from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import pickle
import threading

import numpy as np
import pytest
import tifffile

from xdart.gui.tools.rsm_values import (
    RSMFrameSelector,
    RSMScanMemberForm,
    RSMToolFormV2,
    RSMToolPreflightRefused,
    prepare_rsm_tool_v2,
)
from xrd_tools.analysis.module_transaction import (
    MetadataColumnSelector,
    ModuleSourceGroupReceipt,
)
from xrd_tools.analysis.rsm_geometry_asset import (
    CANONICAL_RSM_GEOMETRY_LOCATOR,
    install_canonical_rsm_geometry_asset,
    rsm_geometry_asset_input,
)
from xrd_tools.analysis.rsm_operation import (
    RSMImageConditioning,
    RSMNormalizationPolicy,
    RSMOperationRefused,
    make_rsm_common_grid,
    prepare_rsm_operation_v2,
)
from xrd_tools.core.geometry import PixelQMap
from xrd_tools.io.analysis_artifact import AnalysisArtifactOverwrite
from xrd_tools.sources.spec import SpecSource


_ROLES = ("mu", "eta", "chi", "phi", "nu", "del")


def _write_member(
    root: Path,
    ordinal: int,
    *,
    energy_eV: float = 13000.007,
    ub: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
) -> RSMScanMemberForm:
    source = root / f"source-{ordinal}"
    images = source / "images"
    images.mkdir(parents=True)
    spec = source / f"RSM_synth_{ordinal}"
    spec.write_text(
        f"""#F RSM_synth_{ordinal}
#E 1
#D Mon Jan 15 10:30:00 2024
#O0 energy  mu  chi  phi  nu  del

#S 1 ascan eta {ordinal} {ordinal + 1} 1 1
#D Mon Jan 15 10:30:00 2024
#P0 {energy_eV} {10 + ordinal} 20 30 40 50
#G3 {' '.join(str(value) for value in ub)}
#N 4
#L eta  Seconds  Seconds  foil status
{ordinal}.0 1 10 101
{ordinal + 1}.0 2 20 110
""",
        encoding="utf-8",
    )
    for label in range(2):
        tifffile.imwrite(
            images / f"member{ordinal}_{label:04d}.tif",
            np.arange(20, dtype=np.uint16).reshape(4, 5) + ordinal + label,
        )
    motors = tuple(
        (role, MetadataColumnSelector(role, 0)) for role in _ROLES
    )
    return RSMScanMemberForm(
        spec,
        "1.1",
        images,
        f"member{ordinal}_",
        RSMFrameSelector(0, 1, 1),
        (195, 487),
        "uint16",
        0,
        motors,
    )


def _form(root: Path, members: tuple[RSMScanMemberForm, ...]) -> RSMToolFormV2:
    install_canonical_rsm_geometry_asset(project_root=root)
    output = root / "output"
    output.mkdir(exist_ok=True)
    return RSMToolFormV2(
        root,
        rsm_geometry_asset_input(CANONICAL_RSM_GEOMETRY_LOCATOR),
        members,
        RSMImageConditioning(0.0, None, None),
        RSMNormalizationPolicy.identity(),
        (4, 5, 6),
        1,
        16 * 1024 * 1024,
        16 * 1024 * 1024,
        output / "rsm-v2.nexus",
        AnalysisArtifactOverwrite.CREATE_NEW,
    )


def test_two_member_preview_is_ordered_image_free_and_factory_owned(
    tmp_path,
    monkeypatch,
):
    members = (_write_member(tmp_path, 0), _write_member(tmp_path, 1))
    form = _form(tmp_path, members)
    runtime_calls = []
    import xrd_tools.analysis.rsm_operation as rsm_operation
    import xrd_tools.core.geometry.xu_runtime as runtime_module
    import xrayutilities as xu

    original_runtime_session = runtime_module.xu_runtime_session
    nthreads_before = xu.config.NTHREADS

    def counted_runtime(*args, **kwargs):
        runtime_calls.append(True)
        return original_runtime_session(*args, **kwargs)

    def forbidden_pixel_q(*_args, **_kwargs):
        raise AssertionError("RSM v2 must use one active owner, not PixelQMap.pixel_q")

    def forbidden_phase_b(*_args, **_kwargs):
        raise AssertionError("RSM v2 Preview must not decode, admit, or write")

    monkeypatch.setattr(runtime_module, "xu_runtime_session", counted_runtime)
    monkeypatch.setattr(PixelQMap, "pixel_q", forbidden_pixel_q)
    monkeypatch.setattr(SpecSource, "load_frame", forbidden_phase_b)
    monkeypatch.setattr(rsm_operation, "admit_module_artifact", forbidden_phase_b)
    monkeypatch.setattr(rsm_operation, "write_rsm", forbidden_phase_b)
    preflight = prepare_rsm_tool_v2(form)

    assert runtime_calls == [True]
    assert xu.config.NTHREADS == nthreads_before
    assert not Path(form.output_path).exists()
    assert preflight.is_current(form)
    assert preflight.request.module._rsm_v2_bound is True
    assert Path(
        preflight.request.module.source.members[0].analysis.resolved_root
    ).name == "RSM_synth_0"
    assert Path(
        preflight.request.module.source.members[1].analysis.resolved_root
    ).name == "RSM_synth_1"
    assert tuple(item.ordinal for item in preflight.request.preflight.members) == (0, 1)
    assert preflight.summary.total_selected_frames == 4
    assert preflight.summary.output_fingerprint == preflight.request.module.output.fingerprint
    assert preflight.summary.geometry_asset_raw_sha256 == (
        preflight.request.preflight.geometry_asset_receipt.raw_sha256
    )
    assert preflight.summary.geometry_asset_semantic_fingerprint == (
        preflight.request.preflight.geometry_asset_receipt.semantic_fingerprint
    )
    assert preflight.summary.normalization_divisor_range == (1.0, 1.0)
    assert preflight.summary.mask_intent == "none"
    assert tuple(item.selected_frame_count for item in preflight.summary.members) == (2, 2)
    assert tuple(item.dependency_file_count for item in preflight.summary.members) == (3, 3)
    assert preflight.summary.union_q_bounds == preflight.request.preflight.common_grid.bounds
    assert preflight.request.provenance["schema_version"] == "rsm-operation-v2-intent"
    assert set(preflight.request.provenance) == {
        "schema_version",
        "kind",
        "source_group",
        "asset",
        "effective_geometry",
        "members",
        "common_grid",
        "conditioning",
        "normalization",
        "plan",
        "runtime_requirements",
        "output",
        "holds",
    }

    protected = (
        preflight,
        preflight.summary,
        *preflight.summary.members,
        preflight.request,
        preflight.request.plan,
        preflight.request.preflight,
        preflight.request.preflight.common_grid,
        *preflight.request.preflight.members,
    )
    for value in protected:
        with pytest.raises(TypeError):
            copy.copy(value)
        with pytest.raises(TypeError):
            copy.deepcopy(value)
        with pytest.raises(TypeError):
            pickle.dumps(value)
        with pytest.raises(TypeError):
            replace(value)
        if hasattr(copy, "replace"):
            with pytest.raises(TypeError):
                copy.replace(value)


def test_visible_member_order_changes_group_preflight_and_request(tmp_path):
    first = _write_member(tmp_path, 0)
    second = _write_member(tmp_path, 1)
    forward = prepare_rsm_tool_v2(_form(tmp_path, (first, second)))
    reverse = prepare_rsm_tool_v2(_form(tmp_path, (second, first)))

    assert forward.form.fingerprint != reverse.form.fingerprint
    assert (
        forward.request.module.source.fingerprint
        != reverse.request.module.source.fingerprint
    )
    assert forward.request.preflight.fingerprint != reverse.request.preflight.fingerprint
    assert forward.request.module.fingerprint != reverse.request.module.fingerprint
    assert forward.request.preflight.common_grid.bounds == reverse.request.preflight.common_grid.bounds
    assert not Path(forward.form.output_path).exists()


def test_group_preflight_factory_rejects_foreign_ordered_source_relation(tmp_path):
    import xrd_tools.analysis.rsm_operation as rsm_operation

    members = (_write_member(tmp_path, 0), _write_member(tmp_path, 1))
    accepted = prepare_rsm_tool_v2(_form(tmp_path, members)).request
    preflight = accepted.preflight
    reversed_group = ModuleSourceGroupReceipt.from_members(
        tuple(reversed(accepted.module.source.members))
    )
    with pytest.raises(RSMOperationRefused) as refused:
        rsm_operation._make_rsm_group_preflight_v2(
            project_root=Path(preflight.project_root),
            asset=preflight.geometry_asset_receipt,
            effective=preflight.effective_geometry,
            bindings=preflight.ordered_geometry_bindings,
            members=preflight.members,
            group=reversed_group,
            common_grid=preflight.common_grid,
            plan=accepted.plan,
        )
    assert refused.value.code == "RSM_MEMBER_IDENTITY_MISMATCH"


def test_member_shape_refuses_before_runtime_or_q_work(tmp_path, monkeypatch):
    member = replace(_write_member(tmp_path, 0), detector_shape=(4, 5))
    form = _form(tmp_path, (member,))
    import xrd_tools.core.geometry.xu_runtime as runtime_module

    monkeypatch.setattr(
        runtime_module,
        "xu_runtime_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("runtime must not start for a shape mismatch")
        ),
    )
    with pytest.raises(Exception) as refused:
        prepare_rsm_tool_v2(form)
    assert getattr(refused.value, "code", None) == "RSM_MEMBER_IDENTITY_MISMATCH"
    assert not Path(form.output_path).exists()


def test_form_rejects_duplicate_members_and_grid_limits(tmp_path):
    member = _write_member(tmp_path, 0)
    install_canonical_rsm_geometry_asset(project_root=tmp_path)
    output = tmp_path / "result.nexus"
    common = dict(
        project_root=tmp_path,
        geometry_asset=rsm_geometry_asset_input(CANONICAL_RSM_GEOMETRY_LOCATOR),
        conditioning=RSMImageConditioning(0.0, None, None),
        normalization=RSMNormalizationPolicy.identity(),
        chunk_size=1,
        max_frame_bytes=1024,
        max_chunk_bytes=1024,
        output_path=output,
    )
    with pytest.raises(TypeError, match="invalid"):
        RSMToolFormV2(members=(member, member), bins=(2, 2, 2), **common)
    with pytest.raises(TypeError, match="invalid"):
        RSMToolFormV2(members=(member,), bins=(1_000_001, 2, 2), **common)
    with pytest.raises(TypeError, match="invalid"):
        RSMToolFormV2(members=(member,), bins=(201, 200, 200), **common)


def test_common_grid_refuses_float32_axis_collapse():
    collapsed = (1.0e20, 1.0e20 + 1.0e12)
    with pytest.raises(RSMOperationRefused) as refused:
        make_rsm_common_grid(
            ((collapsed, (0.0, 1.0), (0.0, 1.0)),),
            (2, 2, 2),
        )
    assert refused.value.code == "RSM_COMMON_GRID_INVALID"


def test_same_source_disjoint_members_pass_and_overlap_refuses_before_grid(
    tmp_path,
    monkeypatch,
):
    import xrd_tools.analysis.rsm_operation as rsm_operation

    base = _write_member(tmp_path, 0)
    first = replace(base, frame_selector=RSMFrameSelector(0, 0, 1))
    second = replace(base, frame_selector=RSMFrameSelector(1, 1, 1))
    disjoint = prepare_rsm_tool_v2(_form(tmp_path, (first, second)))
    assert tuple(
        contribution.label
        for member in disjoint.request.preflight.members
        for contribution in member.contributions
    ) == (0, 1)

    grid_calls = []
    original_grid = rsm_operation.make_rsm_common_grid

    def counted_grid(*args, **kwargs):
        grid_calls.append(True)
        return original_grid(*args, **kwargs)

    monkeypatch.setattr(rsm_operation, "make_rsm_common_grid", counted_grid)
    overlapping = replace(base, frame_selector=RSMFrameSelector(0, 1, 1))
    with pytest.raises(RSMToolPreflightRefused) as refused:
        prepare_rsm_tool_v2(_form(tmp_path, (overlapping, second)))
    assert refused.value.code == "RSM_SOURCE_GROUP_INVALID"
    assert grid_calls == []


def test_per_member_energy_and_ub_are_ordered_source_facts(tmp_path, monkeypatch):
    import xrd_tools.analysis.rsm_operation as rsm_operation

    first_ub = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    second_ub = (2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0)
    members = (
        _write_member(tmp_path, 0, energy_eV=12001.0, ub=first_ub),
        _write_member(tmp_path, 1, energy_eV=14002.0, ub=second_ub),
    )
    observed = []

    def q_bounds_spy(
        _mapper,
        _angles,
        energy_eV,
        ub,
        **_kwargs,
    ):
        observed.append((float(energy_eV), tuple(float(value) for value in ub.flat)))
        offset = float(len(observed))
        return (
            (offset, offset + 1.0),
            (offset + 2.0, offset + 3.0),
            (offset + 4.0, offset + 5.0),
        )

    monkeypatch.setattr(
        rsm_operation,
        "_resolve_exact_rsm_q_bounds_active",
        q_bounds_spy,
    )
    preflight = prepare_rsm_tool_v2(_form(tmp_path, members))
    assert observed == [(12001.0, first_ub), (14002.0, second_ub)]
    assert tuple(item.energy_eV for item in preflight.request.preflight.members) == (
        12001.0,
        14002.0,
    )
    assert tuple(item.ub for item in preflight.request.preflight.members) == (
        (first_ub[0:3], first_ub[3:6], first_ub[6:9]),
        (second_ub[0:3], second_ub[3:6], second_ub[6:9]),
    )


@pytest.mark.parametrize(
    ("source_facts", "expected_code"),
    (
        ((float("nan"), np.eye(3)), "RSM_MEMBER_ENERGY_INVALID"),
        ((13000.0, np.ones((2, 2))), "RSM_MEMBER_UB_INVALID"),
    ),
)
def test_member_energy_and_ub_fail_with_stable_v2_codes(
    tmp_path,
    monkeypatch,
    source_facts,
    expected_code,
):
    import xrd_tools.analysis.rsm_operation as rsm_operation

    member = _write_member(tmp_path, 0)
    monkeypatch.setattr(
        rsm_operation,
        "get_energy_and_UB",
        lambda *_args, **_kwargs: source_facts,
    )
    with pytest.raises(RSMToolPreflightRefused) as refused:
        prepare_rsm_tool_v2(_form(tmp_path, (member,)))
    assert refused.value.code == expected_code


def test_xu_nthreads_restores_after_q_failure_and_cancellation(tmp_path, monkeypatch):
    import xrayutilities as xu
    import xrd_tools.analysis.rsm_operation as rsm_operation

    member = _write_member(tmp_path, 0)
    before = xu.config.NTHREADS

    def fail_inside_owner(*_args, **_kwargs):
        assert xu.config.NTHREADS == 1
        raise RSMOperationRefused("RSM_MEMBER_Q_BOUNDS_INVALID")

    monkeypatch.setattr(
        rsm_operation,
        "_resolve_exact_rsm_q_bounds_active",
        fail_inside_owner,
    )
    with pytest.raises(RSMToolPreflightRefused) as refused:
        prepare_rsm_tool_v2(_form(tmp_path, (member,)))
    assert refused.value.code == "RSM_MEMBER_Q_BOUNDS_INVALID"
    assert xu.config.NTHREADS == before

    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(RSMToolPreflightRefused) as refused:
        prepare_rsm_tool_v2(_form(tmp_path, (member,)), cancel_token=cancelled)
    assert refused.value.code == "CANCELLED"
    assert xu.config.NTHREADS == before


def test_core_v2_rejects_duplicate_forms_and_limits_before_runtime(
    tmp_path,
    monkeypatch,
):
    members = (_write_member(tmp_path, 0), _write_member(tmp_path, 1))
    baseline = prepare_rsm_tool_v2(_form(tmp_path, members))
    request = baseline.request

    import xrd_tools.core.geometry.xu_runtime as runtime_module

    monkeypatch.setattr(
        runtime_module,
        "xu_runtime_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid core intent must refuse before XU runtime")
        ),
    )
    common = dict(
        source_group=request.module.source,
        output=request.module.output,
        geometry_asset_receipt=request.preflight.geometry_asset_receipt,
        member_form_fingerprints=tuple(item.fingerprint for item in members),
        member_motor_selectors=tuple(item.motor_selectors for item in members),
        conditioning=request.plan.conditioning,
        normalization=request.plan.normalization,
        bins=(4, 5, 6),
        chunk_size=1,
        max_frame_bytes=16 * 1024 * 1024,
        max_chunk_bytes=16 * 1024 * 1024,
        project_root=tmp_path,
    )

    duplicate = dict(common)
    duplicate["member_form_fingerprints"] = (
        members[0].fingerprint,
        members[0].fingerprint,
    )
    with pytest.raises(RSMOperationRefused) as refused:
        prepare_rsm_operation_v2(**duplicate)
    assert refused.value.code == "RSM_SOURCE_GROUP_INVALID"

    invalid_cases = (
        ("bins", (1_000_001, 2, 2), "RSM_COMMON_GRID_INVALID"),
        ("bins", (201, 200, 200), "RSM_COMMON_GRID_INVALID"),
        ("chunk_size", 1025, "Q_MEMORY_LIMIT_EXCEEDED"),
        ("max_frame_bytes", 4 * 1024**3 + 1, "FRAME_MEMORY_LIMIT_EXCEEDED"),
        ("max_chunk_bytes", 4 * 1024**3 + 1, "Q_MEMORY_LIMIT_EXCEEDED"),
    )
    for key, value, code in invalid_cases:
        invalid = dict(common)
        invalid[key] = value
        with pytest.raises(RSMOperationRefused) as refused:
            prepare_rsm_operation_v2(**invalid)
        assert refused.value.code == code


@pytest.mark.parametrize(
    ("limit_name", "accepted_limit", "reduced_limit"),
    (
        ("_MAX_MANIFEST_FILES", 8192, 5),
        ("_MAX_MANIFEST_BYTES", 512 * 1024, 6500),
        ("_MAX_PROVENANCE_BYTES", 1024 * 1024, 9000),
    ),
)
def test_group_dependency_and_canonical_byte_limits_fail_closed(
    tmp_path,
    monkeypatch,
    limit_name,
    accepted_limit,
    reduced_limit,
):
    import xrd_tools.analysis.rsm_operation as rsm_operation

    assert getattr(rsm_operation, limit_name) == accepted_limit
    monkeypatch.setattr(rsm_operation, limit_name, reduced_limit)
    members = (_write_member(tmp_path, 0), _write_member(tmp_path, 1))
    form = _form(tmp_path, members)
    with pytest.raises(RSMToolPreflightRefused) as refused:
        prepare_rsm_tool_v2(form)
    assert refused.value.code == "RSM_SOURCE_GROUP_LIMIT_EXCEEDED"
    assert not Path(form.output_path).exists()
