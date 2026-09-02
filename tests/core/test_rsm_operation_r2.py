from __future__ import annotations

import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import pickle
import stat
import threading
import weakref

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
    ModuleDisposition,
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
    RSMOperationCleanupPendingV2,
    RSMOperationExecutionV2,
    RSMOperationRefused,
    RSMOperationVerificationErrorV2,
    make_rsm_common_grid,
    prepare_rsm_operation_v2,
    run_rsm_operation_v2,
)
from xrd_tools.core.allocator_pressure import (
    AllocatorPressureCallFailed,
    AllocatorPressureUnavailable,
)
from xrd_tools.core.geometry import Diffractometer, PixelQMap
from xrd_tools.core.geometry.xu_runtime import XuRuntimeUnsupported
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


@pytest.mark.parametrize("failure_site", ("make_hxrd", "init_area", "area"))
def test_q_backend_exceptions_are_stable_tool_refusals(
    tmp_path,
    monkeypatch,
    failure_site,
):
    import xrayutilities as xu

    member = _write_member(tmp_path, 0)
    form = _form(tmp_path, (member,))
    nthreads_before = xu.config.NTHREADS

    class FailingAng2Q:
        def init_area(self, *_args, **_kwargs):
            if failure_site == "init_area":
                raise RuntimeError("backend-private init detail")

        def area(self, *_args, **_kwargs):
            if failure_site == "area":
                raise RuntimeError("backend-private area detail")
            raise AssertionError("area must not run for this failure site")

    class FailingHxrd:
        Ang2Q = FailingAng2Q()

    def failing_make_hxrd(*_args, **_kwargs):
        if failure_site == "make_hxrd":
            raise RuntimeError("backend-private construction detail")
        return FailingHxrd()

    monkeypatch.setattr(Diffractometer, "make_hxrd", failing_make_hxrd)
    with pytest.raises(RSMToolPreflightRefused) as refused:
        prepare_rsm_tool_v2(form)
    assert refused.value.code == "RSM_MEMBER_Q_BOUNDS_INVALID"
    assert "backend-private" not in str(refused.value)
    assert xu.config.NTHREADS == nthreads_before
    assert not Path(form.output_path).exists()


@pytest.mark.parametrize("failure_site", ("make_hxrd", "init_area", "area"))
@pytest.mark.parametrize(
    ("error_type", "expected_code"),
    (
        (RSMOperationRefused, "CANCELLED"),
        (XuRuntimeUnsupported, "XU_RUNTIME_TEST_REFUSAL"),
    ),
)
def test_q_backend_preserves_typed_refusals(
    tmp_path,
    monkeypatch,
    failure_site,
    error_type,
    expected_code,
):
    import xrayutilities as xu

    member = _write_member(tmp_path, 0)
    form = _form(tmp_path, (member,))
    nthreads_before = xu.config.NTHREADS

    def raise_typed():
        raise error_type(expected_code)

    class FailingAng2Q:
        def init_area(self, *_args, **_kwargs):
            if failure_site == "init_area":
                raise_typed()

        def area(self, *_args, **_kwargs):
            if failure_site == "area":
                raise_typed()
            raise AssertionError("area must not run for this failure site")

    class FailingHxrd:
        Ang2Q = FailingAng2Q()

    def failing_make_hxrd(*_args, **_kwargs):
        if failure_site == "make_hxrd":
            raise_typed()
        return FailingHxrd()

    monkeypatch.setattr(Diffractometer, "make_hxrd", failing_make_hxrd)
    with pytest.raises(RSMToolPreflightRefused) as refused:
        prepare_rsm_tool_v2(form)
    assert refused.value.code == expected_code
    assert xu.config.NTHREADS == nthreads_before
    assert not Path(form.output_path).exists()


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


def test_two_member_science_uses_one_grid_and_persists_exact_attestation(
    tmp_path,
    monkeypatch,
):
    from contextlib import contextmanager

    import xrd_tools.analysis.rsm_operation as rsm_operation
    import xrd_tools.rsm.gridding as rsm_gridding

    monkeypatch.setattr(
        rsm_operation,
        "_resolve_exact_rsm_q_bounds_active",
        lambda *_args, **_kwargs: (
            (-8.0, 8.0),
            (-8.0, 8.0),
            (-8.0, 8.0),
        ),
    )
    first_ub = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    second_ub = (2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0)
    members = (
        _write_member(tmp_path, 0, energy_eV=12001.0, ub=first_ub),
        _write_member(tmp_path, 1, energy_eV=14002.0, ub=second_ub),
    )
    prepared = prepare_rsm_tool_v2(_form(tmp_path, members)).request
    for name in (
        "combine_grids",
        "get_common_grid",
    ):
        monkeypatch.setattr(
            rsm_gridding,
            name,
            lambda *_args, _name=name, **_kwargs: (_ for _ in ()).throw(
                AssertionError(f"forbidden RSM v2 grid path used: {_name}")
            ),
        )
    real_source_lease = rsm_operation.requalified_analysis_source
    source_lease_count = 0
    max_source_leases = 0
    source_lease_entries = 0

    @contextmanager
    def counted_source_lease(*args, **kwargs):
        nonlocal source_lease_count, max_source_leases, source_lease_entries
        with real_source_lease(*args, **kwargs) as source:
            source_lease_count += 1
            source_lease_entries += 1
            max_source_leases = max(max_source_leases, source_lease_count)
            try:
                yield source
            finally:
                source_lease_count -= 1

    monkeypatch.setattr(
        rsm_operation,
        "requalified_analysis_source",
        counted_source_lease,
    )

    def full_detector_frame(self, index):
        ordinal = int(
            str(self.name).split(" [", 1)[0].rsplit("_", 1)[-1]
        )
        return np.full(
            (195, 487),
            10 + ordinal + int(index),
            dtype=np.float64,
        )

    monkeypatch.setattr(SpecSource, "load_frame", full_detector_frame)
    real_gridder = rsm_operation.StreamingGridder
    real_pixel_q = PixelQMap.pixel_q
    instances = []
    observed = []
    mapped = []

    def record_pixel_q(
        self,
        angles,
        energy,
        *,
        UB=None,
        roi=None,
        image_shape=None,
        runtime_session=None,
    ):
        assert runtime_session is not None and runtime_session.active
        mapped.append(
            (
                float(energy),
                tuple(float(value) for value in UB.flat),
                len(angles[0]),
            )
        )
        return real_pixel_q(
            self,
            angles,
            energy,
            UB=UB,
            roi=roi,
            image_shape=image_shape,
            runtime_session=runtime_session,
        )

    monkeypatch.setattr(PixelQMap, "pixel_q", record_pixel_q)

    class CountedGridder(real_gridder):
        def __init__(self, *args, **kwargs):
            instances.append(self)
            super().__init__(*args, **kwargs)

        def add_leased(self, lease, angles, energy, *, UB, **kwargs):
            observed.append(
                (
                    float(energy),
                    tuple(float(value) for value in UB.flat),
                    lease.frame_count,
                )
            )
            return super().add_leased(
                lease,
                angles,
                energy,
                UB=UB,
                **kwargs,
            )

    monkeypatch.setattr(rsm_operation, "StreamingGridder", CountedGridder)
    result = run_rsm_operation_v2(prepared)

    assert result.terminal.disposition is ModuleDisposition.COMMITTED
    assert result.payload is not None
    assert result.payload.schema_version == 2
    assert len(instances) == 1
    assert source_lease_entries == 2
    assert max_source_leases == 1
    assert source_lease_count == 0
    assert len(observed) == 4
    assert mapped == observed
    assert [item[2] for item in observed] == [1, 1, 1, 1]
    assert observed == [
        (12001.0, first_ub, 1),
        (12001.0, first_ub, 1),
        (14002.0, second_ub, 1),
        (14002.0, second_ub, 1),
    ]
    attestation = json.loads(result.payload.execution_attestation_json)
    assert attestation["selected_scan_count"] == 2
    assert attestation["selected_frame_count"] == 4
    assert attestation["science_chunk_count"] == 4
    assert attestation["q_release_check_chunk_count"] == 4
    assert attestation["frame_release_check_frame_count"] == 4
    assert [item["mask_policy"] for item in attestation["member_masks"]] == [
        "none",
        "none",
    ]
    assert result.payload.inspection.source_fingerprint == (
        prepared.module.source.fingerprint
    )
    assert result.payload.provenance_json == prepared.provenance_json
    assert result.payload.intensity.shape == prepared.plan.bins
    assert not np.any(np.isinf(result.payload.intensity))
    execution = RSMOperationExecutionV2(prepared)
    with pytest.raises(TypeError):
        copy.copy(execution)
    with pytest.raises(TypeError):
        copy.deepcopy(execution)


def test_member_static_masks_are_independent_and_attested(tmp_path, monkeypatch):
    import xrd_tools.analysis.rsm_operation as rsm_operation

    monkeypatch.setattr(
        rsm_operation,
        "_resolve_exact_rsm_q_bounds_active",
        lambda *_args, **_kwargs: (
            (-8.0, 8.0),
            (-8.0, 8.0),
            (-8.0, 8.0),
        ),
    )
    members = (_write_member(tmp_path, 0), _write_member(tmp_path, 1))
    form = replace(
        _form(tmp_path, members),
        conditioning=RSMImageConditioning(0.0, None, 100.0),
    )
    prepared = prepare_rsm_tool_v2(form).request
    r0, _r1, c0, _c1 = prepared.plan.effective_geometry.roi
    real_derive = rsm_operation._derive_rsm_v2_static_mask
    mask_roots = {}

    def capture_mask_roots(*args, **kwargs):
        full, cropped, receipt = real_derive(*args, **kwargs)
        member = args[1]
        mask_roots[member.fingerprint] = (
            weakref.ref(full),
            weakref.ref(cropped),
        )
        return full, cropped, receipt

    real_lease = rsm_operation._make_rsm_v2_chunk_lease

    def require_only_cropped_mask(*args, **kwargs):
        member = args[1]
        full_ref, cropped_ref = mask_roots[member.fingerprint]
        assert full_ref() is None
        assert cropped_ref() is kwargs["cropped_static_mask"]
        lease = real_lease(*args, **kwargs)
        masked_row = r0 + member.ordinal
        masked_column = c0 + member.ordinal
        assert np.isnan(
            lease._conditioned[:, masked_row, masked_column]
        ).all()
        assert np.isfinite(
            lease._conditioned[:, masked_row, masked_column + 2]
        ).all()
        return lease

    monkeypatch.setattr(
        rsm_operation,
        "_derive_rsm_v2_static_mask",
        capture_mask_roots,
    )
    monkeypatch.setattr(
        rsm_operation,
        "_make_rsm_v2_chunk_lease",
        require_only_cropped_mask,
    )

    def member_specific_hot_pixel(self, index):
        ordinal = int(
            str(self.name).split(" [", 1)[0].rsplit("_", 1)[-1]
        )
        frame = np.full(
            (195, 487),
            10 + ordinal + int(index),
            dtype=np.float64,
        )
        frame[r0 + ordinal, c0 + ordinal] = 500
        return frame

    monkeypatch.setattr(SpecSource, "load_frame", member_specific_hot_pixel)
    result = run_rsm_operation_v2(prepared)

    assert result.terminal.disposition is ModuleDisposition.COMMITTED
    attestation = json.loads(result.payload.execution_attestation_json)
    masks = attestation["member_masks"]
    assert [item["mask_policy"] for item in masks] == [
        "exact-all-selected-frames-static-hot-v1",
        "exact-all-selected-frames-static-hot-v1",
    ]
    assert [item["full_masked_pixel_count"] for item in masks] == [1, 1]
    assert [item["cropped_masked_pixel_count"] for item in masks] == [1, 1]
    assert masks[0]["full_raw_digest"] != masks[1]["full_raw_digest"]
    assert masks[0]["cropped_raw_digest"] != masks[1]["cropped_raw_digest"]
    assert (
        masks[0]["mask_receipt_fingerprint"]
        != masks[1]["mask_receipt_fingerprint"]
    )
    assert all(
        reference() is None
        for pair in mask_roots.values()
        for reference in pair
    )


def test_science_release_failure_never_admits_or_writes(tmp_path, monkeypatch):
    import xrd_tools.analysis.rsm_operation as rsm_operation

    monkeypatch.setattr(
        rsm_operation,
        "_resolve_exact_rsm_q_bounds_active",
        lambda *_args, **_kwargs: (
            (-8.0, 8.0),
            (-8.0, 8.0),
            (-8.0, 8.0),
        ),
    )
    member = _write_member(tmp_path, 0)
    prepared = prepare_rsm_tool_v2(_form(tmp_path, (member,))).request
    monkeypatch.setattr(
        SpecSource,
        "load_frame",
        lambda _self, index: np.full(
            (195, 487),
            10 + int(index),
            dtype=np.float64,
        ),
    )

    def release_failure(*_args, **_kwargs):
        raise rsm_operation.RSMGridChunkReleaseError()

    monkeypatch.setattr(
        rsm_operation.StreamingGridder,
        "add_leased",
        release_failure,
    )
    monkeypatch.setattr(
        rsm_operation,
        "admit_module_artifact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("release failure must stop before admission")
        ),
    )
    result = run_rsm_operation_v2(prepared)

    assert result.terminal.disposition is ModuleDisposition.REFUSED
    assert result.terminal.code == "RSM_CHUNK_RELEASE_FAILED"
    assert not Path(prepared.module.output.target).exists()


def test_allocator_pressure_unavailable_refuses_before_science_or_output(
    tmp_path,
    monkeypatch,
):
    import xrd_tools.analysis.rsm_operation as rsm_operation

    monkeypatch.setattr(
        rsm_operation,
        "_resolve_exact_rsm_q_bounds_active",
        lambda *_args, **_kwargs: (
            (-8.0, 8.0),
            (-8.0, 8.0),
            (-8.0, 8.0),
        ),
    )
    prepared = prepare_rsm_tool_v2(
        _form(tmp_path, (_write_member(tmp_path, 0),))
    ).request
    monkeypatch.setattr(
        rsm_operation,
        "bind_darwin_allocator_pressure_relief",
        lambda: (_ for _ in ()).throw(AllocatorPressureUnavailable()),
    )
    monkeypatch.setattr(
        rsm_operation,
        "StreamingGridder",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unavailable allocator pressure must stop before science")
        ),
    )

    result = run_rsm_operation_v2(prepared)

    assert result.terminal.disposition is ModuleDisposition.REFUSED
    assert result.terminal.code == "RSM_ALLOCATOR_PRESSURE_UNAVAILABLE"
    assert not Path(prepared.module.output.target).exists()


def test_xu_platform_refusal_precedes_darwin_allocator_binding(
    tmp_path,
    monkeypatch,
):
    import xrd_tools.analysis.rsm_operation as rsm_operation
    import xrd_tools.core.geometry.xu_runtime as xu_runtime

    monkeypatch.setattr(
        rsm_operation,
        "_resolve_exact_rsm_q_bounds_active",
        lambda *_args, **_kwargs: (
            (-8.0, 8.0),
            (-8.0, 8.0),
            (-8.0, 8.0),
        ),
    )
    prepared = prepare_rsm_tool_v2(
        _form(tmp_path, (_write_member(tmp_path, 0),))
    ).request

    class UnsupportedRuntime:
        execution_record = None

        def __enter__(self):
            raise XuRuntimeUnsupported("XU_PLATFORM_UNVALIDATED")

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        xu_runtime,
        "xu_runtime_session",
        lambda *_args, **_kwargs: UnsupportedRuntime(),
    )
    monkeypatch.setattr(
        rsm_operation,
        "bind_darwin_allocator_pressure_relief",
        lambda: (_ for _ in ()).throw(
            AssertionError("unvalidated platform must refuse before Darwin binding")
        ),
    )

    result = run_rsm_operation_v2(prepared)

    assert result.terminal.disposition is ModuleDisposition.REFUSED
    assert result.terminal.code == "XU_PLATFORM_UNVALIDATED"
    assert not Path(prepared.module.output.target).exists()


@pytest.mark.parametrize("fail_at_call", (1, 2))
def test_allocator_pressure_call_failure_is_bounded_and_writes_nothing(
    tmp_path,
    monkeypatch,
    fail_at_call,
):
    import xrd_tools.analysis.rsm_operation as rsm_operation

    monkeypatch.setattr(
        rsm_operation,
        "_resolve_exact_rsm_q_bounds_active",
        lambda *_args, **_kwargs: (
            (-8.0, 8.0),
            (-8.0, 8.0),
            (-8.0, 8.0),
        ),
    )
    prepared = prepare_rsm_tool_v2(
        _form(tmp_path, (_write_member(tmp_path, 0),))
    ).request
    monkeypatch.setattr(
        SpecSource,
        "load_frame",
        lambda _self, index: np.full(
            (195, 487),
            10 + int(index),
            dtype=np.float64,
        ),
    )
    calls = []

    class FailingPressure:
        def relieve(self) -> None:
            calls.append(len(calls) + 1)
            if len(calls) == fail_at_call:
                raise AllocatorPressureCallFailed()

    monkeypatch.setattr(
        rsm_operation,
        "bind_darwin_allocator_pressure_relief",
        lambda: FailingPressure(),
    )
    monkeypatch.setattr(
        rsm_operation,
        "admit_module_artifact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("allocator-pressure failure must stop before admission")
        ),
    )

    result = run_rsm_operation_v2(prepared)

    assert result.terminal.disposition is ModuleDisposition.FAILED
    assert result.terminal.code == "RSM_ALLOCATOR_PRESSURE_FAILED"
    assert calls == list(range(1, fail_at_call + 1))
    assert not Path(prepared.module.output.target).exists()


def test_allocator_pressure_runs_at_two_root_death_fences_only(
    tmp_path,
    monkeypatch,
):
    import xrd_tools.analysis.rsm_operation as rsm_operation

    monkeypatch.setattr(
        rsm_operation,
        "_resolve_exact_rsm_q_bounds_active",
        lambda *_args, **_kwargs: (
            (-8.0, 8.0),
            (-8.0, 8.0),
            (-8.0, 8.0),
        ),
    )
    members = (_write_member(tmp_path, 0), _write_member(tmp_path, 1))
    prepared = prepare_rsm_tool_v2(_form(tmp_path, members)).request
    monkeypatch.setattr(
        SpecSource,
        "load_frame",
        lambda self, index: np.full(
            (195, 487),
            10
            + int(str(self.name).split(" [", 1)[0].rsplit("_", 1)[-1])
            + int(index),
            dtype=np.float64,
        ),
    )
    real_make_lease = rsm_operation._make_rsm_v2_chunk_lease
    lease_root_refs = []

    def capture_lease_roots(*args, **kwargs):
        lease = real_make_lease(*args, **kwargs)
        lease_root_refs.extend(
            (lease._raw_root_ref, lease._conditioned_root_ref)
        )
        return lease

    real_to_volume = rsm_operation.StreamingGridder.to_volume
    gridder_refs = []
    volume_root_refs = []

    def capture_final_roots(gridder):
        gridder_refs.append(weakref.ref(gridder))
        volume = real_to_volume(gridder)
        volume_root_refs.append(weakref.ref(volume.intensity))
        return volume

    calls = []

    class RecordingPressure:
        __slots__ = ()

        def relieve(self) -> None:
            assert lease_root_refs
            assert all(reference() is None for reference in lease_root_refs)
            if not volume_root_refs:
                calls.append("pre-finalization")
                return
            assert all(reference() is None for reference in volume_root_refs)
            assert all(reference() is None for reference in gridder_refs)
            calls.append("post-projection")

    monkeypatch.setattr(
        rsm_operation,
        "_make_rsm_v2_chunk_lease",
        capture_lease_roots,
    )
    monkeypatch.setattr(
        rsm_operation.StreamingGridder,
        "to_volume",
        capture_final_roots,
    )
    monkeypatch.setattr(
        rsm_operation,
        "bind_darwin_allocator_pressure_relief",
        lambda: RecordingPressure(),
    )

    result = run_rsm_operation_v2(prepared)

    assert result.terminal.disposition is ModuleDisposition.COMMITTED
    assert calls == ["pre-finalization", "post-projection"]
    assert len(lease_root_refs) == 8
    assert len(gridder_refs) == len(volume_root_refs) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("frame_count", 2),
        ("q_root_count", 0),
        ("release_passed", False),
    ),
)
def test_science_release_receipt_is_revalidated_at_operation_boundary(
    tmp_path,
    monkeypatch,
    field,
    value,
):
    import xrd_tools.analysis.rsm_operation as rsm_operation

    monkeypatch.setattr(
        rsm_operation,
        "_resolve_exact_rsm_q_bounds_active",
        lambda *_args, **_kwargs: (
            (-8.0, 8.0),
            (-8.0, 8.0),
            (-8.0, 8.0),
        ),
    )
    prepared = prepare_rsm_tool_v2(
        _form(tmp_path, (_write_member(tmp_path, 0),))
    ).request
    monkeypatch.setattr(
        SpecSource,
        "load_frame",
        lambda _self, index: np.full(
            (195, 487),
            10 + int(index),
            dtype=np.float64,
        ),
    )
    real_add = rsm_operation.StreamingGridder.add_leased

    def corrupt_release(*args, **kwargs):
        receipt = real_add(*args, **kwargs)
        object.__setattr__(receipt, field, value)
        return receipt

    monkeypatch.setattr(
        rsm_operation.StreamingGridder,
        "add_leased",
        corrupt_release,
    )
    result = run_rsm_operation_v2(prepared)

    assert result.terminal.disposition is ModuleDisposition.REFUSED
    assert result.terminal.code == "RSM_CHUNK_RELEASE_FAILED"
    assert not Path(prepared.module.output.target).exists()


def test_science_release_receipt_exact_class_clone_is_refused(
    tmp_path,
    monkeypatch,
):
    import xrd_tools.analysis.rsm_operation as rsm_operation

    monkeypatch.setattr(
        rsm_operation,
        "_resolve_exact_rsm_q_bounds_active",
        lambda *_args, **_kwargs: (
            (-8.0, 8.0),
            (-8.0, 8.0),
            (-8.0, 8.0),
        ),
    )
    prepared = prepare_rsm_tool_v2(
        _form(tmp_path, (_write_member(tmp_path, 0),))
    ).request
    monkeypatch.setattr(
        SpecSource,
        "load_frame",
        lambda _self, index: np.full(
            (195, 487),
            10 + int(index),
            dtype=np.float64,
        ),
    )
    real_add = rsm_operation.StreamingGridder.add_leased

    def clone_release(*args, **kwargs):
        receipt = real_add(*args, **kwargs)
        clone = object.__new__(type(receipt))
        for name in ("frame_count", "q_root_count", "release_passed"):
            object.__setattr__(clone, name, getattr(receipt, name))
        return clone

    monkeypatch.setattr(
        rsm_operation.StreamingGridder,
        "add_leased",
        clone_release,
    )
    monkeypatch.setattr(
        rsm_operation,
        "admit_module_artifact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cloned release receipt must stop before admission")
        ),
    )

    result = run_rsm_operation_v2(prepared)

    assert result.terminal.disposition is ModuleDisposition.REFUSED
    assert result.terminal.code == "RSM_CHUNK_RELEASE_FAILED"
    assert not Path(prepared.module.output.target).exists()


def test_memory_plan_refuses_before_source_decode(tmp_path, monkeypatch):
    import xrd_tools.analysis.rsm_operation as rsm_operation

    monkeypatch.setattr(
        rsm_operation,
        "_resolve_exact_rsm_q_bounds_active",
        lambda *_args, **_kwargs: (
            (-8.0, 8.0),
            (-8.0, 8.0),
            (-8.0, 8.0),
        ),
    )
    member = _write_member(tmp_path, 0)
    prepared = prepare_rsm_tool_v2(_form(tmp_path, (member,))).request
    monkeypatch.setattr(rsm_operation, "_MAX_RSM_RESIDENT_BYTES", 1)
    monkeypatch.setattr(
        SpecSource,
        "load_frame",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("memory refusal must precede detector decode")
        ),
    )
    result = run_rsm_operation_v2(prepared)

    assert result.terminal.disposition is ModuleDisposition.REFUSED
    assert result.terminal.code == "Q_MEMORY_LIMIT_EXCEEDED"
    assert not Path(prepared.module.output.target).exists()


def test_static_mask_science_charge_is_exact_before_decode(tmp_path, monkeypatch):
    import xrd_tools.analysis.rsm_operation as rsm_operation

    monkeypatch.setattr(
        rsm_operation,
        "_resolve_exact_rsm_q_bounds_active",
        lambda *_args, **_kwargs: (
            (-8.0, 8.0),
            (-8.0, 8.0),
            (-8.0, 8.0),
        ),
    )
    member_form = _write_member(tmp_path, 0)
    form = replace(
        _form(tmp_path, (member_form,)),
        conditioning=RSMImageConditioning(0.0, None, 100.0),
    )
    prepared = prepare_rsm_tool_v2(form).request
    member = prepared.preflight.members[0]
    frame_pixels = int(np.prod(member.detector_shape))
    raw_bytes = frame_pixels * np.dtype(np.float64).itemsize
    float_bytes = frame_pixels * np.dtype(np.float64).itemsize
    old_science = rsm_operation._rsm_v2_chunk_owned_bytes(
        raw_bytes=raw_bytes,
        float_bytes=float_bytes,
        pixel_count=frame_pixels,
        mask_state_bytes=0,
        science=True,
    )
    fixed_grid = 4 * int(np.prod(prepared.plan.bins)) * np.dtype(np.float64).itemsize
    cropped_mask_bytes = int(np.prod(member.cropped_shape)) * np.dtype(bool).itemsize
    monkeypatch.setattr(
        rsm_operation,
        "_MAX_RSM_RESIDENT_BYTES",
        fixed_grid + old_science + cropped_mask_bytes,
    )
    monkeypatch.setattr(
        SpecSource,
        "load_frame",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("static-mask accounting must precede detector decode")
        ),
    )

    result = run_rsm_operation_v2(prepared)

    assert result.terminal.disposition is ModuleDisposition.REFUSED
    assert result.terminal.code == "Q_MEMORY_LIMIT_EXCEEDED"
    assert not Path(prepared.module.output.target).exists()


def test_finalization_public_copies_and_projection_are_charged_before_decode(
    tmp_path,
    monkeypatch,
):
    import xrd_tools.analysis.rsm_operation as rsm_operation

    monkeypatch.setattr(
        rsm_operation,
        "_resolve_exact_rsm_q_bounds_active",
        lambda *_args, **_kwargs: (
            (-8.0, 8.0),
            (-8.0, 8.0),
            (-8.0, 8.0),
        ),
    )
    form = replace(
        _form(tmp_path, (_write_member(tmp_path, 0),)),
        bins=(100, 100, 100),
    )
    prepared = prepare_rsm_tool_v2(form).request
    voxels = int(np.prod(prepared.plan.bins))
    old_projection_only_charge = voxels * (
        5 * np.dtype(np.float64).itemsize
        + 2 * np.dtype(np.float32).itemsize
    )
    monkeypatch.setattr(
        rsm_operation,
        "_MAX_RSM_RESIDENT_BYTES",
        old_projection_only_charge + 1,
    )
    monkeypatch.setattr(
        SpecSource,
        "load_frame",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("finalization accounting must precede detector decode")
        ),
    )

    result = run_rsm_operation_v2(prepared)

    assert result.terminal.disposition is ModuleDisposition.REFUSED
    assert result.terminal.code == "Q_MEMORY_LIMIT_EXCEEDED"
    assert not Path(prepared.module.output.target).exists()


def test_declared_raw_dtype_fences_cross_chunk_decoder_drift(tmp_path, monkeypatch):
    import xrd_tools.analysis.rsm_operation as rsm_operation

    monkeypatch.setattr(
        rsm_operation,
        "_resolve_exact_rsm_q_bounds_active",
        lambda *_args, **_kwargs: (
            (-8.0, 8.0),
            (-8.0, 8.0),
            (-8.0, 8.0),
        ),
    )
    prepared = prepare_rsm_tool_v2(
        _form(tmp_path, (_write_member(tmp_path, 0),))
    ).request

    def drifting_dtype(_self, index):
        dtype = np.float64 if int(index) == 0 else np.float32
        return np.full((195, 487), 10 + int(index), dtype=dtype)

    monkeypatch.setattr(SpecSource, "load_frame", drifting_dtype)
    result = run_rsm_operation_v2(prepared)

    assert result.terminal.disposition is ModuleDisposition.REFUSED
    assert result.terminal.code == "FRAME_LAYOUT_INVALID"
    assert not Path(prepared.module.output.target).exists()


@pytest.mark.parametrize(
    ("drift_kind", "expected_code"),
    (
        ("source", "SOURCE_REVISION_CHANGED"),
        ("asset", "RSM_GEOMETRY_ASSET_IDENTITY_MISMATCH"),
        ("output-parent", "OUTPUT_IDENTITY_MISMATCH"),
    ),
)
def test_v2_prepublish_drift_refuses_without_committing(
    tmp_path,
    monkeypatch,
    drift_kind,
    expected_code,
):
    import xrd_tools.analysis.rsm_operation as rsm_operation

    monkeypatch.setattr(
        rsm_operation,
        "_resolve_exact_rsm_q_bounds_active",
        lambda *_args, **_kwargs: (
            (-8.0, 8.0),
            (-8.0, 8.0),
            (-8.0, 8.0),
        ),
    )
    member = _write_member(tmp_path, 0)
    prepared = prepare_rsm_tool_v2(_form(tmp_path, (member,))).request
    output = Path(prepared.module.output.target)
    monkeypatch.setattr(
        SpecSource,
        "load_frame",
        lambda _self, index: np.full(
            (195, 487),
            10 + int(index),
            dtype=np.float64,
        ),
    )
    real_write = rsm_operation.write_rsm
    if drift_kind == "source":
        drift_path = Path(member.spec_path)
        original = drift_path.stat()

        def drift() -> None:
            os.utime(
                drift_path,
                ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000),
            )

        def restore() -> None:
            os.utime(
                drift_path,
                ns=(original.st_atime_ns, original.st_mtime_ns),
            )

    elif drift_kind == "asset":
        drift_path = tmp_path / CANONICAL_RSM_GEOMETRY_LOCATOR
        original = drift_path.stat()

        def drift() -> None:
            os.utime(
                drift_path,
                ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000),
            )

        def restore() -> None:
            os.utime(
                drift_path,
                ns=(original.st_atime_ns, original.st_mtime_ns),
            )

    else:
        drift_path = output.parent
        original_mode = stat.S_IMODE(drift_path.stat().st_mode)
        changed_mode = original_mode ^ stat.S_IXGRP

        def drift() -> None:
            os.chmod(drift_path, changed_mode)

        def restore() -> None:
            os.chmod(drift_path, original_mode)

    writes = []

    def write_then_drift(*args, **kwargs):
        real_write(*args, **kwargs)
        writes.append(True)
        drift()

    monkeypatch.setattr(rsm_operation, "write_rsm", write_then_drift)
    try:
        result = run_rsm_operation_v2(prepared)
    finally:
        restore()

    assert writes == [True]
    assert result.terminal.disposition is ModuleDisposition.REFUSED
    assert result.terminal.code == expected_code
    assert not output.exists()


@pytest.mark.parametrize("cancel_after_chunks", (1, 2))
def test_v2_cancellation_at_science_boundaries_restores_runtime(
    tmp_path,
    monkeypatch,
    cancel_after_chunks,
):
    import xrayutilities as xu
    import xrd_tools.analysis.rsm_operation as rsm_operation

    monkeypatch.setattr(
        rsm_operation,
        "_resolve_exact_rsm_q_bounds_active",
        lambda *_args, **_kwargs: (
            (-8.0, 8.0),
            (-8.0, 8.0),
            (-8.0, 8.0),
        ),
    )
    member = _write_member(tmp_path, 0)
    prepared = prepare_rsm_tool_v2(_form(tmp_path, (member,))).request
    monkeypatch.setattr(
        SpecSource,
        "load_frame",
        lambda _self, index: np.full(
            (195, 487),
            10 + int(index),
            dtype=np.float64,
        ),
    )
    real_add = rsm_operation.StreamingGridder.add_leased
    calls = []

    def counted_add(*args, **kwargs):
        result = real_add(*args, **kwargs)
        calls.append(result.frame_count)
        return result

    monkeypatch.setattr(
        rsm_operation.StreamingGridder,
        "add_leased",
        counted_add,
    )
    monkeypatch.setattr(
        rsm_operation.StreamingGridder,
        "to_volume",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cancelled science must not finalize the volume")
        ),
    )
    cancelled = threading.Event()

    def cancel_at_boundary(progress):
        if (
            progress.stage == "science"
            and len(calls) == cancel_after_chunks
            and not cancelled.is_set()
        ):
            cancelled.set()

    before = xu.config.NTHREADS
    result = run_rsm_operation_v2(
        prepared,
        cancel_token=cancelled,
        progress_callback=cancel_at_boundary,
    )

    assert result.terminal.disposition is ModuleDisposition.CANCELLED
    assert result.terminal.code == "CANCELLED"
    assert calls == [1] * cancel_after_chunks
    assert xu.config.NTHREADS == before
    assert not Path(prepared.module.output.target).exists()


@pytest.mark.parametrize("cancel_after_mask_frames", (1, 2))
def test_v2_cancellation_at_static_mask_boundaries_prevents_science(
    tmp_path,
    monkeypatch,
    cancel_after_mask_frames,
):
    import xrayutilities as xu
    import xrd_tools.analysis.rsm_operation as rsm_operation

    monkeypatch.setattr(
        rsm_operation,
        "_resolve_exact_rsm_q_bounds_active",
        lambda *_args, **_kwargs: (
            (-8.0, 8.0),
            (-8.0, 8.0),
            (-8.0, 8.0),
        ),
    )
    form = replace(
        _form(tmp_path, (_write_member(tmp_path, 0),)),
        conditioning=RSMImageConditioning(0.0, None, 100.0),
    )
    prepared = prepare_rsm_tool_v2(form).request
    monkeypatch.setattr(
        SpecSource,
        "load_frame",
        lambda _self, index: np.full(
            (195, 487),
            10 + int(index),
            dtype=np.float64,
        ),
    )
    monkeypatch.setattr(
        rsm_operation.StreamingGridder,
        "add_leased",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cancelled static-mask pass must not start science")
        ),
    )
    cancelled = threading.Event()
    mask_callbacks = []

    def cancel_at_mask_boundary(progress):
        if progress.stage == "static-mask":
            mask_callbacks.append(progress.completed)
            if len(mask_callbacks) == cancel_after_mask_frames:
                cancelled.set()

    before = xu.config.NTHREADS
    result = run_rsm_operation_v2(
        prepared,
        cancel_token=cancelled,
        progress_callback=cancel_at_mask_boundary,
    )

    assert result.terminal.disposition is ModuleDisposition.CANCELLED
    assert result.terminal.code == "CANCELLED"
    assert len(mask_callbacks) == cancel_after_mask_frames
    assert xu.config.NTHREADS == before
    assert not Path(prepared.module.output.target).exists()


def test_v2_cancellation_on_final_source_close_prevents_finalization(
    tmp_path,
    monkeypatch,
):
    from contextlib import contextmanager

    import xrayutilities as xu
    import xrd_tools.analysis.rsm_operation as rsm_operation

    monkeypatch.setattr(
        rsm_operation,
        "_resolve_exact_rsm_q_bounds_active",
        lambda *_args, **_kwargs: (
            (-8.0, 8.0),
            (-8.0, 8.0),
            (-8.0, 8.0),
        ),
    )
    prepared = prepare_rsm_tool_v2(
        _form(tmp_path, (_write_member(tmp_path, 0),))
    ).request
    monkeypatch.setattr(
        SpecSource,
        "load_frame",
        lambda _self, index: np.full(
            (195, 487),
            10 + int(index),
            dtype=np.float64,
        ),
    )
    cancelled = threading.Event()
    real_source_lease = rsm_operation.requalified_analysis_source

    @contextmanager
    def cancel_on_source_close(*args, **kwargs):
        with real_source_lease(*args, **kwargs) as source:
            yield source
        cancelled.set()

    monkeypatch.setattr(
        rsm_operation,
        "requalified_analysis_source",
        cancel_on_source_close,
    )
    monkeypatch.setattr(
        rsm_operation.StreamingGridder,
        "to_volume",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cancelled source close must not finalize the volume")
        ),
    )

    before = xu.config.NTHREADS
    result = run_rsm_operation_v2(prepared, cancel_token=cancelled)

    assert result.terminal.disposition is ModuleDisposition.CANCELLED
    assert result.terminal.code == "CANCELLED"
    assert xu.config.NTHREADS == before
    assert not Path(prepared.module.output.target).exists()


@pytest.mark.parametrize(
    ("name", "mutated"),
    (
        ("EPSILON", 2e-8),
        ("DIGITS", 7),
        ("NTHREADS", 9),
    ),
)
def test_v2_refuses_restores_and_never_attests_xu_global_mutation(
    tmp_path,
    monkeypatch,
    name,
    mutated,
):
    import xrayutilities as xu
    import xrd_tools.analysis.rsm_operation as rsm_operation

    monkeypatch.setattr(
        rsm_operation,
        "_resolve_exact_rsm_q_bounds_active",
        lambda *_args, **_kwargs: (
            (-8.0, 8.0),
            (-8.0, 8.0),
            (-8.0, 8.0),
        ),
    )
    prepared = prepare_rsm_tool_v2(
        _form(tmp_path, (_write_member(tmp_path, 0),))
    ).request
    monkeypatch.setattr(
        SpecSource,
        "load_frame",
        lambda _self, index: np.full(
            (195, 487),
            10 + int(index),
            dtype=np.float64,
        ),
    )
    before = (xu.config.EPSILON, xu.config.DIGITS, xu.config.NTHREADS)
    changed = False

    def mutate_after_science(progress):
        nonlocal changed
        if progress.stage == "science" and not changed:
            setattr(xu.config, name, mutated)
            changed = True

    result = run_rsm_operation_v2(
        prepared,
        progress_callback=mutate_after_science,
    )

    assert changed is True
    assert result.terminal.disposition is ModuleDisposition.REFUSED
    assert result.terminal.code == "XU_RUNTIME_CONFIG_MUTATED"
    assert (xu.config.EPSILON, xu.config.DIGITS, xu.config.NTHREADS) == before
    assert not Path(prepared.module.output.target).exists()


def test_transient_v2_reload_retry_never_replays_science_or_writer(
    tmp_path,
    monkeypatch,
):
    import xrd_tools.analysis.rsm_operation as rsm_operation

    monkeypatch.setattr(
        rsm_operation,
        "_resolve_exact_rsm_q_bounds_active",
        lambda *_args, **_kwargs: (
            (-8.0, 8.0),
            (-8.0, 8.0),
            (-8.0, 8.0),
        ),
    )
    member = _write_member(tmp_path, 0)
    prepared = prepare_rsm_tool_v2(_form(tmp_path, (member,))).request
    monkeypatch.setattr(
        SpecSource,
        "load_frame",
        lambda _self, index: np.full(
            (195, 487),
            10 + int(index),
            dtype=np.float64,
        ),
    )
    counts = {"science": 0, "writer": 0, "read": 0}
    real_add = rsm_operation.StreamingGridder.add_leased
    real_write = rsm_operation.write_rsm
    real_read = rsm_operation.read_analysis_artifact

    def counted_add(*args, **kwargs):
        counts["science"] += 1
        return real_add(*args, **kwargs)

    def counted_write(*args, **kwargs):
        counts["writer"] += 1
        return real_write(*args, **kwargs)

    def transient_read(*args, **kwargs):
        counts["read"] += 1
        if counts["read"] == 1:
            raise OSError("transient detached reload fault")
        return real_read(*args, **kwargs)

    monkeypatch.setattr(rsm_operation.StreamingGridder, "add_leased", counted_add)
    monkeypatch.setattr(rsm_operation, "write_rsm", counted_write)
    monkeypatch.setattr(rsm_operation, "read_analysis_artifact", transient_read)
    execution = RSMOperationExecutionV2(prepared)
    with pytest.raises(RSMOperationVerificationErrorV2) as failed:
        execution.run()
    assert failed.value.execution is execution
    recovered = execution.retry_verification()
    assert recovered.terminal.disposition is ModuleDisposition.COMMITTED
    assert execution.retry_verification() is recovered
    assert counts == {"science": 2, "writer": 1, "read": 2}


def test_mutated_mask_receipt_reload_failure_retains_exact_execution(
    tmp_path,
    monkeypatch,
):
    import xrd_tools.analysis.rsm_operation as rsm_operation

    monkeypatch.setattr(
        rsm_operation,
        "_resolve_exact_rsm_q_bounds_active",
        lambda *_args, **_kwargs: (
            (-8.0, 8.0),
            (-8.0, 8.0),
            (-8.0, 8.0),
        ),
    )
    prepared = prepare_rsm_tool_v2(
        _form(tmp_path, (_write_member(tmp_path, 0),))
    ).request
    monkeypatch.setattr(
        SpecSource,
        "load_frame",
        lambda _self, index: np.full(
            (195, 487),
            10 + int(index),
            dtype=np.float64,
        ),
    )
    counts = {"science": 0, "writer": 0, "read": 0}
    real_add = rsm_operation.StreamingGridder.add_leased
    real_write = rsm_operation.write_rsm
    real_read = rsm_operation.read_analysis_artifact
    execution = RSMOperationExecutionV2(prepared)

    def counted_add(*args, **kwargs):
        counts["science"] += 1
        return real_add(*args, **kwargs)

    def counted_write(*args, **kwargs):
        counts["writer"] += 1
        return real_write(*args, **kwargs)

    def mutate_after_read(*args, **kwargs):
        counts["read"] += 1
        payload = real_read(*args, **kwargs)
        if counts["read"] == 1:
            object.__setattr__(
                execution._mask_receipts[0],
                "fingerprint",
                "0" * 64,
            )
        return payload

    monkeypatch.setattr(rsm_operation.StreamingGridder, "add_leased", counted_add)
    monkeypatch.setattr(rsm_operation, "write_rsm", counted_write)
    monkeypatch.setattr(rsm_operation, "read_analysis_artifact", mutate_after_read)

    with pytest.raises(RSMOperationVerificationErrorV2) as failed:
        execution.run()
    assert failed.value.execution is execution
    assert Path(prepared.module.output.target).is_file()
    with pytest.raises(RSMOperationVerificationErrorV2) as retried:
        execution.retry_verification()
    assert retried.value.execution is execution
    assert counts == {"science": 2, "writer": 1, "read": 2}


def test_v2_cleanup_retry_never_replays_science_or_writer(
    tmp_path,
    monkeypatch,
):
    import xrd_tools.analysis.rsm_operation as rsm_operation
    import xrd_tools.io.output_transaction as transaction_api

    monkeypatch.setattr(
        rsm_operation,
        "_resolve_exact_rsm_q_bounds_active",
        lambda *_args, **_kwargs: (
            (-8.0, 8.0),
            (-8.0, 8.0),
            (-8.0, 8.0),
        ),
    )
    member = _write_member(tmp_path, 0)
    initial = _form(tmp_path, (member,))
    target = Path(initial.output_path)
    target.write_bytes(b"prior operator output")
    prepared = prepare_rsm_tool_v2(
        replace(initial, overwrite=AnalysisArtifactOverwrite.REPLACE)
    ).request
    monkeypatch.setattr(
        SpecSource,
        "load_frame",
        lambda _self, index: np.full(
            (195, 487),
            10 + int(index),
            dtype=np.float64,
        ),
    )
    real_unlink = transaction_api._unlink
    real_add = rsm_operation.StreamingGridder.add_leased
    real_write = rsm_operation.write_rsm
    failures = []
    counts = {"science": 0, "writer": 0}

    def fail_backup_once(path):
        if ".xdart-replacing-" in Path(path).name and not failures:
            failures.append("backup")
            raise OSError("backup cleanup fault")
        return real_unlink(path)

    def counted_add(*args, **kwargs):
        counts["science"] += 1
        return real_add(*args, **kwargs)

    def counted_write(*args, **kwargs):
        counts["writer"] += 1
        return real_write(*args, **kwargs)

    monkeypatch.setattr(transaction_api, "_unlink", fail_backup_once)
    monkeypatch.setattr(rsm_operation.StreamingGridder, "add_leased", counted_add)
    monkeypatch.setattr(rsm_operation, "write_rsm", counted_write)
    with pytest.raises(RSMOperationCleanupPendingV2) as pending:
        run_rsm_operation_v2(prepared)
    recovered = pending.value.retry_cleanup()
    assert recovered.terminal.disposition is ModuleDisposition.COMMITTED
    assert recovered.payload is not None
    assert pending.value.execution.retry_cleanup() is recovered
    assert failures == ["backup"]
    assert counts == {"science": 2, "writer": 1}
