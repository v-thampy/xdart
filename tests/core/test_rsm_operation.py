from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

import xrd_tools.analysis.rsm_operation as rsm_operation

from xrd_tools.analysis.module_transaction import (
    MetadataColumnSelector,
    ModuleDisposition,
    ModuleKind,
    ModuleOutputRequest,
    ModuleSourceReceipt,
    ModuleTerminalResult,
)
from xrd_tools.analysis.rsm_operation import (
    RSMDetectorGeometry,
    RSMImageConditioning,
    RSMNormalizationMode,
    RSMNormalizationPolicy,
    RSMOperationPlan,
    RSMOperationRefused,
    RSMOperationCleanupPending,
    RSMOperationExecution,
    RSMOperationVerificationError,
    RSMOperationResult,
    condition_rsm_images,
    prepare_rsm_operation,
    required_rsm_selectors,
    resolve_exact_rsm_q_bounds,
    rsm_normalization_divisors,
    rsm_static_hot_mask,
    run_rsm_operation,
)
from xrd_tools.analysis.scan_operations import (
    AnalysisDisposition,
    MetadataTablePlan,
    run_metadata_table,
)
from xrd_tools.core.geometry import (
    DetectorHeader,
    Diffractometer,
    ImageOrientation,
    PixelQMap,
)
from xrd_tools.core.scan import SourceKind
from xrd_tools.io.analysis_artifact import (
    AnalysisArtifactKind,
    AnalysisArtifactOverwrite,
)
from xrd_tools.sources.spec import SpecSource


def _geometry(
    *,
    header: DetectorHeader | None = None,
    roi=(0, -1, 0, -1),
) -> RSMDetectorGeometry:
    header = header or DetectorHeader(
        cch1=97.0,
        cch2=243.0,
        pwidth1=0.172,
        pwidth2=0.172,
        distance=1014.7173,
        Nch1=195,
        Nch2=487,
    )
    return RSMDetectorGeometry(
        header,
        tuple(
            (role, MetadataColumnSelector(role, 0))
            for role in ("mu", "eta", "chi", "phi", "nu", "del")
        ),
        ImageOrientation(),
        roi,
    )


def _notebook_normalization(
    *,
    seconds_occurrence: int = 0,
) -> RSMNormalizationPolicy:
    return RSMNormalizationPolicy(
        RSMNormalizationMode.FOIL_TRANSMISSION_EXPOSURE,
        MetadataColumnSelector("foil status", 0),
        MetadataColumnSelector("Seconds", seconds_occurrence),
        (1.06, 3.04, 4.65, 9.5),
    )


def _plan(*, bins=(40, 40, 40)) -> RSMOperationPlan:
    return RSMOperationPlan(
        _geometry(),
        RSMImageConditioning(1.0e-6, 2.0e10, 100.0),
        _notebook_normalization(),
        bins=bins,
        chunk_size=8,
    )


def _synthetic_spec_source(
    tmp_path: Path,
    *,
    selected: tuple[int, ...] = (0, 2),
    bins: tuple[int, int, int] = (4, 5, 6),
    max_frame_bytes: int = 1024 * 1024,
    max_chunk_bytes: int = 1024 * 1024,
):
    source_root = tmp_path / "source"
    image_root = source_root / "images"
    image_root.mkdir(parents=True)
    spec_path = source_root / "RSM_synth"
    spec_path.write_text(
        """#F RSM_synth
#E 1
#D Mon Jan 15 10:30:00 2024
#O0 energy  mu  chi  phi  nu  del

#S 1 ascan eta 0 2 2 1
#D Mon Jan 15 10:30:00 2024
#P0 13000.007 10 20 30 40 50
#G3 1 0 0 0 1 0 0 0 1
#N 4
#L eta  Seconds  Seconds  foil status
0 1 10 101
1 2 20 110
2 3 40 1010
""",
        encoding="utf-8",
    )
    frames = (
        np.asarray(
            [
                [200, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
            ],
            dtype=np.uint16,
        ),
        np.arange(20, dtype=np.uint16).reshape(4, 5) + 30,
        np.asarray(
            [
                [200, 22, 23, 24, 25],
                [26, 27, 28, 29, 30],
                [31, 32, 33, 34, 35],
                [36, 37, 38, 39, 40],
            ],
            dtype=np.uint16,
        ),
    )
    for index, frame in enumerate(frames):
        tifffile.imwrite(image_root / f"frame_{index:04d}.tif", frame)
    plan = RSMOperationPlan(
        RSMDetectorGeometry(
            DetectorHeader(1.5, 2.0, 0.172, 0.172, 1014.7173, 4, 5),
            tuple(
                (role, MetadataColumnSelector(role, 0))
                for role in ("mu", "eta", "chi", "phi", "nu", "del")
            ),
            ImageOrientation(),
            (0, -1, 0, -1),
        ),
        RSMImageConditioning(1.0e-6, 2.0e10, 100.0),
        _notebook_normalization(seconds_occurrence=1),
        bins=bins,
        chunk_size=1,
        max_frame_bytes=max_frame_bytes,
        max_chunk_bytes=max_chunk_bytes,
    )
    selectors = required_rsm_selectors(plan)
    table = run_metadata_table(
        MetadataTablePlan(
            spec_path,
            kind=SourceKind.SPEC,
            scan="1.1",
            image_dir=image_root,
            image_stem="frame_",
            column_projection=tuple(
                (selector.name, selector.occurrence) for selector in selectors
            ),
        )
    )
    assert table.disposition is AnalysisDisposition.COMPLETED
    source = ModuleSourceReceipt.from_metadata_table(
        table,
        kind=ModuleKind.RSM,
        selected_labels=selected,
        resolved_selectors=selectors,
    )
    output_root = tmp_path / "output"
    output_root.mkdir()
    output = ModuleOutputRequest(
        output_root / "rsm.nexus",
        AnalysisArtifactKind.RSM,
        AnalysisArtifactOverwrite.CREATE_NEW,
    )
    return source, output, plan, frames


def _prepared_rsm(tmp_path: Path, **kwargs):
    source, output, plan, frames = _synthetic_spec_source(tmp_path, **kwargs)
    request = prepare_rsm_operation(
        source,
        output,
        plan,
        project_root=tmp_path,
    )
    return request, frames


def _synthetic_volume(request, *, intensity=None):
    axes = tuple(
        np.linspace(bounds[0], bounds[1], size)
        for bounds, size in zip(request.preflight.q_bounds, request.plan.bins)
    )
    values = (
        np.zeros(request.plan.bins, dtype=float)
        if intensity is None
        else np.asarray(intensity)
    )
    return rsm_operation.RSMVolume(*axes, values)


def test_normalization_policy_is_mandatory_and_plan_fingerprint_bound():
    plan = _plan()
    assert len(plan.fingerprint) == 64
    assert plan.normalization.exposure_selector == MetadataColumnSelector(
        "Seconds", 0
    )

    changed = RSMOperationPlan(
        plan.geometry,
        plan.conditioning,
        _notebook_normalization(seconds_occurrence=1),
        bins=plan.bins,
        chunk_size=plan.chunk_size,
    )
    assert changed.fingerprint != plan.fingerprint

    with pytest.raises(TypeError, match="typed policies"):
        RSMOperationPlan(plan.geometry, plan.conditioning, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="8,000,000"):
        _plan(bins=(201, 200, 200))

    overlapping = RSMNormalizationPolicy(
        RSMNormalizationMode.FOIL_TRANSMISSION_EXPOSURE,
        MetadataColumnSelector("mu", 0),
        MetadataColumnSelector("Seconds", 0),
        (1.06, 3.04, 4.65, 9.5),
    )
    with pytest.raises(ValueError, match="disjoint from motor selectors"):
        RSMOperationPlan(
            plan.geometry,
            plan.conditioning,
            overlapping,
            bins=plan.bins,
        )


def test_required_projection_keeps_exact_seconds_occurrence():
    selectors = required_rsm_selectors(
        RSMOperationPlan(
            _geometry(),
            RSMImageConditioning(0.0, None, None),
            _notebook_normalization(seconds_occurrence=1),
            bins=(2, 2, 2),
        )
    )
    assert MetadataColumnSelector("Seconds", 1) in selectors
    assert MetadataColumnSelector("Seconds", 0) not in selectors
    assert len(selectors) == 8


def test_foil_and_seconds_divisor_matches_notebook_formula():
    foil = np.asarray([101.0, 110.0, 1010.0, 1101.0])
    seconds = np.asarray([1.0, 2.0, 0.5, 3.0])
    actual = rsm_normalization_divisors(
        _notebook_normalization(),
        4,
        foil_status=foil,
        exposure_seconds=seconds,
    )
    digits = np.asarray(
        [
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 1],
        ],
        dtype=float,
    )
    expected = np.exp(-digits @ np.asarray([1.06, 3.04, 4.65, 9.5])) * seconds
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
    assert actual.flags.writeable is False


@pytest.mark.parametrize(
    ("foil", "seconds", "code"),
    (
        ([np.nan], [1.0], "FOIL_STATUS_INVALID"),
        ([1.5], [1.0], "FOIL_STATUS_INVALID"),
        ([-1.0], [1.0], "FOIL_STATUS_INVALID"),
        ([10000.0], [1.0], "FOIL_STATUS_INVALID"),
        ([101.0], [0.0], "EXPOSURE_INVALID"),
        ([101.0], [np.inf], "EXPOSURE_INVALID"),
    ),
)
def test_normalization_rejects_invalid_frame_values(foil, seconds, code):
    with pytest.raises(RSMOperationRefused) as refused:
        rsm_normalization_divisors(
            _notebook_normalization(),
            1,
            foil_status=np.asarray(foil),
            exposure_seconds=np.asarray(seconds),
        )
    assert refused.value.code == code


def test_normalization_rejects_underflowed_transmission():
    policy = RSMNormalizationPolicy(
        RSMNormalizationMode.FOIL_TRANSMISSION_EXPOSURE,
        MetadataColumnSelector("foil status", 0),
        MetadataColumnSelector("Seconds", 0),
        (1.0e308, 1.0e308, 1.0e308, 1.0e308),
    )
    with pytest.raises(RSMOperationRefused) as refused:
        rsm_normalization_divisors(
            policy,
            1,
            foil_status=np.asarray([9999.0]),
            exposure_seconds=np.asarray([1.0]),
        )
    assert refused.value.code == "TRANSMISSION_INVALID"


def test_conditioning_and_static_hot_mask_match_notebook_rule():
    conditioning = RSMImageConditioning(1.0e-6, 2.0e10, 100.0)
    frames = np.asarray(
        [
            [[200.0, 2.0e10 + 1.0], [2.0, 3.0]],
            [[200.0, 2.0e10 + 1.0], [4.0, 3.0]],
            [[200.0, 2.0e10 + 1.0], [2.0, 3.0]],
        ]
    )
    mask = rsm_static_hot_mask((frames[:2], frames[2:]), conditioning)
    np.testing.assert_array_equal(mask, [[True, False], [False, False]])
    assert mask.flags.writeable is False

    conditioned = condition_rsm_images(frames[:1], conditioning, static_mask=mask)
    assert np.isnan(conditioned[0, 0, 0])
    assert np.isnan(conditioned[0, 0, 1])
    assert conditioned[0, 1, 0] == 2.000001


def test_exact_q_preflight_uses_every_cropped_pixel_and_is_chunked(monkeypatch):
    header = DetectorHeader(1.0, 2.0, 0.1, 0.1, 100.0, 5, 6)
    mapper = PixelQMap(Diffractometer.psic(), header)
    calls = []

    def pixel_q(self, angles, energy, *, UB, roi, image_shape):
        calls.append((tuple(len(item) for item in angles), image_shape, roi))
        n, rows, columns = image_shape
        row = np.arange(rows, dtype=float)[None, :, None]
        column = np.arange(columns, dtype=float)[None, None, :]
        qx = np.broadcast_to(row, image_shape) + angles[0][:, None, None]
        qy = np.broadcast_to(column, image_shape) + angles[1][:, None, None]
        qz = np.broadcast_to(row * 10.0 + column, image_shape)
        qz = qz + angles[2][:, None, None]
        return qx, qy, qz

    monkeypatch.setattr(PixelQMap, "pixel_q", pixel_q)
    angles = tuple(
        np.asarray(values, dtype=float)
        for values in (
            [0, 10, 20],
            [1, 2, 3],
            [4, 5, 6],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        )
    )
    bounds = resolve_exact_rsm_q_bounds(
        mapper,
        angles,
        13000.007,
        np.eye(3),
        roi=(0, -1, 0, -1),
        chunk_size=2,
        max_frame_bytes=1024 * 1024,
        max_chunk_bytes=1024 * 1024,
    )
    assert calls == [
        ((2, 2, 2, 2, 2, 2), (2, 4, 5), (0, -1, 0, -1)),
        ((1, 1, 1, 1, 1, 1), (1, 4, 5), (0, -1, 0, -1)),
    ]
    assert bounds == ((0.0, 23.0), (1.0, 7.0), (4.0, 40.0))


def test_exact_q_preflight_honors_pre_cancel_without_mapping(monkeypatch):
    mapper = PixelQMap(
        Diffractometer.psic(),
        DetectorHeader(1.0, 2.0, 0.1, 0.1, 100.0, 5, 6),
    )
    calls = []

    def pixel_q(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("q mapping must not start")

    monkeypatch.setattr(PixelQMap, "pixel_q", pixel_q)
    cancel = Event()
    cancel.set()
    with pytest.raises(RSMOperationRefused) as refused:
        resolve_exact_rsm_q_bounds(
            mapper,
            tuple(np.asarray([0.0]) for _ in range(6)),
            13000.007,
            np.eye(3),
            roi=None,
            chunk_size=1,
            max_frame_bytes=1024 * 1024,
            max_chunk_bytes=1024 * 1024,
            cancel_token=cancel,
        )
    assert refused.value.code == "CANCELLED"
    assert calls == []


@pytest.mark.parametrize(
    (
        "frame_count",
        "chunk_size",
        "max_frame_bytes",
        "max_chunk_bytes",
        "expected_code",
    ),
    (
        (1, 1, 239, 1024 * 1024, "FRAME_MEMORY_LIMIT_EXCEEDED"),
        (2, 2, 1024 * 1024, 1439, "Q_MEMORY_LIMIT_EXCEEDED"),
    ),
)
def test_exact_q_preflight_enforces_memory_before_mapping(
    monkeypatch,
    frame_count,
    chunk_size,
    max_frame_bytes,
    max_chunk_bytes,
    expected_code,
):
    mapper = PixelQMap(
        Diffractometer.psic(),
        DetectorHeader(1.0, 2.0, 0.1, 0.1, 100.0, 5, 6),
    )
    calls = []

    def pixel_q(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("q mapping must not start")

    monkeypatch.setattr(PixelQMap, "pixel_q", pixel_q)
    with pytest.raises(RSMOperationRefused) as refused:
        resolve_exact_rsm_q_bounds(
            mapper,
            tuple(np.zeros(frame_count) for _ in range(6)),
            13000.007,
            np.eye(3),
            roi=None,
            chunk_size=chunk_size,
            max_frame_bytes=max_frame_bytes,
            max_chunk_bytes=max_chunk_bytes,
        )
    assert refused.value.code == expected_code
    assert calls == []


def test_tiny_roi_cannot_bypass_full_frame_admission(tmp_path, monkeypatch):
    source, output, base, _frames = _synthetic_spec_source(tmp_path)
    header = base.geometry.header
    plan = RSMOperationPlan(
        RSMDetectorGeometry(
            DetectorHeader(
                header.cch1,
                header.cch2,
                header.pwidth1,
                header.pwidth2,
                header.distance,
                1000,
                1000,
            ),
            base.geometry.motor_selectors,
            base.geometry.image_orientation,
            (0, 2, 0, 2),
        ),
        base.conditioning,
        base.normalization,
        bins=base.bins,
        chunk_size=base.chunk_size,
        max_frame_bytes=1024,
        max_chunk_bytes=1024 * 1024,
    )
    calls = []

    def forbidden(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("prepare must reject before image decode or q mapping")

    monkeypatch.setattr(SpecSource, "load_frame", forbidden)
    monkeypatch.setattr(PixelQMap, "pixel_q", forbidden)
    with pytest.raises(RSMOperationRefused) as refused:
        prepare_rsm_operation(source, output, plan, project_root=tmp_path)
    assert refused.value.code == "FRAME_MEMORY_LIMIT_EXCEEDED"
    assert calls == []


def test_detector_contract_rejects_noncanonical_r1_geometry():
    geometry = _geometry()
    with pytest.raises(ValueError, match="identity orientation"):
        RSMDetectorGeometry(
            geometry.header,
            geometry.motor_selectors,
            ImageOrientation(rotation=180),
            geometry.roi,
        )
    with pytest.raises(ValueError, match="contained"):
        _geometry(roi=(0, 196, 0, 487))


def test_prepare_is_image_free_and_binds_exact_spec_science(tmp_path, monkeypatch):
    source, output, plan, _frames = _synthetic_spec_source(tmp_path)

    def no_decode(*_args, **_kwargs):
        raise AssertionError("RSM preflight must not decode detector images")

    monkeypatch.setattr(SpecSource, "load_frame", no_decode)
    request = prepare_rsm_operation(
        source,
        output,
        plan,
        project_root=tmp_path,
    )
    preflight = request.preflight
    assert preflight.energy_eV == 13000.007
    assert preflight.ub == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    assert preflight.detector_shape == (4, 5)
    assert preflight.cropped_shape == (3, 4)
    assert [item.label for item in preflight.contributions] == [0, 2]
    values = [_contribution.values for _contribution in preflight.contributions]
    assert [("Seconds", 1, 10.0) in item for item in values] == [True, False]
    assert ("Seconds", 1, 40.0) in values[1]
    expected_divisors = rsm_normalization_divisors(
        plan.normalization,
        2,
        foil_status=np.asarray([101.0, 1010.0]),
        exposure_seconds=np.asarray([10.0, 40.0]),
    )
    np.testing.assert_allclose(
        [item.normalization_divisor for item in preflight.contributions],
        expected_divisors,
        rtol=0,
        atol=0,
    )
    assert preflight.source_relative_path == "source/RSM_synth"
    assert json.loads(preflight.source_options_json)["image_dir"] == "source/images"
    assert request.provenance["source"]["preflight"] == preflight.to_provenance()

    def absolute_strings(value):
        if type(value) is str:
            return [value] if Path(value).is_absolute() else []
        if type(value) is dict:
            return [
                item
                for child in value.values()
                for item in absolute_strings(child)
            ]
        if type(value) is list:
            return [item for child in value for item in absolute_strings(child)]
        return []

    assert absolute_strings(request.provenance) == []
    assert request.module.plan_fingerprint == plan.fingerprint


@pytest.mark.parametrize(
    "overwrite",
    (
        AnalysisArtifactOverwrite.CREATE_NEW,
        AnalysisArtifactOverwrite.REPLACE,
    ),
)
def test_prepare_refuses_symbolic_link_output_target(tmp_path, overwrite):
    source, _output, plan, _frames = _synthetic_spec_source(tmp_path)
    target = tmp_path / "output" / "rsm.nexus"
    target.symlink_to(tmp_path / "operator-owned.nexus")
    output = ModuleOutputRequest(
        target,
        AnalysisArtifactKind.RSM,
        overwrite,
    )

    with pytest.raises(RSMOperationRefused) as refused:
        prepare_rsm_operation(source, output, plan, project_root=tmp_path)

    assert refused.value.code == "OUTPUT_SYMLINK_UNSUPPORTED"


def test_prepare_canonicalizes_output_parent_alias(tmp_path):
    source, _output, plan, _frames = _synthetic_spec_source(tmp_path)
    admitted = tmp_path / "admitted"
    outside = tmp_path / "outside"
    admitted.mkdir()
    outside.mkdir()
    alias = tmp_path / "output-alias"
    alias.symlink_to(admitted, target_is_directory=True)
    output = ModuleOutputRequest(
        alias / "rsm.nexus",
        AnalysisArtifactKind.RSM,
        AnalysisArtifactOverwrite.CREATE_NEW,
    )

    request = prepare_rsm_operation(
        source,
        output,
        plan,
        project_root=tmp_path,
    )
    assert request.module.output.target == str(admitted / "rsm.nexus")

    alias.unlink()
    alias.symlink_to(outside, target_is_directory=True)
    result = run_rsm_operation(request)

    assert result.terminal.disposition is ModuleDisposition.COMMITTED
    assert (admitted / "rsm.nexus").is_file()
    assert not (outside / "rsm.nexus").exists()


def test_run_refuses_replaced_canonical_output_parent_before_science(
    tmp_path,
    monkeypatch,
):
    source, _output, plan, _frames = _synthetic_spec_source(tmp_path)
    admitted = tmp_path / "admitted"
    outside = tmp_path / "outside"
    admitted.mkdir()
    outside.mkdir()
    output = ModuleOutputRequest(
        admitted / "rsm.nexus",
        AnalysisArtifactKind.RSM,
        AnalysisArtifactOverwrite.CREATE_NEW,
    )
    request = prepare_rsm_operation(
        source,
        output,
        plan,
        project_root=tmp_path,
    )
    held = tmp_path / "admitted-before-retarget"
    admitted.rename(held)
    admitted.symlink_to(outside, target_is_directory=True)
    science_calls = []

    def forbidden_science(*_args, **_kwargs):
        science_calls.append(True)
        raise AssertionError("output authority must be requalified before science")

    monkeypatch.setattr(rsm_operation, "run_rsm", forbidden_science)
    result = run_rsm_operation(request)

    assert result.terminal.disposition is ModuleDisposition.REFUSED
    assert result.terminal.code == "OUTPUT_IDENTITY_MISMATCH"
    assert science_calls == []
    assert not (outside / "rsm.nexus").exists()


def test_run_refuses_same_path_replacement_output_parent_before_science(
    tmp_path,
    monkeypatch,
):
    request, _frames = _prepared_rsm(tmp_path)
    parent = Path(request.module.output.target).parent
    held = tmp_path / "output-before-ordinary-replacement"
    parent.rename(held)
    parent.mkdir()
    science_calls = []

    def forbidden_science(*_args, **_kwargs):
        science_calls.append(True)
        raise AssertionError("replacement directory inherited Preview authority")

    monkeypatch.setattr(rsm_operation, "run_rsm", forbidden_science)
    result = run_rsm_operation(request)

    assert result.terminal.disposition is ModuleDisposition.REFUSED
    assert result.terminal.code == "OUTPUT_IDENTITY_MISMATCH"
    assert science_calls == []
    assert tuple(parent.iterdir()) == ()
    assert tuple(held.iterdir()) == ()


def test_output_retarget_during_admission_is_refused_before_publication(
    tmp_path,
    monkeypatch,
):
    request, _frames = _prepared_rsm(tmp_path)
    parent = Path(request.module.output.target).parent
    held = tmp_path / "output-before-admission-retarget"
    outside = tmp_path / "outside"
    outside.mkdir()
    original_admit = rsm_operation.admit_module_artifact
    retargets = []

    def admit_then_retarget(*args, **kwargs):
        output = original_admit(*args, **kwargs)
        parent.rename(held)
        parent.symlink_to(outside, target_is_directory=True)
        retargets.append(True)
        return output

    monkeypatch.setattr(
        rsm_operation,
        "admit_module_artifact",
        admit_then_retarget,
    )
    result = run_rsm_operation(request)

    assert retargets == [True]
    assert result.terminal.disposition is ModuleDisposition.REFUSED
    assert result.terminal.code == "OUTPUT_IDENTITY_MISMATCH"
    assert tuple(outside.iterdir()) == ()
    assert tuple(held.iterdir()) == ()


def test_request_factory_rejects_foreign_preflight(tmp_path):
    request, _frames = _prepared_rsm(tmp_path)
    original = request.preflight
    changed_bounds = (
        (original.q_bounds[0][0], original.q_bounds[0][1] + 1.0e-6),
        original.q_bounds[1],
        original.q_bounds[2],
    )
    forged = rsm_operation.RSMPreflightReceipt(
        original.project_root,
        original.source_relative_path,
        original.source_scan,
        original.source_options_json,
        original.primary_revision,
        original.files,
        original.contributions,
        original.energy_eV,
        original.ub,
        changed_bounds,
        original.detector_shape,
        original.cropped_shape,
        original.source_fingerprint,
        original.module_source_fingerprint,
        original.table_fingerprint,
        original.plan_fingerprint,
        rsm_operation._REQUEST_FACTORY,
    )
    with pytest.raises(ValueError, match="exact bound source projection"):
        rsm_operation.RSMOperationRequest(
            request.module,
            request.plan,
            forged,
            request.output_authority,
            request.provenance_json,
            rsm_operation._REQUEST_FACTORY,
        )


def test_request_validation_preflight_remains_cancellable(tmp_path, monkeypatch):
    source, output, plan, _frames = _synthetic_spec_source(tmp_path)
    cancel = Event()
    original = rsm_operation._capture_preflight
    calls = []

    def cancel_second_capture(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            cancel.set()
        return original(*args, **kwargs)

    monkeypatch.setattr(
        rsm_operation,
        "_capture_preflight",
        cancel_second_capture,
    )
    with pytest.raises(RSMOperationRefused) as refused:
        prepare_rsm_operation(
            source,
            output,
            plan,
            project_root=tmp_path,
            cancel_token=cancel,
        )
    assert refused.value.code == "CANCELLED"
    assert calls == [True, True]


def test_static_hot_preflight_requires_at_least_two_frames(tmp_path):
    source, output, plan, _frames = _synthetic_spec_source(
        tmp_path,
        selected=(0,),
    )
    with pytest.raises(
        RSMOperationRefused,
        match="static-hot masking requires at least two frames",
    ) as refused:
        prepare_rsm_operation(source, output, plan, project_root=tmp_path)
    assert refused.value.code == "STATIC_MASK_REQUIRES_MULTIPLE_FRAMES"


def test_operation_commits_and_strictly_reloads_rsm(tmp_path):
    request, _frames = _prepared_rsm(tmp_path)
    progress = []
    result = run_rsm_operation(request, progress_callback=progress.append)

    assert result.terminal.disposition is ModuleDisposition.COMMITTED
    assert result.payload is not None
    assert result.payload.kind is AnalysisArtifactKind.RSM
    assert result.payload.inspection.shape == request.plan.bins
    assert tuple(name for name, _axis in result.payload.axes) == ("h", "k", "l")
    assert result.payload.sigma is None
    assert result.payload.provenance_json == request.provenance_json
    assert result.payload.result_fingerprint == result.terminal.commit.result_fingerprint
    assert result.payload.intensity.flags.writeable is False
    assert all(axis.flags.writeable is False for _name, axis in result.payload.axes)
    assert progress
    assert [item.revision for item in progress] == list(
        range(1, len(progress) + 1)
    )
    assert all(item.total == progress[0].total for item in progress)
    assert all(
        left.completed <= right.completed
        for left, right in zip(progress, progress[1:])
    )


def test_noncommitted_result_requires_no_payload(tmp_path):
    request, _frames = _prepared_rsm(tmp_path)
    terminal = ModuleTerminalResult(
        request.module,
        ModuleDisposition.REFUSED,
        "TEST_REFUSAL",
    )
    assert RSMOperationResult(request, terminal).payload is None
    with pytest.raises(TypeError, match="operation result is invalid"):
        RSMOperationResult(request, terminal, object())  # type: ignore[arg-type]


def test_execution_uses_one_lease_two_passes_and_numerator_normalization(
    tmp_path, monkeypatch
):
    request, frames = _prepared_rsm(tmp_path)
    loads = []
    leases = []
    captured = {}
    original_load = SpecSource.load_frame
    original_lease = rsm_operation.requalified_analysis_source

    def counting_load(self, label):
        loads.append(label)
        return original_load(self, label)

    @contextmanager
    def counting_lease(*args, **kwargs):
        leases.append(True)
        with original_lease(*args, **kwargs) as source:
            yield source

    def capture_run(plan, source, *, scan_labels=None):
        chunks = list(source.iter_chunks(plan.chunk_size))
        captured["plan"] = plan
        captured["labels"] = [labels for _images, labels in chunks]
        captured["images"] = np.concatenate(
            [images for images, _labels in chunks], axis=0
        )
        captured["scan_labels"] = scan_labels
        return SimpleNamespace(payload=_synthetic_volume(request))

    monkeypatch.setattr(SpecSource, "load_frame", counting_load)
    monkeypatch.setattr(rsm_operation, "requalified_analysis_source", counting_lease)
    monkeypatch.setattr(rsm_operation, "run_rsm", capture_run)
    result = run_rsm_operation(request)

    assert result.terminal.disposition is ModuleDisposition.COMMITTED
    assert leases == [True]
    assert loads == [0, 2, 0, 2]
    assert captured["labels"] == [[0], [2]]
    assert captured["scan_labels"] == [request.preflight.source_scan]
    runtime = captured["plan"]
    assert runtime.q_bounds == request.preflight.q_bounds
    assert runtime.energy == request.preflight.energy_eV
    np.testing.assert_allclose(runtime.UB, request.preflight.ub)
    assert runtime.corrections is None
    assert runtime.gi is None
    assert runtime.static_mask.shape == request.preflight.cropped_shape
    assert runtime.static_mask[0, 0]
    divisors = np.asarray(
        [item.normalization_divisor for item in request.preflight.contributions]
    )
    expected = np.stack([frames[0], frames[2]]).astype(float) + 1.0e-6
    expected /= divisors[:, None, None]
    np.testing.assert_allclose(captured["images"], expected, rtol=0, atol=0)


def test_pre_and_mid_static_pass_cancellation_write_nothing(tmp_path, monkeypatch):
    request, _frames = _prepared_rsm(tmp_path)
    cancel = Event()
    cancel.set()
    result = run_rsm_operation(request, cancel_token=cancel)
    assert result.terminal.disposition is ModuleDisposition.CANCELLED
    assert not Path(request.module.output.target).exists()

    second_root = tmp_path / "mid"
    second_root.mkdir()
    request, _frames = _prepared_rsm(second_root)
    original_load = SpecSource.load_frame
    loads = []
    cancel = Event()

    def cancelling_load(self, label):
        value = original_load(self, label)
        loads.append(label)
        if len(loads) == 1:
            cancel.set()
        return value

    monkeypatch.setattr(SpecSource, "load_frame", cancelling_load)
    result = run_rsm_operation(request, cancel_token=cancel)
    assert result.terminal.disposition is ModuleDisposition.CANCELLED
    assert loads == [0]
    assert not Path(request.module.output.target).exists()


def test_mid_science_cancellation_stops_before_output(tmp_path, monkeypatch):
    request, _frames = _prepared_rsm(tmp_path)
    original_load = SpecSource.load_frame
    loads = []
    cancel = Event()

    def cancelling_load(self, label):
        value = original_load(self, label)
        loads.append(label)
        if len(loads) == 3:
            cancel.set()
        return value

    monkeypatch.setattr(SpecSource, "load_frame", cancelling_load)
    result = run_rsm_operation(request, cancel_token=cancel)
    assert result.terminal.disposition is ModuleDisposition.CANCELLED
    assert loads == [0, 2, 0]
    assert not Path(request.module.output.target).exists()


def test_float64_frame_bound_and_invalid_science_stop_before_output(
    tmp_path, monkeypatch
):
    source, output, plan, _frames = _synthetic_spec_source(
        tmp_path,
        max_frame_bytes=100,
    )
    with pytest.raises(RSMOperationRefused) as refused:
        prepare_rsm_operation(source, output, plan, project_root=tmp_path)
    assert refused.value.code == "FRAME_MEMORY_LIMIT_EXCEEDED"
    assert not Path(output.target).exists()

    second_root = tmp_path / "invalid"
    second_root.mkdir()
    request, _frames = _prepared_rsm(second_root)
    monkeypatch.setattr(
        rsm_operation,
        "run_rsm",
        lambda *_args, **_kwargs: SimpleNamespace(payload=object()),
    )
    result = run_rsm_operation(request)
    assert result.terminal.disposition is ModuleDisposition.FAILED
    assert result.terminal.code == "INVALID_SCIENCE_RESULT"
    assert not Path(request.module.output.target).exists()


def test_aggregate_chunk_bound_stops_before_output(tmp_path):
    request, _frames = _prepared_rsm(tmp_path, max_chunk_bytes=800)
    result = run_rsm_operation(request)
    assert result.terminal.disposition is ModuleDisposition.FAILED
    assert result.terminal.code == "SCIENCE_FAILED"
    assert "max_chunk_bytes=800" in result.terminal.diagnostic
    assert not Path(request.module.output.target).exists()


def test_science_axes_must_match_the_exact_fixed_grid(tmp_path, monkeypatch):
    request, _frames = _prepared_rsm(tmp_path)
    volume = _synthetic_volume(request)
    shifted = np.array(volume.h, copy=True)
    shifted[1] += (shifted[2] - shifted[1]) * 0.1
    wrong = rsm_operation.RSMVolume(
        shifted,
        volume.k,
        volume.l,
        volume.intensity,
    )
    monkeypatch.setattr(
        rsm_operation,
        "run_rsm",
        lambda *_args, **_kwargs: SimpleNamespace(payload=wrong),
    )
    result = run_rsm_operation(request)
    assert result.terminal.disposition is ModuleDisposition.FAILED
    assert result.terminal.code == "INVALID_SCIENCE_RESULT"
    assert not Path(request.module.output.target).exists()


def test_normalization_overflow_is_refused_before_output(tmp_path):
    source, output, plan, _frames = _synthetic_spec_source(tmp_path)
    extreme = RSMNormalizationPolicy(
        RSMNormalizationMode.FOIL_TRANSMISSION_EXPOSURE,
        plan.normalization.foil_selector,
        plan.normalization.exposure_selector,
        (356.0, 356.0, 356.0, 356.0),
    )
    extreme_plan = RSMOperationPlan(
        plan.geometry,
        plan.conditioning,
        extreme,
        bins=plan.bins,
        chunk_size=plan.chunk_size,
        max_frame_bytes=plan.max_frame_bytes,
        max_chunk_bytes=plan.max_chunk_bytes,
    )
    request = prepare_rsm_operation(
        source,
        output,
        extreme_plan,
        project_root=tmp_path,
    )
    assert all(
        item.normalization_divisor > 0
        for item in request.preflight.contributions
    )
    result = run_rsm_operation(request)
    assert result.terminal.disposition is ModuleDisposition.REFUSED
    assert result.terminal.code == "NORMALIZATION_RESULT_INVALID"
    assert not Path(request.module.output.target).exists()


def test_transient_reload_retry_never_replays_science_or_writer(
    tmp_path, monkeypatch
):
    request, _frames = _prepared_rsm(tmp_path)
    counts = {"science": 0, "writer": 0, "read": 0}
    original_run = rsm_operation.run_rsm
    original_write = rsm_operation.write_rsm
    original_read = rsm_operation.read_analysis_artifact

    def counted_run(*args, **kwargs):
        counts["science"] += 1
        return original_run(*args, **kwargs)

    def counted_write(*args, **kwargs):
        counts["writer"] += 1
        return original_write(*args, **kwargs)

    def transient_read(*args, **kwargs):
        counts["read"] += 1
        if counts["read"] == 1:
            raise OSError("transient detached reload fault")
        return original_read(*args, **kwargs)

    monkeypatch.setattr(rsm_operation, "run_rsm", counted_run)
    monkeypatch.setattr(rsm_operation, "write_rsm", counted_write)
    monkeypatch.setattr(rsm_operation, "read_analysis_artifact", transient_read)
    execution = RSMOperationExecution(request)
    with pytest.raises(RSMOperationVerificationError) as failed:
        execution.run()
    assert failed.value.execution is execution
    recovered = execution.retry_verification()
    assert recovered.terminal.disposition is ModuleDisposition.COMMITTED
    assert execution.retry_verification() is recovered
    assert counts == {"science": 1, "writer": 1, "read": 2}


def test_strict_reload_rejects_writer_corrupted_fixed_grid_without_replay(
    tmp_path, monkeypatch
):
    request, _frames = _prepared_rsm(tmp_path)
    original_write = rsm_operation.write_rsm
    original_run = rsm_operation.run_rsm
    counts = {"science": 0, "writer": 0}

    def counted_run(*args, **kwargs):
        counts["science"] += 1
        return original_run(*args, **kwargs)

    def corrupt_axis(entry, volume, **kwargs):
        counts["writer"] += 1
        shifted = np.array(volume.h, copy=True)
        shifted[1] += (shifted[2] - shifted[1]) * 0.1
        corrupt = rsm_operation.RSMVolume(
            shifted,
            volume.k,
            volume.l,
            volume.intensity,
        )
        return original_write(entry, corrupt, **kwargs)

    monkeypatch.setattr(rsm_operation, "run_rsm", counted_run)
    monkeypatch.setattr(rsm_operation, "write_rsm", corrupt_axis)
    execution = RSMOperationExecution(request)
    with pytest.raises(
        RSMOperationVerificationError,
        match="payload no longer matches",
    ) as failed:
        execution.run()
    assert failed.value.execution is execution
    assert Path(request.module.output.target).is_file()
    with pytest.raises(
        RSMOperationVerificationError,
        match="payload no longer matches",
    ):
        execution.retry_verification()
    assert counts == {"science": 1, "writer": 1}


def test_cleanup_pending_retries_without_replaying_science_or_writer(
    tmp_path, monkeypatch
):
    import xrd_tools.io.output_transaction as transaction_api

    initial, _frames = _prepared_rsm(tmp_path)
    target = Path(initial.module.output.target)
    target.write_bytes(b"prior operator output")
    replacement = ModuleOutputRequest(
        target,
        AnalysisArtifactKind.RSM,
        AnalysisArtifactOverwrite.REPLACE,
    )
    request = prepare_rsm_operation(
        initial.module.source,
        replacement,
        initial.plan,
        project_root=tmp_path,
    )
    original_unlink = transaction_api._unlink
    original_write = rsm_operation.write_rsm
    original_run = rsm_operation.run_rsm
    failures = []
    counts = {"science": 0, "writer": 0}

    def fail_backup_once(path):
        if ".xdart-replacing-" in Path(path).name and not failures:
            failures.append("backup")
            raise OSError("backup cleanup fault")
        return original_unlink(path)

    def counted_write(*args, **kwargs):
        counts["writer"] += 1
        return original_write(*args, **kwargs)

    def counted_run(*args, **kwargs):
        counts["science"] += 1
        return original_run(*args, **kwargs)

    monkeypatch.setattr(transaction_api, "_unlink", fail_backup_once)
    monkeypatch.setattr(rsm_operation, "write_rsm", counted_write)
    monkeypatch.setattr(rsm_operation, "run_rsm", counted_run)
    with pytest.raises(RSMOperationCleanupPending) as pending:
        run_rsm_operation(request)
    recovered = pending.value.retry_cleanup()
    assert recovered.terminal.disposition is ModuleDisposition.COMMITTED
    assert recovered.payload is not None
    assert pending.value.execution.retry_cleanup() is recovered
    assert failures == ["backup"]
    assert counts == {"science": 1, "writer": 1}


def test_existing_create_new_output_is_late_refusal_without_overwrite(tmp_path):
    request, _frames = _prepared_rsm(tmp_path)
    target = Path(request.module.output.target)
    original = b"operator-owned"
    target.write_bytes(original)
    result = run_rsm_operation(request)
    assert result.terminal.disposition is ModuleDisposition.REFUSED
    assert result.terminal.code == "OUTPUT_EXISTS"
    assert target.read_bytes() == original


def test_callback_failure_is_inert_and_execution_is_one_shot(tmp_path):
    request, _frames = _prepared_rsm(tmp_path)
    execution = RSMOperationExecution(request)
    callbacks = []

    def failing_callback(progress):
        callbacks.append(progress)
        raise RuntimeError("observer fault")

    result = execution.run(progress_callback=failing_callback)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED
    assert callbacks
    with pytest.raises(RuntimeError, match="one-shot"):
        execution.run()
    assert execution.retry_cleanup() is result
    assert execution.retry_verification() is result


def test_source_drift_before_science_and_at_prepublish_is_refused(
    tmp_path, monkeypatch
):
    request, _frames = _prepared_rsm(tmp_path)
    contribution = request.preflight.contributions[0]
    relative = request.preflight.files[contribution.file_ordinal].relative_path
    raw = Path(request.preflight.project_root) / relative
    raw.write_bytes(raw.read_bytes() + b"drift")
    result = run_rsm_operation(request)
    assert result.terminal.disposition is ModuleDisposition.REFUSED
    assert result.terminal.code == "SOURCE_IDENTITY_MISMATCH"
    assert not Path(request.module.output.target).exists()

    second_root = tmp_path / "prepublish"
    second_root.mkdir()
    request, _frames = _prepared_rsm(second_root)
    contribution = request.preflight.contributions[0]
    relative = request.preflight.files[contribution.file_ordinal].relative_path
    raw = Path(request.preflight.project_root) / relative
    original_write = rsm_operation.write_rsm

    def drifting_write(*args, **kwargs):
        original_write(*args, **kwargs)
        raw.write_bytes(raw.read_bytes() + b"drift")

    monkeypatch.setattr(rsm_operation, "write_rsm", drifting_write)
    result = run_rsm_operation(request)
    assert result.terminal.disposition is ModuleDisposition.REFUSED
    assert result.terminal.code == "SOURCE_REVISION_CHANGED"
    assert not Path(request.module.output.target).exists()


def test_source_drift_at_science_lease_terminal_fence_writes_nothing(
    tmp_path, monkeypatch
):
    request, _frames = _prepared_rsm(tmp_path)
    contribution = request.preflight.contributions[0]
    relative = request.preflight.files[contribution.file_ordinal].relative_path
    raw = Path(request.preflight.project_root) / relative

    def drifting_science(*_args, **_kwargs):
        raw.write_bytes(raw.read_bytes() + b"drift")
        return SimpleNamespace(payload=_synthetic_volume(request))

    monkeypatch.setattr(rsm_operation, "run_rsm", drifting_science)
    result = run_rsm_operation(request)
    assert result.terminal.disposition is ModuleDisposition.REFUSED
    assert result.terminal.code == "SOURCE_REVISION_CHANGED"
    assert not Path(request.module.output.target).exists()
