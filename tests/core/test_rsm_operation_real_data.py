from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from xrd_tools.analysis.module_transaction import (
    MetadataColumnSelector,
    ModuleDisposition,
    ModuleKind,
    ModuleOutputRequest,
    ModuleSourceReceipt,
)
from xrd_tools.analysis.rsm_geometry_asset import (
    install_canonical_rsm_geometry_asset,
    lower_rsm_effective_geometry,
    rsm_effective_pixel_q_map,
)
from xrd_tools.analysis.rsm_operation import (
    RSMDetectorGeometry,
    RSMImageConditioning,
    RSMNormalizationMode,
    RSMNormalizationPolicy,
    RSMOperationPlan,
    prepare_rsm_operation,
    required_rsm_selectors,
    resolve_exact_rsm_q_bounds,
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
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.io.analysis_artifact import (
    AnalysisArtifactKind,
    AnalysisArtifactOverwrite,
)


_SPEC_SHA256 = "12e0d05219cb0309f7fb5f170c8fb12c507e7603dcd71b3954721d1b6e0c90b1"
_Q_BOUNDS = (
    (0.9991454998430791, 1.0507535379970891),
    (0.8498770429375965, 1.0673826805071531),
    (2.8230899655686454, 3.18407314269762),
)
_ORACLES = {
    (40, 40, 40): (
        5531,
        "1f5c827c333a9b919db9491c5155353e0b4a0adf13ccb619652c36443febf9e6",
    ),
    (200, 200, 200): (
        502759,
        "33dec932699ebe41e92deec57c658259b9f369778730a18374859afa221749ad",
    ),
}


def _rsm_root() -> Path:
    configured = os.environ.get("XDART_TEST_DATA")
    if not configured:
        pytest.skip("XDART_TEST_DATA is not configured")
    root = Path(configured) / "RSM"
    if not root.is_dir():
        pytest.skip("authenticated RSM test data is unavailable")
    return root


def _scan43_request(
    root: Path,
    output_path: Path,
    bins: tuple[int, int, int],
):
    spec_path = root / "STO_align"
    assert hashlib.sha256(spec_path.read_bytes()).hexdigest() == _SPEC_SHA256
    geometry = RSMDetectorGeometry(
        DetectorHeader(
            cch1=97.0,
            cch2=243.0,
            pwidth1=0.172,
            pwidth2=0.172,
            distance=1014.7173,
            Nch1=195,
            Nch2=487,
        ),
        tuple(
            (role, MetadataColumnSelector(role, 0))
            for role in ("mu", "eta", "chi", "phi", "nu", "del")
        ),
        ImageOrientation(),
        (0, -1, 0, -1),
    )
    plan = RSMOperationPlan(
        geometry,
        RSMImageConditioning(1.0e-6, 2.0e10, 100.0),
        RSMNormalizationPolicy(
            RSMNormalizationMode.FOIL_TRANSMISSION_EXPOSURE,
            MetadataColumnSelector("foil status", 0),
            MetadataColumnSelector("Seconds", 0),
            (1.06, 3.04, 4.65, 9.5),
        ),
        bins=bins,
        chunk_size=8,
        max_frame_bytes=64 * 1024 * 1024,
        max_chunk_bytes=256 * 1024 * 1024,
    )
    selectors = required_rsm_selectors(plan)
    reader = {
        "detector_shape": (195, 487),
        "raw_dtype": "int32",
        "raw_header_skip": 0,
        "threshold": None,
        "rotation": 0,
    }
    source_spec = SourceSpec(
        spec_path,
        SourceKind.SPEC,
        options={
            "scan": "43.1",
            "image_dir": str(root / "images"),
            "image_stem": "b_thampy_STO_align_scan43_",
            "read_image_kwargs": reader,
            "metadata_column_projection": tuple(
                (selector.name, selector.occurrence) for selector in selectors
            ),
        },
    )
    table = run_metadata_table(MetadataTablePlan(source_spec))
    assert table.disposition is AnalysisDisposition.COMPLETED
    assert table.labels == tuple(range(61))
    source = ModuleSourceReceipt.from_metadata_table(
        table,
        kind=ModuleKind.RSM,
        selected_labels=table.labels,
        resolved_selectors=selectors,
    )
    output = ModuleOutputRequest(
        output_path,
        AnalysisArtifactKind.RSM,
        AnalysisArtifactOverwrite.CREATE_NEW,
    )
    request = prepare_rsm_operation(
        source,
        output,
        plan,
        project_root=root,
    )
    options = json.loads(request.preflight.source_options_json)
    assert options["image_dir"] == "images"
    assert options["read_image_kwargs"] == {
        "detector_shape": [195, 487],
        "raw_dtype": "int32",
        "raw_header_skip": 0,
        "rotation": 0,
        "threshold": None,
    }
    assert request.preflight.q_bounds == _Q_BOUNDS
    assert len(request.preflight.contributions) == 61
    assert all(
        request.preflight.files[item.file_ordinal].relative_path
        == f"images/b_thampy_STO_align_scan43_{item.label:04d}.raw"
        for item in request.preflight.contributions
    )
    return request


def _float32_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype=np.float32)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


@pytest.mark.slow
def test_scan43_effective_asset_preserves_every_exact_q_value_image_free(
    tmp_path, monkeypatch
):
    from xrayutilities import config

    from xrd_tools.sources.spec import SpecSource

    root = _rsm_root()
    output = root / f".xdart-rsm-a2-q-only-{os.getpid()}-{tmp_path.name}.nexus"
    assert not output.exists()

    def forbidden_decode(*_args, **_kwargs):
        raise AssertionError("RSM A2 q equivalence must not decode detector data")

    monkeypatch.setattr(SpecSource, "load_frame", forbidden_decode)
    request = _scan43_request(root, output, (40, 40, 40))
    asset_project = tmp_path / "asset-project"
    asset_project.mkdir()
    effective = lower_rsm_effective_geometry(
        install_canonical_rsm_geometry_asset(project_root=asset_project)
    )
    effective_mapper = rsm_effective_pixel_q_map(effective)
    legacy_mapper = PixelQMap(
        Diffractometer.psic(), request.plan.geometry.header
    )
    contribution_values = tuple(
        {
            (name, occurrence): value
            for name, occurrence, value in contribution.values
        }
        for contribution in request.preflight.contributions
    )
    angles = tuple(
        np.asarray(
            [
                values[(selector.name, selector.occurrence)]
                for values in contribution_values
            ],
            dtype=np.float64,
        )
        for _role, selector in request.plan.geometry.motor_selectors
    )
    ub = np.asarray(request.preflight.ub, dtype=np.float64)
    before = config.NTHREADS
    for start in range(0, len(angles[0]), request.plan.chunk_size):
        stop = min(start + request.plan.chunk_size, len(angles[0]))
        chunk = tuple(values[start:stop] for values in angles)
        effective_q = effective_mapper.pixel_q(
            chunk,
            request.preflight.energy_eV,
            UB=ub,
            roi=effective.roi,
        )
        legacy_q = legacy_mapper.pixel_q(
            chunk,
            request.preflight.energy_eV,
            UB=ub,
            roi=request.plan.geometry.roi,
        )
        assert all(
            np.array_equal(effective_axis, legacy_axis)
            for effective_axis, legacy_axis in zip(
                effective_q, legacy_q, strict=True
            )
        )
    effective_bounds = resolve_exact_rsm_q_bounds(
        effective_mapper,
        angles,
        request.preflight.energy_eV,
        ub,
        roi=effective.roi,
        chunk_size=request.plan.chunk_size,
        max_frame_bytes=request.plan.max_frame_bytes,
        max_chunk_bytes=request.plan.max_chunk_bytes,
    )
    assert effective_bounds == request.preflight.q_bounds == _Q_BOUNDS
    assert config.NTHREADS == before
    assert not output.exists()


@pytest.mark.slow
@pytest.mark.parametrize("bins", tuple(_ORACLES))
def test_scan43_matches_authenticated_notebook_rsm(tmp_path, bins):
    root = _rsm_root()
    output = root / (
        f".xdart-rsm-science-{os.getpid()}-{tmp_path.name}-{bins[0]}.nexus"
    )
    assert not output.exists()
    try:
        request = _scan43_request(root, output, bins)
        result = run_rsm_operation(request)
        assert result.terminal.disposition is ModuleDisposition.COMMITTED
        assert result.payload is not None
        expected_occupied, expected_sha = _ORACLES[bins]
        assert int(np.isfinite(result.payload.intensity).sum()) == expected_occupied
        assert _float32_sha256(result.payload.intensity) == expected_sha
        assert result.payload.inspection.result_fingerprint == (
            result.terminal.commit.result_fingerprint
        )
    finally:
        if output.exists():
            output.unlink()
