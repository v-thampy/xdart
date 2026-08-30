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
from xrd_tools.analysis.scan_operations import (
    AnalysisDisposition,
    MetadataTablePlan,
    run_metadata_table,
)
from xrd_tools.analysis.stitch_operation import (
    StitchGeometryInput,
    StitchGeometryKind,
    StitchOperationPlan,
    capture_stitch_geometry,
    prepare_stitch_operation,
    run_stitch_operation,
)
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.io.analysis_artifact import (
    AnalysisArtifactKind,
    AnalysisArtifactOverwrite,
)


_GONIOMETER_SHA256 = "c74676659d821b6460f37e8b42af02767d127263a0079da67fe6e0d291edbdea"
_SPEC_SHA256 = "dbec5be7cb8bbde76b942ce477e10830ddb6bac1b6fc1965fe67df63bb0e3db8"


def _stitching_root() -> Path:
    configured = os.environ.get("XDART_TEST_DATA")
    if not configured:
        pytest.skip("XDART_TEST_DATA is not configured")
    root = Path(configured) / "stitching"
    if not root.is_dir():
        pytest.skip("authenticated Stitching test data is unavailable")
    return root


def _scan14_request(
    root: Path,
    output_path: Path,
    labels: tuple[int, ...],
):
    data = root / "data"
    calibration = data / "calibration"
    spec_path = calibration / "LaB6_17keV"
    assert hashlib.sha256(spec_path.read_bytes()).hexdigest() == _SPEC_SHA256
    source_spec = SourceSpec(
        spec_path,
        SourceKind.SPEC,
        options={
            "scan": "14.1",
            "image_dir": str(data / "images"),
            "image_stem": "b_thampy_LaB6_17keV_scan14_",
            "read_image_kwargs": {
                "detector_shape": (195, 1475),
                "raw_dtype": "int32",
                "threshold": 8.0e5,
                "rotation": 0,
            },
        },
    )
    table = run_metadata_table(MetadataTablePlan(source_spec))
    assert table.disposition is AnalysisDisposition.COMPLETED
    assert table.labels == tuple(range(287))
    source = ModuleSourceReceipt.from_metadata_table(
        table,
        kind=ModuleKind.STITCH,
        selected_labels=labels,
        resolved_selectors=(
            MetadataColumnSelector("del"),
            MetadataColumnSelector("nu"),
        ),
    )
    geometry = capture_stitch_geometry(
        StitchGeometryInput(
            calibration / "MG_gonio_object_5images.json",
            StitchGeometryKind.PYFAI_GONIOMETER_JSON,
            expected_sha256=_GONIOMETER_SHA256,
            source_motors=(("del_angle", "del"), ("nu_angle", "nu")),
            image_rotation=0,
        )
    )
    plan = StitchOperationPlan(
        geometry,
        mode="1d",
        npt_1d=1500,
        radial_range=(1.0, 6.2),
        use_detector_mask=True,
        max_frame_bytes=64 * 1024 * 1024,
    )
    output = ModuleOutputRequest(
        output_path,
        AnalysisArtifactKind.STITCH_1D,
        AnalysisArtifactOverwrite.CREATE_NEW,
    )
    request = prepare_stitch_operation(
        source,
        output,
        plan,
        project_root=root,
    )
    return request


@pytest.mark.slow
def test_scan14_quick_stitch_commits_with_calibrated_geometry(tmp_path):
    root = _stitching_root()
    labels = tuple(range(0, 287, 19))
    request = _scan14_request(root, tmp_path / "scan14_quick.nexus", labels)
    assert request.manifest.source_relative_path == "data/calibration/LaB6_17keV"
    assert request.manifest.source_scan == "14.1"
    assert json.loads(request.manifest.source_options_json)["image_dir"] == (
        "data/images"
    )
    assert len(request.manifest.contributions) == len(labels)
    assert [item.label for item in request.manifest.contributions] == list(labels)
    assert len(request.manifest.files) == 288
    assert request.manifest.files[0].relative_path == (
        "data/calibration/LaB6_17keV"
    )
    assert [
        request.manifest.files[item.file_ordinal].relative_path
        for item in request.manifest.contributions
    ] == [
        f"data/images/b_thampy_LaB6_17keV_scan14_{label:04d}.raw"
        for label in labels
    ]
    assert all(item.source_frame_index == 0 for item in request.manifest.contributions)
    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED
    axis = result.payload.axis("q")
    intensity = result.payload.intensity
    finite = np.isfinite(intensity)
    assert finite.sum() > 0.75 * intensity.size
    assert np.nanmax(intensity) > np.nanmedian(intensity) * 5.0
    expected_peaks = (1.51, 2.14, 2.62, 3.02)
    ratios = []
    for center in expected_peaks:
        peak = (axis >= center - 0.08) & (axis <= center + 0.08)
        neighborhood = (axis >= center - 0.20) & (axis <= center + 0.20)
        ratios.append(
            float(np.nanmax(intensity[peak]) / np.nanmedian(intensity[neighborhood]))
        )
    assert sum(ratio > 1.2 for ratio in ratios) >= 3, ratios
    assert result.payload.inspection.result_fingerprint == (
        result.terminal.commit.result_fingerprint
    )


@pytest.mark.slow
def test_scan14_full_287_streams_and_persists_diagnostics(tmp_path):
    root = _stitching_root()
    labels = tuple(range(287))
    request = _scan14_request(root, tmp_path / "scan14_full.nexus", labels)
    assert len(request.manifest.contributions) == 287
    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED
    assert result.payload.intensity.shape == (1500,)
    assert result.payload.coverage.shape == (1500,)
    assert result.payload.normalization.shape == (1500,)
    assert np.isfinite(result.payload.intensity).sum() > 0.75 * 1500
    assert np.any(result.payload.coverage > 0)
    assert np.any(result.payload.normalization > 0)
