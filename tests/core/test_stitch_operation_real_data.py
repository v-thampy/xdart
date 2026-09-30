from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import replace
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
    XuStitchOperationPlan,
    capture_stitch_geometry,
    prepare_stitch_operation,
    run_stitch_operation,
)
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.io.analysis_artifact import (
    AnalysisArtifactKind,
    AnalysisArtifactOverwrite,
    read_analysis_artifact,
)
from xrd_tools.analysis.xu_stitch_calibration import install_canonical_xu_stitch_calibration
from xrd_tools.core.containers import convert_radial_axis
from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
from xrd_tools.integrate.stitch_conditioning import StitchConditioning
from xrd_tools.integrate.xu_stitch import resolve_xu_stitch_effective_geometry
from xrd_tools.io.frame_view import read_contributing_frame_view
from xrd_tools.io.image import read_image
from xrd_tools.io.output_path import resolve_finite_output_target


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


@pytest.fixture(scope="module")
def isolated_scan14_project(tmp_path_factory):
    """Keep canonical calibration installation and output away from the fixture.

    SPEC image discovery binds row numbers to the complete sorted image list,
    so retain all 287 files even though the quick test reduces only 16 frames.
    Copies also avoid changing source inode identities or escaping Project via
    symlinks while another operation may hold the original fixture receipt.
    """
    original = _stitching_root()
    project = tmp_path_factory.mktemp("native-stitch-scan14-project")
    calibration = project / "data/calibration"
    images = project / "data/images"
    calibration.mkdir(parents=True)
    images.mkdir()
    for name, digest in (
        ("LaB6_17keV", _SPEC_SHA256),
        ("MG_gonio_object_5images.json", _GONIOMETER_SHA256),
    ):
        source = original / "data/calibration" / name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
        shutil.copy2(source, calibration / name)
    files = sorted((original / "data/images").glob("b_thampy_LaB6_17keV_scan14_*.raw"))
    assert [path.name for path in files] == [
        f"b_thampy_LaB6_17keV_scan14_{index:04d}.raw" for index in range(287)
    ]
    assert sum(path.stat().st_size for path in files) == 330_193_500
    for source in files:
        shutil.copy2(source, images / source.name)
    return project


@pytest.mark.slow
@pytest.mark.parametrize("backend", ("multigeometry", "xu_hist"))
def test_scan14_native_records_preserve_legacy_science_and_raw_links(
    isolated_scan14_project, backend,
):
    project = isolated_scan14_project
    labels = tuple(range(0, 287, 19))
    assert len(labels) == 16
    legacy = _scan14_request(project, project / f"{backend}-legacy.nexus", labels)
    if backend == "xu_hist":
        calibration = install_canonical_xu_stitch_calibration(project_root=project)
        with xu_runtime_session() as runtime:
            effective = resolve_xu_stitch_effective_geometry(calibration, runtime).projection
        plan = XuStitchOperationPlan(
            calibration, effective, 1.0, 5.2, npt_1d=760,
            max_frame_bytes=64 * 1024 * 1024,
        )
        table = run_metadata_table(MetadataTablePlan(legacy.module.source.analysis.source_spec))
        assert table.disposition is AnalysisDisposition.COMPLETED
        selectors = tuple(MetadataColumnSelector(name) for name in ("del", "nu", "energy")
                          if name in {column.name for column in table.columns})
        source = ModuleSourceReceipt.from_metadata_table(
            table, kind=ModuleKind.STITCH, selected_labels=labels,
            resolved_selectors=selectors,
        )
        legacy = prepare_stitch_operation(source, legacy.module.output, plan, project_root=project)

    # The notebook reader masks values strictly ABOVE 800000. The shared Int
    # manual policy uses the same inclusive upper limit on native int32 data;
    # an absent lower limit preserves the legacy treatment of negative values.
    native_plan = replace(
        legacy.plan, native_records=True,
        conditioning=StitchConditioning("manual", threshold_max=800_000.0),
    )
    family = f"scan14-{backend}-native"
    native_output = ModuleOutputRequest(
        resolve_finite_output_target(project, family, operation_token="stitch-1d"),
        AnalysisArtifactKind.STITCH_1D, artifact_family=family,
    )
    native = prepare_stitch_operation(
        legacy.module.source, native_output, native_plan, project_root=project,
    )
    assert [item.label for item in native.manifest.contributions] == list(labels)
    binding = native.provenance["frame_members"]
    assert (binding["version"], binding["count"]) == (1, 16)
    for request in (legacy, native):
        result = run_stitch_operation(request)
        assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal

    reference = read_analysis_artifact(legacy.module.output.target)
    payload = read_analysis_artifact(native_output.target)
    assert reference.schema_version == (5 if backend == "xu_hist" else 4)
    assert payload.schema_version == (7 if backend == "xu_hist" else 6)
    assert payload.native_unit == "q_A^-1"
    np.testing.assert_array_equal(payload.axis("q"), reference.axis("q"))
    np.testing.assert_array_equal(payload.intensity, reference.intensity)
    np.testing.assert_array_equal(payload.coverage, reference.coverage)
    np.testing.assert_array_equal(payload.normalization, reference.normalization)
    expected_two_theta = convert_radial_axis(
        reference.axis("q"), "q_A^-1", "2th_deg", payload.wavelength_A,
    ).astype("<f4")
    np.testing.assert_array_equal(payload.radial_axis("2th_deg"), expected_two_theta)
    assert not np.allclose(np.diff(expected_two_theta), np.diff(expected_two_theta)[0], rtol=1e-5)
    assert (payload.execution_attestation_digest is not None) == (backend == "xu_hist")

    rows = payload.contributing_frames
    assert [row["identity"]["frame_index"] for row in rows] == list(labels)
    assert [row["identity"]["contribution_index"] for row in rows] == list(range(16))
    assert len({row["identity"]["source_id"] for row in rows}) == 1
    for row, label in zip(rows, labels, strict=True):
        assert row["identity"]["scan_id"] == "14.1"
        assert row["source_path"] == f"data/images/b_thampy_LaB6_17keV_scan14_{label:04d}.raw"
        assert row["source_frame_index"] == 0
        snapshot = row["source_snapshot"]
        assert np.dtype(snapshot["raw_dtype"]) == np.dtype("int32")
        assert snapshot["raw_shape"] == [195, 1475]
        assert snapshot["raw_header_bytes"] == snapshot["rotation"] == 0
        assert row["mask_baked"] is False
        assert row["thumbnail"]["mask_baked"] is False
    for row in (rows[0], rows[-1]):
        frame = read_contributing_frame_view(
            native_output.target, row["identity"]["frame_index"],
            scan=row["key"].split("/")[0][5:], full_raw=True, artifact_payload=payload,
        )
        expected_raw = read_image(
            project / row["source_path"], detector_shape=(195, 1475),
            raw_dtype="int32", preserve_dtype=True, exact_frame=True,
        )
        np.testing.assert_array_equal(frame.raw, expected_raw)
        assert frame.thumbnail is not None


@pytest.mark.slow
def test_scan14_native_result_records_the_frame_check(isolated_scan14_project):
    # Contiguous runs around every flagged frame keep each frame's same-row
    # neighbours, so the verdicts equal the full 287-frame screen's.
    project = isolated_scan14_project
    labels = tuple(label for low, high in ((47, 58), (94, 126), (159, 170), (267, 278))
                   for label in range(low, high))
    legacy = _scan14_request(project, project / "screen-legacy.nexus", labels)
    plan = replace(legacy.plan, native_records=True,
                   conditioning=StitchConditioning("manual", threshold_max=800_000.0))
    output = ModuleOutputRequest(
        resolve_finite_output_target(project, "scan14-screen", operation_token="stitch-1d"),
        AnalysisArtifactKind.STITCH_1D, artifact_family="scan14-screen",
    )
    result = run_stitch_operation(
        prepare_stitch_operation(legacy.module.source, output, plan, project_root=project))
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
    rows = read_analysis_artifact(output.target).contributing_frames
    verdicts = {row["identity"]["frame_index"]: (row["screen"], row["excluded"]) for row in rows}
    assert sorted(verdicts) == sorted(labels)
    assert sorted(label for label, (_screen, excluded) in verdicts.items() if excluded) == [52, 164]
    assert sorted(label for label, (screen, excluded) in verdicts.items()
                  if screen == "partial" and not excluded) == [99, 106, 120, 272]
    assert result.frame_screen["excluded"] == [52, 164]


@pytest.mark.slow
def test_scan14_native_stitch_2d_merges_past_pyfais_unsplit_bins(isolated_scan14_project):
    """All 287 frames in 2-D (STITCH-2D-20260930): pyFAI's pixel splitting leaves 12 frames a NaN last q
    bin and many a bin of negative round-off. The map commits with every bin finite and nonnegative,
    the last q bin holding the other frames' data, with I(q) beside it."""
    project = isolated_scan14_project
    legacy = _scan14_request(project, project / "map-legacy.nexus", tuple(range(287)))
    plan = replace(legacy.plan, mode="2d", npt_rad_2d=1000, npt_azim_2d=360,
                   azimuth_range=(-180.0, 180.0), native_records=True,
                   # the map over the plan's q range (its own is Auto otherwise): the last bin at its edge
                   map_unit=legacy.plan.unit, map_radial_range=legacy.plan.radial_range)
    output = ModuleOutputRequest(
        resolve_finite_output_target(project, "scan14-map", operation_token="stitch-2d"),
        AnalysisArtifactKind.STITCH_2D, artifact_family="scan14-map",
    )
    result = run_stitch_operation(
        prepare_stitch_operation(legacy.module.source, output, plan, project_root=project))
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
    payload = read_analysis_artifact(output.target, expected_kind=AnalysisArtifactKind.STITCH_2D)
    assert payload.intensity.shape == payload.coverage.shape == (1000, 360)
    for values in (payload.coverage, payload.normalization):
        assert np.isfinite(values).all() and (values >= 0).all()
    covered = payload.coverage[-1] > 0
    assert covered.sum() > 100 and np.isfinite(payload.intensity[-1][covered]).all()
    assert payload.profile.intensity.shape == (1500,)


@pytest.mark.slow
def test_scan14_xu_polarization_tracks_multigeometry(isolated_scan14_project):
    """XU applies pyFAI's polarization: the XU/MultiGeometry ratio is unchanged by it."""
    project = isolated_scan14_project
    labels = tuple(range(0, 287, 11))
    mg = _scan14_request(project, project / "pol-mg-legacy.nexus", labels)
    calibration = install_canonical_xu_stitch_calibration(project_root=project)
    with xu_runtime_session() as runtime:
        effective = resolve_xu_stitch_effective_geometry(calibration, runtime).projection
    table = run_metadata_table(MetadataTablePlan(mg.module.source.analysis.source_spec))
    xu_source = ModuleSourceReceipt.from_metadata_table(
        table, kind=ModuleKind.STITCH, selected_labels=labels,
        resolved_selectors=tuple(MetadataColumnSelector(name) for name in ("del", "nu", "energy")
                                 if name in {column.name for column in table.columns}))
    intensity = {}
    for factor in (None, .99):
        plans = {
            "mg": (mg.module.source, replace(mg.plan, npt_1d=760, radial_range=(1.0, 5.2),
                                             polarization_factor=factor)),
            "xu": (xu_source, XuStitchOperationPlan(calibration, effective, 1.0, 5.2, npt_1d=760,
                                                    max_frame_bytes=64 * 1024 * 1024,
                                                    polarization_factor=factor)),
        }
        for name, (source, plan) in plans.items():
            output = ModuleOutputRequest(project / f"pol-{name}-{factor}.nexus", AnalysisArtifactKind.STITCH_1D,
                                         AnalysisArtifactOverwrite.CREATE_NEW)
            result = run_stitch_operation(prepare_stitch_operation(source, output, plan, project_root=project))
            assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
            intensity[name, factor] = np.asarray(result.payload.intensity, dtype=float)
    plain = intensity["mg", None]
    rings = np.isfinite(np.stack(list(intensity.values()))).all(axis=0) & (plain > np.nanmedian(plain))
    effect = np.median(np.abs(intensity["mg", .99][rings] / plain[rings] - 1))
    ratio_on = intensity["xu", .99][rings] / intensity["mg", .99][rings]
    ratio_off = intensity["xu", None][rings] / plain[rings]
    drift = np.abs(ratio_on / ratio_off - 1)
    print(f"polarization effect median {effect:.4g}; XU/MG ratio drift median {np.median(drift):.4g} "
          f"p90 {np.percentile(drift, 90):.4g} max {drift.max():.4g}; bins {int(rings.sum())}")
    assert effect > .02, effect  # polarization moves the MultiGeometry intensities
    assert np.median(drift) < effect / 10
