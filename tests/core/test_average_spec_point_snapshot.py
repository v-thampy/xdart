"""Average metadata admission uses bounded SPEC points, with real parser custody."""
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import time

import numpy as np
import pytest
import tifffile

from xrd_tools.io.metadata import MetadataInputTooLarge, read_image_metadata_observed
from xrd_tools.sources.execution_graph import (
    SourceRevisionChanged, qualify_source_execution_graph, validate_source_execution_graph,
)
from xrd_tools.sources.point_images import PointImagePolicy
from xrd_tools.sources.selection import image_series_spec


@pytest.mark.parametrize("relative,sha,scan,index,epoch", [
    ("RSM/STO_align", "12e0d05219cb0309f7fb5f170c8fb12c507e7603dcd71b3954721d1b6e0c90b1", 43, 0, 10055.891),
    ("RSM/STO_align", "12e0d05219cb0309f7fb5f170c8fb12c507e7603dcd71b3954721d1b6e0c90b1", 43, 1, 10057.903),
    ("RSM/STO_align", "12e0d05219cb0309f7fb5f170c8fb12c507e7603dcd71b3954721d1b6e0c90b1", 43, 60, 10177.966),
    ("RSM/STO_align", "12e0d05219cb0309f7fb5f170c8fb12c507e7603dcd71b3954721d1b6e0c90b1", 1196, 0, 285619.647),
    ("RSM/STO_align", "12e0d05219cb0309f7fb5f170c8fb12c507e7603dcd71b3954721d1b6e0c90b1", 1196, 60, 286238.813),
    ("stitching/data/efield/R2C1_extralong", "529fcc8c3800961a6eb8437639005c26926d48b05f9f03a84afdf37dafecd927", 1, 0, 23.776),
    ("stitching/data/efield/R2C1_extralong", "529fcc8c3800961a6eb8437639005c26926d48b05f9f03a84afdf37dafecd927", 1, 40, 47.584),
    ("stitching/data/efield/R2C1_extralong", "529fcc8c3800961a6eb8437639005c26926d48b05f9f03a84afdf37dafecd927", 10, 0, 703.972),
    ("stitching/data/efield/R2C1_extralong", "529fcc8c3800961a6eb8437639005c26926d48b05f9f03a84afdf37dafecd927", 10, 1238, 6509.799),
    ("stitching/data/efield/R2C1", "8623222fed86dca0edc77f9b632dd5241ef06886b780501989a3c38d8422df77", 7, 0, 2042.762),
    ("stitching/data/efield/R2C1", "8623222fed86dca0edc77f9b632dd5241ef06886b780501989a3c38d8422df77", 7, 314, 2663.617),
])
def test_real_spec_points_match_ordinary_reader_and_recorded_epochs(
        relative, sha, scan, index, epoch):
    """Real inherited headers, duplicate columns and final rows, through the public metadata API."""
    root = Path(os.environ.get("XDART_TEST_DATA", "/Users/vthampy/repos/test_data"))
    path = root / relative
    if not path.is_file():
        pytest.skip("authenticated real SPEC input is unavailable")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
    # Metadata-only final scans need no detector payload; the recorded image name selects the point.
    image = path.parent / "images" / f"b_thampy_{path.name}_scan{scan}_{index:04}.raw"
    reference = read_image_metadata_observed(image, "spec", meta_dir=path.parent)
    assert reference.source_path == path and reference.values["Epoch"] == epoch
    actual = read_image_metadata_observed(image, "spec", meta_dir=path.parent,
        max_input_bytes=1 << 16, spec_point_snapshot=True)
    assert actual == reference  # all values, inherited motors, source path and byte revision


def test_actual_sto_scan43_average_preserves_mean_counts_and_metadata(tmp_path):
    """Named real large-SPEC workload, through Average's ordinary public execution path."""
    from xrd_tools.core import PONI
    from xrd_tools.core.energy import energy_eV_to_wavelength_m
    from xrd_tools.core.geometry import DetectorCalibration
    from xrd_tools.integrate.calibration import detector_calibration_to_integrator
    from xrd_tools.io.read import get_1d, get_metadata
    from xrd_tools.io.spec import read_spec_scan_table
    from xrd_tools.reduction import AverageScanRecipe, Integration1DPlan, ReductionPlan
    from xrd_tools.reduction.average import AverageScanRunner, get_average_finite_counts
    from xrd_tools.reduction.core import integrate_1d
    from xrd_tools.session.experiment_state import CalibrationState, FactStatus, PoniValues

    root = Path(os.environ.get("XDART_TEST_DATA", "/Users/vthampy/repos/test_data")) / "RSM"
    spec = root / "STO_align"
    selected = root / "images/b_thampy_STO_align_scan43_0000.raw"
    if not spec.is_file() or not selected.is_file():
        pytest.skip("authenticated actual STO Average workload is unavailable")
    assert hashlib.sha256(spec.read_bytes()).hexdigest() == "12e0d05219cb0309f7fb5f170c8fb12c507e7603dcd71b3954721d1b6e0c90b1"
    source = image_series_spec(selected, metadata_format="spec")
    files = source.options["files"]
    assert tuple(Path(path).name for path in files) == tuple(
        f"b_thampy_STO_align_scan43_{index:04}.raw" for index in range(61))
    shape = (195, 487)
    source = replace(source, options={**source.options, "meta_dir": str(root),
        "detector_shape": shape, "raw_dtype": "int32", "raw_header_skip": 0})
    config = {"pixel1": .000172, "pixel2": .000172, "max_shape": list(shape), "orientation": 3}
    values = PoniValues(1.0147173, 97 * .000172, 243 * .000172, 0., 0., 0., 1e-10)
    calibration = CalibrationState(values, "Detector", config, status=FactStatus.PRESENT)
    runner = AverageScanRunner(AverageScanRecipe(source, tmp_path / "sto_scan43.nxs",
        ReductionPlan(integration_1d=Integration1DPlan(npt=256, method="numpy"), mask_saturation=False),
        calibration=calibration))
    started = time.monotonic()
    progress = []
    result = runner.start(progress_cb=lambda row: progress.append(
        (row.stage, row.completed, row.total, time.monotonic() - started)))
    elapsed = time.monotonic() - started
    close_started = time.monotonic()
    assert runner.close() is result
    close_elapsed = time.monotonic() - close_started
    print(f"STO_SCAN43_AVERAGE start={elapsed:.6f}s close={close_elapsed:.6f}s disposition={result.disposition}")
    for stage in dict.fromkeys(row[0] for row in progress):
        rows = [row for row in progress if row[0] == stage]
        print(f"STO_SCAN43_PROGRESS {stage} first={rows[0]} last={rows[-1]}")
    assert result.disposition == "COMMITTED", result
    assert result.contributor_extent == 61 and result.committed_labels == (1,)
    np.testing.assert_array_equal(get_average_finite_counts(result.target).values,
                                  np.full(shape, 61, np.uint32))
    # The independently accumulated actual RAW mean is reduced by the trusted integration owner.
    summed = np.zeros(shape, np.float64)
    for path in files:
        assert Path(path).stat().st_size == 379860
        summed += np.fromfile(path, dtype=np.int32).reshape(shape)
    columns, motors, points = read_spec_scan_table(spec, "43.1")
    assert points == 61 and motors["energy"] == 13000.007
    integrator = detector_calibration_to_integrator(DetectorCalibration(
        PONI(values.dist, values.poni1, values.poni2, 0., 0., 0.,
             energy_eV_to_wavelength_m(motors["energy"]), "Detector"), config))
    expected = integrate_1d(summed / 61, integrator, npt=256, method="numpy", unit="q_A^-1",
                            mask=np.zeros(shape, bool))
    actual = get_1d(result.target, 1)
    # The persisted schema stores axes, intensity and numeric scan-data as float32.
    np.testing.assert_array_equal(actual.q, expected.radial.astype(np.float32))
    np.testing.assert_array_equal(actual.intensity, expected.intensity.astype(np.float32))
    # ... except per-frame times, kept at float64 since 5d921d84 (float32 rounds them to 128 s steps).
    np.testing.assert_array_equal(get_metadata(result.target)["scan_data"]["Epoch"],
                                  np.asarray([np.mean(columns["Epoch"])], np.float64))


def _images(root, name="experiment", count=2, scan=1):
    files = tuple(root / f"{name}_scan{scan}_{index:04}.tif" for index in range(count))
    for index, path in enumerate(files):
        tifffile.imwrite(path, np.full((4, 4), 1 + 2 * index, np.uint16))
    return image_series_spec(files[0], metadata_format="spec")


def _spec(root):
    path = root / "experiment"
    path.write_text("#F experiment\n#E 1\n#O0 theta  phi\n\n"
        "#S 2 ascan theta 0 1 1 1\n#P0 5 6\n#N 2\n#L theta  I0\n5 100\n6 200\n"
        + "#C unrelated padding\n" * 10000
        + "#F experiment\n#E 2\n#O0 theta  chi\n#S 1 ascan theta 0 1 1 1\n#P0 7 8\n#N 2\n"
        "#L theta  I0\n7 300\n8 400\n\n#S 1 duplicate\n#P0 9 10\n#N 2\n"
        "#L theta  I0\n9 900\n")
    return path


def test_real_large_spec_admits_average_with_original_point_metadata(tmp_path):
    from xrd_tools.reduction import AverageScanRecipe, Integration1DPlan, ReductionPlan
    from xrd_tools.reduction.average import AverageScanRunner, get_average_finite_counts
    from xrd_tools.session.experiment_state import CalibrationState, FactStatus, PoniValues

    root = Path(os.environ.get("XDART_TEST_DATA", "/Users/vthampy/repos/test_data"))
    spec_path = root / "stitching/data/efield/R2C1_extralong"
    if not spec_path.is_file():
        pytest.skip("real large SPEC input is unavailable")
    assert spec_path.stat().st_size == 459239
    source = _images(tmp_path, name=spec_path.name)
    source = replace(source, options={**source.options, "meta_dir": str(spec_path.parent)})
    references = [read_image_metadata_observed(path, "spec", meta_dir=spec_path.parent)
                  for path in source.options["files"]]
    assert all(row.source_path == spec_path and row.values for row in references)
    graph = qualify_source_execution_graph(source, reader_binding="average_closed_v1")
    assert {row.metadata_file.path for row in graph.stamp.metadata_sources} == {str(spec_path)}
    for path, reference in zip(source.options["files"], references):
        actual = read_image_metadata_observed(path, "spec", meta_dir=spec_path.parent,
            max_input_bytes=1 << 16, spec_point_snapshot=True)
        assert actual.values == reference.values
        assert actual.source_revision == reference.source_revision
    calibration = CalibrationState(
        PoniValues(.2, .001, .001, 0., 0., 0., 1e-10), "Detector",
        {"pixel1": 1e-4, "pixel2": 1e-4, "max_shape": [4, 4], "orientation": 3},
        status=FactStatus.PRESENT,
    )
    runner = AverageScanRunner(AverageScanRecipe(source, tmp_path / "average.nxs",
        ReductionPlan(integration_1d=Integration1DPlan(npt=4, method="numpy")),
        calibration=calibration))
    result = runner.start()
    assert result.disposition == "COMMITTED", result
    assert runner.close() is result
    np.testing.assert_array_equal(get_average_finite_counts(result.target).values,
                                  np.full((4, 4), 2, np.uint32))


@pytest.mark.parametrize("index", [0, 1, 99])
def test_point_snapshot_preserves_scan_occurrence_motor_updates_and_last_point(tmp_path, index):
    path = _spec(tmp_path)
    image = tmp_path / f"experiment_scan1_{index:04}.tif"
    reference = read_image_metadata_observed(image, "spec")
    assert reference.values["I0"] == (300 if index == 0 else 400)
    assert reference.values["chi"] == 8
    actual = read_image_metadata_observed(image, "spec", max_input_bytes=1 << 16,
                                         spec_point_snapshot=True)
    assert actual == reference
    with pytest.raises(MetadataInputTooLarge):
        read_image_metadata_observed(image, "spec", max_input_bytes=1 << 16)
    assert actual.source_path == path


@pytest.mark.parametrize("metadata_format", ["spec", "auto"])
def test_grouped_average_binds_large_spec_and_refuses_changed_source(tmp_path, metadata_format):
    path = _spec(tmp_path)
    source = _images(tmp_path, count=4)
    source = replace(source, options={**source.options, "metadata_format": metadata_format,
        "point_images": PointImagePolicy(2, "average").to_dict()})
    graph = qualify_source_execution_graph(source, reader_binding="average_closed_v1")
    assert {row.metadata_file.path for row in graph.stamp.metadata_sources} == {str(path)}
    validate_source_execution_graph(graph)
    path.write_text(path.read_text().replace("8 400", "8 401"))
    with pytest.raises(SourceRevisionChanged):
        validate_source_execution_graph(graph)


def test_point_snapshot_retains_parser_payload_line_and_physical_file_bounds(tmp_path):
    path = _spec(tmp_path)
    image = tmp_path / "experiment_scan1_0000.tif"
    with path.open("r+b") as stream:
        stream.truncate((1 << 25) + 1)
    with pytest.raises(MetadataInputTooLarge, match="SPEC file"):
        read_image_metadata_observed(image, "spec", max_input_bytes=1 << 16,
                                     spec_point_snapshot=True)
    path.write_text("#F experiment\n#E 1\n#O0 theta\n#S 1 ascan theta 0 1 1 1\n"
                    "#P0 1\n#N 1\n#L I0\n" + "1 " * (1 << 16) + "\n")
    with pytest.raises(MetadataInputTooLarge, match="line"):
        read_image_metadata_observed(image, "spec", max_input_bytes=1 << 16,
                                     spec_point_snapshot=True)
    path.write_text("#F experiment\n#E 1\n#S 1 ascan theta 0 1 1 1\n#N 1\n#L I0\n"
                    + "#C retained comment\n" * 5000 + "1\n")
    with pytest.raises(MetadataInputTooLarge, match="point snapshot"):
        read_image_metadata_observed(image, "spec", max_input_bytes=1 << 16,
                                     spec_point_snapshot=True)


def test_point_snapshot_stops_at_the_next_file_header(tmp_path):
    path = _spec(tmp_path)
    path.write_text(path.read_text().replace("#C unrelated padding\n" * 10000, ""))
    image = tmp_path / "experiment_scan2_0000.tif"
    reference = read_image_metadata_observed(image, "spec")
    assert reference.values["I0"] == 100
    actual = read_image_metadata_observed(image, "spec", max_input_bytes=1 << 16,
                                         spec_point_snapshot=True)
    assert actual == reference


def test_point_snapshot_refuses_bytes_changed_and_restored_during_read(tmp_path, monkeypatch):
    path = _spec(tmp_path)
    original = path.read_bytes()
    image = tmp_path / "experiment_scan1_0000.tif"
    real_open = Path.open
    mutated = []
    class Reading:
        def __init__(self, stream): self.stream = stream
        def __enter__(self): return self
        def __exit__(self, *args): return self.stream.__exit__(*args)
        def fileno(self): return self.stream.fileno()
        def readline(self, size):
            value = self.stream.readline(size)
            if not mutated:
                with real_open(path, "wb") as writer:
                    writer.write(original.replace(b"8 400", b"8 401"))
                with real_open(path, "wb") as writer:
                    writer.write(original)
                mutated.append(True)
            return value
    def opening(value, *args, **kwargs):
        stream = real_open(value, *args, **kwargs)
        return Reading(stream) if value == path and args == ("rb",) else stream
    monkeypatch.setattr(Path, "open", opening)
    with pytest.raises(OSError, match="metadata input changed during bounded read"):
        read_image_metadata_observed(image, "spec", max_input_bytes=1 << 16,
                                     spec_point_snapshot=True)
    assert mutated == [True] and path.read_bytes() == original
