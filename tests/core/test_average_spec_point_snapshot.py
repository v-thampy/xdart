"""Average metadata admission uses bounded SPEC points, with real parser custody."""
from dataclasses import replace
import os
from pathlib import Path

import numpy as np
import pytest
import tifffile

from xrd_tools.io.metadata import MetadataInputTooLarge, read_image_metadata_observed
from xrd_tools.sources.execution_graph import (
    SourceRevisionChanged, qualify_source_execution_graph, validate_source_execution_graph,
)
from xrd_tools.sources.point_images import PointImagePolicy
from xrd_tools.sources.selection import image_series_spec


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
