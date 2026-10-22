"""Confirmed consecutive exposures use one logical scan-point source."""

from dataclasses import FrozenInstanceError, replace
import json

import h5py
import numpy as np
import pytest
import tifffile

from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.integrate.stitch_conditioning import StitchConditioning
from xrd_tools.sources.registry import open_source


def _spec_images(tmp_path, values=(10, 6, 12, 8)):
    path = tmp_path / "calibrant"
    # The command planned three points; only two were recorded.
    path.write_text("#F calibrant\n#E 1\n#O0 del  nu\n\n"
                    "#S 1 ascan del 10 12 2 1\n#P0 10 0\n#N 3\n"
                    "#L del  nu  i0\n10 0 100\n11 0 200\n")
    for index, value in enumerate(values):
        array = np.full((2, 3), value, dtype=np.uint16)
        tifffile.imwrite(tmp_path / f"calibrant_scan001_{index:04}.tif", array)
    return SourceSpec(path, SourceKind.SPEC, options={
        "scan": "1.1", "image_dir": str(tmp_path), "require_complete_images": True})


def _confirmed(spec, policy):
    return replace(spec, options={**spec.options, "point_images": policy.to_dict()})


@pytest.mark.parametrize("spec_state", ["absent", "not-spec", "other-scan"])
def test_auto_raw_sidecars_without_matching_spec_keep_run_and_average_admission(tmp_path, spec_state):
    from tests.core.test_metadata import _PDI_V1_CONTENT
    from xrd_tools.io.metadata import read_image_metadata_observed
    from xrd_tools.sources.execution_graph import qualify_source_execution_graph
    from xrd_tools.sources.point_images import inspect_point_images
    from xrd_tools.sources.selection import image_series_spec

    paths = [tmp_path / f"ordinary_scan1_{index:04}.raw" for index in range(2)]
    for path, value in zip(paths, (10, 20)):
        np.full((2, 3), value, np.int32).tofile(path)
        path.with_suffix(".raw.pdi").write_text(_PDI_V1_CONTENT)
    if spec_state == "not-spec":
        (tmp_path / "ordinary").write_text("ordinary metadata, not a SPEC file\n")
    elif spec_state == "other-scan":
        (tmp_path / "ordinary").write_text("#F ordinary\n#E 1\n#O0 del\n\n"
            "#S 2 ascan del 10 11 1 1\n#P0 10\n#N 2\n#L del  i0\n10 100\n11 100\n")
    spec = image_series_spec(paths[0], metadata_format="auto")
    spec = replace(spec, options={**spec.options, "detector_shape": (2, 3)})
    observed = read_image_metadata_observed(paths[0], "auto")
    assert observed.source_path == paths[0].with_suffix(".raw.pdi")
    assert inspect_point_images(spec) is None
    for reader_binding in (None, "average_closed_v1"):
        qualified = qualify_source_execution_graph(spec, reader_binding=reader_binding)
        assert qualified is not None
    source = open_source(spec)
    assert source.frame_indices == [1, 2]
    assert source.metadata_for(1)["i0"] == 100
    np.testing.assert_array_equal(source.load_frame(2), np.full((2, 3), 20))


@pytest.mark.parametrize("metadata_format", ["spec", "auto"])
def test_tiff_grouping_in_multiscan_folder_uses_exact_spec_scan(tmp_path, metadata_format):
    from xrd_tools.sources.execution_graph import qualify_source_execution_graph
    from xrd_tools.sources.point_images import inspect_point_images, PointImagePolicy, PointImagesRequired
    from xrd_tools.sources.selection import image_series_spec

    _spec_images(tmp_path)
    spec_path = tmp_path / "calibrant"
    spec_path.write_text(spec_path.read_text() + "\n#S 2 ascan del 20 21 1 1\n"
        "#P0 20 0\n#N 3\n#L del  nu  i0\n20 0 300\n21 0 400\n")
    for index in range(2):
        tifffile.imwrite(tmp_path / f"calibrant_scan002_{index:04}.tif", np.ones((2, 3), np.uint16))
    spec = image_series_spec(tmp_path / "calibrant_scan001_0000.tif", metadata_format=metadata_format)
    detection = inspect_point_images(spec)
    assert (detection.scan_label, detection.point_count, detection.image_count) == ("1.1", 2, 4)
    with pytest.raises(PointImagesRequired):
        qualify_source_execution_graph(spec)
    confirmed = _confirmed(spec, PointImagePolicy(2, "average"))
    source = open_source(confirmed)
    assert source.frame_indices == [1, 2]
    np.testing.assert_equal(source.load_frame(1), 8.)
    np.testing.assert_equal(source.load_frame(2), 10.)
    assert source.metadata_for(2)["del"] == 11
    assert inspect_point_images(image_series_spec(tmp_path / "calibrant_scan002_0000.tif",
                                                  metadata_format=metadata_format)) is None
    subset = replace(spec, options={**spec.options, "files": spec.options["files"][:2]})
    with pytest.raises(ValueError, match="exact recorded scan image series"):
        open_source(_confirmed(subset, PointImagePolicy(2, "average")))


def test_confirmed_auto_grouping_binds_matching_spec_and_source_mutations(tmp_path):
    from tests.core.test_metadata import _PDI_V1_CONTENT
    from xrd_tools.analysis.scan_operations import (
        AnalysisDisposition, AnalysisSourceLeaseRefused, MetadataTablePlan,
        requalified_analysis_source, run_metadata_table,
    )
    from xrd_tools.io.metadata import read_image_metadata_observed
    from xrd_tools.sources.execution_graph import (
        SourceRevisionChanged, qualify_source_execution_graph, validate_source_execution_graph,
    )
    from xrd_tools.sources.point_images import PointImagePolicy
    from xrd_tools.sources.selection import image_series_spec

    _spec_images(tmp_path)
    first = tmp_path / "calibrant_scan001_0000.tif"
    sidecar = first.with_suffix(".tif.pdi")
    sidecar.write_text(_PDI_V1_CONTENT)
    assert read_image_metadata_observed(first, "auto").source_path == sidecar
    spec_path = tmp_path / "calibrant"
    confirmed = _confirmed(image_series_spec(first, metadata_format="auto"), PointImagePolicy(2, "average"))
    graph = qualify_source_execution_graph(confirmed)
    assert {row.metadata_file.path for row in graph.stamp.metadata_sources} == {str(spec_path)}
    table = run_metadata_table(MetadataTablePlan(confirmed))
    assert table.disposition is AnalysisDisposition.COMPLETED, table.diagnostics
    assert spec_path in {path for path, _resolved, _revision in table.receipt.dependency_revisions}
    spec_path.write_text(spec_path.read_text() + "\n#C source changed after admission\n")
    with pytest.raises(SourceRevisionChanged):
        validate_source_execution_graph(graph)
    with pytest.raises(AnalysisSourceLeaseRefused):
        with requalified_analysis_source(table.receipt):
            pass


def _eiger(tmp_path):
    master = tmp_path / "two_per_point_master.h5"
    shard = tmp_path / "two_per_point_data_000001.h5"
    images = np.array([[[10, 0], [4, 65535]], [[6, 2], [8, 9]],
                       [[12, 4], [7, 8]], [[8, 6], [9, 10]]], dtype=np.uint16)
    with h5py.File(shard, "w") as stream:
        stream.create_dataset("entry/data/data", data=images)
    with h5py.File(master, "w") as stream:
        entry = stream.create_group("entry1")
        entry.attrs["NX_class"] = "NXentry"
        data = entry.create_group("data")
        data.attrs["NX_class"] = "NXdata"
        data["data_000001"] = h5py.ExternalLink(shard.name, "/entry/data/data")
        specific = entry.create_group("instrument/detector/detectorSpecific")
        specific.create_dataset("nimages", data=2)
        specific.create_dataset("ntrigger", data=2)
        specific.create_dataset("eiger_fw_version", data="release-2022.1.2")  # the DECTRIS vendor block
    return SourceSpec(master, SourceKind.NEXUS_STACK), images


def test_detection_is_pixel_free_and_requires_confirmation(tmp_path):
    from xrd_tools.sources.point_images import inspect_point_images, PointImagesRequired

    spec = _spec_images(tmp_path)
    detected = inspect_point_images(spec)
    assert (detected.point_count, detected.image_count, detected.images_per_point) == (2, 4, 2)
    assert detected.scan_label == "1.1" and detected.source_spec == spec
    with pytest.raises(PointImagesRequired) as error:
        open_source(spec)
    assert error.value.detection == detected
    assert "4 images" in str(error.value) and "2 scan points" in str(error.value)
    # Detection uses TIFF layout, not a decode: corrupt the pixel payload alone.
    path = tmp_path / "calibrant_scan001_0000.tif"
    with tifffile.TiffFile(path) as stream:
        pixel_offset = stream.pages[0].dataoffsets[0]
    with path.open("r+b") as stream:
        stream.truncate(pixel_offset)
    assert inspect_point_images(spec) == detected


@pytest.mark.parametrize("operation,expected", [("average", 8), ("difference", -4), ("select", 6)])
def test_spec_groups_native_images_and_preserves_point_metadata(tmp_path, operation, expected):
    from xrd_tools.sources.point_images import PointImagePolicy

    spec = _spec_images(tmp_path)
    policy = PointImagePolicy(2, operation, image_index=2 if operation == "select" else 1)
    source = open_source(_confirmed(spec, policy))
    assert source.frame_indices == [0, 1]
    assert source.load_frame(0).dtype == np.float64
    np.testing.assert_equal(source.load_native_frame(0), expected)
    assert source.metadata_for(1)["del"] == 11 and source.metadata_for(1)["i0"] == 200
    membership = source.point_raw_locators(1)
    expected_indices = (3,) if operation == "select" else (2, 3)
    assert tuple(int(path.stem.rsplit("_", 1)[1]) for path, _ in membership) == expected_indices
    assert source.raw_locator_for(1) == membership[0]
    provenance = source.metadata_for(1)["point_images"]
    assert provenance["operation"] == operation and provenance["images_per_point"] == 2
    assert len(provenance["raw_locators"]) == len(expected_indices)
    json.dumps(provenance)
    reopened = open_source(source.spec)
    np.testing.assert_equal(reopened.load_frame(1), source.load_frame(1))
    assert dict(reopened.spec.options)["point_images"] == policy.to_dict()
    assert source.frame_for(1).source_frame_index == 0
    with pytest.raises(IndexError):
        source.load_point_frame(-1)
    with pytest.raises(IndexError):
        source.load_point_frame(2)


def test_conditioning_is_per_input_and_union_masks_before_combination(tmp_path):
    from xrd_tools.sources.point_images import PointImagePolicy

    spec = _spec_images(tmp_path)
    first = np.array([[0, 20, 65535], [2, 3, 4]], dtype=np.uint16)
    second = np.array([[12, 2, 8], [4, 5, 6]], dtype=np.uint16)
    tifffile.imwrite(tmp_path / "calibrant_scan001_0000.tif", first)
    tifffile.imwrite(tmp_path / "calibrant_scan001_0001.tif", second)
    source = open_source(_confirmed(spec, PointImagePolicy(2, "average")))
    manual = source.load_point_frame(0, StitchConditioning("manual", 1, 15))
    assert np.isnan(manual[0]).all()  # averaging first would incorrectly admit each of these
    np.testing.assert_equal(manual[1], [3, 4, 5])
    auto = source.load_point_frame(0, StitchConditioning("auto"))
    assert np.isnan(auto[0, 2]) and auto[0, 0] == 6
    selected = open_source(_confirmed(spec, PointImagePolicy(2, "select", image_index=2)))
    np.testing.assert_equal(selected.load_point_frame(0, StitchConditioning("auto")), second)
    difference = open_source(_confirmed(spec, PointImagePolicy(2, "difference", reverse=True)))
    assert difference.load_point_frame(0)[0, 0] == -12


def test_existing_one_to_one_and_nonintegral_refusal_are_preserved(tmp_path):
    from xrd_tools.sources.point_images import inspect_point_images, PointImagePolicy

    spec = _spec_images(tmp_path, (10, 6))
    assert inspect_point_images(spec) is None
    source = open_source(spec)
    assert source.frame_indices == [0, 1]
    assert source.load_native_frame(0).dtype == np.uint16
    assert "point_images" not in source.metadata_for(0)
    with pytest.raises(ValueError, match="does not match"):
        open_source(_confirmed(spec, PointImagePolicy(2, "average")))
    tifffile.imwrite(tmp_path / "calibrant_scan001_0002.tif", np.ones((2, 3), np.uint16))
    assert inspect_point_images(spec) is None
    with pytest.raises(ValueError, match="2 metadata points but 3 image frames"):
        open_source(spec)


@pytest.mark.parametrize("kwargs", [
    {"images_per_point": 0, "operation": "average"},
    {"images_per_point": True, "operation": "average"},
    {"images_per_point": 2, "operation": "median"},
    {"images_per_point": 3, "operation": "difference"},
    {"images_per_point": 2, "operation": "select", "image_index": 3},
    {"images_per_point": 2, "operation": "select", "image_index": 0},
    {"images_per_point": 2, "operation": "average", "reverse": 1},
])
def test_policy_refuses_invalid_counts_and_operations(kwargs):
    from xrd_tools.sources.point_images import PointImagePolicy
    with pytest.raises((ValueError, TypeError)):
        PointImagePolicy(**kwargs)


def test_policy_is_immutable_serializable_and_operation_compatibility_is_explicit():
    from xrd_tools.sources.point_images import PointImagePolicy

    policy = PointImagePolicy(3, "select", image_index=3)
    assert policy == PointImagePolicy.from_dict(json.loads(json.dumps(policy.to_dict())))
    with pytest.raises(FrozenInstanceError):
        policy.image_index = 1
    assert policy.compatible_with(4) and not policy.compatible_with(2)
    assert PointImagePolicy(2, "average").compatible_with(3)
    assert PointImagePolicy(2, "difference").compatible_with(2)
    assert not PointImagePolicy(2, "difference").compatible_with(3)


def test_eiger_detection_and_confirmed_wrapper_leave_base_source_unchanged(tmp_path):
    from xrd_tools.sources.point_images import inspect_point_images, PointImagePolicy
    from xrd_tools.sources.nexus import NexusStackSource

    spec, images = _eiger(tmp_path)
    detection = inspect_point_images(spec)
    assert (detection.point_count, detection.image_count, detection.images_per_point) == (2, 4, 2)
    original = open_source(spec)
    assert type(original) is NexusStackSource and original.frame_indices == [0, 1, 2, 3]
    source = open_source(_confirmed(spec, PointImagePolicy(2, "difference")))
    assert source.frame_indices == [0, 1]
    np.testing.assert_equal(source.load_frame(1), images[3].astype(float) - images[2])
    assert source.raw_locator_for(1) == (spec.uri, 2)
    assert source.point_raw_locators(1) == ((spec.uri, 2), (spec.uri, 3))
    assert source.metadata_for(0)["point_images"]["raw_locators"][1]["frame"] == 1
    assert np.isnan(source.load_point_frame(0, StitchConditioning("auto"))[1, 1])
    reopened = open_source(source.spec)
    np.testing.assert_equal(reopened.load_frame(1), source.load_frame(1))
    source.close()
    reopened.close()
    original.close()


@pytest.mark.parametrize('operation', ['average', 'difference', 'select'])
def test_grouped_bluesky_counters_remain_point_rows(tmp_path, operation):
    from tests.core.test_metadata_provider import _bluesky, NF, IMG
    from xrd_tools.sources.point_images import PointImagePolicy
    path, hy, monitor, epoch = _bluesky(tmp_path / 'points.nxs')
    with h5py.File(path, 'r+') as stream:
        del stream['entry/data/eiger_image']
        signal = stream.create_dataset('entry/data/eiger_image',
            data=np.full((NF * 2, *IMG), 10, np.uint32), chunks=(2, *IMG))
        signal.attrs['signal_type'] = 'detector'
    spec = SourceSpec(path, SourceKind.NEXUS_STACK, options={
        'scan_point_count': NF, 'point_images': PointImagePolicy(2, operation, image_index=2).to_dict()})
    source = open_source(spec)
    try:
        for index in range(NF):
            row = source.metadata_for(index)
            assert row['i0'] == monitor[index] and row['EPOCH'] == epoch[index]
            assert row['hy'] == hy[index]
    finally:
        source.close()


def test_nexus_without_explicit_point_count_does_not_infer_from_motor_positions(tmp_path):
    from xrd_tools.sources.point_images import inspect_point_images, PointImagePolicy

    path = tmp_path / "raw.nxs"
    with h5py.File(path, "w") as stream:
        stream.create_dataset("entry/instrument/detector/data", data=np.ones((4, 2, 2), np.uint16))
        stream.create_dataset("entry/sample/del", data=[1, 1, 2, 2])
    spec = SourceSpec(path, SourceKind.NEXUS_STACK)
    assert inspect_point_images(spec) is None
    with pytest.raises(ValueError, match="point count"):
        open_source(_confirmed(spec, PointImagePolicy(2, "average")))


def test_eiger_acquisition_count_mismatch_refuses_explicit_grouping(tmp_path):
    from xrd_tools.sources.point_images import inspect_point_images, PointImagePolicy

    spec, _ = _eiger(tmp_path)
    with h5py.File(spec.uri, "r+") as stream:
        stream["entry1/instrument/detector/detectorSpecific/ntrigger"][()] = 3
    assert inspect_point_images(spec) is None
    with pytest.raises(ValueError, match="point count"):
        open_source(_confirmed(spec, PointImagePolicy(2, "average")))


@pytest.mark.parametrize("operation,expected_signal,expected_variance,multiplier", [
    ("average", 8., 4., 1.), ("difference", -4., 16., 0.), ("select", 6., 6., 1.),
])
def test_point_statistics_use_native_counts_before_combination(
        tmp_path, operation, expected_signal, expected_variance, multiplier):
    from xrd_tools.sources.point_images import PointImagePolicy
    source = open_source(_confirmed(_spec_images(tmp_path), PointImagePolicy(
        2, operation, image_index=2 if operation == "select" else 1)))
    try:
        result = source.load_point_statistics(0, StitchConditioning("auto"), poisson=True)
        np.testing.assert_equal(result.signal, expected_signal)
        np.testing.assert_equal(result.variance, expected_variance)
        assert result.exposure_multiplier == multiplier
    finally:
        getattr(source, "close", lambda: None)()


def test_native_transform_runs_before_difference_and_invalidates_variance(tmp_path):
    from xrd_tools.sources.point_images import PointImagePolicy
    source = open_source(_confirmed(_spec_images(tmp_path), PointImagePolicy(2, "difference")))
    def prepare(raw):
        image = raw.astype(float) + 5.
        image[raw > 9] = np.nan
        return image
    try:
        result = source.load_point_statistics(0, poisson=True, native_transform=prepare)
        assert np.isnan(result.signal).all() and np.isnan(result.variance).all()
        result = source.load_point_statistics(1, poisson=True,
            native_transform=lambda raw: raw.astype(float) + 5.)
        np.testing.assert_equal(result.signal, -4.)
        np.testing.assert_equal(result.variance, 20.)
    finally:
        getattr(source, "close", lambda: None)()


def test_one_based_names_are_acquisition_ordinals_not_spec_row_numbers(tmp_path):
    from xrd_tools.sources.point_images import PointImagePolicy
    spec = _spec_images(tmp_path)
    for index in reversed(range(4)):
        (tmp_path / f"calibrant_scan001_{index:04}.tif").rename(
            tmp_path / f"calibrant_scan001_{index + 1:04}.tif")
    source = open_source(_confirmed(spec, PointImagePolicy(2, "average")))
    np.testing.assert_equal(source.load_frame(0), 8.)
    np.testing.assert_equal(source.load_frame(1), 10.)
    assert source.metadata_for(0)["i0"] == 100.
    assert source.metadata_for(1)["i0"] == 200.


def test_container_point_statistics_cross_native_blocks_without_losing_operands(tmp_path):
    from xrd_tools.sources.point_images import PointImagePolicy
    spec, images = _eiger(tmp_path)
    source = open_source(_confirmed(spec, PointImagePolicy(2, "average")))
    try:
        chunks = list(source.iter_point_statistics(1, poisson=True,
                                                   conditioning=StitchConditioning("auto")))
        assert [labels for _results, labels in chunks] == [[0], [1]]
        for point, (results, _labels) in enumerate(chunks):
            assert len(results) == 1
            expected = images[point * 2:point * 2 + 2].astype(float)
            expected[expected >= 65535] = np.nan
            np.testing.assert_equal(results[0].signal, expected.mean(axis=0))
            np.testing.assert_equal(results[0].variance, np.maximum(expected, 1.).sum(axis=0) / 4.)
    finally:
        source.close()
