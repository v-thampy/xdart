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
