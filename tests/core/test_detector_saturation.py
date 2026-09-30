"""Decision 39: the automatic ceiling is the value the file records, else the detector's documented
counter limit, else the data type's maximum; Auto leaves out native values at or above it, and the
native exclusions (negatives, the uint32 sentinel, the dtype ceiling) stay."""
from pathlib import Path
import os

import h5py
import numpy as np
import pytest
import tifffile

from xrd_tools.core.invalid import (
    PILATUS_COUNTER_LIMIT, DetectorSaturation, detector_value_mask, documented_saturation,
    resolve_detector_saturation, saturation_count,
)
from xrd_tools.io.image import pilatus_count_cutoff, read_detector_image_layout
from xrd_tools.io.nexus import _read_saturation
from xrd_tools.session.threshold_editing import ThresholdState, threshold_transition
from xrd_tools.sources.descriptor import describe_container


def test_the_recorded_value_comes_first_even_above_the_pilatus_counter():
    # R20-P1: DECTRIS shows Count_cutoff 1,208,517 in a rate-corrected PILATUS3 header.
    assert resolve_detector_saturation(recorded=1_208_517, detector="Pilatus300kw", dtype="<i4") \
        == DetectorSaturation(1_208_517, "recorded")
    assert resolve_detector_saturation(recorded=None, detector="Pilatus300kw", dtype="<i4") \
        == DetectorSaturation(PILATUS_COUNTER_LIMIT, "detector")
    assert resolve_detector_saturation(detector="Pilatus 300kw", dtype="<i4").value == 1_048_575
    assert resolve_detector_saturation(detector="Eiger2 CdTe 1M", dtype="<u4") \
        == DetectorSaturation(4294967295, "dtype")
    assert resolve_detector_saturation(dtype="<u2") == DetectorSaturation(65535, "dtype")


def test_a_value_the_data_type_cannot_reach_leaves_the_data_type_rule():
    # uint16 frames never exceed 65535, which is always left out: nothing changes, nothing is recorded
    assert resolve_detector_saturation(detector="Pilatus100k", dtype="<u2") == DetectorSaturation(65535, "dtype")
    assert resolve_detector_saturation(recorded=299963, dtype="<u2") == DetectorSaturation(65535, "dtype")
    assert resolve_detector_saturation(recorded=65535, dtype="<u2") == DetectorSaturation(65535, "dtype")
    assert resolve_detector_saturation(recorded=65534, dtype="<u2") == DetectorSaturation(65534, "recorded")


@pytest.mark.parametrize("name", ["Pilatus4_2M", "pilatus4_cdte_1m", "Pilatus4 260kw"])
def test_pilatus4_takes_no_documented_limit(name):
    assert documented_saturation(name) is None
    assert resolve_detector_saturation(detector=name, dtype="<i4").source == "dtype"


def test_float_data_takes_no_counter_limit_and_no_dtype_ceiling():
    assert resolve_detector_saturation(detector="Pilatus100k", dtype="<f4") is None
    # a value the file records applies to any data
    assert resolve_detector_saturation(recorded=299963, detector=None, dtype="<f4").value == 299963


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -5, 0, 1.5, [1, 2], "299963", True, b"7", None])
def test_a_malformed_recorded_value_falls_to_the_next_rule(value):
    assert saturation_count(value) is None
    assert resolve_detector_saturation(recorded=value, detector="Pilatus100k", dtype="<i4").source == "detector"


def test_whole_float_and_numpy_counts_are_accepted():
    assert saturation_count(np.int64(299963)) == 299963
    assert saturation_count(np.uint32(599940)) == 599940
    assert saturation_count(1048575.0) == 1048575


def test_auto_leaves_out_values_at_or_above_the_ceiling_and_keeps_the_native_exclusions():
    raw = np.array([[1048574, 1048575, 1048576, -1]], dtype=np.int32)
    assert detector_value_mask(None, raw, enabled=True, saturation=1048575).tolist() == [[False, True, True, True]]
    # R20 boundary row 2: a uint16 65535 stays out under a higher recorded cutoff.
    raw16 = np.array([[65534, 65535]], dtype=np.uint16)
    assert detector_value_mask(None, raw16, enabled=True, saturation=299963).tolist() == [[False, True]]
    # the uint32 sentinel stays out, and nothing is masked with Auto off
    raw32 = np.array([[299962, 299963, 4294967295]], dtype=np.uint32)
    assert detector_value_mask(None, raw32, enabled=True, saturation=299963).tolist() == [[False, True, True]]
    assert detector_value_mask(None, raw32, enabled=False, saturation=299963) is None
    # the dtype rule alone, as before: only the ceiling itself
    assert detector_value_mask(None, raw32, enabled=True).tolist() == [[False, False, True]]
    with pytest.raises(TypeError):
        detector_value_mask(None, raw32, enabled=True, saturation=np.int64(5))


def test_eiger_masters_record_their_saturation():
    data = Path(os.environ.get("XDART_TEST_DATA", "")) / "eiger"
    master = data / "eiger_w2s3_test_2_scan001_master.h5"
    if not master.is_file():
        pytest.skip("XDART_TEST_DATA Eiger master unavailable")
    assert describe_container(master).saturation == 299963
    assert read_detector_image_layout(master).saturation == 299963


def test_the_nexus_reader_falls_back_to_the_count_cutoff_and_ignores_malformed_values(tmp_path):
    path = tmp_path / "scan.h5"
    with h5py.File(path, "w") as handle:
        detector = handle.create_group("entry/instrument/detector")
        detector["saturation_value"] = np.float64(np.nan)
        detector.create_group("detectorSpecific")["countrate_correction_count_cutoff"] = np.uint32(599940)
    with h5py.File(path, "r") as handle:
        assert _read_saturation(handle["entry"]) == 599940
        assert _read_saturation(None) is None
    with h5py.File(path, "a") as handle:
        del handle["entry/instrument/detector/detectorSpecific"]
    with h5py.File(path, "r") as handle:
        assert _read_saturation(handle["entry"]) is None


def test_a_pilatus_header_records_its_count_cutoff(tmp_path):
    header = ("# Detector: PILATUS3 1M, S/N 10-0001\r\n# Exposure_time 1.0000000 s\r\n"
              "# Count_cutoff 1208517 counts\r\n# Threshold_setting: 6000 eV\r\n")
    image = tmp_path / "frame_00001.tif"
    tifffile.imwrite(image, np.zeros((4, 6), dtype=np.int32), description=header)
    assert read_detector_image_layout(image).saturation == 1208517
    plain = tmp_path / "plain.tif"
    tifffile.imwrite(plain, np.zeros((4, 6), dtype=np.uint16))
    assert read_detector_image_layout(plain).saturation is None
    # a CBF's header mapping, as fabio returns it; two different cutoffs are ambiguous
    assert pilatus_count_cutoff({"_array_data.header_contents": header, "X-Binary-Size": "96"}) == 1208517
    assert pilatus_count_cutoff(header + "# Count_cutoff 1048500 counts\r\n") is None
    assert pilatus_count_cutoff(None) is None


def test_typing_the_hints_number_is_manual_and_keeps_it():
    # R20 boundary row 1: Auto leaves out >= S and its default maximum is the largest value it keeps,
    # S - 1; typing S is manual and keeps S (manual leaves out only > max).
    auto = ThresholdState()
    hint = 1048575
    typed = threshold_transition(auto, "max", str(hint), float(hint - 1))
    assert typed == ThresholdState(True, 0.0, float(hint), False)
    # the default itself keeps Auto: the same values are kept (a Run harvest of a shown default)
    assert threshold_transition(auto, "max", str(hint - 1), float(hint - 1)) is None
    # clearing the box keeps Auto; a minimum keeps Auto's ceiling as the largest kept value
    assert threshold_transition(auto, "max", "", float(hint - 1)) == auto
    assert threshold_transition(auto, "min", "4", float(hint - 1)) == ThresholdState(
        True, 4.0, float(hint - 1), False)
