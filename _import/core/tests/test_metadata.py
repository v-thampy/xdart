"""Tests for ssrl_xrd_tools.io.metadata."""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import pytest

from ssrl_xrd_tools.io.metadata import (
    _extract_scan_info,
    _find_sidecar,
    _parse_kv_pairs,
    read_image_metadata,
    read_pdi_metadata,
    read_txt_metadata,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_TXT_CONTENT = """\
# Counters
i0 = 1000.0, i1 = 500.0
# Motors
del = 15.5, eta = 0.2
User: testuser, time: Mon Jan 15 10:30:00 2024  # Temp
"""

# Primary PDI format: All Counters / All Motors sections.
# After newlines are replaced with ';', the layout becomes:
#   All Counters;i0 = 100.0;i1 = 50.0;;# All Motors;del = 15.5;eta = 0.2;#;1706000000.0
_PDI_V1_CONTENT = """\
All Counters
i0 = 100.0
i1 = 50.0

# All Motors
del = 15.5
eta = 0.2
#
1706000000.0"""

# Fallback PDI format: Diffractometer Motor Positions section.
# After newline replacement the region looks like:
#   ...for image;# 2Theta = 30.0;Theta = 15.0;# Calculated...
_PDI_V2_CONTENT = """\
# Pilatus image header
# Diffractometer Motor Positions for image
# 2Theta = 30.0
Theta = 15.0
# Calculated Detector Calibration Parameters for image:
wavelength = 1.54
"""

# Minimal PDI that matches neither primary nor fallback regex.
_PDI_MINIMAL_CONTENT = """\
# No recognisable sections here
foo bar baz
"""


@pytest.fixture
def txt_meta_file(tmp_path: Path) -> Path:
    p = tmp_path / "scan_0001.txt"
    p.write_text(_TXT_CONTENT)
    return p


@pytest.fixture
def pdi_meta_file_v1(tmp_path: Path) -> Path:
    p = tmp_path / "scan_0001.pdi"
    p.write_text(_PDI_V1_CONTENT)
    return p


@pytest.fixture
def pdi_meta_file_v2(tmp_path: Path) -> Path:
    p = tmp_path / "scan_0001_v2.pdi"
    p.write_text(_PDI_V2_CONTENT)
    return p


@pytest.fixture
def pdi_meta_file_minimal(tmp_path: Path) -> Path:
    p = tmp_path / "scan_minimal.pdi"
    p.write_text(_PDI_MINIMAL_CONTENT)
    return p


# ---------------------------------------------------------------------------
# TestReadTxtMetadata
# ---------------------------------------------------------------------------


class TestReadTxtMetadata:
    def test_basic_parsing(self, txt_meta_file: Path) -> None:
        result = read_txt_metadata(txt_meta_file)

        assert isinstance(result, dict)
        assert result["i0"] == pytest.approx(1000.0)
        assert result["i1"] == pytest.approx(500.0)
        assert result["del"] == pytest.approx(15.5)
        assert result["eta"] == pytest.approx(0.2)

    def test_epoch_present_and_correct(self, txt_meta_file: Path) -> None:
        result = read_txt_metadata(txt_meta_file)

        assert "epoch" in result
        expected = time.mktime(
            datetime.strptime("Mon Jan 15 10:30:00 2024", "%a %b %d %H:%M:%S %Y").timetuple()
        )
        assert result["epoch"] == pytest.approx(expected)

    def test_returns_flat_dict_of_floats(self, txt_meta_file: Path) -> None:
        result = read_txt_metadata(txt_meta_file)
        assert all(isinstance(v, float) for v in result.values())

    def test_missing_file(self, tmp_path: Path) -> None:
        result = read_txt_metadata(tmp_path / "nonexistent.txt")
        assert result == {}

    def test_malformed_file(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.txt"
        bad.write_text("this is not a valid metadata file\nno counters or motors here\n")
        result = read_txt_metadata(bad)
        assert result == {}

    def test_accepts_str_path(self, txt_meta_file: Path) -> None:
        result = read_txt_metadata(str(txt_meta_file))
        assert result["i0"] == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# TestReadPdiMetadata
# ---------------------------------------------------------------------------


class TestReadPdiMetadata:
    def test_primary_format_counters(self, pdi_meta_file_v1: Path) -> None:
        result = read_pdi_metadata(pdi_meta_file_v1)

        assert result["i0"] == pytest.approx(100.0)
        assert result["i1"] == pytest.approx(50.0)

    def test_primary_format_motors(self, pdi_meta_file_v1: Path) -> None:
        result = read_pdi_metadata(pdi_meta_file_v1)

        assert result["del"] == pytest.approx(15.5)
        assert result["eta"] == pytest.approx(0.2)

    def test_epoch_extraction(self, pdi_meta_file_v1: Path) -> None:
        result = read_pdi_metadata(pdi_meta_file_v1)

        assert "epoch" in result
        assert result["epoch"] == pytest.approx(1706000000.0)

    def test_fallback_format_motors(self, pdi_meta_file_v2: Path) -> None:
        result = read_pdi_metadata(pdi_meta_file_v2)

        assert result["2Theta"] == pytest.approx(30.0)
        assert result["Theta"] == pytest.approx(15.0)

    def test_fallback_format_twotheta_alias(self, pdi_meta_file_v2: Path) -> None:
        result = read_pdi_metadata(pdi_meta_file_v2)

        assert "TwoTheta" in result
        assert result["TwoTheta"] == pytest.approx(result["2Theta"])

    def test_fallback_format_counters_empty(self, pdi_meta_file_v2: Path) -> None:
        result = read_pdi_metadata(pdi_meta_file_v2)
        # No Counters section in fallback format — keys from All Counters are absent
        assert "i0" not in result
        assert "i1" not in result

    def test_minimal_fallback_last_resort(self, pdi_meta_file_minimal: Path) -> None:
        result = read_pdi_metadata(pdi_meta_file_minimal)

        assert result["TwoTheta"] == pytest.approx(0.0)
        assert result["Theta"] == pytest.approx(0.0)

    def test_missing_file(self, tmp_path: Path) -> None:
        result = read_pdi_metadata(tmp_path / "nonexistent.pdi")
        assert result == {}

    def test_accepts_str_path(self, pdi_meta_file_v1: Path) -> None:
        result = read_pdi_metadata(str(pdi_meta_file_v1))
        assert result["i0"] == pytest.approx(100.0)

    def test_returns_flat_dict_of_floats(self, pdi_meta_file_v1: Path) -> None:
        result = read_pdi_metadata(pdi_meta_file_v1)
        assert all(isinstance(v, float) for v in result.values())


# ---------------------------------------------------------------------------
# TestReadImageMetadata
# ---------------------------------------------------------------------------


class TestReadImageMetadata:
    def test_txt_sidecar_found(self, tmp_path: Path) -> None:
        image = tmp_path / "scan_0001.tif"
        image.touch()
        sidecar = tmp_path / "scan_0001.txt"
        sidecar.write_text(_TXT_CONTENT)

        result = read_image_metadata(image, meta_format="txt")
        assert result["i0"] == pytest.approx(1000.0)
        assert result["del"] == pytest.approx(15.5)

    def test_pdi_sidecar_found(self, tmp_path: Path) -> None:
        image = tmp_path / "scan_0001.tif"
        image.touch()
        sidecar = tmp_path / "scan_0001.pdi"
        sidecar.write_text(_PDI_V1_CONTENT)

        result = read_image_metadata(image, meta_format="pdi")
        assert result["del"] == pytest.approx(15.5)

    def test_sidecar_not_found(self, tmp_path: Path) -> None:
        image = tmp_path / "scan_0001.tif"
        image.touch()

        result = read_image_metadata(image, meta_format="txt")
        assert result == {}

    def test_appended_extension(self, tmp_path: Path) -> None:
        """Sidecar named image.tif.txt (extension appended) is found."""
        image = tmp_path / "scan_0001.tif"
        image.touch()
        sidecar = Path(str(image) + ".txt")
        sidecar.write_text(_TXT_CONTENT)

        result = read_image_metadata(image, meta_format="txt")
        assert result["i0"] == pytest.approx(1000.0)

    def test_replaced_extension_takes_priority(self, tmp_path: Path) -> None:
        """When both foo.txt and foo.tif.txt exist, foo.txt (replaced ext) wins."""
        image = tmp_path / "scan_0001.tif"
        image.touch()
        replaced = tmp_path / "scan_0001.txt"
        replaced.write_text(_TXT_CONTENT)
        appended = Path(str(image) + ".txt")
        # Write different i0 value to distinguish
        appended.write_text(_TXT_CONTENT.replace("i0 = 1000.0", "i0 = 9999.0"))

        result = read_image_metadata(image, meta_format="txt")
        assert result["i0"] == pytest.approx(1000.0)

    def test_unknown_format_returns_empty(self, tmp_path: Path) -> None:
        image = tmp_path / "scan_0001.tif"
        image.touch()
        result = read_image_metadata(image, meta_format="log")
        assert result == {}

    def test_accepts_str_path(self, tmp_path: Path) -> None:
        image = tmp_path / "scan_0001.tif"
        image.touch()
        (tmp_path / "scan_0001.txt").write_text(_TXT_CONTENT)

        result = read_image_metadata(str(image), meta_format="txt")
        assert result["i0"] == pytest.approx(1000.0)

    def test_default_format_is_txt(self, tmp_path: Path) -> None:
        image = tmp_path / "scan_0001.tif"
        image.touch()
        (tmp_path / "scan_0001.txt").write_text(_TXT_CONTENT)

        result = read_image_metadata(image)
        assert result["i0"] == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# TestHelpers
# ---------------------------------------------------------------------------


class TestHelpers:
    # --- _parse_kv_pairs ---

    def test_parse_kv_pairs_comma_equals(self) -> None:
        text = "i0 = 100.0, i1 = 200.0"
        result = _parse_kv_pairs(text, r",|=")
        assert result["i0"] == pytest.approx(100.0)
        assert result["i1"] == pytest.approx(200.0)

    def test_parse_kv_pairs_semicolon_equals(self) -> None:
        text = "del = 15.5;eta = 0.2"
        result = _parse_kv_pairs(text, r";|=")
        assert result["del"] == pytest.approx(15.5)
        assert result["eta"] == pytest.approx(0.2)

    def test_parse_kv_pairs_bad_values_skipped(self) -> None:
        text = "good = 42.0, bad = notanumber, also_good = 7.0"
        result = _parse_kv_pairs(text, r",|=")
        assert result["good"] == pytest.approx(42.0)
        assert result["also_good"] == pytest.approx(7.0)
        assert "bad" not in result

    def test_parse_kv_pairs_whitespace_key(self) -> None:
        text = "  motor name  = 3.14"
        result = _parse_kv_pairs(text, r",|=")
        assert result["motor"] == pytest.approx(3.14)

    def test_parse_kv_pairs_empty_string(self) -> None:
        result = _parse_kv_pairs("", r",|=")
        assert result == {}

    def test_parse_kv_pairs_returns_floats(self) -> None:
        result = _parse_kv_pairs("x = 1, y = 2", r",|=")
        assert all(isinstance(v, float) for v in result.values())

    # --- _find_sidecar ---

    def test_find_sidecar_replaced_ext(self, tmp_path: Path) -> None:
        image = tmp_path / "foo.tif"
        image.touch()
        sidecar = tmp_path / "foo.txt"
        sidecar.touch()

        found = _find_sidecar(image, "txt")
        assert found == sidecar

    def test_find_sidecar_appended_ext(self, tmp_path: Path) -> None:
        image = tmp_path / "foo.tif"
        image.touch()
        sidecar = Path(str(image) + ".txt")
        sidecar.touch()

        found = _find_sidecar(image, "txt")
        assert found == sidecar

    def test_find_sidecar_none_when_missing(self, tmp_path: Path) -> None:
        image = tmp_path / "foo.tif"
        image.touch()

        assert _find_sidecar(image, "txt") is None

    def test_find_sidecar_prefers_replaced_ext(self, tmp_path: Path) -> None:
        image = tmp_path / "foo.tif"
        image.touch()
        replaced = tmp_path / "foo.txt"
        replaced.touch()
        appended = Path(str(image) + ".txt")
        appended.touch()

        found = _find_sidecar(image, "txt")
        assert found == replaced

    # --- _extract_scan_info ---

    def test_extract_scan_info_standard(self, tmp_path: Path) -> None:
        p = tmp_path / "prefix_specname_scan42_0003.tif"
        spec_fname, scan_num, img_num = _extract_scan_info(p)

        assert spec_fname == "specname"
        assert scan_num == 42
        assert img_num == 3

    def test_extract_scan_info_with_b_prefix(self, tmp_path: Path) -> None:
        p = tmp_path / "b_prefix_specname_scan10_0001.edf"
        spec_fname, scan_num, img_num = _extract_scan_info(p)

        assert spec_fname == "specname"
        assert scan_num == 10
        assert img_num == 1

    def test_extract_scan_info_no_match(self, tmp_path: Path) -> None:
        p = tmp_path / "notanssrlfile.tif"
        spec_fname, scan_num, img_num = _extract_scan_info(p)

        assert spec_fname is None
        assert scan_num is None

    def test_extract_scan_info_multi_part_spec_name(self, tmp_path: Path) -> None:
        """Spec filenames with underscores: prefix_part1_part2_scan5_0002.tif."""
        p = tmp_path / "smpl_spec_file_scan5_0002.tif"
        spec_fname, scan_num, img_num = _extract_scan_info(p)

        assert spec_fname == "spec_file"
        assert scan_num == 5
        assert img_num == 2

    def test_extract_scan_info_zero_padded_numbers(self, tmp_path: Path) -> None:
        p = tmp_path / "run_myspec_scan001_0099.tif"
        spec_fname, scan_num, img_num = _extract_scan_info(p)

        assert scan_num == 1
        assert img_num == 99
