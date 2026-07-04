"""Tests for xrd_tools.io.metadata."""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import pytest

import xrd_tools.io.metadata as metadata_module
from xrd_tools.io.metadata import (
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

_FIXTURE_DIR = Path(__file__).parent / "fixtures"

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

_STRUCTURED_CONTENT = """\
[metadata]
sampleName={sample}
scanNumber=1
exposureTime={exposure}
timeStamp=1773873180.171
dateTime=@DateTime(2026-03-18 12:33:00.171)
emptyValue=
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


@pytest.fixture
def clear_auto_sidecar_cache():
    metadata_module._AUTO_SIDECAR_CACHE.clear()
    yield
    metadata_module._AUTO_SIDECAR_CACHE.clear()


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

    def test_sidecar_extension_matching_is_case_insensitive(self, tmp_path: Path) -> None:
        image = tmp_path / "scan_0001.tif"
        image.touch()
        sidecar = tmp_path / "scan_0001.TXT"
        sidecar.write_text(_TXT_CONTENT)

        found = _find_sidecar(image, "txt")
        result = read_image_metadata(image, meta_format="txt")

        assert found == sidecar
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

    def test_structured_metadata_appended_extension_configparser_fixture(
        self,
        tmp_path: Path,
    ) -> None:
        image = tmp_path / "scan_0001.tif"
        image.touch()
        sidecar = Path(str(image) + ".metadata")
        sidecar.write_text(
            (_FIXTURE_DIR / "aps_qxrd_trimmed.tif.metadata").read_text()
        )

        result = read_image_metadata(image, meta_format="metadata")

        assert result["sampleName"] == "P25_C5"
        assert result["exposureTime"] == 1
        assert isinstance(result["exposureTime"], int)
        assert result["timeStamp"] == pytest.approx(1773873180.171)
        assert result["dateTime"] == "@DateTime(2026-03-18 12:33:00.171)"
        assert result["emptyValue"] == ""
        assert result["repeatedKey"] == "new"

    def test_structured_metadata_replaced_extension_takes_priority(
        self,
        tmp_path: Path,
    ) -> None:
        image = tmp_path / "scan_0001.tif"
        image.touch()
        replaced = tmp_path / "scan_0001.metadata"
        replaced.write_text(_STRUCTURED_CONTENT.format(sample="replaced", exposure=1))
        appended = Path(str(image) + ".metadata")
        appended.write_text(_STRUCTURED_CONTENT.format(sample="appended", exposure=2))

        result = read_image_metadata(image, meta_format="metadata")

        assert result["sampleName"] == "replaced"
        assert result["exposureTime"] == 1

    def test_structured_metadata_colon_fallback(self, tmp_path: Path) -> None:
        image = tmp_path / "scan_0001.tif"
        image.touch()
        sidecar = tmp_path / "scan_0001.meta"
        sidecar.write_text(
            "alpha: 1\n"
            "beta: 2.5\n"
            "dateTime: @DateTime(2026-03-18 12:33:00.171)\n"
            "empty:\n"
            "alpha: 3\n"
        )

        result = read_image_metadata(image, meta_format="meta")

        assert result["alpha"] == 3
        assert result["beta"] == pytest.approx(2.5)
        assert result["dateTime"] == "@DateTime(2026-03-18 12:33:00.171)"
        assert result["empty"] == ""

    def test_auto_metadata_discovers_sidecar_and_uses_directory_cache(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        clear_auto_sidecar_cache,
    ) -> None:
        image1 = tmp_path / "scan_0001.tif"
        image2 = tmp_path / "scan_0002.tif"
        image1.touch()
        image2.touch()
        Path(str(image1) + ".metadata").write_text(
            _STRUCTURED_CONTENT.format(sample="first", exposure=1)
        )
        Path(str(image2) + ".metadata").write_text(
            _STRUCTURED_CONTENT.format(sample="second", exposure=2)
        )

        first = read_image_metadata(image1, meta_format="auto")

        assert first["sampleName"] == "first"
        assert metadata_module._AUTO_SIDECAR_CACHE[(tmp_path, ".tif")] == (
            "append",
            ".metadata",
        )

        def fail_discovery(_image_path: Path):
            raise AssertionError("auto discovery should use cached convention")

        monkeypatch.setattr(metadata_module, "_discover_auto_sidecar", fail_discovery)
        second = read_image_metadata(image2, meta_format="auto")

        assert second["sampleName"] == "second"
        assert second["exposureTime"] == 2

    def test_auto_metadata_discovers_aps_appended_metadata_fixture(
        self,
        tmp_path: Path,
        clear_auto_sidecar_cache,
    ) -> None:
        image = tmp_path / "aps_qxrd_trimmed.tif"
        image.touch()
        Path(str(image) + ".metadata").write_text(
            (_FIXTURE_DIR / "aps_qxrd_trimmed.tif.metadata").read_text()
        )

        result = read_image_metadata(image, meta_format="auto")

        assert result["sampleName"] == "P25_C5"
        assert result["exposureTime"] == 1
        assert metadata_module._AUTO_SIDECAR_CACHE[(tmp_path, ".tif")] == (
            "append",
            ".metadata",
        )

    def test_auto_metadata_discovers_case_insensitive_appended_sidecar(
        self,
        tmp_path: Path,
        clear_auto_sidecar_cache,
    ) -> None:
        image = tmp_path / "scan_0001.tif"
        image.touch()
        Path(str(image) + ".METADATA").write_text(
            _STRUCTURED_CONTENT.format(sample="upper", exposure=4)
        )

        result = read_image_metadata(image, meta_format="auto")

        assert result["sampleName"] == "upper"
        assert metadata_module._AUTO_SIDECAR_CACHE[(tmp_path, ".tif")] == (
            "append",
            ".METADATA",
        )

    def test_auto_metadata_discovers_ssrl_txt_sidecar(
        self,
        tmp_path: Path,
        clear_auto_sidecar_cache,
    ) -> None:
        image = tmp_path / "scan_0001.tif"
        image.touch()
        (tmp_path / "scan_0001.txt").write_text(_TXT_CONTENT)

        result = read_image_metadata(image, meta_format="auto")

        assert result["i0"] == pytest.approx(1000.0)
        assert result["del"] == pytest.approx(15.5)

    def test_auto_metadata_does_not_cache_negative_late_sidecar(
        self,
        tmp_path: Path,
        clear_auto_sidecar_cache,
    ) -> None:
        image = tmp_path / "scan_0001.tif"
        image.touch()

        assert read_image_metadata(image, meta_format="auto") == {}
        assert (tmp_path, ".tif") not in metadata_module._AUTO_SIDECAR_CACHE

        Path(str(image) + ".metadata").write_text(
            _STRUCTURED_CONTENT.format(sample="late", exposure=3)
        )
        result = read_image_metadata(image, meta_format="auto")

        assert result["sampleName"] == "late"
        assert metadata_module._AUTO_SIDECAR_CACHE[(tmp_path, ".tif")] == (
            "append",
            ".metadata",
        )

    def test_none_meta_format_uses_auto_metadata(
        self,
        tmp_path: Path,
        clear_auto_sidecar_cache,
    ) -> None:
        image = tmp_path / "scan_0001.tif"
        image.touch()
        Path(str(image) + ".metadata").write_text(
            _STRUCTURED_CONTENT.format(sample="none-auto", exposure=1)
        )

        result = read_image_metadata(image, meta_format=None)

        assert result["sampleName"] == "none-auto"

    def test_explicit_format_accepts_one_or_two_field_sidecar(
        self,
        tmp_path: Path,
    ) -> None:
        # BL-3: an EXPLICIT meta_format is a deliberate user choice, so a 1-2
        # field sidecar must NOT be silently dropped by the AUTO min-pairs=3
        # plausibility gate (it used to return {} with a misleading "unknown
        # meta_format" warning).
        image = tmp_path / "scan_0001.tif"
        image.touch()
        sidecar = tmp_path / "scan_0001.metadata"
        sidecar.write_text("alpha=1\nthis is junk\nbeta=2\n")

        result = read_image_metadata(image, meta_format="metadata")

        assert result == {"alpha": 1, "beta": 2}

    def test_structured_metadata_reads_undecodable_values_with_replacement(
        self,
        tmp_path: Path,
    ) -> None:
        image = tmp_path / "scan_0001.tif"
        image.touch()
        sidecar = tmp_path / "scan_0001.metadata"
        sidecar.write_bytes(
            b"[metadata]\nalpha=1\nbeta=2.5\nbad=abc\xffdef\n"
        )

        result = read_image_metadata(image, meta_format="metadata")

        assert result["alpha"] == 1
        assert result["beta"] == pytest.approx(2.5)
        assert result["bad"] == "abc\ufffddef"

    def test_txt_format_regression_does_not_use_structured_fallback(
        self,
        tmp_path: Path,
    ) -> None:
        image = tmp_path / "scan_0001.tif"
        image.touch()
        (tmp_path / "scan_0001.txt").write_text(
            "alpha=1\nbeta=2\ngamma=3\n"
        )

        result = read_image_metadata(image, meta_format="txt")

        assert result == {}


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
    #
    # Three filename layouts are recognised, see _extract_scan_info docstring:
    #   1. ``b_<username>_<specname>_scan<N>_<M>.<ext>``     (Pilatus convention)
    #   2. ``checkout_<specname>_scan<N>_<M>.<ext>``         (one specific user)
    #   3. ``<specname>_scan<N>_<M>.<ext>``                  (generic; no strip)
    # And the trailing ``_scan<N>_<M>`` token's <M> is either an integer
    # (single-frame TIF/RAW/CBF/EDF) or the literal "master" (Eiger HDF5).

    def test_extract_scan_info_generic_no_prefix_strip(self, tmp_path: Path) -> None:
        """Rule 3: filenames that don't match b_ or checkout_ keep
        their whole prefix as the SPEC name."""
        p = tmp_path / "eiger_NbH_ctrl_eta3p0_scan001_0003.tif"
        spec_fname, scan_num, img_num = _extract_scan_info(p)

        assert spec_fname == "eiger_NbH_ctrl_eta3p0"
        assert scan_num == 1
        assert img_num == 3

    def test_extract_scan_info_with_b_prefix(self, tmp_path: Path) -> None:
        """Rule 1: ``b_<username>_<specname>_scan...`` strips through the
        second underscore (the one after the username)."""
        p = tmp_path / "b_thampy_STO_align_scan20_0000.raw"
        spec_fname, scan_num, img_num = _extract_scan_info(p)

        assert spec_fname == "STO_align"
        assert scan_num == 20
        assert img_num == 0

    def test_extract_scan_info_with_checkout_prefix(self, tmp_path: Path) -> None:
        """Rule 2: ``checkout_<specname>_scan...`` strips the literal
        ``checkout_`` prefix.  The one non-``b_`` username at the
        beamline."""
        p = tmp_path / "checkout_NbH_eta3p0_scan001_0001.tif"
        spec_fname, scan_num, img_num = _extract_scan_info(p)

        assert spec_fname == "NbH_eta3p0"
        assert scan_num == 1
        assert img_num == 1

    def test_extract_scan_info_eiger_master(self, tmp_path: Path) -> None:
        """Eiger master HDF5 filenames end in ``_scan<N>_master`` —
        no per-frame index in the filename itself, so
        ``image_number`` returns 0."""
        p = tmp_path / "eiger_NbH_ctrl_eta3p0_scan001_master.h5"
        spec_fname, scan_num, img_num = _extract_scan_info(p)

        assert spec_fname == "eiger_NbH_ctrl_eta3p0"
        assert scan_num == 1
        assert img_num == 0

    def test_extract_scan_info_no_match(self, tmp_path: Path) -> None:
        p = tmp_path / "notanssrlfile.tif"
        spec_fname, scan_num, img_num = _extract_scan_info(p)

        assert spec_fname is None
        assert scan_num is None

    def test_extract_scan_info_multi_part_spec_name(self, tmp_path: Path) -> None:
        """Generic (Rule 3) with an underscore-bearing SPEC name."""
        p = tmp_path / "smpl_spec_file_scan5_0002.tif"
        spec_fname, scan_num, img_num = _extract_scan_info(p)

        assert spec_fname == "smpl_spec_file"
        assert scan_num == 5
        assert img_num == 2

    def test_extract_scan_info_zero_padded_numbers(self, tmp_path: Path) -> None:
        """Generic (Rule 3): zero-padded scan/image numbers parse as ints."""
        p = tmp_path / "run_myspec_scan001_0099.tif"
        spec_fname, scan_num, img_num = _extract_scan_info(p)

        assert spec_fname == "run_myspec"
        assert scan_num == 1
        assert img_num == 99


# ---------------------------------------------------------------------------
# SPEC file location: extensionless name, searched in the image's folder plus
# up to two parent directories above it.
# ---------------------------------------------------------------------------

_SPEC_MINIMAL = """#F myscan
#E 1
#D today
#O0 th  chi  phi

#S 5 ascan th 0 2 2 1
#D today
#P0 0 5 10
#N 4
#L th  Epoch  i0  det
0 1 100 10
1 2 110 20
2 3 120 30
"""


@pytest.mark.parametrize("levels_up", [0, 1, 2])
def test_read_spec_metadata_finds_extensionless_spec_within_two_parents(
        tmp_path: Path, levels_up: int) -> None:
    """The SPEC file is extensionless and is found in the image's own folder
    (level 0) or up to two parent directories above it (levels 1 and 2)."""
    pytest.importorskip("silx")

    img_dir = tmp_path / "a" / "b" / "c"
    img_dir.mkdir(parents=True)
    image = img_dir / "myscan_scan5_0001.tif"
    image.write_bytes(b"")  # content irrelevant — only the name is parsed

    spec_dir = img_dir
    for _ in range(levels_up):
        spec_dir = spec_dir.parent
    spec = spec_dir / "myscan"          # extensionless (SSRL convention)
    assert spec.suffix == ""
    spec.write_text(_SPEC_MINIMAL)

    md = read_image_metadata(image, meta_format="SPEC")
    # The file was found and parsed: scan 5 / image index 1 per-point counter +
    # the constant (non-scanned) #O/#P motors.  (th is in both tables; motors
    # win the merge, so it's the motor position, hence we check i0 / chi / phi.)
    assert md["i0"] == 110.0    # per-point counter at image index 1
    assert md["chi"] == 5.0     # constant non-scanned motor
    assert md["phi"] == 10.0


def test_read_spec_metadata_ignores_spec_beyond_two_parents(tmp_path: Path) -> None:
    """A SPEC file three or more levels above the image folder is NOT picked up."""
    pytest.importorskip("silx")

    img_dir = tmp_path / "a" / "b" / "c"
    img_dir.mkdir(parents=True)
    image = img_dir / "myscan_scan5_0001.tif"
    image.write_bytes(b"")
    (tmp_path / "myscan").write_text(_SPEC_MINIMAL)  # tmp_path is 3 levels up

    assert read_image_metadata(image, meta_format="SPEC") == {}
