"""BL-3 — auto-sidecar discovery must not latch a junk companion.

PRODUCTION-WIRED (per the blocker-wave systemic corrective): real files on disk
through the real ``read_image_metadata`` — no monkeypatched sidecar seam.  The
review's failure: with ``auto`` now the default, a per-frame ``.poni`` (colon
pairs) / ``img.tif.log`` (JSON) / oversize / binary companion could sort first
alphabetically and latch as "the metadata", poisoning scan_data for every frame.
"""

import logging

import pytest

import xrd_tools.io.metadata as metadata
from xrd_tools.io.metadata import read_image_metadata


@pytest.fixture(autouse=True)
def _reset_auto_cache():
    metadata._AUTO_SIDECAR_CACHE.clear()
    metadata._DIR_LISTING_CACHE.clear()
    yield
    metadata._AUTO_SIDECAR_CACHE.clear()
    metadata._DIR_LISTING_CACHE.clear()


def _image(tmp_path):
    img = tmp_path / "scan_0001.tif"
    img.touch()
    return img


def test_real_metadata_wins_over_poni_and_log_decoys(tmp_path):
    img = _image(tmp_path)
    # Decoys the OLD alphabetical scan latched (both parse to >= 3 pairs):
    (tmp_path / "scan_0001.tif.log").write_text('{"a": 1, "b": 2, "c": 3}\n')
    (tmp_path / "scan_0001.tif.poni").write_text(
        "Detector: Eiger\nDistance: 0.15\nPoni1: 0.05\nWavelength: 1.0e-10\n")
    # The REAL metadata sidecar:
    (tmp_path / "scan_0001.tif.metadata").write_text(
        "i0=1234\nmotor_th=2.0\nchi=90\n")

    result = read_image_metadata(img, meta_format="auto")

    assert result == {"i0": 1234, "motor_th": 2.0, "chi": 90}  # .metadata, not poni/log


def test_poni_only_companion_does_not_latch(tmp_path):
    img = _image(tmp_path)
    (tmp_path / "scan_0001.tif.poni").write_text(
        "Detector: Eiger\nDistance: 0.15\nPoni1: 0.05\n")  # >= 3 colon pairs

    assert read_image_metadata(img, meta_format="auto") == {}  # excluded by allow-list


def test_generic_name_value_txt_reads_via_ssrl_fallback(tmp_path):
    img = _image(tmp_path)
    # A generic name=value .txt, NOT the SSRL "# Counters/# Motors" format:
    (tmp_path / "scan_0001.tif.txt").write_text("i0=42\nmotor=1.5\ntemp=300\n")

    result = read_image_metadata(img, meta_format="auto")

    assert result == {"i0": 42, "motor": 1.5, "temp": 300}


def test_binary_companion_is_rejected(tmp_path):
    img = _image(tmp_path)
    (tmp_path / "scan_0001.tif.metadata").write_bytes(b"\x00\x01\x02\xff" * 64)

    assert read_image_metadata(img, meta_format="auto") == {}  # binary, not garbage pairs


def test_oversize_sidecar_is_skipped(tmp_path):
    img = _image(tmp_path)
    (tmp_path / "scan_0001.tif.metadata").write_text(
        "".join(f"k{i}=v{i}\n" for i in range(300_000)))  # > 1 MiB

    assert read_image_metadata(img, meta_format="auto") == {}


def test_ranking_is_deterministic_txt_before_metadata(tmp_path):
    img = _image(tmp_path)
    (tmp_path / "scan_0001.tif.txt").write_text("i0=1\nth=2\nchi=3\n")
    (tmp_path / "scan_0001.tif.metadata").write_text("other=9\nx=8\ny=7\n")

    result = read_image_metadata(img, meta_format="auto")

    assert result == {"i0": 1, "th": 2, "chi": 3}  # txt ranks first in the allow-list


def test_auto_locked_convention_is_logged(tmp_path, caplog):
    img = _image(tmp_path)
    (tmp_path / "scan_0001.tif.metadata").write_text("i0=1\nth=2\nchi=3\n")

    with caplog.at_level(logging.INFO, logger="xrd_tools.io.metadata"):
        read_image_metadata(img, meta_format="auto")

    assert any("auto locked onto" in r.getMessage() for r in caplog.records)


def test_explicit_metadata_format_still_bypasses_allow_list_and_threshold(tmp_path):
    # BL-3: an explicit format is a deliberate choice — a 1-field sidecar with a
    # non-allow-listed extension still loads (allow-list is AUTO-only).
    img = _image(tmp_path)
    (tmp_path / "scan_0001.custommeta").write_text("only_one=5\n")

    result = read_image_metadata(img, meta_format="custommeta")

    assert result == {"only_one": 5}


def test_auto_discovery_lists_directory_at_most_once_s20(tmp_path, monkeypatch):
    # S-20: the exact-case is_file() fast path finds a correctly-cased sidecar
    # WITHOUT re-listing the directory per candidate.  The first BL-3 rewrite did
    # ~16 iterdir/frame (2 per _existing_path_case_insensitive x 8 candidates) --
    # worse than the single iterdir it replaced.  Now <= 1 (the misses before the
    # hit share one mtime-cached listing; a first-extension hit needs 0).
    import pathlib

    _image(tmp_path)
    (tmp_path / "scan_0001.tif.metadata").write_text("i0=1\nth=2\nchi=3\n")

    calls = {"n": 0}
    real = pathlib.Path.iterdir

    def counting(self):
        calls["n"] += 1
        return real(self)

    monkeypatch.setattr(pathlib.Path, "iterdir", counting)
    result = read_image_metadata(tmp_path / "scan_0001.tif", meta_format="auto")

    assert result == {"i0": 1, "th": 2, "chi": 3}
    assert calls["n"] <= 1, f"expected <=1 directory listing, got {calls['n']}"


def test_junk_pdi_is_rejected_not_fabricated_bw(tmp_path):
    # BW / .pdi junk-latch: a junk .pdi (no recognizable section) must return {}
    # -- the old last-resort fabricated {TwoTheta:0.0, Theta:0.0}, so a junk .pdi
    # latched as valid metadata and bypassed every BL-3 gate.
    from xrd_tools.io.metadata import read_pdi_metadata

    (tmp_path / "junk.pdi").write_text("this is not a pdi file at all\n")
    assert read_pdi_metadata(tmp_path / "junk.pdi") == {}

    # and auto discovery does not latch a junk .pdi companion
    img = _image(tmp_path)
    (tmp_path / "scan_0001.tif.pdi").write_text("garbage;;not;a;pdi;file\n")
    assert read_image_metadata(img, meta_format="auto") == {}
