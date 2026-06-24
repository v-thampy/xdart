"""Persistence of the canonical Diffractometer (design §4 / §6 step 3).

A reloaded scan must reconstruct the full instrument geometry for offline
stitch/RSM — so the canonical object round-trips through a capability-gated
``/entry/diffractometer`` group, additively (absent group → None, no crash).
"""
from __future__ import annotations

from pathlib import Path

import h5py

from xrd_tools.core.geometry import (
    Diffractometer,
    ImageOrientation,
)
from xrd_tools.io.nexus import write_diffractometer
from xrd_tools.io.read import ProcessedScan, get_diffractometer
from xrd_tools.io.schema import CAPABILITIES, detect_capabilities

_FIXTURES = Path(__file__).parent / "fixtures"
_GONIO_V1 = _FIXTURES / "gonio_robl_v1.json"


def _calibrated_diff() -> Diffractometer:
    """A complete fitted Diffractometer (both views + calibration + mount)."""
    return Diffractometer.from_pyfai_goniometer(
        _GONIO_V1, source_motors="del", base=Diffractometer.psic(),
        image_orientation=ImageOrientation(rotation=180))


def _write_entry(path: Path, diff) -> None:
    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        write_diffractometer(entry, diff)


# ---------------------------------------------------------------------------
# write -> read round-trip (the step-3 gate)
# ---------------------------------------------------------------------------

def test_write_read_roundtrip(tmp_path):
    diff = _calibrated_diff()
    p = tmp_path / "scan.nxs"
    _write_entry(p, diff)
    back = get_diffractometer(p)
    assert back == diff
    # the full object survived: both views + calibration + mount + preset
    assert back.preset == "fitted"
    assert back.sample_circles == ("x+", "z-", "y+", "z-")
    assert back.calibration.detector_config.get("orientation") == 3
    assert back.calibration.image_orientation.rotation == 180
    assert back.rot2.is_active


def test_preset_diffractometer_roundtrip(tmp_path):
    diff = Diffractometer.psic()
    p = tmp_path / "scan.nxs"
    _write_entry(p, diff)
    assert get_diffractometer(p) == diff


# ---------------------------------------------------------------------------
# back-compat: absent group / file → None (never raise, never synthesize)
# ---------------------------------------------------------------------------

def test_absent_group_returns_none(tmp_path):
    p = tmp_path / "old.nxs"
    with h5py.File(p, "w") as f:
        f.create_group("entry")  # no diffractometer group (a pre-existing file)
    assert get_diffractometer(p) is None


def test_absent_entry_returns_none(tmp_path):
    p = tmp_path / "weird.nxs"
    with h5py.File(p, "w") as f:
        f.create_group("something_else")
    assert get_diffractometer(p) is None


def test_write_none_clears_group(tmp_path):
    p = tmp_path / "scan.nxs"
    _write_entry(p, _calibrated_diff())
    assert get_diffractometer(p) is not None
    # rewriting with None clears it (e.g. geometry removed)
    with h5py.File(p, "a") as f:
        write_diffractometer(f["entry"], None)
    assert get_diffractometer(p) is None


# ---------------------------------------------------------------------------
# capability feature-detection (ADR-0002, presence-detected)
# ---------------------------------------------------------------------------

def test_capability_registered():
    cap = CAPABILITIES["diffractometer"]
    assert cap.marker == "diffractometer"
    assert cap.kind == "group"
    assert cap.introduced <= 2  # additive at v2, no schema bump

def test_capability_detected_after_write(tmp_path):
    p = tmp_path / "scan.nxs"
    _write_entry(p, _calibrated_diff())
    with h5py.File(p, "r") as f:
        caps = detect_capabilities(f["entry"])
    assert "diffractometer" in caps

def test_capability_absent_on_old_file(tmp_path):
    p = tmp_path / "old.nxs"
    with h5py.File(p, "w") as f:
        f.create_group("entry")
        caps = detect_capabilities(f["entry"])
    assert "diffractometer" not in caps


# ---------------------------------------------------------------------------
# ProcessedScan.diffractometer property (the offline-stitch/RSM handle)
# ---------------------------------------------------------------------------

def test_processed_scan_property(tmp_path):
    diff = _calibrated_diff()
    p = tmp_path / "scan.nxs"
    _write_entry(p, diff)
    scan = ProcessedScan(p)
    assert scan.diffractometer == diff
    # cached (the flag means a second read does not re-open)
    assert scan.diffractometer is scan.diffractometer

def test_processed_scan_property_none_on_old_file(tmp_path):
    p = tmp_path / "old.nxs"
    with h5py.File(p, "w") as f:
        f.create_group("entry")
    assert ProcessedScan(p).diffractometer is None

def test_processed_scan_refresh_resets(tmp_path):
    p = tmp_path / "scan.nxs"
    with h5py.File(p, "w") as f:
        f.create_group("entry")
    scan = ProcessedScan(p)
    assert scan.diffractometer is None
    # write a geometry, then refresh -> the property re-reads
    with h5py.File(p, "a") as f:
        write_diffractometer(f["entry"], Diffractometer.psic())
    scan.refresh_metadata()
    assert scan.diffractometer == Diffractometer.psic()
