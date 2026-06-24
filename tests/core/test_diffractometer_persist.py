"""Persistence of the canonical Diffractometer (design §4 / §6 step 3).

A reloaded scan must reconstruct the full instrument geometry for offline
stitch/RSM — so the canonical object round-trips through a capability-gated
``/entry/diffractometer`` group, additively (absent group → None, no crash).
"""
from __future__ import annotations

from pathlib import Path

import h5py

import numpy as np

from xrd_tools.core.geometry import (
    Diffractometer,
    DiffractometerGeometry,
    ImageOrientation,
)
from xrd_tools.io.nexus import write_diffractometer, write_per_frame_geometry
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


# ---------------------------------------------------------------------------
# step-4 seam: a Diffractometer is a drop-in for the duck-typed Scan.geometry
# ---------------------------------------------------------------------------

def _read_geom_group(path: Path) -> dict:
    out = {}
    with h5py.File(path, "r") as f:
        g = f["entry/per_frame_geometry"]
        for k in ("rot1", "rot2", "rot3", "incident_angle"):
            out[k] = np.asarray(g[k])
    return out


def test_diffractometer_is_writer_dropin(tmp_path):
    """write_per_frame_geometry (the sole Scan.geometry consumer) must produce
    byte-identical output from a Diffractometer and the legacy class."""
    pd = __import__("pandas")
    scan_data = pd.DataFrame({
        "nu": [2.0, 4.0, 6.0], "del": [15.0, 30.0, 45.0],
        "eta": [0.5, 0.5, 0.5],
    })
    frames = [0, 1, 2]

    def _write(geometry):
        p = tmp_path / f"{type(geometry).__name__}.nxs"
        with h5py.File(p, "w") as f:
            write_per_frame_geometry(f.create_group("entry"), scan_data,
                                     frames, geometry)
        return _read_geom_group(p)

    new = _write(Diffractometer.psic())
    old = _write(DiffractometerGeometry.psic())
    for k in old:
        np.testing.assert_array_equal(new[k], old[k])
