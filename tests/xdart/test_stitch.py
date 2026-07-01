"""Focused tests for the xdart-side stitch gathering (``run_stitch``).

These exercise the LiveScan-specific logic ``run_stitch`` owns — frame-id
alignment of scan_data, normalization, skip-of-frames-without-raw, and the
fail-clearly-on-NaN guards — *without* pyFAI: the actual MultiGeometry
integration (``xrd_tools.integrate.multi.stitch_images``) is
monkeypatched to capture the gathered images/angles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xdart.modules.ewald.stitch import run_stitch


# ---------------------------------------------------------------------------
# Minimal ducks (no pyFAI, no Qt)
# ---------------------------------------------------------------------------

class _DuckPONI:
    dist = 0.1; poni1 = 0.05; poni2 = 0.05; rot1 = 0.0; rot2 = 0.0; rot3 = 0.0


class _DuckFrame:
    def __init__(self, idx, *, raw=True, lazy=False):
        self.idx = idx
        self.poni = _DuckPONI()
        # lazy=True starts with map_raw=None but _lazy_load_raw can hydrate
        # it (simulates a reloaded-from-disk frame).
        self.map_raw = (np.ones((4, 4), dtype=float) * (idx + 1)
                        if raw and not lazy else None)
        self.bg_raw = 0
        self._raw_available = raw

    def _lazy_load_raw(self):
        # A frame with no source on disk stays None (run_stitch skips it).
        if self._raw_available and self.map_raw is None:
            self.map_raw = np.ones((4, 4), dtype=float)


class _DuckGeometry:
    """Returns preset per-frame rotations (radians), ignoring the motors."""

    def __init__(self, rot1_rad, rot2_rad=None, motors=("tth",)):
        self._rot1 = np.asarray(rot1_rad, dtype=float)
        self._rot2 = (np.zeros_like(self._rot1) if rot2_rad is None
                      else np.asarray(rot2_rad, dtype=float))
        self._motors = tuple(motors)

    def all_referenced_motors(self):
        return self._motors

    def derive_per_frame(self, motors):
        return {"rot1": self._rot1, "rot2": self._rot2}


class _DuckScan:
    def __init__(self, frames, geometry, scan_data):
        self.frames = frames
        self.geometry = geometry
        self.scan_data = scan_data
        self.stitched_1d = None
        self.stitched_2d = None
        self.stitch_skipped = None


def _patch_stitch_images(monkeypatch):
    """Replace the ssrl stitch_images with a capture stub; return the bag."""
    captured = {}

    def _fake(images, base_poni, rot1, rot2, **kw):
        captured["images"] = images
        captured["rot1"] = np.asarray(rot1, dtype=float)
        captured["rot2"] = np.asarray(rot2, dtype=float)
        captured["mode"] = kw.get("mode")
        return f"RESULT_{kw.get('mode')}"

    monkeypatch.setattr(
        "xrd_tools.integrate.multi.stitch_images", _fake,
    )
    return captured


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_run_stitch_1d_gathers_all_frames(monkeypatch):
    cap = _patch_stitch_images(monkeypatch)
    n = 3
    frames = [_DuckFrame(i) for i in range(n)]
    geom = _DuckGeometry(np.deg2rad([10.0, 11.0, 12.0]))
    scan_data = pd.DataFrame({"tth": [10.0, 11.0, 12.0]}, index=range(n))
    scan = _DuckScan(frames, geom, scan_data)

    run_stitch(scan, mode="1d")

    assert scan.stitched_1d == "RESULT_1d"
    # All three images handed through as a list (no np.stack copy).
    assert isinstance(cap["images"], list) and len(cap["images"]) == 3
    np.testing.assert_allclose(cap["rot1"], [10.0, 11.0, 12.0], atol=1e-4)


# ---------------------------------------------------------------------------
# Fail-clearly guards (the P2 review items)
# ---------------------------------------------------------------------------

def test_run_stitch_missing_norm_column_raises(monkeypatch):
    _patch_stitch_images(monkeypatch)
    frames = [_DuckFrame(i) for i in range(2)]
    geom = _DuckGeometry(np.deg2rad([10.0, 11.0]))
    scan_data = pd.DataFrame({"tth": [10.0, 11.0]}, index=range(2))
    scan = _DuckScan(frames, geom, scan_data)

    with pytest.raises(ValueError, match="not found in scan_data"):
        run_stitch(scan, mode="1d", norm_motor="i0")


def test_run_stitch_nan_norm_raises_with_frame_ids(monkeypatch):
    _patch_stitch_images(monkeypatch)
    # Frame ids [0, 1, 2] but scan_data only has rows for 0 and 2 → after
    # reindex to frame ids, frame 1's monitor is NaN.
    frames = [_DuckFrame(i) for i in range(3)]
    geom = _DuckGeometry(np.deg2rad([10.0, 11.0, 12.0]))
    scan_data = pd.DataFrame(
        {"tth": [10.0, 12.0], "i0": [1e6, 1.2e6]}, index=[0, 2],
    )
    scan = _DuckScan(frames, geom, scan_data)

    with pytest.raises(ValueError, match=r"NaN/inf for frame\(s\) \[1\]"):
        run_stitch(scan, mode="1d", norm_motor="i0")


def test_run_stitch_nan_rotation_raises_with_frame_ids(monkeypatch):
    _patch_stitch_images(monkeypatch)
    frames = [_DuckFrame(i) for i in range(3)]
    # Frame 1 gets a NaN derived rotation (e.g. its angle motor was missing).
    geom = _DuckGeometry(np.array([np.deg2rad(10.0), np.nan, np.deg2rad(12.0)]))
    scan_data = pd.DataFrame({"tth": [10.0, 11.0, 12.0]}, index=range(3))
    scan = _DuckScan(frames, geom, scan_data)

    with pytest.raises(ValueError, match=r"NaN detector rotation for frame\(s\) \[1\]"):
        run_stitch(scan, mode="1d")


# ---------------------------------------------------------------------------
# Skip-frames-without-raw + geometry alignment
# ---------------------------------------------------------------------------

def test_run_stitch_skips_frames_without_raw_and_aligns_geometry(monkeypatch):
    cap = _patch_stitch_images(monkeypatch)
    # Frame 1 has no raw data and can't be lazy-loaded → skipped.  The
    # surviving images (0, 2) must pair with rotations (10, 12), NOT the
    # full (10, 11, 12) — otherwise pyFAI would mis-pair angle to image.
    f0, f1, f2 = _DuckFrame(0), _DuckFrame(1, raw=False), _DuckFrame(2)
    geom = _DuckGeometry(np.deg2rad([10.0, 11.0, 12.0]))
    scan_data = pd.DataFrame({"tth": [10.0, 11.0, 12.0]}, index=range(3))
    scan = _DuckScan([f0, f1, f2], geom, scan_data)

    run_stitch(scan, mode="1d")

    assert len(cap["images"]) == 2
    np.testing.assert_allclose(cap["rot1"], [10.0, 12.0], atol=1e-4)
    # The partial skip is recorded on the scan so the GUI can warn the merge is a
    # subset (frame 1 had no raw data); the surviving frames still stitched.
    assert scan.stitch_skipped == [1]


# ---------------------------------------------------------------------------
# Memory hygiene + scan_data alignment
# ---------------------------------------------------------------------------

def test_run_stitch_clears_lazy_loaded_raw_keeps_resident(monkeypatch):
    """Raw arrays we hydrate for the stitch (incl. frames[0], which the
    memory-guard probe loads first) are released afterward; frames the
    wrangler already had resident are left alone."""
    _patch_stitch_images(monkeypatch)
    f0 = _DuckFrame(0, lazy=True)   # loaded by the memory-guard probe
    f1 = _DuckFrame(1, lazy=True)   # loaded in the gather loop
    f2 = _DuckFrame(2)              # already resident (wrangler hand-off)
    geom = _DuckGeometry(np.deg2rad([10.0, 11.0, 12.0]))
    scan_data = pd.DataFrame({"tth": [10.0, 11.0, 12.0]}, index=range(3))
    scan = _DuckScan([f0, f1, f2], geom, scan_data)

    run_stitch(scan, mode="1d")

    assert f0.map_raw is None and f1.map_raw is None   # freed
    assert f2.map_raw is not None                      # retained


def test_run_stitch_clears_lazy_raw_even_on_error(monkeypatch):
    """The lazy-raw cleanup runs in a finally, so a guard that raises mid-
    stitch still releases the arrays."""
    _patch_stitch_images(monkeypatch)
    f0 = _DuckFrame(0, lazy=True)
    geom = _DuckGeometry(np.array([np.nan, np.deg2rad(11.0)]))  # NaN rot → raise
    scan_data = pd.DataFrame({"tth": [10.0, 11.0]}, index=range(2))
    scan = _DuckScan([f0, _DuckFrame(1)], geom, scan_data)

    with pytest.raises(ValueError):
        run_stitch(scan, mode="1d")
    assert f0.map_raw is None


def test_run_stitch_duplicate_scan_data_index_raises(monkeypatch):
    """A non-unique scan_data index can't be reindexed to frame ids;
    rather than silently fall back to positional alignment, raise."""
    _patch_stitch_images(monkeypatch)
    frames = [_DuckFrame(0), _DuckFrame(1)]
    geom = _DuckGeometry(np.deg2rad([10.0, 11.0]))
    scan_data = pd.DataFrame({"tth": [10.0, 11.0]}, index=[5, 5])  # duplicate
    scan = _DuckScan(frames, geom, scan_data)

    with pytest.raises(ValueError, match="cannot align scan_data"):
        run_stitch(scan, mode="1d")


# ---------------------------------------------------------------------------
# GUI stitch worker (stitchThread) — Phase 1a
# ---------------------------------------------------------------------------

def _ensure_qapp():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from pyqtgraph.Qt import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_stitch_thread_runs_and_flags_ok(monkeypatch):
    """stitchThread.run() drives run_stitch on its scan and sets ok=True on
    success (the GUI's stitch_thread_finished keys the 'complete' message on it)."""
    _ensure_qapp()
    _patch_stitch_images(monkeypatch)
    from xdart.gui.tabs.static_scan.scan_threads import stitchThread
    frames = [_DuckFrame(i) for i in range(3)]
    geom = _DuckGeometry(np.deg2rad([10.0, 11.0, 12.0]))
    scan_data = pd.DataFrame({"tth": [10.0, 11.0, 12.0]}, index=range(3))
    scan = _DuckScan(frames, geom, scan_data)
    th = stitchThread(scan)
    th.mode = "1d"
    th.params = {"unit": "q_A^-1", "method": "BBox", "npt_1d": 500}
    th.run()                       # synchronous — no event loop needed
    assert th.ok is True
    assert scan.stitched_1d == "RESULT_1d"


def test_stitch_thread_error_routes_to_signal_and_clears_ok():
    """A failing reduction (no geometry → run_stitch raises) is caught: ok stays
    False and the message is emitted on errorSig (the GUI surfaces it)."""
    _ensure_qapp()
    from xdart.gui.tabs.static_scan.scan_threads import stitchThread
    scan = _DuckScan([_DuckFrame(0)], None, pd.DataFrame())   # geometry=None
    th = stitchThread(scan)
    th.mode = "1d"
    errs = []
    th.errorSig.connect(lambda m: errs.append(m))
    th.run()
    assert th.ok is False
    assert errs and "geometry" in errs[0].lower()
