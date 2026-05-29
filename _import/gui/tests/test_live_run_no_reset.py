"""Regression tests for the live-run guards that keep multi-scan Eiger
(Image Directory, non-batch) plots updating per frame.

Two destructive paths used to fire on the async file-thread a few ms
after ``new_scan`` and wipe the in-memory per-frame state the live
display depends on:

* ``fileHandlerThread.set_datafile`` reloaded the scan from disk,
  replacing ``scan.frames`` with the (lagging) on-disk index.
* ``H5Viewer.data_reset`` (wired to the async ``sigNewFile``) cleared
  ``data_1d`` / ``data_2d`` / ``frames``.

Both are now gated by a live-run flag set for the duration of a
non-batch wrangler run.  These tests pin that contract.
"""

from __future__ import annotations

import os
from types import MethodType, SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from xdart.gui.tabs.static_scan.h5viewer import H5Viewer
from xdart.gui.tabs.static_scan.scan_threads import fileHandlerThread


class _NullLock:
    """Minimal context manager standing in for the shared file lock."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_sphere():
    calls = []
    scan = SimpleNamespace(
        data_file="old.nxs",
        name="old",
        skip_2d=False,
    )
    scan.set_datafile = lambda fname: calls.append(fname)
    return scan, calls


def _file_thread(live_run):
    scan, calls = _fake_sphere()
    thread = SimpleNamespace(
        file_lock=_NullLock(),
        fname="/data/scan_42.nxs",
        live_run=live_run,
        scan=scan,
        sigNewFile=SimpleNamespace(emit=lambda *a: None),
        sigUpdate=SimpleNamespace(emit=lambda *a: None),
    )
    thread.set_datafile = MethodType(fileHandlerThread.set_datafile, thread)
    return thread, scan, calls


def test_set_datafile_live_run_repoints_without_reload():
    """In a live run, set_datafile must NOT call scan.set_datafile
    (which reloads frames from the lagging on-disk index) — it only
    repoints data_file + name."""
    thread, scan, calls = _file_thread(live_run=True)

    thread.set_datafile()

    assert calls == []  # no disk reload
    assert scan.data_file == "/data/scan_42.nxs"
    assert scan.name == "scan_42"


def test_set_datafile_non_live_reloads_from_disk():
    """Outside a live run (batch / viewer / end-of-run auto-load) the
    full reload still runs so frames come back from the finished file."""
    thread, scan, calls = _file_thread(live_run=False)

    thread.set_datafile()

    assert calls == ["/data/scan_42.nxs"]  # scan.set_datafile was called


def _reset_viewer(live_run_active):
    viewer = SimpleNamespace(
        live_run_active=live_run_active,
        scan=SimpleNamespace(data_file="scan.nxs"),
        _h5pool=SimpleNamespace(closed=[]),
        frames=SimpleNamespace(cleared=False),
        frame_ids=SimpleNamespace(cleared=False),
        data_1d={1: "a", 2: "b"},
        data_2d={1: "x"},
        data_lock=_NullLock(),
        new_scan=False,
    )
    viewer._h5pool.close = lambda f: viewer._h5pool.closed.append(f)
    viewer.frames.clear = lambda: setattr(viewer.frames, "cleared", True)
    viewer.frame_ids.clear = lambda: setattr(viewer.frame_ids, "cleared", True)
    viewer.data_reset = MethodType(H5Viewer.data_reset, viewer)
    return viewer


def test_data_reset_suppressed_during_live_run():
    """data_reset must be a no-op while a live run is active — the live
    display's per-frame caches must survive the async sigNewFile."""
    viewer = _reset_viewer(live_run_active=True)

    viewer.data_reset()

    assert viewer.data_1d == {1: "a", 2: "b"}
    assert viewer.data_2d == {1: "x"}
    assert viewer.frames.cleared is False
    assert viewer.frame_ids.cleared is False
    assert viewer._h5pool.closed == []


def test_data_reset_clears_when_not_live():
    """Outside a live run data_reset still wipes everything (manual file
    open / end-of-run reload)."""
    viewer = _reset_viewer(live_run_active=False)

    viewer.data_reset()

    assert viewer.data_1d == {}
    assert viewer.data_2d == {}
    assert viewer.frames.cleared is True
    assert viewer.frame_ids.cleared is True
    assert viewer._h5pool.closed == ["scan.nxs"]
    assert viewer.new_scan is True
