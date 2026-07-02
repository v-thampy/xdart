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


def _fake_scan():
    calls = []
    scan = SimpleNamespace(
        data_file="old.nxs",
        name="old",
        skip_2d=False,
        # The lazy frame series carries its OWN data_file (captured at build).
        frames=SimpleNamespace(data_file="old.nxs"),
    )
    scan.set_datafile = lambda fname: calls.append(fname)
    return scan, calls


def _file_thread(live_run):
    scan, calls = _fake_scan()
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
    # Crash regression (2026-06-18): the lazy frame series' data_file must be
    # repointed too — else a post-run Reintegrate that falls to disk for an
    # evicted frame opens the stale init-time default.nxs (FileNotFoundError).
    assert scan.frames.data_file == "/data/scan_42.nxs"


def test_set_datafile_non_live_reloads_from_disk():
    """Outside a live run (batch / viewer / end-of-run auto-load) the
    full reload still runs so frames come back from the finished file."""
    thread, scan, calls = _file_thread(live_run=False)

    thread.set_datafile()

    assert calls == ["/data/scan_42.nxs"]  # scan.set_datafile was called


def test_set_datafile_no_nxs_repoints_without_load():
    """Int 1D (XYE) sets ``no_nxs`` because it never writes a .nxs.  Even in a
    non-live (batch) run, set_datafile must then repoint only — never call
    scan.set_datafile (which would try to load/create the absent .nxs)."""
    thread, scan, calls = _file_thread(live_run=False)
    thread.no_nxs = True

    thread.set_datafile()

    assert calls == []                       # no load/create attempted
    assert scan.data_file == "/data/scan_42.nxs"
    assert scan.name == "scan_42"
    assert scan.frames.data_file == "/data/scan_42.nxs"   # frame series repointed too


def _reset_viewer(live_run_active):
    from xdart.modules.frame_publication import PublicationStore

    viewer = SimpleNamespace(
        live_run_active=live_run_active,
        scan=SimpleNamespace(data_file="scan.nxs"),
        _h5pool=SimpleNamespace(closed=[]),
        frames=SimpleNamespace(cleared=False),
        frame_ids=SimpleNamespace(cleared=False),
        data_1d={1: "a", 2: "b"},
        data_2d={1: "x"},
        publication_store=PublicationStore(),
        data_lock=_NullLock(),
        new_scan=False,
        cancel_calls=0,
    )
    viewer._h5pool.close = lambda f: viewer._h5pool.closed.append(f)
    viewer.frames.clear = lambda: setattr(viewer.frames, "cleared", True)
    viewer.frame_ids.clear = lambda: setattr(viewer.frame_ids, "cleared", True)
    viewer.cancel_pending_loads = lambda: setattr(
        viewer, "cancel_calls", viewer.cancel_calls + 1)
    viewer.data_reset = MethodType(H5Viewer.data_reset, viewer)
    return viewer


def test_data_reset_suppressed_during_live_run():
    """data_reset must be a no-op while a live run is active — the live
    display's per-frame caches must survive the async sigNewFile."""
    viewer = _reset_viewer(live_run_active=True)

    viewer.data_reset()

    assert viewer.data_1d == {1: "a", 2: "b"}
    assert viewer.data_2d == {1: "x"}
    assert viewer.publication_store.generation == 0
    assert viewer.frames.cleared is False
    assert viewer.frame_ids.cleared is False
    assert viewer._h5pool.closed == []
    assert viewer.cancel_calls == 0


def test_data_reset_clears_when_not_live():
    """Outside a live run data_reset still wipes everything (manual file
    open / end-of-run reload)."""
    viewer = _reset_viewer(live_run_active=False)

    viewer.data_reset()

    assert viewer.data_1d == {}
    assert viewer.data_2d == {}
    assert viewer.publication_store.generation == 1
    assert viewer.frames.cleared is True
    assert viewer.frame_ids.cleared is True
    assert viewer._h5pool.closed == ["scan.nxs"]
    assert viewer.new_scan is True
    assert viewer.cancel_calls == 1


# ── Frame-click freeze guard (path #1: data_changed) ───────────────────────
# Clicking an evicted frame in the Frames list during a run used to call
# load_frames_data -> _teardown_load_worker.thread.wait(2000) on the GUI thread,
# re-fired by each writer save's sigUpdate -> multi-minute beachball.  The
# disk-load branch is now gated on _run_writing (the run-state owner's single
# flag, parallel to the displayframe's _processing_active reader-side guard).

def _frame_select_viewer(run_writing=False, selected=('5',), cached_2d=()):
    """Minimal H5Viewer stand-in exercising data_changed's normal/HDF5 branch."""
    calls = {'load': [], 'sig': 0, 'cancel': 0}
    items = [SimpleNamespace(text=(lambda s=s: s)) for s in selected]
    viewer = SimpleNamespace(
        _run_writing=run_writing,
        viewer_mode=None,                       # normal/integration mode
        frame_ids=[],
        update_2d=True,
        data_1d={},
        data_2d={int(i): 'x' for i in cached_2d},
        scan=SimpleNamespace(frames=SimpleNamespace(index=list(range(100)))),
        ui=SimpleNamespace(listData=SimpleNamespace(selectedItems=lambda: items)),
        sigUpdate=SimpleNamespace(emit=lambda: calls.__setitem__('sig', calls['sig'] + 1)),
    )
    viewer.load_frames_data = lambda ids, l2d: calls['load'].append((list(ids), l2d))
    viewer.cancel_pending_loads = lambda: calls.__setitem__('cancel', calls['cancel'] + 1)
    # data_changed now routes its terminal render through the 100 ms debounce
    # Coalescer (freeze fix); simulate the timer firing synchronously so the
    # sigUpdate-count assertions still hold (a burst debounces to one emit).
    viewer._update_coalesce_timer = SimpleNamespace(
        start=lambda: viewer.sigUpdate.emit())
    # data_changed also debounces the blocking DISK LOAD (freeze fix part 2);
    # simulate the load timer firing synchronously so the load assertions hold.
    viewer._pending_load_ids = None
    viewer._pending_load_2d = True
    viewer._flush_pending_load = MethodType(H5Viewer._flush_pending_load, viewer)
    viewer._load_coalesce_timer = SimpleNamespace(
        start=lambda: viewer._flush_pending_load())
    viewer.data_changed = MethodType(H5Viewer.data_changed, viewer)
    viewer.set_run_writing = MethodType(H5Viewer.set_run_writing, viewer)
    return viewer, calls


def test_frame_click_during_run_serves_cache_no_disk_load():
    """Path #1: an evicted-frame click during a run must NOT call
    load_frames_data (the GUI-thread thread.wait(2000) + sigUpdate churn)."""
    viewer, calls = _frame_select_viewer(run_writing=True, selected=('5',))
    viewer.data_changed()
    assert calls['load'] == []          # no disk load while the writer is active
    assert calls['sig'] == 1            # selection still refreshes from cache


def test_frame_click_when_idle_loads_from_disk():
    """Outside a run the same evicted selection loads normally (guard is
    directional — it must not suppress idle / post-run loads)."""
    viewer, calls = _frame_select_viewer(run_writing=False, selected=('5',))
    viewer.data_changed()
    assert calls['load'] == [([5], True)]
    assert calls['sig'] == 1


def test_set_run_writing_cancels_on_start_reloads_on_end():
    """set_run_writing is the single switch the run-state owner drives: cancel
    any in-flight load on the rising edge; re-fire the standing selection on the
    falling edge so a frame skipped during the run loads once the file is idle."""
    viewer, calls = _frame_select_viewer(run_writing=False, selected=('5',))
    viewer.set_run_writing(True)
    assert viewer._run_writing is True
    assert calls['cancel'] == 1         # stale worker cancelled at run start
    assert calls['load'] == []          # nothing loaded while writing
    viewer.set_run_writing(False)
    assert viewer._run_writing is False
    assert calls['load'] == [([5], True)]   # falling edge re-loaded the selection


def test_set_run_writing_no_reload_without_selection():
    """No standing selection -> the falling edge is a quiet no-op."""
    viewer, calls = _frame_select_viewer(run_writing=True, selected=())
    viewer.set_run_writing(False)
    assert calls['load'] == []


class _ScanItem:
    def __init__(self, text):
        self._text = text

    def data(self, _role):
        return self._text


def _scan_click_viewer(*, current_file="/data/old.nxs", run_writing=False,
                       pending=False):
    calls = []
    viewer = SimpleNamespace(
        _suspend_scan_selection_loads=False,
        viewer_mode=None,
        dirname="/data",
        _browser_scan_reset_pending=pending,
        _run_writing=run_writing,
        new_scan_loaded=False,
        file_thread=SimpleNamespace(fname=current_file),
        set_file=lambda fpath: calls.append(fpath),
    )
    viewer.scans_clicked = MethodType(H5Viewer.scans_clicked, viewer)
    return viewer, calls


def test_manual_scan_select_arms_deferred_reset_only_for_real_file_load():
    viewer, calls = _scan_click_viewer(current_file="/data/old.nxs")

    viewer.scans_clicked(_ScanItem("new.nxs"))

    assert calls == ["/data/new.nxs"]
    assert viewer._browser_scan_reset_pending is True
    assert viewer.new_scan_loaded is True


def test_same_file_browser_click_does_not_arm_deferred_reset():
    viewer, calls = _scan_click_viewer(current_file="/data/scan.nxs")

    viewer.scans_clicked(_ScanItem("scan.nxs"))

    assert calls == ["/data/scan.nxs"]
    assert viewer._browser_scan_reset_pending is False
    assert viewer.new_scan_loaded is False


def test_same_file_second_click_preserves_existing_deferred_reset():
    # A genuine first click can be followed by Qt's release-side duplicate
    # same-file signal after set_file has already repointed file_thread.fname.
    # That duplicate must not clear the valid pending reset.
    viewer, calls = _scan_click_viewer(
        current_file="/data/scan.nxs", pending=True,
    )

    viewer.scans_clicked(_ScanItem("scan.nxs"))

    assert calls == ["/data/scan.nxs"]
    assert viewer._browser_scan_reset_pending is True


def test_run_guarded_browser_click_does_not_arm_deferred_reset():
    viewer, calls = _scan_click_viewer(
        current_file="/data/old.nxs", run_writing=True,
    )

    viewer.scans_clicked(_ScanItem("new.nxs"))

    assert calls == ["/data/new.nxs"]
    assert viewer._browser_scan_reset_pending is False
    assert viewer.new_scan_loaded is False
