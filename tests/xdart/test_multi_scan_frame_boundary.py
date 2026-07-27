# -*- coding: utf-8 -*-
"""Frame-driven scan-boundary acceptance tests.

In a multi-scan Image Directory run the wrangler's ``new_scan`` signal races the
frame stream (it is emitted from the wrangler run thread while per-frame
``sigUpdate`` comes from the sink writer thread — no Qt cross-thread ordering —
plus Eiger prefetch read-ahead), so ``new_scan(scanB)`` can arrive AFTER scanB's
frames 1..N already landed.  The panel boundary is therefore FRAME-DRIVEN:
``update_data`` derives each frame's scan identity from its ``source_file`` and
rescopes BEFORE appending, so the triggering frame becomes #1 of the new scan and
its own frames can never be dropped.  A late ``new_scan`` is identity-guarded and
becomes a no-op for the destructive clears.

These are the acceptance gate for the fix: the reverted signal-keyed fix (which
cleared on the ``new_scan`` signal) would FAIL ``test_frame_driven_boundary_
survives_late_new_scan`` — it wiped scanB frames 1..8 and restarted at 9.
"""
from threading import RLock
from types import SimpleNamespace, MethodType

import pandas as pd

from tests.xdart._accepted_run import admitted_worker, accepted_run  # noqa: E402
from xdart.gui.tabs.static_scan.display_logic import LifecycleCause
from xdart.gui.tabs.static_scan.static_scan_widget import (
    staticWidget, _scan_key_from_source,
)
from xdart.modules.frame_publication import PublicationStore


def _boundary_worker(*, batch_mode):
    """A worker stub that publishes a carrier AND its admission ledger.

    O-1b: the host reads run policy from the accepted object and qualifies it by
    identity (review §45.3), so a rig must admit the same object it publishes.
    """
    return admitted_worker(
        SimpleNamespace(
            _published_frames={}, mask=None, detector_shape=None),
        batch_mode=batch_mode)


def _boundary_host():
    pin_clears = []          # S-18 pin-only drops (render-path compat clears)
    scan = SimpleNamespace(
        name="scanA", gi=False, incidence_motor="th", single_img=False,
        series_average=False, global_mask=None, scan_lock=RLock(),
        frames=SimpleNamespace(index=[], _in_memory={}),
        scan_data=pd.DataFrame(),
    )
    host = SimpleNamespace(
        scan=scan, _run_saw_frame=False, _pending_frames={}, _scan_info_rows={},
        frames={}, frame_ids=[], publication_store=PublicationStore(),
        data_lock=RLock(), viewer_rows_1d={}, viewer_rows_2d={},
        h5viewer=SimpleNamespace(
            dirname="", live_run_active=False, scan_name="scanA",
            auto_last=False, latest_idx=0,
            set_file=lambda fname, **k: None, update_scans=lambda: None,
            update=lambda: None),
        wrangler=SimpleNamespace(thread=_boundary_worker(batch_mode=False)),
        integratorTree=SimpleNamespace(
            get_args=lambda n: None, set_image_units=lambda: None),
        _update_timer=SimpleNamespace(stop=lambda: None, trigger=lambda: None),
        _list_timer=SimpleNamespace(stop=lambda: None, trigger=lambda: None),
        _flush_pending_update=lambda: None,
        displayframe=SimpleNamespace(
            set_axes=lambda: None, clear_overlay=lambda *a, **k: None,
            idxs=[], idxs_1d=[], idxs_2d=[],
            _raw_resolve_failed=set(), _raw_full_shape=None,
            stitch_display_mode=None,
            _waterfall_history=None,   # S-14: tests set .ids to seed the accumulator
            _accumulator_lifecycle_log=[],   # V2 owner log (S-14 observation)
            _clear_pinned_slice_cuts=lambda **k: pin_clears.append(k)),
        _controls_v2_enabled=lambda: False,
        _refresh_controls_v2_profile=lambda *a, **k: None,
        _fit_controls_height=lambda *a, **k: None,
        metawidget=SimpleNamespace(update=lambda: None),
    )
    for m in ("update_data", "new_scan", "_rescope_frame_panel_to",
              "_sync_h5viewer_save_dir", "_seed_append_processed_frame_browser"):
        setattr(host, m, MethodType(getattr(staticWidget, m), host))
    host._pin_clears = pin_clears
    return host, scan


def _arrive(host, idx, scan_stem):
    """Simulate a per-frame sigUpdate: stash the frame in the wrangler slot (with
    its source_file scan identity) and call update_data."""
    host.wrangler.thread._published_frames[idx] = SimpleNamespace(
        source_file="/data/%s_master.h5" % scan_stem)
    host.update_data(idx)


def test_scan_key_from_source():
    assert _scan_key_from_source("/d/scanA_0001.tif") == "scanA"
    assert _scan_key_from_source("/d/scanB_master.h5") == "scanB"
    assert _scan_key_from_source("/d/eiger_2_master.h5") == "eiger_2"
    assert _scan_key_from_source("") is None
    assert _scan_key_from_source(None) is None


def test_numeric_nxs_keeps_full_stem_for_titles_and_legends():
    """F2 (maintainer decision, 2026-07-12): a numeric `.nxs`/`.h5`/`.hdf5`
    container keeps its FULL stem as the scan identity — the trailing `00005` is
    part of the name and MUST show in plot titles / legend labels (and must NOT
    merge two numeric `.nxs` into one scan).  Only Eiger `_master` and per-file
    image series (`.tif`) drop a suffix."""
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
        scan_name_from_source, imageThread)
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import imageWrangler

    # The reported case: the numeric suffix survives (was being dropped).
    assert _scan_key_from_source("/d/LaB6_0710_1025pm_00005.nxs") == "LaB6_0710_1025pm_00005"
    assert _scan_key_from_source("/d/Pt_10nm_00013.nxs") == "Pt_10nm_00013"
    # ... while series / master / plain still behave:
    assert scan_name_from_source("/d/foo_0001.tif") == "foo"
    assert scan_name_from_source("/d/eiger_2_master.h5") == "eiger_2"
    assert scan_name_from_source("/d/LaB6.nxs") == "LaB6"

    # PARITY GUARD: the worker, the GUI boundary parser, and the append-target
    # resolver must derive the SAME name for every source, so titles/legends,
    # written filenames, overlays and run-end catch-up never disagree (the
    # three-way divergence F2 fixed).
    for path in ("LaB6_0710_1025pm_00005.nxs", "Pt_10nm_00013.nxs",
                 "x_00013.h5", "x_00013_master.h5", "foo_0001.tif", "LaB6.nxs"):
        canonical = scan_name_from_source(path)
        assert _scan_key_from_source(path) == canonical, path
        assert imageWrangler._append_scan_name_for_source(path) == canonical, path
        assert imageThread._eiger_scan_name(path) == canonical, path


def test_frame_driven_boundary_survives_late_new_scan():
    host, scan = _boundary_host()
    # scanA frames 1..3 (name already scanA -> no rescope, just append)
    for i in (1, 2, 3):
        _arrive(host, i, "scanA")
    assert list(scan.frames.index) == [1, 2, 3]
    assert scan.name == "scanA"

    # scanB frames 1..8 arrive BEFORE new_scan(scanB): the FIRST rescopes the panel
    # (scanA frames gone), the rest append; all 8 survive.
    for i in range(1, 9):
        _arrive(host, i, "scanB")
    assert scan.name == "scanB"
    assert list(scan.frames.index) == [1, 2, 3, 4, 5, 6, 7, 8]

    # The LATE new_scan(scanB) fires — identity-guarded, must NOT wipe scanB frames.
    host.new_scan("scanB", "/data/scanB_master.h5", False, "th", False, False)
    assert list(scan.frames.index) == [1, 2, 3, 4, 5, 6, 7, 8], \
        "late new_scan dropped the new scan's own frames (the reverted-fix bug)"


def test_out_of_sync_new_scan_defers_to_frame_stream():
    # A new_scan for a DIFFERENT scan than the one currently rendering must NOT
    # clear — directory mode bursts new_scan signals out of order, so clearing on
    # one wipes the in-flight scan's frames (the reported restart-at-9 bug: the
    # log showed new_scan(LaB6) firing while scan 03271005 was at frame 8).  The
    # frame stream (source_file) owns the transition and clears when the new scan's
    # frames actually arrive.
    host, scan = _boundary_host()
    for i in (1, 2, 3):
        _arrive(host, i, "scanA")
    assert list(scan.frames.index) == [1, 2, 3]
    # out-of-sync new_scan(scanB) while scanA renders -> DEFER (no clear, no rename)
    host.new_scan("scanB", "/data/scanB_master.h5", False, "th", False, False)
    assert list(scan.frames.index) == [1, 2, 3], "out-of-sync new_scan must not clear"
    assert scan.name == "scanA"
    # scanB's first frame arrives -> the frame-driven boundary clears scanA + rescopes
    _arrive(host, 1, "scanB")
    assert scan.name == "scanB"
    assert list(scan.frames.index) == [1]


def test_same_scan_rerun_clears_panel():
    # RE-RUNNING the same scan (same name) must clear the panel AND the overlay
    # accumulator, else the new run's (name, idx) row-ids collide with the old
    # run's and get first-occurrence-dedup dropped (stale curves under new labels).
    host, scan = _boundary_host()
    host.new_scan("scanA", "/data/scanA_master.h5", False, "th", False, False)
    for i in range(1, 6):
        _arrive(host, i, "scanA")
    assert list(scan.frames.index) == [1, 2, 3, 4, 5]
    # the overlay accumulator now holds scanA rows
    host.displayframe._waterfall_history = SimpleNamespace(
        ids=[("scanA", i) for i in range(1, 6)])
    # RE-RUN scanA
    host.new_scan("scanA", "/data/scanA_master.h5", False, "th", False, False)
    assert list(scan.frames.index) == [], "same-name re-run must clear the panel"
    # V2 (Stage 5): the S-14 reset goes through the single owner — history
    # nulled AND the reset logged with the exact cause + code site.
    assert host.displayframe._waterfall_history is None, \
        "same-name re-run must clear the overlay accumulator (S-14)"
    log = host.displayframe._accumulator_lifecycle_log
    assert [e.cause for e in log] == [LifecycleCause.SAME_NAME_RERUN]
    assert "S-14" in log[-1].site


def test_abab_rerun_clears_overlay_s14():
    # S-14 COMPLETION: A->B->A.  The prev==name guard missed this (prev=B != A);
    # deriving 'seen names' from the accumulator ITSELF catches it -- scanA
    # already has rows, so re-scoping to scanA must clear (else scanA's new frames
    # are dropped by dedup against the first scanA run).
    host, scan = _boundary_host()
    host.displayframe._waterfall_history = SimpleNamespace(
        ids=[("scanA", 1), ("scanA", 2), ("scanB", 1)])
    host._rescope_frame_panel_to("scanA")     # re-run scanA, already accumulated
    assert host.displayframe._waterfall_history is None, \
        "A->B->A re-run must clear the overlay accumulator (S-14 completion)"
    log = host.displayframe._accumulator_lifecycle_log
    assert [e.cause for e in log] == [LifecycleCause.SAME_NAME_RERUN]


def test_different_scan_boundary_keeps_overlay_s14():
    # S-14 must NOT over-fire: a boundary to a name NOT yet in the accumulator
    # appends (OV-6 cross-scan comparison), so it must not clear.
    host, scan = _boundary_host()
    host.displayframe._waterfall_history = SimpleNamespace(ids=[("scanA", 1)])
    host._rescope_frame_panel_to("scanC")     # a NEW name, not in the accumulator
    assert host.displayframe._waterfall_history is not None, \
        "a boundary to a NEW name must NOT clear the overlay (OV-6 append)"
    assert host.displayframe._accumulator_lifecycle_log == [], \
        "no owner reset may be logged for an OV-6 append boundary"


def test_batch_mode_never_frame_rescopes():
    # Batch suppresses per-frame update_data, but assert the boundary block is inert
    # under a batch-flagged wrangler even if update_data is reached.
    host, scan = _boundary_host()
    # O-1b: batch mode is frozen run policy; the worker carries no mirror for the
    # boundary block to read, so a batch run states it on the accepted object.
    host.wrangler.thread = _boundary_worker(batch_mode=True)
    for i in (1, 2, 3):
        _arrive(host, i, "scanA")
    # a scanB frame under batch must NOT rescope
    _arrive(host, 4, "scanB")
    assert scan.name == "scanA"
    assert list(scan.frames.index) == [1, 2, 3, 4]


def test_rescope_follows_scans_panel_on_first_frame():
    # scans_select_after_run (early): the Scans panel must follow the new scan as
    # soon as its frames rescope the panel (first-frame boundary) -- not wait for
    # wrangler_finished.  The writer opened <name>.nxs at run start, so it is
    # already listed by now.
    host, scan = _boundary_host()
    scans_updates = []
    host.h5viewer.update_scans = lambda: scans_updates.append(True)
    host._rescope_frame_panel_to("scanB")
    assert scans_updates, "rescope did not re-select the Scans panel on the boundary"
