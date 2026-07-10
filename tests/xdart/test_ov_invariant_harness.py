# -*- coding: utf-8 -*-
"""QW-3 — the OV acceptance contract as executable sequences (design §4.3).

Each ledgered OV bug is one short scripted sequence against the REAL
adapter + accumulator stack via :mod:`tests.xdart.ov_harness`.  The harness
re-checks the whole acceptance contract (count monotonic except allowed
resets; one strictly-monotonic grid; no constant-clamped rows; pins ⊆
history) after EVERY event — the assertions below are the per-bug OUTCOME
checks on top of that standing contract.

Ledger: live_findings_ledger.md rows OV-1..OV-7c, S-16, S-17, BL-6 and the
"Acceptance test that covers the OV family" composed sequence.
"""

from __future__ import annotations

import numpy as np

from tests.xdart.ov_harness import (
    CLEAR,
    INCOMPATIBLE_GRID,
    NORM_CHANGE,
    REINTEGRATE,
    OVHarness,
    _is_live_sentinel,
)


def _ids(harness):
    return tuple(harness.history.ids)


# ── OV-1: Overlay entry seed → click B → both persist ─────────────────────


def test_ov1_entry_seed_then_click_appends():
    h = OVHarness()
    h.publish(0, select="only")            # Overlay entry with frame A shown
    assert h.persistent_count == 1         # seeded with the displayed trace
    h.publish(1, select=None)              # B processed, not yet selected
    h.click(1)                             # click B → A must NOT be erased
    assert set(_ids(h)) == {("scanA", 0), ("scanA", 1)}
    h.click(0)                             # C/D-style continued browsing
    assert h.persistent_count == 2
    assert h.resets_observed == []


# ── OV-2: click far outside the resident window → old traces survive ──────


def test_ov2_evicted_click_preserves_history():
    h = OVHarness(max_heavy_items=4)
    for i in range(12):                    # the window slides past 0..7
        h.publish(i)
    assert h.persistent_count == 12        # accumulator kept every row
    assert not h.store.get(0).view.has_1d  # 0 really is store-evicted
    h.click(0)                             # the OV-2 click
    assert h.persistent_count == 12        # whole plot does NOT redraw bare
    assert set(_ids(h)) == {("scanA", i) for i in range(12)}
    assert (0, "1d") in h.hydration_requests   # rehydration was queued
    assert h.resets_observed == []


# ── OV-3: hydration completion APPENDS (never fresh-plots) ─────────────────


def test_ov3_hydration_completion_appends_not_fresh_plots():
    h = OVHarness(max_heavy_items=4)
    for i in range(8):
        h.publish(i)
    h.publish(8, select=None)              # processed but never rendered
    h.evict(8)
    h.click(8)                             # select the evicted frame
    assert h.persistent_count == 8         # preserved while hydrating
    assert (8, "1d") in h.hydration_requests
    h.hydration_complete(8)                # worker lands the 1D row
    assert h.persistent_count == 9         # appended onto the overlay
    assert set(_ids(h)) == {("scanA", i) for i in range(9)}

    # The stale-generation flavour: the completion outlived its selection and
    # joins via the pending-append queue (BR-2/OV-3 path).
    h.publish(9, select=None)
    h.evict(9)
    h.deselect_all()
    h.hydration_complete(9, stale=True)
    assert h.persistent_count == 10
    assert ("scanA", 9) in _ids(h)
    assert h.resets_observed == []


# ── OV-4: current frame evicted during live → overlay survives ────────────


def test_ov4_current_frame_evicted_during_live_overlay_survives():
    h = OVHarness(max_heavy_items=4)
    h.widget._processing_active = True     # live run, auto-last growth
    for i in range(6):
        h.publish(i)
    h.click(5)                             # Auto-Last current frame
    h.evict(5)                             # its heavy payload is thinned
    assert h.persistent_count == 6         # overlay does NOT clear
    h.render("live-tick")
    assert h.persistent_count == 6
    assert set(_ids(h)) == {("scanA", i) for i in range(6)}
    assert h.resets_observed == []


# ── OV-5: empty-selection / control repaints never wipe ───────────────────


def test_ov5_empty_selection_and_repaints_never_wipe():
    h = OVHarness()
    for i in range(3):
        h.publish(i)
    h.deselect_all()                       # whitespace click
    assert h.persistent_count == 3
    h.unit_toggle()                        # plotUnit repaint, empty selection
    assert h.persistent_count == 3
    h.image_unit_toggle()                  # imageUnit repaint
    assert h.persistent_count == 3
    h.norm_change(real=False)              # norm refresh echo (no real change)
    assert h.persistent_count == 3
    h.render("run-end repaint")
    assert h.persistent_count == 3
    h.click(1)                             # reselect: dedupe, no double row
    assert h.persistent_count == 3
    assert h.resets_observed == []


# ── OV-6: compatible cross-scan APPENDS; incompatible grid resets ──────────


def test_ov6_compatible_cross_scan_appends_incompatible_resets():
    h = OVHarness()
    for i in range(3):
        h.publish(i)
    h.rescope("scanB", compatible=True)    # scan boundary, same axis+npt
    assert h.persistent_count == 3         # survives the boundary
    h.publish(0)                           # scan B's frames APPEND
    h.publish(1)
    assert h.persistent_count == 5
    assert ("scanA", 0) in _ids(h) and ("scanB", 0) in _ids(h)  # scan-qualified

    h.rescope("scanC", compatible=False)   # DIFFERENT npt → allowed reset
    h.publish(0)
    h.assert_reset_observed(INCOMPATIBLE_GRID)
    assert _ids(h) == (("scanC", 0),)


# ── BL-6: same axis+npt, different radial_range → reinterp onto one grid ──


def test_bl6_cross_scan_reinterp_lands_peak_at_physical_position():
    h = OVHarness()
    h.publish(0, peak=2.0)
    x_a = np.asarray(h.history.x, dtype=float)
    h.rescope("scanB", compatible=True, x_range=(1.5, 5.5))
    h.publish(0, peak=3.0)                 # same npt, shifted radial_range
    hist = h.history
    np.testing.assert_allclose(np.asarray(hist.x), x_a)   # keeps A's grid
    row = np.asarray(hist.rows)[list(hist.ids).index(("scanB", 0))]
    peak_x = float(hist.x[int(np.argmax(row))])
    # The peak lands at the correct PHYSICAL q (~3.0), not at scan A's bin
    # for scan B's index (the OV-6 misgrid BL-6 reopened).
    assert abs(peak_x - 3.0) < 2 * (x_a[1] - x_a[0])
    assert h.resets_observed == []


# ── OV-7: pinned cuts survive norm/unit rebuilds ───────────────────────────


def test_ov7_pinned_cuts_survive_norm_and_unit_rebuilds():
    h = OVHarness(slice_mode=True)
    h.publish(0, select="only")
    h.move_live_cut(-10.0, 2.0)
    h.pin_current_cut()
    h.move_live_cut(0.0)
    h.pin_current_cut()
    h.move_live_cut(10.0)                  # live cut off both pins
    pins = set(h.widget._pinned_slice_cuts)
    assert len(pins) == 2
    h.norm_change(real=False)              # norm repaint: pins survive
    assert pins <= set(_ids(h))
    h.unit_toggle()                        # unit rebuild relabels; pins stay
    assert pins <= set(_ids(h))
    h.render("bg/levels repaint")
    assert pins <= set(_ids(h))
    assert h.persistent_count == 2
    assert h.resets_observed == []


# ── OV-7b: Pin ABSORBS the live current cut when c/w equal ─────────────────


def test_ov7b_pin_absorbs_matching_live_cut():
    h = OVHarness(slice_mode=True)
    h.publish(0, select="only")
    h.move_live_cut(-10.0, 2.0)
    h.pin_current_cut()                    # pin at the live c/w → absorbed
    hist = h.history
    assert hist.count == 1                 # 2 traces would be the OV-7b dup
    assert not any(_is_live_sentinel(i) for i in hist.ids)
    assert not any("current" in n for n in hist.names)

    h.move_live_cut(0.0)                   # current REAPPEARS beside the pin
    hist = h.history
    assert hist.count == 2
    assert any(_is_live_sentinel(i) for i in hist.ids)

    h.pin_current_cut()                    # second pin absorbs again
    hist = h.history
    assert hist.count == 2                 # two pins, no lingering sentinel
    assert not any(_is_live_sentinel(i) for i in hist.ids)

    h.move_live_cut(-10.0)                 # re-dial ONTO pin 1 → suppressed
    hist = h.history
    assert hist.count == 2
    assert not any(_is_live_sentinel(i) for i in hist.ids)
    assert h.persistent_count == 2


# ── OV-7c: the live current previews the NEXT free slot above pins ─────────


def test_ov7c_live_current_previews_next_free_slot():
    h = OVHarness(slice_mode=True)
    h.publish(0, select="only")
    h.move_live_cut(-10.0, 2.0)
    h.pin_current_cut()
    h.move_live_cut(0.0)
    h.pin_current_cut()
    h.move_live_cut(10.0)                  # live at a new center
    hist = h.history
    live_pos = [k for k, i in enumerate(hist.ids) if _is_live_sentinel(i)]
    assert live_pos == [2]                 # the slot above the two pins

    h.pin_current_cut()                    # Pin freezes it IN PLACE
    hist = h.history
    assert not any(_is_live_sentinel(i) for i in hist.ids)
    assert len([i for i in hist.ids
                if isinstance(i, tuple) and len(i) >= 3]) == 3
    # the frozen pin holds slot 2 (no jump), and its name is no longer live
    assert "current" not in hist.names[2]

    h.move_live_cut(20.0)                  # the NEXT current takes slot 3
    hist = h.history
    live_pos = [k for k, i in enumerate(hist.ids) if _is_live_sentinel(i)]
    assert live_pos == [3]


# ── S-17: an empty incoming grid never wipes ───────────────────────────────


def test_s17_empty_grid_publication_never_wipes():
    h = OVHarness()
    h.publish(0)
    h.publish(1)
    x_before = np.asarray(h.history.x, dtype=float).copy()
    h.publish(2, empty=True)               # x.size == 0 arrives mid-run
    assert h.persistent_count == 2         # the accumulator is preserved
    np.testing.assert_allclose(np.asarray(h.history.x), x_before)
    h.render("post-empty repaint")
    assert h.persistent_count == 2
    assert h.resets_observed == []


# ── unit-flip: RELABEL, never reset ────────────────────────────────────────


def test_unit_flip_relabels_never_resets():
    h = OVHarness()
    h.publish(0, peak=2.0)
    h.publish(1, peak=3.0)
    x_q = np.asarray(h.history.x, dtype=float).copy()
    unit_q = h.history.unit
    ids_q = _ids(h)

    h.unit_toggle()                        # Q → 2θ
    hist = h.history
    assert hist.count == 2                 # count unchanged: no reset
    assert tuple(hist.ids) == ids_q
    assert hist.unit != unit_q             # ... but the axis is RELABELED
    # λ = 1 Å: 2θ = 2·asin(qλ/4π) in degrees — the grid converts physically.
    expected = np.degrees(2.0 * np.arcsin(x_q / (4.0 * np.pi)))
    np.testing.assert_allclose(np.asarray(hist.x), expected, rtol=1e-6)

    h.unit_toggle()                        # 2θ → Q round-trips
    hist = h.history
    assert hist.count == 2
    assert hist.unit == unit_q
    np.testing.assert_allclose(np.asarray(hist.x), x_q, rtol=1e-6)
    assert h.resets_observed == []


# ── S-16 class: real norm change resets; norm repaint does not ─────────────


def test_s16_norm_change_classes():
    h = OVHarness()
    h.publish(0)
    h.publish(1)
    h.norm_change(real=False)              # repaint echo: NO reset
    assert h.persistent_count == 2

    h.click(1)                             # selection narrows (still 2 rows)
    assert h.persistent_count == 2
    h.norm_change(real=True)               # REAL channel change: allowed reset
    h.assert_reset_observed(NORM_CHANGE)
    assert h.persistent_count == 1         # rebuilt from the current render
    assert _ids(h) == (("scanA", 1),)

    h.click(0)                             # re-accumulates under the new norm
    assert h.persistent_count == 2


# ── Clear: the canonical allowed reset — history AND pins together ────────


def test_clear_resets_history_and_pins_together():
    h = OVHarness(slice_mode=True)
    h.publish(0, select="only")
    h.move_live_cut(-10.0, 2.0)
    h.pin_current_cut()
    h.move_live_cut(0.0)
    h.pin_current_cut()
    assert h.persistent_count == 2
    h.deselect_all()                       # pins keep rendering (recipes)
    assert h.persistent_count == 2
    h.clear()                              # the Clear button
    h.assert_reset_observed(CLEAR)
    assert h.persistent_count == 0
    assert h.widget._pinned_slice_cuts == {}   # reset TOGETHER (round-4 hole)


# ── reintegrate-finish: allowed to reset (regrid); dedupes when identical ──


def test_reintegrate_finish_regrid_is_an_allowed_reset():
    h = OVHarness()
    for i in range(4):
        h.publish(i)
    h.click(2)
    h.reintegrate_finish(npt=64)           # the pass regridded the scan
    h.assert_reset_observed(REINTEGRATE)
    assert h.persistent_count == 1         # rebuilt from the current render
    assert _ids(h) == (("scanA", 2),)
    assert np.asarray(h.history.x).size == 64


def test_reintegrate_finish_same_grid_keeps_rows():
    h = OVHarness()
    for i in range(3):
        h.publish(i)
    h.reintegrate_finish()                 # identical grid: rows dedupe
    assert h.persistent_count == 3         # allowed ≠ required: no shrink
    assert h.pending_reset == REINTEGRATE  # the window was never needed


# ── the ledger's canonical composed sequence ───────────────────────────────


def test_composed_canonical_ledger_sequence():
    # "accumulator count is MONOTONIC through every step of: Overlay-mode
    # entry (seeded with the displayed trace) → resident click → evicted
    # click → deselect-all → unit toggle → hydration completion → repaint."
    # The harness enforces monotonicity after every event; this sequence
    # composes all of them and pins the end state.
    h = OVHarness(max_heavy_items=6)
    h.publish(0, select="only")            # entry: seeded with displayed trace
    assert h.persistent_count == 1
    for i in range(1, 10):                 # live growth (window slides)
        h.publish(i)
    assert h.persistent_count == 10
    h.click(8)                             # resident click
    assert not h.store.get(0).view.has_1d  # 0 slid out of the heavy window
    h.click(0)                             # evicted click
    assert h.persistent_count == 10
    h.deselect_all()
    assert h.persistent_count == 10
    h.unit_toggle()
    assert h.persistent_count == 10
    h.hydration_complete(0)                # completion lands
    assert h.persistent_count == 10        # dedupe: appended once, ever
    h.render("repaint")
    assert h.persistent_count == 10
    assert set(_ids(h)) == {("scanA", i) for i in range(10)}
    assert h.resets_observed == []         # nothing was allowed to reset
