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
import pytest

from tests.xdart.ov_harness import (
    CLEAR,
    INCOMPATIBLE_GRID,
    METHOD_SWITCH,
    REINTEGRATE,
    SAME_NAME_RERUN,
    InvariantViolation,
    OVHarness,
    _is_live_sentinel,
)
from xdart.gui.tabs.static_scan.display_logic import (
    AccumulatorLifecycle,
    LifecycleCause,
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
    # Stage 5: the reset is owner-logged at the accumulate gate (exact
    # code-site attribution), and the sequence ends settled.
    assert h.resets_observed[-1][2].startswith("_overlay_waterfall_payload")
    h.assert_lifecycle_settled()


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

    _state, payload = h.unit_toggle()      # Q → 2θ
    hist = h.history
    assert hist.count == 2                 # count unchanged: no reset
    assert tuple(hist.ids) == ids_q
    # V1 Stage 3: relabel-not-reset is STRUCTURAL — storage is acquisition-
    # native, so the flip touches NOTHING in the stored history ...
    assert hist.unit == unit_q
    np.testing.assert_allclose(np.asarray(hist.x), x_q, rtol=0)
    # ... and the RELABEL lives in the rendered payload: λ = 1 Å ⇒
    # 2θ = 2·asin(qλ/4π) in degrees — the display grid converts physically.
    expected = np.degrees(2.0 * np.arcsin(x_q / (4.0 * np.pi)))
    np.testing.assert_allclose(
        np.asarray(payload.traces[0].x), expected, rtol=1e-6)
    assert "°" in str(payload.axis_x.unit)             # degrees on screen

    _state, payload = h.unit_toggle()      # 2θ → Q round-trips
    hist = h.history
    assert hist.count == 2
    assert hist.unit == unit_q
    np.testing.assert_allclose(np.asarray(hist.x), x_q, rtol=0)
    np.testing.assert_allclose(np.asarray(payload.traces[0].x), x_q, rtol=0)
    assert "°" not in str(payload.axis_x.unit)         # back to Å⁻¹
    assert h.resets_observed == []


# ── S-16 DISSOLVED (V1 Stage 4): norm change preserves + re-scales at draw ─


def test_s16_dissolved_norm_change_preserves_and_rescales_at_draw():
    # Before Stage 4 this sequence pinned the S-16 contract: a REAL channel
    # change was an allowed reset (with the narrowed selection, the
    # accumulator rebuilt to 1 row).  Since Stage 4 the rows are stored
    # acquisition-native and the norm divides at draw, so the SAME sequence
    # now proves the flip: count PRESERVED, ids untouched, every rendered
    # row re-scaled by the new channel's per-row monitor value.
    h = OVHarness()
    h.publish(0)
    _state, payload = h.publish(1)
    base = [np.asarray(t.y, dtype=float).copy() for t in payload.traces]
    h.norm_change(real=False)              # repaint echo: no change at all
    assert h.persistent_count == 2

    h.click(1)                             # selection narrows (still 2 rows)
    assert h.persistent_count == 2
    # REAL channel change → re-render, NEVER a reset (pre-Stage-4 this
    # dropped frame 0's row; INV-1 would now flag that shrink as a bug).
    _state, payload = h.norm_change(real=True, channel="i1")
    assert h.persistent_count == 2         # PRESERVED (S-16 dissolved)
    assert _ids(h) == (("scanA", 0), ("scanA", 1))
    assert len(payload.traces) == 2        # both rows still render
    for before, trace in zip(base, payload.traces):
        # harness frames carry scan_info {"i0": 2.0, "i1": 4.0, ...}: the
        # draw-time norm divides each row by ITS monitor value for "i1".
        np.testing.assert_allclose(
            np.asarray(trace.y, dtype=float), before / 4.0)

    _state, payload = h.norm_change(real=True, channel="i0")   # switch again
    assert h.persistent_count == 2
    for before, trace in zip(base, payload.traces):
        np.testing.assert_allclose(
            np.asarray(trace.y, dtype=float), before / 2.0)
    assert h.resets_observed == []         # NOTHING reset across all of it


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
    assert len(h.widget._overlay_hydrated_pending_append_labels) == 0
    h.assert_lifecycle_settled()


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
    h.assert_lifecycle_settled()


def test_reintegrate_finish_same_grid_rebuilds_with_owner_logged_reset():
    # Stage 5 renegotiation (soundness gap a closed): production
    # integrator_thread_finished ALWAYS resets through the owner
    # (clear_overlay(REINTEGRATE)) and the follow-up render rebuilds from
    # the current selection — an identical regrid lands back on the SAME
    # count, and the pre-Stage-5 harness left its window ambiguously armed
    # ("allowed ≠ required").  The owner log now proves the reset fired:
    # the window is consumed EXACTLY even though the count never shrank.
    h = OVHarness()
    for i in range(3):
        h.publish(i)
    h.reintegrate_finish()                 # identical grid: reset + rebuild
    h.assert_reset_observed(REINTEGRATE)   # consumed via the owner LOG
    assert h.persistent_count == 3         # rebuilt to the same count
    assert set(_ids(h)) == {("scanA", i) for i in range(3)}
    h.assert_lifecycle_settled()


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


# ── SAME_NAME_RERUN (S-14): re-running an accumulated name resets ──────────


def test_same_name_rerun_resets_with_exact_cause():
    # Consecutive A→A: re-scoping to a name that already has rows resets
    # through the owner with SAME_NAME_RERUN, so the new run's
    # (name, frame_idx) row-ids never collide with (and get dedup-dropped
    # against) the old run's.
    h = OVHarness()
    for i in range(3):
        h.publish(i)
    h.rescope("scanA", compatible=True)    # same name re-run
    h.assert_reset_observed(SAME_NAME_RERUN)
    assert h.persistent_count == 0
    h.publish(0)
    assert _ids(h) == (("scanA", 0),)      # the NEW run's row appended
    h.assert_lifecycle_settled()


def test_abab_same_name_rerun_resets_but_new_name_appends():
    # A→B→A: the seen-set derives from the accumulator ITSELF, so the
    # return to scanA resets even though the immediately-previous scan was
    # B; a boundary to a NEW name (B) appends (OV-6 cross-scan comparison).
    h = OVHarness()
    h.publish(0)
    h.publish(1)
    h.rescope("scanB", compatible=True)    # NEW name: appends, no reset
    h.publish(0)
    assert h.persistent_count == 3
    assert ("scanA", 0) in _ids(h) and ("scanB", 0) in _ids(h)
    h.rescope("scanA", compatible=True)    # back to an accumulated name
    h.assert_reset_observed(SAME_NAME_RERUN)
    assert h.persistent_count == 0
    h.publish(0)
    assert _ids(h) == (("scanA", 0),)
    h.assert_lifecycle_settled()


# ── METHOD_SWITCH: leaving Overlay/Waterfall drops the trio together ───────


def test_method_switch_resets_and_reentry_starts_fresh():
    h = OVHarness()
    for i in range(3):
        h.publish(i)
    h.method_switch("Single")              # non-accumulating method
    h.assert_reset_observed(METHOD_SWITCH)
    assert h.persistent_count == 0
    assert h.resets_observed[-1][2] == "plot_payload[non-accumulating method]"
    h.method_switch("Overlay")             # re-entry: FRESH accumulation
    assert h.persistent_count == 3         # rebuilt from the live selection
    assert set(_ids(h)) == {("scanA", i) for i in range(3)}
    h.assert_lifecycle_settled()


def test_method_switch_clears_pins_history_and_queue_together():
    # The round-4 closure at the method-switch site: pins + history + the
    # pending-append queue all reset TOGETHER through the owner (pre-Stage-5
    # the plot_payload wipe nulled ONLY the history, stranding pins/queue).
    h = OVHarness(slice_mode=True)
    h.publish(0, select="only")
    h.move_live_cut(-10.0, 2.0)
    h.pin_current_cut()
    h.move_live_cut(0.0)
    assert h.widget._pinned_slice_cuts
    h.method_switch("Sum")
    h.assert_reset_observed(METHOD_SWITCH)
    assert h.persistent_count == 0
    assert h.widget._pinned_slice_cuts == {}
    assert len(h.widget._overlay_hydrated_pending_append_labels) == 0
    h.assert_lifecycle_settled()


# ── Stage-5 soundness closures as executable proofs ────────────────────────


def test_unowned_wipe_is_flagged_as_inv1_violation():
    # Gap (a)/(b) backstop: a rogue direct `_waterfall_history = None` (the
    # pattern the src ratchet forbids) has NO owner-logged cause, so the
    # first render that cannot rebuild every row trips INV-1 exactly.
    h = OVHarness(max_heavy_items=4)
    for i in range(10):
        h.publish(i)
    h.widget._waterfall_history = None     # rogue wipe, bypassing the owner
    with pytest.raises(InvariantViolation, match="no owner-logged cause"):
        h.render("rogue direct wipe")


def test_owner_reset_without_armed_expectation_is_flagged():
    # Every reset must be EXPECTED as well as caused: an owner reset the
    # sequence never armed fails the step check.
    h = OVHarness()
    h.publish(0)
    AccumulatorLifecycle(h.widget).reset(
        LifecycleCause.CLEAR, site="test[unexpected]")
    with pytest.raises(InvariantViolation, match="NO armed expectation"):
        h.render("unexpected owner reset")


def test_armed_cause_must_match_owner_logged_cause():
    # Gap (b): attribution is by the cause the code site LOGGED, never by
    # arming order — a mismatched window is a violation, not a consumption.
    h = OVHarness()
    h.publish(0)
    h.expect_reset(CLEAR)
    AccumulatorLifecycle(h.widget).reset(
        LifecycleCause.METHOD_SWITCH, site="test[mismatch]")
    with pytest.raises(InvariantViolation, match="attribution mismatch"):
        h.render("mismatched cause")


def test_armed_window_is_explicit_never_silent():
    # Gap (a): windows cannot stack, cannot linger silently — either the
    # owner consumes them or the sequence cancels them explicitly.
    h = OVHarness()
    h.publish(0)
    h.expect_reset(CLEAR)
    with pytest.raises(AssertionError, match="still armed"):
        h.expect_reset(REINTEGRATE)        # double-arm forbidden
    with pytest.raises(AssertionError, match="never consumed"):
        h.assert_lifecycle_settled()
    h.cancel_expected_reset()              # explicit retirement
    assert h.resets_cancelled == [CLEAR]
    h.assert_lifecycle_settled()
    h.render("after cancel")               # and the contract still holds
    assert h.persistent_count == 1
