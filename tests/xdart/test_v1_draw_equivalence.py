# -*- coding: utf-8 -*-
"""V1 Stage-2 equivalence proof: stored rows ≈ draw-time render of natives.

The point of this stage (canonical-grid plan §Stage-2): before Stage 3 flips
storage to acquisition-native rows, PROVE that rendering the pre-transform
(native_x, native_y) through the new pure draw-time functions reproduces what
today's build-time transforms store.  Production-wired: every sequence drives
the real adapter → ``append_row`` → ``accumulate_waterfall`` path via
:mod:`tests.xdart.ov_harness`; the natives are dual-captured behind
``XDART_OV_DUALCAPTURE=1`` (tests only) into the RowMeta slots reserved in
Stage 1.

For each accumulated row:

* EXACT equality (``assert_array_equal``) wherever no interp ordering differs
  — plain appends, unit flips (one λ: a pure x-relabel), norm channels
  (scalar division), BL-6 drift in the native unit (the SAME ``np.interp``
  call in the same space), slice pins, cross-scan compatible appends.
* tolerance PLUS physical peak-position assertions (the
  test_bl6_overlay_xgrid.py pattern) where the orderings genuinely differ:
  today converts per-row THEN interps on the display grid, draw-time interps
  on the native grid THEN converts the axis — linear interp does not commute
  with the nonlinear Q↔2θ map, so values match only approximately while the
  peak must land at the same physical position either way.

A failure here is a discovery about today's transform behavior — do NOT bend
the pure functions to reproduce a bug (mark the case xfail(strict=True) and
report it instead).
"""

from __future__ import annotations

import numpy as np
import pytest

from xrd_tools.session.display_logic import (
    accumulate_waterfall,
    convert_2d_radial,
    render_waterfall_view,
    x_axis_for_unit,
)

from tests.xdart.ov_harness import OVHarness


@pytest.fixture(autouse=True)
def _dualcapture(monkeypatch):
    """Stage-2 scaffolding: append_row stashes each row's pre-transform
    (native_x, native_y) in RowMeta.  Tests only — unset means zero cost."""
    monkeypatch.setenv("XDART_OV_DUALCAPTURE", "1")


def rebuild_native_history(stored):
    """Re-accumulate the dual-captured PRE-transform rows — the Stage-3
    storage model — through the very same ``accumulate_waterfall`` (native
    unit, native grids; BL-6 drift alignment happens in native space)."""
    native = None
    for k, meta in enumerate(stored.row_meta):
        assert meta is not None and meta.native_x is not None, (
            f"row {k} ({stored.ids[k]!r}): dual-capture missing")
        native = accumulate_waterfall(
            native,
            reset_key=stored.reset_key,
            unit=meta.source_unit,
            label="",
            x=meta.native_x,
            rows=[meta.native_y],
            ids=[stored.ids[k]],
            names=[stored.names[k]],
            metadata=[dict(stored.metadata[k])],
            row_meta=[meta],
        )
    return native


def render_native(h, native):
    """Draw-time render exactly as Stage 3 will call it: the CURRENT plotUnit
    combo text and the CURRENT norm channel, over the native history."""
    return render_waterfall_view(
        native,
        want_unit=h.widget.ui.plotUnit.currentText(),
        norm_channel=h.widget.get_normChannel(),
        max_rows=None,
    )


def assert_equivalent_exact(h):
    """stored == render(native) bit-for-bit: grid, every row, ids, axis."""
    stored = h.history
    native = rebuild_native_history(stored)
    x_disp, rows_disp, ids_disp, axis = render_native(h, native)
    assert ids_disp == tuple(stored.ids)
    np.testing.assert_array_equal(x_disp, np.asarray(stored.x, dtype=float))
    np.testing.assert_array_equal(
        rows_disp, np.atleast_2d(np.asarray(stored.rows, dtype=float)))
    assert (axis.label, axis.unit) == (stored.label, stored.unit)
    return stored, x_disp, rows_disp


def _peak_x(x, row):
    return float(np.asarray(x)[int(np.argmax(np.asarray(row)))])


# ── exact cases (no interp ordering in play) ───────────────────────────────


def test_plain_appends_render_exact():
    h = OVHarness()
    for k in range(4):
        h.publish(k)
    h.publish(4, empty=True)          # S-17: contributes no row, wipes nothing
    stored, _x, rows = assert_equivalent_exact(h)
    assert stored.count == 4 and rows.shape[0] == 4


def test_unit_flip_q_to_tth_and_back_render_exact():
    # One λ everywhere: the flip is a pure x-relabel — rows are untouched in
    # both worlds, and the grids come from the same convert_2d_radial call.
    h = OVHarness()
    h.publish(0)
    h.publish(1)
    h.unit_toggle()                                   # Q → 2θ
    h.publish(2)
    stored, x_disp, _rows = assert_equivalent_exact(h)
    assert stored.unit == x_axis_for_unit("2th_deg")[1]     # really flipped
    assert x_disp[0] > 5.0                                  # degrees, not Å⁻¹
    h.unit_toggle()                                   # 2θ → Q
    h.publish(3)
    stored, _x, _rows = assert_equivalent_exact(h)
    assert stored.unit == x_axis_for_unit("q_A^-1")[1]
    assert stored.count == 4


def test_norm_channel_rows_render_exact():
    h = OVHarness()
    h.publish(0)
    h.norm_change(real=True, channel="i0")   # S-16 rebuild under the channel
    h.publish(1)
    stored, _x, rows = assert_equivalent_exact(h)
    # Guard against a vacuous pass: the stored rows really are scaled by the
    # i0 monitor (2.0), i.e. the draw-time norm actually did something.
    for k, meta in enumerate(stored.row_meta):
        assert meta.norm_channel == "i0" and meta.norm_value == 2.0
        np.testing.assert_array_equal(
            np.asarray(stored.rows)[k], meta.native_y / 2.0)
    assert rows.shape[0] == 2


def test_norm_plus_unit_flip_compose_exact():
    # Scalar norm + one-λ conversion compose without any resampling: exact.
    h = OVHarness()
    h.norm_change(real=True, channel="i1")
    h.publish(0)
    h.publish(1)
    h.unit_toggle()
    h.publish(2)
    stored, _x, _rows = assert_equivalent_exact(h)
    assert stored.count == 3
    assert stored.unit == x_axis_for_unit("2th_deg")[1]


def test_bl6_drift_grid_same_unit_render_exact():
    # Same axis+npt, different radial_range (BL-6): today interps the new
    # row onto the prior grid in native space — the native rebuild runs the
    # SAME np.interp with the same operands, so equality is exact.
    h = OVHarness()
    h.publish(0)
    h.publish(1)
    h.rescope("scanB", compatible=True, x_range=(1.5, 5.5))
    h.publish(2, peak=3.0)
    stored, x_disp, rows = assert_equivalent_exact(h)
    assert stored.count == 3
    # Physical sanity (the test_bl6_overlay_xgrid pattern): the drift row's
    # peak sits at q≈3.0 on the shared grid, not at scan A's index position.
    dx = float(np.max(np.diff(x_disp)))
    assert abs(_peak_x(x_disp, rows[2]) - 3.0) < 2 * dx


def test_cross_scan_compatible_append_render_exact():
    # Identical grid across the boundary: append verbatim, no interp at all.
    h = OVHarness()
    h.publish(0)
    h.publish(1)
    h.rescope("scanB", compatible=True)
    h.publish(2)
    stored, _x, _rows = assert_equivalent_exact(h)
    assert stored.count == 3
    assert {i[0] for i in stored.ids} == {"scanA", "scanB"}


def test_slice_pin_and_live_cut_render_exact():
    # 2D→1D slice projections (irreversible, build-time in both worlds):
    # the natives are the projected curves pre-norm/pre-conversion, so with
    # no conversion in play the render is exact — pin AND live current cut.
    h = OVHarness(slice_mode=True)
    h.publish(0, select="only")
    h.move_live_cut(-10.0, 2.0)
    h.pin_current_cut()
    h.move_live_cut(10.0, 2.0)        # live sentinel re-emerges off the pin
    stored, _x, rows = assert_equivalent_exact(h)
    assert stored.count == 2          # the pin + the live current cut
    assert rows.shape[0] == 2
    # The two windows scale differently (per-χ cake scaling): really 2 cuts.
    assert not np.allclose(rows[0], rows[1])


def test_unit_flip_without_wavelength_stays_native_no_lie():
    # D2's honesty rule, both worlds: no λ anywhere ⇒ the conversion request
    # is refused and the axis stays native — today per-row (no conversion
    # fires), draw-time stack-wide (native axis returned honestly).
    h = OVHarness(wavelength_m=None)
    h.publish(0)
    h.unit_toggle()                   # asks for 2θ; nothing can convert
    h.publish(1)
    stored, x_disp, _rows = assert_equivalent_exact(h)
    assert stored.unit == x_axis_for_unit("q_A^-1")[1]   # still Å⁻¹
    assert float(x_disp[-1]) == pytest.approx(5.0)       # still the q grid


# ── the tolerance case: convert/interp ordering genuinely differs ──────────


def test_bl6_drift_after_unit_flip_tolerance_and_physical_peaks():
    """Drift grid + active Q→2θ conversion — THE ordering case.

    Today: the drift row is converted to 2θ with its λ, THEN np.interp'd onto
    the 2θ display grid.  Draw-time: it is np.interp'd onto the native q grid,
    THEN the axis converts (one λ ⇒ rows untouched).  Linear interpolation
    does not commute with the nonlinear q↔2θ map, so the drift row matches
    only to a tolerance — while the grid, the non-drift rows, and the peak's
    PHYSICAL position must all agree exactly/physically.
    """
    h = OVHarness()
    h.publish(0)
    h.publish(1)
    h.unit_toggle()                                   # display 2θ from here
    h.rescope("scanB", compatible=True, x_range=(1.5, 5.5))
    h.publish(2, peak=3.0)

    stored = h.history
    native = rebuild_native_history(stored)
    x_disp, rows_disp, ids_disp, axis = render_native(h, native)
    stored_rows = np.atleast_2d(np.asarray(stored.rows, dtype=float))

    assert ids_disp == tuple(stored.ids)
    # The display grid and axis come from the same conversion either way.
    np.testing.assert_array_equal(x_disp, np.asarray(stored.x, dtype=float))
    assert (axis.label, axis.unit) == (stored.label, stored.unit)
    # Non-drift rows (scan A, native grid == display grid source): exact.
    np.testing.assert_array_equal(rows_disp[0], stored_rows[0])
    np.testing.assert_array_equal(rows_disp[1], stored_rows[1])
    # The drift row: interp-in-2θ vs interp-in-q.  Tolerance, not identity —
    # but TIGHT (observed max|Δ| ≈ 3.3e-3 on a ~100-amplitude peak; a real
    # transform bug shifts whole bins, orders of magnitude above this), so
    # the tolerance cannot paper over a genuine transform defect.
    assert np.allclose(rows_disp[2], stored_rows[2], rtol=1e-3, atol=0.05), (
        f"max |Δ| = {np.max(np.abs(rows_disp[2] - stored_rows[2])):.4f}")
    # Physical peak position: q=3.0 converted with the carried λ, in BOTH
    # renders, within the bin resolution (the test_bl6 pattern).
    expected_tth = float(convert_2d_radial(
        np.asarray([3.0]), data_unit="q_A^-1", want_tth=True, want_q=False,
        wavelength_m=1e-10)[0])
    dx = float(np.max(np.diff(x_disp)))
    assert abs(_peak_x(x_disp, stored_rows[2]) - expected_tth) < 2 * dx
    assert abs(_peak_x(x_disp, rows_disp[2]) - expected_tth) < 2 * dx
