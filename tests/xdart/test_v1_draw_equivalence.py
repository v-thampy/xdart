# -*- coding: utf-8 -*-
"""V1 Stage-3 draw equivalence: rendered output vs expected curves.

Stage 2 proved stored_row ≈ render(native) with dual-captured natives; the
Stage-3 FLIP made native storage + draw-time transforms THE production path,
so this suite pins the flipped world directly (canonical-grid plan §Stage-3):

* STORAGE is acquisition-native: ``history.unit`` is the native integration
  unit token, ``history.rows`` are un-normed / un-converted — pinned bitwise
  against the harness's synthetic profiles, reproduced here to the bit
  (float32 acquisition store, float64 accumulation).
* The RENDERED payload (``_history_to_payload`` → ``render_waterfall_view``,
  the real adapter path via :mod:`tests.xdart.ov_harness`) equals
  first-principles EXPECTED CURVES: norm = row / monitor, conversion =
  ``convert_2d_radial`` with the carried λ, BL-6 drift = the same
  ``np.interp`` operands.  Equality is EXACT (``assert_array_equal``)
  everywhere — the Stage-2 convert/interp ordering tolerance dissolved with
  the flip (drift alignment and display conversion both run in native
  space now) — plus physical peak-position assertions (the
  test_bl6_overlay_xgrid pattern) wherever an interp or conversion fires.
* The D1 policy is exercised through the harness: a cross-scan append whose
  NATIVE unit differs (scan B integrated in 2θ onto a q accumulator)
  canonicalizes with the row's own carried wavelength and renders at the
  correct physical position in BOTH display units.

A failure here is a transform bug or an unplanned storage-semantics change —
do NOT bend the expectations (mark xfail(strict=True) and report instead).
"""

from __future__ import annotations

import numpy as np
import pytest

from xrd_tools.session.display_logic import (
    convert_2d_radial,
    x_axis_for_unit,
)

from tests.xdart.ov_harness import OVHarness

_LAM = 1e-10                     # the harness default wavelength (1 Å)
_NPT = OVHarness.NPT
_X_RANGE = OVHarness.X_RANGE
_CHI = np.asarray([-20.0, -10.0, 0.0, 10.0, 20.0])
_AXIS_Q = x_axis_for_unit("q_A^-1")      # ('Q', 'Å⁻¹')
_AXIS_TTH = x_axis_for_unit("2th_deg")   # ('2θ', '°')


def native_profile(label, *, npt=_NPT, x_range=_X_RANGE, peak=None,
                   amplitude=100.0):
    """The harness's synthetic frame, reproduced to the BIT: the acquisition
    store is float32 (IntegrationResult1D), the accumulator float64 — so the
    expected native row is ``float64(float32(profile))``."""
    radial = np.linspace(x_range[0], x_range[1], npt)
    if peak is None:
        peak = x_range[0] + 0.25 * (x_range[1] - x_range[0])
    profile = (
        amplitude * np.exp(-0.5 * ((radial - peak) / 0.15) ** 2)
        + np.linspace(1.0, 2.0, npt)
        + float(label)
    )
    return (radial.astype(np.float32).astype(float),
            profile.astype(np.float32).astype(float))


def native_cut(label, chi_value, *, npt=_NPT, x_range=_X_RANGE, peak=None,
               amplitude=100.0):
    """The harness cake's single-χ slice projection (the c/w window that
    selects exactly one χ bin), reproduced to the bit: the float64 profile
    scales per χ, the cake stores float32, nanmean over one bin is exact."""
    radial = np.linspace(x_range[0], x_range[1], npt)
    if peak is None:
        peak = x_range[0] + 0.25 * (x_range[1] - x_range[0])
    profile = (
        amplitude * np.exp(-0.5 * ((radial - peak) / 0.15) ** 2)
        + np.linspace(1.0, 2.0, npt)
        + float(label)
    )
    cake = (profile[:, None]
            * (1.0 + 0.1 * np.arange(_CHI.size))[None, :]).astype(np.float32)
    k = int(np.argmin(np.abs(_CHI - chi_value)))
    return (radial.astype(np.float32).astype(float),
            cake[:, k].astype(float))


def q_to_tth(values):
    return convert_2d_radial(
        np.asarray(values, dtype=float), data_unit="q_A^-1",
        want_tth=True, want_q=False, wavelength_m=_LAM)


def _peak_x(x, row):
    return float(np.asarray(x)[int(np.argmax(np.asarray(row)))])


def assert_native_storage(h, *, unit="q_A^-1"):
    """The Stage-3 storage pin: the history is acquisition-native no matter
    what the display combo says."""
    hist = h.history
    assert hist.unit == unit
    assert len(hist.row_meta) == hist.count


def assert_rendered(payload, h, *, x, rows, axis):
    """The payload's rendered view equals the expected display curves
    bitwise, trace-for-trace in accumulation order."""
    hist = h.history
    assert payload is not None
    assert payload.display_ids == tuple(hist.ids)      # ≤256: no decimation
    assert payload.overlaid_ids == tuple(hist.ids)
    assert (payload.axis_x.label, payload.axis_x.unit) == axis
    assert len(payload.traces) == len(rows)
    for k, expected in enumerate(rows):
        np.testing.assert_array_equal(np.asarray(payload.traces[k].x), x)
        np.testing.assert_array_equal(np.asarray(payload.traces[k].y),
                                      expected)


# ── native storage + plain rendering ────────────────────────────────────────


def test_plain_appends_render_native_exact():
    h = OVHarness()
    for k in range(4):
        _state, payload = h.publish(k)
    _state, payload = h.publish(4, empty=True)   # S-17: no row, no wipe
    xs, profs = zip(*(native_profile(k) for k in range(4)))
    assert_native_storage(h)
    np.testing.assert_array_equal(np.asarray(h.history.x), xs[0])
    np.testing.assert_array_equal(
        np.atleast_2d(np.asarray(h.history.rows, dtype=float)),
        np.vstack(profs))
    assert_rendered(payload, h, x=xs[0], rows=profs, axis=_AXIS_Q)


def test_unit_flip_q_to_tth_and_back_is_pure_draw_relabel():
    h = OVHarness()
    h.publish(0)
    h.publish(1)
    _state, payload = h.unit_toggle()                 # display Q → 2θ
    _state, payload = h.publish(2)
    x_q, _ = native_profile(0)
    profs = [native_profile(k)[1] for k in range(3)]
    # Storage: untouched by the flip — native unit, native grid, all rows.
    assert_native_storage(h)
    np.testing.assert_array_equal(np.asarray(h.history.x), x_q)
    # Display: the SAME rows on the converted grid (one λ ⇒ zero row touches).
    assert_rendered(payload, h, x=q_to_tth(x_q), rows=profs, axis=_AXIS_TTH)
    _state, payload = h.unit_toggle()                 # 2θ → Q round-trips
    _state, payload = h.publish(3)
    profs.append(native_profile(3)[1])
    assert_native_storage(h)
    assert_rendered(payload, h, x=x_q, rows=profs, axis=_AXIS_Q)


def test_norm_channel_applies_at_draw_only():
    h = OVHarness()
    h.publish(0)
    h.norm_change(real=True, channel="i0")   # re-render, no reset (Stage 4)
    _state, payload = h.publish(1)
    x_q, prof0 = native_profile(0)
    _, prof1 = native_profile(1)
    # Storage stays UN-normed; the channel/monitor ride as provenance.
    # Stage 4 (S-16 dissolved): the channel change PRESERVED frame 0's row,
    # so its RowMeta keeps the provenance captured at ITS append (no channel
    # yet); frame 1, appended after the change, records "i0".  The draw-time
    # norm reads the CURRENT channel + per-row `metadata`, never RowMeta.
    assert_native_storage(h)
    np.testing.assert_array_equal(
        np.atleast_2d(np.asarray(h.history.rows, dtype=float)),
        np.vstack([prof0, prof1]))
    assert [m.norm_channel for m in h.history.row_meta] == [None, "i0"]
    assert [m.norm_value for m in h.history.row_meta] == [None, 2.0]
    # Display: BOTH rows divided by the i0 monitor (2.0) at draw.
    assert_rendered(payload, h, x=x_q, rows=[prof0 / 2.0, prof1 / 2.0],
                    axis=_AXIS_Q)


def test_norm_plus_unit_flip_compose_exact():
    h = OVHarness()
    h.norm_change(real=True, channel="i1")   # i1 monitor = 4.0
    h.publish(0)
    h.publish(1)
    h.unit_toggle()
    _state, payload = h.publish(2)
    x_q, _ = native_profile(0)
    rows = [native_profile(k)[1] / 4.0 for k in range(3)]
    assert_native_storage(h)
    assert_rendered(payload, h, x=q_to_tth(x_q), rows=rows, axis=_AXIS_TTH)


# ── BL-6 drift + cross-scan appends ────────────────────────────────────────


def test_bl6_drift_grid_same_unit_renders_interped_row_exact():
    h = OVHarness()
    h.publish(0)
    h.publish(1)
    h.rescope("scanB", compatible=True, x_range=(1.5, 5.5))
    _state, payload = h.publish(2, peak=3.0)
    x_a, prof0 = native_profile(0)
    _, prof1 = native_profile(1)
    x_b, prof2 = native_profile(2, x_range=(1.5, 5.5), peak=3.0)
    # The drift row is aligned onto scan A's grid by the SAME np.interp the
    # accumulator runs (native space) — bitwise-reproducible.
    exp2 = np.interp(x_a, x_b, prof2)
    assert_native_storage(h)
    np.testing.assert_array_equal(np.asarray(h.history.x), x_a)
    assert_rendered(payload, h, x=x_a, rows=[prof0, prof1, exp2],
                    axis=_AXIS_Q)
    dx = float(np.max(np.diff(x_a)))
    assert abs(_peak_x(x_a, payload.traces[2].y) - 3.0) < 2 * dx


def test_cross_scan_compatible_append_renders_exact():
    h = OVHarness()
    h.publish(0)
    h.publish(1)
    h.rescope("scanB", compatible=True)
    _state, payload = h.publish(2)
    x_q, _ = native_profile(0)
    rows = [native_profile(k)[1] for k in range(3)]
    assert {i[0] for i in h.history.ids} == {"scanA", "scanB"}
    assert_native_storage(h)
    assert_rendered(payload, h, x=x_q, rows=rows, axis=_AXIS_Q)


def test_bl6_drift_under_conversion_is_exact_since_the_flip():
    """Drift grid + active Q→2θ display — Stage 2's ONE tolerance case.

    Post-flip both the BL-6 alignment (append) and the display conversion
    (draw) run in NATIVE q space, so the ordering ambiguity is gone and the
    rendered drift row is bitwise ``np.interp``-in-q relabeled to 2θ."""
    h = OVHarness()
    h.publish(0)
    h.publish(1)
    h.unit_toggle()                                   # display 2θ from here
    h.rescope("scanB", compatible=True, x_range=(1.5, 5.5))
    _state, payload = h.publish(2, peak=3.0)
    x_a, prof0 = native_profile(0)
    _, prof1 = native_profile(1)
    x_b, prof2 = native_profile(2, x_range=(1.5, 5.5), peak=3.0)
    exp2 = np.interp(x_a, x_b, prof2)
    assert_native_storage(h)                          # storage still native q
    assert_rendered(payload, h, x=q_to_tth(x_a), rows=[prof0, prof1, exp2],
                    axis=_AXIS_TTH)
    # Physical: the peak sits at 2θ(q=3.0, λ) on the display grid.
    x_disp = q_to_tth(x_a)
    expected_tth = float(q_to_tth([3.0])[0])
    dx = float(np.max(np.diff(x_disp)))
    assert abs(_peak_x(x_disp, payload.traces[2].y) - expected_tth) < 2 * dx


# ── D1: cross-NATIVE-unit appends (the new Stage-3 code path) ──────────────


def test_d1_cross_native_unit_append_canonicalizes_with_carried_lambda():
    """Scan A integrated natively in q; scan B natively in 2θ, same npt (the
    reset key is unit-blind).  The incoming rows canonicalize onto the q
    accumulator with their OWN carried λ, then BL-6-interp — and render at
    the correct physical position in BOTH display units."""
    h = OVHarness()
    h.publish(0)
    h.publish(1)
    tth_range = (float(q_to_tth([1.5])[0]), float(q_to_tth([5.5])[0]))
    peak_tth = float(q_to_tth([3.0])[0])
    h.rescope("scanB", compatible=True, native_unit="2th_deg",
              x_range=tth_range)
    _state, payload = h.publish(2, peak=peak_tth)

    x_a, prof0 = native_profile(0)
    _, prof1 = native_profile(1)
    x_b_tth, prof2 = native_profile(2, x_range=tth_range, peak=peak_tth)
    # The canonicalization the accumulator runs: scan B's 2θ grid → q with
    # the row's carried λ, then interp onto scan A's grid — same operands.
    x_b_q = convert_2d_radial(
        x_b_tth, data_unit="2th_deg", want_tth=False, want_q=True,
        wavelength_m=_LAM)
    exp2 = np.interp(x_a, x_b_q, prof2)

    hist = h.history
    assert hist.count == 3
    assert_native_storage(h, unit="q_A^-1")           # grid stays scan A's
    np.testing.assert_array_equal(np.asarray(hist.x), x_a)
    assert [m.source_unit for m in hist.row_meta] == [
        "q_A^-1", "q_A^-1", "2th_deg"]                # provenance kept
    assert_rendered(payload, h, x=x_a, rows=[prof0, prof1, exp2],
                    axis=_AXIS_Q)
    dx = float(np.max(np.diff(x_a)))
    assert abs(_peak_x(x_a, payload.traces[2].y) - 3.0) < 2 * dx
    assert h.resets_observed == []                    # append, never reset

    _state, payload = h.unit_toggle()                 # display 2θ
    x_disp = q_to_tth(x_a)
    assert_rendered(payload, h, x=x_disp, rows=[prof0, prof1, exp2],
                    axis=_AXIS_TTH)
    dx = float(np.max(np.diff(x_disp)))
    assert abs(_peak_x(x_disp, payload.traces[2].y) - peak_tth) < 2 * dx


# ── slice projections (2D→1D stays build-time) ─────────────────────────────


def test_slice_pin_and_live_cut_render_native_exact():
    h = OVHarness(slice_mode=True)
    h.publish(0, select="only")
    h.move_live_cut(-10.0, 2.0)
    h.pin_current_cut()
    _state, payload = h.move_live_cut(10.0, 2.0)      # live re-emerges
    x_q, cut_pin = native_cut(0, -10.0)
    _, cut_live = native_cut(0, 10.0)
    hist = h.history
    assert hist.count == 2                            # the pin + the current
    assert_native_storage(h)
    np.testing.assert_array_equal(
        np.atleast_2d(np.asarray(hist.rows, dtype=float)),
        np.vstack([cut_pin, cut_live]))
    assert_rendered(payload, h, x=x_q, rows=[cut_pin, cut_live],
                    axis=_AXIS_Q)
    # Two genuinely different χ windows (per-χ cake scaling).
    assert not np.allclose(cut_pin, cut_live)


# ── D2 honesty: no wavelength anywhere ─────────────────────────────────────


def test_unit_flip_without_wavelength_stays_native_no_lie():
    h = OVHarness(wavelength_m=None)
    h.publish(0)
    h.unit_toggle()                   # asks for 2θ; nothing can convert
    _state, payload = h.publish(1)
    x_q, prof0 = native_profile(0)
    _, prof1 = native_profile(1)
    assert_native_storage(h)
    # The axis stays native Q — never a 2θ label over unconverted values.
    assert_rendered(payload, h, x=x_q, rows=[prof0, prof1], axis=_AXIS_Q)
    assert float(payload.traces[0].x[-1]) == pytest.approx(5.0)
