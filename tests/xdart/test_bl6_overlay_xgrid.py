"""BL-6 / S-17 — cross-scan overlay x-grid correctness.

Production-wired: drives the REAL ``accumulate_waterfall`` (the payload-owned
Overlay/Waterfall accumulator) with real arrays — no monkeypatched seam.  BL-6:
two scans with the same axis kind + npt but a DIFFERENT radial_range are
grid-compatible (range is excluded from the reset key on purpose), so without a
value-level reinterp scan B's intensities render at scan A's x positions.  S-17:
one empty-grid publication must not wipe the accumulator.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.display_logic

from xrd_tools.session.display_logic import accumulate_waterfall


def _peak_row(x, peak_x):
    row = np.zeros_like(x)
    row[int(np.argmin(np.abs(x - peak_x)))] = 100.0
    return row


def test_cross_scan_different_range_reinterps_to_common_x():
    # Scan A establishes the accumulator grid (range 1..5).
    xA = np.linspace(1.0, 5.0, 200)
    histA = accumulate_waterfall(
        None, reset_key="grid", unit="q_A^-1", x=xA,
        rows=[np.zeros_like(xA)], ids=[("A", 0)], names=["A/0"])
    assert np.allclose(histA.x, xA)

    # Scan B: SAME npt (200), DIFFERENT range (1.5..5.5) with a peak at the
    # physical position q = 3.0.  Grid-compatible by key, different x values.
    xB = np.linspace(1.5, 5.5, 200)
    histB = accumulate_waterfall(
        histA, reset_key="grid", unit="q_A^-1", x=xB,
        rows=[_peak_row(xB, 3.0)], ids=[("B", 0)], names=["B/0"])

    # The accumulator KEEPS scan A's grid and reinterps scan B onto it.
    assert np.allclose(histB.x, xA)
    b_pos = list(histB.ids).index(("B", 0))
    b_row = np.asarray(histB.rows)[b_pos]
    # The peak must land at the correct PHYSICAL x (~3.0), not at scan A's
    # position for scan B's index (the OV-6 misgrid).
    peak_x = float(histB.x[int(np.argmax(b_row))])
    assert abs(peak_x - 3.0) < 2 * (xA[1] - xA[0])


def test_same_grid_values_are_not_needlessly_reinterped():
    # Identical grid + values: append verbatim (no interp drift), correct x.
    x = np.linspace(1.0, 5.0, 128)
    hist = accumulate_waterfall(
        None, reset_key="grid", unit="q_A^-1", x=x,
        rows=[_peak_row(x, 2.5)], ids=[("A", 0)], names=["A/0"])
    hist = accumulate_waterfall(
        hist, reset_key="grid", unit="q_A^-1", x=x.copy(),
        rows=[_peak_row(x, 4.0)], ids=[("A", 1)], names=["A/1"])
    assert np.allclose(hist.x, x)
    for ident, want in ((("A", 0), 2.5), (("A", 1), 4.0)):
        row = np.asarray(hist.rows)[list(hist.ids).index(ident)]
        assert abs(float(hist.x[int(np.argmax(row))]) - want) < 2 * (x[1] - x[0])


def test_empty_grid_publication_does_not_wipe_accumulator():
    # S-17: one empty-grid publication (x.size == 0) must PRESERVE the overlay.
    x = np.linspace(1.0, 5.0, 100)
    hist = accumulate_waterfall(
        None, reset_key="grid", unit="q_A^-1", x=x,
        rows=[np.ones_like(x)], ids=[("A", 0)], names=["A/0"])

    after = accumulate_waterfall(
        hist, reset_key="grid", unit="q_A^-1", x=np.empty(0),
        rows=np.empty((0, 0)), ids=[], names=[])

    assert np.allclose(after.x, x)
    assert list(after.ids) == [("A", 0)]          # not wiped
    assert np.asarray(after.rows).shape[0] == 1   # the prior row survives
