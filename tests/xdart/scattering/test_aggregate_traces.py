"""Average and Sum of the curves shown, for display (Int's view and Stitch's).

Maintainer, 2026-09-30: "but Average/Sum require identical axes - can this be changed and made more flexible,
at least for sum? This is only for display, right" -- "The simplest way might be to concatenate the arrays ...
The other approach could be to bin them together on a common grid". Curves on different grids of one axis are
interpolated onto one grid spanning them, as fine as the finest; each point takes the curves that reach it, so
a stretch only one curve reaches shows that curve as it is (as concatenating them would) and an overlap adds.
"""
import numpy as np
import pytest

from xdart.gui.tabs.scattering.display_values import DisplayFrameKey
from xdart.gui.tabs.scattering.events import RunIdentity
from xdart.gui.tabs.scattering.shell_values import AxisProjection, TraceProjection
from xdart.gui.tabs.scattering.shell_widgets import aggregate_traces, foldable


def _trace(ordinal, axis, intensity, *, label="Q", unit="Å⁻¹"):
    frame = DisplayFrameKey(RunIdentity(1, "fold"), "scan", "/out/result.nxs", ordinal, ordinal)
    return TraceProjection(frame, AxisProjection(np.asarray(axis, dtype=float), label, unit),
                           np.asarray(intensity, dtype=float), f"scan_{ordinal}")


def _expected(traces, x, mode):
    """Point by point, the curves reaching it: np.interp of each, by hand."""
    out = []
    for value in x:
        reached = [np.interp(value, t.axis.values, t.intensity) for t in traces
                   if t.axis.values.min() <= value <= t.axis.values.max()]
        out.append(np.nan if not reached else sum(reached) / (len(reached) if mode == "Average" else 1))
    return np.asarray(out)


def test_one_grid_folds_point_by_point_as_before():
    axis = np.linspace(1., 2., 11)
    traces = (_trace(1, axis, np.arange(11.)), _trace(2, axis, 2 * np.arange(11.)))
    (summed,) = aggregate_traces(traces, "Sum")
    assert summed.axis is traces[0].axis and summed.title == "Sum"
    np.testing.assert_array_equal(summed.intensity, 3 * np.arange(11.))
    # a grid equal to rounding (numerically equivalent bins) is the same grid: the first's axis is kept
    nudged = _trace(3, axis * (1 + 1e-15), np.ones(11))
    (average,) = aggregate_traces((traces[0], nudged), "Average")
    assert average.axis is traces[0].axis
    np.testing.assert_array_equal(average.intensity, (np.arange(11.) + 1) / 2)


@pytest.mark.parametrize("mode", ("Sum", "Average"))
def test_curves_on_different_grids_fold_on_one_grid_spanning_them(mode):
    """Two scans' ranges overlapping on 2-3, one with other Pts: below 2 and above 3 each is shown as it is,
    between them both are added (Sum) or averaged (Average); the grid is as fine as the finer curve."""
    low = _trace(1, np.linspace(1., 3., 201), np.linspace(10., 30., 201) ** 1.5)
    high = _trace(2, np.linspace(2., 4., 51), np.cos(np.linspace(2., 4., 51)) + 5)
    (folded,) = aggregate_traces((low, high), mode)
    grid = folded.axis.values
    assert (grid[0], grid[-1], grid.size) == (1., 4., 301)  # 0.01 apart, the finer curve's
    assert (folded.axis.label, folded.axis.unit, folded.title, folded.frame) == ("Q", "Å⁻¹", mode, low.frame)
    np.testing.assert_allclose(folded.intensity, _expected((low, high), grid, mode), rtol=1e-12)
    alone = grid < 2.
    np.testing.assert_allclose(folded.intensity[alone], np.interp(grid[alone], low.axis.values, low.intensity),
                               rtol=1e-12)
    # the order of the curves and of their axes does not change the fold (bar the first's frame)
    backwards = _trace(3, high.axis.values[::-1], high.intensity[::-1])
    np.testing.assert_allclose(aggregate_traces((backwards, low), mode)[0].intensity, folded.intensity, rtol=1e-12)


def test_disjoint_ranges_show_each_curve_and_nothing_between():
    left = _trace(1, np.linspace(0., 1., 11), np.full(11, 2.))
    right = _trace(2, np.linspace(2., 3., 11), np.full(11, 5.))
    (summed,) = aggregate_traces((left, right), "Sum")
    grid, values = summed.axis.values, summed.intensity
    assert np.all(values[grid <= 1.] == 2.) and np.all(values[grid >= 2.] == 5.)
    assert np.all(np.isnan(values[(grid > 1. + 1e-9) & (grid < 2. - 1e-9)]))


def test_a_missing_value_leaves_that_curve_out_where_it_is_missing():
    axis = np.linspace(0., 1., 11)
    gappy = np.ones(11)
    gappy[5] = np.nan
    traces = (_trace(1, axis, gappy), _trace(2, np.linspace(0., 1., 21), np.full(21, 3.)))
    (average,) = aggregate_traces(traces, "Average")
    near = np.abs(average.axis.values - .5) < .06  # the finer grid's points beside the gap
    assert np.all(average.intensity[near] == 3.) and np.all(average.intensity[~near] == 2.)


def test_curves_on_different_axes_show_the_first():
    q = _trace(1, np.linspace(1., 2., 11), np.ones(11))
    two_theta = _trace(2, np.linspace(10., 20., 11), np.ones(11), label="2θ", unit="°")
    assert not foldable((q, two_theta)) and aggregate_traces((q, two_theta), "Sum") == (q,)
    unordered = _trace(3, [1., 1.5, 1.2, 2.], np.ones(4))  # not monotonic: no grid to put it on
    assert not foldable((q, unordered)) and aggregate_traces((q, unordered), "Sum") == (q,)
    assert foldable((q, _trace(4, np.linspace(0., 3., 7), np.ones(7)))) and foldable(())


def test_the_grid_stays_bounded():
    """At most 50,000 points (never fewer than the longest curve's), and 20 million interpolated values in all."""
    fine = _trace(1, np.linspace(0., 1e-3, 1001), np.ones(1001))
    wide = _trace(2, np.linspace(0., 100., 11), np.ones(11))
    assert aggregate_traces((fine, wide), "Sum")[0].axis.values.size == 50_000
    many = tuple(_trace(i, np.linspace(0., 1. + i, 2_000), np.ones(2_000)) for i in range(1, 1_001))
    assert aggregate_traces(many, "Sum")[0].axis.values.size == 20_000
