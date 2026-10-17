"""The shared fit display (Codex's Stitch/Int reuse audit R2: "Extend the scientific display owner for
fit/data/residual panels, so both Stitch diagnostics and Int's analysis fit windows can use the same
display capability"). Its consumers' own rows pin what each one shows: Stitch's fit windows in
``test_stitch_calibration_fit_dialog.py`` and ``test_stitch_guided_calibration.py``, the phase-fit dialog in
``test_p37_analysis_mount.py``."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.gui_utils import RectViewBox
from xdart.gui.tabs.scattering.fit_view import (
    FitPanel, FitProjection, FitSeries, FitView, draw_series,
)
from xdart.gui.widgets.scientific_plot import trace_colors


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _brush(item):
    return item.opts["symbolBrush"]


def test_a_fit_projection_draws_data_model_groups_and_residuals_on_the_1d_plots_conventions() -> None:
    _app()
    view = FitView(link_x=True)
    try:
        x = np.arange(5.0)
        projection = FitProjection(
            upper=FitPanel((FitSeries(x, x ** 2, "data", "data"), FitSeries(x, x ** 2 + .1, "fit", "fit"),
                            FitSeries(x[:2], x[:2], "group", "row 0", 0), FitSeries(x[2:], x[2:], "group", "row 1", 1),
                            FitSeries(x[:1], x[:1], "rejected", "rejected")),
                           x_label="angle (°)", y_label="centre (px)", title="a title", markers=(1.5, 2.5)),
            lower=FitPanel((FitSeries(x, x * 0, "residual"),), x_label="angle (°)", y_label="residual",
                           zero_line=True),
            summary="Direct-beam distance: 1 mm")
        view.reconcile(projection)
        items = view.plot.listDataItems()
        assert [item.name() for item in items] == ["data", "fit", "row 0", "row 1", "rejected"]
        np.testing.assert_array_equal(items[1].yData, x ** 2 + .1)
        colours = trace_colors("Default", 2)
        assert [tuple(pg.mkBrush(_brush(item)).color().getRgb()[:3]) for item in items[2:4]] == list(colours)
        assert items[4].opts["symbol"] == "x" and items[0].opts["pen"] is None
        assert view.plot.getPlotItem().titleLabel.text == "a title"
        assert view.plot.getPlotItem().getAxis("bottom").labelText == "angle (°)"
        markers = [item for item in view.plot.getPlotItem().items if isinstance(item, pg.InfiniteLine)]
        assert sorted(line.value() for line in markers) == [1.5, 2.5]
        zero = [item for item in view.residuals.getPlotItem().items if isinstance(item, pg.InfiniteLine)]
        assert [line.value() for line in zero] == [0.0] and len(view.residuals.listDataItems()) == 1
        assert view.summary.text() == "Direct-beam distance: 1 mm" and view.projection is projection
        # the 1-D plot's conventions: box zoom (its view box), the seaborn grid, the readout on the upper panel
        assert isinstance(view.plot.getPlotItem().vb, RectViewBox)
        assert isinstance(view.residuals.getPlotItem().vb, RectViewBox)
        assert view._readout.curve is view.plot
        # a reconcile replaces everything: no item of the last projection stays
        view.reconcile(FitProjection(lower=FitPanel(zero_line=True)))
        assert not view.plot.listDataItems() and not view.residuals.listDataItems()
        assert not [item for item in view.plot.getPlotItem().items if isinstance(item, pg.InfiniteLine)]
        assert view.plot.getPlotItem().getAxis("bottom").labelText == ""
    finally:
        view.close()
        view.deleteLater()


def test_a_series_role_is_its_look_and_an_unknown_role_is_refused() -> None:
    _app()
    with pytest.raises(ValueError, match="unknown fit series role"):
        FitSeries(np.zeros(1), np.zeros(1), "scatter")
    plot = pg.PlotWidget()  # retired by the Qt test boundary (a second PlotWidget.close() raises)
    # the same primitive draws on any plot (the analysis dialogs' older path draws through it)
    fit = draw_series(plot, FitSeries(np.arange(3.0), np.ones(3), "fit", "fit"))
    background = draw_series(plot, FitSeries(np.arange(3.0), np.zeros(3), "background", "background"))
    assert fit.opts["pen"].width() == 2 and background.opts["pen"].style() != fit.opts["pen"].style()
    assert [item.name() for item in plot.listDataItems()] == ["fit", "background"]


def test_the_phase_fit_dialog_shows_its_pattern_fit_and_residual_through_the_shared_fit_display() -> None:
    """The second consumer: the phase-fit dialog's two page-built plots (a data and fit plot, an x-linked
    residual plot) are the display's panels, drawn only through ``reconcile``. Its result rows are pinned in
    ``test_p37_analysis_mount.py``."""
    _app()
    from xdart.gui.analysis.phase_fit_dialog import PhaseFitDialog
    dialog = PhaseFitDialog(analysis_context=None)
    try:
        assert isinstance(dialog.fit_view, FitView)
        assert (dialog.plot, dialog.resid_plot) == (dialog.fit_view.plot, dialog.fit_view.residuals)
        assert isinstance(dialog.plot.getPlotItem().vb, RectViewBox)
        x = np.linspace(1.0, 3.0, 5)
        dialog.set_live_pattern(x, x ** 2, "q (Å⁻¹)")
        shown = dialog.fit_view.projection
        assert [series.role for series in shown.upper.series] == ["data_line"]
        assert (shown.upper.x_label, shown.upper.y_label) == ("q (Å⁻¹)", "Intensity")
        assert shown.lower.zero_line and not shown.lower.series
        overlay = SimpleNamespace(x=x, traces={"fit": x ** 2 + .5, "residual": x * 0})
        dialog._draw_outcome(SimpleNamespace(overlay=overlay, result=SimpleNamespace(payload=[]),
                                             label="current", params={}))
        shown = dialog.fit_view.projection
        assert [(series.role, series.name) for series in shown.upper.series] == [
            ("data_line", "data"), ("fit", "fit")]
        assert [series.role for series in shown.lower.series] == ["residual"]
        assert shown.lower.y_label == "resid"
        assert [item.name() for item in dialog.plot.listDataItems()] == ["data", "fit"]
        dialog._clear_fit()
        assert not dialog.resid_plot.listDataItems() and dialog.fit_view.projection.upper == shown.upper
    finally:
        dialog.close()


def test_the_fit_range_is_the_views_kept_over_every_redraw_and_a_drag_says_where_it_went() -> None:
    """FIT-TOOLS-2026-10 (Codex's design review F7): the fit range is a FitView capability, not a page's plot. Shown
    within the data's extent and clamped to it, kept over each redraw, hidden with None; setting it is silent, a drag
    of one edge emits the range, sorted."""
    _app()
    view = FitView()
    moved = []
    view.rangeChanged.connect(lambda low, high: moved.append((low, high)))
    try:
        region = view._region
        assert view.fit_range is None and not region.isVisible()
        view.set_fit_range((3.0, 1.5), limits=(1.0, 4.0))
        assert view.fit_range == (1.5, 3.0) and region.isVisible() and region in view.plot.getPlotItem().items
        view.set_fit_range((0.0, 9.0), limits=(1.0, 4.0))
        assert view.fit_range == (1.0, 4.0) and not moved
        x = np.linspace(1.0, 4.0, 7)
        view.reconcile(FitProjection(FitPanel((FitSeries(x, x, "data", "data"),), markers=(2.0,))))
        assert view.fit_range == (1.0, 4.0) and region in view.plot.getPlotItem().items and region.isVisible()
        assert [item.name() for item in view.plot.listDataItems()] == ["data"]
        markers = [item for item in view.plot.getPlotItem().items if isinstance(item, pg.InfiniteLine)]
        assert [line.value() for line in markers] == [2.0]
        region.lines[1].setValue(2.5)  # the user drags the upper edge
        assert view.fit_range == (1.0, 2.5) and moved[-1] == (1.0, 2.5)
        region.lines[0].setValue(3.0)  # past the other edge: still low, high
        assert moved[-1] == (2.5, 3.0)
        view.set_fit_range(None)
        assert view.fit_range is None and not region.isVisible()
        count = len(moved)
        region.lines[0].setValue(1.2)  # a hidden range is no fit range
        assert len(moved) == count
    finally:
        view.close()
        view.deleteLater()


def test_a_displayed_fit_result_is_the_views_series_fit_background_components_markers_and_residual() -> None:
    from xdart.gui.tabs.scattering.fit_view import fit_result_series
    x = np.arange(3.0)
    result = SimpleNamespace(fit=x + 1, background=x * 0, residual=x - 1, marker_positions=np.array([1.0, 2.0]))
    upper, markers, lower = fit_result_series(x, result, (("alpha", x * 2),))
    assert [(series.role, series.name) for series in upper] == [
        ("fit", "fit"), ("background", "background"), ("component", "alpha")]
    assert markers == (1.0, 2.0) and [series.role for series in lower] == ["residual"]
    empty = SimpleNamespace(fit=None, background=None, residual=None, marker_positions=None)
    assert fit_result_series(x, empty) == ((), (), ())
