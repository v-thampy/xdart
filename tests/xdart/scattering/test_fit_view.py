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
