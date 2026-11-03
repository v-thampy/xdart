"""NEXT-128-WINDOW-POLISH item 14, fit-line contrast (the maintainer's live check, 2026-11-03: "the fit lines are now
very hard to see"): PF4 darkened the data points (his "Dark slate (Recommended)") and left the overlays in colours
chosen for a dark plot, while every fit plot is drawn on the light seaborn panel (``SEABORN_BG``) in both themes. Each
overlay's colour now reads on that panel (WCAG 1.4.11: at least 3:1 for a graphical object; the fit, its background
and the residual at least 4.5:1), the lines are wider, the fit range's edges and the peak markers are coloured for the
panel, and the trend's curves take a palette made for it."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyqtgraph")

from pyqtgraph.Qt import QtWidgets


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _luminance(colour):
    def channel(value):
        value = value / 255.0
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
    red, green, blue = (channel(value) for value in colour[:3])
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _ratio(first, second):
    high, low = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _panel():
    from xdart.gui.themes.dark import SEABORN_BG
    return tuple(int(SEABORN_BG[index:index + 2], 16) for index in (1, 3, 5))


def test_every_overlay_reads_on_the_plots_panel_and_stands_apart_from_the_data(qapp):
    from xdart.gui.tabs.scattering import fit_view
    panel = _panel()
    for role, (kind, colour, size) in fit_view._ROLES.items():
        if colour == "group":
            continue
        assert colour is not None, f"{role}: an explicit colour, not the theme's foreground (the data's colour)"
        floor = 4.5 if role in {"fit", "background", "residual", "data", "data_line"} else 3.0
        assert _ratio(colour, panel) >= floor, (role, colour, round(_ratio(colour, panel), 2))
        if role not in {"data", "data_line"}:
            assert colour != fit_view._DATA and _ratio(colour, fit_view._DATA) >= 1.5, role
    assert fit_view._ROLES["fit"][2] >= 2.5 and fit_view._ROLES["background"][2] >= 2.0
    assert fit_view._ROLES["residual"][2] >= 1.5 and fit_view._ROLES["component"][2] >= 1.5
    assert _ratio(fit_view._MARKER, panel) >= 3.0


def test_the_fit_ranges_edges_and_a_drawn_fit_use_the_panels_colours(qapp):
    from xdart.gui.tabs.scattering.fit_view import FitPanel, FitProjection, FitSeries, FitView, _ROLES
    panel = _panel()
    view = FitView(None, name="contrastView", top_bar=True)
    x = np.linspace(1., 4., 301)
    y = 20 + 500 * np.exp(-.5 * ((x - 2.5) / .03) ** 2)
    view.reconcile(FitProjection(FitPanel((FitSeries(x, y, "data", "data"), FitSeries(x, y, "fit", "fit"),
                                           FitSeries(x, 0 * x + 20, "background", "background")))))
    view.set_fit_range((2.0, 3.0), limits=(1., 4.))
    for line in view._region.lines:
        colour = line.pen.color().getRgb()[:3]
        assert _ratio(colour, panel) >= 3.0, colour
        assert line.pen.widthF() >= 1.5
    drawn = {item.name(): item for item in view.plot.getPlotItem().listDataItems()}
    pen = drawn["fit"].opts["pen"]
    assert pen.color().getRgb()[:3] == _ROLES["fit"][1] and pen.widthF() == _ROLES["fit"][2]


def test_the_trends_curves_read_on_the_panel(qapp):
    from xdart.gui.analysis.peak_fit_util import TREND_COLORS
    from xdart.gui.tabs.scattering.trend_view import TrendView
    panel = _panel()
    assert len(TREND_COLORS) >= 6 and len(set(TREND_COLORS)) == len(TREND_COLORS)
    assert all(_ratio(colour, panel) >= 3.0 for colour in TREND_COLORS), [
        (colour, round(_ratio(colour, panel), 2)) for colour in TREND_COLORS]
    view = TrendView(None, name="contrastTrend")
    view.set_rows([(str(frame), float(frame), {f"center_{peak}": 2.0 + peak for peak in range(8)})
                   for frame in range(3)])
    colours = [item.opts["pen"].color().getRgb()[:3] for item in view.plot.getPlotItem().listDataItems()]
    assert len(colours) == 8 and colours[:len(TREND_COLORS)] == list(TREND_COLORS)[:8]
