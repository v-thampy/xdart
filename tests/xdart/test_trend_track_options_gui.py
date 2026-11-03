"""NEXT-128-WINDOW-POLISH item 14, Track options in the trend (the maintainer, 2026-11-03: "the track options should
also show the other peak pars like fraction for PV and height (even though it is derived), as well as the peak
independent bg pars"): Track and Track 2 list every family; a background family is drawn whatever peaks the chooser
shows, and the chooser lists peaks only."""
from __future__ import annotations

import pytest

pytest.importorskip("pyqtgraph")

from pyqtgraph.Qt import QtWidgets


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _curves(view):
    return [item.name() for item in view.plot.getPlotItem().listDataItems()]


def test_track_lists_every_family_and_a_background_family_ignores_the_peak_chooser(qapp):
    from xdart.gui.tabs.scattering.trend_view import TrendView
    view = TrendView(None, name="trackOptions")
    rows = [(str(frame), float(frame), {"center_0": 2.5, "center_1": 3.1, "fwhm_0": .07, "fwhm_1": .09,
                                        "height_0": 600., "height_1": 300., "amplitude_0": 40., "amplitude_1": 30.,
                                        "sigma_0": .03, "sigma_1": .04, "fraction_0": .4, "fraction_1": .5,
                                        "bg_slope": 2. + frame, "bg_intercept": 20.})
            for frame in range(3)]
    view.set_rows(rows)
    texts = [view.family.itemText(index) for index in range(view.family.count())]
    assert texts == ["Peak center", "FWHM", "Height", "Amplitude", "σ", "Lorentzian fraction", "Background slope",
                     "Background intercept"]
    assert [view.right_family.itemText(index) for index in range(view.right_family.count())] == ["None", *texts]
    assert view._members == (0, 1)  # the chooser lists the peaks, never a background "peak"
    view.family.setCurrentIndex(view.family.findData("bg_slope"))
    view.redraw()
    assert _curves(view) == ["bg_slope"]
    assert view.plot.getPlotItem().getAxis("left").labelText == "Background slope"
    view._show_members({0, 1})  # every peak hidden: the background is still drawn
    assert _curves(view) == ["bg_slope"]
    view.family.setCurrentIndex(view.family.findData("fraction"))
    view.redraw()
    assert _curves(view) == []
    view._show_members(set())
    assert _curves(view) == ["fraction_0", "fraction_1"]
    assert view.plot.getPlotItem().getAxis("left").labelText == "Lorentzian fraction"
    view.family.setCurrentIndex(view.family.findData("sigma"))
    view.redraw()
    view.set_value_unit("Å⁻¹")
    view.redraw()
    assert view.plot.getPlotItem().getAxis("left").labelText == "σ (Å⁻¹)"
