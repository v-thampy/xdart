# -*- coding: utf-8 -*-
"""The fitting trend (ParamTrendMixin) gains a second, right-hand-axis family
selector so cross-family parameters (e.g. peak center ~2 vs amplitude ~1e5)
overlay cleanly.  Offscreen GUI; a pyqtgraph teardown SIGSEGV is a known flake —
just rerun."""
import pytest


@pytest.fixture(scope="module")
def qapp():
    from pyqtgraph.Qt import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_param_trend_second_family_on_right_axis(qapp):
    from xdart.gui.tabs.static_scan.peak_fit_dialog import PeakFitDialog

    dlg = PeakFitDialog(lambda: None)          # no provider needed for the trend
    try:
        # two frames, two families: center (x-unit) + amplitude (intensity).
        dlg._accumulate_frame_params(0, {"center_0": 2.0, "amplitude_0": 1.0e5})
        dlg._accumulate_frame_params(1, {"center_0": 2.1, "amplitude_0": 1.1e5})

        # both families offered; the right combo also has a None sentinel.
        assert dlg.param_family_combo.findData("center") >= 0
        assert dlg.param_family_combo2.findData(None) == 0
        assert dlg.param_family_combo2.findData("amplitude") >= 0

        # left = center, right = amplitude.
        dlg.param_family_combo.setCurrentIndex(
            dlg.param_family_combo.findData("center"))
        dlg.param_family_combo2.setCurrentIndex(
            dlg.param_family_combo2.findData("amplitude"))
        dlg._redraw_param_plot()

        left = {it.name() for it in dlg.param_plot.getPlotItem().listDataItems()}
        assert "center_0" in left and "amplitude_0" not in left
        assert len(dlg.param_right_vb.addedItems) == 1
        assert dlg.param_right_vb.addedItems[0].name() == "amplitude_0"

        # turning the right family off clears the right ViewBox.
        dlg.param_family_combo2.setCurrentIndex(
            dlg.param_family_combo2.findData(None))
        dlg._redraw_param_plot()
        assert len(dlg.param_right_vb.addedItems) == 0
    finally:
        dlg.close()


def test_param_trend_right_skips_when_same_as_left(qapp):
    """Picking the SAME family on both axes draws it once (left only)."""
    from xdart.gui.tabs.static_scan.peak_fit_dialog import PeakFitDialog

    dlg = PeakFitDialog(lambda: None)
    try:
        dlg._accumulate_frame_params(0, {"center_0": 2.0})
        dlg._accumulate_frame_params(1, {"center_0": 2.1})
        dlg.param_family_combo.setCurrentIndex(
            dlg.param_family_combo.findData("center"))
        dlg.param_family_combo2.setCurrentIndex(
            dlg.param_family_combo2.findData("center"))
        dlg._redraw_param_plot()
        assert len(dlg.param_plot.getPlotItem().listDataItems()) == 1
        assert len(dlg.param_right_vb.addedItems) == 0
    finally:
        dlg.close()
