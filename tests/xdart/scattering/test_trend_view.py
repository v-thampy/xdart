"""The shared trend (``tabs/scattering/trend_view.py``; FIT-TOOLS-2026-10, Codex's design review F4, F7): fitted
parameters across results, rows keyed by result, a second family on the right-hand axis that the Scan Plot shares.
Int's fit dialogs feed it (``test_fit_tools_preservation.py``); the Scan Plot draws its right axis through it."""
from __future__ import annotations

import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.tabs.scattering.trend_view import TrendView


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _curves(view):
    return [(item.name(), list(item.getData()[0]), list(item.getData()[1]))
            for item in view.plot.getPlotItem().listDataItems()]


def _choose(combo, family):
    combo.setCurrentIndex(combo.findData(family))


def test_rows_are_keyed_by_result_drawn_by_x_repeats_kept_and_an_unavailable_x_listed_not_drawn(qapp):
    """Rows with and without an x, mixed: those with one sorted by it (ties in the results' order), the others
    after them, never compared with a number."""
    view = TrendView()
    try:
        assert not view.family.isEnabled() and not view.save.isEnabled() and not view.overlay.isEnabled()
        view.set_row("b", 2.0, {"center_0": 2.1})
        view.set_row("a", 1.0, {"center_0": 2.0})
        view.set_row("c", 2.0, {"center_0": 2.2})  # another result at the same x: its own row
        view.set_row("d", None, {"center_0": 9.9})  # no x for this result
        view.set_row("a", 1.0, {"center_0": 2.05})  # a re-fit updates its row
        view.set_row("e", 3.0, {})  # nothing fitted: nothing added
        assert view.table() == (("a", "b", "c", "d"), (1.0, 2.0, 2.0, None), {"center_0": [2.05, 2.1, 2.2, 9.9]})
        assert view.family.isEnabled() and view.save.isEnabled() and view.family.currentData() == "center"
        assert _curves(view) == [("center_0", [1.0, 2.0, 2.0], [2.05, 2.1, 2.2])]
        assert view.note.isVisibleTo(view) and view.note.text() == "1 result(s) without frame are listed, not drawn"
        view.clear()
        assert view.table() == ((), (), {}) and not _curves(view) and not view.family.isEnabled()
        assert [view.right_family.itemData(i) for i in range(view.right_family.count())] == [None]
    finally:
        view.deleteLater()


def test_in_result_order_the_x_is_each_results_place(qapp):
    view = TrendView(order=True, x_label="Result", x_name="result")
    try:
        for identity in ("scan 9", "scan 2", "scan 5"):
            view.set_row(identity, None, {"amplitude_0": float(len(identity))})
        assert view.table()[:2] == (("scan 9", "scan 2", "scan 5"), (0.0, 1.0, 2.0))
        assert _curves(view)[0][1] == [0.0, 1.0, 2.0] and not view.note.isVisibleTo(view)
    finally:
        view.deleteLater()


def test_a_second_family_goes_on_the_right_axis_once_and_none_turns_it_off(qapp):
    """The fitting trend's right axis (moved from ParamTrendMixin's own rows, test_param_trend_right_axis)."""
    view = TrendView()
    try:
        view.set_value_unit("q (Å⁻¹)")
        view.set_row("0", 0, {"center_0": 2.0, "amplitude_0": 1.0e5})
        view.set_row("1", 1, {"center_0": 2.1, "amplitude_0": 1.1e5})
        assert view.family.findData("center") >= 0 and view.right_family.findData(None) == 0
        assert view.right_family.findData("amplitude") >= 0
        _choose(view.family, "center")
        _choose(view.right_family, "amplitude")
        assert [name for name, *_xy in _curves(view)] == ["center_0"]
        assert [item.name() for item in view.right_view_box.addedItems] == ["amplitude_0"]
        assert view.right_view_box.addedItems[0].opts["pen"].style() == QtCore.Qt.PenStyle.DashLine
        assert "amplitude_0 (R)" in [label.text for _sample, label in view.legend.items]
        assert view.right_axis.isVisible() and view.right_axis.labelText == "Amplitude (Intensity)"
        assert view.plot.getPlotItem().getAxis("left").labelText == "Peak center (q (Å⁻¹))"
        _choose(view.right_family, "center")  # the same family on both: drawn once, on the left
        assert [name for name, *_xy in _curves(view)] == ["center_0"] and not view.right_view_box.addedItems
        _choose(view.right_family, None)
        assert not view.right_view_box.addedItems and not view.right_axis.isVisible()
    finally:
        view.deleteLater()


def test_overlay_draws_every_peak_of_the_family_and_a_choice_survives_new_rows(qapp):
    view = TrendView()
    try:
        view.set_row("0", 0, {"center_0": 1.0, "center_1": 2.0, "fwhm_0": 0.1, "fwhm_1": 0.2})
        _choose(view.family, "fwhm")
        assert [name for name, *_xy in _curves(view)] == ["fwhm_0"]
        view.overlay.setChecked(True)
        assert [name for name, *_xy in _curves(view)] == ["fwhm_0", "fwhm_1"]
        view.set_row("1", 1, {"center_0": 1.1, "center_1": 2.1, "fwhm_0": 0.11, "fwhm_1": 0.21})
        assert view.family.currentData() == "fwhm" and len(_curves(view)) == 2
    finally:
        view.deleteLater()


def test_save_csv_writes_the_rows_through_the_shared_browse_rule(qapp, monkeypatch, tmp_path):
    view = TrendView()
    messages, asked = [], []
    view.messageChanged.connect(messages.append)
    target = tmp_path / "trend.csv"

    def save_dialog(*args):
        asked.append(args)
        return str(target), "CSV"

    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", save_dialog)
    try:
        view.set_row("1", 1, {"center_0": 2.1})
        view.set_row("0", 0, {"center_0": 2.0, "amplitude_0": 1.0e5})
        view.save.click()
        assert asked[0][1] == "Save parameters vs frame" and asked[0][2].endswith("fit_params_vs_frame.csv")
        assert asked[0][3] == "CSV files (*.csv)"
        assert target.read_bytes() == b"frame,center_0,amplitude_0\r\n0,2.0,100000.0\r\n1,2.1,nan\r\n"
        assert messages == [f"Saved {target}"]
        monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                            lambda *args: (str(tmp_path / "missing" / "trend.csv"), "CSV"))
        view.save.click()
        assert messages[-1] == "Could not save CSV (see log)."
    finally:
        view.deleteLater()


def test_the_scan_plot_draws_its_right_axis_through_the_trends_helpers():
    from xdart.gui.analysis import plot_axes
    from xdart.gui.tabs.scattering import trend_view
    assert (plot_axes.attach_right_axis, plot_axes.add_right_series, plot_axes.set_right_axis_visible) == (
        trend_view.attach_right_axis, trend_view.add_right_series, trend_view.set_right_axis_visible)
