"""NEXT-128-PEAK-FIT, the maintainer's popup items (2026-10-31), on the shared owners (hard rule 14): the Peak Fitter
and Phase Analysis follow the main window's theme and Int's display conventions (Claude, 2026-10-31).

His words: "the tool popup windows should follow the same theming/styling as the main window - so the color of the
Fit button for example. also the x-axis label is q_A^-1 instead of the label the main window uses with the unicode
for Angstrom for example. additionallly it will be good to have a label row above the plot to show the label and also
the other options shown in int 1d for example (the color map and log buttons)", and "for the top plot, the raw data
should be shown with markers (same marker size as the main window) and the fits with solid lines. there should also
be a toggle button in the middle row that has the track etc option to show the different fit components".

Frozen before the change: red on 1aa5cde1 at the new behaviour. Real pages, the page's analysis worker and real fits;
the theme rows read the one stylesheet renderer.
"""
import re

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from tests.xdart.scattering.test_fit_tools_preservation import _close_page, _page, _peaks, _settle, _show, _trace


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def isolated_library(tmp_path, monkeypatch):
    monkeypatch.setenv("XDART_PHASE_LIBRARY", str(tmp_path / "phase-library"))


def _rule(qss, selector):
    """The properties of the first rule whose selector list names ``selector`` exactly."""
    qss = re.sub(r"/\*.*?\*/", "", qss, flags=re.S)
    for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", qss):
        if selector in (part.strip() for part in selectors.split(",")):
            return dict((key.strip(), value.strip()) for key, _sep, value in
                        (line.partition(":") for line in body.split(";") if ":" in line))
    raise AssertionError(f"no rule for {selector}")


def _names(dialog):
    return [item.name() for item in dialog.plot.listDataItems()]


def _fit(qapp, page, dialog):
    dialog.fit_btn.click()
    assert page._consume_analysis_update(_settle(page))
    qapp.processEvents()


# ---- S1 the main window's theme -----------------------------------------------------------------------------

def test_s1_the_fit_button_takes_the_main_windows_selection_accent():
    from xdart.gui.themes import render_qss
    from xdart.gui.themes.accent import ACCENT_COLOR_NAMES
    for theme in ("dark", "light"):
        for accent in ACCENT_COLOR_NAMES:
            qss = render_qss(theme, accent_color=accent)
            fit, checked = _rule(qss, "QPushButton#peakFitGo"), _rule(qss, "QPushButton:checked")
            assert (fit["background-color"], fit["color"]) == (checked["background-color"], checked["color"]), (
                theme, accent)
            hover, checked_hover = _rule(qss, "QPushButton#peakFitGo:hover"), _rule(qss, "QPushButton:checked:hover")
            assert hover["background-color"] == checked_hover["background-color"], (theme, accent)


# ---- S2 to S5 on Int's page ---------------------------------------------------------------------------------

def test_s2_s3_both_tools_label_their_axes_as_int_does_under_ints_top_bar(qapp, monkeypatch):
    from xdart.gui.tabs.scattering.shell_widgets import _axis_presentation, color_map_choices
    page = _page()
    try:
        trace, anchor = _trace(*_peaks())
        label, unit = _axis_presentation(trace.label, trace.axis_unit)
        assert unit == "Å⁻¹"
        for kind in ("peak", "phase"):
            dialog = _show(page, monkeypatch, kind, trace, anchor)
            qapp.processEvents()
            axis = dialog.plot.getPlotItem().getAxis("bottom")
            assert (axis.labelText, axis.labelUnits) == (label, unit), kind
            assert dialog.trend._value_unit == unit, kind
            view = dialog.fit_view
            assert [view.color_map.itemText(i) for i in range(view.color_map.count())] == list(color_map_choices())
            assert view.log_scale.isCheckable() and view.log_scale.text() == "Log"
            assert trace.title in view.title.text(), (kind, view.title.text())
            for widget in (view.title, view.color_map, view.log_scale):
                assert widget.toolTip(), (kind, widget)
            top = view.title.mapTo(view, QtCore.QPoint(0, 0)).y()
            assert top < view.plot.mapTo(view, QtCore.QPoint(0, 0)).y(), kind  # the bar sits above the plot
    finally:
        _close_page(page, qapp)


def test_s3_s4_log_draws_the_upper_panel_on_one_scale_and_the_data_are_ints_markers(qapp, monkeypatch):
    from xdart.gui.widgets.scientific_plot import data_curve_style, scaled_intensity
    page = _page()
    try:
        trace, anchor = _trace(*_peaks())
        dialog = _show(page, monkeypatch, "peak", trace, anchor)
        _fit(qapp, page, dialog)
        data, fit, background = dialog.plot.listDataItems()
        int_size = data_curve_style((0, 0, 0))["symbolSize"]
        assert data.opts["symbol"] == "o" and data.opts["symbolSize"] == int_size and data.opts["pen"] is None
        assert fit.opts["symbol"] is None and fit.opts["pen"].style() == QtCore.Qt.PenStyle.SolidLine
        linear = [item.getData()[1] for item in (data, fit, background)]
        residual = dialog.resid_plot.listDataItems()[0].getData()[1]
        dialog.fit_view.log_scale.setChecked(True)
        qapp.processEvents()
        logged = [item.getData()[1] for item in dialog.plot.listDataItems()]
        joined = scaled_intensity(np.concatenate(linear), "Log")  # Int's Log, one shift for the whole panel
        np.testing.assert_allclose(np.concatenate(logged), joined)
        assert dialog.plot.getPlotItem().getAxis("left").labelText == "Log Intensity"
        np.testing.assert_array_equal(dialog.resid_plot.listDataItems()[0].getData()[1], residual)  # stays linear
        dialog.fit_view.log_scale.setChecked(False)
        qapp.processEvents()
        np.testing.assert_array_equal(dialog.plot.listDataItems()[1].getData()[1], linear[1])
        phase = _show(page, monkeypatch, "phase", trace, anchor)
        qapp.processEvents()
        (points,) = phase.plot.listDataItems()
        assert points.name() == "data" and points.opts["symbol"] == "o" and points.opts["pen"] is None
        assert points.opts["symbolSize"] == int_size
    finally:
        _close_page(page, qapp)


def test_s5_a_components_toggle_in_the_trend_row_shows_each_fitted_peak(qapp, monkeypatch):
    from lmfit import Parameters
    from xrd_tools.analysis.fitting.fit import get_peak_model
    page = _page()
    try:
        trace, anchor = _trace(*_peaks())
        dialog = _show(page, monkeypatch, "peak", trace, anchor)
        toggle = dialog.fit_view.components_toggle
        assert toggle.isCheckable() and not toggle.isChecked() and toggle.toolTip()
        parent = toggle.parentWidget()
        while parent is not None and parent is not dialog.trend:
            parent = parent.parentWidget()
        assert parent is dialog.trend, "the toggle sits in the trend's row"
        row_y = dialog.trend.family_row.mapTo(dialog.trend, QtCore.QPoint(0, 0)).y()
        assert abs(toggle.mapTo(dialog.trend, QtCore.QPoint(0, 0)).y() - row_y) < toggle.height()
        _fit(qapp, page, dialog)
        result = page._peak_result
        assert _names(dialog) == ["data", "fit", "background"]
        toggle.setChecked(True)
        qapp.processEvents()
        assert _names(dialog) == ["data", "fit", "background", "p0", "p1", "p2"]
        items = dialog.plot.listDataItems()
        x = items[1].getData()[0]
        parameters = Parameters()
        for name, value in zip(result.parameter_names, result.parameter_values):
            parameters.add(name, value=value)
        for index, item in enumerate(items[3:]):
            expected = get_peak_model("pseudovoigt", prefix=f"p{index}_").eval(params=parameters, x=x)
            np.testing.assert_allclose(item.getData()[1], expected, rtol=1e-12, atol=0)
            assert item.opts["pen"].style() == QtCore.Qt.PenStyle.SolidLine  # the fits: solid lines
        total = sum(item.getData()[1] for item in items[3:]) + items[2].getData()[1]
        np.testing.assert_allclose(total, items[1].getData()[1], rtol=1e-9, atol=1e-9 * float(np.max(total)))
        toggle.setChecked(False)
        qapp.processEvents()
        assert _names(dialog) == ["data", "fit", "background"]
        phase = _show(page, monkeypatch, "phase", trace, anchor)
        assert phase.fit_view.components_toggle.isChecked()  # Phase Analysis keeps drawing its phases
        parent = phase.fit_view.components_toggle.parentWidget()
        while parent is not None and parent is not phase.trend:
            parent = parent.parentWidget()
        assert parent is phase.trend
    finally:
        _close_page(page, qapp)


def test_s5_peak_components_come_from_one_headless_owner():
    """The notebook plot and the fit windows evaluate a peak fit's components through one function."""
    from xrd_tools.analysis.fitting.fit import fit_peaks, peak_component_curves
    x = np.linspace(1.0, 3.0, 301)
    y = 5.0 + 40.0 * np.exp(-0.5 * ((x - 1.8) / 0.03) ** 2) + 25.0 * np.exp(-0.5 * ((x - 2.3) / 0.04) ** 2)
    fit = fit_peaks(x, y, positions=[1.8, 2.3], model="gaussian", background="constant")
    names = tuple(fit.params)
    values = tuple(float(fit.params[name].value) for name in names)
    curves = peak_component_curves("gaussian", names, values, x)
    assert [name for name, _values in curves] == ["p0", "p1"]
    background = fit.fit_result.eval_components(x=x)["bg_"]
    np.testing.assert_allclose(sum(values for _name, values in curves) + background, fit.best_fit, rtol=1e-9)
