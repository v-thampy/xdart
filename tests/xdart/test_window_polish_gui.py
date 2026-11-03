"""NEXT-128-WINDOW-POLISH: the maintainer's Peak Fitting and Phase Analysis changes (2026-11-02), on the shared owners
(hard rule 14). His words: "make it like the phase analysis window (ss2) - controls on the left, plots on the right",
"put the Peaks and "Auto" (for detecting peaks in the same row and the question mark right most in the same row",
"question mark in the Range row also all the way to the right aligned with the question marks above", "the raw data
can barely be discerned. need darker more prominent colors to make it more visible", "Full should be the default and
the ranges displayed should be like they are for other default "auto " values - lighter that can be overwritten when
full is unselected. then when full is selected the full range should show again", "when it opens the left panel with
the controls is not fully expanded - the question marks are hidden" and "the height of the phases window can be
reduced by 50%". His clickable answers (2026-11-03): "Dark slate (Recommended)" and "Same toggle in both
(Recommended)".

Frozen before the product change: red on 034723d4. The geometry rows assert relations only, since fonts and themes
differ; they apply no application theme, which would restyle every later test in the process (the parent was red
without one too). The Full rows run on the real Int page and its fit route.
"""
import numpy as np
import pytest
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets

from tests.xdart.scattering.test_fit_tools_preservation import _close_page, _page, _peaks, _show, _trace

KINDS = ("peak", "phase")


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def isolated_library(tmp_path, monkeypatch):
    monkeypatch.setenv("XDART_PHASE_LIBRARY", str(tmp_path / "phase-library"))


@pytest.fixture
def dark(qapp):
    """The geometry rows' application (named for his dark screenshots); no theme is applied process-wide."""
    return qapp


def _dialog(kind):
    from xdart.gui.analysis.peak_fit_dialog import PeakFitDialog
    from xdart.gui.analysis.phase_fit_dialog import PhaseFitDialog
    return (PeakFitDialog if kind == "peak" else PhaseFitDialog)(vnext_submit=lambda *_args: None)


def _settle(qapp):
    for _ in range(4):
        qapp.processEvents()


def _first_card(dialog):
    from xdart.gui.widgets.controls_panel import SectionCard
    widget = dialog.model_row if hasattr(dialog, "model_row") else dialog.phases_card
    while widget is not None and not isinstance(widget, SectionCard):
        widget = widget.parentWidget()
    return widget


def _left(widget, ancestor):
    return widget.mapTo(ancestor, QtCore.QPoint(0, 0)).x()


def _right(widget, ancestor):
    return widget.mapTo(ancestor, QtCore.QPoint(widget.width(), 0)).x()


def _helps(widget):
    from xdart.gui.widgets.controls_panel import ActionButton
    return [button for button in widget.findChildren(ActionButton) if button.text() == "?" and button.isVisible()]


# ---- PF1 the Peak Fitter laid out as Phase Analysis: one shared split, options left, plots right -------------------

@pytest.mark.parametrize("kind", KINDS)
def test_pf1_both_fit_windows_mount_one_split_with_the_options_left_of_the_plots(dark, kind):
    from xdart.gui.tabs.scattering.tool_window import ToolWindowSplit
    dialog = _dialog(kind)
    try:
        dialog.show()
        _settle(dark)
        split = dialog.split
        assert type(split) is ToolWindowSplit, "the shared split owner, not a window's own splitter"
        card = _first_card(dialog)
        assert split.controls.isAncestorOf(card)
        for display in (dialog.fit_view, dialog.trend, dialog.table):
            assert split.displays.isAncestorOf(display), display
        assert _right(card, dialog) <= _left(dialog.fit_view, dialog), "the options sit left of the plots"
        top = lambda widget: widget.mapTo(dialog, QtCore.QPoint(0, 0)).y()
        assert top(card) < top(dialog.fit_view) + dialog.fit_view.height(), "side by side, not stacked"
        assert top(dialog.fit_view) < top(card) + card.height()
    finally:
        dialog.close()


# ---- PF2, PF3 Peaks and Auto in one row; every "?" rightmost in its row and in one column -------------------------

def test_pf2_pf3_peaks_and_auto_share_a_row_and_every_question_mark_is_one_column(dark):
    dialog = _dialog("peak")
    try:
        dialog.show()
        _settle(dark)
        card = _first_card(dialog)
        assert dialog.count_row.isAncestorOf(dialog.auto_check), "Auto sits in the Peaks row"
        rows = (dialog.model_row, dialog.bkg_row, dialog.background_template_row, dialog.count_row,
                dialog.range_row)
        rights = [_right(row.help_button, card) for row in rows]
        assert max(rights) - min(rights) <= 1, rights
        # each one-line row ends in its "?" (the template row's "?" ends its own button line, after a stretch)
        for row in (dialog.model_row, dialog.bkg_row, dialog.count_row, dialog.range_row):
            button = row.help_button
            for child in row.findChildren(QtWidgets.QWidget):
                if child.parentWidget() is row and child.isVisible() and child is not button:
                    assert _right(child, card) <= _left(button, card), (row, child, "the ? is rightmost")
        assert len(_helps(card)) == len(rows) + 1, "one ? per row and the card's own; no separate Auto row"
        assert _right(dialog.auto_check, card) <= _left(dialog.count_row.help_button, card)
        assert _right(dialog.full_btn, card) <= _left(dialog.range_row.help_button, card)
    finally:
        dialog.close()


# ---- PF6 the Peak Fitter's advanced options in its options column ----------------------------------------------------

def test_pf6_the_peak_fitters_advanced_options_sit_in_its_options_column(dark):
    """His addition, 2026-11-03: "i think there's enogh space in the peak fitting window left panel that the advanced
    options can be accommodated there". Frozen red on 80834b4d (they were the shared Advanced dialog's rows)."""
    from xdart.gui.tabs.scattering.advanced_editor import AdvancedDialog
    dialog = _dialog("peak")
    try:
        dialog.show()
        _settle(dark)
        assert not dialog.findChildren(AdvancedDialog) and not hasattr(dialog, "advanced_btn"), "no Advanced window"
        editors = (dialog.adv_centers, dialog.adv_sigma_init, dialog.adv_sigma_min, dialog.adv_sigma_max,
                   dialog.adv_center_delta, dialog.adv_fraction, dialog.adv_maxfev, dialog.use_uncertainties)
        column = dialog.split.controls.widget()
        assert all(column.isAncestorOf(editor) and editor.isVisible() for editor in editors)
        rows = {editor.parentWidget() for editor in editors}  # each editor's row; σ min and max share one
        assert len(rows) == 7
        rights = [_right(row.help_button, column) for row in rows] + [_right(dialog.model_row.help_button, column)]
        assert max(rights) - min(rights) <= 1, "the advanced rows' ? sit in the fit rows' ? column"
    finally:
        dialog.close()


# ---- PF4, PA2 the data points stand out on the plot background, in both themes and both windows ---------------------

def _luminance(color):
    def channel(value):
        value /= 255.0
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
    red, green, blue = (channel(float(value)) for value in color.getRgb()[:3])
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast(first, second):
    light, deep = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (light + 0.05) / (deep + 0.05)


def test_pf4_pa2_the_data_points_stand_out_on_the_plot_background(qapp):
    from xdart.gui.tabs.scattering.fit_view import FitSeries, draw_series
    from xdart.gui.themes.dark import SEABORN_BG  # both themes' plot background (apply_theme sets it for each)
    from xdart.gui.widgets.scientific_plot import trace_colors
    background = pg.mkColor(SEABORN_BG)
    reference = _contrast(pg.mkColor(trace_colors("Default", 1)[0]), background)  # Int's own 1-D trace
    x, y = _peaks()
    trace, _anchor = _trace(x, y)
    for kind in KINDS:
        dialog = _dialog(kind)
        try:
            dialog.set_vnext_trace(trace)
            (data,) = [item for item in dialog.plot.listDataItems() if item.name() == "data"]
            color = pg.mkBrush(data.opts["symbolBrush"]).color()
            assert _contrast(color, background) >= reference, (kind, color.getRgb())
            assert color.getRgb()[:3] == (52, 52, 64), "his choice: Dark slate (Recommended)"
        finally:
            dialog.close()
    plot = pg.PlotWidget()
    try:
        line = draw_series(plot, FitSeries(x, y, "data_line"))  # Stitch's and RSM's fit windows draw data lines
        assert line.opts["pen"].color().getRgb()[:3] == (52, 52, 64)
    finally:
        plot.deleteLater()


# ---- PF5 Full: on by default, the full range shown greyed, typed over once off, shown again once on ----------------

def _bounds(dialog, kind):
    values = dialog.vnext_fit_values()
    return values["fit_bounds"] if kind == "peak" else values["fit_range"]


def _range_editors(dialog, kind):
    return dialog.range_row.editors if kind == "peak" else dialog.fit_range_row.editors


@pytest.mark.parametrize("kind", KINDS)
def test_pf5_full_is_on_by_default_and_shows_the_full_range_again_without_clearing(qapp, monkeypatch, kind):
    whole = (1.0, 4.0) if kind == "peak" else None  # the Peak Fitter's bounds are clamped; Phase's None is the trace
    page = _page()
    try:
        trace, anchor = _trace(*_peaks())
        dialog = _show(page, monkeypatch, kind, trace, anchor)
        dialog.show()
        dialog.activateWindow()
        _settle(qapp)
        full, editors = dialog.full_btn, _range_editors(dialog, kind)
        low = editors[0]
        assert full.isCheckable() and full.isChecked() and full.text() == "Full" and full.toolTip()

        def shows_the_full_range(expected=("1", "4")):
            assert [editor.text() for editor in editors] == ["", ""]
            assert [editor.placeholderText() for editor in editors] == list(expected), "never cleared"
            assert not any(editor.isEnabled() for editor in editors) and all(editor.toolTip() for editor in editors)

        shows_the_full_range()
        assert _bounds(dialog, kind) == whole and dialog.fit_view.fit_range == (1.0, 4.0)

        full.click()  # off: the greyed values can be typed over
        assert not full.isChecked() and all(editor.isEnabled() for editor in editors)
        assert [editor.placeholderText() for editor in editors] == ["1", "4"] and _bounds(dialog, kind) == whole
        QtTest.QTest.mouseClick(low, QtCore.Qt.MouseButton.LeftButton)
        QtTest.QTest.keyClicks(low, "2")
        QtTest.QTest.keyClick(low, QtCore.Qt.Key.Key_Return)
        _settle(qapp)
        assert _bounds(dialog, kind) == (2.0, 4.0) and dialog.fit_view.fit_range == (2.0, 4.0)

        full.click()  # on again: the full range shows again
        assert full.isChecked()
        shows_the_full_range()
        assert _bounds(dialog, kind) == whole and dialog.fit_view.fit_range == (1.0, 4.0)

        dialog.fit_view._region.lines[0].setValue(2.5)  # a range dragged on the plot is no longer Full
        assert not full.isChecked() and low.isEnabled() and low.text() == "2.5"
        assert _bounds(dialog, kind) == (2.5, 4.0)

        full.click()
        narrow, _anchor = _trace(*_peaks(np.linspace(2.0, 4.0, 401)), label=4)
        monkeypatch.setattr(page, "_capture_analysis_trace", lambda _kind: (narrow, anchor))
        dialog.refresh_btn.click()  # Reload: a new trace while Full is on shows its own full range
        _settle(qapp)
        shows_the_full_range(("2", "4"))
        assert _bounds(dialog, kind) == ((2.0, 4.0) if kind == "peak" else None)
    finally:
        _close_page(page, qapp)


@pytest.mark.parametrize("kind", KINDS)
def test_pf5_saved_settings_keep_full_and_a_typed_range(qapp, monkeypatch, tmp_path, kind):
    keys = ("range_low", "range_high") if kind == "peak" else ("fit_range_low", "fit_range_high")
    page = _page()
    try:
        trace, anchor = _trace(*_peaks())
        dialog = _show(page, monkeypatch, kind, trace, anchor)
        full, editors = dialog.full_btn, _range_editors(dialog, kind)
        path = tmp_path / f"{kind}-settings.json"
        dialog._settings_path_chooser = lambda _action, _start: str(path)
        assert type(dialog.settings_io.save()).__name__ == "ActionCompleted"
        saved_full = dialog._settings_document(dialog)["fields"]
        assert (saved_full[keys[0]], saved_full[keys[1]]) == (None, None), "Full is saved as the whole trace"
        full.click()
        editors[0].setText("2")
        editors[0].editingFinished.emit()
        assert _bounds(dialog, kind) == (2.0, 4.0)
        assert type(dialog.settings_io.load()).__name__ == "ActionCompleted"
        assert full.isChecked() and [editor.text() for editor in editors] == ["", ""]
        assert [editor.placeholderText() for editor in editors] == ["1", "4"] and not editors[0].isEnabled()
        full.click()
        editors[0].setText("2")
        editors[0].editingFinished.emit()
        assert type(dialog.settings_io.save()).__name__ == "ActionCompleted"
        full.click()
        assert full.isChecked()
        assert type(dialog.settings_io.load()).__name__ == "ActionCompleted"
        assert not full.isChecked() and editors[0].isEnabled() and editors[0].text() == "2"
        assert editors[1].text() == "" and editors[1].placeholderText() == "4"
        assert _bounds(dialog, kind) == (2.0, 4.0)
    finally:
        _close_page(page, qapp)


@pytest.mark.parametrize("theme", ("dark", "light"))
def test_pf5_locked_hints_read_as_the_other_locked_values(theme):
    """Added after the freeze (test completion, disclosed): a disabled QLineEdit draws its placeholder at about half
    the contrast of its disabled text (offscreen dark theme: lightness 49 against 98 for the Peaks count beside it),
    so Full's greyed range hardly showed. The owner's theme gives a locked field's hint the locked text's colour."""
    import re
    from xdart.gui.themes import render_qss
    qss = re.sub(r"/\*.*?\*/", "", render_qss(theme), flags=re.S)
    found = {}
    for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", qss):
        if "QLineEdit#controlsLineEdit:disabled" in (part.strip() for part in selectors.split(",")):
            found.update((key.strip(), value.strip()) for key, _sep, value in
                         (line.partition(":") for line in body.split(";") if ":" in line))
    assert found.get("placeholder-text-color") == found["color"], found


# ---- PA1 every "?" is visible when the window opens (Int's controls column never starves) -------------------------

@pytest.mark.parametrize("kind", KINDS)
def test_pa1_every_question_mark_is_visible_when_the_window_opens(dark, kind):
    dialog = _dialog(kind)
    try:
        dialog.show()
        _settle(dark)
        area = _first_card(dialog)
        while area is not None and not isinstance(area, QtWidgets.QScrollArea):
            area = area.parentWidget()
        assert area is not None, "the options scroll in their own pane"
        assert area.horizontalScrollBar().maximum() == 0, "no option is cut off at the splitter"
        viewport = area.viewport()
        helps = _helps(area.widget())
        assert helps
        for button in helps:
            assert _right(button, viewport) <= viewport.width(), button.toolTip()
    finally:
        dialog.close()


# ---- PA3 the Phases list is half its earlier height ------------------------------------------------------------------

def test_pa3_the_phase_list_is_half_its_earlier_height(dark):
    dialog = _dialog("phase")
    try:
        dialog.show()
        _settle(dark)
        browser = dialog.phase_source.browser
        # 034723d4 opened it at 220 px (its size hint; at most 230): half of that, never below the list's own minimum
        # (the themed list's is about 50 px; an unthemed Fusion list's is 118, which then sets the height)
        assert browser.height() <= max(110, browser.minimumSizeHint().height()) < 220
        assert browser.height() >= browser.minimumSizeHint().height()
    finally:
        dialog.close()
