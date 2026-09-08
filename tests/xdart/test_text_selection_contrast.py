"""Real Qt coverage for shared text-selection styling, including Poni."""

from __future__ import annotations

import pytest
from pyqtgraph.Qt import QtGui, QtWidgets

from xdart.gui.themes import apply_theme, typography


@pytest.fixture
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    typography.capture_application_baseline(app)
    original_qss = app.styleSheet()
    original_font = QtGui.QFont(app.font())
    try:
        yield app
    finally:
        app.setStyleSheet(original_qss)
        app.setFont(original_font)
        typography.restore_platform_class_fonts(app, typography.DEFAULT_FONT_SCALE)
        app.processEvents()


def _luminance(color):
    channels = [color.redF(), color.greenF(), color.blueF()]
    linear = [
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in channels
    ]
    return sum(weight * value for weight, value in zip((0.2126, 0.7152, 0.0722), linear))


def _contrast(first, second):
    light, dark = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


def _assert_selection_palette(editor):
    palette = editor.palette()
    for group in (QtGui.QPalette.Active, QtGui.QPalette.Inactive):
        background = palette.color(group, QtGui.QPalette.Highlight)
        foreground = palette.color(group, QtGui.QPalette.HighlightedText)
        assert background.name() == "#8f98b8"
        assert foreground.name() == "#1a1a1a"
        assert _contrast(foreground, background) >= 4.5
        assert _contrast(background, palette.color(group, QtGui.QPalette.Base)) >= 2.0


def _assert_selection_painted(editor):
    # Verify pixels, not just a palette role the widget never consumes.
    image = editor.grab().toImage()
    colors = {
        image.pixelColor(x, y).name()
        for y in range(image.height()) for x in range(image.width())
    }
    assert "#8f98b8" in colors
    assert "#1a1a1a" in colors


@pytest.mark.parametrize("theme_name", ["dark", "light"])
def test_periwinkle_poni_selection_is_visible_in_real_controls(qapp, theme_name):
    from xdart.gui.tabs.scattering.controls_projection import project_controls
    from xdart.gui.tabs.scattering.state_machine import RunPhase
    from xdart.gui.widgets.controls_panel import ControlsPanel, FormRow
    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.session.run_configuration import RunIntent

    apply_theme(qapp, theme_name, accent_color="periwinkle_muted")
    panel = ControlsPanel()
    try:
        intent = RunIntent(poni_file="/project/detxn26_detyn6p5_eta4p5.poni")
        panel.reconcile(project_controls(RunIntentStore(intent).snapshot(), None, RunPhase.IDLE))
        panel.resize(520, 1600)
        panel.show()
        qapp.processEvents()
        panel.activateWindow()
        qapp.processEvents()
        editor = next(
            row.editor for row in panel.findChildren(FormRow)
            if row.path == ("Signal", "poni_file")
        )
        assert isinstance(editor, QtWidgets.QLineEdit)
        editor.setFocus()
        editor.selectAll()
        qapp.processEvents()
        assert editor.isActiveWindow()
        assert editor.hasFocus()
        assert editor.selectedText() == editor.text()
        assert editor.selectedText()
        _assert_selection_palette(editor)
        _assert_selection_painted(editor)
    finally:
        panel.close()


@pytest.mark.parametrize("theme_name", ["dark", "light"])
def test_shared_line_edit_keeps_selection_readable_in_an_inactive_window(qapp, theme_name):
    apply_theme(qapp, theme_name, accent_color="periwinkle_muted")
    editor = QtWidgets.QLineEdit("Selected input text")
    other_window = QtWidgets.QLineEdit("other window")
    try:
        editor.show()
        other_window.show()
        qapp.processEvents()
        editor.activateWindow()
        qapp.processEvents()
        editor.setFocus()
        editor.selectAll()
        other_window.activateWindow()
        qapp.processEvents()
        assert not editor.isActiveWindow()
        assert not editor.hasFocus()
        assert editor.selectedText() == editor.text()
        _assert_selection_palette(editor)
        _assert_selection_painted(editor)
    finally:
        other_window.close()
        editor.close()


@pytest.mark.parametrize("theme_name", ["dark", "light"])
@pytest.mark.parametrize("widget_type", [
    QtWidgets.QLineEdit, QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox,
    QtWidgets.QComboBox, QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit,
])
def test_shared_editors_receive_readable_selection(qapp, theme_name, widget_type):
    apply_theme(qapp, theme_name, accent_color="periwinkle_muted")
    widget = widget_type()
    try:
        if isinstance(widget, QtWidgets.QComboBox):
            widget.setEditable(True)
        widget.show()
        qapp.processEvents()
        _assert_selection_palette(widget)
        if isinstance(widget, (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox)):
            _assert_selection_palette(widget.findChild(QtWidgets.QLineEdit))
    finally:
        widget.close()
