"""NEXT-128-HELP-FOLLOWUPS (GUI row F1): Return in a Peak Fitter field commits it and presses no button. The maintainer
picked "Return and focus fixes (Recommended)". Frozen before the product change (DESIGN-HELP-FOLLOWUPS-128.md): F1 is red
on 72050876 (Return opened Help, Fable's B6 on HELP-GUIDES).

NEXT-128-WINDOW-POLISH: F2 (the window "?" never takes the first focus) is retired with that "?" (TQ, "Remove it
(Recommended)"; test_help_guides_gui pins its absence), and F1 turns Full off before typing a range, since Full is on by
default and locks the range while it is (PF5).
"""
import pytest
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def isolated_library(tmp_path, monkeypatch):
    monkeypatch.setenv("XDART_PHASE_LIBRARY", str(tmp_path / "phase-library"))


def _help_shown():
    from xdart.gui.widgets.context_help import HelpWindow
    return [window for window in QtWidgets.QApplication.topLevelWidgets()
            if isinstance(window, HelpWindow) and window.isVisible()]


def _tool_window(kind, source):
    from xdart.gui.analysis.peak_fit_dialog import PeakFitDialog
    from xdart.gui.analysis.phase_fit_dialog import PhaseFitDialog
    return (PeakFitDialog if kind == "peak" else PhaseFitDialog)(vnext_submit=lambda *_args: None, source=source)


def _first_option(dialog):
    """The options column's first option: in the window's focus chain, its first control that takes keyboard focus,
    shown and enabled, other than a "?" (a help action)."""
    from xdart.gui.widgets.context_help import _help_action
    column, widget = dialog.split.controls.widget(), dialog
    while (widget := widget.nextInFocusChain()) is not dialog:
        if (column.isAncestorOf(widget) and widget.focusPolicy() & QtCore.Qt.FocusPolicy.TabFocus
                and widget.isVisible() and widget.isEnabled() and not _help_action(widget)):
            return widget
    return None


@pytest.mark.parametrize("source", (True, False), ids=("int", "plain"))
@pytest.mark.parametrize("kind", ("peak", "phase"))
def test_the_window_opens_with_its_first_option_focused(qapp, kind, source):
    """NEXT-129-SMALL-FIXES item 5 (pre-existing; HELP-FOLLOWUPS residual): the window opened with keyboard focus on no
    option (the options column itself, since the shared split; a section's "?" before it). Its first option takes the
    first focus: the Source card's in Int's window, else the first section's. Return and Escape keep their one owner,
    RetainedDialogKeyPolicy, whose rows are F1 below and test_fit_dialog_esc."""
    dialog = _tool_window(kind, source)
    try:
        dialog.show()
        dialog.activateWindow()
        qapp.processEvents()
        focus, first = QtWidgets.QApplication.focusWidget(), _first_option(dialog)
        assert first is not None and focus is first, (focus, first)
        assert (dialog.source_card if source else dialog.split._card_widgets[0]).isAncestorOf(focus)
        if kind == "peak" and not source:
            assert focus is dialog.model_combo
    finally:
        dialog.close()


def test_f1_return_in_a_peak_fitter_field_commits_it_and_presses_no_button(qapp):
    """Phase Analysis's W11, for the Peak Fitter: Return reached the window, which pressed its default button, the
    first section's "?", so Help opened on every committed value."""
    from xdart.gui.analysis.peak_fit_dialog import PeakFitDialog
    dialog = PeakFitDialog(vnext_submit=lambda *_args: None)
    try:
        dialog.show()
        dialog.activateWindow()
        qapp.processEvents()
        assert not _help_shown()
        dialog.full_btn.setChecked(False)  # NEXT-128-WINDOW-POLISH PF5: Full is on by default and locks the range
        line = dialog.range_lo
        QtTest.QTest.mouseClick(line, QtCore.Qt.MouseButton.LeftButton)
        qapp.processEvents()
        line.selectAll()
        QtTest.QTest.keyClicks(line, "0.7")
        QtTest.QTest.keyClick(line, QtCore.Qt.Key.Key_Return)
        qapp.processEvents()
        assert dialog._fit_lo == pytest.approx(0.7), "Return commits the value"
        assert not _help_shown() and dialog.isVisible(), "and presses no button"
        # A focused button still answers its own Return.
        button = next(button for button in dialog.findChildren(QtWidgets.QPushButton) if button.text() == "?")
        button.setFocus()  # a section's "?", whose Return opens Help
        QtTest.QTest.keyClick(button, QtCore.Qt.Key.Key_Return)
        qapp.processEvents()
        assert _help_shown(), "a focused ? opens Help on Return"
    finally:
        dialog.close()
