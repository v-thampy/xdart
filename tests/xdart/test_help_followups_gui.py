"""NEXT-128-HELP-FOLLOWUPS (GUI rows F1, F2): Return in a Peak Fitter field commits it and presses no button, and the
window "?" never takes the first keyboard focus. The maintainer picked "Return and focus fixes (Recommended)".
Frozen before the product change (DESIGN-HELP-FOLLOWUPS-128.md): F1 is red on 72050876 (Return opened Help, Fable's B6
on HELP-GUIDES); F2 pins 72050876's creation order.
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
        assert button is not dialog.guide_button  # a section's "?", whose Return opens Help
        button.setFocus()
        QtTest.QTest.keyClick(button, QtCore.Qt.Key.Key_Return)
        qapp.processEvents()
        assert _help_shown(), "a focused ? opens Help on Return"
    finally:
        dialog.close()


@pytest.mark.parametrize("tool", ("peak", "phase"))
def test_f2_the_window_question_mark_never_takes_the_first_focus(qapp, tool):
    """Fable's B2 on HELP-GUIDES: the guide "?" is made last, so the window's first control keeps the initial focus and
    a stray Space on opening never launches the browser."""
    from xdart.gui.analysis.peak_fit_dialog import PeakFitDialog
    from xdart.gui.analysis.phase_fit_dialog import PhaseFitDialog
    dialog = (PeakFitDialog if tool == "peak" else PhaseFitDialog)(vnext_submit=lambda *_args: None)
    try:
        dialog.show()
        dialog.activateWindow()
        qapp.processEvents()
        assert dialog.focusWidget() is not dialog.guide_button
    finally:
        dialog.close()
