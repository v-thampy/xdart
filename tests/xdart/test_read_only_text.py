"""The shared read-only text owner (``widgets/read_only_text.py``; Codex's combined stage-2b review F2, hard rule
14): a pane and a window of text the user reads but does not edit, and every page, tool and pop-up that shows such
text mounting it (RSM's and Stitch's Preview, Stitch's saved provenance, the calibration window's two summaries).
The reuse fence (tests/xdart/pages/test_gui_reuse_fence.py) refuses a page's own text editor."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("pyqtgraph")

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from xdart.gui.pages.services import empty_host_services
from xdart.gui.widgets.read_only_text import ReadOnlyTextPane, ReadOnlyTextWindow


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _Status:
    def show(self, text, timeout_ms=0):
        pass


def _typed(pane, text):
    """Keys typed into the pane as a user types them."""
    for character in text:
        for kind in (QtCore.QEvent.Type.KeyPress, QtCore.QEvent.Type.KeyRelease):
            QtWidgets.QApplication.sendEvent(pane, QtGui.QKeyEvent(kind, QtCore.Qt.Key.Key_A,
                                                                   QtCore.Qt.KeyboardModifier.NoModifier, character))


def test_the_pane_shows_text_that_cannot_be_typed_over_and_is_the_themed_editor(qapp):
    """Read-only, its text set whole (and the plain-text calls its consumers make keep working); typing changes
    nothing. It is a QPlainTextEdit, so the themes' input rules (``themes/dark.py``) style it."""
    pane = ReadOnlyTextPane()
    try:
        assert isinstance(pane, QtWidgets.QPlainTextEdit) and pane.isReadOnly()
        pane.set_text("2 RSMs\n\nRSM 1")
        assert pane.toPlainText() == "2 RSMs\n\nRSM 1"
        _typed(pane, "xy")
        assert pane.toPlainText() == "2 RSMs\n\nRSM 1"
        pane.appendPlainText("saved")
        assert pane.toPlainText() == "2 RSMs\n\nRSM 1\nsaved"
        pane.set_text("")
        assert pane.toPlainText() == ""
    finally:
        pane.deleteLater()


def test_the_window_is_one_pane_with_its_title_and_size(qapp):
    parent = QtWidgets.QWidget()
    try:
        window = ReadOnlyTextWindow("RSM preview", (760, 560), parent)
        assert window.parent() is parent and not window.isModal()
        assert window.windowTitle() == "RSM preview" and (window.width(), window.height()) == (760, 560)
        assert type(window.pane) is ReadOnlyTextPane and window.findChildren(QtWidgets.QPlainTextEdit) == [
            window.pane]
        window.set_text("1 RSMs")
        assert window.pane.toPlainText() == "1 RSMs"
    finally:
        parent.deleteLater()


@pytest.mark.parametrize(("page_kind", "title", "size"), (("rsm", "RSM preview", (760, 560)),
                                                          ("stitch", "Stitch preview", (680, 480))))
def test_each_pages_preview_is_the_shared_window(qapp, page_kind, title, size):
    """RSM's and Stitch's Preview: the shared window, its title and size as before, its pane the page's
    ``preview_text`` (Preview sets its text, then shows the window)."""
    from xdart.gui.pages.values import PageCleanup
    if page_kind == "rsm":
        from xdart.gui.tabs.rsm.page import RSMPage as Page
    else:
        from xdart.gui.tabs.stitch.page import StitchPage as Page
    page = Page(empty_host_services(_Status()))
    try:
        dialog = page.preview_dialog
        assert type(dialog) is ReadOnlyTextWindow and dialog.parent() is page
        assert page.preview_text is dialog.pane and page.preview_text.isReadOnly()
        assert dialog.windowTitle() == title and (dialog.width(), dialog.height()) == size
        assert not dialog.isVisible()
    finally:
        close = page.close_page if page_kind == "rsm" else page.shutdown
        assert close().status is PageCleanup.CLEAN
        page.deleteLater()
        qapp.processEvents()


def test_stitchs_saved_provenance_is_the_shared_window(qapp):
    """The metadata window's Provenance… opens the shared window ("Saved Stitch provenance", 720 × 520) on the
    saved provenance, indented; closing the metadata window closes it."""
    from xdart.gui.tabs.stitch.metadata import StitchMetadataDialog
    provenance = json.dumps({"program": "xdart", "frames": [1, 2]})
    dialog = StitchMetadataDialog(provenance)
    try:
        window = dialog.provenance_dialog
        assert type(window) is ReadOnlyTextWindow and window.parent() is dialog
        assert dialog.provenance_text is window.pane and dialog.provenance_text.isReadOnly()
        assert window.windowTitle() == "Saved Stitch provenance" and (window.width(), window.height()) == (720, 520)
        assert dialog.provenance_text.toPlainText() == json.dumps(json.loads(provenance), indent=2)
        dialog.show()
        dialog.provenance_button.click()
        assert window.isVisible()
        dialog.close()
        assert not window.isVisible()
    finally:
        dialog.deleteLater()


def test_the_calibration_windows_summaries_are_shared_panes(qapp, tmp_path):
    """The shared calibration window's fit summary (4 · Calibrate and fit) and direct-beam summary (2 · Rough
    geometry) are the shared pane, sized as before: at least eight lines, at most nine."""
    from xdart.gui.tabs.calibration.dialog import CalibrationContext, CalibrationDialog
    from xdart.gui.pages.values import PageCleanup
    dialog = CalibrationDialog(CalibrationContext("RSM", tmp_path))
    try:
        for pane in (dialog.fit_summary, dialog.beam_summary):
            assert type(pane) is ReadOnlyTextPane and pane.isReadOnly()
        assert dialog.fit_summary.minimumHeight() == dialog.fit_summary.fontMetrics().lineSpacing() * 8
        assert dialog.beam_summary.maximumHeight() == dialog.beam_summary.fontMetrics().lineSpacing() * 9
    finally:
        assert dialog.shutdown().status is PageCleanup.CLEAN
        dialog.deleteLater()
