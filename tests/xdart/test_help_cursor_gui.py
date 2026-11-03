"""NEXT-128-HELP-CURSOR (row C1): clicking a figure topic in Scientific help, with nothing cached, draws its figures
without an override cursor.

The maintainer's new Mac (Mac17,8, macOS 27.2) crashed on every such click (four crash reports, 2026-11-02): Help set
Qt's WaitCursor while it drew, Qt 6.11.1 builds that cursor on macOS from an ICC-tagged PNG, and its
qt_mac_cgImageFormatForImage releases the colour space it created before CGImageCreate reads it. Frozen before the
product change: red on 8b8ac227, which set the WaitCursor during the cold draw.
"""
import pytest
from pyqtgraph.Qt import QtCore, QtGui, QtTest, QtWidgets


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_c1_a_figure_topic_clicked_cold_draws_without_an_override_cursor(qapp, monkeypatch):
    from xrd_tools import geometry_figures as gf
    from xdart.gui.widgets.context_help import owner_for
    overrides = []
    original = QtWidgets.QApplication.setOverrideCursor

    def track(cursor):
        overrides.append(QtGui.QCursor(cursor).shape())
        return original(cursor)

    monkeypatch.setattr(QtWidgets.QApplication, "setOverrideCursor", track)
    root = QtWidgets.QWidget(); owner = owner_for(root)
    try:
        cache = gf.teaching_cache(); cache.clear()
        root.show(); owner.show_index(); window = owner.window
        QtTest.QTest.keyClicks(window.search_row.editor, "Multi-step detector calibration")
        qapp.processEvents()
        scans = window.browser.scans
        item = next(scans.item(row) for row in range(scans.count())
                    if scans.item(row).data(int(QtCore.Qt.ItemDataRole.UserRole)) == "calibration.multi-step")
        # The maintainer's path: a mouse click on the result, through BrowserView's selection, into show_topic.
        QtTest.QTest.mouseClick(scans.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=scans.visualItemRect(item).center())
        qapp.processEvents()
        assert window.topic_id == "calibration.multi-step"
        assert cache.stats()["figure_misses"] >= 1, "the figures were drawn cold"
        assert window.figure_view is not None and window.figure_view.isVisible()
        assert overrides == [], "no override cursor while Help draws"
        assert QtWidgets.QApplication.overrideCursor() is None
    finally:
        root.close(); root.deleteLater(); qapp.processEvents()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
