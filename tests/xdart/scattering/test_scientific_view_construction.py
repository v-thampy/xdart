"""ScientificView construction: its image canvases are shown only through their panes."""
import pytest

pytest.importorskip("pyqtgraph")

from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.tabs.scattering.scientific_view import ScientificView


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_a_scientific_view_never_shows_its_image_canvases_as_windows(qapp):
    """Live 2026-10-08: the shared image widget showed itself in its constructor, before its pane parented it, so
    every image pane opened as its own window for an instant: three as the Int page builds, and more for each tool
    or help window that builds a view. On macOS such a window takes and returns the activation, the trigger of the
    Int 2D endless loop. A canvas is shown only through its pane; that it shows with its pane is unchanged."""
    shown = []

    class WindowsShown(QtCore.QObject):
        def eventFilter(self, watched, event):
            if (event.type() == QtCore.QEvent.Type.Show and isinstance(watched, QtWidgets.QWidget)
                    and watched.isWindow()):
                shown.append(type(watched).__name__)
            return False

    spy = WindowsShown()
    qapp.installEventFilter(spy)
    try:
        view = ScientificView()
    finally:
        qapp.removeEventFilter(spy)
    try:
        panes = (view.raw, view.cake, view.waterfall)
        assert all(pane.canvas.isVisibleTo(pane) for pane in panes)
        assert shown == []
    finally:
        view.deleteLater()
