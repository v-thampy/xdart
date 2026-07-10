"""Esc must NOT close the Peak / Phase fit popups (match the metadata plotter) --
the default QDialog Esc->reject would discard the whole fit setup.  Dialog imports
are deferred into the tests (importing them pulls the static_scan GUI stack; keep
module collection headless-safe, per test_gui_logging)."""
import pytest
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


@pytest.fixture
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _survives_escape(dlg, qapp):
    dlg.show()
    try:
        ev = QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress,
                             QtCore.Qt.Key.Key_Escape,
                             QtCore.Qt.KeyboardModifier.NoModifier)
        dlg.keyPressEvent(ev)
        qapp.processEvents()
        return dlg.isVisible()           # still open after Esc
    finally:
        dlg.close()


def test_peak_fit_dialog_swallows_escape(qapp):
    from xdart.gui.tabs.static_scan.peak_fit_dialog import PeakFitDialog
    assert _survives_escape(PeakFitDialog(analysis_context=None), qapp)


def test_phase_fit_dialog_swallows_escape(qapp):
    from xdart.gui.tabs.static_scan.phase_fit_dialog import PhaseFitDialog
    assert _survives_escape(PhaseFitDialog(analysis_context=None), qapp)
