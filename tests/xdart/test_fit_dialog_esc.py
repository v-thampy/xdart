"""Esc must NOT close the Peak / Phase fit popups (match the metadata plotter) --
the default QDialog Esc->reject would discard the whole fit setup.  Dialog imports
are deferred so module collection remains headless-safe, per test_gui_logging."""
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
        dlg.deleteLater()


def test_peak_fit_dialog_swallows_escape(qapp):
    from xdart.gui.analysis.peak_fit_dialog import PeakFitDialog
    assert _survives_escape(PeakFitDialog(analysis_context=None), qapp)


def test_phase_fit_dialog_swallows_escape(qapp):
    from xdart.gui.analysis.phase_fit_dialog import PhaseFitDialog
    assert _survives_escape(PhaseFitDialog(vnext_submit=lambda *_args: None), qapp)  # PHASE-ANALYSIS-128


def test_tool_windows_inherit_one_key_policy():
    """The retained windows mount the shared policy rather than copying its handler."""
    from xdart.gui.analysis.peak_fit_dialog import PeakFitDialog
    from xdart.gui.analysis.phase_fit_dialog import PhaseFitDialog
    from xdart.gui.tabs.calibration.dialog import CalibrationDialog
    from xdart.gui.widgets.retained_dialog import RetainedDialogKeyPolicy

    for dialog_type in (PeakFitDialog, PhaseFitDialog, CalibrationDialog):
        assert issubclass(dialog_type, RetainedDialogKeyPolicy)
        assert 'keyPressEvent' not in dialog_type.__dict__
        assert dialog_type.keyPressEvent is RetainedDialogKeyPolicy.keyPressEvent
