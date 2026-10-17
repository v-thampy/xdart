"""Shared calibration close prompt, real Qt events and pair writer/reader."""
import time
from pathlib import Path
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets
from PySide6.QtTest import QTest
from tests.xdart.scattering.test_calibration_file_link import _geometry, qapp
from xdart.gui.tabs.calibration.dialog import CalibrationDialog, CalibrationContext


def _close_with(dialog, qapp, button_text):
    errors = []
    timer = QtCore.QTimer()
    def choose():
        box = QtWidgets.QApplication.activeModalWidget()
        if isinstance(box, QtWidgets.QMessageBox):
            candidates = [b for b in box.buttons() if button_text in b.text()]
            if not candidates:
                errors.append(box.text()); box.reject()
            else:
                QTest.mouseClick(candidates[0], QtCore.Qt.MouseButton.LeftButton)
    timer.timeout.connect(choose); timer.start(10)
    try:
        dialog.close()
    finally:
        timer.stop()
    assert not errors


def test_changed_loaded_setup_close_can_cancel_or_discard_without_rewriting(qapp, tmp_path):
    path = tmp_path / 'initial.json'
    _geometry(path, 90, 1.0147)
    before = path.read_bytes(), path.with_suffix('.json.provenance.json').read_bytes()
    dialog = CalibrationDialog(CalibrationContext('RSM', tmp_path))
    try:
        dialog.load_calibration(path); dialog.show(); qapp.processEvents()
        baseline = dialog.geometry_editor.value()
        phi = dialog.geometry_editor.axis_box('phi')
        phi.setCurrentIndex(phi.findData(-1))
        _close_with(dialog, qapp, 'Cancel')
        assert dialog.isVisible() and dialog.geometry_editor.value() != baseline
        _close_with(dialog, qapp, 'Discard')
        assert not dialog.isVisible() and dialog.geometry_editor.value() == baseline
        assert before == (path.read_bytes(), path.with_suffix('.json.provenance.json').read_bytes())
    finally:
        dialog.shutdown(); dialog.deleteLater(); qapp.processEvents()


def test_save_on_close_writes_and_adopts_new_valid_pair(qapp, tmp_path, monkeypatch):
    from xrd_tools.integrate.stitch_calibration import save_calibration_pair, paired_xu_calibration, pair_validation_angles
    from xrd_tools.analysis.rsm_geometry_asset import capture_rsm_calibration_geometry
    initial = tmp_path / 'initial.json'
    _geometry(initial, 90, 1.0147)
    dialog = CalibrationDialog(CalibrationContext('RSM', tmp_path))
    new = tmp_path / 'edited.json'
    adopted = []
    try:
        dialog.load_calibration(initial)
        # Produce an authentic rough/direct-beam pair, with its truthful provenance.
        fit = dialog._rough_fit()
        saved = save_calibration_pair(fit, tmp_path / 'rough.json',
                                      validation_angles=pair_validation_angles(fit), overwrite=False)
        assert saved.xu_path is not None, saved.xu_refusal
        dialog.load_calibration(saved.pyfai_path)
        before = [Path(p).read_bytes() for p in (saved.pyfai_path, saved.xu_path)]
        dialog.calibrationSaved.connect(lambda path, backend: adopted.append(path))
        dialog.show(); qapp.processEvents()
        phi = dialog.geometry_editor.axis_box('phi'); phi.setCurrentIndex(phi.findData(-1))
        monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName', staticmethod(
            lambda *args: (str(new), 'JSON (*.json)')))
        _close_with(dialog, qapp, 'Save new')
        deadline = time.monotonic() + 15
        while dialog.isVisible() and time.monotonic() < deadline:
            qapp.processEvents(); time.sleep(.01)
        assert not dialog.isVisible() and adopted == [str(new)]
        assert Path(paired_xu_calibration(new)) == new.with_name(new.stem + '.xu.json')
        receipt = capture_rsm_calibration_geometry(new.name, project_root=tmp_path)
        assert receipt is not None
        assert [Path(p).read_bytes() for p in (saved.pyfai_path, saved.xu_path)] == before
    finally:
        dialog.shutdown(); dialog.deleteLater(); qapp.processEvents()
