"""Focused phase-source bindings found while implementing the shared owner."""
import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from tests.xdart.scattering.test_phase_reference_overlays import _cif


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_native_mixed_fit_keeps_cif_texture_and_frees_only_lattice_peaks(qapp, tmp_path, monkeypatch):
    from xdart.gui.analysis.phase_fit_dialog import PhaseFitDialog
    monkeypatch.setenv('XDART_PHASE_LIBRARY', str(tmp_path / 'phases'))
    dialog = PhaseFitDialog()
    try:
        dialog.phase_source.add_cif_paths((_cif(tmp_path),))
        dialog.phase_source.add_lattice('Position-only', a=4.1, space_group=221)
        dialog._show_pattern(np.linspace(1., 4., 31), np.ones(31), 'q')
        request = dialog.build_fit_request()
        assert request is not None
        analyzer = request[1]
        assert analyzer.plan.config.fit_kw['texture'] == {'CsCl': 'none', 'Position-only': 'free'}
        assert tuple(phase.source_kind for phase in analyzer.phases) == ('cif', 'lattice')
        assert not dialog.texture_combo.isEnabled()
        assert 'lattice' in dialog.texture_combo.toolTip().lower()
        dialog.texture_combo.setCurrentIndex(1)
        request = dialog.build_fit_request()
        assert request[1].plan.config.fit_kw['texture'] == {'CsCl': 'none', 'Position-only': 'free'}
        dialog.phase_source.set_selection((dialog.phase_source.selection[0],))
        assert dialog.texture_combo.isEnabled()
        assert dialog.build_fit_request()[1].plan.config.fit_kw['texture'] == 'march_dollase'
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_phase_visibility_keeps_the_selected_library_id_until_remove(qapp, tmp_path, monkeypatch):
    from xdart.gui.analysis.phase_source import PhaseSourceWidget
    monkeypatch.setenv('XDART_PHASE_LIBRARY', str(tmp_path / 'phases'))
    source = PhaseSourceWidget()
    try:
        entry = source.add_lattice('positions', a=4.1, space_group=221)
        source.browser.scans.item(0).setCheckState(QtCore.Qt.CheckState.Unchecked)
        assert source.selection == ((entry.id, False),)
        source.browser.scans.item(0).setCheckState(QtCore.Qt.CheckState.Checked)
        assert source.selection == ((entry.id, True),)
        source.browser.scans.setCurrentRow(0)
        source.remove_btn.click()
        assert source.selection == ()
        assert source.library.resolve(entry.id).entry == entry
    finally:
        source.close()
        source.deleteLater()
        qapp.processEvents()
