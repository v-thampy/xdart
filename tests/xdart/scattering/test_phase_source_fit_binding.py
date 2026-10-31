"""Focused phase-source bindings found while implementing the shared owner."""
import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from tests.xdart.scattering.test_phase_reference_overlays import _cif


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_native_mixed_fit_keeps_cif_texture_and_frees_only_lattice_peaks(qapp, tmp_path, monkeypatch):
    """PHASE-ANALYSIS-128 successor: the one displayed fit path. Texture applies to CIF phases (the hold is lifted);
    a lattice phase always fits free heights, whatever the texture."""
    from xdart.gui.analysis.phase_fit_dialog import PhaseFitDialog
    from xrd_tools.analysis.display_fit_operations import (DisplayedPhaseFitPlan, DisplayedTraceInput,
                                                           DisplayedTraceReceipt, run_displayed_phase_fit)
    monkeypatch.setenv('XDART_PHASE_LIBRARY', str(tmp_path / 'phases'))
    dialog = PhaseFitDialog(vnext_submit=lambda *_args: None)
    try:
        dialog.phase_source.add_cif_paths((_cif(tmp_path),))
        dialog.phase_source.add_lattice('Position-only', a=4.1, space_group=221)
        q = np.linspace(1., 4., 31)
        trace = DisplayedTraceInput(q, np.ones(31), 'q', '1/angstrom', 'q', 1, DisplayedTraceReceipt(
            'displayed-trace-v1', 1, 'run', '1', tmp_path / 'output.nxs', 'q', 0))

        def names(**options):
            values = dict(dialog.vnext_fit_values(), max_nfev=1, **options)
            result = run_displayed_phase_fit(DisplayedPhaseFitPlan(trace, wavelength_angstrom=1.5406, **values))
            assert result.code == 'OK', (result.code, result.diagnostics)
            return set(result.parameter_names)

        assert tuple(entry.source_kind for entry in dialog.vnext_fit_values()['phase_entries']) == ('cif', 'lattice')
        plain = names()
        assert any(name.startswith('p1_pk') for name in plain) and not any(name.startswith('p0_pk') for name in plain)
        assert dialog.texture_combo.isEnabled() and 'lattice' in dialog.texture_combo.toolTip().lower()
        dialog.texture_combo.setCurrentIndex(1)
        assert dialog.vnext_fit_values()['texture'] == 'march_dollase'
        textured = names()
        assert 'p0_march_r' in textured and any(name.startswith('p1_pk') for name in textured)
        assert 'p1_march_r' not in textured
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
