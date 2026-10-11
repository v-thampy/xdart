"""Real shared-source fitting retains plain identities and existing CIF recipes."""
import time

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from tests.xdart.scattering.test_phase_reference_overlays import _cif
from xrd_tools.analysis.display_fit_operations import DisplayedTraceInput, DisplayedTraceReceipt
from xrd_tools.analysis.fit_snapshot import FitAxis, FitInputKey, FitInputSnapshot


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _trace(root):
    return DisplayedTraceInput(np.linspace(1., 4., 101), np.ones(101)*10,
        'q', 'A^-1', 'selected', 1,
        DisplayedTraceReceipt('displayed-trace-v1', 1, 'run', '1', root/'scan.nxs', '1'))


def _close(qapp, dialog):
    dialog.close()
    dialog.deleteLater()
    qapp.processEvents()


def test_library_cif_input_key_and_notebook_keep_the_existing_path_recipe(qapp, tmp_path, monkeypatch):
    from xdart.gui.analysis.phase_fit_dialog import PhaseFitDialog
    from xdart.gui.analysis.fit_export import write_fit_notebook, plain_settings
    from xrd_tools.notebooks import FitNotebookRecipe, capture_snapshot
    from xrd_tools.notebooks.recipe import recipe_to_json, recipe_from_json, recipe_analyzer
    monkeypatch.setenv('XDART_PHASE_LIBRARY', str(tmp_path/'library'))
    dialog = PhaseFitDialog(vnext_submit=lambda *_: None)
    try:
        dialog.phase_source.add_cif_paths((_cif(tmp_path),))
        dialog.set_vnext_trace(_trace(tmp_path))
        dialog._export_wavelength = 1.5406
        dialog.max_nfev.setValue(1)
        snapshot = dialog.export_snapshot()  # Parent fails through the real key boundary.
        assert isinstance(snapshot.key, FitInputKey)
        values = dialog.vnext_fit_values()
        assert FitInputKey('same', '', '', 1, settings=plain_settings(values)).settings
        settings = dialog.export_actions.settings()
        entry = dialog.phase_source.selected_entries()[0]
        assert settings['cif_paths'] == (str(entry.cif_path),)
        assert settings['phase_names'] == (entry.name,)
        assert settings['expected_hashes'] == (entry.id,)
        assert 'phase_entries' not in settings
        for suffix in ('.ipynb', '.py'):
            path = tmp_path/('phase'+suffix)
            write_fit_notebook(path, snapshot, 'phase', settings)
            text = path.read_text()
            assert 'fit-notebook-recipe-v1' in text
            assert 'phase_entries' not in text
        captured = capture_snapshot(snapshot, 'selected.npz')
        recipe = FitNotebookRecipe('phase', captured.reference, settings=settings)
        restored = recipe_from_json(recipe_to_json(recipe))
        result = recipe_analyzer(restored, data_root=tmp_path).analyze(snapshot)
        assert result.result.code == 'OK', result.message
    finally:
        _close(qapp, dialog)


def test_lattice_notebook_exports_are_explicitly_unavailable_without_a_new_recipe(qapp, tmp_path, monkeypatch):
    from xdart.gui.analysis.phase_fit_dialog import PhaseFitDialog
    monkeypatch.setenv('XDART_PHASE_LIBRARY', str(tmp_path/'library'))
    dialog = PhaseFitDialog(vnext_submit=lambda *_: None)
    try:
        dialog.phase_source.add_lattice('positions', a=4.1, space_group=221)
        dialog.set_vnext_trace(_trace(tmp_path))
        assert all(not button.isEnabled() for button in dialog.export_actions.buttons.values())
        assert all('lattice' in button.toolTip().lower() for button in dialog.export_actions.buttons.values())
        dialog.export_actions.choose('jupyter')
        assert 'PHASE_LATTICE_NOTEBOOK_UNAVAILABLE' in dialog.status.text()
        with pytest.raises(ValueError, match='PHASE_LATTICE_NOTEBOOK_UNAVAILABLE'):
            dialog.export_fit_values()
        assert not tuple(tmp_path.glob('*.ipynb'))
        dialog.phase_source.set_selection(())
        assert all(button.isEnabled() for button in dialog.export_actions.buttons.values())
    finally:
        _close(qapp, dialog)


def test_shared_host_fits_actual_lattice_entries_with_plain_input_keys(qapp, tmp_path, monkeypatch):
    from xdart.gui.analysis.snapshot_fit_host import SnapshotFitHost
    from xrd_tools.analysis.phase_references import phase_model
    monkeypatch.setenv('XDART_PHASE_LIBRARY', str(tmp_path/'library'))
    x = np.linspace(1., 4., 101)
    snapshot = FitInputSnapshot(FitInputKey('selected', '', 'admitted', 1),
        (FitAxis('q', 'q_A^-1', x),), np.ones(101)*10, wavelength_angstrom=1.5406)
    host = SnapshotFitHost('phase', current=lambda: snapshot, scope=lambda: snapshot.key)
    try:
        entry = host.dialog.phase_source.add_lattice('positions', a=4.1, space_group=221)
        model = phase_model(entry, wavelength_angstrom=1.5406, q_range=(1., 4.))
        y = np.ones(101)*10
        for peak in model.peaks:
            y += 100*np.exp(-.5*((x-peak.q)/.025)**2)
        snapshot = FitInputSnapshot(snapshot.key, snapshot.axes, y, wavelength_angstrom=1.5406)
        host.dialog.max_nfev.setValue(1)
        host.dialog.set_vnext_trace(snapshot.trace_input())
        host.dialog.fit_btn.click()
        deadline = time.monotonic()+30
        while host.computing and time.monotonic()<deadline:
            qapp.processEvents()
            time.sleep(.005)
        assert host.outcomes, host.dialog.status.text()
        result = host.outcomes[0].result
        assert result is not None, host.outcomes[0].message
        assert result.code == 'OK', result.diagnostics
        assert result.phase_fractions_reason
        assert not any(name.startswith('frac_') for name in host.outcomes[0].params)
        assert isinstance(host.snapshot.key, FitInputKey)
    finally:
        host.dialog.close()
        deadline = time.monotonic()+30
        while host.shutdown().status.value == 'pending' and time.monotonic()<deadline:
            qapp.processEvents()
            time.sleep(.005)
        host.dialog.deleteLater()
        qapp.processEvents()
