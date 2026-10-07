"""Frozen native follow-ups: actual RSM/shared owners, no replacement science or page owners."""
import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from tests.core.test_rsm_viewer_model import _projection
from tests.core.test_rsm_operation_r2 import _write_member
from xdart.gui.tabs.calibration.dialog import CalibrationContext, CalibrationDialog
from tests.xdart.pages.test_rsm_page import _page, _close, _wait, _synthetic_source, _with_source


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_cyclic_planes_have_notebook_axes_values_and_uncluttered_titles(qapp):
    page = _page(qapp)
    try:
        source = _projection(shape=(3, 4, 5))
        page._show('shown.nexus', source)
        values, snapshot = page.shown
        panels = page.view.keyed_projection.panels
        assert [p.key.instance for p in panels] == ['HK', 'KL', 'LH', 'H', 'K', 'L']
        assert [p.title for p in panels[:3]] == [
            'Mean over L [2.000, 3.000]', 'Mean over H [0.000, 1.000]', 'Mean over K [1.000, 2.000]']
        assert all(p.title == '' for p in panels[3:])
        for p, horizontal, vertical in zip(panels[:3], (0, 1, 2), (1, 2, 0), strict=True):
            np.testing.assert_array_equal(p.x_axis.values, values.axes[horizontal][1])
            np.testing.assert_array_equal(p.y_axis.values, values.axes[vertical][1])
        data = values.intensity
        for p, expected in zip(snapshot.products[:3], (data.mean(2).T, data.mean(0).T, data.mean(1)), strict=True):
            np.testing.assert_array_equal(p.values, expected)
        page._map_choice.setCurrentText('Single plane')
        page._accept_map_choice()
        thin = page.shown[1]
        h, k, l = thin.state.axis_indices
        for p, expected in zip(thin.products[:3], (data[:, :, l].T, data[h, :, :].T, data[:, k, :]), strict=True):
            np.testing.assert_array_equal(p.values, expected)
    finally:
        page.close_page(); page.deleteLater(); qapp.processEvents()


def test_ub_source_populates_readonly_rows_and_preserves_typed_override(qapp, tmp_path):
    member = _write_member(tmp_path, 0)
    page = _page(qapp)
    try:
        page.set_field('project_root', str(tmp_path))
        page.set_sources((str(member.spec_path),))
        page.set_field('scan_mode', 'Select Scans'); page.set_field('scan', '1')
        assert [page._value(f'ub_row{i}') for i in (1, 2, 3)] == ['1 0 0', '0 1 0', '0 0 1']
        assert all(not page._rows[f'ub_row{i}'].editor.isEnabled() for i in (1, 2, 3))
        assert page._rows['ub_spec_path'].isHidden() and page._rows['ub_scan'].isHidden()
        from xdart.gui.tabs.rsm.page import _UB_TYPED, _UB_SOURCE, _UB_SPEC
        page.set_field('ub_from', _UB_TYPED)
        for i, text in enumerate(('2 0 0', '0 2 0', '0 0 2'), 1):
            page.set_field(f'ub_row{i}', text)
        page.set_field('ub_from', _UB_SOURCE)
        assert page._value('ub_row1') == '1 0 0'
        page.set_field('ub_from', _UB_TYPED)
        assert page._value('ub_row1') == '2 0 0'
        page.set_field('ub_from', _UB_SPEC)
        page.set_field('ub_spec_path', str(member.spec_path)); page.set_field('ub_scan', '1')
        assert not page._rows['ub_spec_path'].isHidden() and not page._rows['ub_scan'].isHidden()
        layout = page._rows['ub_row1'].parentWidget().layout()
        assert layout.indexOf(page._rows['ub_scan']) < layout.indexOf(page._rows['ub_row1'])
        assert page._value('ub_row1') == '1 0 0'
        page.set_field('ub_scan', '99')
        assert all(page._value(f'ub_row{i}') == '' for i in (1, 2, 3))
        page.set_field('ub_from', _UB_SOURCE)
        page.set_field('scan', '99'); page.set_field('scan_mode', 'All Scans')
        assert page._value('ub_row1') == '1 0 0'
    finally:
        page.close_page(); page.deleteLater(); qapp.processEvents()


def test_fresh_session_browse_remembers_in_process_without_persisting(qapp, tmp_path, monkeypatch):
    from xdart.utils.browse import browse_start_dir, remember_browse_path
    from xdart.utils.session import load_session
    monkeypatch.setenv('XDART_SESSION_FRESH', '1')
    session = tmp_path / 'session.json'
    monkeypatch.setenv('XDART_SESSION_FILE', str(session))
    last, fallback = tmp_path / 'picked', tmp_path / 'fallback'
    last.mkdir(); fallback.mkdir()
    remember_browse_path(last / 'new.nexus')
    assert browse_start_dir(fallback=str(fallback)) == str(last)
    assert load_session() == {} and not session.exists()


def test_calibration_save_uses_shared_last_folder(qapp, tmp_path, monkeypatch):
    from xdart.utils.browse import remember_browse_path
    monkeypatch.delenv('XDART_SESSION_FRESH', raising=False)
    monkeypatch.setenv('XDART_SESSION_FILE', str(tmp_path / 'session.json'))
    last = tmp_path / 'picked'; last.mkdir()
    remember_browse_path(last)
    dialog = CalibrationDialog(CalibrationContext('RSM', tmp_path))
    offered = []
    # Qt's external chooser is the boundary; the shared Browse/Save path stays real.
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName', staticmethod(
        lambda parent, title, start, filt: offered.append(start) or ('', filt)))
    try:
        assert dialog._save_path(beam=True) is None
        assert offered == [str(last / dialog._default_name(True).name)]
    finally:
        dialog.close(); dialog.deleteLater(); qapp.processEvents()


def test_saved_rsm_metadata_reuses_guarded_reader_and_int_dialog(qapp, tmp_path):
    from xrd_tools.analysis.rsm_values import prepare_rsm_tool_v2
    from xrd_tools.analysis.rsm_operation import run_rsm_operation_v2
    from xrd_tools.analysis.stitch_metadata import read_stitch_metadata
    from xrd_tools.analysis.scan_operations import AnalysisDisposition
    from xdart.gui.tabs.scattering.analysis_mount import MetadataResultDialog
    spec = _synthetic_source(tmp_path, full=True)
    page = _page(qapp)
    try:
        _with_source(page, tmp_path)
        page.set_field('grid', 'Quick 40×40×40')
        result = run_rsm_operation_v2(prepare_rsm_tool_v2(page.build_form()).request)
        tables = read_stitch_metadata(result.payload, source_root=tmp_path)
        assert len(tables) == 2
        assert all(t.table.disposition is AnalysisDisposition.COMPLETED for t in tables)
        assert all(t.selected_labels == (0, 1) for t in tables)
        assert all({'eta', 'mu', 'phi', 'del'}.issubset(c.name for c in t.table.columns) for t in tables)
        page.reopen(str(page.build_form().output_path))
        _wait(page, qapp, lambda: page.shown is not None and not page.active())
        page.browser.metadata.click()
        _wait(page, qapp, lambda: page._metadata_dialog.table.rowCount() == 2)
        assert type(page._metadata_dialog) is MetadataResultDialog
        assert page._metadata_dialog.source_selector.count() == 2
        assert page._metadata_dialog.plot_button.isEnabled()
        page._metadata_dialog.close()
        page._show_metadata()
        assert page._metadata_reading is not None and not page.preview_button.isEnabled()
        page._metadata_dialog.close()  # close during the actual worker; late completion cannot reopen it
        _wait(page, qapp, lambda: page._metadata_reading is None)
        assert page._metadata_dialog is None
        spec.write_text(spec.read_text() + '\n')
        refused = read_stitch_metadata(result.payload, source_root=tmp_path)
        assert all(t.table.disposition is AnalysisDisposition.REFUSED for t in refused)
        assert all('Source changed' in t.table.code for t in refused)
    finally:
        _close(page, qapp)
