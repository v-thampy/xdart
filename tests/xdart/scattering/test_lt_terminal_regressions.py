"""Terminal discriminators for existing LT F2/C1/C3/G1 contracts, frozen before direct fixes."""
import json
from pathlib import Path

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets

from tests.xdart.scattering.test_stitch_calibration_fit_dialog import _wait


@pytest.fixture(scope='module')
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.mark.parametrize('name', ('Stitch', 'RSM'))
def test_unresolved_legacy_format_choice_survives_settings_roundtrip(qapp, tmp_path, name):
    from tests.xdart.scattering.test_lt_data_format import _page, _settings, _shut
    document_of, apply = _settings(name)
    page = _page(name, tmp_path)
    restored = _page(name, tmp_path / 'restored')
    try:
        document = document_of(page)
        document['fields'].pop('data_format')
        document['fields'].pop('input_mode')
        document['fields']['source_kind'] = 'Spec'
        document['calibration'] = {
            'fields': {'source': str(tmp_path / 'cal.nxs')},
            'choices': {'source_kind': {'data': 'nexus_stack', 'text': 'NeXus images + metadata'}},
        }
        apply(page, document)
        assert page._format_choice and 'NeXus' in page._format_choice
        saved = json.loads(json.dumps(document_of(page)))
        apply(restored, saved)
        assert restored._format_choice == page._format_choice, 'saving is not a user format choice'
        assert restored._calibration_settings_now()['fields']['source'] == str(tmp_path / 'cal.nxs')
        # Choosing the shown format explicitly settles the ambiguity, including after the next restore.
        editor = restored._rows['data_format'].editor
        editor.setFocus()
        QtTest.QTest.keyClick(editor, QtCore.Qt.Key.Key_Down)
        QtTest.QTest.keyClick(editor, QtCore.Qt.Key.Key_Up)
        qapp.processEvents()
        assert not restored._format_choice
        apply(page, json.loads(json.dumps(document_of(restored))))
        assert not page._format_choice
    finally:
        _shut(qapp, restored)
        _shut(qapp, page)


@pytest.mark.parametrize('draft', (False, True), ids=('unchanged', 'unfinished-valid-edit'))
def test_config_collapse_with_focused_zero_preserves_orientation(qapp, tmp_path, draft):
    from tests.xdart.scattering.test_orientation_diagnosis import dialog_at_fit
    from tests.xdart.scattering.test_lt_calibration_compact import _box, _config, _instrument, _press
    dialog = dialog_at_fit(tmp_path)
    dialog.show()
    dialog.activateWindow()
    qapp.processEvents()
    try:
        button, _card = _config(_box(_instrument(dialog), 'Diffractometer'))
        _press(button)
        zero = dialog.geometry_editor.zero_edit('del')
        zero.setFocus()
        qapp.processEvents()
        assert zero.hasFocus()
        if draft:
            from tests.xdart.scattering.test_calibration_quick_chain import _type
            _type(zero, '1.25')
        before = (dialog.geometry_editor.value(), dialog.fit, dialog._orientation_key())
        _press(button)
        _press(button)
        assert (dialog.geometry_editor.value(), dialog.fit, dialog._orientation_key()) == before
    finally:
        dialog.close()
        _wait(dialog)
        dialog.deleteLater()
        qapp.processEvents()


def test_missing_seed_file_cannot_make_staged_orientation_ready(qapp, tmp_path):
    from tests.xdart.scattering.test_lt_calibration_compact import _checklist, _orientation_row, _show, _staged
    from tests.xdart.scattering.test_calibration_quick_chain import _commit
    dialog, saved, _value = _staged(qapp, tmp_path)
    try:
        table = _checklist(dialog)
        _show(dialog, '4 · Calibrate and fit')
        dialog.refine_seed.setCurrentIndex(dialog.refine_seed.findData('file'))
        _commit(dialog._fields['refine_seed_path'], str(tmp_path / 'missing.json'))
        _wait(dialog)
        assert not dialog.orientation_button.isEnabled(), _orientation_row(table)
        assert _orientation_row(table)[0] == 'Needs input'
        # The same actual file becomes a valid fixed-geometry route after selection.
        _commit(dialog._fields['refine_seed_path'], str(saved))
        _wait(dialog)
        assert dialog.orientation_button.isEnabled(), _orientation_row(table)
        dialog.open_orientation_check()
        assert dialog.orientation_panel.run_button.isEnabled()
    finally:
        dialog.close()
        _wait(dialog)
        dialog.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize(('mode_1d', 'mode_2d'), (('q_oop', 'q_chi'), ('chi_gi', 'exit_angles')))
def test_other_gi_modes_record_frozen_ranges_and_reintegrate_exactly(tmp_path, mode_1d, mode_2d):
    from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _intent, _run_to_terminal, _write_tiff
    from tests.xdart.scattering.test_p1b_output_graph import write_poni
    from xdart.gui.tabs.scattering.display_values import StandardEventKind
    from xdart.gui.tabs.scattering.page import ScatteringWorkspace
    from xrd_tools.core.provenance import read_provenance
    from xrd_tools.io import open_scan
    from xrd_tools.io.output_transaction import capture_target_snapshot
    from xrd_tools.reduction import ReintegrateSuccessorPlan, run_reintegrate_successor
    raw = tmp_path / 'sample_0001.tif'
    _write_tiff(raw, 50)
    _write_tiff(tmp_path / 'sample_0002.tif', 60)
    poni = tmp_path / 'cal.poni'
    write_poni(poni)
    intent = _intent(raw, tmp_path / 'out', poni, processing_mode='Int 2D')
    intent.gi.enabled, intent.gi.incidence_motor, intent.gi.th_val = True, 'Manual', .2
    intent.gi.mode_1d, intent.gi.mode_2d = mode_1d, mode_2d
    executor, identity, events = _run_to_terminal(intent, request_value=9612)
    try:
        terminal = next(event for event in events if event.kind in _TERMINAL)
        assert terminal.kind is StandardEventKind.FINISHED, terminal.primary
    finally:
        executor.close(identity)
    artifact = next((tmp_path / 'out').rglob('*.nexus'))
    config = read_provenance(artifact)['config']
    run = config['run_configuration']
    assert run['bai_1d_args'].get('azimuth_range') == config['bai_1d_args']['azimuth_range']
    for key in ('radial_range', 'azimuth_range'):
        assert run['bai_2d_args'].get(key) == config['bai_2d_args'][key]
    scan = open_scan(artifact)
    labels = tuple(int(label) for label in scan.frame_indices)
    one_d = {label: (np.array(scan.get_1d(label).q), np.array(scan.get_1d(label).intensity)) for label in labels}
    two_d = {label: np.array(scan.get_2d(label).intensity) for label in labels}
    for dimension in ('1d', '2d'):
        plan = ReintegrateSuccessorPlan.from_artifact(
            artifact, entry='entry', dimension=dimension, source_root=str(tmp_path),
            expected_target_snapshot=capture_target_snapshot(artifact), expected_labels=labels,
            preparation=ScatteringWorkspace._reintegrate_preparation(intent, dimension))
        result = run_reintegrate_successor(plan)
        assert result.disposition == 'COMMITTED', result
        output = open_scan(Path(result.output_artifact))
        for label in labels:
            if dimension == '1d':
                np.testing.assert_array_equal(output.get_1d(label).q, one_d[label][0])
                np.testing.assert_array_equal(output.get_1d(label).intensity, one_d[label][1])
            else:
                np.testing.assert_array_equal(output.get_2d(label).intensity, two_d[label])
