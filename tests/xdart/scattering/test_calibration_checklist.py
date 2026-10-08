"""Frozen P1 C1/C2: owner facts, literal teaching steps and actual dialog actions."""
from dataclasses import replace
import re

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.calibration.dialog import CalibrationContext, CalibrationDialog
from xdart.gui.tabs.scattering.results_table import ResultsTable
from xdart.gui.widgets.controls_panel import SectionCard


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _table(dialog):
    return next((table for table in dialog.findChildren(ResultsTable)
                 if tuple(table.horizontalHeaderItem(c).text() for c in range(table.columnCount()))
                 == ('Step', 'State', 'What to do')), None)


def _state(dialog, step):
    return _table(dialog).item(step-1, 1).text()


def test_checklist_fact_matrix_matches_owner_states_and_topic_steps(qapp, tmp_path):
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    try:
        table = _table(dialog)
        assert table is not None and table.rowCount() == 8
        from xrd_tools.integrate.calibration_checklist import ChecklistFacts, calibration_checklist
        from xrd_tools.help_topics import get_topic
        body = get_topic('calibration.multi-step').text
        steps = tuple(' '.join(text.split()) for text in
            re.findall(r'^\d\. (.*?)(?=^\d\. |\n\nUnits)', body, re.M | re.S))
        assert len(steps) == 7
        for index, text in enumerate(steps):
            label = re.split(r'[:.]', text, maxsplit=1)[0]
            assert table.item(index, 0).text() == label
            assert table.item(index, 0).toolTip() == text
        base = ChecklistFacts()
        rows = calibration_checklist(base)
        assert tuple(row.state for row in rows) == ('Needs input',)*7+('Optional',)
        ready = replace(base, instrument_ready=True, rough_ready=True, source_ready=True,
            frame_count=3, complete_count=3, fit_present=True, saved_current_fit=True,
            instrument_message='configured', rough_message='measured', source_message='metadata ready',
            frame_message='3 Complete', fit_message='Converged; original diagnostic')
        rows = calibration_checklist(ready)
        assert tuple(row.state for row in rows) == ('Done',)*5+('Optional', 'Done', 'Optional')
        assert tuple(row.target for row in rows) == (
            'instrument', 'rough', 'select', 'calibrate', 'fit', 'inspect', 'save', 'orientation')
        assert rows[0].message == 'configured' and rows[1].message == 'measured'
        assert rows[2].message == 'metadata ready' and rows[3].message == '3 Complete'
        assert rows[4].message == rows[5].message == 'Converged; original diagnostic'
        for change, index, state in (
            ({'rough_mode': 'manual'}, 1, 'Done'),
            ({'rough_mode': 'skip'}, 1, 'Not applicable'),
            ({'rough_stale': True, 'rough_message': 'original stale reason'}, 1, 'Out of date'),
            ({'rough_ready': False, 'rough_message': 'original refusal'}, 1, 'Needs input'),
            ({'complete_count': 2, 'fit_present': False}, 3, 'Needs input'),
            ({'fit_present': False, 'saved_current_fit': False}, 6, 'Needs input'),
            ({'fit_present': False, 'fit_outdated': True}, 4, 'Out of date'),
        ):
            fact = replace(ready, **change)
            projected = calibration_checklist(fact)
            assert projected[index].state == state
            if 'rough_message' in change:
                assert projected[index].message == change['rough_message']
        # Loading a file does not manufacture a current saved ring fit.
        assert calibration_checklist(replace(base, instrument_ready=True))[6].state == 'Needs input'
        assert not any(word in row.message.lower() for row in rows for word in ('quality passed', 'good fit'))
    finally:
        dialog.close(); dialog.deleteLater(); qapp.processEvents()


def test_checklist_real_fourc_walk_navigation_and_invalidation(qapp, tmp_path, monkeypatch):
    from xrd_tools.session.instrument import InstrumentGeometry
    from xrd_tools.core.geometry import ImageOrientation
    from xrd_tools.core.geometry.camera import stitch_camera
    from tests.xdart.scattering.test_direct_beam_readiness import _fit_container
    from tests.xdart.scattering.test_stitch_calibration_fit_dialog import _single_arm_frames, _wait, _save_to
    from tests.core.test_single_detector_arm import _powder_scan
    from xrd_tools.integrate.stitch_calibration import capture_calibration_frames, adopt_calibration_pair
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path)); dialog.show()
    try:
        card = next((c for c in dialog.findChildren(SectionCard)
                     if c.title.text() == 'CALIBRATION CHECKLIST'), None)
        assert card is not None and dialog.layout().indexOf(card) < dialog.layout().indexOf(dialog.tabs)
        card.set_collapsed(False)
        dialog.set_instrument(InstrumentGeometry('fourc', motors=(('tth', 'del'),), plane='vertical'))
        _fit_container(dialog, tmp_path, 270); _wait(dialog)
        assert _state(dialog, 2) == 'Done'
        rough_target = tmp_path/'calibration'/'rough.json'
        _save_to(monkeypatch, rough_target); dialog.save_rough_button.click(); _wait(dialog)
        assert rough_target.exists(), dialog.status.text()
        assert _state(dialog, 7) != 'Done'
        # The next actual fixture is a different detector/setup, explicitly selected as such.
        dialog.rough_mode.setCurrentIndex(dialog.rough_mode.findData('skip'))
        dialog.set_instrument(InstrumentGeometry('fourc', plane='vertical'))
        dialog.camera.set_value(stitch_camera(ImageOrientation()))
        dialog._set_field_text('beam_detector', 'Pilatus300kw')
        fit_source = _powder_scan(tmp_path/'fit-scan', (6., 18., 30., 42.),
                                  {'nu_offset': .02, 'del_offset': -.05})
        dialog.set_source_context(fit_source, image_rotation=0); _wait(dialog)
        dialog.geometry_editor.zero_edit('tth').setText(f'{-np.rad2deg(.05):.12g}')
        (tmp_path/'frames').mkdir()
        pairs = _single_arm_frames(tmp_path)
        frames = capture_calibration_frames(dialog._spec(), (0, 1, 2, 3), dialog.session_dir,
                                            rotation=0, arm=('tth', None))
        dialog.frames = tuple(adopt_calibration_pair(frame, pair.poni_path, pair.npt_path)
                              for frame, pair in zip(frames, pairs))
        dialog._frame_motors = ('tth', None)
        dialog._render_frames()
        assert _state(dialog, 4) == 'Done' and _state(dialog, 5) == 'Needs input'
        dialog.fit_button.click(); _wait(dialog)
        assert dialog.fit is not None, dialog.status.text()
        assert _state(dialog, 5) == 'Done' and _state(dialog, 6) == 'Optional'
        fit = dialog.fit
        spec = _powder_scan(tmp_path/'scan', (6., 10.7, 15.3, 20., 24.7, 29.3, 34.),
                            {'nu_offset': .02, 'del_offset': -.05})
        dialog.set_source_context(spec, image_rotation=0); _wait(dialog)
        assert dialog.fit is fit and dialog._calibrant_source_ready()
        target = tmp_path/'calibration'/'ring.json'
        _save_to(monkeypatch, target); dialog.save_button.click(); _wait(dialog)
        assert target.exists() and _state(dialog, 7) == 'Done', dialog.status.text()
        # Every row navigates to an existing owner; activation does not start science.
        table = _table(dialog)
        for row in range(8):
            for column in range(3):
                assert table.item(row, column).toolTip()
            table.cellActivated.emit(row, 0)
            assert not dialog.active()
        for widget in (card, card.title, card.status, card.toggle, card.help_button, table):
            assert widget.toolTip()
        for column in range(3):
            assert table.horizontalHeaderItem(column).toolTip()
        dialog.refine_seed.setCurrentIndex(dialog.refine_seed.findData('fit'))
        dialog._fields['refine_frames'].setText('7')
        # These controls do not change the fit. Refine itself must clear the save marker.
        _save_to(monkeypatch, target)
        monkeypatch.setattr(QtWidgets.QMessageBox, 'question',
            staticmethod(lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes))
        dialog.save_button.click(); _wait(dialog)
        assert _state(dialog, 7) == 'Done'
        dialog.refine_button.click()
        assert dialog.active() and _state(dialog, 7) == 'Needs input'
        _wait(dialog)
        assert dialog.fit is not None and _state(dialog, 7) == 'Needs input', dialog.status.text()
        dialog._fields['mount_freedom'].setText('1')
        assert dialog.fit is None and _state(dialog, 7) == 'Needs input'
    finally:
        if dialog._saved_setup is not None and dialog._setup_state() != dialog._saved_setup:
            dialog.load_calibration(dialog._loaded_calibration_identity[0]); _wait(dialog)
        dialog.close(); _wait(dialog); dialog.deleteLater(); qapp.processEvents()
