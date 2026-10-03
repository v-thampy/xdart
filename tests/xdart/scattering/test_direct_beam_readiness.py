"""Measured endpoint direction and actual scan axes through the retained operation owner."""
import os
import time
from pathlib import Path
from threading import Event

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.calibration.dialog import CalibrationContext, CalibrationDialog
from xdart.gui.widgets.controls_panel import SectionCard
from xrd_tools.core.geometry import ImageOrientation
from xrd_tools.core.geometry.camera import stitch_camera
from xrd_tools.integrate.direct_beam import infer_direct_beam_motion
from tests.core.test_direct_beam import beam_images, config
from tests.xdart.scattering.test_stitch_guided_calibration import _beam_container


@pytest.fixture(scope='module')
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def detect(dialog):
    dialog._beam_probe_timer.stop()
    dialog._detect_beam()
    settle(dialog)


def settle(dialog, timeout=15):
    end = time.monotonic() + timeout
    while dialog.active() and time.monotonic() < end:
        QtWidgets.QApplication.processEvents()
        time.sleep(.005)
    assert not dialog.active(), dialog.status.text()


def test_blank_source_and_rough_tab_use_the_shared_readiness_header(qapp, tmp_path):
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    try:
        assert isinstance(dialog.beam_readiness, SectionCard)
        assert 'beam_motor' not in dialog._fields and not hasattr(dialog, 'motion')
        assert not dialog.beam_button.isEnabled()
        dialog._show_tab('rough')
        assert 'Select a direct-beam source' in dialog.status.text()
        assert dialog.beam_readiness.valid_marker.property('accent') == 'processing'
        dialog._show_tab('select')
        assert 'calibrant' in dialog.status.text()
        dialog._show_tab('rough')
        assert 'direct-beam' in dialog.status.text()
    finally:
        dialog.close()


@pytest.mark.parametrize('tab,context', [('instrument', 'diffractometer'), ('select', 'calibrant scan'),
                                       ('calibrate', 'Calibrate')])
def test_background_detection_finishes_with_the_current_tabs_context(qapp, tmp_path, tab, context):
    nexus, _, _ = _beam_container(tmp_path)
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    try:
        dialog.beam_source_kind.setCurrentIndex(dialog.beam_source_kind.findData('nexus'))
        dialog._set_field_text('beam_detector', 'Pilatus100k')
        dialog._set_field_text('beam_source', nexus)
        dialog._show_tab(tab)
        detect(dialog)
        assert dialog._beam_detection is not None, dialog.beam_readiness.status.text()
        assert context in dialog.status.text()
        assert 'progress' not in dialog.status.text().lower()
        assert 'beam_detect' not in dialog.status.text().lower()
    finally:
        dialog.close()


@pytest.mark.parametrize('command,columns,reason', [
    ('timescan 1 1', '#L Seconds\n0\n1', 'No scanned motor'),
    ('ascan eta -1 1 1 1', '#L eta\n-1\n1', 'Unsupported direct-beam motor eta'),
    ('a2scan del -1 1 nu -1 1 1 1', '#L del  nu\n-1 -1\n1 1', 'multiple scanned motors'),
])
def test_non_detector_or_ambiguous_scan_disables_fit(qapp, tmp_path, command, columns, reason):
    spec = tmp_path / 'direct.spec'
    spec.write_text('#F direct\n#O0 del  nu  eta\n#S 1 ' + command + '\n#P0 0 0 0\n#N 1\n' + columns + '\n')
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    try:
        dialog._set_field_text('beam_source', spec)
        detect(dialog)
        assert not dialog.beam_button.isEnabled()
        assert reason in dialog.beam_readiness.status.text()
    finally:
        dialog.close()


@pytest.mark.parametrize('rotation,motion', [(0, 'horizontal'), (90, 'vertical')])
def test_nexus_motion_comes_from_the_oriented_endpoints_and_ignores_old_settings(qapp, tmp_path, rotation, motion):
    nexus, _, _ = _beam_container(tmp_path)
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    try:
        dialog.apply_settings({'fields': {'beam_motor': 'eta'},
                               'choices': {'motion': {'data': None, 'text': 'horizontal'}}})
        dialog.beam_source_kind.setCurrentIndex(dialog.beam_source_kind.findData('nexus'))
        dialog._set_field_text('beam_detector', 'Pilatus100k')
        dialog.camera.set_value(stitch_camera(ImageOrientation(rotation=rotation)))
        dialog._set_field_text('beam_source', nexus)
        detect(dialog)
        assert dialog.beam_button.isEnabled(), dialog.beam_readiness.status.text()
        assert dialog._direct_beam_request()[1] == 'del'
        assert dialog._direct_beam_request()[2]['motion'] == motion
        assert dialog.beam_readiness.valid_marker.property('accent') == 'source'
        dialog.beam_button.click()
        settle(dialog)
        assert dialog._distance_result is not None, dialog.status.text()
        assert dialog._distance_result.distance_m == pytest.approx(.5, rel=1e-3)
    finally:
        dialog.close()


@pytest.mark.parametrize('fault', ['blank', 'stationary', 'diagonal', 'saturated'])
def test_unusable_or_ambiguous_endpoints_refuse_an_axis(fault):
    angles = [-1., 0., 1.]
    images, _, _ = beam_images(angles, zero=0)
    if fault == 'blank':
        images = [np.zeros_like(images[0])] * 3
    elif fault == 'stationary':
        images = [images[1]] * 3
    elif fault == 'diagonal':
        images = [images[0], images[1], np.roll(images[-1], 20, axis=0)]
    options = {'saturation': 100.} if fault == 'saturated' else {}
    with pytest.raises(ValueError, match='No usable|Ambiguous'):
        infer_direct_beam_motion(images, detector='Detector', detector_config=config(), **options)


def test_unusable_first_and_last_rows_use_next_valid_endpoints():
    images, _, _ = beam_images([-1., 0., 1.], zero=0)
    assert infer_direct_beam_motion([None, *images, None], detector='Detector', detector_config=config()) == 'horizontal'
    cancelled = Event(); cancelled.set()
    with pytest.raises(ValueError, match='cancelled'):
        infer_direct_beam_motion(images, detector='Detector', detector_config=config(), cancelled=cancelled)


def test_background_detection_cannot_adopt_after_source_or_camera_changes(qapp, tmp_path):
    nexus, _, _ = _beam_container(tmp_path)
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    try:
        dialog.beam_source_kind.setCurrentIndex(dialog.beam_source_kind.findData('nexus'))
        dialog._set_field_text('beam_detector', 'Pilatus100k')
        dialog._set_field_text('beam_source', nexus)
        dialog._beam_probe_timer.stop()
        dialog._detect_beam()
        assert dialog.active()
        dialog.camera.set_value(stitch_camera(ImageOrientation(rotation=90)))
        dialog._set_field_text('beam_source', '')
        dialog._beam_probe_timer.stop()
        settle(dialog)
        assert dialog._beam_detection is None and not dialog.beam_button.isEnabled()
        assert 'Select a direct-beam source' in dialog.beam_readiness.status.text()
        dialog._set_field_text('beam_source', nexus)
        detect(dialog)
        assert dialog._direct_beam_request()[2]['motion'] == 'vertical'
        dialog._set_field_text('beam_detector', 'Pilatus300k')
        assert dialog._beam_detection is None and not dialog.beam_button.isEnabled()
    finally:
        dialog.close()


def test_real_eiger_scan_fits_after_a_quarter_turn(qapp, tmp_path):
    root = Path(os.environ['XDART_TEST_DATA']) / 'stitching/GIXRD_Fatigue_Radha'
    assert (root / 'direct_beam_dist_p33m_scan001_master.h5').is_file()
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    try:
        dialog.beam_source_kind.setCurrentIndex(dialog.beam_source_kind.findData('spec_container'))
        dialog._set_field_text('beam_detector', 'Eiger2_500k')
        dialog.camera.set_value(stitch_camera(ImageOrientation(rotation=90)))
        dialog._set_field_text('beam_source', root / 'direct_beam_dist_p33m')
        detect(dialog)
        assert dialog._direct_beam_request()[1] == 'del'
        assert dialog._direct_beam_request()[2]['motion'] == 'horizontal'
        dialog.beam_button.click()
        settle(dialog)
        result = dialog._distance_result
        assert result is not None, dialog.status.text()
        assert result.distance_m == pytest.approx(.335321322566, rel=1e-7)
        assert result.used.sum() == 51
    finally:
        dialog.close()


def test_new_calibrant_spec_resets_source_overrides_and_auto_energy_keeps_manual_choice(qapp, tmp_path):
    first = tmp_path / 'first.spec'; second = tmp_path / 'second.spec'
    first.write_text('#F first\n#O0 energy\n#S 1 ascan del -1 1 1 1\n#P0 11270\n#N 1\n#L del\n-1\n1\n')
    second.write_text(first.read_text().replace('11270', '16000'))
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    try:
        dialog._set_field_text('source', first)
        dialog._source_details(); settle(dialog)
        assert dialog._text('rough_energy') == ''
        assert dialog._fields['rough_energy'].placeholderText() == 'Auto · 11.2700 keV from SPEC'
        for key, value in [('scan', '99'), ('images', tmp_path / 'old'), ('stem', 'old'), ('rough_energy', '18')]:
            dialog._set_field_text(key, value)
        dialog._set_field_text('source', second)
        assert [dialog._text(key) for key in ('scan', 'images', 'stem', 'rough_energy')] == ['1', '', '', '']
        dialog._source_details(); settle(dialog)
        assert dialog._fields['rough_energy'].placeholderText() == 'Auto · 16.0000 keV from SPEC'
        dialog._set_field_text('rough_energy', '17')
        dialog._source_details(); settle(dialog)
        assert dialog._text('rough_energy') == '17'
        assert dialog.settings()['fields']['rough_energy'] == '17'
        dialog._set_field_text('rough_energy', '')
        assert dialog.settings()['fields']['rough_energy'] == ''
        dialog._set_field_text('scan', '99')
        dialog._source_details(); settle(dialog)
        assert dialog._source_energy_eV is None
        assert 'unavailable' in dialog._fields['rough_energy'].placeholderText()
        assert dialog._rough_wavelength_A() is None
        assert dialog._parts['entry'][0] == 'source_kind'
    finally:
        dialog.close()


def test_distance_seed_routes_by_verified_source_and_frames_keep_their_recorded_source(qapp, tmp_path):
    nexus, _, _ = _beam_container(tmp_path)
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    try:
        dialog.beam_source_kind.setCurrentIndex(dialog.beam_source_kind.findData('nexus'))
        dialog._set_field_text('beam_detector', 'Pilatus100k')
        dialog._set_field_text('beam_source', nexus)
        detect(dialog); dialog.beam_button.click(); settle(dialog)
        dialog._set_field_text('source', tmp_path / 'stale.spec')
        dialog._source_details(); settle(dialog)
        dialog.apply_distance_button.click()
        assert dialog.tabs.currentWidget() is dialog._tab_pages['select']
        dialog.source_kind.setCurrentIndex(dialog.source_kind.findData('nexus_stack'))
        dialog._set_field_text('source', nexus)
        dialog._source_details(); settle(dialog)
        assert dialog._calibrant_source_ready()
        dialog.apply_distance_button.click()
        assert dialog.tabs.currentWidget() is dialog._tab_pages['calibrate']
        assert str(nexus) in dialog.active_source_row.editor.text()
        import shutil
        shutil.copyfile(nexus, tmp_path / 'beam_scan001_master.h5')
        spec = tmp_path / 'beam'
        spec.write_text('#F beam\n#O0 nu  energy\n#S 1 ascan del 1 -1 4 1\n#P0 4 16000\n#N 1\n#L del\n1\n.5\n0\n-.5\n-1\n')
        dialog.source_kind.setCurrentIndex(dialog.source_kind.findData('spec_container'))
        dialog._set_field_text('source', spec)
        dialog._enumerate(); settle(dialog)
        dialog._prepare(); settle(dialog)
        assert dialog.frames, dialog.status.text()
        assert str(spec) in dialog.active_source_row.editor.text()
        dialog._set_field_text('source', tmp_path / 'another.nxs')
        assert str(spec) in dialog.active_source_row.editor.text()
        assert 'Selected source differs' in dialog.active_source_row.editor.text()
        dialog._fit()
        assert 'another calibrant source' in dialog.status.text()
        dialog._refine()
        assert 'valid calibrant source' in dialog.status.text()
    finally:
        dialog.close()


def test_changed_source_cannot_adopt_old_energy_or_stem(qapp, tmp_path):
    first = tmp_path / 'first.spec'; other = tmp_path / 'other.spec'
    first.write_text('#F first\n#O0 energy\n#S 1 ascan del -1 1 1 1\n#P0 11270\n#N 1\n#L del\n-1\n1\n')
    other.write_text(first.read_text().replace('11270', '16000'))
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    try:
        dialog._set_field_text('source', first)
        dialog._source_details()
        dialog._set_field_text('source', other)
        settle(dialog)
        assert dialog._source_energy_eV is None
        assert 'unavailable' in dialog._fields['rough_energy'].placeholderText()
        dialog._source_details(); settle(dialog)
        assert dialog._source_energy_eV == 16000
    finally:
        dialog.close()
