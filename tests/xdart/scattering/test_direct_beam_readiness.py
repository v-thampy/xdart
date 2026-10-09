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


@pytest.mark.parametrize('outcome', ['complete', 'refused', 'cancelled'])
def test_job_buttons_wait_for_the_background_probe_and_keep_the_fit(qapp, tmp_path, monkeypatch, outcome):
    """CAL-BUSY-ACTION-DISCARDS-FIT (maintainer, 2026-10-09: "disable buttons if possible during probe"): the background
    direct-beam probe keeps the tabs open for typing, so the buttons that start a job wait while it runs, and Refine or
    Fit asked for anyway keeps the current fit and says to wait. Before, either discarded the fit and started nothing
    (_start drops a job while one runs). The buttons return however the probe ends: complete, refused or cancelled."""
    import xrd_tools.integrate.direct_beam as direct_beam
    from tests.core.test_stitch_autocalibration import ROW_ANGLES, five_parameter_seed, ring_scan
    from xrd_tools.integrate.stitch_autocalibration import refine_arm_from_scan
    fit = refine_arm_from_scan(ring_scan(tmp_path / 'scan', [(d, 0.) for d in (6., 20., 34.)]),
                               five_parameter_seed(ROW_ANGLES), max_frames=3)
    entered, release = Event(), Event()
    original = direct_beam.infer_direct_beam_motion

    def held(*args, **kwargs):
        entered.set()
        assert release.wait(30)
        if outcome == 'refused':
            raise ValueError('the direct-beam scan does not move the beam')
        return original(*args, **kwargs)
    monkeypatch.setattr(direct_beam, 'infer_direct_beam_motion', held)
    nexus, _, _ = _beam_container(tmp_path)
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    buttons = (dialog.list_button, dialog.prepare_button, dialog.calibrate_button, dialog.fit_button,
               dialog.refine_button)
    try:
        dialog.fit = fit
        dialog.beam_source_kind.setCurrentIndex(dialog.beam_source_kind.findData('nexus'))
        dialog._set_field_text('beam_detector', 'Pilatus100k')
        dialog._set_field_text('beam_source', nexus)
        dialog._beam_probe_timer.stop()
        dialog._detect_beam()
        assert entered.wait(10) and dialog._job == 'beam_detect' and dialog._owner.busy
        QtWidgets.QApplication.processEvents()
        assert all(dialog.tabs.isTabEnabled(index) for index in range(dialog.tabs.count()))  # typing goes on
        assert [button.text() for button in buttons if button.isEnabled()] == []
        dialog.refine_button.click()  # a disabled button takes no press
        assert dialog.fit is fit and dialog._job == 'beam_detect'
        for asked in (dialog._refine, dialog._fit):  # asked for anyway: the fit stays and the status says to wait
            asked()
            assert dialog.fit is fit and 'wait' in dialog.status.text().lower(), dialog.status.text()
        if outcome == 'cancelled':
            dialog.cancel_button.click()
        release.set()
        settle(dialog)
        assert [button.text() for button in buttons if not button.isEnabled()] == []
        assert dialog.fit is fit
    finally:
        release.set()
        dialog.close()


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


@pytest.mark.parametrize('rotation,warning', [
    (270, None),
    (90, 'The direct beam moved the opposite way to this geometry: del -1 → 1° moved it along the rows from '),
    (0, 'The direct beam moved across this geometry: del -1 → 1° moved it along the columns from '),
])
def test_a_fit_whose_beam_runs_against_the_arm_says_how_the_detector_is_mounted(qapp, tmp_path, rotation, warning):
    """The container's beam moves to smaller stored columns as del rises. Image turn 270 (camera z+, x+: columns
    up, the mounting rotation 90° CW) moves it with the arm; turn 90 (z−, x−, 270° CW) fits it as well, but its
    geometry moves the beam the other way; turn 0 (x+, z−) moves it across the columns del moves it along. The
    fit used to keep |slope| and say nothing (maintainer, 2026-10-06: "why 90 degrees doesn't work"); now the
    fit's status, its window and the distance rows say so, with the mounting to try."""
    nexus, _, _ = _beam_container(tmp_path)
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    try:
        dialog.beam_source_kind.setCurrentIndex(dialog.beam_source_kind.findData('nexus'))
        dialog._set_field_text('beam_detector', 'Pilatus100k')
        dialog.camera.set_value(stitch_camera(ImageOrientation(rotation=rotation)))
        dialog._set_field_text('beam_source', nexus)
        detect(dialog)
        dialog.beam_button.click()
        settle(dialog)
        assert dialog._distance_result is not None, dialog.status.text()
        assert dialog._distance_result.distance_m == pytest.approx(.5, rel=1e-3)
        shown = (dialog.status.text(), dialog._direct_beam_fit_window.summary.text(),
                 dialog.beam_summary.toPlainText().splitlines()[0])
        if warning is None:
            assert not any('direct beam moved' in text for text in shown), shown
        else:
            assert all(warning in text for text in shown), shown
        if rotation == 90:
            assert all('try 90° CW rather than 270° CW' in text for text in shown), shown
    finally:
        dialog.close()


def _fit_container(dialog, tmp_path, rotation):
    """The five-frame del container fitted under image turn ``rotation``; its result."""
    nexus, _, _ = _beam_container(tmp_path)
    dialog.beam_source_kind.setCurrentIndex(dialog.beam_source_kind.findData('nexus'))
    dialog._set_field_text('beam_detector', 'Pilatus100k')
    dialog.camera.set_value(stitch_camera(ImageOrientation(rotation=rotation)))
    dialog._set_field_text('beam_source', nexus)
    detect(dialog)
    dialog.beam_button.click()
    settle(dialog)
    assert dialog._distance_result is not None, dialog.status.text()
    return dialog._distance_result


def _surfaces(dialog):
    """The advice's three surfaces: the status line, the fit window's summary, the first distance row."""
    return (dialog.status.text(), dialog._direct_beam_fit_window.summary.text(),
            dialog.beam_summary.toPlainText().splitlines()[0])


def _assert_advice(dialog, *phrases):
    shown = _surfaces(dialog)
    assert all(phrase in text for text in shown for phrase in phrases), shown


def _assert_no_advice(dialog):
    shown = _surfaces(dialog)
    assert not any('direct beam moved' in text for text in shown), shown
    assert shown[0].startswith('Direct-beam distance: 500 mm') and shown[2].startswith('Distance: 500 mm'), shown


def _reverse(dialog, circle):
    """The Diffractometer section's sense box for ``circle`` turned over (its data: 1 psic's, −1 reversed)."""
    box = dialog.geometry_editor.axis_box(circle)
    box.setCurrentIndex(box.findData(-box.currentData()))


def test_a_sense_edit_after_the_fit_updates_its_advice_everywhere(qapp, tmp_path):
    """Codex review B1: the advice is about the arm the controls describe now, not the one at the fit. Turn 270
    (mounting 90° CW) agrees; reversed, del's arm moves the beam backwards; restored, it agrees again. The
    fit itself (its distance, held reading and wavelength) never changes."""
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    try:
        result = _fit_container(dialog, tmp_path, 270)
        wavelength = dialog._rough_wavelength_A()
        _assert_no_advice(dialog)
        _reverse(dialog, 'del')
        _assert_advice(dialog, 'The direct beam moved the opposite way to this geometry', "or check del's sense")
        _reverse(dialog, 'del')
        _assert_no_advice(dialog)
        assert dialog._distance_result is result and dialog._distance_fresh()
        assert dialog._distance_held == {'nu': 4.} and dialog._rough_wavelength_A() == wavelength
    finally:
        dialog.close()


def test_correcting_the_sense_clears_the_advice_everywhere(qapp, tmp_path):
    """Turn 90 (mounting 270° CW) disagrees with the container's beam; with del reversed its arm agrees."""
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    try:
        _fit_container(dialog, tmp_path, 90)
        _assert_advice(dialog, 'The direct beam moved the opposite way to this geometry',
                       'try 90° CW rather than 270° CW')
        _reverse(dialog, 'del')
        _assert_no_advice(dialog)
    finally:
        dialog.close()


def test_a_camera_edit_after_the_fit_asks_for_a_new_fit_not_old_advice(qapp, tmp_path):
    """A camera edit leaves the fit stale (its extraction turned the images): the old mounting advice goes from
    every surface and the status asks for a new fit, which the scan read again under the new camera still needs."""
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    try:
        _fit_container(dialog, tmp_path, 90)
        _assert_advice(dialog, 'try 90° CW rather than 270° CW')
        dialog.camera.set_value(stitch_camera(ImageOrientation(rotation=270)))
        assert dialog.status.text() == dialog._STALE_DISTANCE
        assert dialog._distance_warning is None
        assert not dialog._direct_beam_fit_window.summary.text().startswith('⚠')
        assert dialog.beam_summary.toPlainText().startswith('Distance: 500 mm')
        detect(dialog)
        assert not dialog._distance_fresh() and dialog._distance_warning is None
        assert not dialog.beam_summary.toPlainText().startswith('⚠')
    finally:
        dialog.close()


def test_saving_a_geometry_against_its_scan_keeps_the_advice_beside_the_saved_pair(qapp, tmp_path, monkeypatch):
    """The rough save is of the arm the advice is about: a reversed del arm saved over a scan it contradicts says
    so with the saved pair (advice only: the pair is written)."""
    from tests.xdart.scattering.test_stitch_calibration_fit_dialog import _save_to, _wait
    # a fresh session keeps the saved folder out of the shared session's browse history, which
    # test_a_saved_calibration_is_named_after_its_scan_and_never_over_another expects empty
    monkeypatch.setenv('XDART_SESSION_FRESH', '1')
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    try:
        _fit_container(dialog, tmp_path, 270)
        _reverse(dialog, 'del')
        target = (tmp_path / 'calibration' / 'reversed-del.json').resolve()
        _save_to(monkeypatch, target)
        dialog.save_rough_button.click()
        _wait(dialog, timeout=120)
        assert target.is_file(), dialog.status.text()
        advice, saved = dialog.status.text().split('\n', 1)
        assert advice.startswith('⚠ The direct beam moved the opposite way to this geometry'), advice
        assert target.name in saved, saved
    finally:
        dialog.close()


@pytest.mark.parametrize('key,value', [('beam_pixel', '117, 243'), ('beam_strip', '3')])
def test_an_extraction_edit_after_the_fit_drops_its_advice_everywhere(qapp, tmp_path, key, value):
    """Codex B1 re-review: the beam pixel and strip are the fit's extraction inputs, so editing either leaves
    the fit stale; its mounting advice must go from every surface and the status ask for a new fit."""
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    try:
        _fit_container(dialog, tmp_path, 90)
        _assert_advice(dialog, 'try 90° CW rather than 270° CW')
        dialog._set_field_text(key, value)
        assert not dialog._distance_fresh() and dialog._distance_warning is None
        assert dialog.status.text() == dialog._STALE_DISTANCE
        assert not dialog._direct_beam_fit_window.summary.text().startswith('⚠')
        assert dialog.beam_summary.toPlainText().startswith('Distance: 500 mm')
    finally:
        dialog.close()


def test_a_mount_edit_is_advice_about_the_arm_it_makes(qapp, tmp_path):
    """The mount's tilt and roll are step 1's arm, not the fit's identity: the fit stays fresh. A mount the arm
    cannot be built from leaves nothing to compare (no advice, the distance shown); a valid one brings it back."""
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    try:
        _fit_container(dialog, tmp_path, 90)
        _assert_advice(dialog, 'try 90° CW rather than 270° CW')
        dialog._set_field_text('rough_mount', '1, 2, 3')
        assert dialog._distance_fresh() and dialog._distance_warning is None
        _assert_no_advice(dialog)
        dialog._set_field_text('rough_mount', '0, 0')
        _assert_advice(dialog, 'try 90° CW rather than 270° CW')
    finally:
        dialog.close()


def test_a_typed_rough_geometry_has_no_scan_to_compare(qapp, tmp_path):
    """With Rough geometry typed rather than measured no scanned motor is known, so there is no advice to show."""
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    try:
        _fit_container(dialog, tmp_path, 90)
        _assert_advice(dialog, 'try 90° CW rather than 270° CW')
        dialog.rough_mode.setCurrentIndex(dialog.rough_mode.findData('manual'))
        assert dialog._distance_warning is None
        assert not any(text.startswith('⚠') for text in _surfaces(dialog)), _surfaces(dialog)
    finally:
        dialog.close()


def test_every_field_the_fit_identity_and_the_rough_arm_read_refreshes_the_advice(qapp, tmp_path):
    """Codex B1 re-review owner checkpoint: the trigger list was hand-picked and missed fields. Record every
    field the fit's identity (its source, reader and extraction options) and step 1's arm read, across the
    source kinds and both rough modes, and require each to be one the advice follows."""
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    try:
        read, text = set(), dialog._text
        dialog._text = lambda key: read.add(key) or text(key)
        for key, value in (('beam_source', tmp_path / 'direct'), ('beam_scan', '1'), ('beam_images', tmp_path),
                           ('beam_stem', 'direct_scan1_'), ('beam_roi', '0, 0, 10, 10'), ('mask', tmp_path / 'm.npy'),
                           ('saturation', '1000'), ('beam_pixel', '97, 243'), ('rough_distance', '500'),
                           ('rough_energy', '16')):
            dialog._set_field_text(key, str(value))
        for kind in ('spec', 'nexus'):
            dialog.beam_source_kind.setCurrentIndex(dialog.beam_source_kind.findData(kind))
            dialog._spec(beam=True)
            dialog._beam_fit_options(with_mask=False)
        dialog.beam_source_kind.setCurrentIndex(dialog.beam_source_kind.findData('spec_container'))
        for key, value in (('beam_images', ''), ('beam_stem', ''), ('beam_image_file', tmp_path / 'b.h5')):
            dialog._set_field_text(key, str(value))
        dialog._spec(beam=True)
        dialog.rough_mode.setCurrentIndex(dialog.rough_mode.findData('manual'))
        assert dialog._rough_arm() is not None
        assert read <= set(CalibrationDialog._DIRECT_BEAM_ADVICE_FIELDS), sorted(
            read - set(CalibrationDialog._DIRECT_BEAM_ADVICE_FIELDS))
    finally:
        dialog.close()


def test_a_motion_check_that_fails_leaves_the_fit_shown(qapp, tmp_path, monkeypatch):
    """The motion check is advice: if it cannot run, the distance, its rows and its window still appear."""
    import xdart.gui.tabs.calibration.dialog as calibration_dialog

    def broken(*_args, **_kwargs):
        raise RuntimeError('motion check unavailable')
    monkeypatch.setattr(calibration_dialog, 'direct_beam_motion_mismatch', broken)
    nexus, _, _ = _beam_container(tmp_path)
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    try:
        dialog.beam_source_kind.setCurrentIndex(dialog.beam_source_kind.findData('nexus'))
        dialog._set_field_text('beam_detector', 'Pilatus100k')
        dialog.camera.set_value(stitch_camera(ImageOrientation(rotation=90)))
        dialog._set_field_text('beam_source', nexus)
        detect(dialog)
        dialog.beam_button.click()
        settle(dialog)
        assert dialog._distance_result is not None and dialog._distance_warning is None, dialog.status.text()
        assert dialog.status.text().startswith('Direct-beam distance: 500 mm'), dialog.status.text()
        assert dialog.beam_summary.toPlainText().startswith('Distance: 500 mm')
        assert dialog._direct_beam_fit_window.summary.text().startswith('Direct-beam distance: 500 mm')
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
        summary = dialog.beam_readiness.status.text()
        assert summary.startswith('del scan (-1, 1) · 51 pts · Δx ')
        assert ' px, Δy ' in summary and summary.endswith(')')
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
