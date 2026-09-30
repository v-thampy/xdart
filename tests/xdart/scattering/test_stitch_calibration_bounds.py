"""Calibrate window: one motor-zero estimate per circle (the Diffractometer panel) and the fits'
freedom about the estimates under Advanced (ARM-ZERO-BOUNDS-20260929; maintainer, 2026-09-29:
"this is to make sure the sample displacement in particular doesn't go to unrealistic values")."""
import json

import numpy as np
import pytest

from xrd_tools.core.geometry import Diffractometer
from tests.core.test_stacked_arm_extension import TRUTH
from tests.xdart.scattering.test_stitch_calibration_fit_dialog import (  # noqa: F401  (qapp)
    _ROW_ANGLES, _five_parameter_seed, _guided_frames, _page_with_dialog, _ring_scan, _wait, qapp,
)

SOURCES = {'del_angle': 'del', 'nu_angle': 'nu'}
#: TRUTH's motor zeros (orientation 3: zero = −offset / (π/180))
TRUTH_ZEROS = {'del': -np.rad2deg(TRUTH['del_offset']), 'nu': -np.rad2deg(TRUTH['nu_offset'])}


def _type_zeros(dialog, zeros):
    for circle, zero in zeros.items():
        dialog.geometry_editor.zero_edit(circle).setText(f'{zero:.12g}')


def test_the_direct_beam_and_the_rough_arm_read_the_panels_zeros(qapp, tmp_path):
    """One zero per circle: 'Known normal-incidence zero' is gone; the rough arm puts both moving
    circles at their zeros and the direct-beam fit's tangent zero is the beam motor's; an older
    settings file's zero moves into the panel, never over one already there."""
    page, dialog = _page_with_dialog(tmp_path)
    try:
        assert 'beam_zero' not in dialog._fields
        _type_zeros(dialog, {'del': 2.3, 'nu': -.8})
        dialog.rough_mode.setCurrentIndex(dialog.rough_mode.findData('manual'))
        for key, value in (('rough_distance', '390'), ('beam_pixel', '700, 95'), ('rough_wavelength', '0.7293')):
            dialog._fields[key].setText(value)
        record = dialog._rough_arm()
        assert Diffractometer.from_pyfai_goniometer(record, source_motors=SOURCES).stacked_arm.motor_zeros() \
            == pytest.approx({'del': 2.3, 'nu': -.8})
        started = []
        dialog._start = lambda kind, arguments: started.append((kind, arguments))
        dialog.rough_mode.setCurrentIndex(dialog.rough_mode.findData('scan'))
        dialog._fields['beam_source'].setText(str(tmp_path / 'beam'))
        dialog._distance()
        (kind, (_spec, motor, options)), = started
        assert kind == 'distance' and motor == 'del' and options['angle_zero_deg'] == 2.3
        dialog._fields['beam_motor'].setText('nu')  # a nu scan: nu's zero
        started.clear()
        dialog._distance()
        assert started[0][1][2]['angle_zero_deg'] == -.8
    finally:
        dialog.close()
        page.shutdown()
    page, dialog = _page_with_dialog(tmp_path / 'older')
    try:
        dialog.apply_settings({'fields': {'beam_zero': '2.3', 'beam_motor': 'del'},
                               'instrument_geometry': {'geometry': 'psic', 'motors': {'del': 'del', 'nu': 'nu'}}})
        assert dialog.geometry_editor.value().arm_zeros() == {'del': 2.3}
        dialog.apply_settings({'fields': {'beam_zero': '2.3', 'beam_motor': 'del'},
                               'instrument_geometry': {'geometry': 'psic', 'motors': {'del': 'del', 'nu': 'nu'},
                                                       'zeros': {'del': 1.5}}})
        assert dialog.geometry_editor.value().arm_zeros() == {'del': 1.5}  # the panel's own stays
        dialog.apply_settings({'fields': {'beam_zero': '1.2', 'beam_motor': 'tth'},
                               'instrument_geometry': {'geometry': 'twoc', 'motors': {'tth': 'tth', 'th': 'th'}}})
        assert dialog.geometry_editor.value().zeros == (('tth', 1.2),)
    finally:
        dialog.close()
        page.shutdown()


def test_the_advanced_bounds_reach_the_fit_and_its_summary_reports_contact(qapp, tmp_path):
    (tmp_path / 'frames').mkdir()
    frames = _guided_frames(tmp_path)  # TRUTH: the sample 0.5–2.2 mm off the rotation centre
    page, dialog = _page_with_dialog(tmp_path)
    try:
        assert {key: dialog._text(key) for key in dialog._BOUND_DEFAULTS} == {
            'zero_freedom': '2', 'mount_freedom': '2', 'displacement_limit': '0.05', 'distance_freedom': '5'}
        dialog.frames, dialog._frame_motors = frames, ('del', 'nu')
        dialog._render_frames()
        _type_zeros(dialog, TRUTH_ZEROS)
        assert 'Fits start from: del zero 2.86' in dialog.bounds_note.text()
        dialog._fields['displacement_limit'].setText('0.02')
        dialog.fit_button.click()
        _wait(dialog)
        assert dialog.fit is not None, dialog.status.text()
        recorded = dialog.fit.asset['provenance']['fit_bounds']
        assert recorded['estimates']['sample_offset_mm'] == .02
        assert recorded['estimates']['zeros_deg'] == pytest.approx(TRUTH_ZEROS)
        assert recorded['estimates']['zero_sources'] == {'del': 'typed', 'nu': 'typed'}
        table = recorded['parameters']
        assert all(abs(table[f'sample_offset{k}_mm']['fitted']) <= .02 * (1 + 1e-9) for k in (1, 2, 3))
        assert any(table[f'sample_offset{k}_mm']['at_bound'] for k in (1, 2, 3))
        summary = dialog.fit_summary.toPlainText()
        assert 'Fit parameters (initial → fitted, correction; bounds)' in summary and 'AT BOUND' in summary
        assert 'sample offset 1: 0 →' in summary and 'del zero: 2.86479' in summary
        assert 'at a fit bound: ' in dialog.fit.message
        dialog._fields['zero_freedom'].setText('x')  # a malformed bound is refused, never guessed
        dialog.fit_button.click()
        _wait(dialog)
        assert dialog.fit is None and 'Zero freedom' in dialog.status.text()
        dialog.reset_bounds_button.click()
        assert {key: dialog._text(key) for key in dialog._BOUND_DEFAULTS} == dialog._BOUND_DEFAULTS
        settings = dialog.settings()
        assert settings['instrument_geometry']['zeros'] == pytest.approx(TRUTH_ZEROS)
        assert settings['fields']['displacement_limit'] == '0.05'
    finally:
        dialog.close()
        page.shutdown()


def test_editing_a_zero_or_a_bound_leaves_the_fit_and_the_prediction_stale(qapp, tmp_path):
    (tmp_path / 'frames').mkdir()
    frames = _guided_frames(tmp_path)
    page, dialog = _page_with_dialog(tmp_path)
    try:
        dialog.frames, dialog._frame_motors = frames, ('del', 'nu')
        _type_zeros(dialog, TRUTH_ZEROS)
        dialog._fields['displacement_limit'].setText('5')
        for edit in (lambda: dialog.geometry_editor.zero_edit('del').setText('2.9'),
                     lambda: dialog._fields['mount_freedom'].setText('1'),
                     lambda: dialog._fields['rough_mount'].setText('.5, 0')):
            dialog.fit_button.click()
            _wait(dialog)
            assert dialog.fit is not None, dialog.status.text()
            dialog._prediction = (dialog._calibrated_key(), {'predicted': True})
            assert dialog._seed_options()['prediction'] == {'predicted': True}
            edit()
            assert dialog.fit is None and not dialog.save_button.isEnabled()
            assert dialog._seed_options()['prediction'] is None  # the prediction's bounds changed
    finally:
        dialog.close()
        page.shutdown()


@pytest.mark.parametrize('typed', [False, True])
def test_refine_starts_from_the_seed_calibrations_zeros_unless_one_is_typed(qapp, tmp_path, typed):
    spec = _ring_scan(tmp_path / 'scan')
    seed = _five_parameter_seed(_ROW_ANGLES)
    seed_zeros = Diffractometer.from_pyfai_goniometer(seed, source_motors=SOURCES).stacked_arm.motor_zeros()
    seed_path = tmp_path / 'calibration' / 'five.json'
    seed_path.parent.mkdir()
    seed_path.write_text(json.dumps(seed))
    page, dialog = _page_with_dialog(tmp_path, spec, seed_path)
    try:
        dialog._fields['refine_frames'].setText('12')
        dialog._fields['displacement_limit'].setText('5')  # TRUTH's sample sits mm off the centre
        if typed:
            _type_zeros(dialog, {'del': 2.5})
        assert dialog.refine_seed.currentData() == 'file'
        dialog.refine_button.click()
        _wait(dialog, timeout=120)
        assert dialog.fit is not None, dialog.status.text()
        estimates = dialog.fit.asset['provenance']['fit_bounds']['estimates']
        if typed:
            assert estimates['zero_sources'] == {'del': 'typed', 'nu': 'seed'}
            assert estimates['zeros_deg'] == pytest.approx({'del': 2.5, 'nu': seed_zeros['nu']})
        else:
            assert estimates['zero_sources'] == {'del': 'seed', 'nu': 'seed'}
            assert estimates['zeros_deg'] == pytest.approx(seed_zeros)
        assert estimates['sample_offset_mm'] == 5.
        assert 'Fit parameters (initial → fitted' in dialog.fit_summary.toPlainText()
    finally:
        dialog.close()
        page.shutdown()
