"""Calibrate window: one motor-zero estimate per circle (the Diffractometer panel) and the fits'
freedom about the estimates under Advanced (ARM-ZERO-BOUNDS-20260929; maintainer, 2026-09-29:
"this is to make sure the sample displacement in particular doesn't go to unrealistic values")."""
import json
from pathlib import Path

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
        (kind, (_spec, motor, options, held)), = started
        assert kind == 'distance' and motor == 'del' and options['angle_zero_deg'] == 2.3
        assert held == ('nu',)  # its reading during the scan places the beam (the other circle)
        dialog._fields['beam_motor'].setText('nu')  # a nu scan: nu's zero
        started.clear()
        dialog._distance()
        assert started[0][1][2]['angle_zero_deg'] == -.8 and started[0][1][3] == ('del',)
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


def _captured(dialog):
    started = []
    dialog._start = lambda kind, arguments: started.append((kind, arguments))
    return started


@pytest.mark.parametrize('record,motor,zeros', [
    ({'geometry': 'psic', 'motors': {'del': 'del', 'nu': 'nu'}}, 'del', {'del': 2.3}),
    ({'geometry': 'twoc', 'motors': {'th': 'th', 'tth': 'tth'}}, 'tth', {'del': 2.3}),
    ({'geometry': 'twoc', 'motors': {'th': 'th', 'tth': 'tth'}, 'plane': 'horizontal'}, 'tth', {'nu': 2.3}),
])
def test_the_first_opening_keeps_an_older_windows_normal_incidence_zero(qapp, tmp_path, record, motor, zeros):
    """Codex review F1: the page read its instrument before the window migrated the zero, then put
    that older record back; the page's reader migrates it too, so the rough arm and the fits keep it."""
    from xdart.gui.pages.services import empty_host_services
    from xdart.gui.tabs.stitch.page import StitchPage
    from tests.xdart.scattering.test_stitch_calibration_fit_dialog import _Status
    page = StitchPage(empty_host_services(_Status()))
    page.set_field("project_root", str(tmp_path))
    page._set_calibration_settings({'fields': {'beam_zero': '2.3', 'beam_motor': motor}, 'choices': {},
                                    'instrument_geometry': record})
    assert page._instrument().arm_zeros() == zeros
    page._calibrate()
    dialog = page._calibration_dialog
    try:
        assert dialog.geometry_editor.value().arm_zeros() == zeros
        dialog.rough_mode.setCurrentIndex(dialog.rough_mode.findData('manual'))
        for key, value in (('rough_distance', '390'), ('beam_pixel', '240, 97'), ('rough_wavelength', '0.7293')):
            dialog._fields[key].setText(value)
        bound = ({'del_angle': 'del', 'nu_angle': 'nu'} if record['geometry'] == 'psic'
                 else {'nu_angle' if 'nu' in zeros else 'del_angle': 'tth'})  # a single arm: its one circle
        arm = Diffractometer.from_pyfai_goniometer(dialog._rough_arm(), source_motors=bound).stacked_arm
        moving = {'del': 0., 'nu': 0.} if record['geometry'] == 'psic' else {}
        assert arm.motor_zeros() == pytest.approx({**moving, **zeros})  # psic's nu at its reading 0
        started = _captured(dialog)
        dialog.fit_button.click()
        (kind, (_frames, options)), = started
        assert kind == 'fit' and dict(options['bounds'].zeros_deg) == zeros
    finally:
        dialog.close()
        page.shutdown()


def test_a_malformed_zero_is_refused_by_every_operation_and_a_typed_zero_is_supplied(qapp, tmp_path):
    """Codex review F4: '-' or '.' in a zero box is not a blank one: Fit, Refine, the direct-beam fit
    and Save as calibration refuse it by name and start nothing; an explicit 0 is typed."""
    page, dialog = _page_with_dialog(tmp_path)
    try:
        dialog.rough_mode.setCurrentIndex(dialog.rough_mode.findData('manual'))
        for key, value in (('rough_distance', '390'), ('beam_pixel', '240, 97'), ('rough_wavelength', '0.7293')):
            dialog._fields[key].setText(value)
        dialog.refine_seed.setCurrentIndex(dialog.refine_seed.findData('rough'))
        dialog._fields['source'].setText(str(tmp_path / 'calibrant'))
        started = _captured(dialog)
        for text in ('-', '.'):
            dialog.geometry_editor.zero_edit('del').setText(text)
            for button in (dialog.fit_button, dialog.refine_button, dialog.beam_button):
                button.click()
                assert started == [] and "del's motor zero" in dialog.status.text(), (text, button.text())
            dialog.save_rough_button.click()
            assert started == [] and "del's motor zero" in dialog.status.text()
        dialog.geometry_editor.zero_edit('del').setText('0')
        assert dialog._bounds().resolved(['del', 'nu']).zero_sources == (('del', 'typed'), ('nu', 'default'))
        dialog.geometry_editor.zero_edit('del').setText('')
        assert dialog._bounds().resolved(['del', 'nu']).zero_sources == (('del', 'default'), ('nu', 'default'))
    finally:
        dialog.close()
        page.shutdown()


def test_a_direct_beam_result_is_used_only_with_the_inputs_it_was_fitted_from(qapp, tmp_path):
    """Codex review F5: a changed zero (or source, or detector) leaves the direct-beam fit stale: its
    beam pixel and distance are not applied, saved or used for the rough arm until it is refitted."""
    import h5py
    from tests.core.test_direct_beam import beam_images
    angles = np.array([1., .5, 0., -.5, -1.])
    images, _, beam = beam_images(angles, shape=(65, 487), pitches=(516e-6, 172e-6), distance=.5, zero=0)
    spec = tmp_path / 'direct.spec'
    spec.write_text('#F direct\n#O0 eta\n#S 8 ascan del 1 -1 4 1\n#P0 0\n#N 1\n#L del\n'
                    + '\n'.join(map(str, angles)) + '\n')
    nexus = tmp_path / 'direct.nxs'
    with h5py.File(nexus, 'w') as handle:
        handle.create_group('measurement').attrs['NX_class'] = 'NXentry'
        handle['measurement/data/image'] = np.asarray(images).astype('uint32')
        handle['measurement/data/del'] = angles
    master = tmp_path / 'direct_master.h5'
    with h5py.File(master, 'w') as handle:
        handle.create_group('measurement').attrs['NX_class'] = 'NXentry'
        handle['measurement/data/data_000001'] = h5py.ExternalLink(nexus.name, '/measurement/data/image')
        handle['measurement/instrument/detector/detectorSpecific/nimages'] = 1
    page, dialog = _page_with_dialog(tmp_path)
    try:
        dialog.beam_source_kind.setCurrentIndex(dialog.beam_source_kind.findData('spec_container'))
        for key, value in dict(beam_source=spec, beam_image_file=master, beam_scan='8.1', beam_entry='measurement',
                               beam_detector='Pilatus100k', beam_pixel=', '.join(map(str, beam)),
                               rough_wavelength='0.7293').items():
            dialog._set_field_text(key, str(value))
        dialog.beam_button.click()
        _wait(dialog)
        assert dialog._distance_result is not None and dialog._distance_fresh(), dialog.status.text()
        assert dialog._rough_record() is not None
        dialog.geometry_editor.zero_edit('del').setText('.3')  # the zero the result was fitted at changed
        assert not dialog._distance_fresh() and dialog._rough_record() is None
        dialog.apply_distance_button.click()
        assert dialog._text('distance_seed') == '' and 'fit the direct-beam distance again' in dialog.status.text()
        dialog.save_rough_button.click()
        assert 'fit the direct-beam distance again' in dialog.status.text()
        dialog.geometry_editor.zero_edit('del').setText('')  # its own inputs again: it applies
        dialog.apply_distance_button.click()
        assert float(dialog._text('distance_seed')) == pytest.approx(500., rel=1e-4)
        dialog._fields['beam_detector'].setCurrentIndex(dialog._fields['beam_detector'].findData('Pilatus300kw'))
        assert not dialog._distance_fresh()  # another detector: stale too
    finally:
        dialog.close()
        page.shutdown()


def test_a_resumed_session_fits_and_predicts_with_its_own_zeros_and_bounds(qapp, tmp_path, monkeypatch):
    """Codex review F6: session A saved at del zero 2.3°, ±0.02 mm and a 385–395 mm distance box; the
    window then set to B's values; reopening A restores A's before its prediction starts. The
    Refine request and the prediction's key follow the distance box (Codex review F3)."""
    from xrd_tools.integrate.stitch_calibration import save_calibration_session
    (tmp_path / 'frames').mkdir()
    frames = _guided_frames(tmp_path)
    page, dialog = _page_with_dialog(tmp_path)
    try:
        dialog.frames, dialog._frame_motors = frames, ('del', 'nu')
        _type_zeros(dialog, {'del': 2.3})
        dialog._fields['displacement_limit'].setText('0.02')
        dialog._fields['distance_bounds'].setText('385, 395')
        (tmp_path / 'session-a').mkdir()
        path = save_calibration_session(tmp_path / 'session-a', frames, arm=('del', 'nu'),
                                        fit_policy=dialog._fit_policy())
        _type_zeros(dialog, {'del': 1., 'nu': .5})  # session B's window
        dialog._fields['displacement_limit'].setText('0.05')
        dialog._fields['distance_bounds'].setText('')
        started = _captured(dialog)
        monkeypatch.setattr('xdart.gui.widgets.file_dialogs.browse', lambda *args, **kwargs: [str(path)])
        dialog._resume()
        assert dialog.geometry_editor.value().arm_zeros() == {'del': 2.3}
        assert (dialog._text('displacement_limit'), dialog._text('distance_bounds')) == ('0.02', '385, 395')
        (kind, (key, seed)), = started
        assert kind == 'predict' and key == dialog._calibrated_key()
        assert dict(seed['bounds'].zeros_deg) == {'del': 2.3} and seed['bounds'].sample_offset_mm == .02
        assert seed['distance'] == {'distance_seed_m': None, 'distance_bounds_m': (.385, .395)}
        dialog._prediction = (dialog._calibrated_key(), {'predicted': True})
        dialog._fields['distance_bounds'].setText('380, 381')  # the prediction's box changed: stale
        assert dialog._seed_options()['prediction'] is None
        dialog._fields['distance_bounds'].setText('385, 395')
        dialog._prediction = (dialog._calibrated_key(), {'predicted': True})
        dialog._fields['distance_seed'].setText('390')  # and its distance seed: stale too
        assert dialog._seed_options()['prediction'] is None
        dialog._fields['distance_seed'].setText('')
        started.clear()
        dialog._fields['distance_bounds'].setText('385, 395')
        dialog.refine_seed.setCurrentIndex(dialog.refine_seed.findData('file'))
        seed_file = tmp_path / 'seed.json'
        seed_file.write_text(json.dumps(_five_parameter_seed(_ROW_ANGLES)))
        dialog._fields['refine_seed_path'].setText(str(seed_file))
        dialog._fields['source'].setText(str(tmp_path / 'calibrant'))
        dialog.refine_button.click()
        (kind, (_spec, _seed, options)), = started
        assert kind == 'refine' and options['distance_bounds_m'] == (.385, .395)
        older = tmp_path / 'session-b'
        older.mkdir()
        older = save_calibration_session(older, frames, arm=('del', 'nu'))  # version 3: no policy recorded
        monkeypatch.setattr('xdart.gui.widgets.file_dialogs.browse', lambda *args, **kwargs: [str(older)])
        started.clear()
        dialog._resume()
        assert 'records no fit bounds' in dialog.status.text()
        assert dialog.geometry_editor.value().arm_zeros() == {'del': 2.3}  # the window's own are kept
    finally:
        dialog.close()
        page.shutdown()
