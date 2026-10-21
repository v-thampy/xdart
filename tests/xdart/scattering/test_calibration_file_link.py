"""A selected committed calibration updates the real shared authoring window."""
import json

import pytest

from xdart.gui.pages.services import empty_host_services
from xrd_tools.core.geometry import ImageOrientation
from xrd_tools.core.geometry.camera import stitch_camera
from xrd_tools.integrate.calibration_seed import rough_arm_record
from xrd_tools.session.instrument import InstrumentGeometry
from tests.xdart.pages.test_stitch_page import _Status, qapp  # noqa: F401


def _geometry(path, rotation, distance):
    record = rough_arm_record(distance_m=distance, detector='Pilatus100k', beam_px=(97.25, 243.5),
                              wavelength_m=1e-10, camera=ImageOrientation(rotation=rotation))
    path.write_text(json.dumps(record))
    path.with_suffix('.json.provenance.json').write_text(json.dumps({
        'image_rotation': rotation, 'image_flip': False,
        'instrument_geometry': InstrumentGeometry().to_record(),
        'source_motors': {'del_angle': 'del', 'nu_angle': 'nu'},
    }))


@pytest.mark.parametrize('page_name', ('RSM', 'Stitch'))
def test_selected_file_updates_new_and_retained_window_and_same_file_keeps_edits(qapp, tmp_path, page_name):
    from xdart.gui.tabs.rsm.page import RSMPage
    from xdart.gui.tabs.stitch.page import StitchPage
    page = (RSMPage if page_name == 'RSM' else StitchPage)(empty_host_services(_Status()))
    first, second = tmp_path / 'first.json', tmp_path / 'second.json'
    _geometry(first, 90, 1.0147)
    _geometry(second, 270, .9)
    try:
        page.set_field('project_root', str(tmp_path))
        page.set_field('geometry_path', str(first))
        page._calibrate()
        dialog = page._calibration_dialog
        assert dialog.camera.value() == stitch_camera(ImageOrientation(rotation=90))
        assert float(dialog._text('rough_distance')) == pytest.approx(1014.7)
        assert tuple(map(float, dialog._text('beam_pixel').split(','))) == pytest.approx((97.25, 243.5))
        assert dialog._text('beam_detector') == 'Pilatus100k'
        assert dialog._text('refine_seed_path') == str(first)
        assert first.name in dialog.windowTitle()
        dialog._set_field_text('source', str(tmp_path / 'my_calibrant.spec'))
        dialog._set_field_text('rms', '.023')
        page.set_field('geometry_path', str(second))
        assert page._calibration_dialog is dialog
        assert dialog.camera.value() == stitch_camera(ImageOrientation(rotation=270))
        assert float(dialog._text('rough_distance')) == pytest.approx(900)
        assert dialog._text('source').endswith('my_calibrant.spec') and dialog._text('rms') == '.023'
        assert dialog.fit is None and not dialog.save_button.isEnabled()
        dialog._set_field_text('rough_distance', '901')
        dialog.hide()
        page._calibrate()
        assert dialog._text('rough_distance') == '901'
        # Saving new bytes at the same path is a new loaded geometry on reopening.
        _geometry(second, 90, .8)
        dialog.hide()
        page._calibrate()
        assert dialog.camera.value() == stitch_camera(ImageOrientation(rotation=90))
        assert float(dialog._text('rough_distance')) == pytest.approx(800)
        before = dialog.settings()
        bad = tmp_path / 'bad.json'
        bad.write_text('{}')
        page.set_field('geometry_path', str(bad))
        assert dialog.settings() == before
        assert 'load' in dialog.status.text().lower()
    finally:
        close = page.close_page if page_name == 'RSM' else page.shutdown
        close()
        page.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize('page_name', ('RSM', 'Stitch'))
def test_loaded_geometry_inputs_preserve_manual_values_and_explicit_edits_on_restore(qapp, tmp_path, page_name):
    from xdart.gui.tabs.calibration.dialog import CalibrationContext, CalibrationDialog
    from xrd_tools.core.energy import wavelength_m_to_energy_eV
    from xrd_tools.integrate.stitch_calibration import (
        CalibrationSessionResume, load_calibration_session, save_calibration_session,
    )
    from tests.xdart.scattering.test_calibration_input_defaults import _dispose, settle
    from tests.xdart.scattering.test_calibration_quick_chain import _type
    geometry = tmp_path / 'loaded.json'
    _geometry(geometry, 90, .5)
    record = json.loads(geometry.read_text())
    recorded = float(wavelength_m_to_energy_eV(record['wavelength'])) - 40.
    record['source_energy_eV'] = recorded
    geometry.write_text(json.dumps(record))
    dialog = CalibrationDialog(CalibrationContext(page_name, tmp_path))
    restored = CalibrationDialog(CalibrationContext(page_name, tmp_path))
    try:
        dialog.load_calibration(geometry)
        settle(dialog)
        assert dialog._text('rough_energy') == ''
        assert dialog._rough_arm()['wavelength'] == record['wavelength']
        assert dialog._rough_arm()['source_energy_eV'] == recorded
        assert dialog._beam_fit_options()['beam_pixel'] is None
        original = dialog._text('beam_pixel')
        dialog.camera.set_value(stitch_camera(ImageOrientation(rotation=270)))
        assert dialog._text('beam_pixel') != original
        assert dialog._beam_fit_options()['beam_pixel'] is None
        # Both page settings and the real session codec retain imported provenance, without rereading the file.
        saved = dialog.settings()
        (tmp_path / 'session').mkdir()
        session = save_calibration_session(tmp_path / 'session', (), resume=CalibrationSessionResume(saved))
        _, _, _, resume = load_calibration_session(session, with_resume=True)
        restored.apply_settings(resume.setup)
        settle(restored)
        assert restored.settings() == saved
        assert restored._beam_fit_options()['beam_pixel'] is None
        assert restored._rough_arm()['wavelength'] == record['wavelength']
        # Calibrated PONI wavelength is already effective: never add the imported offset again.
        from xrd_tools.core.containers import PONI
        from xrd_tools.integrate.refine import ControlFrame
        from tests.core.test_stitch_guided_calibration import _frames
        (tmp_path / 'poni').mkdir()
        control = ControlFrame([1, 2], [3, 4], [0, 0], {'del': 10., 'nu': 2.})
        restored.frames = _frames(tmp_path / 'poni', (control, control),
                                 PONI(dist=.5, poni1=.01, poni2=.02, wavelength=.8e-10,
                                      detector='Pilatus100k'))
        assert restored._rough_wavelength_A() == pytest.approx(.8, rel=1e-12)
        restored.frames = ()
        # A same-value user edit is still deliberate; camera transformations and restore must not erase it.
        _type(dialog._fields['beam_pixel'], dialog._text('beam_pixel'))
        expected = tuple(map(float, dialog._text('beam_pixel').split(',')))
        _type(dialog._fields['rough_energy'], repr(recorded / 1000))
        dialog.load_calibration(geometry)  # same-file reopening keeps both deliberate inputs
        assert dialog._beam_fit_options()['beam_pixel'] == expected
        assert dialog._rough_wavelength_A() == pytest.approx(record['wavelength'] * 1e10, rel=1e-12)
        restored.apply_settings(dialog.settings())
        assert restored._beam_fit_options()['beam_pixel'] == expected
        assert restored._text('rough_energy') == dialog._text('rough_energy')
    finally:
        # Return the fixture's camera before closing: its unsaved-setup confirmation is unrelated to this row.
        dialog.camera.set_value(stitch_camera(ImageOrientation(rotation=90)))
        for window in (dialog, restored):
            window.close()
            _dispose(window)
