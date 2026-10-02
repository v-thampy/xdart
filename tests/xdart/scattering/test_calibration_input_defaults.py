"""Real source commits, camera coordinates and chooser cancellation in the retained window."""
import time
import os
from pathlib import Path

import h5py
import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets

from xdart.gui.tabs.calibration.dialog import CalibrationContext, CalibrationDialog
from xrd_tools.core.energy import energy_eV_to_wavelength_m, wavelength_m_to_energy_eV
from xrd_tools.core.geometry import ImageOrientation
from xrd_tools.core.geometry.camera import stitch_camera


@pytest.fixture(scope='module')
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def settle(dialog, timeout=5):
    limit = time.monotonic() + timeout
    while dialog.active() and time.monotonic() < limit:
        QtWidgets.QApplication.processEvents()
        time.sleep(.005)  # release the GIL as the normal Qt event loop does for the worker
    assert not dialog.active()


def commit(dialog, key, text):
    dialog._set_field_text(key, str(text))
    QtTest.QTest.keyClick(dialog._fields[key], QtCore.Qt.Key.Key_Return)
    settle(dialog)


def test_detector_change_drops_the_previous_detectors_beam_pixel(qapp, tmp_path):
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    try:
        dialog._set_field_text('beam_detector', 'Pilatus100k')
        dialog._set_field_text('beam_pixel', '280, 97')
        dialog._set_field_text('beam_detector', 'Eiger2_500k')
        assert dialog._text('beam_pixel') == ''
        assert '1028 × 512' in dialog._image_size.text()
        assert 'found on the frame' in dialog._fields['beam_pixel'].placeholderText()
    finally:
        dialog.close()


@pytest.mark.parametrize('rotation,flip', [(r, f) for r in (0, 90, 180, 270) for f in (False, True)])
def test_typed_beam_pixel_follows_the_camera_but_not_pyfais_origin(qapp, tmp_path, rotation, flip):
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    try:
        dialog._set_field_text('beam_detector', 'Pilatus100k')
        dialog.camera.set_value(stitch_camera(ImageOrientation()))
        dialog._set_field_text('beam_pixel', '280, 97')
        image = np.zeros((195, 487), dtype=np.uint8)
        image[97, 280] = 1
        orientation = ImageOrientation(rotation=rotation, flip_horizontal=flip)
        dialog.camera.set_value(stitch_camera(orientation))
        y, x = np.argwhere(orientation.apply(image))[0]
        assert tuple(map(float, dialog._text('beam_pixel').split(','))) == (x, y)
        dialog._set_field_text('beam_orientation', '1')
        assert tuple(map(float, dialog._text('beam_pixel').split(','))) == (x, y)
    finally:
        dialog.close()


def test_escape_dismisses_a_real_file_chooser_and_keeps_calibration_open(qapp, tmp_path):
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    dialog.show()
    dismissed = []
    def escape():
        chooser = QtWidgets.QApplication.activeModalWidget()
        assert isinstance(chooser, QtWidgets.QFileDialog)
        QtTest.QTest.keyClick(chooser, QtCore.Qt.Key.Key_Escape)
        dismissed.append(True)
    try:
        QtCore.QTimer.singleShot(50, escape)
        dialog._browse('beam_source', 'spec')
        assert dismissed and dialog.isVisible() and dialog._text('beam_source') == ''
        # The release of Escape can reach the retained parent after a native chooser exits.
        QtTest.QTest.keyClick(dialog, QtCore.Qt.Key.Key_Escape)
        assert dialog.isVisible()
    finally:
        dialog.close()


def test_saved_wavelength_migrates_to_energy_with_the_same_physical_value(qapp, tmp_path):
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    try:
        dialog.apply_settings({'fields': {'rough_wavelength': '0.7293'}})
        energy = float(wavelength_m_to_energy_eV(.7293e-10)) / 1000
        assert dialog._formrows['rough_energy'].label.text() == 'Energy (keV)'
        assert dialog._parts['rough_energy'][0] == 'calibrant'
        assert dialog._formrows['rough_energy'].isAncestorOf(dialog._fields['rough_energy'])
        assert float(dialog._text('rough_energy')) == pytest.approx(energy, rel=1e-12)
        assert dialog._rough_wavelength_A() == pytest.approx(.7293, rel=1e-12)
        assert 'rough_wavelength' not in dialog.settings()['fields']
        again = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
        try:
            again.apply_settings(dialog.settings())
            assert again._rough_wavelength_A() == dialog._rough_wavelength_A()
        finally:
            again.close()
    finally:
        dialog.close()


def test_committed_spec_scan_updates_energy_without_reading_images(qapp, tmp_path):
    spec = tmp_path / 'beam'
    spec.write_text('#F beam\n#O0 energy\n#S 1 ascan del 1 -1 1 1\n#P0 11270\n#N 1\n#L del\n1\n-1\n'
                    '#S 2 ascan del 1 -1 1 1\n#P0 16000\n#N 1\n#L del\n1\n-1\n')
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    dialog.show()
    try:
        commit(dialog, 'source', spec)
        assert float(dialog._text('rough_energy')) == 11.27
        assert dialog._rough_wavelength_A() == pytest.approx(float(energy_eV_to_wavelength_m(11270)) * 1e10)
        commit(dialog, 'scan', '2')
        assert float(dialog._text('rough_energy')) == 16
        commit(dialog, 'scan', '99')
        assert dialog._text('rough_energy') == ''  # no stale scan energy or first-scan fallback
    finally:
        dialog.close()


def test_committed_nexus_wavelength_updates_energy(qapp, tmp_path):
    nexus = tmp_path / 'beam.nxs'
    with h5py.File(nexus, 'w') as handle:
        handle['entry/data/data'] = np.zeros((2, 3, 5), dtype=np.uint16)
        handle['entry/instrument/monochromator/wavelength'] = 1.1
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    dialog.show()
    try:
        dialog.source_kind.setCurrentIndex(dialog.source_kind.findData('nexus_stack'))
        commit(dialog, 'source', nexus)
        assert float(dialog._text('rough_energy')) == pytest.approx(float(wavelength_m_to_energy_eV(1.1e-10))/1000)
        assert dialog._rough_wavelength_A() == pytest.approx(1.1)
    finally:
        dialog.close()


def test_spec_container_uses_auto_folder_and_padded_stem(qapp, tmp_path):
    from tests.xdart.scattering.test_stitch_guided_calibration import _beam_container
    nexus, _, _ = _beam_container(tmp_path)
    master = tmp_path / 'beam_scan001_master.h5'
    nexus.rename(master)
    spec = tmp_path / 'beam'
    spec.write_text('#F beam\n#O0 nu  energy\n#S 1 ascan del 1 -1 4 1\n#P0 4 16000\n#N 1\n#L del\n'
                    '1\n.5\n0\n-.5\n-1\n')
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    dialog.show()
    try:
        dialog.beam_source_kind.setCurrentIndex(dialog.beam_source_kind.findData('spec_container'))
        commit(dialog, 'beam_source', spec)
        assert dialog._row_visible('beam_images') and dialog._row_visible('beam_stem')
        assert not dialog._row_visible('beam_image_file')
        assert 'beam_scan001_' in dialog._fields['beam_stem'].placeholderText()
        source = dialog._spec(beam=True)
        assert source.options['image_dir'] == str(tmp_path)
        assert source.options.get('image_file') is None  # the shared source chooses the actual container
        dialog._set_field_text('beam_detector', 'Pilatus100k')
        dialog.motion.setCurrentText('horizontal')  # the synthetic spot moves horizontally
        dialog.beam_button.click()
        settle(dialog)
        assert dialog._distance_result is not None, dialog.status.text()
        assert dialog._distance_result.distance_m == pytest.approx(.5, rel=1e-3)
    finally:
        dialog.close()


def test_direct_beam_source_does_not_overwrite_the_calibrants_energy(qapp, tmp_path):
    spec = tmp_path / 'direct'
    spec.write_text('#F direct\n#O0 energy\n#S 1 ascan del 1 -1 1 1\n#P0 11270\n#N 1\n#L del\n1\n-1\n')
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    try:
        dialog._set_field_text('rough_energy', '17')
        commit(dialog, 'beam_source', spec)
        assert dialog._text('rough_energy') == '17'
    finally:
        dialog.close()


@pytest.mark.parametrize('geometry,plane,motor,expected', [
    ('psic', 'vertical', 'delta', 'vertical'), ('psic', 'vertical', 'nu', 'horizontal'),
    ('fourc', 'vertical', 'tth', 'vertical'), ('twoc', 'horizontal', 'tth', 'horizontal'),
])
def test_auto_motion_uses_the_configured_circle_and_keeps_an_override(qapp, tmp_path, geometry, plane, motor, expected):
    from xrd_tools.session.instrument import InstrumentGeometry
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    try:
        dialog.set_instrument(InstrumentGeometry(geometry, (('del', 'delta'),) if geometry == 'psic' else (), plane))
        dialog._set_field_text('beam_source', tmp_path / 'source')
        dialog._set_field_text('beam_motor', motor)
        assert dialog._direct_beam_request()[2]['motion'] == expected
        dialog.motion.setCurrentText('horizontal' if expected == 'vertical' else 'vertical')
        assert dialog._direct_beam_request()[2]['motion'] != expected
        again = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
        try:
            again.apply_settings(dialog.settings())
            assert again._direct_beam_request()[2]['motion'] == dialog._direct_beam_request()[2]['motion']
        finally:
            again.close()
    finally:
        dialog.close()


def test_the_real_eiger_del_scan_fits_with_auto_motion_and_no_energy(qapp, tmp_path):
    root = Path(os.environ.get('XDART_TEST_DATA', '/nonexistent')) / 'stitching/GIXRD_Fatigue_Radha'
    if not (root / 'direct_beam_dist_p33m_scan001_master.h5').is_file():
        pytest.skip('direct-beam Eiger scan unavailable')
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    try:
        dialog.beam_source_kind.setCurrentIndex(dialog.beam_source_kind.findData('spec_container'))
        dialog._set_field_text('beam_detector', 'Eiger2_500k')
        dialog._set_field_text('beam_source', root / 'direct_beam_dist_p33m')
        assert dialog._text('rough_energy') == ''
        dialog.beam_button.click()
        settle(dialog, timeout=30)
        assert dialog._distance_result is not None, dialog.status.text()
        assert dialog._distance_result.distance_m == pytest.approx(.335321322566, rel=1e-7)
        assert np.count_nonzero(dialog._distance_result.used) == 51
        assert dialog._text('rough_energy') == ''
    finally:
        dialog.close()
