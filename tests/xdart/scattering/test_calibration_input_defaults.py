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
from xrd_tools.core.geometry.camera import DetectorCamera, stitch_camera


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


def test_calibrant_spec_container_lists_and_stages_padded_images(qapp, tmp_path):
    from tests.xdart.scattering.test_stitch_guided_calibration import _beam_container
    from xrd_tools.sources import SourceKind
    nexus, _, _ = _beam_container(tmp_path)
    nexus.rename(tmp_path / 'beam_scan001_master.h5')
    spec = tmp_path / 'beam'
    spec.write_text('#F beam\n#O0 nu  energy\n#S 1 ascan del 1 -1 4 1\n#P0 4 7000.6716669638\n'
                    '#N 1\n#L del\n1\n.5\n0\n-.5\n-1\n')
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    dialog.show()
    try:
        index = dialog.source_kind.findData('spec_container')
        assert index >= 0
        dialog.source_kind.setCurrentIndex(index)
        commit(dialog, 'source', spec)
        assert dialog._text('rough_energy') == ''
        assert 'Auto · 7.0007 keV from SPEC' in dialog._fields['rough_energy'].placeholderText()
        assert dialog._fields['rough_energy'].maximumWidth() == 300
        assert dialog._row_visible('images') and dialog._row_visible('entry')
        assert 'beam_scan001_' in dialog._fields['stem'].placeholderText()
        assert dialog._spec().kind is SourceKind.SPEC
        assert dialog.frames == ()  # Refine reads the scan without staged manual-calibration frames.
        assert dialog._calibrant_source_ready()
        assert '5 images' in dialog.active_source_row.editor.text()
        dialog.list_button.click()
        settle(dialog)
        assert len(dialog._choices) == 5, dialog.status.text()
        dialog.prepare_button.click()
        settle(dialog)
        assert len(dialog.frames) == 5, dialog.status.text()
        assert all(Path(frame.image_path).is_file() for frame in dialog.frames)
        again = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
        try:
            again.apply_settings(dialog.settings())
            assert again.source_kind.currentData() == 'spec_container'
            assert again._spec() == dialog._spec()
        finally:
            again.close()
    finally:
        dialog.close()


def test_calibrant_metadata_only_spec_is_not_ready_for_refinement(qapp, tmp_path):
    spec = tmp_path / 'beam'
    spec.write_text('#F beam\n#O0 nu  energy\n#S 1 ascan del 1 -1 2 1\n#P0 0 11000\n'
                    '#N 1\n#L del\n1\n0\n-1\n')
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    try:
        dialog.source_kind.setCurrentIndex(dialog.source_kind.findData('spec_container'))
        commit(dialog, 'source', spec)
        assert '11.0000 keV from SPEC' in dialog._fields['rough_energy'].placeholderText()
        assert not dialog._calibrant_source_ready()
        assert '0 images' in dialog.active_source_row.editor.text()
        assert 'folder' in dialog.active_source_row.editor.text().lower()
        dialog.refine_button.click()
        assert not dialog.active()
        assert dialog.fit is None
        assert 'calibrant source' in dialog.status.text().lower()
    finally:
        dialog.close()


def test_rough_rows_share_existing_controls_and_keep_mode_visibility(qapp, tmp_path):
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    try:
        assert dialog._parts['beam_entry'][0] == 'beam_source_kind'
        assert dialog._parts['rough_mount'][0] == 'beam_pixel'
        assert 'beam_motor' not in dialog._fields and not hasattr(dialog, 'motion')
        assert dialog._label('beam_pixel').text() == 'Direct beam pixel (x, y)'
        assert dialog.beam_readiness.title.isHidden()
        assert dialog.beam_readiness.objectName() == 'controlsInlineReadiness'
        assert dialog.tabs.cornerWidget().text() == '?'
        for kind in ('spec_series', 'spec_container', 'nexus'):
            dialog.beam_source_kind.setCurrentIndex(dialog.beam_source_kind.findData(kind))
            assert dialog._row_visible('beam_entry') == (kind != 'spec_series')
        dialog.rough_mode.setCurrentIndex(dialog.rough_mode.findData('manual'))
        assert dialog._row_visible('beam_pixel') and dialog._row_visible('rough_mount')
        assert not dialog._row_visible('beam_entry') and dialog.beam_readiness.isHidden()
        dialog.rough_mode.setCurrentIndex(dialog.rough_mode.findData('skip'))
        assert not dialog._row_visible('beam_pixel') and not dialog._row_visible('rough_mount')
        dialog.rough_mode.setCurrentIndex(dialog.rough_mode.findData('scan'))
        assert dialog._row_visible('beam_entry') and not dialog.beam_readiness.isHidden()
    finally:
        dialog.close()


def test_calibrant_nexus_context_still_selects_the_nexus_choice(qapp, tmp_path):
    from xrd_tools.sources import SourceKind, SourceSpec
    dialog = CalibrationDialog(CalibrationContext('RSM', tmp_path))
    try:
        dialog.set_source_context(SourceSpec(tmp_path / 'scan.nxs', SourceKind.NEXUS_STACK))
        assert dialog.source_kind.currentData() == 'nexus_stack'
        assert not dialog._row_visible('images') and dialog._row_visible('entry')
    finally:
        dialog.close()


def test_detector_picker_describes_raw_defaults_and_manual_header_reads_pixels(qapp, tmp_path):
    from xrd_tools.core.scan import SourceKind, SourceSpec
    from xrd_tools.integrate.stitch_calibration import capture_calibration_frames
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path, detector='Eiger2_500k'))
    again = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    try:
        picker = dialog._fields['beam_detector']
        assert dialog._text('beam_detector') == 'Eiger2_500k'
        assert '512 × 1028' in picker.currentText() and 'RAW int32' in picker.currentText()
        assert not dialog._row_visible('raw')
        assert dialog._fields['shape'].isReadOnly() and dialog._fields['dtype'].isReadOnly()
        assert dialog._parts['header'][0] == 'beam_detector'
        assert dialog.camera.parent().layout().labelForField(dialog.camera).text() == 'Camera orientation (lab frame)'
        spec = tmp_path / 'small'
        spec.write_text('#F small\n#O0 nu\n#S 1 ascan del 0 0 0 1\n#P0 0\n#N 1\n#L del\n0\n')
        image = np.arange(24, dtype='<u2').reshape(4, 6)
        (tmp_path / 'small_scan1_0000.raw').write_bytes(b'header16bytes!!!' + image.tobytes())
        source = SourceSpec(spec, SourceKind.SPEC, options={'scan': '1', 'image_dir': str(tmp_path),
            'read_image_kwargs': {'detector_shape': image.shape, 'raw_dtype': 'uint16', 'raw_header_skip': 16}})
        dialog.set_source_context(source)
        assert dialog.raw_layout.currentData() == 'manual' and dialog._row_visible('raw')
        assert not dialog._fields['shape'].isReadOnly() and not dialog._fields['dtype'].isReadOnly()
        staged = capture_calibration_frames(dialog._spec(), [0], tmp_path / 'staged')
        import fabio
        np.testing.assert_array_equal(fabio.open(staged[0].image_path).data, image)
        again.apply_settings(dialog.settings())
        assert again.raw_layout.currentData() == 'manual' and again._reader() == dialog._reader()
        again.raw_layout.setCurrentIndex(again.raw_layout.findData('factory'))
        assert again._text('shape') == '512, 1028' and again._text('dtype') == 'int32'
        assert again._text('header') == '16'  # byte offset is independent of the detector model
    finally:
        dialog.close()
        again.close()


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


def test_camera_names_the_diffractometers_lab_frame(qapp, tmp_path):
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    try:
        label = dialog.camera.parentWidget().layout().labelForField(dialog.camera)
        assert 'lab frame' in label.text().lower()
        assert 'Diffractometer' in dialog.camera.rotation.toolTip()
        assert 'x up' in dialog.camera.rotation.toolTip()
        assert 'z to the left' in dialog.camera.fliplr.toolTip()
        assert 'pyFAI\'s frame, rows x+, columns z−' in dialog.camera.rotation.toolTip()
        assert 'does not determine' in dialog.camera.rotation.toolTip()
    finally:
        dialog.close()


@pytest.mark.parametrize('page', ['Stitch', 'RSM'])
def test_detector_header_tracks_the_model_and_camera_image_correction(qapp, tmp_path, page):
    dialog = CalibrationDialog(CalibrationContext(page, tmp_path))
    try:
        dialog._set_field_text('beam_detector', 'Eiger2_500k')
        assert dialog._detector_panel.title() == 'Detector · Eiger2 500k · 1028 × 512 px (W × H) · orientation: upright'
        dialog.camera.set_value(DetectorCamera('z+', 'x-'))
        assert '512 × 1028 px (W × H)' in dialog._detector_panel.title()
        assert dialog._detector_panel.title().endswith('orientation: 90° CW, horizontal flip')
        assert dialog._image_size.text() == '512 × 1028 px (W × H)'
        title = dialog._detector_panel.title()
        dialog._set_field_text('beam_orientation', '1')
        assert dialog._detector_panel.title() == title
        dialog._set_field_text('beam_detector', 'Pilatus100k')
        assert dialog._detector_panel.title().startswith('Detector · Pilatus 100k ·')
        dialog.camera.default_button.click()
        assert dialog._detector_panel.title().endswith('orientation: upright')
        assert dialog._text('beam_orientation') == '1'
    finally:
        dialog.close()


def test_saturation_shows_the_documented_auto_value_and_keeps_manual_overrides(qapp, tmp_path):
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    try:
        dialog._set_field_text('beam_source', tmp_path / 'beam')
        field = dialog._fields['saturation']
        assert dialog._text('saturation') == ''  # the hint is Auto, as in Int's threshold
        assert '1048575' in field.placeholderText()
        assert dialog._beam_fit_options()['saturation'] == 1048575
        dialog._set_field_text('saturation', '12345')
        dialog._set_field_text('beam_detector', 'Eiger2_500k')
        assert field.text() == '12345'
        assert dialog._beam_fit_options()['saturation'] == 12345
        dialog._set_field_text('saturation', '')
        assert 'acquisition' in field.placeholderText()
        assert 'saturation' not in dialog._beam_fit_options()
        again = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
        try:
            again.apply_settings(dialog.settings())
            assert again._text('saturation') == ''
            assert again._fields['saturation'].placeholderText() == field.placeholderText()
        finally:
            again.close()
    finally:
        dialog.close()


@pytest.mark.parametrize('rotation,flip', [(r, f) for r in (0, 90, 180, 270) for f in (False, True)])
@pytest.mark.parametrize('page', ['Stitch', 'RSM'])
def test_typed_beam_pixel_follows_the_camera_but_not_pyfais_origin(qapp, tmp_path, rotation, flip, page):
    dialog = CalibrationDialog(CalibrationContext(page, tmp_path))
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
        dialog.camera.default_button.click()
        assert tuple(map(float, dialog._text('beam_pixel').split(','))) == (280, 97)
        assert dialog._orientation() == ImageOrientation()
        assert dialog._text('beam_orientation') == '1'
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
        assert dialog._text('rough_energy') == f'{energy:.4f}'
        assert dialog._rough_wavelength_A() == pytest.approx(.7293, rel=5e-6)
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
        assert dialog._text('rough_energy') == ''
        assert dialog._source_energy_eV == 11270
        assert dialog._rough_wavelength_A() == pytest.approx(float(energy_eV_to_wavelength_m(11270)) * 1e10)
        commit(dialog, 'scan', '2')
        assert dialog._text('rough_energy') == ''
        assert dialog._source_energy_eV == 16000
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
        assert dialog._text('rough_energy') == ''
        assert dialog._source_energy_eV == pytest.approx(float(wavelength_m_to_energy_eV(1.1e-10)))
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
        from tests.xdart.scattering.test_direct_beam_readiness import detect
        detect(dialog)
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
        from tests.xdart.scattering.test_direct_beam_readiness import detect
        detect(dialog)
        dialog.beam_button.click()
        settle(dialog, timeout=30)
        assert dialog._distance_result is not None, dialog.status.text()
        assert dialog._distance_result.distance_m == pytest.approx(.335321322566, rel=1e-7)
        assert np.count_nonzero(dialog._distance_result.used) == 51
        assert dialog._text('rough_energy') == ''
    finally:
        dialog.close()
