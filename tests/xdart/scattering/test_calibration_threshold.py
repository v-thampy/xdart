"""Calibration mounts Int's threshold row without changing staged detector data."""
import hashlib
import json

import fabio
import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.calibration.dialog import CalibrationContext, CalibrationDialog
from xdart.gui.widgets.controls_panel import RangeRow
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.integrate.stitch_calibration import capture_calibration_frames, load_calibration_session
from xrd_tools.integrate.stitch_conditioning import StitchConditioning


@pytest.fixture(scope='module')
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def edit(row, bound, value):
    if bound == 'enabled':
        row._toggle[1].setChecked(value)
    else:
        box = row.editors[0 if bound == 'min' else 1]
        box.setText(str(value))
        box.setModified(True)
        box.editingFinished.emit()


def frame(root):
    root.mkdir()
    path = root / 'beam'
    path.write_text('#F beam\n#O0 del nu\n#S 1 ascan del 0 0 0 1\n#P0 0 0\n#N 2\n#L del  nu\n0 0\n')
    image = np.resize(np.array([2, 5, 10, 11], dtype='<i4'), (195, 487))
    image.tofile(root / 'beam_scan1_0000.raw')
    spec = SourceSpec(path, SourceKind.SPEC, options={'scan': '1', 'image_dir': str(root),
        'read_image_kwargs': {'detector_shape': image.shape, 'raw_dtype': 'int32'}})
    return capture_calibration_frames(spec, [0], root / 'staged')[0], image


def test_calibration_threshold_uses_shared_transitions_and_restores_its_session(qapp, tmp_path):
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    again = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    try:
        row = dialog.threshold_range
        assert type(row) is RangeRow
        assert dialog._conditioning().mode == 'off'
        edit(row, 'max', 10)
        edit(row, 'min', 5)
        assert dialog._conditioning().to_dict() == {'mode': 'manual', 'threshold_min': 5., 'threshold_max': 10.}
        edit(row, 'min', 11)  # refusal leaves the last valid band
        assert dialog._conditioning().threshold_min == 5
        edit(row, 'enabled', False)
        again.apply_settings(dialog.settings())
        assert again._conditioning().mode == 'off'
        edit(again.threshold_range, 'enabled', True)
        assert again._conditioning() == StitchConditioning('manual', 5., 10.)
        staged, _ = frame(tmp_path / 'source')
        again.frames = (staged,)
        again.session_dir = tmp_path / 'source' / 'staged'
        again._save_session()
        path = again.session_dir / 'calibration-session.json'
        assert json.loads(path.read_text())['version'] == 6
        _, _, policy = load_calibration_session(path, with_policy=True)
        dialog._apply_fit_policy(policy)
        assert dialog._conditioning() == again._conditioning()
        edit(row, 'max', '')
        assert dialog._conditioning().mode == 'auto'
        assert row.editors[1].text() == '' and 'Auto' in row.editors[1].placeholderText()
        edit(row, 'min', 1)
        assert dialog._conditioning().threshold_max == 1048574  # Pilatus's documented ceiling minus one
    finally:
        dialog.close()
        again.close()


def test_detector_selection_supplies_whole_auto_counts_to_frame_prediction(qapp, tmp_path):
    from xrd_tools.integrate.calibration_seed import rough_arm_record
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path, detector='Eiger2_500k'))
    staged, image = frame(tmp_path / 'source')
    try:
        picker = dialog._fields['beam_detector']
        picker.setCurrentIndex(picker.findData('Pilatus100k'))
        assert dialog._text('beam_detector') == 'Pilatus100k'
        assert 'Auto' in dialog._fields['saturation'].placeholderText()
        assert '1048575' in dialog._fields['saturation'].placeholderText()
        dialog.frames = (staged,)
        dialog.session_dir = tmp_path / 'source' / 'staged'
        edit(dialog.threshold_range, 'enabled', True)
        record = rough_arm_record(distance_m=.05, detector='Pilatus100k', beam_px=(243, 97),
                                  wavelength_m=.7293e-10)
        dialog._prediction = (dialog._calibrated_key(), record)
        dialog._render_frames()  # the actual ring prediction path from the live traceback
        assert dialog.frame_table.item(0, 5).text() not in ('—', 'none')
        assert type(dialog._saturation_value()) is int
        assert dialog._saturation_value() == 1048575
        assert type(dialog._beam_fit_options()['saturation']) is int
        assert dialog._beam_fit_options()['saturation'] == 1048575
        dialog._set_field_text('saturation', '12345.0')
        assert dialog._saturation_value() == 12345
        assert type(dialog._saturation_value()) is int
        for malformed in ('1.5', '0', '-1', 'nan', 'inf'):
            dialog._set_field_text('saturation', malformed)
            with pytest.raises(ValueError, match='positive whole count'):
                dialog._saturation_value()
        np.testing.assert_array_equal(fabio.open(staged.image_path).data, image)
    finally:
        dialog.close()


def test_the_calibration_band_masks_pyfai_and_prediction_without_rewriting_pixels(qapp, tmp_path):
    from xrd_tools.integrate.calibration_seed import rough_arm_record
    from xrd_tools.core.geometry.camera import STITCH_CAMERA
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path, detector='Pilatus100k'))
    dialog.camera.set_value(STITCH_CAMERA)  # frame() stages unchanged native pyFAI pixels.
    staged, image = frame(tmp_path / 'source')
    before = hashlib.sha256(open(staged.image_path, 'rb').read()).hexdigest()
    static = np.zeros(image.shape, bool)
    static[0, 1] = True  # a kept value, independently masked
    np.save(tmp_path / 'mask.npy', static)
    try:
        dialog.frames = (staged,)
        dialog.session_dir = tmp_path / 'source' / 'staged'
        dialog._set_field_text('mask', tmp_path / 'mask.npy')
        edit(dialog.threshold_range, 'max', 10)
        edit(dialog.threshold_range, 'min', 5)
        mask_path = dialog._pyfai_mask(staged)
        np.testing.assert_array_equal(fabio.open(mask_path).data.astype(bool), static | (image < 5) | (image > 10))
        record = rough_arm_record(distance_m=.05, detector='Pilatus100k', beam_px=(243, 97),
                                  wavelength_m=.7293e-10)
        dialog._prediction = (dialog._calibrated_key(), record)
        assert dialog._expected_rings(staged)
        edit(dialog.threshold_range, 'min', 10)
        edit(dialog.threshold_range, 'max', 10.5)
        edit(dialog.threshold_range, 'min', 10.5)  # no pixel survives
        assert dialog._expected_rings(staged) == ()
        assert hashlib.sha256(open(staged.image_path, 'rb').read()).hexdigest() == before
        np.testing.assert_array_equal(fabio.open(staged.image_path).data, image)
        # Calibration threshold is absent from direct-beam options; the shared mask remains there.
        dialog.beam_source_kind.setCurrentIndex(dialog.beam_source_kind.findData('spec'))
        dialog._set_field_text('beam_source', tmp_path / 'source' / 'beam')
        dialog._set_field_text('beam_images', tmp_path / 'source')
        options = dialog._beam_fit_options()
        assert 'conditioning' not in options
        np.testing.assert_array_equal(options['mask'], static)
    finally:
        dialog.close()
