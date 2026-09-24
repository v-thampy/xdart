"""Calibrate dialog: arm model choice, automatic refinement, cancel, summary, save and adoption."""
import json
from pathlib import Path
from threading import Event
import time

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.pages.operation_owner import OperationCancelled
from xdart.gui.pages.services import empty_host_services
from xdart.gui.tabs.stitch.calibration import StitchCalibrationDialog, _run_job
from xdart.gui.tabs.stitch.page import StitchPage
from xrd_tools.core.geometry.stacked_arm import EXTENDED_ARM_MARKER
from xrd_tools.core.scan import SourceKind, SourceSpec
from tests.core.test_stacked_arm_extension import ANGLES, TRUTH, _calibration, _ray, _ring_controls

_ROW_ANGLES = [(d, n) for n in (-10., -3., 4., 11.) for d in (6., 20., 34.)]


class _Status:
    def show(self, text, timeout_ms=0):
        pass


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _wait(dialog, timeout=60):
    until = time.monotonic() + timeout
    while dialog.active() and time.monotonic() < until:
        QtWidgets.QApplication.processEvents()
        time.sleep(.005)
    for _ in range(5):  # the 50 ms poll delivers the terminal
        QtWidgets.QApplication.processEvents()
        time.sleep(.06)
    QtWidgets.QApplication.processEvents()
    assert not dialog.active(), dialog.status.text()


def _five_parameter_seed(angles):
    """The guided five-parameter fit of exact controls: the usual refinement seed."""
    from xrd_tools.core.containers import PONI
    from xrd_tools.integrate.refine import refine_stacked_arm
    controls, calibrant = _ring_controls(TRUTH, _calibration(), angles)
    seed = PONI(dist=.392, poni1=.0353, poni2=.0447, wavelength=.7293e-10, detector='Pilatus300kw')
    five = refine_stacked_arm(seed, controls, nu_motor='nu', del_motor='del', calibrant=calibrant,
                              detector_config={'orientation': 3}, extended=False)
    record = five.diffractometer.stacked_arm.to_pyfai_goniometer(five.diffractometer.calibration)
    record['detector_config'] = {'orientation': 3}
    return record


def _ring_scan(root):
    """A SPEC scan of LaB6 ring images from the independent extended-arm ray."""
    from pyFAI.calibrant import get_calibrant
    calibrant = get_calibrant('LaB6')
    calibrant.wavelength = .7293e-10
    q_rings = 2 * np.pi / np.asarray(calibrant.dSpacing)[:25]
    rows, cols = np.indices((195, 1475), dtype=float)
    images = root / 'images'
    images.mkdir(parents=True)
    for index, (delta, nu) in enumerate(_ROW_ANGLES):
        q, _ = _ray(TRUTH, _calibration(), delta, nu, rows, cols)
        image = 10 + 5000 * np.exp(-((q[..., None] - q_rings) ** 2) / (2 * .008 ** 2)).sum(-1)
        image.astype('<i4').tofile(images / f'ring_scan1_{index:04d}.raw')
    lines = '\n'.join(f'{d} {n}' for d, n in _ROW_ANGLES)
    (root / 'ring').write_text('#F ring\n#E 1\n#D today\n#O0 del nu\n#S 1 ascan del 6 34 11 1\n'
                               f'#D today\n#P0 0 0\n#N 2\n#L del  nu\n{lines}\n')
    return SourceSpec(root / 'ring', SourceKind.SPEC, options={
        'scan': '1', 'image_dir': str(images), 'image_stem': 'ring_scan1_',
        'read_image_kwargs': {'detector_shape': (195, 1475), 'raw_dtype': 'int32'}})


def _page_with_dialog(tmp_path, spec=None, seed_path=None):
    page = StitchPage(empty_host_services(_Status()))
    page.set_field("project_root", str(tmp_path))
    if spec is not None:
        page.set_sources((str(spec.uri),))
        for key, value in (("scan", "1"), ("image_dir", spec.options['image_dir']),
                           ("image_stem", "ring_scan1_"), ("detector_rows", "195"),
                           ("detector_cols", "1475"), ("raw_dtype", "int32")):
            page.set_field(key, value)
    if seed_path is not None:
        page.set_field("geometry_kind", "Goniometer JSON")
        page.set_field("geometry_path", str(seed_path))
    page._calibrate()
    return page, page._calibration_dialog


def _guided_frames(tmp_path):
    from tests.core.test_stitch_guided_calibration import _frames
    controls, _calibrant = _ring_controls(TRUTH, _calibration(), ANGLES[::2])
    return _frames(tmp_path / 'frames', controls, _calibration().poni)


def _save_to(monkeypatch, target):
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName',
                        staticmethod(lambda *args, **kwargs: (str(target), '')))


def test_guided_extended_fit_saves_a_v2_arm_that_the_page_adopts(qapp, tmp_path, monkeypatch):
    (tmp_path / 'frames').mkdir()
    frames = _guided_frames(tmp_path)
    page, dialog = _page_with_dialog(tmp_path)
    try:
        assert not dialog.arm_model.isEnabled()  # no frames yet
        dialog.frames = frames
        dialog._frame_motors = ('del', 'nu')
        dialog._render_frames()
        assert dialog.arm_model.isEnabled()
        dialog.arm_model.setCurrentIndex(dialog.arm_model.findData('stacked_extended'))
        dialog.fit_button.click()
        _wait(dialog)
        assert dialog.fit is not None, dialog.status.text()
        assert dialog.fit.asset['provenance']['arm_model'] == 'stacked_extended'
        summary = dialog.fit_summary.toPlainText()
        assert 'Parameters (± 1σ):' in summary and 'mount_tilt' in summary and 'sample_offset3' in summary
        dialog.arm_model.setCurrentIndex(0)  # a changed choice invalidates the fit and its summary
        assert dialog.fit is None and not dialog.fit_summary.toPlainText()
        assert not dialog.save_button.isEnabled()
        dialog.arm_model.setCurrentIndex(1)
        dialog.fit_button.click()
        _wait(dialog)
        target = (tmp_path / 'calibration' / 'extended.json').resolve()
        _save_to(monkeypatch, target)
        dialog.save_button.click()
        assert Path(page._value("geometry_path")) == target
        assert page._value("geometry_kind") == "Goniometer JSON"
        record = json.loads(target.read_text())
        assert EXTENDED_ARM_MARKER in record['trans_function']['constants']
        from xrd_tools.analysis.stitch_operation import (
            StitchGeometryInput, StitchGeometryKind, capture_stitch_geometry,
        )
        from xrd_tools.core.geometry import Diffractometer
        receipt = capture_stitch_geometry(StitchGeometryInput(
            target, StitchGeometryKind.PYFAI_GONIOMETER_JSON,
            source_motors=(('del_angle', 'del'), ('nu_angle', 'nu'))))
        arm = Diffractometer.from_pyfai_goniometer(
            json.loads(receipt.content), source_motors={'del_angle': 'del', 'nu_angle': 'nu'}).stacked_arm
        saved = dict(zip(record['param_names'], record['param']))
        assert arm.extended and arm.mount_tilt == saved['mount_tilt'] != 0
        assert abs(arm.mount_tilt - TRUTH['mount_tilt']) < 2e-3  # NPT files round the points
    finally:
        dialog.close()
        page.shutdown()


def test_five_parameter_guided_save_equals_the_core_save(qapp, tmp_path, monkeypatch):
    from xrd_tools.integrate.stitch_calibration import fit_calibration, save_calibration_fit
    (tmp_path / 'frames').mkdir()
    frames = _guided_frames(tmp_path)
    dialog = StitchCalibrationDialog(tmp_path)
    try:
        dialog.frames = frames
        dialog._frame_motors = ('del', 'nu')
        dialog._render_frames()
        assert dialog.arm_model.currentData() == 'stacked'
        dialog.fit_button.click()
        _wait(dialog)
        assert dialog.fit is not None, dialog.status.text()
        target = (tmp_path / 'calibration' / 'dialog.json').resolve()
        _save_to(monkeypatch, target)
        dialog.save_button.click()
        direct = Path(save_calibration_fit(fit_calibration(frames), tmp_path / 'calibration' / 'core.json'))
        assert target.read_bytes() == direct.read_bytes()
        assert (Path(f'{target}.provenance.json').read_bytes()
                == Path(f'{direct}.provenance.json').read_bytes())
    finally:
        dialog.close()


def test_automatic_refinement_from_the_pages_geometry_saves_and_is_adopted(qapp, tmp_path, monkeypatch):
    spec = _ring_scan(tmp_path / 'scan')
    seed_path = tmp_path / 'calibration' / 'five.json'
    seed_path.parent.mkdir()
    seed_path.write_text(json.dumps(_five_parameter_seed(_ROW_ANGLES)))
    page, dialog = _page_with_dialog(tmp_path, spec, seed_path)
    try:
        assert dialog.refine_seed.currentData() == 'file'
        assert dialog._text('refine_seed_path') == str(seed_path)
        dialog.refine_button.click()  # the calibrant confirmation is required
        assert 'calibrant scan' in dialog.status.text() and not dialog.active()
        dialog.refine_confirm.setChecked(True)
        dialog._fields['refine_frames'].setText('12')
        dialog.refine_button.click()
        _wait(dialog)
        assert dialog.fit is not None, dialog.status.text()
        automatic = dialog.fit.asset['provenance']['automatic_refinement']
        assert automatic['calibrant'] == 'LaB6' and automatic['validation']['chosen'] == 'stacked_extended'
        summary = dialog.fit_summary.toPlainText()
        assert 'Automatic refinement chose the detector mount and sample offset' in summary
        assert 'Held-out σ: five-parameter' in summary and 'nu rows used: 4 of 4' in summary
        target = (tmp_path / 'calibration' / 'refined.json').resolve()
        _save_to(monkeypatch, target)
        dialog.save_button.click()
        assert Path(page._value("geometry_path")) == target
        saved = json.loads(Path(f'{target}.provenance.json').read_text())
        assert saved['automatic_refinement']['validation']['chosen'] == 'stacked_extended'
    finally:
        dialog.close()
        page.shutdown()


def test_cancelled_refinement_ends_cancelled_and_a_refusal_saves_nothing(qapp, tmp_path):
    spec = _ring_scan(tmp_path / 'scan')
    seed = _five_parameter_seed(_ROW_ANGLES)
    cancelled = Event()
    cancelled.set()
    with pytest.raises(OperationCancelled):
        _run_job(('refine', (spec, seed, {'max_frames': 12})), cancelled, lambda *_: True)
    seen = []
    fit = _run_job(('refine', (spec, seed, {'max_frames': 12})), Event(),
                   lambda stage, done, total: seen.append((stage, done, total)) or True)
    assert fit.asset['provenance']['automatic_refinement']
    assert [done for _stage, done, _total in seen] == sorted(done for _stage, done, _total in seen)
    assert {total for _stage, _done, total in seen} == {1000}
    assert {stage.rsplit(' ', 1)[0] for stage, _done, _total in seen} == {
        'Reading frames', 'Finding ring points', 'Validating models', 'Fitting'}

    seed_path = tmp_path / 'five.json'
    seed_path.write_text(json.dumps(seed))
    dialog = StitchCalibrationDialog(tmp_path)
    try:
        dialog.set_source_context(spec, geometry_path=seed_path)
        dialog.refine_confirm.setChecked(True)
        dialog._fields['refine_frames'].setText('2')
        dialog.refine_button.click()
        _wait(dialog)
        assert 'needs 3-256 frames' in dialog.status.text()
        assert dialog.fit is None and not dialog.save_button.isEnabled()
    finally:
        dialog.close()


def test_the_model_choice_is_disabled_with_its_reason(qapp, tmp_path):
    (tmp_path / 'frames').mkdir()
    frames = _guided_frames(tmp_path)
    dialog = StitchCalibrationDialog(tmp_path, backend='xu_hist')
    try:
        dialog.frames = frames
        dialog._frame_motors = ('del', 'nu')
        dialog._render_frames()
        assert not dialog.arm_model.isEnabled() and 'xrayutilities' in dialog.arm_model.toolTip()
        dialog.backend.setCurrentIndex(dialog.backend.findData('multigeometry'))
        assert dialog.arm_model.isEnabled()
        dialog.arm_model.setCurrentIndex(1)
        dialog.scales.setChecked(True)
        assert not dialog.arm_model.isEnabled() and dialog.arm_model.currentData() == 'stacked'
        assert 'motor scales' in dialog.arm_model.toolTip()
        dialog.scales.setChecked(False)
        dialog.frames = tuple(frame for frame in frames if frame.motors['nu'] in (-10., -3.))
        dialog._render_frames()
        assert not dialog.arm_model.isEnabled() and '3 nu positions' in dialog.arm_model.toolTip()
    finally:
        dialog.close()


def test_a_cancelled_guided_fit_ends_cancelled_not_failed(tmp_path):
    (tmp_path / 'frames').mkdir()
    frames = _guided_frames(tmp_path)
    cancelled = Event()
    cancelled.set()
    with pytest.raises(OperationCancelled):
        _run_job(('fit', (frames, {'arm_model': 'stacked_extended'})), cancelled, lambda *_: True)
