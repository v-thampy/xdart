"""Calibrate dialog: automatic arm model, automatic refinement, cancel, summary, saving both files and adoption."""
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
from tests.core.test_stacked_arm_extension import ANGLES, TRUTH, _calibration, _ring_controls
from tests.core.test_stitch_autocalibration import (
    ROW_ANGLES as _ROW_ANGLES, five_parameter_seed as _five_parameter_seed, ring_scan as _ring_scan,
)

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


def _single_arm_frames(tmp_path):
    """Guided frames of twoc's 2θ arm: psic's arm at nu = 0, read from the 2θ motor alone."""
    from dataclasses import replace
    from tests.core.test_stitch_guided_calibration import _frames
    controls, _calibrant = _ring_controls({'nu_offset': .02, 'del_offset': -.05}, _calibration(),
                                          [(delta, 0.) for delta in (6., 18., 30., 42.)])
    controls = [replace(control, motors={'tth': control.motors['del']}) for control in controls]
    return _frames(tmp_path / 'frames', controls, _calibration().poni)


def _horizontal_frames(tmp_path):
    """Guided frames of a horizontal twoc arm: psic's arm at del = 0, read from the 2θ motor alone."""
    from dataclasses import replace
    from tests.core.test_stitch_guided_calibration import _frames
    controls, _calibrant = _ring_controls({'nu_offset': -.02, 'del_offset': .03}, _calibration(),
                                          [(0., nu) for nu in (6., 18., 30., 42.)])
    controls = [replace(control, motors={'tth': control.motors['nu']}) for control in controls]
    return _frames(tmp_path / 'frames', controls, _calibration().poni)


def _save_to(monkeypatch, target):
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName',
                        staticmethod(lambda *args, **kwargs: (str(target), '')))


def test_the_guided_fit_takes_the_extension_and_saves_both_files(qapp, tmp_path, monkeypatch):
    """Decision 40 and always-both: the extension whenever the frames determine it (no model
    choice), and Save writes the pyFAI file and its xrayutilities partner before adopting."""
    (tmp_path / 'frames').mkdir()
    frames = _guided_frames(tmp_path)
    page, dialog = _page_with_dialog(tmp_path)
    try:
        assert 'five-parameter arm; the extension needs frames on at least 3 nu positions' in dialog.arm_note.text()
        dialog.frames = frames
        dialog._frame_motors = ('del', 'nu')
        dialog._render_frames()
        assert 'the detector mount and sample offset, which the frames on 5 nu positions' in dialog.arm_note.text()
        dialog.fit_button.click()
        _wait(dialog)
        assert dialog.fit is not None, dialog.status.text()
        assert dialog.fit.asset['provenance']['arm_model'] == 'stacked_extended'
        summary = dialog.fit_summary.toPlainText()
        assert 'Parameters (± 1σ):' in summary and 'mount_tilt' in summary and 'sample_offset3' in summary
        dialog.scales.setChecked(True)  # a changed option invalidates the fit and says what it changes
        assert dialog.fit is None and not dialog.fit_summary.toPlainText()
        assert not dialog.save_button.isEnabled()
        assert ('five-parameter arm; motor scales are fitted' in dialog.arm_note.text()
                and 'Save writes no xrayutilities file' in dialog.arm_note.text())
        dialog.scales.setChecked(False)
        dialog.fit_button.click()
        _wait(dialog)
        target = (tmp_path / 'calibration' / 'extended.json').resolve()
        _save_to(monkeypatch, target)
        adopted = []
        dialog.calibrationSaved.connect(lambda path, backend: adopted.append((path, backend)))
        dialog.save_button.click()
        _wait(dialog)
        assert adopted == [(str(target), 'multigeometry')]  # once, with both files written
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
        partner = json.loads((tmp_path / 'calibration' / 'extended.xu.json').read_text())
        assert partner['version'] == 3  # its mount is the arm's own tilt
        assert partner['detector']['tilt_deg'] == pytest.approx(abs(np.degrees(arm.mount_tilt)), abs=1e-6)
        assert 'extended.xu.json' in dialog.status.text() and 'extended.xu.json' in dialog.fit_summary.toPlainText()
    finally:
        dialog.close()
        page.shutdown()


def test_a_twoc_arm_is_fitted_saved_and_adopted_with_nu_held_at_zero(qapp, tmp_path, monkeypatch):
    from xrd_tools.core.geometry.stacked_arm import is_single_arm
    from xrd_tools.session.instrument import InstrumentGeometry
    (tmp_path / 'frames').mkdir()
    frames = _single_arm_frames(tmp_path)
    page, dialog = _page_with_dialog(tmp_path)
    try:
        dialog.set_instrument(InstrumentGeometry('twoc'))
        dialog.frames, dialog._frame_motors = frames, ('tth', None)
        dialog._render_frames()
        assert 'five-parameter arm; a single 2θ arm holds one circle at zero' in dialog.arm_note.text()
        dialog.fit_button.click()
        _wait(dialog)
        assert dialog.fit is not None, dialog.status.text()
        assert dialog.fit.message == ('Converged; one detector arm, nu held at zero; motor scales fixed at degrees; '
                                      'five-parameter arm: a single 2θ arm holds one circle at zero; the extension '
                                      'needs both')
        provenance, record = dialog.fit.asset['provenance'], dialog.fit.asset['calibration']
        assert provenance['source_motors'] == {'del_angle': 'tth'}
        assert provenance['instrument_geometry'] == InstrumentGeometry('twoc').to_record()
        assert is_single_arm(record) and record['pos_names'] == ['del_angle']
        fitted = dict(zip(record['param_names'], record['param']))
        assert fitted['nu_offset'] == pytest.approx(.02, abs=2e-3)  # the arm's yaw
        assert fitted['del_offset'] == pytest.approx(-.05, abs=2e-3)
        target = (tmp_path / 'calibration' / 'twoc.json').resolve()
        _save_to(monkeypatch, target)
        dialog.save_button.click()
        _wait(dialog)
        assert Path(page._value("geometry_path")) == target
        assert page._instrument() == InstrumentGeometry('twoc')
        assert page._value("source_motors") == "del_angle=tth"
        assert page._arm_problem() is None
        partner = json.loads((tmp_path / 'calibration' / 'twoc.xu.json').read_text())
        assert partner['acquisition']['source_motors'] == {'del': 'tth', 'nu': None}
    finally:
        dialog.close()
        page.shutdown()


def test_a_horizontal_twoc_arm_is_fitted_saved_and_adopted_with_del_held_at_zero(qapp, tmp_path, monkeypatch):
    from xrd_tools.session.instrument import InstrumentGeometry
    (tmp_path / 'frames').mkdir()
    frames = _horizontal_frames(tmp_path)
    page, dialog = _page_with_dialog(tmp_path)
    try:
        dialog.set_instrument(InstrumentGeometry('twoc', plane='horizontal'))
        dialog.frames, dialog._frame_motors = frames, (None, 'tth')
        dialog._render_frames()
        dialog.fit_button.click()
        _wait(dialog)
        assert dialog.fit is not None, dialog.status.text()
        assert dialog.fit.message.startswith('Converged; one detector arm, del held at zero; motor scales fixed at degrees')
        provenance, record = dialog.fit.asset['provenance'], dialog.fit.asset['calibration']
        assert provenance['source_motors'] == {'nu_angle': 'tth'} and record['pos_names'] == ['nu_angle']
        assert provenance['instrument_geometry'] == InstrumentGeometry('twoc', plane='horizontal').to_record()
        fitted = dict(zip(record['param_names'], record['param']))
        assert fitted['nu_offset'] == pytest.approx(-.02, abs=2e-3)
        assert fitted['del_offset'] == pytest.approx(.03, abs=2e-3)  # the detector's fixed pitch
        target = (tmp_path / 'calibration' / 'twoc-horizontal.json').resolve()
        _save_to(monkeypatch, target)
        dialog.save_button.click()
        _wait(dialog)
        assert page._instrument() == InstrumentGeometry('twoc', plane='horizontal')
        assert page._value("source_motors") == "nu_angle=tth"
        assert page._arm_problem() is None
        page._set_instrument(InstrumentGeometry('twoc'))  # the vertical plane cannot take it
        assert 'Choose the horizontal scattering plane' in page._arm_problem()
    finally:
        dialog.close()
        page.shutdown()


def test_a_twoc_arms_xrayutilities_file_covers_the_typed_range_with_nu_at_zero(qapp, tmp_path, monkeypatch):
    """Was: the window's xrayutilities fit of a twoc arm. Now every save writes the partner, checked
    over the typed range; the held circle stays at zero."""
    from xrd_tools.session.instrument import InstrumentGeometry
    (tmp_path / 'frames').mkdir()
    frames = _single_arm_frames(tmp_path)
    page, dialog = _page_with_dialog(tmp_path)
    try:
        dialog.set_instrument(InstrumentGeometry('twoc'))
        dialog.frames, dialog._frame_motors = frames, ('tth', None)
        dialog._render_frames()
        dialog._fields['del_domain'].setText('0, 50')  # the typed range covers 2θ; nu stays at zero
        dialog.fit_button.click()
        _wait(dialog)
        target = (tmp_path / 'calibration' / 'twoc.json').resolve()
        _save_to(monkeypatch, target)
        dialog.save_button.click()
        _wait(dialog)
        asset = json.loads((tmp_path / 'calibration' / 'twoc.xu.json').read_text())
        assert asset['version'] == 3 and asset['acquisition']['source_motors'] == {'del': 'tth', 'nu': None}
        assert asset['validation']['validated_scan_domain_source_deg'] == {'del': [0., 50.], 'nu': [0., 0.]}
    finally:
        dialog.close()
        page.shutdown()


def test_a_horizontal_twoc_arms_xrayutilities_file_covers_the_typed_range_with_del_at_zero(qapp, tmp_path, monkeypatch):
    from xrd_tools.session.instrument import InstrumentGeometry
    (tmp_path / 'frames').mkdir()
    frames = _horizontal_frames(tmp_path)
    page, dialog = _page_with_dialog(tmp_path)
    try:
        dialog.set_instrument(InstrumentGeometry('twoc', plane='horizontal'))
        dialog.frames, dialog._frame_motors = frames, (None, 'tth')
        dialog._render_frames()
        dialog._fields['nu_domain'].setText('0, 50')  # the typed range covers 2θ; del stays at zero
        dialog.fit_button.click()
        _wait(dialog)
        target = (tmp_path / 'calibration' / 'twoc-horizontal.json').resolve()
        _save_to(monkeypatch, target)
        dialog.save_button.click()
        _wait(dialog)
        asset = json.loads((tmp_path / 'calibration' / 'twoc-horizontal.xu.json').read_text())
        assert asset['acquisition']['source_motors'] == {'del': None, 'nu': 'tth'}
        assert asset['validation']['validated_scan_domain_source_deg'] == {'del': [0., 0.], 'nu': [0., 50.]}
    finally:
        dialog.close()
        page.shutdown()


def test_a_fit_belongs_to_the_instrument_its_frames_were_staged_with(qapp, tmp_path, monkeypatch):
    # Codex round 11 F1: motors renamed after staging, and the geometry changed after a fit
    from xrd_tools.session.instrument import InstrumentGeometry
    (tmp_path / 'frames').mkdir()
    frames = _guided_frames(tmp_path)
    page, dialog = _page_with_dialog(tmp_path)
    try:
        dialog.frames, dialog._frame_motors = frames, ('del', 'nu')
        dialog._render_frames()
        dialog.set_instrument(InstrumentGeometry('psic', (('del', 'delta'), ('nu', 'gamma'))))
        dialog.fit_button.click()
        assert dialog.fit is None and not dialog.active()
        assert 'staged reading del and nu; the Geometry now names delta and gamma' in dialog.status.text()
        dialog.set_instrument(InstrumentGeometry('psic'))
        dialog.fit_button.click()
        _wait(dialog)
        assert dialog.fit is not None and dialog.save_button.isEnabled(), dialog.status.text()
        dialog.set_instrument(InstrumentGeometry('psic'))  # the same instrument keeps the fit
        assert dialog.fit is not None
        dialog.set_instrument(InstrumentGeometry('fourc'))  # another one leaves it unsaveable
        assert dialog.fit is None and not dialog.save_button.isEnabled()
        dialog.set_instrument(InstrumentGeometry('psic', (('eta', 'halpha'),)))  # any circle counts
        dialog.fit_button.click()
        _wait(dialog)
        dialog.geometry_editor.motor_edit('chi').setText('chi2')
        assert dialog.fit is None
        dialog.fit_button.click()
        _wait(dialog)
        fitted = dialog.fit.asset['provenance']['instrument_geometry']
        assert fitted == dialog.geometry_editor.value().to_record()
        target = (tmp_path / 'calibration' / 'psic.json').resolve()
        _save_to(monkeypatch, target)
        dialog.save_button.click()
        _wait(dialog)
        # adoption keeps the fitted instrument and its motors
        assert page._instrument().to_record() == fitted
        assert page._value("source_motors") == "del_angle=del, nu_angle=nu"
    finally:
        dialog.close()
        page.shutdown()


def test_the_guided_save_equals_the_core_save(qapp, tmp_path, monkeypatch):
    from xrd_tools.integrate.stitch_calibration import (
        fit_calibration, pair_validation_angles, save_calibration_pair,
    )
    (tmp_path / 'frames').mkdir()
    frames = _guided_frames(tmp_path)
    dialog = StitchCalibrationDialog(tmp_path)
    try:
        dialog.frames = frames
        dialog._frame_motors = ('del', 'nu')
        dialog._render_frames()
        dialog.fit_button.click()
        _wait(dialog)
        assert dialog.fit is not None, dialog.status.text()
        target = (tmp_path / 'calibration' / 'dialog.json').resolve()
        _save_to(monkeypatch, target)
        dialog.save_button.click()
        _wait(dialog)
        geometry = dialog.geometry_editor.value().to_record()  # the window records its diffractometer
        fit = fit_calibration(frames, instrument_geometry=geometry, arm_model='auto')
        direct = save_calibration_pair(fit, tmp_path / 'calibration' / 'core.json',
                                       validation_angles=pair_validation_angles(fit))
        assert target.read_bytes() == Path(direct.pyfai_path).read_bytes()
        assert (Path(f'{target}.provenance.json').read_bytes()
                == Path(f'{direct.pyfai_path}.provenance.json').read_bytes())
        assert (tmp_path / 'calibration' / 'dialog.xu.json').read_bytes() == Path(direct.xu_path).read_bytes()
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
        assert not dialog.refine_seed_note.text()  # a detector arm: it can start the refinement
        older = tmp_path / 'calibration' / 'independent.json'  # an independent-angle calibration
        older.write_text(json.dumps({'trans_function': {'constants': {'pi': 3.141592653589793}}}))
        dialog._fields['refine_seed_path'].setText(str(older))
        assert 'independent angles' in dialog.refine_seed_note.text()  # said at once, not after Refine
        dialog._fields['refine_seed_path'].setText(str(seed_path))
        assert not dialog.refine_seed_note.text()
        dialog._fields['refine_frames'].setText('12')
        dialog.refine_button.click()  # no calibrant confirmation: the calibrant chosen says what it is
        _wait(dialog)
        assert dialog.fit is not None, dialog.status.text()
        assert dialog.refine_seed.currentData() == 'fit'  # a further refinement starts from this one
        automatic = dialog.fit.asset['provenance']['automatic_refinement']
        assert automatic['calibrant'] == 'LaB6' and automatic['validation']['chosen'] == 'stacked_extended'
        # as the guided fit, it records which motor drives which circle (Codex round 11 residual)
        assert dialog.fit.asset['provenance']['instrument_geometry'] == dialog.geometry_editor.value().to_record()
        summary = dialog.fit_summary.toPlainText()
        assert 'Automatic refinement chose the detector mount and sample offset' in summary
        assert 'Held-out σ: five-parameter' in summary and 'nu rows used: 4 of 4' in summary
        target = (tmp_path / 'calibration' / 'refined.json').resolve()
        _save_to(monkeypatch, target)
        dialog.save_button.click()
        _wait(dialog)
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
        'Screening the scan', 'Finding ring points', 'Validating models', 'Fitting'}

    seed_path = tmp_path / 'five.json'
    seed_path.write_text(json.dumps(seed))
    dialog = StitchCalibrationDialog(tmp_path)
    try:
        dialog.set_source_context(spec, geometry_path=seed_path)
        dialog._fields['refine_frames'].setText('2')
        dialog.refine_button.click()
        _wait(dialog)
        assert 'needs 3-256 frames' in dialog.status.text()
        assert dialog.fit is None and not dialog.save_button.isEnabled()
    finally:
        dialog.close()


def test_the_arm_note_says_which_model_and_why(qapp, tmp_path):
    (tmp_path / 'frames').mkdir()
    frames = _guided_frames(tmp_path)
    dialog = StitchCalibrationDialog(tmp_path)
    try:
        dialog.frames = frames
        dialog._frame_motors = ('del', 'nu')
        dialog._render_frames()
        assert 'the detector mount and sample offset' in dialog.arm_note.text()
        dialog.scales.setChecked(True)
        assert 'five-parameter arm; motor scales are fitted' in dialog.arm_note.text()
        assert 'no xrayutilities file' in dialog.arm_note.text()
        dialog.scales.setChecked(False)
        dialog.frames = tuple(frame for frame in frames if frame.motors['nu'] in (-10., -3.))
        dialog._render_frames()
        assert 'at least 3 nu positions (2 so far)' in dialog.arm_note.text()
    finally:
        dialog.close()


def test_a_cancelled_guided_fit_ends_cancelled_not_failed(tmp_path):
    (tmp_path / 'frames').mkdir()
    frames = _guided_frames(tmp_path)
    cancelled = Event()
    cancelled.set()
    with pytest.raises(OperationCancelled):
        _run_job(('fit', (frames, {'arm_model': 'stacked_extended'})), cancelled, lambda *_: True)


def test_an_independent_angle_page_geometry_does_not_become_the_refinement_seed(qapp, tmp_path):
    """The page's older independent-angle calibration still predicts rings, but the refinement
    keeps starting from the current fit instead of refusing later."""
    older = tmp_path / 'older.json'
    older.write_text(json.dumps({'trans_function': {'constants': {'pi': 3.141592653589793}}}))
    dialog = StitchCalibrationDialog(tmp_path)
    try:
        dialog.set_source_context(SourceSpec(tmp_path / 'scan', SourceKind.SPEC, options={'scan': '1'}),
                                  geometry_path=older)
        assert dialog._text('refine_seed_path') == str(older) and dialog.refine_seed.currentData() == 'fit'
        dialog.refine_seed.setCurrentIndex(dialog.refine_seed.findData('file'))  # chosen anyway: said at once
        assert 'independent angles' in dialog.refine_seed_note.text()
    finally:
        dialog.close()


def test_the_xrayutilities_file_is_checked_over_the_scan_and_a_refusal_keeps_the_pyfai_file(qapp, tmp_path, monkeypatch):
    """Was: the window's XU extended fit. Now Save checks the partner on every frame of the scan
    whose metadata was read; a fit xrayutilities cannot take is saved and adopted without one."""
    (tmp_path / 'frames').mkdir()
    frames = _guided_frames(tmp_path)
    dialog = StitchCalibrationDialog(tmp_path)
    try:
        dialog.frames = frames
        dialog._frame_motors = ('del', 'nu')
        dialog._choices = [(index, {'del': d, 'nu': n}, None) for index, (d, n) in enumerate(ANGLES)]
        dialog._render_frames()
        dialog.fit_button.click()
        _wait(dialog)
        target = (tmp_path / 'calibration' / 'arm.json').resolve()
        _save_to(monkeypatch, target)
        adopted = []
        dialog.calibrationSaved.connect(lambda path, backend: adopted.append((path, backend)))
        dialog.save_button.click()
        _wait(dialog, timeout=120)
        assert adopted == [(str(target), 'multigeometry')], dialog.status.text()
        provenance = json.loads((tmp_path / 'calibration' / 'arm.xu.json.provenance.json').read_text())
        transfer = provenance['xu_transfer']
        assert transfer['validation_frames'] == len(ANGLES) and transfer['max_abs_dq_A^-1'] <= 1e-8
        # the partner records its instrument too (Codex round 11 residual)
        assert provenance['instrument_geometry'] == dialog.geometry_editor.value().to_record()
        dialog.scales.setChecked(True)
        dialog.fit_button.click()
        _wait(dialog)
        target = (tmp_path / 'calibration' / 'scaled.json').resolve()
        _save_to(monkeypatch, target)
        dialog.save_button.click()
        _wait(dialog)
        assert adopted[-1] == (str(target), 'multigeometry') and target.is_file()
        assert not (tmp_path / 'calibration' / 'scaled.xu.json').exists()
        assert 'no xrayutilities file' in dialog.status.text() and 'motor scales' in dialog.status.text()
    finally:
        dialog.close()