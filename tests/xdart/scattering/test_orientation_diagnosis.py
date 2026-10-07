"""Frozen worker, focus, bounded compute and direct-beam-consistency oracle."""
import threading
import time
import tracemalloc

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def dialog_at_fit(tmp_path, rotation=270):
    from xdart.gui.tabs.calibration.dialog import CalibrationDialog, CalibrationContext
    from tests.xdart.scattering.test_direct_beam_readiness import _fit_container
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    _fit_container(dialog, tmp_path, rotation)
    return dialog


def wait(dialog):
    from tests.xdart.scattering.test_direct_beam_readiness import settle
    settle(dialog)
    assert not dialog.active()


def geometry_state(dialog):
    return (dialog.camera.value(), dialog.geometry_editor.value(), dialog._text('beam_pixel'),
            dialog._text('rough_distance'), dialog._text('rough_mount'))


def test_diagnosis_uses_calibration_worker_cancels_joins_and_changes_no_geometry(qapp, tmp_path):
    from xdart.gui.pages.values import PageCleanup
    dialog = dialog_at_fit(tmp_path); dialog.show()
    try:
        prior = geometry_state(dialog); owner = dialog._owner
        dialog.open_orientation_check(); panel = dialog.orientation_panel
        panel.run_button.click()
        assert dialog.active() and dialog._owner is owner
        wait(dialog)
        finding = panel.finding
        assert finding is not None and finding.worker_thread_id != threading.get_ident()
        assert geometry_state(dialog) == prior and panel.finding_pane.toPlainText() == finding.reason
        panel.run_button.click(); dialog._cancel(); wait(dialog)
        assert geometry_state(dialog) == prior
        panel.run_button.click(); dialog.close(); wait(dialog)
        assert dialog.shutdown().status is PageCleanup.CLEAN
        assert not dialog.isVisible() and geometry_state(dialog) == prior
    finally:
        dialog.close(); dialog.deleteLater(); qapp.processEvents()


def test_staged_ring_run_uses_saved_calibration_json_cancels_and_closes_without_edits(qapp, tmp_path):
    from dataclasses import replace
    import hashlib
    import json
    from pathlib import Path
    from xdart.gui.pages.values import PageCleanup
    from xdart.gui.tabs.calibration.dialog import CalibrationDialog, CalibrationContext
    from xrd_tools import geometry_figures as gf
    from xrd_tools.core.scan import SourceKind, SourceSpec
    from xrd_tools.core.geometry import ImageOrientation
    from xrd_tools.core.geometry.camera import stitch_camera
    from xrd_tools.core.energy import wavelength_m_to_energy_eV
    from xrd_tools.integrate.calibration_seed import rough_arm_record
    from tests.core.test_geometry_figures import reference

    value = gf.GeometryInputs(); camera = stitch_camera(ImageOrientation())
    images = tmp_path/'images'; images.mkdir()
    for index, angle in enumerate((0., 15.)):
        _, image = reference(replace(value, readings=(('del', angle), ('nu', 0.))), camera)
        (images/f'rings_scan1_{index:04d}.raw').write_bytes(image.astype('<f4').tobytes())
    spec = tmp_path/'rings'
    spec.write_text('#F rings\n#O0 del nu energy\n#S 1 ascan del 0 15 1 1\n'
                    f'#P0 0 0 {wavelength_m_to_energy_eV(value.wavelength_m)}\n'
                    '#N 2\n#L del  nu\n0 0\n15 0\n')
    record = rough_arm_record(detector=value.detector, distance_m=value.distance_m,
        wavelength_m=value.wavelength_m, beam_px=(
            (value.shape[1]-1)/2+value.beam_offset_px[0],
            (value.shape[0]-1)/2+value.beam_offset_px[1]),
        camera=ImageOrientation(), orientation=value.first_pixel)
    saved = tmp_path/'fixed-arm.json'; saved.write_text(json.dumps(record))
    source = SourceSpec(spec, SourceKind.SPEC, options={'scan': '1', 'image_dir': str(images),
        'image_stem': 'rings', 'read_image_kwargs': {'detector_shape': value.shape, 'raw_dtype': '<f4'}})
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path)); dialog.show()
    try:
        dialog.set_source_context(source, image_rotation=0); wait(dialog)
        dialog._enumerate(); wait(dialog)
        dialog._prepare(); wait(dialog)
        assert len(dialog.frames) == 2, dialog.status.text()
        dialog.load_calibration(saved)
        dialog._source_details(); wait(dialog)
        dialog.rough_mode.setCurrentIndex(dialog.rough_mode.findData('skip'))
        seed = dialog._seed_options()
        assert seed['file'] == str(saved) and all(seed[k] is None for k in ('record', 'prediction', 'rough'))
        paths = (saved, spec, *(Path(f.image_path) for f in dialog.frames), *images.iterdir())
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        prior = geometry_state(dialog); owner = dialog._owner
        dialog.open_orientation_check(); panel = dialog.orientation_panel
        panel.run_button.click(); worker = owner._worker
        assert dialog.active() and worker is not None and dialog._job == 'orientation'
        wait(dialog)
        assert not worker.is_alive() and owner._identity is None
        assert panel.finding is not None and panel.finding.worker_thread_id != threading.get_ident(), dialog.status.text()
        assert dialog._orientation_geometry is not None and panel.figure_view.keyed_projection
        reason = panel.finding.reason.lower()
        assert 'declared beam' in reason and 'current calibration record' in reason
        assert 'zero-angle' in reason and 'consistency check' in reason and 'not an independent identification' in reason
        assert geometry_state(dialog) == prior
        panel.run_button.click(); worker = owner._worker
        panel.cancel_button.click()
        assert owner._cancel_commanded, 'the real worker must accept cancellation before terminal delivery'
        wait(dialog)
        assert not worker.is_alive() and owner._identity is None and panel.finding is None
        panel.run_button.click(); worker = owner._worker
        dialog.close()
        assert dialog._close_pending and owner._cancel_commanded
        wait(dialog)
        assert not worker.is_alive() and owner._identity is None
        assert dialog.shutdown().status is PageCleanup.CLEAN and not dialog.isVisible()
        assert geometry_state(dialog) == prior
        assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths} == before
    finally:
        dialog.close(); dialog.deleteLater(); qapp.processEvents()


def test_scoring_is_off_thread_with_frozen_reference_budget_and_bounded_frame_storage(qapp):
    from xrd_tools import geometry_figures as gf
    from tests.core.test_geometry_figures import inputs, reference
    from xrd_tools.core.geometry.camera import DetectorCamera
    value = inputs(); start_load = time.perf_counter()
    _, image = reference(value, DetectorCamera('x-', 'z+'))
    loading = time.perf_counter()-start_load
    frame = gf.OrientationFrame(image, value, None)
    timings = []; results = []; errors = []
    def compute():
        try:
            for _ in range(4):  # one cold and three warm; never hidden warm-up work
                start = time.perf_counter(); result = gf.diagnose_orientation((frame,))
                timings.append(time.perf_counter()-start); results.append(result)
            tracemalloc.start()
            result = gf.diagnose_orientation((frame,))
            retained, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
            results.append((retained, peak, result))
        except BaseException as error:
            errors.append(error)
    worker = threading.Thread(target=compute); worker.start()
    while worker.is_alive():
        qapp.processEvents(); worker.join(.01)
    if errors:
        raise errors[0]
    retained, peak, finding = results[-1]
    print({'loading_s': loading, 'cold_s': timings[0], 'warm_s': timings[1:],
           'peak_bytes': peak, 'retained_bytes': retained})
    assert max(timings[1:]) <= 1.0
    assert finding.evaluated_hypotheses == 32
    assert finding.worker_thread_id != threading.get_ident()
    pixels = image.size
    assert peak <= image.nbytes + 8*pixels + 2*pixels + 2*1024**2
    assert retained <= 2*1024**2
    assert not any(isinstance(v, np.ndarray) and v.size >= pixels for v in vars(finding).values())


@pytest.mark.parametrize('activation_order', ['before', 'after'])
@pytest.mark.parametrize('destination', ['camera', 'panel'])
def test_show_in_editor_and_panel_click_resolve_help_draft_once_and_keep_focus(qapp, tmp_path, activation_order, destination):
    from xdart.gui.widgets.controls_panel import FormRow
    from xdart.gui.widgets.context_help import owner_for
    dialog = dialog_at_fit(tmp_path); dialog.show(); dialog.activateWindow()
    dialog.open_orientation_check(); panel = dialog.orientation_panel
    row = FormRow(label='Draft note', path=('test', 'draft'), value='unchanged')
    row.set_help_topic('geometry.orientation', button=True); dialog.layout().addWidget(row)
    commits = []; row.valueChanged.connect(lambda *args: commits.append(args))
    try:
        prior = geometry_state(dialog)
        row.editor.setFocus(); qapp.processEvents()
        QtTest.QTest.keyClicks(row.editor, ' draft')
        QtTest.QTest.mouseClick(row.help_button, QtCore.Qt.MouseButton.LeftButton)
        qapp.processEvents(); owner = owner_for(dialog)
        assert owner._held_row is row and not commits
        event = QtCore.QEvent(QtCore.QEvent.Type.WindowActivate)
        if activation_order == 'before':
            QtWidgets.QApplication.sendEvent(dialog, event)
        if destination == 'camera':
            target = dialog.camera.rotation
            QtTest.QTest.mouseClick(panel.show_camera_button, QtCore.Qt.MouseButton.LeftButton)
        else:
            target = panel.step_row.editor
            QtTest.QTest.mouseClick(target, QtCore.Qt.MouseButton.LeftButton)
            target.hidePopup()
        if activation_order == 'after':
            QtWidgets.QApplication.sendEvent(dialog, event)
        for _ in range(3):
            qapp.processEvents()
        assert target.hasFocus() and owner._held_row is None
        assert len(commits) == 1 or (row.refusal() and row.editor.property('invalid'))
        assert geometry_state(dialog) == prior and not row._help_focus_transfer
    finally:
        dialog.close(); dialog.deleteLater(); qapp.processEvents()


def test_live_schematic_tracks_current_geometry_and_selected_readbacks(qapp, tmp_path):
    dialog = dialog_at_fit(tmp_path)
    try:
        dialog.show_instrument_schematic(); window = dialog.instrument_schematic
        first = window.projection
        assert first.panels and window.figure_view.keyed_projection
        from tests.xdart.scattering.test_direct_beam_readiness import _reverse
        _reverse(dialog, 'del'); qapp.processEvents()
        assert window.projection != first
        del_circle = next(c for c in window.projection.circles if c.name == 'del')
        assert del_circle.axis == dict(dialog.geometry_editor.value().circle_axes())['del']
        dialog.set_schematic_readbacks({m: 7. for _, m in dialog.geometry_editor.value().motors})
        assert any(c.angle_deg == 7. for c in window.projection.circles)
        dialog.set_schematic_readbacks({})
        assert window.legend.toPlainText().startswith('Missing motor readback:')
    finally:
        dialog.close(); dialog.deleteLater(); qapp.processEvents()


def test_existing_direct_beam_advice_is_in_the_consistent_set_after_camera_and_sense_edits(qapp, tmp_path):
    from tests.xdart.scattering.test_direct_beam_readiness import _reverse, _surfaces, detect
    from xrd_tools.core.geometry.camera import DetectorCamera
    dialog = dialog_at_fit(tmp_path)
    try:
        dialog.open_orientation_check(); panel = dialog.orientation_panel
        panel.run_button.click(); wait(dialog)
        assert panel.finding and 'single-axis' in panel.finding.reason.lower()
        original_camera = dialog.camera.value()
        original_senses = dialog.geometry_editor.value().arm_senses()
        _reverse(dialog, 'del')
        assert panel.finding is None
        advice = _surfaces(dialog)
        assert any('direct beam moved' in text for text in advice)
        panel.run_button.click(); wait(dialog)
        suggested = DetectorCamera(*(axis[0]+('-' if axis[1] == '+' else '+')
                                      for axis in dialog.camera.value().to_xu()))
        current_senses = dialog.geometry_editor.value().arm_senses()
        assert any(c.camera == suggested and c.senses == (current_senses['del'], current_senses['nu'])
                   for c in panel.finding.candidates)
        assert any(c.camera == original_camera and c.senses[0] == original_senses['del']
                   for c in panel.finding.candidates)
        assert {c.senses[0] for c in panel.finding.candidates} == {-1, 1}
        assert {c.senses[1] for c in panel.finding.candidates} == {-1, 1}
        assert 'nu sense is unmeasured' in panel.finding.reason.lower()
        labels = [panel.candidate_row.editor.itemText(i) for i in range(panel.candidate_row.editor.count())]
        assert all('nu unmeasured' in label for label in labels)
        assert any('keeps configured senses' in label for label in labels)
        dialog.camera.set_value(suggested)
        assert panel.finding is None and dialog.status.text() == dialog._STALE_DISTANCE
        detect(dialog); dialog.beam_button.click(); wait(dialog)
        panel.run_button.click(); wait(dialog)
        assert any(c.camera == suggested for c in panel.finding.candidates)
        dialog._set_field_text('beam_strip', '3')
        assert panel.finding is None
    finally:
        dialog.close(); dialog.deleteLater(); qapp.processEvents()
