"""HELP-RESIDUALS-20261011 frozen GUI oracle, calibration and status (Claude, October 11): GA1, GA2, GA4 and GB1.

Real dialogs, real orientation jobs and real page outcome handlers; new names are reached inside test bodies.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from xrd_tools.analysis.module_transaction import ModuleDisposition


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_ga1_al2o3_opens_pyfai_calib2_as_alpha_al2o3_with_its_head_start(tmp_path):
    from xdart.gui.tabs.calibration.dialog import _pyfai_arguments
    from tests.core.test_calibration_seed import _frame, _record
    folder = tmp_path / 'frame-20'
    folder.mkdir()
    frame = replace(_frame({'del': 20., 'nu': 0.}), image_path=str(folder / 'scan14_0020.edf'))
    seed = dict(calibrant='Al2O3', del_motor='del', nu_motor='nu', record=None, frames=(),
                file=str(_record(tmp_path)), detector=None)
    arguments = _pyfai_arguments(frame, seed)
    assert arguments[:3] == ('-c', 'alpha_Al2O3', '--poni'), arguments
    assert Path(arguments[3]).parent == folder and Path(arguments[3]).is_file()


def _wait(dialog):
    from tests.xdart.scattering.test_direct_beam_readiness import settle
    settle(dialog)
    assert not dialog.active()


def _figure_shows(dialog, candidate):
    from xrd_tools import geometry_figures as gf
    expected, _ = gf._display_prediction(dialog._orientation_geometry, candidate.camera, candidate.senses,
                                         dialog._orientation_figure.display_binning)
    shown = dialog._orientation_figure.curves
    return len(shown) == len(expected.curves) and all(
        np.array_equal(actual.xy, oracle.xy) for actual, oracle in zip(shown, expected.curves))


def test_ga2_the_orientation_figure_follows_the_selected_hypothesis(qapp, tmp_path):
    from xdart.gui.tabs.calibration.dialog import CalibrationContext, CalibrationDialog
    from xrd_tools import geometry_figures as gf
    from xrd_tools.core.energy import wavelength_m_to_energy_eV
    from xrd_tools.core.geometry import ImageOrientation
    from xrd_tools.core.geometry.camera import stitch_camera
    from xrd_tools.core.scan import SourceKind, SourceSpec
    from xrd_tools.integrate.calibration_seed import rough_arm_record
    from tests.core.test_geometry_figures import reference
    value = gf.GeometryInputs()
    camera = stitch_camera(ImageOrientation())
    images = tmp_path / 'images'
    images.mkdir()
    for index, angle in enumerate((0., 15.)):
        _, image = reference(replace(value, readings=(('del', angle), ('nu', 0.))), camera)
        (images / f'rings_scan1_{index:04d}.raw').write_bytes(image.astype('<f4').tobytes())
    spec = tmp_path / 'rings'
    spec.write_text('#F rings\n#O0 del nu energy\n#S 1 ascan del 0 15 1 1\n'
                    f'#P0 0 0 {wavelength_m_to_energy_eV(value.wavelength_m)}\n'
                    '#N 2\n#L del  nu\n0 0\n15 0\n')
    record = rough_arm_record(detector=value.detector, distance_m=value.distance_m,
                              wavelength_m=value.wavelength_m,
                              beam_px=((value.shape[1] - 1) / 2 + value.beam_offset_px[0],
                                       (value.shape[0] - 1) / 2 + value.beam_offset_px[1]),
                              camera=ImageOrientation(), orientation=value.first_pixel)
    saved = tmp_path / 'fixed-arm.json'
    saved.write_text(json.dumps(record))
    source = SourceSpec(spec, SourceKind.SPEC, options={'scan': '1', 'image_dir': str(images),
        'image_stem': 'rings', 'read_image_kwargs': {'detector_shape': value.shape, 'raw_dtype': '<f4'}})
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    dialog.show()
    try:
        # Staged a quarter turn: the configured camera is not the candidate list's first.
        dialog.set_source_context(source, image_rotation=90)
        _wait(dialog)
        dialog._enumerate(); _wait(dialog)
        dialog._prepare(); _wait(dialog)
        dialog.load_calibration(saved)
        dialog.camera.set_value(stitch_camera(ImageOrientation(rotation=90)))  # the staged camera
        dialog._source_details(); _wait(dialog)
        dialog.rough_mode.setCurrentIndex(dialog.rough_mode.findData('skip'))
        dialog.open_orientation_check()
        panel = dialog.orientation_panel
        panel.run_button.click()
        _wait(dialog)
        finding = panel.finding
        assert finding is not None and len(finding.candidates) >= 2, dialog.status.text()
        selected = finding.candidates[panel.candidate_row.editor.currentData()]
        assert _figure_shows(dialog, selected), 'the figure draws the hypothesis the selector shows'
        # A finding whose candidates arrive reordered keeps the same camera and senses selected.
        panel.show_finding(replace(finding, candidates=tuple(reversed(finding.candidates))))
        again = panel.finding.candidates[panel.candidate_row.editor.currentData()]
        assert (again.camera, again.senses) == (selected.camera, selected.senses)
    finally:
        # Undo the staged camera edit, or close opens the modal unsaved-setup prompt (reloading the same file is a no-op).
        if dialog._saved_setup is not None:
            dialog.camera.set_value(dialog._saved_setup[1])
        dialog.close(); _wait(dialog); dialog.deleteLater(); qapp.processEvents()


def test_ga4_fit_steps_show_one_line_and_keep_the_full_summary_as_tooltip(qapp, tmp_path, monkeypatch):
    from xdart.gui.tabs.calibration.dialog import CalibrationContext, CalibrationDialog
    from xrd_tools.session.instrument import InstrumentGeometry
    from xrd_tools.core.geometry import ImageOrientation
    from xrd_tools.core.geometry.camera import stitch_camera
    from xrd_tools.integrate.stitch_calibration import adopt_calibration_pair, capture_calibration_frames
    from tests.xdart.scattering.test_stitch_calibration_fit_dialog import _single_arm_frames
    from tests.core.test_single_detector_arm import _powder_scan
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    dialog.show()
    try:
        dialog.set_instrument(InstrumentGeometry('fourc', plane='vertical'))
        dialog.rough_mode.setCurrentIndex(dialog.rough_mode.findData('skip'))
        dialog.camera.set_value(stitch_camera(ImageOrientation()))
        dialog._set_field_text('beam_detector', 'Pilatus300kw')
        fit_source = _powder_scan(tmp_path / 'fit-scan', (6., 18., 30., 42.), {'nu_offset': .02, 'del_offset': -.05})
        dialog.set_source_context(fit_source, image_rotation=0); _wait(dialog)
        dialog.geometry_editor.zero_edit('tth').setText(f'{-np.rad2deg(.05):.12g}')
        (tmp_path / 'frames').mkdir()
        pairs = _single_arm_frames(tmp_path)
        frames = capture_calibration_frames(dialog._spec(), (0, 1, 2, 3), dialog.session_dir,
                                            rotation=0, arm=('tth', None))
        dialog.frames = tuple(adopt_calibration_pair(frame, pair.poni_path, pair.npt_path)
                              for frame, pair in zip(frames, pairs))
        dialog._frame_motors = ('tth', None)
        dialog._render_frames()
        dialog.fit_button.click(); _wait(dialog)
        assert dialog.fit is not None, dialog.status.text()
        summary = dialog.fit_summary.toPlainText()
        assert '\n' in summary, 'the fit summary is multiline'
        table = dialog.checklist_table
        for row in (4, 5):
            cell = table.item(row, 2)
            assert '\n' not in cell.text() and cell.text() == summary.splitlines()[0], cell.text()
            assert cell.toolTip() == summary
    finally:
        dialog.close(); _wait(dialog); dialog.deleteLater(); qapp.processEvents()


def _terminal_text(result, family):
    """``tests/xdart/scattering/test_refusal_explanations_gui.py``'s page projection: the readiness text."""
    from xdart.gui.pages.operation_owner import OperationIdentity, OperationTerminalStatus
    from xdart.gui.pages.services import empty_host_services
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    class Status:
        def show(self, text, timeout_ms=0):
            pass
    identity = OperationIdentity(1, result.request)
    if family == 'stitch':
        from xdart.gui.tabs.stitch.page import StitchPage
        from xdart.gui.tools.stitch_owner import (
            StitchOwnerAction, StitchOwnerOutcome, StitchOwnerOutcomeKind, StitchOwnerUpdate)
        page = StitchPage(empty_host_services(Status()))
        update = StitchOwnerUpdate(identity, StitchOwnerAction.RUN, terminal_status=OperationTerminalStatus.RETURNED,
                                   outcome=StitchOwnerOutcome(StitchOwnerOutcomeKind.RESULT, result=result))
        accept = page._accept_outcome
    else:
        from xdart.gui.tabs.rsm.page import RSMPage
        from xdart.gui.tools.rsm_owner import RSMOwnerAction, RSMOwnerOutcome, RSMOwnerOutcomeKind, RSMOwnerUpdate
        page = RSMPage(empty_host_services(Status()))
        update = RSMOwnerUpdate(identity, RSMOwnerAction.RUN, terminal_status=OperationTerminalStatus.RETURNED,
                                outcome=RSMOwnerOutcome(RSMOwnerOutcomeKind.RESULT, result=result))
        accept = page._accept
    try:
        accept(update)
        return page.run_bar.readinessLabel.text()
    finally:
        (page.shutdown if family == 'stitch' else page.close_page)()
        page.deleteLater(); app.processEvents()


def _variants(result):
    terminal = result.terminal
    return (replace(result, terminal=replace(terminal, disposition=ModuleDisposition.FAILED)),
            replace(result, terminal=replace(terminal, diagnostic='')))


def test_gb1_stitch_and_rsm_terminals_read_code_colon_reason(qapp, tmp_path, monkeypatch):
    from xrd_tools.analysis.stitch_operation import run_stitch_operation
    from tests.core.test_native_stitch_operation import _native
    (tmp_path / 'stitch').mkdir()
    request = _native(tmp_path / 'stitch', 'multigeometry', 'q_A^-1', grouped=False)
    np.save(request.plan.mask_path, np.ones((195, 487), dtype=np.uint8))
    refused = run_stitch_operation(request)
    code, reason = refused.terminal.code, refused.terminal.diagnostic
    assert refused.terminal.disposition is ModuleDisposition.REFUSED and reason
    failed, bare = _variants(refused)
    assert f'refused: {code}: {reason}' in _terminal_text(refused, 'stitch')
    assert f'failed: {code}: {reason}' in _terminal_text(failed, 'stitch')
    text = _terminal_text(bare, 'stitch')
    assert f'refused: {code}' in text and not text.rstrip().endswith(':'), text

    from tests.core.test_rsm_operation_r2 import _form, _write_member
    from xrd_tools.analysis.rsm_values import prepare_rsm_tool_v2
    from xrd_tools.analysis.rsm_operation import run_rsm_operation_v2
    form = _form(tmp_path / 'rsm', (_write_member(tmp_path / 'rsm', 0),))
    form = replace(form, output_path=tmp_path / 'rsm' / 'output' / 'late' / 'rsm-v2.nexus')
    rsm_request = prepare_rsm_tool_v2(form).request
    (tmp_path / 'rsm' / 'output').mkdir(parents=True, exist_ok=True)
    (tmp_path / 'rsm' / 'output' / 'late').write_text('blocking file')
    rsm_refused = run_rsm_operation_v2(rsm_request)
    rsm_code, rsm_reason = rsm_refused.terminal.code, rsm_refused.terminal.diagnostic
    assert rsm_refused.terminal.disposition is ModuleDisposition.REFUSED and rsm_reason
    rsm_failed, rsm_bare = _variants(rsm_refused)
    assert f'RSM refused: {rsm_code}: {rsm_reason}' in _terminal_text(rsm_refused, 'rsm')
    assert f'RSM failed: {rsm_code}: {rsm_reason}' in _terminal_text(rsm_failed, 'rsm')
    assert _terminal_text(rsm_bare, 'rsm').rstrip().endswith(f'RSM refused: {rsm_code}')
