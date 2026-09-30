"""Actual pyFAI Save -> matched points -> guided row; native window controls."""
from dataclasses import replace
import json
import os
from pathlib import Path
from threading import Event

import numpy as np
import pytest
import tifffile

from xdart.gui.tabs.scattering.experiment_authoring import prepare_calibration_request, run_calibration
from xdart.gui.tabs.scattering.operation_values import OperationIdentity, OperationTerminalStatus
from xdart.gui.tabs.stitch.calibration import StitchCalibrationDialog
from xrd_tools.core.scan import SourceSpec, SourceKind


@pytest.mark.parametrize('action', ['save', 'one_click', 'seeded', 'switched_image', 'close', 'missing_points', 'failed_save', 'child_failure'])
def test_real_guided_child_save_captures_only_matched_enabled_points(tmp_path, monkeypatch, action):
    hooks = tmp_path / 'hooks'; hooks.mkdir()
    (hooks/'sitecustomize.py').write_text('''
import os
from pathlib import Path
import numpy as np
from silx.gui import qt
scratch = Path(__file__).parent
qt.QSettings.setPath(qt.QSettings.IniFormat, qt.QSettings.UserScope, str(scratch))
original_exec = qt.QApplication.exec

def automated_exec(app):
    def finish():
        try:
            from pyFAI.gui.tasks.IntegrationTask import IntegrationTask
            from pyFAI.gui.model.PeakModel import PeakModel
            from pyFAI.calibrant import get_calibrant
            from pyFAI.detectors import detector_factory
            action = os.environ['XDART_TEST_GUIDED_ACTION']
            task = next(w for w in app.allWidgets() if isinstance(w, IntegrationTask))
            if action == 'seeded':  # what pyFAI opened with, before anything is changed
                import json
                model = task.model()
                (scratch / 'seeded.json').write_text(json.dumps({
                    'valid': model.fittedGeometry().isValid(),
                    'calibrant': model.experimentSettingsModel().calibrantModel().calibrant().name}))
            if action != 'close':
                settings = task.model().experimentSettingsModel()
                if action == 'switched_image':
                    from pyFAI.io import image as image_io
                    with settings.image().lockContext() as model:
                        model.setFilename(str(scratch / 'other.tif'))
                        model.setValue(image_io.read_image_data(str(scratch / 'other.tif')))
                        model.setSynchronized(True)
                settings.detectorModel().setDetector(detector_factory('Detector', {'pixel1': 1e-4, 'pixel2': 1e-4, 'max_shape': [8, 10]}))
                settings.calibrantModel().setCalibrant(get_calibrant('LaB6'))
                settings.wavelength().setValue(1e-10)
                geometry = task.model().fittedGeometry()
                with geometry.lockContext():
                    for name, value in [('distance',.1), ('poni1',.0004), ('poni2',.0005), ('rotation1',0.), ('rotation2',0.), ('rotation3',0.), ('wavelength',1e-10)]:
                        getattr(geometry,name)().setValue(value)
                task._geometryTabs.setGeometryModel(geometry)
                if action != 'missing_points':
                    peak = PeakModel(); peak.setName('ring'); peak.setRingNumber(1)
                    peak.setCoords(np.array([[1.,2.],[3.,4.]])); peak.setEnabled(True)
                    task.model().peakSelectionModel().append(peak)
                if action == 'failed_save':
                    from pyFAI.gui.dialog import MessageBox
                    MessageBox.exception = lambda *args, **kwargs: None
                    def broken(self):
                        raise RuntimeError('writer unavailable')
                    IntegrationTask._IntegrationTask__getPoni = broken
                qt.QFileDialog.exec = lambda dialog: 0  # a guided frame saves without a file dialog
                task._savePoniButton.click()
                if action in ('one_click', 'seeded'):
                    return  # the save itself closes pyFAI and returns to xdart
            app.exit(7 if action == 'child_failure' else 0)
        except BaseException:
            import traceback; traceback.print_exc(); app.exit(98)
    qt.QTimer.singleShot(0, finish)
    qt.QTimer.singleShot(15000, lambda: app.exit(99))
    return original_exec()
qt.QApplication.exec = automated_exec
''')
    monkeypatch.setenv('PYTHONPATH', str(hooks) + os.pathsep + str(Path(__file__).resolve().parents[3] / 'src'))
    monkeypatch.setenv('XDART_TEST_GUIDED_ACTION', action)
    image = tmp_path/'selected.tif'
    tifffile.imwrite(image, np.arange(80, dtype=np.uint16).reshape(8,10))
    tifffile.imwrite(hooks/'other.tif', (100 + np.arange(80, dtype=np.uint16)).reshape(8,10))
    arguments = ()
    if action == 'seeded':  # a predicted geometry and the calibrant, as a guided frame opens
        from pyFAI.detectors import Detector
        from pyFAI.integrator.azimuthal import AzimuthalIntegrator
        from xrd_tools.integrate.calibration_seed import FrameSeed, write_seed_poni
        ai = AzimuthalIntegrator(dist=.1, poni1=.0004, poni2=.0005, wavelength=1e-10,
                                 detector=Detector(1e-4, 1e-4, max_shape=(8, 10)))
        arguments = ('-c', 'LaB6', '--poni', str(write_seed_poni(FrameSeed(ai, ()), tmp_path / 'seed.poni')))
    request = replace(prepare_calibration_request(str(image)), points_directory=str(tmp_path),
                      arguments=arguments)
    terminal = run_calibration(request, OperationIdentity(1), Event(), lambda *_: None, lambda _: True)
    if action in ('switched_image', 'missing_points', 'child_failure'):
        assert terminal.status is OperationTerminalStatus.FAILED
        assert terminal.payload.control_points_path is None
        assert not terminal.payload.candidates
        assert terminal.payload.loaded_image_sha256 is None
        if action == 'switched_image':
            assert 'loaded pyFAI image differs' in terminal.diagnostic
    else:
        assert terminal.status is OperationTerminalStatus.RETURNED, terminal.diagnostic
        if action == 'seeded':
            assert json.loads((hooks / 'seeded.json').read_text()) == {'valid': True, 'calibrant': 'LaB6'}
        if action in ('save', 'one_click', 'seeded'):
            result = terminal.payload
            saved = Path(result.last_saved_path)  # named by xdart in the frame's folder
            assert saved.parent == tmp_path and saved.name.startswith('pyfai-') and saved.suffix == '.poni'
        if action == 'save':
            assert Path(result.control_points_path).parent == tmp_path
            from pyFAI.control_points import ControlPoints
            points = ControlPoints(result.control_points_path)
            group = points.get(lbl=points.get_labels()[0])
            assert group.ring == 0
            assert group.points == [[1.,2.],[3.,4.]]
            assert Path(result.control_points_path).is_file()
            from xrd_tools.integrate.stitch_calibration import (
                CalibrationFrame, adopt_calibration_pair, calibration_image_identity,
                save_calibration_session, load_calibration_session,
            )
            import hashlib
            frame = CalibrationFrame('selected', str(image), 'image_file', {}, 4,
                str(image), 0, str(image), hashlib.sha256(image.read_bytes()).hexdigest(),
                (8,10), 0, {'del':12.,'nu':3.}, None)
            assert result.loaded_image_shape == frame.shape
            assert result.loaded_image_sha256 == calibration_image_identity(tifffile.imread(image))[1]
            adopted = adopt_calibration_pair(frame, result.last_saved_path, result.control_points_path,
                                             model_image_sha256=result.loaded_image_sha256)
            assert adopted.complete
            assert adopted.motors == {'del':12.,'nu':3.}
            assert adopted.pairing_policy == 'pyfai-model-image-v1'
            session = save_calibration_session(tmp_path, (adopted,))
            assert load_calibration_session(session) == (adopted,)
        elif action not in ('one_click', 'seeded'):
            assert terminal.payload.last_saved_path is None
            assert terminal.payload.control_points_path is None
    assert not list(tmp_path.glob('.xdart-calibration-*'))


def test_modeless_window_exposes_guided_source_controls_and_preserves_session(tmp_path):
    dialog = StitchCalibrationDialog(tmp_path)
    source = SourceSpec(tmp_path/'beam', SourceKind.SPEC, options={'scan':'14', 'image_dir':str(tmp_path/'images')})
    dialog.set_source_context(source, image_rotation=180, del_motor='delta', nu_motor='gamma')
    assert not dialog.isModal()
    assert dialog.tabs.count() == 3  # 3 · Calibrate and fit holds the fit (maintainer, 2026-09-29)
    assert dialog._spec().options['scan'] == '14'
    assert dialog.rotation.currentData() == 180
    assert not dialog.scales.isChecked()
    assert dialog._text('del_motor') == 'delta'
    previous = dialog.session_dir
    dialog._new_session()
    assert dialog.session_dir != previous
    assert not dialog.active()
    from xdart.gui.pages.values import PageCleanup
    assert dialog.shutdown().status is PageCleanup.CLEAN
    dialog.close()


def test_calibration_rows_name_their_source_image_and_a_save_selects_the_next_frame(tmp_path):
    from xrd_tools.integrate.stitch_calibration import CalibrationFrame
    dialog = StitchCalibrationDialog(tmp_path)

    def frame(index, raw, within=0):
        return CalibrationFrame(f'frame-{index}', str(tmp_path/'beam'), 'spec', {}, index,
            str(tmp_path/'images'/raw), within, str(tmp_path/f'frame-{index}'/f'{Path(raw).stem}.edf'),
            'sha', (8, 10), 0, {'del': 5., 'nu': 0.}, None)
    done = dict(poni_path='p', npt_path='n', control_image_sha256='c', pairing_policy='pyfai-model-image-v1')
    dialog.frames = (frame(0, 'scan14_0000.raw'), replace(frame(40, 'scan14_0040.raw'), **done),
                     frame(143, 'run.h5', 7))
    dialog._render_frames()
    assert dialog.frame_table.item(0, 2).text() == 'scan14_0000.raw'  # the source, not the staged copy
    assert dialog.frame_table.item(2, 2).text() == 'run.h5 #7'
    assert 'staged for pyFAI' in dialog.frame_table.item(0, 2).toolTip()
    dialog._select_next_frame(dialog.frames[0])  # 40 is complete: 143 is next
    assert dialog._selected_frame().index == 143
    dialog._select_next_frame(dialog.frames[2])  # none after it: the first still pending
    assert dialog._selected_frame().index == 0
    from xdart.gui.pages.values import PageCleanup
    assert dialog.shutdown().status is PageCleanup.CLEAN
    dialog.close()


def test_a_frame_opens_pyfai_with_its_calibrant_energy_and_predicted_geometry(tmp_path):
    from xdart.gui.tabs.stitch.calibration import _pyfai_arguments
    from tests.core.test_calibration_seed import _frame, _record
    folder = tmp_path / 'frame-20'
    folder.mkdir()
    frame = replace(_frame({'del': 20., 'nu': 0.}), image_path=str(folder / 'scan14_0020.edf'))
    seed = dict(calibrant='LaB6', del_motor='del', nu_motor='nu', record=None, frames=(),
                file=str(_record(tmp_path)), detector=None)
    arguments = _pyfai_arguments(frame, seed)
    assert arguments[:3] == ('-c', 'LaB6', '--poni')
    assert Path(arguments[3]).parent == folder and Path(arguments[3]).is_file()
    # no calibration yet: the calibrant, the frame's energy and a saved frame's detector
    assert _pyfai_arguments(frame, dict(seed, file=None, detector='PilatusCdTe300kw')) == (
        '-c', 'LaB6', '-e', '17.000018', '-D', 'PilatusCdTe300kw')


def test_the_frame_table_lists_each_frames_expected_rings(tmp_path):
    from tests.core.test_calibration_seed import _frame, _record
    dialog = StitchCalibrationDialog(tmp_path)
    dialog.frames = (replace(_frame({'del': 20., 'nu': 0.}), identity='frame-20'),)
    dialog._render_frames()
    assert dialog.frame_table.item(0, 5).text() == '—'  # nothing to predict from yet
    assert 'calibrate one frame' in dialog.frame_table.item(0, 5).toolTip()  # it then predicts the rest
    dialog._fields['refine_seed_path'].setText(str(_record(tmp_path)))
    dialog._render_frames()
    cell = dialog.frame_table.item(0, 5)
    assert cell.text() and cell.text() != '—' and cell.toolTip().startswith('Ring ')
    from xdart.gui.pages.values import PageCleanup
    assert dialog.shutdown().status is PageCleanup.CLEAN
    dialog.close()


def test_a_rough_geometry_comes_first_and_predicts_the_rings(tmp_path):
    from xdart.gui.tabs.stitch.calibration import _pyfai_arguments
    from pyFAI.io.ponifile import PoniFile
    from tests.core.test_calibration_seed import _frame, _record
    dialog = StitchCalibrationDialog(tmp_path)
    assert [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())] == [
        '1 · Geometry', '2 · Select frames', '3 · Calibrate and fit']
    visible = dialog._row_visible
    assert visible('beam_source') and not visible('rough_distance')  # a direct-beam scan by default
    dialog.rough_mode.setCurrentIndex(dialog.rough_mode.findData('manual'))
    assert visible('rough_distance') and visible('beam_pixel') and not visible('beam_source')
    dialog.rough_mode.setCurrentIndex(dialog.rough_mode.findData('scan'))  # and back: every scan row returns
    assert all(visible(key) for key in ('beam_source', 'beam_scan', 'beam_images', 'beam_stem'))
    assert not dialog._fit_options.isHidden()  # the strip and ROI, under Fit options
    dialog.rough_mode.setCurrentIndex(dialog.rough_mode.findData('manual'))
    assert dialog._fit_options.isHidden()
    # the Detector panel serves every step: its rows stay, whatever the rough geometry
    assert all(visible(key) for key in ('beam_detector', 'raw', 'beam_size', 'beam_orientation', 'saturation', 'mask'))
    folder = tmp_path / 'frame-20'
    folder.mkdir()
    dialog.frames = (replace(_frame({'del': 20., 'nu': 0.}), identity='frame-20',
                             image_path=str(folder / 'scan14_0020.edf')),)
    dialog._render_frames()
    assert dialog.frame_table.item(0, 5).text() == '—'  # no distance typed yet
    for key, value in (('rough_distance', '500'), ('beam_pixel', '737, 97'), ('beam_detector', 'Pilatus300kw')):
        dialog._set_field_text(key, value)
    dialog._fields['refine_seed_path'].setText(str(_record(tmp_path)))  # a calibration file too
    dialog._render_frames()
    assert dialog.frame_table.item(0, 5).toolTip().startswith('Ring ')
    arguments = _pyfai_arguments(dialog.frames[0], dialog._seed_options())
    seeded = PoniFile(arguments[3])
    assert seeded.dist == pytest.approx(0.5)  # this session's rough geometry, not the file
    # the beam pixel is (x, y) = (column, row), as the direct-beam fit reads it
    assert (seeded.poni2, seeded.poni1) == pytest.approx((737.5 * 172e-6, 97.5 * 172e-6))
    # a beamline whose del motor is "delta": the zero motor "del" names the circle
    dialog._frame_motors = ('delta', 'nu')
    dialog.frames = (replace(dialog.frames[0], motors={'delta': 20., 'nu': 0.}),)
    assert dialog._rough_record() is not None
    dialog.rough_mode.setCurrentIndex(dialog.rough_mode.findData('skip'))
    assert not visible('beam_pixel') and dialog._seed_options()['rough'] is None
    assert PoniFile(_pyfai_arguments(dialog.frames[0], dialog._seed_options())[3]).dist == pytest.approx(0.39)
    from xdart.gui.pages.values import PageCleanup
    assert dialog.shutdown().status is PageCleanup.CLEAN
    dialog.close()


def test_the_geometry_comes_first_and_names_the_arm_motors(tmp_path):
    dialog = StitchCalibrationDialog(tmp_path)
    assert dialog.tabs.tabText(0) == '1 · Geometry'
    assert dialog._tab_pages['rough'].isAncestorOf(dialog.geometry_editor)
    source = SourceSpec(tmp_path/'beam', SourceKind.SPEC, options={'scan': '14'})
    dialog.set_source_context(source, del_motor='delta', nu_motor='nu')
    assert dialog.geometry_editor.value().stitch_arm() == ('delta', 'nu')
    assert dialog.settings()['instrument_geometry']['motors']['del'] == 'delta'
    dialog.geometry_editor.geometry.setCurrentIndex(dialog.geometry_editor.geometry.findData('twoc'))
    assert dialog.geometry_editor.value().stitch_arm() == ('tth', None)  # psic's arm, nu held at zero
    label = dialog.geometry_editor._form.labelForField(dialog.geometry_editor._rows['tth']).text()
    assert label == 'tth (detector) · z-  ·  used by Stitch'  # its axis: psic's del, vertical plane
    from xdart.gui.pages.values import PageCleanup
    assert dialog.shutdown().status is PageCleanup.CLEAN
    dialog.close()


def _finish(dialog, timeout=30):
    import time
    from pyqtgraph.Qt import QtWidgets
    until = time.monotonic() + timeout
    while dialog.active() and time.monotonic() < until:
        QtWidgets.QApplication.processEvents()
        time.sleep(.005)
    for _ in range(5):  # the 50 ms poll delivers the terminal
        QtWidgets.QApplication.processEvents()
        time.sleep(.06)
    assert not dialog.active(), dialog.status.text()


def test_a_twoc_scan_is_read_and_staged_as_one_2theta_arm(tmp_path):
    from xrd_tools.core.geometry.stacked_arm import is_single_arm
    from xrd_tools.session.instrument import InstrumentGeometry
    spec = tmp_path / 'powder'
    spec.write_text('#F powder\n#E 1\n#D Thu Jan 1 00:00:00 1970\n#C test\n'
                    '#O0 tth th\n#S 1 ascan tth 10 30 2 1\n#D today\n#P0 10 5\n#N 1\n#L tth\n10\n20\n30\n')
    images = tmp_path / 'images'
    images.mkdir()
    for index in range(3):
        (images / f'powder_scan1_{index:04d}.raw').write_bytes(np.zeros((4, 6), '<i4').tobytes())
    source = SourceSpec(spec, SourceKind.SPEC, options={'scan': '1', 'image_dir': str(images),
        'image_stem': 'powder_scan1_', 'read_image_kwargs': {'detector_shape': (4, 6), 'raw_dtype': '<i4'}})
    dialog = StitchCalibrationDialog(tmp_path)
    try:
        dialog.set_instrument(InstrumentGeometry('twoc'))
        dialog.set_source_context(source)
        dialog._enumerate()
        _finish(dialog)
        assert dialog.selection.columnCount() == 3, dialog.status.text()  # Use, Frame, tth: no nu
        assert dialog.selection.horizontalHeaderItem(2).text() == 'tth (°)'
        assert [dialog.selection.item(row, 2).text() for row in range(3)] == ['10', '20', '30']
        dialog._prepare()
        _finish(dialog)
        assert dialog._frame_motors == ('tth', None)
        assert [frame.motors for frame in dialog.frames] == [{'tth': 10.}, {'tth': 20.}, {'tth': 30.}]
        assert 'holds one circle at zero' in dialog.arm_note.text()
        # the direct-beam zero "del" names twoc's 2θ: the rough geometry is a single arm
        dialog.rough_mode.setCurrentIndex(dialog.rough_mode.findData('manual'))
        for key, value in (('rough_distance', '500'), ('beam_pixel', '3, 2'), ('beam_detector', 'Pilatus300kw')):
            dialog._set_field_text(key, value)
        assert dialog._text('beam_motor') == 'del' and is_single_arm(dialog._rough_record())
        # renaming the 2θ motor after staging: the frames read tth, so Fit refuses
        dialog.set_instrument(InstrumentGeometry('twoc', (('tth', 'ttheta'),)))
        dialog._fit()
        assert ('staged reading tth as a vertical 2θ arm; the Geometry now names ttheta as a vertical 2θ arm'
                in dialog.status.text())
        assert dialog.fit is None
    finally:
        from xdart.gui.pages.values import PageCleanup
        assert dialog.shutdown().status is PageCleanup.CLEAN
        dialog.close()


def test_a_horizontal_twoc_scan_is_staged_and_resumed_as_nu(tmp_path, monkeypatch):
    from xdart.gui.widgets import file_dialogs
    from xrd_tools.core.geometry.stacked_arm import is_single_arm
    from xrd_tools.session.instrument import InstrumentGeometry
    spec = tmp_path / 'powder'
    spec.write_text('#F powder\n#E 1\n#D Thu Jan 1 00:00:00 1970\n#C test\n'
                    '#O0 tth th\n#S 1 ascan tth 10 30 2 1\n#D today\n#P0 10 5\n#N 1\n#L tth\n10\n20\n30\n')
    images = tmp_path / 'images'
    images.mkdir()
    for index in range(3):
        (images / f'powder_scan1_{index:04d}.raw').write_bytes(np.zeros((4, 6), '<i4').tobytes())
    source = SourceSpec(spec, SourceKind.SPEC, options={'scan': '1', 'image_dir': str(images),
        'image_stem': 'powder_scan1_', 'read_image_kwargs': {'detector_shape': (4, 6), 'raw_dtype': '<i4'}})
    dialog = StitchCalibrationDialog(tmp_path)
    other = StitchCalibrationDialog(tmp_path)
    try:
        dialog.set_instrument(InstrumentGeometry('twoc', plane='horizontal'))
        dialog.set_source_context(source)
        dialog._enumerate()
        _finish(dialog)
        assert dialog._enumerated_motors == (None, 'tth'), dialog.status.text()
        dialog._prepare()
        _finish(dialog)
        assert dialog._frame_motors == (None, 'tth')
        session = dialog.session_dir / 'calibration-session.json'
        assert json.loads(session.read_text())['arm'] == {'del': None, 'nu': 'tth'}
        assert 'holds one circle at zero' in dialog.arm_note.text()
        # the direct-beam default "del" means the arm's one circle: the rough geometry holds del
        dialog.rough_mode.setCurrentIndex(dialog.rough_mode.findData('manual'))
        for key, value in (('rough_distance', '500'), ('beam_pixel', '3, 2'), ('beam_detector', 'Pilatus300kw')):
            dialog._set_field_text(key, value)
        rough = dialog._rough_record()
        assert dialog._text('beam_motor') == 'del' and is_single_arm(rough) and rough['pos_names'] == ['nu_angle']
        # the same motor name, resumed in a window set to the vertical plane: the session keeps its plane
        monkeypatch.setattr(file_dialogs, 'browse', lambda *args, **kwargs: [str(session)])
        other.set_instrument(InstrumentGeometry('twoc'))
        other._resume()
        assert other._frame_motors == (None, 'tth')
        other._fit()
        assert ('staged reading tth as a horizontal 2θ arm; the Geometry now names tth as a vertical 2θ arm'
                in other.status.text())
    finally:
        from xdart.gui.pages.values import PageCleanup
        for window in (dialog, other):
            assert window.shutdown().status is PageCleanup.CLEAN
            window.close()


def test_reading_metadata_takes_each_circles_motor_from_the_scan(tmp_path):
    from xdart.gui.tabs.stitch.calibration import _dispatch_job
    from xrd_tools.session.instrument import InstrumentGeometry
    spec = tmp_path / 'beam'
    spec.write_text('#F beam\n#E 1\n#D Thu Jan 1 00:00:00 1970\n#C test\n'
                    '#O0 delta nu\n#S 1 ascan delta 1 3 2 1\n#D today\n#P0 0 4\n#N 2\n#L delta  nu\n1 4\n2 5\n3 6\n')
    images = tmp_path / 'images'
    images.mkdir()
    for index in range(3):
        (images / f'beam_scan1_{index:04d}.raw').write_bytes(np.zeros((4, 6), '<i4').tobytes())
    source = SourceSpec(spec, SourceKind.SPEC, options={'scan': '1', 'image_dir': str(images),
        'read_image_kwargs': {'detector_shape': (4, 6), 'raw_dtype': '<i4'}})
    rows, names, record = _dispatch_job(('enumerate', (source, InstrumentGeometry().to_record())),
                                        Event(), lambda *_: None)
    assert 'delta' in names and InstrumentGeometry.from_record(record).stitch_arm() == ('delta', 'nu')
    assert rows[0][1] == {'delta': 1.0, 'nu': 4.0}


def test_staged_frames_survive_pyfai_ring_extraction(tmp_path):
    # pyFAI's ring extractor converts to float32 without copying float32 input and fills
    # the detector gaps in place; a float64 staged frame keeps the image xdart checks.
    from pyFAI.detectors import detector_factory
    from pyFAI.gui.peak_picker import PeakPicker
    from pyFAI.io import image as pyfai_image
    from xrd_tools.integrate.stitch_calibration import calibration_image_identity, capture_calibration_frames
    spec = tmp_path / 'beam'
    spec.write_text('#F beam\n#E 1\n#D Thu Jan 1 00:00:00 1970\n#C test\n'
                    '#O0 del nu\n#S 1 ascan del 1 2 1 1\n#D today\n#P0 0 4\n#N 2\n#L del  nu\n1 4\n2 5\n')
    images = tmp_path / 'images'; images.mkdir()
    counts = np.arange(195 * 1475, dtype='<i4').reshape(195, 1475) % 1000
    counts[:, 487:494] = -1  # the module gap, as a Pilatus writes it
    for index in range(2):
        (images / f'beam_scan1_{index:04d}.raw').write_bytes(counts.tobytes())
    source = SourceSpec(spec, SourceKind.SPEC, options={'scan': '1', 'image_dir': str(images),
        'read_image_kwargs': {'detector_shape': (195, 1475), 'raw_dtype': '<i4'}})
    staged, = capture_calibration_frames(source, (0,), tmp_path / 'session')
    image = pyfai_image.read_image_data(staged.image_path)  # what pyFAI-calib2 loads
    before = calibration_image_identity(image)
    PeakPicker(data=image, mask=None, detector=detector_factory('Pilatus300kw'))
    assert calibration_image_identity(image) == before


def test_direct_beam_window_handoff_uses_actual_api_and_seed_not_bounds(tmp_path):
    from tests.core.test_direct_beam import beam_images, config
    from xdart.gui.tabs.stitch.calibration import _run_job
    from xrd_tools.integrate.direct_beam import calibrate_direct_beam
    angles = np.array([5.5,6.,6.5,7.,7.5,8.,8.5])
    images, _, beam = beam_images(angles)
    spec = tmp_path/'direct'
    spec.write_text('#F direct\n#E 1\n#D today\n#C test\n#O0 del\n'
        '#S 1 ascan del 5.5 8.5 6 1\n#D today\n#P0 0\n#N 1\n#L del\n' + '\n'.join(map(str, angles)) + '\n')
    for index, array in enumerate(images):
        tifffile.imwrite(tmp_path/f'direct_scan1_{index:04d}.tif', array)
    source = SourceSpec(spec, SourceKind.SPEC, options={'scan':'1','image_dir':str(tmp_path)})
    options = dict(detector='Detector', detector_config=config(), beam_pixel=beam,
                   angle_zero_deg=7., rotation=0)
    result, energy, held = _run_job(('distance', (source, 'del', options, ())), Event(), lambda *_: None)
    assert energy is None and held == {}  # the scan records no energy; no other circle is held
    expected = calibrate_direct_beam(angles, images, detector='Detector', detector_config=config(),
                                     beam_pixel=beam, angle_zero_deg=7.)
    assert result.distance_m == pytest.approx(expected.distance_m)
    assert result.point_ids == tuple(('1.1', index) for index in range(7))
    dialog = StitchCalibrationDialog(tmp_path)
    for key, value in (('beam_source', spec), ('beam_scan', '1'), ('beam_images', tmp_path)):
        dialog._set_field_text(key, str(value))
    # as the window's own job records it: the result and the inputs it was fitted from (Codex F5)
    dialog._distance_result, dialog._distance_key = result, dialog._direct_beam_key()
    dialog._apply_distance()  # no confirmation: the result applies
    assert float(dialog._text('distance_seed')) == pytest.approx(80., rel=2e-5)
    assert dialog._text('distance_bounds') == ''
    assert dialog._distance_provenance['regression_std_m'] >= 0
    assert 'residuals_px' in dialog._distance_provenance
    dialog.close()


@pytest.mark.parametrize('source_kind', ['spec_container', 'nexus'])
def test_direct_beam_fit_button_reads_container_sources(tmp_path, source_kind):
    import h5py
    from pyqtgraph.Qt import QtWidgets, QtTest
    from tests.core.test_direct_beam import beam_images
    angles = np.array([1., .5, 0., -.5, -1.])
    images, _, beam = beam_images(angles, shape=(65, 487),
        pitches=(516e-6, 172e-6), distance=.5, zero=0)
    spec = tmp_path / 'direct.spec'
    spec.write_text('#F direct\n#O0 eta\n#S 8 ascan del 1 -1 4 1\n'
        '#P0 0\n#N 1\n#L del\n' + '\n'.join(map(str, angles)) + '\n')
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
    dialog = StitchCalibrationDialog(tmp_path)
    dialog.beam_source_kind.setCurrentIndex(dialog.beam_source_kind.findData(source_kind))
    # the NeXus case leaves the beam pixel blank: it is found on the frame at the zero
    typed = ', '.join(map(str, beam)) if source_kind == 'spec_container' else ''
    for key, value in dict(beam_source=spec if source_kind == 'spec_container' else nexus,
            beam_image_file=master, beam_scan='8.1', beam_entry='measurement',
            beam_detector='Pilatus100k', beam_pixel=typed).items():
        dialog._set_field_text(key, value)
    assert dialog._fields['beam_pixel'].placeholderText().startswith('auto · where the direct beam hits')
    assert dialog._fields['beam_scan'].isHidden() == (source_kind == 'nexus')
    assert dialog._fields['beam_images'].parentWidget().isHidden()
    dialog.beam_button.click()
    for _ in range(100):
        QtWidgets.QApplication.processEvents()
        if not dialog.active():
            break
        QtTest.QTest.qWait(20)
    assert not dialog.active()
    result = dialog._distance_result
    assert result is not None, dialog.status.text()
    path = master if source_kind == 'spec_container' else nexus
    assert result.frame_ids == tuple((path, i) for i in range(5))
    assert result.point_ids == (tuple(('8.1', i) for i in range(5))
                               if source_kind == 'spec_container' else result.frame_ids)
    assert result.distance_mm == pytest.approx(500., rel=1e-4)
    assert result.used.all()
    assert result.beam_pixel == pytest.approx(beam, abs=.1)
    if not typed:
        assert 'found on the frame nearest the zero' in dialog.beam_summary.toPlainText()
        assert dialog._fields['beam_pixel'].placeholderText().startswith('auto · the direct beam found at 243.0, 32.0')
    assert dialog._text('distance_seed') == ''
    dialog.apply_distance_button.click()
    assert float(dialog._text('distance_seed')) == pytest.approx(500., rel=1e-4)
    assert dialog._text('distance_bounds') == ''
    # The saved fit provenance must remain JSON serializable for NeXus path IDs.
    json.dumps(dialog._distance_provenance)
    dialog.close()


def _fit_direct_beam(dialog):
    from pyqtgraph.Qt import QtWidgets, QtTest
    dialog.beam_button.click()
    for _ in range(1500):
        QtWidgets.QApplication.processEvents()
        if not dialog.active():
            break
        QtTest.QTest.qWait(20)
    assert not dialog.active(), dialog.status.text()


def test_a_direct_beam_fit_shows_its_distance_and_the_scan_supplies_the_wavelength(tmp_path):
    """Maintainer, 2026-09-29: "for Direct beam distance fit, it will be good to pop up a window with the
    fit that also shows the fit distance. also the fit distance should show in the output message box.
    also why the message to type in wavelength to save calibration. why doesn't it get it from the spec
    file?" A SPEC image series recording the energy: the fit opens its window with the distance, the
    status line says it, and with no calibrant scan read the wavelength is the direct-beam scan's."""
    from tests.core.test_direct_beam import beam_images
    from xrd_tools.core.energy import energy_eV_to_wavelength_m
    angles = np.array([1., .5, 0., -.5, -1.])
    images, _, beam = beam_images(angles, shape=(65, 487), pitches=(516e-6, 172e-6), distance=.5, zero=0)
    spec = tmp_path / 'direct'
    spec.write_text('#F direct\n#E 1\n#D today\n#C test\n#O0 del  nu  energy\n#S 1 ascan del 1 -1 4 1\n'
                    '#D today\n#P0 0 0 17000.018\n#N 1\n#L del\n' + '\n'.join(map(str, angles)) + '\n')
    for index, array in enumerate(images):
        tifffile.imwrite(tmp_path / f'direct_scan1_{index:04d}.tif', array)
    dialog = StitchCalibrationDialog(tmp_path)
    try:
        dialog.show()  # as the page opens it
        dialog.beam_source_kind.setCurrentIndex(dialog.beam_source_kind.findData('spec_series'))
        for key, value in dict(beam_source=spec, beam_scan='1', beam_images=tmp_path,
                               beam_detector='Pilatus100k', beam_pixel=', '.join(map(str, beam))).items():
            dialog._set_field_text(key, str(value))
        assert dialog._rough_wavelength_A() is None and not dialog.show_fit_button.isEnabled()
        dialog.geometry_editor.zero_edit('nu').setText('1.5')  # nu held at 0 during the scan, off its zero
        _fit_direct_beam(dialog)
        result = dialog._distance_result
        assert result is not None, dialog.status.text()
        assert dialog.status.text().startswith(f'Direct-beam distance: {result.distance_mm:.6g} mm (regression SE')
        assert dialog.status.text().endswith('nu held at 0° · the scan records 17000 eV')
        assert dialog._distance_held == {'nu': 0.}
        # the rough arm puts the beam on the fitted pixel where the scan saw it (del at its zero, nu at 0),
        # not at the zero pose (nu at 1.5): the exact angles are the core row's
        from xrd_tools.integrate.calibration_seed import _direct_beam_hit
        seed = dialog._rough_arm()
        x, y = result.zero_beam_pixel
        (row, column), _ = _direct_beam_hit(seed, del_motor='del', nu_motor='nu', readings={'del': 0., 'nu': 0.})
        assert (row, column) == pytest.approx((y, x), abs=1e-6)
        (row, column), _ = _direct_beam_hit(seed, del_motor='del', nu_motor='nu', readings={'del': 0., 'nu': 1.5})
        assert abs(column - x) > 10
        assert dialog.beam_summary.toPlainText().startswith('Distance: ')
        window = dialog._direct_beam_fit_window
        assert window is not None and window.isVisible() and dialog.show_fit_button.isEnabled()
        assert window.summary.text().startswith(f'Direct-beam distance: {result.distance_mm:.6g} mm')
        assert 'of 5 frames used' in window.summary.text()
        wavelength = float(energy_eV_to_wavelength_m(17000.018))
        assert dialog._rough_wavelength_A() == pytest.approx(wavelength * 1e10, rel=1e-12, abs=0)
        fit = dialog._rough_fit()  # Save as calibration no longer asks for it
        assert fit.asset['calibration']['wavelength'] == pytest.approx(wavelength, rel=1e-12, abs=0)
        dialog._set_field_text('beam_scan', '2')  # the fit no longer matches its inputs: typed again
        assert dialog._rough_wavelength_A() is None
        dialog.hide()  # the fit window goes with the calibration window
        assert not window.isVisible()
    finally:
        dialog.close()


def test_return_in_a_box_commits_it_and_presses_no_button(tmp_path):
    """Return or Enter in any box of the calibration window pressed its default button, "Fit direct-beam
    distance", and started a job; now it only commits the box."""
    from pyqtgraph.Qt import QtCore, QtWidgets, QtTest
    dialog = StitchCalibrationDialog(tmp_path)
    try:
        dialog.show()
        clicked = []
        for button in dialog.findChildren(QtWidgets.QPushButton):
            button.clicked.connect(lambda *_a, text=button.text(): clicked.append(text))
        edit = dialog._fields['beam_pixel']
        committed = []
        edit.editingFinished.connect(lambda: committed.append(edit.text()))
        edit.setFocus()
        QtTest.QTest.keyClicks(edit, '280, 97')
        for key in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            QtTest.QTest.keyClick(edit, key)
            QtWidgets.QApplication.processEvents()
        assert clicked == [] and not dialog.active() and dialog.isVisible()
        assert committed and committed[0] == '280, 97'
    finally:
        dialog.close()


def test_direct_beam_container_job_refuses_count_mismatch_and_honours_cancel(tmp_path):
    import h5py
    from xdart.gui.pages.operation_owner import OperationCancelled
    from xdart.gui.tabs.stitch.calibration import _run_job
    spec = tmp_path / 'direct.spec'
    spec.write_text('#F direct\n#O0 eta\n#S 8 ascan del 1 -1 4 1\n'
        '#P0 0\n#N 1\n#L del\n1\n.5\n0\n-.5\n-1\n')
    nexus = tmp_path / 'direct.nxs'
    with h5py.File(nexus, 'w') as handle:
        handle.create_group('entry').attrs['NX_class'] = 'NXentry'
        handle['entry/data/image'] = np.zeros((4, 65, 487), dtype='uint32')
    source = SourceSpec(spec, SourceKind.SPEC,
                        options={'scan': '8.1', 'image_file': str(nexus)})
    options = dict(detector='Pilatus100k', beam_pixel=(243, 32), angle_zero_deg=0., rotation=0)
    with pytest.raises(ValueError, match='count mismatch'):
        _run_job(('distance', (source, 'del', options, ())), Event(), lambda *_: None)
    cancelled = Event(); cancelled.set()
    with pytest.raises(OperationCancelled):
        _run_job(('distance', (source, 'del', options, ())), cancelled, lambda *_: None)


def test_the_detector_and_its_orientation_are_choices_saved_with_the_settings(tmp_path):
    from pyqtgraph.Qt import QtWidgets
    dialog = StitchCalibrationDialog(tmp_path)
    again = StitchCalibrationDialog(tmp_path)
    try:
        detector, orientation = dialog._fields['beam_detector'], dialog._fields['beam_orientation']
        assert isinstance(detector, QtWidgets.QComboBox) and isinstance(orientation, QtWidgets.QComboBox)
        assert [detector.itemData(row) for row in range(3)] == ['Pilatus100k', 'Pilatus300kw', 'Eiger1M']
        assert dialog._text('beam_detector') == 'Pilatus300kw' and dialog._text('beam_orientation') == '3'
        # the image the beam pixel is read in: width (x, columns) × height (y, rows)
        assert dialog._image_size.text() == '1475 × 195 px (x × y): x 0–1474, y 0–194 · image rotation 0°'
        # the hint does not suggest the centre: the beam need not be there (the 17-2 nu scan's is at 279, 97)
        assert dialog._row_visible('beam_size') and 'direct beam' in dialog._fields['beam_pixel'].placeholderText()
        # the first-pixel corner is not a rotation: it never swaps x and y
        assert [orientation.itemText(row).split(' · ')[1].startswith('first pixel') for row in range(4)] == [True] * 4
        dialog._set_field_text('beam_orientation', '2')
        assert dialog._image_size.text().startswith('1475 × 195 px')
        dialog._set_field_text('beam_orientation', '3')
        dialog._set_field_text('beam_detector', 'Pilatus100k')
        assert dialog._text('shape') == '195, 487'  # the RAW reader follows the detector
        assert dialog._image_size.text().startswith('487 × 195 px')
        dialog.rotation.setCurrentIndex(dialog.rotation.findData(90))  # a turned image swaps x and y
        assert dialog._image_size.text().startswith('195 × 487 px') and 'image rotation 90°' in dialog._image_size.text()
        dialog.rotation.setCurrentIndex(dialog.rotation.findData(0))
        dialog._set_field_text('beam_orientation', '2')
        for key, value in (('rough_distance', '400'), ('beam_pixel', '240, 97')):
            dialog._set_field_text(key, value)
        dialog.rough_mode.setCurrentIndex(dialog.rough_mode.findData('manual'))
        assert dialog._rough_record()['detector_config'] == {'orientation': 2}
        again.apply_settings(dialog.settings())
        assert (again._text('beam_detector'), again._text('beam_orientation')) == ('Pilatus100k', '2')
        assert again._text('shape') == '195, 487'
        # an older window's direct-beam mask and saturation and the refinement's count limit are now
        # the Detector panel's saturation and mask
        again.apply_settings({'fields': {'beam_mask': str(tmp_path / 'm.edf'), 'beam_saturation': '',
                                         'refine_max_counts': '900000'}})
        assert (again._text('mask'), again._text('saturation')) == (str(tmp_path / 'm.edf'), '900000')
        saved = dialog.settings()
        saved['fields']['beam_detector'] = 'SomeFutureDetector'  # a name this pyFAI does not list
        again.apply_settings(saved)
        assert again._text('beam_detector') == 'SomeFutureDetector'  # kept, not replaced
    finally:
        dialog.close()
        again.close()


def test_after_one_calibrated_frame_the_others_are_predicted_from_it(tmp_path):
    """The rough geometry is 200 px off; once frame 0 is calibrated its arm predicts the rest,
    in the table and in the seed pyFAI opens with."""
    import time
    from pyFAI.io.ponifile import PoniFile
    from pyqtgraph.Qt import QtWidgets
    from xdart.gui.tabs.stitch.calibration import _pyfai_arguments
    from tests.core.test_stacked_arm_extension import _calibration, _ring_controls
    from tests.core.test_stitch_guided_calibration import _frames
    from xrd_tools.core.geometry.stacked_arm import StackedDetectorArm
    from xrd_tools.integrate.calibration_seed import predict_frame_seed, ring_summary
    truth = {'nu_offset': .02, 'del_offset': -.05}
    controls, _calibrant = _ring_controls(truth, _calibration(), [(5., -10.), (45., -10.), (25., 9.5), (5., 29.)])
    (tmp_path / 'frames').mkdir()
    calibrated = _frames(tmp_path / 'frames', controls, _calibration().poni)
    waiting = tuple(replace(frame, poni_path='', poni_sha256='', npt_path='', npt_sha256='', pairing_policy='',
                            control_image_sha256='') for frame in calibrated[1:])
    true_arm = StackedDetectorArm(**truth).to_pyfai_goniometer(_calibration())
    dialog = StitchCalibrationDialog(tmp_path)
    try:
        # the truth's motor zeros (del 2.86°, nu −1.15°), typed beside the motors: every fit starts there
        dialog.geometry_editor.zero_edit('del').setText(f'{np.rad2deg(.05):.12g}')
        dialog.geometry_editor.zero_edit('nu').setText(f'{-np.rad2deg(.02):.12g}')
        dialog.rough_mode.setCurrentIndex(dialog.rough_mode.findData('manual'))
        for key, value in (('rough_distance', '390'), ('beam_pixel', f'{.045 / 172e-6 + 200:.1f}, {.035 / 172e-6:.1f}')):
            dialog._set_field_text(key, value)
        dialog.frames = (calibrated[0], *waiting)
        dialog._render_frames()
        expected = [ring_summary(predict_frame_seed(true_arm, frame).rings) for frame in waiting]
        assert dialog.frame_table.item(1, 5).text() != expected[0]  # the rough guess misnumbers them
        dialog._start('predict', (dialog._calibrated_key(), dialog._seed_options()))
        until = time.monotonic() + 60
        while dialog.active() and time.monotonic() < until:  # sleep, as the app's event loop idles:
            QtWidgets.QApplication.processEvents()        # qWait holds the GIL from the worker
            time.sleep(.005)
        assert not dialog.active() and dialog._prediction is not None, dialog.status.text()
        assert [dialog.frame_table.item(row, 5).text() for row in (1, 2, 3)] == expected
        seeded = PoniFile(_pyfai_arguments(waiting[0], dialog._seed_options())[3])
        true_seed = predict_frame_seed(true_arm, waiting[0]).integrator
        assert (seeded.poni1, seeded.poni2) == pytest.approx((true_seed.poni1, true_seed.poni2), abs=1e-3)
        dialog._prediction = (('another frame', ''), dialog._prediction[1])  # fitted to other frames: unused
        assert dialog._seed_options()['prediction'] is None
    finally:
        dialog.close()


def test_a_cancel_during_the_prediction_keeps_the_frame_pyfai_saved(tmp_path):
    """The prediction runs after the frame's PONI and points are adopted: a cancel there drops
    only the prediction, never the calibrated frame (Codex round 16)."""
    from xdart.gui.tabs.stitch.calibration import _run_job
    from tests.core.test_stacked_arm_extension import _calibration, _ring_controls
    from tests.core.test_stitch_guided_calibration import _frames
    (tmp_path / 'frames').mkdir()
    controls, _calibrant = _ring_controls({'nu_offset': .02, 'del_offset': -.05}, _calibration(),
                                          [(5., -10.), (45., -10.)])
    calibrated = _frames(tmp_path / 'frames', controls, _calibration().poni)
    waiting = replace(calibrated[0], poni_path='', poni_sha256='', npt_path='', npt_sha256='', pairing_policy='',
                      control_image_sha256='')
    seed = {'frames': (), 'del_motor': 'del', 'nu_motor': 'nu'}
    request = ('import', (waiting, calibrated[0].poni_path, calibrated[0].npt_path, seed))
    cancelled = Event()
    cancelled.set()
    adopted, prediction = _run_job(request, cancelled, lambda *_: True)
    assert adopted.complete and prediction is None
    adopted, prediction = _run_job(request, Event(), lambda *_: True)
    assert adopted.complete and prediction is not None


def test_the_detector_mask_reaches_pyfai_turned_as_the_frames_are(tmp_path):
    import fabio
    from xdart.gui.tabs.stitch.calibration import _pyfai_arguments
    from tests.core.test_calibration_seed import _frame
    mask = np.zeros((195, 1475), bool)
    mask[:10, :20] = True  # drawn on the stored image
    np.save(tmp_path / 'mask.npy', mask)
    dialog = StitchCalibrationDialog(tmp_path)
    try:
        dialog._fields['mask'].setText(str(tmp_path / 'mask.npy'))
        dialog.rotation.setCurrentIndex(dialog.rotation.findData(90))
        path = dialog._pyfai_mask()
        np.testing.assert_array_equal(fabio.open(path).data.astype(bool), np.rot90(mask))  # 90° counterclockwise
        np.testing.assert_array_equal(dialog._loaded_mask(), np.rot90(mask))  # the expected rings' mask too
        seed = {**dialog._seed_options(), 'mask': path}
        arguments = _pyfai_arguments(_frame({'del': 20., 'nu': 0.}, shape=(1475, 195)), seed)
        assert arguments[arguments.index('-m') + 1] == path
        dialog._fields['mask'].setText(str(tmp_path / 'missing.npy'))
        with pytest.raises(ValueError, match="the Detector panel's mask cannot be read"):
            dialog._pyfai_mask()
    finally:
        dialog.close()


def test_the_rough_geometry_puts_the_beam_where_the_fit_says_it_is_at_the_zero(tmp_path):
    """Codex review F1: with no frame at the zero, the spot found on the nearest frame only places the
    strips; the rough geometry (ring prediction, Save as calibration, the refinement seed) takes the
    fitted zero-angle beam."""
    import h5py
    from pyqtgraph.Qt import QtWidgets, QtTest
    from tests.core.test_calibration_seed import _frame
    from tests.core.test_direct_beam import beam_images
    from xrd_tools.integrate.calibration_seed import predict_frame_seed
    angles = np.array([1., .6, -.6, -1.])  # the zero is 0°, and no frame is at it
    images, _, beam = beam_images(angles, shape=(65, 487), pitches=(516e-6, 172e-6), distance=.5, zero=0)
    nexus = tmp_path / 'direct.nxs'
    with h5py.File(nexus, 'w') as handle:
        handle.create_group('measurement').attrs['NX_class'] = 'NXentry'
        handle['measurement/data/image'] = np.asarray(images).astype('uint32')
        handle['measurement/data/del'] = angles
    dialog = StitchCalibrationDialog(tmp_path)
    try:
        dialog.beam_source_kind.setCurrentIndex(dialog.beam_source_kind.findData('nexus'))
        for key, value in dict(beam_source=nexus, beam_entry='measurement', beam_detector='Pilatus100k',
                               beam_pixel='', rough_wavelength='1').items():
            dialog._set_field_text(key, value)
        dialog.beam_button.click()
        for _ in range(1500):
            QtWidgets.QApplication.processEvents()
            if not dialog.active():
                break
            QtTest.QTest.qWait(20)
        QtTest.QTest.qWait(200)  # the poll delivers the terminal
        result = dialog._distance_result
        assert result is not None, dialog.status.text()
        assert abs(result.beam_pixel[0] - beam[0]) > 1  # found on a frame off the zero
        assert result.zero_beam_pixel == pytest.approx(beam, abs=.05)
        assert 'Beam pixel at the zero' in dialog.beam_summary.toPlainText()
        seed = predict_frame_seed(dialog._rough_arm(), _frame({'del': 0., 'nu': 0.}, shape=(195, 487)))
        x, y = beam
        assert float(seed.integrator.tth(np.array([y]), np.array([x]))[0]) == pytest.approx(0, abs=1e-4)
    finally:
        dialog.close()
