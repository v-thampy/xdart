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


@pytest.mark.parametrize('action', ['save', 'switched_image', 'close', 'missing_points', 'failed_save', 'child_failure'])
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
                target = scratch / 'saved.poni'
                if action == 'failed_save':
                    from pyFAI.gui.dialog import MessageBox
                    MessageBox.exception = lambda *args, **kwargs: None
                    target = scratch / 'missing' / 'saved.poni'
                def choose(dialog):
                    dialog.selectFile(str(target)); return 1
                qt.QFileDialog.exec = choose
                task._savePoniButton.click()
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
    request = replace(prepare_calibration_request(str(image)), points_directory=str(tmp_path))
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
        if action == 'save':
            result = terminal.payload
            assert result.last_saved_path == str(hooks/'saved.poni')
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
        else:
            assert terminal.payload.last_saved_path is None
            assert terminal.payload.control_points_path is None
    assert not list(tmp_path.glob('.xdart-calibration-*'))


def test_modeless_window_exposes_guided_source_controls_and_preserves_session(tmp_path):
    dialog = StitchCalibrationDialog(tmp_path, backend='xu_hist')
    source = SourceSpec(tmp_path/'beam', SourceKind.SPEC, options={'scan':'14', 'image_dir':str(tmp_path/'images')})
    dialog.set_source_context(source, image_rotation=180, del_motor='delta', nu_motor='gamma')
    assert not dialog.isModal()
    assert dialog.tabs.count() == 4
    assert dialog._spec().options['scan'] == '14'
    assert dialog.rotation.currentData() == 180
    assert dialog.backend.currentData() == 'xu_hist'
    assert dialog._text('del_motor') == 'delta'
    previous = dialog.session_dir
    dialog._new_session()
    assert dialog.session_dir != previous
    assert not dialog.active()
    from xdart.gui.pages.values import PageCleanup
    assert dialog.shutdown().status is PageCleanup.CLEAN
    dialog.close()


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
    result = _run_job(('distance', (source,'del',options)), Event(), lambda *_:None)
    expected = calibrate_direct_beam(angles, images, detector='Detector', detector_config=config(),
                                     beam_pixel=beam, angle_zero_deg=7.)
    assert result.distance_m == pytest.approx(expected.distance_m)
    assert result.point_ids == tuple(range(7))
    dialog = StitchCalibrationDialog(tmp_path)
    dialog._distance_result = result
    dialog._apply_distance()
    assert dialog._text('distance_seed') == ''
    dialog.compatible.setChecked(True)
    dialog._apply_distance()
    assert float(dialog._text('distance_seed')) == pytest.approx(80., rel=2e-5)
    assert dialog._text('distance_bounds') == ''
    assert dialog._distance_provenance['regression_std_m'] >= 0
    assert 'residuals_px' in dialog._distance_provenance
    dialog.close()
