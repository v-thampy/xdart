"""Guided calibration uses matched real files and the released numerical owners."""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest
import tifffile

from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.integrate.stitch_calibration import (
    CalibrationFrame, adopt_calibration_pair, capture_calibration_frames,
    fit_calibration, load_calibration_session, load_control_frame,
    save_calibration_fit, suggested_frame_indices,
    save_calibration_session, validate_frame_files, calibration_image_identity,
)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frames(tmp_path, controls, poni, shape=(195, 1475)):
    from pyFAI.control_points import ControlPoints
    from pyFAI.calibrant import get_calibrant
    from pyFAI.detectors import detector_factory
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    frames = []
    for index, control in enumerate(controls):
        directory = tmp_path / str(index)
        directory.mkdir()
        image = directory / 'frame.tif'
        tifffile.imwrite(image, np.zeros(shape, dtype=np.float32))
        points = ControlPoints(calibrant=get_calibrant('LaB6'), wavelength=poni.wavelength)
        for ring in np.unique(control.rings):
            selected = control.rings == ring
            points.append(points=np.column_stack([control.rows[selected], control.cols[selected]]).tolist(), ring=int(ring))
        npt = directory / 'frame.npt'
        points.save(str(npt))
        seed = directory / 'frame.poni'
        AzimuthalIntegrator(dist=poni.dist, poni1=poni.poni1, poni2=poni.poni2,
            rot1=poni.rot1, rot2=poni.rot2, rot3=poni.rot3, wavelength=poni.wavelength,
            detector=detector_factory('Pilatus300kw', config={'orientation':3})).save(str(seed))
        frame = CalibrationFrame(str(index), str(image), 'image_file', {}, index,
            str(image), 0, str(image), _digest(image), shape, 0, dict(control.motors), None)
        frames.append(adopt_calibration_pair(frame, seed, npt))
    return tuple(frames)


def test_exact_raw_staging_rotation_angles_and_resume(tmp_path):
    spec = tmp_path / 'beam'
    spec.write_text('#F beam\n#E 1\n#D Thu Jan 1 00:00:00 1970\n#C test\n'
                    '#O0 del nu\n#S 1 ascan del 1 3 2 1\n#D today\n#P0 0 4\n#N 2\n#L del  nu\n1 4\n2 5\n3 6\n')
    images = tmp_path / 'images'; images.mkdir()
    original = np.arange(24, dtype='<i4').reshape(4, 6)
    for index in range(3):
        (images / f'beam_scan1_{index:04d}.raw').write_bytes((original + index * 100).tobytes())
    source = SourceSpec(spec, SourceKind.SPEC, options={'scan': '1', 'image_dir': str(images),
        'read_image_kwargs': {'detector_shape': (4, 6), 'raw_dtype': '<i4'}})
    frames = capture_calibration_frames(source, (2, 0), tmp_path / 'session', rotation=180)
    assert [f.index for f in frames] == [2, 0]
    assert [f.motors for f in frames] == [{'del': 3., 'nu': 6.}, {'del': 1., 'nu': 4.}]
    assert np.array_equal(tifffile.imread(frames[0].image_path), np.rot90(original + 200, 2))
    assert load_calibration_session(tmp_path / 'session/calibration-session.json') == frames
    with pytest.raises(ValueError, match='existing SPEC'):
        capture_calibration_frames(replace(source, options={**source.options, 'scan': '999'}), (0,), tmp_path / 'invalid')
    assert len(suggested_frame_indices([(f.index, f.motors, None) for f in frames])) == 2


def test_resumed_control_points_require_retained_image_binding(tmp_path):
    from xrd_tools.core.containers import PONI
    from xrd_tools.integrate.refine import ControlFrame
    control = ControlFrame([1,2], [3,4], [0,0], {'del':10.,'nu':2.})
    frame, = _frames(tmp_path, (control,), PONI(dist=.39, poni1=.01, poni2=.02,
        wavelength=1e-10, detector='Pilatus300kw'))
    session = save_calibration_session(tmp_path, (frame,))
    assert load_calibration_session(session)[0].complete
    value = json.loads(session.read_text())
    value['frames'][0]['control_image_sha256'] = '0' * 64
    session.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='different calibration image'):
        load_calibration_session(session)
    # Older sessions preserve their files but cannot inherit automatic binding.
    value['version'] = 1
    value['frames'][0].pop('control_image_sha256')
    value['frames'][0].pop('pairing_policy')
    session.write_text(json.dumps(value))
    old, = load_calibration_session(session)
    assert not old.complete
    assert old.poni_path == frame.poni_path and old.npt_path == frame.npt_path
    with pytest.raises(ValueError, match='matched PONI'):
        validate_frame_files(old)
    with pytest.raises(ValueError, match='loaded pyFAI image differs'):
        adopt_calibration_pair(old, frame.poni_path, frame.npt_path,
                               model_image_sha256=calibration_image_identity(np.ones(frame.shape))[1])


def test_mg_fit_saved_json_reloads_actual_importer_and_bad_inputs_refuse(tmp_path):
    from tests.core.test_refine import _truth, _synth_control_frames, _FRAMES
    from xrd_tools.core.geometry import Diffractometer
    from pyFAI.goniometer import Goniometer
    truth = _truth()
    controls = _synth_control_frames(truth, _FRAMES, per_ring=8)
    frames = _frames(tmp_path, controls, truth.calibration.poni)
    result = fit_calibration(frames, distance_bounds_m=(.37, .41))
    path = save_calibration_fit(result, tmp_path / 'fitted.json')
    loaded = Diffractometer.from_pyfai_goniometer(path,
        source_motors={'nu_angle': 'nu', 'del_angle': 'del'})
    assert .37 <= loaded.calibration.poni.dist <= .41
    assert result.rms_q < .006
    actual = Goniometer.sload(path)
    assert .37 <= actual.get_ai((3., 15.)).dist <= .41
    from xrd_tools.analysis.stitch_operation import (
        StitchGeometryInput, StitchGeometryKind, capture_stitch_geometry,
    )
    receipt = capture_stitch_geometry(StitchGeometryInput(path, StitchGeometryKind.PYFAI_GONIOMETER_JSON,
        source_motors=(('del_angle','del'),('nu_angle','nu'))))
    assert json.loads(receipt.content)['detector_config'] == {'orientation':3}
    assert json.loads((tmp_path / 'fitted.json.provenance.json').read_text())['frames'][0]['index'] == 0
    with pytest.raises(FileExistsError):
        save_calibration_fit(result, path)
    with pytest.raises(ValueError, match='matched PONI'):
        load_control_frame(replace(frames[0], npt_path=''))
    with pytest.raises(ValueError, match='bounds'):
        fit_calibration(frames, distance_bounds_m=(.4, .3))
    with pytest.raises(ValueError, match='RMS'):
        fit_calibration(frames, max_rms_q=1e-8)
    with open(frames[0].npt_path, 'a') as stream:
        stream.write('\n# changed')
    with pytest.raises(ValueError, match='changed'):
        fit_calibration(frames)


def test_nexus_staging_uses_complete_metadata_provider_and_exact_frame(tmp_path):
    from tests.core.test_metadata_provider import _bluesky
    from xrd_tools.integrate.stitch_calibration import source_frame_choices
    from xrd_tools.sources import open_source
    path, hy, i0, _ = _bluesky(tmp_path / 'raw.nxs')
    spec = SourceSpec(path, SourceKind.NEXUS_STACK)
    choices = source_frame_choices(spec, ('hy', 'i0'))
    assert choices[2][1] == {'hy': float(hy[2]), 'i0': float(i0[2])}
    frames = capture_calibration_frames(spec, (2,), tmp_path/'staged', motor_names=('hy','i0'))
    source = open_source(spec)
    try:
        assert np.array_equal(tifffile.imread(frames[0].image_path), source.load_frame(2))
    finally:
        source.close()
    assert frames[0].raw_index == 2
    assert frames[0].source_options['dataset_path'] == '/entry/data/eiger_image'


def test_xu_notebook_fit_and_authored_admission_preserve_v1(tmp_path):
    from pyFAI.calibrant import get_calibrant
    from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
    from xrd_tools.core.containers import PONI
    from xrd_tools.integrate.refine import ControlFrame
    from xrd_tools.analysis.xu_stitch_calibration import (
        canonical_surface_resource_bytes, capture_xu_stitch_calibration,
        XuStitchCalibrationInput, parse_xu_stitch_calibration_bytes, XuStitchCalibrationRefused)
    calibration = json.loads(canonical_surface_resource_bytes())
    detector, acquisition = calibration['detector'], calibration['acquisition']
    wl = acquisition['xdart_wavelength_A'] * 1e-10
    lab6 = get_calibrant('LaB6'); lab6.set_wavelength(wl)
    q_ref = 2 * np.pi / np.asarray(lab6.dSpacing)
    controls = []
    with xu_runtime_session() as runtime:
        xu = runtime.xu
        qc = xu.QConversion(['x+', 'z-', 'y+', 'z-'], ['x+', 'z-'], [0, 1, 0])
        hx = xu.HXRD([0, 1, 0], [0, 0, 1], geometry='real', en=acquisition['energy_eV'], qconv=qc)
        hx.Ang2Q.init_area('x-', 'z+', cch1=detector['cch1_px'], cch2=detector['cch2_px'],
            Nch1=195, Nch2=1475, pwidth1=.172, pwidth2=.172, distance=detector['distance_m'] * 1000)
        for de, nu in [(5., -10.), (20., -5.), (30., 3.), (40., 9.5)]:
            qx, qy, qz = hx.Ang2Q.area(0., 0., 0., 0., nu-acquisition['detector_offsets_deg']['nu'],
                                    de-acquisition['detector_offsets_deg']['del'])
            q = np.hypot(np.hypot(qx, qy), qz)
            rows, cols, rings = [], [], []
            for ri, qr in enumerate(q_ref[:20]):
                yy, xx = np.where(np.abs(q-qr) < .001)
                select = np.linspace(0, max(0, len(yy)-1), min(12, len(yy))).astype(int)
                rows.extend(yy[select]); cols.extend(xx[select]); rings.extend([ri]*len(select))
            controls.append(ControlFrame(rows, cols, rings, {'del': de, 'nu': nu}))
    poni = PONI(dist=detector['distance_m'], poni1=detector['cch1_px']*.000172,
                poni2=detector['cch2_px']*.000172, detector='Pilatus300kw', wavelength=wl)
    frames = _frames(tmp_path, controls, poni)
    fit = fit_calibration(frames, backend='xu_hist', distance_bounds_m=(.38,.40),
                          domain={'del':[0.,50.], 'nu':[-15.,30.]})
    assert fit.rms_q < .002
    path = save_calibration_fit(fit, tmp_path/'xu.json')
    receipt = capture_xu_stitch_calibration(XuStitchCalibrationInput(path), project_root=tmp_path)
    assert receipt.projection.value['version'] == 2
    assert .38 <= receipt.projection.value['detector']['distance_m'] <= .40
    constrained = fit_calibration(frames, backend='xu_hist', distance_seed_m=.395,
        distance_bounds_m=(.390, .400), domain={'del':[0.,50.], 'nu':[-15.,30.]})
    assert .390 <= constrained.asset['calibration']['detector']['distance_m'] <= .400
    bad = json.loads(receipt.content)
    bad['validation']['geometry_rms_q_A^-1'] = .1
    with pytest.raises(XuStitchCalibrationRefused):
        parse_xu_stitch_calibration_bytes(json.dumps(bad, sort_keys=True, separators=(',', ':')).encode())
    original = canonical_surface_resource_bytes()
    changed = json.loads(original); changed['detector']['distance_m'] += .001
    with pytest.raises(XuStitchCalibrationRefused):
        parse_xu_stitch_calibration_bytes(json.dumps(changed, sort_keys=True, separators=(',', ':')).encode())
    assert parse_xu_stitch_calibration_bytes(original).content == original


def test_mg_distance_bounds_constrain_optimizer_not_only_initial_seed():
    from tests.core.test_refine import _truth, _synth_control_frames, _FRAMES
    from xrd_tools.integrate.refine import refine_goniometer
    truth = _truth()
    controls = _synth_control_frames(truth, _FRAMES, per_ring=8)
    distance = truth.calibration.poni.dist
    lower, upper = distance * 1.002, distance * 1.006
    result = refine_goniometer(truth.calibration.poni, controls, rot1_motor='nu', rot2_motor='del',
        detector_config={'orientation':3}, distance_seed_m=(lower+upper)/2,
        distance_bounds_m=(lower,upper))
    assert lower <= result.params['dist'] <= upper
    assert result.params['dist'] > distance


def test_calibration_refuses_same_count_missing_interior_raw_frame(tmp_path):
    from tests.core.test_spec_source import _spec
    from xrd_tools.integrate.stitch_calibration import source_frame_choices
    path = _spec(tmp_path)
    for index in (0, 2, 3):
        np.ones((2, 3), dtype=np.uint16).tofile(tmp_path / f'myscan_scan5_{index}.raw')
    spec = SourceSpec(path, SourceKind.SPEC, options={'scan': '5', 'image_dir': str(tmp_path),
        'read_image_kwargs': {'detector_shape': (2, 3), 'raw_dtype': 'uint16'}})
    with pytest.raises(ValueError, match='number'):
        source_frame_choices(spec, ('th', 'chi'))
