"""Del-on-nu calibration against explicit rotations and real powder controls."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from xrd_tools.core.containers import PONI
from xrd_tools.core.geometry import DetectorCalibration, Diffractometer
from xrd_tools.integrate.multi import create_multigeometry_integrators_from_geometry


def _r1(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1., 0, 0], [0, c, s], [0, -s, c]])


def _r2(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, -s], [0, 1., 0], [s, 0, c]])


def _r3(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def _model():
    from xrd_tools.core.geometry import StackedDetectorArm
    return StackedDetectorArm(nu_motor='pitch', del_motor='swing',
        nu_scale=np.deg2rad(1.006), del_scale=np.deg2rad(.997),
        nu_offset=.031, del_offset=-.043, beam_rotation=.017)


def _calibration():
    return DetectorCalibration(PONI(dist=.39, poni1=.028, poni2=.041,
        wavelength=.729e-10, detector='Pilatus300kw'), detector_config={'orientation': 3})


def _independent_stacked_controls():
    """Powder controls from physical pixel rays, without product Euler mapping."""
    from pyFAI.calibrant import get_calibrant
    from xrd_tools.integrate.refine import ControlFrame
    cal = _calibration()
    model = _model()
    rows, cols = np.indices((195, 1475))
    ray = np.stack(((rows + .5) * .000172 - cal.poni.poni1,
                    (cols + .5) * .000172 - cal.poni.poni2,
                    np.full(rows.shape, cal.poni.dist)))
    ring_q = 2 * np.pi / np.asarray(get_calibrant('LaB6').dspacing)
    rng = np.random.default_rng(12)
    controls = []
    for delta, nu in ((6., -10.), (15., 3.), (25., 6.), (40., 9.), (20., 16.)):
        rotation = (_r1(model.nu_offset + model.nu_scale * nu)
                    @ _r2(model.del_offset + model.del_scale * delta))
        lab = np.einsum('ij,jkl->ikl', rotation, ray)
        lab /= np.linalg.norm(lab, axis=0)
        lab[2] -= 1
        q = 2 * np.pi / (cal.poni.wavelength * 1e10) * np.linalg.norm(lab, axis=0)
        points = []
        for ring, target in enumerate(ring_q[:24]):
            ys, xs = np.where(abs(q - target) < .003)
            if len(ys) >= 6:
                indices = rng.choice(len(ys), min(12, len(ys)), replace=False)
                points.extend((y, x, ring) for y, x in zip(ys[indices], xs[indices]))
        y, x, rings = np.asarray(points).T
        controls.append(ControlFrame(y, x, rings, {'del': delta, 'nu': nu}))
    return controls, cal


def test_stacked_json_and_runtime_match_independent_matrix(tmp_path):
    from pyFAI.goniometer import Goniometer
    model, calibration = _model(), _calibration()
    asset = model.to_pyfai_goniometer(calibration)
    path = tmp_path / 'stacked.json'
    path.write_text(json.dumps(asset))
    gonio = Goniometer.sload(path)
    loaded = Diffractometer.from_pyfai_goniometer(path,
        source_motors={'del_angle': 'swing', 'nu_angle': 'pitch'})
    assert loaded.stacked_arm == model
    assert Diffractometer.from_json(loaded.to_json()) == loaded
    assert set(loaded.all_referenced_motors()) == {'swing', 'pitch'}
    for delta in (-45., -20., 0., 20., 45.):
        for nu in (-29., -10., 0., 10., 29.):
            expected = (_r3(model.beam_rotation) @ _r1(model.nu_offset + model.nu_scale * nu)
                        @ _r2(model.del_offset + model.del_scale * delta))
            ai, = create_multigeometry_integrators_from_geometry(loaded,
                {'swing': np.array([delta]), 'pitch': np.array([nu])})
            np.testing.assert_allclose(ai.rotation_matrix(), expected, atol=2e-14, rtol=0)
            direct = gonio.get_ai((delta, nu))
            np.testing.assert_allclose(direct.rotation_matrix(), expected, atol=2e-14, rtol=0)
            np.testing.assert_allclose(ai.qArray(), direct.qArray(), atol=2e-12, rtol=0)
    # A normal pyFAI save must retain the explicit model and its expressions.
    gonio.save(str(tmp_path / 'resaved.json'))
    assert Diffractometer.from_pyfai_goniometer(tmp_path / 'resaved.json',
        source_motors={'del_angle': 'swing', 'nu_angle': 'pitch'}).stacked_arm == model
    with pytest.raises(NotImplementedError, match='stacked'):
        loaded.to_diffractometer_geometry()


@pytest.mark.parametrize('motors', [
    {'swing': [1.]}, {'swing': [1., 2.], 'pitch': [2.]},
    {'swing': [1.], 'pitch': [np.nan]}, {'swing': [np.inf], 'pitch': [2.]},
])
def test_stacked_requires_finite_paired_motor_arrays(motors):
    loaded = Diffractometer(stacked_arm=_model(), calibration=_calibration())
    with pytest.raises((KeyError, ValueError)):
        create_multigeometry_integrators_from_geometry(loaded, motors)


def test_imported_stacked_q_and_chi_match_canonical_xu(tmp_path):
    from tests.core.test_stitch_two_angle_math import _receipt
    from xrd_tools.core.geometry import StackedDetectorArm
    from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
    from xrd_tools.integrate.xu_stitch import resolve_xu_stitch_effective_geometry
    receipt = _receipt(tmp_path, canonical=True)
    det = receipt.projection.value['detector']
    acq = receipt.projection.value['acquisition']
    offsets = acq['detector_offsets_deg']
    cal = DetectorCalibration(PONI(dist=det['distance_m'],
        poni1=(det['cch1_px'] + .5) * det['pixel1_m'],
        poni2=(det['cch2_px'] + .5) * det['pixel2_m'],
        wavelength=acq['xu_mapping_wavelength_A'] * 1e-10, detector='Pilatus300kw'),
        detector_config={'orientation': 3})
    # The immutable canonical asset records the former camera frame and positive scales.
    model = StackedDetectorArm(nu_scale=np.pi / 180, del_scale=np.pi / 180,
                              nu_offset=-np.deg2rad(offsets['nu']),
                              del_offset=-np.deg2rad(offsets['del']))
    geometry = Diffractometer.from_pyfai_goniometer(model.to_pyfai_goniometer(cal),
        source_motors={'del_angle': 'del', 'nu_angle': 'nu'})
    # Float64 point calls isolate mapping arithmetic from qArray's float32 grid.
    rows, cols = np.array([0., 0., 194., 194., 97.]), np.array([0., 1474., 0., 1474., 737.])
    with xu_runtime_session() as session:
        effective = resolve_xu_stitch_effective_geometry(receipt, session)
        for delta, nu in ((-40., -29.), (0., 0.), (20., 10.), (40., 29.), (40., -29.)):
            ai, = create_multigeometry_integrators_from_geometry(geometry,
                {'del': [delta], 'nu': [nu]})
            qx, qy, qz = effective.hxrd.Ang2Q.area(0., 0., 0., 0.,
                nu - offsets['nu'], delta - offsets['del'])
            points = (rows.astype(int), cols.astype(int))
            q = np.hypot(np.hypot(qx, qy), qz)[points]
            chi = -np.arctan2(qx, qz)[points]
            np.testing.assert_allclose(ai.qFunction(rows, cols) / 10, q, atol=8e-14, rtol=0)
            error = np.angle(np.exp(1j * (ai.chi(rows, cols) - chi)))
            np.testing.assert_allclose(error, 0, atol=2e-13, rtol=0)


def test_stacked_marker_cannot_override_other_expressions(tmp_path):
    asset = _model().to_pyfai_goniometer(_calibration())
    asset['trans_function']['rot3_expr'] = '0.0'
    with pytest.raises(ValueError, match='stacked'):
        Diffractometer.from_pyfai_goniometer(asset)
    from xrd_tools.analysis.stitch_operation import (
        StitchGeometryInput, StitchGeometryKind, StitchOperationRefused, capture_stitch_geometry,
    )
    path = tmp_path / 'invalid.json'
    path.write_text(json.dumps(asset))
    with pytest.raises(StitchOperationRefused, match='invalid stacked-arm') as refused:
        capture_stitch_geometry(StitchGeometryInput(path, StitchGeometryKind.PYFAI_GONIOMETER_JSON,
            source_motors=(('del_angle', 'del'), ('nu_angle', 'nu'))))
    assert refused.value.code == 'GEOMETRY_PARSE_FAILED'


def test_stacked_persisted_geometry_and_per_frame_rotations(tmp_path):
    import h5py
    import pandas as pd
    from xrd_tools.io.nexus import write_per_frame_geometry
    from xrd_tools.io.read import get_diffractometer
    from tests.core.test_diffractometer_persist import _write_entry
    model, cal = _model(), _calibration()
    geometry = Diffractometer(stacked_arm=model, calibration=cal)
    path = tmp_path / 'scan.nexus'
    _write_entry(path, geometry)
    assert get_diffractometer(path) == geometry
    motors = {'swing': np.array([10., 40.]), 'pitch': np.array([-10., 29.])}
    with h5py.File(path, 'a') as handle:
        write_per_frame_geometry(handle['entry'], pd.DataFrame(motors), [0, 1], geometry)
        saved = handle['entry/per_frame_geometry']
        expected = geometry.to_pyfai_per_frame(motors)
        for key in ('rot1', 'rot2', 'rot3'):
            # The existing per-frame schema stores radians as float32.
            assert saved[key].dtype == np.dtype('float32')
            np.testing.assert_array_equal(saved[key], expected[key].astype(np.float32))


def test_legacy_affine_serialization_does_not_gain_model_field():
    geometry = Diffractometer.psic()
    assert 'stacked_arm' not in json.loads(geometry.to_json())
    assert Diffractometer.from_json(geometry.to_json()).to_json() == geometry.to_json()


@pytest.mark.slow
def test_real_guided_stacked_fit_save_reload_and_holdout(tmp_path):
    import pyFAI
    from pyFAI.control_points import ControlPoints
    from pyFAI.goniometer import Goniometer
    from xrd_tools.core.scan import SourceKind, SourceSpec
    from xrd_tools.integrate.stitch_calibration import (
        capture_calibration_frames, adopt_calibration_pair, fit_calibration,
        save_calibration_fit, load_control_frame,
    )
    root = os.environ.get('XDART_TEST_DATA')
    if not root:
        pytest.skip('XDART_TEST_DATA is not configured')
    data = Path(root) / 'stitching/data'
    source = SourceSpec(data / 'calibration/LaB6_17keV', SourceKind.SPEC,
        options={'scan': '14.1', 'image_dir': str(data / 'images'),
                 'image_stem': 'b_thampy_LaB6_17keV_scan14_',
                 'read_image_kwargs': {'detector_shape': (195, 1475),
                                       'raw_dtype': 'int32', 'threshold': 8e5}})
    indices = (0, 9, 18, 27, 35, 55, 65, 75, 130)
    # The archived images/points use the former x-/z+ frame: rotate both into native x+/z-.
    staged = capture_calibration_frames(source, indices, tmp_path / 'session', rotation=180)
    control_dir = data / 'LaB6_Calibration_Poni_and_Data'
    frames = []
    for frame in staged:
        ai = pyFAI.load(str(control_dir / f'LaB6_17keV_scan14_{frame.index:04d}.poni'))
        rows, cols = frame.shape
        ai.poni1, ai.poni2 = rows * ai.pixel1 - ai.poni1, cols * ai.pixel2 - ai.poni2
        ai.rot1, ai.rot2 = -ai.rot1, -ai.rot2
        path = Path(frame.image_path).parent
        ai.save(str(path / 'native.poni'))
        points = ControlPoints(str(control_dir / f'LaB6_17keV_scan14_{frame.index:04d}.npt'))
        native = ControlPoints(calibrant=points.calibrant, wavelength=points.calibrant.wavelength)
        for label in points.get_labels():
            group = points.get(lbl=label)
            native.append(points=[[rows - 1 - y, cols - 1 - x] for y, x in group.points], ring=group.ring)
        native.save(str(path / 'native.npt'))
        frames.append(adopt_calibration_pair(frame, path / 'native.poni', path / 'native.npt'))
    frames = tuple(frames)
    fit = fit_calibration(frames)
    assert fit.rms_q < .012, fit.rms_q
    assert fit.n_points == 2074
    path = save_calibration_fit(fit, tmp_path / 'stacked-fit.json')
    loaded = Diffractometer.from_pyfai_goniometer(path,
        source_motors={'del_angle': 'del', 'nu_angle': 'nu'})
    assert loaded.stacked_arm is not None
    assert loaded.stacked_arm.nu_scale == -np.deg2rad(1.)
    assert loaded.stacked_arm.del_scale == -np.deg2rad(1.)
    native = Goniometer.sload(path)
    for frame in frames:
        motors = {key: np.array([value]) for key, value in frame.motors.items()}
        ai, = create_multigeometry_integrators_from_geometry(loaded, motors)
        direct = native.get_ai((frame.motors['del'], frame.motors['nu']))
        np.testing.assert_allclose(ai.qArray(), direct.qArray(), atol=2e-12, rtol=0)
    training = tuple(frame for frame in frames if frame.index not in (35, 65))
    heldout_fit = fit_calibration(training)
    heldout = Diffractometer.from_pyfai_goniometer(heldout_fit.asset['calibration'],
        source_motors={'del_angle': 'del', 'nu_angle': 'nu'})
    residuals = []
    for frame in frames:
        if frame.index not in (35, 65):
            continue
        control, calibrant, _ = load_control_frame(frame)
        ai, = create_multigeometry_integrators_from_geometry(heldout,
            {key: np.array([value]) for key, value in frame.motors.items()})
        expected = 2 * np.pi / np.asarray(calibrant.dspacing)[control.rings]
        residuals.extend(ai.qArray()[control.rows, control.cols] / 10 - expected)
    heldout_rms = float(np.sqrt(np.mean(np.square(residuals))))
    assert heldout_rms < .018, heldout_rms
    print(f'full RMS={fit.rms_q:.9g}; held-out RMS={heldout_rms:.9g}; JSON={path}')
