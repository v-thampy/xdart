"""Del-on-nu arm with detector mount and sample offset: pose, JSON, carrier, fit."""
import json

import numpy as np
import pytest

from xrd_tools.core.containers import PONI
from xrd_tools.core.geometry import DetectorCalibration, Diffractometer, StackedDetectorArm
from xrd_tools.core.geometry.stacked_arm import stacked_arm_transformation
from xrd_tools.integrate.multi import (
    GeometryIntegratorSeries,
    create_multigeometry_integrator_series_from_geometry,
    create_multigeometry_integrators_from_geometry,
)
from xrd_tools.integrate.refine import ControlFrame, refine_stacked_arm

SOURCES = {'del_angle': 'del', 'nu_angle': 'nu'}
WAVELENGTH = .7293e-10
TRUTH = dict(nu_offset=.02, del_offset=-.05, mount_tilt=np.deg2rad(.9),
             mount_roll=np.deg2rad(-.35), sample_offset1=5e-4,
             sample_offset2=-1.2e-3, sample_offset3=2.2e-3)
ANGLES = [(d, n) for n in (-10., -3., 4., 11., 18.) for d in (6., 18., 30., 42.)]


def _r1(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1., 0, 0], [0, c, s], [0, -s, c]])


def _r2(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, -s], [0, 1., 0], [s, 0, c]])


def _r3(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def _calibration(dist=.39, poni1=.035, poni2=.045):
    return DetectorCalibration(
        PONI(dist=dist, poni1=poni1, poni2=poni2, wavelength=WAVELENGTH, detector='Pilatus300kw'),
        detector_config={'orientation': 3})


def _ray(params, cal, delta, nu, rows, cols, beam_rotation=0.):
    """Independent lab ray: lab = R3(b) (R1(n) R2(d) R3(roll) R1(tilt) v - s)."""
    p = cal.poni
    v = np.stack(((rows + .5) * 172e-6 - p.poni1, (cols + .5) * 172e-6 - p.poni2,
                  np.full(np.shape(rows), p.dist)))
    pose = (_r1(params.get('nu_offset', 0.) + np.deg2rad(nu))
            @ _r2(params.get('del_offset', 0.) + np.deg2rad(delta))
            @ _r3(params.get('mount_roll', 0.)) @ _r1(params.get('mount_tilt', 0.)))
    s = np.array([params.get(f'sample_offset{i}', 0.) for i in (1, 2, 3)])
    lab = np.einsum('ij,j...->i...', _r3(beam_rotation) @ pose, v)
    lab -= (_r3(beam_rotation) @ s).reshape((3,) + (1,) * np.ndim(rows))
    tth = np.arctan2(np.hypot(lab[0], lab[1]), lab[2])
    return 4 * np.pi / (WAVELENGTH * 1e10) * np.sin(tth / 2), np.degrees(np.arctan2(lab[0], lab[1]))


def test_extended_pose_matches_independent_ray_at_every_pixel(tmp_path):
    from pyFAI.goniometer import Goniometer
    arm = StackedDetectorArm(beam_rotation=.2, **TRUTH)
    cal = _calibration()
    record = arm.to_pyfai_goniometer(cal)
    assert 'xdart_stacked_arm_v2' in record['trans_function']['constants']
    diff = Diffractometer.from_pyfai_goniometer(record, source_motors=SOURCES)
    assert diff.stacked_arm == arm
    path = tmp_path / 'extended.json'
    path.write_text(json.dumps(record))
    stock = Goniometer.sload(str(path))
    rows, cols = np.indices((195, 1475), dtype=float)
    angles = ((0., 0.), (30., 20.), (-30., -20.), (45., 29.))
    integrators = create_multigeometry_integrators_from_geometry(
        diff, {'del': np.array([a[0] for a in angles]), 'nu': np.array([a[1] for a in angles])})
    for (delta, nu), product in zip(angles, integrators):
        q, chi = _ray(TRUTH, cal, delta, nu, rows, cols, beam_rotation=.2)
        for ai in (product, stock.get_ai((delta, nu))):
            np.testing.assert_allclose(ai.qFunction(rows, cols) / 10, q, atol=1e-12, rtol=0)
            wrapped = (np.degrees(ai.chi(rows, cols)) - chi + 180) % 360 - 180
            np.testing.assert_allclose(wrapped, 0, atol=1e-9)


def test_zero_extension_keeps_the_v1_encoding_and_carrier():
    arm = StackedDetectorArm(nu_offset=.02, del_offset=-.05, beam_rotation=.1)
    assert not arm.extended
    record = arm.to_pyfai_goniometer(_calibration())
    assert record['trans_function'] == stacked_arm_transformation()
    assert len(record['param_names']) == 8
    diff = Diffractometer.from_pyfai_goniometer(record, source_motors=SOURCES)
    stored = json.loads(diff.to_json())['stacked_arm']
    assert not {'mount_tilt', 'mount_roll', 'sample_offset1'} & set(stored)
    assert Diffractometer.from_json(diff.to_json()).stacked_arm == arm
    series = create_multigeometry_integrator_series_from_geometry(
        diff, {'del': np.array([5., 20.]), 'nu': np.array([-3., 9.])})
    assert series.translations is None


def test_extended_import_is_strict():
    record = StackedDetectorArm(**TRUTH).to_pyfai_goniometer(_calibration())
    tampered = json.loads(json.dumps(record))
    tampered['trans_function']['rot1_expr'] += '+0'
    with pytest.raises(ValueError, match='stacked arm'):
        Diffractometer.from_pyfai_goniometer(tampered, source_motors=SOURCES)
    zero = json.loads(json.dumps(record))
    zero['param'][8:] = [0.] * 5
    with pytest.raises(ValueError, match='nonzero extension'):
        Diffractometer.from_pyfai_goniometer(zero, source_motors=SOURCES)
    renamed = json.loads(json.dumps(record))
    renamed['param_names'] = renamed['param_names'][:8] + renamed['param_names'][9:] + ['mount_tilt']
    with pytest.raises(ValueError, match='stacked arm'):
        Diffractometer.from_pyfai_goniometer(renamed, source_motors=SOURCES)


def test_series_carries_sample_offset_positions_only_when_present():
    motors = {'del': np.array([5., 20., 40.]), 'nu': np.array([-10., 3., 16.])}
    cal = _calibration()
    tilt_only = Diffractometer.from_pyfai_goniometer(
        StackedDetectorArm(mount_tilt=.01).to_pyfai_goniometer(cal), source_motors=SOURCES)
    assert create_multigeometry_integrator_series_from_geometry(tilt_only, motors).translations is None
    shifted = Diffractometer.from_pyfai_goniometer(
        StackedDetectorArm(**TRUTH).to_pyfai_goniometer(cal), source_motors=SOURCES)
    series = create_multigeometry_integrator_series_from_geometry(shifted, motors)
    assert len(series.translations) == 3
    for (dist, poni1, poni2), ai in zip(series.translations, series):
        assert (ai.dist, ai.poni1, ai.poni2) == (dist, poni1, poni2)
    assert set(shifted.to_pyfai_per_frame(motors)) == {'rot1', 'rot2', 'rot3', 'incident_angle'}
    with pytest.raises(ValueError, match='differ in length'):
        GeometryIntegratorSeries(cal, series.rotations, series.translations[:2])


def test_control_frame_keeps_subpixel_positions():
    frame = ControlFrame([1.7, 2.2], [3.9, 4.1], [0, 1], {'nu': 0., 'del': 5.})
    assert frame.rows.tolist() == [1, 2] and frame.cols.tolist() == [3, 4]
    assert frame.row_positions.tolist() == [1.7, 2.2]
    assert frame.col_positions.tolist() == [3.9, 4.1]
    with pytest.raises(ValueError, match='sub-pixel'):
        ControlFrame([1, 2], [3, 4], [0, 1], {'nu': 0.}, row_positions=[1.])


def _ring_controls(params, cal, angles):
    """Exact sub-pixel ring points from the independent ray (Newton along rows)."""
    from pyFAI.calibrant import get_calibrant
    calibrant = get_calibrant('LaB6')
    calibrant.wavelength = WAVELENGTH
    q_ring = 2 * np.pi / np.asarray(calibrant.dSpacing)
    grid = np.arange(0., 194.75, .25)
    columns = np.arange(3.3, 1475., 17.7)
    controls = []
    for delta, nu in angles:
        rows_g, cols_g = np.meshgrid(grid, columns, indexing='ij')
        q, _ = _ray(params, cal, delta, nu, rows_g, cols_g)
        points = []
        for ring, target in enumerate(q_ring[:30]):
            s = q - target
            i, j = np.nonzero(s[:-1] * s[1:] < 0)
            r = grid[i] + .25 * s[i, j] / (s[i, j] - s[i + 1, j])
            c = columns[j]
            for _ in range(3):
                f0 = _ray(params, cal, delta, nu, r, c)[0] - target
                slope = (_ray(params, cal, delta, nu, r + 1e-3, c)[0]
                         - _ray(params, cal, delta, nu, r - 1e-3, c)[0]) / 2e-3
                r = r - f0 / slope
            points += [(a, b, ring) for a, b in zip(r, c)]
        y, x, rings = np.asarray(points).T
        controls.append(ControlFrame(y, x, rings, {'del': delta, 'nu': nu}))
    return controls, calibrant


def test_refine_stacked_arm_recovers_an_extended_truth():
    truth = _calibration()
    controls, calibrant = _ring_controls(TRUTH, truth, ANGLES)
    seed = PONI(dist=.392, poni1=.0353, poni2=.0447, wavelength=WAVELENGTH, detector='Pilatus300kw')
    extended = refine_stacked_arm(seed, controls, nu_motor='nu', del_motor='del', calibrant=calibrant,
                                  detector_config={'orientation': 3})
    assert extended.success, extended.message
    assert extended.rms_q < 1e-8
    assert set(extended.param_errors) >= {'mount_tilt', 'sample_offset3'}
    five = refine_stacked_arm(seed, controls, nu_motor='nu', del_motor='del', calibrant=calibrant,
                              detector_config={'orientation': 3}, extended=False)
    motors = {'del': np.array([a[0] for a in ANGLES]), 'nu': np.array([a[1] for a in ANGLES])}
    rows, cols = np.indices((195, 1475), dtype=float)
    worst = {}
    for name, fit in (('extended', extended), ('five', five)):
        integrators = create_multigeometry_integrators_from_geometry(fit.diffractometer, motors)
        worst[name] = max(float(np.abs(ai.qFunction(rows, cols) / 10
                                       - _ray(TRUTH, truth, d, n, rows, cols)[0]).max())
                          for (d, n), ai in zip(ANGLES, integrators))
    assert worst['extended'] < 1e-7, worst
    assert worst['five'] > 1e-3, worst


def test_arm_extension_needs_three_nu_positions():
    truth = _calibration()
    controls, calibrant = _ring_controls(TRUTH, truth, [(10., 0.), (30., 0.), (20., 5.)])
    with pytest.raises(ValueError, match='at least 3 nu positions'):
        refine_stacked_arm(truth.poni, controls, nu_motor='nu', del_motor='del',
                           calibrant=calibrant, detector_config={'orientation': 3})


def test_guided_fit_rejects_unknown_or_xu_arm_extension():
    from xrd_tools.integrate.stitch_calibration import fit_calibration
    with pytest.raises(ValueError, match='unknown arm model'):
        fit_calibration([object()] * 3, arm_model='surface')
    with pytest.raises(ValueError, match='MultiGeometry-only'):
        fit_calibration([object()] * 3, backend='xu_hist', arm_model='stacked_extended')
