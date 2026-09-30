"""Motor-zero estimates and fit bounds for the detector arm (maintainer, 2026-09-29).

The public convention is physical angle = motor readback − motor-zero reading; the arm's internal
offset is −(signed scale) × zero. A fit starts at the supplied estimates and stays within the
maintainer's freedom about them: the zeros and the mount ±2°, each sample displacement ±0.05 mm
(or ±0.02), the distance ±5 % of its seed unless bounded explicitly, the energy only when asked;
it reports each parameter's initial and fitted value and whether it reached its bound, and never
widens one ("this is to make sure the sample displacement in particular doesn't go to
unrealistic values").
"""
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from xrd_tools.core.containers import PONI
from xrd_tools.core.geometry import DetectorCalibration, Diffractometer, StackedDetectorArm
from xrd_tools.core.geometry.stacked_arm import orientation_signs
from xrd_tools.integrate.multi import create_multigeometry_integrators_from_geometry
from xrd_tools.integrate.refine import ArmFitBounds, ControlFrame, refine_stacked_arm
from tests.core.test_stacked_arm_extension import ANGLES, TRUTH, WAVELENGTH, _calibration, _ring_controls

SOURCES = {'del_angle': 'del', 'nu_angle': 'nu'}
ZEROS = {'del': 2.3, 'nu': -.8}  # motor-zero readings (degrees)
PIXEL = 172e-6


def _offsets(zeros=ZEROS, orientation=3):
    """The internal offsets these motor zeros give: −(signed scale) × zero."""
    nu_sign, del_sign = orientation_signs(orientation)
    return {'nu_offset': -nu_sign * np.deg2rad(zeros['nu']), 'del_offset': -del_sign * np.deg2rad(zeros['del'])}


def _seed(orientation, dist=.392):
    """Orientation 3's seed pixel (.0353, .0447 m) as this orientation places it."""
    from pyFAI.detectors import detector_factory
    rows, cols = np.array([.0353 / PIXEL - .5]), np.array([.0447 / PIXEL - .5])
    poni1, poni2, _ = detector_factory('Pilatus300kw', config={'orientation': orientation}
                                       ).calc_cartesian_positions(d1=rows, d2=cols)
    return PONI(dist=dist, poni1=float(poni1[0]), poni2=float(poni2[0]), wavelength=WAVELENGTH,
                detector='Pilatus300kw')


def _fit(controls, calibrant, orientation=3, **options):
    options.setdefault('extended', False)
    return refine_stacked_arm(_seed(orientation), controls, nu_motor='nu', del_motor='del',
                              calibrant=calibrant, detector_config={'orientation': orientation}, **options)


@pytest.mark.parametrize('orientation', [3, 1])
def test_nonzero_zeros_are_recovered_from_their_estimates(orientation):
    """The fit starts at the estimates, not at zero, and recovers the motor zeros in pyFAI's
    default orientation and in one that reverses both circles' signs."""
    controls, calibrant = _ring_controls(_offsets(), _calibration(), ANGLES)
    fit = _fit(controls, calibrant, orientation, bounds=ArmFitBounds(zeros_deg={'del': 1.5, 'nu': 0.}))
    assert fit.success and fit.rms_q < 1e-8, fit.message
    table = fit.fit_bounds
    assert (table['del_zero_deg']['initial'], table['nu_zero_deg']['initial']) == (1.5, 0.)
    assert table['del_zero_deg']['fitted'] == pytest.approx(2.3, abs=1e-6)
    assert table['nu_zero_deg']['fitted'] == pytest.approx(-.8, abs=1e-6)
    assert (table['del_zero_deg']['lower'], table['del_zero_deg']['upper']) == pytest.approx((-.5, 3.5))
    assert (table['nu_zero_deg']['lower'], table['nu_zero_deg']['upper']) == pytest.approx((-2., 2.))
    assert not any(row['at_bound'] for row in table.values())
    zeros = fit.diffractometer.stacked_arm.motor_zeros()
    assert zeros == pytest.approx(ZEROS, abs=1e-6)


def test_a_zero_outside_its_freedom_stops_at_the_bound_and_says_so():
    controls, calibrant = _ring_controls(_offsets(), _calibration(), ANGLES)
    fit = _fit(controls, calibrant, bounds=ArmFitBounds(zeros_deg={'del': 0., 'nu': -.8}))
    row = fit.fit_bounds['del_zero_deg']
    assert (row['lower'], row['upper']) == pytest.approx((-2., 2.))  # never widened
    assert row['fitted'] == pytest.approx(2., abs=1e-9) and row['at_bound']


@pytest.mark.parametrize('limit_mm', [.05, .02])
def test_each_sample_displacement_is_held_within_its_limit(limit_mm):
    """TRUTH's sample sits 0.5–2.2 mm off the rotation centre; the fit may not follow it there."""
    zeros = {'nu': -np.rad2deg(TRUTH['nu_offset']), 'del': -np.rad2deg(TRUTH['del_offset'])}
    controls, calibrant = _ring_controls(TRUTH, _calibration(), ANGLES)
    fit = _fit(controls, calibrant, extended=True,
               bounds=ArmFitBounds(zeros_deg=zeros, sample_offset_mm=limit_mm))
    table = fit.fit_bounds
    for k in (1, 2, 3):
        row = table[f'sample_offset{k}_mm']
        assert row['initial'] == 0. and (row['lower'], row['upper']) == pytest.approx((-limit_mm, limit_mm))
        assert abs(row['fitted']) <= limit_mm * (1 + 1e-12)
        assert row['at_bound'] == (limit_mm - abs(row['fitted']) <= 1e-6 * limit_mm)
    assert any(table[f'sample_offset{k}_mm']['at_bound'] for k in (1, 2, 3))
    for name in ('mount_tilt_deg', 'mount_roll_deg'):
        assert (table[name]['lower'], table[name]['upper']) == pytest.approx((-2., 2.))


def test_the_distance_is_held_within_five_percent_of_its_seed_unless_bounded_explicitly():
    controls, calibrant = _ring_controls(_offsets(), _calibration(), ANGLES[::2])
    bounds = ArmFitBounds(zeros_deg=ZEROS)
    row = _fit(controls, calibrant, bounds=bounds).fit_bounds['distance_mm']
    assert (row['initial'], row['lower'], row['upper']) == pytest.approx((392., 372.4, 411.6))
    assert row['fitted'] == pytest.approx(390., abs=1e-6) and not row['at_bound']
    explicit = _fit(controls, calibrant, bounds=bounds, distance_bounds_m=(.385, .395)).fit_bounds['distance_mm']
    assert (explicit['lower'], explicit['upper']) == pytest.approx((385., 395.))
    far = refine_stacked_arm(replace_dist(_seed(3), .30), controls, nu_motor='nu', del_motor='del',
                             calibrant=calibrant, detector_config={'orientation': 3}, extended=False,
                             bounds=bounds).fit_bounds['distance_mm']
    assert (far['lower'], far['upper']) == pytest.approx((285., 315.)) and far['at_bound']


def replace_dist(poni, dist):
    from dataclasses import replace
    return replace(poni, dist=dist)


def test_a_single_arm_keeps_its_held_circle_as_a_fixed_angle():
    """A vertical 2θ arm moves del; its held nu is the arm's fixed yaw, not a motor zero: the table
    names it so, and the arm stores it as +scale × angle (no motor reading to subtract)."""
    held = {'nu_offset': np.deg2rad(-.5), 'del_offset': -np.deg2rad(1.)}  # yaw −0.5°, del zero +1°
    controls, calibrant = _ring_controls(held, _calibration(), [(d, 0.) for d in (6., 18., 30., 42., 54.)])
    controls = [ControlFrame(c.row_positions, c.col_positions, c.rings, {'del': c.motors['del']})
                for c in controls]
    fit = refine_stacked_arm(_seed(3), controls, nu_motor=None, del_motor='del', calibrant=calibrant,
                             detector_config={'orientation': 3}, extended=False,
                             bounds=ArmFitBounds(zeros_deg={'del': .5}))
    assert fit.success and fit.rms_q < 1e-8, fit.message
    table = fit.fit_bounds
    assert 'nu_zero_deg' not in table and 'del_held_deg' not in table
    assert table['del_zero_deg']['fitted'] == pytest.approx(1., abs=1e-6)
    assert table['nu_held_deg']['fitted'] == pytest.approx(-.5, abs=1e-6)
    assert (table['nu_held_deg']['lower'], table['nu_held_deg']['upper']) == pytest.approx((-2., 2.))
    assert fit.diffractometer.stacked_arm.nu_offset == pytest.approx(np.deg2rad(-.5), abs=1e-9)


@pytest.mark.parametrize('orientation', [3, 2])
@pytest.mark.parametrize('mount', [(0., 0.), (.6, -.3)])
def test_the_rough_arm_puts_the_direct_beam_on_its_pixel_with_both_circles_at_their_zeros(orientation, mount):
    from xrd_tools.integrate.calibration_seed import rough_arm_record
    from xrd_tools.integrate.multi import create_multigeometry_integrator_series_from_geometry
    beam = (700.3, 95.6)  # x, y in the oriented image
    record = rough_arm_record(distance_m=.39, detector='Pilatus300kw', beam_px=beam, zeros_deg=ZEROS,
                              orientation=orientation, wavelength_m=WAVELENGTH, mount_deg=mount)
    diff = Diffractometer.from_pyfai_goniometer(record, source_motors=SOURCES)
    assert diff.stacked_arm.motor_zeros() == pytest.approx(ZEROS)
    ai = next(iter(create_multigeometry_integrator_series_from_geometry(
        diff, {'del': np.array([ZEROS['del']]), 'nu': np.array([ZEROS['nu']])})))
    z, y, x = (float(np.ravel(v)[0]) for v in ai.calc_pos_zyx(d1=np.array([beam[1]]), d2=np.array([beam[0]])))
    assert abs(y) < 1e-9 and abs(x) < 1e-9  # on the direct beam ...
    assert z == pytest.approx(.39, abs=1e-9)  # ... at the typed distance along it


def _least_squares_calls(monkeypatch):
    """Every SciPy solve's start and box (the real solver runs: this only records)."""
    import scipy.optimize
    real, calls = scipy.optimize.least_squares, []

    def recording(fun, x0, **options):
        lower, upper = options.get('bounds', (-np.inf, np.inf))
        calls.append((np.array(x0, dtype=float), np.broadcast_to(lower, np.shape(x0)).copy(),
                      np.broadcast_to(upper, np.shape(x0)).copy()))
        return real(fun, x0, **options)
    monkeypatch.setattr(scipy.optimize, 'least_squares', recording)
    return calls


def _assert_boxed(calls, bounds, orientation=3):
    """Each solve's angles and displacements are boxed about the estimates (in internal units)."""
    nu_sign, del_sign = orientation_signs(orientation)
    zeros = dict(bounds.zeros_deg)
    half = np.deg2rad(bounds.zero_freedom_deg)
    assert calls
    centres = [-nu_sign * np.deg2rad(zeros.get('nu', 0.)), -del_sign * np.deg2rad(zeros.get('del', 0.))]
    assert calls[0][0][3:5] == pytest.approx(centres, abs=1e-15)  # the first solve starts at the estimates
    for _start, lower, upper in calls:
        for index, circle, sign in ((3, 'nu', nu_sign), (4, 'del', del_sign)):
            centre = -sign * np.deg2rad(zeros.get(circle, 0.))
            assert (lower[index], upper[index]) == pytest.approx((centre - half, centre + half), abs=1e-15)
        assert np.isfinite(lower[0]) and np.isfinite(upper[0])
        assert np.isinf(lower[1:3]).all() and np.isinf(upper[1:3]).all()  # the PONI stays free
        if len(lower) >= 10:
            limit = bounds.sample_offset_mm * 1e-3
            assert lower[7:10] == pytest.approx([-limit] * 3) and upper[7:10] == pytest.approx([limit] * 3)


def _subpixel_frames(tmp_path, controls, poni, orientation=3):
    """Calibration frames whose control points keep their sub-pixel positions."""
    from pyFAI.calibrant import get_calibrant
    from pyFAI.control_points import ControlPoints
    from pyFAI.detectors import detector_factory
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    from xrd_tools.integrate.stitch_calibration import CalibrationFrame, adopt_calibration_pair
    import hashlib
    frames = []
    for index, control in enumerate(controls):
        directory = tmp_path / str(index)
        directory.mkdir(parents=True)
        image = directory / 'frame.tif'
        tifffile.imwrite(image, np.zeros((195, 1475), dtype=np.float32))
        points = ControlPoints(calibrant=get_calibrant('LaB6'), wavelength=poni.wavelength)
        for ring in np.unique(control.rings):
            chosen = control.rings == ring
            points.append(points=np.column_stack([control.row_positions[chosen],
                                                  control.col_positions[chosen]]).tolist(), ring=int(ring))
        npt = directory / 'frame.npt'
        points.save(str(npt))
        seed = directory / 'frame.poni'
        AzimuthalIntegrator(dist=poni.dist, poni1=poni.poni1, poni2=poni.poni2, wavelength=poni.wavelength,
                            detector=detector_factory('Pilatus300kw', config={'orientation': orientation})
                            ).save(str(seed))
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        frame = CalibrationFrame(str(index), str(image), 'image_file', {}, index, str(image), 0, str(image),
                                 digest, (195, 1475), 0, dict(control.motors), None)
        frames.append(adopt_calibration_pair(frame, seed, npt))
    return tuple(frames)


def test_the_manual_fit_and_the_prediction_box_every_solve(tmp_path, monkeypatch):
    """The guided fit's preliminary and extended solves, its five-parameter fallback and the
    session prediction all take the same box about the same estimates."""
    from xrd_tools.integrate.calibration_seed import session_prediction
    from xrd_tools.integrate.stitch_calibration import fit_calibration
    inside = dict(_offsets(), mount_tilt=np.deg2rad(.4), mount_roll=np.deg2rad(-.3),
                  sample_offset1=1e-5, sample_offset2=-1.5e-5, sample_offset3=1.8e-5)
    controls, _ = _ring_controls(inside, _calibration(), ANGLES[::2])
    frames = _subpixel_frames(tmp_path / 'wide', controls, _seed(3))
    bounds = ArmFitBounds(zeros_deg={'del': 2., 'nu': -.5}, sample_offset_mm=.02)
    calls = _least_squares_calls(monkeypatch)
    fit = fit_calibration(frames, arm_model='auto', bounds=bounds)
    assert fit.asset['provenance']['arm_model'] == 'stacked_extended'
    assert {len(start) for start, *_box in calls} == {5, 10}  # the preliminary and the extended solve
    _assert_boxed(calls, bounds)
    recorded = fit.asset['provenance']['fit_bounds']
    assert recorded['estimates']['zeros_deg'] == {'del': 2., 'nu': -.5}
    assert recorded['estimates']['zero_sources'] == {'del': 'typed', 'nu': 'typed'}
    assert recorded['estimates']['sample_offset_mm'] == .02
    assert recorded['parameters']['del_zero_deg']['fitted'] == pytest.approx(2.3, abs=1e-6)
    calls.clear()  # three nu positions 0.3° apart, pixel-centre points: the extension is refused
    from tests.core.test_stitch_guided_calibration import _frames
    narrow, _ = _ring_controls(_offsets(), _calibration(), [(d, n) for n in (-.3, 0., .3) for d in (6., 30.)])
    (tmp_path / 'narrow').mkdir()
    fallback = fit_calibration(_frames(tmp_path / 'narrow', narrow, _seed(3)), arm_model='auto', bounds=bounds)
    assert fallback.asset['provenance']['arm_model'] == 'stacked'
    assert 5 in {len(start) for start, *_box in calls}
    _assert_boxed(calls, bounds)
    calls.clear()
    assert session_prediction(frames[:3], bounds=bounds) is not None
    _assert_boxed(calls, bounds)


def test_the_automatic_refinement_boxes_every_fold_and_starts_from_the_seeds_zeros(tmp_path, monkeypatch):
    """Each validation fold and the final fit take the box; with no zero typed it is centred on the
    seed calibration's own zeros, and a typed one wins (its source is recorded)."""
    from tests.core.test_stitch_autocalibration import ROW_ANGLES, five_parameter_seed, ring_scan
    from xrd_tools.integrate.stitch_autocalibration import refine_arm_from_scan
    seed = five_parameter_seed(ROW_ANGLES)
    seed_zeros = Diffractometer.from_pyfai_goniometer(seed, source_motors=SOURCES).stacked_arm.motor_zeros()
    spec = ring_scan(tmp_path / 'scan')
    calls = _least_squares_calls(monkeypatch)
    wide = ArmFitBounds(sample_offset_mm=5.)
    fit = refine_arm_from_scan(spec, seed, bounds=wide)
    recorded = fit.asset['provenance']['fit_bounds']['estimates']
    assert recorded['zeros_deg'] == pytest.approx(seed_zeros)
    assert recorded['zero_sources'] == {'del': 'seed', 'nu': 'seed'}
    validation = fit.asset['provenance']['automatic_refinement']['validation']
    assert len(validation['folds']) == 4  # every fold was fitted, each within the box below
    _assert_boxed(calls, ArmFitBounds(zeros_deg=seed_zeros, sample_offset_mm=5.))
    calls.clear()
    typed = ArmFitBounds(zeros_deg={'del': 2.5, 'nu': -1.}, sample_offset_mm=5.)
    fit = refine_arm_from_scan(spec, seed, bounds=typed)
    recorded = fit.asset['provenance']['fit_bounds']['estimates']
    assert recorded['zeros_deg'] == {'del': 2.5, 'nu': -1.}
    assert recorded['zero_sources'] == {'del': 'typed', 'nu': 'typed'}
    _assert_boxed(calls, typed)


def _controls_from(diff, angles, calibrant):
    """Exact sub-pixel ring points of a geometry (Newton along rows on its own q)."""
    q_ring = 2 * np.pi / np.asarray(calibrant.dspacing)
    motors = {'del': np.array([a[0] for a in angles]), 'nu': np.array([a[1] for a in angles])}
    grid, columns = np.arange(0., 194.5, .5), np.arange(5.1, 1475., 29.3)
    rows_g, cols_g = np.meshgrid(grid, columns, indexing='ij')
    controls = []
    for (delta, nu), ai in zip(angles, create_multigeometry_integrators_from_geometry(diff, motors)):
        q = ai.qFunction(rows_g, cols_g) / 10
        points = []
        for ring, target in enumerate(q_ring[:30]):
            s = q - target
            i, j = np.nonzero(s[:-1] * s[1:] < 0)
            r = grid[i] + .5 * s[i, j] / (s[i, j] - s[i + 1, j])
            c = columns[j]
            for _ in range(4):
                slope = (ai.qFunction(r + 1e-3, c) - ai.qFunction(r - 1e-3, c)) / 2e-2
                r = r - (ai.qFunction(r, c) / 10 - target) / slope
            points += [(a, b, ring) for a, b in zip(r, c)]
        y, x, rings = np.asarray(points).T
        controls.append(ControlFrame(y, x, rings, {'del': delta, 'nu': nu}))
    return controls


@pytest.mark.parametrize('orientation', [3, 1])
def test_a_saved_fit_with_nonzero_zeros_reloads_with_the_same_2theta_and_chi(tmp_path, orientation):
    """The saved calibration keeps the fitted total offsets: through the importer and the per-frame
    integrator it gives the generating arm's 2θ and χ at every pixel (nothing subtracts the zero
    again); xrayutilities pairs orientation 3 and still refuses another by name."""
    from pyFAI.calibrant import get_calibrant
    from xrd_tools.integrate.stitch_calibration import fit_calibration, save_calibration_pair
    nu_sign, del_sign = orientation_signs(orientation)
    arm = StackedDetectorArm(nu_scale=nu_sign * np.pi / 180, del_scale=del_sign * np.pi / 180,
                             mount_tilt=np.deg2rad(.4), mount_roll=np.deg2rad(-.3),
                             sample_offset1=2e-5, sample_offset2=-1e-5, sample_offset3=3e-5,
                             **_offsets(orientation=orientation))
    poni = replace_dist(_seed(orientation), .39)
    truth = Diffractometer.from_pyfai_goniometer(
        arm.to_pyfai_goniometer(DetectorCalibration(poni, detector_config={'orientation': orientation})),
        source_motors=SOURCES)
    calibrant = get_calibrant('LaB6')
    calibrant.wavelength = WAVELENGTH
    angles = ANGLES[::2]
    frames = _subpixel_frames(tmp_path / 'frames', _controls_from(truth, angles, calibrant), _seed(orientation),
                              orientation)
    fit = fit_calibration(frames, arm_model='stacked_extended', bounds=ArmFitBounds(zeros_deg=ZEROS))
    assert fit.rms_q < 1e-8, fit.message
    saved = save_calibration_pair(fit, tmp_path / 'calibration' / 'arm.json', validation_angles=angles)
    record = json.loads(Path(saved.pyfai_path).read_text())
    loaded = Diffractometer.from_pyfai_goniometer(record, source_motors=SOURCES)
    assert loaded.stacked_arm.motor_zeros() == pytest.approx(ZEROS, abs=1e-7)
    motors = {'del': np.array([a[0] for a in angles]), 'nu': np.array([a[1] for a in angles])}
    for mine, theirs in zip(create_multigeometry_integrators_from_geometry(loaded, motors),
                            create_multigeometry_integrators_from_geometry(truth, motors)):
        assert np.abs(mine.twoThetaArray() - theirs.twoThetaArray()).max() < 1e-8
        assert np.abs((mine.chiArray() - theirs.chiArray() + np.pi) % (2 * np.pi) - np.pi).max() < 1e-8
    if orientation == 3:
        assert saved.xu_path is not None and saved.xu_refusal == ''
    else:
        assert saved.xu_path is None and 'orientation' in saved.xu_refusal


def test_the_instrument_records_its_motor_zeros_and_older_records_read_none():
    from xrd_tools.session.instrument import InstrumentGeometry
    geometry = InstrumentGeometry('psic', zeros=(('del', 2.3), ('nu', -.8)))
    assert geometry.arm_zeros() == {'del': 2.3, 'nu': -.8}
    record = geometry.to_record()
    assert record['zeros'] == {'del': 2.3, 'nu': -.8}
    assert InstrumentGeometry.from_record(record) == geometry
    assert 'zeros' not in InstrumentGeometry('psic').to_record()
    older = {'geometry': 'psic', 'motors': {'del': 'del', 'nu': 'nu'}}
    assert InstrumentGeometry.from_record(older).zeros == ()
    assert InstrumentGeometry('twoc', zeros=(('tth', 1.5),)).arm_zeros() == {'del': 1.5}
    assert InstrumentGeometry('twoc', zeros=(('tth', 1.5),), plane='horizontal').arm_zeros() == {'nu': 1.5}
    assert InstrumentGeometry.pick('psic', ['del', 'nu', 'eta'], keep=geometry).zeros == geometry.zeros
    with pytest.raises(ValueError, match='twoc has no circle'):
        InstrumentGeometry('twoc', zeros=(('del', 1.),))
    with pytest.raises(ValueError, match='finite'):
        InstrumentGeometry('psic', zeros=(('del', float('nan')),))


def test_a_zero_keeps_its_source_through_every_fit_path(tmp_path):
    """The automatic arm model fits twice (the extension, then maybe five parameters) and the
    xrayutilities fit once more through the pyFAI one: a zero left blank stays 'default', never
    becomes 'typed', however often the estimates are resolved."""
    from xrd_tools.integrate.stitch_calibration import fit_calibration
    once = ArmFitBounds(zeros_deg={'del': 2.}).resolved(['del', 'nu'])
    assert dict(once.zero_sources) == {'del': 'typed', 'nu': 'default'}
    assert dict(once.resolved(['del', 'nu']).zero_sources) == {'del': 'typed', 'nu': 'default'}
    seeded = ArmFitBounds().resolved(['del', 'nu'], seed_zeros={'del': 2.5, 'nu': -.5})
    assert dict(seeded.resolved(['del', 'nu']).zero_sources) == {'del': 'seed', 'nu': 'seed'}
    controls, _ = _ring_controls(_offsets(), _calibration(), ANGLES[::2])
    frames = _subpixel_frames(tmp_path, controls, _seed(3))
    for arm_model in ('auto', 'stacked'):
        fit = fit_calibration(frames, arm_model=arm_model, bounds=ArmFitBounds(zeros_deg={'del': 2.}))
        assert fit.asset['provenance']['fit_bounds']['estimates']['zero_sources'] == {'del': 'typed', 'nu': 'default'}


def test_a_fitted_motor_scale_keeps_the_motor_zero_within_its_bounds():
    """Codex review F2: with the motor scales fitted the box was on the offset, so a 0.8 scale let the
    motor zero reach +2.3° past its ±2°, reported inside. The fitted-scale fit now bounds the zero."""
    from xrd_tools.integrate.refine import refine_goniometer
    controls, calibrant = _ring_controls({'nu_offset': 0., 'del_offset': 0.}, _calibration(), ANGLES)
    # a del motor turning 0.8° per reading degree, zero at +2.3: rotation = 0.8 (reading − 2.3) degrees
    controls = [ControlFrame(c.row_positions, c.col_positions, c.rings,
                             {'nu': c.motors['nu'], 'del': c.motors['del'] / .8 + 2.3}) for c in controls]
    fit = refine_goniometer(_seed(3), controls, rot1_motor='nu', rot2_motor='del', calibrant=calibrant,
                            detector_config={'orientation': 3}, fit_scales=True, model='stacked_del_on_nu',
                            bounds=ArmFitBounds(zeros_deg={'del': 0., 'nu': 0.}))
    row = fit.fit_bounds['del_zero_deg']
    assert (row['lower'], row['upper']) == pytest.approx((-2., 2.))
    assert row['fitted'] == pytest.approx(2., abs=1e-9) and row['at_bound']
    arm = fit.diffractometer.stacked_arm
    assert -arm.del_offset / arm.del_scale == pytest.approx(row['fitted'], abs=1e-9)  # the saved zero is the bounded one


def test_the_automatic_refinement_and_the_prediction_take_an_explicit_distance_box(tmp_path, monkeypatch):
    """Codex review F3: explicit distance bounds reach every fold and the final fit of the automatic
    refinement and the prediction, and are recorded; bounds excluding the seed are refused."""
    from tests.core.test_stitch_autocalibration import ROW_ANGLES, five_parameter_seed, ring_scan
    from xrd_tools.integrate.calibration_seed import session_prediction
    from xrd_tools.integrate.stitch_autocalibration import refine_arm_from_scan
    seed, spec = five_parameter_seed(ROW_ANGLES), ring_scan(tmp_path / 'scan')
    calls = _least_squares_calls(monkeypatch)
    fit = refine_arm_from_scan(spec, seed, bounds=ArmFitBounds(sample_offset_mm=5.), distance_bounds_m=(.385, .395))
    assert calls and all((lower[0], upper[0]) == (.385, .395) for _start, lower, upper in calls)
    assert fit.asset['provenance']['distance_bounds_m'] == [.385, .395]
    assert fit.asset['provenance']['fit_bounds']['parameters']['distance_mm']['lower'] == pytest.approx(385.)
    with pytest.raises(ValueError, match='seed inside them'):  # the seed's .39 m lies outside: refused
        refine_arm_from_scan(spec, seed, bounds=ArmFitBounds(sample_offset_mm=5.), distance_bounds_m=(.30, .31))
    calls.clear()
    controls, _ = _ring_controls(_offsets(), _calibration(), ANGLES[::2])
    frames = _subpixel_frames(tmp_path / 'frames', controls, _seed(3))
    assert session_prediction(frames[:3], bounds=ArmFitBounds(zeros_deg=ZEROS), distance_bounds_m=(.385, .395))
    assert calls and all((lower[0], upper[0]) == (.385, .395) for _start, lower, upper in calls)


def test_a_session_records_its_fit_policy_and_older_sessions_still_read(tmp_path):
    """Codex review F6: version 4 keeps the zeros, bounds and distance the session fitted with."""
    from xrd_tools.integrate.stitch_calibration import load_calibration_session, save_calibration_session
    controls, _ = _ring_controls(_offsets(), _calibration(), ANGLES[:3])
    frames = _subpixel_frames(tmp_path / 'frames', controls, _seed(3))
    policy = {'bounds': ArmFitBounds(zeros_deg={'del': 2.3}, sample_offset_mm=.02).to_record(),
              'distance_seed_m': .39, 'distance_bounds_m': [.385, .395]}
    path = save_calibration_session(tmp_path, frames, arm=('del', 'nu'), fit_policy=policy)
    loaded, arm, recorded = load_calibration_session(path, with_arm=True, with_policy=True)
    assert json.loads(Path(path).read_text())['version'] == 4 and arm == ('del', 'nu') and len(loaded) == 3
    assert recorded == policy and ArmFitBounds.from_record(recorded['bounds']).sample_offset_mm == .02
    (tmp_path / 'older').mkdir()
    older = save_calibration_session(tmp_path / 'older', frames, arm=('del', 'nu'))  # version 3
    assert load_calibration_session(older, with_arm=True, with_policy=True)[1:] == (('del', 'nu'), None)
    broken = json.loads(Path(path).read_text())
    broken['fit_policy']['bounds']['sample_offset_mm'] = -1.
    Path(path).write_text(json.dumps(broken))
    with pytest.raises(ValueError, match='positive'):
        load_calibration_session(path, with_policy=True)
