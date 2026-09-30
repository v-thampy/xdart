"""A guided calibration frame's predicted geometry and expected calibrant rings."""
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from xrd_tools.integrate.calibration_seed import (
    expected_rings, predict_frame_seed, ring_summary, rough_arm_record, write_seed_poni,
)
from xrd_tools.integrate.stitch_calibration import CalibrationFrame


def _frame(motors, energy_eV=17000.018, shape=(195, 1475)):
    return CalibrationFrame('frame-0', 'scan', 'spec', {}, 0, 'raw', 0, 'image', 'sha', shape, 0,
                            motors, energy_eV)


def _record(tmp_path, names=('del_angle', 'nu_angle')):
    # an independent-Euler MultiGeometry calibration, as the test data's files are written
    record = {'content': 'Goniometer calibration v2', 'detector': 'Pilatus 300kw',
              'detector_config': {'orientation': 3}, 'wavelength': 7.2932e-11,
              'param_names': ['dist', 'poni1', 'poni2', 'rot1_offset', 'rot1_scale', 'rot2_offset', 'rot2_scale'],
              'param': [0.39, 0.035, 0.043, 0.0, math.pi / 180, 0.0, math.pi / 180],
              'pos_names': list(names),
              'trans_function': {'content': 'GeometryTransformation', 'param_names': [
                  'dist', 'poni1', 'poni2', 'rot1_offset', 'rot1_scale', 'rot2_offset', 'rot2_scale'],
                  'pos_names': list(names), 'dist_expr': 'dist', 'poni1_expr': 'poni1', 'poni2_expr': 'poni2',
                  'rot1_expr': f'rot1_scale * {names[1]} + rot1_offset',
                  'rot2_expr': f'rot2_scale * {names[0]} + rot2_offset', 'rot3_expr': '0.0', 'constants': {}}}
    path = tmp_path / 'calibration.json'
    path.write_text(json.dumps(record))
    return path


@pytest.mark.parametrize('names', [('del_angle', 'nu_angle'), ('del_value', 'nu_value')])
def test_a_frame_is_seeded_at_its_own_angles_with_pyfai_ring_numbers(tmp_path, names):
    from pyFAI.calibrant import get_calibrant
    seed = predict_frame_seed(_record(tmp_path, names), _frame({'del': 20., 'nu': 0.}))
    ai = seed.integrator
    assert (ai.rot1, ai.rot2) == pytest.approx((0., math.radians(20.)))  # nu -> rot1, del -> rot2
    assert ai.wavelength == pytest.approx(12.398419843320026 / 17.000018 * 1e-10, abs=0)  # the frame's energy
    reference = get_calibrant('LaB6')
    reference.wavelength = ai.wavelength
    two_theta = np.degrees(ai.center_array((195, 1475), unit='2th_rad', scale=False))
    valid = ~np.asarray(ai.detector.mask, dtype=bool)
    low, high = two_theta[valid].min(), two_theta[valid].max()
    expected = [n for n, angle in enumerate(reference.get_2th(), 1)
                if angle is not None and low <= math.degrees(angle) <= high]
    assert [ring.number for ring in seed.rings] == expected and expected
    for ring in seed.rings:  # pyFAI's numbering: ring n is the calibrant's n-th d-spacing
        assert ring.d_spacing_A == pytest.approx(reference.dspacing[ring.number - 1])
    path = write_seed_poni(seed, tmp_path / 'seed.poni', calibrant='LaB6')
    from pyFAI.io.ponifile import PoniFile
    loaded = PoniFile(str(path))
    assert (loaded.dist, loaded.rot2) == pytest.approx((ai.dist, ai.rot2))
    assert 'Calibrant: LaB6' in path.read_text()


def test_a_seed_refuses_a_calibration_it_cannot_apply(tmp_path):
    with pytest.raises(ValueError, match='detector does not match'):
        predict_frame_seed(_record(tmp_path), _frame({'del': 0., 'nu': 0.}, shape=(8, 10)))
    with pytest.raises(ValueError, match='one del and one nu'):
        predict_frame_seed(_record(tmp_path, ('two_theta', 'chi')), _frame({'del': 0., 'nu': 0.}))
    assert ring_summary(()) == ''
    assert expected_rings.__doc__


def test_ring_summary_collapses_runs():
    from xrd_tools.integrate.calibration_seed import ExpectedRing
    rings = [ExpectedRing(n, 0., 0.) for n in (1, 2, 3, 5, 7, 8)]
    assert ring_summary(rings) == '1–3, 5, 7–8'


@pytest.mark.skipif(not os.environ.get('XDART_TEST_DATA'), reason='needs XDART_TEST_DATA')
def test_the_old_calibration_predicts_the_rings_of_the_fitted_first_frame():
    # 2026-09-25 live session: frame 0 (del 5, nu -10, 17.000018 keV) fitted in pyFAI shows
    # LaB6 rings 1-3 (10.07, 14.25, 17.48 deg); the page's old calibration predicts the same.
    calibration = Path(os.environ['XDART_TEST_DATA']) / 'stitching/data/calibration/MG_gonio_object.json'
    seed = predict_frame_seed(calibration, _frame({'del': 5., 'nu': -10.}))
    assert [ring.number for ring in seed.rings] == [1, 2, 3]
    assert [round(ring.two_theta_deg, 2) for ring in seed.rings] == [10.07, 14.25, 17.48]


def test_detector_choices_list_the_common_detectors_first_and_every_one_builds():
    from pyFAI.detectors import detector_factory
    from xrd_tools.integrate.calibration_seed import detector_choices, detector_shape
    choices = detector_choices()
    names = [name for _label, name in choices]
    assert names[:7] == ['Pilatus100k', 'Pilatus300kw', 'Eiger1M', 'Eiger2_500k', 'Eiger4M',
                         'RayonixMx225', 'RayonixMx300']
    assert choices[5][0] == 'MarCCD 225 / Rayonix MX225' and len(set(names)) == len(names) > 7
    assert 'Detector' not in names  # the generic one needs pixel sizes, not a name
    assert all(detector_factory(name).shape is not None for name in names)
    rest = [label.casefold() for label, _name in choices[7:]]
    assert rest == sorted(rest)
    assert detector_shape('Pilatus100k') == (195, 487) and detector_shape('no such detector') is None


@pytest.mark.parametrize('orientation', [3, 1, 2, 4])
def test_the_detector_orientation_reaches_the_seed_pyfai_opens(tmp_path, orientation):
    import pyFAI
    from pyFAI.io.ponifile import PoniFile
    from xrd_tools.integrate.calibration_seed import write_seed_poni
    # off-centre: a reflection moves this pixel (the detector's centre, 737, 97, hid it; Codex R16-F1)
    x, y = 100, 40
    record = rough_arm_record(distance_m=.4, detector='Pilatus300kw', beam_px=(x, y),
                              orientation=orientation)
    # the default leaves the record as it was; another orientation travels in the detector config
    assert record['detector_config'] == ({} if orientation == 3 else {'orientation': orientation})
    seed = predict_frame_seed(record, _frame({'del': 0., 'nu': 0.}))
    assert int(seed.integrator.detector.orientation) == orientation
    path = write_seed_poni(seed, tmp_path / 'seed.poni')
    assert int(PoniFile(str(path)).detector.orientation) == orientation
    # the beam pixel is where the direct beam hits at normal incidence, in every orientation,
    # in the seed as pyFAI reloads it
    for geometry in (seed.integrator, pyFAI.load(str(path))):
        assert float(geometry.tth(np.array([y]), np.array([x]))[0]) == pytest.approx(0, abs=1e-12)
    if orientation == 3:  # the default record is the pitch formula it always was, to the bit
        params = dict(zip(record['param_names'], record['param']))
        assert (params['poni1'], params['poni2']) == ((y + .5) * 172e-6, (x + .5) * 172e-6)


@pytest.mark.parametrize('orientation', [1, 2, 4])
def test_every_orientation_predicts_orientation_3s_geometry_with_the_arm_tilted(orientation):
    """pyFAI reflects before rotating and after (F·R·F, Codex R18-F1): with one-degree scales the
    seed turned the detector the wrong way once the arm moved (24° off at nu 12°). The scales
    carry the orientation's signs, so every orientation predicts orientation 3's geometry."""
    shape, (x, y) = (195, 1475), (100, 40)

    def seed(orientation, motors, **zero):
        record = rough_arm_record(distance_m=.5, detector='Pilatus300kw', beam_px=(x, y),
                                  orientation=orientation, **zero)
        return predict_frame_seed(record, _frame(motors))

    away = np.ones(shape, bool)
    away[y, x] = False  # χ is undefined at the beam
    for pose in ((30., 0.), (0., 12.), (30., 12.), (-20., -7.)):
        seeds = [seed(code, {'del': pose[0], 'nu': pose[1]}) for code in (3, orientation)]
        (tth3, chi3), (tth, chi) = ([s.integrator.center_array(shape, unit=unit, scale=False)
                                     for unit in ('2th_rad', 'chi_rad')] for s in seeds)
        assert np.abs(tth - tth3)[away].max() < 1e-9
        assert np.abs(np.angle(np.exp(1j * (chi - chi3))))[away].max() < 1e-9
        assert seeds[0].rings == seeds[1].rings
    # a known normal-incidence zero on a reversed circle still puts the beam on its pixel
    for circle in ('del', 'nu'):
        at_zero = seed(orientation, {'del': 4. if circle == 'del' else 0., 'nu': 4. if circle == 'nu' else 0.},
                       zeros_deg={circle: 4.})
        assert float(at_zero.integrator.tth(np.array([y]), np.array([x]))[0]) == pytest.approx(0, abs=1e-12)


def test_one_calibrated_frame_predicts_the_other_frames_rings(tmp_path):
    """A rough guess 200 px off misnumbers a far frame's rings; the arm fitted to the first
    calibrated frame places every frame's rings as the true arm does."""
    from tests.core.test_stacked_arm_extension import _calibration, _ring_controls
    from tests.core.test_stitch_guided_calibration import _frames
    from xrd_tools.core.geometry.stacked_arm import StackedDetectorArm
    from xrd_tools.integrate.calibration_seed import session_prediction
    truth = {'nu_offset': .02, 'del_offset': -.05}
    controls, _calibrant = _ring_controls(truth, _calibration(), [(5., -10.), (45., -10.), (25., 9.5), (5., 29.)])
    frames = _frames(tmp_path, controls, _calibration().poni)
    true_arm = StackedDetectorArm(**truth).to_pyfai_goniometer(_calibration())

    def two_theta(record, frame):
        seed = predict_frame_seed(record, frame)
        return np.degrees(seed.integrator.center_array(frame.shape, unit='2th_rad', scale=False))

    rough = rough_arm_record(distance_m=.39, detector='Pilatus300kw', beam_px=(.045 / 172e-6 + 200, .035 / 172e-6))
    assert (ring_summary(predict_frame_seed(rough, frames[1]).rings)
            != ring_summary(predict_frame_seed(true_arm, frames[1]).rings))
    assert session_prediction(frames[:0]) is None
    predicted = session_prediction(frames[:1])
    for frame in frames:
        assert np.abs(two_theta(predicted, frame) - two_theta(true_arm, frame)).max() < .05
        assert (ring_summary(predict_frame_seed(predicted, frame).rings)
                == ring_summary(predict_frame_seed(true_arm, frame).rings))


def test_the_calibrations_mask_removes_the_rings_that_fall_only_on_it(tmp_path):
    frame = _frame({'del': 20., 'nu': 0.})
    seed = predict_frame_seed(_record(tmp_path), frame)
    assert len(seed.rings) >= 2
    two_theta = np.degrees(seed.integrator.center_array((195, 1475), unit='2th_rad', scale=False))
    mask = two_theta >= (seed.rings[-2].two_theta_deg + seed.rings[-1].two_theta_deg) / 2
    masked = predict_frame_seed(_record(tmp_path), frame, mask=mask)
    assert [ring.number for ring in masked.rings] == [ring.number for ring in seed.rings[:-1]]
    # a mask of another shape is not this frame's: the rings are predicted without it
    assert predict_frame_seed(_record(tmp_path), frame, mask=np.ones((2, 2), bool)).rings == seed.rings


def test_a_rough_arm_holds_the_mount_tilt_and_roll_it_is_given():
    """A direct-beam calibration holds the detector's other rotations fixed (maintainer, 2026-09-29:
    "all other rots held to zero or some fixed value"): the zeros, and the mount's tilt and roll."""
    from xrd_tools.core.geometry import Diffractometer

    def arm(**mount):
        record = rough_arm_record(distance_m=.5, detector='Pilatus300kw', beam_px=(100, 40), **mount)
        return Diffractometer.from_pyfai_goniometer(record, source_motors={'del_angle': 'del', 'nu_angle': 'nu'}).stacked_arm

    assert not arm().extended
    tilted = arm(mount_deg=(.5, -.3))
    assert tilted.extended and (tilted.mount_tilt, tilted.mount_roll) == pytest.approx(
        (math.radians(.5), math.radians(-.3)))
    with pytest.raises(ValueError, match="single detector arm"):
        rough_arm_record(distance_m=.5, detector='Pilatus300kw', beam_px=(100, 40), nu_motor=None,
                         mount_deg=(.5, 0.))


def test_a_circle_with_no_frames_and_no_range_is_checked_at_zero():
    """With nu fixed at 0 a direct-beam calibration types only del's range: nu is checked at 0."""
    from xrd_tools.integrate.stitch_calibration import CalibrationFit, pair_validation_angles
    fit = CalibrationFit('multigeometry', {'provenance': {'frames': [], 'source_motors': {
        'del_angle': 'del', 'nu_angle': 'nu'}}}, 0., 0., 0, '')
    angles = pair_validation_angles(fit, domain={'del': [0., 30.]}, grid=4)
    assert angles == [(0., 0.), (10., 0.), (20., 0.), (30., 0.)]
    assert pair_validation_angles(fit) == []


@pytest.mark.parametrize('orientation', [3, 1, 2, 4])
@pytest.mark.parametrize('zero_motor,zero_deg', [('del', 0.), ('del', 4.), ('nu', -3.)])
def test_a_tilted_mount_keeps_the_direct_beam_on_its_pixel_at_the_given_distance(orientation, zero_motor, zero_deg):
    """Codex review F2: the rough arm promises the beam hits beam_px at the zero pose; with a tilted
    mount the PONI moves to keep that, and the distance is the direct beam's (pyFAI's directDist)."""
    x, y = 700, 95

    def seed(**mount):
        record = rough_arm_record(distance_m=.5, detector='Pilatus300kw', beam_px=(x, y), orientation=orientation,
                                  zeros_deg={zero_motor: zero_deg}, **mount)
        motors = {'del': 0., 'nu': 0., zero_motor: zero_deg}
        return predict_frame_seed(record, _frame(motors)).integrator

    for mount in ((0., 0.), (.5, 0.), (.5, -.3), (-1.2, .8)):
        integrator = seed(mount_deg=mount)
        assert float(integrator.tth(np.array([y]), np.array([x]))[0]) == pytest.approx(0, abs=1e-9), mount
        assert integrator.getFit2D()['directDist'] == pytest.approx(500., abs=1e-6), mount


@pytest.mark.parametrize('orientation', [3, 1, 2, 4])
def test_a_direct_beam_seen_off_the_zero_pose_is_placed_where_it_was_seen(orientation):
    """Maintainer, 2026-09-29: an automatic refinement from a direct-beam geometry "looks totally off".
    17-2's direct-beam scan 3 turns nu and holds del at 0, while del's zero is 2.3°: the fitted pixel is
    where the beam hit at that pose. Placed at the zero pose instead, the rough arm was 2.3° (about 92
    pixels at 395 mm) off. Now the beam hits the pixel at the scan's pose, 2.3° from it at the zero
    pose; with no pose given it hits at the zero pose, as before."""
    from xrd_tools.integrate.calibration_seed import predict_frame_seed
    x, y = 279.19, 97.
    zeros, scan = {'del': 2.3, 'nu': 0.}, {'del': 0., 'nu': 0.}

    def tth_at_beam(record, motors):
        integrator = predict_frame_seed(record, _frame(motors)).integrator
        return math.degrees(float(integrator.tth(np.array([y]), np.array([x]))[0]))

    rough = dict(distance_m=.395, detector='Pilatus300kw', beam_px=(x, y), orientation=orientation, zeros_deg=zeros)
    record = rough_arm_record(**rough, beam_readings={'del': 0.})
    assert tth_at_beam(record, scan) == pytest.approx(0, abs=1e-7)
    assert tth_at_beam(record, zeros) == pytest.approx(2.3, abs=1e-6)
    assert tth_at_beam(rough_arm_record(**rough), zeros) == pytest.approx(0, abs=1e-7)


def test_an_energy_refined_calibration_seeds_frames_at_the_recorded_energy():
    """CAL-2: frames are calibrated at the energy the scan recorded, so a seed from an energy-refined
    calibration uses its recorded energy (source_energy_eV) when the frame has none of its own."""
    from pyFAI import units
    from xrd_tools.core.energy import energy_eV_to_wavelength_m
    from xrd_tools.integrate.calibration_seed import predict_frame_seed
    record = rough_arm_record(distance_m=.4, detector='Pilatus300kw', beam_px=(200., 90.),
                              wavelength_m=energy_eV_to_wavelength_m(17100.))
    frame = _frame({'del': 20., 'nu': 0.}, energy_eV=None)  # a scan without an energy column
    assert predict_frame_seed(record, frame).integrator.wavelength == pytest.approx(
        energy_eV_to_wavelength_m(17100.), rel=1e-12, abs=0)  # metres: no default 1e-12 abs
    refined = {**record, 'source_energy_eV': 17000.}
    assert predict_frame_seed(refined, frame).integrator.wavelength == pytest.approx(
        units.hc / 17.0 * 1e-10, rel=1e-12, abs=0)
