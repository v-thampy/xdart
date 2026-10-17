"""Frozen product-interface oracle. References use pyFAI, never the HTML mockups.

A4 is fixed before product implementation: segment AUC, the constants below,
support weights and conservative SE. A failed ensemble is a design checkpoint.
"""
from dataclasses import replace
import os
from pathlib import Path

import numpy as np
import pytest

from xrd_tools.core.geometry.camera import DetectorCamera, stitch_orientation
from xrd_tools.core.geometry.diffractometer import ImageOrientation, Diffractometer
from xrd_tools.core.energy import wavelength_m_to_energy_eV
from xrd_tools.integrate.calibration_seed import expected_rings, rough_arm_record, predict_frame_seed
from xrd_tools.integrate.stitch_calibration import CalibrationFrame

SHAPE = (1043, 981)
PITCH = 172e-6
SCORE_POLICY = dict(arc_step_px=1., segment_px=32., ring_offsets=(-.75, 0., .75),
                    background_offsets=(-4.75, -4., -3.25, 3.25, 4., 4.75),
                    minimum_segments=4, minimum_support=.05, margin_se=3.)
SEEDS = tuple(range(16))
BACKGROUNDS = (.5, 5., 50.)
SIGNAL = 200.
POSITIVE_CASES = (((-190., 229.), (0., 0.), (0., 0.)),
                  ((-190., 229.), (2., -1.), (0., 0.)),
                  ((-190., 229.), (0., 0.), (15., 7.)))


def api():
    from xrd_tools import geometry_figures
    return geometry_figures


def inputs(offset=(-190., 229.), tilt=(0., 0.), pose=(0., 0.), shape=SHAPE):
    return api().GeometryInputs(shape=shape, detector='Pilatus1M', distance_m=.2,
                                wavelength_m=1e-10, beam_offset_px=offset,
                                mount_deg=tilt, readings=(('del', pose[0]), ('nu', pose[1])))


def reference(value, camera, senses=(1, 1)):
    """Separate reference setup: existing arm/seed owners and upstream calibrant image.
    Geometry inputs are fixed lab/pixel offsets, not a beam refit for each candidate.
    """
    from pyFAI.calibrant import get_calibrant
    orientation = stitch_orientation(camera)
    oriented_shape = value.shape[::-1] if orientation.swaps_axes else value.shape
    centre = (np.array(oriented_shape)[::-1] - 1) / 2
    beam = centre + value.beam_offset_px
    record = rough_arm_record(detector=value.detector, distance_m=value.distance_m,
                              wavelength_m=value.wavelength_m, beam_px=beam, camera=orientation,
                              mount_deg=value.mount_deg, orientation=value.first_pixel,
                              senses=dict(zip(('del', 'nu'), senses)))
    frame = CalibrationFrame('f', 's', 'spec', {}, 0, 'raw', 0, 'i', 'sha', oriented_shape,
                             0, dict(value.readings), wavelength_m_to_energy_eV(value.wavelength_m))
    seed = predict_frame_seed(record, frame)
    calibrant = get_calibrant('LaB6'); calibrant.wavelength = value.wavelength_m
    image = calibrant.fake_calibration_image(seed.integrator, shape=oriented_shape,
               Imax=SIGNAL, resolution=np.degrees(PITCH/value.distance_m))
    return seed.integrator, orientation.unapply(image)


def test_camera_candidates_match_transforms_on_non_square_images():
    gf = api(); candidates = gf.camera_candidates()
    assert len(candidates) == len(set(c.camera for c in candidates)) == 8
    raw = np.arange(35).reshape(5, 7)
    for candidate in candidates:
        transform = stitch_orientation(candidate.camera)
        actual = candidate.orientation.apply(raw)
        np.testing.assert_array_equal(actual, transform.apply(raw))
        for x, y in ((0., 0.), (6., 4.), (1.25, 2.75)):
            xy = transform.apply_pixel((x, y), raw.shape)
            assert candidate.raw_pixel(xy, raw.shape) == pytest.approx((x, y))
        assert candidate.label and candidate.tooltip
        assert candidate.lab_up == tuple(candidate.camera.raw_vector('x+'))
        assert candidate.lab_z == tuple(candidate.camera.raw_vector('z+'))


def test_first_pixel_corners_match_pyfai_orientation():
    from pyFAI.detectors.orientation import Orientation
    gf = api()
    for code in (1, 2, 3, 4):
        figure = gf.first_pixel_figure(code, (5, 7))
        assert figure.corner_name == Orientation(code).name
        expected = {1: (0., 0.), 2: (6., 0.), 3: (6., 4.), 4: (0., 4.)}[code]
        assert figure.markers[0].xy == expected
        assert 'separate' in figure.caption.lower()


@pytest.mark.parametrize('tilt,pose', [((0., 0.), (0., 0.)),
                       ((3., -2.), (0., 0.)), ((1., 2.), (40., 12.))])
def test_ring_curves_match_pyfai_contours_and_predicted_ring_metadata(tilt, pose):
    from scipy.ndimage import map_coordinates
    from contourpy import contour_generator
    from scipy.spatial import cKDTree
    gf = api(); value = inputs(tilt=tilt, pose=pose)
    for camera in (DetectorCamera('x-', 'z+'), DetectorCamera('z-', 'x-')):
        ai, _ = reference(value, camera)
        prediction = gf.predict_ring_curves(value, camera)
        shape = value.shape[::-1] if stitch_orientation(camera).swaps_axes else value.shape
        rings = expected_rings(ai, 'LaB6', shape)
        assert prediction.rings == rings
        tth = ai.center_array(shape, unit='2th_rad', scale=False)
        for curve in prediction.curves:
            xy = np.asarray(curve.xy)
            assert not xy.flags.writeable and xy.shape[1] == 2
            oriented = np.asarray([stitch_orientation(camera).apply_pixel(p, value.shape) for p in xy])
            valid = np.isfinite(oriented).all(axis=1)
            oriented = oriented[valid]
            if not len(oriented):
                continue
            delta = abs(ai.tth(oriented[:, 1], oriented[:, 0]) - np.radians(curve.two_theta_deg))
            assert delta.max() / (PITCH/value.distance_m) <= .05
            contours = contour_generator(z=tth).lines(np.radians(curve.two_theta_deg))
            assert contours
            # Dense contour distance bounds the normal distance, independently of the kernel.
            cloud = np.concatenate(contours)
            assert cKDTree(cloud).query(oriented)[0].max() <= .5
            assert np.isfinite(map_coordinates(tth, oriented[:, ::-1].T, order=1)).all()
        assert prediction.beam_xy is not None
        if pose == (40., 12.):
            assert prediction.beam_off_detector and prediction.arrows


def test_residual_examples_show_declared_geometric_errors():
    gf = api(); examples = gf.residual_example()
    assert set(examples) == {'good fit', 'zero offset', 'wrong mounting'}
    for name, figure in examples.items():
        assert figure.units == 'Å⁻¹' and figure.caption and figure.curves
        assert any(m.role == 'rejected' for m in figure.markers)
    good, zero, wrong = (np.array(examples[k].curves[0].xy)[:, 1]
                         for k in ('good fit', 'zero offset', 'wrong mounting'))
    assert np.max(abs(good)) < 1e-10
    assert np.ptp(zero) > 0 and np.ptp(wrong) > 0
    assert not np.allclose(zero, wrong)


def assert_policy(finding):
    assert dict(finding.policy) == SCORE_POLICY
    for item in finding.candidates:
        if not item.ring_scores:
            continue
        weights = np.array([r.support for r in item.ring_scores]); weights /= weights.sum()
        assert item.score == pytest.approx(sum(w*r.score for w, r in zip(weights, item.ring_scores)))
        assert item.standard_error == pytest.approx(sum(w*r.standard_error for w, r in zip(weights, item.ring_scores)))
        for ring in item.ring_scores:
            assert ring.score == pytest.approx(np.mean(ring.segment_scores))
            assert ring.standard_error == pytest.approx(np.std(ring.segment_scores, ddof=1)/np.sqrt(len(ring.segment_scores)))
            assert ring.support >= .05 and len(ring.segment_scores) >= 4


def test_orientation_scores_identifiable_ties_masks_and_refusals():
    gf = api(); camera = DetectorCamera('x-', 'z+'); value = inputs()
    _, image = reference(value, camera)
    original = image.copy()
    result = gf.score_orientation(image, value)
    assert result.identifiable and result.best.camera == camera
    assert result.margin > 3*result.margin_standard_error
    assert_policy(result)
    # Constant background is legitimate, including zero counts.
    zero_background = np.where(image > 10., image, 0.)
    assert gf.score_orientation(zero_background, value).identifiable
    mask = np.zeros(value.shape, bool); mask[:100] = True
    image[110, 110] = 1e9
    masked = gf.score_orientation(image, value, mask=mask, saturation=1e6)
    assert masked.exclusions.geometric == int(mask.sum()) and masked.exclusions.value == 1
    assert masked.identifiable
    np.testing.assert_array_equal(image[mask], original[mask])
    refused = gf.score_orientation(image, value, mask=np.ones(value.shape, bool))
    assert not refused.identifiable and refused.reason == 'Insufficient unmasked ring support.'
    with pytest.raises(ValueError, match='shape'):
        gf.score_orientation(image[:-1], value)
    for offset in ((0., 0.), (0., 229.), (100., 100.)):
        control = inputs(offset=offset)
        _, control_image = reference(control, camera)
        finding = gf.score_orientation(control_image, control)
        assert not finding.identifiable and len(finding.tied) >= 2
        assert camera in tuple(c.camera for c in finding.tied)
        assert 'not identifiable' in finding.reason.lower()
        if offset == (100., 100.):
            assert 'diagonal' in finding.reason.lower()


# About five minutes locally (313 s): slow, so pr.yml's core job skips it, and its own budget instead of the
# job-wide --timeout, as the notebook smoke has (maintainer, 2026-10-11).
@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_frozen_photon_ensemble_never_tunes_and_preserves_diagonal_ties():
    gf = api()
    for offset, tilt, pose in POSITIVE_CASES:
        for candidate in gf.camera_candidates():
            value = inputs(offset, tilt, pose)
            _, photons = reference(value, candidate.camera)
            for background in BACKGROUNDS:
                for seed in SEEDS:
                    image = np.random.default_rng(seed).poisson(photons+background)
                    finding = gf.score_orientation(image, value)
                    assert finding.identifiable and finding.best.camera == candidate.camera, (
                        offset, tilt, pose, candidate.label, background, seed, finding.reason)
                    assert_policy(finding)
    for offset in ((0., 0.), (0., 229.), (100., 100.)):
        value = inputs(offset); camera = DetectorCamera('x-', 'z+')
        _, photons = reference(value, camera)
        for background in BACKGROUNDS:
            for seed in SEEDS:
                finding = gf.score_orientation(np.random.default_rng(seed).poisson(photons+background), value)
                assert not finding.identifiable and camera in tuple(c.camera for c in finding.tied)


def test_zero_and_moved_frames_resolve_camera_and_senses_against_ground_truth():
    gf = api(); truth = DetectorCamera('x-', 'z+'); value = inputs()
    frames = []
    for pose in ((0., 0.), (15., 0.), (0., 12.)):
        current = replace(value, readings=(('del', pose[0]), ('nu', pose[1])))
        _, image = reference(current, truth, senses=(-1, 1))
        frames.append(gf.OrientationFrame(image, current, None))
    result = gf.diagnose_orientation(tuple(frames))
    assert result.identifiable and result.best.camera == truth and result.best.senses == (-1, 1)
    mirror = inputs(offset=(0., 0.)); frames = []
    for pose in ((0., 0.), (15., 0.)):
        current = replace(mirror, readings=(('del', pose[0]), ('nu', pose[1])))
        _, image = reference(current, truth)
        frames.append(gf.OrientationFrame(image, current, None))
    tie = gf.diagnose_orientation(tuple(frames))
    assert not tie.identifiable and 'flip' in tie.reason.lower() and 'sense' in tie.reason.lower()


def test_known_feature_reports_explicit_backend_chi_or_refuses():
    gf = api(); value = inputs(); camera = DetectorCamera('x-', 'z+')
    result = gf.feature_azimuth(value, camera, (200., 300.), expected_lab='x+')
    assert result.pyfai_convention and result.xrayutilities_convention
    assert np.isfinite((result.pyfai_chi_deg, result.xrayutilities_chi_deg)).all()
    assert result.expected_lab == 'x+' and result.reason
    assert gf.feature_azimuth(value, camera, (200., 300.), expected_lab=None).reason == 'Declare the known feature’s lab direction.'


def test_instrument_figures_match_presets_zeros_and_reversed_circle_stacks():
    from xrd_tools.session.instrument import InstrumentGeometry
    from xrayutilities.math import getVector, ArbRotation
    gf = api()
    for geometry, preset in (('psic', Diffractometer.psic()), ('fourc', Diffractometer.fourc()),
                             ('twoc', Diffractometer.two_circle())):
        value = InstrumentGeometry(geometry)
        figure = gf.instrument_figure(value, {m: 0. for _, m in value.motors})
        assert tuple(c.axis for c in figure.circles) == preset.sample_circles+preset.detector_circles
        assert all(c.angle_deg == 0 for c in figure.circles)
        # LT-SMALL-FIXES-20261014, LT-007: one shared explanation for the three views, not one per view
        assert len(figure.panels) == 3 and all(p.arrows for p in figure.panels)
        assert sum(bool(p.caption) for p in figure.panels) == 1
        reversed_value = replace(value, zeros=((value.circles[0], 3.),), reversed_circles=(value.circles[0],))
        readings = {m: 8. for _, m in value.motors}
        forward = gf.instrument_figure(value, readings)
        reverse = gf.instrument_figure(reversed_value, readings)
        assert reverse.circles[0].angle_deg == 5.
        assert reverse.circles[0].axis != figure.circles[0].axis
        axes = dict(reversed_value.circle_axes())
        for role in ('sample', 'detector'):
            prefix = np.eye(3)
            for circle, drawn, original in zip(value.circles, reverse.circles, forward.circles):
                if value.role(circle) != role:
                    continue
                axis = getVector(axes[circle])
                angle = readings[value.motor(circle)] - (reversed_value.zero(circle) or 0.)
                assert drawn.angle_deg == angle
                np.testing.assert_allclose(drawn.lab_axis, prefix @ axis, atol=1e-12, rtol=0)
                prefix = prefix @ ArbRotation(axis, angle).matrix
                if role == 'detector':
                    np.testing.assert_allclose(drawn.lab_axis, original.lab_axis, atol=1e-12, rtol=0)
        if geometry != 'twoc':
            assert not np.allclose(reverse.circles[1].lab_axis, forward.circles[1].lab_axis)
    assert gf.instrument_figure(InstrumentGeometry('psic'), {}).reason.startswith('Missing motor readback:')
    assert gf.instrument_figure(Diffractometer.psic_halpha(), {}).reason.startswith('Unsupported schematic geometry:')


def test_rsm_direct_beam_scan1_consistent_set_keeps_reflection_and_untrusted_sense_ties():
    from xrd_tools.integrate.direct_beam import calibrate_direct_beam_scan
    gf = api(); root = Path(os.environ.get('XDART_TEST_DATA', '/missing'))/'RSM'
    if not (root/'direct_beam').is_file():
        pytest.skip('XDART_TEST_DATA lacks RSM/direct_beam')
    camera = DetectorCamera('z-', 'x-')
    fit = calibrate_direct_beam_scan(root/'direct_beam', '1', motor='del', image_dir=root/'images',
        filename_template='b_thampy_direct_beam_scan{scan}_{point:04d}.raw',
        reader_options={'detector_shape': (195, 487), 'raw_dtype': 'int32', 'raw_header_skip': 0},
        detector='Pilatus100k', motion='vertical', angle_zero_deg=0., beam_pixel=None,
        orientation=stitch_orientation(camera))
    assert fit.used.all() and fit.distance_m == pytest.approx(1.014649, abs=1e-5)
    trusted = gf.consistent_motion_candidates(fit, camera, detector='Pilatus100k', motor='del',
                                             held={'nu': 0.}, trusted_senses={'del': 1})
    cameras = {c.camera for c in trusted.candidates}
    assert camera in cameras and DetectorCamera('z+', 'x+') not in cameras
    assert DetectorCamera('z+', 'x-') in cameras
    assert 'single-axis' in trusted.reason.lower() and 'nu' in trusted.reason.lower()
    untrusted = gf.consistent_motion_candidates(fit, camera, detector='Pilatus100k', motor='del', held={'nu': 0.})
    assert any(c.camera == DetectorCamera('z+', 'x+') and c.senses[0] == -1 for c in untrusted.candidates)
