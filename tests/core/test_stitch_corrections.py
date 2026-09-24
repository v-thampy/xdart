"""pyFAI corrections in the streaming MultiGeometry Stitch: solid angle,
polarization, Poisson sigma, and hash-bound dark and flat images."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from xrd_tools.analysis.module_transaction import ModuleDisposition
from xrd_tools.analysis.stitch_operation import (
    StitchOperationPlan, StitchOperationRefused, prepare_stitch_operation, run_stitch_operation,
)
from xrd_tools.integrate.frame_screen import FrameScreen, FrameScreenPolicy
from xrd_tools.integrate.multi import _resolve_streaming_ranges, stitch_1d_streaming
from xrd_tools.io.analysis_artifact import read_analysis_artifact
from tests.core.test_stitch_operation import _prepared

SHAPE = (9, 13)


def _ai():
    from pyFAI.detectors import Detector
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    return AzimuthalIntegrator(dist=.39, poni1=3.7 * 75e-6, poni2=5.9 * 172e-6, rot2=np.deg2rad(4.),
                               detector=Detector(pixel1=75e-6, pixel2=172e-6, max_shape=SHAPE),
                               wavelength=.729e-10)


def test_one_frame_matches_pyfai_with_dark_flat_polarization_and_poisson():
    from xrd_tools.integrate.detector_mask import mask_with_detector
    rng = np.random.default_rng(3)
    raw = rng.poisson(60, SHAPE).astype(float) + 20.
    dark = 20. + rng.uniform(-1, 1, SHAPE)
    flat = rng.uniform(.8, 1.2, SHAPE)
    dark[0, 0] = np.nan
    flat[1, 1] = 0.
    monitor, npt, radial = 3., 24, (0., 2.)
    kwargs = dict(npt=npt, radial_range=radial, method='full', normalization=[monitor],
                  polarization_factor=.95, error_model='poisson')
    payload, _ = stitch_1d_streaming([raw], 1, lambda: iter([_ai()]), dark=dark, flat=flat, **kwargs)

    fixed_radial, fixed_azimuth = _resolve_streaming_ranges(
        lambda: iter([_ai()]), 1, unit='q_A^-1', radial_range=radial, azimuth_range=None)
    ai = _ai()
    ai.setChiDiscAtPi()
    invalid = np.zeros(SHAPE, bool)
    invalid[0, 0] = invalid[1, 1] = True
    solid = ai.detector.pixel1 * ai.detector.pixel2 / ai.dist ** 2
    common = dict(correctSolidAngle=True, polarization_factor=.95,
                  dark=np.where(invalid, 0., dark), flat=np.where(invalid, 1., flat),
                  mask=mask_with_detector(ai, invalid), radial_range=fixed_radial,
                  azimuth_range=fixed_azimuth, unit='q_A^-1', method='full', safe=True,
                  normalization_factor=monitor * solid)
    # pyFAI's Poisson model treats the dark as noiseless; a measured dark adds its counts.
    expected = ai.integrate1d_ng(raw, npt, variance=np.maximum(raw, 1.) + np.where(invalid, 0., dark),
                                 **common)
    np.testing.assert_allclose(payload.intensity, expected.intensity, rtol=1e-12, atol=0)
    np.testing.assert_allclose(payload.sigma, expected.sigma, rtol=1e-12, atol=0)
    noiseless = ai.integrate1d_ng(raw, npt, error_model='poisson', **common)
    finite = np.isfinite(payload.sigma) & (payload.sigma > 0)
    assert np.all(payload.sigma[finite] > noiseless.sigma[finite])

    # Subtracting the dark before pyFAI would also lose the raw frame's full counts.
    subtracted, _ = stitch_1d_streaming([raw - np.nan_to_num(dark)], 1, lambda: iter([_ai()]),
                                        flat=flat, **kwargs)
    finite = np.isfinite(payload.sigma) & (payload.sigma > 0)
    assert np.all(subtracted.sigma[finite] < payload.sigma[finite])


def test_dark_subtracted_screen_catches_a_dropped_frame_on_a_pedestal():
    pedestal, signal = 100., 50.
    images = [np.full(SHAPE, pedestal + signal) for _ in range(6)]
    images[3] = np.full(SHAPE, pedestal)
    dark = np.full(SHAPE, pedestal)
    angles = [2., 4., 6., 8., 10., 12.]

    def ais():
        from pyFAI.detectors import Detector
        from pyFAI.integrator.azimuthal import AzimuthalIntegrator
        return (AzimuthalIntegrator(dist=.39, poni1=3.7 * 75e-6, poni2=5.9 * 172e-6,
                                    rot2=np.deg2rad(a), wavelength=.729e-10,
                                    detector=Detector(pixel1=75e-6, pixel2=172e-6, max_shape=SHAPE))
                for a in angles)

    without = FrameScreen(FrameScreenPolicy(), list(range(6)), [0] * 6)
    stitch_1d_streaming(images, 6, ais, npt=30, radial_range=(0., 3.), method='full', frame_screen=without)
    assert without.summary()['excluded'] == []  # 100/150 of its neighbours looks fine
    with_dark = FrameScreen(FrameScreenPolicy(), list(range(6)), [0] * 6)
    stitch_1d_streaming(images, 6, ais, npt=30, radial_range=(0., 3.), method='full',
                        frame_screen=with_dark, dark=dark)
    assert with_dark.summary()['excluded'] == [3]


def test_run_stitch_applies_corrections_only_on_the_streaming_merge():
    from xrd_tools.analysis.plans import StitchPlan, run_stitch
    from xrd_tools.corrections.stack import CorrectionStack
    from tests.core.test_stitch_geometry import _stitch_diff_and_src
    diff, src = _stitch_diff_and_src()
    eager = StitchPlan(diffractometer=diff, mode='1d', npt_1d=100,
                       corrections=CorrectionStack(polarization_factor=.9))
    with pytest.raises(ValueError, match='need the streaming MultiGeometry stitch'):
        run_stitch(eager, src)
    with pytest.raises(ValueError, match='air absorption'):
        run_stitch(replace(eager, streaming_multigeometry=True,
                           corrections=CorrectionStack(air_absorption_mu=1.)), src)
    streaming = replace(eager, streaming_multigeometry=True, radial_range=(.5, 3.))
    polarized = run_stitch(streaming, src).payload
    plain = run_stitch(replace(streaming, corrections=None), src).payload
    identity = run_stitch(replace(streaming, corrections=CorrectionStack()), src).payload
    np.testing.assert_array_equal(identity.intensity, plain.intensity)
    assert not np.allclose(polarized.intensity, plain.intensity, equal_nan=True)
    sigma = run_stitch(replace(streaming, error_model='poisson'), src).payload.sigma
    assert sigma is not None and np.isfinite(sigma).any()


def test_plan_defaults_validation_and_fingerprints(tmp_path):
    plan = _prepared(tmp_path).plan
    assert (plan.correct_solid_angle, plan.polarization_factor, plan.error_model) == (True, .99, None)
    variants = [replace(plan, correct_solid_angle=False), replace(plan, polarization_factor=None),
                replace(plan, polarization_factor=.9), replace(plan, error_model='poisson')]
    assert len({plan.fingerprint, *(item.fingerprint for item in variants)}) == 5
    for bad in (dict(polarization_factor=1.5), dict(polarization_factor=True),
                dict(error_model='azimuthal'), dict(correct_solid_angle=1)):
        with pytest.raises((TypeError, ValueError)):
            replace(plan, **bad)
    np.save(tmp_path / 'dark.npy', np.ones((195, 487)))
    (tmp_path / 'dark.txt').write_text('1')
    with pytest.raises(StitchOperationRefused) as refused:
        replace(plan, dark_path=tmp_path / 'dark.txt')
    assert refused.value.code == 'DARK_FORMAT_UNSUPPORTED'
    dark = replace(plan, dark_path=tmp_path / 'dark.npy')
    assert dark.fingerprint != plan.fingerprint


def _run(tmp_path, name, **changes):
    (tmp_path / name).mkdir(parents=True, exist_ok=True)
    request = _prepared(tmp_path / name)
    plan = replace(request.plan, **changes)
    request = prepare_stitch_operation(request.module.source, request.module.output, plan,
                                       project_root=tmp_path / name)
    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
    return request, read_analysis_artifact(request.module.output.target)


def _project_file(tmp_path, name, filename, value):
    (tmp_path / name).mkdir(parents=True, exist_ok=True)
    path = tmp_path / name / filename
    np.save(path, value)
    return path


def test_dark_and_flat_reach_pyfai_through_the_operation(tmp_path):
    _, plain = _run(tmp_path, 'plain')
    dark_value = 2.5
    dark_path = _project_file(tmp_path, 'dark', 'dark.npy', np.full((195, 487), dark_value))
    request, dark = _run(tmp_path, 'dark', dark_path=dark_path)
    # pyFAI removes the dark per unit of pixel coverage: ΔI·Σnorm = dark·count
    # (to the float32 precision the artifact stores).
    covered = plain.coverage > 0
    np.testing.assert_allclose(((plain.intensity - dark.intensity) * dark.normalization)[covered],
                               dark_value * dark.coverage[covered], rtol=1e-4)
    recorded = json.loads(dark.provenance_json)['plan']['corrections']
    assert recorded['dark']['relative_path'] == 'dark.npy'
    assert recorded['dark']['sha256'] == request.plan._dark_receipt.sha256
    assert (recorded['solid_angle'], recorded['polarization_factor']) == (True, .99)

    flat_value = np.full((195, 487), 2.)
    flat_value[:, :40] = 0.  # dead columns are masked, not divided by zero
    flat_path = _project_file(tmp_path, 'flat', 'flat.npy', flat_value)
    _, flat = _run(tmp_path, 'flat', flat_path=flat_path)
    recorded = json.loads(flat.provenance_json)['plan']['corrections']['flat']
    assert (recorded['min_fraction'], recorded['masked_pixels'], recorded['median']) == (.1, 40 * 195, 2.)
    assert np.all(flat.coverage <= plain.coverage + 1e-9) and np.any(flat.coverage < plain.coverage - 1e-6)
    same = (flat.coverage > 0) & np.isclose(flat.coverage, plain.coverage)
    np.testing.assert_allclose(flat.intensity[same], plain.intensity[same] / 2., rtol=1e-6)


def test_poisson_sigma_persists_and_a_changed_dark_refuses(tmp_path):
    dark_path = _project_file(tmp_path, 'poisson', 'dark.npy', np.full((195, 487), 1.))
    _, stored = _run(tmp_path, 'poisson', error_model='poisson', dark_path=dark_path)
    assert stored.sigma is not None and np.isfinite(stored.sigma).any()

    changed_path = _project_file(tmp_path, 'changed', 'dark.npy', np.full((195, 487), 1.))
    request = _prepared(tmp_path / 'changed')
    request = prepare_stitch_operation(request.module.source, request.module.output,
                                       replace(request.plan, dark_path=changed_path),
                                       project_root=tmp_path / 'changed')
    np.save(changed_path, np.full((195, 487), 3.))
    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.REFUSED
    assert result.terminal.code == 'DARK_IDENTITY_MISMATCH'
    assert not Path(request.module.output.target).exists()


def test_dark_is_oriented_like_the_mask(tmp_path):
    from xrd_tools.analysis.stitch_native import native_detector_image, native_mask
    from xrd_tools.analysis.stitch_operation import (
        StitchGeometryInput, StitchGeometryKind, capture_stitch_geometry,
    )
    plan = _prepared(tmp_path).plan
    geometry = capture_stitch_geometry(StitchGeometryInput(
        plan.geometry.lexical_path, StitchGeometryKind.PYFAI_GONIOMETER_JSON,
        source_motors=(('del_angle', 'del'), ('nu_angle', 'nu')), image_rotation=90))
    pattern = np.zeros((487, 195))  # raw file orientation of the rotated detector
    pattern[:40, :7] = 1.
    np.save(tmp_path / 'mask.npy', pattern.astype(np.uint8))
    np.save(tmp_path / 'dark.npy', 5. * pattern)
    oriented = replace(plan, geometry=geometry, native_records=True, radial_range=None,
                       mask_path=tmp_path / 'mask.npy', dark_path=tmp_path / 'dark.npy')
    dark = native_detector_image(oriented, oriented._dark_receipt)
    assert dark.shape == (195, 487)
    np.testing.assert_array_equal(dark > 0, native_mask(oriented))


def test_one_frame_stitch_matches_int_up_to_multigeometry_solid_angle_scale():
    from xrd_tools.integrate.single import integrate_1d
    rng = np.random.default_rng(11)
    raw = rng.poisson(80, SHAPE).astype(float)
    monitor, npt = 2., 24
    radial, fixed_azimuth = _resolve_streaming_ranges(
        lambda: iter([_ai()]), 1, unit='q_A^-1', radial_range=None, azimuth_range=None)
    stitched, _ = stitch_1d_streaming([raw], 1, lambda: iter([_ai()]), npt=npt, radial_range=radial,
                                      method='full', normalization=[monitor], polarization_factor=.95,
                                      error_model='poisson')
    ai = _ai()
    ai.setChiDiscAtPi()
    single = integrate_1d(raw, ai, npt=npt, unit='q_A^-1', method='full', radial_range=radial,
                          azimuth_range=fixed_azimuth, error_model='poisson',
                          polarization_factor=.95, normalization_factor=monitor)
    # MultiGeometry normalizes by the absolute solid angle pixel1·pixel2/dist²; pyFAI
    # returns float32, so the two agree to its precision.
    scale = ai.dist ** 2 / (ai.detector.pixel1 * ai.detector.pixel2)
    finite = np.isfinite(single.intensity) & (single.intensity != 0)
    assert finite.sum() > npt // 2
    np.testing.assert_allclose(stitched.intensity[finite], single.intensity[finite] * scale, rtol=1e-6)
    np.testing.assert_allclose(stitched.sigma[finite], single.sigma[finite] * scale, rtol=1e-6)


def test_a_multi_frame_dark_is_refused(tmp_path):
    import tifffile
    plan = _prepared(tmp_path).plan
    tifffile.imwrite(tmp_path / 'darks.tif', np.ones((2, 195, 487), dtype=np.float32))
    with pytest.raises(StitchOperationRefused) as refused:
        replace(plan, dark_path=tmp_path / 'darks.tif')
    assert refused.value.code == 'DARK_FORMAT_UNSUPPORTED'
    tifffile.imwrite(tmp_path / 'dark.tif', np.ones((195, 487), dtype=np.float32))
    assert replace(plan, dark_path=tmp_path / 'dark.tif')._dark_receipt.shape == (195, 487)


def test_native_run_persists_poisson_sigma_with_its_frame_records(tmp_path):
    from tests.core.test_native_stitch_operation import _native
    native = _native(tmp_path, 'multigeometry', 'q_A^-1', grouped=False)
    request = prepare_stitch_operation(native.module.source, native.module.output,
                                       replace(native.plan, error_model='poisson'), project_root=tmp_path)
    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
    payload = read_analysis_artifact(request.module.output.target)
    assert payload.schema_version == 6 and payload.sigma is not None
    assert np.isfinite(payload.sigma).any()
    assert len(payload.contributing_frames) == len(request.module.source.selected_labels)


def test_two_dimensional_merge_applies_dark_and_flat():
    from xrd_tools.integrate.multi import stitch_2d_streaming
    raw = np.random.default_rng(5).poisson(60, SHAPE).astype(float) + 7.
    kwargs = dict(npt_rad=12, npt_azim=8, method='full', radial_range=None, azimuth_range=None)
    plain, plain_diag = stitch_2d_streaming([raw], 1, lambda: iter([_ai()]), **kwargs)
    dark, dark_diag = stitch_2d_streaming([raw], 1, lambda: iter([_ai()]),
                                          dark=np.full(SHAPE, 7.), **kwargs)
    covered = np.asarray(plain_diag.coverage) > 0
    removed = (plain.intensity - dark.intensity) * np.asarray(dark_diag.normalization)
    np.testing.assert_allclose(removed[covered], 7. * np.asarray(dark_diag.coverage)[covered], rtol=1e-5)
    flat, _ = stitch_2d_streaming([raw], 1, lambda: iter([_ai()]), flat=np.full(SHAPE, 2.), **kwargs)
    np.testing.assert_allclose(flat.intensity[covered], plain.intensity[covered] / 2., rtol=1e-6)


def test_a_rebuilt_old_multigeometry_record_drops_corrections_it_never_applied():
    from xrd_tools.analysis.plans import StitchPlan
    from xrd_tools.corrections.stack import CorrectionStack
    record = {'backend': 'multigeometry', 'streaming_multigeometry': True,
              'corrections': CorrectionStack(polarization_factor=.9).to_dict()}
    assert StitchPlan.from_provenance(record).corrections is None  # written before corrections applied
    current = StitchPlan.from_provenance({**record, 'error_model': None})
    assert current.corrections.polarization_factor == .9
    histogram = StitchPlan.from_provenance({**record, 'backend': 'pyfai_hist', 'streaming_multigeometry': False})
    assert histogram.corrections.polarization_factor == .9  # histogram backends always applied them


def test_the_frame_screen_statistic_names_the_dark():
    from xrd_tools.integrate.frame_screen import FrameScreenPolicy
    assert 'dark-subtracted' in FrameScreenPolicy().to_provenance()['statistic']
