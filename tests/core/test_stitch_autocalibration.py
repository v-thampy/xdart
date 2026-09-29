"""Automatic stacked-arm refinement on synthetic LaB6 ring scans (independent extended-arm ray)."""
import numpy as np

from xrd_tools.core.scan import SourceKind, SourceSpec
from tests.core.test_stacked_arm_extension import TRUTH, _calibration, _ray, _ring_controls

ROW_ANGLES = [(d, n) for n in (-10., -3., 4., 11.) for d in (6., 20., 34.)]


def five_parameter_seed(angles):
    """The guided five-parameter fit of exact controls: the usual refinement seed."""
    from xrd_tools.core.containers import PONI
    from xrd_tools.integrate.refine import refine_stacked_arm
    controls, calibrant = _ring_controls(TRUTH, _calibration(), angles)
    seed = PONI(dist=.392, poni1=.0353, poni2=.0447, wavelength=.7293e-10, detector='Pilatus300kw')
    five = refine_stacked_arm(seed, controls, nu_motor='nu', del_motor='del', calibrant=calibrant,
                              detector_config={'orientation': 3}, extended=False)
    record = five.diffractometer.stacked_arm.to_pyfai_goniometer(five.diffractometer.calibration)
    record['detector_config'] = {'orientation': 3}
    return record


def ring_scan(root, angles=ROW_ANGLES, scale=None):
    """A SPEC scan of LaB6 ring images; ``scale`` maps a frame index to a count factor."""
    from pyFAI.calibrant import get_calibrant
    calibrant = get_calibrant('LaB6')
    calibrant.wavelength = .7293e-10
    q_rings = 2 * np.pi / np.asarray(calibrant.dspacing)[:25]
    rows, cols = np.indices((195, 1475), dtype=float)
    images = root / 'images'
    images.mkdir(parents=True)
    for index, (delta, nu) in enumerate(angles):
        q, _ = _ray(TRUTH, _calibration(), delta, nu, rows, cols)
        image = 10 + 5000 * np.exp(-((q[..., None] - q_rings) ** 2) / (2 * .008 ** 2)).sum(-1)
        image *= (scale or {}).get(index, 1.0)
        image.astype('<i4').tofile(images / f'ring_scan1_{index:04d}.raw')
    lines = '\n'.join(f'{d} {n}' for d, n in angles)
    (root / 'ring').write_text('#F ring\n#E 1\n#D today\n#O0 del nu\n#S 1 ascan del 6 34 11 1\n'
                               f'#D today\n#P0 0 0\n#N 2\n#L del  nu\n{lines}\n')
    return SourceSpec(root / 'ring', SourceKind.SPEC, options={
        'scan': '1', 'image_dir': str(images), 'image_stem': 'ring_scan1_',
        'read_image_kwargs': {'detector_shape': (195, 1475), 'raw_dtype': 'int32'}})


def test_a_partial_frame_is_screened_against_the_whole_scan_before_selection(tmp_path):
    from xrd_tools.integrate.stitch_autocalibration import refine_arm_from_scan
    angles = [(d, n) for n in (-10., -3., 4., 11.) for d in (6., 10., 14., 18., 22., 26., 30., 34.)]
    partial = angles.index((14., -3.))  # the stratified choice with 2 frames per row picks del 14 and 30
    spec = ring_scan(tmp_path / 'scan', angles, scale={partial: .1})
    fit = refine_arm_from_scan(spec, five_parameter_seed(ROW_ANGLES), max_frames=8)
    automatic = fit.asset['provenance']['automatic_refinement']
    assert [item['index'] for item in automatic['excluded_frames']] == [partial]
    assert 'partial frame' in automatic['excluded_frames'][0]['reason']
    used = [frame['index'] for frame in fit.asset['provenance']['frames']]
    assert partial not in used and len(used) == 8


def test_automatic_refinement_is_reproducible_and_leaves_the_global_generator_alone(tmp_path):
    from xrd_tools.integrate.stitch_autocalibration import refine_arm_from_scan
    spec = ring_scan(tmp_path / 'scan')
    seed = five_parameter_seed(ROW_ANGLES)
    np.random.seed(1)
    first = refine_arm_from_scan(spec, seed, max_frames=12)
    after = np.random.random()
    np.random.seed(2)  # a different prior state must not change the refinement
    second = refine_arm_from_scan(spec, seed, max_frames=12)
    assert first.asset['calibration'] == second.asset['calibration']
    assert first.n_points == second.n_points and first.rms_q == second.rms_q
    np.random.seed(1)
    assert np.random.random() == after  # the caller's generator state is restored
