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


def ring_scan(root, angles=ROW_ANGLES, scale=None, energy_ratio=1.0):
    """A SPEC scan of LaB6 ring images; ``scale`` maps a frame index to a count factor;
    ``energy_ratio``: the beam's true energy over the recorded one."""
    from pyFAI.calibrant import get_calibrant
    calibrant = get_calibrant('LaB6')
    calibrant.wavelength = .7293e-10
    q_rings = 2 * np.pi / np.asarray(calibrant.dspacing)[:25] / energy_ratio
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


def test_calibration_threshold_filters_the_scan_search_and_records_the_band(tmp_path):
    from xrd_tools.integrate.stitch_autocalibration import extract_scan_ring_points, refine_arm_from_scan
    from xrd_tools.integrate.stitch_conditioning import StitchConditioning
    spec = ring_scan(tmp_path / 'scan', [(d, 0.) for d in (6., 20., 34.)])
    originals = {}
    for path in (tmp_path / 'scan' / 'images').glob('*.raw'):
        image = np.fromfile(path, dtype='<i4').reshape(195, 1475)
        image[:, :700] = 0  # left out by the lower bound, independently of detector gaps
        image.tofile(path)
        originals[path] = path.read_bytes()
    band = StitchConditioning('manual', 1., 6000.)
    seed = five_parameter_seed(ROW_ANGLES)
    points = extract_scan_ring_points(spec, seed, max_frames=3, conditioning=band)
    assert all(control.col_positions.min() >= 699 for control in points.controls)
    result = refine_arm_from_scan(spec, seed, max_frames=3, conditioning=band)
    policy = result.asset['provenance']['automatic_refinement']['conditioning']
    assert policy == band.to_dict()
    assert all(path.read_bytes() == raw for path, raw in originals.items())


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


def test_a_seed_beam_rotation_is_not_carried_into_the_refined_arm(tmp_path):
    """|q| cannot measure a rotation about the beam: a new fit takes the design value 0, whatever
    the seed geometry held (the maintainer's saved calibration carried −0.87° from its seed)."""
    from xrd_tools.integrate.stitch_autocalibration import refine_arm_from_scan
    seed = five_parameter_seed(ROW_ANGLES)
    seed['param'][seed['param_names'].index('beam_rotation')] = np.radians(-.87)
    fit = refine_arm_from_scan(ring_scan(tmp_path / 'scan'), seed, max_frames=12)
    calibration = fit.asset['calibration']
    assert dict(zip(calibration['param_names'], calibration['param']))['beam_rotation'] == 0.


def test_the_calibrations_mask_keeps_ring_points_off_its_pixels(tmp_path):
    """One mask for the whole calibration (maintainer, 2026-09-29): the ring search skips its pixels,
    the refinement records which mask it used, and a mask of another shape is refused."""
    import pytest
    from xrd_tools.integrate.stitch_autocalibration import extract_scan_ring_points, refine_arm_from_scan
    spec = ring_scan(tmp_path / 'scan')
    seed = five_parameter_seed(ROW_ANGLES)
    mask = np.zeros((195, 1475), bool)
    mask[:, :700] = True
    unmasked = extract_scan_ring_points(spec, seed, max_frames=12)
    assert any(control.col_positions.min() < 600 for control in unmasked.controls)
    masked = extract_scan_ring_points(spec, seed, max_frames=12, mask=mask)
    assert all(control.col_positions.min() >= 699 for control in masked.controls)
    fit = refine_arm_from_scan(spec, seed, max_frames=12, mask=mask)
    assert len(fit.asset['provenance']['automatic_refinement']['mask_sha256']) == 64
    with pytest.raises(ValueError, match=r'the mask is \(2, 2\), the oriented image \(195, 1475\)'):
        extract_scan_ring_points(spec, seed, max_frames=12, mask=np.zeros((2, 2), bool))


def test_the_automatic_refinement_refines_the_energy_within_its_bound(tmp_path):
    """Maintainer, 2026-09-29: "add a refine energy toggle … just one number for +-that number". A scan
    of a beam 40 eV above the recorded energy: Refine finds it within its uncertainty and keeps the
    recorded energy with the calibration; refining again from that calibration is still ± the
    recorded energy, not the refined one."""
    import pytest
    from xrd_tools.core.energy import wavelength_m_to_energy_eV
    from xrd_tools.integrate.refine import ArmFitBounds
    from xrd_tools.integrate.stitch_autocalibration import refine_arm_from_scan
    recorded = float(wavelength_m_to_energy_eV(.7293e-10))
    spec = ring_scan(tmp_path / 'scan', energy_ratio=(recorded + 40.) / recorded)
    # TRUTH's sample sits 0.5–2.2 mm off the rotation centre: allow it, or the energy absorbs the
    # displacement the default ±0.05 mm holds back (ARM-ZERO-BOUNDS-20260929)
    wide = ArmFitBounds(sample_offset_mm=5.)
    fit = refine_arm_from_scan(spec, five_parameter_seed(ROW_ANGLES), max_frames=12, energy_bound_eV=200.,
                               bounds=wide)
    energy = fit.asset['provenance']['energy_refinement']
    assert abs(energy['refined_eV'] - (recorded + 40.)) < max(3 * energy['sigma_eV'], 1.), energy
    assert fit.asset['calibration']['source_energy_eV'] == pytest.approx(recorded)
    assert f"energy {energy['refined_eV']:.1f}" in fit.message
    again = refine_arm_from_scan(spec, fit.asset['calibration'], max_frames=12, energy_bound_eV=200., bounds=wide)
    assert again.asset['provenance']['energy_refinement']['recorded_eV'] == pytest.approx(recorded)
    assert again.asset['calibration']['source_energy_eV'] == pytest.approx(recorded)
    kept = refine_arm_from_scan(spec, fit.asset['calibration'], max_frames=12, bounds=wide)  # the energy stays
    assert 'energy_refinement' not in kept.asset['provenance']
    assert kept.asset['calibration']['source_energy_eV'] == pytest.approx(recorded)
