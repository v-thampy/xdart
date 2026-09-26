"""A guided calibration frame's predicted geometry and expected calibrant rings."""
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from xrd_tools.integrate.calibration_seed import (
    expected_rings, predict_frame_seed, ring_summary, write_seed_poni,
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
    assert ai.wavelength == pytest.approx(12.398419843320026 / 17.000018 * 1e-10)  # the frame's energy
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
