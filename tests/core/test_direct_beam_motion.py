"""A direct-beam fit keeps |slope| (``calibrate_direct_beam``), so a detector mounted the other way round fits its
scan as well as the right mounting and saves a geometry that moves the beam backwards. The maintainer, 2026-10-06:
"i would also like to understand why 90 degrees doesn't work" (RSM's Pilatus 100K: both 90° and 270° fitted
RSM/direct_beam scan 1 with the same residuals). ``direct_beam_motion_mismatch`` places the saved arm's own
direct beam at the scan's end angles and compares it with the measured centres."""
import os
from pathlib import Path

import numpy as np
import pytest

from tests.core.test_direct_beam import beam_images
from xrd_tools.core.geometry import ImageOrientation
from xrd_tools.integrate.calibration_seed import direct_beam_motion_mismatch, rough_arm_record
from xrd_tools.integrate.direct_beam import calibrate_direct_beam, calibrate_direct_beam_scan

#: The Pilatus 100K turned a quarter, as the calibration window's images are: 487 rows, 195 columns.
SHAPE, PITCH, DISTANCE = (487, 195), 172e-6, .5
ANGLES = np.array([-1., -.5, 0., .5, 1.])


def _fit_and_arm(images, motion):
    """The window's two steps: the fit in the turned image, then the rough arm from its distance and beam."""
    result = calibrate_direct_beam(ANGLES, images, detector='Detector',
                                   detector_config=dict(pixel1=PITCH, pixel2=PITCH, max_shape=SHAPE),
                                   beam_pixel=None, angle_zero_deg=0., motion=motion)
    record = rough_arm_record(distance_m=result.distance_m, detector='Pilatus100k', beam_px=result.zero_beam_pixel,
                              camera=ImageOrientation(rotation=90))
    return result, record


def test_a_beam_moving_as_the_del_arm_moves_it_agrees():
    """pyFAI's frame has rows up (x+): +del raises the detector, so the beam meets it lower, at smaller rows.
    ``beam_images`` moves it by −D·tan(del)/pitch along the rows."""
    images, centres, _ = beam_images(ANGLES, shape=SHAPE, pitches=(PITCH, PITCH), distance=DISTANCE, zero=0.,
                                     motion='vertical')
    assert centres[-1] < centres[0]
    result, record = _fit_and_arm(images, 'vertical')
    assert direct_beam_motion_mismatch(record, result, motor='del', held={'nu': 0.}) is None


def test_the_same_scan_mounted_the_other_way_round_is_named():
    """The images a detector turned 180° records: the fit is as good, the geometry moves the beam backwards."""
    images, _, _ = beam_images(ANGLES, shape=SHAPE, pitches=(PITCH, PITCH), distance=DISTANCE, zero=0.,
                               motion='vertical')
    turned = [np.ascontiguousarray(image[::-1, ::-1]) for image in images]
    result, record = _fit_and_arm(turned, 'vertical')
    assert result.used.all() and result.distance_m == pytest.approx(DISTANCE, rel=1e-3)
    mismatch = direct_beam_motion_mismatch(record, result, motor='del', held={'nu': 0.})
    assert mismatch is not None and not mismatch.quarter_turn
    assert mismatch.motor == 'del' and mismatch.axis == 'rows' and mismatch.angles_deg == (-1., 1.)
    assert mismatch.measured_px[1] > mismatch.measured_px[0] and mismatch.model_px[1] < mismatch.model_px[0]
    assert str(mismatch).startswith('The direct beam moved the opposite way to this geometry: del -1 → 1° moved '
                                    'it along the rows from ')


def test_a_beam_moving_across_the_arms_direction_is_a_quarter_turn():
    """del moves the beam along the rows in this geometry; a scan whose beam crossed the columns is a camera a
    quarter turn off."""
    images, _, _ = beam_images(ANGLES, shape=SHAPE, pitches=(PITCH, PITCH), distance=DISTANCE, zero=0.,
                               motion='horizontal')
    result, record = _fit_and_arm(images, 'horizontal')
    mismatch = direct_beam_motion_mismatch(record, result, motor='del', held={'nu': 0.})
    assert mismatch is not None and mismatch.quarter_turn and mismatch.axis == 'columns'
    assert str(mismatch).startswith('The direct beam moved across this geometry: del -1 → 1° moved it along the '
                                    'columns from ')


@pytest.mark.parametrize('turn,agrees', [(90, True), (270, False)])
def test_rsm_direct_beam_scan_1_picks_the_mounting_the_notebook_uses(turn, agrees):
    """RSM/direct_beam scan 1 (del −1…1°, nu 0, the Pilatus 100K .raw of the Stanton beamtime). Image turn 90 is
    the camera ('z-', 'x-') of RSM_process.ipynb (the mounting window's 270°); turn 270 is ('z+', 'x+') (its 90°).
    Both fit; only the first moves the beam with the arm (measured in pyFAI: rows 346 → 140 for both models, the
    data 346 → 140 at turn 90 and 140 → 346 at turn 270)."""
    configured = os.environ.get('XDART_TEST_DATA')
    if not configured:
        pytest.skip('XDART_TEST_DATA is not configured')
    root = Path(configured) / 'RSM'
    if not (root / 'direct_beam').is_file():
        pytest.skip('XDART_TEST_DATA lacks RSM/direct_beam')
    orientation = ImageOrientation(rotation=turn)
    result = calibrate_direct_beam_scan(
        root / 'direct_beam', '1', motor='del', image_dir=root / 'images',
        filename_template='b_thampy_direct_beam_scan{scan}_{point:04d}.raw',
        reader_options={'detector_shape': (195, 487), 'raw_dtype': 'int32', 'raw_header_skip': 0},
        detector='Pilatus100k', motion='vertical', angle_zero_deg=0., beam_pixel=None, orientation=orientation)
    assert result.used.all() and result.distance_m == pytest.approx(1.014649, abs=1e-5)
    record = rough_arm_record(distance_m=result.distance_m, detector='Pilatus100k', beam_px=result.zero_beam_pixel,
                              beam_readings={'nu': 0.}, camera=orientation)
    mismatch = direct_beam_motion_mismatch(record, result, motor='del', held={'nu': 0.})
    if agrees:
        assert mismatch is None
    else:
        assert mismatch is not None and not mismatch.quarter_turn
        assert mismatch.measured_px == pytest.approx((140.1, 346.1), abs=.5)
        assert mismatch.model_px == pytest.approx((346., 140.), abs=1.)
