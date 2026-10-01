"""The shared per-frame preparation (AUD-9 stage 2b C3, design §4): Stitch's xrayutilities path and RSM's
gridding prepare a frame with the same functions, and Stitch's dark and flat are admitted by Int's owners
at run time too (Codex Stitch/Int reuse audit R4)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from xrd_tools.corrections.frame_preparation import (
    FramePreparationRefused,
    correction_image,
    frame_valid,
    frame_weights,
    polarization_axes,
    polarization_weight,
    prepare_frame_signal,
)


def test_a_dark_and_a_flat_are_admitted_by_ints_owners_with_neutral_values_where_unusable():
    dark = np.array([[1.0, np.nan], [2.0, 3.0]])
    values, invalid = correction_image(dark, "dark")
    np.testing.assert_array_equal(values, [[1.0, 0.0], [2.0, 3.0]])
    np.testing.assert_array_equal(invalid, [[False, True], [False, False]])
    flat = np.array([[1.0, 0.0], [0.05, 1.1]])  # 0 is not positive; 0.05 is below 10 % of the median
    values, invalid = correction_image(flat, "flat", min_fraction=0.1)
    np.testing.assert_array_equal(values, [[1.0, 1.0], [1.0, 1.1]])
    np.testing.assert_array_equal(invalid, [[False, True], [True, False]])
    usable = np.ones((2, 2))
    assert correction_image(usable, "flat")[0] is usable  # no copy when nothing is unusable
    assert correction_image(None, "dark") == (None, None)


def test_an_all_nan_dark_is_refused_as_int_refuses_it():
    """Int's ``prepare_dark``: "dark has no finite pixel" (Stitch's run-time preparation used to mask it all,
    a blank result)."""
    with pytest.raises(ValueError, match="dark has no finite pixel"):
        correction_image(np.full((2, 2), np.nan), "dark")


def test_an_xrayutilities_stitch_refuses_an_all_nan_dark_at_run_time(tmp_path):
    """Behavioural fail-before: on the parent the run's dark preparation masked every pixel."""
    from xrd_tools.core.scan import ScanFrame
    from xrd_tools.integrate.xu_stitch import run_xu_hist_stitch_1d
    from xrd_tools.sources.memory import MemoryFrameSource
    from tests.core.test_xu_stitch import _receipt
    receipt = _receipt(tmp_path)
    frames = [ScanFrame(index=0, image=np.full((195, 1475), 20, dtype=np.uint16),
                        metadata={"del": 14.0, "nu": -10.0, "mon": 1.0})]
    with pytest.raises(ValueError, match="dark has no finite pixel"):
        run_xu_hist_stitch_1d(receipt, MemoryFrameSource(frames), frame_indices=(0,), npt=32, monitor_key="mon",
                              max_frame_bytes=16 * 1024 * 1024, dark=np.full((195, 1475), np.nan))


def test_the_signal_takes_the_dark_then_the_divisor_in_place_and_refuses_overflow():
    image = np.array([[10.0, 20.0]])
    result = prepare_frame_signal(image, dark=np.array([[2.0, 4.0]]), divisor=2.0)
    assert result is image
    np.testing.assert_array_equal(image, [[4.0, 8.0]])
    with pytest.raises(FramePreparationRefused) as refused:
        prepare_frame_signal(np.array([[1e308]]), dark=None, divisor=1e-10)
    assert refused.value.code == "NUMERIC_OVERFLOW"


def test_valid_pixels_and_weights():
    detector = np.array([[True, False, False, False]])
    mask = np.array([[False, True, False, False]])
    image = np.array([[1.0, 1.0, np.nan, 1.0]])
    solid = np.array([[1.0, 1.0, 1.0, 0.0]])
    np.testing.assert_array_equal(frame_valid(detector, mask=mask, finite=(image,), positive=(solid,)),
                                  [[False, False, False, False]])
    np.testing.assert_array_equal(frame_valid(np.zeros((1, 2), bool), finite=(np.ones((1, 2)),)), [[True, True]])
    assert frame_weights(solid) is solid
    weights = frame_weights(np.array([2.0, 3.0]), polarization=np.array([0.5, 0.5]), flat=np.array([1.0, 2.0]))
    np.testing.assert_array_equal(weights, [1.0, 3.0])


@pytest.mark.parametrize("factor", (0.0, 0.5, 0.99))
def test_the_polarization_weight_is_pyfais_in_closed_form(factor):
    """``P = ½(1 + cos²2θ − f·cos2χ·sin²2θ)``, χ from the horizontal: 1 along the beam, (1 − f)/2 at 2θ = 90°
    in the horizontal plane and (1 + f)/2 in the vertical one."""
    beam, vertical = polarization_axes([0.0, 1.0, 0.0], "z+")
    wavelength = 1.0
    k = 2 * math.pi / wavelength

    def q_of(direction):
        direction = np.asarray(direction, dtype=np.float64)
        return k * direction - k * beam
    horizontal = np.cross(beam, vertical)
    cases = ((beam, 1.0), (horizontal, (1 - factor) / 2), (vertical, (1 + factor) / 2))
    q = np.array([q_of(direction) for direction, _expected in cases])
    weights = polarization_weight(q[:, 0], q[:, 1], q[:, 2], factor=factor, wavelength_A=wavelength, beam=beam,
                                  vertical=vertical)
    np.testing.assert_allclose(weights, [expected for _direction, expected in cases], rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="not elastic"):
        polarization_weight(np.array([0.0]), np.array([0.0]), np.array([k]), factor=factor, wavelength_A=wavelength,
                            beam=beam, vertical=vertical)
    with pytest.raises(ValueError, match="perpendicular"):
        polarization_axes([0.0, 1.0, 0.0], "y+")
