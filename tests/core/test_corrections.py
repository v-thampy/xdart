"""Tests for ssrl_xrd_tools.corrections."""

from __future__ import annotations

import numpy as np
import pytest
from pyFAI.detectors import Detector
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

from ssrl_xrd_tools.corrections import (
    apply_flatfield,
    apply_mask,
    apply_threshold,
    combine_masks,
    correct_image,
    normalize_monitor,
    normalize_stack,
    polarization_correction,
    scale_to_range,
    solid_angle_correction,
    subtract_dark,
)


def _small_ai() -> AzimuthalIntegrator:
    """Create a generic 100x100 AzimuthalIntegrator for correction tests."""
    det = Detector(pixel1=75e-6, pixel2=75e-6, max_shape=(100, 100))
    return AzimuthalIntegrator(
        dist=0.2,
        poni1=0.00375,
        poni2=0.00375,
        wavelength=1e-10,
        detector=det,
    )


def test_subtract_dark():
    image = np.full((10, 10), 100.0)
    dark = np.full((10, 10), 10.0)

    result = subtract_dark(image, dark)
    np.testing.assert_allclose(result, 90.0, rtol=1e-12, atol=1e-12)

    image_nan = image.copy()
    image_nan[0, 0] = np.nan
    result_nan = subtract_dark(image_nan, dark)
    assert np.isnan(result_nan[0, 0])


def test_apply_flatfield():
    image = np.full((10, 10), 100.0)
    flat = np.full((10, 10), 2.0)

    result = apply_flatfield(image, flat)
    np.testing.assert_allclose(result, 50.0, rtol=1e-12, atol=1e-12)

    flat_bad = flat.copy()
    flat_bad[5, 5] = 0.01
    result_bad = apply_flatfield(image, flat_bad, min_flat=0.1)
    assert np.isnan(result_bad[5, 5])


def test_apply_threshold():
    image = np.array([0.0, 50.0, 100.0, 200.0], dtype=float)

    high = apply_threshold(image, threshold=150.0)
    assert np.isnan(high[3])
    np.testing.assert_allclose(high[:3], [0.0, 50.0, 100.0], rtol=1e-12, atol=1e-12)

    high_low = apply_threshold(image, threshold=150.0, low=25.0)
    assert np.isnan(high_low[0])
    assert np.isnan(high_low[3])
    np.testing.assert_allclose(high_low[1:3], [50.0, 100.0], rtol=1e-12, atol=1e-12)


def test_apply_mask():
    image = np.ones((10, 10), dtype=float)
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, 0] = True
    mask[5, 5] = True

    result = apply_mask(image, mask)
    assert np.isnan(result[0, 0])
    assert np.isnan(result[5, 5])
    assert np.isfinite(result[1, 1])


def test_combine_masks():
    mask1 = np.zeros((5, 5), dtype=bool)
    mask2 = np.zeros((5, 5), dtype=bool)
    mask1[0, 0] = True
    mask2[1, 1] = True

    combined = combine_masks(mask1, mask2)
    assert combined is not None
    assert combined[0, 0]
    assert combined[1, 1]

    assert combine_masks(None, None) is None

    only_first = combine_masks(mask1, None)
    assert only_first is not None
    np.testing.assert_array_equal(only_first, mask1)


def test_correct_image_pipeline():
    image = np.array([[100.0, 200.0], [300.0, 400.0]])
    dark = np.array([[10.0, 10.0], [10.0, 10.0]])
    flat = np.array([[2.0, 2.0], [2.0, 0.05]])  # last pixel below min_flat -> NaN
    mask = np.array([[False, True], [False, False]])

    result = correct_image(
        image,
        dark=dark,
        flat=flat,
        mask=mask,
        threshold=100.0,
    )

    expected = np.array([[45.0, np.nan], [np.nan, np.nan]])
    np.testing.assert_allclose(result, expected, equal_nan=True, rtol=1e-12, atol=1e-12)


def test_polarization_correction():
    ai = _small_ai()
    image = np.ones((100, 100), dtype=float)

    result = polarization_correction(image, ai, polarization_factor=0.99)
    assert result.shape == image.shape
    assert result.dtype == np.float64
    assert not np.allclose(result, 1.0)  # correction applied
    center = result[40:60, 40:60]
    assert np.all(np.isfinite(center))


def test_solid_angle_correction():
    ai = _small_ai()
    image = np.ones((100, 100), dtype=float)

    result = solid_angle_correction(image, ai)
    assert result.shape == image.shape
    assert result.dtype == np.float64
    center = result[50, 50]
    corner = result[0, 0]
    assert np.isfinite(center)
    assert np.isfinite(corner)
    assert not np.isclose(center, corner)


def test_normalize_monitor():
    image = np.ones((10, 10), dtype=float) * 1000.0
    result = normalize_monitor(image, monitor=50000.0, reference=100000.0)
    np.testing.assert_allclose(result, 2000.0, rtol=1e-12, atol=1e-12)


def test_normalize_monitor_zero(caplog):
    image = np.ones((10, 10), dtype=float) * 1000.0
    with caplog.at_level("WARNING"):
        result = normalize_monitor(image, monitor=0.0, reference=100000.0)
    np.testing.assert_allclose(result, image, rtol=1e-12, atol=1e-12)
    assert "monitor value" in caplog.text


def test_normalize_stack():
    images = [
        np.ones((4, 4), dtype=float) * 100.0,
        np.ones((4, 4), dtype=float) * 100.0,
        np.ones((4, 4), dtype=float) * 100.0,
    ]
    monitors = [100.0, 200.0, 50.0]

    # reference defaults to mean(monitors) = 350/3
    ref = np.mean(monitors)
    result = normalize_stack(images, monitors)

    assert len(result) == 3
    np.testing.assert_allclose(result[0], 100.0 * ref / 100.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result[1], 100.0 * ref / 200.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result[2], 100.0 * ref / 50.0, rtol=1e-12, atol=1e-12)


def test_scale_to_range():
    intensity = np.linspace(10.0, 100.0, 50)
    scaled = scale_to_range(intensity, vmin=0.0, vmax=1.0)
    np.testing.assert_allclose(np.nanmin(scaled), 0.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.nanmax(scaled), 1.0, rtol=1e-12, atol=1e-12)
