"""Tests for ssrl_xrd_tools.integrate.single."""

from __future__ import annotations

import numpy as np
import pytest

from ssrl_xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from ssrl_xrd_tools.integrate.single import integrate_1d, integrate_2d, integrate_scan


@pytest.mark.slow
def test_integrate_1d_returns_correct_type(ai_fixture, synthetic_image):
    result = integrate_1d(
        synthetic_image,
        ai_fixture,
        npt=500,
        unit="q_A^-1",
        correctSolidAngle=False,
    )

    assert isinstance(result, IntegrationResult1D)
    assert result.radial.shape == (500,)
    assert result.intensity.shape == (500,)
    assert ("q" in result.unit.lower()) or (result.unit == "q_A^-1")


@pytest.mark.slow
def test_integrate_1d_with_mask(ai_fixture, synthetic_image, synthetic_mask):
    unmasked = integrate_1d(
        synthetic_image,
        ai_fixture,
        npt=500,
        unit="q_A^-1",
        correctSolidAngle=False,
    )
    masked = integrate_1d(
        synthetic_image,
        ai_fixture,
        npt=500,
        unit="q_A^-1",
        mask=synthetic_mask,
        correctSolidAngle=False,
    )

    assert isinstance(masked, IntegrationResult1D)
    assert masked.intensity.shape == (500,)
    assert not np.allclose(masked.intensity, unmasked.intensity, rtol=1e-6, atol=1e-12)


@pytest.mark.slow
def test_integrate_2d_returns_correct_type(ai_fixture, synthetic_image):
    result = integrate_2d(
        synthetic_image,
        ai_fixture,
        npt_rad=200,
        npt_azim=100,
        unit="q_A^-1",
        correctSolidAngle=False,
    )

    assert isinstance(result, IntegrationResult2D)
    assert result.radial.shape == (200,)
    assert result.azimuthal.shape == (100,)
    assert result.intensity.shape == (200, 100)


@pytest.mark.slow
def test_integrate_2d_transpose_convention(ai_fixture, synthetic_image):
    npt_rad, npt_azim = 160, 70
    result = integrate_2d(
        synthetic_image,
        ai_fixture,
        npt_rad=npt_rad,
        npt_azim=npt_azim,
        unit="q_A^-1",
        correctSolidAngle=False,
    )

    assert result.intensity.shape == (npt_rad, npt_azim)
    assert result.intensity.shape != (npt_azim, npt_rad)


@pytest.mark.slow
def test_integrate_scan_sum(ai_fixture, synthetic_image):
    images = np.stack([synthetic_image] * 3, axis=0)

    single = integrate_1d(
        synthetic_image,
        ai_fixture,
        npt=500,
        unit="q_A^-1",
        correctSolidAngle=False,
    )
    summed = integrate_scan(
        images,
        ai_fixture,
        npt=500,
        unit="q_A^-1",
        reduce="sum",
        correctSolidAngle=False,
    )

    assert isinstance(summed, IntegrationResult1D)
    np.testing.assert_allclose(summed.intensity, 3.0 * single.intensity, rtol=1e-6, atol=1e-8)


@pytest.mark.slow
def test_integrate_scan_mean(ai_fixture, synthetic_image):
    images = np.stack([synthetic_image] * 3, axis=0)

    single = integrate_1d(
        synthetic_image,
        ai_fixture,
        npt=500,
        unit="q_A^-1",
        correctSolidAngle=False,
    )
    meaned = integrate_scan(
        images,
        ai_fixture,
        npt=500,
        unit="q_A^-1",
        reduce="mean",
        correctSolidAngle=False,
    )

    assert isinstance(meaned, IntegrationResult1D)
    np.testing.assert_allclose(meaned.intensity, single.intensity, rtol=1e-6, atol=1e-8)


def test_integrate_scan_invalid_reduce(ai_fixture, synthetic_image):
    images = np.stack([synthetic_image, synthetic_image], axis=0)

    with pytest.raises(ValueError, match="reduce must be 'sum' or 'mean'"):
        integrate_scan(
            images,
            ai_fixture,
            npt=200,
            unit="q_A^-1",
            reduce="invalid",
            correctSolidAngle=False,
        )


@pytest.mark.slow
def test_integrate_1d_polarization_factor(ai_fixture, synthetic_image):
    base = integrate_1d(
        synthetic_image,
        ai_fixture,
        npt=500,
        unit="q_A^-1",
        correctSolidAngle=False,
    )
    pol = integrate_1d(
        synthetic_image,
        ai_fixture,
        npt=500,
        unit="q_A^-1",
        polarization_factor=0.99,
        correctSolidAngle=False,
    )

    assert isinstance(pol, IntegrationResult1D)
    assert pol.intensity.shape == (500,)
    assert not np.allclose(pol.intensity, base.intensity, rtol=1e-6, atol=1e-12)


@pytest.mark.slow
def test_integrate_1d_normalization_factor(ai_fixture, synthetic_image):
    norm1 = integrate_1d(
        synthetic_image,
        ai_fixture,
        npt=500,
        unit="q_A^-1",
        normalization_factor=1.0,
        correctSolidAngle=False,
    )
    norm2 = integrate_1d(
        synthetic_image,
        ai_fixture,
        npt=500,
        unit="q_A^-1",
        normalization_factor=2.0,
        correctSolidAngle=False,
    )

    np.testing.assert_allclose(norm2.intensity, norm1.intensity / 2.0, rtol=1e-6, atol=1e-8)
