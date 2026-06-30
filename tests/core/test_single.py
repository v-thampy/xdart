"""Tests for xrd_tools.integrate.single."""

from __future__ import annotations

import numpy as np
import pytest

from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from xrd_tools.integrate.single import integrate_1d, integrate_2d, integrate_scan


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
def test_integrate_2d_nan_fills_empty_bins_not_real_zeros(ai_fixture, synthetic_image):
    """The standard q-χ cake NaN-masks empty (count==0) bins -- uncovered corners
    / masked gaps that CSR fills with 0 -- so they don't average into the cake→1D
    projection or the 2D aggregate, while GENUINE zero-signal bins (count>0,
    intensity 0) are PRESERVED.  Keyed on count, never on the value; mirrors the
    GI _to_result_2d fix."""
    # (1) Real zeros preserved: an all-zero image -> every COVERED bin (count>0)
    # reads 0 and must stay 0, never NaN.  (If value==0 were nulled, the finite
    # set would be empty and the size check would fail.)
    zeros = np.zeros_like(synthetic_image)
    rz = integrate_2d(zeros, ai_fixture, npt_rad=200, npt_azim=100,
                      unit="q_A^-1", correctSolidAngle=False)
    finite = rz.intensity[np.isfinite(rz.intensity)]
    assert finite.size > 0
    np.testing.assert_array_equal(finite, 0.0)
    # (2) Empty bins NaN'd: a real cake has uncovered q-χ corners (count==0) -> NaN,
    # not the CSR 0-fill that would drag the χ-projection.  (Also proves count is
    # exposed for method='csr' -- else the fix would be a no-op and no NaN appear.)
    res = integrate_2d(synthetic_image, ai_fixture, npt_rad=200, npt_azim=100,
                       unit="q_A^-1", correctSolidAngle=False)
    assert np.isnan(res.intensity).any()
    finite_res = res.intensity[np.isfinite(res.intensity)]
    assert finite_res.min() >= -1e-9            # no surviving 0/dummy drag below 0


@pytest.mark.slow
def test_integrate_radial_returns_pooled_chi_profile(ai_fixture, synthetic_image):
    # Azimuthal profile mode (I vs χ over a q band): a chi_deg ±180° axis, and the
    # POOLED quantity -- it must differ from an unweighted cake-row mean-of-means
    # projection (the whole point of using integrate_radial, not a cake collapse).
    from xrd_tools.integrate.single import integrate_radial
    res = integrate_radial(synthetic_image, ai_fixture, npt=180, npt_rad=500,
                           radial_unit="q_A^-1", radial_range=(0.5, 5.0),
                           correctSolidAngle=False)
    assert res.unit == "chi_deg"
    assert res.radial.shape == (180,)
    assert res.radial.min() >= -180.0 - 1e-6 and res.radial.max() <= 180.0 + 1e-6
    assert np.isfinite(res.intensity).any()

    # Pooled (count-weighted) vs cake-row nanmean (mean-of-means): they differ.
    cake = integrate_2d(synthetic_image, ai_fixture, npt_rad=500, npt_azim=180,
                        unit="q_A^-1", radial_range=(0.5, 5.0),
                        correctSolidAngle=False)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        proj = np.nanmean(cake.intensity, axis=0)      # I(χ) cake-row projection
    both = np.isfinite(res.intensity) & np.isfinite(proj)
    assert both.sum() > 0
    assert not np.allclose(res.intensity[both], proj[both], rtol=1e-3)

    # A narrower q band gives a different profile (the band is integrated over).
    res2 = integrate_radial(synthetic_image, ai_fixture, npt=180, npt_rad=500,
                            radial_unit="q_A^-1", radial_range=(1.0, 2.0),
                            correctSolidAngle=False)
    b2 = np.isfinite(res.intensity) & np.isfinite(res2.intensity)
    assert not np.allclose(res.intensity[b2], res2.intensity[b2], rtol=1e-3)


def test_integrate_radial_drops_integrate1d_only_kwargs(ai_fixture, synthetic_image):
    """A 1D-χ reduction carries the full Int arg set, but pyFAI's integrate_radial
    rejects `safe` / `error_model` / `chi_offset` — they must be dropped, not
    forwarded (regression: integrate_radial got an unexpected keyword argument
    'safe' on a non-GI χ-axis integration)."""
    from xrd_tools.integrate.single import integrate_radial
    res = integrate_radial(synthetic_image, ai_fixture, npt=180, npt_rad=500,
                           radial_unit="q_A^-1", radial_range=(0.5, 5.0),
                           correctSolidAngle=False,
                           safe=True, error_model="poisson", chi_offset=90.0)
    assert res.unit == "chi_deg"
    assert res.radial.shape == (180,)
    assert np.isfinite(res.intensity).any()


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
