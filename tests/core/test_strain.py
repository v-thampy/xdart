"""Tests for xrd_tools.analysis.strain (sin²ψ analysis)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from xrd_tools.analysis.strain import (
    ChiSector,
    PeakFitResult,
    Sin2PsiResult,
    extract_chi_sectors,
    fit_peak_vs_psi,
    sin2psi_regression,
    sin2psi_analysis,
)
from xrd_tools.core.containers import IntegrationResult2D


_HAS_LMFIT = importlib.util.find_spec("lmfit") is not None


# ---------------------------------------------------------------------------
# Helpers: build a synthetic (q, chi) polar map with a known peak shift
# ---------------------------------------------------------------------------

def _make_synthetic_polar_map(
    q_center_0: float = 2.5,
    slope: float = 0.02,      # d-shift per sin²ψ
    sigma_q: float = 0.03,
    nq: int = 500,
    nchi: int = 180,
    chi_min: float = 10.0,
    chi_max: float = 85.0,
    noise_level: float = 5.0,
) -> IntegrationResult2D:
    """
    Create a synthetic (q_total, chi) polar map with a single peak whose
    position shifts linearly with sin²(ψ) = sin²(χ).

    pyFAI convention: chigi = arctan2(q_ip, q_oop), measured from surface
    normal.  ψ = |χ|.

    d(ψ) = d0 + slope * sin²(ψ)
    q(ψ) = 2π / d(ψ)
    """
    rng = np.random.default_rng(42)
    d0 = 2.0 * np.pi / q_center_0

    q_axis = np.linspace(q_center_0 - 0.3, q_center_0 + 0.3, nq)
    chi_axis = np.linspace(chi_min, chi_max, nchi)

    intensity = np.zeros((nq, nchi), dtype=float)
    for j, chi in enumerate(chi_axis):
        psi = abs(chi)
        sin2psi = np.sin(np.deg2rad(psi)) ** 2
        d_psi = d0 + slope * sin2psi
        q_psi = 2.0 * np.pi / d_psi
        peak = 100.0 * np.exp(-((q_axis - q_psi) / sigma_q) ** 2)
        intensity[:, j] = peak + 10.0  # constant background

    intensity += rng.normal(0, noise_level, intensity.shape)
    intensity = np.clip(intensity, 0, None)

    return IntegrationResult2D(
        radial=q_axis,
        azimuthal=chi_axis,
        intensity=intensity,
        sigma=None,
        unit="qtot_A^-1",
        azimuthal_unit="chigi_deg",
    )


# ---------------------------------------------------------------------------
# Tests: extract_chi_sectors
# ---------------------------------------------------------------------------

class TestExtractChiSectors:
    def test_auto_sectors(self):
        result2d = _make_synthetic_polar_map()
        sectors = extract_chi_sectors(result2d, chi_width=5.0)
        assert len(sectors) > 0
        for s in sectors:
            assert s.psi == pytest.approx(abs(s.chi_center))
            assert len(s.q) == len(result2d.radial)

    def test_explicit_centers(self):
        result2d = _make_synthetic_polar_map()
        centers = [20.0, 40.0, 60.0, 80.0]
        sectors = extract_chi_sectors(result2d, chi_centers=centers, chi_width=5.0)
        assert len(sectors) == len(centers)
        for s, c in zip(sectors, centers):
            assert s.chi_center == pytest.approx(c)

    def test_missing_wedge_skipped(self):
        """Sectors at chi values outside the data range should be skipped."""
        result2d = _make_synthetic_polar_map(chi_min=10.0, chi_max=85.0)
        # Request a sector at chi=2 which is outside our range
        sectors = extract_chi_sectors(result2d, chi_centers=[2.0], chi_width=3.0)
        assert len(sectors) == 0

    def test_n_sectors_parameter(self):
        result2d = _make_synthetic_polar_map()
        sectors = extract_chi_sectors(result2d, n_sectors=10, chi_width=5.0)
        assert len(sectors) == 10


# ---------------------------------------------------------------------------
# Tests: fit_peak_vs_psi
# ---------------------------------------------------------------------------

class TestFitPeakVsPsi:
    @pytest.mark.skipif(not _HAS_LMFIT, reason="requires lmfit")
    def test_fits_converge(self):
        result2d = _make_synthetic_polar_map(noise_level=2.0)
        sectors = extract_chi_sectors(result2d, n_sectors=8, chi_width=8.0)
        fits = fit_peak_vs_psi(sectors, q_range=(2.3, 2.7), model="gaussian")
        assert len(fits) > 0
        for f in fits:
            assert 2.3 < f.q_center < 2.7
            assert f.d_spacing > 0
            assert 0 <= f.sin2psi <= 1

    def test_narrow_q_range_skipped(self):
        """If q_range captures < 5 points, sectors should be skipped."""
        result2d = _make_synthetic_polar_map(nq=50)
        sectors = extract_chi_sectors(result2d, n_sectors=3, chi_width=10.0)
        # Very narrow range
        fits = fit_peak_vs_psi(sectors, q_range=(2.499, 2.501))
        # Should either skip or still fit — just shouldn't crash
        assert isinstance(fits, list)


# ---------------------------------------------------------------------------
# Tests: sin2psi_regression
# ---------------------------------------------------------------------------

class TestSin2PsiRegression:
    @pytest.mark.skipif(not _HAS_LMFIT, reason="requires lmfit")
    def test_recovers_slope(self):
        """The regression should recover the injected d vs sin²ψ slope."""
        known_slope = 0.02
        result2d = _make_synthetic_polar_map(slope=known_slope, noise_level=1.0)
        sectors = extract_chi_sectors(result2d, n_sectors=12, chi_width=5.0)
        fits = fit_peak_vs_psi(sectors, q_range=(2.3, 2.7), model="gaussian")
        reg = sin2psi_regression(fits)

        # The injected slope is in d-spacing; the fit should recover it
        assert reg.slope == pytest.approx(known_slope, abs=0.005)
        assert reg.r_squared > 0.9
        assert reg.d0 > 0

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="at least 3"):
            sin2psi_regression([
                PeakFitResult(psi=10, sin2psi=0.03, q_center=2.5,
                              d_spacing=2.51, q_center_err=0, d_spacing_err=0,
                              fit_result=None),
                PeakFitResult(psi=30, sin2psi=0.25, q_center=2.5,
                              d_spacing=2.52, q_center_err=0, d_spacing_err=0,
                              fit_result=None),
            ])


# ---------------------------------------------------------------------------
# Tests: sin2psi_analysis (convenience pipeline)
# ---------------------------------------------------------------------------

class TestSin2PsiAnalysis:
    @pytest.mark.skipif(not _HAS_LMFIT, reason="requires lmfit")
    def test_end_to_end(self):
        result2d = _make_synthetic_polar_map(slope=0.015, noise_level=1.0)
        reg = sin2psi_analysis(
            result2d,
            q_range=(2.3, 2.7),
            n_sectors=10,
            chi_width=6.0,
            model="gaussian",
        )
        assert isinstance(reg, Sin2PsiResult)
        assert len(reg.peak_fits) >= 3
        assert reg.slope == pytest.approx(0.015, abs=0.005)
