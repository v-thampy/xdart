"""Tests for xrd_tools.analysis.fitting.phase_fitting.

Uses synthetic two-phase data (FCC Au + FCC Cu) generated from known
lattice parameters and pseudo-Voigt profiles.  The fitter should recover
the input parameters within tolerance.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from xrd_tools.analysis.fitting.background import snip_1d
from xrd_tools.analysis.phase import PhaseModel, PeakData

_HAS_LMFIT = importlib.util.find_spec("lmfit") is not None
pytestmark = pytest.mark.skipif(not _HAS_LMFIT, reason="requires lmfit")

if _HAS_LMFIT:
    from xrd_tools.analysis.fitting.phase_fitting import (
        PhaseFitter,
        MultiPhaseResult,
        _metric_tensor,
        _q_from_hkl,
        _pseudo_voigt,
        _caglioti_sigma,
    )


# ---------------------------------------------------------------------------
# Helpers: build synthetic phases without pymatgen
# ---------------------------------------------------------------------------

def _make_fcc_phase(name: str, a: float) -> PhaseModel:
    """Create a synthetic FCC phase with known peak positions.

    For FCC, allowed reflections have all-even or all-odd Miller indices.
    We include (111), (200), (220), (311), (222).
    """
    hkls = [(1, 1, 1), (2, 0, 0), (2, 2, 0), (3, 1, 1), (2, 2, 2)]
    # Relative intensities (roughly physical for FCC)
    rel_intensities = [100.0, 47.0, 22.0, 24.0, 7.0]

    phase = PhaseModel(name=name, structure=None)
    phase.peaks = []
    for hkl, intensity in zip(hkls, rel_intensities):
        h, k, l = hkl
        d = a / np.sqrt(h**2 + k**2 + l**2)
        q = 2 * np.pi / d
        phase.peaks.append(PeakData(q=q, intensity=intensity, hkl=hkl, d_spacing=d))
    return phase


def _synthetic_pattern(
    q: np.ndarray,
    phases: list[PhaseModel],
    scales: list[float],
    sigma: float = 0.015,
    fraction: float = 0.5,
    q_shift: float = 0.0,
    noise_level: float = 0.5,
    bg_level: float = 5.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate a synthetic multi-phase XRD pattern."""
    rng = np.random.default_rng(seed)
    y = np.full_like(q, bg_level)
    x_shifted = q - q_shift

    for phase, scale in zip(phases, scales):
        max_int = max(pk.intensity for pk in phase.peaks) if phase.peaks else 1.0
        for pk in phase.peaks:
            amp = (pk.intensity / max_int) * scale
            y += _pseudo_voigt(x_shifted, pk.q, amp, sigma, fraction)

    y += rng.normal(0, noise_level, size=len(q))
    y = np.clip(y, 0, None)
    return y


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def q_axis():
    return np.linspace(1.5, 8.0, 2000)


@pytest.fixture
def au_phase():
    return _make_fcc_phase("Au", a=4.0782)


@pytest.fixture
def cu_phase():
    return _make_fcc_phase("Cu", a=3.6149)


@pytest.fixture
def two_phase_data(q_axis, au_phase, cu_phase):
    """Synthetic Au+Cu pattern with known parameters."""
    y = _synthetic_pattern(
        q_axis,
        [au_phase, cu_phase],
        scales=[50.0, 30.0],
        sigma=0.015,
        fraction=0.5,
        q_shift=0.005,
        noise_level=0.3,
        bg_level=3.0,
    )
    return y


# ---------------------------------------------------------------------------
# Unit tests: analytical helpers
# ---------------------------------------------------------------------------

class TestMetricTensor:
    def test_cubic(self):
        """For cubic a=b=c, α=β=γ=90, d(hkl) = a/sqrt(h²+k²+l²)."""
        a = 4.0
        G = _metric_tensor(a, a, a, 90, 90, 90)
        hkl = np.array([[1, 1, 1]])
        q = _q_from_hkl(hkl, G)
        expected_d = a / np.sqrt(3)
        expected_q = 2 * np.pi / expected_d
        np.testing.assert_allclose(q, [expected_q], rtol=1e-6)

    def test_tetragonal(self):
        """Tetragonal: a=b≠c."""
        a, c = 3.0, 5.0
        G = _metric_tensor(a, a, c, 90, 90, 90)
        # (001): d = c
        hkl = np.array([[0, 0, 1]])
        q = _q_from_hkl(hkl, G)
        np.testing.assert_allclose(q, [2 * np.pi / c], rtol=1e-6)

        # (100): d = a
        hkl = np.array([[1, 0, 0]])
        q = _q_from_hkl(hkl, G)
        np.testing.assert_allclose(q, [2 * np.pi / a], rtol=1e-6)

    def test_multiple_hkl(self):
        a = 4.0782  # Au
        G = _metric_tensor(a, a, a, 90, 90, 90)
        hkl = np.array([[1, 1, 1], [2, 0, 0], [2, 2, 0]])
        q = _q_from_hkl(hkl, G)
        for i, (h, k, l) in enumerate(hkl):
            d = a / np.sqrt(h**2 + k**2 + l**2)
            np.testing.assert_allclose(q[i], 2 * np.pi / d, rtol=1e-6)


class TestPseudoVoigt:
    def test_positive(self):
        x = np.linspace(-1, 1, 100)
        y = _pseudo_voigt(x, 0.0, 1.0, 0.1, 0.5)
        assert np.all(y >= 0)

    def test_peak_at_center(self):
        x = np.linspace(-2, 2, 1000)
        y = _pseudo_voigt(x, 0.0, 1.0, 0.1, 0.5)
        assert np.argmax(y) == np.argmin(np.abs(x))

    def test_pure_gaussian(self):
        """fraction=0 → pure Gaussian."""
        x = np.linspace(-1, 1, 500)
        y = _pseudo_voigt(x, 0.0, 1.0, 0.1, 0.0)
        # Gaussian should have faster tail decay than Lorentzian
        y_lor = _pseudo_voigt(x, 0.0, 1.0, 0.1, 1.0)
        # At large |x|, Gaussian tails < Lorentzian tails
        assert y[-1] < y_lor[-1]


class TestCagliotiSigma:
    def test_positive(self):
        sig = _caglioti_sigma(3.0, 1e-4, 0.0, 4e-4)
        assert sig > 0

    def test_array(self):
        q = np.array([2.0, 4.0, 6.0])
        sig = _caglioti_sigma(q, 1e-4, 0.0, 4e-4)
        assert sig.shape == (3,)
        # Width should increase with Q when U > 0
        assert sig[-1] > sig[0]


# ---------------------------------------------------------------------------
# Integration tests: PhaseFitter
# ---------------------------------------------------------------------------

class TestPhaseFitter:
    def test_init_from_arrays(self, q_axis, two_phase_data):
        fitter = PhaseFitter(q_axis, two_phase_data)
        assert fitter.x.shape == q_axis.shape
        assert fitter.background.shape == q_axis.shape

    def test_add_phase(self, q_axis, two_phase_data, au_phase, cu_phase):
        fitter = PhaseFitter(q_axis, two_phase_data)
        fitter.add_phase(au_phase)
        fitter.add_phase(cu_phase)
        assert len(fitter.phases) == 2
        assert fitter._hkl_arrays[0].shape[1] == 3

    def test_add_phase_empty_raises(self, q_axis, two_phase_data):
        fitter = PhaseFitter(q_axis, two_phase_data)
        empty_phase = PhaseModel(name="empty")
        with pytest.raises(ValueError, match="no peaks"):
            fitter.add_phase(empty_phase)

    def test_build_parameters(self, q_axis, two_phase_data, au_phase, cu_phase):
        fitter = PhaseFitter(q_axis, two_phase_data)
        fitter.add_phase(au_phase)
        fitter.add_phase(cu_phase)
        params = fitter.build_parameters()
        assert "q_shift" in params
        assert "p0_scale" in params
        assert "p1_scale" in params
        assert "p0_U" in params  # Caglioti
        assert "p0_fraction" in params

    def test_build_parameters_no_caglioti(self, q_axis, two_phase_data, au_phase):
        fitter = PhaseFitter(q_axis, two_phase_data)
        fitter.add_phase(au_phase)
        params = fitter.build_parameters(caglioti=False)
        assert "p0_sigma" in params
        assert "p0_U" not in params

    def test_march_dollase_refuses_structureless_fixed_q(
        self, q_axis, two_phase_data, au_phase,
    ):
        fitter = PhaseFitter(q_axis, two_phase_data)
        fitter.add_phase(au_phase)
        with pytest.raises(ValueError, match="structure-backed phase"):
            fitter.build_parameters(texture="march_dollase")

    def test_march_dollase_allowed_with_lattice_metric(self, q_axis, two_phase_data):
        class _Lattice:
            a = 4.0782
            b = 4.0782
            c = 4.0782
            alpha = 90.0
            beta = 90.0
            gamma = 90.0

        class _Structure:
            lattice = _Lattice()

        phase = _make_fcc_phase("Au-structured", a=4.0782)
        phase.structure = _Structure()
        fitter = PhaseFitter(q_axis, two_phase_data)
        fitter.add_phase(phase)
        params = fitter.build_parameters(texture="march_dollase")
        assert "p0_march_r" in params

    def test_eval_model(self, q_axis, two_phase_data, au_phase, cu_phase):
        fitter = PhaseFitter(q_axis, two_phase_data)
        fitter.add_phase(au_phase)
        fitter.add_phase(cu_phase)
        params = fitter.build_parameters()
        y_model = fitter.eval_model(params)
        assert y_model.shape == q_axis.shape
        # Model should be positive
        assert np.all(y_model >= 0)

    def test_fit_recovers_scales(self, q_axis, au_phase, cu_phase):
        """The fitter should recover that Au has a larger scale than Cu."""
        y = _synthetic_pattern(
            q_axis,
            [au_phase, cu_phase],
            scales=[80.0, 20.0],
            sigma=0.015,
            fraction=0.5,
            q_shift=0.0,
            noise_level=0.1,
            bg_level=2.0,
        )
        fitter = PhaseFitter(q_axis, y)
        fitter.add_phase(au_phase)
        fitter.add_phase(cu_phase)
        result = fitter.fit(caglioti=False)

        assert result.success
        # Au should have a larger scale
        assert result.phase_scale(0) > result.phase_scale(1)
        # Phase fractions should sum to 1
        fracs = result.phase_fractions()
        np.testing.assert_allclose(sum(fracs.values()), 1.0, atol=1e-10)

    def test_fit_with_nans_succeeds(self, q_axis, au_phase, cu_phase):
        """NaNs in the data must be ignored, not crash the fit (CARRYOVER 3).

        lmfit's default nan_policy is 'raise', so a plain .fit() over data with
        NaN raised ValueError("NaN values detected ...").  PhaseFitter.fit now
        defaults to nan_policy='omit'."""
        y = _synthetic_pattern(
            q_axis, [au_phase, cu_phase], scales=[80.0, 20.0], sigma=0.015,
            fraction=0.5, q_shift=0.0, noise_level=0.1, bg_level=2.0,
        )
        y_nan = y.copy()
        y_nan[[100, 500, 1500]] = np.nan        # scattered dummy/masked points
        fitter = PhaseFitter(q_axis, y_nan)
        fitter.add_phase(au_phase)
        fitter.add_phase(cu_phase)
        result = fitter.fit(caglioti=False)     # no nan_policy override
        assert result.success
        assert result.phase_scale(0) > result.phase_scale(1)
        assert np.isfinite(result.lmfit_result.redchi)

    def test_nan_policy_omit_is_noop_for_clean_data(self, q_axis, au_phase, cu_phase):
        """The new default must not change results when data has no NaNs."""
        y = _synthetic_pattern(
            q_axis, [au_phase, cu_phase], scales=[80.0, 20.0], sigma=0.015,
            fraction=0.5, q_shift=0.0, noise_level=0.1, bg_level=2.0,
        )

        def _fit(nan_policy):
            f = PhaseFitter(q_axis, y)
            f.add_phase(au_phase)
            f.add_phase(cu_phase)
            return f.fit(caglioti=False, nan_policy=nan_policy)

        r_omit = _fit("omit")
        r_raise = _fit("raise")                 # also proves the override binds
        assert r_omit.success and r_raise.success
        np.testing.assert_allclose(
            r_omit.lmfit_result.redchi, r_raise.lmfit_result.redchi, rtol=0, atol=0)
        np.testing.assert_allclose(
            [r_omit.phase_scale(0), r_omit.phase_scale(1)],
            [r_raise.phase_scale(0), r_raise.phase_scale(1)], rtol=0, atol=0)

    def test_fit_recovers_q_shift(self, q_axis, au_phase):
        """Applied Q-shift should be recovered."""
        true_shift = 0.01
        y = _synthetic_pattern(
            q_axis,
            [au_phase],
            scales=[50.0],
            sigma=0.015,
            q_shift=true_shift,
            noise_level=0.1,
            bg_level=2.0,
        )
        fitter = PhaseFitter(q_axis, y)
        fitter.add_phase(au_phase)
        result = fitter.fit(caglioti=False, q_shift_bound=0.05)

        assert result.success
        # Should recover shift within ±0.005
        np.testing.assert_allclose(result.q_shift, true_shift, atol=0.005)

    def test_no_phases_raises(self, q_axis, two_phase_data):
        # When no phases / no fit-background / no amorphous component
        # are configured, build_model raises ValueError with the
        # "No fit content" message (broadened from the old "No phases"
        # by commit 219a4c7 — fit-background-only baseline fits are
        # now legal, so the error is about *any* fit content missing).
        fitter = PhaseFitter(q_axis, two_phase_data)
        with pytest.raises(ValueError, match="No fit content"):
            fitter.fit()

    def test_summary(self, q_axis, two_phase_data, au_phase, cu_phase):
        fitter = PhaseFitter(q_axis, two_phase_data)
        fitter.add_phase(au_phase)
        fitter.add_phase(cu_phase)
        result = fitter.fit(caglioti=False)
        txt = result.summary()
        assert "Au" in txt
        assert "Cu" in txt
        assert "Q-shift" in txt


class TestMultiPhaseResult:
    def test_phase_fractions_sum(self, q_axis, two_phase_data, au_phase, cu_phase):
        fitter = PhaseFitter(q_axis, two_phase_data)
        fitter.add_phase(au_phase)
        fitter.add_phase(cu_phase)
        result = fitter.fit(caglioti=False)
        fracs = result.phase_fractions()
        assert len(fracs) == 2
        np.testing.assert_allclose(sum(fracs.values()), 1.0, atol=1e-10)

    def test_redchi(self, q_axis, two_phase_data, au_phase, cu_phase):
        fitter = PhaseFitter(q_axis, two_phase_data)
        fitter.add_phase(au_phase)
        fitter.add_phase(cu_phase)
        result = fitter.fit(caglioti=False)
        assert result.redchi > 0
        assert np.isfinite(result.redchi)
