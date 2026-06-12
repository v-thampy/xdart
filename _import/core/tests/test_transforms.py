"""Tests for ssrl_xrd_tools.transforms unit conversion functions."""

from __future__ import annotations

import numpy as np
import pytest

from ssrl_xrd_tools.transforms import (
    HC_KEV_ANGSTROM,
    d_to_q,
    energy_to_wavelength,
    q_to_d,
    q_to_tth,
    tth_to_q,
    wavelength_to_energy,
)


# ---------------------------------------------------------------------------
# energy ↔ wavelength
# ---------------------------------------------------------------------------

class TestEnergyWavelength:
    def test_canonical_value(self):
        """12.398 keV → 1.0 Å (hc in keV·Å ≈ 12.398)."""
        lam = energy_to_wavelength(HC_KEV_ANGSTROM)
        np.testing.assert_allclose(lam, 1.0, rtol=1e-10)

    def test_roundtrip_scalar(self):
        for energy in (5.0, 10.0, 12.398, 20.0, 100.0):
            recovered = wavelength_to_energy(energy_to_wavelength(energy))
            np.testing.assert_allclose(recovered, energy, rtol=1e-10,
                                       err_msg=f"failed at energy={energy}")

    def test_roundtrip_array(self):
        energies = np.array([5.0, 10.0, 12.398, 25.0, 50.0])
        recovered = wavelength_to_energy(energy_to_wavelength(energies))
        np.testing.assert_allclose(recovered, energies, rtol=1e-10)

    def test_symmetry(self):
        """energy_to_wavelength and wavelength_to_energy are the same operation."""
        # Both are HC / x, so they must be exactly symmetric.
        e = 17.5
        lam = energy_to_wavelength(e)
        np.testing.assert_allclose(wavelength_to_energy(lam), e, rtol=1e-12)

    def test_returns_ndarray_for_array_input(self):
        result = energy_to_wavelength(np.array([10.0, 12.0]))
        assert isinstance(result, np.ndarray)

    def test_returns_scalar_for_scalar_input(self):
        result = energy_to_wavelength(12.398)
        # numpy scalar, not a Python float, but must not be an ndarray with shape
        assert np.ndim(result) == 0


# ---------------------------------------------------------------------------
# q ↔ 2θ
# ---------------------------------------------------------------------------

class TestQTth:
    def test_roundtrip_scalar(self):
        """tth_to_q(q_to_tth(q, E), E) ≈ q for representative values."""
        energy = 12.0
        for q in (0.5, 1.0, 3.0, 5.0, 8.0):
            tth = q_to_tth(q, energy)
            recovered = tth_to_q(tth, energy)
            np.testing.assert_allclose(recovered, q, rtol=1e-6,
                                       err_msg=f"failed at q={q}, E={energy}")

    def test_roundtrip_array(self):
        energy = 12.0
        q_arr = np.linspace(0.1, 8.0, 200)
        recovered = tth_to_q(q_to_tth(q_arr, energy), energy)
        np.testing.assert_allclose(recovered, q_arr, rtol=1e-6)

    def test_roundtrip_multiple_energies(self):
        for energy in (8.0, 12.398, 17.0, 25.0):
            q = np.linspace(0.1, 4.0 * np.pi / energy_to_wavelength(energy) * 0.9, 50)
            recovered = tth_to_q(q_to_tth(q, energy), energy)
            np.testing.assert_allclose(recovered, q, rtol=1e-6,
                                       err_msg=f"failed at energy={energy}")

    def test_known_value_60_degrees(self):
        """
        At λ = 1 Å (E ≈ 12.398 keV), d = 1 Å → q = 2π Å⁻¹.
        Bragg: 2d·sin θ = λ  →  sin θ = 0.5  →  θ = 30°  →  2θ = 60°.
        """
        energy = HC_KEV_ANGSTROM      # λ = 1 Å
        q = 2.0 * np.pi               # d = 2π/q = 1 Å
        tth = q_to_tth(q, energy)
        np.testing.assert_allclose(tth, 60.0, rtol=1e-6)

    def test_known_value_tth_to_q(self):
        """Inverse of the 60° case: 2θ = 60° → q = 2π at λ = 1 Å."""
        energy = HC_KEV_ANGSTROM
        q = tth_to_q(60.0, energy)
        np.testing.assert_allclose(q, 2.0 * np.pi, rtol=1e-6)

    def test_q_zero_gives_tth_zero(self):
        np.testing.assert_allclose(q_to_tth(0.0, 12.0), 0.0, atol=1e-10)

    def test_tth_zero_gives_q_zero(self):
        np.testing.assert_allclose(tth_to_q(0.0, 12.0), 0.0, atol=1e-10)

    def test_array_input_returns_ndarray(self):
        q_arr = np.array([1.0, 2.0, 3.0])
        tth = q_to_tth(q_arr, 12.0)
        assert isinstance(tth, np.ndarray)
        assert tth.shape == (3,)

    def test_monotonically_increasing(self):
        """q and 2θ must be strictly monotonically related."""
        q = np.linspace(0.1, 5.0, 100)
        tth = q_to_tth(q, 12.0)
        assert np.all(np.diff(tth) > 0)


# ---------------------------------------------------------------------------
# d ↔ q
# ---------------------------------------------------------------------------

class TestDQ:
    def test_roundtrip_scalar(self):
        for d in (0.5, 1.0, 2.0, 3.5, 10.0):
            recovered = q_to_d(d_to_q(d))
            np.testing.assert_allclose(recovered, d, rtol=1e-10,
                                       err_msg=f"failed at d={d}")

    def test_roundtrip_array(self):
        d_arr = np.linspace(0.5, 10.0, 200)
        recovered = q_to_d(d_to_q(d_arr))
        np.testing.assert_allclose(recovered, d_arr, rtol=1e-10)

    def test_d1_gives_2pi(self):
        """q = 2π / d; for d = 1 Å → q = 2π Å⁻¹."""
        np.testing.assert_allclose(d_to_q(1.0), 2.0 * np.pi, rtol=1e-10)

    def test_q2pi_gives_d1(self):
        """q_to_d(2π) = 1 Å."""
        np.testing.assert_allclose(q_to_d(2.0 * np.pi), 1.0, rtol=1e-10)

    def test_symmetry(self):
        """d_to_q and q_to_d are the same operation (2π / x)."""
        x = 3.14
        np.testing.assert_allclose(d_to_q(x), q_to_d(x), rtol=1e-12)

    def test_array_input_returns_ndarray(self):
        d_arr = np.array([1.0, 2.0, 3.0])
        result = d_to_q(d_arr)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)

    def test_inversely_proportional(self):
        """Doubling d halves q."""
        d = 2.5
        np.testing.assert_allclose(d_to_q(2 * d), d_to_q(d) / 2, rtol=1e-10)


# ---------------------------------------------------------------------------
# Cross-function consistency
# ---------------------------------------------------------------------------

class TestCrossConsistency:
    def test_q_tth_d_triangle(self):
        """
        The three conversions must be internally consistent:
        q → 2θ → q  and  q → d → q  should agree.
        """
        energy = 12.0
        d_values = np.array([1.0, 2.0, 3.5, 5.0])
        q_from_d = d_to_q(d_values)
        tth = q_to_tth(q_from_d, energy)
        q_from_tth = tth_to_q(tth, energy)
        np.testing.assert_allclose(q_from_tth, q_from_d, rtol=1e-6)

    def test_bragg_law_consistency(self):
        """
        Bragg's law: 2d sin θ = λ.
        Verify that d-spacing recovered from 2θ matches the input d.
        """
        energy = HC_KEV_ANGSTROM     # λ = 1 Å
        lam = energy_to_wavelength(energy)
        for d in (1.0, 2.0, 3.0):
            tth = q_to_tth(d_to_q(d), energy)
            theta_rad = np.deg2rad(tth / 2.0)
            d_bragg = lam / (2.0 * np.sin(theta_rad))
            np.testing.assert_allclose(d_bragg, d, rtol=1e-6,
                                       err_msg=f"Bragg consistency failed at d={d}")
