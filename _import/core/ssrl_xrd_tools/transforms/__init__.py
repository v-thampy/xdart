"""
Unit conversion helpers used throughout ``ssrl_xrd_tools``.

Implemented here:
- ``q_to_tth`` / ``tth_to_q``
- ``d_to_q`` / ``q_to_d``
- ``energy_to_wavelength`` / ``wavelength_to_energy``
"""

from __future__ import annotations

import numpy as np

HC_KEV_ANGSTROM = 12.398


def energy_to_wavelength(energy_keV: float | np.ndarray) -> float | np.ndarray:
    """
    Convert photon energy to wavelength.

    Parameters
    ----------
    energy_keV : float or ndarray
        Photon energy in keV.

    Returns
    -------
    float or np.ndarray
        Wavelength in Angstroms.
    """
    return HC_KEV_ANGSTROM / np.asarray(energy_keV, dtype=float)


def wavelength_to_energy(wavelength_A: float | np.ndarray) -> float | np.ndarray:
    """
    Convert wavelength to photon energy.

    Parameters
    ----------
    wavelength_A : float or ndarray
        Wavelength in Angstroms.

    Returns
    -------
    float or np.ndarray
        Photon energy in keV.
    """
    return HC_KEV_ANGSTROM / np.asarray(wavelength_A, dtype=float)


def q_to_tth(q: float | np.ndarray, energy_keV: float) -> float | np.ndarray:
    """
    Convert momentum transfer to 2theta.

    Uses the project-wide formula from ``CLAUDE.md``:
    ``tth = 2 * rad2deg(arcsin(12.398 / (4 * pi * energy_keV) * q))``.

    Parameters
    ----------
    q : float or ndarray
        Momentum transfer in Angstrom^-1.
    energy_keV : float
        Photon energy in keV.

    Returns
    -------
    float or np.ndarray
        2theta in degrees.
    """
    q_arr = np.asarray(q, dtype=float)
    arg = HC_KEV_ANGSTROM * q_arr / (4.0 * np.pi * float(energy_keV))
    return 2.0 * np.rad2deg(np.arcsin(arg))


def tth_to_q(tth_deg: float | np.ndarray, energy_keV: float) -> float | np.ndarray:
    """
    Convert 2theta to momentum transfer.

    Parameters
    ----------
    tth_deg : float or ndarray
        2theta in degrees.
    energy_keV : float
        Photon energy in keV.

    Returns
    -------
    float or np.ndarray
        Momentum transfer in Angstrom^-1.
    """
    tth_rad = np.deg2rad(np.asarray(tth_deg, dtype=float))
    wavelength = energy_to_wavelength(float(energy_keV))
    return (4.0 * np.pi / wavelength) * np.sin(tth_rad / 2.0)


def d_to_q(d_A: float | np.ndarray) -> float | np.ndarray:
    """
    Convert d-spacing to momentum transfer.

    Parameters
    ----------
    d_A : float or ndarray
        d-spacing in Angstroms.

    Returns
    -------
    float or np.ndarray
        Momentum transfer in Angstrom^-1.
    """
    return 2.0 * np.pi / np.asarray(d_A, dtype=float)


def q_to_d(q: float | np.ndarray) -> float | np.ndarray:
    """
    Convert momentum transfer to d-spacing.

    Parameters
    ----------
    q : float or ndarray
        Momentum transfer in Angstrom^-1.

    Returns
    -------
    float or np.ndarray
        d-spacing in Angstroms.
    """
    return 2.0 * np.pi / np.asarray(q, dtype=float)


__all__ = [
    "HC_KEV_ANGSTROM",
    "d_to_q",
    "energy_to_wavelength",
    "q_to_d",
    "q_to_tth",
    "tth_to_q",
    "wavelength_to_energy",
]
