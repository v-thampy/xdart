# ssrl_xrd_tools/core/containers.py
"""
Shared data containers for calibration and azimuthal integration results.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PONI:
    """
    pyFAI point-of-normal-incidence calibration geometry.

    Parameters
    ----------
    dist : float
        Sample-to-detector distance (m).
    poni1, poni2 : float
        PONI coordinates in detector plane (m).
    rot1, rot2, rot3 : float, optional
        Detector rotations (rad).
    wavelength : float, optional
        Wavelength (m); 0 if unknown.
    detector : str, optional
        pyFAI detector name or identifier.
    """

    dist: float
    poni1: float
    poni2: float
    rot1: float = 0.0
    rot2: float = 0.0
    rot3: float = 0.0
    wavelength: float = 0.0
    detector: str = ""


@dataclass(slots=True)
class IntegrationResult1D:
    """
    Result of 1D azimuthal integration.

    Parameters
    ----------
    radial : ndarray
        Radial axis (2θ or q, depending on ``unit``).
    intensity : ndarray
        Integrated intensity, same length as ``radial``.
    sigma : ndarray or None, optional
        Estimated standard deviation per bin.
    unit : str, optional
        Radial axis unit, e.g. ``"2th_deg"``, ``"q_nm^-1"``.
    """

    radial: np.ndarray
    intensity: np.ndarray
    sigma: np.ndarray | None = None
    unit: str = "2th_deg"

    def __post_init__(self) -> None:
        self.radial = np.asarray(self.radial, dtype=float)
        self.intensity = np.asarray(self.intensity, dtype=float)
        if self.sigma is not None:
            self.sigma = np.asarray(self.sigma, dtype=float)
        if self.radial.shape != self.intensity.shape:
            raise ValueError(
                f"radial shape {self.radial.shape} != intensity shape "
                f"{self.intensity.shape}"
            )
        if self.sigma is not None and self.sigma.shape != self.radial.shape:
            raise ValueError(
                f"sigma shape {self.sigma.shape} != radial shape "
                f"{self.radial.shape}"
            )


@dataclass(slots=True)
class IntegrationResult2D:
    """
    Result of 2D (cake) azimuthal integration.

    Parameters
    ----------
    radial : ndarray
        Radial axis (1D).
    azimuthal : ndarray
        Azimuthal axis (1D), e.g. χ in degrees.
    intensity : ndarray
        2D array of shape ``(len(radial), len(azimuthal))``.
    sigma : ndarray or None, optional
        Per-pixel uncertainty, same shape as ``intensity``.
    unit : str, optional
        Radial axis unit.
    """

    radial: np.ndarray
    azimuthal: np.ndarray
    intensity: np.ndarray
    sigma: np.ndarray | None = None
    unit: str = "2th_deg"

    def __post_init__(self) -> None:
        self.radial = np.asarray(self.radial, dtype=float)
        self.azimuthal = np.asarray(self.azimuthal, dtype=float)
        self.intensity = np.asarray(self.intensity, dtype=float)
        if self.sigma is not None:
            self.sigma = np.asarray(self.sigma, dtype=float)
        if self.intensity.ndim != 2:
            raise ValueError("intensity must be a 2D array")
        nr, naz = len(self.radial), len(self.azimuthal)
        expected = (nr, naz)
        if self.intensity.shape != expected:
            raise ValueError(
                f"intensity shape {self.intensity.shape} != {expected} "
                f"(radial × azimuthal)"
            )
        if self.sigma is not None and self.sigma.shape != self.intensity.shape:
            raise ValueError(
                f"sigma shape {self.sigma.shape} != intensity shape "
                f"{self.intensity.shape}"
            )
