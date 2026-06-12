"""
Beam- and geometry-related correction helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator


def _as_float_image(image: np.ndarray) -> np.ndarray:
    """Return image as float64 ndarray."""
    return np.asarray(image, dtype=float)


def polarization_correction(
    image: np.ndarray,
    ai: AzimuthalIntegrator,
    polarization_factor: float = 0.99,
) -> np.ndarray:
    """
    Apply polarization correction using pyFAI's per-pixel polarization array.

    Parameters
    ----------
    image : ndarray
        Input detector image.
    ai : AzimuthalIntegrator
        pyFAI azimuthal integrator used to compute the polarization array.
    polarization_factor : float, optional
        Horizontal polarization factor (typical synchrotron values are close to
        1.0; default 0.99).

    Returns
    -------
    np.ndarray
        Polarization-corrected image in ``float64``.

    Notes
    -----
    This is an alternative to passing ``polarization_factor`` directly to
    pyFAI integration functions. Do not apply both corrections simultaneously.
    """
    image_arr = _as_float_image(image)
    pol = np.asarray(
        ai.polarization(shape=image_arr.shape, factor=float(polarization_factor)),
        dtype=float,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        corrected = image_arr / pol
    corrected[pol < 1e-10] = np.nan
    corrected[np.isnan(image_arr)] = np.nan
    return corrected


def solid_angle_correction(
    image: np.ndarray,
    ai: AzimuthalIntegrator,
) -> np.ndarray:
    """
    Apply solid-angle correction using pyFAI's per-pixel solid-angle array.

    Parameters
    ----------
    image : ndarray
        Input detector image.
    ai : AzimuthalIntegrator
        pyFAI azimuthal integrator used to compute the solid-angle array.

    Returns
    -------
    np.ndarray
        Solid-angle-corrected image in ``float64``.

    Notes
    -----
    pyFAI ``integrate1d`` / ``integrate2d`` apply solid-angle correction by
    default (``correctSolidAngle=True``). This helper is useful when a
    pre-corrected image is required for direct image analysis or RSM workflows.
    """
    image_arr = _as_float_image(image)
    sa = np.asarray(ai.solidAngleArray(shape=image_arr.shape), dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        corrected = image_arr / sa
    corrected[sa < 1e-10] = np.nan
    corrected[np.isnan(image_arr)] = np.nan
    return corrected


def absorption_correction(
    image: np.ndarray,
    mu_t: float,
    ai: AzimuthalIntegrator,
) -> np.ndarray:
    """
    Apply a simplified absorption correction factor from scattering angle.

    Parameters
    ----------
    image : ndarray
        Input detector image.
    mu_t : float
        Effective absorption-thickness product (μ·t) for the sample.
    ai : AzimuthalIntegrator
        pyFAI azimuthal integrator used to compute the 2θ map.

    Returns
    -------
    np.ndarray
        Absorption-corrected image in ``float64``.

    Notes
    -----
    This uses a simplified transmission-geometry factor:

    ``abs_factor = 1 / (1 - exp(-mu_t / cos(2theta)))``.

    Real absorption corrections depend on experimental geometry (transmission
    vs reflection, sample orientation, depth profile, etc.). Users should
    validate and adapt this formula for their specific setup.
    """
    image_arr = _as_float_image(image)
    tth = np.asarray(ai.twoThetaArray(shape=image_arr.shape), dtype=float)
    cos_tth = np.cos(tth)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        denom = 1.0 - np.exp(-float(mu_t) / cos_tth)
        abs_factor = 1.0 / denom
        corrected = image_arr * abs_factor

    # Guard numerical singularities / non-physical values.
    corrected[np.abs(cos_tth) < 1e-10] = np.nan
    corrected[np.abs(denom) < 1e-10] = np.nan
    corrected[~np.isfinite(corrected)] = np.nan
    corrected[np.isnan(image_arr)] = np.nan
    return corrected


__all__ = [
    "absorption_correction",
    "polarization_correction",
    "solid_angle_correction",
]
