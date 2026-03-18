# ssrl_xrd_tools/integrate/single.py
"""
Single-image and scan-level azimuthal integration via pyFAI.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from ssrl_xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D

if TYPE_CHECKING:
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator

logger = logging.getLogger(__name__)


def integrate_1d(
    image: np.ndarray,
    ai: AzimuthalIntegrator,
    npt: int = 1000,
    unit: str = "q_A^-1",
    method: str = "csr",
    mask: np.ndarray | None = None,
    radial_range: tuple[float, float] | None = None,
    azimuth_range: tuple[float, float] | None = None,
    error_model: str | None = None,
    polarization_factor: float | None = None,
    normalization_factor: float | None = None,
    **kwargs: Any,
) -> IntegrationResult1D:
    """
    Integrate a single detector image to a 1D pattern.

    Parameters
    ----------
    image : ndarray
        2D detector image.
    ai : AzimuthalIntegrator
        Configured pyFAI integrator.
    npt : int, optional
        Number of radial bins.
    unit : str, optional
        Radial unit, e.g. ``"q_A^-1"``, ``"2th_deg"``.
    method : str, optional
        Integration method passed to pyFAI, e.g. ``"csr"``, ``"BBox"``.
    mask : ndarray or None, optional
        Boolean mask; ``True`` marks bad pixels.
    radial_range : tuple of float or None, optional
        ``(min, max)`` limits on the radial axis.
    azimuth_range : tuple of float or None, optional
        ``(min, max)`` limits on the azimuthal axis in degrees.
    error_model : str or None, optional
        pyFAI error model, e.g. ``"poisson"``. When set, ``sigma`` is
        populated in the result.
    polarization_factor : float or None, optional
        Synchrotron X-rays are horizontally polarized. Typical values:
        ~0.99 for bending-magnet beamlines, ~1.0 for undulators. When
        ``None`` (default), pyFAI's own default is preserved.
    normalization_factor : float or None, optional
        Scales the result by ``1 / normalization_factor``. Useful for
        monitor normalization (e.g. dividing by *i1* counts). When
        ``None`` (default), pyFAI's own default is preserved.
    **kwargs
        Additional keyword arguments forwarded to ``ai.integrate1d``.

    Returns
    -------
    IntegrationResult1D
        Radial axis, intensity, and (optionally) sigma.
    """
    extra: dict[str, Any] = dict(**kwargs)
    if polarization_factor is not None:
        extra["polarization_factor"] = polarization_factor
    if normalization_factor is not None:
        extra["normalization_factor"] = normalization_factor
    result = ai.integrate1d(
        image,
        npt,
        unit=unit,
        method=method,
        mask=mask,
        radial_range=radial_range,
        azimuth_range=azimuth_range,
        error_model=error_model,
        **extra,
    )
    sigma = result.sigma if result.sigma is not None else None
    unit_str = str(result.unit) if result.unit is not None else unit
    return IntegrationResult1D(
        radial=np.asarray(result.radial, dtype=float),
        intensity=np.asarray(result.intensity, dtype=float),
        sigma=np.asarray(sigma, dtype=float) if sigma is not None else None,
        unit=unit_str,
    )


def integrate_2d(
    image: np.ndarray,
    ai: AzimuthalIntegrator,
    npt_rad: int = 1000,
    npt_azim: int = 1000,
    unit: str = "q_A^-1",
    method: str = "csr",
    mask: np.ndarray | None = None,
    radial_range: tuple[float, float] | None = None,
    azimuth_range: tuple[float, float] | None = None,
    error_model: str | None = None,
    polarization_factor: float | None = None,
    normalization_factor: float | None = None,
    **kwargs: Any,
) -> IntegrationResult2D:
    """
    Integrate a single detector image to a 2D cake pattern.

    Parameters
    ----------
    image : ndarray
        2D detector image.
    ai : AzimuthalIntegrator
        Configured pyFAI integrator.
    npt_rad : int, optional
        Number of radial bins.
    npt_azim : int, optional
        Number of azimuthal bins.
    unit : str, optional
        Radial unit, e.g. ``"q_A^-1"``, ``"2th_deg"``.
    method : str, optional
        Integration method passed to pyFAI.
    mask : ndarray or None, optional
        Boolean mask; ``True`` marks bad pixels.
    radial_range : tuple of float or None, optional
        ``(min, max)`` limits on the radial axis.
    azimuth_range : tuple of float or None, optional
        ``(min, max)`` azimuthal range in degrees.
    error_model : str or None, optional
        pyFAI error model.  When set, ``sigma`` is populated in the result.
    polarization_factor : float or None, optional
        Synchrotron X-rays are horizontally polarized. Typical values:
        ~0.99 for bending-magnet beamlines, ~1.0 for undulators. When
        ``None`` (default), pyFAI's own default is preserved.
    normalization_factor : float or None, optional
        Scales the result by ``1 / normalization_factor``. Useful for
        monitor normalization (e.g. dividing by *i1* counts). When
        ``None`` (default), pyFAI's own default is preserved.
    **kwargs
        Additional keyword arguments forwarded to ``ai.integrate2d``.

    Returns
    -------
    IntegrationResult2D
        Radial axis, azimuthal axis, intensity ``(npt_rad, npt_azim)``,
        and (optionally) sigma of the same shape.

    Notes
    -----
    pyFAI returns ``intensity`` with shape ``(npt_azim, npt_rad)``.  This
    function transposes to ``(npt_rad, npt_azim)`` to match our convention.
    """
    extra: dict[str, Any] = dict(**kwargs)
    if polarization_factor is not None:
        extra["polarization_factor"] = polarization_factor
    if normalization_factor is not None:
        extra["normalization_factor"] = normalization_factor
    result = ai.integrate2d(
        image,
        npt_rad,
        npt_azim,
        unit=unit,
        method=method,
        mask=mask,
        radial_range=radial_range,
        azimuth_range=azimuth_range,
        error_model=error_model,
        **extra,
    )
    # pyFAI intensity shape is (npt_azim, npt_rad); transpose to (npt_rad, npt_azim)
    intensity = np.asarray(result.intensity, dtype=float).T
    sigma = (
        np.asarray(result.sigma, dtype=float).T
        if result.sigma is not None
        else None
    )
    # result.unit is a tuple (radial_unit, azimuth_unit); take the radial part
    unit_str = str(result.unit[0]) if isinstance(result.unit, tuple) else str(result.unit)
    return IntegrationResult2D(
        radial=np.asarray(result.radial, dtype=float),
        azimuthal=np.asarray(result.azimuthal, dtype=float),
        intensity=intensity,
        sigma=sigma,
        unit=unit_str,
    )


def integrate_scan(
    images: np.ndarray,
    ai: AzimuthalIntegrator,
    npt: int = 1000,
    unit: str = "q_A^-1",
    method: str = "csr",
    mask: np.ndarray | None = None,
    reduce: str = "sum",
    polarization_factor: float | None = None,
    normalization_factor: float | None = None,
    **kwargs: Any,
) -> IntegrationResult1D:
    """
    Integrate a 3D image stack and reduce to a single 1D pattern.

    Parameters
    ----------
    images : ndarray
        3D array of shape ``(n_frames, ny, nx)``.
    ai : AzimuthalIntegrator
        Configured pyFAI integrator.
    npt : int, optional
        Number of radial bins.
    unit : str, optional
        Radial unit.
    method : str, optional
        Integration method.
    mask : ndarray or None, optional
        Boolean mask; ``True`` marks bad pixels.
    reduce : {'sum', 'mean'}
        How to combine per-frame patterns.
    polarization_factor : float or None, optional
        Synchrotron X-rays are horizontally polarized. Typical values:
        ~0.99 for bending-magnet beamlines, ~1.0 for undulators. When
        ``None`` (default), pyFAI's own default is preserved.
        Forwarded to :func:`integrate_1d`.
    normalization_factor : float or None, optional
        Scales each frame's result by ``1 / normalization_factor``. When
        ``None`` (default), pyFAI's own default is preserved.
        Forwarded to :func:`integrate_1d`.
    **kwargs
        Additional keyword arguments forwarded to :func:`integrate_1d`.

    Returns
    -------
    IntegrationResult1D
        Combined 1D pattern.

    Raises
    ------
    ValueError
        If ``images`` is not 3D, or ``reduce`` is not ``'sum'`` or ``'mean'``.
    """
    images = np.asarray(images, dtype=float)
    if images.ndim != 3:
        raise ValueError(f"images must be 3D (n_frames, ny, nx), got shape {images.shape}")
    if reduce not in {"sum", "mean"}:
        raise ValueError(f"reduce must be 'sum' or 'mean', got {reduce!r}")

    frame_results = [
        integrate_1d(
            images[i],
            ai,
            npt=npt,
            unit=unit,
            method=method,
            mask=mask,
            polarization_factor=polarization_factor,
            normalization_factor=normalization_factor,
            **kwargs,
        )
        for i in range(images.shape[0])
    ]

    radial = frame_results[0].radial
    all_intensity = np.stack([r.intensity for r in frame_results], axis=0)
    combined = np.nansum(all_intensity, axis=0) if reduce == "sum" else np.nanmean(all_intensity, axis=0)

    all_sigma = [r.sigma for r in frame_results]
    if all(s is not None for s in all_sigma):
        stacked_var = np.stack([s ** 2 for s in all_sigma], axis=0)  # type: ignore[operator]
        if reduce == "sum":
            combined_sigma = np.sqrt(np.nansum(stacked_var, axis=0))
        else:
            n = images.shape[0]
            combined_sigma = np.sqrt(np.nansum(stacked_var, axis=0)) / n
    else:
        combined_sigma = None

    return IntegrationResult1D(
        radial=radial,
        intensity=combined,
        sigma=combined_sigma,
        unit=frame_results[0].unit,
    )


__all__ = [
    "integrate_1d",
    "integrate_2d",
    "integrate_scan",
]
