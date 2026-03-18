# ssrl_xrd_tools/integrate/multi.py
"""
Multi-image stitching via pyFAI MultiGeometry.

The key pattern: when the detector is scanned to different angular positions
(in-plane ``del`` / ``rot1`` and out-of-plane ``nu`` / ``rot2``), every image
gets its own AzimuthalIntegrator with the detector angle encoded.
``create_multigeometry_integrators`` builds that list; ``stitch_1d`` /
``stitch_2d`` perform the stitched integration.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ssrl_xrd_tools.core.containers import (
    PONI,
    IntegrationResult1D,
    IntegrationResult2D,
)
from ssrl_xrd_tools.integrate.calibration import poni_to_integrator

if TYPE_CHECKING:
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator

logger = logging.getLogger(__name__)


def create_multigeometry_integrators(
    base_poni: PONI,
    rot1_angles: np.ndarray | Sequence[float],
    rot2_angles: np.ndarray | Sequence[float] | None = None,
) -> list[AzimuthalIntegrator]:
    """
    Build a per-image list of AzimuthalIntegrators for a detector-angle scan.

    Each integrator starts from ``base_poni`` and has its ``rot1`` (and
    optionally ``rot2``) shifted by the corresponding scan angle.

    Parameters
    ----------
    base_poni : PONI
        Calibration geometry at the zero-angle detector position.
    rot1_angles : array-like of float
        Per-image in-plane detector rotation offsets **in degrees**
        (e.g. the ``del`` / ``tth`` motor values).
    rot2_angles : array-like of float or None, optional
        Per-image out-of-plane detector rotation offsets **in degrees**
        (e.g. the ``nu`` motor values).  If ``None``, only ``rot1`` varies.

    Returns
    -------
    list of AzimuthalIntegrator
        One integrator per image, ready to pass to :class:`MultiGeometry`.
    """
    rot1 = np.asarray(rot1_angles, dtype=float)
    rot2 = np.zeros_like(rot1) if rot2_angles is None else np.asarray(rot2_angles, dtype=float)
    if rot1.shape != rot2.shape:
        raise ValueError(
            f"rot1_angles length {rot1.shape} != rot2_angles length {rot2.shape}"
        )

    base_ai = poni_to_integrator(base_poni)
    base_rot1 = float(base_ai.rot1)
    base_rot2 = float(base_ai.rot2)

    integrators: list[AzimuthalIntegrator] = []
    for r1_deg, r2_deg in zip(rot1, rot2):
        ai = poni_to_integrator(base_poni)
        ai.rot1 = base_rot1 + float(np.deg2rad(r1_deg))
        ai.rot2 = base_rot2 + float(np.deg2rad(r2_deg))
        integrators.append(ai)

    logger.debug(
        "Created %d per-angle integrators (rot2_varied=%s)",
        len(integrators),
        rot2_angles is not None,
    )
    return integrators


def stitch_1d(
    images: list[np.ndarray] | np.ndarray,
    integrators: list[AzimuthalIntegrator],
    npt: int = 1000,
    unit: str = "q_A^-1",
    method: str = "BBox",
    radial_range: tuple[float, float] | None = None,
    mask: np.ndarray | None = None,
    normalization: np.ndarray | None = None,
    **kwargs: Any,
) -> IntegrationResult1D:
    """
    Stitch a list of images at different detector angles into a 1D pattern.

    Parameters
    ----------
    images : list of ndarray or 3-D ndarray
        Per-image detector frames, one per integrator.
    integrators : list of AzimuthalIntegrator
        Per-image integrators from :func:`create_multigeometry_integrators`.
    npt : int, optional
        Number of radial bins.
    unit : str, optional
        Radial unit, e.g. ``"q_A^-1"``, ``"2th_deg"``.
    method : str, optional
        Integration method.  Default is ``"BBox"``; MultiGeometry works best
        with histogram-based methods.
    radial_range : tuple of float or None, optional
        ``(min, max)`` radial range applied at MultiGeometry construction.
    mask : ndarray or None, optional
        Single detector mask applied to every image.
    normalization : array-like of float or None, optional
        Per-image monitor counts.  Each image is divided by its corresponding
        value before integration.
    **kwargs
        Extra keyword arguments forwarded to ``mg.integrate1d``.

    Returns
    -------
    IntegrationResult1D
    """
    from pyFAI.multi_geometry import MultiGeometry

    img_list = _prepare_images(images, normalization)
    lst_mask = [mask] * len(img_list) if mask is not None else None

    mg = MultiGeometry(integrators, unit=unit, radial_range=radial_range)
    result = mg.integrate1d(img_list, npt, lst_mask=lst_mask, method=method, **kwargs)

    sigma = result.sigma if result.sigma is not None else None
    unit_str = str(result.unit) if result.unit is not None else unit
    return IntegrationResult1D(
        radial=np.asarray(result.radial, dtype=float),
        intensity=np.asarray(result.intensity, dtype=float),
        sigma=np.asarray(sigma, dtype=float) if sigma is not None else None,
        unit=unit_str,
    )


def stitch_2d(
    images: list[np.ndarray] | np.ndarray,
    integrators: list[AzimuthalIntegrator],
    npt_rad: int = 1000,
    npt_azim: int = 1000,
    unit: str = "q_A^-1",
    method: str = "BBox",
    radial_range: tuple[float, float] | None = None,
    azimuth_range: tuple[float, float] | None = None,
    mask: np.ndarray | None = None,
    **kwargs: Any,
) -> IntegrationResult2D:
    """
    Stitch a list of images at different detector angles into a 2D cake.

    Parameters
    ----------
    images : list of ndarray or 3-D ndarray
        Per-image detector frames, one per integrator.
    integrators : list of AzimuthalIntegrator
        Per-image integrators from :func:`create_multigeometry_integrators`.
    npt_rad : int, optional
        Number of radial bins.
    npt_azim : int, optional
        Number of azimuthal bins.
    unit : str, optional
        Radial unit.
    method : str, optional
        Integration method.
    radial_range : tuple of float or None, optional
        ``(min, max)`` radial range applied at MultiGeometry construction.
    azimuth_range : tuple of float or None, optional
        ``(min, max)`` azimuthal range (degrees) applied at MultiGeometry
        construction.
    mask : ndarray or None, optional
        Single detector mask applied to every image.
    **kwargs
        Extra keyword arguments forwarded to ``mg.integrate2d``.

    Returns
    -------
    IntegrationResult2D
        Intensity has shape ``(npt_rad, npt_azim)`` (transposed from pyFAI).
    """
    from pyFAI.multi_geometry import MultiGeometry

    img_list = _prepare_images(images, normalization=None)
    lst_mask = [mask] * len(img_list) if mask is not None else None

    mg = MultiGeometry(
        integrators,
        unit=unit,
        radial_range=radial_range,
        azimuth_range=azimuth_range,
    )
    result = mg.integrate2d(
        img_list, npt_rad, npt_azim, lst_mask=lst_mask, method=method, **kwargs
    )

    # pyFAI returns intensity (npt_azim, npt_rad); transpose to (npt_rad, npt_azim)
    intensity = np.asarray(result.intensity, dtype=float).T
    sigma = (
        np.asarray(result.sigma, dtype=float).T
        if result.sigma is not None
        else None
    )
    unit_str = (
        str(result.unit[0]) if isinstance(result.unit, tuple) else str(result.unit)
    )
    return IntegrationResult2D(
        radial=np.asarray(result.radial, dtype=float),
        azimuthal=np.asarray(result.azimuthal, dtype=float),
        intensity=intensity,
        sigma=sigma,
        unit=unit_str,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prepare_images(
    images: list[np.ndarray] | np.ndarray,
    normalization: np.ndarray | Sequence[float] | None,
) -> list[np.ndarray]:
    """Convert images to a list and apply optional per-image normalisation."""
    if isinstance(images, np.ndarray):
        if images.ndim == 3:
            img_list: list[np.ndarray] = [images[i] for i in range(images.shape[0])]
        elif images.ndim == 2:
            img_list = [images]
        else:
            raise ValueError(f"images ndarray must be 2D or 3D, got shape {images.shape}")
    else:
        img_list = [np.asarray(im, dtype=float) for im in images]

    if normalization is not None:
        norm = np.asarray(normalization, dtype=float)
        if norm.shape != (len(img_list),):
            raise ValueError(
                f"normalization length {norm.shape} != number of images {len(img_list)}"
            )
        img_list = [img / n for img, n in zip(img_list, norm)]

    return img_list


__all__ = [
    "create_multigeometry_integrators",
    "stitch_1d",
    "stitch_2d",
]
