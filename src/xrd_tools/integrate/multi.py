# xrd_tools/integrate/multi.py
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

from xrd_tools.core.containers import (
    PONI,
    IntegrationResult1D,
    IntegrationResult2D,
)
from xrd_tools.integrate.calibration import poni_to_integrator

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


def create_multigeometry_integrators_from_geometry(
    diffractometer: Any,
    motors: Any,
    *,
    base_calibration: Any = None,
) -> list[AzimuthalIntegrator]:
    """Per-frame integrators from a :class:`Diffractometer` — the calibrated path.

    Closes stitching GAP A + GAP B vs :func:`create_multigeometry_integrators`:

    * **GAP A (fitted scales, not a hardwired ``deg2rad``):** per-frame rotations
      come from ``diffractometer.to_pyfai_per_frame(motors)``, so a *calibrated*
      goniometer's fitted per-axis scales/offsets are used (an uncalibrated preset
      whose ``AngleMapping.sign == 1`` reduces exactly to the old ``deg2rad`` path).
    * **GAP B (panel mount preserved):** the base geometry is a
      :class:`DetectorCalibration` carrying ``Detector_config`` (orientation), built
      via :func:`detector_calibration_to_integrator` — the orientation is no longer
      silently dropped.

    Per-frame ``rotN = base.poni.rotN + to_pyfai_per_frame()[rotN]`` (the same
    decomposition :meth:`Diffractometer.from_pyfai_goniometer` produces).

    Parameters
    ----------
    diffractometer : Diffractometer
        The instrument geometry (preset-built or gonio-fitted).
    motors : Mapping[str, array-like]
        Per-frame motor columns (degrees), keyed by motor name.
    base_calibration : DetectorCalibration, optional
        The base detector calibration; defaults to
        ``diffractometer.calibration``.

    Returns
    -------
    list of AzimuthalIntegrator
        One per frame, ready for :class:`MultiGeometry` / the histogram backends.
    """
    from xrd_tools.integrate.calibration import (  # noqa: PLC0415
        detector_calibration_to_integrator,
    )

    cal = base_calibration if base_calibration is not None else getattr(
        diffractometer, "calibration", None)
    if cal is None:
        raise ValueError(
            "no DetectorCalibration: pass base_calibration= or use a "
            "Diffractometer carrying one (from_pyfai_goniometer / a fitted "
            "geometry). The base dist/poni/Detector_config is required to stitch.")

    # Compute ONLY the per-frame detector rotations (rot1/2/3) directly — a stitch
    # never needs the GI incidence, so a stitch scan that lacks the incidence motor
    # (e.g. psic's `eta`) must not crash; to_pyfai_per_frame stays strict for GI.
    def _frame_rot(mapping: Any) -> np.ndarray | None:
        if not mapping.is_active:
            return None
        if mapping.source_motor not in motors:
            raise KeyError(
                f"stitch geometry needs motor {mapping.source_motor!r} (it drives "
                f"a detector rotation) but the source provides {sorted(motors)}")
        return np.deg2rad(mapping.apply(
            np.asarray(motors[mapping.source_motor], dtype=float)))

    r1 = _frame_rot(diffractometer.rot1)
    r2 = _frame_rot(diffractometer.rot2)
    r3 = _frame_rot(diffractometer.rot3)
    nframes = next((len(r) for r in (r1, r2, r3) if r is not None), None)
    if nframes is None:  # no active detector rotation — use any motor column length
        col = next(iter(motors.values()), None)
        nframes = len(np.atleast_1d(np.asarray(col))) if col is not None else 1
    base = cal.poni
    integrators: list[AzimuthalIntegrator] = []
    for i in range(nframes):
        integrators.append(detector_calibration_to_integrator(
            cal,
            rot1=float(base.rot1) + (float(r1[i]) if r1 is not None else 0.0),
            rot2=float(base.rot2) + (float(r2[i]) if r2 is not None else 0.0),
            rot3=float(base.rot3) + (float(r3[i]) if r3 is not None else 0.0),
        ))
    logger.debug("Created %d per-frame integrators from a Diffractometer "
                 "(preset=%s)", len(integrators),
                 getattr(diffractometer, "preset", "?"))
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
    normalization: np.ndarray | None = None,
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
    normalization : array-like of float or None, optional
        Per-image monitor counts.  Each image is divided by its
        corresponding value before integration.  Matches the
        ``normalization`` parameter on :func:`stitch_1d` — both stitching
        paths share the same :func:`_prepare_images` helper so the math
        is identical.
    **kwargs
        Extra keyword arguments forwarded to ``mg.integrate2d``.

    Returns
    -------
    IntegrationResult2D
        Intensity has shape ``(npt_rad, npt_azim)`` (transposed from pyFAI).
    """
    from pyFAI.multi_geometry import MultiGeometry

    img_list = _prepare_images(images, normalization)
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
        if not np.all(np.isfinite(norm)):
            raise ValueError("normalization contains non-finite (nan/inf) values")
        if np.any(norm == 0):
            raise ValueError("normalization contains zero values (would divide by zero)")
        img_list = [img / n for img, n in zip(img_list, norm)]

    return img_list


def stitch_images(
    images: list[np.ndarray] | np.ndarray,
    base_poni: PONI,
    rot1_angles: np.ndarray | Sequence[float],
    rot2_angles: np.ndarray | Sequence[float] | None = None,
    *,
    mode: str = "1d",
    npt_1d: int = 2000,
    npt_rad_2d: int = 1500,
    npt_azim_2d: int = 720,
    unit: str = "q_A^-1",
    method: str = "BBox",
    radial_range: tuple[float, float] | None = None,
    azimuth_range: tuple[float, float] | None = None,
    mask: np.ndarray | None = None,
    normalization: np.ndarray | Sequence[float] | None = None,
) -> IntegrationResult1D | IntegrationResult2D:
    """Stitch a detector-angle image stack into a 1D or 2D pattern.

    High-level entry point that builds the per-image MultiGeometry
    integrators from ``base_poni`` + per-image ``rot1``/``rot2`` offsets
    (degrees) and dispatches to :func:`stitch_1d` / :func:`stitch_2d`.
    This is the orchestration the xdart GUI used to carry inline; keeping
    it here lets headless callers stitch without reimplementing the
    integrator-build + dispatch.

    Parameters mirror :func:`stitch_1d` / :func:`stitch_2d`; ``mode``
    selects which.  ``rot2_angles`` that are all-zero (or ``None``) are
    treated as a pure ``rot1`` scan.
    """
    # Fail early on a count mismatch — feeding MultiGeometry an unequal
    # number of images and integrators silently mispairs images with the
    # wrong detector angle (or raises deep inside pyFAI).
    rot1 = np.asarray(rot1_angles, dtype=float)
    # Count images the same way _prepare_images interprets them: a 3-D
    # ndarray is a stack (count = shape[0]); a 2-D ndarray is a single
    # image (count = 1, NOT shape[0]); anything else is a sequence.
    if isinstance(images, np.ndarray):
        n_images = images.shape[0] if images.ndim == 3 else 1
    else:
        n_images = len(images)
    if n_images != rot1.shape[0]:
        raise ValueError(
            f"stitch_images: {n_images} images != {rot1.shape[0]} angles; "
            "one detector angle is required per image."
        )

    rot2 = (
        rot2_angles
        if rot2_angles is not None and np.any(np.asarray(rot2_angles))
        else None
    )
    integrators = create_multigeometry_integrators(
        base_poni, rot1_angles=rot1_angles, rot2_angles=rot2,
    )
    if mode == "1d":
        return stitch_1d(
            images, integrators, npt=npt_1d, unit=unit, method=method,
            radial_range=radial_range, mask=mask, normalization=normalization,
        )
    if mode == "2d":
        return stitch_2d(
            images, integrators, npt_rad=npt_rad_2d, npt_azim=npt_azim_2d,
            unit=unit, method=method, radial_range=radial_range,
            azimuth_range=azimuth_range, mask=mask, normalization=normalization,
        )
    raise ValueError(f"mode must be '1d' or '2d', got {mode!r}")


__all__ = [
    "create_multigeometry_integrators",
    "create_multigeometry_integrators_from_geometry",
    "stitch_1d",
    "stitch_2d",
    "stitch_images",
]
