# ssrl_xrd_tools/integrate/gid.py
"""
Grazing-incidence X-ray diffraction (GIXRD) integration via
``pyFAI.integrator.fiber.FiberIntegrator``.

Requires pyFAI >= 2025.01 (FiberIntegrator replaces pygix).

Key pyFAI behaviour notes
--------------------------
- ``integrate_fiber`` / ``integrate1d_fiber`` / ``integrate2d_fiber`` accept
  ``incident_angle``, ``tilt_angle`` (radians) and ``sample_orientation`` as
  keyword arguments.  These override the stored cache on every call.
  After each integration pyFAI rewrites ``fi._cache_parameters`` from its unit
  object, so ``fi.incident_angle`` is **unreliable** between calls.
  Solution: cache the angles on the ``FiberIntegrator`` instance as
  ``_gi_incident_angle``, ``_gi_tilt_angle``, and ``_gi_sample_orientation``,
  and re-inject them on every integration call.
- All angles stored / injected in **radians**; public functions accept degrees
  by default (``angle_unit="deg"``).

Parameter-name mapping (public API → pyFAI fiber API)
-----------------------------------------------------
    npt              → npt_ip = npt_oop  (same value for intermediate 2-D)
    npt_rad          → npt_ip            (in-plane = "radial" output axis)
    npt_azim         → npt_oop           (out-of-plane = "azimuthal" output)
    radial_range     → ip_range
    azimuth_range    → oop_range

Result-shape convention
-----------------------
pyFAI returns 2-D results with shape ``(npt_oop, npt_ip)``.  All 2-D helpers
transpose to ``(npt_ip, npt_oop) = (npt_rad, npt_azim)`` so that
``IntegrationResult2D.radial`` = in-plane axis and ``.azimuthal`` = OOP axis.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from ssrl_xrd_tools.core.containers import PONI, IntegrationResult1D, IntegrationResult2D
from ssrl_xrd_tools.integrate.calibration import poni_to_integrator

if TYPE_CHECKING:
    from pyFAI.integrator.fiber import FiberIntegrator

logger = logging.getLogger(__name__)

_DEFAULT_UNIT_IP = "qip_A^-1"
_DEFAULT_UNIT_OOP = "qoop_A^-1"

# Private attribute names used to persist GI geometry between calls
_ATTR_INC = "_gi_incident_angle"
_ATTR_TILT = "_gi_tilt_angle"
_ATTR_ORIENT = "_gi_sample_orientation"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _deg2rad_or_pass(value: float, angle_unit: str) -> float:
    return float(np.deg2rad(value)) if angle_unit == "deg" else float(value)


def _effective_gi_params(
    fi: FiberIntegrator,
    incident_angle: float | None,
    tilt_angle: float | None,
    sample_orientation: int | None,
) -> tuple[float, float, int]:
    """
    Return (incident_rad, tilt_rad, orientation) merging caller overrides with
    the cached values stored on *fi* by :func:`create_fiber_integrator`.

    Caller values are in **degrees** and are converted here.  Cache values are
    already in radians.  The cache is updated if new values are provided.
    """
    inc = float(getattr(fi, _ATTR_INC, 0.0))
    tilt = float(getattr(fi, _ATTR_TILT, 0.0))
    orient = int(getattr(fi, _ATTR_ORIENT, 1))

    if incident_angle is not None:
        inc = float(np.deg2rad(incident_angle))
        setattr(fi, _ATTR_INC, inc)
    if tilt_angle is not None:
        tilt = float(np.deg2rad(tilt_angle))
        setattr(fi, _ATTR_TILT, tilt)
    if sample_orientation is not None:
        orient = int(sample_orientation)
        setattr(fi, _ATTR_ORIENT, orient)

    return inc, tilt, orient


def _unit_str(unit: Any) -> str:
    return str(unit[0]) if isinstance(unit, tuple) else str(unit)


def _to_result_2d(result: Any, unit_fallback: str) -> IntegrationResult2D:
    """Convert ``Integrate2dFiberResult`` → ``IntegrationResult2D``."""
    # pyFAI shape: (npt_oop, npt_ip) → transpose to (npt_ip, npt_oop)
    intensity = np.asarray(result.intensity, dtype=float).T
    sigma = (
        np.asarray(result.sigma, dtype=float).T
        if result.sigma is not None
        else None
    )
    ip_unit = getattr(result, "ip_unit", None)
    return IntegrationResult2D(
        radial=np.asarray(result.inplane, dtype=float),
        azimuthal=np.asarray(result.outofplane, dtype=float),
        intensity=intensity,
        sigma=sigma,
        unit=str(ip_unit) if ip_unit is not None else unit_fallback,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_fiber_integrator(
    poni: PONI,
    incident_angle: float,
    tilt_angle: float = 0.0,
    sample_orientation: int = 1,
    angle_unit: str = "deg",
) -> FiberIntegrator:
    """
    Create a ``FiberIntegrator`` from a project ``PONI`` dataclass.

    Parameters
    ----------
    poni : PONI
        Calibration geometry.
    incident_angle : float
        Grazing-incidence angle of the X-ray beam relative to the sample surface.
    tilt_angle : float, optional
        In-plane tilt of the sample.
    sample_orientation : int, optional
        EXIF-style sample orientation flag (1–8).  Default 1 = detector
        horizontal, beam from left.
    angle_unit : {'deg', 'rad'}, optional
        Unit for ``incident_angle`` and ``tilt_angle``.  Default ``"deg"``.

    Returns
    -------
    FiberIntegrator
        Configured pyFAI grazing-incidence integrator.

    Raises
    ------
    ImportError
        If ``FiberIntegrator`` is unavailable (pyFAI < 2025.01).
    """
    ai = poni_to_integrator(poni)
    try:
        fi = ai.promote("FiberIntegrator")
    except (AttributeError, Exception) as exc:
        raise ImportError(
            "FiberIntegrator requires pyFAI >= 2025.01. "
            "Upgrade with:  pip install -U pyFAI"
        ) from exc

    inc = _deg2rad_or_pass(incident_angle, angle_unit)
    tilt = _deg2rad_or_pass(tilt_angle, angle_unit)
    fi.reset_integrator(inc, tilt, int(sample_orientation))

    # Persist so integration helpers can re-inject them on every call
    setattr(fi, _ATTR_INC, inc)
    setattr(fi, _ATTR_TILT, tilt)
    setattr(fi, _ATTR_ORIENT, int(sample_orientation))

    logger.debug(
        "FiberIntegrator created: incident=%.4f rad tilt=%.4f rad orientation=%d",
        inc, tilt, sample_orientation,
    )
    return fi


def integrate_gi_1d(
    image: np.ndarray,
    fi: FiberIntegrator,
    npt: int = 1000,
    unit: str = "qoop_A^-1",
    method: str = "no",
    mask: np.ndarray | None = None,
    radial_range: tuple[float, float] | None = None,
    azimuth_range: tuple[float, float] | None = None,
    incident_angle: float | None = None,
    tilt_angle: float | None = None,
    sample_orientation: int | None = None,
    **kwargs: Any,
) -> IntegrationResult1D:
    """
    1-D grazing-incidence integration via ``FiberIntegrator.integrate1d_fiber``.

    Parameters
    ----------
    image : ndarray
        2-D detector image.
    fi : FiberIntegrator
        Configured fiber integrator from :func:`create_fiber_integrator`.
    npt : int, optional
        Number of output bins.  Passed as both ``npt_ip`` and ``npt_oop``.
    unit : str, optional
        Out-of-plane (OOP) coordinate unit, e.g. ``"qoop_A^-1"``.
        Override in-plane unit via ``unit_ip=`` in ``**kwargs``.
    method : str, optional
        Integration method.  Default ``"no"`` (no pixel-splitting).
    mask : ndarray or None, optional
        Boolean bad-pixel mask.
    radial_range : tuple of float or None, optional
        In-plane range ``(min, max)`` → ``ip_range``.
    azimuth_range : tuple of float or None, optional
        Out-of-plane range ``(min, max)`` → ``oop_range``.
    incident_angle, tilt_angle : float or None, optional
        Override geometry for this call (degrees).  Cached for future calls.
    sample_orientation : int or None, optional
        Override sample orientation for this call.
    **kwargs
        Forwarded to ``fi.integrate1d_fiber``.

    Returns
    -------
    IntegrationResult1D
    """
    inc, tilt, orient = _effective_gi_params(fi, incident_angle, tilt_angle, sample_orientation)
    unit_ip = kwargs.pop("unit_ip", _DEFAULT_UNIT_IP)

    result = fi.integrate1d_fiber(
        image,
        npt_ip=npt,
        npt_oop=npt,
        unit_ip=unit_ip,
        unit_oop=unit,
        sample_orientation=orient,
        method=method,
        mask=mask,
        ip_range=radial_range,
        oop_range=azimuth_range,
        incident_angle=inc,
        tilt_angle=tilt,
        **kwargs,
    )
    sigma = result.sigma if result.sigma is not None else None
    # Use .integrated (not .radial) for fiber/GI results to avoid pyFAI warning
    radial = getattr(result, "integrated", None)
    if radial is None:
        radial = result.radial
    return IntegrationResult1D(
        radial=np.asarray(radial, dtype=float),
        intensity=np.asarray(result.intensity, dtype=float),
        sigma=np.asarray(sigma, dtype=float) if sigma is not None else None,
        unit=_unit_str(result.unit),
    )


def integrate_gi_2d(
    image: np.ndarray,
    fi: FiberIntegrator,
    npt_rad: int = 500,
    npt_azim: int = 500,
    unit: str = "qip_A^-1",
    method: str = "no",
    mask: np.ndarray | None = None,
    radial_range: tuple[float, float] | None = None,
    azimuth_range: tuple[float, float] | None = None,
    incident_angle: float | None = None,
    tilt_angle: float | None = None,
    sample_orientation: int | None = None,
    **kwargs: Any,
) -> IntegrationResult2D:
    """
    2-D grazing-incidence map ``(Q_ip, Q_oop)`` via
    ``FiberIntegrator.integrate2d_fiber``.

    Parameters
    ----------
    image : ndarray
        2-D detector image.
    fi : FiberIntegrator
        Configured fiber integrator.
    npt_rad : int, optional
        In-plane bins → radial axis of output.
    npt_azim : int, optional
        Out-of-plane bins → azimuthal axis of output.
    unit : str, optional
        In-plane unit, e.g. ``"qip_A^-1"``.
        Override OOP unit via ``unit_oop=`` in ``**kwargs``.
    method : str, optional
        Integration method.  Default ``"no"``.
    mask : ndarray or None, optional
        Boolean bad-pixel mask.
    radial_range : tuple of float or None, optional
        In-plane range → ``ip_range``.
    azimuth_range : tuple of float or None, optional
        Out-of-plane range → ``oop_range``.
    incident_angle, tilt_angle : float or None, optional
        Override geometry for this call (degrees).
    sample_orientation : int or None, optional
        Override sample orientation.
    **kwargs
        Forwarded to ``fi.integrate2d_fiber``.

    Returns
    -------
    IntegrationResult2D
        ``radial`` = in-plane axis (npt_rad,),
        ``azimuthal`` = out-of-plane axis (npt_azim,),
        ``intensity.shape == (npt_rad, npt_azim)``.
    """
    inc, tilt, orient = _effective_gi_params(fi, incident_angle, tilt_angle, sample_orientation)
    unit_oop = kwargs.pop("unit_oop", _DEFAULT_UNIT_OOP)

    result = fi.integrate2d_fiber(
        image,
        npt_ip=npt_rad,
        npt_oop=npt_azim,
        unit_ip=unit,
        unit_oop=unit_oop,
        sample_orientation=orient,
        method=method,
        mask=mask,
        ip_range=radial_range,
        oop_range=azimuth_range,
        incident_angle=inc,
        tilt_angle=tilt,
        **kwargs,
    )
    return _to_result_2d(result, unit_fallback=unit)


def integrate_gi_polar(
    image: np.ndarray,
    fi: FiberIntegrator,
    npt_rad: int = 500,
    npt_azim: int = 500,
    unit: str = "q_A^-1",
    method: str = "no",
    mask: np.ndarray | None = None,
    incident_angle: float | None = None,
    tilt_angle: float | None = None,
    sample_orientation: int | None = None,
    **kwargs: Any,
) -> IntegrationResult2D:
    """
    2-D polar map ``(Q_total, polar_angle χ)`` via
    ``FiberIntegrator.integrate2d_polar``.

    Units are fixed internally as ``qtot_A^-1`` (or ``qtot_nm^-1``) vs
    ``chigi_deg``.  The ``unit`` parameter controls only the Q-axis prefix.

    Parameters
    ----------
    image : ndarray
        2-D detector image.
    fi : FiberIntegrator
        Configured fiber integrator.
    npt_rad : int, optional
        Q-total bins → radial axis.
    npt_azim : int, optional
        Polar-angle bins → azimuthal axis.
    unit : str, optional
        Hint for Q unit: ``"q_A^-1"`` → ``qtot_A^-1``;
        anything else → ``qtot_nm^-1``.
    method : str, optional
        Integration method.  Default ``"no"``.
    mask : ndarray or None, optional
        Boolean bad-pixel mask.
    incident_angle, tilt_angle : float or None, optional
        Override geometry (degrees).
    sample_orientation : int or None, optional
        Override sample orientation.
    **kwargs
        Forwarded to ``integrate2d_polar`` / ``integrate2d_fiber``.

    Returns
    -------
    IntegrationResult2D
        ``radial`` = Q_total axis, ``azimuthal`` = polar angle (chigi).
    """
    inc, tilt, orient = _effective_gi_params(fi, incident_angle, tilt_angle, sample_orientation)
    radial_unit = "A^-1" if "A^-1" in unit else "nm^-1"

    result = fi.integrate2d_polar(
        polar_degrees=True,
        radial_unit=radial_unit,
        data=image,
        npt_ip=npt_rad,
        npt_oop=npt_azim,
        sample_orientation=orient,
        method=method,
        mask=mask,
        incident_angle=inc,
        tilt_angle=tilt,
        **kwargs,
    )
    return _to_result_2d(result, unit_fallback=f"qtot_{radial_unit}")


def integrate_gi_exitangles(
    image: np.ndarray,
    fi: FiberIntegrator,
    npt_rad: int = 500,
    npt_azim: int = 500,
    unit: str = "q_A^-1",
    method: str = "no",
    mask: np.ndarray | None = None,
    incident_angle: float | None = None,
    tilt_angle: float | None = None,
    sample_orientation: int | None = None,
    **kwargs: Any,
) -> IntegrationResult2D:
    """
    2-D exit-angle map ``(horizontal exit angle, vertical exit angle)`` via
    ``FiberIntegrator.integrate2d_exitangles``.

    The direct reciprocal-space map ``(Q_xy, Q_z)`` for GIWAXS.  Units are
    exit-angles in degrees by default; pass ``angle_degrees=False`` in
    ``**kwargs`` for radians.

    Parameters
    ----------
    image : ndarray
        2-D detector image.
    fi : FiberIntegrator
        Configured fiber integrator.
    npt_rad : int, optional
        Horizontal exit-angle bins → radial axis.
    npt_azim : int, optional
        Vertical exit-angle bins → azimuthal axis.
    unit : str, optional
        Kept for API consistency; unused (units are fixed as exit angles).
    method : str, optional
        Integration method.  Default ``"no"``.
    mask : ndarray or None, optional
        Boolean bad-pixel mask.
    incident_angle, tilt_angle : float or None, optional
        Override geometry (degrees).
    sample_orientation : int or None, optional
        Override sample orientation.
    **kwargs
        Forwarded to ``integrate2d_exitangles`` / ``integrate2d_fiber``.
        Pass ``angle_degrees=False`` for radian output.

    Returns
    -------
    IntegrationResult2D
        ``radial`` = horizontal exit angle axis,
        ``azimuthal`` = vertical exit angle axis.
    """
    inc, tilt, orient = _effective_gi_params(fi, incident_angle, tilt_angle, sample_orientation)
    angle_degrees = kwargs.pop("angle_degrees", True)

    result = fi.integrate2d_exitangles(
        angle_degrees=angle_degrees,
        data=image,
        npt_ip=npt_rad,
        npt_oop=npt_azim,
        sample_orientation=orient,
        method=method,
        mask=mask,
        incident_angle=inc,
        tilt_angle=tilt,
        **kwargs,
    )
    return _to_result_2d(result, unit_fallback="exit_angle_horz_deg")


def integrate_gi_polar_1d(
    image: np.ndarray,
    fi: FiberIntegrator,
    npt: int = 1000,
    unit: str = "q_A^-1",
    method: str = "no",
    mask: np.ndarray | None = None,
    incident_angle: float | None = None,
    tilt_angle: float | None = None,
    sample_orientation: int | None = None,
    **kwargs: Any,
) -> IntegrationResult1D:
    """
    1-D polar-coordinate integration: intensity vs Q_total (chi-integrated).

    Wraps ``FiberIntegrator.integrate1d_polar``, which azimuthally integrates
    the full ``(Q, χ)`` space and returns a single radial profile in Q.

    Parameters
    ----------
    image : ndarray
        2-D detector image.
    fi : FiberIntegrator
        Configured fiber integrator from :func:`create_fiber_integrator`.
    npt : int, optional
        Number of output bins.  Passed as both ``npt_ip`` and ``npt_oop`` to
        the underlying 2-D polar map before azimuthal reduction.
    unit : str, optional
        Hint for the Q axis unit: ``"q_A^-1"`` (or any string containing
        ``"A^-1"``) → ``"A^-1"``; anything else → ``"nm^-1"``.
    method : str, optional
        Integration method.  Default ``"no"`` (no pixel-splitting).
    mask : ndarray or None, optional
        Boolean bad-pixel mask.
    incident_angle, tilt_angle : float or None, optional
        Override geometry for this call (degrees).  Cached for future calls.
    sample_orientation : int or None, optional
        Override sample orientation for this call.
    **kwargs
        Forwarded to ``fi.integrate1d_polar``.  Useful options include
        ``radial_integration`` (bool, default ``False``) and
        ``polar_degrees`` (bool, default ``True``).

    Returns
    -------
    IntegrationResult1D
        ``radial`` = Q_total axis, ``intensity`` = chi-integrated profile.
    """
    inc, tilt, orient = _effective_gi_params(fi, incident_angle, tilt_angle, sample_orientation)
    radial_unit = "A^-1" if "A^-1" in unit else "nm^-1"

    result = fi.integrate1d_polar(
        polar_degrees=True,
        radial_unit=radial_unit,
        data=image,
        npt_ip=npt,
        npt_oop=npt,
        sample_orientation=orient,
        method=method,
        mask=mask,
        incident_angle=inc,
        tilt_angle=tilt,
        **kwargs,
    )
    sigma = result.sigma if result.sigma is not None else None
    # Use .integrated (not .radial) for fiber/GI results to avoid pyFAI warning
    radial = getattr(result, "integrated", None)
    if radial is None:
        radial = result.radial
    return IntegrationResult1D(
        radial=np.asarray(radial, dtype=float),
        intensity=np.asarray(result.intensity, dtype=float),
        sigma=np.asarray(sigma, dtype=float) if sigma is not None else None,
        unit=_unit_str(result.unit),
    )


def integrate_gi_exitangles_1d(
    image: np.ndarray,
    fi: FiberIntegrator,
    npt: int = 1000,
    method: str = "no",
    mask: np.ndarray | None = None,
    incident_angle: float | None = None,
    tilt_angle: float | None = None,
    sample_orientation: int | None = None,
    **kwargs: Any,
) -> IntegrationResult1D:
    """
    1-D exit-angle integration: intensity vs horizontal exit angle (vertically integrated).

    Wraps ``FiberIntegrator.integrate1d_exitangles``, which integrates the
    ``(2θ_h, 2θ_v)`` exit-angle space and returns a single profile vs the
    horizontal exit angle.

    Parameters
    ----------
    image : ndarray
        2-D detector image.
    fi : FiberIntegrator
        Configured fiber integrator from :func:`create_fiber_integrator`.
    npt : int, optional
        Number of output bins.  Passed as both ``npt_ip`` and ``npt_oop``.
    method : str, optional
        Integration method.  Default ``"no"`` (no pixel-splitting).
    mask : ndarray or None, optional
        Boolean bad-pixel mask.
    incident_angle, tilt_angle : float or None, optional
        Override geometry for this call (degrees).  Cached for future calls.
    sample_orientation : int or None, optional
        Override sample orientation for this call.
    **kwargs
        Forwarded to ``fi.integrate1d_exitangles``.  Pass
        ``angle_degrees=False`` to get radian output, or
        ``vertical_integration=False`` to integrate along the horizontal
        axis instead.

    Returns
    -------
    IntegrationResult1D
        ``radial`` = horizontal exit-angle axis (degrees by default),
        ``intensity`` = vertically integrated profile.
    """
    inc, tilt, orient = _effective_gi_params(fi, incident_angle, tilt_angle, sample_orientation)
    angle_degrees = kwargs.pop("angle_degrees", True)

    result = fi.integrate1d_exitangles(
        angle_degrees=angle_degrees,
        data=image,
        npt_ip=npt,
        npt_oop=npt,
        sample_orientation=orient,
        method=method,
        mask=mask,
        incident_angle=inc,
        tilt_angle=tilt,
        **kwargs,
    )
    sigma = result.sigma if result.sigma is not None else None
    # Use .integrated (not .radial) for fiber/GI results to avoid pyFAI warning
    radial = getattr(result, "integrated", None)
    if radial is None:
        radial = result.radial
    return IntegrationResult1D(
        radial=np.asarray(radial, dtype=float),
        intensity=np.asarray(result.intensity, dtype=float),
        sigma=np.asarray(sigma, dtype=float) if sigma is not None else None,
        unit=_unit_str(result.unit),
    )


__all__ = [
    "create_fiber_integrator",
    "integrate_gi_1d",
    "integrate_gi_2d",
    "integrate_gi_exitangles",
    "integrate_gi_exitangles_1d",
    "integrate_gi_polar",
    "integrate_gi_polar_1d",
]
