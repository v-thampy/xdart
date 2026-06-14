# xrd_tools/integrate/gid.py
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

from xrd_tools.core.containers import PONI, IntegrationResult1D, IntegrationResult2D
from xrd_tools.integrate.calibration import poni_to_integrator

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


def _to_result_2d(result: Any, unit_fallback: str,
                  azim_unit_fallback: str = "qoop_A^-1") -> IntegrationResult2D:
    """Convert pyFAI grazing-incidence 2D result → ``IntegrationResult2D``.

    Handles both pyFAI versions:
      * pyFAI 2025.x: ``integrate2d_grazing_incidence`` returns a plain
        ``Integrate2dResult`` whose ``.radial`` axis (length ``npt_ip``)
        is the in-plane axis and ``.azimuthal`` (length ``npt_oop``) is
        the out-of-plane axis.
      * pyFAI 2026.x+: returns ``Integrate2dFiberResult`` with explicit
        ``.inplane`` / ``.outofplane`` attributes — the renamed aliases
        for the same two axes.
    """
    # pyFAI shape: (npt_oop, npt_ip) → transpose to (npt_ip, npt_oop)
    intensity = np.asarray(result.intensity, dtype=float).T
    sigma = (
        np.asarray(result.sigma, dtype=float).T
        if result.sigma is not None
        else None
    )
    inplane = getattr(result, "inplane", None)
    if inplane is None:
        inplane = result.radial
    outofplane = getattr(result, "outofplane", None)
    if outofplane is None:
        outofplane = result.azimuthal
    ip_unit = getattr(result, "ip_unit", None)
    oop_unit = getattr(result, "oop_unit", None)
    return IntegrationResult2D(
        radial=np.asarray(inplane, dtype=float),
        azimuthal=np.asarray(outofplane, dtype=float),
        intensity=intensity,
        sigma=sigma,
        unit=str(ip_unit) if ip_unit is not None else unit_fallback,
        # pyFAI 2025.x results carry no ip/oop unit attrs, so the fallback
        # must match the TRANSFORM: hardcoding qoop here labeled the polar
        # chi axis "Q_oop (A^-1)" in the GUI (wrong unit AND wrong name).
        azimuthal_unit=(str(oop_unit) if oop_unit is not None
                        else azim_unit_fallback),
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

    # Disable pyFAI's legacy mask heuristic that auto-inverts masks
    # with > 50% masked pixels (e.g. threshold masks).
    fi.USE_LEGACY_MASK_NORMALIZATION = False

    # Persist so integration helpers can re-inject them on every call
    setattr(fi, _ATTR_INC, inc)
    setattr(fi, _ATTR_TILT, tilt)
    setattr(fi, _ATTR_ORIENT, int(sample_orientation))

    logger.debug(
        "FiberIntegrator created: incident=%.4f rad tilt=%.4f rad orientation=%d",
        inc, tilt, sample_orientation,
    )
    return fi


def _nan_empty_1d(result):
    """Return a 1D result's intensity with EMPTY bins set to NaN.

    pyFAI fills bins with no contributing pixels (``count == 0``) with 0, so a
    frozen output range wider than the data (the GI-freeze coverage pad) — and
    any fully-masked gap — would plot/aggregate as a spurious flat 0/dummy line
    (the points-at-negative-Q artifact).  NaN-marking those bins keeps them out
    of the plot; the aggregations are NaN-aware (nanmean/nansum) and the
    equivalence spine compares with equal_nan.  Defensive: if no per-bin
    ``count`` is exposed (some fiber methods), the intensity is returned
    unchanged.
    """
    intensity = np.asarray(result.intensity, dtype=float)
    count = getattr(result, "count", None)
    if count is not None:
        count = np.asarray(count)
        if count.shape == intensity.shape:
            intensity = np.where(count == 0, np.nan, intensity)
    return intensity


def integrate_gi_1d(
    image: np.ndarray,
    fi: FiberIntegrator,
    npt: int = 1000,
    npt_oop: int | None = None,
    unit: str = "qoop_A^-1",
    method: str = "no",
    mask: np.ndarray | None = None,
    radial_range: tuple[float, float] | None = None,
    azimuth_range: tuple[float, float] | None = None,
    incident_angle: float | None = None,
    tilt_angle: float | None = None,
    sample_orientation: int | None = None,
    vertical_integration: bool = True,
    **kwargs: Any,
) -> IntegrationResult1D:
    """
    1-D grazing-incidence integration via
    ``FiberIntegrator.integrate1d_grazing_incidence``.

    Parameters
    ----------
    image : ndarray
        2-D detector image.
    fi : FiberIntegrator
        Configured fiber integrator from :func:`create_fiber_integrator`.
    npt : int, optional
        Number of output bins for the in-plane axis (``npt_ip``).
    npt_oop : int or None, optional
        Number of output bins for the out-of-plane axis.  If *None*,
        defaults to ``npt`` (same resolution for both axes).
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
    vertical_integration : bool, optional
        If *True* (default), integrate over IP and return a 1-D OOP profile
        (Q_oop).  If *False*, integrate over OOP and return a 1-D IP profile
        (Q_ip).
    **kwargs
        Forwarded to ``fi.integrate1d_grazing_incidence``.

    Returns
    -------
    IntegrationResult1D
    """
    inc, tilt, orient = _effective_gi_params(fi, incident_angle, tilt_angle, sample_orientation)
    unit_ip = kwargs.pop("unit_ip", _DEFAULT_UNIT_IP)
    _npt_oop = npt_oop if npt_oop is not None else npt

    result = fi.integrate1d_grazing_incidence(
        image,
        npt_ip=npt,
        npt_oop=_npt_oop,
        unit_ip=unit_ip,
        unit_oop=unit,
        sample_orientation=orient,
        method=method,
        mask=mask,
        ip_range=radial_range,
        oop_range=azimuth_range,
        incident_angle=inc,
        tilt_angle=tilt,
        vertical_integration=vertical_integration,
        **kwargs,
    )
    sigma = result.sigma if result.sigma is not None else None
    radial = getattr(result, "integrated", None)
    if radial is None:
        radial = result.radial
    return IntegrationResult1D(
        radial=np.asarray(radial, dtype=float),
        intensity=_nan_empty_1d(result),
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
    ``FiberIntegrator.integrate2d_grazing_incidence``.

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
        Forwarded to ``fi.integrate2d_grazing_incidence``.

    Returns
    -------
    IntegrationResult2D
        ``radial`` = in-plane axis (npt_rad,),
        ``azimuthal`` = out-of-plane axis (npt_azim,),
        ``intensity.shape == (npt_rad, npt_azim)``.
    """
    inc, tilt, orient = _effective_gi_params(fi, incident_angle, tilt_angle, sample_orientation)
    unit_oop = kwargs.pop("unit_oop", _DEFAULT_UNIT_OOP)

    result = fi.integrate2d_grazing_incidence(
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
    radial_range: tuple[float, float] | None = None,
    azimuth_range: tuple[float, float] | None = None,
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
    if azimuth_range is not None:
        # pyFAI wraps out-of-domain polar requests instead of clamping.
        azimuth_range = (max(float(azimuth_range[0]), -180.0),
                         min(float(azimuth_range[1]), 180.0))

    result = fi.integrate2d_polar(
        polar_degrees=True,
        radial_unit=radial_unit,
        data=image,
        npt_ip=npt_rad,
        npt_oop=npt_azim,
        sample_orientation=orient,
        method=method,
        mask=mask,
        ip_range=radial_range,
        oop_range=azimuth_range,
        incident_angle=inc,
        tilt_angle=tilt,
        **kwargs,
    )
    return _to_result_2d(result, unit_fallback=f"qtot_{radial_unit}",
                         azim_unit_fallback="chigi_deg")


def integrate_gi_exitangles(
    image: np.ndarray,
    fi: FiberIntegrator,
    npt_rad: int = 500,
    npt_azim: int = 500,
    unit: str = "q_A^-1",
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
        ip_range=radial_range,
        oop_range=azimuth_range,
        incident_angle=inc,
        tilt_angle=tilt,
        **kwargs,
    )
    return _to_result_2d(result, unit_fallback="exit_angle_horz_deg",
                         azim_unit_fallback="exit_angle_vert_deg")


def integrate_gi_polar_1d(
    image: np.ndarray,
    fi: FiberIntegrator,
    npt: int = 1000,
    unit: str = "q_A^-1",
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
    1-D polar-coordinate integration: intensity vs Q_total (chi-integrated).

    Fast path
    ---------
    When ``azimuth_range is None`` (the common "integrate over all χ" case),
    this is numerically identical to a standard ``AzimuthalIntegrator.integrate1d``
    on the q magnitude — every detector pixel ends up in the same q bin either
    way, because q is invariant under sample rotation.  Since
    ``FiberIntegrator`` IS an ``AzimuthalIntegrator``, we call its inherited
    ``integrate1d`` directly: a single-pass 1-D rebin, much faster than the
    2D-rebin-then-collapse path of ``integrate1d_polar``.

    When ``azimuth_range`` is set (partial-χ wedge), we fall back to
    ``FiberIntegrator.integrate1d_polar`` so the χ filter is applied
    on the intermediate 2-D ``(q_ip, q_oop)`` grid where the polar mask
    is well-defined.

    Parameters
    ----------
    image : ndarray
        2-D detector image.
    fi : FiberIntegrator
        Configured fiber integrator from :func:`create_fiber_integrator`.
    npt : int, optional
        Number of output radial bins.  In the slow path, also used as both
        ``npt_ip`` and ``npt_oop`` for the intermediate 2-D polar map.
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
        Forwarded to the chosen integrator.  Useful options for the slow path
        include ``radial_integration`` (bool, default ``False``) and
        ``polar_degrees`` (bool, default ``True``).

    Returns
    -------
    IntegrationResult1D
        ``radial`` = Q_total axis, ``intensity`` = chi-integrated profile.
    """
    inc, tilt, orient = _effective_gi_params(fi, incident_angle, tilt_angle, sample_orientation)
    radial_unit = "A^-1" if "A^-1" in unit else "nm^-1"

    # --- fast path: full χ → standard 1D azimuthal integration --------------
    if azimuth_range is None:
        # Strip kwargs only meaningful to fiber-specific methods so they
        # don't poison the standard integrate1d call.
        _fiber_only = {
            "polar_degrees", "radial_integration",
            "ip_range", "oop_range",
            "sample_orientation", "incident_angle", "tilt_angle",
            "npt_ip", "npt_oop", "vertical_integration", "unit_oop",
        }
        std_kwargs = {k: v for k, v in kwargs.items() if k not in _fiber_only}
        std_unit = f"q_{radial_unit}"
        result = fi.integrate1d(
            data=image,
            npt=npt,
            unit=std_unit,
            method=method,
            mask=mask,
            radial_range=radial_range,
            **std_kwargs,
        )
        sigma = result.sigma if result.sigma is not None else None
        return IntegrationResult1D(
            radial=np.asarray(result.radial, dtype=float),
            intensity=_nan_empty_1d(result),
            sigma=np.asarray(sigma, dtype=float) if sigma is not None else None,
            unit=_unit_str(result.unit) if result.unit is not None else std_unit,
        )

    # --- slow path: restricted χ wedge → polar method -----------------------
    result = fi.integrate1d_polar(
        polar_degrees=True,
        radial_unit=radial_unit,
        data=image,
        npt_ip=npt,
        npt_oop=npt,
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
        intensity=_nan_empty_1d(result),
        sigma=np.asarray(sigma, dtype=float) if sigma is not None else None,
        unit=_unit_str(result.unit),
    )


def integrate_gi_exitangles_1d(
    image: np.ndarray,
    fi: FiberIntegrator,
    npt: int = 1000,
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
        ip_range=radial_range,
        oop_range=azimuth_range,
        **kwargs,
    )
    sigma = result.sigma if result.sigma is not None else None
    # Use .integrated (not .radial) for fiber/GI results to avoid pyFAI warning
    radial = getattr(result, "integrated", None)
    if radial is None:
        radial = result.radial
    return IntegrationResult1D(
        radial=np.asarray(radial, dtype=float),
        intensity=_nan_empty_1d(result),
        sigma=np.asarray(sigma, dtype=float) if sigma is not None else None,
        unit=_unit_str(result.unit),
    )


# ---------------------------------------------------------------------------
# Common-grid freeze (pure axis math)
# ---------------------------------------------------------------------------
#
# A grazing-incidence angle-dependence scan integrates many frames that must
# share ONE output grid (the stacked NeXus writer rejects per-frame axis drift).
# The GUI "freezes" a common output range from a few scout integrations and
# hands the same explicit range to every frame.  The *pure* part — given one or
# more scout results, compute the padded, unioned output range and which
# integration arg key it maps to — lives here so it is headless + testable and
# reusable by future RSM / stitching axis logic.  The GUI keeps only the Qt
# orchestration (pick + run the scout integrations, write the result into args).


def gi_1d_output_axis_key(gi_mode_1d: str | None) -> str:
    """Which 1D integration range arg controls the *output* axis for a GI mode.

    gid maps ``radial_range`` → in-plane (ip) and ``azimuth_range`` → out-of-plane
    (oop) output, so:

      * ``q_total`` / ``q_ip`` → in-plane output  → ``radial_range``
      * ``q_oop`` / ``exit_angle`` → out-of-plane output → ``azimuth_range``

    Freezing the wrong key leaves the real output axis auto-ranging per incidence
    angle, so it drifts across an angle scan → a non-uniform stack the writer
    rejects.  The 1D output axis is always stored in ``IntegrationResult1D.radial``
    regardless of mode.
    """
    return ('azimuth_range'
            if gi_mode_1d in ('q_oop', 'exit_angle')
            else 'radial_range')


def _padded_union_range(axes, pad_fraction: float = 0.02):
    """Padded union of the finite extents of several axis arrays.

    Returns ``(lo - pad, hi + pad)`` where ``lo`` is the min over all axes' finite
    minima and ``hi`` the max over their finite maxima (so the range brackets
    EVERY scout, not just one), with ``pad = max(span * pad_fraction, 1e-9)``.

    Returns ``None`` when no axis has finite samples, or the union is *collapsed*
    (span <= 0 — e.g. every scout degenerate at a 0° incidence): freezing a
    padded tiny range from a collapsed axis would clamp the whole scan onto it
    and blank the output, so the caller leaves the range unfrozen and surfaces
    the problem instead.
    """
    los, his = [], []
    for axis in axes:
        if axis is None:
            continue
        arr = np.asarray(axis, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        los.append(float(arr.min()))
        his.append(float(arr.max()))
    if not los:
        return None
    lo, hi = min(los), max(his)
    span = hi - lo
    if span <= 0:
        return None
    pad = max(span * pad_fraction, 1e-9)
    return lo - pad, hi + pad


def freeze_common_axis(results, *, gi_mode_1d: str | None = None,
                       pad_fraction: float = 0.02):
    """Compute the frozen 1D output-axis range from one or more scout results.

    Parameters
    ----------
    results
        A single :class:`IntegrationResult1D` or an iterable of them — the scout
        integrations (e.g. the lowest- and highest-incidence frames) whose output
        extents should be bracketed.
    gi_mode_1d
        The GI 1D mode; selects the output arg key (see
        :func:`gi_1d_output_axis_key`).
    pad_fraction
        Small fractional margin on the unioned span (default 2%).  It is
        load-bearing for COVERAGE: a fresh per-frame integration can land a few
        bins beyond the scout's auto-range extent (binning discretization), and
        without the margin the frozen range would CLIP that real data (see
        tests/.../test_gi_freeze_covers_last_frame_extent).  The margin makes
        empty bins beyond the real data; those should be filled with NaN (not a
        spurious dummy) so they are not plotted — see the q_total floor below
        and the NaN-empty follow-up.  For ``q_total`` the low margin is clamped
        to 0 (a magnitude can't be negative).

    Returns
    -------
    (key, range)
        ``key`` is the integration arg name (``'radial_range'`` or
        ``'azimuth_range'``); ``range`` is the padded union ``(lo, hi)`` covering
        every scout, or ``None`` if no scout yields a finite, non-collapsed
        extent (caller should then leave the range unfrozen).
    """
    if isinstance(results, IntegrationResult1D):
        results = [results]
    key = gi_1d_output_axis_key(gi_mode_1d)
    rng = _padded_union_range(
        [getattr(r, 'radial', None) for r in results], pad_fraction)
    # q_total is a MAGNITUDE (>= 0).  The symmetric pad_fraction must not push
    # its lower bound below 0 — a negative frozen radial_range makes pyFAI bin
    # empty q < 0 cells that integrate to a spurious flat dummy line at low Q
    # (visible as points at negative Q in the 1D plot).  q_ip (signed in-plane)
    # and q_oop (this codebase allows a signed out-of-plane scout) are left
    # alone; ``None`` defaults to q_total.  Mirrors _clamp_angle in
    # freeze_common_axes_2d (which keeps angular ranges within [-180, 180]).
    if rng is not None and gi_mode_1d in (None, 'q_total'):
        rng = (max(0.0, float(rng[0])), float(rng[1]))
    return key, rng


def freeze_common_axes_2d(results, *, gi_mode_2d: str = 'qip_qoop',
                          pad_fraction: float = 0.02) -> dict:
    """Compute the frozen 2D ranges from one or more scout results.

    Returns a ``{arg_key: (lo, hi)}`` dict for the GI 2D range keys — ``x_range``/
    ``y_range`` for ``qip_qoop``, else ``radial_range``/``azimuth_range`` — each
    the padded union (covering every scout) of that axis' extents
    (``.radial`` → x/radial, ``.azimuthal`` → y/azimuth).  A key whose union is
    missing or collapsed is omitted, so the caller leaves it unfrozen.
    """
    if isinstance(results, IntegrationResult2D):
        results = [results]
    results = list(results)
    x_key, y_key = (('x_range', 'y_range') if gi_mode_2d == 'qip_qoop'
                    else ('radial_range', 'azimuth_range'))
    out: dict = {}
    rx = _padded_union_range(
        [getattr(r, 'radial', None) for r in results], pad_fraction)
    ry = _padded_union_range(
        [getattr(r, 'azimuthal', None) for r in results], pad_fraction)
    def _clamp_angle(rng):
        # pyFAI's polar/exit-angle domain is [-180, 180); a padded scout
        # range that overflows it (e.g. -183..186 from a full-wedge scout)
        # gets WRAPPED by pyFAI (-183 -> +177), collapsing the request to a
        # sliver and producing an all-dummy cake.
        if rng is None:
            return None
        return (max(float(rng[0]), -180.0), min(float(rng[1]), 180.0))

    angle_x = gi_mode_2d == 'exit_angles'
    angle_y = gi_mode_2d in ('q_chi', 'exit_angles')
    if rx is not None:
        out[x_key] = _clamp_angle(rx) if angle_x else rx
    if ry is not None:
        out[y_key] = _clamp_angle(ry) if angle_y else ry
    return out


__all__ = [
    "create_fiber_integrator",
    "integrate_gi_1d",
    "integrate_gi_2d",
    "integrate_gi_exitangles",
    "integrate_gi_exitangles_1d",
    "integrate_gi_polar",
    "integrate_gi_polar_1d",
    "gi_1d_output_axis_key",
    "freeze_common_axis",
    "freeze_common_axes_2d",
]
