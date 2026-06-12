"""
Calibration helpers bridging ``xrd_tools`` containers and pyFAI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyFAI
from pyFAI.detectors import detector_factory
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

from xrd_tools.core.containers import PONI
from xrd_tools.io.image import get_detector_mask as _get_detector_mask

if TYPE_CHECKING:
    from pyFAI.detectors import Detector
    from pyFAI.integrator.fiber import FiberIntegrator

logger = logging.getLogger(__name__)


def _as_path(path: Path | str) -> Path:
    return path if isinstance(path, Path) else Path(path)


def load_poni(path: Path | str) -> PONI:
    """
    Load a pyFAI ``.poni`` file into the project ``PONI`` dataclass.

    Uses :class:`pyFAI.io.ponifile.PoniFile` rather than the legacy
    ``pyFAI.load()`` entry point.  ``pyFAI.load()`` would silently
    swallow parse failures and return a default-initialised
    ``AzimuthalIntegrator`` (``dist=1.0``, no wavelength, generic
    ``Detector``), which made save→load round-trip failures look
    like value mismatches rather than parse errors.  The ``PoniFile``
    parser raises on bad input and works identically across pyFAI
    2025.x and 2026.x .poni format variants.

    Parameters
    ----------
    path : Path or str
        Path to a ``.poni`` calibration file.

    Returns
    -------
    PONI
        Calibration geometry extracted from the file.
    """
    from pyFAI.io.ponifile import PoniFile

    pf = PoniFile(str(_as_path(path)))
    det = pf.detector
    detector_name = getattr(det, "name", "") if det is not None else ""
    wl = pf.wavelength
    return PONI(
        dist=float(pf.dist),
        poni1=float(pf.poni1),
        poni2=float(pf.poni2),
        rot1=float(pf.rot1),
        rot2=float(pf.rot2),
        rot3=float(pf.rot3),
        wavelength=0.0 if wl is None else float(wl),
        detector=str(detector_name or ""),
    )


def save_poni(poni: PONI, path: Path | str) -> None:
    """
    Save a project ``PONI`` dataclass to a pyFAI ``.poni`` file.

    Routes through :class:`pyFAI.io.ponifile.PoniFile` for the same
    reason :func:`load_poni` does — the ``PoniFile.write`` path is
    version-stable across pyFAI 2025.x / 2026.x, while
    ``AzimuthalIntegrator.save`` switched its on-disk format between
    minor versions.  Falls back to ``ai.save()`` if the dataclass
    can't be expressed via the public ``PoniFile`` constructor (very
    old pyFAI without ``read_from_dict``).

    Parameters
    ----------
    poni : PONI
        Calibration geometry to save.
    path : Path or str
        Output ``.poni`` path.
    """
    out_path = _as_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ai = poni_to_integrator(poni)
    try:
        from pyFAI.io.ponifile import PoniFile

        pf = PoniFile(ai)
        with open(out_path, "w") as f:
            pf.write(f)
    except Exception:
        # Last-resort fallback — older pyFAI without PoniFile, or a
        # quirk in PoniFile.write on this version.  Logs at debug
        # because the legacy save path normally works too.
        logger.debug("PoniFile.write failed; falling back to ai.save", exc_info=True)
        ai.save(str(out_path))


def poni_to_integrator(poni: PONI) -> AzimuthalIntegrator:
    """
    Convert a project ``PONI`` dataclass to a pyFAI integrator.

    Parameters
    ----------
    poni : PONI
        Calibration geometry container.

    Returns
    -------
    AzimuthalIntegrator
        Configured pyFAI azimuthal integrator.
    """
    # 'Detector' is pyFAI's generic base-class name — treat it as unspecified.
    _det_name = poni.detector or ""
    detector = get_detector(_det_name) if _det_name and _det_name.lower() != "detector" else None
    return AzimuthalIntegrator(
        dist=float(poni.dist),
        poni1=float(poni.poni1),
        poni2=float(poni.poni2),
        rot1=float(poni.rot1),
        rot2=float(poni.rot2),
        rot3=float(poni.rot3),
        wavelength=float(poni.wavelength) if poni.wavelength else None,
        detector=detector,
    )


def get_detector(name: str | Detector) -> Detector:
    """
    Get a detector instance from pyFAI's registry.

    Parameters
    ----------
    name : str or Detector
        pyFAI detector name (e.g. ``"Pilatus300k"``) or an already-constructed
        pyFAI ``Detector`` instance, which is returned as-is.

    Returns
    -------
    Detector
        Configured pyFAI detector instance.

    Raises
    ------
    ValueError
        If the detector name string is not recognized by pyFAI.
    """
    from pyFAI.detectors import Detector as _Detector
    if isinstance(name, _Detector):
        return name
    try:
        return detector_factory(name)
    except Exception as exc:
        raise ValueError(
            f"Unknown pyFAI detector {name!r}. "
            "Use a detector name from the pyFAI registry."
        ) from exc


def get_detector_mask(name: str) -> np.ndarray | None:
    """
    Get the bad-pixel mask for a detector from pyFAI's registry.

    Parameters
    ----------
    name : str
        pyFAI detector name.

    Returns
    -------
    np.ndarray or None
        Boolean mask, or ``None`` if the detector is unknown.
    """
    return _get_detector_mask(name)


def poni_to_fiber_integrator(
    poni: PONI,
    incident_angle: float,
    tilt_angle: float = 0.0,
    sample_orientation: int = 1,
    angle_unit: str = "deg",
) -> FiberIntegrator:
    """
    Convert a project ``PONI`` dataclass to a pyFAI FiberIntegrator.

    This is a convenience re-export of
    :func:`~xrd_tools.integrate.gid.create_fiber_integrator`.
    The ``gid`` version caches incident/tilt angles on the instance so that
    the ``integrate_gi_*`` helpers can re-inject them on every call (pyFAI
    resets its internal cache after each integration).

    Parameters
    ----------
    poni : PONI
        Calibration geometry container.
    incident_angle : float
        Incidence angle of the X-ray beam on the sample surface.
    tilt_angle : float, optional
        Tilt angle of the sample.
    sample_orientation : int, optional
        EXIF-convention sample orientation (1–8).  Default ``1`` means the
        detector is horizontal with the beam arriving from the left.
    angle_unit : str, optional
        ``"deg"`` (default) or ``"rad"``.  If ``"deg"``, angles are
        converted to radians internally because FiberIntegrator works in
        radians.

    Returns
    -------
    FiberIntegrator
        Configured pyFAI fiber integrator.

    Raises
    ------
    ImportError
        If the installed pyFAI version does not support FiberIntegrator.
    """
    from xrd_tools.integrate.gid import create_fiber_integrator

    return create_fiber_integrator(
        poni,
        incident_angle=incident_angle,
        tilt_angle=tilt_angle,
        sample_orientation=sample_orientation,
        angle_unit=angle_unit,
    )


__all__ = [
    "get_detector",
    "get_detector_mask",
    "load_poni",
    "poni_to_fiber_integrator",
    "poni_to_integrator",
    "save_poni",
]
