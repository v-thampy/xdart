"""
Calibration helpers bridging ``ssrl_xrd_tools`` containers and pyFAI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyFAI
from pyFAI.detectors import detector_factory
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

from ssrl_xrd_tools.core.containers import PONI
from ssrl_xrd_tools.io.image import get_detector_mask as _get_detector_mask

if TYPE_CHECKING:
    from pyFAI.detectors import Detector
    from pyFAI.integrator.fiber import FiberIntegrator

logger = logging.getLogger(__name__)


def _as_path(path: Path | str) -> Path:
    return path if isinstance(path, Path) else Path(path)


def load_poni(path: Path | str) -> PONI:
    """
    Load a pyFAI ``.poni`` file into the project ``PONI`` dataclass.

    Parameters
    ----------
    path : Path or str
        Path to a ``.poni`` calibration file.

    Returns
    -------
    PONI
        Calibration geometry extracted from the file.
    """
    ai = pyFAI.load(_as_path(path))
    detector_name = getattr(getattr(ai, "detector", None), "name", "")
    return PONI(
        dist=float(ai.dist),
        poni1=float(ai.poni1),
        poni2=float(ai.poni2),
        rot1=float(ai.rot1),
        rot2=float(ai.rot2),
        rot3=float(ai.rot3),
        wavelength=0.0 if ai.wavelength is None else float(ai.wavelength),
        detector=str(detector_name or ""),
    )


def save_poni(poni: PONI, path: Path | str) -> None:
    """
    Save a project ``PONI`` dataclass to a pyFAI ``.poni`` file.

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
    :func:`~ssrl_xrd_tools.integrate.gid.create_fiber_integrator`.
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
    from ssrl_xrd_tools.integrate.gid import create_fiber_integrator

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
