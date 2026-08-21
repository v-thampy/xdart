"""
Calibration helpers bridging ``xrd_tools`` containers and pyFAI.
"""

from __future__ import annotations

import inspect, json, logging, math, os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from xrd_tools.core.containers import PONI
from xrd_tools.io.image import get_detector_mask as _get_detector_mask

# pyFAI is imported LAZILY (inside the functions that use it) — never at module
# level.  pyFAI loads a Qt binding at import (PyQt5 by default when QT_API is
# unset), so an eager import here would let merely importing this module pin the
# wrong Qt binding in a notebook/headless process.  xrd_tools is headless-first;
# the concrete integrator/detector are pulled in only when actually building one.
if TYPE_CHECKING:
    from pyFAI.detectors import Detector
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    from pyFAI.integrator.fiber import FiberIntegrator
    from xrd_tools.core.geometry.diffractometer import DetectorCalibration

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


def load_detector_calibration(
    path: Path | str, *, data: bytes | None = None,
) -> DetectorCalibration:
    """Strictly load a bounded PONI 2.x file without losing detector config."""
    from pyFAI.detectors import ALL_DETECTORS
    from pyFAI.io.ponifile import PoniFile
    from xrd_tools.core.geometry.diffractometer import DetectorCalibration

    if data is None:
        with open(_as_path(path), "rb") as stream:
            data = stream.read((1 << 20) + 1)
    if not data or len(data) > (1 << 20):
        raise ValueError("PONI must be nonempty and no larger than 1 MiB")
    text = data.decode("utf-8", errors="strict")
    allowed = {"poni_version", "detector", "detector_config", "distance", "dist", "poni1",
               "poni2", "rot1", "rot2", "rot3", "wavelength"}
    raw: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line: raise ValueError("malformed noncomment PONI line")
        key, value = line.split(":", 1)
        key = key.strip().casefold()
        if not key or key in raw or key not in allowed: raise ValueError("duplicate or unsupported PONI key")
        raw[key] = value.strip()
    if "distance" in raw and "dist" in raw: raise ValueError("duplicate PONI distance")
    def no_duplicate_json(pairs):
        result = dict(pairs)
        if len(result) != len(pairs): raise ValueError("duplicate detector-config key")
        return result
    try:
        version = float(raw["poni_version"])
        config = json.loads(raw["detector_config"], object_pairs_hook=no_duplicate_json,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid PONI version or detector config") from exc
    if version not in {2.0, 2.1} or type(config) is not dict: raise ValueError(
        "only configured PONI 2 and 2.1 are supported")
    if version == 2.1 and "orientation" not in config: raise ValueError("PONI 2.1 requires detector orientation")
    config.setdefault("orientation", 3)
    if type(config["orientation"]) is not int or config["orientation"] not in range(1, 5):
        raise ValueError("detector orientation must be an integer from 1 through 4")
    json.dumps(config, allow_nan=False)
    name = raw.get("detector", "").strip()
    unsafe_name = (not name or any(mark in name for mark in ("\0", "/", "\\", "://"))
                   or len(name) > 1 and name[0].isalpha() and name[1] == ":"
                   or os.path.exists(name))
    folded = name.casefold()
    detector_class = (ALL_DETECTORS.get(folded) or ALL_DETECTORS.get(
        folded.replace(" ", "_")) or ALL_DETECTORS.get(folded.replace(" ", "")))
    if unsafe_name or detector_class is None: raise ValueError("detector must be a simple registered name")
    constructor_keys = set(inspect.getfullargspec(detector_class).args) - {"self"}
    if not set(config) <= constructor_keys | {"binning"} or any(
        token in key.casefold() for key in config for token in
        ("spline", "file", "path", "uri", "url")
    ): raise ValueError("detector configuration contains an unsafe key")
    def pathlike(value) -> bool:
        if type(value) is str:
            return True
        items = value.values() if type(value) is dict else value if type(value) is list else ()
        return any(pathlike(item) for item in items)
    if pathlike(config): raise ValueError("detector configuration contains a path-like value")
    mapping = dict(raw, poni_version=version, detector=name, detector_config=config)
    pf = PoniFile(mapping)
    detector = pf.detector
    values = (pf.dist, pf.poni1, pf.poni2, pf.rot1, pf.rot2, pf.rot3)
    wavelength = 0.0 if pf.wavelength is None else float(pf.wavelength)
    try: geometry = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc: raise ValueError("PONI geometry is incomplete") from exc
    if not all(map(math.isfinite, (*geometry, wavelength))) or geometry[0] <= 0 or wavelength < 0: raise ValueError(
        "PONI geometry is non-finite or out of range")
    poni = PONI(*geometry, wavelength, detector.__class__.__name__)
    calibration = DetectorCalibration(poni, detector.get_config())
    normalized = dict(calibration.detector_config)
    json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if any(key not in normalized or json.dumps(normalized[key], sort_keys=True) != json.dumps(value, sort_keys=True) for key, value in config.items()): raise ValueError(
        "detector configuration did not survive construction")
    rebuilt = detector_calibration_to_integrator(calibration).detector
    identity = lambda item: (type(item), item.shape, item.max_shape, item.pixel1,
        item.pixel2, int(item.orientation),
        DetectorCalibration(poni, item.get_config()).to_json())
    if identity(detector) != identity(rebuilt): raise ValueError("detector configuration did not survive reconstruction")
    return calibration


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
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
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


def detector_calibration_to_integrator(
    cal, *, rot1: float | None = None, rot2: float | None = None,
    rot3: float | None = None,
) -> AzimuthalIntegrator:
    """Build a pyFAI integrator from a :class:`DetectorCalibration`, honouring
    its ``Detector_config`` (panel orientation) — unlike :func:`poni_to_integrator`,
    which rebuilds the detector at default config and silently drops a non-default
    orientation (stitching GAP B).

    ``rot1``/``rot2``/``rot3`` override the base rotations to build a *per-frame*
    integrator (base calibration ⊕ ``Diffractometer.to_pyfai_per_frame`` rotations).

    Parameters
    ----------
    cal : DetectorCalibration
        Base calibration (``poni`` + ``detector_config``).
    rot1, rot2, rot3 : float, optional
        Per-frame rotations (rad); ``None`` keeps the base ``poni`` value.
    """
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator  # noqa: PLC0415
    from pyFAI.detectors import detector_factory  # noqa: PLC0415

    p = cal.poni
    cfg = dict(cal.detector_config or {})
    name = p.detector or ""
    if name and name.lower() != "detector":
        det = detector_factory(name, config=cfg) if cfg else detector_factory(name)
    else:
        det = None
    return AzimuthalIntegrator(
        dist=float(p.dist), poni1=float(p.poni1), poni2=float(p.poni2),
        rot1=float(p.rot1 if rot1 is None else rot1),
        rot2=float(p.rot2 if rot2 is None else rot2),
        rot3=float(p.rot3 if rot3 is None else rot3),
        wavelength=float(p.wavelength) if p.wavelength else None,
        detector=det,
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
    from pyFAI.detectors import Detector as _Detector, detector_factory
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
    "load_detector_calibration",
    "poni_to_fiber_integrator",
    "poni_to_integrator",
    "save_poni",
]
