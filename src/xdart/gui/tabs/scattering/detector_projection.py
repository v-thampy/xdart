"""Compact detector facts projected from the mounted calibration."""

from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path

from xrd_tools.core.containers import PONI


def detector_summary(poni_file: str, mask_file: str) -> str:
    """Mirror the production detector, distance, and fitted summary."""

    return _detector_projection(poni_file, mask_file)[0]


def detector_calibration_ready(poni_file: str) -> bool:
    """Mirror production's parsed-PONI calibration readiness fact."""

    return _detector_projection(poni_file, "")[1]


def poni_detector(poni_file: str) -> str:
    """The mounted PONI's detector name ("" when none or unreadable), for
    decision 39's documented counter limits (``resolve_detector_saturation``)."""

    return _detector_projection(poni_file, "")[2]


def _detector_projection(
    poni_file: str,
    mask_file: str,
) -> tuple[str, bool, str]:
    path = Path(str(poni_file or "")).expanduser()
    if poni_file:
        try:
            stat = path.stat()
            return _cached_poni_projection(
                str(path.resolve()), stat.st_mtime_ns, stat.st_size,
                stat.st_ctime_ns, stat.st_dev, stat.st_ino, stat.st_mode,
            )
        except Exception:
            # The production readiness contract fails closed for every parser
            # error (including PyYAML's non-ValueError syntax exceptions).
            return f"{path.name} · selected", False, ""
    mask_name = Path(str(mask_file or "")).name
    return (
        (f"{mask_name} · mask selected", False, "")
        if mask_name
        else ("not configured", False, "")
    )


@lru_cache(maxsize=32)
def _cached_poni_projection(
    path: str,
    _mtime_ns: int,
    _size: int,
    _ctime_ns: int,
    _device: int,
    _inode: int,
    _mode: int,
) -> tuple[str, bool, str]:
    poni = PONI.from_poni_file(path)
    detector = str(poni.detector or "").strip()
    if detector.lower() in {"", "none", "detector"}:
        detector = ""
    try:
        distance = float(poni.dist)
    except (TypeError, ValueError):
        distance = 0.0
    parts = [detector] if detector else []
    if math.isfinite(distance) and distance > 0.0:
        parts.append(f"{distance * 1000.0:.1f}mm")
    if not parts:
        return f"{Path(path).name} · selected", True, detector
    return " · ".join((*parts, "fitted")), True, detector


__all__ = [
    "detector_calibration_ready",
    "detector_summary",
    "poni_detector",
]
