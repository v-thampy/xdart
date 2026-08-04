"""Compact detector facts projected from the mounted calibration."""

from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path

from xrd_tools.core.containers import PONI
from xrd_tools.session.detector_limits import detector_saturation_ceiling


def detector_summary(poni_file: str, mask_file: str) -> str:
    """Mirror the production detector, distance, and fitted summary."""

    return _detector_projection(poni_file, mask_file)[0]


def detector_calibration_ready(poni_file: str) -> bool:
    """Mirror production's parsed-PONI calibration readiness fact."""

    return _detector_projection(poni_file, "")[1]


def poni_saturation_ceiling(poni_file: str) -> float | None:
    """The raw-dtype saturation ceiling of the PONI-declared detector family
    (the value Mask Saturated keys off), or ``None`` when no valid PONI is
    mounted or the family's raw dtype is unknown (LV-UI-11: the GUI shows a
    blank manual-threshold default rather than a guessed number)."""

    return _detector_projection(poni_file, "")[2]


def _detector_projection(
    poni_file: str,
    mask_file: str,
) -> tuple[str, bool, float | None]:
    path = Path(str(poni_file or "")).expanduser()
    if poni_file:
        try:
            stat = path.stat()
            return _cached_poni_projection(
                str(path.resolve()), stat.st_mtime_ns, stat.st_size
            )
        except Exception:
            # The production readiness contract fails closed for every parser
            # error (including PyYAML's non-ValueError syntax exceptions).
            return f"{path.name} · selected", False, None
    mask_name = Path(str(mask_file or "")).name
    return (
        (f"{mask_name} · mask selected", False, None)
        if mask_name
        else ("not configured", False, None)
    )


@lru_cache(maxsize=32)
def _cached_poni_projection(
    path: str,
    _mtime_ns: int,
    _size: int,
) -> tuple[str, bool, float | None]:
    poni = PONI.from_poni_file(path)
    detector = str(poni.detector or "").strip()
    if detector.lower() in {"", "none", "detector"}:
        detector = ""
    ceiling = detector_saturation_ceiling(detector)
    try:
        distance = float(poni.dist)
    except (TypeError, ValueError):
        distance = 0.0
    parts = [detector] if detector else []
    if math.isfinite(distance) and distance > 0.0:
        parts.append(f"{distance * 1000.0:.1f}mm")
    if not parts:
        return f"{Path(path).name} · selected", True, ceiling
    return " · ".join((*parts, "fitted")), True, ceiling


__all__ = [
    "detector_calibration_ready",
    "detector_summary",
    "poni_saturation_ceiling",
]
