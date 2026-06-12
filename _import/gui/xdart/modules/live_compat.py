"""Compatibility helpers for xdart live-object naming transitions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


# Maps legacy serialized class names (as they appear in pre-rename .nxs
# provenance) to the current canonical names + module paths.  Keys are the
# historical strings on disk; values are today's locations
# (ewald.frame / ewald.scan / ewald.frame_series).  Pure string mapping —
# nothing here is imported.
LEGACY_LIVE_NAME_ALIASES: dict[str, str] = {
    "EwaldArch": "LiveFrame",
    "xdart.modules.ewald.arch.EwaldArch": "xdart.modules.ewald.frame.LiveFrame",
    "EwaldSphere": "LiveScan",
    "xdart.modules.ewald.sphere.EwaldSphere": "xdart.modules.ewald.scan.LiveScan",
    "ArchSeries": "LiveFrameSeries",
    "xdart.modules.ewald.arch_series.ArchSeries": (
        "xdart.modules.ewald.frame_series.LiveFrameSeries"
    ),
}


def normalize_live_class_name(value: str) -> str:
    """Return the canonical Live* name for a legacy serialized class name."""
    return LEGACY_LIVE_NAME_ALIASES.get(value, value)


def normalize_live_class_names(value: Any) -> Any:
    """Recursively normalize legacy Ewald*/ArchSeries names in reader data.

    Old xdart 0.37.x files may include class names inside provenance or
    configuration dictionaries.  The on-disk schema stays unchanged here; this
    helper only normalizes values after reading them.
    """
    if isinstance(value, str):
        return normalize_live_class_name(value)
    if isinstance(value, Mapping):
        return {
            key: normalize_live_class_names(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(normalize_live_class_names(item) for item in value)
    if isinstance(value, list):
        return [normalize_live_class_names(item) for item in value]
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
    ):
        try:
            return type(value)(
                normalize_live_class_names(item) for item in value
            )
        except TypeError:
            return value
    return value


__all__ = [
    "LEGACY_LIVE_NAME_ALIASES",
    "normalize_live_class_name",
    "normalize_live_class_names",
]
