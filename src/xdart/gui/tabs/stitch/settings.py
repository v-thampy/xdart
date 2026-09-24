"""Saved Stitch settings: every page input, the source files, the threshold policy
and the calibration dialog, as one versioned JSON document.

File ▸ Save/Load Settings writes and reads it, and the page restores the last
session's document at start-up. Per-use confirmations in the calibration dialog
are never kept. Unknown keys are ignored and missing ones keep their defaults, so
a document from another version restores what it can.
"""
from __future__ import annotations

from dataclasses import replace
import json
import math
from typing import Any

SETTINGS_KIND = "xdart-stitch-settings"
SETTINGS_VERSION = 1
#: Where the last session's settings live in the session file.
SESSION_KEY = "stitch_settings"
#: Applied first, in this order: each resets other fields to its own defaults.
_FIRST = ("backend", "source_kind", "unit")
#: Shown from the source selection, not an input of its own.
_DERIVED = frozenset({"spec_path"})
#: Restored as one threshold policy, not field by field.
_THRESHOLD = frozenset({"threshold_min", "threshold", "threshold_enabled"})


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def page_settings(page) -> dict[str, Any]:
    """The page's current settings document."""
    state = page._threshold_state
    return {
        "kind": SETTINGS_KIND,
        "version": SETTINGS_VERSION,
        "fields": {key: _plain(page._value(key)) for key in page._fields if key not in _DERIVED},
        "sources": {str(kind): [str(path) for path in paths]
                    for kind, paths in page._paths_by_kind.items()},
        "threshold": {"apply_threshold": bool(state.apply_threshold),
                      "threshold_min": _plain(state.threshold_min),
                      "threshold_max": _plain(state.threshold_max),
                      "mask_saturation": bool(state.mask_saturation)},
        "calibration": page._calibration_settings_now(),
    }


def validate_settings(document: Any) -> dict[str, Any]:
    """Return the document if it is Stitch settings this version can read."""
    if (type(document) is not dict or document.get("kind") != SETTINGS_KIND
            or type(document.get("version")) is not int):
        raise ValueError("not an xdart Stitch settings file")
    if document["version"] > SETTINGS_VERSION:
        raise ValueError("these Stitch settings come from a newer xdart")
    for key, kind in (("fields", dict), ("sources", dict), ("threshold", dict)):
        if type(document.get(key, {})) is not kind:
            raise ValueError(f"Stitch settings {key} are malformed")
    calibration = document.get("calibration")
    if calibration is not None and type(calibration) is not dict:
        raise ValueError("Stitch settings calibration is malformed")
    return document


def apply_page_settings(page, document: Any) -> None:
    """Restore a settings document onto the page (see the module docstring)."""
    from xrd_tools.session.threshold_editing import ThresholdState

    document = validate_settings(document)
    fields = {key: value for key, value in document.get("fields", {}).items()
              if key in page._fields and key not in _DERIVED}
    for key in _FIRST:
        if key in fields:
            page.set_field(key, fields[key])
    page._paths_by_kind = {str(kind): tuple(str(path) for path in paths)
                           for kind, paths in document.get("sources", {}).items()
                           if type(paths) is list}
    page.set_sources(page._paths_by_kind.get(page._value("source_kind"), ()))
    for key, value in fields.items():
        if key not in _FIRST and key not in _THRESHOLD:
            page.set_field(key, value)
    threshold = document.get("threshold")
    if threshold:
        state = ThresholdState(bool(threshold.get("apply_threshold")),
                               threshold.get("threshold_min"), threshold.get("threshold_max"),
                               bool(threshold.get("mask_saturation")))
        page._threshold_state = state
        mode = page._threshold_mode
        for key, value in (("threshold_min", state.threshold_min if mode == "manual" else 0.),
                           ("threshold", state.threshold_max),
                           ("threshold_enabled", mode != "off")):
            page._fields[key] = replace(page._fields[key], value=value)
        page._render_ranges()
    page._set_calibration_settings(document.get("calibration"))


def dump_settings(document: dict[str, Any]) -> str:
    return json.dumps(validate_settings(document), indent=2, sort_keys=True)


def load_settings(text: str) -> dict[str, Any]:
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("not an xdart Stitch settings file") from error
    return validate_settings(document)


__all__ = [
    "SESSION_KEY", "SETTINGS_KIND", "SETTINGS_VERSION", "apply_page_settings",
    "dump_settings", "load_settings", "page_settings", "validate_settings",
]
