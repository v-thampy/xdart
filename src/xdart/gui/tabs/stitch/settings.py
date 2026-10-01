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
_FIRST = ("backend", "source_kind", "unit", "map_axis")
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
                    for kind, paths in page._paths_by_kind.items() if paths},
        "threshold": {"apply_threshold": bool(state.apply_threshold),
                      "threshold_min": _plain(state.threshold_min),
                      "threshold_max": _plain(state.threshold_max),
                      "mask_saturation": bool(state.mask_saturation)},
        "calibration": page._calibration_settings_now(),
        # grazing incidence as Int sets it (its axis and ranges are fields above)
        "gi": dict(page._gi),
        # Append skips an output already stitched with these settings; "Overwrite" is Replace
        "write_mode": page.run_bar.write_mode(),
        # Stitch 1D or Stitch 2D
        "mode": page._mode,
        # Int's Cores: the frames pyFAI integrates at once
        "cores": page.run_bar.get_cores(),
    }


def validate_settings(document: Any) -> dict[str, Any]:
    """Return the document if it is Stitch settings this version can read."""
    if (type(document) is not dict or document.get("kind") != SETTINGS_KIND
            or type(document.get("version")) is not int):
        raise ValueError("not an xdart Stitch settings file")
    if document["version"] > SETTINGS_VERSION:
        raise ValueError("these Stitch settings come from a newer xdart")
    for key, kind in (("fields", dict), ("sources", dict), ("threshold", dict), ("gi", dict)):
        if type(document.get(key, {})) is not kind:
            raise ValueError(f"Stitch settings {key} are malformed")
    calibration = document.get("calibration")
    if calibration is not None and (type(calibration) is not dict or any(
            type(calibration.get(key, {})) is not dict for key in ("fields", "choices"))):
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
        if key not in _FIRST and key not in _THRESHOLD and key != "normalization":
            page.set_field(key, value)
    if "normalization" in fields:  # last: the saved choice wins over a monitor's implied one
        page.set_field("normalization", fields["normalization"])
    saved = document.get("fields", {})
    if "camera" not in saved and str(saved.get("image_rotation", "")).strip() in ("0", "90", "180", "270"):
        from xrd_tools.core.geometry import ImageOrientation  # older settings: the rotation's unmirrored camera
        from .page import camera_text
        page.set_field("camera", camera_text(ImageOrientation(rotation=int(saved["image_rotation"]))))
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
    if document.get("write_mode") in ("Append", "Overwrite"):
        page.run_bar.set_write_mode(document["write_mode"])
        page._write_mode_changed()
    if document.get("mode") in ("Stitch 1D", "Stitch 2D"):
        page.select_mode(document["mode"])
    if type(document.get("cores")) is int:
        spin = page.run_bar.coresSpin
        spin.setValue(max(spin.minimum(), min(document["cores"], spin.maximum())))
    gi = document.get("gi", {})
    page._gi.update({key: bool(value) if key == "enabled" else None if value is None else str(value)
                     for key, value in gi.items() if key in page._gi})
    if "tilt_chi" in gi and "tilt_angle" not in gi:  # older settings: a Manual χ, now Int's tilt angle
        from xrd_tools.integrate.stitch_gi import tilt_angle_from_chi
        try:
            page._gi["tilt_angle"] = f"{tilt_angle_from_chi(float(gi['tilt_chi']), int(page._gi['orientation'])):g}"
        except (TypeError, ValueError, KeyError):
            pass  # an unreadable one: level
    # the mapping follows the restored geometry, and readiness rechecks the calibration's arm
    page._instrument_changed()
    # As after load_form: the XU asset owns some fields, now that its paths are back.
    page._sync_advanced()
    page._refresh_xu_geometry()
    page._render_gi()
    from xdart.gui.tabs.stitch.page import _LEGACY_AXES
    for key in ("unit", "map_axis"):  # Grazing's axes exist once Grazing is back
        value = _LEGACY_AXES.get(fields.get(key), fields.get(key))
        if value in page._fields[key].choices:
            page.set_field(key, value)
    for key in ("q_min", "q_max", "map_min", "map_max"):  # an axis's range, after the axis that resets it
        if key in fields:
            page.set_field(key, fields[key])


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
