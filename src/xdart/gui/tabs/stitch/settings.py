"""Saved Stitch settings: every page input, the source files, the threshold policy
and the calibration dialog, as one versioned JSON document.

File ▸ Save/Load Settings writes and reads it, and the page restores the last
session's document at start-up, through the shared settings owner
(``xdart.gui.tools.page_settings``: the document's checks and text, the file and
the session). This module is Stitch's own part: what it saves and the order it
applies in. Per-use confirmations in the calibration dialog are never kept.
Unknown keys are ignored and missing ones keep their defaults, so a document
from another version restores what it can.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from xdart.gui.tools.page_settings import (
    SettingsKind, restore_sources, restore_write_mode, saved_fields, saved_threshold,
)

#: The document's kind and version, its key in the session file, and its own object section.
STITCH_SETTINGS = SettingsKind("xdart-stitch-settings", 1, "stitch_settings", "Stitch", sections=("gi",))
SETTINGS_KIND = STITCH_SETTINGS.kind
SETTINGS_VERSION = STITCH_SETTINGS.version
#: Where the last session's settings live in the session file.
SESSION_KEY = STITCH_SETTINGS.session_key
#: Applied first, in this order: each resets other fields to its own defaults.
_FIRST = ("backend", "source_kind", "unit", "map_axis")
#: Shown from the source selection, not an input of its own.
_DERIVED = frozenset({"spec_path"})
#: Restored as one threshold policy, not field by field.
_THRESHOLD = frozenset({"threshold_min", "threshold", "threshold_enabled"})


def page_settings(page) -> dict[str, Any]:
    """The page's current settings document."""
    return STITCH_SETTINGS.document(
        page, (key for key in page._fields if key not in _DERIVED),
        # grazing incidence as Int sets it (its axis and ranges are fields above)
        gi=dict(page._gi),
        # Append skips an output already stitched with these settings; "Overwrite" is Replace
        write_mode=page.run_bar.write_mode(),
        # Stitch 1D or Stitch 2D
        mode=page._mode,
        # Int's Cores: the frames pyFAI integrates at once
        cores=page.run_bar.get_cores(),
    )


#: Return the document if it is Stitch settings this version can read.
validate_settings = STITCH_SETTINGS.validate
dump_settings = STITCH_SETTINGS.dump
load_settings = STITCH_SETTINGS.load


def apply_page_settings(page, document: Any) -> None:
    """Restore a settings document onto the page (see the module docstring)."""
    document = validate_settings(document)
    fields = saved_fields(page, document, _DERIVED)
    for key in _FIRST:
        if key in fields:
            page.set_field(key, fields[key])
    restore_sources(page, document)
    for key in page._fields:  # the page's own order, as RSM's: a document's (sorted in its file) is no order of the
        # page's, and in it a blank Save Path came before the Project that fills it
        if key in fields and key not in _FIRST and key not in _THRESHOLD and key != "normalization":
            page.set_field(key, fields[key])
    if "normalization" in fields:  # last: the saved choice wins over a monitor's implied one
        page.set_field("normalization", fields["normalization"])
    saved = document.get("fields", {})
    if "camera" not in saved and str(saved.get("image_rotation", "")).strip() in ("0", "90", "180", "270"):
        from xrd_tools.core.geometry import ImageOrientation  # older settings: the rotation's unmirrored camera
        from .page import camera_text
        page.set_field("camera", camera_text(ImageOrientation(rotation=int(saved["image_rotation"]))))
    state = saved_threshold(document)
    if state is not None:
        page._threshold_state = state
        mode = page._threshold_mode
        for key, value in (("threshold_min", state.threshold_min if mode == "manual" else 0.),
                           ("threshold", state.threshold_max),
                           ("threshold_enabled", mode != "off")):
            page._fields[key] = replace(page._fields[key], value=value)
        page._render_ranges()
    page._set_calibration_settings(document.get("calibration"))
    restore_write_mode(page, document)
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


__all__ = [
    "SESSION_KEY", "SETTINGS_KIND", "SETTINGS_VERSION", "STITCH_SETTINGS", "apply_page_settings",
    "dump_settings", "load_settings", "page_settings", "validate_settings",
]
