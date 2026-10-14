"""Saved RSM settings: every page input, the source files, the threshold policy, the calibration window and
Append ⇄ Replace, as one versioned JSON document (G12, ledger row 3: "File ▸ Save/Load Settings and last-session
restore covering every input, the Advanced options and any calibration panel ... mirror the Stitch page").

The inputs are every row of Project, Experiment (the Calibration, the Mask File and the refinement toggle),
Source (with the Selection and the UB group), Processing (Coordinates, three grid counts and the
Normalization group) and Advanced (Offset, All-frame hot, the decoder, Angle mapping, frames, column occurrences, Corrections, chunk
and memory). The Threshold is Int's policy, kept as one state; the calibration window's own ``settings()`` never
hold a per-use confirmation.

File ▸ Save/Load Settings writes and reads it, and the page restores the last session's document at start-up,
through the shared settings owner (``xdart.gui.tools.page_settings``), as Stitch's. This module is RSM's own part:
what it saves and the order it applies in. A document is checked first: a value its row cannot hold (a choice
this version lacks) refuses it whole, naming the row. Unknown keys are ignored, and a missing field or section
(the sources, the threshold, the calibration window, Append ⇄ Replace) keeps the page's own, so an older or
partial document restores what it holds: the scan-43 one the tests ship names no Project.
"""
from __future__ import annotations

from typing import Any

from xdart.gui.tools.page_settings import (
    SettingsKind, check_fields, document_data_format, restore_sources, restore_write_mode, saved_fields,
    saved_threshold,
)

#: The document's kind and version, and its key in the session file.
RSM_SETTINGS = SettingsKind("xdart-rsm-settings", 1, "rsm_settings", "RSM", sections=("point_images",))
SESSION_KEY = RSM_SETTINGS.session_key
#: Not saved field by field: the SPEC files row shows the source selection, and the Threshold is one policy.
_NOT_FIELDS = frozenset({"spec_path", "threshold_enabled", "threshold_min", "threshold"})
#: Choices that are the first scan's own column names: a saved name stays whatever that scan records.
_OPEN_CHOICES = frozenset({"monitor", "foil_column", "exposure_column"})


def page_settings(page) -> dict[str, Any]:
    """The page's current settings document."""
    document = RSM_SETTINGS.document(
        page, (key for key in page._fields if key not in _NOT_FIELDS),
        # Append skips an output already gridded with these settings; "Overwrite" is Replace
        write_mode=page.run_bar.write_mode(),
        point_images=page.point_images.document(),
    )
    if page._output_suggested:  # the Output follows the Source until a name is chosen, and so it does restored
        document["fields"]["output_path"] = ""
    from .page import _UB_TYPED
    if page._value("ub_from") != _UB_TYPED:
        for key, value in zip(("ub_row1", "ub_row2", "ub_row3"), page._ub_typed_rows, strict=True):
            document["fields"][key] = value
    return document


def apply_page_settings(page, document: Any) -> None:
    """Restore a settings document onto the page (see the module docstring)."""
    document = RSM_SETTINGS.validate(document)
    if "point_images" in document:
        page.point_images.restore(document["point_images"])
    formats, conflict = document_data_format(document)  # an older document's Source kind, migrated first
    fields = saved_fields(page, document, _NOT_FIELDS)
    fields.update(formats)
    legacy = document.get("fields", {}).get("grid")
    if legacy is not None:
        from xdart.gui.tabs.rsm.page import _GRID_FIELDS, _GRIDS
        if legacy not in _GRIDS:
            raise ValueError(f"Grid: {legacy!r} is not one of its choices")
        for key, value in zip(_GRID_FIELDS, _GRIDS[legacy], strict=True):
            fields.setdefault(key, str(value))
    check_fields(page, fields, _OPEN_CHOICES)
    first = ("data_format", "input_mode")  # first: the format and input make the kind that shows its own rows
    for key in first:
        if key in fields:
            page.set_field(key, fields[key])
    if "sources" in document:  # a document without them keeps the page's
        restore_sources(page, document)
    for key in page._fields:  # the page's own order: a document's (sorted in its file) is no order of the page's
        if key in fields and key not in (*first, "normalization"):
            page.set_field(key, fields[key])
    if "normalization" in fields:  # last, as Stitch's: once the Source names the columns it reads
        page.set_field("normalization", fields["normalization"])
    state = saved_threshold(document)
    if state is not None:
        page._threshold_state = state
        page._render_threshold()
    if "calibration" in document:  # a document without it keeps the window's
        page._set_calibration_settings(document["calibration"])
    restore_write_mode(page, document)
    page._set_format_choice(conflict)  # and page._edited(), as before


__all__ = ["RSM_SETTINGS", "SESSION_KEY", "apply_page_settings", "page_settings"]
