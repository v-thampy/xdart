# -*- coding: utf-8 -*-
"""Overlay/Waterfall identity helpers shared by the GUI adapter and plot mixin."""

from __future__ import annotations

import re

import numpy as np

from .display_constants import Chi, Qip_s, Qoop_s, Qtot_s, Th
from .display_logic import (
    frame_index_from_qualified_id,
    overlay_grid_keys_compatible,
    overlay_grid_reset_key,
    qualified_frame_id,
    scan_key_from_qualified_id,
)


def current_scan_key(widget):
    """Stable scan key for row identity; prefer the visible scan name."""
    scan = getattr(widget, "scan", None)
    if scan is None:
        return None
    name = getattr(scan, "name", None)
    if name not in (None, "", "null_main"):
        return name
    return getattr(scan, "data_file", None) or name


def qualified_row_id_for_widget(widget, frame_idx):
    return qualified_frame_id(current_scan_key(widget), frame_idx)


def qualified_row_ids_for_widget(widget, frame_idxs):
    return tuple(qualified_row_id_for_widget(widget, idx) for idx in frame_idxs)


def frame_index_from_row_id(row_id):
    return frame_index_from_qualified_id(row_id)


def scan_key_from_row_id(row_id):
    return scan_key_from_qualified_id(row_id)


def row_id_belongs_to_widget_scan(widget, row_id):
    scan_key = scan_key_from_row_id(row_id)
    return scan_key is None or scan_key == current_scan_key(widget)


def current_axis_info(widget):
    try:
        idx = int(widget.ui.plotUnit.currentIndex())
    except Exception:
        return {"source": "1d", "slice_axis": None, "axis": None}
    info = getattr(widget, "_plot_axis_info", ())
    if 0 <= idx < len(info):
        return dict(info[idx] or {})
    return {"source": "1d", "slice_axis": None, "axis": None}


def slice_enabled(widget):
    try:
        return bool(widget.ui.slice.isChecked())
    except Exception:
        return False


def overlay_needs_2d(widget, axis_info=None):
    axis_info = axis_info or current_axis_info(widget)
    source = axis_info.get("source", "1d")
    return (source == "2d") or (source == "1d_2d" and slice_enabled(widget))


def overlay_slice_key(widget, needs_2d):
    if not needs_2d:
        return None
    try:
        return (
            float(widget.ui.slice_center.value()),
            float(widget.ui.slice_width.value()),
        ) if slice_enabled(widget) else (None, None)
    except Exception:
        return (None, None)


def _plot_unit_text(widget):
    try:
        return str(widget.ui.plotUnit.currentText())
    except Exception:
        return ""


def _axis_token_from_text(text):
    lowered = str(text or "").lower()
    head = re.sub(r"\s*\(.*?\)\s*$", "", lowered).strip()
    if Qip_s.lower() in lowered:
        return "q_ip"
    if Qoop_s.lower() in lowered:
        return "q_oop"
    if Qtot_s.lower() in lowered:
        return "q_total"
    if "exit" in lowered:
        return "exit_angle"
    if f"{Chi.lower()}gi" in lowered or "chigi" in lowered:
        return "chi_gi"
    if Chi.lower() in lowered or "chi" in lowered:
        return "chi"
    if Th.lower() in lowered or "theta" in lowered or "2th" in lowered:
        return "radial"
    if head.startswith("q") or "q" in head:
        return "radial"
    return head or "unknown"


def overlay_axis_kind(widget, axis_info=None, needs_2d=None):
    axis_info = axis_info or current_axis_info(widget)
    if needs_2d is None:
        needs_2d = overlay_needs_2d(widget, axis_info)
    scan = getattr(widget, "scan", None)
    gi = bool(getattr(scan, "gi", False)) if scan is not None else False
    axis = axis_info.get("axis")
    if needs_2d:
        if gi:
            return _axis_token_from_text(_plot_unit_text(widget))
        if axis == "azimuthal":
            return "chi"
        return "radial"
    if gi:
        args = getattr(scan, "bai_1d_args", {}) or {}
        return str(args.get("gi_mode_1d") or _axis_token_from_text(_plot_unit_text(widget)))
    if axis == "azimuthal":
        return "chi"
    args = getattr(scan, "bai_1d_args", {}) or {}
    unit = str(args.get("unit", "") or "").lower()
    if "chigi" in unit:
        return "chi_gi"
    if "chi" in unit:
        return "chi"
    return "radial"


def _positive_int(value):
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def _planned_npt(widget, axis_info, needs_2d):
    scan = getattr(widget, "scan", None)
    if scan is None:
        return None
    if needs_2d:
        args = getattr(scan, "bai_2d_args", {}) or {}
        if axis_info.get("axis") == "azimuthal":
            keys = ("npt_azim", "npt_azimuthal", "numpoints_azimuthal")
        else:
            keys = ("npt_rad", "npt_radial", "numpoints_radial")
    else:
        args = getattr(scan, "bai_1d_args", {}) or {}
        keys = ("numpoints", "npt", "npt_1d")
    for key in keys:
        npt = _positive_int(args.get(key))
        if npt is not None:
            return npt
    return None


def _axis_len(values):
    if values is None:
        return None
    try:
        return int(np.asarray(values).size)
    except Exception:
        return None


def _actual_npt_from_frame(first_frame, axis_info, needs_2d):
    if first_frame is None:
        return None
    if needs_2d:
        result = getattr(first_frame, "int_2d", None)
        if result is None:
            return None
        attr = "azimuthal" if axis_info.get("axis") == "azimuthal" else "radial"
        return _axis_len(getattr(result, attr, None))
    result = getattr(first_frame, "int_1d", None)
    if result is None:
        return None
    return _axis_len(getattr(result, "radial", None))


def overlay_grid_key_for_widget(widget, *, npt=None, first_frame=None, axis_info=None):
    axis_info = axis_info or current_axis_info(widget)
    needs_2d = overlay_needs_2d(widget, axis_info)
    if npt is None:
        npt = _actual_npt_from_frame(first_frame, axis_info, needs_2d)
    if npt is None:
        npt = _planned_npt(widget, axis_info, needs_2d)
    return overlay_grid_reset_key(
        overlay_axis_kind(widget, axis_info, needs_2d),
        npt,
        needs_2d,
        overlay_slice_key(widget, needs_2d),
    )


def overlay_identity_for_widget(widget, frame_idx, *, npt=None, first_frame=None, axis_info=None):
    """Return ``(grid_key, row_id)`` for one Overlay/Waterfall row.

    Slice identity is intentionally isolated here: OV-6 still keeps slice range in
    the grid key (matching the current reset behavior), while OV-7 can move that
    component into ``row_id`` locally without touching payload/seeding call sites.
    """
    axis_info = axis_info or current_axis_info(widget)
    grid_key = overlay_grid_key_for_widget(
        widget, npt=npt, first_frame=first_frame, axis_info=axis_info)
    row_id = qualified_row_id_for_widget(widget, frame_idx)
    return grid_key, row_id


def overlay_grid_keys_match(left, right):
    return overlay_grid_keys_compatible(left, right)
