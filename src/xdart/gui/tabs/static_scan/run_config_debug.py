# -*- coding: utf-8 -*-
"""Silent-by-default structured logging for run-configuration ownership.

R4-D diagnostics deliberately inspect every mutable carrier involved in a run
boundary.  They are gated by ``XDART_RUN_CONFIG_DEBUG`` so the normal GUI does
not traverse Qt widgets, parameter trees, or scan state for logging.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable


def run_config_debug_enabled() -> bool:
    value = os.environ.get("XDART_RUN_CONFIG_DEBUG", "")
    return value.lower() not in ("", "0", "false", "no", "off")


def bump_run_config_debug_generation(owner, kind: str) -> int | None:
    """Advance a diagnostic-only run/config generation when tracing is on."""
    if not run_config_debug_enabled() or owner is None:
        return None
    attr = f"_run_config_debug_{str(kind)}_generation"
    try:
        generation = int(getattr(owner, attr, 0) or 0) + 1
        setattr(owner, attr, generation)
        return generation
    except Exception:
        return None


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(val) for val in value]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        try:
            return [_jsonable(val) for val in value]
        except TypeError:
            pass
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _safe_attr(owner, name, default=None):
    try:
        return getattr(owner, name, default)
    except Exception:
        return default


def _parameter_value(parameters, *paths):
    for path in paths:
        try:
            return parameters.child(*path).value()
        except Exception:
            continue
    return None


def _object_identity(value) -> dict:
    if value is None:
        return {"present": False, "object_id": None, "type": None}
    shape = _safe_attr(value, "shape")
    try:
        shape = list(shape) if shape is not None else None
    except TypeError:
        shape = str(shape)
    return {
        "present": True,
        "object_id": hex(id(value)),
        "type": type(value).__name__,
        "shape": shape,
        "dtype": (
            str(_safe_attr(value, "dtype"))
            if _safe_attr(value, "dtype") is not None
            else None
        ),
    }


def _path_identity(value) -> dict:
    text = "" if value is None else str(value)
    return {
        "configured": text,
        "absolute": os.path.abspath(os.path.expanduser(text)) if text else "",
    }


def _scan_key(scan):
    name = _safe_attr(scan, "name")
    if name not in (None, "", "null_main"):
        return name
    data_file = _safe_attr(scan, "data_file")
    return data_file or None


def _scan_summary(scan) -> dict:
    if scan is None:
        return {"present": False}
    a1 = dict(_safe_attr(scan, "bai_1d_args", {}) or {})
    a2 = dict(_safe_attr(scan, "bai_2d_args", {}) or {})
    return {
        "present": True,
        "object_id": hex(id(scan)),
        "scan_key": _scan_key(scan),
        "name": _safe_attr(scan, "name"),
        "data_file": _path_identity(_safe_attr(scan, "data_file")),
        "gi": bool(_safe_attr(scan, "gi", False)),
        "gi_config": dict(_safe_attr(scan, "gi_config", {}) or {}),
        "incidence_motor": _safe_attr(scan, "incidence_motor"),
        "th_mtr": _safe_attr(scan, "th_mtr"),
        "sample_orientation": _safe_attr(scan, "sample_orientation"),
        "tilt_angle": _safe_attr(scan, "tilt_angle"),
        "gi_mode_1d": a1.get("gi_mode_1d"),
        "gi_mode_2d": a2.get("gi_mode_2d"),
        "unit_1d": a1.get("unit"),
        "unit_2d": a2.get("unit"),
        "poni": _object_identity(_safe_attr(scan, "_cached_poni")),
        "global_mask": _object_identity(_safe_attr(scan, "global_mask")),
    }


def _controls_summary(widget) -> dict:
    if widget is None:
        return {"present": False}
    panel = _safe_attr(widget, "controls_v2")
    gi_values = []
    if panel is not None:
        try:
            gi_values = [
                edit.value
                for edit in panel.current_form_edits()
                if tuple(edit.path) == ("GI", "Grazing")
            ]
        except Exception:
            gi_values = []

    state = _safe_attr(widget, "_controls_v2_threshold_state")
    threshold = dict(state) if isinstance(state, dict) else None
    poni_path = ""
    get_poni_path = _safe_attr(widget, "_controls_v2_poni_path")
    if callable(get_poni_path):
        try:
            poni_path = get_poni_path()
        except Exception:
            poni_path = ""
    wrangler = _safe_attr(widget, "wrangler")
    parameters = _safe_attr(wrangler, "parameters")
    return {
        "present": panel is not None,
        "gi_visible_values": gi_values,
        "threshold": threshold,
        "poni_file": _path_identity(poni_path),
        "mask_file": _path_identity(
            _parameter_value(parameters, ("Signal", "mask_file"))
            if parameters is not None
            else ""
        ),
    }


def _carrier_summary(owner) -> dict:
    if owner is None:
        return {"present": False}
    return {
        "present": True,
        "object_id": hex(id(owner)),
        "gi": _safe_attr(owner, "gi"),
        "incidence_motor": _safe_attr(owner, "incidence_motor"),
        "sample_orientation": _safe_attr(owner, "sample_orientation"),
        "tilt_angle": _safe_attr(owner, "tilt_angle"),
        "threshold": {
            "apply": _safe_attr(owner, "apply_threshold"),
            "min": _safe_attr(owner, "threshold_min"),
            "max": _safe_attr(owner, "threshold_max"),
            "mask_saturation": _safe_attr(owner, "mask_sentinel"),
        },
        "poni_file": _path_identity(_safe_attr(owner, "poni_file")),
        "poni": _object_identity(_safe_attr(owner, "poni")),
        "mask_file": _path_identity(_safe_attr(owner, "mask_file")),
        "scan": _scan_summary(_safe_attr(owner, "scan")),
    }


def _wrangler_summary(wrangler) -> dict:
    if wrangler is None:
        return {"present": False}
    parameters = _safe_attr(wrangler, "parameters")
    hidden = {
        "gi": _parameter_value(parameters, ("GI", "Grazing")),
        "incidence_motor": _parameter_value(parameters, ("GI", "th_motor")),
        "sample_orientation": _parameter_value(
            parameters, ("GI", "sample_orientation")
        ),
        "tilt_angle": _parameter_value(parameters, ("GI", "tilt_angle")),
        "threshold": {
            "apply": _parameter_value(parameters, ("Mask", "Threshold")),
            "min": _parameter_value(parameters, ("Mask", "min")),
            "max": _parameter_value(parameters, ("Mask", "max")),
            "mask_saturation": _parameter_value(
                parameters, ("MaskSat", "mask_sentinel")
            ),
        },
        "poni_file": _path_identity(
            _parameter_value(
                parameters,
                ("Signal", "poni_file"),
                ("Calibration", "poni_file"),
            )
        ),
        "mask_file": _path_identity(
            _parameter_value(parameters, ("Signal", "mask_file"))
        ),
    }
    summary = _carrier_summary(wrangler)
    summary["hidden_parameters"] = hidden
    summary["worker"] = _carrier_summary(_safe_attr(wrangler, "thread"))
    return summary


def _generations(widget) -> dict:
    if widget is None:
        return {}
    display = _safe_attr(widget, "displayframe")
    publication = _safe_attr(widget, "publication_store")
    viewer = _safe_attr(widget, "h5viewer")
    return {
        "diagnostic_run": _safe_attr(
            widget, "_run_config_debug_run_generation"
        ),
        "diagnostic_config": _safe_attr(
            widget, "_run_config_debug_config_generation"
        ),
        "runend": _safe_attr(widget, "_runend_generation"),
        "display": _safe_attr(display, "display_generation"),
        "publication": _safe_attr(publication, "generation"),
        "h5_load": _safe_attr(viewer, "_load_generation"),
    }


def parameter_change_summary(changes) -> list[dict]:
    """Return compact pyqtgraph parameter-tree changes for R4-D traces."""
    result = []
    for change in changes or ():
        try:
            param, change_type, value = change[:3]
        except (TypeError, ValueError):
            result.append({"change": str(change)})
            continue
        path = []
        current = param
        for _ in range(12):
            if current is None:
                break
            try:
                path.append(str(current.name()))
            except Exception:
                break
            try:
                current = current.parent()
            except Exception:
                break
        result.append({
            "path": list(reversed(path)),
            "change_type": change_type,
            "value": _jsonable(value),
        })
    return result


def run_config_debug_log(
    logger,
    event: str,
    *,
    widget=None,
    wrangler=None,
    origin: str = "",
    level: str = "info",
    **fields,
) -> None:
    """Log one ownership snapshot, doing no inspection when tracing is off."""
    if not run_config_debug_enabled():
        return
    wrangler = wrangler if wrangler is not None else _safe_attr(widget, "wrangler")
    scan = _safe_attr(widget, "scan")
    payload = {
        "event": event,
        "origin": origin,
        "t": round(time.monotonic(), 6),
        "run_active": bool(_safe_attr(widget, "_run_active", False)),
        "generations": _generations(widget),
        "controls": _controls_summary(widget),
        "shared_scan": _scan_summary(scan),
        "wrangler": _wrangler_summary(wrangler),
    }
    payload.update({str(key): _jsonable(val) for key, val in fields.items()})
    message = "RUN_CONFIG_DEBUG " + json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    )
    log = getattr(logger, level, None)
    if not callable(log):
        log = logger.info
    log(message)
