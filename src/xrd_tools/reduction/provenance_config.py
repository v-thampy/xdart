"""Pure reduction-provenance config assembly.

This module intentionally accepts duck-typed scan/plan objects so the headless
core and the xdart GUI writer can share the same provenance dictionary builder
without importing each other.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def build_reduction_config(
    scan_or_plan: Any, *, include_inputs: bool = True
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(config, inputs)`` for ``write_provenance``.

    ``scan_or_plan`` may be a GUI-style scan with ``bai_1d_args`` /
    ``bai_2d_args``, a headless ``ReductionPlan``, a ``(scan, plan)`` tuple, or
    an object exposing ``scan``/``plan`` attributes.

    ``include_inputs`` gates the (potentially expensive) raw-input enumeration.
    ``_inputs_from_scan`` may walk the ENTIRE frame series to collect source
    paths, and for a GUI ``LiveFrameSeries`` each non-resident frame triggers a
    disk read under ``file_lock`` (``__getitem__``) -- ruinous on the GUI thread
    while a live run holds that same lock.  Callers that only need the
    integration ``config`` (e.g. the display-provenance snapshot, which discards
    ``inputs``) pass ``include_inputs=False`` to skip the walk; the authoritative
    provenance writers (nexus writer, headless core) keep the default.
    """

    scan, plan = _split_scan_plan(scan_or_plan)

    config: dict[str, Any] = {}
    if scan is not None and _has_bai_args(scan):
        # Preserve the GUI writer's exact config shape and insertion order.
        config["bai_1d_args"] = dict(getattr(scan, "bai_1d_args"))
        config["bai_2d_args"] = dict(getattr(scan, "bai_2d_args"))
    elif plan is not None:
        config.update(_config_from_plan(plan))

    if scan is not None and hasattr(scan, "gi"):
        config["gi"] = bool(getattr(scan, "gi"))
    elif plan is not None:
        config["gi"] = getattr(plan, "gi", None) is not None

    if scan is not None and getattr(scan, "gi_config", None):
        config["gi_config"] = dict(getattr(scan, "gi_config"))
    elif plan is not None:
        gi_config = _gi_config_from_plan(plan)
        if gi_config:
            config["gi_config"] = gi_config

    gi_diag = _first_truthy(
        getattr(scan, "gi_freeze_diagnostic", None) if scan is not None else None,
        _plan_extra(plan).get("gi_freeze_diagnostic") if plan is not None else None,
    )
    if gi_diag:
        config["gi_freeze_diagnostic"] = str(gi_diag)

    if scan is not None:
        geom = getattr(scan, "geometry", None)
        if geom is not None:
            config["geometry"] = _geometry_config(geom)

    return config, (_inputs_from_scan(scan) if include_inputs else {})


def _split_scan_plan(value: Any) -> tuple[Any | None, Any | None]:
    if isinstance(value, tuple) and len(value) == 2:
        first, second = value
        if _looks_like_plan(first) and not _looks_like_plan(second):
            return second, first
        return first, second

    scan = getattr(value, "scan", None)
    if scan is None:
        scan = getattr(value, "_scan", None)
    plan = getattr(value, "plan", None)
    if plan is None:
        plan = getattr(value, "_plan", None)

    if _looks_like_plan(value):
        plan = value
    elif scan is None:
        scan = value
    return scan, plan


def _looks_like_plan(value: Any) -> bool:
    return (
        value is not None
        and hasattr(value, "integration_1d")
        and hasattr(value, "integration_2d")
    )


def _has_bai_args(value: Any) -> bool:
    return hasattr(value, "bai_1d_args") and hasattr(value, "bai_2d_args")


def _config_from_plan(plan: Any) -> dict[str, Any]:
    return {
        "bai_1d_args": _integration_1d_args(
            getattr(plan, "integration_1d", None),
            getattr(plan, "gi", None),
        ),
        "bai_2d_args": _integration_2d_args(
            getattr(plan, "integration_2d", None),
            getattr(plan, "gi", None),
        ),
    }


def _integration_1d_args(plan: Any, gi: Any) -> dict[str, Any]:
    if plan is None:
        return {}
    out: dict[str, Any] = {
        "npt": getattr(plan, "npt", None),
        "unit": getattr(plan, "unit", None),
        "method": getattr(plan, "method", None),
        "radial_range": _jsonable_range(getattr(plan, "radial_range", None)),
        "azimuth_range": _jsonable_range(getattr(plan, "azimuth_range", None)),
    }
    if getattr(plan, "npt_rad", 1000) != 1000:
        out["chi_npt_rad"] = getattr(plan, "npt_rad")
    _copy_optional(out, "monitor", getattr(plan, "monitor_key", None))
    _copy_optional(out, "error_model", getattr(plan, "error_model", None))
    _copy_optional(
        out, "polarization_factor", getattr(plan, "polarization_factor", None),
    )
    extra = _mapping(getattr(plan, "extra", None))
    out.update(extra)
    if gi is not None:
        out.setdefault("gi_mode_1d", _enum_value(getattr(gi, "mode_1d", "q_total")))
        out.setdefault("gi_method_1d", getattr(gi, "method", None))
        npt_oop = getattr(gi, "npt_oop", None)
        if npt_oop is not None:
            out.setdefault("npt_oop", int(npt_oop))
    return out


def _integration_2d_args(plan: Any, gi: Any) -> dict[str, Any]:
    if plan is None:
        return {}
    out: dict[str, Any] = {
        "npt_rad": getattr(plan, "npt_rad", None),
        "npt_azim": getattr(plan, "npt_azim", None),
        "unit": getattr(plan, "unit", None),
        "method": getattr(plan, "method", None),
        "radial_range": _jsonable_range(getattr(plan, "radial_range", None)),
        "azimuth_range": _jsonable_range(getattr(plan, "azimuth_range", None)),
    }
    azimuth_offset = float(getattr(plan, "azimuth_offset", 0.0) or 0.0)
    if azimuth_offset:
        out["chi_offset"] = azimuth_offset
    _copy_optional(out, "monitor", getattr(plan, "monitor_key", None))
    _copy_optional(out, "error_model", getattr(plan, "error_model", None))
    _copy_optional(
        out, "polarization_factor", getattr(plan, "polarization_factor", None),
    )
    extra = _mapping(getattr(plan, "extra", None))
    out.update(extra)
    if gi is not None:
        out.setdefault("gi_mode_2d", _enum_value(getattr(gi, "mode_2d", "qip_qoop")))
        out.setdefault("gi_method_2d", getattr(gi, "method", None))
        npt_oop = getattr(gi, "npt_oop", None)
        if npt_oop is not None:
            out.setdefault("npt_oop", int(npt_oop))
    return out


def _gi_config_from_plan(plan: Any) -> dict[str, Any]:
    gi = getattr(plan, "gi", None)
    if gi is None:
        return {}
    return {
        "gi_mode_1d": _enum_value(getattr(gi, "mode_1d", "q_total")),
        "gi_mode_2d": _enum_value(getattr(gi, "mode_2d", "qip_qoop")),
        "incidence_motor": str(getattr(gi, "incidence_motor", "") or ""),
        "th_val": float(getattr(gi, "incident_angle", 0.0) or 0.0),
        "tilt_angle": float(getattr(gi, "tilt_angle", 0.0) or 0.0),
        "sample_orientation": int(getattr(gi, "sample_orientation", 1) or 1),
    }


def _geometry_config(geom: Any) -> dict[str, Any]:
    return {
        "convention": getattr(geom, "convention", getattr(geom, "preset", "")),
        "mapping_json": geom.to_json(),
        "motor_sources": {
            m: m for m in geom.all_referenced_motors()
        },
    }


def _inputs_from_scan(scan: Any | None) -> dict[str, Any]:
    if scan is None:
        return {}

    inputs: dict[str, Any] = {}
    if hasattr(scan, "raw_files") and getattr(scan, "raw_files"):
        inputs["raw_files"] = list(getattr(scan, "raw_files"))
    if hasattr(scan, "meta_file") and getattr(scan, "meta_file"):
        inputs["meta_file"] = str(getattr(scan, "meta_file"))

    if "raw_files" not in inputs:
        raw_files = _raw_files_from_scan(scan)
        if raw_files:
            inputs["raw_files"] = raw_files
    if "meta_file" not in inputs:
        meta_file = _meta_file_from_scan(scan)
        if meta_file:
            inputs["meta_file"] = meta_file
    return inputs


def _raw_files_from_scan(scan: Any) -> list[str]:
    metadata = getattr(scan, "metadata", None)
    image_paths = getattr(metadata, "image_paths", None)
    if image_paths:
        return [str(p) for p in image_paths]

    h5_path = getattr(metadata, "h5_path", None)
    if h5_path:
        return [str(h5_path)]

    frames = getattr(scan, "frames", None) or ()
    out: list[str] = []
    seen: set[str] = set()
    for frame in frames:
        source_path = getattr(frame, "source_path", None)
        if source_path is None:
            continue
        text = str(source_path)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _meta_file_from_scan(scan: Any) -> str | None:
    metadata = getattr(scan, "metadata", None)
    extra = _mapping(getattr(metadata, "extra", None))
    for key in ("meta_file", "metadata_file", "spec_path"):
        value = extra.get(key)
        if value:
            return str(value)
    return None


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _plan_extra(plan: Any) -> dict[str, Any]:
    return _mapping(getattr(plan, "extra", None))


def _copy_optional(out: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        out[key] = value


def _jsonable_range(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(float(v) for v in value)
    if isinstance(value, list):
        return [float(v) for v in value]
    return value


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _first_truthy(*values: Any) -> Any:
    for value in values:
        if value:
            return value
    return None


__all__ = ["build_reduction_config"]
