"""Pure reduction-provenance config assembly.

This module intentionally accepts duck-typed scan/plan objects so the headless
core and the xdart GUI writer can share the same provenance dictionary builder
without importing each other.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
import math
from pathlib import Path
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
    paths, and for a lazy GUI frame collection each non-resident frame can
    trigger a disk read under its file lock -- ruinous on the GUI thread
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
        gi_config = dict(getattr(scan, "gi_config"))
        # gi_mode lives authoritatively in bai_*_args (the GUI Axis field edits
        # them directly); the scan-carried copy can lag an edit.  Reconcile so
        # one written config can never contradict itself — a no-op whenever the
        # two already agree.
        for mode_key, args_key in (("gi_mode_1d", "bai_1d_args"),
                                   ("gi_mode_2d", "bai_2d_args")):
            mode = (config.get(args_key) or {}).get(mode_key)
            if mode is not None and mode_key in gi_config:
                gi_config[mode_key] = mode
        config["gi_config"] = gi_config
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

    # O-1a-W1A: the accepted run configuration, as the detached JSON-native
    # projection the run owner attached to this scan BEFORE the output opened.
    # Additive and value-only -- this function never reaches back to the GUI or
    # re-derives the projection, and a scan without one writes nothing new.
    run_configuration = getattr(scan, "run_configuration_provenance", None) \
        if scan is not None else None
    if isinstance(run_configuration, Mapping) and run_configuration:
        config["run_configuration"] = dict(run_configuration)

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
    # Do NOT hydrate a lazy frame series just to read
    # source paths.  Iterating it triggers a per-frame disk load under
    # its file lock, and a legacy GUI frame carries no
    # ``source_path`` -- so this walk hydrates every frame and returns [] today
    # (S-19).  On the WRITER thread that lengthens the flush's lock hold, which is
    # exactly the contention that amplified BB-1.  A lazy series exposes an
    # ``_in_memory`` map of already-resident frames: iterate THOSE (no disk I/O).
    # A plain list (headless ``Scan``, whose ``Frame``s DO carry ``source_path``)
    # has no ``_in_memory`` and iterates unchanged.  Written provenance is byte-
    # identical either way -- GUI stays [], headless keeps its real raw_files.
    resident = getattr(frames, "_in_memory", None)
    if resident is not None:
        # Snapshot the resident map under its cache lock: the GUI thread stashes
        # into ``_in_memory`` during a streaming flush, and iterating a live dict
        # raises "dictionary changed size during iteration" on the writer thread.
        lock = getattr(frames, "_cache_lock", None)
        if lock is not None:
            with lock:
                frame_iter = list(resident.values())
        else:
            frame_iter = list(resident.values())
    else:
        frame_iter = frames
    out: list[str] = []
    seen: set[str] = set()
    for frame in frame_iter:
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
    """One JSON-native range projection owned by reduction provenance.

    O-1a-W1R (review §39.2 W1R-P1-7): this pair used to be the only bounded
    recursive normalization in the tree, so reduction provenance generalized
    it rather than adding a second writer schema authority.  The session module
    re-exports :func:`jsonable_run_value` as a compatibility name; it does not
    own another implementation.  The numeric coercion below is kept because a
    reduction range is numeric by contract.
    """
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return _jsonable(tuple(float(v) for v in value))
    return _jsonable(value)


def _enum_value(value: Any) -> Any:
    return _jsonable(getattr(value, "value", value))


def _jsonable(value: Any) -> Any:
    return jsonable_run_value(value, path="reduction_config")


def jsonable_run_value(value: Any, *, path: str = "provenance") -> Any:
    """Return the one detached, deterministic, JSON-native value projection."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(
                f"{path}: non-finite float {number!r} has no JSON form"
            )
        return number
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return jsonable_run_value(value.value, path=f"{path}.value")
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"{path}: JSON object keys must be str, got "
                    f"{type(key).__module__}.{type(key).__qualname__}"
                )
            out[key] = jsonable_run_value(item, path=f"{path}.{key}")
        return out
    if isinstance(value, (list, tuple)):
        return [
            jsonable_run_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{path}: run provenance values must be JSON-native; unsupported "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _first_truthy(*values: Any) -> Any:
    for value in values:
        if value:
            return value
    return None


__all__ = ["build_reduction_config", "jsonable_run_value"]
