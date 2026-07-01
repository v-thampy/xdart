"""Headless source/read-result readiness composition.

``xrd_tools.core.scan.SourceCapabilities`` is what a frame source advertises
about itself.  ``xrd_tools.session.readiness.SourceCaps`` / ``ResultCaps`` are
the readiness-layer projections consumed by run gates and analysis launchers.
This module is the pure bridge between those two shapes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xrd_tools.core.scan import (
    SourceCapabilities as CoreSourceCapabilities,
    SourceKind,
    SourceSpec,
)
from xrd_tools.session.readiness import ResultCaps, SourceCaps

__all__ = ["describe_source_readiness", "capabilities_for_processed"]

_ENERGY_KEYS = frozenset({
    "energy",
    "energy_ev",
    "energy_eV",
    "energy_kev",
    "energy_keV",
    "wavelength",
    "wavelength_a",
    "wavelength_A",
})
_PSI_KEYS = frozenset({"psi", "sin2psi", "sin^2psi", "chi", "eta"})
_PHASE_CAPS = frozenset({"phase_result", "phase_fit", "phase_fractions"})


def describe_source_readiness(spec_or_source: Any, *, probe: bool = True) -> SourceCaps:
    """Project a source, URI, or :class:`SourceSpec` into readiness caps.

    True-live / unknown-length sources keep ``raw_reachable=True`` even when a
    frame-0 probe cannot yet load an image; a live acquisition may legitimately
    have no frame zero at gate time.
    """

    source = _open_source(spec_or_source)
    if source is None:
        return _caps_from_classification(spec_or_source)

    source_caps = _core_caps(source)
    frame_indices, frame_count = _frame_indices(source)
    unknown_length = frame_count is None
    live_unknown = bool(
        source_caps.is_streaming
        or getattr(source, "kind", None) == SourceKind.LIVE
        or unknown_length
    )
    has_frames = bool(live_unknown or (frame_count is not None and frame_count > 0))

    info = _classify(spec_or_source)
    has_raw = bool(
        live_unknown
        or getattr(source_caps, "has_raw_references", False)
        or getattr(info, "has_raw", False)
        or (has_frames and hasattr(source, "load_frame"))
    )

    if probe and has_raw:
        reachable = _probe_first_frame(source)
        raw_reachable = bool(reachable or live_unknown)
    else:
        raw_reachable = bool(has_raw or live_unknown)

    first_metadata = _first_metadata(source, frame_indices)
    motors = _motors(source)
    return SourceCaps(
        has_frames=has_frames,
        has_raw=has_raw,
        raw_reachable=raw_reachable,
        has_metadata=bool(
            source_caps.has_metadata
            or source_caps.has_scan_manifest
            or first_metadata
            or motors
        ),
        has_motors=bool(motors),
        has_energy=bool(_has_energy(first_metadata) or _attr_known(source, ("energy", "wavelength"))),
        has_geometry=bool(
            source_caps.has_geometry
            or _attr_known(source, ("geometry", "poni", "integrator"))
        ),
        has_psi_metadata=_has_any_key(first_metadata, _PSI_KEYS),
    )


def capabilities_for_processed(metadata: Mapping[str, Any]) -> ResultCaps:
    """Project already-materialized processed metadata into result caps.

    This consumes ``metadata["capabilities"]`` as written by
    :func:`xrd_tools.io.read.get_metadata`; it intentionally does not reopen an
    HDF5 file or call ``io.schema.detect_capabilities``.
    """

    caps = {str(cap) for cap in metadata.get("capabilities", ()) or ()}
    has_1d = bool(
        metadata.get("has_1d")
        or caps & {"axis_kind_1d", "sigma_1d", "multi_result_1d"}
    )
    has_2d = bool(
        metadata.get("has_2d")
        or caps & {"two_d_kind", "sigma_2d", "multi_result_2d"}
    )
    has_raw = bool(caps & {"frames_record", "source_base"})
    has_scan_metadata = _metadata_has_scan_table(metadata) or bool(
        metadata.get("positioners")
        or metadata.get("n_frames")
        or _array_len(metadata.get("frames"))
    )
    return ResultCaps(
        has_1d=has_1d,
        has_2d=has_2d,
        has_raw=has_raw,
        raw_reachable=has_raw,
        has_scan_metadata=has_scan_metadata,
        has_rsm="rsm" in caps,
        has_phase_result=bool(caps & _PHASE_CAPS),
        has_psi_metadata=has_scan_metadata,
    )


def _open_source(value: Any) -> Any | None:
    if hasattr(value, "frame_indices") and hasattr(value, "load_frame"):
        return value
    try:
        from xrd_tools.sources.registry import open_source

        return open_source(value)
    except Exception:
        return None


def _caps_from_classification(value: Any) -> SourceCaps:
    info = _classify(value)
    frame_count = getattr(info, "n_frames", 0) if info is not None else _count_frames(value)
    has_frames = bool(frame_count and frame_count > 0)
    has_raw = bool(getattr(info, "has_raw", False))
    return SourceCaps(
        has_frames=has_frames,
        has_raw=has_raw,
        raw_reachable=has_raw and has_frames,
        has_metadata=False,
    )


def _core_caps(source: Any) -> CoreSourceCapabilities:
    caps = getattr(source, "capabilities", None)
    if isinstance(caps, CoreSourceCapabilities):
        return caps
    return CoreSourceCapabilities()


def _frame_indices(source: Any) -> tuple[list[int], int | None]:
    try:
        indices = [int(idx) for idx in source.frame_indices]
    except Exception:
        return [], None
    return indices, len(indices)


def _probe_first_frame(source: Any) -> bool:
    try:
        from xrd_tools.sources.probe import probe_first_frame

        reachable, _image = probe_first_frame(source)
        return bool(reachable)
    except Exception:
        return False


def _classify(value: Any) -> Any | None:
    uri = _uri(value)
    if uri is None:
        return None
    try:
        from xrd_tools.io.image_source import classify_image_source

        return classify_image_source(uri)
    except Exception:
        return None


def _count_frames(value: Any) -> int:
    uri = _uri(value)
    if uri is None:
        return 0
    try:
        from xrd_tools.io.image import count_frames

        return int(count_frames(uri))
    except Exception:
        return 0


def _uri(value: Any) -> str | Path | None:
    if isinstance(value, SourceSpec):
        return value.uri
    if isinstance(value, (str, Path)):
        return value
    spec = getattr(value, "spec", None)
    if isinstance(spec, SourceSpec):
        return spec.uri
    return None


def _first_metadata(source: Any, frame_indices: list[int]) -> Mapping[str, Any]:
    if not frame_indices or not hasattr(source, "metadata_for"):
        return {}
    try:
        metadata = source.metadata_for(frame_indices[0])
    except Exception:
        return {}
    return dict(metadata or {})


def _motors(source: Any) -> Mapping[str, Any]:
    try:
        motors = getattr(source, "motors", None)
    except Exception:
        return {}
    return dict(motors or {})


def _attr_known(source: Any, names: tuple[str, ...]) -> bool:
    for name in names:
        try:
            value = getattr(source, name, None)
        except Exception:
            continue
        if value is not None:
            return True
    return False


def _has_energy(metadata: Mapping[str, Any]) -> bool:
    for key, value in metadata.items():
        if str(key) not in _ENERGY_KEYS:
            continue
        if _finite_value(value) or value is not None:
            return True
    return False


def _has_any_key(metadata: Mapping[str, Any], keys: frozenset[str]) -> bool:
    normalized = {str(key).lower() for key in metadata}
    return bool(normalized & {key.lower() for key in keys})


def _finite_value(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _metadata_has_scan_table(metadata: Mapping[str, Any]) -> bool:
    scan_data = metadata.get("scan_data")
    if scan_data is None:
        return False
    empty = getattr(scan_data, "empty", None)
    if empty is not None:
        return not bool(empty)
    return True


def _array_len(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(len(value))
    except Exception:
        return 0
