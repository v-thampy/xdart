# xrd_tools/core/metadata.py
"""
Source-agnostic scan metadata.

Readers (SPEC, NeXus, Tiled) build ``ScanMetadata`` instances; processing code
consumes them without knowing the data source.

Two metadata models live here, at DIFFERENT granularities — they are
intentionally distinct, NOT redundant (so there is nothing to "collapse"):

* :class:`ScanMetadata` — the **scan-level** ingestion record: one per scan,
  carrying energy/wavelength, per-scan-point motor ``angles`` + ``counters``
  arrays, the UB matrix, sample/scan-type/source provenance, and image paths.
  Built once by the format readers.
* :class:`HeterogeneousMetadata` — a **single frame's** ``raw`` + ``numeric``
  metadata bag (immutable), carried on each :class:`FrameView`/reduced frame.

A scan-level record relates to a per-frame bag by slicing the scan-point arrays
at a given index; the natural bridge is one-to-many (one ``ScanMetadata`` ->
N ``HeterogeneousMetadata``), not an alias.  (Resolution of restructure item
#8b: the two were flagged as candidates to merge, but they model different
things — kept separate by design.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from xrd_tools.core.strictness import StrictnessError


INCIDENCE_MOTOR_SEARCH_ORDER: tuple[str, ...] = (
    "th",
    "theta",
    "eta",
    "halpha",
    "gth",
    "gonth",
)

_MISSING = object()


class IncidenceAngleUnresolved(StrictnessError):
    """A GI frame's incidence angle could not be resolved from metadata."""


def _metadata_get_case_insensitive(metadata: Mapping[str, Any], key: Any) -> Any:
    key_lower = str(key).lower()
    for existing, value in metadata.items():
        if str(existing).lower() == key_lower:
            return value
    return _MISSING


def _float_or_unresolved(value: Any, *, motor: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise IncidenceAngleUnresolved(
            f"GI incidence motor {motor!r} was found but value {value!r} "
            "could not be converted to an incidence angle."
        ) from exc


def resolve_incident_angle(
    metadata: Mapping[str, Any],
    incidence_motor: str | None,
) -> float:
    """Resolve a GI incidence angle in degrees from a frame metadata bag.

    ``incidence_motor`` may be a manual numeric angle or a motor name.  When no
    explicit motor is provided, common GI motor names are tried in
    :data:`INCIDENCE_MOTOR_SEARCH_ORDER`.
    """

    metadata = metadata or {}
    if incidence_motor is not None and str(incidence_motor).strip():
        try:
            return float(incidence_motor)
        except (TypeError, ValueError):
            motor = str(incidence_motor)
            value = _metadata_get_case_insensitive(metadata, motor)
            if value is not _MISSING:
                return _float_or_unresolved(value, motor=motor)
            raise IncidenceAngleUnresolved(
                "GI incidence motor {!r} is not a number and was not found "
                "in the frame metadata; refusing to integrate at a degenerate "
                "0°. Set 'Theta Motor' to Manual and enter the incidence "
                "angle.".format(incidence_motor)
            )

    first_error: IncidenceAngleUnresolved | None = None
    for motor in INCIDENCE_MOTOR_SEARCH_ORDER:
        value = _metadata_get_case_insensitive(metadata, motor)
        if value is _MISSING:
            continue
        try:
            return _float_or_unresolved(value, motor=motor)
        except IncidenceAngleUnresolved as exc:
            if first_error is None:
                first_error = exc

    if first_error is not None:
        raise first_error
    raise IncidenceAngleUnresolved(
        "GI incidence angle could not be resolved from a manual value or "
        "metadata motors: {}.".format(", ".join(INCIDENCE_MOTOR_SEARCH_ORDER))
    )


def resolve_monitor_norm(metadata: Mapping[str, Any], key: str | None) -> float | None:
    """Resolve a guarded monitor normalization value from frame metadata.

    Returns ``None`` when no monitor is configured, the key is absent, or the
    value is nonnumeric, non-finite, zero, or negative.
    """

    if key is None or not str(key).strip():
        return None
    value = _metadata_get_case_insensitive(metadata or {}, key)
    if value is _MISSING:
        return None
    try:
        norm = float(value)
    except (TypeError, ValueError):
        return None
    return norm if np.isfinite(norm) and norm > 0.0 else None


def numeric_metadata(metadata: Mapping[str, Any] | None) -> dict[str, float]:
    """Return finite scalar numeric values from heterogeneous metadata."""

    out: dict[str, float] = {}
    for key, value in (metadata or {}).items():
        try:
            arr = np.asarray(value)
            if arr.shape != ():
                continue
            numeric = float(arr)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(numeric):
            continue
        out[str(key)] = numeric
    return out


@dataclass(frozen=True, slots=True)
class HeterogeneousMetadata:
    """Per-frame metadata preserving raw values plus a numeric view."""

    raw: Mapping[str, Any] = field(default_factory=dict)
    numeric: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        raw = MappingProxyType(dict(self.raw or {}))
        numeric = dict(self.numeric or numeric_metadata(raw))
        object.__setattr__(self, "raw", raw)
        object.__setattr__(self, "numeric", MappingProxyType(numeric))


@dataclass(slots=True)
class ScanMetadata:
    """
    Scan-level metadata and per-point arrays, independent of file format.

    Parameters
    ----------
    scan_id : str
        Unique identifier for the scan (e.g. ``"sample_scan12"``).
    energy : float
        Photon energy in keV.
    wavelength : float
        Wavelength in Angstroms.
    angles : dict of str to ndarray
        Motor name → values at each scan point (same length along scan axis).
    counters : dict of str to ndarray
        Counter name → values at each scan point.
    ub_matrix : ndarray or None, optional
        3×3 UB matrix in Å⁻¹, if available.
    sample_name : str, optional
    scan_type : str, optional
        Scan command or type label.
    source : str, optional
        Provenance label, e.g. ``"spec"``, ``"tiled"``, ``"hdf5"``.
    image_paths : list of Path, optional
        Paths to per-point images when not using a single HDF5 stack.
    h5_path : Path or None, optional
        Single HDF5 master or stack path when applicable.
    extra : dict, optional
        Additional source-specific fields without breaking the common API.
    """

    scan_id: str
    energy: float
    wavelength: float
    angles: dict[str, np.ndarray]
    counters: dict[str, np.ndarray]
    ub_matrix: np.ndarray | None = None
    sample_name: str = ""
    scan_type: str = ""
    source: str = ""
    image_paths: list[Path] = field(default_factory=list)
    h5_path: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ub_matrix is not None:
            self.ub_matrix = np.asarray(self.ub_matrix, dtype=float)
        self.angles = {
            k: np.asarray(v, dtype=float) for k, v in self.angles.items()
        }
        self.counters = {
            k: np.asarray(v, dtype=float) for k, v in self.counters.items()
        }
        self.image_paths = [
            p if isinstance(p, Path) else Path(p) for p in self.image_paths
        ]
        if self.h5_path is not None and not isinstance(self.h5_path, Path):
            self.h5_path = Path(self.h5_path)
