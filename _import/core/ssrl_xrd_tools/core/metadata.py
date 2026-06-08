# ssrl_xrd_tools/core/metadata.py
"""
Source-agnostic scan metadata.

Readers (SPEC, NeXus, Tiled) build ``ScanMetadata`` instances; processing code
consumes them without knowing the data source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from pathlib import Path
from typing import Any, Mapping

import numpy as np


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
