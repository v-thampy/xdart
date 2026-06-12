"""Canonical scan/frame contracts for headless XRD processing.

The types in this module are intentionally GUI-free.  They are the shared
input-side contracts for reduction, RSM, stitching, notebooks, and thin GUIs.
Older modules may re-export these classes while their internal callers migrate.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np

# NOTE: keep this module import-light (no h5py/fabio/pyFAI/Qt): it is part
# of the Qt-free core-contracts surface that pure display-logic / CI
# environments import.  Heavy readers (e.g. xrd_tools.io.image.read_image)
# are imported inside the methods that need them.
from xrd_tools.core.containers import PONI
from xrd_tools.core.metadata import (
    HeterogeneousMetadata,
    ScanMetadata,
    numeric_metadata,
)
ImageLoader = Callable[["ScanFrame"], np.ndarray]


def _metadata_get_case_insensitive(metadata: Mapping[str, Any], key: str) -> Any:
    key_lower = key.lower()
    for existing, value in metadata.items():
        if str(existing).lower() == key_lower:
            return value
    return None


@dataclass(frozen=True, slots=True)
class SourceCapabilities:
    """Capabilities advertised by a :class:`FrameSource` implementation."""

    is_streaming: bool = False
    supports_random_access: bool = True
    supports_chunks: bool = True
    supports_prefetch: bool = False
    has_metadata: bool = False
    has_geometry: bool = False
    has_raw_references: bool = False
    has_thumbnails: bool = False


class SourceKind(str, Enum):
    """Known source families."""

    MEMORY = "memory"
    IMAGE_FILE = "image_file"
    TIFF_SERIES = "tiff_series"
    NEXUS_STACK = "nexus_stack"
    EIGER_MASTER = "eiger_master"
    PROCESSED_NEXUS = "processed_nexus"
    SPEC = "spec"
    TILED = "tiled"
    LIVE = "live"
    UNKNOWN = "unknown"


def coerce_source_kind(value: SourceKind | str) -> SourceKind:
    if isinstance(value, SourceKind):
        return value
    return SourceKind(str(value))


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """Serializable description of an image/frame source."""

    uri: str | Path
    kind: SourceKind | str = SourceKind.UNKNOWN
    metadata_uri: str | Path | None = None
    entry: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", coerce_source_kind(self.kind))
        object.__setattr__(self, "options", MappingProxyType(dict(self.options or {})))


@dataclass(frozen=True, slots=True)
class FrameGeometry:
    """Minimal per-frame geometry needed by GI, RSM, and stitching."""

    rot1: float | None = None
    rot2: float | None = None
    rot3: float | None = None
    incident_angle: float | None = None
    poni: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.poni is not None:
            object.__setattr__(self, "poni", MappingProxyType(dict(self.poni)))


@dataclass(frozen=True, slots=True)
class MaskSpec:
    """Detector mask that can be resolved once a frame shape is known."""

    values: Any

    def to_bool(self, image_shape: tuple[int, int]) -> np.ndarray:
        arr = np.asarray(self.values)
        if arr.ndim == 2:
            if arr.shape != image_shape:
                raise ValueError(
                    f"mask shape {arr.shape} does not match image shape {image_shape}"
                )
            return arr.astype(bool, copy=False)
        if arr.ndim != 1:
            raise ValueError(f"flat mask must be 1D; got shape {arr.shape}")

        n_pixels = int(np.prod(image_shape))
        if arr.dtype == bool:
            if arr.size != n_pixels:
                raise ValueError(
                    f"flat boolean mask length {arr.size} does not match "
                    f"image shape {image_shape}"
                )
            return arr.reshape(image_shape)

        flat = np.asarray(arr, dtype=int).ravel()
        if np.any(flat < 0) or np.any(flat >= n_pixels):
            raise ValueError(f"flat mask indices out of bounds for image shape {image_shape}")
        out = np.zeros(n_pixels, dtype=bool)
        out[flat] = True
        return out.reshape(image_shape)


@dataclass(slots=True)
class ScanFrame:
    """One detector frame plus enough provenance to load it lazily."""

    index: int
    image: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: Path | str | None = None
    source_frame_index: int | None = None
    background: np.ndarray | float | None = None
    mask: np.ndarray | MaskSpec | None = None
    normalization_factor: float | None = None
    loader: ImageLoader | None = None
    geometry: FrameGeometry | None = None
    source_identity: str | None = None

    def __post_init__(self) -> None:
        self.metadata = dict(self.metadata or {})
        if self.source_path is not None and not isinstance(self.source_path, Path):
            self.source_path = Path(self.source_path)

    @property
    def metadata_view(self) -> HeterogeneousMetadata:
        return HeterogeneousMetadata(self.metadata)

    @property
    def metadata_raw(self) -> Mapping[str, Any]:
        return self.metadata_view.raw

    @property
    def metadata_numeric(self) -> Mapping[str, float]:
        return self.metadata_view.numeric

    def load_image(self) -> np.ndarray:
        """Return this frame's image, loading from provenance if needed."""

        if self.image is not None:
            return np.asarray(self.image)
        if self.loader is not None:
            self.image = np.asarray(self.loader(self))
            return self.image
        if self.source_path is None:
            raise ValueError(
                f"Frame {self.index} has no image, loader, or source_path."
            )

        path = Path(self.source_path)
        ext = path.suffix.lower()
        if ext in {".h5", ".hdf5", ".nxs"} and self.source_frame_index is not None:
            # Keep the canonical contracts import-light: xrd_tools.io.nexus
            # imports core containers, so importing this at module load time
            # creates a circular import through xrd_tools.core.__init__.
            from xrd_tools.io.nexus import open_nexus_image_stack

            with open_nexus_image_stack(path) as stack:
                self.image = np.asarray(stack[int(self.source_frame_index)])
        else:
            from xrd_tools.io.image import read_image

            self.image = np.asarray(read_image(path))
        return self.image

    @property
    def label(self) -> str:
        return str(self.index)


@runtime_checkable
class FrameSource(Protocol):
    """Frame stream consumed by reduction, RSM, stitching, and notebooks."""

    @property
    def frame_indices(self) -> list[int]:
        ...

    @property
    def capabilities(self) -> SourceCapabilities:
        ...

    def load_frame(self, index: int) -> np.ndarray:
        ...

    def iter_chunks(self, chunk_size: int) -> Iterator[tuple[np.ndarray, list[int]]]:
        ...


@dataclass(slots=True)
class Scan:
    """Ordered set of frames with scan-level reduction context."""

    name: str
    frames: list[ScanFrame]
    poni: PONI | None = None
    integrator: Any = None
    metadata: ScanMetadata | None = None
    energy: float | None = None
    wavelength: float | None = None
    # Diffractometer angle mapping (``DiffractometerGeometry``): when set,
    # NexusSink derives /entry/per_frame_geometry from scan_data at finish.
    geometry: Any | None = None
    motors: dict[str, np.ndarray] = field(default_factory=dict)
    output_path: Path | str | None = None
    sample_name: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    capabilities: SourceCapabilities = field(
        default_factory=lambda: SourceCapabilities(
            is_streaming=False,
            supports_random_access=True,
            supports_chunks=True,
            has_metadata=True,
            has_geometry=True,
            has_raw_references=True,
        )
    )
    _frame_by_index: dict[int, ScanFrame] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.frames = sorted(list(self.frames), key=lambda f: f.index)
        indices = [int(f.index) for f in self.frames]
        if len(indices) != len(set(indices)):
            seen: set[int] = set()
            dupes: set[int] = set()
            for idx in indices:
                if idx in seen:
                    dupes.add(idx)
                seen.add(idx)
            raise ValueError(f"Scan contains duplicate frame indices: {sorted(dupes)}")
        self.motors = {k: np.asarray(v, dtype=float) for k, v in self.motors.items()}
        if self.output_path is not None and not isinstance(self.output_path, Path):
            self.output_path = Path(self.output_path)
        self._frame_by_index = {int(frame.index): frame for frame in self.frames}

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterable[ScanFrame]:
        return iter(self.frames)

    @property
    def frame_indices(self) -> list[int]:
        return [int(frame.index) for frame in self.frames]

    @property
    def energy_keV(self) -> float | None:
        return self.energy

    @property
    def energy_eV(self) -> float | None:
        return None if self.energy is None else float(self.energy) * 1000.0

    def load_frame(self, index: int) -> np.ndarray:
        try:
            frame = self._frame_by_index[int(index)]
        except KeyError as exc:
            raise KeyError(f"Scan has no frame {index}") from exc
        return np.asarray(frame.load_image())

    def iter_chunks(self, chunk_size: int) -> Iterator[tuple[np.ndarray, list[int]]]:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0; got {chunk_size}")
        indices = self.frame_indices
        for start in range(0, len(indices), chunk_size):
            chunk_indices = indices[start:start + chunk_size]
            loaded_here: list[ScanFrame] = []
            images: list[np.ndarray] = []
            try:
                for idx in chunk_indices:
                    frame = self._frame_by_index[int(idx)]
                    was_empty = frame.image is None
                    images.append(np.asarray(frame.load_image()))
                    if was_empty and frame.image is not None:
                        loaded_here.append(frame)
                yield np.stack(images), chunk_indices
            finally:
                for frame in loaded_here:
                    frame.image = None

    def to_metadata(self) -> ScanMetadata | None:
        if self.metadata is not None:
            return self.metadata

        wavelength_A: float | None = None
        if self.wavelength is not None:
            wavelength_A = float(self.wavelength)
        elif self.poni is not None and self.poni.wavelength:
            wavelength_A = float(self.poni.wavelength) * 1e10

        energy_keV = self.energy
        if energy_keV is None and wavelength_A and wavelength_A > 0:
            energy_keV = 12.398 / wavelength_A
        if energy_keV is None or wavelength_A is None:
            return None

        counters: dict[str, np.ndarray] = {}
        for key in ("i0", "i1", "monitor", "mon", "seconds"):
            vals = [
                _metadata_get_case_insensitive(f.metadata, key) for f in self.frames
            ]
            vals = [v for v in vals if v is not None]
            if len(vals) == len(self.frames):
                counters[key] = np.asarray(vals, dtype=float)

        return ScanMetadata(
            scan_id=self.name,
            energy=float(energy_keV),
            wavelength=float(wavelength_A),
            angles=self.motors,
            counters=counters,
            sample_name=self.sample_name,
            source="core.Scan",
            image_paths=[
                Path(f.source_path) for f in self.frames
                if f.source_path is not None
            ],
            h5_path=None,
            extra=self.extra.copy(),
        )

    def to_scan_data(self):
        """Per-frame condition table as a pandas DataFrame."""

        import pandas as pd

        idx = self.frame_indices
        keys: list[Any] = []
        seen: set = set()
        for frame in self.frames:
            for key in (frame.metadata or {}):
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        data = {
            str(key): [(frame.metadata or {}).get(key) for frame in self.frames]
            for key in keys
        }
        df = pd.DataFrame(data, index=idx) if data else pd.DataFrame(index=idx)
        for name, arr in self.motors.items():
            values = np.asarray(arr)
            if values.ndim == 1 and values.shape[0] == len(idx):
                df[str(name)] = values
        return df

    @property
    def scan_data(self):
        """Pandas-compatible per-frame metadata table for RSM-style consumers."""
        return self.to_scan_data()


Frame = ScanFrame


__all__ = [
    "Frame",
    "FrameGeometry",
    "FrameSource",
    "HeterogeneousMetadata",
    "ImageLoader",
    "MaskSpec",
    "Scan",
    "ScanFrame",
    "SourceCapabilities",
    "SourceKind",
    "SourceSpec",
    "coerce_source_kind",
    "numeric_metadata",
]
