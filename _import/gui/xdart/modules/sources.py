"""xdart live-object adapters for architecture-v2 headless sources."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from ssrl_xrd_tools.core import FrameGeometry, SourceCapabilities, SourceKind
from ssrl_xrd_tools.core.scan import Scan, ScanFrame


class LiveScanFrameSource:
    """FrameSource adapter over an xdart ``LiveScan``.

    The adapter is intentionally Qt-free.  It keeps xdart's mutable
    ``LiveScan``/``LiveFrame`` objects at the edge while exposing the
    canonical ssrl ``FrameSource`` contract to reduction, RSM, stitching,
    and future notebook tools.
    """

    kind = SourceKind.LIVE

    def __init__(self, live_scan: Any, *, frame_indices: Iterable[int] | None = None) -> None:
        self.live_scan = live_scan
        self.name = str(getattr(live_scan, "name", "scan"))
        self._frame_indices = (
            [int(i) for i in frame_indices]
            if frame_indices is not None
            else [int(i) for i in getattr(live_scan.frames, "index", [])]
        )
        self.capabilities = SourceCapabilities(
            is_streaming=bool(getattr(live_scan, "live", False)),
            supports_random_access=True,
            supports_chunks=True,
            has_metadata=True,
            has_geometry=bool(getattr(live_scan, "geometry", None)),
        )
        self.poni = _first_poni(live_scan, self._frame_indices)
        self.wavelength = _wavelength_angstrom(live_scan)
        self.output_path = getattr(live_scan, "data_file", None)
        self.motors = _numeric_motors(getattr(live_scan, "scan_data", None))

    @property
    def frame_indices(self) -> list[int]:
        return list(self._frame_indices)

    @property
    def scan_data(self):
        return getattr(self.live_scan, "scan_data", None)

    @property
    def energy_eV(self) -> float | None:
        if self.wavelength is None or self.wavelength <= 0:
            return None
        return 12398.0 / float(self.wavelength)

    def metadata_for(self, index: int) -> dict[str, Any]:
        frame = self.live_scan.frames[int(index)]
        metadata = dict(getattr(frame, "scan_info", {}) or {})
        metadata.update(_scan_data_row(getattr(self.live_scan, "scan_data", None), int(index)))
        return metadata

    def load_frame(self, index: int) -> np.ndarray:
        frame = self.live_scan.frames[int(index)]
        if getattr(frame, "map_raw", None) is None:
            lazy = getattr(frame, "_lazy_load_raw", None)
            if callable(lazy):
                lazy()
        image = getattr(frame, "map_raw", None)
        if image is None:
            raise ValueError(f"LiveScan frame {index} has no loadable raw image")
        return np.asarray(image)

    def frame_for(self, index: int) -> ScanFrame:
        live_frame = self.live_scan.frames[int(index)]
        return ScanFrame(
            index=int(index),
            image=getattr(live_frame, "map_raw", None),
            metadata=self.metadata_for(int(index)),
            source_path=_resolved_source_path(live_frame),
            source_frame_index=int(getattr(live_frame, "source_frame_idx", 0) or 0),
            background=getattr(live_frame, "bg_raw", None),
            mask=getattr(live_frame, "mask", None),
            loader=lambda _frame, idx=int(index): self.load_frame(idx),
            geometry=_frame_geometry(live_frame),
            source_identity=str(getattr(live_frame, "source_file", "") or index),
        )

    def iter_chunks(self, chunk_size: int):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0; got {chunk_size}")
        labels = self.frame_indices
        for start in range(0, len(labels), chunk_size):
            chunk = labels[start:start + chunk_size]
            yield np.stack([self.load_frame(i) for i in chunk]), chunk

    def clear_frame_image(self, index: int) -> None:
        """Drop the live-frame raw image cache after downstream sinks publish it."""

        try:
            frame = self.live_scan.frames[int(index)]
        except Exception:
            return
        if hasattr(frame, "map_raw"):
            frame.map_raw = None

    def to_scan(self, *, name: str | None = None, **kwargs: Any) -> Scan:
        return Scan(
            name or self.name,
            [self.frame_for(i) for i in self.frame_indices],
            poni=kwargs.pop("poni", self.poni),
            wavelength=kwargs.pop("wavelength", self.wavelength),
            motors=kwargs.pop("motors", self.motors),
            output_path=kwargs.pop("output_path", self.output_path),
            extra={"source": "xdart.LiveScanFrameSource", **kwargs.pop("extra", {})},
            **kwargs,
        )


def source_from_live_scan(
    live_scan: Any,
    *,
    frame_indices: Iterable[int] | None = None,
) -> LiveScanFrameSource:
    return LiveScanFrameSource(live_scan, frame_indices=frame_indices)


def _resolved_source_path(frame: Any) -> Path | None:
    resolver = getattr(frame, "_resolved_source_path", None)
    path = resolver() if callable(resolver) else getattr(frame, "source_file", "")
    return Path(path) if path else None


def _first_poni(live_scan: Any, labels: list[int]) -> Any | None:
    if not labels:
        return None
    try:
        return getattr(live_scan.frames[int(labels[0])], "poni", None)
    except Exception:
        return None


def _wavelength_angstrom(live_scan: Any) -> float | None:
    try:
        wavelength_m = float((getattr(live_scan, "mg_args", {}) or {}).get("wavelength", 0))
    except (TypeError, ValueError):
        return None
    return wavelength_m * 1e10 if wavelength_m > 0 else None


def _numeric_motors(scan_data: Any) -> dict[str, np.ndarray]:
    if scan_data is None or not hasattr(scan_data, "columns"):
        return {}
    out: dict[str, np.ndarray] = {}
    for col in scan_data.columns:
        try:
            out[str(col)] = np.asarray(scan_data[col].values, dtype=float)
        except (TypeError, ValueError):
            continue
    return out


def _scan_data_row(scan_data: Any, index: int) -> dict[str, Any]:
    if scan_data is None or not hasattr(scan_data, "loc"):
        return {}
    try:
        row = scan_data.loc[int(index)]
    except (KeyError, TypeError, ValueError):
        return {}
    if hasattr(row, "to_dict"):
        return {str(k): v for k, v in row.to_dict().items()}
    return {}


def _frame_geometry(frame: Any) -> FrameGeometry | None:
    incident = None
    if getattr(frame, "gi", False):
        resolver = getattr(frame, "_get_incident_angle", None)
        if callable(resolver):
            try:
                incident = float(resolver())
            except Exception:
                incident = None
    return FrameGeometry(incident_angle=incident) if incident is not None else None


__all__ = [
    "LiveScanFrameSource",
    "source_from_live_scan",
]
