"""xdart live-object adapters for architecture-v2 headless sources."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ssrl_xrd_tools.core import FrameGeometry, FrameView, SourceCapabilities, SourceKind
from ssrl_xrd_tools.core.scan import Scan, ScanFrame
from ssrl_xrd_tools.reduction import (
    CompositeSink,
    NexusSink,
    ReductionPlan,
    ReductionResult,
    ReductionSink,
    XYESink,
    run_reduction,
)

from xdart.modules.frame_publication import (
    FramePublication,
    publication_from_frame_view,
)
from xdart.modules.reduction import plan_from_live_scan


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


@dataclass(slots=True)
class HeadlessRawRef:
    """Display-compatible raw reference around a headless ``ScanFrame``."""

    frame: ScanFrame

    @property
    def map_raw(self):
        return self.frame.image

    @property
    def bg_raw(self):
        return self.frame.background

    @property
    def mask(self):
        return self.frame.mask

    @property
    def thumbnail(self):
        return None


@dataclass(slots=True)
class PublicationSink:
    """Reduction sink that publishes FramePublication snapshots."""

    callback: Callable[[FramePublication], None]
    generation: int = 0
    validate: bool = True
    _scan_name: str = field(default="", init=False, repr=False)

    def begin(self, scan: Scan, plan: ReductionPlan) -> None:
        self._scan_name = scan.name

    def write(self, frame: ScanFrame, reduction) -> None:
        raw = None if frame.image is None else np.asarray(frame.image)
        view = FrameView.from_results(
            label=int(frame.index),
            result_1d=reduction.result_1d,
            result_2d=reduction.result_2d,
            raw=raw,
            metadata_raw=reduction.metadata,
            metadata_numeric=dict(getattr(frame, "metadata_numeric", {}) or {}),
            source_path=None if frame.source_path is None else str(frame.source_path),
            source_frame_index=frame.source_frame_index,
        )
        publication = publication_from_frame_view(
            view,
            generation=self.generation,
            source_identity=f"{self._scan_name}:{frame.index}",
            raw_ref=HeadlessRawRef(frame),
            raw_status=("ready" if raw is not None else "lazy"),
            validate=self.validate,
        )
        self.callback(publication)

    def finish(self, result: ReductionResult) -> None:
        return None


@dataclass(slots=True)
class ReductionJob:
    """A complete xdart-to-ssrl reduction call description."""

    source: LiveScanFrameSource
    plan: ReductionPlan
    sink: ReductionSink
    chunk_size: int = 1
    clear_frame_images: bool = False
    run_kwargs: dict[str, Any] = field(default_factory=dict)

    def run(self) -> ReductionResult:
        return run_reduction(
            self.plan,
            self.source,
            self.sink,
            chunk_size=self.chunk_size,
            clear_frame_images=self.clear_frame_images,
            **self.run_kwargs,
        )


def source_from_live_scan(
    live_scan: Any,
    *,
    frame_indices: Iterable[int] | None = None,
) -> LiveScanFrameSource:
    return LiveScanFrameSource(live_scan, frame_indices=frame_indices)


def build_reduction_job(
    live_scan: Any,
    *,
    frame_indices: Iterable[int] | None = None,
    integrate_1d: bool = True,
    integrate_2d: bool | None = None,
    sinks: Iterable[ReductionSink] | None = None,
    publish: Callable[[FramePublication], None] | None = None,
    output_nexus: str | Path | None = None,
    output_xye: str | Path | None = None,
    chunk_size: int = 1,
    clear_frame_images: bool = True,
    **run_kwargs: Any,
) -> ReductionJob:
    """Build the architecture-v2 reduction job for a live scan.

    RESTRUCTURE-TODO(WS-X1): image/nexus/live/reintegrate compute paths now
    call the headless reducer through ``reduce_live_frames``.  Promote the
    wrangler threads themselves to build and run ``ReductionJob`` directly
    once save/progress/cancel semantics are represented entirely as sinks.
    """

    source = LiveScanFrameSource(live_scan, frame_indices=frame_indices)
    plan = plan_from_live_scan(
        live_scan,
        integrate_1d=integrate_1d,
        integrate_2d=integrate_2d,
        gi_incident_angle=_safe_first_incident_angle(live_scan, source.frame_indices),
    )
    sink_list: list[ReductionSink] = list(sinks or [])
    if publish is not None:
        sink_list.append(PublicationSink(publish))
    if output_nexus is not None:
        sink_list.append(NexusSink(output_nexus, overwrite=True))
    if output_xye is not None:
        sink_list.append(XYESink(output_xye))
    sink: ReductionSink
    if not sink_list:
        from ssrl_xrd_tools.reduction import MemorySink

        sink = MemorySink()
    elif len(sink_list) == 1:
        sink = sink_list[0]
    else:
        sink = CompositeSink(tuple(sink_list))
    return ReductionJob(
        source=source,
        plan=plan,
        sink=sink,
        chunk_size=chunk_size,
        clear_frame_images=clear_frame_images,
        run_kwargs=run_kwargs,
    )


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


def _safe_first_incident_angle(live_scan: Any, labels: list[int]) -> float | None:
    if not getattr(live_scan, "gi", False) or not labels:
        return None
    try:
        return float(live_scan.frames[int(labels[0])]._get_incident_angle())
    except Exception:
        return None


__all__ = [
    "LiveScanFrameSource",
    "PublicationSink",
    "ReductionJob",
    "build_reduction_job",
    "source_from_live_scan",
]
