"""Scan/frame-oriented headless reduction primitives.

The intent of this module is to give GUIs and notebooks one small, stable
surface for common reduction jobs while keeping the numerical work in
``ssrl_xrd_tools.integrate``.  xdart should eventually build these objects
from its UI state and display the returned results, rather than owning
integration loops itself.
"""

from __future__ import annotations

import os
import queue
import threading
import time
import uuid
import warnings
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Protocol, runtime_checkable

import numpy as np

from ssrl_xrd_tools.core.containers import (
    IntegrationResult1D,
    IntegrationResult2D,
    PONI,
)
from ssrl_xrd_tools.core.metadata import ScanMetadata
from ssrl_xrd_tools.core.scan import (
    FrameSource as CoreFrameSource,
    ImageLoader,
    MaskSpec as CoreMaskSpec,
    Scan as CoreScan,
    ScanFrame,
)
from ssrl_xrd_tools.io.export import write_xye
from ssrl_xrd_tools.io.image import read_image
from ssrl_xrd_tools.io.nexus import (
    open_nexus_image_stack,
    open_nexus_writer,
    upsert_scan_metadata,
    write_nexus_frame,
)

if TYPE_CHECKING:  # C4 — tighter Scan.integrator type without forcing the import
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator

ProgressCallback = Callable[["ReductionProgress"], None]


class GIFreezeError(ValueError):
    """Raised when the GI output-range freeze pre-pass cannot produce a grid.

    Subclasses :class:`ValueError` so existing broad ``except ValueError``
    callers still catch it, while letting GUIs translate this *specific* GI
    failure (a blank or degenerate scout frame) into actionable guidance
    without matching on the message text.
    """


def poni_to_integrator(*args: Any, **kwargs: Any) -> Any:
    from ssrl_xrd_tools.integrate.calibration import poni_to_integrator as _impl

    return _impl(*args, **kwargs)


def poni_to_fiber_integrator(*args: Any, **kwargs: Any) -> Any:
    from ssrl_xrd_tools.integrate.calibration import (
        poni_to_fiber_integrator as _impl,
    )

    return _impl(*args, **kwargs)


def integrate_1d(*args: Any, **kwargs: Any) -> IntegrationResult1D:
    from ssrl_xrd_tools.integrate.single import integrate_1d as _impl

    return _impl(*args, **kwargs)


def integrate_2d(*args: Any, **kwargs: Any) -> IntegrationResult2D:
    from ssrl_xrd_tools.integrate.single import integrate_2d as _impl

    return _impl(*args, **kwargs)


def integrate_gi_1d(*args: Any, **kwargs: Any) -> IntegrationResult1D:
    from ssrl_xrd_tools.integrate.gid import integrate_gi_1d as _impl

    return _impl(*args, **kwargs)


def integrate_gi_2d(*args: Any, **kwargs: Any) -> IntegrationResult2D:
    from ssrl_xrd_tools.integrate.gid import integrate_gi_2d as _impl

    return _impl(*args, **kwargs)


def integrate_gi_exitangles(*args: Any, **kwargs: Any) -> IntegrationResult2D:
    from ssrl_xrd_tools.integrate.gid import integrate_gi_exitangles as _impl

    return _impl(*args, **kwargs)


def integrate_gi_exitangles_1d(*args: Any, **kwargs: Any) -> IntegrationResult1D:
    from ssrl_xrd_tools.integrate.gid import integrate_gi_exitangles_1d as _impl

    return _impl(*args, **kwargs)


def integrate_gi_polar(*args: Any, **kwargs: Any) -> IntegrationResult2D:
    from ssrl_xrd_tools.integrate.gid import integrate_gi_polar as _impl

    return _impl(*args, **kwargs)


def integrate_gi_polar_1d(*args: Any, **kwargs: Any) -> IntegrationResult1D:
    from ssrl_xrd_tools.integrate.gid import integrate_gi_polar_1d as _impl

    return _impl(*args, **kwargs)


@dataclass(slots=True)
class CancelToken:
    """Small cancellation primitive shared by GUI and headless callers."""

    cancelled: bool = False

    def cancel(self) -> None:
        self.cancelled = True


@dataclass(slots=True)
class Frame:
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

    def __post_init__(self) -> None:
        if self.source_path is not None and not isinstance(self.source_path, Path):
            self.source_path = Path(self.source_path)

    def load_image(self) -> np.ndarray:
        """Return this frame's image, loading from provenance if needed.

        **Side effect**: caches the loaded array in ``self.image`` so
        repeat calls don't re-read disk.  Pair with
        ``run_reduction(clear_frame_images=True)`` to release the
        cached array once the frame has been written to its sink.
        """
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
            with open_nexus_image_stack(path) as stack:
                self.image = np.asarray(stack[int(self.source_frame_index)])
        else:
            self.image = np.asarray(read_image(path))
        return self.image

    @property
    def label(self) -> str:
        return str(self.index)


@dataclass(frozen=True, slots=True)
class MaskSpec:
    """Detector mask that can be resolved once a frame shape is known.

    ``values`` may be a 2D mask image, a flat boolean mask, or flat detector
    pixel indices.  This lets GUIs preserve flat index masks when the first
    image has not been loaded yet.
    """

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


@runtime_checkable
class FrameSource(Protocol):
    """Minimal image-stream boundary shared by reduction, RSM, and stitching."""

    @property
    def frame_indices(self) -> list[int]:
        ...

    def load_frame(self, index: int) -> np.ndarray:
        ...

    def iter_chunks(self, chunk_size: int) -> Iterator[tuple[np.ndarray, list[int]]]:
        ...


@dataclass(slots=True)
class Scan:
    """Ordered set of frames with scan-level reduction context."""

    name: str
    frames: list[Frame]
    poni: PONI | None = None
    # ``integrator`` is typed as Any at runtime so importing this module
    # never pulls pyFAI in; the TYPE_CHECKING alias above gives the IDE
    # an ``AzimuthalIntegrator`` hint without paying the import cost.
    integrator: "AzimuthalIntegrator | None" = None
    metadata: ScanMetadata | None = None
    energy: float | None = None
    wavelength: float | None = None
    motors: dict[str, np.ndarray] = field(default_factory=dict)
    output_path: Path | str | None = None
    sample_name: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    _frame_by_index: dict[int, Frame] = field(init=False, repr=False)

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

    def __iter__(self) -> Iterable[Frame]:
        return iter(self.frames)

    @property
    def frame_indices(self) -> list[int]:
        """Ordered labels exposed through the shared frame-source boundary."""
        return [int(frame.index) for frame in self.frames]

    @property
    def energy_keV(self) -> float | None:
        """Photon energy in keV, matching :class:`ScanMetadata.energy`."""
        return self.energy

    @property
    def energy_eV(self) -> float | None:
        """Photon energy in eV for RSM/gridder consumers."""
        return None if self.energy is None else float(self.energy) * 1000.0

    def load_frame(self, index: int) -> np.ndarray:
        """Load one detector image by frame label."""
        try:
            frame = self._frame_by_index[int(index)]
        except KeyError as exc:
            raise KeyError(f"Scan has no frame {index}") from exc
        return np.asarray(frame.load_image())

    def iter_chunks(self, chunk_size: int) -> Iterator[tuple[np.ndarray, list[int]]]:
        """Yield bounded image chunks for streaming consumers such as RSM."""
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0; got {chunk_size}")
        indices = self.frame_indices
        for start in range(0, len(indices), chunk_size):
            chunk_indices = indices[start:start + chunk_size]
            loaded_here: list[Frame] = []
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
        """Return explicit metadata or synthesize minimal NeXus metadata."""
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
            source="reduction.Scan",
            image_paths=[
                Path(f.source_path) for f in self.frames
                if f.source_path is not None
            ],
            h5_path=None,
            extra=self.extra.copy(),
        )

    def to_scan_data(self):
        """Per-frame condition table as a pandas DataFrame (one row per frame).

        Index = :attr:`frame_indices`; columns = the union of every
        ``Frame.metadata`` key (first-seen order) plus any per-frame
        :attr:`motors` array, one value per frame.  Keys missing on a given
        frame become ``NaN``; a motor array overrides a same-named metadata
        column (the motor array is the authoritative per-frame record).

        This is the headless analog of the GUI's ``scan.scan_data`` — the table
        :class:`NexusSink` persists under ``/entry/scan_data`` so that
        variable-correlated analyses (lattice vs stress, texture vs angle,
        property vs time/temperature) can correlate each integrated frame with
        the experimental conditions that produced it.
        """
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


# Architecture-v2 canonical aliases.  The legacy definitions above remain
# temporarily as an internal migration cushion; public reduction imports now
# resolve to the headless core contracts in ``ssrl_xrd_tools.core.scan``.
Frame = ScanFrame
MaskSpec = CoreMaskSpec
FrameSource = CoreFrameSource
Scan = CoreScan


@dataclass(slots=True)
class Integration1DPlan:
    """1D integration settings for one reduction output."""

    npt: int = 1000
    unit: str = "q_A^-1"
    method: str = "csr"
    radial_range: tuple[float, float] | None = None
    azimuth_range: tuple[float, float] | None = None
    monitor_key: str | None = None
    error_model: str | None = None
    polarization_factor: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.npt <= 0:
            raise ValueError(f"Integration1DPlan.npt must be > 0; got {self.npt}")


@dataclass(slots=True)
class Integration2DPlan:
    """2D integration settings for one reduction output."""

    npt_rad: int = 1000
    npt_azim: int = 360
    unit: str = "q_A^-1"
    method: str = "csr"
    radial_range: tuple[float, float] | None = None
    azimuth_range: tuple[float, float] | None = None
    azimuth_offset: float = 0.0
    monitor_key: str | None = None
    error_model: str | None = None
    polarization_factor: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.npt_rad <= 0 or self.npt_azim <= 0:
            raise ValueError(
                "Integration2DPlan.npt_rad and npt_azim must both be > 0; "
                f"got ({self.npt_rad}, {self.npt_azim})"
            )


class GI1DMode(str, Enum):
    """Supported grazing-incidence 1D output coordinates."""

    Q_TOTAL = "q_total"
    Q_IP = "q_ip"
    Q_OOP = "q_oop"
    EXIT_ANGLE = "exit_angle"


class GI2DMode(str, Enum):
    """Supported grazing-incidence 2D output coordinates."""

    QIP_QOOP = "qip_qoop"
    Q_CHI = "q_chi"
    EXIT_ANGLES = "exit_angles"


@dataclass(frozen=True, slots=True)
class GIMode:
    """Grazing-incidence reduction parameters.

    When present on a :class:`ReductionPlan`, ``run_reduction`` builds a
    pyFAI :class:`FiberIntegrator` from ``scan.poni`` + these settings
    and dispatches to :func:`integrate_gi_1d` / :func:`integrate_gi_2d`
    instead of the standard pyFAI integrator path.

    Encoding the GI parameters as a single optional sum-type field
    (rather than a ``gi: bool`` flag with five sibling fields) means
    invalid configurations like ``gi=False, gi_incident_angle=2.5``
    aren't representable.
    """

    incident_angle: float | None = None
    incidence_motor: str | None = None
    tilt_angle: float = 0.0
    sample_orientation: int = 1
    method: str = "no"
    mode_1d: GI1DMode | str = GI1DMode.Q_TOTAL
    mode_2d: GI2DMode | str = GI2DMode.QIP_QOOP
    npt_oop: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode_1d", _coerce_gi_1d_mode(self.mode_1d))
        object.__setattr__(self, "mode_2d", _coerce_gi_2d_mode(self.mode_2d))
        if self.incident_angle is not None:
            object.__setattr__(self, "incident_angle", float(self.incident_angle))
        if self.npt_oop is not None and int(self.npt_oop) <= 0:
            raise ValueError(f"GIMode.npt_oop must be > 0; got {self.npt_oop}")
        if self.npt_oop is not None:
            object.__setattr__(self, "npt_oop", int(self.npt_oop))


@dataclass(slots=True)
class ReductionPlan:
    """Reduction settings — the *content* of a reduction job.

    Execution policy (``chunk_size``, ``clear_frame_images``) lives on
    :func:`run_reduction` instead, so the same plan can be saved once
    and run with different chunking on different scans.
    """

    integration_1d: Integration1DPlan | None = field(default_factory=Integration1DPlan)
    integration_2d: Integration2DPlan | None = None
    gi: GIMode | None = None
    mask: np.ndarray | MaskSpec | None = None
    threshold_min: float | None = None
    threshold_max: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.integration_1d is None and self.integration_2d is None:
            raise ValueError(
                "ReductionPlan must include integration_1d or integration_2d."
            )


@dataclass(slots=True)
class FrameReduction:
    """Reduction products for one frame."""

    frame_index: int
    result_1d: IntegrationResult1D | None = None
    result_2d: IntegrationResult2D | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReductionProgress:
    """Progress event emitted by :func:`run_reduction`."""

    scan_name: str
    stage: str
    frame_index: int | None
    completed: int
    total: int
    message: str = ""


@dataclass(slots=True)
class ReductionResult:
    """Summary returned by :func:`run_reduction`."""

    scan_name: str
    frames: dict[int, FrameReduction]
    n_processed: int
    cancelled: bool = False
    output_path: Path | None = None
    failed: bool = False
    error: str | None = None


class ReductionSink(Protocol):
    """Destination for frame reduction products."""

    def begin(self, scan: Scan, plan: ReductionPlan) -> None: ...
    def write(self, frame: Frame, reduction: FrameReduction) -> None: ...
    def finish(self, result: ReductionResult) -> None: ...


@dataclass(slots=True)
class MemorySink:
    """In-memory sink for notebooks, tests, and xdart display handoff."""

    frames: dict[int, FrameReduction] = field(default_factory=dict)

    def begin(self, scan: Scan, plan: ReductionPlan) -> None:
        self.frames.clear()

    def write(self, frame: Frame, reduction: FrameReduction) -> None:
        self.frames[int(frame.index)] = reduction

    def finish(self, result: ReductionResult) -> None:
        return None


@dataclass(slots=True)
class CompositeSink:
    """Fan out reduction products to multiple sinks."""

    sinks: tuple[ReductionSink, ...]

    def begin(self, scan: Scan, plan: ReductionPlan) -> None:
        for sink in self.sinks:
            sink.begin(scan, plan)

    def write(self, frame: Frame, reduction: FrameReduction) -> None:
        for sink in self.sinks:
            sink.write(frame, reduction)

    def replace(self, frame: Frame, reduction: FrameReduction) -> None:
        for sink in self.sinks:
            _emit_sink_replace(sink, frame, reduction)

    def finish(self, result: ReductionResult) -> None:
        errors: list[BaseException] = []
        for sink in self.sinks:
            try:
                sink.finish(result)
            except BaseException as exc:  # pragma: no cover - defensive fan-out
                errors.append(exc)
        if errors:
            raise errors[0]

    def abort(self, result: ReductionResult) -> None:
        errors: list[BaseException] = []
        for sink in self.sinks:
            abort = getattr(sink, "abort", None)
            try:
                if callable(abort):
                    abort(result)
                else:
                    sink.finish(result)
            except BaseException as exc:  # pragma: no cover - defensive fan-out
                errors.append(exc)
        if errors:
            raise errors[0]


def _emit_sink_replace(
    sink: ReductionSink, frame: Frame, reduction: FrameReduction
) -> None:
    """Re-emit an already-written frame as a *replace*.

    Used when a session is fed an index it has already processed (reintegrate /
    replace re-feed).  Sinks that distinguish replace from first-write expose a
    ``replace`` hook; the rest fall back to ``write`` because their ``write`` is
    already idempotent per frame index (MemorySink/XYESink overwrite by index,
    NexusSink upserts the frame slot).
    """

    replace = getattr(sink, "replace", None)
    if callable(replace):
        replace(frame, reduction)
    else:
        sink.write(frame, reduction)


@dataclass(slots=True)
class XYESink:
    """Write 1D reductions as one ``.xye`` file per frame."""

    directory: Path | str
    pattern: str = "{scan}_{frame:04d}.xye"
    _scan_name: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.directory, Path):
            self.directory = Path(self.directory)

    def begin(self, scan: Scan, plan: ReductionPlan) -> None:
        self._scan_name = scan.name
        self.directory.mkdir(parents=True, exist_ok=True)

    def write(self, frame: Frame, reduction: FrameReduction) -> None:
        result = reduction.result_1d
        if result is None:
            return
        path = self.directory / self.pattern.format(
            scan=self._scan_name,
            frame=int(frame.index),
            label=frame.label,
        )
        write_xye(path, result.radial, result.intensity, result.sigma)

    def finish(self, result: ReductionResult) -> None:
        return None


@dataclass(slots=True)
class NexusSink:
    """Frame-by-frame NeXus sink backed by ``ssrl_xrd_tools.io.nexus``."""

    path: Path | str
    entry: str = "entry"
    compression: str | None = "lzf"
    overwrite: bool = False
    swmr: bool = False
    flush_every: int | None = 16
    atomic: bool | None = None
    _h5: Any | None = field(default=None, init=False, repr=False)
    _n_written: int = field(default=0, init=False, repr=False)
    _scan: "Scan | None" = field(default=None, init=False, repr=False)
    _active_path: Path | None = field(default=None, init=False, repr=False)
    _tmp_path: Path | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            self.path = Path(self.path)

    def begin(self, scan: Scan, plan: ReductionPlan) -> None:
        if self.flush_every is not None and self.flush_every <= 0:
            raise ValueError(f"flush_every must be > 0 or None; got {self.flush_every}")
        self._n_written = 0
        # Stash the scan so finish() can persist its per-frame condition table
        # (scan_data) alongside the integrated stacks (core provenance).
        self._scan = scan
        use_atomic = (
            bool(self.atomic)
            if self.atomic is not None
            else (not self.swmr and (self.overwrite or not self.path.exists()))
        )
        self._tmp_path = None
        self._active_path = self.path
        if use_atomic:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._tmp_path = self.path.with_name(
                f".{self.path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            self._active_path = self._tmp_path
        self._h5 = open_nexus_writer(
            self._active_path,
            metadata=scan.to_metadata(),
            entry=self.entry,
            compression=self.compression,
            swmr=self.swmr,
            overwrite=True if use_atomic else self.overwrite,
        )

    def write(self, frame: Frame, reduction: FrameReduction) -> None:
        if self._h5 is None:
            raise RuntimeError("NexusSink.write called before begin().")
        write_nexus_frame(
            self._h5,
            frame.index,
            result_1d=reduction.result_1d,
            result_2d=reduction.result_2d,
            entry=self.entry,
            compression=self.compression,
        )
        self._n_written += 1
        if self.flush_every is not None and self._n_written % self.flush_every == 0:
            self._h5.flush()

    def finish(self, result: ReductionResult) -> None:
        if self._h5 is None:
            return
        tmp_path = self._tmp_path
        try:
            # Persist the per-frame condition table (scan_data) after the final
            # integrated frame, before close.  A finish-time single upsert is
            # correct for a batch sink; only the GUI's SWMR live consumers would
            # need scan_data mid-run (they'd require a per-write incremental
            # upsert).  Any failure here propagates (after abort() below) so a
            # lost condition table is surfaced, not silent — and abort()
            # preserves the frames written so far (atomic mode keeps the tmp
            # as <output>.partial instead of unlinking it, T0-6/S7).
            scan = self._scan
            if scan is not None:
                scan_data = scan.to_scan_data()
                if scan_data is not None and len(scan_data.columns):
                    upsert_scan_metadata(
                        self._h5[self.entry], scan_data, scan.frame_indices,
                    )
            self._h5.flush()
            self._h5.close()
            self._h5 = None
            self._scan = None
            if tmp_path is not None:
                tmp_path.replace(self.path)
        except BaseException:
            try:
                self.abort(result)
            finally:
                raise
        finally:
            self._active_path = None
            self._tmp_path = None

    def abort(self, result: ReductionResult) -> None:
        """Failure teardown.  NEVER destroys written data (T0-6/S7): in atomic
        mode the tmp file holds every frame written this run, so instead of
        unlinking it (which converted a finish-time failure into deletion of
        the whole run) it is preserved as ``<output>.partial`` and a warning
        names it.  Non-atomic mode writes in place — nothing to move."""
        h5 = self._h5
        tmp_path = self._tmp_path
        self._h5 = None
        self._scan = None
        self._active_path = None
        self._tmp_path = None
        if h5 is not None:
            try:
                h5.close()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
        if tmp_path is not None and tmp_path.exists():
            partial = Path(str(self.path) + ".partial")
            try:
                tmp_path.replace(partial)
                warnings.warn(
                    f"NexusSink.abort: run failed — frames written so far are "
                    f"preserved at {partial} (not a finalized scan file).",
                    RuntimeWarning, stacklevel=2,
                )
            except Exception:  # pragma: no cover - best-effort preservation
                # Could not even rename: leave the tmp where it is rather
                # than deleting it.
                warnings.warn(
                    f"NexusSink.abort: run failed — could not rename the "
                    f"partial output; data left at {tmp_path}.",
                    RuntimeWarning, stacklevel=2,
                )


@dataclass(slots=True)
class ReductionSession:
    """Incremental headless reduction engine for one scan/run.

    ``ReductionSession`` is the stateful counterpart to
    :func:`run_reduction`.  It owns the executor, per-thread pyFAI
    integrators, sink lifecycle, progress, and cancellation for the scan
    lifetime, so callers can feed chunks without rebuilding CSR-LUTs or
    reopening sinks every chunk.
    """

    plan: ReductionPlan
    source: Scan | FrameSource
    sink: ReductionSink | Iterable[ReductionSink] | None = None
    chunk_size: int = 1
    clear_frame_images: bool = False
    progress_cb: ProgressCallback | None = None
    cancel_token: CancelToken | None = None
    executor: Any | None = None
    gi_freeze_mode: str | None = None
    # Execution policy.  "chunked" (default) keeps the existing
    # ``process(chunk)`` submit-then-drain-in-order loop.  "streaming" exposes
    # ``submit(frame)`` — each frame is dispatched to the persistent pool the
    # instant it is read (no chunk barrier), a bounded in-flight window keeps
    # the reader from outrunning integration, and ONE internal writer/consumer
    # thread drains completed reductions and calls the sink by frame index
    # (HDF5 is not thread-safe → exactly one thread ever touches the sink).
    execution: str = "chunked"
    # Max frames in flight (submitted but not yet written) in streaming mode.
    # ``None`` → 2× the pool's worker count.  This bounds peak memory and stops
    # the reader starving the pool.
    inflight_max: int | None = None
    # S2: whether completed FrameReduction objects (including full 2D arrays)
    # are retained in ``self._products`` for the session's lifetime.  True
    # (default) preserves the historical contract (``result.frames`` holds
    # every reduction — what headless run_reduction() callers consume).
    # False bounds memory for sink-driven runs where the data already lands
    # on disk per frame: ~1.4 MB/frame of 2D cake → ~14 GB retained on a
    # 10k-frame scan.  With False, ``result.frames`` is EMPTY — read results
    # back from the sink's output.  Replace/re-feed detection is tracked
    # independently (``_seen_idxs``), so A1 idempotency is unaffected.
    retain_products: bool = True
    scan: Scan = field(init=False)
    result: ReductionResult | None = field(default=None, init=False)
    integrator_provider_builds: int = field(default=0, init=False)
    _sink: ReductionSink = field(init=False, repr=False)
    _worker: Any | None = field(default=None, init=False, repr=False)
    _owns_worker: bool = field(default=False, init=False, repr=False)
    _integrators: _ReductionIntegratorProvider = field(init=False, repr=False)
    _plan_masks: dict[tuple[int, int], np.ndarray | None] = field(
        default_factory=dict, init=False, repr=False,
    )
    _products: dict[int, FrameReduction] = field(default_factory=dict, init=False, repr=False)
    _seen_idxs: set[int] = field(default_factory=set, init=False, repr=False)
    _completed: int = field(default=0, init=False, repr=False)
    _cancelled: bool = field(default=False, init=False, repr=False)
    _failure: BaseException | None = field(default=None, init=False, repr=False)
    _started: bool = field(default=False, init=False, repr=False)
    _finished: bool = field(default=False, init=False, repr=False)
    _output_path: Path | None = field(default=None, init=False, repr=False)
    _freeze_policy: str | None = field(default=None, init=False, repr=False)
    _initial_incident_angle: float | None = field(default=None, init=False, repr=False)
    _gi_freeze_applied: bool = field(default=False, init=False, repr=False)
    _scan_frame_positions: dict[int, int] = field(
        default_factory=dict, init=False, repr=False,
    )
    # Streaming-mode machinery (execution="streaming"); unused when chunked.
    _semaphore: Any = field(default=None, init=False, repr=False)
    _write_queue: Any = field(default=None, init=False, repr=False)
    _writer_thread: Any = field(default=None, init=False, repr=False)
    _stream_started: bool = field(default=False, init=False, repr=False)
    _submitted: int = field(default=0, init=False, repr=False)
    _writer_ident: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0; got {self.chunk_size}")
        self.scan = _coerce_to_scan(self.source)
        self._sink = _coerce_sink(self.sink)
        self.cancel_token = self.cancel_token or CancelToken()
        self._freeze_policy = _normalize_gi_freeze_mode(self.gi_freeze_mode)
        self._output_path = _sink_path(self._sink) or (
            self.scan.output_path if isinstance(self.scan.output_path, Path) else None
        )
        self._scan_frame_positions = {
            int(frame.index): pos for pos, frame in enumerate(self.scan.frames)
        }

        ai = None
        fi = None
        if self.plan.gi is not None:
            if self.scan.poni is None:
                raise ValueError("GI reduction requires scan.poni.")
            self._initial_incident_angle = _resolve_gi_incident_angle(
                self.scan.frames[0] if self.scan.frames else None,
                self.plan.gi,
            )
            if self._freeze_policy in {"first_frame", "scout_union"}:
                self._apply_gi_freeze(self._freeze_policy)
        else:
            ai = self.scan.integrator
            if ai is None and self.scan.poni is None:
                raise ValueError("Reduction requires scan.integrator or scan.poni.")

        self._integrators = _ReductionIntegratorProvider(
            scan=self.scan,
            plan=self.plan,
            ai=ai,
            fi=fi,
            initial_incident_angle=self._initial_incident_angle,
        )
        self.integrator_provider_builds = 1
        # S8: the monitor warn-once set is per-process; without this reset a
        # dead monitor in scan 1 silences the warning for every later scan in
        # a long GUI session.  Per-session scope = once per run.
        _warned_monitor_keys.clear()
        # Validate BEFORE acquiring resources: sink.begin() opens an h5 handle
        # (atomic NexusSink also creates its hidden .tmp) and _coerce_executor
        # may build an owned pool — a ValueError after those leaks both, with
        # no abort path that ever cleans the orphaned tmp.
        if self.execution not in ("chunked", "streaming"):
            raise ValueError(
                f"execution must be 'chunked' or 'streaming'; got {self.execution!r}"
            )
        self._worker, self._owns_worker = _coerce_executor(self.executor)
        self._sink.begin(self.scan, self.plan)
        self._started = True
        _emit(self.progress_cb, self.scan.name, "start", None, 0, len(self.scan))

        if self.execution == "streaming":
            self._init_streaming()

    def __enter__(self) -> ReductionSession:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            # A body exception is already propagating — finish for cleanup but
            # do NOT raise (that would mask the original exception).
            self._failure = self._failure or exc
            self.finish(raise_on_failure=False)
        else:
            # Body succeeded; surface a swallowed write/sink failure (fail-loud)
            # so even a bare ``with ReductionSession(...) as s: s.process()``
            # can't silently report success on a failed write.
            self.finish()

    @property
    def frames(self) -> dict[int, FrameReduction]:
        """Completed frame reductions accumulated so far."""

        return self._products

    def release_products(self, indices) -> None:
        """Drop retained :class:`FrameReduction` objects for *indices*.

        For persistent chunked sessions whose caller harvests each chunk's
        results from :attr:`frames` (the serial/true-live per-frame pattern):
        without releasing, a session reused across a long watch run retains
        every frame's products — the same unbounded growth that
        ``retain_products=False`` solves for sink-driven streaming.  Replace /
        re-feed detection is unaffected (``_seen_idxs`` is kept), so a
        released-then-re-fed index still counts as a replace, not a new
        completion."""
        for idx in indices:
            self._products.pop(int(idx), None)

    def process(
        self,
        frames_or_chunk: Iterable[Frame] | tuple[Any, Iterable[int]] | None = None,
        images: Iterable[np.ndarray | None] | None = None,
    ) -> None:
        """Reduce the next frames/chunk.

        With no argument, the session streams its original source using
        ``chunk_size``.  Supplying frames (and optional image arrays) lets a GUI
        feed newly-acquired chunks while preserving this session's executor,
        integrators, sinks, and progress accounting.
        """

        if self.execution == "streaming":
            raise RuntimeError(
                "execution='streaming' uses submit(); process() is chunked-only"
            )
        if self._finished:
            raise RuntimeError("ReductionSession.process called after finish().")
        if self._cancelled or self.cancel_token.cancelled:
            self._cancelled = True
            return

        try:
            if frames_or_chunk is None:
                for chunk, chunk_images in _iter_reduction_chunks(
                    self.source, self.scan, self.chunk_size,
                ):
                    self._process_chunk(chunk, chunk_images)
                    if self._cancelled:
                        break
                return

            chunk, chunk_images = self._normalize_process_input(frames_or_chunk, images)
            self._register_process_frames(chunk)
            self._process_chunk(chunk, chunk_images)
        except BaseException as exc:
            self._failure = exc
            raise

    def _init_streaming(self) -> None:
        """Set up the bounded in-flight window + single writer thread.

        Reuses the persistent pool (``self._worker``) and the per-thread
        integrator provider; only adds a ``Semaphore`` (the in-flight bound), a
        FIFO queue, and one consumer thread.  Called once from ``__post_init__``
        AFTER the executor + integrators + GI freeze are in place, so the freeze
        (which needs first+last frames) is fixed before any frame is submitted.
        """
        if self._worker is None:
            # Streaming needs a real pool; build a default owned one when the
            # caller passed executor=None/False.
            self._worker, self._owns_worker = ThreadPoolExecutor(), True
        n_workers = getattr(self._worker, "_max_workers", None) or 4
        bound = (
            self.inflight_max
            if self.inflight_max and self.inflight_max > 0
            else max(2, 2 * n_workers)
        )
        self.inflight_max = bound
        self._semaphore = threading.Semaphore(bound)
        self._write_queue = queue.Queue()
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name=f"reduction-writer-{self.scan.name}",
            daemon=True,
        )
        self._writer_thread.start()
        self._stream_started = True

    def submit(self, frame: Frame, image: np.ndarray | None = None) -> None:
        """Stream one frame into the pool immediately (execution="streaming").

        Blocks when ``inflight_max`` frames are already in flight (bounded
        memory), then dispatches integration to the persistent pool and hands
        the ``(frame, future)`` to the single writer thread, which writes it to
        the sink by frame index once it completes.  Out-of-order completion is
        fine — the sink/writer is index-addressed.  A no-op once cancelled.
        """
        if self.execution != "streaming":
            raise RuntimeError("submit() requires execution='streaming'")
        if self._finished:
            raise RuntimeError("ReductionSession.submit called after finish().")
        if self._failure is not None:
            raise self._failure
        if self._cancelled or self.cancel_token.cancelled:
            self._cancelled = True
            return
        self._register_process_frames([frame])
        # Bounded in-flight: blocks the reader so it can't outrun integration
        # (flat peak memory).  Released by the writer thread per frame.
        # Timed-acquire loop: poll the cancel token so Stop/Pause can interrupt
        # a full in-flight window without waiting for a worker slot to free.
        # Returns cleanly (no raise) so the caller's stop-check handles it safely
        # (raising here escapes through run() which has no except, tearing down
        # the QThread — the same trap the GIFreezeError fix addressed).
        while not self._semaphore.acquire(timeout=0.1):
            if self._cancelled or self.cancel_token.cancelled:
                self._cancelled = True
                return
            if (self._writer_thread is not None
                    and not self._writer_thread.is_alive()):
                # T0-7/S1 belt-and-suspenders: the writer died, so no slot
                # will ever free.  Record the failure and return cleanly
                # (matching the cancel path — raising here would escape the
                # caller's run() and tear down the QThread); the NEXT submit
                # raises the recorded failure at its fail-loud precheck.
                self._failure = self._failure or RuntimeError(
                    "ReductionSession writer thread died; run cannot proceed"
                )
                self._cancelled = True
                return
        if self._cancelled or self.cancel_token.cancelled:
            self._cancelled = True
            self._semaphore.release()
            return
        try:
            future = self._worker.submit(self._stream_reduce, frame, image)
        except BaseException as exc:
            # Pool/interpreter-level dispatch failure: the in-flight permit
            # acquired above must be returned, else later submits deadlock
            # on a semaphore that can never refill.  Record fail-loud.
            self._semaphore.release()
            self._failure = self._failure or exc
            self._cancelled = True
            raise
        self._submitted += 1
        self._write_queue.put((frame, future))

    def _stream_reduce(self, frame: Frame, image: np.ndarray | None):
        """Worker-thread task: integrate, then run the sink's per-frame
        ``worker_process`` hook (if any) so expensive per-frame prep — e.g.
        xdart's thumbnail + raw-free — happens in PARALLEL across the pool
        rather than serially on the single writer thread.  The writer then only
        does the index-addressed HDF5 write.  Cancellation/errors propagate
        through the future to the writer loop unchanged.
        """
        reduction = _reduce_frame(
            frame, image, self.plan, self._integrators, self._plan_masks,
            self.cancel_token,
        )
        worker_process = getattr(self._sink, "worker_process", None)
        if callable(worker_process):
            worker_process(frame, reduction)
        return reduction

    def _writer_loop(self) -> None:
        """The single consumer thread: drain completed frames → sink by index.

        The ONLY thread that ever calls the sink (``write``/``replace``),
        satisfying the HDF5-single-writer invariant.  Releases one in-flight
        slot per frame so ``submit`` can proceed.  Records (does not raise)
        failures so a bad write surfaces in ``finish`` without deadlocking the
        bounded window.  A re-fed index is a *replace* (not a new completion),
        preserving the A1 idempotency contract.
        """
        self._writer_ident = threading.get_ident()
        while True:
            item = self._write_queue.get()
            if item is _STREAM_SENTINEL:
                self._write_queue.task_done()
                break
            frame, future = item
            try:
                try:
                    reduction = future.result()
                except _ReductionCancelled:
                    self._cancelled = True
                    if self.clear_frame_images:
                        frame.image = None
                        frame.background = None
                    continue
                except BaseException as exc:  # integration failure for one frame
                    self._failure = self._failure or exc
                    if self.clear_frame_images:
                        frame.image = None
                        frame.background = None
                    continue
                try:
                    idx = int(frame.index)
                    replacing = idx in self._seen_idxs
                    self._seen_idxs.add(idx)
                    if self.retain_products:
                        self._products[idx] = reduction
                    if replacing:
                        _emit_sink_replace(self._sink, frame, reduction)
                    else:
                        self._sink.write(frame, reduction)
                        self._completed += 1
                except BaseException as exc:
                    self._failure = self._failure or exc
                else:
                    # T0-7/S1: a progress-callback or image-clear exception
                    # must be RECORDED, not allowed to escape — an escape
                    # kills this thread, after which submit() blocks forever
                    # on the in-flight semaphore and finish() join()s a dead
                    # thread and reports SUCCESS with frames missing.
                    try:
                        _emit(self.progress_cb, self.scan.name, "write",
                              frame.index, self._completed, len(self.scan))
                        if self.clear_frame_images:
                            frame.image = None
                            frame.background = None
                            _clear_source_frame_image(self.source, frame.index)
                    except BaseException as exc:
                        self._failure = self._failure or exc
            finally:
                self._semaphore.release()
                self._write_queue.task_done()

    def drain(self, timeout: float | None = None, poll: float = 0.05) -> bool:
        """Block until every SUBMITTED frame has been written, WITHOUT closing
        the session (non-terminal — unlike :meth:`finish`).  Returns ``True`` if
        the writer fully drained, ``False`` if it timed out / was cancelled.

        For ``execution="streaming"`` this waits on the writer queue: the writer
        thread calls ``task_done()`` for every item (both the per-frame
        ``finally`` and the sentinel branch in :meth:`_writer_loop`), so this
        returns once the in-flight window has fully drained and the sink has
        written each completed frame — yet the writer thread keeps idling on
        ``_write_queue.get()`` (no sentinel is pushed), so the session stays OPEN
        and :meth:`submit` works unchanged afterward.

        This is what lets a caller quiesce the writer at a frame boundary (e.g. a
        GUI Pause: drain, flush the sink to disk, browse, then resume submitting)
        without the terminal teardown :meth:`finish` performs.  A per-frame
        failure recorded during the drain still surfaces at the eventual
        :meth:`finish` (fail-loud preserved).  No-op (returns ``True``) for
        chunked execution and before the stream starts.

        ``timeout`` BOUNDS the wait: a single in-flight worker that never returns
        (a stalled detector/NFS read or a runaway pyFAI call — a running
        ``ThreadPoolExecutor`` future can't be cancelled) would otherwise hang the
        caller forever.  With ``timeout=None`` (the default, used by terminal
        teardown) this is the original unbounded ``join()``.  With a timeout it
        polls ``unfinished_tasks`` under the queue's own condition and ALSO bails
        early once ``cancel_token`` trips (Stop/close), so a paused-then-stopped
        run can break out promptly instead of stranding the thread.
        """
        if not (self.execution == "streaming" and self._stream_started
                and self._write_queue is not None):
            return True
        q = self._write_queue
        if timeout is None:
            q.join()
            return True
        deadline = time.monotonic() + timeout
        while True:
            with q.all_tasks_done:          # the same condition join() waits on
                if q.unfinished_tasks == 0:
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self.cancel_token.cancelled:
                    return False
                q.all_tasks_done.wait(min(poll, remaining))

    def finish(self, raise_on_failure: bool = True,
               join_timeout: float | None = None) -> ReductionResult:
        """Drain, flush the sink, and return the :class:`ReductionResult`.

        By default this is FAIL-LOUD: if any frame reduction or sink write
        failed (``self._failure``), ``finish`` re-raises that original exception
        (preserving its traceback) AFTER the result is built and the sink is
        aborted/closed — so a data-writing run can never silently report success
        (the failure info is still available on ``self.result`` / the return
        value of a ``raise_on_failure=False`` call).  Pass
        ``raise_on_failure=False`` to inspect ``result.failed`` and tolerate
        partial failures instead (e.g. cleanup paths that are already handling an
        exception, or freeze-only sessions with no write sink).

        ``join_timeout`` bounds the writer-thread join for streaming sessions.
        ``None`` (default) is unbounded — safe for normal runs where workers
        complete promptly.  GUI sessions should pass a finite timeout (e.g. 60 s)
        so a stalled NFS/pyFAI worker can't wedge Stop/close indefinitely; if the
        join times out, the cancel token is tripped, the result is marked failed,
        and a ``TimeoutError`` is recorded as the failure."""
        if self._finished and self.result is not None:
            # Idempotent: a re-call after a raised first finish() returns the
            # preserved (possibly failed) result rather than re-raising.
            return self.result

        _writer_timed_out = False
        if self.execution == "streaming" and self._stream_started:
            # No more submits: tell the writer to drain the queue and exit, then
            # join it so only completed frames are flushed (never a torn frame).
            self._write_queue.put(_STREAM_SENTINEL)
            if self._writer_thread is not None:
                self._writer_thread.join(timeout=join_timeout)
                if self._writer_thread.is_alive():
                    # Writer is still alive after the timeout (a stalled worker
                    # held the future.result() call and the sentinel hasn't been
                    # processed).  Cancel any remaining in-flight work via the
                    # cancel token so the worker unblocks at its next check, and
                    # flag the result as failed so the caller gets a loud error
                    # rather than a silent hang.
                    self.cancel_token.cancel()
                    self._cancelled = True
                    if self._failure is None:
                        self._failure = TimeoutError(
                            f"ReductionSession.finish(): writer thread did not "
                            f"exit within {join_timeout}s; a worker may be "
                            f"stalled (stalled NFS read or runaway pyFAI call). "
                            f"Session result is incomplete."
                        )
                    warnings.warn(
                        f"ReductionSession.finish(): writer join timed out after "
                        f"{join_timeout}s; session result is incomplete.",
                        RuntimeWarning, stacklevel=2,
                    )
                    # Do NOT null the thread handle — it is still alive and
                    # process-scoped; let the interpreter reap it on exit.
                    # Signal fast-exit to the cleanup below: the pool cannot be
                    # shut down with wait=True because its futures are still live.
                    _writer_timed_out = True
                else:
                    self._writer_thread = None

        self.result = ReductionResult(
            scan_name=self.scan.name,
            frames=self._products,
            n_processed=self._completed,
            cancelled=self._cancelled or self.cancel_token.cancelled,
            output_path=self._output_path,
            failed=self._failure is not None,
            error=None if self._failure is None else str(self._failure),
        )
        try:
            if self._started:
                if _writer_timed_out:
                    # T0-5: the writer thread is STILL ALIVE (a stalled worker
                    # holds it in future.result()) and may yet call
                    # sink.write() on its h5 handle.  Tearing the sink down
                    # here would (a) race h5.close() against an in-flight
                    # write — h5py handles are not thread-safe — and (b) in
                    # atomic-mode NexusSink, abort() used to unlink the tmp
                    # file holding every frame written so far.  Leave the sink
                    # untouched; the recorded TimeoutError (raised below) is
                    # the loud signal, and the on-disk file keeps whatever was
                    # written.
                    # Name the actual on-disk location: an atomic-mode
                    # NexusSink writes into a hidden tmp file that never gets
                    # promoted on this path — without naming it, "left as-is"
                    # reads as total loss for a new output file.
                    data_loc = (getattr(self._sink, "_tmp_path", None)
                                or getattr(self._sink, "_active_path", None)
                                or getattr(self._sink, "path", None))
                    where = (f" Frames written so far are in {data_loc}"
                             " (un-finalized)." if data_loc else "")
                    warnings.warn(
                        "ReductionSession.finish(): writer join timed out — "
                        "skipping sink finish/abort (writer may still be "
                        f"writing); output for {self.scan.name!r} is left "
                        f"un-finalized.{where}",
                        RuntimeWarning, stacklevel=2,
                    )
                elif self._failure is None:
                    self._sink.finish(self.result)
                else:
                    abort = getattr(self._sink, "abort", None)
                    if callable(abort):
                        abort(self.result)
                    else:
                        self._sink.finish(self.result)
        finally:
            if self._owns_worker and self._worker is not None:
                # If the writer join timed out, the pool has stalled futures —
                # shut down without waiting (don't re-hang here).
                wait_for_pool = not _writer_timed_out
                self._worker.shutdown(wait=wait_for_pool, cancel_futures=True)
            self._worker = None
            self._finished = True

        _emit(
            self.progress_cb,
            self.scan.name,
            "finish",
            None,
            self._completed,
            len(self.scan),
        )
        # Fail-loud (rail 1+3): re-raise the ORIGINAL reduction/sink-write
        # exception so the real traceback survives (no generic wrapper).  The
        # result is already stored on self.result (rail 2) for retrieval.
        if raise_on_failure and self._failure is not None:
            raise self._failure
        return self.result

    def _apply_gi_freeze(self, freeze_policy: str) -> None:
        if self._gi_freeze_applied or self.plan.gi is None:
            return
        self.plan = _apply_gi_freeze_policy(
            self.plan,
            self.scan,
            freeze_policy=freeze_policy,
            fi=None,
            initial_incident_angle=self._initial_incident_angle,
        )
        self._gi_freeze_applied = True

    def _normalize_process_input(
        self,
        frames_or_chunk: Iterable[Frame] | tuple[Any, Iterable[int]],
        images: Iterable[np.ndarray | None] | None,
    ) -> tuple[list[Frame], list[np.ndarray | None]]:
        if isinstance(frames_or_chunk, tuple) and len(frames_or_chunk) == 2:
            chunk_images, labels = frames_or_chunk
            frame_by_index = {int(frame.index): frame for frame in self.scan.frames}
            chunk = [frame_by_index[int(label)] for label in labels]
            return chunk, _chunk_images_as_list(chunk_images, [int(label) for label in labels])

        chunk = list(frames_or_chunk)
        if images is None:
            return chunk, [None] * len(chunk)
        chunk_images = [None if image is None else np.asarray(image) for image in images]
        if len(chunk_images) != len(chunk):
            raise ValueError(
                f"got {len(chunk_images)} images for {len(chunk)} reduction frames"
            )
        return chunk, chunk_images

    def _register_process_frames(self, chunk: list[Frame]) -> None:
        """Keep the session scan inventory in sync with explicitly-fed chunks.

        GUI/live callers commonly open the session from the first available
        chunk, then feed later chunks as fresh ``Frame`` objects.  The reducer can
        compute those frames without registering them, but sinks, scan metadata,
        progress totals, and future replay/debug hooks need the session's scan to
        describe the whole run.  This is O(new frames) for ordered acquisition and
        only sorts when callers feed out-of-order labels.
        """

        if not chunk:
            return

        frames = self.scan.frames
        positions = self._scan_frame_positions
        last_index = int(frames[-1].index) if frames else None
        needs_sort = False

        for frame in chunk:
            idx = int(frame.index)
            pos = positions.get(idx)
            if pos is None:
                positions[idx] = len(frames)
                frames.append(frame)
                self.scan._frame_by_index[idx] = frame
                if last_index is not None and idx < last_index:
                    needs_sort = True
                last_index = idx
                continue

            # Replace the frame object for this label with the caller's latest
            # explicit frame.  xdart builds a fresh headless Frame for the chunk
            # it feeds, so identity equality is not expected even for the first
            # chunk used to open the session.
            frames[pos] = frame
            self.scan._frame_by_index[idx] = frame

        if needs_sort:
            frames.sort(key=lambda item: int(item.index))
            positions.clear()
            positions.update({int(frame.index): pos for pos, frame in enumerate(frames)})

    def _process_chunk(
        self,
        chunk: list[Frame],
        chunk_images: list[np.ndarray | None],
    ) -> None:
        if not chunk:
            return
        _emit(
            self.progress_cb,
            self.scan.name,
            "chunk",
            chunk[0].index,
            self._completed,
            len(self.scan),
        )
        pending: list[tuple[Frame, Any]] = []
        for frame, raw_image in zip(chunk, chunk_images):
            if self.cancel_token.cancelled:
                self._cancelled = True
                break
            _emit(self.progress_cb, self.scan.name, "load", frame.index, self._completed, len(self.scan))
            _emit(self.progress_cb, self.scan.name, "integrate", frame.index, self._completed, len(self.scan))
            if self._worker is None:
                try:
                    reduction = _reduce_frame(
                        frame,
                        raw_image,
                        self.plan,
                        self._integrators,
                        self._plan_masks,
                        cancel_token=self.cancel_token,
                    )
                except _ReductionCancelled:
                    self._cancelled = True
                    break
                pending.append((frame, reduction))
            else:
                pending.append((
                    frame,
                    self._worker.submit(
                        _reduce_frame,
                        frame,
                        raw_image,
                        self.plan,
                        self._integrators,
                        # PERF: share the session's persistent mask cache with
                        # the worker (ThreadPoolExecutor => shared memory) so the
                        # bool mask is expanded once per detector shape, not once
                        # per frame per worker.  Keyed by image shape; a dict set
                        # is atomic under the GIL and a concurrent first-write
                        # recomputes the identical array, so sharing is safe.
                        self._plan_masks,
                        self.cancel_token,
                    ),
                ))

        pos = -1
        try:
            for pos, (frame, reduction_or_future) in enumerate(pending):
                try:
                    reduction = (
                        reduction_or_future
                        if self._worker is None
                        else reduction_or_future.result()
                    )
                except _ReductionCancelled:
                    self._cancelled = True
                    _cancel_pending_futures(pending[pos + 1:], worker=self._worker)
                    break
                idx = int(frame.index)
                # Re-feeding an already-processed index (reintegrate / replace
                # re-feed) is a *replace*, not a new completion: overwrite the
                # product, re-emit to the sink as a replace where supported, and do
                # not double-count progress -- ``n_processed`` must never exceed the
                # number of distinct frames in the scan.
                replacing = idx in self._seen_idxs
                self._seen_idxs.add(idx)
                if self.retain_products:
                    self._products[idx] = reduction
                if replacing:
                    _emit_sink_replace(self._sink, frame, reduction)
                else:
                    self._sink.write(frame, reduction)
                    self._completed += 1
                _emit(self.progress_cb, self.scan.name, "write", frame.index, self._completed, len(self.scan))
                if self.clear_frame_images:
                    frame.image = None
                    frame.background = None
                    _clear_source_frame_image(self.source, frame.index)
                if self.cancel_token.cancelled:
                    self._cancelled = True
                    _cancel_pending_futures(pending[pos + 1:], worker=self._worker)
                    break
        except BaseException:
            # An arbitrary error (a worker raise out of future.result(), or a
            # sink.write failure) previously exited the loop WITHOUT cancelling
            # the tail futures or releasing the chunk's image refs -- a
            # persistent GUI session then held them until close.  Cancel and
            # release, then re-raise the original error.
            _cancel_pending_futures(pending[pos + 1:], worker=self._worker)
            if self.clear_frame_images:
                for _frame, _ in pending:
                    try:
                        _frame.image = None
                        _frame.background = None
                    except Exception:
                        pass
            raise


def run_reduction(
    plan: ReductionPlan,
    scan: Scan | FrameSource,
    sink: ReductionSink | Iterable[ReductionSink] | None = None,
    *,
    chunk_size: int = 1,
    clear_frame_images: bool = False,
    progress_cb: ProgressCallback | None = None,
    cancel_token: CancelToken | None = None,
    executor: Any | None = None,
    gi_freeze_mode: str | None = None,
    execution: str = "chunked",
    inflight_max: int | None = None,
    retain_products: bool | None = None,
) -> ReductionResult:
    """Run a headless reduction job over all frames in ``scan`` or a source.

    Parameters
    ----------
    plan
        Content of the reduction (what to integrate, mask, thresholds,
        optional :class:`GIMode`).
    scan
        Frames + scan-level context (PONI / integrator / motors), or any
        :class:`FrameSource` that can be materialized into one.
    sink
        Where to send per-frame :class:`FrameReduction`.  Defaults to
        an in-memory :class:`MemorySink`.
    chunk_size
        Frames per progress chunk.  Larger values amortise the
        ``"chunk"`` progress event over more frames but don't change
        the per-frame compute path.  Default 1.
    clear_frame_images
        Set each frame's cached image to ``None`` after writing to the
        sink.  Cheap memory bound for long lazy-loaded scans.
    progress_cb
        Called as ``cb(ReductionProgress)`` after every stage.
    cancel_token
        Polled per frame; cancellation stops at the next frame
        boundary (pyFAI doesn't yield mid-integration).
    executor
        Optional execution policy for per-frame work inside each chunk.  Pass
        an executor with ``submit()``, ``True`` for a default
        :class:`ThreadPoolExecutor`, or an integer worker count.  Sink writes
        remain ordered on the caller thread.
    gi_freeze_mode
        Optional grazing-incidence common-grid freeze policy.  ``"first_frame"``
        scouts the first frame; ``"scout_union"`` scouts first+last (or
        ``plan.extra["gi_freeze_scout_indices"]``) and freezes the missing
        output-axis ranges before the main reduction.  Explicit caller ranges
        are preserved.
    execution
        ``"chunked"`` (default) integrates frames in chunks; ``"streaming"``
        submits each frame to a bounded in-flight window drained by one writer
        thread (out-of-order completion, single-writer sink) — the same engine
        xdart's GUI uses by default, exposed here so notebook/headless callers
        get it without hand-driving :class:`ReductionSession`.
    inflight_max
        Streaming only: max frames in flight (defaults to ``2 × workers``).
        Bounds peak memory for a fast source feeding a slower reduce.
    retain_products
        Whether ``result.frames`` accumulates every :class:`FrameReduction`
        (full 2D arrays — ~14 GB on a 10k-frame 2D scan).  ``None`` (default)
        auto-selects: ``False`` for STREAMING runs into a durable sink (the
        data lands on disk per frame; read it back from the file), ``True``
        otherwise (MemorySink / no sink / chunked — ``result.frames`` is the
        only way to get results back).  Pass an explicit bool to override.
    """
    if retain_products is None:
        retain_products = not (
            execution == "streaming"
            and sink is not None
            and not isinstance(sink, MemorySink)
        )
    with ReductionSession(
        plan,
        scan,
        sink,
        chunk_size=chunk_size,
        clear_frame_images=clear_frame_images,
        progress_cb=progress_cb,
        cancel_token=cancel_token,
        executor=executor,
        gi_freeze_mode=gi_freeze_mode,
        execution=execution,
        inflight_max=inflight_max,
        retain_products=retain_products,
    ) as session:
        if execution == "streaming":
            # Streaming drains via submit() (process() is rejected); feed every
            # frame, then finish() joins the writer and flushes.
            for frame in session.scan:
                if session.cancel_token.cancelled:
                    break
                session.submit(frame)
        else:
            session.process()
        return session.finish()


def _normalize_gi_freeze_mode(mode: str | None) -> str | None:
    if mode is None:
        return None
    value = str(mode).strip().lower()
    if value in {"", "none", "pre_frozen", "pre-frozen"}:
        return None
    if value not in {"first_frame", "first-frame", "scout_union", "scout-union"}:
        raise ValueError(
            "gi_freeze_mode must be None, 'first_frame', or 'scout_union'; "
            f"got {mode!r}"
        )
    return value.replace("-", "_")


def _apply_gi_freeze_policy(
    plan: ReductionPlan,
    scan: Scan,
    *,
    freeze_policy: str | None,
    fi: Any,
    initial_incident_angle: float | None,
) -> ReductionPlan:
    """Return a copy of *plan* with missing GI output ranges frozen.

    The pre-pass is intentionally bounded: live mode can use ``first_frame``,
    while batch mode can use ``scout_union`` over first+last or an explicit
    ``plan.extra["gi_freeze_scout_indices"]`` iterable.  Existing explicit
    ranges win; the freeze only fills missing output-axis ranges so notebook
    callers can still choose their own grids exactly.
    """

    if freeze_policy is None or plan.gi is None or not scan.frames:
        return plan

    needs_1d = _gi_1d_freeze_key(plan)
    needs_2d = _gi_2d_freeze_keys(plan)
    if needs_1d is None and not needs_2d:
        return plan

    scout_indices = _gi_freeze_scout_indices(plan, scan, freeze_policy)
    if not scout_indices:
        return plan

    scout_integrators = _ReductionIntegratorProvider(
        scan=scan,
        plan=plan,
        ai=None,
        fi=fi,
        initial_incident_angle=initial_incident_angle,
    )
    scout_results_1d: list[IntegrationResult1D] = []
    scout_results_2d: list[IntegrationResult2D] = []
    masks: dict[tuple[int, int], np.ndarray | None] = {}
    for idx in scout_indices:
        frame = scan._frame_by_index[int(idx)]
        was_empty = frame.image is None
        reduction = _reduce_frame(frame, None, plan, scout_integrators, masks)
        if reduction.result_1d is not None:
            scout_results_1d.append(reduction.result_1d)
        if reduction.result_2d is not None:
            if _is_all_dummy_2d(reduction.result_2d):
                continue
            scout_results_2d.append(reduction.result_2d)
        if was_empty:
            frame.image = None

    out = plan
    if needs_1d is not None and scout_results_1d:
        from ssrl_xrd_tools.integrate.gid import freeze_common_axis

        key, rng = freeze_common_axis(
            scout_results_1d,
            gi_mode_1d=out.gi.mode_1d.value,
        )
        if rng is not None and key == needs_1d:
            out = _replace_integration_1d_range(out, key, rng)
        elif key == needs_1d:
            # Mirror the 2D branch's fail-loud: a degenerate scout (blank /
            # all-masked / collapsed span -> rng None) silently skipped the
            # 1D freeze, leaving per-frame auto axes that the writer's
            # uniform-axes validator rejects MID-RUN, frames already on disk
            # and far from the root cause.
            raise GIFreezeError(
                "GI 1D freeze scout produced a degenerate axis range; "
                "check the incident angle / mask / threshold."
            )
    elif needs_1d is not None:
        raise GIFreezeError(
            "GI 1D freeze scout produced no usable 1D results; "
            "check the incident angle / incidence motor."
        )
    if needs_2d and scout_results_2d:
        from ssrl_xrd_tools.integrate.gid import freeze_common_axes_2d

        ranges = freeze_common_axes_2d(
            scout_results_2d,
            gi_mode_2d=out.gi.mode_2d.value,
        )
        out = _replace_integration_2d_ranges(
            out,
            {
                key: value
                for key, value in ranges.items()
                if key in needs_2d
            },
        )
    elif needs_2d:
        raise GIFreezeError(
            "GI 2D freeze scout produced no non-dummy 2D results; "
            "check the incident angle / incidence motor."
        )
    return out


def _is_all_dummy_2d(result: IntegrationResult2D, *, dummy: float = -1.0) -> bool:
    intensity = getattr(result, "intensity", None)
    if intensity is None:
        return False
    arr = np.asarray(intensity, dtype=float)
    if arr.size == 0:
        return True
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return True
    return bool(np.all(finite <= dummy))


def _gi_freeze_scout_indices(
    plan: ReductionPlan,
    scan: Scan,
    freeze_policy: str,
) -> list[int]:
    extra = getattr(plan, "extra", None) or {}
    explicit = extra.get("gi_freeze_scout_indices") if isinstance(extra, dict) else None
    if explicit is not None:
        allowed = set(scan.frame_indices)
        out = []
        for value in explicit:
            idx = int(value)
            if idx not in allowed:
                raise ValueError(f"GI freeze scout frame {idx} is not in scan {scan.name!r}")
            if idx not in out:
                out.append(idx)
        return out
    if freeze_policy == "first_frame" or len(scan.frames) == 1:
        return [int(scan.frames[0].index)]
    return [int(scan.frames[0].index), int(scan.frames[-1].index)]


def _gi_1d_freeze_key(plan: ReductionPlan) -> str | None:
    if plan.gi is None or plan.integration_1d is None:
        return None
    from ssrl_xrd_tools.integrate.gid import gi_1d_output_axis_key

    key = gi_1d_output_axis_key(plan.gi.mode_1d.value)
    return key if getattr(plan.integration_1d, key) is None else None


def _gi_2d_freeze_keys(plan: ReductionPlan) -> set[str]:
    if plan.gi is None or plan.integration_2d is None:
        return set()
    p2d = plan.integration_2d
    if plan.gi.mode_2d is GI2DMode.QIP_QOOP:
        out: set[str] = set()
        if p2d.extra.get("x_range") is None and p2d.radial_range is None:
            out.add("x_range")
        if p2d.extra.get("y_range") is None and p2d.azimuth_range is None:
            out.add("y_range")
        return out
    out = set()
    if p2d.radial_range is None:
        out.add("radial_range")
    if p2d.azimuth_range is None:
        out.add("azimuth_range")
    return out


def _replace_integration_1d_range(
    plan: ReductionPlan,
    key: str,
    value: tuple[float, float],
) -> ReductionPlan:
    if plan.integration_1d is None:
        return plan
    if key == "radial_range":
        p1d = replace(plan.integration_1d, radial_range=tuple(map(float, value)))
    elif key == "azimuth_range":
        p1d = replace(plan.integration_1d, azimuth_range=tuple(map(float, value)))
    else:
        return plan
    return replace(plan, integration_1d=p1d)


def _replace_integration_2d_ranges(
    plan: ReductionPlan,
    ranges: dict[str, tuple[float, float]],
) -> ReductionPlan:
    if not ranges or plan.integration_2d is None:
        return plan
    p2d = plan.integration_2d
    extra = dict(p2d.extra)
    kwargs: dict[str, Any] = {}
    for key, value in ranges.items():
        frozen = tuple(map(float, value))
        if key == "x_range":
            extra["x_range"] = frozen
        elif key == "y_range":
            extra["y_range"] = frozen
        elif key == "radial_range":
            kwargs["radial_range"] = frozen
        elif key == "azimuth_range":
            kwargs["azimuth_range"] = frozen
    p2d = replace(p2d, extra=extra, **kwargs)
    return replace(plan, integration_2d=p2d)


class _ReductionIntegratorProvider:
    """Per-thread integrator cache for executor-backed reductions."""

    def __init__(
        self,
        *,
        scan: Scan,
        plan: ReductionPlan,
        ai: Any,
        fi: Any,
        initial_incident_angle: float | None,
    ) -> None:
        self.scan = scan
        self.plan = plan
        self.ai = ai
        self.fi = fi
        self.initial_incident_angle = initial_incident_angle
        self._local = threading.local()
        self._owner_thread = threading.get_ident()

    def standard(self) -> Any:
        if self.scan.poni is None:
            return self.ai
        if threading.get_ident() == self._owner_thread and self.ai is not None:
            return self.ai
        ai = getattr(self._local, "ai", None)
        if ai is None:
            ai = poni_to_integrator(self.scan.poni)
            self._local.ai = ai
        return ai

    def fiber(self) -> Any:
        if self.scan.poni is None:
            return self.fi
        if threading.get_ident() == self._owner_thread and self.fi is not None:
            return self.fi
        fi = getattr(self._local, "fi", None)
        if fi is None:
            gi = self.plan.gi
            if gi is None:
                return None
            fi = poni_to_fiber_integrator(
                self.scan.poni,
                incident_angle=float(self.initial_incident_angle or 0.0),
                tilt_angle=float(gi.tilt_angle),
                sample_orientation=int(gi.sample_orientation),
            )
            self._local.fi = fi
        return fi


def _coerce_executor(executor: Any | None):
    if executor is None or executor is False:
        return None, False
    if executor is True:
        return ThreadPoolExecutor(), True
    if isinstance(executor, int):
        if executor <= 0:
            raise ValueError(f"executor worker count must be > 0; got {executor}")
        return ThreadPoolExecutor(max_workers=executor), True
    if hasattr(executor, "submit"):
        return executor, False
    raise TypeError(
        "executor must be None, False, True, a positive worker count, "
        "or an object with submit()"
    )


class _ReductionCancelled(Exception):
    """Internal sentinel used to stop queued worker tasks without failure."""


# Pushed onto a streaming session's write queue by ``finish`` to tell the
# single writer/consumer thread to drain and exit.
_STREAM_SENTINEL = object()


def _cancel_pending_futures(pending: list[tuple[Frame, Any]], *, worker: Any | None) -> None:
    if worker is None:
        return
    for _frame, candidate in pending:
        cancel = getattr(candidate, "cancel", None)
        if callable(cancel):
            cancel()


def _reduce_frame(
    frame: Frame,
    raw_image: np.ndarray | None,
    plan: ReductionPlan,
    integrators: _ReductionIntegratorProvider,
    plan_masks: dict[tuple[int, int], np.ndarray | None],
    cancel_token: CancelToken | None = None,
) -> FrameReduction:
    if cancel_token is not None and cancel_token.cancelled:
        raise _ReductionCancelled
    if raw_image is not None:
        frame.image = np.asarray(raw_image)
    image = np.asarray(frame.load_image(), dtype=float)
    if cancel_token is not None and cancel_token.cancelled:
        raise _ReductionCancelled
    if image.ndim != 2:
        raise ValueError(f"Frame {frame.index} image must be 2D; got shape {image.shape}")
    _validate_frame_inputs(frame, image.shape)
    image = _apply_thresholds(image, plan)
    image = _subtract_background(image, frame.background)
    plan_mask = _cached_mask_for_shape(
        plan.mask,
        image.shape,
        "ReductionPlan.mask",
        plan_masks,
    )
    mask = _combined_mask(plan_mask, frame.mask, image.shape)

    if plan.gi is not None:
        fi = integrators.fiber()
        incident_angle = _resolve_gi_incident_angle(frame, plan.gi)
        r1d = (
            _run_gi_1d(
                image,
                fi,
                plan.integration_1d,
                plan.gi,
                mask=mask,
                incident_angle=incident_angle,
                normalization_factor=_normalization_for(frame, plan.integration_1d),
            )
            if plan.integration_1d is not None else None
        )
        r2d = (
            _run_gi_2d(
                image,
                fi,
                plan.integration_2d,
                plan.gi,
                mask=mask,
                incident_angle=incident_angle,
                normalization_factor=_normalization_for(frame, plan.integration_2d),
            )
            if plan.integration_2d is not None else None
        )
    else:
        ai = integrators.standard()
        r1d = (
            integrate_1d(
                image,
                ai,
                npt=plan.integration_1d.npt,
                unit=plan.integration_1d.unit,
                method=plan.integration_1d.method,
                mask=mask,
                radial_range=plan.integration_1d.radial_range,
                azimuth_range=plan.integration_1d.azimuth_range,
                error_model=plan.integration_1d.error_model,
                polarization_factor=plan.integration_1d.polarization_factor,
                normalization_factor=_normalization_for(frame, plan.integration_1d),
                **plan.integration_1d.extra,
            )
            if plan.integration_1d is not None else None
        )
        r2d = (
            integrate_2d(
                image,
                ai,
                npt_rad=plan.integration_2d.npt_rad,
                npt_azim=plan.integration_2d.npt_azim,
                unit=plan.integration_2d.unit,
                method=plan.integration_2d.method,
                mask=mask,
                radial_range=plan.integration_2d.radial_range,
                azimuth_range=_integration_azimuth_range(plan.integration_2d),
                error_model=plan.integration_2d.error_model,
                polarization_factor=plan.integration_2d.polarization_factor,
                normalization_factor=_normalization_for(frame, plan.integration_2d),
                **plan.integration_2d.extra,
            )
            if plan.integration_2d is not None else None
        )
        if r2d is not None and plan.integration_2d.azimuth_offset:
            r2d.azimuthal = r2d.azimuthal + float(plan.integration_2d.azimuth_offset)

    return FrameReduction(
        frame_index=frame.index,
        result_1d=r1d,
        result_2d=r2d,
        metadata=dict(frame.metadata),
    )


def _apply_thresholds(image: np.ndarray, plan: ReductionPlan) -> np.ndarray:
    if plan.threshold_min is None and plan.threshold_max is None:
        return image
    out = np.array(image, dtype=float, copy=True)
    bad = np.zeros(out.shape, dtype=bool)
    if plan.threshold_min is not None:
        bad |= out < float(plan.threshold_min)
    if plan.threshold_max is not None:
        bad |= out > float(plan.threshold_max)
    out[bad] = np.nan
    return out


def _subtract_background(
    image: np.ndarray,
    background: np.ndarray | float | None,
) -> np.ndarray:
    if background is None:
        return image
    bg = np.asarray(background, dtype=float)
    if bg.ndim > 0 and bg.shape != image.shape:
        raise ValueError(
            f"background shape {bg.shape} does not match image shape {image.shape}"
        )
    return image - bg


def _integration_azimuth_range(
    plan: Integration2DPlan,
) -> tuple[float, float] | None:
    if plan.azimuth_range is None:
        return None
    if not plan.azimuth_offset:
        return plan.azimuth_range
    lo, hi = plan.azimuth_range
    offset = float(plan.azimuth_offset)
    return lo - offset, hi - offset


def _coerce_gi_1d_mode(mode: GI1DMode | str) -> GI1DMode:
    if isinstance(mode, GI1DMode):
        return mode
    aliases = {
        "qip": GI1DMode.Q_IP,
        "q_ip": GI1DMode.Q_IP,
        "qoop": GI1DMode.Q_OOP,
        "q_oop": GI1DMode.Q_OOP,
        "qtot": GI1DMode.Q_TOTAL,
        "q_total": GI1DMode.Q_TOTAL,
        "qtotal": GI1DMode.Q_TOTAL,
        "polar": GI1DMode.Q_TOTAL,
        "exit": GI1DMode.EXIT_ANGLE,
        "exit_angle": GI1DMode.EXIT_ANGLE,
    }
    key = str(mode).strip().lower()
    try:
        return aliases[key]
    except KeyError as exc:
        allowed = ", ".join(m.value for m in GI1DMode)
        raise ValueError(f"unknown GI 1D mode {mode!r}; expected one of {allowed}") from exc


def _coerce_gi_2d_mode(mode: GI2DMode | str) -> GI2DMode:
    if isinstance(mode, GI2DMode):
        return mode
    aliases = {
        "qip_qoop": GI2DMode.QIP_QOOP,
        "qip-qoop": GI2DMode.QIP_QOOP,
        "gi2d": GI2DMode.QIP_QOOP,
        "q_chi": GI2DMode.Q_CHI,
        "q-chi": GI2DMode.Q_CHI,
        "polar": GI2DMode.Q_CHI,
        "exit": GI2DMode.EXIT_ANGLES,
        "exit_angle": GI2DMode.EXIT_ANGLES,
        "exit_angles": GI2DMode.EXIT_ANGLES,
    }
    key = str(mode).strip().lower()
    try:
        return aliases[key]
    except KeyError as exc:
        allowed = ", ".join(m.value for m in GI2DMode)
        raise ValueError(f"unknown GI 2D mode {mode!r}; expected one of {allowed}") from exc


def _resolve_gi_incident_angle(frame: Frame | None, gi: GIMode) -> float:
    if gi.incident_angle is not None:
        return float(gi.incident_angle)
    if frame is not None and frame.geometry is not None:
        if frame.geometry.incident_angle is not None:
            return float(frame.geometry.incident_angle)
    if frame is not None and gi.incidence_motor:
        value = _metadata_get_case_insensitive(frame.metadata, gi.incidence_motor)
        try:
            angle = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Frame {frame.index} cannot resolve GI incident angle from "
                f"metadata motor {gi.incidence_motor!r}."
            ) from exc
        if np.isfinite(angle):
            return angle
    detail = (
        f"Frame {frame.index} " if frame is not None else ""
    )
    raise ValueError(
        detail
        + "GI reduction requires GIMode.incident_angle, "
        "Frame.geometry.incident_angle, or GIMode.incidence_motor metadata."
    )


def _gi_plan_extra(
    plan: Integration1DPlan | Integration2DPlan,
    normalization_factor: float | None,
) -> dict[str, Any]:
    extra = dict(plan.extra)
    if plan.error_model is not None:
        extra.setdefault("error_model", plan.error_model)
    if plan.polarization_factor is not None:
        extra.setdefault("polarization_factor", plan.polarization_factor)
    if normalization_factor is not None:
        extra.setdefault("normalization_factor", normalization_factor)
    return extra


def _run_gi_1d(
    image: np.ndarray,
    fi: Any,
    plan: Integration1DPlan,
    gi: GIMode,
    *,
    mask: np.ndarray | None,
    incident_angle: float,
    normalization_factor: float | None,
) -> IntegrationResult1D:
    extra = _gi_plan_extra(plan, normalization_factor)
    npt_oop = extra.pop("npt_oop", gi.npt_oop if gi.npt_oop is not None else plan.npt)
    common = dict(
        npt=plan.npt,
        method=gi.method,
        mask=mask,
        radial_range=plan.radial_range,
        azimuth_range=plan.azimuth_range,
        incident_angle=incident_angle,
        tilt_angle=gi.tilt_angle,
        sample_orientation=gi.sample_orientation,
    )
    if gi.mode_1d is GI1DMode.Q_IP:
        return integrate_gi_1d(
            image,
            fi,
            unit="qip_A^-1",
            npt_oop=npt_oop,
            vertical_integration=False,
            **common,
            **extra,
        )
    if gi.mode_1d is GI1DMode.Q_OOP:
        return integrate_gi_1d(
            image,
            fi,
            unit="qoop_A^-1",
            npt_oop=npt_oop,
            vertical_integration=True,
            **common,
            **extra,
        )
    if gi.mode_1d is GI1DMode.EXIT_ANGLE:
        return integrate_gi_exitangles_1d(
            image,
            fi,
            **common,
            **extra,
        )
    return integrate_gi_polar_1d(
        image,
        fi,
        unit=plan.unit,
        **common,
        **extra,
    )


def _run_gi_2d(
    image: np.ndarray,
    fi: Any,
    plan: Integration2DPlan,
    gi: GIMode,
    *,
    mask: np.ndarray | None,
    incident_angle: float,
    normalization_factor: float | None,
) -> IntegrationResult2D:
    extra = _gi_plan_extra(plan, normalization_factor)
    # The qip/qoop output ranges ride in plan.extra as x_range/y_range; they
    # are only meaningful for the QIP_QOOP transform.  Pop them BEFORE
    # branching so they never leak into pyFAI's polar/exit-angle calls as
    # unknown kwargs (pyFAI warns 'wrong or deprecated' and IGNORES them).
    x_range = extra.pop("x_range", plan.radial_range)
    y_range = extra.pop("y_range", plan.azimuth_range)
    common = dict(
        npt_rad=plan.npt_rad,
        npt_azim=plan.npt_azim,
        method=gi.method,
        mask=mask,
        incident_angle=incident_angle,
        tilt_angle=gi.tilt_angle,
        sample_orientation=gi.sample_orientation,
    )
    # GI ignores plan.azimuth_offset (Vivek, Jun 10): the chi offset is a
    # TRANSMISSION display convention (rotate the cake's chi origin).  In GI
    # the requested window goes to FiberIntegrator's polar/exit-angle/q-space
    # grids directly -- shifting it by the transmission offset displaced the
    # integrated wedge by 90 deg (GUI default) and, for qip_qoop, applied a
    # chi ANGLE offset to a q-space range.
    if gi.mode_2d is GI2DMode.Q_CHI:
        return integrate_gi_polar(
            image,
            fi,
            unit=plan.unit,
            radial_range=plan.radial_range,
            azimuth_range=plan.azimuth_range,
            **common,
            **extra,
        )
    if gi.mode_2d is GI2DMode.EXIT_ANGLES:
        return integrate_gi_exitangles(
            image,
            fi,
            unit=plan.unit,
            radial_range=plan.radial_range,
            azimuth_range=plan.azimuth_range,
            **common,
            **extra,
        )
    return integrate_gi_2d(
        image,
        fi,
        unit=_qip_qoop_unit(plan.unit),
        radial_range=x_range,
        azimuth_range=y_range,
        **common,
        **extra,
    )


def _qip_qoop_unit(unit: str | None) -> str:
    """Return a valid in-plane FiberIntegrator unit for qip/qoop maps.

    GUI state can legitimately carry a stale standard-AI unit such as
    ``q_A^-1`` when a user switches into GI qip/qoop mode.  Treat that as an
    unspecified GI unit and fall back to the FiberIntegrator default instead
    of letting pyFAI fail deep in unit parsing.
    """
    text = str(unit or "").strip()
    if text.startswith("qip_"):
        return text
    return "qip_A^-1"


def _cached_mask_for_shape(
    mask: np.ndarray | MaskSpec | None,
    image_shape: tuple[int, int],
    name: str,
    cache: dict[tuple[int, int], np.ndarray | None],
) -> np.ndarray | None:
    if mask is None:
        return None
    if image_shape not in cache:
        cache[image_shape] = _as_bool_mask(mask, name, image_shape=image_shape)
    return cache[image_shape]


def _combined_mask(
    plan_mask: np.ndarray | None,
    frame_mask: np.ndarray | MaskSpec | None,
    image_shape: tuple[int, int],
) -> np.ndarray | None:
    frame_mask = _as_bool_mask(frame_mask, "Frame.mask", image_shape=image_shape)
    if plan_mask is not None and plan_mask.shape != image_shape:
        raise ValueError(
            f"ReductionPlan.mask shape {plan_mask.shape} does not match "
            f"image shape {image_shape}"
        )
    if frame_mask is not None and frame_mask.shape != image_shape:
        raise ValueError(
            f"Frame.mask shape {frame_mask.shape} does not match image shape {image_shape}"
        )
    if plan_mask is None:
        return frame_mask
    if frame_mask is None:
        return plan_mask
    return plan_mask | frame_mask


# S8: monitor keys already warned about (once per key per process) — a bad
# monitor means frames are written UN-normalized, which must not be silent.
_warned_monitor_keys: set[str] = set()


def _normalization_for(
    frame: Frame,
    plan: Integration1DPlan | Integration2DPlan,
) -> float | None:
    if frame.normalization_factor is not None:
        return float(frame.normalization_factor)
    if plan.monitor_key is not None:
        key = plan.monitor_key
        value = frame.metadata.get(key)
        if value is None:
            value = frame.metadata.get(key.upper())
        if value is None:
            value = frame.metadata.get(key.lower())
        try:
            norm = float(value)
        except (TypeError, ValueError):
            norm = None
        if norm is not None and np.isfinite(norm) and norm != 0:
            return norm
        # S8: the monitor was configured but unusable — the frame is about to
        # be written UN-normalized.  Warn once per monitor key (not per frame:
        # a dead monitor on a 10k-frame scan must not emit 10k warnings).
        if key not in _warned_monitor_keys:
            _warned_monitor_keys.add(key)
            warnings.warn(
                f"monitor {key!r} is missing/zero/non-finite on frame "
                f"{frame.index} (value={value!r}); affected frames are "
                f"written UN-normalized.  (Warned once per monitor key.)",
                RuntimeWarning, stacklevel=2,
            )
    return None


def _validate_frame_inputs(frame: Frame, image_shape: tuple[int, int]) -> None:
    if frame.background is not None:
        bg = np.asarray(frame.background)
        if bg.ndim > 0 and bg.shape != image_shape:
            raise ValueError(
                f"Frame {frame.index} background shape {bg.shape} does not "
                f"match image shape {image_shape}"
            )
    if frame.mask is not None:
        mask = _as_bool_mask(frame.mask, "Frame.mask", image_shape=image_shape)
        if mask.shape != image_shape:
            raise ValueError(
                f"Frame {frame.index} mask shape {mask.shape} does not "
                f"match image shape {image_shape}"
            )


def _as_bool_mask(
    mask: np.ndarray | MaskSpec | None,
    name: str,
    *,
    image_shape: tuple[int, int] | None = None,
) -> np.ndarray | None:
    if mask is None:
        return None
    if isinstance(mask, MaskSpec):
        if image_shape is None:
            raise ValueError(f"{name} requires image shape to resolve MaskSpec.")
        return mask.to_bool(image_shape)
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D boolean mask; got shape {arr.shape}")
    return arr.astype(bool, copy=False)


def _metadata_get_case_insensitive(metadata: dict[str, Any], key: str) -> Any:
    if key in metadata:
        return metadata[key]
    key_lower = key.lower()
    for candidate, value in metadata.items():
        if str(candidate).lower() == key_lower:
            return value
    return None


def _emit(
    cb: ProgressCallback | None,
    scan_name: str,
    stage: str,
    frame_index: int | None,
    completed: int,
    total: int,
) -> None:
    if cb is not None:
        cb(ReductionProgress(scan_name, stage, frame_index, completed, total))


def _sink_path(sink: ReductionSink) -> Path | None:
    if isinstance(sink, CompositeSink):
        for child in sink.sinks:
            path = _sink_path(child)
            if path is not None:
                return path
    path = getattr(sink, "path", None)
    return path if isinstance(path, Path) else None


def _iter_reduction_chunks(
    source: Scan | FrameSource,
    scan: Scan,
    chunk_size: int,
) -> Iterator[tuple[list[Frame], list[np.ndarray | None]]]:
    """Yield scan frames paired with optional source-loaded image arrays.

    ``run_reduction`` materializes every source into a canonical ``Scan`` so
    geometry, metadata, and writer provenance are uniform.  The actual pixels
    should still come from ``FrameSource.iter_chunks`` when available: NeXus
    and Eiger sources can then hold one file handle and read a contiguous stack
    slice instead of reopening the file once per frame.
    """

    frame_by_index = {int(frame.index): frame for frame in scan.frames}
    if not isinstance(source, Scan):
        iter_chunks = getattr(source, "iter_chunks", None)
        if callable(iter_chunks):
            for images, labels in iter_chunks(chunk_size):
                frame_labels = [int(label) for label in labels]
                chunk_frames = []
                for label in frame_labels:
                    try:
                        chunk_frames.append(frame_by_index[label])
                    except KeyError as exc:
                        raise ValueError(
                            f"source yielded frame {label}, which is not present "
                            f"in materialized scan {scan.name!r}"
                        ) from exc
                yield chunk_frames, _chunk_images_as_list(images, frame_labels)
            return

    for start in range(0, len(scan.frames), chunk_size):
        chunk_frames = scan.frames[start:start + chunk_size]
        yield chunk_frames, [None] * len(chunk_frames)


def _chunk_images_as_list(images: Any, labels: list[int]) -> list[np.ndarray]:
    """Normalize a source chunk payload into one image per label."""

    if len(labels) == 1:
        arr = np.asarray(images)
        if arr.ndim == 2:
            return [arr]

    if isinstance(images, np.ndarray):
        if images.shape[0] != len(labels):
            raise ValueError(
                f"source chunk returned {images.shape[0]} images for "
                f"{len(labels)} frame labels"
            )
        return [np.asarray(images[i]) for i in range(len(labels))]

    out = [np.asarray(image) for image in images]
    if len(out) != len(labels):
        raise ValueError(
            f"source chunk returned {len(out)} images for {len(labels)} frame labels"
        )
    return out


def _clear_source_frame_image(source: Scan | FrameSource, index: int) -> None:
    """Best-effort hook for sources that own mutable image caches."""

    clear = getattr(source, "clear_frame_image", None)
    if callable(clear):
        clear(int(index))


def _coerce_sink(
    sink: ReductionSink | Iterable[ReductionSink] | None,
) -> ReductionSink:
    if sink is None:
        return MemorySink()
    if hasattr(sink, "begin") and hasattr(sink, "write") and hasattr(sink, "finish"):
        return sink  # type: ignore[return-value]
    sinks = tuple(sink)
    if not sinks:
        return MemorySink()
    if len(sinks) == 1:
        return sinks[0]
    return CompositeSink(sinks)


def _coerce_to_scan(source: Scan | FrameSource) -> Scan:
    if isinstance(source, Scan):
        return source
    to_scan = getattr(source, "to_scan", None)
    if callable(to_scan):
        kwargs = {}
        for name in (
            "poni",
            "integrator",
            "metadata",
            "energy",
            "wavelength",
            "motors",
            "output_path",
            "sample_name",
            "extra",
        ):
            if hasattr(source, name):
                kwargs[name] = getattr(source, name)
        return to_scan(**kwargs)
    if not hasattr(source, "frame_indices") or not hasattr(source, "load_frame"):
        raise TypeError(f"object does not implement FrameSource: {type(source)!r}")

    frames: list[Frame] = []
    for idx in source.frame_indices:
        metadata_for = getattr(source, "metadata_for", None)
        metadata = metadata_for(idx) if callable(metadata_for) else {}
        frames.append(
            Frame(
                index=int(idx),
                metadata=dict(metadata or {}),
                loader=lambda frame, src=source, label=int(idx): src.load_frame(label),
            )
        )
    return Scan(getattr(source, "name", "source"), frames)
