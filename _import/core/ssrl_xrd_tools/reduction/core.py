"""Scan/frame-oriented headless reduction primitives.

The intent of this module is to give GUIs and notebooks one small, stable
surface for common reduction jobs while keeping the numerical work in
``ssrl_xrd_tools.integrate``.  xdart should eventually build these objects
from its UI state and display the returned results, rather than owning
integration loops itself.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ssrl_xrd_tools.core.containers import (
    IntegrationResult1D,
    IntegrationResult2D,
    PONI,
)
from ssrl_xrd_tools.core.metadata import ScanMetadata
from ssrl_xrd_tools.integrate.calibration import (
    poni_to_fiber_integrator,
    poni_to_integrator,
)
from ssrl_xrd_tools.integrate.gid import integrate_gi_1d, integrate_gi_2d
from ssrl_xrd_tools.integrate.single import integrate_1d, integrate_2d
from ssrl_xrd_tools.io.image import read_image
from ssrl_xrd_tools.io.nexus import (
    open_nexus_image_stack,
    open_nexus_writer,
    write_nexus_frame,
)

ImageLoader = Callable[["Frame"], np.ndarray]
ProgressCallback = Callable[["ReductionProgress"], None]


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
    mask: np.ndarray | None = None
    normalization_factor: float | None = None
    loader: ImageLoader | None = None

    def __post_init__(self) -> None:
        if self.source_path is not None and not isinstance(self.source_path, Path):
            self.source_path = Path(self.source_path)

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
            with open_nexus_image_stack(path) as stack:
                self.image = np.asarray(stack[int(self.source_frame_index)])
        else:
            self.image = np.asarray(read_image(path))
        return self.image

    @property
    def label(self) -> str:
        return str(self.index)


@dataclass(slots=True)
class Scan:
    """Ordered set of frames with scan-level reduction context."""

    name: str
    frames: list[Frame]
    poni: PONI | None = None
    integrator: Any | None = None
    metadata: ScanMetadata | None = None
    energy: float | None = None
    wavelength: float | None = None
    motors: dict[str, np.ndarray] = field(default_factory=dict)
    output_path: Path | str | None = None
    sample_name: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.frames = sorted(list(self.frames), key=lambda f: f.index)
        self.motors = {k: np.asarray(v, dtype=float) for k, v in self.motors.items()}
        if self.output_path is not None and not isinstance(self.output_path, Path):
            self.output_path = Path(self.output_path)

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterable[Frame]:
        return iter(self.frames)

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
                f.metadata[key] for f in self.frames
                if key in f.metadata and f.metadata[key] is not None
            ]
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


@dataclass(slots=True)
class ReductionPlan:
    """Reduction settings shared by notebooks, CLIs, and xdart."""

    integrate_1d: bool = True
    integrate_2d: bool = False
    gi: bool = False
    npt_1d: int = 1000
    npt_rad_2d: int = 1000
    npt_azim_2d: int = 360
    unit: str = "q_A^-1"
    method_1d: str = "csr"
    method_2d: str = "csr"
    mask: np.ndarray | None = None
    radial_range: tuple[float, float] | None = None
    azimuth_range: tuple[float, float] | None = None
    error_model: str | None = None
    polarization_factor: float | None = None
    normalization_factors: Mapping[int, float] | None = None
    threshold_min: float | None = None
    threshold_max: float | None = None
    chunk_size: int = 1
    clear_frame_images: bool = False
    gi_incident_angle: float | None = None
    gi_tilt_angle: float = 0.0
    gi_sample_orientation: int = 1
    gi_method: str = "no"
    extra_1d: dict[str, Any] = field(default_factory=dict)
    extra_2d: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.integrate_1d and not self.integrate_2d:
            raise ValueError("ReductionPlan must enable integrate_1d or integrate_2d.")
        if self.npt_1d <= 0:
            raise ValueError(f"npt_1d must be > 0; got {self.npt_1d}")
        if self.npt_rad_2d <= 0 or self.npt_azim_2d <= 0:
            raise ValueError(
                "npt_rad_2d and npt_azim_2d must both be > 0; "
                f"got ({self.npt_rad_2d}, {self.npt_azim_2d})"
            )
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0; got {self.chunk_size}")
        if self.gi and self.gi_incident_angle is None:
            raise ValueError("gi=True requires gi_incident_angle.")


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
class NexusSink:
    """Frame-by-frame NeXus sink backed by ``ssrl_xrd_tools.io.nexus``."""

    path: Path | str
    entry: str = "entry"
    compression: str | None = "lzf"
    overwrite: bool = False
    swmr: bool = False
    _h5: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            self.path = Path(self.path)

    def begin(self, scan: Scan, plan: ReductionPlan) -> None:
        self._h5 = open_nexus_writer(
            self.path,
            metadata=scan.to_metadata(),
            entry=self.entry,
            compression=self.compression,
            swmr=self.swmr,
            overwrite=self.overwrite,
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
        self._h5.flush()

    def finish(self, result: ReductionResult) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None


def run_reduction(
    plan: ReductionPlan,
    scan: Scan,
    sink: ReductionSink | None = None,
    *,
    progress_cb: ProgressCallback | None = None,
    cancel_token: CancelToken | None = None,
) -> ReductionResult:
    """Run a headless reduction job over all frames in ``scan``."""
    if sink is None:
        sink = MemorySink()
    cancel_token = cancel_token or CancelToken()

    total = len(scan)
    output_path = _sink_path(sink) or (
        scan.output_path if isinstance(scan.output_path, Path) else None
    )
    products: dict[int, FrameReduction] = {}
    completed = 0
    cancelled = False

    ai = None
    fi = None
    if plan.gi:
        if scan.poni is None:
            raise ValueError("GI reduction requires scan.poni.")
        fi = poni_to_fiber_integrator(
            scan.poni,
            incident_angle=float(plan.gi_incident_angle),
            tilt_angle=float(plan.gi_tilt_angle),
            sample_orientation=int(plan.gi_sample_orientation),
        )
    else:
        ai = scan.integrator if scan.integrator is not None else None
        if ai is None:
            if scan.poni is None:
                raise ValueError("Reduction requires scan.integrator or scan.poni.")
            ai = poni_to_integrator(scan.poni)

    sink.begin(scan, plan)
    _emit(progress_cb, scan.name, "start", None, completed, total)
    try:
        for chunk_start in range(0, total, plan.chunk_size):
            chunk = scan.frames[chunk_start : chunk_start + plan.chunk_size]
            _emit(
                progress_cb,
                scan.name,
                "chunk",
                chunk[0].index if chunk else None,
                completed,
                total,
            )
            for frame in chunk:
                if cancel_token.cancelled:
                    cancelled = True
                    break
                _emit(progress_cb, scan.name, "load", frame.index, completed, total)
                image = np.asarray(frame.load_image(), dtype=float)
                image = _apply_thresholds(image, plan)
                mask = _combined_mask(plan.mask, frame.mask)
                norm = _normalization_for(frame, plan)

                _emit(progress_cb, scan.name, "integrate", frame.index, completed, total)
                if plan.gi:
                    r1d = (
                        integrate_gi_1d(
                            image,
                            fi,
                            npt=plan.npt_1d,
                            unit=plan.unit,
                            method=plan.gi_method,
                            mask=mask,
                            radial_range=plan.radial_range,
                            azimuth_range=plan.azimuth_range,
                            **plan.extra_1d,
                        )
                        if plan.integrate_1d else None
                    )
                    r2d = (
                        integrate_gi_2d(
                            image,
                            fi,
                            npt_rad=plan.npt_rad_2d,
                            npt_azim=plan.npt_azim_2d,
                            unit=plan.unit,
                            method=plan.gi_method,
                            mask=mask,
                            radial_range=plan.radial_range,
                            azimuth_range=plan.azimuth_range,
                            **plan.extra_2d,
                        )
                        if plan.integrate_2d else None
                    )
                else:
                    r1d = (
                        integrate_1d(
                            image,
                            ai,
                            npt=plan.npt_1d,
                            unit=plan.unit,
                            method=plan.method_1d,
                            mask=mask,
                            radial_range=plan.radial_range,
                            azimuth_range=plan.azimuth_range,
                            error_model=plan.error_model,
                            polarization_factor=plan.polarization_factor,
                            normalization_factor=norm,
                            **plan.extra_1d,
                        )
                        if plan.integrate_1d else None
                    )
                    r2d = (
                        integrate_2d(
                            image,
                            ai,
                            npt_rad=plan.npt_rad_2d,
                            npt_azim=plan.npt_azim_2d,
                            unit=plan.unit,
                            method=plan.method_2d,
                            mask=mask,
                            radial_range=plan.radial_range,
                            azimuth_range=plan.azimuth_range,
                            error_model=plan.error_model,
                            polarization_factor=plan.polarization_factor,
                            normalization_factor=norm,
                            **plan.extra_2d,
                        )
                        if plan.integrate_2d else None
                    )

                reduction = FrameReduction(
                    frame_index=frame.index,
                    result_1d=r1d,
                    result_2d=r2d,
                    metadata=dict(frame.metadata),
                )
                sink.write(frame, reduction)
                products[int(frame.index)] = reduction
                completed += 1
                _emit(progress_cb, scan.name, "write", frame.index, completed, total)
                if plan.clear_frame_images:
                    frame.image = None
            if cancelled:
                break
    finally:
        result = ReductionResult(
            scan_name=scan.name,
            frames=products,
            n_processed=completed,
            cancelled=cancelled or cancel_token.cancelled,
            output_path=output_path,
        )
        sink.finish(result)

    _emit(progress_cb, scan.name, "finish", None, completed, total)
    return result


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


def _combined_mask(
    plan_mask: np.ndarray | None,
    frame_mask: np.ndarray | None,
) -> np.ndarray | None:
    if plan_mask is None:
        return frame_mask
    if frame_mask is None:
        return plan_mask
    return np.asarray(plan_mask, dtype=bool) | np.asarray(frame_mask, dtype=bool)


def _normalization_for(frame: Frame, plan: ReductionPlan) -> float | None:
    if plan.normalization_factors is not None and frame.index in plan.normalization_factors:
        return float(plan.normalization_factors[frame.index])
    if frame.normalization_factor is not None:
        return float(frame.normalization_factor)
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
    path = getattr(sink, "path", None)
    return path if isinstance(path, Path) else None
