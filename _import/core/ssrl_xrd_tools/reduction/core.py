"""Scan/frame-oriented headless reduction primitives.

The intent of this module is to give GUIs and notebooks one small, stable
surface for common reduction jobs while keeping the numerical work in
``ssrl_xrd_tools.integrate``.  xdart should eventually build these objects
from its UI state and display the returned results, rather than owning
integration loops itself.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Protocol, runtime_checkable

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
    upsert_scan_metadata,
    write_nexus_frame,
)

if TYPE_CHECKING:  # C4 — tighter Scan.integrator type without forcing the import
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator

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

    incident_angle: float
    tilt_angle: float = 0.0
    sample_orientation: int = 1
    method: str = "no"


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
    flush_every: int | None = 16
    _h5: Any | None = field(default=None, init=False, repr=False)
    _n_written: int = field(default=0, init=False, repr=False)
    _scan: "Scan | None" = field(default=None, init=False, repr=False)

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
        self._n_written += 1
        if self.flush_every is not None and self._n_written % self.flush_every == 0:
            self._h5.flush()

    def finish(self, result: ReductionResult) -> None:
        if self._h5 is None:
            return
        try:
            # Persist the per-frame condition table (scan_data) after the final
            # integrated frame, before close.  A finish-time single upsert is
            # correct for a batch sink; only the GUI's SWMR live consumers would
            # need scan_data mid-run (they'd require a per-write incremental
            # upsert).  Any failure here propagates (after the file is closed
            # below) so a lost condition table is surfaced, not silent — the
            # integrated stacks written during write() are already intact.
            scan = self._scan
            if scan is not None:
                scan_data = scan.to_scan_data()
                if scan_data is not None and len(scan_data.columns):
                    upsert_scan_metadata(
                        self._h5[self.entry], scan_data, scan.frame_indices,
                    )
        finally:
            self._h5.flush()
            self._h5.close()
            self._h5 = None
            self._scan = None


def run_reduction(
    plan: ReductionPlan,
    scan: Scan,
    sink: ReductionSink | None = None,
    *,
    chunk_size: int = 1,
    clear_frame_images: bool = False,
    progress_cb: ProgressCallback | None = None,
    cancel_token: CancelToken | None = None,
) -> ReductionResult:
    """Run a headless reduction job over all frames in ``scan``.

    Parameters
    ----------
    plan
        Content of the reduction (what to integrate, mask, thresholds,
        optional :class:`GIMode`).
    scan
        Frames + scan-level context (PONI / integrator / motors).
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
    """
    if sink is None:
        sink = MemorySink()
    cancel_token = cancel_token or CancelToken()
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0; got {chunk_size}")

    total = len(scan)
    output_path = _sink_path(sink) or (
        scan.output_path if isinstance(scan.output_path, Path) else None
    )
    products: dict[int, FrameReduction] = {}
    completed = 0
    cancelled = False

    ai = None
    fi = None
    if plan.gi is not None:
        if scan.poni is None:
            raise ValueError("GI reduction requires scan.poni.")
        fi = poni_to_fiber_integrator(
            scan.poni,
            incident_angle=float(plan.gi.incident_angle),
            tilt_angle=float(plan.gi.tilt_angle),
            sample_orientation=int(plan.gi.sample_orientation),
        )
    else:
        ai = scan.integrator if scan.integrator is not None else None
        if ai is None:
            if scan.poni is None:
                raise ValueError("Reduction requires scan.integrator or scan.poni.")
            ai = poni_to_integrator(scan.poni)

    plan_masks: dict[tuple[int, int], np.ndarray | None] = {}
    sink.begin(scan, plan)
    _emit(progress_cb, scan.name, "start", None, completed, total)
    try:
        for chunk_start in range(0, total, chunk_size):
            chunk = scan.frames[chunk_start : chunk_start + chunk_size]
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
                if image.ndim != 2:
                    raise ValueError(
                        f"Frame {frame.index} image must be 2D; got shape {image.shape}"
                    )
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

                _emit(progress_cb, scan.name, "integrate", frame.index, completed, total)
                if plan.gi is not None:
                    gi_method = plan.gi.method
                    r1d = (
                        integrate_gi_1d(
                            image,
                            fi,
                            npt=plan.integration_1d.npt,
                            unit=plan.integration_1d.unit,
                            method=gi_method,
                            mask=mask,
                            radial_range=plan.integration_1d.radial_range,
                            azimuth_range=plan.integration_1d.azimuth_range,
                            **plan.integration_1d.extra,
                        )
                        if plan.integration_1d is not None else None
                    )
                    r2d = (
                        integrate_gi_2d(
                            image,
                            fi,
                            npt_rad=plan.integration_2d.npt_rad,
                            npt_azim=plan.integration_2d.npt_azim,
                            unit=plan.integration_2d.unit,
                            method=gi_method,
                            mask=mask,
                            radial_range=plan.integration_2d.radial_range,
                            azimuth_range=_integration_azimuth_range(plan.integration_2d),
                            **plan.integration_2d.extra,
                        )
                        if plan.integration_2d is not None else None
                    )
                else:
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
                            normalization_factor=_normalization_for(
                                frame, plan.integration_1d
                            ),
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
                            normalization_factor=_normalization_for(
                                frame, plan.integration_2d
                            ),
                            **plan.integration_2d.extra,
                        )
                        if plan.integration_2d is not None else None
                    )
                    if r2d is not None and plan.integration_2d.azimuth_offset:
                        r2d.azimuthal = (
                            r2d.azimuthal + float(plan.integration_2d.azimuth_offset)
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
                if clear_frame_images:
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
            return None
        if np.isfinite(norm) and norm != 0:
            return norm
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
    path = getattr(sink, "path", None)
    return path if isinstance(path, Path) else None
