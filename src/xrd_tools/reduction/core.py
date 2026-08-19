"""Scan/frame-oriented headless reduction primitives.

The intent of this module is to give GUIs and notebooks one small, stable
surface for common reduction jobs while keeping the numerical work in
``xrd_tools.integrate``.  xdart should eventually build these objects
from its UI state and display the returned results, rather than owning
integration loops itself.
"""

from __future__ import annotations

import copy
import logging
import os
import queue
import threading
import time
import warnings
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Protocol, runtime_checkable
from uuid import uuid4

import numpy as np

from xrd_tools.core.containers import (
    IntegrationResult1D,
    IntegrationResult2D,
    PONI,
)
from xrd_tools.core.frame_view import DEFAULT_MODE_KEY
from xrd_tools.core.invalid import combine_detector_masks, detector_value_mask
from xrd_tools.core.metadata import ScanMetadata, resolve_monitor_norm
from xrd_tools.core.scan import (
    FrameSource as CoreFrameSource,
    ImageLoader,
    MaskSpec as CoreMaskSpec,
    Scan as CoreScan,
    ScanFrame,
)
from xrd_tools.core.strictness import (
    GIAllDummyError,
    MissingNormalizationError,
    StrictnessError,
    StrictPolicy,
)
from xrd_tools.io.export import write_xye
from xrd_tools.io.image import read_image
from xrd_tools.io.nexus import (
    open_nexus_writer,
    open_nexus_image_stack,
    resolve_stack_compression,
)
from xrd_tools.io.record_writer import (
    NexusRecordWriter,
    RecordWrite,
    ResultMode,
    WriterFinalization,
    WriterTransactionBinding,
)
from xrd_tools.io.append import (
    AppendCommittedPrefix,
    AppendDecision,
    AppendDisposition,
    AppendIntent,
    AppendPreflight,
    AppendPreflightCleanupError,
    AppendPreflightState,
    AppendRefused,
    begin_same_run_lineage,
    extend_same_run_lineage,
    prepare_append_preflight,
    seal_append_epoch,
    truncate_append_epoch,
)
from xrd_tools.io.output_transaction import (
    LeaseOwner,
    OutputReceiptCapability,
    OutputReceiptCapabilityProvider,
    OwnerToken,
    StreamSeedMode,
    StreamTerminal,
    TransactionPhase,
    TransactionSnapshot,
    XyeSnapshot,
    get_output_transaction_coordinator,
)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # C4 — tighter Scan.integrator type without forcing the import
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator

ProgressCallback = Callable[["ReductionProgress"], None]


def supports_durable_xye_receipts(value: object) -> bool:
    """Probe the public, typed durable-XYE receipt capability contract."""
    if not isinstance(value, OutputReceiptCapabilityProvider):
        return False
    capabilities = value.output_receipt_capabilities
    return (
        isinstance(capabilities, frozenset)
        and all(isinstance(item, OutputReceiptCapability) for item in capabilities)
        and OutputReceiptCapability.DURABLE_XYE in capabilities
    )


class OutputSinkKind(str, Enum):
    """Typed description of output families present in a sink graph."""

    MEMORY = "memory"
    NEXUS = "nexus"
    XYE = "xye"


class NexusTerminalDisposition(str, Enum):
    COMMITTED = "committed"
    ABORTED = "aborted"


@dataclass(frozen=True, slots=True)
class NexusTerminalResult:
    """Detached, validated H23 terminal truth returned by a Nexus sink."""

    disposition: NexusTerminalDisposition
    transaction: TransactionSnapshot
    commit_identity: StreamTerminal | None = None

    def __post_init__(self) -> None:
        if type(self.transaction) is not TransactionSnapshot:
            raise TypeError("Nexus terminal result requires a TransactionSnapshot")
        if self.disposition is NexusTerminalDisposition.COMMITTED:
            if self.transaction.phase is not TransactionPhase.COMMITTED:
                raise ValueError("committed Nexus result requires COMMITTED transaction")
            if type(self.commit_identity) is not StreamTerminal:
                raise ValueError("committed Nexus result requires exact StreamTerminal")
            return
        if self.disposition is NexusTerminalDisposition.ABORTED:
            if self.transaction.phase is not TransactionPhase.ABORTED:
                raise ValueError("aborted Nexus result requires ABORTED transaction")
            if self.commit_identity is not None:
                raise ValueError("aborted Nexus result cannot carry commit identity")
            return
        raise ValueError("unknown Nexus terminal disposition")


@runtime_checkable
class OutputSinkKindProvider(Protocol):
    @property
    def output_sink_kinds(self) -> frozenset[OutputSinkKind]: ...


@runtime_checkable
class OutputSinkChildrenProvider(Protocol):
    """Public immutable delegation edge for a composite/proxy sink graph."""

    @property
    def output_sink_children(self) -> tuple[object, ...]: ...


class UnclassifiedOutputSinkGraph(TypeError):
    """A sink graph has no complete public output-family classification."""


def classify_output_sink_graph(value: object) -> frozenset[OutputSinkKind]:
    """Recursively classify the actual public sink graph at admission time."""
    active: set[int] = set()

    def visit(node: object) -> set[OutputSinkKind]:
        if node is None:
            return set()
        identity = id(node)
        if identity in active:
            raise UnclassifiedOutputSinkGraph("output sink graph contains a cycle")
        active.add(identity)
        try:
            if isinstance(node, OutputSinkChildrenProvider):
                children = node.output_sink_children
                if type(children) is not tuple:
                    raise UnclassifiedOutputSinkGraph(
                        "output_sink_children must be an immutable tuple"
                    )
                kinds: set[OutputSinkKind] = set()
                for child in children:
                    kinds.update(visit(child))
                return kinds
            if not isinstance(node, OutputSinkKindProvider):
                raise UnclassifiedOutputSinkGraph(
                    f"sink {type(node).__name__} has no public output requirement"
                )
            kinds = node.output_sink_kinds
            if (not isinstance(kinds, frozenset)
                    or not all(isinstance(item, OutputSinkKind) for item in kinds)):
                raise UnclassifiedOutputSinkGraph(
                    "output_sink_kinds must be a typed frozenset"
                )
            return set(kinds)
        finally:
            active.remove(identity)

    return frozenset(visit(value))


def requires_active_xye_output(value: object) -> bool:
    """Return whether a public sink graph contains an active XYE writer."""
    try:
        kinds = classify_output_sink_graph(value)
    except UnclassifiedOutputSinkGraph:
        return False
    return OutputSinkKind.XYE in kinds

# H10 §14.3 admission-lifecycle trace: OFF unless ``XDART_H10_ADMISSION_TRACE``
# names a file to append to (no default location, so an unset gate writes
# nowhere).  One tab-separated ``monotonic  thread  event  k=v`` line per event
# makes the submit/publication/worker/writer/sink/pool order readable for one
# frame identity.  Diagnostic only: no state, no owner, no public API.
_ADMISSION_TRACE_ENV = "XDART_H10_ADMISSION_TRACE"
_TRACE_UNREPRESENTABLE = "<unrepresentable>"


def _admission_trace(event: str, **fields: Any) -> None:
    """Append one diagnostic event line — fire-and-forget (§19.4).

    A trace never raises because of the DATA it was given: each field is
    formatted under its own guard and a value whose ``__repr__`` raises becomes
    one stable marker, so hostile data can neither escape nor replace the
    failure the caller must see.  The COMPLETE body then runs under ONE outer
    ``BaseException`` guard, so an interrupt inside optional trace machinery
    changes no lifecycle fact; outside it, interrupts behave normally."""
    try:
        if not (path := os.environ.get(_ADMISSION_TRACE_ENV)):
            return
        parts = [f"{time.monotonic():.6f}", f"thread={threading.get_ident()}", event]
        for key, value in fields.items():
            try:
                parts.append(f"{key}={value!r}")
            except BaseException:        # hostile field data, not a run failure
                parts.append(f"{key}={_TRACE_UNREPRESENTABLE}")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write("\t".join(parts) + "\n")
    except BaseException:  # a diagnostic may never change a lifecycle fact
        pass


class GIFreezeError(ValueError):
    """Raised when the GI output-range freeze pre-pass cannot produce a grid.

    Subclasses :class:`ValueError` so existing broad ``except ValueError``
    callers still catch it, while letting GUIs translate this *specific* GI
    failure (a blank or degenerate scout frame) into actionable guidance
    without matching on the message text.
    """


def poni_to_integrator(*args: Any, **kwargs: Any) -> Any:
    from xrd_tools.integrate.calibration import poni_to_integrator as _impl

    return _impl(*args, **kwargs)


def poni_to_fiber_integrator(*args: Any, **kwargs: Any) -> Any:
    from xrd_tools.integrate.calibration import (
        poni_to_fiber_integrator as _impl,
    )

    return _impl(*args, **kwargs)


def integrate_1d(*args: Any, **kwargs: Any) -> IntegrationResult1D:
    from xrd_tools.integrate.single import integrate_1d as _impl

    return _impl(*args, **kwargs)


def integrate_radial(*args: Any, **kwargs: Any) -> IntegrationResult1D:
    from xrd_tools.integrate.single import integrate_radial as _impl

    return _impl(*args, **kwargs)


def integrate_2d(*args: Any, **kwargs: Any) -> IntegrationResult2D:
    from xrd_tools.integrate.single import integrate_2d as _impl

    return _impl(*args, **kwargs)


def integrate_gi_1d(*args: Any, **kwargs: Any) -> IntegrationResult1D:
    from xrd_tools.integrate.gid import integrate_gi_1d as _impl

    return _impl(*args, **kwargs)


def integrate_gi_2d(*args: Any, **kwargs: Any) -> IntegrationResult2D:
    from xrd_tools.integrate.gid import integrate_gi_2d as _impl

    return _impl(*args, **kwargs)


def integrate_gi_exitangles(*args: Any, **kwargs: Any) -> IntegrationResult2D:
    from xrd_tools.integrate.gid import integrate_gi_exitangles as _impl

    return _impl(*args, **kwargs)


def integrate_gi_exitangles_1d(*args: Any, **kwargs: Any) -> IntegrationResult1D:
    from xrd_tools.integrate.gid import integrate_gi_exitangles_1d as _impl

    return _impl(*args, **kwargs)


def integrate_gi_polar(*args: Any, **kwargs: Any) -> IntegrationResult2D:
    from xrd_tools.integrate.gid import integrate_gi_polar as _impl

    return _impl(*args, **kwargs)


def integrate_gi_polar_1d(*args: Any, **kwargs: Any) -> IntegrationResult1D:
    from xrd_tools.integrate.gid import integrate_gi_polar_1d as _impl

    return _impl(*args, **kwargs)


def integrate_gi_azimuthal_1d(*args: Any, **kwargs: Any) -> IntegrationResult1D:
    from xrd_tools.integrate.gid import integrate_gi_azimuthal_1d as _impl

    return _impl(*args, **kwargs)


@dataclass(slots=True)
class CancelToken:
    """Small cancellation primitive shared by GUI and headless callers."""

    cancelled: bool = False

    def cancel(self) -> None:
        self.cancelled = True


# Architecture-v2 canonical aliases: the reduction-facing names resolve to
# the headless core contracts in ``xrd_tools.core.scan`` (the duplicate
# legacy definitions were deleted in the 1.0 monorepo migration, S4).
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
    # Azimuthal Mode A (unit='chi_deg') only: the radial sampling across the
    # q (or 2theta) band the I-vs-chi profile is pooled over.  ``npt`` is the
    # chi-bin count; this is the band resolution.  Ignored by the radial path.
    npt_rad: int = 1000
    # S-4: chi_offset re-added to the chi OUTPUT axis of a Mode-A (chi_deg)
    # reduction, mirroring Integration2DPlan.azimuth_offset -- so the written 1D
    # chi axis matches the 2D cake chi instead of the raw pyFAI frame.  Ignored by
    # the radial (q/2theta) path.
    azimuth_offset: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.npt <= 0:
            raise ValueError(f"Integration1DPlan.npt must be > 0; got {self.npt}")
        if self.npt_rad <= 0:
            raise ValueError(
                f"Integration1DPlan.npt_rad must be > 0; got {self.npt_rad}")


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
    CHI_GI = "chi_gi"


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
    method: str = "cython"
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
    # R3-C: opt-in detector-saturation masking in the HEADLESS reduction path.
    # When True, _reduce_frame excludes the dtype-derived saturation ceiling
    # (np.iinfo(dtype).max, e.g. uint16 65535) using the same fraction-guarded
    # policy as the GUI (xrd_tools.core.invalid.saturation_pixels): masked only
    # when a whole module sits at the ceiling (>1e-4 of the frame), never a few
    # genuinely-saturated Bragg pixels.  Default False is behavior-preserving;
    # core never hardcodes 65535 (a float-dtype frame -> ceiling None -> no-op).
    mask_saturation: bool = False
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
    mode_1d: str | None = None
    mode_2d: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    corrected_image: np.ndarray | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    thumbnail: np.ndarray | None = field(default=None, repr=False, compare=False)
    _thumbnail_mask_baked: bool = field(default=False, repr=False, compare=False)
    write_frame_record: bool = True


def _plan_mode_keys(plan: ReductionPlan | None) -> tuple[str, str]:
    gi = getattr(plan, "gi", None)
    if gi is None:
        return DEFAULT_MODE_KEY, DEFAULT_MODE_KEY
    return (
        str(getattr(getattr(gi, "mode_1d", None), "value", getattr(gi, "mode_1d", None))
            or DEFAULT_MODE_KEY),
        str(getattr(getattr(gi, "mode_2d", None), "value", getattr(gi, "mode_2d", None))
            or DEFAULT_MODE_KEY),
    )


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


class FrameOutcome(str, Enum):
    """Typed per-item compute outcome (H10-C1).  ``COMPLETED`` means the
    reduction produced a typed result; the terminal outcomes are exactly the
    paths that previously disappeared inside the writer loop."""

    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED_BEFORE_COMPLETION = "cancelled_before_completion"


@dataclass(frozen=True, slots=True)
class FrameOutcomeReceipt:
    """One per-item outcome from the streaming writer loop (H10-C1).

    Emitted on the WRITER thread, once per drained item: ``COMPLETED`` fires
    BEFORE the sink write hook (compute success is a distinct fact from a
    successful write — an accounting owner must never infer it from
    accepted-minus-written counts), ``FAILED``/``CANCELLED_BEFORE_COMPLETION``
    fire where the writer loop records-and-continues.  ``replacing`` is True
    for a re-fed index (a replace attempt, not a first completion).

    ``attempt`` is the exact per-label acceptance revision minted by
    ``accept_cb`` for the submission that produced this outcome, so an older
    overlapping attempt can never be mistaken for the latest one.  It is
    ``None`` when no acceptance hook is installed.
    """

    frame_index: int
    outcome: FrameOutcome
    replacing: bool
    produced_1d: bool
    produced_2d: bool
    error: str | None = None
    attempt: int | None = None


class ReductionSink(Protocol):
    """Destination for frame reduction products.

    Required hooks: ``begin`` (once, before the first write), ``write`` (once per
    frame index, on the single writer thread), ``finish`` (once, after the last
    write).  The engine also PROBES these OPTIONAL hooks by ``getattr`` and calls
    them when present — implement only the ones a sink needs:

    * ``replace(frame, reduction)`` — re-fed index (reintegration); falls back to
      ``write`` when absent.
    * ``abort(result)`` — finalize on a failed/cancelled run instead of
      ``finish``.
    * ``worker_process(frame, reduction)`` — per-frame prep run on the POOL
      worker thread (NOT the writer), e.g. a thumbnail; lets expensive per-frame
      work fan out instead of serializing on the writer.
    * ``flush(*, force=False)`` — force pending buffered output to its backing
      store (pause / end-of-run).  The save *cadence* (when to call it) is the
      caller's policy (e.g. xdart's ``FlushPolicy``), not the sink's — see
      ADR-0004 §4.
    """

    def begin(self, scan: Scan, plan: ReductionPlan) -> None: ...
    def write(self, frame: Frame, reduction: FrameReduction) -> None: ...
    def finish(self, result: ReductionResult) -> None: ...


@dataclass(slots=True)
class MemorySink:
    """In-memory sink for notebooks, tests, and xdart display handoff."""

    frames: dict[int, FrameReduction] = field(default_factory=dict)

    @property
    def output_sink_kinds(self) -> frozenset[OutputSinkKind]:
        return frozenset({OutputSinkKind.MEMORY})

    def begin(self, scan: Scan, plan: ReductionPlan) -> None:
        self.frames.clear()

    def write(self, frame: Frame, reduction: FrameReduction) -> frozenset[ResultMode]:
        self.frames[int(frame.index)] = reduction

    def finish(self, result: ReductionResult) -> None:
        return None


@dataclass(frozen=True, slots=True)
class CompositeSink:
    """Fan out reduction products to multiple sinks."""

    sinks: tuple[ReductionSink, ...]
    worker_process: Any = field(default=None, init=False, repr=False)
    output_receipt_capabilities: frozenset[OutputReceiptCapability] = field(
        default_factory=frozenset, init=False,
    )
    output_sink_kinds: frozenset[OutputSinkKind] = field(
        default_factory=frozenset, init=False,
    )
    writer_batch_size: int = field(default=1, init=False)

    @property
    def output_sink_children(self) -> tuple[object, ...]:
        return self.sinks

    def __post_init__(self) -> None:
        capabilities: set[OutputReceiptCapability] = set()
        kinds: set[OutputSinkKind] = set()
        for sink in self.sinks:
            provided = getattr(sink, "output_receipt_capabilities", frozenset())
            if (isinstance(provided, frozenset)
                    and all(isinstance(item, OutputReceiptCapability)
                            for item in provided)):
                capabilities.update(provided)
            provided_kinds = getattr(sink, "output_sink_kinds", frozenset())
            if (isinstance(provided_kinds, frozenset)
                    and all(isinstance(item, OutputSinkKind)
                            for item in provided_kinds)):
                kinds.update(provided_kinds)
        object.__setattr__(
            self, "output_receipt_capabilities", frozenset(capabilities),
        )
        object.__setattr__(self, "output_sink_kinds", frozenset(kinds))
        sizes = (_sink_writer_batch_size(sink) for sink in self.sinks)
        object.__setattr__(self, "writer_batch_size", max(sizes, default=1))
        workers = tuple(
            hook for sink in self.sinks
            if callable(hook := getattr(sink, "worker_process", None))
        )
        if workers:
            def process(frame, reduction):
                for hook in workers:
                    hook(frame, reduction)
            object.__setattr__(self, "worker_process", process)

    def begin(self, scan: Scan, plan: ReductionPlan) -> None:
        self._begin(scan, plan, self._p0_order()[1])

    def _begin(self, scan, plan, p0) -> None:
        for sink in self._p0_order()[0] if p0 else self.sinks:
            if type(sink) is CompositeSink:
                sink._begin(scan, plan, p0)
            else:
                sink.begin(scan, plan)

    def bind_session(self, facade: Any) -> None:
        for sink in self.sinks:
            bind = getattr(sink, "bind_session", None)
            if callable(bind):
                bind(facade)

    def write(self, frame: Frame, reduction: FrameReduction) -> None:
        for sink in self.sinks:
            sink.write(frame, reduction)

    def write_batch(self, items: Iterable[tuple[Frame, FrameReduction]]) -> None:
        batch = tuple(items)
        for sink in self.sinks:
            _emit_sink_write_batch(sink, batch)

    def _bind_run_saturation_mask(self, state) -> None:
        for sink in self.sinks:
            bind = getattr(sink, "_bind_run_saturation_mask", None)
            if callable(bind):
                bind(state)

    def replace(self, frame: Frame, reduction: FrameReduction) -> None:
        for sink in self.sinks:
            _emit_sink_replace(sink, frame, reduction)

    @staticmethod
    def _p0_nexus_count(sink):
        kind = type(sink)
        if kind is not CompositeSink:
            return (
                0 if kind in {MemorySink, TransactionalXYESink}
                else 1 if kind is NexusSink else None
            )
        counts = tuple(map(CompositeSink._p0_nexus_count, sink.sinks))
        return None if None in counts else sum(counts)

    def _p0_order(self):
        counts = tuple(map(self._p0_nexus_count, self.sinks))
        if type(self) is not CompositeSink or None in counts or sum(counts) != 1:
            return self.sinks, False
        owner = counts.index(1)
        return (self.sinks[owner], *self.sinks[:owner], *self.sinks[owner + 1:]), True

    def _terminal(self, result, *, failed, p0) -> NexusTerminalResult | None:
        values, error = [], None
        sinks, nexus_first = self._p0_order() if p0 else (self.sinks, False)
        for position, sink in enumerate(sinks):
            hook = getattr(sink, "abort", None) if failed else sink.finish
            try:
                value = (sink._terminal(result, failed=failed, p0=p0)
                         if type(sink) is CompositeSink else
                         (hook if callable(hook) else sink.finish)(result))
            except BaseException as exc:  # pragma: no cover - defensive fan-out
                if nexus_first and position == 0:
                    raise
                error = exc if error is None else error
            else:
                if nexus_first and position == 0 and type(value) is not NexusTerminalResult:
                    raise RuntimeError("CompositeSink Nexus branch did not settle")
                values.append(value)
        if error is not None:
            raise error
        terminals = [value for value in values if type(value) is NexusTerminalResult]
        if len(terminals) > 1:
            raise RuntimeError("CompositeSink terminal requires exactly one Nexus owner")
        return terminals[0] if terminals else None

    def finish(self, result: ReductionResult) -> NexusTerminalResult | None:
        return self._terminal(result, failed=False, p0=self._p0_order()[1])

    def abort(self, result: ReductionResult | None) -> NexusTerminalResult | None:
        return self._terminal(result, failed=True, p0=self._p0_order()[1])

    def flush(self, *, force: bool = False) -> None:
        for sink in self.sinks:
            flush = getattr(sink, "flush", None)
            if callable(flush):
                flush(force=force)

    def _settle_deferred_publication_drops(
        self, frame: Frame, reduction: FrameReduction,
    ) -> None:
        for sink in self.sinks:
            settle = getattr(sink, "_settle_deferred_publication_drops", None)
            if callable(settle):
                settle(frame, reduction)

    def perf_snapshot(self) -> dict[str, float]:
        values: dict[str, float] = {}
        for sink in self.sinks:
            snapshot = getattr(sink, "perf_snapshot", None)
            if callable(snapshot):
                for key, elapsed in snapshot().items():
                    values[key] = values.get(key, 0.0) + float(elapsed)
        return values


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


def _sink_writer_batch_size(sink: object) -> int:
    value = getattr(sink, "writer_batch_size", 1)
    if type(value) is not int or value < 1:
        raise TypeError("sink writer_batch_size must be a positive exact int")
    return value


def _emit_sink_write_batch(sink: ReductionSink, items: tuple[tuple, ...]) -> None:
    write_batch = getattr(sink, "write_batch", None)
    if callable(write_batch):
        write_batch(items)
        return
    for frame, reduction in items:
        sink.write(frame, reduction)


_XYE_WRITE_END = object()


def _performance_timing_enabled() -> bool:
    return (
        bool(os.environ.get("XDART_PERF"))
        or os.environ.get("XDART_PERF_QUARTILES", "").strip() == "1"
    )


@dataclass(slots=True)
class XYESink:
    """Write 1D reductions as one ``.xye`` file per frame."""

    directory: Path | str
    pattern: str = "{scan}_{frame:04d}.xye"
    _scan_name: str = field(default="", init=False, repr=False)
    _perf_enabled: bool = field(default=False, init=False, repr=False)
    _perf_write: float = field(default=0.0, init=False, repr=False)
    _perf_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False,
    )
    _pending: Any = field(default=None, init=False, repr=False)
    _worker: Any = field(default=None, init=False, repr=False)
    _errors: list[BaseException] = field(default_factory=list, init=False, repr=False)

    @property
    def output_sink_kinds(self) -> frozenset[OutputSinkKind]:
        return frozenset({OutputSinkKind.XYE})

    def __post_init__(self) -> None:
        if not isinstance(self.directory, Path):
            self.directory = Path(self.directory)
        self._perf_enabled = _performance_timing_enabled()

    def begin(self, scan: Scan, plan: ReductionPlan) -> None:
        self._scan_name = scan.name
        with self._perf_lock:
            self._perf_write = 0.0
        self._errors.clear()
        self.directory.mkdir(parents=True, exist_ok=True)
        pending: queue.Queue[object] = queue.Queue(maxsize=16)
        self._pending = pending

        def write_pending() -> None:
            while True:
                item = pending.get()
                try:
                    if item is _XYE_WRITE_END:
                        return
                    path, radial, intensity, sigma = item
                    started = time.perf_counter() if self._perf_enabled else 0.0
                    try:
                        write_xye(path, radial, intensity, sigma)
                    except BaseException as error:
                        self._errors.append(error)
                    finally:
                        if self._perf_enabled:
                            with self._perf_lock:
                                self._perf_write += (
                                    time.perf_counter() - started
                                )
                finally:
                    pending.task_done()

        self._worker = threading.Thread(
            target=write_pending,
            name="xrd-tools-xye-writer",
            daemon=True,
        )
        self._worker.start()

    def write(self, frame: Frame, reduction: FrameReduction) -> None:
        result = reduction.result_1d
        if result is None:
            return
        path = self.directory / self.pattern.format(
            scan=self._scan_name,
            frame=int(frame.index),
            label=frame.label,
        )
        pending = self._pending
        if pending is None:
            raise RuntimeError("XYESink.write called before begin().")
        pending.put((path, result.radial, result.intensity, result.sigma))

    def finish(self, result: ReductionResult) -> None:
        pending, worker = self._pending, self._worker
        if pending is not None and worker is not None:
            pending.put(_XYE_WRITE_END)
            worker.join()
        self._pending = None
        self._worker = None
        if self._errors:
            error = self._errors[0]
            self._errors.clear()
            raise error

    def abort(self, result: ReductionResult) -> None:
        self.finish(result)

    def perf_snapshot(self) -> dict[str, float]:
        with self._perf_lock:
            return {"sink_xye_write": self._perf_write}


@dataclass(frozen=True, slots=True)
class _TransactionalXYEHandoff:
    identity: tuple[int, XyeSnapshot]
    descriptors: tuple[tuple[int, ResultMode, str], ...]
    kind: str


@dataclass(slots=True)
class TransactionalXYESink:
    """Publish dynamic 1-D values through the shared H23 transaction owner."""

    directory: Path | str
    stale_paths: tuple[Path | str, ...] = field(default=(), kw_only=True)
    pattern: str = field(default="{scan}_{frame:04d}.xye", kw_only=True)
    prefix: str | None = field(default=None, kw_only=True)
    _run_owner: OwnerToken = field(init=False, repr=False)
    _transaction: Any = field(init=False, repr=False)
    _canonical_directory: Path = field(init=False, repr=False)
    _stale_paths: tuple[Path | str, ...] = field(init=False, repr=False)
    _pattern: str = field(init=False, repr=False)
    _canonical_target: str = field(init=False, repr=False)
    _boundary: Any = field(default=None, init=False, repr=False)
    _scan_name: str = field(default="", init=False, repr=False)
    _mode: ResultMode | None = field(default=None, init=False, repr=False)
    _transition_kind: str | None = field(default=None, init=False, repr=False)
    _handoff: _TransactionalXYEHandoff | None = field(
        default=None, init=False, repr=False,
    )
    _transition_ordinal: int = field(default=0, init=False, repr=False)
    _published_any: bool = field(default=False, init=False, repr=False)
    _stage_nonce: str = field(init=False, repr=False)
    _stage_paths: dict[int, Path] = field(default_factory=dict, init=False, repr=False)
    _ready_labels: set[int] = field(default_factory=set, init=False, repr=False)
    _pending: Any = field(default=None, init=False, repr=False)
    _worker: Any = field(default=None, init=False, repr=False)
    _errors: list[BaseException] = field(default_factory=list, init=False, repr=False)
    _perf_enabled: bool = field(default=False, init=False, repr=False)
    _perf_values: dict[str, float] = field(
        default_factory=dict, init=False, repr=False,
    )
    _perf_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False,
    )

    def __post_init__(self) -> None:
        if self.pattern != "{scan}_{frame:04d}.xye":
            raise ValueError(
                "transactional XYE supports only the direct per-frame pattern"
            )
        if self.prefix is not None:
            if type(self.prefix) is not str or self.prefix not in {
                "iq", "itth", "iqip", "iqoop", "iexit",
            }:
                raise ValueError("transactional XYE prefix is invalid")
            self.pattern = f"{self.prefix}_{{scan}}_{{frame:04d}}.xye"
        self.stale_paths = tuple(self.stale_paths)
        self._stale_paths = self.stale_paths
        self._pattern = self.pattern
        owner = OwnerToken(f"transactional-xye:{self.directory}")
        transaction = get_output_transaction_coordinator().prepare_xye(
            self.directory, run_owner=owner,
        )
        normalized = transaction.snapshot().directory
        self.directory = Path(normalized)
        self._canonical_directory = self.directory
        self._run_owner = owner
        self._transaction = transaction
        self._canonical_target = f"xye:{normalized}"
        self._stage_nonce = uuid4().hex
        self._perf_enabled = _performance_timing_enabled()

    @property
    def output_sink_kinds(self) -> frozenset[OutputSinkKind]:
        return frozenset({OutputSinkKind.XYE})

    @property
    def output_receipt_capabilities(self) -> frozenset[OutputReceiptCapability]:
        return frozenset({OutputReceiptCapability.DURABLE_XYE})

    @property
    def canonical_target(self) -> str:
        return self._canonical_target

    @property
    def _settlement_pending(self) -> bool:
        return self._transition_kind is not None

    @property
    def _has_canonical_output(self) -> bool:
        return self._published_any

    def bind_session(self, boundary: Any) -> None:
        if getattr(boundary, "defer_epoch_durability", None) is not True:
            raise TypeError(
                "transactional XYE requires the deferred dynamic writer boundary"
            )
        if self._boundary is not None and self._boundary is not boundary:
            raise RuntimeError("transactional XYE boundary identity changed")
        self._boundary = boundary

    def begin(self, scan: Scan, plan: ReductionPlan) -> None:
        boundary = self._boundary
        if boundary is None:
            raise RuntimeError("transactional XYE has no bound writer boundary")
        if plan.integration_1d is None:
            raise ValueError("transactional XYE requires one 1-D result mode")
        mode_1d, mode_2d = _plan_mode_keys(plan)
        mode = ResultMode.one_d(mode_1d)
        if self._canonical_target not in boundary.targets_for(mode):
            raise ValueError("transactional XYE target does not match the 1-D mode")
        if (
            plan.integration_2d is not None
            and self._canonical_target in boundary.targets_for(
                ResultMode.two_d(mode_2d)
            )
        ):
            raise ValueError("transactional XYE target cannot satisfy a 2-D mode")
        self._scan_name = scan.name
        self._mode = mode
        self._errors.clear()
        with self._perf_lock:
            self._perf_values = {
                "sink_xye_write": 0.0,
                "sink_xye_worker_format": 0.0,
                "sink_xye_enqueue_wait": 0.0,
                "sink_xye_queue_high_water": 0,
                "sink_xye_drain": 0.0,
                "sink_xye_promotion": 0.0,
                "sink_xye_generated": 0,
            }
        self._canonical_directory.mkdir(parents=True, exist_ok=True)
        pending: queue.Queue[object] = queue.Queue(maxsize=16)
        self._pending = pending

        def write_pending() -> None:
            while True:
                item = pending.get()
                try:
                    if item is _XYE_WRITE_END:
                        return
                    label, radial, intensity, sigma = item
                    try:
                        self._format_stage(
                            label, radial, intensity, sigma, worker=True,
                        )
                    except BaseException as error:
                        self._errors.append(error)
                finally:
                    pending.task_done()

        self._worker = threading.Thread(
            target=write_pending,
            name="xrd-tools-xye-writer",
            daemon=True,
        )
        self._worker.start()

    def _path_for(self, frame: Frame) -> Path:
        path = os.path.normcase(os.path.abspath(str(
            self._canonical_directory / self._pattern.format(
                scan=self._scan_name,
                frame=int(frame.index),
                label=frame.label,
            )
        )))
        if os.path.dirname(path) != str(self._canonical_directory):
            raise ValueError(
                "transactional XYE output path must stay in its canonical directory"
            )
        return Path(path)

    def _stage_path(self, label: int) -> Path:
        return self._canonical_directory / (
            f".xdart-xye-{self._stage_nonce}-{int(label)}.tmp"
        )

    def _format_stage(
        self, label: int, radial, intensity, sigma, *, worker: bool,
    ) -> None:
        label = int(label)
        stage = self._stage_path(label)
        self._stage_paths[label] = stage
        self._ready_labels.discard(label)
        started = time.perf_counter() if self._perf_enabled else 0.0
        try:
            write_xye(stage, radial, intensity, sigma)
        finally:
            if self._perf_enabled and worker:
                elapsed = max(0.0, time.perf_counter() - started)
                with self._perf_lock:
                    self._perf_values["sink_xye_worker_format"] += elapsed
                    self._perf_values["sink_xye_write"] += elapsed
        self._ready_labels.add(label)
        if self._perf_enabled:
            with self._perf_lock:
                self._perf_values["sink_xye_generated"] += 1

    def _drain_worker(self, *, terminal: bool) -> None:
        pending, worker = self._pending, self._worker
        started = time.perf_counter() if self._perf_enabled else 0.0
        if pending is not None and worker is not None:
            if terminal:
                pending.put(_XYE_WRITE_END)
                worker.join()
                self._pending = None
                self._worker = None
            else:
                pending.join()
        if self._perf_enabled:
            with self._perf_lock:
                self._perf_values["sink_xye_drain"] += max(
                    0.0, time.perf_counter() - started,
                )

    def _cleanup_staging(self) -> None:
        failures: list[BaseException] = []
        for label, path in tuple(self._stage_paths.items()):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except BaseException as error:
                failures.append(error)
                continue
            self._stage_paths.pop(label, None)
            self._ready_labels.discard(label)
        if failures:
            raise failures[0]

    def write(self, frame: Frame, reduction: FrameReduction) -> None:
        result = reduction.result_1d
        if result is None:
            return
        mode = ResultMode.one_d(
            str(reduction.mode_1d or DEFAULT_MODE_KEY)
        )
        if self._mode is None:
            raise RuntimeError("TransactionalXYESink.write called before begin().")
        if mode != self._mode:
            raise ValueError("transactional XYE result mode changed during the run")
        pending = self._pending
        if pending is None:
            raise RuntimeError("TransactionalXYESink worker is not active")
        label = int(frame.index)
        path = self._path_for(frame)
        self._transaction.stage(
            self._run_owner,
            label,
            (path, mode, result.radial, result.intensity, result.sigma),
        )
        self._stage_paths[label] = self._stage_path(label)
        started = time.perf_counter() if self._perf_enabled else 0.0
        pending.put((
            label, result.radial, result.intensity, result.sigma,
        ))
        if self._perf_enabled:
            with self._perf_lock:
                self._perf_values["sink_xye_enqueue_wait"] += max(
                    0.0, time.perf_counter() - started,
                )
                self._perf_values["sink_xye_queue_high_water"] = max(
                    self._perf_values["sink_xye_queue_high_water"],
                    pending.qsize(),
                )

    def replace(self, frame: Frame, reduction: FrameReduction) -> None:
        self.write(frame, reduction)

    def flush(self, *, force: bool = False) -> None:
        return None

    def finish(self, result: ReductionResult) -> None:
        return None

    def abort(self, result: ReductionResult | None) -> None:
        if self._settlement_pending:
            raise RuntimeError("attempted XYE publication requires exact retry")
        self._drain_worker(terminal=True)
        self._transaction.abandon(run_owner=self._run_owner)
        self._cleanup_staging()
        if self._errors:
            error = self._errors[0]
            self._errors.clear()
            raise error

    def _publish_values(self, values) -> tuple[tuple[int, ResultMode, str], ...]:
        if self._errors:
            error = self._errors[0]
            self._errors.clear()
            raise error
        descriptors = []
        started = time.perf_counter() if self._perf_enabled else 0.0
        for label, value in values:
            path, mode, radial, intensity, sigma = value
            label = int(label)
            stage = self._stage_path(label)
            if label not in self._ready_labels or not stage.exists():
                self._format_stage(
                    label, radial, intensity, sigma, worker=False,
                )
            os.replace(stage, path)
            self._stage_paths.pop(label, None)
            self._ready_labels.discard(label)
            descriptors.append((int(label), mode, self._canonical_target))
        if self._perf_enabled:
            with self._perf_lock:
                self._perf_values["sink_xye_promotion"] += max(
                    0.0, time.perf_counter() - started,
                )
        return tuple(descriptors)

    def _settle_transition(self, kind: str, boundary: Any):
        if kind not in {"epoch", "finish"}:
            raise ValueError("unknown transactional XYE transition")
        if boundary is not self._boundary:
            raise RuntimeError("transactional XYE settlement changed boundary")
        if self._transition_kind not in {None, kind}:
            raise RuntimeError("another XYE transition remains unsettled")
        self._transition_kind = kind
        handoff = self._handoff
        if handoff is None:
            published: list[tuple[int, ResultMode, str]] = []

            def publisher(values) -> None:
                published.extend(self._publish_values(values))

            try:
                self._drain_worker(terminal=kind == "finish")
                snapshot = self._transaction.snapshot()
                if snapshot.retryable:
                    token = snapshot.cleanup_token
                    if token is None:
                        raise RuntimeError("retryable XYE transaction lost its token")
                    snapshot = self._transaction.retry_publication(
                        token, publisher=publisher,
                    )
                elif kind == "epoch":
                    snapshot = self._transaction.publish_epoch(
                        run_owner=self._run_owner,
                        stale_paths=self._stale_paths,
                        publisher=publisher,
                    )
                else:
                    snapshot = self._transaction.publish(
                        run_owner=self._run_owner,
                        stale_paths=self._stale_paths,
                        publisher=publisher,
                    )
            except BaseException:
                if not self._transaction.snapshot().retryable:
                    self._transition_kind = None
                raise
            self._transition_ordinal += 1
            handoff = _TransactionalXYEHandoff(
                (self._transition_ordinal, snapshot), tuple(published), kind,
            )
            self._handoff = handoff
            self._published_any = self._published_any or bool(published)
        receipts = tuple(
            boundary.capture_receipt(label, mode, target)
            for label, mode, target in handoff.descriptors
        )
        boundary.commit_durable(receipts)
        return handoff.identity

    def perf_snapshot(self) -> dict[str, float]:
        with self._perf_lock:
            return dict(self._perf_values)

    def _acknowledge_transition(self, identity) -> None:
        handoff = self._handoff
        if handoff is None or identity is not handoff.identity:
            raise RuntimeError("transactional XYE promotion identity changed")
        self._handoff = None
        self._transition_kind = None


@dataclass(slots=True)
class NexusSink:
    """Headless value adapter for :class:`NexusRecordWriter`.

    The shared writer owns the open handle, row cursors, H10 durability
    receipts, flush cadence, partial-file preservation, and terminal state.
    This adapter only converts reduction-domain values into ``RecordWrite`` and
    ``WriterFinalization`` objects.  ``file_lock`` is optional and borrowed;
    without one this retains the pre-cutover single-session serialization
    contract rather than claiming cross-session exclusion.  H23-C3 supplies
    the canonical GUI lock at its composition boundary.
    """

    path: Path | str
    entry: str = "entry"
    # Default honors XDART_INTEGRATED_COMPRESSION (gzip when unset) so a headless
    # run picks up the same override as the GUI; pass compression= to bypass.
    compression: str | None = field(default_factory=resolve_stack_compression)
    overwrite: bool = False
    flush_every: int | None = 16
    atomic: bool | None = None
    complete_record: bool = True
    source_base: Path | str | None = None
    write_thumbnails: bool = True
    thumbnail_max: int = 256
    run_configuration_provenance: Mapping[str, Any] | None = field(
        default=None, repr=False,
    )
    source_execution_provenance: Mapping[str, Any] | None = field(
        default=None, repr=False,
    )
    source_snapshots_provenance: Mapping[str, Mapping[str, Any]] | None = None
    file_lock: Any | None = None
    append_preflight: AppendPreflight | None = None
    same_run_intent: AppendIntent | None = None
    allow_unbound_same_run: bool = False
    incremental_finalization: bool = False
    durable_fsync: bool = True
    _writer: NexusRecordWriter | None = field(default=None, init=False, repr=False)
    _transaction: Any | None = field(default=None, init=False, repr=False)
    _lease: Any | None = field(default=None, init=False, repr=False)
    _attempt: Any | None = field(default=None, init=False, repr=False)
    _transaction_owners: tuple[Any, Any, dict] | None = field(
        default=None, init=False, repr=False,
    )
    _scan: "Scan | None" = field(default=None, init=False, repr=False)
    _plan: "ReductionPlan | None" = field(default=None, init=False, repr=False)
    _run_saturation_mask: "_RunSaturationMask | None" = field(
        default=None, init=False, repr=False,
    )
    _session_facade: Any | None = field(default=None, init=False, repr=False)
    _primary_mode_1d: str = field(default=DEFAULT_MODE_KEY, init=False, repr=False)
    _primary_mode_2d: str = field(default=DEFAULT_MODE_KEY, init=False, repr=False)
    _extension_owner: Any | None = field(default=None, init=False, repr=False)
    _pending_append_decision: Any | None = field(default=None, init=False, repr=False)
    _epoch_decision: Any | None = field(default=None, init=False, repr=False)
    _source_snapshots: dict[str, dict[str, Any]] = field(
        default_factory=dict, init=False, repr=False,
    )
    _run_configuration: dict[str, Any] | None = field(
        default=None, init=False, repr=False,
    )
    _fast_regenerable: bool = field(default=False, init=False, repr=False)
    _source_execution: dict[str, Any] | None = field(
        default=None, init=False, repr=False,
    )
    _terminal_result: NexusTerminalResult | None = field(
        default=None, init=False, repr=False,
    )
    _deferred_publication_drops: dict[int, tuple[ResultMode, ...]] = field(
        default_factory=dict, init=False, repr=False,
    )
    _defer_publication_drop_settlement: bool = field(
        default=False, init=False, repr=False,
    )
    _writer_batch_size: int = field(default=1, init=False, repr=False)
    _nexus_record_batch_size: int | None = field(default=None, init=False, repr=False)
    _pending_record_writes: list[tuple[RecordWrite, tuple[ResultMode, ...], int]] = field(
        default_factory=list, init=False, repr=False,
    )
    _settled_buffered_drop_labels: set[int] = field(
        default_factory=set, init=False, repr=False,
    )
    _existing_append_intent: AppendIntent | None = field(
        default=None, init=False, repr=False,
    )
    _existing_append_prefix: AppendCommittedPrefix | None = field(
        default=None, init=False, repr=False,
    )
    _existing_append_pending: bool = field(default=False, init=False, repr=False)
    _perf_nexus_write: float = field(default=0.0, init=False, repr=False)
    _perf_nexus_flush: float = field(default=0.0, init=False, repr=False)
    _perf_nexus_enabled: bool = field(default=False, init=False, repr=False)
    _perf_nexus_lock: Any = field(
        default_factory=threading.Lock, init=False, repr=False,
    )

    @classmethod
    def for_existing_append(
        cls,
        path: Path | str,
        intent: AppendIntent,
        *,
        committed_prefix: AppendCommittedPrefix | None = None,
        **sink_values: Any,
    ) -> "NexusSink":
        """Build a sink that qualifies one existing target lazily at begin."""
        if type(intent) is not AppendIntent:
            raise TypeError("existing Append requires an exact AppendIntent")
        if (committed_prefix is not None
                and type(committed_prefix) is not AppendCommittedPrefix):
            raise TypeError(
                "existing Append prefix must be an exact AppendCommittedPrefix"
            )
        if any(name in sink_values for name in (
            "append_preflight", "same_run_intent", "allow_unbound_same_run",
        )):
            raise ValueError("existing Append owns its admission policy")
        sink = cls(path, **sink_values)
        if sink.overwrite:
            raise ValueError("existing Append cannot overwrite its target")
        if not sink.durable_fsync:
            raise ValueError("diagnostic no-fsync is unavailable for Append")
        sink._existing_append_intent = intent
        sink._existing_append_prefix = committed_prefix
        sink._existing_append_pending = True
        return sink

    @property
    def output_sink_kinds(self) -> frozenset[OutputSinkKind]:
        return frozenset({OutputSinkKind.NEXUS})

    @property
    def writer_batch_size(self) -> int:
        return self._writer_batch_size

    def _configure_writer_batch_size(self, value: int) -> None:
        if type(value) is not int or not 1 <= value <= 16:
            raise TypeError("Nexus writer batch size must be an exact int in [1, 16]")
        if self._writer is not None:
            raise RuntimeError("Nexus writer batch size must bind before begin")
        self._writer_batch_size = value

    @property
    def nexus_record_batch_size(self) -> int | None:
        return self._nexus_record_batch_size

    def _configure_nexus_record_batch_size(self, value: int) -> None:
        if type(value) is not int or not 1 <= value <= 16:
            raise TypeError("Nexus record batch size must be an exact int in [1, 16]")
        if self._writer is not None:
            raise RuntimeError("Nexus record batch size must bind before begin")
        self._nexus_record_batch_size = value

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            self.path = Path(self.path)
        if type(self.durable_fsync) is not bool:
            raise TypeError("Nexus durable_fsync must be an exact bool")
        if not self.durable_fsync and self.append_preflight is not None:
            raise ValueError(
                "diagnostic no-fsync is unavailable for Append"
            )
        if self.overwrite and self.append_preflight is not None:
            raise ValueError("Overwrite cannot consume an Append preflight")
        if self.same_run_intent is not None and self.append_preflight is not None:
            raise ValueError("same-run creation and cross-run Append are distinct policies")
        if self.allow_unbound_same_run and (
            not self.overwrite
            or self.append_preflight is not None
            or self.same_run_intent is not None
        ):
            raise ValueError(
                "unbound same-run adoption requires a fresh Overwrite owner"
            )
        from xrd_tools.reduction.provenance_config import jsonable_run_value

        if self.run_configuration_provenance is not None:
            self._run_configuration = jsonable_run_value(
                self.run_configuration_provenance,
                path="run_configuration",
            )
        config = self._run_configuration or {}
        self._fast_regenerable = bool(
            self.overwrite and config.get("output_mode") == "Overwrite"
            and config.get("live_mode") is False)
        if self.source_execution_provenance is not None:
            self._source_execution = jsonable_run_value(
                self.source_execution_provenance,
                path="source_execution",
            )
        self.run_configuration_provenance = None
        self.source_execution_provenance = None
        snapshots = self.source_snapshots_provenance or {}
        self._source_snapshots = {
            str(path): jsonable_run_value(snapshot, path=f"source_snapshot[{path!r}]")
            for path, snapshot in snapshots.items()
        }
        self.source_snapshots_provenance = None

    @property
    def _tmp_path(self) -> Path | None:
        writer = self._writer
        if writer is None or writer.phase.value not in {"active", "partial"}:
            return None
        active = writer.active_path
        return active if active is not None and active != self.path else None

    def bind_session(self, facade: Any) -> None:
        if self._writer is not None and self._writer.phase.value in {"active", "partial"}:
            raise RuntimeError("NexusSink session facade cannot change during a write")
        self._session_facade = facade

    def _release_terminal_lease(self) -> None:
        if self._transaction_owners is None:
            return
        owners = self._transaction_owners[2]
        try:
            for role in tuple(owners):
                self._transaction.release_lease_owner(
                    self._lease, role, owners[role]
                )
                del owners[role]
        except BaseException:
            if self.append_preflight is not None:
                self.append_preflight._terminal(
                    self.append_preflight._cleanup_failure_state())
            raise
        if not owners:
            self._transaction_owners = None
            self._extension_owner = None
            self._pending_append_decision = None
            self._epoch_decision = None

    def _settle_unstarted_transaction(self) -> None:
        snapshot = self._transaction.snapshot()
        if snapshot.phase.value == "aborted":
            self._release_terminal_lease()
            return
        if snapshot.pending_actions:
            if snapshot.cleanup_token is None:
                raise RuntimeError("pending output cleanup has no exact owner")
            self._transaction.retry_cleanup(snapshot.cleanup_token)
        self._transaction.abandon(self._lease)
        self._release_terminal_lease()

    def _typed_terminal(
        self, snapshot: TransactionSnapshot,
    ) -> NexusTerminalResult:
        if snapshot.phase is TransactionPhase.COMMITTED:
            writer = self._writer
            if writer is None:
                raise RuntimeError("committed Nexus transaction lost its writer")
            outcome = writer.finish()
            result = NexusTerminalResult(
                NexusTerminalDisposition.COMMITTED,
                snapshot,
                outcome.stream_terminal,
            )
        elif snapshot.phase is TransactionPhase.ABORTED:
            result = NexusTerminalResult(
                NexusTerminalDisposition.ABORTED, snapshot, None,
            )
        else:
            raise RuntimeError(
                f"Nexus transaction is not terminal: {snapshot.phase.value}"
            )
        if self._terminal_result is not None and result != self._terminal_result:
            raise RuntimeError("Nexus terminal retry changed its exact result")
        self._terminal_result = self._terminal_result or result
        return self._terminal_result

    def _abort_composed(self) -> NexusTerminalResult:
        if self._terminal_result is not None:
            return self._terminal_result
        writer = self._writer
        phase = self._transaction.snapshot().phase.value
        if writer is not None and writer.phase.value == "finished" and phase == "cleanup-pending":
            snapshot = self._transaction.snapshot()
            self._transaction.retry_cleanup(snapshot.cleanup_token)
            phase = self._transaction.snapshot().phase.value
        if (writer is not None and writer.phase.value == "finished"
                and phase in {"epoch-committed", "committed"}):
            if phase == "epoch-committed":
                self._transaction.commit_stream(self._attempt, lease=self._lease)
            self._release_terminal_lease()
        else:
            if writer is not None and writer.phase.value != "aborted":
                writer.abort()
            if self._attempt is None:
                self._settle_unstarted_transaction()
            else:
                snapshot = self._transaction.abort_stream(
                    self._attempt, lease=self._lease,
                    retain_partial=bool(writer and writer.written_labels),
                )
                if snapshot.partial_path:
                    warnings.warn(f"writer abort preserved non-final data at {snapshot.partial_path}", RuntimeWarning, stacklevel=2)
                self._release_terminal_lease()
        if self.append_preflight is not None:
            state = (
                AppendPreflightState.COMMITTED
                if self._transaction.snapshot().phase.value == "committed"
                else AppendPreflightState.ABORTED
            )
            self.append_preflight._terminal(state)
        return self._typed_terminal(self._transaction.snapshot())

    def _prepare_transaction(self):
        if self._existing_append_intent is not None and self.append_preflight is None:
            if not self._existing_append_pending:
                raise RuntimeError("existing Append qualification already failed")
            self._existing_append_pending = False
            try:
                self.append_preflight = prepare_append_preflight(
                    self.path,
                    self._existing_append_intent,
                    committed_prefix=self._existing_append_prefix,
                    file_lock=self.file_lock,
                )
            except AppendPreflightCleanupError as error:
                self.append_preflight = error.owner
                raise
        if self.append_preflight is not None:
            binding = self.append_preflight._consume(self.path)
            self._transaction = binding.transaction
            self._lease = binding.lease
            self._transaction_owners = (
                binding.transaction_owner,
                binding.target_owner,
                binding.owners,
            )
            try:
                self._attempt = binding.transaction.begin_stream(
                    admission=binding.transaction.admission,
                    transaction_owner=binding.transaction_owner,
                    target_owner=binding.target_owner,
                    lease=binding.lease,
                    file_lock=self.file_lock,
                    seed_mode=StreamSeedMode.PRESERVE_BASE,
                )
            except BaseException as primary:
                try:
                    self._settle_unstarted_transaction()
                except BaseException as cleanup:
                    self.append_preflight._terminal(
                        self.append_preflight._cleanup_failure_state())
                    raise primary from cleanup
                self.append_preflight._terminal(AppendPreflightState.ABORTED)
                raise
            return binding.decision
        coordinator = get_output_transaction_coordinator()
        transaction_owner = OwnerToken("nexus-sink-transaction")
        target_owner = OwnerToken("nexus-sink-target")
        owners = {role: OwnerToken(f"nexus-sink-{role.value}") for role in LeaseOwner}
        with (nullcontext() if self.file_lock is None else self.file_lock):
            transaction = coordinator.admit(
                self.path,
                transaction_owner=transaction_owner,
                target_owner=target_owner,
                durable_fsync=self.durable_fsync,
                fast_regenerable=self._fast_regenerable,
            )
            lease = transaction.acquire_lease(
                admission=transaction.admission,
                transaction_owner=transaction_owner,
                target_owner=target_owner,
                owners=owners,
            )
        self._transaction = transaction
        self._lease = lease
        self._transaction_owners = (transaction_owner, target_owner, owners)
        try:
            if self.same_run_intent is not None:
                if transaction.admission.snapshot.exists and not self.overwrite:
                    raise ValueError(
                        "existing target requires an exact Append preflight")
                decision = begin_same_run_lineage(self.same_run_intent)
            else:
                decision = None
        except BaseException as primary:
            try:
                self._settle_unstarted_transaction()
            except BaseException as cleanup:
                raise primary from cleanup
            raise
        if decision is not None and decision.disposition is not AppendDisposition.WRITE:
            transaction.abandon(lease)
            self._release_terminal_lease()
            raise AppendRefused(decision)
        try:
            self._attempt = transaction.begin_stream(
                admission=transaction.admission,
                transaction_owner=transaction_owner,
                target_owner=target_owner,
                lease=lease,
                file_lock=self.file_lock,
                seed_mode=(
                    StreamSeedMode.EMPTY_REPLACEMENT
                    if self.overwrite
                    else StreamSeedMode.PRESERVE_BASE
                ),
            )
        except BaseException as primary:
            try:
                self._settle_unstarted_transaction()
            except BaseException as cleanup:
                raise primary from cleanup
            raise
        return decision

    def begin(self, scan: Scan, plan: ReductionPlan) -> None:
        if self._writer is not None and self._writer.phase.value in {"active", "partial"}:
            raise RuntimeError("NexusSink.begin called before the prior writer terminated")
        self._scan = scan
        self._plan = plan
        self._perf_nexus_enabled = _performance_timing_enabled()
        with self._perf_nexus_lock:
            self._perf_nexus_write = 0.0
            self._perf_nexus_flush = 0.0
        if not self._source_snapshots:
            snapshots = (scan.extra or {}).get("source_snapshots") or {}
            from xrd_tools.reduction.provenance_config import jsonable_run_value
            self._source_snapshots = {
                str(path): jsonable_run_value(
                    snapshot, path=f"source_snapshot[{path!r}]",
                )
                for path, snapshot in snapshots.items()
            }
        self._writer = None
        self._attempt = None
        self._terminal_result = None
        self._deferred_publication_drops.clear()
        self._pending_record_writes.clear()
        self._settled_buffered_drop_labels.clear()
        self._primary_mode_1d, self._primary_mode_2d = _plan_mode_keys(plan)
        append_decision = self._prepare_transaction()
        try:
            writer = NexusRecordWriter(
                self.path,
                entry=self.entry,
                compression=self.compression,
                overwrite=self.overwrite,
                atomic=False,
                flush_every=self.flush_every,
                complete_record=self.complete_record,
                source_base=self.source_base,
                file_lock=self.file_lock,
                opener=open_nexus_writer,
                transaction_binding=WriterTransactionBinding(
                    self._transaction, self._attempt, self._lease,
                ),
                append_decision=append_decision,
                fast_regenerable=self._fast_regenerable,
            )
            self._writer = writer
            if self._session_facade is not None:
                writer.bind_session(self._session_facade)
            writer.begin(
                metadata=scan.to_metadata(),
                primary_mode_1d=self._primary_mode_1d,
                primary_mode_2d=self._primary_mode_2d,
            )
            if self.append_preflight is not None:
                self.append_preflight._set_consumer(self._queue_append_decision)
            if (self.append_preflight is not None
                    or self.same_run_intent is not None
                    or self.allow_unbound_same_run):
                self._extension_owner = OwnerToken("same-run-extension")
        except BaseException as primary:
            try:
                self._abort_composed()
            except BaseException as cleanup:
                raise primary from cleanup
            raise

    @property
    def extension_owner(self):
        if self._extension_owner is None:
            raise RuntimeError("sink has no active same-run extension owner")
        return self._extension_owner

    def _live_extension_capability(self):
        writer = self._writer
        if (writer is None or self._transaction_owners is None
                or self._extension_owner is None
                or writer.phase.value not in {"active", "finished"}):
            return None
        preflight = self.append_preflight
        if preflight is not None:
            consumer = preflight._consumer
            if (preflight._state is not AppendPreflightState.BOUND
                    or getattr(consumer, "__self__", None) is not self
                    or getattr(consumer, "__func__", None)
                    is not type(self)._queue_append_decision):
                return None
            intent = preflight._intent
        elif type(self.same_run_intent) is AppendIntent:
            intent = self.same_run_intent
        else:
            return None
        return self.extend_live, self.extension_owner, intent

    def _queue_append_decision(self, decision) -> None:
        self._pending_append_decision = decision

    def _apply_pending_extension(self) -> None:
        decision = self._pending_append_decision
        if decision is None:
            return
        if self._writer.phase.value == "finished":
            self._open_owned_epoch(decision)
        else:
            self._writer.extend_append(decision)
        if self._pending_append_decision is decision:
            self._pending_append_decision = None

    def _open_owned_epoch(self, decision) -> None:
        transaction_owner, target_owner, _owners = self._transaction_owners
        try:
            self._attempt = self._transaction.begin_stream_epoch(
                self._attempt,
                admission=self._transaction.admission,
                transaction_owner=transaction_owner,
                target_owner=target_owner,
                lease=self._lease,
            )
            self._writer = None
            writer = NexusRecordWriter(
                self.path,
                entry=self.entry,
                compression=self.compression,
                overwrite=False,
                atomic=False,
                flush_every=self.flush_every,
                complete_record=self.complete_record,
                source_base=self.source_base,
                file_lock=self.file_lock,
                opener=open_nexus_writer,
                transaction_binding=WriterTransactionBinding(
                    self._transaction, self._attempt, self._lease,
                ),
                append_decision=decision,
                fast_regenerable=self._fast_regenerable,
            )
            self._writer = writer
            if self._session_facade is not None:
                writer.bind_session(self._session_facade)
            writer.begin(
                metadata=self._scan.to_metadata(),
                primary_mode_1d=self._primary_mode_1d,
                primary_mode_2d=self._primary_mode_2d,
            )
        except BaseException as primary:
            try:
                self._abort_composed()
            except BaseException as cleanup:
                if self.append_preflight is not None:
                    self.append_preflight._terminal(
                        self.append_preflight._cleanup_failure_state())
                raise primary from cleanup
            raise

    def extend_live(self, owner, intent: AppendIntent):
        if owner is not self._extension_owner:
            raise RuntimeError("same-run extension requires the exact live owner")
        preflight = self.append_preflight
        prior = preflight._intent if preflight is not None else self.same_run_intent
        writer = self._writer
        if writer is None or writer.phase.value not in {
            "active", "finished",
        }:
            raise RuntimeError("same-run extension requires one owned writer")
        if preflight is not None:
            if intent == prior:
                return preflight._decision
            preflight.extend(intent)
            decision = preflight._decision
            if self._pending_append_decision is not decision:
                raise RuntimeError(
                    "bound Append preflight did not queue its exact decision"
                )
            return decision
        if prior is None:
            if not self.allow_unbound_same_run:
                raise RuntimeError("same-run extension has no bound source owner")
            if writer.phase.value != "active":
                raise RuntimeError(
                    "first same-run lineage must bind before writer finalization"
                )
            decision = begin_same_run_lineage(intent)
            if decision.disposition is not AppendDisposition.WRITE:
                raise AppendRefused(decision)
            writer.adopt_append(decision)
            self.same_run_intent = intent
            return decision
        base = self._pending_append_decision or self._epoch_decision or writer.append_decision
        decision = extend_same_run_lineage(
            base, prior, intent)
        if decision.disposition is AppendDisposition.REFUSE:
            raise AppendRefused(decision)
        if intent == prior:
            return decision
        self.same_run_intent = intent
        self._queue_append_decision(decision)
        return decision

    def write(self, frame: Frame, reduction: FrameReduction) -> None:
        return self._write_or_replace(frame, reduction, replace_existing=False)

    def write_batch(self, items: Iterable[tuple[Frame, FrameReduction]]) -> None:
        self._write_batch(tuple(items), replace_existing=False)

    def _write_or_replace(
        self, frame: Frame, reduction: FrameReduction, *, replace_existing: bool,
    ) -> None:
        dropped = self._write_batch(
            ((frame, reduction),), replace_existing=replace_existing,
        )
        return frozenset(dropped[0])

    def _write_batch(self, items: tuple[tuple, ...], *, replace_existing: bool
                     ) -> tuple[tuple[ResultMode, ...], ...]:
        if not items:
            return ()
        if replace_existing:
            self._drain_pending_record_writes(force=True)
        writer = self._writer
        if writer is None:
            raise RuntimeError("NexusSink.write called before begin().")
        self._apply_pending_extension()
        writer = self._writer
        for frame, reduction in items: self.worker_process(frame, reduction)
        prepared = tuple(self._prepare_frame_write(
            frame, reduction, replace_existing=replace_existing,
        ) for frame, reduction in items)
        if self._nexus_record_batch_size is not None and not replace_existing:
            for (_frame, reduction), (record, dropped) in zip(items, prepared):
                key = id(reduction)
                self._pending_record_writes.append((record, dropped, key))
                if self._defer_publication_drop_settlement:
                    self._deferred_publication_drops[key] = dropped
            self._drain_pending_record_writes(force=False)
            return tuple(dropped for _record, dropped in prepared)
        records = tuple(record for record, _dropped in prepared)
        if self._perf_nexus_enabled:
            started = time.perf_counter()
            try:
                writer.write_batch(records)
            finally:
                self._record_nexus_perf("write", started)
        else:
            writer.write_batch(records)
        for (frame, reduction), (_record, dropped) in zip(items, prepared):
            if self._defer_publication_drop_settlement:
                self._deferred_publication_drops[id(reduction)] = dropped
            else:
                for mode in dropped:
                    writer.drop_publication(int(frame.index), mode)
        return tuple(dropped for _record, dropped in prepared)

    def _drain_pending_record_writes(self, *, force: bool) -> None:
        size = self._nexus_record_batch_size
        writer = self._writer
        if size is None or writer is None:
            return
        while self._pending_record_writes and (
            force or len(self._pending_record_writes) >= size
        ):
            count = len(self._pending_record_writes) if force else size
            batch = tuple(self._pending_record_writes[:count])
            del self._pending_record_writes[:count]
            try:
                records = tuple(item[0] for item in batch)
                if self._perf_nexus_enabled:
                    started = time.perf_counter()
                    try:
                        writer.write_batch(records)
                    finally:
                        self._record_nexus_perf("write", started)
                else:
                    writer.write_batch(records)
                for record, dropped, key in batch:
                    if self._defer_publication_drop_settlement:
                        if record.label not in self._settled_buffered_drop_labels:
                            continue
                        self._settled_buffered_drop_labels.remove(record.label)
                        self._deferred_publication_drops.pop(key, None)
                    for mode in dropped:
                        writer.drop_publication(record.label, mode)
            except BaseException:
                abandoned = batch + tuple(self._pending_record_writes)
                self._pending_record_writes.clear()
                for record, _dropped, key in abandoned:
                    self._settled_buffered_drop_labels.discard(record.label)
                    self._deferred_publication_drops.pop(key, None)
                raise

    def _prepare_frame_write(self, frame: Frame, reduction: FrameReduction, *,
                             replace_existing: bool
                             ) -> tuple[RecordWrite, tuple[ResultMode, ...]]:
        mode_1d = str(getattr(reduction, "mode_1d", None)
                      or self._primary_mode_1d or DEFAULT_MODE_KEY)
        mode_2d = str(getattr(reduction, "mode_2d", None)
                      or self._primary_mode_2d or DEFAULT_MODE_KEY)
        result_1d = reduction.result_1d
        result_2d = reduction.result_2d
        drop_1d = result_1d is not None and not np.isfinite(
            np.asarray(result_1d.intensity, dtype=float)
        ).any()
        intensity_2d = (
            None if result_2d is None
            else np.asarray(result_2d.intensity, dtype=float)
        )
        drop_2d = result_2d is not None and (
            not np.isfinite(intensity_2d).any()
            or np.isclose(intensity_2d, -1.0, equal_nan=False).mean() >= 0.95
            or not np.isfinite(np.asarray(result_2d.radial, dtype=float)).any()
            or not np.isfinite(np.asarray(result_2d.azimuthal, dtype=float)).any()
        )
        dropped = tuple(
            mode for mode, dropped in (
                (ResultMode.one_d(mode_1d), drop_1d),
                (ResultMode.two_d(mode_2d), drop_2d),
            ) if dropped
        )
        record = self._write_frame_record(
            frame, reduction,
            result_1d=None if drop_1d else result_1d,
            result_2d=None if drop_2d else result_2d,
            mode_1d=mode_1d,
            mode_2d=mode_2d,
            replace_existing=replace_existing,
        )
        return record, dropped

    def _settle_deferred_publication_drops(
        self, frame: Frame, reduction: FrameReduction,
    ) -> None:
        key = id(reduction)
        label = int(frame.index)
        if any(item[0].label == label for item in self._pending_record_writes):
            self._deferred_publication_drops.pop(key, None)
            self._settled_buffered_drop_labels.add(label)
            return
        modes = self._deferred_publication_drops.pop(key, ())
        writer = self._writer
        if writer is None and modes:
            raise RuntimeError("deferred publication drop lost its writer")
        for mode in modes:
            writer.drop_publication(int(frame.index), mode)

    def worker_process(self, frame: Frame, reduction: FrameReduction) -> None:
        """Prepare the persisted thumbnail on the parallel reduction worker."""
        if reduction.thumbnail is not None:
            return
        prepared = self._prepare_frame_thumbnail(
            frame, corrected_image=reduction.corrected_image,
        )
        reduction.thumbnail, reduction._thumbnail_mask_baked = prepared

    def _bind_run_saturation_mask(self, state: "_RunSaturationMask") -> None:
        self._run_saturation_mask = state

    def _prepare_frame_thumbnail(
        self,
        frame: Frame,
        *,
        corrected_image: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, bool]:
        from xrd_tools.io.nexus_record import make_thumbnail_array

        if not self.write_thumbnails or frame.image is None:
            return None, False
        raw = np.asarray(frame.image)
        if corrected_image is None:
            image = np.asarray(raw, dtype=np.float32)
            background = frame.background
            if background is not None:
                bg = np.asarray(background, dtype=np.float32)
                if bg.shape == () or bg.shape == image.shape:
                    image = image - bg
        else:
            image = np.asarray(corrected_image, dtype=np.float32)
        plan = self._plan
        static_mask = None
        if plan is not None:
            static_mask = _as_bool_mask(
                plan.mask, "ReductionPlan.mask", image_shape=raw.shape,
            )
            static_mask = _combined_mask(static_mask, frame.mask, raw.shape)
        run_mask = self._run_saturation_mask
        resolved_mask = (
            run_mask.combine(static_mask, raw.shape)
            if run_mask is not None and run_mask.seeded
            else detector_value_mask(
                static_mask,
                raw,
                enabled=bool(plan is not None and plan.mask_saturation),
            )
        )
        return (
            make_thumbnail_array(
                image,
                mask_flat=(
                    None if resolved_mask is None else np.flatnonzero(resolved_mask)
                ),
                max_size=self.thumbnail_max,
            ),
            resolved_mask is not None,
        )

    def _write_frame_record(
        self,
        frame: Frame,
        reduction: FrameReduction,
        *,
        result_1d: IntegrationResult1D | None,
        result_2d: IntegrationResult2D | None,
        mode_1d: str,
        mode_2d: str,
        prepared: tuple[np.ndarray | None, bool] | None = None,
        replace_existing: bool = False,
    ) -> RecordWrite:
        thumb, mask_baked = reduction.thumbnail, reduction._thumbnail_mask_baked
        path = getattr(frame, "source_path", None)
        return RecordWrite(
            label=int(frame.index),
            result_1d=result_1d,
            result_2d=result_2d,
            mode_1d=mode_1d,
            mode_2d=mode_2d,
            thumbnail=thumb,
            thumbnail_mask_baked=mask_baked,
            mask_baked=mask_baked,
            source_path=path,
            source_frame_index=int(getattr(frame, "source_frame_index", None) or 0),
            source_snapshot=self._source_snapshots.get(str(path), {}),
            timestamp=(getattr(frame, "metadata", None) or {}).get("timestamp"),
            metadata=dict(
                getattr(reduction, "metadata", None)
                or getattr(frame, "metadata", None)
                or {}
            ),
            write_frame_record=bool(getattr(reduction, "write_frame_record", True)),
            replace_existing=replace_existing,
        )

    def replace(self, frame: Frame, reduction: FrameReduction):
        return self._write_or_replace(frame, reduction, replace_existing=True)

    def flush(self, *, force: bool = False) -> None:
        if self._writer is None:
            return
        self._apply_pending_extension()
        if self._writer.phase.value == "finished":
            return
        self._drain_pending_record_writes(force=force)
        if self._perf_nexus_enabled:
            started = time.perf_counter()
            try:
                self._writer.flush(force=force)
            finally:
                self._record_nexus_perf("flush", started)
        else:
            self._writer.flush(force=force)

    def _record_nexus_perf(self, kind: str, started: float) -> None:
        elapsed = max(0.0, time.perf_counter() - started)
        with self._perf_nexus_lock:
            if kind == "write":
                self._perf_nexus_write += elapsed
            else:
                self._perf_nexus_flush += elapsed

    def perf_snapshot(self) -> dict[str, float]:
        if not self._perf_nexus_enabled:
            return {}
        with self._perf_nexus_lock:
            return {
                "sink_nexus_write": self._perf_nexus_write,
                "sink_nexus_flush": self._perf_nexus_flush,
            }

    def _writer_finalization(self, writer: NexusRecordWriter) -> WriterFinalization:
        scan = self._scan
        plan = self._plan
        if scan is None:
            raise RuntimeError("NexusSink has no bound scan values")
        from xrd_tools import __version__ as _xrd_tools_version
        from xrd_tools.core.geometry import Diffractometer
        from xrd_tools.reduction.provenance_config import build_reduction_config

        scan_data = (
            None
            if self.incremental_finalization
            else (scan.to_scan_data() if writer.fresh else None)
        )
        if plan is None:
            config = inputs = None
        else:
            config, inputs = build_reduction_config(
                (scan, plan), include_inputs=writer.fresh
            )
            if self._run_configuration is not None:
                config["run_configuration"] = copy.deepcopy(
                    self._run_configuration,
                )
            if self._source_execution is not None:
                config["source_execution"] = copy.deepcopy(
                    self._source_execution,
                )
        geometry = scan.geometry if self.complete_record else None
        return WriterFinalization(
            scan_data=scan_data,
            frame_indices=(tuple(int(x) for x in scan.frame_indices)
                           if scan_data is not None else ()),
            geometry=geometry,
            diffractometer=(geometry if isinstance(geometry, Diffractometer)
                            else None),
            provenance_config=config,
            provenance_inputs=inputs,
            program_version=_xrd_tools_version,
            detector_calibration=(scan.extra or {}).get("detector_calibration"),
            global_mask=(scan.extra or {}).get("global_mask"),
            detector_shape=(scan.extra or {}).get("detector_shape"),
            stitched_1d=(scan.extra or {}).get("stitched_1d"),
            stitched_2d=(scan.extra or {}).get("stitched_2d"),
            stitched_provenance=(scan.extra or {}).get("stitched_provenance"),
        )

    def truncate_epoch(self, written_labels) -> None:
        writer = self._writer
        if writer is None or writer.append_decision is None:
            return
        if self.append_preflight is not None:
            self.append_preflight.truncate(written_labels)
            return
        if self.same_run_intent is None:
            raise RuntimeError("pending lineage has no same-run source owner")
        decision = self._pending_append_decision or writer.append_decision
        decision, self.same_run_intent = truncate_append_epoch(
            decision, self.same_run_intent, written_labels)
        self._queue_append_decision(decision)

    def commit_epoch(self, result: ReductionResult):
        if getattr(result, "cancelled", False):
            raise RuntimeError("a cancelled result cannot commit a live epoch")
        if self._writer is None:
            raise RuntimeError("epoch commit requires an owned Append-lineage writer")
        self._apply_pending_extension()
        self._drain_pending_record_writes(force=True)
        writer = self._writer
        if writer.phase.value == "active":
            writer.finish(self._writer_finalization(writer))
        else:
            writer.finish()
        anchor = seal_append_epoch(writer.append_decision)
        self._transaction.commit_stream_epoch(self._attempt, lease=self._lease)
        self._epoch_decision = anchor
        if self.append_preflight is not None:
            self.append_preflight._commit_epoch(anchor)
        return anchor

    def finish(self, result: ReductionResult) -> NexusTerminalResult:
        if self._terminal_result is not None:
            return self._terminal_result
        self._apply_pending_extension()
        self._drain_pending_record_writes(force=True)
        writer = self._writer
        if writer is None:
            raise RuntimeError("NexusSink finish has no transaction writer")
        if (writer.phase.value == "finished"
                and self._transaction.snapshot().phase is TransactionPhase.EPOCH_COMMITTED
                and self._pending_append_decision is None):
            self._scan = None
            self._plan = None
            return self._abort_composed()
        if getattr(result, "cancelled", False) and not writer.written_labels:
            self._scan = None
            self._plan = None
            terminal = self._abort_composed()
            if self.append_preflight is not None:
                self.append_preflight._terminal(AppendPreflightState.ABORTED)
            return terminal
        if getattr(result, "cancelled", False):
            self.truncate_epoch(writer.written_labels)
            self._apply_pending_extension()
            writer = self._writer
        if (writer.phase.value == "finished"
                and self._transaction.snapshot().phase.value == "committed"
                and self._transaction_owners is None):
            self._scan = None
            self._plan = None
            return self._typed_terminal(self._transaction.snapshot())
        # A retry resumes the writer's frozen finite state machine; rebuilding
        # provenance/metadata here would create a second authority for it.
        if writer.phase.value != "active":
            writer.finish()
            snapshot = self._transaction.commit_stream(
                self._attempt, lease=self._lease,
            )
            self._release_terminal_lease()
            if self.append_preflight is not None:
                self.append_preflight._terminal(AppendPreflightState.COMMITTED)
            self._scan = None
            self._plan = None
            return self._typed_terminal(snapshot)
        try:
            finalization = self._writer_finalization(writer)
        except BaseException as primary:
            try:
                self._abort_composed()
            except BaseException as cleanup:
                raise primary from cleanup
            raise
        writer.finish(finalization)
        snapshot = self._transaction.commit_stream(
            self._attempt, lease=self._lease,
        )
        self._release_terminal_lease()
        if self.append_preflight is not None:
            self.append_preflight._terminal(AppendPreflightState.COMMITTED)
        self._scan = None
        self._plan = None
        return self._typed_terminal(snapshot)

    def abort(self, result: ReductionResult | None) -> NexusTerminalResult | None:
        if self._terminal_result is not None:
            return self._terminal_result
        if self._nexus_record_batch_size is not None:
            self._pending_record_writes.clear()
            self._settled_buffered_drop_labels.clear()
            self._deferred_publication_drops.clear()
        self._scan = None
        self._plan = None
        if self._transaction is not None and self._transaction_owners is not None:
            try:
                return self._abort_composed()
            except BaseException:
                if self.append_preflight is not None:
                    self.append_preflight._terminal(
                        self.append_preflight._cleanup_failure_state())
                raise
        if self._transaction is None:
            preflight = self.append_preflight
            if preflight is None:
                return None
            state = preflight.snapshot.state
            if state is AppendPreflightState.RESERVED:
                state = preflight.abort().state
            elif state is AppendPreflightState.RETRYABLE:
                state = preflight.retry_cleanup().state
            if state not in {
                AppendPreflightState.NOOP,
                AppendPreflightState.COMMITTED,
                AppendPreflightState.ABORTED,
            }:
                raise RuntimeError(
                    f"Append preflight cleanup remains unsettled: {state.value}"
                )
            return None
        return self._typed_terminal(self._transaction.snapshot())


@dataclass(frozen=True, slots=True)
class BoundOutputSinkGraph:
    """One immutable dynamic-admission graph and the exact sink it executes."""

    sink: ReductionSink | None
    families: frozenset[OutputSinkKind]
    nexus_sink: NexusSink | None = None
    transactional_xye_sink: TransactionalXYESink | None = None

    def __post_init__(self) -> None:
        if (not isinstance(self.families, frozenset)
                or not all(isinstance(item, OutputSinkKind)
                           for item in self.families)):
            raise TypeError("bound sink families must be a typed frozenset")
        if self.nexus_sink is not None and type(self.nexus_sink) is not NexusSink:
            raise TypeError("bound Nexus owner must be an exact NexusSink")
        if (
            self.transactional_xye_sink is not None
            and type(self.transactional_xye_sink) is not TransactionalXYESink
        ):
            raise TypeError(
                "bound XYE owner must be an exact TransactionalXYESink"
            )


def bind_dynamic_output_sink(value: object) -> BoundOutputSinkGraph:
    """Bind the finite P0 dynamic-sink envelope to its executable snapshot.

    Arbitrary providers and delegation proxies are deliberately unsupported:
    only exact built-ins and recursively rebuilt exact ``CompositeSink`` values
    can cross this admission boundary.
    """
    active: set[int] = set()

    def bind(node: object) -> BoundOutputSinkGraph:
        if node is None:
            return BoundOutputSinkGraph(None, frozenset(), None, None)
        node_type = type(node)
        direct = {
            MemorySink: OutputSinkKind.MEMORY,
            NexusSink: OutputSinkKind.NEXUS,
            TransactionalXYESink: OutputSinkKind.XYE,
        }
        if node_type in direct:
            return BoundOutputSinkGraph(
                node, frozenset({direct[node_type]}),
                node if node_type is NexusSink else None,
                node if node_type is TransactionalXYESink else None,
            )
        if node_type is not CompositeSink:
            raise UnclassifiedOutputSinkGraph(
                f"dynamic sink {node_type.__name__} is outside the supported envelope"
            )
        identity = id(node)
        if identity in active:
            raise UnclassifiedOutputSinkGraph("dynamic sink graph contains a cycle")
        active.add(identity)
        try:
            children = node.sinks
            if type(children) is not tuple:
                raise UnclassifiedOutputSinkGraph(
                    "CompositeSink children must be an immutable tuple"
                )
            bound_children = tuple(bind(child) for child in children)
            if any(child.sink is None for child in bound_children):
                raise UnclassifiedOutputSinkGraph(
                    "CompositeSink children must be supported sink values"
                )
        finally:
            active.remove(identity)
        executable = CompositeSink(tuple(
            child.sink for child in bound_children
        ))
        nexus_children = tuple(
            child.nexus_sink for child in bound_children
            if child.nexus_sink is not None
        )
        if len(nexus_children) > 1:
            raise UnclassifiedOutputSinkGraph(
                "dynamic CompositeSink has multiple Nexus transaction owners"
            )
        xye_children = tuple(
            child.transactional_xye_sink for child in bound_children
            if child.transactional_xye_sink is not None
        )
        if len(xye_children) > 1:
            raise UnclassifiedOutputSinkGraph(
                "dynamic CompositeSink has multiple XYE transaction owners"
            )
        return BoundOutputSinkGraph(
            executable,
            frozenset().union(*(child.families for child in bound_children)),
            nexus_children[0] if nexus_children else None,
            xye_children[0] if xye_children else None,
        )

    return bind(value)


class _RunSaturationMask:
    """Session-owned, immutable detector-value mask seeded by frame one."""

    __slots__ = ("enabled", "_seeded", "_mask", "_lock")

    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._seeded = False
        self._mask: np.ndarray | None = None
        self._lock = threading.Lock()

    @property
    def seeded(self) -> bool:
        with self._lock:
            return self._seeded

    @property
    def mask(self) -> np.ndarray | None:
        with self._lock:
            return self._mask

    def seed(self, raw_image: object | None = None) -> None:
        with self._lock:
            if self._seeded:
                return
            if not self.enabled:
                self._seeded = True
                return
            if raw_image is None:
                raise RuntimeError(
                    "enabled run saturation mask requires the first raw frame"
                )
            resolved = detector_value_mask(
                None,
                np.asarray(raw_image),
                enabled=True,
            )
            if resolved is not None:
                resolved = np.array(resolved, dtype=bool, copy=True)
                resolved.setflags(write=False)
            self._mask = resolved
            self._seeded = True

    def apply(self, mask: np.ndarray | None, raw_image: object) -> np.ndarray | None:
        self.seed(raw_image)
        return self.combine(mask, np.asarray(raw_image).shape)

    def combine(
        self,
        mask: np.ndarray | None,
        image_shape: tuple[int, ...],
    ) -> np.ndarray | None:
        if len(image_shape) != 2:
            raise ValueError(f"detector image must be 2D; got shape {image_shape}")
        with self._lock:
            if not self._seeded:
                raise RuntimeError("run saturation mask used before first-frame seed")
            value_mask = self._mask
        return combine_detector_masks(mask, value_mask, image_shape)


@dataclass(slots=True)
class ReductionSession:
    """Incremental headless reduction engine for one scan/run.

    ``ReductionSession`` is the stateful counterpart to
    :func:`run_reduction`.  It owns the executor, per-thread pyFAI
    integrators, sink lifecycle, progress, and cancellation for the scan
    lifetime, so callers can feed chunks without rebuilding CSR-LUTs or
    reopening sinks every chunk.

    Streaming callers drive ``submit``/``pause``/``resume`` from one
    orchestrating thread; concurrent submitters are unsupported.  A streaming
    ``executor`` must also be asynchronous — ``submit()`` has to return before
    the submitted callable needs its decision — while its Future need expose
    only a blocking, no-argument ``result()``.  That is not structurally
    probeable, so an inline executor is unsupported for streaming (use chunked
    execution or an async adapter), not rejected up front.  An ``accept_cb``
    has shape ``(frame, publisher) -> int`` and publishes that exact attempt."""

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
    # D7: per-degradation loud/graceful policy.  Default LOUD — a headless run
    # RAISES on missing-normalization / GI-all-dummy instead of writing bad
    # data; the xdart GUI passes StrictPolicy.graceful() (never abort a save).
    strict: StrictPolicy = field(default_factory=StrictPolicy.loud)
    # H10-C1: ONE narrow synchronous acceptance-admission hook, invoked on the
    # CALLER thread inside submit() in publication order (§18.2) — Future bound,
    # the same undecided ticket queued, inventory staged, THEN this authority runs
    # and only its ACCEPTED opens the worker and writer — so an
    # accounting owner's `accepted` state is visible before any outcome, write
    # or public completion callback for that item can run.  It returns the
    # per-label attempt revision, which travels with the queued item onto every
    # FrameOutcomeReceipt.  Streaming-mode only; not an observer framework.
    # AUTHORITATIVE: publish/return one exact-int attempt; pre-publication
    # failure rejects, post-publication acceptance is final.
    accept_cb: Callable[[Frame, Callable[[int], None]], int] | None = None
    # H10-C1: ONE narrow per-item outcome receipt from the streaming writer
    # loop (see FrameOutcomeReceipt).  Not a general callback framework: a
    # single optional callable, streaming-mode only, exceptions caught+logged
    # so a listener can never kill the writer (T0-7/S1 discipline).
    outcome_cb: Callable[[FrameOutcomeReceipt], None] | None = None
    # Package-owned dynamic authorities.  Unlike observers, failures are sticky
    # and stop the write/publication path.
    outcome_authority_cb: Callable[[FrameOutcomeReceipt], None] | None = None
    written_authority_cb: Callable[[Frame, FrameReduction, int | None], None] | None = None
    batch_settled_authority_cb: Callable[[int], None] | None = None
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
    _frame_masks: dict[tuple[int, tuple[int, int]], tuple[Any, np.ndarray | None]] = field(
        default_factory=dict, init=False, repr=False,
    )
    _run_saturation_mask: _RunSaturationMask = field(init=False, repr=False)
    # S8: per-SCAN monitor warn-once state (shared with pool workers like
    # _plan_masks; set.add is GIL-atomic).  Session-owned so a dead monitor
    # warns again on the next scan and concurrent sessions don't cross-talk.
    _warned_monitor_keys: set[str] = field(
        default_factory=set, init=False, repr=False,
    )
    _products: dict[int, FrameReduction] = field(default_factory=dict, init=False, repr=False)
    _seen_idxs: set[int] = field(default_factory=set, init=False, repr=False)
    # H10-C1 §4.1: the DISTINCT labels whose top-level sink hook returned
    # successfully at least once — an identity set, not a post-sink counter.
    _written_labels: set[int] = field(default_factory=set, init=False, repr=False)
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
    _inflight: Any = field(default=None, init=False, repr=False)
    _write_queue: Any = field(default=None, init=False, repr=False)
    _writer_thread: Any = field(default=None, init=False, repr=False)
    _writer_batch_size: int = field(default=1, init=False, repr=False)
    _stream_started: bool = field(default=False, init=False, repr=False)
    _submitted: int = field(default=0, init=False, repr=False)
    # §13.2.3: ATTEMPT-axis facts for the private cancellation diagnostic —
    # `_written_attempts` counts queue items whose top-level sink hook returned
    # successfully; `_dropped_attempts` counts queue items the writer recorded
    # as cancelled-before-completion.  Same axis as `_submitted`, counted
    # directly at their exact writer-loop branches — never derived by
    # subtracting the distinct-label completion total from the attempt total.
    _written_attempts: int = field(default=0, init=False, repr=False)
    _dropped_attempts: int = field(default=0, init=False, repr=False)
    _writer_ident: int | None = field(default=None, init=False, repr=False)
    # Phase 4a: cooperative pause.  pause() quiesces the writer at a frame
    # boundary and rejects further submit()/process() until resume().
    # pause/resume/submit are called from ONE orchestrating thread (drain's
    # contract); pause is never concurrent with submit.
    _paused: bool = field(default=False, init=False, repr=False)
    _state_lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False,
    )
    _perf_quartiles_enabled: bool = field(
        default=False, init=False, repr=False,
    )
    _perf_compute_seconds: float = field(default=0.0, init=False, repr=False)
    _perf_compute_count: int = field(default=0, init=False, repr=False)
    _perf_compute_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False,
    )

    @property
    def _completed(self) -> int:
        """DERIVED completed-progress (§4.1): the number of DISTINCT labels
        whose top-level sink hook returned successfully at least once during
        the run.  There is no independently incremented post-sink counter — a
        re-fed label cannot inflate this and a later failed replacement cannot
        decrement it."""
        return len(self._written_labels)

    def _mark_cancelled(self) -> None:
        with self._state_lock:
            self._cancelled = True

    def _record_failure(self, exc: BaseException) -> None:
        with self._state_lock:
            if self._failure is None:
                self._failure = exc

    def _current_failure(self) -> BaseException | None:
        with self._state_lock:
            return self._failure

    def _is_cancelled(self) -> bool:
        with self._state_lock:
            return self._cancelled or self.cancel_token.cancelled

    def _terminal_state(self) -> tuple[bool, BaseException | None]:
        with self._state_lock:
            return (
                self._cancelled or self.cancel_token.cancelled,
                self._failure,
            )

    @property
    def sink_terminal_safe(self) -> bool:
        """Whether no timed-out writer can still invoke the sink."""
        return self._writer_thread is None or not self._writer_thread.is_alive()

    def _shutdown_worker(self, *, wait: bool = True) -> None:
        if self._owns_worker and self._worker is not None:
            _admission_trace("pool_shutdown_enter", wait=wait, owned=True)
            self._worker.shutdown(wait=wait, cancel_futures=True)
            _admission_trace("pool_shutdown_exit", owned=True)
        self._worker = None

    def _rollback_construction(self, primary: BaseException) -> None:
        self._record_failure(primary)
        try:
            if self._started:
                self.finish(raise_on_failure=False)
        finally:
            self._shutdown_worker()
            self._finished = True

    def __post_init__(self) -> None:
        self._perf_quartiles_enabled = (
            os.environ.get("XDART_PERF_QUARTILES", "").strip() == "1"
        )
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0; got {self.chunk_size}")
        self.scan = _coerce_to_scan(self.source)
        self._sink = _coerce_sink(self.sink)
        self._run_saturation_mask = _RunSaturationMask(self.plan.mask_saturation)
        bind_run_mask = getattr(self._sink, "_bind_run_saturation_mask", None)
        if callable(bind_run_mask):
            bind_run_mask(self._run_saturation_mask)
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
        # Validate BEFORE acquiring resources: sink.begin() opens an h5 handle
        # (atomic NexusSink also creates its hidden .tmp) and _coerce_executor
        # may build an owned pool — a ValueError after those leaks both, with
        # no abort path that ever cleans the orphaned tmp.
        if self.execution not in ("chunked", "streaming"):
            raise ValueError(
                f"execution must be 'chunked' or 'streaming'; got {self.execution!r}"
            )
        try:
            self._worker, self._owns_worker = _coerce_executor(self.executor)
            self._sink.begin(self.scan, self.plan)
            self._started = True
            _emit(self.progress_cb, self.scan.name, "start", None, 0, len(self.scan))
            if self.execution == "streaming":
                self._init_streaming()
        except BaseException as primary:
            try:
                self._rollback_construction(primary)
            except BaseException as cleanup:
                raise primary from cleanup
            raise

    def __enter__(self) -> ReductionSession:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            # A body exception is already propagating — finish for cleanup but
            # do NOT raise (that would mask the original exception).
            self._record_failure(exc)
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

    @property
    def saturation_mask_seeded(self) -> bool:
        return self._run_saturation_mask.seeded

    @property
    def saturation_mask(self) -> np.ndarray | None:
        return self._run_saturation_mask.mask

    def perf_snapshot(self) -> dict[str, float]:
        """Return aggregate worker compute timing without per-frame history."""
        if not self._perf_quartiles_enabled:
            return {}
        with self._perf_compute_lock:
            return {
                "reducer_compute": float(self._perf_compute_seconds),
                "reducer_compute_count": float(self._perf_compute_count),
            }

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
        if self._paused:
            raise RuntimeError(
                "ReductionSession.process called while paused; call resume() first"
            )
        if self._is_cancelled():
            self._mark_cancelled()
            return

        try:
            if frames_or_chunk is None:
                for chunk, chunk_images in _iter_reduction_chunks(
                    self.source, self.scan, self.chunk_size,
                ):
                    self._process_chunk(chunk, chunk_images)
                    if self._is_cancelled():
                        break
                return

            chunk, chunk_images = self._normalize_process_input(frames_or_chunk, images)
            self._register_process_frames(chunk)
            self._process_chunk(chunk, chunk_images)
        except BaseException as exc:
            self._record_failure(exc)
            raise

    def _init_streaming(self) -> None:
        """Set up the bounded in-flight window + single writer thread.

        Reuses the persistent pool (``self._worker``) and the per-thread
        integrator provider; only adds one identity-membership capacity owner,
        a FIFO queue, and one consumer thread.  Called once from ``__post_init__``
        AFTER the executor + integrators + GI freeze are in place, so the freeze
        (which needs first+last frames) is fixed before any frame is submitted.
        """
        if self._worker is None:
            # Streaming needs a real pool; build a default owned one when the
            # caller passed executor=None/False.  MEM-3: cap it at the RAM-aware
            # throughput knee (~4) instead of Python's ``min(32, cpu+4)`` default
            # — each worker deep-copies the integrator, so an unbounded default
            # pool duplicated the geometry ~20x for no throughput gain.
            from xrd_tools.core import reduction_worker_cap
            self._worker = ThreadPoolExecutor(max_workers=reduction_worker_cap())
            self._owns_worker = True
        n_workers = getattr(self._worker, "_max_workers", None) or 4
        bound = (
            self.inflight_max
            if self.inflight_max and self.inflight_max > 0
            else max(2, 2 * n_workers)
        )
        self.inflight_max = bound
        self._writer_batch_size = min(_sink_writer_batch_size(self._sink), bound)
        self._inflight = _InFlightWindow(bound)
        self._write_queue = queue.Queue()
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name=f"reduction-writer-{self.scan.name}",
            daemon=True,
        )
        self._writer_thread.start()
        self._stream_started = True

    def submit(self, frame: Frame, image: np.ndarray | None = None) -> bool:
        """Stream one frame into the pool immediately (execution="streaming").

        Blocks when ``inflight_max`` frames are already in flight (bounded
        memory), then dispatches integration to the persistent pool and hands
        the ``(frame, future)`` to the single writer thread, which writes it to
        the sink by frame index once it completes.  Out-of-order completion is
        fine — the sink/writer is index-addressed.

        Returns ``True`` when the frame was ACCEPTED (registered in the scan
        inventory, dispatched to the pool, and handed to the writer); ``False``
        when it was DROPPED without being submitted because the session was
        cancelled, or the writer died, while waiting for a worker slot.  A
        dropped frame is NOT registered and does NOT advance ``frames_submitted``
        — so a Stop racing a ``submit`` can't leave a phantom frame the caller
        believes was processed (the accepted-vs-cancelled state leak).  The drop
        paths RETURN (never raise) so they don't escape the caller's ``run()``
        loop and tear the QThread down (the GIFreezeError trap); caller-contract
        violations (after ``finish()``, after a recorded failure, while paused)
        still RAISE.  An acceptance-admission failure (``accept_cb`` raising)
        also RAISES loudly: pre-proof facts roll back; published acceptance is
        final and wakes under cancellation.

        One frame is ONE publication transaction owned by a single private
        ticket (§18.2).  The work is dispatched first, but the ticket keeps it
        — and the writer — gated until this call publishes the item's one
        decision, and the authority runs only once a Future exists, the same
        PENDING ticket is queued and the inventory is staged, so acceptance
        stays exactly where §4.1 froze it.  Every failure before the authority
        publishes rejects that ticket with no Future method or public effect;
        inventory, submitted count and membership restore.  Afterwards an
        interrupt fails/cancels and completes the SAME immutable receipt wake.
        ``accept_cb(frame, publish_acceptance)`` must publish and return the
        same positive exact int.  Called from one orchestrating thread
        (as are ``pause``/``resume``); a streaming executor must be
        asynchronous — ``submit()`` returns before the callable needs its
        decision."""
        if self.execution != "streaming":
            raise RuntimeError("submit() requires execution='streaming'")
        if self._finished:
            raise RuntimeError("ReductionSession.submit called after finish().")
        failure = self._current_failure()
        if failure is not None:
            raise failure
        if self._paused:
            raise RuntimeError(
                "ReductionSession.submit called while paused; call resume() first"
            )
        if self._is_cancelled():
            self._mark_cancelled()
            return False
        label = int(frame.index)
        ticket = _StreamPublication(frame)
        try:
            while not self._inflight.try_acquire(ticket, 0.1):
                if self._is_cancelled():
                    self._mark_cancelled()
                    return False
                if (self._writer_thread is not None
                        and not self._writer_thread.is_alive()):
                    self._record_failure(RuntimeError(
                        "ReductionSession writer thread died; run cannot proceed"
                    ))
                    self._mark_cancelled()
                    return False
            if self._is_cancelled():
                self._mark_cancelled()
                self._inflight.release(ticket)
                return False
            _admission_trace("permit_acquired", label=label)
            image = self._prime_saturation_mask(frame, image)
            ticket.future = self._worker.submit(
                self._ticketed_stream_reduce, ticket, image)
            _admission_trace("future_bound", label=label)
            self._write_queue.put(ticket)
            _admission_trace("queue_published", label=label)
            self._stage_publication(ticket)
            _admission_trace("inventory_staged", label=label)
            self._emit_accepted(frame, ticket.store_accepted)
            receipt = ticket.decision
            if receipt is None or receipt[0] == _TICKET_REJECTED:
                raise RuntimeError(
                    "acceptance authority returned without an ACCEPTED receipt")
            ticket.complete_wake()
        except BaseException as exc:
            receipt = ticket.decision
            if receipt is not None and receipt[0] == _TICKET_ACCEPTED:
                self._record_failure(exc)
                self._mark_cancelled()
                self.cancel_token.cancel()
                ticket.complete_wake()
            else:
                self._reject_publication(ticket, exc, label)
            raise
        return True

    def _reject_publication(self, ticket: _StreamPublication,
                            exc: BaseException, label: int) -> None:
        """Undo one UNACCEPTED submission, invariant-first (§18.3).

        Ordered so the run's lifecycle is consistent before anything that can
        raise on its own: publish the shared REJECTED decision (waking a worker
        or writer parked on it), restore the staged inventory/submitted facts
        exactly, make the original the run's sticky failure and cancel, remove
        capacity membership exactly once — and only THEN write the diagnostic.  Trace
        I/O may therefore change WHICH exception reaches the caller (an
        interrupt supersedes the stored original), not what the run recorded."""
        ticket.reject_and_wake()
        undo = ticket.take_rejection_undo()
        if undo is not None:
            undo()
        self._record_failure(exc)
        self._mark_cancelled()
        self._inflight.release(ticket)
        _admission_trace("submit_rejected", label=label, error=exc)

    def _stage_publication(self, ticket: _StreamPublication) -> None:
        """Stage this item's reversible facts — the private submitted-attempt
        count and the scan inventory — arming the EXACT undo on the ticket
        BEFORE any mutation, so even a staging failure landing mid-mutation
        restores frame order, ``_frame_by_index``, the position map, prior
        object identity and the submitted count.  Registration goes through the
        same :meth:`_register_process_frames` the chunked path uses, so fresh
        and replacement semantics cannot drift; only an out-of-order fresh
        label (the one re-sorting case) pays an O(n) snapshot.  ``submit`` runs
        on one orchestrating thread, so absolute restores are exact."""
        frame = ticket.frame
        frames = self.scan.frames
        positions = self._scan_frame_positions
        by_index = self.scan._frame_by_index
        idx = int(frame.index)
        pos = positions.get(idx)
        prior_frame = frames[pos] if pos is not None else None
        had_entry = idx in by_index
        prior_entry = by_index.get(idx)
        prior_submitted = self._submitted
        resort = (list(frames), dict(positions)) if (
            pos is None and frames and idx < int(frames[-1].index)) else None

        def _undo() -> None:                    # armed BEFORE any mutation
            self._submitted = prior_submitted
            if resort is not None:              # staging rebuilt the ordering
                frames[:] = resort[0]
                positions.clear()
                positions.update(resort[1])
            elif pos is None:                   # appended in order
                if frames and frames[-1] is frame:
                    frames.pop()
                positions.pop(idx, None)
            else:                               # replaced in place
                frames[pos] = prior_frame
            if had_entry:
                by_index[idx] = prior_entry
            else:
                by_index.pop(idx, None)

        ticket.unstage = _undo
        self._submitted += 1
        self._register_process_frames([frame])

    def _ticketed_stream_reduce(self, ticket: _StreamPublication,
                                image: np.ndarray | None):
        """The dispatched worker callable (§18.2): observe this item's ONE
        publication decision, then run the unchanged :meth:`_stream_reduce`.
        A REJECTED item exits here — before reduction, ``worker_process``
        prep or any public effect — including when a hostile executor scheduled
        it and then failed the caller's ``submit``."""
        label = int(ticket.frame.index)
        _admission_trace("submitted_callable_entered", label=label)
        state, _attempt = ticket.await_decision()
        if state == _TICKET_REJECTED:
            _admission_trace("callable_unaccepted", label=label, state=state)
            raise _ReductionCancelled(
                f"frame {label} was never admitted; reduction not started"
            )
        _admission_trace("ticket_accepted", label=label)
        return self._stream_reduce(ticket.frame, image)

    def _stream_reduce(self, frame: Frame, image: np.ndarray | None):
        """Worker-thread task: integrate, then run the sink's per-frame
        ``worker_process`` hook (if any) so expensive per-frame prep — e.g.
        xdart's thumbnail + raw-free — happens in PARALLEL across the pool
        rather than serially on the single writer thread.  The writer then only
        does the index-addressed HDF5 write.  Cancellation/errors from the
        REDUCTION propagate through the future to the writer loop unchanged.

        Returns ``(reduction, prep_error)``.  A ``worker_process`` failure
        happens AFTER a valid typed result exists, so it is carried back beside
        that result instead of destroying it: the writer still reports the
        typed COMPLETED outcome with its exact produced modes, then records the
        prep failure on the fail-loud run path and writes nothing.
        """
        label = int(frame.index)
        worker_process = getattr(self._sink, "worker_process", None)
        _admission_trace("reduction_begin", label=label)
        compute_started = (
            time.perf_counter() if self._perf_quartiles_enabled else 0.0
        )
        try:
            reduction = _reduce_frame(
                frame, image, self.plan, self._integrators, self._plan_masks,
                self._frame_masks,
                self.cancel_token, self._warned_monitor_keys,
                include_corrected_image=callable(worker_process),
                run_saturation_mask=self._run_saturation_mask,
                strict=self.strict,
            )
        finally:
            if self._perf_quartiles_enabled:
                elapsed = max(0.0, time.perf_counter() - compute_started)
                with self._perf_compute_lock:
                    self._perf_compute_seconds += elapsed
                    self._perf_compute_count += 1
        _admission_trace("reduction_end", label=label)
        prep_error: BaseException | None = None
        try:
            if callable(worker_process):
                _admission_trace("worker_process_begin", label=label)
                worker_process(frame, reduction)
                _admission_trace("worker_process_end", label=label)
        except BaseException as exc:
            prep_error = exc
        finally:
            reduction.corrected_image = None
        return reduction, prep_error

    def _emit_accepted(self, frame: Frame, publish_acceptance:
                       Callable[[int | None], None]) -> int | None:
        """Run the two-argument authority: publish/return one positive exact
        int; no callback is raw ``ACCEPTED(None)`` or retried."""
        cb = self.accept_cb
        if cb is None:
            publish_acceptance(None)
            return None
        published: object = _MISSING

        def _publish(attempt: int) -> None:
            nonlocal published
            _require_positive_exact_int(attempt, "published attempt")
            publish_acceptance(attempt)
            published = attempt

        attempt = cb(frame, _publish)
        _require_positive_exact_int(attempt, "returned attempt")
        if published is _MISSING:
            raise RuntimeError("acceptance authority returned without a receipt")
        if attempt != published:
            raise RuntimeError(f"published {published!r}, returned {attempt!r}")
        return attempt

    def _emit_outcome(self, frame_index: int, outcome: FrameOutcome, *,
                      replacing: bool, reduction: FrameReduction | None = None,
                      error: str | None = None,
                      attempt: int | None = None) -> None:
        """Deliver one :class:`FrameOutcomeReceipt` to ``outcome_cb`` (writer
        thread).  A listener exception is caught + logged — it must never
        escape the writer loop (T0-7/S1)."""
        receipt = FrameOutcomeReceipt(
            frame_index=int(frame_index),
            outcome=outcome,
            replacing=bool(replacing),
            produced_1d=getattr(reduction, "result_1d", None) is not None,
            produced_2d=getattr(reduction, "result_2d", None) is not None,
            error=error,
            attempt=attempt,
        )
        authority = self.outcome_authority_cb
        if authority is not None:
            authority(receipt)
        cb = self.outcome_cb
        if cb is None:
            return
        try:
            cb(receipt)
        except Exception:
            logger.exception("ReductionSession outcome_cb listener raised")

    def _resolve_writer_ticket(self, ticket: "_StreamPublication") -> tuple:
        idx = int(ticket.frame.index)
        state, attempt = ticket.await_decision()
        if state == _TICKET_REJECTED:
            return "rejected", ticket, attempt, None, None
        _admission_trace("writer_item_dequeued", label=idx, attempt=attempt)
        try:
            reduction, prep_error = ticket.future.result()
        except _ReductionCancelled as exc:
            return "cancelled", ticket, attempt, None, exc
        except BaseException as exc:
            return "failed", ticket, attempt, None, exc
        if prep_error is not None:
            return "prep_failed", ticket, attempt, reduction, prep_error
        return "ready", ticket, attempt, reduction, None

    def _settle_writer_barrier(self, resolved: tuple) -> None:
        kind, ticket, attempt, reduction, error = resolved
        frame, idx = ticket.frame, int(ticket.frame.index)
        replacing = idx in self._seen_idxs
        try:
            if kind == "rejected":
                _admission_trace("writer_dropped_unaccepted", label=idx)
                return
            if kind == "cancelled":
                self._dropped_attempts += 1
                outcome = FrameOutcome.CANCELLED_BEFORE_COMPLETION
            else:
                outcome = FrameOutcome.FAILED if kind == "failed" else FrameOutcome.COMPLETED
            emitted = True
            try:
                self._emit_outcome(
                    idx, outcome, replacing=replacing, reduction=reduction,
                    error=(f"{type(error).__name__}: {error}"
                           if kind == "failed" else None), attempt=attempt)
            except BaseException as exc:
                emitted = False
                self._record_failure(exc)
            if kind == "cancelled":
                self._mark_cancelled()
            elif kind == "failed" or (kind == "prep_failed" and emitted):
                self._record_failure(error)
            if self.clear_frame_images:
                frame.image = None
                frame.background = None
        finally:
            self._complete_stream_publication(ticket)

    def _settle_writer_batch(self, batch: tuple[tuple, ...]) -> None:
        outcomes_ok = True
        for ticket, attempt, reduction, replacing in batch:
            frame = ticket.frame
            try:
                self._emit_outcome(
                    int(frame.index), FrameOutcome.COMPLETED,
                    replacing=replacing, reduction=reduction, attempt=attempt)
            except BaseException as exc:
                self._record_failure(exc)
                outcomes_ok = False
                if self.clear_frame_images:
                    frame.image = None
                    frame.background = None
        try:
            if not outcomes_ok:
                return
            items = tuple((item[0].frame, item[2]) for item in batch)
            try:
                if batch[0][3]:
                    _emit_sink_replace(self._sink, *items[0])
                elif self._writer_batch_size == 1:
                    self._sink.write(*items[0])
                else:
                    _emit_sink_write_batch(self._sink, items)
            except BaseException as exc:
                self._record_failure(exc)
                return
            batch_settled = True
            for ticket, attempt, reduction, _replacing in batch:
                frame = ticket.frame
                idx = int(frame.index)
                try:
                    self._seen_idxs.add(idx)
                    if self.retain_products:
                        self._products[idx] = reduction
                    if self.written_authority_cb is not None:
                        self.written_authority_cb(frame, reduction, attempt)
                    settle = getattr(
                        self._sink, "_settle_deferred_publication_drops", None,
                    )
                    if callable(settle):
                        settle(frame, reduction)
                    self._written_labels.add(idx)
                    self._written_attempts += 1
                    post_write = getattr(self._sink, "_post_write", None)
                    if callable(post_write):
                        post_write(frame, reduction)
                except BaseException as exc:
                    self._record_failure(exc)
                    batch_settled = False
                else:
                    try:
                        _emit(self.progress_cb, self.scan.name, "write",
                              frame.index, self._completed, len(self.scan))
                        if self.clear_frame_images:
                            frame.image = None
                            frame.background = None
                            _clear_source_frame_image(self.source, frame.index)
                    except BaseException as exc:
                        self._record_failure(exc)
                        batch_settled = False
            authority = self.batch_settled_authority_cb
            if batch_settled and authority is not None:
                try:
                    authority(len(batch))
                except BaseException as exc:
                    self._record_failure(exc)
        finally:
            for ticket, _attempt, _reduction, _replacing in batch:
                self._complete_stream_publication(ticket)

    def _writer_loop(self) -> None:
        """Drain queued tickets on the single existing writer lane."""
        self._writer_ident = threading.get_ident()
        pending_ticket = None
        pending_resolved = None
        while True:
            if pending_resolved is not None:
                resolved, pending_resolved = pending_resolved, None
            else:
                item = pending_ticket or self._write_queue.get()
                pending_ticket = None
                if item is _STREAM_SENTINEL:
                    _admission_trace("writer_sentinel_exit")
                    self._write_queue.task_done()
                    break
                resolved = self._resolve_writer_ticket(item)
            if resolved[0] != "ready":
                self._settle_writer_barrier(resolved)
                continue
            _kind, ticket, attempt, reduction, _error = resolved
            idx = int(ticket.frame.index)
            replacing = idx in self._seen_idxs
            if replacing:
                self._settle_writer_batch(((ticket, attempt, reduction, True),))
                continue
            batch = [(ticket, attempt, reduction, False)]
            batch_labels = {idx}
            stop_after_batch = False
            while len(batch) < self._writer_batch_size:
                try:
                    item = self._write_queue.get_nowait()
                except queue.Empty:
                    break
                if item is _STREAM_SENTINEL:
                    stop_after_batch = True
                    break
                next_idx = int(item.frame.index)
                if (item.decision is None or next_idx in self._seen_idxs or
                        next_idx in batch_labels):
                    pending_ticket = item
                    break
                candidate = self._resolve_writer_ticket(item)
                if candidate[0] != "ready":
                    pending_resolved = candidate
                    break
                _, next_ticket, next_attempt, next_reduction, _ = candidate
                batch.append((next_ticket, next_attempt, next_reduction, False))
                batch_labels.add(next_idx)
            self._settle_writer_batch(tuple(batch))
            if stop_after_batch:
                _admission_trace("writer_sentinel_exit")
                self._write_queue.task_done()
                break

    def _complete_stream_publication(self, ticket: _StreamPublication) -> None:
        try:
            try:
                self._inflight.release(ticket)
            except BaseException as exc:
                self._record_failure(exc)
        finally:
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

    def pause(self, timeout: float | None = None) -> bool:
        """Quiesce the writer at a frame boundary; reject submits until resume.

        Phase 4a — the pause-safe guarantee the GUI's ``_enter_pause`` hand-
        rolls today: sets :attr:`is_paused`, then :meth:`drain`\\ s the in-flight
        window so the single writer thread is provably idle (streaming).  The
        caller may then flush its sink / browse without racing a write.

        Returns whether the writer fully quiesced within ``timeout`` (``True``),
        or timed out / was cancelled (``False`` — unflushed frames remain and
        flush on :meth:`resume`/:meth:`finish`; RS-1 tolerance preserved).
        Idempotent; a no-op returning ``True`` once finished or cancelled (a
        cancelled session is never marked paused).  Chunked execution has no
        in-flight window, so this only sets the flag (drain is a no-op).

        Called from the SAME thread as :meth:`submit` (cooperative; never
        concurrent with it)."""
        if self._finished or self._is_cancelled():
            return True
        self._paused = True
        return self.drain(timeout=timeout)

    def resume(self) -> None:
        """Re-allow :meth:`submit` / :meth:`process` after :meth:`pause`.

        No-op if not paused or already finished."""
        if not self._finished:
            self._paused = False

    @property
    def is_paused(self) -> bool:
        """True iff paused and not yet finished."""
        return self._paused and not self._finished

    @property
    def is_running(self) -> bool:
        """True iff the session is active — begun (the sink is open) and not
        finished/cancelled.  ``_started`` is set at construction for both
        execution modes, so this reads as running across chunked and
        streaming runs (the GUI run-state seam, Phase 4d)."""
        return (self._started
                and not (self._finished or self._is_cancelled()))

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
                _admission_trace("writer_join_begin", timeout=join_timeout)
                self._writer_thread.join(timeout=join_timeout)
                _admission_trace("writer_join_end",
                                 alive=self._writer_thread.is_alive())
                if self._writer_thread.is_alive():
                    # Writer is still alive after the timeout (a stalled worker
                    # held the future.result() call and the sentinel hasn't been
                    # processed).  Cancel any remaining in-flight work via the
                    # cancel token so the worker unblocks at its next check, and
                    # flag the result as failed so the caller gets a loud error
                    # rather than a silent hang.
                    self.cancel_token.cancel()
                    self._mark_cancelled()
                    self._record_failure(TimeoutError(
                            f"ReductionSession.finish(): writer thread did not "
                            f"exit within {join_timeout}s; a worker may be "
                            f"stalled (stalled NFS read or runaway pyFAI call). "
                            f"Session result is incomplete."
                    ))
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

        cancelled, failure = self._terminal_state()
        self.result = ReductionResult(
            scan_name=self.scan.name,
            frames=self._products,
            n_processed=self._completed,
            cancelled=cancelled,
            output_path=self._output_path,
            failed=failure is not None,
            error=None if failure is None else str(failure),
        )
        # §13.2.3: the drop count is the DIRECT attempt-keyed tally from the
        # writer loop's cancelled-before-completion branch — never derived by
        # subtracting a distinct-label total from an attempt total (two
        # successful writes of one label would fabricate a "dropped" item).
        # All three logged quantities share the attempt axis.  Private
        # diagnostic only; public projections stay distinct-label facts.
        dropped = int(self._dropped_attempts)
        if cancelled and dropped:
            logger.info(
                "cancelled: %d attempts submitted, %d written, %d dropped in-flight",
                self._submitted, self._written_attempts, dropped,
            )
        try:
            if self._started:
                if _writer_timed_out:
                    # T0-5: a live writer may still call sink.write(); closing
                    # its HDF5 handle would race that write.  Leave the exact
                    # sink untouched and report its unfinalized data location.
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
                elif failure is None:
                    _admission_trace("sink_finish_begin")
                    self._sink.finish(self.result)
                    _admission_trace("sink_finish_end")
                else:
                    abort = getattr(self._sink, "abort", None)
                    if callable(abort):
                        _admission_trace("sink_abort_begin")
                        abort(self.result)
                        _admission_trace("sink_abort_end")
                    else:
                        _admission_trace("sink_finish_begin", on_failure=True)
                        self._sink.finish(self.result)
                        _admission_trace("sink_finish_end", on_failure=True)
        finally:
            # A timed-out writer can own stalled futures; never re-hang here.
            self._shutdown_worker(wait=not _writer_timed_out)
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
        failure = self._current_failure()
        if raise_on_failure and failure is not None:
            raise failure
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
            warned_monitor_keys=self._warned_monitor_keys,
            run_saturation_mask=self._run_saturation_mask,
        )
        self._gi_freeze_applied = True

    def _prime_saturation_mask(
        self,
        frame: Frame,
        image: np.ndarray | None,
    ) -> np.ndarray | None:
        """Seed once from the first image already entering this session."""
        if self._run_saturation_mask.seeded:
            return image
        if not self._run_saturation_mask.enabled:
            self._run_saturation_mask.seed()
            return image
        raw = np.asarray(image) if image is not None else np.asarray(frame.load_image())
        self._run_saturation_mask.seed(raw)
        return raw

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
                self._mark_cancelled()
                break
            raw_image = self._prime_saturation_mask(frame, raw_image)
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
                        self._frame_masks,
                        cancel_token=self.cancel_token,
                        warned_monitor_keys=self._warned_monitor_keys,
                        run_saturation_mask=self._run_saturation_mask,
                        strict=self.strict,
                    )
                except _ReductionCancelled:
                    self._mark_cancelled()
                    break
                except BaseException as exc:
                    # Per-frame integration failure (incl. a loud StrictnessError):
                    # record + SKIP this frame, never abort the whole chunk — mirror
                    # the streaming submit path so chunked is symmetric and honours
                    # "reject per frame, never abort a whole-scan save".  The other
                    # frames still process; finish() re-raises once (fail-loud).
                    self._record_failure(exc)
                    continue
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
                        self._frame_masks,
                        self.cancel_token,
                        self._warned_monitor_keys,
                        run_saturation_mask=self._run_saturation_mask,
                        strict=self.strict,
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
                    self._mark_cancelled()
                    _cancel_pending_futures(pending[pos + 1:], worker=self._worker)
                    break
                except BaseException as exc:
                    # Per-frame integration failure (incl. a loud StrictnessError)
                    # surfacing from the worker future: record + SKIP, never abort
                    # the chunk — symmetric with the streaming submit path.  The
                    # outer BaseException handler still covers sink/IO failures.
                    self._record_failure(exc)
                    continue
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
                # The top-level sink hook returned: record the DISTINCT label
                # identity (§4.1) — a re-fed index never double-counts — plus
                # the attempt-axis write fact (kept exact in both modes).
                self._written_labels.add(idx)
                self._written_attempts += 1
                _emit(self.progress_cb, self.scan.name, "write", frame.index, self._completed, len(self.scan))
                if self.clear_frame_images:
                    frame.image = None
                    frame.background = None
                    _clear_source_frame_image(self.source, frame.index)
                if self.cancel_token.cancelled:
                    self._mark_cancelled()
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
                # D6: cancel() cannot stop a future that is already RUNNING;
                # that worker re-pins frame.image (top of _reduce_frame)
                # AFTER an immediate clear, leaving the raw held until
                # session close.  Order the clear AFTER the running tail has
                # finished -- pyFAI integrations terminate, and the
                # cancelled-before-start futures resolve instantly, so this
                # wait is bounded by the in-flight tail of one chunk.
                _wait_pending_futures(pending[pos + 1:], worker=self._worker)
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
    clear_frame_images: bool | None = None,
    progress_cb: ProgressCallback | None = None,
    cancel_token: CancelToken | None = None,
    executor: Any | None = None,
    gi_freeze_mode: str | None = None,
    execution: str | None = None,
    inflight_max: int | None = None,
    retain_products: bool | None = None,
    strict: StrictPolicy | None = None,
) -> ReductionResult:
    """Run a headless reduction job over all frames in ``scan`` or a source.

    This one-orchestrating-thread owner uses raw ``ACCEPTED(None)``; direct
    sessions may supply ``accept_cb(frame, publish_acceptance)``.

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
        Set each frame's cached image/background to ``None`` after writing to
        the sink.  ``None`` (default) auto-selects ``True`` for streaming
        durable-sink runs and ``False`` otherwise.  Cheap memory bound for
        long lazy-loaded scans.
    progress_cb
        Called as ``cb(ReductionProgress)`` after every stage.
    cancel_token
        Polled per frame; cancellation stops at the next frame
        boundary (pyFAI doesn't yield mid-integration).
    executor
        Optional execution policy for per-frame work inside each chunk.  Pass
        an executor with ``submit()``, ``True`` for a default
        :class:`ThreadPoolExecutor`, or an integer worker count.  Sink writes
        remain ordered on the caller thread.  A returned Future need expose
        only a blocking, no-argument ``result()``; a ``"streaming"`` executor
        must also be asynchronous (``submit()`` returns before the submitted
        callable needs its decision), driven from one orchestrating thread.
    gi_freeze_mode
        Optional grazing-incidence common-grid freeze policy.  ``"first_frame"``
        scouts the first frame; ``"scout_union"`` scouts first+last (or
        ``plan.extra["gi_freeze_scout_indices"]``) and freezes the missing
        output-axis ranges before the main reduction.  Explicit caller ranges
        are preserved.
    execution
        ``None`` (default) auto-selects ``"streaming"`` for durable sinks such
        as :class:`NexusSink` / :class:`XYESink`, and ``"chunked"`` for
        in-memory/no-sink calls.  Pass ``"chunked"`` or ``"streaming"``
        explicitly to override.  Streaming submits each frame to a bounded
        in-flight window drained by one writer thread (out-of-order completion,
        single-writer sink) — the same engine xdart's GUI uses by default,
        exposed here so notebook/headless callers get it without hand-driving
        :class:`ReductionSession`.
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
    strict
        :class:`~xrd_tools.core.strictness.StrictPolicy` — whether per-frame
        degradations (missing normalization, an all-dummy 2D integration) RAISE
        or degrade.  ``None`` (default) = :meth:`StrictPolicy.loud`: a headless
        run fails loudly instead of persisting bad data.  Pass
        ``StrictPolicy.graceful()`` for the GUI's never-abort behavior.
    """
    sink_obj = _coerce_sink(sink)
    durable_sink = not _sink_is_memory_only(sink_obj)
    if execution is None:
        execution = "streaming" if durable_sink else "chunked"
    if clear_frame_images is None:
        clear_frame_images = execution == "streaming" and durable_sink
    if retain_products is None:
        retain_products = not (execution == "streaming" and durable_sink)
    with ReductionSession(
        plan,
        scan,
        sink_obj,
        chunk_size=chunk_size,
        clear_frame_images=clear_frame_images,
        progress_cb=progress_cb,
        cancel_token=cancel_token,
        executor=executor,
        gi_freeze_mode=gi_freeze_mode,
        execution=execution,
        inflight_max=inflight_max,
        retain_products=retain_products,
        strict=strict if strict is not None else StrictPolicy.loud(),
    ) as session:
        if execution == "streaming":
            # Streaming drains via submit() (process() is rejected); feed every
            # frame, then finish() joins the writer and flushes.  submit()
            # returns False when it drops a frame (cancel / writer-death mid-
            # wait) — stop feeding promptly rather than spin the remaining frames
            # against a cancelled session.
            #
            # SOURCE-FED input seam (R2-R1): the pixels come from the ORIGINAL
            # FrameSource's sustained cursor via ``iter_chunks`` (ONE open handle,
            # native-dtype reads) on THIS producer thread, and are handed to
            # ``submit(frame, image)`` as decoded numpy arrays — so no live h5py/
            # fabio handle ever crosses into a reduction worker and the public
            # ``open_source(...) -> run_reduction(...)`` path never reopens the
            # file per frame.  The in-flight window bounds memory, so
            # the cursor cannot outrun the workers.  A plain ``Scan`` source
            # yields ``None`` images here and keeps its existing per-frame
            # (in-memory / lazy) load behavior unchanged.
            stop = False
            for chunk_frames, chunk_images in _iter_reduction_chunks(
                    session.source, session.scan, chunk_size):
                for frame, image in zip(chunk_frames, chunk_images):
                    if session.cancel_token.cancelled:
                        stop = True
                        break
                    if not session.submit(frame, image):
                        stop = True
                        break
                if stop:
                    break
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


@dataclass(frozen=True, slots=True)
class PrepareDiagnostics:
    """Outcome of the whole-scan GI prepare pass (ADR-0006).

    ``status``:
      - ``"frozen"`` — scout indices pinned into ``plan.extra``; the downstream
        freeze (``ReductionSession(..., gi_freeze_mode="scout_union")`` or
        xdart's adapter) unions over them.
      - ``"skip"`` — nothing to do: non-GI plan, GI ranges already pinned, fixed/
        manual incidence, ``<2`` frames, or a single distinct incidence.
      - ``"unverifiable"`` — the whole-scan extent could NOT be established (the
        source can't be cheaply swept, or ``<2`` readable incidences): the caller
        WARNS and proceeds on the chunk/first-frame freeze (T0-4 policy).

    ``scout_metadata`` carries the resolved (read-only) metadata of the extreme
    frames as provenance — it is NOT a loadable source ref.  To LOAD a scout
    image, use ``scout_indices`` against the same source:
    ``source.frame_for(idx)`` (a lazy ``ScanFrame`` with ``source_path`` /
    ``source_frame_index`` / loader) or ``source.load_frame(idx)`` — no
    re-enumeration needed (the source you passed to ``prepare_gi_freeze`` is the
    loader).  Deeply immutable so a consumer can't mutate the provenance.
    """

    status: str
    reason: str = ""
    scout_indices: tuple[int, ...] = ()
    scout_metadata: tuple[MappingProxyType, ...] = ()


def _resolve_incidence(meta: Any, motor: Any) -> float | None:
    """Case-insensitive lookup of the incidence motor in a frame's metadata →
    float, or None if absent/non-numeric.  Verbatim port of xdart's
    ``_resolve_incidence_from_meta`` so the headless extremes match the GUI's."""
    if not isinstance(meta, dict):
        return None
    ml = str(motor).lower()
    for key, val in meta.items():
        if str(key).lower() == ml:
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
    return None


def _scan_manifest(source: Any):
    """Probe the optional ``FrameSource.scan_manifest()`` capability (the
    Protocol is name-only ``runtime_checkable``, so a ``getattr`` probe lets a
    ``Scan`` / duck source without the method work).  Returns ``None`` on any
    failure — the caller treats that as 'unverifiable'."""
    fn = getattr(source, "scan_manifest", None)
    if not callable(fn):
        return None
    try:
        return fn()
    except Exception:
        return None


def _incidence_extremes(manifest: Any, motor: Any):
    """The image-free half of xdart's ``_gi_whole_scan_scout_entries``: from a
    ``[(frame_index, metadata), ...]`` manifest, decide the global GI scout.

    Returns ``(status, extremes)``:
      - ``("skip", [])`` — fixed/manual angle, ``<2`` frames, or one distinct
        incidence (the chunk/session freeze was never clipped);
      - ``("unverifiable", [])`` — no manifest, or ``<2`` readable incidences
        (cannot establish a global range → warn-and-proceed);
      - ``("found", [(lo_idx, lo_meta), (hi_idx, hi_meta)])`` — a real sweep; the
        extremes are chosen BY RESOLVED INCIDENCE VALUE, never positionally.
    """
    try:
        float(motor)            # fixed/manual: one angle for the whole scan
        return "skip", []
    except (TypeError, ValueError):
        pass
    if manifest is None:
        return "unverifiable", []
    if len(manifest) < 2:
        return "skip", []       # single-frame scan: no incidence range
    resolved = []
    for idx, meta in manifest:
        ang = _resolve_incidence(meta, motor)
        if ang is not None:
            resolved.append((ang, int(idx), meta))
    if len(resolved) < 2:
        # ≥2 frames but we can't read incidence for two of them: cannot
        # establish the global range → fail to 'unverifiable' (warn-and-proceed).
        return "unverifiable", []
    lo = min(resolved, key=lambda r: r[0])
    hi = max(resolved, key=lambda r: r[0])
    if lo[0] == hi[0]:          # no incidence sweep → chunk grid is fine
        return "skip", []
    return "found", [(lo[1], lo[2]), (hi[1], hi[2])]


def prepare_gi_freeze(
    source: Any,
    plan: ReductionPlan,
    *,
    incidence_motor: Any = None,
) -> tuple[ReductionPlan, PrepareDiagnostics]:
    """Whole-scan GI prepass (ADR-0006): scout *source*'s full metadata extent
    and return a COPY of *plan* with ``extra["gi_freeze_scout_indices"]`` pinned
    to the GLOBAL incidence extremes, plus a :class:`PrepareDiagnostics`.

    Computes WHICH FRAMES only — it never loads detector images and never
    integrates.  Hand the returned plan to the freeze step
    (``ReductionSession(plan2, source, gi_freeze_mode="scout_union")``, or xdart's
    ``freeze_live_scan_gi_ranges``) and the existing ``_apply_gi_freeze_policy``
    unions those pinned frames instead of chunk-1's first/last — the fix for the
    codex-P1 chunk-clip.  GI-only; non-GI plans pass through with ``"skip"``.
    Never raises for an unenumerable source.

    ``incidence_motor`` defaults to ``plan.gi.incidence_motor``.
    """
    if plan.gi is None:
        return plan, PrepareDiagnostics("skip", reason="non-GI plan")
    # Already-pinned ranges: the freeze is a no-op, so don't even enumerate
    # (preserves the T0-3 silent skip).
    if _gi_1d_freeze_key(plan) is None and not _gi_2d_freeze_keys(plan):
        return plan, PrepareDiagnostics(
            "skip", reason="GI output ranges already pinned")
    motor = incidence_motor
    if motor is None:
        motor = getattr(plan.gi, "incidence_motor", None)
    manifest = _scan_manifest(source)
    status, extremes = _incidence_extremes(manifest, motor)
    if status != "found":
        reason = {
            "skip": "fixed/single incidence or <2 frames",
            "unverifiable": "whole-scan incidence extent could not be "
                            "established — warn and proceed on the chunk freeze",
        }.get(status, "")
        return plan, PrepareDiagnostics(status, reason=reason)
    indices = tuple(int(idx) for idx, _meta in extremes)
    meta = tuple(MappingProxyType(dict(m)) for _idx, m in extremes)
    new_extra = {**plan.extra, "gi_freeze_scout_indices": list(indices)}
    return replace(plan, extra=new_extra), PrepareDiagnostics(
        "frozen",
        reason=f"scout extremes pinned to frames {indices}",
        scout_indices=indices,
        scout_metadata=meta,
    )


def _apply_gi_freeze_policy(
    plan: ReductionPlan,
    scan: Scan,
    *,
    freeze_policy: str | None,
    fi: Any,
    initial_incident_angle: float | None,
    warned_monitor_keys: set[str] | None = None,
    run_saturation_mask: _RunSaturationMask | None = None,
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
        reduction = _reduce_frame(frame, None, plan, scout_integrators, masks,
                                  warned_monitor_keys=warned_monitor_keys,
                                  run_saturation_mask=run_saturation_mask)
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
        from xrd_tools.integrate.gid import freeze_common_axis

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
        from xrd_tools.integrate.gid import freeze_common_axes_2d

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
    from xrd_tools.integrate.gid import gi_1d_output_axis_key

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
            # Per-worker AI (pyFAI AIs aren't safe to share across threads).
            # Deep-copy the base integrator instead of rebuilding from
            # scan.poni: poni_to_integrator() cannot recover a GENERIC/unnamed
            # detector's pixel size (PONI carries only a detector *name*), so a
            # worker rebuild drops _pixel1/_pixel2 -> None and integrate1d
            # crashes in calc_cartesian_positions.  A deepcopy keeps the
            # detector (pixel sizes intact), stays thread-isolated, and is
            # geometrically identical to the owner thread's AI (strengthening
            # live==batch==reload equivalence).  Fall back to a poni rebuild
            # only when there is no base AI (pure-PONI, named-detector path).
            ai = copy.deepcopy(self.ai) if self.ai is not None \
                else poni_to_integrator(self.scan.poni)
            self._local.ai = ai
        return ai

    def fiber(self) -> Any:
        if self.scan.poni is None:
            return self.fi
        if threading.get_ident() == self._owner_thread and self.fi is not None:
            return self.fi
        fi = getattr(self._local, "fi", None)
        if fi is None:
            if self.fi is not None:
                # Same reasoning as standard(): deepcopy the base
                # FiberIntegrator to keep a generic detector's pixel size
                # (a poni rebuild would drop it) and stay thread-isolated.
                fi = copy.deepcopy(self.fi)
            else:
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
        # MEM-3: cap the "just give me a pool" default at the RAM-aware knee.
        from xrd_tools.core import reduction_worker_cap
        return ThreadPoolExecutor(max_workers=reduction_worker_cap()), True
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


# How often a parked worker or writer wakes to re-check its item's decision.
# The orchestrating caller publishes microseconds later on EVERY path, failure
# transactions included, and an expired wake is never an outcome (§19.4).
_TICKET_DECISION_TIMEOUT = 60.0
_TICKET_ACCEPTED, _TICKET_REJECTED = "ACCEPTED", "REJECTED"
_MISSING = object()


def _require_positive_exact_int(value: Any, name: str) -> None:
    if type(value) is not int or value < 1:
        raise TypeError(f"{name} must be a positive exact int; got {value!r}")


class _InFlightWindow:
    __slots__ = ("limit", "_members", "_changed")

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._members: dict[_StreamPublication, None] = {}
        self._changed = threading.Event()

    def try_acquire(self, ticket: _StreamPublication, timeout: float) -> bool:
        if ticket in self._members:
            return True
        if len(self._members) < self.limit:
            self._members[ticket] = None
            return True
        self._changed.wait(timeout)
        self._changed.clear()
        return False

    def release(self, ticket: _StreamPublication) -> bool:
        first: BaseException | None = None
        try:
            removed = self._members.pop(ticket, _MISSING) is not _MISSING
        except BaseException as exc:
            first = exc
            try:
                removed = self._members.pop(ticket, _MISSING) is not _MISSING
            except BaseException:
                raise first
        if removed or first is not None:
            try:
                self._changed.set()
            except BaseException as exc:
                first = first or exc
                try:
                    self._changed.set()
                except BaseException:
                    raise first
        if first is not None:
            raise first
        return removed


class _StreamPublication:
    __slots__ = ("frame", "future", "unstage", "_decision", "_decided")

    def __init__(self, frame: Frame) -> None:
        self.frame = frame
        self.future: Any = None
        self.unstage: Callable[[], None] | None = None
        self._decision: tuple[str, int | None] | None = None
        self._decided = threading.Event()

    @property
    def decision(self) -> tuple[str, int | None] | None:
        return self._decision

    def store_accepted(self, attempt: int | None) -> None:
        if attempt is not None:
            _require_positive_exact_int(attempt, "published attempt")
        receipt = (_TICKET_ACCEPTED, attempt)
        current = self._decision
        if current is None:
            self._decision = receipt
        elif current != receipt:
            raise RuntimeError("contradictory publication decision")

    def reject_and_wake(self) -> None:
        current = self._decision
        if current is None:
            self._decision = (_TICKET_REJECTED, None)
        elif current[0] != _TICKET_REJECTED:
            raise RuntimeError("accepted publication cannot be rejected")
        self._decided.set()

    def complete_wake(self) -> None:
        if self._decision is None:
            raise RuntimeError("cannot wake an undecided publication")
        self._decided.set()

    def take_rejection_undo(self) -> Callable[[], None] | None:
        receipt = self._decision
        if receipt is not None and receipt[0] == _TICKET_ACCEPTED:
            return None
        undo = self.unstage
        self.unstage = None
        return undo

    def await_decision(self, timeout: float = _TICKET_DECISION_TIMEOUT
                       ) -> tuple[str, int | None]:
        while True:
            self._decided.wait(timeout)
            receipt = self._decision
            if receipt is not None:
                return receipt


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


def _wait_pending_futures(pending: list[tuple[Frame, Any]], *, worker: Any | None) -> None:
    """Block until every pending future has resolved (done or cancelled).

    Error-path companion to :func:`_cancel_pending_futures` (D6): callers
    that are about to release the pending frames' image refs must first wait
    out the already-running tail, or a still-running ``_reduce_frame``
    re-pins ``frame.image`` after the clear.  Exceptions/cancellations are
    swallowed here -- the caller is re-raising the original error.
    """
    if worker is None:
        return
    for _frame, candidate in pending:
        result = getattr(candidate, "result", None)
        if not callable(result):
            continue
        try:
            result()
        except BaseException:
            pass


def _reduce_frame(
    frame: Frame,
    raw_image: np.ndarray | None,
    plan: ReductionPlan,
    integrators: _ReductionIntegratorProvider,
    plan_masks: dict[tuple[int, int], np.ndarray | None],
    frame_masks: dict[tuple[int, tuple[int, int]], tuple[Any, np.ndarray | None]] | None = None,
    cancel_token: CancelToken | None = None,
    warned_monitor_keys: set[str] | None = None,
    *,
    include_corrected_image: bool = False,
    run_saturation_mask: _RunSaturationMask | None = None,
    strict: StrictPolicy | None = None,
) -> FrameReduction:
    if cancel_token is not None and cancel_token.cancelled:
        raise _ReductionCancelled
    if raw_image is not None:
        frame.image = np.asarray(raw_image)
    raw_image_arr = np.asarray(frame.load_image())  # pre-float: integer dtype for the saturation ceiling
    image = raw_image_arr.astype(float)
    if cancel_token is not None and cancel_token.cancelled:
        raise _ReductionCancelled
    if image.ndim != 2:
        raise ValueError(f"Frame {frame.index} image must be 2D; got shape {image.shape}")
    _validate_frame_inputs(frame, image.shape, frame_masks)
    corrected_image = (
        _thumbnail_corrected_image(raw_image_arr, frame.background)
        if include_corrected_image
        else None
    )
    image = _apply_thresholds(image, plan)
    image = _subtract_background(image, frame.background)
    plan_mask = _cached_mask_for_shape(
        plan.mask,
        image.shape,
        "ReductionPlan.mask",
        plan_masks,
    )
    mask = _combined_mask(plan_mask, frame.mask, image.shape, frame_masks)
    mask = _apply_saturation_mask(
        mask,
        raw_image_arr,
        plan,
        run_saturation_mask=run_saturation_mask,
    )

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
                normalization_factor=_normalization_for(
                    frame, plan.integration_1d, warned_monitor_keys, strict=strict),
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
                normalization_factor=_normalization_for(
                    frame, plan.integration_2d, warned_monitor_keys, strict=strict),
            )
            if plan.integration_2d is not None else None
        )
    else:
        ai = integrators.standard()
        p1 = plan.integration_1d
        if p1 is not None and str(p1.unit or "").lower() == "chi_deg":
            # Non-GI azimuthal profile (Mode A): the output axis is chi, while
            # radial_range is the q band to integrate over.  Mirror
            # xdart.LiveFrame.integrate_1d's legacy dispatch to
            # integrate_radial; pyFAI's normal integrate1d would interpret the
            # q band as a chi output range and persist a garbage sliver.
            chi_extra = dict(p1.extra)
            chi_extra.pop("error_model", None)
            chi_extra.pop("variance", None)
            # S-4 (input half): shift an explicit chi range by -azimuth_offset into
            # the raw pyFAI frame BEFORE integrating -- the exact mirror of the 2D
            # (_integration_azimuth_range) -- so an explicit panel-frame range read
            # off the offset-labeled axes integrates the SAME bins as the 2D.  The
            # output axis is relabeled +offset below.  (Raw range passed straight
            # through would integrate 90deg off the 2D for the default offset.)
            _chi_azimuth_range = _integration_azimuth_range(p1)
            if _chi_azimuth_range is not None:
                chi_extra.setdefault("azimuth_range", _chi_azimuth_range)
            r1d = integrate_radial(
                image,
                ai,
                npt=p1.npt,
                npt_rad=p1.npt_rad,
                radial_unit="q_A^-1",
                method=p1.method,
                mask=mask,
                radial_range=p1.radial_range,
                polarization_factor=p1.polarization_factor,
                normalization_factor=_normalization_for(
                    frame, p1, warned_monitor_keys, strict=strict),
                **chi_extra,
            )
        else:
            r1d = (
                integrate_1d(
                    image,
                    ai,
                    npt=p1.npt,
                    unit=p1.unit,
                    method=p1.method,
                    mask=mask,
                    radial_range=p1.radial_range,
                    azimuth_range=_integration_azimuth_range(p1),
                    error_model=p1.error_model,
                    polarization_factor=p1.polarization_factor,
                    normalization_factor=_normalization_for(
                        frame, p1, warned_monitor_keys, strict=strict),
                    **p1.extra,
                )
                if p1 is not None else None
            )
        # S-4: re-add chi_offset to the 1D chi OUTPUT axis (mirror the 2D
        # r2d.azimuthal += azimuth_offset below), so the written Mode-A (chi_deg)
        # chi axis matches the 2D cake chi instead of staying in the raw pyFAI
        # frame 90deg out.  Standard mode only (GI keeps azimuth_offset 0).
        if (r1d is not None and p1 is not None
                and str(getattr(p1, "unit", "") or "").lower() == "chi_deg"
                and getattr(p1, "azimuth_offset", 0.0)):
            r1d.radial = (np.asarray(r1d.radial, dtype=float)
                          + float(p1.azimuth_offset))
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
                    frame, plan.integration_2d, warned_monitor_keys, strict=strict),
                **plan.integration_2d.extra,
            )
            if plan.integration_2d is not None else None
        )
        if r2d is not None and plan.integration_2d.azimuth_offset:
            r2d.azimuthal = r2d.azimuthal + float(plan.integration_2d.azimuth_offset)

    # D7 loud: a 2D integration with no usable data (all-dummy) would persist
    # nothing meaningful.  Under a loud policy RAISE (the streaming writer
    # records the failure + skips the frame, then re-raises at finish() — never
    # writing bad data); under graceful the all-dummy result is returned and
    # dropped per-frame downstream (the publication gate / writer), never
    # aborting a whole-scan save.
    if (strict is not None and strict.gi_all_dummy
            and r2d is not None and _is_all_dummy_2d(r2d)):
        raise GIAllDummyError(
            f"frame {frame.index}: the 2D integration is entirely dummy "
            "(no usable data); pass StrictPolicy.graceful() to drop it "
            "per-frame instead of raising."
        )
    mode_1d, mode_2d = _plan_mode_keys(plan)
    return FrameReduction(
        frame_index=frame.index,
        result_1d=r1d,
        result_2d=r2d,
        mode_1d=mode_1d,
        mode_2d=mode_2d,
        metadata=dict(frame.metadata),
        corrected_image=corrected_image,
    )


def _thumbnail_corrected_image(
    raw_image: np.ndarray,
    background: np.ndarray | float | None,
) -> np.ndarray:
    """Return a float32 raw-minus-background image for transient thumbnails."""
    raw = np.asarray(raw_image)
    if background is None:
        return np.array(raw, dtype=np.float32, copy=True)
    bg = np.asarray(background, dtype=np.float32)
    if bg.ndim == 0 and float(bg) == 0.0:
        return np.array(raw, dtype=np.float32, copy=True)
    return np.asarray(raw, dtype=np.float32) - bg


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
    if bg.shape == () and float(bg) == 0.0:
        return image
    if bg.ndim > 0 and bg.shape != image.shape:
        raise ValueError(
            f"background shape {bg.shape} does not match image shape {image.shape}"
        )
    return image - bg


def _integration_azimuth_range(
    plan: "Integration1DPlan | Integration2DPlan",
) -> tuple[float, float] | None:
    # S-4: shared by the 2D cake and the 1D Mode-A chi branch -- both carry
    # azimuth_range + azimuth_offset, and both shift the input by -offset (raw
    # frame) and re-add +offset at output.
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
        "chigi": GI1DMode.CHI_GI,
        "chi_gi": GI1DMode.CHI_GI,
        "chi": GI1DMode.CHI_GI,
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
    if gi.mode_1d is GI1DMode.CHI_GI:
        # Azimuthal profile: I vs χ_GI over a q_total band.  ``common`` passes
        # npt=plan.npt as the χ_GI output-bin count; the second pts box (npt_oop)
        # is the q_total sampling across the integrated band.
        return integrate_gi_azimuthal_1d(
            image,
            fi,
            npt_q=npt_oop,
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
    frame_mask_cache: dict[tuple[int, tuple[int, int]], tuple[Any, np.ndarray | None]] | None = None,
) -> np.ndarray | None:
    frame_mask = _cached_frame_mask_for_shape(
        frame_mask,
        image_shape,
        frame_mask_cache,
    )
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


def _cached_frame_mask_for_shape(
    mask: np.ndarray | MaskSpec | None,
    image_shape: tuple[int, int],
    cache: dict[tuple[int, tuple[int, int]], tuple[Any, np.ndarray | None]] | None,
) -> np.ndarray | None:
    if mask is None:
        return None
    if not isinstance(mask, MaskSpec) or cache is None:
        return _as_bool_mask(mask, "Frame.mask", image_shape=image_shape)
    owner = mask.values
    key = (id(owner), image_shape)
    cached = cache.get(key)
    if cached is not None and cached[0] is owner:
        return cached[1]
    resolved = _as_bool_mask(mask, "Frame.mask", image_shape=image_shape)
    cache[key] = (owner, resolved)
    return resolved


def _apply_saturation_mask(
    mask,
    raw_image,
    plan,
    *,
    run_saturation_mask: _RunSaturationMask | None = None,
):
    """Union the toggle-qualified detector value mask into ``mask``.

    A session supplies ``run_saturation_mask`` so the first native frame owns
    one immutable mask for the scan.  The no-state path remains dynamic for
    direct/private callers.  Disabled behavior is an exact no-op.
    """
    if run_saturation_mask is not None:
        return run_saturation_mask.apply(mask, raw_image)
    return detector_value_mask(
        mask,
        raw_image,
        enabled=bool(plan.mask_saturation),
    )


# S8: fallback warn-state for direct (sessionless) calls.  Sessions own
# their per-scan set — see ReductionSession._warned_monitor_keys — so a dead
# monitor warns once per SCAN, not once per process.  A bad monitor means
# frames are written UN-normalized, which must not be silent.
_warned_monitor_keys: set[str] = set()


def _normalization_for(
    frame: Frame,
    plan: Integration1DPlan | Integration2DPlan,
    warned_keys: set[str] | None = None,
    *,
    strict: StrictPolicy | None = None,
) -> float | None:
    if frame.normalization_factor is not None:
        return float(frame.normalization_factor)
    if plan.monitor_key is not None:
        key = plan.monitor_key
        # S-6: delegate lookup + guard to the ONE canonical resolver
        # (case-insensitive; rejects missing/zero/negative/non-finite) so the
        # reduction spine and the GUI mirror never disagree about whether a frame
        # is normalized.  The exact/upper/lower lookup here missed MIXED-case
        # monitor keys -> the spine wrote UN-normalized data while ``map_norm``
        # claimed normalization (and it accepted negatives the resolver rejects).
        norm = resolve_monitor_norm(frame.metadata, key)
        if norm is not None:
            return norm
        value = _metadata_get_case_insensitive(frame.metadata, key)
        # S8: the monitor was configured but unusable — the frame is about to
        # be written UN-normalized.  D7 loud: RAISE so a scripted/batch run
        # fails instead of silently persisting un-normalized data; graceful
        # keeps the warn-once below.
        if strict is not None and strict.missing_normalization:
            raise MissingNormalizationError(
                f"monitor {key!r} is missing/zero/non-finite on frame "
                f"{frame.index} (value={value!r}); the frame would be written "
                f"UN-normalized.  Pass StrictPolicy.graceful() to allow it."
            )
        # Warn once per monitor key per scan (not per frame: a dead monitor on
        # a 10k-frame scan must not emit 10k warnings; per scan, not per
        # process: the next scan's dead monitor must not be silenced by this
        # one's).  set.add is GIL-atomic; a racing double-warn is harmless.
        warned = _warned_monitor_keys if warned_keys is None else warned_keys
        if key not in warned:
            warned.add(key)
            warnings.warn(
                f"monitor {key!r} is missing/zero/non-finite on frame "
                f"{frame.index} (value={value!r}); affected frames are "
                f"written UN-normalized.  (Warned once per monitor key "
                f"per scan.)",
                RuntimeWarning, stacklevel=2,
            )
    return None


def _validate_frame_inputs(
    frame: Frame,
    image_shape: tuple[int, int],
    frame_mask_cache: dict[tuple[int, tuple[int, int]], tuple[Any, np.ndarray | None]] | None = None,
) -> None:
    if frame.background is not None:
        bg = np.asarray(frame.background)
        if bg.ndim > 0 and bg.shape != image_shape:
            raise ValueError(
                f"Frame {frame.index} background shape {bg.shape} does not "
                f"match image shape {image_shape}"
            )
    if frame.mask is not None:
        mask = _cached_frame_mask_for_shape(
            frame.mask,
            image_shape,
            frame_mask_cache,
        )
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


def _sink_is_memory_only(sink: ReductionSink) -> bool:
    """Whether a sink leaves ``ReductionResult.frames`` as the only product."""
    if isinstance(sink, MemorySink):
        return True
    if isinstance(sink, CompositeSink):
        return all(_sink_is_memory_only(child) for child in sink.sinks)
    return False


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
