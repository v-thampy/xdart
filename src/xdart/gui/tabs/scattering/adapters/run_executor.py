from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import logging
import os
from pathlib import Path
from queue import Empty, Full, Queue, SimpleQueue
from threading import Event, Lock, Thread, current_thread, get_ident
from time import monotonic
from typing import Any, Callable, Mapping
import numpy as np
from xdart.modules.frame_publication import (
    FramePublication,
    PublicationStore,
    canonical_frame_source_identity,
)
from xrd_tools.core.scan import SourceKind
from xrd_tools.reduction import FrameBackgroundPlan, resolve_frame_background
from xrd_tools.integrate.calibration import (
    detector_calibration_to_integrator,
    load_detector_calibration,
)
from xrd_tools.session.frame_record_store import FrameRecordStore
from xrd_tools.io.output_transaction import (
    StreamTerminal,
    stream_terminal_object_revision,
)
from xrd_tools.session.run_configuration import FrozenRunConfiguration
from xrd_tools.sources import open_source
from xrd_tools.sources.cursor import open_container_cursor
from xrd_tools.sources.nexus import NexusStackSource
from xrd_tools.sources.probe import ProbeState
from ..contracts import (
    AdmittedOutput, AdmissionFailure, AdmissionReceipt, AdmissionReleased,
    AdmissionToken,
    SourceCapture, SourceExecutionIdentityV1, StartCapture,
    executor_start_inputs_are_valid,
)
from ..acquisition_runtime import AcquisitionRuntime, TerminalPauseFailure
from ..display_values import (
    DisplayFrameCatalog,
    DisplayFrameKey,
    StandardDisplayPayload,
    StandardEventKind,
    StandardRunEvent,
    StandardQuartileTiming,
    StandardTerminalTiming,
)
from ..display_runtime import (
    DisplayArtifact,
    RunDisplayState,
)
from ..display_retirement import (
    DisplayRetirementOwner,
    NO_DISPLAY_RETIREMENT,
)
from ..events import (
    CleanupStatus, DetachedDiagnostic, DurablePaused, ExecutorAccepted, ExecutorClosed,
    ExecutorStartFailed, RunIdentity, detach_exception,
)
from ..output_preflight import (
    _background_frame_fact, _merge_background_binding,
    DeferredDirectoryPlan, LiveDirectoryAttempt, LiveDirectoryGroup,
    OutputDisposition, PlannedOutput, SourceRevisionChanged,
    native_int_reduction_plan, materialize_deferred_output,
    live_directory_groups, materialize_live_directory_group,
    prepare_output as build_admission_receipt, source_snapshots,
    target_state_matches, validate_admitted_receipt, validate_planned_source,
)
from .target_reservation import (
    AdmissionOperation as _AdmissionOperation,
    RunResources,
)
from .dynamic_output import DynamicOutputAdapter, HeavyResidencyFact

logger = logging.getLogger(__name__)

# Preserve the established test-injection seam while production remains strict.
load_poni = load_detector_calibration
poni_to_integrator = detector_calibration_to_integrator

_CONTAINER_READ_CHUNK_FRAMES = 8
_SOURCE_PREFETCH_FRAMES = 4
_DISPLAY_PROJECTION_FRAMES = 8
_LIVE_DIRECTORY_POLL_S = 0.1

_SOURCE_SUBMISSION_END = object()


def _resolve_background_before_submit(frame, plan: FrameBackgroundPlan, binding,
                                      *, cancelled, retryable=False, frame_fact=None) -> bool:
    """The sole immediate-pre-submit Background array insertion seam."""
    if type(plan) is not FrameBackgroundPlan: raise TypeError("Background plan must be exact")
    frame.background = None; frame.background_dependency_bytes = None
    frame.background_dependency_fingerprint = None
    if plan.mode == "None": return True
    fact = binding[1] if binding is not None else frame_fact
    if fact is None: raise RuntimeError("Background resolution lost its frame fact")
    result = resolve_frame_background(plan, fact, cancelled=cancelled)
    if result.disposition == "RETRYABLE" and retryable: return False
    if result.disposition == "CANCELLED": raise RuntimeError("admission cancelled")
    if result.disposition != "RESOLVED" or result.background is None:
        raise SourceRevisionChanged("Background dependency is not presently resolvable")
    pair = result.descriptor_bytes, result.fingerprint
    if binding is not None:
        if pair != binding[2:]: raise SourceRevisionChanged("Background dependency changed after qualification")
        pair = binding[2], binding[3]
    frame.background = result.background
    frame.background_dependency_bytes, frame.background_dependency_fingerprint = pair
    return True


def _qualify_background_bindings(configuration, scan, item, decision, policy,
                                  prior, stop_signal):
    plan = configuration.background
    if plan.mode == "None": return prior
    requirements = policy.allocation.requirements
    shape = requirements.height, requirements.width
    selector = None if item.descriptor is None else item.descriptor.dataset_path
    if not prior and configuration.output_mode == "Append":
        labels = tuple(range(item.source_stamp.first_label,
            item.source_stamp.first_label + item.source_stamp.frame_count))
        skipped = tuple(label for label in labels if label not in decision.labels)
        if skipped:
            from xrd_tools.io.record_writer import _read_persisted_background_bindings
            prior = _read_persisted_background_bindings(item.target, skipped, requirements.background_binding_bytes)
    bindings = ()
    for existing in prior:
        bindings = _merge_background_binding(bindings, existing,
            limit=requirements.background_binding_bytes)
    frames = {int(frame.index): frame for frame in scan.frames}
    for label in (() if configuration.live_mode else decision.labels):
        frame = frames.get(label)
        if frame is None: raise ValueError("Background qualification lost an admitted frame")
        fact = _background_frame_fact(frame, plan, shape, selector)
        expected = next((value for value in bindings if value[0] == label), None)
        try:
            if not _resolve_background_before_submit(frame, plan, expected,
                    cancelled=stop_signal, retryable=configuration.live_mode, frame_fact=fact):
                raise SourceRevisionChanged("Live Background dependency is retryable")
            binding = (label, fact, frame.background_dependency_bytes,
                       frame.background_dependency_fingerprint)
            bindings = _merge_background_binding(bindings, binding,
                limit=requirements.background_binding_bytes)
        finally:
            frame.background = None; frame.background_dependency_bytes = None
            frame.background_dependency_fingerprint = None
    return bindings


def _background_ready(run, output, frame) -> bool:
    plan = run.configuration.background
    if plan.mode == "None": return _resolve_background_before_submit(frame, plan, None, cancelled=run.stop_signal)
    while True:
        binding = output.background_binding(int(frame.index))
        if binding is None:
            if not run.configuration.live_mode:
                raise RuntimeError("admitted Background binding is absent")
            item, requirements = output.background_admission_context()
            selector = None if item.descriptor is None else item.descriptor.dataset_path
            fact = _background_frame_fact(
                frame, plan, (requirements.height, requirements.width), selector)
            try:
                if not _resolve_background_before_submit(
                        frame, plan, None, cancelled=run.stop_signal,
                        retryable=True, frame_fact=fact):
                    run.stop_signal.wait(_LIVE_DIRECTORY_POLL_S); continue
                binding = output.admit_background_binding((int(frame.index), fact,
                    frame.background_dependency_bytes, frame.background_dependency_fingerprint)); frame.background_dependency_bytes, frame.background_dependency_fingerprint = binding[2:]
            except BaseException:
                frame.background = None; frame.background_dependency_bytes = None
                frame.background_dependency_fingerprint = None
                raise
            return True
        if _resolve_background_before_submit(frame, plan, binding,
                cancelled=run.stop_signal, retryable=run.configuration.live_mode): return True
        run.stop_signal.wait(_LIVE_DIRECTORY_POLL_S)
_QUARTILE_STAGE_NAMES = (
    "reducer_compute",
    "source_read",
    "submit_wait",
    "writer_batch",
    "writer_flush",
    "xye",
    "completion_display",
    "finish_wait",
)


def _quartile_stage_totals(
    cumulative: Mapping[str, float],
) -> dict[str, float]:
    value = lambda key: max(0.0, float(cumulative.get(key, 0.0)))
    return {
        "reducer_compute": value("reducer_compute"),
        "source_read": value("source_read"),
        "submit_wait": value("submit_wait"),
        "writer_batch": value("sink_nexus_write"),
        "writer_flush": value("sink_nexus_flush"),
        "xye": (
            value("sink_xye_write") + value("sink_xye_promotion")
        ),
        "completion_display": sum(value(key) for key in (
            "session_record_upsert",
            "session_frame_listeners",
            "session_progress_listeners",
            "display_projection",
        )),
        "finish_wait": value("finish_wait"),
    }


def _quartile_compute_count(cumulative: Mapping[str, float]) -> int:
    raw = cumulative.get("reducer_compute_count", 0)
    value = float(raw)
    count = int(value)
    if not np.isfinite(value) or value < 0.0 or value != count:
        raise ValueError("reducer compute count is invalid")
    return count


@dataclass(frozen=True, slots=True)
class _RunQuartileBoundary:
    quartile: int
    completed: int
    total: int
    frame_count: int
    wall_seconds: float


@dataclass(frozen=True, slots=True)
class _FrameProjectionItem:
    frame_index: int; record: Any; frame_mask_qualified: bool


@dataclass(frozen=True, slots=True)
class _CheckpointProjectionItem:
    artifact: str; labels: tuple[int, ...]


@dataclass(slots=True)
class _RunQuartileCapture:
    """Fixed-size temporal snapshots; no per-frame timing history."""

    total: int
    started_at: float
    thresholds: tuple[int, int, int] = field(init=False)
    frame_counts: tuple[int, int, int, int] = field(init=False)
    completed: int = field(default=0, init=False)
    _next_quartile: int = field(default=0, init=False, repr=False)
    _previous_wall: float = field(default=0.0, init=False, repr=False)
    _previous_stages: dict[str, float] = field(
        default_factory=lambda: dict.fromkeys(_QUARTILE_STAGE_NAMES, 0.0),
        init=False,
        repr=False,
    )
    _wall_rows: list[float] = field(default_factory=list, init=False, repr=False)
    _previous_compute_count: int = field(default=0, init=False, repr=False)
    _compute_counts: list[int] = field(default_factory=list, init=False, repr=False)
    _stage_rows: dict[str, list[float]] = field(
        default_factory=lambda: {
            name: [] for name in _QUARTILE_STAGE_NAMES
        },
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if (
            type(self.total) is not int
            or self.total < 4
            or type(self.started_at) is not float
            or not np.isfinite(self.started_at)
        ):
            raise TypeError("quartile capture boundary is invalid")
        first = (self.total + 3) // 4
        second = (self.total + 1) // 2
        third = (3 * self.total + 3) // 4
        self.thresholds = (first, second, third)
        self.frame_counts = (
            first,
            second - first,
            third - second,
            self.total - third,
        )

    def observe(
        self,
        completed: int,
        *,
        now: float,
        cumulative: Mapping[str, float],
    ) -> _RunQuartileBoundary | None:
        if type(completed) is not int or completed != self.completed + 1:
            raise ValueError("quartile completion sequence is not contiguous")
        self.completed = completed
        if (
            self._next_quartile >= 3
            or completed < self.thresholds[self._next_quartile]
        ):
            return None
        wall = max(0.0, float(now) - self.started_at)
        stages = _quartile_stage_totals(cumulative)
        compute_count = _quartile_compute_count(cumulative)
        wall_delta = max(0.0, wall - self._previous_wall)
        self._wall_rows.append(wall_delta)
        for name in _QUARTILE_STAGE_NAMES:
            total = stages[name]
            self._stage_rows[name].append(max(
                0.0, total - self._previous_stages[name],
            ))
        self._previous_wall = wall
        self._previous_stages = stages
        self._compute_counts.append(
            max(0, compute_count - self._previous_compute_count)
        )
        self._previous_compute_count = compute_count
        quartile = self._next_quartile + 1
        frame_count = self.frame_counts[self._next_quartile]
        self._next_quartile += 1
        return _RunQuartileBoundary(
            quartile,
            completed,
            self.total,
            frame_count,
            wall_delta,
        )

    def finish(
        self,
        *,
        completed: int,
        now: float,
        cumulative: Mapping[str, float],
    ) -> StandardQuartileTiming | None:
        if (
            completed != self.total
            or self.completed != self.total
            or self._next_quartile != 3
        ):
            return None
        wall = max(0.0, float(now) - self.started_at)
        stages = _quartile_stage_totals(cumulative)
        compute_count = _quartile_compute_count(cumulative)
        wall_rows = tuple(
            self._wall_rows
            + [max(0.0, wall - self._previous_wall)]
        )
        details = [("wall", wall_rows)]
        details.extend(
            (
                name,
                tuple(self._stage_rows[name] + [max(
                    0.0, stages[name] - self._previous_stages[name],
                )]),
            )
            for name in _QUARTILE_STAGE_NAMES
        )
        return StandardQuartileTiming(
            self.frame_counts,
            tuple(details),
            tuple(self._compute_counts + [max(
                0, compute_count - self._previous_compute_count,
            )]),
        )
_DISPLAY_PROJECTION_END = object()


def _terminal_durable_progress(run) -> tuple[int, int]:
    completed = max(0, int(run.completed))
    return completed, max(int(run.total), completed)


def _session_terminal_commit_identity(
    session: object,
    artifact: Path | None = None,
) -> StreamTerminal | None:
    terminal = getattr(session, "terminal_result", None)
    commit_identity = getattr(terminal, "commit_identity", None)
    if type(commit_identity) is not StreamTerminal:
        return None
    if stream_terminal_object_revision(commit_identity) is None:
        return None
    if (
        artifact is not None
        and os.path.normcase(os.path.abspath(str(artifact)))
        != commit_identity.target
    ):
        return None
    return commit_identity

@dataclass(slots=True)
class _StandardRun:
    configuration: FrozenRunConfiguration | None
    identity: RunIdentity
    scan: Any | None
    source: Any | None
    session: Any | None
    records: FrameRecordStore | None
    artifact: Path
    max_display_items: int = 2
    capture: SourceCapture | None = None
    sink: Any | None = None
    output: DynamicOutputAdapter | None = None
    worker: Thread | None = None
    stop_requested: bool = False
    stop_signal: Event = field(default_factory=Event)
    closed: bool = False
    cleanup_status: CleanupStatus = CleanupStatus.CLEANUP_PENDING
    primary: DetachedDiagnostic | None = None
    cleanup_failures: list[DetachedDiagnostic] = field(default_factory=list)
    cleanup_lock: Lock = field(default_factory=Lock)
    terminal_emitted: bool = False
    artifacts: list[Path] = field(default_factory=list)
    completed: int = 0
    total: int = 0
    current_total: int = 0
    current_completed: int = 0
    current_published: int = 0
    current_epoch_published: int = 0
    files_discovered: int = 0
    files_processed: int = 0
    files_skipped: int = 0
    processed_live_revisions: dict[str, LiveDirectoryAttempt] = field(
        default_factory=dict
    )
    live_revision_lock: Lock = field(default_factory=Lock)
    current_file_total: int = 0
    current_files_incremental: bool = False
    resources: RunResources | None = None
    display: RunDisplayState = field(init=False)
    context_runtime: AcquisitionRuntime | None = None
    unpublished_display_retired: bool = False
    frames_by_label: dict[int, Any] = field(default_factory=dict)
    perf_enabled: bool = field(init=False)
    perf_quartiles_enabled: bool = field(init=False)
    perf_started_at: float | None = None
    perf_quartiles: _RunQuartileCapture | None = None
    perf_values: dict[str, float] = field(default_factory=dict)
    perf_lock: Lock = field(default_factory=Lock)
    display_projection_queue: Queue[object] | None = None
    display_projection_worker: Thread | None = None
    display_projection_errors: list[BaseException] = field(default_factory=list)
    light_projection_error: BaseException | None = None
    light_projection_error_lock: Lock = field(default_factory=Lock)
    gui_thread_id: int = field(default_factory=get_ident)
    command_failure: DetachedDiagnostic | None = None
    resource_facts: list[Any] = field(default_factory=list)
    pending_partition_count: int = 1
    terminal_commit_identity: StreamTerminal | None = None

    def __post_init__(self) -> None:
        self.perf_quartiles_enabled = (
            os.environ.get("XDART_PERF_QUARTILES", "").strip() == "1"
        )
        self.perf_enabled = (
            bool(os.environ.get("XDART_PERF"))
            or self.perf_quartiles_enabled
        )
        self.display = RunDisplayState(
            self.identity,
            max_payload_items=self.max_display_items,
        )


def _perf_add(run: _StandardRun, key: str, elapsed: float) -> None:
    if not run.perf_enabled:
        return
    with run.perf_lock:
        run.perf_values[key] = run.perf_values.get(key, 0.0) + max(
            0.0, float(elapsed)
        )


def _combined_perf_snapshot(run: _StandardRun) -> dict[str, float]:
    with run.perf_lock:
        values = dict(run.perf_values)
    snapshot = getattr(run.session, "perf_snapshot", None)
    if callable(snapshot):
        for key, value in snapshot().items():
            values[key] = values.get(key, 0.0) + float(value)
    return values


def _observe_quartile_completion(run: _StandardRun) -> None:
    if not run.perf_quartiles_enabled:
        return
    try:
        started_at = run.perf_started_at
        if started_at is None or run.total < 4:
            return
        capture = run.perf_quartiles
        if capture is None:
            capture = _RunQuartileCapture(run.total, started_at)
            run.perf_quartiles = capture
        completed = capture.completed + 1
        boundary_due = (
            capture._next_quartile < 3
            and completed == capture.thresholds[capture._next_quartile]
        )
        boundary = capture.observe(
            completed,
            now=monotonic() if boundary_due else started_at,
            cumulative=(
                _combined_perf_snapshot(run) if boundary_due else {}
            ),
        )
        if boundary is not None:
            logger.info(
                "[PERF-QUARTILE] q=%d frames=%d/%d bucket-frames=%d "
                "wall=%.2fs mean=%.4fs/frame",
                boundary.quartile,
                boundary.completed,
                boundary.total,
                boundary.frame_count,
                boundary.wall_seconds,
                boundary.wall_seconds / boundary.frame_count,
            )
    except Exception:
        # Diagnostics must never change run correctness or terminal outcome.
        run.perf_quartiles_enabled = False
        run.perf_quartiles = None
        logger.exception("[PERF-QUARTILE] telemetry disabled after snapshot failure")


def _directory_status(
    run: _StandardRun,
    *,
    state: str = "Running",
    in_flight_processed: int = 0,
) -> str:
    processed, skipped, pending, discovered = _directory_counts(
        run,
        in_flight_processed=in_flight_processed,
    )
    status = (
        f"{state} · {processed} processed · {skipped} skipped · "
        f"{pending} pending · {discovered} discovered"
    )
    return status


def _directory_counts(
    run: _StandardRun,
    *,
    in_flight_processed: int = 0,
) -> tuple[int, int, int, int]:
    discovered = max(0, int(run.files_discovered))
    processed = max(
        0,
        min(discovered, int(run.files_processed) + in_flight_processed),
    )
    skipped = max(
        0,
        min(discovered - processed, int(run.files_skipped)),
    )
    pending = max(0, discovered - processed - skipped)
    return processed, skipped, pending, discovered


def _directory_event_fields(
    run: _StandardRun,
    *,
    in_flight_processed: int = 0,
) -> dict[str, int]:
    processed, skipped, pending, discovered = _directory_counts(
        run,
        in_flight_processed=in_flight_processed,
    )
    return {
        "files_processed": processed,
        "files_skipped": skipped,
        "files_pending": pending,
        "files_discovered": discovered,
    }


def _terminal_in_flight_files(run: _StandardRun) -> int:
    """Physical files durably completed by the current unfinished fold."""

    if run.current_file_total <= 0:
        return 0
    if run.current_files_incremental:
        return min(run.current_file_total, run.current_completed)
    return (
        run.current_file_total
        if run.current_total > 0
        and run.current_completed >= run.current_total
        else 0
    )


def _planned_physical_file_count(item: PlannedOutput) -> int:
    """Physical source files consumed by one planned output."""

    if item.source_spec.kind is SourceKind.TIFF_SERIES:
        return max(1, len(item.source_stamp.members))
    return 1 + len({
        value.file.path for value in item.source_stamp.external_members
    })


def _physical_file_key(path: Path | str) -> str:
    return os.path.normcase(os.path.realpath(path))


@contextmanager
def _run_effect(run: _StandardRun):
    runtime = run.context_runtime
    if runtime is None:
        yield
        return
    with runtime._live_effect():
        yield


def _planned_primary_paths(item: PlannedOutput) -> tuple[Path, ...]:
    """Physical candidates that own their own directory counter slots."""

    members = tuple(
        Path(value.path) for value in item.source_stamp.members
    )
    return members or (item.source_path,)


def _candidate_file_owners(
    primary_groups: tuple[tuple[Path, ...], ...],
) -> dict[str, int]:
    """Bind every discovered candidate to exactly one planned output."""

    owners: dict[str, int] = {}
    for output_index, paths in enumerate(primary_groups):
        for path in paths:
            key = _physical_file_key(path)
            previous = owners.setdefault(key, output_index)
            if previous != output_index:
                raise ValueError(
                    "physical directory candidate belongs to multiple outputs: "
                    f"{path}"
                )
    return owners


def _claim_physical_files(
    member_paths: tuple[Path, ...],
    discovered_paths: tuple[Path, ...],
    claimed: set[Path],
    *,
    candidate_owners: dict[str, int],
    output_index: int,
) -> int:
    """Claim one output's candidates and otherwise-unowned sidecars."""

    member_keys = {
        _physical_file_key(path) for path in member_paths
    }
    selected = tuple(
        path
        for path in discovered_paths
        if path not in claimed
        and (key := _physical_file_key(path)) in member_keys
        and candidate_owners.get(key, output_index) == output_index
    )
    claimed.update(selected)
    return len(selected)


def _claim_output_physical_files(
    item: PlannedOutput,
    discovered_paths: tuple[Path, ...],
    claimed: set[Path],
    *,
    candidate_owners: dict[str, int],
    output_index: int,
) -> int:
    return _claim_physical_files(
        item.group.member_paths,
        discovered_paths,
        claimed,
        candidate_owners=candidate_owners,
        output_index=output_index,
    )


def _eager_directory_file_counts(
    outputs: tuple[AdmittedOutput, ...],
    discovered_paths: tuple[Path, ...],
) -> tuple[tuple[int, ...], int]:
    """Assign each physical directory entry to at most one eager output."""

    if not discovered_paths:
        counts = tuple(
            _planned_physical_file_count(output.item) for output in outputs
        )
        return counts, 0
    candidate_owners = _candidate_file_owners(tuple(
        _planned_primary_paths(output.item) for output in outputs
    ))
    claimed: set[Path] = set()
    counts = tuple(
        _claim_output_physical_files(
            output.item,
            discovered_paths,
            claimed,
            candidate_owners=candidate_owners,
            output_index=output_index,
        )
        for output_index, output in enumerate(outputs)
    )
    return counts, len(discovered_paths) - len(claimed)


class StandardRunExecutor:

    def __init__(self, *, max_display_items: int=2, join_timeout: float=5.0) -> None:
        if type(max_display_items) is not int or max_display_items < 1:
            raise ValueError('max_display_items must be a positive integer')
        self._lock = Lock()
        self._events: SimpleQueue[StandardRunEvent] = SimpleQueue()
        self._active: _StandardRun | None = None
        self._admission: _AdmissionOperation | None = None
        self._retirement: DisplayRetirementOwner | None = None
        self._max_display_items = max_display_items
        self._join_timeout = float(join_timeout)

    def start(self, configuration, source: SourceCapture, run_identity: RunIdentity, admission: AdmissionReceipt):
        with self._lock:
            operation = self._admission
            owned_admission = operation is not None and operation.result is admission and (type(admission) is AdmissionReceipt)
            active = self._active
            valid_admission = (
                owned_admission
                and executor_start_inputs_are_valid(
                    configuration, source, run_identity
                )
                and admission.source_capture is source
                and admission.request_id is source.request_id
                and admission.candidate.matches(configuration)
                and not operation.cancelled.is_set()
                and active is None
                and admission.display_retirement
                is operation.retirement_receipt
            )
            resources = operation.transfer(admission) if valid_admission else None
            if resources is not None:
                self._admission = None
                if operation.retirement_owner is self._retirement:
                    self._retirement = None
        if not valid_admission:
            if operation is not None and owned_admission:
                operation.request_cancel()
                self._cleanup_admission(operation)
                status = operation.cleanup_receipt().cleanup_status
            else:
                status = CleanupStatus.CLEANED
            return ExecutorStartFailed(run_identity, status)
        if resources is None:
            return ExecutorStartFailed(run_identity, CleanupStatus.CLEANUP_PENDING)
        run = _StandardRun(
            configuration, run_identity, None, None, None, None,
            Path(configuration.save_path),
            max_display_items=self._max_display_items,
            capture=source,
            resources=resources,
        )
        run.display.set_factories(FrameRecordStore, PublicationStore)
        run.display.bind_transport(event_sink=self._events.put)
        if type(admission.deferred_directory) is DeferredDirectoryPlan and admission.deferred_directory.live:
            run.context_runtime = AcquisitionRuntime()
            run.context_runtime._arm_live()
        with self._lock:
            self._active = run
        worker = Thread(target=self._run, args=(run,), name='scattering-standard', daemon=True)
        run.worker = worker
        try:
            worker.start()
        except Exception as error:
            run.worker = None
            receipt = self._cleanup(run, detach_exception(error, 'thread.start'))
            if receipt.cleanup_status is CleanupStatus.CLEANED:
                with self._lock:
                    if self._active is run:
                        self._active = None
            return ExecutorStartFailed(run_identity, receipt.cleanup_status, primary=receipt.primary, cleanup_failures=receipt.cleanup_failures)
        return ExecutorAccepted(run_identity)

    def begin_admission(self, capture: StartCapture) -> AdmissionToken:
        if type(capture) is not StartCapture:
            raise TypeError('admission requires StartCapture')
        token = AdmissionToken(capture.request_id, capture.intent_snapshot.revision)
        with self._lock:
            active = self._active
            if (
                active is not None
                and active.unpublished_display_retired
            ):
                self._active = None
                active = None
            if self._admission is not None or (
                active is not None and not active.closed
            ):
                raise RuntimeError('executor already owns an operation')
            owner = self._retirement
            if active is not None and (
                owner is None
                or owner.run_identity is not active.identity
            ):
                owner = DisplayRetirementOwner(
                    active.identity,
                    lambda identity=active.identity: self._close_run(
                        identity
                    ),
                )
                self._retirement = owner
            operation = _AdmissionOperation(
                token, capture, retirement_owner=owner
            )
            if owner is not None:
                operation.retirement_receipt = owner.proof
            self._admission = operation
        worker = Thread(target=self._perform_admission, args=(operation,), name='scattering-admission', daemon=True)
        try:
            worker.start()
        except Exception:
            operation.finish_worker(None)
            operation.request_cancel()
            self._cleanup_admission(operation)
            raise
        return token

    def poll_admission(self, token: AdmissionToken) -> AdmissionReceipt | AdmissionFailure | None:
        with self._lock:
            operation = self._admission
            if operation is None or operation.token is not token:
                return None
            return operation.result

    def cancel_admission(self, token: AdmissionToken) -> AdmissionReleased:
        with self._lock:
            operation = self._admission
            if operation is None or operation.token is not token:
                return AdmissionReleased(token, CleanupStatus.CLEANED)
        operation.request_cancel()
        operation.retain_retirement_release()
        self._cleanup_admission(operation)
        released = replace(
            operation.cleanup_receipt(),
            display_retirement=operation.retirement_receipt,
        )
        if released.cleanup_status is CleanupStatus.CLEANED:
            operation.consume_retirement_release()
            self._forget_clean_admission(operation)
        return released

    def release_admission(self, token: AdmissionToken) -> AdmissionReleased:
        return self.cancel_admission(token)

    def _cleanup_admission(self, operation: _AdmissionOperation) -> None:
        automatic_retry = True
        while operation.begin_cleanup():
            try:
                succeeded = operation.cleanup_once()
            finally:
                cleaned, requested = operation.finish_cleanup()
            if cleaned:
                break
            if requested:
                automatic_retry = False
                continue
            if succeeded or not automatic_retry:
                break
            automatic_retry = False
        self._forget_clean_admission(operation)

    def _forget_clean_admission(
        self, operation: _AdmissionOperation
    ) -> None:
        if operation.cleanup_receipt().cleanup_status is not CleanupStatus.CLEANED:
            return
        if operation.retirement_release_is_pending():
            return
        with self._lock:
            if self._admission is operation:
                self._admission = None
            if (
                operation.retirement_owner is self._retirement
                and operation.retirement_receipt.cleanup_status
                is CleanupStatus.CLEANED
            ):
                self._retirement = None

    def _perform_admission(self, operation: _AdmissionOperation) -> None:
        result: AdmissionReceipt | AdmissionFailure | None = None
        try:
            proof = self._retire_display(operation)
            if proof.cleanup_status is not CleanupStatus.CLEANED:
                result = AdmissionFailure(
                    operation.token,
                    "Historical display cleanup remains pending.",
                )
            elif not operation.cancelled.is_set():
                result = replace(
                    build_admission_receipt(
                        operation.capture,
                        cancelled=operation.cancelled.is_set,
                        session_owner=operation.register_directory_session,
                    ),
                    display_retirement=proof,
                )
        except Exception as error:
            result = AdmissionFailure(
                operation.token,
                detach_exception(error, 'admission').message,
            )
        finally:
            operation.finish_worker(result)
            if operation.cancelled.is_set():
                self._cleanup_admission(operation)

    def _retire_display(self, operation: _AdmissionOperation):
        owner = operation.retirement_owner
        if owner is None:
            operation.retirement_receipt = NO_DISPLAY_RETIREMENT
            return NO_DISPLAY_RETIREMENT
        owner.attempt()
        proof = owner.proof
        operation.retirement_receipt = proof
        if proof.cleanup_status is not CleanupStatus.CLEANED:
            return proof
        with self._lock:
            active = self._active
            if (
                active is not None
                and (
                    active.identity is not owner.run_identity
                    or not active.closed
                )
            ):
                return replace(
                    proof,
                    cleanup_status=CleanupStatus.CLEANUP_PENDING,
                )
            if active is not None:
                self._active = None
        return proof

    def stop(self, run_identity: RunIdentity) -> None:
        run = self._exact_run(run_identity)
        if run is None or run.closed:
            return
        run.stop_requested = True
        run.stop_signal.set()
        session = run.session
        output = run.output
        if output is not None:
            runtime = run.context_runtime
            if runtime is None:
                output.stop()
            else:
                runtime.request_stop(output)
        elif session is not None:
            runtime = run.context_runtime
            if runtime is None:
                session.stop()
            else:
                runtime.stop(session)
        elif run.context_runtime is not None:
            run.context_runtime.request_stop(None)

    def pause(self, run_identity: RunIdentity) -> DurablePaused:
        run = self._exact_run(run_identity)
        if run is None or run.closed:
            raise RuntimeError("acquisition is not pausable")
        runtime = run.context_runtime
        if runtime is None:
            raise RuntimeError("acquisition context is not ready")
        try:
            return runtime.pause(
                run.session,
                run_identity,
                self._join_timeout,
                drain_projection=lambda timeout: self._drain_display_projection(
                    run, timeout
                ),
                session_supplier=lambda: run.session,
            )
        except TerminalPauseFailure as failure:
            failure.cleanup_receipt = self._terminate_projection_failure(
                run, failure.diagnostic
            )
            raise

    def resume(self, run_identity: RunIdentity) -> None:
        run = self._exact_run(run_identity)
        if run is None or run.closed:
            raise RuntimeError("acquisition is not resumable")
        runtime = run.context_runtime
        if runtime is None:
            raise RuntimeError("acquisition context is not ready")
        runtime.resume(run.session)

    def close(self, run_identity: RunIdentity) -> ExecutorClosed:
        with self._lock:
            owner = self._retirement
        if (
            owner is not None
            and owner.run_identity is run_identity
        ):
            receipt = owner.attempt()
            if receipt.cleanup_status is CleanupStatus.CLEANED:
                with self._lock:
                    active = self._active
                    if (
                        active is not None
                        and active.identity is run_identity
                    ):
                        self._active = None
            return receipt
        return self._close_run(run_identity)

    def _close_run(self, run_identity: RunIdentity) -> ExecutorClosed:
        run = self._exact_run(run_identity)
        if run is None:
            return ExecutorClosed(run_identity, CleanupStatus.CLEANUP_PENDING)
        try:
            self.stop(run_identity)
        except Exception as error:
            with run.cleanup_lock:
                run.cleanup_failures.append(detach_exception(error, 'session.stop'))
        context_runtime = run.context_runtime
        if context_runtime is not None:
            context_runtime.retire()
        display_clean = run.display.retire(
            join_timeout=self._join_timeout
        )
        worker = run.worker
        worker_was_alive = worker is not None and worker is not current_thread() and worker.is_alive()
        if worker_was_alive:
            worker.join(timeout=self._join_timeout)
        if worker is not None and worker.is_alive():
            return self._receipt(run)
        if not display_clean and worker_was_alive:
            display_clean = run.display.retire(join_timeout=self._join_timeout)
        if not run.closed:
            worker = self._start_cleanup_retry(run)
            if worker is not None and worker is not current_thread():
                worker.join(timeout=self._join_timeout)
        if not display_clean:
            run.cleanup_status = CleanupStatus.CLEANUP_PENDING
        elif run.closed and all(value is None for value in (
            run.session, run.sink, run.output, run.source, run.resources,
        )) and not getattr(run.display, 'light_1d_cleanup_unresolved', lambda: False)():
            run.cleanup_status = CleanupStatus.CLEANED
        return self._receipt(run)

    def drain_events(self) -> tuple[StandardRunEvent, ...]:
        events: list[StandardRunEvent] = []
        while not self._events.empty():
            events.append(self._events.get())
        return tuple(events)

    def _start_display_projection(self, run: _StandardRun) -> None:
        """Build display publications off the single HDF5 writer thread.

        The completion listener transfers one exact frame label into a
        bounded queue.  Projection is CPU-only and owns no Qt object; public
        events remain arrays-free and are still coalesced by the page's normal
        125 ms drain.
        """
        if run.display_projection_worker is not None:
            raise RuntimeError("display projection worker already exists")
        pending: Queue[object] = Queue(maxsize=_DISPLAY_PROJECTION_FRAMES)
        run.display_projection_queue = pending
        run.display_projection_errors.clear()

        def project_pending() -> None:
            while True:
                item = pending.get()
                failed = False
                try:
                    if item is _DISPLAY_PROJECTION_END:
                        return
                    started = monotonic()
                    if type(item) is _FrameProjectionItem:
                        self._frame_ready_owned(run, item, None, None)
                    elif type(item) is _CheckpointProjectionItem:
                        owner = run.display.artifacts.get(item.artifact)
                        if owner is None:
                            raise RuntimeError("checkpoint projection lost its exact owner")
                        run.display.mark_checkpoint_recoverable(owner, item.labels)
                    else:
                        raise TypeError("unknown display projection item")
                    _perf_add(
                        run,
                        "display_projection",
                        monotonic() - started,
                    )
                except BaseException as error:
                    # Publish the exact projection failure before task_done()
                    # wakes any bounded Pause waiter.
                    run.display_projection_errors.append(error)
                    failed = True
                finally:
                    pending.task_done()
                if failed:
                    return

        worker = Thread(
            target=project_pending,
            name="scattering-display-projection",
            daemon=True,
        )
        run.display_projection_worker = worker
        worker.start()

    @staticmethod
    def _drain_display_projection(
        run: _StandardRun,
        timeout: float,
    ) -> bool:
        """Wait for already-accepted projection work without retiring it."""

        pending = run.display_projection_queue
        worker = run.display_projection_worker
        if pending is None or worker is None:
            if run.display_projection_errors:
                raise TerminalPauseFailure(
                    run.display_projection_errors[0]
                )
            return True
        deadline = monotonic() + max(0.0, float(timeout))
        with pending.all_tasks_done:
            while pending.unfinished_tasks:
                if run.display_projection_errors:
                    raise TerminalPauseFailure(
                        run.display_projection_errors[0]
                    )
                if not worker.is_alive():
                    raise TerminalPauseFailure(RuntimeError(
                        "display projection worker stopped before durable pause"
                    ))
                remaining = deadline - monotonic()
                if remaining <= 0.0:
                    return False
                pending.all_tasks_done.wait(timeout=min(remaining, 0.01))
            if run.display_projection_errors:
                raise TerminalPauseFailure(
                    run.display_projection_errors[0]
                )
            if not worker.is_alive():
                raise TerminalPauseFailure(RuntimeError(
                    "display projection worker stopped before durable pause"
                ))
        return True

    def _terminate_projection_failure(
        self,
        run: _StandardRun,
        diagnostic: DetachedDiagnostic,
    ) -> ExecutorClosed:
        """Fail one exact active run and retire its non-retryable owners."""

        with run.cleanup_lock:
            if run.command_failure is None:
                run.command_failure = diagnostic
            if run.primary is None:
                run.primary = diagnostic
        run.stop_requested = True
        run.stop_signal.set()
        session = run.session
        output = run.output
        runtime = run.context_runtime
        if output is not None:
            try:
                if runtime is None:
                    output.stop()
                else:
                    runtime.request_stop(output)
            except BaseException as error:
                with run.cleanup_lock:
                    run.cleanup_failures.append(detach_exception(
                        error, "dynamic_output.stop"
                    ))
        elif session is not None:
            try:
                if runtime is None:
                    session.stop()
                else:
                    runtime.terminal_stop(session)
            except BaseException as error:
                with run.cleanup_lock:
                    run.cleanup_failures.append(detach_exception(
                        error, "session.stop"
                    ))
        projection_retired = self._retire_failed_display_projection(run)
        worker = run.worker
        if (
            worker is not None
            and worker is not current_thread()
            and worker.ident is not None
        ):
            worker.join(timeout=self._join_timeout)
        if projection_retired and (worker is None or not worker.is_alive()):
            receipt = self._cleanup(run, diagnostic)
        else:
            run.cleanup_status = CleanupStatus.CLEANUP_PENDING
            receipt = self._receipt(run)
        completed, total = _terminal_durable_progress(run)
        self._terminal_event(
            run,
            StandardEventKind.FAILED,
            receipt,
            completed,
            total,
        )
        return receipt

    def _retire_failed_display_projection(self, run: _StandardRun) -> bool:
        """Discard non-retryable queued work and release the failed worker."""

        pending = run.display_projection_queue
        worker = run.display_projection_worker
        if (
            worker is not None
            and worker is not current_thread()
            and worker.ident is not None
            and worker.is_alive()
        ):
            worker.join(timeout=self._join_timeout)
        if worker is not None and worker.is_alive():
            with run.cleanup_lock:
                run.cleanup_failures.append(detach_exception(
                    TimeoutError("display projection worker did not retire"),
                    "display_projection.retire",
                ))
            return False
        if pending is not None:
            while True:
                try:
                    pending.get_nowait()
                except Empty:
                    break
                else:
                    pending.task_done()
        run.display_projection_queue = None
        run.display_projection_worker = None
        run.display_projection_errors.clear()
        return True

    @staticmethod
    def _finish_display_projection(run: _StandardRun) -> None:
        pending = run.display_projection_queue
        worker = run.display_projection_worker
        if pending is None or worker is None:
            return
        while worker.is_alive():
            try:
                pending.put(_DISPLAY_PROJECTION_END, timeout=0.05)
            except Full:
                continue
            worker.join()
            break
        run.display_projection_queue = None
        run.display_projection_worker = None
        if run.display_projection_errors:
            raise run.display_projection_errors[0]

    def frame_catalog(
        self, run_identity: RunIdentity
    ) -> DisplayFrameCatalog | None:
        run = self._exact_run(run_identity)
        if run is None:
            return None
        with self._lock:
            return run.display.catalog_snapshot()

    def processed_live_revisions(
        self,
        run_identity: RunIdentity,
    ) -> tuple[LiveDirectoryAttempt, ...]:
        """Exact latest processed source attempt per physical Live target."""

        run = self._exact_run(run_identity)
        if run is None:
            return ()
        with run.live_revision_lock:
            return tuple(run.processed_live_revisions.values())

    def acquisition_context(self, run_identity: RunIdentity):
        run = self._exact_run(run_identity)
        runtime = None if run is None else run.context_runtime
        return None if runtime is None else runtime.context

    @staticmethod
    def _commit_gate(run: _StandardRun):
        context = None if run.context_runtime is None else run.context_runtime.context
        return None if context is None else context.commit_gate

    def _drain_light_lineage(
        self, run: _StandardRun, owner: DisplayArtifact,
    ) -> None:
        self._finish_display_projection(run)
        run.display.verify_light_1d(owner, self._commit_gate(run))

    def _release_predecessor_before_target(
        self, run: _StandardRun, target: Path,
    ) -> None:
        output = run.output
        predecessor = (
            None if output is None
            else output.predecessor_owner_for_target(target)
        )
        if predecessor is None:
            return
        self._settle_predecessor(run, output, predecessor)

    def _settle_predecessor(self, run, output, predecessor) -> None:
        result = output.finish_predecessor(predecessor)
        self._finish_display_projection(run)
        self._project_new_durable(run, output)
        if getattr(result, "failed", False):
            raise RuntimeError(result.error or "predecessor settlement failed")
        run.display.mark_hydration_closed(predecessor)
        if not run.display.release_light_1d(
            predecessor, reason="lineage-replaced",
        ):
            raise RuntimeError("prior light-1D lineage cleanup remains pending")
        output.drop_released_predecessor(predecessor)

    def _construct(self, run: _StandardRun, *, item: PlannedOutput | None=None, labels: tuple[int, ...] | None=None, decision: AdmittedOutput | None=None) -> _StandardRun:
        configuration, capture = (run.configuration, run.capture)
        if configuration is None or capture is None:
            raise RuntimeError('run construction shell is incomplete')
        source_spec = item.source_spec if item is not None else configuration.thaw_source_spec()
        if source_spec is None or not configuration.poni_file or (not configuration.save_path):
            raise ValueError('Standard execution requires source, PONI, and output paths')
        artifact = item.target if item is not None else Path(configuration.save_path)
        output = run.output
        predecessor = (
            None if output is None or item is None
            else output.predecessor_owner(item)
        )
        if predecessor is not None:
            self._settle_predecessor(run, output, predecessor)
        # Establish the item identity before validation/open.  A failure at any
        # later construction seam must be reported against this artifact, not
        # the previously completed one.
        run.artifact = artifact
        cancelled = lambda: run.stop_requested

        def discard_source() -> None:
            source_owner = run.source
            if source_owner is not None:
                close = getattr(source_owner, 'close', None)
                if callable(close):
                    close()
            run.source = None
            run.scan = None

        try:
            if item is not None:
                validate_planned_source(item, cancelled=cancelled)
            if item is not None and item.candidate is not None and (item.descriptor is not None) and (item.descriptor.kind in {SourceKind.NEXUS_STACK, SourceKind.EIGER_MASTER}):
                cursor = open_container_cursor(item.source_path, entry=item.source_spec.entry or 'entry', candidate=item.candidate)
                try:
                    run.source = NexusStackSource(item.source_path, entry=item.source_spec.entry or 'entry', cursor=cursor)
                except Exception:
                    cursor.close()
                    raise
            else:
                run.source = open_source(source_spec)
            admission = None if run.resources is None else run.resources.admission
            assets = None if admission is None else admission.scientific_assets
            calibration = (
                assets.detector_calibration
                if assets is not None else load_poni(configuration.poni_file)
            )
            if calibration is None:
                raise ValueError('accepted PONI asset is unavailable')
            poni = getattr(calibration, "poni", calibration)
            run.scan = run.source.to_scan(poni=poni, integrator=poni_to_integrator(calibration), output_path=artifact)
        except SourceRevisionChanged:
            discard_source()
            raise
        except Exception as error:
            if (
                isinstance(error, RuntimeError)
                and error.args == ('admission cancelled',)
            ):
                discard_source()
                raise
            if item is not None:
                try:
                    validate_planned_source(item, cancelled=cancelled)
                except SourceRevisionChanged as drift:
                    discard_source()
                    raise drift from error
                except RuntimeError as cancellation:
                    if cancellation.args == ('admission cancelled',):
                        discard_source()
                        raise cancellation from error
                    raise
            raise
        run.scan.gi_config = configuration.gi.scan_config().copy()
        run.current_total = (
            item.source_stamp.frame_count
            if item is not None
            else len(run.scan.frames)
        )
        plan = replace(
            native_int_reduction_plan(configuration),
            mask=None if assets is None else assets.mask,
        )
        try:
            npt = int(configuration.bai_1d_args.get("npt", 0))
        except (TypeError, ValueError):
            npt = 0
        if item is None or decision is None:
            raise RuntimeError("dynamic output requires one admitted output")
        output = run.output
        if output is None:
            output = DynamicOutputAdapter(configuration); run.output = output
        preparation = output.prepare_admission(
            run.scan, plan, item, decision, run.stop_signal,
            qualify=lambda policy, prior: _qualify_background_bindings(
                configuration, run.scan, item, decision, policy, prior,
                run.stop_signal))
        if configuration.output_mode == "Overwrite" and not target_state_matches(preparation.effective):
            raise RuntimeError(f"output target changed after Background qualification: {item.target}")
        run.display.set_factories(FrameRecordStore, PublicationStore)
        if not run.display.configured:
            descriptor_bytes = (
                item.descriptor.frame_bytes
                if item is not None and item.descriptor is not None
                else None
            )
            first_image = next(
                (
                    np.asarray(image)
                    for frame in getattr(run.scan, "frames", ())
                    if (image := getattr(frame, "image", None)) is not None
                ),
                None,
            )
            run.display.configure(
                partition_count=run.pending_partition_count,
                npt=npt,
                frame_bytes=(
                    descriptor_bytes
                    if descriptor_bytes is not None
                    else None if first_image is None else first_image.nbytes
                ),
            )
        mask = None if assets is None else assets.mask
        owner = run.display.artifacts.get(str(artifact))
        newly_adopted_owner = owner is None
        if owner is None:
            if configuration.live_mode and run.display.artifacts:
                run.display.admit_additional_partition()
            owner = run.display.add_artifact(
                artifact,
                str(
                    getattr(run.scan, "name", "")
                    or Path(source_spec.uri).stem
                    or "scan"
                ),
                mask=mask,
                mask_saturation=bool(
                    getattr(plan, "mask_saturation", False)
                ),
                measurement_mode=(
                    "GI" if configuration.gi.enabled else "Standard"
                ),
                gi_incidence_motor=(
                    configuration.gi.incidence_motor
                    if configuration.gi.enabled else ""
                ),
                gi_resolved_motor=(
                    configuration.gi.effective_motor
                    if configuration.gi.enabled else ""
                ),
                gi_mode_1d=(
                    configuration.gi.mode_1d if configuration.gi.enabled else ""
                ),
                gi_mode_2d=(
                    configuration.gi.mode_2d if configuration.gi.enabled else ""
                ),
                source_base=configuration.project_root or None,
                wavelength_m=(
                    float(poni.wavelength)
                    if getattr(poni, "wavelength", None)
                    and float(poni.wavelength) > 0.0
                    else None
                ),
            )
        run.records = owner.records
        run.display.bind_checkpoint_hydration(owner)
        run_provenance = configuration.as_provenance()
        if admission is not None:
            run_provenance['scientific_signature'] = admission.candidate.processing_mapping()
        if run.stop_requested:
            discard_source()
            raise RuntimeError("admission cancelled")
        run.session, created = output.activate(
            preparation,
            record_store=run.records,
            run_provenance=run_provenance,
            cancelled=cancelled,
            publication_store=owner.publications,
            display_owner=owner,
            display_state=run.display,
            source_owner=run.source,
            gui_thread_id=run.gui_thread_id,
            light_cancel=lambda: run.display.cancel_light_1d(
                owner, self._commit_gate(run),
            ),
            light_drain=lambda: self._drain_light_lineage(run, owner),
            light_verify=lambda: run.display.verify_light_1d(
                owner, self._commit_gate(run),
            ),
            on_frame_completed=lambda event: self._frame_ready(run, event),
            on_checkpoint_recoverable=lambda event: self._checkpoint_ready(run, event),
            resource_fact_sink=run.resource_facts.append,
        )
        run.sink = output
        write_labels = set(output.write_labels)
        persisted_prefix_labels = output.persisted_prefix_labels
        run.current_completed = max(0, run.current_total - len(write_labels))
        run.current_published = run.current_completed
        run.current_epoch_published = 0
        self._seed_persisted_prefix_navigation(
            run,
            owner,
            persisted_prefix_labels,
            newly_adopted=newly_adopted_owner,
        )
        if run.session is not None and run.display_projection_worker is None:
            self._start_display_projection(run)
        if run.session is not None or persisted_prefix_labels:
            self._adopt_acquisition_context(
                run, owner, source_path=str(source_spec.uri)
            )
        return run

    @staticmethod
    def _seed_persisted_prefix_navigation(
        run: _StandardRun,
        owner: DisplayArtifact,
        labels: tuple[int, ...],
        *,
        newly_adopted: bool,
    ) -> None:
        if type(newly_adopted) is not bool:
            raise TypeError("persisted-prefix adoption flag must be exact")
        if not newly_adopted:
            return
        if type(labels) is not tuple:
            raise TypeError("persisted prefix labels must be an exact tuple")
        if len(labels) != run.current_completed:
            raise ValueError("persisted prefix cardinality changed")
        if any(type(label) is not int for label in labels):
            raise TypeError("persisted prefix labels must be exact integers")
        capacity = run.display.navigation_capacity
        retained = labels[-capacity:]
        absolute_base = run.completed + len(labels) - len(retained)
        rows = tuple(
            (
                owner.source_scan,
                str(owner.artifact),
                label,
                absolute_base + offset,
            )
            for offset, label in enumerate(retained, start=1)
        )
        run.display.seed_navigation_prefix_at_work_ordinals(rows)

    def _adopt_acquisition_context(
        self, run: _StandardRun, owner: DisplayArtifact, *, source_path: str
    ):
        created = run.context_runtime is None
        runtime = run.context_runtime or AcquisitionRuntime()
        run.context_runtime = runtime
        previous = (
            None
            if runtime.context is None
            else runtime.context.hydration_owner
        )
        context = runtime.adopt(run, owner, source_path)
        if created or context.hydration_owner != previous:
            self._events.put(StandardRunEvent(
                run.identity, StandardEventKind.CONTEXT_READY
            ))
        return context

    def _run(self, run: _StandardRun) -> None:
        started_at = monotonic()
        if run.perf_quartiles_enabled:
            run.perf_started_at = started_at
        primary: DetachedDiagnostic | None = None
        stopped = False
        try:
            core_count = int(getattr(run.configuration, "max_cores", 0))
        except (TypeError, ValueError):
            core_count = 0
        try:
            configuration = run.configuration
            if run.scan is not None and run.session is not None:
                stopped = self._execute_current(run, construct=False)
            elif type(configuration) is not FrozenRunConfiguration:
                stopped = self._execute_current(run)
            else:
                stopped = self._execute_admitted(run)
        except Exception as error:
            if (
                run.stop_requested
                and isinstance(error, RuntimeError)
                and error.args == ('admission cancelled',)
            ):
                stopped = True
            else:
                primary = detach_exception(error)
        if run.command_failure is not None:
            primary = run.command_failure
        work_elapsed = max(0.0, monotonic() - started_at)
        cleanup_started_at = monotonic()
        receipt = self._cleanup(run, primary)
        receipt = self._release_unpublished_terminal(run, receipt)
        completed, total = _terminal_durable_progress(run)
        elapsed = max(0.0, monotonic() - started_at)
        cleanup_elapsed = max(
            0.0,
            elapsed - (cleanup_started_at - started_at),
        )
        if primary is None and receipt.cleanup_status is CleanupStatus.CLEANED:
            kind = StandardEventKind.STOPPED if stopped else StandardEventKind.FINISHED
        else:
            kind = StandardEventKind.FAILED
        self._terminal_event(
            run,
            kind,
            receipt,
            completed,
            total,
            elapsed=elapsed,
            work_elapsed=work_elapsed,
            cleanup_elapsed=cleanup_elapsed,
            core_count=core_count,
        )

    def _release_unpublished_terminal(
        self,
        run: _StandardRun,
        receipt: ExecutorClosed,
    ) -> ExecutorClosed:
        """Retire a clean run whose display identity was never published."""

        runtime = run.context_runtime
        if (
            receipt.cleanup_status is not CleanupStatus.CLEANED
            or runtime is not None and runtime.context is not None
        ):
            return receipt
        stage = "unpublished_runtime.retire"
        try:
            if runtime is not None:
                runtime.retire()
            stage = "unpublished_display.retire"
            display_clean = run.display.retire(
                join_timeout=self._join_timeout
            )
        except BaseException as error:
            with run.cleanup_lock:
                run.cleanup_failures.append(detach_exception(
                    error, stage,
                ))
                run.cleanup_status = CleanupStatus.CLEANUP_PENDING
                return self._receipt(run)
        if not display_clean:
            with run.cleanup_lock:
                run.cleanup_status = CleanupStatus.CLEANUP_PENDING
                return self._receipt(run)
        with self._lock:
            if self._active is run:
                run.unpublished_display_retired = True
        return receipt

    def _execute_admitted(self, run: _StandardRun) -> bool:
        receipt = None if run.resources is None else run.resources.admission
        if type(receipt) is not AdmissionReceipt:
            raise RuntimeError('executor lost its consumed admission receipt')
        if run.stop_requested:
            return True
        outputs = receipt.outputs
        resources = run.resources
        try:
            with _run_effect(run):
                validate_admitted_receipt(
                    receipt,
                    None if resources is None else resources.directory_session,
                    cancelled=lambda: run.stop_requested,
                )
        except RuntimeError as error:
            if (
                run.stop_requested
                and error.args == ('admission cancelled',)
            ):
                return True
            raise
        if run.stop_requested:
            return True
        deferred = receipt.deferred_directory
        if type(deferred) is DeferredDirectoryPlan:
            if deferred.live:
                return self._execute_live_directory(run, receipt, deferred)
            return self._execute_deferred_directory(run, receipt, deferred)
        run.pending_partition_count = max(1, len(outputs))
        run.total = sum(output.item.source_stamp.frame_count for output in outputs)
        run.files_discovered = receipt.directory_discovered_file_count
        output_file_counts, unowned_files = _eager_directory_file_counts(
            outputs,
            getattr(receipt, "directory_discovered_paths", ()),
        )
        if run.files_discovered:
            run.files_skipped = unowned_files
        self._events.put(StandardRunEvent(
            run.identity,
            StandardEventKind.DISCOVERY,
            total=run.total,
            detail=(
                _directory_status(run)
                if run.files_discovered
                else f"{('GI' if run.configuration.gi.enabled else 'Standard')} · {len(outputs)} admitted candidates"
            ),
            **_directory_event_fields(run),
        ))
        stopped = False
        for decision, output_file_count in zip(
            outputs, output_file_counts, strict=True
        ):
            item = decision.item
            if run.stop_requested:
                stopped = True
                break
            run.current_total = item.source_stamp.frame_count
            run.current_file_total = (
                output_file_count
                if run.files_discovered
                else 0
            )
            run.current_files_incremental = (
                bool(run.current_file_total)
                and item.source_spec.kind is SourceKind.TIFF_SERIES
            )
            self._construct(run, item=item, labels=decision.labels, decision=decision)
            run.completed += run.current_completed
            stopped = self._execute_current(run, construct=False) or stopped
            if run.current_files_incremental:
                run.files_processed += min(
                    run.current_file_total,
                    run.current_completed,
                )
            elif (
                run.current_file_total
                and run.current_completed >= run.current_total
            ):
                run.files_processed += run.current_file_total
            run.current_file_total = 0
            run.current_files_incremental = False
            if run.files_discovered:
                self._events.put(StandardRunEvent(
                    run.identity,
                    StandardEventKind.DISCOVERY,
                    completed=run.completed,
                    total=run.total,
                    artifact=str(item.target),
                    detail=_directory_status(run),
                    artifact_completed=run.current_published,
                    artifact_total=run.current_total,
                    **_directory_event_fields(run),
                ))
        return stopped

    def _execute_live_directory(
        self,
        run: _StandardRun,
        receipt: AdmissionReceipt,
        deferred: DeferredDirectoryPlan,
    ) -> bool:
        """Watch one directory and extend each exact output lineage in place."""

        resources = run.resources
        session = None if resources is None else resources.directory_session
        configuration = run.configuration
        if (
            session is None
            or type(configuration) is not FrozenRunConfiguration
            or not deferred.live
        ):
            raise RuntimeError("live directory run lost its source owner")

        settled_revisions: dict[
            str,
            tuple[tuple[object, ...], SourceExecutionIdentityV1 | None],
        ] = {}
        retry_revisions: dict[str, tuple[object, ...]] = {}
        discovered_paths: set[Path] = set(deferred.discovered_paths)
        processed_paths: set[Path] = set()
        skipped_paths: set[Path] = set()
        last_projection: tuple[int, int, int, int, int] | None = None

        def publish(*, force: bool = False) -> None:
            nonlocal last_projection
            run.files_discovered = len(discovered_paths)
            run.files_processed = len(processed_paths)
            run.files_skipped = len(skipped_paths - processed_paths)
            projection = (
                run.files_processed,
                run.files_skipped,
                max(
                    0,
                    run.files_discovered
                    - run.files_processed
                    - run.files_skipped,
                ),
                run.files_discovered,
            )
            if not force and projection == last_projection:
                return
            last_projection = projection
            self._events.put(StandardRunEvent(
                run.identity,
                StandardEventKind.DISCOVERY,
                completed=run.completed,
                total=run.total,
                artifact=str(run.artifact),
                detail=_directory_status(run, state="Watching"),
                artifact_completed=run.current_published,
                artifact_total=run.current_total,
                **_directory_event_fields(run),
            ))

        with _run_effect(run):
            publish(force=True)
        while not run.stop_requested:
            with _run_effect(run):
                observation = session.observe(refresh=True)
                groups = live_directory_groups(receipt, observation)
                for group in groups:
                    discovered_paths.update(group.physical_paths)
                publish()

            for group in groups:
                if run.stop_requested:
                    return True
                revision = tuple(group.revision)
                target_key = _physical_file_key(group.target)
                settled = settled_revisions.get(target_key)
                if settled is not None and settled[0] == revision:
                    settled_identity = settled[1]
                    if settled_identity is None:
                        continue
                    with run.live_revision_lock:
                        retained = run.processed_live_revisions.get(target_key)
                    if (
                        retained is None
                        or retained.decision is None
                        or retained.decision.item.source_stamp.execution_identity_v1
                        != settled_identity
                    ):
                        raise RuntimeError(
                            "settled Live source identity lost its exact attempt"
                        )
                    with _run_effect(run):
                        try:
                            validate_planned_source(
                                retained.decision.item,
                                cancelled=lambda: run.stop_requested,
                            )
                        except SourceRevisionChanged:
                            settled_revisions.pop(target_key, None)
                        else:
                            continue
                try:
                    with _run_effect(run):
                        reprobe = (
                            retry_revisions.pop(target_key, None) == revision
                        )
                        self._release_predecessor_before_target(
                            run, group.target,
                        )
                        attempt = materialize_live_directory_group(
                            receipt,
                            configuration,
                            session,
                            group,
                            cancelled=lambda: run.stop_requested,
                            reprobe=reprobe,
                        )
                        if attempt.state is ProbeState.IN_PROGRESS:
                            if attempt.revision_changed:
                                retry_revisions[target_key] = revision
                            continue
                        if attempt.state is not ProbeState.READY:
                            settled_revisions[target_key] = (revision, None)
                            skipped_paths.update(group.physical_paths)
                            publish()
                            continue

                        decision = attempt.decision
                        if decision is None:  # pragma: no cover - typed invariant
                            raise RuntimeError(
                                "READY Live attempt lost its output"
                            )
                        if target_key != _physical_file_key(decision.item.target):
                            raise RuntimeError(
                                "Live output target changed after JIT"
                            )
                        item = decision.item
                        run.current_file_total = len(group.physical_paths)
                        run.current_files_incremental = (
                            item.source_spec.kind is SourceKind.TIFF_SERIES
                        )
                        try:
                            self._construct(
                                run,
                                item=item,
                                labels=decision.labels,
                                decision=decision,
                            )
                        except SourceRevisionChanged:
                            source_owner = run.source
                            if source_owner is not None:
                                close = getattr(source_owner, "close", None)
                                if callable(close):
                                    close()
                            run.source = None
                            run.scan = None
                            run.records = None
                            run.frames_by_label.clear()
                            run.current_total = 0
                            run.current_completed = 0
                            run.current_published = 0
                            run.current_file_total = 0
                            run.current_files_incremental = False
                            retry_revisions[target_key] = revision
                            publish(force=True)
                            continue
                        output = run.output
                        if output is None:
                            raise RuntimeError(
                                "Live output graph was not constructed"
                            )
                        run.total += len(output.write_labels)
                        publish(force=True)
                except RuntimeError as error:
                    if (
                        run.stop_requested
                        and error.args == ("admission cancelled",)
                    ):
                        return True
                    raise
                def settle_live(stopped: bool) -> None:
                    if run.current_files_incremental:
                        processed_paths.update(
                            group.physical_paths[:run.current_completed]
                        )
                    elif run.current_completed >= run.current_total:
                        processed_paths.update(group.physical_paths)
                    skipped_paths.difference_update(processed_paths)
                    if (
                        not stopped
                        and run.current_completed >= run.current_total
                    ):
                        with run.live_revision_lock:
                            run.processed_live_revisions[target_key] = attempt
                        settled_revisions[target_key] = (
                            revision,
                            decision.item.source_stamp.execution_identity_v1,
                        )
                    run.current_file_total = 0
                    run.current_files_incremental = False
                    publish(force=True)

                stopped = self._execute_current(
                    run,
                    construct=False,
                    retain_session=True,
                    settle=settle_live,
                )
                if stopped or run.stop_requested:
                    return True

            run.stop_signal.wait(_LIVE_DIRECTORY_POLL_S)
        return True

    def _execute_deferred_directory(
        self,
        run: _StandardRun,
        receipt: AdmissionReceipt,
        deferred: DeferredDirectoryPlan,
    ) -> bool:
        resources = run.resources
        session = None if resources is None else resources.directory_session
        if session is None:
            raise RuntimeError("deferred directory run lost its index session")
        run.pending_partition_count = max(1, len(deferred.entries))
        run.files_discovered = deferred.discovered_file_count
        run.files_skipped = 0
        claimed_files: set[Path] = set()
        candidate_owners = _candidate_file_owners(tuple(
            entry.physical_paths for entry in deferred.entries
        ))
        self._events.put(StandardRunEvent(
            run.identity,
            StandardEventKind.DISCOVERY,
            completed=run.completed,
            total=run.total,
            detail=_directory_status(run),
            **_directory_event_fields(run),
        ))
        stopped = False
        for output_index, entry in enumerate(deferred.entries):
            if run.stop_requested:
                stopped = True
                break
            try:
                self._release_predecessor_before_target(
                    run, entry.target,
                )
                decision, _ready_files, skipped_files = (
                    materialize_deferred_output(
                        receipt,
                        session,
                        entry,
                        cancelled=lambda: run.stop_requested,
                    )
                )
            except RuntimeError as error:
                if (
                    run.stop_requested
                    and error.args == ("admission cancelled",)
                ):
                    stopped = True
                    break
                raise
            if decision is None:
                skipped_claims = _claim_physical_files(
                    tuple(
                        Path(state.path)
                        for state in entry.protected_states
                    ),
                    deferred.discovered_paths,
                    claimed_files,
                    candidate_owners=candidate_owners,
                    output_index=output_index,
                )
                run.files_skipped += max(skipped_files, skipped_claims)
                self._events.put(StandardRunEvent(
                    run.identity,
                    StandardEventKind.DISCOVERY,
                    completed=run.completed,
                    total=run.total,
                    detail=_directory_status(run),
                    **_directory_event_fields(run),
                ))
                continue
            run.files_skipped += skipped_files
            item = decision.item
            run.current_total = item.source_stamp.frame_count
            run.total += run.current_total
            run.current_file_total = _claim_output_physical_files(
                item,
                deferred.discovered_paths,
                claimed_files,
                candidate_owners=candidate_owners,
                output_index=output_index,
            )
            run.current_files_incremental = (
                item.source_spec.kind is SourceKind.TIFF_SERIES
            )
            self._construct(
                run,
                item=item,
                labels=decision.labels,
                decision=decision,
            )
            run.completed += run.current_completed
            self._events.put(StandardRunEvent(
                run.identity,
                StandardEventKind.DISCOVERY,
                completed=run.completed,
                total=run.total,
                artifact=str(item.target),
                detail=_directory_status(run),
                artifact_completed=run.current_published,
                artifact_total=run.current_total,
                **_directory_event_fields(run),
            ))
            stopped = self._execute_current(run, construct=False) or stopped
            if run.current_files_incremental:
                run.files_processed += min(
                    run.current_file_total,
                    run.current_completed,
                )
            elif run.current_completed >= run.current_total:
                run.files_processed += run.current_file_total
            run.current_file_total = 0
            run.current_files_incremental = False
            self._events.put(StandardRunEvent(
                run.identity,
                StandardEventKind.DISCOVERY,
                completed=run.completed,
                total=run.total,
                artifact=str(item.target),
                detail=_directory_status(run),
                artifact_completed=run.current_published,
                artifact_total=run.current_total,
                **_directory_event_fields(run),
            ))
            if stopped:
                break
        if not stopped:
            run.files_skipped = max(
                run.files_skipped,
                run.files_discovered - run.files_processed,
            )
        return stopped

    def _execute_current(
        self,
        run: _StandardRun,
        *,
        construct: bool = True,
        retain_session: bool = False,
        settle: Callable[[bool], None] | None = None,
    ) -> bool:
        if construct and run.session is None:
            self._construct(run)
        scan, session, output = (run.scan, run.session, run.output)
        if (
            scan is not None
            and session is None
            and output is not None
            and not output.write_labels
        ):
            with _run_effect(run):
                close = getattr(run.source, "close", None)
                if callable(close):
                    close()
                run.source = None
                run.scan = None
                run.records = None
                run.frames_by_label.clear()
                if run.artifact not in run.artifacts:
                    run.artifacts.append(run.artifact)
                owner = run.display.artifacts.get(str(run.artifact))
                if owner is None:
                    raise RuntimeError(
                        "persisted-prefix display artifact is missing"
                    )
                run.display.mark_hydration_closed(owner)
                stopped = bool(run.stop_requested)
                if settle is not None:
                    settle(stopped)
                return stopped
        if scan is None or session is None or output is None:
            raise RuntimeError('scattering executor lost its constructed owners')
        write_labels = set(output.write_labels)
        run.frames_by_label = {
            int(frame.index): frame
            for frame in scan.frames
            if int(frame.index) in write_labels
        }
        with _run_effect(run):
            session.start()
        if isinstance(run.source, NexusStackSource) and write_labels:
            self._submit_container_source(run, output)
        else:
            for frame in scan.frames:
                if int(frame.index) not in write_labels:
                    continue
                if run.stop_requested:
                    break
                runtime = run.context_runtime
                submit_started = monotonic()
                if not _background_ready(run, output, frame): break
                if not (
                    output.submit(frame)
                    if runtime is None
                    else runtime.submit(output, frame)
                ):
                    _perf_add(run, "submit_wait", monotonic() - submit_started)
                    break
                _perf_add(run, "submit_wait", monotonic() - submit_started)
        with _run_effect(run):
            finish_started = monotonic()
            session_error: BaseException | None = None
            projection_error: BaseException | None = None
            result = None
            finished_current = not (
                retain_session and not run.stop_requested
            )
            try:
                if not finished_current:
                    result = output.commit_epoch()
                else:
                    result = output.finish_current()
            except BaseException as error:
                session_error = error
            if session_error is None:
                try:
                    self._finish_display_projection(run)
                except BaseException as error:
                    projection_error = error
            _perf_add(run, "finish_wait", monotonic() - finish_started)
            if session_error is not None:
                raise session_error.with_traceback(session_error.__traceback__)
            if projection_error is not None:
                try:
                    self.stop(run.identity)
                except BaseException as error:
                    run.cleanup_failures.append(detach_exception(
                        error, "dynamic_output.stop"
                    ))
                raise projection_error.with_traceback(
                    projection_error.__traceback__
                )
            self._project_new_durable(run, output)
            with run.light_projection_error_lock:
                light_error = run.light_projection_error
            if light_error is not None:
                raise light_error.with_traceback(light_error.__traceback__)
            if result is None:  # pragma: no cover - defensive type narrowing
                raise RuntimeError(
                    "scattering session returned no terminal result"
                )
            if run.perf_enabled:
                snapshot = getattr(session, "perf_snapshot", None)
                if callable(snapshot):
                    for key, elapsed in snapshot().items():
                        _perf_add(run, key, elapsed)
            if getattr(result, "failed", False):
                raise RuntimeError(
                    result.error or "scattering reduction failed"
                )
            if finished_current:
                run.terminal_commit_identity = (
                    _session_terminal_commit_identity(session, run.artifact)
                )
                owner = run.display.artifacts.get(str(run.artifact))
                if owner is None:
                    raise RuntimeError("finished display artifact is missing")
                run.display.mark_hydration_closed(owner)
            run.current_published = max(
                run.current_published, run.current_completed,
            )
            close = getattr(run.source, "close", None)
            if callable(close):
                close()
            run.source = None
            run.scan = None
            run.records = None
            run.frames_by_label.clear()
            stopped = bool(
                getattr(result, "cancelled", False) or run.stop_requested
            )
            if settle is not None:
                settle(stopped)
            return stopped

    @staticmethod
    def _project_new_durable(run: _StandardRun, output) -> None:
        def apply(artifact, durable_labels, new_labels):
            owner = run.display.artifacts.get(artifact)
            if owner is None:
                raise RuntimeError(
                    f"durable output lost display artifact {artifact}"
                )
            run.display.mark_durable(owner, durable_labels)
            run.completed += len(new_labels)
            path = Path(artifact)
            if path == run.artifact:
                run.current_completed = min(
                    run.current_total,
                    run.current_completed + len(new_labels),
                )
                run.current_published = max(
                    run.current_published, run.current_completed,
                )
            if new_labels and path not in run.artifacts:
                run.artifacts.append(path)
        output.project_new_durable(apply)

    @staticmethod
    def _submit_container_source(run: _StandardRun, output: Any) -> None:
        """Overlap bounded container reads with reduction backpressure.

        The executor thread remains the sole HDF5/cursor owner.  One bounded
        consumer thread performs only ``session.submit`` calls, allowing the
        next source frame to decode while the previous frame waits for a
        reduction slot.  The exact allocation grant bounds that native-frame
        queue; no h5py object crosses threads.
        """
        source = run.source
        if not isinstance(source, NexusStackSource):
            raise TypeError("container submission requires NexusStackSource")

        def publish_direct_fact() -> None:
            take = getattr(source, "take_direct_chunk_fact", None)
            fact = take() if callable(take) else None
            if fact is not None:
                run.resource_facts.append(fact)
                logger.info("%s", fact.log_line("[SOURCE-READ]"))

        wanted = tuple(run.frames_by_label)
        if wanted != tuple(source.frame_indices):
            observed_fallbacks = 0
            try:
                for label in wanted:
                    if run.stop_requested:
                        return
                    frame = run.frames_by_label[label]
                    image = source.load_frame(label)
                    observed_fallbacks += 1
                    if not _background_ready(run, output, frame): break
                    runtime = run.context_runtime
                    accepted = (
                        output.submit(frame, image)
                        if runtime is None
                        else runtime.submit(output, frame, image)
                    )
                    if not accepted:
                        break
            finally:
                note = getattr(source, "note_direct_chunk_bypass", None)
                if callable(note):
                    note("non-complete source selection uses random access",
                         observed_fallbacks)
                publish_direct_fact()
            return

        allocation = getattr(source, "allocation", None)
        queue_depth = (_SOURCE_PREFETCH_FRAMES if allocation is None
                       else int(allocation.queue_depth))
        pending: Queue[object] = Queue(maxsize=queue_depth)
        accepting = Event()
        accepting.set()
        consumer_done = Event()
        consumer_errors: list[BaseException] = []

        def submit_pending() -> None:
            try:
                while True:
                    if not accepting.is_set():
                        return
                    try:
                        item = pending.get(timeout=0.05)
                    except Empty:
                        continue
                    if item is _SOURCE_SUBMISSION_END:
                        return
                    frame, image = item
                    if not accepting.is_set() or run.stop_requested:
                        accepting.clear()
                        return
                    if not _background_ready(run, output, frame):
                        accepting.clear(); return
                    runtime = run.context_runtime
                    submit_started = monotonic()
                    accepted = (
                        output.submit(frame, image)
                        if runtime is None
                        else runtime.submit(output, frame, image)
                    )
                    _perf_add(
                        run, "submit_wait", monotonic() - submit_started
                    )
                    if not accepted:
                        accepting.clear()
                        return
            except BaseException as error:
                consumer_errors.append(error)
                accepting.clear()
            finally:
                consumer_done.set()

        consumer = Thread(
            target=submit_pending,
            name="scattering-source-submit",
            daemon=True,
        )
        consumer.start()

        def enqueue(item: object) -> bool:
            while accepting.is_set() and not consumer_done.is_set():
                if run.stop_requested:
                    accepting.clear()
                    return False
                try:
                    pending.put(item, timeout=0.05)
                    return True
                except Full:
                    continue
            return False

        vnext_chunks = getattr(source, "_iter_vnext_chunks", None)
        public_chunks = source.iter_chunks
        canonical = getattr(type(source), "_canonical_iter_chunks", None)
        cancelled = lambda: (run.stop_requested or run.stop_signal.is_set()
                             or not accepting.is_set())
        chunks = iter(
            vnext_chunks(_CONTAINER_READ_CHUNK_FRAMES, cancelled)
            if (callable(vnext_chunks)
                and getattr(public_chunks, "__func__", None) is canonical)
            else public_chunks(_CONTAINER_READ_CHUNK_FRAMES)
        )
        producer_error: BaseException | None = None
        try:
            while accepting.is_set() and not run.stop_requested:
                read_started = monotonic()
                try:
                    block, labels = next(chunks)
                except StopIteration:
                    _perf_add(
                        run, "source_read", monotonic() - read_started
                    )
                    break
                _perf_add(run, "source_read", monotonic() - read_started)
                for offset, label in enumerate(labels):
                    if not accepting.is_set() or run.stop_requested:
                        break
                    frame = run.frames_by_label.get(int(label))
                    if frame is None:
                        continue
                    if not enqueue((frame, np.asarray(block[offset]))):
                        break
        except BaseException as error:
            producer_error = error
            accepting.clear()
            runtime = run.context_runtime
            try:
                if runtime is None:
                    output.stop()
                else:
                    runtime.request_stop(output)
            except BaseException as stop_error:
                run.cleanup_failures.append(detach_exception(
                    stop_error, "dynamic_output.stop"
                ))
        finally:
            close_chunks = getattr(chunks, "close", None)
            if callable(close_chunks):
                close_chunks()
            if run.stop_requested:
                accepting.clear()
            if accepting.is_set() and not consumer_done.is_set():
                enqueue(_SOURCE_SUBMISSION_END)
            consumer.join()
            publish_direct_fact()

        if producer_error is not None:
            raise producer_error
        if consumer_errors:
            raise consumer_errors[0]

    def _frame_ready(self, run: _StandardRun, event: Any) -> None:
        started = monotonic()
        try:
            with run.light_projection_error_lock:
                if run.light_projection_error is not None:
                    return
                try:
                    label = int(event.frame_index)
                    records = run.records
                    if records is None:
                        raise RuntimeError("display completion lost its record store")
                    record = records.get(label)
                    if record is None:
                        raise RuntimeError("display completion lost its exact record")
                    owner = run.display.artifacts.get(str(run.artifact))
                    if owner is None:
                        raise RuntimeError("display completion lost its exact owner")
                    view = record.active_view()
                    if owner.light_lease is not None:
                        run.display.publish_light_1d(
                            owner,
                            record,
                            source_identity=canonical_frame_source_identity(
                                view,
                                source_base=owner.source_base,
                                fallback_path=owner.artifact,
                            ),
                        )
                except BaseException as error:
                    run.light_projection_error = error
                    return
            pending = run.display_projection_queue
            session = run.session
            frame = run.frames_by_label.get(label)
            image = None if frame is None else frame.image
            if pending is not None and session is not None:
                try:
                    if image is not None: run.display.stamp_saturation_ceiling(owner, image)
                    if (session.saturation_mask_seeded
                            and not owner.saturation_mask_seeded):
                        run.display.stamp_saturation_mask(owner, session.saturation_mask)
                except BaseException as error:
                    run.display_projection_errors.append(error)
                    return
                while True:
                    if run.display_projection_errors:
                        return
                    try:
                        pending.put(_FrameProjectionItem(label, record, bool(
                            frame is not None and frame.mask is not None)), timeout=0.05)
                        break
                    except Full:
                        continue
            else:
                run.display_projection_errors.append(RuntimeError(
                    "display completion lost its projection queue or session"))
        finally:
            _perf_add(run, "display_callback", monotonic() - started)
            _observe_quartile_completion(run)

    def _frame_ready_owned(
        self,
        run: _StandardRun,
        item: _FrameProjectionItem,
        image: np.ndarray | None,
        session: Any,
    ) -> None:
        label, record = item.frame_index, item.record
        if type(label) is not int: raise TypeError("display projection item must be an exact int")
        scan = run.scan
        if scan is None:
            raise RuntimeError("display projection lost its scan")
        label = int(label)
        if record.label != label:
            raise RuntimeError("display projection record identity changed")
        owner = run.display.artifacts.get(str(run.artifact))
        if owner is None:
            raise RuntimeError("display projection lost its exact owner")
        view = replace(
            record.active_view(),
            raw=None,
        )
        configuration = run.configuration
        is_gi = bool(configuration is not None and configuration.gi.enabled)
        mode = 'GI' if is_gi else 'Standard'
        navigation = run.display.append_navigation(
            owner.source_scan,
            str(owner.artifact),
            label,
        )
        run.current_epoch_published += 1
        run.current_published = min(
            run.current_total,
            run.current_completed + run.current_epoch_published,
        )
        key = navigation.appended
        source_identity = canonical_frame_source_identity(
            view,
            source_base=owner.source_base,
            fallback_path=owner.artifact,
        )
        publication = FramePublication(
            view,
            record=record,
            source_identity=source_identity,
            source_base=owner.source_base,
            scan_key=owner.source_scan,
        )
        run.display.retain_frame(
            owner,
            key,
            record,
            publication,
            source_identity=source_identity,
            frame_mask_qualified=item.frame_mask_qualified,
        )
        payload = StandardDisplayPayload(
            0,
            key,
            f"{mode} · {scan.name} · frame {label}",
            view,
            "running",
            measurement_mode=mode,
            gi_incidence_motor=(
                configuration.gi.incidence_motor if is_gi else ""
            ),
            gi_resolved_motor=(
                configuration.gi.effective_motor if is_gi else ""
            ),
            gi_mode_1d=configuration.gi.mode_1d if is_gi else "",
            gi_mode_2d=configuration.gi.mode_2d if is_gi else "",
            wavelength_m=owner.wavelength_m,
        )
        self._publish_payload(
            run,
            payload,
            run.completed - run.current_completed + run.current_published,
            run.total,
            navigation=navigation,
        )

    def _checkpoint_ready(self, run: _StandardRun, event: Any) -> None:
        pending = run.display_projection_queue
        if pending is None:
            run.display_projection_errors.append(RuntimeError(
                "checkpoint recovery lost its projection queue"))
            return
        item = _CheckpointProjectionItem(str(run.artifact), tuple(event.labels))
        while not run.display_projection_errors:
            try:
                pending.put(item, timeout=0.05)
                return
            except Full:
                continue

    def _publish_payload(
        self,
        run: _StandardRun,
        payload: StandardDisplayPayload,
        completed: int,
        total: int,
        *,
        navigation=None,
    ) -> None:
        key = payload.frame_key
        if type(key) is not DisplayFrameKey:
            return
        with self._lock:
            if self._active is not run or run.closed:
                return
            run.display.put_payload(payload)
        artifact_completed = run.current_published
        self._events.put(StandardRunEvent(
            run.identity,
            StandardEventKind.FRAME_READY,
            completed=completed,
            total=total,
            artifact=key.artifact,
            detail=(
                _directory_status(
                    run,
                )
                if run.files_discovered
                else payload.status
            ),
            frame_key=key,
            navigation_delta=navigation,
            artifact_completed=artifact_completed,
            artifact_total=run.current_total,
            **_directory_event_fields(run),
        ))

    def _cleanup(self, run: _StandardRun, primary: DetachedDiagnostic | None=None) -> ExecutorClosed:
        with run.cleanup_lock:
            if primary is not None and run.primary is None:
                run.primary = primary
            if run.closed:
                return self._receipt(run)
            output = run.output
            if output is not None:
                output_finished = False
                try:
                    output.finish_all(stopped=run.stop_requested)
                except Exception as error:
                    run.cleanup_failures.append(detach_exception(
                        error, 'dynamic_output.finish'
                    ))
                else:
                    output_finished = True
            try:
                self._finish_display_projection(run)
            except Exception as error:
                run.cleanup_failures.append(detach_exception(
                    error, 'display_projection.finish'
                ))
            else:
                if output is not None:
                    try:
                        finalized = (
                            output.finalized_display_owners()
                            if isinstance(output, DynamicOutputAdapter)
                            else ()
                        )
                        self._project_new_durable(run, output)
                        for owner in finalized:
                            run.display.mark_hydration_closed(owner)
                    except Exception as error:
                        run.cleanup_failures.append(detach_exception(
                            error, 'dynamic_output.project'
                        ))
                    else:
                        if output_finished:
                            run.output = None
                            run.session = None
                            run.sink = None
            source = run.source
            if source is not None:
                try:
                    close = getattr(source, 'close', None)
                    if close is not None:
                        close()
                except Exception as error:
                    run.cleanup_failures.append(detach_exception(error, 'source.close'))
                else:
                    run.source = None
            resources = run.resources
            if resources is not None:
                for context, error in resources.cleanup():
                    run.cleanup_failures.append(detach_exception(error, context))
                if resources.cleaned:
                    run.resources = None
            non_display_clean = all(value is None for value in (
                run.session, run.sink, run.output, run.source, run.resources,
            ))
            display_clean = True
            if (
                non_display_clean
                and run.primary is not None
                and run.completed == 0
                and not run.display.payloads
                and not run.display.catalog_snapshot().entries
            ):
                # A construction/first-frame failure can leave the display
                # pointing at a lease and custody slot that lower-layer
                # settlement has already terminalized.  There is no published
                # or partial historical display in this exact zero-frame
                # shape, so retire it here on the executor worker before the
                # FAILED receipt is emitted.  Any genuinely pending owner
                # keeps ``display_clean`` false and cleanup remains fail-closed.
                try:
                    display_clean = run.display.retire(
                        join_timeout=self._join_timeout
                    )
                except Exception as error:
                    display_clean = False
                    run.cleanup_failures.append(detach_exception(
                        error, 'failed_display.retire'
                    ))
            if run.output is None:
                for owner in tuple(run.display.artifacts.values()):
                    if owner.light_lease is None:
                        try:
                            run.display.cancel_light_1d(owner, None)
                        except Exception as error:
                            run.cleanup_failures.append(detach_exception(
                                error, 'display_subscription.cancel'))
            cleaned = (
                non_display_clean
                and display_clean
                and not run.display.light_1d_cleanup_unresolved()
            )
            run.cleanup_status = CleanupStatus.CLEANED if cleaned else CleanupStatus.CLEANUP_PENDING
            if cleaned:
                run.closed = True
                run.configuration = None
                run.capture = None
                run.scan = None
                run.records = None
                run.frames_by_label.clear()
            return self._receipt(run)

    def _terminal_event(
        self,
        run: _StandardRun,
        kind: StandardEventKind,
        receipt: ExecutorClosed,
        completed: int,
        total: int,
        *,
        elapsed: float | None = None,
        work_elapsed: float = 0.0,
        cleanup_elapsed: float = 0.0,
        core_count: int = 0,
    ) -> None:
        with run.cleanup_lock:
            if run.terminal_emitted:
                return
            run.terminal_emitted = True
        detail = (
            receipt.primary.message
            if receipt.primary is not None
            else receipt.cleanup_failures[0].message
            if receipt.cleanup_failures
            else _directory_status(
                run,
                state=(
                    "Stopped"
                    if kind is StandardEventKind.STOPPED
                    else "Failed"
                    if kind is StandardEventKind.FAILED
                    else "Complete"
                ),
                in_flight_processed=_terminal_in_flight_files(run),
            )
            if run.files_discovered
            else next(
                reversed(run.display.payloads.values())
            ).measurement_mode
            if run.display.payloads
            else "Standard"
        )
        perf: dict[str, float] = {}
        if run.perf_enabled:
            with run.perf_lock:
                perf = dict(run.perf_values)
        timing_details: list[tuple[str, float]] = []
        for name, key in (
            ("source_read", "source_read"),
            ("submit_wait", "submit_wait"),
            ("writer_batch", "sink_nexus_write"),
            ("writer_flush", "sink_nexus_flush"),
            ("xye", "sink_xye_write"),
            ("finish_wait", "finish_wait"),
        ):
            if key in perf:
                timing_details.append((name, float(perf[key])))
        display_keys = ("display_callback", "display_projection")
        if any(key in perf for key in display_keys):
            timing_details.append((
                "display",
                float(sum(perf.get(key, 0.0) for key in display_keys)),
            ))
        quartile_timing = None
        capture = run.perf_quartiles
        if (
            elapsed is not None
            and run.perf_quartiles_enabled
            and capture is not None
            and run.perf_started_at is not None
        ):
            try:
                quartile_timing = capture.finish(
                    completed=completed,
                    now=run.perf_started_at + float(elapsed),
                    cumulative=perf,
                )
            except Exception:
                logger.exception(
                    "[PERF-QUARTILE] terminal snapshot was unavailable"
                )
        timing = (
            None
            if elapsed is None
            else StandardTerminalTiming(
                float(elapsed),
                float(work_elapsed),
                float(cleanup_elapsed),
                tuple(timing_details),
                quartile_timing,
            )
        )
        self._events.put(StandardRunEvent(
            run.identity,
            kind,
            completed=completed,
            total=total,
            artifact=str(run.artifact),
            detail=detail,
            cleanup_status=receipt.cleanup_status,
            primary=receipt.primary,
            cleanup_failures=receipt.cleanup_failures,
            artifacts=tuple(str(item) for item in run.artifacts),
            artifact_completed=run.current_completed,
            artifact_total=run.current_total,
            terminal_timing=timing,
            terminal_commit_identity=(
                run.terminal_commit_identity
                if kind is StandardEventKind.FINISHED
                and receipt.cleanup_status is CleanupStatus.CLEANED
                else None
            ),
            **_directory_event_fields(
                run,
                in_flight_processed=_terminal_in_flight_files(run),
            ),
        ))
        measured_elapsed = 0.0 if elapsed is None else elapsed
        throughput = (
            completed / measured_elapsed
            if completed and measured_elapsed > 0.0
            else 0.0
        )
        logger.info("Total Frames Processed: %d", completed)
        if elapsed is None:
            logger.info("Total Time: unavailable")
            logger.info(
                "[PERF-SUMMARY] outcome=%s frames=%d/%d cores=%s | "
                "total=unavailable | output=%s",
                kind.value,
                completed,
                total,
                core_count if core_count > 0 else "unknown",
                run.artifact,
            )
        else:
            logger.info("Total Time: %.2fs", measured_elapsed)
            logger.info(
                "[PERF-SUMMARY] outcome=%s frames=%d/%d cores=%s | "
                "total=%.2fs work=%.2fs cleanup=%.2fs | "
                "throughput=%.1f frames/s | output=%s",
                kind.value,
                completed,
                total,
                core_count if core_count > 0 else "unknown",
                measured_elapsed,
                work_elapsed,
                cleanup_elapsed,
                throughput,
                run.artifact,
            )
        for fact in run.resource_facts:
            logger.info("%s", fact.log_line("[PERF-RESOURCES]"))
        if run.perf_enabled:
            logger.info(
                "[PERF-DETAIL] source-read=%.2fs submit/backpressure=%.2fs "
                "finish/drain=%.2fs display-callback=%.2fs "
                "display-projection=%.2fs "
                "(parallel timers may overlap)",
                perf.get("source_read", 0.0),
                perf.get("submit_wait", 0.0),
                perf.get("finish_wait", 0.0),
                perf.get("display_callback", 0.0),
                perf.get("display_projection", 0.0),
            )
            logger.info(
                "[PERF-WRITER] nexus-write=%.2fs nexus-flush=%.2fs "
                "nexus-integrated=%.2fs named-modes=%.2fs "
                "frame-record=%.2fs h5-flush=%.2fs xye=%.2fs | "
                "record-upsert=%.2fs completion-listeners=%.2fs "
                "progress-listeners=%.2fs",
                perf.get("sink_nexus_write", 0.0),
                perf.get("sink_nexus_flush", 0.0),
                perf.get("sink_nexus_integrated", 0.0),
                perf.get("sink_nexus_named_modes", 0.0),
                perf.get("sink_nexus_frame_record", 0.0),
                perf.get("sink_nexus_h5_flush", 0.0),
                perf.get("sink_xye_write", 0.0),
                perf.get("session_record_upsert", 0.0),
                perf.get("session_frame_listeners", 0.0),
                perf.get("session_progress_listeners", 0.0),
            )
        if quartile_timing is not None:
            logger.info(
                "[PERF-QUARTILES] frames=%s",
                "/".join(str(value) for value in quartile_timing.frame_counts),
            )
            for name, values in quartile_timing.details:
                if name == "reducer_compute":
                    logger.info(
                        "[PERF-QUARTILES] reducer-compute "
                        "q1=%.3fms/frame(n=%d) q2=%.3fms/frame(n=%d) "
                        "q3=%.3fms/frame(n=%d) q4=%.3fms/frame(n=%d)",
                        *tuple(
                            item
                            for value, count in zip(
                                values,
                                quartile_timing.compute_counts,
                                strict=True,
                            )
                            for item in (
                                1000.0 * value / count if count else 0.0,
                                count,
                            )
                        ),
                    )
                    continue
                logger.info(
                    "[PERF-QUARTILES] %s q1=%.3fs q2=%.3fs "
                    "q3=%.3fs q4=%.3fs",
                    name,
                    *values,
                )

    def _exact_run(self, identity: RunIdentity) -> _StandardRun | None:
        with self._lock:
            run = self._active
        return run if run is not None and identity is run.identity else None

    def _start_cleanup_retry(self, run: _StandardRun) -> Thread | None:
        with self._lock:
            worker = run.worker
            if run.closed or (worker is not None and worker.is_alive()):
                return worker
            retry = Thread(target=self._cleanup, args=(run,), name='scattering-cleanup', daemon=True)
            run.worker = retry
            try:
                retry.start()
            except Exception as error:
                run.worker = None
                with run.cleanup_lock:
                    run.cleanup_failures.append(detach_exception(error, 'cleanup.retry'))
                return None
            return retry

    @staticmethod
    def _receipt(run: _StandardRun) -> ExecutorClosed:
        return ExecutorClosed(run.identity, run.cleanup_status, run.primary, tuple(run.cleanup_failures))
