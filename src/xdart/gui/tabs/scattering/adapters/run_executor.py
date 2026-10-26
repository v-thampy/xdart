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
from xdart.utils import log_events
from xdart.modules.frame_publication import (
    PublicationStore,
    canonical_frame_source_identity,
    publication_from_frame_view,
)
from xrd_tools.core.scan import SourceKind, SourceReadError
from xrd_tools.io.append import AppendRefused, saved_energy_selection
from xrd_tools.io.metadata import MetadataSourceChanged, any_auto_sidecar
from xrd_tools.reduction import FrameBackgroundPlan, resolve_frame_background
from xrd_tools.reduction.core import EnergyGridWidened, WarmIntegrators, energy_scan_grid_changes
from xrd_tools.reduction.background import (
    _PreparedSeriesFrameBackground,
    _PreparedSingleFrameBackground,
    _prepare_single_frame_background,
    _rebind_series_frame_background,
    _rebind_single_frame_background,
    series_capsule_requested,
)
from xrd_tools.integrate.calibration import (
    apply_sensor_parallax,
    detector_calibration_record,
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
from xrd_tools.sources.adapters import registry_identity
from xrd_tools.sources.base import source_metadata_energy_eV
from xrd_tools.sources.point_images import PointImagesRequired
from xrd_tools.sources.cursor import open_container_cursor
from xrd_tools.sources.execution_graph import (
    LivePrefixProof, published_metadata_drift, validate_consumed_metadata,
)
from xrd_tools.sources.nexus import NexusStackSource
from xrd_tools.sources.metadata_provider import RowsMetadataProvider
from xrd_tools.sources.point_images import PointImageSource
from xrd_tools.sources.probe import ProbeState
from xrd_tools.sources.selection import DirectorySourceSpec, live_metadata_format
from ..contracts import (
    AdmittedOutput, AdmissionFailure, AdmissionReceipt, AdmissionReleased,
    AdmissionToken,
    SourceCapture, SourceExecutionIdentityV1, StartCapture,
    executor_start_inputs_are_valid,
)
from ..acquisition_runtime import AcquisitionRuntime, TerminalPauseFailure
from ..display_values import (
    new_frames_note,
    DisplayFrameCatalog,
    DisplayFrameKey,
    StandardDisplayPayload,
    StandardEventKind,
    StandardRunEvent,
    StandardQuartileTiming,
    StandardTerminalTiming,
    companion_views_2d,
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
    live_directory_groups, live_group_coverage, live_readiness_debug, materialize_live_directory_group,
    observe_live_members,
    prepare_output as build_admission_receipt, source_snapshots,
    target_state_matches, validate_admitted_receipt, validate_planned_source,
)
from .target_reservation import (
    AdmissionOperation as _AdmissionOperation,
    RunResources,
)
from .dynamic_output import DynamicOutputAdapter, HeavyResidencyFact

logger = logging.getLogger(__name__)


def _run_summary(configuration) -> str:
    """A Run's banner line: processing, output mode, Live, cores, its source and where it writes."""
    try:
        spec = configuration.thaw_source_spec()
    except Exception:  # a legacy configuration has none to thaw
        spec = None
    source = (getattr(spec, "uri", None) or getattr(spec, "root", None)
              or getattr(configuration, "project_root", "") or "")
    cores = getattr(configuration, "max_cores", None)
    mode = (f"{getattr(configuration, 'processing_mode', '?')}, {getattr(configuration, 'output_mode', '?')}"
            f"{', Live' if getattr(configuration, 'live_mode', False) else ''}, "
            f"{_counted(cores, 'core') if type(cores) is int else '? cores'}")
    output = getattr(configuration, "save_path", "") or ""
    return " · ".join(filter(None, (mode, f"{source} → {output}" if source and output else source or output)))


def _counted(count: int, noun: str) -> str:
    """``1 frame``, ``2 frames``."""
    return f"{count} {noun}{'' if count == 1 else 's'}"

# Preserve the established test-injection seam while production remains strict.
load_poni = load_detector_calibration
poni_to_integrator = detector_calibration_to_integrator

_CONTAINER_READ_CHUNK_FRAMES = 8
_SOURCE_PREFETCH_FRAMES = 4
_DISPLAY_PROJECTION_FRAMES = 8
_LIVE_DIRECTORY_POLL_S = 0.1
# A Live Run never waits forever (maintainer, 2026-10-24: "I don't want a live run to stop or be waiting forever"): a
# frame waiting this long for its own metadata (its sidecar, or its SPEC row) is noted in the log and the Live status
# ("When waiting for the sidecar the log output should show it is waiting"), and after the limit for its kind it is
# processed without it; a SPEC row that never comes (an aborted point), or a Run that normalizes by a monitor from it,
# skips its scan from that frame on instead ("it is fine to skip the entire scan altogether"). The limits were "10 s
# for .txt, 30 s for SPEC"; a .txt now waits 150 s (maintainer, 2026-10-26: "exposures can be as long as 120s", his
# pick "150 s"). It is only the most a frame waits: one whose .txt holds still and reads cleanly goes on at once.
_METADATA_WAIT_NOTE_S = 2.0
_METADATA_WAIT_LIMIT_S = {"sidecar": 150.0, "spec": 30.0}
# A scan found changed is reduced again once its images and sidecars have held still this long: an acquisition can
# write an image and its sidecar a moment later (maintainer, 2026-10-24: "Go ahead with your recommendation").
_REBUILD_QUIET_S = 2.0
# Why a directory Run skips a scan whose files changed while they were read, once and then once more (not an Append
# refusal: nothing was left unchanged, and Replace would only read the same files again).
_CHANGED_WHILE_READ = "its files changed while they were read; run Append again"

_SOURCE_SUBMISSION_END = object()


def _resolve_background_before_submit(frame, plan, binding, *, cancelled,
                                      retryable=False, frame_fact=None, capsule_owner=None):
    from xrd_tools.reduction.background import resolve_background_before_submit
    return resolve_background_before_submit(frame, plan, binding, cancelled=cancelled,
        retryable=retryable, frame_fact=frame_fact, capsule_owner=capsule_owner,
        resolver=resolve_frame_background)


def _qualify_background_bindings(configuration, scan, item, decision, policy,
                                  prior, stop_signal, capsule_owner=None):
    plan = configuration.background
    if plan.mode == "None": return prior
    requirements = policy.allocation.requirements
    shape = requirements.height, requirements.width
    selector = None if item.descriptor is None else item.descriptor.dataset_path
    if not prior and configuration.output_mode == "Append":
        labels = tuple(int(frame.index) for frame in scan.frames)
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
                    cancelled=stop_signal, retryable=configuration.live_mode,
                    frame_fact=fact, capsule_owner=capsule_owner):
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
                        retryable=True, frame_fact=fact,
                        capsule_owner=run):
                    run.stop_signal.wait(_LIVE_DIRECTORY_POLL_S); continue
                binding = output.admit_background_binding((int(frame.index), fact,
                    frame.background_dependency_bytes, frame.background_dependency_fingerprint)); frame.background_dependency_bytes, frame.background_dependency_fingerprint = binding[2:]
            except BaseException:
                frame.background = None; frame.background_dependency_bytes = None
                frame.background_dependency_fingerprint = None
                raise
            return True
        if _resolve_background_before_submit(frame, plan, binding,
                cancelled=run.stop_signal, retryable=run.configuration.live_mode,
                capsule_owner=run): return True
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
    "finish_writer_seal",
    "finish_display_drain",
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
        # finish_wait covers two unrelated things: the writer's terminal seal
        # (whose real cost is the one deferred integrated-science readback) and
        # draining the display-projection worker backlog.  Split so a live run
        # says which dominates the end-of-run tail.
        "finish_writer_seal": value("finish_writer_seal"),
        "finish_display_drain": value("finish_display_drain"),
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


def _coverage_notice(run) -> str:
    """The Run-start notice when the output this Run appends to keeps a “Record angular coverage” choice that the
    pill did not ask for (COVERAGE-SUPPORT-20261013 contract B: an output keeps the choice it was created with)."""
    recorded = getattr(run.session, "recorded_angular_support", None)
    if recorded is None or run.configuration is None:
        return ""
    from xrd_tools.analysis.angular_support import DEFAULT_RECORD_ANGULAR_SUPPORT
    requested = run.configuration.run_options.get("record_angular_support", DEFAULT_RECORD_ANGULAR_SUPPORT)
    if type(requested) is not bool or (recorded == "on") is requested:
        return ""
    if recorded == "off":
        return (f"{run.artifact.name} was created with angular coverage off, and Append keeps that: this Run "
                "records none. Overwrite or a new output records it.")
    return (f"{run.artifact.name} was created recording angular coverage, and Append keeps that: this Run "
            "records it. Overwrite or a new output can turn it off.")


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
    point_plan: Any | None = None
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
    # Frames this Run reduced and saved (made durable); an Append's outputs' earlier frames are not among them.
    frames_new: int = 0
    # An Append Run, which reports frames_new; kept here because cleanup drops the configuration before the end.
    appends: bool = False
    # A Live Run's output name for each (folder, scan) so far: a scan keeps its output's name (flat outputs,
    # output_preflight._name_output_groups).
    live_output_names: dict[tuple[str, str], str] = field(default_factory=dict)
    # Each constructed output's family, as its planned output named it (StandardRunEvent.artifact_families).
    artifact_families: dict[str, str] = field(default_factory=dict)
    # Scans a directory Append skipped instead of failing the run: (output, reason).
    append_skips: list[tuple[str, str, str]] = field(default_factory=list)
    # Scans an Append reduced again whole, their processed images changed on disk: (output, reason), once each
    # finished; the current output's while it runs.
    append_rebuilds: list[tuple[str, str]] = field(default_factory=list)
    pending_rebuild: tuple[str, str] | None = None
    # The scans whose "New Scan" banner the log shows and whose "End of Scan" it has not yet: output name -> start.
    open_scans: dict[str, float] = field(default_factory=dict)
    processed_live_revisions: dict[str, LiveDirectoryAttempt] = field(
        default_factory=dict
    )
    live_revision_lock: Lock = field(default_factory=Lock)
    # Live reuse: physical target key -> the proof of its published prefix,
    # and SPEC readiness by (SPEC file, scan).  Run-owned; cleared with the loop.
    live_proofs: dict[str, Any] = field(default_factory=dict)
    live_key_snapshot: Any = None
    live_target_keys: dict[str, str] = field(default_factory=dict)
    live_spec_cache: dict[Any, Any] = field(default_factory=dict)
    # The proof of what the Live poll admitted, for the construction it calls.
    live_admission_proof: Any = None
    # A finite Append's grown scan: the energy selection its output saved for the frames it holds (_construct's prior).
    prior_energy_selection: Any = None
    # The next construction reduces a scan whole again because its new frames widened its energy grid.
    rebuild_widened: bool = False
    # Live outputs that stopped growing because a published frame's metadata
    # changed: physical target key -> the reason every later status carries.
    live_blocked: dict[str, str] = field(default_factory=dict)
    # Live outputs whose next frame has waited _METADATA_WAIT_NOTE_S for its own
    # metadata: physical target key -> the note the status carries meanwhile.
    live_waiting: dict[str, str] = field(default_factory=dict)
    # Frames a Live Run processed without their metadata: "image (output)".
    metadata_given_up: list[str] = field(default_factory=list)
    # Under Auto: whether the Run has seen a sidecar beside any image (its first listing shows one, or an admission
    # read or waited on one). A folder with none does not hold each frame for the sidecar limit.
    sidecars_seen: bool = False
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
    # Set once a recorded projection error has failed this artifact's display
    # channel; durable acknowledgements stop counting as publications then.
    display_projection_failed: bool = False
    light_projection_error: BaseException | None = None
    light_projection_error_lock: Lock = field(default_factory=Lock)
    gui_thread_id: int = field(default_factory=get_ident)
    command_failure: DetachedDiagnostic | None = None
    resource_facts: list[Any] = field(default_factory=list)
    pending_partition_count: int = 1
    terminal_commit_identity: StreamTerminal | None = None
    # The last output this Run finished writing (its path, completed and total frames, commit), and whether the Run's
    # current output is a saved one it only shows: a Run that ends showing one ends on the last frame it processed
    # (maintainer, 2026-10-26: "Last frame processed"; Codex N-068bf509).
    last_written: tuple[Path, int, int, StreamTerminal | None] | None = None
    showing_saved: bool = False
    background_capsule: _PreparedSingleFrameBackground | _PreparedSeriesFrameBackground | None = None
    energy_notice_sent: bool = False
    coverage_notice_sent: bool = False

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


def _live_proof(configuration, graph, record, registry) -> LivePrefixProof:
    """A published Live output's proof, with its members' sidecar observations from ``record``."""
    return LivePrefixProof(graph, metadata_format=live_metadata_format(configuration.thaw_source_spec()),
                           record=record, registry=registry)


def _target_stat(path: Path) -> tuple[int, int, int, int] | None:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


@dataclass(slots=True)
class _CoveredOutput:
    """A Live scan its output's committed record covers, watched by a cheap stat of that output and the adapter
    registrations at most every LivePrefixProof.PERIOD_S (Codex N-0243a1bb F3): once the output is deleted, replaced or
    rewritten while the scan's images are not, the scan is looked at again."""
    target: Path
    signature: tuple[int, int, int, int]  # the output as its record was read (unchanged across that read)
    registry: tuple
    checked_at: float

    def holds(self, registry: tuple) -> bool:
        if monotonic() - self.checked_at < LivePrefixProof.PERIOD_S:
            return True
        if registry != self.registry or _target_stat(self.target) != self.signature:
            return False
        self.checked_at = monotonic()
        return True


class _LivePollTimes:
    """XDART_PERF: where one Live poll spent its time, by step, counted and summed (Codex N-0243a1bb: measure the
    startup work before choosing another cache or reducing validation)."""

    __slots__ = ("steps",)

    def __init__(self) -> None:
        self.steps: dict[str, list] = {}

    def add(self, step: str, started: float) -> None:
        tally = self.steps.setdefault(step, [0, 0.0])
        tally[0] += 1
        tally[1] += monotonic() - started

    @property
    def busy(self) -> bool:
        return any(step not in ("listing", "key") for step in self.steps)

    def line(self) -> str:
        return ", ".join(f"{step} {count}× {seconds:.3f} s" for step, (count, seconds) in self.steps.items())


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
        f"{state} · {processed} processed{new_frames_note(_frames_new(run))} · {skipped} skipped · "
        f"{pending} pending · {discovered} discovered"
    )
    return " · ".join((status, *run.live_blocked.values(), *run.live_waiting.values()))


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
        "frames_new": _frames_new(run),
    }


def _frames_new(run: _StandardRun) -> int | None:
    """An Append Run's frames saved in this Run; None for any other Run."""
    return run.frames_new if run.appends else None


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


def _live_physical_file_key(run: _StandardRun, snapshot: object, path: Path | str) -> str:
    """Cache one Live target's physical identity for the current directory listing."""
    if run.live_key_snapshot is not snapshot:
        run.live_key_snapshot = snapshot
        run.live_target_keys.clear()
    lexical = str(path)
    key = run.live_target_keys.get(lexical)
    if key is None:
        key = _physical_file_key(lexical)
        run.live_target_keys[lexical] = key
    return key


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


def _remaining_tiff_candidates(deferred: DeferredDirectoryPlan, start_index: int) -> int:
    """Count frozen TIFF frame candidates from this output through the finite Run."""
    remaining = 0
    for index, entry in enumerate(deferred.entries[start_index:], start_index):
        if entry.completed is not None or entry.skip_reason:
            continue
        # The directory snapshot owns image files individually; tiff_series is
        # the source kind of the grouped output, not a per-file candidate owner.
        count = sum(candidate.adapter_id == "image_file" for candidate in entry.candidates)
        if index == start_index and entry.committed_prefix is not None:
            count = max(0, count - len(entry.committed_prefix.stamp.members))
        remaining += count
    return remaining


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


def _skips_failed_scans(run: _StandardRun) -> bool:
    """Whether ``run`` is a directory Run (Live or not), whose loops skip a scan that fails and go on."""
    admission = getattr(getattr(run, "resources", None), "admission", None)  # a run double may carry no resources
    return (type(getattr(admission, "deferred_directory", None)) is DeferredDirectoryPlan
            or bool(getattr(admission, "directory_discovered_file_count", 0)))


def _energy_span(selection: Any) -> str:
    """An energy selection's span, for a message: its frames' lowest to highest energy, or its one energy."""
    energies = (selection or {}).get("frame_energies_eV") or () if isinstance(selection, dict) else ()
    if energies:
        return f"{min(energies):.1f}-{max(energies):.1f} eV"
    energy = selection.get("energy_eV") if isinstance(selection, dict) else None
    return "one energy" if energy is None else f"one energy, {energy:.1f} eV"


def _read_container(read, source, what: str, *, typed: bool):
    """``read()``, one of a container's frame reads in the executor. In a directory Run (``typed``) its failure is that
    scan's SourceReadError, as a frame's own image read is (Codex N-69301b5f NW-R2): the loop skips the scan and the
    Run goes on; a single-source Run fails with the error itself, as before."""
    try:
        return read()
    except (InterruptedError, SourceReadError):
        raise
    except (OSError, ValueError) as error:
        if not typed:
            raise
        raise SourceReadError(f"{what} of {Path(str(getattr(source, 'path', '') or 'its source')).name} could not be "
                              f"read: {type(error).__name__}: {error}") from error


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
        # Built pyFAI tables kept from one Run to the next (beamline 2026-10-23: each Run's first frame rebuilt them,
        # about 8 s for a Rayonix on that PC); each Run fits them to its own memory plan.
        self._warm_tables = WarmIntegrators(1)

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
            appends=configuration.output_mode == "Append",
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
            point_images = error.detection if isinstance(error, PointImagesRequired) else None
            live_point_images = point_images is not None and operation.capture.intent_snapshot.thaw().live_mode
            result = AdmissionFailure(
                operation.token,
                ('Multiple images per point are unavailable in Live; choose a finite run.'
                 if live_point_images else detach_exception(error, 'admission').message),
                append_refused=isinstance(error, AppendRefused),
                point_images=None if live_point_images else point_images,
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
        deadline = monotonic() + max(0.0, self._join_timeout)

        def remaining() -> float:
            return max(0.0, deadline - monotonic())

        try:
            self.stop(run_identity)
        except Exception as error:
            with run.cleanup_lock:
                run.cleanup_failures.append(detach_exception(error, 'session.stop'))
        context_runtime = run.context_runtime
        if context_runtime is not None:
            context_runtime.retire()
        display_clean = run.display.retire(
            join_timeout=remaining()
        )
        worker = run.worker
        worker_was_alive = worker is not None and worker is not current_thread() and worker.is_alive()
        if worker_was_alive:
            worker.join(timeout=remaining())
        if worker is not None and worker.is_alive():
            return self._receipt(run)
        if not display_clean and worker_was_alive:
            display_clean = run.display.retire(join_timeout=remaining())
        if not run.closed:
            worker = self._start_cleanup_retry(run)
            if worker is not None and worker is not current_thread():
                worker.join(timeout=remaining())
        if not display_clean:
            run.cleanup_status = CleanupStatus.CLEANUP_PENDING
        elif run.closed and all(value is None for value in (
            run.session, run.sink, run.output, run.source, run.resources,
        )) and not getattr(run.display, 'light_1d_cleanup_unresolved', lambda: False)():
            run.cleanup_status = CleanupStatus.CLEANED
            # The worker and display are now retired. Do not keep its closed
            # source alive through AcquisitionContext's scan frame loaders.
            run.context_runtime = None
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
        run.display_projection_failed = False

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
        if run.display_projection_errors:
            run.display_projection_failed = True
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
            run.display_projection_failed = True
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

    def live_reuse_records(self, run_identity: RunIdentity) -> tuple[tuple, ...]:
        """What a Live Run keeps between polls for reuse, as records of primitive values whose
        first is the frame label served, or ``None`` for an output's or the Run's own state."""
        run = self._exact_run(run_identity)
        session = None if run is None or run.resources is None else run.resources.directory_session
        records = [] if session is None else list(session.reuse_records())
        for (path, scan), (revision, count, read) in tuple(() if run is None else run.live_spec_cache.items()):
            records.append((None, str(path), scan, *revision, count,
                            *(() if read is None else (line.decode('latin-1') for line in read.lines))))
        for target, proof in tuple(() if run is None else run.live_proofs.items()):
            stamp = proof.graph.stamp
            records.append((None, target, proof.verified_at, *(value for entry in proof.registry for value in entry),
                            *(value for key, seen in tuple(proof.spec_revisions.items()) for value in (key, *seen))))
            for label, (path, seen) in enumerate(proof.observations, stamp.first_label + proof.start):
                records.append((label, path, *(value for sidecar, revision in seen or () for value in
                                               (sidecar, *(revision or (None,))))))
            for label, source in enumerate(stamp.metadata_sources, stamp.first_label):
                if source.read_values is not None:
                    records.append((label, *source.read_values, *source.read_values.values()))
        return tuple(records)

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

    def _retire_current_target(self, run: _StandardRun, target: Path) -> None:
        """Settle ``target``'s output when it is the current one and stops growing, as a predecessor settles: its
        writer finishes and its display closes now, not when another output starts or the Run ends."""
        output = run.output
        owner = None if output is None else output.current_owner_for_target(target)
        if owner is None:
            return
        self._settle_predecessor(run, output, owner)
        run.session = None

    def _settle_failed_scan(self, run: _StandardRun, target: Path) -> None:
        """Settle ``target``'s output, the current one, after its scan's own image could not be read during reduction
        (a SourceReadError; Codex N-69301b5f NW-R2): its writer finishes as it stands, earlier epochs kept and the
        failed one's frames never published; its display closes; the caller skips the scan and the Run goes on. A
        failure of anything else there, the writer's own included, still fails the Run."""
        output = run.output
        owner = None if output is None else output.current_owner_for_target(target)
        if owner is not None:
            result, primary = output.settle_failed_current(owner)
            self._close_scan(run, owner.artifact)
            self._finish_display_projection(run)
            self._project_new_durable(run, output)
            if getattr(result, "failed", False) and not isinstance(primary, SourceReadError):
                raise RuntimeError(result.error or "scan settlement failed")
            run.display.mark_hydration_closed(owner)
            if not run.display.release_light_1d(owner, reason="lineage-replaced"):
                raise RuntimeError("prior light-1D lineage cleanup remains pending")
            output.drop_released_predecessor(owner, aborted=True)
        # Settlement can make submitted container frames durable. Count those before dropping only the unsaved work.
        run.total -= max(0, run.current_total - run.current_completed)
        run.session = None
        self._abandon_construction(run)

    def _block_live_target(
        self, run: _StandardRun, target: Path, target_key: str,
        drift: tuple[str, ...],
    ) -> None:
        """Stop extending one Live output whose published frames' metadata changed.

        Its published frames, their metadata and lineage stay as written; its
        open writer is finished through the output owner, and a failure there
        still fails the Run.  The reason stays in every later status.
        """

        self._retire_current_target(run, target)
        names = ", ".join(sorted({Path(path).name for path in drift}))
        reason = (
            f"{Path(target).name} stopped growing: {names} changed after "
            "its frame was reduced; published frames kept"
        )
        run.live_blocked[target_key] = reason
        run.live_waiting.pop(target_key, None)
        log_events.event(
            logger, logging.WARNING, "%s stops growing: the metadata of frames it already processed changed (%s); its "
            "frames so far are kept and its later frames are skipped in this Run [LIVE]", Path(target).name, names)

    @staticmethod
    def _abandon_construction(run: _StandardRun) -> None:
        """Close the source a scan's construction opened and forget its frames, after that construction refused."""
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
        run.pending_rebuild = None  # a refused construction reduced nothing again

    def _skip_directory_scan(self, run: _StandardRun, target: Path, error: BaseException | str) -> None:
        """One scan of a directory Run that could not be processed: skipped, named, and listed at the Run's end; the
        other scans still run (maintainer, 2026-10-24: "if something fails, a live run or a directory run shouldn't
        stop. That scan/frame should be logged and skipped and the remaining scan processing should continue").
        ``error`` may already be its reason."""
        reason = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
        run.append_skips.append((str(target), "scan", reason))
        log_events.event(logger, logging.WARNING, "%s skipped: it could not be processed; frames already saved are "
                         "kept, and the other scans continue. Details: %s [SCAN-SKIPPED]", Path(target).name, reason)

    @staticmethod
    def _widened_rebuild(run: _StandardRun, receipt: AdmissionReceipt, session: Any, entry: DeferredDirectoryEntry,
                         reason: str) -> AdmittedOutput:
        """The admission that reduces ``entry``'s scan again whole into its output (a REPLACE, every frame qualified
        afresh, none taken from its record): its new frames widened its energy scan's q grid."""
        decision, _ready, _skipped = materialize_deferred_output(
            receipt, session, entry, cancelled=lambda: run.stop_requested)
        if decision is None:
            raise ValueError("its scan has no READY frames left to reduce again")
        stamp = decision.item.source_stamp
        labels = tuple(range(stamp.first_label, stamp.first_label + stamp.frame_count))
        return AdmittedOutput(decision.item, OutputDisposition.REPLACE, labels, decision.fact, reason)

    def _skip_failed_directory_scan(self, run: _StandardRun, target: Path, error: BaseException | str, *,
                                    constructed: bool) -> None:
        """A directory Run's scan whose construction refused (``constructed`` False) or whose image could not be read
        while it was reduced (a SourceReadError): its output settled as it stands, its unfinished frames out of the
        totals, its files skipped, the scan named; the Run goes on."""
        claimed = run.current_file_total
        if constructed:
            with _run_effect(run):
                self._settle_failed_scan(run, target)
        else:
            planned = run.current_total
            with _run_effect(run):
                self._retire_current_target(run, target)
            self._abandon_construction(run)
            run.total -= planned
        run.files_skipped += claimed
        self._skip_directory_scan(run, target, error)
        self._events.put(StandardRunEvent(
            run.identity,
            StandardEventKind.DISCOVERY,
            completed=run.completed,
            total=run.total,
            detail=_directory_status(run),
            **_directory_event_fields(run),
        ))

    def _skip_live_append(
        self, run: _StandardRun, target: Path, target_key: str, error: AppendRefused | None, *, constructed: bool,
        reason: str | None = None, changed: bool = False, failed: bool = False, widened: bool = False,
    ) -> None:
        """Leave one Live output as it is after its Append refused: later polls skip its frames, the status names
        it, and the Run's end lists the reason. The output, when it is the current one, is retired first (with the
        Run's progress as it stands), and a failure there still fails the Run; a refusal during construction then
        closes the source it opened. ``reason`` stands for ``error``'s when there is none (a rebuild typed against an
        output this Run grows, ``changed``: one of its processed images changed on disk)."""
        with _run_effect(run):
            self._retire_current_target(run, target)
        if constructed:
            self._abandon_construction(run)
        reason = detach_exception(error, "append").message if reason is None else reason
        skip_kind = "scan" if failed or widened else "append"
        run.append_skips.append((str(target), skip_kind, reason))
        run.live_blocked[target_key] = (
            f"{Path(target).name} stopped growing: "
            f"{'its energy span widened' if widened else 'it could not be processed' if failed else 'its Append was refused'}"
            "; published frames kept"
        )
        run.live_waiting.pop(target_key, None)
        log_events.event(
            logger, logging.WARNING, "%s stops growing: %s; its frames so far are kept and its later frames are "
            "skipped in this Run%s. Details: %s [%s]", Path(target).name,
            "its new frames' energies widen the q grid it was reduced on" if widened
            else "an image it already processed changed on disk" if changed else "it could not be processed" if failed
            else "it can no longer be extended",
            "; a directory Run reprocesses the whole scan" if widened
            else "; the next Run reprocesses the whole scan" if changed else "", reason,
            "SCAN-SKIPPED" if skip_kind == "scan" else "APPEND-SKIPPED")

    def _watch_metadata_wait(self, run: _StandardRun, target: Path | None, target_key: str,
                             waiting: tuple[str, str] | None,
                             waits: dict[str, tuple[str, float, str, str, float | None]],
                             given_up: dict[str, set[str]], *, retried: bool = False,
                             admitted: frozenset[str] | None = None) -> bool:
        """Time the frame an output waits on for its own metadata (``waiting``: (image, kind), as
        ``LiveDirectoryAttempt.waiting_metadata``; None: it no longer waits, its scan is covered, or its files left the
        directory). After _METADATA_WAIT_NOTE_S the log and the Live status say so; after the limit for its kind a frame
        whose sidecar never came is given up on, processed without its metadata; a sidecar still changing, a SPEC row,
        or a Run normalizing by a monitor from it skips the scan from that frame on, as does a sidecar that comes for a
        frame given up on (Codex N-69301b5f NW-R1: never read unsettled; True: the caller moves on). ``retried``: the
        attempt did not admit the frame, nor name it (its guarded read met the file changing, a listing was due, a
        candidate went stale), or a READY attempt's ``admitted`` images leave it out (a fresh listing cut before it):
        the same frame still waits, with its one deadline (Codex N-0c4e4237). A sidecar settles within the limit
        from its first sighting, not from its image's: one first seen near the limit is looked at again, not skipped
        at the look that found it (beamline 2026-10-25, four scans skipped "did not settle within 10 s" a moment after
        their sidecars appeared); the first sighting is kept, so the whole wait stays bounded."""
        if waiting is not None and waiting[1] == "late":
            run.live_waiting.pop(target_key, None)
            waits.pop(target_key, None)
            self._skip_live_append(run, target, target_key, None, constructed=False, failed=True, reason=(
                f"the metadata of {waiting[0]} arrived after the Run stopped waiting for it"))
            return True
        now, held = monotonic(), waits.get(target_key)
        if waiting is None and held is not None and (retried or admitted is not None and held[0] not in admitted):
            waiting = (held[0], held[3])  # not admitted: it still waits, as it was last seen
        if waiting is None or held is None or held[0] != waiting[0]:
            if run.live_waiting.pop(target_key, None) is not None and held is not None:
                log_events.event(logger, logging.INFO, "%s no longer waits for the metadata of %s [LIVE]",
                                 held[2], held[0])
            if waiting is None:
                waits.pop(target_key, None)
            else:
                waits[target_key] = (waiting[0], now, Path(target).name, waiting[1],
                                     now if waiting[1] == "settling" else None)
            return False
        sighted = held[4] if held[4] is not None else (now if waiting[1] == "settling" else None)
        waits[target_key] = (*held[:3], waiting[1], sighted)  # its latest kind, and its sidecar's first sighting
        (image, kind), name = waiting, held[2]
        limit, quiet = _METADATA_WAIT_LIMIT_S["sidecar" if kind == "settling" else kind], False
        if kind == "sidecar" and not run.sidecars_seen and live_metadata_format(
                run.configuration.thaw_source_spec()) == "auto":
            # Auto, and no sidecar beside any image so far (an empty folder at the Run's start, or one whose images
            # have none). The first frame waits the full limit, as any other does: a .txt can land seconds after its
            # image (beamline 2026-10-26: 5.1 s on the share), and a frame given up on is processed without it, as are
            # all later ones (maintainer, 2026-10-26: "shouldn't this be 150s like the other case then to be safe?").
            # Once the Run has given up on one, the folder is known to have none and later frames go on at once,
            # without a warning each.
            quiet = bool(run.metadata_given_up)
            limit = 0.0 if quiet else limit
        if now - (sighted if kind == "settling" else held[1]) < limit:
            if target_key not in run.live_waiting and now - held[1] >= _METADATA_WAIT_NOTE_S:
                run.live_waiting[target_key] = f"{name} waits for the metadata of {image}"
                log_events.event(logger, logging.INFO, "%s is waiting for the metadata of %s [LIVE]", name, image)
            return False
        run.live_waiting.pop(target_key, None)
        waits.pop(target_key, None)
        if kind == "spec":  # its row never came: a SPEC scan that stopped there (an aborted point)
            self._skip_live_append(run, target, target_key, None, constructed=False, failed=True, reason=(
                f"the SPEC row of {image} did not arrive within {limit:g} s"))
            return True
        if kind == "settling":  # present but never holding still: the rare case, not read unsettled
            self._skip_live_append(run, target, target_key, None, constructed=False, failed=True, reason=(
                f"the metadata of {image} did not settle within {limit:g} s"))
            return True
        if any(str((args or {}).get("monitor") or "").strip()
               for args in (run.configuration.bai_1d_args, run.configuration.bai_2d_args)):
            self._skip_live_append(run, target, target_key, None, constructed=False, failed=True, reason=(
                f"the metadata of {image} did not arrive within {limit:g} s, and this Run normalizes by a monitor "
                "from it"))
            return True
        given_up.setdefault(target_key, set()).add(image)
        if not quiet:
            run.metadata_given_up.append(f"{image} ({name})")
            log_events.event(logger, logging.WARNING, "%s: the metadata of %s did not arrive within %g s; the frame "
                             "is processed without it [LIVE]", name, image, limit)
        return False

    def _open_scan(self, run: _StandardRun, artifact: Path, note: str = "") -> None:
        """The log's ``New Scan`` banner for ``artifact``'s output (maintainer, 2026-10-24)."""
        name = Path(artifact).name
        run.open_scans[name] = monotonic()
        log_events.banner(logger, "New Scan · %s%s", name, f" ({note})" if note else "", kind="scan")

    def _close_scan(self, run: _StandardRun, artifact: Path, frames: int | None = None) -> None:
        """The log's ``End of Scan`` banner for ``artifact``'s output, with its frames and its time since it opened."""
        name = Path(artifact).name
        started = run.open_scans.pop(name, None)
        if started is None:
            return
        counted = "" if frames is None else f" · {_counted(int(frames), 'frame')}"
        log_events.banner(logger, "End of Scan · %s%s · %.1f s", name, counted, monotonic() - started,
                          kind="scan-end")

    def _settle_predecessor(self, run, output, predecessor) -> None:
        result = output.finish_predecessor(predecessor)
        self._close_scan(run, predecessor.artifact, getattr(result, "n_processed", None))
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

    def _construct(self, run: _StandardRun, *, item: PlannedOutput | None=None, labels: tuple[int, ...] | None=None, decision: AdmittedOutput | None=None, remaining_frames: int | None=None) -> _StandardRun:
        proof = getattr(run, 'live_admission_proof', None)
        if proof is not None:
            run.live_admission_proof = None
            proof = proof if item is not None and proof.graph is item.graph else None
        prior_energy, run.prior_energy_selection = getattr(run, 'prior_energy_selection', None), None
        widened_rebuild, run.rebuild_widened = getattr(run, 'rebuild_widened', False), False
        # A finite Append's scan that grew, with the energy selection its output saved: the frames it holds are not
        # read here either (only that selection used their metadata); the binding extends it, or they are read.
        grown = (prior_energy if proof is not None and item.graph.infers_point_images
                 and type(prior_energy) is dict else None)
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
        run.artifact, run.showing_saved = artifact, False
        if family := getattr(item, "artifact_family", ""):
            run.artifact_families[str(artifact)] = family
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
            # A Live item re-checks the proof of what its poll admitted; only a
            # failed proof validates the whole graph again.
            if item is not None and (proof is None or not proof.current(cancelled=cancelled)):
                validate_planned_source(item, cancelled=cancelled)
            if item is not None and item.candidate is not None and (item.descriptor is not None) and (item.descriptor.kind in {SourceKind.NEXUS_STACK, SourceKind.EIGER_MASTER}):
                cursor = open_container_cursor(item.source_path, entry=item.source_spec.entry or 'entry', candidate=item.candidate)
                try:
                    run.source = NexusStackSource(item.source_path, entry=item.source_spec.entry or 'entry', cursor=cursor)
                    if (rows := item.graph.stamp.metadata_sources):  # a Live SPEC container: each frame's admitted row (R5b)
                        run.source.bind_metadata_provider(RowsMetadataProvider(
                            (value.spec_values for value in rows), rows[0].spec_columns or ()))
                    if source_spec.options.get('point_images') is not None:
                        run.source = PointImageSource(run.source, source_spec)
                except Exception:
                    cursor.close()
                    raise
            else:
                run.source = open_source(source_spec)
            if item is not None and hasattr(run.source, 'retain_metadata'):
                # The proved prefix's frames keep the values their admission read, and a live frame admitted without
                # a metadata file (one the Run gave up waiting for) is reduced without one, whatever lands after
                # (Codex N-69301b5f NW-R1: the saved row, the reduction and the admitted record agree).
                stamp = item.graph.stamp
                retained = {label: values for label, source in enumerate(
                    stamp.metadata_sources[:0 if proof is None else proof.start], stamp.first_label)
                    if (values := source.read_values if source.spec_values is None else source.spec_values) is not None}
                retained.update((label, {}) for label, source in enumerate(stamp.metadata_sources, stamp.first_label)
                                if source.metadata_file is None and source.read_values is not None)
                if grown is not None:
                    retained.update((label, {}) for label in range(stamp.first_label, stamp.first_label + proof.start))
                if retained:
                    run.source.retain_metadata(retained)
            admission = None if run.resources is None else run.resources.admission
            assets = None if admission is None else admission.scientific_assets
            calibration = (
                assets.detector_calibration
                if assets is not None else load_poni(configuration.poni_file)
            )
            if calibration is None:
                raise ValueError('accepted PONI asset is unavailable')
            if assets is None and configuration.poni_v3_override is not None:
                override = configuration.poni_v3_override
                calibration = apply_sensor_parallax(
                    calibration,
                    material=override.material,
                    thickness_m=override.thickness_m,
                    parallax=override.parallax,
                )
            from xrd_tools.session.readiness import bind_native_int_source_energy, effective_ring_geometry
            run.scan = run.source.to_scan(output_path=artifact, cancelled=cancelled)
            if item is not None:
                # A live Run reduces only the SPEC values its admission read.
                validate_consumed_metadata(
                    item.graph,
                    {int(frame.index): frame.metadata for frame in run.scan.frames},
                    start=0 if proof is None else proof.start,
                )
            from xrd_tools.session.readiness import PriorEnergyUnsettled

            def bind(prior):
                return bind_native_int_source_energy(run.scan, calibration,
                    metadata_wavelength_A=(None if item is None or item.descriptor is None
                                           else item.descriptor.wavelength),
                    metadata_energy_eV=source_metadata_energy_eV(run.source), prior=prior)
            try:
                calibration = bind(None if grown is None else (dict(grown), proof.start))
            except PriorEnergyUnsettled:  # its energy needs the frames the output holds: read them, as before
                run.source.retain_metadata({})
                run.scan = run.source.to_scan(output_path=artifact, cancelled=cancelled)
                calibration = bind(None)
            integrator = run.scan.integrator
            poni = run.scan.poni
            if calibration.parallax is not None:
                extra = dict(getattr(run.scan, "extra", None) or {})
                extra["detector_calibration"] = detector_calibration_record(
                    calibration, integrator=integrator,
                )
                run.scan.extra = extra
            mask = None if assets is None else assets.mask
            flat = None if assets is None else assets.flat
            dark = None if assets is None else assets.dark
            if mask is not None or flat is not None or dark is not None:
                # Persist the applied mask, flat and dark: Reintegrate reuses only
                # the output's own copies, never the source files.
                extra = dict(getattr(run.scan, "extra", None) or {})
                if mask is not None:
                    extra["global_mask"] = np.flatnonzero(np.asarray(mask, dtype=bool))
                if flat is not None:
                    extra["flatfield"] = flat
                if dark is not None:
                    extra["darkcurrent"] = dark
                run.scan.extra = extra
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
            if (
                isinstance(error, MetadataSourceChanged)
                and item is not None
                and not item.graph.infers_point_images
            ):
                # A live Run's reduction read met a metadata file changing under
                # it (a SPEC file still being appended): read again next poll.
                discard_source()
                raise SourceRevisionChanged(str(error)) from error
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
            len(run.scan.frames)
        )
        for key, name in (("flat", "flat field"), ("dark", "dark")):
            if getattr(configuration, f"{key}_file", "") and (
                    assets is None or getattr(assets, f"{key}_bytes") is None):
                raise ValueError(f"the configured {name} was not admitted")
        plan = replace(
            native_int_reduction_plan(configuration, companion_modes_2d=True,
                energy_selection=run.scan.extra['energy_selection']),
            mask=None if assets is None else assets.mask,
            flat=None if assets is None else assets.flat,
            dark=None if assets is None else assets.dark,
        )
        ring_geometry = effective_ring_geometry(calibration, integrator=integrator, plan=plan)
        run.point_plan = plan if isinstance(run.source, PointImageSource) else None
        try:
            npt = int(configuration.bai_1d_args.get("npt", 0))
        except (TypeError, ValueError):
            npt = 0
        if item is None or decision is None:
            raise RuntimeError("dynamic output requires one admitted output")
        output = run.output
        keeps_session = output is not None and output.owns_target(item.target) and output.session is not None
        if (configuration.output_mode == "Append" and decision.disposition is OutputDisposition.WRITE
                and not keeps_session and len(decision.labels) < item.source_stamp.frame_count
                and (plan.extra.get("energy_selection") or {}).get("frame_energies_eV")):
            # An energy scan in q has one common grid, from its lowest to its highest energy; frames that widen the
            # one an earlier Run's rows were reduced on cannot join its stack (maintainer, 2026-10-25: "Rebuild it
            # whole"). An output with an existing session is not checked: it keeps its first grid and energies,
            # dropping the points beyond, and a frame they do not list takes the first point's (maintainer,
            # 2026-10-25: "it is fine to drop the points that don't lie on the initial grid rather than stopping";
            # "It's fine to stick with the energy of the first point if the scan parameters are not known").
            baseline = prior_energy if type(prior_energy) is dict else saved_energy_selection(item.target)
            if energy_scan_grid_changes(plan, baseline, integrator, poni):
                discard_source()
                raise EnergyGridWidened(f"its frames' energies now span {_energy_span(plan.extra.get('energy_selection'))}"
                                        f"; its saved rows were reduced over {_energy_span(baseline)}")
        if output is None:
            output = DynamicOutputAdapter(configuration, self._warm_tables); run.output = output
        preparation = output.prepare_admission(
            run.scan, plan, item, decision, run.stop_signal,
            qualify=lambda policy, prior: _qualify_background_bindings(
                configuration, run.scan, item, decision, policy, prior,
                run.stop_signal, run))
        replaces = decision.disposition is OutputDisposition.REPLACE
        if (configuration.output_mode == "Overwrite" or replaces) and not target_state_matches(preparation.effective):
            raise ValueError(f"output target changed after Background qualification: {item.target}")
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
            self._configure_display(
                run,
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
        resumed = owner is not None and owner.hydration_closed
        if owner is not None and replaces:
            # Replaced whole (a Live Run showed it from its saved record, or finished it before another scan began,
            # then its images changed): its display was closed then, and its old frames leave navigation with the
            # replacement's first frame.
            if not owner.hydration_closed:
                raise RuntimeError(f"a replaced output's display is still open: {artifact}")
            run.display.retire_artifact_navigation(str(artifact))
        if owner is not None and owner.hydration_closed:
            # An output this Run closed (shown from its saved record, or settled to admit another) grows again: its
            # writer starts a new display generation (Codex N-0243a1bb F2: the closed store could not take its allocation).
            run.display.begin_active_generation(owner, cancelled=cancelled)
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
                saturation=getattr(plan, "saturation", None),
                threshold_min=plan.threshold_min,
                threshold_max=plan.threshold_max,
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
        if newly_adopted_owner or resumed:
            self._open_scan(run, artifact, "reprocessed whole" if replaces else "continued" if resumed else "")
        run.records = owner.records
        run.display.bind_checkpoint_hydration(owner)
        run_provenance = configuration.as_provenance()
        if admission is not None:
            run_provenance['scientific_signature'] = admission.candidate.processing_mapping()
        run.pending_rebuild = (str(item.target), decision.reason) if replaces else None
        if replaces:  # the Run asked for Append; this output is replaced whole (Codex N-a8a6c887)
            run_provenance["append_replacement"] = {
                "requested_output_mode": "Append", "effective_output_mode": "Overwrite", "reason": decision.reason}
            log_events.event(
                logger, logging.INFO, "%s: %s, so the whole scan is reprocessed and its output replaced. Details: %s "
                "[APPEND-REBUILD]", Path(item.target).name,
                "its new frames' energies widen the q grid its frames were reduced on" if widened_rebuild
                else "images it already processed changed on disk", decision.reason)
        if run.stop_requested:
            discard_source()
            raise RuntimeError("admission cancelled")
        run.session, created = output.activate(
            preparation,
            ring_geometry=ring_geometry,
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
            remaining_frames=remaining_frames,
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
    def _configure_display(run: _StandardRun, *, npt: int, frame_bytes: int | None) -> None:
        """The Run's one display configuration, at its first output: a constructed one, after its Background
        qualification, or an already-processed scan's presentation, which has no Background."""
        run.display.configure(
            partition_count=run.pending_partition_count, npt=npt, frame_bytes=frame_bytes,
        )

    @staticmethod
    def _seed_persisted_prefix_navigation(
        run: _StandardRun,
        owner: DisplayArtifact,
        labels: tuple[int, ...],
        *,
        newly_adopted: bool,
        completed: int | None = None,
    ) -> None:
        """``completed``: the output's completed frames, the Run's current output's when None (a saved output shown
        while another is open passes its own)."""
        if type(newly_adopted) is not bool:
            raise TypeError("persisted-prefix adoption flag must be exact")
        if not newly_adopted:
            return
        if type(labels) is not tuple:
            raise TypeError("persisted prefix labels must be an exact tuple")
        if len(labels) != (run.current_completed if completed is None else completed):
            raise ValueError("persisted prefix cardinality changed")
        if any(type(label) is not int for label in labels):
            raise TypeError("persisted prefix labels must be exact integers")
        capacity = run.display.navigation_capacity
        retained = labels[-capacity:]
        # Live progress counts only new frames, so run.completed omits earlier
        # outputs' persisted prefixes that the catalog already holds; a second
        # Append output then reused a spent ordinal (Windows report, 2026-10-22).
        # New output owners start after every ordinal already issued. Retain
        # the caller's absolute base when it supplies an already represented
        # prefix; ordinary same-owner growth returned above without reseeding.
        issued = run.completed
        if not run.display.navigation_represents(str(owner.artifact), retained):
            issued = max(issued, run.display.navigation_work_high_water)
        absolute_base = issued + len(labels) - len(retained)
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
        self, run: _StandardRun, owner: DisplayArtifact, *, source_path: str,
        saved_only: bool = False,
    ):
        created = run.context_runtime is None
        runtime = run.context_runtime or AcquisitionRuntime()
        run.context_runtime = runtime
        previous = (
            None
            if runtime.context is None
            else runtime.context.hydration_owner
        )
        context = runtime.adopt(run, owner, source_path, saved_only=saved_only)
        # The native binder already resolved these values for this output.
        # Announce only after its acquisition context has actually been adopted.
        selection = {} if saved_only else getattr(run.scan, "extra", {}).get("energy_selection", {})
        energy = ""
        if selection.get("disagreement") and not run.energy_notice_sent:
            energy = (f"Using the scan energy, {selection['source_energy_eV'] / 1000:.3f} keV; "
                      f"the PONI says {selection['calibration_energy_eV'] / 1000:.3f} keV.")
            if isinstance(run.capture.source, DirectorySourceSpec):
                energy = f"{run.artifact.name}: {energy}"
        coverage = "" if saved_only or run.coverage_notice_sent else _coverage_notice(run)
        detail = " ".join(part for part in (energy, coverage) if part)
        if created or context.hydration_owner != previous or detail:
            self._events.put(StandardRunEvent(
                run.identity, StandardEventKind.CONTEXT_READY,
                artifact=str(owner.artifact), detail=detail,
            ))
            run.energy_notice_sent |= bool(energy)
            run.coverage_notice_sent |= bool(coverage)
        return context

    def _run(self, run: _StandardRun) -> None:
        started_at = monotonic()
        log_events.banner(logger, "New Run · %s", _run_summary(run.configuration), kind="run")
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
        for target, kind, reason in getattr(receipt, "skipped_outputs", ()):  # scans its admission skipped (GI)
            if kind == "scan":
                self._skip_directory_scan(run, Path(target), reason)
                continue
            run.append_skips.append((target, "append", reason))
            log_events.event(
                logger, logging.WARNING, "%s skipped: it cannot be extended, so its existing results are left "
                "unchanged (Replace reprocesses it). Details: %s [APPEND-SKIPPED]", Path(target).name, reason)
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
            try:
                self._construct(run, item=item, labels=decision.labels, decision=decision)
            except ValueError as error:  # a directory Run's scan (GI admits them all first): skipped, the rest run
                if not run.files_discovered or isinstance(error, SourceRevisionChanged):  # its admitted files changed
                    raise
                self._skip_failed_directory_scan(run, item.target, error, constructed=False)
                continue
            run.completed += run.current_completed
            try:
                stopped = self._execute_current(run, construct=False) or stopped
            except SourceReadError as error:
                if not run.files_discovered:
                    raise
                self._skip_failed_directory_scan(run, item.target, error, constructed=True)
                continue
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
        # Outputs this Run has already settled once (admitted, presented or published): only these take the
        # already-processed census below; a first sight is admitted as before.
        seen_targets: set[str] = set()
        # Outputs that did not cover their scan, by its images and the output's stat: no second read while neither
        # changes (Claude cloud review of 4c28033b, finding 5: a scan still waiting opened its output every poll).
        uncovered: dict[str, tuple[tuple[object, ...], tuple[int, int, int, int]]] = {}
        # A found output's committed image count by output file version, for its readiness waits.
        committed_counts: dict[str, tuple[tuple[int, int, int, int], int | None]] = {}
        # The image each output's next frame waits on for its metadata, since when, the output's name, the wait's
        # latest kind and when its sidecar was first seen; and the images each output stopped waiting for.
        metadata_waits: dict[str, tuple[str, float, str, str, float | None]] = {}
        given_up: dict[str, set[str]] = {}
        # A scan found changed: its revision and when its rebuild may start (_REBUILD_QUIET_S).
        rebuild_holds: dict[str, tuple[tuple[object, ...], float]] = {}
        # SPEC readiness by SPEC file revision: no read while it is unchanged.
        spec_cache = run.live_spec_cache
        spec_cache.clear()
        run.live_proofs.clear()
        # Each image's sidecar as last seen; admission waits until it holds still.
        sidecar_observations: dict[str, tuple[tuple[str, object], ...]] = {}
        # Under Auto the folder's first listing says whether sidecars are expected at all (run.sidecars_seen).
        meta_auto = live_metadata_format(configuration.thaw_source_spec()) == "auto"
        discovered_paths: set[Path] = set(deferred.discovered_paths)
        processed_paths: set[Path] = set()
        skipped_paths: set[Path] = set()
        last_projection: tuple[object, ...] | None = None

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
                # A newly stopped or waiting output is news even when no count moved.
                tuple(run.live_blocked.values()),
                tuple(run.live_waiting.values()),
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
        # The last poll's groups, reused while the session returns the same
        # Snapshot object under the same adapter registrations.
        grouped: tuple[object, tuple[object, ...], tuple[object, ...]] | None = None
        polls, loop_started, first_written = 0, monotonic(), False
        while not run.stop_requested:
            times, poll_started = _LivePollTimes(), monotonic()
            with _run_effect(run):
                observation = session.observe(refresh=True)
                registry = registry_identity()
                if grouped is None or grouped[0] is not observation.discovered_snapshot or grouped[1] != registry:
                    grouped = (observation.discovered_snapshot, registry,
                               live_directory_groups(receipt, observation, run.live_output_names))
                groups = grouped[2]
                if polls == 0 and meta_auto:
                    # What the folder holds at the Run's start: a sidecar beside any image means each frame waits for
                    # its own as it lands; none means the first frame waits for its own, and once the Run has given
                    # up on it the folder is known to have none.
                    run.sidecars_seen = any_auto_sidecar(path for group in groups for path in group.physical_paths)
                for group in groups:
                    discovered_paths.update(group.physical_paths)
                publish()
            times.add("listing", poll_started)

            watched: list[Path] = []  # files the next poll must see change at once (D1)
            polled: set[str] = set()  # this poll's outputs: a waiting note outlives none of them
            for group in groups:
                if run.stop_requested:
                    return True
                revision = tuple(group.revision)
                started = monotonic()
                target_key = _live_physical_file_key(
                    run, observation.discovered_snapshot, group.target,
                )
                polled.add(target_key)
                times.add("key", started)
                if target_key in run.live_blocked:
                    # This output stopped growing; later frames are skipped.
                    skipped_paths.update(
                        path for path in group.physical_paths
                        if path not in processed_paths
                    )
                    publish()
                    continue
                settled = settled_revisions.get(target_key)
                if settled is not None:
                    seen_targets.add(target_key)
                    if settled[0] == revision and type(settled[1]) is _CoveredOutput:
                        if settled[1].holds(registry):
                            continue
                        # Its output changed (deleted, replaced or rewritten) while its images did not: look again.
                        settled_revisions.pop(target_key, None)
                        settled = None
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
                    proof = run.live_proofs.get(target_key)
                    with _run_effect(run):
                        # The drift sweep: each settled output's proof is
                        # re-checked within the sweep period (a container,
                        # which grows in place, every poll); only a change
                        # reaches the strict validator below.
                        if proof is not None and proof.graph is retained.decision.item.graph and (
                                proof.current if retained.decision.item.descriptor is None
                                else proof.holds)(cancelled=lambda: run.stop_requested):
                            continue
                        run.live_proofs.pop(target_key, None)
                        # Discovery and registrations as they are before the
                        # strict check: a proof rebuilt on its pass holds both.
                        registry = registry_identity()
                        seen = observe_live_members(configuration, [
                            member.path for member in retained.decision.item.source_stamp.members])
                        try:
                            validate_planned_source(
                                retained.decision.item,
                                cancelled=lambda: run.stop_requested,
                            )
                        except SourceRevisionChanged:
                            settled_revisions.pop(target_key, None)
                        else:
                            run.live_proofs[target_key] = _live_proof(
                                configuration, retained.decision.item.graph, seen, registry)
                            continue
                held = rebuild_holds.get(target_key)
                if held is not None and held[0] == revision and monotonic() < held[1]:
                    continue  # a scan found changed waits for its images and sidecars to hold still
                owned = run.output is not None and run.output.owns_target(group.target)
                started = monotonic()
                completed = signature = None
                if not owned:
                    # Coverage counts only for the output its record was read from: the same output before and
                    # after the read (Codex N-770941b2: one deleted or replaced during the read stayed covered).
                    signature = _target_stat(group.target)
                    if signature is not None and uncovered.get(target_key) != (revision, signature):
                        completed = live_group_coverage(receipt, group)
                        if completed is not None and _target_stat(group.target) != signature:
                            completed = None
                        if completed is None:
                            uncovered[target_key] = (revision, signature)
                times.add("coverage", started)
                if completed is not None:
                    # A scan its output already holds, image for image: shown once from its committed record, as a
                    # directory Append shows it, then a quiet no-op until its images change; nothing is probed, read,
                    # released or refused (beamline 2026-10-23: each finished scan was probed and its sidecars read
                    # over two listings, about 40 s on the share before the first frame; files the beamline touched
                    # after a scan were refused one by one, and settling the scan still growing to look at them
                    # failed it at its next frame, "cannot bind an allocation to a populated store").
                    if target_key not in seen_targets and target_key not in run.processed_live_revisions:
                        started = monotonic()
                        with _run_effect(run):
                            self._present_live_saved_output(run, group, completed)
                        times.add("shown", started)
                        seen_targets.add(target_key)
                    settled_revisions[target_key] = (
                        revision, _CoveredOutput(group.target, signature, registry, monotonic()))
                    self._watch_metadata_wait(run, group.target, target_key, None, metadata_waits, given_up)
                    processed_paths.update(group.physical_paths)
                    publish(force=True)
                    continue
                constructing = False
                try:
                    with _run_effect(run):
                        reprobe = (
                            retry_revisions.pop(target_key, None) == revision
                        )
                        with run.live_revision_lock:
                            retained = run.processed_live_revisions.get(
                                target_key
                            )
                        published = (
                            0 if retained is None or retained.decision is None
                            else len(retained.decision.item.source_stamp.members)
                        )
                        watched.extend(group.physical_paths[published:])
                        # The registrations this attempt is qualified under.
                        registry = registry_identity()
                        started = monotonic()
                        attempt = materialize_live_directory_group(
                            receipt,
                            configuration,
                            session,
                            group,
                            cancelled=lambda: run.stop_requested,
                            reprobe=reprobe,
                            sidecar_observations=sidecar_observations,
                            published_members=published,
                            published_graph=(
                                None if retained is None or retained.decision is None
                                else retained.decision.item.graph
                            ),
                            proof=run.live_proofs.get(target_key),
                            spec_cache=spec_cache,
                            owned=owned,
                            committed_cache=committed_counts,
                            given_up=frozenset(given_up.get(target_key, ())),
                            # The folder is known to have no sidecars (one frame given up on after the note period):
                            # its frames are admitted together, not one given up on per poll.
                            sidecar_optional=meta_auto and not run.sidecars_seen and bool(run.metadata_given_up),
                        )
                        times.add("admission", started)
                        if not run.sidecars_seen and (
                                attempt.waiting_metadata is not None and attempt.waiting_metadata[1] == "settling"
                                or attempt.decision is not None and any(
                                    source.metadata_file is not None
                                    for source in attempt.decision.item.graph.stamp.metadata_sources)):
                            run.sidecars_seen = True  # one is there: the frames after it wait for theirs
                        if self._watch_metadata_wait(
                                run, group.target, target_key, attempt.waiting_metadata, metadata_waits, given_up,
                                retried=attempt.state is ProbeState.IN_PROGRESS,
                                admitted=None if attempt.decision is None else frozenset(
                                    Path(member.path).name for member in attempt.decision.item.source_stamp.members)):
                            skipped_paths.update(path for path in group.physical_paths if path not in processed_paths)
                            publish(force=True)
                            continue
                        live_readiness_debug(
                            "live_attempt", target=group.target.name,
                            owned=owned,
                            state=attempt.state.name, partial=attempt.partial, reason=attempt.reason,
                            disposition=None if attempt.decision is None else attempt.decision.disposition.name)
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
                        if target_key != _live_physical_file_key(
                                run, observation.discovered_snapshot, decision.item.target):
                            raise RuntimeError(
                                "Live output target changed after JIT"
                            )
                        late = next((Path(source.source_path).name for source in decision.item.graph.stamp.metadata_sources
                                     if source.metadata_file is not None
                                     and Path(source.source_path).name in given_up.get(target_key, ())), None)
                        if late is not None:  # its sidecar came between the look and admission's read
                            self._watch_metadata_wait(run, group.target, target_key, (late, "late"), metadata_waits,
                                                      given_up)
                            skipped_paths.update(path for path in group.physical_paths if path not in processed_paths)
                            publish(force=True)
                            continue
                        drift = (
                            () if retained is None or retained.decision is None
                            else published_metadata_drift(
                                retained.decision.item.graph,
                                decision.item.graph,
                            )
                        )
                        if drift:
                            self._block_live_target(
                                run, group.target, target_key, drift,
                            )
                            skipped_paths.update(
                                path for path in group.physical_paths
                                if path not in processed_paths
                            )
                            publish(force=True)
                            continue
                        if attempt.partial and len(
                            decision.item.source_stamp.members
                        ) <= published and (
                            retained is not None and retained.decision is not None
                            and decision.item.source_stamp.members
                            == retained.decision.item.source_stamp.members[
                                :len(decision.item.source_stamp.members)
                            ]
                        ):
                            # Nothing beyond the published frames is ready yet.
                            # A raw rewrite must still reach the Append guard.
                            continue
                        if (decision.disposition is OutputDisposition.REPLACE and run.output is not None
                                and owned):
                            # An image of the scan this Run grows changed after it was reduced: it stops growing, as
                            # its refusal always made it (maintainer, 2026-10-24: "Stop growing it"); a later Run
                            # finds it changed and rebuilds it whole.
                            self._skip_live_append(run, group.target, target_key, None, constructed=False,
                                                   reason=decision.reason, changed=True)
                            skipped_paths.update(
                                path for path in group.physical_paths
                                if path not in processed_paths
                            )
                            publish(force=True)
                            continue
                        if decision.disposition is OutputDisposition.REPLACE:
                            # A scan found changed is reduced again only once its images and sidecars have held still
                            # for _REBUILD_QUIET_S (an image can land before its sidecar); then the next attempt goes on.
                            held = rebuild_holds.get(target_key)
                            if held is None or held[0] != revision:
                                rebuild_holds[target_key] = (revision, monotonic() + _REBUILD_QUIET_S)
                                continue
                            rebuild_holds.pop(target_key)
                        # Only an output that runs now takes over: a scan still waiting, refused or stopped above
                        # leaves the current one open, and its next frame extends it in place (Codex N-0243a1bb F1:
                        # settled to look at a neighbour, it failed that frame on its own repopulated display store).
                        self._release_predecessor_before_target(run, group.target)
                        item = decision.item
                        run.current_file_total = len(group.physical_paths)
                        run.current_files_incremental = (
                            item.source_spec.kind is SourceKind.TIFF_SERIES
                        )
                        try:
                            run.live_admission_proof = attempt.proof
                            constructing = True
                            started = monotonic()
                            self._construct(
                                run,
                                item=item,
                                labels=decision.labels,
                                decision=decision,
                            )
                            times.add("construct", started)
                        except EnergyGridWidened as widened:
                            # An earlier Run's output whose new frames widen the q grid it was reduced on (the output
                            # this Run grows is never checked): it stops growing, named; a directory Run reduces it
                            # again whole (maintainer, 2026-10-25: "Rebuild it whole").
                            self._skip_live_append(run, group.target, target_key, None, constructed=True,
                                                   reason=str(widened), widened=True)
                            skipped_paths.update(path for path in group.physical_paths if path not in processed_paths)
                            publish(force=True)
                            continue
                        except SourceRevisionChanged:
                            self._abandon_construction(run)
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
                except AppendRefused as error:
                    # One scan's Append refusal skips that scan: Live keeps watching the others and the end of the
                    # Run lists it, as a directory Run does (beamline 2026-10-23: a first image rewritten after it
                    # was reduced failed the whole Live Run).
                    live_readiness_debug(
                        "live_refused", target=group.target.name, constructed=constructing,
                        owned=owned, reason=str(error))
                    self._skip_live_append(run, group.target, target_key, error, constructed=constructing)
                    skipped_paths.update(
                        path for path in group.physical_paths
                        if path not in processed_paths
                    )
                    publish(force=True)
                    continue
                except (ValueError, OSError) as error:
                    if isinstance(error, SourceRevisionChanged):
                        retry_revisions[target_key] = revision  # its files changed while it was read: again
                        continue
                    if constructing and isinstance(error, OSError) and not isinstance(error, SourceReadError):
                        raise  # the output cannot be written: the Run fails (maintainer, 2026-10-24)
                    # One scan's own problem (a GI motor missing in a frame, an energy-scan refusal, a read error, an
                    # output collision, an unreadable image) skips that scan; the Run goes on (maintainer, 2026-10-24:
                    # "Skip that scan, keep the Run").
                    live_readiness_debug(
                        "live_refused", target=group.target.name, constructed=constructing,
                        owned=owned, reason=str(error))
                    self._skip_live_append(run, group.target, target_key, None, constructed=constructing,
                                           reason=f"{type(error).__name__}: {error}", failed=True)
                    skipped_paths.update(
                        path for path in group.physical_paths
                        if path not in processed_paths
                    )
                    publish(force=True)
                    continue
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
                        run.live_proofs[target_key] = _live_proof(
                            configuration, decision.item.graph, sidecar_observations, registry)
                        if not attempt.partial:
                            # A partial attempt leaves members still waiting
                            # for their metadata: retry them next poll.
                            settled_revisions[target_key] = (
                                revision,
                                decision.item.source_stamp.execution_identity_v1,
                            )
                        if run.pending_rebuild is not None:  # reduced again whole, so said at the Run's end
                            run.append_rebuilds.append(run.pending_rebuild)
                            run.pending_rebuild = None
                    run.current_file_total = 0
                    run.current_files_incremental = False
                    publish(force=True)

                started = monotonic()
                try:
                    stopped = self._execute_current(
                        run,
                        construct=False,
                        retain_session=True,
                        settle=settle_live,
                    )
                except SourceReadError as error:
                    # One scan's own image could not be read while it was reduced: that scan stops growing, what it
                    # published before is kept, and the Run goes on (Codex N-69301b5f NW-R2).
                    with _run_effect(run):
                        self._settle_failed_scan(run, group.target)
                    self._skip_live_append(run, group.target, target_key, None, constructed=False, failed=True,
                                           reason=f"{type(error).__name__}: {error}")
                    skipped_paths.update(path for path in group.physical_paths if path not in processed_paths)
                    publish(force=True)
                    continue
                times.add("written", started)
                if run.perf_enabled and not first_written and run.current_epoch_published:
                    first_written = True
                    logger.info("[LIVE-POLL] first epoch with new frames finished %.3f s after the first poll "
                                "began (admission and checks before it not included)", monotonic() - loop_started)
                if stopped or run.stop_requested:
                    return True

            for key in [key for key in metadata_waits if key not in polled]:  # its files left the directory
                self._watch_metadata_wait(run, None, key, None, metadata_waits, given_up)
            with run.live_revision_lock:
                containers = tuple(value.decision.item.source_stamp for value in run.processed_live_revisions.values()
                                   if value.decision is not None and value.decision.item.descriptor is not None)
            if run.perf_enabled and (polls < 2 or times.busy):
                logger.info("[LIVE-POLL] poll %d, %d groups, %.3f s: %s", polls, len(groups),
                            monotonic() - poll_started, times.line())
            polls += 1
            # Containers grow in place and their external chunks are never listed.
            session.defer_listing((*watched, *(Path(state.path) for stamp in containers for state in (
                stamp.file, *(member.file for member in stamp.external_members), *stamp.dependency_files))))
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
        # Files admission already found processed count as processed from the start, not pending until each scan is
        # shown (maintainer, 2026-10-23: "it says there are 24 images or so remaining and then goes through them in a
        # couple of seconds"); claimed here, their presentation counts nothing again.
        for output_index, entry in enumerate(deferred.entries):
            if entry.completed is not None:
                run.files_processed += min(len(entry.completed.labels), _claim_physical_files(
                    tuple(candidate.path for candidate in entry.candidates), deferred.discovered_paths,
                    claimed_files, candidate_owners=candidate_owners, output_index=output_index))
        self._events.put(StandardRunEvent(
            run.identity,
            StandardEventKind.DISCOVERY,
            completed=run.completed,
            total=run.total,
            detail=_directory_status(run),
            **_directory_event_fields(run),
        ))
        stopped = False
        # An entry whose files changed while they were read (a scan still being written) is taken once more; the
        # files it had claimed stay claimed.
        retried: set[int] = set()
        reclaimed: dict[int, int] = {}
        position = 0
        while position < len(deferred.entries):
            output_index, entry = position, deferred.entries[position]
            position += 1
            if run.stop_requested:
                stopped = True
                break
            started = monotonic()
            try:
                self._release_predecessor_before_target(
                    run, entry.target,
                )
                if entry.completed is not None:  # admitted as already processed, its source never captured
                    stopped = self._present_already_processed(
                        run, deferred, entry, entry.completed, claimed_files,
                        candidate_owners=candidate_owners,
                        output_index=output_index,
                    )
                    logger.debug(
                        "[APPEND-SCAN] output=%s already processed (%.0f ms)",
                        entry.target, 1000 * (monotonic() - started),
                    )
                    if stopped:
                        break
                    continue
                # A scan that grew keeps the frames its output holds as their record says: only the frames after
                # them are probed, read and proved (once, here), as a Live Run's published prefix is.
                committed, proofs, recheck = getattr(entry, "committed_prefix", None), [], None  # (doubles: none)
                if committed is not None:
                    start, registry = len(committed.stamp.members), registry_identity()
                    record = observe_live_members(
                        run.configuration, [value.path for value in entry.candidates[start:]])

                    def recheck(graph, *, start=start, record=record, registry=registry, proofs=proofs):
                        proofs.append(LivePrefixProof(
                            graph, record=record, registry=registry, start=start,
                            metadata_format=live_metadata_format(run.configuration.thaw_source_spec())))
                        return proofs[-1].holds(cancelled=lambda: run.stop_requested)
                elif all(Path(value.path).suffix.lower() in {".tif", ".tiff"}
                         for value in entry.candidates):
                    # A finite new TIFF scan has the same qualification proof
                    # available to construction as a Live admission.  The
                    # proof is made after its JIT graph exists, then _construct
                    # reuses it instead of validating the graph a second time.
                    registry, record = registry_identity(), observe_live_members(
                        run.configuration, [value.path for value in entry.candidates])

                    def recheck(graph, *, record=record, registry=registry, proofs=proofs):
                        # The inexpensive LivePrefixProof below protects the
                        # interval from materialization through construction;
                        # it does not qualify the graph's detector metadata.
                        # Finite NEW scans therefore perform the same full
                        # graph validation as the ordinary path exactly once.
                        from xrd_tools.sources.execution_graph import validate_source_execution_graph
                        validate_source_execution_graph(
                            graph, cancelled=lambda: run.stop_requested)
                        proofs.append(LivePrefixProof(
                            graph, record=record, registry=registry,
                            metadata_format=live_metadata_format(run.configuration.thaw_source_spec())))
                        return proofs[-1].holds(cancelled=lambda: run.stop_requested)
                decision, _ready_files, skipped_files = (
                    materialize_deferred_output(
                        receipt,
                        session,
                        entry,
                        cancelled=lambda: run.stop_requested,
                        prefix=committed, recheck=recheck,
                    )
                )
            except RuntimeError as error:
                if (
                    run.stop_requested
                    and error.args == ("admission cancelled",)
                ):
                    stopped = True
                    break
                if not isinstance(error, AppendRefused):
                    raise
                # One scan's Append refusal skips that scan; the rest of the
                # directory still runs and the end of the run lists it
                # (maintainer, 2026-10-23). Nothing was constructed for it.
                reason = detach_exception(error, "append").message
                run.append_skips.append((str(entry.target), "append", reason))
                log_events.event(
                    logger, logging.WARNING, "%s skipped: it cannot be extended, so its existing results are left "
                    "unchanged (Replace reprocesses it). Details: %s [APPEND-SKIPPED]", Path(entry.target).name,
                    reason)
                decision, skipped_files = None, 0
            except (ValueError, OSError) as error:  # one scan's own problem: that scan is skipped, the rest still runs
                if isinstance(error, SourceRevisionChanged):  # its files changed while they were read: once more
                    if output_index not in retried:
                        retried.add(output_index)
                        position -= 1
                        continue
                    error = _CHANGED_WHILE_READ
                self._skip_directory_scan(run, entry.target, error)
                decision, skipped_files = None, 0
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
            run.current_file_total = reclaimed.pop(output_index) if output_index in reclaimed else (
                _claim_output_physical_files(
                    item,
                    deferred.discovered_paths,
                    claimed_files,
                    candidate_owners=candidate_owners,
                    output_index=output_index,
                ))
            run.current_files_incremental = (
                item.source_spec.kind is SourceKind.TIFF_SERIES
            )
            remaining_frames = (
                _remaining_tiff_candidates(deferred, output_index)
                if item.source_spec.kind is SourceKind.TIFF_SERIES else None
            )
            admitted = monotonic()
            try:
                # The proof of what was admitted: construction re-checks it instead of validating the scan again.
                run.live_admission_proof = proofs[-1] if proofs else None
                run.prior_energy_selection = None if committed is None else getattr(
                    entry, "committed_energy_selection", None)
                try:
                    self._construct(
                        run,
                        item=item,
                        labels=decision.labels,
                        decision=decision,
                        remaining_frames=remaining_frames,
                    )
                except EnergyGridWidened as widened:
                    # Its new frames widen the q grid its energy scan was reduced on: the whole scan is reduced again
                    # on the wider one, into its output (maintainer, 2026-10-25: "Rebuild it whole"). Construction
                    # refused before it changed anything but its source, which it closed.
                    decision = self._widened_rebuild(run, receipt, session, entry, str(widened))
                    item = decision.item
                    run.rebuild_widened = True
                    self._construct(run, item=item, labels=decision.labels, decision=decision,
                                    remaining_frames=remaining_frames)
            except ValueError as error:  # (an OSError: its output cannot be written, and the Run fails)
                if isinstance(error, SourceRevisionChanged):  # its files changed while they were read: once more
                    if output_index not in retried:
                        retried.add(output_index)
                        reclaimed[output_index] = run.current_file_total
                        run.total -= run.current_total
                        run.files_skipped -= skipped_files
                        self._abandon_construction(run)
                        position -= 1
                        continue
                    error = _CHANGED_WHILE_READ
                self._skip_failed_directory_scan(run, item.target, error, constructed=False)
                continue
            constructed = monotonic()
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
            try:
                stopped = self._execute_current(run, construct=False) or stopped
            except SourceReadError as error:  # one scan's own image could not be read: it is skipped, the rest runs
                self._skip_failed_directory_scan(run, item.target, error, constructed=True)
                continue
            if run.pending_rebuild is not None:  # said at the end only when the scan was reduced again whole
                if run.current_completed >= run.current_total:
                    run.append_rebuilds.append(run.pending_rebuild)
                run.pending_rebuild = None
            logger.debug(
                "[APPEND-SCAN] output=%s frames=%d admit=%.0f ms construct=%.0f ms run=%.0f ms",
                item.target, len(decision.labels), 1000 * (admitted - started),
                1000 * (constructed - admitted), 1000 * (monotonic() - constructed),
            )
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

    def _present_already_processed(
        self,
        run: _StandardRun,
        deferred: DeferredDirectoryPlan,
        entry,
        completed,
        claimed_files: set[Path],
        *,
        candidate_owners: dict[str, int],
        output_index: int,
    ) -> bool:
        """A scan its output already holds, shown as an Append no-op shows it (its saved frames in navigation, its
        files counted processed) without reading the scan again: no source, graph, writer or warning
        (``output_preflight._already_processed``). Returns whether Stop was requested."""
        run.current_file_total = _claim_physical_files(
            tuple(candidate.path for candidate in entry.candidates),
            deferred.discovered_paths,
            claimed_files,
            candidate_owners=candidate_owners,
            output_index=output_index,
        )
        run.current_files_incremental = True
        self._follow_saved_output(run, entry.target, completed)
        owner = self._present_saved_output(run, entry.target, entry.candidates[0], completed)
        for finished in (False, True):
            if finished:
                if run.artifact not in run.artifacts:
                    run.artifacts.append(run.artifact)
                run.display.mark_hydration_closed(owner)
                run.records = None
                run.files_processed += min(run.current_file_total, run.current_completed)
                run.current_file_total = 0
                run.current_files_incremental = False
            self._events.put(StandardRunEvent(
                run.identity,
                StandardEventKind.DISCOVERY,
                completed=run.completed,
                total=run.total,
                artifact=str(entry.target),
                detail=_directory_status(run),
                artifact_completed=run.current_published,
                artifact_total=run.current_total,
                **_directory_event_fields(run),
            ))
        return bool(run.stop_requested)

    def _present_live_saved_output(self, run: _StandardRun, group: LiveDirectoryGroup, completed) -> None:
        """A Live Run's view of a scan its output already holds, as a directory Append shows it; the poll counts its
        files and publishes the status. The output a Live epoch holds open keeps its own session and progress: this
        scan gets no writer, so nothing is settled, and its presentation changes none of that output's fields."""
        idle = run.output is None or not run.output.has_current
        if idle:
            self._follow_saved_output(run, group.target, completed)
        owner = self._present_saved_output(run, group.target, group.plan.candidates[0], completed)
        if group.target not in run.artifacts:
            run.artifacts.append(group.target)
        run.display.mark_hydration_closed(owner)
        if idle:
            run.records = None
            run.current_file_total = 0
            run.current_files_incremental = False

    @staticmethod
    def _follow_saved_output(run: _StandardRun, target: Path, completed) -> None:
        """Point the Run's current-output fields at a saved output it shows while no other output is open, as an
        Append no-op's dormant output leaves them. It has no stream commit in this Run, so a terminal event naming it
        carries none (the last written output's commit made the page drop a finite Append's FINISHED event: no Browse,
        no plot)."""
        run.artifact, run.session, run.showing_saved = target, None, True
        run.current_total = run.current_completed = run.current_published = len(completed.labels)
        run.current_epoch_published = 0
        run.terminal_commit_identity = None

    def _present_saved_output(self, run: _StandardRun, target: Path, first, completed):
        """Show an output's saved frames as an Append no-op shows them (navigation, display owner, acquisition
        context) from its committed record, without reading its scan again: no source, graph, writer or warning
        (``output_preflight._already_processed``). It sets none of the Run's current-output fields (the caller does,
        when no other output is open); the Run's totals count its frames. Returns the display owner; the caller counts
        files."""
        configuration = run.configuration
        labels = tuple(completed.labels)
        run.total += len(labels)
        plan = native_int_reduction_plan(configuration, companion_modes_2d=True)
        admission = None if run.resources is None else run.resources.admission
        assets = None if admission is None else admission.scientific_assets
        try:
            npt = int(configuration.bai_1d_args.get("npt", 0))
        except (TypeError, ValueError):
            npt = 0
        run.display.set_factories(FrameRecordStore, PublicationStore)
        if not run.display.configured:
            self._configure_display(run, npt=npt, frame_bytes=None)
        owner = run.display.artifacts.get(str(target))
        newly_adopted_owner = owner is None
        if owner is None:
            from xrd_tools.sources.adapters import get_adapter
            if configuration.live_mode and run.display.artifacts:
                run.display.admit_additional_partition()  # as a Live Run's construction admits each later output
            owner = run.display.add_artifact(
                target,
                get_adapter(first.adapter_id).scan_name(first.path),
                mask=None if assets is None else assets.mask,
                mask_saturation=bool(getattr(plan, "mask_saturation", False)),
                saturation=getattr(plan, "saturation", None),
                threshold_min=plan.threshold_min,
                threshold_max=plan.threshold_max,
                measurement_mode="GI" if configuration.gi.enabled else "Standard",
                gi_incidence_motor=configuration.gi.incidence_motor if configuration.gi.enabled else "",
                gi_resolved_motor=configuration.gi.effective_motor if configuration.gi.enabled else "",
                gi_mode_1d=configuration.gi.mode_1d if configuration.gi.enabled else "",
                gi_mode_2d=configuration.gi.mode_2d if configuration.gi.enabled else "",
                source_base=configuration.project_root or None,
                wavelength_m=completed.wavelength_m,  # the output's own, as it was processed
            )
        run.display.bind_checkpoint_hydration(owner)
        self._seed_persisted_prefix_navigation(
            run, owner, labels, newly_adopted=newly_adopted_owner, completed=len(labels))
        self._adopt_acquisition_context(run, owner, source_path=str(first.path), saved_only=True)
        run.completed += len(labels)
        return owner

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
        if isinstance(run.source, (NexusStackSource, PointImageSource)) and write_labels:
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
            seal_started = monotonic()
            try:
                if not finished_current:
                    result = output.commit_epoch()
                else:
                    result = output.finish_current(stopped=run.stop_requested)
            except BaseException as error:
                session_error = error
            _perf_add(run, "finish_writer_seal", monotonic() - seal_started)
            if session_error is None:
                drain_started = monotonic()
                try:
                    self._finish_display_projection(run)
                except BaseException as error:
                    projection_error = error
                _perf_add(
                    run, "finish_display_drain", monotonic() - drain_started,
                )
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
                if isinstance(primary := getattr(session, "primary_error", None), SourceReadError):
                    raise primary  # one scan's own image could not be read: its caller skips that scan
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
            if finished_current:
                run.last_written = (run.artifact, run.current_completed, run.current_total,
                                    run.terminal_commit_identity)
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
            run.frames_new += len(new_labels)
            path = Path(artifact)
            if path == run.artifact:
                run.current_completed = min(
                    run.current_total,
                    run.current_completed + len(new_labels),
                )
                # Durable rows may outlive a failed display callback. Their
                # acknowledgement proves a navigation publication only while
                # this artifact's display projection channel is intact.
                if not run.display_projection_failed:
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
        if not isinstance(source, (NexusStackSource, PointImageSource)):
            raise TypeError("container submission requires NexusStackSource")
        typed = _skips_failed_scans(run)

        def publish_direct_fact() -> None:
            take = getattr(source, "take_direct_chunk_fact", None)
            fact = take() if callable(take) else None
            if fact is not None:
                run.resource_facts.append(fact)
                log_events.diagnostic(logger, "%s", fact.log_line("[SOURCE-READ]"))

        wanted = tuple(run.frames_by_label)
        if wanted != tuple(source.frame_indices):
            observed_fallbacks = 0
            try:
                for label in wanted:
                    if run.stop_requested:
                        return
                    frame = run.frames_by_label[label]
                    image = _read_container(lambda: source.load_frame(label), source, f"frame {label}", typed=typed)
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

        cancelled = lambda: (run.stop_requested or run.stop_signal.is_set()
                             or not accepting.is_set())
        grouped = isinstance(source, PointImageSource)
        if grouped:
            from xrd_tools.reduction.core import _condition_point_operand, _wants_poisson
            plan = run.point_plan
            if plan is None:
                raise RuntimeError('grouped container lost its frozen reduction plan')
            chunks = iter(source.iter_point_statistics(1,
                poisson=_wants_poisson(plan), native_transform=lambda native: _condition_point_operand(native, plan),
                cancelled=cancelled))
        else:
            vnext_chunks = getattr(source, "_iter_vnext_chunks", None)
            public_chunks = source.iter_chunks
            canonical = getattr(type(source), "_canonical_iter_chunks", None)
            chunks = iter(vnext_chunks(_CONTAINER_READ_CHUNK_FRAMES, cancelled)
                if (callable(vnext_chunks) and getattr(public_chunks, "__func__", None) is canonical)
                else public_chunks(_CONTAINER_READ_CHUNK_FRAMES))
        producer_error: BaseException | None = None
        try:
            while accepting.is_set() and not run.stop_requested:
                read_started = monotonic()
                try:
                    block, labels = _read_container(lambda: next(chunks), source, "its next frames", typed=typed)
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
                    if grouped:
                        statistics = block[offset]
                        frame.point_statistics_loader = None
                        frame.point_variance = statistics.variance
                        frame.point_exposure_multiplier = statistics.exposure_multiplier
                        image = statistics.signal
                    else:
                        image = np.asarray(block[offset])
                    if not enqueue((frame, image)):
                        break
        except BaseException as error:
            producer_error = error
            accepting.clear()
            runtime = run.context_runtime
            try:
                if isinstance(error, SourceReadError):  # that scan's own read: only its graph stops, the Run goes on
                    output.stop_current()
                elif runtime is None:
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
        publication = publication_from_frame_view(
            view,
            record=record,
            source_identity=source_identity,
            source_base=owner.source_base,
            fallback_path=owner.artifact,
            validate=False,
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
            extra_views_2d=companion_views_2d(record),
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
                # a one-scan Append says its new frames as it runs too (Codex N-acb7b192 B-R3)
                else f"{payload.status}{new_frames_note(_frames_new(run))}"
            ),
            frame_key=key,
            navigation_delta=navigation,
            artifact_completed=artifact_completed,
            artifact_total=run.current_total,
            **_directory_event_fields(run),
        ))

    def _cleanup(self, run: _StandardRun, primary: DetachedDiagnostic | None=None) -> ExecutorClosed:
        run.live_proofs.clear(); run.live_spec_cache.clear()
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
            run.background_capsule = None
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
                and not run.display.payloads
                and all(owner.publications.allocation is None
                        and owner.publications._light_1d is None
                        for owner in run.display.artifacts.values())
            ):
                # Failed construction can leave references to an already
                # released lease/cancelled slot. Append counts persisted rows,
                # so a nonzero catalog does not imply retained display arrays.
                # Retire only after those arrays are gone; actual pending
                # owners still withhold the CLEANED receipt.
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
                run.point_plan = None
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
        warnings = tuple(
            f"Skipped {Path(output).name}: {reason}" if kind == "scan"
            else f"Append skipped {Path(output).name}: {reason}"
            for output, kind, reason in run.append_skips
        )
        rebuilds = list(run.append_rebuilds)
        if run.pending_rebuild is not None and kind is StandardEventKind.FINISHED:  # a one-scan Run's output
            rebuilds.append(run.pending_rebuild)
        rebuilt = tuple(
            f"Append reprocessed {Path(output).name} whole: {reason}"
            for output, reason in rebuilds
        )
        if warnings:
            log_events.event(logger, logging.WARNING, "Skipped in this Run (%d): %s [RUN-SKIPPED]", len(warnings),
                             "; ".join(warnings))
        if run.metadata_given_up:
            log_events.event(logger, logging.WARNING, "Processed without their metadata (%d): %s [LIVE]",
                             len(run.metadata_given_up), ", ".join(run.metadata_given_up))
        if receipt.cleanup_status is CleanupStatus.CLEANUP_PENDING:
            logger.warning(
                "[RUN-CLEANUP] output=%s retained=%s light_pending=%s failures=%s",
                run.artifact,
                tuple(name for name in ("session", "sink", "output", "source", "resources")
                      if getattr(run, name) is not None),
                run.display.light_1d_cleanup_unresolved(),
                tuple(f"{error.operation}: {error.message}"
                      for error in receipt.cleanup_failures),
            )
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
        if kind is StandardEventKind.FAILED:
            diagnostic = receipt.primary or next(iter(receipt.cleanup_failures), None)
            logger.error(
                "[RUN-FAILED] output=%s error=%s: %s",
                run.artifact,
                diagnostic.type_qualname if diagnostic is not None else "ExecutionError",
                detail,
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
            ("xye_worker_format", "sink_xye_worker_format"),
            ("xye_enqueue_wait", "sink_xye_enqueue_wait"),
            ("xye_queue_high_water", "sink_xye_queue_high_water"),
            ("xye_drain", "sink_xye_drain"),
            ("xye_promotion", "sink_xye_promotion"),
            ("xye_generated", "sink_xye_generated"),
            ("finish_wait", "finish_wait"),
            ("finish_writer_seal", "finish_writer_seal"),
            ("finish_display_drain", "finish_display_drain"),
        ):
            if key in perf:
                timing_details.append((name, float(perf[key])))
        timing_details.extend(
            (key, float(value)) for key, value in perf.items()
            if key.startswith("writer_finish_")
        )
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
        shown = (run.artifact, run.current_completed, run.current_total, run.terminal_commit_identity)
        if run.showing_saved and run.last_written is not None:
            # It ended showing a saved scan it did not change: its end shows the last output it wrote, with that
            # output's own commit (a saved scan has none in this Run).
            shown = run.last_written
        self._events.put(StandardRunEvent(
            run.identity,
            kind,
            completed=completed,
            total=total,
            artifact=str(shown[0]),
            detail=detail,
            cleanup_status=receipt.cleanup_status,
            primary=receipt.primary,
            cleanup_failures=receipt.cleanup_failures,
            artifacts=tuple(str(item) for item in run.artifacts),
            artifact_completed=shown[1],
            artifact_total=shown[2],
            terminal_timing=timing,
            terminal_commit_identity=(
                shown[3]
                if kind is StandardEventKind.FINISHED
                and receipt.cleanup_status is CleanupStatus.CLEANED
                else None
            ),
            warnings=warnings,
            rebuilt=rebuilt,
            artifact_families=tuple(
                (str(item), run.artifact_families[str(item)])
                for item in run.artifacts if str(item) in run.artifact_families
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
        for name in list(run.open_scans):  # the scans still open close with the Run, ahead of its totals
            self._close_scan(run, Path(name), run.current_completed if name == Path(str(run.artifact)).name else None)
        # End of Run below says it plainly; these are for diagnosis (maintainer, 2026-10-24)
        log_events.diagnostic(logger, "Total Frames Processed: %d", completed, shown=run.perf_enabled)
        if elapsed is None:
            log_events.diagnostic(logger, "Total Time: unavailable", shown=run.perf_enabled)
            log_events.diagnostic(
                logger,
                "[PERF-SUMMARY] outcome=%s frames=%d/%d cores=%s | "
                "total=unavailable | output=%s",
                kind.value,
                completed,
                total,
                core_count if core_count > 0 else "unknown",
                run.artifact,
                shown=run.perf_enabled,
            )
        else:
            log_events.diagnostic(logger, "Total Time: %.2fs", measured_elapsed, shown=run.perf_enabled)
            log_events.diagnostic(
                logger,
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
                shown=run.perf_enabled,
            )
        for fact in run.resource_facts:
            log_events.diagnostic(logger, "%s", fact.log_line("[PERF-RESOURCES]"), shown=run.perf_enabled)
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
            if "sink_xye_generated" in perf:
                logger.info(
                    "[PERF-XYE] worker-format=%.3fs enqueue-wait=%.3fs "
                    "queue-high-water=%d drain=%.3fs promotion=%.3fs "
                    "files=%d (parallel timers may overlap)",
                    perf.get("sink_xye_worker_format", 0.0),
                    perf.get("sink_xye_enqueue_wait", 0.0),
                    perf.get("sink_xye_queue_high_water", 0),
                    perf.get("sink_xye_drain", 0.0),
                    perf.get("sink_xye_promotion", 0.0),
                    perf["sink_xye_generated"],
                )
            if "writer_finish_close" in perf:
                logger.info(
                    "[PERF-WRITER-FINISH] %s (science-readback is included in close)",
                    " ".join(f"{key.removeprefix('writer_finish_')}={value:.3f}s"
                             for key, value in perf.items()
                             if key.startswith("writer_finish_")),
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
        new = new_frames_note(_frames_new(run))
        log_events.banner(
            logger, "End of Run · %s · %s%s · %.1f s", kind.value.capitalize(),
            _counted(completed, "frame") + ("" if run.files_discovered else new),
            f" · {_counted(run.files_processed, 'file')} processed{new}, {run.files_skipped} skipped"
            if run.files_discovered else "", measured_elapsed, kind="run-end")

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
