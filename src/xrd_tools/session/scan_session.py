# -*- coding: utf-8 -*-
"""The headless :class:`ScanSession` + its immutable event types.

``ScanSession`` wraps a streaming ``ReductionSession`` and a ``ReductionSink``.
The user's sink is wrapped in an internal event-emitting decorator
(:class:`_EventSink`) that forwards every hook the engine probes
(``begin``/``write``/``replace``/``finish``/``abort``/``worker_process``/
``flush``) and, after each ``write``/``replace``, fires ``on_frame_completed``
on the session's single writer thread — preserving the HDF5 single-writer
invariant (ADR-0004 §1).

Threading (ADR-0004): ``on_frame_completed`` and the completion-side
``on_progress`` fire on the WRITER thread; ``on_state_change`` and the
submit-side ``on_progress`` fire on the caller thread.  A callback that raises
is caught + logged — a listener can never kill the writer (the T0-7/S1
false-success trap).  A Qt bridge marshals ``on_frame_completed`` via a
``QueuedConnection``.

This module is Qt-free (numpy only via the result containers).  Per ADR-0005's
refinement of ADR-0004 §4, the *persist-before-evict* bookkeeping now lives here:
the optional ``record_store`` is upserted after each sink ``write``, and a
buffering sink marks its records persisted from ``flush`` (see :class:`ScanSession`).
Only the Qt/file-handle flush *action* and the ``FlushPolicy`` *timing* remain
xdart-adapter concerns; the session exposes ``flush`` as a contract pass-through to
the sink.

H10-C1: the session composes a :class:`~xrd_tools.session.stage_accounting.
StageLedger` — the typed stage-accounting authority (accepted / completed /
written / persisted / durable as identity sets, §4.1 of the H10 handoff).
Acceptance is admitted from INSIDE the engine's ``submit`` (the narrow
``accept_cb`` seam), the instant acceptance becomes irreversible and before
its ACCEPTED decision opens worker and writer, so no write or completion can
ever publish work the ledger has not yet accepted; the ledger mints that
submission's per-label ``attempt_revision`` there.  Typed compute outcomes
arrive through the engine's per-item
:class:`~xrd_tools.reduction.FrameOutcomeReceipt` carrying the same attempt
identity (so a failed sink write stays distinguishable from a failed compute,
and an older overlapping attempt cannot overwrite a newer one's state), and
``written`` is recorded when the top-level sink hook returns.  Persistence /
durability advance only on explicit target-qualified receipts delivered to
:attr:`ScanSession.accounting` by the flush-boundary owner.
``frames_submitted``/``frames_completed`` are DERIVED compatibility
projections of the ledger's identity facts — distinct accepted labels and
distinct labels ever successfully written — never independent counters.
``FrameEvent.generation`` stays a render-staleness stamp, excluded from every
accounting identity.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from xrd_tools.core import DEFAULT_MODE_KEY, FrameRecord, FrameView
from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from xrd_tools.io import AppendDecision, AppendIntent
from xrd_tools.reduction import (
    Frame,
    FrameOutcome,
    FrameOutcomeReceipt,
    NexusTerminalDisposition,
    NexusTerminalResult,
    OutputSinkKind,
    ReductionPlan,
    ReductionResult,
    ReductionSession,
    StrictPolicy,
    bind_dynamic_output_sink,
)
from .dynamic_accounting import (
    DynamicAttemptState,
    DynamicRunAccounting,
    DynamicRunState,
)
from .frame_record_store import FrameRecordStore
from .policy import (SessionPolicy, _int, requirements_from,
                     resolve_session_policy)
from .stage_accounting import (
    ItemDisposition, ResultMode, StageLedger, StageReceipt, StageSnapshot,
    freeze_target_map)

logger = logging.getLogger(__name__)
_ATTEMPT_MISSING = object()


class _StageBoundaryFacade:
    """The PRIVATE stage boundary a sink binds to; the session stays sole owner."""
    __slots__ = ("_session",)
    def __init__(self, session: "ScanSession") -> None:
        self._session = session
    def targets_for(self, mode: ResultMode) -> frozenset[str]:
        return self._session.accounting.targets_by_mode.get(mode, frozenset())
    def capture_receipt(self, label: int, mode: ResultMode,
                        target: str) -> StageReceipt:
        return self._session.accounting.receipt(label, mode, target)
    def commit_durable(self, receipts: Iterable[StageReceipt]) -> None:
        self._session.record_durable(receipts)
    def commit_publication_drop(self, label: int, mode: ResultMode,
                                expected_revision: int) -> None:
        self._session.record_publication_dropped(
            label, mode, expected_revision=expected_revision)


# ── immutable events ────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class FrameEvent:
    """One frame finished reducing (ADR-0003: single-result + the mode it was
    computed under).  Immutable; built from the engine's ``FrameReduction``."""

    frame_index: int
    mode_key: Any                      # GI (mode_1d, mode_2d) value tuple, or None
    result_1d: IntegrationResult1D | None
    result_2d: IntegrationResult2D | None
    metadata: Mapping[str, Any]
    generation: int                    # caller-owned stale-render stamp (ADR-0004 §2)
    timestamp: float                   # wall-clock completion (time.time())


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """Absolute (not delta) progress counts; may fire from two threads, so
    consumers treat it as idempotent."""

    submitted: int
    completed: int
    total: int | None


@dataclass(frozen=True, slots=True)
class StateChangeEvent:
    """Run-state transition (fires on the caller thread)."""

    is_running: bool
    is_paused: bool


# ── internal sink decorator ───────────────────────────────────────────────────

class _EventSink:
    """Wrap the user's sink: forward every probed hook, and after each
    ``write``/``replace`` fire the completion callback on the writer thread.

    Forwarding the *optional* hooks (``replace``/``abort``/``worker_process``/
    ``flush``) is essential — defining them unconditionally would otherwise make
    the engine treat a plain sink as replace/abort-capable, or (if omitted)
    disable the parallel ``worker_process`` thumbnail path.  Each forwards to the
    inner sink only when the inner sink actually provides it.
    """

    def __init__(self, inner, on_completed: Callable[[Frame, Any], None], *,
                 defer_terminal: bool = False) -> None:
        self._inner = inner
        self._on_completed = on_completed
        self._defer_terminal = bool(defer_terminal)
        worker_process = getattr(inner, "worker_process", None)
        if callable(worker_process):
            self.worker_process = worker_process

    def begin(self, scan, plan) -> None:
        if self._inner is not None:
            self._inner.begin(scan, plan)

    def _bind_run_saturation_mask(self, state) -> None:
        bind = getattr(self._inner, "_bind_run_saturation_mask", None)
        if callable(bind):
            bind(state)

    def write(self, frame, reduction) -> None:
        if self._inner is not None:
            self._inner.write(frame, reduction)

    def replace(self, frame, reduction) -> None:
        inner_replace = getattr(self._inner, "replace", None)
        if callable(inner_replace):
            inner_replace(frame, reduction)
        elif self._inner is not None:
            # No replace hook → the engine would have called write(); match it.
            self._inner.write(frame, reduction)

    def _post_write(self, frame, reduction) -> None:
        self._on_completed(frame, reduction)

    def _settle_deferred_publication_drops(self, frame, reduction) -> None:
        settle = getattr(self._inner, "_settle_deferred_publication_drops", None)
        if callable(settle):
            settle(frame, reduction)

    def finish(self, result):
        if self._defer_terminal or self._inner is None:
            return None
        return self._inner.finish(result)

    def abort(self, result):
        if self._defer_terminal:
            return None
        inner_abort = getattr(self._inner, "abort", None)
        if callable(inner_abort):
            return inner_abort(result)
        if self._inner is not None:
            return self._inner.finish(result)
        return None

    def flush(self, *, force: bool = False) -> None:
        f = getattr(self._inner, "flush", None)
        if callable(f):
            f(force=force)
            return
        # Interim: the xdart QtNexusSink still exposes the historical private
        # `_flush`; honour it until the bridge renames it (ADR-0004 §4).
        _f = getattr(self._inner, "_flush", None)
        if callable(_f):
            _f(force=force)


def _mode_key_from_plan(plan: ReductionPlan):
    """The GI sub-mode key (``(mode_1d, mode_2d)`` values) a result was computed
    under, or ``None`` for a standard scan — ADR-0003's per-completion mode tag."""
    gi = getattr(plan, "gi", None)
    if gi is None:
        return None
    m1 = getattr(gi, "mode_1d", None)
    m2 = getattr(gi, "mode_2d", None)
    return (getattr(m1, "value", m1), getattr(m2, "value", m2))


def _dimension_modes(mode_key: Any) -> tuple[str, str]:
    if isinstance(mode_key, tuple) and len(mode_key) == 2:
        m1, m2 = mode_key
        return str(m1 or DEFAULT_MODE_KEY), str(m2 or DEFAULT_MODE_KEY)
    return DEFAULT_MODE_KEY, DEFAULT_MODE_KEY


def _required_modes_from_plan(plan: ReductionPlan, mode_key: Any) -> tuple[ResultMode, ...]:
    """The run's frozen required result modes: one per integration dimension
    the plan declares, under the plan's (GI) mode keys."""
    mode_1d, mode_2d = _dimension_modes(mode_key)
    modes: list[ResultMode] = []
    if getattr(plan, "integration_1d", None) is not None:
        modes.append(ResultMode.one_d(mode_1d))
    if getattr(plan, "integration_2d", None) is not None:
        modes.append(ResultMode.two_d(mode_2d))
    return tuple(modes)


def _executor_request(executor, executor_workers) -> tuple[int | None, bool]:
    """``(requested_workers, is_caller_pool)``.  An integer ``executor=N`` is a
    positive worker REQUEST; a caller pool must PROVE a positive capacity, fit
    the grant exactly, is never shut down, and two spellings must agree."""
    if executor_workers is not None:
        _int("executor_workers", executor_workers, low=1)
    if executor is None:
        return (None if executor_workers is None
                else int(executor_workers)), False
    if isinstance(executor, (int, float)):
        _int("executor", executor, low=1)
        if executor_workers is not None and int(executor_workers) != executor:
            raise ValueError(f"executor={executor} and executor_workers="
                             f"{executor_workers} must agree")
        return int(executor), False
    declared = getattr(executor, "_max_workers", None)
    if declared is not None and executor_workers is not None:
        _int("_max_workers", declared, low=1)
        if int(declared) != int(executor_workers):
            raise ValueError(f"_max_workers={declared} and executor_workers="
                             f"{executor_workers} must agree")
    if declared is None:
        declared = executor_workers
    if declared is None:
        raise ValueError(
            "a descriptor-managed source needs an external executor exposing a "
            "positive _max_workers, or an explicit executor_workers")
    return _int("caller pool capacity", declared, low=1), True


def required_result_modes(plan: ReductionPlan) -> tuple[ResultMode, ...]:
    return _required_modes_from_plan(plan, _mode_key_from_plan(plan))


def _freeze_result_arrays(result):
    """Mark a result's ndarray fields read-only IN PLACE (zero-copy) so a
    FrameEvent listener cannot retroactively corrupt the shared, already-written
    arrays (the completion fires AFTER the sink's write, and the event holds the
    SAME ndarray objects the sink stored — a listener writing into them would
    poison persisted/cached data).  This makes the "immutable event" contract
    real without the deep-copy that would defeat retain_products=False.  Returns
    the same object.  Defensive: skips anything not a writeable ndarray."""
    if result is None:
        return result
    for attr in ("radial", "azimuthal", "intensity", "sigma"):
        arr = getattr(result, attr, None)
        if isinstance(arr, np.ndarray) and arr.flags.writeable:
            try:
                arr.flags.writeable = False
            except (ValueError, AttributeError):
                pass  # a view that doesn't own its data / can't toggle — leave it
    return result


# ── the session ───────────────────────────────────────────────────────────────

class ScanSession:
    """Drive a streaming scan reduction by commands in / events out.

    Construction arms the underlying streaming ``ReductionSession`` (its writer
    thread starts + ``sink.begin`` runs), so :meth:`start` is an idempotent
    confirmation.  Feed frames with :meth:`submit`; consume results by
    registering :meth:`on_frame_completed`.  Always :meth:`finish` (or use it as
    a context manager) to drain the writer + finalize the sink.

    ``record_store`` is optional and dormant for existing callers.  When supplied,
    completed frame records are upserted after the sink write.  Set
    ``record_store_persisted_on_write=True`` ONLY for sinks whose ``write`` hook is
    itself durable (the frame is on disk when ``write`` returns).  Do NOT set it
    for a buffering sink — notably xdart's ``QtNexusSink``, whose ``write`` only
    stashes in memory and whose durable boundary is a separate ``flush``: marking
    persisted-on-write there would let heavy arrays be evicted before they are
    written (persist-before-evict violation).  Such a caller must instead call
    ``record_store.mark_persisted(labels)`` from its flush completion.

    ``obligations`` (H10-C1) declares the run's frozen output obligation /
    target identities (e.g. ``("nexus:/data/run42.nxs",)``) for the composed
    :attr:`accounting` ledger.  Persisted/durable state advances only on
    explicit :class:`~xrd_tools.session.stage_accounting.StageReceipt`\\ s for
    those targets; with no declared obligation nothing is ever reported
    durable (no vacuous recoverability claim).
    """

    @classmethod
    def new_accounting(cls, plan, targets_by_mode):
        return cls(plan, None, targets_by_mode=targets_by_mode,
                   _accounting_only=True).accounting

    def __init__(
        self,
        plan: ReductionPlan,
        source: Any,
        sink: Any = None,
        *,
        executor: Any | None = None,
        inflight_max: int | None = None,
        gi_freeze_mode: str | None = None,
        cancel_token: Any | None = None,
        clear_frame_images: bool = False,
        record_store: FrameRecordStore | None = None,
        record_store_persisted_on_write: bool = False,
        strict: StrictPolicy | None = None,
        obligations: Iterable[str] = (),
        targets_by_mode: Mapping[ResultMode, Iterable[str]] | None = None,
        store_targets_by_mode: Mapping[ResultMode, Iterable[str]] | None = None,
        write_targets_by_mode: Mapping[ResultMode, Iterable[str]] | None = None,
        accounting: StageLedger | None = None,
        dynamic_accounting: DynamicRunAccounting | None = None,
        policy: SessionPolicy | None = None,
        envelope_bytes: int | None = None,
        executor_workers: int | None = None,
        _accounting_only: bool = False,
    ) -> None:
        self._lock = threading.RLock()
        self._frame_cbs: list[Callable[[FrameEvent], None]] = []
        self._progress_cbs: list[Callable[[ProgressEvent], None]] = []
        self._state_cbs: list[Callable[[StateChangeEvent], None]] = []
        self._generation = 0
        self._mode_key = _mode_key_from_plan(plan)
        required = _required_modes_from_plan(plan, self._mode_key)
        if (
            dynamic_accounting is not None
            and type(dynamic_accounting) is not DynamicRunAccounting
        ):
            raise TypeError("dynamic_accounting must be an exact DynamicRunAccounting")
        self._dynamic_accounting = dynamic_accounting
        if dynamic_accounting is not None:
            if accounting is not None and accounting is not dynamic_accounting.ledger:
                raise ValueError(
                    "dynamic accounting must borrow the exact supplied StageLedger"
                )
            accounting = dynamic_accounting.ledger
        if accounting is None:
            self._accounting = StageLedger(
                required_modes=required,
                obligations=obligations,
                targets_by_mode=targets_by_mode,
            )
        else:
            if not isinstance(accounting, StageLedger):
                raise TypeError("accounting must be a StageLedger")
            if tuple(accounting.required_modes) != tuple(required):
                raise ValueError(
                    "accounting required modes must exactly match the plan")
            if targets_by_mode is None:
                declared = frozenset(str(target) for target in obligations)
                expected_targets = {mode: declared for mode in required}
            else:
                if tuple(obligations):
                    raise ValueError(
                        "a non-empty global obligations set cannot be combined "
                        "with an explicit targets_by_mode map")
                expected_targets = freeze_target_map(
                    "targets_by_mode", targets_by_mode, required)
            if dict(accounting.targets_by_mode) != dict(expected_targets):
                raise ValueError(
                    "accounting target map must exactly match the session")
            self._accounting = accounting
        if _accounting_only:
            return
        self._dynamic_nexus_sink = None
        if dynamic_accounting is not None:
            binding = bind_dynamic_output_sink(sink)
            if OutputSinkKind.XYE in binding.families:
                raise TypeError("dynamic XYE output remains outside the C2 envelope")
            nexus = binding.nexus_sink
            if nexus is not None:
                if nexus.allow_unbound_same_run and nexus.same_run_intent is None:
                    raise ValueError(
                        "dynamic ScanSession refuses unbound same-run adoption"
                    )
                if nexus.flush_every is not None:
                    raise ValueError("dynamic Nexus sink requires flush_every=None")
                expected = f"nexus:{nexus.path}"
                if any(
                    self._accounting.targets_by_mode.get(mode) != frozenset((expected,))
                    for mode in required
                ):
                    raise ValueError(
                        "dynamic Nexus target must exactly match every required mode"
                    )
            sink = binding.sink
            self._dynamic_nexus_sink = nexus
        applicable = self._accounting.targets_by_mode
        self._store_targets = (
            applicable if store_targets_by_mode is None else
            freeze_target_map("store_targets_by_mode", store_targets_by_mode,
                              required, allow_empty=True, within=applicable))
        self._record_store_persisted_on_write = bool(record_store_persisted_on_write)
        if write_targets_by_mode is None:
            if self._record_store_persisted_on_write:
                raise ValueError(
                    "record_store_persisted_on_write=True must declare exactly one "
                    "applicable target per mode via write_targets_by_mode")
            self._write_targets: Mapping[ResultMode, frozenset[str]] = MappingProxyType({})
        elif not self._record_store_persisted_on_write:
            raise ValueError(
                "write_targets_by_mode is meaningless without "
                "record_store_persisted_on_write=True")
        else:
            self._write_targets = freeze_target_map(
                "write_targets_by_mode", write_targets_by_mode, required,
                allow_empty=False, exactly_one=True, within=applicable)
        # One projection lock; fixed order session -> ledger -> store.
        self._projection_lock = threading.RLock()
        self._blocked: dict[int, set[ResultMode]] = {}
        self._pending: dict[int, set[ResultMode]] = {}
        self._swept = False
        self._record_store = record_store
        self._perf_enabled = bool(os.environ.get("XDART_PERF"))
        self._perf_values: dict[str, float] = {}
        self._policy = policy
        allocation = self._resolve_resource_envelope(
            source, plan, executor, executor_workers, envelope_bytes,
            inflight_max)
        if allocation is not None:
            # The grants are the ACTUAL owned configuration, not report fields.
            inflight_max = allocation.reduction_inflight
            if executor is None or isinstance(executor, int):
                executor = allocation.workers
        self._user_sink = sink
        self._dynamic_boundary = (
            None if dynamic_accounting is None else dynamic_accounting.writer_boundary
        )
        self._dynamic_owner_token = object() if dynamic_accounting is not None else None
        self._dynamic_submit_token = None
        self._dynamic_stop_requested = False
        self._dynamic_frozen_result: ReductionResult | None = None
        self._dynamic_primary_error: BaseException | None = None
        self._dynamic_finish_seal = None
        self._dynamic_epoch_seal = None
        self._dynamic_epoch_anchor = None
        self._dynamic_settled_epoch_anchor = None
        self._dynamic_epoch_notified = False
        self._terminal_result: NexusTerminalResult | None = None
        self._dynamic_graph_terminal_settled = False
        self._dynamic_terminal_settled = False
        self._terminal_state_emitted = False
        self._dynamic_extend_live = None
        self._dynamic_extension_owner = None
        self._dynamic_current_intent = None
        nexus = self._dynamic_nexus_sink
        prior_facade = None if nexus is None else nexus._session_facade
        if hasattr(sink, "bind_session"):
            sink.bind_session(
                self._dynamic_boundary
                if dynamic_accounting is not None
                else _StageBoundaryFacade(self)
            )
        event_sink = _EventSink(
            sink, self._on_completed, defer_terminal=False,
        )
        # Streaming + retain_products=False: per-frame results are delivered via
        # events (and persisted by a durable sink), so the session does not also
        # hoard every FrameReduction (the S2 ~14 GB-on-10k-frames trap).
        try:
            self._session = ReductionSession(
                plan,
                source,
                event_sink,
                execution="streaming",
                executor=executor,
                inflight_max=inflight_max,
                gi_freeze_mode=gi_freeze_mode,
                cancel_token=cancel_token,
                # Strictness policy (default loud, matching ReductionSession).  The GUI
                # write path (open_live_scan_session) passes graceful() so a per-frame
                # degradation skips-and-defers instead of aborting the whole-scan save —
                # the streaming path's per-frame contract (B-1 regression fix).
                strict=strict if strict is not None else StrictPolicy.loud(),
                retain_products=False,
                # The writer nulls frame.image after each write so the source-array
                # reference doesn't pin ~18 MB/frame for the session's life (xdart's
                # PERF-3); a later consumer reloads via Frame.load_image.  Default
                # off — a notebook caller keeping the source frames opts in.
                clear_frame_images=clear_frame_images,
                # H10-C1: acceptance is admitted synchronously inside submit(), as
                # the last fallible step before ACCEPTED opens the worker and
                # writer (so `accepted` is never behind a completion), and per-item
                # compute outcomes
                # (success/failure/cancellation) feed the stage ledger from the
                # writer loop — never inferred from accepted-minus-written counts.
                accept_cb=self._on_accepted,
                outcome_cb=(
                    None if dynamic_accounting is not None else self._on_outcome
                ),
                outcome_authority_cb=(
                    self._on_dynamic_outcome
                    if dynamic_accounting is not None else None
                ),
                written_authority_cb=(
                    self._on_dynamic_written
                    if dynamic_accounting is not None else None
                ),
            )
        except BaseException as primary:
            try:
                if nexus is not None:
                    nexus.bind_session(prior_facade)
            except BaseException as cleanup:
                raise primary from cleanup
            raise
        self._event_sink = event_sink
        try:
            if (dynamic_accounting is not None
                    and self._dynamic_nexus_sink is not None
                    and type(self._dynamic_nexus_sink.same_run_intent) is AppendIntent):
                self._dynamic_extend_live = self._dynamic_nexus_sink.extend_live
                self._dynamic_extension_owner = self._dynamic_nexus_sink.extension_owner
                self._dynamic_current_intent = self._dynamic_nexus_sink.same_run_intent
            if dynamic_accounting is not None:
                self._dynamic_boundary.bind_live_session(
                    self, self._dynamic_owner_token,
                )
        except BaseException as primary:
            try:
                try:
                    self._session._rollback_construction(primary)
                finally:
                    if nexus is not None:
                        nexus.bind_session(prior_facade)
                if nexus is not None:
                    terminal = nexus._terminal_result
                    if (type(terminal) is not NexusTerminalResult
                            or terminal.disposition is not NexusTerminalDisposition.ABORTED):
                        raise RuntimeError("construction cleanup did not abort Nexus")
            except BaseException as cleanup:
                raise primary from cleanup
            raise
        if dynamic_accounting is not None:
            event_sink._defer_terminal = True
            if self._dynamic_nexus_sink is not None:
                self._dynamic_nexus_sink._defer_publication_drop_settlement = True

    # -- context manager ---------------------------------------------------
    def __enter__(self) -> "ScanSession":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Mirror ReductionSession.__exit__: never raise a fresh failure during an
        # exception unwind; surface the run failure only on a clean exit.
        self.finish(raise_on_failure=exc_type is None)

    def _resolve_resource_envelope(self, source, plan, executor,
                                   executor_workers, envelope_bytes,
                                   inflight_max):
        """The coordinated descriptor-backed path: descriptor -> requirements ->
        one policy -> bind the exact allocation, all BEFORE the sink hook, the
        ``ReductionSession`` or any read.  A materialized C1 ``Scan`` is not
        descriptor-managed, so its minimal duck is untouched."""
        bind = getattr(source, "bind_allocation", None)
        if not callable(bind):
            return None
        declared, caller_pool = _executor_request(executor, executor_workers)
        if inflight_max is not None:
            _int("inflight_max", inflight_max, low=1)
        requirements = requirements_from(source.container_descriptor(), plan)
        prior = self._policy
        # A cadence-only SessionPolicy(allocation=None) is NOT explicit.
        explicit = None if prior is None else prior.allocation
        policy = resolve_session_policy(
            requirements, envelope_bytes=envelope_bytes,
            requested_workers=declared,
            requests=(None if inflight_max is None
                      else {"reduction_inflight": int(inflight_max)}),
            flush=None if prior is None else prior.flush, allocation=explicit)
        alloc = policy.allocation
        if caller_pool and declared != alloc.workers:
            raise ValueError(f"caller pool declares {declared} workers but the "
                             f"allocation grants {alloc.workers} exactly")
        if explicit is not None:
            if declared is not None and not caller_pool and declared < alloc.workers:
                raise ValueError(f"owned worker request {declared} is below the "
                                 f"explicit grant {alloc.workers}")
            if inflight_max is not None and int(inflight_max) != alloc.reduction_inflight:
                raise ValueError(f"inflight_max {int(inflight_max)} != explicit "
                                 f"grant {alloc.reduction_inflight}")
        self._policy = policy
        bind(alloc)
        return alloc
    def bind_session_policy(self, policy: SessionPolicy) -> None:
        """Associate the ONE run policy the image/source owner already resolved;
        a second, different policy object is rejected."""
        if self._policy is not None and self._policy is not policy:
            raise ValueError("a different session policy is already bound")
        self._policy = policy
    @property
    def policy(self) -> SessionPolicy | None:
        """The run's one immutable policy, or ``None`` when cadence-only."""
        return self._policy

    @property
    def record_store(self) -> FrameRecordStore | None:
        return self._record_store

    @property
    def accounting(self) -> StageLedger:
        """The composed stage-accounting authority (H10-C1).  Flush-boundary
        owners deliver persisted/durable :class:`StageReceipt`\\ s here."""
        return self._accounting

    def accounting_snapshot(self) -> StageSnapshot:
        """The public typed accounting snapshot (§4.1 exposure contract)."""
        return self._accounting.snapshot()

    # -- commands in -------------------------------------------------------
    def start(self) -> None:
        """Idempotent: the writer is armed at construction; emit the initial
        running state once."""
        self._emit_state()

    def submit(
        self,
        frame: Frame,
        image: np.ndarray | None = None,
        *,
        attempt_token: object = _ATTEMPT_MISSING,
    ) -> bool:
        """Feed one frame.

        Returns True when accepted, False when DROPPED (cancelled / writer-dead
        while waiting on a full in-flight window).  CALLER-CONTRACT VIOLATIONS
        RAISE rather than return False (mirroring ``ReductionSession.submit``):
        calling submit() after :meth:`finish`, or while paused, raises
        ``RuntimeError`` — these are misuse, kept loud on purpose, not a normal
        "dropped" outcome.  An accounting-authority exception also RAISES loudly.
        Before publishing an acceptance proof, the engine rejects the ticket,
        undoes its staged facts, and leaves no ledger fact, scan inventory, queue,
        reduction, sink, outcome, or completion.  Once the authority publishes the attempt,
        acceptance is irreversible even if it then raises: the item remains
        inventoried and ledger-visible, the session sticky-fails and cancels, and the
        accepted writer path completes it.  Advances submitted-progress only
        when accepted; a call that does not return (an
        operator interrupt in the engine's acceptance tail) emits no submit-side
        event, and the totals it already advanced surface on the next one.

        Acceptance itself is recorded by :meth:`_on_accepted`, which the engine
        invokes from inside its own ``submit`` at the exact point acceptance
        becomes irreversible — never here, after the writer may already have
        completed the frame.

        Called from one orchestrating thread, as are :meth:`pause`/
        :meth:`resume`.  A streaming ``executor`` must be asynchronous (its
        ``submit()`` returns before the submitted callable needs its admission
        decision); its Future need expose only a blocking ``result()``."""
        dynamic = self._dynamic_accounting
        if dynamic is None:
            if attempt_token is not _ATTEMPT_MISSING:
                raise ValueError("static ScanSession rejects an attempt_token")
        else:
            if attempt_token is _ATTEMPT_MISSING:
                raise TypeError("dynamic ScanSession submit is missing attempt_token")
            dynamic.validate_submission(attempt_token, int(frame.index))
            if self._dynamic_submit_token is not None:
                raise RuntimeError("dynamic submit capability is already in use")
            self._dynamic_submit_token = attempt_token
        try:
            accepted = self._session.submit(frame, image)
        finally:
            if dynamic is not None:
                self._dynamic_submit_token = None
        if accepted:
            self._emit_progress()
        else:
            if dynamic is None:
                self._accounting.record_refused(int(frame.index))
        return accepted

    def pause(self, timeout: float | None = None) -> bool:
        """Quiesce the writer at a frame boundary (delegates to
        ``ReductionSession.pause``).  Returns whether it fully drained."""
        drained = self._session.pause(timeout=timeout)
        self._emit_state()
        return drained

    def resume(self) -> None:
        self._session.resume()
        self._emit_state()

    def stop(self) -> None:
        """Cooperative cancel (sets the cancel token); the writer stops at the
        next boundary.  Call :meth:`finish` to drain + finalize."""
        if self._dynamic_accounting is not None and not self._dynamic_stop_requested:
            self._dynamic_accounting.stop()
            self._dynamic_stop_requested = True
        self._session.cancel_token.cancel()
        self._emit_state()

    def finish(self, *, raise_on_failure: bool = True,
               join_timeout: float | None = None) -> ReductionResult:
        """Drain the writer, finalize the sink, return the result.  Idempotent:
        a second finish() returns the same result and does NOT re-emit a
        state-change event (so a bridge that tears down on the running→finished
        transition can't double-fire)."""
        if self._dynamic_accounting is None:
            was_running = self.is_running
            try:
                result = self._session.finish(
                    raise_on_failure=raise_on_failure, join_timeout=join_timeout)
            finally:
                self._final_sweep()
            if was_running:
                self._emit_state()
            return result
        return self._finish_dynamic(
            raise_on_failure=raise_on_failure, join_timeout=join_timeout,
        )

    @property
    def terminal_result(self) -> NexusTerminalResult | None:
        return self._terminal_result

    def _freeze_dynamic_result(
        self, join_timeout: float | None,
    ) -> ReductionResult:
        if self._dynamic_frozen_result is None:
            self._dynamic_frozen_result = self._session.finish(
                raise_on_failure=False, join_timeout=join_timeout,
            )
            self._dynamic_primary_error = self._session._current_failure()
        return self._dynamic_frozen_result

    def _mark_dynamic_failure(self, error: BaseException) -> ReductionResult:
        if self._dynamic_primary_error is None:
            self._dynamic_primary_error = error
        result = self._dynamic_frozen_result
        if result is None:
            raise RuntimeError("dynamic terminal result was not frozen")
        if not result.failed:
            result = replace(result, failed=True, error=str(error))
            self._dynamic_frozen_result = result
        return result

    def _settle_dynamic_graph(self, result, *, failed):
        if self._dynamic_graph_terminal_settled:
            return self._terminal_result
        sink = self._user_sink
        if sink is None:
            value = None
        elif failed:
            abort = getattr(sink, "abort", None)
            value = (abort if callable(abort) else sink.finish)(result)
        else:
            value = sink.finish(result)
        if self._dynamic_nexus_sink is not None:
            if type(value) is not NexusTerminalResult:
                raise RuntimeError("dynamic Nexus graph returned no typed terminal")
            self._terminal_result = value
        elif value is not None:
            raise RuntimeError("dynamic Memory graph returned Nexus terminal truth")
        self._dynamic_graph_terminal_settled = True
        return self._terminal_result

    def _finish_dynamic(
        self, *, raise_on_failure: bool, join_timeout: float | None,
    ) -> ReductionResult:
        result = self._freeze_dynamic_result(join_timeout)
        if not self._session.sink_terminal_safe:
            raise self._dynamic_primary_error or RuntimeError(
                "dynamic writer remains able to call the sink"
            )
        if self._dynamic_terminal_settled:
            if raise_on_failure and result.failed:
                error = self._dynamic_primary_error
                if error is not None:
                    raise error
            return result
        boundary = self._dynamic_boundary
        owner = self._dynamic_owner_token
        failed = bool(result.failed)
        stopped = bool(self._dynamic_stop_requested or result.cancelled)

        if not failed and self._dynamic_finish_seal is None:
            try:
                self._event_sink.flush(force=True)
                self._dynamic_finish_seal = boundary.prepare_session_finish(
                    self, owner, stopped=stopped,
                )
            except BaseException as error:
                result = self._mark_dynamic_failure(error)
                failed = True

        if failed:
            value = self._settle_dynamic_graph(result, failed=True)
            if value is not None and value.disposition is NexusTerminalDisposition.COMMITTED:
                if self._dynamic_finish_seal is None:
                    raise RuntimeError("committed dynamic failure has no finish seal")
                boundary.session_finished(
                    self, owner, self._dynamic_finish_seal, value.commit_identity,
                )
            else:
                if value is not None and value.disposition is not NexusTerminalDisposition.ABORTED:
                    raise RuntimeError("dynamic abort returned contradictory terminal truth")
                boundary.epoch_aborted(
                    self, owner,
                    str(self._dynamic_primary_error or "dynamic run aborted"),
                )
            self._dynamic_terminal_settled = True
        else:
            value = self._settle_dynamic_graph(result, failed=False)
            if value is None:
                if stopped:
                    boundary.session_stopped(
                        self, owner, self._dynamic_finish_seal,
                        "dynamic session stopped without a Nexus transaction",
                    )
                else:
                    boundary.session_finished(
                        self, owner, self._dynamic_finish_seal, owner,
                    )
            elif value.disposition is NexusTerminalDisposition.COMMITTED:
                boundary.session_finished(
                    self, owner, self._dynamic_finish_seal, value.commit_identity,
                )
            elif stopped and value.disposition is NexusTerminalDisposition.ABORTED:
                boundary.session_stopped(
                    self, owner, self._dynamic_finish_seal,
                    "dynamic session stopped before a canonical prefix",
                )
            else:
                error = RuntimeError("dynamic finish resolved to ABORTED")
                self._mark_dynamic_failure(error)
                boundary.epoch_aborted(self, owner, str(error))
            self._dynamic_terminal_settled = True

        if self._record_store is not None:
            with self._projection_lock:
                self._reconcile_locked(self._record_store.labels())
        self._final_sweep()
        if not self._terminal_state_emitted:
            self._terminal_state_emitted = True
            self._emit_state()
        result = self._dynamic_frozen_result
        if raise_on_failure and result.failed:
            error = self._dynamic_primary_error
            if error is not None:
                raise error
        return result

    def commit_epoch(self):
        """Commit one dynamic H23 epoch while retaining this exact session."""
        if self._dynamic_accounting is None or self._dynamic_nexus_sink is None:
            raise RuntimeError("commit_epoch requires one dynamic Nexus owner")
        if self._dynamic_frozen_result is not None or self._dynamic_stop_requested:
            raise RuntimeError("terminal dynamic session cannot commit another epoch")
        if self._dynamic_epoch_notified:
            return self._dynamic_settled_epoch_anchor
        if self._dynamic_epoch_seal is None:
            if not self._session.drain():
                raise RuntimeError("dynamic epoch writer did not drain")
            failure = self._session._current_failure()
            if failure is not None:
                raise failure
            self._event_sink.flush(force=True)
            self._dynamic_epoch_seal = self._dynamic_boundary.prepare_epoch_commit(
                self, self._dynamic_owner_token,
            )
        if self._dynamic_epoch_anchor is None:
            current = ReductionResult(
                self.scan.name, {}, self._session._completed,
            )
            self._dynamic_epoch_anchor = self._dynamic_nexus_sink.commit_epoch(current)
        anchor = self._dynamic_epoch_anchor
        self._dynamic_boundary.epoch_committed(
            self, self._dynamic_owner_token, self._dynamic_epoch_seal, anchor,
        )
        self._dynamic_settled_epoch_anchor = anchor
        self._dynamic_epoch_seal = None
        self._dynamic_epoch_anchor = None
        self._dynamic_epoch_notified = True
        if self._record_store is not None:
            with self._projection_lock:
                self._reconcile_locked(self._record_store.labels())
        return anchor

    def extend_live(self, intent: AppendIntent) -> AppendDecision:
        """Continue this exact committed same-run output lineage."""
        if type(intent) is not AppendIntent:
            raise TypeError("extend_live requires an exact AppendIntent")
        extend_live = self._dynamic_extend_live
        owner = self._dynamic_extension_owner
        current_intent = self._dynamic_current_intent
        if extend_live is None or owner is None or current_intent is None:
            raise RuntimeError(
                "session has no captured same-run continuation capability"
            )
        if self._dynamic_settled_epoch_anchor is None:
            raise RuntimeError(
                "same-run continuation requires one settled committed epoch"
            )
        if (
            self._dynamic_epoch_seal is not None
            or self._dynamic_finish_seal is not None
            or self._dynamic_frozen_result is not None
            or self._dynamic_terminal_settled
            or self._dynamic_primary_error is not None
        ):
            raise RuntimeError("same-run continuation has pending terminal work")
        if self._dynamic_submit_token is not None:
            raise RuntimeError("same-run continuation cannot overlap submission")
        if (
            self._dynamic_stop_requested
            or self._session.cancel_token.cancelled
            or self._session.is_paused
            or not self._session.is_running
        ):
            raise RuntimeError("same-run continuation requires an active session")
        if self._session._current_failure() is not None:
            raise RuntimeError("same-run continuation refuses a failed session")
        if not self._session.drain(timeout=0.0):
            raise RuntimeError("same-run continuation requires a drained writer")
        snapshot = self._dynamic_accounting.snapshot()
        latest_states = tuple(
            snapshot.attempt_states[tokens[-1]]
            for tokens in snapshot.attempts.values() if tokens
        )
        if (
            snapshot.state is not DynamicRunState.ACTIVE
            or snapshot.in_flight
            or snapshot.retry_owned
            or any(state in {
                DynamicAttemptState.FAILED,
                DynamicAttemptState.FAILED_RETRYABLE,
                DynamicAttemptState.CANCELLED,
            } for state in latest_states)
        ):
            raise RuntimeError("same-run continuation refuses a dirty epoch")
        replay = intent == current_intent
        if not replay and not self._dynamic_epoch_notified:
            raise RuntimeError(
                "a different successor requires a newly committed epoch"
            )
        decision = extend_live(owner, intent)
        if not replay:
            self._dynamic_current_intent = intent
            self._dynamic_epoch_notified = False
        return decision

    def _final_sweep(self) -> None:
        store = self._record_store
        with self._projection_lock:
            if self._swept:
                return
            self._swept = True
            if store is None:
                return
            for label in store.labels():
                try:
                    if self._fenced_locked(int(label)):
                        continue        # a blocked pair keeps its heavy data
                    store.release_heavy(label)
                except Exception:
                    logger.exception(
                        "ScanSession final sweep failed for label %r", label)
    def flush(self, *, force: bool = False) -> None:
        """Contract pass-through to the sink's optional ``flush`` hook (ADR-0004
        §4).  No-op for a sink without one."""
        self._event_sink.flush(force=force)

    def set_generation(self, generation: int) -> None:
        """Set the stale-render stamp put on subsequent events (ADR-0004 §2).
        Caller-owned; the session never auto-advances it (esp. not on
        pause/resume)."""
        with self._lock:
            self._generation = int(generation)

    # -- state out ---------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._session.is_running

    @property
    def is_paused(self) -> bool:
        return self._session.is_paused

    @property
    def frames_submitted(self) -> int:
        """DERIVED (§4.1): the number of DISTINCT accepted label identities.
        A re-feed of a known label cannot increase it."""
        return self._accounting.accepted_label_count()

    @property
    def frames_completed(self) -> int:
        """DERIVED (§4.1): the number of DISTINCT labels whose top-level sink
        hook returned successfully at least once during the run.  Monotonic —
        a later replacement whose write fails cannot decrement it — and a
        re-feed cannot inflate it."""
        return self._accounting.written_label_count()

    @property
    def saturation_mask_seeded(self) -> bool:
        return self._session.saturation_mask_seeded

    @property
    def saturation_mask(self) -> np.ndarray | None:
        return self._session.saturation_mask

    @property
    def scan(self):
        """The underlying session's scan (frame inventory / context)."""
        return self._session.scan

    # -- events out --------------------------------------------------------
    # Each registration returns an UNSUBSCRIBE callable so a notebook / the Qt
    # bridge / a remote client can detach without tearing down the session
    # (append-only listeners would otherwise leak across re-subscribes).  The
    # handle is idempotent — calling it twice is a no-op.
    def on_frame_completed(self, cb: Callable[[FrameEvent], None]) -> Callable[[], None]:
        return self._subscribe(self._frame_cbs, cb)

    def on_progress(self, cb: Callable[[ProgressEvent], None]) -> Callable[[], None]:
        return self._subscribe(self._progress_cbs, cb)

    def on_state_change(self, cb: Callable[[StateChangeEvent], None]) -> Callable[[], None]:
        return self._subscribe(self._state_cbs, cb)

    def _subscribe(self, registry: list, cb: Callable) -> Callable[[], None]:
        with self._lock:
            registry.append(cb)

        def _unsubscribe() -> None:
            with self._lock:
                try:
                    registry.remove(cb)
                except ValueError:
                    pass            # already removed / never present — idempotent
        return _unsubscribe

    # -- internals ---------------------------------------------------------
    _OUTCOME_DISPOSITIONS = {
        FrameOutcome.COMPLETED: ItemDisposition.COMPLETED,
        FrameOutcome.FAILED: ItemDisposition.FAILED,
        FrameOutcome.CANCELLED_BEFORE_COMPLETION:
            ItemDisposition.CANCELLED_BEFORE_COMPLETION,
    }

    def _on_accepted(self, frame: Frame, publish_acceptance: Callable[[int], None]) -> int:
        """Caller-thread acceptance admission (the engine's ``accept_cb``).

        Invoked inside ``ReductionSession.submit`` in publication order — Future
        bound, the same undecided ticket queued, inventory staged, THEN this
        authority runs, ACCEPTED opens the worker and writer — so
        the new attempt's accepted/PENDING state is ledger-visible before any
        outcome, sink write or public completion callback for it can run.
        """
        dynamic = self._dynamic_accounting
        subject = int(frame.index)
        publisher = publish_acceptance
        if dynamic is not None:
            token = self._dynamic_submit_token
            if token is None:
                raise RuntimeError("dynamic acceptance lost its exact submit token")
            def publish_dynamic(ledger_attempt: int) -> None:
                self._dynamic_epoch_anchor = None
                self._dynamic_epoch_notified = False
                publish_acceptance(ledger_attempt)
            subject = token
            publisher = publish_dynamic
        authority = dynamic if dynamic is not None else self._accounting
        return authority.record_accepted(
            subject, publish_acceptance=publisher,
        )

    def record_persisted(self, receipts: Iterable[StageReceipt]) -> None:
        if self._dynamic_accounting is not None:
            raise RuntimeError(
                "dynamic persisted truth is owned only by the writer boundary"
            )
        batch = tuple(receipts)
        with self._projection_lock:
            self._accounting.record_persisted(batch)
            self._reconcile_locked({int(r.label) for r in batch})

    def record_durable(self, receipts: Iterable[StageReceipt]) -> None:
        if self._dynamic_accounting is not None:
            raise RuntimeError(
                "dynamic durable truth is owned only by the writer boundary"
            )
        batch = tuple(receipts)
        with self._projection_lock:
            self._accounting.record_durable(batch)
            self._reconcile_locked({int(r.label) for r in batch})

    def record_publication_dropped(self, label: int, mode: ResultMode, *,
                                   expected_revision: int) -> None:
        if self._dynamic_accounting is not None:
            raise RuntimeError(
                "dynamic publication drop is owned only by the writer boundary"
            )
        with self._projection_lock:
            self._accounting.record_publication_dropped(
                label, mode, expected_revision=expected_revision)
            self._reconcile_locked({int(label)})

    def _fenced_locked(self, label: int) -> set[ResultMode]:
        return (self._blocked.get(label, set()) | self._pending.get(label, set()))

    def _reconcile_locked(self, labels: Iterable[int]) -> None:
        """Publish each label's COMPLETE projection in ONE atomic replacement."""
        store = self._record_store
        if store is None:
            return
        snapshot = self._accounting.snapshot()
        for label in labels:
            fenced = self._fenced_locked(label)
            hydratable: list[tuple[str, str]] = []
            durable: list[tuple[str, str]] = []
            dropped: list[tuple[str, str]] = []
            for mode in self._accounting.required_modes:
                if mode in fenced:
                    continue
                key = (mode.kind, mode.key)
                if (label, mode) in snapshot.publication_dropped:
                    dropped.append(key)
                    continue
                applies = snapshot.targets_by_mode.get(mode, frozenset())
                if any((label, mode, target) in snapshot.persisted
                       for target in self._store_targets.get(mode, applies)):
                    hydratable.append(key)          # this store can recover it
                if applies and all((label, mode, target) in snapshot.durable
                                   for target in applies):
                    durable.append(key)             # EVERY applicable target
            try:
                store.replace_projection(label, hydratable=hydratable,
                                         durable=durable, dropped=dropped)
            except Exception:
                # Fail toward retention: re-fence what we could not publish.
                logger.exception("ScanSession store projection publish failed")
                self._blocked.setdefault(label, set()).update(
                    self._accounting.required_modes)
    def _clear_projection_locked(self, label: int,
                                 modes: Iterable[ResultMode]) -> None:
        modes = tuple(modes)
        if not modes:
            return
        self._blocked.setdefault(label, set()).update(modes)
        self._reconcile_locked((label,))
    def _settle_locked(self, label: int) -> None:
        settled = self._pending.pop(label, set())
        blocked = self._blocked.get(label)
        if blocked is not None:
            blocked -= settled
            if not blocked:
                self._blocked.pop(label, None)
    def _on_outcome(self, receipt: FrameOutcomeReceipt) -> None:
        """Writer-thread per-item compute outcome → the stage ledger.  A
        COMPLETED outcome mints new result revisions for exactly the produced
        modes; FAILED/CANCELLED_BEFORE_COMPLETION are typed terminal
        dispositions with no revision (prior durable state stays truthful)."""
        mode_1d, mode_2d = _dimension_modes(self._mode_key)
        produced: list[ResultMode] = []
        if receipt.outcome is FrameOutcome.COMPLETED:
            if receipt.produced_1d:
                produced.append(ResultMode.one_d(mode_1d))
            if receipt.produced_2d:
                produced.append(ResultMode.two_d(mode_2d))
        label = int(receipt.frame_index)
        with self._projection_lock:
            affected = tuple(produced) or tuple(
                mode for mode in self._accounting.required_modes
                if self._accounting.current_revision(label, mode) >= 1)
            before = {mode: self._accounting.current_revision(label, mode)
                      for mode in affected}
            self._clear_projection_locked(label, affected)
            self._accounting.record_outcome(
                label,
                self._OUTCOME_DISPOSITIONS[receipt.outcome],
                produced=produced,
                error=receipt.error,
                attempt=receipt.attempt,
            )
            minted = {mode for mode in affected
                      if self._accounting.current_revision(label, mode)
                      != before[mode]}
            if minted:
                self._pending.setdefault(label, set()).update(minted)
            unchanged = set(affected) - minted
            if unchanged:
                blocked = self._blocked.get(label)
                if blocked is not None:
                    blocked -= unchanged
                    if not blocked:
                        self._blocked.pop(label, None)
                self._reconcile_locked((label,))

    def _produced_modes(self, receipt: FrameOutcomeReceipt) -> tuple[ResultMode, ...]:
        mode_1d, mode_2d = _dimension_modes(self._mode_key)
        modes = []
        if receipt.produced_1d:
            modes.append(ResultMode.one_d(mode_1d))
        if receipt.produced_2d:
            modes.append(ResultMode.two_d(mode_2d))
        return tuple(modes)

    def _on_dynamic_outcome(self, receipt: FrameOutcomeReceipt) -> None:
        dynamic = self._dynamic_accounting
        if dynamic is None or receipt.attempt is None:
            raise RuntimeError("dynamic outcome has no exact ledger attempt")
        token = dynamic.token_for_ledger_attempt(
            int(receipt.frame_index), receipt.attempt,
        )
        produced = self._produced_modes(receipt)
        label = int(receipt.frame_index)
        with self._projection_lock:
            affected = produced or tuple(
                mode for mode in self._accounting.required_modes
                if self._accounting.current_revision(label, mode) >= 1
            )
            self._clear_projection_locked(label, affected)
            if receipt.outcome is FrameOutcome.COMPLETED:
                dynamic.record_completed(token, produced=produced)
                if produced:
                    self._pending.setdefault(label, set()).update(produced)
            elif receipt.outcome is FrameOutcome.FAILED:
                dynamic.record_failed(
                    token, error=receipt.error or "dynamic reduction failed",
                    retryable=not self._dynamic_stop_requested,
                )
            else:
                dynamic.record_cancelled(
                    token, reason=receipt.error or "dynamic reduction cancelled",
                )

    def _on_dynamic_written(
        self, frame: Frame, reduction: Any, ledger_attempt: int | None,
    ) -> None:
        dynamic = self._dynamic_accounting
        if dynamic is None or ledger_attempt is None:
            raise RuntimeError("dynamic write has no exact ledger attempt")
        token = dynamic.token_for_ledger_attempt(int(frame.index), ledger_attempt)
        mode_1d, mode_2d = _dimension_modes(self._mode_key)
        modes = []
        if getattr(reduction, "result_1d", None) is not None:
            modes.append(ResultMode.one_d(mode_1d))
        if getattr(reduction, "result_2d", None) is not None:
            modes.append(ResultMode.two_d(mode_2d))
        dynamic.record_written(token, modes=modes)

    def _record_written(self, event: FrameEvent) -> None:
        """The TOP-LEVEL sink hook returned successfully for this event.

        Certifies the event's exact modes and records the label in the
        historical write identity behind ``frames_completed`` — so it runs even
        for a result with no modes.  Guarded like the record-store upsert: an
        accounting error is logged, never allowed to turn a successful write
        into a run failure."""
        mode_1d, mode_2d = _dimension_modes(event.mode_key)
        modes: list[ResultMode] = []
        if event.result_1d is not None:
            modes.append(ResultMode.one_d(mode_1d))
        if event.result_2d is not None:
            modes.append(ResultMode.two_d(mode_2d))
        try:
            self._accounting.record_written(event.frame_index, modes)
        except Exception:
            logger.exception("ScanSession stage-accounting record_written failed")

    def _on_completed(self, frame: Frame, reduction: Any) -> None:
        """Writer-thread completion hook (called by _EventSink after the sink
        write).  Builds the immutable FrameEvent + advances completion progress.
        A listener exception is caught — it must never escape the writer loop."""
        with self._lock:
            generation = self._generation
            cbs = tuple(self._frame_cbs)
        event = FrameEvent(
            frame_index=int(getattr(reduction, "frame_index", getattr(frame, "index", -1))),
            mode_key=self._mode_key,
            # Freeze the shared result arrays read-only + the metadata into a
            # read-only view, so a listener can't retroactively corrupt the
            # already-written/cached data (the event is the bridge's sole data
            # contract — it must be tamper-evident).  Both are zero-copy.
            result_1d=_freeze_result_arrays(getattr(reduction, "result_1d", None)),
            result_2d=_freeze_result_arrays(getattr(reduction, "result_2d", None)),
            metadata=MappingProxyType(dict(getattr(reduction, "metadata", {}) or {})),
            generation=generation,
            timestamp=time.time(),
        )
        if self._dynamic_accounting is None:
            self._record_written(event)
        started = time.perf_counter() if self._perf_enabled else 0.0
        self._upsert_record_store(frame, event)
        self._perf_add("session_record_upsert", started)
        started = time.perf_counter() if self._perf_enabled else 0.0
        for cb in cbs:
            try:
                cb(event)
            except Exception:
                logger.exception("ScanSession.on_frame_completed listener raised")
        self._perf_add("session_frame_listeners", started)
        started = time.perf_counter() if self._perf_enabled else 0.0
        self._emit_progress()
        self._perf_add("session_progress_listeners", started)

    def _perf_add(self, key: str, started: float) -> None:
        if not self._perf_enabled:
            return
        self._perf_values[key] = self._perf_values.get(key, 0.0) + max(
            0.0, time.perf_counter() - started,
        )

    def perf_snapshot(self) -> dict[str, float]:
        """Return writer-side timings without exposing mutable state."""
        values = dict(self._perf_values)
        snapshot = getattr(self._user_sink, "perf_snapshot", None)
        if callable(snapshot):
            values.update(snapshot())
        return values

    def _mint_write_receipts_locked(self, label: int) -> None:
        """The write IS the durability boundary: EXACTLY one declared target/mode."""
        if not self._record_store_persisted_on_write:
            return
        receipts = [self._accounting.receipt(label, mode, target)
                    for mode, targets in self._write_targets.items()
                    for target in targets
                    if self._accounting.current_revision(label, mode) >= 1]
        if receipts:
            self._accounting.record_durable(receipts)
    def _upsert_record_store(self, frame: Frame, event: FrameEvent) -> None:
        if self._record_store is None:
            return
        mode_1d, mode_2d = _dimension_modes(event.mode_key)
        try:
            view = FrameView.from_results(
                label=event.frame_index,
                result_1d=event.result_1d,
                result_2d=event.result_2d,
                metadata_raw=event.metadata,
                metadata_numeric=getattr(frame, "metadata_numeric", None),
                incident_angle=getattr(getattr(frame, "geometry", None), "incident_angle", None),
                source_path=getattr(frame, "source_path", None),
                source_frame_index=getattr(frame, "source_frame_index", None),
            )
            record = FrameRecord.from_view(view, mode_1d=mode_1d, mode_2d=mode_2d)
        except Exception:
            logger.exception("ScanSession record_store view build failed")
            return
        label = int(event.frame_index)
        with self._projection_lock:
            try:
                self._record_store.upsert(
                    record,
                    source_identity=getattr(frame, "source_identity", None),
                )
            except Exception:
                logger.exception("ScanSession record_store upsert failed")
                return
            self._settle_locked(label)
            try:
                self._mint_write_receipts_locked(label)
            except Exception:
                logger.exception("ScanSession write-boundary receipts failed")
            self._reconcile_locked((label,))

    def _emit_progress(self) -> None:
        # Both totals are DERIVED identity projections (§4.1), read straight
        # from the ledger: absolute, monotonic, and immune to re-feed inflation.
        submitted = self._accounting.accepted_label_count()
        completed = self._accounting.written_label_count()
        with self._lock:
            cbs = tuple(self._progress_cbs)
        try:
            total = len(self._session.scan)
        except Exception:
            total = None
        event = ProgressEvent(submitted=submitted, completed=completed, total=total)
        for cb in cbs:
            try:
                cb(event)
            except Exception:
                logger.exception("ScanSession.on_progress listener raised")

    def _emit_state(self) -> None:
        event = StateChangeEvent(is_running=self.is_running, is_paused=self.is_paused)
        with self._lock:
            cbs = tuple(self._state_cbs)
        for cb in cbs:
            try:
                cb(event)
            except Exception:
                logger.exception("ScanSession.on_state_change listener raised")
