"""One bounded typed hydration transport for the vNext scattering display.

The single owner of background preview reads (E4-R): one active read, one
latest queued reference, worker-thread execution through the shared one-open
:func:`read_frame_preview`, and one terminal :class:`HydrationCompletion` for
every admitted token.  It replaces the deleted private flight protocol (whose
old module name may not appear here — the census oracle scans this package).
The queued reference carries only the typed ``HydrationRequest`` (owner, exact
stores, gate, read key, token), the frozen values-only projection derived ONCE
at submit from the exact carried target, presentation facts, and the resolved
catalog key — never arrays, open handles, widgets or callbacks.  The read
result is consumed only by the ONE owner-bound target port
(``RunDisplayState.commit_preview``), which validates under its own lock plus
the exact request-carried ``CommitGate`` and publishes last; a commit
exception retains that exact prepared commit for one idempotent retry.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from threading import Lock, RLock, Thread, current_thread

from xdart.modules.display_context import (
    HydrationRequest,
    Prepared2DCatalogCommit,
    Prepared2DFrameCommit,
    Viewer2DCatalogHydrationRequest,
    Viewer2DCleanupReceipt,
    Viewer2DCleanupState,
    Viewer2DDisposal,
    Viewer2DDisposalToken,
    Viewer2DFrameHydrationRequest,
    Viewer2DReadActivation,
    Viewer2DReceiptPhase,
)
from xrd_tools.session.viewer_1d import (
    VIEWER_1D_R,
    Viewer1DBatch,
    Viewer1DBatchHydrationRequest,
    Viewer1DCleanupPendingNotice,
    Viewer1DCommitReceipt,
    Viewer1DDisposal,
    Viewer1DReadFailure,
    Viewer1DReadBudgetState,
    Viewer1DReadOperation,
    _Viewer1DReaderControl,
    _mint_viewer_1d_budget_receipt,
    _new_prepared_viewer_1d_commit,
    _new_viewer_1d_cleanup_notice,
    _retire_viewer_1d_budget_receipt,
    begin_viewer_1d_read,
    mint_viewer_1d_pass_two_permit,
    mint_viewer_1d_transfer,
    viewer_1d_disposal_is_current,
    viewer_1d_request_is_canonical,
    viewer_1d_transfer_is_released,
    viewer_1d_budget,
)
from xrd_tools.io.viewer_2d import (
    CATALOG_RESERVATION,
    catalog_viewer_2d,
    read_viewer_2d_frame,
    viewer_2d_memory_ledger,
    viewer_2d_selected_ledger,
)
from xrd_tools.io.frame_preview import (
    DetectorPreviewProjection,
    FramePreview,
    read_frame_preview,
)
from xrd_tools.session.hydration import (
    HydrationCompletion,
    HydrationOutcome,
    HydrationToken,
)

from .display_values import DisplayFrameKey

_VIEWER_2D_REQUEST_TYPES = (Viewer2DCatalogHydrationRequest, Viewer2DFrameHydrationRequest)
_VIEWER_1D_REQUEST_TYPES = (Viewer1DBatchHydrationRequest,)
_VIEWER_REQUEST_TYPES = (*_VIEWER_2D_REQUEST_TYPES, *_VIEWER_1D_REQUEST_TYPES)


def _viewer_activation_matches(request, activation):
    try:
        if type(activation) is not Viewer2DReadActivation:
            return False
        request.__post_init__()
        activation.__post_init__()
        receipt = activation.receipt
        if type(request) is Viewer2DCatalogHydrationRequest:
            return (activation.token is request.token
                and activation.phase is Viewer2DReceiptPhase.CATALOG_R
                and receipt.capacity == viewer_2d_memory_ledger(1, 1).budget
                and receipt.reserved == CATALOG_RESERVATION)
        ledger = viewer_2d_selected_ledger(request.catalog, request.label)
        return (activation.token is request.token
            and activation.phase is Viewer2DReceiptPhase.FRAME_A
            and receipt.identity is request.receipt_identity
            and receipt.capacity == ledger.budget
            and receipt.reserved == ledger.admission)
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return False


class _HydrationTicket:
    """One admission's terminal fact.  Tokens are VALUES, so ticket OBJECT
    IDENTITY separates admissions and survives diagnostic-deque rotation."""

    __slots__ = ("_lock", "_token", "_completion")

    def __init__(self, token: HydrationToken) -> None:
        if type(token) is not HydrationToken:
            raise TypeError("an admission ticket requires one exact token")
        self._lock = Lock()
        self._token = token
        self._completion: HydrationCompletion | None = None

    @property
    def token(self) -> HydrationToken:
        return self._token

    def result(self) -> HydrationCompletion | None:
        with self._lock:
            return self._completion

    def _settle(self, completion: HydrationCompletion) -> bool:
        """First-wins settlement; returns whether THIS call settled it."""
        if (
            type(completion) is not HydrationCompletion
            or completion.token != self._token
        ):
            return False
        with self._lock:
            if self._completion is not None:
                return False
            self._completion = completion
            return True


class _EntryState(str, Enum):
    READING = "reading"
    PREPARED = "prepared"
    DELIVERY_PENDING = "delivery_pending"
    CLEANUP_BLOCKED = "cleanup_blocked"


class HydrationTransportCleanupToken:
    __slots__ = ()


class _DeliveryGuard:
    __slots__ = ("claimed",)

    def __init__(self): self.claimed = False


@dataclass(frozen=True, slots=True)
class HydrationTransportCleanupReceipt:
    token: object
    state: Viewer2DCleanupState


@dataclass(slots=True)
class _TransportEntry:
    """The sole active-or-latest request and any unsettled custody."""

    request: HydrationRequest
    projection: DetectorPreviewProjection | None
    key: DisplayFrameKey | None
    closed: bool
    token: HydrationToken
    ticket: _HydrationTicket
    committed_token: HydrationToken | None = None
    committed_ticket: _HydrationTicket | None = None
    state: _EntryState = _EntryState.READING
    disposal: object | None = None
    viewer_1d_receipt: object | None = None
    viewer_1d_transfer: object | None = None
    cleanup_token: HydrationTransportCleanupToken | None = None
    terminal: tuple | None = None
    retrying: bool = False
    delivery_guard: _DeliveryGuard | None = None


@dataclass(frozen=True, slots=True)
class _DetachedDelivery:
    entry: _TransportEntry
    completion: HydrationCompletion
    guard: _DeliveryGuard
    notify: bool
    active_slot: bool
    clear: bool
    restore: _EntryState | None = None


@dataclass(frozen=True, slots=True)
class DetachedHydrationMutation:
    ticket: _HydrationTicket | None = None
    deliveries: tuple[_DetachedDelivery, ...] = ()
    replacement: _TransportEntry | None = None

    @property
    def token(self):
        return None if self.ticket is None else self.ticket.token


@dataclass(frozen=True, slots=True)
class _Viewer1DExecutionTerminal:
    outcome: HydrationOutcome
    diagnostic: str | None
    control: BaseException | None = None


@dataclass(frozen=True, slots=True)
class _Viewer1DBlockedTerminal:
    outcome: HydrationOutcome
    diagnostic: str | None
    notice: Viewer1DCleanupPendingNotice


class _Viewer1DUnwind(BaseException):
    def __init__(self, control, entry=None, guard=None):
        self.control, self.entry, self.guard = control, entry, guard


def _viewer_1d_cleanup_current(entry):
    disposal = entry.disposal
    return (type(disposal) is Viewer1DDisposal
        and entry.viewer_1d_transfer is disposal.transfer
        and (viewer_1d_disposal_is_current(disposal)
             or viewer_1d_transfer_is_released(disposal.transfer)))


@dataclass(frozen=True, slots=True)
class PreparedHydrationCommit:
    """The one frozen value the target-port operation may consume."""

    request: HydrationRequest
    token: HydrationToken
    key: DisplayFrameKey | None
    closed: bool
    preview: FramePreview
    projection: DetectorPreviewProjection | None


class HydrationTransport:
    """Externally observable, internally synchronized single-lane transport."""

    __slots__ = ("_viewer_1d_claim", "__dict__")

    def __init__(self, commit, derive, *, completion_sink=None) -> None:
        # commit/derive are owner-bound at construction, never queue-carried.
        self._commit = commit
        self._derive = derive
        self._completion_sink = completion_sink
        self._lock = RLock()
        self._viewer_1d_claim = object()
        self._active: _TransportEntry | None = None
        self._queued: _TransportEntry | None = None
        self._worker: Thread | None = None
        self._retired = False
        self._counters: dict[HydrationOutcome, int] = {
            outcome: 0 for outcome in HydrationOutcome
        }
        self._completions: deque[HydrationCompletion] = deque(maxlen=32)

    # -- observability ------------------------------------------------------ #

    @property
    def active_token(self) -> HydrationToken | None:
        with self._lock:
            return self._active.token if self._active is not None else None

    @property
    def queued_token(self) -> HydrationToken | None:
        with self._lock:
            return self._queued.token if self._queued is not None else None

    @property
    def worker(self) -> Thread | None:
        with self._lock:
            return self._worker

    def completions(self) -> tuple[HydrationCompletion, ...]:
        with self._lock:
            return tuple(self._completions)

    def counters(self) -> dict[HydrationOutcome, int]:
        with self._lock:
            return dict(self._counters)

    # -- admission ---------------------------------------------------------- #

    def submit(
        self, request, *, closed: bool = False
    ) -> HydrationToken | None:
        """Admit one typed request, or refuse with ``None`` (tokens only)."""
        ticket = self._submit_admission(request, closed=closed)
        return None if ticket is None else ticket.token

    def _submit_admission(
        self, request, *, closed: bool = False
    ) -> _HydrationTicket | None:
        mutation = self.submit_detached(request, closed=closed)
        self.dispatch_detached(mutation)
        return mutation.ticket

    def submit_detached(self, request, *, closed: bool = False):
        if type(request) is Viewer1DBatchHydrationRequest:
            if not viewer_1d_request_is_canonical(request, self):
                return DetachedHydrationMutation()
        elif (type(request) not in (HydrationRequest, *_VIEWER_REQUEST_TYPES)
                or getattr(request, "read_key", None) is None
                or request.token is None or not request.enqueueable):
            return DetachedHydrationMutation()
        token = request.token
        with self._lock:
            if (self._retired or request.commit_gate.cancelled
                    or type(request) is Viewer1DBatchHydrationRequest
                    and request.admitted_provider_identity is not self
                    or any(entry is not None and entry.state is _EntryState.DELIVERY_PENDING
                           for entry in (self._active, self._queued))):
                return DetachedHydrationMutation()
            active = self._active
            if (active is not None and active.state is _EntryState.CLEANUP_BLOCKED
                    and type(active.request) is Viewer1DBatchHydrationRequest):
                prior = active.request
                terminal = active.terminal
                if (not _viewer_1d_cleanup_current(active)
                        or active.cleanup_token is None
                        or type(terminal) is not _Viewer1DBlockedTerminal
                        or terminal.notice.request is not prior
                        or terminal.notice.transport_token is not active.token
                        or type(request) is not Viewer1DBatchHydrationRequest
                        or request.owner_identity is not prior.owner_identity
                        or request.owner_request_claim is not prior.owner_request_claim
                        or request.port is not prior.port
                        or request.commit_gate is not prior.commit_gate
                        or request.admitted_provider_identity is not prior.admitted_provider_identity
                        or request.generation <= prior.generation):
                    return DetachedHydrationMutation()
            if (
                type(request) is HydrationRequest
                and
                active is not None
                and type(active.request) is HydrationRequest
                and active.request.read_key == request.read_key
            ):
                displaced = active.token
                displaced_ticket = active.ticket
                active.closed = bool(closed)
                deliveries = []
                if displaced != token:
                    active.token = token
                    active.ticket = _HydrationTicket(token)
                    deliveries.append(self._capture_locked(
                        active, displaced, HydrationOutcome.SUPERSEDED,
                        displaced_ticket, clear=False, restore=_EntryState.READING))
                # An equal token represents the same in-flight presentation.
                # Keep its identity: _execute checks that exact object before
                # committing the read already in progress.
                queued = self._queued
                if queued is not None:
                    deliveries.append(self._capture_locked(
                        queued, queued.token, HydrationOutcome.SUPERSEDED,
                        queued.ticket, clear=True))
                return DetachedHydrationMutation(active.ticket, tuple(deliveries))
            projection, key = ((None, None) if type(request) is not HydrationRequest
                               else self._derive(request))
            entry = _TransportEntry(
                request, projection, key, bool(closed), token,
                _HydrationTicket(token),
            )
            queued = self._queued
            if queued is not None:
                delivery = self._capture_locked(
                    queued, queued.token, HydrationOutcome.SUPERSEDED,
                    queued.ticket, clear=True)
                return DetachedHydrationMutation(entry.ticket, (delivery,), entry)
            self._queued = entry
            return DetachedHydrationMutation(entry.ticket)

    def dispatch_detached(self, mutation) -> None:
        if type(mutation) is not DetachedHydrationMutation:
            return
        pending = mutation
        control = None
        while True:
            while pending.deliveries:
                with self._lock:
                    deliveries = tuple(delivery for delivery in pending.deliveries
                        if not self._delivery_finalized_locked(delivery))
                    if any(not self._delivery_valid_locked(delivery, claimed=False)
                           for delivery in deliveries):
                        return
                    for delivery in deliveries:
                        delivery.guard.claimed = True
                for delivery in deliveries:
                    if not delivery.notify:
                        continue
                    try:
                        request = delivery.entry.request
                        if type(request) in _VIEWER_REQUEST_TYPES:
                            request.port.complete(delivery.completion)
                        elif self._completion_sink is not None:
                            self._completion_sink(delivery.completion)
                    except BaseException as error:
                        if type(request) is Viewer1DBatchHydrationRequest:
                            if (_viewer_1d_control(error) and control is None): control = error
                        elif not isinstance(error, Exception):
                            raise
                with self._lock:
                    if any(not self._delivery_valid_locked(delivery, claimed=True)
                           for delivery in deliveries):
                        return
                    for delivery in deliveries:
                        entry = delivery.entry
                        entry.delivery_guard = None
                        if delivery.clear:
                            if self._active is entry: self._active = None
                            if self._queued is entry: self._queued = None
                        elif delivery.restore is not None:
                            entry.state = delivery.restore
                            if (delivery.active_slot and self._worker is None
                                    and delivery.restore is _EntryState.READING):
                                self._active = None
                                self._queued = entry
                    replacement = pending.replacement
                    if replacement is not None and not self._retired:
                        if self._queued is None and self._active is not replacement:
                            self._queued = replacement
                        elif self._queued is not replacement and self._active is not replacement:
                            return
                pending = DetachedHydrationMutation()
            with self._lock:
                thread = self._ensure_worker_locked()
            if thread is None:
                if control is not None: raise control
                return
            try:
                thread.start()
            except BaseException as error:
                with self._lock:
                    one_d_start = (self._worker is thread and self._queued is not None
                        and type(self._queued.request) is Viewer1DBatchHydrationRequest)
                if one_d_start:
                    if _viewer_1d_control(error) and control is None: control = error
                elif not isinstance(error, Exception):
                    raise
                with self._lock:
                    failed = None
                    if self._worker is thread:
                        self._worker = None
                        queued = self._queued
                        if (queued is not None and queued.state is _EntryState.READING
                                and queued.delivery_guard is None):
                            failed = self._capture_locked(
                                queued, queued.token, HydrationOutcome.FAILED,
                                queued.ticket, "transport worker failed to start",
                                clear=True)
                if failed is None:
                    return
                pending = DetachedHydrationMutation(deliveries=(failed,))
                continue
            if control is not None: raise control
            return

    # -- cancellation and retirement ---------------------------------------- #

    def cancel_gate(self, gate) -> None:
        """Drop the queued reference carrying *gate*; an active read is never
        forcibly interrupted — its cancelled gate refuses the commit instead."""
        self.dispatch_detached(self.cancel_gate_detached(gate))

    def cancel_gate_detached(self, gate):
        with self._lock:
            if any(entry is not None and entry.state is _EntryState.DELIVERY_PENDING
                   for entry in (self._active, self._queued)):
                return DetachedHydrationMutation()
            queued = self._queued
            if queued is not None and queued.request.commit_gate is gate:
                delivery = self._capture_locked(
                    queued, queued.token, HydrationOutcome.CANCELLED,
                    queued.ticket, clear=True)
                return DetachedHydrationMutation(deliveries=(delivery,))
            return DetachedHydrationMutation()

    def retains_gate(self, gate) -> bool:
        """Whether any admitted request still carries *gate* (stores/gate)."""
        with self._lock:
            return any(
                entry is not None and entry.request.commit_gate is gate
                for entry in (self._active, self._queued)
            )

    def retire(self, *, join_timeout: float) -> bool:
        """Cancel the queued reference and join or report the exact worker."""
        with self._lock:
            if any(entry is not None and entry.state is _EntryState.DELIVERY_PENDING
                   for entry in (self._active, self._queued)):
                return False
            self._retired = True
            queued = self._queued
            mutation = DetachedHydrationMutation()
            if queued is not None and queued.state is not _EntryState.DELIVERY_PENDING:
                mutation = DetachedHydrationMutation(deliveries=(self._capture_locked(
                    queued, queued.token, HydrationOutcome.CANCELLED,
                    queued.ticket, clear=True),))
            worker = self._worker
            start_in_flight = worker is not None and worker.ident is None
        self.dispatch_detached(mutation)
        if (
            worker is not None
            and worker is not current_thread()
            and worker.ident is not None
        ):
            worker.join(timeout=max(0.0, float(join_timeout)))
        with self._lock:
            blocked = self._active is not None or self._queued is not None
            worker_slot_empty = self._worker is None
        return (not start_in_flight and (worker is None or not worker.is_alive())
                and worker_slot_empty and not blocked)

    # -- worker ------------------------------------------------------------- #

    def _ensure_worker_locked(self):
        if self._active is not None:
            return None
        worker = self._worker
        if worker is not None:
            return None
        if self._queued is None or self._retired:
            return None
        thread = None
        def run():
            try:
                self._run(thread)
            except _Viewer1DUnwind as unwind:
                with self._lock:
                    entry, guard = unwind.entry, unwind.guard
                    if (entry is not None and self._active is entry
                            and entry.state is _EntryState.CLEANUP_BLOCKED
                            and entry.delivery_guard is guard and guard.claimed
                            and type(entry.terminal) is _Viewer1DBlockedTerminal
                            and entry.terminal.notice.acknowledgement is not None):
                        entry.delivery_guard = None
                    if self._worker is thread: self._worker = None
                self.dispatch_detached(DetachedHydrationMutation())
                raise unwind.control
        thread = Thread(
            target=run,
            name="scattering-preview-transport",
            daemon=True,
        )
        self._worker = thread
        return thread

    def _run(self, worker) -> None:
        while True:
            with self._lock:
                if self._retired or self._queued is None:
                    if self._worker is worker: self._worker = None
                    return
                if (self._queued.state is not _EntryState.READING
                        or self._queued.delivery_guard is not None):
                    if self._worker is worker: self._worker = None
                    return
                entry, self._queued = self._queued, None
                self._active = entry
                token, ticket = entry.token, entry.ticket
            result = self._execute(entry, token, ticket)
            if result is None:
                delivery = None
                with self._lock:
                    if (self._active is entry and entry.state is _EntryState.READING
                            and entry.delivery_guard is None):
                        if self._queued is None and not self._retired:
                            self._active = None
                            self._queued = entry
                            continue
                        delivery = self._capture_locked(
                            entry, entry.token, (HydrationOutcome.CANCELLED
                                if self._retired else HydrationOutcome.SUPERSEDED),
                            entry.ticket, clear=True)
                    elif (self._active is entry and entry.state is _EntryState.CLEANUP_BLOCKED
                            and type(entry.terminal) is _Viewer1DBlockedTerminal
                            and entry.terminal.notice.acknowledgement is not None
                            and entry.delivery_guard is not None
                            and entry.delivery_guard.claimed):
                        entry.delivery_guard = None
                    if delivery is None and self._worker is worker: self._worker = None
                if delivery is None: return
                self.dispatch_detached(DetachedHydrationMutation(deliveries=(delivery,)))
                continue
            terminal = result if type(result) is _Viewer1DExecutionTerminal else None
            outcome, diagnostic = ((terminal.outcome, terminal.diagnostic)
                                   if terminal is not None else result)
            with self._lock:
                if not self._entry_current_locked(entry, token, ticket):
                    if self._worker is worker: self._worker = None
                    return
                final = entry.token
                current_ticket = entry.ticket
                committed = entry.committed_token
                committed_ticket = entry.committed_ticket
                if (
                    committed_ticket is not None
                    and committed_ticket is not current_ticket
                ):
                    guard = _DeliveryGuard()
                    deliveries = (
                        self._capture_locked(entry, committed, outcome,
                            committed_ticket, diagnostic, clear=True, guard=guard),
                        self._capture_locked(entry, final,
                            HydrationOutcome.ALREADY_RESIDENT, current_ticket,
                            clear=True, guard=guard),
                    )
                else:
                    deliveries = (self._capture_locked(
                        entry, final, outcome, current_ticket, diagnostic,
                        clear=True),)
            try:
                self.dispatch_detached(DetachedHydrationMutation(deliveries=deliveries))
            except BaseException as control:
                if terminal is not None: raise _Viewer1DUnwind(control) from None
                raise
            if terminal is not None and terminal.control is not None:
                raise _Viewer1DUnwind(terminal.control) from None

    def _entry_current_locked(self, entry, token, ticket):
        return (self._active is entry and entry.token is token and entry.ticket is ticket
                and entry.delivery_guard is None
                and entry.state in (_EntryState.READING, _EntryState.PREPARED))

    def _execute(self, entry: _TransportEntry, token, ticket):
        if type(entry.request) is Viewer1DBatchHydrationRequest:
            return self._execute_viewer_1d(entry, token, ticket)
        if type(entry.request) in _VIEWER_2D_REQUEST_TYPES:
            return self._execute_viewer(entry, token, ticket)
        try:
            preview = read_frame_preview(
                entry.request.read_key,
                detector_projection=entry.projection,
            )
        except Exception as error:
            return HydrationOutcome.FAILED, _diagnostic(error)
        with self._lock:
            if not self._entry_current_locked(entry, token, ticket):
                return None
            closed = entry.closed
            entry.committed_token = token
            entry.committed_ticket = ticket
        prepared = PreparedHydrationCommit(
            entry.request, token, entry.key, closed, preview, entry.projection
        )
        for attempt in (0, 1):
            with self._lock:
                if not self._entry_current_locked(entry, token, ticket):
                    return None
            try:
                outcome = self._commit(prepared)
            except Exception as error:
                if attempt:
                    return HydrationOutcome.FAILED, _diagnostic(error)
                continue
            if type(outcome) is HydrationOutcome:
                return outcome, preview.detector_diagnostic
            return HydrationOutcome.FAILED, "target port returned no outcome"
        return HydrationOutcome.FAILED, "unreachable"

    def _execute_viewer_1d(self, entry, token, ticket):
        caught = None
        for _ in (0, 1):
            try:
                if caught is None: return self._execute_viewer_1d_body(entry, token, ticket)
                return self._recover_viewer_1d(entry, token, ticket, caught)
            except _Viewer1DUnwind:
                raise
            except BaseException as error:
                if caught is None: caught = error
        raise caught

    def _execute_viewer_1d_body(self, entry, token, ticket):
        request = entry.request
        receipt = _mint_viewer_1d_budget_receipt(
            request, viewer_1d_budget(), VIEWER_1D_R, current_thread().ident, self)
        entry.viewer_1d_receipt = receipt
        transfer = mint_viewer_1d_transfer(request, receipt, self)
        entry.viewer_1d_transfer = transfer
        operation = begin_viewer_1d_read(request, receipt, transfer)
        if type(operation) is Viewer1DReadFailure:
            return self._dispose_viewer_1d(entry, operation.disposal,
                HydrationOutcome.FAILED, operation.diagnostic, token, ticket)
        _retire_viewer_1d_budget_receipt(receipt, "transferred")
        permit = mint_viewer_1d_pass_two_permit(operation, receipt, self)
        batch = operation.complete(permit)
        if type(batch) is Viewer1DReadFailure:
            return self._dispose_viewer_1d(entry, batch.disposal,
                HydrationOutcome.FAILED, batch.diagnostic, token, ticket)
        if type(batch) is not Viewer1DBatch: raise RuntimeError("viewer 1-D returned no batch")
        prepared_identity = object()
        transfer.install_batch(batch, prepared_identity, receipt.identity)
        prepared = _new_prepared_viewer_1d_commit(
            request, receipt.identity, prepared_identity, batch.batch_identity, transfer)
        with self._lock:
            if (not self._entry_current_locked(entry, token, ticket)
                    or entry.viewer_1d_transfer is not transfer):
                raise RuntimeError("viewer 1-D active lineage changed")
            entry.state = _EntryState.PREPARED
        error = None
        for attempt in (0, 1):
            try:
                returned = request.port.commit(prepared)
            except BaseException as caught:
                if transfer.owner_receipt is not None:
                    return _Viewer1DExecutionTerminal(HydrationOutcome.HYDRATED, None,
                        caught if _viewer_1d_control(caught) else None)
                if not _viewer_1d_control(caught) and attempt == 0:
                    error = caught
                    continue
                raise
            if (type(returned) is Viewer1DCommitReceipt
                    and transfer.owner_receipt is returned):
                return _Viewer1DExecutionTerminal(HydrationOutcome.HYDRATED, None)
            if transfer.owner_receipt is not None:
                return _Viewer1DExecutionTerminal(HydrationOutcome.HYDRATED, None)
            break
        raise RuntimeError(_viewer_1d_diagnostic(error)
                           if error else "viewer target refused batch")

    def _recover_viewer_1d(self, entry, token, ticket, error):
        original = error.control if type(error) is _Viewer1DReaderControl else error
        control = original if _viewer_1d_control(original) else None
        transfer = entry.viewer_1d_transfer
        if transfer is not None and transfer.owner_receipt is not None:
            return _Viewer1DExecutionTerminal(HydrationOutcome.HYDRATED, None, control)
        if transfer is None:
            receipt = entry.viewer_1d_receipt
            if receipt is not None and receipt.state is Viewer1DReadBudgetState.ACTIVE:
                _retire_viewer_1d_budget_receipt(receipt, "abandoned")
            return _Viewer1DExecutionTerminal(HydrationOutcome.FAILED,
                _viewer_1d_diagnostic(original), control)
        if viewer_1d_transfer_is_released(transfer):
            receipt = entry.viewer_1d_receipt
            if receipt is not None and receipt.state is Viewer1DReadBudgetState.ACTIVE:
                _retire_viewer_1d_budget_receipt(receipt, "abandoned")
            return _Viewer1DExecutionTerminal(HydrationOutcome.FAILED,
                _viewer_1d_diagnostic(original), control)
        disposal = (error.disposal if type(error) is _Viewer1DReaderControl
                    else transfer.disposal(_viewer_1d_diagnostic(original)))
        return self._dispose_viewer_1d(entry, disposal, HydrationOutcome.FAILED,
            _viewer_1d_diagnostic(original), token, ticket, control=control)

    def _dispose_viewer_1d(self, entry, disposal, outcome, diagnostic, token, ticket,
                           *, control=None):
        with self._lock:
            if (not self._entry_current_locked(entry, token, ticket)
                    or type(disposal) is not Viewer1DDisposal
                    or entry.viewer_1d_transfer is not disposal.transfer
                    or not viewer_1d_disposal_is_current(disposal)):
                return None
        try: cleaned = disposal.release()
        except BaseException as escaped:
            escaped = getattr(escaped, "control", escaped)
            if control is None and _viewer_1d_control(escaped): control = escaped
            cleaned = viewer_1d_transfer_is_released(disposal.transfer)
        if cleaned:
            receipt = entry.viewer_1d_receipt
            if receipt is not None and receipt.state is Viewer1DReadBudgetState.ACTIVE:
                _retire_viewer_1d_budget_receipt(receipt, "abandoned")
            return _Viewer1DExecutionTerminal(outcome, diagnostic, control)
        notice = _new_viewer_1d_cleanup_notice(
            entry.request, diagnostic, self, self._viewer_1d_claim)
        guard, outer = _DeliveryGuard(), HydrationTransportCleanupToken()
        guard.claimed = True
        terminal = _Viewer1DBlockedTerminal(outcome, diagnostic, notice)
        with self._lock:
            if (not self._entry_current_locked(entry, token, ticket)
                    or not viewer_1d_disposal_is_current(disposal)): return None
            entry.disposal = disposal
            entry.state, entry.cleanup_token, entry.terminal = _EntryState.CLEANUP_BLOCKED, outer, terminal
            entry.delivery_guard = guard
        pending_control = control
        for _ in (0, 1):
            try: entry.request.port.cleanup_pending(notice)
            except BaseException as escaped:
                if pending_control is None and _viewer_1d_control(escaped):
                    pending_control = escaped
            if notice.acknowledgement is not None: break
        if notice.acknowledgement is None: return None
        if pending_control is not None:
            raise _Viewer1DUnwind(pending_control, entry, guard)
        return None

    def _execute_viewer(self, entry, token, ticket):
        request = entry.request
        with self._lock:
            if not self._entry_current_locked(entry, token, ticket):
                return None
        try:
            activation = request.port.activate(request)
        except Exception as error:
            return HydrationOutcome.FAILED, _diagnostic(error)
        if not _viewer_activation_matches(request, activation):
            try:
                disposal = Viewer2DDisposal(
                    request, activation, None, Viewer2DDisposalToken())
            except (AttributeError, TypeError, ValueError, RuntimeError):
                diagnostic = (getattr(activation, "diagnostic", None)
                              if type(activation) is Viewer2DReadActivation else None)
                diagnostic = (diagnostic if type(diagnostic) is str and diagnostic
                              else "viewer port returned no owning activation")
                return HydrationOutcome.FAILED, diagnostic
            return self._dispose_viewer(
                entry, disposal, HydrationOutcome.FAILED,
                "viewer activation phase, receipt, or budget does not match request",
                token, ticket)
        prepared = None
        try:
            if type(request) is Viewer2DCatalogHydrationRequest:
                prepared = Prepared2DCatalogCommit(
                    request, activation,
                    catalog_viewer_2d(request.path, policy=request.policy))
            else:
                prepared = Prepared2DFrameCommit(
                    request, activation,
                    read_viewer_2d_frame(
                        request.catalog, request.label, policy=request.policy),
                    request.receipt_identity)
            with self._lock:
                entry.state = _EntryState.PREPARED
                if not self._entry_current_locked(entry, token, ticket):
                    return None
            outcome = None
            for _ in (0, 1):
                with self._lock:
                    if not self._entry_current_locked(entry, token, ticket):
                        return None
                try:
                    outcome = request.port.commit(prepared)
                    break
                except Exception:
                    continue
            if type(outcome) is not HydrationOutcome:
                outcome = HydrationOutcome.FAILED
            if outcome in (HydrationOutcome.HYDRATED,
                           HydrationOutcome.ALREADY_RESIDENT):
                return outcome, None
            diagnostic = "viewer target refused prepared value"
        except Exception as error:
            outcome, diagnostic = HydrationOutcome.FAILED, _diagnostic(error)
        disposal_token = Viewer2DDisposalToken()
        try:
            disposal = Viewer2DDisposal(
                request, activation, prepared, disposal_token)
        except (AttributeError, TypeError, ValueError, RuntimeError) as error:
            disposal = Viewer2DDisposal(request, activation, None, disposal_token)
            outcome, diagnostic = HydrationOutcome.FAILED, _diagnostic(error)
        return self._dispose_viewer(
            entry, disposal, outcome, diagnostic, token, ticket)

    def _dispose_viewer(self, entry, disposal, outcome, diagnostic, token, ticket):
        with self._lock:
            if not self._entry_current_locked(entry, token, ticket):
                return None
        entry.disposal = disposal
        try:
            receipt = entry.request.port.dispose(disposal)
        except Exception:
            receipt = None
        if (type(receipt) is Viewer2DCleanupReceipt
                and receipt.token is disposal.token
                and receipt.state is Viewer2DCleanupState.CLEANED):
            return outcome, diagnostic
        with self._lock:
            entry.cleanup_token = HydrationTransportCleanupToken()
            entry.terminal = (outcome, diagnostic)
            entry.state = _EntryState.CLEANUP_BLOCKED
            self._worker = None
        return None, None

    # -- completion bookkeeping --------------------------------------------- #

    def _capture_locked(self, entry, token, outcome, ticket,
                        diagnostic=None, *, clear, restore=None, guard=None):
        completion = HydrationCompletion(token, outcome, diagnostic or None)
        notify = ticket is None or ticket._settle(completion)
        if notify or type(entry.request) not in _VIEWER_1D_REQUEST_TYPES:
            self._counters[outcome] += 1
            self._completions.append(completion)
        guard = _DeliveryGuard() if guard is None else guard
        active_slot = self._active is entry
        if not active_slot and self._queued is not entry:
            raise RuntimeError("delivery entry has no transport slot")
        entry.state = _EntryState.DELIVERY_PENDING
        entry.delivery_guard = guard
        return _DetachedDelivery(entry, completion, guard, notify, active_slot, clear, restore)

    def _delivery_valid_locked(self, delivery, *, claimed):
        entry = delivery.entry
        slot = self._active if delivery.active_slot else self._queued
        return (delivery.guard.claimed is claimed and slot is entry
                and entry.state is _EntryState.DELIVERY_PENDING
                and entry.delivery_guard is delivery.guard)

    def _delivery_finalized_locked(self, delivery):
        entry = delivery.entry
        if not delivery.guard.claimed or entry.delivery_guard is not None:
            return False
        if delivery.clear:
            return self._active is not entry and self._queued is not entry
        slot = self._active if delivery.active_slot else self._queued
        return slot is entry and entry.state is delivery.restore

    def _recover_viewer_1d_capture_locked(
            self, entry, ticket, outcome, diagnostic, was_unsettled):
        completion, guard = ticket.result(), entry.delivery_guard
        if (was_unsettled and self._active is entry and entry.ticket is ticket
                and entry.state is _EntryState.DELIVERY_PENDING
                and type(guard) is _DeliveryGuard and not guard.claimed
                and type(completion) is HydrationCompletion
                and completion.token is entry.token and completion.outcome is outcome
                and completion.diagnostic == (diagnostic or None)):
            return _DetachedDelivery(entry, completion, guard, True, True, True)
        return None

    def blocked_cleanup_token(self, expected_transport_token):
        with self._lock:
            entry = self._active
            if (entry is not None and entry.state is _EntryState.CLEANUP_BLOCKED
                    and entry.token is expected_transport_token
                    and (type(entry.request) not in _VIEWER_1D_REQUEST_TYPES
                         or entry.delivery_guard is None
                         and self._worker is None
                         and _viewer_1d_cleanup_current(entry)
                         and type(entry.terminal) is _Viewer1DBlockedTerminal
                         and entry.terminal.notice.acknowledgement is not None)):
                return entry.cleanup_token
            return None

    def retry_blocked_cleanup(self, outer_token):
        with self._lock:
            entry = self._active
            if (entry is None or entry.state is not _EntryState.CLEANUP_BLOCKED
                    or entry.cleanup_token is not outer_token or entry.retrying
                    or type(entry.request) in _VIEWER_1D_REQUEST_TYPES
                    and (entry.delivery_guard is not None or self._worker is not None
                         or type(entry.terminal) is not _Viewer1DBlockedTerminal
                         or entry.terminal.notice.acknowledgement is None
                         or not _viewer_1d_cleanup_current(entry))):
                return HydrationTransportCleanupReceipt(
                    outer_token, Viewer2DCleanupState.STALE)
            entry.retrying = True
            disposal = entry.disposal
        control = None
        try:
            receipt = (disposal.retry() if type(disposal) is Viewer1DDisposal
                       else entry.request.port.dispose(disposal))
        except BaseException as escaped:
            receipt = None
            if type(disposal) is Viewer1DDisposal:
                escaped = getattr(escaped, "control", escaped)
                if _viewer_1d_control(escaped): control = escaped
            elif not isinstance(escaped, Exception):
                raise
        with self._lock:
            entry.retrying = False
            if (self._active is not entry or entry.cleanup_token is not outer_token):
                state = Viewer2DCleanupState.STALE
                mutation = DetachedHydrationMutation()
            elif (type(disposal) is Viewer1DDisposal and receipt is not True
                    or type(disposal) is not Viewer1DDisposal
                    and (type(receipt) is not Viewer2DCleanupReceipt
                         or receipt.token is not disposal.token
                         or receipt.state is not Viewer2DCleanupState.CLEANED)):
                state = Viewer2DCleanupState.CLEANUP_PENDING
                mutation = DetachedHydrationMutation()
            else:
                terminal = entry.terminal
                outcome, diagnostic = ((terminal.outcome, terminal.diagnostic)
                    if type(terminal) is _Viewer1DBlockedTerminal else terminal)
                if (type(disposal) is Viewer1DDisposal
                        and entry.viewer_1d_receipt is not None
                        and entry.viewer_1d_receipt.state is Viewer1DReadBudgetState.ACTIVE):
                    _retire_viewer_1d_budget_receipt(entry.viewer_1d_receipt, "abandoned")
                was_unsettled = entry.ticket.result() is None
                try:
                    delivery = self._capture_locked(
                        entry, entry.token, outcome, entry.ticket, diagnostic,
                        clear=True)
                except BaseException as escaped:
                    if type(disposal) is not Viewer1DDisposal or not _viewer_1d_control(escaped):
                        raise
                    if control is None: control = escaped
                    delivery = self._recover_viewer_1d_capture_locked(
                        entry, entry.ticket, outcome, diagnostic, was_unsettled)
                if delivery is None:
                    state = Viewer2DCleanupState.CLEANUP_PENDING
                    mutation = DetachedHydrationMutation()
                else:
                    state = Viewer2DCleanupState.CLEANED
                    mutation = DetachedHydrationMutation(deliveries=(delivery,))
                    entry.disposal = None
        self.dispatch_detached(mutation)
        if control is not None: raise control
        return HydrationTransportCleanupReceipt(outer_token, state)


def _diagnostic(error: BaseException) -> str:
    return str(error) or type(error).__name__


def _viewer_1d_diagnostic(error: BaseException) -> str:
    try: value = str(error) or type(error).__name__
    except BaseException: value = type(error).__name__
    while len(value.encode("utf8")) > 256: value = value[:-1]
    return value


def _viewer_1d_control(error):
    return isinstance(error, MemoryError) or (isinstance(error, BaseException)
                                               and not isinstance(error, Exception))


__all__ = [
    "DetachedHydrationMutation",
    "HydrationTransportCleanupReceipt",
    "HydrationTransportCleanupToken",
    "HydrationTransport",
    "PreparedHydrationCommit",
]
