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

_VIEWER_REQUEST_TYPES = (Viewer2DCatalogHydrationRequest, Viewer2DFrameHydrationRequest)


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
    disposal: Viewer2DDisposal | None = None
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

    def __init__(self, commit, derive, *, completion_sink=None) -> None:
        # commit/derive are owner-bound at construction, never queue-carried.
        self._commit = commit
        self._derive = derive
        self._completion_sink = completion_sink
        self._lock = RLock()
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
        if (
            type(request) not in (HydrationRequest, *_VIEWER_REQUEST_TYPES)
            or getattr(request, "read_key", None) is None
            or request.token is None
            or not request.enqueueable
        ):
            return DetachedHydrationMutation()
        token = request.token
        with self._lock:
            if (self._retired or request.commit_gate.cancelled
                    or any(entry is not None and entry.state is _EntryState.DELIVERY_PENDING
                           for entry in (self._active, self._queued))):
                return DetachedHydrationMutation()
            active = self._active
            if (
                type(request) is HydrationRequest
                and
                active is not None
                and type(active.request) is HydrationRequest
                and active.request.read_key == request.read_key
            ):
                displaced = active.token
                displaced_ticket = active.ticket
                active.token = token
                active.closed = bool(closed)
                deliveries = []
                if displaced != token:
                    active.ticket = _HydrationTicket(token)
                    deliveries.append(self._capture_locked(
                        active, displaced, HydrationOutcome.SUPERSEDED,
                        displaced_ticket, clear=False, restore=_EntryState.READING))
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
                    except Exception:
                        pass
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
                return
            try:
                thread.start()
                return
            except Exception:
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
            self._run(thread)
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
                with self._lock:
                    if (self._active is entry and entry.state is _EntryState.READING
                            and entry.delivery_guard is None):
                        self._active = None
                        self._queued = entry
                        continue
                    if self._worker is worker: self._worker = None
                return
            outcome, diagnostic = result
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
            self.dispatch_detached(DetachedHydrationMutation(deliveries=deliveries))

    def _entry_current_locked(self, entry, token, ticket):
        return (self._active is entry and entry.token is token and entry.ticket is ticket
                and entry.delivery_guard is None
                and entry.state in (_EntryState.READING, _EntryState.PREPARED))

    def _execute(self, entry: _TransportEntry, token, ticket):
        if type(entry.request) in _VIEWER_REQUEST_TYPES:
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
        self._counters[outcome] += 1
        self._completions.append(completion)
        notify = ticket is None or ticket._settle(completion)
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

    def blocked_cleanup_token(self, expected_transport_token):
        with self._lock:
            entry = self._active
            if (entry is not None and entry.state is _EntryState.CLEANUP_BLOCKED
                    and entry.token is expected_transport_token):
                return entry.cleanup_token
            return None

    def retry_blocked_cleanup(self, outer_token):
        with self._lock:
            entry = self._active
            if (entry is None or entry.state is not _EntryState.CLEANUP_BLOCKED
                    or entry.cleanup_token is not outer_token or entry.retrying):
                return HydrationTransportCleanupReceipt(
                    outer_token, Viewer2DCleanupState.STALE)
            entry.retrying = True
            disposal = entry.disposal
        try:
            receipt = entry.request.port.dispose(disposal)
        except Exception:
            receipt = None
        with self._lock:
            entry.retrying = False
            if (self._active is not entry or entry.cleanup_token is not outer_token):
                state = Viewer2DCleanupState.STALE
                mutation = DetachedHydrationMutation()
            elif (type(receipt) is not Viewer2DCleanupReceipt
                    or receipt.token is not disposal.token
                    or receipt.state is not Viewer2DCleanupState.CLEANED):
                state = Viewer2DCleanupState.CLEANUP_PENDING
                mutation = DetachedHydrationMutation()
            else:
                state = Viewer2DCleanupState.CLEANED
                outcome, diagnostic = entry.terminal
                entry.disposal = None
                delivery = self._capture_locked(
                    entry, entry.token, outcome, entry.ticket, diagnostic,
                    clear=True)
                mutation = DetachedHydrationMutation(deliveries=(delivery,))
        self.dispatch_detached(mutation)
        return HydrationTransportCleanupReceipt(outer_token, state)


def _diagnostic(error: BaseException) -> str:
    return str(error) or type(error).__name__


__all__ = [
    "DetachedHydrationMutation",
    "HydrationTransportCleanupReceipt",
    "HydrationTransportCleanupToken",
    "HydrationTransport",
    "PreparedHydrationCommit",
]
