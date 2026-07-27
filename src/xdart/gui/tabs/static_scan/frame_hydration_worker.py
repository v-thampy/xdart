# -*- coding: utf-8 -*-
"""Background frame-hydration worker (greenfield Phase 3 / D2).

Scroll-back to a frame that has been evicted from the in-memory window needs its
heavy payload (cake / full raw) rehydrated from ``scan.frames`` / the ``.nxs``.
Doing that h5py read on the GUI thread froze the UI for ~5 s per evicted frame
(``display_data._hydrate_frame_from_disk``'s idle branch).  This worker moves the
read OFF the GUI thread: it pulls records/publications through
``*.get_or_hydrate`` (which invokes the registered hydrator) on its own thread,
then emits :attr:`sigHydrated` so the GUI re-renders — by which point the heavy
payload is resident in the shared store.

Staleness is the CALLER's concern: every request carries the
``displayFrameWidget.display_generation`` it was made under, echoed back in the
signal, so a selection/mode change that bumped the generation makes the GUI drop
the late result.  The worker itself never reaches into GUI state.
"""

import logging
from collections import deque
from dataclasses import dataclass
from threading import Condition

from pyqtgraph import Qt

from .display_logic import (
    ConsumerKind,
    HydrationSupersedeAction,
    SupersedeReason,
    hydration_supersede_action,
)
from xdart.modules.display_context import HydrationOwner
from .browse_debug import browse_debug_log, sequence_summary

logger = logging.getLogger(__name__)

_MAX_PENDING_REQUESTS = 64


@dataclass
class _HydrationRequest:
    labels: tuple
    generation: int
    purpose: str
    consumer: ConsumerKind
    #: X1 O-3 (c3): the display context this request was made UNDER, captured
    #: at request time and echoed on completion.  A generation alone cannot
    #: authorize a completion — two contexts can be at the same generation, and
    #: a completion held across a context switch would then be admitted by the
    #: owner it no longer belongs to.  Values only: two strings.
    context_token: str = ""
    context_scan_key: str = ""
    #: c3R-b (§9.2.3): the EXACT stores this request lands in, resolved when it
    #: was made.  The worker must never ask a live provider where a request
    #: belongs — that is how a read made under a browse landed in the resumed
    #: run's store.
    stores: tuple = ()
    #: The context's commit authority and the epoch this request was minted
    #: under (§9.2.4).  Read is unlocked; the INSERT goes through the gate.
    commit_gate: object = None
    epoch: int = 0
    #: The file identity the request was made against, so a completion can be
    #: refused on source as well as on scan key.
    context_source: str = ""

    @property
    def owner(self):
        """The complete values-only identity this request belongs to (§10.3)."""
        return HydrationOwner.of(
            context_token=self.context_token,
            scan_key=self.context_scan_key,
            source=self.context_source,
            epoch=self.epoch)

    @property
    def owner(self):
        """The complete values-only identity this request belongs to."""
        return HydrationOwner.of(
            context_token=self.context_token,
            scan_key=self.context_scan_key,
            source=self.context_source,
            epoch=self.epoch)


class FrameHydrationWorker(Qt.QtCore.QThread):
    """One persistent thread draining a request queue into the store.

    ``request(label, generation)`` enqueues a hydration; ``run`` pops FIFO,
    calls ``store.get_or_hydrate(label)`` (the heavy read), and emits
    ``sigHydrated(label, generation)`` when it yields a payload.  Cheap when the
    payload is already resident — ``get_or_hydrate`` returns without a read — so
    duplicate requests from rapid scroll-back are nearly free and need no
    dedupe.  ``stop()`` drains and joins; safe to call once at teardown.

    ``store`` may also be a zero-arg provider returning one store or an iterable
    of stores.  Stores may expose ``hydration_purposes`` so a raw-preview request
    does not wake an integrated-results-only disk hydrator.
    """

    #: ``(label, generation, owner)`` — label echoes the request, generation
    #: gates staleness, and ``owner`` is the ``(context_token, scan_key)`` pair
    #: the request was made under so the GUI can refuse a completion that
    #: belongs to a context it has since left (X1 O-3 c3).  A two-argument slot
    #: may still connect; the extra argument is simply not delivered to it.
    sigHydrated = Qt.QtCore.Signal(object, int, object)

    def __init__(self, store, parent=None):
        super().__init__(parent)
        self._store = store
        self._cond = Condition()
        self._queue: deque = deque()
        self._queued: set[tuple[object, int, str, ConsumerKind]] = set()
        self._newest_gen = -1        # highest generation ever requested (P3)
        self._stop = False

    def _stores(self):
        source = self._store
        if callable(source) and not hasattr(source, "get_or_hydrate"):
            source = source()
        if source is None:
            return ()
        if (
            hasattr(source, "get_or_hydrate")
            or hasattr(source, "get_1d_many_or_hydrate")
        ):
            return (source,)
        return tuple(store for store in source if store is not None)

    @staticmethod
    def _consumer(value):
        if isinstance(value, ConsumerKind):
            return value
        try:
            return ConsumerKind(str(value))
        except ValueError:
            return ConsumerKind.PLOT_1D

    @staticmethod
    def _reason(value):
        if isinstance(value, SupersedeReason):
            return value
        try:
            return SupersedeReason(str(value))
        except ValueError:
            return SupersedeReason.GENERATION

    @staticmethod
    def _token(label, generation, purpose, consumer, owner=None):
        # §9.2.6 / §10.3: the COMPLETE owner is part of the identity — context
        # token, scan key, SOURCE identity and epoch.  Without the context the
        # same label at the same generation in two sub-scans collapsed to one
        # request; without the source, two sub-scans of one container did.
        owner = owner if owner is not None else HydrationOwner()
        return (label, int(generation), str(purpose or "full"), consumer,
                owner.as_tuple())

    def _discard_locked(self, request: _HydrationRequest) -> None:
        for label in request.labels:
            self._queued.discard(
                self._token(label, request.generation,
                            request.purpose, request.consumer,
                            request.owner))

    def _drain_stale_locked(self, reason=SupersedeReason.GENERATION) -> None:
        if not self._queue:
            return
        reason = self._reason(reason)
        keep = deque()
        for request in self._queue:
            if int(request.generation) >= self._newest_gen:
                keep.append(request)
                continue
            action = hydration_supersede_action(request.consumer, reason)
            if action is HydrationSupersedeAction.COMPLETE_AND_APPEND:
                keep.append(request)
            else:
                self._discard_locked(request)
        self._queue = keep

    def _trim_pending_locked(self) -> None:
        while len(self._queue) > _MAX_PENDING_REQUESTS:
            self._discard_locked(self._queue.popleft())

    def cancel_stale_before(
            self, generation: int,
            *, reason=SupersedeReason.GENERATION) -> None:
        """Drop queued work from generations older than ``generation``."""
        generation = int(generation)
        with self._cond:
            if generation > self._newest_gen:
                self._newest_gen = generation
            self._drain_stale_locked(reason)
            self._cond.notify()

    def request(
            self, label, generation: int, *, purpose: str = "full",
            consumer=ConsumerKind.PLOT_1D,
            supersede_reason=SupersedeReason.SELECTION,
            context_token: str = "", context_scan_key: str = "",
            stores=None, commit_gate=None, epoch: int = 0,
            context_source: str = "") -> None:
        """Enqueue a hydration request (non-blocking; returns immediately).

        ``context_token``/``context_scan_key`` name the display context at
        REQUEST time (X1 O-3 c3).  They are carried untouched to the completion
        so admission can compare against the context that is selected THEN.
        """
        generation = int(generation)
        context_token = str(context_token or "")
        context_scan_key = str(context_scan_key or "")
        context_source = str(context_source or "")
        # §9.2.3: resolve the target NOW.  Doing it in `run()` meant the store
        # was chosen after the display may already have moved on.
        if stores is None:
            stores = self._stores()
        stores = tuple(stores or ())
        purpose = str(purpose or "full")
        consumer = self._consumer(consumer)
        supersede_reason = self._reason(supersede_reason)
        with self._cond:
            if self._stop:
                return
            if generation > self._newest_gen:
                self._newest_gen = generation
                self._drain_stale_locked(supersede_reason)
            owner = HydrationOwner.of(
                context_token=context_token, scan_key=context_scan_key,
                source=context_source, epoch=epoch)
            token = self._token(label, generation, purpose, consumer, owner)
            if token in self._queued:
                browse_debug_log(
                    logger,
                    "hydration_worker_enqueue",
                    labels=sequence_summary((label,)),
                    generation=generation,
                    purpose=purpose,
                    consumer=consumer.value,
                    granted=False,
                    suppressed_by="duplicate_token",
                    queue_depth=len(self._queue),
                )
                return
            self._queued.add(token)
            if (
                purpose == "1d"
                and self._queue
                and self._queue[-1].generation == generation
                and self._queue[-1].purpose == purpose
                and self._queue[-1].consumer is consumer
                # §9.2.6 / §10.3: never coalesce across contexts, SOURCES or
                # epochs — the whole owner has to match, not part of it.
                and self._queue[-1].owner == owner
            ):
                self._queue[-1].labels = (*self._queue[-1].labels, label)
            else:
                self._queue.append(
                    _HydrationRequest((label,), generation, purpose, consumer,
                                      context_token, context_scan_key,
                                      stores, commit_gate, epoch,
                                      context_source))
            browse_debug_log(
                logger,
                "hydration_worker_enqueue",
                labels=sequence_summary((label,)),
                generation=generation,
                purpose=purpose,
                consumer=consumer.value,
                granted=True,
                queue_depth=len(self._queue),
            )
            self._trim_pending_locked()
            self._cond.notify()

    def _pop_batch_locked(self):
        request = self._queue.popleft()
        self._discard_locked(request)
        return request

    @staticmethod
    def _store_supports_purpose(store, purpose: str) -> bool:
        purposes = getattr(store, "hydration_purposes", None)
        return purposes is None or purpose in purposes

    @staticmethod
    def _commit_kwargs(request):
        """The commit-gate arguments a store insertion must be qualified by."""
        if getattr(request, "commit_gate", None) is None:
            return {}
        return {"commit_gate": request.commit_gate,
                "commit_epoch": request.epoch}

    def _hydrate_full(self, request, label, purpose: str) -> bool:
        hydrated = False
        commit = self._commit_kwargs(request)
        for store in (request.stores or ()):
            if not self._store_supports_purpose(store, purpose):
                continue
            getter = getattr(store, "get_or_hydrate", None)
            if getter is None:
                continue
            try:
                hydrated = getter(label, **commit) is not None or hydrated
            except Exception:
                logger.debug("background hydration failed for %s", label,
                             exc_info=True)
        return hydrated

    def _hydrate_1d_many(self, request, labels) -> bool:
        hydrated = False
        commit = self._commit_kwargs(request)
        for store in (request.stores or ()):
            getter = getattr(store, "get_1d_many_or_hydrate", None)
            if getter is None:
                continue
            try:
                hydrated = bool(getter(labels, **commit)) or hydrated
            except Exception:
                logger.debug("background 1D hydration failed for %s", labels,
                             exc_info=True)
        return hydrated

    def run(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                request = self._pop_batch_locked()
                labels = list(request.labels)
                generation = request.generation
                purpose = request.purpose
                consumer = request.consumer
                owner = request.owner
                newest = self._newest_gen
            if (
                generation < newest
                and hydration_supersede_action(
                    consumer, SupersedeReason.SELECTION
                ) is HydrationSupersedeAction.CANCEL
            ):
                # P3 coalesce: a newer selection/mode superseded this request,
                # so don't even hit disk for a frame the user already scrolled
                # past — the GUI would drop the result anyway.
                browse_debug_log(
                    logger,
                    "hydration_worker_result",
                    labels=sequence_summary(labels),
                    generation=generation,
                    newest_generation=newest,
                    purpose=purpose,
                    consumer=consumer.value,
                    success=False,
                    emitted=False,
                    suppressed_by="stale_generation",
                )
                continue
            if purpose == "1d":
                success = self._hydrate_1d_many(request, tuple(labels))
            else:
                success = False
                for label in labels:
                    success = (
                        self._hydrate_full(request, label, purpose) or success)
            # The GUI handler still re-checks generation == the live
            # display_generation (a change that landed during the read). Emit
            # even when hydration failed so GUI-side pending dedupe can clear the
            # request key for this generation.
            emitted_label = tuple(labels) if purpose == "1d" else labels[-1]
            browse_debug_log(
                logger,
                "hydration_worker_result",
                labels=sequence_summary(labels),
                generation=generation,
                newest_generation=newest,
                purpose=purpose,
                consumer=consumer.value,
                success=bool(success),
                emitted=True,
            )
            self.sigHydrated.emit(emitted_label, generation, owner)

    def stop(self, timeout_ms: int = 8000) -> bool:
        """Signal the loop to exit and join (idempotent).

        Returns ``True`` iff the thread actually stopped within ``timeout_ms``.
        A ``False`` return means an in-flight disk read is still running (bounded
        by ``catch_h5py_file``'s retry cap) — the caller MUST keep the handle so
        the QThread object isn't destroyed while its thread runs (P1)."""
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        if self.isRunning():
            return bool(self.wait(timeout_ms))
        return True
