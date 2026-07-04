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

logger = logging.getLogger(__name__)

_MAX_PENDING_REQUESTS = 64


@dataclass
class _HydrationRequest:
    labels: tuple
    generation: int
    purpose: str
    consumer: ConsumerKind


class FrameHydrationWorker(Qt.QtCore.QThread):
    """One persistent thread draining a request queue into the store.

    ``request(label, generation)`` enqueues a hydration; ``run`` pops FIFO,
    calls ``store.get_or_hydrate(label)`` (the heavy read), and emits
    ``sigHydrated(label, generation)`` when it yields a payload.  Cheap when the
    payload is already resident — ``get_or_hydrate`` returns without a read — so
    duplicate requests from rapid scroll-back are nearly free and need no
    dedupe.  ``stop()`` drains and joins; safe to call once at teardown.

    ``store`` may also be a zero-arg provider returning one store or an iterable
    of stores.  The live app uses that to hydrate the authoritative
    ``FrameRecordStore`` first while the transitional ``PublicationStore`` still
    exists as a fallback projection.
    """

    #: (label, generation) — label echoes the request; generation gates staleness.
    sigHydrated = Qt.QtCore.Signal(object, int)

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
    def _token(label, generation, purpose, consumer):
        return (label, int(generation), str(purpose or "full"), consumer)

    def _discard_locked(self, request: _HydrationRequest) -> None:
        for label in request.labels:
            self._queued.discard(
                self._token(label, request.generation,
                            request.purpose, request.consumer))

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
            supersede_reason=SupersedeReason.SELECTION) -> None:
        """Enqueue a hydration request (non-blocking; returns immediately)."""
        generation = int(generation)
        purpose = str(purpose or "full")
        consumer = self._consumer(consumer)
        supersede_reason = self._reason(supersede_reason)
        with self._cond:
            if self._stop:
                return
            if generation > self._newest_gen:
                self._newest_gen = generation
                self._drain_stale_locked(supersede_reason)
            token = self._token(label, generation, purpose, consumer)
            if token in self._queued:
                return
            self._queued.add(token)
            if (
                purpose == "1d"
                and self._queue
                and self._queue[-1].generation == generation
                and self._queue[-1].purpose == purpose
                and self._queue[-1].consumer is consumer
            ):
                self._queue[-1].labels = (*self._queue[-1].labels, label)
            else:
                self._queue.append(
                    _HydrationRequest((label,), generation, purpose, consumer))
            self._trim_pending_locked()
            self._cond.notify()

    def _pop_batch_locked(self):
        request = self._queue.popleft()
        self._discard_locked(request)
        return list(request.labels), request.generation, request.purpose, request.consumer

    def _hydrate_full(self, label) -> bool:
        hydrated = False
        for store in self._stores():
            getter = getattr(store, "get_or_hydrate", None)
            if getter is None:
                continue
            try:
                hydrated = getter(label) is not None or hydrated
            except Exception:
                logger.debug("background hydration failed for %s", label,
                             exc_info=True)
        return hydrated

    def _hydrate_1d_many(self, labels) -> bool:
        hydrated = False
        for store in self._stores():
            getter = getattr(store, "get_1d_many_or_hydrate", None)
            if getter is None:
                continue
            try:
                hydrated = bool(getter(labels)) or hydrated
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
                labels, generation, purpose, consumer = self._pop_batch_locked()
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
                continue
            if purpose == "1d":
                self._hydrate_1d_many(tuple(labels))
            else:
                for label in labels:
                    self._hydrate_full(label)
            # The GUI handler still re-checks generation == the live
            # display_generation (a change that landed during the read). Emit
            # even when hydration failed so GUI-side pending dedupe can clear the
            # request key for this generation.
            emitted_label = tuple(labels) if purpose == "1d" else labels[-1]
            self.sigHydrated.emit(emitted_label, generation)

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
