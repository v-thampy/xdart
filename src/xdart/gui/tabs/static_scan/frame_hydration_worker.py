# -*- coding: utf-8 -*-
"""Background frame-hydration worker (greenfield Phase 3 / D2).

Scroll-back to a frame that has been evicted from the in-memory window needs its
heavy payload (cake / full raw) rehydrated from ``scan.frames`` / the ``.nxs``.
Doing that h5py read on the GUI thread froze the UI for ~5 s per evicted frame
(``display_data._hydrate_frame_from_disk``'s idle branch).  This worker moves the
read OFF the GUI thread: it pulls publications through
``PublicationStore.get_or_hydrate`` (which invokes the registered hydrator) on
its own thread, then emits :attr:`sigHydrated` so the GUI re-renders — by which
point the heavy payload is resident in the shared store.

Staleness is the CALLER's concern: every request carries the
``displayFrameWidget.display_generation`` it was made under, echoed back in the
signal, so a selection/mode change that bumped the generation makes the GUI drop
the late result.  The worker itself never reaches into GUI state.
"""

import logging
from collections import deque
from threading import Condition

from pyqtgraph import Qt

logger = logging.getLogger(__name__)


class FrameHydrationWorker(Qt.QtCore.QThread):
    """One persistent thread draining a request queue into the store.

    ``request(label, generation)`` enqueues a hydration; ``run`` pops FIFO,
    calls ``store.get_or_hydrate(label)`` (the heavy read), and emits
    ``sigHydrated(label, generation)`` when it yields a payload.  Cheap when the
    payload is already resident — ``get_or_hydrate`` returns without a read — so
    duplicate requests from rapid scroll-back are nearly free and need no
    dedupe.  ``stop()`` drains and joins; safe to call once at teardown.
    """

    #: (label, generation) — label echoes the request; generation gates staleness.
    sigHydrated = Qt.QtCore.Signal(object, int)

    def __init__(self, store, parent=None):
        super().__init__(parent)
        self._store = store
        self._cond = Condition()
        self._queue: deque = deque()
        self._newest_gen = -1        # highest generation ever requested (P3)
        self._stop = False

    def request(self, label, generation: int) -> None:
        """Enqueue a hydration request (non-blocking; returns immediately)."""
        generation = int(generation)
        with self._cond:
            if self._stop:
                return
            if generation > self._newest_gen:
                self._newest_gen = generation
            self._queue.append((label, generation))
            self._cond.notify()

    def run(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                label, generation = self._queue.popleft()
                newest = self._newest_gen
            if generation < newest:
                # P3 coalesce: a newer selection/mode superseded this request,
                # so don't even hit disk for a frame the user already scrolled
                # past — the GUI would drop the result anyway.
                continue
            try:
                publication = self._store.get_or_hydrate(label)
            except Exception:
                logger.debug("background hydration failed for %s", label,
                             exc_info=True)
                publication = None
            if publication is not None:
                # The GUI handler still re-checks generation == the live
                # display_generation (a change that landed during the read).
                self.sigHydrated.emit(label, generation)

    def stop(self, timeout_ms: int = 5000) -> bool:
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
