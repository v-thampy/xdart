# -*- coding: utf-8 -*-
"""Background whole-scan aggregation worker (greenfield Step 7b, the wiring).

A whole-scan Sum/Average over a scan longer than the bounded display store can't
be built by collapsing resident traces (it would silently drop evicted frames —
the Round-12 bug).  :mod:`xdart.modules.scan_aggregate` computes it correctly
from the on-disk primary stack ⊕ the in-memory tail, but that is a chunked HDF5
read — too slow for the GUI thread.  This worker moves it OFF the GUI thread,
mirroring :class:`FrameHydrationWorker`: ``request`` enqueues, ``run`` computes,
``sigAggregated(key, generation, result)`` hands the result back for a re-render.

Staleness is the caller's concern: each request carries the
``displayFrameWidget.display_generation`` it was made under, echoed back so a
selection/mode change that bumped the generation makes the GUI drop the late
result.  Requests coalesce by ``key`` (only the newest survives the queue) and a
request superseded by a newer generation is skipped before the read.
"""

import logging
from collections import deque
from threading import Condition

from pyqtgraph import Qt

from xdart.modules.scan_aggregate import (
    whole_scan_aggregate_1d,
    whole_scan_aggregate_2d,
)

logger = logging.getLogger(__name__)


class AggregationWorker(Qt.QtCore.QThread):
    """One persistent thread computing whole-scan aggregates off the GUI thread.

    ``request(key, generation, scan, dim, method, norm_channel)`` enqueues;
    ``run`` pops, computes :func:`whole_scan_aggregate_1d`/``_2d``, and emits
    ``sigAggregated(key, generation, result)`` (``result`` may be ``None`` when
    nothing is on disk yet / the read failed — the GUI caches that too so it
    doesn't re-dispatch every render).  ``stop()`` drains and joins."""

    #: (key, generation, result) — key echoes the request, generation gates
    #: staleness, result is an Aggregated1D/2D or None.
    sigAggregated = Qt.QtCore.Signal(object, int, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cond = Condition()
        self._queue: deque = deque()
        self._newest_gen = -1
        self._stop = False

    def request(self, key, generation, scan, dim, method, norm_channel) -> None:
        """Enqueue an aggregation (non-blocking).  Coalesces by ``key``: a queued
        request for the same key is dropped in favour of this newer one."""
        generation = int(generation)
        with self._cond:
            if self._stop:
                return
            if generation > self._newest_gen:
                self._newest_gen = generation
            self._queue = deque(
                item for item in self._queue if item[0] != key)
            self._queue.append((key, generation, scan, dim, method, norm_channel))
            self._cond.notify()

    def run(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                key, generation, scan, dim, method, norm_channel = self._queue.popleft()
                newest = self._newest_gen
            if generation < newest:
                # A newer selection/mode superseded this request before we read.
                continue
            try:
                fn = whole_scan_aggregate_2d if dim == "2d" else whole_scan_aggregate_1d
                result = fn(scan, method=method, norm_channel=norm_channel)
            except Exception:
                logger.debug("background aggregation failed for %s", key,
                             exc_info=True)
                result = None
            # The GUI handler re-checks generation == the live display_generation.
            self.sigAggregated.emit(key, generation, result)

    def stop(self, timeout_ms: int = 8000) -> bool:
        """Signal the loop to exit and join (idempotent).  Returns ``True`` iff
        the thread stopped within ``timeout_ms`` (a ``False`` means an in-flight
        read is still running — the caller keeps the handle, P1)."""
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        if self.isRunning():
            return bool(self.wait(timeout_ms))
        return True
