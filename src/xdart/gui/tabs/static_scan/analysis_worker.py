# -*- coding: utf-8 -*-
"""Background analysis workers (analyzer framework Steps 3-4).

Off-GUI-thread runners that drive ANY :class:`xrd_tools.analysis.runner.Analyzer`
(peak fit now; phase fit / sin2psi later) — the workers never know which.

* :class:`LiveAnalysisWorker` — latest-wins live preview.  ``request`` coalesces
  to the single newest job and a generation gate drops a job superseded before
  it runs, so the fit always tracks the newest frame and never builds a backlog
  (frames arrive ~1 s apart in practice, far slower than a fit, so most are
  caught; when the fitter lags it simply skips to the freshest).  Modelled on
  :class:`AggregationWorker`.

The analysis itself is a READ-ONLY consumer of already-published frame data; it
never touches the integration pipeline.
"""

import logging
from collections import deque
from threading import Condition

from pyqtgraph import Qt

from xrd_tools.analysis.plans import run_roi_signals
from xrd_tools.analysis.runner import batch_params_table, run_batch

logger = logging.getLogger(__name__)


def _run_analyzer(analyzer, inp):
    """Branch ONCE on granularity — the only analysis-specific line in the
    workers.  Frame-unit analyzers fit one pattern; scan-unit analyzers
    (sin2psi/texture) consume the set (here, the single current input)."""
    if getattr(analyzer, "unit", "frame") == "scan":
        return analyzer.analyze_scan([inp])
    return analyzer.analyze(inp)


class LiveAnalysisWorker(Qt.QtCore.QThread):
    """Latest-wins live analysis thread.

    ``request(label, generation, analyzer, inp)`` enqueues (coalescing to the
    newest job); ``run`` drops a job whose generation has been superseded, else
    runs the analyzer and emits ``sigAnalyzed(label, generation, outcome)``
    (``outcome`` is ``None`` on failure).  ``stop()`` drains and joins."""

    #: (label, generation, outcome) — generation gates staleness in the GUI slot.
    sigAnalyzed = Qt.QtCore.Signal(object, int, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cond = Condition()
        self._queue: deque = deque()
        self._newest_gen = -1
        self._stop = False

    def request(self, label, generation, analyzer, inp) -> None:
        """Enqueue an analysis (non-blocking).  Coalesces to the single newest
        job — latest-wins, so an in-flight fit is never starved by a backlog."""
        generation = int(generation)
        with self._cond:
            if self._stop:
                return
            if generation > self._newest_gen:
                self._newest_gen = generation
            self._queue.clear()
            self._queue.append((label, generation, analyzer, inp))
            self._cond.notify()

    def run(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                label, generation, analyzer, inp = self._queue.popleft()
                newest = self._newest_gen
            if generation < newest:
                continue  # a newer frame superseded this before we fit it
            try:
                outcome = _run_analyzer(analyzer, inp)
            except Exception:
                logger.debug("live analysis failed", exc_info=True)
                outcome = None
            self.sigAnalyzed.emit(label, generation, outcome)

    def stop(self, timeout_ms: int = 4000) -> bool:
        """Signal the loop to exit and join (idempotent)."""
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        if self.isRunning():
            return bool(self.wait(timeout_ms))
        return True


class BatchAnalysisWorker(Qt.QtCore.QThread):
    """Run an analyzer across EVERY frame's pattern off the GUI thread.

    Unlike :class:`LiveAnalysisWorker` (latest-wins), batch processes the full
    input set in order — emitting progress and returning the per-frame parameter
    table for the vs-frame plot.  A thin thread around the headless
    :func:`run_batch` / :func:`batch_params_table`; cancellable between frames."""

    #: (done, total) progress after each frame.
    sigProgress = Qt.QtCore.Signal(int, int)
    #: (label, params) streamed per frame — grows the live vs-frame trend.
    sigFrameFit = Qt.QtCore.Signal(object, object)
    #: (labels, columns) on completion — or (None, None) if cancelled.
    sigBatchDone = Qt.QtCore.Signal(object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._analyzer = None
        self._inputs = []
        self._cancel = False

    def configure(self, analyzer, inputs) -> None:
        """Set the analyzer + the per-frame inputs for the next ``start()``."""
        self._analyzer = analyzer
        self._inputs = list(inputs)
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        outcomes = run_batch(
            self._analyzer, self._inputs,
            on_progress=lambda done, total: self.sigProgress.emit(done, total),
            on_frame=lambda out: self.sigFrameFit.emit(out.label, out.params),
            should_cancel=lambda: self._cancel)
        if self._cancel:
            self.sigBatchDone.emit(None, None)   # cancelled -> no partial plot
            return
        labels, columns = batch_params_table(outcomes)
        self.sigBatchDone.emit(labels, columns)

    def stop(self, timeout_ms: int = 8000) -> bool:
        """Request cancel + join (idempotent).  Longer default than the live
        worker — a frame's fit must finish before the loop re-checks cancel."""
        self._cancel = True
        if self.isRunning():
            return bool(self.wait(timeout_ms))
        return True


class RoiStatsWorker(Qt.QtCore.QThread):
    """Reduce ROI signals over EVERY raw frame of a source off the GUI thread.

    The ROI counterpart of :class:`BatchAnalysisWorker`: a thin thread around the
    headless :func:`xrd_tools.analysis.run_roi_signals`, streaming each frame's
    per-ROI stats so the Scan Plot table fills its computed columns
    incrementally, with progress + cancel.  Loading each raw frame is the real
    cost (the reduce is cheap), so it runs OFF the GUI thread; the math stays
    headless."""

    #: (done, total) progress after each frame.
    sigProgress = Qt.QtCore.Signal(int, int)
    #: (frame_index, {column_name: value}) streamed per frame — grows the columns.
    sigFrameStat = Qt.QtCore.Signal(object, object)
    #: the completed AnalysisResult on success, or ``None`` if cancelled.
    sigRoiDone = Qt.QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._signals = ()
        self._source = None
        self._x_key = None
        self._mask_saturation = False
        self._cancel = False

    def configure(self, signals, source, *, x_key=None,
                  mask_saturation=False) -> None:
        """Set the ROI signals + the raw source for the next ``start()``."""
        self._signals = tuple(signals)
        self._source = source
        self._x_key = x_key
        self._mask_saturation = bool(mask_saturation)
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            result = run_roi_signals(
                self._signals, self._source,
                x_key=self._x_key, mask_saturation=self._mask_saturation,
                on_progress=lambda done, total: self.sigProgress.emit(done, total),
                on_frame=lambda f, row: self.sigFrameStat.emit(f, row),
                should_cancel=lambda: self._cancel)
        except Exception:
            logger.exception("ROI-stats worker failed")
            self.sigRoiDone.emit(None)
            return
        if self._cancel:
            self.sigRoiDone.emit(None)   # cancelled -> abandon the partial columns
            return
        self.sigRoiDone.emit(result)

    def stop(self, timeout_ms: int = 8000) -> bool:
        """Request cancel + join (idempotent)."""
        self._cancel = True
        if self.isRunning():
            return bool(self.wait(timeout_ms))
        return True
