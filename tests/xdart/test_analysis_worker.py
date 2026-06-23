# -*- coding: utf-8 -*-
"""Analyzer framework Step 3: the latest-wins live analysis worker drives ANY
:class:`xrd_tools.analysis.runner.Analyzer` off the GUI thread, branches once on
``unit``, and gates stale results by generation.  Uses fake analyzers (no fitting
backend) so the worker contract is tested without lmfit."""
import threading
import time

import numpy as np
import pytest

from pyqtgraph import Qt
from xdart.gui.tabs.static_scan.analysis_worker import (
    BatchAnalysisWorker, LiveAnalysisWorker, _run_analyzer,
)
from xrd_tools.analysis.runner import AnalysisInput, AnalysisOutcome

_DIRECT = Qt.QtCore.Qt.ConnectionType.DirectConnection


@pytest.fixture(scope="module")
def qapp():
    from pyqtgraph.Qt import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


class FakeFrameAnalyzer:
    """Frame-unit Analyzer with no backend — records the thread it ran on and
    echoes the input label so the test can assert routing + off-thread work."""
    kind = "fake"
    unit = "frame"

    def __init__(self):
        self.threads = []

    def analyze(self, inp):
        self.threads.append(threading.get_ident())
        return AnalysisOutcome(label=inp.label, ok=True,
                               params={"center_0": float(inp.label)})

    def analyze_scan(self, inputs):
        raise AssertionError("frame-unit analyzer must not route to analyze_scan")


class FakeScanAnalyzer:
    """Scan-unit Analyzer: a SET of inputs -> one outcome."""
    kind = "fake_scan"
    unit = "scan"

    def analyze(self, inp):
        raise AssertionError("scan-unit analyzer must not route to analyze()")

    def analyze_scan(self, inputs):
        return AnalysisOutcome(label="scan", ok=True, params={"n": len(inputs)})


def _inp(label):
    return AnalysisInput(label=str(label), x=np.array([1.0, 2.0]),
                         y=np.array([1.0, 2.0]))


def test_run_analyzer_branches_once_on_unit():
    """The ONLY analysis-specific line: frame-unit -> analyze(); scan-unit ->
    analyze_scan([inp]).  This single branch is what keeps the worker reusable."""
    out_f = _run_analyzer(FakeFrameAnalyzer(), _inp(3))
    assert out_f.ok and out_f.label == "3"
    out_s = _run_analyzer(FakeScanAnalyzer(), _inp(3))
    assert out_s.ok and out_s.params["n"] == 1


def test_worker_runs_off_thread_and_echoes_generation(qapp):
    caller = threading.get_ident()
    analyzer = FakeFrameAnalyzer()
    done = threading.Event()
    emitted = []
    worker = LiveAnalysisWorker()
    worker.sigAnalyzed.connect(
        lambda label, gen, outcome:
            (emitted.append((label, gen, outcome)), done.set()), _DIRECT)
    worker.start()
    try:
        worker.request("7", 3, analyzer, _inp(7))
        assert done.wait(5.0), "worker never emitted sigAnalyzed"
        label, gen, outcome = emitted[-1]
        assert (label, gen) == ("7", 3)            # label + generation echoed
        assert outcome.ok and outcome.params["center_0"] == 7.0
        assert analyzer.threads and analyzer.threads[0] != caller  # ran OFF caller
    finally:
        stopped = worker.stop()
    assert stopped is True
    assert not worker.isRunning()


def test_worker_latest_wins_drops_superseded(qapp):
    """Two requests enqueued before the loop runs: only the NEWEST is analyzed
    (latest-wins coalescing + the generation gate), so the fit never lags."""
    analyzer = FakeFrameAnalyzer()
    done = threading.Event()
    emitted = []

    def on(label, gen, outcome):
        emitted.append((label, gen))
        if gen == 2:
            done.set()

    worker = LiveAnalysisWorker()
    worker.sigAnalyzed.connect(on, _DIRECT)
    worker.request("1", 1, analyzer, _inp(1))
    worker.request("2", 2, analyzer, _inp(2))      # supersedes gen-1
    worker.start()
    try:
        assert done.wait(5.0)
        time.sleep(0.1)                            # let a (wrong) gen-1 emit happen
    finally:
        worker.stop()
    gens = [g for _, g in emitted]
    assert 2 in gens
    assert 1 not in gens, f"superseded gen-1 should be dropped; got {emitted}"
    assert len(analyzer.threads) == 1              # only the newest was analyzed


def test_worker_emits_none_when_analyzer_raises(qapp):
    """A raising analyzer can't kill the loop; it emits outcome=None so the GUI
    slot knows the generation completed but draws nothing."""
    class BoomAnalyzer:
        kind = "boom"
        unit = "frame"

        def analyze(self, inp):
            raise RuntimeError("backend gone")

        def analyze_scan(self, inputs):
            raise RuntimeError("backend gone")

    done = threading.Event()
    emitted = []
    worker = LiveAnalysisWorker()
    worker.sigAnalyzed.connect(
        lambda l, g, o: (emitted.append((l, g, o)), done.set()), _DIRECT)
    worker.start()
    try:
        worker.request("1", 1, BoomAnalyzer(), _inp(1))
        assert done.wait(5.0)
        assert worker.isRunning()                  # one bad fit can't kill it
        assert emitted[-1][2] is None              # outcome is None on failure
    finally:
        worker.stop()


def test_request_after_stop_is_a_noop(qapp):
    analyzer = FakeFrameAnalyzer()
    emitted = []
    worker = LiveAnalysisWorker()
    worker.sigAnalyzed.connect(
        lambda l, g, o: emitted.append((l, g)), _DIRECT)
    worker.start()
    worker.stop()
    worker.request("5", 1, analyzer, _inp(5))      # after stop -> dropped
    time.sleep(0.2)
    assert emitted == []
    assert analyzer.threads == []


# ── Batch worker (every frame, in order, vs-frame table) ──────────────────


def test_batch_worker_runs_all_frames_and_returns_table(qapp):
    analyzer = FakeFrameAnalyzer()
    done = threading.Event()
    result = {}
    progress = []
    worker = BatchAnalysisWorker()
    worker.sigProgress.connect(lambda d, t: progress.append((d, t)), _DIRECT)
    worker.sigBatchDone.connect(
        lambda labels, cols: (result.update(labels=labels, cols=cols), done.set()),
        _DIRECT)
    worker.configure(analyzer, [_inp(0), _inp(1), _inp(2)])
    worker.start()
    try:
        assert done.wait(5.0), "batch worker never finished"
        assert result["labels"] == ["0", "1", "2"]
        assert result["cols"]["center_0"] == [0.0, 1.0, 2.0]   # vs-frame series
        assert progress[-1] == (3, 3)
    finally:
        worker.stop()


def test_batch_worker_cancel_emits_none(qapp):
    """Cancelling mid-run emits (None, None) so the GUI shows no partial plot."""
    started = threading.Event()
    done = threading.Event()
    result = {}

    class SlowAnalyzer:
        kind = "slow"
        unit = "frame"

        def analyze(self, inp):
            started.set()
            time.sleep(0.05)
            return AnalysisOutcome(label=inp.label, ok=True,
                                   params={"center_0": 1.0})

        def analyze_scan(self, inputs):
            raise AssertionError

    worker = BatchAnalysisWorker()
    worker.sigBatchDone.connect(
        lambda labels, cols: (result.update(labels=labels, cols=cols), done.set()),
        _DIRECT)
    worker.configure(SlowAnalyzer(), [_inp(i) for i in range(50)])
    worker.start()
    try:
        assert started.wait(2.0)
        worker.cancel()
        assert done.wait(5.0)
        assert result["labels"] is None and result["cols"] is None
    finally:
        worker.stop()


# ── ROI-stats worker (per-frame stats over a source -> computed columns) ───


def _img_stack():
    """3 frames: a bright 2x2 block whose mean rises 10, 20, 30."""
    out = []
    for f in range(3):
        im = np.zeros((4, 4), dtype=float)
        im[1:3, 1:3] = (f + 1) * 10.0
        out.append(im)
    return out


def test_roi_stats_worker_streams_and_returns_result(qapp):
    """Streams per-frame stats AND the completed series equals a direct headless
    run_roi_signals — the mini live≡batch spine for the ROI worker."""
    from xdart.gui.tabs.static_scan.analysis_worker import RoiStatsWorker
    from xrd_tools.analysis.plans import RoiSignal, run_roi_signals
    from xrd_tools.core.roi import RoiSpec
    from xrd_tools.sources import MemoryFrameSource

    src = MemoryFrameSource(_img_stack())
    sig = RoiSignal(roi=RoiSpec(center_x=1.5, center_y=1.5, width_x=2, width_y=2),
                    reducer="mean", name="roiA")
    done = threading.Event()
    streamed, progress, result = [], [], {}
    worker = RoiStatsWorker()
    worker.sigFrameStat.connect(
        lambda f, row: streamed.append((f, row["roiA"])), _DIRECT)
    worker.sigProgress.connect(lambda d, t: progress.append((d, t)), _DIRECT)
    worker.sigRoiDone.connect(
        lambda res: (result.update(res=res), done.set()), _DIRECT)
    worker.configure((sig,), src)
    worker.start()
    try:
        assert done.wait(5.0), "ROI worker never finished"
        assert [f for f, _ in streamed] == [0, 1, 2]
        assert progress[-1] == (3, 3)
        direct = run_roi_signals((sig,), src).payload
        np.testing.assert_allclose(result["res"].payload.series["roiA"],
                                   direct.series["roiA"])
        np.testing.assert_allclose([v for _, v in streamed], direct.series["roiA"])
    finally:
        worker.stop()


def test_roi_stats_worker_cancel_emits_none(qapp):
    """Cancelling mid-run emits None (the dialog then abandons partial columns)."""
    from xdart.gui.tabs.static_scan.analysis_worker import RoiStatsWorker
    from xrd_tools.analysis.plans import RoiSignal
    from xrd_tools.core.roi import RoiSpec
    from xrd_tools.sources import MemoryFrameSource

    started = threading.Event()
    done = threading.Event()
    result = {}

    class _SlowSource(MemoryFrameSource):
        def load_frame(self, index):
            started.set()
            time.sleep(0.05)
            return super().load_frame(index)

    src = _SlowSource([np.zeros((4, 4)) for _ in range(50)])
    worker = RoiStatsWorker()
    worker.sigRoiDone.connect(
        lambda res: (result.update(res=res), done.set()), _DIRECT)
    worker.configure((RoiSignal(roi=RoiSpec.full_frame(), name="full"),), src)
    worker.start()
    try:
        assert started.wait(2.0)
        worker.cancel()
        assert done.wait(5.0)
        assert result["res"] is None
    finally:
        worker.stop()
