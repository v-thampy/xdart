# -*- coding: utf-8 -*-
"""Tests for QtNexusSink — xdart's v2 .nxs write driven through the ssrl
ReductionSink interface (the streaming write path)."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest


def _r1d(value, nq=16):
    from ssrl_xrd_tools.core.containers import IntegrationResult1D
    return IntegrationResult1D(
        radial=np.linspace(0.5, 5.0, nq, dtype=np.float32),
        intensity=np.full(nq, float(value), dtype=np.float32),
        sigma=np.ones(nq, dtype=np.float32),
        unit="q_A^-1",
    )


def _r2d(value, nq=16, nchi=6):
    from ssrl_xrd_tools.core.containers import IntegrationResult2D
    return IntegrationResult2D(
        radial=np.linspace(0.5, 5.0, nq, dtype=np.float32),
        azimuthal=np.linspace(-180, 180, nchi, endpoint=False).astype(np.float32),
        intensity=np.full((nq, nchi), float(value), dtype=np.float32),
        sigma=None, unit="q_A^-1")


def _reduction(idx, *, with_2d=False):
    from ssrl_xrd_tools.reduction.core import FrameReduction
    return FrameReduction(frame_index=idx, result_1d=_r1d(idx + 1),
                          result_2d=_r2d(idx + 1) if with_2d else None)


def _headless(idx):
    # The bits QtNexusSink / _frame_norm read off the ssrl Frame.
    return SimpleNamespace(index=idx, metadata={}, normalization_factor=None)


def _live_frame(idx):
    from xdart.modules.ewald.frame import LiveFrame
    fr = LiveFrame(idx=idx, map_raw=np.full((8, 8), idx + 1, dtype=np.float32))
    fr.bg_raw = 0
    fr.scan_info = {"i0": float(idx + 1)}
    fr.source_file = ""
    fr.source_frame_idx = 0
    fr.skip_map_raw = True
    return fr


class _FakeHost:
    """Minimal imageWranglerThread stand-in for the sink's shared machinery."""

    def __init__(self, *, xye_only=False, batch_mode=True, live_save_interval=1000):
        self.xye_only = xye_only
        self.batch_mode = batch_mode
        self._lsi = live_save_interval
        self.gi = False
        self.incidence_motor = None
        self.series_average = False
        self.file_lock = threading.RLock()
        self._xye_lock = threading.RLock()
        self._xye_buffer = []
        self.sigUpdate = SimpleNamespace(emit=lambda idx=None: self._signals.append(idx))
        self._signals = []
        self.xye_written = []
        self.data_1d = {}
        self.data_2d = {}
        self._published_frames = {}

    @property
    def LIVE_SAVE_INTERVAL(self):
        return self._lsi

    def _flush_xye_buffer(self, scan, published_idxs=None):
        with self._xye_lock:
            buf, self._xye_buffer = self._xye_buffer, []
        for idx, _frame in buf:
            if published_idxs is None or int(idx) in published_idxs:
                self.xye_written.append(int(idx))


def _minimal_plan():
    from ssrl_xrd_tools.reduction import ReductionPlan
    return ReductionPlan(integration_2d=None)


def _drive(sink, host, n):
    """Register + write n frames through the sink, then finish."""
    sink.begin(None, None)
    for i in range(n):
        live = _live_frame(i)
        sink.register(live)
        sink.write(_headless(i), _reduction(i))
    sink.finish(SimpleNamespace(cancelled=False, n_processed=n))


def test_sink_writes_all_frames_to_nxs_and_pops_register(tmp_path):
    from xdart.modules.ewald import LiveScan
    from xdart.gui.tabs.static_scan.wranglers.qt_nexus_sink import QtNexusSink

    nxs = str(tmp_path / "scan.nxs")
    scan = LiveScan(data_file=nxs)
    scan.skip_2d = True
    scan.frames._in_memory_cap = 16        # cap < N < interval (persist-before-evict)
    host = _FakeHost(batch_mode=True)
    sink = QtNexusSink(host, scan, _minimal_plan(), mask=None)

    N = 25
    _drive(sink, host, N)

    # The register map is bounded: every frame is popped once written.
    assert sink._registry == {}
    # All frames persisted -> reload sees every one with its int_1d.
    reloaded = LiveScan(data_file=nxs)
    reloaded.load_from_h5()
    assert len(reloaded.frames.index) == N
    for i in range(N):
        fr = reloaded.frames[i]
        assert fr.int_1d is not None
        np.testing.assert_allclose(np.asarray(fr.int_1d.intensity)[0], float(i + 1),
                                   atol=1e-4)
    # XYE rows flushed for every published frame; one end-of-run refresh (-1),
    # and NO per-frame signals (batch is silent during the run).
    assert sorted(host.xye_written) == list(range(N))
    assert host._signals == [-1]


def test_sink_persist_before_evict_no_unsaved_eviction(tmp_path):
    """The streaming sink must flush before _in_memory evicts an unsaved frame —
    so even with N > cap and a high interval, reload has every frame."""
    from xdart.modules.ewald import LiveScan
    from xdart.gui.tabs.static_scan.wranglers.qt_nexus_sink import QtNexusSink

    nxs = str(tmp_path / "scan.nxs")
    scan = LiveScan(data_file=nxs)
    scan.skip_2d = True
    scan.frames._in_memory_cap = 16
    host = _FakeHost(batch_mode=True, live_save_interval=100000)  # interval >> N
    sink = QtNexusSink(host, scan, _minimal_plan(), mask=None)

    N = 60
    _drive(sink, host, N)
    # The cap bound forced flushes; never more than cap unsaved in memory.
    assert scan.frames.unsaved_in_memory_count() == 0
    reloaded = LiveScan(data_file=nxs)
    reloaded.load_from_h5()
    assert len(reloaded.frames.index) == N


def test_worker_process_makes_thumbnail_off_the_writer(tmp_path):
    """The thumbnail is made by worker_process (the parallel pool hook), so the
    single writer thread never has to -- this is what closes the streaming 2D
    gap vs chunked."""
    from xdart.gui.tabs.static_scan.wranglers.qt_nexus_sink import QtNexusSink
    from xdart.modules.ewald import LiveScan

    scan = LiveScan(data_file=str(tmp_path / "s.nxs"))
    scan.skip_2d = False                       # 2D mode -> thumbnail wanted
    sink = QtNexusSink(_FakeHost(batch_mode=True), scan, _minimal_plan(), mask=None)
    sink.begin(None, None)
    live = _live_frame(0)
    assert live.thumbnail is None
    sink.register(live)
    sink.worker_process(_headless(0), _reduction(0))
    assert live.thumbnail is not None          # made on the worker, not the writer


def test_live_mode_hands_off_via_published_frames(tmp_path):
    """#3 (display contract): in live (non-batch) mode the sink does the
    single-source-of-truth hand-off the serial path uses — it stashes the
    fully-hydrated LiveFrame into host._published_frames and emits a lightweight
    sigUpdate, doing NO copy_for_display / data_1d / data_2d work on the writer
    thread (the GUI-thread update_data consumer owns that, incl. the publication
    that the cake renders from).  worker_process still KEEPS map_raw for live."""
    from xdart.gui.tabs.static_scan.wranglers.qt_nexus_sink import QtNexusSink
    from xdart.modules.ewald import LiveScan

    scan = LiveScan(data_file=str(tmp_path / "s.nxs"))
    scan.skip_2d = False
    host = _FakeHost(batch_mode=False)          # live
    sink = QtNexusSink(host, scan, _minimal_plan(), mask=None)
    sink.begin(None, None)
    live = _live_frame(0)
    assert live.map_raw is not None
    sink.register(live)
    sink.worker_process(_headless(0), _reduction(0, with_2d=True))
    assert live.thumbnail is not None           # thumbnail made on the worker (2D)
    assert live.map_raw is not None             # raw KEPT for the live display
    sink.write(_headless(0), _reduction(0, with_2d=True))
    # Fully-hydrated frame handed off via _published_frames; per-frame sigUpdate
    # emitted.  The frame carries int_2d (so update_data can build the cake's
    # publication) and map_raw.
    assert host._published_frames.get(0) is live
    assert live.int_2d is not None and live.map_raw is not None
    assert 0 in host._signals                   # lightweight queued sigUpdate
    # The writer thread did NO display-cache work — update_data (GUI thread) owns
    # data_1d / data_2d / publication / scan_data now.
    assert 0 not in host.data_1d and 0 not in host.data_2d


def test_sink_xye_only_writes_xye_no_nxs(tmp_path):
    from xdart.gui.tabs.static_scan.wranglers.qt_nexus_sink import QtNexusSink
    from xdart.modules.ewald import LiveScan

    nxs = str(tmp_path / "scan_xye.nxs")
    scan = LiveScan(data_file=nxs)
    scan.skip_2d = True
    host = _FakeHost(xye_only=True, batch_mode=True)
    sink = QtNexusSink(host, scan, _minimal_plan(), mask=None)

    _drive(sink, host, 10)
    # XYE rows written; no frames stashed into the scan (.nxs untouched).
    assert sorted(host.xye_written) == list(range(10))
    assert len(scan.frames.index) == 0


def test_close_reduction_session_surfaces_write_failure():
    """BLOCKER 2 (xdart side): a fail-loud finish() at scan close surfaces the
    write failure (records it, shows a label, stops the run) instead of the old
    silent debug-swallow — and BOTH session slots are still closed even if the
    first raises."""
    from types import MethodType
    from xdart.gui.tabs.static_scan.wranglers.wrangler_widget import wranglerThread

    labels, closed = [], []

    def boom_finish():
        closed.append("first")
        raise RuntimeError("disk full")

    def ok_finish():
        closed.append("second")

    w = SimpleNamespace(
        _reduction_session=SimpleNamespace(finish=boom_finish),   # raises first
        _reduction_session_key="k",
        _streaming_session=SimpleNamespace(finish=ok_finish),     # must still close
        _streaming_sink=object(),
        _streaming_scan_id=1,
        _reduction_write_error=None,
        command="run",
        showLabel=SimpleNamespace(emit=lambda m: labels.append(m)),
    )
    w._close_reduction_session = MethodType(
        wranglerThread._close_reduction_session, w)

    w._close_reduction_session()

    assert isinstance(w._reduction_write_error, RuntimeError)   # recorded, not swallowed
    assert w.command == "stop"                                  # run halted
    assert labels and "Save FAILED" in labels[0]                # user-visible
    assert closed == ["first", "second"]                       # second closed despite first raising
    assert w._streaming_session is None and w._reduction_session is None


def test_resume_parity_streaming_nxs_matches_unpaused(tmp_path):
    """Pause spec acceptance (#2): a run PAUSED mid-stream — drain + flush the
    sink to .nxs at a frame boundary, then resume submitting on the same open
    session — produces the SAME .nxs as an un-paused run.  Pausing never drops or
    corrupts a frame.  The pause-time flush is modelled by sink._flush(force=True)
    partway through (exactly what _enter_pause calls after session.drain())."""
    from xdart.modules.ewald import LiveScan
    from xdart.gui.tabs.static_scan.wranglers.qt_nexus_sink import QtNexusSink

    N = 12

    def _run(nxs, pause_after=None):
        scan = LiveScan(data_file=nxs)
        scan.skip_2d = True
        sink = QtNexusSink(_FakeHost(batch_mode=True), scan, _minimal_plan(), mask=None)
        sink.begin(None, None)
        for i in range(N):
            sink.register(_live_frame(i))
            sink.write(_headless(i), _reduction(i))
            if pause_after is not None and i == pause_after:
                sink._flush(force=True)        # the pause-time flush at a boundary
        sink.finish(SimpleNamespace(cancelled=False, n_processed=N))
        return nxs

    unpaused = _run(str(tmp_path / "unpaused.nxs"))
    paused = _run(str(tmp_path / "paused.nxs"), pause_after=5)   # "pause" after 6 frames

    ra = LiveScan(data_file=unpaused); ra.load_from_h5()
    rb = LiveScan(data_file=paused);   rb.load_from_h5()
    assert list(ra.frames.index) == list(rb.frames.index) == list(range(N))
    for i in range(N):
        np.testing.assert_allclose(
            np.asarray(ra.frames[i].int_1d.intensity),
            np.asarray(rb.frames[i].int_1d.intensity),
            err_msg=f"frame {i} differs between paused and un-paused .nxs",
        )
