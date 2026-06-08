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


def _reduction(idx):
    from ssrl_xrd_tools.reduction.core import FrameReduction
    return FrameReduction(frame_index=idx, result_1d=_r1d(idx + 1), result_2d=None)


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


def test_live_mode_publishes_display_and_keeps_raw(tmp_path):
    """#3: in live (non-batch) mode the sink does the per-frame display publish
    (data_1d/data_2d + sigUpdate) and worker_process KEEPS map_raw (the 2D panel
    reads it), unlike batch which frees it."""
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
    sink.worker_process(_headless(0), _reduction(0))
    assert live.thumbnail is not None           # thumbnail made (2D)
    assert live.map_raw is not None             # but raw kept for the display
    sink.write(_headless(0), _reduction(0))
    # per-frame display publish happened + the frame's raw is in data_2d.
    assert 0 in host.data_1d and 0 in host.data_2d
    assert host.data_2d[0]['map_raw'] is not None
    assert 0 in host._signals                   # per-frame sigUpdate emitted


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
