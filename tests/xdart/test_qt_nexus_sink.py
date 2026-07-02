# -*- coding: utf-8 -*-
"""Tests for QtNexusSink — xdart's v2 .nxs write driven through the ssrl
ReductionSink interface (the streaming write path)."""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest


def _r1d(value, nq=16, *, unit="q_A^-1"):
    from xrd_tools.core.containers import IntegrationResult1D
    return IntegrationResult1D(
        radial=np.linspace(0.5, 5.0, nq, dtype=np.float32),
        intensity=np.full(nq, float(value), dtype=np.float32),
        sigma=np.ones(nq, dtype=np.float32),
        unit=unit,
    )


def _r2d(value, nq=16, nchi=6, *, unit="q_A^-1", azimuthal_unit="chi_deg"):
    from xrd_tools.core.containers import IntegrationResult2D
    return IntegrationResult2D(
        radial=np.linspace(0.5, 5.0, nq, dtype=np.float32),
        azimuthal=np.linspace(-180, 180, nchi, endpoint=False).astype(np.float32),
        intensity=np.full((nq, nchi), float(value), dtype=np.float32),
        sigma=None, unit=unit, azimuthal_unit=azimuthal_unit)


def _reduction(idx, *, with_2d=False):
    from xrd_tools.reduction.core import FrameReduction
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

    @contextmanager
    def _h5pool_bracket(self, scan):
        # mirrors imageThread._h5pool_bracket — pause/resume the real h5 pool
        # around the QtNexusSink write (the same path the pre-DRY flush took).
        from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
            _get_h5pool)
        _get_h5pool().pause(scan.data_file)
        try:
            yield
        finally:
            _get_h5pool().resume(scan.data_file)


def _minimal_plan():
    from xrd_tools.reduction import ReductionPlan
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


def test_qt_sink_persists_accumulated_gi_modes(tmp_path):
    from xdart.modules.ewald import LiveScan
    from xdart.gui.tabs.static_scan.wranglers.qt_nexus_sink import QtNexusSink
    from xrd_tools.io import read_frame_record
    from xrd_tools.reduction.core import FrameReduction

    nxs = str(tmp_path / "gi_modes.nxs")
    scan = LiveScan(data_file=nxs)
    scan.gi = True
    scan.skip_2d = False
    scan.bai_1d_args["gi_mode_1d"] = "q_total"
    scan.bai_2d_args["gi_mode_2d"] = "qip_qoop"
    host = _FakeHost(batch_mode=True)
    host.gi = True
    host.incidence_motor = "th"
    sink = QtNexusSink(host, scan, _minimal_plan(), mask=None)
    sink.begin(scan, _minimal_plan())

    active_1d = _r1d(1.0, unit="q_A^-1")
    extra_1d = _r1d(9.0, unit="qip_A^-1")
    active_2d = _r2d(2.0, unit="qip_A^-1", azimuthal_unit="qoop_A^-1")
    extra_2d = _r2d(7.0, unit="q_A^-1", azimuthal_unit="chi_deg")
    live = _live_frame(0)
    live.gi = True
    live.scan_info = {"th": 0.2, "i0": 1.0}
    live.gi_1d = {"qtotal": active_1d, "qip": extra_1d}
    live.gi_2d = {"gi2d": active_2d, "polar": extra_2d}
    sink.register(live)
    sink.write(
        _headless(0),
        FrameReduction(
            frame_index=0,
            result_1d=active_1d,
            result_2d=active_2d,
            mode_1d="q_total",
            mode_2d="qip_qoop",
        ),
    )
    sink.flush(force=True)

    rec = read_frame_record(nxs, 0)
    assert set(rec.modes_1d) == {"q_total", "q_ip"}
    assert set(rec.modes_2d) == {"qip_qoop", "q_chi"}
    np.testing.assert_allclose(rec.view_1d("q_ip").intensity_1d, extra_1d.intensity)
    np.testing.assert_allclose(rec.view_2d("q_chi").intensity_2d, extra_2d.intensity.T)


def test_first_forced_batch_flush_replaces_skeleton_atomically(tmp_path, monkeypatch):
    """The first real batch save after initialize_scan's skeleton file should
    use the writer's atomic ``mode='w'`` path.  This avoids growing integrated
    stacks in place while the GUI may already be browsing the skeleton."""
    from xdart.modules.ewald import LiveScan
    from xdart.gui.tabs.static_scan.wranglers.qt_nexus_sink import QtNexusSink
    import xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread as iwt

    class _Pool:
        def pause(self, path):
            pass

        def resume(self, path):
            pass

    monkeypatch.setattr(iwt, "_get_h5pool", lambda: _Pool())

    scan = LiveScan(data_file=str(tmp_path / "scan.nxs"))
    scan.skip_2d = True
    host = _FakeHost(batch_mode=True)
    sink = QtNexusSink(host, scan, _minimal_plan(), mask=None)
    sink.begin(scan, _minimal_plan())
    for i in range(3):
        live = _live_frame(i)
        sink.register(live)
        sink.write(_headless(i), _reduction(i))

    modes = []

    def fake_save(*, mode="a", **kwargs):
        modes.append(mode)
        scan.frames.mark_persisted(scan.frames.index)

    monkeypatch.setattr(scan, "_save_to_nexus", fake_save)
    sink.flush(force=True)

    assert modes == ["w"]


def test_forced_batch_flush_appends_after_frames_are_persisted(tmp_path, monkeypatch):
    from xdart.modules.ewald import LiveScan
    from xdart.gui.tabs.static_scan.wranglers.qt_nexus_sink import QtNexusSink
    import xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread as iwt

    class _Pool:
        def pause(self, path):
            pass

        def resume(self, path):
            pass

    monkeypatch.setattr(iwt, "_get_h5pool", lambda: _Pool())

    scan = LiveScan(data_file=str(tmp_path / "scan.nxs"))
    scan.skip_2d = True
    host = _FakeHost(batch_mode=True)
    sink = QtNexusSink(host, scan, _minimal_plan(), mask=None)
    sink.begin(scan, _minimal_plan())
    live = _live_frame(0)
    sink.register(live)
    sink.write(_headless(0), _reduction(0))
    scan.frames.mark_persisted(scan.frames.index)

    modes = []

    def fake_save(*, mode="a", **kwargs):
        modes.append(mode)
        scan.frames.mark_persisted(scan.frames.index)

    monkeypatch.setattr(scan, "_save_to_nexus", fake_save)
    sink.flush(force=True)

    assert modes == ["a"]


def test_qt_sink_marks_record_store_persisted_after_nexus_save(tmp_path, monkeypatch):
    from xdart.modules.ewald import LiveScan
    from xdart.gui.tabs.static_scan.wranglers.qt_nexus_sink import QtNexusSink
    import xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread as iwt

    class _Pool:
        def pause(self, path):
            pass

        def resume(self, path):
            pass

    class _Store:
        def __init__(self):
            self.persisted = []

        def mark_persisted(self, labels, *, modes=None):
            events.append("record_store")
            self.persisted.append((set(labels), tuple(modes or ())))

    monkeypatch.setattr(iwt, "_get_h5pool", lambda: _Pool())

    scan = LiveScan(data_file=str(tmp_path / "scan.nxs"))
    scan.skip_2d = True
    host = _FakeHost(batch_mode=True)
    events = []
    store = _Store()
    sink = QtNexusSink(
        host, scan, _minimal_plan(), mask=None, record_store=store
    )
    sink.begin(scan, _minimal_plan())
    for i in range(2):
        live = _live_frame(i)
        sink.register(live)
        sink.write(_headless(i), _reduction(i))

    saved = []

    def fake_save(*, mode="a", **kwargs):
        events.append("nexus")
        saved.append(mode)
        scan.frames.mark_persisted(scan.frames.index)

    monkeypatch.setattr(scan, "_save_to_nexus", fake_save)
    assert store.persisted == []
    sink.flush(force=True)

    assert saved == ["w"]
    assert events == ["nexus", "record_store"]
    assert store.persisted == [({0, 1}, (("1d", "default"),))]


def test_qt_sink_marks_only_written_record_store_modes(tmp_path, monkeypatch):
    from xdart.modules.ewald import LiveScan
    from xdart.gui.tabs.static_scan.wranglers.qt_nexus_sink import QtNexusSink
    import xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread as iwt

    class _Pool:
        def pause(self, path):
            pass

        def resume(self, path):
            pass

    class _Store:
        def __init__(self):
            self.calls = []

        def mark_persisted(self, labels, *, modes=None):
            self.calls.append((tuple(labels), tuple(modes or ())))

    monkeypatch.setattr(iwt, "_get_h5pool", lambda: _Pool())

    scan = LiveScan(data_file=str(tmp_path / "scan.nxs"))
    scan.skip_2d = False
    host = _FakeHost(batch_mode=True)
    store = _Store()
    sink = QtNexusSink(
        host, scan, _minimal_plan(), mask=None, record_store=store
    )
    sink.begin(scan, _minimal_plan())
    live = _live_frame(0)
    sink.register(live)
    sink.write(_headless(0), _reduction(0, with_2d=True))

    def fake_save(*, mode="a", **kwargs):
        scan.frames.mark_persisted(scan.frames.index)
        return {"entry/integrated_2d": [0]}

    monkeypatch.setattr(scan, "_save_to_nexus", fake_save)
    sink.flush(force=True)

    assert store.calls == [((0,), (("1d", "default"),))]


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


def test_worker_process_uses_core_corrected_image_for_thumbnail(tmp_path):
    from xdart.gui.tabs.static_scan.wranglers.qt_nexus_sink import QtNexusSink
    from xdart.modules.ewald import LiveScan

    scan = LiveScan(data_file=str(tmp_path / "s.nxs"))
    scan.skip_2d = False
    sink = QtNexusSink(_FakeHost(batch_mode=True), scan, _minimal_plan(), mask=None)
    sink.begin(None, None)
    live = _live_frame(0)
    reduction = _reduction(0)
    reduction.corrected_image = np.full((8, 8), 7.0, dtype=np.float32)
    seen = {}

    def fake_thumbnail(*, global_mask=None, corrected_image=None):
        seen["global_mask"] = global_mask
        seen["corrected_image"] = corrected_image
        live.thumbnail = np.zeros((2, 2), dtype=np.float32)

    live.make_thumbnail = fake_thumbnail
    sink.register(live)
    sink.worker_process(_headless(0), reduction)

    assert live.thumbnail is not None
    assert seen["global_mask"] is None
    np.testing.assert_array_equal(seen["corrected_image"], reduction.corrected_image)


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

    def boom_finish(**kw):      # kw absorbs join_timeout=
        closed.append("first")
        raise RuntimeError("disk full")

    def ok_finish(**kw):
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


def test_close_reduction_session_reports_cancelled_unwritten_frames(caplog):
    from types import MethodType
    from xdart.gui.tabs.static_scan.wranglers.wrangler_widget import wranglerThread

    labels = []

    class _Session:
        frames_submitted = 5

        def finish(self, **kw):
            return SimpleNamespace(n_processed=3, cancelled=True)

    w = SimpleNamespace(
        _reduction_session=None,
        _reduction_session_key=None,
        _streaming_session=_Session(),
        _streaming_sink=object(),
        _streaming_scan_id=1,
        _streaming_record_store=object(),
        _scan_session_adapter=object(),
        _gi_prepass_scan_id=1,
        _reduction_write_error=None,
        command="stop",
        showLabel=SimpleNamespace(emit=lambda m: labels.append(m)),
    )
    w._close_reduction_session = MethodType(
        wranglerThread._close_reduction_session, w)

    with caplog.at_level(logging.INFO):
        w._close_reduction_session()

    assert labels == [
        "Stopped with 2 frame(s) un-written "
        "(submitted=5, written=3) — source data intact; re-run Append/batch to recover"
    ]
    assert "Total Files Processed (durable after cancel): 3" in caplog.text
    assert "submitted=5, written=3" in caplog.text


def test_resume_parity_streaming_nxs_matches_unpaused(tmp_path):
    """Pause spec acceptance (#2): a run PAUSED mid-stream — drain + flush the
    sink to .nxs at a frame boundary, then resume submitting on the same open
    session — produces the SAME .nxs as an un-paused run.  Pausing never drops or
    corrupts a frame.  The pause-time flush is modelled by sink.flush(force=True)
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
                sink.flush(force=True)        # the pause-time flush at a boundary
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


def test_finish_and_abort_clear_unwritten_registry(tmp_path):
    """T0-8: frames whose reduction failed/was cancelled mid-flight are never
    popped by write()/replace() — finish() and abort() must clear the registry
    so they don't pin LiveFrames (and their raw images) for the scan's life."""
    from types import SimpleNamespace
    from xdart.modules.ewald import LiveScan
    from xdart.gui.tabs.static_scan.wranglers.qt_nexus_sink import QtNexusSink

    for teardown in ("finish", "abort"):
        scan = LiveScan(data_file=str(tmp_path / f"reg_{teardown}.nxs"))
        scan.skip_2d = True
        host = _FakeHost(batch_mode=True)
        sink = QtNexusSink(host, scan, _minimal_plan(), mask=None)
        sink.begin(scan, _minimal_plan())
        # Two in-flight frames that will never reach write() (failed/cancelled).
        sink.register(SimpleNamespace(idx=0))
        sink.register(SimpleNamespace(idx=1))
        assert len(sink._registry) == 2

        getattr(sink, teardown)(result=None)

        assert sink._registry == {}, f"{teardown}() left frames pinned"


def test_write_without_registration_fails_loud(tmp_path):
    """P1b (codex pre-merge): a registry miss in write() must RAISE — silently
    returning made the session count the frame as written while its data never
    reached the .nxs or the display (fail-loud writer contract)."""
    from types import SimpleNamespace
    from xdart.modules.ewald import LiveScan
    from xdart.gui.tabs.static_scan.wranglers.qt_nexus_sink import QtNexusSink

    scan = LiveScan(data_file=str(tmp_path / "unregistered.nxs"))
    sink = QtNexusSink(_FakeHost(batch_mode=True), scan, _minimal_plan(), mask=None)
    sink.begin(scan, _minimal_plan())

    with pytest.raises(RuntimeError, match="register"):
        sink.write(SimpleNamespace(index=7), SimpleNamespace())


# ---------------------------------------------------------------------------
# Phase 1a — streaming conformance under a REAL session (thread discipline,
# replace semantics, abort flush, persist-before-evict threshold)
# ---------------------------------------------------------------------------

def _spy_session_drive(host, scan, lives, monkeypatch, *, executor=3):
    """Drive QtNexusSink through a real streaming ReductionSession wrapped
    in the thread-tracking spy from tests.core.contracts.

    The session drive loop (monkeypatched integrator + submit/finish) is the
    shared ``contracts.drive_streaming``; only the QtNexusSink-specific setup
    (register the LiveFrames, supply 4×4 frames + the 16-bin _r1d stub) stays
    here.
    """
    from tests.core.contracts import ThreadSpySink, drive_streaming
    from xdart.gui.tabs.static_scan.wranglers.qt_nexus_sink import QtNexusSink
    from xrd_tools.reduction import Frame

    sink = QtNexusSink(host, scan, _minimal_plan(), mask=None)
    for lv in lives:
        sink.register(lv)
    spy = ThreadSpySink(inner=sink)
    frames = [Frame(int(lv.idx), image=np.full((4, 4), int(lv.idx) + 1.0))
              for lv in lives]
    drive_streaming(
        spy, monkeypatch, frames=frames, plan=_minimal_plan(), executor=executor,
        integrate_1d=lambda image, ai, **kw: _r1d(float(np.sum(image))),
    )
    return spy, sink


def test_qt_sink_worker_process_runs_on_pool_workers(tmp_path, monkeypatch):
    """The PERF-5 thumbnail work fans out on POOL workers (never the writer
    thread, never the caller) while write() stays on the one writer thread —
    the discipline the .nxs single-writer design depends on, exercised
    through a real parallel session for the first time."""
    from xdart.modules.ewald import LiveScan

    scan = LiveScan(data_file=str(tmp_path / "wp.nxs"))
    scan.skip_2d = False                  # thumbnails must NOT be skipped
    host = _FakeHost(batch_mode=True)
    lives = [_live_frame(i) for i in range(8)]
    spy, sink = _spy_session_drive(host, scan, lives, monkeypatch)

    writer = spy.threads_for("write")
    workers = spy.threads_for("worker_process")
    assert len(writer) == 1
    assert workers and not (workers & writer)
    assert threading.get_ident() not in writer
    # the worker_process work product is real: every frame got its thumbnail
    for lv in lives:
        assert lv.thumbnail is not None, f"frame {lv.idx} missing thumbnail"
    assert sink._registry == {}


def test_qt_sink_replace_upserts_without_recounting(tmp_path):
    """replace() (re-fed index): hydrates + upserts the in-memory frame but
    does NOT advance the new-frame save counter and does NOT re-buffer the
    XYE row — the original write already did both."""
    from types import SimpleNamespace
    from xdart.modules.ewald import LiveScan
    from xdart.gui.tabs.static_scan.wranglers.qt_nexus_sink import QtNexusSink
    from xrd_tools.reduction.core import FrameReduction

    scan = LiveScan(data_file=str(tmp_path / "replace.nxs"))
    scan.skip_2d = True
    host = _FakeHost(batch_mode=True, live_save_interval=1000)
    sink = QtNexusSink(host, scan, _minimal_plan(), mask=None)
    sink.begin(scan, _minimal_plan())
    for i in range(3):
        live = _live_frame(i)
        sink.register(live)
        sink.write(_headless(i), _reduction(i))

    counter_before = sink._since_save
    with host._xye_lock:
        xye_before = len(host._xye_buffer)

    sink.replace(_headless(1), FrameReduction(frame_index=1,
                                              result_1d=_r1d(99.0)))

    assert float(scan.frames[1].int_1d.intensity[0]) == 99.0   # upserted
    assert sink._since_save == counter_before                  # not recounted
    with host._xye_lock:
        assert len(host._xye_buffer) == xye_before             # not re-buffered

    sink.finish(SimpleNamespace(cancelled=False, n_processed=3))


def test_qt_sink_abort_flushes_completed_frames(tmp_path):
    """abort(): completed frames are flushed to the .nxs (never deleted —
    the sink writes into the live file, not a temp), registry cleared."""
    from types import SimpleNamespace
    from xdart.modules.ewald import LiveScan
    from xdart.gui.tabs.static_scan.wranglers.qt_nexus_sink import QtNexusSink

    nxs = str(tmp_path / "abort.nxs")
    scan = LiveScan(data_file=nxs)
    scan.skip_2d = True
    host = _FakeHost(batch_mode=True, live_save_interval=1000)
    sink = QtNexusSink(host, scan, _minimal_plan(), mask=None)
    sink.begin(scan, _minimal_plan())
    for i in range(2):
        live = _live_frame(i)
        sink.register(live)
        sink.write(_headless(i), _reduction(i))
    sink.register(SimpleNamespace(idx=2))      # in-flight, never written

    sink.abort(SimpleNamespace(cancelled=True, n_processed=2))

    assert sink._registry == {}
    reloaded = LiveScan(data_file=nxs)
    reloaded.load_from_h5()
    assert sorted(int(i) for i in reloaded.frames.index) == [0, 1]
    assert reloaded.frames[0].int_1d is not None


def test_qt_sink_persist_before_evict_threshold(tmp_path, monkeypatch):
    """The save cadence fires BEFORE LiveFrameSeries could evict an unsaved
    frame: with cap C, the flush threshold is C - 8 (the margin), so the
    Nth unsaved in-memory frame can never reach the eviction boundary."""
    from types import SimpleNamespace
    from xdart.modules.ewald import LiveScan
    from xdart.gui.tabs.static_scan.wranglers.qt_nexus_sink import QtNexusSink

    scan = LiveScan(data_file=str(tmp_path / "evict.nxs"))
    scan.skip_2d = True
    scan.frames._in_memory_cap = 12            # threshold = 12 - 8 = 4
    host = _FakeHost(batch_mode=True, live_save_interval=1000)
    sink = QtNexusSink(host, scan, _minimal_plan(), mask=None)

    saves = []
    orig_save = scan._save_to_nexus
    monkeypatch.setattr(scan, "_save_to_nexus",
                        lambda *a, **k: (saves.append((sink._since_save,
                                                       k.get("mode", "a"))),
                                         orig_save(*a, **k))[1])

    sink.begin(scan, _minimal_plan())
    for i in range(3):                          # below threshold: no save
        live = _live_frame(i)
        sink.register(live)
        sink.write(_headless(i), _reduction(i))
    assert saves == []

    live = _live_frame(3)                       # 4th write crosses threshold
    sink.register(live)
    sink.write(_headless(3), _reduction(3))
    assert len(saves) == 1
    assert saves[0] == (4, "w")                 # first save is atomic
    assert sink._since_save == 0                # counter reset by the flush

    sink.finish(SimpleNamespace(cancelled=False, n_processed=4))
