# -*- coding: utf-8 -*-
"""Phase 4c-1 — ScanSessionAdapter contract (offscreen, no Qt event loop).

Drives a REAL streaming ReductionSession through the adapter on a duck
host + duck sink: register+submit, quiesce(writer idle)+resume continuation,
and the stop-on-write-failure translation that must never raise into the
wrangler run() loop.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np

from xrd_tools.core.containers import IntegrationResult1D
from xrd_tools.reduction import Frame, ReductionPlan, Scan
from xrd_tools.session import ScanSession
import xrd_tools.reduction.core as reduction_core
from xdart.gui.tabs.static_scan.wranglers.scan_session import ScanSessionAdapter


def _r1d(v):
    return IntegrationResult1D(radial=np.array([0.0, 1.0]),
                              intensity=np.array([v, v + 1.0]),
                              sigma=None, unit="q_A^-1")


def _live(idx):
    """Duck LiveFrame with the attrs frame_from_live_frame reads."""
    return SimpleNamespace(idx=idx, map_raw=np.full((2, 2), idx, dtype=float),
                           bg_raw=None, scan_info={"i0": float(idx + 1)},
                           source_file="", source_frame_idx=0, mask=None)


class _DuckSink:
    """Records register/write/flush; the writer-thread single-writer sink."""
    def __init__(self, boom=False):
        self.boom = boom
        self.registered, self.written, self.flushes = [], [], 0
        self.unregistered = []
        self._reg = {}

    def register(self, live):
        self._reg[int(live.idx)] = live
        self.registered.append(int(live.idx))

    def unregister(self, index):
        self._reg.pop(int(index), None)
        self.unregistered.append(int(index))

    def begin(self, scan, plan):
        pass

    def write(self, frame, reduction):
        if self.boom:
            raise RuntimeError("disk full")
        self.written.append(int(frame.index))

    def finish(self, result):
        pass

    def abort(self, result):
        pass

    def flush(self, *, force=False):
        self.flushes += 1


def _adapter(sink, *, host=None, n_workers=2):
    # codex P2: drive the adapter through the PUBLIC xrd_tools.session.ScanSession
    # (the production object after the 4f-bridge), not a bare ReductionSession,
    # so wrapper-specific behaviour (_EventSink hook forwarding, submit()->bool,
    # flush contract, pause/resume) is exercised by the contract tests.
    scan = Scan("a", [Frame(0, image=np.zeros((2, 2)))], integrator=object())
    session = ScanSession(
        ReductionPlan(integration_2d=None), scan,
        sink=sink, executor=n_workers,
    )
    host = host or SimpleNamespace(showLabel=SimpleNamespace(emit=lambda m: None),
                                   command_lock=threading.RLock(), command='start')
    return ScanSessionAdapter(host, scan, session, sink), session, host


def test_adapter_submits_all_frames(monkeypatch):
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))
    sink = _DuckSink()
    adapter, session, host = _adapter(sink)
    for i in range(6):
        assert adapter.submit(_live(i)) is True
    session.finish()
    assert sorted(sink.written) == [0, 1, 2, 3, 4, 5]
    assert sorted(sink.registered) == [0, 1, 2, 3, 4, 5]


def test_adapter_quiesce_then_resume_continues(monkeypatch):
    """quiesce() drains (writer idle), flush fires, resume() re-allows submit;
    the run completes with all frames."""
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))
    sink = _DuckSink()
    adapter, session, host = _adapter(sink)
    for i in range(3):
        adapter.submit(_live(i))
    assert adapter.quiesce(timeout=10) is True
    assert adapter.is_paused
    assert sorted(sink.written) == [0, 1, 2]      # writer idle: all written
    adapter.flush()
    assert sink.flushes >= 1
    adapter.resume()
    assert not adapter.is_paused
    for i in range(3, 6):
        assert adapter.submit(_live(i)) is True    # not rejected after resume
    session.finish()
    assert sorted(sink.written) == [0, 1, 2, 3, 4, 5]


def test_adapter_submit_failure_stops_without_raising(monkeypatch):
    """A sink write failure (re-raised at submit's fail-loud precheck) is
    translated to command='stop' and a False return — never a raise that
    would tear down the wrangler QThread."""
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))
    sink = _DuckSink(boom=True)
    host = SimpleNamespace(showLabel=SimpleNamespace(emit=lambda m: None),
                           command_lock=threading.RLock(), command='start')
    adapter, session, _ = _adapter(sink, host=host)
    # Submit until the writer records the failure and the next submit's
    # fail-loud precheck re-raises it (caught by the adapter -> stop).
    stopped = False
    for i in range(8):
        if not adapter.submit(_live(i)):
            stopped = True
            break
    assert stopped
    assert host.command == 'stop'
    session.finish(raise_on_failure=False)


def test_adapter_unregisters_a_dropped_frame(monkeypatch):
    """A frame the session DROPS (submit returns False because the session was
    cancelled mid-submit) is rolled back out of the sink registration rather
    than left pinned until finish() (review #4 / codex pre-bridge cleanup)."""
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))
    sink = _DuckSink()
    adapter, session, _ = _adapter(sink)
    session.stop()                               # cooperative cancel -> next submit dropped
    assert adapter.submit(_live(0)) is False     # dropped, no raise
    assert 0 in sink.registered                  # it WAS registered first...
    assert 0 in sink.unregistered                # ...then rolled back
    session.finish(raise_on_failure=False)


def test_adapter_resume_is_noop_without_pause(monkeypatch):
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))
    adapter, session, _ = _adapter(_DuckSink())
    adapter.resume()                                # no-op, no error
    assert not adapter.is_paused
    session.finish()
