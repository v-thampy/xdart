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
from xrd_tools.reduction import Frame, ReductionPlan, ReductionSession, Scan
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
        self._reg = {}

    def register(self, live):
        self._reg[int(live.idx)] = live
        self.registered.append(int(live.idx))

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

    def _flush(self, *, force=False):
        self.flushes += 1


def _adapter(sink, *, host=None, n_workers=2):
    session = ReductionSession(
        ReductionPlan(integration_2d=None),
        Scan("a", [Frame(0, image=np.zeros((2, 2)))], integrator=object()),
        sink=sink, execution="streaming", executor=n_workers,
    )
    host = host or SimpleNamespace(showLabel=SimpleNamespace(emit=lambda m: None),
                                   command_lock=threading.RLock(), command='start')
    return ScanSessionAdapter(host, session.scan, session, sink), session, host


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


def test_adapter_resume_is_noop_without_pause(monkeypatch):
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))
    adapter, session, _ = _adapter(_DuckSink())
    adapter.resume()                                # no-op, no error
    assert not adapter.is_paused
    session.finish()
