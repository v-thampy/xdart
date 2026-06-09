"""Tests for the streaming sink-driven execution mode on ReductionSession.

The chunked path (``process``) is the reference oracle: streaming must produce
the same per-frame products, regardless of worker completion order, while
bounding the in-flight window and driving the sink from exactly one thread.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from ssrl_xrd_tools.core.containers import IntegrationResult1D
from ssrl_xrd_tools.reduction import (
    CancelToken,
    Frame,
    MemorySink,
    ReductionPlan,
    Scan,
    ReductionSession,
    run_reduction,
)
import ssrl_xrd_tools.reduction.core as reduction_core


def _r1d(value: float) -> IntegrationResult1D:
    return IntegrationResult1D(
        radial=np.array([0.0, 1.0]),
        intensity=np.array([value, value + 1.0]),
        sigma=None,
        unit="q_A^-1",
    )


def _frames(n: int) -> list[Frame]:
    return [Frame(i, image=np.full((2, 2), i, dtype=float)) for i in range(n)]


def _plan() -> ReductionPlan:
    return ReductionPlan(integration_2d=None)


def _stream(plan, frames, sink, **kw):
    """Open a streaming session, submit every frame, finish.  Returns (session,
    result) so callers can assert on counters."""
    session = ReductionSession(
        plan, Scan("stream", frames, integrator=object()),
        sink=sink, execution="streaming", **kw,
    )
    for fr in frames:
        session.submit(fr)
    return session, session.finish()


# ---------------------------------------------------------------------------
# Equivalence with the chunked oracle
# ---------------------------------------------------------------------------

def test_streaming_output_matches_chunked(monkeypatch):
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))
    plan = _plan()
    chunked = run_reduction(plan, Scan("c", _frames(8), integrator=object()),
                            executor=3, chunk_size=8)
    sink = MemorySink()
    session, result = _stream(plan, _frames(8), sink, executor=3)

    assert sorted(sink.frames) == sorted(chunked.frames) == list(range(8))
    assert result.n_processed == 8
    assert session.integrator_provider_builds == 1   # persistent provider
    for idx in chunked.frames:
        np.testing.assert_allclose(
            sink.frames[idx].result_1d.intensity,
            chunked.frames[idx].result_1d.intensity,
        )


def test_streaming_correct_under_scrambled_completion(monkeypatch):
    """Make low-index frames finish LAST (sleep longer) so completion order is
    the reverse of submission order — the index-addressed writer must still
    pair every result with its own frame."""
    def slow(image, ai, **kw):
        v = float(np.sum(image))           # frame i -> 4*i
        time.sleep(0.002 * (40 - v))       # frame 0 slowest, frame 9 fastest
        return _r1d(v)
    monkeypatch.setattr(reduction_core, "integrate_1d", slow)

    sink = MemorySink()
    _stream(_plan(), _frames(10), sink, executor=5)
    for i in range(10):
        expected = float(np.sum(np.full((2, 2), i)))   # 4*i
        np.testing.assert_allclose(
            sink.frames[i].result_1d.intensity, [expected, expected + 1.0],
        )


# ---------------------------------------------------------------------------
# Bounded in-flight window
# ---------------------------------------------------------------------------

def test_streaming_respects_inflight_bound(monkeypatch):
    lock = threading.Lock()
    state = {"cur": 0, "peak": 0}

    def tracked(image, ai, **kw):
        with lock:
            state["cur"] += 1
            state["peak"] = max(state["peak"], state["cur"])
        time.sleep(0.01)
        with lock:
            state["cur"] -= 1
        return _r1d(float(np.sum(image)))

    monkeypatch.setattr(reduction_core, "integrate_1d", tracked)
    sink = MemorySink()
    session, _ = _stream(_plan(), _frames(24), sink, executor=8, inflight_max=3)

    # In-flight (submitted-but-unwritten) is capped, so no more than
    # inflight_max integrations can be running at once.
    assert state["peak"] <= 3
    assert session.inflight_max == 3
    assert len(sink.frames) == 24


def test_streaming_default_inflight_is_twice_workers(monkeypatch):
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))
    sink = MemorySink()
    session, _ = _stream(_plan(), _frames(4), sink, executor=4)
    assert session.inflight_max == 8     # 2 x workers


# ---------------------------------------------------------------------------
# Replace idempotency (A1) — re-fed index doesn't double-count
# ---------------------------------------------------------------------------

def test_streaming_replace_is_idempotent(monkeypatch):
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))
    plan = _plan()
    frames = _frames(3)
    sink = MemorySink()
    session = ReductionSession(
        plan, Scan("s", frames, integrator=object()),
        sink=sink, execution="streaming", executor=2,
    )
    for fr in frames:
        session.submit(fr)
    # Re-feed index 1 (reintegration) — must be a replace, not a new completion.
    session.submit(Frame(1, image=np.full((2, 2), 1, dtype=float)))
    result = session.finish()

    assert result.n_processed == 3       # distinct frames, not 4
    assert sorted(sink.frames) == [0, 1, 2]


# ---------------------------------------------------------------------------
# Cancellation — flush completed only, no torn frame
# ---------------------------------------------------------------------------

def test_streaming_stop_flushes_completed_only(monkeypatch):
    token = CancelToken()
    seen = {"n": 0}

    def integ(image, ai, **kw):
        with threading.Lock():
            seen["n"] += 1
        if seen["n"] >= 4:
            token.cancel()
        return _r1d(float(np.sum(image)))

    monkeypatch.setattr(reduction_core, "integrate_1d", integ)
    plan = _plan()
    frames = _frames(30)
    sink = MemorySink()
    session = ReductionSession(
        plan, Scan("s", frames, integrator=object()),
        sink=sink, execution="streaming", executor=2, cancel_token=token,
    )
    for fr in frames:
        session.submit(fr)
    result = session.finish()

    assert result.cancelled is True
    # Only genuinely-completed frames were written; none are torn/half.
    assert len(sink.frames) <= 30
    assert all(r.result_1d is not None for r in sink.frames.values())


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def test_process_rejected_in_streaming_mode(monkeypatch):
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))
    session = ReductionSession(
        _plan(), Scan("s", _frames(2), integrator=object()),
        sink=MemorySink(), execution="streaming", executor=2,
    )
    with pytest.raises(RuntimeError, match="streaming"):
        session.process(_frames(2))
    session.finish()


def test_submit_rejected_in_chunked_mode(monkeypatch):
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))
    session = ReductionSession(
        _plan(), Scan("s", _frames(2), integrator=object()),
        sink=MemorySink(), executor=2,
    )
    with pytest.raises(RuntimeError, match="streaming"):
        session.submit(Frame(0, image=np.zeros((2, 2))))
    session.finish()


# ---------------------------------------------------------------------------
# Fail-loud on sink/write failure (BLOCKER 2)
# ---------------------------------------------------------------------------
class _BoomSink:
    """A sink whose write() always fails — to exercise fail-loud finish()."""

    def begin(self, scan, plan):
        pass

    def write(self, frame, reduction):
        raise RuntimeError("disk full")

    def finish(self, result):
        pass

    def abort(self, result):
        pass


def test_streaming_write_failure_surfaces(monkeypatch):
    """A streaming sink WRITE failure must SURFACE (fail-loud), never be silently
    swallowed.  It re-raises the ORIGINAL exception at the earliest point the
    main thread touches the session after the writer records it — the next
    submit() (existing guard) or finish() (this fix, for a last-frame failure)."""
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))
    session = ReductionSession(
        _plan(), Scan("boom", _frames(4), integrator=object()),
        sink=_BoomSink(), execution="streaming", executor=2,
    )
    with pytest.raises(RuntimeError, match="disk full"):
        for fr in _frames(4):
            session.submit(fr)
        session.finish()
    # The failure is recorded (preserved) for inspection, not lost.
    assert session._failure is not None
    assert "disk full" in str(session._failure)


def test_finish_raise_on_failure_false_returns_failed_result(monkeypatch):
    """The opt-out escape hatch: a last-frame write failure surfaces only at
    finish(); raise_on_failure=False returns the failed result (preserved on the
    session) instead of raising, so a caller can inspect result.failed."""
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))
    session = ReductionSession(
        _plan(), Scan("boom", _frames(1), integrator=object()),
        sink=_BoomSink(), execution="streaming", executor=2,
    )
    # A single frame: its write fails on the writer thread, so the ONLY point
    # the main thread observes it is finish() (no later submit to surface it).
    session.submit(_frames(1)[0])
    result = session.finish(raise_on_failure=False)
    assert result.failed and "disk full" in (result.error or "")
    assert session.result is result
