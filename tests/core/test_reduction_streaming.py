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

from xrd_tools.core.containers import IntegrationResult1D
from xrd_tools.reduction import (
    CancelToken,
    Frame,
    MemorySink,
    ReductionPlan,
    Scan,
    ReductionSession,
    run_reduction,
)
import xrd_tools.reduction.core as reduction_core


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
from tests.core.contracts import BoomSink as _ContractBoomSink


def _BoomSink():
    """write() always fails — exercise fail-loud finish().  (Shared double
    from tests.core.contracts; message kept for the match= asserts.)"""
    return _ContractBoomSink(boom_on="write", message="disk full")


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


def test_run_reduction_streaming_matches_chunked(monkeypatch):
    """#8a: run_reduction(execution="streaming") drives the streaming engine and
    yields the same per-frame products as the chunked default -- notebook/headless
    callers get streaming without hand-driving ReductionSession."""
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))
    chunked = run_reduction(_plan(), Scan("s", _frames(8), integrator=object()),
                            executor=2)
    streaming = run_reduction(_plan(), Scan("s", _frames(8), integrator=object()),
                              execution="streaming", executor=2)
    assert chunked.n_processed == streaming.n_processed == 8
    assert sorted(streaming.frames) == sorted(chunked.frames) == list(range(8))
    for i in range(8):
        np.testing.assert_array_equal(
            np.asarray(streaming.frames[i].result_1d.intensity, float),
            np.asarray(chunked.frames[i].result_1d.intensity, float),
        )


# ---------------------------------------------------------------------------
# Non-terminal drain() (Pause) — quiesce the writer at a frame boundary
# WITHOUT closing the session, so submit() works again on resume.
# ---------------------------------------------------------------------------

def test_streaming_drain_is_non_terminal(monkeypatch):
    """drain() writes every submitted-so-far frame but keeps the session OPEN:
    the writer thread stays alive and submit() works after drain (the GUI Pause
    primitive)."""
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))
    sink = MemorySink()
    session = ReductionSession(
        _plan(), Scan("s", _frames(6), integrator=object()),
        sink=sink, execution="streaming", executor=2,
    )
    for fr in _frames(6)[:3]:
        session.submit(fr)
    session.drain()
    # The 3 submitted frames are written; the writer is STILL ALIVE (no sentinel).
    assert sorted(sink.frames) == [0, 1, 2]
    assert session._writer_thread is not None and session._writer_thread.is_alive()
    assert not session._finished
    # Session still open -> more submits + another drain work (resume).
    for fr in _frames(6)[3:]:
        session.submit(fr)
    session.drain()
    assert sorted(sink.frames) == [0, 1, 2, 3, 4, 5]
    # Re-entrant drain with nothing pending returns immediately.
    session.drain()
    result = session.finish()
    assert result.n_processed == 6
    assert sorted(sink.frames) == list(range(6))


def test_streaming_drain_noop_before_start_and_for_chunked(monkeypatch):
    """drain() is a harmless no-op for chunked execution and before the first
    submit (no writer thread to join)."""
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))
    chunked = ReductionSession(_plan(), Scan("c", _frames(2), integrator=object()),
                               sink=MemorySink(), executor=2)
    chunked.drain()            # no-op (chunked)
    chunked.process(_frames(2))
    chunked.finish()

    streaming = ReductionSession(_plan(), Scan("s", _frames(2), integrator=object()),
                                 sink=MemorySink(), execution="streaming", executor=2)
    streaming.drain()          # no-op (stream not started yet)
    streaming.finish()


def test_streaming_drain_timeout_and_cancel_bail(monkeypatch):
    """drain(timeout=) BOUNDS the wait so a hung worker (stalled IO / runaway
    pyFAI — an uncancellable running future) can't deadlock a GUI pause, and it
    bails early once the cancel token trips (Stop/close)."""
    gate = threading.Event()

    def blocking(image, ai, **kw):
        gate.wait(5)           # never set in the test window -> writer hangs
        return _r1d(float(np.sum(image)))

    monkeypatch.setattr(reduction_core, "integrate_1d", blocking)
    token = CancelToken()
    session = ReductionSession(
        _plan(), Scan("s", _frames(2), integrator=object()),
        sink=MemorySink(), execution="streaming", executor=1, cancel_token=token,
    )
    session.submit(_frames(2)[0])

    # Worker stuck in integrate -> drain can't complete -> times out (bounded).
    t0 = time.monotonic()
    assert session.drain(timeout=0.2) is False
    assert time.monotonic() - t0 < 2.0          # did NOT hang on the stuck worker

    # Cancel (Stop/close) -> a subsequent drain bails promptly, not after 5 s.
    token.cancel()
    t1 = time.monotonic()
    assert session.drain(timeout=5.0) is False
    assert time.monotonic() - t1 < 2.0

    gate.set()                                  # release the worker, finish clean
    session.finish(raise_on_failure=False)


# ---------------------------------------------------------------------------
# #2 — cancel-aware submit() under backpressure
# ---------------------------------------------------------------------------

def test_submit_cancel_aware_under_backpressure(monkeypatch):
    """#2: submit() must not block indefinitely when the in-flight window is full
    and a cancel is requested (Stop/Pause).  It polls the cancel token with a
    bounded acquire so the dispatch loop can proceed to its stop-check promptly."""
    gate = threading.Event()

    def blocking(image, ai, **kw):
        gate.wait(5)               # worker stalls -> window fills -> submit blocks
        return _r1d(float(np.sum(image)))

    monkeypatch.setattr(reduction_core, "integrate_1d", blocking)
    token = CancelToken()
    session = ReductionSession(
        _plan(), Scan("s", _frames(4), integrator=object()),
        sink=MemorySink(), execution="streaming", executor=1,
        cancel_token=token, inflight_max=1,   # window=1 so 2nd submit would block
    )
    session.submit(_frames(4)[0])   # fills the window
    # Second submit in a thread; it would block on the semaphore without the fix.
    done = []
    t = threading.Thread(target=lambda: (session.submit(_frames(4)[1]),
                                         done.append(True)))
    t.start()
    time.sleep(0.15)             # give it time to block in acquire loop
    assert not done              # still blocked

    token.cancel()               # cancel → submit should return promptly
    t.join(timeout=2)
    assert done == [True]        # returned within the poll interval
    assert session._cancelled    # marked cancelled, not raised through

    gate.set()
    session.finish(raise_on_failure=False)


# ---------------------------------------------------------------------------
# #4 — bounded finish() writer join
# ---------------------------------------------------------------------------

def test_finish_join_timeout_loud_on_stuck_worker(monkeypatch):
    """#4: finish(join_timeout=) must not hang indefinitely when a worker is
    stalled.  After the timeout it cancels, marks the result failed, records a
    TimeoutError, and RETURNS (does not hang) — the caller gets a loud error,
    not a silent success."""
    gate = threading.Event()

    def blocking(image, ai, **kw):
        gate.wait(5)
        return _r1d(float(np.sum(image)))

    monkeypatch.setattr(reduction_core, "integrate_1d", blocking)
    session = ReductionSession(
        _plan(), Scan("s", _frames(1), integrator=object()),
        sink=MemorySink(), execution="streaming", executor=1,
    )
    session.submit(_frames(1)[0])

    t0 = time.monotonic()
    result = session.finish(raise_on_failure=False, join_timeout=0.3)
    assert time.monotonic() - t0 < 2.0         # did NOT hang
    assert result.failed and result.error       # marked failed + recorded error
    assert "timed out" in result.error.lower() or isinstance(
        session._failure, TimeoutError)

    gate.set()   # release the worker so the thread can exit cleanly


def test_finish_join_timeout_normal_session_still_succeeds(monkeypatch):
    """A normal (non-stalled) session with join_timeout= still finishes cleanly
    and still re-raises on result.failed (BLOCKER 2 preserved)."""
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))
    sink = MemorySink()
    session = ReductionSession(
        _plan(), Scan("s", _frames(4), integrator=object()),
        sink=sink, execution="streaming", executor=2,
    )
    for fr in _frames(4):
        session.submit(fr)
    result = session.finish(join_timeout=30.0)   # large timeout, should complete fast
    assert result.n_processed == 4
    assert sorted(sink.frames) == list(range(4))


# ---------------------------------------------------------------------------
# T0-5/6/7 — destructive-teardown + writer-survival hardening
# ---------------------------------------------------------------------------

class _SpySink(MemorySink):
    """MemorySink recording whether finish()/abort() were called."""

    def __init__(self):
        super().__init__()
        self.finish_called = False
        self.abort_called = False

    def finish(self, result):
        self.finish_called = True
        return super().finish(result)

    def abort(self, result):
        self.abort_called = True


def test_finish_timeout_does_not_touch_sink_while_writer_alive(monkeypatch):
    """T0-5: on a writer-join timeout the writer thread is STILL ALIVE and may
    yet write -- finish() must NOT call sink.finish()/abort() (atomic-mode
    NexusSink.abort used to unlink the tmp holding every written frame, and
    closing the h5 handle races the in-flight write)."""
    gate = threading.Event()

    def blocking(image, ai, **kw):
        gate.wait(5)
        return _r1d(float(np.sum(image)))

    monkeypatch.setattr(reduction_core, "integrate_1d", blocking)
    sink = _SpySink()
    session = ReductionSession(
        _plan(), Scan("s", _frames(1), integrator=object()),
        sink=sink, execution="streaming", executor=1,
    )
    session.submit(_frames(1)[0])

    with pytest.warns(RuntimeWarning):
        result = session.finish(raise_on_failure=False, join_timeout=0.3)

    assert result.failed                      # loud failure, not silence
    assert sink.finish_called is False        # sink left untouched
    assert sink.abort_called is False

    gate.set()    # release the worker so the thread can exit cleanly


def test_nexus_sink_abort_preserves_partial(tmp_path):
    """T0-6/S7: in atomic mode every frame written so far lives in the tmp
    file; abort() must preserve it as <output>.partial, never unlink it."""
    from xrd_tools.reduction import NexusSink

    out = tmp_path / "run.nxs"
    sink = NexusSink(out, overwrite=True)
    sink.begin(Scan("s", _frames(1), integrator=object()), _plan())
    assert sink._tmp_path is not None and sink._tmp_path.exists()
    tmp = sink._tmp_path

    with pytest.warns(RuntimeWarning, match="partial"):
        sink.abort(result=None)

    partial = tmp_path / "run.nxs.partial"
    assert partial.exists(), "aborted run's data must be preserved"
    assert not tmp.exists()
    assert not out.exists()                   # never half-promoted


def test_nexus_sink_finish_failure_preserves_partial(tmp_path, monkeypatch):
    """T0-6/S7: a finish-time failure (e.g. scan_data upsert) must surface
    loudly AND leave the written frames recoverable as <output>.partial."""
    from xrd_tools.reduction import NexusSink
    from xrd_tools.reduction.core import Scan as _Scan

    out = tmp_path / "run2.nxs"
    sink = NexusSink(out, overwrite=True)
    scan = _Scan("s", _frames(1), integrator=object())
    sink.begin(scan, _plan())

    def boom(self):
        raise RuntimeError("scan_data upsert failed")

    monkeypatch.setattr(type(scan), "to_scan_data", boom)
    with pytest.raises(RuntimeError, match="upsert failed"):
        with pytest.warns(RuntimeWarning, match="partial"):
            sink.finish(result=None)

    assert (tmp_path / "run2.nxs.partial").exists()
    assert not out.exists()


def test_writer_loop_survives_raising_progress_callback(monkeypatch):
    """T0-7/S1: an exception from the post-write progress callback must be
    RECORDED as the session failure, not allowed to kill the writer thread --
    a dead writer made submit() spin forever on the in-flight window and
    finish() join a dead thread and report SUCCESS with frames missing."""
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))

    def bad_cb(progress, *a, **k):
        # Only the post-write emission runs on the writer thread; the 'start'
        # stage is emitted synchronously from submit() and is out of scope.
        if getattr(progress, "stage", "") == "write":
            raise RuntimeError("boom from progress callback")

    frames = _frames(2)
    session = ReductionSession(
        _plan(), Scan("s", frames, integrator=object()),
        sink=MemorySink(), execution="streaming", executor=1,
        progress_cb=bad_cb,
    )
    session.submit(frames[0])
    assert session.drain(timeout=5)           # writer ALIVE: queue drains
    assert session._writer_thread.is_alive()
    # The callback failure was recorded -> the next submit raises it loudly
    # (fail-loud contract), instead of silently proceeding or deadlocking.
    deadline = time.monotonic() + 5
    while session._failure is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert isinstance(session._failure, RuntimeError)
    with pytest.raises(RuntimeError, match="progress callback"):
        session.submit(frames[1])
    result = session.finish(raise_on_failure=False)
    assert result.failed
    assert result.n_processed == 1            # the write itself succeeded


def test_submit_detects_dead_writer(monkeypatch):
    """T0-7 belt-and-suspenders: if the writer thread dies anyway, a blocked
    submit() must detect it, record a failure, and return -- not spin on the
    in-flight semaphore forever."""
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))
    frames = _frames(2)
    session = ReductionSession(
        _plan(), Scan("s", frames, integrator=object()),
        sink=MemorySink(), execution="streaming", executor=1,
    )
    session.submit(frames[0])
    assert session.drain(timeout=5)

    # Simulate a dead writer + a full in-flight window.
    dead = threading.Thread(target=lambda: None)
    dead.start(); dead.join()
    session._writer_thread = dead
    while session._semaphore.acquire(blocking=False):
        pass

    t0 = time.monotonic()
    session.submit(frames[1])                 # returns; previously spun forever
    assert time.monotonic() - t0 < 5.0
    assert session._cancelled
    assert isinstance(session._failure, RuntimeError)
    assert "writer thread died" in str(session._failure)


# ---------------------------------------------------------------------------
# Phase 4a — pause / resume / is_paused
# ---------------------------------------------------------------------------

def test_pause_quiesces_writer_and_rejects_submit(monkeypatch):
    """pause() drains the in-flight window (writer provably idle), flips
    is_paused, and rejects further submits until resume()."""
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))
    sink = MemorySink()
    session = ReductionSession(
        _plan(), Scan("p", _frames(6), integrator=object()),
        sink=sink, execution="streaming", executor=2,
    )
    for fr in _frames(3):
        session.submit(fr)

    assert session.pause(timeout=10) is True       # fully quiesced
    assert session.is_paused
    assert sorted(sink.frames) == [0, 1, 2]        # everything written

    with pytest.raises(RuntimeError, match="paused"):
        session.submit(_frames(1)[0])

    session.resume()
    assert not session.is_paused
    for fr in _frames(6)[3:]:                        # continue the same session
        session.submit(fr)
    result = session.finish()
    assert result.n_processed == 6
    assert sorted(sink.frames) == [0, 1, 2, 3, 4, 5]


def test_pause_resume_matches_uninterrupted_baseline(monkeypatch):
    """A paused-then-resumed run yields byte-identical products to one that
    never paused (pause is a quiesce, not a data operation)."""
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))

    base = MemorySink()
    s0 = ReductionSession(_plan(), Scan("base", _frames(8), integrator=object()),
                          sink=base, execution="streaming", executor=2)
    for fr in _frames(8):
        s0.submit(fr)
    s0.finish()

    paused = MemorySink()
    s1 = ReductionSession(_plan(), Scan("paused", _frames(8), integrator=object()),
                          sink=paused, execution="streaming", executor=2)
    for fr in _frames(8)[:4]:
        s1.submit(fr)
    assert s1.pause(timeout=10)
    s1.resume()
    for fr in _frames(8)[4:]:
        s1.submit(fr)
    s1.finish()

    assert sorted(base.frames) == sorted(paused.frames)
    for idx in base.frames:
        np.testing.assert_array_equal(
            base.frames[idx].result_1d.intensity,
            paused.frames[idx].result_1d.intensity,
        )


def test_process_rejected_while_paused(monkeypatch):
    """Chunked process() also refuses while paused (uniform contract for the
    GUI run-state model); resume() re-allows it."""
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))
    session = ReductionSession(_plan(), Scan("c", _frames(2), integrator=object()),
                              sink=MemorySink(), execution="chunked")
    assert session.pause() is True                  # no in-flight window: flag only
    assert session.is_paused
    with pytest.raises(RuntimeError, match="paused"):
        session.process()
    session.resume()
    session.process()
    assert session.finish().n_processed == 2


def test_pause_is_noop_when_finished_or_cancelled(monkeypatch):
    """pause() after finish() / on a cancelled session is a no-op returning
    True, and a cancelled session is never marked paused."""
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))
    s = ReductionSession(_plan(), Scan("f", _frames(2), integrator=object()),
                         sink=MemorySink(), execution="streaming", executor=2)
    for fr in _frames(2):
        s.submit(fr)
    s.finish()
    assert s.pause() is True
    assert not s.is_paused                          # finished dominates

    token = CancelToken()
    s2 = ReductionSession(_plan(), Scan("x", _frames(2), integrator=object()),
                          sink=MemorySink(), execution="streaming", executor=2,
                          cancel_token=token)
    token.cancel()
    assert s2.pause() is True
    assert not s2.is_paused                          # cancelled is never paused
    s2.finish(raise_on_failure=False)


def test_is_running_lifecycle(monkeypatch):
    """is_running (Phase 4d): True from construction (sink open) until finish;
    False after finish and on a cancelled session."""
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))
    s = ReductionSession(_plan(), Scan("r", _frames(2), integrator=object()),
                         sink=MemorySink(), execution="streaming", executor=2)
    assert s.is_running
    for fr in _frames(2):
        s.submit(fr)
    assert s.is_running                          # still running mid-stream
    s.finish()
    assert not s.is_running                      # finished

    token = CancelToken()
    s2 = ReductionSession(_plan(), Scan("c", _frames(1), integrator=object()),
                          sink=MemorySink(), execution="streaming", executor=2,
                          cancel_token=token)
    assert s2.is_running
    token.cancel()
    assert not s2.is_running                     # cancelled
    s2.finish(raise_on_failure=False)


def test_submit_returns_bool_and_drops_cleanly_on_cancel(monkeypatch):
    """submit() returns True for an accepted frame and False for one dropped
    because the session was cancelled — and a dropped frame is NEITHER
    registered in the scan inventory NOR counted as submitted.  This is the
    accepted-vs-cancelled fix: a Stop racing a submit must not leave a phantom
    frame the caller believes was processed."""
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))
    token = CancelToken()
    s = ReductionSession(_plan(), Scan("s", _frames(2), integrator=object()),
                         sink=MemorySink(), execution="streaming", executor=2,
                         cancel_token=token)
    # A fresh later frame is accepted -> registered + counted.
    assert s.submit(Frame(2, image=np.full((2, 2), 2.0))) is True
    submitted_after_accept = s._submitted
    inventory_after_accept = len(s.scan.frames)
    assert 2 in s.scan._frame_by_index

    # Cancel, then submit another fresh frame: dropped (False, no raise), and it
    # leaves NO trace in the inventory or the submitted counter.
    token.cancel()
    assert s.submit(Frame(99, image=np.full((2, 2), 9.0))) is False
    assert s._submitted == submitted_after_accept
    assert len(s.scan.frames) == inventory_after_accept
    assert 99 not in s.scan._frame_by_index
    s.finish(raise_on_failure=False)
