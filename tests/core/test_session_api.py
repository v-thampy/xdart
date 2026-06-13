# -*- coding: utf-8 -*-
"""Phase 4f — the headless ``xrd_tools.session.ScanSession`` contract.

Offscreen, Qt-free.  Asserts the ADR-0003 / ADR-0004 contract: single-result
immutable ``FrameEvent``s, completion events on the WRITER thread, a listener
exception that cannot kill the run, the caller-owned ``generation`` stamp that
pause/resume never bumps, progress + state events, and the GI mode key.
"""
from __future__ import annotations

import dataclasses
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

_EXAMPLE = (Path(__file__).resolve().parents[2]
            / "examples" / "headless_scan_session.py")

from xrd_tools.core.containers import IntegrationResult1D
from xrd_tools.reduction import (
    Frame,
    GI1DMode,
    GI2DMode,
    GIMode,
    MemorySink,
    ReductionPlan,
    Scan,
)
import xrd_tools.reduction.core as reduction_core
from xrd_tools.session import (
    FrameEvent,
    ProgressEvent,
    ScanSession,
    StateChangeEvent,
)
from xrd_tools.session.scan_session import _mode_key_from_plan


def _r1d(value: float) -> IntegrationResult1D:
    return IntegrationResult1D(radial=np.array([0.0, 1.0]),
                              intensity=np.array([value, value + 1.0]),
                              sigma=None, unit="q_A^-1")


def _frames(n: int) -> list[Frame]:
    return [Frame(i, image=np.full((2, 2), i, dtype=float)) for i in range(n)]


def _standard_session(n=4, **kw) -> ScanSession:
    return ScanSession(ReductionPlan(integration_2d=None),
                       Scan("s", _frames(n), integrator=object()),
                       sink=MemorySink(), executor=2, **kw)


@pytest.fixture(autouse=True)
def _fake_integrate(monkeypatch):
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))


# ── lifecycle + single-result events ────────────────────────────────────────

def test_lifecycle_emits_one_single_result_event_per_frame():
    events: list[FrameEvent] = []
    sess = _standard_session(4)
    sess.on_frame_completed(events.append)
    sess.start()
    for fr in _frames(4):
        assert sess.submit(fr) is True
    sess.finish()

    assert len(events) == 4
    assert {e.frame_index for e in events} == {0, 1, 2, 3}
    for e in events:
        assert e.result_1d is not None          # single-result, populated
        assert e.mode_key is None                # standard scan → trivial key
        assert e.timestamp > 0
    assert sess.frames_completed == 4
    assert sess.frames_submitted == 4
    assert not sess.is_running                   # finished


def test_frame_event_is_immutable():
    e = FrameEvent(frame_index=0, mode_key=None, result_1d=_r1d(1.0),
                   result_2d=None, metadata={}, generation=0, timestamp=1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.frame_index = 5            # type: ignore[misc]


def test_completion_events_fire_on_the_writer_thread():
    """ADR-0004 §1: on_frame_completed fires on the single writer thread, never
    the caller — so a Qt bridge MUST marshal via QueuedConnection."""
    main_ident = threading.get_ident()
    idents: list[int] = []
    sess = _standard_session(3)
    sess.on_frame_completed(lambda e: idents.append(threading.get_ident()))
    sess.start()
    for fr in _frames(3):
        sess.submit(fr)
    sess.finish()

    assert len(idents) == 3
    assert all(i != main_ident for i in idents)      # off the caller thread
    assert len(set(idents)) == 1                       # exactly one writer thread


def test_listener_exception_cannot_kill_the_run():
    """A raising on_frame_completed listener is caught + logged; every frame
    still completes and the run does not fail (the T0-7/S1 false-success trap)."""
    seen: list[int] = []
    sess = _standard_session(5)

    def _boom(_e):
        raise RuntimeError("listener blew up")

    sess.on_frame_completed(_boom)
    sess.on_frame_completed(lambda e: seen.append(e.frame_index))  # still runs
    sess.start()
    for fr in _frames(5):
        sess.submit(fr)
    result = sess.finish()                              # must not raise

    assert sorted(seen) == [0, 1, 2, 3, 4]
    assert sess.frames_completed == 5
    assert not getattr(result, "failed", False)


# ── generation (ADR-0004 §2) ─────────────────────────────────────────────────

def test_pause_resume_does_not_bump_generation():
    events: list[FrameEvent] = []
    sess = _standard_session(6)
    sess.on_frame_completed(events.append)
    sess.set_generation(7)
    sess.start()

    for fr in _frames(6)[:3]:
        sess.submit(fr)
    assert sess.pause(timeout=10) is True               # drain (completions fire)
    sess.resume()
    for fr in _frames(6)[3:]:
        sess.submit(fr)
    sess.finish()

    assert len(events) == 6
    assert all(e.generation == 7 for e in events)       # pause/resume never bumped

    # sensitivity: an explicit set_generation DOES change subsequent stamps.
    events2: list[FrameEvent] = []
    sess2 = _standard_session(2)
    sess2.on_frame_completed(events2.append)
    sess2.set_generation(9)
    sess2.start()
    for fr in _frames(2):
        sess2.submit(fr)
    sess2.finish()
    assert all(e.generation == 9 for e in events2)


# ── progress + state events ──────────────────────────────────────────────────

def test_progress_events_carry_absolute_counts():
    progress: list[ProgressEvent] = []
    sess = _standard_session(3)
    sess.on_progress(progress.append)
    sess.start()
    for fr in _frames(3):
        sess.submit(fr)
    sess.finish()

    assert progress, "expected progress events"
    last = progress[-1]
    assert last.submitted == 3
    assert last.completed == 3
    assert last.total == 3
    # monotonic non-decreasing counts
    assert [p.completed for p in progress] == sorted(p.completed for p in progress)


def test_state_events_fire_on_pause_resume_finish():
    states: list[StateChangeEvent] = []
    sess = _standard_session(2)
    sess.on_state_change(states.append)
    sess.start()
    sess.submit(_frames(2)[0])
    assert sess.pause(timeout=10) is True
    assert sess.is_paused
    sess.resume()
    assert not sess.is_paused
    sess.submit(_frames(2)[1])
    sess.finish()

    # paused-state and finished-state were observed
    assert any(s.is_paused for s in states)
    assert states[-1].is_running is False


def test_context_manager_finishes_on_exit():
    events: list[FrameEvent] = []
    with _standard_session(2) as sess:
        sess.on_frame_completed(events.append)
        for fr in _frames(2):
            sess.submit(fr)
    assert len(events) == 2                              # drained on __exit__
    assert not sess.is_running


# ── GI mode key (ADR-0003) ───────────────────────────────────────────────────

def test_mode_key_standard_is_none_gi_is_mode_tuple():
    assert _mode_key_from_plan(ReductionPlan(integration_2d=None)) is None
    gi_plan = ReductionPlan(
        integration_2d=None,
        gi=GIMode(incident_angle=0.2, mode_1d=GI1DMode.Q_TOTAL,
                  mode_2d=GI2DMode.QIP_QOOP),
    )
    assert _mode_key_from_plan(gi_plan) == ("q_total", "qip_qoop")


def test_headless_example_runs_qt_free_in_a_fresh_interpreter():
    """The shipped no-Qt example must run end-to-end in a clean interpreter with
    no Qt/pyqtgraph imported (Difference 2 — the headless path is real, not just
    asserted in-process where Qt may already be loaded by another test)."""
    if not _EXAMPLE.exists():
        pytest.skip("example not found")
    probe = (
        "import runpy, sys; runpy.run_path(sys.argv[1], run_name='__main__'); "
        "leaked=[m for m in sys.modules if m.split('.')[0] in "
        "('PySide6','PyQt5','PyQt6','qtpy','pyqtgraph')]; "
        "assert not leaked, leaked"
    )
    proc = subprocess.run([sys.executable, "-c", probe, str(_EXAMPLE)],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert "Qt-free. OK" in proc.stdout, proc.stdout


# ── the Phase-1 sink contract survives the event-sink wrapper ────────────────

def test_event_sink_wrapper_preserves_single_writer_contract():
    """Driving the Phase-1 ThreadSpySink THROUGH a ScanSession must still satisfy
    the HDF5 single-writer discipline: the wrapper's forwarding may not move
    write() off the one writer thread, nor disable the pool-thread worker_process
    (ADR-0004 §1 / Difference 6 contract harness)."""
    from tests.core.contracts import ThreadSpySink

    spy = ThreadSpySink(inner=MemorySink())
    caller = threading.get_ident()
    sess = ScanSession(ReductionPlan(integration_2d=None),
                       Scan("c", _frames(4), integrator=object()),
                       sink=spy, executor=2)
    sess.start()
    for fr in _frames(4):
        sess.submit(fr)
    sess.finish()

    hooks = spy.hooks()
    assert hooks[0] == "begin" and hooks[-1] == "finish"
    assert sorted(spy.frames_for("write")) == [0, 1, 2, 3]
    writer_threads = spy.threads_for("write")
    assert len(writer_threads) == 1 and writer_threads != {caller}
    # worker_process (forwarded by the wrapper) ran on pool threads, off the writer
    wp_threads = spy.threads_for("worker_process")
    assert wp_threads and not (wp_threads & writer_threads)
    assert sorted(spy.frames_for("worker_process")) == [0, 1, 2, 3]
