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
    FrameRecordStore,
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
    from tests.core.contracts import ThreadSpySink, assert_streaming_contract

    spy = ThreadSpySink(inner=MemorySink())
    caller = threading.get_ident()
    sess = ScanSession(ReductionPlan(integration_2d=None),
                       Scan("c", _frames(4), integrator=object()),
                       sink=spy, executor=2)
    sess.start()
    for fr in _frames(4):
        sess.submit(fr)
    sess.finish()

    # the SAME single-writer contract used for a raw ReductionSession; the spy
    # injects worker_process, so the wrapper must still fan it to pool workers
    # even though MemorySink does not define it (expect_worker_process=True).
    assert_streaming_contract(spy, caller, n_frames=4, expect_worker_process=True)


# ── adversarial-audit hardening (the event contract must be tamper-evident +
#    thread-pinned before the xdart bridge builds on it) ───────────────────────

def test_frame_event_result_arrays_are_read_only():
    """FATAL fix: the event's result arrays are the SAME ndarrays the sink
    stored, so they must be read-only — else a listener could retroactively
    corrupt already-persisted/cached data."""
    sink = MemorySink()
    sess = _standard_session(3)
    # rebuild with our own sink so we can inspect what it stored
    sess = ScanSession(ReductionPlan(integration_2d=None),
                       Scan("ro", _frames(3), integrator=object()),
                       sink=sink, executor=2)
    events: list[FrameEvent] = []
    sess.on_frame_completed(events.append)
    sess.start()
    for fr in _frames(3):
        sess.submit(fr)
    sess.finish()

    assert events
    e = events[0]
    with pytest.raises(ValueError):
        e.result_1d.intensity[0] = 999.0          # read-only enforced
    with pytest.raises(ValueError):
        e.result_1d.radial[0] = 999.0
    # the sink stored the SAME object, so it is protected too
    assert sink.frames[e.frame_index].result_1d.intensity[0] != 999.0


def test_frame_event_metadata_is_read_only():
    """metadata is a read-only mapping, so a listener can't corrupt the view
    other listeners (or the bridge) see for the same frame."""
    sess = _standard_session(1)
    events: list[FrameEvent] = []
    sess.on_frame_completed(events.append)
    sess.start()
    sess.submit(_frames(1)[0])
    sess.finish()
    assert events
    with pytest.raises(TypeError):                # MappingProxyType
        events[0].metadata["poison"] = True


def test_flush_delegates_to_public_then_private_then_noop():
    """ScanSession.flush() (via the event-sink wrapper) prefers the sink's public
    `flush`, falls back to the historical private `_flush` (the QtNexusSink shim),
    and is a silent no-op for a sink with neither (ADR-0004 §4)."""
    from xrd_tools.session.scan_session import _EventSink
    from types import SimpleNamespace

    calls = []
    pub = SimpleNamespace(flush=lambda *, force=False: calls.append(("pub", force)))
    _EventSink(pub, lambda f, r: None).flush(force=True)
    assert calls == [("pub", True)]

    calls.clear()
    priv = SimpleNamespace(_flush=lambda *, force=False: calls.append(("priv", force)))
    _EventSink(priv, lambda f, r: None).flush(force=True)
    assert calls == [("priv", True)]

    neither = SimpleNamespace()                   # no flush, no _flush
    _EventSink(neither, lambda f, r: None).flush()   # must not raise


def test_submit_raises_after_finish_and_while_paused():
    """Caller-contract violations stay LOUD: submit() after finish() or while
    paused RAISES (not a False 'dropped' return) — mirrors ReductionSession."""
    sess = _standard_session(3)
    sess.start()
    sess.submit(_frames(3)[0])
    sess.finish()
    with pytest.raises(RuntimeError, match="after finish"):
        sess.submit(_frames(3)[1])

    sess2 = _standard_session(3)
    sess2.start()
    sess2.submit(_frames(3)[0])
    assert sess2.pause(timeout=10) is True
    with pytest.raises(RuntimeError, match="paused"):
        sess2.submit(_frames(3)[1])
    sess2.resume()
    sess2.finish()


def test_progress_fires_from_both_caller_and_writer_threads():
    """ADR-0004 §1: on_progress fires on the caller thread (submit side) AND the
    writer thread (completion side) — the dual-thread guarantee the bridge's
    QueuedConnection design assumes."""
    main = threading.get_ident()
    idents: list[int] = []
    sess = _standard_session(4)
    sess.on_progress(lambda p: idents.append(threading.get_ident()))
    sess.start()
    for fr in _frames(4):
        sess.submit(fr)
    sess.finish()
    assert any(i == main for i in idents)         # submit-side (caller)
    assert any(i != main for i in idents)         # completion-side (writer)


def test_state_change_always_fires_on_caller_thread():
    """ADR-0004 §1: on_state_change fires on the orchestrating (caller) thread —
    the bridge maps it straight to sigPaused/sigResuming WITHOUT QueuedConnection."""
    main = threading.get_ident()
    idents: list[int] = []
    sess = _standard_session(2)
    sess.on_state_change(lambda s: idents.append(threading.get_ident()))
    sess.start()
    sess.submit(_frames(2)[0])
    assert sess.pause(timeout=10) is True
    sess.resume()
    sess.submit(_frames(2)[1])
    sess.finish()
    assert idents and all(i == main for i in idents)


def test_finish_with_no_frames_fires_no_completions():
    """Completions fire ONLY after a real write/replace — finishing an empty
    (or cancelled) run must emit zero on_frame_completed events, or the bridge
    would publish a frame that was never written."""
    events: list[FrameEvent] = []
    sess = _standard_session(2)
    sess.on_frame_completed(events.append)
    sess.start()
    sess.finish()                                  # no submit
    assert events == []
    assert sess.frames_completed == 0


def test_event_registration_returns_idempotent_unsubscribe():
    """on_*() returns an unsubscribe handle so a bridge/notebook can detach
    without tearing down the session; calling it twice is a no-op."""
    seen: list[int] = []
    sess = _standard_session(4)
    off = sess.on_frame_completed(lambda e: seen.append(e.frame_index))
    sess.start()
    sess.submit(_frames(4)[0])
    sess.pause(timeout=10)              # drain -> the first completion fires
    n_before = len(seen)
    assert n_before == 1
    off()                               # detach
    off()                               # idempotent: second call must not raise
    sess.resume()
    for fr in _frames(4)[1:]:
        sess.submit(fr)
    sess.finish()
    assert len(seen) == n_before        # no further events after unsubscribe


def test_double_finish_is_idempotent_no_extra_state_event():
    """finish() is idempotent and does not re-emit a state-change on the second
    call (so a bridge tearing down on running->finished can't double-fire)."""
    states: list[StateChangeEvent] = []
    sess = _standard_session(2)
    sess.on_state_change(states.append)
    sess.start()
    for fr in _frames(2):
        sess.submit(fr)
    r1 = sess.finish()
    n_after_first = len(states)
    r2 = sess.finish()                             # idempotent
    assert r2 is r1 or r2 == r1
    assert len(states) == n_after_first            # no extra state event


# ── 4f-bridge: clear_frame_images pass-through (xdart PERF-3 raw-nulling) ─────
def test_clear_frame_images_true_nulls_source_images_after_write():
    """ScanSession threads clear_frame_images to its inner ReductionSession, so
    the writer nulls frame.image post-write (the xdart streaming path passes
    True via open_live_scan_session to release ~18 MB/frame)."""
    frames = _frames(3)
    sess = ScanSession(ReductionPlan(integration_2d=None),
                       Scan("s", frames, integrator=object()),
                       sink=MemorySink(), executor=2, clear_frame_images=True)
    for fr in frames:
        sess.submit(fr)
    sess.finish()
    assert all(fr.image is None for fr in frames)


def test_clear_frame_images_default_keeps_images():
    frames = _frames(3)
    sess = ScanSession(ReductionPlan(integration_2d=None),
                       Scan("s", frames, integrator=object()),
                       sink=MemorySink(), executor=2)        # default False
    for fr in frames:
        sess.submit(fr)
    sess.finish()
    assert all(fr.image is not None for fr in frames)


def test_optional_record_store_receives_completed_frame_records():
    store = FrameRecordStore(max_heavy_items=None)
    frames = _frames(2)
    frames[0].source_path = "/tmp/source.tif"
    frames[0].source_frame_index = 0
    sess = ScanSession(
        ReductionPlan(integration_2d=None),
        Scan("s", frames, integrator=object()),
        sink=MemorySink(),
        executor=2,
        record_store=store,
    )
    for fr in frames:
        sess.submit(fr)
    sess.finish()

    rec = store.get(0)
    assert rec is not None
    assert rec.modes_1d == ("default",)
    np.testing.assert_allclose(rec.view_1d().intensity_1d, [0.0, 1.0])
    assert store.source_identity(0) == "/tmp/source.tif#0"


def test_optional_record_store_can_mark_completed_writes_persisted_for_eviction():
    store = FrameRecordStore(max_heavy_items=1)
    frames = _frames(2)
    sess = ScanSession(
        ReductionPlan(integration_2d=None),
        Scan("s", frames, integrator=object()),
        sink=MemorySink(),
        executor=2,
        record_store=store,
        record_store_persisted_on_write=True,
    )
    for fr in frames:
        sess.submit(fr)
    sess.finish()

    assert store.get(0) is not None and store.get(1) is not None
    assert not store.has_heavy_payload(0)
    assert store.has_heavy_payload(1)


def test_live_store_config_wired_through_scan_session_evicts_persisted_completions():
    # A-prep2: pin the exact live-store config (max_heavy_items=64 mirror of
    # LiveFrameSeries._in_memory_cap; require_persisted_for_eviction) end-to-end
    # through ScanSession with record_store_persisted_on_write=True.  Completing
    # more frames than the heavy cap thins the persisted overflow, never an
    # unpersisted frame (none here, since each write marks itself persisted).
    cap = 64
    store = FrameRecordStore(
        max_heavy_items=cap, require_persisted_for_eviction=True
    )
    frames = _frames(cap + 3)
    sess = ScanSession(
        ReductionPlan(integration_2d=None),
        Scan("s", frames, integrator=object()),
        sink=MemorySink(),
        executor=2,
        record_store=store,
        record_store_persisted_on_write=True,
    )
    for fr in frames:
        sess.submit(fr)
    sess.finish()

    # Every completed frame is in the store...
    assert len(store) == cap + 3
    # ...but heavy arrays are bounded at the cap: exactly the overflow is thinned.
    heavy = sum(1 for fr in frames if store.has_heavy_payload(fr.index))
    assert heavy == cap
    thinned = sum(1 for fr in frames if not store.has_heavy_payload(fr.index))
    assert thinned == 3
    # Thinned records keep their light fields (labels/axes/metadata survive).
    a_thinned = next(fr for fr in frames if not store.has_heavy_payload(fr.index))
    rec = store.get(a_thinned.index)
    assert rec is not None
    assert rec.view_1d().intensity_1d is None
