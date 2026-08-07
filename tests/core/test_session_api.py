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
    ResultMode,
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


def _frame_record_store_array_nbytes(store: FrameRecordStore) -> int:
    total = 0
    for record in store.snapshot().values():
        views = tuple(record.results_1d.values()) + tuple(record.results_2d.values())
        for view in views:
            for attr in (
                "intensity_1d",
                "sigma_1d",
                "intensity_2d",
                "sigma_2d",
                "raw",
                "thumbnail",
            ):
                array = getattr(view, attr, None)
                if array is not None:
                    total += np.asarray(array).nbytes
            for attr in ("axis_1d", "axis_2d_x", "axis_2d_y"):
                values = getattr(getattr(view, attr, None), "values", None)
                if values is not None:
                    total += np.asarray(values).nbytes
    return total


def test_h10_c4_external_ledger_is_exact_identity_and_validated_pre_engine():
    """C4 composes the already-created per-output ledger into ScanSession.
    Mode/target mismatches are rejected before ``sink.begin`` or engine work;
    callers that omit the seam retain the default one-ledger construction."""
    import inspect

    from xrd_tools.session import StageLedger

    assert "accounting" in inspect.signature(ScanSession).parameters, (
        "C4 owner missing: ScanSession must accept the one external StageLedger"
    )
    mode = ResultMode.one_d()
    target_map = {mode: ("nexus:/tmp/c4.nxs",)}

    class BeginSpy(MemorySink):
        def __init__(self):
            super().__init__()
            self.begins = 0

        def begin(self, scan, plan):
            self.begins += 1
            return super().begin(scan, plan)

    def source():
        return Scan("c4", _frames(1), integrator=object())

    exact = StageLedger(required_modes=(mode,), targets_by_mode=target_map)
    sink = BeginSpy()
    session = ScanSession(
        ReductionPlan(integration_2d=None), source(), sink=sink, executor=1,
        accounting=exact, targets_by_mode=target_map)
    assert session.accounting is exact
    assert sink.begins == 1
    session.finish(raise_on_failure=False)

    wrong_target = StageLedger(
        required_modes=(mode,),
        targets_by_mode={mode: ("nexus:/tmp/other.nxs",)},
    )
    sink = BeginSpy()
    with pytest.raises(ValueError, match="target map"):
        ScanSession(
            ReductionPlan(integration_2d=None), source(), sink=sink, executor=1,
            accounting=wrong_target, targets_by_mode=target_map)
    assert sink.begins == 0

    wrong_mode = ResultMode.two_d()
    wrong_modes = StageLedger(
        required_modes=(wrong_mode,),
        targets_by_mode={wrong_mode: ("nexus:/tmp/c4.nxs",)},
    )
    sink = BeginSpy()
    with pytest.raises(ValueError, match="required modes"):
        ScanSession(
            ReductionPlan(integration_2d=None), source(), sink=sink, executor=1,
            accounting=wrong_modes, targets_by_mode=target_map)
    assert sink.begins == 0

    default = ScanSession(
        ReductionPlan(integration_2d=None), source(), sink=MemorySink(),
        executor=1, targets_by_mode=target_map)
    assert default.accounting is not exact
    default.finish(raise_on_failure=False)


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


def test_live_scan_session_adapter_wires_store_with_live_source_identity(tmp_path):
    from types import SimpleNamespace

    from xdart.modules.reduction import frame_from_live_frame, open_live_scan_session

    source = tmp_path / "raw_master.h5"
    live_frames = [
        SimpleNamespace(
            idx=i,
            map_raw=np.full((2, 2), i + 1, dtype=float),
            bg_raw=None,
            scan_info={},
            source_file=str(source),
            source_frame_idx=i + 10,
            mask=None,
            poni=None,
            integrator=object(),
        )
        for i in range(2)
    ]
    store = FrameRecordStore(max_heavy_items=None)
    sess = open_live_scan_session(
        live_frames,
        ReductionPlan(integration_2d=None),
        scan_name="live",
        record_store=store,
    )

    converted = [frame_from_live_frame(live) for live in live_frames]
    for frame in converted:
        sess.submit(frame)
    sess.finish()

    assert len(store) == len(converted)
    for frame in converted:
        rec = store.get(frame.index)
        assert rec is not None
        assert store.source_identity(frame.index) == (
            f"{frame.source_path}#{frame.source_frame_index}"
        )


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
        obligations=("nexus:s",),
        write_targets_by_mode={ResultMode.one_d(): ("nexus:s",)},
    )
    for fr in frames:
        sess.submit(fr)
    assert sess.pause(timeout=20.0), "the writer did not drain"

    # Bounded DURING the run: exactly the durable overflow is thinned.
    assert store.get(0) is not None and store.get(1) is not None
    assert not store.has_heavy_payload(0)
    assert store.has_heavy_payload(1)

    sess.resume()
    sess.finish()
    # H10-C2-A: the ONE post-terminal sweep then releases the remaining
    # current-durable heavy data as well; the light records stay.
    assert not store.has_heavy_payload(1)
    assert store.get(0) is not None and store.get(1) is not None


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
        obligations=("nexus:s",),
        write_targets_by_mode={ResultMode.one_d(): ("nexus:s",)},
    )
    for fr in frames:
        sess.submit(fr)
    assert sess.pause(timeout=20.0), "the writer did not drain"

    # Every completed frame is in the store...
    assert len(store) == cap + 3
    # ...but heavy arrays are bounded at the cap DURING the run: exactly the
    # durable overflow is thinned.
    heavy = sum(1 for fr in frames if store.has_heavy_payload(fr.index))
    assert heavy == cap
    thinned = sum(1 for fr in frames if not store.has_heavy_payload(fr.index))
    assert thinned == 3
    # Thinned records keep their light fields (labels/axes/metadata survive).
    a_thinned = next(fr for fr in frames if not store.has_heavy_payload(fr.index))
    rec = store.get(a_thinned.index)
    assert rec is not None
    assert rec.view_1d().intensity_1d is None

    sess.resume()
    sess.finish()
    # H10-C2-A: the post-terminal sweep releases the rest of the current-durable
    # heavy data; every light record is retained.
    assert len(store) == cap + 3
    assert sum(1 for fr in frames if store.has_heavy_payload(fr.index)) == 0


def test_long_live_scan_record_store_plateaus_under_item_bound():
    """H8 pre-flip gate: a long ScanSession cannot grow a light tail forever."""
    n_frames = 5000
    heavy_cap = 64
    item_cap = 512
    retained_kb_per_submitted_frame_budget = 0.20
    store = FrameRecordStore(
        max_items=item_cap,
        max_heavy_items=heavy_cap,
        require_persisted_for_eviction=True,
    )
    frames = _frames(n_frames)
    sess = ScanSession(
        ReductionPlan(integration_2d=None),
        Scan("s", frames, integrator=object()),
        sink=MemorySink(),
        executor=2,
        record_store=store,
        record_store_persisted_on_write=True,
        obligations=("nexus:s",),
        write_targets_by_mode={ResultMode.one_d(): ("nexus:s",)},
    )
    for frame in frames:
        sess.submit(frame)
    sess.finish()

    assert len(store) <= item_cap
    assert sum(store.has_heavy_payload(frame.index) for frame in frames) <= heavy_cap
    retained_kb_per_frame = (
        _frame_record_store_array_nbytes(store) / 1024.0 / n_frames
    )
    assert retained_kb_per_frame < retained_kb_per_submitted_frame_budget


# ── H10-C2-B: the coordinated session's resource envelope ────────────────────
#
# A materialized ``Scan`` is NOT descriptor-managed: the accepted C1 minimal
# executor duck and every existing row above stay green.  A descriptor-managed
# source opts into envelope resolution, early executor validation and the typed
# pre-effect error.


class _DescriptorManagedSource:
    """The narrow descriptor-managed contract ScanSession coordinates."""

    def __init__(self, n=4, *, height=2, width=2, itemsize=2):
        self.frame_indices = range(n)
        self.name = "descriptor-source"
        self.integrator = object()
        self.allocation = None
        self.bind_calls = []
        self.effects = []
        self._height, self._width, self._itemsize = height, width, itemsize

    def container_descriptor(self):
        from xrd_tools.sources.descriptor import ContainerDescriptor
        self.effects.append("descriptor")
        return ContainerDescriptor(
            path=Path("/data/descriptor-source.h5"),
            dataset_path="/entry/data/data",
            frame_count=len(self.frame_indices),
            frame_shape=(self._height, self._width),
            dtype=np.dtype(f"uint{self._itemsize * 8}"))

    def bind_allocation(self, allocation):
        if self.allocation is not None and self.allocation != allocation:
            raise ValueError("a second, different allocation was bound")
        self.bind_calls.append(allocation)
        self.allocation = allocation

    def to_scan(self, **kwargs):
        """Real sources carry their reduction context through ``to_scan``; the
        bare-duck path in ``_coerce_to_scan`` drops it."""
        return Scan(self.name,
                    [Frame(int(i), image=np.full((2, 2), i, dtype=float))
                     for i in self.frame_indices],
                    **kwargs)

    def frame_for(self, index):
        self.effects.append(("frame_for", index))
        return Frame(int(index), image=np.full((2, 2), index, dtype=float))

    def load_frame(self, index):
        self.effects.append(("load_frame", index))
        return np.full((2, 2), index, dtype=float)


class _CallerPool:
    """A caller-owned executor: ScanSession validates it, never shuts it down."""

    def __init__(self, max_workers=None):
        if max_workers is not None:
            self._max_workers = max_workers
        self.shutdown_calls = 0

    def submit(self, fn, *a, **kw):
        raise AssertionError("no work may be submitted before validation")

    def shutdown(self, *a, **kw):
        self.shutdown_calls += 1


def _envelope_session(source, **kw):
    kw.setdefault("envelope_bytes", 64 * 1024 ** 3)
    return ScanSession(ReductionPlan(integration_2d=None), source,
                       sink=MemorySink(), **kw)


def test_c2b_materialized_scan_keeps_the_accepted_c1_minimal_duck():
    """No descriptor, no envelope: the C1 contract is untouched."""
    session = _standard_session(2)
    try:
        assert session.policy is None or session.policy.allocation is None
    finally:
        session.finish(raise_on_failure=False)


def test_c2b_descriptor_managed_source_is_bound_before_any_frame_effect():
    source = _DescriptorManagedSource()
    session = _envelope_session(source, executor=1)
    try:
        assert len(source.bind_calls) == 1
        assert source.effects[0] == "descriptor"
        assert not any(isinstance(e, tuple) for e in source.effects)
        assert session.policy.allocation is source.allocation
    finally:
        session.finish(raise_on_failure=False)


def test_c2b_undersize_envelope_raises_before_any_bind_or_sink_effect():
    from xrd_tools.session.policy import SessionEnvelopeError

    source = _DescriptorManagedSource()
    sink = MemorySink()
    with pytest.raises(SessionEnvelopeError) as exc:
        ScanSession(ReductionPlan(integration_2d=None), source, sink=sink,
                    envelope_bytes=4096)
    assert exc.value.available_bytes == 4096
    assert exc.value.required_bytes > 4096
    assert source.bind_calls == []
    assert source.effects == ["descriptor"]
    assert not getattr(sink, "results", None)


def test_c2b_managed_external_executor_must_declare_positive_capacity():
    source = _DescriptorManagedSource()
    pool = _CallerPool(max_workers=None)          # cannot prove its bound
    with pytest.raises(ValueError):
        _envelope_session(source, executor=pool)
    assert pool.shutdown_calls == 0, "a caller pool is never shut down"
    assert source.bind_calls == []


def test_c2b_managed_external_executor_capacity_may_be_declared_explicitly():
    source = _DescriptorManagedSource()
    pool = _CallerPool(max_workers=None)
    session = _envelope_session(source, executor=pool, executor_workers=2)
    try:
        # the pool's declared capacity IS its request, and must equal the grant
        assert source.allocation.workers == 2
        assert pool.shutdown_calls == 0
    finally:
        session.finish(raise_on_failure=False)
        assert pool.shutdown_calls == 0


def test_c2b_managed_external_executor_beyond_the_grant_is_refused_early():
    source = _DescriptorManagedSource()
    pool = _CallerPool(max_workers=64)
    with pytest.raises(ValueError):
        _envelope_session(source, executor=pool,
                          envelope_bytes=3 * 1024 ** 3)   # grants far fewer
    assert pool.shutdown_calls == 0
    assert source.bind_calls == []


def test_c2b_an_explicit_policy_is_revalidated_not_trusted():
    import dataclasses as _dc
    from xrd_tools.session.policy import (
        SessionResourceRequirements,
        resolve_session_policy,
    )

    req = SessionResourceRequirements(height=2, width=2, native_itemsize=2,
                                      modes_1d=1, npt_1d=1000)
    good = resolve_session_policy(req, envelope_bytes=64 * 1024 ** 3, env={},
                                  requests={"workers": 1,
                                            "reduction_inflight": 1})
    source = _DescriptorManagedSource()
    session = _envelope_session(source, executor=1, policy=good)
    try:
        # the caller's EXACT object is bound, never an equal reconstruction
        assert source.allocation is good.allocation
    finally:
        session.finish(raise_on_failure=False)

    forged = _dc.replace(good, allocation=_dc.replace(
        good.allocation, counts={**good.allocation.counts, "queue_depth": 999}))
    with pytest.raises(ValueError):
        _envelope_session(_DescriptorManagedSource(), executor=1, policy=forged)


def test_c2b_integer_executor_is_a_worker_request_not_the_owned_count():
    """For a descriptor-managed source an integer ``executor`` is a REQUEST;
    the coordinator owns a pool sized to the granted ``allocation.workers``."""
    source = _DescriptorManagedSource()
    session = _envelope_session(source, executor=64)
    try:
        alloc = source.allocation
        assert alloc.workers >= 1
        assert alloc.workers <= 64
        assert session.policy.allocation is alloc
    finally:
        session.finish(raise_on_failure=False)


def test_c2b_the_granted_inflight_bound_is_actually_wired_not_just_reported():
    """``allocation.reduction_inflight`` must become the real coordinated
    inflight bound - never an unconsumed report field."""
    from xrd_tools.session.policy import minimum_bytes, requirements_from

    source = _DescriptorManagedSource()
    plan = ReductionPlan(integration_2d=None)
    # Pin the envelope to exactly ``M`` so every count is granted its MINIMUM:
    # the granted inflight is then 1, while ReductionSession's own default is
    # ``max(2, 2 * n_workers)`` = 2.  A grant that merely coincides with that
    # default cannot discriminate, so the bound must be pinned apart from it.
    envelope = minimum_bytes(requirements_from(source.container_descriptor(), plan))
    session = _envelope_session(source, executor=1, envelope_bytes=envelope)
    try:
        alloc = source.allocation
        assert (alloc.workers, alloc.reduction_inflight) == (1, 1)
        assert session._session.inflight_max == alloc.reduction_inflight == 1
    finally:
        session.finish(raise_on_failure=False)


def test_c2b_a_caller_inflight_request_must_match_an_explicit_policy():
    """With an explicit policy a caller ``inflight_max`` is a request that must
    agree with the granted bound, and disagreement rejects before effects."""
    from xrd_tools.session.policy import (
        SessionResourceRequirements,
        resolve_session_policy,
    )

    req = SessionResourceRequirements(height=2, width=2, native_itemsize=2,
                                      modes_1d=1, npt_1d=1000)
    good = resolve_session_policy(req, envelope_bytes=64 * 1024 ** 3, env={},
                                  requests={"workers": 1,
                                            "reduction_inflight": 1})
    source = _DescriptorManagedSource()
    session = _envelope_session(
        source, executor=1, policy=good,
        inflight_max=good.allocation.reduction_inflight)
    try:
        assert source.allocation is good.allocation
    finally:
        session.finish(raise_on_failure=False)

    conflicting = _DescriptorManagedSource()
    with pytest.raises(ValueError):
        _envelope_session(conflicting, executor=1, policy=good,
                          inflight_max=good.allocation.reduction_inflight + 5)
    assert conflicting.bind_calls == []


# ── correction 1: the executor / inflight boundary, all before any effect ────

def _alloc_for(source, plan=None, **kw):
    from xrd_tools.session.policy import requirements_from, resolve_session_policy
    plan = plan or ReductionPlan(integration_2d=None)
    req = requirements_from(source.container_descriptor(), plan)
    return resolve_session_policy(req, envelope_bytes=64 * 1024 ** 3, env={},
                                  **kw)


def test_c1_non_positive_or_malformed_integer_executor_rejects_before_bind():
    for bad in (0, -1, 2.5):
        source = _DescriptorManagedSource()
        with pytest.raises(ValueError):
            _envelope_session(source, executor=bad)
        assert source.bind_calls == []
        assert source.effects in ([], ["descriptor"])


def test_c1_conflicting_executor_and_executor_workers_reject_before_bind():
    source = _DescriptorManagedSource()
    with pytest.raises(ValueError):
        _envelope_session(source, executor=4, executor_workers=2)
    assert source.bind_calls == []
    # the same value twice is one unambiguous request
    ok = _DescriptorManagedSource()
    session = _envelope_session(ok, executor=2, executor_workers=2)
    try:
        assert len(ok.bind_calls) == 1
    finally:
        session.finish(raise_on_failure=False)


def test_c1_non_positive_inflight_max_rejects_before_bind():
    for bad in (0, -3, 1.5):
        source = _DescriptorManagedSource()
        with pytest.raises(ValueError):
            _envelope_session(source, executor=1, inflight_max=bad)
        assert source.bind_calls == []


def test_c1_caller_pool_must_equal_the_grant_exactly():
    """A pool SMALLER than the grant is as wrong as a larger one: the policy
    would claim workers the pool cannot supply."""
    explicit = _alloc_for(_DescriptorManagedSource(), requests={"workers": 2,
                                                               "reduction_inflight": 2})
    grant = explicit.allocation.workers
    for capacity in (grant - 1, grant + 1):
        source = _DescriptorManagedSource()
        pool = _CallerPool(max_workers=capacity)
        with pytest.raises(ValueError):
            _envelope_session(source, executor=pool, policy=explicit)
        assert source.bind_calls == []
        assert pool.shutdown_calls == 0
    exact = _DescriptorManagedSource()
    pool = _CallerPool(max_workers=grant)
    session = _envelope_session(exact, executor=pool, policy=explicit)
    try:
        assert exact.allocation is explicit.allocation
        assert pool.shutdown_calls == 0
    finally:
        session.finish(raise_on_failure=False)
        assert pool.shutdown_calls == 0


def test_c1_caller_pool_max_workers_and_explicit_capacity_must_agree():
    source = _DescriptorManagedSource()
    pool = _CallerPool(max_workers=4)
    with pytest.raises(ValueError):
        _envelope_session(source, executor=pool, executor_workers=2)
    assert source.bind_calls == [] and pool.shutdown_calls == 0


def test_c1_explicit_grant_above_an_owned_worker_request_rejects_before_bind():
    explicit = _alloc_for(_DescriptorManagedSource(),
                          requests={"workers": 3, "reduction_inflight": 6})
    assert explicit.allocation.workers == 3
    source = _DescriptorManagedSource()
    with pytest.raises(ValueError):
        _envelope_session(source, executor=1, policy=explicit)   # owned req < grant
    assert source.bind_calls == []


def test_c1_explicit_inflight_max_must_equal_the_granted_bound():
    explicit = _alloc_for(_DescriptorManagedSource(),
                          requests={"workers": 2, "reduction_inflight": 4})
    source = _DescriptorManagedSource()
    with pytest.raises(ValueError):
        _envelope_session(source, executor=2, policy=explicit,
                          inflight_max=explicit.allocation.reduction_inflight + 1)
    assert source.bind_calls == []


def test_c1_a_cadence_only_prior_policy_still_resolves_and_clamps():
    """``SessionPolicy(allocation=None)`` is NOT an explicit allocation: its
    requests are clamped normally instead of being equality-checked."""
    from xrd_tools.session.policy import FlushPolicy, SessionPolicy

    cadence_only = SessionPolicy(flush=FlushPolicy(interval=3))
    source = _DescriptorManagedSource()
    session = _envelope_session(source, executor=2, inflight_max=3,
                                policy=cadence_only)
    try:
        alloc = source.allocation
        assert alloc is not None
        assert alloc.reduction_inflight <= 3           # clamped, not rejected
        assert session.policy.flush.interval == 3      # the cadence survives
    finally:
        session.finish(raise_on_failure=False)


def test_event_sink_exposes_worker_process_only_when_inner_sink_owns_it():
    """A plain sink must not make reduction build a discarded corrected image."""
    from types import SimpleNamespace

    from xrd_tools.session.scan_session import _EventSink

    plain = _EventSink(SimpleNamespace(), lambda _frame, _reduction: None)
    assert getattr(plain, "worker_process", None) is None

    calls = []
    hooked = _EventSink(
        SimpleNamespace(
            worker_process=lambda frame, reduction: calls.append((frame, reduction))
        ),
        lambda _frame, _reduction: None,
    )
    worker_process = getattr(hooked, "worker_process", None)
    assert callable(worker_process)
    worker_process("frame", "reduction")
    assert calls == [("frame", "reduction")]
