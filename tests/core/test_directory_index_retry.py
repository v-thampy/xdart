# -*- coding: utf-8 -*-
"""R1 — DirectoryIndex bounded retry/provisional handling (record_probe) and
the persistent poll_forever producer.  A fake, injectable clock throughout —
no wall-clock sleeps.
"""

from __future__ import annotations

from pathlib import Path

from xrd_tools.sources.directory_index import DirectoryIndex, RetryState
from xrd_tools.sources.probe import ProbeResult, ProbeState


class _FakeClock:
    """Monotonic fake clock: advance() moves time forward; calling the
    instance returns the current reading (matches Callable[[], float])."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def _touch(path: Path, data: bytes = b"x") -> Path:
    path.write_bytes(data)
    return path


# ---- terminal results clear/never start retry tracking ---------------------


def test_ready_result_is_returned_unchanged_and_not_tracked(tmp_path):
    _touch(tmp_path / "a.nxs")
    index = DirectoryIndex(tmp_path, clock=_FakeClock())
    index.poll()

    result = ProbeResult(ProbeState.READY, reason="ok")
    effective = index.record_probe(tmp_path / "a.nxs", result)

    assert effective is result
    assert index.retry_state(tmp_path / "a.nxs") is None


def test_processed_output_and_invalid_are_terminal_too(tmp_path):
    _touch(tmp_path / "a.nxs")
    index = DirectoryIndex(tmp_path, clock=_FakeClock())
    index.poll()

    for state in (ProbeState.PROCESSED_OUTPUT, ProbeState.INVALID):
        result = ProbeResult(state, reason="terminal")
        effective = index.record_probe(tmp_path / "a.nxs", result)
        assert effective is result
        assert index.retry_state(tmp_path / "a.nxs") is None


# ---- provisional results: bounded retry window ------------------------------


def test_in_progress_starts_a_retry_window_and_surfaces_as_in_progress(tmp_path):
    _touch(tmp_path / "a.nxs")
    clock = _FakeClock()
    index = DirectoryIndex(tmp_path, clock=clock, retry_deadline=10.0)
    index.poll()

    raw = ProbeResult(ProbeState.IN_PROGRESS, reason="still writing")
    effective = index.record_probe(tmp_path / "a.nxs", raw)

    assert effective.state is ProbeState.IN_PROGRESS
    state = index.retry_state(tmp_path / "a.nxs")
    assert isinstance(state, RetryState)
    assert state.attempts == 1
    assert state.first_seen_at == 0.0


def test_imageless_shell_stays_provisional_within_the_deadline_not_retired(tmp_path):
    """The exact policy from the handoff: a newly readable, no-detector-
    dataset shell must NOT be immediately trusted as imageless."""
    _touch(tmp_path / "a.nxs")
    clock = _FakeClock()
    index = DirectoryIndex(tmp_path, clock=clock, retry_deadline=10.0)
    index.poll()

    raw_imageless = ProbeResult(ProbeState.IMAGELESS, reason="no detector dataset yet")
    effective = index.record_probe(tmp_path / "a.nxs", raw_imageless)
    assert effective.state is ProbeState.IN_PROGRESS   # NOT surfaced as imageless yet

    clock.advance(5.0)   # still within the 10s window
    effective = index.record_probe(tmp_path / "a.nxs", raw_imageless)
    assert effective.state is ProbeState.IN_PROGRESS
    assert index.retry_state(tmp_path / "a.nxs").attempts == 2


def test_retry_exhausts_after_the_deadline_and_surfaces_the_raw_verdict(tmp_path):
    _touch(tmp_path / "a.nxs")
    clock = _FakeClock()
    index = DirectoryIndex(tmp_path, clock=clock, retry_deadline=10.0)
    index.poll()

    raw = ProbeResult(ProbeState.IMAGELESS, reason="no detector dataset")
    index.record_probe(tmp_path / "a.nxs", raw)          # t=0, starts window

    clock.advance(9.0)
    still_provisional = index.record_probe(tmp_path / "a.nxs", raw)
    assert still_provisional.state is ProbeState.IN_PROGRESS

    clock.advance(2.0)   # t=11 >= 10s deadline
    final = index.record_probe(tmp_path / "a.nxs", raw)

    assert final is raw
    assert final.state is ProbeState.IMAGELESS
    assert index.retry_state(tmp_path / "a.nxs") is None   # window cleared


def test_per_call_retry_deadline_overrides_the_constructor_default(tmp_path):
    _touch(tmp_path / "a.nxs")
    clock = _FakeClock()
    index = DirectoryIndex(tmp_path, clock=clock, retry_deadline=1000.0)
    index.poll()

    raw = ProbeResult(ProbeState.IN_PROGRESS, reason="still writing")
    index.record_probe(tmp_path / "a.nxs", raw)
    clock.advance(5.0)
    final = index.record_probe(tmp_path / "a.nxs", raw, retry_deadline=1.0)

    # Exhausted the SHORT per-call deadline, not the 1000s constructor
    # default -- an IN_PROGRESS raw verdict that never once resolved
    # escalates to INVALID rather than retrying forever (see
    # test_in_progress_exhaustion_escalates_to_invalid).
    assert final.state is ProbeState.INVALID


# ---- reset on file-stamp change ---------------------------------------------


def test_retry_window_resets_when_the_file_stamp_changes_via_record_probe(tmp_path):
    a = _touch(tmp_path / "a.nxs", data=b"short")
    clock = _FakeClock()
    index = DirectoryIndex(tmp_path, clock=clock, retry_deadline=10.0)
    index.poll()

    raw = ProbeResult(ProbeState.IN_PROGRESS, reason="still writing")
    index.record_probe(a, raw)
    clock.advance(3.0)
    index.record_probe(a, raw)
    assert index.retry_state(a).attempts == 2

    # File content changes -> new version stamp -> poll() picks it up as
    # "changed" and clears the stale window; the next record_probe starts
    # a FRESH window at the current clock reading, not accumulating attempts.
    a.write_bytes(b"a much longer payload now")
    index.poll()
    assert index.retry_state(a) is None   # cleared by poll()'s changed-set sweep

    clock.advance(1.0)
    index.record_probe(a, raw)
    state = index.retry_state(a)
    assert state.attempts == 1
    assert state.first_seen_at == 4.0   # fresh window, not resumed from t=0


def test_poll_clears_retry_state_for_removed_candidates(tmp_path):
    a = _touch(tmp_path / "a.nxs")
    clock = _FakeClock()
    index = DirectoryIndex(tmp_path, clock=clock)
    index.poll()

    index.record_probe(a, ProbeResult(ProbeState.IN_PROGRESS, reason="writing"))
    assert index.retry_state(a) is not None

    a.unlink()
    index.poll()

    assert index.retry_state(a) is None


def test_reconfigure_clears_all_retry_state(tmp_path):
    a = _touch(tmp_path / "a.nxs")
    clock = _FakeClock()
    index = DirectoryIndex(tmp_path, clock=clock)
    index.poll()
    index.record_probe(a, ProbeResult(ProbeState.IN_PROGRESS, reason="writing"))
    assert index.retry_state(a) is not None

    index.reconfigure(name_filter="anything")

    assert index.retry_state(a) is None


# ---- retry state is unrelated-candidate-safe --------------------------------


def test_retry_tracking_for_one_candidate_does_not_affect_another(tmp_path):
    a = _touch(tmp_path / "a.nxs")
    b = _touch(tmp_path / "b.nxs")
    clock = _FakeClock()
    index = DirectoryIndex(tmp_path, clock=clock)
    index.poll()

    index.record_probe(a, ProbeResult(ProbeState.IN_PROGRESS, reason="writing"))
    assert index.retry_state(a) is not None
    assert index.retry_state(b) is None


# ---- poll_forever: persistent, cancellable, no wall-clock sleep ------------


def test_poll_forever_yields_a_snapshot_per_iteration_and_stops_cleanly(tmp_path):
    _touch(tmp_path / "a.nxs")
    index = DirectoryIndex(tmp_path, clock=_FakeClock())

    slept = []
    counter = {"n": 0}

    def _should_stop():
        return counter["n"] >= 3

    def _fake_sleep(seconds):
        slept.append(seconds)
        counter["n"] += 1

    snapshots = list(index.poll_forever(
        interval=5.0, should_stop=_should_stop, sleep=_fake_sleep))

    assert len(snapshots) == 3
    assert all(s.generation >= 1 for s in snapshots)
    assert slept == [5.0, 5.0, 5.0]


def test_poll_forever_stops_immediately_when_should_stop_is_already_true(tmp_path):
    index = DirectoryIndex(tmp_path, clock=_FakeClock())
    snapshots = list(index.poll_forever(
        interval=1.0, should_stop=lambda: True, sleep=lambda s: None))
    assert snapshots == []
