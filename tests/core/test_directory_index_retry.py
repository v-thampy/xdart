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
    # One file per state: a terminal verdict is sticky for its stamp (R1-R1),
    # so probing two different terminal states on the SAME path would return
    # the first (sticky) verdict, not the second — use distinct paths.
    for i, state in enumerate((ProbeState.PROCESSED_OUTPUT, ProbeState.INVALID)):
        name = f"f{i}.nxs"
        _touch(tmp_path / name)
        index = DirectoryIndex(tmp_path, clock=_FakeClock())
        index.poll()
        result = ProbeResult(state, reason="terminal")
        effective = index.record_probe(tmp_path / name, result)
        assert effective is result
        assert index.retry_state(tmp_path / name) is None


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


# ===========================================================================
# R1-R1 — exhausted retry is TERMINAL for the unchanged stamp (gates 1 + 2)
# ===========================================================================

def test_r1r1_exhausted_imageless_stays_terminal_on_repeated_probes(tmp_path):
    """Gate 1: after the window exhausts to IMAGELESS, repeated probes of the
    SAME stamp keep returning IMAGELESS — they must NOT reopen a fresh window
    (the held defect was in_progress -> imageless -> in_progress)."""
    _touch(tmp_path / "shell.nxs")
    clock = _FakeClock()
    index = DirectoryIndex(tmp_path, clock=clock, retry_deadline=10.0)
    index.poll()
    raw = ProbeResult(ProbeState.IMAGELESS, reason="no det")
    p = tmp_path / "shell.nxs"

    assert index.record_probe(p, raw).state is ProbeState.IN_PROGRESS
    clock.advance(11.0)
    assert index.record_probe(p, raw).state is ProbeState.IMAGELESS   # exhausted
    # Repeated probes of the same stamp stay terminal, never in_progress:
    for _ in range(3):
        assert index.record_probe(p, raw).state is ProbeState.IMAGELESS
        assert index.retry_state(p) is None   # no reopened window


def test_r1r1_exhausted_in_progress_stays_terminal_invalid_on_repeated_probes(tmp_path):
    """Gate 1 (INVALID variant): a never-resolving IN_PROGRESS escalates to
    INVALID at exhaustion and STAYS INVALID on repeated same-stamp probes."""
    _touch(tmp_path / "stuck.nxs")
    clock = _FakeClock()
    index = DirectoryIndex(tmp_path, clock=clock, retry_deadline=5.0)
    index.poll()
    raw = ProbeResult(ProbeState.IN_PROGRESS, reason="still writing")
    p = tmp_path / "stuck.nxs"

    index.record_probe(p, raw)
    clock.advance(6.0)
    assert index.record_probe(p, raw).state is ProbeState.INVALID
    for _ in range(3):
        assert index.record_probe(p, raw).state is ProbeState.INVALID
        assert index.retry_state(p) is None


def test_r1r1_stamp_change_clears_terminal_and_opens_one_fresh_window(tmp_path):
    """Gate 2: a size/mtime change after a terminal resolution clears it and
    permits exactly ONE fresh bounded window (back to IN_PROGRESS)."""
    p = _touch(tmp_path / "shell.nxs", data=b"short")
    clock = _FakeClock()
    index = DirectoryIndex(tmp_path, clock=clock, retry_deadline=10.0)
    index.poll()
    raw = ProbeResult(ProbeState.IMAGELESS, reason="no det")

    index.record_probe(p, raw)
    clock.advance(11.0)
    assert index.record_probe(p, raw).state is ProbeState.IMAGELESS   # terminal

    # bytes change -> poll sees it -> terminal resolution cleared
    p.write_bytes(b"a longer payload now")
    index.poll()
    assert index.retry_state(p) is None

    # fresh window opens; NOT immediately terminal again
    eff = index.record_probe(p, raw)
    assert eff.state is ProbeState.IN_PROGRESS
    assert index.retry_state(p).attempts == 1


def test_r1r1_reconfigure_clears_terminal_resolution(tmp_path):
    """Gate 2 (reconfigure variant): a reconfiguration also clears a terminal
    resolution, so the candidate can open a fresh window after re-poll."""
    _touch(tmp_path / "shell.nxs")
    clock = _FakeClock()
    index = DirectoryIndex(tmp_path, clock=clock, retry_deadline=10.0)
    index.poll()
    raw = ProbeResult(ProbeState.IMAGELESS, reason="no det")
    p = tmp_path / "shell.nxs"
    index.record_probe(p, raw)
    clock.advance(11.0)
    assert index.record_probe(p, raw).state is ProbeState.IMAGELESS

    index.reconfigure(name_filter="")   # any real change (whitespace filter matches all)
    index.poll()
    assert index.record_probe(p, raw).state is ProbeState.IN_PROGRESS   # fresh window


def test_r1r1_terminal_resolution_survives_an_unchanged_poll(tmp_path):
    """A terminal resolution must persist across an unchanged poll — the
    R1-R5 unchanged-poll early-return must not touch _terminal."""
    _touch(tmp_path / "shell.nxs")
    clock = _FakeClock()
    index = DirectoryIndex(tmp_path, clock=clock, retry_deadline=10.0)
    index.poll()
    raw = ProbeResult(ProbeState.IMAGELESS, reason="no det")
    p = tmp_path / "shell.nxs"
    index.record_probe(p, raw)
    clock.advance(11.0)
    assert index.record_probe(p, raw).state is ProbeState.IMAGELESS

    index.poll()   # unchanged
    assert index.last_delta.unchanged is True
    assert index.record_probe(p, raw).state is ProbeState.IMAGELESS   # still sticky


def test_r1r1_transient_in_progress_on_the_deadline_does_not_condemn_an_imageless_shell(tmp_path):
    """A stable readable-but-imageless shell probed IMAGELESS several times,
    then a lone TRANSIENT IN_PROGRESS (a momentary lock/stall) landing on the
    deadline, must still resolve to IMAGELESS — NOT be condemned to INVALID.
    The escalation keys on whether the file was EVER readable in the window,
    not on the single last sample (adversarial BUG-1)."""
    _touch(tmp_path / "shell.nxs")
    clock = _FakeClock()
    index = DirectoryIndex(tmp_path, clock=clock, retry_deadline=10.0)
    index.poll()
    p = tmp_path / "shell.nxs"
    imageless = ProbeResult(ProbeState.IMAGELESS, reason="no det")
    transient = ProbeResult(ProbeState.IN_PROGRESS, reason="momentary lock")

    assert index.record_probe(p, imageless).state is ProbeState.IN_PROGRESS  # window opens
    clock.advance(3.0)
    assert index.record_probe(p, imageless).state is ProbeState.IN_PROGRESS  # readable again
    clock.advance(8.0)   # t=11 >= 10: exhausts, but on a transient IN_PROGRESS
    resolved = index.record_probe(p, transient)
    assert resolved.state is ProbeState.IMAGELESS   # NOT INVALID
    # and it stays IMAGELESS (sticky) on repeat
    assert index.record_probe(p, transient).state is ProbeState.IMAGELESS


def test_r1r1_never_readable_window_still_escalates_to_invalid(tmp_path):
    """The converse: a window that was NEVER once readable (all IN_PROGRESS)
    still escalates to INVALID — the imageless-carry-forward must not weaken
    the genuine stuck/corrupt-file escalation."""
    _touch(tmp_path / "stuck.nxs")
    clock = _FakeClock()
    index = DirectoryIndex(tmp_path, clock=clock, retry_deadline=5.0)
    index.poll()
    p = tmp_path / "stuck.nxs"
    raw = ProbeResult(ProbeState.IN_PROGRESS, reason="never readable")

    index.record_probe(p, raw)
    clock.advance(2.0)
    index.record_probe(p, raw)
    clock.advance(4.0)   # t=6 >= 5
    assert index.record_probe(p, raw).state is ProbeState.INVALID


# ===========================================================================
# R1-R6 — stale / unknown probe completions create NO state (gate 9)
# ===========================================================================

def test_r1r6_unknown_path_probe_creates_no_state(tmp_path):
    """Gate 9: record_probe for a path absent from the current snapshot is
    ignored — returned unchanged, with no retry/terminal state created."""
    _touch(tmp_path / "a.nxs")
    index = DirectoryIndex(tmp_path, clock=_FakeClock())
    index.poll()
    ghost = tmp_path / "ghost.nxs"
    raw = ProbeResult(ProbeState.IN_PROGRESS, reason="late")

    eff = index.record_probe(ghost, raw)
    assert eff is raw                       # returned unchanged
    assert index.retry_state(ghost) is None  # no window created


def test_r1r6_stale_stamp_probe_creates_no_state(tmp_path):
    """Gate 9: a completion whose expected_stamp no longer matches the current
    candidate (a late result for an older version) creates no state."""
    p = _touch(tmp_path / "a.nxs", data=b"one")
    index = DirectoryIndex(tmp_path, clock=_FakeClock())
    index.poll()
    stale_stamp = (999, 999)   # never the real stamp
    raw = ProbeResult(ProbeState.IN_PROGRESS, reason="late")

    eff = index.record_probe(p, raw, expected_stamp=stale_stamp)
    assert eff is raw
    assert index.retry_state(p) is None


def test_r1r6_probe_candidate_rejects_a_removed_candidate(tmp_path):
    """Gate 9: probe_candidate on a candidate whose file was removed since the
    snapshot raises ValueError and records no state — before any file open."""
    import pytest
    a = _touch(tmp_path / "a.nxs")
    index = DirectoryIndex(tmp_path, clock=_FakeClock())
    index.poll()
    candidate = index.snapshot.candidates[0]

    a.unlink()
    index.poll()   # candidate now removed from snapshot

    with pytest.raises(ValueError):
        index.probe_candidate(candidate)
    assert index.retry_state(a) is None


def test_r1r6_probe_candidate_rejects_a_reconfigured_away_candidate(tmp_path):
    """Gate 9: probe_candidate on a candidate filtered out by a reconfigure
    (no longer in the snapshot) raises ValueError and records no state."""
    import pytest
    _touch(tmp_path / "sample_bg.nxs")
    index = DirectoryIndex(tmp_path, clock=_FakeClock())
    index.poll()
    candidate = index.snapshot.candidates[0]

    index.reconfigure(name_filter="-bg")   # excludes sample_bg
    index.poll()
    assert index.snapshot.candidates == ()

    with pytest.raises(ValueError):
        index.probe_candidate(candidate)
    assert index.retry_state(candidate.path) is None


# ===========================================================================
# R1-R7 — full candidate identity rejects an old-owner late completion
# ===========================================================================
import contextlib

from xrd_tools.core.scan import SourceKind
from xrd_tools.sources.adapters import (
    SourceFormatAdapter, _ADAPTERS, register_adapter)


@contextlib.contextmanager
def _isolated_adapters():
    saved = dict(_ADAPTERS)
    try:
        yield
    finally:
        _ADAPTERS.clear()
        _ADAPTERS.update(saved)


def _widget_adapter(idn):
    return SourceFormatAdapter(
        id=idn, kinds=(SourceKind.TILED,),
        is_candidate=lambda p: p.suffix == ".widget",
        scan_name=lambda p: p.stem,
        probe=lambda p: ProbeResult(ProbeState.READY, reason=idn, kind=SourceKind.TILED),
        open=lambda spec: None)


def test_r1r7_late_completion_from_old_owner_is_rejected_after_owner_flip(tmp_path):
    """An owner flip A->B on unchanged bytes shares the stamp.  A late result
    from A, delivered with A's Candidate identity, must be rejected: no retry
    or terminal state created, returned unchanged.  A subsequent probe through
    B then returns B's real result — not A's stale sticky verdict."""
    with _isolated_adapters():
        register_adapter(_widget_adapter("owner_a"))
        p = tmp_path / "x.widget"
        p.write_bytes(b"x")
        index = DirectoryIndex(tmp_path, clock=_FakeClock())
        index.poll()
        cand_a = index.snapshot.candidates[0]
        assert cand_a.adapter_id == "owner_a"

        register_adapter(_widget_adapter("owner_b"))   # flip precedence, same bytes
        index.poll()
        assert index.snapshot.candidates[0].adapter_id == "owner_b"

        # A's late result, carrying A's identity (path+stamp+adapter_id):
        a_result = ProbeResult(ProbeState.READY, reason="owner_a late", kind=SourceKind.TILED)
        effective = index.record_probe(cand_a, a_result)

        assert effective is a_result                 # ignored, returned unchanged
        assert index.retry_state(p) is None          # no window
        assert index._terminal.get(p) is None        # no sticky terminal

        # B's real result is what the current candidate resolves to
        cand_b = index.snapshot.candidates[0]
        b_result = ProbeResult(ProbeState.READY, reason="owner_b", kind=SourceKind.TILED)
        assert index.record_probe(cand_b, b_result).reason == "owner_b"


def test_r1r7_record_probe_rejects_explicit_mismatched_adapter_id(tmp_path):
    """The bare-path form gains an expected_adapter_id guard: a completion
    tagged for an owner that is not the current owner creates no state."""
    with _isolated_adapters():
        register_adapter(_widget_adapter("owner_b"))
        p = tmp_path / "x.widget"
        p.write_bytes(b"x")
        index = DirectoryIndex(tmp_path, clock=_FakeClock())
        index.poll()
        assert index.snapshot.candidates[0].adapter_id == "owner_b"

        eff = index.record_probe(
            p, ProbeResult(ProbeState.IN_PROGRESS, reason="from A"),
            expected_adapter_id="owner_a")   # A is not the current owner
        assert eff.reason == "from A"        # ignored
        assert index.retry_state(p) is None


def test_r1r7_probe_candidate_still_works_for_a_current_candidate(tmp_path):
    """No false rejection: probe_candidate on a fresh, current candidate
    records normally (the full-identity check must not over-reject)."""
    with _isolated_adapters():
        register_adapter(_widget_adapter("owner_a"))
        (tmp_path / "x.widget").write_bytes(b"x")
        index = DirectoryIndex(tmp_path, clock=_FakeClock())
        index.poll()
        c = index.snapshot.candidates[0]
        assert index.probe_candidate(c).reason == "owner_a"
