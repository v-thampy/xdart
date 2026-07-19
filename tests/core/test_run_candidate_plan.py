# -*- coding: utf-8 -*-
"""H19 §4 — ``RunCandidatePlan``: the immutable Source-card -> Run handoff seam.

Production-wired: a real :class:`DirectoryIndex` over real temp files produces
the :class:`Snapshot` the plan freezes; the filesystem is really mutated
(removed / rewritten / adapter-owner-flipped) and re-polled to drive the
transitions.  The reprobe->adopt growth cycle runs through the real
:meth:`DirectoryIndex.probe_candidate` with a small controllable adapter (the
same technique the DirectoryIndex probe/owner-flip tests use).  No fake on the
seam under test.

Covers the orchestrator corrections:
  RP1 — a grown baseline candidate is withheld (``changed``), not stranded;
        recover via reprobe + ``adopt`` exactly once; removal/owner-flip stay
        fail-closed.
  RP2 — ``reconcile`` enforces the frozen root/recursive/name_filter config.
  RP3 — per-candidate ``validate`` is O(1) (plan-owned cached lookup) + a
        3600-candidate benchmark.
  RP4 — the identity-only property is ``baseline_current`` (not ``all_ready``).
"""

from __future__ import annotations

import contextlib
import dataclasses
import time

import pytest

from xrd_tools.core.scan import SourceKind
from xrd_tools.sources import RunCandidatePlan, RunReconcile, SupersededPlanError
from xrd_tools.sources.adapters import (
    SourceFormatAdapter, _ADAPTERS, register_adapter)
from xrd_tools.sources.directory_index import DirectoryIndex, StaleCandidateError
from xrd_tools.sources.discover import Candidate
from xrd_tools.sources.probe import ProbeResult, ProbeState


def _touch(path, data=b"x"):
    path.write_bytes(data)
    return path


@contextlib.contextmanager
def _isolated_adapters():
    saved = dict(_ADAPTERS)
    try:
        yield
    finally:
        _ADAPTERS.clear()
        _ADAPTERS.update(saved)


def _nxs_override_adapter(idn):
    """Higher-precedence real adapter that claims ``.nxs`` by name, to flip the
    owning adapter of an existing candidate without a byte change."""
    return SourceFormatAdapter(
        id=idn, kinds=(SourceKind.TILED,),
        is_candidate=lambda p: p.suffix == ".nxs",
        scan_name=lambda p: p.stem,
        probe=lambda p: ProbeResult(
            ProbeState.READY, reason=idn, kind=SourceKind.TILED),
        open=lambda spec: None)


def _grow_adapter(idn="grow"):
    """Real adapter for ``.grow`` files whose probe verdict is content-driven:
    READY once the file contains ``b"ready"``, else IN_PROGRESS (a growing
    shell).  Lets the reprobe->adopt cycle run deterministically through the
    real DirectoryIndex.probe_candidate path."""
    def probe(p):
        data = p.read_bytes()
        if b"ready" in data:
            return ProbeResult(ProbeState.READY, reason="ready",
                               kind=SourceKind.TILED)
        return ProbeResult(ProbeState.IN_PROGRESS, reason="growing",
                           kind=SourceKind.TILED)
    return SourceFormatAdapter(
        id=idn, kinds=(SourceKind.TILED,),
        is_candidate=lambda p: p.suffix == ".grow",
        scan_name=lambda p: p.stem, probe=probe, open=lambda spec: None)


def _index_with(tmp_path, *names):
    for n in names:
        _touch(tmp_path / n)
    index = DirectoryIndex(tmp_path)
    index.poll()
    return index


# ── from_snapshot: freeze the ordered baseline ────────────────────────────

def test_from_snapshot_freezes_ordered_identity_and_config(tmp_path):
    index = DirectoryIndex(tmp_path, recursive=True, name_filter="scan")
    _touch(tmp_path / "scan_1.nxs")
    _touch(tmp_path / "scan_2.nxs")
    _touch(tmp_path / "scan_10.nxs")
    _touch(tmp_path / "other.nxs")             # excluded by the "scan" filter
    snap = index.poll()

    plan = RunCandidatePlan.from_snapshot(snap)

    assert plan.candidates == snap.candidates          # exact frozen baseline
    assert plan.generation == snap.generation
    assert plan.root == snap.root
    assert plan.recursive is True
    assert plan.name_filter == "scan"
    assert [p.name for p in plan.paths] == [
        "scan_1.nxs", "scan_2.nxs", "scan_10.nxs"]     # natural order
    assert set(plan.by_path()) == {c.path for c in snap.candidates}
    assert len(plan) == 3
    assert bool(plan) is True


def test_empty_plan_is_falsy(tmp_path):
    plan = RunCandidatePlan.from_snapshot(DirectoryIndex(tmp_path).poll())
    assert len(plan) == 0
    assert bool(plan) is False
    assert plan.paths == ()


# ── validate: fail-closed single-candidate check ──────────────────────────

def test_validate_returns_current_on_identity_match(tmp_path):
    index = _index_with(tmp_path, "a.nxs")
    plan = RunCandidatePlan.from_snapshot(index.snapshot)
    current = index.poll().by_path()[tmp_path / "a.nxs"]   # unchanged re-poll
    assert plan.validate(current) is current


def test_validate_fails_closed_on_removal(tmp_path):
    index = _index_with(tmp_path, "a.nxs")
    plan = RunCandidatePlan.from_snapshot(index.snapshot)
    (tmp_path / "a.nxs").unlink()
    fresh = index.poll()
    assert (tmp_path / "a.nxs") not in fresh.by_path()
    with pytest.raises(StaleCandidateError):
        plan.validate(None, path=tmp_path / "a.nxs")


def test_validate_fails_closed_on_stamp_change(tmp_path):
    index = _index_with(tmp_path, "a.nxs")
    plan = RunCandidatePlan.from_snapshot(index.snapshot)
    _touch(tmp_path / "a.nxs", b"much longer content changes the size stamp")
    current = index.poll().by_path()[tmp_path / "a.nxs"]
    with pytest.raises(StaleCandidateError):
        plan.validate(current)                # old bytes never opened as-is


def test_validate_fails_closed_on_owner_flip(tmp_path):
    with _isolated_adapters():
        index = _index_with(tmp_path, "a.nxs")
        plan = RunCandidatePlan.from_snapshot(index.snapshot)
        baseline = plan.by_path()[tmp_path / "a.nxs"]
        register_adapter(_nxs_override_adapter("nxs_override"))
        current = index.poll().by_path()[tmp_path / "a.nxs"]
        assert current.adapter_id == "nxs_override"
        assert current.adapter_id != baseline.adapter_id
        assert current.version_stamp == baseline.version_stamp
        with pytest.raises(StaleCandidateError):
            plan.validate(current)


def test_validate_rejects_live_appended_path_with_keyerror(tmp_path):
    index = _index_with(tmp_path, "a.nxs")
    plan = RunCandidatePlan.from_snapshot(index.snapshot)
    _touch(tmp_path / "b.nxs")
    appeared = index.poll().by_path()[tmp_path / "b.nxs"]
    with pytest.raises(KeyError):
        plan.validate(appeared)


def test_validate_needs_a_path_when_current_is_none(tmp_path):
    plan = RunCandidatePlan.from_snapshot(_index_with(tmp_path, "a.nxs").snapshot)
    with pytest.raises(ValueError):
        plan.validate(None)


# ── reconcile: run order, categorized transitions, live appends ────────────

def test_reconcile_unchanged_returns_frozen_run_order(tmp_path):
    index = _index_with(tmp_path, "scan_1.nxs", "scan_2.nxs")
    plan = RunCandidatePlan.from_snapshot(index.snapshot)

    result = plan.reconcile(index.poll())                     # unchanged

    assert isinstance(result, RunReconcile)
    assert result.run_order == plan.candidates
    assert result.changed == ()
    assert result.removed == ()
    assert result.owner_flipped == ()
    assert result.appended == ()
    assert result.baseline_current is True


def test_reconcile_reports_removed_fail_closed(tmp_path):
    index = _index_with(tmp_path, "scan_1.nxs", "scan_2.nxs")
    plan = RunCandidatePlan.from_snapshot(index.snapshot)
    (tmp_path / "scan_1.nxs").unlink()

    result = plan.reconcile(index.poll())

    assert [p.name for p in result.removed] == ["scan_1.nxs"]
    assert [c.path.name for c in result.run_order] == ["scan_2.nxs"]
    assert result.changed == ()
    assert result.baseline_current is False


def test_reconcile_reports_owner_flip_fail_closed(tmp_path):
    with _isolated_adapters():
        index = _index_with(tmp_path, "scan_1.nxs", "scan_2.nxs")
        plan = RunCandidatePlan.from_snapshot(index.snapshot)
        register_adapter(_nxs_override_adapter("nxs_override"))

        result = plan.reconcile(index.poll())

        assert {c.path.name for c in result.owner_flipped} == {
            "scan_1.nxs", "scan_2.nxs"}
        assert result.run_order == ()
        assert result.changed == ()
        assert result.baseline_current is False


# ── RP1: a growing baseline candidate is withheld, not stranded ────────────

def test_reconcile_withholds_a_grown_candidate_as_changed(tmp_path):
    """Same-path stamp change is reported as ``changed`` (the CURRENT
    candidate), withheld from ``run_order`` — never stranded into limbo."""
    index = _index_with(tmp_path, "scan_1.nxs", "scan_2.nxs")
    plan = RunCandidatePlan.from_snapshot(index.snapshot)
    _touch(tmp_path / "scan_1.nxs", b"grown by more bytes than before")

    result = plan.reconcile(index.poll())

    assert [c.path.name for c in result.changed] == ["scan_1.nxs"]
    assert result.changed[0].version_stamp != plan.by_path()[
        tmp_path / "scan_1.nxs"].version_stamp        # it is the CURRENT candidate
    assert [c.path.name for c in result.run_order] == ["scan_2.nxs"]
    assert result.removed == ()
    assert result.baseline_current is False


def test_grown_candidate_is_not_stranded_across_repeated_polls(tmp_path):
    """RP1 core repro: after a stamp change, repeated reconciliation keeps the
    path recoverable via ``changed`` — it is never lost from every bucket."""
    index = _index_with(tmp_path, "growing.nxs")
    plan = RunCandidatePlan.from_snapshot(index.snapshot)
    _touch(tmp_path / "growing.nxs", b"a much longer body")

    for _ in range(3):
        result = plan.reconcile(index.poll())
        # present in exactly one place (changed), never dropped from all buckets
        assert [c.path.name for c in result.changed] == ["growing.nxs"]
        assert result.run_order == ()
        assert result.removed == ()
        assert result.appended == ()


def test_reprobe_ready_then_adopt_consumes_the_growth_exactly_once(tmp_path):
    """The full consumer cycle through the real index probe: grow -> changed ->
    probe_candidate READY -> adopt -> run_order (once); repeated unchanged polls
    do not re-offer it as changed."""
    with _isolated_adapters():
        register_adapter(_grow_adapter())
        _touch(tmp_path / "g.grow", b"grow")             # not yet ready
        index = DirectoryIndex(tmp_path)
        index.poll()
        plan = RunCandidatePlan.from_snapshot(index.snapshot)

        _touch(tmp_path / "g.grow", b"grow ready now")    # finalized
        snap = index.poll()
        result = plan.reconcile(snap)
        assert [c.path.name for c in result.changed] == ["g.grow"]
        assert result.run_order == ()

        current = result.changed[0]
        probe = index.probe_candidate(current)
        assert probe.state is ProbeState.READY
        plan = plan.adopt(current, probe)                 # adopt exactly once

        again = plan.reconcile(index.poll())              # unchanged now
        assert [c.path.name for c in again.run_order] == ["g.grow"]
        assert again.changed == ()
        assert again.baseline_current is True

        # a repeated unchanged poll does not duplicate or re-offer it
        third = plan.reconcile(index.poll())
        assert [c.path.name for c in third.run_order] == ["g.grow"]
        assert third.changed == ()


def test_second_growth_transition_repeats_the_safe_cycle(tmp_path):
    with _isolated_adapters():
        register_adapter(_grow_adapter())
        _touch(tmp_path / "g.grow", b"grow ready")
        index = DirectoryIndex(tmp_path)
        index.poll()
        plan = RunCandidatePlan.from_snapshot(index.snapshot)
        assert plan.reconcile(index.poll()).baseline_current is True

        # grow again -> changed again -> reprobe READY -> adopt -> run_order
        _touch(tmp_path / "g.grow", b"grow ready and even larger")
        result = plan.reconcile(index.poll())
        assert [c.path.name for c in result.changed] == ["g.grow"]
        current = result.changed[0]
        probe = index.probe_candidate(current)
        assert probe.state is ProbeState.READY
        plan = plan.adopt(current, probe)
        assert [c.path.name for c in plan.reconcile(index.snapshot).run_order] \
            == ["g.grow"]


def test_in_progress_growth_stays_retryable_and_does_not_block_ready_sibling(
        tmp_path):
    with _isolated_adapters():
        register_adapter(_grow_adapter())
        _touch(tmp_path / "a.grow", b"grow ready")        # a stays ready
        _touch(tmp_path / "b.grow", b"grow")              # b is a growing shell
        index = DirectoryIndex(tmp_path)
        index.poll()
        plan = RunCandidatePlan.from_snapshot(index.snapshot)

        _touch(tmp_path / "b.grow", b"grow still")        # b grows, still not ready
        result = plan.reconcile(index.poll())

        # a is unchanged and openable; b is withheld pending its reprobe
        assert [c.path.name for c in result.run_order] == ["a.grow"]
        assert [c.path.name for c in result.changed] == ["b.grow"]
        # reprobe b -> IN_PROGRESS -> not adopted, still retryable next poll
        assert index.probe_candidate(result.changed[0]).state \
            is ProbeState.IN_PROGRESS
        later = plan.reconcile(index.poll())
        assert [c.path.name for c in later.run_order] == ["a.grow"]   # never blocked
        assert [c.path.name for c in later.changed] == ["b.grow"]


def test_adopt_rejects_owner_flip_and_unchanged_and_unknown(tmp_path):
    with _isolated_adapters():
        index = _index_with(tmp_path, "a.nxs")
        plan = RunCandidatePlan.from_snapshot(index.snapshot)
        baseline = plan.by_path()[tmp_path / "a.nxs"]

        ready = ProbeResult(ProbeState.READY, reason="ready")
        with pytest.raises(ValueError):                   # unchanged: nothing to adopt
            plan.adopt(baseline, ready)
        with pytest.raises(KeyError):                     # not in baseline
            plan.adopt(
                Candidate(tmp_path / "z.nxs", baseline.adapter_id, 1, 1), ready)
        # owner flip is not an adoptable growth
        flipped = Candidate(tmp_path / "a.nxs", "some_other_owner",
                            baseline.size + 1, baseline.mtime_ns + 1)
        with pytest.raises(StaleCandidateError):
            plan.adopt(flipped, ready)


def test_adopt_requires_ready_probe_evidence(tmp_path):
    """A changed name-only candidate cannot be adopted merely because a caller
    forgot to inspect its probe verdict.  IN_PROGRESS remains withheld."""
    index = _index_with(tmp_path, "a.nxs")
    plan = RunCandidatePlan.from_snapshot(index.snapshot)
    _touch(tmp_path / "a.nxs", b"grown but not proven ready")
    current = plan.reconcile(index.poll()).changed[0]

    with pytest.raises(ValueError, match="READY"):
        plan.adopt(current, ProbeResult(
            ProbeState.IN_PROGRESS, reason="writer is still finalizing"))


def test_reconcile_appends_later_candidates_in_natural_order(tmp_path):
    index = _index_with(tmp_path, "scan_1.nxs", "scan_2.nxs")
    plan = RunCandidatePlan.from_snapshot(index.snapshot)
    _touch(tmp_path / "scan_10.nxs")
    _touch(tmp_path / "scan_3.nxs")

    result = plan.reconcile(index.poll())

    assert [c.path.name for c in result.run_order] == ["scan_1.nxs", "scan_2.nxs"]
    assert result.changed == () and result.removed == ()
    assert [c.path.name for c in result.appended] == ["scan_3.nxs", "scan_10.nxs"]
    assert result.baseline_current is True


def test_reconcile_preserves_frozen_order_not_snapshot_order(tmp_path):
    index = _index_with(tmp_path, "scan_1.nxs", "scan_2.nxs", "scan_10.nxs")
    snap = index.poll()
    reversed_candidates = tuple(reversed(snap.candidates))
    plan = RunCandidatePlan(
        generation=snap.generation, candidates=reversed_candidates,
        root=snap.root, recursive=snap.recursive, name_filter=snap.name_filter)

    result = plan.reconcile(snap)

    assert [c.path.name for c in snap.candidates] == [
        "scan_1.nxs", "scan_2.nxs", "scan_10.nxs"]
    assert [c.path.name for c in result.run_order] == [
        "scan_10.nxs", "scan_2.nxs", "scan_1.nxs"]        # FROZEN order
    assert result.baseline_current is True


# ── RP2: reconcile enforces the frozen source configuration ────────────────

def test_reconcile_rejects_a_different_root(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    plan = RunCandidatePlan.from_snapshot(_index_with(a, "s.nxs").snapshot)
    other = DirectoryIndex(b)
    _touch(b / "s.nxs")
    with pytest.raises(SupersededPlanError):
        plan.reconcile(other.poll())


def test_reconcile_rejects_a_recursion_change(tmp_path):
    _touch(tmp_path / "s.nxs")
    plan = RunCandidatePlan.from_snapshot(DirectoryIndex(tmp_path).poll())
    recursive = DirectoryIndex(tmp_path, recursive=True)
    with pytest.raises(SupersededPlanError):
        plan.reconcile(recursive.poll())


def test_reconcile_rejects_a_name_filter_change(tmp_path):
    _touch(tmp_path / "scan.nxs")
    plan = RunCandidatePlan.from_snapshot(
        DirectoryIndex(tmp_path, name_filter="scan").poll())
    filtered = DirectoryIndex(tmp_path, name_filter="other")
    with pytest.raises(SupersededPlanError):
        plan.reconcile(filtered.poll())


def test_reconcile_rejects_a_coincident_generation_from_another_index(tmp_path):
    """Generation alone is insufficient: two separate indexes on different roots
    can share a generation number; config identity is what rejects it."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _touch(a / "s.nxs")
    _touch(b / "s.nxs")
    ia = DirectoryIndex(a)
    ib = DirectoryIndex(b)
    snap_a = ia.poll()
    snap_b = ib.poll()
    assert snap_a.generation == snap_b.generation      # coincident generation
    plan = RunCandidatePlan.from_snapshot(snap_a)
    with pytest.raises(SupersededPlanError):
        plan.reconcile(snap_b)


def test_superseded_plan_error_is_a_valueerror(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _touch(a / "s.nxs")
    plan = RunCandidatePlan.from_snapshot(DirectoryIndex(a).poll())
    _touch(b / "s.nxs")
    try:
        plan.reconcile(DirectoryIndex(b).poll())
    except ValueError as exc:
        assert isinstance(exc, SupersededPlanError)
    else:
        raise AssertionError("expected SupersededPlanError")


# ── RP3: O(1) per-candidate lookup + 3600-candidate benchmark ──────────────

def test_by_path_is_cached_plan_owned_state(tmp_path):
    index = _index_with(tmp_path, "scan_1.nxs", "scan_2.nxs")
    plan = RunCandidatePlan.from_snapshot(index.snapshot)
    # the lookup is built once and reused (same object) — not rebuilt per call
    assert plan.by_path() is plan.by_path()


def test_validate_is_o1_over_a_large_plan_benchmark(tmp_path):
    """3600-candidate benchmark: validate() must not rebuild an N-entry map per
    call (RP3 measured ~0.646 s for the O(N^2) version at 2d79037b)."""
    cands = tuple(
        Candidate(tmp_path / f"scan_{i:05d}.nxs", "nexus_hdf5", 100, 1000 + i)
        for i in range(3600))
    plan = RunCandidatePlan(
        generation=1, candidates=cands, root=tmp_path,
        recursive=False, name_filter=None)

    start = time.perf_counter()
    for c in plan.candidates:
        assert plan.validate(c) is c
    elapsed = time.perf_counter() - start

    # generous bound, an order of magnitude under the old O(N^2) 0.646 s
    assert elapsed < 0.25, f"3600 validate() calls took {elapsed:.3f}s"


# ── value-only / immutability ─────────────────────────────────────────────

def test_plan_and_result_are_frozen_value_types(tmp_path):
    index = _index_with(tmp_path, "a.nxs")
    plan = RunCandidatePlan.from_snapshot(index.snapshot)
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.candidates = ()                       # type: ignore[misc]
    result = plan.reconcile(index.poll())
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.run_order = ()                      # type: ignore[misc]
    assert all(isinstance(c, Candidate) for c in plan.candidates)


def test_reconcile_result_has_no_all_ready_name(tmp_path):
    """RP4: the identity-only property is baseline_current; the misleading
    all_ready name is gone."""
    index = _index_with(tmp_path, "a.nxs")
    result = RunCandidatePlan.from_snapshot(index.snapshot).reconcile(index.poll())
    assert hasattr(result, "baseline_current")
    assert not hasattr(result, "all_ready")


def test_stale_candidate_error_is_a_valueerror(tmp_path):
    index = _index_with(tmp_path, "a.nxs")
    plan = RunCandidatePlan.from_snapshot(index.snapshot)
    (tmp_path / "a.nxs").unlink()
    index.poll()
    try:
        plan.validate(None, path=tmp_path / "a.nxs")
    except ValueError as exc:
        assert isinstance(exc, StaleCandidateError)
    else:
        raise AssertionError("expected StaleCandidateError")
