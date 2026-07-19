# -*- coding: utf-8 -*-
"""H19 §4 — ``RunCandidatePlan``: the immutable Source-card -> Run handoff seam.

Production-wired: a real :class:`DirectoryIndex` over real temp files produces
the :class:`Snapshot` the plan freezes; the filesystem is really mutated
(removed / rewritten / adapter-owner-flipped) and re-polled to drive the
stale-candidate contract.  No fakes on the seam under test — the only injected
double is a higher-precedence real ``SourceFormatAdapter`` to flip ownership,
exactly as the DirectoryIndex owner-flip tests do.
"""

from __future__ import annotations

import contextlib
import dataclasses

import pytest

from xrd_tools.core.scan import SourceKind
from xrd_tools.sources import RunCandidatePlan, RunReconcile
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
    """A higher-precedence real adapter that claims ``.nxs`` by name so a
    registration flips the owning adapter of an existing candidate without a
    byte change (externally-registered, most-recent wins)."""
    return SourceFormatAdapter(
        id=idn, kinds=(SourceKind.TILED,),
        is_candidate=lambda p: p.suffix == ".nxs",
        scan_name=lambda p: p.stem,
        probe=lambda p: ProbeResult(
            ProbeState.READY, reason=idn, kind=SourceKind.TILED),
        open=lambda spec: None)


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
    # natural order (scan_1 < scan_2 < scan_10), preserved from the snapshot
    assert [p.name for p in plan.paths] == [
        "scan_1.nxs", "scan_2.nxs", "scan_10.nxs"]
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
        plan.validate(current)


def test_validate_fails_closed_on_owner_flip(tmp_path):
    with _isolated_adapters():
        index = _index_with(tmp_path, "a.nxs")
        plan = RunCandidatePlan.from_snapshot(index.snapshot)
        baseline = plan.by_path()[tmp_path / "a.nxs"]
        register_adapter(_nxs_override_adapter("nxs_override"))
        current = index.poll().by_path()[tmp_path / "a.nxs"]
        assert current.adapter_id == "nxs_override"
        assert current.adapter_id != baseline.adapter_id       # owner flipped
        assert current.version_stamp == baseline.version_stamp  # bytes unchanged
        with pytest.raises(StaleCandidateError):
            plan.validate(current)


def test_validate_rejects_live_appended_path_with_keyerror(tmp_path):
    index = _index_with(tmp_path, "a.nxs")
    plan = RunCandidatePlan.from_snapshot(index.snapshot)
    _touch(tmp_path / "b.nxs")
    appeared = index.poll().by_path()[tmp_path / "b.nxs"]
    with pytest.raises(KeyError):
        plan.validate(appeared)                               # not in baseline


def test_validate_needs_a_path_when_current_is_none(tmp_path):
    plan = RunCandidatePlan.from_snapshot(_index_with(tmp_path, "a.nxs").snapshot)
    with pytest.raises(ValueError):
        plan.validate(None)


# ── reconcile: run order + fail-closed stale + live appends ────────────────

def test_reconcile_unchanged_returns_frozen_run_order(tmp_path):
    index = _index_with(tmp_path, "scan_1.nxs", "scan_2.nxs")
    plan = RunCandidatePlan.from_snapshot(index.snapshot)

    result = plan.reconcile(index.poll())                     # unchanged

    assert isinstance(result, RunReconcile)
    assert result.run_order == plan.candidates
    assert result.stale == ()
    assert result.appended == ()
    assert result.all_ready is True


def test_reconcile_holds_back_removed_candidate(tmp_path):
    index = _index_with(tmp_path, "scan_1.nxs", "scan_2.nxs")
    plan = RunCandidatePlan.from_snapshot(index.snapshot)
    (tmp_path / "scan_1.nxs").unlink()

    result = plan.reconcile(index.poll())

    assert [p.name for p in result.stale] == ["scan_1.nxs"]
    assert [c.path.name for c in result.run_order] == ["scan_2.nxs"]
    assert result.appended == ()
    assert result.all_ready is False


def test_reconcile_holds_back_stamp_changed_candidate(tmp_path):
    index = _index_with(tmp_path, "scan_1.nxs", "scan_2.nxs")
    plan = RunCandidatePlan.from_snapshot(index.snapshot)
    _touch(tmp_path / "scan_1.nxs", b"rewritten with a different size stamp")

    result = plan.reconcile(index.poll())

    assert [p.name for p in result.stale] == ["scan_1.nxs"]
    assert [c.path.name for c in result.run_order] == ["scan_2.nxs"]
    assert result.all_ready is False


def test_reconcile_holds_back_owner_flipped_candidate(tmp_path):
    with _isolated_adapters():
        index = _index_with(tmp_path, "scan_1.nxs", "scan_2.nxs")
        plan = RunCandidatePlan.from_snapshot(index.snapshot)
        register_adapter(_nxs_override_adapter("nxs_override"))

        result = plan.reconcile(index.poll())

        # both paths' owners flipped -> both are stale, none processed
        assert {p.name for p in result.stale} == {"scan_1.nxs", "scan_2.nxs"}
        assert result.run_order == ()
        assert result.all_ready is False


def test_reconcile_appends_later_candidates_in_natural_order(tmp_path):
    index = _index_with(tmp_path, "scan_1.nxs", "scan_2.nxs")
    plan = RunCandidatePlan.from_snapshot(index.snapshot)
    _touch(tmp_path / "scan_10.nxs")      # appears AFTER the plan was frozen
    _touch(tmp_path / "scan_3.nxs")

    result = plan.reconcile(index.poll())

    # frozen baseline still runs, in frozen order
    assert [c.path.name for c in result.run_order] == ["scan_1.nxs", "scan_2.nxs"]
    assert result.stale == ()
    # live-discovered scans appended in natural order (scan_3 < scan_10)
    assert [c.path.name for c in result.appended] == ["scan_3.nxs", "scan_10.nxs"]
    assert result.all_ready is True


def test_reconcile_preserves_frozen_order_not_snapshot_order(tmp_path):
    """run_order follows the FROZEN plan order, never a re-globbed snapshot
    order — proven by a hand-built plan whose candidates are in reverse of the
    natural-sorted snapshot the index produces."""
    index = _index_with(tmp_path, "scan_1.nxs", "scan_2.nxs", "scan_10.nxs")
    snap = index.poll()
    by_path = snap.by_path()
    # deliberately non-natural (reversed) baseline order
    reversed_candidates = tuple(reversed(snap.candidates))
    plan = RunCandidatePlan(
        generation=snap.generation, candidates=reversed_candidates,
        root=snap.root, recursive=snap.recursive, name_filter=snap.name_filter)

    result = plan.reconcile(snap)

    # snapshot natural order is scan_1, scan_2, scan_10; the plan is the reverse
    assert [c.path.name for c in snap.candidates] == [
        "scan_1.nxs", "scan_2.nxs", "scan_10.nxs"]
    assert [c.path.name for c in result.run_order] == [
        "scan_10.nxs", "scan_2.nxs", "scan_1.nxs"]
    assert result.stale == ()
    assert all(c is by_path[c.path] for c in result.run_order)  # current values


# ── value-only / immutability ─────────────────────────────────────────────

def test_plan_and_result_are_frozen_value_types(tmp_path):
    index = _index_with(tmp_path, "a.nxs")
    plan = RunCandidatePlan.from_snapshot(index.snapshot)
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.candidates = ()                       # type: ignore[misc]
    result = plan.reconcile(index.poll())
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.run_order = ()                      # type: ignore[misc]
    # the plan carries only immutable Candidate values — no open handle / source
    assert all(isinstance(c, Candidate) for c in plan.candidates)


def test_stale_candidate_error_is_a_valueerror(tmp_path):
    index = _index_with(tmp_path, "a.nxs")
    plan = RunCandidatePlan.from_snapshot(index.snapshot)
    (tmp_path / "a.nxs").unlink()
    index.poll()
    try:
        plan.validate(None, path=tmp_path / "a.nxs")
    except ValueError as exc:               # existing except ValueError catches it
        assert isinstance(exc, StaleCandidateError)
    else:
        raise AssertionError("expected StaleCandidateError")
