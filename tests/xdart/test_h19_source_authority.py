# -*- coding: utf-8 -*-
"""H19 Source-card authority at the GUI -> processing boundary."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

pytest.importorskip("pyqtgraph")

from xrd_tools.core.scan import SourceKind
from xrd_tools.sources.directory_index import IndexDelta, Snapshot
from xrd_tools.sources.directory_session import (
    CandidateObservation,
    DirectoryObservation,
)
from xrd_tools.sources.discover import Candidate
from xrd_tools.sources.probe import ProbeResult, ProbeState
from xrd_tools.sources.run_plan import RunCandidatePlan


def _candidate(root: Path, name: str, stamp: int = 1, owner="nexus_hdf5"):
    return Candidate(root / name, owner, stamp, stamp)


def _result(state=ProbeState.READY):
    return ProbeResult(state, kind=SourceKind.NEXUS_STACK)


def _observation(root, rows, generation=1):
    candidates = tuple(candidate for candidate, _ in rows)
    snapshot = Snapshot(generation, candidates, root, False, None)
    ready = tuple(
        candidate for candidate, result in rows
        if result.state is ProbeState.READY
    )
    return DirectoryObservation(
        request_generation=1,
        discovered_snapshot=snapshot,
        ready_snapshot=Snapshot(generation, ready, root, False, None),
        candidates=tuple(CandidateObservation(*row) for row in rows),
        delta=IndexDelta(),
        content_opens=0,
        stale_drops=0,
        elapsed_s=0.0,
    )


class _Session:
    def __init__(self, *observations):
        self._observations = deque(observations)
        self.calls = 0

    def observe(self):
        self.calls += 1
        if len(self._observations) > 1:
            return self._observations.popleft()
        return self._observations[0]


class _RaisingSession:
    def __init__(self, exc):
        self.exc = exc

    def observe(self):
        raise self.exc


class _Signal:
    def __init__(self):
        self.values = []

    def emit(self, value):
        self.values.append(value)


def _worker(plan, session):
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
        imageThread,
    )

    host = SimpleNamespace(
        source_run_plan=plan,
        source_index_session=session,
        _source_plan_reported=set(),
        _eiger_master_queue=deque(),
        _eiger_done_masters=set(),
        _eiger_retry_after={},
        _eiger_zero_frame_seen={},
        showLabel=_Signal(),
        live_mode=True,
        inp_type="Image Directory",
        command="start",
    )
    host._h19_ready_master_paths = MethodType(
        imageThread._h19_ready_master_paths, host)
    host._eiger_refill_master_queue = MethodType(
        imageThread._eiger_refill_master_queue, host)
    host._eiger_pop_next_master = MethodType(
        imageThread._eiger_pop_next_master, host)
    host._h19_live_directory_armed = MethodType(
        imageThread._h19_live_directory_armed, host)
    return host


def test_empty_frozen_baseline_arms_and_accepts_first_ready_append(tmp_path):
    source = _candidate(tmp_path, "scan_1.nxs")
    plan = RunCandidatePlan.from_snapshot(
        Snapshot(0, (), tmp_path, False, None))
    worker = _worker(
        plan,
        _Session(_observation(tmp_path, ((source, _result()),), generation=1)),
    )

    assert worker._h19_live_directory_armed() is True
    worker._eiger_refill_master_queue()
    assert list(worker._eiger_master_queue) == [str(source.path)]


def test_authoritative_refill_honors_zero_frame_retry_clock(
    tmp_path, monkeypatch,
):
    import xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread as iwt

    source = _candidate(tmp_path, "scan_1.nxs")
    plan = RunCandidatePlan.from_snapshot(
        Snapshot(1, (source,), tmp_path, False, None))
    observation = _observation(tmp_path, ((source, _result()),), generation=1)
    worker = _worker(plan, _Session(observation))
    worker._eiger_retry_after[str(source.path)] = 100.0

    monkeypatch.setattr(iwt.time, "monotonic", lambda: 50.0)
    worker._eiger_refill_master_queue()
    assert list(worker._eiger_master_queue) == []

    monkeypatch.setattr(iwt.time, "monotonic", lambda: 101.0)
    worker._eiger_refill_master_queue()
    assert list(worker._eiger_master_queue) == [str(source.path)]


def test_observation_failure_preserves_queue_and_surfaces_status(tmp_path):
    source = _candidate(tmp_path, "scan_1.nxs")
    plan = RunCandidatePlan.from_snapshot(
        Snapshot(1, (source,), tmp_path, False, None))
    worker = _worker(plan, _RaisingSession(OSError("share unavailable")))
    worker._eiger_master_queue.append(str(source.path))

    with pytest.raises(RuntimeError, match="share unavailable"):
        worker._eiger_pop_next_master()

    assert list(worker._eiger_master_queue) == [str(source.path)]
    assert worker.showLabel.values
    assert "temporarily unavailable" in worker.showLabel.values[-1]


def test_worker_uses_frozen_order_and_ready_appends_without_reglob(
    tmp_path, monkeypatch,
):
    import xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread as iwt

    scan2 = _candidate(tmp_path, "scan_2.nxs")
    scan10 = _candidate(tmp_path, "scan_10.nxs")
    scan11 = _candidate(tmp_path, "scan_11.nxs")
    baseline = Snapshot(1, (scan2, scan10), tmp_path, False, None)
    plan = RunCandidatePlan.from_snapshot(baseline)
    observation = _observation(
        tmp_path,
        ((scan2, _result()), (scan10, _result()), (scan11, _result())),
        generation=2,
    )
    worker = _worker(plan, _Session(observation))

    monkeypatch.setattr(
        iwt,
        "_paths_with_suffix",
        lambda *_args, **_kwargs: pytest.fail("H19 worker re-globbed"),
    )
    worker._eiger_refill_master_queue()
    assert [Path(path).name for path in worker._eiger_master_queue] == [
        "scan_2.nxs", "scan_10.nxs", "scan_11.nxs",
    ]


def test_changed_pending_candidate_does_not_block_ready_sibling(tmp_path):
    original = _candidate(tmp_path, "scan_1.nxs", stamp=1)
    sibling = _candidate(tmp_path, "scan_2.nxs", stamp=1)
    changed = _candidate(tmp_path, "scan_1.nxs", stamp=2)
    plan = RunCandidatePlan.from_snapshot(
        Snapshot(1, (original, sibling), tmp_path, False, None))
    observation = _observation(
        tmp_path,
        ((changed, _result(ProbeState.IN_PROGRESS)), (sibling, _result())),
        generation=2,
    )

    worker = _worker(plan, _Session(observation))
    assert worker._h19_ready_master_paths() == (sibling.path,)
    assert worker.source_run_plan.paths == (original.path, sibling.path)


def test_changed_ready_candidate_is_adopted_once_in_original_slot(tmp_path):
    original = _candidate(tmp_path, "scan_1.nxs", stamp=1)
    sibling = _candidate(tmp_path, "scan_2.nxs", stamp=1)
    changed = _candidate(tmp_path, "scan_1.nxs", stamp=2)
    plan = RunCandidatePlan.from_snapshot(
        Snapshot(1, (original, sibling), tmp_path, False, None))
    observation = _observation(
        tmp_path,
        ((changed, _result()), (sibling, _result())),
        generation=2,
    )
    session = _Session(observation)
    worker = _worker(plan, session)

    assert worker._h19_ready_master_paths() == (changed.path, sibling.path)
    adopted = worker.source_run_plan
    assert adopted.candidates[0] == changed
    assert worker._h19_ready_master_paths() == (changed.path, sibling.path)
    assert worker.source_run_plan == adopted


def test_removed_owner_flip_and_nonready_append_fail_closed(tmp_path):
    removed = _candidate(tmp_path, "gone.nxs")
    prior_owner = _candidate(tmp_path, "owner.nxs", owner="nexus_hdf5")
    flipped = _candidate(tmp_path, "owner.nxs", owner="external")
    pending = _candidate(tmp_path, "new.nxs")
    plan = RunCandidatePlan.from_snapshot(
        Snapshot(1, (removed, prior_owner), tmp_path, False, None))
    observation = _observation(
        tmp_path,
        ((flipped, _result()), (pending, _result(ProbeState.IN_PROGRESS))),
        generation=2,
    )
    worker = _worker(plan, _Session(observation))

    assert worker._h19_ready_master_paths() == ()
    assert ("removed", str(removed.path)) in worker._source_plan_reported
    assert ("owner", str(flipped.path)) in worker._source_plan_reported


def test_pop_revalidates_queue_and_never_consumes_candidate_that_went_pending(
    tmp_path,
):
    source = _candidate(tmp_path, "scan.nxs")
    plan = RunCandidatePlan.from_snapshot(
        Snapshot(1, (source,), tmp_path, False, None))
    ready = _observation(tmp_path, ((source, _result()),), generation=1)
    pending = _observation(
        tmp_path, ((source, _result(ProbeState.IN_PROGRESS)),), generation=1)
    worker = _worker(plan, _Session(ready, pending))

    worker._eiger_refill_master_queue()
    assert list(worker._eiger_master_queue) == [str(source.path)]
    assert worker._eiger_pop_next_master() is None
