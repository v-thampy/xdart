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


def _file_candidate(path: Path, owner="nexus_hdf5"):
    stat = path.stat()
    return Candidate(
        path, owner, int(stat.st_size), int(stat.st_mtime_ns))


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

    def observe(self, **_kwargs):
        self.calls += 1
        if len(self._observations) > 1:
            return self._observations.popleft()
        return self._observations[0]


class _RaisingSession:
    def __init__(self, exc):
        self.exc = exc

    def observe(self, **_kwargs):
        raise self.exc


class _Signal:
    def __init__(self):
        self.values = []

    def emit(self, value):
        self.values.append(value)


class _ManySignal:
    def __init__(self):
        self.values = []

    def emit(self, *values):
        self.values.append(values)


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
        _h19_seed_pending=False,
        _h19_ready_master_candidates={},
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
        worker._eiger_refill_master_queue()

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


def test_pop_point_revalidates_and_never_consumes_changed_candidate(tmp_path):
    path = tmp_path / "scan.nxs"
    path.write_bytes(b"ready")
    source = _file_candidate(path)
    plan = RunCandidatePlan.from_snapshot(
        Snapshot(1, (source,), tmp_path, False, None))
    ready = _observation(tmp_path, ((source, _result()),), generation=1)
    session = _Session(ready)
    worker = _worker(plan, session)

    worker._eiger_refill_master_queue()
    assert list(worker._eiger_master_queue) == [str(source.path)]
    path.write_bytes(b"changed bytes")
    assert worker._eiger_pop_next_master() is None
    # One authoritative follow-up is allowed, but the same sticky stale
    # identity must then yield rather than spin/requeue forever.
    assert session.calls == 2


def test_first_refill_uses_frozen_ready_seed_without_observation(tmp_path):
    path = tmp_path / "scan.nxs"
    path.write_bytes(b"ready")
    source = _file_candidate(path)
    plan = RunCandidatePlan.from_snapshot(
        Snapshot(1, (source,), tmp_path, False, None))
    worker = _worker(
        plan, _RaisingSession(AssertionError("seed re-observed")))
    worker._h19_seed_pending = True
    worker._h19_ready_master_candidates = {str(path): source}

    worker._eiger_refill_master_queue()

    assert worker._eiger_pop_next_master() == str(path)


def test_pop_does_not_reobserve_each_already_queued_master(tmp_path):
    paths = (tmp_path / "scan_1.nxs", tmp_path / "scan_2.nxs")
    for path in paths:
        path.write_bytes(path.name.encode())
    sources = tuple(_file_candidate(path) for path in paths)
    plan = RunCandidatePlan.from_snapshot(
        Snapshot(1, sources, tmp_path, False, None))
    observation = _observation(
        tmp_path, tuple((source, _result()) for source in sources),
        generation=1,
    )
    session = _Session(observation)
    worker = _worker(plan, session)

    worker._eiger_refill_master_queue()
    popped = (
        worker._eiger_pop_next_master(),
        worker._eiger_pop_next_master(),
    )

    assert popped == tuple(str(path) for path in paths)
    assert session.calls == 1


def test_seed_drain_observes_only_when_frozen_plan_has_pending_work(tmp_path):
    first_path = tmp_path / "scan_1.nxs"
    second_path = tmp_path / "scan_2.nxs"
    first_path.write_bytes(b"first")
    second_path.write_bytes(b"second")
    first = _file_candidate(first_path)
    second = _file_candidate(second_path)
    plan = RunCandidatePlan.from_snapshot(
        Snapshot(1, (first,), tmp_path, False, None))
    followup = _observation(
        tmp_path,
        ((first, _result()), (second, _result())),
        generation=2,
    )
    session = _Session(followup)
    worker = _worker(plan, session)
    worker._h19_seed_pending = True
    worker._h19_pending_count = 1
    worker._h19_ready_master_candidates = {str(first.path): first}

    def skip_first(path, _candidate):
        if path != str(first.path):
            return False
        worker._eiger_done_masters.add(path)
        return True

    worker._eiger_skip_complete_append_master = skip_first
    worker._eiger_refill_master_queue()

    assert worker._eiger_pop_next_master() == str(second.path)
    assert session.calls == 1


def test_real_session_observes_only_after_frozen_queue_exhausts(
        tmp_path, monkeypatch):
    import h5py
    import numpy as np
    from xrd_tools.sources.directory_session import DirectoryIndexSession

    paths = (tmp_path / "scan_1.nxs", tmp_path / "scan_2.nxs")
    for path in paths:
        with h5py.File(path, "w") as handle:
            detector = handle.create_group("entry/instrument/detector")
            detector.create_dataset(
                "data", data=np.zeros((2, 3, 4), dtype=np.uint16))

    session = DirectoryIndexSession()
    try:
        session.configure(tmp_path, suffixes=(".nxs",))
        observation = session.observe()
        assert len(observation.ready_snapshot.candidates) == 2
        plan = RunCandidatePlan.from_snapshot(observation.ready_snapshot)
        worker = _worker(plan, session)
        worker._h19_seed_pending = True
        worker._h19_ready_master_candidates = {
            str(candidate.path): candidate for candidate in plan.candidates}

        calls = 0
        real_observe = session.observe

        def observe(*, refresh=True):
            nonlocal calls
            calls += 1
            return real_observe(refresh=refresh)

        monkeypatch.setattr(session, "observe", observe)

        worker._eiger_refill_master_queue()
        popped = (
            worker._eiger_pop_next_master(),
            worker._eiger_pop_next_master(),
        )
        assert popped == tuple(str(path) for path in paths)
        assert calls == 0

        worker._eiger_done_masters.update(popped)
        worker._eiger_refill_master_queue()
        assert calls == 1
        assert list(worker._eiger_master_queue) == []
    finally:
        session.close()


def test_real_session_converges_unprobed_tail_after_seed_bulk_skip(
        tmp_path, monkeypatch):
    import h5py
    import numpy as np
    from xrd_tools.sources.directory_session import DirectoryIndexSession

    paths = tuple(tmp_path / f"scan_{idx}.nxs" for idx in range(6))
    for path in paths:
        with h5py.File(path, "w") as handle:
            detector = handle.create_group("entry/instrument/detector")
            detector.create_dataset(
                "data", data=np.zeros((2, 3, 4), dtype=np.uint16))

    session = DirectoryIndexSession(max_probes_per_observation=2)
    try:
        session.configure(tmp_path, suffixes=(".nxs",))
        observation = session.observe()
        assert len(observation.ready_snapshot.candidates) == 2
        assert observation.pending_count == 4
        plan = RunCandidatePlan.from_snapshot(observation.ready_snapshot)
        worker = _worker(plan, session)
        worker._h19_seed_pending = True
        worker._h19_pending_count = observation.pending_count
        worker._h19_ready_master_candidates = {
            str(candidate.path): candidate for candidate in plan.candidates}

        calls = []
        real_observe = session.observe

        def observe(*, refresh=True):
            calls.append(refresh)
            return real_observe(refresh=refresh)

        monkeypatch.setattr(session, "observe", observe)

        def skip(path, _candidate):
            worker._eiger_done_masters.add(path)
            return True

        worker._eiger_skip_complete_append_master = skip
        worker._eiger_refill_master_queue()

        assert worker._eiger_pop_next_master() is None
        assert worker._eiger_done_masters == {str(path) for path in paths}
        assert worker._h19_pending_count == 0
        assert calls == [False, False]
    finally:
        session.close()


@pytest.mark.parametrize("external", (False, True))
def test_container_count_optimization_requires_self_contained_final_cursor(
        tmp_path, external):
    import h5py
    import numpy as np
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
        imageThread,
    )
    from xrd_tools.sources.cursor import ContainerCursor

    master = tmp_path / "scan.nxs"
    if external:
        target = tmp_path / "target.h5"
        with h5py.File(target, "w") as handle:
            handle.create_dataset(
                "frames", data=np.zeros((2, 3, 4), dtype=np.uint16))
        with h5py.File(master, "w") as handle:
            detector = handle.create_group("entry/instrument/detector")
            detector["data"] = h5py.ExternalLink(target.name, "/frames")
    else:
        with h5py.File(master, "w") as handle:
            detector = handle.create_group("entry/instrument/detector")
            detector.create_dataset(
                "data", data=np.zeros((2, 3, 4), dtype=np.uint16))

    candidate = _file_candidate(master)
    cursor = ContainerCursor(master, candidate=candidate).open()
    try:
        signal = _ManySignal()
        host = SimpleNamespace(
            sigContainerCount=signal,
            _eiger_master_candidate=candidate,
            _eiger_cursor=cursor,
            _eiger_descriptor=cursor.descriptor,
        )

        imageThread._emit_container_count(
            host, master, cursor.frame_count, authoritative=True)

        assert signal.values == [(
            str(master),
            2,
            candidate.version_stamp,
            not external,
        )]
    finally:
        cursor.close()
