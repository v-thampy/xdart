"""C2 public-seam composition discriminators.

The rows intentionally drive E6 discovery values into the accepted H10
dynamic owner and then into the one high-level H23 LiveScan session.  They do
not import or inspect writer, transaction, lease, or target internals.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import ast
import gc
import inspect
import threading
import time

import h5py
import numpy as np
import pytest

from tests.core._vnext_p0_c2_bridge_support import (
    DeterministicIntegrator,
    accounting_for,
    append_intent,
    discover,
    live_scan,
    observe_source_fact,
    open_session,
    retryable_attempt,
    successful_attempt,
    submission_attempt,
)


def _rows(target: Path) -> tuple[int, ...]:
    with h5py.File(target, "r") as handle:
        return tuple(
            int(value)
            for value in handle["entry/integrated_1d/frame_index"][()]
        )


def _write_ready_nexus(target: Path, frames: int = 1) -> None:
    with h5py.File(target, "w") as handle:
        entry = handle.create_group("entry")
        detector = entry.create_group("instrument/detector")
        detector.create_dataset(
            "data",
            data=np.arange(frames * 12, dtype=np.uint16).reshape(frames, 3, 4),
        )
        entry.create_dataset("end_time", data=np.bytes_("2026-08-07T00:00:00"))


@pytest.mark.parametrize(
    "filename", ("growing.nxs", "growing.h5", "armed_master.h5"),
)
def test_growing_container_retries_same_key_and_extends_one_live_owner(
    tmp_path,
    filename,
):
    from xrd_tools.sources import ProbeState, open_source
    from xrd_tools.sources.directory_session import DirectoryIndexSession

    source = tmp_path / filename
    linked = tmp_path / "armed_data_000001.h5"
    ordinary = filename != "armed_master.h5"
    if filename == "armed_master.h5":
        with h5py.File(source, "w") as handle:
            entry = handle.create_group("entry")
            entry.attrs["NX_class"] = "NXentry"
            data = entry.create_group("data")
            data.attrs["NX_class"] = "NXdata"
            data["data_000001"] = h5py.ExternalLink(
                str(linked), "/entry/data/data",
            )
        suffixes = ("_master.h5",)
    else:
        initial_value = 3 if filename.endswith(".nxs") else 5
        with h5py.File(source, "w") as handle:
            entry = handle.create_group("entry")
            detector = entry.create_group("instrument/detector")
            detector.create_dataset(
                "data", data=np.full((1, 3, 4), initial_value, dtype=np.uint16),
                maxshape=(None, 3, 4), chunks=(1, 3, 4),
            )
        suffixes = (Path(filename).suffix,)
    source_owner = DirectoryIndexSession(probe_candidates=False)
    source_owner.configure(tmp_path, suffixes=suffixes)
    discovered = source_owner.observe()
    candidate = next(
        item for item in discovered.discovered_snapshot.candidates
        if item.path == source
    )
    provisional = source_owner.probe_candidate(candidate, refresh=False)
    expected_initial_state = (
        ProbeState.READY if ordinary else ProbeState.IN_PROGRESS
    )
    assert provisional.result.state is expected_initial_state
    assert provisional.result.descriptor is not None
    assert provisional.result.descriptor.state is expected_initial_state
    if ordinary:
        opened = open_source(source)
        try:
            np.testing.assert_array_equal(
                opened.load_frame(0),
                np.full((3, 4), initial_value, dtype=np.uint16),
            )
        finally:
            close = getattr(opened, "close", None)
            if callable(close):
                close()
    first_fact = observe_source_fact(tmp_path, filename, logical_identity=0)
    target = tmp_path / f"{Path(filename).stem}-processed.nexus"
    _ledger, accounting, mode, target_name = accounting_for(target)
    key0 = discover(accounting, first_fact, group="detector", ordinal=0, label=0)
    if ordinary:
        first_attempt = successful_attempt(
            accounting, key0, first_fact.source_revision, mode,
        )
        first_intent = append_intent(
            tmp_path, extent=1, labels=(0,), generation=0,
            source_identity=first_fact.source_identity,
        )
        live = live_scan(target, first_intent, (0,))
        session = open_session(live, accounting)
        session.flush(force=True)
        session.commit_epoch()
    else:
        failed = retryable_attempt(
            accounting, key0, first_fact.source_revision,
            "missing Eiger ExternalLink dependency",
        )

    if filename == "armed_master.h5":
        master_stamp = source.stat().st_mtime_ns
        with h5py.File(linked, "w") as handle:
            handle.create_dataset(
                "entry/data/data",
                data=np.stack((
                    np.full((3, 4), 11, dtype=np.uint16),
                    np.full((3, 4), 13, dtype=np.uint16),
                )),
            )
        assert source.stat().st_mtime_ns == master_stamp
    else:
        with h5py.File(source, "a") as handle:
            data = handle["entry/instrument/detector/data"]
            data.resize((2, 3, 4))
            data[1] = np.full((3, 4), initial_value + 4, dtype=np.uint16)
    current = source_owner.observe().discovered_snapshot.candidates
    candidate = next(item for item in current if item.path == source)
    ready = source_owner.probe_candidate(candidate, refresh=False)
    assert ready.result.state is ProbeState.READY
    opened = open_source(source)
    try:
        expected0 = 11 if not ordinary else initial_value
        expected1 = 13 if not ordinary else initial_value + 4
        np.testing.assert_array_equal(
            opened.load_frame(0), np.full((3, 4), expected0, dtype=np.uint16),
        )
        np.testing.assert_array_equal(
            opened.load_frame(1), np.full((3, 4), expected1, dtype=np.uint16),
        )
    finally:
        close = getattr(opened, "close", None)
        if callable(close):
            close()
    retry_fact = observe_source_fact(tmp_path, filename, logical_identity=0)
    if ordinary:
        successor = first_attempt
    else:
        successor = successful_attempt(
            accounting,
            key0,
            max(first_fact.source_revision + 1, retry_fact.source_revision),
            mode,
        )
        first_intent = append_intent(
            tmp_path, extent=1, labels=(0,), generation=0,
            source_identity=first_fact.source_identity,
        )
        live = live_scan(target, first_intent, (0,))
        session = open_session(live, accounting)
        session.flush(force=True)
        session.commit_epoch()
    assert accounting.owner_census().count(session) == 1

    second_fact = observe_source_fact(tmp_path, filename, logical_identity=1)
    key1 = discover(accounting, second_fact, group="detector", ordinal=1, label=1)
    second = successful_attempt(
        accounting,
        key1,
        max(successor.source_revision + 1, second_fact.source_revision),
        mode,
    )
    session.extend(append_intent(
        tmp_path, extent=2, labels=(0, 1), generation=1,
        source_identity=first_fact.source_identity,
    ))
    live.frames.add(1)
    session.flush(force=True)
    session.commit_epoch()
    session.finish(finalize=True)

    snapshot = accounting.snapshot()
    assert snapshot.attempts[key0] == (
        (successor,) if ordinary else (failed, successor)
    )
    assert snapshot.attempts[key1] == (second,)
    assert snapshot.durable_attempts[(key0, mode, target_name)] is successor
    assert snapshot.durable_attempts[(key1, mode, target_name)] is second
    assert _rows(target) == (0, 1)
    assert session not in accounting.owner_census()
    source_owner.close()


@pytest.mark.parametrize("truncation", ("incomplete_object", "decoder_exception"))
def test_partial_tiff_retry_has_no_receipt_then_same_key_succeeds(
    tmp_path, truncation,
):
    from xrd_tools.sources import ProbeState, open_source
    from xrd_tools.sources.directory_session import DirectoryIndexSession

    source = tmp_path / "frame_0001.tif"
    tifffile = pytest.importorskip("tifffile")
    image = np.arange(12, dtype=np.uint16).reshape(3, 4)
    complete = tmp_path / "complete.tif"
    tifffile.imwrite(complete, image)
    complete_bytes = complete.read_bytes()
    complete.unlink()
    cutoff = len(complete_bytes) // 3 if truncation == "incomplete_object" else 8
    source.write_bytes(complete_bytes[:cutoff])
    source_owner = DirectoryIndexSession(probe_candidates=False)
    source_owner.configure(tmp_path, suffixes=(".tif",))
    candidate = source_owner.observe().discovered_snapshot.candidates[0]
    assert source_owner.probe_candidate(candidate, refresh=False).result.state \
        is ProbeState.IN_PROGRESS
    partial = observe_source_fact(tmp_path, source.name, logical_identity=0)
    target = tmp_path / f"partial-tiff-{truncation}.nexus"
    ledger, accounting, mode, target_name = accounting_for(target)
    key = discover(accounting, partial, group="tiff", ordinal=0, label=0)
    failed = retryable_attempt(accounting, key, partial.source_revision, "partial TIFF")
    assert ledger.snapshot().persisted == ledger.snapshot().durable == frozenset()

    source.write_bytes(complete_bytes)
    candidate = source_owner.observe().discovered_snapshot.candidates[0]
    ready = source_owner.probe_candidate(candidate, refresh=False)
    assert ready.result.state is ProbeState.READY
    opened = open_source(source)
    assert np.array_equal(opened.load_frame(0), image)
    completed = observe_source_fact(tmp_path, source.name, logical_identity=0)
    successor = successful_attempt(
        accounting,
        key,
        max(partial.source_revision + 1, completed.source_revision),
        mode,
    )
    intent = append_intent(
        tmp_path, extent=1, labels=(0,), generation=0,
        source_identity=partial.source_identity,
    )
    session = open_session(live_scan(target, intent, (0,)), accounting)
    session.flush(force=True)
    session.commit_epoch()
    session.finish(finalize=True)

    snapshot = accounting.snapshot()
    assert snapshot.attempts[key] == (failed, successor)
    assert snapshot.durable_attempts[(key, mode, target_name)] is successor
    assert _rows(target) == (0,)
    source_owner.close()


def test_ready_to_open_drift_is_retry_owned_and_opens_no_h23_session(tmp_path):
    from xrd_tools.sources.directory_index import StaleCandidateError
    from xrd_tools.sources.directory_session import DirectoryIndexSession

    source = tmp_path / "ready.nxs"
    _write_ready_nexus(source)
    owner = DirectoryIndexSession(probe_candidates=False)
    try:
        owner.configure(tmp_path, suffixes=(".nxs",))
        observation = owner.observe()
        candidate = next(
            value for value in observation.discovered_snapshot.candidates
            if value.path == source
        )
        fact = observe_source_fact(tmp_path, source.name, logical_identity=0)
        target = tmp_path / "drift-output.nexus"
        ledger, accounting, mode, target_name = accounting_for(target)
        key = discover(accounting, fact, group="drift", ordinal=0, label=0)
        owner_census = accounting.owner_census()

        with h5py.File(source, "a") as handle:
            handle["entry"].attrs["revision"] = 2
        with pytest.raises(StaleCandidateError):
            owner.probe_candidate(candidate, refresh=True)
        failed = retryable_attempt(accounting, key, fact.source_revision, "READY drift")
        assert accounting.owner_census() == owner_census
        assert not target.exists()
        assert ledger.snapshot().persisted == ledger.snapshot().durable == frozenset()

        current = observe_source_fact(tmp_path, source.name, logical_identity=0)
        successor = successful_attempt(
            accounting,
            key,
            max(fact.source_revision + 1, current.source_revision),
            mode,
        )
        intent = append_intent(
            tmp_path, extent=1, labels=(0,), generation=0,
            source_identity=fact.source_identity,
        )
        session = open_session(live_scan(target, intent, (0,)), accounting)
        session.flush(force=True)
        session.commit_epoch()
        session.finish(finalize=True)
    finally:
        owner.close()

    snapshot = accounting.snapshot()
    assert snapshot.attempts[key] == (failed, successor)
    assert snapshot.durable_attempts[(key, mode, target_name)] is successor


def test_interleaved_groups_advance_independent_contiguous_high_water(tmp_path):
    source = tmp_path / "interleaved.nxs"
    source.write_bytes(b"catalog")
    facts = tuple(
        observe_source_fact(tmp_path, source.name, logical_identity=value)
        for value in range(4)
    )
    target = tmp_path / "interleaved-output.nexus"
    _ledger, accounting, mode, _target_name = accounting_for(
        target, max_groups=2, max_outstanding=4,
    )
    keys = []
    for label, (group, ordinal) in enumerate(
        (("a", 0), ("b", 0), ("a", 1), ("b", 1))
    ):
        key = discover(accounting, facts[label], group=group, ordinal=ordinal, label=label)
        successful_attempt(accounting, key, facts[label].source_revision + label, mode)
        keys.append(key)
    before = accounting.snapshot()
    assert before.high_water["a"].written == 1
    assert before.high_water["b"].written == 1

    intent = append_intent(
        tmp_path, extent=4, labels=range(4), generation=0,
        source_identity=facts[0].source_identity,
    )
    session = open_session(live_scan(target, intent, range(4)), accounting)
    session.flush(force=True)
    session.commit_epoch()
    session.finish(finalize=True)
    after = accounting.snapshot()
    assert after.high_water["a"].durable == 1
    assert after.high_water["b"].durable == 1
    assert after.durable == frozenset(
        (key, mode, f"nexus:{target}") for key in keys
    )
    assert _rows(target) == (0, 1, 2, 3)


def test_failed_attempt_never_supplies_positive_stage_or_durability(tmp_path):
    source = tmp_path / "successor.h5"
    source.write_bytes(b"attempt-a")
    first = observe_source_fact(tmp_path, source.name, logical_identity=0)
    target = tmp_path / "successor-output.nexus"
    ledger, accounting, mode, target_name = accounting_for(target)
    key = discover(accounting, first, group="g", ordinal=0, label=0)
    failed = retryable_attempt(accounting, key, first.source_revision, "revision changed")
    source.write_bytes(b"attempt-b-complete")
    second = observe_source_fact(tmp_path, source.name, logical_identity=0)
    successor = successful_attempt(
        accounting,
        key,
        max(first.source_revision + 1, second.source_revision),
        mode,
    )
    intent = append_intent(
        tmp_path, extent=1, labels=(0,), generation=0,
        source_identity=first.source_identity,
    )
    session = open_session(live_scan(target, intent, (0,)), accounting)
    session.flush(force=True)
    session.commit_epoch()
    session.finish(finalize=True)

    dynamic = accounting.snapshot()
    stage = ledger.snapshot()
    assert dynamic.completed_attempts[key] is successor
    assert dynamic.written_attempts[(key, mode)] is successor
    assert dynamic.persisted_attempts[(key, mode, target_name)] is successor
    assert dynamic.durable_attempts[(key, mode, target_name)] is successor
    assert failed not in dynamic.completed_attempts.values()
    assert stage.durable == frozenset(((0, mode, target_name),))


def test_same_run_owner_refuses_duplicate_foreign_and_abort_restores_epoch(tmp_path):
    source = tmp_path / "same-run.nxs"
    source.write_bytes(b"same-run")
    first_fact = observe_source_fact(tmp_path, source.name, logical_identity=0)
    target = tmp_path / "same-run-output.nexus"
    _ledger, accounting, mode, _target_name = accounting_for(target)
    key0 = discover(accounting, first_fact, group="g", ordinal=0, label=0)
    successful_attempt(accounting, key0, first_fact.source_revision, mode)
    first = append_intent(
        tmp_path, extent=1, labels=(0,), generation=0,
        source_identity=first_fact.source_identity,
    )
    live = live_scan(target, first, (0,))
    session = open_session(live, accounting)
    session.flush(force=True)
    session.commit_epoch()
    epoch_a = target.read_bytes()
    epoch_revision = accounting.snapshot().epoch_revision
    assert accounting.owner_census().count(session) == 1

    with pytest.raises((TypeError, ValueError)):
        append_intent(
            tmp_path, extent=2, labels=(0, 0), generation=1,
            source_identity=first_fact.source_identity,
        )
    foreign = append_intent(
        tmp_path, extent=2, labels=(0, 1), generation=1,
        source_identity="foreign/source",
    )
    with pytest.raises(Exception):
        session.extend(foreign)

    second_fact = observe_source_fact(tmp_path, source.name, logical_identity=1)
    key1 = discover(accounting, second_fact, group="g", ordinal=1, label=1)
    successful_attempt(accounting, key1, second_fact.source_revision + 1, mode)
    session.extend(append_intent(
        tmp_path, extent=2, labels=(0, 1), generation=1,
        source_identity=first_fact.source_identity,
    ))
    live.frames.add(1)
    session.flush(force=True)
    session.abort()

    assert target.read_bytes() == epoch_a
    snapshot = accounting.snapshot()
    assert snapshot.epoch_revision == epoch_revision
    assert key1 not in frozenset(key for key, _mode, _target in snapshot.durable)
    assert session not in accounting.owner_census()


def test_stop_is_preterminal_and_commits_only_preaccepted_ready_prefix(tmp_path):
    source = tmp_path / "stopped.nxs"
    source.write_bytes(b"stopped")
    facts = tuple(
        observe_source_fact(tmp_path, source.name, logical_identity=value)
        for value in range(3)
    )
    target = tmp_path / "stopped-output.nexus"
    ledger, accounting, mode, target_name = accounting_for(target)
    key0 = discover(accounting, facts[0], group="g", ordinal=0, label=0)
    attempt0 = successful_attempt(accounting, key0, facts[0].source_revision, mode)
    key1 = discover(accounting, facts[1], group="g", ordinal=1, label=1)
    attempt1 = accounting.begin_attempt(key1, source_revision=facts[1].source_revision + 1)
    accounting.record_enqueued(attempt1)
    accounting.record_accepted(attempt1)
    key2 = discover(accounting, facts[2], group="g", ordinal=2, label=2)
    retry = retryable_attempt(accounting, key2, facts[2].source_revision + 2, "partial")

    first = append_intent(
        tmp_path, extent=1, labels=(0,), generation=0,
        source_identity=facts[0].source_identity,
    )
    live = live_scan(target, first, (0,))
    session = open_session(live, accounting)
    session.flush(force=True)
    session.commit_epoch()
    session.extend(append_intent(
        tmp_path, extent=2, labels=(0, 1), generation=1,
        source_identity=facts[0].source_identity,
    ))

    stopped = accounting.stop()
    assert stopped.state.value == "active"
    with pytest.raises(RuntimeError, match="frontier"):
        discover(accounting, facts[2], group="late", ordinal=0, label=99)
    accounting.record_completed(attempt1, produced=(mode,))
    accounting.record_written(attempt1, modes=(mode,))
    live.frames.add(1)
    session.flush(force=True)
    session.commit_epoch()
    session.finish(finalize=False)

    snapshot = accounting.snapshot()
    assert _rows(target) == (0, 1)
    assert snapshot.durable_attempts[(key0, mode, target_name)] is attempt0
    assert snapshot.durable_attempts[(key1, mode, target_name)] is attempt1
    assert key2 not in frozenset(key for key, _mode, _target in snapshot.durable)
    assert snapshot.attempt_states[retry].value == "failed"
    assert snapshot.in_flight == snapshot.retry_owned == frozenset()
    # The retry failed before acceptance, so the run owns no StageLedger attempt
    # that could truthfully receive a terminal disposition.
    assert 2 not in ledger.snapshot().dispositions
    with pytest.raises(RuntimeError, match="terminal"):
        accounting.record_accepted(attempt1)
    assert session not in accounting.owner_census()


def test_event_sink_exposes_worker_process_only_for_callable_inner_hook():
    from xrd_tools.session.scan_session import _EventSink

    class Plain:
        pass

    class Worker:
        def worker_process(self, frame, reduction):
            self.seen = (frame, reduction)

    plain = _EventSink(Plain(), lambda *_args: None)
    worker_inner = Worker()
    worker = _EventSink(worker_inner, lambda *_args: None)
    assert not callable(getattr(plain, "worker_process", None))
    assert callable(getattr(worker, "worker_process", None))
    worker.worker_process("frame", "reduction")
    assert worker_inner.seen == ("frame", "reduction")


def test_c2_has_exactly_the_h10_and_bounded_xye_stage_queues():
    import xrd_tools.reduction.core as core

    tree = ast.parse(inspect.getsource(core))
    queues = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "queue"
        and node.func.attr == "Queue"
    ]
    assert len(queues) == 2
    assert sorted(
        ast.unparse(node) for node in queues
    ) == ["queue.Queue()", "queue.Queue(maxsize=16)"]
    classes = {
        node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
    }

    def method(class_name, method_name):
        return next(
            node for node in classes[class_name].body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == method_name
        )

    xye_begin = method("XYESink", "begin")
    xye_queues = [node for node in ast.walk(xye_begin) if node in queues]
    assert [ast.unparse(node) for node in xye_queues] == [
        "queue.Queue(maxsize=16)",
    ]
    streaming_init = method("ReductionSession", "_init_streaming")
    streaming_queues = [node for node in ast.walk(streaming_init) if node in queues]
    assert [ast.unparse(node) for node in streaming_queues] == ["queue.Queue()"]
    assignment = next(
        node for node in ast.walk(streaming_init)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == "_write_queue"
            for target in node.targets
        )
    )
    assert ast.unparse(assignment.value) == "queue.Queue()"
    xye = classes["XYESink"]
    names = {node.id for node in ast.walk(xye) if isinstance(node, ast.Name)}
    attrs = {node.attr for node in ast.walk(xye) if isinstance(node, ast.Attribute)}
    assert {
        "_StreamPublication", "_InFlightWindow", "record_accepted",
        "record_written", "record_durable",
    }.isdisjoint(names | attrs)
    assert not hasattr(core.XYESink(Path("unused")), "output_receipt_capabilities")


def test_session_normalizer_is_the_exact_canonical_reexport():
    import importlib
    from xrd_tools.reduction.provenance_config import jsonable_run_value as canonical
    from xrd_tools.session import jsonable_run_value as compatibility

    assert compatibility is canonical
    provenance = importlib.import_module("xrd_tools.reduction.provenance_config")
    run_config = importlib.import_module("xrd_tools.session.run_configuration")
    provenance_tree = ast.parse(inspect.getsource(provenance))
    run_config_tree = ast.parse(inspect.getsource(run_config))
    definitions = [
        (module_name, node)
        for module_name, tree_value in (
            ("provenance", provenance_tree), ("session", run_config_tree)
        )
        for node in ast.walk(tree_value)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "jsonable_run_value"
    ]
    assert [(name, node.name) for name, node in definitions] == [
        ("provenance", "jsonable_run_value"),
    ]
    forbidden = ("xdart", "PyQt", "PySide", "qtpy", "pyqtgraph")
    imports = []
    for tree_value in (provenance_tree, run_config_tree):
        for node in ast.walk(tree_value):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
    assert not [name for name in imports if name.startswith(forbidden)]


def test_source_snapshot_and_mask_provenance_are_durable_and_conflict_atomic(
    tmp_path,
):
    from xrd_tools.io import record_writer as writer_api
    from xrd_tools.session import ItemDisposition, ResultMode, StageLedger

    source = tmp_path / "provenance-source.nxs"
    source.write_bytes(b"immutable-source")
    fact = observe_source_fact(tmp_path, source.name, logical_identity=0)
    snapshot = {
        "adapter_id": "nexus_hdf5",
        "size": source.stat().st_size,
        "mtime_ns": fact.source_revision,
        "frame_count": 1,
        "dataset_path": "/entry/data/data",
        "self_contained": True,
    }
    expected_snapshot = dict(snapshot)
    thumbnail = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    thumbnail_mask = np.array([[False, True], [False, False]], dtype=bool)
    expected_mask = thumbnail_mask.copy()
    target = tmp_path / "provenance-output.nexus"
    mode, target_name = ResultMode.one_d(), f"nexus:{target}"
    ledger = StageLedger(
        required_modes=(mode,), targets_by_mode={mode: (target_name,)},
    )
    ledger.record_discovered(0)
    attempt = ledger.record_accepted(0)
    ledger.record_outcome(
        0,
        ItemDisposition.COMPLETED,
        produced=(mode,),
        attempt=attempt,
    )
    ledger.record_written(0, (mode,))

    class Facade:
        def targets_for(self, selected):
            return ledger.targets_by_mode[selected]

        def capture_receipt(self, label, selected, target_value):
            return ledger.receipt(label, selected, target_value)

        def commit_durable(self, receipts):
            ledger.record_durable(receipts)

    record = writer_api.RecordWrite(
        label=0,
        result_1d=live_scan(target, None, (0,)).frames[0].int_1d,
        source_path=source,
        source_frame_index=0,
        source_snapshot=snapshot,
        thumbnail=thumbnail,
        thumbnail_mask_baked=False,
        mask_baked=False,
        thumbnail_mask=thumbnail_mask,
    )
    snapshot["size"] = -1
    thumbnail_mask[:] = False
    writer = writer_api.NexusRecordWriter(target, atomic=False, flush_every=None)
    writer.bind_session(Facade())
    writer.begin()
    writer.write(record)
    writer.flush(force=True)
    assert ledger.snapshot().durable == frozenset(((0, mode, target_name),))

    with h5py.File(target, "r") as handle:
        frame = handle["entry/frames/frame_0000"]
        source_group = frame["source"]
        assert {
            "adapter_id": source_group.attrs["adapter_id"],
            "size": int(source_group.attrs["file_size"]),
            "mtime_ns": int(source_group.attrs["file_mtime_ns"]),
            "frame_count": int(source_group.attrs["frame_count"]),
            "dataset_path": source_group.attrs["dataset_path"],
            "self_contained": bool(source_group.attrs["self_contained"]),
        } == expected_snapshot
        assert bool(frame["thumbnail"].attrs["mask_baked"]) is False
        assert bool(frame.attrs["mask_baked"]) is False
        np.testing.assert_array_equal(frame["thumbnail_mask"][()], expected_mask)
    before = target.read_bytes()
    conflicting = writer_api.RecordWrite(
        label=0,
        result_1d=live_scan(target, None, (0,)).frames[0].int_1d,
        source_path=source,
        source_snapshot={**expected_snapshot, "size": expected_snapshot["size"] + 1},
    )
    with pytest.raises((ValueError, writer_api.WriterStateError)):
        writer.write(conflicting)
    assert target.read_bytes() == before
    writer.finish()


def test_close_revalidates_compact_frame_proof_after_checkpoint_corruption(
    tmp_path,
):
    source = tmp_path / "proof-source.nxs"
    source.write_bytes(b"source")
    fact = observe_source_fact(tmp_path, source.name, logical_identity=0)
    target = tmp_path / "proof-output.nexus"
    _ledger, accounting, mode, _target = accounting_for(target)
    key = discover(accounting, fact, group="proof", ordinal=0, label=0)
    successful_attempt(accounting, key, fact.source_revision, mode)
    intent = append_intent(
        tmp_path, extent=1, labels=(0,), generation=0,
        source_identity=fact.source_identity,
    )
    session = open_session(live_scan(target, intent, (0,)), accounting)
    session.flush(force=True)

    writer = session.sink._writer
    assert writer._transaction_binding is not None
    assert writer._dirty_frames == {}
    assert writer._durable_frame_proofs
    proof = writer._durable_frame_proofs[0]
    assert writer._frame_row_digest(0)[0] == proof.digest
    frame_group = writer._h5["entry/frames/frame_0000"]
    frame_group.attrs["mask_baked"] = not bool(frame_group.attrs["mask_baked"])
    writer._h5.flush()
    assert writer._frame_row_digest(0)[0] != proof.digest
    with pytest.raises(Exception, match="frame proof changed"):
        writer._close_handle()
    session.abort()


def test_one_shot_refuses_a_second_owner_during_live_cadence(tmp_path):
    from xdart.modules.reduction import write_live_scan_to_nexus

    source = tmp_path / "cadence.nxs"
    source.write_bytes(b"cadence")
    fact = observe_source_fact(tmp_path, source.name, logical_identity=0)
    target = tmp_path / "cadence-output.nexus"
    _ledger, accounting, mode, _target_name = accounting_for(target)
    key = discover(accounting, fact, group="g", ordinal=0, label=0)
    successful_attempt(accounting, key, fact.source_revision, mode)
    intent = append_intent(
        tmp_path, extent=1, labels=(0,), generation=0,
        source_identity=fact.source_identity,
    )
    live = live_scan(target, intent, (0,))
    session = open_session(live, accounting)
    session.flush(force=True)
    session.commit_epoch()
    before = target.read_bytes()
    before_census = accounting.owner_census()
    with pytest.raises(Exception):
        write_live_scan_to_nexus(
            live,
            replace=True,
            accounting=accounting.writer_boundary,
        )
    assert target.read_bytes() == before
    assert accounting.owner_census() == before_census
    session.finish(finalize=True)


def test_real_reduction_session_refeed_uses_explicit_atomic_replace(
    tmp_path, monkeypatch,
):
    import xrd_tools.reduction.core as reduction_core
    from xrd_tools.core.containers import IntegrationResult1D
    from xrd_tools.reduction import Frame, NexusSink, ReductionPlan, Scan
    from xrd_tools.session import ResultMode, ScanSession

    monkeypatch.setattr(
        reduction_core, "integrate_1d",
        lambda image, _integrator, **_kwargs: IntegrationResult1D(
            radial=np.array([0.0, 1.0]),
            intensity=np.full(2, float(np.sum(image))),
            sigma=None, unit="q_A^-1",
        ),
    )
    target = tmp_path / "explicit-refeed.nexus"
    old = Frame(
        0, image=np.ones((2, 2)), source_path=tmp_path / "old.tif",
        source_frame_index=1,
    )
    new = Frame(
        0, image=np.full((2, 2), 3.0), source_path=tmp_path / "new.tif",
        source_frame_index=8,
    )
    mode = ResultMode.one_d()
    target_name = f"nexus:{target}"
    session = ScanSession(
        ReductionPlan(integration_2d=None),
        Scan("replace", [old], integrator=object()),
        sink=NexusSink(target, overwrite=True, atomic=False, flush_every=None),
        executor=1, targets_by_mode={mode: (target_name,)},
    )
    assert session.submit(old)
    assert session.pause(timeout=5)
    session.flush(force=True)
    stale = session.accounting.receipt(0, mode, target_name)
    session.accounting.record_durable((stale,))
    session.resume()
    assert session.submit(new)
    assert session.pause(timeout=5)
    session.flush(force=True)
    fresh = session.accounting.receipt(0, mode, target_name)
    assert fresh.revision == stale.revision + 1
    assert session.frames_submitted == session.frames_completed == 1
    session.accounting.record_durable((stale,))
    assert (0, mode, target_name) in session.accounting_snapshot().durable
    assert session.accounting.receipt(0, mode, target_name) == fresh
    session.resume()
    session.finish()
    with h5py.File(target, "r") as handle:
        frame = handle["entry/frames/frame_0000"]
        assert int(frame["source/frame_index"][()]) == 8
        assert frame["source/path"].asstr()[()] == str(tmp_path / "new.tif")


def test_public_durable_xye_capability_is_typed_and_composite_aggregated():
    from xrd_tools.io import OutputReceiptCapability
    from xrd_tools.reduction import CompositeSink, supports_durable_xye_receipts

    class Capable:
        output_receipt_capabilities = frozenset({
            OutputReceiptCapability.DURABLE_XYE,
        })

    class LegacyPrivateClaim:
        _xye_receipt_capable = True
        output_receipt_capabilities = frozenset({"durable-xye-v1"})

    capable = Capable()
    legacy = LegacyPrivateClaim()
    assert supports_durable_xye_receipts(capable)
    assert not supports_durable_xye_receipts(legacy)
    composite = CompositeSink((legacy, capable))
    assert composite.output_receipt_capabilities == frozenset({
        OutputReceiptCapability.DURABLE_XYE,
    })
    assert supports_durable_xye_receipts(composite)


def test_dynamic_bare_xye_refuses_before_any_positive_fact_or_target_mutation(
    tmp_path,
):
    from xdart.modules.reduction import (
        DynamicXyeReceiptBoundaryRequired,
        open_live_scan_session,
    )
    from xrd_tools.reduction import ReductionPlan
    from xrd_tools.session import (
        DynamicAccountingLimits,
        DynamicRunAccounting,
        ResultMode,
        StageLedger,
    )

    xye = tmp_path / "bare.xye"
    mode, target_name = ResultMode.one_d(), f"xye:{xye}"
    ledger = StageLedger(
        required_modes=(mode,), targets_by_mode={mode: (target_name,)},
    )
    accounting = DynamicRunAccounting(
        ledger,
        run_generation=1,
        limits=DynamicAccountingLimits(1, 2, 1),
    )
    live = SimpleNamespace(
        idx=0,
        map_raw=np.ones((2, 2), dtype=np.float32),
        bg_raw=None,
        scan_info={},
        source_file=str(tmp_path / "raw.tif"),
        source_frame_idx=0,
        mask=None,
        poni=None,
        integrator=object(),
    )
    before_dynamic = accounting.snapshot()
    before_stage = ledger.snapshot()
    before_census = accounting.owner_census()
    with pytest.raises(DynamicXyeReceiptBoundaryRequired):
        open_live_scan_session(
            (live,),
            ReductionPlan(integration_2d=None),
            scan_name="bare-xye",
            sink=object(),
            xye_target=target_name,
            accounting=accounting,
        )
    assert accounting.snapshot() == before_dynamic
    assert ledger.snapshot() == before_stage
    assert not xye.exists()
    assert accounting.owner_census() == before_census


def test_dynamic_nested_actual_xye_cannot_be_laundered_by_receipt_capability(
    tmp_path,
):
    from xdart.modules.reduction import (
        DynamicXyeReceiptBoundaryRequired,
        open_live_scan_session,
    )
    from xrd_tools.io import OutputReceiptCapability
    from xrd_tools.reduction import CompositeSink, ReductionPlan, XYESink
    from xrd_tools.session import (
        DynamicAccountingLimits, DynamicRunAccounting, ResultMode, StageLedger,
    )

    class UnrelatedReceiptOwner:
        output_receipt_capabilities = frozenset({
            OutputReceiptCapability.DURABLE_XYE,
        })

        def begin(self, *_args):
            raise AssertionError("sink graph mutated before XYE refusal")

    directory = tmp_path / "must-not-exist"
    nested = CompositeSink((
        CompositeSink((UnrelatedReceiptOwner(), XYESink(directory))),
    ))
    mode = ResultMode.one_d()
    ledger = StageLedger(
        required_modes=(mode,), targets_by_mode={mode: ("nexus:unused",)},
    )
    accounting = DynamicRunAccounting(
        ledger, run_generation=1,
        limits=DynamicAccountingLimits(1, 2, 1),
    )
    live = SimpleNamespace(
        idx=0, map_raw=np.ones((2, 2)), bg_raw=None, scan_info={},
        source_file=str(tmp_path / "raw.tif"), source_frame_idx=0,
        mask=None, poni=None, integrator=object(),
    )
    before_dynamic = accounting.snapshot()
    before_stage = ledger.snapshot()
    before_census = accounting.owner_census()
    with pytest.raises(DynamicXyeReceiptBoundaryRequired):
        open_live_scan_session(
            (live,), ReductionPlan(integration_2d=None), sink=nested,
            accounting=accounting, xye_receipt_boundary=UnrelatedReceiptOwner(),
        )
    assert accounting.snapshot() == before_dynamic
    assert ledger.snapshot() == before_stage
    assert accounting.owner_census() == before_census
    assert not directory.exists()


def _dynamic_sink_case(tmp_path, name):
    from xrd_tools.reduction import ReductionPlan

    target = tmp_path / f"{name}.nexus"
    _ledger, accounting, _mode, _target_name = accounting_for(target)
    live = SimpleNamespace(
        idx=0, map_raw=np.ones((2, 2)), bg_raw=None, scan_info={},
        source_file=str(tmp_path / f"{name}.tif"), source_frame_idx=0,
        mask=None, poni=None, integrator=DeterministicIntegrator(),
    )
    return target, accounting, live, ReductionPlan(integration_2d=None)


def test_d1_direct_nexus_dynamic_graph_is_bound_and_admitted(tmp_path):
    from xdart.modules.reduction import open_live_scan_session
    from xrd_tools.reduction import NexusSink

    target, accounting, live, plan = _dynamic_sink_case(tmp_path, "direct-nexus")
    session = open_live_scan_session(
        (live,), plan,
        sink=NexusSink(target, overwrite=True, atomic=False, flush_every=None),
        accounting=accounting,
        nexus_target=f"nexus:{target}",
    )
    session.finish(raise_on_failure=False)
    assert target.exists()


def test_d1_nexus_memory_dynamic_composite_is_bound_and_admitted(tmp_path):
    from xdart.modules.reduction import open_live_scan_session
    from xrd_tools.reduction import CompositeSink, MemorySink, NexusSink

    target, accounting, live, plan = _dynamic_sink_case(tmp_path, "nexus-memory")
    session = open_live_scan_session(
        (live,), plan,
        sink=CompositeSink((
            MemorySink(), NexusSink(
                target, overwrite=True, atomic=False, flush_every=None,
            ),
        )),
        accounting=accounting,
        nexus_target=f"nexus:{target}",
    )
    session.finish(raise_on_failure=False)
    assert target.exists()


def test_d1_swapping_delegation_proxy_refuses_before_begin_or_effect(tmp_path):
    from xdart.modules.reduction import (
        DynamicXyeReceiptBoundaryRequired, open_live_scan_session,
    )
    from xrd_tools.reduction import MemorySink, XYESink

    class SwappingProxy:
        def __init__(self, child):
            self.child = child
            self.begun = False

        @property
        def output_sink_children(self):
            return (self.child,)

        def begin(self, scan, plan):
            self.begun = True
            self.child = XYESink(directory)
            self.child.begin(scan, plan)
            self.child.finish(SimpleNamespace())
            raise DynamicXyeReceiptBoundaryRequired("proxy mutated during begin")

    target, accounting, live, plan = _dynamic_sink_case(tmp_path, "swapping")
    directory = tmp_path / "swapped-xye"
    proxy = SwappingProxy(MemorySink())
    before = accounting.snapshot()
    with pytest.raises(DynamicXyeReceiptBoundaryRequired):
        open_live_scan_session(
            (live,), plan, sink=proxy, accounting=accounting,
            nexus_target=f"nexus:{target}",
        )
    assert proxy.begun is False
    assert not directory.exists()
    assert accounting.snapshot() == before


def test_d1_original_composite_mutation_cannot_change_bound_execution(tmp_path):
    from xdart.modules.reduction import open_live_scan_session
    from xrd_tools.reduction import CompositeSink, MemorySink

    class LateEffectSink:
        def finish(self, _result):
            marker.write_text("executed", encoding="utf-8")

    target, accounting, live, plan = _dynamic_sink_case(tmp_path, "bound-copy")
    marker = tmp_path / "late-effect.txt"
    original = CompositeSink((MemorySink(),))
    session = open_live_scan_session(
        (live,), plan, sink=original, accounting=accounting,
        nexus_target=f"nexus:{target}",
    )
    object.__setattr__(original, "sinks", (LateEffectSink(),))
    session.finish(raise_on_failure=False)
    assert not marker.exists()


def test_dynamic_accounting_subclass_refuses_before_xye_safety(tmp_path):
    from xdart.modules.reduction import open_live_scan_session
    from xrd_tools.reduction import XYESink
    from xrd_tools.session import (
        DynamicAccountingLimits, DynamicRunAccounting, ResultMode, StageLedger,
    )

    class DerivedDynamicAccounting(DynamicRunAccounting):
        pass

    target, _accounting, live, plan = _dynamic_sink_case(tmp_path, "subclass")
    mode = ResultMode.one_d()
    ledger = StageLedger(
        required_modes=(mode,),
        targets_by_mode={mode: (f"nexus:{target}",)},
    )
    derived = DerivedDynamicAccounting(
        ledger, run_generation=1,
        limits=DynamicAccountingLimits(1, 2, 1),
    )
    directory = tmp_path / "subclass-xye"
    before = derived.snapshot()
    with pytest.raises(TypeError, match="DynamicRunAccounting"):
        open_live_scan_session(
            (live,), plan, sink=XYESink(directory), accounting=derived,
            nexus_target=f"nexus:{target}",
        )
    assert derived.snapshot() == before
    assert not directory.exists()
    assert not target.exists()


def test_dynamic_accounting_subclass_refuses_before_direct_source_materialization(
    tmp_path,
):
    from xdart.modules.reduction import open_live_scan_session
    from xrd_tools.reduction import NexusSink, ReductionPlan
    from xrd_tools.session import (
        DynamicAccountingLimits, DynamicRunAccounting, ResultMode, StageLedger,
    )

    class DerivedDynamicAccounting(DynamicRunAccounting):
        pass

    class MustNotMaterialize:
        def __iter__(self):
            raise AssertionError("dynamic subclass admission touched the source")

    target = tmp_path / "subclass-direct.nexus"
    mode, target_name = ResultMode.one_d(), f"nexus:{target}"
    ledger = StageLedger(
        required_modes=(mode,), targets_by_mode={mode: (target_name,)},
    )
    accounting = DerivedDynamicAccounting(
        ledger, run_generation=1,
        limits=DynamicAccountingLimits(1, 2, 1),
    )
    before = accounting.snapshot()
    with pytest.raises(TypeError, match="DynamicRunAccounting"):
        open_live_scan_session(
            MustNotMaterialize(), ReductionPlan(integration_2d=None),
            sink=NexusSink(
                target, overwrite=True, atomic=False, flush_every=None,
            ),
            accounting=accounting, nexus_target=target_name,
        )
    assert accounting.snapshot() == before
    assert not target.exists()


def test_dynamic_unclassified_proxy_fails_closed_before_begin(tmp_path):
    from xdart.modules.reduction import (
        DynamicXyeReceiptBoundaryRequired, open_live_scan_session,
    )

    class UnclassifiedProxy:
        def __init__(self):
            self.begun = False

        def begin(self, *_args):
            self.begun = True
            raise AssertionError("unclassified sink began")

    target, accounting, live, plan = _dynamic_sink_case(tmp_path, "proxy")
    proxy = UnclassifiedProxy()
    before = accounting.snapshot()
    with pytest.raises(DynamicXyeReceiptBoundaryRequired):
        open_live_scan_session(
            (live,), plan, sink=proxy, accounting=accounting,
            nexus_target=f"nexus:{target}",
        )
    assert proxy.begun is False
    assert accounting.snapshot() == before


def test_dynamic_delegated_proxy_classifies_its_actual_xye_child(tmp_path):
    from xdart.modules.reduction import (
        DynamicXyeReceiptBoundaryRequired, open_live_scan_session,
    )
    from xrd_tools.reduction import XYESink

    class DelegatedProxy:
        def __init__(self, child):
            self.child = child

        @property
        def output_sink_children(self):
            return (self.child,)

        def begin(self, scan, plan):
            self.child.begin(scan, plan)
            raise AssertionError("delegated XYE proxy began")

    target, accounting, live, plan = _dynamic_sink_case(tmp_path, "delegated")
    directory = tmp_path / "delegated-xye"
    proxy = DelegatedProxy(XYESink(directory))
    before = accounting.snapshot()
    with pytest.raises(DynamicXyeReceiptBoundaryRequired):
        open_live_scan_session(
            (live,), plan, sink=proxy, accounting=accounting,
            nexus_target=f"nexus:{target}",
        )
    assert accounting.snapshot() == before
    assert not directory.exists()


def test_dynamic_admission_reclassifies_mutated_composite_children(tmp_path):
    from xdart.modules.reduction import (
        DynamicXyeReceiptBoundaryRequired, open_live_scan_session,
    )
    from xrd_tools.reduction import CompositeSink, MemorySink, XYESink

    class MustNotBegin:
        output_sink_kinds = frozenset()

        def begin(self, *_args):
            raise AssertionError("mutated graph reached sink begin")

    target, accounting, live, plan = _dynamic_sink_case(tmp_path, "mutated")
    directory = tmp_path / "mutated-xye"
    sink = CompositeSink((MemorySink(),))
    object.__setattr__(sink, "sinks", (XYESink(directory), MustNotBegin()))
    before = accounting.snapshot()
    with pytest.raises(DynamicXyeReceiptBoundaryRequired):
        open_live_scan_session(
            (live,), plan, sink=sink, accounting=accounting,
            nexus_target=f"nexus:{target}",
        )
    assert accounting.snapshot() == before
    assert not directory.exists()


def test_admitted_composite_children_are_immutable_for_session_lifetime(tmp_path):
    from xdart.modules.reduction import open_live_scan_session
    from xrd_tools.reduction import CompositeSink, MemorySink, XYESink

    target, accounting, live, plan = _dynamic_sink_case(tmp_path, "frozen")
    sink = CompositeSink((MemorySink(),))
    session = open_live_scan_session(
        (live,), plan, sink=sink, accounting=accounting,
        nexus_target=f"nexus:{target}",
    )
    try:
        with pytest.raises((AttributeError, TypeError)):
            sink.sinks = (XYESink(tmp_path / "late-xye"),)
    finally:
        session.finish(raise_on_failure=False)
    assert not (tmp_path / "late-xye").exists()


def test_standalone_xye_sink_remains_available_outside_dynamic_runs(tmp_path):
    from xrd_tools.reduction import ReductionPlan, Scan, XYESink

    sink = XYESink(tmp_path / "standalone")
    sink.begin(Scan("standalone", []), ReductionPlan(integration_2d=None))
    sink.finish(SimpleNamespace())
    assert (tmp_path / "standalone").is_dir()


def test_static_composite_ignores_legacy_child_terminal_sentinel():
    from xrd_tools.reduction import CompositeSink, ReductionResult

    sentinel = object()

    class LegacySink:
        def finish(self, _result):
            return sentinel

        def abort(self, _result):
            return sentinel

    sink = CompositeSink((LegacySink(),))
    result = ReductionResult("legacy-static", {}, 0)
    assert sink.finish(result) is None
    assert sink.abort(result) is None


# ---------------------------------------------------------------------------
# C2 final-foundation: exact dynamic token -> engine -> H23 terminal bridge
# ---------------------------------------------------------------------------


def _armed_submission(accounting, *, label=0, revision=1, logical=None):
    from xrd_tools.session import DynamicFrameIdentity

    key = DynamicFrameIdentity("c2-final-source", label if logical is None else logical)
    accounting.discover(key, group="scan", ordinal=int(label), output_label=int(label))
    return key, submission_attempt(accounting, key, revision)


def _open_final_session(tmp_path, name, sink, accounting, live, plan, *, store=None,
                        nexus_target=None, executor=1):
    from xdart.modules.reduction import open_live_scan_session

    return open_live_scan_session(
        (live,), plan, sink=sink, accounting=accounting, executor=executor,
        record_store=store,
        nexus_target=nexus_target or f"nexus:{tmp_path / (name + '.nexus')}",
    )


@pytest.mark.parametrize("composite", (False, True), ids=("direct", "memory-nexus"))
def test_c2_dynamic_genuine_frame_reaches_exact_written_then_typed_commit(
    tmp_path, monkeypatch, composite,
):
    from xrd_tools.reduction import (
        CompositeSink, MemorySink, NexusSink, NexusTerminalDisposition,
    )

    target, accounting, live, plan = _dynamic_sink_case(tmp_path, f"final-{composite}")
    mode = accounting.ledger.required_modes[0]
    target_name = next(iter(accounting.ledger.targets_by_mode[mode]))
    key, token = _armed_submission(accounting)
    memory = MemorySink()
    nexus = NexusSink(target, overwrite=True, atomic=False, flush_every=None)
    sink = CompositeSink((memory, nexus)) if composite else nexus
    facades, graph_finishes, memory_finishes = [], [], []
    original_bind = NexusSink.bind_session
    original_graph_finish = CompositeSink.finish
    original_memory_finish = MemorySink.finish

    def bind_nexus(owner, facade):
        if owner is nexus:
            facades.append(facade)
        return original_bind(owner, facade)

    def bind_memory(owner, facade):
        if owner is memory:
            facades.append(facade)

    def count_graph_finish(owner, result):
        if composite and memory in owner.sinks and nexus in owner.sinks:
            graph_finishes.append(result)
        return original_graph_finish(owner, result)

    def count_memory_finish(owner, result):
        if owner is memory:
            memory_finishes.append(result)
        return original_memory_finish(owner, result)

    monkeypatch.setattr(NexusSink, "bind_session", bind_nexus)
    monkeypatch.setattr(MemorySink, "bind_session", bind_memory, raising=False)
    monkeypatch.setattr(CompositeSink, "finish", count_graph_finish)
    monkeypatch.setattr(MemorySink, "finish", count_memory_finish)
    session = _open_final_session(
        tmp_path, f"final-{composite}", sink, accounting, live, plan,
        nexus_target=target_name,
    )
    events = []
    session.on_frame_completed(events.append)
    assert session.submit(session.scan.frames[0], attempt_token=token) is True
    assert session.pause(timeout=5.0) is True
    session.flush(force=True)

    staged = accounting.snapshot()
    assert staged.completed_attempts[key] is token
    assert staged.written_attempts[(key, mode)] is token
    assert staged.pending_durable == frozenset(((key, mode, target_name),))
    assert staged.durable == frozenset()
    assert _rows(target) == (0,)
    assert [event.frame_index for event in events] == [0]
    if composite:
        assert 0 in memory.frames
        assert facades == [accounting.writer_boundary, accounting.writer_boundary]
        assert facades[0] is facades[1]
    else:
        assert facades == [accounting.writer_boundary]

    result = session.finish()
    assert result.failed is False
    assert session.terminal_result.disposition is NexusTerminalDisposition.COMMITTED
    assert session.terminal_result.commit_identity is not None
    settled = accounting.snapshot()
    assert settled.durable_attempts[(key, mode, target_name)] is token
    assert settled.state.value == "finished"
    if composite:
        assert len(graph_finishes) == len(memory_finishes) == 1
    else:
        assert graph_finishes == memory_finishes == []


def test_c2_dynamic_submit_refuses_every_invalid_capability_before_engine(tmp_path):
    from xdart.modules.reduction import open_live_scan_session
    from xrd_tools.reduction import Frame, NexusSink
    from xrd_tools.session import (
        DynamicAccountingLimits, DynamicAttemptToken, DynamicFrameIdentity,
        DynamicRunAccounting, StageLedger,
    )

    class MustNotSubmit:
        _max_workers = 1

        def submit(self, *_args, **_kwargs):
            raise AssertionError("invalid dynamic token reached executor.submit")

    target, accounting, live, plan = _dynamic_sink_case(tmp_path, "invalid-token")
    mode = accounting.ledger.required_modes[0]
    target_name = next(iter(accounting.ledger.targets_by_mode[mode]))
    keys = []
    tokens = []
    for label in range(5):
        key = DynamicFrameIdentity("invalid-source", label)
        accounting.discover(key, group="scan", ordinal=label, output_label=label)
        keys.append(key)
        tokens.append(submission_attempt(accounting, key, 1))
    stale = tokens[2]
    accounting.record_failed(stale, error="retry", retryable=True)
    submission_attempt(accounting, keys[2], 2)
    preaccepted = tokens[3]
    accounting.record_accepted(preaccepted)
    foreign_ledger = StageLedger(
        required_modes=(mode,), targets_by_mode={mode: (target_name,)},
    )
    foreign = DynamicRunAccounting(
        foreign_ledger, run_generation=accounting.snapshot().run_generation,
        limits=DynamicAccountingLimits(1, 2, 1),
    )
    foreign_key = DynamicFrameIdentity("foreign-source", 0)
    foreign.discover(foreign_key, group="foreign", ordinal=0, output_label=0)
    foreign_token = submission_attempt(foreign, foreign_key, 1)
    wrong_generation = DynamicAttemptToken(keys[1], 999, 1, 1)
    session = open_live_scan_session(
        (live,), plan,
        sink=NexusSink(target, overwrite=True, atomic=False, flush_every=None),
        accounting=accounting, executor=MustNotSubmit(), nexus_target=target_name,
    )
    base_frame = session.scan.frames[0]
    cases = (
        (base_frame, None, "missing"),
        (base_frame, foreign_token, "foreign"),
        (base_frame, wrong_generation, "generation"),
        (Frame(2, image=np.ones((2, 2))), stale, "stale"),
        (Frame(3, image=np.ones((2, 2))), preaccepted, "accepted"),
        (Frame(99, image=np.ones((2, 2))), tokens[4], "label"),
    )
    for frame, token, match in cases:
        before = accounting.snapshot()
        before_inventory = tuple(session.scan.frame_indices)
        kwargs = {} if token is None else {"attempt_token": token}
        with pytest.raises((TypeError, ValueError), match=match):
            session.submit(frame, **kwargs)
        assert accounting.snapshot() == before
        assert tuple(session.scan.frame_indices) == before_inventory
    session.stop()
    session.finish(raise_on_failure=False)

    static = open_live_scan_session(
        (live,), plan, sink=None, executor=MustNotSubmit(),
    )
    with pytest.raises(ValueError, match="static"):
        static.submit(static.scan.frames[0], attempt_token=tokens[0])
    static.stop()
    static.finish(raise_on_failure=False)


def test_c2_attempt_integer_bijection_is_exact_across_same_label_retry(tmp_path):
    target = tmp_path / "bijection.nexus"
    _ledger, accounting, _mode, _target = accounting_for(target)
    key, first = _armed_submission(accounting)
    first_published = []
    first_revision = accounting.record_accepted(
        first, publish_acceptance=first_published.append,
    )
    accounting.record_failed(first, error="retry", retryable=True)
    second = submission_attempt(accounting, key, 2)
    second_published = []
    second_revision = accounting.record_accepted(
        second, publish_acceptance=second_published.append,
    )

    assert (first_revision, second_revision) == (1, 2)
    assert first_published == [1]
    assert second_published == [2]
    assert accounting.token_for_ledger_attempt(0, 1) is first
    assert accounting.token_for_ledger_attempt(0, 2) is second
    replay = []
    assert accounting.record_accepted(
        second, publish_acceptance=replay.append,
    ) == 2
    assert replay == [2]
    assert accounting.ledger.current_attempt(0) == 2


@pytest.mark.parametrize("composite", (False, True), ids=("direct", "memory-nexus"))
def test_scan_session_commit_epoch_then_extend_live_reuses_exact_capability(
    tmp_path, monkeypatch, composite,
):
    from xdart.modules.reduction import open_live_scan_session
    from xrd_tools.io import AppendRefused, get_output_transaction_coordinator
    import xrd_tools.io.append as append_module
    from xrd_tools.io.output_transaction import OutputTransaction
    from xrd_tools.reduction import CompositeSink, Frame, MemorySink, NexusSink
    from xrd_tools.session import DynamicFrameIdentity

    target, accounting, live, plan = _dynamic_sink_case(
        tmp_path, f"same-run-public-{composite}",
    )
    mode = accounting.ledger.required_modes[0]
    target_name = next(iter(accounting.ledger.targets_by_mode[mode]))
    source_identity = "c2-public-same-run"
    first_intent = append_intent(
        tmp_path, extent=1, labels=(0,), generation=0,
        source_identity=source_identity,
    )
    key0, token0 = _armed_submission(accounting)
    memory = MemorySink()
    nexus = NexusSink(
        target, overwrite=True, atomic=False, flush_every=None,
        same_run_intent=first_intent,
    )
    sink = CompositeSink((memory, nexus)) if composite else nexus

    coordinator = get_output_transaction_coordinator()
    admissions, leases, qualifications = [], [], []
    original_admit = coordinator.admit
    original_acquire = OutputTransaction.acquire_lease
    original_qualify = append_module.qualify_append

    def count_admit(*args, **kwargs):
        admissions.append(args[0])
        return original_admit(*args, **kwargs)

    def count_acquire(owner, *args, **kwargs):
        leases.append(owner.admission.target)
        return original_acquire(owner, *args, **kwargs)

    def count_qualify(*args, **kwargs):
        qualifications.append(args[0])
        return original_qualify(*args, **kwargs)

    monkeypatch.setattr(coordinator, "admit", count_admit)
    monkeypatch.setattr(OutputTransaction, "acquire_lease", count_acquire)
    monkeypatch.setattr(append_module, "qualify_append", count_qualify)

    session = open_live_scan_session(
        (live,), plan, sink=sink, accounting=accounting, executor=1,
        nexus_target=target_name,
    )
    assert session.submit(session.scan.frames[0], attempt_token=token0)
    anchor_a = session.commit_epoch()
    epoch_a = target.read_bytes()
    assert accounting.snapshot().durable_attempts[(key0, mode, target_name)] is token0

    second_intent = append_intent(
        tmp_path, extent=2, labels=(0, 1), generation=1,
        source_identity=source_identity,
    )
    decision_b = session.extend_live(second_intent)
    assert session.extend_live(second_intent) is decision_b
    third_intent = append_intent(
        tmp_path, extent=3, labels=(0, 1, 2), generation=2,
        source_identity=source_identity,
    )
    with pytest.raises(RuntimeError, match="different successor"):
        session.extend_live(third_intent)

    key1 = DynamicFrameIdentity("c2-final-source", 1)
    accounting.discover(key1, group="scan", ordinal=1, output_label=1)
    token1 = submission_attempt(accounting, key1, 2)
    frame1 = Frame(1, image=np.full((2, 2), 2.0))
    assert session.submit(frame1, attempt_token=token1)
    with pytest.raises(RuntimeError, match="drained writer|dirty epoch"):
        session.extend_live(second_intent)
    anchor_b = session.commit_epoch()
    assert anchor_b is not anchor_a

    regressed = append_intent(
        tmp_path, extent=3, labels=(0, 1, 2), generation=1,
        source_identity=source_identity,
    )
    before_refusal = accounting.snapshot()
    before_bytes = target.read_bytes()
    with pytest.raises(AppendRefused):
        session.extend_live(regressed)
    assert accounting.snapshot() == before_refusal
    assert target.read_bytes() == before_bytes

    session.finish()
    with pytest.raises(RuntimeError, match="terminal|active"):
        session.extend_live(third_intent)
    snapshot = accounting.snapshot()
    assert _rows(target) == (0, 1)
    assert snapshot.ledger_attempts[token0] == 1
    assert snapshot.ledger_attempts[token1] == 1
    assert snapshot.durable_attempts[(key0, mode, target_name)] is token0
    assert snapshot.durable_attempts[(key1, mode, target_name)] is token1
    if composite:
        assert tuple(memory.frames) == (0, 1)
    assert admissions == [target]
    assert leases == [str(target)]
    assert qualifications == []
    assert epoch_a != target.read_bytes()


def test_scan_session_extend_live_refusals_and_stop_preserve_committed_epoch(
    tmp_path, monkeypatch,
):
    from xdart.modules.reduction import open_live_scan_session
    from xrd_tools.reduction import NexusSink

    target, accounting, live, plan = _dynamic_sink_case(tmp_path, "same-run-stop")
    mode = accounting.ledger.required_modes[0]
    target_name = next(iter(accounting.ledger.targets_by_mode[mode]))
    source_identity = "c2-public-stop"
    first_intent = append_intent(
        tmp_path, extent=1, labels=(0,), generation=0,
        source_identity=source_identity,
    )
    second_intent = append_intent(
        tmp_path, extent=2, labels=(0, 1), generation=1,
        source_identity=source_identity,
    )
    third_intent = append_intent(
        tmp_path, extent=3, labels=(0, 1, 2), generation=2,
        source_identity=source_identity,
    )
    _key0, token0 = _armed_submission(accounting)
    nexus = NexusSink(
        target, overwrite=True, atomic=False, flush_every=None,
        same_run_intent=first_intent,
    )
    calls = []
    original_extend = NexusSink.extend_live

    def count_extend(owner, capability, intent):
        if owner is nexus:
            calls.append((capability, intent))
        return original_extend(owner, capability, intent)

    monkeypatch.setattr(NexusSink, "extend_live", count_extend)
    session = open_live_scan_session(
        (live,), plan, sink=nexus, accounting=accounting, executor=1,
        nexus_target=target_name,
    )
    before = accounting.snapshot()
    with pytest.raises(RuntimeError, match="settled committed epoch"):
        session.extend_live(second_intent)
    assert calls == []
    assert accounting.snapshot() == before

    assert session.submit(session.scan.frames[0], attempt_token=token0)
    session.commit_epoch()
    epoch_a = target.read_bytes()
    assert session.pause(timeout=5.0)
    with pytest.raises(RuntimeError, match="active session"):
        session.extend_live(second_intent)
    assert calls == []
    session.resume()

    decision = session.extend_live(second_intent)
    assert session.extend_live(second_intent) is decision
    with pytest.raises(RuntimeError, match="different successor"):
        session.extend_live(third_intent)
    assert len(calls) == 2
    session.stop()
    with pytest.raises(RuntimeError, match="active session|terminal"):
        session.extend_live(second_intent)
    assert len(calls) == 2
    session.finish(raise_on_failure=False)
    assert target.read_bytes() == epoch_a
    assert _rows(target) == (0,)
    assert accounting.snapshot().state.value == "stopped"
    assert session not in accounting.owner_census()


def test_scan_session_capture_fault_aborts_graph_but_keeps_dynamic_light_reusable(
    tmp_path, monkeypatch,
):
    from xdart.modules.reduction import open_live_scan_session
    from xrd_tools.reduction import NexusSink
    from xrd_tools.session import (
        Light1DBufferLayout, Light1DCleanupHooks, Light1DLayout,
        Light1DModeLayout, SessionResourceAuthority,
        acquire_light_1d_retention,
    )

    target, accounting, live, plan = _dynamic_sink_case(tmp_path, "capture-fault")
    target_name = next(iter(accounting.ledger.targets_by_mode[
        accounting.ledger.required_modes[0]
    ]))
    intent = append_intent(
        tmp_path, extent=1, labels=(0,), generation=0,
        source_identity="c2-capture-fault",
    )
    coordinate = Light1DBufferLayout(4, 8, "capture-q", "<f8", shared=True)
    intensity = Light1DBufferLayout(4, 8, "capture-i", "<f8")
    layout = Light1DLayout(
        modes=(Light1DModeLayout("default", coordinate, intensity),),
        active_mode="default",
    )
    authority = SessionResourceAuthority(capacity_bytes=4096)
    lease = acquire_light_1d_retention(
        authority, owner="capture-fault", generation=1, layout=layout,
        requested_rows=1,
        compatibility_byte_ceiling=(layout.shared_bytes
                                    + layout.per_row_unique_ndarray_bytes),
        gui_thread_id=threading.get_ident(),
    )
    accounting.bind_light_1d(lease, cleanup_hooks=Light1DCleanupHooks())
    reserved = authority.snapshot().reserved_bytes
    _key, token = _armed_submission(accounting)
    before = accounting.snapshot()
    baseline_owners = accounting.owner_census()

    def fail_capture(_owner):
        raise RuntimeError("forced continuation capture fault")

    monkeypatch.setattr(NexusSink, "extension_owner", property(fail_capture))
    nexus = NexusSink(
        target, overwrite=True, atomic=False, flush_every=None,
        same_run_intent=intent,
    )
    with pytest.raises(RuntimeError, match="forced continuation capture fault"):
        open_live_scan_session(
            (live,), plan,
            sink=nexus,
            accounting=accounting, executor=1, nexus_target=target_name,
        )
    assert accounting.snapshot() == before
    assert accounting.owner_census() == baseline_owners
    assert authority.snapshot().reserved_bytes == reserved
    assert nexus._terminal_result.disposition.value == "aborted"
    assert not target.exists()
    monkeypatch.undo()
    fresh = NexusSink(
        target, overwrite=True, atomic=False, flush_every=None,
        same_run_intent=intent,
    )
    session = open_live_scan_session(
        (live,), plan, sink=fresh, accounting=accounting, executor=1,
        nexus_target=target_name,
    )
    assert session.submit(session.scan.frames[0], attempt_token=token)
    assert session.finish().failed is False
    assert authority.snapshot().reserved_bytes == 0


@pytest.mark.parametrize("composite", (False, True), ids=("direct", "memory-nexus"))
def test_c2_post_begin_cleanup_replay_keeps_accounting_reusable(
    tmp_path, monkeypatch, composite,
):
    from concurrent.futures import ThreadPoolExecutor as RealThreadPoolExecutor
    from xdart.modules.reduction import open_live_scan_session
    import xrd_tools.reduction.core as reduction_core
    from xrd_tools.io.output_transaction import OutputTransaction
    from xrd_tools.reduction import (
        CompositeSink, MemorySink, NexusSink, NexusTerminalDisposition,
    )
    from xrd_tools.session import (
        Light1DBufferLayout, Light1DCleanupHooks, Light1DCleanupPending,
        Light1DLayout, Light1DModeLayout, SessionResourceAuthority,
        acquire_light_1d_retention,
    )

    target, accounting, live, plan = _dynamic_sink_case(
        tmp_path, f"post-begin-{composite}",
    )
    target_name = next(iter(accounting.ledger.targets_by_mode[
        accounting.ledger.required_modes[0]
    ]))
    key, token = _armed_submission(accounting)
    coordinate = Light1DBufferLayout(4, 8, "construct-q", "<f8", shared=True)
    intensity = Light1DBufferLayout(4, 8, "construct-i", "<f8")
    layout = Light1DLayout(
        modes=(Light1DModeLayout("default", coordinate, intensity),),
        active_mode="default",
    )
    authority = SessionResourceAuthority(capacity_bytes=4096)
    lease = acquire_light_1d_retention(
        authority, owner="constructor-retry", generation=1, layout=layout,
        requested_rows=1,
        compatibility_byte_ceiling=(layout.shared_bytes
                                    + layout.per_row_unique_ndarray_bytes),
        gui_thread_id=threading.get_ident(),
    )
    light_failures = [RuntimeError("reachable light cleanup fault")]

    def fail_light_once():
        if light_failures:
            raise light_failures.pop()

    accounting.bind_light_1d(
        lease, cleanup_hooks=Light1DCleanupHooks(verify=fail_light_once),
    )
    before_accounting = accounting.snapshot()
    before_owners = accounting.owner_census()
    reserved = authority.snapshot().reserved_bytes
    nexus = NexusSink(target, overwrite=True, atomic=False, flush_every=None)
    sink = CompositeSink((MemorySink(), nexus)) if composite else nexus
    shutdowns = []

    class TrackingExecutor(RealThreadPoolExecutor):
        def shutdown(self, wait=True, *, cancel_futures=False):
            shutdowns.append((bool(wait), bool(cancel_futures)))
            return super().shutdown(wait=wait, cancel_futures=cancel_futures)

    starts = [RuntimeError("forced reduction writer start fault")]
    original_start = threading.Thread.start

    def fail_writer_start(owner):
        if owner.name.startswith("reduction-writer-") and starts:
            raise starts.pop()
        return original_start(owner)

    h23_failures = [OSError("construction H23 cleanup fault")]
    original_release = OutputTransaction.release_lease_owner

    def fail_h23_cleanup_once(owner, *args, **kwargs):
        if owner is nexus._transaction and h23_failures:
            raise h23_failures.pop()
        return original_release(owner, *args, **kwargs)

    monkeypatch.setattr(reduction_core, "ThreadPoolExecutor", TrackingExecutor)
    monkeypatch.setattr(threading.Thread, "start", fail_writer_start)
    monkeypatch.setattr(
        OutputTransaction, "release_lease_owner", fail_h23_cleanup_once,
    )
    with pytest.raises(RuntimeError, match="forced reduction writer start fault") as caught:
        open_live_scan_session(
            (live,), plan,
            sink=sink,
            accounting=accounting, executor=1, nexus_target=target_name,
        )
    replay_errors = []
    try:
        terminal = sink.abort(None)
    except OSError as error:
        replay_errors.append(error)
        terminal = sink.abort(None)
    assert isinstance(caught.value.__cause__, OSError)
    assert "construction H23 cleanup fault" in str(caught.value.__cause__)
    assert replay_errors == []
    assert shutdowns == [(True, True)]
    assert accounting.snapshot() == before_accounting
    assert accounting.owner_census() == before_owners
    assert authority.snapshot().reserved_bytes == reserved
    assert light_failures

    assert terminal.disposition is NexusTerminalDisposition.ABORTED
    fresh_nexus = NexusSink(
        target, overwrite=True, atomic=False, flush_every=None,
    )
    fresh_sink = (
        CompositeSink((MemorySink(), fresh_nexus)) if composite else fresh_nexus
    )
    finish_calls = []
    original_finish = NexusSink.finish

    def count_finish(owner, result):
        if owner is fresh_nexus:
            finish_calls.append(result)
        return original_finish(owner, result)

    monkeypatch.setattr(NexusSink, "finish", count_finish)
    session = open_live_scan_session(
        (live,), plan, sink=fresh_sink, accounting=accounting, executor=1,
        nexus_target=target_name,
    )
    assert session.submit(session.scan.frames[0], attempt_token=token)
    with pytest.raises(Light1DCleanupPending, match="reachable light cleanup fault"):
        session.finish()
    assert accounting.snapshot().state.value == "cleanup-pending"
    assert authority.snapshot().reserved_bytes == reserved
    assert len(finish_calls) == 1
    assert session.finish().failed is False
    assert accounting.snapshot().state.value == "finished"
    assert accounting.snapshot().durable_attempts[
        (key, accounting.ledger.required_modes[0], target_name)
    ] is token
    assert authority.snapshot().reserved_bytes == 0
    assert len(finish_calls) == 1


@pytest.mark.parametrize("sink_kind", ("none", "memory", "memory-composite"))
def test_dynamic_targetless_graph_finishes_written_without_false_durable(
    tmp_path, monkeypatch, sink_kind,
):
    from xdart.modules.reduction import open_live_scan_session
    from xrd_tools.reduction import CompositeSink, MemorySink, ReductionPlan
    from xrd_tools.session import (
        DynamicAccountingLimits, DynamicFrameIdentity, DynamicRunAccounting,
        ResultMode, StageLedger,
    )

    mode = ResultMode.one_d()
    ledger = StageLedger(required_modes=(mode,))
    accounting = DynamicRunAccounting(
        ledger, run_generation=1,
        limits=DynamicAccountingLimits(1, 2, 1),
    )
    key = DynamicFrameIdentity(f"targetless-{sink_kind}", 0)
    accounting.discover(key, group="scan", ordinal=0, output_label=0)
    token = submission_attempt(accounting, key, 1)
    live = SimpleNamespace(
        idx=0, map_raw=np.ones((2, 2)), bg_raw=None, scan_info={},
        source_file=str(tmp_path / f"targetless-{sink_kind}.tif"),
        source_frame_idx=0, mask=None, poni=None,
        integrator=DeterministicIntegrator(),
    )
    memory = None if sink_kind == "none" else MemorySink()
    sink = (
        None if memory is None else
        CompositeSink((memory,)) if sink_kind == "memory-composite" else memory
    )
    memory_calls, graph_calls = [], []
    original_memory_finish = MemorySink.finish
    original_composite_finish = CompositeSink.finish

    def count_memory_finish(owner, result):
        if owner is memory:
            memory_calls.append(result)
        return original_memory_finish(owner, result)

    def count_composite_finish(owner, result):
        if memory in owner.sinks:
            graph_calls.append(result)
        return original_composite_finish(owner, result)

    monkeypatch.setattr(MemorySink, "finish", count_memory_finish)
    monkeypatch.setattr(CompositeSink, "finish", count_composite_finish)
    session = open_live_scan_session(
        (live,), ReductionPlan(integration_2d=None), sink=sink,
        accounting=accounting, executor=1,
    )
    assert session.submit(session.scan.frames[0], attempt_token=token)
    result = session.finish()
    snapshot = accounting.snapshot()
    assert result.failed is False
    assert snapshot.state.value == "finished"
    assert snapshot.accepted == snapshot.completed == snapshot.written == {key}
    assert snapshot.written_attempts[(key, mode)] is token
    assert snapshot.persisted == snapshot.durable == frozenset()
    assert snapshot.publication_dropped == frozenset()
    assert session.terminal_result is None
    if memory is not None:
        assert tuple(memory.frames) == (0,)
        assert len(memory_calls) == 1
        assert len(graph_calls) == (1 if sink_kind == "memory-composite" else 0)
    else:
        assert memory_calls == graph_calls == []


def test_dynamic_targetless_completion_without_written_cannot_finish(tmp_path):
    from xrd_tools.session import (
        DynamicAccountingLimits, DynamicFrameIdentity, DynamicRunAccounting,
        ResultMode, StageLedger,
    )

    class LiveOwner:
        pass

    mode = ResultMode.one_d()
    accounting = DynamicRunAccounting(
        StageLedger(required_modes=(mode,)), run_generation=1,
        limits=DynamicAccountingLimits(1, 2, 1),
    )
    key = DynamicFrameIdentity("targetless-unwritten", 0)
    accounting.discover(key, group="scan", ordinal=0, output_label=0)
    token = submission_attempt(accounting, key, 1)
    accounting.record_accepted(token)
    accounting.record_completed(token, produced=(mode,))
    session, owner = LiveOwner(), object()
    boundary = accounting.writer_boundary
    boundary.bind_live_session(session, owner)
    with pytest.raises(RuntimeError, match="unresolved dynamic work"):
        boundary.prepare_session_finish(session, owner)
    assert (key, mode) not in accounting.snapshot().written_attempts
    boundary.epoch_aborted(session, owner, "negative targetless oracle")


def test_scan_session_extend_live_refuses_recorded_writer_failure(tmp_path, monkeypatch):
    from xdart.modules.reduction import open_live_scan_session
    from xrd_tools.reduction import Frame, NexusSink
    from xrd_tools.session import DynamicFrameIdentity

    target, accounting, live, plan = _dynamic_sink_case(tmp_path, "extend-failure")
    mode = accounting.ledger.required_modes[0]
    target_name = next(iter(accounting.ledger.targets_by_mode[mode]))
    source_identity = "c2-public-failure"
    first_intent = append_intent(
        tmp_path, extent=1, labels=(0,), generation=0,
        source_identity=source_identity,
    )
    second_intent = append_intent(
        tmp_path, extent=2, labels=(0, 1), generation=1,
        source_identity=source_identity,
    )
    _key0, token0 = _armed_submission(accounting)
    nexus = NexusSink(
        target, overwrite=True, atomic=False, flush_every=None,
        same_run_intent=first_intent,
    )
    session = open_live_scan_session(
        (live,), plan, sink=nexus, accounting=accounting, executor=1,
        nexus_target=target_name,
    )
    assert session.submit(session.scan.frames[0], attempt_token=token0)
    session.commit_epoch()
    epoch_a = target.read_bytes()
    session.extend_live(second_intent)

    key1 = DynamicFrameIdentity("c2-final-source", 1)
    accounting.discover(key1, group="scan", ordinal=1, output_label=1)
    token1 = submission_attempt(accounting, key1, 2)
    original_write = NexusSink.write

    def write_then_fail(owner, frame, reduction):
        value = original_write(owner, frame, reduction)
        if owner is nexus and int(frame.index) == 1:
            raise OSError("recorded same-run write failure")
        return value

    monkeypatch.setattr(NexusSink, "write", write_then_fail)
    assert session.submit(Frame(1, image=np.ones((2, 2))), attempt_token=token1)
    assert session.pause(timeout=5.0)
    session.resume()
    before = accounting.snapshot()
    with pytest.raises(RuntimeError, match="failed session"):
        session.extend_live(second_intent)
    assert accounting.snapshot() == before
    result = session.finish(raise_on_failure=False)
    assert result.failed is True
    assert target.read_bytes() == epoch_a
    assert _rows(target) == (0,)
    assert accounting.snapshot().state.value == "aborted"


@pytest.mark.parametrize(
    "case",
    ("automatic-flush", "target-mismatch", "multi-nexus", "unbound-same-run"),
)
def test_c2_dynamic_nexus_admission_refuses_pre_effect(tmp_path, case):
    from xdart.modules.reduction import (
        DynamicXyeReceiptBoundaryRequired, open_live_scan_session,
    )
    from xrd_tools.reduction import CompositeSink, NexusSink

    actual = tmp_path / f"{case}.nexus"
    declared = actual if case != "target-mismatch" else tmp_path / "declared.nexus"
    _ledger, accounting, _mode, target_name = accounting_for(declared)
    if case == "automatic-flush":
        sink = NexusSink(actual, overwrite=True, atomic=False, flush_every=1)
    elif case == "target-mismatch":
        sink = NexusSink(actual, overwrite=True, atomic=False, flush_every=None)
    elif case == "unbound-same-run":
        sink = NexusSink(
            actual, overwrite=True, atomic=False, flush_every=None,
            allow_unbound_same_run=True,
        )
    else:
        sink = CompositeSink((
            NexusSink(actual, overwrite=True, atomic=False, flush_every=None),
            NexusSink(tmp_path / "second.nexus", overwrite=True, atomic=False,
                      flush_every=None),
        ))

    class MustNotMaterialize:
        def __iter__(self):
            raise AssertionError("invalid sink admission touched the source")

    before = accounting.snapshot()
    with pytest.raises((ValueError, DynamicXyeReceiptBoundaryRequired)):
        open_live_scan_session(
            MustNotMaterialize(),
            __import__("xrd_tools.reduction", fromlist=["ReductionPlan"]).ReductionPlan(
                integration_2d=None,
            ),
            sink=sink, accounting=accounting, nexus_target=target_name,
        )
    assert accounting.snapshot() == before
    assert not actual.exists()
    assert not (tmp_path / "second.nexus").exists()


@pytest.mark.parametrize("composite", (False, True), ids=("direct", "memory-nexus"))
def test_c2_already_active_nexus_refuses_before_foreign_or_accounting_mutation(
    tmp_path, composite,
):
    from xdart.modules.reduction import open_live_scan_session
    from xrd_tools.reduction import (
        CompositeSink, Frame, MemorySink, NexusSink, ReductionResult, Scan,
    )

    target, accounting, live, plan = _dynamic_sink_case(tmp_path, "foreign-active")
    target_name = next(iter(accounting.ledger.targets_by_mode[
        accounting.ledger.required_modes[0]
    ]))
    _key, token = _armed_submission(accounting)
    nexus = NexusSink(target, overwrite=True, atomic=False, flush_every=None)
    foreign = Scan(
        "foreign-active", (Frame(0, image=np.ones((2, 2))),),
        integrator=DeterministicIntegrator(),
    )
    nexus.begin(foreign, plan)
    sink = CompositeSink((MemorySink(), nexus)) if composite else nexus
    before_accounting = accounting.snapshot()
    before_owners = accounting.owner_census()
    before = (
        nexus._session_facade,
        nexus._defer_publication_drop_settlement,
        nexus._writer,
        nexus._transaction,
        nexus._transaction.snapshot(),
        nexus._transaction_owners,
        target.stat().st_size,
    )
    with pytest.raises(RuntimeError, match="facade cannot change"):
        open_live_scan_session(
            (live,), plan, sink=sink, accounting=accounting, executor=1,
            nexus_target=target_name,
        )
    gc.collect()
    after_accounting = accounting.snapshot()
    after_owners = accounting.owner_census()
    after = (
        nexus._session_facade,
        nexus._defer_publication_drop_settlement,
        nexus._writer,
        nexus._transaction,
        nexus._transaction.snapshot(),
        nexus._transaction_owners,
        target.stat().st_size,
    )
    nexus.abort(ReductionResult("foreign-cleanup", {}, 0, failed=True))
    assert after_accounting == before_accounting
    assert after_owners == before_owners
    assert after == before

    fresh = NexusSink(target, overwrite=True, atomic=False, flush_every=None)
    session = open_live_scan_session(
        (live,), plan, sink=fresh, accounting=accounting, executor=1,
        nexus_target=target_name,
    )
    assert session.submit(session.scan.frames[0], attempt_token=token)
    assert session.finish().failed is False


def test_c2_all_nan_is_written_before_pending_drop_and_committed_canonical(tmp_path):
    from xrd_tools.reduction import NexusSink

    target, accounting, live, plan = _dynamic_sink_case(tmp_path, "all-nan-final")
    live.integrator = DeterministicIntegrator(all_nan=True)
    mode = accounting.ledger.required_modes[0]
    target_name = next(iter(accounting.ledger.targets_by_mode[mode]))
    key, token = _armed_submission(accounting)
    session = _open_final_session(
        tmp_path, "all-nan-final",
        NexusSink(target, overwrite=True, atomic=False, flush_every=None),
        accounting, live, plan, nexus_target=target_name,
    )
    assert session.submit(session.scan.frames[0], attempt_token=token)
    assert session.pause(timeout=5.0)
    session.flush(force=True)
    pending = accounting.snapshot()
    assert pending.written_attempts[(key, mode)] is token
    assert pending.pending_publication_dropped == frozenset(((key, mode),))
    assert pending.publication_dropped == frozenset()
    session.finish()
    final = accounting.snapshot()
    assert final.publication_dropped_attempts[(key, mode)] is token
    assert final.durable == frozenset()


@pytest.mark.parametrize("failure_owner", ("sink", "accounting"))
def test_c2_authority_and_sink_failures_never_publish_false_write(
    tmp_path, monkeypatch, failure_owner,
):
    from xrd_tools.reduction import NexusSink
    from xrd_tools.session import DynamicRunState

    target, accounting, live, plan = _dynamic_sink_case(
        tmp_path, f"failure-{failure_owner}",
    )
    mode = accounting.ledger.required_modes[0]
    target_name = next(iter(accounting.ledger.targets_by_mode[mode]))
    key, token = _armed_submission(accounting)
    nexus = NexusSink(target, overwrite=True, atomic=False, flush_every=None)
    writes = []
    original_write = NexusSink.write
    if failure_owner == "sink":
        def fail_write(owner, frame, reduction):
            if owner is nexus:
                raise OSError("sink write fault")
            return original_write(owner, frame, reduction)

        monkeypatch.setattr(NexusSink, "write", fail_write)
    else:
        def count_write(owner, frame, reduction):
            if owner is nexus:
                writes.append(int(frame.index))
            return original_write(owner, frame, reduction)

        monkeypatch.setattr(NexusSink, "write", count_write)
        owner_type = type(accounting)
        original = owner_type.record_completed

        def fail_outcome(owner, *args, **kwargs):
            if owner is accounting:
                raise RuntimeError("accounting authority fault")
            return original(owner, *args, **kwargs)

        monkeypatch.setattr(owner_type, "record_completed", fail_outcome)
    session = _open_final_session(
        tmp_path, f"failure-{failure_owner}", nexus, accounting, live, plan,
        nexus_target=target_name,
    )
    events = []
    session.on_frame_completed(events.append)
    assert session.submit(session.scan.frames[0], attempt_token=token)
    result = session.finish(raise_on_failure=False)
    snap = accounting.snapshot()
    assert result.failed is True
    assert (key, mode) not in snap.written_attempts
    assert snap.durable == frozenset()
    assert events == []
    assert snap.state is DynamicRunState.ABORTED
    if failure_owner == "sink":
        assert snap.completed_attempts[key] is token
    else:
        assert writes == []


def test_c2_raise_on_failure_false_preserves_failed_terminal_disposition(
    tmp_path, monkeypatch,
):
    from xrd_tools.reduction import NexusSink, NexusTerminalDisposition
    import xrd_tools.session.scan_session as scan_module

    target, accounting, live, plan = _dynamic_sink_case(tmp_path, "false-failure")
    mode = accounting.ledger.required_modes[0]
    target_name = next(iter(accounting.ledger.targets_by_mode[mode]))
    key, token = _armed_submission(accounting)
    nexus = NexusSink(target, overwrite=True, atomic=False, flush_every=None)

    original_post_write = scan_module._EventSink._post_write

    def fail_after_written(owner, *args):
        original_post_write(owner, *args)
        raise RuntimeError("post-written run failure")

    monkeypatch.setattr(scan_module._EventSink, "_post_write", fail_after_written)
    session = _open_final_session(
        tmp_path, "false-failure", nexus, accounting, live, plan,
        nexus_target=target_name,
    )
    assert session.submit(session.scan.frames[0], attempt_token=token)
    result = session.finish(raise_on_failure=False)
    snapshot = accounting.snapshot()
    assert result.failed is True
    assert snapshot.written_attempts[(key, mode)] is token
    assert snapshot.durable == frozenset()
    assert snapshot.state.value == "aborted"
    assert session.terminal_result.disposition is NexusTerminalDisposition.ABORTED
    assert not target.exists()


def test_c2_writer_timeout_defers_all_terminal_owners_until_writer_exit(
    tmp_path, monkeypatch,
):
    from xdart.modules.reduction import open_live_scan_session
    from xrd_tools.reduction import NexusSink
    import xrd_tools.session.scan_session as scan_module

    entered, release = threading.Event(), threading.Event()

    class BlockingIntegrator:
        detector = None

        def integrate1d(self, _image, npt, *, unit="q_A^-1", **_kwargs):
            entered.set()
            if not release.wait(5.0):
                raise RuntimeError("blocking integration was not released")
            return SimpleNamespace(
                radial=np.linspace(0.0, 1.0, int(npt)),
                intensity=np.ones(int(npt)), sigma=None, unit=unit,
            )

    target, accounting, live, plan = _dynamic_sink_case(tmp_path, "writer-timeout")
    live.integrator = BlockingIntegrator()
    target_name = next(iter(accounting.ledger.targets_by_mode[
        accounting.ledger.required_modes[0]
    ]))
    _key, token = _armed_submission(accounting)
    nexus = NexusSink(target, overwrite=True, atomic=False, flush_every=None)
    session = open_live_scan_session(
        (live,), plan, sink=nexus, accounting=accounting, executor=1,
        nexus_target=target_name,
    )
    abort_calls, sweeps, states = [], [], []
    original_abort = NexusSink.abort
    original_sweep = scan_module.ScanSession._final_sweep

    def count_abort(owner, result=None):
        if owner is nexus:
            abort_calls.append(result)
        return original_abort(owner, result)

    def count_sweep(owner):
        if owner is session:
            sweeps.append(owner)
        return original_sweep(owner)

    monkeypatch.setattr(NexusSink, "abort", count_abort)
    monkeypatch.setattr(scan_module.ScanSession, "_final_sweep", count_sweep)
    session.on_state_change(states.append)
    assert session.submit(session.scan.frames[0], attempt_token=token)
    assert entered.wait(2.0)
    try:
        with pytest.raises(TimeoutError, match="writer thread did not exit"):
            session.finish(raise_on_failure=False, join_timeout=0.01)
        assert session._session.sink_terminal_safe is False
        assert abort_calls == sweeps == states == []
        assert accounting.snapshot().state.value == "active"
    finally:
        release.set()
    for _ in range(500):
        if session._session.sink_terminal_safe:
            break
        time.sleep(0.01)
    assert session._session.sink_terminal_safe is True
    result = session.finish(raise_on_failure=False)
    assert result.failed is result.cancelled is True
    assert len(abort_calls) == len(sweeps) == len(states) == 1
    assert accounting.snapshot().state.value == "aborted"


@pytest.mark.parametrize("written_prefix", (False, True), ids=("empty", "prefix"))
def test_c2_stop_terminal_disposition_is_exact(tmp_path, written_prefix):
    from xrd_tools.reduction import NexusSink, NexusTerminalDisposition

    target, accounting, live, plan = _dynamic_sink_case(tmp_path, f"stop-{written_prefix}")
    mode = accounting.ledger.required_modes[0]
    target_name = next(iter(accounting.ledger.targets_by_mode[mode]))
    key, token = _armed_submission(accounting)
    session = _open_final_session(
        tmp_path, f"stop-{written_prefix}",
        NexusSink(target, overwrite=True, atomic=False, flush_every=None),
        accounting, live, plan, nexus_target=target_name,
    )
    if written_prefix:
        assert session.submit(session.scan.frames[0], attempt_token=token)
        assert session.pause(timeout=5.0)
    session.stop()
    result = session.finish(raise_on_failure=False)
    assert result.cancelled is True
    assert accounting.snapshot().state.value == "stopped"
    expected = (
        NexusTerminalDisposition.COMMITTED
        if written_prefix else NexusTerminalDisposition.ABORTED
    )
    assert session.terminal_result.disposition is expected
    if written_prefix:
        assert accounting.snapshot().durable_attempts[(key, mode, target_name)] is token
    else:
        assert accounting.snapshot().durable == frozenset()


def test_c2_stop_after_committed_epoch_preserves_prefix_without_reopen(
    tmp_path, monkeypatch,
):
    from xdart.modules.reduction import open_live_scan_session
    from xrd_tools.io import NexusRecordWriter
    from xrd_tools.reduction import NexusSink, NexusTerminalDisposition

    target, accounting, live, plan = _dynamic_sink_case(
        tmp_path, "stop-committed-epoch",
    )
    mode = accounting.ledger.required_modes[0]
    target_name = next(iter(accounting.ledger.targets_by_mode[mode]))
    intent = append_intent(
        tmp_path, extent=1, labels=(0,), generation=0,
        source_identity="c2-stop-committed-epoch",
    )
    key, token = _armed_submission(accounting)
    nexus = NexusSink(
        target, overwrite=True, atomic=False, flush_every=None,
        same_run_intent=intent,
    )
    begins, truncations, stopped_notifications = [], [], []
    original_begin = NexusRecordWriter.begin
    original_truncate = NexusSink.truncate_epoch
    boundary_type = type(accounting.writer_boundary)
    original_finished = boundary_type.session_finished

    def count_begin(owner, *args, **kwargs):
        if owner.target == target:
            begins.append(owner)
        return original_begin(owner, *args, **kwargs)

    def count_truncate(owner, labels):
        if owner is nexus:
            truncations.append(tuple(labels))
        return original_truncate(owner, labels)

    def count_finished(owner, *args, **kwargs):
        if owner is accounting.writer_boundary:
            stopped_notifications.append(args)
        return original_finished(owner, *args, **kwargs)

    monkeypatch.setattr(NexusRecordWriter, "begin", count_begin)
    monkeypatch.setattr(NexusSink, "truncate_epoch", count_truncate)
    monkeypatch.setattr(boundary_type, "session_finished", count_finished)
    session = open_live_scan_session(
        (live,), plan, sink=nexus, accounting=accounting, executor=1,
        nexus_target=target_name,
    )
    assert session.submit(session.scan.frames[0], attempt_token=token)
    epoch_identity = session.commit_epoch()
    committed_bytes = target.read_bytes()
    session.stop()
    try:
        result = session.finish(raise_on_failure=False)
    except BaseException:
        session.finish(raise_on_failure=False)
        raise
    assert result.cancelled is True
    assert session.terminal_result.disposition is NexusTerminalDisposition.COMMITTED
    assert session.terminal_result.commit_identity is not None
    assert target.read_bytes() == committed_bytes
    assert _rows(target) == (0,)
    assert len(begins) == 1
    assert truncations == []
    snapshot = accounting.snapshot()
    assert snapshot.state.value == "stopped"
    assert epoch_identity is not None
    assert len(stopped_notifications) == 1
    assert snapshot.durable_attempts[(key, mode, target_name)] is token


def test_c2_finish_retry_preserves_seal_and_terminal_identity(tmp_path, monkeypatch):
    from xrd_tools.io.output_transaction import OutputTransaction
    from xrd_tools.reduction import NexusSink

    target, accounting, live, plan = _dynamic_sink_case(tmp_path, "finish-retry-final")
    target_name = next(iter(accounting.ledger.targets_by_mode[
        accounting.ledger.required_modes[0]
    ]))
    _key, token = _armed_submission(accounting)
    session = _open_final_session(
        tmp_path, "finish-retry-final",
        NexusSink(target, overwrite=True, atomic=False, flush_every=None),
        accounting, live, plan, nexus_target=target_name,
    )
    assert session.submit(session.scan.frames[0], attempt_token=token)
    original = OutputTransaction.release_lease_owner
    failures = [OSError("terminal lease cleanup fault")]

    def fail_once(owner, *args, **kwargs):
        if failures:
            raise failures.pop()
        return original(owner, *args, **kwargs)

    monkeypatch.setattr(OutputTransaction, "release_lease_owner", fail_once)
    with pytest.raises(OSError, match="terminal lease cleanup fault"):
        session.finish()
    assert accounting.snapshot().epoch_revision == 0
    result = session.finish()
    identity = session.terminal_result.commit_identity
    assert result.failed is False
    assert identity is not None
    assert session.finish().failed is False
    assert session.terminal_result.commit_identity is identity
    assert accounting.snapshot().epoch_revision == 1


def test_c2_abort_retry_notifies_only_after_exact_aborted(tmp_path, monkeypatch):
    from xrd_tools.io.output_transaction import OutputTransaction
    from xrd_tools.reduction import NexusSink

    target, accounting, live, plan = _dynamic_sink_case(tmp_path, "abort-retry-final")
    target_name = next(iter(accounting.ledger.targets_by_mode[
        accounting.ledger.required_modes[0]
    ]))
    _key, token = _armed_submission(accounting)
    nexus = NexusSink(target, overwrite=True, atomic=False, flush_every=None)
    original_write = NexusSink.write

    def fail_write(owner, frame, reduction):
        if owner is nexus:
            raise OSError("writer failure")
        return original_write(owner, frame, reduction)

    monkeypatch.setattr(NexusSink, "write", fail_write)
    session = _open_final_session(
        tmp_path, "abort-retry-final", nexus, accounting, live, plan,
        nexus_target=target_name,
    )
    assert session.submit(session.scan.frames[0], attempt_token=token)
    original_release = OutputTransaction.release_lease_owner
    failures = [OSError("abort cleanup fault")]

    def fail_release(owner, *args, **kwargs):
        if failures:
            raise failures.pop()
        return original_release(owner, *args, **kwargs)

    monkeypatch.setattr(OutputTransaction, "release_lease_owner", fail_release)
    with pytest.raises(OSError, match="abort cleanup fault"):
        session.finish(raise_on_failure=False)
    assert accounting.snapshot().state.value == "active"
    result = session.finish(raise_on_failure=False)
    assert result.failed is True
    assert accounting.snapshot().state.value == "aborted"


def test_c2_post_commit_accounting_retry_does_not_rerun_h23(tmp_path, monkeypatch):
    from xrd_tools.reduction import NexusSink

    target, accounting, live, plan = _dynamic_sink_case(tmp_path, "light-retry-final")
    target_name = next(iter(accounting.ledger.targets_by_mode[
        accounting.ledger.required_modes[0]
    ]))
    _key, token = _armed_submission(accounting)
    nexus = NexusSink(target, overwrite=True, atomic=False, flush_every=None)
    session = _open_final_session(
        tmp_path, "light-retry-final", nexus, accounting, live, plan,
        nexus_target=target_name,
    )
    assert session.submit(session.scan.frames[0], attempt_token=token)
    finish_calls = []
    original_finish = NexusSink.finish

    def count_finish(owner, result):
        if owner is nexus:
            finish_calls.append(result)
        return original_finish(owner, result)

    monkeypatch.setattr(NexusSink, "finish", count_finish)
    boundary_type = type(accounting.writer_boundary)
    original_notify = boundary_type.session_finished
    failures = [RuntimeError("light cleanup fault")]

    def fail_once(owner, *args, **kwargs):
        if owner is accounting.writer_boundary and failures:
            raise failures.pop()
        return original_notify(owner, *args, **kwargs)

    monkeypatch.setattr(boundary_type, "session_finished", fail_once)
    with pytest.raises(RuntimeError, match="light cleanup fault"):
        session.finish()
    committed = session.terminal_result
    assert committed.commit_identity is not None
    assert len(finish_calls) == 1
    session.finish()
    assert session.terminal_result is committed
    assert len(finish_calls) == 1
    assert accounting.snapshot().state.value == "finished"


def test_c2_final_sweep_and_terminal_event_wait_for_settlement(
    tmp_path, monkeypatch,
):
    from xrd_tools.reduction import NexusSink
    from xrd_tools.session import FrameRecordStore

    target, accounting, live, plan = _dynamic_sink_case(tmp_path, "sweep-final")
    target_name = next(iter(accounting.ledger.targets_by_mode[
        accounting.ledger.required_modes[0]
    ]))
    _key, token = _armed_submission(accounting)
    store = FrameRecordStore()
    nexus = NexusSink(target, overwrite=True, atomic=False, flush_every=None)
    session = _open_final_session(
        tmp_path, "sweep-final", nexus, accounting, live, plan,
        store=store, nexus_target=target_name,
    )
    assert session.submit(session.scan.frames[0], attempt_token=token)
    states = []
    session.on_state_change(states.append)
    released = []
    original_release = FrameRecordStore.release_heavy

    def record_release(owner, label):
        if owner is store:
            released.append(int(label))
        return original_release(owner, label)

    monkeypatch.setattr(FrameRecordStore, "release_heavy", record_release)
    boundary_type = type(accounting.writer_boundary)
    original_notify = boundary_type.session_finished
    failures = [RuntimeError("terminal notification fault")]

    def fail_once(owner, *args, **kwargs):
        if owner is accounting.writer_boundary and failures:
            raise failures.pop()
        return original_notify(owner, *args, **kwargs)

    monkeypatch.setattr(boundary_type, "session_finished", fail_once)
    with pytest.raises(RuntimeError, match="terminal notification fault"):
        session.finish()
    assert released == []
    assert states == []
    session.finish()
    assert released == [0]
    assert len(states) == 1
    session.finish()
    assert released == [0]
    assert len(states) == 1


def test_publication_store_borrows_the_only_light_1d_arrays_and_obeys_grant():
    from dataclasses import fields, is_dataclass
    from collections.abc import Mapping
    from xdart.modules.frame_publication import PublicationStore
    from xrd_tools.session import (
        Light1DBufferLayout,
        Light1DCleanupHooks,
        Light1DLayout,
        Light1DModeData,
        Light1DModeLayout,
        Light1DRecord,
        Light1DStaleGeneration,
        SessionResourceAuthority,
        SessionResourceRequirements,
        acquire_light_1d_retention,
        resolve_session_policy,
    )

    def independent_publication_arrays(store, lease):
        """Test-side object-graph census independent of the production census."""
        arrays = {}
        seen = set()

        def visit(value):
            identity = id(value)
            if identity in seen:
                return
            seen.add(identity)
            if isinstance(value, np.ndarray):
                root = value
                while isinstance(root.base, np.ndarray):
                    root = root.base
                arrays[id(root)] = int(root.nbytes)
                return
            if isinstance(value, Mapping):
                for key, item in value.items():
                    visit(key)
                    visit(item)
                return
            if isinstance(value, (tuple, list, set, frozenset)):
                for item in value:
                    visit(item)
                return
            if is_dataclass(value) and not isinstance(value, type):
                for descriptor in fields(value):
                    visit(getattr(value, descriptor.name))

        for name, value in store.__dict__.items():
            if name not in {"_lock", "_light_1d"} and not callable(value):
                visit(value)
        for root_id in lease.owned_buffer_ids:
            arrays.pop(root_id, None)
        return arrays

    requirements = SessionResourceRequirements(
        height=4,
        width=4,
        native_itemsize=2,
        modes_1d=2,
        npt_1d=4,
        sigma_1d=1,
        modes_2d=0,
    )
    allocation = resolve_session_policy(
        requirements,
        envelope_bytes=4 * 1024 ** 3,
        requests={"publication_items": 1, "record_items": 1},
        env={},
    ).allocation
    axis_layout = Light1DBufferLayout(4, 8, "q", "<f8", shared=True)
    layout = Light1DLayout(
        modes=(
            Light1DModeLayout(
                "raw",
                axis_layout,
                Light1DBufferLayout(4, 8, "raw-i", "<f8"),
                Light1DBufferLayout(4, 8, "raw-s", "<f8"),
            ),
            Light1DModeLayout(
                "bg",
                axis_layout,
                Light1DBufferLayout(4, 4, "bg-i", "<f4"),
            ),
        ),
        active_mode="bg",
    )
    authority = SessionResourceAuthority.from_allocation(allocation)
    lease = acquire_light_1d_retention(
        authority,
        owner="c2-publication",
        generation=7,
        layout=layout,
        requested_rows=2,
        compatibility_byte_ceiling=(
            layout.shared_bytes + layout.per_row_unique_ndarray_bytes
        ),
        gui_thread_id=threading.get_ident(),
    )
    assert lease.row_cap == 1
    store = PublicationStore()
    store.bind_allocation(allocation)
    store.bind_light_1d(lease)
    with pytest.raises(ValueError, match="allocation-bound"):
        store.set_max_heavy_items(99)

    axis = np.linspace(0.0, 1.0, 4, dtype=np.float64)

    def record(label, generation=7):
        return Light1DRecord(
            row_identity=label,
            generation=generation,
            active_mode="bg",
            modes={
                "raw": Light1DModeData(
                    axis,
                    np.full(4, label + 1, dtype=np.float64),
                    np.full(4, label / 10, dtype=np.float64),
                ),
                "bg": Light1DModeData(
                    axis,
                    np.full(4, label + 2, dtype=np.float32),
                ),
            },
            provenance={"source": "scan.nxs", "logical": label},
        )

    first = store.publish_light_1d(record(0), source_identity="scan.nxs#0")
    canonical = lease.borrow(0)
    try:
        assert first.borrow.modes["raw"].coordinate is canonical.modes["raw"].coordinate
        assert first.borrow.modes["raw"].intensity is canonical.modes["raw"].intensity
        assert first.borrow.modes["bg"].intensity is canonical.modes["bg"].intensity
    finally:
        canonical.close()
    census = store.ndarray_owner_census()
    assert census["lease"] == lease.owned_buffer_ids
    assert census["publication"] == frozenset()
    assert independent_publication_arrays(store, lease) == {}
    assert lease.unique_owned_ndarray_bytes == lease.reserved_ndarray_bytes
    assert authority.snapshot().categories["light_1d"] == lease.reserved_ndarray_bytes

    hidden_copy = np.arange(4, dtype=np.float64)
    store._hidden_publication_copy = hidden_copy
    try:
        mutated = store.ndarray_owner_census()
        assert mutated["publication"] == frozenset((id(hidden_copy),))
        assert mutated["publication"].isdisjoint(mutated["lease"])
        assert independent_publication_arrays(store, lease) == {
            id(hidden_copy): hidden_copy.nbytes,
        }
    finally:
        del store._hidden_publication_copy

    second = store.publish_light_1d(record(1), source_identity="scan.nxs#1")
    assert store.labels() == (1,)
    assert lease.keys() == (1,)
    assert first.borrow.closed is True
    with pytest.raises(Light1DStaleGeneration):
        store.publish_light_1d(record(2, generation=6), source_identity="stale")
    assert store.labels() == (1,)

    del first, second
    store.clear()
    gc.collect()
    lease.release(reason="terminal", hooks=Light1DCleanupHooks())
    assert authority.snapshot().reserved_bytes == allocation.assigned_bytes
    assert authority.snapshot().categories.get("light_1d", 0) == 0
    assert store.ndarray_owner_census() == {
        "lease": frozenset(),
        "publication": frozenset(),
    }
    assert independent_publication_arrays(store, lease) == {}


def _c2_retry_graph(memory_count, nexus):
    """Return an admitted exact graph with Memory before the Nexus branch."""
    from xrd_tools.reduction import CompositeSink, MemorySink

    memories = [MemorySink() for _ in range(memory_count)]
    if memory_count == 1:
        return CompositeSink((memories[0], nexus)), memories
    inner = CompositeSink((memories[1], nexus))
    return CompositeSink((memories[0], inner)), memories


@pytest.mark.parametrize(
    "case",
    (
        "flat-finish",
        "flat-abort",
        "flat-constructor-replay",
        "nested-finish",
        "nested-abort",
    ),
)
def test_c2_exact_composite_retry_settles_nexus_before_memory(
    tmp_path, monkeypatch, case,
):
    from xdart.modules.reduction import open_live_scan_session
    from xrd_tools.io.output_transaction import OutputTransaction
    from xrd_tools.reduction import (
        CompositeSink, MemorySink, NexusSink, NexusTerminalDisposition,
        NexusTerminalResult, ReductionResult,
    )

    target, accounting, live, plan = _dynamic_sink_case(tmp_path, case)
    target_name = next(iter(accounting.ledger.targets_by_mode[
        accounting.ledger.required_modes[0]
    ]))
    _key, token = _armed_submission(accounting)
    nexus = NexusSink(target, overwrite=True, atomic=False, flush_every=None)
    graph, memories = _c2_retry_graph(
        2 if case.startswith("nested") else 1, nexus,
    )
    memory_calls = {id(memory): [] for memory in memories}
    nexus_calls = []
    original_memory_finish = MemorySink.finish
    original_nexus_finish = NexusSink.finish
    original_nexus_abort = NexusSink.abort

    def count_memory(owner, result):
        if id(owner) in memory_calls:
            memory_calls[id(owner)].append(result)
        return original_memory_finish(owner, result)

    def count_nexus_finish(owner, result):
        if owner is nexus:
            nexus_calls.append(result)
        return original_nexus_finish(owner, result)

    def count_nexus_abort(owner, result):
        if owner is nexus:
            nexus_calls.append(result)
        return original_nexus_abort(owner, result)

    monkeypatch.setattr(MemorySink, "finish", count_memory)
    monkeypatch.setattr(NexusSink, "finish", count_nexus_finish)
    monkeypatch.setattr(NexusSink, "abort", count_nexus_abort)
    original_release = OutputTransaction.release_lease_owner
    failures = [OSError("forced H23 terminal cleanup failure")]

    def fail_release_once(owner, *args, **kwargs):
        if owner is nexus._transaction and failures:
            raise failures.pop()
        return original_release(owner, *args, **kwargs)

    monkeypatch.setattr(OutputTransaction, "release_lease_owner", fail_release_once)

    if case == "flat-constructor-replay":
        original_start = threading.Thread.start
        start_failures = [RuntimeError("forced reduction writer start failure")]

        def fail_writer_start(owner):
            if owner.name.startswith("reduction-writer-") and start_failures:
                raise start_failures.pop()
            return original_start(owner)

        monkeypatch.setattr(threading.Thread, "start", fail_writer_start)
        with pytest.raises(
            RuntimeError, match="forced reduction writer start failure",
        ) as caught:
            open_live_scan_session(
                (live,), plan, sink=graph, accounting=accounting, executor=1,
                nexus_target=target_name,
            )
        assert isinstance(caught.value.__cause__, OSError)
        assert len(nexus_calls) == 1
        assert type(nexus_calls[0]) is ReductionResult
        assert nexus_calls[0].failed is True
        assert all(calls == [] for calls in memory_calls.values())
        terminal = graph.abort(None)
        assert nexus_calls == [nexus_calls[0], None]
        assert type(terminal) is NexusTerminalResult
        assert terminal.disposition is NexusTerminalDisposition.ABORTED
    else:
        session = open_live_scan_session(
            (live,), plan, sink=graph, accounting=accounting, executor=1,
            nexus_target=target_name,
        )
        if case.endswith("abort"):
            original_write = NexusSink.write

            def fail_write(owner, frame, reduction):
                if owner is nexus:
                    raise OSError("forced Nexus write failure")
                return original_write(owner, frame, reduction)

            monkeypatch.setattr(NexusSink, "write", fail_write)
        assert session.submit(session.scan.frames[0], attempt_token=token)
        with pytest.raises(OSError, match="forced H23 terminal cleanup failure"):
            session.finish(raise_on_failure=not case.endswith("abort"))
        assert len(nexus_calls) == 1
        assert all(calls == [] for calls in memory_calls.values())
        result = session.finish(raise_on_failure=False)
        terminal = session.terminal_result
        assert result.failed is case.endswith("abort")
        assert type(terminal) is NexusTerminalResult
        assert terminal.disposition is (
            NexusTerminalDisposition.ABORTED
            if case.endswith("abort") else NexusTerminalDisposition.COMMITTED
        )
        assert len(nexus_calls) == 2

    assert all(len(calls) == 1 for calls in memory_calls.values())


def _c2_bind_light_owner(accounting, name):
    from xrd_tools.session import (
        Light1DBufferLayout, Light1DCleanupHooks, Light1DLayout,
        Light1DModeLayout, SessionResourceAuthority,
        acquire_light_1d_retention,
    )

    coordinate = Light1DBufferLayout(4, 8, f"{name}-q", "<f8", shared=True)
    intensity = Light1DBufferLayout(4, 8, f"{name}-i", "<f8")
    layout = Light1DLayout((Light1DModeLayout(
        "default", coordinate, intensity,
    ),), "default")
    authority = SessionResourceAuthority(capacity_bytes=4096)
    lease = acquire_light_1d_retention(
        authority, owner=name, generation=1, layout=layout, requested_rows=1,
        compatibility_byte_ceiling=(
            layout.shared_bytes + layout.per_row_unique_ndarray_bytes
        ),
        gui_thread_id=threading.get_ident(),
    )
    accounting.bind_light_1d(lease, cleanup_hooks=Light1DCleanupHooks())
    return authority


@pytest.mark.parametrize(
    "failure_kind,composite",
    (
        ("writer-start", False),
        ("writer-start", True),
        ("final-bind", False),
        ("final-bind", True),
    ),
    ids=("writer-direct", "writer-composite", "bind-direct", "bind-composite"),
)
def test_c2_failed_constructor_restores_borrowed_facade_before_replay(
    tmp_path, monkeypatch, failure_kind, composite,
):
    from xdart.modules.reduction import open_live_scan_session
    from xrd_tools.io.output_transaction import OutputTransaction
    from xrd_tools.reduction import (
        CompositeSink, MemorySink, NexusSink, NexusTerminalDisposition,
    )

    name = f"facade-{failure_kind}-{composite}"
    target, accounting, live, plan = _dynamic_sink_case(tmp_path, name)
    target_name = next(iter(accounting.ledger.targets_by_mode[
        accounting.ledger.required_modes[0]
    ]))
    authority = _c2_bind_light_owner(accounting, name)
    reserved = authority.snapshot().reserved_bytes
    existing = None
    if failure_kind == "final-bind":
        existing = open_live_scan_session(
            (live,), plan, sink=MemorySink(), accounting=accounting, executor=1,
            nexus_target=target_name,
        )
        token = None
    else:
        _key, token = _armed_submission(accounting)
    before = accounting.snapshot()
    owners_before = accounting.owner_census()
    nexus = NexusSink(target, overwrite=True, atomic=False, flush_every=None)
    prior_facade = object()
    prior_defer = True
    nexus.bind_session(prior_facade)
    nexus._defer_publication_drop_settlement = prior_defer
    graph = CompositeSink((MemorySink(), nexus)) if composite else nexus
    original_release = OutputTransaction.release_lease_owner
    cleanup_failures = [OSError("forced construction H23 cleanup failure")]

    def fail_cleanup_once(owner, *args, **kwargs):
        if owner is nexus._transaction and cleanup_failures:
            raise cleanup_failures.pop()
        return original_release(owner, *args, **kwargs)

    monkeypatch.setattr(OutputTransaction, "release_lease_owner", fail_cleanup_once)
    if failure_kind == "writer-start":
        original_start = threading.Thread.start
        start_failures = [RuntimeError("forced reduction writer start failure")]

        def fail_writer_start(owner):
            if owner.name.startswith("reduction-writer-") and start_failures:
                raise start_failures.pop()
            return original_start(owner)

        monkeypatch.setattr(threading.Thread, "start", fail_writer_start)
        expected = "forced reduction writer start failure"
    else:
        expected = "dynamic accounting already has a bound live session"

    with pytest.raises(RuntimeError, match=expected) as caught:
        open_live_scan_session(
            (live,), plan, sink=graph, accounting=accounting, executor=1,
            nexus_target=target_name,
        )
    assert isinstance(caught.value.__cause__, OSError)
    assert nexus._session_facade is prior_facade
    assert nexus._defer_publication_drop_settlement is prior_defer
    assert accounting.snapshot() == before
    assert accounting.owner_census() == owners_before
    assert authority.snapshot().reserved_bytes == reserved
    terminal = graph.abort(None)
    assert terminal.disposition is NexusTerminalDisposition.ABORTED
    assert nexus._session_facade is prior_facade
    assert nexus._defer_publication_drop_settlement is prior_defer

    if existing is None:
        fresh = NexusSink(target, overwrite=True, atomic=False, flush_every=None)
        session = open_live_scan_session(
            (live,), plan, sink=fresh, accounting=accounting, executor=1,
            nexus_target=target_name,
        )
        assert session.submit(session.scan.frames[0], attempt_token=token)
        assert session.finish().failed is False
    else:
        assert accounting.owner_census().count(existing) == 1
        existing.stop()
        assert existing.finish(raise_on_failure=False).cancelled is True
    assert authority.snapshot().reserved_bytes == 0


@pytest.mark.parametrize("nested", (False, True), ids=("flat", "nested"))
def test_c2_exact_composite_begin_runs_nexus_branch_before_memory(
    tmp_path, monkeypatch, nested,
):
    from xdart.modules.reduction import open_live_scan_session
    from xrd_tools.io import NexusRecordWriter
    from xrd_tools.reduction import (
        CompositeSink, MemorySink, NexusSink, NexusTerminalDisposition,
    )

    target, accounting, live, plan = _dynamic_sink_case(
        tmp_path, f"begin-order-{nested}",
    )
    target_name = next(iter(accounting.ledger.targets_by_mode[
        accounting.ledger.required_modes[0]
    ]))
    nexus = NexusSink(target, overwrite=True, atomic=False, flush_every=None)
    graph, memories = _c2_retry_graph(2 if nested else 1, nexus)
    sentinels = {id(memory): object() for memory in memories}
    for memory in memories:
        memory.frames[99] = sentinels[id(memory)]
    begin_calls = {id(memory): 0 for memory in memories}
    finish_calls = {id(memory): 0 for memory in memories}
    original_memory_begin = MemorySink.begin
    original_memory_finish = MemorySink.finish
    original_writer_begin = NexusRecordWriter.begin

    def count_memory_begin(owner, scan, reduction_plan):
        if id(owner) in begin_calls:
            begin_calls[id(owner)] += 1
        return original_memory_begin(owner, scan, reduction_plan)

    def count_memory_finish(owner, result):
        if id(owner) in finish_calls:
            finish_calls[id(owner)] += 1
        return original_memory_finish(owner, result)

    def fail_nexus_begin(owner, *args, **kwargs):
        if owner.target == target:
            raise RuntimeError("forced Nexus writer begin failure")
        return original_writer_begin(owner, *args, **kwargs)

    monkeypatch.setattr(MemorySink, "begin", count_memory_begin)
    monkeypatch.setattr(MemorySink, "finish", count_memory_finish)
    monkeypatch.setattr(NexusRecordWriter, "begin", fail_nexus_begin)
    with pytest.raises(RuntimeError, match="forced Nexus writer begin failure"):
        open_live_scan_session(
            (live,), plan, sink=graph, accounting=accounting, executor=1,
            nexus_target=target_name,
        )
    assert all(count == 0 for count in begin_calls.values())
    assert all(count == 0 for count in finish_calls.values())
    assert all(
        memory.frames == {99: sentinels[id(memory)]} for memory in memories
    )
    assert nexus._terminal_result.disposition is NexusTerminalDisposition.ABORTED
    assert nexus._writer is None or nexus._writer.phase.value not in {"active", "partial"}
    assert nexus._transaction_owners is None
