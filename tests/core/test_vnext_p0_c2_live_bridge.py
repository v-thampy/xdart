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

import h5py
import numpy as np
import pytest

from tests.core._vnext_p0_c2_bridge_support import (
    accounting_for,
    append_intent,
    discover,
    live_scan,
    observe_source_fact,
    open_session,
    retryable_attempt,
    successful_attempt,
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
        mask=None, poni=None, integrator=object(),
    )
    return target, accounting, live, ReductionPlan(integration_2d=None)


def test_dynamic_accounting_subclass_cannot_skip_xye_safety(tmp_path):
    from xdart.modules.reduction import (
        DynamicXyeReceiptBoundaryRequired, open_live_scan_session,
    )
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
    with pytest.raises(DynamicXyeReceiptBoundaryRequired):
        open_live_scan_session(
            (live,), plan, sink=XYESink(directory), accounting=derived,
            nexus_target=f"nexus:{target}",
        )
    assert derived.snapshot() == before
    assert not directory.exists()
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
