"""C2 public-seam composition discriminators.

The rows intentionally drive E6 discovery values into the accepted H10
dynamic owner and then into the one high-level H23 LiveScan session.  They do
not import or inspect writer, transaction, lease, or target internals.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import gc
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
    ("filename", "initial"),
    (
        ("growing.nxs", b"nexus-shell"),
        ("growing.h5", b"hdf-shell"),
        ("armed_master.h5", b"missing-external-link-member"),
    ),
)
def test_growing_container_retries_same_key_and_extends_one_live_owner(
    tmp_path,
    filename,
    initial,
):
    source = tmp_path / filename
    source.write_bytes(initial)
    first_fact = observe_source_fact(tmp_path, filename, logical_identity=0)
    target = tmp_path / f"{Path(filename).stem}-processed.nexus"
    _ledger, accounting, mode, target_name = accounting_for(target)
    key0 = discover(accounting, first_fact, group="detector", ordinal=0, label=0)
    failed = retryable_attempt(
        accounting, key0, first_fact.source_revision, "transient source revision",
    )

    source.write_bytes(initial + b"-stable")
    retry_fact = observe_source_fact(tmp_path, filename, logical_identity=0)
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
    assert snapshot.attempts[key0] == (failed, successor)
    assert snapshot.attempts[key1] == (second,)
    assert snapshot.durable_attempts[(key0, mode, target_name)] is successor
    assert snapshot.durable_attempts[(key1, mode, target_name)] is second
    assert _rows(target) == (0, 1)
    assert session not in accounting.owner_census()


def test_partial_tiff_retry_has_no_receipt_then_same_key_succeeds(tmp_path):
    source = tmp_path / "frame_0001.tif"
    source.write_bytes(b"II\x2a\x00partial")
    partial = observe_source_fact(tmp_path, source.name, logical_identity=0)
    target = tmp_path / "partial-tiff.nexus"
    ledger, accounting, mode, target_name = accounting_for(target)
    key = discover(accounting, partial, group="tiff", ordinal=0, label=0)
    failed = retryable_attempt(accounting, key, partial.source_revision, "partial TIFF")
    assert ledger.snapshot().persisted == ledger.snapshot().durable == frozenset()

    source.write_bytes(b"II\x2a\x00complete-image-payload")
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
    assert stopped.state.value == "stopped"
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


def test_publication_store_borrows_the_only_light_1d_arrays_and_obeys_grant():
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
    assert lease.unique_owned_ndarray_bytes == lease.reserved_ndarray_bytes
    assert authority.snapshot().categories["light_1d"] == lease.reserved_ndarray_bytes

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
    assert store.ndarray_owner_census() == {
        "lease": frozenset(),
        "publication": frozenset(),
    }
