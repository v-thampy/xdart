from __future__ import annotations

import hashlib
import importlib
import copy
import json
import os
from pathlib import Path
import threading

import h5py
import numpy as np
import pytest

from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from xrd_tools.io.nexus import write_integrated_stack
from tests.core.v2_fixture_factory import current_entry
from xrd_tools.io.output_transaction import (
    CleanupIncomplete,
    LeaseOwner,
    LeaseUnavailable,
    OutputTransactionCoordinator,
    OwnerToken,
    RetryAction,
    TargetChanged,
    TransactionPhase,
    get_output_transaction_coordinator,
)
from xrd_tools.session import (
    ItemDisposition,
    ResultMode,
    StageLedger,
    StageReceipt,
)
from xrd_tools.io.schema import (
    MULTI_RESULT_MODES_ATTR,
    PROCESSED_SCHEMA_NAME,
    PROCESSED_SCHEMA_VERSION,
    PRIMARY_MODE_ATTR,
    SCHEMA_NAME_ATTR,
    SCHEMA_VERSION_ATTR,
    SOURCE_BASE_ATTR,
    read_current_mode_layout,
)


class _Pool:
    def __init__(self, *, fail_resume: int = 0) -> None:
        self.events: list[tuple[str, str]] = []
        self.fail_resume = int(fail_resume)

    def pause(self, target) -> None:
        self.events.append(("pause", os.fspath(target)))

    def resume(self, target) -> None:
        self.events.append(("resume", os.fspath(target)))
        if self.fail_resume:
            self.fail_resume -= 1
            raise OSError("resume blocked")


class _Facade:
    def __init__(self, target: Path) -> None:
        self.target = f"nexus:{target}"
        self.durable: list[StageReceipt] = []

    def targets_for(self, _mode):
        return frozenset((self.target,))

    def capture_receipt(self, label, mode, target):
        return StageReceipt(int(label), mode, 1, target)

    def commit_durable(self, receipts):
        self.durable.extend(receipts)


class _LedgerFacade:
    """Bind the real immutable H10 ledger to the H23 writer boundary."""

    def __init__(self, ledger: StageLedger) -> None:
        self.ledger = ledger

    def targets_for(self, mode):
        return self.ledger.targets_by_mode.get(mode, frozenset())

    def capture_receipt(self, label, mode, target):
        return self.ledger.receipt(label, mode, target)

    def commit_durable(self, receipts):
        self.ledger.record_durable(receipts)

    def commit_publication_drop(self, label, mode, expected_revision):
        self.ledger.record_publication_dropped(
            label, mode, expected_revision=expected_revision,
        )


def _r1(value: float) -> IntegrationResult1D:
    return IntegrationResult1D(
        radial=np.linspace(0.1, 1.0, 8),
        intensity=np.full(8, value),
        sigma=np.full(8, value / 10),
        unit="q_A^-1",
    )


def _r2(value: float) -> IntegrationResult2D:
    raw = np.arange(48, dtype=float).reshape(8, 6) + value
    return IntegrationResult2D(
        radial=np.linspace(0.1, 1.0, 8),
        azimuthal=np.linspace(-1.0, 1.0, 6),
        intensity=raw,
        sigma=raw / 10,
        unit="q_A^-1",
        azimuthal_unit="chi_deg",
    )


def _transaction(tmp_path: Path, *, prior: bytes = b"prior"):
    target = tmp_path / "out.nexus"
    if prior is not None:
        target.write_bytes(prior)
    coordinator = OutputTransactionCoordinator()
    transaction_owner = OwnerToken("transaction")
    target_owner = OwnerToken("target")
    owners = {role: OwnerToken(role.value) for role in LeaseOwner}
    transaction = coordinator.admit(
        target,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
    )
    lease = transaction.acquire_lease(
        admission=transaction.admission,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        owners=owners,
    )
    return (
        coordinator,
        transaction,
        target,
        transaction_owner,
        target_owner,
        owners,
        lease,
    )


def _release(transaction, lease, owners) -> None:
    for role in (LeaseOwner.RUN, LeaseOwner.SESSION, LeaseOwner.SOURCE,
                 LeaseOwner.CLEANUP):
        transaction.release_lease_owner(lease, role, owners[role])


def _bound_writer(tmp_path, *, prior=None, complete_record=False, facade_type=_Facade):
    from xrd_tools.io.record_writer import NexusRecordWriter, WriterTransactionBinding

    (_coordinator, transaction, target, transaction_owner, target_owner,
     owners, lease) = _transaction(tmp_path, prior=prior)
    pool = _Pool()
    attempt = transaction.begin_stream(
        admission=transaction.admission, transaction_owner=transaction_owner,
        target_owner=target_owner, lease=lease, pool=pool,
        file_lock=threading.RLock(),
    )
    writer = NexusRecordWriter(
        target, atomic=False, flush_every=None, complete_record=complete_record,
        file_lock=threading.RLock(),
        transaction_binding=WriterTransactionBinding(transaction, attempt, lease),
    )
    facade = facade_type(target)
    writer.bind_session(facade)
    writer.begin()
    return writer, transaction, attempt, lease, owners, pool, facade, target


def _begin_finite_overwrite_sink(
    tmp_path,
    filename,
    scan_name,
    *,
    extra=None,
    plan=None,
    bind_session=False,
    same_run_intent=None,
):
    """Build the common fresh finite-Overwrite product route used below."""
    from xrd_tools.core.scan import Scan
    from xrd_tools.reduction import NexusSink, ReductionPlan

    target = tmp_path / filename
    sink = NexusSink(
        target,
        overwrite=True,
        flush_every=None,
        run_configuration_provenance={
            "output_mode": "Overwrite",
            "live_mode": False,
        },
        **({} if same_run_intent is None
           else {"same_run_intent": same_run_intent}),
    )
    facade = _Facade(target) if bind_session else None
    if facade is not None:
        sink.bind_session(facade)
    sink.begin(
        Scan(scan_name, [], extra={} if extra is None else extra),
        ReductionPlan(integration_2d=None) if plan is None else plan,
    )
    return target, sink, facade


def test_failed_checkpoint_publication_retries_the_same_staged_delta(
    tmp_path,
):
    """A seal that raises must not lose the labels it was about to cover.

    `_checkpoint_staged_receipts` reads the staged window; only a COMPLETED
    seal closes it.  When it emptied the window up front instead, a transient
    failure in `commit_checkpoint_recoverable` left `flush()` retryable and
    `_pending` intact, but the delta was gone: the retry sealed a checkpoint
    covering no labels, so those frames' heavy arrays stayed resident until
    close and bounded-memory release silently stopped working for that window.
    """
    from xrd_tools.core.scan import ScanFrame
    from xrd_tools.io.record_writer import WriterIncomplete
    from xrd_tools.reduction import (
        FrameReduction,
        NexusTerminalDisposition,
        ReductionResult,
    )

    _target, sink, facade = _begin_finite_overwrite_sink(
        tmp_path,
        "fast-retry.nexus",
        "fast",
        bind_session=True,
    )
    recoveries = []
    fail_next = [True]

    def flaky_recoverable(*values):
        if fail_next[0]:
            fail_next[0] = False
            raise OSError("transient checkpoint publication failure")
        recoveries.append(values)

    facade.commit_checkpoint_recoverable = flaky_recoverable
    writer = sink._writer

    sink.write(ScanFrame(0), FrameReduction(0, result_1d=_r1(3)))
    staged_before = tuple(writer._staged_since_checkpoint)
    assert staged_before, "the frame should be staged before the first seal"

    with pytest.raises(WriterIncomplete):
        writer.flush(force=True)

    # The failed attempt must leave the window intact AND the flush retryable.
    assert tuple(writer._staged_since_checkpoint) == staged_before
    assert writer._since_flush > 0

    writer.flush(force=True)

    assert len(recoveries) == 1
    _checkpoint, receipts, _drops, frame_labels, _thumbnails = recoveries[0]
    assert tuple(frame_labels) == (0,)
    assert tuple(int(receipt.label) for receipt in receipts) == (0,)
    # The successful seal is what closes the window.
    assert writer._staged_since_checkpoint == {}

    # And the Run still finishes normally after the retry: the single durable
    # publication happens at close exactly as on an unimpeded fast Run.
    terminal = sink.finish(ReductionResult("fast", {}, 1))
    assert terminal.disposition is NexusTerminalDisposition.COMMITTED
    assert len(facade.durable) == 1


def test_grouped_semantic_reads_preserve_order_axes_sigma_and_two_observations(
    tmp_path, monkeypatch,
):
    from xrd_tools.io.record_writer import RecordWrite

    (writer, transaction, attempt, lease, owners, _pool, _facade, target) = (
        _bound_writer(tmp_path))
    groups = []
    real_group_read = writer._read_grouped_mode

    def trace_group(rows, phase, *args):
        groups.append((phase, rows[0].group_name, tuple(row.label for row in rows)))
        return real_group_read(rows, phase, *args)

    monkeypatch.setattr(writer, "_read_grouped_mode", trace_group)
    records = []
    for label in range(10):
        one_d, two_d = _r1(label + 1), _r2(label + 1)
        if label % 2:
            one_d.sigma = two_d.sigma = None
        records.append(RecordWrite(label=label, result_1d=one_d, result_2d=two_d))
    writer.write_batch(records)
    writer.flush(force=True)

    expected_groups = [
        ("integrated_1d", tuple(range(8))),
        ("integrated_1d", (8, 9)),
        ("integrated_2d", tuple(range(8))),
        ("integrated_2d", (8, 9)),
    ]
    assert groups == [("checkpoint", name, labels) for name, labels in expected_groups]
    assert tuple(writer.grouped_semantic_read_volume.values()) == ((4, 20), (0, 0))
    checkpoint = {key: proof.digest
                  for key, proof in writer._durable_mode_proofs.items()}
    assert all(
        not isinstance(getattr(proof, field), np.ndarray)
        for proof in writer._durable_mode_proofs.values()
        for field in proof.__slots__
    )
    close_observed = {}
    close_reads = {}
    real_getitem = h5py.Dataset.__getitem__
    real_reverify = writer._reverify_durable_mode_proof

    def trace_close_reads(dataset, item):
        if (
            writer._pending_owner == "close"
            and "/integrated_" in dataset.name
            and dataset.name.rsplit("/", 1)[-1]
            in {"frame_index", "axis_1", "intensity", "sigma", "axis_2"}
        ):
            close_reads.setdefault(dataset.name, []).append(item)
        return real_getitem(dataset, item)

    def trace_reverify(proof, *args):
        evidence = real_reverify(proof, *args)
        close_observed[(proof.group_name, proof.label)] = evidence.observed_hexdigest()
        return evidence

    monkeypatch.setattr(h5py.Dataset, "__getitem__", trace_close_reads)
    monkeypatch.setattr(writer, "_reverify_durable_mode_proof", trace_reverify)
    writer.finish()
    assert groups[4:] == [("close", name, labels) for name, labels in expected_groups]
    assert close_observed == checkpoint
    assert tuple(writer.grouped_semantic_read_volume.values()) == ((4, 20), (4, 20))
    expected_slices = [slice(0, 8), slice(8, 10)]
    for group_name in ("integrated_1d", "integrated_2d"):
        prefix = f"/{writer.entry}/{group_name}"
        assert close_reads[f"{prefix}/axis_1"] == [()]
        assert close_reads[f"{prefix}/frame_index"] == expected_slices
        assert close_reads[f"{prefix}/intensity"] == expected_slices
        assert all(
            isinstance(item, slice)
            for item in close_reads[f"{prefix}/frame_index"]
        )
    assert close_reads[f"/{writer.entry}/integrated_2d/axis_2"] == [()]
    assert transaction.commit_stream(attempt, lease=lease).phase is TransactionPhase.COMMITTED
    _release(transaction, lease, owners)

    with h5py.File(target, "r") as handle:
        one_d, two_d = handle["entry/integrated_1d"], handle["entry/integrated_2d"]
        np.testing.assert_array_equal(one_d["axis_1"][()], _r1(1).radial.astype("f4"))
        np.testing.assert_array_equal(two_d["axis_1"][()], _r2(1).radial.astype("f4"))
        np.testing.assert_array_equal(two_d["axis_2"][()], _r2(1).azimuthal.astype("f4"))
        for label in range(10):
            np.testing.assert_array_equal(one_d["intensity"][label], _r1(label + 1).intensity)
            np.testing.assert_array_equal(two_d["intensity"][label], _r2(label + 1).intensity.T)
            if label % 2:
                assert np.isnan(one_d["sigma"][label]).all()
                assert np.isnan(two_d["sigma"][label]).all()
            else:
                np.testing.assert_allclose(one_d["sigma"][label], _r1(label + 1).sigma)
                np.testing.assert_allclose(two_d["sigma"][label], _r2(label + 1).sigma.T)


@pytest.mark.parametrize("corrupt_shifted_row", (False, True))
def test_bound_close_keeps_label_content_proof_after_middle_publication_drop(
    tmp_path, monkeypatch, corrupt_shifted_row,
):
    from xrd_tools.io.record_writer import RecordWrite, WriterIncomplete

    class DropFacade(_Facade):
        def __init__(self, target):
            super().__init__(target)
            self.dropped = []

        def commit_publication_drop(self, label, mode, expected_revision):
            self.dropped.append((int(label), mode, int(expected_revision)))

    (writer, transaction, attempt, lease, owners, _pool, facade, target) = (
        _bound_writer(tmp_path, facade_type=DropFacade)
    )
    mode = ResultMode.one_d()
    writer.write_batch(
        RecordWrite(label=label, result_1d=_r1(label + 1))
        for label in (0, 1, 2)
    )
    writer.flush(force=True)
    before = writer._durable_mode_proofs[("integrated_1d", 2)]
    assert before.row == 2

    writer.mark_publication_dropped(1, mode, expected_revision=1)
    writer.flush(force=True)
    shifted = writer._durable_mode_proofs[("integrated_1d", 2)]
    assert shifted.row == 1
    assert shifted.digest == before.digest
    assert facade.dropped == [(1, mode, 1)]

    if corrupt_shifted_row:
        real_verify = writer._seal_verified_stream_close

        def corrupt_then_verify(binding):
            with h5py.File(target, "r+") as handle:
                intensity = handle["entry/integrated_1d/intensity"]
                intensity[shifted.row, 0] = (
                    float(intensity[shifted.row, 0]) + 100.0
                )
            return real_verify(binding)

        monkeypatch.setattr(
            writer, "_seal_verified_stream_close", corrupt_then_verify,
        )
        with pytest.raises(WriterIncomplete, match="durable-row proof changed"):
            writer.finish()
        assert transaction.snapshot().phase is TransactionPhase.INTEGRITY_HOLD
        return

    writer.finish()
    assert transaction.commit_stream(
        attempt, lease=lease,
    ).phase is TransactionPhase.COMMITTED
    _release(transaction, lease, owners)


def test_grouped_semantic_reads_split_sparse_replacements_and_clear_on_failure(
    tmp_path, monkeypatch,
):
    from xrd_tools.io.record_writer import NexusRecordWriter, RecordWrite, WriterIncomplete

    seed_path = tmp_path / "seed.nexus"
    seed = NexusRecordWriter(seed_path, atomic=False, flush_every=None,
                             complete_record=True)
    seed.begin()
    seed.write_batch(
        RecordWrite(label=label, result_1d=_r1(label + 1), result_2d=_r2(label + 1))
        for label in range(10)
    )
    seed.finish()

    class DropFacade(_Facade):
        def __init__(self, target):
            super().__init__(target)
            self.dropped = []

        def commit_publication_drop(self, label, mode, expected_revision):
            self.dropped.append((label, mode, expected_revision))

    (writer, transaction, attempt, lease, owners, _pool, facade, target) = (
        _bound_writer(tmp_path, prior=seed_path.read_bytes(), complete_record=True,
                      facade_type=DropFacade))
    groups = []
    real_group_read = writer._read_grouped_mode

    def trace_group(rows, phase, *args):
        groups.append((phase, rows[0].group_name, tuple(row.label for row in rows)))
        return real_group_read(rows, phase, *args)

    monkeypatch.setattr(writer, "_read_grouped_mode", trace_group)
    writer.write_batch(
        RecordWrite(label=label, result_1d=_r1(100 + label),
                    result_2d=_r2(100 + label), replace_existing=True)
        for label in (0, 2, 3, 9)
    )
    dropped_mode = ResultMode.one_d()
    writer.mark_publication_dropped(5, dropped_mode, expected_revision=1)
    real_verify = writer._verify_mode_row
    failed = []

    def fail_once(evidence, expected, observed=None):
        if expected.label == 2 and not failed:
            failed.append(expected.label)
            raise OSError("grouped row observation unavailable")
        return real_verify(evidence, expected, observed)

    monkeypatch.setattr(writer, "_verify_mode_row", fail_once)
    with pytest.raises(WriterIncomplete, match="grouped row observation unavailable"):
        writer.flush(force=True)
    assert writer._semantic_mode_observation is None
    groups.clear()
    writer.flush(force=True)
    sparse = [(name, labels) for name in ("integrated_1d", "integrated_2d")
              for labels in ((0,), (2, 3), (9,))]
    assert groups == [("checkpoint", name, labels) for name, labels in sparse]
    assert facade.dropped == [(5, dropped_mode, 1)]
    assert ("integrated_1d", 5) in writer._durable_absence_proofs
    writer.finish()
    assert groups[6:] == [("close", name, labels) for name, labels in sparse]
    assert writer._semantic_mode_observation is None
    assert transaction.commit_stream(attempt, lease=lease).phase is TransactionPhase.COMMITTED
    _release(transaction, lease, owners)
    with h5py.File(target, "r") as handle:
        one_d, two_d = handle["entry/integrated_1d"], handle["entry/integrated_2d"]
        assert tuple(one_d["frame_index"][()]) == (0, 1, 2, 3, 4, 6, 7, 8, 9)
        assert tuple(two_d["frame_index"][()]) == tuple(range(10))
        for label in (0, 2, 3, 9):
            row_1d = list(one_d["frame_index"][()]).index(label)
            np.testing.assert_array_equal(one_d["intensity"][row_1d], _r1(100 + label).intensity)
            np.testing.assert_array_equal(two_d["intensity"][label], _r2(100 + label).intensity.T)


def test_grouped_close_rejects_rank_drift_in_frame_index(tmp_path, monkeypatch):
    from xrd_tools.io.record_writer import RecordWrite, WriterIncomplete

    (writer, transaction, _attempt, _lease, _owners, _pool, _facade, _target) = (
        _bound_writer(tmp_path)
    )
    writer.write_batch(
        RecordWrite(label=label, result_1d=_r1(label + 1))
        for label in range(2)
    )
    writer.flush(force=True)
    real_getitem = h5py.Dataset.__getitem__

    def rank_drift(dataset, item):
        observed = real_getitem(dataset, item)
        if (
            writer._pending_owner == "close"
            and dataset.name.endswith("/integrated_1d/frame_index")
            and isinstance(item, slice)
        ):
            return np.asarray(observed).reshape(-1, 1)
        return observed

    monkeypatch.setattr(h5py.Dataset, "__getitem__", rank_drift)
    with pytest.raises(WriterIncomplete, match="grouped durability read lost rows"):
        writer.finish()
    assert transaction.snapshot().phase is TransactionPhase.INTEGRITY_HOLD


def test_stream_attempt_keeps_one_lease_through_open_flush_and_commit(tmp_path):
    (coordinator, transaction, target, transaction_owner, target_owner,
     owners, lease) = _transaction(tmp_path)
    pool = _Pool()
    lock = threading.RLock()

    attempt = transaction.begin_stream(
        admission=transaction.admission,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease,
        pool=pool,
        file_lock=lock,
    )
    assert attempt.target == os.path.normcase(os.path.abspath(target))
    with pytest.raises(LeaseUnavailable):
        contender = coordinator.admit(
            target,
            transaction_owner=OwnerToken("other transaction"),
            target_owner=OwnerToken("other target"),
        )
        contender.acquire_lease(
            admission=contender.admission,
            transaction_owner=contender._transaction_owner,
            target_owner=contender._target_owner,
            owners={role: OwnerToken(f"other {role.value}") for role in LeaseOwner},
        )

    transaction.authorize_stream_mutation(attempt, lease=lease)
    target.write_bytes(b"durable checkpoint")
    evidence = hashlib.sha256(b"expected=observed:row-0").hexdigest()
    descriptor = os.open(target, os.O_RDONLY)
    try:
        checkpoint = transaction.seal_stream_checkpoint(
            attempt,
            lease=lease,
            descriptor=descriptor,
            evidence_digest=evidence,
            evidence_bytes=len(b"expected=observed:row-0"),
        )
    finally:
        os.close(descriptor)
    assert checkpoint.size == len(b"durable checkpoint")
    assert checkpoint.evidence_digest == evidence
    assert checkpoint.evidence_bytes == len(b"expected=observed:row-0")
    with pytest.raises(LeaseUnavailable):
        contender = coordinator.admit(
            target,
            transaction_owner=OwnerToken("flush contender"),
            target_owner=OwnerToken("flush target"),
        )
        contender.acquire_lease(
            admission=contender.admission,
            transaction_owner=contender._transaction_owner,
            target_owner=contender._target_owner,
            owners={role: OwnerToken(f"flush {role.value}") for role in LeaseOwner},
        )

    terminal = transaction.seal_stream_terminal(attempt, lease=lease)
    assert terminal.digest == hashlib.sha256(b"durable checkpoint").hexdigest()
    snapshot = transaction.commit_stream(attempt, lease=lease)
    assert snapshot.phase is TransactionPhase.COMMITTED
    assert pool.events == [("pause", str(target)), ("resume", str(target))]
    _release(transaction, lease, owners)


def test_checkpoint_rejects_same_stat_changed_bytes(tmp_path):
    (_coordinator, transaction, target, transaction_owner, target_owner,
     _owners, lease) = _transaction(tmp_path)
    attempt = transaction.begin_stream(
        admission=transaction.admission,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease,
        pool=_Pool(),
        file_lock=threading.RLock(),
    )
    transaction.authorize_stream_mutation(attempt, lease=lease)
    target.write_bytes(b"AAAA")
    descriptor = os.open(target, os.O_RDONLY)
    try:
        transaction.seal_stream_checkpoint(
            attempt,
            lease=lease,
            descriptor=descriptor,
            evidence_digest=hashlib.sha256(b"row-0").hexdigest(),
            evidence_bytes=len(b"row-0"),
        )
    finally:
        os.close(descriptor)
    accepted = target.stat()
    target.write_bytes(b"BBBB")
    os.utime(target, ns=(accepted.st_atime_ns, accepted.st_mtime_ns))

    with pytest.raises(TargetChanged, match="checkpoint"):
        transaction.authorize_stream_mutation(attempt, lease=lease)
    assert transaction.snapshot().phase is TransactionPhase.INTEGRITY_HOLD
    assert RetryAction.ROLLBACK in transaction.snapshot().pending_actions


@pytest.mark.parametrize("boundary", ("final", "epoch"))
def test_stream_terminal_commit_transfers_one_full_seal_under_exact_exclusion(
    tmp_path, monkeypatch, boundary,
):
    module = importlib.import_module("xrd_tools.io.output_transaction")

    class TrackingLock:
        def __init__(self):
            self.lock = threading.RLock()
            self.depth = 0

        def __enter__(self):
            self.lock.acquire()
            self.depth += 1
            return self

        def __exit__(self, *_args):
            self.depth -= 1
            self.lock.release()

    (_coordinator, transaction, target, transaction_owner, target_owner,
     owners, lease) = _transaction(tmp_path)
    pool, lock = _Pool(), TrackingLock()
    attempt = transaction.begin_stream(
        admission=transaction.admission,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease,
        pool=pool,
        file_lock=lock,
    )
    transaction.authorize_stream_mutation(attempt, lease=lease)
    target.write_bytes(b"AAAA")
    hashes = 0
    stat_checks = 0
    real_hash = module._sha256_handle
    real_revision_check = module._stream_revision_mismatch

    def count_hash(handle):
        nonlocal hashes
        observed = os.fstat(handle.fileno())
        current = target.stat()
        if (observed.st_dev, observed.st_ino) == (current.st_dev, current.st_ino):
            assert lock.depth > 0
            hashes += 1
        return real_hash(handle)

    def check_stat_under_exclusion(path, receipt, **kwargs):
        nonlocal stat_checks
        assert lock.depth > 0
        stat_checks += 1
        return real_revision_check(path, receipt, **kwargs)

    monkeypatch.setattr(module, "_sha256_handle", count_hash)
    monkeypatch.setattr(module, "_stream_revision_mismatch", check_stat_under_exclusion)
    with lock:
        transaction.seal_stream_terminal(attempt, lease=lease)
    if boundary == "final":
        snapshot = transaction.commit_stream(attempt, lease=lease)
    else:
        snapshot = transaction.commit_stream_epoch(attempt, lease=lease)
        assert snapshot.phase is TransactionPhase.EPOCH_COMMITTED
        snapshot = transaction.commit_stream(attempt, lease=lease)
    assert snapshot.phase is TransactionPhase.COMMITTED
    assert hashes == 1
    assert stat_checks >= 1
    _release(transaction, lease, owners)


def test_stream_terminal_commit_rejects_stat_or_identity_drift_without_rehash(
    tmp_path, monkeypatch,
):
    module = importlib.import_module("xrd_tools.io.output_transaction")

    for drift in ("stat", "identity"):
        root = tmp_path / drift
        root.mkdir()
        (_coordinator, transaction, target, transaction_owner, target_owner,
         _owners, lease) = _transaction(root)
        attempt = transaction.begin_stream(
            admission=transaction.admission,
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
            pool=_Pool(),
            file_lock=threading.RLock(),
        )
        transaction.authorize_stream_mutation(attempt, lease=lease)
        target.write_bytes(b"AAAA")
        transaction.seal_stream_terminal(attempt, lease=lease)
        if drift == "stat":
            accepted = target.stat()
            os.utime(
                target,
                ns=(accepted.st_atime_ns, accepted.st_mtime_ns + 1_000_000_000),
            )
        else:
            replacement = root / "replacement.nexus"
            replacement.write_bytes(b"AAAA")
            os.replace(replacement, target)

        with monkeypatch.context() as patch:
            patch.setattr(
                module,
                "_sha256_handle",
                lambda *_args: (_ for _ in ()).throw(
                    AssertionError("terminal commit rehashed the full target")
                ),
            )
            with pytest.raises(TargetChanged, match="terminal seal"):
                transaction.commit_stream(attempt, lease=lease)
        assert transaction.snapshot().phase is TransactionPhase.INTEGRITY_HOLD


def test_stream_terminal_cleanup_retry_reuses_seal_and_retains_exclusion(
    tmp_path, monkeypatch,
):
    module = importlib.import_module("xrd_tools.io.output_transaction")

    class TrackingLock:
        def __init__(self):
            self.lock = threading.RLock()
            self.depth = 0

        def __enter__(self):
            self.lock.acquire()
            self.depth += 1
            return self

        def __exit__(self, *_args):
            self.depth -= 1
            self.lock.release()

    for boundary in ("final", "epoch"):
        root = tmp_path / boundary
        root.mkdir()
        (coordinator, transaction, target, transaction_owner, target_owner,
         owners, lease) = _transaction(root)
        pool, lock = _Pool(), TrackingLock()
        attempt = transaction.begin_stream(
            admission=transaction.admission,
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
            pool=pool,
            file_lock=lock,
        )
        transaction.authorize_stream_mutation(attempt, lease=lease)
        target.write_bytes(b"terminal")
        real_hash = module._sha256_handle
        real_unlink = module._unlink
        real_revision_check = module._stream_revision_mismatch
        hashes = 0
        stat_checks = 0
        failed = False
        sealed_receipt = None
        sealed_stat = None

        def count_hash(handle):
            nonlocal hashes
            observed = os.fstat(handle.fileno())
            current = target.stat()
            if (observed.st_dev, observed.st_ino) == (current.st_dev, current.st_ino):
                hashes += 1
            return real_hash(handle)

        def check_stat_under_exclusion(path, receipt, **kwargs):
            nonlocal stat_checks
            assert lock.depth > 0
            stat_checks += 1
            return real_revision_check(path, receipt, **kwargs)

        def fail_backup_once(path):
            nonlocal failed
            if Path(path) == transaction.backup and not failed:
                failed = True
                raise OSError("backup cleanup fault")
            return real_unlink(path)

        with monkeypatch.context() as patch:
            patch.setattr(module, "_sha256_handle", count_hash)
            patch.setattr(module, "_stream_revision_mismatch", check_stat_under_exclusion)
            patch.setattr(module, "_unlink", fail_backup_once)
            with lock:
                transaction.seal_stream_terminal(attempt, lease=lease)
            sealed_receipt = transaction._stream_terminal_receipt
            sealed_stat = transaction._stream_terminal_stat
            commit = (
                transaction.commit_stream
                if boundary == "final"
                else transaction.commit_stream_epoch
            )
            with pytest.raises(CleanupIncomplete):
                commit(attempt, lease=lease)
            held = transaction.snapshot()
            assert held.phase is TransactionPhase.CLEANUP_PENDING
            # Admission fingerprints before it reaches the lease refusal; do
            # not charge that independent contender read to terminal retry.
            with monkeypatch.context() as admission_patch:
                admission_patch.setattr(module, "_sha256_handle", real_hash)
                with pytest.raises(LeaseUnavailable):
                    contender = coordinator.admit(
                        target,
                        transaction_owner=OwnerToken("retry contender"),
                        target_owner=OwnerToken("retry target"),
                    )
                    contender.acquire_lease(
                        admission=contender.admission,
                        transaction_owner=contender._transaction_owner,
                        target_owner=contender._target_owner,
                        owners={
                            role: OwnerToken(f"retry {role.value}")
                            for role in LeaseOwner
                        },
                    )
            complete = transaction.retry_cleanup(held.cleanup_token)
            expected = (
                TransactionPhase.COMMITTED
                if boundary == "final"
                else TransactionPhase.EPOCH_COMMITTED
            )
            assert complete.phase is expected
            assert transaction._stream_terminal_receipt is sealed_receipt
            assert transaction._stream_terminal_stat is sealed_stat
            if boundary == "epoch":
                complete = transaction.commit_stream(attempt, lease=lease)
                assert complete.phase is TransactionPhase.COMMITTED
            assert hashes == 1, boundary
            assert stat_checks >= 2
        _release(transaction, lease, owners)


def test_routine_checkpoint_never_hashes_or_captures_the_whole_target(
    tmp_path, monkeypatch,
):
    module = importlib.import_module("xrd_tools.io.output_transaction")
    (_coordinator, transaction, target, transaction_owner, target_owner,
     _owners, lease) = _transaction(tmp_path, prior=b"P" * (3 * 1024 * 1024))
    attempt = transaction.begin_stream(
        admission=transaction.admission,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease,
        pool=_Pool(),
        file_lock=threading.RLock(),
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("routine checkpoint performed a whole-file read")

    monkeypatch.setattr(module, "_sha256_handle", forbidden)
    monkeypatch.setattr(module, "_capture_target", forbidden)
    transaction.authorize_stream_mutation(attempt, lease=lease)
    with target.open("r+b") as handle:
        handle.seek(1024)
        handle.write(b"dirty-row")
        handle.flush()
        checkpoint = transaction.seal_stream_checkpoint(
            attempt,
            lease=lease,
            descriptor=handle.fileno(),
            evidence_digest=hashlib.sha256(b"expected=observed:dirty-row").hexdigest(),
            evidence_bytes=len(b"dirty-row"),
        )
    assert checkpoint.evidence_bytes == len(b"dirty-row")


def test_rollback_resume_failure_retains_lease_until_exact_retry(tmp_path):
    (coordinator, transaction, target, transaction_owner, target_owner,
     owners, lease) = _transaction(tmp_path)
    pool = _Pool(fail_resume=1)
    attempt = transaction.begin_stream(
        admission=transaction.admission,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease,
        pool=pool,
        file_lock=threading.RLock(),
    )
    transaction.authorize_stream_mutation(attempt, lease=lease)
    target.write_bytes(b"partial")

    with pytest.raises(CleanupIncomplete):
        transaction.abort_stream(attempt, lease=lease)
    assert target.read_bytes() == b"prior"
    with pytest.raises(LeaseUnavailable):
        contender = coordinator.admit(
            target,
            transaction_owner=OwnerToken("cleanup contender"),
            target_owner=OwnerToken("cleanup target"),
        )
        contender.acquire_lease(
            admission=contender.admission,
            transaction_owner=contender._transaction_owner,
            target_owner=contender._target_owner,
            owners={role: OwnerToken(f"cleanup {role.value}") for role in LeaseOwner},
        )

    cleanup = transaction.snapshot().cleanup_token
    assert cleanup is not None
    transaction.retry_cleanup(cleanup)
    snapshot = transaction.abort_stream(attempt, lease=lease)
    assert snapshot.phase is TransactionPhase.ABORTED
    _release(transaction, lease, owners)


def test_stream_retire_post_rename_observation_retries_from_exact_receipt(
    tmp_path, monkeypatch,
):
    module = importlib.import_module("xrd_tools.io.output_transaction")
    (_coordinator, transaction, target, transaction_owner, target_owner,
     owners, lease) = _transaction(tmp_path)
    attempt = transaction.begin_stream(
        admission=transaction.admission,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease,
        pool=_Pool(),
        file_lock=threading.RLock(),
    )
    transaction.authorize_stream_mutation(attempt, lease=lease)
    target.write_bytes(b"uncommitted stream bytes")
    partial = transaction._stream_partial
    real_capture = module._capture_target
    failed = False

    def fail_first_destination_observation(path, **kwargs):
        nonlocal failed
        if Path(path) == partial and partial.exists() and not target.exists() and not failed:
            failed = True
            raise OSError("post-rename observation failed")
        return real_capture(path, **kwargs)

    monkeypatch.setattr(module, "_capture_target", fail_first_destination_observation)
    with pytest.raises(CleanupIncomplete):
        transaction.abort_stream(attempt, lease=lease)
    receipt = transaction._stream_partial_receipt
    assert receipt is not None and partial.read_bytes() == b"uncommitted stream bytes"
    cleanup = transaction.snapshot().cleanup_token
    assert cleanup is not None
    transaction.retry_cleanup(cleanup)
    assert transaction.abort_stream(attempt, lease=lease).phase is TransactionPhase.ABORTED
    assert target.read_bytes() == b"prior"
    _release(transaction, lease, owners)


def test_stream_retire_replace_error_after_rename_keeps_exact_receipt(
    tmp_path, monkeypatch,
):
    module = importlib.import_module("xrd_tools.io.output_transaction")
    (_coordinator, transaction, target, transaction_owner, target_owner,
     owners, lease) = _transaction(tmp_path)
    attempt = transaction.begin_stream(
        admission=transaction.admission,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease,
        pool=_Pool(),
        file_lock=threading.RLock(),
    )
    transaction.authorize_stream_mutation(attempt, lease=lease)
    target.write_bytes(b"uncommitted stream bytes")
    partial = transaction._stream_partial
    real_replace = module._replace
    failed = False

    def replace_then_raise(source, destination):
        nonlocal failed
        result = real_replace(source, destination)
        if Path(destination) == partial and not failed:
            failed = True
            raise OSError("rename completion was reported as an error")
        return result

    monkeypatch.setattr(module, "_replace", replace_then_raise)
    with pytest.raises(CleanupIncomplete):
        transaction.abort_stream(attempt, lease=lease)
    receipt = transaction._stream_partial_receipt
    assert receipt is not None
    assert partial.read_bytes() == b"uncommitted stream bytes"
    assert not target.exists()

    transaction.retry_cleanup(transaction.snapshot().cleanup_token)
    assert transaction.abort_stream(attempt, lease=lease).phase is TransactionPhase.ABORTED
    assert target.read_bytes() == b"prior"
    assert not partial.exists()
    _release(transaction, lease, owners)


def test_stream_retire_receipt_refuses_replaced_partial_after_observation_fault(
    tmp_path, monkeypatch,
):
    module = importlib.import_module("xrd_tools.io.output_transaction")
    (_coordinator, transaction, target, transaction_owner, target_owner,
     _owners, lease) = _transaction(tmp_path)
    attempt = transaction.begin_stream(
        admission=transaction.admission,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease,
        pool=_Pool(),
        file_lock=threading.RLock(),
    )
    transaction.authorize_stream_mutation(attempt, lease=lease)
    target.write_bytes(b"owned bytes")
    partial = transaction._stream_partial
    real_capture = module._capture_target
    failed = False

    def fail_once(path, **kwargs):
        nonlocal failed
        if Path(path) == partial and partial.exists() and not target.exists() and not failed:
            failed = True
            raise OSError("post-rename observation failed")
        return real_capture(path, **kwargs)

    monkeypatch.setattr(module, "_capture_target", fail_once)
    with pytest.raises(CleanupIncomplete):
        transaction.abort_stream(attempt, lease=lease)
    partial.unlink()
    partial.write_bytes(b"foreign replacement")
    cleanup = transaction.snapshot().cleanup_token
    with pytest.raises(CleanupIncomplete):
        transaction.retry_cleanup(cleanup)
    snapshot = transaction.snapshot()
    assert snapshot.phase is TransactionPhase.CLEANUP_PENDING
    assert RetryAction.STREAM_RETIRE in snapshot.pending_actions
    assert target.read_bytes() == b"prior"


def test_stream_partial_absence_without_authorized_cleanup_stays_held(
    tmp_path, monkeypatch,
):
    module = importlib.import_module("xrd_tools.io.output_transaction")
    (_coordinator, transaction, target, transaction_owner, target_owner,
     _owners, lease) = _transaction(tmp_path)
    attempt = transaction.begin_stream(
        admission=transaction.admission,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease, pool=_Pool(), file_lock=threading.RLock(),
    )
    partial = transaction._stream_partial
    real_capture = module._capture_target
    failed = False

    def fail_once(path, **kwargs):
        nonlocal failed
        if Path(path) == partial and partial.exists() and not target.exists() and not failed:
            failed = True
            raise OSError("post-rename observation failed")
        return real_capture(path, **kwargs)

    monkeypatch.setattr(module, "_capture_target", fail_once)
    with pytest.raises(CleanupIncomplete):
        transaction.abort_stream(attempt, lease=lease)
    partial.unlink()
    with pytest.raises(CleanupIncomplete):
        transaction.retry_cleanup(transaction.snapshot().cleanup_token)
    assert RetryAction.STREAM_RETIRE in transaction.snapshot().pending_actions
    assert target.read_bytes() == b"prior"


def test_stream_partial_authorized_unlink_observation_retries(tmp_path, monkeypatch):
    module = importlib.import_module("xrd_tools.io.output_transaction")
    (_coordinator, transaction, target, transaction_owner, target_owner,
     owners, lease) = _transaction(tmp_path)
    attempt = transaction.begin_stream(
        admission=transaction.admission,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease, pool=_Pool(), file_lock=threading.RLock(),
    )
    partial = transaction._stream_partial
    real_unlink = module._unlink
    failed = False

    def unlink_then_raise(path):
        nonlocal failed
        result = real_unlink(path)
        if Path(path) == partial and not failed:
            failed = True
            raise OSError("unlink observation failed")
        return result

    monkeypatch.setattr(module, "_unlink", unlink_then_raise)
    with pytest.raises(CleanupIncomplete):
        transaction.abort_stream(attempt, lease=lease)
    transaction.retry_cleanup(transaction.snapshot().cleanup_token)
    assert transaction.abort_stream(attempt, lease=lease).phase is TransactionPhase.ABORTED
    assert target.read_bytes() == b"prior" and not partial.exists()
    _release(transaction, lease, owners)


def test_bound_writer_checkpoint_is_semantic_k_payload_not_file_size(
    tmp_path, monkeypatch,
):
    from xrd_tools.io.record_writer import (
        NexusRecordWriter,
        RecordWrite,
        WriterTransactionBinding,
    )
    target = tmp_path / "large.nexus"
    _seed_target(target, (0,), tmp_path)
    with h5py.File(target, "r+") as handle:
        handle.create_dataset("unrelated_padding", data=np.zeros(3 * 1024 * 1024,
                                                                  dtype=np.uint8))
    (coordinator, transaction, target, transaction_owner, target_owner,
     owners, lease) = _transaction(tmp_path, prior=target.read_bytes())
    pool = _Pool()
    attempt = transaction.begin_stream(
        admission=transaction.admission,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease,
        pool=pool,
        file_lock=threading.RLock(),
    )
    writer = NexusRecordWriter(
        target,
        atomic=False,
        flush_every=None,
        complete_record=False,
        file_lock=threading.RLock(),
        transaction_binding=WriterTransactionBinding(transaction, attempt, lease),
    )
    facade = _Facade(target)
    writer.bind_session(facade)
    writer.begin()
    writer.write(RecordWrite(label=3, result_1d=_r1(3), result_2d=_r2(3)))
    module = importlib.import_module("xrd_tools.io.output_transaction")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("routine writer checkpoint read the whole file")

    with monkeypatch.context() as patch:
        patch.setattr(module, "_capture_target", forbidden)
        patch.setattr(module, "_sha256_handle", forbidden)
        writer.flush(force=True)
    assert {(r.label, r.mode) for r in facade.durable} == {
        (3, ResultMode.one_d("default")),
        (3, ResultMode.two_d("default")),
    }
    rows, read_bytes = writer.checkpoint_read_volume
    assert rows == 2
    assert 0 < read_bytes < 32 * 1024
    assert len(writer._durable_mode_proofs) == 2
    assert all(
        not isinstance(getattr(proof, field), np.ndarray)
        for proof in writer._durable_mode_proofs.values()
        for field in proof.__slots__
    ), "abort proof metadata must not become a second ndarray-buffer owner"
    writer.finish()
    assert transaction.commit_stream(attempt, lease=lease).phase is TransactionPhase.COMMITTED
    _release(transaction, lease, owners)
    assert pool.events == [("pause", str(target)), ("resume", str(target))]


def test_bound_writer_withholds_durable_receipts_until_checkpoint_seals(
    tmp_path, monkeypatch,
):
    from xrd_tools.io.record_writer import (
        NexusRecordWriter,
        RecordWrite,
        WriterIncomplete,
        WriterTransactionBinding,
    )

    (coordinator, transaction, target, transaction_owner, target_owner,
     owners, lease) = _transaction(tmp_path, prior=None)
    attempt = transaction.begin_stream(
        admission=transaction.admission,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease,
        pool=_Pool(),
        file_lock=threading.RLock(),
    )
    writer = NexusRecordWriter(
        target,
        atomic=False,
        flush_every=None,
        complete_record=False,
        file_lock=threading.RLock(),
        transaction_binding=WriterTransactionBinding(transaction, attempt, lease),
    )
    facade = _Facade(target)
    writer.bind_session(facade)
    writer.begin()
    writer.write(RecordWrite(label=3, result_1d=_r1(3)))
    with monkeypatch.context() as patch:
        patch.setattr(
            transaction,
            "seal_stream_checkpoint",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("checkpoint seal failed")
            ),
        )
        with pytest.raises(WriterIncomplete, match="flush incomplete"):
            writer.flush(force=True)
    assert facade.durable == []
    writer.abort()
    transaction.abort_stream(attempt, lease=lease)
    _release(transaction, lease, owners)


def test_h10_commit_failure_cannot_reopen_rollback_below_durable_floor(tmp_path):
    """The irreversible floor precedes even a partially successful H10 call."""
    from xrd_tools.io.record_writer import (
        NexusRecordWriter,
        RecordWrite,
        WriterIncomplete,
        WriterTransactionBinding,
    )

    (coordinator, transaction, target, transaction_owner, target_owner,
     owners, lease) = _transaction(tmp_path, prior=None)
    pool = _Pool()
    attempt = transaction.begin_stream(
        admission=transaction.admission,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease,
        pool=pool,
        file_lock=threading.RLock(),
    )
    writer = NexusRecordWriter(
        target,
        atomic=False,
        flush_every=None,
        complete_record=False,
        file_lock=threading.RLock(),
        transaction_binding=WriterTransactionBinding(transaction, attempt, lease),
    )

    class CommitThenFailFacade(_Facade):
        def commit_durable(self, receipts):
            snapshot = transaction.snapshot()
            assert snapshot.durable_floor is not None
            assert RetryAction.ROLLBACK not in snapshot.pending_actions
            super().commit_durable(receipts)
            raise OSError("H10 acknowledgement was lost after commit")

    facade = CommitThenFailFacade(target)
    writer.bind_session(facade)
    writer.begin()
    writer.write(RecordWrite(label=3, result_1d=_r1(3), result_2d=_r2(3)))

    with pytest.raises(WriterIncomplete, match="flush incomplete"):
        writer.flush(force=True)
    assert {(receipt.label, receipt.mode) for receipt in facade.durable} == {
        (3, ResultMode.one_d("default")),
        (3, ResultMode.two_d("default")),
    }

    writer.abort()
    snapshot = transaction.abort_stream(attempt, lease=lease)
    assert snapshot.phase is TransactionPhase.ABORTED
    assert snapshot.durable_floor is not None
    assert snapshot.partial_path == str(target)
    with h5py.File(target, "r") as handle:
        assert tuple(handle["entry/integrated_1d/frame_index"][()]) == (3,)
        assert tuple(handle["entry/integrated_2d/frame_index"][()]) == (3,)
    _release(transaction, lease, owners)
    assert pool.events == [("pause", str(target)), ("resume", str(target))]


def test_publication_drop_checkpoint_is_an_irreversible_static_h10_floor(tmp_path):
    from xrd_tools.io.record_writer import (
        NexusRecordWriter,
        RecordWrite,
        WriterTransactionBinding,
    )

    target = tmp_path / "drop-floor.nexus"
    seed = NexusRecordWriter(
        target, atomic=False, flush_every=None, complete_record=False,
    )
    seed.begin()
    seed.write(RecordWrite(label=0, result_1d=_r1(1)))
    seed.write(RecordWrite(label=1, result_1d=_r1(2)))
    seed.finish()
    prior = target.read_bytes()

    (_coordinator, transaction, target, transaction_owner, target_owner,
     owners, lease) = _transaction(tmp_path, prior=prior)
    pool = _Pool()
    attempt = transaction.begin_stream(
        admission=transaction.admission,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease,
        pool=pool,
        file_lock=threading.RLock(),
    )
    mode = ResultMode.one_d()
    target_name = f"nexus:{target}"
    ledger = StageLedger(
        required_modes=(mode,), targets_by_mode={mode: (target_name,)},
    )
    accepted = ledger.record_accepted(0)
    ledger.record_outcome(
        0, ItemDisposition.COMPLETED, produced=(mode,), attempt=accepted,
    )
    ledger.record_written(0, (mode,))
    writer = NexusRecordWriter(
        target,
        atomic=False,
        flush_every=None,
        complete_record=False,
        file_lock=threading.RLock(),
        transaction_binding=WriterTransactionBinding(transaction, attempt, lease),
    )
    writer.bind_session(_LedgerFacade(ledger))
    writer.begin()
    writer.mark_publication_dropped(0, mode, expected_revision=1)
    writer.flush(force=True)

    assert ledger.snapshot().publication_dropped == frozenset(((0, mode),))
    assert transaction.snapshot().durable_floor is not None
    writer.abort()
    snapshot = transaction.abort_stream(attempt, lease=lease)
    assert snapshot.phase is TransactionPhase.ABORTED
    assert snapshot.durable_floor is not None
    assert snapshot.partial_path == str(target)
    with h5py.File(target, "r") as handle:
        assert tuple(handle["entry/integrated_1d/frame_index"][()]) == (1,)
        np.testing.assert_array_equal(
            handle["entry/integrated_1d/intensity"][0], _r1(2).intensity,
        )
    _release(transaction, lease, owners)
    assert pool.events == [("pause", str(target)), ("resume", str(target))]


def test_durable_abort_refuses_row_corruption_during_controlled_close(
    tmp_path,
    monkeypatch,
):
    """A successful close cannot bless changed bytes for an H10 durable row."""
    from xrd_tools.io.record_writer import (
        NexusRecordWriter,
        RecordWrite,
        WriterIncomplete,
        WriterTransactionBinding,
    )

    (coordinator, transaction, target, transaction_owner, target_owner,
     owners, lease) = _transaction(tmp_path, prior=None)
    pool = _Pool()
    attempt = transaction.begin_stream(
        admission=transaction.admission,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease,
        pool=pool,
        file_lock=threading.RLock(),
    )
    writer = NexusRecordWriter(
        target,
        atomic=False,
        flush_every=None,
        complete_record=False,
        file_lock=threading.RLock(),
        transaction_binding=WriterTransactionBinding(transaction, attempt, lease),
    )
    facade = _Facade(target)
    writer.bind_session(facade)
    writer.begin()
    writer.write(RecordWrite(label=3, result_1d=_r1(3)))
    writer.flush(force=True)
    real_verify = writer._seal_verified_stream_close

    def corrupt_then_verify(binding):
        with h5py.File(target, "r+") as handle:
            intensity = handle["entry/integrated_1d/intensity"]
            intensity[0, 0] = float(intensity[0, 0]) + 100.0
        return real_verify(binding)

    monkeypatch.setattr(writer, "_seal_verified_stream_close", corrupt_then_verify)
    with pytest.raises(WriterIncomplete, match="durable-row proof changed"):
        writer.abort()

    snapshot = transaction.snapshot()
    assert snapshot.phase is TransactionPhase.INTEGRITY_HOLD
    assert snapshot.durable_floor is not None
    assert RetryAction.ROLLBACK not in snapshot.pending_actions
    assert RetryAction.POOL_RESUME in snapshot.pending_actions
    assert facade.durable
    assert snapshot.cleanup_token is not None
    retried = transaction.retry_cleanup(snapshot.cleanup_token)
    assert retried.phase is TransactionPhase.INTEGRITY_HOLD
    assert RetryAction.POOL_RESUME in retried.pending_actions
    assert pool.events == [("pause", str(target))]


def test_durable_abort_ties_semantic_readback_to_close_descriptor_stat(
    tmp_path,
    monkeypatch,
):
    """A race after row proof retains the integrity owner instead of blessing."""
    from xrd_tools.io.record_writer import (
        NexusRecordWriter,
        RecordWrite,
        WriterIncomplete,
        WriterTransactionBinding,
    )

    (coordinator, transaction, target, transaction_owner, target_owner,
     owners, lease) = _transaction(tmp_path, prior=None)
    attempt = transaction.begin_stream(
        admission=transaction.admission,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease,
        pool=_Pool(),
        file_lock=threading.RLock(),
    )
    writer = NexusRecordWriter(
        target,
        atomic=False,
        flush_every=None,
        complete_record=False,
        file_lock=threading.RLock(),
        transaction_binding=WriterTransactionBinding(transaction, attempt, lease),
    )
    facade = _Facade(target)
    writer.bind_session(facade)
    writer.begin()
    writer.write(RecordWrite(label=3, result_1d=_r1(3)))
    writer.flush(force=True)
    real_seal = transaction.seal_stream_close

    def race_after_proof(*args, **kwargs):
        with target.open("ab") as handle:
            handle.write(b"foreign-race")
        return real_seal(*args, **kwargs)

    monkeypatch.setattr(transaction, "seal_stream_close", race_after_proof)
    with pytest.raises(
        WriterIncomplete,
        match="descriptor changed after semantic verification",
    ):
        writer.abort()

    snapshot = transaction.snapshot()
    assert snapshot.phase is TransactionPhase.INTEGRITY_HOLD
    assert snapshot.durable_floor is not None
    assert RetryAction.ROLLBACK not in snapshot.pending_actions
    assert RetryAction.POOL_RESUME in snapshot.pending_actions
    assert facade.durable


def test_durable_abort_retries_transient_semantic_observation_exactly(
    tmp_path,
    monkeypatch,
):
    """A failed observation retains the close owner; a later proof may resolve."""
    from xrd_tools.io.record_writer import (
        NexusRecordWriter,
        RecordWrite,
        WriterIncomplete,
        WriterTransactionBinding,
    )

    (coordinator, transaction, target, transaction_owner, target_owner,
     owners, lease) = _transaction(tmp_path, prior=None)
    pool = _Pool()
    attempt = transaction.begin_stream(
        admission=transaction.admission,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease,
        pool=pool,
        file_lock=threading.RLock(),
    )
    writer = NexusRecordWriter(
        target,
        atomic=False,
        flush_every=None,
        complete_record=False,
        file_lock=threading.RLock(),
        transaction_binding=WriterTransactionBinding(transaction, attempt, lease),
    )
    facade = _Facade(target)
    writer.bind_session(facade)
    writer.begin()
    writer.write(RecordWrite(label=3, result_1d=_r1(3)))
    writer.flush(force=True)
    real_verify = writer._reverify_durable_mode_proof
    failures = []

    def fail_once(proof, *args):
        if not failures:
            failures.append("observation")
            raise OSError("post-close semantic observation unavailable")
        return real_verify(proof, *args)

    monkeypatch.setattr(writer, "_reverify_durable_mode_proof", fail_once)
    with pytest.raises(WriterIncomplete, match="observation unavailable"):
        writer.abort()
    held = transaction.snapshot()
    assert held.phase is TransactionPhase.INTEGRITY_HOLD
    assert RetryAction.POOL_RESUME in held.pending_actions
    assert pool.events == [("pause", str(target))]

    writer.abort()
    complete = transaction.abort_stream(attempt, lease=lease)
    assert complete.phase is TransactionPhase.ABORTED
    assert complete.partial_path == str(target)
    assert pool.events == [("pause", str(target)), ("resume", str(target))]
    _release(transaction, lease, owners)


def test_sink_role_release_retries_only_remaining_exact_owners(tmp_path):
    from xrd_tools.reduction import NexusSink

    class Releaser:
        def __init__(self):
            self.calls = []
            self.failed = False

        def release_lease_owner(self, _lease, role, _owner):
            self.calls.append(role)
            if role is LeaseOwner.SESSION and not self.failed:
                self.failed = True
                raise OSError("release fault")

    sink = NexusSink(tmp_path / "release.nexus")
    sink._transaction = Releaser()
    sink._lease = object()
    owners = {role: object() for role in LeaseOwner}
    sink._transaction_owners = (object(), object(), owners)
    with pytest.raises(OSError, match="release fault"):
        sink._release_terminal_lease()
    assert LeaseOwner.RUN not in owners and LeaseOwner.SESSION in owners
    sink._release_terminal_lease()
    assert sink._transaction.calls.count(LeaseOwner.RUN) == 1
    assert sink._transaction_owners is None


def test_sink_constructor_failure_aborts_seed_and_releases_lease(tmp_path, monkeypatch):
    from xrd_tools.core.scan import Scan
    from xrd_tools.reduction import NexusSink, ReductionPlan
    module = importlib.import_module("xrd_tools.reduction.core")
    target = tmp_path / "constructor.nexus"
    sink = NexusSink(target)
    monkeypatch.setattr(
        module,
        "NexusRecordWriter",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("constructor fault")),
    )
    with pytest.raises(ValueError, match="constructor fault"):
        sink.begin(Scan("fault", []), ReductionPlan())
    assert sink._transaction.snapshot().phase is TransactionPhase.ABORTED
    assert sink._transaction_owners is None
    assert not target.exists()


def test_bound_preflight_constructor_failure_reports_aborted_not_bound(
    tmp_path, monkeypatch,
):
    from xrd_tools.core.scan import Scan
    from xrd_tools.io import AppendPreflightState, prepare_append_preflight
    from xrd_tools.reduction import NexusSink, ReductionPlan
    module = importlib.import_module("xrd_tools.reduction.core")
    target = tmp_path / "preflight-constructor.nexus"
    preflight = prepare_append_preflight(
        target, _intent(tmp_path, extent=1, labels=(0,)),
    )
    sink = NexusSink(target, append_preflight=preflight)
    monkeypatch.setattr(
        module,
        "NexusRecordWriter",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("constructor fault")),
    )

    with pytest.raises(ValueError, match="constructor fault"):
        sink.begin(Scan("fault", []), ReductionPlan())
    assert preflight.snapshot.state is AppendPreflightState.ABORTED
    assert sink._transaction.snapshot().phase is TransactionPhase.ABORTED
    assert sink._transaction_owners is None
    assert not target.exists()


def test_sink_begin_stream_failure_does_not_leak_lease(tmp_path, monkeypatch):
    from xrd_tools.core.scan import Scan
    from xrd_tools.reduction import NexusSink, ReductionPlan
    transaction_module = importlib.import_module("xrd_tools.io.output_transaction")
    monkeypatch.setattr(
        transaction_module.OutputTransaction,
        "begin_stream",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("begin fault")),
    )
    sink = NexusSink(tmp_path / "begin_stream.nexus")
    with pytest.raises(OSError, match="begin fault"):
        sink.begin(Scan("fault", []), ReductionPlan())
    assert sink._transaction.snapshot().phase is TransactionPhase.ABORTED
    assert sink._transaction_owners is None


def test_nonappend_admission_and_lease_share_the_borrowed_file_lock(
    tmp_path, monkeypatch,
):
    from xrd_tools.core.scan import Scan
    from xrd_tools.reduction import NexusSink, ReductionPlan, ReductionResult
    coordinator = get_output_transaction_coordinator()
    real_admit = coordinator.admit

    class BorrowedLock:
        active = False

        def __enter__(self):
            assert not self.active
            self.active = True
            return self

        def __exit__(self, *_exc):
            self.active = False

    lock = BorrowedLock()

    def admit(*args, **kwargs):
        assert lock.active, "canonical admission escaped the borrowed file lock"
        return real_admit(*args, **kwargs)

    monkeypatch.setattr(coordinator, "admit", admit)
    sink = NexusSink(tmp_path / "locked-admission.nexus", file_lock=lock)
    sink.begin(Scan("locked", []), ReductionPlan())
    sink.abort(ReductionResult("locked", {}, 0, failed=True))
    assert sink._transaction.snapshot().phase is TransactionPhase.ABORTED
    assert sink._transaction_owners is None


def test_bound_preflight_begin_failure_settles_as_aborted(tmp_path, monkeypatch):
    from xrd_tools.core.scan import Scan
    from xrd_tools.io import AppendPreflightState, prepare_append_preflight
    from xrd_tools.reduction import NexusSink, ReductionPlan
    transaction_module = importlib.import_module("xrd_tools.io.output_transaction")
    target = tmp_path / "preflight-begin.nexus"
    preflight = prepare_append_preflight(
        target, _intent(tmp_path, extent=1, labels=(0,)),
    )
    monkeypatch.setattr(
        transaction_module.OutputTransaction,
        "begin_stream",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("begin fault")),
    )
    sink = NexusSink(target, append_preflight=preflight)

    with pytest.raises(OSError, match="begin fault"):
        sink.begin(Scan("fault", []), ReductionPlan())
    assert preflight.snapshot.state is AppendPreflightState.ABORTED
    assert sink._transaction.snapshot().phase is TransactionPhase.ABORTED
    assert sink._transaction_owners is None


def test_existing_target_requires_preflight_and_overwrite_rejects_one(tmp_path):
    from xrd_tools.core.scan import Scan
    from xrd_tools.io import prepare_append_preflight
    from xrd_tools.reduction import NexusSink, ReductionPlan

    existing = tmp_path / "existing.nexus"
    _seed_target(existing, (0,), tmp_path)
    preserved = existing.read_bytes()
    sink = NexusSink(
        existing,
        same_run_intent=_intent(tmp_path, extent=1, labels=(0,)),
    )
    with pytest.raises(ValueError, match="exact Append preflight"):
        sink.begin(Scan("existing", []), ReductionPlan())
    assert existing.read_bytes() == preserved
    assert sink._transaction_owners is None

    fresh = tmp_path / "fresh.nexus"
    preflight = prepare_append_preflight(
        fresh, _intent(tmp_path, extent=1, labels=(0,)))
    with pytest.raises(ValueError, match="Overwrite cannot consume"):
        NexusSink(fresh, overwrite=True, append_preflight=preflight)
    preflight.abort()


def test_sink_close_failure_never_restores_until_writer_closes(tmp_path, monkeypatch):
    from xrd_tools.core.scan import Scan
    from xrd_tools.reduction import NexusSink, ReductionPlan, ReductionResult
    target = tmp_path / "close-owner.nexus"
    sink = NexusSink(target)
    sink.begin(Scan("fault", []), ReductionPlan())
    writer = sink._writer
    real_close = writer._close_handle
    monkeypatch.setattr(
        writer,
        "_close_handle",
        lambda: (_ for _ in ()).throw(OSError("close fault")),
    )
    result = ReductionResult("fault", {}, 0, failed=True)
    with pytest.raises(Exception, match="close fault"):
        sink.abort(result)
    assert target.exists()
    assert sink._transaction.snapshot().phase is TransactionPhase.EXECUTING
    monkeypatch.setattr(writer, "_close_handle", real_close)
    sink.abort(result)
    assert sink._transaction.snapshot().phase is TransactionPhase.ABORTED
    assert not target.exists()


def _append_api():
    module = importlib.import_module("xrd_tools.io.append")
    return (
        module.AppendDisposition,
        module.AppendExternalMember,
        module.AppendIntent,
        module.AppendSource,
        module.commit_append_lineage,
        module.qualify_append,
    )


def _intent(tmp_path: Path, *, extent: int, labels, source_identity="beam/run",
            science="science-v1", source_base=None, member_path=None,
            generation=0, modes=("1d:default", "2d:default"),
            dataset_paths=()):
    (_Disposition, AppendExternalMember, AppendIntent, AppendSource,
     _commit, _qualify) = _append_api()
    member_path = member_path or (tmp_path / "member.h5")
    member = AppendExternalMember(
        path=str(member_path),
        dataset_path="/entry/data/data",
        size=100 + extent,
        mtime_ns=1_000 + extent,
        source_start=0,
        source_stop=extent,
        ordinal=0,
    )
    source = AppendSource(
        path=str(tmp_path / "master.h5"),
        adapter_id="nexus_hdf5",
        size=200 + extent,
        mtime_ns=2_000 + extent,
        extent=extent,
        dataset_paths=tuple(dataset_paths),
        external_members=(member,),
        generation=generation,
    )
    return AppendIntent(
        entry="entry",
        source_base=str(source_base or tmp_path),
        source_identity=source_identity,
        science_fingerprint=science,
        modes=modes,
        source=source,
        labels=tuple(int(x) for x in labels),
    )


def _eiger_append_source(
    root: Path,
    extents: tuple[int, ...],
    *,
    generation: int,
):
    import xrd_tools.sources.registry  # noqa: F401  (register built-in owners)
    from xrd_tools.core.scan import SourceKind, SourceSpec
    from xrd_tools.sources.execution_graph import (
        append_source_from_execution_graph,
        qualify_source_execution_graph,
    )

    root.mkdir(parents=True, exist_ok=True)
    members = []
    for ordinal, extent in enumerate(extents, 1):
        member = root / f"scan_data_{ordinal:06d}.h5"
        if not member.exists():
            with h5py.File(member, "w") as handle:
                handle.create_dataset(
                    "entry/data/data",
                    data=np.full((extent, 2, 3), ordinal, dtype="u2"),
                    chunks=(1, 2, 3),
                )
        members.append(member)
    master = root / "scan_master.h5"
    if not master.exists():
        with h5py.File(master, "w") as handle:
            entry = handle.create_group("entry")
            entry.attrs["NX_class"] = "NXentry"
            data = entry.create_group("data")
            data.attrs["NX_class"] = "NXdata"
    with h5py.File(master, "r+") as handle:
        data = handle["entry/data"]
        for ordinal, member in enumerate(members, 1):
            name = f"data_{ordinal:06d}"
            if name not in data:
                data[name] = h5py.ExternalLink(
                    member.name, "/entry/data/data"
                )
    graph = qualify_source_execution_graph(
        SourceSpec(master, SourceKind.EIGER_MASTER, entry="entry")
    )
    return append_source_from_execution_graph(
        graph, generation=generation
    )


def _intent_for_source(tmp_path: Path, source, labels):
    (_Disposition, _Member, AppendIntent, _Source,
     _commit, _qualify) = _append_api()
    return AppendIntent(
        entry="entry",
        source_base=str(tmp_path),
        source_identity="beam/run",
        science_fingerprint="science-v1",
        modes=("1d:default", "2d:default"),
        source=source,
        labels=tuple(labels),
    )


class _LiveFrames:
    def __init__(self, labels=()):
        self._values = {}
        self._persisted = set()
        for label in labels:
            self.add(label)

    @property
    def index(self):
        return sorted(self._values)

    def __getitem__(self, label):
        return self._values[int(label)]

    def add(self, label):
        from types import SimpleNamespace
        self._values[int(label)] = SimpleNamespace(
            idx=int(label), int_1d=_r1(label), int_2d=None,
            gi_1d={}, gi_2d={}, scan_info={}, map_raw=None, bg_raw=None,
            source_file="", source_frame_idx=0, mask=None, poni=None,
            thumbnail=None,
        )

    def mark_persisted(self, labels):
        self._persisted.update(int(label) for label in labels)


def _live_scan(target, intent, labels):
    from types import SimpleNamespace
    return SimpleNamespace(
        name="live", data_file=str(target), source_base=target.parent,
        file_lock=threading.RLock(), frames=_LiveFrames(labels),
        skip_2d=True, bai_1d_args={}, bai_2d_args={}, gi=False,
        gi_config={}, scan_data=None, geometry=None, global_mask=None,
        detector_shape=None, mg_args={"wavelength": 1e-10},
        _same_run_intent=intent,
    )


def _assert_lease_available(target):
    coordinator = get_output_transaction_coordinator()
    transaction_owner = OwnerToken("availability transaction")
    target_owner = OwnerToken("availability target")
    owners = {role: OwnerToken(f"availability {role.value}") for role in LeaseOwner}
    transaction = coordinator.admit(
        target, transaction_owner=transaction_owner, target_owner=target_owner)
    lease = transaction.acquire_lease(
        admission=transaction.admission,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        owners=owners,
    )
    transaction.abandon(lease)
    _release(transaction, lease, owners)


def _seed_target(path: Path, labels, source_base: Path) -> None:
    with h5py.File(path, "w") as handle:
        entry = current_entry(handle)
        entry.attrs[SOURCE_BASE_ATTR] = str(source_base)
        _write_result_rows(entry, labels)


def _write_result_rows(entry, labels) -> None:
    labels = tuple(labels)
    write_integrated_stack(
        entry, frame_indices=labels,
        results_1d=[_r1(label) for label in labels],
        results_2d=[_r2(label) for label in labels],
    )


def test_existing_stream_startup_full_hashes_prior_exactly_three_times(
    tmp_path, monkeypatch,
):
    module = importlib.import_module("xrd_tools.io.output_transaction")
    prior = b"immutable prior bytes"
    real_hash = module._sha256_handle
    prior_hashes = []

    def count_prior_hash(handle):
        observed = os.fstat(handle.fileno())
        if observed.st_size == len(prior):
            prior_hashes.append((observed.st_dev, observed.st_ino))
        return real_hash(handle)

    monkeypatch.setattr(module, "_sha256_handle", count_prior_hash)
    (_coordinator, transaction, target, transaction_owner, target_owner,
     owners, lease) = _transaction(tmp_path, prior=prior)
    pool = _Pool()
    attempt = transaction.begin_stream(
        admission=transaction.admission,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease,
        pool=pool,
        file_lock=threading.RLock(),
        seed_mode=module.StreamSeedMode.EMPTY_REPLACEMENT,
    )

    assert len(prior_hashes) == 3
    assert prior_hashes[0] == prior_hashes[1]
    assert prior_hashes[2] == prior_hashes[0]
    assert target.read_bytes() == b""
    assert transaction.backup.read_bytes() == prior
    assert transaction.abort_stream(
        attempt, lease=lease,
    ).phase is TransactionPhase.ABORTED
    assert target.read_bytes() == prior
    _release(transaction, lease, owners)


def test_preserved_stream_seed_hashes_during_copy_and_keeps_destination_readback(
    tmp_path, monkeypatch,
):
    module = importlib.import_module("xrd_tools.io.output_transaction")
    prior = b"P" * (2 * module._HASH_CHUNK_BYTES + 17)
    (_coordinator, transaction, target, transaction_owner, target_owner,
     owners, lease) = _transaction(tmp_path, prior=prior)
    prior_identity = (
        transaction.admission.snapshot.device,
        transaction.admission.snapshot.inode,
    )
    real_hash = module._sha256_handle
    real_read = module.os.read
    real_seal = module._descriptor_content_receipt
    hashed_objects = []
    copied_bytes = 0
    source_seals = []

    def observe_hash(handle):
        observed = os.fstat(handle.fileno())
        hashed_objects.append((
            (observed.st_dev, observed.st_ino),
            observed.st_size,
        ))
        return real_hash(handle)

    def observe_read(descriptor, size):
        nonlocal copied_bytes
        block = real_read(descriptor, size)
        observed = os.fstat(descriptor)
        if (observed.st_dev, observed.st_ino) == prior_identity:
            copied_bytes += len(block)
        return block

    def observe_seal(descriptor, path, role, **kwargs):
        if role == "stream-seed-source":
            source_seals.append((
                kwargs.get("evidence_digest"),
                kwargs.get("expected_stat"),
            ))
        return real_seal(descriptor, path, role, **kwargs)

    monkeypatch.setattr(module, "_sha256_handle", observe_hash)
    monkeypatch.setattr(module.os, "read", observe_read)
    monkeypatch.setattr(module, "_descriptor_content_receipt", observe_seal)
    attempt = transaction.begin_stream(
        admission=transaction.admission,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease,
        pool=_Pool(),
        file_lock=threading.RLock(),
    )

    destination_identity = (
        transaction._stream_reservation.identity.device,
        transaction._stream_reservation.identity.inode,
    )
    assert [identity for identity, size in hashed_objects if size == len(prior)] == [
        prior_identity,
        prior_identity,
        destination_identity,
    ]
    assert copied_bytes == len(prior)
    assert len(source_seals) == 1
    assert source_seals[0][0] == hashlib.sha256(prior).hexdigest()
    assert source_seals[0][1][:4] == (
        prior_identity[0],
        prior_identity[1],
        len(prior),
        transaction.admission.snapshot.mtime_ns,
    )
    assert type(source_seals[0][1][4]) is int
    assert transaction._stream_checkpoint.evidence_digest == hashlib.sha256(
        prior,
    ).hexdigest()
    assert target.read_bytes() == prior
    assert transaction.abort_stream(
        attempt, lease=lease,
    ).phase is TransactionPhase.ABORTED
    assert target.read_bytes() == prior
    _release(transaction, lease, owners)


def test_preserved_stream_seed_mutation_holds_then_retries_exact_rollback(
    tmp_path, monkeypatch,
):
    module = importlib.import_module("xrd_tools.io.output_transaction")
    prior = b"Q" * (2 * module._HASH_CHUNK_BYTES + 17)
    (_coordinator, transaction, target, transaction_owner, target_owner,
     owners, lease) = _transaction(tmp_path, prior=prior)
    pool = _Pool()
    prior_identity = (
        transaction.admission.snapshot.device,
        transaction.admission.snapshot.inode,
    )
    real_read = module.os.read
    mutated = False

    def mutate_after_first_copy_read(descriptor, size):
        nonlocal mutated
        block = real_read(descriptor, size)
        observed = os.fstat(descriptor)
        if (
            block
            and not mutated
            and (observed.st_dev, observed.st_ino) == prior_identity
        ):
            with transaction.backup.open("r+b") as handle:
                handle.seek(-1, os.SEEK_END)
                handle.write(b"R")
                handle.flush()
                os.fsync(handle.fileno())
            mutated = True
        return block

    monkeypatch.setattr(module.os, "read", mutate_after_first_copy_read)
    with pytest.raises(TargetChanged, match="stream-seed-source"):
        transaction.begin_stream(
            admission=transaction.admission,
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
            pool=pool,
            file_lock=threading.RLock(),
        )

    held = transaction.snapshot()
    assert mutated
    assert held.phase is TransactionPhase.ROLLBACK_PENDING
    assert RetryAction.ROLLBACK in held.pending_actions
    assert RetryAction.POOL_RESUME in held.pending_actions
    assert held.cleanup_token is not None
    assert not target.exists()
    assert transaction.backup.read_bytes().endswith(b"R")

    mutated_stat = transaction.backup.stat()
    with transaction.backup.open("r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        handle.write(b"Q")
        handle.flush()
        os.fsync(handle.fileno())
    os.utime(
        transaction.backup,
        ns=(
            mutated_stat.st_atime_ns,
            transaction.admission.snapshot.mtime_ns,
        ),
    )
    recovered = transaction.retry_cleanup(held.cleanup_token)
    assert recovered.phase is TransactionPhase.READY_TO_RETRY
    assert recovered.pending_actions == ()
    assert target.read_bytes() == prior
    assert not transaction.backup.exists()
    assert transaction.abandon(lease).phase is TransactionPhase.ABORTED
    _release(transaction, lease, owners)


def test_existing_replacement_uses_transaction_admission_without_recapture(
    tmp_path, monkeypatch,
):
    from xrd_tools.core.scan import Scan
    from xrd_tools.io.output_transaction import capture_target_snapshot
    from xrd_tools.reduction import NexusSink, ReductionPlan

    core = importlib.import_module("xrd_tools.reduction.core")
    target = tmp_path / "receipt-qualified.nexus"
    _seed_existing_append_target(
        target,
        tmp_path,
        _intent(
            tmp_path,
            extent=1,
            labels=(0,),
            modes=("1d:default",),
        ),
    )
    expected = capture_target_snapshot(target)
    before = target.read_bytes()
    sink = NexusSink.for_existing_replacement(
        target,
        expected_target_snapshot=expected,
        dimension="1d",
        labels=(0,),
        audit_bytes=b"{}",
        selected_plan={},
        selected_gi_mode=None,
        source_execution={},
        append_lineage=None,
        source_base=tmp_path,
        file_lock=threading.RLock(),
        flush_every=None,
    )

    monkeypatch.setattr(
        core,
        "capture_target_snapshot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("replacement admission recaptured the full target")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        core,
        "NexusRecordWriter",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("stop after receipt qualification")
        ),
    )
    with pytest.raises(RuntimeError, match="receipt qualification"):
        sink.begin(Scan("receipt", []), ReductionPlan(integration_2d=None))

    assert sink._transaction.admission.snapshot == expected
    assert sink._transaction.snapshot().phase is TransactionPhase.ABORTED
    assert target.read_bytes() == before
    _assert_lease_available(target)


@pytest.mark.parametrize(
    ("provenance", "overwrite", "fast"),
    (
        ({"output_mode": "Overwrite", "live_mode": False}, True, True),
        ({"output_mode": "Overwrite", "live_mode": True}, True, False),
        ({"output_mode": "Append", "live_mode": False}, True, False),
        ({"output_mode": "Overwrite"}, True, False),
        ({"output_mode": "Overwrite", "live_mode": "false"}, True, False),
        ({"output_mode": "Overwrite", "live_mode": False}, False, False),
    ),
)
def test_frozen_provenance_routes_only_exact_overwrite_to_fast_regenerable(
    tmp_path, monkeypatch, provenance, overwrite, fast,
):
    from xrd_tools.core.scan import Scan
    from xrd_tools.reduction import NexusSink, ReductionPlan

    core = importlib.import_module("xrd_tools.reduction.core")
    transaction_module = importlib.import_module("xrd_tools.io.output_transaction")
    target = tmp_path / "route.nexus"
    from tests.core._processed_fixture import write_recognized_result

    # A RECOGNIZED prior: an ordinary Replace refuses an unrecognized
    # occupant since OWNER-GATE-RAW-TARGET-20260905, and arbitrary bytes
    # are exactly that. The subject here is backup/restore, not content.
    prior = write_recognized_result(target)
    real_hash = transaction_module._sha256_handle
    hashes = []

    def observe_hash(handle):
        hashes.append(os.fstat(handle.fileno()).st_size)
        return real_hash(handle)

    def reject_writer(*_args, **_kwargs):
        raise RuntimeError("stop after transaction routing")

    monkeypatch.setattr(transaction_module, "_sha256_handle", observe_hash)
    monkeypatch.setattr(core, "NexusRecordWriter", reject_writer)
    sink = NexusSink(
        target,
        overwrite=overwrite,
        run_configuration_provenance=provenance,
    )
    assert sink._fast_regenerable is fast
    if not overwrite:
        return
    with pytest.raises(RuntimeError, match="transaction routing"):
        sink.begin(Scan("route", []), ReductionPlan(integration_2d=None))

    assert (hashes == []) is fast
    assert target.read_bytes() == prior
    assert not sink._transaction.backup.exists()
    assert sink._transaction.snapshot().phase is TransactionPhase.ABORTED
    _assert_lease_available(target)

@pytest.mark.parametrize("fast", (1, np.bool_(True)))
def test_fast_regenerable_capability_requires_an_exact_bool(tmp_path, fast):
    from xrd_tools.io.record_writer import NexusRecordWriter

    with pytest.raises(TypeError, match="exact bool"):
        NexusRecordWriter(tmp_path / "unbound.nexus", fast_regenerable=fast)


def test_fast_regenerable_capability_cannot_be_used_unbound(tmp_path):
    from xrd_tools.io.record_writer import NexusRecordWriter

    with pytest.raises(ValueError, match="transaction-bound Overwrite"):
        NexusRecordWriter(
            tmp_path / "unbound.nexus",
            overwrite=True,
            fast_regenerable=True,
        )


def test_fast_regenerable_writer_requires_matching_transaction_capability(
    tmp_path,
):
    from xrd_tools.io.record_writer import (
        NexusRecordWriter,
        WriterTransactionBinding,
    )

    (
        _coordinator,
        transaction,
        target,
        _transaction_owner,
        _target_owner,
        owners,
        lease,
    ) = _transaction(tmp_path)
    try:
        with pytest.raises(ValueError, match="transaction-bound Overwrite"):
            NexusRecordWriter(
                target,
                overwrite=True,
                atomic=False,
                fast_regenerable=True,
                transaction_binding=WriterTransactionBinding(
                    transaction, object(), lease,
                ),
            )
    finally:
        transaction.abandon(lease)
        _release(transaction, lease, owners)


def test_fast_regenerable_finish_skips_checkpoint_payload_scans_but_verifies_science_at_close(
    tmp_path, monkeypatch,
):
    from xrd_tools.core.scan import ScanFrame
    from xrd_tools.reduction import (
        FrameReduction,
        NexusTerminalDisposition,
        ReductionResult,
    )

    transaction_module = importlib.import_module("xrd_tools.io.output_transaction")
    target = tmp_path / "fast-finish.nexus"
    from tests.core._processed_fixture import write_recognized_result

    # A RECOGNIZED prior: an ordinary Replace refuses an unrecognized
    # occupant since OWNER-GATE-RAW-TARGET-20260905, and arbitrary bytes
    # are exactly that. The subject here is backup/restore, not content.
    write_recognized_result(target)
    real_fsync = transaction_module.os.fsync
    fsync_calls = []

    def forbidden_hash(_handle):
        raise AssertionError("fast regenerable output performed a whole-file hash")

    def observe_fsync(descriptor):
        fsync_calls.append(os.fstat(descriptor).st_ino)
        return real_fsync(descriptor)

    monkeypatch.setattr(transaction_module, "_sha256_handle", forbidden_hash)
    monkeypatch.setattr(transaction_module.os, "fsync", observe_fsync)
    target, sink, facade = _begin_finite_overwrite_sink(
        tmp_path,
        "fast-finish.nexus",
        "fast",
        bind_session=True,
    )
    recoveries = []
    facade.commit_checkpoint_recoverable = (
        lambda *values: recoveries.append(values)
    )
    writer = sink._writer

    monkeypatch.setattr(
        writer,
        "_verify_dirty_evidence",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("fast checkpoint reread persisted rows")
        ),
    )
    sink.write(ScanFrame(0), FrameReduction(0, result_1d=_r1(3)))
    terminal = sink.finish(ReductionResult("fast", {}, 1))

    assert terminal.disposition is NexusTerminalDisposition.COMMITTED
    assert len(facade.durable) == 1
    assert writer.checkpoint_read_volume == (0, 0)
    assert writer.grouped_semantic_read_volume["checkpoint"] == (0, 0)
    assert writer.grouped_semantic_read_volume["close"] == (1, 1)
    assert writer._durable_frame_proofs == {}
    assert writer._dirty_frames == {}
    assert writer._dirty_indexed == {}
    assert writer._dirty_absent_modes == set()
    # The fast path publishes checkpoint RECOVERABILITY ("written and re-readable
    # from the artifact") but still no DURABILITY: len(facade.durable) == 1 above
    # proves the single durable publication happens at close, not per checkpoint.
    # Without the recoverable projection the display store may never release a
    # heavy payload under a live projection and memory grows with frame count.
    # Every cost assertion above still holds: no checkpoint reads, no payload
    # scans, no whole-file hash, no _verify_dirty_evidence.
    assert len(recoveries) == 1
    _checkpoint, receipts, drops, frame_labels, thumbnails = recoveries[0]
    assert tuple(frame_labels) == (0,)
    assert tuple(drops) == () and tuple(thumbnails) == ()
    assert tuple(int(receipt.label) for receipt in receipts) == (0,)
    assert all(
        proof.row_count == 1
        and len(proof.sigma_presence) == 1
        and not isinstance(proof.radial, np.ndarray)
        and not isinstance(proof.azimuthal, np.ndarray)
        for proof in writer._fast_mode_groups.values()
    )
    assert fsync_calls
    assert not sink._transaction.backup.exists()
    with h5py.File(target, "r") as handle:
        group = handle["entry/integrated_1d"]
        assert group["frame_index"].shape == (1,)
        assert group["intensity"].shape == (1, 8)
        np.testing.assert_array_equal(group["intensity"][0], _r1(3).intensity)


def test_fast_regenerable_finite_overwrite_carries_product_same_run_lineage(
    tmp_path,
):
    """The GUI's ordinary Run always binds a same-run intent.

    Regression: the fresh finite Overwrite fast path is exactly the route the
    non-Live, non-Append GUI Run takes, and that route always constructs the
    sink with ``same_run_intent``.  A guard that rejects any non-None append
    decision therefore fails every ordinary Run before its first frame.
    """
    from xrd_tools.core.scan import ScanFrame
    from xrd_tools.reduction import (
        FrameReduction,
        NexusTerminalDisposition,
        ReductionResult,
    )

    intent = _intent(tmp_path, extent=1, labels=(0,))
    target, sink, _facade = _begin_finite_overwrite_sink(
        tmp_path,
        "fast-same-run.nexus",
        "fast-same-run",
        same_run_intent=intent,
    )
    writer = sink._writer

    assert writer._fast_regenerable is True
    assert writer._append_decision is not None
    assert sink._transaction.fast_regenerable is True

    sink.write(ScanFrame(0), FrameReduction(0, result_1d=_r1(5)))
    terminal = sink.finish(ReductionResult("fast-same-run", {}, 1))

    assert terminal.disposition is NexusTerminalDisposition.COMMITTED
    with h5py.File(target, "r") as handle:
        group = handle["entry/integrated_1d"]
        assert group["frame_index"].shape == (1,)
        np.testing.assert_array_equal(group["intensity"][0], _r1(5).intensity)
        lineage = json.loads(
            handle["entry/reduction/config/append_lineage"][()]
        )
    assert lineage["epochs"][-1]["labels"] == [0]


@pytest.mark.parametrize(
    "changed",
    (
        "frame-index",
        "q",
        "q-unit",
        "intensity",
        "sigma",
        "chi",
        "chi-unit",
        "two-d-kind",
        "axis-kind",
        "mode-inventory",
        "stack-shape-after-checkpoint",
    ),
)
def test_fast_regenerable_close_rejects_changed_scientific_value(
    tmp_path, monkeypatch, changed,
):
    from xrd_tools.core.scan import ScanFrame
    from xrd_tools.io.record_writer import WriterIncomplete
    from xrd_tools.reduction import (
        FrameReduction,
        ReductionResult,
    )

    target, sink, facade = _begin_finite_overwrite_sink(
        tmp_path,
        "fast-science.nexus",
        "science",
        bind_session=True,
    )
    sink.write(
        ScanFrame(0),
        FrameReduction(0, result_1d=_r1(5), result_2d=_r2(5)),
    )
    writer = sink._writer
    one_d = writer._h5["entry/integrated_1d"]
    two_d = writer._h5["entry/integrated_2d"]
    if changed == "frame-index":
        one_d["frame_index"][0] = np.int64(9)
    elif changed == "q":
        one_d["axis_1"][0] += np.float32(1)
    elif changed == "q-unit":
        one_d["axis_1"].attrs["units"] = "changed"
    elif changed == "intensity":
        one_d["intensity"][0, 0] += np.float32(1)
    elif changed == "sigma":
        two_d["sigma"][0, 0, 0] += np.float32(1)
    elif changed == "chi":
        two_d["axis_2"][0] += np.float32(1)
    elif changed == "chi-unit":
        two_d["axis_2"].attrs["units"] = "changed"
    elif changed == "two-d-kind":
        two_d.attrs["two_d_kind"] = "changed"
    elif changed == "axis-kind":
        one_d.attrs["axis_kind"] = "azimuthal"
    elif changed == "stack-shape-after-checkpoint":
        real_close = writer._close_handle

        def break_shape_after_checkpoint():
            intensity = writer._h5["entry/integrated_1d/intensity"]
            intensity.resize((0, intensity.shape[1]))
            writer._h5.flush()
            return real_close()

        monkeypatch.setattr(writer, "_close_handle", break_shape_after_checkpoint)
    else:
        one_d.attrs[PRIMARY_MODE_ATTR] = "chi_q"
    with pytest.raises(
        WriterIncomplete,
        match="fast close|processed input|row|shape|durability",
    ):
        sink.finish(ReductionResult("science", {}, 1))
    phase = sink._transaction.snapshot().phase
    assert phase is (
        TransactionPhase.EXECUTING
        if changed == "mode-inventory"
        else TransactionPhase.INTEGRITY_HOLD
    )
    assert facade.durable == []
    sink.abort(None)
    assert writer.phase.value == "aborted"
    assert sink._transaction.snapshot().phase is TransactionPhase.ABORTED
    assert not target.exists()
    assert facade.durable == []


def test_fast_science_failure_abort_restores_exact_prior_target(tmp_path):
    from xrd_tools.core.scan import ScanFrame
    from xrd_tools.io.record_writer import WriterIncomplete
    from xrd_tools.reduction import FrameReduction, ReductionResult

    target = tmp_path / "fast-prior.nexus"
    from tests.core._processed_fixture import write_recognized_result

    # A RECOGNIZED prior: an ordinary Replace refuses an unrecognized
    # occupant since OWNER-GATE-RAW-TARGET-20260905, and arbitrary bytes
    # are exactly that. The subject here is backup/restore, not content.
    prior = write_recognized_result(target)
    target, sink, facade = _begin_finite_overwrite_sink(
        tmp_path,
        target.name,
        "prior",
        bind_session=True,
    )
    sink.write(ScanFrame(0), FrameReduction(0, result_1d=_r1(5)))
    writer = sink._writer
    writer._h5["entry/integrated_1d/intensity"][0, 0] += np.float32(1)

    with pytest.raises(WriterIncomplete, match="fast close"):
        sink.finish(ReductionResult("prior", {}, 1))
    sink.abort(None)

    assert writer.phase.value == "aborted"
    assert sink._transaction.snapshot().phase is TransactionPhase.ABORTED
    assert target.read_bytes() == prior
    assert facade.durable == []
    _assert_lease_available(target)


def test_fast_terminal_science_accepts_allclose_axes_and_mixed_sigma(tmp_path):
    from xrd_tools.core.scan import ScanFrame
    from xrd_tools.reduction import (
        FrameReduction,
        NexusTerminalDisposition,
        ReductionResult,
    )

    target, sink, _facade = _begin_finite_overwrite_sink(
        tmp_path,
        "fast-both.nexus",
        "both",
    )
    for label in range(3):
        base_1d = _r1(label + 1)
        base_2d = _r2(label + 1)
        delta = np.float64(label) * 1e-7
        result_1d = IntegrationResult1D(
            radial=base_1d.radial + delta,
            intensity=base_1d.intensity,
            sigma=base_1d.sigma if label == 1 else None,
            unit=base_1d.unit,
        )
        result_2d = IntegrationResult2D(
            radial=base_2d.radial + delta,
            azimuthal=base_2d.azimuthal + delta,
            intensity=base_2d.intensity,
            sigma=base_2d.sigma if label == 1 else None,
            unit=base_2d.unit,
            azimuthal_unit=base_2d.azimuthal_unit,
        )
        sink.write(
            ScanFrame(label),
            FrameReduction(
                label,
                result_1d=result_1d,
                result_2d=result_2d,
            ),
        )

    terminal = sink.finish(ReductionResult("both", {}, 3))
    writer = sink._writer
    assert terminal.disposition is NexusTerminalDisposition.COMMITTED
    assert writer.grouped_semantic_read_volume["close"] == (2, 6)
    with h5py.File(target, "r") as handle:
        one_d = handle["entry/integrated_1d"]
        two_d = handle["entry/integrated_2d"]
        np.testing.assert_array_equal(
            one_d["axis_1"][()], np.asarray(_r1(1).radial, np.float32),
        )
        np.testing.assert_array_equal(
            two_d["axis_1"][()], np.asarray(_r2(1).radial, np.float32),
        )
        assert np.isnan(one_d["sigma"][[0, 2]]).all()
        assert np.isnan(two_d["sigma"][[0, 2]]).all()


def test_fast_later_batch_rejects_divergent_axis_before_mutation(tmp_path):
    from xrd_tools.core.scan import ScanFrame
    from xrd_tools.reduction import FrameReduction, ReductionResult

    _target, sink, _facade = _begin_finite_overwrite_sink(
        tmp_path,
        "fast-axis-authority.nexus",
        "axis-authority",
    )
    first = _r1(1)
    sink.write(ScanFrame(0), FrameReduction(0, result_1d=first))
    changed = IntegrationResult1D(
        radial=first.radial + 1.0,
        intensity=first.intensity,
        sigma=first.sigma,
        unit=first.unit,
    )

    with pytest.raises(ValueError, match="axis/unit or row shape"):
        sink.write(ScanFrame(1), FrameReduction(1, result_1d=changed))

    writer = sink._writer
    assert writer._row_cursors["integrated_1d"] == {0: 0}
    assert writer._h5["entry/integrated_1d/intensity"].shape == (1, 8)
    with pytest.warns(RuntimeWarning, match="preserved non-final data"):
        sink.abort(ReductionResult("axis-authority", {}, 1, failed=True))


def test_fast_regenerable_rejects_accidental_repeat_before_mutation(tmp_path):
    from xrd_tools.core.scan import ScanFrame
    from xrd_tools.io.record_writer import WriterStateError
    from xrd_tools.reduction import FrameReduction

    _target, sink, _facade = _begin_finite_overwrite_sink(
        tmp_path,
        "fast-repeat.nexus",
        "repeat",
    )
    frame = ScanFrame(0, metadata={"timestamp": "first"})
    sink.write(frame, FrameReduction(0, result_1d=_r1(1)))
    writer = sink._writer
    before = np.asarray(
        writer._h5["entry/integrated_1d/intensity"][0]
    ).copy()

    with pytest.raises(WriterStateError, match="already owns frame label 0"):
        sink.write(
            ScanFrame(0, metadata={"timestamp": "second"}),
            FrameReduction(0, result_1d=_r1(2)),
        )

    np.testing.assert_array_equal(
        writer._h5["entry/integrated_1d/intensity"][0], before,
    )
    assert writer._h5["entry/frames/frame_0000/timestamp"].asstr()[()] == "first"
    with pytest.warns(RuntimeWarning, match="preserved non-final data"):
        sink.abort(None)


@pytest.mark.parametrize("corrupt", (False, True))
def test_fast_average_counts_are_authenticated_once_at_terminal_close(
    tmp_path, monkeypatch, corrupt,
):
    from tests.core.test_h23_record_writer import _average_counts
    from xrd_tools.core.scan import ScanFrame
    from xrd_tools.io.record_writer import WriterIncomplete
    from xrd_tools.reduction import (
        FrameReduction,
        NexusTerminalDisposition,
        ReductionResult,
    )

    counts = _average_counts(np.array([[3, 2, 0], [1, 3, 2]]), 3)
    _target, sink, facade = _begin_finite_overwrite_sink(
        tmp_path,
        f"fast-average-{corrupt}.nexus",
        "avg",
        extra={"average_finite_counts": counts},
        bind_session=True,
    )
    sink.write(ScanFrame(1), FrameReduction(1, result_1d=_r1(2)))
    writer = sink._writer
    if corrupt:
        real_flush = writer._flush_handle

        def corrupt_after_flush():
            real_flush()
            writer._h5[
                "entry/frames/frame_0001/finite_counts"
            ][0, 0] = np.uint32(1)

        monkeypatch.setattr(writer, "_flush_handle", corrupt_after_flush)
        with pytest.raises(WriterIncomplete, match="Average count"):
            sink.finish(ReductionResult("avg", {}, 1))
        assert sink._transaction.snapshot().phase is TransactionPhase.INTEGRITY_HOLD
        assert facade.durable == []
        return

    terminal = sink.finish(ReductionResult("avg", {}, 1))
    assert terminal.disposition is NexusTerminalDisposition.COMMITTED
    assert writer.checkpoint_read_volume == (0, 0)
    assert writer.grouped_semantic_read_volume["checkpoint"] == (0, 0)


def test_fast_terminal_science_reads_result_rows_in_bounded_exact_slabs(
    tmp_path, monkeypatch,
):
    from xrd_tools.core.scan import ScanFrame
    from xrd_tools.reduction import (
        FrameReduction,
        ReductionResult,
    )

    _target, sink, _facade = _begin_finite_overwrite_sink(
        tmp_path,
        "fast-close-slabs.nexus",
        "slice",
    )
    for label in range(10):
        sink.write(
            ScanFrame(label),
            FrameReduction(
                label,
                result_1d=_r1(label + 1),
                result_2d=_r2(label + 1),
            ),
        )
    writer = sink._writer
    reads = {}
    real_getitem = h5py.Dataset.__getitem__

    def trace(dataset, item):
        if (
            writer._pending_owner == "close"
            and "/integrated_" in dataset.name
            and dataset.name.rsplit("/", 1)[-1]
            in {"frame_index", "intensity", "sigma", "axis_1", "axis_2"}
        ):
            reads.setdefault(dataset.name, []).append(item)
        return real_getitem(dataset, item)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", trace)
    sink.finish(ReductionResult("slice", {}, 10))

    slabs = [slice(0, 8), slice(8, 10)]
    for name in ("integrated_1d", "integrated_2d"):
        root = f"/entry/{name}"
        assert reads[root + "/axis_1"] == [()]
        for leaf in ("frame_index", "intensity", "sigma"):
            assert reads[root + "/" + leaf] == slabs
    assert reads["/entry/integrated_2d/axis_2"] == [()]
    assert writer.grouped_semantic_read_volume["close"] == (4, 20)


@pytest.mark.parametrize("changed", ("unexpected-sigma", "unexpected-dimension"))
def test_fast_terminal_science_rejects_unrequested_result_storage(
    tmp_path, changed,
):
    from xrd_tools.core.scan import ScanFrame
    from xrd_tools.io.record_writer import WriterIncomplete
    from xrd_tools.reduction import (
        FrameReduction,
        ReductionResult,
    )

    _target, sink, _facade = _begin_finite_overwrite_sink(
        tmp_path,
        "fast-unexpected-sigma.nexus",
        "unexpected-sigma",
    )
    result = _r1(2)
    result = IntegrationResult1D(
        radial=result.radial,
        intensity=result.intensity,
        sigma=None,
        unit=result.unit,
    )
    sink.write(ScanFrame(0), FrameReduction(0, result_1d=result))
    one_d = sink._writer._h5["entry/integrated_1d"]
    if changed == "unexpected-sigma":
        one_d.create_dataset(
            "sigma",
            data=np.full((1, result.radial.size), np.nan, dtype=np.float32),
            maxshape=(None, result.radial.size),
        )
    else:
        from xrd_tools.io.nexus import write_integrated_stack

        write_integrated_stack(
            sink._writer._h5["entry"],
            frame_indices=[0],
            results_2d=[_r2(2)],
        )
    with pytest.raises(WriterIncomplete, match="fast close"):
        sink.finish(ReductionResult("unrequested", {}, 1))
    assert sink._transaction.snapshot().phase is TransactionPhase.INTEGRITY_HOLD


@pytest.mark.parametrize("remove_extra_mode", (False, True))
def test_fast_terminal_science_authenticates_named_mode_inventory(
    tmp_path, remove_extra_mode,
):
    from xrd_tools.core.scan import ScanFrame
    from xrd_tools.io.record_writer import WriterIncomplete
    from xrd_tools.reduction import (
        FrameReduction,
        GIMode,
        NexusTerminalDisposition,
        ReductionPlan,
        ReductionResult,
    )

    target, sink, _facade = _begin_finite_overwrite_sink(
        tmp_path,
        "fast-named-modes.nexus",
        "named",
        plan=ReductionPlan(
            gi=GIMode(mode_1d="q_total", mode_2d="qip_qoop")
        ),
    )
    frame = ScanFrame(0)
    primary_2d = _r2(1)
    sink.write(
        frame,
        FrameReduction(
            0,
            result_1d=_r1(1),
            result_2d=IntegrationResult2D(
                radial=primary_2d.radial,
                azimuthal=primary_2d.azimuthal,
                intensity=primary_2d.intensity,
                sigma=primary_2d.sigma,
                unit="qip_A^-1",
                azimuthal_unit="qoop_A^-1",
            ),
            mode_1d="q_total",
            mode_2d="qip_qoop",
        ),
    )
    q_ip = _r1(2)
    sink.write(
        frame,
        FrameReduction(
            0,
            result_1d=IntegrationResult1D(
                radial=q_ip.radial,
                intensity=q_ip.intensity,
                sigma=q_ip.sigma,
                unit="qip_A^-1",
            ),
            result_2d=_r2(2),
            mode_1d="q_ip",
            mode_2d="q_chi",
        ),
    )
    if remove_extra_mode:
        one_d = sink._writer._h5["entry/integrated_1d"]
        del one_d["q_ip"]
        one_d.attrs[MULTI_RESULT_MODES_ATTR] = ["q_total"]
        if "q_ip" in sink._writer._row_cursors:
            raise AssertionError("mode cursor must use its full group path")
        with pytest.raises(WriterIncomplete, match="fast close"):
            sink.finish(ReductionResult("named", {}, 1))
        assert sink._transaction.snapshot().phase is TransactionPhase.INTEGRITY_HOLD
        return

    terminal = sink.finish(ReductionResult("named", {}, 1))
    assert terminal.disposition is NexusTerminalDisposition.COMMITTED
    assert sink._writer.grouped_semantic_read_volume["close"] == (4, 4)
    with h5py.File(target, "r") as handle:
        np.testing.assert_array_equal(
            handle["entry/integrated_1d"].attrs[MULTI_RESULT_MODES_ATTR],
            ["q_total", "q_ip"],
        )
        np.testing.assert_array_equal(
            handle["entry/integrated_2d"].attrs[MULTI_RESULT_MODES_ATTR],
            ["qip_qoop", "q_chi"],
        )


def test_same_stat_mutation_during_pool_pause_refuses_before_target_move(
    tmp_path,
):
    module = importlib.import_module("xrd_tools.io.output_transaction")
    prior = b"original"
    mutated = b"mutated!"
    (_coordinator, transaction, target, transaction_owner, target_owner,
     owners, lease) = _transaction(tmp_path, prior=prior)
    admitted = target.stat()

    class MutatingPool(_Pool):
        def pause(self, path):
            super().pause(path)
            target.write_bytes(mutated)
            os.utime(
                target,
                ns=(admitted.st_atime_ns, admitted.st_mtime_ns),
            )

    pool = MutatingPool()
    with pytest.raises(TargetChanged, match="changed after admission"):
        transaction.begin_stream(
            admission=transaction.admission,
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
            pool=pool,
            file_lock=threading.RLock(),
            seed_mode=module.StreamSeedMode.EMPTY_REPLACEMENT,
        )

    assert target.read_bytes() == mutated
    assert not transaction.backup.exists()
    assert pool.events == [("pause", str(target)), ("resume", str(target))]


def test_existing_overwrite_routes_empty_seed_without_prior_copy_and_abort_restores(
    tmp_path, monkeypatch,
):
    from xrd_tools.core.scan import Scan
    from xrd_tools.reduction import NexusSink, ReductionPlan

    core = importlib.import_module("xrd_tools.reduction.core")
    target = tmp_path / "overwrite-empty-seed.nexus"
    from tests.core._processed_fixture import write_recognized_result

    # A RECOGNIZED prior: an ordinary Replace refuses an unrecognized
    # occupant since OWNER-GATE-RAW-TARGET-20260905, and arbitrary bytes
    # are exactly that. The subject here is backup/restore, not content.
    prior = write_recognized_result(target)
    sink = NexusSink(target, overwrite=True)
    observed = []

    def reject_writer(*_args, **_kwargs):
        observed.append(
            (target.read_bytes(), Path(sink._transaction.backup).read_bytes())
        )
        raise RuntimeError("observe empty overwrite seed")

    monkeypatch.setattr(core, "NexusRecordWriter", reject_writer)
    with pytest.raises(RuntimeError, match="observe empty overwrite seed"):
        sink.begin(Scan("overwrite", []), ReductionPlan())

    assert observed == [(b"", prior)]
    assert target.read_bytes() == prior
    assert sink._transaction.snapshot().phase is TransactionPhase.ABORTED
    assert not sink._transaction.backup.exists()
    _assert_lease_available(target)


def test_fresh_overwrite_empty_seed_has_no_backup_and_typed_zero_checkpoint(
    tmp_path,
):
    module = importlib.import_module("xrd_tools.io.output_transaction")
    StreamSeedMode = module.StreamSeedMode
    (_coordinator, transaction, target, transaction_owner, target_owner,
     owners, lease) = _transaction(tmp_path, prior=None)
    pool = _Pool()

    with pytest.raises(TypeError, match="seed_mode must be a StreamSeedMode"):
        transaction.begin_stream(
            admission=transaction.admission,
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
            pool=pool,
            file_lock=threading.RLock(),
            seed_mode=StreamSeedMode.EMPTY_REPLACEMENT.value,
        )

    attempt = transaction.begin_stream(
        admission=transaction.admission,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease,
        pool=pool,
        file_lock=threading.RLock(),
        seed_mode=StreamSeedMode.EMPTY_REPLACEMENT,
    )
    checkpoint = transaction._stream_checkpoint
    assert type(checkpoint) is module._StreamStatReceipt
    assert checkpoint.size == 0
    assert checkpoint.evidence_bytes == 0
    assert checkpoint.evidence_digest == hashlib.sha256(b"").hexdigest()
    assert target.read_bytes() == b""
    assert not transaction.backup.exists()

    assert transaction.abort_stream(
        attempt, lease=lease,
    ).phase is TransactionPhase.ABORTED
    assert not target.exists()
    assert pool.events == [("pause", str(target)), ("resume", str(target))]
    _release(transaction, lease, owners)


def test_existing_append_routes_preserve_seed(tmp_path, monkeypatch):
    from xrd_tools.core.scan import Scan, ScanFrame
    from xrd_tools.io import prepare_append_preflight
    from xrd_tools.reduction import (
        FrameReduction,
        NexusSink,
        ReductionPlan,
        ReductionResult,
    )

    target = tmp_path / "append-preserve-seed.nexus"
    first = _intent(
        tmp_path, extent=1, labels=(0,), generation=0,
        modes=("1d:default",),
    )
    initial = NexusSink(
        target,
        overwrite=True,
        source_base=tmp_path,
        same_run_intent=first,
        flush_every=None,
    )
    initial.begin(Scan("append-preserve", []), ReductionPlan(integration_2d=None))
    initial.write(ScanFrame(0), FrameReduction(0, result_1d=_r1(0)))
    initial.finish(ReductionResult("append-preserve", {}, 1))
    prior = target.read_bytes()

    second = _intent(
        tmp_path, extent=2, labels=(0, 1), generation=0,
        modes=("1d:default",),
    )
    preflight = prepare_append_preflight(target, second)
    resumed = NexusSink(
        target,
        source_base=tmp_path,
        append_preflight=preflight,
        flush_every=None,
    )
    core = importlib.import_module("xrd_tools.reduction.core")
    transaction_module = importlib.import_module("xrd_tools.io.output_transaction")
    real_begin = transaction_module.OutputTransaction.begin_stream
    policies = []
    observed = []

    def trace_begin(self, *args, **kwargs):
        policies.append(kwargs.get("seed_mode"))
        return real_begin(self, *args, **kwargs)

    def reject_writer(*_args, **_kwargs):
        observed.append(target.read_bytes())
        raise RuntimeError("observe preserved append seed")

    monkeypatch.setattr(
        transaction_module.OutputTransaction, "begin_stream", trace_begin,
    )
    monkeypatch.setattr(core, "NexusRecordWriter", reject_writer)
    with pytest.raises(RuntimeError, match="observe preserved append seed"):
        resumed.begin(
            Scan("append-preserve", []), ReductionPlan(integration_2d=None),
        )

    assert policies == [transaction_module.StreamSeedMode.PRESERVE_BASE]
    assert observed == [prior]
    assert target.read_bytes() == prior
    assert preflight.snapshot.state.value == "aborted"
    _assert_lease_available(target)


def test_science_fingerprint_rejects_unsupported_values_deterministically():
    module = importlib.import_module("xrd_tools.io.append")

    class Unsupported:
        pass

    with pytest.raises(TypeError, match="unsupported science fingerprint value"):
        module.science_fingerprint({"value": Unsupported()})


def test_append_writer_lineage_n_to_m_to_k_and_final_noop(tmp_path):
    (Disposition, _Member, _Intent, _Source, commit_lineage,
     qualify) = _append_api()
    target = tmp_path / "grow.nexus"
    first = _intent(tmp_path, extent=2, labels=(0, 1))
    decision = qualify(target, first)
    assert decision.disposition is Disposition.WRITE
    _seed_target(target, (0, 1), tmp_path)
    with h5py.File(target, "r+") as handle:
        commit_lineage(handle["entry"], decision, written_labels=(0, 1))

    second = _intent(tmp_path, extent=5, labels=(0, 1, 2, 3, 4))
    decision = qualify(target, second)
    assert decision.disposition is Disposition.WRITE
    with h5py.File(target, "r+") as handle:
        _write_result_rows(handle["entry"], (2, 3, 4))
        commit_lineage(handle["entry"], decision, written_labels=(2, 3, 4))

    third = _intent(tmp_path, extent=6, labels=tuple(range(6)))
    decision = qualify(target, third)
    assert decision.disposition is Disposition.WRITE
    with h5py.File(target, "r+") as handle:
        _write_result_rows(handle["entry"], (5,))
        commit_lineage(handle["entry"], decision, written_labels=(5,))

    final = qualify(target, _intent(tmp_path, extent=6, labels=tuple(range(6))))
    assert final.disposition is Disposition.SKIP


def test_legacy_empty_dataset_paths_refuse_same_extent_eiger_skip(
    tmp_path,
):
    from dataclasses import replace

    (Disposition, _Member, _Intent, _Source, commit_lineage,
     qualify) = _append_api()
    target = tmp_path / "legacy-eiger-same-extent.nexus"
    current_source = _eiger_append_source(
        tmp_path / "same-extent-source", (2,), generation=1
    )
    assert current_source.dataset_paths == ("/entry/data/data_000001",)
    assert current_source.external_members[0].dataset_path == "/entry/data/data"
    legacy = _intent_for_source(
        tmp_path,
        replace(current_source, dataset_paths=(), generation=0),
        (0, 1),
    )
    initial = qualify(target, legacy)
    assert initial.disposition is Disposition.WRITE
    _seed_target(target, legacy.labels, tmp_path)
    with h5py.File(target, "r+") as handle:
        commit_lineage(handle["entry"], initial, written_labels=legacy.labels)
    before = target.read_bytes()

    current = _intent_for_source(
        tmp_path, current_source, (0, 1),
    )
    decision = qualify(target, current)

    assert decision.disposition is Disposition.REFUSE
    assert decision.reason == "persisted source has no exact dataset selectors"
    assert target.read_bytes() == before


def test_legacy_empty_dataset_paths_refuse_eiger_growth_write(tmp_path):
    from dataclasses import replace

    (Disposition, _Member, _Intent, _Source, commit_lineage,
     qualify) = _append_api()
    target = tmp_path / "legacy-eiger-growth.nexus"
    source_root = tmp_path / "growth-source"
    initial_source = _eiger_append_source(
        source_root, (2,), generation=0
    )
    legacy = _intent_for_source(
        tmp_path, replace(initial_source, dataset_paths=()), (0, 1)
    )
    initial = qualify(target, legacy)
    assert initial.disposition is Disposition.WRITE
    _seed_target(target, legacy.labels, tmp_path)
    with h5py.File(target, "r+") as handle:
        commit_lineage(handle["entry"], initial, written_labels=legacy.labels)

    current_source = _eiger_append_source(
        source_root, (2, 3), generation=1
    )
    current = _intent_for_source(
        tmp_path, current_source, (0, 1, 2, 3, 4),
    )
    decision = qualify(target, current)

    assert decision.disposition is Disposition.REFUSE
    assert decision.reason == "persisted source has no exact dataset selectors"


def test_legacy_empty_dataset_paths_refuse_unmatched_eiger_growth_selectors(
    tmp_path,
):
    from dataclasses import replace

    (Disposition, _Member, _Intent, _Source, commit_lineage,
     qualify) = _append_api()
    target = tmp_path / "legacy-eiger-selector-mismatch.nexus"
    source_root = tmp_path / "mismatch-source"
    initial_source = _eiger_append_source(
        source_root, (2,), generation=0
    )
    legacy = _intent_for_source(
        tmp_path, replace(initial_source, dataset_paths=()), (0, 1)
    )
    initial = qualify(target, legacy)
    assert initial.disposition is Disposition.WRITE
    _seed_target(target, legacy.labels, tmp_path)
    with h5py.File(target, "r+") as handle:
        commit_lineage(handle["entry"], initial, written_labels=legacy.labels)
    before = target.read_bytes()

    current_source = _eiger_append_source(
        source_root, (2, 1), generation=1
    )
    mismatched = _intent_for_source(
        tmp_path,
        replace(
            current_source,
            dataset_paths=(
                "/entry/data/not-the-authenticated-member",
                "/entry/data/data_000002",
            ),
        ),
        (0, 1, 2),
    )
    decision = qualify(target, mismatched)

    assert decision.disposition is Disposition.REFUSE
    assert decision.reason == "persisted source has no exact dataset selectors"
    assert target.read_bytes() == before


def test_append_refuses_malformed_earlier_epoch_member_history(tmp_path):
    import xrd_tools.io.append as append_module
    target = tmp_path / "earlier-history.nexus"
    first = _intent(tmp_path, extent=2, labels=(0, 1))
    first_decision = append_module.qualify_append(target, first)
    _seed_target(target, (0, 1), tmp_path)
    with h5py.File(target, "r+") as handle:
        append_module.commit_append_lineage(
            handle["entry"], first_decision, written_labels=(0, 1),
        )
    second = _intent(tmp_path, extent=3, labels=(0, 1, 2))
    second_decision = append_module.qualify_append(target, second)
    with h5py.File(target, "r+") as handle:
        _write_result_rows(handle["entry"], (2,))
        append_module.commit_append_lineage(
            handle["entry"], second_decision, written_labels=(2,),
        )
        dataset = handle["entry/reduction/config/append_lineage"]
        lineage = json.loads(dataset[()].decode())
        lineage["epochs"][0]["source"]["external_members"][0]["ordinal"] = 7
        del handle["entry/reduction/config/append_lineage"]
        handle["entry/reduction/config"].create_dataset(
            "append_lineage", data=json.dumps(
                lineage, sort_keys=True, separators=(",", ":")),
        )
    before = target.read_bytes()

    decision = append_module.qualify_append(
        target, _intent(tmp_path, extent=4, labels=(0, 1, 2, 3)),
    )
    assert decision.disposition is append_module.AppendDisposition.REFUSE
    assert "ordinals" in decision.reason
    assert target.read_bytes() == before


def test_append_refuses_partial_external_selector_inventory_byte_exact(
    tmp_path,
):
    from dataclasses import replace
    import xrd_tools.io.append as append_module

    (_Disposition, AppendExternalMember, _AppendIntent, AppendSource,
     _commit, _qualify) = _append_api()
    members = tuple(AppendExternalMember(
        path=str(tmp_path / f"member-{ordinal}.h5"),
        dataset_path="/entry/data/data", size=10, mtime_ns=20,
        source_start=ordinal, source_stop=ordinal + 1, ordinal=ordinal,
    ) for ordinal in range(2))
    source = AppendSource(
        path=str(tmp_path / "master.h5"), adapter_id="nexus_hdf5",
        size=30, mtime_ns=40, extent=2,
        dataset_paths=("/entry/data/a", "/entry/data/b"),
        external_members=members,
    )
    with pytest.raises(ValueError, match="cover every member"):
        replace(source, dataset_paths=source.dataset_paths[:1])

    intent = _intent_for_source(tmp_path, source, (0, 1))
    target = tmp_path / "partial-selector-history.nexus"
    decision = append_module.qualify_append(target, intent)
    _seed_target(target, intent.labels, tmp_path)
    with h5py.File(target, "r+") as handle:
        append_module.commit_append_lineage(
            handle["entry"], decision, written_labels=intent.labels,
        )
        dataset = handle["entry/reduction/config/append_lineage"]
        lineage = json.loads(dataset[()].decode())
        lineage["epochs"][0]["source"]["dataset_paths"] = (
            lineage["epochs"][0]["source"]["dataset_paths"][:1]
        )
        del handle["entry/reduction/config/append_lineage"]
        handle["entry/reduction/config"].create_dataset(
            "append_lineage", data=json.dumps(
                lineage, sort_keys=True, separators=(",", ":"),
            ),
        )
    before = target.read_bytes()

    refused = append_module.qualify_append(target, intent)

    assert refused.disposition is append_module.AppendDisposition.REFUSE
    assert "cover every member" in refused.reason
    assert target.read_bytes() == before


@pytest.mark.parametrize(
    "mutation",
    ("pending", "gap", "foreign_source", "foreign_science", "source_base",
     "member_history", "missing_lineage"),
)
def test_append_refusal_families_leave_target_byte_exact(tmp_path, mutation):
    (Disposition, _Member, _Intent, _Source, commit_lineage,
     qualify) = _append_api()
    target = tmp_path / "refuse.nexus"
    first = _intent(tmp_path, extent=2, labels=(0, 1))
    decision = qualify(target, first)
    _seed_target(target, (0, 1), tmp_path)
    with h5py.File(target, "r+") as handle:
        commit_lineage(handle["entry"], decision, written_labels=(0, 1))
        if mutation == "pending":
            raw = handle["entry/reduction/config/append_lineage"][()].decode()
            import json
            value = json.loads(raw)
            value["state"] = "pending"
            del handle["entry/reduction/config/append_lineage"]
            handle["entry/reduction/config"].create_dataset(
                "append_lineage", data=json.dumps(value, sort_keys=True,
                                                   separators=(",", ":")))
        elif mutation == "gap":
            del handle["entry/integrated_2d/frame_index"]
            handle["entry/integrated_2d"].create_dataset(
                "frame_index", data=np.asarray([0, 2], dtype=np.int64))
        elif mutation == "member_history":
            raw = handle["entry/reduction/config/append_lineage"][()].decode()
            import json
            value = json.loads(raw)
            value["epochs"][0]["source"]["external_members"][0][
                "dataset_path"] = "/wrong"
            del handle["entry/reduction/config/append_lineage"]
            handle["entry/reduction/config"].create_dataset(
                "append_lineage", data=json.dumps(value, sort_keys=True,
                                                   separators=(",", ":")))
        elif mutation == "missing_lineage":
            del handle["entry/reduction/config/append_lineage"]

    kwargs = {}
    if mutation == "foreign_source":
        kwargs["source_identity"] = "other/run"
    if mutation == "foreign_science":
        kwargs["science"] = "science-v2"
    if mutation == "source_base":
        kwargs["source_base"] = tmp_path / "other-base"
    proposed = _intent(tmp_path, extent=3, labels=(0, 1, 2), **kwargs)
    before = target.read_bytes()
    refused = qualify(target, proposed)
    after = target.read_bytes()
    assert refused.disposition is Disposition.REFUSE
    assert after == before


@pytest.mark.parametrize(
    "mutation,reason",
    (("size", "source size regressed"),
     ("mtime_ns", "source mtime_ns regressed"),
     ("digest", "digest evidence was downgraded")),
)
def test_live_extension_refuses_top_level_regression_and_digest_downgrade(
    tmp_path, mutation, reason,
):
    from dataclasses import replace
    module = importlib.import_module("xrd_tools.io.append")
    prior = _intent(
        tmp_path, extent=1, labels=(0,), generation=0,
        modes=("1d:default",),
    )
    prior = replace(prior, source=replace(prior.source, digest="exact-prior"))
    decision = module.begin_same_run_lineage(prior)
    current = _intent(
        tmp_path, extent=2, labels=(0, 1), generation=1,
        modes=("1d:default",),
    )
    if mutation == "digest":
        source = replace(current.source, digest=None)
    else:
        source = replace(
            current.source,
            digest="exact-current",
            **{mutation: getattr(prior.source, mutation) - 1},
        )
    refused = module.extend_same_run_lineage(
        decision, prior, replace(current, source=source))
    assert refused.disposition.value == "refuse"
    assert reason in refused.reason


def test_first_run_same_owner_extends_without_readmit_reopen_or_cursor_rebuild(
    tmp_path, monkeypatch,
):
    from xrd_tools.core.scan import Scan, ScanFrame
    from xrd_tools.io import get_output_transaction_coordinator
    from xrd_tools.reduction import (
        FrameReduction, NexusSink, ReductionPlan, ReductionResult,
    )
    core = importlib.import_module("xrd_tools.reduction.core")
    coordinator = get_output_transaction_coordinator()
    admits, opens = [], []
    real_admit, real_open = coordinator.admit, core.open_nexus_writer

    def admit(*args, **kwargs):
        admits.append(args[0])
        return real_admit(*args, **kwargs)

    def open_writer(*args, **kwargs):
        opens.append(args[0])
        return real_open(*args, **kwargs)

    monkeypatch.setattr(coordinator, "admit", admit)
    monkeypatch.setattr(core, "open_nexus_writer", open_writer)
    target = tmp_path / "same-run.nexus"
    first = _intent(
        tmp_path, extent=1, labels=(0,), generation=0,
        modes=("1d:default",),
    )
    sink = NexusSink(
        target, overwrite=True, source_base=tmp_path,
        flush_every=None, same_run_intent=first,
    )
    sink.begin(Scan("same-run", []), ReductionPlan(integration_2d=None))
    owner = sink.extension_owner
    copied_owner = copy.copy(owner)
    assert copied_owner == owner and copied_owner is not owner
    with pytest.raises(RuntimeError, match="exact live owner"):
        sink.extend_live(copied_owner, first)
    sink.write(ScanFrame(0), FrameReduction(0, result_1d=_r1(0)))
    cursors = sink._writer._row_cursors

    # Exact replay is idempotent; the next generation contributes only label 1.
    assert sink.extend_live(owner, first) is sink._writer.append_decision
    second = _intent(
        tmp_path, extent=2, labels=(0, 1), generation=1,
        modes=("1d:default",),
    )
    decision = sink.extend_live(owner, second)
    assert decision.write_labels == (0, 1)
    sink.write(ScanFrame(1), FrameReduction(1, result_1d=_r1(1)))
    assert sink._writer._row_cursors is cursors
    assert len(admits) == len(opens) == 1

    with pytest.raises(LeaseUnavailable):
        contender = coordinator.admit(
            target,
            transaction_owner=OwnerToken("contender transaction"),
            target_owner=OwnerToken("contender target"),
        )
        contender.acquire_lease(
            admission=contender.admission,
            transaction_owner=contender._transaction_owner,
            target_owner=contender._target_owner,
            owners={role: OwnerToken(f"contender {role.value}")
                    for role in LeaseOwner},
        )
    sink.finish(ReductionResult("same-run", {}, 2))
    with h5py.File(target, "r") as handle:
        assert tuple(handle["entry/integrated_1d/frame_index"][()]) == (0, 1)
        lineage = json.loads(
            handle["entry/reduction/config/append_lineage"][()].decode())
    assert lineage["state"] == "committed"
    assert lineage["epochs"][-1]["source"]["generation"] == 1


def test_same_run_extension_rollback_restores_prior_bytes(tmp_path):
    from xrd_tools.core.scan import Scan, ScanFrame
    from xrd_tools.reduction import FrameReduction, NexusSink, ReductionPlan, ReductionResult
    target = tmp_path / "rollback.nexus"
    _seed_target(target, (7,), tmp_path)
    prior = target.read_bytes()
    first = _intent(
        tmp_path, extent=1, labels=(0,), generation=0,
        modes=("1d:default",),
    )
    sink = NexusSink(
        target, overwrite=True, source_base=tmp_path,
        same_run_intent=first, flush_every=None,
    )
    sink.begin(Scan("rollback", []), ReductionPlan(integration_2d=None))
    sink.write(ScanFrame(0), FrameReduction(0, result_1d=_r1(0)))
    sink.extend_live(sink.extension_owner, _intent(
        tmp_path, extent=2, labels=(0, 1), generation=1,
        modes=("1d:default",),
    ))
    sink.abort(ReductionResult("rollback", {}, 1, failed=True))
    assert target.read_bytes() == prior


def test_committed_epoch_extension_rollback_restores_epoch_not_original(
    tmp_path, monkeypatch,
):
    from xrd_tools.core.scan import Scan, ScanFrame
    from xrd_tools.io import AppendRefused, get_output_transaction_coordinator
    from xrd_tools.reduction import FrameReduction, NexusSink, ReductionPlan, ReductionResult
    core = importlib.import_module("xrd_tools.reduction.core")
    coordinator = get_output_transaction_coordinator()
    admits, opens = [], []
    real_admit, real_open = coordinator.admit, core.open_nexus_writer
    monkeypatch.setattr(
        coordinator, "admit",
        lambda *a, **k: (admits.append(a[0]), real_admit(*a, **k))[1],
    )
    monkeypatch.setattr(
        core, "open_nexus_writer",
        lambda *a, **k: (opens.append(a[0]), real_open(*a, **k))[1],
    )
    target = tmp_path / "epoch-rollback.nexus"
    _seed_target(target, (7,), tmp_path)
    original = target.read_bytes()
    first = _intent(
        tmp_path, extent=1, labels=(0,), generation=0,
        modes=("1d:default",),
    )
    sink = NexusSink(
        target, overwrite=True, source_base=tmp_path,
        same_run_intent=first, flush_every=None,
        run_configuration_provenance={
            "output_mode": "Overwrite", "live_mode": True,
        },
    )
    sink.begin(Scan("epoch", []), ReductionPlan(integration_2d=None))
    owner, lease = sink.extension_owner, sink._lease
    sink.write(ScanFrame(0), FrameReduction(0, result_1d=_r1(0)))
    anchor = sink.commit_epoch(ReductionResult("epoch", {}, 1))
    epoch_a = target.read_bytes()
    assert epoch_a != original
    assert anchor.skip_labels == (0,) and anchor.write_labels == ()
    assert sink._transaction.snapshot().phase is TransactionPhase.EPOCH_COMMITTED
    assert sink.extend_live(owner, first) is anchor
    assert len(opens) == 1
    with pytest.raises(LeaseUnavailable):
        contender = coordinator.admit(
            target,
            transaction_owner=OwnerToken("epoch contender transaction"),
            target_owner=OwnerToken("epoch contender target"),
        )
        contender.acquire_lease(
            admission=contender.admission,
            transaction_owner=contender._transaction_owner,
            target_owner=contender._target_owner,
            owners={role: OwnerToken(f"epoch contender {role.value}")
                    for role in LeaseOwner},
        )
    admission_count = len(admits)

    second = _intent(
        tmp_path, extent=2, labels=(0, 1), generation=1,
        modes=("1d:default",),
    )
    with pytest.raises(AppendRefused, match="generation did not advance"):
        sink.extend_live(owner, _intent(
            tmp_path, extent=2, labels=(0, 1), generation=0,
            modes=("1d:default",),
        ))
    decision = sink.extend_live(owner, second)
    assert decision.skip_labels == (0,) and decision.write_labels == (1,)
    sink.write(ScanFrame(1), FrameReduction(1, result_1d=_r1(1)))
    assert sink._lease is lease
    assert len(admits) == admission_count and len(opens) == 2
    sink.abort(ReductionResult("epoch", {}, 1, failed=True))
    assert target.read_bytes() == epoch_a


def test_owned_epoch_open_constructor_fault_restores_a_and_releases(tmp_path, monkeypatch):
    from xrd_tools.core.scan import Scan, ScanFrame
    from xrd_tools.reduction import FrameReduction, NexusSink, ReductionPlan, ReductionResult
    core = importlib.import_module("xrd_tools.reduction.core")
    target = tmp_path / "epoch-open-fault.nexus"
    first = _intent(
        tmp_path, extent=1, labels=(0,), generation=0,
        modes=("1d:default",),
    )
    sink = NexusSink(
        target, overwrite=True, source_base=tmp_path,
        same_run_intent=first, flush_every=None,
    )
    sink.begin(Scan("fault", []), ReductionPlan(integration_2d=None))
    sink.write(ScanFrame(0), FrameReduction(0, result_1d=_r1(0)))
    sink.commit_epoch(ReductionResult("fault", {}, 1))
    epoch_a = target.read_bytes()
    sink.extend_live(sink.extension_owner, _intent(
        tmp_path, extent=2, labels=(0, 1), generation=1,
        modes=("1d:default",),
    ))
    monkeypatch.setattr(
        core, "NexusRecordWriter",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("epoch constructor")),
    )
    with pytest.raises(ValueError, match="epoch constructor"):
        sink.write(ScanFrame(1), FrameReduction(1, result_1d=_r1(1)))
    assert target.read_bytes() == epoch_a
    assert sink._transaction.snapshot().phase is TransactionPhase.ABORTED
    assert not sink._transaction.backup.exists()
    assert not sink._transaction._stream_partial.exists()
    assert sink._transaction_owners is None
    _assert_lease_available(target)


def test_owned_epoch_seed_fault_restores_committed_epoch_authority(
    tmp_path, monkeypatch,
):
    (_coordinator, transaction, target, transaction_owner, target_owner,
     owners, lease) = _transaction(tmp_path)
    attempt = transaction.begin_stream(
        admission=transaction.admission,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease, pool=_Pool(), file_lock=threading.RLock(),
    )
    transaction.authorize_stream_mutation(attempt, lease=lease)
    target.write_bytes(b"epoch-a")
    transaction.seal_stream_terminal(attempt, lease=lease)
    transaction.commit_stream_epoch(attempt, lease=lease)
    real_copy = transaction._copy_stream_seed

    def fail_after_seed():
        real_copy()
        raise OSError("epoch seed fault")

    monkeypatch.setattr(transaction, "_copy_stream_seed", fail_after_seed)
    with pytest.raises(OSError, match="epoch seed fault"):
        transaction.begin_stream_epoch(
            attempt,
            admission=transaction.admission,
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
        )
    assert target.read_bytes() == b"epoch-a"
    assert transaction.snapshot().phase is TransactionPhase.EPOCH_COMMITTED
    assert not transaction.backup.exists() and not transaction._stream_partial.exists()
    transaction.commit_stream(attempt, lease=lease)
    _release(transaction, lease, owners)


def test_sink_epoch_seed_fault_terminalizes_restored_epoch(tmp_path, monkeypatch):
    from xrd_tools.core.scan import Scan, ScanFrame
    from xrd_tools.reduction import FrameReduction, NexusSink, ReductionPlan, ReductionResult
    target = tmp_path / "sink-epoch-seed-fault.nexus"
    first = _intent(
        tmp_path, extent=1, labels=(0,), generation=0,
        modes=("1d:default",),
    )
    sink = NexusSink(
        target, overwrite=True, source_base=tmp_path,
        same_run_intent=first, flush_every=None,
    )
    sink.begin(Scan("fault", []), ReductionPlan(integration_2d=None))
    sink.write(ScanFrame(0), FrameReduction(0, result_1d=_r1(0)))
    sink.commit_epoch(ReductionResult("fault", {}, 1))
    epoch_a = target.read_bytes()
    sink.extend_live(sink.extension_owner, _intent(
        tmp_path, extent=2, labels=(0, 1), generation=1,
        modes=("1d:default",),
    ))
    real_copy = sink._transaction._copy_stream_seed
    transaction_module = importlib.import_module("xrd_tools.io.output_transaction")
    real_unlink = transaction_module._unlink
    unlink_failed = False

    def fail_after_seed():
        real_copy()
        raise OSError("epoch seed fault")

    def fail_partial_unlink_once(path):
        nonlocal unlink_failed
        if Path(path) == sink._transaction._stream_partial and not unlink_failed:
            unlink_failed = True
            raise OSError("partial cleanup fault")
        return real_unlink(path)

    monkeypatch.setattr(sink._transaction, "_copy_stream_seed", fail_after_seed)
    monkeypatch.setattr(transaction_module, "_unlink", fail_partial_unlink_once)
    with pytest.raises(OSError, match="epoch seed fault"):
        sink.write(ScanFrame(1), FrameReduction(1, result_1d=_r1(1)))
    assert target.read_bytes() == epoch_a
    assert sink._transaction.snapshot().phase is TransactionPhase.COMMITTED
    assert sink._transaction_owners is None
    _assert_lease_available(target)


def test_epoch_backup_cleanup_retry_keeps_lock_pool_and_epoch_authority(
    tmp_path, monkeypatch,
):
    module = importlib.import_module("xrd_tools.io.output_transaction")

    class TrackingLock:
        def __init__(self):
            self.lock = threading.RLock()
            self.depth = 0

        def __enter__(self):
            self.lock.acquire()
            self.depth += 1
            return self

        def __exit__(self, *_args):
            self.depth -= 1
            self.lock.release()

    (coordinator, transaction, target, transaction_owner, target_owner,
     owners, lease) = _transaction(tmp_path)
    pool, lock = _Pool(), TrackingLock()
    attempt = transaction.begin_stream(
        admission=transaction.admission,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease, pool=pool, file_lock=lock,
    )
    transaction.authorize_stream_mutation(attempt, lease=lease)
    epoch_a = b"epoch-a"
    target.write_bytes(epoch_a)
    transaction.seal_stream_terminal(attempt, lease=lease)
    real_unlink, failed = module._unlink, False

    def fail_once(path):
        nonlocal failed
        if Path(path) == transaction.backup:
            assert lock.depth > 0
            if not failed:
                failed = True
                raise OSError("backup cleanup fault")
        return real_unlink(path)

    monkeypatch.setattr(module, "_unlink", fail_once)
    with pytest.raises(CleanupIncomplete):
        transaction.commit_stream_epoch(attempt, lease=lease)
    snapshot = transaction.snapshot()
    assert snapshot.phase is TransactionPhase.CLEANUP_PENDING
    assert {RetryAction.BACKUP_UNLINK, RetryAction.POOL_RESUME} <= set(
        snapshot.pending_actions)
    assert target.read_bytes() == epoch_a
    assert pool.events == [("pause", str(target))]
    with pytest.raises(LeaseUnavailable):
        contender = coordinator.admit(
            target,
            transaction_owner=OwnerToken("epoch cleanup contender"),
            target_owner=OwnerToken("epoch cleanup target"),
        )
        contender.acquire_lease(
            admission=contender.admission,
            transaction_owner=contender._transaction_owner,
            target_owner=contender._target_owner,
            owners={role: OwnerToken(f"epoch cleanup {role.value}")
                    for role in LeaseOwner},
        )

    snapshot = transaction.retry_cleanup(snapshot.cleanup_token)
    assert snapshot.phase is TransactionPhase.EPOCH_COMMITTED
    assert set(snapshot.pending_actions) == {RetryAction.POOL_RESUME}
    assert pool.events == [("pause", str(target))]
    next_attempt = transaction.begin_stream_epoch(
        attempt,
        admission=transaction.admission,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease,
    )
    transaction.abort_stream(next_attempt, lease=lease)
    assert target.read_bytes() == epoch_a
    assert pool.events[-1] == ("resume", str(target))
    _release(transaction, lease, owners)


def test_cross_run_append_preflight_extends_the_bound_writer(tmp_path):
    from xrd_tools.core.scan import Scan, ScanFrame
    from xrd_tools.io import prepare_append_preflight
    from xrd_tools.reduction import FrameReduction, NexusSink, ReductionPlan, ReductionResult
    target = tmp_path / "cross-run.nexus"
    first = _intent(
        tmp_path, extent=1, labels=(0,), generation=0,
        modes=("1d:default",),
    )
    initial = NexusSink(
        target, overwrite=True, source_base=tmp_path,
        same_run_intent=first, flush_every=None,
    )
    initial.begin(Scan("cross-run", []), ReductionPlan(integration_2d=None))
    initial.write(ScanFrame(0), FrameReduction(0, result_1d=_r1(0)))
    initial.finish(ReductionResult("cross-run", {}, 1))

    second = _intent(
        tmp_path, extent=2, labels=(0, 1), generation=0,
        modes=("1d:default",),
    )
    preflight = prepare_append_preflight(target, second)
    assert preflight.snapshot.skip_labels == (0,)
    assert preflight.snapshot.write_labels == (1,)
    resumed = NexusSink(
        target, source_base=tmp_path, append_preflight=preflight,
        flush_every=None,
    )
    resumed.begin(Scan("cross-run", []), ReductionPlan(integration_2d=None))
    resumed.write(ScanFrame(1), FrameReduction(1, result_1d=_r1(1)))
    anchor = resumed.commit_epoch(ReductionResult("cross-run", {}, 1))
    assert anchor.skip_labels == (0, 1) and anchor.write_labels == ()
    third = _intent(
        tmp_path, extent=3, labels=(0, 1, 2), generation=1,
        modes=("1d:default",),
    )
    preflight.extend(third)
    assert preflight.extend(third) == preflight.snapshot
    with pytest.raises(importlib.import_module("xrd_tools.io").AppendRefused,
                       match="generation did not advance"):
        preflight.extend(_intent(
            tmp_path, extent=3, labels=(0, 1, 2), generation=0,
            modes=("1d:default",),
        ))
    assert preflight.snapshot.skip_labels == (0, 1)
    assert preflight.snapshot.write_labels == (2,)
    resumed.write(ScanFrame(2), FrameReduction(2, result_1d=_r1(2)))
    resumed.finish(ReductionResult("cross-run", {}, 2))
    assert preflight.snapshot.state.value == "committed"
    final = importlib.import_module("xrd_tools.io.append").qualify_append(
        target,
        _intent(tmp_path, extent=3, labels=(0, 1, 2), generation=0,
                modes=("1d:default",)),
    )
    assert final.disposition.value == "skip"


def test_noop_release_failure_is_retryable_not_integrity_hold(tmp_path, monkeypatch):
    from xrd_tools.io import prepare_append_preflight
    transaction_module = importlib.import_module("xrd_tools.io.output_transaction")
    append_module = importlib.import_module("xrd_tools.io.append")
    target = tmp_path / "noop-release.nexus"
    intent = _intent(
        tmp_path, extent=1, labels=(0,), modes=("1d:default",))
    decision = append_module.begin_same_run_lineage(intent)
    _seed_target(target, (0,), tmp_path)
    with h5py.File(target, "r+") as handle:
        append_module.commit_append_lineage(
            handle["entry"], decision, written_labels=(0,))
    preflight = prepare_append_preflight(target, intent)
    real_release = transaction_module.OutputTransaction.release_lease_owner
    failed = False

    def fail_once(self, *args, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("release fault")
        return real_release(self, *args, **kwargs)

    monkeypatch.setattr(
        transaction_module.OutputTransaction, "release_lease_owner", fail_once)
    with pytest.raises(OSError, match="release fault"):
        preflight.complete_noop()
    assert preflight.snapshot.state.value == "retryable"
    assert preflight.retry_cleanup().state.value == "noop"
    _assert_lease_available(target)


def test_committed_preflight_release_failure_retries_without_abandon(
    tmp_path, monkeypatch,
):
    from xrd_tools.core.scan import Scan, ScanFrame
    from xrd_tools.io import prepare_append_preflight
    from xrd_tools.reduction import FrameReduction, NexusSink, ReductionPlan, ReductionResult
    transaction_module = importlib.import_module("xrd_tools.io.output_transaction")
    target = tmp_path / "commit-release.nexus"
    intent = _intent(
        tmp_path, extent=1, labels=(0,), modes=("1d:default",))
    preflight = prepare_append_preflight(target, intent)
    sink = NexusSink(
        target, source_base=tmp_path, append_preflight=preflight,
        flush_every=None,
    )
    sink.begin(Scan("release", []), ReductionPlan(integration_2d=None))
    sink.write(ScanFrame(0), FrameReduction(0, result_1d=_r1(0)))
    real_release = transaction_module.OutputTransaction.release_lease_owner
    failed = False

    def fail_once(self, *args, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("release fault")
        return real_release(self, *args, **kwargs)

    monkeypatch.setattr(
        transaction_module.OutputTransaction, "release_lease_owner", fail_once)
    with pytest.raises(OSError, match="release fault"):
        sink.finish(ReductionResult("release", {}, 1))
    assert sink._transaction.snapshot().phase is TransactionPhase.COMMITTED
    assert preflight.snapshot.state.value == "retryable"
    assert preflight.retry_cleanup().state.value == "committed"
    _assert_lease_available(target)


def test_nonappend_terminal_release_failure_abort_only_settles_cleanup(
    tmp_path, monkeypatch,
):
    from xrd_tools.core.scan import Scan, ScanFrame
    from xrd_tools.reduction import FrameReduction, NexusSink, ReductionPlan, ReductionResult
    transaction_module = importlib.import_module("xrd_tools.io.output_transaction")
    target = tmp_path / "replace-release.nexus"
    sink = NexusSink(target, overwrite=True, flush_every=None)
    sink.begin(Scan("release", []), ReductionPlan(integration_2d=None))
    sink.write(ScanFrame(0), FrameReduction(0, result_1d=_r1(0)))
    real_release = transaction_module.OutputTransaction.release_lease_owner
    failed = False

    def fail_once(self, *args, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("release fault")
        return real_release(self, *args, **kwargs)

    monkeypatch.setattr(
        transaction_module.OutputTransaction, "release_lease_owner", fail_once)
    result = ReductionResult("release", {}, 1)
    with pytest.raises(OSError, match="release fault"):
        sink.finish(result)
    committed = target.read_bytes()
    sink.abort(ReductionResult("release", {}, 1, failed=True))
    assert target.read_bytes() == committed
    assert sink._transaction_owners is None
    _assert_lease_available(target)


def test_cancelled_pending_epoch_restores_last_committed_epoch(tmp_path):
    from xrd_tools.core.scan import Scan, ScanFrame
    from xrd_tools.reduction import FrameReduction, NexusSink, ReductionPlan, ReductionResult
    target = tmp_path / "cancel-pending-epoch.nexus"
    first = _intent(
        tmp_path, extent=1, labels=(0,), generation=0,
        modes=("1d:default",),
    )
    sink = NexusSink(
        target, overwrite=True, source_base=tmp_path,
        same_run_intent=first, flush_every=None,
    )
    sink.begin(Scan("cancel", []), ReductionPlan(integration_2d=None))
    sink.write(ScanFrame(0), FrameReduction(0, result_1d=_r1(0)))
    sink.commit_epoch(ReductionResult("cancel", {}, 1))
    epoch_a = target.read_bytes()
    sink.extend_live(sink.extension_owner, _intent(
        tmp_path, extent=2, labels=(0, 1), generation=1,
        modes=("1d:default",),
    ))
    sink.finish(ReductionResult("cancel", {}, 1, cancelled=True))
    assert target.read_bytes() == epoch_a
    assert sink._transaction_owners is None
    _assert_lease_available(target)


def test_abort_never_rolls_back_a_published_h10_durable_extension(tmp_path):
    """A monotonic H10 durable receipt must name bytes abort cannot remove."""
    from xrd_tools.core.scan import Scan, ScanFrame
    from xrd_tools.reduction import (
        FrameReduction,
        NexusSink,
        ReductionPlan,
        ReductionResult,
    )

    target = tmp_path / "durable-extension-abort.nexus"
    mode = ResultMode.one_d()
    target_name = f"nexus:{target}"
    ledger = StageLedger(
        required_modes=(mode,),
        targets_by_mode={mode: (target_name,)},
    )
    first = _intent(
        tmp_path, extent=1, labels=(0,), generation=0,
        modes=("1d:default",),
    )
    sink = NexusSink(
        target, overwrite=True, source_base=tmp_path,
        same_run_intent=first, flush_every=None,
    )
    sink.bind_session(_LedgerFacade(ledger))
    sink.begin(Scan("durable", []), ReductionPlan(integration_2d=None))

    attempt = ledger.record_accepted(0)
    ledger.record_outcome(
        0, ItemDisposition.COMPLETED, produced=(mode,), attempt=attempt,
    )
    sink.write(ScanFrame(0), FrameReduction(0, result_1d=_r1(0)))
    ledger.record_written(0, (mode,))
    sink.commit_epoch(ReductionResult("durable", {}, 1))

    sink.extend_live(sink.extension_owner, _intent(
        tmp_path, extent=2, labels=(0, 1), generation=1,
        modes=("1d:default",),
    ))
    attempt = ledger.record_accepted(1)
    ledger.record_outcome(
        1, ItemDisposition.COMPLETED, produced=(mode,), attempt=attempt,
    )
    sink.write(ScanFrame(1), FrameReduction(1, result_1d=_r1(1)))
    ledger.record_written(1, (mode,))
    sink.flush(force=True)

    durable_before_abort = {
        label for label, durable_mode, durable_target
        in ledger.snapshot().durable
        if durable_mode == mode and durable_target == target_name
    }
    assert durable_before_abort == {0, 1}, "periodic durability was lost"

    sink.abort(ReductionResult("durable", {}, 2, failed=True))
    with h5py.File(target, "r") as handle:
        disk_labels = set(int(label) for label in (
            handle["entry/integrated_1d/frame_index"][()]
        ))
    assert disk_labels == durable_before_abort
    assert sink._transaction_owners is None
    snapshot = sink._transaction.snapshot()
    assert snapshot.phase is TransactionPhase.ABORTED
    assert snapshot.durable_floor is not None
    assert snapshot.partial_path == str(target)
    _assert_lease_available(target)


def test_cancelled_written_epoch_commits_only_exact_written_prefix(tmp_path):
    from xrd_tools.core.scan import Scan, ScanFrame
    from xrd_tools.reduction import FrameReduction, NexusSink, ReductionPlan, ReductionResult
    target = tmp_path / "cancel-written-prefix.nexus"
    intent = _intent(
        tmp_path, extent=3, labels=(0, 1, 2), generation=0,
        modes=("1d:default",),
    )
    sink = NexusSink(
        target, overwrite=True, source_base=tmp_path,
        same_run_intent=intent, flush_every=None,
    )
    sink.begin(Scan("cancel", []), ReductionPlan(integration_2d=None))
    sink.write(ScanFrame(0), FrameReduction(0, result_1d=_r1(0)))
    sink.finish(ReductionResult("cancel", {}, 1, cancelled=True))
    with h5py.File(target, "r") as handle:
        assert tuple(handle["entry/integrated_1d/frame_index"][()]) == (0,)
        lineage = json.loads(
            handle["entry/reduction/config/append_lineage"][()].decode())
    assert lineage["state"] == "committed"
    assert lineage["epochs"][-1]["labels"] == [0]
    assert lineage["epochs"][-1]["source"]["extent"] == 1


def test_many_to_one_stop_keeps_full_source_extent_once_output_is_written(tmp_path):
    from xrd_tools.io import AppendImageMember, AppendIntent, AppendSource
    append_module = importlib.import_module("xrd_tools.io.append")
    members = tuple(AppendImageMember(
        path=str(tmp_path / f"source-{index}.tif"), size=1, mtime_ns=2,
        source_start=index, source_stop=index + 1, ordinal=index,
    ) for index in range(3))
    intent = AppendIntent(
        "entry", str(tmp_path), "average", "science", ("1d:default",),
        AppendSource(
            str(tmp_path / "source-0.tif"), "image-series:.tif", 3, 2, 3,
            image_members=members,
        ),
        (1,),
    )
    decision = append_module.begin_same_run_lineage(intent)
    truncated, truncated_intent = append_module.truncate_append_epoch(
        decision, intent, (1,))
    assert truncated.write_labels == (1,)
    assert truncated_intent.source.extent == 3
    assert len(truncated_intent.source.image_members) == 3


@pytest.mark.parametrize(
    "member_kwargs,match",
    [
        ({"path": ""}, "path is required"),
        ({"dataset_path": ""}, "dataset path is required"),
        ({"source_start": 1, "source_stop": 2}, "contiguous from zero"),
        ({"source_stop": 0}, "range is invalid"),
        ({"ordinal": 1}, "ordered ordinals"),
    ],
)
def test_source_member_identity_is_total(tmp_path, member_kwargs, match):
    from xrd_tools.io import AppendExternalMember, AppendSource
    values = dict(
        path=str(tmp_path / "member.h5"), dataset_path="/entry/data/data",
        size=1, mtime_ns=2, source_start=0, source_stop=1, ordinal=0,
    )
    values.update(member_kwargs)
    if (not values["path"] or not values["dataset_path"]
            or values["source_stop"] <= values["source_start"]):
        with pytest.raises(ValueError, match=match):
            AppendExternalMember(**values)
        return
    member = AppendExternalMember(**values)
    with pytest.raises(ValueError, match=match):
        AppendSource(
            path=str(tmp_path / "master.h5"), adapter_id="hdf5",
            size=1, mtime_ns=2, extent=1, external_members=(member,),
        )


def test_preflight_owns_lease_before_qualification_and_rejects_copy(
    tmp_path, monkeypatch,
):
    from xrd_tools.io import get_output_transaction_coordinator, prepare_append_preflight
    module = importlib.import_module("xrd_tools.io.append")
    coordinator = get_output_transaction_coordinator()
    target = tmp_path / "preflight.nexus"
    intent = _intent(tmp_path, extent=1, labels=(0,), modes=("1d:default",))
    real_qualify = module.qualify_append
    real_admit = coordinator.admit

    class BorrowedLock:
        active = False

        def __enter__(self):
            assert not self.active
            self.active = True
            return self

        def __exit__(self, *_exc):
            self.active = False

    lock = BorrowedLock()

    def admit(*args, **kwargs):
        assert lock.active, "target admission must share the borrowed file lock"
        return real_admit(*args, **kwargs)

    def qualify(path, value):
        assert lock.active, "qualification must remain in the admission boundary"
        with pytest.raises(LeaseUnavailable):
            contender = coordinator.admit(
                path,
                transaction_owner=OwnerToken("qualification contender"),
                target_owner=OwnerToken("qualification target"),
            )
            contender.acquire_lease(
                admission=contender.admission,
                transaction_owner=contender._transaction_owner,
                target_owner=contender._target_owner,
                owners={role: OwnerToken(f"qualification {role.value}")
                        for role in LeaseOwner},
            )
        return real_qualify(path, value)

    monkeypatch.setattr(coordinator, "admit", admit)
    monkeypatch.setattr(module, "qualify_append", qualify)
    preflight = prepare_append_preflight(target, intent, file_lock=lock)
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(preflight)
    preflight.abort()


def test_preflight_reserved_generation_replay_is_exact_and_monotonic(tmp_path):
    from xrd_tools.io import AppendRefused, prepare_append_preflight

    current = _intent(
        tmp_path, extent=1, labels=(0,), generation=1,
        modes=("1d:default",),
    )
    preflight = prepare_append_preflight(tmp_path / "generation.nexus", current)
    assert preflight.extend(current) == preflight.snapshot
    with pytest.raises(AppendRefused, match="generation did not advance"):
        preflight.extend(_intent(
            tmp_path, extent=1, labels=(0,), generation=0,
            modes=("1d:default",),
        ))
    assert preflight.snapshot.source_generation == 1


def test_reserved_preflight_rejects_a_higher_generation_source_regression(tmp_path):
    from dataclasses import replace
    from xrd_tools.io import AppendRefused, prepare_append_preflight

    current = _intent(
        tmp_path, extent=1, labels=(0,), generation=0,
        modes=("1d:default",),
    )
    preflight = prepare_append_preflight(tmp_path / "reserved-regression.nexus", current)
    regressed = replace(
        current,
        source=replace(
            current.source,
            size=current.source.size - 1,
            generation=1,
        ),
    )
    with pytest.raises(AppendRefused, match="source size regressed"):
        preflight.extend(regressed)
    assert preflight.snapshot.source_generation == 0
    preflight.abort()
    preflight.abort()


def test_qualification_cleanup_failure_returns_exact_retry_owner(
    tmp_path, monkeypatch,
):
    from xrd_tools.io import AppendPreflightCleanupError, prepare_append_preflight
    module = importlib.import_module("xrd_tools.io.append")
    transaction_module = importlib.import_module("xrd_tools.io.output_transaction")
    real_qualify = module.qualify_append
    real_abandon = transaction_module.OutputTransaction.abandon
    calls = 0

    def fail_once(self, lease):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("cleanup fault")
        return real_abandon(self, lease)

    monkeypatch.setattr(
        module, "qualify_append",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("qualify fault")),
    )
    monkeypatch.setattr(transaction_module.OutputTransaction, "abandon", fail_once)
    target = tmp_path / "qualification-cleanup.nexus"
    intent = _intent(tmp_path, extent=1, labels=(0,), modes=("1d:default",))
    with pytest.raises(AppendPreflightCleanupError) as excinfo:
        prepare_append_preflight(target, intent)
    owner = excinfo.value.owner
    assert owner.target == target and owner.snapshot.state.value == "retryable"
    owner.retry_cleanup()
    assert owner.snapshot.state.value == "aborted"

    monkeypatch.setattr(module, "qualify_append", real_qualify)
    recovered = prepare_append_preflight(target, intent)
    recovered.abort()


def test_preflight_abort_preserves_integrity_hold_after_target_appears(tmp_path):
    from xrd_tools.io import prepare_append_preflight

    target = tmp_path / "appeared.nexus"
    preflight = prepare_append_preflight(
        target, _intent(tmp_path, extent=1, labels=(0,), modes=("1d:default",)))
    target.write_bytes(b"foreign")
    with pytest.raises(TargetChanged, match="changed before abandonment"):
        preflight.abort()
    assert preflight.snapshot.state.value == "integrity_hold"
    assert preflight.retry_cleanup().state.value == "integrity_hold"


def test_noop_preflight_rechecks_target_before_reporting_terminal(tmp_path):
    from xrd_tools.io import prepare_append_preflight
    target = tmp_path / "noop-mutation.nexus"
    intent = _intent(
        tmp_path, extent=1, labels=(0,), modes=("1d:default",))
    seed = importlib.import_module("xrd_tools.io.append").begin_same_run_lineage(intent)
    _seed_target(target, (0,), tmp_path)
    with h5py.File(target, "r+") as handle:
        del handle["entry/integrated_2d"]
        importlib.import_module("xrd_tools.io.append").commit_append_lineage(
            handle["entry"], seed, written_labels=(0,))
    preflight = prepare_append_preflight(target, intent)
    assert preflight.snapshot.disposition.value == "skip"
    target.write_bytes(b"foreign occupant")
    with pytest.raises(TargetChanged, match="changed before abandonment"):
        preflight.complete_noop()
    assert preflight.snapshot.state.value == "integrity_hold"


def _image_series_intent(tmp_path, count, *, generation):
    from xrd_tools.io import AppendImageMember, AppendIntent, AppendSource

    members = []
    for index in range(count):
        path = tmp_path / f"series-{index:04d}.tif"
        if not path.exists():
            path.write_bytes(bytes((index + 1,)) * (index + 3))
        stat = path.stat()
        members.append(AppendImageMember(
            path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
            source_start=index, source_stop=index + 1, ordinal=index,
        ))
    return AppendIntent(
        entry="entry", source_base=str(tmp_path),
        source_identity="image-series:scan-17", science_fingerprint="science-v1",
        modes=("1d:default",),
        source=AppendSource(
            path=str(tmp_path / "series-0000.tif"),
            adapter_id="image-series:.tif",
            size=sum(member.size for member in members),
            mtime_ns=max(member.mtime_ns for member in members),
            extent=count, image_members=tuple(members), generation=generation,
        ),
        labels=tuple(range(count)),
    )


def _commit_image_series_target(target, intent):
    module = importlib.import_module("xrd_tools.io.append")
    decision = module.qualify_append(target, intent)
    assert decision.disposition is module.AppendDisposition.WRITE
    if not target.exists():
        _seed_target(target, intent.labels, Path(intent.source_base))
    else:
        with h5py.File(target, "r+") as handle:
            _write_result_rows(handle["entry"], decision.write_labels)
    with h5py.File(target, "r+") as handle:
        module.commit_append_lineage(
            handle["entry"], decision, written_labels=decision.write_labels,
        )


def _decode_image_series_prefix(target):
    from xrd_tools.io import decode_committed_append_prefix

    with h5py.File(target, "r") as handle:
        return decode_committed_append_prefix(handle)


def test_committed_prefix_decoder_returns_exact_cumulative_image_series_intent(
    tmp_path,
):
    io_module = importlib.import_module("xrd_tools.io")
    target = tmp_path / "decoder.nexus"
    first = _image_series_intent(tmp_path, 2, generation=0)
    final = _image_series_intent(tmp_path, 3, generation=1)
    _commit_image_series_target(target, first)
    _commit_image_series_target(target, final)

    assert "AppendCommittedPrefix" in io_module.__all__
    assert "decode_committed_append_prefix" in io_module.__all__
    with h5py.File(target, "r") as handle:
        handle_id = handle.id
        prefix = io_module.decode_committed_append_prefix(handle)
        assert handle.id is handle_id and handle.id.valid
        assert "entry" in handle

    assert type(prefix) is io_module.AppendCommittedPrefix
    assert prefix.target == os.path.normcase(os.path.abspath(target))
    assert prefix.committed_labels == (0, 1, 2)
    assert prefix.intent.entry == "entry"
    assert prefix.intent.source_base == os.path.normcase(os.path.abspath(tmp_path))
    assert prefix.intent.source_identity == "image-series:scan-17"
    assert prefix.intent.science_fingerprint == "science-v1"
    assert prefix.intent.modes == ("1d:default",)
    assert prefix.intent.source.generation == 1
    assert tuple(member.path for member in prefix.intent.source.image_members) == tuple(
        os.path.normcase(os.path.abspath(tmp_path / f"series-{index:04d}.tif"))
        for index in range(3)
    )
    assert tuple(
        (member.source_start, member.source_stop, member.ordinal)
        for member in prefix.intent.source.image_members
    ) == ((0, 1, 0), (1, 2, 1), (2, 3, 2))
    assert prefix.lineage_json == json.dumps(
        json.loads(prefix.lineage_json), sort_keys=True, separators=(",", ":"),
    )
    with pytest.raises((AttributeError, TypeError)):
        prefix.target = "different"


def test_prefix_bound_preflight_refuses_disappeared_target_instead_of_first_run_write(
    tmp_path,
):
    from xrd_tools.io import AppendRefused, prepare_append_preflight

    target = tmp_path / "disappeared.nexus"
    _commit_image_series_target(
        target, _image_series_intent(tmp_path, 2, generation=0),
    )
    prefix = _decode_image_series_prefix(target)
    target.unlink()

    with pytest.raises(AppendRefused, match="committed Append prefix target disappeared"):
        prepare_append_preflight(
            target, _image_series_intent(tmp_path, 3, generation=1),
            committed_prefix=prefix,
        )
    assert not target.exists()
    _assert_lease_available(target)


def test_prefix_bound_preflight_refuses_divergent_same_label_lineage(tmp_path):
    from dataclasses import replace
    from xrd_tools.io import (
        AppendCommittedPrefix, AppendRefused, decode_committed_append_prefix,
        prepare_append_preflight,
    )

    target = tmp_path / "divergent.nexus"
    _commit_image_series_target(
        target, _image_series_intent(tmp_path, 2, generation=0),
    )
    _commit_image_series_target(
        target, _image_series_intent(tmp_path, 3, generation=1),
    )
    prefix = _decode_image_series_prefix(target)
    with h5py.File(target, "r+") as handle:
        dataset = handle["entry/reduction/config/append_lineage"]
        lineage = json.loads(dataset[()].decode())
        lineage["epochs"][0]["source"]["generation"] = 1
        del handle["entry/reduction/config/append_lineage"]
        handle["entry/reduction/config"].create_dataset(
            "append_lineage",
            data=json.dumps(lineage, sort_keys=True, separators=(",", ":")),
        )
    before = target.read_bytes()

    with pytest.raises(AppendRefused, match="committed Append epochs diverged"):
        prepare_append_preflight(
            target, _image_series_intent(tmp_path, 4, generation=2),
            committed_prefix=prefix,
        )
    assert target.read_bytes() == before
    _assert_lease_available(target)

    for mutation in (
        "bool-version", "float-label", "bool-member-ordinal", "bool-digest",
        "string-image-members", "null-image-member",
    ):
        typed_target = tmp_path / f"typed-{mutation}.nexus"
        _commit_image_series_target(
            typed_target, _image_series_intent(tmp_path, 2, generation=0),
        )
        typed_prefix = _decode_image_series_prefix(typed_target)
        with h5py.File(typed_target, "r+") as handle:
            dataset = handle["entry/reduction/config/append_lineage"]
            typed_lineage = json.loads(dataset[()].decode())
            if mutation == "bool-version":
                typed_lineage["version"] = True
            elif mutation == "float-label":
                typed_lineage["epochs"][0]["labels"][0] = 0.0
            elif mutation == "bool-member-ordinal":
                typed_lineage["epochs"][0]["source"]["image_members"][0][
                    "ordinal"
                ] = False
            elif mutation == "bool-digest":
                typed_lineage["epochs"][0]["source"]["digest"] = True
            elif mutation == "string-image-members":
                typed_lineage["epochs"][0]["source"]["image_members"] = "x"
            else:
                typed_lineage["epochs"][0]["source"]["image_members"] = [None]
            del handle["entry/reduction/config/append_lineage"]
            handle["entry/reduction/config"].create_dataset(
                "append_lineage", data=json.dumps(
                    typed_lineage, sort_keys=True, separators=(",", ":"),
                ),
            )
        typed_before = typed_target.read_bytes()
        with h5py.File(typed_target, "r") as handle:
            with pytest.raises(ValueError, match="not committed|exact JSON"):
                decode_committed_append_prefix(handle)
        with pytest.raises(AppendRefused, match="not committed|exact JSON"):
            prepare_append_preflight(
                typed_target, _image_series_intent(tmp_path, 3, generation=1),
                committed_prefix=typed_prefix,
            )
        assert typed_target.read_bytes() == typed_before
        _assert_lease_available(typed_target)

    for mutation, value in (
        ("schema-version-float", 2.5),
        ("schema-version-inf", float("inf")),
        ("schema-version-nan", float("nan")),
        ("schema-version-bool", True),
        ("schema-version-text", str(PROCESSED_SCHEMA_VERSION)),
        ("schema-version-array", np.asarray(
            [PROCESSED_SCHEMA_VERSION], dtype=np.int64,
        )),
        ("source-base-int", 7),
        ("schema-name-int", 7),
        ("schema-name-array", np.asarray(
            [PROCESSED_SCHEMA_NAME.encode("utf-8")],
        )),
        ("lineage-scalar-int", 7),
    ):
        scalar_target = tmp_path / f"scalar-{mutation}.nexus"
        _commit_image_series_target(
            scalar_target, _image_series_intent(tmp_path, 2, generation=0),
        )
        scalar_prefix = _decode_image_series_prefix(scalar_target)
        with h5py.File(scalar_target, "r+") as handle:
            entry = handle["entry"]
            if mutation == "source-base-int":
                entry.attrs[SOURCE_BASE_ATTR] = value
            elif mutation.startswith("schema-name"):
                entry.attrs[SCHEMA_NAME_ATTR] = value
            elif mutation == "lineage-scalar-int":
                del entry["reduction/config/append_lineage"]
                entry["reduction/config"].create_dataset(
                    "append_lineage", data=value,
                )
            else:
                entry.attrs[SCHEMA_VERSION_ATTR] = value
        scalar_before = scalar_target.read_bytes()
        expected_error = (
            "current xdart .nexus record" if mutation.startswith("schema-")
            else "source base" if mutation == "source-base-int"
            else "lineage scalar"
        )
        with h5py.File(scalar_target, "r") as handle:
            with pytest.raises(ValueError, match=expected_error):
                decode_committed_append_prefix(handle)
        with pytest.raises(AppendRefused, match=expected_error):
            prepare_append_preflight(
                scalar_target, _image_series_intent(tmp_path, 3, generation=1),
                committed_prefix=scalar_prefix,
            )
        assert scalar_target.read_bytes() == scalar_before
        _assert_lease_available(scalar_target)

    primary_target = tmp_path / "primary-mode-int.nexus"
    _commit_image_series_target(
        primary_target, _image_series_intent(tmp_path, 2, generation=0),
    )
    original_prefix = _decode_image_series_prefix(primary_target)
    primary_lineage = json.loads(original_prefix.lineage_json)
    primary_lineage["modes"] = ["1d:7"]
    primary_intent = replace(original_prefix.intent, modes=("1d:7",))
    primary_prefix = AppendCommittedPrefix(
        str(primary_target), primary_intent, json.dumps(
            primary_lineage, sort_keys=True, separators=(",", ":"),
        ),
    )
    with h5py.File(primary_target, "r+") as handle:
        entry = handle["entry"]
        entry["integrated_1d"].attrs[PRIMARY_MODE_ATTR] = 7
        entry["integrated_1d"].attrs[MULTI_RESULT_MODES_ATTR] = ["q_total"]
        with pytest.raises(ValueError, match="requires one bounded text scalar"):
            read_current_mode_layout(entry["integrated_1d"], "1d")
        del entry["reduction/config/append_lineage"]
        entry["reduction/config"].create_dataset(
            "append_lineage", data=primary_prefix.lineage_json,
        )
    primary_before = primary_target.read_bytes()
    with h5py.File(primary_target, "r") as handle:
        with pytest.raises(ValueError, match="current xdart .nexus record"):
            decode_committed_append_prefix(handle)
    successor = replace(
        _image_series_intent(tmp_path, 3, generation=1), modes=("1d:7",),
    )
    with pytest.raises(AppendRefused, match="current xdart .nexus record"):
        prepare_append_preflight(
            primary_target, successor, committed_prefix=primary_prefix,
        )
    assert primary_target.read_bytes() == primary_before
    _assert_lease_available(primary_target)


@pytest.mark.parametrize(
    "case,dimension,primary,modes,subgroup,error",
    (
        (
            "primary-1d", "1d", "qip_qoop", ("1d:qip_qoop",), None,
            "unknown canonical 1d mode key",
        ),
        (
            "non-primary-1d", "1d", "q_total",
            ("1d:q_total", "1d:qip_qoop"), "qip_qoop", "malformed current mode inventory",
        ),
        (
            "primary-2d", "2d", "q_total", ("2d:q_total",), None,
            "unknown canonical 2d mode key",
        ),
        (
            "non-primary-2d", "2d", "qip_qoop",
            ("2d:qip_qoop", "2d:q_total"), "q_total", "malformed current mode inventory",
        ),
    ),
)
def test_prefix_bound_preflight_refuses_dimension_incompatible_modes(
    tmp_path, case, dimension, primary, modes, subgroup, error,
):
    from dataclasses import replace
    from xrd_tools.io import (
        AppendCommittedPrefix, AppendRefused, decode_committed_append_prefix,
        prepare_append_preflight,
    )

    target = tmp_path / f"wrong-dimension-{case}.nexus"
    _commit_image_series_target(
        target, _image_series_intent(tmp_path, 2, generation=0),
    )
    original = _decode_image_series_prefix(target)
    lineage = json.loads(original.lineage_json)
    lineage["modes"] = list(modes)
    intent = replace(original.intent, modes=modes)
    prefix = AppendCommittedPrefix(
        str(target), intent,
        json.dumps(lineage, sort_keys=True, separators=(",", ":")),
    )
    with h5py.File(target, "r+") as handle:
        entry = handle["entry"]
        top = entry[f"integrated_{dimension}"]
        if subgroup is not None:
            # Keep the child payload schema-valid so only its mode identity is
            # wrong. Copying also preserves independent row-dataset ownership.
            top.copy(top, subgroup)
        top.attrs[PRIMARY_MODE_ATTR] = primary
        top.attrs[MULTI_RESULT_MODES_ATTR] = [mode.split(":", 1)[1] for mode in modes]
        with pytest.raises(ValueError, match=error):
            read_current_mode_layout(top, dimension)
        del entry["reduction/config/append_lineage"]
        entry["reduction/config"].create_dataset(
            "append_lineage", data=prefix.lineage_json,
        )
    before = target.read_bytes()

    with h5py.File(target, "r") as handle:
        with pytest.raises(ValueError, match="current xdart .nexus record"):
            decode_committed_append_prefix(handle)
    successor = replace(
        _image_series_intent(tmp_path, 3, generation=1), modes=modes,
    )
    with pytest.raises(AppendRefused, match="current xdart .nexus record"):
        prepare_append_preflight(
            target, successor, committed_prefix=prefix,
        )
    assert target.read_bytes() == before
    _assert_lease_available(target)


def test_prefix_bound_preflight_accepts_concurrent_exact_successor_as_noop(
    tmp_path,
):
    from xrd_tools.io import AppendDisposition, AppendPreflightState
    from xrd_tools.io import prepare_append_preflight

    target = tmp_path / "concurrent-successor.nexus"
    _commit_image_series_target(
        target, _image_series_intent(tmp_path, 2, generation=0),
    )
    prefix = _decode_image_series_prefix(target)
    successor = _image_series_intent(tmp_path, 3, generation=1)
    _commit_image_series_target(target, successor)
    before = target.read_bytes()

    preflight = prepare_append_preflight(
        target, successor, committed_prefix=prefix,
    )
    assert preflight.snapshot.disposition is AppendDisposition.SKIP
    assert preflight.snapshot.skip_labels == (0, 1, 2)
    assert preflight.complete_noop().state is AppendPreflightState.NOOP
    assert target.read_bytes() == before
    _assert_lease_available(target)


def test_reserved_prefix_bound_preflight_extend_retains_anchor(tmp_path, monkeypatch):
    from xrd_tools.io import prepare_append_preflight

    module = importlib.import_module("xrd_tools.io.append")
    target = tmp_path / "reserved-anchor.nexus"
    _commit_image_series_target(
        target, _image_series_intent(tmp_path, 2, generation=0),
    )
    prefix = _decode_image_series_prefix(target)
    preflight = prepare_append_preflight(
        target, _image_series_intent(tmp_path, 3, generation=1),
        committed_prefix=prefix,
    )
    before = target.read_bytes()
    seen = []
    real_qualify = module.qualify_append

    def qualify(path, intent, *, committed_prefix=None):
        seen.append(committed_prefix)
        return real_qualify(path, intent, committed_prefix=committed_prefix)

    monkeypatch.setattr(module, "qualify_append", qualify)
    snapshot = preflight.extend(_image_series_intent(tmp_path, 4, generation=2))
    assert seen == [prefix]
    assert snapshot.skip_labels == (0, 1) and snapshot.write_labels == (2, 3)
    preflight.abort()
    assert target.read_bytes() == before
    _assert_lease_available(target)


def _seed_existing_append_target(target, source_base, intent):
    from xrd_tools.core.scan import Scan, ScanFrame
    from xrd_tools.reduction import FrameReduction, NexusSink, ReductionPlan, ReductionResult

    sink = NexusSink(
        target, overwrite=True, atomic=False, flush_every=None,
        source_base=source_base, same_run_intent=intent,
    )
    sink.begin(Scan("seed", []), ReductionPlan(integration_2d=None))
    for label in intent.labels:
        sink.write(
            ScanFrame(label), FrameReduction(label, result_1d=_r1(label)),
        )
    sink.finish(ReductionResult("seed", {}, len(intent.labels)))


def test_existing_append_sink_qualifies_once_and_reuses_owner_for_extension(
    tmp_path, monkeypatch,
):
    from xrd_tools.core.scan import Scan, ScanFrame
    from xrd_tools.io import AppendDisposition
    from xrd_tools.reduction import FrameReduction, NexusSink, ReductionPlan, ReductionResult
    import xrd_tools.reduction.core as reduction_core

    target = tmp_path / "lazy-existing.nexus"
    first = _intent(
        tmp_path, extent=1, labels=(0,), modes=("1d:default",),
    )
    _seed_existing_append_target(target, tmp_path, first)
    second = _intent(
        tmp_path, extent=2, labels=(0, 1), modes=("1d:default",),
    )
    calls = []
    real_prepare = reduction_core.prepare_append_preflight

    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return real_prepare(*args, **kwargs)

    monkeypatch.setattr(reduction_core, "prepare_append_preflight", prepare)
    sink = NexusSink.for_existing_append(
        target, second, atomic=False, flush_every=None, source_base=tmp_path,
    )
    assert calls == []
    sink.begin(Scan("existing", []), ReductionPlan(integration_2d=None))
    owner = (sink, sink._transaction, sink._lease, sink.extension_owner)
    assert len(calls) == 1
    sink.write(ScanFrame(1), FrameReduction(1, result_1d=_r1(1)))
    sink.commit_epoch(ReductionResult("existing", {}, 1))
    third = _intent(
        tmp_path, extent=3, labels=(0, 1, 2), generation=1,
        modes=("1d:default",),
    )
    assert sink.extend_live(sink.extension_owner, third).disposition is AppendDisposition.WRITE
    sink.write(ScanFrame(2), FrameReduction(2, result_1d=_r1(2)))
    sink.finish(ReductionResult("existing", {}, 2))

    assert len(calls) == 1
    assert sink is owner[0] and sink._transaction is owner[1]
    assert sink._lease is owner[2] and owner[3] is not None
    with h5py.File(target, "r") as handle:
        assert tuple(handle["entry/integrated_1d/frame_index"][()]) == (0, 1, 2)


def test_existing_append_refusal_is_typed_preserves_target_and_has_no_owner(
    tmp_path, monkeypatch,
):
    from xrd_tools.core.scan import Scan
    from xrd_tools.io import AppendRefused
    from xrd_tools.reduction import NexusSink, ReductionPlan
    import xrd_tools.reduction.core as reduction_core

    target = tmp_path / "lazy-refusal.nexus"
    first = _intent(
        tmp_path, extent=1, labels=(0,), modes=("1d:default",),
    )
    _seed_existing_append_target(target, tmp_path, first)
    before = target.read_bytes()
    refused = _intent(
        tmp_path, extent=2, labels=(0, 1), science="different-science",
        modes=("1d:default",),
    )
    calls = 0
    real_prepare = reduction_core.prepare_append_preflight

    def prepare(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_prepare(*args, **kwargs)

    monkeypatch.setattr(reduction_core, "prepare_append_preflight", prepare)
    sink = NexusSink.for_existing_append(target, refused, source_base=tmp_path)
    assert sink.abort(None) is None
    with pytest.raises(AppendRefused, match="foreign Append identity"):
        sink.begin(Scan("refused", []), ReductionPlan(integration_2d=None))
    assert calls == 1 and sink.append_preflight is None
    assert sink.abort(None) is None and sink.abort(None) is None
    with pytest.raises(RuntimeError, match="qualification already failed"):
        sink.begin(Scan("refused", []), ReductionPlan(integration_2d=None))
    assert calls == 1 and target.read_bytes() == before
    sink.abort(None)
    _assert_lease_available(target)


def test_existing_append_retryable_cleanup_stays_in_sink_and_abort_retries(
    tmp_path, monkeypatch,
):
    from xrd_tools.core.scan import Scan
    from xrd_tools.io import AppendPreflightCleanupError, AppendPreflightState
    from xrd_tools.reduction import NexusSink, ReductionPlan
    import xrd_tools.io.append as append_module
    import xrd_tools.io.output_transaction as transaction_module
    import xrd_tools.reduction.core as reduction_core

    target = tmp_path / "lazy-retryable.nexus"
    intent = _intent(
        tmp_path, extent=1, labels=(0,), modes=("1d:default",),
    )
    _seed_existing_append_target(target, tmp_path, intent)
    real_abandon = transaction_module.OutputTransaction.abandon
    attempts = {"prepare": 0, "abandon": 0, "open": 0}

    def fail_qualification(*_args, **_kwargs):
        raise OSError("qualification fault")

    def fail_abandon_once(owner, lease):
        attempts["abandon"] += 1
        if attempts["abandon"] == 1:
            raise OSError("cleanup fault")
        return real_abandon(owner, lease)

    real_prepare = reduction_core.prepare_append_preflight

    def prepare(*args, **kwargs):
        attempts["prepare"] += 1
        return real_prepare(*args, **kwargs)

    monkeypatch.setattr(append_module, "qualify_append", fail_qualification)
    monkeypatch.setattr(transaction_module.OutputTransaction, "abandon", fail_abandon_once)
    monkeypatch.setattr(reduction_core, "prepare_append_preflight", prepare)
    monkeypatch.setattr(
        reduction_core, "open_nexus_writer",
        lambda *_a, **_k: attempts.__setitem__("open", attempts["open"] + 1),
    )
    sink = NexusSink.for_existing_append(target, intent, source_base=tmp_path)
    with pytest.raises(AppendPreflightCleanupError) as excinfo:
        sink.begin(Scan("retryable", []), ReductionPlan(integration_2d=None))
    owner = excinfo.value.owner
    assert sink.append_preflight is owner
    assert owner.snapshot.state is AppendPreflightState.RETRYABLE
    assert attempts == {"prepare": 1, "abandon": 1, "open": 0}
    assert sink.abort(None) is None
    assert owner.snapshot.state is AppendPreflightState.ABORTED
    assert attempts == {"prepare": 1, "abandon": 2, "open": 0}
    assert sink.abort(None) is None and sink.append_preflight is owner
    _assert_lease_available(target)


def test_existing_append_integrity_hold_abort_never_reports_settled(
    tmp_path, monkeypatch,
):
    from xrd_tools.core.scan import Scan
    from xrd_tools.io import AppendPreflightCleanupError, AppendPreflightState
    from xrd_tools.reduction import NexusSink, ReductionPlan
    import xrd_tools.io.append as append_module

    target = tmp_path / "lazy-integrity-hold.nexus"
    intent = _intent(
        tmp_path, extent=1, labels=(0,), modes=("1d:default",),
    )
    _seed_existing_append_target(target, tmp_path, intent)

    def mutate_then_fail(path, *_args, **_kwargs):
        Path(path).write_bytes(b"foreign replacement")
        raise OSError("qualification fault")

    monkeypatch.setattr(append_module, "qualify_append", mutate_then_fail)
    sink = NexusSink.for_existing_append(target, intent, source_base=tmp_path)
    with pytest.raises(AppendPreflightCleanupError) as excinfo:
        sink.begin(Scan("integrity", []), ReductionPlan(integration_2d=None))
    owner = excinfo.value.owner
    assert sink.append_preflight is owner
    assert owner.snapshot.state is AppendPreflightState.INTEGRITY_HOLD
    for _ in range(2):
        with pytest.raises(RuntimeError, match="cleanup remains unsettled: integrity_hold"):
            sink.abort(None)
        assert sink.append_preflight is owner
        assert owner.snapshot.state is AppendPreflightState.INTEGRITY_HOLD
