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
    PROCESSED_SCHEMA_NAME,
    PROCESSED_SCHEMA_VERSION,
    PRIMARY_MODE_ATTR,
    SCHEMA_NAME_ATTR,
    SCHEMA_VERSION_ATTR,
    SOURCE_BASE_ATTR,
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


def test_terminal_commit_rechecks_exact_bytes_not_only_stat_receipt(
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
    target.write_bytes(b"AAAA")
    transaction.seal_stream_terminal(attempt, lease=lease)
    accepted = target.stat()
    target.write_bytes(b"BBBB")
    os.utime(target, ns=(accepted.st_atime_ns, accepted.st_mtime_ns))
    monkeypatch.setattr(module, "_stream_stat_matches", lambda *_args: True)

    with pytest.raises(TargetChanged, match="terminal seal changed"):
        transaction.commit_stream(attempt, lease=lease)
    assert transaction.snapshot().phase is TransactionPhase.INTEGRITY_HOLD


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

    def fail_first_destination_observation(path):
        nonlocal failed
        if Path(path) == partial and partial.exists() and not target.exists() and not failed:
            failed = True
            raise OSError("post-rename observation failed")
        return real_capture(path)

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

    def fail_once(path):
        nonlocal failed
        if Path(path) == partial and partial.exists() and not target.exists() and not failed:
            failed = True
            raise OSError("post-rename observation failed")
        return real_capture(path)

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

    def fail_once(path):
        nonlocal failed
        if Path(path) == partial and partial.exists() and not target.exists() and not failed:
            failed = True
            raise OSError("post-rename observation failed")
        return real_capture(path)

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
    with h5py.File(target, "w") as handle:
        handle.create_group("entry")
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
        assert tuple(handle["entry/integrated_1d/frame_index"][()]) == ()
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

    def fail_once(proof):
        if not failures:
            failures.append("observation")
            raise OSError("post-close semantic observation unavailable")
        return real_verify(proof)

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
    existing.write_bytes(b"preserved")
    sink = NexusSink(
        existing,
        same_run_intent=_intent(tmp_path, extent=1, labels=(0,)),
    )
    with pytest.raises(ValueError, match="exact Append preflight"):
        sink.begin(Scan("existing", []), ReductionPlan())
    assert existing.read_bytes() == b"preserved"
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
            generation=0, modes=("1d:default", "2d:default")):
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
        entry = handle.create_group("entry")
        entry.attrs[SCHEMA_NAME_ATTR] = PROCESSED_SCHEMA_NAME
        entry.attrs[SCHEMA_VERSION_ATTR] = PROCESSED_SCHEMA_VERSION
        entry.attrs[SOURCE_BASE_ATTR] = str(source_base)
        for name in ("integrated_1d", "integrated_2d"):
            group = entry.create_group(name)
            group.create_dataset("frame_index", data=np.asarray(labels, dtype=np.int64))


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
        for name in ("integrated_1d", "integrated_2d"):
            del handle[f"entry/{name}/frame_index"]
            handle[f"entry/{name}"].create_dataset(
                "frame_index", data=np.arange(5, dtype=np.int64))
        commit_lineage(handle["entry"], decision, written_labels=(2, 3, 4))

    third = _intent(tmp_path, extent=6, labels=tuple(range(6)))
    decision = qualify(target, third)
    assert decision.disposition is Disposition.WRITE
    with h5py.File(target, "r+") as handle:
        for name in ("integrated_1d", "integrated_2d"):
            del handle[f"entry/{name}/frame_index"]
            handle[f"entry/{name}"].create_dataset(
                "frame_index", data=np.arange(6, dtype=np.int64))
        commit_lineage(handle["entry"], decision, written_labels=(5,))

    final = qualify(target, _intent(tmp_path, extent=6, labels=tuple(range(6))))
    assert final.disposition is Disposition.SKIP


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
        for name in ("integrated_1d", "integrated_2d"):
            del handle[f"entry/{name}/frame_index"]
            handle[f"entry/{name}"].create_dataset(
                "frame_index", data=np.arange(3, dtype=np.int64),
            )
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
    target = tmp_path / "same-run.nxs"
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


def test_empty_owned_writer_adopts_first_lineage_without_readmit_or_reopen(
    tmp_path, monkeypatch,
):
    from xdart.modules.reduction import open_live_scan_nexus_session
    from xrd_tools.io import AppendRefused
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
    target = tmp_path / "empty-owned.nexus"
    live = _live_scan(target, None, ())
    session = open_live_scan_nexus_session(live, replace=True)

    session.flush(force=True)
    writer = session.sink._writer
    cursors = writer._row_cursors
    assert target.exists()
    assert writer.append_decision is None
    assert len(admits) == len(opens) == 1
    with pytest.raises(LeaseUnavailable):
        contender = coordinator.admit(
            target,
            transaction_owner=OwnerToken("empty contender transaction"),
            target_owner=OwnerToken("empty contender target"),
        )
        contender.acquire_lease(
            admission=contender.admission,
            transaction_owner=contender._transaction_owner,
            target_owner=contender._target_owner,
            owners={role: OwnerToken(f"empty contender {role.value}")
                    for role in LeaseOwner},
        )
    admitted_with_contender = len(admits)

    with pytest.raises(AppendRefused, match="absent source has no labels"):
        session.extend(_intent(
            tmp_path, extent=1, labels=(), generation=0,
            modes=("1d:default",),
        ))
    assert session.same_run_intent is None
    assert session.sink.same_run_intent is None
    assert writer.append_decision is None

    intent = _intent(
        tmp_path, extent=1, labels=(0,), generation=0,
        modes=("1d:default",),
    )
    live._same_run_intent = intent
    live.frames.add(0)
    session.flush(force=True)
    assert session.sink._writer is writer
    assert writer._row_cursors is cursors
    assert len(admits) == admitted_with_contender
    assert opens == [target]
    session.finish(finalize=True)

    with h5py.File(target, "r") as handle:
        assert tuple(handle["entry/integrated_1d/frame_index"][()]) == (0,)
        lineage = json.loads(
            handle["entry/reduction/config/append_lineage"][()].decode())
    assert lineage["state"] == "committed"
    assert lineage["epochs"][-1]["labels"] == [0]
    _assert_lease_available(target)


def test_intent_bound_zero_frame_replace_owns_target_before_first_frame(
    tmp_path,
):
    """Production binds a valid lineage before requesting the empty Replace
    owner. That ordering must still open one writer/lease and close the path in
    which a foreign target could appear before frame zero."""
    from xdart.modules.reduction import open_live_scan_nexus_session

    target = tmp_path / "intent-bound-empty.nexus"
    intent = _intent(
        tmp_path,
        extent=1,
        labels=(0,),
        generation=0,
        modes=("1d:default",),
    )
    live = _live_scan(target, intent, ())
    session = open_live_scan_nexus_session(live, replace=True)

    session.flush(force=True)

    assert session._begun
    assert session.sink._transaction is not None
    assert session.sink._writer is not None
    assert target.exists()
    with pytest.raises(FileExistsError):
        target.open("xb").close()

    session.abort()
    _assert_lease_available(target)


def test_same_run_extension_rollback_restores_prior_bytes(tmp_path):
    from xrd_tools.core.scan import Scan, ScanFrame
    from xrd_tools.reduction import FrameReduction, NexusSink, ReductionPlan, ReductionResult
    target = tmp_path / "rollback.nxs"
    prior = b"immutable prior bytes"
    target.write_bytes(prior)
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
    target = tmp_path / "epoch-rollback.nxs"
    original = b"pre-run target"
    target.write_bytes(original)
    first = _intent(
        tmp_path, extent=1, labels=(0,), generation=0,
        modes=("1d:default",),
    )
    sink = NexusSink(
        target, overwrite=True, source_base=tmp_path,
        same_run_intent=first, flush_every=None,
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
    target = tmp_path / "cross-run.nxs"
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


def test_replace_session_selects_all_once_then_only_unpersisted_rows(tmp_path):
    from xdart.modules.reduction import open_live_scan_nexus_session

    target = tmp_path / "replace-dirty.nexus"
    intent = _intent(
        tmp_path, extent=3, labels=(0, 1, 2), modes=("1d:default",))
    live = _live_scan(target, intent, (0, 1, 2))
    session = open_live_scan_nexus_session(live, replace=True)
    with pytest.raises(TypeError, match="authority cannot be copied"):
        copy.copy(session)
    assert session._select() == [0, 1, 2]
    live.frames._persisted.update((0, 1))
    session._begun = True
    assert session._select() == [2]


def test_one_shot_live_scan_wrapper_preserves_primary_over_abort_failure(monkeypatch):
    module = importlib.import_module("xdart.modules.reduction")

    class FailingSession:
        def flush(self, **_kwargs):
            raise ValueError("write primary")

        def abort(self):
            raise OSError("abort cleanup")

    monkeypatch.setattr(
        module, "open_live_scan_nexus_session", lambda *_a, **_k: FailingSession(),
    )
    with pytest.raises(ValueError, match="write primary") as excinfo:
        module.write_live_scan_to_nexus(object())
    assert isinstance(excinfo.value.__cause__, OSError)
    assert str(excinfo.value.__cause__) == "abort cleanup"


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


def test_first_run_stop_commits_only_the_written_source_prefix(tmp_path):
    from xdart.modules.reduction import open_live_scan_nexus_session

    target = tmp_path / "first-stop.nexus"
    intent = _intent(
        tmp_path, extent=3, labels=(0, 1, 2), generation=0,
        modes=("1d:default",),
    )
    session = open_live_scan_nexus_session(
        _live_scan(target, intent, (0, 1)), replace=True)
    session.flush(force=True)
    session.finish(finalize=False)

    with h5py.File(target, "r") as handle:
        assert tuple(handle["entry/integrated_1d/frame_index"][()]) == (0, 1)
        lineage = json.loads(
            handle["entry/reduction/config/append_lineage"][()].decode())
        assert "stitched_1d" not in handle["entry"]
    assert lineage["state"] == "committed"
    assert lineage["epochs"][-1]["labels"] == [0, 1]
    assert lineage["epochs"][-1]["source"]["extent"] == 2
    assert session.sink._transaction_owners is None
    _assert_lease_available(target)


def test_stop_never_commits_a_partially_complete_append_label(tmp_path):
    from xdart.modules.reduction import open_live_scan_nexus_session

    target = tmp_path / "pending-mode-stop.nexus"
    intent = _intent(
        tmp_path, extent=1, labels=(0,), generation=0,
        modes=("1d:default", "2d:default"),
    )
    live = _live_scan(target, intent, (0,))
    live.skip_2d = False
    session = open_live_scan_nexus_session(live, replace=True)
    session.flush(force=True)
    session.finish(finalize=False)

    assert not target.exists()
    assert session.sink._transaction_owners is None


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


def test_cross_run_stop_after_epoch_commit_uses_only_current_prefix(tmp_path):
    from xrd_tools.core.scan import Scan, ScanFrame
    from xrd_tools.io import prepare_append_preflight
    from xrd_tools.reduction import FrameReduction, NexusSink, ReductionPlan, ReductionResult
    from xdart.modules.reduction import open_live_scan_nexus_session

    target = tmp_path / "cross-stop.nexus"
    first = _intent(
        tmp_path, extent=1, labels=(0,), generation=0,
        modes=("1d:default",),
    )
    seed = NexusSink(
        target, overwrite=True, source_base=tmp_path,
        same_run_intent=first, flush_every=None,
    )
    seed.begin(Scan("seed", []), ReductionPlan(integration_2d=None))
    seed.write(ScanFrame(0), FrameReduction(0, result_1d=_r1(0)))
    seed.finish(ReductionResult("seed", {}, 1))

    second = _intent(
        tmp_path, extent=2, labels=(0, 1), generation=0,
        modes=("1d:default",),
    )
    live = _live_scan(target, second, (1,))
    preflight = prepare_append_preflight(target, second)
    session = open_live_scan_nexus_session(live, append_preflight=preflight)
    session.flush(force=True)
    session.commit_epoch()
    assert session._written == []

    third = _intent(
        tmp_path, extent=4, labels=(0, 1, 2, 3), generation=1,
        modes=("1d:default",),
    )
    session.extend(third)
    live.frames.add(2)
    session.flush(force=True)  # public extend synchronizes host observation
    session.finish(finalize=False)

    with h5py.File(target, "r") as handle:
        assert tuple(handle["entry/integrated_1d/frame_index"][()]) == (0, 1, 2)
        lineage = json.loads(
            handle["entry/reduction/config/append_lineage"][()].decode())
        assert "stitched_1d" not in handle["entry"]
    assert lineage["state"] == "committed"
    assert lineage["epochs"][-1]["labels"] == [2]
    assert lineage["epochs"][-1]["source"]["extent"] == 3
    assert session.sink._transaction_owners is None
    _assert_lease_available(target)


def test_dynamic_accounting_promotes_only_committed_h23_epochs(tmp_path):
    """The public LiveScan session, not a test-only writer helper, owns the
    attempt-to-lineage boundary: epoch A is canonical; flushed but aborted B
    never becomes an H10 durable receipt."""
    from xdart.modules.reduction import open_live_scan_nexus_session
    from xrd_tools.session import (
        DynamicAccountingLimits,
        DynamicFrameIdentity,
        DynamicRunAccounting,
    )

    target = tmp_path / "dynamic-epoch-accounting.nexus"
    nexus_target = f"nexus:{target}"
    ledger = StageLedger(
        required_modes=(ResultMode.one_d(),),
        targets_by_mode={ResultMode.one_d(): (nexus_target,)},
    )
    accounting = DynamicRunAccounting(
        ledger,
        run_generation=1,
        limits=DynamicAccountingLimits(
            max_groups=2, max_attempts_per_frame=4, max_outstanding=4,
        ),
    )

    def stage(label, source_revision):
        key = DynamicFrameIdentity("growing-master.h5", label)
        accounting.discover(
            key, group="detector", ordinal=label, output_label=label,
        )
        attempt = accounting.begin_attempt(
            key, source_revision=source_revision,
        )
        accounting.record_enqueued(attempt)
        accounting.record_accepted(attempt)
        accounting.record_completed(attempt, produced=(ResultMode.one_d(),))
        accounting.record_written(attempt, modes=(ResultMode.one_d(),))
        return key, attempt

    first = _intent(
        tmp_path, extent=1, labels=(0,), generation=0,
        modes=("1d:default",),
    )
    live = _live_scan(target, first, (0,))
    key0, attempt0 = stage(0, 1)
    session = open_live_scan_nexus_session(
        live, replace=True, accounting=accounting.writer_boundary,
    )
    session.flush(force=True)
    assert ledger.snapshot().accepted == ledger.snapshot().completed == frozenset((0,))
    assert (0, ResultMode.one_d()) in ledger.snapshot().written
    assert ledger.snapshot().durable == frozenset()
    assert accounting.snapshot().pending_durable == frozenset((
        (key0, ResultMode.one_d(), nexus_target),
    ))
    session.commit_epoch()
    epoch_a = target.read_bytes()
    assert accounting.snapshot().durable_attempts[
        (key0, ResultMode.one_d(), nexus_target)
    ] == attempt0

    second = _intent(
        tmp_path, extent=2, labels=(0, 1), generation=1,
        modes=("1d:default",),
    )
    session.extend(second)
    live.frames.add(1)
    key1, attempt1 = stage(1, 2)
    session.flush(force=True)
    assert (key1, ResultMode.one_d(), nexus_target) in (
        accounting.snapshot().pending_durable
    )
    assert session.sink._transaction.snapshot().durable_floor is None
    assert ledger.snapshot().durable == frozenset((
        (0, ResultMode.one_d(), nexus_target),
    ))

    session.abort()
    assert target.read_bytes() == epoch_a
    snap = accounting.snapshot()
    assert snap.durable == frozenset((
        (key0, ResultMode.one_d(), nexus_target),
    ))
    assert (key1, ResultMode.one_d(), nexus_target) not in snap.durable_attempts
    assert snap.attempt_states[attempt1].value == "completed"
    assert attempt1 in snap.aborted_epoch_attempts
    assert 1 in ledger.snapshot().accepted
    assert 1 in ledger.snapshot().completed
    assert (1, ResultMode.one_d()) in ledger.snapshot().written
    assert ledger.snapshot().durable == frozenset((
        (0, ResultMode.one_d(), nexus_target),
    ))


def test_dynamic_abort_after_committed_epoch_with_no_new_work_is_terminal(tmp_path):
    """An already-accounted committed prefix is sufficient authority for a
    no-new-work abort; no second epoch seal exists or is needed."""
    from xdart.modules.reduction import open_live_scan_nexus_session
    from xrd_tools.session import (
        DynamicAccountingLimits,
        DynamicFrameIdentity,
        DynamicRunAccounting,
    )

    target = tmp_path / "dynamic-abort-after-epoch.nexus"
    mode, nexus_target = ResultMode.one_d(), f"nexus:{target}"
    ledger = StageLedger(
        required_modes=(mode,), targets_by_mode={mode: (nexus_target,)},
    )
    accounting = DynamicRunAccounting(
        ledger,
        run_generation=1,
        limits=DynamicAccountingLimits(1, 2, 1),
    )
    key = DynamicFrameIdentity("growing-master.h5", 0)
    accounting.discover(key, group="detector", ordinal=0, output_label=0)
    attempt = accounting.begin_attempt(key, source_revision=1)
    accounting.record_accepted(attempt)
    accounting.record_completed(attempt, produced=(mode,))
    accounting.record_written(attempt, modes=(mode,))

    intent = _intent(
        tmp_path, extent=1, labels=(0,), generation=0,
        modes=("1d:default",),
    )
    session = open_live_scan_nexus_session(
        _live_scan(target, intent, (0,)),
        replace=True,
        accounting=accounting.writer_boundary,
    )
    session.flush(force=True)
    session.commit_epoch()
    committed = target.read_bytes()

    session.abort()
    session.abort()  # exact terminal replay is idempotent
    assert session._closed
    assert target.read_bytes() == committed
    snapshot = accounting.snapshot()
    assert snapshot.state.value == "aborted"
    assert snapshot.durable_attempts[(key, mode, nexus_target)] is attempt
    assert snapshot.pending_durable == frozenset()
    assert session not in accounting.owner_census()
    assert session.sink._transaction_owners is None
    _assert_lease_available(target)


def test_dynamic_stop_without_new_rows_discards_epoch_and_preserves_prior(tmp_path):
    from xdart.modules.reduction import open_live_scan_nexus_session
    from xrd_tools.session import (
        DynamicAccountingLimits,
        DynamicFrameIdentity,
        DynamicRunAccounting,
    )

    target = tmp_path / "dynamic-stop-empty.nexus"
    mode, target_name = ResultMode.one_d(), f"nexus:{target}"
    ledger = StageLedger(
        required_modes=(mode,), targets_by_mode={mode: (target_name,)},
    )
    accounting = DynamicRunAccounting(
        ledger, run_generation=1,
        limits=DynamicAccountingLimits(2, 3, 2),
    )
    key0 = DynamicFrameIdentity("growing", 0)
    accounting.discover(key0, group="g", ordinal=0, output_label=0)
    attempt0 = accounting.begin_attempt(key0, source_revision=1)
    accounting.record_accepted(attempt0)
    accounting.record_completed(attempt0, produced=(mode,))
    accounting.record_written(attempt0, modes=(mode,))
    first = _intent(
        tmp_path, extent=1, labels=(0,), generation=0,
        modes=("1d:default",),
    )
    live = _live_scan(target, first, (0,))
    session = open_live_scan_nexus_session(
        live, replace=True, accounting=accounting.writer_boundary,
    )
    session.flush(force=True)
    session.commit_epoch()
    epoch_a = target.read_bytes()
    session.extend(_intent(
        tmp_path, extent=2, labels=(0, 1), generation=1,
        modes=("1d:default",),
    ))
    accounting.stop()
    session.finish(finalize=False)
    assert target.read_bytes() == epoch_a
    snapshot = accounting.snapshot()
    assert snapshot.state.value == "stopped"
    assert snapshot.durable == frozenset(((key0, mode, target_name),))
    assert snapshot.in_flight == snapshot.retry_owned == frozenset()


def test_dynamic_stop_commits_only_the_exact_ready_prefix(tmp_path):
    from xdart.modules.reduction import open_live_scan_nexus_session
    from xrd_tools.session import (
        DynamicAccountingLimits,
        DynamicFrameIdentity,
        DynamicRunAccounting,
    )

    target = tmp_path / "dynamic-stop-prefix.nexus"
    mode, target_name = ResultMode.one_d(), f"nexus:{target}"
    ledger = StageLedger(
        required_modes=(mode,), targets_by_mode={mode: (target_name,)},
    )
    accounting = DynamicRunAccounting(
        ledger, run_generation=1,
        limits=DynamicAccountingLimits(2, 3, 2),
    )

    def stage(label, revision):
        key = DynamicFrameIdentity("growing", label)
        accounting.discover(key, group="g", ordinal=label, output_label=label)
        attempt = accounting.begin_attempt(key, source_revision=revision)
        accounting.record_accepted(attempt)
        accounting.record_completed(attempt, produced=(mode,))
        accounting.record_written(attempt, modes=(mode,))
        return key

    key0 = stage(0, 1)
    first = _intent(
        tmp_path, extent=1, labels=(0,), generation=0,
        modes=("1d:default",),
    )
    live = _live_scan(target, first, (0,))
    session = open_live_scan_nexus_session(
        live, replace=True, accounting=accounting.writer_boundary,
    )
    session.flush(force=True)
    session.commit_epoch()
    session.extend(_intent(
        tmp_path, extent=2, labels=(0, 1), generation=1,
        modes=("1d:default",),
    ))
    live.frames.add(1)
    key1 = stage(1, 2)
    session.flush(force=True)
    accounting.stop()
    session.finish(finalize=False)
    with h5py.File(target, "r") as handle:
        assert tuple(handle["entry/integrated_1d/frame_index"][()]) == (0, 1)
    snapshot = accounting.snapshot()
    assert snapshot.state.value == "stopped"
    assert snapshot.durable == frozenset((
        (key0, mode, target_name), (key1, mode, target_name),
    ))
    assert snapshot.in_flight == snapshot.retry_owned == frozenset()


def test_dynamic_all_nan_writer_drop_waits_for_exact_h23_commit(tmp_path):
    from xdart.modules.reduction import open_live_scan_nexus_session
    from xrd_tools.session import (
        DynamicAccountingLimits,
        DynamicFrameIdentity,
        DynamicRunAccounting,
    )

    target = tmp_path / "dynamic-all-nan.nexus"
    mode, target_name = ResultMode.one_d(), f"nexus:{target}"
    ledger = StageLedger(
        required_modes=(mode,), targets_by_mode={mode: (target_name,)},
    )
    accounting = DynamicRunAccounting(
        ledger, run_generation=1,
        limits=DynamicAccountingLimits(1, 2, 1),
    )
    key = DynamicFrameIdentity("nan-source", 0)
    accounting.discover(key, group="g", ordinal=0, output_label=0)
    attempt = accounting.begin_attempt(key, source_revision=1)
    accounting.record_accepted(attempt)
    accounting.record_completed(attempt, produced=(mode,))
    accounting.record_written(attempt, modes=(mode,))
    intent = _intent(
        tmp_path, extent=1, labels=(0,), generation=0,
        modes=("1d:default",),
    )
    live = _live_scan(target, intent, (0,))
    live.frames[0].int_1d = IntegrationResult1D(
        radial=np.linspace(0.1, 1.0, 8),
        intensity=np.full(8, np.nan),
        sigma=np.full(8, np.nan),
        unit="q_A^-1",
    )
    session = open_live_scan_nexus_session(
        live, replace=True, accounting=accounting.writer_boundary,
    )
    session.flush(force=True)
    assert accounting.snapshot().pending_publication_dropped == frozenset((
        (key, mode),
    ))
    assert ledger.snapshot().publication_dropped == frozenset()
    session.commit_epoch()
    assert accounting.snapshot().publication_dropped == frozenset(((key, mode),))
    assert ledger.snapshot().publication_dropped == frozenset(((0, mode),))
    assert ledger.snapshot().durable == frozenset()
    with h5py.File(target, "r") as handle:
        assert "integrated_1d" not in handle["entry"]
    session.finish(commit_empty=True)


def _dynamic_revision_session(tmp_path, name):
    from xdart.modules.reduction import open_live_scan_nexus_session
    from xrd_tools.session import (
        DynamicAccountingLimits,
        DynamicFrameIdentity,
        DynamicRunAccounting,
    )

    target = tmp_path / f"{name}.nexus"
    mode, target_name = ResultMode.one_d(), f"nexus:{target}"
    ledger = StageLedger(
        required_modes=(mode,), targets_by_mode={mode: (target_name,)},
    )
    accounting = DynamicRunAccounting(
        ledger, run_generation=1,
        limits=DynamicAccountingLimits(1, 4, 1),
    )
    key = DynamicFrameIdentity("same-path-growing-source", 0)
    accounting.discover(key, group="g", ordinal=0, output_label=0)
    intent = _intent(
        tmp_path, extent=1, labels=(0,), generation=0,
        modes=("1d:default",),
    )
    live = _live_scan(target, intent, (0,))
    session = open_live_scan_nexus_session(
        live, replace=True, accounting=accounting.writer_boundary,
    )

    def stage(revision, result):
        token = accounting.begin_attempt(key, source_revision=revision)
        accounting.record_accepted(token)
        accounting.record_completed(token, produced=(mode,))
        accounting.record_written(token, modes=(mode,))
        live.frames[0].int_1d = result
        return token

    return target, mode, target_name, ledger, accounting, key, live, session, stage


def test_dynamic_current_drop_removes_older_row_before_same_epoch_commit(tmp_path):
    (target, mode, target_name, ledger, accounting, key, _live, session,
     stage) = _dynamic_revision_session(tmp_path, "valid-then-drop")
    first = stage(1, _r1(11))
    session.flush(force=True)
    assert accounting.snapshot().pending_durable == frozenset((
        (key, mode, target_name),
    ))

    dropped = IntegrationResult1D(
        radial=np.linspace(0.1, 1.0, 8),
        intensity=np.full(8, np.nan),
        sigma=np.full(8, np.nan),
        unit="q_A^-1",
    )
    second = stage(2, dropped)
    session.flush(replace_frame_indices=(0,), force=True)
    pending = accounting.snapshot()
    assert pending.pending_durable == frozenset()
    assert pending.pending_publication_dropped == frozenset(((key, mode),))
    session.commit_epoch()

    with h5py.File(target, "r") as handle:
        group = handle["entry"].get("integrated_1d")
        assert group is None or 0 not in tuple(group["frame_index"][()])
    final = accounting.snapshot()
    assert final.publication_dropped_attempts[(key, mode)] is second
    assert (key, mode, target_name) not in final.durable_attempts
    assert ledger.snapshot().publication_dropped == frozenset(((0, mode),))
    assert first is not second
    session.finish(commit_empty=True)


def test_dynamic_newer_valid_row_retires_same_epoch_pending_drop(tmp_path):
    (target, mode, target_name, ledger, accounting, key, _live, session,
     stage) = _dynamic_revision_session(tmp_path, "drop-then-valid")
    dropped = IntegrationResult1D(
        radial=np.linspace(0.1, 1.0, 8),
        intensity=np.full(8, np.nan),
        sigma=np.full(8, np.nan),
        unit="q_A^-1",
    )
    first = stage(1, dropped)
    session.flush(force=True)
    assert accounting.snapshot().pending_publication_dropped == frozenset((
        (key, mode),
    ))

    second = stage(2, _r1(22))
    session.flush(replace_frame_indices=(0,), force=True)
    pending = accounting.snapshot()
    assert pending.pending_publication_dropped == frozenset()
    assert pending.pending_durable == frozenset(((key, mode, target_name),))
    session.commit_epoch()

    with h5py.File(target, "r") as handle:
        group = handle["entry/integrated_1d"]
        assert tuple(group["frame_index"][()]) == (0,)
        np.testing.assert_allclose(group["intensity"][0], 22)
    final = accounting.snapshot()
    assert final.durable_attempts[(key, mode, target_name)] is second
    assert final.publication_dropped == frozenset()
    assert ledger.snapshot().durable == frozenset(((0, mode, target_name),))
    assert first is not second
    session.finish(commit_empty=True)


def test_dynamic_finish_refuses_unknown_h23_phase_without_publishing(tmp_path):
    (_target, mode, target_name, ledger, accounting, key, _live, session,
     stage) = _dynamic_revision_session(tmp_path, "unknown-finish-phase")
    stage(1, _r1(1))
    session.flush(force=True)
    real_phase = session._transaction_phase
    session._transaction_phase = lambda: "publishing"
    with pytest.raises(RuntimeError, match="terminal transaction phase"):
        session.finish()
    assert ledger.snapshot().durable == frozenset()
    assert accounting.snapshot().pending_durable == frozenset((
        (key, mode, target_name),
    ))
    assert session._accounting_finish_seal is not None
    assert not session._closed

    session._transaction_phase = real_phase
    session.finish()
    assert ledger.snapshot().durable == frozenset(((0, mode, target_name),))


def test_dynamic_abort_refuses_unknown_h23_phase_without_settlement(tmp_path):
    (_target, mode, target_name, ledger, accounting, key, _live, session,
     stage) = _dynamic_revision_session(tmp_path, "unknown-abort-phase")
    stage(1, _r1(1))
    session.flush(force=True)
    real_phase = session._transaction_phase
    session._transaction_phase = lambda: "rolling-back"
    with pytest.raises(RuntimeError, match="terminal transaction phase"):
        session.abort()
    assert ledger.snapshot().durable == frozenset()
    assert accounting.snapshot().pending_durable == frozenset((
        (key, mode, target_name),
    ))
    assert not session._closed

    session._transaction_phase = real_phase
    session.abort()
    assert accounting.snapshot().state.value == "aborted"
    assert accounting.snapshot().pending_durable == frozenset()


def test_live_session_finish_releases_bound_light_1d_authority(tmp_path):
    from xdart.modules.reduction import open_live_scan_nexus_session
    from xrd_tools.session import (
        DynamicAccountingLimits,
        DynamicRunAccounting,
        Light1DBufferLayout,
        Light1DCleanupHooks,
        Light1DLayout,
        Light1DModeLayout,
        SessionResourceAuthority,
        acquire_light_1d_retention,
    )

    target = tmp_path / "dynamic-light-release.nexus"
    nexus_target = f"nexus:{target}"
    ledger = StageLedger(
        required_modes=(ResultMode.one_d(),),
        targets_by_mode={ResultMode.one_d(): (nexus_target,)},
    )
    accounting = DynamicRunAccounting(
        ledger, run_generation=8,
        limits=DynamicAccountingLimits(
            max_groups=2, max_attempts_per_frame=2, max_outstanding=2,
        ),
    )
    authority = SessionResourceAuthority(capacity_bytes=4096)
    layout = Light1DLayout(
        modes=(Light1DModeLayout(
            "raw",
            Light1DBufferLayout(8, 8, "axis", "<f8", shared=True),
            Light1DBufferLayout(8, 8, "intensity", "<f8"),
        ),),
        active_mode="raw",
    )
    lease = acquire_light_1d_retention(
        authority, owner="dynamic-run", generation=8, layout=layout,
        requested_rows=8, compatibility_byte_ceiling=4096,
        gui_thread_id=threading.get_ident(),
    )
    accounting.bind_light_1d(
        lease, cleanup_hooks=Light1DCleanupHooks(),
    )
    session = open_live_scan_nexus_session(
        _live_scan(target, None, ()), replace=True,
        accounting=accounting.writer_boundary,
    )
    session.flush(force=True)
    session.finish(commit_empty=True)
    assert lease.state.value == "released"
    assert authority.snapshot().reserved_bytes == 0
    assert accounting.snapshot().state.value == "finished"
    terminal_census = accounting.owner_census()
    assert session not in terminal_census
    assert lease not in terminal_census


def test_live_session_light_release_fault_is_cleanup_pending_then_retryable(tmp_path):
    from xdart.modules.reduction import open_live_scan_nexus_session
    from xrd_tools.session import (
        DynamicAccountingLimits,
        DynamicRunAccounting,
        Light1DBufferLayout,
        Light1DCleanupHooks,
        Light1DCleanupPending,
        Light1DLayout,
        Light1DModeLayout,
        SessionResourceAuthority,
        acquire_light_1d_retention,
    )

    target = tmp_path / "dynamic-release-retry.nexus"
    mode, nexus_target = ResultMode.one_d(), f"nexus:{target}"
    accounting = DynamicRunAccounting(
        StageLedger(required_modes=(mode,), targets_by_mode={mode: (nexus_target,)}),
        run_generation=9,
        limits=DynamicAccountingLimits(2, 2, 2),
    )
    authority = SessionResourceAuthority(capacity_bytes=1024)
    layout = Light1DLayout(
        (Light1DModeLayout(
            "raw", Light1DBufferLayout(4, 8, "axis", "<f8", shared=True),
            Light1DBufferLayout(4, 8, "intensity", "<f8"),
        ),),
        "raw",
    )
    lease = acquire_light_1d_retention(
        authority, owner="retry", generation=9, layout=layout,
        requested_rows=2, compatibility_byte_ceiling=1024,
        gui_thread_id=threading.get_ident(),
    )
    failures = [OSError("release fault")]

    def release_hook():
        if failures:
            raise failures.pop()

    hooks = Light1DCleanupHooks(release=release_hook)
    accounting.bind_light_1d(lease, cleanup_hooks=hooks)
    session = open_live_scan_nexus_session(
        _live_scan(target, None, ()), replace=True,
        accounting=accounting.writer_boundary,
    )
    session.flush(force=True)
    with pytest.raises(Light1DCleanupPending) as caught:
        session.finish(commit_empty=True)
    assert lease.state.value == "cleanup-pending"
    assert accounting.snapshot().state.value == "cleanup-pending"
    assert not session._closed
    assert authority.snapshot().reserved_bytes == lease.reserved_ndarray_bytes
    pending = accounting.snapshot().cleanup_receipt
    assert pending is not None
    assert pending.light_cleanup_receipt is caught.value.receipt
    assert pending.retry_token is caught.value.token

    session.finish(commit_empty=True)
    assert session._closed
    assert lease.state.value == "released"
    assert authority.snapshot().reserved_bytes == 0
    assert accounting.snapshot().cleanup_receipt is not None
    assert caught.value.token == accounting.snapshot().cleanup_receipt.retry_token
    assert session not in accounting.owner_census()
    assert lease not in accounting.owner_census()


def test_dynamic_finish_seal_survives_canonical_cleanup_fault_and_exact_retry(
    tmp_path, monkeypatch,
):
    from xdart.modules.reduction import open_live_scan_nexus_session
    from xrd_tools.session import (
        DynamicAccountingLimits,
        DynamicFrameIdentity,
        DynamicRunAccounting,
    )

    target = tmp_path / "dynamic-finish-seal-retry.nexus"
    mode, target_name = ResultMode.one_d(), f"nexus:{target}"
    ledger = StageLedger(
        required_modes=(mode,), targets_by_mode={mode: (target_name,)},
    )
    accounting = DynamicRunAccounting(
        ledger, run_generation=1,
        limits=DynamicAccountingLimits(2, 2, 2),
    )
    key = DynamicFrameIdentity("seal-source", 0)
    accounting.discover(key, group="g", ordinal=0, output_label=0)
    attempt = accounting.begin_attempt(key, source_revision=1)
    accounting.record_accepted(attempt)
    accounting.record_completed(attempt, produced=(mode,))
    accounting.record_written(attempt, modes=(mode,))
    intent = _intent(
        tmp_path, extent=1, labels=(0,), generation=0,
        modes=("1d:default",),
    )
    session = open_live_scan_nexus_session(
        _live_scan(target, intent, (0,)),
        replace=True,
        accounting=accounting.writer_boundary,
    )
    session.flush(force=True)
    assert accounting.snapshot().pending_durable == frozenset((
        (key, mode, target_name),
    ))
    transaction_module = importlib.import_module("xrd_tools.io.output_transaction")
    original = transaction_module.OutputTransaction.release_lease_owner
    failed = False

    def fail_once(owner, *args, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("terminal lease cleanup fault")
        return original(owner, *args, **kwargs)

    monkeypatch.setattr(
        transaction_module.OutputTransaction, "release_lease_owner", fail_once,
    )
    with pytest.raises(OSError, match="terminal lease cleanup fault"):
        session.finish()
    seal = session._accounting_finish_seal
    assert seal is not None
    assert session.sink._transaction.snapshot().phase is TransactionPhase.COMMITTED
    assert ledger.snapshot().durable == frozenset()
    with pytest.raises(RuntimeError, match="sealed"):
        accounting.discover(
            DynamicFrameIdentity("seal-source", 1),
            group="g", ordinal=1, output_label=1,
        )
    with pytest.raises(RuntimeError, match="seal"):
        session.flush(force=True)

    session.finish()
    assert session._closed
    assert session._accounting_finish_seal is None
    assert ledger.snapshot().durable == frozenset(((0, mode, target_name),))
    assert accounting.snapshot().state.value == "finished"
    assert session not in accounting.owner_census()


def test_dynamic_light_drain_callback_holds_neither_owner_lock(tmp_path):
    from xdart.modules.reduction import open_live_scan_nexus_session
    from xrd_tools.session import (
        DynamicAccountingLimits,
        DynamicRunAccounting,
        Light1DBufferLayout,
        Light1DCleanupHooks,
        Light1DLayout,
        Light1DModeData,
        Light1DModeLayout,
        Light1DRecord,
        Light1DStaleGeneration,
        SessionResourceAuthority,
        acquire_light_1d_retention,
    )

    target = tmp_path / "dynamic-drain-locks.nexus"
    mode, target_name = ResultMode.one_d(), f"nexus:{target}"
    accounting = DynamicRunAccounting(
        StageLedger(
            required_modes=(mode,), targets_by_mode={mode: (target_name,)},
        ),
        run_generation=11,
        limits=DynamicAccountingLimits(1, 1, 1),
    )
    authority = SessionResourceAuthority(capacity_bytes=1024)
    layout = Light1DLayout((Light1DModeLayout(
        "raw",
        Light1DBufferLayout(4, 8, "axis", "<f8", shared=True),
        Light1DBufferLayout(4, 8, "intensity", "<f8"),
    ),), "raw")
    lease = acquire_light_1d_retention(
        authority,
        owner="drain",
        generation=11,
        layout=layout,
        requested_rows=1,
        compatibility_byte_ceiling=1024,
        gui_thread_id=threading.get_ident(),
    )
    token_box = []
    issuer = threading.Thread(
        target=lambda: token_box.append(lease.issue_hydration_token(0)),
    )
    issuer.start()
    issuer.join()
    start = threading.Event()
    done = threading.Event()
    axis = np.arange(4, dtype=np.float64)
    record = Light1DRecord(
        0, 11, "raw",
        {"raw": Light1DModeData(axis, np.arange(4, dtype=np.float64))},
    )

    def worker():
        start.wait()
        try:
            accounting.snapshot()  # needs DynamicRunAccounting._lock
            lease.complete_hydration(token_box[0], record)  # needs lease._lock
        except Light1DStaleGeneration:
            pass
        finally:
            done.set()

    hydration = threading.Thread(target=worker)
    hydration.start()

    def drain():
        start.set()
        assert done.wait(2), "cleanup callback retained an owner lock"
        hydration.join(timeout=2)

    hooks = Light1DCleanupHooks(drain=drain)
    accounting.bind_light_1d(lease, cleanup_hooks=hooks)
    session = open_live_scan_nexus_session(
        _live_scan(target, None, ()),
        replace=True,
        accounting=accounting.writer_boundary,
    )
    session.flush(force=True)
    session.finish(commit_empty=True)
    assert not hydration.is_alive()
    assert lease.state.value == "released"
    assert authority.snapshot().reserved_bytes == 0


def test_dynamic_live_owner_census_has_one_session_lease_and_active_writer(tmp_path):
    from xdart.modules.reduction import open_live_scan_nexus_session
    from xrd_tools.session import (
        DynamicAccountingLimits,
        DynamicFrameIdentity,
        DynamicRunAccounting,
    )

    target = tmp_path / "dynamic-owner-census.nexus"
    mode, nexus_target = ResultMode.one_d(), f"nexus:{target}"
    ledger = StageLedger(
        required_modes=(mode,), targets_by_mode={mode: (nexus_target,)},
    )
    accounting = DynamicRunAccounting(
        ledger, run_generation=1,
        limits=DynamicAccountingLimits(2, 2, 2),
    )
    intent = _intent(
        tmp_path, extent=1, labels=(0,), generation=0,
        modes=("1d:default",),
    )
    live = _live_scan(target, intent, (0,))
    key = DynamicFrameIdentity("master", 0)
    accounting.discover(key, group="g", ordinal=0, output_label=0)
    token = accounting.begin_attempt(key, source_revision=1)
    accounting.record_accepted(token)
    accounting.record_completed(token, produced=(mode,))
    accounting.record_written(token, modes=(mode,))
    session = open_live_scan_nexus_session(
        live, replace=True, accounting=accounting.writer_boundary,
    )
    session.flush(force=True)
    sink, transaction, lease, writer_a = (
        session.sink, session.sink._transaction, session.sink._lease,
        session.sink._writer,
    )
    session.commit_epoch()
    session.extend(_intent(
        tmp_path, extent=2, labels=(0, 1), generation=1,
        modes=("1d:default",),
    ))
    live.frames.add(1)
    key1 = DynamicFrameIdentity("master", 1)
    accounting.discover(key1, group="g", ordinal=1, output_label=1)
    token1 = accounting.begin_attempt(key1, source_revision=2)
    accounting.record_accepted(token1)
    accounting.record_completed(token1, produced=(mode,))
    accounting.record_written(token1, modes=(mode,))
    session.flush(force=True)
    writer_b = session.sink._writer
    assert session.sink is sink
    assert session.sink._transaction is transaction
    assert session.sink._lease is lease
    assert writer_b is not writer_a and writer_a.phase.value == "finished"
    assert writer_b.phase.value in {"active", "partial"}
    census = accounting.owner_census()
    assert sum(value is ledger for value in census) == 1
    assert sum(value is session for value in census) == 1
    assert sum(value is lease for value in census) == 1
    assert sum(value is writer_b for value in census) == 1
    assert all(value is not writer_a for value in census)
    session.abort()
    terminal_census = accounting.owner_census()
    assert session not in terminal_census
    assert lease not in terminal_census
    assert writer_b not in terminal_census


def test_stop_with_no_rows_in_new_epoch_restores_last_committed_epoch(tmp_path):
    from xdart.modules.reduction import open_live_scan_nexus_session
    target = tmp_path / "empty-stop-epoch.nexus"
    first = _intent(
        tmp_path, extent=1, labels=(0,), generation=0,
        modes=("1d:default",),
    )
    live = _live_scan(target, first, (0,))
    session = open_live_scan_nexus_session(live, replace=True)
    session.flush(force=True)
    session.commit_epoch()
    epoch_a = target.read_bytes()
    session.extend(_intent(
        tmp_path, extent=2, labels=(0, 1), generation=1,
        modes=("1d:default",),
    ))
    session.finish(finalize=False)
    assert target.read_bytes() == epoch_a
    assert session.sink._transaction_owners is None
    _assert_lease_available(target)


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
    target = tmp_path / "preflight.nxs"
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
    preflight = prepare_append_preflight(tmp_path / "generation.nxs", current)
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
    preflight = prepare_append_preflight(tmp_path / "reserved-regression.nxs", current)
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
    target = tmp_path / "qualification-cleanup.nxs"
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

    target = tmp_path / "appeared.nxs"
    preflight = prepare_append_preflight(
        target, _intent(tmp_path, extent=1, labels=(0,), modes=("1d:default",)))
    target.write_bytes(b"foreign")
    with pytest.raises(TargetChanged, match="changed before abandonment"):
        preflight.abort()
    assert preflight.snapshot.state.value == "integrity_hold"
    assert preflight.retry_cleanup().state.value == "integrity_hold"


def test_noop_preflight_rechecks_target_before_reporting_terminal(tmp_path):
    from xrd_tools.io import prepare_append_preflight
    target = tmp_path / "noop-mutation.nxs"
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
            for name in ("integrated_1d", "integrated_2d"):
                del handle[f"entry/{name}/frame_index"]
                handle[f"entry/{name}"].create_dataset(
                    "frame_index", data=np.asarray(intent.labels, dtype=np.int64),
                )
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
    target = tmp_path / "decoder.nxs"
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

    target = tmp_path / "disappeared.nxs"
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

    target = tmp_path / "divergent.nxs"
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
        typed_target = tmp_path / f"typed-{mutation}.nxs"
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
        scalar_target = tmp_path / f"scalar-{mutation}.nxs"
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
        with h5py.File(scalar_target, "r") as handle:
            with pytest.raises(ValueError, match="schema|source base|lineage scalar"):
                decode_committed_append_prefix(handle)
        with pytest.raises(AppendRefused, match="schema|source base|lineage scalar"):
            prepare_append_preflight(
                scalar_target, _image_series_intent(tmp_path, 3, generation=1),
                committed_prefix=scalar_prefix,
            )
        assert scalar_target.read_bytes() == scalar_before
        _assert_lease_available(scalar_target)

    primary_target = tmp_path / "primary-mode-int.nxs"
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
        del entry["reduction/config/append_lineage"]
        entry["reduction/config"].create_dataset(
            "append_lineage", data=primary_prefix.lineage_json,
        )
    primary_before = primary_target.read_bytes()
    with h5py.File(primary_target, "r") as handle:
        with pytest.raises(ValueError, match="primary mode"):
            decode_committed_append_prefix(handle)
    successor = replace(
        _image_series_intent(tmp_path, 3, generation=1), modes=("1d:7",),
    )
    with pytest.raises(AppendRefused, match="primary mode"):
        prepare_append_preflight(
            primary_target, successor, committed_prefix=primary_prefix,
        )
    assert primary_target.read_bytes() == primary_before
    _assert_lease_available(primary_target)


def test_prefix_bound_preflight_accepts_concurrent_exact_successor_as_noop(
    tmp_path,
):
    from xrd_tools.io import AppendDisposition, AppendPreflightState
    from xrd_tools.io import prepare_append_preflight

    target = tmp_path / "concurrent-successor.nxs"
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
    target = tmp_path / "reserved-anchor.nxs"
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
