"""H23-C0/C1 finite oracle for the unmounted output transaction kernel.

The implementation import is deliberately deferred into each test.  On the
exact H23 parent every node therefore records its own contract-absence failure
instead of collapsing collection into one import error.
"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
import hashlib
import importlib
import os
from pathlib import Path
import threading

import pytest


def _api():
    return importlib.import_module("xrd_tools.io.output_transaction")


def _owners(api, prefix: str = "owner"):
    return {
        api.LeaseOwner.RUN: api.OwnerToken(f"{prefix}-run"),
        api.LeaseOwner.SESSION: api.OwnerToken(f"{prefix}-session"),
        api.LeaseOwner.SOURCE: api.OwnerToken(f"{prefix}-source"),
        api.LeaseOwner.CLEANUP: api.OwnerToken(f"{prefix}-cleanup"),
    }


def _prepared(
    tmp_path: Path,
    *,
    prior: bytes | None = b"prior",
    durable_fsync: bool = True,
    fast_regenerable: bool = False,
):
    api = _api()
    target = tmp_path / "result.nexus"
    if prior is not None:
        target.write_bytes(prior)
    coordinator = api.OutputTransactionCoordinator()
    transaction_owner = api.OwnerToken("transaction")
    target_owner = api.OwnerToken("target")
    transaction = coordinator.admit(
        target,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        durable_fsync=durable_fsync,
        fast_regenerable=fast_regenerable,
    )
    owners = _owners(api)
    lease = transaction.acquire_lease(
        admission=transaction.admission,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        owners=owners,
    )
    return (
        api,
        target,
        coordinator,
        transaction,
        transaction_owner,
        target_owner,
        owners,
        lease,
    )


def test_untouched_integrity_hold_can_abandon_only_after_exact_restore(
    tmp_path: Path,
) -> None:
    (
        api,
        target,
        _coordinator,
        transaction,
        _transaction_owner,
        _target_owner,
        owners,
        lease,
    ) = _prepared(tmp_path, prior=None)

    target.write_bytes(b"foreign occupant")
    with pytest.raises(api.TargetChanged, match="changed before abandonment"):
        transaction.abandon(lease)
    assert transaction.snapshot().phase is api.TransactionPhase.INTEGRITY_HOLD
    assert target.read_bytes() == b"foreign occupant"
    with pytest.raises(api.TargetChanged, match="changed before abandonment"):
        transaction.abandon(lease)

    target.unlink()
    retired = transaction.abandon(lease)
    assert retired.phase is api.TransactionPhase.ABORTED
    released = None
    for role in api.LeaseOwner:
        released = transaction.release_lease_owner(lease, role, owners[role])
    assert released is not None
    assert released.remaining_owners == ()


@pytest.mark.parametrize(
    ("durable_fsync", "expected_calls"),
    ((True, 2), (False, 0)),
    ids=("durable", "diagnostic-no-fsync"),
)
def test_descriptor_receipts_can_skip_only_diagnostic_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    durable_fsync: bool,
    expected_calls: int,
) -> None:
    api = _api()
    target = tmp_path / "diagnostic.nexus"
    target.write_bytes(b"semantic-content")
    descriptor = os.open(target, os.O_RDONLY)
    calls: list[int] = []
    monkeypatch.setattr(api.os, "fsync", calls.append)
    try:
        content = api._descriptor_content_receipt(
            descriptor,
            target,
            "diagnostic-content",
            durable_fsync=durable_fsync,
        )
        stream = api._descriptor_stream_stat_receipt(
            descriptor,
            target,
            evidence_digest=content.snapshot.digest,
            evidence_bytes=content.snapshot.size,
            ordinal=1,
            role="diagnostic-stream",
            durable_fsync=durable_fsync,
        )
    finally:
        os.close(descriptor)
    assert len(calls) == expected_calls
    assert content.snapshot.digest == hashlib.sha256(
        b"semantic-content"
    ).hexdigest()
    assert stream.evidence_digest == content.snapshot.digest
    assert stream.evidence_bytes == len(b"semantic-content")

    coordinator = api.OutputTransactionCoordinator()
    transaction = coordinator.admit(
        tmp_path / "owned.nexus",
        transaction_owner=api.OwnerToken("transaction"),
        target_owner=api.OwnerToken("target"),
        durable_fsync=durable_fsync,
    )
    assert transaction._durable_fsync is durable_fsync


class _Pool:
    def __init__(self, *, fail_pause: int = 0, fail_resume: int = 0):
        self.fail_pause = fail_pause
        self.fail_resume = fail_resume
        self.pauses: list[str] = []
        self.resumes: list[str] = []

    def pause(self, path):
        self.pauses.append(str(path))
        if len(self.pauses) <= self.fail_pause:
            raise OSError("pool pause failed")

    def resume(self, path):
        self.resumes.append(str(path))
        if len(self.resumes) <= self.fail_resume:
            raise OSError("pool resume failed")


class _PartialPausePool:
    """Pool double whose failing pause has already acquired exclusion."""

    def __init__(self):
        self.depth = 0
        self.pauses: list[str] = []
        self.resumes: list[str] = []

    def pause(self, path):
        self.pauses.append(str(path))
        self.depth += 1
        if len(self.pauses) == 1:
            raise OSError("pool pause failed after acquiring exclusion")

    def resume(self, path):
        self.resumes.append(str(path))
        if self.depth:
            self.depth -= 1


class _DepthPool:
    """Pool double that exposes exclusion depth at filesystem boundaries."""

    def __init__(self):
        self.depth = 0
        self.pauses: list[str] = []
        self.resumes: list[str] = []

    def pause(self, path):
        self.pauses.append(str(path))
        self.depth += 1

    def resume(self, path):
        self.resumes.append(str(path))
        assert self.depth > 0
        self.depth -= 1


class _TerminalInterruption(BaseException):
    pass


def _executing_stream(
    tmp_path: Path,
    *,
    durable_fsync: bool = True,
    fast_regenerable: bool = False,
    content: bytes = b"AAAA",
):
    prepared = _prepared(
        tmp_path,
        durable_fsync=durable_fsync,
        fast_regenerable=fast_regenerable,
    )
    (
        _api_module,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = prepared
    attempt = transaction.begin_stream(
        admission=transaction.admission,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease,
        pool=_DepthPool(),
        file_lock=threading.RLock(),
    )
    transaction.authorize_stream_mutation(attempt, lease=lease)
    target.write_bytes(content)
    return prepared, attempt


def _execute(
    transaction,
    write,
    *,
    transaction_owner,
    target_owner,
    lease,
    pool=None,
    validate_writer_result=None,
):
    return transaction.execute(
        write,
        admission=transaction.admission,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease,
        pool=pool or _Pool(),
        validate_writer_result=validate_writer_result,
    )


def test_writer_result_validator_observes_one_exact_captured_candidate(
    tmp_path: Path,
) -> None:
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path)
    seen = []

    def validator(path, captured):
        seen.append((Path(path), captured))
        assert Path(path) == transaction._candidate
        assert api._capture_target(str(path)) == captured

    outcome = _execute(
        transaction,
        lambda path: path.write_bytes(b"validated result"),
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease,
        validate_writer_result=validator,
    )
    assert outcome.phase is api.TransactionPhase.COMMITTED
    assert target.read_bytes() == b"validated result"
    assert len(seen) == 1


@pytest.mark.parametrize("prior", (None, b"exact prior"))
def test_writer_result_validator_exception_rolls_back_exact_prior(
    tmp_path: Path,
    prior: bytes | None,
) -> None:
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=prior)
    writes = []

    def refuse(_path, _captured):
        raise ValueError("semantic refusal")

    with pytest.raises(ValueError, match="semantic refusal"):
        _execute(
            transaction,
            lambda path: (writes.append("writer"), path.write_bytes(b"candidate")),
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
            validate_writer_result=refuse,
        )
    assert (target.read_bytes() if target.exists() else None) == prior
    assert transaction.snapshot().phase is api.TransactionPhase.READY_TO_RETRY
    assert writes == ["writer"]


def test_writer_result_validator_mutation_is_caught_before_publication(
    tmp_path: Path,
) -> None:
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=None)

    def mutate(path, _captured):
        path.write_bytes(b"changed after semantic receipt")

    with pytest.raises(api.TargetChanged, match="validated writer result"):
        _execute(
            transaction,
            lambda path: path.write_bytes(b"validated candidate"),
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
            validate_writer_result=mutate,
        )
    assert not target.exists()
    assert transaction.snapshot().phase is api.TransactionPhase.READY_TO_RETRY


def test_invalid_writer_result_validator_consumes_no_state(tmp_path: Path) -> None:
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path)
    writes = []
    with pytest.raises(TypeError, match="validator"):
        _execute(
            transaction,
            lambda path: writes.append(path),
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
            validate_writer_result=object(),
        )
    assert writes == []
    assert target.read_bytes() == b"prior"
    assert transaction.snapshot().phase is api.TransactionPhase.LEASED


def test_validator_primary_survives_retryable_rollback_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=None)
    real_unlink = api._unlink
    failed = []
    writes = []

    def fail_candidate_once(path):
        if Path(path) == transaction._candidate and not failed:
            failed.append("candidate")
            raise OSError("candidate cleanup fault")
        return real_unlink(path)

    monkeypatch.setattr(api, "_unlink", fail_candidate_once)
    with pytest.raises(ValueError, match="semantic refusal") as raised:
        _execute(
            transaction,
            lambda path: (writes.append("writer"), path.write_bytes(b"candidate")),
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
            validate_writer_result=lambda _path, _snapshot: (
                _ for _ in ()
            ).throw(ValueError("semantic refusal")),
        )
    assert isinstance(raised.value.__cause__, OSError)
    pending = transaction.snapshot()
    assert pending.phase is api.TransactionPhase.CLEANUP_PENDING
    assert pending.pending_actions == (api.RetryAction.CANDIDATE_UNLINK,)
    assert pending.cleanup_token is not None
    recovered = transaction.retry_cleanup(pending.cleanup_token)
    assert recovered.phase is api.TransactionPhase.READY_TO_RETRY
    assert not target.exists()
    assert writes == ["writer"]


@pytest.mark.parametrize(
    ("durable_fsync", "expected_fsyncs"),
    ((True, 1), (False, 0)),
    ids=("durable", "diagnostic"),
)
def test_stream_terminal_composes_one_optional_fsync_and_one_exact_stat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    durable_fsync: bool,
    expected_fsyncs: int,
) -> None:
    (prepared, attempt) = _executing_stream(
        tmp_path,
        durable_fsync=durable_fsync,
    )
    api, _target, _coordinator, transaction, *_rest, lease = prepared
    fsyncs: list[int] = []
    helper_calls: list[
        tuple[str, bool, tuple[int, int, int, int, int] | None]
    ] = []
    real_content = api._descriptor_content_receipt
    real_stat = api._descriptor_stream_stat_receipt

    def content_receipt(*args, **kwargs):
        helper_calls.append((
            "content",
            kwargs.get("durable_fsync"),
            kwargs.get("expected_stat"),
        ))
        return real_content(*args, **kwargs)

    def stat_receipt(*args, **kwargs):
        helper_calls.append((
            "stat",
            kwargs.get("durable_fsync"),
            kwargs.get("expected_stat"),
        ))
        return real_stat(*args, **kwargs)

    monkeypatch.setattr(api.os, "fsync", fsyncs.append)
    monkeypatch.setattr(api, "_descriptor_content_receipt", content_receipt)
    monkeypatch.setattr(api, "_descriptor_stream_stat_receipt", stat_receipt)
    terminal = transaction.seal_stream_terminal(attempt, lease=lease)

    assert len(fsyncs) == expected_fsyncs
    assert [call[:2] for call in helper_calls] == [
        ("content", False),
        ("stat", False),
    ]
    assert helper_calls[0][2] is not None
    assert helper_calls[0][2] == helper_calls[1][2]
    assert terminal.size == 4


def test_stream_terminal_refuses_same_length_mutation_during_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (prepared, attempt) = _executing_stream(tmp_path)
    api, target, _coordinator, transaction, *_rest, lease = prepared
    transaction.seal_stream_terminal(attempt, lease=lease)
    prior_receipt = transaction._stream_terminal_receipt
    prior_stat = transaction._stream_terminal_stat
    accepted = target.stat()
    real_fsync = api.os.fsync
    real_close = api.os.close

    def fsync_then_mutate(descriptor: int) -> None:
        real_fsync(descriptor)
        writer = os.open(target, os.O_WRONLY)
        try:
            os.write(writer, b"BBBB")
        finally:
            real_close(writer)
        os.utime(
            target,
            ns=(accepted.st_atime_ns, accepted.st_mtime_ns),
        )

    monkeypatch.setattr(api.os, "fsync", fsync_then_mutate)
    with pytest.raises(api.TargetChanged, match="changed during fsync"):
        transaction.seal_stream_terminal(attempt, lease=lease)

    assert transaction.snapshot().phase is api.TransactionPhase.INTEGRITY_HOLD
    assert transaction._stream_terminal_receipt is prior_receipt
    assert transaction._stream_terminal_stat is prior_stat
    assert target.read_bytes() == b"BBBB"


def test_stream_terminal_refuses_same_length_mutation_between_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (prepared, attempt) = _executing_stream(tmp_path)
    api, target, _coordinator, transaction, *_rest, lease = prepared
    transaction.seal_stream_terminal(attempt, lease=lease)
    prior_receipt = transaction._stream_terminal_receipt
    prior_stat = transaction._stream_terminal_stat
    accepted = target.stat()
    real_stat_receipt = api._descriptor_stream_stat_receipt

    def mutate_between_receipts(*args, **kwargs):
        target.write_bytes(b"BBBB")
        os.utime(
            target,
            ns=(accepted.st_atime_ns, accepted.st_mtime_ns),
        )
        return real_stat_receipt(*args, **kwargs)

    monkeypatch.setattr(
        api,
        "_descriptor_stream_stat_receipt",
        mutate_between_receipts,
    )
    with pytest.raises(api.TargetChanged, match="descriptor|seal"):
        transaction.seal_stream_terminal(attempt, lease=lease)

    assert transaction.snapshot().phase is api.TransactionPhase.INTEGRITY_HOLD
    assert transaction._stream_terminal_receipt is prior_receipt
    assert transaction._stream_terminal_stat is prior_stat
    with pytest.raises(api.TransactionStateError, match="integrity hold"):
        transaction.commit_stream(attempt, lease=lease)


def test_stream_terminal_constructor_failure_retains_prior_pair_and_holds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (prepared, attempt) = _executing_stream(tmp_path)
    api, _target, _coordinator, transaction, *_rest, lease = prepared
    transaction.seal_stream_terminal(attempt, lease=lease)
    prior_receipt = transaction._stream_terminal_receipt
    prior_stat = transaction._stream_terminal_stat

    def fail_constructor(*_args, **_kwargs):
        raise RuntimeError("terminal constructor fault")

    monkeypatch.setattr(api, "StreamTerminal", fail_constructor)
    with pytest.raises(api.TargetChanged, match="content seal failed") as raised:
        transaction.seal_stream_terminal(attempt, lease=lease)

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert transaction.snapshot().phase is api.TransactionPhase.INTEGRITY_HOLD
    assert transaction._stream_terminal_receipt is prior_receipt
    assert transaction._stream_terminal_stat is prior_stat


@pytest.mark.parametrize(
    "failure_type",
    (KeyboardInterrupt, SystemExit, _TerminalInterruption),
)
def test_stream_terminal_constructor_base_exception_holds_then_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_type: type[BaseException],
) -> None:
    (prepared, attempt) = _executing_stream(tmp_path)
    api, _target, _coordinator, transaction, *_rest, lease = prepared
    transaction.seal_stream_terminal(attempt, lease=lease)
    prior_receipt = transaction._stream_terminal_receipt
    prior_stat = transaction._stream_terminal_stat

    def interrupt_constructor(*_args, **_kwargs):
        raise failure_type("terminal constructor interruption")

    monkeypatch.setattr(api, "StreamTerminal", interrupt_constructor)
    with pytest.raises(failure_type, match="constructor interruption"):
        transaction.seal_stream_terminal(attempt, lease=lease)

    assert transaction.snapshot().phase is api.TransactionPhase.INTEGRITY_HOLD
    assert transaction._stream_terminal_receipt is prior_receipt
    assert transaction._stream_terminal_stat is prior_stat
    with pytest.raises(api.TransactionStateError, match="integrity hold"):
        transaction.commit_stream(attempt, lease=lease)


def test_stream_terminal_close_failure_retains_prior_pair_and_holds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (prepared, attempt) = _executing_stream(tmp_path)
    api, _target, _coordinator, transaction, *_rest, lease = prepared
    transaction.seal_stream_terminal(attempt, lease=lease)
    prior_receipt = transaction._stream_terminal_receipt
    prior_stat = transaction._stream_terminal_stat
    real_close = api.os.close

    def close_then_fail(descriptor: int) -> None:
        real_close(descriptor)
        raise OSError("descriptor close fault")

    monkeypatch.setattr(api.os, "close", close_then_fail)
    with pytest.raises(api.TargetChanged, match="content seal failed") as raised:
        transaction.seal_stream_terminal(attempt, lease=lease)

    assert isinstance(raised.value.__cause__, OSError)
    assert transaction.snapshot().phase is api.TransactionPhase.INTEGRITY_HOLD
    assert transaction._stream_terminal_receipt is prior_receipt
    assert transaction._stream_terminal_stat is prior_stat


@pytest.mark.parametrize(
    "failure_type",
    (KeyboardInterrupt, SystemExit, _TerminalInterruption),
)
def test_stream_terminal_close_base_exception_holds_then_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_type: type[BaseException],
) -> None:
    (prepared, attempt) = _executing_stream(tmp_path)
    api, _target, _coordinator, transaction, *_rest, lease = prepared
    transaction.seal_stream_terminal(attempt, lease=lease)
    prior_receipt = transaction._stream_terminal_receipt
    prior_stat = transaction._stream_terminal_stat
    real_close = api.os.close

    def close_then_interrupt(descriptor: int) -> None:
        real_close(descriptor)
        raise failure_type("terminal close interruption")

    monkeypatch.setattr(api.os, "close", close_then_interrupt)
    with pytest.raises(failure_type, match="close interruption"):
        transaction.seal_stream_terminal(attempt, lease=lease)

    assert transaction.snapshot().phase is api.TransactionPhase.INTEGRITY_HOLD
    assert transaction._stream_terminal_receipt is prior_receipt
    assert transaction._stream_terminal_stat is prior_stat
    with pytest.raises(api.TransactionStateError, match="integrity hold"):
        transaction.commit_stream(attempt, lease=lease)


def test_stream_terminal_token_failure_retains_prior_pair_and_holds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (prepared, attempt) = _executing_stream(tmp_path)
    api, _target, coordinator, transaction, *_rest, lease = prepared
    transaction.seal_stream_terminal(attempt, lease=lease)
    prior_receipt = transaction._stream_terminal_receipt
    prior_stat = transaction._stream_terminal_stat

    def fail_token() -> int:
        raise RuntimeError("terminal token fault")

    monkeypatch.setattr(coordinator, "_next_ordinal", fail_token)
    with pytest.raises(api.TargetChanged, match="content seal failed") as raised:
        transaction.seal_stream_terminal(attempt, lease=lease)

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert transaction.snapshot().phase is api.TransactionPhase.INTEGRITY_HOLD
    assert transaction._stream_terminal_receipt is prior_receipt
    assert transaction._stream_terminal_stat is prior_stat


def test_public_tokens_snapshots_and_outcomes_are_immutable_values(
    tmp_path: Path,
) -> None:
    (
        api,
        _target,
        _coordinator,
        transaction,
        _transaction_owner,
        _target_owner,
        _owners_by_role,
        _lease,
    ) = _prepared(tmp_path)

    token = api.OwnerToken("equal-value")
    equal_token = api.OwnerToken("equal-value")
    assert token == equal_token
    assert token is not equal_token
    assert transaction.admission.snapshot.exists
    assert transaction.admission.snapshot.digest

    public_io = importlib.import_module("xrd_tools.io")
    assert public_io.OutputTransactionCoordinator is api.OutputTransactionCoordinator
    assert (
        public_io.get_output_transaction_coordinator()
        is public_io.get_output_transaction_coordinator()
    )

    for value, field, replacement in (
        (token, "label", "changed"),
        (transaction.admission, "target", "changed"),
        (transaction.snapshot(), "phase", api.TransactionPhase.ABORTED),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(value, field, replacement)


def test_equal_valued_foreign_transaction_admission_target_and_lease_refuse(
    tmp_path: Path,
) -> None:
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path)
    writes: list[str] = []

    candidates = (
        {"transaction_owner": api.OwnerToken(transaction_owner.label)},
        {"target_owner": api.OwnerToken(target_owner.label)},
        {"admission": replace(transaction.admission)},
        {"lease": replace(lease)},
    )
    for replacement in candidates:
        kwargs = {
            "admission": transaction.admission,
            "transaction_owner": transaction_owner,
            "target_owner": target_owner,
            "lease": lease,
            "pool": _Pool(),
        }
        kwargs.update(replacement)
        with pytest.raises(api.OwnershipRefused):
            transaction.execute(
                lambda path: writes.append(str(path)),
                **kwargs,
            )

    assert writes == []
    assert target.read_bytes() == b"prior"


def test_target_appearance_after_absent_admission_refuses_before_write(
    tmp_path: Path,
) -> None:
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=None)
    target.write_bytes(b"foreign appearance")
    writes: list[str] = []

    with pytest.raises(api.TargetChanged):
        _execute(
            transaction,
            lambda path: writes.append(str(path)),
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
        )

    assert writes == []
    assert target.read_bytes() == b"foreign appearance"


def test_same_stat_content_mutation_after_admission_refuses(
    tmp_path: Path,
) -> None:
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=b"original")
    admitted = target.stat()
    target.write_bytes(b"mutated!")
    os.utime(target, ns=(admitted.st_atime_ns, admitted.st_mtime_ns))
    changed = target.stat()
    assert changed.st_size == admitted.st_size
    assert changed.st_mtime_ns == admitted.st_mtime_ns

    with pytest.raises(api.TargetChanged):
        _execute(
            transaction,
            lambda path: path.write_bytes(b"wrong"),
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
        )

    assert target.read_bytes() == b"mutated!"


def test_competing_normalized_target_leases_have_exactly_one_winner(
    tmp_path: Path,
) -> None:
    api = _api()
    target = tmp_path / "result.nexus"
    alias_dir = tmp_path / "alias"
    alias_dir.mkdir()
    normalized_alias = alias_dir / ".." / target.name
    coordinator = api.OutputTransactionCoordinator()
    transactions = []
    for index in range(2):
        transaction_owner = api.OwnerToken(f"transaction-{index}")
        target_owner = api.OwnerToken(f"target-{index}")
        transaction = coordinator.admit(
            target if index == 0 else normalized_alias,
            transaction_owner=transaction_owner,
            target_owner=target_owner,
        )
        transactions.append(
            (transaction, transaction_owner, target_owner, _owners(api, str(index)))
        )

    barrier = threading.Barrier(2)
    outcomes: list[tuple[str, int]] = []
    guard = threading.Lock()

    def contend(index: int) -> None:
        transaction, transaction_owner, target_owner, owners = transactions[index]
        barrier.wait()
        try:
            transaction.acquire_lease(
                admission=transaction.admission,
                transaction_owner=transaction_owner,
                target_owner=target_owner,
                owners=owners,
            )
        except api.LeaseUnavailable:
            result = ("refused", index)
        else:
            result = ("winner", index)
        with guard:
            outcomes.append(result)

    threads = [threading.Thread(target=contend, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(2)

    assert sorted(kind for kind, _index in outcomes) == ["refused", "winner"]
    assert all(not thread.is_alive() for thread in threads)


def test_lease_retirement_waits_for_every_owner_and_cleanup_qualification(
    tmp_path: Path,
) -> None:
    (
        api,
        target,
        coordinator,
        transaction,
        transaction_owner,
        target_owner,
        owners,
        lease,
    ) = _prepared(tmp_path, prior=None)

    with pytest.raises(api.TransactionStateError):
        transaction.release_lease_owner(
            lease,
            api.LeaseOwner.CLEANUP,
            owners[api.LeaseOwner.CLEANUP],
        )

    _execute(
        transaction,
        lambda path: path.write_bytes(b"complete"),
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease,
    )
    for role in (api.LeaseOwner.RUN, api.LeaseOwner.SESSION, api.LeaseOwner.SOURCE):
        snapshot = transaction.release_lease_owner(lease, role, owners[role])
        assert snapshot.active
        assert api.LeaseOwner.CLEANUP in snapshot.remaining_owners

    competitor_owner = api.OwnerToken("competitor-transaction")
    competitor_target = api.OwnerToken("competitor-target")
    competitor = coordinator.admit(
        target,
        transaction_owner=competitor_owner,
        target_owner=competitor_target,
    )
    with pytest.raises(api.LeaseUnavailable):
        competitor.acquire_lease(
            admission=competitor.admission,
            transaction_owner=competitor_owner,
            target_owner=competitor_target,
            owners=_owners(api, "competitor"),
        )

    equal_cleanup_owner = api.OwnerToken(owners[api.LeaseOwner.CLEANUP].label)
    with pytest.raises(api.OwnershipRefused):
        transaction.release_lease_owner(
            lease,
            api.LeaseOwner.CLEANUP,
            equal_cleanup_owner,
        )
    retired = transaction.release_lease_owner(
        lease,
        api.LeaseOwner.CLEANUP,
        owners[api.LeaseOwner.CLEANUP],
    )
    assert not retired.active
    assert retired.remaining_owners == ()
    winner = competitor.acquire_lease(
        admission=competitor.admission,
        transaction_owner=competitor_owner,
        target_owner=competitor_target,
        owners=_owners(api, "competitor"),
    )
    assert winner.target == transaction.admission.target


def test_pool_pause_failure_before_staging_preserves_prior_and_is_retryable(
    tmp_path: Path,
) -> None:
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path)
    pool = _Pool(fail_pause=1)
    writes: list[str] = []

    with pytest.raises(OSError, match="pool pause failed"):
        _execute(
            transaction,
            lambda path: writes.append(str(path)),
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
            pool=pool,
        )

    snapshot = transaction.snapshot()
    assert snapshot.phase is api.TransactionPhase.CLEANUP_PENDING
    assert snapshot.pending_actions == (api.RetryAction.POOL_RESUME,)
    assert snapshot.cleanup_token is not None
    assert target.read_bytes() == b"prior"
    assert writes == []

    recovered = transaction.retry_cleanup(snapshot.cleanup_token)
    assert recovered.phase is api.TransactionPhase.READY_TO_RETRY
    assert recovered.pending_actions == ()

    _execute(
        transaction,
        lambda path: (writes.append("write"), path.write_bytes(b"new")),
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease,
        pool=pool,
    )
    assert target.read_bytes() == b"new"
    assert writes == ["write"]


def test_canonical_stage_retries_transient_windows_sharing_violation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=b"prior")
    real_replace = api._replace
    calls: list[tuple[Path, Path]] = []
    writes: list[Path] = []

    def transient_replace(source, destination):
        calls.append((Path(source), Path(destination)))
        if len(calls) <= 2:
            raise PermissionError(13, "Access is denied", str(destination), 5)
        return real_replace(source, destination)

    monkeypatch.setattr(api, "_REPLACE_RETRY_DELAY_S", 0.0, raising=False)
    monkeypatch.setattr(api, "_replace", transient_replace)
    result = _execute(
        transaction,
        lambda path: (writes.append(path), path.write_bytes(b"new")),
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease,
    )

    assert len(calls) == 3
    assert len(writes) == 1
    assert target.read_bytes() == b"new"
    assert result.phase is api.TransactionPhase.COMMITTED


def test_canonical_stream_epoch_extension_retries_under_one_exclusion_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=b"prior")
    real_replace = api._replace
    pool = _DepthPool()
    calls: list[tuple[Path, Path]] = []

    attempt = transaction.begin_stream(
        admission=transaction.admission,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease,
        pool=pool,
        file_lock=threading.RLock(),
    )
    transaction.authorize_stream_mutation(attempt, lease=lease)
    target.write_bytes(b"epoch-a")
    transaction.seal_stream_terminal(attempt, lease=lease)
    committed = transaction.commit_stream_epoch(attempt, lease=lease)
    assert committed.phase is api.TransactionPhase.EPOCH_COMMITTED
    assert target.read_bytes() == b"epoch-a"
    assert pool.depth == 1

    def transient_replace(source, destination):
        assert pool.depth == 1
        calls.append((Path(source), Path(destination)))
        if len(calls) <= 2:
            raise PermissionError(13, "Access is denied", str(destination), 5)
        return real_replace(source, destination)

    monkeypatch.setattr(api, "_REPLACE_RETRY_DELAY_S", 0.0)
    monkeypatch.setattr(api, "_replace", transient_replace)
    next_attempt = transaction.begin_stream_epoch(
        attempt,
        admission=transaction.admission,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease,
    )

    assert len(calls) == 3
    assert target.read_bytes() == b"epoch-a"
    assert pool.depth == 1

    monkeypatch.setattr(api, "_replace", real_replace)
    result = transaction.abort_stream(next_attempt, lease=lease)
    assert result.phase is api.TransactionPhase.ABORTED
    assert target.read_bytes() == b"epoch-a"
    assert pool.depth == 0


def test_canonical_stage_accepts_exact_move_despite_replace_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=b"prior")
    real_replace = api._replace
    calls = 0
    writes: list[Path] = []

    def moved_then_error(source, destination):
        nonlocal calls
        calls += 1
        real_replace(source, destination)
        raise PermissionError(13, "post-move status unavailable", str(destination), 5)

    monkeypatch.setattr(api, "_REPLACE_RETRY_DELAY_S", 0.0, raising=False)
    monkeypatch.setattr(api, "_replace", moved_then_error)
    result = _execute(
        transaction,
        lambda path: (writes.append(path), path.write_bytes(b"new")),
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease,
    )

    assert calls == 1
    assert len(writes) == 1
    assert target.read_bytes() == b"new"
    assert result.phase is api.TransactionPhase.COMMITTED


def test_canonical_stage_accepts_exact_move_despite_generic_oserror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive receipt resolution outranks the syscall's generic error."""
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=b"prior")
    real_replace = api._replace
    calls = 0
    writes: list[Path] = []

    def moved_then_error(source, destination):
        nonlocal calls
        calls += 1
        real_replace(source, destination)
        raise OSError(5, "post-move status unavailable", str(destination))

    monkeypatch.setattr(api, "_replace", moved_then_error)
    result = _execute(
        transaction,
        lambda path: (writes.append(path), path.write_bytes(b"new")),
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease,
    )

    assert calls == 1
    assert len(writes) == 1
    assert target.read_bytes() == b"new"
    assert result.phase is api.TransactionPhase.COMMITTED


def test_unsealed_mutation_after_durable_floor_holds_and_never_rolls_back(
    tmp_path: Path,
) -> None:
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=b"prior")
    attempt = transaction.begin_stream(
        admission=transaction.admission,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease,
        pool=_DepthPool(),
        file_lock=threading.RLock(),
    )
    transaction.authorize_stream_mutation(attempt, lease=lease)
    target.write_bytes(b"durable")
    descriptor = os.open(target, os.O_RDONLY)
    try:
        checkpoint = transaction.seal_stream_checkpoint(
            attempt,
            lease=lease,
            descriptor=descriptor,
            evidence_digest=hashlib.sha256(b"row").hexdigest(),
            evidence_bytes=3,
        )
    finally:
        os.close(descriptor)
    transaction.promote_stream_checkpoint(
        attempt, checkpoint, lease=lease,
    )

    transaction.authorize_stream_mutation(attempt, lease=lease)
    target.write_bytes(b"unsealed")
    with pytest.raises(api.TransactionStateError, match="irreversible durable floor"):
        transaction.abort_stream(attempt, lease=lease)

    snapshot = transaction.snapshot()
    assert snapshot.phase is api.TransactionPhase.INTEGRITY_HOLD
    assert snapshot.durable_floor is checkpoint
    assert api.RetryAction.ROLLBACK not in snapshot.pending_actions
    assert target.read_bytes() == b"unsealed"
    assert transaction.backup.read_bytes() == b"prior"


def test_unobserved_mutation_after_durable_floor_is_never_blessed(
    tmp_path: Path,
) -> None:
    """Inode continuity alone cannot prove the sealed durable bytes survived."""
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=b"prior")
    attempt = transaction.begin_stream(
        admission=transaction.admission,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease,
        pool=_DepthPool(),
        file_lock=threading.RLock(),
    )
    transaction.authorize_stream_mutation(attempt, lease=lease)
    target.write_bytes(b"durable")
    descriptor = os.open(target, os.O_RDONLY)
    try:
        checkpoint = transaction.seal_stream_checkpoint(
            attempt,
            lease=lease,
            descriptor=descriptor,
            evidence_digest=hashlib.sha256(b"row").hexdigest(),
            evidence_bytes=3,
        )
    finally:
        os.close(descriptor)
    transaction.promote_stream_checkpoint(
        attempt, checkpoint, lease=lease,
    )

    # Model mutation through a path that failed to call the authorization
    # boundary.  The abort resolver must still compare the exact sealed stat
    # facts rather than laundering the current bytes as a terminal receipt.
    target.write_bytes(b"tampered-after-seal")
    with pytest.raises(api.TargetChanged, match="durability floor changed"):
        transaction.abort_stream(attempt, lease=lease)

    snapshot = transaction.snapshot()
    assert snapshot.phase is api.TransactionPhase.INTEGRITY_HOLD
    assert snapshot.durable_floor is checkpoint
    assert target.read_bytes() == b"tampered-after-seal"
    assert transaction.backup.read_bytes() == b"prior"


def test_durable_floor_cleanup_retry_reaches_positive_aborted_terminal(
    tmp_path: Path,
) -> None:
    """Retried cleanup must preserve the verified floor and finish ABORTED."""
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=b"prior")
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
    target.write_bytes(b"durable")
    descriptor = os.open(target, os.O_RDONLY)
    try:
        checkpoint = transaction.seal_stream_checkpoint(
            attempt,
            lease=lease,
            descriptor=descriptor,
            evidence_digest=hashlib.sha256(b"row").hexdigest(),
            evidence_bytes=3,
        )
    finally:
        os.close(descriptor)
    transaction.promote_stream_checkpoint(
        attempt, checkpoint, lease=lease,
    )

    with pytest.raises(api.CleanupIncomplete):
        transaction.abort_stream(attempt, lease=lease)
    pending = transaction.snapshot()
    assert pending.pending_actions == (api.RetryAction.POOL_RESUME,)
    assert pending.cleanup_token is not None
    assert target.read_bytes() == b"durable"

    complete = transaction.retry_cleanup(pending.cleanup_token)
    assert complete.phase is api.TransactionPhase.ABORTED
    assert complete.pending_actions == ()
    assert complete.partial_path == str(target)
    assert complete.durable_floor is checkpoint
    assert pool.resumes == [str(target.resolve()), str(target.resolve())]


def test_canonical_stage_retry_exhaustion_preserves_exact_prior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=b"prior")
    calls: list[tuple[Path, Path]] = []
    writes: list[Path] = []

    def blocked_replace(source, destination):
        calls.append((Path(source), Path(destination)))
        raise PermissionError(13, "Access is denied", str(destination), 5)

    monkeypatch.setattr(api, "_REPLACE_RETRIES", 3, raising=False)
    monkeypatch.setattr(api, "_REPLACE_RETRY_DELAY_S", 0.0, raising=False)
    monkeypatch.setattr(api, "_replace", blocked_replace)
    with pytest.raises(PermissionError, match="held open by another program"):
        _execute(
            transaction,
            lambda path: writes.append(path),
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
        )

    assert len(calls) == 3
    assert writes == []
    assert target.read_bytes() == b"prior"
    snapshot = transaction.snapshot()
    assert snapshot.phase is api.TransactionPhase.READY_TO_RETRY
    assert snapshot.pending_actions == ()


def test_canonical_stage_does_not_retry_generic_replace_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=b"prior")
    calls = 0
    writes: list[Path] = []

    def fail_replace(_source, _destination):
        nonlocal calls
        calls += 1
        raise OSError("non-sharing replacement failure")

    monkeypatch.setattr(api, "_REPLACE_RETRIES", 3, raising=False)
    monkeypatch.setattr(api, "_REPLACE_RETRY_DELAY_S", 0.0, raising=False)
    monkeypatch.setattr(api, "_replace", fail_replace)
    with pytest.raises(OSError, match="non-sharing replacement failure"):
        _execute(
            transaction,
            lambda path: writes.append(path),
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
        )

    assert calls == 1
    assert writes == []
    assert target.read_bytes() == b"prior"


def test_canonical_stage_never_retries_failed_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=b"prior")
    real_capture = api._capture_target
    replace_calls = 0
    observation_failed = False

    def blocked_replace(_source, destination):
        nonlocal replace_calls
        replace_calls += 1
        raise PermissionError(13, "Access is denied", str(destination), 5)

    def fail_first_resolution(path, *, hash_content=True):
        nonlocal observation_failed
        if (
            replace_calls
            and not observation_failed
            and Path(path) == transaction.backup
        ):
            observation_failed = True
            raise OSError("staging observation unavailable")
        return real_capture(path, hash_content=hash_content)

    monkeypatch.setattr(api, "_REPLACE_RETRIES", 3, raising=False)
    monkeypatch.setattr(api, "_REPLACE_RETRY_DELAY_S", 0.0, raising=False)
    monkeypatch.setattr(api, "_replace", blocked_replace)
    monkeypatch.setattr(api, "_capture_target", fail_first_resolution)
    with pytest.raises(PermissionError, match="Access is denied") as caught:
        _execute(
            transaction,
            lambda path: path.write_bytes(b"new"),
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
        )

    assert replace_calls == 1
    assert isinstance(caught.value.__cause__, OSError)
    assert "staging observation unavailable" in str(caught.value.__cause__)
    assert target.read_bytes() == b"prior"


def test_partial_pool_pause_retains_resume_owner_before_retry_or_abort(
    tmp_path: Path,
) -> None:
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path)
    pool = _PartialPausePool()
    writes: list[str] = []

    with pytest.raises(OSError, match="after acquiring exclusion"):
        _execute(
            transaction,
            lambda path: writes.append(str(path)),
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
            pool=pool,
        )

    pending = transaction.snapshot()
    assert pending.phase is api.TransactionPhase.CLEANUP_PENDING
    assert pending.pending_actions == (api.RetryAction.POOL_RESUME,)
    assert pending.cleanup_token is not None
    assert pool.depth == 1
    assert writes == []
    with pytest.raises(api.TransactionStateError):
        transaction.abort(
            admission=transaction.admission,
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
        )
    with pytest.raises(api.TransactionStateError):
        _execute(
            transaction,
            lambda path: writes.append(str(path)),
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
            pool=pool,
        )

    recovered = transaction.retry_cleanup(pending.cleanup_token)
    assert recovered.phase is api.TransactionPhase.READY_TO_RETRY
    assert recovered.pending_actions == ()
    assert pool.depth == 0

    committed = _execute(
        transaction,
        lambda path: (writes.append(str(path)), path.write_bytes(b"new")),
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease,
        pool=pool,
    )
    assert committed.phase is api.TransactionPhase.COMMITTED
    assert pool.depth == 0
    assert target.read_bytes() == b"new"
    assert len(writes) == 1


def test_empty_pending_without_terminal_proof_enters_integrity_hold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=b"durable prior")
    pool = _Pool(fail_pause=1)
    real_capture = api._capture_target
    target_observations = 0
    writes: list[Path] = []

    def fail_first_terminal_verification(path, *, hash_content=True):
        nonlocal target_observations
        if Path(path) == target:
            target_observations += 1
            if target_observations == 2:
                raise OSError("terminal receipt observation transient")
        return real_capture(path, hash_content=hash_content)

    monkeypatch.setattr(api, "_capture_target", fail_first_terminal_verification)

    with pytest.raises(OSError, match="pool pause failed"):
        _execute(
            transaction,
            lambda path: writes.append(path),
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
            pool=pool,
        )

    pending = transaction.snapshot()
    assert pending.cleanup_token is not None
    assert pending.pending_actions == (api.RetryAction.POOL_RESUME,)

    with pytest.raises(api.CleanupIncomplete):
        transaction.retry_cleanup(pending.cleanup_token)

    hold = transaction.snapshot()
    assert hold.phase is api.TransactionPhase.INTEGRITY_HOLD
    assert hold.pending_actions == ()
    assert hold.retryable is False
    assert writes == []
    with pytest.raises(api.TransactionStateError):
        _execute(
            transaction,
            lambda path: writes.append(path),
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
            pool=pool,
        )
    with pytest.raises(api.TransactionStateError):
        transaction.abort(
            admission=transaction.admission,
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
        )

    recovered = transaction.retry_cleanup(pending.cleanup_token)
    assert recovered.phase is api.TransactionPhase.READY_TO_RETRY
    assert recovered.pending_actions == ()
    assert target.read_bytes() == b"durable prior"
    assert pool.resumes == [str(target.resolve())]
    assert writes == []


def test_same_stat_mutation_at_existing_target_stage_is_captured_and_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=b"original")
    pool = _DepthPool()
    admitted = target.stat()
    real_replace = api._replace
    injected: list[str] = []

    def mutate_at_stage(source, destination):
        if Path(source) == target and not injected:
            injected.append("mutated")
            target.write_bytes(b"mutated!")
            os.utime(target, ns=(admitted.st_atime_ns, admitted.st_mtime_ns))
        return real_replace(source, destination)

    monkeypatch.setattr(api, "_replace", mutate_at_stage)
    writer_paths: list[Path] = []

    with pytest.raises(api.TargetChanged):
        _execute(
            transaction,
            lambda path: writer_paths.append(path),
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
            pool=pool,
        )

    assert injected == ["mutated"]
    assert writer_paths == []
    assert not target.exists()
    assert transaction.backup.read_bytes() == b"mutated!"
    held = transaction.snapshot()
    assert held.phase is api.TransactionPhase.ROLLBACK_PENDING
    assert api.RetryAction.STAGE_PRIOR in held.pending_actions
    assert api.RetryAction.ROLLBACK in held.pending_actions
    assert api.RetryAction.POOL_RESUME in held.pending_actions
    assert pool.depth == 1


def test_absent_target_appearance_at_publication_is_not_clobbered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=None)
    real_capture = api._capture_target
    target_captures = 0

    def appear_after_last_validation(path, *, hash_content=True):
        nonlocal target_captures
        snapshot = real_capture(path, hash_content=hash_content)
        if Path(path) == target and not snapshot.exists:
            target_captures += 1
            if target_captures == 2:
                target.write_bytes(b"foreign appearance")
        return snapshot

    monkeypatch.setattr(api, "_capture_target", appear_after_last_validation)
    writer_paths: list[Path] = []

    def writer(path: Path) -> None:
        writer_paths.append(path)
        path.write_bytes(b"candidate result")

    with pytest.raises(api.TargetChanged):
        _execute(
            transaction,
            writer,
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
        )

    assert target.read_bytes() == b"foreign appearance"
    assert len(writer_paths) == 1
    assert writer_paths[0] != target
    assert writer_paths[0].read_bytes() == b"candidate result"
    pending = transaction.snapshot()
    assert pending.phase is api.TransactionPhase.ROLLBACK_PENDING
    assert pending.pending_actions[0] is api.RetryAction.ROLLBACK
    assert pending.cleanup_token is not None

    target.unlink()
    recovered = transaction.retry_cleanup(pending.cleanup_token)
    assert recovered.phase is api.TransactionPhase.READY_TO_RETRY
    assert not recovered.writer_succeeded
    assert not writer_paths[0].exists()
    committed = _execute(
        transaction,
        writer,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease,
    )
    assert committed.phase is api.TransactionPhase.COMMITTED
    assert target.read_bytes() == b"candidate result"
    assert len(writer_paths) == 2


def test_known_nonpublication_clears_attempt_and_allows_fresh_execute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=b"durable prior")
    real_link = api._link
    failed: list[str] = []
    writes: list[Path] = []

    def fail_publication_before_link(source, destination):
        if Path(destination) == target and Path(source) != transaction.backup and not failed:
            failed.append("publication")
            raise OSError("hard link unsupported before publication")
        return real_link(source, destination)

    monkeypatch.setattr(api, "_link", fail_publication_before_link)

    def writer(path: Path) -> None:
        writes.append(path)
        path.write_bytes(b"candidate result")

    with pytest.raises(OSError, match="hard link unsupported"):
        _execute(
            transaction,
            writer,
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
        )

    retryable = transaction.snapshot()
    assert retryable.phase is api.TransactionPhase.READY_TO_RETRY
    assert retryable.retryable
    assert not retryable.writer_succeeded
    assert retryable.pending_actions == ()
    assert target.read_bytes() == b"durable prior"
    assert not transaction.backup.exists()
    assert not writes[0].exists()

    committed = _execute(
        transaction,
        writer,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease,
    )
    assert committed.phase is api.TransactionPhase.COMMITTED
    assert committed.writer_succeeded
    assert target.read_bytes() == b"candidate result"
    assert len(writes) == 2


def test_post_link_observation_resolves_owned_publication_without_writer_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=None)
    real_capture = api._capture_target
    failed: list[str] = []
    writes: list[Path] = []

    def fail_first_post_link_capture(path, *, hash_content=True):
        candidate = transaction._candidate
        if (
            Path(path) == target
            and target.exists()
            and candidate.exists()
            and os.path.samefile(target, candidate)
            and not failed
        ):
            failed.append("post-link")
            raise OSError("final fingerprint transient")
        return real_capture(path, hash_content=hash_content)

    monkeypatch.setattr(api, "_capture_target", fail_first_post_link_capture)

    def writer(path: Path) -> None:
        writes.append(path)
        path.write_bytes(b"published candidate")

    try:
        _execute(
            transaction,
            writer,
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
        )
    except OSError as exc:
        assert "final fingerprint transient" in str(exc)

    snapshot = transaction.snapshot()
    if snapshot.phase is not api.TransactionPhase.COMMITTED:
        assert snapshot.cleanup_token is not None
        snapshot = transaction.retry_cleanup(snapshot.cleanup_token)

    assert failed == ["post-link"]
    assert snapshot.phase is api.TransactionPhase.COMMITTED
    assert snapshot.writer_succeeded
    assert snapshot.pending_actions == ()
    assert target.read_bytes() == b"published candidate"
    assert len(writes) == 1
    assert not writes[0].exists()


def test_publication_requires_exact_private_source_before_link_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior = b"QQQQQQQQQQQQ"
    result = b"RRRRRRRRRRRR"
    foreign = b"foreign-candidate"
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=prior)
    pool = _DepthPool()
    real_link = api._link
    replaced: list[str] = []
    writes: list[Path] = []

    def replace_source_after_real_publication_link(source, destination):
        result_value = real_link(source, destination)
        if (
            Path(source) == transaction._candidate
            and Path(destination) == target
            and not replaced
        ):
            replacement = transaction._candidate.with_name(
                transaction._candidate.name + ".foreign-link-source"
            )
            replacement.write_bytes(foreign)
            os.replace(replacement, transaction._candidate)
            replaced.append("candidate")
        return result_value

    monkeypatch.setattr(api, "_link", replace_source_after_real_publication_link)

    def writer(path: Path) -> None:
        writes.append(path)
        path.write_bytes(result)

    with pytest.raises(api.OutputTransactionError):
        _execute(
            transaction,
            writer,
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
            pool=pool,
        )

    held = transaction.snapshot()
    assert replaced == ["candidate"]
    assert held.phase is api.TransactionPhase.ROLLBACK_PENDING
    assert api.RetryAction.PUBLICATION in held.pending_actions
    assert api.RetryAction.ROLLBACK in held.pending_actions
    assert api.RetryAction.POOL_RESUME in held.pending_actions
    assert held.cleanup_token is not None
    assert target.read_bytes() == result
    assert transaction._candidate.read_bytes() == foreign
    assert transaction.backup.read_bytes() == prior
    assert pool.depth == 1
    assert len(writes) == 1

    transaction._candidate.unlink()
    with pytest.raises(api.CleanupIncomplete):
        transaction.retry_cleanup(held.cleanup_token)
    still_held = transaction.snapshot()
    assert still_held.phase is api.TransactionPhase.ROLLBACK_PENDING
    assert not transaction._candidate.exists()
    assert target.read_bytes() == result
    assert pool.depth == 1

    os.link(target, transaction._candidate)
    committed = transaction.retry_cleanup(held.cleanup_token)
    assert committed.phase is api.TransactionPhase.COMMITTED
    assert committed.pending_actions == ()
    assert target.read_bytes() == result
    assert not transaction._candidate.exists()
    assert not transaction.backup.exists()
    assert pool.depth == 0
    assert len(pool.resumes) == 1
    assert len(writes) == 1


def test_publication_terminal_observation_retries_after_authorized_source_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior = b"SSSSSSSSSSSS"
    result = b"TTTTTTTTTTTT"
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=prior)
    pool = _DepthPool()
    real_capture = api._capture_target
    failed: list[str] = []
    writes: list[Path] = []

    def fail_terminal_after_candidate_cleanup(path, *, hash_content=True):
        if (
            Path(path) == target
            and target.exists()
            and not transaction._candidate.exists()
            and transaction.backup.exists()
            and not failed
        ):
            failed.append("terminal")
            raise OSError("publication terminal observation transient")
        return real_capture(path, hash_content=hash_content)

    monkeypatch.setattr(api, "_capture_target", fail_terminal_after_candidate_cleanup)

    def writer(path: Path) -> None:
        writes.append(path)
        path.write_bytes(result)

    with pytest.raises(api.CleanupIncomplete):
        _execute(
            transaction,
            writer,
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
            pool=pool,
        )

    pending = transaction.snapshot()
    assert failed == ["terminal"]
    assert pending.phase is api.TransactionPhase.CLEANUP_PENDING
    assert api.RetryAction.PUBLICATION in pending.pending_actions
    assert api.RetryAction.POOL_RESUME in pending.pending_actions
    assert pending.cleanup_token is not None
    assert not transaction._candidate.exists()
    assert target.read_bytes() == result
    assert transaction.backup.read_bytes() == prior
    assert pool.depth == 1

    committed = transaction.retry_cleanup(pending.cleanup_token)
    assert committed.phase is api.TransactionPhase.COMMITTED
    assert committed.pending_actions == ()
    assert target.read_bytes() == result
    assert not transaction.backup.exists()
    assert pool.depth == 0
    assert len(pool.resumes) == 1
    assert len(writes) == 1


@pytest.mark.parametrize("reservation", ["candidate", "backup"])
def test_private_reservation_observation_failure_retains_exact_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reservation: str,
) -> None:
    prior = None if reservation == "candidate" else b"durable prior"
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=prior)
    private_path = transaction._candidate if reservation == "candidate" else transaction.backup
    real_capture = api._capture_target
    failed: list[str] = []
    writes: list[Path] = []

    def fail_first_private_observation(path, *, hash_content=True):
        if Path(path) == private_path and private_path.exists() and not failed:
            failed.append(reservation)
            raise OSError(f"{reservation} reservation fingerprint transient")
        return real_capture(path, hash_content=hash_content)

    monkeypatch.setattr(api, "_capture_target", fail_first_private_observation)

    with pytest.raises(OSError, match="reservation fingerprint transient"):
        _execute(
            transaction,
            lambda path: writes.append(path),
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
        )

    snapshot = transaction.snapshot()
    if snapshot.pending_actions:
        assert snapshot.cleanup_token is not None
        snapshot = transaction.retry_cleanup(snapshot.cleanup_token)

    assert failed == [reservation]
    assert snapshot.phase is api.TransactionPhase.READY_TO_RETRY
    assert not snapshot.writer_succeeded
    assert snapshot.pending_actions == ()
    assert writes == []
    if prior is None:
        assert not target.exists()
    else:
        assert target.read_bytes() == prior
    assert not transaction.backup.exists()
    assert not transaction._candidate.exists()


def test_staged_prior_observation_failure_resolves_and_restores_exact_prior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=b"durable prior")
    real_capture = api._capture_target
    failed: list[str] = []
    writes: list[Path] = []

    def fail_first_post_move_observation(path, *, hash_content=True):
        if (
            Path(path) == transaction.backup
            and transaction.backup.exists()
            and not target.exists()
            and not failed
        ):
            failed.append("post-move")
            raise OSError("staged prior fingerprint transient")
        return real_capture(path, hash_content=hash_content)

    monkeypatch.setattr(api, "_capture_target", fail_first_post_move_observation)

    with pytest.raises(OSError, match="staged prior fingerprint transient"):
        _execute(
            transaction,
            lambda path: writes.append(path),
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
        )

    snapshot = transaction.snapshot()
    if snapshot.pending_actions:
        assert snapshot.cleanup_token is not None
        snapshot = transaction.retry_cleanup(snapshot.cleanup_token)

    assert failed == ["post-move"]
    assert snapshot.phase is api.TransactionPhase.READY_TO_RETRY
    assert not snapshot.writer_succeeded
    assert writes == []
    assert target.read_bytes() == b"durable prior"
    assert not transaction.backup.exists()
    assert not transaction._candidate.exists()


def test_writer_primary_survives_candidate_observation_and_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=b"durable prior")
    real_capture = api._capture_target
    failed: list[str] = []
    writes: list[Path] = []

    def fail_first_post_writer_observation(path, *, hash_content=True):
        candidate = transaction._candidate
        if (
            Path(path) == candidate
            and candidate.exists()
            and candidate.read_bytes() == b"partial candidate"
            and not failed
        ):
            failed.append("post-writer")
            raise OSError("candidate fingerprint transient")
        return real_capture(path, hash_content=hash_content)

    monkeypatch.setattr(api, "_capture_target", fail_first_post_writer_observation)

    def writer(path: Path) -> None:
        writes.append(path)
        path.write_bytes(b"partial candidate")
        raise RuntimeError("writer primary fault")

    with pytest.raises(RuntimeError, match="writer primary fault") as caught:
        _execute(
            transaction,
            writer,
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
        )
    assert isinstance(caught.value.__cause__, OSError)

    snapshot = transaction.snapshot()
    if snapshot.pending_actions:
        assert snapshot.cleanup_token is not None
        snapshot = transaction.retry_cleanup(snapshot.cleanup_token)

    assert failed == ["post-writer"]
    assert snapshot.phase is api.TransactionPhase.READY_TO_RETRY
    assert not snapshot.writer_succeeded
    assert len(writes) == 1
    assert target.read_bytes() == b"durable prior"
    assert not transaction.backup.exists()
    assert not writes[0].exists()


def test_h10_exclusion_remains_owned_across_target_restoring_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=b"durable prior")
    pool = _DepthPool()
    real_link = api._link
    restore_depths: list[int] = []

    def fail_first_prior_restore(source, destination):
        if Path(source) == transaction.backup and Path(destination) == target:
            restore_depths.append(pool.depth)
            if len(restore_depths) == 1:
                raise OSError("rollback restore transient")
        return real_link(source, destination)

    monkeypatch.setattr(api, "_link", fail_first_prior_restore)
    writes: list[Path] = []

    def writer(path: Path) -> None:
        writes.append(path)
        path.write_bytes(b"partial candidate")
        raise RuntimeError("writer fault requiring rollback")

    with pytest.raises(RuntimeError, match="writer fault requiring rollback"):
        _execute(
            transaction,
            writer,
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
            pool=pool,
        )

    pending = transaction.snapshot()
    assert pending.phase is api.TransactionPhase.ROLLBACK_PENDING
    assert api.RetryAction.ROLLBACK in pending.pending_actions
    assert api.RetryAction.POOL_RESUME in pending.pending_actions
    assert pending.cleanup_token is not None
    assert pool.depth == 1
    assert pool.resumes == []

    recovered = transaction.retry_cleanup(pending.cleanup_token)
    assert recovered.phase is api.TransactionPhase.READY_TO_RETRY
    assert recovered.pending_actions == ()
    assert restore_depths == [1, 1]
    assert pool.depth == 0
    assert len(pool.resumes) == 1
    assert target.read_bytes() == b"durable prior"
    assert not transaction.backup.exists()
    assert not writes[0].exists()


def test_same_inode_staged_prior_corruption_holds_until_exact_bytes_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior = b"AAAAAAAAAAAA"
    changed = b"BBBBBBBBBBBB"
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=prior)
    pool = _DepthPool()
    real_capture = api._capture_target
    failed: list[int] = []
    writes: list[Path] = []

    def fail_two_post_move_observations(path, *, hash_content=True):
        if (
            Path(path) == transaction.backup
            and transaction.backup.exists()
            and not target.exists()
            and len(failed) < 2
        ):
            failed.append(len(failed) + 1)
            raise OSError("staged prior observation transient")
        return real_capture(path, hash_content=hash_content)

    monkeypatch.setattr(api, "_capture_target", fail_two_post_move_observations)

    with pytest.raises(OSError, match="staged prior observation transient"):
        _execute(
            transaction,
            lambda path: writes.append(path),
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
            pool=pool,
        )

    pending = transaction.snapshot()
    assert failed == [1, 2]
    assert pending.phase is api.TransactionPhase.ROLLBACK_PENDING
    assert api.RetryAction.STAGE_PRIOR in pending.pending_actions
    assert api.RetryAction.ROLLBACK in pending.pending_actions
    assert api.RetryAction.POOL_RESUME in pending.pending_actions
    assert pending.cleanup_token is not None
    assert not target.exists()
    assert transaction.backup.read_bytes() == prior
    assert writes == []
    assert pool.depth == 1

    admitted = transaction.admission.snapshot
    owned_stat = transaction.backup.stat()
    transaction.backup.write_bytes(changed)
    os.utime(
        transaction.backup,
        ns=(owned_stat.st_atime_ns, admitted.mtime_ns),
    )
    changed_stat = transaction.backup.stat()
    assert (changed_stat.st_dev, changed_stat.st_ino) == (
        admitted.device,
        admitted.inode,
    )
    assert changed_stat.st_size == admitted.size
    assert changed_stat.st_mtime_ns == admitted.mtime_ns

    with pytest.raises(api.CleanupIncomplete):
        transaction.retry_cleanup(pending.cleanup_token)

    held = transaction.snapshot()
    assert held.phase is api.TransactionPhase.ROLLBACK_PENDING
    assert api.RetryAction.STAGE_PRIOR in held.pending_actions
    assert api.RetryAction.ROLLBACK in held.pending_actions
    assert api.RetryAction.POOL_RESUME in held.pending_actions
    assert not target.exists()
    assert transaction.backup.read_bytes() == changed
    assert pool.depth == 1
    assert pool.resumes == []

    held_stat = transaction.backup.stat()
    transaction.backup.write_bytes(prior)
    os.utime(
        transaction.backup,
        ns=(held_stat.st_atime_ns, admitted.mtime_ns),
    )
    restored_stat = transaction.backup.stat()
    assert (restored_stat.st_dev, restored_stat.st_ino) == (
        admitted.device,
        admitted.inode,
    )

    recovered = transaction.retry_cleanup(pending.cleanup_token)
    assert recovered.phase is api.TransactionPhase.READY_TO_RETRY
    assert recovered.pending_actions == ()
    assert target.read_bytes() == prior
    assert not transaction.backup.exists()
    assert pool.depth == 0
    assert len(pool.resumes) == 1
    assert writes == []


def test_replacement_inode_staged_prior_forgery_remains_foreign_hold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior = b"CCCCCCCCCCCC"
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=prior)
    pool = _DepthPool()
    real_capture = api._capture_target
    failed: list[int] = []
    writes: list[Path] = []

    def fail_two_post_move_observations(path, *, hash_content=True):
        if (
            Path(path) == transaction.backup
            and transaction.backup.exists()
            and not target.exists()
            and len(failed) < 2
        ):
            failed.append(len(failed) + 1)
            raise OSError("staged prior observation transient")
        return real_capture(path, hash_content=hash_content)

    monkeypatch.setattr(api, "_capture_target", fail_two_post_move_observations)

    with pytest.raises(OSError, match="staged prior observation transient"):
        _execute(
            transaction,
            lambda path: writes.append(path),
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
            pool=pool,
        )

    pending = transaction.snapshot()
    assert pending.cleanup_token is not None
    admitted = transaction.admission.snapshot
    admitted_inode = (admitted.device, admitted.inode)
    forged = transaction.backup.with_name(transaction.backup.name + ".forged")
    forged.write_bytes(prior)
    forged_stat = forged.stat()
    os.utime(forged, ns=(forged_stat.st_atime_ns, admitted.mtime_ns))
    os.replace(forged, transaction.backup)
    replacement_stat = transaction.backup.stat()
    assert (replacement_stat.st_dev, replacement_stat.st_ino) != admitted_inode
    assert replacement_stat.st_size == admitted.size
    assert replacement_stat.st_mtime_ns == admitted.mtime_ns
    assert transaction.backup.read_bytes() == prior

    with pytest.raises(api.CleanupIncomplete):
        transaction.retry_cleanup(pending.cleanup_token)

    held = transaction.snapshot()
    assert held.phase is api.TransactionPhase.ROLLBACK_PENDING
    assert api.RetryAction.STAGE_PRIOR in held.pending_actions
    assert api.RetryAction.ROLLBACK in held.pending_actions
    assert api.RetryAction.POOL_RESUME in held.pending_actions
    assert not target.exists()
    assert transaction.backup.read_bytes() == prior
    assert (transaction.backup.stat().st_dev, transaction.backup.stat().st_ino) == (
        replacement_stat.st_dev,
        replacement_stat.st_ino,
    )
    assert pool.depth == 1
    assert pool.resumes == []
    assert writes == []


def test_rollback_link_integrity_race_removes_only_owned_invalid_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior = b"DDDDDDDDDDDD"
    changed = b"EEEEEEEEEEEE"
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=prior)
    pool = _DepthPool()
    real_link = api._link
    real_unlink = api._unlink
    mutated: list[str] = []
    removed_depths: list[int] = []
    writes: list[Path] = []

    def mutate_prior_inside_link(source, destination):
        if (
            Path(source) == transaction.backup
            and Path(destination) == target
            and not mutated
        ):
            source_stat = transaction.backup.stat()
            transaction.backup.write_bytes(changed)
            os.utime(
                transaction.backup,
                ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns),
            )
            mutated.append("prior")
        return real_link(source, destination)

    def record_target_unlink(path) -> None:
        if Path(path) == target:
            removed_depths.append(pool.depth)
            if len(removed_depths) == 1:
                raise OSError("invalid rollback final unlink transient")
        real_unlink(path)

    monkeypatch.setattr(api, "_link", mutate_prior_inside_link)
    monkeypatch.setattr(api, "_unlink", record_target_unlink)

    def writer(path: Path) -> None:
        writes.append(path)
        path.write_bytes(b"partial-data")
        raise RuntimeError("writer forces rollback")

    with pytest.raises(RuntimeError, match="writer forces rollback"):
        _execute(
            transaction,
            writer,
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
            pool=pool,
        )

    held = transaction.snapshot()
    assert mutated == ["prior"]
    assert removed_depths == [1]
    assert target.read_bytes() == changed
    assert transaction.backup.read_bytes() == changed
    assert held.phase is api.TransactionPhase.ROLLBACK_PENDING
    assert api.RetryAction.ROLLBACK in held.pending_actions
    assert api.RetryAction.POOL_RESUME in held.pending_actions
    assert held.cleanup_token is not None
    assert pool.depth == 1
    assert pool.resumes == []
    assert len(writes) == 1

    with pytest.raises(api.CleanupIncomplete):
        transaction.retry_cleanup(held.cleanup_token)
    assert removed_depths == [1, 1]
    assert not target.exists()
    assert transaction.backup.read_bytes() == changed
    assert pool.depth == 1

    admitted = transaction.admission.snapshot
    backup_stat = transaction.backup.stat()
    transaction.backup.write_bytes(prior)
    os.utime(
        transaction.backup,
        ns=(backup_stat.st_atime_ns, admitted.mtime_ns),
    )
    recovered = transaction.retry_cleanup(held.cleanup_token)
    assert recovered.phase is api.TransactionPhase.READY_TO_RETRY
    assert recovered.pending_actions == ()
    assert target.read_bytes() == prior
    assert not transaction.backup.exists()
    assert pool.depth == 0
    assert len(pool.resumes) == 1


def test_rollback_backup_unlink_reverifies_restored_final_before_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior = b"LLLLLLLLLLLL"
    changed = b"MMMMMMMMMMMM"
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=prior)
    pool = _DepthPool()
    real_unlink = api._unlink
    mutated_depths: list[int] = []
    writes: list[Path] = []

    def mutate_prior_inside_backup_unlink(path) -> None:
        if Path(path) == transaction.backup and not mutated_depths:
            backup_stat = transaction.backup.stat()
            transaction.backup.write_bytes(changed)
            os.utime(
                transaction.backup,
                ns=(backup_stat.st_atime_ns, backup_stat.st_mtime_ns),
            )
            mutated_depths.append(pool.depth)
        real_unlink(path)

    monkeypatch.setattr(api, "_unlink", mutate_prior_inside_backup_unlink)

    def writer(path: Path) -> None:
        writes.append(path)
        path.write_bytes(b"partial-data")
        raise RuntimeError("writer forces rollback")

    with pytest.raises(RuntimeError, match="writer forces rollback"):
        _execute(
            transaction,
            writer,
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
            pool=pool,
        )

    held = transaction.snapshot()
    assert mutated_depths == [1]
    assert held.phase is api.TransactionPhase.ROLLBACK_PENDING
    assert api.RetryAction.ROLLBACK in held.pending_actions
    assert api.RetryAction.POOL_RESUME in held.pending_actions
    assert held.cleanup_token is not None
    assert target.read_bytes() == changed
    assert not transaction.backup.exists()
    assert pool.depth == 1
    assert pool.resumes == []

    admitted = transaction.admission.snapshot
    target_stat = target.stat()
    target.write_bytes(prior)
    os.utime(target, ns=(target_stat.st_atime_ns, admitted.mtime_ns))
    recovered = transaction.retry_cleanup(held.cleanup_token)
    assert recovered.phase is api.TransactionPhase.READY_TO_RETRY
    assert recovered.pending_actions == ()
    assert target.read_bytes() == prior
    assert pool.depth == 0
    assert len(pool.resumes) == 1
    assert len(writes) == 1


def test_rollback_requires_exact_private_source_before_link_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior = b"UUUUUUUUUUUU"
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=prior)
    pool = _DepthPool()
    real_link = api._link
    removed: list[str] = []
    writes: list[Path] = []

    def remove_source_after_real_rollback_link(source, destination):
        result_value = real_link(source, destination)
        if (
            Path(source) == transaction.backup
            and Path(destination) == target
            and not removed
        ):
            transaction.backup.unlink()
            removed.append("backup")
        return result_value

    monkeypatch.setattr(api, "_link", remove_source_after_real_rollback_link)

    def writer(path: Path) -> None:
        writes.append(path)
        path.write_bytes(b"partial-data")
        raise RuntimeError("writer forces rollback")

    with pytest.raises(RuntimeError, match="writer forces rollback"):
        _execute(
            transaction,
            writer,
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
            pool=pool,
        )

    held = transaction.snapshot()
    assert removed == ["backup"]
    assert held.phase is api.TransactionPhase.ROLLBACK_PENDING
    assert api.RetryAction.ROLLBACK in held.pending_actions
    assert api.RetryAction.POOL_RESUME in held.pending_actions
    assert held.cleanup_token is not None
    assert target.read_bytes() == prior
    assert not transaction.backup.exists()
    assert pool.depth == 1
    assert len(writes) == 1

    with pytest.raises(api.CleanupIncomplete):
        transaction.retry_cleanup(held.cleanup_token)
    assert transaction.snapshot().phase is api.TransactionPhase.ROLLBACK_PENDING
    assert not transaction.backup.exists()
    assert pool.depth == 1

    os.link(target, transaction.backup)
    recovered = transaction.retry_cleanup(held.cleanup_token)
    assert recovered.phase is api.TransactionPhase.READY_TO_RETRY
    assert recovered.pending_actions == ()
    assert target.read_bytes() == prior
    assert not transaction.backup.exists()
    assert pool.depth == 0
    assert len(pool.resumes) == 1
    assert len(writes) == 1


def test_rollback_terminal_observation_retries_after_authorized_source_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior = b"VVVVVVVVVVVV"
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=prior)
    pool = _DepthPool()
    real_capture = api._capture_target
    failed: list[str] = []
    writes: list[Path] = []
    writer_attempted: list[str] = []

    def fail_terminal_after_backup_cleanup(path, *, hash_content=True):
        if (
            Path(path) == target
            and writer_attempted
            and target.exists()
            and not transaction.backup.exists()
            and not failed
        ):
            failed.append("terminal")
            raise OSError("rollback terminal observation transient")
        return real_capture(path, hash_content=hash_content)

    monkeypatch.setattr(api, "_capture_target", fail_terminal_after_backup_cleanup)

    def writer(path: Path) -> None:
        writer_attempted.append("writer")
        writes.append(path)
        path.write_bytes(b"partial-data")
        raise RuntimeError("writer forces rollback")

    with pytest.raises(RuntimeError, match="writer forces rollback"):
        _execute(
            transaction,
            writer,
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
            pool=pool,
        )

    pending = transaction.snapshot()
    assert failed == ["terminal"]
    assert pending.phase is api.TransactionPhase.ROLLBACK_PENDING
    assert api.RetryAction.ROLLBACK in pending.pending_actions
    assert api.RetryAction.POOL_RESUME in pending.pending_actions
    assert pending.cleanup_token is not None
    assert target.read_bytes() == prior
    assert not transaction.backup.exists()
    assert pool.depth == 1

    recovered = transaction.retry_cleanup(pending.cleanup_token)
    assert recovered.phase is api.TransactionPhase.READY_TO_RETRY
    assert recovered.pending_actions == ()
    assert target.read_bytes() == prior
    assert pool.depth == 0
    assert len(pool.resumes) == 1
    assert len(writes) == 1


def test_publication_link_integrity_race_never_commits_changed_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior = b"FFFFFFFFFFFF"
    result = b"GGGGGGGGGGGG"
    changed = b"HHHHHHHHHHHH"
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=prior)
    pool = _DepthPool()
    real_link = api._link
    real_unlink = api._unlink
    mutated: list[str] = []
    removed_depths: list[int] = []
    writes: list[Path] = []

    def mutate_candidate_inside_link(source, destination):
        if (
            Path(source) == transaction._candidate
            and Path(destination) == target
            and not mutated
        ):
            source_stat = transaction._candidate.stat()
            transaction._candidate.write_bytes(changed)
            os.utime(
                transaction._candidate,
                ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns),
            )
            mutated.append("candidate")
        return real_link(source, destination)

    def record_target_unlink(path) -> None:
        if Path(path) == target:
            removed_depths.append(pool.depth)
        real_unlink(path)

    monkeypatch.setattr(api, "_link", mutate_candidate_inside_link)
    monkeypatch.setattr(api, "_unlink", record_target_unlink)

    def writer(path: Path) -> None:
        writes.append(path)
        path.write_bytes(result)

    with pytest.raises(api.OutputTransactionError):
        _execute(
            transaction,
            writer,
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
            pool=pool,
        )

    snapshot = transaction.snapshot()
    assert mutated == ["candidate"]
    assert removed_depths == [1]
    assert target.read_bytes() == prior
    assert snapshot.phase is not api.TransactionPhase.COMMITTED
    assert len(writes) == 1
    assert pool.depth == 0 or api.RetryAction.POOL_RESUME in snapshot.pending_actions


def test_publication_candidate_unlink_reverifies_final_before_backup_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior = b"NNNNNNNNNNNN"
    result = b"OOOOOOOOOOOO"
    changed = b"PPPPPPPPPPPP"
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=prior)
    pool = _DepthPool()
    real_unlink = api._unlink
    mutated_depths: list[int] = []
    writes: list[Path] = []

    def mutate_candidate_inside_cleanup_unlink(path) -> None:
        if Path(path) == transaction._candidate and not mutated_depths:
            candidate_stat = transaction._candidate.stat()
            transaction._candidate.write_bytes(changed)
            os.utime(
                transaction._candidate,
                ns=(candidate_stat.st_atime_ns, candidate_stat.st_mtime_ns),
            )
            mutated_depths.append(pool.depth)
        real_unlink(path)

    monkeypatch.setattr(api, "_unlink", mutate_candidate_inside_cleanup_unlink)

    def writer(path: Path) -> None:
        writes.append(path)
        path.write_bytes(result)

    with pytest.raises(api.OutputTransactionError):
        _execute(
            transaction,
            writer,
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
            pool=pool,
        )

    snapshot = transaction.snapshot()
    assert mutated_depths == [1]
    assert snapshot.phase is not api.TransactionPhase.COMMITTED
    assert target.read_bytes() == prior
    assert not transaction.backup.exists()
    assert len(writes) == 1
    assert pool.depth == 0
    assert len(pool.resumes) == 1


def test_candidate_reservation_replacement_is_foreign_before_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior = b"IIIIIIIIIIII"
    foreign = b"foreign-candidate"
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=prior)
    real_close = api.os.close
    reservation_identity: list[tuple[int, int]] = []
    writes: list[Path] = []

    def replace_candidate_after_close(descriptor: int) -> None:
        descriptor_stat = os.fstat(descriptor)
        real_close(descriptor)
        if transaction._candidate.exists() and not reservation_identity:
            path_stat = transaction._candidate.stat()
            if (path_stat.st_dev, path_stat.st_ino) == (
                descriptor_stat.st_dev,
                descriptor_stat.st_ino,
            ):
                reservation_identity.append(
                    (descriptor_stat.st_dev, descriptor_stat.st_ino)
                )
                replacement = transaction._candidate.with_name(
                    transaction._candidate.name + ".foreign"
                )
                replacement.write_bytes(foreign)
                os.replace(replacement, transaction._candidate)

    monkeypatch.setattr(api.os, "close", replace_candidate_after_close)

    with pytest.raises(api.OutputTransactionError):
        _execute(
            transaction,
            lambda path: writes.append(path),
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
        )

    assert len(reservation_identity) == 1
    replacement_stat = transaction._candidate.stat()
    assert (replacement_stat.st_dev, replacement_stat.st_ino) != reservation_identity[0]
    assert transaction._candidate.read_bytes() == foreign
    assert target.read_bytes() == prior
    assert writes == []
    assert transaction.snapshot().phase is not api.TransactionPhase.COMMITTED


def test_backup_reservation_replacement_refuses_before_target_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior = b"JJJJJJJJJJJJ"
    foreign = b"foreign-backup"
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=prior)
    real_close = api.os.close
    real_replace = api._replace
    reservation_identity: list[tuple[int, int]] = []
    replace_calls: list[tuple[Path, Path]] = []
    writes: list[Path] = []

    def replace_backup_after_close(descriptor: int) -> None:
        descriptor_stat = os.fstat(descriptor)
        real_close(descriptor)
        if transaction.backup.exists() and not reservation_identity:
            path_stat = transaction.backup.stat()
            if (path_stat.st_dev, path_stat.st_ino) == (
                descriptor_stat.st_dev,
                descriptor_stat.st_ino,
            ):
                reservation_identity.append(
                    (descriptor_stat.st_dev, descriptor_stat.st_ino)
                )
                replacement = transaction.backup.with_name(
                    transaction.backup.name + ".foreign"
                )
                replacement.write_bytes(foreign)
                os.replace(replacement, transaction.backup)

    def record_replace(source, destination):
        replace_calls.append((Path(source), Path(destination)))
        return real_replace(source, destination)

    monkeypatch.setattr(api.os, "close", replace_backup_after_close)
    monkeypatch.setattr(api, "_replace", record_replace)

    with pytest.raises(api.OutputTransactionError):
        _execute(
            transaction,
            lambda path: writes.append(path),
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
        )

    assert len(reservation_identity) == 1
    replacement_stat = transaction.backup.stat()
    assert (replacement_stat.st_dev, replacement_stat.st_ino) != reservation_identity[0]
    assert transaction.backup.read_bytes() == foreign
    assert target.read_bytes() == prior
    assert writes == []
    assert replace_calls == []
    assert transaction.snapshot().phase is not api.TransactionPhase.COMMITTED


def test_writer_namespace_replacement_is_preserved_and_never_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior = b"KKKKKKKKKKKK"
    replacement = b"writer-replacement"
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=prior)
    real_link = api._link
    reserved_identity: list[tuple[int, int]] = []
    publication_links: list[tuple[Path, Path]] = []
    writes: list[Path] = []

    def record_publication_link(source, destination):
        if Path(source) == transaction._candidate and Path(destination) == target:
            publication_links.append((Path(source), Path(destination)))
        return real_link(source, destination)

    monkeypatch.setattr(api, "_link", record_publication_link)

    def writer(path: Path) -> None:
        writes.append(path)
        reserved = path.stat()
        reserved_identity.append((reserved.st_dev, reserved.st_ino))
        path.unlink()
        path.write_bytes(replacement)

    with pytest.raises(api.OutputTransactionError):
        _execute(
            transaction,
            writer,
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
        )

    assert len(writes) == 1
    assert len(reserved_identity) == 1
    replacement_stat = writes[0].stat()
    assert (replacement_stat.st_dev, replacement_stat.st_ino) != reserved_identity[0]
    assert writes[0].read_bytes() == replacement
    assert target.read_bytes() == prior
    assert publication_links == []
    assert transaction.snapshot().phase is not api.TransactionPhase.COMMITTED


def test_positive_stage_receipt_outranks_generic_post_move_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path)
    real_replace = api._replace
    attempts: list[str] = []

    def stage_then_fail(source, destination):
        real_replace(source, destination)
        if not attempts:
            attempts.append("failed")
            raise OSError("stage failed after replace")

    monkeypatch.setattr(api, "_replace", stage_then_fail)
    writes: list[str] = []
    result = _execute(
        transaction,
        lambda path: (writes.append(str(path)), path.write_bytes(b"new")),
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        lease=lease,
    )

    assert len(writes) == 1
    assert target.read_bytes() == b"new"
    assert not transaction.backup.exists()
    assert result.phase is api.TransactionPhase.COMMITTED


@pytest.mark.parametrize("prior", [None, b"durable prior"])
def test_writer_failure_restores_prior_or_removes_partial(
    tmp_path: Path,
    prior: bytes | None,
) -> None:
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=prior)
    writes: list[str] = []

    def fail_writer(path: Path):
        writes.append("write")
        path.write_bytes(b"partial")
        raise OSError("writer failed")

    with pytest.raises(OSError, match="writer failed"):
        _execute(
            transaction,
            fail_writer,
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
        )

    assert writes == ["write"]
    if prior is None:
        assert not target.exists()
    else:
        assert target.read_bytes() == prior
    assert transaction.snapshot().phase is api.TransactionPhase.READY_TO_RETRY


def test_failed_rollback_retains_exact_cleanup_token_and_retries_without_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path)
    real_link = api._link
    attempts: list[str] = []
    writes: list[str] = []

    def fail_restore_link_once(source, destination):
        if Path(destination) == target and not attempts:
            attempts.append("failed")
            raise OSError("rollback link failed")
        return real_link(source, destination)

    monkeypatch.setattr(api, "_link", fail_restore_link_once)

    def fail_writer(path: Path):
        writes.append("write")
        path.write_bytes(b"partial")
        raise RuntimeError("writer fault")

    with pytest.raises(RuntimeError, match="writer fault"):
        _execute(
            transaction,
            fail_writer,
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
        )

    pending = transaction.snapshot()
    assert pending.phase is api.TransactionPhase.ROLLBACK_PENDING
    assert pending.pending_actions == (
        api.RetryAction.ROLLBACK,
        api.RetryAction.POOL_RESUME,
    )
    cleanup_token = pending.cleanup_token
    assert cleanup_token is not None
    with pytest.raises(api.OwnershipRefused):
        transaction.retry_cleanup(replace(cleanup_token))

    recovered = transaction.retry_cleanup(cleanup_token)
    assert recovered.phase is api.TransactionPhase.READY_TO_RETRY
    assert recovered.pending_actions == ()
    assert target.read_bytes() == b"prior"
    assert writes == ["write"]
    assert transaction.retry_cleanup(cleanup_token) == recovered


def test_rollback_refuses_foreign_occupant_and_retains_prior_and_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=b"durable prior")
    real_replace = api._replace
    real_link = os.link
    injected: list[str] = []

    def inject_before_legacy_restore(source, destination):
        if Path(destination) == target and not injected:
            injected.append("replace")
            target.write_bytes(b"foreign occupant")
        return real_replace(source, destination)

    def inject_before_no_clobber_restore(source, destination):
        if Path(destination) == target and not injected:
            injected.append("link")
            target.write_bytes(b"foreign occupant")
        return real_link(source, destination)

    monkeypatch.setattr(api, "_replace", inject_before_legacy_restore)
    monkeypatch.setattr(api, "_link", inject_before_no_clobber_restore, raising=False)
    writer_paths: list[Path] = []

    def fail_writer(path: Path) -> None:
        writer_paths.append(path)
        path.write_bytes(b"partial candidate")
        raise RuntimeError("writer fault")

    with pytest.raises(RuntimeError, match="writer fault"):
        _execute(
            transaction,
            fail_writer,
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
        )

    assert injected in (["replace"], ["link"])
    assert target.read_bytes() == b"foreign occupant"
    assert len(writer_paths) == 1
    assert writer_paths[0] != target
    assert writer_paths[0].read_bytes() == b"partial candidate"
    assert transaction.backup.read_bytes() == b"durable prior"
    pending = transaction.snapshot()
    assert pending.phase is api.TransactionPhase.ROLLBACK_PENDING
    assert api.RetryAction.ROLLBACK in pending.pending_actions
    assert pending.cleanup_token is not None

    with pytest.raises(api.CleanupIncomplete):
        transaction.retry_cleanup(pending.cleanup_token)
    assert target.read_bytes() == b"foreign occupant"
    assert transaction.backup.read_bytes() == b"durable prior"
    assert writer_paths[0].read_bytes() == b"partial candidate"

    target.unlink()
    recovered = transaction.retry_cleanup(pending.cleanup_token)
    assert recovered.phase is api.TransactionPhase.READY_TO_RETRY
    assert target.read_bytes() == b"durable prior"
    assert not transaction.backup.exists()
    assert not writer_paths[0].exists()


def test_failed_backup_unlink_retains_retry_phase_without_replaying_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path)
    real_unlink = api._unlink
    writes: list[str] = []
    attempts: list[str] = []

    def fail_backup_once(path):
        if Path(path) == transaction.backup and not attempts:
            attempts.append("failed")
            raise OSError("backup unlink failed")
        return real_unlink(path)

    monkeypatch.setattr(api, "_unlink", fail_backup_once)

    def writer(path: Path):
        writes.append("write")
        path.write_bytes(b"complete")

    with pytest.raises(api.CleanupIncomplete):
        _execute(
            transaction,
            writer,
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
        )

    pending = transaction.snapshot()
    assert pending.phase is api.TransactionPhase.CLEANUP_PENDING
    assert pending.pending_actions == (api.RetryAction.BACKUP_UNLINK,)
    assert pending.cleanup_token is not None
    assert target.read_bytes() == b"complete"
    assert transaction.backup.read_bytes() == b"prior"

    complete = transaction.retry_cleanup(pending.cleanup_token)
    assert complete.phase is api.TransactionPhase.COMMITTED
    assert complete.pending_actions == ()
    assert writes == ["write"]
    assert not transaction.backup.exists()
    assert transaction.retry_cleanup(pending.cleanup_token) == complete


def test_failed_pool_resume_retains_pool_owner_without_replaying_writer(
    tmp_path: Path,
) -> None:
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=None)
    pool = _Pool(fail_resume=1)
    writes: list[str] = []

    def writer(path: Path):
        writes.append("write")
        path.write_bytes(b"complete")

    with pytest.raises(api.CleanupIncomplete):
        _execute(
            transaction,
            writer,
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
            pool=pool,
        )

    pending = transaction.snapshot()
    assert pending.phase is api.TransactionPhase.CLEANUP_PENDING
    assert pending.pending_actions == (api.RetryAction.POOL_RESUME,)
    assert pending.cleanup_token is not None
    assert writes == ["write"]

    complete = transaction.retry_cleanup(pending.cleanup_token)
    assert complete.phase is api.TransactionPhase.COMMITTED
    assert complete.pending_actions == ()
    assert pool.resumes == [str(target.resolve()), str(target.resolve())]
    assert writes == ["write"]


def test_unowned_backup_refuses_without_destroying_only_prior(
    tmp_path: Path,
) -> None:
    (
        api,
        target,
        _coordinator,
        transaction,
        transaction_owner,
        target_owner,
        _owners_by_role,
        lease,
    ) = _prepared(tmp_path, prior=None)
    transaction.backup.write_bytes(b"only durable prior")
    writes: list[str] = []

    with pytest.raises(api.TransactionStateError, match="unowned backup"):
        _execute(
            transaction,
            lambda path: writes.append(str(path)),
            transaction_owner=transaction_owner,
            target_owner=target_owner,
            lease=lease,
        )

    assert writes == []
    assert not target.exists()
    assert transaction.backup.read_bytes() == b"only durable prior"


def test_xye_stale_tail_failure_withholds_publication_and_retries_exact_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()
    coordinator = api.OutputTransactionCoordinator()
    run_owner = api.OwnerToken("xye-run")
    xye = coordinator.prepare_xye(tmp_path, run_owner=run_owner)
    xye.stage(run_owner, 0, b"zero")
    xye.stage(run_owner, 1, b"one")
    stale = [tmp_path / "0007.xye", tmp_path / "0008.xye"]
    for path in stale:
        path.write_bytes(b"stale")
    real_unlink = api._unlink
    failed: list[str] = []

    def fail_once(path):
        if Path(path) == stale[1] and not failed:
            failed.append("failed")
            raise OSError("stale unlink failed")
        return real_unlink(path)

    monkeypatch.setattr(api, "_unlink", fail_once)
    publications: list[tuple[tuple[int, bytes], ...]] = []

    def publish(entries):
        publications.append(tuple(entries))

    with pytest.raises(api.CleanupIncomplete):
        xye.publish(
            run_owner=run_owner,
            stale_paths=stale,
            publisher=publish,
        )

    pending = xye.snapshot()
    assert not pending.complete
    assert pending.retryable
    assert pending.pending_stale == (str(stale[1].resolve()),)
    assert pending.staged_indices == (0, 1)
    assert pending.cleanup_token is not None
    assert publications == []
    with pytest.raises(api.OwnershipRefused):
        xye.retry_publication(replace(pending.cleanup_token), publisher=publish)

    complete = xye.retry_publication(pending.cleanup_token, publisher=publish)
    assert complete.complete
    assert not complete.retryable
    assert complete.pending_stale == ()
    assert complete.staged_indices == ()
    assert publications == [((0, b"zero"), (1, b"one"))]
    assert xye.retry_publication(pending.cleanup_token, publisher=publish) == complete
    assert publications == [((0, b"zero"), (1, b"one"))]


def test_xye_first_attempt_freezes_staged_tuple_through_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()
    coordinator = api.OutputTransactionCoordinator()
    run_owner = api.OwnerToken("frozen-xye-run")
    xye = coordinator.prepare_xye(tmp_path, run_owner=run_owner)
    xye.stage(run_owner, 0, b"zero")
    xye.stage(run_owner, 1, b"one")
    stale = tmp_path / "0008.xye"
    stale.write_bytes(b"stale")
    real_unlink = api._unlink
    failed: list[str] = []

    def fail_once(path):
        if Path(path) == stale and not failed:
            failed.append("failed")
            raise OSError("stale unlink failed")
        return real_unlink(path)

    monkeypatch.setattr(api, "_unlink", fail_once)
    publications: list[tuple[tuple[int, bytes], ...]] = []
    with pytest.raises(api.CleanupIncomplete):
        xye.publish(
            run_owner=run_owner,
            stale_paths=(stale,),
            publisher=lambda entries: publications.append(tuple(entries)),
        )

    pending = xye.snapshot()
    frozen_indices = pending.staged_indices
    assert frozen_indices == (0, 1)
    with pytest.raises(api.TransactionStateError):
        xye.stage(run_owner, 2, b"late")
    assert xye.snapshot().staged_indices == frozen_indices

    complete = xye.retry_publication(
        pending.cleanup_token,
        publisher=lambda entries: publications.append(tuple(entries)),
    )
    assert complete.complete
    assert publications == [((0, b"zero"), (1, b"one"))]


def test_xye_outside_stale_paths_refuse_atomically_before_unlink_or_publish(
    tmp_path: Path,
) -> None:
    api = _api()
    directory = tmp_path / "xye"
    directory.mkdir()
    sibling = tmp_path / "xye-sibling"
    sibling.mkdir()
    traversal = directory / ".." / "foreign-traversal.xye"
    prefix_confusion = sibling / "foreign-prefix.xye"
    traversal.write_bytes(b"traversal foreign")
    prefix_confusion.write_bytes(b"prefix foreign")

    coordinator = api.OutputTransactionCoordinator()
    run_owner = api.OwnerToken("contained-xye-run")
    xye = coordinator.prepare_xye(directory, run_owner=run_owner)
    xye.stage(run_owner, 0, b"zero")
    publications: list[tuple] = []

    with pytest.raises(api.TransactionStateError):
        xye.publish(
            run_owner=run_owner,
            stale_paths=(traversal, prefix_confusion),
            publisher=lambda entries: publications.append(tuple(entries)),
        )

    assert traversal.read_bytes() == b"traversal foreign"
    assert prefix_confusion.read_bytes() == b"prefix foreign"
    assert publications == []
    snapshot = xye.snapshot()
    assert snapshot.staged_indices == (0,)
    assert snapshot.pending_stale == ()
    assert snapshot.cleanup_token is None


def test_empty_xye_publication_preserves_a_prior_tail(tmp_path: Path) -> None:
    api = _api()
    coordinator = api.OutputTransactionCoordinator()
    run_owner = api.OwnerToken("empty-xye-run")
    xye = coordinator.prepare_xye(tmp_path, run_owner=run_owner)
    prior = tmp_path / "0009.xye"
    prior.write_bytes(b"prior durable run")
    publications: list[tuple] = []

    outcome = xye.publish(
        run_owner=run_owner,
        stale_paths=(prior,),
        publisher=lambda entries: publications.append(tuple(entries)),
    )

    assert outcome.complete
    assert not outcome.retryable
    assert outcome.cleanup_token is None
    assert prior.read_bytes() == b"prior durable run"
    assert publications == []


def test_xye_incremental_epochs_clean_prior_tail_once_and_retain_run_owner(
    tmp_path: Path,
) -> None:
    api = _api()
    coordinator = api.OutputTransactionCoordinator()
    run_owner = api.OwnerToken("incremental-xye-run")
    xye = coordinator.prepare_xye(tmp_path, run_owner=run_owner)
    stale = tmp_path / "0009.xye"
    stale.write_bytes(b"prior")
    publications: list[tuple[int, ...]] = []

    def publish(entries):
        publications.append(tuple(index for index, _value in entries))
        for index, value in entries:
            (tmp_path / f"{index:04d}.xye").write_bytes(value)

    xye.stage(run_owner, 0, b"zero")
    first = xye.publish_epoch(
        run_owner=run_owner,
        stale_paths=(stale,),
        publisher=publish,
    )
    assert not first.complete and not first.retryable
    assert not stale.exists()
    assert (tmp_path / "0000.xye").read_bytes() == b"zero"

    xye.stage(run_owner, 1, b"one")
    second = xye.publish_epoch(
        run_owner=run_owner,
        # A fresh directory observation contains the run's first epoch.  It is
        # not stale and must not be deleted by the second epoch.
        stale_paths=tuple(tmp_path.glob("*.xye")),
        publisher=publish,
    )
    assert not second.complete and not second.retryable
    assert (tmp_path / "0000.xye").read_bytes() == b"zero"
    assert (tmp_path / "0001.xye").read_bytes() == b"one"

    terminal = xye.publish(
        run_owner=run_owner,
        stale_paths=tuple(tmp_path.glob("*.xye")),
        publisher=publish,
    )
    assert terminal.complete and not terminal.retryable
    assert publications == [(0,), (1,)]


def test_xye_nonterminal_retry_freezes_epoch_then_allows_later_terminal_epoch(
    tmp_path: Path,
) -> None:
    api = _api()
    coordinator = api.OutputTransactionCoordinator()
    run_owner = api.OwnerToken("retryable-incremental-xye-run")
    xye = coordinator.prepare_xye(tmp_path, run_owner=run_owner)
    xye.stage(run_owner, 0, b"zero")
    attempts: list[tuple[int, ...]] = []

    def fail_once(entries):
        labels = tuple(index for index, _value in entries)
        attempts.append(labels)
        if len(attempts) == 1:
            raise OSError("epoch publication failed")

    with pytest.raises(OSError, match="epoch publication failed"):
        xye.publish_epoch(
            run_owner=run_owner,
            stale_paths=(),
            publisher=fail_once,
        )
    pending = xye.snapshot()
    assert pending.retryable and not pending.complete
    assert pending.staged_indices == (0,)
    with pytest.raises(api.TransactionStateError):
        xye.stage(run_owner, 1, b"too early")

    retried = xye.retry_publication(
        pending.cleanup_token,
        publisher=fail_once,
    )
    assert not retried.complete and not retried.retryable
    xye.stage(run_owner, 1, b"one")
    terminal = xye.publish(
        run_owner=run_owner,
        stale_paths=(),
        publisher=lambda entries: attempts.append(tuple(
            index for index, _value in entries)),
    )
    assert terminal.complete
    assert attempts == [(0,), (0,), (1,)]


def test_kernel_is_qt_free_and_has_only_headless_transaction_mounts() -> None:
    api = _api()
    module_path = Path(api.__file__).resolve()
    source_root = module_path.parents[2]
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    forbidden_imports = ("PyQt", "PySide", "pyqtgraph", "qtpy", "xdart")
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not [name for name in imported if name.startswith(forbidden_imports)]

    allowed = {
        module_path,
        (module_path.parent / "__init__.py").resolve(),
        (module_path.parent / "analysis_artifact.py").resolve(),
    }
    public_names = (
        "OutputTransactionCoordinator",
        "OutputTransaction",
        "XyeOutputTransaction",
        "get_output_transaction_coordinator",
    )
    offenders = []
    for path in source_root.rglob("*.py"):
        resolved = path.resolve()
        if resolved in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        if "xrd_tools.io.output_transaction" in text:
            offenders.append(str(path.relative_to(source_root)))
            continue
        if any(name in text for name in public_names):
            offenders.append(str(path.relative_to(source_root)))
    assert sorted(offenders) == sorted([
        "xdart/gui/tabs/scattering/adapters/browse_loader.py",
        "xdart/gui/tabs/scattering/adapters/external_operation.py",
        "xdart/gui/tabs/scattering/adapters/run_executor.py",
        "xdart/gui/tabs/scattering/browse_1d_hydration.py",
        "xdart/gui/tabs/scattering/browse_1d_projection.py",
        "xdart/gui/tabs/scattering/browse_values.py",
        "xdart/gui/tabs/scattering/context_controller.py",
        "xdart/gui/tabs/scattering/display_values.py",
        "xdart/gui/tabs/scattering/page.py",
        "xdart/gui/tabs/scattering/processed_browser.py",
        "xdart/gui/tabs/scattering/workspace_operations.py",
        "xrd_tools/io/append.py",
        "xrd_tools/io/record_writer.py",
        "xrd_tools/reduction/average.py",
        "xrd_tools/reduction/core.py",
        "xrd_tools/reduction/reintegrate.py",
    ])


def test_c2_headless_transaction_kernel_purity_and_mount_split() -> None:
    """C2 sub-oracle; the GUI transaction mount remains deferred to P1 XYE."""
    api = _api()
    module_path = Path(api.__file__).resolve()
    source_root = module_path.parents[2]
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    forbidden = ("PyQt", "PySide", "pyqtgraph", "qtpy", "xdart")
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not [name for name in imports if name.startswith(forbidden)]

    public_names = (
        "OutputTransactionCoordinator",
        "OutputTransaction",
        "XyeOutputTransaction",
        "get_output_transaction_coordinator",
    )
    allowed = {
        "xrd_tools/io/__init__.py",
        "xrd_tools/io/analysis_artifact.py",
        "xrd_tools/io/append.py",
        "xrd_tools/io/record_writer.py",
        "xrd_tools/reduction/average.py",
        "xrd_tools/reduction/core.py",
        "xrd_tools/reduction/reintegrate.py",
    }
    observed: set[str] = set()
    for base in (source_root / "xrd_tools", source_root / "xdart/modules"):
        for path in base.rglob("*.py"):
            if path.resolve() == module_path:
                continue
            rel = str(path.relative_to(source_root))
            text = path.read_text(encoding="utf-8")
            if ("xrd_tools.io.output_transaction" in text
                    or any(name in text for name in public_names)):
                observed.add(rel)
    assert observed == allowed, (
        "the C2/headless transaction surface has unexpected mounts or lost an "
        f"allowed route: {sorted(observed)}"
    )
