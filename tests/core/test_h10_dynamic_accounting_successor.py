"""Finite public contract for dynamic H10 attempt/accounting ownership.

These tests intentionally use only the future public value/state seam.  Source
watching, stable-read policy, GUI behavior, and H23 filesystem mechanics stay
outside this module.
"""
from __future__ import annotations

import copy
import gc
from dataclasses import FrozenInstanceError
import weakref

import numpy as np
import pytest

from xrd_tools.session import ItemDisposition, ResultMode, StageLedger


def _api():
    from xrd_tools.session import (
        DynamicAccountingLimits,
        DynamicAttemptState,
        DynamicFrameIdentity,
        DynamicRunAccounting,
        DynamicRunState,
    )

    return (
        DynamicAccountingLimits,
        DynamicAttemptState,
        DynamicFrameIdentity,
        DynamicRunAccounting,
        DynamicRunState,
    )


MODE = ResultMode.one_d()
TARGET = "nexus:/tmp/dynamic-accounting.nxs"
_BOUND_OWNERS = {}


class _BoundOwner:
    pass


def _bind(accounting):
    owner, token = _BoundOwner(), object()
    accounting.writer_boundary.bind_live_session(owner, token)
    _BOUND_OWNERS[id(accounting)] = (owner, token)


def _commit(accounting, identity):
    owner, token = _BOUND_OWNERS[id(accounting)]
    seal = accounting.writer_boundary.prepare_epoch_commit(owner, token)
    accounting.writer_boundary.epoch_committed(owner, token, seal, identity)


def _abort(accounting, reason):
    owner, token = _BOUND_OWNERS[id(accounting)]
    accounting.writer_boundary.epoch_aborted(owner, token, reason)


def _accounting(*, max_groups=2, max_attempts=4, max_outstanding=4):
    (Limits, _AttemptState, _Identity, Accounting, _RunState) = _api()
    ledger = StageLedger(
        required_modes=(MODE,),
        targets_by_mode={MODE: (TARGET,)},
    )
    accounting = Accounting(
        ledger,
        run_generation=7,
        limits=Limits(
            max_groups=max_groups,
            max_attempts_per_frame=max_attempts,
            max_outstanding=max_outstanding,
        ),
    )
    _bind(accounting)
    return ledger, accounting


def _key(source, logical):
    return _api()[2](source, logical)


def _discover(accounting, source, logical, *, group="g", ordinal=None, label=None):
    logical = int(logical)
    key = _key(source, logical)
    accounting.discover(
        key,
        group=group,
        ordinal=logical if ordinal is None else ordinal,
        output_label=logical if label is None else label,
    )
    return key


def _successful_attempt(accounting, key, *, source_revision=1):
    token = accounting.begin_attempt(key, source_revision=source_revision)
    accounting.record_enqueued(token)
    accounting.record_accepted(token)
    accounting.record_completed(token, produced=(MODE,))
    accounting.record_written(token, modes=(MODE,))
    return token


def _stage_writer_durable(accounting, label):
    facade = accounting.writer_boundary
    receipt = facade.capture_receipt(label, MODE, TARGET)
    facade.commit_durable((receipt,))
    return receipt


class _MutablePayloadIdentity:
    def __init__(self, payload):
        self.value = 1
        self.payload = payload

    def __hash__(self):
        return self.value

    def __eq__(self, other):
        return self is other


@pytest.mark.parametrize("slot", ("source", "logical", "group"))
def test_dynamic_identity_grammar_refuses_mutable_payload_hashables(slot):
    """Caller mutation can never remap an owned discovery dictionary key."""
    (_Limits, _AttemptState, Identity, _Accounting, _RunState) = _api()
    hidden = np.ones(1_000_000, dtype=np.float64)
    hidden_ref = weakref.ref(hidden)
    payload = _MutablePayloadIdentity(hidden)

    if slot == "source":
        with pytest.raises(TypeError):
            Identity(payload, 0)
    elif slot == "logical":
        with pytest.raises(TypeError):
            Identity("master.nxs", payload)
    else:
        _ledger, accounting = _accounting()
        key = Identity("master.nxs", 0)
        with pytest.raises(TypeError):
            accounting.discover(
                key, group=payload, ordinal=0, output_label=0,
            )

    del payload, hidden
    gc.collect()
    assert hidden_ref() is None


def test_provisional_retry_is_one_discovery_and_one_accepted_durable_frame():
    ledger, accounting = _accounting()
    key = _discover(accounting, "master.nxs", 0)

    provisional = accounting.begin_attempt(key, source_revision=1)
    accounting.record_enqueued(provisional)
    accounting.record_failed(provisional, error="partial source", retryable=True)
    accepted = _successful_attempt(accounting, key, source_revision=2)
    immediate = ledger.snapshot()
    assert immediate.accepted == immediate.completed == frozenset((0,))
    assert (0, MODE) in immediate.written
    receipt = _stage_writer_durable(accounting, 0)

    # A writer checkpoint is exact evidence for this pending lineage epoch,
    # but not canonical H10 durability before the H23 epoch commits.
    pending = accounting.snapshot()
    assert pending.discovered == frozenset((key,))
    assert pending.enqueued == frozenset((key,))
    assert ledger.snapshot().enqueued == frozenset((key,))
    assert pending.accepted == frozenset((key,))
    assert pending.durable == frozenset()
    assert pending.pending_durable == frozenset(((key, MODE, TARGET),))
    assert pending.attempts[key] == (provisional, accepted)

    _commit(accounting, "epoch-a")
    final = accounting.snapshot()
    assert final.accepted == final.completed == final.written == frozenset((key,))
    assert final.persisted == final.durable == frozenset(((key, MODE, TARGET),))
    assert final.durable_attempts[(key, MODE, TARGET)] == accepted
    assert final.high_water["g"].durable == 0
    assert ledger.snapshot().accepted == frozenset((0,))
    assert (0, MODE, TARGET) in ledger.snapshot().durable
    assert receipt.revision == 1


def test_only_the_exact_bound_session_owner_can_promote_an_epoch():
    ledger, accounting = _accounting()
    key = _discover(accounting, "owner", 0)
    _successful_attempt(accounting, key)
    _stage_writer_durable(accounting, 0)
    owner, token = _BOUND_OWNERS[id(accounting)]
    seal = accounting.writer_boundary.prepare_epoch_commit(owner, token)
    with pytest.raises(RuntimeError, match="exact bound live session"):
        accounting.writer_boundary.epoch_committed(
            _BoundOwner(), token, seal, "bad",
        )
    with pytest.raises(RuntimeError, match="exact bound live session"):
        accounting.writer_boundary.epoch_committed(owner, object(), seal, "bad")
    with pytest.raises(RuntimeError, match="exact prepare seal"):
        accounting.writer_boundary.epoch_committed(owner, token, object(), "bad")
    assert ledger.snapshot().durable == frozenset()
    accounting.writer_boundary.epoch_committed(owner, token, seal, "good")
    assert ledger.snapshot().durable == frozenset(((0, MODE, TARGET),))


def test_completed_attempt_consumes_capacity_until_exact_canonical_receipt():
    _ledger, accounting = _accounting(max_outstanding=1)
    key0 = _discover(accounting, "capacity", 0)
    key1 = _discover(accounting, "capacity", 1)
    _successful_attempt(accounting, key0)
    with pytest.raises(ValueError, match="outstanding"):
        accounting.begin_attempt(key1, source_revision=1)
    _commit(accounting, "capacity-no-receipt")
    with pytest.raises(ValueError, match="outstanding"):
        accounting.begin_attempt(key1, source_revision=1)
    _stage_writer_durable(accounting, 0)
    _commit(accounting, "capacity-canonical")
    assert accounting.begin_attempt(key1, source_revision=1)


def test_retry_reuses_one_logical_outstanding_slot_at_capacity():
    _ledger, accounting = _accounting(max_attempts=3, max_outstanding=1)
    key0 = _discover(accounting, "retry-cap", 0)
    key1 = _discover(accounting, "retry-cap", 1)
    first = accounting.begin_attempt(key0, source_revision=1)
    accounting.record_enqueued(first)
    accounting.record_accepted(first)
    accounting.record_failed(first, error="partial", retryable=True)
    retry = accounting.begin_attempt(key0, source_revision=2)
    accounting.record_enqueued(retry)
    with pytest.raises(ValueError, match="outstanding"):
        accounting.begin_attempt(key1, source_revision=1)
    assert accounting.snapshot().enqueued == frozenset((key0,))


def test_begin_attempt_skips_outstanding_scan_through_registered_key_ceiling(
    monkeypatch,
):
    (_Limits, AttemptState, _Identity, Accounting, _RunState) = _api()

    def fail_scan(_self):
        raise AssertionError("registered-key ceiling must not scan outstanding state")

    monkeypatch.setattr(Accounting, "_outstanding_keys", fail_scan)

    _ledger, accounting = _accounting(max_attempts=3, max_outstanding=2)
    key0 = _discover(accounting, "fast-path", 0)
    key1 = _discover(accounting, "fast-path", 1)
    assert type(accounting._attempts) is dict
    assert accounting._attempts == {key0: [], key1: []}

    first = accounting.begin_attempt(key0, source_revision=1)
    accounting.record_failed(first, error="partial", retryable=True)
    retry = accounting.begin_attempt(key0, source_revision=2)
    accounting.record_failed(retry, error="terminal", retryable=False)
    accounting.begin_attempt(key1, source_revision=1)

    _ledger, accounting = _accounting(max_outstanding=2)
    key0 = _discover(accounting, "uncanonical", 0)
    key1 = _discover(accounting, "uncanonical", 1)
    completed = _successful_attempt(accounting, key0)
    snap = accounting.snapshot()
    assert snap.attempt_states[completed] is AttemptState.COMPLETED
    assert (key0, MODE, TARGET) not in snap.durable
    accounting.begin_attempt(key1, source_revision=1)


def test_same_path_revision_growth_does_not_cross_a_durable_hole():
    _ledger, accounting = _accounting()
    key0 = _discover(accounting, "growing.h5", 0)
    key1 = _discover(accounting, "growing.h5", 1)

    later = _successful_attempt(accounting, key1, source_revision=1)
    _stage_writer_durable(accounting, 1)
    _commit(accounting, "later-first")
    snap = accounting.snapshot()
    assert snap.durable_attempts[(key1, MODE, TARGET)] == later
    assert snap.high_water["g"].durable == -1

    partial = accounting.begin_attempt(key0, source_revision=1)
    accounting.record_failed(partial, error="short read", retryable=True)
    assert accounting.snapshot().high_water["g"].durable == -1
    current = _successful_attempt(accounting, key0, source_revision=2)
    _stage_writer_durable(accounting, 0)
    _commit(accounting, "hole-filled")
    snap = accounting.snapshot()
    assert snap.attempts[key0] == (partial, current)
    assert snap.high_water["g"].durable == 1


def test_failed_accepted_retry_does_not_erase_prior_durable_supplier():
    ledger, accounting = _accounting()
    key = _discover(accounting, "same-path", 0)
    first = _successful_attempt(accounting, key, source_revision=1)
    _stage_writer_durable(accounting, 0)
    _commit(accounting, "success")

    retry = accounting.begin_attempt(key, source_revision=2)
    accounting.record_accepted(retry)
    accounting.record_failed(retry, error="changed source was partial", retryable=True)
    _commit(accounting, "failed-retry")
    snap = accounting.snapshot()
    assert snap.completed_attempts[key] == retry
    assert snap.written_attempts[(key, MODE)] == first
    assert snap.durable_attempts[(key, MODE, TARGET)] == first
    assert snap.high_water["g"].durable == 0
    assert ledger.snapshot().durable == frozenset(((0, MODE, TARGET),))


def test_two_interleaved_groups_have_independent_contiguous_high_water():
    _ledger, accounting = _accounting(max_groups=2)
    a0 = _discover(accounting, "source-a", 10, group="a", ordinal=0, label=10)
    b0 = _discover(accounting, "source-b", 20, group="b", ordinal=0, label=20)
    a1 = _discover(accounting, "source-a", 11, group="a", ordinal=1, label=11)
    for key, label in ((a0, 10), (b0, 20), (a1, 11)):
        _successful_attempt(accounting, key)
        _stage_writer_durable(accounting, label)
    _commit(accounting, "interleaved")
    snap = accounting.snapshot()
    assert snap.high_water["a"].durable == 1
    assert snap.high_water["b"].durable == 0
    with pytest.raises(ValueError, match="group limit"):
        _discover(accounting, "source-c", 30, group="c", ordinal=0, label=30)


def test_duplicate_discovery_is_typed_refusal_including_exact_replay():
    _ledger, accounting = _accounting()
    key = _discover(accounting, "same", 0, group="a", ordinal=0, label=5)
    with pytest.raises(ValueError, match="duplicate discovery"):
        accounting.discover(key, group="a", ordinal=0, output_label=5)
    with pytest.raises(ValueError, match="remap"):
        accounting.discover(key, group="b", ordinal=0, output_label=5)
    with pytest.raises(ValueError, match="ordinal"):
        accounting.discover(_key("other", 0), group="a", ordinal=0, output_label=6)
    with pytest.raises(ValueError, match="output label"):
        accounting.discover(_key("other", 1), group="a", ordinal=1, output_label=5)


def test_all_five_group_high_waters_and_attempt_suppliers_are_exact():
    _ledger, accounting = _accounting(max_outstanding=8)
    keys = [
        _discover(accounting, "stages", ordinal, label=ordinal)
        for ordinal in range(5)
    ]
    attempts = []
    for key in keys:
        token = accounting.begin_attempt(key, source_revision=1)
        accounting.record_accepted(token)
        attempts.append(token)
    for token in attempts[:4]:
        accounting.record_completed(token, produced=(MODE,))
    for token in attempts[:3]:
        accounting.record_written(token, modes=(MODE,))
    persisted = accounting.writer_boundary.capture_receipt(1, MODE, TARGET)
    accounting.record_persisted((persisted,))
    _stage_writer_durable(accounting, 0)
    _commit(accounting, "stages")

    snap = accounting.snapshot()
    high = snap.high_water["g"]
    assert (high.accepted, high.completed, high.written,
            high.persisted, high.durable) == (4, 3, 2, 1, 0)
    assert snap.completed_attempts[keys[3]] == attempts[3]
    assert snap.written_attempts[(keys[2], MODE)] == attempts[2]
    assert snap.persisted_attempts[(keys[1], MODE, TARGET)] == attempts[1]
    assert snap.durable_attempts[(keys[0], MODE, TARGET)] == attempts[0]


def test_foreign_regressed_and_excess_attempts_are_refused_atomically():
    _ledger, accounting = _accounting(max_attempts=3, max_outstanding=1)
    key = _discover(accounting, "growing", 0)
    other = _discover(accounting, "growing", 1)
    first = accounting.begin_attempt(key, source_revision=2)
    with pytest.raises(ValueError, match="outstanding"):
        accounting.begin_attempt(other, source_revision=1)
    accounting.record_failed(first, error="retry", retryable=True)
    with pytest.raises(ValueError, match="source revision"):
        accounting.begin_attempt(key, source_revision=1)
    second = accounting.begin_attempt(key, source_revision=2)
    accounting.record_failed(second, error="terminal", retryable=False)
    third = accounting.begin_attempt(key, source_revision=3)
    accounting.record_failed(third, error="terminal", retryable=False)
    with pytest.raises(ValueError, match="attempt limit"):
        accounting.begin_attempt(key, source_revision=4)
    foreign = type(first)(
        key=_key("foreign", 0),
        run_generation=first.run_generation,
        revision=first.revision,
        source_revision=first.source_revision,
    )
    with pytest.raises(ValueError, match="foreign attempt"):
        accounting.record_enqueued(foreign)


@pytest.mark.parametrize("boundary", ("accepted", "completed", "written", "persisted"))
def test_stop_freezes_frontier_without_laundering_incomplete_work(boundary):
    _ledger, accounting = _accounting()
    key = _discover(accounting, "stop", 0)
    token = accounting.begin_attempt(key, source_revision=1)
    accounting.record_enqueued(token)
    accounting.record_accepted(token)
    if boundary in {"completed", "written", "persisted"}:
        accounting.record_completed(token, produced=(MODE,))
    if boundary in {"written", "persisted"}:
        accounting.record_written(token, modes=(MODE,))
    if boundary == "persisted":
        receipt = accounting.writer_boundary.capture_receipt(0, MODE, TARGET)
        accounting.record_persisted((receipt,))

    stopped = accounting.stop()
    assert stopped.state.value == "active"
    assert stopped.durable == frozenset()
    assert key in stopped.accepted
    assert key in stopped.retry_owned or key in stopped.in_flight
    with pytest.raises(RuntimeError, match="accepted frontier"):
        accounting.begin_attempt(key, source_revision=2)
    with pytest.raises(RuntimeError, match="accepted frontier"):
        _discover(accounting, "late", 1)


def test_committed_epoch_a_survives_aborted_epoch_b_without_false_receipt():
    ledger, accounting = _accounting()
    key0 = _discover(accounting, "source", 0)
    token0 = _successful_attempt(accounting, key0)
    _stage_writer_durable(accounting, 0)
    _commit(accounting, "a")

    key1 = _discover(accounting, "source", 1)
    token1 = _successful_attempt(accounting, key1, source_revision=2)
    _stage_writer_durable(accounting, 1)
    assert (key1, MODE, TARGET) in accounting.snapshot().pending_durable
    _abort(accounting, "writer rollback")

    snap = accounting.snapshot()
    assert snap.durable == frozenset(((key0, MODE, TARGET),))
    assert snap.durable_attempts[(key0, MODE, TARGET)] == token0
    assert (key1, MODE, TARGET) not in snap.durable_attempts
    assert snap.attempt_states[token1].value == "completed"
    assert token1 in snap.aborted_epoch_attempts
    assert key1 not in snap.in_flight
    assert 1 in ledger.snapshot().accepted
    assert 1 in ledger.snapshot().completed
    assert (1, MODE) in ledger.snapshot().written
    assert ledger.snapshot().durable == frozenset(((0, MODE, TARGET),))


def test_publication_drop_is_exact_attempt_truth_and_never_durable():
    ledger, accounting = _accounting()
    key = _discover(accounting, "nan", 0)
    token = _successful_attempt(accounting, key)
    facade = accounting.writer_boundary
    receipt = facade.capture_receipt(0, MODE, TARGET)
    facade.commit_publication_drop(0, MODE, receipt.revision)
    with pytest.raises(ValueError, match="publication-dropped"):
        facade.commit_durable((receipt,))
    _commit(accounting, "drop")
    snap = accounting.snapshot()
    assert snap.publication_dropped == frozenset(((key, MODE),))
    assert snap.publication_dropped_attempts[(key, MODE)] == token
    assert snap.durable == frozenset()
    assert ledger.snapshot().publication_dropped == frozenset(((0, MODE),))


def test_multimode_partial_drop_is_not_a_terminal_or_durable_frame():
    (Limits, _AttemptState, Identity, Accounting, _RunState) = _api()
    mode_1d, mode_2d = ResultMode.one_d(), ResultMode.two_d()
    target = "nexus:/tmp/multimode.nxs"
    ledger = StageLedger(
        required_modes=(mode_1d, mode_2d),
        targets_by_mode={mode_1d: (target,), mode_2d: (target,)},
    )
    accounting = Accounting(
        ledger, run_generation=1, limits=Limits(2, 2, 2),
    )
    _bind(accounting)
    key = Identity("multi", 0)
    accounting.discover(key, group="g", ordinal=0, output_label=0)
    token = accounting.begin_attempt(key, source_revision=1)
    accounting.record_accepted(token)
    accounting.record_completed(token, produced=(mode_1d, mode_2d))
    accounting.record_written(token, modes=(mode_1d, mode_2d))
    dropped = accounting.writer_boundary.capture_receipt(0, mode_1d, target)
    accounting.writer_boundary.commit_publication_drop(
        0, mode_1d, dropped.revision,
    )
    _commit(accounting, "partial-drop")
    snap = accounting.snapshot()
    assert key in snap.in_flight
    assert snap.high_water["g"].durable == -1
    assert (key, mode_1d) in snap.publication_dropped
    assert not snap.durable


@pytest.mark.parametrize("empty", ("modes", "targets"))
def test_empty_required_modes_or_targets_never_create_vacuous_high_water(empty):
    (Limits, _AttemptState, Identity, Accounting, _RunState) = _api()
    if empty == "modes":
        ledger = StageLedger(required_modes=())
    else:
        ledger = StageLedger(required_modes=(MODE,), obligations=())
    accounting = Accounting(
        ledger, run_generation=1, limits=Limits(1, 1, 1),
    )
    _bind(accounting)
    key = Identity("empty", 0)
    accounting.discover(key, group="g", ordinal=0, output_label=0)
    token = accounting.begin_attempt(key, source_revision=1)
    accounting.record_accepted(token)
    if empty == "targets":
        accounting.record_completed(token, produced=(MODE,))
        accounting.record_written(token, modes=(MODE,))
    _commit(accounting, f"empty-{empty}")
    high = accounting.snapshot().high_water["g"]
    assert high.persisted == high.durable == -1


def test_cancel_failure_and_retry_owned_states_name_the_exact_attempt():
    _ledger, accounting = _accounting()
    k0 = _discover(accounting, "source", 0)
    k1 = _discover(accounting, "source", 1)
    failed = accounting.begin_attempt(k0, source_revision=1)
    cancelled = accounting.begin_attempt(k1, source_revision=1)
    accounting.record_accepted(failed)
    accounting.record_accepted(cancelled)
    accounting.record_failed(failed, error="retry me", retryable=True)
    accounting.record_cancelled(cancelled, reason="Stop")
    accounting.record_failed(failed, error="retry me", retryable=True)
    accounting.record_cancelled(cancelled, reason="Stop")
    with pytest.raises(ValueError, match="contradictory failed"):
        accounting.record_failed(failed, error="different", retryable=True)
    with pytest.raises(ValueError, match="contradictory cancelled"):
        accounting.record_cancelled(cancelled, reason="different")
    snap = accounting.snapshot()
    assert snap.attempt_states[failed].value == "failed-retryable"
    assert snap.attempt_states[cancelled].value == "cancelled"
    assert snap.retry_owned == frozenset((k0,))
    assert snap.in_flight == frozenset()
    assert not (snap.in_flight & snap.retry_owned)
    assert snap.errors[failed] == "retry me"


def test_generation_is_exact_and_the_dynamic_owner_borrows_one_stage_ledger():
    ledger, accounting = _accounting()
    assert accounting.ledger is ledger
    key = _discover(accounting, "source", 0)
    token = accounting.begin_attempt(key, source_revision=1)
    with pytest.raises(FrozenInstanceError):
        token.source_revision = 2
    forged = type(token)(
        key=token.key,
        run_generation=token.run_generation + 1,
        revision=token.revision,
        source_revision=token.source_revision,
    )
    with pytest.raises(ValueError, match="generation"):
        accounting.record_enqueued(forged)

    # The public owner graph contains exactly the borrowed ledger, never a
    # hidden second StageLedger or a detached writer receipt publisher.
    gc.collect()
    ledger_ids = {
        id(value) for value in accounting.owner_census()
        if isinstance(value, StageLedger)
    }
    assert ledger_ids == {id(ledger)}
    assert accounting.writer_boundary.accounting is accounting
    assert not hasattr(accounting, "commit_durable")


def test_late_old_attempt_cannot_certify_a_newer_result_revision():
    ledger, accounting = _accounting(max_attempts=4)
    key = _discover(accounting, "late-write", 0)
    old = accounting.begin_attempt(key, source_revision=1)
    accounting.record_accepted(old)
    accounting.record_completed(old, produced=(MODE,))
    new = accounting.begin_attempt(key, source_revision=2)
    accounting.record_accepted(new)
    accounting.record_completed(new, produced=(MODE,))
    accounting.record_written(new, modes=(MODE,))
    with pytest.raises(ValueError, match="stale/out-of-order"):
        accounting.record_written(old, modes=(MODE,))
    assert ledger.snapshot().written == frozenset(((0, MODE),))
    assert accounting.snapshot().written_attempts[(key, MODE)] is new


def test_enqueue_replay_after_acceptance_is_exact_but_late_first_enqueue_is_not():
    _ledger, accounting = _accounting(max_attempts=3)
    key = _discover(accounting, "enqueue-replay", 0)
    enqueued = accounting.begin_attempt(key, source_revision=1)
    accounting.record_enqueued(enqueued)
    accounting.record_accepted(enqueued)
    accounting.record_enqueued(enqueued)
    accounting.record_completed(enqueued, produced=(MODE,))
    direct = accounting.begin_attempt(key, source_revision=2)
    accounting.record_accepted(direct)
    with pytest.raises(ValueError, match="only a provisional"):
        accounting.record_enqueued(direct)


def test_exact_old_write_replay_is_noop_after_a_newer_completion():
    ledger, accounting = _accounting(max_attempts=4)
    key = _discover(accounting, "write-replay", 0)
    old = accounting.begin_attempt(key, source_revision=1)
    accounting.record_accepted(old)
    accounting.record_completed(old, produced=(MODE,))
    accounting.record_written(old, modes=(MODE,))
    new = accounting.begin_attempt(key, source_revision=2)
    accounting.record_accepted(new)
    accounting.record_completed(new, produced=(MODE,))
    accounting.record_written(old, modes=(MODE,))  # already-recorded identity
    assert ledger.current_revision(0, MODE) == 2
    assert (0, MODE) not in ledger.snapshot().written


def test_abort_settles_an_accepted_pending_attempt_in_the_borrowed_ledger():
    ledger, accounting = _accounting()
    key = _discover(accounting, "pending-abort", 0)
    token = accounting.begin_attempt(key, source_revision=1)
    accounting.record_accepted(token)
    _abort(accounting, "operator abort")
    snap = accounting.snapshot()
    assert snap.attempt_states[token] is _api()[1].CANCELLED
    assert token in snap.aborted_epoch_attempts
    assert ledger.snapshot().dispositions[0] is ItemDisposition.CANCELLED_BEFORE_COMPLETION


def test_normal_finish_seal_refuses_unresolved_work_but_stop_can_settle_it():
    ledger, accounting = _accounting()
    key = _discover(accounting, "finish-pending", 0)
    token = accounting.begin_attempt(key, source_revision=1)
    accounting.record_accepted(token)
    owner, owner_token = _BOUND_OWNERS[id(accounting)]
    with pytest.raises(RuntimeError, match="unresolved"):
        accounting.writer_boundary.prepare_session_finish(
            owner, owner_token, stopped=False,
        )
    seal = accounting.writer_boundary.prepare_session_finish(
        owner, owner_token, stopped=True,
    )
    accounting.writer_boundary.session_stopped(
        owner, owner_token, seal, "Stop",
    )
    assert accounting.snapshot().state is _api()[4].STOPPED
    assert ledger.snapshot().dispositions[0] is ItemDisposition.CANCELLED_BEFORE_COMPLETION


def test_dynamic_accounting_rejects_a_stage_ledger_subclass_clone():
    class Clone(StageLedger):
        pass

    (Limits, _AttemptState, _Identity, Accounting, _RunState) = _api()
    with pytest.raises(TypeError, match="existing H10 StageLedger"):
        Accounting(
            Clone(required_modes=(MODE,), targets_by_mode={MODE: (TARGET,)}),
            run_generation=1,
            limits=Limits(1, 1, 1),
        )


def test_latest_attempt_in_flight_does_not_erase_prior_durable_high_water():
    _ledger, accounting = _accounting()
    key = _discover(accounting, "revision-growth", 0)
    first = _successful_attempt(accounting, key, source_revision=1)
    _stage_writer_durable(accounting, 0)
    _commit(accounting, "durable-a")
    assert accounting.snapshot().durable_attempts[(key, MODE, TARGET)] is first

    latest = accounting.begin_attempt(key, source_revision=2)
    accounting.record_accepted(latest)
    accepted = accounting.snapshot()
    assert accepted.in_flight == frozenset((key,))
    assert accepted.high_water["g"].durable == 0
    assert accepted.durable_attempts[(key, MODE, TARGET)] is first

    accounting.record_completed(latest, produced=(MODE,))
    completed = accounting.snapshot()
    assert completed.in_flight == frozenset((key,))
    assert completed.high_water["g"].durable == 0
    assert completed.durable_attempts[(key, MODE, TARGET)] is first


def test_unaccepted_retry_owner_is_visible_before_stop_and_settled_at_boundary():
    _ledger, accounting = _accounting()
    key = _discover(accounting, "unaccepted-retry", 0)
    token = accounting.begin_attempt(key, source_revision=1)
    accounting.record_failed(token, error="partial source", retryable=True)
    stopped = accounting.stop()
    assert stopped.state.value == "active"
    assert stopped.retry_owned == frozenset((key,))
    owner, owner_token = _BOUND_OWNERS[id(accounting)]
    seal = accounting.writer_boundary.prepare_session_finish(
        owner, owner_token, stopped=True,
    )
    accounting.writer_boundary.session_stopped(
        owner, owner_token, seal, "operator Stop",
    )
    terminal = accounting.snapshot()
    assert terminal.retry_owned == terminal.in_flight == frozenset()
    assert terminal.attempt_states[token].value == "failed"


def test_stop_freeze_keeps_terminal_unlatched_and_disposes_distinct_remainders():
    ledger, accounting = _accounting()
    accepted_key = _discover(accounting, "stop-two-phase", 0)
    accepted = accounting.begin_attempt(accepted_key, source_revision=1)
    accounting.record_accepted(accepted)
    retry_key = _discover(accounting, "stop-two-phase", 1)
    retry = accounting.begin_attempt(retry_key, source_revision=1)
    accounting.record_failed(retry, error="partial", retryable=True)

    frozen = accounting.stop()
    assert frozen.state is _api()[4].ACTIVE
    assert frozen.in_flight == frozenset((accepted_key,))
    assert frozen.retry_owned == frozenset((retry_key,))
    owner, owner_token = _BOUND_OWNERS[id(accounting)]
    seal = accounting.writer_boundary.prepare_session_finish(
        owner, owner_token, stopped=False,
    )
    assert accounting.snapshot().state is _api()[4].ACTIVE
    accounting.writer_boundary.session_stopped(
        owner, owner_token, seal, "operator Stop",
    )

    terminal = accounting.snapshot()
    assert terminal.state is _api()[4].STOPPED
    assert terminal.attempt_states[accepted] is _api()[1].CANCELLED
    assert terminal.attempt_states[retry] is _api()[1].FAILED
    assert ledger.snapshot().dispositions[0] is ItemDisposition.CANCELLED_BEFORE_COMPLETION
    assert 1 not in ledger.snapshot().dispositions
    with pytest.raises(RuntimeError):
        accounting.writer_boundary.session_stopped(
            owner, owner_token, seal, "double disposition",
        )


def test_live_attempt_must_be_settled_before_retry_and_superseded_positive_is_atomic():
    ledger, accounting = _accounting(max_attempts=4)
    key = _discover(accounting, "settled-retry", 0)
    old = accounting.begin_attempt(key, source_revision=1)
    with pytest.raises(ValueError, match="live attempt"):
        accounting.begin_attempt(key, source_revision=2)
    accounting.record_enqueued(old)
    with pytest.raises(ValueError, match="live attempt"):
        accounting.begin_attempt(key, source_revision=2)
    accounting.record_failed(old, error="partial", retryable=True)
    new = accounting.begin_attempt(key, source_revision=2)
    assert accounting.snapshot().attempt_states[old].value == "failed"

    before_ledger = ledger.snapshot()
    before = accounting.snapshot()
    accounting.record_enqueued(old)  # exact already-recorded replay is a no-op
    assert ledger.snapshot() == before_ledger
    assert accounting.snapshot().attempt_states == before.attempt_states
    for transition in (
        lambda: accounting.record_accepted(old),
        lambda: accounting.record_completed(old, produced=(MODE,)),
    ):
        with pytest.raises(ValueError, match="superseded"):
            transition()
        assert ledger.snapshot() == before_ledger
        assert accounting.snapshot().attempt_states == before.attempt_states
    assert accounting.snapshot().attempt_states[new].value == "provisional"
    accounting.record_accepted(new)
    with pytest.raises(ValueError, match="live attempt"):
        accounting.begin_attempt(key, source_revision=3)


@pytest.mark.parametrize("positive", ("persisted", "durable"))
def test_canonical_drop_refuses_later_positive_receipt_without_pending_mutation(positive):
    _ledger, accounting = _accounting()
    key = _discover(accounting, f"drop-then-{positive}", 0)
    token = _successful_attempt(accounting, key)
    facade = accounting.writer_boundary
    receipt = facade.capture_receipt(0, MODE, TARGET)
    facade.commit_publication_drop(0, MODE, receipt.revision)
    _commit(accounting, "drop")
    before = accounting.snapshot()
    with pytest.raises(ValueError, match="publication-dropped"):
        if positive == "persisted":
            accounting.record_persisted((receipt,))
        else:
            facade.commit_durable((receipt,))
    after = accounting.snapshot()
    assert after.pending_persisted == before.pending_persisted == frozenset()
    assert after.pending_durable == before.pending_durable == frozenset()
    assert after.pending_publication_dropped == frozenset()
    assert after.publication_dropped_attempts[(key, MODE)] is token


@pytest.mark.parametrize("positive", ("persisted", "durable"))
def test_canonical_positive_refuses_later_drop_without_pending_mutation(positive):
    _ledger, accounting = _accounting()
    key = _discover(accounting, f"{positive}-then-drop", 0)
    token = _successful_attempt(accounting, key)
    facade = accounting.writer_boundary
    receipt = facade.capture_receipt(0, MODE, TARGET)
    if positive == "persisted":
        accounting.record_persisted((receipt,))
    else:
        facade.commit_durable((receipt,))
    _commit(accounting, positive)
    before = accounting.snapshot()
    with pytest.raises(ValueError, match="persisted/durable"):
        facade.commit_publication_drop(0, MODE, receipt.revision)
    after = accounting.snapshot()
    assert after.pending_persisted == before.pending_persisted == frozenset()
    assert after.pending_durable == before.pending_durable == frozenset()
    assert after.pending_publication_dropped == frozenset()
    canonical = (
        after.persisted_attempts if positive == "persisted"
        else after.durable_attempts
    )
    assert canonical[(key, MODE, TARGET)] is token


def test_exact_canonical_receipt_replays_are_noops_not_new_pending_epochs():
    _ledger, accounting = _accounting()
    key = _discover(accounting, "canonical-replay", 0)
    token = _successful_attempt(accounting, key)
    facade = accounting.writer_boundary
    receipt = facade.capture_receipt(0, MODE, TARGET)
    facade.commit_durable((receipt,))
    _commit(accounting, "durable")
    facade.commit_durable((receipt,))
    accounting.record_persisted((receipt,))
    snap = accounting.snapshot()
    assert snap.pending_persisted == snap.pending_durable == frozenset()
    assert snap.durable_attempts[(key, MODE, TARGET)] is token

    other_ledger, other = _accounting()
    other_key = _discover(other, "drop-replay", 0)
    other_token = _successful_attempt(other, other_key)
    other_receipt = other.writer_boundary.capture_receipt(0, MODE, TARGET)
    other.writer_boundary.commit_publication_drop(
        0, MODE, other_receipt.revision,
    )
    _commit(other, "drop")
    other.writer_boundary.commit_publication_drop(
        0, MODE, other_receipt.revision,
    )
    other_snap = other.snapshot()
    assert other_snap.pending_publication_dropped == frozenset()
    assert other_snap.publication_dropped_attempts[(other_key, MODE)] is other_token
    assert other_ledger.snapshot().durable == frozenset()


def test_writer_boundary_is_a_unique_deferred_authority():
    _ledger, accounting = _accounting()
    with pytest.raises(TypeError, match="unique"):
        copy.copy(accounting.writer_boundary)
    with pytest.raises(TypeError, match="unique"):
        copy.deepcopy(accounting.writer_boundary)


def test_one_dynamic_overlay_may_borrow_each_exact_stage_ledger():
    (Limits, _AttemptState, _Identity, Accounting, _RunState) = _api()
    ledger = StageLedger(
        required_modes=(MODE,), targets_by_mode={MODE: (TARGET,)},
    )
    first = Accounting(
        ledger, run_generation=1, limits=Limits(1, 1, 1),
    )
    with pytest.raises(RuntimeError, match="already has a dynamic accounting owner"):
        Accounting(ledger, run_generation=2, limits=Limits(1, 1, 1))
    independent = Accounting(
        StageLedger(
            required_modes=(MODE,), targets_by_mode={MODE: (TARGET,)},
        ),
        run_generation=1,
        limits=Limits(1, 1, 1),
    )
    assert first.ledger is ledger
    assert independent.ledger is not ledger
    first_ref = weakref.ref(first)
    del first
    gc.collect()
    assert first_ref() is None
    with pytest.raises(RuntimeError, match="already has a dynamic accounting owner"):
        Accounting(ledger, run_generation=3, limits=Limits(1, 1, 1))
    independent_ledger_ref = weakref.ref(independent.ledger)
    del independent
    gc.collect()
    assert independent_ledger_ref() is None


def test_newer_durable_retires_only_older_pending_drop_for_same_mode():
    _ledger, accounting = _accounting()
    key = _discover(accounting, "drop-a-durable-b", 0)
    first = _successful_attempt(accounting, key, source_revision=1)
    facade = accounting.writer_boundary
    receipt_a = facade.capture_receipt(0, MODE, TARGET)
    facade.commit_publication_drop(0, MODE, receipt_a.revision)
    assert accounting.snapshot().pending_publication_dropped == frozenset(((key, MODE),))

    second = _successful_attempt(accounting, key, source_revision=2)
    receipt_b = facade.capture_receipt(0, MODE, TARGET)
    facade.commit_durable((receipt_b,))
    snap = accounting.snapshot()
    assert snap.pending_publication_dropped == frozenset()
    assert snap.pending_durable == frozenset(((key, MODE, TARGET),))
    assert first is not second


@pytest.mark.parametrize("positive", ("persisted", "durable"))
def test_newer_drop_retires_only_older_pending_positive_for_same_mode(positive):
    _ledger, accounting = _accounting()
    key = _discover(accounting, f"{positive}-a-drop-b", 0)
    first = _successful_attempt(accounting, key, source_revision=1)
    facade = accounting.writer_boundary
    receipt_a = facade.capture_receipt(0, MODE, TARGET)
    if positive == "persisted":
        accounting.record_persisted((receipt_a,))
    else:
        facade.commit_durable((receipt_a,))

    second = _successful_attempt(accounting, key, source_revision=2)
    receipt_b = facade.capture_receipt(0, MODE, TARGET)
    facade.commit_publication_drop(0, MODE, receipt_b.revision)
    snap = accounting.snapshot()
    assert snap.pending_persisted == frozenset()
    assert snap.pending_durable == frozenset()
    assert snap.pending_publication_dropped == frozenset(((key, MODE),))
    assert first is not second


def test_same_attempt_drop_durable_contradiction_is_atomic():
    _ledger, accounting = _accounting()
    key = _discover(accounting, "same-attempt-conflict", 0)
    _successful_attempt(accounting, key)
    facade = accounting.writer_boundary
    receipt = facade.capture_receipt(0, MODE, TARGET)
    facade.commit_publication_drop(0, MODE, receipt.revision)
    before = accounting.snapshot()
    with pytest.raises(ValueError, match="publication-dropped"):
        facade.commit_durable((receipt,))
    after = accounting.snapshot()
    assert after.pending_publication_dropped == before.pending_publication_dropped
    assert after.pending_persisted == before.pending_persisted == frozenset()
    assert after.pending_durable == before.pending_durable == frozenset()


def test_written_frame_high_water_waits_for_every_current_required_mode():
    (Limits, _AttemptState, Identity, Accounting, _RunState) = _api()
    one_d, two_d = ResultMode.one_d(), ResultMode.two_d()
    ledger = StageLedger(
        required_modes=(one_d, two_d),
        targets_by_mode={one_d: (TARGET,), two_d: (TARGET,)},
    )
    accounting = Accounting(
        ledger, run_generation=1, limits=Limits(1, 3, 1),
    )
    _bind(accounting)
    key = Identity("mixed-written", 0)
    accounting.discover(key, group="g", ordinal=0, output_label=0)
    first = accounting.begin_attempt(key, source_revision=1)
    accounting.record_accepted(first)
    accounting.record_completed(first, produced=(one_d, two_d))
    accounting.record_written(first, modes=(one_d,))
    partial = accounting.snapshot()
    assert partial.written == frozenset()
    assert partial.high_water["g"].written == -1
    assert partial.written_attempts == {(key, one_d): first}

    second = accounting.begin_attempt(key, source_revision=2)
    accounting.record_accepted(second)
    accounting.record_completed(second, produced=(two_d,))
    accounting.record_written(second, modes=(two_d,))
    complete = accounting.snapshot()
    assert complete.written == frozenset((key,))
    assert complete.high_water["g"].written == 0
    assert complete.written_attempts == {
        (key, one_d): first,
        (key, two_d): second,
    }


def test_mode_composite_durable_truth_and_finish_use_one_canonical_rule():
    (Limits, _AttemptState, Identity, Accounting, _RunState) = _api()
    one_d, two_d = ResultMode.one_d(), ResultMode.two_d()
    ledger = StageLedger(
        required_modes=(one_d, two_d),
        targets_by_mode={one_d: (TARGET,), two_d: (TARGET,)},
    )
    accounting = Accounting(
        ledger, run_generation=1, limits=Limits(1, 3, 1),
    )
    _bind(accounting)
    key = Identity("mode-composite", 0)
    accounting.discover(key, group="g", ordinal=0, output_label=0)

    first = accounting.begin_attempt(key, source_revision=1)
    accounting.record_accepted(first)
    accounting.record_completed(first, produced=(one_d,))
    accounting.record_written(first, modes=(one_d,))
    one_receipt = accounting.writer_boundary.capture_receipt(0, one_d, TARGET)
    accounting.writer_boundary.commit_durable((one_receipt,))

    second = accounting.begin_attempt(key, source_revision=2)
    accounting.record_accepted(second)
    accounting.record_completed(second, produced=(two_d,))
    accounting.record_written(second, modes=(two_d,))
    two_receipt = accounting.writer_boundary.capture_receipt(0, two_d, TARGET)
    accounting.writer_boundary.commit_durable((two_receipt,))
    _commit(accounting, "mode-composite")

    snap = accounting.snapshot()
    assert snap.high_water["g"].durable == 0
    assert key not in snap.in_flight
    owner, owner_token = _BOUND_OWNERS[id(accounting)]
    seal = accounting.writer_boundary.prepare_session_finish(
        owner, owner_token, stopped=False,
    )
    accounting.writer_boundary.epoch_prepare_failed(owner, owner_token, seal)


@pytest.mark.parametrize("transition", ("persisted", "durable", "drop"))
def test_target_truth_requires_exact_supplier_mode_to_be_written(transition):
    ledger, accounting = _accounting()
    key = _discover(accounting, f"pre-written-{transition}", 0)
    token = accounting.begin_attempt(key, source_revision=1)
    accounting.record_accepted(token)
    accounting.record_completed(token, produced=(MODE,))
    facade = accounting.writer_boundary
    receipt = facade.capture_receipt(0, MODE, TARGET)

    def transition_once():
        if transition == "persisted":
            accounting.record_persisted((receipt,))
        elif transition == "durable":
            facade.commit_durable((receipt,))
        else:
            facade.commit_publication_drop(0, MODE, receipt.revision)

    before = accounting.snapshot()
    with pytest.raises(ValueError, match="written"):
        transition_once()
    after = accounting.snapshot()
    assert after.pending_persisted == before.pending_persisted == frozenset()
    assert after.pending_durable == before.pending_durable == frozenset()
    assert (
        after.pending_publication_dropped
        == before.pending_publication_dropped
        == frozenset()
    )
    assert ledger.snapshot().persisted == ledger.snapshot().durable == frozenset()
    assert ledger.snapshot().publication_dropped == frozenset()

    accounting.record_written(token, modes=(MODE,))
    transition_once()
    _commit(accounting, f"post-written-{transition}")
    final = accounting.snapshot()
    if transition == "persisted":
        assert final.persisted_attempts[(key, MODE, TARGET)] is token
        assert final.durable == frozenset()
    elif transition == "durable":
        assert final.durable_attempts[(key, MODE, TARGET)] is token
    else:
        assert final.publication_dropped_attempts[(key, MODE)] is token
        assert final.durable == frozenset()


def test_later_multimode_epoch_abort_does_not_relabel_committed_supplier():
    (Limits, _AttemptState, Identity, Accounting, _RunState) = _api()
    one_d, two_d = ResultMode.one_d(), ResultMode.two_d()
    ledger = StageLedger(
        required_modes=(one_d, two_d),
        targets_by_mode={one_d: (TARGET,), two_d: (TARGET,)},
    )
    accounting = Accounting(
        ledger, run_generation=1, limits=Limits(1, 3, 1),
    )
    _bind(accounting)
    key = Identity("multimode-abort", 0)
    accounting.discover(key, group="g", ordinal=0, output_label=0)

    committed = accounting.begin_attempt(key, source_revision=1)
    accounting.record_accepted(committed)
    accounting.record_completed(committed, produced=(one_d,))
    accounting.record_written(committed, modes=(one_d,))
    receipt = accounting.writer_boundary.capture_receipt(0, one_d, TARGET)
    accounting.writer_boundary.commit_durable((receipt,))
    _commit(accounting, "committed-1d")

    rolled_back = accounting.begin_attempt(key, source_revision=2)
    accounting.record_accepted(rolled_back)
    accounting.record_completed(rolled_back, produced=(two_d,))
    accounting.record_written(rolled_back, modes=(two_d,))
    receipt = accounting.writer_boundary.capture_receipt(0, two_d, TARGET)
    accounting.writer_boundary.commit_durable((receipt,))
    _abort(accounting, "rollback later 2d epoch")

    snap = accounting.snapshot()
    assert snap.durable_attempts[(key, one_d, TARGET)] is committed
    assert (key, two_d, TARGET) not in snap.durable_attempts
    assert committed not in snap.aborted_epoch_attempts
    assert rolled_back in snap.aborted_epoch_attempts
    assert ledger.snapshot().durable == frozenset(((0, one_d, TARGET),))


def test_new_inflight_attempt_does_not_break_monotonic_stage_high_water_chain():
    _ledger, accounting = _accounting()
    key = _discover(accounting, "monotonic-chain", 0)
    _successful_attempt(accounting, key, source_revision=1)
    _stage_writer_durable(accounting, 0)
    _commit(accounting, "durable-prefix")

    current = accounting.begin_attempt(key, source_revision=2)
    accounting.record_accepted(current)
    snap = accounting.snapshot()
    high = snap.high_water["g"]
    assert (
        high.accepted,
        high.completed,
        high.written,
        high.persisted,
        high.durable,
    ) == (0, 0, 0, 0, 0)
    assert key in snap.in_flight


def test_one_mode_cannot_mix_target_receipts_from_different_attempts():
    (Limits, _AttemptState, Identity, Accounting, _RunState) = _api()
    first_target = "nexus:/tmp/target-a.nxs"
    second_target = "xye:/tmp/target-b"
    ledger = StageLedger(
        required_modes=(MODE,),
        targets_by_mode={MODE: (first_target, second_target)},
    )
    accounting = Accounting(
        ledger, run_generation=1, limits=Limits(1, 3, 1),
    )
    _bind(accounting)
    key = Identity("split-target-supplier", 0)
    accounting.discover(key, group="g", ordinal=0, output_label=0)

    first = accounting.begin_attempt(key, source_revision=1)
    accounting.record_accepted(first)
    accounting.record_completed(first, produced=(MODE,))
    accounting.record_written(first, modes=(MODE,))
    receipt = accounting.writer_boundary.capture_receipt(0, MODE, first_target)
    accounting.writer_boundary.commit_durable((receipt,))
    _commit(accounting, "first-target")

    second = accounting.begin_attempt(key, source_revision=2)
    accounting.record_accepted(second)
    accounting.record_completed(second, produced=(MODE,))
    accounting.record_written(second, modes=(MODE,))
    receipt = accounting.writer_boundary.capture_receipt(0, MODE, second_target)
    accounting.writer_boundary.commit_durable((receipt,))
    _commit(accounting, "second-target")

    snap = accounting.snapshot()
    assert snap.durable_attempts[(key, MODE, first_target)] is first
    assert snap.durable_attempts[(key, MODE, second_target)] is second
    assert snap.high_water["g"].durable == -1
    assert snap.in_flight == frozenset((key,))


def _terminal_accounting(case):
    ledger, accounting = _accounting(max_groups=2, max_outstanding=4)
    key = _discover(accounting, f"terminal-{case}", 0)
    token = accounting.begin_attempt(key, source_revision=1)
    accounting.record_accepted(token)
    if case == "stopped-committed-prefix":
        remainder = _discover(accounting, f"terminal-{case}", 1)
        remainder_token = accounting.begin_attempt(remainder, source_revision=1)
        accounting.record_accepted(remainder_token)
        accounting.stop()
    accounting.record_completed(token, produced=(MODE,))
    accounting.record_written(token, modes=(MODE,))
    facade = accounting.writer_boundary
    receipt = facade.capture_receipt(0, MODE, TARGET)
    facade.commit_durable((receipt,))
    owner, owner_token = _BOUND_OWNERS[id(accounting)]

    if case == "abort":
        facade.epoch_aborted(owner, owner_token, "operator Abort")
    else:
        if case == "stopped-discard":
            accounting.stop()
        seal = facade.prepare_session_finish(
            owner, owner_token, stopped=case.startswith("stopped-"),
        )
        if case == "stopped-discard":
            facade.session_stopped(
                owner, owner_token, seal, "operator Stop",
            )
        else:
            facade.session_finished(
                owner, owner_token, seal, f"terminal-{case}",
            )
    return ledger, accounting, token, receipt


@pytest.mark.parametrize(
    "case", ("abort", "finish", "stopped-discard", "stopped-committed-prefix"),
)
def test_terminal_run_cannot_rebind_restage_or_resurrect_an_epoch(case):
    ledger, accounting, token, receipt = _terminal_accounting(case)
    facade = accounting.writer_boundary
    terminal_snapshot = accounting.snapshot()
    terminal_ledger = ledger.snapshot()
    new_owner, new_owner_token = _BoundOwner(), object()
    foreign_seal = object()
    materialized = []

    def terminal_receipts():
        materialized.append(True)
        yield receipt

    operations = (
        lambda: accounting.record_persisted(terminal_receipts()),
        lambda: facade.commit_durable(terminal_receipts()),
        lambda: facade.commit_publication_drop(0, MODE, receipt.revision),
        lambda: facade.capture_receipt(0, MODE, TARGET),
        lambda: accounting.record_written(token, modes=(MODE,)),
        lambda: accounting.discover(
            _key(f"terminal-{case}", 2),
            group="g", ordinal=2, output_label=2,
        ),
        lambda: facade.prepare_epoch_commit(new_owner, new_owner_token),
        lambda: facade.epoch_committed(
            new_owner, new_owner_token, foreign_seal, "resurrected",
        ),
        lambda: facade.bind_live_session(new_owner, new_owner_token),
    )
    for operation in operations:
        with pytest.raises(RuntimeError, match="terminal"):
            operation()
        assert accounting.snapshot() == terminal_snapshot
        assert ledger.snapshot() == terminal_ledger
    assert materialized == []


@pytest.mark.parametrize("transition", ("persisted", "durable"))
def test_receipt_staging_rechecks_terminal_intent_after_materialization(
    transition,
):
    ledger, accounting = _accounting()
    key = _discover(accounting, f"terminal-iterator-{transition}", 0)
    _successful_attempt(accounting, key)
    facade = accounting.writer_boundary
    receipt = facade.capture_receipt(0, MODE, TARGET)
    owner, owner_token = _BOUND_OWNERS[id(accounting)]
    terminal = []

    def receipts():
        facade.epoch_aborted(owner, owner_token, "terminal during iteration")
        terminal.append((accounting.snapshot(), ledger.snapshot()))
        yield receipt

    stage = (
        accounting.record_persisted
        if transition == "persisted"
        else facade.commit_durable
    )
    with pytest.raises(RuntimeError, match="terminal"):
        stage(receipts())
    assert terminal
    assert accounting.snapshot() == terminal[0][0]
    assert ledger.snapshot() == terminal[0][1]
