from __future__ import annotations

import copy
import gc
import pickle
import threading
import weakref

import numpy as np
import pytest

import xrd_tools.core.physical_memory as module
from xrd_tools.core.physical_memory import (
    PhysicalRootAuthority,
    PhysicalRootExchangePhase,
    PhysicalRootLease,
)


class _InjectedCut(BaseException):
    pass


def _retain(
    authority: PhysicalRootAuthority,
    *items: tuple[object, np.ndarray],
) -> dict[object, PhysicalRootLease]:
    reservation = authority.reserve()
    for semantic, value in items:
        reservation.reserve(value, semantic)
    return reservation.commit()


def _snapshot(authority: PhysicalRootAuthority) -> tuple[int, int, int, tuple]:
    return (
        authority.retained_bytes,
        authority.retained_root_count,
        authority.semantic_references,
        authority.retained_roots,
    )


def _open_journal(exchange) -> module._ExchangeJournal:
    journal = exchange._journal
    assert type(journal) is module._ExchangeJournal
    return journal


def _receipt(exchange) -> module._ExchangeReceipt:
    receipt = _open_journal(exchange).receipt
    assert receipt is not None
    return receipt


def test_exchange_projects_survivors_incoming_and_shared_roots_at_final_cap(
) -> None:
    old = np.arange(8, dtype=np.uint8)
    shared = np.arange(8, dtype=np.uint8)
    new = np.arange(8, dtype=np.uint8)
    authority = PhysicalRootAuthority(16)
    leases = _retain(
        authority, ("victim", old), ("survivor", shared),
    )
    victim = leases["victim"]

    exchange = authority.exchange((victim,))
    exchange.reserve(shared, "shared-alias")
    # Reusing a victim semantic is valid; the same semantic on a survivor is
    # rejected in the separate collision test.
    exchange.reserve(new, "victim")
    assert exchange.projected_bytes == 16

    incoming = exchange.prepare()
    assert exchange.phase is PhysicalRootExchangePhase.STAGED
    assert exchange.prepared_leases is incoming
    assert exchange.final_projected_bytes == 16
    with pytest.raises(TypeError):
        incoming["forged"] = incoming["victim"]  # type: ignore[index]
    for accounting in (
        lambda: authority.retained_bytes,
        lambda: authority.retained_roots,
        lambda: authority.retained_root_count,
        lambda: authority.semantic_references,
    ):
        with pytest.raises(RuntimeError, match="accounting is busy"):
            accounting()
    exchange.commit()
    assert exchange.phase is PhysicalRootExchangePhase.COMMIT_PENDING
    with pytest.raises(RuntimeError, match="accounting is busy"):
        _ = authority.retained_bytes
    accepted = exchange.accept()
    assert exchange.phase is PhysicalRootExchangePhase.ACCEPTED
    assert _snapshot(authority)[:3] == (16, 2, 3)
    assert set(map(id, authority.retained_roots)) == {id(shared), id(new)}
    assert accepted["victim"] is incoming["victim"]
    assert accepted["shared-alias"] is incoming["shared-alias"]

    # The old token is stale after acceptance and cannot retire its accepted
    # same-semantic replacement.
    before = _snapshot(authority)
    victim.release()
    assert victim.released
    assert _snapshot(authority) == before

    accepted["victim"].release()
    accepted["shared-alias"].release()
    leases["survivor"].release()
    assert _snapshot(authority)[:3] == (0, 0, 0)


def test_exchange_final_capacity_excludes_replaced_root_staging_overlap() -> None:
    old = np.arange(8, dtype=np.uint8)
    new = np.arange(8, dtype=np.uint8)
    authority = PhysicalRootAuthority(8)
    victim = _retain(authority, ("row", old))["row"]
    exchange = authority.exchange((victim,))
    exchange.reserve(new, "row")
    assert exchange.projected_bytes == 8
    exchange.prepare()
    with pytest.raises(RuntimeError, match="accounting is busy"):
        _ = authority.retained_roots
    exchange.commit()
    with pytest.raises(RuntimeError, match="accounting is busy"):
        _ = authority.retained_roots
    exchange.accept()["row"].release()


def test_exchange_rejects_nonexact_duplicate_foreign_released_and_stale_victims(
) -> None:
    first = PhysicalRootAuthority(32)
    second = PhysicalRootAuthority(32)
    root = np.arange(4, dtype=np.uint8)
    lease = _retain(first, ("first", root))["first"]
    foreign = _retain(second, ("second", root.copy()))["second"]

    with pytest.raises(TypeError, match="exact tuple"):
        first.exchange([lease])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="duplicated"):
        first.exchange((lease, lease))
    with pytest.raises(ValueError, match="foreign"):
        first.exchange((foreign,))

    released = _retain(first, ("released", root.copy()))["released"]
    released.release()
    with pytest.raises(ValueError, match="released"):
        first.exchange((released,))

    replacement = root.copy()
    exchange = first.exchange((lease,))
    exchange.reserve(replacement, "first")
    exchange.prepare()
    exchange.commit()
    current = exchange.accept()["first"]
    with pytest.raises(ValueError, match="stale"):
        first.exchange((lease,))
    current.release()
    foreign.release()


def test_exchange_allows_victim_semantic_but_rejects_survivor_collision_and_drift(
) -> None:
    victim_root = np.arange(4, dtype=np.uint8)
    survivor_root = np.arange(4, dtype=np.uint8)
    authority = PhysicalRootAuthority(32)
    leases = _retain(
        authority,
        ("victim", victim_root),
        ("survivor", survivor_root),
    )
    exchange = authority.exchange((leases["victim"],))
    with pytest.raises(ValueError, match="already retained"):
        exchange.reserve(victim_root.copy(), "survivor")
    exchange.reserve(victim_root.copy(), "victim")

    # Exchange creation is ungated. Any committed-base drift before prepare is
    # detected rather than silently replacing a different graph.
    leases["survivor"].release()
    with pytest.raises(RuntimeError, match="base state drifted"):
        exchange.prepare()
    assert authority.semantic_references == 1
    leases["victim"].release()


def test_empty_exchange_uses_distinct_staged_and_rolled_back_state_identities(
) -> None:
    authority = PhysicalRootAuthority(0)
    base = authority._snapshot_state()
    exchange = authority.exchange(())
    assert exchange.prepare() == {}
    assert exchange.phase is PhysicalRootExchangePhase.STAGED
    staged = authority._snapshot_state()
    assert staged is not base
    assert staged.roots == () and staged.bindings == ()
    exchange.rollback()
    rolled = authority._snapshot_state()
    assert rolled is not staged and rolled is not base
    assert rolled.roots == () and rolled.bindings == ()
    assert exchange.phase is PhysicalRootExchangePhase.ROLLED_BACK
    exchange.rollback()
    assert exchange.phase is PhysicalRootExchangePhase.ROLLED_BACK
    with pytest.raises(RuntimeError, match="rolled back"):
        exchange.prepare()
    with pytest.raises(RuntimeError, match="rolled back"):
        exchange.commit()


@pytest.mark.parametrize(
    "alias",
    (
        pytest.param(copy.copy, id="copy"),
        pytest.param(copy.deepcopy, id="deepcopy"),
        pytest.param(pickle.dumps, id="pickle"),
    ),
)
def test_exchange_rejects_linear_journal_aliasing_without_mutation(alias) -> None:
    authority = PhysicalRootAuthority(8)
    exchange = authority.exchange(())
    callbacks = 0

    class Semantic:
        def __hash__(self) -> int:
            nonlocal callbacks
            callbacks += 1
            return 41

        def __eq__(self, _other: object) -> bool:
            nonlocal callbacks
            callbacks += 1
            return False

    exchange.reserve(np.arange(4, dtype=np.uint8), Semantic())
    prior_callbacks = callbacks
    journal = exchange._journal
    state = authority._snapshot_state()

    with pytest.raises(TypeError, match="cannot be (copied|serialized)"):
        alias(exchange)
    assert exchange._journal is journal
    assert authority._snapshot_state() is state
    assert callbacks == prior_callbacks
    exchange.rollback()


@pytest.mark.parametrize("phase", ("staged", "pending"))
def test_exchange_gate_refuses_all_mutations_and_rollback_restores_victims(
    phase: str,
) -> None:
    victim_root = np.arange(4, dtype=np.uint8)
    survivor_root = np.arange(4, dtype=np.uint8)
    incoming_root = np.arange(4, dtype=np.uint8)
    authority = PhysicalRootAuthority(32)
    leases = _retain(
        authority,
        ("victim", victim_root),
        ("survivor", survivor_root),
    )
    exchange = authority.exchange((leases["victim"],))
    competing = authority.exchange(())
    exchange.reserve(incoming_root, "incoming")
    incoming = exchange.prepare()["incoming"]
    if phase == "pending":
        exchange.commit()

    for lease in (leases["victim"], leases["survivor"], incoming):
        with pytest.raises(RuntimeError, match="busy"):
            lease.release()
        assert not lease.released
    with pytest.raises(RuntimeError, match="busy"):
        authority.close()
    with pytest.raises(RuntimeError):
        authority.reserve()
    with pytest.raises(RuntimeError, match="busy"):
        authority.exchange(())
    with pytest.raises(RuntimeError, match="busy"):
        competing.prepare()
    with pytest.raises(RuntimeError, match="busy"):
        authority._cancel_reservation()
    with pytest.raises(RuntimeError, match="busy"):
        authority._commit({}, {})

    exchange.rollback()
    assert authority.retained_roots == (victim_root, survivor_root)
    assert authority.semantic_references == 2
    incoming.release()
    assert incoming.released
    assert authority.semantic_references == 2
    leases["victim"].release()
    leases["survivor"].release()


def test_exchange_rollback_refuses_to_overwrite_unknown_third_state() -> None:
    root = np.arange(4, dtype=np.uint8)
    authority = PhysicalRootAuthority(16)
    victim = _retain(authority, ("victim", root))["victim"]
    exchange = authority.exchange((victim,))
    exchange.prepare()
    receipt = _receipt(exchange)
    third = module._AuthorityState(
        receipt.staged.roots,
        receipt.staged.bindings,
        receipt.staged.gate,
        receipt.staged.phase,
        receipt.staged.closed,
        receipt.staged.terminal_evidence,
    )
    with authority._lock:
        authority._state = third
    with pytest.raises(RuntimeError, match="state drifted"):
        exchange.rollback()
    assert authority._snapshot_state() is third
    assert authority._snapshot_state().gate is receipt.staged.gate

    # Test-only restoration proves the journal remains recoverable.
    with authority._lock:
        authority._state = receipt.staged
    exchange.rollback()
    victim.release()


def test_prepare_recovers_post_real_swap_interruption(monkeypatch) -> None:
    root = np.arange(4, dtype=np.uint8)
    authority = PhysicalRootAuthority(16)
    exchange = authority.exchange(())
    exchange.reserve(root, "incoming")
    real_swap = authority._swap_state
    interrupted = False

    def swap_then_interrupt(expected, replacement):
        nonlocal interrupted
        swapped = real_swap(expected, replacement)
        if not interrupted:
            interrupted = True
            raise _InjectedCut("prepare cut")
        return swapped

    monkeypatch.setattr(authority, "_swap_state", swap_then_interrupt)
    with pytest.raises(_InjectedCut, match="prepare cut"):
        exchange.prepare()
    prepared_receipt = _receipt(exchange)
    prepared_mapping = prepared_receipt.leases
    accepted_marker = prepared_receipt.accepted_marker
    rolled_marker = prepared_receipt.rolled_back_marker
    assert authority._snapshot_state() is prepared_receipt.staged
    assert exchange.phase is PhysicalRootExchangePhase.STAGED
    assert exchange.prepare() is prepared_mapping
    assert _receipt(exchange).accepted_marker is accepted_marker
    assert _receipt(exchange).rolled_back_marker is rolled_marker
    incoming = prepared_mapping["incoming"]
    exchange.rollback()
    incoming.release()
    assert incoming.released


@pytest.mark.parametrize("transition", ("commit", "accept", "rollback"))
def test_exchange_transition_fault_after_real_swap_is_retryable(
    monkeypatch, transition: str,
) -> None:
    old = np.arange(4, dtype=np.uint8)
    new = np.arange(4, dtype=np.uint8)
    authority = PhysicalRootAuthority(16)
    victim = _retain(authority, ("value", old))["value"]
    exchange = authority.exchange((victim,))
    exchange.reserve(new, "value")
    incoming = exchange.prepare()["value"]
    if transition == "accept":
        exchange.commit()

    real_swap = authority._swap_state
    interrupted = False

    def swap_then_interrupt(expected, replacement):
        nonlocal interrupted
        swapped = real_swap(expected, replacement)
        if not interrupted:
            interrupted = True
            raise _InjectedCut(f"{transition} cut")
        return swapped

    monkeypatch.setattr(authority, "_swap_state", swap_then_interrupt)
    operation = getattr(exchange, transition)
    with pytest.raises(_InjectedCut, match=f"{transition} cut"):
        operation()
    operation()

    if transition == "accept":
        assert authority.retained_roots == (new,)
        with pytest.raises(RuntimeError, match="accepted"):
            exchange.rollback()
        incoming.release()
        victim.release()
    else:
        if transition == "commit":
            # Same-direction commit retry succeeded; rollback is still allowed
            # from COMMIT_PENDING and accept is then the opposite terminal.
            exchange.rollback()
        assert authority.retained_roots == (old,)
        with pytest.raises(RuntimeError, match="rolled back"):
            exchange.accept()
        incoming.release()
        victim.release()


def test_claim_bind_and_base_drift_refusal_leave_authority_ungated() -> None:
    old = np.arange(8, dtype=np.uint8)
    new = np.arange(8, dtype=np.uint8)
    authority = PhysicalRootAuthority(8)
    victim = _retain(authority, ("old", old))["old"]
    exchange = authority.exchange((victim,))
    claim = exchange.claim(8)
    exchange.bind(claim, new, "new")
    assert exchange.projected_bytes == 8
    victim.release()
    with pytest.raises(RuntimeError, match="base state drifted"):
        exchange.prepare()
    exchange.rollback()
    assert exchange.phase is PhysicalRootExchangePhase.ROLLED_BACK
    assert type(exchange._journal) is module._TerminalRecord
    # No STAGED gate was published on refusal.
    reservation = authority.reserve()
    reservation.rollback()


def test_bind_matches_claim_token_by_identity_without_user_callbacks() -> None:
    root = np.arange(8, dtype=np.uint8)
    authority = PhysicalRootAuthority(8)
    exchange = authority.exchange(())
    claim = exchange.claim(8)
    callback_count = 0

    class HostileToken:
        def __hash__(self) -> int:
            nonlocal callback_count
            callback_count += 1
            raise _InjectedCut("token hash")

        def __eq__(self, _other: object) -> bool:
            nonlocal callback_count
            callback_count += 1
            raise _InjectedCut("token equality")

    before = exchange._journal
    with pytest.raises(ValueError, match="capacity token is not owned"):
        exchange.bind(HostileToken(), root, "row")
    assert callback_count == 0
    assert exchange._journal is before

    fact = exchange.bind(claim, root, "row")
    assert fact.root is root
    exchange.rollback()


def test_prepare_cas_false_at_exact_source_remains_retryable(monkeypatch) -> None:
    root = np.arange(4, dtype=np.uint8)
    authority = PhysicalRootAuthority(8)
    exchange = authority.exchange(())
    exchange.reserve(root, "incoming")
    base = authority._snapshot_state()
    monkeypatch.setattr(authority, "_swap_state", lambda _old, _new: False)

    with pytest.raises(RuntimeError, match="did not complete"):
        exchange.prepare()
    assert authority._snapshot_state() is base
    assert exchange.phase is PhysicalRootExchangePhase.PREPARED
    journal = _open_journal(exchange)
    assert journal.receipt is not None
    assert tuple(journal.roots) == ("incoming",)
    assert journal.roots["incoming"].root is root
    exchange.rollback()
    assert exchange.phase is PhysicalRootExchangePhase.ROLLED_BACK
    assert type(exchange._journal) is module._TerminalRecord


@pytest.mark.parametrize(
    "phase", ("prepared", "base-drift", "staged", "unknown"),
)
def test_failed_early_accept_never_claims_terminal_intent(
    monkeypatch, phase: str,
) -> None:
    old = np.arange(4, dtype=np.uint8)
    new = np.arange(4, dtype=np.uint8)
    authority = PhysicalRootAuthority(16)
    victim = _retain(authority, ("old", old))["old"]
    exchange = authority.exchange((victim,))
    exchange.reserve(new, "new")
    real_swap = authority._swap_state
    staged_state = None

    if phase in {"prepared", "base-drift"}:
        monkeypatch.setattr(
            authority, "_swap_state", lambda _expected, _replacement: False,
        )
        with pytest.raises(RuntimeError, match="did not complete"):
            exchange.prepare()
        monkeypatch.setattr(authority, "_swap_state", real_swap)
        if phase == "base-drift":
            victim.release()
        expected_error = "authority state drifted"
    else:
        exchange.prepare()
        staged_state = authority._snapshot_state()
        expected_error = "not committed"
        if phase == "unknown":
            third = module._AuthorityState(
                staged_state.roots,
                staged_state.bindings,
                object(),
                module._GatePhase.STAGED,
                staged_state.closed,
                staged_state.terminal_evidence,
            )
            with authority._lock:
                assert authority._state is staged_state
                authority._state = third
            expected_error = "authority state drifted"

    journal = _open_journal(exchange)
    incoming = _receipt(exchange).leases["new"]
    assert journal.terminal_intent is None
    with pytest.raises(RuntimeError, match=expected_error):
        exchange.accept()
    assert exchange._journal is journal
    assert journal.terminal_intent is None

    if phase == "unknown":
        assert staged_state is not None
        with authority._lock:
            authority._state = staged_state
    exchange.rollback()
    assert exchange.phase is PhysicalRootExchangePhase.ROLLED_BACK
    incoming.release()
    victim.release()


def test_preprepare_rollback_discards_staging_even_after_base_drift() -> None:
    old = np.arange(4, dtype=np.uint8)
    incoming = np.arange(4, dtype=np.uint8)
    authority = PhysicalRootAuthority(8)
    victim = _retain(authority, ("old", old))["old"]
    exchange = authority.exchange((victim,))
    exchange.reserve(incoming, "incoming")
    victim.release()

    exchange.rollback()
    assert exchange.phase is PhysicalRootExchangePhase.ROLLED_BACK
    assert type(exchange._journal) is module._TerminalRecord
    assert authority.semantic_references == 0


def test_private_release_accepts_only_opaque_token() -> None:
    root = np.arange(4, dtype=np.uint8)
    authority = PhysicalRootAuthority(8)
    lease = _retain(authority, ("semantic", root))["semantic"]
    with pytest.raises(TypeError, match="opaque lease token"):
        authority._release("semantic")
    assert authority.semantic_references == 1
    lease.release()


@pytest.mark.parametrize("transition", ("prepare", "commit", "accept", "rollback"))
def test_transition_post_real_swap_false_return_self_heals(
    monkeypatch, transition: str,
) -> None:
    old = np.arange(4, dtype=np.uint8)
    new = np.arange(4, dtype=np.uint8)
    authority = PhysicalRootAuthority(16)
    victim = _retain(authority, ("value", old))["value"]
    exchange = authority.exchange((victim,))
    exchange.reserve(new, "value")
    if transition != "prepare":
        exchange.prepare()
    if transition == "accept":
        exchange.commit()

    real_swap = authority._swap_state
    first = True

    def swap_then_false(expected, replacement):
        nonlocal first
        swapped = real_swap(expected, replacement)
        if first:
            first = False
            assert swapped
            return False
        return swapped

    monkeypatch.setattr(authority, "_swap_state", swap_then_false)
    getattr(exchange, transition)()
    expected_phase = {
        "prepare": PhysicalRootExchangePhase.STAGED,
        "commit": PhysicalRootExchangePhase.COMMIT_PENDING,
        "accept": PhysicalRootExchangePhase.ACCEPTED,
        "rollback": PhysicalRootExchangePhase.ROLLED_BACK,
    }[transition]
    assert exchange.phase is expected_phase

    if transition == "accept":
        exchange.accept()["value"].release()
        victim.release()
    else:
        exchange.rollback()
        victim.release()


@pytest.mark.parametrize("transition", ("prepare", "commit", "accept", "rollback"))
def test_transition_pre_real_swap_baseexception_remains_retryable(
    monkeypatch, transition: str,
) -> None:
    old = np.arange(4, dtype=np.uint8)
    new = np.arange(4, dtype=np.uint8)
    authority = PhysicalRootAuthority(16)
    victim = _retain(authority, ("value", old))["value"]
    exchange = authority.exchange((victim,))
    exchange.reserve(new, "value")
    if transition != "prepare":
        exchange.prepare()
    if transition == "accept":
        exchange.commit()

    real_swap = authority._swap_state
    first = True

    def interrupt_before_swap(expected, replacement):
        nonlocal first
        if first:
            first = False
            raise _InjectedCut(f"{transition} pre-swap")
        return real_swap(expected, replacement)

    monkeypatch.setattr(authority, "_swap_state", interrupt_before_swap)
    operation = getattr(exchange, transition)
    with pytest.raises(_InjectedCut, match=f"{transition} pre-swap"):
        operation()
    operation()

    if transition == "accept":
        assert exchange.phase is PhysicalRootExchangePhase.ACCEPTED
        exchange.accept()["value"].release()
        victim.release()
    else:
        exchange.rollback()
        assert exchange.phase is PhysicalRootExchangePhase.ROLLED_BACK
        victim.release()


@pytest.mark.parametrize("drift", ("release", "close", "reservation"))
def test_prepared_receipt_rollback_discards_after_legal_ungated_drift(
    monkeypatch, drift: str,
) -> None:
    old = np.arange(4, dtype=np.uint8)
    incoming = np.arange(4, dtype=np.uint8)
    incoming_ref = weakref.ref(incoming)
    authority = PhysicalRootAuthority(16)
    victim = _retain(authority, ("old", old))["old"]
    exchange = authority.exchange((victim,))
    exchange.reserve(incoming, "incoming")
    real_swap = authority._swap_state

    def interrupt_before_swap(_expected, _replacement):
        raise _InjectedCut("prepare before swap")

    monkeypatch.setattr(authority, "_swap_state", interrupt_before_swap)
    with pytest.raises(_InjectedCut, match="prepare before swap"):
        exchange.prepare()
    monkeypatch.setattr(authority, "_swap_state", real_swap)
    assert exchange.phase is PhysicalRootExchangePhase.PREPARED

    held = None
    if drift == "release":
        victim.release()
    elif drift == "close":
        authority.close()
    else:
        held = authority.reserve()
    del incoming
    exchange.rollback()
    assert exchange.phase is PhysicalRootExchangePhase.ROLLED_BACK
    assert type(exchange._journal) is module._TerminalRecord
    gc.collect()
    assert incoming_ref() is None
    if held is not None:
        held.rollback()


def test_terminal_phase_survives_later_release_and_close() -> None:
    old = np.arange(4, dtype=np.uint8)
    survivor = np.arange(4, dtype=np.uint8)
    incoming = np.arange(4, dtype=np.uint8)
    authority = PhysicalRootAuthority(24)
    leases = _retain(
        authority, ("old", old), ("survivor", survivor),
    )
    exchange = authority.exchange((leases["old"],))
    exchange.reserve(incoming, "incoming")
    exchange.prepare()
    exchange.commit()
    accepted = exchange.accept()
    accepted["incoming"].release()
    leases["survivor"].release()
    leases["old"].release()
    authority.close()

    assert exchange.phase is PhysicalRootExchangePhase.ACCEPTED
    assert exchange.accept() is accepted
    assert exchange.commit() is accepted
    with pytest.raises(RuntimeError, match="accepted"):
        exchange.rollback()

    authority2 = PhysicalRootAuthority(8)
    root2 = np.arange(4, dtype=np.uint8)
    victim2 = _retain(authority2, ("old", root2))["old"]
    rolled = authority2.exchange((victim2,))
    rolled.prepare()
    rolled.rollback()
    victim2.release()
    authority2.close()
    assert rolled.phase is PhysicalRootExchangePhase.ROLLED_BACK
    rolled.rollback()
    with pytest.raises(RuntimeError, match="rolled back"):
        rolled.accept()
    with pytest.raises(RuntimeError, match="rolled back"):
        rolled.commit()


@pytest.mark.parametrize("terminal", ("accept", "rollback"))
def test_post_swap_interruption_terminal_marker_survives_release_and_close(
    monkeypatch, terminal: str,
) -> None:
    old = np.arange(4, dtype=np.uint8)
    incoming_root = np.arange(4, dtype=np.uint8)
    authority = PhysicalRootAuthority(8)
    victim = _retain(authority, ("old", old))["old"]
    exchange = authority.exchange((victim,))
    exchange.reserve(incoming_root, "incoming")
    incoming = exchange.prepare()["incoming"]
    if terminal == "accept":
        exchange.commit()
    real_swap = authority._swap_state

    def swap_then_interrupt(expected, replacement):
        assert real_swap(expected, replacement)
        raise _InjectedCut(f"{terminal} terminal cut")

    monkeypatch.setattr(authority, "_swap_state", swap_then_interrupt)
    with pytest.raises(_InjectedCut, match=f"{terminal} terminal cut"):
        getattr(exchange, terminal)()
    monkeypatch.setattr(authority, "_swap_state", real_swap)

    if terminal == "accept":
        incoming.release()
        victim.release()
        authority.close()
        assert exchange.phase is PhysicalRootExchangePhase.ACCEPTED
        assert exchange.accept()["incoming"] is incoming
    else:
        victim.release()
        authority.close()
        assert exchange.phase is PhysicalRootExchangePhase.ROLLED_BACK
        exchange.rollback()
        incoming.release()


def test_terminal_compaction_drops_losing_roots_outside_authority_lock() -> None:
    authority = PhysicalRootAuthority(8)
    old = np.arange(4, dtype=np.uint8)
    incoming = np.arange(4, dtype=np.uint8)
    old_ref = weakref.ref(old)
    callback_lock_state: list[tuple[bool, bool, object, object]] = []

    def observe_accept_finalizer() -> None:
        callback_lock_state.append((
            authority._lock._is_owned(),
            exchange._lock._is_owned(),
            getattr(module._EXCHANGE_TLS, "token", None),
            exchange.phase,
        ))

    old_finalizer = weakref.finalize(
        old,
        observe_accept_finalizer,
    )
    victim = _retain(authority, ("old", old))["old"]
    exchange = authority.exchange((victim,))
    exchange.reserve(incoming, "incoming")
    exchange.prepare()
    exchange.commit()
    del old
    assert old_ref() is not None
    accepted = exchange.accept()
    gc.collect()
    assert old_ref() is None
    assert old_finalizer.alive is False
    assert callback_lock_state == [(
        False, False, None, PhysicalRootExchangePhase.ACCEPTED,
    )]
    assert type(exchange._journal) is module._TerminalRecord
    accepted["incoming"].release()

    authority2 = PhysicalRootAuthority(8)
    retained = np.arange(4, dtype=np.uint8)
    rejected = np.arange(4, dtype=np.uint8)
    rejected_ref = weakref.ref(rejected)
    rollback_lock_state: list[tuple[bool, bool, object, object]] = []

    def observe_rollback_finalizer() -> None:
        rollback_lock_state.append((
            authority2._lock._is_owned(),
            exchange2._lock._is_owned(),
            getattr(module._EXCHANGE_TLS, "token", None),
            exchange2.phase,
        ))

    rejected_finalizer = weakref.finalize(
        rejected,
        observe_rollback_finalizer,
    )
    victim2 = _retain(authority2, ("retained", retained))["retained"]
    exchange2 = authority2.exchange((victim2,))
    exchange2.reserve(rejected, "rejected")
    exchange2.prepare()
    del rejected
    assert rejected_ref() is not None
    exchange2.rollback()
    gc.collect()
    assert rejected_ref() is None
    assert rejected_finalizer.alive is False
    assert rollback_lock_state == [(
        False, False, None, PhysicalRootExchangePhase.ROLLED_BACK,
    )]
    assert type(exchange2._journal) is module._TerminalRecord
    assert exchange2._journal.leases is None
    victim2.release()


def test_open_cow_and_failed_candidate_drop_refs_after_operation_cleanup() -> None:
    authority = PhysicalRootAuthority(4)
    exchange = authority.exchange(())
    observations: list[tuple[str, bool, bool, object, object]] = []

    def observe(label: str) -> None:
        observations.append((
            label,
            authority._lock._is_owned(),
            exchange._lock._is_owned(),
            getattr(module._EXCHANGE_TLS, "token", None),
            exchange.phase,
        ))

    old_journal = _open_journal(exchange)
    old_journal_ref = weakref.ref(old_journal)
    old_finalizer = weakref.finalize(old_journal, observe, "journal")
    del old_journal
    exchange.claim(0)
    gc.collect()
    assert old_journal_ref() is None
    assert old_finalizer.alive is False

    def reject_candidate():
        candidate = np.arange(8, dtype=np.uint8)
        candidate_ref = weakref.ref(candidate)
        finalizer = weakref.finalize(candidate, observe, "candidate")
        with pytest.raises(ValueError, match="exceed limit"):
            exchange.reserve(candidate, "too-large")
        return candidate_ref, finalizer

    candidate_ref, candidate_finalizer = reject_candidate()
    gc.collect()
    assert candidate_ref() is None
    assert candidate_finalizer.alive is False
    assert observations == [
        (
            "journal", False, False, None,
            PhysicalRootExchangePhase.OPEN,
        ),
        (
            "candidate", False, False, None,
            PhysicalRootExchangePhase.OPEN,
        ),
    ]
    exchange.rollback()


def test_close_during_legacy_reservation_preserves_legacy_semantics() -> None:
    authority = PhysicalRootAuthority(16)
    retained = np.arange(4, dtype=np.uint8)
    pending = np.arange(4, dtype=np.uint8)
    lease = _retain(authority, ("retained", retained))["retained"]
    reservation = authority.reserve()
    reservation.reserve(pending, "pending")

    authority.close()
    assert authority.retained_bytes == 0
    assert authority.retained_root_count == 0
    assert authority.semantic_references == 0
    with pytest.raises(RuntimeError, match="not active"):
        reservation.commit()
    reservation.rollback()
    lease.release()
    with pytest.raises(RuntimeError, match="closed"):
        authority.reserve()


@pytest.mark.parametrize(
    ("first_direction", "second_direction"),
    (
        ("accept", "accept"),
        ("accept", "rollback"),
        ("rollback", "accept"),
        ("rollback", "rollback"),
    ),
)
def test_interrupted_terminal_survives_every_later_terminal_direction(
    monkeypatch, first_direction: str, second_direction: str,
) -> None:
    authority = PhysicalRootAuthority(32)
    original = np.arange(4, dtype=np.uint8)
    first_root = np.arange(4, dtype=np.uint8)
    second_root = np.arange(4, dtype=np.uint8)
    original_lease = _retain(authority, ("original", original))["original"]
    first = authority.exchange((original_lease,))
    first.reserve(first_root, "first")
    first_incoming = first.prepare()["first"]
    if first_direction == "accept":
        first.commit()
    first_receipt = _receipt(first)
    first_marker = (
        first_receipt.accepted_marker
        if first_direction == "accept"
        else first_receipt.rolled_back_marker
    )
    real_swap = authority._swap_state
    cut = True

    def first_terminal_cut(expected, replacement):
        nonlocal cut
        swapped = real_swap(expected, replacement)
        if cut:
            cut = False
            raise _InjectedCut("first terminal")
        return swapped

    monkeypatch.setattr(authority, "_swap_state", first_terminal_cut)
    with pytest.raises(_InjectedCut, match="first terminal"):
        getattr(first, first_direction)()
    monkeypatch.setattr(authority, "_swap_state", real_swap)
    assert module._has_terminal_marker(
        authority._snapshot_state(), first_marker,
    )

    current = first_incoming if first_direction == "accept" else original_lease
    second = authority.exchange((current,))
    second.reserve(second_root, "second")
    second_incoming = second.prepare()["second"]
    if second_direction == "accept":
        second.commit()
        second.accept()
    else:
        second.rollback()

    # The later terminal state preserved the exact older marker, so recovery
    # does not depend on terminal direction or stack order.
    assert module._has_terminal_marker(
        authority._snapshot_state(), first_marker,
    )
    if first_direction == "accept":
        assert first.accept()["first"] is first_incoming
        assert first.phase is PhysicalRootExchangePhase.ACCEPTED
    else:
        first.rollback()
        assert first.phase is PhysicalRootExchangePhase.ROLLED_BACK

    for lease in (second_incoming, first_incoming, original_lease):
        lease.release()


def test_terminal_markers_recover_non_lifo_and_prune_dead_history(
    monkeypatch,
) -> None:
    authority = PhysicalRootAuthority(64)
    original = np.arange(4, dtype=np.uint8)
    a_root = np.arange(4, dtype=np.uint8)
    b_root = np.arange(4, dtype=np.uint8)
    c_root = np.arange(4, dtype=np.uint8)
    d_root = np.arange(4, dtype=np.uint8)
    original_lease = _retain(authority, ("original", original))["original"]

    def interrupted_accept(victim, root, semantic):
        exchange = authority.exchange((victim,))
        exchange.reserve(root, semantic)
        incoming = exchange.prepare()[semantic]
        exchange.commit()
        receipt = _receipt(exchange)
        marker = receipt.accepted_marker
        marker_ref = weakref.ref(marker)
        losing_marker = receipt.rolled_back_marker
        losing_marker_ref = weakref.ref(losing_marker)
        real_swap = authority._swap_state

        def terminal_cut(expected, replacement):
            assert real_swap(expected, replacement)
            raise _InjectedCut(f"{semantic} cut")

        monkeypatch.setattr(authority, "_swap_state", terminal_cut)
        with pytest.raises(_InjectedCut, match=f"{semantic} cut"):
            exchange.accept()
        monkeypatch.setattr(authority, "_swap_state", real_swap)
        del receipt, marker, losing_marker
        return exchange, incoming, marker_ref, losing_marker_ref

    first, first_incoming, first_marker_ref, first_losing_ref = interrupted_accept(
        original_lease, a_root, "a",
    )
    second, second_incoming, second_marker_ref, second_losing_ref = interrupted_accept(
        first_incoming, b_root, "b",
    )
    state = authority._snapshot_state()
    assert first_marker_ref() is not None
    assert second_marker_ref() is not None
    assert first_losing_ref() is not None
    assert second_losing_ref() is not None
    assert first_marker_ref() is not second_marker_ref()
    assert module._has_terminal_marker(state, first_marker_ref())
    assert module._has_terminal_marker(state, second_marker_ref())

    # Recover the older operation first. Its marker dies after durable local
    # compaction, while the newer unresolved marker stays live.
    first.accept()
    gc.collect()
    assert first_marker_ref() is None
    assert first_losing_ref() is None
    assert second_marker_ref() is not None
    assert second_losing_ref() is not None

    third = authority.exchange((second_incoming,))
    third.reserve(c_root, "c")
    third_incoming = third.prepare()["c"]
    third.commit()
    third.accept()
    gc.collect()
    state = authority._snapshot_state()
    assert len(state.terminal_evidence) <= 2
    assert module._has_terminal_marker(state, second_marker_ref())

    second.accept()
    gc.collect()
    assert second_marker_ref() is None
    assert second_losing_ref() is None
    fourth = authority.exchange((third_incoming,))
    fourth.reserve(d_root, "d")
    fourth_incoming = fourth.prepare()["d"]
    fourth.commit()
    fourth.accept()
    gc.collect()
    assert len(authority._snapshot_state().terminal_evidence) == 1

    for lease in (
        fourth_incoming,
        third_incoming,
        second_incoming,
        first_incoming,
        original_lease,
    ):
        lease.release()


@pytest.mark.parametrize("second_direction", ("accept", "rollback"))
def test_later_terminal_prunes_marker_that_died_after_its_prepare(
    monkeypatch, second_direction: str,
) -> None:
    authority = PhysicalRootAuthority(32)
    original = np.arange(4, dtype=np.uint8)
    first_root = np.arange(4, dtype=np.uint8)
    second_root = np.arange(4, dtype=np.uint8)
    original_lease = _retain(authority, ("original", original))["original"]

    first = authority.exchange((original_lease,))
    first.reserve(first_root, "first")
    first_incoming = first.prepare()["first"]
    first.commit()
    first_receipt = _receipt(first)
    first_marker = first_receipt.accepted_marker
    first_marker_ref = weakref.ref(first_marker)
    real_swap = authority._swap_state
    cut = True

    def first_terminal_cut(expected, replacement):
        nonlocal cut
        swapped = real_swap(expected, replacement)
        if cut:
            cut = False
            raise _InjectedCut("first terminal")
        return swapped

    monkeypatch.setattr(authority, "_swap_state", first_terminal_cut)
    with pytest.raises(_InjectedCut, match="first terminal"):
        first.accept()
    monkeypatch.setattr(authority, "_swap_state", real_swap)

    # The second exchange freezes its staged/pending states while the first
    # marker is still live.  Its later terminal transition must not republish
    # that weakref after the first exchange has compacted and released it.
    second = authority.exchange((first_incoming,))
    second.reserve(second_root, "second")
    second_incoming = second.prepare()["second"]
    second.commit()
    second_receipt = _receipt(second)
    second_marker = (
        second_receipt.accepted_marker
        if second_direction == "accept"
        else second_receipt.rolled_back_marker
    )
    assert module._has_terminal_marker(
        authority._snapshot_state(), first_marker,
    )

    assert first.accept()["first"] is first_incoming
    del first_receipt, first_marker
    gc.collect()
    assert first_marker_ref() is None

    if second_direction == "accept":
        assert second.accept()["second"] is second_incoming
    else:
        second.rollback()
    state = authority._snapshot_state()
    assert len(state.terminal_evidence) == 1
    assert state.terminal_evidence[0]() is second_marker
    assert module._has_terminal_marker(state, second_marker)

    second_incoming.release()
    first_incoming.release()
    original_lease.release()


@pytest.mark.parametrize(
    "property_name", ("prepared_leases", "final_projected_bytes"),
)
@pytest.mark.parametrize("source_phase", ("staged", "pending"))
def test_interrupted_rollback_properties_terminalize_without_payload(
    monkeypatch, property_name: str, source_phase: str,
) -> None:
    authority = PhysicalRootAuthority(16)
    old = np.arange(4, dtype=np.uint8)
    incoming_root = np.arange(4, dtype=np.uint8)
    victim = _retain(authority, ("old", old))["old"]
    exchange = authority.exchange((victim,))
    exchange.reserve(incoming_root, "incoming")
    incoming = exchange.prepare()["incoming"]
    if source_phase == "pending":
        exchange.commit()
    real_swap = authority._swap_state
    cut = True

    def rollback_cut(expected, replacement):
        nonlocal cut
        swapped = real_swap(expected, replacement)
        if cut:
            cut = False
            raise _InjectedCut("rollback terminal")
        return swapped

    monkeypatch.setattr(authority, "_swap_state", rollback_cut)
    with pytest.raises(_InjectedCut, match="rollback terminal"):
        exchange.rollback()
    monkeypatch.setattr(authority, "_swap_state", real_swap)

    with pytest.raises(RuntimeError, match="rolled back"):
        getattr(exchange, property_name)
    assert exchange.phase is PhysicalRootExchangePhase.ROLLED_BACK
    assert type(exchange._journal) is module._TerminalRecord
    assert exchange._journal.leases is None
    assert exchange._journal.projected_bytes is None
    incoming.release()
    victim.release()


@pytest.mark.parametrize("winner", ("accept", "rollback"))
def test_terminal_intent_has_one_concurrent_winner(
    monkeypatch, winner: str,
) -> None:
    authority = PhysicalRootAuthority(16)
    old = np.arange(4, dtype=np.uint8)
    new = np.arange(4, dtype=np.uint8)
    victim = _retain(authority, ("old", old))["old"]
    exchange = authority.exchange((victim,))
    exchange.reserve(new, "new")
    incoming = exchange.prepare()["new"]
    exchange.commit()
    receipt = _receipt(exchange)
    entered = threading.Event()
    release = threading.Event()
    real_swap = authority._swap_state

    def held_terminal_swap(expected, replacement):
        if expected is receipt.pending:
            entered.set()
            assert release.wait(1.0)
        return real_swap(expected, replacement)

    monkeypatch.setattr(authority, "_swap_state", held_terminal_swap)
    result: list[object] = []
    errors: list[BaseException] = []

    def run_winner() -> None:
        try:
            result.append(getattr(exchange, winner)())
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=run_winner)
    thread.start()
    assert entered.wait(1.0)
    loser = "rollback" if winner == "accept" else "accept"
    with pytest.raises(RuntimeError, match=(
        "acceptance is pending" if winner == "accept"
        else "rollback is pending"
    )):
        getattr(exchange, loser)()
    release.set()
    thread.join(1.0)
    assert not thread.is_alive()
    assert errors == []
    if winner == "accept":
        assert len(result) == 1
        assert result[0] is exchange.accept()
        with pytest.raises(RuntimeError, match="accepted"):
            exchange.rollback()
    else:
        assert result == [None]
        exchange.rollback()
        with pytest.raises(RuntimeError, match="rolled back"):
            exchange.accept()
    incoming.release()
    victim.release()


@pytest.mark.parametrize("direction", ("accept", "rollback"))
def test_concurrent_same_terminal_direction_fails_fast_then_is_idempotent(
    monkeypatch, direction: str,
) -> None:
    authority = PhysicalRootAuthority(16)
    old = np.arange(4, dtype=np.uint8)
    new = np.arange(4, dtype=np.uint8)
    victim = _retain(authority, ("old", old))["old"]
    exchange = authority.exchange((victim,))
    exchange.reserve(new, "new")
    incoming = exchange.prepare()["new"]
    if direction == "accept":
        exchange.commit()
    receipt = _receipt(exchange)
    held_source = (
        receipt.pending if direction == "accept" else receipt.staged
    )
    entered = threading.Event()
    release = threading.Event()
    real_swap = authority._swap_state

    def held_terminal_swap(expected, replacement):
        if expected is held_source:
            entered.set()
            assert release.wait(1.0)
        return real_swap(expected, replacement)

    monkeypatch.setattr(authority, "_swap_state", held_terminal_swap)
    results: list[object] = []
    errors: list[BaseException] = []

    def terminal() -> None:
        try:
            results.append(getattr(exchange, direction)())
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=terminal)
    thread.start()
    assert entered.wait(1.0)
    with pytest.raises(RuntimeError, match="busy"):
        getattr(exchange, direction)()
    release.set()
    thread.join(1.0)
    assert not thread.is_alive()
    assert errors == []
    if direction == "accept":
        assert results == [exchange.accept()]
    else:
        assert results == [None]
        exchange.rollback()
    incoming.release()
    victim.release()


@pytest.mark.parametrize("competitor", ("prepare", "rollback"))
def test_concurrent_prepare_retains_exact_receipt_and_marker_identities(
    monkeypatch, competitor: str,
) -> None:
    authority = PhysicalRootAuthority(16)
    root = np.arange(4, dtype=np.uint8)
    exchange = authority.exchange(())
    exchange.reserve(root, "root")
    entered = threading.Event()
    release = threading.Event()
    real_swap = authority._swap_state

    def held_prepare_swap(expected, replacement):
        entered.set()
        assert release.wait(1.0)
        return real_swap(expected, replacement)

    monkeypatch.setattr(authority, "_swap_state", held_prepare_swap)
    results: list[object] = []
    errors: list[BaseException] = []

    def run_prepare() -> None:
        try:
            results.append(exchange.prepare())
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=run_prepare)
    thread.start()
    assert entered.wait(1.0)
    journal = _open_journal(exchange)
    receipt = _receipt(exchange)
    leases = receipt.leases
    accepted_marker = receipt.accepted_marker
    rolled_back_marker = receipt.rolled_back_marker
    with pytest.raises(RuntimeError, match="busy"):
        getattr(exchange, competitor)()
    assert exchange._journal is journal
    assert journal.terminal_intent is None
    assert _receipt(exchange) is receipt
    assert receipt.accepted_marker is accepted_marker
    assert receipt.rolled_back_marker is rolled_back_marker
    release.set()
    thread.join(1.0)
    assert not thread.is_alive()
    assert errors == [] and len(results) == 1
    assert results[0] is leases
    assert _receipt(exchange) is receipt
    assert receipt.accepted_marker is accepted_marker
    assert receipt.rolled_back_marker is rolled_back_marker
    if competitor == "prepare":
        assert exchange.prepare() is results[0]
    exchange.rollback()


@pytest.mark.parametrize("transition", ("commit", "accept", "rollback"))
def test_transition_false_before_real_swap_is_retryable(
    monkeypatch, transition: str,
) -> None:
    authority = PhysicalRootAuthority(16)
    old = np.arange(4, dtype=np.uint8)
    new = np.arange(4, dtype=np.uint8)
    victim = _retain(authority, ("old", old))["old"]
    exchange = authority.exchange((victim,))
    exchange.reserve(new, "new")
    incoming = exchange.prepare()["new"]
    if transition == "accept":
        exchange.commit()
    real_swap = authority._swap_state
    first = True

    def false_before_swap(expected, replacement):
        nonlocal first
        if first:
            first = False
            return False
        return real_swap(expected, replacement)

    monkeypatch.setattr(authority, "_swap_state", false_before_swap)
    with pytest.raises(RuntimeError, match="did not complete"):
        getattr(exchange, transition)()
    getattr(exchange, transition)()
    if transition == "accept":
        assert exchange.phase is PhysicalRootExchangePhase.ACCEPTED
    else:
        exchange.rollback()
        assert exchange.phase is PhysicalRootExchangePhase.ROLLED_BACK
    incoming.release()
    victim.release()


@pytest.mark.parametrize("terminal", ("accept", "rollback"))
@pytest.mark.parametrize("mutation", ("release", "close", "reservation"))
def test_terminal_recovery_survives_legacy_mutation_inside_fault_cut(
    monkeypatch, terminal: str, mutation: str,
) -> None:
    authority = PhysicalRootAuthority(24)
    old = np.arange(4, dtype=np.uint8)
    survivor_root = np.arange(4, dtype=np.uint8)
    new = np.arange(4, dtype=np.uint8)
    leases = _retain(
        authority, ("old", old), ("survivor", survivor_root),
    )
    exchange = authority.exchange((leases["old"],))
    exchange.reserve(new, "new")
    incoming = exchange.prepare()["new"]
    if terminal == "accept":
        exchange.commit()
    real_swap = authority._swap_state
    later_leases: list[PhysicalRootLease] = []

    def swap_mutate_interrupt(expected, replacement):
        assert real_swap(expected, replacement)
        if mutation == "release":
            leases["survivor"].release()
        elif mutation == "close":
            authority.close()
        else:
            reservation = authority.reserve()
            reservation.reserve(
                np.arange(4, dtype=np.uint8), "later-legacy",
            )
            later_leases.extend(reservation.commit().values())
        raise _InjectedCut("terminal mutation cut")

    monkeypatch.setattr(authority, "_swap_state", swap_mutate_interrupt)
    with pytest.raises(_InjectedCut, match="terminal mutation cut"):
        getattr(exchange, terminal)()
    monkeypatch.setattr(authority, "_swap_state", real_swap)
    if terminal == "accept":
        assert exchange.accept()["new"] is incoming
    else:
        exchange.rollback()
    incoming.release()
    leases["old"].release()
    leases["survivor"].release()
    for lease in later_leases:
        lease.release()


class _CallbackSemantic:
    def __init__(self, callback=None, *, equal: bool = False) -> None:
        self.callback = callback
        self.equal = equal

    def __hash__(self) -> int:
        if self.callback is not None:
            self.callback("hash")
        return 17

    def __eq__(self, other) -> bool:
        if self.callback is not None:
            self.callback("eq")
        return self.equal


def _all_exchange_entries(exchange, root) -> tuple[tuple[str, object], ...]:
    return (
        ("phase", lambda: exchange.phase),
        ("claim", lambda: exchange.claim(0)),
        ("bind", lambda: exchange.bind(object(), root, "bind")),
        ("reserve", lambda: exchange.reserve(root, "reserve")),
        ("projected", lambda: exchange.projected_bytes),
        ("prepare", exchange.prepare),
        ("commit", exchange.commit),
        ("accept", exchange.accept),
        ("rollback", exchange.rollback),
        ("leases", lambda: exchange.prepared_leases),
        ("final", lambda: exchange.final_projected_bytes),
    )


def test_semantic_hash_and_equality_reentry_leave_exact_journal() -> None:
    authority = PhysicalRootAuthority(32)
    exchange = authority.exchange(())
    root = np.arange(4, dtype=np.uint8)
    observed: list[tuple[str, str, str]] = []
    callback_state: list[tuple[str, bool, bool, bool]] = []

    def reenter(kind: str) -> None:
        callback_state.append((
            kind,
            authority._lock._is_owned(), exchange._lock._is_owned(),
            getattr(module._EXCHANGE_TLS, "token", None) is not None,
        ))
        for name, entry in _all_exchange_entries(exchange, root):
            try:
                entry()
            except RuntimeError as error:
                observed.append((kind, name, str(error)))
            else:  # pragma: no cover - exact adversarial failure signal
                raise AssertionError(f"nested {name} unexpectedly succeeded")

    semantic = _CallbackSemantic(reenter)
    initial = _open_journal(exchange)
    exchange.reserve(root, semantic)
    after_hash = _open_journal(exchange)
    assert after_hash is not initial
    assert after_hash.generation == initial.generation + 1
    assert {name for _kind, name, _message in observed} == {
        name for name, _entry in _all_exchange_entries(exchange, root)
    }
    assert all(
        "callback reentry" in message
        for _kind, _name, message in observed
    )
    assert callback_state
    assert {state[1:] for state in callback_state} == {
        (False, False, True),
    }
    assert getattr(module._EXCHANGE_TLS, "token", None) is None

    class BindSemantic:
        def __hash__(self) -> int:
            reenter("bind-hash")
            return 29

        def __eq__(self, _other: object) -> bool:
            reenter("bind-eq")
            return False

    observed.clear()
    claim = exchange.claim(root.nbytes)
    exchange.bind(claim, root.copy(), BindSemantic())
    assert observed
    assert {state[1:] for state in callback_state} == {
        (False, False, True),
    }
    assert getattr(module._EXCHANGE_TLS, "token", None) is None

    observed.clear()
    semantic.callback = reenter
    semantic.equal = True
    collision = _CallbackSemantic(equal=False)
    before_collision = exchange._journal
    with pytest.raises(ValueError, match="duplicated"):
        exchange.reserve(root.copy(), collision)
    assert exchange._journal is before_collision
    assert observed
    equality_states = [state for state in callback_state if state[0] == "eq"]
    assert equality_states == [("eq", False, False, True)]
    equality_failures = [
        (name, message)
        for kind, name, message in observed
        if kind == "eq"
    ]
    assert {name for name, _message in equality_failures} == {
        name for name, _entry in _all_exchange_entries(exchange, root)
    }
    assert all(
        "callback reentry" in message
        for _name, message in equality_failures
    )
    assert getattr(module._EXCHANGE_TLS, "token", None) is None

    def interrupt(_kind: str) -> None:
        raise _InjectedCut("hash callback")

    before_cut = exchange._journal
    with pytest.raises(_InjectedCut, match="hash callback"):
        exchange.reserve(root.copy(), _CallbackSemantic(interrupt))
    assert exchange._journal is before_cut
    assert exchange.phase is PhysicalRootExchangePhase.OPEN
    assert getattr(module._EXCHANGE_TLS, "token", None) is None
    exchange.rollback()


@pytest.mark.parametrize(
    "entry_name",
    (
        "phase", "claim", "bind", "reserve", "projected", "prepare",
        "commit", "accept", "rollback", "leases", "final",
    ),
)
def test_callback_blocks_every_other_thread_entry_promptly(
    entry_name: str,
) -> None:
    authority = PhysicalRootAuthority(32)
    exchange = authority.exchange(())
    root = np.arange(4, dtype=np.uint8)
    entered = threading.Event()
    release = threading.Event()
    once = False
    callback_locks: list[tuple[bool, bool]] = []

    def held_hash(_kind: str) -> None:
        nonlocal once
        if once:
            return
        once = True
        callback_locks.append((
            authority._lock._is_owned(), exchange._lock._is_owned(),
        ))
        entered.set()
        assert release.wait(1.0)

    outer_errors: list[BaseException] = []

    def outer() -> None:
        try:
            exchange.reserve(root, _CallbackSemantic(held_hash))
        except BaseException as error:
            outer_errors.append(error)

    outer_thread = threading.Thread(target=outer)
    outer_thread.start()
    assert entered.wait(1.0)
    results: list[object] = []

    entries = dict(_all_exchange_entries(exchange, root))

    def call() -> None:
        try:
            results.append(entries[entry_name]())
        except BaseException as error:
            results.append(error)

    reader = threading.Thread(target=call)
    reader.start()
    reader.join(1.0)
    assert not reader.is_alive()
    assert len(results) == 1
    assert isinstance(results[0], RuntimeError)
    assert "busy" in str(results[0])
    release.set()
    outer_thread.join(1.0)
    assert not outer_thread.is_alive()
    assert outer_errors == []
    assert callback_locks == [(False, False)]
    exchange.rollback()


def test_two_exchange_cross_callbacks_fail_before_second_lock_without_deadlock(
) -> None:
    first = PhysicalRootAuthority(16).exchange(())
    second = PhysicalRootAuthority(16).exchange(())
    barrier = threading.Barrier(2)
    results: list[tuple[str, object]] = []

    def semantic_for(name: str, other):
        fired = False

        def callback(_kind: str) -> None:
            nonlocal fired
            if fired:
                return
            fired = True
            barrier.wait(timeout=1.0)
            try:
                _ = other.phase
            except BaseException as error:
                results.append((name, error))
            else:  # pragma: no cover - exact adversarial failure signal
                results.append((name, None))

        return _CallbackSemantic(callback)

    errors: list[BaseException] = []

    def run(exchange, semantic) -> None:
        try:
            exchange.reserve(np.arange(4, dtype=np.uint8), semantic)
        except BaseException as error:
            errors.append(error)

    threads = (
        threading.Thread(target=run, args=(first, semantic_for("first", second))),
        threading.Thread(target=run, args=(second, semantic_for("second", first))),
    )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(1.0)
        assert not thread.is_alive()
    assert errors == []
    assert {name for name, _error in results} == {"first", "second"}
    assert all(
        isinstance(error, RuntimeError) and "callback reentry" in str(error)
        for _name, error in results
    )
    first.rollback()
    second.rollback()


def test_cross_exchange_callback_rejects_mutator_before_second_lock() -> None:
    first = PhysicalRootAuthority(16).exchange(())
    second = PhysicalRootAuthority(16).exchange(())
    second_journal = second._journal
    observed: list[tuple[bool, bool, bool, BaseException]] = []

    def reenter(_kind: str) -> None:
        try:
            second.claim(0)
        except BaseException as error:
            observed.append((
                first._lock._is_owned(),
                second._lock._is_owned(),
                getattr(module._EXCHANGE_TLS, "token", None) is not None,
                error,
            ))

    first.reserve(
        np.arange(4, dtype=np.uint8), _CallbackSemantic(reenter),
    )
    assert observed
    for first_locked, second_locked, tls_active, error in observed:
        assert (first_locked, second_locked, tls_active) == (
            False, False, True,
        )
        assert isinstance(error, RuntimeError)
        assert "callback reentry" in str(error)
    assert second._journal is second_journal
    assert getattr(module._EXCHANGE_TLS, "token", None) is None
    first.rollback()
    second.rollback()


def test_callback_authority_drift_refuses_before_journal_publication() -> None:
    authority = PhysicalRootAuthority(16)
    old = np.arange(4, dtype=np.uint8)
    victim = _retain(authority, ("old", old))["old"]
    exchange = authority.exchange((victim,))
    initial = exchange._journal
    fired = False

    def drift(_kind: str) -> None:
        nonlocal fired
        if not fired:
            fired = True
            victim.release()

    with pytest.raises(RuntimeError, match="base state drifted"):
        exchange.reserve(
            np.arange(4, dtype=np.uint8), _CallbackSemantic(drift),
        )
    assert exchange._journal is initial
    assert _open_journal(exchange).receipt is None
    assert tuple(_open_journal(exchange).roots) == ()
    exchange.rollback()
