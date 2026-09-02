from __future__ import annotations

import copy
import gc
import pickle
import threading
import weakref

import numpy as np
import pytest

import xrd_tools.io.browse_1d_cache as module
from xrd_tools.core.physical_memory import (
    PhysicalRootAuthority,
    PhysicalRootExchange,
    PhysicalRootExchangePhase,
)
from xrd_tools.io.browse_1d_cache import (
    Browse1DCache,
    Browse1DCachePhase,
    Browse1DRowKey,
    default_browse_1d_cache_budget,
)


class _InjectedCut(BaseException):
    pass


def _readonly(values, *, dtype=np.float64) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    array.flags.writeable = False
    return array


def _operation(
    cache: Browse1DCache,
    frame: int,
    label: int,
    *rows: tuple[str, np.ndarray],
):
    return cache.begin_store(frame, label, tuple(rows))


def _prepare_only(cache: Browse1DCache, operation) -> None:
    state = cache._snapshot_state()
    assert state.phase is Browse1DCachePhase.PREPARING
    cache._drive_prepare(state, operation)
    assert cache.phase is Browse1DCachePhase.STAGED_PENDING


def _commit_only(cache: Browse1DCache, operation) -> None:
    state = cache._snapshot_state()
    assert state.phase is Browse1DCachePhase.STAGED_PENDING
    cache._drive_commit(state, operation)
    assert cache.phase is Browse1DCachePhase.ACCEPT_PENDING


def _cache_transition_kind(expected, replacement) -> str | None:
    pair = (expected.phase, replacement.phase)
    if replacement.phase is Browse1DCachePhase.BLOCKED:
        return "blocked"
    if pair == (Browse1DCachePhase.OPEN, Browse1DCachePhase.PREPARING):
        if replacement.journal is not None and replacement.journal.closing:
            return "close-install"
        return "install"
    if pair == (
        Browse1DCachePhase.PREPARING,
        Browse1DCachePhase.PREPARING,
    ) and replacement.driver is not None:
        return "prepare-claim"
    if pair == (
        Browse1DCachePhase.PREPARING,
        Browse1DCachePhase.STAGED_PENDING,
    ):
        return "prepare-settle"
    if pair == (
        Browse1DCachePhase.PREPARING,
        Browse1DCachePhase.ROLLBACK_PENDING,
    ):
        return "prepare-rollback-request"
    if pair == (
        Browse1DCachePhase.PREPARING,
        Browse1DCachePhase.ROLLBACK_READY,
    ):
        return "prepare-failure-settle"
    if pair == (
        Browse1DCachePhase.ROLLBACK_PENDING,
        Browse1DCachePhase.ROLLBACK_READY,
    ):
        return "prepare-rollback-ready"
    if pair == (
        Browse1DCachePhase.STAGED_PENDING,
        Browse1DCachePhase.COMMITTING,
    ):
        return "commit-claim"
    if pair == (
        Browse1DCachePhase.COMMITTING,
        Browse1DCachePhase.ACCEPT_PENDING,
    ):
        return "commit-settle"
    if pair == (
        Browse1DCachePhase.COMMITTING,
        Browse1DCachePhase.STAGED_PENDING,
    ):
        return "commit-failure-settle"
    if pair == (
        Browse1DCachePhase.ACCEPT_PENDING,
        Browse1DCachePhase.ACCEPTING,
    ):
        return "accept-claim"
    if pair == (
        Browse1DCachePhase.ACCEPTING,
        Browse1DCachePhase.OPEN,
    ):
        return "accept-terminal"
    if pair == (
        Browse1DCachePhase.ACCEPTING,
        Browse1DCachePhase.ACCEPT_PENDING,
    ):
        return "accept-failure-settle"
    if pair == (
        Browse1DCachePhase.STAGED_PENDING,
        Browse1DCachePhase.ROLLBACK_READY,
    ):
        return "rollback-request"
    if pair == (
        Browse1DCachePhase.ROLLBACK_READY,
        Browse1DCachePhase.ROLLING_BACK,
    ):
        return "rollback-claim"
    if pair == (
        Browse1DCachePhase.ROLLING_BACK,
        Browse1DCachePhase.OPEN,
    ):
        return "rollback-terminal"
    if pair == (
        Browse1DCachePhase.ROLLING_BACK,
        Browse1DCachePhase.ROLLBACK_READY,
    ):
        return "rollback-failure-settle"
    if pair == (
        Browse1DCachePhase.ACCEPTING,
        Browse1DCachePhase.CLOSING_AUTHORITY,
    ):
        return "close-exchange-settle"
    if pair == (
        Browse1DCachePhase.CLOSING_AUTHORITY,
        Browse1DCachePhase.CLOSING_AUTHORITY,
    ) and replacement.driver is not None:
        return "close-claim"
    if pair == (
        Browse1DCachePhase.CLOSING_AUTHORITY,
        Browse1DCachePhase.CLOSING_AUTHORITY,
    ) and expected.driver is not None and replacement.driver is None:
        return "close-driver-clear"
    if pair == (
        Browse1DCachePhase.CLOSING_AUTHORITY,
        Browse1DCachePhase.CLOSED,
    ):
        return "close-terminal"
    if pair == (Browse1DCachePhase.OPEN, Browse1DCachePhase.OPEN):
        if len(replacement.borrows) == len(expected.borrows) + 1:
            return "borrow-install"
        if len(replacement.borrows) + 1 == len(expected.borrows):
            return "borrow-release"
    return None


def _install_cache_cas_cut(
    cache: Browse1DCache,
    monkeypatch,
    transition: str,
    cut: str,
) -> None:
    real_cas = cache._cas_state
    fired = False

    def cut_cas(expected, replacement):
        nonlocal fired
        if not fired and _cache_transition_kind(expected, replacement) == transition:
            fired = True
            if cut == "after":
                assert real_cas(expected, replacement)
            raise _InjectedCut(f"{transition}-{cut}")
        return real_cas(expected, replacement)

    monkeypatch.setattr(cache, "_cas_state", cut_cas)


def test_default_budget_is_five_percent_capped_with_gib_fallback(
    monkeypatch,
) -> None:
    gib = 1 << 30
    assert default_browse_1d_cache_budget(10 * gib) == gib // 2
    assert default_browse_1d_cache_budget(100 * gib) == gib
    assert default_browse_1d_cache_budget(True) == gib
    monkeypatch.setattr(module, "_detect_physical_ram_bytes", lambda: None)
    assert default_browse_1d_cache_budget() == gib


def test_cache_operation_and_borrow_refuse_copy_deepcopy_and_pickle() -> None:
    cache = Browse1DCache(64)
    operation = _operation(cache, 1, 1, ("i", _readonly(np.arange(4))))
    assert operation.run() == "accepted"
    borrowed = cache.borrow(1, 1, "i")
    try:
        for owner in (cache, operation, borrowed):
            with pytest.raises(TypeError, match="cannot be copied"):
                copy.copy(owner)
            with pytest.raises(TypeError, match="cannot be copied"):
                copy.deepcopy(owner)
            with pytest.raises(TypeError, match="cannot be pickled"):
                pickle.dumps(owner)
            with pytest.raises(TypeError, match="cannot be pickled"):
                owner.__reduce__()
            with pytest.raises(TypeError, match="cannot be pickled"):
                owner.__reduce_ex__(5)
    finally:
        borrowed.release()


def test_borrow_key_and_array_are_getter_only_and_pin_exact_row() -> None:
    first = _readonly(np.arange(4))
    second = _readonly(np.arange(4, 8))
    cache = Browse1DCache(first.nbytes)
    _operation(cache, 1, 1, ("i", first)).run()
    borrowed = cache.borrow(1, 1, "i")
    exact_key = Browse1DRowKey(1, 1, "i")
    assert borrowed.key == exact_key
    assert borrowed.array is first
    with pytest.raises(AttributeError):
        borrowed.key = Browse1DRowKey(2, 2, "i")
    with pytest.raises(AttributeError):
        borrowed.array = second
    with pytest.raises(RuntimeError, match="borrowed"):
        _operation(cache, 1, 1, ("i", second))
    assert borrowed.key == exact_key
    assert borrowed.array is first
    borrowed.release()
    _operation(cache, 1, 1, ("i", second)).run()
    replacement = cache.borrow(1, 1, "i")
    assert replacement.array is second
    replacement.release()


def test_whole_readonly_rows_share_one_root_and_catalog_survives_eviction(
) -> None:
    base = _readonly(np.arange(8), dtype=np.uint8)
    first = base[:4]
    second = base[4:]
    incoming = _readonly(np.arange(8, 16), dtype=np.uint8)
    cache = Browse1DCache(8)
    cache.record_scalars(1, 7, (("temperature", 300.0),))

    assert _operation(
        cache, 1, 7, ("intensity", first), ("sigma", second),
    ).run() == "accepted"
    assert cache.resident_bytes == 8
    assert cache.resident_root_count == 1
    assert cache.resident_keys == (
        Browse1DRowKey(1, 7, "intensity"),
        Browse1DRowKey(1, 7, "sigma"),
    )

    borrowed = cache.borrow(1, 7, "intensity")
    assert borrowed.array is first
    assert borrowed.array.flags.writeable is False
    with pytest.raises(ValueError, match="exceed the cache budget"):
        _operation(cache, 2, 8, ("intensity", incoming))
    assert cache.resident_keys[0] == borrowed.key

    borrowed.release()
    assert _operation(cache, 2, 8, ("intensity", incoming)).run() == "accepted"
    assert cache.resident_keys == (Browse1DRowKey(2, 8, "intensity"),)
    assert dict(cache.scalars(1, 7)) == {"temperature": 300.0}


def test_resident_row_limit_uses_lru_and_counts_shared_roots(
    monkeypatch,
) -> None:
    monkeypatch.setattr(module, "_MAX_BROWSE_1D_RESIDENT_ROWS", 2)
    shared = _readonly(np.arange(8), dtype=np.uint8)
    cache = Browse1DCache(1 << 20)
    _operation(
        cache,
        1,
        1,
        ("first", shared[:4]),
        ("second", shared[4:]),
    ).run()
    assert cache.resident_root_count == 1
    recent = cache.borrow(1, 1, "first")
    recent.release()

    _operation(
        cache, 2, 2, ("third", _readonly([8], dtype=np.uint8)),
    ).run()

    assert cache.resident_keys == (
        Browse1DRowKey(1, 1, "first"),
        Browse1DRowKey(2, 2, "third"),
    )


def test_resident_row_limit_refuses_before_journal_when_rows_are_pinned(
    monkeypatch,
) -> None:
    monkeypatch.setattr(module, "_MAX_BROWSE_1D_RESIDENT_ROWS", 2)
    cache = Browse1DCache(1 << 20)
    _operation(
        cache,
        1,
        1,
        ("borrowed", _readonly([1], dtype=np.uint8)),
        ("protected", _readonly([2], dtype=np.uint8)),
    ).run()
    borrowed = cache.borrow(1, 1, "borrowed")
    before = cache._snapshot_state()
    roots = cache._authority.retained_roots
    try:
        with pytest.raises(ValueError, match="resident row limit"):
            cache.begin_store(
                2,
                2,
                (("incoming", _readonly([3], dtype=np.uint8)),),
                _expected_state=before,
                _protected_rows=(before.rows[1],),
            )
        assert cache._snapshot_state() is before
        assert cache._authority.retained_roots == roots
        assert cache.phase is Browse1DCachePhase.OPEN
    finally:
        borrowed.release()


def test_resident_row_limit_refuses_oversized_batch_atomically(
    monkeypatch,
) -> None:
    monkeypatch.setattr(module, "_MAX_BROWSE_1D_RESIDENT_ROWS", 2)
    cache = Browse1DCache(1 << 20)
    before = cache._snapshot_state()
    with pytest.raises(ValueError, match="resident row limit"):
        _operation(
            cache,
            1,
            1,
            ("first", _readonly([1], dtype=np.uint8)),
            ("second", _readonly([2], dtype=np.uint8)),
            ("third", _readonly([3], dtype=np.uint8)),
        )
    assert cache._snapshot_state() is before
    assert cache.resident_keys == ()


def test_resident_row_limit_replacement_does_not_double_count(
    monkeypatch,
) -> None:
    monkeypatch.setattr(module, "_MAX_BROWSE_1D_RESIDENT_ROWS", 1)
    cache = Browse1DCache(1 << 20)
    _operation(
        cache, 1, 1, ("row", _readonly([1], dtype=np.uint8)),
    ).run()
    replacement = _readonly([2], dtype=np.uint8)
    assert _operation(cache, 1, 1, ("row", replacement)).run() == "accepted"
    assert cache.resident_keys == (Browse1DRowKey(1, 1, "row"),)
    with cache.borrow(1, 1, "row") as borrowed:
        assert borrowed.array is replacement


def test_preparing_snapshot_retains_exact_prior_rows_until_accept() -> None:
    first = _readonly(np.arange(4), dtype=np.uint8)
    second = _readonly(np.arange(4, 8), dtype=np.uint8)
    cache = Browse1DCache(4)
    _operation(cache, 1, 1, ("i", first)).run()
    prior = cache._snapshot_state().rows
    operation = _operation(cache, 2, 2, ("i", second))
    preparing = cache._snapshot_state()
    assert preparing.phase is Browse1DCachePhase.PREPARING
    assert preparing.rows is prior
    assert preparing.journal is not None
    assert preparing.journal.prior_rows is prior
    assert preparing.journal.survivor_rows == ()
    assert operation.run() == "accepted"
    assert cache.borrow(2, 2, "i").array is second


def test_rollback_restores_exact_prior_rows_and_root_authority() -> None:
    prior_array = _readonly(np.arange(4), dtype=np.uint8)
    incoming = _readonly(np.arange(4, 8), dtype=np.uint8)
    cache = Browse1DCache(4)
    _operation(cache, 1, 1, ("i", prior_array)).run()
    before = cache._snapshot_state()
    prior_rows = before.rows
    prior_row = prior_rows[0]
    prior_roots = cache._authority.retained_roots
    assert len(prior_roots) == 1
    assert prior_roots[0] is prior_array
    assert cache._authority.semantic_references == 1

    operation = _operation(cache, 1, 1, ("i", incoming))
    _prepare_only(cache, operation)
    assert operation.rollback() == "rolled-back"
    after = cache._snapshot_state()
    assert after.rows is prior_rows
    assert after.rows[0] is prior_row
    assert after.rows[0].array is prior_array
    restored_roots = cache._authority.retained_roots
    assert len(restored_roots) == 1
    assert restored_roots[0] is prior_roots[0]
    assert cache._authority.retained_bytes == prior_array.nbytes
    assert cache._authority.retained_root_count == 1
    assert cache._authority.semantic_references == 1


def test_accept_retains_exact_survivor_and_replaces_only_victim() -> None:
    survivor = _readonly(np.arange(4), dtype=np.uint8)
    victim = _readonly(np.arange(4, 8), dtype=np.uint8)
    incoming = _readonly(np.arange(8, 12), dtype=np.uint8)
    cache = Browse1DCache(8)
    _operation(
        cache,
        1,
        1,
        ("survivor", survivor),
        ("victim", victim),
    ).run()
    before = cache._snapshot_state()
    survivor_row, victim_row = before.rows
    assert survivor_row.array is survivor
    assert victim_row.array is victim

    assert _operation(cache, 1, 1, ("victim", incoming)).run() == "accepted"
    after = cache._snapshot_state()
    assert len(after.rows) == 2
    assert after.rows[0] is survivor_row
    assert after.rows[0].lease is survivor_row.lease
    assert after.rows[0].array is survivor
    assert after.rows[1] is not victim_row
    assert after.rows[1].array is incoming
    roots = cache._authority.retained_roots
    assert len(roots) == 2
    assert any(root is survivor for root in roots)
    assert any(root is incoming for root in roots)
    assert not any(root is victim for root in roots)
    assert cache.resident_bytes == survivor.nbytes + incoming.nbytes
    assert cache.resident_root_count == 2
    assert cache._authority.semantic_references == 2


def test_row_validation_refuses_mutable_wide_and_record_fallbacks() -> None:
    cache = Browse1DCache(128)
    with pytest.raises(ValueError, match="read-only"):
        _operation(cache, 1, 1, ("mutable", np.arange(4)))
    wide = np.empty((2, 4), dtype=np.int64)
    wide[...] = np.arange(8).reshape(2, 4)
    assert wide.base is None
    wide.flags.writeable = False
    with pytest.raises(ValueError, match="numeric and 1-D"):
        _operation(cache, 1, 1, ("wide", wide))
    row = wide[0]
    assert row.ndim == 1 and not row.flags.writeable
    with pytest.raises(ValueError, match="wide root"):
        _operation(cache, 1, 1, ("record-row", row))


@pytest.mark.parametrize(
    "dtype",
    (
        np.dtype([("value", "f8")]),
        np.dtype("S4"),
        np.dtype("U4"),
        np.dtype("datetime64[ns]"),
        np.dtype(bool),
    ),
)
def test_row_validation_refuses_every_nonnumeric_fallback_dtype(dtype) -> None:
    row = np.zeros(4, dtype=dtype)
    row.flags.writeable = False
    cache = Browse1DCache(256)
    with pytest.raises(ValueError, match="numeric and 1-D"):
        _operation(cache, 1, 1, ("row", row))
    assert cache.phase is Browse1DCachePhase.OPEN
    assert cache.resident_bytes == 0


def test_readonly_leaf_refuses_writable_physical_root() -> None:
    root = np.arange(8, dtype=np.float64)
    row = root[:4]
    row.flags.writeable = False
    assert root.flags.writeable and not row.flags.writeable
    cache = Browse1DCache(root.nbytes)
    with pytest.raises(ValueError, match="physical root must be read-only"):
        _operation(cache, 1, 1, ("row", row))
    root[0] = 99.0
    assert cache.resident_bytes == 0


@pytest.mark.parametrize("cut", ("before", "after"))
def test_borrow_admission_cas_cut_never_orphans_non_evictable_binding(
    monkeypatch, cut: str,
) -> None:
    row = _readonly(np.arange(4))
    cache = Browse1DCache(row.nbytes)
    _operation(cache, 1, 1, ("i", row)).run()
    _install_cache_cas_cut(cache, monkeypatch, "borrow-install", cut)
    if cut == "before":
        with pytest.raises(_InjectedCut, match="borrow-install-before"):
            cache.borrow(1, 1, "i")
        assert cache.outstanding_borrows == 0
    else:
        borrowed = cache.borrow(1, 1, "i")
        assert borrowed.array is row
        assert cache.outstanding_borrows == 1
        monkeypatch.undo()
        borrowed.release()
    monkeypatch.undo()
    assert cache.outstanding_borrows == 0


@pytest.mark.parametrize("cut", ("before", "after"))
def test_borrow_release_cas_cut_is_exactly_retryable(
    monkeypatch, cut: str,
) -> None:
    row = _readonly(np.arange(4))
    cache = Browse1DCache(row.nbytes)
    _operation(cache, 1, 1, ("i", row)).run()
    borrowed = cache.borrow(1, 1, "i")
    _install_cache_cas_cut(cache, monkeypatch, "borrow-release", cut)
    if cut == "before":
        with pytest.raises(_InjectedCut, match="borrow-release-before"):
            borrowed.release()
        assert not borrowed.released
        assert cache.outstanding_borrows == 1
        monkeypatch.undo()
        borrowed.release()
    else:
        borrowed.release()
    monkeypatch.undo()
    assert borrowed.released
    assert cache.outstanding_borrows == 0


@pytest.mark.parametrize("action", ("install", "release"))
def test_borrow_cas_postswap_effect_survives_later_open_cow(
    monkeypatch, action: str,
) -> None:
    row = _readonly(np.arange(4))
    cache = Browse1DCache(row.nbytes)
    _operation(cache, 1, 1, ("i", row)).run()
    borrowed = None if action == "install" else cache.borrow(1, 1, "i")
    transition = f"borrow-{action}"
    real_cas = cache._cas_state
    fired = False

    def swap_advance_then_cut(expected, replacement):
        nonlocal fired
        if not fired and _cache_transition_kind(expected, replacement) == transition:
            fired = True
            assert real_cas(expected, replacement)
            advanced = module._state_with(
                replacement,
                catalog=replacement.catalog
                + (module._CatalogRecord(9, 9, (("status", "newer"),)),),
            )
            assert real_cas(replacement, advanced)
            raise _InjectedCut(f"{transition}-after-newer-cow")
        return real_cas(expected, replacement)

    monkeypatch.setattr(cache, "_cas_state", swap_advance_then_cut)
    if action == "install":
        borrowed = cache.borrow(1, 1, "i")
        assert borrowed.array is row
        assert cache.outstanding_borrows == 1
    else:
        assert borrowed is not None
        borrowed.release()
        assert borrowed.released
        assert cache.outstanding_borrows == 0
    assert dict(cache.scalars(9, 9)) == {"status": "newer"}
    monkeypatch.undo()
    if borrowed is not None and not borrowed.released:
        borrowed.release()


@pytest.mark.parametrize("action", ("install", "release"))
def test_borrow_cas_postswap_effect_survives_later_preparing_state(
    monkeypatch, action: str,
) -> None:
    first = _readonly(np.arange(4))
    second = _readonly(np.arange(4, 8))
    cache = Browse1DCache(first.nbytes + second.nbytes)
    _operation(cache, 1, 1, ("i", first)).run()
    borrowed = None if action == "install" else cache.borrow(1, 1, "i")
    transition = f"borrow-{action}"
    real_cas = cache._cas_state
    followups = []
    fired = False

    def swap_prepare_then_cut(expected, replacement):
        nonlocal fired
        if not fired and _cache_transition_kind(expected, replacement) == transition:
            fired = True
            assert real_cas(expected, replacement)
            followups.append(_operation(cache, 2, 2, ("j", second)))
            assert cache.phase is Browse1DCachePhase.PREPARING
            raise _InjectedCut(f"{transition}-after-preparing")
        return real_cas(expected, replacement)

    monkeypatch.setattr(cache, "_cas_state", swap_prepare_then_cut)
    if action == "install":
        borrowed = cache.borrow(1, 1, "i")
        assert borrowed.array is first
        state = cache._snapshot_state()
        assert len(state.borrows) == 1
    else:
        assert borrowed is not None
        borrowed.release()
        assert borrowed.released
        assert cache._snapshot_state().borrows == ()
    monkeypatch.undo()
    assert len(followups) == 1
    assert followups[0].rollback() == "rolled-back"
    if borrowed is not None and not borrowed.released:
        borrowed.release()


def test_borrow_release_postswap_effect_survives_later_close(monkeypatch) -> None:
    row = _readonly(np.arange(4))
    cache = Browse1DCache(row.nbytes)
    _operation(cache, 1, 1, ("i", row)).run()
    borrowed = cache.borrow(1, 1, "i")
    real_cas = cache._cas_state
    fired = False

    def swap_close_then_cut(expected, replacement):
        nonlocal fired
        if (
            not fired
            and _cache_transition_kind(expected, replacement)
            == "borrow-release"
        ):
            fired = True
            assert real_cas(expected, replacement)
            cache.close()
            assert cache.phase is Browse1DCachePhase.CLOSED
            raise _InjectedCut("borrow-release-after-close")
        return real_cas(expected, replacement)

    monkeypatch.setattr(cache, "_cas_state", swap_close_then_cut)
    borrowed.release()
    assert borrowed.released
    assert cache.phase is Browse1DCachePhase.CLOSED


def test_borrow_token_and_row_identity_are_exact_and_non_evictable() -> None:
    row = _readonly(np.arange(4), dtype=np.uint8)
    replacement = _readonly(np.arange(4, 8), dtype=np.uint8)
    cache = Browse1DCache(4)
    _operation(cache, 1, 1, ("i", row)).run()
    borrowed = cache.borrow(1, 1, "i")
    state = cache._snapshot_state()
    binding = state.borrows[0]
    assert borrowed.array is state.rows[0].array

    forged = module._BorrowToken(binding.row_identity)
    with pytest.raises(ValueError, match="invalid"):
        cache._release_borrow(forged, binding.row_identity)
    with pytest.raises(RuntimeError, match="borrowed"):
        _operation(cache, 1, 1, ("i", replacement))
    borrowed.release()
    _operation(cache, 1, 1, ("i", replacement)).run()
    assert cache.borrow(1, 1, "i").array is replacement


def test_prepare_active_recovery_only_records_rollback_intent(
    monkeypatch,
) -> None:
    cache = Browse1DCache(32)
    operation = _operation(cache, 1, 1, ("i", _readonly(np.arange(4))))
    journal = cache._snapshot_state().journal
    assert journal is not None
    exchange = journal.exchange
    entered = threading.Event()
    release = threading.Event()
    real_prepare = PhysicalRootExchange.prepare
    real_rollback = PhysicalRootExchange.rollback
    rollback_calls = 0

    def held_prepare(self):
        if self is exchange:
            assert not cache._lock.locked()
            entered.set()
            assert release.wait(1.0)
        return real_prepare(self)

    def counted_rollback(self):
        nonlocal rollback_calls
        if self is exchange:
            rollback_calls += 1
            assert release.is_set()
            assert not cache._lock.locked()
        return real_rollback(self)

    monkeypatch.setattr(PhysicalRootExchange, "prepare", held_prepare)
    monkeypatch.setattr(PhysicalRootExchange, "rollback", counted_rollback)
    results: list[str] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            results.append(operation.run())
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    assert entered.wait(1.0)
    active_state = cache._snapshot_state()
    driver = active_state.driver
    assert driver is not None
    owner = driver.owner()
    assert owner is not None
    assert isinstance(driver.owner, weakref.ReferenceType)
    with pytest.raises(AttributeError):
        driver.owner = weakref.ref(owner)
    owner_drops: list[bool] = []
    owner_ref = weakref.ref(
        owner, lambda _value: owner_drops.append(cache._lock.locked()),
    )
    del owner
    with pytest.raises(RuntimeError, match="busy"):
        operation.run()
    assert cache.recover() == Browse1DCachePhase.ROLLBACK_PENDING.value
    assert cache.phase is Browse1DCachePhase.ROLLBACK_PENDING
    assert rollback_calls == 0
    release.set()
    thread.join(1.0)
    assert not thread.is_alive()
    assert errors == []
    assert results == ["rolled-back"]
    assert rollback_calls == 1
    assert owner_ref() is None
    assert owner_drops == [False]
    assert cache.phase is Browse1DCachePhase.OPEN
    assert cache.resident_keys == ()


def test_terminal_marker_direction_is_shared_by_independent_helpers() -> None:
    cache = Browse1DCache(64)
    rolled = _operation(cache, 1, 1, ("i", _readonly(np.arange(4))))
    assert cache.recover() == "rolled-back"
    assert rolled.recover() == "rolled-back"

    accepted = _operation(
        cache, 2, 2, ("i", _readonly(np.arange(4, 8))),
    )
    _prepare_only(cache, accepted)
    assert cache.recover() == "accepted"
    with pytest.raises(RuntimeError, match="accepted"):
        accepted.rollback()
    assert accepted.recover() == "accepted"


def test_staged_commit_and_late_rollback_have_one_cas_winner(monkeypatch) -> None:
    cache = Browse1DCache(32)
    operation = _operation(cache, 1, 1, ("i", _readonly(np.arange(4))))
    _prepare_only(cache, operation)
    barrier = threading.Barrier(2)
    real_cas = cache._cas_state

    def racing_cas(expected, replacement):
        if (
            expected.phase is Browse1DCachePhase.STAGED_PENDING
            and replacement.phase in {
                Browse1DCachePhase.COMMITTING,
                Browse1DCachePhase.ROLLBACK_READY,
            }
        ):
            barrier.wait(timeout=1.0)
        return real_cas(expected, replacement)

    monkeypatch.setattr(cache, "_cas_state", racing_cas)
    results: list[str] = []
    errors: list[BaseException] = []

    def invoke(function) -> None:
        try:
            results.append(function())
        except BaseException as error:
            errors.append(error)

    threads = (
        threading.Thread(target=invoke, args=(operation.run,)),
        threading.Thread(target=invoke, args=(operation.rollback,)),
    )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(1.0)
        assert not thread.is_alive()
    assert errors == []
    assert len(set(results)) == 1
    assert results[0] in {"accepted", "rolled-back"}
    assert cache.phase is Browse1DCachePhase.OPEN


@pytest.mark.parametrize("driver_phase", ("commit", "accept", "rollback"))
def test_same_direction_helper_is_exclusive(
    monkeypatch, driver_phase: str,
) -> None:
    cache = Browse1DCache(32)
    operation = _operation(cache, 1, 1, ("i", _readonly(np.arange(4))))
    _prepare_only(cache, operation)
    if driver_phase == "accept":
        _commit_only(cache, operation)
    entered = threading.Event()
    release = threading.Event()
    method_name = driver_phase
    real_method = getattr(PhysicalRootExchange, method_name)
    exchange = cache._snapshot_state().journal.exchange

    def held(self):
        if self is exchange:
            entered.set()
            assert release.wait(1.0)
        return real_method(self)

    monkeypatch.setattr(PhysicalRootExchange, method_name, held)
    errors: list[BaseException] = []

    def drive() -> None:
        try:
            if driver_phase == "rollback":
                operation.rollback()
            else:
                operation.run()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=drive)
    thread.start()
    assert entered.wait(1.0)
    with pytest.raises(RuntimeError, match="busy"):
        operation.run()
    release.set()
    thread.join(1.0)
    assert not thread.is_alive()
    assert errors == []
    assert cache.phase is Browse1DCachePhase.OPEN


@pytest.mark.parametrize("method", ("prepare", "commit", "accept", "rollback"))
@pytest.mark.parametrize("cut", ("before", "after"))
def test_exchange_baseexception_fault_cuts_are_recoverable(
    monkeypatch, method: str, cut: str,
) -> None:
    cache = Browse1DCache(32)
    operation = _operation(cache, 1, 1, ("i", _readonly(np.arange(4))))
    if method in {"commit", "accept", "rollback"}:
        _prepare_only(cache, operation)
    if method == "accept":
        _commit_only(cache, operation)
    exchange = cache._snapshot_state().journal.exchange
    real_method = getattr(PhysicalRootExchange, method)
    fired = False

    def cut_method(self):
        nonlocal fired
        if self is not exchange or fired:
            return real_method(self)
        fired = True
        if cut == "after":
            real_method(self)
        raise _InjectedCut(f"{method}-{cut}")

    monkeypatch.setattr(PhysicalRootExchange, method, cut_method)
    with pytest.raises(_InjectedCut, match=f"{method}-{cut}"):
        if method == "rollback":
            operation.rollback()
        else:
            operation.run()
    monkeypatch.setattr(PhysicalRootExchange, method, real_method)

    if method == "accept" and cut == "after":
        assert operation.terminal_direction == "accepted"
    elif method == "rollback" and cut == "after":
        assert operation.terminal_direction == "rolled-back"
    else:
        assert operation.terminal_direction is None
    terminal = operation.recover()
    if method == "prepare" and cut == "before":
        assert terminal == "rolled-back"
    elif method == "rollback":
        assert terminal == "rolled-back"
    else:
        assert terminal == "accepted"
    assert cache.phase is Browse1DCachePhase.OPEN


@pytest.mark.parametrize(
    ("method", "transition", "terminal"),
    (
        ("prepare", "prepare-failure-settle", "rolled-back"),
        ("commit", "commit-failure-settle", "accepted"),
        ("accept", "accept-failure-settle", "accepted"),
        ("rollback", "rollback-failure-settle", "rolled-back"),
    ),
)
@pytest.mark.parametrize("cut", ("before", "after"))
def test_failure_settlement_cache_cas_cuts_are_recoverable(
    monkeypatch,
    method: str,
    transition: str,
    terminal: str,
    cut: str,
) -> None:
    cache = Browse1DCache(32)
    operation = _operation(cache, 1, 1, ("i", _readonly(np.arange(4))))
    if method in {"commit", "accept", "rollback"}:
        _prepare_only(cache, operation)
    if method == "accept":
        _commit_only(cache, operation)
    state = cache._snapshot_state()
    assert state.journal is not None
    exchange = state.journal.exchange
    real_method = getattr(PhysicalRootExchange, method)

    def fail_before_effect(self):
        if self is exchange:
            raise _InjectedCut(f"{method}-failure")
        return real_method(self)

    _install_cache_cas_cut(cache, monkeypatch, transition, cut)
    monkeypatch.setattr(PhysicalRootExchange, method, fail_before_effect)
    with pytest.raises(_InjectedCut, match=f"{transition}-{cut}"):
        if method == "rollback":
            operation.rollback()
        else:
            operation.run()
    monkeypatch.undo()
    assert operation.terminal_direction is None
    failed = cache._snapshot_state()
    if cut == "before":
        assert failed.driver is not None
        assert failed.driver.owner() is None
    if method == "rollback":
        result = operation.rollback()
    else:
        result = operation.recover()
    expected_terminal = (
        "accepted"
        if method == "prepare" and cut == "before"
        else terminal
    )
    assert result == expected_terminal
    assert operation.terminal_direction == expected_terminal
    assert cache.phase is Browse1DCachePhase.OPEN


@pytest.mark.parametrize("cut", ("before", "after"))
def test_close_driver_clear_cache_cas_cut_is_recoverable(
    monkeypatch, cut: str,
) -> None:
    cache = Browse1DCache(32)
    _operation(cache, 1, 1, ("i", _readonly(np.arange(4)))).run()

    def close_failure():
        raise _InjectedCut("authority-close")

    _install_cache_cas_cut(cache, monkeypatch, "close-driver-clear", cut)
    monkeypatch.setattr(cache._authority, "close", close_failure)
    with pytest.raises(_InjectedCut, match=f"close-driver-clear-{cut}"):
        cache.close()
    monkeypatch.undo()
    failed = cache._snapshot_state()
    assert failed.phase is Browse1DCachePhase.CLOSING_AUTHORITY
    if cut == "before":
        assert failed.driver is not None
        assert failed.driver.owner() is None
    else:
        assert failed.driver is None
    assert cache.recover() == "closed"
    assert cache.phase is Browse1DCachePhase.CLOSED


@pytest.mark.parametrize("cut", ("before", "after"))
def test_blocked_transition_cache_cas_cut_retains_or_recovers_exact_journal(
    monkeypatch, cut: str,
) -> None:
    cache = Browse1DCache(32)
    operation = _operation(cache, 1, 1, ("i", _readonly(np.arange(4))))
    _prepare_only(cache, operation)
    state = cache._snapshot_state()
    journal = state.journal
    assert journal is not None
    exchange = journal.exchange
    real_phase = PhysicalRootExchange.phase
    real_commit = PhysicalRootExchange.commit

    def no_commit(self):
        if self is exchange:
            return self.prepared_leases
        return real_commit(self)

    def failed_phase(self):
        if self is exchange:
            return PhysicalRootExchangePhase.FAILED
        return real_phase.__get__(self, type(self))

    _install_cache_cas_cut(cache, monkeypatch, "blocked", cut)
    monkeypatch.setattr(PhysicalRootExchange, "commit", no_commit)
    monkeypatch.setattr(PhysicalRootExchange, "phase", property(failed_phase))
    with pytest.raises(_InjectedCut, match=f"blocked-{cut}"):
        operation.run()
    monkeypatch.undo()
    assert operation.terminal_direction is None
    failed = cache._snapshot_state()
    assert failed.journal is journal
    if cut == "before":
        assert operation.recover() == "accepted"
        assert cache.phase is Browse1DCachePhase.OPEN
    else:
        assert failed.phase is Browse1DCachePhase.BLOCKED
        with pytest.raises(RuntimeError, match="blocked"):
            operation.recover()
        assert cache._snapshot_state() is failed
        assert cache._snapshot_state().journal is journal


def test_prejournal_reserve_and_cleanup_fault_leave_authority_ungated(
    monkeypatch,
) -> None:
    cache = Browse1DCache(64)
    real_reserve = PhysicalRootExchange.reserve
    real_rollback = PhysicalRootExchange.rollback

    def reserve_then_cut(self, value, semantic):
        real_reserve(self, value, semantic)
        raise _InjectedCut("reserve-after")

    def rollback_before(self):
        raise _InjectedCut("cleanup-before")

    monkeypatch.setattr(PhysicalRootExchange, "reserve", reserve_then_cut)
    monkeypatch.setattr(PhysicalRootExchange, "rollback", rollback_before)
    with pytest.raises(_InjectedCut, match="cleanup-before"):
        _operation(cache, 1, 1, ("i", _readonly(np.arange(4))))
    assert cache.phase is Browse1DCachePhase.OPEN
    assert cache.resident_bytes == 0
    assert cache._authority.semantic_references == 0

    monkeypatch.setattr(PhysicalRootExchange, "reserve", real_reserve)
    monkeypatch.setattr(PhysicalRootExchange, "rollback", real_rollback)
    assert _operation(
        cache, 2, 2, ("i", _readonly(np.arange(4, 8))),
    ).run() == "accepted"


@pytest.mark.parametrize("cut", ("before", "after"))
def test_initial_cache_cas_fault_never_strands_unowned_exchange(
    monkeypatch, cut: str,
) -> None:
    cache = Browse1DCache(32)
    _install_cache_cas_cut(cache, monkeypatch, "install", cut)
    with pytest.raises(_InjectedCut, match=f"install-{cut}"):
        _operation(cache, 1, 1, ("i", _readonly(np.arange(4))))
    monkeypatch.undo()
    if cut == "after":
        assert cache.phase is Browse1DCachePhase.PREPARING
        assert cache.recover() == "rolled-back"
    else:
        assert cache.phase is Browse1DCachePhase.OPEN
    assert cache.phase is Browse1DCachePhase.OPEN
    assert cache.resident_keys == ()
    assert cache.resident_bytes == 0


@pytest.mark.parametrize(
    "transition",
    (
        "prepare-claim",
        "prepare-settle",
        "commit-claim",
        "commit-settle",
        "accept-claim",
        "accept-terminal",
    ),
)
@pytest.mark.parametrize("cut", ("before", "after"))
def test_accept_path_cache_cas_faults_are_recoverable(
    monkeypatch, transition: str, cut: str,
) -> None:
    row = _readonly(np.arange(4))
    cache = Browse1DCache(32)
    operation = _operation(cache, 1, 1, ("i", row))
    _install_cache_cas_cut(cache, monkeypatch, transition, cut)
    with pytest.raises(_InjectedCut, match=f"{transition}-{cut}"):
        operation.run()
    monkeypatch.undo()
    expected_before_recovery = (
        "accepted"
        if transition == "accept-terminal" and cut == "after"
        else None
    )
    assert operation.terminal_direction == expected_before_recovery
    state = cache._snapshot_state()
    if transition.endswith("claim") and cut == "after":
        assert state.driver is not None
        assert state.driver.owner() is None
    assert operation.recover() == "accepted"
    assert operation._marker is None
    assert operation.terminal_direction == "accepted"
    assert cache.phase is Browse1DCachePhase.OPEN
    borrowed = cache.borrow(1, 1, "i")
    assert borrowed.array is row
    borrowed.release()


@pytest.mark.parametrize(
    "transition",
    ("rollback-request", "rollback-claim", "rollback-terminal"),
)
@pytest.mark.parametrize("cut", ("before", "after"))
def test_rollback_path_cache_cas_faults_are_recoverable(
    monkeypatch, transition: str, cut: str,
) -> None:
    cache = Browse1DCache(32)
    operation = _operation(cache, 1, 1, ("i", _readonly(np.arange(4))))
    _prepare_only(cache, operation)
    _install_cache_cas_cut(cache, monkeypatch, transition, cut)
    with pytest.raises(_InjectedCut, match=f"{transition}-{cut}"):
        operation.rollback()
    monkeypatch.undo()
    expected_before_recovery = (
        "rolled-back"
        if transition == "rollback-terminal" and cut == "after"
        else None
    )
    assert operation.terminal_direction == expected_before_recovery
    assert operation.rollback() == "rolled-back"
    assert operation._marker is None
    assert operation.terminal_direction == "rolled-back"
    assert cache.phase is Browse1DCachePhase.OPEN
    assert cache.resident_keys == ()


@pytest.mark.parametrize(
    "transition", ("prepare-rollback-request", "prepare-rollback-ready"),
)
@pytest.mark.parametrize("cut", ("before", "after"))
def test_prestage_rollback_cache_cas_faults_are_recoverable(
    monkeypatch, transition: str, cut: str,
) -> None:
    cache = Browse1DCache(32)
    operation = _operation(cache, 1, 1, ("i", _readonly(np.arange(4))))
    _install_cache_cas_cut(cache, monkeypatch, transition, cut)
    with pytest.raises(_InjectedCut, match=f"{transition}-{cut}"):
        operation.rollback()
    monkeypatch.undo()
    assert operation.terminal_direction is None
    assert operation.rollback() == "rolled-back"
    assert cache.phase is Browse1DCachePhase.OPEN
    assert cache.resident_keys == ()


@pytest.mark.parametrize(
    "transition",
    (
        "close-install",
        "close-exchange-settle",
        "close-claim",
        "close-terminal",
    ),
)
@pytest.mark.parametrize("cut", ("before", "after"))
def test_close_path_cache_cas_faults_are_recoverable(
    monkeypatch, transition: str, cut: str,
) -> None:
    cache = Browse1DCache(32)
    _operation(cache, 1, 1, ("i", _readonly(np.arange(4)))).run()
    _install_cache_cas_cut(cache, monkeypatch, transition, cut)
    with pytest.raises(_InjectedCut, match=f"{transition}-{cut}"):
        cache.close()
    monkeypatch.undo()
    if transition == "close-install" and cut == "before":
        assert cache.phase is Browse1DCachePhase.OPEN
        cache.close()
    else:
        assert cache.recover() == "closed"
    assert cache.phase is Browse1DCachePhase.CLOSED


def test_unknown_third_state_retains_exact_journal_and_refuses() -> None:
    cache = Browse1DCache(32)
    operation = _operation(cache, 1, 1, ("i", _readonly(np.arange(4))))
    state = cache._snapshot_state()
    journal = state.journal
    assert journal is not None
    third = module._state_with(
        state,
        phase=Browse1DCachePhase.BLOCKED,
        driver=None,
        replace_driver=True,
    )
    with cache._lock:
        cache._state = third
    with pytest.raises(RuntimeError, match="blocked"):
        operation.run()
    assert cache._snapshot_state() is third
    assert cache._snapshot_state().journal is journal


@pytest.mark.parametrize("cut", ("before", "after"))
def test_close_refuses_borrow_and_recovers_authority_close_fault(
    monkeypatch, cut: str,
) -> None:
    cache = Browse1DCache(32)
    _operation(cache, 1, 1, ("i", _readonly(np.arange(4)))).run()
    borrowed = cache.borrow(1, 1, "i")
    with pytest.raises(RuntimeError, match="outstanding borrows"):
        cache.close()
    borrowed.release()

    real_close = cache._authority.close
    fired = False

    def cut_close():
        nonlocal fired
        if fired:
            return real_close()
        fired = True
        if cut == "after":
            real_close()
        raise _InjectedCut(f"close-{cut}")

    monkeypatch.setattr(cache._authority, "close", cut_close)
    with pytest.raises(_InjectedCut, match=f"close-{cut}"):
        cache.close()
    assert cache.phase is Browse1DCachePhase.CLOSING_AUTHORITY
    monkeypatch.setattr(cache._authority, "close", real_close)
    assert cache.recover() == "closed"
    assert cache.phase is Browse1DCachePhase.CLOSED
    with pytest.raises(RuntimeError, match="closed"):
        cache.borrow(1, 1, "i")


def test_close_and_borrow_race_has_one_exact_state_winner(monkeypatch) -> None:
    cache = Browse1DCache(32)
    row = _readonly(np.arange(4))
    _operation(cache, 1, 1, ("i", row)).run()
    entered = threading.Event()
    release = threading.Event()
    real_cas = cache._cas_state

    def held_close_cas(expected, replacement):
        if (
            expected.phase is Browse1DCachePhase.OPEN
            and replacement.phase is Browse1DCachePhase.PREPARING
            and replacement.journal is not None
            and replacement.journal.closing
        ):
            entered.set()
            assert release.wait(1.0)
        return real_cas(expected, replacement)

    monkeypatch.setattr(cache, "_cas_state", held_close_cas)
    errors: list[BaseException] = []

    def close() -> None:
        try:
            cache.close()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=close)
    thread.start()
    assert entered.wait(1.0)
    borrowed = cache.borrow(1, 1, "i")
    assert borrowed.array is row
    release.set()
    thread.join(1.0)
    assert not thread.is_alive()
    assert len(errors) == 1
    assert cache.phase is Browse1DCachePhase.OPEN
    assert cache.outstanding_borrows == 1
    borrowed.release()


def test_close_accepts_empty_exchange_before_authority_close(monkeypatch) -> None:
    cache = Browse1DCache(32)
    _operation(cache, 1, 1, ("i", _readonly(np.arange(4)))).run()
    cache.record_scalars(1, 1, (("temperature", 300.0),))
    events: list[str] = []
    real_accept = PhysicalRootExchange.accept
    real_close = cache._authority.close

    def accept(self):
        assert not cache._lock.locked()
        result = real_accept(self)
        events.append("exchange-accepted")
        return result

    def close_authority():
        assert not cache._lock.locked()
        assert events == ["exchange-accepted"]
        events.append("authority-closed")
        return real_close()

    monkeypatch.setattr(PhysicalRootExchange, "accept", accept)
    monkeypatch.setattr(cache._authority, "close", close_authority)
    cache.close()
    assert events == ["exchange-accepted", "authority-closed"]
    assert cache.phase is Browse1DCachePhase.CLOSED
    closed = cache._snapshot_state()
    assert closed.rows == ()
    assert closed.catalog == ()
    assert closed.borrows == ()
    assert closed.journal is None
    assert closed.driver is None


@pytest.mark.parametrize("direction", ("accept", "rollback"))
def test_completion_marker_survives_later_operation_and_prunes_non_lifo(
    monkeypatch, direction: str,
) -> None:
    cache = Browse1DCache(128)
    first = _operation(cache, 1, 1, ("i", _readonly(np.arange(4))))
    if direction == "rollback":
        _prepare_only(cache, first)
    first_marker = first._marker
    assert first_marker is not None
    first_marker_ref = weakref.ref(first_marker)
    real_cas = cache._cas_state
    cut = True

    def terminal_cut(expected, replacement):
        nonlocal cut
        if (
            cut
            and replacement.phase is Browse1DCachePhase.OPEN
            and replacement.journal is None
            and replacement.terminal_evidence
            and replacement.terminal_evidence[-1]() is first_marker
        ):
            cut = False
            assert real_cas(expected, replacement)
            raise _InjectedCut(f"{direction} terminal")
        return real_cas(expected, replacement)

    monkeypatch.setattr(cache, "_cas_state", terminal_cut)
    with pytest.raises(_InjectedCut, match=f"{direction} terminal"):
        if direction == "accept":
            first.run()
        else:
            first.rollback()
    monkeypatch.setattr(cache, "_cas_state", real_cas)
    assert cache.phase is Browse1DCachePhase.OPEN
    assert first_marker_ref() is first_marker

    second = _operation(cache, 2, 2, ("i", _readonly(np.arange(4, 8))))
    assert second.run() == "accepted"
    assert module._marker_is_present(cache._snapshot_state(), first_marker)
    assert first.terminal_direction == (
        "accepted" if direction == "accept" else "rolled-back"
    )
    assert first.recover() == (
        "accepted" if direction == "accept" else "rolled-back"
    )
    del first_marker
    gc.collect()
    assert first_marker_ref() is None

    third = _operation(cache, 3, 3, ("i", _readonly(np.arange(8, 12))))
    assert third.run() == "accepted"
    evidence = cache._snapshot_state().terminal_evidence
    assert len(evidence) == 1
    assert evidence[0]() is None
