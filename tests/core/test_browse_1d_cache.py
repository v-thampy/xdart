from __future__ import annotations

import copy
from dataclasses import replace
import pickle
import weakref

import numpy as np
import pytest

import xrd_tools.io.browse_1d_cache as module
from xrd_tools.io.browse_1d_cache import (
    Browse1DCache, Browse1DCachePhase, Browse1DRowKey,
    default_browse_1d_cache_budget,
)


def _readonly(values, *, dtype=np.float64) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    array.flags.writeable = False
    return array


def _store(
    cache: Browse1DCache,
    frame: int,
    label: int,
    *rows: tuple[str, np.ndarray],
):
    return cache.store_rows(frame, label, tuple(rows))


def _naive_plan_rows(cache, state, pending, protected_rows=()):
    """The pre-H2-B planner retained as an exact ordering/error oracle."""

    incoming_keys = {row.key for row in pending}
    borrowed = cache._borrowed_identities(state)
    protected = {id(row) for row in protected_rows}
    if len(protected) != len(protected_rows) or any(
        not any(row is resident for resident in state.rows)
        for row in protected_rows
    ):
        raise RuntimeError("protected Browse 1-D rows changed")
    mandatory = tuple(row for row in state.rows if row.key in incoming_keys)
    if any(id(row) in protected for row in mandatory):
        raise ValueError("protected Browse 1-D row cannot be replaced")
    if any(id(row.row_identity) in borrowed for row in mandatory):
        raise RuntimeError("borrowed Browse 1-D row cannot be replaced")
    survivors = [row for row in state.rows if row.key not in incoming_keys]
    victims = list(mandatory)
    while (
        len(survivors) + len(pending) > module._MAX_BROWSE_1D_RESIDENT_ROWS
        or cache._projected_unique_bytes(tuple(survivors), pending)
        > cache._budget
    ):
        row_limit_exceeded = (
            len(survivors) + len(pending)
            > module._MAX_BROWSE_1D_RESIDENT_ROWS
        )
        eligible = [
            row for row in survivors
            if id(row.row_identity) not in borrowed and id(row) not in protected
        ]
        if not eligible:
            message = (
                "Browse 1-D rows exceed the resident row limit"
                if row_limit_exceeded
                else "Browse 1-D rows exceed the cache budget"
            )
            raise ValueError(message)
        oldest = min(eligible, key=lambda row: row.touch)
        survivors.remove(oldest)
        victims.append(oldest)
    return tuple(survivors), tuple(victims)


def test_default_budget_is_five_percent_capped_with_gib_fallback(
    monkeypatch,
) -> None:
    gib = 1 << 30
    assert default_browse_1d_cache_budget(10 * gib) == gib // 2
    assert default_browse_1d_cache_budget(100 * gib) == gib
    assert default_browse_1d_cache_budget(True) == gib
    monkeypatch.setattr(module, "_detect_physical_ram_bytes", lambda: None)
    assert default_browse_1d_cache_budget() == gib


def test_cache_store_and_borrow_refuse_copy_deepcopy_and_pickle() -> None:
    cache = Browse1DCache(64)
    operation = _store(cache, 1, 1, ("i", _readonly(np.arange(4))))
    assert operation == "accepted"
    borrowed = cache.borrow(1, 1, "i")
    try:
        for owner in (cache, borrowed):
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
    _store(cache, 1, 1, ("i", first))
    borrowed = cache.borrow(1, 1, "i")
    exact_key = Browse1DRowKey(1, 1, "i")
    assert borrowed.key == exact_key
    assert borrowed.array is first
    with pytest.raises(AttributeError):
        borrowed.key = Browse1DRowKey(2, 2, "i")
    with pytest.raises(AttributeError):
        borrowed.array = second
    with pytest.raises(RuntimeError, match="borrowed"):
        _store(cache, 1, 1, ("i", second))
    assert borrowed.key == exact_key
    assert borrowed.array is first
    borrowed.release()
    _store(cache, 1, 1, ("i", second))
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

    assert _store(
        cache, 1, 7, ("intensity", first), ("sigma", second),
    ) == "accepted"
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
        _store(cache, 2, 8, ("intensity", incoming))
    assert cache.resident_keys[0] == borrowed.key

    borrowed.release()
    assert _store(cache, 2, 8, ("intensity", incoming)) == "accepted"
    assert cache.resident_keys == (Browse1DRowKey(2, 8, "intensity"),)
    assert dict(cache.scalars(1, 7)) == {"temperature": 300.0}


def test_resident_row_limit_uses_lru_and_counts_shared_roots(
    monkeypatch,
) -> None:
    monkeypatch.setattr(module, "_MAX_BROWSE_1D_RESIDENT_ROWS", 2)
    shared = _readonly(np.arange(8), dtype=np.uint8)
    cache = Browse1DCache(1 << 20)
    _store(
        cache,
        1,
        1,
        ("first", shared[:4]),
        ("second", shared[4:]),
    )
    assert cache.resident_root_count == 1
    recent = cache.borrow(1, 1, "first")
    recent.release()

    _store(
        cache, 2, 2, ("third", _readonly([8], dtype=np.uint8)),
    )

    assert cache.resident_keys == (
        Browse1DRowKey(1, 1, "first"),
        Browse1DRowKey(2, 2, "third"),
    )


def test_resident_row_limit_refuses_before_admission_when_rows_are_pinned(
    monkeypatch,
) -> None:
    monkeypatch.setattr(module, "_MAX_BROWSE_1D_RESIDENT_ROWS", 2)
    cache = Browse1DCache(1 << 20)
    _store(
        cache,
        1,
        1,
        ("borrowed", _readonly([1], dtype=np.uint8)),
        ("protected", _readonly([2], dtype=np.uint8)),
    )
    borrowed = cache.borrow(1, 1, "borrowed")
    before = cache._snapshot_state()
    roots = cache._authority.retained_roots
    try:
        with pytest.raises(ValueError, match="resident row limit"):
            cache.store_rows(
                2,
                2,
                (("incoming", _readonly([3], dtype=np.uint8)),),
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
        _store(
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
    _store(
        cache, 1, 1, ("row", _readonly([1], dtype=np.uint8)),
    )
    replacement = _readonly([2], dtype=np.uint8)
    assert _store(cache, 1, 1, ("row", replacement)) == "accepted"
    assert cache.resident_keys == (Browse1DRowKey(1, 1, "row"),)
    with cache.borrow(1, 1, "row") as borrowed:
        assert borrowed.array is replacement


@pytest.mark.parametrize(
    ("row_limit", "budget"),
    ((16, 1 << 20), (16, 30), (6, 1 << 20), (5, 23), (5, 22)),
)
def test_one_pass_planner_matches_naive_shared_root_lru_oracle(
    monkeypatch,
    row_limit: int,
    budget: int,
) -> None:
    shared = _readonly(np.arange(8), dtype=np.uint8)
    cache = Browse1DCache(1 << 20)
    _store(
        cache,
        1,
        1,
        ("a", shared[:4]),
        ("b", shared[4:]),
        ("c", _readonly(np.arange(4), dtype=np.uint8)),
        ("d", _readonly(np.arange(5), dtype=np.uint8)),
        ("e", _readonly(np.arange(6), dtype=np.uint8)),
        ("f", _readonly(np.arange(7), dtype=np.uint8)),
    )
    first_borrow = cache.borrow(1, 1, "a")
    last_borrow = cache.borrow(1, 1, "f")
    try:
        state = cache._snapshot_state()
        # Equal touches prove the heap's original-position tie break matches
        # repeated stable min/remove behavior.
        state = replace(
            state,
            rows=tuple(
                replace(row, touch=4)
                if row.key.name in {"d", "e"}
                else row
                for row in state.rows
            ),
        )
        protected = (
            next(row for row in state.rows if row.key.name == "b"),
        )
        pending = cache._normalize_rows(
            1,
            1,
            (
                ("c", _readonly(np.arange(8), dtype=np.uint8)),
                ("g", shared[:2]),
            ),
            state.clock,
        )
        monkeypatch.setattr(
            module, "_MAX_BROWSE_1D_RESIDENT_ROWS", row_limit,
        )
        cache._budget = budget

        def outcome(planner):
            try:
                survivors, victims = planner(
                    cache, state, pending, protected,
                )
            except Exception as error:
                return type(error), str(error)
            return (
                tuple(id(row) for row in survivors),
                tuple(id(row) for row in victims),
            )

        expected = outcome(
            lambda owner, exact, incoming, pinned:
                _naive_plan_rows(owner, exact, incoming, pinned)
        )
        actual = outcome(
            lambda owner, exact, incoming, pinned:
                owner._plan_rows(exact, incoming, pinned)
        )
        assert actual == expected
    finally:
        first_borrow.release()
        last_borrow.release()


def test_one_pass_planner_preserves_mixed_replacement_error_precedence() -> None:
    cache = Browse1DCache(1 << 20)
    _store(
        cache,
        1,
        1,
        ("borrowed", _readonly([1], dtype=np.uint8)),
        ("protected", _readonly([2], dtype=np.uint8)),
    )
    borrowed = cache.borrow(1, 1, "borrowed")
    try:
        state = cache._snapshot_state()
        protected = (
            next(row for row in state.rows if row.key.name == "protected"),
        )
        pending = cache._normalize_rows(
            1,
            1,
            (
                ("borrowed", _readonly([3], dtype=np.uint8)),
                ("protected", _readonly([4], dtype=np.uint8)),
            ),
            state.clock,
        )

        with pytest.raises(
            ValueError, match="protected Browse 1-D row cannot be replaced",
        ):
            _naive_plan_rows(cache, state, pending, protected)
        with pytest.raises(
            ValueError, match="protected Browse 1-D row cannot be replaced",
        ):
            cache._plan_rows(state, pending, protected)
    finally:
        borrowed.release()


def test_one_pass_planner_preserves_row_cap_before_private_fact_validation(
    monkeypatch,
) -> None:
    shared = _readonly(np.arange(8), dtype=np.uint8)
    cache = Browse1DCache(1 << 20)
    _store(
        cache,
        1,
        1,
        ("oldest", shared[:4]),
        ("retained", shared[4:]),
    )
    state = cache._snapshot_state()
    state = replace(
        state,
        rows=(
            replace(
                state.rows[0],
                fact=replace(
                    state.rows[0].fact,
                    nbytes=state.rows[0].fact.nbytes + 1,
                ),
            ),
            state.rows[1],
        ),
    )
    pending = cache._normalize_rows(
        2,
        2,
        (("incoming", _readonly([9], dtype=np.uint8)),),
        state.clock,
    )
    monkeypatch.setattr(module, "_MAX_BROWSE_1D_RESIDENT_ROWS", 2)

    expected = _naive_plan_rows(cache, state, pending)
    actual = cache._plan_rows(state, pending)
    assert tuple(id(row) for row in actual[0]) == tuple(
        id(row) for row in expected[0]
    )
    assert tuple(id(row) for row in actual[1]) == tuple(
        id(row) for row in expected[1]
    )


def test_row_validation_refuses_mutable_wide_and_record_fallbacks() -> None:
    cache = Browse1DCache(128)
    with pytest.raises(ValueError, match="read-only"):
        _store(cache, 1, 1, ("mutable", np.arange(4)))
    wide = np.empty((2, 4), dtype=np.int64)
    wide[...] = np.arange(8).reshape(2, 4)
    assert wide.base is None
    wide.flags.writeable = False
    with pytest.raises(ValueError, match="numeric and 1-D"):
        _store(cache, 1, 1, ("wide", wide))
    row = wide[0]
    assert row.ndim == 1 and not row.flags.writeable
    with pytest.raises(ValueError, match="wide root"):
        _store(cache, 1, 1, ("record-row", row))


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
        _store(cache, 1, 1, ("row", row))
    assert cache.phase is Browse1DCachePhase.OPEN
    assert cache.resident_bytes == 0


def test_readonly_leaf_refuses_writable_physical_root() -> None:
    root = np.arange(8, dtype=np.float64)
    row = root[:4]
    row.flags.writeable = False
    assert root.flags.writeable and not row.flags.writeable
    cache = Browse1DCache(root.nbytes)
    with pytest.raises(ValueError, match="physical root must be read-only"):
        _store(cache, 1, 1, ("row", row))
    root[0] = 99.0
    assert cache.resident_bytes == 0


def _row(values) -> np.ndarray:
    value = np.asarray(values, dtype=np.float64)
    value.setflags(write=False)
    return value

def test_failed_fill_discards_evicted_rereadable_victim_without_restoring_fifo(monkeypatch) -> None:
    cache = Browse1DCache(32)
    cache.store_rows(1, 1, (("old", _row(range(4))),))
    original = cache._authority.reserve
    def fail_reserve():
        raise RuntimeError("incoming reservation failed")
    monkeypatch.setattr(cache._authority, "reserve", fail_reserve)
    with pytest.raises(RuntimeError, match="incoming reservation failed"):
        cache.store_rows(2, 2, (("new", _row(range(4, 8))),))
    assert cache.resident_keys == ()
    monkeypatch.setattr(cache._authority, "reserve", original)
    assert cache.store_rows(2, 2, (("new", _row(range(4, 8))),)) == "accepted"
    assert tuple(key.name for key in cache.resident_keys) == ("new",)
    cache.close()

def test_stale_complete_bundle_discards_incoming_without_partial_publication() -> None:
    cache = Browse1DCache(64)
    with pytest.raises(InterruptedError, match="stale"):
        cache.store_rows(1, 1, (("a", _row(range(4))), ("b", _row(range(4, 8)))), cancelled=lambda: True)
    assert cache.resident_keys == ()
    assert cache.resident_bytes == 0
    cache.close()

def test_close_refuses_borrows_then_releases_all_actual_charges() -> None:
    cache = Browse1DCache(64)
    cache.store_rows(1, 1, (("i", _row(range(4))),))
    borrow = cache.borrow(1, 1, "i")
    with pytest.raises(RuntimeError, match="outstanding borrows"):
        cache.close()
    borrow.release(); cache.close()
    assert cache.phase is Browse1DCachePhase.CLOSED


def test_evicted_array_dies_before_replacement_capacity_is_reserved(monkeypatch):
    cache = Browse1DCache(32)
    victim = _readonly(range(4))
    reference = weakref.ref(victim)
    cache.store_rows(1, 1, (("old", victim),))
    del victim
    reserve = cache._authority.reserve

    def require_detached():
        assert reference() is None
        assert cache._authority.retained_bytes == 0
        return reserve()

    monkeypatch.setattr(cache._authority, "reserve", require_detached)
    cache.store_rows(2, 2, (("new", _readonly(range(4, 8))),))
    cache.close()


def test_close_retries_failed_eviction_and_releases_surviving_rows(monkeypatch):
    cache = Browse1DCache(64)
    cache.store_rows(1, 1, (("first", _readonly(range(4))), ("second", _readonly(range(4)))))
    release = cache._authority._release
    failures = []

    def fail_once(token):
        if not failures:
            failures.append(token)
            raise OSError("lease release failed")
        return release(token)

    monkeypatch.setattr(cache._authority, "_release", fail_once)
    with pytest.raises(OSError, match="lease release failed"):
        cache.store_rows(2, 2, (("new", _readonly(range(4))),))
    assert cache._authority.retained_bytes == 64
    cache.close()
    assert cache.phase is Browse1DCachePhase.CLOSED
    assert cache._authority.retained_bytes == 0


def test_late_cancel_discards_whole_bundle_and_allows_reread_without_locked_callback():
    cache = Browse1DCache(64)
    calls = []

    def cancelled():
        assert cache._lock.acquire(blocking=False)
        cache._lock.release()
        with pytest.raises(RuntimeError, match="in progress"):
            cache.close()
        calls.append(True)
        return len(calls) == 2

    rows = (("a", _readonly(range(4))), ("b", _readonly(range(4))))
    with pytest.raises(InterruptedError):
        cache.store_rows(1, 1, rows, cancelled=cancelled)
    assert len(calls) == 2
    assert cache.resident_keys == ()
    assert cache.resident_bytes == 0
    assert cache.store_rows(1, 1, rows) == "accepted"
    cache.close()


def test_scalar_catalog_replaces_values_and_missing_is_empty():
    cache = Browse1DCache(32)
    assert dict(cache.scalars(1, 1)) == {}
    cache.record_scalars(1, 1, (("temperature", 300.0),))
    cache.record_scalars(1, 1, (("temperature", 310.0),))
    assert dict(cache.scalars(1, 1)) == {"temperature": 310.0}
    cache.close()
