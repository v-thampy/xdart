from __future__ import annotations

import numpy as np
import pytest

from xrd_tools.io.browse_1d_cache import Browse1DCache, Browse1DCachePhase, default_browse_1d_cache_budget


def _row(values) -> np.ndarray:
    value = np.asarray(values, dtype=np.float64)
    value.setflags(write=False)
    return value


def test_default_budget_is_five_percent_capped_with_gib_fallback() -> None:
    gib = 1 << 30
    assert default_browse_1d_cache_budget(10 * gib) == gib // 2
    assert default_browse_1d_cache_budget(100 * gib) == gib
    assert default_browse_1d_cache_budget(True) == gib


def test_borrow_is_getter_only_and_pins_exact_shared_root() -> None:
    cache = Browse1DCache(64)
    root = _row(range(4))
    assert cache.store_rows(1, 1, (("i", root), ("alias", root))) == "accepted"
    assert cache.resident_root_count == 1
    borrow = cache.borrow(1, 1, "i")
    assert borrow.array is root
    with pytest.raises(RuntimeError, match="borrowed"):
        cache.store_rows(1, 1, (("i", _row(range(4))),))
    borrow.release()
    assert cache.outstanding_borrows == 0
    cache.close()


def test_lru_evicts_unborrowed_victim_before_reserving_replacement() -> None:
    cache = Browse1DCache(32)
    first, second = _row(range(4)), _row(range(4, 8))
    cache.store_rows(1, 1, (("first", first),))
    cache.store_rows(2, 2, (("second", second),))
    assert tuple(key.name for key in cache.resident_keys) == ("second",)
    assert cache.resident_bytes == second.nbytes
    cache.close()


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


def test_row_cap_refuses_atomic_incoming_bundle_when_all_rows_are_pinned(monkeypatch) -> None:
    import xrd_tools.io.browse_1d_cache as module
    monkeypatch.setattr(module, "_MAX_BROWSE_1D_RESIDENT_ROWS", 2)
    cache = Browse1DCache(1024)
    cache.store_rows(1, 1, (("a", _row([1])), ("b", _row([2]))))
    first, second = cache.borrow(1, 1, "a"), cache.borrow(1, 1, "b")
    with pytest.raises(ValueError, match="resident row limit"):
        cache.store_rows(2, 2, (("c", _row([3])),))
    assert tuple(key.name for key in cache.resident_keys) == ("a", "b")
    first.release(); second.release(); cache.close()


def test_close_refuses_borrows_then_releases_all_actual_charges() -> None:
    cache = Browse1DCache(64)
    cache.store_rows(1, 1, (("i", _row(range(4))),))
    borrow = cache.borrow(1, 1, "i")
    with pytest.raises(RuntimeError, match="outstanding borrows"):
        cache.close()
    borrow.release(); cache.close()
    assert cache.phase is Browse1DCachePhase.CLOSED
