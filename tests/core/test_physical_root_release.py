"""INT-PERF-DIRECT-20261015: the physical-root authority releases a lease batch in one validated transition.

Frozen before the change, with Codex's pre-writer CR1/CR2 contracts. O4a uses only the existing single-lease API and
must pass on the parent. O4b and L1 use the new ``release_leases``, imported inside each row (the parent lacks it).
"""
from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

from xrd_tools.core import physical_memory
from xrd_tools.core.physical_memory import PhysicalRootAuthority


def _held(authority, *roots, shared=None):
    """One committed lease per root (semantic ``s<i>``); ``shared`` names a second lease on the first root."""
    reservation = authority.reserve()
    for index, root in enumerate(roots):
        reservation.reserve(root, f"s{index}")
    if shared is not None:
        reservation.reserve(roots[0][:1], shared)
    return reservation.commit()


def test_o4a_single_lease_release_keeps_its_contract():
    base, other = np.zeros(64, np.uint8), np.zeros(32, np.uint8)
    authority = PhysicalRootAuthority(1024)
    leases = _held(authority, base, other, shared="view")
    leases["view"].release()
    assert (authority.retained_bytes, authority.retained_root_count) == (96, 2), "a shared root stays charged"
    leases["view"].release()  # a second release is a no-op
    assert authority.semantic_references == 2
    reservation = authority.reserve()  # releasing during an active reservation keeps it
    leases["s1"].release()
    reservation.reserve(np.zeros(16, np.uint8), "late")
    late = reservation.commit()
    assert (authority.retained_bytes, authority.semantic_references) == (64 + 16, 2)
    authority.close()
    leases["s0"].release()  # stale after the authority closed: a no-op
    late["late"].release()
    assert leases["s0"].released and late["late"].released


def test_o4b_release_leases_is_one_checked_batch():
    from xrd_tools.core.physical_memory import release_leases
    base, other, third = (np.zeros(size, np.uint8) for size in (64, 32, 16))
    authority = PhysicalRootAuthority(1024)
    leases = _held(authority, base, other, third, shared="view")
    release_leases(())  # empty: a no-op
    release_leases([leases["view"], leases["view"], leases["s1"]])  # a duplicate lease counts once
    assert leases["view"].released and leases["s1"].released
    assert (authority.retained_bytes, authority.semantic_references) == (64 + 16, 2), "the shared root survives"
    release_leases([leases["view"], leases["s1"]])  # all already released: a no-op
    foreign = PhysicalRootAuthority(1024)
    stranger = _held(foreign, np.zeros(8, np.uint8))["s0"]
    with pytest.raises(ValueError, match="one authority"):
        release_leases([leases["s2"], stranger])
    with pytest.raises(TypeError):
        release_leases([leases["s2"], object()])
    assert not leases["s2"].released and not stranger.released, "a refused batch releases nothing"
    assert (authority.retained_bytes, foreign.retained_bytes) == (80, 8)
    reservation = authority.reserve()  # an active reservation is preserved
    release_leases([leases["s2"]])
    reservation.reserve(np.zeros(4, np.uint8), "late")
    late = reservation.commit()
    assert (authority.retained_bytes, authority.semantic_references) == (64 + 4, 2)
    authority.close()
    release_leases([leases["s0"], late["late"]])  # stale after the authority closed: a no-op
    assert leases["s0"].released and late["late"].released
    stranger.release()
    foreign.close()


@pytest.mark.parametrize("winning_semantic", ("s0", "s1"), ids=("same_lease", "other_lease"))
def test_l1_a_lost_swap_retries_and_drops_roots_outside_the_lock(monkeypatch, winning_semantic):
    from xrd_tools.core.physical_memory import release_leases
    authority = PhysicalRootAuthority(1024)
    root = np.zeros(64, np.uint8)
    leases = _held(authority, root, np.zeros(32, np.uint8))
    observed = []
    weakref.finalize(root, lambda: observed.append(authority._lock._is_owned()))
    del root
    real = physical_memory._replacement_state
    calls = []

    def replacement_state(*args, **kwargs):  # the real delta; its first use lets another release win the swap
        calls.append(1)
        built = real(*args, **kwargs)
        if len(calls) == 1:
            # If the winner also removes s0, its root is held only by the losing snapshot: replacing that
            # snapshot must not run its finalizer under the lock. The other-lease case requires a new delta.
            leases[winning_semantic].release()
        return built
    monkeypatch.setattr(physical_memory, "_replacement_state", replacement_state)
    release_leases([leases["s0"]])
    gc.collect()
    assert len(calls) == (2 if winning_semantic == "s0" else 3), "the lost swap recomputes live victims"
    assert observed == [False], "the released root died once, with the authority lock not held"
    assert authority.retained_bytes == (32 if winning_semantic == "s0" else 0)
    leases["s1"].release()
    assert (authority.retained_bytes, authority.retained_root_count, authority.semantic_references) == (0, 0, 0)
    authority.close()
