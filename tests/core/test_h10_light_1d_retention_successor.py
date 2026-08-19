"""Finite Qt-free byte-owner contract for retained light 1-D rows."""
from __future__ import annotations

from collections import deque
from dataclasses import replace
import copy
import gc
import json
from pathlib import Path
import subprocess
import sys
import threading
import weakref

import numpy as np
import pytest


def _api():
    from xrd_tools.session import (
        GUIThreadHydrationRefused,
        Light1DBorrow,
        Light1DBufferLayout,
        Light1DCleanupHooks,
        Light1DCleanupPending,
        Light1DHydrationToken,
        Light1DLayout,
        Light1DModeData,
        Light1DModeLayout,
        Light1DRecord,
        Light1DRetentionLease,
        Light1DStaleGeneration,
        Light1DUnavailable,
        SessionResourceAuthority,
        acquire_light_1d_retention,
    )

    return locals()


def _layout(*, n=4):
    api = _api()
    Buffer = api["Light1DBufferLayout"]
    Mode = api["Light1DModeLayout"]
    Layout = api["Light1DLayout"]
    axis = Buffer(
        length=n, itemsize=8, owner_key="q-axis", dtype="<f8", shared=True,
    )
    return Layout(
        modes=(
            Mode(
                mode="raw",
                coordinate=axis,
                intensity=Buffer(n, 8, "raw-intensity", "<f8"),
                uncertainty=Buffer(n, 8, "raw-sigma", "<f8"),
            ),
            Mode(
                mode="bg-subtracted",
                coordinate=axis,
                intensity=Buffer(n, 4, "bg-intensity", "<f4"),
                uncertainty=None,
            ),
        ),
        active_mode="bg-subtracted",
    )


def _record(label, axis, *, n=4, generation=3):
    api = _api()
    Data, Record = api["Light1DModeData"], api["Light1DRecord"]
    return Record(
        row_identity=label,
        generation=generation,
        active_mode="bg-subtracted",
        modes={
            "raw": Data(
                coordinate=axis,
                intensity=np.full(n, label, dtype=np.float64),
                uncertainty=np.full(n, label / 10, dtype=np.float64),
            ),
            "bg-subtracted": Data(
                coordinate=axis,
                intensity=np.full(n, label, dtype=np.float32),
                uncertainty=None,
            ),
        },
        provenance={"source": "scan.nxs", "logical": int(label)},
    )


def _lease(*, rows=3, ceiling=None, capacity=1_000_000, generation=3,
           committed=None):
    api = _api()
    layout = _layout()
    Authority = api["SessionResourceAuthority"]
    acquire = api["acquire_light_1d_retention"]
    authority = Authority(
        capacity_bytes=capacity,
        committed_bytes=committed or {},
    )
    requested = layout.shared_bytes + rows * layout.per_row_unique_ndarray_bytes
    lease = acquire(
        authority,
        owner="browse:scan",
        generation=generation,
        layout=layout,
        requested_rows=rows,
        compatibility_byte_ceiling=requested if ceiling is None else ceiling,
        gui_thread_id=threading.get_ident(),
    )
    return authority, lease


def test_layout_accounts_every_mode_sigma_dtype_and_unique_shared_axis():
    authority, lease = _lease(rows=2)
    layout = lease.layout
    assert layout.shared_bytes == 4 * 8
    assert layout.per_row_unique_ndarray_bytes == (4 * 8 + 4 * 8 + 4 * 4)
    assert lease.reserved_ndarray_bytes == (
        layout.shared_bytes + 2 * layout.per_row_unique_ndarray_bytes
    )
    assert lease.row_cap == 2
    assert authority.snapshot().categories == {
        "light_1d": lease.reserved_ndarray_bytes,
    }


def test_exact_byte_ceiling_reduces_legacy_row_request_without_a_floor():
    layout = _layout()
    ceiling = layout.shared_bytes + 3 * layout.per_row_unique_ndarray_bytes
    _authority, lease = _lease(rows=4096, ceiling=ceiling)
    assert lease.row_cap == 3
    assert lease.reserved_ndarray_bytes == ceiling
    assert lease.requested_rows == 4096


def test_one_authority_bounds_multiple_live_leases_without_overcommit():
    api = _api()
    layout = _layout()
    one = layout.shared_bytes + layout.per_row_unique_ndarray_bytes
    authority = api["SessionResourceAuthority"](capacity_bytes=2 * one)
    first = api["acquire_light_1d_retention"](
        authority, owner="scan-a", generation=1, layout=layout,
        requested_rows=2, compatibility_byte_ceiling=2 * one,
        gui_thread_id=threading.get_ident(),
    )
    second = api["acquire_light_1d_retention"](
        authority, owner="scan-b", generation=1, layout=layout,
        requested_rows=2, compatibility_byte_ceiling=2 * one,
        gui_thread_id=threading.get_ident(),
    )
    snapshot = authority.snapshot()
    assert first.reserved_ndarray_bytes + second.reserved_ndarray_bytes <= (
        authority.capacity_bytes
    )
    assert first.row_cap == 2
    assert second.row_cap == 0
    assert snapshot.reserved_bytes <= snapshot.capacity_bytes
    assert snapshot.reservation_count == 2


def test_zero_byte_grant_still_fences_duplicate_owner_generation():
    api = _api()
    layout = _layout()
    authority = api["SessionResourceAuthority"](capacity_bytes=0)
    first = api["acquire_light_1d_retention"](
        authority, owner="zero", generation=1, layout=layout,
        requested_rows=1, compatibility_byte_ceiling=0,
        gui_thread_id=threading.get_ident(),
    )
    assert first.row_cap == first.reserved_ndarray_bytes == 0
    assert authority.snapshot().reservation_count == 1
    with pytest.raises(RuntimeError, match="already has"):
        api["acquire_light_1d_retention"](
            authority, owner="zero", generation=1, layout=layout,
            requested_rows=1, compatibility_byte_ceiling=0,
            gui_thread_id=threading.get_ident(),
        )


def test_only_coordinate_buffers_may_be_declared_shared():
    api = _api()
    Buffer, Mode, Layout = (
        api["Light1DBufferLayout"], api["Light1DModeLayout"],
        api["Light1DLayout"],
    )
    with pytest.raises(ValueError, match="only coordinate"):
        Layout((Mode(
            "raw", Buffer(4, 8, "axis", "<f8", shared=True),
            Buffer(4, 8, "intensity", "<f8", shared=True),
        ),), "raw")


def test_shared_flag_is_an_exact_immutable_bool_before_grant_accounting():
    api = _api()
    Buffer = api["Light1DBufferLayout"]

    class MutableTruth:
        def __init__(self):
            self.value = True

        def __bool__(self):
            return self.value

    flag = MutableTruth()
    with pytest.raises(TypeError, match="shared must be an exact bool"):
        Buffer(4, 8, "axis", "<f8", shared=flag)


def test_deterministic_oldest_eviction_uses_grant_not_requested_rows():
    api = _api()
    _authority, lease = _lease(rows=5, ceiling=(
        _layout().shared_bytes + 3 * _layout().per_row_unique_ndarray_bytes
    ))
    axis = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    for label in range(5):
        lease.retain(
            _record(label, axis),
            grant_id=lease.grant_id,
            generation=lease.generation,
        )
    assert lease.keys() == (2, 3, 4)
    assert lease.evicted == (0, 1)
    first = lease.get(3)
    second = lease.borrow(3)
    assert isinstance(first, api["Light1DBorrow"])
    assert isinstance(second, api["Light1DBorrow"])
    assert first is not second
    assert first.row_identity == second.row_identity == 3
    first.close()
    second.close()
    assert lease.keys() == (2, 3, 4)  # access is not LRU mutation


def test_retain_does_not_walk_the_resident_root_prefix():
    """A new row consults only its roots and the inverse ownership index."""
    class NoPrefixWalkDict(dict):
        def __iter__(self):
            raise AssertionError("resident ownership prefix was iterated")

        def items(self):
            raise AssertionError("resident root prefix was walked")

        def values(self):
            raise AssertionError("resident ownership values were walked")

    _authority, lease = _lease(rows=4)
    axis = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    for label in (0, 1):
        lease.retain(
            _record(label, axis),
            grant_id=lease.grant_id,
            generation=lease.generation,
        )

    lease._roots = NoPrefixWalkDict(lease._roots)
    lease._root_owners = NoPrefixWalkDict(lease._root_owners)
    lease.retain(
        _record(2, axis),
        grant_id=lease.grant_id,
        generation=lease.generation,
    )

    assert lease.contains(0)
    assert lease.contains(2)
    assert lease.retained_count == 3
    assert lease.oldest_row_identity == 0


def test_shared_root_index_survives_zero_rows_and_unique_index_does_not_pin():
    _authority, lease = _lease(rows=1)
    axis = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    lease.retain(
        _record(0, axis),
        grant_id=lease.grant_id,
        generation=lease.generation,
    )
    shared_root = lease._shared_roots["q-axis"]
    shared_id = id(shared_root)
    unique_ids = {
        id(root) for owner_key, root in lease._roots[0].items()
        if not lease.layout._groups[owner_key].shared
    }

    assert lease.retire(
        0, grant_id=lease.grant_id, generation=lease.generation,
    )
    shared_entry = lease._root_owners[shared_id]
    assert shared_entry.root is shared_root
    assert shared_entry.owner_key == "q-axis"
    assert shared_entry.rows == set()
    assert not (unique_ids & lease._root_owners.keys())

    # With a one-row grant, this succeeds only if no retired unique root is
    # kept alive by the inverse index while its payload receipt is pruned.
    gc.collect()
    lease.retain(
        _record(1, axis),
        grant_id=lease.grant_id,
        generation=lease.generation,
    )
    assert lease.keys() == (1,)
    assert lease._root_owners[shared_id].rows == {1}


@pytest.mark.parametrize(
    ("candidate_owner", "resident_owner"),
    (("raw-intensity", "raw-intensity"),
     ("raw-sigma", "raw-intensity")),
)
def test_inverse_root_ownership_preserves_shared_and_foreign_alias_semantics(
    candidate_owner, resident_owner,
):
    _authority, lease = _lease(rows=3)
    axis = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    lease.retain(
        _record(0, axis),
        grant_id=lease.grant_id,
        generation=lease.generation,
    )
    lease.retain(
        _record(1, axis),
        grant_id=lease.grant_id,
        generation=lease.generation,
    )

    # The one declared shared owner is intentionally reused by both rows.
    assert lease._roots[0]["q-axis"] is lease._roots[1]["q-axis"]
    before = (
        lease.keys(), lease.owned_buffer_ids,
        lease.unique_owned_ndarray_bytes,
    )
    canonical, roots, supplied, views = lease._canonicalize_record(
        _record(2, axis),
    )
    attacked = dict(roots)
    attacked[candidate_owner] = lease._roots[0][resident_owner]

    with pytest.raises(ValueError, match="another owner"):
        lease._retain_validated(canonical, attacked, supplied, views)

    assert (
        lease.keys(), lease.owned_buffer_ids,
        lease.unique_owned_ndarray_bytes,
    ) == before

    # A normal same-row replacement remains legal and moves that row to the
    # newest insertion position, exactly as the prior map implementation did.
    lease.retain(
        _record(0, axis),
        grant_id=lease.grant_id,
        generation=lease.generation,
    )
    assert lease.keys() == (1, 0)
    assert lease.retained_count == 2
    assert lease.oldest_row_identity == 1


def test_inverse_root_ownership_tracks_retire_hydration_and_cleanup():
    api = _api()
    authority, lease = _lease(rows=2)
    axis = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    lease.retain(
        _record(0, axis),
        grant_id=lease.grant_id,
        generation=lease.generation,
    )

    tokens = []
    worker = threading.Thread(
        target=lambda: tokens.append(lease.issue_hydration_token(1)),
    )
    worker.start()
    worker.join(timeout=2)
    assert not worker.is_alive() and len(tokens) == 1
    lease.complete_hydration(tokens[0], _record(1, axis))
    assert lease.retained_count == 2
    assert lease.contains(0) and lease.contains(1)
    assert lease.oldest_row_identity == 0

    assert lease.retire(
        0, grant_id=lease.grant_id, generation=lease.generation,
    )
    assert not lease.contains(0)
    assert lease.contains(1)
    assert lease.retained_count == 1
    assert lease.oldest_row_identity == 1

    receipt = lease.release(
        reason="close", hooks=api["Light1DCleanupHooks"](),
    )
    assert receipt.released_bytes == lease.reserved_ndarray_bytes
    assert lease.retained_count == 0
    assert lease.oldest_row_identity is None
    assert not lease.contains(1)
    assert lease._root_owners == {}
    assert authority.snapshot().reserved_bytes == 0


def test_hydration_abandon_and_fenced_completion_leave_root_index_unchanged():
    api = _api()
    _authority, lease = _lease(rows=2)
    axis = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    lease.retain(
        _record(0, axis),
        grant_id=lease.grant_id,
        generation=lease.generation,
    )

    def index_state():
        return {
            root_id: (id(entry.root), entry.owner_key, frozenset(entry.rows))
            for root_id, entry in lease._root_owners.items()
        }

    expected = index_state()
    tokens = []
    worker = threading.Thread(
        target=lambda: tokens.append(lease.issue_hydration_token(1)),
    )
    worker.start()
    worker.join(timeout=2)
    lease.abandon_hydration(tokens.pop())
    assert index_state() == expected

    worker = threading.Thread(
        target=lambda: tokens.append(lease.issue_hydration_token(2)),
    )
    worker.start()
    worker.join(timeout=2)
    lease.fence()
    with pytest.raises(api["Light1DStaleGeneration"]):
        lease.complete_hydration(tokens.pop(), _record(2, axis))
    assert index_state() == expected


def test_cleanup_retry_keeps_partial_root_index_exact_then_empties_it(
    monkeypatch,
):
    api = _api()
    authority, lease = _lease(rows=2)
    axis = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    for label in (0, 1):
        lease.retain(
            _record(label, axis),
            grant_id=lease.grant_id,
            generation=lease.generation,
        )
    first_unique_ids = {
        id(root) for owner_key, root in lease._roots[0].items()
        if not lease.layout._groups[owner_key].shared
    }
    original = api["Light1DRetentionLease"]._retire_row
    calls = 0

    def fail_second(owner, key):
        nonlocal calls
        if owner is lease:
            calls += 1
            if calls == 2:
                raise OSError("injected partial clear")
        return original(owner, key)

    monkeypatch.setattr(
        api["Light1DRetentionLease"], "_retire_row", fail_second,
    )
    with pytest.raises(api["Light1DCleanupPending"]) as pending:
        lease.release(reason="close", hooks=api["Light1DCleanupHooks"]())
    assert not lease.contains(0) and lease.contains(1)
    assert not (first_unique_ids & lease._root_owners.keys())
    assert any(entry.rows == {1} for entry in lease._root_owners.values())

    monkeypatch.setattr(api["Light1DRetentionLease"], "_retire_row", original)
    receipt = lease.retry_cleanup(pending.value.token)
    assert receipt.released_bytes == lease.reserved_ndarray_bytes
    assert lease._root_owners == {}
    assert authority.snapshot().reserved_bytes == 0


def test_cleanup_receipt_allocation_failure_keeps_shared_index_retryable():
    api = _api()

    class FailSecondAppend(deque):
        def __init__(self, values=()):
            super().__init__(values)
            self.calls = 0

        def append(self, value):
            self.calls += 1
            if self.calls == 2:
                raise MemoryError("injected shared receipt allocation")
            return super().append(value)

    authority, lease = _lease(rows=1)
    axis = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    lease.retain(
        _record(0, axis),
        grant_id=lease.grant_id,
        generation=lease.generation,
    )
    shared_root = lease._shared_roots["q-axis"]
    lease._retired_payload_groups = FailSecondAppend(
        lease._retired_payload_groups,
    )

    with pytest.raises(api["Light1DCleanupPending"]) as pending:
        lease.release(reason="close", hooks=api["Light1DCleanupHooks"]())
    shared_entry = lease._root_owners[id(shared_root)]
    assert shared_entry.root is shared_root
    assert shared_entry.rows == set()
    assert lease._shared_roots["q-axis"] is shared_root

    retry_token = pending.value.token
    del pending, shared_entry, shared_root
    gc.collect()
    receipt = lease.retry_cleanup(retry_token)
    assert receipt.released_bytes == lease.reserved_ndarray_bytes
    assert lease._root_owners == {}
    assert authority.snapshot().reserved_bytes == 0


def test_more_than_512_light_rows_stay_resident_without_hydration():
    api = _api()
    Buffer, Mode, Layout = (
        api["Light1DBufferLayout"], api["Light1DModeLayout"],
        api["Light1DLayout"],
    )
    layout = Layout(
        modes=(Mode("raw", Buffer(1, 8, "axis", "<f8", shared=True),
                    Buffer(1, 8, "intensity", "<f8")),),
        active_mode="raw",
    )
    authority = api["SessionResourceAuthority"](capacity_bytes=20_000)
    lease = api["acquire_light_1d_retention"](
        authority, owner="browse", generation=1, layout=layout,
        requested_rows=513, compatibility_byte_ceiling=20_000,
        gui_thread_id=threading.get_ident(),
    )
    axis = np.asarray([1.0], dtype=np.float64)
    Data, Record = api["Light1DModeData"], api["Light1DRecord"]
    for label in range(513):
        record = Record(label, 1, "raw", {
            "raw": Data(axis, np.asarray([label], dtype=np.float64), None),
        })
        lease.retain(record, grant_id=lease.grant_id, generation=1)
    assert lease.row_cap >= 513
    for label in (0, 256, 512):
        with lease.get(label) as shell:
            assert shell.row_identity == label
    assert lease.hydration_authorizations == 0


def test_shell_borrows_private_canonical_record_and_reuses_equal_shared_axis():
    api = _api()
    _authority, lease = _lease(rows=2)
    axis = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    record = _record(0, axis)
    lease.retain(record, grant_id=lease.grant_id, generation=lease.generation)
    shell_record = lease.borrow(0)
    assert isinstance(shell_record, api["Light1DBorrow"])
    assert shell_record.row_identity == record.row_identity
    canonical_axis = shell_record.modes["raw"].coordinate
    assert canonical_axis is not axis
    np.testing.assert_array_equal(canonical_axis, axis)
    assert lease.unique_owned_ndarray_bytes == (
        lease.layout.shared_bytes + lease.layout.per_row_unique_ndarray_bytes
    )
    shell_record.close()

    copied_axis = axis.copy()
    lease.retain(
        _record(1, copied_axis),
        grant_id=lease.grant_id,
        generation=lease.generation,
    )
    with lease.borrow(1) as second:
        assert second.modes["raw"].coordinate is canonical_axis
    assert lease.unique_owned_ndarray_bytes == (
        lease.layout.shared_bytes
        + (2 * lease.layout.per_row_unique_ndarray_bytes)
    )

    with pytest.raises(ValueError, match="changed values"):
        lease.retain(
            _record(1, copied_axis + 1),
            grant_id=lease.grant_id,
            generation=lease.generation,
        )


def test_positive_release_waits_for_every_tracked_shell_borrow():
    api = _api()
    authority, lease = _lease(rows=1)
    axis = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    lease.retain(
        _record(0, axis), grant_id=lease.grant_id, generation=lease.generation,
    )
    shell = lease.borrow(0)
    with pytest.raises(api["Light1DCleanupPending"]) as caught:
        lease.release(reason="close", hooks=api["Light1DCleanupHooks"]())
    assert authority.snapshot().categories["light_1d"] == (
        lease.reserved_ndarray_bytes
    )
    shell.close()
    receipt = lease.retry_cleanup(caught.value.token)
    assert receipt.released_bytes == lease.reserved_ndarray_bytes
    assert authority.snapshot().reserved_bytes == 0


@pytest.mark.parametrize(
    ("target", "field", "replacement"),
    (
        ("authority", "capacity_bytes", 10_000_000),
        ("authority", "parent_allocation", object()),
        ("lease", "authority", object()),
        ("lease", "grant_id", "forged"),
        ("lease", "owner", "forged"),
        ("lease", "generation", 999),
        ("lease", "layout", object()),
        ("lease", "requested_rows", 999),
        ("lease", "row_cap", 999),
        ("lease", "reserved_ndarray_bytes", 10_000_000),
        ("lease", "shared_bytes", 0),
        ("lease", "per_row_unique_ndarray_bytes", 1),
    ),
)
def test_public_authority_and_lease_grant_fields_are_immutable(
    target, field, replacement,
):
    authority, lease = _lease(rows=1)
    owner = authority if target == "authority" else lease
    before = getattr(owner, field)
    with pytest.raises(AttributeError):
        setattr(owner, field, replacement)
    assert getattr(owner, field) is before or getattr(owner, field) == before


def test_borrow_close_detaches_record_before_unregistering_for_release(monkeypatch):
    api = _api()
    authority, lease = _lease(rows=1)
    axis = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    lease.retain(
        _record(0, axis), grant_id=lease.grant_id, generation=lease.generation,
    )
    shell = lease.borrow(0)
    detached = threading.Event()
    proceed = threading.Event()
    original = api["Light1DRetentionLease"]._close_borrow

    def pause_after_unregister(self, ordinal, handle):
        original(self, ordinal, handle)
        detached.set()
        assert proceed.wait(2)

    monkeypatch.setattr(
        api["Light1DRetentionLease"], "_close_borrow", pause_after_unregister,
    )
    closer = threading.Thread(target=shell.close)
    closer.start()
    assert detached.wait(2)
    try:
        receipt = lease.release(
            reason="close", hooks=api["Light1DCleanupHooks"](),
        )
        assert receipt.released_bytes == lease.reserved_ndarray_bytes
        with pytest.raises(api["Light1DStaleGeneration"]):
            _ = shell.modes
    finally:
        proceed.set()
        closer.join(timeout=2)
    assert not closer.is_alive()
    assert authority.snapshot().reserved_bytes == 0


def test_release_waits_for_extracted_canonical_array_and_it_stays_immutable():
    api = _api()
    authority, lease = _lease(rows=1)
    axis = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    lease.retain(
        _record(0, axis), grant_id=lease.grant_id, generation=lease.generation,
    )
    shell = lease.borrow(0)
    extracted = shell.modes["raw"].intensity
    extracted_ref = weakref.ref(extracted)
    backing = extracted.base
    backing_ref = weakref.ref(backing)
    derived_backing = backing.toreadonly()
    derived_ref = weakref.ref(derived_backing)
    payload = backing.obj

    assert type(backing) is memoryview and backing.readonly
    assert type(payload) is bytes
    with pytest.raises(ValueError):
        extracted.setflags(write=True)
    shell.close()
    with pytest.raises(api["Light1DCleanupPending"]) as caught:
        lease.release(reason="close", hooks=api["Light1DCleanupHooks"]())
    assert caught.value.receipt.failed_step == "verify"
    assert authority.snapshot().reserved_bytes == lease.reserved_ndarray_bytes
    assert extracted_ref() is extracted
    np.testing.assert_array_equal(extracted, np.full(4, 0, dtype=np.float64))

    del extracted
    gc.collect()
    assert extracted_ref() is None
    assert backing_ref() is backing
    with pytest.raises(api["Light1DCleanupPending"]):
        lease.retry_cleanup(caught.value.token)

    del backing
    gc.collect()
    assert backing_ref() is None
    assert derived_ref() is derived_backing
    with pytest.raises(api["Light1DCleanupPending"]):
        lease.retry_cleanup(caught.value.token)

    del derived_backing
    gc.collect()
    assert derived_ref() is None
    with pytest.raises(api["Light1DCleanupPending"]):
        lease.retry_cleanup(caught.value.token)

    del payload
    gc.collect()
    receipt = lease.retry_cleanup(caught.value.token)
    assert receipt.released_bytes == lease.reserved_ndarray_bytes
    assert authority.snapshot().reserved_bytes == 0


def test_evicted_extracted_array_keeps_its_granted_slot_until_detached():
    api = _api()
    authority, lease = _lease(rows=1)
    axis = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    lease.retain(
        _record(0, axis), grant_id=lease.grant_id, generation=lease.generation,
    )
    shell = lease.borrow(0)
    extracted = shell.modes["raw"].intensity
    shell.close()

    with pytest.raises(api["Light1DUnavailable"], match="alias"):
        lease.retain(
            _record(1, axis),
            grant_id=lease.grant_id,
            generation=lease.generation,
        )
    assert lease.keys() == ()
    assert lease.evicted == (0,)
    assert authority.snapshot().reserved_bytes == lease.reserved_ndarray_bytes
    np.testing.assert_array_equal(extracted, np.full(4, 0, dtype=np.float64))

    del extracted
    gc.collect()
    lease.retain(
        _record(1, axis), grant_id=lease.grant_id, generation=lease.generation,
    )
    assert lease.keys() == (1,)
    assert lease.unique_owned_ndarray_bytes == (
        lease.layout.shared_bytes + lease.layout.per_row_unique_ndarray_bytes
    )


def test_same_row_replacement_cannot_bypass_retired_alias_capacity():
    api = _api()
    authority, lease = _lease(rows=1)
    axis = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    lease.retain(
        _record(0, axis), grant_id=lease.grant_id, generation=lease.generation,
    )
    shell = lease.borrow(0)
    extracted = shell.modes["raw"].intensity
    shell.close()

    with pytest.raises(api["Light1DUnavailable"], match="alias"):
        lease.retain(
            _record(0, axis),
            grant_id=lease.grant_id,
            generation=lease.generation,
        )
    assert lease.keys() == ()
    assert lease.unique_owned_ndarray_bytes == lease.reserved_ndarray_bytes
    assert authority.snapshot().reserved_bytes == lease.reserved_ndarray_bytes

    del extracted
    gc.collect()
    lease.retain(
        _record(0, axis), grant_id=lease.grant_id, generation=lease.generation,
    )
    # A replacement with no escaped alias retires and frees the exact prior
    # payloads before installing the new row.
    lease.retain(
        _record(0, axis), grant_id=lease.grant_id, generation=lease.generation,
    )
    assert lease.keys() == (0,)
    assert lease.unique_owned_ndarray_bytes == lease.reserved_ndarray_bytes


class _CleanupHookPayloadOwner:
    def __init__(self, payload):
        self.payload = payload

    def verify(self):
        assert self.payload is not None


def test_positive_release_severs_cleanup_hook_callback_ownership():
    api = _api()
    authority, lease = _lease(rows=1)
    hidden = np.ones(1_000_000, dtype=np.float64)
    hidden_ref = weakref.ref(hidden)
    owner = _CleanupHookPayloadOwner(hidden)
    owner_ref = weakref.ref(owner)
    hooks = api["Light1DCleanupHooks"](verify=owner.verify)

    lease.release(reason="close", hooks=hooks)
    assert authority.snapshot().reserved_bytes == 0
    assert lease._cleanup_hooks is None
    del hooks, owner, hidden
    gc.collect()
    assert owner_ref() is None
    assert hidden_ref() is None


def test_discarded_borrow_handle_stays_owned_until_explicit_detach():
    api = _api()
    authority, lease = _lease(rows=1)
    axis = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    lease.retain(
        _record(0, axis), grant_id=lease.grant_id, generation=lease.generation,
    )
    shell = lease.borrow(0)
    shell_ref = weakref.ref(shell)
    extracted = shell.modes["raw"].intensity
    del shell
    gc.collect()
    assert shell_ref() is not None
    assert lease.active_borrow_count == 1
    with pytest.raises(api["Light1DCleanupPending"]) as caught:
        lease.release(reason="close", hooks=api["Light1DCleanupHooks"]())
    assert authority.snapshot().reserved_bytes == lease.reserved_ndarray_bytes
    shell_ref().close()
    del extracted
    receipt = lease.retry_cleanup(caught.value.token)
    assert receipt.released_bytes == lease.reserved_ndarray_bytes


def test_retain_owns_private_immutable_roots_not_mutable_source_aliases():
    api = _api()
    authority, lease = _lease(rows=1)
    axis_root = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    axis_view = axis_root.view()
    axis_alias = axis_root.view()
    record = _record(0, axis_view)
    intensity = record.modes["raw"].intensity
    intensity_alias = intensity.view()
    lease.retain(record, grant_id=lease.grant_id, generation=lease.generation)
    with lease.borrow(0) as shell:
        canonical_axis = shell.modes["raw"].coordinate
        canonical_intensity = shell.modes["raw"].intensity
        assert canonical_axis is not axis_view
        assert canonical_intensity is not intensity
        assert not canonical_axis.flags.writeable
        assert not canonical_intensity.flags.writeable
        expected_axis = canonical_axis.copy()
        expected_intensity = canonical_intensity.copy()

        assert axis_root.flags.writeable and axis_alias.flags.writeable
        assert intensity.flags.writeable and intensity_alias.flags.writeable
        axis_alias[:] = -1
        intensity_alias[:] = -2
        np.testing.assert_array_equal(canonical_axis, expected_axis)
        np.testing.assert_array_equal(canonical_intensity, expected_intensity)

        canonical_axis_ref = weakref.ref(canonical_axis)
        canonical_intensity_ref = weakref.ref(canonical_intensity)

    del canonical_axis, canonical_intensity, expected_axis, expected_intensity
    receipt = lease.release(
        reason="close", hooks=api["Light1DCleanupHooks"](),
    )
    assert receipt.released_bytes == lease.reserved_ndarray_bytes
    gc.collect()
    assert canonical_axis_ref() is None
    assert canonical_intensity_ref() is None
    assert authority.snapshot().reserved_bytes == 0
    assert record.modes["raw"].intensity is intensity


@pytest.mark.parametrize("fault", ("length", "dtype", "missing-mode", "sigma"))
def test_record_layout_shape_dtype_and_sigma_mismatch_is_refused(fault):
    api = _api()
    _authority, lease = _lease(rows=1)
    axis = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    record = _record(0, axis)
    modes = dict(record.modes)
    raw = modes["raw"]
    if fault == "length":
        modes["raw"] = replace(raw, intensity=np.ones(3, dtype=np.float64))
    elif fault == "dtype":
        modes["raw"] = replace(raw, intensity=np.ones(4, dtype=np.float32))
    elif fault == "missing-mode":
        modes.pop("bg-subtracted")
    else:
        modes["raw"] = replace(raw, uncertainty=None)
    malformed = replace(record, modes=modes)
    with pytest.raises(ValueError, match="layout"):
        lease.retain(
            malformed, grant_id=lease.grant_id, generation=lease.generation,
        )
    assert lease.keys() == () and lease.owned_buffer_ids == frozenset()


def test_provenance_cannot_hide_an_unaccounted_array_owner():
    api = _api()
    axis = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    with pytest.raises(ValueError, match="unaccounted array buffer"):
        api["Light1DRecord"](
            row_identity=0,
            generation=3,
            active_mode="raw",
            modes=_record(0, axis).modes,
            provenance={"hidden": np.ones(64, dtype=np.float64)},
        )


def test_provenance_is_recursively_frozen_against_late_array_injection():
    api = _api()
    nested = {"members": ["scan.nxs"]}
    record = api["Light1DRecord"](
        0,
        3,
        "raw",
        {"raw": api["Light1DModeData"](
            np.arange(4, dtype=np.float64),
            np.arange(4, dtype=np.float64),
        )},
        provenance={"source": nested},
    )
    nested["members"].append(np.ones(64, dtype=np.float64))
    assert record.provenance["source"]["members"] == ("scan.nxs",)


class _PayloadString(str):
    pass


@pytest.mark.parametrize(
    "slot", ("provenance-value", "provenance-key", "owner-key", "dtype"),
)
def test_closed_metadata_grammar_refuses_payload_bearing_scalar_subclasses(slot):
    api = _api()
    hidden = np.ones(1_000_000, dtype=np.float64)
    hidden_ref = weakref.ref(hidden)
    payload = _PayloadString("payload")
    payload.hidden = hidden

    if slot.startswith("provenance"):
        axis = np.linspace(0.0, 1.0, 4, dtype=np.float64)
        template = _record(0, axis)
        provenance = (
            {"payload": payload}
            if slot == "provenance-value"
            else {payload: "value"}
        )
        with pytest.raises(TypeError):
            api["Light1DRecord"](
                row_identity=0,
                generation=3,
                active_mode="bg-subtracted",
                modes=template.modes,
                provenance=provenance,
            )
        del provenance, template, axis
    else:
        owner_key = payload if slot == "owner-key" else "axis"
        dtype = payload if slot == "dtype" else "<f8"
        with pytest.raises(TypeError):
            api["Light1DBufferLayout"](4, 8, owner_key, dtype)
        del owner_key, dtype

    del payload, hidden
    gc.collect()
    assert hidden_ref() is None


class _PayloadIdentity:
    def __init__(self):
        self.payload = np.ones(1_000_000, dtype=np.float64)

    __hash__ = object.__hash__


@pytest.mark.parametrize(
    "slot", ("row", "layout-mode", "layout-active", "record-mode", "token"),
)
def test_light_identity_grammar_refuses_payload_bearing_hashables(slot):
    api = _api()
    hidden = _PayloadIdentity()
    hidden_ref = weakref.ref(hidden)
    Buffer, Mode, Layout = (
        api["Light1DBufferLayout"], api["Light1DModeLayout"],
        api["Light1DLayout"],
    )
    axis_spec = Buffer(4, 8, "axis", "<f8", shared=True)
    intensity_spec = Buffer(4, 8, "intensity", "<f8")
    axis = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    data = api["Light1DModeData"](
        coordinate=axis, intensity=np.ones(4, dtype=np.float64),
    )

    def construct():
        if slot == "row":
            return replace(_record(0, axis), row_identity=hidden)
        if slot == "layout-mode":
            return Mode(hidden, axis_spec, intensity_spec)
        if slot == "layout-active":
            declared = Mode("raw", axis_spec, intensity_spec)
            return Layout((declared,), hidden)
        if slot == "record-mode":
            return api["Light1DRecord"](
                row_identity=0,
                generation=3,
                active_mode=hidden,
                modes={hidden: data},
            )
        return api["Light1DHydrationToken"]("grant", 3, hidden, 1)

    with pytest.raises(TypeError, match="identity"):
        construct()
    del hidden
    gc.collect()
    assert hidden_ref() is None


def test_exact_dynamic_frame_and_result_mode_identities_remain_supported():
    api = _api()
    from xrd_tools.session import DynamicFrameIdentity, ResultMode

    Buffer, Mode, Layout = (
        api["Light1DBufferLayout"], api["Light1DModeLayout"],
        api["Light1DLayout"],
    )
    result_mode = ResultMode.one_d("raw")
    axis_spec = Buffer(4, 8, "axis", "<f8", shared=True)
    layout = Layout(
        (Mode(result_mode, axis_spec, Buffer(4, 8, "intensity", "<f8")),),
        result_mode,
    )
    identity = DynamicFrameIdentity(("master.nxs", "detector"), 7)
    axis = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    record = api["Light1DRecord"](
        row_identity=identity,
        generation=3,
        active_mode=result_mode,
        modes={result_mode: api["Light1DModeData"](
            coordinate=axis,
            intensity=np.ones(4, dtype=np.float64),
        )},
    )
    authority = api["SessionResourceAuthority"](capacity_bytes=1_000)
    lease = api["acquire_light_1d_retention"](
        authority,
        owner="dynamic-light",
        generation=3,
        layout=layout,
        requested_rows=1,
        compatibility_byte_ceiling=1_000,
        gui_thread_id=threading.get_ident(),
    )
    lease.retain(record, grant_id=lease.grant_id, generation=3)
    assert lease.keys() == (identity,)


def test_eviction_diagnostics_are_bounded_by_the_granted_row_cap():
    _authority, lease = _lease(rows=2)
    axis = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    for label in range(20):
        lease.retain(
            _record(label, axis),
            grant_id=lease.grant_id,
            generation=lease.generation,
        )
    assert lease.keys() == (18, 19)
    assert lease.evicted == (16, 17)
    assert len(lease.evicted) <= lease.row_cap


def test_positive_release_clears_eviction_identity_ownership():
    api = _api()
    authority, lease = _lease(rows=1)
    axis = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    for label in (0, 1):
        lease.retain(
            _record(label, axis),
            grant_id=lease.grant_id,
            generation=lease.generation,
        )
    assert lease.evicted == (0,)
    lease.release(reason="close", hooks=api["Light1DCleanupHooks"]())
    assert lease.evicted == ()
    assert authority.snapshot().reserved_bytes == 0


def test_heavy_and_thumbnail_policy_caps_are_unchanged_by_light_eviction():
    api = _api()
    from xrd_tools.session import SessionResourceRequirements, resolve_session_policy

    requirements = SessionResourceRequirements(
        height=4, width=4, native_itemsize=2,
        modes_1d=2, npt_1d=4, sigma_1d=1,
        modes_2d=1, npt_rad=4, npt_azim=4,
    )
    allocation = resolve_session_policy(
        requirements,
        envelope_bytes=4 * 1024 ** 3,
        requests={"record_items": 10, "publication_items": 10},
        env={},
    ).allocation
    immutable_caps = (
        allocation.record_heavy_items,
        allocation.publication_heavy_items,
        allocation.thumbnail_items,
    )
    assert immutable_caps[2] == 512
    authority = api["SessionResourceAuthority"].from_allocation(allocation)
    layout = _layout()
    lease = api["acquire_light_1d_retention"](
        authority,
        owner="browse-light",
        generation=1,
        layout=layout,
        requested_rows=2,
        compatibility_byte_ceiling=(
            layout.shared_bytes + 2 * layout.per_row_unique_ndarray_bytes
        ),
        gui_thread_id=threading.get_ident(),
    )
    before = authority.snapshot()
    axis = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    for label in range(4):
        lease.retain(
            _record(label, axis, generation=1),
            grant_id=lease.grant_id,
            generation=1,
        )
    after = authority.snapshot()
    assert dict(before.committed_bytes) == dict(after.committed_bytes)
    assert dict(after.committed_bytes) == dict(allocation.categories)
    assert after.categories["light_1d"] == lease.reserved_ndarray_bytes
    assert lease.keys() == (2, 3)
    receipt = lease.release(
        reason="close", hooks=api["Light1DCleanupHooks"](),
    )
    assert receipt.released_bytes == lease.reserved_ndarray_bytes
    assert (
        allocation.record_heavy_items,
        allocation.publication_heavy_items,
        allocation.thumbnail_items,
    ) == immutable_caps
    assert dict(authority.snapshot().categories) == dict(allocation.categories)


def test_generation_fence_refuses_new_hydration_completions():
    api = _api()
    _authority, lease = _lease(rows=2)
    axis = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    first = _record(0, axis)
    lease.retain(first, grant_id=lease.grant_id, generation=3)
    tokens, refusals = [], []

    def issue():
        try:
            lease.issue_hydration_token(0)
        except BaseException as exc:
            refusals.append(exc)
        tokens.extend((
            lease.issue_hydration_token(98),
            lease.issue_hydration_token(99),
        ))

    worker = threading.Thread(target=issue)
    worker.start()
    worker.join()
    assert len(refusals) == 1
    assert isinstance(refusals[0], api["Light1DUnavailable"])
    first_unseen, second_unseen = tokens
    lease.fence()

    late_first = _record(98, axis)
    late_second = _record(99, axis)
    first_ref = weakref.ref(late_first.modes["raw"].intensity)
    second_ref = weakref.ref(late_second.modes["raw"].intensity)
    with pytest.raises(api["Light1DStaleGeneration"]):
        lease.complete_hydration(first_unseen, late_first)
    with pytest.raises(api["Light1DStaleGeneration"]):
        lease.complete_hydration(second_unseen, late_second)
    # The second authorization reserved the final granted slot before worker
    # allocation, so the old resident row was deterministically evicted.
    assert lease.keys() == ()
    assert lease.evicted == (0,)
    assert id(first_ref()) not in lease.owned_buffer_ids
    assert id(second_ref()) not in lease.owned_buffer_ids


def test_gui_thread_hydration_is_refused_before_io_and_worker_is_authorized():
    api = _api()
    _authority, lease = _lease(rows=1)
    with pytest.raises(api["GUIThreadHydrationRefused"]):
        lease.issue_hydration_token("miss")

    tokens = []
    thread = threading.Thread(
        target=lambda: tokens.append(lease.issue_hydration_token("miss")),
    )
    thread.start()
    thread.join()
    assert len(tokens) == 1
    assert tokens[0].grant_id == lease.grant_id
    assert lease.hydration_authorizations == 1


def test_pending_hydration_tokens_are_one_per_row_and_bounded_by_row_cap():
    api = _api()
    _authority, lease = _lease(rows=1)
    outcomes = []

    def issue():
        first = lease.issue_hydration_token("row")
        outcomes.append(first)
        for row in ("row", "other"):
            try:
                lease.issue_hydration_token(row)
            except BaseException as exc:
                outcomes.append(exc)

    thread = threading.Thread(target=issue)
    thread.start()
    thread.join()
    assert lease.pending_hydration_count == 1
    assert all(isinstance(value, api["Light1DUnavailable"])
               for value in outcomes[1:])
    assert "already owns" in str(outcomes[1])
    assert "shared" in str(outcomes[2])


@pytest.mark.parametrize("stage", ("cancel", "drain", "clear", "detach", "verify", "release"))
def test_partial_release_is_cleanup_pending_and_exact_retry_is_idempotent(stage):
    api = _api()
    authority, lease = _lease(rows=1)
    axis = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    lease.retain(_record(0, axis), grant_id=lease.grant_id, generation=3)
    failed = {stage}
    events = []

    def hook(name):
        def run():
            events.append(name)
            if name in failed:
                failed.remove(name)
                raise OSError(f"{name} fault")
        return run

    hooks = api["Light1DCleanupHooks"](
        cancel=hook("cancel"), drain=hook("drain"), clear=hook("clear"),
        detach=hook("detach"), verify=hook("verify"), release=hook("release"),
    )
    with pytest.raises(api["Light1DCleanupPending"]) as caught:
        lease.release(reason="replacement", hooks=hooks)
    token = caught.value.token
    pending_receipt = caught.value.receipt
    assert pending_receipt is lease.cleanup_receipt
    assert pending_receipt.retry_token is token
    assert pending_receipt.state == "cleanup-pending"
    assert lease.state.value == "cleanup-pending"
    assert authority.snapshot().categories["light_1d"] == lease.reserved_ndarray_bytes

    with pytest.raises(RuntimeError, match="hooks are frozen"):
        lease.retry_cleanup(token, hooks=api["Light1DCleanupHooks"]())

    receipt = lease.retry_cleanup(token, hooks=hooks)
    assert receipt.reason == "replacement"
    assert lease.state.value == "released"
    assert authority.snapshot().reserved_bytes == sum(
        authority.snapshot().committed_bytes.values()
    )
    assert lease.keys() == () and lease.owned_buffer_ids == frozenset()
    assert lease.retry_cleanup(token, hooks=hooks) is receipt
    assert lease.release(reason="replacement", hooks=hooks) is receipt


def test_replacement_cancel_and_close_each_return_one_positive_release_receipt():
    api = _api()
    for reason in ("replacement", "cancellation", "close"):
        authority, lease = _lease(rows=1)
        receipt = lease.release(reason=reason, hooks=api["Light1DCleanupHooks"]())
        assert receipt.grant_id == lease.grant_id
        assert receipt.released_bytes == lease.reserved_ndarray_bytes
        assert authority.snapshot().reserved_bytes == 0
        assert lease.release(reason=reason) is receipt


def test_acquire_uses_explicit_authority_and_never_reads_physical_ram(monkeypatch):
    import xrd_tools.session.policy as policy

    monkeypatch.setattr(
        policy, "default_envelope_bytes",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("RAM read")),
    )
    authority, lease = _lease(rows=1, capacity=512)
    assert lease.authority is authority
    assert lease.reserved_ndarray_bytes <= authority.capacity_bytes
    assert authority.snapshot().reservation_count == 1


def test_authority_can_be_derived_from_one_exact_session_allocation():
    api = _api()
    from xrd_tools.session import SessionResourceRequirements, resolve_session_policy

    requirements = SessionResourceRequirements(
        height=2, width=2, native_itemsize=2,
        modes_1d=1, npt_1d=4,
    )
    allocation = resolve_session_policy(
        requirements, envelope_bytes=2 * 1024 ** 3, env={},
    ).allocation
    authority = api["SessionResourceAuthority"].from_allocation(allocation)
    snap = authority.snapshot()
    assert authority.parent_allocation is allocation
    assert snap.capacity_bytes == (
        allocation.envelope_bytes + allocation.oversize_excess_bytes
    )
    assert dict(snap.committed_bytes) == dict(allocation.categories)
    assert sum(snap.committed_bytes.values()) == allocation.assigned_bytes


def test_allocation_factory_accepts_exact_oversize_without_inventing_headroom():
    api = _api()
    from xrd_tools.session import SessionResourceRequirements, resolve_session_policy
    import xrd_tools.session.policy as policy_module

    requirements = SessionResourceRequirements(
        height=1, width=1, native_itemsize=1,
    )
    allocation = resolve_session_policy(
        requirements,
        envelope_bytes=policy_module.floor_bytes(requirements),
        env={},
    ).allocation
    assert allocation.oversize_excess_bytes > 0
    authority = api["SessionResourceAuthority"].from_allocation(allocation)
    assert authority.capacity_bytes == (
        allocation.envelope_bytes + allocation.oversize_excess_bytes
    )
    assert authority.snapshot().available_bytes == 0


def test_released_owner_requires_a_strictly_newer_generation():
    api = _api()
    authority, lease = _lease(rows=1, generation=3)
    lease.release(reason="replacement", hooks=api["Light1DCleanupHooks"]())
    for generation in (2, 3):
        with pytest.raises(ValueError, match="strictly newer"):
            api["acquire_light_1d_retention"](
                authority,
                owner="browse:scan",
                generation=generation,
                layout=_layout(),
                requested_rows=1,
                compatibility_byte_ceiling=1_000,
                gui_thread_id=threading.get_ident(),
            )
    replacement = api["acquire_light_1d_retention"](
        authority,
        owner="browse:scan",
        generation=4,
        layout=_layout(),
        requested_rows=1,
        compatibility_byte_ceiling=1_000,
        gui_thread_id=threading.get_ident(),
    )
    assert replacement.generation == 4


def test_cleanup_drain_does_not_hold_the_lease_state_lock():
    api = _api()
    _authority, lease = _lease(rows=1)
    axis = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    token_box = []
    issuer = threading.Thread(
        target=lambda: token_box.append(lease.issue_hydration_token(99)),
    )
    issuer.start()
    issuer.join()
    go = threading.Event()
    worker_done = threading.Event()

    def hydration_worker():
        go.wait()
        try:
            lease.complete_hydration(token_box[0], _record(99, axis))
        except api["Light1DStaleGeneration"]:
            pass
        finally:
            worker_done.set()

    worker = threading.Thread(target=hydration_worker)
    worker.start()

    def drain():
        go.set()
        assert worker_done.wait(2), "hydration worker deadlocked on lease state lock"
        worker.join(timeout=2)

    receipt = lease.release(
        reason="close",
        hooks=api["Light1DCleanupHooks"](drain=drain),
    )
    assert receipt.released_bytes == lease.reserved_ndarray_bytes
    assert not worker.is_alive()


def test_cleanup_cannot_discard_an_unretired_hydration_authorization():
    api = _api()
    authority, lease = _lease(rows=1)
    tokens = []
    issuer = threading.Thread(
        target=lambda: tokens.append(lease.issue_hydration_token(99)),
    )
    issuer.start()
    issuer.join()
    with pytest.raises(api["Light1DCleanupPending"]) as caught:
        lease.release(reason="close", hooks=api["Light1DCleanupHooks"]())
    assert caught.value.receipt.failed_step == "clear"
    assert lease.pending_hydration_count == 1
    assert authority.snapshot().reserved_bytes == lease.reserved_ndarray_bytes
    lease.abandon_hydration(tokens[0])
    receipt = lease.retry_cleanup(caught.value.token)
    assert receipt.released_bytes == lease.reserved_ndarray_bytes
    assert authority.snapshot().reserved_bytes == 0


def test_invalid_gui_thread_identity_cannot_leak_a_reservation():
    api = _api()
    authority = api["SessionResourceAuthority"](capacity_bytes=1024)
    with pytest.raises(ValueError, match="gui_thread_id"):
        api["acquire_light_1d_retention"](
            authority,
            owner="invalid",
            generation=1,
            layout=_layout(),
            requested_rows=1,
            compatibility_byte_ceiling=1024,
            gui_thread_id=object(),
        )
    assert authority.snapshot().reservation_count == 0


def test_new_h10_modules_and_lazy_exports_are_import_pure_in_a_subprocess():
    src = Path(__file__).resolve().parents[2] / "src"
    light_script = f"""
import json
import sys
sys.path.insert(0, {str(src)!r})
import xrd_tools.session.light_1d_retention
import xrd_tools.session as session
session.Light1DRetentionLease
forbidden = ("numpy", "h5py", "xdart", "PyQt5", "PyQt6", "PySide2", "PySide6", "qtpy")
print(json.dumps(sorted(name for name in forbidden if name in sys.modules)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", light_script],
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(completed.stdout) == []
    dynamic_script = f"""
import json
import sys
sys.path.insert(0, {str(src)!r})
import xrd_tools.session.dynamic_accounting
import xrd_tools.session as session
session.DynamicRunAccounting
forbidden = ("h5py", "xdart", "PyQt5", "PyQt6", "PySide2", "PySide6", "qtpy")
print(json.dumps(sorted(name for name in forbidden if name in sys.modules)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", dynamic_script],
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(completed.stdout) == []


def test_unique_owners_refuse_shallow_and_deep_copy():
    from xrd_tools.session import (
        DynamicAccountingLimits,
        DynamicRunAccounting,
        ResultMode,
        StageLedger,
    )

    authority, lease = _lease(rows=1)
    mode = ResultMode.one_d()
    accounting = DynamicRunAccounting(
        StageLedger(
            required_modes=(mode,),
            targets_by_mode={mode: ("nexus:/tmp/copy-owner.nxs",)},
        ),
        run_generation=1,
        limits=DynamicAccountingLimits(1, 1, 1),
    )
    for owner in (authority, lease, accounting, accounting.writer_boundary):
        with pytest.raises(TypeError, match="unique"):
            copy.copy(owner)
        with pytest.raises(TypeError, match="unique"):
            copy.deepcopy(owner)


def test_dynamic_light_binding_requires_exact_active_owner_and_hooks():
    api = _api()
    from xrd_tools.session import (
        DynamicAccountingLimits,
        DynamicRunAccounting,
        ResultMode,
        StageLedger,
    )

    def accounting(generation):
        mode = ResultMode.one_d()
        return DynamicRunAccounting(
            StageLedger(
                required_modes=(mode,),
                targets_by_mode={mode: (f"nexus:/tmp/bind-{generation}.nxs",)},
            ),
            run_generation=generation,
            limits=DynamicAccountingLimits(1, 1, 1),
        )

    _authority, lease = _lease(rows=1, generation=3)
    owner = accounting(3)
    hooks = api["Light1DCleanupHooks"]()
    owner.bind_light_1d(lease, cleanup_hooks=hooks)
    owner.bind_light_1d(lease, cleanup_hooks=hooks)
    with pytest.raises(RuntimeError, match=r"light-1D binding cannot change"):
        owner.bind_light_1d(lease, cleanup_hooks=api["Light1DCleanupHooks"]())
    with pytest.raises(TypeError, match="exact light-1D lease"):
        accounting(3).bind_light_1d(object(), cleanup_hooks=hooks)
    with pytest.raises(ValueError, match="another generation"):
        accounting(4).bind_light_1d(lease, cleanup_hooks=hooks)

    _released_authority, released = _lease(
        rows=1, generation=8, capacity=2_000,
    )
    released.release(reason="close", hooks=api["Light1DCleanupHooks"]())
    with pytest.raises(RuntimeError, match="active light-1D lease"):
        accounting(8).bind_light_1d(released, cleanup_hooks=hooks)

    _late_authority, late = _lease(
        rows=1, generation=9, capacity=2_000,
    )
    stopped = accounting(9)
    stopped.stop()
    with pytest.raises(RuntimeError, match="non-active"):
        stopped.bind_light_1d(late, cleanup_hooks=hooks)


@pytest.mark.parametrize("failure", ("cleanup-pending", "unexpected"))
def test_stopped_terminal_cleanup_exact_retry_survives_the_failure_window(
    failure, monkeypatch,
):
    api = _api()
    from xrd_tools.session import (
        DynamicAccountingLimits,
        DynamicRunAccounting,
        DynamicRunState,
        ResultMode,
        StageLedger,
    )

    class LiveOwner:
        pass

    authority, lease = _lease(rows=1, generation=11)
    mode = ResultMode.one_d()
    accounting = DynamicRunAccounting(
        StageLedger(
            required_modes=(mode,),
            targets_by_mode={mode: ("nexus:/tmp/cleanup-retry.nxs",)},
        ),
        run_generation=11,
        limits=DynamicAccountingLimits(1, 1, 1),
    )
    owner, owner_token = LiveOwner(), object()
    accounting.writer_boundary.bind_live_session(owner, owner_token)

    if failure == "cleanup-pending":
        first = True

        def drain():
            nonlocal first
            if first:
                first = False
                raise OSError("drain once")

        hooks = api["Light1DCleanupHooks"](drain=drain)
        expected = api["Light1DCleanupPending"]
    else:
        hooks = api["Light1DCleanupHooks"]()
        release = type(lease).release
        first = True

        def fail_once(self, *args, **kwargs):
            nonlocal first
            if self is lease and first:
                first = False
                raise OSError("unexpected release fault")
            return release(self, *args, **kwargs)

        monkeypatch.setattr(type(lease), "release", fail_once)
        expected = OSError

    accounting.bind_light_1d(lease, cleanup_hooks=hooks)
    accounting.stop()
    seal = accounting.writer_boundary.prepare_session_finish(
        owner, owner_token, stopped=True,
    )
    with pytest.raises(expected):
        accounting.writer_boundary.session_stopped(
            owner, owner_token, seal, "operator Stop",
        )
    assert accounting._terminal_intent is DynamicRunState.STOPPED
    with pytest.raises(RuntimeError, match="terminal"):
        accounting.writer_boundary.bind_live_session(LiveOwner(), object())
    accounting.writer_boundary.session_stopped(
        owner, owner_token, seal, "operator Stop",
    )
    assert accounting.snapshot().state is DynamicRunState.STOPPED
    assert authority.snapshot().reserved_bytes == 0


def test_reused_source_buffer_is_copied_and_charged_for_each_canonical_row():
    _authority, lease = _lease(rows=3)
    axis = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    first = _record(0, axis)
    middle = _record(1, axis)
    lease.retain(first, grant_id=lease.grant_id, generation=lease.generation)
    lease.retain(middle, grant_id=lease.grant_id, generation=lease.generation)
    last = _record(2, axis)
    modes = dict(last.modes)
    modes["raw"] = replace(
        modes["raw"], intensity=first.modes["raw"].intensity,
    )
    lease.retain(
        replace(last, modes=modes),
        grant_id=lease.grant_id,
        generation=lease.generation,
    )
    assert lease.keys() == (0, 1, 2)
    with lease.borrow(0) as first_shell, lease.borrow(2) as last_shell:
        assert (
            first_shell.modes["raw"].intensity
            is not last_shell.modes["raw"].intensity
        )
    assert lease.unique_owned_ndarray_bytes == (
        lease.layout.shared_bytes
        + (3 * lease.layout.per_row_unique_ndarray_bytes)
    )


def test_layout_dtype_identity_rejects_same_itemsize_wrong_dtype():
    api = _api()
    Buffer, Mode, Layout = (
        api["Light1DBufferLayout"], api["Light1DModeLayout"],
        api["Light1DLayout"],
    )
    layout = Layout((Mode(
        "raw",
        Buffer(4, 8, "axis", dtype="<f8", shared=True),
        Buffer(4, 8, "intensity", dtype="<f8"),
    ),), "raw")
    authority = api["SessionResourceAuthority"](capacity_bytes=1024)
    lease = api["acquire_light_1d_retention"](
        authority, owner="dtype", generation=1, layout=layout,
        requested_rows=1, compatibility_byte_ceiling=1024,
        gui_thread_id=threading.get_ident(),
    )
    axis = np.arange(4, dtype=np.float64)
    record = api["Light1DRecord"](
        0, 1, "raw", {"raw": api["Light1DModeData"](
            axis, np.arange(4, dtype=np.int64), None,
        )},
    )
    with pytest.raises(ValueError, match="dtype"):
        lease.retain(record, grant_id=lease.grant_id, generation=1)
    assert lease.keys() == ()


def test_object_containing_dtype_is_never_an_exact_ndarray_byte_claim():
    api = _api()
    Buffer, Mode, Layout = (
        api["Light1DBufferLayout"], api["Light1DModeLayout"],
        api["Light1DLayout"],
    )
    object_dtype = np.dtype(object)
    layout = Layout((Mode(
        "raw",
        Buffer(4, 8, "axis", dtype="<f8", shared=True),
        Buffer(4, object_dtype.itemsize, "intensity", dtype=object_dtype.str),
    ),), "raw")
    authority = api["SessionResourceAuthority"](capacity_bytes=1024)
    lease = api["acquire_light_1d_retention"](
        authority, owner="object-dtype", generation=1, layout=layout,
        requested_rows=1, compatibility_byte_ceiling=1024,
        gui_thread_id=threading.get_ident(),
    )
    record = api["Light1DRecord"](
        0, 1, "raw", {"raw": api["Light1DModeData"](
            np.arange(4, dtype=np.float64),
            np.asarray([object(), object(), object(), object()], dtype=object),
            None,
        )},
    )
    with pytest.raises(ValueError, match="object-containing"):
        lease.retain(record, grant_id=lease.grant_id, generation=1)
    assert lease.keys() == ()


@pytest.mark.parametrize("carrier", ("top-level", "nested-structured"))
def test_dtype_metadata_cannot_escape_the_byte_grant_or_replace_residency(
    carrier,
):
    api = _api()
    _authority, lease = _lease(rows=1)
    axis = np.arange(4, dtype=np.float64)
    original = _record(0, axis)
    lease.retain(
        original, grant_id=lease.grant_id, generation=lease.generation,
    )
    with lease.borrow(0) as resident:
        expected_intensity = resident.modes["raw"].intensity.copy()
    before_keys = lease.keys()
    before_ids = lease.owned_buffer_ids
    before_bytes = lease.unique_owned_ndarray_bytes

    hidden = np.ones(1_000_000, dtype=np.float64)
    hidden_ref = weakref.ref(hidden)
    if carrier == "top-level":
        caller_dtype = np.dtype(np.float64, metadata={"hidden": hidden})
    else:
        nested = np.dtype([
            ("value", np.dtype(np.float64, metadata={"hidden": hidden})),
        ])
        caller_dtype = np.dtype(
            np.float64, metadata={"descriptor": nested},
        )
    replacement = _record(0, axis)
    modes = dict(replacement.modes)
    modes["raw"] = replace(
        modes["raw"], intensity=np.arange(4, dtype=caller_dtype),
    )
    replacement = replace(replacement, modes=modes)

    with pytest.raises(ValueError, match="dtype metadata"):
        lease.retain(
            replacement,
            grant_id=lease.grant_id,
            generation=lease.generation,
        )
    assert lease.keys() == before_keys == (0,)
    assert lease.owned_buffer_ids == before_ids
    assert lease.unique_owned_ndarray_bytes == before_bytes
    with lease.borrow(0) as resident:
        np.testing.assert_array_equal(
            resident.modes["raw"].intensity, expected_intensity,
        )

    del replacement, modes, caller_dtype, hidden
    if carrier == "nested-structured":
        del nested
    gc.collect()
    assert hidden_ref() is None


def test_structured_field_metadata_is_refused_without_minting_residency():
    api = _api()
    Buffer, Mode, Layout = (
        api["Light1DBufferLayout"], api["Light1DModeLayout"],
        api["Light1DLayout"],
    )
    hidden = np.ones(1_000_000, dtype=np.float64)
    hidden_ref = weakref.ref(hidden)
    caller_dtype = np.dtype([
        ("value", np.dtype(np.float64, metadata={"hidden": hidden})),
    ])
    layout = Layout((Mode(
        "raw",
        Buffer(4, 8, "axis", "<f8", shared=True),
        Buffer(4, caller_dtype.itemsize, "intensity", caller_dtype.str),
    ),), "raw")
    authority = api["SessionResourceAuthority"](capacity_bytes=1024)
    lease = api["acquire_light_1d_retention"](
        authority, owner="structured", generation=1, layout=layout,
        requested_rows=1, compatibility_byte_ceiling=1024,
        gui_thread_id=threading.get_ident(),
    )
    record = api["Light1DRecord"](
        0, 1, "raw", {"raw": api["Light1DModeData"](
            np.arange(4, dtype=np.float64),
            np.zeros(4, dtype=caller_dtype),
            None,
        )},
    )
    with pytest.raises(ValueError, match="plain numeric/bool dtype"):
        lease.retain(record, grant_id=lease.grant_id, generation=1)
    assert lease.keys() == ()
    assert lease.owned_buffer_ids == frozenset()

    del record, caller_dtype, hidden
    gc.collect()
    assert hidden_ref() is None


@pytest.mark.parametrize(
    "descriptor",
    ("|b1", "<i2", ">i8", "<u2", ">u8", "<f4", ">f8", "<c8", ">c16"),
)
def test_plain_numeric_and_bool_dtype_descriptors_remain_supported(descriptor):
    api = _api()
    Buffer, Mode, Layout = (
        api["Light1DBufferLayout"], api["Light1DModeLayout"],
        api["Light1DLayout"],
    )
    dtype = np.dtype(descriptor)
    axis_spec = Buffer(4, dtype.itemsize, "axis", dtype.str, shared=True)
    intensity_spec = Buffer(4, dtype.itemsize, "intensity", dtype.str)
    layout = Layout((Mode("raw", axis_spec, intensity_spec),), "raw")
    authority = api["SessionResourceAuthority"](capacity_bytes=1024)
    lease = api["acquire_light_1d_retention"](
        authority, owner=f"plain-{descriptor}", generation=1, layout=layout,
        requested_rows=1, compatibility_byte_ceiling=1024,
        gui_thread_id=threading.get_ident(),
    )
    values = np.asarray((0, 1, 0, 1), dtype=dtype)
    record = api["Light1DRecord"](
        0, 1, "raw", {"raw": api["Light1DModeData"](
            values, values.copy(), None,
        )},
    )
    lease.retain(record, grant_id=lease.grant_id, generation=1)
    with lease.borrow(0) as retained:
        assert retained.modes["raw"].intensity.dtype.str == dtype.str
        assert retained.modes["raw"].intensity.dtype.metadata is None


def test_one_owner_group_cannot_claim_contradictory_dtype_descriptors():
    api = _api()
    Buffer, Mode, Layout = (
        api["Light1DBufferLayout"], api["Light1DModeLayout"],
        api["Light1DLayout"],
    )
    with pytest.raises(ValueError, match="contradictory layouts"):
        Layout((
            Mode(
                "raw",
                Buffer(4, 8, "axis", "<f8", shared=True),
                Buffer(4, 8, "raw", "<f8"),
            ),
            Mode(
                "other",
                Buffer(4, 8, "axis", "<i8", shared=True),
                Buffer(4, 8, "other", "<f8"),
            ),
        ), "raw")


def test_invalid_layout_entry_is_a_typed_refusal_not_attribute_error():
    api = _api()
    with pytest.raises(TypeError, match="Light1DModeLayout"):
        api["Light1DLayout"]((object(),), "raw")


def test_direct_lease_construction_cannot_forge_an_unreserved_owner():
    api = _api()
    authority = api["SessionResourceAuthority"](capacity_bytes=1024)
    with pytest.raises(RuntimeError, match="acquire"):
        api["Light1DRetentionLease"](
            authority,
            grant_id="forged",
            owner="forged",
            generation=1,
            layout=_layout(),
            requested_rows=1,
            row_cap=1,
            reserved_ndarray_bytes=128,
            gui_thread_id=threading.get_ident(),
        )
    assert authority.snapshot().reservation_count == 0


def test_acquire_construction_failure_cancels_the_exact_reservation(monkeypatch):
    api = _api()
    authority = api["SessionResourceAuthority"](capacity_bytes=1024)

    def fail_construction(_self, *_args, **_kwargs):
        raise RuntimeError("lease construction fault")

    with monkeypatch.context() as patch:
        patch.setattr(
            api["Light1DRetentionLease"], "__init__", fail_construction,
        )
        with pytest.raises(RuntimeError, match="construction fault"):
            api["acquire_light_1d_retention"](
                authority, owner="fault", generation=1, layout=_layout(),
                requested_rows=1, compatibility_byte_ceiling=1024,
                gui_thread_id=threading.get_ident(),
            )
    assert authority.snapshot().reservation_count == 0
    recovered = api["acquire_light_1d_retention"](
        authority, owner="fault", generation=1, layout=_layout(),
        requested_rows=1, compatibility_byte_ceiling=1024,
        gui_thread_id=threading.get_ident(),
    )
    assert recovered.generation == 1


def test_acquire_requires_the_exact_resource_authority_type():
    api = _api()

    class DerivedAuthority(api["SessionResourceAuthority"]):
        pass

    authority = DerivedAuthority(capacity_bytes=1024)
    with pytest.raises(TypeError, match="exact SessionResourceAuthority"):
        api["acquire_light_1d_retention"](
            authority, owner="derived", generation=1, layout=_layout(),
            requested_rows=1, compatibility_byte_ceiling=1024,
            gui_thread_id=threading.get_ident(),
        )
    assert authority.snapshot().reservation_count == 0


def test_resident_and_pending_rows_cannot_mint_competing_hydration_owners():
    api = _api()
    _authority, lease = _lease(rows=2)
    axis = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    resident = _record(0, axis)
    lease.retain(resident, grant_id=lease.grant_id, generation=3)
    results = []

    def worker():
        try:
            lease.issue_hydration_token(0)
        except BaseException as exc:
            results.append(exc)
        results.append(lease.issue_hydration_token(1))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert isinstance(results[0], api["Light1DUnavailable"])
    token = results[1]
    assert lease.hydration_authorizations == 1
    with pytest.raises(api["Light1DUnavailable"], match="pending hydration"):
        lease.retain(
            _record(1, axis), grant_id=lease.grant_id, generation=3,
        )
    assert lease.pending_hydration_count == 1
    lease.complete_hydration(token, _record(1, axis))
    with lease.get(1) as shell:
        assert shell.row_identity == 1


def test_hydration_authorization_reserves_a_granted_row_before_worker_io():
    api = _api()
    _authority, lease = _lease(rows=2)
    axis = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    for label in (0, 1):
        lease.retain(
            _record(label, axis), grant_id=lease.grant_id, generation=3,
        )

    tokens = []

    def issue():
        tokens.append(lease.issue_hydration_token(2))
        tokens.append(lease.issue_hydration_token(3))
        with pytest.raises(api["Light1DUnavailable"], match="cap"):
            lease.issue_hydration_token(4)

    thread = threading.Thread(target=issue)
    thread.start()
    thread.join()

    assert lease.keys() == ()
    assert lease.evicted == (0, 1)
    assert lease.pending_hydration_count == lease.row_cap == 2

    lease.complete_hydration(tokens[0], _record(2, axis))
    assert len(lease.keys()) + lease.pending_hydration_count == lease.row_cap
    lease.complete_hydration(tokens[1], _record(3, axis))
    assert lease.keys() == (2, 3)
    assert lease.pending_hydration_count == 0


def test_shared_hydration_initialization_is_serial_then_borrowed_by_workers():
    api = _api()
    _authority, lease = _lease(rows=2)
    first_tokens = []
    first_worker = threading.Thread(
        target=lambda: first_tokens.append(lease.issue_hydration_token(10)),
    )
    first_worker.start()
    first_worker.join()
    first = first_tokens[0]

    errors = []

    def competing_worker():
        try:
            lease.issue_hydration_token(11)
        except BaseException as exc:
            errors.append(exc)

    competing = threading.Thread(target=competing_worker)
    competing.start()
    competing.join()
    assert len(errors) == 1
    assert isinstance(errors[0], api["Light1DUnavailable"])
    assert "shared" in str(errors[0])

    axis = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    lease.complete_hydration(first, _record(10, axis))
    second_tokens = []
    second_worker = threading.Thread(
        target=lambda: second_tokens.append(lease.issue_hydration_token(11)),
    )
    second_worker.start()
    second_worker.join()
    second = second_tokens[0]
    assert not second.may_create_shared
    with lease.borrow(10) as resident:
        canonical_axis = resident.modes["raw"].coordinate
        assert second.shared_roots["q-axis"] is canonical_axis
        assert canonical_axis is not axis
        lease.complete_hydration(second, _record(11, canonical_axis))


def test_foreign_hydration_token_cannot_consume_the_exact_pending_owner():
    api = _api()
    _authority, lease = _lease(rows=1)
    tokens = []

    worker = threading.Thread(
        target=lambda: tokens.append(lease.issue_hydration_token(4)),
    )
    worker.start()
    worker.join()
    exact = tokens[0]
    forged = api["Light1DHydrationToken"](
        exact.grant_id, exact.generation, exact.row_identity, exact.ordinal,
    )
    with pytest.raises(api["Light1DStaleGeneration"], match="stale or foreign"):
        lease.complete_hydration(forged, _record(4, np.arange(4, dtype=np.float64)))
    assert lease.pending_hydration_count == 1
    lease.complete_hydration(exact, _record(4, np.arange(4, dtype=np.float64)))
    assert lease.pending_hydration_count == 0


def test_failed_hydration_can_abandon_its_exact_slot_and_retry():
    api = _api()
    _authority, lease = _lease(rows=1)
    tokens = []

    def worker():
        first = lease.issue_hydration_token(8)
        lease.abandon_hydration(first)
        tokens.append(lease.issue_hydration_token(8))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert lease.pending_hydration_count == 1
    lease.complete_hydration(
        tokens[0], _record(8, np.arange(4, dtype=np.float64)),
    )
    with lease.get(8) as shell:
        assert shell.row_identity == 8


def test_allocation_factory_is_concurrently_canonical_by_exact_identity():
    api = _api()
    from xrd_tools.session import SessionResourceRequirements, resolve_session_policy

    allocation = resolve_session_policy(
        SessionResourceRequirements(height=2, width=2, native_itemsize=2),
        envelope_bytes=2 * 1024 ** 3,
        env={},
    ).allocation
    barrier = threading.Barrier(5)
    authorities = []
    errors = []

    def derive():
        try:
            barrier.wait()
            authorities.append(
                api["SessionResourceAuthority"].from_allocation(allocation)
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=derive) for _ in range(4)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()
    assert errors == []
    assert len({id(authority) for authority in authorities}) == 1
    assert api["SessionResourceAuthority"].from_allocation(allocation) is authorities[0]


def test_parent_allocation_constructor_path_cannot_bypass_canonical_factory():
    api = _api()
    from xrd_tools.session import SessionResourceRequirements, resolve_session_policy

    allocation = resolve_session_policy(
        SessionResourceRequirements(height=2, width=2, native_itemsize=2),
        envelope_bytes=2 * 1024 ** 3,
        env={},
    ).allocation
    with pytest.raises(TypeError, match="from_allocation"):
        api["SessionResourceAuthority"](
            capacity_bytes=allocation.envelope_bytes,
            committed_bytes=allocation.categories,
            parent_allocation=allocation,
        )


class _WriteableFlags:
    def __init__(self) -> None:
        self.writeable = True


class _NumericArrayLike:
    ndim = 1
    shape = (4,)
    dtype = np.dtype(np.float64)
    nbytes = 32
    base = None

    def __init__(self) -> None:
        self.flags = _WriteableFlags()


class _NoOpFenceArray(_NumericArrayLike):
    def setflags(self, *, write):
        pass


class _FailedFenceArray(_NumericArrayLike):
    def setflags(self, *, write):
        raise OSError("write fence failed")


class _WorkingFenceArray(_NumericArrayLike):
    def setflags(self, *, write):
        self.flags.writeable = bool(write)


class _HiddenPayloadArray(_WorkingFenceArray):
    def __init__(self) -> None:
        super().__init__()
        self.hidden_payload = np.ones(1_000_000, dtype=np.float64)


def test_unusable_array_write_fence_is_refused_without_replacing_resident_row(
    monkeypatch,
):
    import xrd_tools.session.light_1d_retention as retention

    _authority, lease = _lease(rows=1)
    axis = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    original = _record(0, axis)
    lease.retain(original, grant_id=lease.grant_id, generation=3)
    shell = lease.get(0)
    canonical_intensity = shell.modes["raw"].intensity
    before_ids = lease.owned_buffer_ids
    replacement = _record(0, axis)

    def fail_fence(_array):
        raise ValueError("light-1D array write fence failed")

    monkeypatch.setattr(retention, "_freeze_array", fail_fence)
    with pytest.raises(ValueError, match="write fence"):
        lease.retain(
            replacement, grant_id=lease.grant_id, generation=3,
        )
    assert shell.row_identity == original.row_identity
    assert shell.modes["raw"].intensity is canonical_intensity
    shell.close()
    assert lease.owned_buffer_ids == before_ids


def test_duck_array_cannot_hide_an_unaccounted_ndarray_payload():
    _authority, lease = _lease(rows=1)
    axis = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    record = _record(0, axis)
    modes = dict(record.modes)
    modes["raw"] = replace(
        modes["raw"], intensity=_HiddenPayloadArray(),
    )
    with pytest.raises(ValueError, match="exact NumPy ndarray"):
        lease.retain(
            replace(record, modes=modes),
            grant_id=lease.grant_id,
            generation=lease.generation,
        )
    assert lease.keys() == ()


def test_non_ndarray_buffer_root_is_refused_as_unaccounted_owner():
    _authority, lease = _lease(rows=1)
    axis = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    numeric_view = np.ndarray(
        shape=(4,), dtype=np.float64, buffer=bytearray(32),
    )
    record = _record(0, axis)
    modes = dict(record.modes)
    modes["raw"] = replace(modes["raw"], intensity=numeric_view)
    with pytest.raises(ValueError, match="exact NumPy ndarray root"):
        lease.retain(
            replace(record, modes=modes),
            grant_id=lease.grant_id,
            generation=3,
        )
    assert lease.keys() == ()


def test_child_b_publication_a1_replacement_funds_fully_assigned_allocation():
    from xrd_tools.session import (
        Light1DCleanupHooks, Light1DFundingMode, SessionResourceAuthority,
        SessionResourceRequirements, acquire_light_1d_retention,
        resolve_session_policy,
    )
    import xrd_tools.session.policy as policy

    requirements = SessionResourceRequirements(
        height=4, width=4, native_itemsize=2,
        modes_1d=2, npt_1d=4, sigma_1d=1,
    )
    allocation = resolve_session_policy(
        requirements, envelope_bytes=policy.floor_bytes(requirements), env={},
    ).allocation
    authority = SessionResourceAuthority.from_allocation(allocation)
    before = authority.snapshot()
    assert before.available_bytes == 0
    layout = _layout()
    one_row = layout.shared_bytes + layout.per_row_unique_ndarray_bytes
    lease = acquire_light_1d_retention(
        authority, owner="child-b-full", generation=1, layout=layout,
        requested_rows=4, compatibility_byte_ceiling=one_row,
        gui_thread_id=threading.get_ident(),
        funding_mode=Light1DFundingMode.REPLACE_PUBLICATION_A1,
        current_lineage_rows=None,
    )
    after = authority.snapshot()
    assert lease.row_cap == 1
    assert after.available_bytes == 0
    assert after.committed_bytes["records"] == before.committed_bytes["records"]
    assert after.committed_bytes["publication"] == (
        before.committed_bytes["publication"] - lease.reserved_ndarray_bytes
    )
    assert after.categories["light_1d"] == lease.reserved_ndarray_bytes
    assert after.reserved_bytes == before.reserved_bytes
    lease.release(reason="test", hooks=Light1DCleanupHooks())
    assert authority.snapshot() == before


def test_child_b_publication_a1_replacement_preserves_record_a1_and_lineage_cap():
    from xrd_tools.session import (
        Light1DCleanupHooks, Light1DFundingMode, SessionResourceAuthority,
        SessionResourceRequirements, acquire_light_1d_retention,
        resolve_session_policy,
    )

    requirements = SessionResourceRequirements(
        height=4, width=4, native_itemsize=2,
        modes_1d=2, npt_1d=4, sigma_1d=1,
    )
    requests = {"record_items": 4, "publication_items": 4}
    draft = resolve_session_policy(
        requirements, envelope_bytes=4 * 1024 ** 3,
        requests=requests, env={},
    ).allocation
    allocation = resolve_session_policy(
        requirements, envelope_bytes=draft.assigned_bytes,
        requests=requests, env={},
    ).allocation
    assert allocation.record_items == allocation.publication_items == 4
    authority = SessionResourceAuthority.from_allocation(allocation)
    layout = _layout()
    ceiling = layout.shared_bytes + 6 * layout.per_row_unique_ndarray_bytes
    baseline = authority.snapshot()

    with pytest.raises(ValueError, match="current_lineage_rows"):
        acquire_light_1d_retention(
            authority, owner="omitted", generation=1, layout=layout,
            requested_rows=6, compatibility_byte_ceiling=ceiling,
            gui_thread_id=threading.get_ident(),
            funding_mode=Light1DFundingMode.REPLACE_PUBLICATION_A1,
        )
    assert authority.snapshot() == baseline
    for invalid in (True, -1, 1.0):
        with pytest.raises((TypeError, ValueError)):
            acquire_light_1d_retention(
                authority, owner=f"invalid-{invalid!r}", generation=1,
                layout=layout, requested_rows=6,
                compatibility_byte_ceiling=ceiling,
                gui_thread_id=threading.get_ident(),
                funding_mode=Light1DFundingMode.REPLACE_PUBLICATION_A1,
                current_lineage_rows=invalid,
            )
        assert authority.snapshot() == baseline

    open_live = acquire_light_1d_retention(
        authority, owner="open-live", generation=1, layout=layout,
        requested_rows=6, compatibility_byte_ceiling=ceiling,
        gui_thread_id=threading.get_ident(),
        funding_mode=Light1DFundingMode.REPLACE_PUBLICATION_A1,
        current_lineage_rows=None,
    )
    assert open_live.row_cap == 4
    assert authority.snapshot().committed_bytes["records"] == (
        baseline.committed_bytes["records"]
    )
    open_live.release(reason="test", hooks=Light1DCleanupHooks())

    finite = acquire_light_1d_retention(
        authority, owner="finite", generation=1, layout=layout,
        requested_rows=6, compatibility_byte_ceiling=ceiling,
        gui_thread_id=threading.get_ident(),
        funding_mode=Light1DFundingMode.REPLACE_PUBLICATION_A1,
        current_lineage_rows=2,
    )
    assert finite.row_cap == 2
    assert finite.reserved_ndarray_bytes <= (
        allocation.publication_items * requirements.result_1d_bytes
    )
    assert authority.snapshot().committed_bytes["records"] == (
        baseline.committed_bytes["records"]
    )
    finite.release(reason="test", hooks=Light1DCleanupHooks())
    assert authority.snapshot() == baseline


def test_child_b_publication_a1_replacement_refuses_layout_without_mutation():
    from xrd_tools.session import (
        Light1DFundingMode, SessionResourceAuthority,
        SessionResourceRequirements, acquire_light_1d_retention,
        resolve_session_policy,
    )
    import xrd_tools.session.policy as policy

    requirements = SessionResourceRequirements(
        height=1, width=1, native_itemsize=1, modes_1d=1, npt_1d=1,
    )
    allocation = resolve_session_policy(
        requirements, envelope_bytes=policy.floor_bytes(requirements), env={},
    ).allocation
    authority = SessionResourceAuthority.from_allocation(allocation)
    before = authority.snapshot()
    with pytest.raises(ValueError, match="publication A1"):
        acquire_light_1d_retention(
            authority, owner="oversize", generation=1, layout=_layout(),
            requested_rows=1, compatibility_byte_ceiling=10_000,
            gui_thread_id=threading.get_ident(),
            funding_mode=Light1DFundingMode.REPLACE_PUBLICATION_A1,
            current_lineage_rows=1,
        )
    assert authority.snapshot() == before
    with pytest.raises(TypeError, match="funding_mode"):
        acquire_light_1d_retention(
            authority, owner="wrong-mode", generation=1, layout=_layout(),
            requested_rows=1, compatibility_byte_ceiling=10_000,
            gui_thread_id=threading.get_ident(), funding_mode="replacement",
            current_lineage_rows=1,
        )
    assert authority.snapshot() == before
    ordinary = SessionResourceAuthority(capacity_bytes=10_000)
    ordinary_before = ordinary.snapshot()
    with pytest.raises(TypeError, match="allocation"):
        acquire_light_1d_retention(
            ordinary, owner="foreign", generation=1, layout=_layout(),
            requested_rows=1, compatibility_byte_ceiling=10_000,
            gui_thread_id=threading.get_ident(),
            funding_mode=Light1DFundingMode.REPLACE_PUBLICATION_A1,
            current_lineage_rows=1,
        )
    assert ordinary.snapshot() == ordinary_before


def test_child_b_publication_a1_replacement_constructor_failure_restores_categories(
    monkeypatch,
):
    from xrd_tools.session import (
        Light1DCleanupHooks, Light1DFundingMode, SessionResourceAuthority,
        SessionResourceRequirements, acquire_light_1d_retention,
        resolve_session_policy,
    )
    import xrd_tools.session.light_1d_retention as retention
    import xrd_tools.session.policy as policy

    requirements = SessionResourceRequirements(
        height=4, width=4, native_itemsize=2,
        modes_1d=2, npt_1d=4, sigma_1d=1,
    )
    allocation = resolve_session_policy(
        requirements, envelope_bytes=policy.floor_bytes(requirements), env={},
    ).allocation
    authority = SessionResourceAuthority.from_allocation(allocation)
    before = authority.snapshot()
    original = retention.Light1DRetentionLease

    def fail_construction(*_args, **_kwargs):
        raise RuntimeError("injected lease construction failure")

    monkeypatch.setattr(retention, "Light1DRetentionLease", fail_construction)
    with pytest.raises(RuntimeError, match="injected lease construction failure"):
        acquire_light_1d_retention(
            authority, owner="constructor", generation=1, layout=_layout(),
            requested_rows=1, compatibility_byte_ceiling=10_000,
            gui_thread_id=threading.get_ident(),
            funding_mode=Light1DFundingMode.REPLACE_PUBLICATION_A1,
            current_lineage_rows=1,
        )
    assert authority.snapshot() == before
    monkeypatch.setattr(retention, "Light1DRetentionLease", original)
    lease = acquire_light_1d_retention(
        authority, owner="constructor", generation=1, layout=_layout(),
        requested_rows=1, compatibility_byte_ceiling=10_000,
        gui_thread_id=threading.get_ident(),
        funding_mode=Light1DFundingMode.REPLACE_PUBLICATION_A1,
        current_lineage_rows=1,
    )
    lease.release(reason="test", hooks=Light1DCleanupHooks())
    assert authority.snapshot() == before


def test_child_b_retained_custody_release_failure_retries_exactly_once():
    from xrd_tools.session import (
        DynamicRunState, Light1DCleanupHooks, Light1DCleanupPending,
        Light1DCustodySlot, Light1DCustodyState,
        Light1DRetainedCustodyReceipt,
    )

    authority, lease = _lease(rows=1, generation=3)
    failures = [OSError("custody cleanup fault")]
    calls = []

    def verify():
        calls.append("verify")
        if failures:
            raise failures.pop()

    hooks = Light1DCleanupHooks(verify=verify)
    slot = Light1DCustodySlot(
        grant_id=lease.grant_id, owner=lease.owner,
        generation=lease.generation, cleanup_hooks=hooks,
    )
    wrong_grant = Light1DCustodySlot(
        grant_id="foreign", owner=lease.owner,
        generation=lease.generation, cleanup_hooks=hooks,
    )
    wrong_owner = Light1DCustodySlot(
        grant_id=lease.grant_id, owner="foreign",
        generation=lease.generation, cleanup_hooks=hooks,
    )
    wrong_generation = Light1DCustodySlot(
        grant_id=lease.grant_id, owner=lease.owner,
        generation=lease.generation + 1, cleanup_hooks=hooks,
    )
    lease_before = (lease.state, lease.keys(), authority.snapshot())
    refusals = (
        (slot, object(), hooks, DynamicRunState.FINISHED, TypeError, "exact light-1D lease"),
        (slot, lease, Light1DCleanupHooks(), DynamicRunState.FINISHED, RuntimeError, "cleanup hooks"),
        (slot, lease, hooks, DynamicRunState.ABORTED, ValueError, "terminal"),
        (wrong_grant, lease, hooks, DynamicRunState.FINISHED, ValueError, "grant"),
        (wrong_owner, lease, hooks, DynamicRunState.FINISHED, ValueError, "owner"),
        (wrong_generation, lease, hooks, DynamicRunState.FINISHED, ValueError, "generation"),
    )
    for target, candidate, candidate_hooks, terminal, error, match in refusals:
        with pytest.raises(error, match=match):
            target.adopt(
                candidate, cleanup_hooks=candidate_hooks, terminal=terminal,
            )
        assert target.state is Light1DCustodyState.PENDING
        assert target.custody_receipt is None
        assert (lease.state, lease.keys(), authority.snapshot()) == lease_before

    custody = slot.adopt(
        lease, cleanup_hooks=hooks, terminal=DynamicRunState.FINISHED,
    )
    assert type(custody) is Light1DRetainedCustodyReceipt
    assert slot.custody_receipt is custody
    assert slot.state is Light1DCustodyState.RETAINED
    assert (custody.grant_id, custody.owner, custody.generation) == (
        lease.grant_id, lease.owner, lease.generation,
    )
    assert custody.terminal == "finished"
    assert custody.retained_rows == len(lease.keys())
    assert custody.reserved_bytes == lease.reserved_ndarray_bytes
    assert slot.adopt(
        lease, cleanup_hooks=hooks, terminal=DynamicRunState.FINISHED,
    ) is custody

    with pytest.raises(Light1DCleanupPending) as caught:
        slot.release(reason="close")
    token = caught.value.token
    assert slot.state is Light1DCustodyState.CLEANUP_PENDING
    assert token is lease.cleanup_receipt.retry_token
    assert authority.snapshot().categories["light_1d"] == (
        lease.reserved_ndarray_bytes
    )
    with pytest.raises(RuntimeError, match="exact retry token"):
        slot.retry_cleanup(object())
    released = slot.retry_cleanup(token)
    assert slot.state is Light1DCustodyState.RELEASED
    assert slot.retry_cleanup(token) is released
    assert calls == ["verify", "verify"]
    assert authority.snapshot().reserved_bytes == 0
