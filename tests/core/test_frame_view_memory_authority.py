from __future__ import annotations

import copy
import gc
import pickle
import sys
from threading import Event, Thread
import weakref

import h5py
import numpy as np
import pytest

from xrd_tools.core.physical_memory import (
    PhysicalRootAuthority,
    PhysicalRootExchange,
    PhysicalRootExchangePhase,
    PhysicalRootLease,
    PhysicalRootReservation,
    physical_root_fact,
)
from xrd_tools.io import frame_view as module


class _InjectedPublicationFailure(BaseException):
    pass


def _install_reader_cache_cas_cut(
    reader,
    monkeypatch,
    *,
    expected_phase,
    replacement_phase,
    cut: str,
) -> None:
    real_cas = reader._cas_reader_cache_state
    fired = False

    def cut_cas(expected, replacement):
        nonlocal fired
        if (
            not fired
            and expected.phase is expected_phase
            and replacement.phase is replacement_phase
        ):
            fired = True
            if cut == "after":
                assert real_cas(expected, replacement)
            raise _InjectedPublicationFailure(
                f"reader cache {expected_phase.value}->{replacement_phase.value} {cut}"
            )
        return real_cas(expected, replacement)

    monkeypatch.setattr(reader, "_cas_reader_cache_state", cut_cas)


def _install_exchange_cut(monkeypatch, method: str, cut: str):
    real_method = getattr(PhysicalRootExchange, method)
    fired = False

    def cut_method(exchange):
        nonlocal fired
        if not fired:
            fired = True
            if cut == "after":
                real_method(exchange)
            raise _InjectedPublicationFailure(f"exchange {method} {cut}")
        return real_method(exchange)

    monkeypatch.setattr(PhysicalRootExchange, method, cut_method)
    return real_method


def _assert_reader_cache_matches_authority(reader) -> None:
    state = reader._snapshot_reader_cache_state()
    assert state.phase is module._ReaderCachePhase.OPEN
    assert state.journal is None
    roots = reader._memory_authority.retained_roots
    assert reader.semantic_references == len(state.entries)
    projected = reader._read_cache
    assert tuple(projected) == tuple(state.entries)
    expected_roots: dict[int, tuple[object, int, int]] = {}
    expected_tokens: set[int] = set()
    for key, entry in state.entries.items():
        assert not entry.lease.released
        assert projected[key] is entry.array
        lease = entry.lease
        assert lease._authority is reader._memory_authority
        token = lease._token
        assert id(token) not in expected_tokens
        expected_tokens.add(id(token))
        fact = physical_root_fact(entry.array)
        identity = id(fact.root)
        prior = expected_roots.get(identity)
        if prior is not None:
            assert prior[0] is fact.root
            assert prior[1] == fact.nbytes
        expected_roots[identity] = (
            fact.root,
            fact.nbytes,
            1 if prior is None else prior[2] + 1,
        )
        assert any(fact.root is retained for retained in roots)
    authority_state = reader._memory_authority._snapshot_state()
    assert not authority_state.closed
    assert len(authority_state.bindings) == len(state.entries)
    assert {id(binding.token) for binding in authority_state.bindings} == expected_tokens
    for entry in state.entries.values():
        lease = entry.lease
        matches = tuple(
            binding
            for binding in authority_state.bindings
            if binding.token is lease._token
        )
        assert len(matches) == 1
        binding = matches[0]
        fact = physical_root_fact(entry.array)
        assert binding.semantic is lease._semantic
        assert lease._token.semantic is lease._semantic
        assert binding.root_identity == id(fact.root)
    assert len(authority_state.roots) == len(expected_roots)
    for item in authority_state.roots:
        expected = expected_roots[item.identity]
        assert item.identity == id(item.root)
        assert item.root is expected[0]
        assert item.nbytes == expected[1]
        assert item.references == expected[2]
    assert all(expected_roots[id(root)][0] is root for root in roots)
    if state.scan_data_columns is not None:
        cached_values = tuple(entry.array for entry in state.entries.values())
        for column in state.scan_data_columns.values():
            assert any(column is value for value in cached_values)


def _assert_cache_items_exact(actual, expected) -> None:
    assert len(actual) == len(expected)
    for (actual_key, actual_value), (expected_key, expected_value) in zip(
        actual,
        expected,
        strict=True,
    ):
        assert actual_key == expected_key
        assert actual_value.array is expected_value.array
        assert actual_value.lease is expected_value.lease


def _reader_marker_is_present(state, marker) -> bool:
    return any(marker_ref() is marker for marker_ref in state.terminal_evidence)


def _assert_authority_closed_exact(authority, leases) -> None:
    state = authority._snapshot_state()
    assert state.closed
    assert state.bindings == ()
    assert state.roots == ()
    for lease in leases:
        assert all(binding.token is not lease._token for binding in state.bindings)
        lease.release()
        assert lease.released


def test_reader_array_bundle_refuses_copy_and_serialization(tmp_path) -> None:
    path = tmp_path / "reader_bundle_linear.nxs"
    _one_dimensional_file(path)
    with module.FrameViewReader(path) as reader:
        bundle = module._ReaderArrayBundle(reader)
        try:
            with pytest.raises(TypeError, match="cannot be copied"):
                copy.copy(bundle)
            with pytest.raises(TypeError, match="cannot be copied"):
                copy.deepcopy(bundle)
            with pytest.raises(TypeError, match="cannot be serialized"):
                pickle.dumps(bundle)
            with pytest.raises(TypeError, match="cannot be serialized"):
                bundle.__reduce__()
        finally:
            bundle.rollback()
        assert reader._snapshot_reader_cache_state().phase is module._ReaderCachePhase.OPEN
        _assert_reader_cache_matches_authority(reader)


@pytest.mark.parametrize("opened", (False, True))
def test_frame_view_reader_refuses_copy_and_serialization(
    tmp_path, opened: bool,
) -> None:
    path = tmp_path / f"reader_linear_{opened}.nxs"
    _one_dimensional_file(path)
    reader = module.FrameViewReader(path)
    if opened:
        reader.__enter__()
    try:
        with pytest.raises(TypeError, match="cannot be copied"):
            copy.copy(reader)
        with pytest.raises(TypeError, match="cannot be copied"):
            copy.deepcopy(reader)
        with pytest.raises(TypeError, match="cannot be serialized"):
            pickle.dumps(reader)
        with pytest.raises(TypeError, match="cannot be serialized"):
            reader.__reduce__()
    finally:
        if opened:
            reader.__exit__(None, None, None)


def _one_dimensional_file(
    path,
    *,
    points: int = 8,
    rows: int = 1,
    source_dtype=np.float32,
    linked_sigma: bool = False,
    scan_alias: bool = False,
) -> None:
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        group = entry.create_group("integrated_1d")
        labels = group.create_dataset(
            "frame_index", data=np.arange(5, 5 + rows, dtype=np.int64),
        )
        group.create_dataset(
            "q", data=np.linspace(0.1, 1.0, points, dtype=source_dtype),
        )
        intensity = group.create_dataset(
            "intensity",
            data=np.arange(rows * points, dtype=source_dtype).reshape(
                rows, points,
            ),
        )
        if linked_sigma:
            group["sigma"] = intensity
        else:
            group.create_dataset(
                "sigma",
                data=np.arange(rows * points, dtype=source_dtype).reshape(
                    rows, points,
                ),
            )
        if scan_alias:
            scan = entry.create_group("scan_data")
            scan["frame_index"] = labels
            scan["monitor"] = labels


def _add_second_1d_mode(
    path,
    *,
    points: int = 8,
    malformed_intensity: bool = False,
    linked_q: bool = False,
    linked_intensity: bool = False,
) -> None:
    with h5py.File(path, "r+") as handle:
        primary = handle["entry/integrated_1d"]
        child = primary.create_group("q_oop")
        child["frame_index"] = primary["frame_index"]
        if linked_q:
            child["q"] = primary["q"]
        else:
            child.create_dataset(
                "q", data=np.linspace(0.2, 1.1, points, dtype=np.float32),
            )
        if linked_intensity:
            child["intensity"] = primary["intensity"]
        else:
            width = points + 1 if malformed_intensity else points
            child.create_dataset(
                "intensity", data=np.arange(width, dtype=np.float32)[None, :],
            )


def _add_scan_monitor(path, *, rows: int = 1) -> None:
    with h5py.File(path, "r+") as handle:
        scan = handle["entry"].create_group("scan_data")
        scan["frame_index"] = handle["entry/integrated_1d/frame_index"]
        scan.create_dataset(
            "monitor",
            data=np.arange(11, 11 + rows, dtype=np.float32),
        )


def _object_addresses(path, names: tuple[str, ...]) -> dict[str, int]:
    with h5py.File(path, "r") as handle:
        return {
            name: int(h5py.h5o.get_info(handle[name].id).addr)
            for name in names
        }


def test_physical_root_authority_counts_ultimate_root_and_rolls_back() -> None:
    base = np.empty((1024,), dtype=np.uint8)
    tiny_view = base[:1]
    assert physical_root_fact(tiny_view).root is base
    assert physical_root_fact(tiny_view).nbytes == base.nbytes

    too_small = PhysicalRootAuthority(base.nbytes - 1)
    with too_small.reserve() as reservation:
        with pytest.raises(ValueError, match="bytes exceed limit"):
            reservation.reserve(tiny_view, "tiny")
    assert too_small.retained_bytes == 0
    assert too_small.retained_root_count == 0
    assert too_small.semantic_references == 0

    authority = PhysicalRootAuthority(base.nbytes)
    reservation = authority.reserve()
    reservation.reserve(tiny_view, "array-view")
    reservation.reserve(memoryview(base), "memory-view")
    with pytest.raises(RuntimeError, match="already active"):
        authority.reserve()
    leases = reservation.commit()
    assert authority.retained_bytes == base.nbytes
    assert authority.retained_root_count == 1
    assert authority.semantic_references == 2
    assert authority.retained_roots == (base,)
    leases["array-view"].release()
    assert authority.retained_bytes == base.nbytes
    assert authority.semantic_references == 1
    leases["memory-view"].release()
    assert authority.retained_bytes == 0
    assert authority.retained_root_count == 0
    authority.close()


@pytest.mark.parametrize("linked_sigma", (True, False))
def test_reader_cache_uses_physical_hdf_identity_after_role_validation(
    tmp_path, monkeypatch, linked_sigma,
) -> None:
    path = tmp_path / f"physical_cache_{linked_sigma}.nxs"
    _one_dimensional_file(
        path, linked_sigma=linked_sigma, scan_alias=True,
    )
    _add_second_1d_mode(path, linked_q=True, linked_intensity=True)
    names = (
        "entry/integrated_1d/frame_index",
        "entry/integrated_1d/q",
        "entry/integrated_1d/intensity",
        "entry/integrated_1d/sigma",
        "entry/integrated_1d/q_oop/frame_index",
        "entry/integrated_1d/q_oop/q",
        "entry/integrated_1d/q_oop/intensity",
        "entry/scan_data/frame_index",
        "entry/scan_data/monitor",
    )
    addresses = _object_addresses(path, names)
    calls: list[int] = []
    qualified_roles: list[str] = []
    real_read_direct = h5py.Dataset.read_direct
    real_qualify_vector = module._qualify_vector

    def tracked(dataset, *args, **kwargs):
        calls.append(int(h5py.h5o.get_info(dataset.id).addr))
        return real_read_direct(dataset, *args, **kwargs)

    def tracked_qualify(*args, **kwargs):
        qualified_roles.append(str(kwargs["role"]))
        return real_qualify_vector(*args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "read_direct", tracked)
    monkeypatch.setattr(module, "_qualify_vector", tracked_qualify)
    with module.FrameViewReader(path) as reader:
        record = reader.read_record(5)
        view = next(iter(record.results_1d.values()))
        assert view.metadata_raw["monitor"] == 5
        assert view.axis_1d is not None
        assert view.axis_1d.values.dtype == np.dtype(np.float64)
        assert view.axis_1d.values.flags.c_contiguous
        assert view.intensity_1d.dtype == np.dtype(np.float64)
        assert view.intensity_1d.flags.c_contiguous
        if linked_sigma:
            assert view.intensity_1d is view.sigma_1d
        else:
            assert view.intensity_1d is not view.sigma_1d
        assert len(record.results_1d) == 2
        assert len({id(item.intensity_1d) for item in record.results_1d.values()}) == 1
        assert reader.retained_bytes > 0
        assert reader.retained_root_count > 0

    assert calls.count(addresses[names[0]]) == 1
    assert calls.count(addresses[names[1]]) == 1
    assert addresses[names[0]] == addresses[names[4]]
    assert addresses[names[1]] == addresses[names[5]]
    assert addresses[names[2]] == addresses[names[6]]
    assert addresses[names[0]] == addresses[names[7]] == addresses[names[8]]
    assert "/entry/scan_data/monitor" in qualified_roles
    if linked_sigma:
        assert addresses[names[2]] == addresses[names[3]]
        assert calls.count(addresses[names[2]]) == 1
    else:
        assert addresses[names[2]] != addresses[names[3]]
        assert calls.count(addresses[names[2]]) == 1
        assert calls.count(addresses[names[3]]) == 1
    assert reader.retained_bytes == 0
    assert reader.retained_root_count == 0
    assert reader.semantic_references == 0


def test_float32_row_reserves_final_float64_bytes_before_hdf_read(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "final_float64_reservation.nxs"
    points = 32
    _one_dimensional_file(path, points=points, source_dtype=np.float32)
    with module.FrameViewReader(path) as reader:
        baseline = reader.retained_bytes

    monkeypatch.setattr(
        module,
        "_MAX_READER_RETAINED_BYTES",
        baseline + points * np.dtype(np.float64).itemsize - 1,
    )
    intensity_address = _object_addresses(
        path, ("entry/integrated_1d/intensity",),
    )["entry/integrated_1d/intensity"]
    reads: list[int] = []
    real_read_direct = h5py.Dataset.read_direct

    def guarded(dataset, *args, **kwargs):
        address = int(h5py.h5o.get_info(dataset.id).addr)
        if address == intensity_address:
            reads.append(address)
        return real_read_direct(dataset, *args, **kwargs)

    with module.FrameViewReader(path) as reader:
        monkeypatch.setattr(h5py.Dataset, "read_direct", guarded)
        with pytest.raises(ValueError, match="bytes exceed limit"):
            reader.read(5)
        assert reads == []
        assert reader.retained_bytes == baseline


def test_vlen_pointer_root_is_claimed_before_allocation_or_cell_read_and_retries(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "vlen_pointer_preflight.nxs"
    rows = 3
    with h5py.File(path, "w") as handle:
        scan = handle.create_group("entry/scan_data")
        scan.create_dataset(
            "frame_index", data=np.arange(5, 5 + rows, dtype=np.int64),
        )
        scan.create_dataset(
            "sample",
            data=np.asarray(["a", "bb", "ccc"], dtype=object),
            dtype=h5py.string_dtype("utf-8"),
        )
    with module.FrameViewReader(path) as admitted:
        baseline = admitted.retained_bytes

    pointer_bytes = rows * np.dtype(object).itemsize
    monkeypatch.setattr(
        module,
        "_MAX_READER_RETAINED_BYTES",
        baseline + pointer_bytes - 1,
    )
    real_empty = np.empty
    real_cell = module._bounded_utf8_item
    real_claim = PhysicalRootExchange.claim
    real_bind = PhysicalRootExchange.bind
    object_allocations: list[tuple[int, ...]] = []
    cell_reads: list[int] = []
    events: list[tuple[str, object]] = []

    def tracked_claim(reservation, nbytes):
        events.append(("claim", int(nbytes)))
        return real_claim(reservation, nbytes)

    def tracked_empty(*args, **kwargs):
        dtype = kwargs.get("dtype", args[1] if len(args) > 1 else float)
        if np.dtype(dtype) == np.dtype(object):
            object_allocations.append(tuple(args[0]))
            events.append(("empty", tuple(args[0])))
        return real_empty(*args, **kwargs)

    def tracked_bind(reservation, token, value, semantic):
        if isinstance(value, np.ndarray) and value.dtype == np.dtype(object):
            events.append(("bind", tuple(value.shape)))
        return real_bind(reservation, token, value, semantic)

    def tracked_cell(dataset, row, *, role):
        cell_reads.append(int(row))
        events.append(("cell", int(row)))
        return real_cell(dataset, row, role=role)

    monkeypatch.setattr(module.np, "empty", tracked_empty)
    monkeypatch.setattr(module, "_bounded_utf8_item", tracked_cell)
    with module.FrameViewReader(path) as reader:
        monkeypatch.setattr(PhysicalRootExchange, "claim", tracked_claim)
        monkeypatch.setattr(PhysicalRootExchange, "bind", tracked_bind)
        prior_cache = dict(reader._read_cache)
        prior_roots = reader.retained_root_count
        prior_refs = reader.semantic_references
        with pytest.raises(ValueError, match="bytes exceed limit"):
            reader.read(5)
        assert object_allocations == []
        assert cell_reads == []
        assert reader._scan_data_columns is None
        assert reader._read_cache == prior_cache
        assert reader.retained_bytes == baseline
        assert reader.retained_root_count == prior_roots
        assert reader.semantic_references == prior_refs

        events.clear()
        monkeypatch.setattr(
            reader._memory_authority,
            "_limit",
            baseline + pointer_bytes,
        )
        view = reader.read(5)
        assert view.metadata_raw["sample"] == "a"
        assert object_allocations == [(rows,)]
        assert cell_reads == [0, 1, 2]
        assert events.index(("claim", pointer_bytes)) < events.index(
            ("empty", (rows,))
        ) < events.index(("bind", (rows,))) < events.index(("cell", 0))
        assert reader._scan_data_columns is not None
        assert reader.retained_bytes == baseline + pointer_bytes


def test_vlen_hardlink_alias_costs_one_root_and_validates_both_roles(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "vlen_hardlink_alias.nxs"
    rows = 2
    values = ("x" * 1024, "y" * 1024)
    _one_dimensional_file(path, rows=rows)
    with h5py.File(path, "r+") as handle:
        scan = handle["entry"].create_group("scan_data")
        scan["frame_index"] = handle["entry/integrated_1d/frame_index"]
        first = scan.create_dataset(
            "a_first",
            data=np.asarray(values, dtype=object),
            dtype=h5py.string_dtype("utf-8"),
        )
        scan["z_alias"] = first
    one_logical_column = rows * np.dtype(object).itemsize + sum(
        max(64, sys.getsizeof(value)) for value in values
    )
    monkeypatch.setattr(
        module, "_MAX_SCAN_DATA_BYTES", one_logical_column,
    )
    real_qualify = module._qualify_vector
    real_cell = module._bounded_utf8_item
    qualified_roles: list[str] = []
    cell_reads: list[tuple[str, int]] = []

    def tracked_qualify(*args, **kwargs):
        qualified_roles.append(str(kwargs["role"]))
        return real_qualify(*args, **kwargs)

    def tracked_cell(dataset, row, *, role):
        cell_reads.append((dataset.name, int(row)))
        return real_cell(dataset, row, role=role)

    monkeypatch.setattr(module, "_qualify_vector", tracked_qualify)
    monkeypatch.setattr(module, "_bounded_utf8_item", tracked_cell)
    with module.FrameViewReader(path) as reader:
        baseline = reader.retained_bytes
        baseline_roots = reader.retained_root_count
        view = reader.read(5)
        assert view.metadata_raw["a_first"] == values[0]
        assert view.metadata_raw["z_alias"] == values[0]
        assert reader._scan_data_columns is not None
        assert (
            reader._scan_data_columns["a_first"]
            is reader._scan_data_columns["z_alias"]
        )
        assert reader.retained_bytes == (
            baseline + rows * np.dtype(object).itemsize
        )
        assert reader.retained_root_count == baseline_roots + 1
    assert cell_reads == [
        ("/entry/scan_data/a_first", 0),
        ("/entry/scan_data/a_first", 1),
    ]
    assert "/entry/scan_data/a_first" in qualified_roles
    assert "/entry/scan_data/z_alias" in qualified_roles


def test_same_hdf_object_with_distinct_dtype_or_transform_reads_distinct_roots(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "cache_key_transform_dtype.nxs"
    _one_dimensional_file(path)
    reads: list[int] = []
    roles: list[str] = []
    real_read = h5py.Dataset.read_direct

    def tracked_read(dataset, *args, **kwargs):
        reads.append(int(h5py.h5o.get_info(dataset.id).addr))
        return real_read(dataset, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "read_direct", tracked_read)
    with module.FrameViewReader(path) as reader:
        dataset = reader._entry["integrated_1d/q"]

        def validator(candidate, *, role):
            roles.append(role)
            return module._qualify_vector(
                candidate,
                role=role,
                max_items=module._MAX_AXIS_POINTS,
                max_bytes=module._MAX_1D_ROW_BYTES,
                numeric=True,
            )

        with module._ReaderArrayBundle(reader) as bundle:
            a64 = bundle.read_hdf(
                dataset,
                role="axis-a64",
                validator=lambda value: validator(value, role="axis-a64"),
                selection=None,
                final_dtype=np.dtype(np.float64),
                transform="variant-a",
                retain=True,
            )
            b64 = bundle.read_hdf(
                dataset,
                role="axis-b64",
                validator=lambda value: validator(value, role="axis-b64"),
                selection=None,
                final_dtype=np.dtype(np.float64),
                transform="variant-b",
                retain=True,
            )
            a32 = bundle.read_hdf(
                dataset,
                role="axis-a32",
                validator=lambda value: validator(value, role="axis-a32"),
                selection=None,
                final_dtype=np.dtype(np.float32),
                transform="variant-a",
                retain=True,
            )
            bundle.finish()
        assert len({id(a64), id(b64), id(a32)}) == 3
        assert len({id(physical_root_fact(value).root) for value in (
            a64, b64, a32,
        )}) == 3
    address = _object_addresses(path, ("entry/integrated_1d/q",))[
        "entry/integrated_1d/q"
    ]
    # One initial axis read plus three distinct cache-key reads.
    assert reads.count(address) == 4
    assert roles == ["axis-a64", "axis-b64", "axis-a32"]


def test_two_transient_rows_share_one_aggregate_bundle_ceiling_and_retry(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "two_transient_rows.nxs"
    points = 16
    _one_dimensional_file(path, points=points)
    with module.FrameViewReader(path) as admitted:
        baseline = admitted.retained_bytes
    row_bytes = points * np.dtype(np.float64).itemsize
    monkeypatch.setattr(
        module,
        "_MAX_READER_RETAINED_BYTES",
        baseline + 2 * row_bytes - 1,
    )
    addresses = _object_addresses(
        path,
        (
            "entry/integrated_1d/intensity",
            "entry/integrated_1d/sigma",
        ),
    )
    reads: list[int] = []
    real_read = h5py.Dataset.read_direct

    def tracked(dataset, *args, **kwargs):
        reads.append(int(h5py.h5o.get_info(dataset.id).addr))
        return real_read(dataset, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "read_direct", tracked)
    with module.FrameViewReader(path) as reader:
        prior_cache = dict(reader._read_cache)
        prior_roots = reader.retained_root_count
        prior_refs = reader.semantic_references
        reads.clear()
        with pytest.raises(ValueError, match="bytes exceed limit"):
            reader.read(5)
        assert reads == [addresses["entry/integrated_1d/intensity"]]
        assert reader._read_cache == prior_cache
        assert reader.retained_bytes == baseline
        assert reader.retained_root_count == prior_roots
        assert reader.semantic_references == prior_refs

        monkeypatch.setattr(
            reader._memory_authority,
            "_limit",
            baseline + 2 * row_bytes,
        )
        reads.clear()
        view = reader.read(5)
        assert reads == [
            addresses["entry/integrated_1d/intensity"],
            addresses["entry/integrated_1d/sigma"],
        ]
        intensity = physical_root_fact(view.intensity_1d)
        sigma = physical_root_fact(view.sigma_1d)
        assert intensity.root is not sigma.root
        assert intensity.nbytes == sigma.nbytes == row_bytes
        retained = reader._memory_authority.retained_roots
        assert all(intensity.root is not root for root in retained)
        assert all(sigma.root is not root for root in retained)


def test_aggregate_axis_limit_refuses_before_late_read_and_publishes_nothing(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "aggregate_axis_limit.nxs"
    points = 16
    _one_dimensional_file(path, points=points)
    _add_second_1d_mode(path, points=points)
    inventory_bytes = np.dtype(np.int64).itemsize
    one_axis_bytes = points * np.dtype(np.float64).itemsize
    monkeypatch.setattr(
        module,
        "_MAX_READER_RETAINED_BYTES",
        inventory_bytes + one_axis_bytes,
    )
    late_address = _object_addresses(
        path, ("entry/integrated_1d/q_oop/q",),
    )["entry/integrated_1d/q_oop/q"]
    late_reads: list[int] = []
    real_read_direct = h5py.Dataset.read_direct

    def guarded(dataset, *args, **kwargs):
        address = int(h5py.h5o.get_info(dataset.id).addr)
        if address == late_address:
            late_reads.append(address)
        return real_read_direct(dataset, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "read_direct", guarded)
    reader = module.FrameViewReader(path)
    with pytest.raises(ValueError, match="bytes exceed limit"):
        reader.__enter__()
    assert late_reads == []
    assert reader._h5 is None
    assert reader._hdf_owner_token is None
    assert reader._entry is None
    assert reader._g1_modes == {}
    assert reader._axis_1d_modes == {}
    assert reader.retained_bytes == 0
    assert reader.retained_root_count == 0
    assert reader.semantic_references == 0


def test_late_malformed_mode_cleans_enter_state_and_reopen_resets_owner(
    tmp_path,
) -> None:
    path = tmp_path / "late_malformed_mode.nxs"
    _one_dimensional_file(path)
    _add_second_1d_mode(path, malformed_intensity=True)
    reader = module.FrameViewReader(path)
    with pytest.raises(ValueError, match="1-D row stack"):
        reader.__enter__()
    assert reader._h5 is None
    assert reader._hdf_owner_token is None
    assert reader._entry is None
    assert reader._g1_modes == {}
    assert reader.retained_bytes == 0
    assert reader.retained_root_count == 0

    with h5py.File(path, "r+") as handle:
        child = handle["entry/integrated_1d/q_oop"]
        del child["intensity"]
        child.create_dataset(
            "intensity", data=np.arange(8, dtype=np.float32)[None, :],
        )
    with reader as opened:
        first_owner = opened._hdf_owner_token
        assert first_owner is not None
        assert opened.retained_bytes > 0
        view = opened.read(5, mode_1d="q_oop")
        np.testing.assert_allclose(view.intensity_1d, np.arange(8))
    assert reader._hdf_owner_token is None
    assert reader.retained_bytes == 0
    assert reader.retained_root_count == 0
    assert reader._axis_1d_modes == {}
    with reader as reopened:
        assert reopened._hdf_owner_token is not first_owner
        assert reopened.retained_bytes > 0
    assert reader.retained_bytes == 0

def test_hdf_open_failure_clears_owner_and_authority(tmp_path) -> None:
    reader = module.FrameViewReader(tmp_path / "missing.nxs")
    with pytest.raises(OSError):
        reader.__enter__()
    assert reader._h5 is None
    assert reader._hdf_owner_token is None
    assert reader._read_cache == {}
    assert reader.retained_bytes == 0
    assert reader.retained_root_count == 0
    assert reader.semantic_references == 0


@pytest.mark.parametrize("failure", ("lease", "cache"))
def test_enter_bundle_publication_baseexception_leaves_zero_owner_state(
    tmp_path, monkeypatch, failure,
) -> None:
    path = tmp_path / f"enter_publication_{failure}.nxs"
    _one_dimensional_file(path)
    reader = module.FrameViewReader(path)

    if failure == "lease":
        def fail_lease(_self, _authority, _semantic):
            raise _InjectedPublicationFailure("lease construction failed")

        monkeypatch.setattr(PhysicalRootLease, "__init__", fail_lease)
    else:
        _install_reader_cache_cas_cut(
            reader,
            monkeypatch,
            expected_phase=module._ReaderCachePhase.PUBLISHING,
            replacement_phase=module._ReaderCachePhase.OPENING,
            cut="before",
        )

    with pytest.raises(_InjectedPublicationFailure):
        reader.__enter__()
    assert reader._h5 is None
    assert reader._hdf_owner_token is None
    assert reader._entry is None
    assert reader._read_cache == {}
    assert reader.retained_bytes == 0
    assert reader.retained_root_count == 0
    assert reader.semantic_references == 0


@pytest.mark.parametrize("failure", ("lease", "cache"))
def test_read_bundle_finish_baseexception_preserves_exact_prior_cache(
    tmp_path, monkeypatch, failure,
) -> None:
    path = tmp_path / f"read_publication_{failure}.nxs"
    _one_dimensional_file(path)
    with h5py.File(path, "r+") as handle:
        scan = handle["entry"].create_group("scan_data")
        scan["frame_index"] = handle["entry/integrated_1d/frame_index"]
        scan.create_dataset("monitor", data=np.array([11.0], dtype=np.float32))

    reader = module.FrameViewReader(path)
    with reader:
        baseline_cache = dict(reader._read_cache)
        baseline_bytes = reader.retained_bytes
        baseline_roots = reader.retained_root_count
        baseline_refs = reader.semantic_references
        if failure == "lease":
            real_init = PhysicalRootLease.__init__

            def fail_lease(_self, _authority, _semantic):
                raise _InjectedPublicationFailure("lease construction failed")

            monkeypatch.setattr(PhysicalRootLease, "__init__", fail_lease)
        else:
            _install_reader_cache_cas_cut(
                reader,
                monkeypatch,
                expected_phase=module._ReaderCachePhase.PREPARING,
                replacement_phase=module._ReaderCachePhase.STAGED_PENDING,
                cut="before",
            )

        with pytest.raises(_InjectedPublicationFailure):
            reader.read(5)
        assert reader._scan_data_columns is None
        assert dict(reader._read_cache) == baseline_cache
        assert reader.retained_bytes == baseline_bytes
        assert reader.retained_root_count == baseline_roots
        assert reader.semantic_references == baseline_refs
        if failure == "lease":
            monkeypatch.setattr(PhysicalRootLease, "__init__", real_init)


def test_finish_mapping_lookup_baseexception_rolls_back_and_same_reader_retries(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "finish_mapping_lookup.nxs"
    _one_dimensional_file(path)
    real_prepared_leases = PhysicalRootExchange.prepared_leases
    real_getter = real_prepared_leases.fget
    assert real_getter is not None

    class LookupBomb(dict):
        def __getitem__(self, key):
            raise _InjectedPublicationFailure("lease lookup failed")

    def prepared_bomb(exchange):
        return LookupBomb(real_getter(exchange))

    with module.FrameViewReader(path) as reader:
        baseline_cache = dict(reader._read_cache)
        baseline_bytes = reader.retained_bytes
        baseline_roots = reader.retained_root_count
        baseline_refs = reader.semantic_references
        monkeypatch.setattr(
            PhysicalRootExchange,
            "prepared_leases",
            property(prepared_bomb),
        )
        with pytest.raises(_InjectedPublicationFailure, match="lease lookup"):
            reader.read(5)
        assert reader._scan_data_columns is None
        assert reader._read_cache == baseline_cache
        assert reader.retained_bytes == baseline_bytes
        assert reader.retained_root_count == baseline_roots
        assert reader.semantic_references == baseline_refs

        monkeypatch.setattr(
            PhysicalRootExchange,
            "prepared_leases",
            real_prepared_leases,
        )
        view = reader.read(5)
        np.testing.assert_allclose(
            view.intensity_1d, np.arange(8, dtype=np.float64),
        )
        assert reader.retained_bytes == baseline_bytes
        assert reader.retained_root_count == baseline_roots
        assert reader.semantic_references == baseline_refs
    assert reader.retained_bytes == 0
    assert reader.retained_root_count == 0
    assert reader.semantic_references == 0


def test_finish_recovers_commit_receipt_when_wrapper_raises_before_return(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "finish_post_commit_return.nxs"
    _one_dimensional_file(path)
    real_commit = PhysicalRootExchange.commit

    def post_commit_failure(exchange):
        real_commit(exchange)
        raise _InjectedPublicationFailure("commit return interrupted")

    with module.FrameViewReader(path) as reader:
        baseline_cache = dict(reader._read_cache)
        baseline_scan = reader._scan_data_columns
        baseline_bytes = reader.retained_bytes
        baseline_roots = reader.retained_root_count
        baseline_refs = reader.semantic_references
        monkeypatch.setattr(
            PhysicalRootExchange, "commit", post_commit_failure,
        )
        with pytest.raises(
            _InjectedPublicationFailure, match="commit return interrupted",
        ):
            reader.read(5)
        assert reader._read_cache == baseline_cache
        assert reader._scan_data_columns is baseline_scan
        assert reader.retained_bytes == baseline_bytes
        assert reader.retained_root_count == baseline_roots
        assert reader.semantic_references == baseline_refs

        monkeypatch.setattr(PhysicalRootExchange, "commit", real_commit)
        view = reader.read(5)
        np.testing.assert_allclose(
            view.intensity_1d, np.arange(8, dtype=np.float64),
        )
        assert reader.retained_bytes == baseline_bytes
        assert reader.retained_root_count == baseline_roots
        assert reader.semantic_references == baseline_refs
    assert reader.retained_bytes == 0
    assert reader.retained_root_count == 0
    assert reader.semantic_references == 0


@pytest.mark.parametrize("cut", ("before", "after"))
def test_finish_retries_transient_lease_release_at_both_fault_cuts(
    tmp_path, monkeypatch, cut: str,
) -> None:
    path = tmp_path / f"finish_release_retry_{cut}.nxs"
    _one_dimensional_file(path)

    with module.FrameViewReader(path) as reader:
        baseline_cache = dict(reader._read_cache)
        baseline_scan = reader._scan_data_columns
        baseline_bytes = reader.retained_bytes
        baseline_roots = reader.retained_root_count
        baseline_refs = reader.semantic_references
        authority = reader._memory_authority
        real_release = authority._release
        failed = False

        def release_then_interrupt(token):
            nonlocal failed
            if not failed:
                failed = True
                if cut == "after":
                    real_release(token)
                raise _InjectedPublicationFailure("release interrupted")
            real_release(token)

        monkeypatch.setattr(authority, "_release", release_then_interrupt)
        with pytest.raises(
            _InjectedPublicationFailure, match="release interrupted",
        ):
            reader.read(5)
        assert failed
        assert reader._read_cache == baseline_cache
        assert reader._scan_data_columns is baseline_scan
        assert reader.retained_bytes == baseline_bytes
        assert reader.retained_root_count == baseline_roots
        assert reader.semantic_references == baseline_refs
        _assert_reader_cache_matches_authority(reader)

        view = reader.read(5)
        np.testing.assert_allclose(
            view.intensity_1d, np.arange(8, dtype=np.float64),
        )
        assert reader.retained_bytes == baseline_bytes
        assert reader.retained_root_count == baseline_roots
        assert reader.semantic_references == baseline_refs
    assert reader.retained_bytes == 0
    assert reader.retained_root_count == 0
    assert reader.semantic_references == 0


def test_read_record_bundle_failure_is_all_or_none_and_output_transfers(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "record_bundle_atomic.nxs"
    _one_dimensional_file(path, scan_alias=False)
    _add_second_1d_mode(path)
    with h5py.File(path, "r+") as handle:
        scan = handle["entry"].create_group("scan_data")
        scan["frame_index"] = handle["entry/integrated_1d/frame_index"]
        scan.create_dataset("monitor", data=np.array([11.0], dtype=np.float32))
    late_address = _object_addresses(
        path, ("entry/integrated_1d/q_oop/intensity",),
    )["entry/integrated_1d/q_oop/intensity"]
    real_read_direct = h5py.Dataset.read_direct
    fail = True

    def guarded(dataset, *args, **kwargs):
        if fail and int(h5py.h5o.get_info(dataset.id).addr) == late_address:
            raise _InjectedPublicationFailure("late mode read failed")
        return real_read_direct(dataset, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "read_direct", guarded)
    with module.FrameViewReader(path) as reader:
        baseline_bytes = reader.retained_bytes
        baseline_roots = reader.retained_root_count
        baseline_refs = reader.semantic_references
        baseline_cache = dict(reader._read_cache)
        with pytest.raises(_InjectedPublicationFailure):
            reader.read_record(5)
        assert reader._scan_data_columns is None
        assert reader._read_cache == baseline_cache
        assert reader.retained_bytes == baseline_bytes
        assert reader.retained_root_count == baseline_roots
        assert reader.semantic_references == baseline_refs

        fail = False
        record = reader.read_record(5)
        assert record.results_1d["q_oop"].intensity_1d is not None
        assert reader._scan_data_columns is not None
        retained_after = reader.retained_bytes
        assert retained_after == baseline_bytes + np.dtype(np.float32).itemsize
        retained_roots = reader._memory_authority.retained_roots
        assert all(
            physical_root_fact(view.intensity_1d).root is not root
            for view in record.results_1d.values()
            for root in retained_roots
        )
        assert all(
            view.sigma_1d is None
            or physical_root_fact(view.sigma_1d).root is not root
            for view in record.results_1d.values()
            for root in retained_roots
        )
    assert reader.retained_bytes == 0
    assert reader.retained_root_count == 0
    assert reader.semantic_references == 0
    assert reader._read_cache == {}
    assert reader._scan_data_columns is None
    assert reader._scan_data_items == ()
    assert reader._axis_1d is None
    assert reader._axis_1d_modes == {}
    assert reader._map_1d == {}
    assert reader._map_1d_modes == {}


@pytest.mark.parametrize("method", ("commit", "accept"))
@pytest.mark.parametrize("cut", ("before", "after"))
def test_reader_cache_exchange_fault_cuts_are_exact_and_retryable(
    tmp_path, monkeypatch, method: str, cut: str,
) -> None:
    path = tmp_path / f"reader_exchange_{method}_{cut}.nxs"
    _one_dimensional_file(path)
    _add_scan_monitor(path)
    with module.FrameViewReader(path) as reader:
        baseline = reader._snapshot_reader_cache_state()
        baseline_items = tuple(baseline.entries.items())
        baseline_scan = baseline.scan_data_columns
        baseline_accounting = (
            reader.retained_bytes,
            reader.retained_root_count,
            reader.semantic_references,
        )
        _install_exchange_cut(monkeypatch, method, cut)
        with pytest.raises(
            _InjectedPublicationFailure,
            match=f"exchange {method} {cut}",
        ):
            reader.read(5)
        current = reader._snapshot_reader_cache_state()
        assert current.phase is module._ReaderCachePhase.OPEN
        assert current.journal is None
        if method == "commit":
            assert current.entries is baseline.entries
            assert current.scan_data_columns is baseline_scan
            _assert_cache_items_exact(
                tuple(current.entries.items()), baseline_items,
            )
            assert (
                reader.retained_bytes,
                reader.retained_root_count,
                reader.semantic_references,
            ) == baseline_accounting
        else:
            assert current.entries is not baseline.entries
            assert current.scan_data_columns is not None
            _assert_cache_items_exact(
                tuple(current.entries.items())[: len(baseline_items)],
                baseline_items,
            )
            _assert_reader_cache_matches_authority(reader)
        monkeypatch.undo()
        view = reader.read(5)
        np.testing.assert_allclose(
            view.intensity_1d, np.arange(8, dtype=np.float64),
        )
        _assert_reader_cache_matches_authority(reader)


@pytest.mark.parametrize("cut", ("before", "after"))
def test_reader_pointer_terminal_swap_fault_is_accepted_without_split_brain(
    tmp_path, monkeypatch, cut: str,
) -> None:
    path = tmp_path / f"reader_pointer_{cut}.nxs"
    _one_dimensional_file(path)
    _add_scan_monitor(path)
    with module.FrameViewReader(path) as reader:
        baseline = reader._snapshot_reader_cache_state()
        _install_reader_cache_cas_cut(
            reader,
            monkeypatch,
            expected_phase=module._ReaderCachePhase.PUBLISHING,
            replacement_phase=module._ReaderCachePhase.OPEN,
            cut=cut,
        )
        with pytest.raises(
            _InjectedPublicationFailure,
            match=f"publishing->open {cut}",
        ):
            reader.read(5)
        current = reader._snapshot_reader_cache_state()
        assert current.phase is module._ReaderCachePhase.OPEN
        assert current.journal is None
        assert current.entries is not baseline.entries
        assert current.scan_data_columns is not None
        _assert_reader_cache_matches_authority(reader)
        monkeypatch.undo()
        assert reader.read(5).metadata_raw["monitor"] == 11.0
        _assert_reader_cache_matches_authority(reader)


def test_reader_cache_rollback_restores_exact_prior_cache_scan_and_order(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "reader_exact_prior_restore.nxs"
    _one_dimensional_file(path, rows=2)
    _add_scan_monitor(path, rows=2)
    with module.FrameViewReader(path) as reader:
        assert reader.read(5).metadata_raw["monitor"] == 11.0
        baseline = reader._snapshot_reader_cache_state()
        baseline_items = tuple(baseline.entries.items())
        baseline_scan = baseline.scan_data_columns
        baseline_accounting = (
            reader.retained_bytes,
            reader.retained_root_count,
            reader.semantic_references,
        )
        _install_exchange_cut(monkeypatch, "commit", "after")
        with pytest.raises(_InjectedPublicationFailure, match="commit after"):
            reader.read(6)
        restored = reader._snapshot_reader_cache_state()
        assert restored.entries is baseline.entries
        assert restored.scan_data_columns is baseline_scan
        _assert_cache_items_exact(
            tuple(restored.entries.items()), baseline_items,
        )
        assert (
            reader.retained_bytes,
            reader.retained_root_count,
            reader.semantic_references,
        ) == baseline_accounting
        monkeypatch.undo()
        assert reader.read(6).metadata_raw["monitor"] == 12.0


def test_reader_cache_accept_preserves_survivors_and_publishes_one_graph(
    tmp_path,
) -> None:
    path = tmp_path / "reader_accept_graph.nxs"
    _one_dimensional_file(path)
    _add_scan_monitor(path)
    with module.FrameViewReader(path) as reader:
        baseline = reader._snapshot_reader_cache_state()
        survivors = tuple(baseline.entries.items())
        baseline_refs = reader.semantic_references
        view = reader.read(5)
        assert view.metadata_raw["monitor"] == 11.0
        accepted = reader._snapshot_reader_cache_state()
        assert accepted.entries is not baseline.entries
        _assert_cache_items_exact(
            tuple(accepted.entries.items())[: len(survivors)], survivors,
        )
        assert reader.semantic_references == baseline_refs + 1
        assert accepted.scan_data_columns is not None
        monitor = accepted.scan_data_columns["monitor"]
        assert any(
            monitor is entry.array for entry in accepted.entries.values()
        )
        _assert_reader_cache_matches_authority(reader)
        returned_root = physical_root_fact(view.intensity_1d).root
        assert all(
            returned_root is not root
            for root in reader._memory_authority.retained_roots
        )


def test_reader_cache_diagnostic_hides_leases_and_terminal_bundle_compacts(
    tmp_path,
) -> None:
    path = tmp_path / "reader_values_only_diagnostic.nxs"
    _one_dimensional_file(path)
    with module.FrameViewReader(path) as reader:
        state = reader._snapshot_reader_cache_state()
        projection = reader._read_cache
        assert tuple(projection) == tuple(state.entries)
        for key, value in projection.items():
            assert type(value) is np.ndarray
            assert value is state.entries[key].array
            assert not isinstance(value, PhysicalRootLease)
        with pytest.raises(TypeError):
            projection[("forged",)] = np.empty(1)

        bundle = module._ReaderArrayBundle(reader)
        marker = bundle._marker
        bundle.finish()
        assert bundle._terminal == "accepted"
        assert bundle._reader is None
        assert bundle._owner is None
        assert bundle._marker is None
        assert bundle._exchange is None
        assert bundle._local == {}
        assert bundle.pending_scan_data_columns is None
        current = reader._snapshot_reader_cache_state()
        assert _reader_marker_is_present(current, marker)
        _assert_reader_cache_matches_authority(reader)


def test_reader_cache_builder_failure_is_lock_free_and_restores_exact_prior(
    tmp_path,
) -> None:
    path = tmp_path / "reader_failing_cache_builder.nxs"
    _one_dimensional_file(path)
    _add_scan_monitor(path)
    with module.FrameViewReader(path) as reader:
        baseline = reader._snapshot_reader_cache_state()
        baseline_items = tuple(baseline.entries.items())
        baseline_scan = baseline.scan_data_columns
        baseline_accounting = (
            reader.retained_bytes,
            reader.retained_root_count,
            reader.semantic_references,
        )
        observations: list[tuple[str, bool, bool]] = []

        class FailingCache(dict):
            def __setitem__(self, key, value):
                observations.append((
                    "setitem",
                    reader._reader_cache_lock._is_owned(),
                    reader._memory_authority._lock._is_owned(),
                ))
                raise _InjectedPublicationFailure("cache insertion failed")

            def __del__(self):
                observations.append((
                    "finalizer",
                    reader._reader_cache_lock._is_owned(),
                    reader._memory_authority._lock._is_owned(),
                ))

        reader._reader_cache_builder_factory = FailingCache
        with pytest.raises(
            _InjectedPublicationFailure, match="cache insertion failed",
        ):
            reader.read(5)
        gc.collect()
        assert ("setitem", False, False) in observations
        assert ("finalizer", False, False) in observations
        restored = reader._snapshot_reader_cache_state()
        assert restored.phase is module._ReaderCachePhase.OPEN
        assert restored.journal is None
        assert restored.entries is baseline.entries
        assert restored.scan_data_columns is baseline_scan
        _assert_cache_items_exact(tuple(restored.entries.items()), baseline_items)
        assert (
            reader.retained_bytes,
            reader.retained_root_count,
            reader.semantic_references,
        ) == baseline_accounting

        reader._reader_cache_builder_factory = dict
        assert reader.read(5).metadata_raw["monitor"] == 11.0
        _assert_reader_cache_matches_authority(reader)


@pytest.mark.parametrize(
    "behavior", ("drop", "reorder", "extra", "substitute"),
)
def test_reader_cache_builder_cannot_silently_change_values_graph(
    tmp_path, behavior: str,
) -> None:
    path = tmp_path / f"reader_builder_{behavior}.nxs"
    _one_dimensional_file(path)
    _add_scan_monitor(path)
    with module.FrameViewReader(path) as reader:
        baseline = reader._snapshot_reader_cache_state()
        observed: list[object] = []

        class AdversarialBuilder(dict):
            def __init__(self, *args, **kwargs):
                assert args == () and kwargs == {}
                super().__init__()
                if behavior == "extra":
                    dict.__setitem__(
                        self, ("extra",), np.zeros(1, dtype=np.float64),
                    )

            def __setitem__(self, key, value):
                observed.append(value)
                assert type(value) is np.ndarray
                assert not isinstance(value, PhysicalRootLease)
                assert not isinstance(value, module._ReaderCacheEntry)
                if behavior == "drop":
                    return
                if behavior == "substitute":
                    value = np.array(value, copy=True)
                dict.__setitem__(self, key, value)
                if behavior == "reorder" and len(self) > 1:
                    items = tuple(dict.items(self))
                    dict.clear(self)
                    for item_key, item_value in reversed(items):
                        dict.__setitem__(self, item_key, item_value)

        reader._reader_cache_builder_factory = AdversarialBuilder
        with pytest.raises(RuntimeError, match="cache builder changed"):
            reader.read(5)
        assert observed
        restored = reader._snapshot_reader_cache_state()
        assert restored.phase is module._ReaderCachePhase.OPEN
        assert restored.journal is None
        assert restored.entries is baseline.entries
        assert restored.scan_data_columns is baseline.scan_data_columns
        _assert_reader_cache_matches_authority(reader)

        reader._reader_cache_builder_factory = dict
        assert reader.read(5).metadata_raw["monitor"] == 11.0
        _assert_reader_cache_matches_authority(reader)


def test_permuted_complete_prepared_lease_mapping_rolls_back_before_commit(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "reader_permuted_prepared_leases.nxs"
    _one_dimensional_file(path)
    _add_scan_monitor(path)
    real_property = PhysicalRootExchange.prepared_leases
    real_getter = real_property.fget
    assert real_getter is not None

    def permuted(exchange):
        admitted = real_getter(exchange)
        items = tuple(admitted.items())
        if len(items) < 2:
            return admitted
        values = tuple(value for _key, value in items)
        return {
            key: values[(index + 1) % len(values)]
            for index, (key, _value) in enumerate(items)
        }

    with module.FrameViewReader(path) as reader:
        baseline = reader._snapshot_reader_cache_state()
        baseline_accounting = (
            reader.retained_bytes,
            reader.retained_root_count,
            reader.semantic_references,
        )
        monkeypatch.setattr(
            PhysicalRootExchange,
            "prepared_leases",
            property(permuted),
        )
        with pytest.raises(RuntimeError, match="prepared lease is invalid"):
            reader.read(5)
        restored = reader._snapshot_reader_cache_state()
        assert restored.entries is baseline.entries
        assert restored.scan_data_columns is baseline.scan_data_columns
        assert (
            reader.retained_bytes,
            reader.retained_root_count,
            reader.semantic_references,
        ) == baseline_accounting
        _assert_reader_cache_matches_authority(reader)

        monkeypatch.setattr(
            PhysicalRootExchange, "prepared_leases", real_property,
        )
        assert reader.read(5).metadata_raw["monitor"] == 11.0
        _assert_reader_cache_matches_authority(reader)


def test_reader_bundle_hash_and_equality_callbacks_run_without_reader_locks(
    tmp_path,
) -> None:
    path = tmp_path / "reader_callback_lock_freedom.nxs"
    _one_dimensional_file(path)
    with module.FrameViewReader(path) as reader:
        observations: list[tuple[str, bool, bool]] = []

        class CallbackKey:
            def __init__(self, name: str) -> None:
                self.name = name

            def __hash__(self) -> int:
                observations.append((
                    "hash",
                    reader._reader_cache_lock._is_owned(),
                    reader._memory_authority._lock._is_owned(),
                ))
                return 17

            def __eq__(self, other: object) -> bool:
                observations.append((
                    "eq",
                    reader._reader_cache_lock._is_owned(),
                    reader._memory_authority._lock._is_owned(),
                ))
                return self is other

        with module._ReaderArrayBundle(reader) as bundle:
            first = np.arange(2, dtype=np.float64)
            second = np.arange(3, dtype=np.float64)
            bundle.adopt((CallbackKey("first"),), first, retain=False)
            bundle.adopt((CallbackKey("second"),), second, retain=False)
            bundle.finish()
        kinds = {kind for kind, _reader_lock, _authority_lock in observations}
        assert kinds == {"hash", "eq"}
        assert all(
            not reader_lock and not authority_lock
            for _kind, reader_lock, authority_lock in observations
        )
        _assert_reader_cache_matches_authority(reader)


def test_live_building_owner_blocks_second_read_and_close_before_mutation(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "reader_live_building_owner.nxs"
    _one_dimensional_file(path)
    reader = module.FrameViewReader(path)
    reader.__enter__()
    handle = reader._h5
    assert handle is not None
    authority_state = reader._memory_authority._snapshot_state()
    entered = Event()
    release = Event()
    real_claim = reader._bundle_exchange_claim
    blocked_once = False

    def held_claim(marker, owner, nbytes):
        nonlocal blocked_once
        if not blocked_once:
            blocked_once = True
            entered.set()
            assert release.wait(1.0)
        return real_claim(marker, owner, nbytes)

    monkeypatch.setattr(reader, "_bundle_exchange_claim", held_claim)
    results: list[object] = []
    failures: list[BaseException] = []

    def run_read() -> None:
        try:
            results.append(reader.read(5))
        except BaseException as error:  # pragma: no cover - asserted below
            failures.append(error)

    worker = Thread(target=run_read)
    worker.start()
    assert entered.wait(1.0)
    busy = reader._snapshot_reader_cache_state()
    assert busy.phase is module._ReaderCachePhase.BUILDING
    try:
        with pytest.raises(RuntimeError, match="publication is busy"):
            reader.read(5)
        with pytest.raises(RuntimeError, match="publication is busy"):
            reader.__exit__(None, None, None)
        with pytest.raises(RuntimeError, match="accounting is busy"):
            _ = reader.retained_bytes
        assert reader._h5 is handle
        assert bool(handle.id.valid)
        assert reader._memory_authority._snapshot_state() is authority_state
    finally:
        release.set()
    worker.join(1.0)
    assert not worker.is_alive()
    assert failures == []
    assert len(results) == 1
    _assert_reader_cache_matches_authority(reader)
    reader.__exit__(None, None, None)
    assert reader._snapshot_reader_cache_state().phase is module._ReaderCachePhase.CLOSED


@pytest.mark.parametrize("cut", ("before", "after"))
def test_fresh_read_recovers_abandoned_open_to_building_admission_cut(
    tmp_path, monkeypatch, cut: str,
) -> None:
    path = tmp_path / f"reader_abandoned_admission_{cut}.nxs"
    _one_dimensional_file(path)
    with module.FrameViewReader(path) as reader:
        _install_reader_cache_cas_cut(
            reader,
            monkeypatch,
            expected_phase=module._ReaderCachePhase.OPEN,
            replacement_phase=module._ReaderCachePhase.BUILDING,
            cut=cut,
        )
        with pytest.raises(
            _InjectedPublicationFailure,
            match=f"open->building {cut}",
        ):
            reader.read(5)
        gc.collect()
        abandoned = reader._snapshot_reader_cache_state()
        if cut == "before":
            assert abandoned.phase is module._ReaderCachePhase.OPEN
            assert abandoned.journal is None
        else:
            assert abandoned.phase is module._ReaderCachePhase.BUILDING
            assert abandoned.journal is not None
            assert abandoned.journal.owner_ref() is None
        monkeypatch.undo()
        np.testing.assert_allclose(
            reader.read(5).intensity_1d,
            np.arange(8, dtype=np.float64),
        )
        _assert_reader_cache_matches_authority(reader)


def test_dead_bundle_recovery_failure_retains_exact_journal_for_retry(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "reader_dead_recovery_retry.nxs"
    _one_dimensional_file(path)
    with module.FrameViewReader(path) as reader:
        bundle = module._ReaderArrayBundle(reader)
        marker = bundle._marker
        del bundle
        gc.collect()
        abandoned = reader._snapshot_reader_cache_state()
        assert abandoned.phase is module._ReaderCachePhase.BUILDING
        assert abandoned.journal is not None
        assert abandoned.journal.owner_ref() is None
        exchange = abandoned.journal.exchange
        assert exchange is not None
        real_rollback = PhysicalRootExchange.rollback
        failed = False

        def fail_once(candidate):
            nonlocal failed
            if candidate is exchange and not failed:
                failed = True
                raise _InjectedPublicationFailure("dead recovery interrupted")
            return real_rollback(candidate)

        monkeypatch.setattr(PhysicalRootExchange, "rollback", fail_once)
        with pytest.raises(
            _InjectedPublicationFailure, match="dead recovery interrupted",
        ):
            reader.read(5)
        retained = reader._snapshot_reader_cache_state()
        assert retained.journal is not None
        assert retained.journal.marker is marker
        assert retained.journal.owner_ref() is None

        np.testing.assert_allclose(
            reader.read(5).intensity_1d,
            np.arange(8, dtype=np.float64),
        )
        _assert_reader_cache_matches_authority(reader)


def test_dead_blocked_bundle_is_retained_and_never_auto_recovered(
    tmp_path,
) -> None:
    path = tmp_path / "reader_dead_blocked.nxs"
    _one_dimensional_file(path)
    with module.FrameViewReader(path) as reader:
        bundle = module._ReaderArrayBundle(reader)
        marker = bundle._marker
        source = reader._snapshot_reader_cache_state()
        blocked = module._reader_cache_state_with(
            source, phase=module._ReaderCachePhase.BLOCKED,
        )
        assert reader._cas_reader_cache_state(source, blocked)
        del bundle
        gc.collect()
        assert blocked.journal is not None
        assert blocked.journal.owner_ref() is None
        with pytest.raises(RuntimeError, match="unrecoverable"):
            reader.read(5)
        assert reader._snapshot_reader_cache_state() is blocked

        assert reader._cas_reader_cache_state(blocked, source)
        assert reader._recover_array_bundle(marker) == "rolled-back"
        _assert_reader_cache_matches_authority(reader)


@pytest.mark.parametrize("callback_kind", ("hash", "eq"))
def test_reentrant_key_callback_drift_is_retained_without_overwrite(
    tmp_path, callback_kind: str,
) -> None:
    path = tmp_path / f"reader_reentrant_{callback_kind}.nxs"
    _one_dimensional_file(path)
    with module.FrameViewReader(path) as reader:
        bundle = module._ReaderArrayBundle(reader)
        installed: dict[str, object] = {}
        armed = callback_kind == "hash"

        class ReentrantKey:
            def __hash__(self) -> int:
                if armed and callback_kind == "hash" and not installed:
                    install_third()
                return 29

            def __eq__(self, other: object) -> bool:
                if armed and callback_kind == "eq" and not installed:
                    install_third()
                return self is other

        def install_third() -> None:
            source = reader._snapshot_reader_cache_state()
            assert not reader._reader_cache_lock._is_owned()
            assert not reader._memory_authority._lock._is_owned()
            third = module._reader_cache_state_with(
                source, phase=module._ReaderCachePhase.BLOCKED,
            )
            assert reader._cas_reader_cache_state(source, third)
            installed.update(source=source, third=third)

        first_key = (ReentrantKey(),)
        if callback_kind == "eq":
            bundle.adopt(first_key, np.arange(2), retain=False)
            armed = True
        with pytest.raises(RuntimeError):
            bundle.adopt((ReentrantKey(),), np.arange(3), retain=False)
        assert installed
        source = installed["source"]
        third = installed["third"]
        assert reader._snapshot_reader_cache_state() is third
        assert third.phase is module._ReaderCachePhase.BLOCKED
        assert third.journal is source.journal
        assert reader._cas_reader_cache_state(third, source)
        bundle.rollback()
        _assert_reader_cache_matches_authority(reader)


def test_equal_colliding_local_hit_cannot_hide_reentrant_state_drift(
    tmp_path,
) -> None:
    path = tmp_path / "reader_equal_local_hit_drift.nxs"
    _one_dimensional_file(path)
    with module.FrameViewReader(path) as reader:
        bundle = module._ReaderArrayBundle(reader)
        installed: dict[str, object] = {}
        armed = False

        class EqualCollision:
            def __hash__(self) -> int:
                return 41

            def __eq__(self, other: object) -> bool:
                if armed and not installed:
                    source = reader._snapshot_reader_cache_state()
                    assert not reader._reader_cache_lock._is_owned()
                    assert not reader._memory_authority._lock._is_owned()
                    third = module._reader_cache_state_with(
                        source, phase=module._ReaderCachePhase.BLOCKED,
                    )
                    assert reader._cas_reader_cache_state(source, third)
                    installed.update(source=source, third=third)
                return True

        first_key = (EqualCollision(),)
        first_value = np.arange(2, dtype=np.float64)
        bundle.adopt(first_key, first_value, retain=False)
        prior_local = bundle._local
        prior_items = tuple(prior_local.items())
        armed = True
        with pytest.raises(RuntimeError, match="no longer building"):
            bundle.cached((EqualCollision(),))
        source = installed["source"]
        third = installed["third"]
        assert reader._snapshot_reader_cache_state() is third
        assert third.journal is source.journal
        assert bundle._local is prior_local
        assert tuple(bundle._local.items()) == prior_items

        assert reader._cas_reader_cache_state(third, source)
        bundle.rollback()
        _assert_reader_cache_matches_authority(reader)


def test_local_cow_insert_late_hash_cut_retains_exact_prior_mapping(
    tmp_path,
) -> None:
    path = tmp_path / "reader_local_insert_drift.nxs"
    _one_dimensional_file(path)
    with module.FrameViewReader(path) as reader:
        bundle = module._ReaderArrayBundle(reader)
        bundle.adopt(("prior",), np.arange(2), retain=False)
        prior_local = bundle._local
        prior_items = tuple(prior_local.items())
        source = reader._snapshot_reader_cache_state()
        installed: dict[str, object] = {}

        class LateHash:
            def __hash__(self) -> int:
                if not installed:
                    current = reader._snapshot_reader_cache_state()
                    assert current is source
                    assert not reader._reader_cache_lock._is_owned()
                    assert not reader._memory_authority._lock._is_owned()
                    third = module._reader_cache_state_with(
                        current, phase=module._ReaderCachePhase.BLOCKED,
                    )
                    assert reader._cas_reader_cache_state(current, third)
                    installed["third"] = third
                return 43

        with pytest.raises(RuntimeError, match="no longer building"):
            bundle._publish_local(
                prior_local,
                (LateHash(),),
                (np.arange(3), object(), False),
            )
        third = installed["third"]
        assert reader._snapshot_reader_cache_state() is third
        assert third.journal is source.journal
        assert bundle._local is prior_local
        assert tuple(bundle._local.items()) == prior_items

        assert reader._cas_reader_cache_state(third, source)
        bundle.rollback()
        _assert_reader_cache_matches_authority(reader)


def test_accepting_driver_is_single_owner_and_other_entries_fail_promptly(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "reader_accept_concurrency.nxs"
    _one_dimensional_file(path)
    reader = module.FrameViewReader(path)
    reader.__enter__()
    entered = Event()
    release = Event()
    real_accept = PhysicalRootExchange.accept

    def held_accept(exchange):
        entered.set()
        assert release.wait(1.0)
        return real_accept(exchange)

    monkeypatch.setattr(PhysicalRootExchange, "accept", held_accept)
    results: list[object] = []
    failures: list[BaseException] = []

    def run_read() -> None:
        try:
            results.append(reader.read(5))
        except BaseException as error:  # pragma: no cover - asserted below
            failures.append(error)

    worker = Thread(target=run_read)
    worker.start()
    assert entered.wait(1.0)
    state = reader._snapshot_reader_cache_state()
    assert state.phase is module._ReaderCachePhase.ACCEPTING
    assert state.journal is not None
    assert state.journal.owner_ref() is not None
    assert state.journal.driver_ref() is not None
    try:
        with pytest.raises(RuntimeError, match="publication is busy"):
            reader.read(5)
        with pytest.raises(RuntimeError, match="publication is busy"):
            reader.__exit__(None, None, None)
        with pytest.raises(RuntimeError, match="owner is busy"):
            reader._recover_array_bundle(state.journal.marker)
    finally:
        release.set()
    worker.join(1.0)
    assert not worker.is_alive()
    assert failures == []
    assert len(results) == 1
    _assert_reader_cache_matches_authority(reader)
    reader.__exit__(None, None, None)


def test_reader_cache_unknown_third_state_retains_journal_and_refuses(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "reader_third_state.nxs"
    _one_dimensional_file(path)
    _add_scan_monitor(path)
    reader = module.FrameViewReader(path)
    reader.__enter__()
    real_cas = reader._cas_reader_cache_state
    captured: dict[str, object] = {}

    def install_third(expected, replacement):
        if (
            replacement.phase is module._ReaderCachePhase.STAGED_PENDING
            and "third" not in captured
        ):
            third = module._reader_cache_state_with(
                expected, phase=module._ReaderCachePhase.BLOCKED,
            )
            captured.update(expected=expected, third=third)
            assert real_cas(expected, third)
            return False
        return real_cas(expected, replacement)

    monkeypatch.setattr(reader, "_cas_reader_cache_state", install_third)
    with pytest.raises(RuntimeError, match="blocked"):
        reader.read(5)
    third = captured["third"]
    expected = captured["expected"]
    assert reader._snapshot_reader_cache_state() is third
    assert third.journal is not None
    assert third.journal.exchange.phase is PhysicalRootExchangePhase.STAGED
    with pytest.raises(RuntimeError, match="busy"):
        _ = reader._read_cache
    with pytest.raises(RuntimeError, match="blocked"):
        reader._recover_array_bundle(third.journal.marker)

    monkeypatch.setattr(reader, "_cas_reader_cache_state", real_cas)
    assert real_cas(third, expected)
    assert reader._recover_array_bundle(expected.journal.marker) == "rolled-back"
    reader.__exit__(None, None, None)
    assert reader._snapshot_reader_cache_state().phase is module._ReaderCachePhase.CLOSED


@pytest.mark.parametrize("method", ("prepare", "commit", "accept"))
@pytest.mark.parametrize("cut", ("before", "after"))
def test_reader_cache_close_exchange_faults_retain_custody_and_retry(
    tmp_path, monkeypatch, method: str, cut: str,
) -> None:
    path = tmp_path / f"reader_close_{method}_{cut}.nxs"
    _one_dimensional_file(path)
    reader = module.FrameViewReader(path)
    reader.__enter__()
    assert reader.retained_bytes > 0
    real_method = _install_exchange_cut(monkeypatch, method, cut)
    with pytest.raises(
        _InjectedPublicationFailure,
        match=f"exchange {method} {cut}",
    ):
        reader.__exit__(None, None, None)
    pending = reader._snapshot_reader_cache_state()
    assert pending.journal is not None
    assert pending.journal.intent is module._ReaderCacheDirection.CLOSED
    with pytest.raises(RuntimeError, match="busy"):
        _ = reader._read_cache
    monkeypatch.setattr(PhysicalRootExchange, method, real_method)
    reader.__exit__(None, None, None)
    assert reader._snapshot_reader_cache_state().phase is module._ReaderCachePhase.CLOSED
    assert reader.retained_bytes == 0
    assert reader.retained_root_count == 0
    assert reader.semantic_references == 0


@pytest.mark.parametrize("cut", ("before", "after"))
def test_reader_cache_authority_close_fault_is_retryable(
    tmp_path, monkeypatch, cut: str,
) -> None:
    path = tmp_path / f"reader_authority_close_{cut}.nxs"
    _one_dimensional_file(path)
    reader = module.FrameViewReader(path)
    reader.__enter__()
    authority = reader._memory_authority
    real_close = authority.close
    fired = False

    def close_cut():
        nonlocal fired
        if not fired:
            fired = True
            if cut == "after":
                real_close()
            raise _InjectedPublicationFailure(f"authority close {cut}")
        return real_close()

    monkeypatch.setattr(authority, "close", close_cut)
    with pytest.raises(
        _InjectedPublicationFailure, match=f"authority close {cut}",
    ):
        reader.__exit__(None, None, None)
    pending = reader._snapshot_reader_cache_state()
    assert pending.phase in {
        module._ReaderCachePhase.CLOSE_AUTHORITY,
        module._ReaderCachePhase.CLOSE_HDF,
    }
    assert pending.journal is not None
    assert pending.journal.intent is module._ReaderCacheDirection.CLOSED
    reader.__exit__(None, None, None)
    assert reader._snapshot_reader_cache_state().phase is module._ReaderCachePhase.CLOSED
    assert reader.retained_bytes == 0


@pytest.mark.parametrize("cut", ("before", "after"))
def test_reader_cache_close_pointer_fault_is_retryable(
    tmp_path, monkeypatch, cut: str,
) -> None:
    path = tmp_path / f"reader_close_pointer_{cut}.nxs"
    _one_dimensional_file(path)
    reader = module.FrameViewReader(path)
    reader.__enter__()
    _install_reader_cache_cas_cut(
        reader,
        monkeypatch,
        expected_phase=module._ReaderCachePhase.CLOSE_HDF,
        replacement_phase=module._ReaderCachePhase.CLOSED,
        cut=cut,
    )
    with pytest.raises(
        _InjectedPublicationFailure,
        match=f"close-hdf->closed {cut}",
    ):
        reader.__exit__(None, None, None)
    monkeypatch.undo()
    reader.__exit__(None, None, None)
    assert reader._snapshot_reader_cache_state().phase is module._ReaderCachePhase.CLOSED
    assert reader.retained_bytes == 0


@pytest.mark.parametrize("cut", ("before", "after"))
def test_reader_cache_close_admission_precedes_hdf_and_authority_mutation(
    tmp_path, monkeypatch, cut: str,
) -> None:
    path = tmp_path / f"reader_close_admission_{cut}.nxs"
    _one_dimensional_file(path)
    reader = module.FrameViewReader(path)
    reader.__enter__()
    handle = reader._h5
    assert handle is not None
    authority = reader._memory_authority
    real_cas = reader._cas_reader_cache_state
    real_exchange = authority.exchange
    real_hdf_close = h5py.File.close
    fired = False
    calls = {"exchange": 0, "hdf_close": 0}

    def cut_cas(expected, replacement):
        nonlocal fired
        if (
            not fired
            and expected.phase is module._ReaderCachePhase.OPEN
            and replacement.phase is module._ReaderCachePhase.CLOSE_ADMITTED
        ):
            fired = True
            if cut == "after":
                assert real_cas(expected, replacement)
            raise _InjectedPublicationFailure(f"close admission {cut}")
        return real_cas(expected, replacement)

    def tracked_exchange(victims):
        calls["exchange"] += 1
        return real_exchange(victims)

    def tracked_hdf_close(candidate):
        if candidate is handle:
            calls["hdf_close"] += 1
        return real_hdf_close(candidate)

    monkeypatch.setattr(reader, "_cas_reader_cache_state", cut_cas)
    monkeypatch.setattr(authority, "exchange", tracked_exchange)
    monkeypatch.setattr(h5py.File, "close", tracked_hdf_close)
    with pytest.raises(
        _InjectedPublicationFailure, match=f"close admission {cut}",
    ):
        reader.__exit__(None, None, None)
    current = reader._snapshot_reader_cache_state()
    assert current.phase is (
        module._ReaderCachePhase.OPEN
        if cut == "before"
        else module._ReaderCachePhase.CLOSE_ADMITTED
    )
    assert calls == {"exchange": 0, "hdf_close": 0}
    assert bool(handle.id.valid)
    assert not authority._snapshot_state().closed

    monkeypatch.setattr(reader, "_cas_reader_cache_state", real_cas)
    reader.__exit__(None, None, None)
    assert calls == {"exchange": 1, "hdf_close": 1}
    assert reader._snapshot_reader_cache_state().phase is module._ReaderCachePhase.CLOSED


@pytest.mark.parametrize("cut", ("before", "after"))
def test_reader_cache_hdf_close_fault_is_exactly_retryable(
    tmp_path, monkeypatch, cut: str,
) -> None:
    path = tmp_path / f"reader_hdf_close_{cut}.nxs"
    _one_dimensional_file(path)
    reader = module.FrameViewReader(path)
    reader.__enter__()
    handle = reader._h5
    assert handle is not None
    authority = reader._memory_authority
    prior_token = reader._hdf_owner_token
    prior_state = reader._snapshot_reader_cache_state()
    prior_keys = tuple(prior_state.entries)
    prior_leases = tuple(
        entry.lease for entry in prior_state.entries.values()
    )
    real_close = h5py.File.close
    fired = False

    def close_cut(candidate):
        nonlocal fired
        if candidate is handle and not fired:
            fired = True
            if cut == "after":
                real_close(candidate)
            raise _InjectedPublicationFailure(f"HDF close {cut}")
        return real_close(candidate)

    monkeypatch.setattr(h5py.File, "close", close_cut)
    with pytest.raises(
        _InjectedPublicationFailure, match=f"HDF close {cut}",
    ):
        reader.__exit__(None, None, None)
    current = reader._snapshot_reader_cache_state()
    assert current.phase is (
        module._ReaderCachePhase.CLOSE_HDF
        if cut == "before"
        else module._ReaderCachePhase.CLOSED
    )
    assert bool(handle.id.valid) is (cut == "before")
    monkeypatch.setattr(h5py.File, "close", real_close)
    reader.__exit__(None, None, None)
    assert reader._snapshot_reader_cache_state().phase is module._ReaderCachePhase.CLOSED
    assert reader._h5 is None
    assert reader._hdf_owner_token is None
    assert reader.retained_bytes == 0
    _assert_authority_closed_exact(authority, prior_leases)

    with reader as reopened:
        assert reopened._memory_authority is not authority
        assert reopened._hdf_owner_token is not prior_token
        reopened_state = reopened._snapshot_reader_cache_state()
        reopened_authority = reopened._memory_authority
        reopened_leases = tuple(
            entry.lease for entry in reopened_state.entries.values()
        )
        assert reopened_state.phase is module._ReaderCachePhase.OPEN
        assert all(
            new_key != old_key
            for new_key in reopened_state.entries
            for old_key in prior_keys
        )
        _assert_reader_cache_matches_authority(reopened)
    _assert_authority_closed_exact(reopened_authority, reopened_leases)


def test_reader_terminal_markers_survive_non_lifo_and_prune_dead_evidence(
    tmp_path,
) -> None:
    path = tmp_path / "reader_terminal_marker_lineage.nxs"
    _one_dimensional_file(path)
    with module.FrameViewReader(path) as reader:
        first = module._ReaderArrayBundle(reader)
        first_marker = first._marker
        first_ref = weakref.ref(first_marker)
        first.finish()

        second = module._ReaderArrayBundle(reader)
        second_marker = second._marker
        second.rollback()
        after_second = reader._snapshot_reader_cache_state()
        assert _reader_marker_is_present(after_second, first_marker)
        assert _reader_marker_is_present(after_second, second_marker)
        assert module._reader_marker_direction(
            after_second, first_marker,
        ) is module._ReaderCacheDirection.ACCEPTED
        assert module._reader_marker_direction(
            after_second, second_marker,
        ) is module._ReaderCacheDirection.ROLLED_BACK
        assert reader._recover_array_bundle(first_marker) == "accepted"
        assert reader._recover_array_bundle(second_marker) == "rolled-back"

        del first_marker
        gc.collect()
        assert first_ref() is None
        third = module._ReaderArrayBundle(reader)
        third_marker = third._marker
        third.finish()
        after_third = reader._snapshot_reader_cache_state()
        live = tuple(ref_item() for ref_item in after_third.terminal_evidence)
        assert None not in live
        assert second_marker in live
        assert third_marker in live
        assert len(live) == 2
        assert reader._recover_array_bundle(second_marker) == "rolled-back"
        assert reader._recover_array_bundle(third_marker) == "accepted"
        _assert_reader_cache_matches_authority(reader)
