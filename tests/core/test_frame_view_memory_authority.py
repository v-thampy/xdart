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
    PhysicalRootLease,
    PhysicalRootReservation,
    physical_root_fact,
)
from xrd_tools.core import FrameRecord, FrameView, axis_from_unit
from xrd_tools.io import write_frame_records
from xrd_tools.io.schema import (
    PROCESSED_SCHEMA_NAME,
    PROCESSED_SCHEMA_VERSION,
)
from xrd_tools.io import frame_view as module


class _InjectedPublicationFailure(BaseException):
    pass


def _assert_reader_cache_matches_authority(reader) -> None:
    state = reader._snapshot_reader_cache_state()
    assert state.phase is module._ReaderCachePhase.OPEN
    assert state.pending is None
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
    path = tmp_path / "reader_bundle_linear.nexus"
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
    path = tmp_path / f"reader_linear_{opened}.nexus"
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
    include_sigma: bool = False,
    scan_alias: bool = False,
) -> None:
    """Write a current, writer-produced 1-D stack before changing a seam.

    The reservation tests deliberately perturb individual arrays below, but
    their starting graph must remain a record-writer graph so admission is not
    itself the behavior under test.
    """
    records = []
    for label in range(5, 5 + rows):
        values = np.arange(points, dtype=source_dtype) + (label - 5) * points
        view = FrameView(
            label=label,
            axis_1d=axis_from_unit(
                "qtot_A^-1", np.linspace(0.1, 1.0, points),
            ),
            intensity_1d=values,
            sigma_1d=values if include_sigma else None,
            metadata_raw={"monitor": float(label)},
        )
        records.append(FrameRecord.from_view(view, mode_1d="q_total"))
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.attrs["ssrl_schema"] = PROCESSED_SCHEMA_NAME
        entry.attrs["ssrl_schema_version"] = PROCESSED_SCHEMA_VERSION
        write_frame_records(entry, records)
        if scan_alias:
            scan = entry.create_group("scan_data")
            labels = entry["integrated_1d/frame_index"]
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
        primary.attrs["primary_mode"] = "q_total"
        primary.attrs["multi_result_modes"] = ("q_total", "q_oop")
        child = primary.create_group("q_oop")
        child.attrs["NX_class"] = "NXdata"
        child.attrs["signal"] = "intensity"
        child.attrs["axes"] = ("frame_index", "axis_1")
        labels = np.asarray(primary["frame_index"], dtype=np.int64)
        child.create_dataset(
            "frame_index", data=labels,
            chunks=(max(1, labels.size),), maxshape=(None,),
        )
        if linked_q:
            child["axis_1"] = primary["axis_1"]
        else:
            child.create_dataset(
                "axis_1", data=np.linspace(0.2, 1.1, points, dtype=np.float32),
            )
        width = points + 1 if malformed_intensity else points
        child.create_dataset(
            "intensity",
            data=np.arange(labels.size * width, dtype=np.float32).reshape(
                labels.size, width,
            ),
            chunks=(1, width), maxshape=(None, width),
        )
        if linked_intensity:
            raise ValueError("current result rows may not alias across modes")


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


@pytest.mark.parametrize("include_sigma", (True, False))
def test_reader_cache_uses_physical_hdf_identity_after_role_validation(
    tmp_path, monkeypatch, include_sigma,
) -> None:
    path = tmp_path / f"physical_cache_{include_sigma}.nexus"
    _one_dimensional_file(
        path, include_sigma=include_sigma, scan_alias=True,
    )
    _add_second_1d_mode(path, linked_q=True)
    names = (
        "entry/integrated_1d/frame_index",
        "entry/integrated_1d/axis_1",
        "entry/integrated_1d/intensity",
        "entry/integrated_1d/q_oop/frame_index",
        "entry/integrated_1d/q_oop/axis_1",
        "entry/integrated_1d/q_oop/intensity",
        "entry/scan_data/frame_index",
        "entry/scan_data/monitor",
    )
    addresses = _object_addresses(
        path,
        names + (("entry/integrated_1d/sigma",) if include_sigma else ()),
    )
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
        assert (view.sigma_1d is not None) is include_sigma
        assert len(record.results_1d) == 2
        assert len({id(item.intensity_1d) for item in record.results_1d.values()}) == 2
        assert reader.retained_bytes > 0
        assert reader.retained_root_count > 0

    assert calls.count(addresses[names[0]]) == 1
    assert calls.count(addresses[names[1]]) == 1
    assert addresses[names[0]] != addresses[names[3]]
    assert addresses[names[1]] == addresses[names[4]]
    assert addresses[names[2]] != addresses[names[5]]
    assert addresses[names[0]] == addresses[names[6]] == addresses[names[7]]
    assert "/entry/scan_data/monitor" in qualified_roles
    if include_sigma:
        assert calls.count(addresses[names[2]]) == 1
        assert calls.count(addresses["entry/integrated_1d/sigma"]) == 1
    else:
        assert "entry/integrated_1d/sigma" not in addresses
        assert calls.count(addresses[names[2]]) == 1
    assert reader.retained_bytes == 0
    assert reader.retained_root_count == 0
    assert reader.semantic_references == 0


def test_float32_row_reserves_final_float64_bytes_before_hdf_read(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "final_float64_reservation.nexus"
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
    path = tmp_path / "vlen_pointer_preflight.nexus"
    rows = 3
    _one_dimensional_file(path, rows=rows)
    with h5py.File(path, "r+") as handle:
        scan = handle.require_group("entry/scan_data")
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
    transient_row_bytes = 8 * np.dtype(np.float64).itemsize
    monkeypatch.setattr(
        module,
        "_MAX_READER_RETAINED_BYTES",
        baseline + pointer_bytes - 1,
    )
    real_empty = np.empty
    real_cell = module._bounded_utf8_item
    real_claim = PhysicalRootReservation.claim
    real_bind = PhysicalRootReservation.bind
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
        monkeypatch.setattr(PhysicalRootReservation, "claim", tracked_claim)
        monkeypatch.setattr(PhysicalRootReservation, "bind", tracked_bind)
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
            baseline + pointer_bytes + transient_row_bytes,
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
    path = tmp_path / "vlen_hardlink_alias.nexus"
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
    path = tmp_path / "cache_key_transform_dtype.nexus"
    _one_dimensional_file(path)
    reads: list[int] = []
    roles: list[str] = []
    real_read = h5py.Dataset.read_direct

    def tracked_read(dataset, *args, **kwargs):
        reads.append(int(h5py.h5o.get_info(dataset.id).addr))
        return real_read(dataset, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "read_direct", tracked_read)
    with module.FrameViewReader(path) as reader:
        dataset = reader._entry["integrated_1d/axis_1"]

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
    address = _object_addresses(path, ("entry/integrated_1d/axis_1",))[
        "entry/integrated_1d/axis_1"
    ]
    # One initial axis read plus three distinct cache-key reads.
    assert reads.count(address) == 4
    assert roles == ["axis-a64", "axis-b64", "axis-a32"]


def test_two_transient_rows_share_one_aggregate_bundle_ceiling_and_retry(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "two_transient_rows.nexus"
    points = 16
    _one_dimensional_file(path, points=points, include_sigma=True)
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
    path = tmp_path / "aggregate_axis_limit.nexus"
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
        path, ("entry/integrated_1d/q_oop/axis_1",),
    )["entry/integrated_1d/q_oop/axis_1"]
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
    path = tmp_path / "late_malformed_mode.nexus"
    _one_dimensional_file(path)
    _add_second_1d_mode(path, malformed_intensity=True)
    reader = module.FrameViewReader(path)
    with pytest.raises(ValueError, match="current xdart"):
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
            chunks=(1, 8), maxshape=(None, 8),
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
    reader = module.FrameViewReader(tmp_path / "missing.nexus")
    with pytest.raises(OSError):
        reader.__enter__()
    assert reader._h5 is None
    assert reader._hdf_owner_token is None
    assert reader._read_cache == {}
    assert reader.retained_bytes == 0
    assert reader.retained_root_count == 0
    assert reader.semantic_references == 0


@pytest.mark.parametrize("failure", ("lease",))
def test_enter_bundle_publication_baseexception_leaves_zero_owner_state(
    tmp_path, monkeypatch, failure,
) -> None:
    path = tmp_path / f"enter_publication_{failure}.nexus"
    _one_dimensional_file(path)
    reader = module.FrameViewReader(path)

    if failure == "lease":
        def fail_lease(_self, _authority, _semantic):
            raise _InjectedPublicationFailure("lease construction failed")

        monkeypatch.setattr(PhysicalRootLease, "__init__", fail_lease)
    with pytest.raises(_InjectedPublicationFailure):
        reader.__enter__()
    assert reader._h5 is None
    assert reader._hdf_owner_token is None
    assert reader._entry is None
    assert reader._read_cache == {}
    assert reader.retained_bytes == 0
    assert reader.retained_root_count == 0
    assert reader.semantic_references == 0


@pytest.mark.parametrize("failure", ("lease",))
def test_read_bundle_finish_baseexception_preserves_exact_prior_cache(
    tmp_path, monkeypatch, failure,
) -> None:
    path = tmp_path / f"read_publication_{failure}.nexus"
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
        with pytest.raises(_InjectedPublicationFailure):
            reader.read(5)
        assert reader._scan_data_columns is None
        assert dict(reader._read_cache) == baseline_cache
        assert reader.retained_bytes == baseline_bytes
        assert reader.retained_root_count == baseline_roots
        assert reader.semantic_references == baseline_refs
        if failure == "lease":
            monkeypatch.setattr(PhysicalRootLease, "__init__", real_init)


def test_read_record_bundle_failure_is_all_or_none_and_output_transfers(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "record_bundle_atomic.nexus"
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


def test_reader_failed_fill_rolls_back_reservation_and_rereads(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "reservation_fill_retry.nexus"
    _one_dimensional_file(path)
    intensity = _object_addresses(
        path, ("entry/integrated_1d/intensity",),
    )["entry/integrated_1d/intensity"]
    real_read_direct = h5py.Dataset.read_direct
    failed = True

    def fail_once(dataset, *args, **kwargs):
        nonlocal failed
        if failed and int(h5py.h5o.get_info(dataset.id).addr) == intensity:
            failed = False
            raise _InjectedPublicationFailure("fill failed")
        return real_read_direct(dataset, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "read_direct", fail_once)
    with module.FrameViewReader(path) as reader:
        baseline = reader.retained_bytes
        with pytest.raises(_InjectedPublicationFailure, match="fill failed"):
            reader.read(5)
        assert reader.retained_bytes == baseline
        assert reader._scan_data_columns is None
        reread = reader.read(5)
        assert reread.axis_1d is not None
        np.testing.assert_allclose(reread.axis_1d.values, np.linspace(0.1, 1.0, 8))


def test_reader_cache_diagnostic_is_values_only_and_bundle_releases_owners(tmp_path) -> None:
    path = tmp_path / "reader_diagnostic.nexus"
    _one_dimensional_file(path)
    with module.FrameViewReader(path) as reader:
        projection = reader._read_cache
        for key, value in projection.items():
            assert type(value) is np.ndarray
            assert value is reader._snapshot_reader_cache_state().entries[key].array
        with pytest.raises(TypeError):
            projection[("forged",)] = np.empty(1)
        bundle = module._ReaderArrayBundle(reader)
        bundle.finish()
        assert bundle._terminal == "accepted"
        assert bundle._reader is None
        assert bundle._owner is None
        assert bundle._reservation is None
        assert bundle._local == {}
        assert bundle.pending_scan_data_columns is None
        _assert_reader_cache_matches_authority(reader)


def test_reader_close_releases_actual_cache_leases(tmp_path) -> None:
    path = tmp_path / "reservation_close_actual_leases.nexus"
    _one_dimensional_file(path)
    reader = module.FrameViewReader(path)
    reader.__enter__()
    reader.read(5)
    leases = tuple(entry.lease for entry in reader._snapshot_reader_cache_state().entries.values())
    assert leases
    reader.__exit__(None, None, None)
    _assert_authority_closed_exact(reader._memory_authority, leases)
    assert reader._read_cache == {}
    assert reader._scan_data_columns is None
    assert reader._scan_data_items == ()
    assert reader._axis_1d is None
    assert reader._axis_1d_modes == {}
    assert reader._map_1d == {}
    assert reader._map_1d_modes == {}


def test_reader_close_lease_release_failure_retains_pending_charge_and_retries(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "reservation_close_lease_retry.nexus"
    _one_dimensional_file(path)
    reader = module.FrameViewReader(path)
    reader.__enter__()
    reader.read(5)
    leases = tuple(
        entry.lease for entry in reader._snapshot_reader_cache_state().entries.values()
    )
    assert leases
    real_release = PhysicalRootLease.release
    failed = False

    def fail_once(lease):
        nonlocal failed
        if not failed:
            failed = True
            raise _InjectedPublicationFailure("lease release failed")
        return real_release(lease)

    monkeypatch.setattr(PhysicalRootLease, "release", fail_once)
    with pytest.raises(_InjectedPublicationFailure, match="lease release failed"):
        reader.__exit__(None, None, None)
    pending = reader._snapshot_reader_cache_state()
    assert pending.phase is module._ReaderCachePhase.CLOSING_LEASES
    assert pending.close_leases
    assert reader._memory_authority.retained_bytes > 0
    assert all(not lease.released for lease in pending.close_leases)

    monkeypatch.setattr(PhysicalRootLease, "release", real_release)
    reader.__exit__(None, None, None)
    assert reader._snapshot_reader_cache_state().phase is module._ReaderCachePhase.CLOSED
    assert reader._memory_authority.retained_bytes == 0
    _assert_authority_closed_exact(reader._memory_authority, leases)


def test_reader_hdf_close_failure_drops_science_arrays_and_retries(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "reservation_hdf_close_retry.nexus"
    _one_dimensional_file(path)
    reader = module.FrameViewReader(path)
    reader.__enter__()
    reader.read(5)
    array_refs = tuple(
        weakref.ref(entry.array)
        for entry in reader._snapshot_reader_cache_state().entries.values()
    )
    handle = reader._h5
    assert handle is not None
    real_close = h5py.File.close
    failed = False

    def fail_once(candidate):
        nonlocal failed
        if candidate is handle and not failed:
            failed = True
            raise _InjectedPublicationFailure("hdf close failed")
        return real_close(candidate)

    monkeypatch.setattr(h5py.File, "close", fail_once)
    with pytest.raises(_InjectedPublicationFailure, match="hdf close failed"):
        reader.__exit__(None, None, None)
    state = reader._snapshot_reader_cache_state()
    assert state.phase is module._ReaderCachePhase.CLOSE_HDF
    assert state.entries == {}
    assert state.scan_data_columns is None
    assert state.close_leases == ()
    assert reader._memory_authority.retained_bytes == 0
    assert reader._entry is None
    assert reader._axis_1d is None
    gc.collect()
    assert all(reference() is None for reference in array_refs)
    reader.__exit__(None, None, None)
    assert reader._snapshot_reader_cache_state().phase is module._ReaderCachePhase.CLOSED


def test_reader_cache_accept_preserves_survivors_and_publishes_one_graph(
    tmp_path,
) -> None:
    path = tmp_path / "reader_accept_graph.nexus"
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


def test_reader_bundle_hash_and_equality_callbacks_run_without_reader_locks(
    tmp_path,
) -> None:
    path = tmp_path / "reader_callback_lock_freedom.nexus"
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
    path = tmp_path / "reader_live_building_owner.nexus"
    _one_dimensional_file(path)
    reader = module.FrameViewReader(path)
    reader.__enter__()
    handle = reader._h5
    assert handle is not None
    authority_state = reader._memory_authority._snapshot_state()
    entered = Event()
    release = Event()
    real_claim = reader._bundle_reservation_claim
    blocked_once = False

    def held_claim(owner, nbytes):
        nonlocal blocked_once
        if not blocked_once:
            blocked_once = True
            entered.set()
            assert release.wait(1.0)
        return real_claim(owner, nbytes)

    monkeypatch.setattr(reader, "_bundle_reservation_claim", held_claim)
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
        held_state = reader._memory_authority._snapshot_state()
        assert held_state.roots == authority_state.roots
        assert held_state.bindings == authority_state.bindings
        assert held_state.gate is not None
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
def test_reader_cache_authority_close_fault_is_retryable(
    tmp_path, monkeypatch, cut: str,
) -> None:
    path = tmp_path / f"reader_authority_close_{cut}.nexus"
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
    assert pending.close_leases == ()
    reader.__exit__(None, None, None)
    assert reader._snapshot_reader_cache_state().phase is module._ReaderCachePhase.CLOSED
    assert reader.retained_bytes == 0


@pytest.mark.parametrize("cut", ("before", "after"))
def test_reader_cache_hdf_close_fault_is_exactly_retryable(
    tmp_path, monkeypatch, cut: str,
) -> None:
    path = tmp_path / f"reader_hdf_close_{cut}.nexus"
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
    assert current.phase is module._ReaderCachePhase.CLOSE_HDF
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
