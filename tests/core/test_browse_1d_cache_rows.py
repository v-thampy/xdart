from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from xrd_tools.core import Axis
from xrd_tools.core.physical_memory import PhysicalRootExchange
from xrd_tools.io import (
    Browse1DCache,
    Browse1DCacheOperation,
    Browse1DLabelStoreCustodyError,
    Browse1DLabelStoreReceipt,
    Frame1DModeRows,
    Frame1DRows,
    FrameScalarCatalog,
    FrameScalarRow,
    browse_1d_row_name,
)
from xrd_tools.io import browse_1d_cache as cache_module


def _readonly(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    array.setflags(write=False)
    return array


def _projection() -> tuple[
    FrameScalarCatalog,
    Frame1DRows,
    dict[str, np.ndarray],
]:
    shared_axis = _readonly([0.5, 1.0, 1.5])
    primary_intensity = _readonly([10.0, 11.0, 12.0])
    primary_sigma = _readonly([1.0, 1.1, 1.2])
    primary_second = _readonly([20.0, 21.0, 22.0])
    secondary_intensity = _readonly([30.0, 31.0, 32.0])
    primary = Frame1DModeRows(
        "q:total/µ",
        Axis("Q", "q_A^-1", False, shared_axis),
        (2, 5),
        (primary_intensity, primary_second),
        (primary_sigma, _readonly([2.0, 2.1, 2.2])),
    )
    secondary = Frame1DModeRows(
        "q|oop",
        Axis("Q_oop", "qoop_A^-1", False, shared_axis),
        (2,),
        (secondary_intensity,),
        None,
    )
    catalog = FrameScalarCatalog(
        "/processed/scan.nxs",
        "entry",
        (
            FrameScalarRow(2, modes_1d=("q:total/µ", "q|oop")),
            FrameScalarRow(5, modes_1d=("q:total/µ",)),
            FrameScalarRow(9),
        ),
        axes_1d=(
            ("q:total/µ", "Q", "q_A^-1", False),
            ("q|oop", "Q_oop", "qoop_A^-1", False),
        ),
    )
    rows = Frame1DRows(
        "/processed/scan.nxs",
        "entry",
        (2, 5, 9),
        (primary, secondary),
        "q:total/µ",
    )
    return catalog, rows, {
        browse_1d_row_name("q:total/µ", "axis"): shared_axis,
        browse_1d_row_name("q:total/µ", "intensity"): primary_intensity,
        browse_1d_row_name("q:total/µ", "sigma"): primary_sigma,
        browse_1d_row_name("q|oop", "axis"): shared_axis,
        browse_1d_row_name("q|oop", "intensity"): secondary_intensity,
    }


def test_row_name_is_versioned_utf8_framed_collision_free_and_bounded() -> None:
    pairs = (
        ("a", "axis"),
        ("a:4:b", "axis"),
        ("a", "intensity"),
        ("a:intensity", "sigma"),
        ("μ/雪:💠", "axis"),
    )
    names = tuple(browse_1d_row_name(*pair) for pair in pairs)
    assert len(set(names)) == len(names)
    unicode_name = browse_1d_row_name("μ/雪:💠", "axis")
    assert unicode_name.startswith(
        f"browse-1d-row-v1:{len('μ/雪:💠'.encode('utf-8'))}:"
    )
    assert len(unicode_name.encode("utf-8")) <= 1024

    for invalid in ("", 1, True, None):
        with pytest.raises(TypeError):
            browse_1d_row_name(invalid, "axis")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        browse_1d_row_name("q", 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        browse_1d_row_name("q", "thumbnail")
    with pytest.raises(ValueError):
        browse_1d_row_name("μ" * 600, "intensity")


def test_label_boundary_stores_one_exact_ordered_mixed_sigma_operation(
    monkeypatch,
) -> None:
    catalog, rows, expected = _projection()
    cache = Browse1DCache(1 << 20)
    calls: list[tuple[int, int, tuple[tuple[str, np.ndarray], ...]]] = []
    real_begin = cache.begin_store

    def begin(frame, label, named, **options):
        calls.append((frame, label, named))
        return real_begin(frame, label, named, **options)

    monkeypatch.setattr(cache, "begin_store", begin)
    monkeypatch.setattr(
        cache,
        "record_scalars",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("scalar catalog must not be mirrored")
        ),
    )
    monkeypatch.setattr(
        cache,
        "scalars",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("scalar catalog must not be queried")
        ),
    )

    receipt = cache.begin_store_1d_label(catalog, rows, 1, 2)

    assert type(receipt) is Browse1DLabelStoreReceipt
    assert type(receipt.operation) is Browse1DCacheOperation
    assert not receipt.complete_empty
    assert not receipt.already_complete
    assert len(calls) == 1
    frame, label, named = calls[0]
    assert (frame, label) == (1, 2)
    assert tuple(name for name, _array in named) == tuple(expected)
    assert all(array is expected[name] for name, array in named)
    assert all(
        (key.frame, key.label) == (1, 2) for key in receipt.keys
    )
    assert tuple(key.name for key in receipt.keys) == tuple(expected)

    operation = receipt.operation
    assert operation is not None
    assert operation.run() == "accepted"
    assert cache.resident_keys == receipt.keys
    # Five logical rows retain four physical roots because both mode axes are
    # the exact same immutable ndarray.
    assert cache.resident_root_count == 4
    assert cache.resident_bytes == sum(
        array.nbytes
        for array in {id(array): array for array in expected.values()}.values()
    )
    for key in receipt.keys:
        with cache.borrow(key.frame, key.label, key.name) as borrowed:
            assert borrowed.array is expected[key.name]


@pytest.mark.parametrize(
    "case",
    ("ordinal", "artifact", "axis", "missing_mode", "extra_mode"),
)
def test_label_mismatch_refuses_before_cache_journal(
    monkeypatch,
    case: str,
) -> None:
    catalog, rows, _expected = _projection()
    ordinal, label = 1, 2
    if case == "ordinal":
        ordinal = 2
    elif case == "artifact":
        rows = Frame1DRows(
            "/processed/other.nxs", rows.entry, rows.labels,
            rows.modes, rows.primary_mode,
        )
    elif case == "axis":
        first = rows.modes[0]
        changed = Frame1DModeRows(
            first.mode,
            Axis("wrong", first.axis.unit, first.axis.log, first.axis.values),
            first.labels,
            first.intensity_rows,
            first.sigma_rows,
        )
        rows = Frame1DRows(
            rows.artifact_path, rows.entry, rows.labels,
            (changed, rows.modes[1]), rows.primary_mode,
        )
    elif case == "missing_mode":
        second = rows.modes[1]
        missing = Frame1DModeRows(
            second.mode, second.axis, (), (), None,
        )
        rows = Frame1DRows(
            rows.artifact_path, rows.entry, rows.labels,
            (rows.modes[0], missing), rows.primary_mode,
        )
    else:
        row = catalog.rows[0]
        catalog = FrameScalarCatalog(
            catalog.artifact_path,
            catalog.entry,
            (
                FrameScalarRow(row.label, modes_1d=("q:total/µ",)),
                *catalog.rows[1:],
            ),
            axes_1d=catalog.axes_1d,
        )

    cache = Browse1DCache(1 << 20)
    state = cache._snapshot_state()
    calls = []
    monkeypatch.setattr(
        cache,
        "begin_store",
        lambda *_args, **_kwargs: calls.append(True),
    )
    with pytest.raises(ValueError):
        cache.begin_store_1d_label(catalog, rows, ordinal, label)
    assert calls == []
    assert cache._snapshot_state() is state
    assert cache.resident_keys == ()


def test_failed_operation_remains_exactly_exposed_for_recovery(
    monkeypatch,
) -> None:
    catalog, rows, _expected = _projection()
    cache = Browse1DCache(1 << 20)
    receipt = cache.begin_store_1d_label(catalog, rows, 1, 2)
    operation = receipt.operation
    assert operation is not None
    real_accept = PhysicalRootExchange.accept
    calls = 0

    def fail_once(exchange):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected accept cut")
        return real_accept(exchange)

    monkeypatch.setattr(PhysicalRootExchange, "accept", fail_once)
    with pytest.raises(RuntimeError, match="injected accept cut"):
        operation.run()
    assert receipt.operation is operation
    assert operation.recover() == "accepted"
    assert receipt.operation is operation
    assert cache.resident_keys == receipt.keys


def test_resident_extra_refuses_before_journal_without_stale_success(
    monkeypatch,
) -> None:
    catalog, rows, _expected = _projection()
    cache = Browse1DCache(1 << 20)
    first = cache.begin_store_1d_label(catalog, rows, 1, 2)
    assert first.operation is not None
    assert first.operation.run() == "accepted"
    prior_keys = cache.resident_keys

    primary = rows.modes[0]
    without_sigma = Frame1DModeRows(
        primary.mode,
        primary.axis,
        primary.labels,
        primary.intensity_rows,
        None,
    )
    changed = Frame1DRows(
        rows.artifact_path,
        rows.entry,
        rows.labels,
        (without_sigma, rows.modes[1]),
        rows.primary_mode,
    )
    calls = []
    real_begin = cache.begin_store

    def begin(*args, **options):
        calls.append((args, options))
        return real_begin(*args, **options)

    monkeypatch.setattr(cache, "begin_store", begin)
    with pytest.raises(ValueError, match="inventory has extra rows"):
        cache.begin_store_1d_label(catalog, changed, 1, 2)
    assert calls == []
    assert cache.resident_keys == prior_keys
    sigma_name = browse_1d_row_name("q:total/µ", "sigma")
    with cache.borrow(1, 2, sigma_name) as borrowed:
        assert borrowed.key.name == sigma_name


def test_partial_eviction_repairs_only_missing_rows_and_preserves_borrow(
    monkeypatch,
) -> None:
    catalog, rows, expected = _projection()
    # Four unique three-point float64 roots exactly fill this cache.  A pinned
    # shared-axis row plus a nine-point competitor therefore evicts every
    # other label component without replacing the outstanding borrow.
    cache = Browse1DCache(4 * 3 * np.dtype(np.float64).itemsize)
    first = cache.begin_store_1d_label(catalog, rows, 1, 2)
    assert first.operation is not None
    assert first.operation.run() == "accepted"

    exact_state = cache._snapshot_state()
    complete = cache.begin_store_1d_label(catalog, rows, 1, 2)
    assert complete.already_complete
    assert not complete.complete_empty
    assert complete.operation is None
    assert complete.keys == first.keys
    assert cache._snapshot_state() is exact_state

    axis_name = browse_1d_row_name("q:total/µ", "axis")
    borrowed = cache.borrow(1, 2, axis_name)
    assert borrowed.array is expected[axis_name]
    competitor = _readonly(np.arange(9, dtype=np.float64))
    assert cache.begin_store(
        99, 99, (("competitor", competitor),),
    ).run() == "accepted"
    assert cache.resident_keys == (
        next(key for key in first.keys if key.name == axis_name),
        cache_module.Browse1DRowKey(99, 99, "competitor"),
    )

    calls: list[tuple[tuple[tuple[str, np.ndarray], ...], dict]] = []
    real_begin = cache.begin_store

    def begin(frame, label, named, **options):
        calls.append((named, options))
        return real_begin(frame, label, named, **options)

    monkeypatch.setattr(cache, "begin_store", begin)
    repaired = cache.begin_store_1d_label(catalog, rows, 1, 2)
    assert repaired.operation is not None
    assert not repaired.already_complete
    assert len(calls) == 1
    missing, options = calls[0]
    assert tuple(name for name, _array in missing) == tuple(
        name for name in expected if name != axis_name
    )
    assert all(array is expected[name] for name, array in missing)
    protected = options["_protected_rows"]
    assert len(protected) == 1
    assert protected[0].array is borrowed.array
    assert repaired.operation.run() == "accepted"
    assert cache.resident_keys == repaired.keys
    assert all(key.name != "competitor" for key in cache.resident_keys)
    assert borrowed.array is expected[axis_name]
    assert not borrowed.released
    borrowed.release()
    assert cache.outstanding_borrows == 0


def test_conflicting_resident_component_refuses_before_journal(
    monkeypatch,
) -> None:
    catalog, rows, _expected = _projection()
    cache = Browse1DCache(1 << 20)
    first = cache.begin_store_1d_label(catalog, rows, 1, 2)
    assert first.operation is not None
    assert first.operation.run() == "accepted"
    state = cache._snapshot_state()

    primary = rows.modes[0]
    changed_intensity = _readonly(primary.intensity_rows[0].copy())
    changed_primary = Frame1DModeRows(
        primary.mode,
        primary.axis,
        primary.labels,
        (changed_intensity, primary.intensity_rows[1]),
        primary.sigma_rows,
    )
    changed = Frame1DRows(
        rows.artifact_path,
        rows.entry,
        rows.labels,
        (changed_primary, rows.modes[1]),
        rows.primary_mode,
    )
    calls = []
    monkeypatch.setattr(
        cache,
        "begin_store",
        lambda *_args, **_kwargs: calls.append(True),
    )

    with pytest.raises(ValueError, match="row conflicts"):
        cache.begin_store_1d_label(catalog, changed, 1, 2)
    assert calls == []
    assert cache._snapshot_state() is state
    assert cache.resident_keys == first.keys


def test_receipt_construction_failure_rolls_back_hidden_journal(
    monkeypatch,
) -> None:
    catalog, rows, _expected = _projection()
    cache = Browse1DCache(1 << 20)

    def refuse_receipt(*_args, **_kwargs):
        raise RuntimeError("injected receipt allocation")

    monkeypatch.setattr(
        cache_module, "Browse1DLabelStoreReceipt", refuse_receipt,
    )
    with pytest.raises(RuntimeError, match="injected receipt allocation"):
        cache.begin_store_1d_label(catalog, rows, 1, 2)
    assert cache.phase.value == "open"
    assert cache.resident_keys == ()
    assert cache.resident_root_count == 0


@pytest.mark.parametrize("cut", ("before", "after"))
def test_receipt_and_rollback_failure_exposes_exact_operation_custody(
    monkeypatch,
    cut: str,
) -> None:
    catalog, rows, _expected = _projection()
    cache = Browse1DCache(1 << 20)
    admitted: list[Browse1DCacheOperation] = []
    real_begin = cache.begin_store
    real_rollback = Browse1DCacheOperation.rollback
    rollback_calls: list[Browse1DCacheOperation] = []

    def begin(*args, **kwargs):
        operation = real_begin(*args, **kwargs)
        admitted.append(operation)
        return operation

    def refuse_receipt(*_args, **_kwargs):
        raise RuntimeError("injected receipt binding")

    def fail_first_rollback(operation):
        rollback_calls.append(operation)
        if len(rollback_calls) == 1:
            if cut == "after":
                assert real_rollback(operation) == "rolled-back"
            raise RuntimeError("injected rollback cut")
        return real_rollback(operation)

    monkeypatch.setattr(cache, "begin_store", begin)
    monkeypatch.setattr(
        cache_module, "Browse1DLabelStoreReceipt", refuse_receipt,
    )
    monkeypatch.setattr(
        Browse1DCacheOperation, "rollback", fail_first_rollback,
    )

    with pytest.raises(Browse1DLabelStoreCustodyError) as caught:
        cache.begin_store_1d_label(catalog, rows, 1, 2)
    error = caught.value
    assert len(admitted) == 1
    assert error.operation is admitted[0]
    assert rollback_calls == [error.operation]
    assert str(error.receipt_error) == "injected receipt binding"
    assert str(error.rollback_error) == "injected rollback cut"
    assert cache.phase.value == (
        "preparing" if cut == "before" else "open"
    )

    assert error.operation.rollback() == "rolled-back"
    assert rollback_calls == [error.operation, error.operation]
    assert cache.phase.value == "open"
    assert cache.resident_keys == ()
    assert cache.resident_root_count == 0


def test_empty_label_returns_immutable_complete_receipt_without_journal() -> None:
    catalog, rows, _expected = _projection()
    cache = Browse1DCache(1 << 20)
    state = cache._snapshot_state()

    receipt = cache.begin_store_1d_label(catalog, rows, 3, 9)

    assert type(receipt) is Browse1DLabelStoreReceipt
    assert receipt.complete_empty
    assert not receipt.already_complete
    assert receipt.operation is None
    assert receipt.keys == ()
    assert cache._snapshot_state() is state
    with pytest.raises(FrozenInstanceError):
        receipt.complete_empty = False  # type: ignore[misc]


def test_ordinary_651_catalog_admits_exact_last_ordinal_without_scalar_copy() -> None:
    labels = tuple(range(1, 652))
    axis = _readonly([1.0])
    intensities = tuple(_readonly([float(label)]) for label in labels)
    catalog = FrameScalarCatalog(
        "/processed/651.nxs",
        "entry",
        tuple(FrameScalarRow(label, modes_1d=("q",)) for label in labels),
        axes_1d=(("q", "Q", "q_A^-1", False),),
    )
    rows = Frame1DRows(
        catalog.artifact_path,
        catalog.entry,
        labels,
        (
            Frame1DModeRows(
                "q",
                Axis("Q", "q_A^-1", False, axis),
                labels,
                intensities,
                None,
            ),
        ),
        "q",
    )
    cache = Browse1DCache(1 << 20)

    receipt = cache.begin_store_1d_label(catalog, rows, 651, 651)

    assert receipt.ordinal == 651
    assert receipt.label == 651
    assert tuple(key.name for key in receipt.keys) == (
        browse_1d_row_name("q", "axis"),
        browse_1d_row_name("q", "intensity"),
    )
    assert receipt.operation is not None
    assert receipt.operation.rollback() == "rolled-back"
    assert cache.resident_keys == ()
