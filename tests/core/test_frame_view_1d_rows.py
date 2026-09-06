from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from threading import Event, Thread

import h5py
import numpy as np
import pytest

from xrd_tools.core import FrameRecord, FrameView, axis_from_unit
from xrd_tools.io import (
    Frame1DModeRows,
    Frame1DRows,
    FrameScalarCatalog,
    FrameScalarRow,
    FrameViewReader,
    write_frame_records,
)
from xrd_tools.io import frame_view as module
from xrd_tools.io.schema import mode_subgroup_name


def _one_d_view(
    label: int,
    scale: float,
    *,
    unit: str,
    sigma: bool,
) -> FrameView:
    q = np.linspace(0.5, 2.0, 4)
    chi = np.linspace(-1.0, 1.0, 3)
    return FrameView(
        label=label,
        axis_1d=axis_from_unit(unit, q),
        intensity_1d=np.arange(4, dtype=float) * scale + label,
        sigma_1d=(
            np.arange(4, dtype=float) * 0.1 + label
            if sigma else None
        ),
        axis_2d_x=axis_from_unit("q_A^-1", q),
        axis_2d_y=axis_from_unit("chi_deg", chi),
        intensity_2d=np.arange(12, dtype=float).reshape(3, 4) + label,
        thumbnail=np.full((2, 3), label, dtype=np.uint8),
        metadata_raw={"monitor": float(label)},
        source_path=f"raw_{label}.h5",
        source_frame_index=label + 100,
    )


def _one_d_file(path: Path, *, alias_modes: bool = False) -> None:
    records: list[FrameRecord] = []
    for label in (2, 5, 9):
        record = FrameRecord.from_view(
            _one_d_view(
                label, 1.0, unit="q_A^-1", sigma=True,
            ),
            mode_1d="q_total",
        )
        if alias_modes or label != 5:
            record = record.with_result_1d(
                "q_oop",
                _one_d_view(
                    label,
                    1.0 if alias_modes else 2.0,
                    unit="q_A^-1" if alias_modes else "qoop_A^-1",
                    sigma=alias_modes,
                ),
                make_active=False,
            )
        records.append(record)
    with h5py.File(path, "w") as handle:
        write_frame_records(handle.create_group("entry"), records)
        if alias_modes:
            parent = handle["entry/integrated_1d"]
            child = parent[mode_subgroup_name("q_oop")]
            for name in ("axis_x", "intensity", "sigma"):
                del child[name]
                child[name] = parent[name]


def _tiny_one_d_file(path: Path, *, points: int) -> None:
    with h5py.File(path, "w") as handle:
        group = handle.create_group("entry").create_group("integrated_1d")
        q = group.create_dataset("axis_x", data=np.arange(points, dtype=np.float32))
        q.attrs["units"] = "q_A^-1"
        group.create_dataset(
            "intensity", data=np.zeros((2, points), dtype=np.float32),
        )
        group.create_dataset(
            "sigma", data=np.ones((2, points), dtype=np.float32),
        )
        group.create_dataset("frame_index", data=np.asarray([2, 5], np.int64))


def _cache_identity(
    reader: FrameViewReader,
) -> tuple[dict[object, int], tuple[int, int, int]]:
    return (
        {key: id(value) for key, value in reader._read_cache.items()},
        (
            reader.retained_bytes,
            reader.retained_root_count,
            reader.semantic_references,
        ),
    )


def test_read_1d_rows_projects_sparse_modes_axes_sigma_and_catalog_descriptors(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "rows.nxs"
    _one_d_file(path)
    projections: list[tuple[int, int, int, int]] = []
    real_projection = module._validate_1d_result_projection

    def tracked_projection(**facts):
        projections.append((
            facts["label_count"],
            facts["membership_count"],
            facts["root_count"],
            facts["projected_bytes"],
        ))
        return real_projection(**facts)

    monkeypatch.setattr(
        module, "_validate_1d_result_projection", tracked_projection,
    )
    with FrameViewReader(path, resolve_source=False) as reader:
        result = reader.read_1d_rows((2, 5, 9, 12))
        empty_result = reader.read_1d_rows((12,))
        catalog = reader.read_scalar_catalog()

    assert type(result) is Frame1DRows
    assert result.artifact_path == str(path)
    assert result.entry == "entry"
    assert result.labels == (2, 5, 9, 12)
    assert tuple(mode.mode for mode in result.modes) == ("q_total", "q_oop")
    assert result.primary_mode == "q_total"
    assert tuple(mode.mode for mode in empty_result.modes) == (
        "q_total", "q_oop",
    )
    assert all(mode.labels == () for mode in empty_result.modes)
    assert empty_result.primary_mode == "q_total"

    primary = result.mode("q_total")
    assert type(primary) is Frame1DModeRows
    assert primary.labels == (2, 5, 9)
    assert primary.axis.label == "Q"
    assert primary.axis.unit == "q_A^-1"
    assert primary.axis.log is False
    assert primary.sigma_rows is not None
    secondary = result.mode("q_oop")
    assert type(secondary) is Frame1DModeRows
    assert secondary.labels == (2, 9)
    assert secondary.axis.label == "Q_oop"
    assert secondary.axis.unit == "qoop_A^-1"
    assert secondary.sigma_rows is None
    assert result.mode("missing") is None

    for mode_rows in result.modes:
        assert type(mode_rows.axis.values) is np.ndarray
        assert mode_rows.axis.values.dtype == np.float64
        assert not mode_rows.axis.values.flags.writeable
        for row in mode_rows.intensity_rows + (mode_rows.sigma_rows or ()):
            assert type(row) is np.ndarray
            assert row.dtype == np.float64
            assert row.flags.c_contiguous
            assert not row.flags.writeable
            with pytest.raises(ValueError):
                row[0] = -1.0
    row = primary.row(5)
    assert row is not None
    assert row[0] is primary.intensity_rows[1]
    assert row[1] is primary.sigma_rows[1]
    assert primary.row(7) is None
    assert catalog.axes_1d == (
        ("q_total", "Q", "q_A^-1", False),
        ("q_oop", "Q_oop", "qoop_A^-1", False),
    )
    assert all(
        type(value) in {str, bool}
        for descriptor in catalog.axes_1d
        for value in descriptor
    )
    assert (4, 5, 10, 320) in projections


def test_read_1d_rows_uses_one_lean_bundle_and_no_unrelated_payloads(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "lean.nxs"
    _one_d_file(path)
    forbidden: list[str] = []
    read_paths: list[str] = []
    real_read_direct = h5py.Dataset.read_direct

    def bomb(name: str):
        def refuse(*_args, **_kwargs):
            forbidden.append(name)
            raise AssertionError(name)

        return refuse

    def guarded_read_direct(dataset, *args, **kwargs):
        read_paths.append(dataset.name)
        if "/integrated_1d/" not in dataset.name:
            raise AssertionError(f"unrelated payload read: {dataset.name}")
        return real_read_direct(dataset, *args, **kwargs)

    with FrameViewReader(path) as reader:
        for name in (
            "_common_fields",
            "_view_for",
            "_source_for_frame",
            "_persisted_source_for_frame",
            "_metadata_for_frame",
            "_geometry_for_frame",
            "_thumbnail_for_frame",
        ):
            monkeypatch.setattr(module.FrameViewReader, name, bomb(name))
        monkeypatch.setattr(module, "_read_2d_row", bomb("_read_2d_row"))
        monkeypatch.setattr(module, "_read_thumbnail", bomb("_read_thumbnail"))
        monkeypatch.setattr(Path, "resolve", bomb("Path.resolve"))
        monkeypatch.setattr(Path, "exists", bomb("Path.exists"))
        monkeypatch.setattr(Path, "stat", bomb("Path.stat"))
        monkeypatch.setattr(h5py.Dataset, "read_direct", guarded_read_direct)
        result = reader.read_1d_rows((2, 9))

    assert result.labels == (2, 9)
    assert forbidden == []
    assert read_paths
    assert all("/integrated_1d/" in name for name in read_paths)
    assert all(not name.endswith("/axis_x") for name in read_paths)


def test_read_1d_rows_preserves_hardlinked_axis_and_row_identity(tmp_path) -> None:
    path = tmp_path / "aliases.nxs"
    _one_d_file(path, alias_modes=True)
    with FrameViewReader(path, resolve_source=False) as reader:
        result = reader.read_1d_rows((2, 5, 9))

    primary = result.mode("q_total")
    secondary = result.mode("q_oop")
    assert primary is not None and secondary is not None
    assert primary.axis.values is secondary.axis.values
    assert all(
        left is right
        for left, right in zip(
            primary.intensity_rows, secondary.intensity_rows, strict=True,
        )
    )
    assert primary.sigma_rows is not None and secondary.sigma_rows is not None
    assert all(
        left is right
        for left, right in zip(
            primary.sigma_rows, secondary.sigma_rows, strict=True,
        )
    )


def test_read_1d_rows_cancellation_rolls_back_exactly_and_retries(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "cancel.nxs"
    _one_d_file(path)
    reads = 0
    cancelling = True
    real_read = module._read_1d_row

    def tracked(*args, **kwargs):
        nonlocal reads
        result = real_read(*args, **kwargs)
        reads += 1
        return result

    monkeypatch.setattr(module, "_read_1d_row", tracked)
    with FrameViewReader(path, resolve_source=False) as reader:
        baseline = _cache_identity(reader)

        def cancelled() -> bool:
            return bool(cancelling and reads >= 2)

        with pytest.raises(InterruptedError, match="1-D row read cancelled"):
            reader.read_1d_rows((2, 5, 9), cancelled=cancelled)
        assert _cache_identity(reader) == baseline
        assert reader._scan_data_columns is None
        cancelling = False
        result = reader.read_1d_rows((2, 5, 9), cancelled=cancelled)
        assert result.mode("q_total").labels == (2, 5, 9)


def test_read_1d_rows_preflight_cancellation_has_zero_rows_and_retries(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "preflight_cancel.nxs"
    _one_d_file(path)
    callbacks = 0
    row_reads = 0
    cancelling = True
    real_read = module._read_1d_row

    def tracked_read(*args, **kwargs):
        nonlocal row_reads
        row_reads += 1
        return real_read(*args, **kwargs)

    monkeypatch.setattr(module, "_read_1d_row", tracked_read)
    with FrameViewReader(path, resolve_source=False) as reader:
        baseline = _cache_identity(reader)

        def cancelled() -> bool:
            nonlocal callbacks
            callbacks += 1
            return bool(cancelling and callbacks == 4)

        with pytest.raises(InterruptedError, match="1-D row read cancelled"):
            reader.read_1d_rows((2, 5, 9), cancelled=cancelled)
        assert row_reads == 0
        assert _cache_identity(reader) == baseline
        cancelling = False
        result = reader.read_1d_rows((2, 5, 9), cancelled=cancelled)
        assert result.mode("q_total").labels == (2, 5, 9)
        assert _cache_identity(reader) == baseline
        with pytest.raises(TypeError, match="exact boolean"):
            reader.read_1d_rows(
                (2, 5), cancelled=lambda: np.bool_(False),
            )
        assert _cache_identity(reader) == baseline


def test_live_1d_preflight_blocks_close_before_hdf_or_row_mutation(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "preflight_close.nxs"
    _one_d_file(path)
    reader = FrameViewReader(path, resolve_source=False).__enter__()
    h5 = reader._h5
    authority = reader._memory_authority
    axis = reader._axis_1d_modes["q_total"]
    baseline = _cache_identity(reader)
    entered = Event()
    release = Event()
    row_reads = 0
    callbacks = 0
    terminal: list[object] = []
    real_read = module._read_1d_row

    def tracked_read(*args, **kwargs):
        nonlocal row_reads
        row_reads += 1
        return real_read(*args, **kwargs)

    def cancelled() -> bool:
        nonlocal callbacks
        callbacks += 1
        if callbacks == 2:
            entered.set()
            assert release.wait(1.0)
        return False

    def run() -> None:
        try:
            terminal.append(
                reader.read_1d_rows((2, 5, 9), cancelled=cancelled),
            )
        except BaseException as error:  # pragma: no cover - asserted below
            terminal.append(error)

    monkeypatch.setattr(module, "_read_1d_row", tracked_read)
    worker = Thread(target=run)
    worker.start()
    try:
        assert entered.wait(1.0)
        assert row_reads == 0
        with pytest.raises(RuntimeError, match="busy"):
            reader.__exit__(None, None, None)
        assert reader._h5 is h5 and h5 is not None and bool(h5.id.valid)
        assert authority._snapshot_state().closed is False
        release.set()
        worker.join(1.0)
        assert not worker.is_alive()
        assert len(terminal) == 1 and type(terminal[0]) is Frame1DRows
        result = terminal[0]
        assert result.mode("q_total").axis is axis
        assert _cache_identity(reader) == baseline
        retry = reader.read_1d_rows((2, 5, 9))
        assert retry.mode("q_total").axis is axis
        assert _cache_identity(reader) == baseline
    finally:
        release.set()
        worker.join(1.0)
        if reader._h5 is not None:
            reader.__exit__(None, None, None)


def test_read_1d_rows_callback_is_lock_free_and_reentry_refuses(tmp_path) -> None:
    path = tmp_path / "callback.nxs"
    _one_d_file(path)
    observations: list[tuple[bool, bool]] = []
    nested: list[str] = []
    with FrameViewReader(path, resolve_source=False) as reader:

        def cancelled() -> bool:
            observations.append((
                reader._reader_cache_lock._is_owned(),
                reader._memory_authority._lock._is_owned(),
            ))
            try:
                reader.read_1d_rows((2,))
            except RuntimeError as error:
                nested.append(str(error))
            return False

        assert reader.read_1d_rows((2, 9), cancelled=cancelled).labels == (2, 9)
    assert observations and set(observations) == {(False, False)}
    assert nested and all("busy" in message for message in nested)


@pytest.mark.parametrize(
    "factory, message",
    [
        (
            lambda axis, row: Frame1DModeRows("", axis, (1,), (row,)),
            "mode",
        ),
        (
            lambda axis, row: Frame1DModeRows("q", axis, (True,), (row,)),
            "nonnegative",
        ),
        (
            lambda axis, row: Frame1DModeRows(
                "q", axis, (1, 1), (row, row),
            ),
            "strictly increasing",
        ),
        (
            lambda axis, _row: Frame1DModeRows(
                "q", axis, (1,), (np.ones(4, dtype=np.float32),),
            ),
            "C float64",
        ),
        (
            lambda axis, row: Frame1DModeRows(
                "q", axis, (1,), (row,), (),
            ),
            "align exactly",
        ),
    ],
)
def test_frame_1d_mode_rows_rejects_malformed_projection(
    factory, message,
) -> None:
    axis = axis_from_unit("q_A^-1", np.arange(4, dtype=np.float64))
    row = np.arange(4, dtype=np.float64)
    with pytest.raises((TypeError, ValueError), match=message):
        factory(axis, row)


def test_frame_1d_rows_freezes_direct_arrays_and_refuses_invalid_selection(
    tmp_path,
) -> None:
    axis = axis_from_unit("q_A^-1", np.arange(4, dtype=np.float64))
    row = np.arange(4, dtype=np.float64)
    mode = Frame1DModeRows("q", axis, (1,), (row,))
    result = Frame1DRows("scan.nxs", "entry", (1,), (mode,), "q")
    assert not row.flags.writeable
    with pytest.raises(FrozenInstanceError):
        result.labels = ()
    with pytest.raises(ValueError, match="selected labels"):
        Frame1DRows("scan.nxs", "entry", (2,), (mode,), "q")
    with pytest.raises(ValueError, match="duplicate mode"):
        Frame1DRows("scan.nxs", "entry", (1,), (mode, mode), "q")
    with pytest.raises(TypeError, match="axes_1d"):
        FrameScalarCatalog(
            "scan.nxs", "entry", (FrameScalarRow(1),), axes_1d=[],
        )
    with pytest.raises(ValueError, match="duplicate mode"):
        FrameScalarCatalog(
            "scan.nxs",
            "entry",
            (FrameScalarRow(1, modes_1d=("q",)),),
            axes_1d=(("q", "Q", "q_A^-1", False),) * 2,
        )
    with pytest.raises(ValueError, match="describe every"):
        FrameScalarCatalog(
            "scan.nxs", "entry", (FrameScalarRow(1, modes_1d=("q",)),),
        )

    path = tmp_path / "partial.nxs"
    _one_d_file(path)
    with FrameViewReader(path, target_frame=2) as reader:
        with pytest.raises(ValueError, match="full-inventory reader"):
            reader.read_1d_rows((2,))
    with FrameViewReader(path) as reader:
        for bad in ([], (), (True,), (2, 2), (5, 2)):
            with pytest.raises((TypeError, ValueError)):
                reader.read_1d_rows(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("points", [0, 1])
def test_1d_projection_refuses_tiny_excess_before_rows_or_cache_mutation(
    tmp_path, monkeypatch, points,
) -> None:
    path = tmp_path / f"tiny_{points}.nxs"
    _tiny_one_d_file(path, points=points)
    row_accesses: list[tuple[str, str]] = []
    real_getitem = h5py.Dataset.__getitem__
    real_read_direct = h5py.Dataset.read_direct
    real_membership_limit = module._MAX_1D_RESULT_MEMBERSHIPS

    def tracked_getitem(dataset, selection):
        row_accesses.append(("getitem", dataset.name))
        return real_getitem(dataset, selection)

    def tracked_read_direct(dataset, *args, **kwargs):
        row_accesses.append(("read_direct", dataset.name))
        return real_read_direct(dataset, *args, **kwargs)

    with FrameViewReader(path, resolve_source=False) as reader:
        baseline = _cache_identity(reader)
        axis = reader._axis_1d_modes["default"]
        monkeypatch.setattr(h5py.Dataset, "__getitem__", tracked_getitem)
        monkeypatch.setattr(h5py.Dataset, "read_direct", tracked_read_direct)
        monkeypatch.setattr(module, "_MAX_1D_RESULT_MEMBERSHIPS", 1)
        with pytest.raises(ValueError, match="membership projection"):
            reader.read_1d_rows((2, 5))
        assert row_accesses == []
        assert _cache_identity(reader) == baseline

        monkeypatch.setattr(
            module, "_MAX_1D_RESULT_MEMBERSHIPS", real_membership_limit,
        )
        result = reader.read_1d_rows((2, 5))
        mode = result.mode("default")
        assert mode is not None
        assert mode.axis is axis
        assert mode.labels == (2, 5)
        assert _cache_identity(reader) == baseline


@pytest.mark.parametrize("points", [0, 1])
def test_1d_projection_numeric_limits_refuse_million_tiny_roots_and_admit_651(
    points,
) -> None:
    row_bytes = points * np.dtype(np.float64).itemsize
    with pytest.raises(ValueError, match="membership projection"):
        module._validate_1d_result_projection(
            label_count=651,
            membership_count=1_000_000,
            root_count=1_000_001,
            projected_bytes=1_000_000 * row_bytes,
        )
    with pytest.raises(ValueError, match="root projection"):
        module._validate_1d_result_projection(
            label_count=1,
            membership_count=1,
            root_count=module._MAX_1D_RESULT_ROOTS + 1,
            projected_bytes=0,
        )
    with pytest.raises(ValueError, match="byte projection"):
        module._validate_1d_result_projection(
            label_count=1,
            membership_count=1,
            root_count=1,
            projected_bytes=module._MAX_1D_RESULT_BYTES + 1,
        )

    ordinary_points = 1000
    module._validate_1d_result_projection(
        label_count=651,
        membership_count=651 * 5,
        root_count=651 * 5 * 2 + 5,
        projected_bytes=(
            651 * 5 * 2 * ordinary_points * np.dtype(np.float64).itemsize
            + 5 * ordinary_points * np.dtype(np.float64).itemsize
        ),
    )
