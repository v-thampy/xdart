from __future__ import annotations

from dataclasses import FrozenInstanceError
from importlib import import_module
from pathlib import Path
from threading import Event, Thread
from types import MappingProxyType

import h5py
import numpy as np
import pytest

from xrd_tools.core import (
    FrameRecord,
    FrameView,
    TwoDKind,
    axis_from_unit,
)
from xrd_tools.core.frame_view import FrameGeometry
from xrd_tools.io import (
    FrameScalarCatalog,
    FrameScalarRow,
    FrameViewReader,
    write_frame_records,
)
from xrd_tools.io import frame_view as module
from tests.core.v2_fixture_factory import current_entry


def _view(
    label: int,
    scale: float,
    *,
    q_unit: str = "q_A^-1",
    x_unit: str = "qip_A^-1",
    y_unit: str = "qoop_A^-1",
    kind: TwoDKind = TwoDKind.QIP_QOOP,
) -> FrameView:
    q = np.linspace(0.5, 2.0, 4)
    chi = np.linspace(-1.0, 1.0, 3)
    return FrameView(
        label=label,
        axis_1d=axis_from_unit(q_unit, q),
        intensity_1d=np.arange(4, dtype=float) * scale + label,
        axis_2d_x=axis_from_unit(x_unit, q),
        axis_2d_y=axis_from_unit(y_unit, chi),
        intensity_2d=(
            np.arange(12, dtype=float).reshape(3, 4) * scale + label
        ),
        two_d_kind=kind,
    )


def _catalog_file(path: Path) -> None:
    labels = (2, 5, 9)
    records: list[FrameRecord] = []
    for label in labels:
        record = FrameRecord.from_view(
            _view(label, 1.0),
            mode_1d="q_total",
            mode_2d="qip_qoop",
        )
        if label != 5:
            record = record.with_result_1d(
                "q_oop",
                _view(label, 2.0, q_unit="qoop_A^-1"),
                make_active=False,
            )
            record = record.with_result_2d(
                "q_chi",
                _view(
                    label,
                    3.0,
                    x_unit="q_A^-1",
                    y_unit="chi_deg",
                    kind=TwoDKind.Q_CHI,
                ),
                make_active=False,
            )
        records.append(record)

    with h5py.File(path, "w") as handle:
        entry = current_entry(handle)
        write_frame_records(entry, records)
        geometry = entry.create_group("per_frame_geometry")
        geometry.create_dataset("frame_index", data=np.asarray(labels, np.int64))
        for name, offset in (
            ("rot1", 0.1),
            ("rot2", 0.2),
            ("rot3", 0.3),
            ("incident_angle", 0.4),
        ):
            geometry.create_dataset(
                name,
                data=np.asarray([label + offset for label in labels], np.float32),
            )
        scan = entry.create_group("scan_data")
        scan.create_dataset("frame_index", data=np.asarray(labels, np.int64))
        scan.create_dataset("monitor", data=np.asarray([10, 20, 30], np.float32))
        scan.create_dataset("enabled", data=np.asarray([True, False, True]))
        scan.create_dataset(
            "sample",
            data=np.asarray(["alpha", "beta", "gamma"], dtype=object),
            dtype=h5py.string_dtype("utf-8"),
        )
        scan.create_dataset(
            "numeric_text",
            data=np.asarray(["1.5", "2.5", "3.5"], dtype=object),
            dtype=h5py.string_dtype("utf-8"),
        )
        frames = entry.create_group("frames")
        for label in labels:
            frame = frames.create_group(f"frame_{label:04d}")
            source = frame.create_group("source")
            source.create_dataset("path", data=np.bytes_(f"raw_{label}.h5"))
            source.create_dataset("frame_index", data=np.int64(label + 100))
            if label != 5:
                thumbnail = frame.create_dataset(
                    "thumbnail",
                    data=np.full((2, 3), label, dtype=np.uint8),
                )
                if label == 9:
                    thumbnail.attrs["mask_baked"] = False


def _assert_array_free(value: object) -> None:
    assert not isinstance(value, (np.ndarray, np.generic, memoryview, bytearray))
    if type(value) in {type(None), bool, int, float, str}:
        return
    if type(value) is TwoDKind:
        return
    if type(value) is MappingProxyType:
        for key, item in value.items():
            _assert_array_free(key)
            _assert_array_free(item)
        return
    if type(value) is tuple:
        for item in value:
            _assert_array_free(item)
        return
    if type(value) is FrameGeometry:
        assert value.poni is None
        for item in (
            value.rot1, value.rot2, value.rot3, value.incident_angle,
        ):
            _assert_array_free(item)
        return
    if type(value) is FrameScalarRow:
        for name in value.__slots__:
            _assert_array_free(getattr(value, name))
        return
    if type(value) is FrameScalarCatalog:
        for name in value.__slots__:
            _assert_array_free(getattr(value, name))
        return
    raise AssertionError(f"unexpected catalog object: {type(value)!r}")


def test_scalar_catalog_projects_all_rows_modes_and_deeply_frozen_facts(
    tmp_path,
) -> None:
    path = tmp_path / "catalog.nexus"
    _catalog_file(path)
    with FrameViewReader(
        path, resolve_source=False, include_thumbnail=False,
    ) as reader:
        catalog = reader.read_scalar_catalog()

    assert catalog.artifact_path == str(path)
    assert catalog.entry == "entry"
    assert catalog.labels == (2, 5, 9)
    assert catalog.row(7) is None
    with pytest.raises(TypeError, match="exact nonnegative"):
        catalog.row(True)
    first = catalog.row(2)
    assert first is catalog.rows[0]
    assert first is not None
    assert dict(first.metadata_raw) == {
        "enabled": True,
        "monitor": 10.0,
        "numeric_text": "1.5",
        "sample": "alpha",
    }
    assert dict(first.metadata_numeric) == {
        "enabled": 1.0,
        "monitor": 10.0,
        "numeric_text": 1.5,
    }
    assert first.geometry is not None
    assert (
        first.geometry.rot1,
        first.geometry.rot2,
        first.geometry.rot3,
        first.geometry.incident_angle,
    ) == tuple(float(np.float32(value)) for value in (2.1, 2.2, 2.3, 2.4))
    assert (first.source_path, first.source_frame_index) == ("raw_2.h5", 102)
    assert (first.has_thumbnail, first.mask_baked) == (True, True)
    assert first.modes_1d == ("q_total", "q_oop")
    assert first.modes_2d == ("qip_qoop", "q_chi")
    assert first.active_mode_1d == "q_total"
    assert first.active_mode_2d == "qip_qoop"
    assert first.two_d_kinds == (
        ("qip_qoop", TwoDKind.QIP_QOOP),
        ("q_chi", TwoDKind.Q_CHI),
    )
    middle = catalog.row(5)
    assert middle is not None
    assert middle.modes_1d == ("q_total",)
    assert middle.modes_2d == ("qip_qoop",)
    assert (middle.has_thumbnail, middle.mask_baked) == (False, False)
    last = catalog.row(9)
    assert last is not None
    assert (last.has_thumbnail, last.mask_baked) == (True, False)
    _assert_array_free(catalog)
    with pytest.raises(FrozenInstanceError):
        catalog.rows = ()
    with pytest.raises(TypeError):
        first.metadata_raw["monitor"] = 0.0

    # The returned graph owns exact immutable strings, never a live source/HDF
    # alias.  Later source mutation cannot rewrite the catalog.
    with h5py.File(path, "r+") as handle:
        source = handle["entry/frames/frame_0002/source"]
        del source["path"]
        source.create_dataset("path", data=np.bytes_("changed.h5"))
    assert first.source_path == "raw_2.h5"


def test_scalar_catalog_qualifies_frame_nodes_once(tmp_path, monkeypatch) -> None:
    path = tmp_path / "one_frame_lookup.nexus"
    _catalog_file(path)
    expected = {
        f"/entry/frames/frame_{label:04d}{suffix}": 1
        for label in (2, 5, 9)
        for suffix in (
            "", "/source", "/source/path", "/source/frame_index", "/thumbnail",
        )
    }
    inspections: dict[str, int] = {}
    real_get = h5py.Group.get

    def tracked_get(group, name, *args, **kwargs):
        node_path = f"{group.name}/{name}"
        if kwargs.get("getlink") and node_path in expected:
            inspections[node_path] = inspections.get(node_path, 0) + 1
        return real_get(group, name, *args, **kwargs)

    with FrameViewReader(path, resolve_source=False) as reader:
        monkeypatch.setattr(h5py.Group, "get", tracked_get)
        catalog = reader.read_scalar_catalog()
    assert tuple(
        (row.source_path, row.source_frame_index, row.has_thumbnail, row.mask_baked)
        for row in catalog.rows
    ) == (
        ("raw_2.h5", 102, True, True),
        ("raw_5.h5", 105, False, False),
        ("raw_9.h5", 109, True, False),
    )
    assert inspections == expected


def test_scalar_catalog_marks_current_average_capability(tmp_path) -> None:
    path = tmp_path / "average.nexus"
    record = FrameRecord.from_view(_view(1, 1.0))
    with h5py.File(path, "w") as handle:
        entry = current_entry(handle)
        write_frame_records(entry, (record,))
        frame = entry.create_group("frames").create_group("frame_0001")
        frame.create_dataset(
            "finite_counts", data=np.ones((2, 3), dtype=np.uint32),
        )
        counts = frame["finite_counts"]
        counts.attrs["average_scan_policy"] = np.bytes_(b"average_scan_v1")
        counts.attrs["contributor_extent"] = np.uint32(1)
        counts.attrs["finite_counts_sha256"] = np.bytes_(b"0" * 64)
        counts.attrs["finite_counts_min"] = np.uint32(1)
        counts.attrs["finite_counts_max"] = np.uint32(1)
        counts.attrs["finite_counts_zero_count"] = np.uint64(0)

    with FrameViewReader(
        path, resolve_source=False, include_thumbnail=False,
    ) as reader:
        catalog = reader.read_scalar_catalog()

    assert catalog.labels == (1,)
    assert catalog.rows[0].averaged is True

    with h5py.File(path, "r+") as handle:
        handle["entry/frames"].create_group("frame_0002")
    with FrameViewReader(
        path, resolve_source=False, include_thumbnail=False,
    ) as reader:
        appended = reader.read_scalar_catalog()
    assert appended.labels == (1, 2)
    assert appended.rows[0].averaged is True
    assert appended.rows[1].averaged is False


def test_scalar_catalog_never_reads_scientific_or_thumbnail_payloads(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "payload_free.nexus"
    _catalog_file(path)
    forbidden: list[tuple[str, str]] = []
    direct_reads: dict[str, int] = {}
    real_getitem = h5py.Dataset.__getitem__
    real_read_direct = h5py.Dataset.read_direct

    def is_forbidden(name: str) -> bool:
        return name.endswith((
            "/intensity", "/sigma", "/thumbnail", "/thumbnail_mask",
        ))

    def guarded_getitem(dataset, selection):
        if is_forbidden(dataset.name):
            forbidden.append(("getitem", dataset.name))
            raise AssertionError(f"payload read through __getitem__: {dataset.name}")
        return real_getitem(dataset, selection)

    def guarded_read_direct(dataset, *args, **kwargs):
        if is_forbidden(dataset.name):
            forbidden.append(("read_direct", dataset.name))
            raise AssertionError(f"payload read through read_direct: {dataset.name}")
        direct_reads[dataset.name] = direct_reads.get(dataset.name, 0) + 1
        return real_read_direct(dataset, *args, **kwargs)

    with FrameViewReader(path, resolve_source=False) as reader:
        monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded_getitem)
        monkeypatch.setattr(h5py.Dataset, "read_direct", guarded_read_direct)
        monkeypatch.setattr(
            module, "_read_1d_row",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("1d")),
        )
        monkeypatch.setattr(
            module, "_read_2d_row",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("2d")),
        )
        monkeypatch.setattr(
            module, "_read_thumbnail",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("thumb")),
        )
        monkeypatch.setattr(
            module.FrameViewReader,
            "_view_for",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("view")),
        )
        assert reader.read_scalar_catalog().labels == (2, 5, 9)
    assert forbidden == []
    assert direct_reads["/entry/scan_data/enabled"] == 1
    assert direct_reads["/entry/scan_data/monitor"] == 1
    assert direct_reads["/entry/scan_data/numeric_text"] == 3
    assert direct_reads["/entry/scan_data/sample"] == 3


def test_scalar_catalog_mid_vlen_cancellation_rolls_back_and_retries(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "cancel.nexus"
    _catalog_file(path)
    sample_reads = 0
    cancelling = True
    real_read_direct = h5py.Dataset.read_direct

    def tracked(dataset, *args, **kwargs):
        nonlocal sample_reads
        result = real_read_direct(dataset, *args, **kwargs)
        if dataset.name.endswith("/scan_data/sample"):
            sample_reads += 1
        return result

    monkeypatch.setattr(h5py.Dataset, "read_direct", tracked)
    with FrameViewReader(path, resolve_source=False) as reader:
        baseline = {key: id(value) for key, value in reader._read_cache.items()}
        counts = (
            reader.retained_bytes,
            reader.retained_root_count,
            reader.semantic_references,
        )

        def cancelled() -> bool:
            return bool(cancelling and sample_reads >= 1)

        with pytest.raises(InterruptedError, match="catalog read cancelled"):
            reader.read_scalar_catalog(cancelled=cancelled)
        assert reader._scan_data_columns is None
        assert {key: id(value) for key, value in reader._read_cache.items()} == baseline
        assert (
            reader.retained_bytes,
            reader.retained_root_count,
            reader.semantic_references,
        ) == counts
        cancelling = False
        catalog = reader.read_scalar_catalog(cancelled=cancelled)
        assert catalog.row(9).metadata_raw["sample"] == "gamma"


def test_scalar_catalog_uses_exact_persisted_source_without_path_probes(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "persisted_source.nexus"
    _catalog_file(path)
    probes: list[str] = []

    def forbidden_probe(*_args, **_kwargs):
        probes.append("probe")
        raise AssertionError("scalar catalog consulted the mutable filesystem")

    with FrameViewReader(path) as reader:
        read_module = import_module("xrd_tools.io.read")
        monkeypatch.setattr(
            read_module, "resolve_source_master", forbidden_probe,
        )
        monkeypatch.setattr(Path, "resolve", forbidden_probe)
        monkeypatch.setattr(Path, "exists", forbidden_probe)
        monkeypatch.setattr(Path, "stat", forbidden_probe)
        catalog = reader.read_scalar_catalog()
    assert probes == []
    assert tuple(
        (row.source_path, row.source_frame_index) for row in catalog.rows
    ) == (("raw_2.h5", 102), ("raw_5.h5", 105), ("raw_9.h5", 109))


def test_scalar_catalog_projection_preflight_refuses_before_inventory_or_rows(
    tmp_path,
) -> None:
    path = tmp_path / "projection_bound.nexus"
    _catalog_file(path)
    module._validate_scalar_catalog_projection(
        label_count=651,
        metadata_column_count=256,
        mode_membership_count=651 * 8,
    )
    with FrameViewReader(path, resolve_source=False) as reader:
        baseline = {key: id(value) for key, value in reader._read_cache.items()}
        inventory_calls = 0

        def adversarial_counts() -> tuple[int, int, int]:
            return 1_000_000, 256, 0

        def forbidden_inventory():
            nonlocal inventory_calls
            inventory_calls += 1
            raise AssertionError("oversized inventory was materialized")

        reader._scalar_catalog_projection_counts = adversarial_counts
        reader._scalar_catalog_inventory = forbidden_inventory
        with pytest.raises(ValueError, match="row projection exceeds limit"):
            reader.read_scalar_catalog()
        assert inventory_calls == 0
        assert reader._scan_data_columns is None
        assert {key: id(value) for key, value in reader._read_cache.items()} == baseline


def test_scalar_catalog_normalizes_nonfinite_persisted_geometry(tmp_path) -> None:
    path = tmp_path / "nonfinite_geometry.nexus"
    _catalog_file(path)
    with h5py.File(path, "r+") as handle:
        group = handle["entry/per_frame_geometry"]
        group["rot1"][0] = np.nan
        group["rot2"][0] = np.inf
        group["rot3"][0] = -np.inf
    with FrameViewReader(path, resolve_source=False) as reader:
        row = reader.read_scalar_catalog().row(2)
    assert row is not None and row.geometry is not None
    assert row.geometry.rot1 is None
    assert row.geometry.rot2 is None
    assert row.geometry.rot3 is None
    assert row.geometry.incident_angle == float(np.float32(2.4))


def test_scalar_catalog_explicitly_refuses_partial_target_frame_reader(
    tmp_path,
) -> None:
    path = tmp_path / "target_frame.nexus"
    _catalog_file(path)
    with FrameViewReader(path, target_frame=2) as reader:
        baseline = {key: id(value) for key, value in reader._read_cache.items()}
        with pytest.raises(ValueError, match="full-inventory reader"):
            reader.read_scalar_catalog()
        assert reader._scan_data_columns is None
        assert {key: id(value) for key, value in reader._read_cache.items()} == baseline


def test_scalar_catalog_cancel_callback_is_lock_free_and_reentry_refuses(
    tmp_path,
) -> None:
    path = tmp_path / "callback.nexus"
    _catalog_file(path)
    observations: list[tuple[bool, bool]] = []
    nested: list[str] = []
    with FrameViewReader(path, resolve_source=False) as reader:

        def cancelled() -> bool:
            observations.append((
                reader._reader_cache_lock._is_owned(),
                reader._memory_authority._lock._is_owned(),
            ))
            try:
                reader.read_scalar_catalog()
            except RuntimeError as error:
                nested.append(str(error))
            return False

        catalog = reader.read_scalar_catalog(cancelled=cancelled)
        assert catalog.labels == (2, 5, 9)
    assert observations and set(observations) == {(False, False)}
    assert nested and all("busy" in message for message in nested)


def test_live_scalar_catalog_read_blocks_close_before_hdf_mutation(
    tmp_path,
) -> None:
    path = tmp_path / "close_busy.nexus"
    _catalog_file(path)
    reader = FrameViewReader(path, resolve_source=False).__enter__()
    entered = Event()
    release = Event()
    terminal: list[object] = []

    def cancelled() -> bool:
        if not entered.is_set():
            entered.set()
            assert release.wait(1.0)
        return False

    def run() -> None:
        try:
            terminal.append(reader.read_scalar_catalog(cancelled=cancelled))
        except BaseException as error:  # pragma: no cover - asserted below
            terminal.append(error)

    worker = Thread(target=run)
    worker.start()
    assert entered.wait(1.0)
    h5 = reader._h5
    authority = reader._memory_authority
    with pytest.raises(RuntimeError, match="busy"):
        reader.__exit__(None, None, None)
    assert reader._h5 is h5 and h5 is not None and bool(h5.id.valid)
    assert authority._snapshot_state().closed is False
    release.set()
    worker.join(1.0)
    assert not worker.is_alive()
    assert len(terminal) == 1 and type(terminal[0]) is FrameScalarCatalog
    reader.__exit__(None, None, None)
    assert authority._snapshot_state().closed is True


@pytest.mark.parametrize(
    "factory, message",
    [
        (lambda: FrameScalarRow(label=True), "exact nonnegative"),
        (
            lambda: FrameScalarRow(label=0, metadata_raw={"x": np.float64(1)}),
            "exact scalar",
        ),
        (
            lambda: FrameScalarRow(
                label=0, geometry=FrameGeometry(poni={"distance": 1.0}),
            ),
            "array-free FrameGeometry",
        ),
        (
            lambda: FrameScalarRow(
                label=0, geometry=FrameGeometry(rot1=np.float32(1)),
            ),
            "exact floats",
        ),
        (
            lambda: FrameScalarRow(
                label=0, geometry=FrameGeometry(rot1=float("nan")),
            ),
            "finite floats",
        ),
        (
            lambda: FrameScalarRow(label=0, source_path=Path("raw.tif")),
            "source_path",
        ),
        (
            lambda: FrameScalarRow(
                label=0,
                modes_2d=("q_chi",),
                two_d_kinds=(("q_chi", "q_chi"),),
            ),
            "TwoDKind",
        ),
        (
            lambda: FrameScalarCatalog(
                "a.nxs", "entry",
                (FrameScalarRow(1), FrameScalarRow(1)),
            ),
            "strictly increasing",
        ),
        (
            lambda: FrameScalarCatalog(
                "a.nxs", "entry",
                (FrameScalarRow(2), FrameScalarRow(1)),
            ),
            "strictly increasing",
        ),
        (
            lambda: FrameScalarRow(1, averaged=1),
            "marker facts",
        ),
    ],
)
def test_scalar_catalog_rejects_negative_duplicate_and_malformed_projections(
    factory, message,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()


def test_scalar_catalog_callback_baseexception_publishes_nothing_and_retries(
    tmp_path,
) -> None:
    path = tmp_path / "baseexception.nexus"
    _catalog_file(path)
    raised = False
    with FrameViewReader(path, resolve_source=False) as reader:
        baseline = {key: id(value) for key, value in reader._read_cache.items()}

        def interrupted() -> bool:
            nonlocal raised
            if not raised:
                raised = True
                raise KeyboardInterrupt("catalog callback interrupted")
            return False

        with pytest.raises(KeyboardInterrupt, match="callback interrupted"):
            reader.read_scalar_catalog(cancelled=interrupted)
        assert reader._scan_data_columns is None
        assert {key: id(value) for key, value in reader._read_cache.items()} == baseline
        assert reader.read_scalar_catalog(cancelled=interrupted).labels == (2, 5, 9)
