"""PERF-BROWSE-OPEN-128 stage B: per-frame facts read on demand, not at catalog time.

Frozen before the product change (design section 8). The catalog keeps labels, modes and scalar metadata; the six
per-frame facts (geometry, source locator, thumbnail presence, mask marker, averaged) come from one codec,
``FrameViewReader.read_frame_facts``, and live in a bounded per-context ``FrameFactsStore``.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from importlib import import_module
from pathlib import Path
from threading import Thread

import h5py
import numpy as np
import pytest

from xrd_tools.core import FrameRecord
from xrd_tools.core.frame_view import FrameGeometry
from xrd_tools.io import FrameScalarRow, FrameViewReader, write_frame_records
from xrd_tools.io import frame_view as module
from tests.core.test_frame_scalar_catalog import _catalog_file, _view
from tests.core.v2_fixture_factory import current_entry


def _api():
    io = import_module("xrd_tools.io")
    return io.FrameFacts, io.FrameFactsFailure, io.FrameFactsStore


def _read_facts(path, labels, **kwargs):
    with FrameViewReader(path, resolve_source=False, include_thumbnail=False, **kwargs) as reader:
        return reader.read_frame_facts(labels)


def _average_file(path: Path) -> None:
    record = FrameRecord.from_view(_view(1, 1.0))
    with h5py.File(path, "w") as handle:
        entry = current_entry(handle)
        write_frame_records(entry, (record,))
        frame = entry.create_group("frames").create_group("frame_0001")
        frame.create_dataset("finite_counts", data=np.ones((2, 3), dtype=np.uint32))
        counts = frame["finite_counts"]
        counts.attrs["average_scan_policy"] = np.bytes_(b"average_scan_v1")
        counts.attrs["contributor_extent"] = np.uint32(1)
        counts.attrs["finite_counts_sha256"] = np.bytes_(b"0" * 64)
        counts.attrs["finite_counts_min"] = np.uint32(1)
        counts.attrs["finite_counts_max"] = np.uint32(1)
        counts.attrs["finite_counts_zero_count"] = np.uint64(0)


def test_frame_facts_read_the_eager_values_for_requested_labels(tmp_path) -> None:
    FrameFacts, _failure, _store = _api()
    path = tmp_path / "catalog.nexus"
    _catalog_file(path)
    facts = _read_facts(path, (2, 5, 9))

    assert tuple(type(item) for item in facts) == (FrameFacts, FrameFacts, FrameFacts)
    first, middle, last = facts
    assert tuple(item.label for item in facts) == (2, 5, 9)
    assert type(first.geometry) is FrameGeometry and first.geometry.poni is None
    assert (
        first.geometry.rot1, first.geometry.rot2, first.geometry.rot3, first.geometry.incident_angle,
    ) == tuple(float(np.float32(value)) for value in (2.1, 2.2, 2.3, 2.4))
    assert (first.source_path, first.source_frame_index) == ("raw_2.h5", 102)
    assert (first.has_thumbnail, first.mask_baked, first.averaged, first.thumbnail_failure) == (
        True, True, False, None,
    )
    assert (middle.source_path, middle.source_frame_index) == ("raw_5.h5", 105)
    assert (middle.has_thumbnail, middle.mask_baked, middle.thumbnail_failure) == (False, False, None)
    assert (last.source_path, last.source_frame_index) == ("raw_9.h5", 109)
    assert (last.has_thumbnail, last.mask_baked, last.thumbnail_failure) == (True, False, None)
    with pytest.raises(FrozenInstanceError):
        first.source_path = "changed.h5"

    # Detached immutable values: a later source rewrite cannot reach them.
    with h5py.File(path, "r+") as handle:
        source = handle["entry/frames/frame_0002/source"]
        del source["path"]
        source.create_dataset("path", data=np.bytes_("changed.h5"))
    assert first.source_path == "raw_2.h5"
    assert _read_facts(path, (2,))[0].source_path == "changed.h5"


def test_catalog_reads_no_frame_node_and_facts_read_each_requested_node_once(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "one_frame_lookup.nexus"
    _catalog_file(path)
    tracked = {
        f"/entry/frames/frame_{label:04d}{suffix}"
        for label in (2, 5, 9)
        for suffix in ("", "/source", "/source/path", "/source/frame_index", "/thumbnail")
    }
    inspections: dict[str, int] = {}
    geometry_calls: list[int] = []
    real_get = h5py.Group.get
    real_geometry = module.FrameViewReader._scalar_catalog_geometry_for_frame

    def tracked_get(group, name, *args, **kwargs):
        node_path = f"{group.name}/{name}"
        if kwargs.get("getlink") and node_path in tracked:
            inspections[node_path] = inspections.get(node_path, 0) + 1
        return real_get(group, name, *args, **kwargs)

    def counted_geometry(reader, frame):
        geometry_calls.append(frame)
        return real_geometry(reader, frame)

    monkeypatch.setattr(module.FrameViewReader, "_scalar_catalog_geometry_for_frame", counted_geometry)
    with FrameViewReader(path, resolve_source=False) as reader:
        monkeypatch.setattr(h5py.Group, "get", tracked_get)
        catalog = reader.read_scalar_catalog()
        assert catalog.labels == (2, 5, 9)
        assert inspections == {}
        assert geometry_calls == []
        facts = reader.read_frame_facts((5,))
    assert facts[0].label == 5
    assert inspections == {
        f"/entry/frames/frame_0005{suffix}": 1
        for suffix in ("", "/source", "/source/path", "/source/frame_index", "/thumbnail")
    }
    assert geometry_calls == [5]


def test_frame_facts_mark_the_current_average_frame(tmp_path) -> None:
    _api()
    path = tmp_path / "average.nexus"
    _average_file(path)
    assert tuple(item.averaged for item in _read_facts(path, (1,))) == (True,)
    with h5py.File(path, "r+") as handle:
        handle["entry/frames"].create_group("frame_0002")
    assert tuple(item.averaged for item in _read_facts(path, (1, 2))) == (True, False)


def test_frame_facts_never_read_payloads_or_probe_the_filesystem(tmp_path, monkeypatch) -> None:
    _api()
    path = tmp_path / "payload_free.nexus"
    _catalog_file(path)
    forbidden: list[str] = []
    real_getitem = h5py.Dataset.__getitem__
    real_read_direct = h5py.Dataset.read_direct

    def is_payload(name: str) -> bool:
        return name.endswith(("/intensity", "/sigma", "/thumbnail", "/thumbnail_mask"))

    def guarded_getitem(dataset, selection):
        if is_payload(dataset.name):
            forbidden.append(dataset.name)
            raise AssertionError(f"payload read: {dataset.name}")
        return real_getitem(dataset, selection)

    def guarded_read_direct(dataset, *args, **kwargs):
        if is_payload(dataset.name):
            forbidden.append(dataset.name)
            raise AssertionError(f"payload read: {dataset.name}")
        return real_read_direct(dataset, *args, **kwargs)

    def probe(*_args, **_kwargs):
        forbidden.append("probe")
        raise AssertionError("frame facts consulted the mutable filesystem")

    def bomb(*_args, **_kwargs):
        raise AssertionError("frame facts read a payload")

    with FrameViewReader(path) as reader:
        monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded_getitem)
        monkeypatch.setattr(h5py.Dataset, "read_direct", guarded_read_direct)
        for name in ("_read_1d_row", "_read_2d_row", "_read_thumbnail"):
            monkeypatch.setattr(module, name, bomb)
        monkeypatch.setattr(module.FrameViewReader, "_view_for", bomb)
        monkeypatch.setattr(import_module("xrd_tools.io.read"), "resolve_source_master", probe)
        monkeypatch.setattr(Path, "resolve", probe)
        monkeypatch.setattr(Path, "exists", probe)
        monkeypatch.setattr(Path, "stat", probe)
        facts = reader.read_frame_facts((2, 5, 9))
    assert forbidden == []
    assert tuple((item.source_path, item.source_frame_index) for item in facts) == (
        ("raw_2.h5", 102), ("raw_5.h5", 105), ("raw_9.h5", 109),
    )


def test_frame_facts_normalize_nonfinite_geometry(tmp_path) -> None:
    _api()
    path = tmp_path / "nonfinite_geometry.nexus"
    _catalog_file(path)
    with h5py.File(path, "r+") as handle:
        group = handle["entry/per_frame_geometry"]
        group["rot1"][0] = np.nan
        group["rot2"][0] = np.inf
        group["rot3"][0] = -np.inf
    geometry = _read_facts(path, (2,))[0].geometry
    assert geometry is not None
    assert (geometry.rot1, geometry.rot2, geometry.rot3) == (None, None, None)
    assert geometry.incident_angle == float(np.float32(2.4))


def _out_of_bounds_thumbnail(frame) -> None:
    del frame["thumbnail"]
    frame.create_dataset("thumbnail", data=np.zeros((1, 300), dtype=np.uint8))


def _group_thumbnail(frame) -> None:
    del frame["thumbnail"]
    frame.create_group("thumbnail")


def _text_mask_marker(frame) -> None:
    frame["thumbnail"].attrs["mask_baked"] = "yes"


@pytest.mark.parametrize("corrupt", (_out_of_bounds_thumbnail, _group_thumbnail, _text_mask_marker))
def test_malformed_thumbnail_is_flagged_on_its_frame_only(tmp_path, corrupt) -> None:
    path = tmp_path / "bad_thumbnail.nexus"
    _catalog_file(path)
    with h5py.File(path, "r+") as handle:
        corrupt(handle["entry/frames/frame_0009"])

    with FrameViewReader(path, resolve_source=False) as reader:
        # The catalog no longer reads any thumbnail, so the file opens.
        assert reader.read_scalar_catalog().labels == (2, 5, 9)
    FrameFacts, _failure, _store = _api()
    first, last = _read_facts(path, (2, 9))
    assert type(first) is FrameFacts and first.thumbnail_failure is None and first.has_thumbnail
    assert type(last) is FrameFacts
    assert (last.has_thumbnail, last.mask_baked) == (False, False)
    assert type(last.thumbnail_failure) is str and 0 < len(last.thumbnail_failure) <= 512
    assert "thumbnail" in last.thumbnail_failure
    # A flagged thumbnail is not absence, and the frame's other facts stay exact.
    assert (last.source_path, last.source_frame_index) == ("raw_9.h5", 109)
    assert last.geometry is not None and last.geometry.incident_angle == float(np.float32(9.4))


def _long_source_path(frame) -> None:
    source = frame["source"]
    del source["path"]
    source.create_dataset("path", data=np.bytes_(b"x" * 5000))


def _source_dataset(frame) -> None:
    del frame["source"]
    frame.create_dataset("source", data=np.int64(1))


def _negative_source_index(frame) -> None:
    source = frame["source"]
    del source["frame_index"]
    source.create_dataset("frame_index", data=np.int64(-3))


@pytest.mark.parametrize("corrupt", (_long_source_path, _source_dataset, _negative_source_index))
def test_malformed_source_fails_only_its_frame(tmp_path, corrupt) -> None:
    path = tmp_path / "bad_source.nexus"
    _catalog_file(path)
    with h5py.File(path, "r+") as handle:
        corrupt(handle["entry/frames/frame_0005"])

    with FrameViewReader(path, resolve_source=False) as reader:
        assert reader.read_scalar_catalog().labels == (2, 5, 9)
    FrameFacts, FrameFactsFailure, _store = _api()
    facts = _read_facts(path, (2, 5, 9))
    assert tuple(type(item) for item in facts) == (FrameFacts, FrameFactsFailure, FrameFacts)
    failure = facts[1]
    assert failure.label == 5
    assert type(failure.reason) is str and 0 < len(failure.reason) <= 512
    assert (facts[0].source_path, facts[2].source_path) == ("raw_2.h5", "raw_9.h5")


def test_frame_facts_cancellation_publishes_nothing_and_retries(tmp_path) -> None:
    _api()
    path = tmp_path / "cancel.nexus"
    _catalog_file(path)
    calls = 0

    def cancelled() -> bool:
        nonlocal calls
        calls += 1
        return calls > 2

    with FrameViewReader(path, resolve_source=False) as reader:
        baseline = {key: id(value) for key, value in reader._read_cache.items()}
        counts = (reader.retained_bytes, reader.retained_root_count, reader.semantic_references)
        with pytest.raises(InterruptedError, match="facts read cancelled"):
            reader.read_frame_facts((2, 5, 9), cancelled=cancelled)
        assert {key: id(value) for key, value in reader._read_cache.items()} == baseline
        assert (reader.retained_bytes, reader.retained_root_count, reader.semantic_references) == counts
        facts = reader.read_frame_facts((2, 5, 9), cancelled=lambda: False)
    assert tuple(item.label for item in facts) == (2, 5, 9)


@pytest.mark.parametrize(
    "labels",
    ([2, 5], (5, 2), (2, 2), (True,), (-1,), ()),
)
def test_frame_facts_refuse_inexact_label_requests(tmp_path, labels) -> None:
    _api()
    path = tmp_path / "labels.nexus"
    _catalog_file(path)
    with FrameViewReader(path, resolve_source=False) as reader:
        with pytest.raises((TypeError, ValueError)):
            reader.read_frame_facts(labels)


def test_frame_facts_refuse_a_partial_target_frame_reader(tmp_path) -> None:
    _api()
    path = tmp_path / "target_frame.nexus"
    _catalog_file(path)
    with FrameViewReader(path, target_frame=2) as reader:
        with pytest.raises(ValueError, match="full-inventory reader"):
            reader.read_frame_facts((2,))


@pytest.mark.parametrize(
    "build, message",
    (
        (lambda F, X: F(label=True), "label"),
        (lambda F, X: F(label=1, geometry=FrameGeometry(poni={"distance": 1.0})), "FrameGeometry"),
        (lambda F, X: F(label=1, geometry=FrameGeometry(rot1=np.float32(1))), "exact floats"),
        (lambda F, X: F(label=1, geometry=FrameGeometry(rot1=float("nan"))), "finite floats"),
        (lambda F, X: F(label=1, source_path=Path("raw.tif")), "source_path"),
        (lambda F, X: F(label=1, source_path="x" * 5000), "source_path"),
        (lambda F, X: F(label=1, source_frame_index=-1), "source_frame_index"),
        (lambda F, X: F(label=1, has_thumbnail=1), "marker facts"),
        (lambda F, X: F(label=1, averaged=1), "marker facts"),
        (lambda F, X: F(label=1, thumbnail_failure=""), "thumbnail_failure"),
        (lambda F, X: F(label=1, thumbnail_failure="x" * 513), "thumbnail_failure"),
        (lambda F, X: F(label=1, has_thumbnail=True, thumbnail_failure="bad"), "thumbnail_failure"),
        (lambda F, X: X(label=1, reason=""), "reason"),
        (lambda F, X: X(label=1, reason="x" * 513), "reason"),
        (lambda F, X: X(label=-1, reason="bad"), "label"),
    ),
)
def test_frame_facts_and_failures_validate_their_fields(build, message) -> None:
    FrameFacts, FrameFactsFailure, _store = _api()
    with pytest.raises((TypeError, ValueError), match=message):
        build(FrameFacts, FrameFactsFailure)


@pytest.mark.parametrize(
    "name, value",
    (
        ("geometry", FrameGeometry(rot1=1.0)),
        ("source_path", "raw.h5"),
        ("source_frame_index", 3),
        ("has_thumbnail", True),
        ("mask_baked", True),
        ("averaged", True),
    ),
)
def test_scalar_rows_no_longer_carry_frame_facts(name, value) -> None:
    assert name not in FrameScalarRow.__slots__
    with pytest.raises(TypeError):
        FrameScalarRow(label=1, **{name: value})


def _facts_batch(FrameFacts, labels, *, path_bytes=0):
    return tuple(
        FrameFacts(label=label, source_path=("p" * path_bytes) if path_bytes else None)
        for label in labels
    )


def test_frame_facts_store_bounds_records_with_lru_eviction() -> None:
    FrameFacts, _failure, FrameFactsStore = _api()
    store = FrameFactsStore()
    assert (store.MAX_RECORDS, store.MAX_BYTES) == (512, 2 * 1024 * 1024)
    batches = [tuple(range(start, start + 128)) for start in range(0, 640, 128)]
    for labels in batches:
        store.commit(_facts_batch(FrameFacts, labels))
        assert store.record_count <= 512
        assert store.retained_bytes <= 2 * 1024 * 1024
    assert store.record_count == 512
    assert all(store.get(label) is None for label in batches[0])
    assert all(type(store.get(label)) is FrameFacts for labels in batches[1:] for label in labels)
    assert store.retained_bytes == sum(
        FrameFactsStore.charge(store.get(label)) for label in store.resident_labels
    )


def test_frame_facts_store_bounds_bytes_by_conservative_charge() -> None:
    FrameFacts, _failure, FrameFactsStore = _api()
    store = FrameFactsStore()
    sample = FrameFacts(label=0, source_path="p" * 4096)
    assert FrameFactsStore.charge(sample) == 512 + 4096
    for start in range(0, 500, 100):
        store.commit(_facts_batch(FrameFacts, range(start, start + 100), path_bytes=4096))
        assert store.retained_bytes <= 2 * 1024 * 1024
    assert store.record_count == (2 * 1024 * 1024) // (512 + 4096)
    assert store.get(0) is None and type(store.get(499)) is FrameFacts


def test_frame_facts_store_retains_a_requests_resident_labels() -> None:
    FrameFacts, _failure, FrameFactsStore = _api()
    store = FrameFactsStore()
    store.commit(_facts_batch(FrameFacts, range(0, 300)))
    store.commit(_facts_batch(FrameFacts, range(300, 512)))
    store.commit(_facts_batch(FrameFacts, range(512, 522)), retain=tuple(range(0, 5)))
    assert all(type(store.get(label)) is FrameFacts for label in range(0, 5))
    assert all(store.get(label) is None for label in range(5, 15))
    assert type(store.get(15)) is FrameFacts and store.record_count == 512


def test_frame_facts_store_refuses_an_oversize_or_inexact_batch_atomically() -> None:
    FrameFacts, FrameFactsFailure, FrameFactsStore = _api()
    store = FrameFactsStore()
    store.commit(_facts_batch(FrameFacts, range(3)))
    before = (store.resident_labels, store.retained_bytes)
    for batch in (
        _facts_batch(FrameFacts, range(10, 523)),
        (FrameFacts(label=7), FrameFacts(label=7)),
        (FrameFacts(label=8), "not facts"),
    ):
        with pytest.raises((TypeError, ValueError)):
            store.commit(batch)
        assert (store.resident_labels, store.retained_bytes) == before
    with pytest.raises((TypeError, ValueError)):
        store.commit(_facts_batch(FrameFacts, (11,)), retain=(99,))
    assert (store.resident_labels, store.retained_bytes) == before


def test_frame_facts_store_keeps_failures_distinct_and_clears() -> None:
    FrameFacts, FrameFactsFailure, FrameFactsStore = _api()
    store = FrameFactsStore()
    failure = FrameFactsFailure(label=4, reason="source locator is malformed")
    store.commit((FrameFacts(label=3), failure))
    assert store.get(4) is failure and type(store.get(3)) is FrameFacts and store.get(5) is None
    assert FrameFactsStore.charge(failure) == 512 + len(failure.reason.encode("utf-8"))
    store.clear()
    assert (store.record_count, store.retained_bytes, store.resident_labels) == (0, 0, ())
    assert store.get(3) is None


def test_frame_facts_store_commits_from_a_worker_while_reads_continue() -> None:
    FrameFacts, _failure, FrameFactsStore = _api()
    store = FrameFactsStore()
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            for start in range(0, 2048, 64):
                store.commit(_facts_batch(FrameFacts, range(start, start + 64)))
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)

    thread = Thread(target=worker)
    thread.start()
    while thread.is_alive():
        for label in store.resident_labels[:8]:
            value = store.get(label)
            assert value is None or type(value) is FrameFacts
    thread.join(5.0)
    assert errors == [] and store.record_count == 512
