"""Headless :class:`FrameRecordStore` tests.

These are the Phase-B foundation tests: no Qt, no xdart, and no live display
flip.  They lock the store invariants before xdart projects onto it.
"""

from __future__ import annotations

import numpy as np

from xrd_tools.core import Axis, FrameRecord, FrameView
from xrd_tools.session import FrameRecordStore


def _view(label=0, *, source="/data/scan_0001.tif", source_frame=0, scale=1.0):
    return FrameView(
        label=label,
        axis_1d=Axis("Q", "q_A^-1", values=np.array([1.0, 2.0, 3.0])),
        intensity_1d=np.array([10.0, 20.0, 30.0]) * scale,
        metadata_raw={"i0": 1.0, "sample": "A"},
        source_path=source,
        source_frame_index=source_frame,
    )


def _record(
    label=0,
    *,
    mode="q_total",
    source="/data/scan_0001.tif",
    source_frame=0,
    scale=1.0,
):
    return FrameRecord.from_view(
        _view(label, source=source, source_frame=source_frame, scale=scale),
        mode_1d=mode,
    )


def test_store_accumulates_modes_for_same_source():
    store = FrameRecordStore(max_heavy_items=None)
    store.upsert(_record(mode="q_total"))
    store.upsert(_record(mode="q_ip", scale=2.0))

    rec = store.get(0)
    assert rec is not None
    assert set(rec.modes_1d) == {"q_total", "q_ip"}
    np.testing.assert_allclose(rec.view_1d("q_ip").intensity_1d, [20.0, 40.0, 60.0])


def test_store_replaces_instead_of_merging_conflicting_sources():
    store = FrameRecordStore(max_heavy_items=None)
    store.upsert(_record(mode="q_total", source="/run/a/frame_0001.tif"))
    store.upsert(_record(mode="q_ip", source="/run/b/frame_0001.tif", scale=3.0))

    rec = store.get(0)
    assert rec is not None
    assert rec.modes_1d == ("q_ip",)
    assert store.source_identity(0) == "/run/b/frame_0001.tif#0"


def test_store_replaces_known_source_with_missing_source_instead_of_splicing():
    store = FrameRecordStore(max_heavy_items=None)
    store.upsert(_record(mode="q_total", source="/run/a/frame_0001.tif"))
    store.upsert(_record(mode="q_ip", source=None, source_frame=None, scale=2.0))

    rec = store.get(0)
    assert rec is not None
    assert rec.modes_1d == ("q_ip",)
    assert store.source_identity(0) == ""


def test_heavy_eviction_waits_until_frame_is_persisted():
    store = FrameRecordStore(max_heavy_items=1)
    store.upsert(_record(label=1))
    store.upsert(_record(label=2, source="/data/scan_0002.tif"))

    assert store.has_heavy_payload(1)
    assert store.has_heavy_payload(2)

    store.mark_persisted(1)

    assert not store.has_heavy_payload(1)
    assert store.has_heavy_payload(2)
    assert store.get(1).view_1d("q_total").intensity_1d is None
    assert store.get(1).view_1d("q_total").metadata_raw["sample"] == "A"


def test_get_or_hydrate_restores_thinned_record():
    store = FrameRecordStore(max_heavy_items=1)
    store.upsert(_record(label=1), persisted=True)
    store.upsert(_record(label=2, source="/data/scan_0002.tif"), persisted=True)
    assert not store.has_heavy_payload(1)
    assert store.has_heavy_payload(2)

    calls = []

    def hydrate(label):
        calls.append(label)
        return _record(label=label, scale=5.0)

    store.set_hydrator(hydrate)
    rec = store.get_or_hydrate(1)

    assert calls == [1]
    assert store.has_heavy_payload(1)
    assert not store.has_heavy_payload(2)
    np.testing.assert_allclose(rec.view_1d("q_total").intensity_1d, [50.0, 100.0, 150.0])


def test_get_or_hydrate_learns_source_identity_when_thinned_record_had_none():
    store = FrameRecordStore(max_heavy_items=0, require_persisted_for_eviction=False)
    store.upsert(_record(label=1, source=None, source_frame=None))
    assert store.source_identity(1) == ""
    assert not store.has_heavy_payload(1)

    def hydrate(label):
        return _record(label=label, source="/data/scan_0001.tif", source_frame=12)

    store.set_hydrator(hydrate)
    rec = store.get_or_hydrate(1)

    assert rec is not None
    assert store.source_identity(1) == "/data/scan_0001.tif#12"


def test_get_or_hydrate_replaces_when_hydrator_returns_conflicting_source():
    store = FrameRecordStore(max_heavy_items=0, require_persisted_for_eviction=False)
    store.upsert(_record(label=1, mode="q_total", source="/data/a.tif"))
    assert store.source_identity(1) == "/data/a.tif#0"

    def hydrate(label):
        return _record(label=label, mode="q_ip", source="/data/b.tif", scale=4.0)

    store.set_hydrator(hydrate)
    rec = store.get_or_hydrate(1)

    assert rec is not None
    assert rec.modes_1d == ("q_ip",)
    assert store.source_identity(1) == "/data/b.tif#0"


def test_snapshot_is_read_only_copy():
    store = FrameRecordStore(max_heavy_items=None)
    store.upsert(_record(label=1))
    snap = store.snapshot()

    try:
        snap[2] = _record(label=2)  # type: ignore[index]
    except TypeError:
        pass
    assert store.labels() == (1,)
