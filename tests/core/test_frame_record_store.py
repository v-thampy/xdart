"""Headless :class:`FrameRecordStore` tests.

These are the Phase-B foundation tests: no Qt, no xdart, and no live display
flip.  They lock the store invariants before xdart projects onto it.
"""

from __future__ import annotations

import numpy as np

from xrd_tools.core import Axis, FrameRecord, FrameView
from xrd_tools.session import FrameRecordStore


def _view(label=0, *, source: "str | None" = "/data/scan_0001.tif",
          source_frame: "int | None" = 0, scale=1.0):
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
    source: "str | None" = "/data/scan_0001.tif",
    source_frame: "int | None" = 0,
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


def test_new_unsaved_mode_does_not_inherit_label_persistence():
    store = FrameRecordStore(max_heavy_items=1)
    store.upsert(_record(label=1, mode="q_total"), persisted=True)
    assert store.is_persisted(1)

    store.upsert(_record(label=1, mode="q_ip", scale=2.0), persisted=False)
    assert not store.is_persisted(1)

    store.upsert(_record(label=2, source="/data/scan_0002.tif"), persisted=True)

    assert store.has_heavy_payload(1)
    assert not store.has_heavy_payload(2)
    rec = store.get(1)
    assert rec is not None
    assert set(rec.modes_1d) == {"q_total", "q_ip"}
    np.testing.assert_allclose(rec.view_1d("q_ip").intensity_1d, [20.0, 40.0, 60.0])


def test_mark_persisted_marks_all_current_modes_for_eviction():
    store = FrameRecordStore(max_heavy_items=1)
    store.upsert(_record(label=1, mode="q_total"), persisted=True)
    store.upsert(_record(label=1, mode="q_ip", scale=2.0), persisted=False)

    store.mark_persisted(1)
    assert store.is_persisted(1)

    store.upsert(_record(label=2, source="/data/scan_0002.tif"), persisted=True)

    assert not store.has_heavy_payload(1)
    assert store.has_heavy_payload(2)
    rec = store.get(1)
    assert rec is not None
    assert rec.view_1d("q_total").intensity_1d is None
    assert rec.view_1d("q_ip").intensity_1d is None


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


def test_get_or_hydrate_does_not_persist_extra_unsaved_mode():
    # P2 regression: a hydrator that returns the persisted disk mode PLUS an
    # extra freshly-computed (unsaved) mode must NOT mark the extra mode
    # persisted — else it could be heavy-evicted before it is written (the
    # 748fcac persist-before-evict bug, re-introduced via get_or_hydrate).
    store = FrameRecordStore(max_heavy_items=1)
    store.upsert(_record(label=1, mode="q_total"), persisted=True)
    store.upsert(_record(label=2, source="/data/scan_0002.tif"), persisted=True)
    assert not store.has_heavy_payload(1)            # thinned (fully persisted)

    def hydrate(label):
        rec = FrameRecord.from_view(_view(label), mode_1d="q_total")  # on-disk mode
        return rec.with_result_1d("q_ip", _view(label, scale=2.0))   # extra unsaved

    store.set_hydrator(hydrate)
    rec = store.get_or_hydrate(1)
    assert set(rec.modes_1d) == {"q_total", "q_ip"}
    assert not store.is_persisted(1)                 # q_ip unsaved -> not fully persisted

    # Heavy pressure must thin the fully-persisted frame, NOT label 1 (q_ip unsaved).
    store.upsert(_record(label=3, source="/data/scan_0003.tif"), persisted=True)
    assert store.has_heavy_payload(1)
    assert store.get(1).view_1d("q_ip").intensity_1d is not None


def test_max_items_evicts_persisted_records_not_unpersisted():
    # max_items full-record eviction (audit gap): evicts a PERSISTED record, never
    # an unpersisted one (persist-before-evict also gates whole-record eviction).
    store = FrameRecordStore(max_items=2, max_heavy_items=None)
    store.upsert(_record(label=1, source="/d/1.tif"), persisted=True)
    store.upsert(_record(label=2, source="/d/2.tif"), persisted=False)
    store.upsert(_record(label=3, source="/d/3.tif"), persisted=True)
    assert len(store) == 2
    assert set(store.labels()) == {2, 3}            # persisted 1 evicted; unpersisted 2 kept

    # All-unpersisted past the cap: nothing is evictable, so the store keeps both.
    store2 = FrameRecordStore(max_items=1, max_heavy_items=None)
    store2.upsert(_record(label=1, source="/d/1.tif"), persisted=False)
    store2.upsert(_record(label=2, source="/d/2.tif"), persisted=False)
    assert len(store2) == 2


def test_concurrent_upsert_mark_hydrate_is_thread_safe():
    # The RLock is the store's headline safety feature; exercise it under
    # contention (upsert / mark_persisted / get_or_hydrate / snapshot from many
    # threads).  Assert no exception, no deadlock, and the bounds hold.
    import threading

    store = FrameRecordStore(max_heavy_items=8, max_items=20)
    store.set_hydrator(lambda label: _record(label=label, scale=3.0))
    errors: list[BaseException] = []

    def worker(base):
        try:
            for i in range(base, base + 80):
                lbl = i % 20
                store.upsert(_record(label=lbl, source=f"/d/{lbl}.tif"),
                             persisted=(i % 2 == 0))
                store.mark_persisted(lbl)
                store.get_or_hydrate(lbl)
                store.snapshot()
        except BaseException as exc:                 # pragma: no cover - diagnostic
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(b,))
               for b in (0, 100, 200, 300)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15.0)

    assert not errors, errors
    assert not any(t.is_alive() for t in threads)   # no deadlock
    assert len(store) <= 20                          # max_items bound held


# --------------------------------------------------------------------------- #
# A-prep2: freeze the config the LIVE store (D3 one-store collapse) will use.
#
# The live store mirrors xdart's LiveFrameSeries in-memory cap
# (``frame_series.py::_in_memory_cap == 64``) and the persist-before-evict
# invariant (``frame_series.py``: ``_in_memory`` never evicts unpersisted
# frames).  These tests pin that exact config so the later eviction wiring lands
# against a tested spec rather than re-deriving it.  Read-only confirmed against
# ``src/xdart/modules/ewald/frame_series.py``; this constant is duplicated, not
# imported, to keep the test xdart-free.
# --------------------------------------------------------------------------- #

LIVE_STORE_HEAVY_CAP = 64
"""Mirror of ``LiveFrameSeries._in_memory_cap`` (xdart frame_series.py)."""


def _live_store() -> FrameRecordStore:
    """The exact config the live FrameRecordStore path will use."""
    return FrameRecordStore(
        max_heavy_items=LIVE_STORE_HEAVY_CAP,
        require_persisted_for_eviction=True,
    )


def test_live_store_config_evicts_only_persisted_under_heavy_pressure():
    # (a) With max_heavy_items=64 and require-persisted eviction, pushing past
    # the cap thins ONLY persisted records; unpersisted heavy frames survive.
    store = _live_store()

    # Fill the cap with persisted frames (eligible for eviction)...
    for i in range(LIVE_STORE_HEAVY_CAP):
        store.upsert(_record(label=i, source=f"/d/{i}.tif"), persisted=True)
    # ...then add one MORE persisted frame: exactly one persisted frame is thinned.
    store.upsert(
        _record(label=LIVE_STORE_HEAVY_CAP, source="/d/cap.tif"), persisted=True
    )

    thinned = [
        i
        for i in range(LIVE_STORE_HEAVY_CAP + 1)
        if not store.has_heavy_payload(i)
    ]
    assert len(thinned) == 1                         # one over the cap -> one thinned
    # The thinned frame keeps its labels/axes/metadata; only arrays are dropped.
    rec = store.get(thinned[0])
    assert rec is not None
    assert rec.view_1d("q_total").intensity_1d is None
    assert rec.view_1d("q_total").metadata_raw["sample"] == "A"


def test_live_store_never_evicts_an_unsaved_extra_mode_under_heavy_pressure():
    # (b) persist-before-evict with an EXTRA UNSAVED record present (simulating a
    # freshly-computed GI sub-mode not yet on disk): the unpersisted frame is
    # NEVER thinned, even when the store is at/over the heavy cap.
    store = _live_store()

    # One frame carries a persisted primary mode PLUS an unsaved extra GI mode.
    store.upsert(_record(label=0, mode="q_total", source="/d/0.tif"), persisted=True)
    store.upsert(_record(label=0, mode="q_ip", source="/d/0.tif", scale=2.0),
                 persisted=False)
    assert not store.is_persisted(0)                 # extra mode unsaved

    # Flood the rest of the cap with fully-persisted frames, then overflow.
    for i in range(1, LIVE_STORE_HEAVY_CAP + 4):
        store.upsert(_record(label=i, source=f"/d/{i}.tif"), persisted=True)

    # The unsaved frame must still hold its heavy arrays (never evicted).
    assert store.has_heavy_payload(0)
    rec = store.get(0)
    assert rec is not None
    assert rec.view_1d("q_ip").intensity_1d is not None
    # Eviction happened among the persisted frames instead.
    persisted_thinned = [
        i
        for i in range(1, LIVE_STORE_HEAVY_CAP + 4)
        if not store.has_heavy_payload(i)
    ]
    assert persisted_thinned                         # some persisted frames thinned


def test_live_store_all_unpersisted_overflow_keeps_everything():
    # Corollary of (b): if EVERY heavy frame is unpersisted, nothing is
    # evictable, so the store exceeds the cap rather than dropping unsaved data.
    store = _live_store()
    for i in range(LIVE_STORE_HEAVY_CAP + 5):
        store.upsert(_record(label=i, source=f"/d/{i}.tif"), persisted=False)
    assert all(
        store.has_heavy_payload(i) for i in range(LIVE_STORE_HEAVY_CAP + 5)
    )


def test_live_store_mark_persisted_only_marks_passed_labels():
    # (c) mark_persisted marks ONLY the labels passed (mirrors "mark from
    # flush() only"): an unmentioned frame stays unpersisted and un-evictable.
    store = _live_store()
    store.upsert(_record(label=1, source="/d/1.tif"), persisted=False)
    store.upsert(_record(label=2, source="/d/2.tif"), persisted=False)
    assert not store.is_persisted(1)
    assert not store.is_persisted(2)

    store.mark_persisted([1])                         # only label 1

    assert store.is_persisted(1)
    assert not store.is_persisted(2)                  # 2 untouched

    # And it only marks the modes that exist on the passed label: marking a
    # missing label is a no-op (does not raise, marks nothing).
    store.mark_persisted([999])
    assert not store.is_persisted(999)


def test_live_store_mark_persisted_marks_all_current_modes_of_passed_label():
    # mark_persisted marks EVERY current mode of the passed label (so a frame
    # with a primary + extra GI mode becomes fully persisted, hence evictable).
    store = _live_store()
    store.upsert(_record(label=1, mode="q_total", source="/d/1.tif"), persisted=True)
    store.upsert(_record(label=1, mode="q_ip", source="/d/1.tif", scale=2.0),
                 persisted=False)
    assert not store.is_persisted(1)                  # q_ip unsaved

    store.mark_persisted(1)                            # flush wrote both modes
    assert store.is_persisted(1)
    rec = store.get(1)
    assert set(rec.modes_1d) == {"q_total", "q_ip"}


def test_live_store_set_hydrator_rehydrates_an_evicted_record_on_access():
    # (d) set_hydrator re-hydrates a thinned (evicted-arrays) record on access:
    # get_or_hydrate calls the hydrator exactly once and restores heavy arrays.
    store = _live_store()
    # Fill + overflow with persisted frames so the oldest is thinned.
    for i in range(LIVE_STORE_HEAVY_CAP + 1):
        store.upsert(_record(label=i, source=f"/d/{i}.tif"), persisted=True)

    thinned = next(
        i for i in range(LIVE_STORE_HEAVY_CAP + 1) if not store.has_heavy_payload(i)
    )

    calls: list[int] = []

    def hydrate(label):
        calls.append(label)
        return _record(label=label, source=f"/d/{label}.tif", scale=5.0)

    store.set_hydrator(hydrate)
    rec = store.get_or_hydrate(thinned)

    assert calls == [thinned]                          # hydrator called once
    assert store.has_heavy_payload(thinned)            # arrays restored
    assert rec is not None
    np.testing.assert_allclose(
        rec.view_1d("q_total").intensity_1d, [50.0, 100.0, 150.0]
    )

    # A frame whose arrays are still resident is returned WITHOUT re-hydrating.
    resident = next(
        i for i in range(LIVE_STORE_HEAVY_CAP + 1) if store.has_heavy_payload(i)
        and i != thinned
    )
    calls.clear()
    store.get_or_hydrate(resident)
    assert calls == []                                 # no hydrator call for resident
