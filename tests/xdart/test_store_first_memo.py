# -*- coding: utf-8 -*-
"""MEM-1[14]: the live render must resolve only NEW/changed publications per tick.

Drives the REAL DisplayDataMixin store-first read path against real
FrameRecordStore + PublicationStore (the test_store_first_reads pattern -- no
fakes on the seam).  Locks the memo's two contracts: O(k) resolver calls per tick
(not O(N)) and byte-equivalence to a fresh build.
"""
from __future__ import annotations

from threading import RLock
from types import MethodType, SimpleNamespace

import numpy as np

from xdart.gui.tabs.static_scan.display_data import DisplayDataMixin
from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
from xdart.modules.frame_publication import (
    PublicationStore,
    publication_from_frame_view,
)
from xrd_tools.core import (
    FrameRecord,
    FrameView,
    IntegrationResult1D,
    assert_frameview_equivalent,
)
from xrd_tools.session import FrameRecordStore


def _r1d(scale=1.0, *, unit="q_A^-1"):
    radial = np.linspace(0.5, 3.5, 4)
    intensity = scale * np.array([2.0, 4.0, 8.0, 16.0])
    return IntegrationResult1D(
        radial=radial, intensity=intensity, sigma=np.sqrt(intensity), unit=unit)


def _host(store, publications):
    host = SimpleNamespace(
        scan=SimpleNamespace(gi=False, bai_1d_args={}, bai_2d_args={}),
        data_lock=RLock(),
        viewer_rows_1d={},
        viewer_rows_2d={},
        publication_store=publications,
        _frame_record_store=store,
        wrangler=None,
        viewer_mode=None,
        display_generation=0,
    )
    for name in ("_active_frame_record_modes", "_active_frame_record_store",
                 "_publication_frame_view", "store_first_frame_view"):
        setattr(host, name, MethodType(getattr(staticWidget, name), host))
    host._coerce_frame_label = staticWidget._coerce_frame_label
    for name in ("_selected_publication_views", "_first_present",
                 "_publication_legacy_parts", "_display_publication_from_view",
                 "_store_first_publication_for_display", "_store_first_pub_cache",
                 "_store_first_cache_ident"):
        setattr(host, name, MethodType(getattr(DisplayDataMixin, name), host))
    host._display_hydration_should_block = (
        lambda allow_blocking_read=None: bool(allow_blocking_read))
    return host


def _populate(n):
    store = FrameRecordStore(max_heavy_items=None)
    publications = PublicationStore(max_heavy_items=None)
    for i in range(1, n + 1):
        v = FrameView.from_results(
            label=i, result_1d=_r1d(scale=float(i)), metadata_raw={"i0": 1.0})
        store.upsert(FrameRecord.from_view(v))
        publications.upsert(publication_from_frame_view(v))
    return store, publications


def _count_builds(host):
    """Wrap _display_publication_from_view (the expensive build) to record calls."""
    calls = []
    orig = host._display_publication_from_view

    def _spy(idx, view):
        calls.append(idx)
        return orig(idx, view)

    host._display_publication_from_view = _spy
    return calls


def _tick(host, n):
    for i in range(1, n + 1):
        host._store_first_publication_for_display(i)


def test_memo_resolves_only_new_labels_per_tick():
    N = 20
    store, publications = _populate(N)
    host = _host(store, publications)
    calls = _count_builds(host)

    # Tick 1 (cold cache): O(N) builds.
    _tick(host, N)
    assert len(calls) == N

    # Tick 2 (nothing changed): ZERO builds -- all cache hits.
    calls.clear()
    _tick(host, N)
    assert calls == []

    # One new frame published: exactly ONE build (O(k)), not O(N).
    v = FrameView.from_results(
        label=N + 1, result_1d=_r1d(scale=99.0), metadata_raw={"i0": 1.0})
    store.upsert(FrameRecord.from_view(v))
    publications.upsert(publication_from_frame_view(v))
    calls.clear()
    _tick(host, N + 1)
    assert calls == [N + 1]


def test_rehydrated_frame_rebuilds_but_others_do_not():
    N = 10
    store, publications = _populate(N)
    host = _host(store, publications)
    _tick(host, N)                                   # warm
    calls = _count_builds(host)

    # Re-publish frame 5 (a new publication object -> identity changes).
    v = FrameView.from_results(
        label=5, result_1d=_r1d(scale=500.0), metadata_raw={"i0": 1.0})
    store.upsert(FrameRecord.from_view(v))
    publications.upsert(publication_from_frame_view(v))

    _tick(host, N)
    assert calls == [5]                              # only the changed label


def test_memoized_publication_equivalent_to_fresh_build():
    store, publications = _populate(3)
    host = _host(store, publications)

    fresh = host._store_first_publication_for_display(2)     # cold build
    cached = host._store_first_publication_for_display(2)    # cache hit
    assert cached is not None and fresh is not None
    assert_frameview_equivalent(cached.view, fresh.view)
    assert cached.generation == fresh.generation

    # A display_generation bump must be reflected on the hit (cheap re-stamp),
    # with the data otherwise byte-identical.
    host.display_generation = 7
    restamped = host._store_first_publication_for_display(2)
    assert_frameview_equivalent(restamped.view, fresh.view)
    assert restamped.generation == 7


def test_memo_dropped_on_store_generation_bump():
    store, publications = _populate(3)
    host = _host(store, publications)
    _tick(host, 3)                                   # warm

    # Scan boundary: both backing stores emptied (publication clear bumps the
    # store generation, which drops the memo).
    host._frame_record_store = FrameRecordStore(max_heavy_items=None)
    publications.clear()
    # Cache dropped + frame gone -> None, never a stale hit.
    assert host._store_first_publication_for_display(2) is None
