# -*- coding: utf-8 -*-
"""PF-1e outcome tests: source-only rows hydrate their raw through the GUI route.

The live-reproduced blocker: an old Append row carries a valid ``source``
reference but no stored thumbnail (``raw_status`` ``"missing"``).
``PublicationStore.get_or_hydrate`` short-circuited on the resident lighter
item, so the hydration worker reported success while the full-purpose
residency check (``view.raw is not None``) scored false — retried, suppressed,
and the raw detector panel stayed blank even though the headless
``load_processed_raw_or_thumbnail`` fallback could read the image the whole
time.  Newly appended rows with thumbnails masked the same gap because the
panel fell back to the thumbnail tier.

These tests drive the PRODUCTION worker route (the registered
``DisplayDataMixin._rehydrate_publication`` hydrator invoked via
``store.get_or_hydrate``, exactly what ``FrameHydrationWorker._hydrate_full``
calls) against a real writer-produced target, and score residency with the
REAL ``displayFrameWidget._view_has_hydration_payload``.  Thumbnail backfill
stays optional: hydration must not rewrite the file.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np

from tests.xdart.test_append_flush_durability import (
    NOLD,
    NNEW,
    _make_1d_only_target,
    _run_int2d_append,
)


def _completed_target(tmp_path):
    """A real completed Int 2D Append target: old rows source-only (no stored
    thumbnail), new tail rows with thumbnails."""
    nxs, _ = _make_1d_only_target(tmp_path)
    _run_int2d_append(nxs, revisit=range(NOLD), new=range(NOLD, NOLD + NNEW))
    return nxs


def _worker_route(nxs, labels):
    """Seed the store as browsing does, then hydrate via the worker's calls."""
    from xdart.gui.tabs.static_scan.display_data import DisplayDataMixin
    from xdart.modules.ewald import LiveScan
    from xdart.modules.frame_publication import (
        PublicationStore,
        publication_from_nexus_frame,
    )

    scan = LiveScan(data_file=nxs)
    scan.load_from_h5()

    class _Host(DisplayDataMixin):
        def __init__(self, scan, store):
            self.scan = scan
            self.publication_store = store
            self._processing_active = False

    store = PublicationStore()
    host = _Host(scan, store)
    store.set_hydrator(host._rehydrate_publication)

    out = {}
    for label in labels:
        seed = publication_from_nexus_frame(nxs, label,
                                            generation=store.generation)
        store.upsert(seed)
        out[label] = (seed, store.get_or_hydrate(label))
    return out


def _scored_full_resident(view) -> bool:
    """The REAL RL-1 tier-accurate residency scoring for a 'full' request."""
    from xdart.gui.tabs.static_scan.display_frame_widget import displayFrameWidget
    # the method reads only its arguments; call unbound to avoid Qt widget init
    return bool(displayFrameWidget._view_has_hydration_payload(None, view, "full"))


def test_source_only_row_hydrates_raw_through_worker_route(tmp_path):
    nxs = _completed_target(tmp_path)
    old_label, new_label = 3, NOLD + 2

    results = _worker_route(nxs, (old_label, new_label))

    seed_old, got_old = results[old_label]
    assert seed_old.raw_status == "missing"          # source-only: no thumbnail
    assert seed_old.view.thumbnail is None
    assert got_old is not None
    assert got_old.view.raw is not None, \
        "source-only row's raw must hydrate via the lazy source fallback"
    assert _scored_full_resident(got_old.view), \
        "full-purpose completion must score RESIDENT (no retry treadmill)"

    seed_new, got_new = results[new_label]
    assert seed_new.raw_status == "thumbnail"
    assert got_new is not None and got_new.view.raw is not None
    assert _scored_full_resident(got_new.view)


def test_live_publication_with_retained_empty_raw_ref_hydrates_source(tmp_path):
    """Live Append keeps a lightweight ``raw_ref`` after ``map_raw`` is
    released.  The reference object is not itself a resident raw payload: a
    source-backed publication in this state must still enter the hydrator.

    This is the production shape that the first PF-1e correction missed.  Its
    browser-seeded test used ``raw_ref=None``, while the real live publication
    had ``raw_ref is not None`` and ``raw_ref.map_raw is None``.
    """
    from xdart.gui.tabs.static_scan.display_data import DisplayDataMixin
    from xdart.modules.ewald import LiveScan
    from xdart.modules.frame_publication import (
        PublicationStore,
        publication_from_live_frame,
    )

    nxs = _completed_target(tmp_path)
    scan = LiveScan(data_file=nxs)
    scan.load_from_h5()
    light_frame = scan.frames[3]
    assert light_frame.map_raw is None
    assert light_frame.source_file

    publication = publication_from_live_frame(light_frame)
    assert publication.raw_ref is light_frame
    assert publication.raw_ref.map_raw is None
    assert publication.view.raw is None

    class _Host(DisplayDataMixin):
        def __init__(self, scan, store):
            self.scan = scan
            self.publication_store = store
            self._processing_active = False

    store = PublicationStore()
    host = _Host(scan, store)
    store.set_hydrator(host._rehydrate_publication)
    store.upsert(publication)

    hydrated = store.get_or_hydrate(3)
    assert hydrated is not None
    assert hydrated.view.raw is not None, (
        "a retained lightweight raw_ref must not suppress lazy source hydration"
    )
    assert _scored_full_resident(hydrated.view)


def test_live_publication_snapshots_resident_raw_before_raw_ref_release(tmp_path):
    """A live publication may borrow raw pixels through ``raw_ref`` while its
    immutable view still has ``raw=None``.  That borrowed array can be released
    by the live-memory bound at any time, so it must not suppress full-purpose
    hydration.  Hydration snapshots the array into the published view before
    the mutable ``LiveFrame`` is freed.

    This is the intermittent cake-present/raw-blank shape seen after an Int 2D
    Append: eligibility observes a populated ``raw_ref.map_raw``, then the live
    frame is thinned before the selected publication is rendered.
    """
    from xdart.gui.tabs.static_scan.display_data import DisplayDataMixin
    from xdart.modules.ewald import LiveScan
    from xdart.modules.frame_publication import (
        PublicationStore,
        publication_from_live_frame,
    )

    nxs = _completed_target(tmp_path)
    scan = LiveScan(data_file=nxs)
    scan.load_from_h5()
    live_frame = scan.frames[3]
    assert live_frame.map_raw is None
    live_frame._lazy_load_raw()
    expected = np.asarray(live_frame.map_raw).copy()

    publication = publication_from_live_frame(live_frame, include_raw=False)
    assert publication.raw_ref is live_frame
    assert publication.raw_ref.map_raw is not None
    assert publication.view.raw is None

    class _Host(DisplayDataMixin):
        def __init__(self, scan, store):
            self.scan = scan
            self.publication_store = store
            self._processing_active = False

    store = PublicationStore()
    host = _Host(scan, store)
    store.set_hydrator(host._rehydrate_publication)
    store.upsert(publication)

    hydrated = store.get_or_hydrate(3)
    assert hydrated is not None
    assert hydrated.view.raw is not None, (
        "borrowed live raw must be promoted into the immutable publication view"
    )
    assert live_frame.free_raw()
    assert live_frame.map_raw is None
    np.testing.assert_array_equal(np.asarray(hydrated.view.raw), expected)
    assert _scored_full_resident(hydrated.view)


def test_fix_gates_on_source_recoverability(tmp_path, monkeypatch):
    """Fail-before pin: with the PF-1e eligibility removed (the 121bc438
    predicate), the worker route returns the light item un-hydrated and the
    full-purpose score is false — the exact blank-panel treadmill."""
    import xdart.modules.frame_publication as fp

    monkeypatch.setattr(fp, "publication_raw_recoverable", lambda _p: False)
    nxs = _completed_target(tmp_path)

    seed, got = _worker_route(nxs, (3,))[3]
    assert seed.raw_status == "missing"
    assert got is not None                    # worker still reports "success"
    assert got.view.raw is None               # ...but the raw never hydrates
    assert not _scored_full_resident(got.view)


def test_thumbnail_backfill_stays_optional(tmp_path):
    """Hydrating a source-only row must not rewrite the file: no thumbnail is
    persisted for the old prefix (acceptance: recoverable raw preview without
    eagerly hydrating or persisting old rows)."""
    nxs = _completed_target(tmp_path)

    def _thumbless_labels():
        out = set()
        with h5py.File(nxs, "r") as f:
            fg = f["entry/frames"]
            for name in fg:
                if "thumbnail" not in fg[name]:
                    out.add(int(name.split("_")[-1]))
        return out

    before = _thumbless_labels()
    assert 3 in before                        # the old prefix is source-only
    mtime_before = os.path.getmtime(nxs)

    got = _worker_route(nxs, (3,))[3][1]
    assert got.view.raw is not None

    assert _thumbless_labels() == before
    assert os.path.getmtime(nxs) == mtime_before


def test_hydrated_raw_matches_headless_source_fallback(tmp_path):
    """The GUI-hydrated raw is the same image the headless reader returns."""
    from xrd_tools.io.image_source import load_processed_raw_or_thumbnail

    nxs = _completed_target(tmp_path)
    got = _worker_route(nxs, (3,))[3][1]
    headless = load_processed_raw_or_thumbnail(nxs, 3)
    headless_arr = np.asarray(getattr(headless, "image", headless))
    np.testing.assert_array_equal(np.asarray(got.view.raw), headless_arr)
