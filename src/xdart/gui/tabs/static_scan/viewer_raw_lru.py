# -*- coding: utf-8 -*-
"""Shared hydrated-raw LRU for the display caches (D5).

``viewer_rows_2d`` payloads may carry full detector arrays (``map_raw``/``bg_raw``,
~18 MB each for an Eiger).  This module owns the ONE bounded LRU that caps
how many stay hydrated, shared by every writer — the GUI thread (H5Viewer
absorb/load paths) and the worker threads (reintegrate publish, full-reload
load_frames) — which is the point: per-writer order lists let thread-side
inserts pile up past the cap (the pre-1.0 D5 gap).

The order list rides ON the shared ``viewer_rows_2d`` object itself (the
``FixSizeOrderedDict`` created once in staticWidget and handed to every
thread), so sharing needs no constructor plumbing.  Callers MUST hold the
``data_lock`` that guards ``viewer_rows_2d``; the helpers do no locking of their
own.  A plain-``dict`` ``viewer_rows_2d`` (bare headless constructions) cannot
carry the attribute — the helpers then no-op, which is safe: those dicts
are private to their owner and never accumulate across writers.

Evicting here only drops the full-resolution pixels (``map_raw``/``bg_raw``
-> None); thumbnails and integrated payloads stay, and the display layer
lazily rehydrates from the source/.nxs on demand.  Persist-before-evict is
not in play: this cache never holds the only copy of unsaved data
(LiveFrameSeries._in_memory owns that invariant).
"""
from __future__ import annotations

__all__ = ["VIEWER_RAW_LIMIT", "remember_viewer_raw_lru", "clear_viewer_raw_lru"]

#: default cap on hydrated full-resolution payloads in viewer_rows_2d.
VIEWER_RAW_LIMIT = 8

_ORDER_ATTR = "_viewer_raw_lru_order"


def remember_viewer_raw_lru(
    viewer_rows_2d,
    idx,
    limit: int = VIEWER_RAW_LIMIT,
    *,
    keep=(),
) -> list[int]:
    """Mark ``idx`` most-recently hydrated; evict past ``limit``.

    Caller must hold the ``data_lock`` guarding ``viewer_rows_2d``.

    Returns the evicted labels so callers that mirror raw payloads in a
    parallel cache (notably the Image Viewer rows) can release those too.
    ``keep`` protects currently displayed labels even if the order is over
    the nominal limit.
    """
    order = getattr(viewer_rows_2d, _ORDER_ATTR, None)
    if order is None:
        order = []
        try:
            setattr(viewer_rows_2d, _ORDER_ATTR, order)
        except AttributeError:
            return []                   # plain dict (headless) — no shared cap
    idx = int(idx)
    if idx in order:
        order.remove(idx)
    order.append(idx)
    limit = max(1, int(limit))
    keep_set = {int(k) for k in keep}
    evicted = []
    while len(order) > limit:
        # Prefer to evict the oldest NON-kept payload so a currently displayed
        # label survives.
        stale = next((candidate for candidate in order
                      if candidate not in keep_set), None)
        if stale is None:
            # MEM-1d: every remaining entry is in the keep-set yet we are STILL
            # over the cap — e.g. Cmd+A select-all on a 5000-frame stack makes
            # the keep-set cover everything, which would disable eviction and
            # hydrate ~64 MB/frame until OOM.  The cap WINS: drop the oldest kept
            # label.  It renders from whatever is resident and re-hydrates on
            # scroll (the store-first path), instead of OOMing on one keystroke.
            stale = order[0]
        order.remove(stale)
        payload = viewer_rows_2d.get(stale)
        if payload is not None:
            payload["map_raw"] = None
            if "bg_raw" in payload:
                payload["bg_raw"] = None   # full bg images pin like raws
        evicted.append(stale)
    return evicted


def clear_viewer_raw_lru(viewer_rows_2d) -> None:
    """Reset the shared order (call where ``viewer_rows_2d`` itself is cleared)."""
    order = getattr(viewer_rows_2d, _ORDER_ATTR, None)
    if order is not None:
        del order[:]
