# -*- coding: utf-8 -*-
"""Shared hydrated-raw LRU for the display caches (D5).

``data_2d`` payloads may carry full detector arrays (``map_raw``/``bg_raw``,
~18 MB each for an Eiger).  This module owns the ONE bounded LRU that caps
how many stay hydrated, shared by every writer — the GUI thread (H5Viewer
absorb/load paths) and the worker threads (reintegrate publish, full-reload
load_frames) — which is the point: per-writer order lists let thread-side
inserts pile up past the cap (the pre-1.0 D5 gap).

The order list rides ON the shared ``data_2d`` object itself (the
``FixSizeOrderedDict`` created once in staticWidget and handed to every
thread), so sharing needs no constructor plumbing.  Callers MUST hold the
``data_lock`` that guards ``data_2d``; the helpers do no locking of their
own.  A plain-``dict`` ``data_2d`` (bare headless constructions) cannot
carry the attribute — the helpers then no-op, which is safe: those dicts
are private to their owner and never accumulate across writers.

Evicting here only drops the full-resolution pixels (``map_raw``/``bg_raw``
-> None); thumbnails and integrated payloads stay, and the display layer
lazily rehydrates from the source/.nxs on demand.  Persist-before-evict is
not in play: this cache never holds the only copy of unsaved data
(LiveFrameSeries._in_memory owns that invariant).
"""
from __future__ import annotations

__all__ = ["HYDRATED_RAW_LIMIT", "remember_hydrated_raw", "clear_hydrated_raw"]

#: default cap on hydrated full-resolution payloads in data_2d.
HYDRATED_RAW_LIMIT = 8

_ORDER_ATTR = "_hydrated_raw_order"


def remember_hydrated_raw(
    data_2d,
    idx,
    limit: int = HYDRATED_RAW_LIMIT,
    *,
    keep=(),
) -> list[int]:
    """Mark ``idx`` most-recently hydrated; evict past ``limit``.

    Caller must hold the ``data_lock`` guarding ``data_2d``.

    Returns the evicted labels so callers that mirror raw payloads in a
    parallel cache (notably the Image Viewer rows) can release those too.
    ``keep`` protects currently displayed labels even if the order is over
    the nominal limit.
    """
    order = getattr(data_2d, _ORDER_ATTR, None)
    if order is None:
        order = []
        try:
            setattr(data_2d, _ORDER_ATTR, order)
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
        stale = next((candidate for candidate in order
                      if candidate not in keep_set), None)
        if stale is None:
            break
        order.remove(stale)
        payload = data_2d.get(stale)
        if payload is not None:
            payload["map_raw"] = None
            if "bg_raw" in payload:
                payload["bg_raw"] = None   # full bg images pin like raws
        evicted.append(stale)
    return evicted


def clear_hydrated_raw(data_2d) -> None:
    """Reset the shared order (call where ``data_2d`` itself is cleared)."""
    order = getattr(data_2d, _ORDER_ATTR, None)
    if order is not None:
        del order[:]
