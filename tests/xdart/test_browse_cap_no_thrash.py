"""Browse 1D cap raise (item 1) — a large Show-All stays fully resident.

With the byte-budget cap, a 3621-frame Show-All keeps every cheap 1D-light
publication resident (len(store) == N), so hydration succeeds once per frame and
there is no evict->re-request thrash.  Heavy raw/2D residency is unchanged
(max_heavy_items governs it), so MEM-1 is preserved.
"""

from types import SimpleNamespace

import numpy as np

from xrd_tools.core import browse_publication_max_items
from xrd_tools.core.containers import IntegrationResult1D
from xdart.modules.frame_publication import (
    PublicationStore, publication_from_live_frame)


def _light_1d_frame(i, nq=2000):
    return SimpleNamespace(
        idx=i,
        int_1d=IntegrationResult1D(
            radial=np.linspace(0.5, 5.0, nq, dtype=np.float32),
            intensity=np.full(nq, float(i), dtype=np.float32),
            sigma=np.ones(nq, dtype=np.float32), unit="q_A^-1"),
        int_2d=None, map_raw=None, mask=None, gi=False, gi_2d={},
        thumbnail=None, bg_raw=0, scan_info={},
        source_file=f"f{i}.tif", source_frame_idx=i)


def test_show_all_3621_stays_fully_resident_no_eviction():
    # The default 512-item cap would evict 3621-512 frames; the browse cap holds
    # them all -> no eviction, so Show All hydrates once and never re-requests.
    n = 3621
    assert browse_publication_max_items(npt=2000) >= n
    store = PublicationStore(max_items=browse_publication_max_items())
    for i in range(1, n + 1):
        store.upsert(publication_from_live_frame(_light_1d_frame(i)))
    assert len(store) == n, f"evicted: only {len(store)}/{n} resident"
    # every frame's 1d is resident (nothing downgraded/evicted)
    labels = set(store.labels())
    assert labels == set(range(1, n + 1))
    for i in (1, 1800, n):     # oldest, middle, newest all still have 1d
        item = store.get(i)
        assert item is not None and item.view.has_1d


def test_default_store_evicts_the_same_flood_confirms_cap_is_the_fix():
    # Control: the OLD default (512) DOES evict a 3621 flood -> the thrash source.
    n = 3621
    store = PublicationStore(max_items=512)
    for i in range(1, n + 1):
        store.upsert(publication_from_live_frame(_light_1d_frame(i)))
    assert len(store) == 512
    assert store.get(1) is None                 # oldest evicted


def test_browse_cap_does_not_touch_heavy_window_mem1():
    # MEM-1: raising max_items must NOT change the heavy (raw/2D) residency window.
    from xrd_tools.core import heavy_window
    store = PublicationStore(max_items=browse_publication_max_items())
    assert store._max_heavy_items == heavy_window()
