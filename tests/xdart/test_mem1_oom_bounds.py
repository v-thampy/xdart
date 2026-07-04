"""MEM-1a — live-mode OOM triad bounds.

Two coupled guards keep a GUI-behind-producer live run from growing unbounded:

* ``_BoundedFrameHandoff`` caps the wrangler→GUI display hand-off (drop-oldest),
  so a stalled ``update_data`` consumer can't let ``_published_frames`` grow to
  tens of GB of pinned LiveFrames.
* ``LiveFrameSeries`` frees each frame's ~18 MB raw when it leaves the write-side
  staging window, so live-mode raw residency is bounded by the cap instead of
  accumulating for every frame ever displayed.

See the memory-load review findings [1] and [5].
"""

import threading


# ── _BoundedFrameHandoff (MEM-1a step 1) ────────────────────────────────────
def test_bounded_frame_handoff_drops_oldest():
    from xdart.gui.tabs.static_scan.wranglers.wrangler_widget import (
        _BoundedFrameHandoff)

    d = _BoundedFrameHandoff(cap=8)
    for i in range(50):          # a producer 6x ahead of a stalled consumer
        d[i] = f"frame{i}"

    assert len(d) <= 8                              # never exceeds the cap
    assert set(d.keys()) == set(range(42, 50))      # keeps NEWEST, drops oldest
    assert d.get(49) == "frame49"                   # the idx about to be popped
    assert d.get(0) is None                         # a dropped idx reads None
    assert d.pop(0, None) is None                   # consumer-safe (no KeyError)

    d.clear()
    assert len(d) == 0


def test_default_handoff_cap_is_enforced():
    from xdart.gui.tabs.static_scan.wranglers.wrangler_widget import (
        _BoundedFrameHandoff, _PUBLISHED_FRAMES_CAP)

    d = _BoundedFrameHandoff()
    for i in range(_PUBLISHED_FRAMES_CAP * 4):
        d[i] = i
    assert len(d) == _PUBLISHED_FRAMES_CAP


def test_bounded_frame_handoff_tolerates_gui_pop_during_eviction():
    from xdart.gui.tabs.static_scan.wranglers.wrangler_widget import (
        _BoundedFrameHandoff)

    class RacingHandoff(_BoundedFrameHandoff):
        def __init__(self):
            super().__init__(cap=1)
            self._race_once = True

        def keys(self):
            keys = list(super().keys())
            if self._race_once and keys:
                self._race_once = False
                super().pop(keys[0], None)
            return keys

    d = RacingHandoff()
    d[0] = "old"
    d[1] = "new"

    assert d == {1: "new"}


# ── raw freed on staging-window eviction (MEM-1a step 2) ─────────────────────
class _FakeFrame:
    """Stand-in for a LiveFrame: only ``idx`` + ``free_raw`` are used by the
    staging cache, and ``map_raw`` stands in for the ~18 MB detector array."""

    def __init__(self, idx):
        self.idx = idx
        self.map_raw = object()
        self.freed = False

    def free_raw(self):
        self.map_raw = None
        self.freed = True
        return True


def _make_series():
    from xdart.modules.ewald.frame_series import LiveFrameSeries
    return LiveFrameSeries("mem1.nxs", threading.Lock())


def test_stash_eviction_frees_raw_of_persisted_frames():
    series = _make_series()
    series._in_memory_cap = 4

    frames = []
    for i in range(20):
        f = _FakeFrame(i)
        frames.append(f)
        series.stash(f)
        series.mark_persisted([i])   # durable => evictable on the next stash

    assert len(series._in_memory) <= series._in_memory_cap
    evicted = [f for f in frames if f.idx not in series._in_memory]
    assert evicted, "expected frames to leave the staging window"
    assert all(f.freed and f.map_raw is None for f in evicted)
    # resident frames keep their raw (still readable by the writer/display)
    for f in frames:
        if f.idx in series._in_memory:
            assert f.map_raw is not None


def test_evict_persisted_beyond_cap_frees_raw():
    series = _make_series()
    series._in_memory_cap = 4

    frames = [_FakeFrame(i) for i in range(10)]
    for f in frames:                       # fill directly (no stash eviction)
        series._in_memory[f.idx] = f
    series.mark_persisted(range(10))

    n = series.evict_persisted_beyond_cap()
    assert n == 6
    assert len(series._in_memory) == 4
    assert sum(1 for f in frames if f.freed) == 6


def test_unsaved_frames_are_never_evicted_or_freed():
    """persist-before-evict must still hold: an UNSAVED frame keeps its raw."""
    series = _make_series()
    series._in_memory_cap = 2

    frames = [_FakeFrame(i) for i in range(6)]
    for f in frames:
        series.stash(f)          # never marked persisted
    # nothing is evictable => cache exceeds cap, no raw freed (data-loss guard)
    assert len(series._in_memory) == 6
    assert not any(f.freed for f in frames)


# ── viewer raw LRU: cap wins over a select-all keep-set (MEM-1d) ─────────────
class _Rows(dict):
    """A dict that can carry the LRU order attribute (like FixSizeOrderedDict)."""


def test_viewer_raw_lru_cap_wins_over_select_all():
    from xdart.gui.tabs.static_scan.viewer_raw_lru import (
        remember_viewer_raw_lru, VIEWER_RAW_LIMIT)

    rows = _Rows()
    n = 100
    for i in range(n):
        rows[i] = {"map_raw": object(), "bg_raw": object()}

    keep = list(range(n))            # Cmd+A: keep-set covers the WHOLE stack
    for i in range(n):
        remember_viewer_raw_lru(rows, i, limit=VIEWER_RAW_LIMIT, keep=keep)

    order = getattr(rows, "_viewer_raw_lru_order")
    assert len(order) <= VIEWER_RAW_LIMIT               # cap wins, no OOM
    hydrated = [i for i in range(n) if rows[i]["map_raw"] is not None]
    assert len(hydrated) <= VIEWER_RAW_LIMIT
    assert set(order).issubset(range(n - VIEWER_RAW_LIMIT, n))  # newest survive


def test_viewer_raw_lru_protects_small_keep_set():
    """A keep-set smaller than the cap still shields displayed labels."""
    from xdart.gui.tabs.static_scan.viewer_raw_lru import (
        remember_viewer_raw_lru, VIEWER_RAW_LIMIT)

    rows = _Rows()
    for i in range(VIEWER_RAW_LIMIT + 20):
        rows[i] = {"map_raw": object()}
    keep = [0, 1]                    # two "displayed" labels, well under the cap
    for i in range(VIEWER_RAW_LIMIT + 20):
        remember_viewer_raw_lru(rows, i, limit=VIEWER_RAW_LIMIT, keep=keep)

    # the protected labels survive even though they are the oldest
    assert rows[0]["map_raw"] is not None
    assert rows[1]["map_raw"] is not None
    order = getattr(rows, "_viewer_raw_lru_order")
    assert len(order) <= VIEWER_RAW_LIMIT
