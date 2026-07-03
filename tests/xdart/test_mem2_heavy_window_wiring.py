"""MEM-2 wiring — the RAM-aware window feeds all three live heavy caps.

The pure ``heavy_window`` math is covered in ``tests/core/test_heavy_window.py``;
this checks the three consumers actually take it:
  * ``PublicationStore`` default (its own RAM-aware default),
  * ``LiveFrameSeries._in_memory_cap`` (staging), and
  * ``FrameRecordStore.max_heavy_items`` (record store)
all resolve to the same window.  ``XDART_HEAVY_WINDOW`` pins it for determinism.
"""

import threading
from types import SimpleNamespace


def test_publication_store_default_is_ram_aware(monkeypatch):
    monkeypatch.setenv("XDART_HEAVY_WINDOW", "24")
    from xdart.modules.frame_publication import PublicationStore
    assert PublicationStore()._max_heavy_items == 24


def test_publication_store_explicit_cap_still_honored():
    from xdart.modules.frame_publication import PublicationStore
    assert PublicationStore(max_heavy_items=7)._max_heavy_items == 7


def test_thread_heavy_window_uses_override_and_caches(monkeypatch):
    monkeypatch.setenv("XDART_HEAVY_WINDOW", "32")
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
        imageThread)
    worker = imageThread.__new__(imageThread)
    worker.detector_shape = (2167, 2070)
    scan = SimpleNamespace(frames=None)
    win = imageThread._heavy_staging_window(worker, scan)
    assert win == 32
    assert worker._heavy_window == 32                 # cached


def test_all_three_caps_share_the_same_window(monkeypatch):
    monkeypatch.setenv("XDART_HEAVY_WINDOW", "40")
    from xdart.modules.frame_publication import PublicationStore
    from xdart.modules.ewald.frame_series import LiveFrameSeries
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
        imageThread)
    from xrd_tools.session import FrameRecordStore

    # (1) publication store — its own RAM-aware default
    assert PublicationStore()._max_heavy_items == 40

    # (2)+(3) staging + record store, as _get_streaming_session wires them
    worker = imageThread.__new__(imageThread)
    worker.detector_shape = (2167, 2070)
    scan = SimpleNamespace(frames=LiveFrameSeries("m.nxs", threading.Lock()))
    window = imageThread._heavy_staging_window(worker, scan)
    scan.frames._in_memory_cap = window
    record_store = FrameRecordStore(max_items=512, max_heavy_items=window)

    assert window == 40
    assert scan.frames._in_memory_cap == 40
    assert record_store._max_heavy_items == 40


def test_thread_window_falls_back_when_shape_unknown(monkeypatch):
    # no override, no detector shape -> the RAM-tier fallback (a valid 16/32/64)
    monkeypatch.delenv("XDART_HEAVY_WINDOW", raising=False)
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
        imageThread)
    worker = imageThread.__new__(imageThread)
    worker.detector_shape = None
    win = imageThread._heavy_staging_window(worker, SimpleNamespace(frames=None))
    assert win in (16, 32, 64)
