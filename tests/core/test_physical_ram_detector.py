"""One physical-RAM detector for every RAM-aware default, Windows included (1.2.10 review M5).

``total_physical_ram_bytes`` read ``os.sysconf`` only, so on Windows the reduction worker cap, the heavy window, the
Live record and Browse publication caps and the Browse 1D cache budget all fell back as if RAM were unknown. The
maintainer's pick (2026-11-11): "Move them (Recommended)". A Windows host is simulated at the operating-system
boundary only: no ``os.sysconf`` (as on Windows) and ``ctypes.windll.kernel32.GlobalMemoryStatusEx`` answering, so
the real Windows reader and the real detector run. Each caller keeps its own policy.
"""
from __future__ import annotations

import ctypes
import os
import sys
from types import SimpleNamespace

import pytest

from xrd_tools.core import staging
from xrd_tools.io import browse_1d_cache

_GIB = 1024 ** 3


def _windows_host(monkeypatch, total_bytes):
    """This process as a Windows host with ``total_bytes`` of RAM (0: GlobalMemoryStatusEx fails)."""

    def global_memory_status_ex(reference):
        if not total_bytes:
            return 0
        reference._obj.ullTotalPhys = total_bytes
        return 1

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delattr(os, "sysconf", raising=False)
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(
        kernel32=SimpleNamespace(GlobalMemoryStatusEx=global_memory_status_ex)), raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    monkeypatch.delenv(staging.ENV_OVERRIDE, raising=False)
    monkeypatch.delenv(staging.REDUCTION_WORKERS_ENV, raising=False)


def test_an_8_gib_windows_host_gets_the_ram_aware_defaults(monkeypatch):
    """The detector reads Windows RAM, and every routed caller sizes from it as on macOS or Linux with 8 GiB."""
    _windows_host(monkeypatch, 8 * _GIB)
    assert staging.total_physical_ram_bytes() == 8 * _GIB
    assert staging.reduction_worker_cap(None) == 2
    assert staging.reduction_worker_cap(6) == 6  # a deliberate Cores choice is never trimmed
    assert staging.heavy_window() == staging.MIN_WINDOW
    assert staging.live_record_store_max_items(1000) == int(0.05 * 8 * _GIB / staging.live_record_trace_bytes(1000))
    assert staging.browse_publication_max_items(1000) == staging.live_record_store_max_items(1000)
    assert browse_1d_cache.default_browse_1d_cache_budget() == 8 * _GIB // 20


def test_a_32_gib_windows_host_moves_only_the_1d_record_caps(monkeypatch):
    """What his pick moves on a 32 GiB Windows PC: the 1D record caps rise; workers, the heavy window and the Browse
    1D cache stay where the unknown-RAM fallback had them."""
    _windows_host(monkeypatch, 32 * _GIB)
    assert staging.reduction_worker_cap(None) == staging.DEFAULT_REDUCTION_WORKERS
    assert staging.heavy_window() == staging.DEFAULT_WINDOW
    assert staging.live_record_store_max_items(None) == 7953
    assert staging.browse_publication_max_items(None) == 4971
    assert browse_1d_cache.default_browse_1d_cache_budget() == _GIB


def test_a_windows_host_whose_ram_cannot_be_read_keeps_every_fallback(monkeypatch):
    """Unknown RAM keeps each caller's own fallback: four workers, the 64-frame window, the 4096-trace floors and the
    1 GiB Browse 1D cache."""
    _windows_host(monkeypatch, 0)
    assert staging.total_physical_ram_bytes() is None
    assert staging.reduction_worker_cap(None) == staging.DEFAULT_REDUCTION_WORKERS
    assert staging.heavy_window() == staging.DEFAULT_WINDOW
    assert staging.live_record_store_max_items(1000) == staging.LIVE_RECORD_MIN_ITEMS
    assert staging.browse_publication_max_items(1000) == staging.LIVE_RECORD_MIN_ITEMS
    assert browse_1d_cache.default_browse_1d_cache_budget() == _GIB


@pytest.mark.skipif(sys.platform != "win32", reason="the real GlobalMemoryStatusEx exists only on Windows")
def test_the_detector_reads_this_windows_host():
    """On a real Windows runner the one detector returns the Windows reading."""
    total = staging.total_physical_ram_bytes()
    assert type(total) is int and total > 0
    assert total == staging.windows_physical_ram_bytes()
