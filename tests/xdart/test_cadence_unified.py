# -*- coding: utf-8 -*-
"""Phase 4b-3 — the serial and streaming save-cadence predicates now route
through ONE headless FlushPolicy (the divergence is closed).

Each predicate keeps its own inputs by design (serial owns the live unsaved
count; the streaming sink tracks only its own counter), so the test proves
each REPRODUCES the standalone policy on its inputs — not that they agree
with each other on differing inputs.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from xrd_tools.reduction import FlushPolicy
from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import imageThread


def _serial(frames_since, *, cap, interval, unsaved, xye_only=False):
    """Drive the real imageThread._save_due on a duck wrangler/scan."""
    frames = SimpleNamespace(
        _in_memory_cap=cap,
        unsaved_in_memory_count=(lambda: unsaved) if unsaved is not None else None,
    )
    scan = SimpleNamespace(frames=frames)
    host = SimpleNamespace(xye_only=xye_only, _frames_since_save=frames_since,
                           LIVE_SAVE_INTERVAL=interval)
    return imageThread._save_due(host, scan), scan


@pytest.mark.parametrize("frames_since", [0, 1, 7, 8, 55, 56, 57])
@pytest.mark.parametrize("unsaved", [None, 3, 55, 56])
@pytest.mark.parametrize("interval", [8, 1000])
@pytest.mark.parametrize("force", [False, True])
def test_serial_save_due_reproduces_flush_policy(frames_since, unsaved, interval, force):
    frames = SimpleNamespace(
        _in_memory_cap=64,
        unsaved_in_memory_count=(lambda: unsaved) if unsaved is not None else None,
    )
    scan = SimpleNamespace(frames=frames)
    host = SimpleNamespace(xye_only=False, _frames_since_save=frames_since,
                           LIVE_SAVE_INTERVAL=interval)
    got = imageThread._save_due(host, scan, force=force)
    expected = FlushPolicy(interval=interval, cap=64).should_flush(
        frames_since_flush=frames_since, unsaved_in_memory=unsaved, force=force)
    assert got is expected


def test_serial_save_due_xye_only_never_saves():
    got, _ = _serial(999, cap=64, interval=8, unsaved=999, xye_only=True)
    assert got is False


def test_streaming_due_to_save_reproduces_flush_policy(tmp_path):
    """The QtNexusSink streaming predicate reproduces FlushPolicy with
    unsaved_in_memory=None (it tracks only its own counter) — batch uses the
    cap pressure bound, live uses LIVE_SAVE_INTERVAL."""
    from xdart.modules.ewald import LiveScan
    from xdart.gui.tabs.static_scan.wranglers.qt_nexus_sink import (
        QtNexusSink, _SAVE_BEFORE_EVICT_MARGIN,
    )
    from xrd_tools.reduction import ReductionPlan

    cap = 12
    for batch_mode, interval in ((True, cap), (False, 8)):
        scan = LiveScan(data_file=str(tmp_path / f"c_{batch_mode}.nxs"))
        scan.frames._in_memory_cap = cap
        host = SimpleNamespace(batch_mode=batch_mode, LIVE_SAVE_INTERVAL=8)
        sink = QtNexusSink(host, scan, ReductionPlan(integration_2d=None), mask=None)
        policy = FlushPolicy(interval=interval, cap=cap,
                             margin=_SAVE_BEFORE_EVICT_MARGIN)
        for since in (0, 1, 3, 4, 7, 8, 11, 12):
            sink._since_save = since
            got = sink._due_to_save()
            expected = policy.should_flush(frames_since_flush=since)
            assert got is expected, (batch_mode, since, got, expected)
