# -*- coding: utf-8 -*-
"""PERF-3 O(new) live tick: during a live Overlay/Waterfall run, the per-tick
resolution must scope to only the not-yet-accumulated frames (O(drain)), not the
whole selection (O(index)).  The accumulator (append-only) owns the rest.
"""
from __future__ import annotations

from types import SimpleNamespace

from xdart.gui.tabs.static_scan.display_controllers import (
    _live_overlay_render_labels,
)


def _widget(*, processing, method, accumulated_ids, scan_name="scan"):
    history = (
        SimpleNamespace(ids=list(accumulated_ids))
        if accumulated_ids is not None else None)
    return SimpleNamespace(
        _processing_active=processing,
        scan=SimpleNamespace(name=scan_name),
        ui=SimpleNamespace(plotMethod=SimpleNamespace(currentText=lambda: method)),
        _waterfall_history=history,
    )


def test_live_overlay_tick_is_o_new_not_o_index():
    N = 3400
    labels = tuple(range(1, N + 1))
    acc = [("scan", i) for i in range(1, N - 2)]        # 1..N-3 accumulated
    w = _widget(processing=True, method="Overlay", accumulated_ids=acc)
    scoped = _live_overlay_render_labels(w, labels)
    assert set(scoped) == {N - 2, N - 1, N}            # only the 3 new frames
    assert len(scoped) == 3                             # O(drain), not O(3400)


def test_waterfall_slice_mode_3tuple_ids_decode():
    # Slice-mode rows are 3-tuples (scan_key, frame_idx, projection_id).
    labels = tuple(range(1, 6))
    acc = [("scan", i, "qz") for i in range(1, 4)]      # 1,2,3 accumulated
    w = _widget(processing=True, method="Waterfall", accumulated_ids=acc)
    assert set(_live_overlay_render_labels(w, labels)) == {4, 5}


def test_live_overlay_tick_does_not_confuse_reused_index_across_scans():
    """Directory scans commonly restart at frame 0; scan A's row is not scan B's."""
    w = _widget(
        processing=True,
        method="Overlay",
        accumulated_ids=[("scanA", 0)],
        scan_name="scanB",
    )

    assert _live_overlay_render_labels(w, (0,)) == (0,)


def test_live_overlay_tick_still_filters_reused_index_in_current_scan():
    w = _widget(
        processing=True,
        method="Overlay",
        accumulated_ids=[("scanA", 0), ("scanB", 0)],
        scan_name="scanB",
    )

    assert _live_overlay_render_labels(w, (0,)) == ()


def test_live_overlay_tick_uses_frame_source_when_scan_identity_is_unset():
    """Pre-scan capture must not collapse two source scans to ``(None, 0)``."""
    w = _widget(
        processing=True,
        method="Overlay",
        accumulated_ids=[("scanA", 0)],
        scan_name=None,
    )
    w.frame = SimpleNamespace(source_file="/data/scanB_0000.tif")

    assert _live_overlay_render_labels(w, (0,)) == (0,)


def test_full_reseed_when_accumulator_empty_or_absent():
    labels = tuple(range(1, 11))
    # Mode entry / reset: accumulator empty -> full reseed (one hit, by design).
    assert _live_overlay_render_labels(
        _widget(processing=True, method="Overlay", accumulated_ids=[]), labels
    ) == labels
    assert _live_overlay_render_labels(
        _widget(processing=True, method="Overlay", accumulated_ids=None), labels
    ) == labels


def test_no_scope_off_live_or_non_overlay():
    labels = tuple(range(1, 11))
    acc = [("scan", i) for i in range(1, 8)]
    # Not processing -> full selection (idle browse must resolve everything).
    assert _live_overlay_render_labels(
        _widget(processing=False, method="Overlay", accumulated_ids=acc), labels
    ) == labels
    # Sum/Average/Single keep their full-selection guards untouched.
    for method in ("Single", "Sum", "Average"):
        assert _live_overlay_render_labels(
            _widget(processing=True, method=method, accumulated_ids=acc), labels
        ) == labels
