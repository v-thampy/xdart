# -*- coding: utf-8 -*-
"""V1 RowMeta capture, production-wired (canonical-grid plan §Stage-1/3).

Drives the REAL adapter → ``append_row`` → ``accumulate_waterfall`` path via
:mod:`tests.xdart.ov_harness` and asserts the per-row transform provenance
(``WaterfallHistory.row_meta``) is captured on a real append: the
acquisition-native ``source_unit``, the row's wavelength, and the norm
channel + monitor value.  Since Stage 3 the rows themselves are stored
acquisition-NATIVE and the provenance is capture-only (norm/conversion run
at draw), so nothing here asserts display output — only that the carried
provenance is present and id-aligned (what the draw-time render consumes).
"""

from __future__ import annotations

import pytest

from tests.xdart.ov_harness import OVHarness


def test_row_meta_captured_on_real_append():
    h = OVHarness()
    h.publish(0)
    hist = h.history
    assert len(hist.row_meta) == hist.count == 1
    meta = hist.row_meta[0]
    assert meta is not None
    assert meta.source_unit == "q_A^-1"
    assert meta.wavelength_m == pytest.approx(1e-10)
    # No norm channel selected: channel/value record "none would apply".
    assert meta.norm_channel is None and meta.norm_value is None
    assert meta.bkg_token is None and meta.projection_id is None

    # A REAL channel change is a re-render since Stage 4 (S-16 dissolved):
    # frame 0's accumulated row is PRESERVED, so its RowMeta keeps the
    # provenance captured at ITS append (no channel yet); frame 1, appended
    # after the change, records the new channel + its own monitor reading
    # from metadata_raw (PROVENANCE only — the stored rows stay un-normed;
    # the draw-time norm reads the CURRENT channel + per-row `metadata`).
    h.norm_change(real=True, channel="i0")
    h.publish(1)
    hist = h.history
    assert hist.count == 2 and len(hist.row_meta) == 2
    assert [m.norm_channel for m in hist.row_meta] == [None, "i0"]
    assert [m.norm_value for m in hist.row_meta] == [None, 2.0]


def test_row_meta_source_unit_stays_native_across_unit_flip():
    h = OVHarness()
    h.publish(0)
    h.unit_toggle()                       # display Q→2θ: a draw-time relabel
    h.publish(1)                          # arrives + is STORED native (Stage 3)
    hist = h.history
    assert hist.count == 2 and len(hist.row_meta) == 2
    # source_unit names the acquisition-native unit; since Stage 3 the
    # history itself is native too — the display flip never touches storage
    # (history.unit == the native token regardless of the combo).
    assert [m.source_unit for m in hist.row_meta] == ["q_A^-1", "q_A^-1"]
    assert hist.unit == "q_A^-1"


def test_row_meta_projection_id_rides_pinned_cut():
    h = OVHarness(slice_mode=True)
    h.publish(0, select="only")
    h.move_live_cut(-10.0, 2.0)
    h.pin_current_cut()
    hist = h.history
    assert hist.count == 1 and len(hist.row_meta) == 1
    meta = hist.row_meta[0]
    assert meta is not None
    recipe = h.widget._pinned_slice_cuts[hist.ids[0]]
    # Duplicated from the pin recipe (self-description; also lives in the id).
    assert meta.projection_id is not None
    assert meta.projection_id == recipe["projection_id"]
    assert meta.source_unit == "q_A^-1"   # the native 2D radial unit
