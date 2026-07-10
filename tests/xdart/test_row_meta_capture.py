# -*- coding: utf-8 -*-
"""V1 Stage-1 RowMeta capture, production-wired (canonical-grid plan §Stage-1).

Drives the REAL adapter → ``append_row`` → ``accumulate_waterfall`` path via
:mod:`tests.xdart.ov_harness` and asserts the per-row transform provenance
(``WaterfallHistory.row_meta``) is captured on a real append: the
acquisition-native ``source_unit`` (read BEFORE the plotUnit conversion), the
row's wavelength, and the norm channel + monitor value actually applied.
Stage 1 is dual-write: rows are still stored post-transform, so nothing here
asserts display output — only that the carried provenance is present and
id-aligned (what the Stage-3 flip will consume).
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
    # No norm channel selected: channel/value record "none applied".
    assert meta.norm_channel is None and meta.norm_value is None
    assert meta.bkg_token is None and meta.projection_id is None
    assert meta.native_x is None and meta.native_y is None   # Stage-2 slots

    # A REAL channel change (S-16 rebuild) re-captures under the new channel;
    # the value is the row's own monitor reading from metadata_raw.
    h.norm_change(real=True, channel="i0")
    h.publish(1)
    hist = h.history
    assert hist.count == 2 and len(hist.row_meta) == 2
    assert [m.norm_channel for m in hist.row_meta] == ["i0", "i0"]
    assert [m.norm_value for m in hist.row_meta] == [2.0, 2.0]


def test_row_meta_source_unit_stays_native_across_unit_flip():
    h = OVHarness()
    h.publish(0)
    h.unit_toggle()                       # display Q→2θ: a relabel
    h.publish(1)                          # this row arrives converted to 2θ
    hist = h.history
    assert hist.count == 2 and len(hist.row_meta) == 2
    # source_unit is captured BEFORE _apply_plot_unit_1d, so it names the
    # acquisition-native unit even when the display (history.unit) does not.
    assert [m.source_unit for m in hist.row_meta] == ["q_A^-1", "q_A^-1"]
    assert hist.unit != "q_A^-1"


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
