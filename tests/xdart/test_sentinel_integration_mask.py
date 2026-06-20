# -*- coding: utf-8 -*-
"""Regression: the uint16 (65535) detector sentinels must be excluded from the
INTEGRATION too (not just the raw display), or they integrate into a huge
spurious high-Q spike when no explicit mask file is applied — and the
thumbnail mask-union must tolerate mixed 1-D flat / 2-D boolean masks (the
collision the sentinel fix surfaced in the GI equivalence spine).
"""
from types import SimpleNamespace, MethodType

import numpy as np

from xdart.gui.tabs.static_scan.wranglers.wrangler_widget import wranglerThread
from xdart.modules.ewald.frame import _make_thumbnail


def _resolve(img, mask_sentinel=True):
    w = SimpleNamespace(mask_sentinel=mask_sentinel)
    w._resolve_frame_mask = MethodType(wranglerThread._resolve_frame_mask, w)
    # R3-A: _resolve_frame_mask emits a one-time advisory when the saturation
    # mask fires; bind it (no showLabel on this holder -> it just logs).
    w._warn_saturation_masked = MethodType(
        wranglerThread._warn_saturation_masked, w)
    scan = SimpleNamespace(_cached_data_mask=None)
    idx = w._resolve_frame_mask(scan, img)
    masked = np.zeros(np.asarray(img).size, dtype=bool)
    if idx is not None and len(idx):
        masked[idx] = True
    return masked.reshape(np.asarray(img).shape)


def test_resolve_frame_mask_excludes_uint16_sentinels():
    """With the 'Mask saturated' toggle ON (default), the uint16 ceiling band
    is masked from integration."""
    img = np.full((100, 100), 500.0)
    img[10:20, :] = 65535.0          # a 1000-px dead-module sentinel band (10%)
    img[0, 0] = -1.0                 # a negative bad pixel
    m = _resolve(img)
    assert m[10:20, :].all()         # the uint16 sentinel band is masked
    assert m[0, 0]                   # negatives still masked
    assert not m[50, 50]             # real pixels untouched


def test_resolve_frame_mask_toggle_off_masks_nothing():
    """'Mask Saturated' is the AUTHORITATIVE on/off (Vivek requirement): OFF masks
    NOTHING — a real saturated Bragg peak, negatives, AND the uint32 sentinel all
    survive into the integration.  Experiments with strong saturating peaks need
    the raw pixels kept; a toggle that masks regardless of state is pointless.
    (Was: OFF still masked the 'unambiguous' invalids — that design is rejected.)
    Independent of the Threshold toggle (intensity band), which this never touches.
    """
    img = np.full((100, 100), 500.0)
    img[10:20, :] = 65535.0          # uint16 ceiling band
    img[0, 0] = -1.0                 # negative
    img[0, 1] = 4294967295.0         # uint32 sentinel
    m = _resolve(img, mask_sentinel=False)
    assert not m.any()               # NOTHING masked when the toggle is off


def test_resolve_frame_mask_ceiling_from_dtype():
    """The integration ceiling is derived from the raw integer dtype: an 8-bit
    frame's dead band sits at 255, not 65535, and is masked (toggle ON)."""
    img = np.full((100, 100), 50, dtype=np.uint8)
    img[10:20, :] = 255              # 10% at the uint8 ceiling
    m = _resolve(img)
    assert m[10:20, :].all()         # uint8 ceiling masked via dtype derivation
    assert not m[50, 50]


def test_resolve_frame_mask_keeps_sparse_saturation():
    """A handful of legitimately-saturated pixels (< 1e-4 of the frame) are NOT
    treated as sentinels — the same fraction-guard the display uses."""
    img = np.full((1000, 1000), 500.0)   # 1e6 px
    img[0, :5] = 65535.0                  # 5 px = 5e-6 fraction (< 1e-4)
    m = _resolve(img)
    assert not m[0, :5].any()             # sparse saturation preserved


def test_resolve_frame_mask_warns_once_when_saturation_fires(caplog):
    """R3-A: a fired saturation mask must not be silent at runtime — it logs a
    one-time advisory (the mask is computed once per scan, cached thereafter),
    and stays quiet when nothing is masked or the toggle is off."""
    import logging
    img = np.full((100, 100), 100, dtype=np.uint16)
    img[:30, :] = 65535                  # 30% dead block -> mask fires

    w = SimpleNamespace(mask_sentinel=True)
    w._resolve_frame_mask = MethodType(wranglerThread._resolve_frame_mask, w)
    w._warn_saturation_masked = MethodType(
        wranglerThread._warn_saturation_masked, w)
    scan = SimpleNamespace(_cached_data_mask=None)
    with caplog.at_level(logging.WARNING):
        w._resolve_frame_mask(scan, img)
        w._resolve_frame_mask(scan, img)   # cache hit -> no recompute, no 2nd warn
    fired = [m for m in caplog.messages if "Mask Saturated" in m]
    assert len(fired) == 1, f"expected one advisory, got {fired}"

    # toggle OFF -> never warns (the saturated block is left in the integration).
    w2 = SimpleNamespace(mask_sentinel=False)
    w2._resolve_frame_mask = MethodType(wranglerThread._resolve_frame_mask, w2)
    w2._warn_saturation_masked = MethodType(
        wranglerThread._warn_saturation_masked, w2)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        w2._resolve_frame_mask(SimpleNamespace(_cached_data_mask=None), img)
    assert not [m for m in caplog.messages if "Mask Saturated" in m]


def test_make_thumbnail_handles_mixed_1d_flat_and_2d_boolean_masks():
    """The mask-union normalizes a 2-D boolean image-mask to flat indices, so a
    flat per-frame mask + a 2-D global mask no longer collide on concatenate."""
    img = np.full((50, 50), 100.0)
    mask_idx = np.array([0, 1, 2])                # 1-D flat indices
    global2d = np.zeros((50, 50), dtype=bool)
    global2d[5, :] = True                          # 2-D boolean image mask
    t = _make_thumbnail(img, mask_idx=mask_idx, global_mask=global2d, max_size=200)
    assert t is not None
    assert np.isnan(t.ravel()[[0, 1, 2]]).all()   # the flat mask applied
    assert np.isnan(t[5, :]).all()                 # the 2-D mask applied
    assert not np.isnan(t[25, 25])                 # untouched pixel finite
