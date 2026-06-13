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


def test_resolve_frame_mask_toggle_off_keeps_saturation_masks_invalids():
    """With the toggle OFF, the uint16 ceiling (65535) is NOT masked — a real
    saturated Bragg peak survives into the integration — while the UNAMBIGUOUS
    invalids (negatives, the uint32 ceiling) are still masked always."""
    img = np.full((100, 100), 500.0)
    img[10:20, :] = 65535.0          # would be a 10% sentinel band when ON
    img[0, 0] = -1.0                 # negative bad pixel (always invalid)
    img[0, 1] = 4294967295.0         # uint32 ceiling (always invalid)
    m = _resolve(img, mask_sentinel=False)
    assert not m[10:20, :].any()     # 65535 NOT masked when the toggle is off
    assert m[0, 0]                   # negatives still masked
    assert m[0, 1]                   # uint32 ceiling still masked
    assert not m[50, 50]             # real pixels untouched


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
