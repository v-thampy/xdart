# -*- coding: utf-8 -*-
"""Regression for image_widget._ceiling_safe_levels — the robust display
level-clamp that keeps an UNMASKED detector ceiling (the 'Mask saturated'
toggle OFF) from blowing out autoscale, WITHOUT ever hiding the pixel.

Two traps the design must avoid (both covered here):
  * Log/Sqrt transform the displayed image IN PLACE before the percentile, so
    the ceiling mask must be taken from the UNTRANSFORMED raw, not displayed.
  * An all-/mostly-ceiling frame leaves an empty non-ceiling population, which
    would yield NaN levels + the 'autoscale to data min/max' regression — the
    helper must fall back to the full finite population.
"""
import numpy as np

from xdart.gui.widgets.image_widget import _ceiling_safe_levels


def test_ceiling_excluded_from_population():
    """A 65535 block (toggle OFF -> still finite) must not drag the upper level
    up to the ceiling; the level is driven by the non-ceiling pixels."""
    raw = np.full((100, 100), 100.0)
    raw[:28, :] = 65535.0          # 28% at the uint16 ceiling
    lo, hi = _ceiling_safe_levels(raw, raw, (2, 98))
    assert hi < 1000.0             # driven by the ~100 background, not 65535
    assert np.isfinite(lo) and np.isfinite(hi)
    # Without exclusion the 98th percentile would BE the ceiling:
    assert np.nanpercentile(raw, 98) == 65535.0


def test_all_ceiling_frame_falls_back_no_nan():
    """Every finite pixel at the ceiling -> fall back to the full population so
    levels are finite (no NaN, no min/max-autoscale regression)."""
    raw = np.full((10, 10), 65535.0)
    lo, hi = _ceiling_safe_levels(raw, raw, (2, 98))
    assert np.isfinite(lo) and np.isfinite(hi)


def test_ceiling_mask_uses_pretransform_raw_not_displayed():
    """For Log/Sqrt the caller passes a transformed `displayed`; the ceiling
    must still be identified from the original `raw` counts."""
    raw = np.full((100, 100), 100.0)
    raw[:28, :] = 65535.0
    displayed = np.log10(raw)              # transformed: ceiling -> ~4.816
    lo, hi = _ceiling_safe_levels(displayed, raw, (2, 98))
    # exclusion keyed on raw==65535 -> upper level is log10(100)=2.0, NOT 4.816
    assert hi < 3.0


def test_already_masked_ceiling_is_noop():
    """When saturation was masked upstream (toggle ON -> NaN in raw), the
    ceiling test is a no-op and levels come from the finite background."""
    raw = np.full((100, 100), 100.0)
    raw[:28, :] = np.nan                   # masked upstream
    lo, hi = _ceiling_safe_levels(raw, raw, (2, 98))
    assert hi == 100.0 and lo == 100.0


def test_all_nonfinite_returns_unit_range():
    out = _ceiling_safe_levels(np.full((4, 4), np.nan), np.full((4, 4), np.nan),
                               (2, 98))
    assert out == (0.0, 1.0)


def test_ceiling_derived_from_integer_dtype_not_hardcoded():
    """R3-D: the saturation ceiling is taken from the raw INTEGER dtype, not a
    hardcoded 65535.  A uint8 frame's 255 ceiling must be excluded from the
    autoscale population (the old hardcoded ==65535 missed it and let the 255
    block blow out the scale)."""
    raw8 = np.full((100, 100), 10, dtype=np.uint8)
    raw8[:28, :] = 255                      # 28% at the uint8 ceiling
    _, hi = _ceiling_safe_levels(raw8.astype(float), raw8, (2, 98))
    assert hi < 200.0                       # driven by the ~10 background, not 255
    # uint16 integer dtype still derives 65535 (same value, now dtype-driven).
    raw16 = np.full((100, 100), 100, dtype=np.uint16)
    raw16[:28, :] = 65535
    _, hi16 = _ceiling_safe_levels(raw16.astype(float), raw16, (2, 98))
    assert hi16 < 1000.0


def test_float_raw_falls_back_to_65535_ceiling():
    """When the integer dtype was lost upstream (float raw), the legacy 65535
    GUI fallback still excludes a 65535 block — behaviour preserved."""
    raw = np.full((100, 100), 100.0)        # float — dtype-derived ceiling is None
    raw[:28, :] = 65535.0
    _, hi = _ceiling_safe_levels(raw, raw, (2, 98))
    assert hi < 1000.0
