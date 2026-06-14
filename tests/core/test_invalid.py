"""Core detector invalid-pixel policy (R3-C)."""

import numpy as np
import pytest

from xrd_tools.core.invalid import (
    UINT32_CEILING,
    integer_saturation_ceiling,
    saturation_pixels,
)


@pytest.mark.parametrize("dtype, expected", [
    (np.uint16, 65535.0),
    (np.uint8, 255.0),
    (np.uint32, 4294967295.0),
    (np.int32, float(np.iinfo(np.int32).max)),
])
def test_ceiling_is_dtype_derived(dtype, expected):
    assert integer_saturation_ceiling(np.zeros((3, 3), dtype=dtype)) == expected


def test_ceiling_none_for_float_never_hardcodes_65535():
    # Core must NOT assume 16-bit / hardcode 65535: a float frame (integer dtype
    # lost upstream) returns None so the caller chooses any fallback.
    assert integer_saturation_ceiling(np.zeros((3, 3), dtype=float)) is None


def test_saturation_pixels_masks_a_dead_module():
    # A whole block at the ceiling (fraction > 1e-4) is a dead/overflowed module.
    a = np.zeros(10000, dtype=float)
    a[:50] = 65535.0                      # 0.5% >> 1e-4
    mask = saturation_pixels(a, ceiling=65535.0)
    assert mask.shape == a.shape
    assert mask.sum() == 50
    assert np.array_equal(np.flatnonzero(mask), np.arange(50))


def test_saturation_pixels_keeps_sparse_saturation():
    # A handful of genuinely-saturated Bragg pixels (fraction <= 1e-4) stay.
    a = np.zeros(100000, dtype=float)
    a[:3] = 65535.0                       # 3e-5 < 1e-4
    assert not saturation_pixels(a, ceiling=65535.0).any()


def test_saturation_pixels_no_ceiling_or_uint32_is_all_false():
    a = np.full(1000, 65535.0)
    assert not saturation_pixels(a, ceiling=None).any()
    # the uint32 dead sentinel is unambiguous — masked ALWAYS by the caller,
    # not through this opt-in fraction gate.
    b = np.full(1000, UINT32_CEILING)
    assert not saturation_pixels(b, ceiling=UINT32_CEILING).any()


def test_saturation_pixels_empty_and_nonfinite():
    assert not saturation_pixels(np.array([]), ceiling=65535.0).any()
    # equality with a finite ceiling already excludes NaN/inf.
    a = np.array([np.nan, np.inf, 65535.0, 65535.0])
    mask = saturation_pixels(a, ceiling=65535.0, min_fraction=0.0)
    assert mask.tolist() == [False, False, True, True]


def test_min_fraction_boundary_is_strict_greater_than():
    # exactly 1e-4 does NOT trip (matches the production '> 1e-4' guard).
    a = np.zeros(10000, dtype=float)
    a[:1] = 65535.0                       # 1/10000 == 1e-4 exactly
    assert not saturation_pixels(a, ceiling=65535.0).any()
    a[:2] = 65535.0                       # 2/10000 > 1e-4
    assert saturation_pixels(a, ceiling=65535.0).sum() == 2
