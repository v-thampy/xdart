"""Core detector invalid-pixel policy (R3-C)."""

import numpy as np
import pytest

from xrd_tools.core.invalid import (
    UINT32_CEILING,
    integer_saturation_ceiling,
    saturation_pixels,
)


def test_native_ceiling_is_resolved_once_independent_of_frame_values(monkeypatch):
    from xrd_tools.core.invalid import _integer_dtype_saturation_ceiling

    _integer_dtype_saturation_ceiling.cache_clear()
    original = np.iinfo
    calls = []

    def observed(dtype):
        calls.append(dtype)
        return original(dtype)

    monkeypatch.setattr(np, "iinfo", observed)
    for value in (0, 123, 65535, 1000):
        assert integer_saturation_ceiling(np.full((2, 2), value, dtype=np.uint16)) == 65535
    assert calls == [np.dtype("uint16")]


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


@pytest.mark.parametrize("count", (1, 2, 50))
@pytest.mark.parametrize("dtype", (np.uint8, np.uint16, np.uint32))
def test_every_saturated_pixel_is_selected(count, dtype):
    a = np.zeros(100000, dtype=dtype)
    a[:count] = np.iinfo(dtype).max
    np.testing.assert_array_equal(
        np.flatnonzero(saturation_pixels(a, ceiling=np.iinfo(dtype).max)),
        np.arange(count),
    )


def test_saturation_pixels_unknown_ceiling_empty_and_nonfinite():
    assert not saturation_pixels(np.ones(10), ceiling=None).any()
    assert not saturation_pixels(np.array([]), ceiling=65535.0).any()
    a = np.array([np.nan, np.inf, 65535.0, 65535.0])
    np.testing.assert_array_equal(saturation_pixels(a, ceiling=65535.0),
                                  [False, False, True, True])
