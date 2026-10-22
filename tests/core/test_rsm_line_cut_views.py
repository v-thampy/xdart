"""FIT-TOOLS-2026-10 (F5/F6): the range-based ``extract_line_cut`` keeps its semantics and copies no volume.

The first row is parent-passing preservation; the second fails on the parent (a full projection copied the volume
through boolean fancy indexing and again inside ``np.nanmean``).
"""
from __future__ import annotations

import tracemalloc
import warnings

import numpy as np

from xrd_tools.rsm.volume import extract_line_cut


def test_extract_line_cut_keeps_its_nanmean_semantics():
    """Parent-passing preservation: the range-based helper equals nanmean over its inclusive box."""
    rng = np.random.default_rng(7)
    h, k, l = np.linspace(0, 1, 7), np.linspace(-1, 0, 8), np.linspace(2, 3, 9)
    intensity = rng.random((7, 8, 9))
    intensity[rng.random((7, 8, 9)) < 0.2] = np.nan
    intensity[3, 2, 4] = np.inf  # nanmean keeps infinities in its mean; so does this helper
    cases = (
        (0, {1: (-0.6, -0.2), 2: (2.25, 2.75)}),
        (1, None),
        (2, {0: (0.1, 0.5)}),
    )
    axes = (h, k, l)
    for axis, ranges in cases:
        masks = [np.ones(len(values), dtype=bool) for values in axes]
        for dim, (lower, upper) in (ranges or {}).items():
            masks[dim] = (axes[dim] >= lower) & (axes[dim] <= upper)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            reference = np.nanmean(np.moveaxis(intensity[np.ix_(*masks)], axis, -1), axis=(0, 1))
        values, profile = extract_line_cut(h, k, l, intensity, axis=axis, fixed_ranges=ranges)
        np.testing.assert_array_equal(values, axes[axis])
        np.testing.assert_allclose(profile, reference, rtol=1e-12, atol=0, equal_nan=True)


def test_full_projection_allocates_no_volume_copy():
    rng = np.random.default_rng(3)
    shape = (200, 200, 200)
    axes = tuple(np.linspace(0.0, 1.0, size) for size in shape)
    intensity = rng.random(shape)
    intensity[rng.random(shape) < 0.1] = np.nan
    tracemalloc.start()
    try:
        values, profile = extract_line_cut(*axes, intensity, axis=0)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert profile.shape == (200,)
    assert peak < intensity.nbytes / 4, (peak, intensity.nbytes)
