"""FIT-TOOLS-2026-10 oracle row 8 (F6): axis-aligned RSM line-outs through a point.

Bandwidth is the notebook's FULL width: a voxel joins a line-out when ``abs(coordinate - centre) <= width/2`` on both
transverse axes (inclusive). The value is the unweighted mean of the box's finite voxels; a bin with none is
unavailable (NaN, count 0), never zero; an infinite voxel inside a box is refused.
"""
from __future__ import annotations

import tracemalloc
import warnings

import numpy as np
import pytest

from xrd_tools.rsm.coordinate_frame import RSMCoordinateFrame
from xrd_tools.rsm.lineout import RSMLineoutRefused, axis_lineouts
from xrd_tools.rsm.volume import RSMVolume


def compute_lineouts(H_m, K_m, L_m, rmap, H0, K0, L0, bandwidth):
    """Integrate a merged RSM volume to produce H, K, L lineouts.
    Returns (L_lo, K_lo, H_lo)."""
    n_m = len(H_m)
    H_mask = np.abs(H_m - H0) <= bandwidth / 2
    K_mask = np.abs(K_m - K0) <= bandwidth / 2
    L_mask = np.abs(L_m - L0) <= bandwidth / 2
    L_lo = np.nanmean(rmap[np.ix_(H_mask, K_mask, np.ones(n_m, dtype=bool))], axis=(0, 1))
    K_lo = np.nanmean(rmap[np.ix_(H_mask, np.ones(n_m, dtype=bool), L_mask)], axis=(0, 2))
    H_lo = np.nanmean(rmap[np.ix_(np.ones(n_m, dtype=bool), K_mask, L_mask)], axis=(1, 2))
    return L_lo, K_lo, H_lo
# ^ copied verbatim from example_notebooks/RSM/RSM_process.ipynb (read-only); rmap is indexed (H, K, L).


def _hand_volume() -> RSMVolume:
    """A 3 x 4 x 5 volume, unequal spacings, I[i, j, m] = 100 i + 10 j + m + 1 with a few empty voxels."""
    h = np.array([0.0, 0.5, 1.0])
    k = np.array([-1.0, -0.5, 0.0, 0.5])
    l = np.array([2.0, 2.25, 2.5, 2.75, 3.0])
    i, j, m = np.meshgrid(np.arange(3), np.arange(4), np.arange(5), indexing="ij")
    intensity = (100 * i + 10 * j + m + 1).astype(float)
    intensity[0, 0, 1] = np.nan           # value 2, inside every box
    intensity[:, 0:3, 4] = np.nan         # the l = 3.0 bin of the l line-out has no finite voxel
    intensity[1, 3, 0] = np.inf           # outside every box (k = 0.5 lies beyond k = -0.5 +- 0.5)
    return RSMVolume(h, k, l, intensity)


def test_hand_calculated_anisotropic_lineouts_inclusive_edges_and_empty_bins():
    volume = _hand_volume()
    # transverse boxes, edges included exactly: h in {0, .5, 1}, k in {-1, -.5, 0}, l in {2.25, 2.5, 2.75}
    lineouts = axis_lineouts(volume, {"h": 0.5, "k": -0.5, "l": 2.5}, {"h": 1.0, "k": 1.0, "l": 0.5})
    assert tuple(lineouts) == ("h", "k", "l")
    expected = {
        # h: mean over j in 0..2, m in 1..3 of 100 i + 10 j + m + 1; i = 0 loses its value 2: (117 - 2) / 8
        "h": ([115 / 8, 113.0, 213.0], [8, 9, 9]),
        # k: mean over i in 0..2, m in 1..3; j = 0 loses its 2: (927 - 2) / 8
        "k": ([925 / 8, 113.0, 123.0, 133.0], [8, 9, 9, 9]),
        # l: mean over i in 0..2, j in 0..2; m = 1 loses its 2: (1008 - 2) / 8; m = 4 has no finite voxel
        "l": ([111.0, 1006 / 8, 113.0, 114.0, np.nan], [9, 8, 9, 9, 0]),
    }
    for name, (values, counts) in expected.items():
        lineout = lineouts[name]
        assert lineout.axis == name and lineout.unit is None
        np.testing.assert_array_equal(lineout.coordinates, volume.axis_values["hkl".index(name)])
        np.testing.assert_allclose(lineout.intensity, values, rtol=1e-12, atol=0, equal_nan=True)
        np.testing.assert_array_equal(lineout.counts, counts)
        np.testing.assert_array_equal(lineout.available, np.asarray(counts) > 0)
    assert dict(lineouts["h"].box) == {"k": (-1.0, 0.0), "l": (2.25, 2.75)}
    only_k = axis_lineouts(volume, (0.5, -0.5, 2.75), 1.0, axes=("k",))
    assert tuple(only_k) == ("k",)
    # one full width 1.0 everywhere: h keeps 0..1, l keeps 2.25..3.0 (2.25 on the edge; 2.0 with its +inf is out)
    assert dict(only_k["k"].box) == {"h": (0.0, 1.0), "l": (2.25, 3.25)}
    assert dict(only_k["k"].box_bins) == {"h": 3, "l": 4}
    # m = 4 is empty for j in 0..2, so those bins keep m in 1..3; j = 3 averages m in 1..4: 100 + 30 + 2.5 + 1
    np.testing.assert_allclose(only_k["k"].intensity, [925 / 8, 113.0, 123.0, 133.5], rtol=1e-12, atol=0)
    np.testing.assert_array_equal(only_k["k"].counts, [8, 9, 9, 12])


def test_named_axes_and_units_follow_the_coordinate_frame():
    q = np.linspace(-1.0, 1.0, 6)
    volume = RSMVolume.from_axes(
        RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU,
        (("qx", q), ("qy", q[:5]), ("qz", q[:4] + 3.0)),
        np.arange(120.0).reshape(6, 5, 4),
    )
    lineouts = axis_lineouts(volume, {"qx": 0.0, "qy": -0.2, "qz": 2.6}, 0.8)
    assert {name: lineout.unit for name, lineout in lineouts.items()} == {
        "qx": "q_A^-1", "qy": "q_A^-1", "qz": "q_A^-1"}
    assert lineouts["qz"].intensity.shape == (4,)
    with pytest.raises(ValueError, match="axis"):
        axis_lineouts(volume, {"qx": 0.0, "qy": 0.0, "qz": 2.6}, 0.8, axes=("h",))
    with pytest.raises(ValueError, match="centre"):
        axis_lineouts(volume, {"qx": 0.0, "qy": 0.0}, 0.8)
    with pytest.raises(ValueError, match="width"):
        axis_lineouts(volume, {"qx": 0.0, "qy": 0.0, "qz": 2.6}, -0.1)
    with pytest.raises(ValueError, match="centre"):
        axis_lineouts(volume, {"qx": np.nan, "qy": 0.0, "qz": 2.6}, 0.8)


def test_infinities_inside_a_box_are_refused():
    volume = _hand_volume()
    # the +inf voxel at (h=.5, k=.5, l=2.0) sits inside the k line-out box through l = 2.0
    with pytest.raises(RSMLineoutRefused) as refused:
        axis_lineouts(volume, {"h": 0.5, "k": -0.5, "l": 2.0}, 1.0, axes=("k",))
    assert refused.value.code == "INFINITE_VOXEL"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_notebook_equality_on_square_volumes(seed):
    rng = np.random.default_rng(seed)
    n = 12
    H, K, L = np.linspace(0.8, 1.2, n), np.linspace(-0.3, 0.1, n), np.linspace(2.5, 3.5, n)
    rmap = rng.random((n, n, n)) * 1e4
    rmap[rng.random((n, n, n)) < 0.3] = np.nan
    rmap[rng.random((n, n, n)) < 0.05] = 0.0
    H0, K0, L0 = H[5] + 0.004, K[6] - 0.01, L[4]
    bandwidth = 0.11
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # the notebook's nanmean on empty bins
        L_lo, K_lo, H_lo = compute_lineouts(H, K, L, rmap, H0, K0, L0, bandwidth)
    ours = axis_lineouts(RSMVolume(H, K, L, rmap), (H0, K0, L0), bandwidth)
    for name, reference in (("h", H_lo), ("k", K_lo), ("l", L_lo)):
        np.testing.assert_array_equal(np.isnan(ours[name].intensity), np.isnan(reference))
        np.testing.assert_allclose(ours[name].intensity, reference, rtol=1e-12, atol=0, equal_nan=True)


def test_a_wide_box_allocates_no_volume_copy():
    rng = np.random.default_rng(3)
    shape = (200, 200, 200)
    axes = tuple(np.linspace(0.0, 1.0, size) for size in shape)
    intensity = rng.random(shape)
    intensity[rng.random(shape) < 0.1] = np.nan
    volume = RSMVolume(*axes, intensity)
    tracemalloc.start()
    try:
        axis_lineouts(volume, (0.5, 0.5, 0.5), 2.0, axes=("h",))  # a box covering the whole cross-section
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < intensity.nbytes / 4, (peak, intensity.nbytes)
