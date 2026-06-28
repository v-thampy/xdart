"""P6.6 — the RSM equivalence spine (the live proxy), WITH corrections on.

The two-gridder Σraw/Σnorm refactor must keep the streaming and single-shot
paths identical and chunk-size-independent EVEN with a per-pixel norm — the
gates that the design notes were never compared on real intensities before.

Uses an angle-driven fake mapper: per-pixel q is a function of the per-frame
ANGLE value (correctly sliced per chunk), so it stays globally consistent across
chunkings, while the real ``xu.Gridder3D`` does the binning.  (The real-xu /
notebook golden is the separate, real-data-gated fixture.)
"""
from __future__ import annotations

import numpy as np
import pytest

from xrd_tools.core.geometry import DetectorHeader


def _header(Nch1=24, Nch2=28):
    return DetectorHeader(cch1=12.0, cch2=14.0, pwidth1=0.172, pwidth2=0.172,
                          distance=500.0, Nch1=Nch1, Nch2=Nch2)


class _AngleMapper:
    """q is a function of the per-frame angle (angles[0]) — so chunking the
    angles correctly chunks q (a frame's q is the same in any chunk layout)."""

    def __init__(self, header):
        self.header = header

    def pixel_q(self, angles, energy, *, UB=None, roi=None, image_shape=None):
        n, H, W = image_shape
        a0 = np.asarray(angles[0], dtype=float).reshape(n, 1, 1)
        y, x = np.mgrid[:H, :W]
        px = ((x - W / 2) / W)[None]
        py = ((y - H / 2) / H)[None]
        qx = np.broadcast_to(0.20 * px + 0.05 * a0, image_shape).astype(float)
        qy = np.broadcast_to(0.20 * py + 0.03 * a0, image_shape).astype(float)
        qz = np.broadcast_to(0.10 * (px + py) + 0.02 * a0, image_shape).astype(float)
        return (np.ascontiguousarray(qx), np.ascontiguousarray(qy),
                np.ascontiguousarray(qz))


def _setup(n_frames=8):
    h = _header()
    mapper = _AngleMapper(h)
    img = np.random.default_rng(0).random((n_frames, h.Nch1, h.Nch2)) + 1.0
    angles = [np.linspace(0.0, 1.4, n_frames)]
    # a strongly-varying weight (what a CorrectionStack supplies, exaggerated)
    weight = np.linspace(0.2, 1.0, h.Nch1 * h.Nch2).reshape(h.Nch1, h.Nch2)
    qx, qy, qz = mapper.pixel_q(angles, 1.0e4, image_shape=img.shape)
    bounds = ((float(qx.min()), float(qx.max())),
              (float(qy.min()), float(qy.max())),
              (float(qz.min()), float(qz.max())))
    return mapper, img, angles, weight, bounds


@pytest.mark.parametrize("weight_on", [False, True])
def test_streaming_equals_single_shot(weight_on):
    pytest.importorskip("xrayutilities")
    from xrd_tools.rsm.gridding import grid_img_data, grid_img_data_streaming
    mapper, img, angles, weight, bounds = _setup()
    w = weight if weight_on else None

    single = grid_img_data(mapper, img, angles, energy=1.0e4, bins=(8, 8, 8),
                           mask_static_pixels=False, weight=w)
    stream = grid_img_data_streaming(mapper, img, angles, energy=1.0e4,
                                     bins=(8, 8, 8), q_bounds=bounds,
                                     chunk_size=3, weight=w)
    np.testing.assert_allclose(single.intensity, stream.intensity, equal_nan=True)


@pytest.mark.parametrize("weight_on", [False, True])
def test_chunk_size_independent_intensity(weight_on):
    """Not just pixel counts (the old test) — the actual binned INTENSITY is
    identical across chunk sizes, with corrections on."""
    pytest.importorskip("xrayutilities")
    from xrd_tools.rsm.gridding import grid_img_data_streaming
    mapper, img, angles, weight, bounds = _setup()
    w = weight if weight_on else None

    def run(cs):
        return grid_img_data_streaming(
            mapper, img, angles, energy=1.0e4, bins=(8, 8, 8),
            q_bounds=bounds, chunk_size=cs, weight=w).intensity

    a, b, c = run(1), run(3), run(8)
    np.testing.assert_allclose(a, b, equal_nan=True)
    np.testing.assert_allclose(a, c, equal_nan=True)


def test_weight_changes_intensity_vs_count_mean():
    """Sanity that the weight is actually doing something (not a no-op that would
    make the equivalence tests vacuous)."""
    pytest.importorskip("xrayutilities")
    from xrd_tools.rsm.gridding import grid_img_data_streaming
    mapper, img, angles, weight, bounds = _setup()
    plain = grid_img_data_streaming(mapper, img, angles, energy=1.0e4,
                                    bins=(8, 8, 8), q_bounds=bounds,
                                    chunk_size=4).intensity
    wtd = grid_img_data_streaming(mapper, img, angles, energy=1.0e4,
                                  bins=(8, 8, 8), q_bounds=bounds,
                                  chunk_size=4, weight=weight).intensity
    both = np.isfinite(plain) & np.isfinite(wtd)
    assert both.any()
    assert not np.allclose(plain[both], wtd[both])
