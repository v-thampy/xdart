"""P6.5 — the xu_hist q-provider (xu_q_frames) for the stitch histogram merge.

Dead-but-proven so the deferred xu_hist stitch backend (P3c) is pure wiring.
|q| (the vector magnitude) is convention-free and gate-checked here; χ (the
azimuth) is flagged PENDING the P3c real-data gate (== pyFAI chiArray), so it is
not asserted against an absolute reference.
"""
from __future__ import annotations

import numpy as np
import pytest

from xrd_tools.core.geometry import DetectorHeader


def _header(Nch1=24, Nch2=28):
    return DetectorHeader(cch1=12.0, cch2=14.0, pwidth1=0.172, pwidth2=0.172,
                          distance=500.0, Nch1=Nch1, Nch2=Nch2)


class _AngleMapper:
    """Controlled finite per-pixel q (isolates the provider from xu geometry —
    the real-xu path is the P3c notebook gate)."""

    def __init__(self, header):
        self.header = header

    def pixel_q(self, angles, energy, *, UB=None, roi=None, image_shape=None):
        n, H, W = image_shape
        a0 = np.asarray(angles[0], dtype=float).reshape(n, 1, 1)
        y, x = np.mgrid[:H, :W]
        px = ((x - W / 2) / W)[None]
        py = ((y - H / 2) / H)[None]
        qx = np.broadcast_to(0.20 * px + 0.05 * a0, image_shape).astype(float)
        qy = np.broadcast_to(0.20 * py + 0.03 * a0 + 0.5, image_shape).astype(float)
        qz = np.broadcast_to(0.10 * (px + py) + 0.02 * a0, image_shape).astype(float)
        return (np.ascontiguousarray(qx), np.ascontiguousarray(qy),
                np.ascontiguousarray(qz))


def _setup(n=3):
    h = _header()
    mapper = _AngleMapper(h)
    imgs = [np.random.default_rng(i).random((h.Nch1, h.Nch2)) + 1.0 for i in range(n)]
    angles = [np.linspace(0.0, 0.6, n)]
    return h, mapper, imgs, angles


def test_qmag_is_vector_magnitude_of_pixel_q():
    from xrd_tools.integrate.stitch_hist import xu_q_frames
    h, mapper, imgs, angles = _setup()
    qx, qy, qz = mapper.pixel_q(angles, 1.0e4, image_shape=(len(imgs), h.Nch1, h.Nch2))
    expected = np.sqrt(qx ** 2 + qy ** 2 + qz ** 2)
    for i, (qmag, _chi, _sig, _w) in enumerate(
            xu_q_frames(imgs, mapper, angles, 1.0e4)):
        np.testing.assert_allclose(qmag, expected[i])


def test_signal_monitor_and_mask():
    from xrd_tools.integrate.stitch_hist import xu_q_frames
    h, mapper, imgs, angles = _setup()
    mask = np.zeros((h.Nch1, h.Nch2), dtype=bool)
    mask[:5, :] = True
    weight = np.full((h.Nch1, h.Nch2), 2.0)
    frames = list(xu_q_frames(imgs, mapper, angles, 1.0e4, weight=weight,
                              mask=mask, normalization=[4.0, 4.0, 4.0]))
    _q, _c, sig, w = frames[0]
    np.testing.assert_allclose(sig, imgs[0] / 4.0)     # monitor divides
    assert np.all(w[:5, :] == 0.0)                     # masked → no weight
    assert np.all(w[5:, :] == 2.0)


def test_bad_monitor_raises():
    from xrd_tools.integrate.stitch_hist import xu_q_frames
    h, mapper, imgs, angles = _setup()
    with pytest.raises(ValueError, match="invalid value"):
        next(xu_q_frames(imgs, mapper, angles, 1.0e4, normalization=[1.0, 0.0, 1.0]))


def test_feeds_stitch_q_grid_to_a_sane_1d_pattern():
    """Structural gate: the (|q|, χ, signal, weight) tuples are consumable by
    stitch_q_grid and produce a finite 1D pattern (so P3c is pure wiring)."""
    from xrd_tools.integrate.stitch_hist import stitch_q_grid, xu_q_frames
    h, mapper, imgs, angles = _setup()
    out = stitch_q_grid(xu_q_frames(imgs, mapper, angles, 1.0e4),
                        mode="1d", npt=64)
    assert out.radial.shape == (64,)
    assert np.isfinite(out.intensity).any()
    assert np.nanmax(out.intensity) > 0
