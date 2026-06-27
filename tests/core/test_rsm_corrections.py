"""P6.2 — the CorrectionStack weight hook for RSM.

The RSM grid folds a CorrectionStack into the Σ(raw·w)/Σ(w) accumulator as the
SAME per-pixel weight stitching uses.  Gates:
* the DetectorHeader → pyFAI ai bridge is correct (solid angle peaks at the beam
  centre — the convention check), and is wavelength-independent;
* the weight flows into the grid and changes intensities the pyFAI-reproducing
  way; ``corrections=None`` is the unchanged count-mean.
"""
from __future__ import annotations

import numpy as np
import pytest

from xrd_tools.core.geometry import DetectorHeader


def _header(Nch1=64, Nch2=80, cch1=30.0, cch2=40.0):
    return DetectorHeader(cch1=cch1, cch2=cch2, pwidth1=0.172, pwidth2=0.172,
                          distance=500.0, Nch1=Nch1, Nch2=Nch2)


class TestHeaderToAiBridge:
    def test_solid_angle_peaks_at_beam_centre(self):
        """The convention check: the max-solid-angle pixel is the beam centre
        (cch1, cch2) — proves poni1/poni2 aren't transposed/flipped."""
        pytest.importorskip("pyFAI")
        from xrd_tools.rsm.corrections import detector_header_to_ai
        h = _header(cch1=30.0, cch2=40.0)
        ai = detector_header_to_ai(h)
        sa = np.asarray(ai.solidAngleArray(shape=(h.Nch1, h.Nch2)))
        peak = np.unravel_index(int(np.argmax(sa)), sa.shape)
        assert abs(peak[0] - h.cch1) <= 1 and abs(peak[1] - h.cch2) <= 1
        assert 0.0 < sa.min() <= sa.max() == pytest.approx(1.0)

    def test_bridge_is_wavelength_independent(self):
        """The angular corrections don't depend on λ — so the weight can be
        computed once with a placeholder wavelength."""
        pytest.importorskip("pyFAI")
        from xrd_tools.corrections.stack import CorrectionStack
        from xrd_tools.rsm.corrections import detector_header_to_ai
        h = _header()
        stack = CorrectionStack(solid_angle=True, polarization_factor=0.95)
        a = detector_header_to_ai(h, wavelength_m=1.0e-10)
        b = detector_header_to_ai(h, wavelength_m=2.0e-10)
        wa = stack.normalization(a, (h.Nch1, h.Nch2))
        wb = stack.normalization(b, (h.Nch1, h.Nch2))
        np.testing.assert_allclose(wa, wb)


class TestRsmCorrectionWeight:
    def test_none_returns_none(self):
        from xrd_tools.rsm.corrections import rsm_correction_weight
        assert rsm_correction_weight(_header(), None) is None

    def test_weight_shape_and_roi(self):
        pytest.importorskip("pyFAI")
        from xrd_tools.corrections.stack import CorrectionStack
        from xrd_tools.rsm.corrections import rsm_correction_weight
        h = _header(Nch1=64, Nch2=80)
        stack = CorrectionStack(solid_angle=True)
        full = rsm_correction_weight(h, stack)
        assert full.shape == (64, 80)
        # ROI crops the weight so it matches the cropped chunk image
        cropped = rsm_correction_weight(h, stack, roi=(10, 40, 5, 45))
        assert cropped.shape == (30, 40)

    def test_solid_angle_weight_is_below_one_off_centre(self):
        pytest.importorskip("pyFAI")
        from xrd_tools.corrections.stack import CorrectionStack
        from xrd_tools.rsm.corrections import rsm_correction_weight
        w = rsm_correction_weight(_header(), CorrectionStack(solid_angle=True))
        assert w.max() == pytest.approx(1.0)     # at the beam centre
        assert w.min() < 1.0                     # falls off with angle


class _FakeMapper:
    """A PixelQMap stand-in that yields controlled, finite per-pixel q — isolates
    the corrections plumbing + the real xu.Gridder3D from xu's angle→q geometry
    (the real-xu end-to-end path is Step-6's vendored-fixture gate)."""

    def __init__(self, header):
        self.header = header

    def pixel_q(self, angles, energy, *, UB=None, roi=None, image_shape=None):
        n, H, W = image_shape
        y, x = np.mgrid[:H, :W]
        base = np.broadcast_to((x - W / 2) / W, image_shape).astype(float)
        base2 = np.broadcast_to((y - H / 2) / H, image_shape).astype(float)
        frame = np.arange(n, dtype=float).reshape(n, 1, 1)
        qx = 0.2 * base + 0.01 * frame
        qy = 0.2 * base2 + 0.01 * frame
        qz = 0.1 * (base + base2) + 0.005 * frame
        return (np.ascontiguousarray(qx), np.ascontiguousarray(qy),
                np.ascontiguousarray(qz))


class TestRsmGridCorrectionsWiring:
    """corrections flow end-to-end through the grid and change I."""

    def test_weight_reweights_the_grid_else_count_mean(self):
        """A per-pixel weight feeds the Σ(raw·w)/Σ(w) grid (so it differs from the
        count mean); weight=None is the plain count-mean. (The CorrectionStack
        magnitude is gated by TestRsmCorrectionWeight; this is the plumbing — a
        strongly-varying weight makes the effect visible.)"""
        pytest.importorskip("xrayutilities")
        from xrd_tools.rsm.gridding import grid_img_data

        h = _header(Nch1=32, Nch2=32)
        mapper = _FakeMapper(h)
        img = np.random.default_rng(0).random((6, 32, 32)) + 1.0
        angles = [np.linspace(0, 0.5, 6)]
        # a strong gradient weight — the kind a CorrectionStack supplies, but
        # large enough to move the weighted mean unmistakably.
        weight = np.linspace(0.1, 1.0, 32 * 32).reshape(32, 32)

        plain = grid_img_data(mapper, img, angles, energy=12000.0,
                              bins=(6, 6, 6), mask_static_pixels=False)
        corrected = grid_img_data(mapper, img, angles, energy=12000.0,
                                  bins=(6, 6, 6), mask_static_pixels=False,
                                  weight=weight)
        both = np.isfinite(plain.intensity) & np.isfinite(corrected.intensity)
        assert both.any()                                # same support
        assert not np.allclose(plain.intensity[both], corrected.intensity[both])
