"""P6.2 — the CorrectionStack weight hook for RSM.

The RSM grid folds a CorrectionStack into the Σraw/Σnorm accumulator as the
SAME per-pixel norm stitching uses.  Gates:
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


class TestGiGridWeight:
    """P6.8 — the GI intensity weight for the RSM grid (footprint·Fresnel·
    absorption), reusing P4's pyFAI-fiber αf. Refraction + per-frame αi + the
    absolute GI signs are the real-data-gated tail (not asserted here)."""

    def test_off_factors_reduce_to_unit_weight(self):
        pytest.importorskip("xrayutilities")
        pytest.importorskip("pyFAI")
        from xrd_tools.corrections.grazing import GICorrectionStack
        from xrd_tools.rsm.corrections import gi_grid_weight
        gi = GICorrectionStack(material="Si", energy_eV=10000.0, footprint=False,
                               fresnel=False, absorption=False, refraction=False)
        w = gi_grid_weight(_header(), gi, incident_angle_deg=0.3)
        np.testing.assert_allclose(w, 1.0)

    def test_footprint_only_is_inv_sin_alpha_i(self):
        pytest.importorskip("xrayutilities")
        pytest.importorskip("pyFAI")
        from xrd_tools.corrections.grazing import GICorrectionStack
        from xrd_tools.rsm.corrections import gi_grid_weight
        gi = GICorrectionStack(material="Si", energy_eV=10000.0, footprint=True,
                               fresnel=False, absorption=False, refraction=False)
        w = gi_grid_weight(_header(), gi, incident_angle_deg=0.3)
        # footprint boost 1/sin αi enters the Σnorm weight
        np.testing.assert_allclose(w, 1.0 / np.sin(np.deg2rad(0.3)))

    def test_rsm_correction_weight_multiplies_gi_in(self):
        """rsm_correction_weight(corrections, gi) == base × GI weight."""
        pytest.importorskip("xrayutilities")
        pytest.importorskip("pyFAI")
        from xrd_tools.corrections.grazing import GICorrectionStack, GISettings
        from xrd_tools.corrections.stack import CorrectionStack
        from xrd_tools.rsm.corrections import (
            gi_grid_weight, rsm_correction_weight,
        )
        h = _header()
        cs = CorrectionStack(solid_angle=True)
        gi = GICorrectionStack(material="Si", energy_eV=10000.0, footprint=True,
                               fresnel=True, absorption=False, refraction=False)
        base = rsm_correction_weight(h, cs)
        giw = gi_grid_weight(h, gi, incident_angle_deg=0.3)
        combined = rsm_correction_weight(
            h, cs, gi=GISettings(corrections=gi, incident_angle_deg=0.3))
        np.testing.assert_allclose(combined, base * giw)

    def test_gi_requires_fixed_incident_angle(self):
        pytest.importorskip("xrayutilities")
        pytest.importorskip("pyFAI")
        from xrd_tools.corrections.grazing import GICorrectionStack, GISettings
        from xrd_tools.rsm.corrections import rsm_correction_weight
        gi = GICorrectionStack(material="Si", energy_eV=10000.0)
        with pytest.raises(ValueError, match="incident_angle_deg"):
            rsm_correction_weight(_header(), None, gi=GISettings(corrections=gi))

    def test_gi_refraction_rejected_in_rsm(self):
        """RSM applies only the GI *intensity* weight, not the q-coordinate
        refraction the stitch GI provider does — so a refraction=True stack would
        silently diverge from GI stitch.  Reject it (default is refraction=True);
        refraction=False is accepted."""
        pytest.importorskip("xrayutilities")
        pytest.importorskip("pyFAI")
        from xrd_tools.corrections.grazing import GICorrectionStack, GISettings
        from xrd_tools.rsm.corrections import rsm_correction_weight
        on = GICorrectionStack(material="Si", energy_eV=10000.0)   # refraction=True default
        with pytest.raises(NotImplementedError, match="refraction"):
            rsm_correction_weight(
                _header(), None,
                gi=GISettings(corrections=on, incident_angle_deg=0.3))
        off = GICorrectionStack(material="Si", energy_eV=10000.0, refraction=False)
        w = rsm_correction_weight(
            _header(), None, gi=GISettings(corrections=off, incident_angle_deg=0.3))
        assert w is not None and np.all(np.isfinite(w))


class TestRsmGridCorrectionsWiring:
    """corrections flow end-to-end through the grid and change I."""

    def test_weight_reweights_the_grid_else_count_mean(self):
        """A per-pixel norm feeds the Σraw/Σnorm grid (so it differs from the
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
        # large enough to move the Σraw/Σnorm mean unmistakably.
        weight = np.linspace(0.1, 1.0, 32 * 32).reshape(32, 32)

        plain = grid_img_data(mapper, img, angles, energy=12000.0,
                              bins=(6, 6, 6), mask_static_pixels=False)
        corrected = grid_img_data(mapper, img, angles, energy=12000.0,
                                  bins=(6, 6, 6), mask_static_pixels=False,
                                  weight=weight)
        both = np.isfinite(plain.intensity) & np.isfinite(corrected.intensity)
        assert both.any()                                # same support
        assert not np.allclose(plain.intensity[both], corrected.intensity[both])
