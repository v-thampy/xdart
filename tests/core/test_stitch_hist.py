"""P3 — the histogram stitch merge (stitch_q_grid) + the pyfai_hist provider.

Gates:
* single-frame pyfai_hist 1D == pyFAI ``integrate1d`` (exact — the Σraw/Σnorm
  scheme, the same one CorrectionStack uses);
* multi-frame pyfai_hist 1D agrees in SHAPE with pyFAI ``MultiGeometry`` (a
  different merge engine; absolute-vs-normalized solid angle ⇒ a global scale);
* the 2D (q, χ) merge produces a sane cake; mask + error paths.
"""
from __future__ import annotations

import numpy as np
import pytest

from xrd_tools.core.containers import PONI


_SHAPE = (195, 487)


def _ai(rot1=0.0, rot2=0.0):
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    from pyFAI.detectors import detector_factory
    det = detector_factory("Pilatus100k")
    return AzimuthalIntegrator(
        dist=0.2, poni1=_SHAPE[0] * 172e-6 / 2, poni2=_SHAPE[1] * 172e-6 / 2,
        rot1=rot1, rot2=rot2, rot3=0.0, detector=det, wavelength=1.0e-10)


def _ring(seed):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:_SHAPE[0], :_SHAPE[1]]
    r = np.sqrt((y - _SHAPE[0] / 2) ** 2 + (x - _SHAPE[1] / 2) ** 2)
    return (500.0 * np.exp(-((r - 60.0) / 10.0) ** 2)
            + rng.poisson(3, size=_SHAPE)).astype(float)


class TestStitchQGrid:
    def test_single_frame_equals_pyfai_integrate1d(self):
        pytest.importorskip("pyFAI")
        from xrd_tools.corrections.stack import CorrectionStack
        from xrd_tools.integrate.stitch_hist import pyfai_q_frames, stitch_q_grid
        ai = _ai(); img = _ring(0)
        stack = CorrectionStack(solid_angle=True, polarization_factor=0.99)
        ref = ai.integrate1d(img, 250, unit="q_A^-1", correctSolidAngle=True,
                             polarization_factor=0.99,
                             method=("no", "histogram", "cython"))
        rng = (ref.radial[0] - (ref.radial[1] - ref.radial[0]) / 2,
               ref.radial[-1] + (ref.radial[1] - ref.radial[0]) / 2)
        out = stitch_q_grid(
            pyfai_q_frames([img], [ai], corrections=stack),
            mode="1d", npt=250, radial_range=rng)
        m = np.isfinite(out.intensity) & (ref.intensity > 0)
        rel = np.abs(out.intensity[m] - ref.intensity[m]) / np.maximum(
            np.abs(ref.intensity[m]), 1e-9)
        assert np.nanmedian(rel) < 1e-5  # exact Σraw/Σnorm per bin

    def test_multiframe_shape_matches_multigeometry(self):
        pytest.importorskip("pyFAI")
        from xrd_tools.corrections.stack import CorrectionStack
        from xrd_tools.integrate.multi import (
            create_multigeometry_integrators, stitch_1d)
        from xrd_tools.integrate.stitch_hist import pyfai_q_frames, stitch_q_grid
        base = PONI(dist=0.2, poni1=_SHAPE[0] * 172e-6 / 2,
                    poni2=_SHAPE[1] * 172e-6 / 2, rot1=0.0, rot2=0.0, rot3=0.0,
                    wavelength=1.0e-10, detector="Pilatus100k")
        ais = create_multigeometry_integrators(base, rot1_angles=np.array([0., 5., 10.]))
        imgs = [_ring(i) for i in range(3)]
        ref = stitch_1d(imgs, ais, npt=300, unit="q_A^-1", polarization_factor=0.99)
        stack = CorrectionStack(solid_angle=True, polarization_factor=0.99)
        out = stitch_q_grid(
            pyfai_q_frames(imgs, ais, corrections=stack),
            mode="1d", npt=300,
            radial_range=(ref.radial[0], ref.radial[-1]))
        m = (np.isfinite(out.intensity) & (ref.intensity > 0)
             & (out.intensity > 0))
        # MultiGeometry uses absolute solid angle ⇒ a single global scale
        scale = np.nanmedian(ref.intensity[m] / out.intensity[m])
        rel = np.abs(out.intensity[m] * scale - ref.intensity[m]) / np.maximum(
            np.abs(ref.intensity[m]), 1e-9)
        assert np.nanmedian(rel) < 0.03   # two different merge engines, ~1.5% typ.

    def test_2d_cake_is_sane(self):
        pytest.importorskip("pyFAI")
        from xrd_tools.integrate.stitch_hist import pyfai_q_frames, stitch_q_grid
        ais = [_ai(rot1=0.0), _ai(rot2=np.deg2rad(5))]
        imgs = [_ring(0), _ring(1)]
        out = stitch_q_grid(pyfai_q_frames(imgs, ais), mode="2d",
                            npt=120, npt_azim=90)
        assert out.intensity.shape == (120, 90)
        assert np.isfinite(out.intensity).any()
        assert np.nanmax(out.intensity) > 0

    def test_mask_excludes_pixels(self):
        pytest.importorskip("pyFAI")
        from xrd_tools.integrate.stitch_hist import pyfai_q_frames
        ai = _ai(); img = _ring(0)
        mask = np.zeros(_SHAPE, dtype=bool); mask[:50, :] = True
        q, chi, sig, w = next(pyfai_q_frames([img], [ai], mask=mask))
        assert np.all(w[:50, :] == 0.0)       # masked pixels carry no weight
        assert np.any(w[50:, :] > 0.0)

    def test_empty_raises(self):
        from xrd_tools.integrate.stitch_hist import stitch_q_grid
        with pytest.raises(ValueError, match="no frames"):
            stitch_q_grid([], mode="1d")
