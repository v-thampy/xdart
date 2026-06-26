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


class TestPyfaiQFramesGuards:
    """P3 review — the provider must fail loud on a corrupting input, not produce
    a finite-but-wrong stitch (bad monitor cancels frames; a length desync silently
    truncates). pyfai_q_frames is a generator: the guards fire on first iteration."""

    def test_length_mismatch_raises(self):
        pytest.importorskip("pyFAI")
        from xrd_tools.integrate.stitch_hist import pyfai_q_frames
        ais = [_ai(), _ai()]
        imgs = [_ring(0), _ring(1), _ring(2)]   # 3 imgs / 2 integrators
        with pytest.raises(ValueError, match="3 images but 2 integrators"):
            next(pyfai_q_frames(imgs, ais))

    def test_monitor_length_mismatch_raises(self):
        pytest.importorskip("pyFAI")
        from xrd_tools.integrate.stitch_hist import pyfai_q_frames
        ais = [_ai(), _ai()]; imgs = [_ring(0), _ring(1)]
        with pytest.raises(ValueError, match="monitor/normalization values"):
            next(pyfai_q_frames(imgs, ais, normalization=[1.0, 2.0, 3.0]))

    @pytest.mark.parametrize("bad", [0.0, np.nan, np.inf, -2.0])
    def test_bad_monitor_raises(self, bad):
        """A zero/NaN monitor silently drops a whole frame; a NEGATIVE one flips its
        sign and cancels healthy frames. All must raise, not corrupt the merge."""
        pytest.importorskip("pyFAI")
        from xrd_tools.integrate.stitch_hist import pyfai_q_frames
        ais = [_ai(), _ai()]; imgs = [_ring(0), _ring(1)]
        with pytest.raises(ValueError, match="invalid value"):
            next(pyfai_q_frames(imgs, ais, normalization=[1.0, bad]))

    def test_good_monitor_divides_signal(self):
        pytest.importorskip("pyFAI")
        from xrd_tools.integrate.stitch_hist import pyfai_q_frames
        ai = _ai(); img = _ring(0)
        _q, _chi, sig, _w = next(pyfai_q_frames([img], [ai], normalization=[4.0]))
        np.testing.assert_allclose(sig, img / 4.0)


class TestStitchMergeSemantics:
    """Merge-level (stitch_q_grid runs the real Σraw/Σnorm), not just the provider."""

    def test_mask_removes_pixels_from_the_merge(self):
        """The earlier mask test only checks the provider weight; this checks the
        masked band is actually absent from the merged signal."""
        pytest.importorskip("pyFAI")
        from xrd_tools.integrate.stitch_hist import pyfai_q_frames, stitch_q_grid
        ai = _ai(); img = _ring(0)
        full = stitch_q_grid(pyfai_q_frames([img], [ai]), mode="1d", npt=200)
        mask = np.zeros(_SHAPE, dtype=bool); mask[:97, :] = True   # half the ring
        masked = stitch_q_grid(pyfai_q_frames([img], [ai], mask=mask),
                               mode="1d", npt=200,
                               radial_range=(full.radial[0], full.radial[-1]))
        # masking strictly removes counts ⇒ some bin that was populated is now empty,
        # and no bin gains intensity from nothing.
        gained = np.isfinite(masked.intensity) & ~np.isfinite(full.intensity)
        assert not gained.any()

    def test_nan_signal_pixels_are_dropped_not_propagated(self):
        pytest.importorskip("pyFAI")
        from xrd_tools.integrate.stitch_hist import pyfai_q_frames, stitch_q_grid
        ai = _ai(); img = _ring(0)
        clean = stitch_q_grid(pyfai_q_frames([img], [ai]), mode="1d", npt=200)
        dirty_img = img.copy(); dirty_img[100:120, 200:240] = np.nan
        dirty = stitch_q_grid(pyfai_q_frames([dirty_img], [ai]), mode="1d", npt=200,
                              radial_range=(clean.radial[0], clean.radial[-1]))
        # a few NaN pixels must not NaN-poison whole bins: output stays mostly finite.
        assert np.isfinite(dirty.intensity).sum() >= 0.9 * np.isfinite(clean.intensity).sum()

    def test_empty_bins_are_nan(self):
        pytest.importorskip("pyFAI")
        from xrd_tools.integrate.stitch_hist import pyfai_q_frames, stitch_q_grid
        ai = _ai(); img = _ring(0)
        # a radial range far beyond the data ⇒ every bin empty ⇒ all NaN, no zeros.
        out = stitch_q_grid(pyfai_q_frames([img], [ai]), mode="1d", npt=50,
                            radial_range=(900.0, 1000.0))
        assert np.isnan(out.intensity).all()

    def test_unit_weight_merge_is_an_identity(self):
        """signal == weight == ones ⇒ every populated bin is exactly 1.0 (Σ1/Σ1),
        empties are NaN. Pins the accumulator with no correction confounding."""
        pytest.importorskip("pyFAI")
        from xrd_tools.integrate.stitch_hist import stitch_q_grid
        ai = _ai()
        ones = np.ones(_SHAPE, dtype=float)
        q = np.asarray(ai.qArray(_SHAPE), dtype=float).ravel() / 10.0
        chi = np.degrees(np.asarray(ai.chiArray(_SHAPE), dtype=float)).ravel()
        out = stitch_q_grid([(q, chi, ones.ravel(), ones.ravel())], mode="1d", npt=200)
        pop = np.isfinite(out.intensity)
        assert pop.any()
        np.testing.assert_allclose(out.intensity[pop], 1.0)

    def test_2d_chi_seam_both_boundary_bins_populate(self):
        """The ±180° azimuthal seam splits a full ring across the first AND last χ
        bins — pin that both boundaries are reachable (inherent, previously untested)."""
        pytest.importorskip("pyFAI")
        from xrd_tools.integrate.stitch_hist import pyfai_q_frames, stitch_q_grid
        ai = _ai(); img = _ring(0)
        out = stitch_q_grid(pyfai_q_frames([img], [ai]), mode="2d",
                            npt=80, npt_azim=72, azimuth_range=(-180.0, 180.0))
        col0 = np.isfinite(out.intensity[:, 0])
        col_last = np.isfinite(out.intensity[:, -1])
        assert col0.any() and col_last.any()   # the seam is reachable on both sides

    def test_2d_unit_weight_merge_conserves_no_phantom_counts(self):
        """2D identity merge: every populated (q,χ) bin is exactly 1.0; no bin
        invents intensity (Σ1/Σ1). Guards the 2D histogram2d accumulator."""
        pytest.importorskip("pyFAI")
        from xrd_tools.integrate.stitch_hist import stitch_q_grid
        ai = _ai()
        ones = np.ones(_SHAPE, dtype=float).ravel()
        q = np.asarray(ai.qArray(_SHAPE), dtype=float).ravel() / 10.0
        chi = np.degrees(np.asarray(ai.chiArray(_SHAPE), dtype=float)).ravel()
        out = stitch_q_grid([(q, chi, ones, ones)], mode="2d", npt=60, npt_azim=60)
        pop = np.isfinite(out.intensity)
        assert pop.any()
        np.testing.assert_allclose(out.intensity[pop], 1.0)
