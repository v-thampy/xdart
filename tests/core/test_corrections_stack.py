"""The shared per-pixel correction stack (P2a).

The load-bearing gate: a binned intensity formed as ``Σ raw / Σ normalization``
(pyFAI's scheme) reproduces pyFAI ``integrate1d(correctSolidAngle, ...,
polarization_factor=...)`` — proving the stack's normalization is correct AND
that the accumulator scheme (not the naive ``Σ(raw·weight)/N``) is the right one.
"""
from __future__ import annotations

import numpy as np
import pytest


def _ai(orientation: int = 3):
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    from pyFAI.detectors import detector_factory
    det = detector_factory("Pilatus300kw", config={"orientation": orientation})
    return AzimuthalIntegrator(dist=0.39, poni1=0.10, poni2=0.12, rot1=0.0,
                               rot2=0.05, rot3=0.0, detector=det,
                               wavelength=0.729e-10)


def _raw(shape):
    return np.random.RandomState(1).gamma(2.0, 50.0, size=shape)


class TestCorrectionStack:
    def test_per_pixel_normalization_is_solidangle_times_polarization(self):
        pytest.importorskip("pyFAI")
        from xrd_tools.corrections.stack import CorrectionStack
        ai = _ai(); shape = ai.detector.shape
        stack = CorrectionStack(solid_angle=True, polarization_factor=0.99)
        norm = stack.normalization(ai, shape)
        ref = (np.asarray(ai.solidAngleArray(shape=shape))
               * np.asarray(ai.polarization(shape=shape, factor=0.99)))
        np.testing.assert_allclose(norm, ref, rtol=0, atol=0)

    def test_scheme_reproduces_pyfai_integrate1d(self):
        """Σraw/Σnorm over q-bins == pyFAI integrate1d on well-populated bins."""
        pytest.importorskip("pyFAI")
        from xrd_tools.corrections.stack import CorrectionStack
        ai = _ai(); shape = ai.detector.shape
        raw = _raw(shape)
        stack = CorrectionStack(solid_angle=True, polarization_factor=0.99)
        norm = stack.normalization(ai, shape)
        q = np.asarray(ai.array_from_unit(unit="q_A^-1"))

        # wide bins (~1900 px/bin) so a ±1-pixel bin-edge assignment difference
        # between np.histogram and pyFAI's exact edges is negligible — the test
        # is the normalization SCHEME, not bin-edge bookkeeping.
        N = 150
        res = ai.integrate1d(raw, N, unit="q_A^-1", correctSolidAngle=True,
                             polarization_factor=0.99,
                             method=("no", "histogram", "cython"))
        dq = res.radial[1] - res.radial[0]
        edges = np.linspace(res.radial[0] - dq / 2, res.radial[-1] + dq / 2, N + 1)
        sig = np.histogram(q.ravel(), edges, weights=raw.ravel())[0]
        nrm = np.histogram(q.ravel(), edges, weights=norm.ravel())[0]
        cnt = np.histogram(q.ravel(), edges)[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            i_mine = sig / nrm
        good = cnt > 200
        rel = np.abs(i_mine[good] - res.intensity[good]) / np.maximum(
            np.abs(res.intensity[good]), 1e-9)
        assert np.nanmedian(rel) < 1e-6
        assert np.nanpercentile(rel, 95) < 3e-3

    def test_corrections_are_not_a_no_op(self):
        pytest.importorskip("pyFAI")
        from xrd_tools.corrections.stack import CorrectionStack
        ai = _ai(); shape = ai.detector.shape
        on = CorrectionStack(solid_angle=True, polarization_factor=0.99).normalization(ai, shape)
        assert not np.allclose(on, 1.0)

    def test_identity_stack(self):
        from xrd_tools.corrections.stack import CorrectionStack
        s = CorrectionStack(solid_angle=False, polarization_factor=None)
        assert s.is_identity
        pytest.importorskip("pyFAI")
        ai = _ai()
        np.testing.assert_array_equal(s.normalization(ai, ai.detector.shape),
                                      np.ones(ai.detector.shape))

    def test_weight_is_inverse_normalization(self):
        pytest.importorskip("pyFAI")
        from xrd_tools.corrections.stack import CorrectionStack
        ai = _ai(); shape = ai.detector.shape
        s = CorrectionStack(solid_angle=True, polarization_factor=0.95)
        norm = s.normalization(ai, shape)
        w = s.weight(ai, shape)
        np.testing.assert_allclose(w[np.isfinite(w)], 1.0 / norm[np.isfinite(w)])

    def test_provenance_roundtrip(self):
        from xrd_tools.corrections.stack import CorrectionStack
        s = CorrectionStack(solid_angle=True, polarization_factor=0.97,
                            air_absorption_mu=0.8)
        assert CorrectionStack.from_dict(s.to_dict()) == s


class TestDetectorCalibrationToIntegrator:
    def test_honors_detector_config_orientation(self):
        pytest.importorskip("pyFAI")
        from xrd_tools.core.containers import PONI
        from xrd_tools.core.geometry import DetectorCalibration
        from xrd_tools.integrate.calibration import detector_calibration_to_integrator
        cal = DetectorCalibration(
            poni=PONI(dist=0.39, poni1=0.10, poni2=0.12, wavelength=0.729e-10,
                      detector="Pilatus300kw"),
            detector_config={"orientation": 3})
        ai = detector_calibration_to_integrator(cal)
        assert int(ai.detector.orientation) == 3

    def test_per_frame_rotation_override(self):
        pytest.importorskip("pyFAI")
        from xrd_tools.core.containers import PONI
        from xrd_tools.core.geometry import DetectorCalibration
        from xrd_tools.integrate.calibration import detector_calibration_to_integrator
        cal = DetectorCalibration(
            poni=PONI(dist=0.39, poni1=0.10, poni2=0.12, wavelength=0.729e-10,
                      detector="Pilatus300kw"),
            detector_config={"orientation": 3})
        ai = detector_calibration_to_integrator(cal, rot2=0.31)
        assert ai.rot2 == pytest.approx(0.31)
        assert ai.rot1 == pytest.approx(0.0)  # base value kept
