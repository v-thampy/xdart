"""P4 — the grazing-incidence stitch flag (StitchPlan.gi + pyfai_gi_q_frames).

The GI per-pixel geometry (αf, q_oop) is delegated to pyFAI's fiber units, so the
convention is pyFAI's own — gate-checked here via the textbook identity
``q_oop == k0·(sin αf + sin αi)``.  The correction *application* reuses the
already-gated P2b GICorrectionStack.  (The ABSOLUTE GI correctness — composition
signs + sample_orientation/tilt — is pending real-data GIXSGUI validation; these
tests pin the wiring + internal consistency, not the absolute physics.)
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


def _gi_diff_and_src():
    """psic Diffractometer (incident_angle active ← eta) + a 3-frame source whose
    eta=0.3° gives αi=0.3° (just above Si αc≈0.179° at 10 keV)."""
    from xrd_tools.core.geometry import DetectorCalibration, Diffractometer
    from xrd_tools.core.scan import ScanFrame
    from xrd_tools.sources import MemoryFrameSource
    base = PONI(dist=0.2, poni1=_SHAPE[0] * 172e-6 / 2, poni2=_SHAPE[1] * 172e-6 / 2,
                rot1=0.0, rot2=0.0, rot3=0.0, wavelength=1.0e-10, detector="Pilatus100k")
    frames = [ScanFrame(i, image=_ring(i),
                        metadata={"nu": float(i), "del": float(5 * i), "eta": 0.3})
              for i in range(3)]
    src = MemoryFrameSource(frames, name="gi-stitch")
    psic = Diffractometer.psic()
    diff = Diffractometer(preset="psic", rot1=psic.rot1, rot2=psic.rot2,
                          incident_angle=psic.incident_angle,
                          calibration=DetectorCalibration(poni=base))
    return diff, src


class TestGISettings:
    """The shared GI config object (StitchPlan.gi / RSMPlan.gi)."""

    def test_roundtrip_to_from_dict(self):
        from xrd_tools.corrections.grazing import GICorrectionStack, GISettings
        gi = GISettings(
            corrections=GICorrectionStack(material="Si", energy_eV=10000.0,
                                          footprint=True, refraction=False),
            incident_angle_deg=0.35, sample_orientation=3, tilt_deg=1.5)
        back = GISettings.from_dict(gi.to_dict())
        assert back.incident_angle_deg == 0.35
        assert back.sample_orientation == 3 and back.tilt_deg == 1.5
        assert back.corrections.material == "Si"
        assert back.corrections.footprint is True and back.corrections.refraction is False

    def test_empty_settings_roundtrip(self):
        from xrd_tools.corrections.grazing import GISettings
        back = GISettings.from_dict(GISettings().to_dict())
        assert back.corrections is None and back.incident_angle_deg is None


class TestGIConvention:
    """Pin the pyFAI fiber convention the GI provider delegates to."""

    def test_qoop_equals_k0_sin_af_plus_ai(self):
        pytest.importorskip("pyFAI")
        import pyFAI.units as U
        from pyFAI.integrator.fiber import FiberIntegrator
        fi = FiberIntegrator(dist=0.2, poni1=_SHAPE[0] * 172e-6 / 2,
                             poni2=_SHAPE[1] * 172e-6 / 2, rot1=0, rot2=0, rot3=0,
                             detector="Pilatus100k", wavelength=1.0e-10)
        air = np.deg2rad(0.3)
        fi.reset_integrator(incident_angle=air, tilt_angle=0.0, sample_orientation=1)
        af = fi.array_from_unit(_SHAPE, "center",
                                U.get_unit_fiber("exit_angle_vert_rad", incident_angle=air))
        qoop = fi.array_from_unit(_SHAPE, "center",
                                  U.get_unit_fiber("qoop_A^-1", incident_angle=air))
        k0 = 2 * np.pi / 1.0   # λ = 1 Å
        np.testing.assert_allclose(qoop, k0 * (np.sin(af) + np.sin(air)),
                                   atol=1e-6)


class TestGIProvider:
    def test_gi_all_off_equals_plain_provider(self):
        """A GICorrectionStack with every factor off (incl. refraction) must yield a
        byte-equal frame to pyfai_q_frames — the GI path adds nothing when idle."""
        pytest.importorskip("pyFAI")
        from xrd_tools.corrections.grazing import GICorrectionStack, GISettings
        from xrd_tools.integrate.stitch_hist import pyfai_gi_q_frames, pyfai_q_frames
        ai = _ai(); img = _ring(0)
        off = GICorrectionStack(material="Si", energy_eV=10000.0, footprint=False,
                                fresnel=False, absorption=False, refraction=False)
        q0, c0, s0, w0 = next(pyfai_q_frames([img], [ai]))
        q1, c1, s1, w1 = next(pyfai_gi_q_frames([img], [ai], gi=off,
                                                incident_angles_deg=[0.3]))
        np.testing.assert_allclose(q1, q0)        # refraction off ⇒ |q| unchanged
        np.testing.assert_allclose(w1, w0)        # all factors off ⇒ unit weight
        np.testing.assert_allclose(s1, s0)

    def test_footprint_only_scales_weight_by_inv_sin_ai(self):
        pytest.importorskip("pyFAI")
        from xrd_tools.corrections.grazing import GICorrectionStack, GISettings
        from xrd_tools.integrate.stitch_hist import pyfai_gi_q_frames, pyfai_q_frames
        ai = _ai(); img = _ring(0); aideg = 0.3
        fp = GICorrectionStack(material="Si", energy_eV=10000.0, footprint=True,
                               fresnel=False, absorption=False, refraction=False)
        _q0, _c0, _s0, w0 = next(pyfai_q_frames([img], [ai]))
        _q1, _c1, _s1, w1 = next(pyfai_gi_q_frames([img], [ai], gi=fp,
                                                   incident_angles_deg=[aideg]))
        # footprint boost 1/sin αi multiplies into the Σnorm weight
        np.testing.assert_allclose(w1, w0 / np.sin(np.deg2rad(aideg)))

    def test_refraction_toggle_rewrites_q_else_leaves_it(self):
        """Provider wiring: refraction=True rewrites |q| (and stays finite/≥0);
        refraction=False leaves |q| byte-equal to the plain provider. (The refraction
        *physics* — shift vanishing above αc — is gate-tested in test_grazing.py.)"""
        pytest.importorskip("pyFAI")
        from xrd_tools.corrections.grazing import GICorrectionStack, GISettings
        from xrd_tools.integrate.stitch_hist import pyfai_gi_q_frames, pyfai_q_frames
        ai = _ai(); img = _ring(0)
        q_plain, *_ = next(pyfai_q_frames([img], [ai]))
        on = GICorrectionStack(material="Si", energy_eV=10000.0, footprint=False,
                               fresnel=False, absorption=False, refraction=True)
        off = GICorrectionStack(material="Si", energy_eV=10000.0, footprint=False,
                                fresnel=False, absorption=False, refraction=False)
        q_on, *_ = next(pyfai_gi_q_frames([img], [ai], gi=on,
                                          incident_angles_deg=[0.3]))   # physical αi
        q_off, *_ = next(pyfai_gi_q_frames([img], [ai], gi=off,
                                           incident_angles_deg=[0.3]))
        assert not np.allclose(q_on, q_plain)              # refraction is applied
        np.testing.assert_allclose(q_off, q_plain)         # off ⇒ untouched
        assert np.all(np.isfinite(q_on)) and np.all(q_on >= 0.0)

    def test_bad_incident_angle_count_raises(self):
        pytest.importorskip("pyFAI")
        from xrd_tools.corrections.grazing import GICorrectionStack
        from xrd_tools.integrate.stitch_hist import pyfai_gi_q_frames
        gi = GICorrectionStack(material="Si", energy_eV=10000.0)
        with pytest.raises(ValueError, match="incident angle"):
            list(pyfai_gi_q_frames([_ring(0), _ring(1)], [_ai(), _ai()], gi=gi,
                                    incident_angles_deg=[0.3]))  # 1 αi / 2 frames


class TestGIRunStitch:
    def test_gi_requires_pyfai_hist_backend(self):
        pytest.importorskip("pyFAI")
        from xrd_tools.analysis.plans import StitchPlan, run_stitch
        from xrd_tools.corrections.grazing import GICorrectionStack, GISettings
        diff, src = _gi_diff_and_src()
        gi = GICorrectionStack(material="Si", energy_eV=10000.0)
        with pytest.raises(ValueError, match="only available on the 'pyfai_hist'"):
            run_stitch(StitchPlan(diffractometer=diff, gi=GISettings(corrections=gi),
                                  backend="multigeometry"), src)

    def test_gi_requires_diffractometer(self):
        pytest.importorskip("pyFAI")
        from xrd_tools.analysis.plans import StitchPlan, run_stitch
        from xrd_tools.corrections.grazing import GICorrectionStack, GISettings
        _diff, src = _gi_diff_and_src()
        base = PONI(dist=0.2, poni1=_SHAPE[0] * 172e-6 / 2, poni2=_SHAPE[1] * 172e-6 / 2,
                    wavelength=1.0e-10, detector="Pilatus100k")
        gi = GICorrectionStack(material="Si", energy_eV=10000.0)
        with pytest.raises(ValueError, match="requires the geometry path"):
            run_stitch(StitchPlan(base_poni=base, rot1_key="nu", rot2_key="del",
                                  backend="pyfai_hist", gi=GISettings(corrections=gi)), src)

    def test_gi_off_matches_non_gi_run(self):
        """An all-off GICorrectionStack run_stitch must equal the plain pyfai_hist run."""
        pytest.importorskip("pyFAI")
        from xrd_tools.analysis.plans import StitchPlan, run_stitch
        from xrd_tools.corrections.grazing import GICorrectionStack, GISettings
        diff, src = _gi_diff_and_src()
        off = GICorrectionStack(material="Si", energy_eV=10000.0, footprint=False,
                                fresnel=False, absorption=False, refraction=False)
        plain = run_stitch(StitchPlan(diffractometer=diff, backend="pyfai_hist",
                                      mode="1d", npt_1d=200), src)
        gi = run_stitch(StitchPlan(diffractometer=diff, backend="pyfai_hist",
                                   mode="1d", npt_1d=200, gi=GISettings(corrections=off)), src)
        np.testing.assert_allclose(gi.payload.intensity, plain.payload.intensity,
                                   equal_nan=True)

    def test_gi_footprint_run_scales_intensity(self):
        """run_stitch GI footprint-only ⇒ I = I_nonGI · sin(αi) (αi=0.3° from eta):
        the 1/sin αi boost in the Σnorm denominator divides the over-illumination
        back out, so the corrected intensity is smaller at grazing."""
        pytest.importorskip("pyFAI")
        from xrd_tools.analysis.plans import StitchPlan, run_stitch
        from xrd_tools.corrections.grazing import GICorrectionStack, GISettings
        diff, src = _gi_diff_and_src()
        fp = GICorrectionStack(material="Si", energy_eV=10000.0, footprint=True,
                               fresnel=False, absorption=False, refraction=False)
        plain = run_stitch(StitchPlan(diffractometer=diff, backend="pyfai_hist",
                                      mode="1d", npt_1d=200), src)
        gi = run_stitch(StitchPlan(diffractometer=diff, backend="pyfai_hist",
                                   mode="1d", npt_1d=200, gi=GISettings(corrections=fp)), src)
        m = np.isfinite(plain.payload.intensity) & np.isfinite(gi.payload.intensity)
        ratio = gi.payload.intensity[m] / plain.payload.intensity[m]
        np.testing.assert_allclose(ratio, np.sin(np.deg2rad(0.3)), rtol=1e-6)

    def test_gi_explicit_incident_angle_override(self):
        pytest.importorskip("pyFAI")
        from xrd_tools.analysis.plans import StitchPlan, run_stitch
        from xrd_tools.corrections.grazing import GICorrectionStack, GISettings
        diff, src = _gi_diff_and_src()
        fp = GICorrectionStack(material="Si", energy_eV=10000.0, footprint=True,
                               fresnel=False, absorption=False, refraction=False)
        plain = run_stitch(StitchPlan(diffractometer=diff, backend="pyfai_hist",
                                      mode="1d", npt_1d=200), src)
        # override αi to 1.0° (ignore the eta=0.3° in the metadata)
        gi = run_stitch(StitchPlan(diffractometer=diff, backend="pyfai_hist",
                                   mode="1d", npt_1d=200,
                                   gi=GISettings(corrections=fp, incident_angle_deg=1.0)), src)
        m = np.isfinite(plain.payload.intensity) & np.isfinite(gi.payload.intensity)
        ratio = gi.payload.intensity[m] / plain.payload.intensity[m]
        np.testing.assert_allclose(ratio, np.sin(np.deg2rad(1.0)), rtol=1e-6)
