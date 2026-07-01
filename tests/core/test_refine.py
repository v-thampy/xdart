"""Tests for refine_goniometer (step 4b — the Refine-button backend).

The committed gate is a **synthetic round-trip**: take a known goniometer
(the real del/nu ``MG_gonio_object.json``), forward-project LaB6 powder rings
onto the detector at several (del, nu) frames to make control points, then run
``refine_goniometer`` from a perturbed seed and assert it recovers the known
per-frame geometry.  (The real-data |q|-RMS / beam-centre gate runs in the
``Multi120_Calibration_*`` notebooks, outside the repo.)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xrd_tools.core.containers import PONI
from xrd_tools.core.geometry import Diffractometer

_FIXTURES = Path(__file__).parent / "fixtures"
_GONIO_DELNU = _FIXTURES / "gonio_del_nu_object.json"


def _truth():
    return Diffractometer.from_pyfai_goniometer(
        _GONIO_DELNU, source_motors={"del_value": "del", "nu_value": "nu"},
        base=Diffractometer.psic())


def _synth_control_frames(truth, frames_dn, *, n_rings=24, tol=0.003,
                          per_ring=12, seed=0):
    """Forward-project LaB6 rings to control points for a known geometry."""
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    from pyFAI.detectors import detector_factory
    from pyFAI.calibrant import get_calibrant
    from xrd_tools.integrate.refine import ControlFrame

    cal = truth.calibration
    wl = cal.poni.wavelength
    det = detector_factory("Pilatus300kw", config={"orientation": 3})
    lab6 = get_calibrant("LaB6"); lab6.set_wavelength(wl)
    q_ring = 2.0 * np.pi / np.asarray(lab6.get_dSpacing())
    rng = np.random.RandomState(seed)

    out = []
    for de, nu in frames_dn:
        pf = truth.to_pyfai_per_frame({"del": np.array([de]), "nu": np.array([nu])})
        ai = AzimuthalIntegrator(
            dist=cal.poni.dist, poni1=cal.poni.poni1, poni2=cal.poni.poni2,
            rot1=cal.poni.rot1 + pf["rot1"][0], rot2=cal.poni.rot2 + pf["rot2"][0],
            rot3=cal.poni.rot3 + pf["rot3"][0], detector=det, wavelength=wl)
        qA = ai.qArray() / 10.0
        rows, cols, rings = [], [], []
        for ri, qr in enumerate(q_ring[:n_rings]):
            ys, xs = np.where(np.abs(qA - qr) < tol)
            if len(ys) < 6:
                continue
            idx = rng.choice(len(ys), size=min(per_ring, len(ys)), replace=False)
            rows += list(ys[idx]); cols += list(xs[idx]); rings += [ri] * len(idx)
        out.append(ControlFrame(rows=rows, cols=cols, rings=rings,
                                motors={"del": de, "nu": nu}))
    return out


def _max_rot_error(truth, fit, frames_dn):
    cal_t, cal_f = truth.calibration, fit.calibration
    err = 0.0
    for de, nu in frames_dn:
        pt = truth.to_pyfai_per_frame({"del": np.array([de]), "nu": np.array([nu])})
        pf = fit.to_pyfai_per_frame({"del": np.array([de]), "nu": np.array([nu])})
        t1 = cal_t.poni.rot1 + pt["rot1"][0]; t2 = cal_t.poni.rot2 + pt["rot2"][0]
        f1 = cal_f.poni.rot1 + pf["rot1"][0]; f2 = cal_f.poni.rot2 + pf["rot2"][0]
        err = max(err, abs(t1 - f1), abs(t2 - f2))
    return err


_FRAMES = [(6.0, 0.0), (15.0, 3.0), (25.0, 6.0), (40.0, 9.0)]


class TestRefineGoniometer:
    def test_synthetic_roundtrip_recovers_geometry(self):
        pytest.importorskip("pyFAI")
        pytest.importorskip("scipy")
        from xrd_tools.integrate.refine import refine_goniometer

        truth = _truth()
        frames = _synth_control_frames(truth, _FRAMES)
        assert sum(len(f.rows) for f in frames) > 200  # enough constraints

        cal = truth.calibration
        # perturb the base poni seed (the fit must find its way back)
        seed = PONI(dist=cal.poni.dist * 1.02, poni1=cal.poni.poni1 * 0.97,
                    poni2=cal.poni.poni2 * 1.03, wavelength=cal.poni.wavelength,
                    detector="Pilatus300kw")
        res = refine_goniometer(
            seed, frames, rot1_motor="nu", rot2_motor="del", calibrant="LaB6",
            detector_config={"orientation": 3}, base=Diffractometer.psic())

        assert res.success
        assert res.rms_q < 0.006              # clean synthetic points
        assert res.condition_number < 1e8     # well-spanned -> well-conditioned
        assert res.frozen_scales == ()        # both motors span -> both scales fit
        d = res.diffractometer
        # the correspondence is recovered: rot1<-nu, rot2<-del
        assert d.rot1.source_motor == "nu" and d.rot1.is_active
        assert d.rot2.source_motor == "del" and d.rot2.is_active
        # rot3 (beam-axis rotation) is pinned at the base, NOT diverged to ~1e9 rad
        assert abs(d.calibration.poni.rot3) < 1e-3
        # the fitted per-frame geometry tracks the truth (held-out frames)
        held = [(10.0, 2.0), (35.0, 8.0), (45.0, 9.0)]
        assert _max_rot_error(truth, d, held) < 3e-3     # rad (~0.17°)
        # the result is a complete, persistable canonical object
        assert d.calibration is not None
        assert d.calibration.detector_config.get("orientation") == 3
        assert d == Diffractometer.from_json(d.to_json())

    def test_rot3_is_not_fit(self):
        """rot3 leaves |q| invariant — fitting it free let LM diverge to ~1e9 rad
        and silently corrupt the azimuth.  It must stay pinned at the base."""
        pytest.importorskip("pyFAI")
        from xrd_tools.integrate.refine import refine_goniometer
        truth = _truth()
        frames = _synth_control_frames(truth, _FRAMES)
        res = refine_goniometer(
            truth.calibration.poni, frames, rot1_motor="nu", rot2_motor="del",
            detector_config={"orientation": 3}, base=Diffractometer.psic())
        # base rot3 was ~7e-12; it must not have wandered
        assert abs(res.diffractometer.calibration.poni.rot3) < 1e-6
        assert abs(res.params["rot3_offset"]) < 1e-6

    def test_non_spanning_axis_is_flagged_not_catastrophic(self):
        """A motor that does not span (del fixed) makes its scale unidentifiable;
        it must be frozen + flagged, never fit to a low-RMS garbage value."""
        pytest.importorskip("pyFAI")
        from xrd_tools.integrate.refine import refine_goniometer
        truth = _truth()
        # del fixed at 15; only nu varies
        frames = _synth_control_frames(truth, [(15.0, 0.0), (15.0, 4.0), (15.0, 9.0)])
        res = refine_goniometer(
            truth.calibration.poni, frames, rot1_motor="nu", rot2_motor="del",
            detector_config={"orientation": 3}, base=Diffractometer.psic())
        assert "del" in res.frozen_scales           # del scale frozen (un-spanned)
        # the frozen scale is the sane deg2rad, NOT a collapsed garbage value
        assert res.diffractometer.rot2.sign == pytest.approx(1.0, abs=1e-6)
        # nu still spans -> its scale is fit and the nu fit is usable
        assert "nu" not in res.frozen_scales

    def test_recovers_motor_zero_offsets(self):
        """The motor-zero offsets are first-class fit params (the missing
        ingredient on real data) — recovered into rot*.offset (degrees)."""
        pytest.importorskip("pyFAI")
        from xrd_tools.integrate.refine import refine_goniometer
        truth = _truth()
        frames = _synth_control_frames(truth, _FRAMES)
        res = refine_goniometer(
            truth.calibration.poni, frames, rot1_motor="nu", rot2_motor="del",
            detector_config={"orientation": 3}, base=Diffractometer.psic())
        # truth offsets: nu (rot1) +1.445°, del (rot2) -8.380°
        assert res.diffractometer.rot1.offset == pytest.approx(1.445, abs=0.1)
        assert res.diffractometer.rot2.offset == pytest.approx(-8.380, abs=0.2)

    def test_empty_frames_raises(self):
        from xrd_tools.integrate.refine import refine_goniometer
        with pytest.raises(ValueError, match="empty"):
            refine_goniometer(
                PONI(dist=0.39, poni1=0.03, poni2=0.05, wavelength=7.3e-11,
                     detector="Pilatus300kw"),
                [], rot1_motor="nu", rot2_motor="del")

    def test_missing_wavelength_raises(self):
        from xrd_tools.integrate.refine import refine_goniometer, ControlFrame
        cf = ControlFrame(rows=[1], cols=[1], rings=[0], motors={"nu": 0, "del": 6})
        with pytest.raises(ValueError, match="wavelength"):
            refine_goniometer(
                PONI(dist=0.39, poni1=0.03, poni2=0.05, detector="Pilatus300kw"),
                [cf], rot1_motor="nu", rot2_motor="del")

    def test_control_frame_coerces_arrays(self):
        from xrd_tools.integrate.refine import ControlFrame
        cf = ControlFrame(rows=[1, 2], cols=[3, 4], rings=[0, 1],
                          motors={"nu": 0.0, "del": 6.0})
        assert cf.rows.dtype == int and cf.rings.tolist() == [0, 1]

    def test_control_frame_rejects_mismatched_lengths(self):
        from xrd_tools.integrate.refine import ControlFrame
        with pytest.raises(ValueError, match="mismatched lengths"):
            ControlFrame(rows=[1, 2], cols=[3], rings=[0, 1],
                         motors={"nu": 0.0, "del": 6.0})

    def test_control_frame_rejects_negative_index(self):
        from xrd_tools.integrate.refine import ControlFrame
        with pytest.raises(ValueError, match="negative index"):
            ControlFrame(rows=[1, -2], cols=[3, 4], rings=[0, 1],
                         motors={"nu": 0.0, "del": 6.0})
