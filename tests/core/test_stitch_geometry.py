"""P1 — stitch consumes the Diffractometer (closes GAP A/B).

Gates:
* an UNCALIBRATED psic (AngleMapping.sign==1) reproduces the legacy ``deg2rad``
  integrator path byte-for-byte (back-compat — refinement is optional);
* a CALIBRATED goniometer's per-frame integrators match pyFAI ``get_ai`` (the
  fitted scales are used, not a hardwired deg2rad — GAP A closed);
* the base ``Detector_config`` orientation survives (GAP B closed).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xrd_tools.core.containers import PONI
from xrd_tools.core.geometry import DetectorCalibration, Diffractometer

_GONIO_DELNU = Path(__file__).parent / "fixtures" / "gonio_del_nu_object.json"


def test_uncalibrated_psic_reproduces_legacy_deg2rad_path():
    pytest.importorskip("pyFAI")
    from xrd_tools.integrate.multi import (
        create_multigeometry_integrators,
        create_multigeometry_integrators_from_geometry,
    )
    base = PONI(dist=0.39, poni1=0.10, poni2=0.12, rot1=0.003, rot2=0.0, rot3=0.0,
                wavelength=0.729e-10, detector="Pilatus300kw")
    nu = np.array([0.0, 3.0, 6.0])
    del_ = np.array([6.0, 20.0, 40.0])

    # legacy path: psic maps rot1<-nu, rot2<-del -> rot1_angles=nu, rot2_angles=del
    legacy = create_multigeometry_integrators(base, rot1_angles=nu, rot2_angles=del_)

    # geometry path: an uncalibrated psic over the same base (sign == 1 ⇒ deg2rad)
    psic = Diffractometer.psic()
    diff = Diffractometer(
        preset="psic", rot1=psic.rot1, rot2=psic.rot2,
        calibration=DetectorCalibration(poni=base, detector_config={"orientation": 3}),
    )
    geom = create_multigeometry_integrators_from_geometry(
        diff, {"nu": nu, "del": del_})

    assert len(legacy) == len(geom) == 3
    for a, b in zip(legacy, geom):
        assert a.rot1 == pytest.approx(b.rot1)
        assert a.rot2 == pytest.approx(b.rot2)
        assert a.rot3 == pytest.approx(b.rot3)


def test_calibrated_goniometer_integrators_match_get_ai():
    """The fitted per-axis scales are used (GAP A): per-frame integrator rotations
    equal pyFAI Goniometer.get_ai(motors)."""
    gonio = pytest.importorskip("pyFAI.goniometer")
    from xrd_tools.integrate.multi import (
        create_multigeometry_integrators_from_geometry,
    )
    g = gonio.Goniometer.sload(str(_GONIO_DELNU))
    diff = Diffractometer.from_pyfai_goniometer(
        _GONIO_DELNU, source_motors={"del_value": "del", "nu_value": "nu"},
        base=Diffractometer.psic())
    nu = np.array([0.0, 5.0, 9.0])
    del_ = np.array([6.0, 20.0, 45.0])
    ais = create_multigeometry_integrators_from_geometry(
        diff, {"nu": nu, "del": del_})
    for i, (d, n) in enumerate(zip(del_, nu)):
        ref = g.get_ai((float(d), float(n)))
        assert ais[i].rot1 == pytest.approx(ref.rot1, abs=1e-12)
        assert ais[i].rot2 == pytest.approx(ref.rot2, abs=1e-12)
        assert ais[i].rot3 == pytest.approx(ref.rot3, abs=1e-12)
        assert ais[i].dist == pytest.approx(ref.dist)


def test_preserves_detector_config_orientation():
    pytest.importorskip("pyFAI")
    from xrd_tools.integrate.multi import (
        create_multigeometry_integrators_from_geometry,
    )
    diff = Diffractometer.from_pyfai_goniometer(
        _GONIO_DELNU, source_motors={"del_value": "del", "nu_value": "nu"},
        base=Diffractometer.psic())
    ais = create_multigeometry_integrators_from_geometry(
        diff, {"nu": np.array([3.0]), "del": np.array([20.0])})
    assert int(ais[0].detector.orientation) == 3


def test_stitch_without_incidence_motor():
    """A stitch never needs the GI incidence — a psic Diffractometer must build
    integrators from nu/del alone, NOT crash on a missing 'eta' column."""
    pytest.importorskip("pyFAI")
    from xrd_tools.integrate.multi import (
        create_multigeometry_integrators_from_geometry,
    )
    base = PONI(dist=0.39, poni1=0.10, poni2=0.12, wavelength=0.729e-10,
                detector="Pilatus300kw")
    psic = Diffractometer.psic()  # incident_angle active <- eta
    diff = Diffractometer(preset="psic", rot1=psic.rot1, rot2=psic.rot2,
                          incident_angle=psic.incident_angle,
                          calibration=DetectorCalibration(poni=base))
    # motors carry nu/del but NOT eta — must not raise
    ais = create_multigeometry_integrators_from_geometry(
        diff, {"nu": np.array([0.0, 3.0]), "del": np.array([6.0, 20.0])})
    assert len(ais) == 2

    # but a missing *rotation* motor still fails loud (and names the motor)
    with pytest.raises(KeyError, match="nu"):
        create_multigeometry_integrators_from_geometry(diff, {"del": np.array([6.0])})


def test_requires_a_calibration():
    from xrd_tools.integrate.multi import (
        create_multigeometry_integrators_from_geometry,
    )
    # a bare preset has no DetectorCalibration
    with pytest.raises(ValueError, match="DetectorCalibration"):
        create_multigeometry_integrators_from_geometry(
            Diffractometer.psic(), {"nu": np.array([0.0]), "del": np.array([6.0])})


def test_run_stitch_geometry_dispatch_equals_legacy_uncalibrated():
    """End-to-end: run_stitch via an uncalibrated-psic Diffractometer reproduces
    the legacy base_poni + rot1_key/rot2_key deg2rad path (back-compat)."""
    pytest.importorskip("pyFAI")
    from xrd_tools.analysis.plans import StitchPlan, run_stitch
    from xrd_tools.core.scan import ScanFrame
    from xrd_tools.sources import MemoryFrameSource

    shape = (195, 487)
    base = PONI(dist=0.2, poni1=shape[0] * 172e-6 / 2, poni2=shape[1] * 172e-6 / 2,
                rot1=0.0, rot2=0.0, rot3=0.0, wavelength=1.0e-10, detector="Pilatus100k")

    def _ring(seed):
        rng = np.random.default_rng(seed)
        y, x = np.mgrid[:shape[0], :shape[1]]
        r = np.sqrt((y - shape[0] / 2) ** 2 + (x - shape[1] / 2) ** 2)
        return (500.0 * np.exp(-((r - 60.0) / 10.0) ** 2)
                + rng.poisson(3, size=shape)).astype(float)

    # del/nu vary; eta (psic incidence) is a real-but-constant motor in the source
    frames = [ScanFrame(i, image=_ring(i),
                        metadata={"nu": float(i), "del": float(5 * i), "eta": 0.3})
              for i in range(3)]
    src = MemoryFrameSource(frames, name="stitch")

    psic = Diffractometer.psic()
    diff = Diffractometer(preset="psic", rot1=psic.rot1, rot2=psic.rot2,
                          incident_angle=psic.incident_angle,
                          calibration=DetectorCalibration(poni=base))

    geom = run_stitch(StitchPlan(diffractometer=diff, mode="1d", npt_1d=200), src)
    # legacy: psic maps rot1<-nu, rot2<-del
    legacy = run_stitch(StitchPlan(base_poni=base, rot1_key="nu", rot2_key="del",
                                   mode="1d", npt_1d=200), src)
    np.testing.assert_allclose(geom.payload.intensity, legacy.payload.intensity,
                               equal_nan=True)
    np.testing.assert_allclose(geom.payload.radial, legacy.payload.radial)


def test_run_stitch_backend_dispatch():
    """run_stitch(backend='pyfai_hist') routes to the streaming histogram merge and
    shape-matches the multigeometry backend; 'xu_hist' is a clear NotImplementedError."""
    pytest.importorskip("pyFAI")
    from xrd_tools.analysis.plans import StitchPlan, run_stitch
    from xrd_tools.core.scan import ScanFrame
    from xrd_tools.corrections.stack import CorrectionStack
    from xrd_tools.sources import MemoryFrameSource

    shape = (195, 487)
    base = PONI(dist=0.2, poni1=shape[0] * 172e-6 / 2, poni2=shape[1] * 172e-6 / 2,
                rot1=0.0, rot2=0.0, rot3=0.0, wavelength=1.0e-10, detector="Pilatus100k")

    def _ring(seed):
        rng = np.random.default_rng(seed)
        y, x = np.mgrid[:shape[0], :shape[1]]
        r = np.sqrt((y - shape[0] / 2) ** 2 + (x - shape[1] / 2) ** 2)
        return (500.0 * np.exp(-((r - 60.0) / 10.0) ** 2)
                + rng.poisson(3, size=shape)).astype(float)

    frames = [ScanFrame(i, image=_ring(i),
                        metadata={"nu": float(i), "del": float(5 * i), "eta": 0.3})
              for i in range(3)]
    src = MemoryFrameSource(frames, name="stitch")
    psic = Diffractometer.psic()
    diff = Diffractometer(preset="psic", rot1=psic.rot1, rot2=psic.rot2,
                          incident_angle=psic.incident_angle,
                          calibration=DetectorCalibration(poni=base))

    mg = run_stitch(StitchPlan(diffractometer=diff, mode="1d", npt_1d=250,
                               backend="multigeometry"), src)
    hist = run_stitch(StitchPlan(diffractometer=diff, mode="1d", npt_1d=250,
                                 backend="pyfai_hist",
                                 corrections=CorrectionStack(solid_angle=True,
                                                             polarization_factor=0.99)),
                      src)
    assert hist.payload.radial.shape == (250,)
    # the histogram backend shape-matches MG (absolute vs normalized solid angle)
    m = (np.isfinite(hist.payload.intensity) & (mg.payload.intensity > 0)
         & (hist.payload.intensity > 0))
    scale = np.nanmedian(mg.payload.intensity[m] / hist.payload.intensity[m])
    rel = np.abs(hist.payload.intensity[m] * scale - mg.payload.intensity[m]) / \
        np.maximum(np.abs(mg.payload.intensity[m]), 1e-9)
    assert np.nanmedian(rel) < 0.05

    with pytest.raises(NotImplementedError, match="xu_hist"):
        run_stitch(StitchPlan(diffractometer=diff, backend="xu_hist"), src)


def _stitch_diff_and_src():
    """A small psic Diffractometer + 3-frame source for the dispatch-guard tests."""
    from xrd_tools.core.scan import ScanFrame
    from xrd_tools.sources import MemoryFrameSource
    shape = (195, 487)
    base = PONI(dist=0.2, poni1=shape[0] * 172e-6 / 2, poni2=shape[1] * 172e-6 / 2,
                rot1=0.0, rot2=0.0, rot3=0.0, wavelength=1.0e-10, detector="Pilatus100k")

    def _ring(seed):
        rng = np.random.default_rng(seed)
        y, x = np.mgrid[:shape[0], :shape[1]]
        r = np.sqrt((y - shape[0] / 2) ** 2 + (x - shape[1] / 2) ** 2)
        return (500.0 * np.exp(-((r - 60.0) / 10.0) ** 2)
                + rng.poisson(3, size=shape)).astype(float)

    frames = [ScanFrame(i, image=_ring(i),
                        metadata={"nu": float(i), "del": float(5 * i), "eta": 0.3})
              for i in range(3)]
    src = MemoryFrameSource(frames, name="stitch")
    psic = Diffractometer.psic()
    diff = Diffractometer(preset="psic", rot1=psic.rot1, rot2=psic.rot2,
                          incident_angle=psic.incident_angle,
                          calibration=DetectorCalibration(poni=base))
    return diff, src


def test_pyfai_hist_rejects_silently_dropped_params():
    """P3 review — the pyfai_hist backend emits q in Å⁻¹ and cannot forward pyFAI
    kwargs, so a non-q unit or leftover extra must fail loud, not mislabel/vanish."""
    pytest.importorskip("pyFAI")
    from xrd_tools.analysis.plans import StitchPlan, run_stitch
    diff, src = _stitch_diff_and_src()

    # a non-q unit would mislabel the q-axis as 2θ
    with pytest.raises(ValueError, match="only emits q in"):
        run_stitch(StitchPlan(diffractometer=diff, backend="pyfai_hist",
                              unit="2th_deg", npt_1d=100), src)
    # pyFAI integrate kwargs would silently vanish
    with pytest.raises(ValueError, match="cannot consume pyFAI"):
        run_stitch(StitchPlan(diffractometer=diff, backend="pyfai_hist",
                              npt_1d=100, extra={"safe": True}), src)
    # method != BBox is ignored — warned, not fatal (still produces a result)
    out = run_stitch(StitchPlan(diffractometer=diff, backend="pyfai_hist",
                                npt_1d=100, method="splitpixel"), src)
    assert out.payload.radial.shape == (100,)


def test_multigeometry_warns_when_corrections_ignored(caplog):
    """The MG backend uses pyFAI's own corrections, NOT the shared CorrectionStack
    — run_stitch must warn so the caller/GUI knows the toggle was a no-op."""
    pytest.importorskip("pyFAI")
    import logging
    from xrd_tools.analysis.plans import StitchPlan, run_stitch
    from xrd_tools.corrections.stack import CorrectionStack
    diff, src = _stitch_diff_and_src()
    with caplog.at_level(logging.WARNING, logger="xrd_tools.analysis.plans"):
        run_stitch(StitchPlan(diffractometer=diff, backend="multigeometry",
                              mode="1d", npt_1d=100,
                              corrections=CorrectionStack(solid_angle=True)), src)
    assert any("IGNORED by the 'multigeometry'" in r.message for r in caplog.records)


def test_pyfai_hist_2d_dispatch_and_no_corrections():
    """2D pyfai_hist uses npt_rad_2d (NOT npt_1d) and corrections=None is unit-weight."""
    pytest.importorskip("pyFAI")
    from xrd_tools.analysis.plans import StitchPlan, run_stitch
    diff, src = _stitch_diff_and_src()
    out = run_stitch(StitchPlan(diffractometer=diff, backend="pyfai_hist", mode="2d",
                                npt_rad_2d=111, npt_azim_2d=40, npt_1d=999,
                                corrections=None), src)
    # crucially npt_1d=999 must NOT leak into the 2D radial axis
    assert out.payload.intensity.shape == (111, 40)
    assert np.isfinite(out.payload.intensity).any()
