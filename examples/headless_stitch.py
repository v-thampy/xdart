"""Headless stitching: ``StitchPlan`` / ``run_stitch`` + persistence + reload.

This is the GUI **three-section layout** (``docs/design/
design_gui_three_section_layout_jun2026.md``) expressed as a runnable, Qt-free
script.  A reader should be able to see, one-to-one, that:

  * **SECTION 1 (DATA)** — the frame source — is the synthetic
    ``MemoryFrameSource`` of ``ScanFrame`` objects built below;
  * **SECTION 2 (EXPERIMENTAL CONFIG / the INSTRUMENT)** — the detector
    calibration + diffractometer geometry — is the ``Diffractometer.psic()``
    carrying a ``DetectorCalibration(poni=PONI(...))`` cell;
  * **SECTION 3 (PROCESSING OPTIONS / the PLAN)** — ranges / bins / axes /
    backend / corrections — is the ``StitchPlan`` cell.

It then RUNS the plan (``run_stitch`` → an ``AnalysisResult`` whose ``.payload``
is an ``IntegrationResult1D``), PERSISTS it to a temp ``.nxs``
(``write_stitched`` + ``StitchPlan.provenance()``), READS it back
(``read_stitched``), and proves the section-3 RELOAD path
(``StitchPlan.from_provenance``) reconstructs the processing options.

No external data files (synthetic ring frames are generated in-script) and no Qt
/ xdart GUI imports — it must run in a headless environment::

    python examples/headless_stitch.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from xrd_tools.analysis.plans import StitchPlan, run_stitch
from xrd_tools.core.containers import PONI
from xrd_tools.core.geometry import DetectorCalibration, Diffractometer
from xrd_tools.core.scan import ScanFrame
from xrd_tools.corrections.grazing import GICorrectionStack, GISettings
from xrd_tools.corrections.stack import CorrectionStack
from xrd_tools.io.nexus import read_stitched, write_stitched
from xrd_tools.sources import MemoryFrameSource

# A small Pilatus-100k-shaped detector for the whole demo.
SHAPE = (195, 487)
N_FRAMES = 4
NPT_1D = 250


# ===== SECTION 1: DATA =====================================================
# The frame source.  In the GUI this is whatever feeds frames (an image series,
# a SPEC scan, a NeXus file); here we synthesize a few ring images so the script
# is self-contained.  Each ScanFrame carries the per-frame goniometer motors in
# its metadata — del/nu vary across the scan, eta (the psic incidence) is a real
# but constant motor.  The plan's geometry path reads these motors by name.
def _ring(seed: int) -> np.ndarray:
    """A single Debye-Scherrer-like ring image (copied from the stitch tests)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[: SHAPE[0], : SHAPE[1]]
    r = np.sqrt((y - SHAPE[0] / 2) ** 2 + (x - SHAPE[1] / 2) ** 2)
    return (
        500.0 * np.exp(-((r - 60.0) / 10.0) ** 2)
        + rng.poisson(3, size=SHAPE)
    ).astype(float)


def _data_source() -> MemoryFrameSource:
    frames = [
        ScanFrame(
            i,
            image=_ring(i),
            metadata={"nu": float(i), "del": float(5 * i), "eta": 0.3},
        )
        for i in range(N_FRAMES)
    ]
    return MemoryFrameSource(frames, name="stitch_demo")


# ===== SECTION 2: EXPERIMENTAL CONFIG ======================================
# The INSTRUMENT: detector calibration (PONI) + the diffractometer convention.
# `Diffractometer.psic()` defines which rotation circles exist and how they map
# onto the motor names in Section 1 (psic: rot1<-nu, rot2<-del, incidence<-eta).
# Wrapping the PONI in a DetectorCalibration is what lets the histogram backend
# build a calibrated per-frame integrator instead of a hardwired deg2rad guess.
# This whole object is what persists under /entry/diffractometer in the .nxs.
def _instrument() -> Diffractometer:
    base = PONI(
        dist=0.2,
        poni1=SHAPE[0] * 172e-6 / 2,
        poni2=SHAPE[1] * 172e-6 / 2,
        rot1=0.0,
        rot2=0.0,
        rot3=0.0,
        wavelength=1.0e-10,
        detector="Pilatus100k",
    )
    psic = Diffractometer.psic()
    return Diffractometer(
        preset="psic",
        rot1=psic.rot1,
        rot2=psic.rot2,
        incident_angle=psic.incident_angle,  # active <- eta (only used in GI mode)
        calibration=DetectorCalibration(poni=base),
    )


# ===== SECTION 3: PROCESSING OPTIONS =======================================
# The PLAN: backend + ranges/bins/axes/corrections.  Two interchangeable merge
# backends share the same geometry (Section 2) and data (Section 1):
#
#   * backend="multigeometry" — pyFAI's MultiGeometry.  It applies pyFAI's OWN
#     solid-angle / polarization corrections internally, so the shared
#     CorrectionStack is intentionally NOT passed here (run_stitch warns if it
#     were — the toggle would be a silent no-op).
#
#   * backend="pyfai_hist" — the streaming per-pixel histogram merge.  Here the
#     corrections ARE the shared CorrectionStack (solid_angle / polarization),
#     applied as per-pixel weights before the histogram.  This is the backend
#     the GI path also rides on.
def _plan_multigeometry(diff: Diffractometer) -> StitchPlan:
    return StitchPlan(
        diffractometer=diff,
        backend="multigeometry",
        mode="1d",
        npt_1d=NPT_1D,
        unit="q_A^-1",
        radial_range=None,  # auto from the data
    )


def _plan_pyfai_hist(diff: Diffractometer) -> StitchPlan:
    return StitchPlan(
        diffractometer=diff,
        backend="pyfai_hist",
        mode="1d",
        npt_1d=NPT_1D,
        unit="q_A^-1",            # pyfai_hist emits |q| in Å^-1 only
        radial_range=None,
        corrections=CorrectionStack(solid_angle=True, polarization_factor=0.99),
    )


def _plan_gi_sketch(diff: Diffractometer) -> StitchPlan:
    """Section-3 GI sketch (built but NOT run here, to keep the demo simple).

    A grazing-incidence stitch is just the pyfai_hist plan plus a GISettings
    cell — the same Section-1/2 inputs, with footprint/refraction toggles and an
    incidence angle (taken from the eta motor, or overridden here)."""
    return StitchPlan(
        diffractometer=diff,
        backend="pyfai_hist",
        mode="1d",
        npt_1d=NPT_1D,
        unit="q_A^-1",
        gi=GISettings(
            corrections=GICorrectionStack(
                material="Si", energy_eV=12398.0, footprint=True, refraction=False
            ),
            incident_angle_deg=0.3,
            sample_orientation=1,
        ),
    )


def main() -> int:
    # --- Section 1 + 2 + 3: assemble the three cells -----------------------
    src = _data_source()
    diff = _instrument()
    plan_mg = _plan_multigeometry(diff)
    plan_hist = _plan_pyfai_hist(diff)
    _ = _plan_gi_sketch(diff)  # built to show the GI shape; not run below
    print(
        f"SECTION 1 data:   {N_FRAMES} synthetic {SHAPE} ring frames "
        f"(motors nu/del vary, eta=0.3)"
    )
    print(
        f"SECTION 2 config: Diffractometer.psic() + "
        f"DetectorCalibration(PONI dist={diff.calibration.poni.dist} m, "
        f"Pilatus100k)"
    )
    print(
        f"SECTION 3 plan:   npt_1d={plan_hist.npt_1d} unit={plan_hist.unit!r} "
        f"backends=[multigeometry, pyfai_hist]"
    )

    # --- RUN: both backends over the same data/instrument ------------------
    res_mg = run_stitch(plan_mg, src)
    res_hist = run_stitch(plan_hist, src)
    stitched_mg = res_mg.payload      # IntegrationResult1D
    stitched_hist = res_hist.payload  # IntegrationResult1D

    assert stitched_hist.radial.shape == (NPT_1D,), stitched_hist.radial.shape
    # The two backends merge the same rings to the same radial axis; their
    # intensities agree up to an absolute-vs-normalized solid-angle scale.
    m = (
        np.isfinite(stitched_hist.intensity)
        & (stitched_mg.intensity > 0)
        & (stitched_hist.intensity > 0)
    )
    scale = np.nanmedian(stitched_mg.intensity[m] / stitched_hist.intensity[m])
    rel = np.abs(stitched_hist.intensity[m] * scale - stitched_mg.intensity[m]) / np.maximum(
        np.abs(stitched_mg.intensity[m]), 1e-9
    )
    assert np.nanmedian(rel) < 0.05, np.nanmedian(rel)
    print(
        f"RUN: multigeometry I[peak]={np.nanmax(stitched_mg.intensity):.1f}, "
        f"pyfai_hist I[peak]={np.nanmax(stitched_hist.intensity):.3g} "
        f"(agree to {np.nanmedian(rel) * 100:.2f}% after scale)"
    )

    with tempfile.TemporaryDirectory() as tmp:
        nxs = Path(tmp) / "stitched_demo.nxs"

        # --- PERSIST: write the pyfai_hist result + its plan provenance ----
        import h5py

        prov = plan_hist.provenance()
        with h5py.File(nxs, "w") as f:
            entry = f.create_group("entry")
            write_stitched(entry, stitched_1d=stitched_hist, provenance=prov)

        # --- READ BACK: intensity + provenance round-trip ------------------
        ds = read_stitched(nxs)
        np.testing.assert_allclose(
            ds["stitched_1d"].values, stitched_hist.intensity, rtol=1e-4
        )
        read_prov = ds.attrs["stitched_1d_provenance"]
        assert read_prov["backend"] == "pyfai_hist", read_prov["backend"]
        assert read_prov["npt_1d"] == NPT_1D, read_prov["npt_1d"]
        assert read_prov["corrections"]["polarization_factor"] == 0.99
        print(
            f"PERSIST -> {nxs.name}: read_stitched 1d shape="
            f"{tuple(ds['stitched_1d'].shape)}, intensity round-trips; "
            f"provenance backend={read_prov['backend']!r} present"
        )

    # --- RELOAD ROUND-TRIP: rebuild the Section-3 plan from provenance ------
    # This is the GUI's section-3 reload path: the processing options come back
    # from the persisted provenance dict (geometry/mask are reattached, not in
    # provenance — they live under /entry/diffractometer).
    reloaded = StitchPlan.from_provenance(plan_hist.provenance())
    assert reloaded.backend == plan_hist.backend
    assert reloaded.npt_1d == plan_hist.npt_1d
    assert reloaded.unit == plan_hist.unit
    assert reloaded.corrections.solid_angle == plan_hist.corrections.solid_angle
    assert (
        reloaded.corrections.polarization_factor
        == plan_hist.corrections.polarization_factor
    )
    assert reloaded.diffractometer is None  # geometry persists separately
    print(
        f"RELOAD: StitchPlan.from_provenance -> backend={reloaded.backend!r} "
        f"npt_1d={reloaded.npt_1d} unit={reloaded.unit!r} "
        f"pol={reloaded.corrections.polarization_factor} -> provenance round-trip OK"
    )

    # --- the whole point: no Qt / xdart GUI stack on the import graph -------
    leaked = sorted({m.split(".")[0] for m in sys.modules} & {"xdart", "pyqtgraph"})
    assert not leaked, f"headless example pulled in the xdart GUI stack: {leaked}"
    print(
        "OK: stitch (multigeometry + pyfai_hist) -> persist -> read-back -> "
        "from_provenance ran headless (no xdart/pyqtgraph)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
