"""Headless RSM (reciprocal-space map) example — the GUI three-section layout,
expressed as a runnable headless script.

This is the **RSM section spec** for ``docs/design/design_gui_three_section_layout_jun2026.md``.
A reader should be able to see that the GUI's three sections ARE the three
objects this script builds:

  * SECTION 1 (Data)               = the frame source / synthetic image stack +
                                     the per-frame diffractometer angle arrays.
  * SECTION 2 (Experimental config)= the INSTRUMENT: a ``PixelQMap`` built from a
                                     ``Diffractometer`` (the 2a circle stack) +
                                     a ``DetectorHeader`` (the 2b xu mm geometry),
                                     plus the 2c GISettings and 2d beam energy.
  * SECTION 3 (Processing options) = the PLAN: an ``RSMPlan`` carrying the
                                     bins / q_bounds / diff_motors / corrections
                                     (CorrectionStack) — the section-3 fields.

It then RUNs the grid (Σ(raw·w)/Σ(w) accumulator over real ``xrayutilities``),
PERSISTs the volume to a temp ``.nxs`` with the plan provenance, READS it back,
and round-trips the plan through ``RSMPlan.from_provenance`` — the reload half of
the section-3 contract.

No Qt / GUI imports.  Synthetic data is generated in-script (no external files);
the only file written is a tempfile ``.nxs`` that is cleaned up on exit.  Run it
in an environment where importing ``xdart`` or Qt would fail — it must still
pass::

    python examples/headless_rsm.py

Requires ``xrayutilities`` + ``pyFAI`` (the real geometry + the correction
weight); both are part of the analysis env.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

from xrd_tools.analysis.plans import RSMPlan
from xrd_tools.corrections.stack import CorrectionStack
from xrd_tools.core.geometry import (
    DetectorHeader,
    Diffractometer,
    DiffractometerConfig,
    PixelQMap,
)
from xrd_tools.io.nexus import read_rsm, write_rsm
from xrd_tools.rsm.corrections import rsm_correction_weight
from xrd_tools.rsm.gridding import grid_img_data_streaming


def main() -> int:
    # ========================================================================
    # ===== SECTION 1: DATA ==================================================
    # ========================================================================
    # The frame source.  In the GUI this is the Wrangler; here we synthesise a
    # small (N, H, W) image stack plus the per-frame diffractometer angle
    # arrays (one array per circle, length N) that the loaded SPEC/NeXus header
    # would supply as ``scan_data`` columns.
    N_FRAMES = 8
    NCH1, NCH2 = 32, 36              # detector rows x cols (H x W)
    rng = np.random.default_rng(0)

    # A synthetic ring-ish stack with a slow per-frame drift — stands in for a
    # real detector movie; the values only have to be finite + non-degenerate.
    yy, xx = np.mgrid[:NCH1, :NCH2]
    r = np.sqrt((yy - NCH1 / 2) ** 2 + (xx - NCH2 / 2) ** 2)
    img = np.stack([
        500.0 * np.exp(-((r - 9.0) / 4.0) ** 2)
        + 3.0 * i
        + rng.poisson(2.0, size=(NCH1, NCH2))
        for i in range(N_FRAMES)
    ]).astype(float)

    # Per-frame circle angles (deg).  The default DiffractometerConfig has
    # 3 sample circles + 1 detector circle => 4 motor arrays, named below in
    # SECTION 3's ``diff_motors``.  Two motors sweep; the other two are fixed.
    angles = [
        np.linspace(0.0, 3.0, N_FRAMES),    # sample circle 1  (sweeps)
        np.linspace(0.0, 1.0, N_FRAMES),    # sample circle 2  (sweeps)
        np.zeros(N_FRAMES),                 # sample circle 3  (fixed)
        np.linspace(2.0, 8.0, N_FRAMES),    # detector circle  (sweeps)
    ]
    print(f"[SECTION 1: DATA] frame stack shape={img.shape} "
          f"({N_FRAMES} frames, {NCH1}x{NCH2}); "
          f"{len(angles)} per-frame angle arrays")

    # ========================================================================
    # ===== SECTION 2: EXPERIMENTAL CONFIG (the instrument) ==================
    # ========================================================================
    # 2a. Diffractometer config — the circle stack.  We build the canonical
    #     Diffractometer from a DiffractometerConfig (byte-equal drop-in); its
    #     sample/detector circle directions ARE the goniometer convention.
    diff = Diffractometer.from_diffractometer_config(DiffractometerConfig())

    # 2b. Detector config — the xu mm geometry (beam centre + pixel size +
    #     sample-detector distance + panel shape).  This is the DetectorHeader
    #     half of section 2b (the PONI/DetectorCalibration is its pyFAI twin).
    header = DetectorHeader(
        cch1=NCH1 / 2.0, cch2=NCH2 / 2.0,    # beam centre (pixels)
        pwidth1=0.075, pwidth2=0.075,        # pixel size (mm)
        distance=830.0,                      # sample->detector (mm)
        Nch1=NCH1, Nch2=NCH2,
    )

    # The section-2 RSM object: the PixelQMap bridges 2a (Diffractometer) and
    # 2b (DetectorHeader) into per-pixel reciprocal-space coordinates.
    mapper = PixelQMap(diff_config=diff, header=header)

    # 2d. Beam — energy (eV).  (The polarization *plane* would live here too;
    #     the polarization *factor* is a section-3 correction.)
    energy_eV = 10000.0

    print(f"[SECTION 2: CONFIG] detector={header.Nch1}x{header.Nch2} px, "
          f"pixel={header.pwidth1}x{header.pwidth2} mm, dist={header.distance} mm, "
          f"beam-centre=({header.cch1}, {header.cch2}) px")
    print(f"[SECTION 2: CONFIG] convention: preset={diff.preset!r}, "
          f"sample_circles={diff.sample_circles}, "
          f"detector_circles={diff.detector_circles}; beam E={energy_eV:.0f} eV")

    # ========================================================================
    # ===== SECTION 3: PROCESSING OPTIONS (the plan) =========================
    # ========================================================================
    # The RSMPlan IS the section-3 field set: Ranges (q_bounds, or auto-scout),
    # Bins (the 3D grid), Axes (hkl / qx,qy,qz), Corrections (CorrectionStack),
    # plus the diff_motors wiring (which scan_data column drives each circle).
    DIFF_MOTORS = ("phi", "chi", "eta", "tth")   # one name per circle (4)
    BINS = (24, 24, 24)
    CHUNK_SIZE = 3

    # Section-3 "Corrections" group — the per-pixel CorrectionStack folded into
    # the Σ(raw·w)/Σ(w) grid as the SAME weight stitching uses.
    corrections = CorrectionStack(solid_angle=True)

    # Compute the grid bounds up front from the real per-pixel q (the GUI's
    # "auto-scout" alternative to typing q_bounds by hand).
    qx, qy, qz = mapper.pixel_q(
        angles, energy_eV, UB=np.eye(3), image_shape=img.shape)
    q_bounds = (
        (float(qx.min()), float(qx.max())),
        (float(qy.min()), float(qy.max())),
        (float(qz.min()), float(qz.max())),
    )

    # The section-3 spec object.  We attach the mapper (section-2 geometry) so
    # ``plan.provenance()`` can note the diffractometer preset; the grid below
    # is driven by grid_img_data_streaming (the most reliable runnable path),
    # but the plan IS the full section-3 record that persists + reloads.
    plan = RSMPlan(
        mapper=mapper,
        diff_motors=DIFF_MOTORS,
        bins=BINS,
        UB=np.eye(3),
        energy=energy_eV,
        chunk_size=CHUNK_SIZE,
        q_bounds=q_bounds,
        corrections=corrections,
    )
    print(f"[SECTION 3: PLAN]   RSMPlan bins={plan.bins} "
          f"diff_motors={plan.diff_motors} chunk_size={plan.chunk_size}")
    print(f"[SECTION 3: PLAN]   q_bounds="
          f"{[(round(lo, 3), round(hi, 3)) for lo, hi in plan.q_bounds]}, "
          f"corrections.solid_angle={plan.corrections.solid_angle}")

    # ========================================================================
    # ===== RUN ==============================================================
    # ========================================================================
    # The most reliable runnable path: drive the streaming gridder directly
    # with the section-2 mapper + section-3 fields.  The corrections weight is
    # the SAME per-pixel CorrectionStack the GUI binds in section 3.  (Driving
    # run_rsm(plan, source) instead would need a source that also exposes a
    # scan_data motor table + energy; the gridding function is the minimal,
    # proven path — see tests/core/test_rsm_equivalence.py.)
    weight = rsm_correction_weight(header, plan.corrections)
    volume = grid_img_data_streaming(
        mapper, img, angles, energy=plan.energy,
        UB=plan.UB, bins=plan.bins, chunk_size=plan.chunk_size,
        q_bounds=plan.q_bounds, weight=weight,
    )
    assert volume.shape == BINS, volume.shape
    n_filled = int(np.isfinite(volume.intensity).sum())
    print(f"[RUN]     gridded RSMVolume shape={volume.shape} "
          f"({n_filled}/{volume.intensity.size} voxels filled); "
          f"weight={'on' if weight is not None else 'off'} "
          f"(solid-angle CorrectionStack)")

    # ========================================================================
    # ===== DISPLAY =========================================================
    # ========================================================================
    # The display layer would slice the volume; here we just summarise an
    # H-K-L projection so the run has a sane, inspectable output.
    finite = np.isfinite(volume.intensity)
    proj = np.nansum(np.where(finite, volume.intensity, 0.0), axis=(1, 2))
    print(f"[DISPLAY] H-axis projection peak at h={volume.h[int(np.argmax(proj))]:.4f}, "
          f"I_total={float(np.nansum(volume.intensity[finite])):.1f}")

    # ========================================================================
    # ===== PERSIST + READ-BACK + RELOAD ROUND-TRIP =========================
    # ========================================================================
    with tempfile.TemporaryDirectory() as tmp:
        nxs = Path(tmp) / "headless_rsm.nxs"

        # Persist: /entry/rsm = the volume + the plan provenance blob.
        with h5py.File(nxs, "w") as f:
            entry = f.create_group("entry")
            write_rsm(entry, volume, provenance=plan.provenance())

        # Read the volume back through the public API.
        reloaded = read_rsm(nxs)
        assert reloaded.shape == volume.shape, reloaded.shape
        np.testing.assert_allclose(reloaded.intensity, volume.intensity,
                                   rtol=1e-5, atol=1e-5, equal_nan=True)
        assert reloaded.provenance is not None, "provenance blob missing on read-back"
        print(f"[PERSIST] wrote + read /entry/rsm: shape round-trips "
              f"{reloaded.shape}; provenance present "
              f"(kind={reloaded.provenance.get('kind')!r})")

        # Reload round-trip: rebuild the section-3 plan from its provenance and
        # confirm the load-bearing fields match (geometry is reattached, not
        # serialized — pass the mapper back in).
        back = RSMPlan.from_provenance(reloaded.provenance, mapper=mapper)
        assert back.bins == plan.bins, (back.bins, plan.bins)
        assert back.q_bounds == plan.q_bounds, (back.q_bounds, plan.q_bounds)
        assert back.diff_motors == plan.diff_motors, (back.diff_motors, plan.diff_motors)
        assert back.energy == plan.energy
        assert back.corrections is not None and back.corrections.solid_angle is True
        assert back.mapper is mapper          # reattached, not from provenance
        print(f"[RELOAD]  provenance round-trip OK: bins/q_bounds/diff_motors match, "
              f"mapper reattached")

    # --- the whole point: no xdart GUI app on the import graph --------------
    # pyFAI may transitively import a Qt *binding* for its optional calibration
    # GUI — a pyFAI packaging detail, not us pulling in a GUI — so the
    # meaningful check is that the xdart app + pyqtgraph (the stack we own)
    # never load.
    gui_roots = {"xdart", "pyqtgraph"}
    leaked = sorted({m.split(".")[0] for m in sys.modules} & gui_roots)
    assert not leaked, f"headless example pulled in the xdart GUI stack: {leaked}"

    print("OK: RSM data -> config -> plan -> grid -> persist -> reload ran "
          "headlessly (no xdart/pyqtgraph GUI stack).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
