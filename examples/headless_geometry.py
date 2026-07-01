"""Headless reference for the canonical ``Diffractometer`` (ADR-0007) — the
GUI's **section-2 "experimental config"** object, exercised with no Qt/GUI.

This script is the *section spec* for the three-section GUI layout
(``docs/design/design_gui_three_section_layout_jun2026.md``): a notebook/script
reads top-to-bottom in exactly the GUI's section order, and the cells map 1:1
onto the panes. The banners below mark the mapping so a reader sees that

  * the SECTION-2 cell IS the instrument record — ``Diffractometer`` (ADR-0007)
    + ``DetectorCalibration`` (PONI + Detector_config + mount) + ``GISettings``
    + beam (energy/wavelength). This is what is persisted once under
    ``/entry/diffractometer`` and restored on reload; and
  * the SECTION-3 cell IS the per-run plan (ranges / bins / axes / corrections),
    here only sketched — this module stays geometry-focused (no full reduction).

The point of the file is the SECTION-2 instrument and its **two derived adapter
views**, the one place the design insists the geometry lives once:

  * ``to_pyfai_per_frame(motors)`` → per-frame pyFAI rotations (rad) + the GI
    ``incident_angle`` (deg) — the writer/reduction view;
  * ``to_qconversion()`` → the xrayutilities ``QConversion`` — the RSM/q-space
    view;
  * ``assemble_circle_angles(diff, scan, indices)`` → the per-frame sample+
    detector angle list (the "one wiring task" RSM and the xu_hist stitch share).
    The circle ORDER + signs are carried in the preset's ``circle_motors``, NOT
    invented here.

It imports no Qt / pyqtgraph / xdart — it must run in an environment where
importing the GUI would fail::

    python examples/headless_geometry.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from xrd_tools.core.containers import PONI
from xrd_tools.core.geometry import (
    DetectorCalibration,
    Diffractometer,
    ImageOrientation,
    assemble_circle_angles,
)
from xrd_tools.corrections.grazing import GICorrectionStack, GISettings
from xrd_tools.integrate.multi import (
    create_multigeometry_integrators_from_geometry,
)


# A tiny synthetic 6-circle scan: per-frame motor positions (degrees).  The
# psic preset reads nu→rot1, del→rot2, eta→incidence; the other circles
# (mu/chi/phi) ride along in the angle assembly.
N_FRAMES = 4


def _synthetic_motors() -> dict[str, np.ndarray]:
    """Per-frame motor columns (degrees), keyed by motor name — what a Wrangler
    would extract from a SPEC/NeXus header (this is SECTION-1 'data')."""
    rng = np.random.default_rng(0)
    return {
        "mu": np.zeros(N_FRAMES),
        "eta": np.full(N_FRAMES, 0.5),                       # GI incidence
        "chi": rng.uniform(-1.0, 1.0, N_FRAMES),
        "phi": rng.uniform(0.0, 2.0, N_FRAMES),
        "nu": np.linspace(2.0, 8.0, N_FRAMES),               # → rot1
        "del": np.linspace(15.0, 45.0, N_FRAMES),            # → rot2
    }


def _synthetic_scan(motors: dict[str, np.ndarray]):
    """A minimal duck-typed scan: a ``scan_data`` DataFrame of per-frame motor
    columns + a ``frame_indices`` list.  This is the exact ``_Scan`` shape
    ``assemble_circle_angles`` consumes (see tests/core/test_circle_angles.py).
    """
    df = pd.DataFrame(motors)

    class _Scan:
        scan_data = df
        frame_indices = list(range(len(df)))

    return _Scan()


def main() -> int:
    motors = _synthetic_motors()

    # ===== SECTION 1: DATA ==================================================
    # The frame source / scan.  Here it is purely the per-frame motor table the
    # geometry consumes (no images needed for a geometry demo).  In the GUI this
    # is the Wrangler; loading data is what populates SECTION 2 below.
    scan = _synthetic_scan(motors)
    print(f"[section 1] data: {N_FRAMES} frames, motor columns "
          f"{sorted(scan.scan_data.columns)}")

    # ===== SECTION 2: EXPERIMENTAL CONFIG (THE INSTRUMENT) ==================
    # The persisted, round-trippable instrument record (/entry/diffractometer):
    #   2a Diffractometer  2b DetectorCalibration  2c GISettings  2d beam.

    # -- 2a. Diffractometer (ADR-0007): preset authors BOTH derived halves. ---
    diff = Diffractometer.psic()
    print(f"[section 2a] Diffractometer preset = {diff.preset!r}")
    print(f"             sample_circles = {diff.sample_circles}")
    print(f"             detector_circles = {diff.detector_circles}")
    print(f"             camera = {diff.camera}")
    # circle_motors carries the q-convention ORDER (sample circles, then
    # detector circles) — authored in the validated preset, never invented.
    print("             circle_motors (order = sample…, detector…) = "
          + ", ".join(m.source_motor for m in diff.circle_motors))
    print(f"             rot1←{diff.rot1.source_motor}  rot2←{diff.rot2.source_motor}"
          f"  incidence←{diff.incident_angle.source_motor}")

    # -- 2b. DetectorCalibration: a PONI + Detector_config + image mount. ------
    cal = DetectorCalibration(
        poni=PONI(dist=0.39, poni1=0.033, poni2=0.050, rot1=0.0027,
                  wavelength=7.7487e-11, detector="Pilatus300kw"),
        detector_config={"orientation": 3},
        image_orientation=ImageOrientation(rotation=180),
    )
    print(f"[section 2b] DetectorCalibration: dist={cal.poni.dist} m, "
          f"poni=({cal.poni.poni1}, {cal.poni.poni2}) m, "
          f"detector={cal.poni.detector!r}")
    print(f"             Detector_config={dict(cal.detector_config)}, "
          f"image rotation={cal.image_orientation.rotation}deg")

    # -- 2c. GISettings: the unified grazing-incidence object (section-2c). ----
    gi = GISettings(
        corrections=GICorrectionStack(material="Si", energy_eV=16000.0),
        incident_angle_deg=0.5,            # fixed αi; None → per-frame from diff
        sample_orientation=1,
    )
    print(f"[section 2c] GISettings: αi={gi.incident_angle_deg}deg, "
          f"sample_orientation={gi.sample_orientation}, "
          f"material={gi.corrections.material!r}")

    # -- 2d. Beam: energy / wavelength (the calibration is the q-source). ------
    energy_eV = 16000.0
    print(f"[section 2d] beam: energy={energy_eV} eV "
          f"(calibration wavelength={cal.poni.wavelength:.4e} m)")

    # ----- Section 2's TWO derived adapter views ---------------------------
    # The whole reason the instrument lives once: it yields both reduction and
    # q-space geometry, no duplicated wiring.

    # View 1 — pyFAI per-frame rotations (rad) + GI incidence (deg).
    pf = diff.to_pyfai_per_frame(motors)
    print("[section 2 → view 1] to_pyfai_per_frame(motors):")
    print(f"             rot1 (rad, ←nu)  shape={pf['rot1'].shape}  "
          f"sample={np.round(pf['rot1'][:2], 5).tolist()}")
    print(f"             rot2 (rad, ←del) shape={pf['rot2'].shape}  "
          f"sample={np.round(pf['rot2'][:2], 5).tolist()}")
    print(f"             rot3 (rad)       all-zero={bool(np.all(pf['rot3'] == 0))}")
    print(f"             incident_angle (deg, ←eta, NOT deg2rad) = "
          f"{pf['incident_angle'][:2].tolist()}")
    # rot* are radians; the GI incidence stays in degrees (design contract).
    assert np.allclose(pf["rot1"], np.deg2rad(motors["nu"]))
    assert np.allclose(pf["incident_angle"], motors["eta"])

    # View 2 — the xrayutilities QConversion (energy-free q-space geometry).
    qconv = diff.to_qconversion()
    print("[section 2 → view 2] to_qconversion():")
    print(f"             {type(qconv).__module__}.{type(qconv).__name__}: "
          f"{len(qconv.sampleAxis)} sample axes + "
          f"{len(qconv.detectorAxis)} detector axes")
    assert len(qconv.sampleAxis) == len(diff.sample_circles)
    assert len(qconv.detectorAxis) == len(diff.detector_circles)

    # assemble_circle_angles — the shared per-frame angle list (RSM + xu_hist
    # stitch).  ORDER is sample circles then detector circles, carried in the
    # preset's circle_motors — NOT invented here.
    angles_all = assemble_circle_angles(diff, scan)
    n_circ = len(diff.sample_circles) + len(diff.detector_circles)
    print("[section 2] assemble_circle_angles(diff, scan):")
    print(f"             {len(angles_all)} circle arrays (== {n_circ} circles), "
          f"each length {angles_all[0].shape[0]} (one per frame)")
    assert len(angles_all) == n_circ
    # the first circle is mu, the last detector circle is del — confirm the
    # preset-carried order drives the lookup (not any guess in this script).
    assert np.array_equal(angles_all[0], motors["mu"])      # 1st sample circle
    assert np.array_equal(angles_all[-1], motors["del"])    # last detector circle

    # index-selected assembly (a subset of frames, re-ordered) uses frame_indices
    sel = [3, 0, 2]
    angles_sel = assemble_circle_angles(diff, scan, indices=sel)
    print(f"             index-selected {sel}: last-circle (del) = "
          f"{angles_sel[-1].tolist()}")
    assert np.array_equal(angles_sel[-1], motors["del"][sel])

    # ===== SECTION 3: PROCESSING OPTIONS (THE PLAN — sketch only) ===========
    # The per-run choices: ranges / bins / axes / corrections.  This module is
    # geometry-focused, so we only NAME the section-3 fields a Plan would carry
    # (a full ReductionPlan/StitchPlan/RSMPlan run lives in the other examples).
    plan_sketch = {
        "ranges": {"radial_range": (0.5, 6.0), "azimuth_range": (-180.0, 180.0)},
        "bins": {"npt_1d": 1000, "npt_rad": 1000, "npt_azim": 360},
        "axes": "q_A^-1 (standard); q_ip/q_oop/χ_GI in GI mode",
        "corrections": "solid-angle + polarization; GI adds footprint/Fresnel/…",
    }
    print(f"[section 3] plan sketch (per-run): ranges={plan_sketch['ranges']}")
    print(f"             bins={plan_sketch['bins']}  axes={plan_sketch['axes']!r}")

    # ===== RUN / DISPLAY / PERSIST =========================================
    # Show the geometry FEEDS pyFAI: per-frame integrators built straight from
    # the instrument (SECTION 2) — the calibrated MultiGeometry path.  Per-frame
    #   rotN = base.poni.rotN + to_pyfai_per_frame()[rotN].
    integrators = create_multigeometry_integrators_from_geometry(
        diff, motors, base_calibration=cal)
    print("[run] create_multigeometry_integrators_from_geometry: "
          f"{len(integrators)} per-frame AzimuthalIntegrators")
    assert len(integrators) == N_FRAMES
    ai0, ai_last = integrators[0], integrators[-1]
    # frame 0: rot2 = base.poni.rot2 (0) + deg2rad(del[0]); confirm it tracks.
    print(f"      frame 0   rot2={ai0.rot2:.5f} rad (dist={ai0.dist} m)")
    print(f"      frame {N_FRAMES - 1}   rot2={ai_last.rot2:.5f} rad")
    assert np.isclose(ai0.rot2, cal.poni.rot2 + np.deg2rad(motors["del"][0]))
    assert np.isclose(ai_last.rot2,
                      cal.poni.rot2 + np.deg2rad(motors["del"][-1]))

    # Persist + round-trip the SECTION-2 instrument record (the /entry/
    # diffractometer blob): JSON is the round-trippable form the .nxs carries.
    with tempfile.TemporaryDirectory() as tmp:
        blob = Path(tmp) / "diffractometer.json"
        blob.write_text(diff.to_json())
        diff2 = Diffractometer.from_json(blob.read_text())
        cal2 = DetectorCalibration.from_json(cal.to_json())
        gi2 = GISettings.from_dict(gi.to_dict())
        print(f"[persist] wrote {blob.name} ({blob.stat().st_size} bytes); "
              "round-tripped Diffractometer + DetectorCalibration + GISettings")
        assert diff2 == diff
        assert cal2.poni == cal.poni
        assert cal2.image_orientation == cal.image_orientation
        assert gi2.incident_angle_deg == gi.incident_angle_deg

    # --- the whole point: no GUI on the import graph -----------------------
    gui_roots = {"xdart", "pyqtgraph"}
    leaked = sorted({m.split(".")[0] for m in sys.modules} & gui_roots)
    assert not leaked, f"geometry example pulled in the GUI stack: {leaked}"

    # ----- summary ---------------------------------------------------------
    print("-" * 70)
    print("SUMMARY — Diffractometer (ADR-0007) as the section-2 instrument:")
    print(f"  preset={diff.preset!r}: {len(diff.sample_circles)} sample + "
          f"{len(diff.detector_circles)} detector circles")
    print(f"  view 1 (pyFAI): rot1/rot2/rot3 (rad) + incidence (deg), "
          f"{N_FRAMES} frames")
    print(f"  view 2 (xu QConversion): {len(qconv.sampleAxis)}+"
          f"{len(qconv.detectorAxis)} axes")
    print(f"  assemble_circle_angles: {len(angles_all)} circles, order from "
          "circle_motors (not invented)")
    print(f"  fed pyFAI: {len(integrators)} per-frame integrators from the "
          "instrument + calibration")
    print("  round-trip: Diffractometer/DetectorCalibration/GISettings JSON ✓")
    print("OK: geometry section-2 reference ran without the xdart/pyqtgraph "
          "GUI stack.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
