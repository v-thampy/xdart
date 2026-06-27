# GUI design — the three-section layout (data / experimental config / processing)

**Status:** design / brainstorm captured Jun 2026, for the stitching + RSM GUI wiring (P7).
Refined by the take-stock review (`wbalkxzey`). Applies first to the **new** Stitch/RSM tools;
the existing Int-1D/2D layout is **not** refactored now (this becomes its eventual template).

## The principle: GUI sections mirror the headless data model

The headless layer already separates concerns cleanly, so the GUI sections should fall out of it
rather than be invented:

| GUI section | Headless object(s) | Lifetime |
|---|---|---|
| **1. Data** | the frame source (Wrangler) — Image-Series / SPEC / NeXus / Tiled | per-run input |
| **2. Experimental config** | `Diffractometer` (ADR-0007) + `DetectorCalibration` + `GISettings` + beam | **the instrument** — persisted once, round-trippable (`/entry/diffractometer`) |
| **3. Processing options** | the **Plan** (`ReductionPlan` / `StitchPlan` / `RSMPlan`) | per-run choices |
| *(Tools)* | actions that **produce section-2 state** (Calibrate→`DetectorCalibration`, Refine→`Diffractometer`, Make-mask→detector mask) | actions |
| *(Controls)* | mode select + Start/Pause/Resume/Stop | run control |

The key seam is **section 2 vs section 3 = the instrument vs the plan** — i.e. the geometry you
configure/persist once vs the reduction choices you make per run. This is exactly the headless
split between the `Diffractometer`+`DetectorCalibration` (written to the `.nxs` once) and the
`*Plan` (provenance of a single run). When the GUI mirrors the data model, load/save and
headless≡GUI equivalence come for free.

---

## Section 1 — Data

The Wrangler / frame source (unchanged in spirit). One addition for the new tools: a **multi-scan
selector** (Stitch and multi-scan RSM both consume a *list* of sources). For the viewers, keep the
wrangler controls minimal (Project Folder + Save Path) per [[viewer_mode_features_jun2026]].

**Loading data is what populates section 2** (see below) — so section 1 and section 2 are coupled:
choosing data triggers inference/auto-population of the experimental config.

---

## Section 2 — Experimental config (the instrument)

Four sub-groups, each backed by a headless object:

- **2a. Diffractometer config** — `Diffractometer` (`preset`: psic / fourc / sixc / twoc / custom;
  the circle stack `sample_circles`/`detector_circles`; `circle_motors` = which motor drives each
  circle). The preset dropdown *is* `Diffractometer.psic()` etc.
- **2b. Detector config** — `DetectorCalibration` (PONI: dist/poni/rot + wavelength) +
  `Detector_config` orientation + the image mount + the detector **mask**. The detector type/shape.
- **2c. Sample & measurement** — `GISettings` (the unified GI object, see below) + UB matrix +
  measurement mode (standard / **grazing incidence** / transmission) + the sample material + the
  incidence-angle source. (Broadened from the original "GI/transmission" — sample mounting lives
  here too.)
- **2d. Beam** — energy / wavelength, and the **polarization plane**. (Distinct from the
  polarization *factor*, which is a section-3 correction — same word, two homes.)

### Field provenance — section 2 is *derived*, not typed

This is the load-bearing insight: most section-2 fields are **populated from the loaded data**, not
hand-entered. Each field has a source, and the GUI should show which:

| Field | Source | How |
|---|---|---|
| Detector geometry (2b) | **loaded** | a **PONI file** → `DetectorCalibration` (`load_poni`) |
| Detector type/shape (2b) | **inferred** | from the image shape / detector name |
| Diffractometer motors (2a) | **inferred** | a **SPEC file header** → motor names → `circle_motors` wiring + the available scan/incidence axes |
| Energy/wavelength (2d) | **inferred / loaded** | SPEC/NeXus header, or the PONI wavelength |
| UB matrix (2c) | **loaded** | from the SPEC/data (or the `.nxs` `ub_matrix` capability) |
| Preset (2a) | **user / inferred** | chosen, or inferred from the motor set (e.g. {nu,del}⊆columns → psic) |
| GI material / mode (2c) | **user** | typed (the sample is the experiment) |
| **Everything, on reload** | **restored** | a v2 `.nxs` → the `/entry/diffractometer` blob + `DetectorCalibration` + `scan_data` repopulate *all* of section 2 |

So section 2 has four field states the UI should distinguish: **auto-inferred** / **loaded-from-file**
/ **user-set** / **restored-from-nxs**. Treating section 2 as "the editable view of the persisted
instrument record" makes the **reload/persistence symmetry** fall out: section 2 is both an *input*
(configure before run) and an *output* (restored from the `.nxs`). This is what makes the
"headless ≡ reload ≡ live" equivalence hold at the GUI level.

### Actions feed section 2

Calibrate / Refine / Make-mask stay as **action buttons** (currently Tools), but their **results
live in section 2**, not in a transient dialog:
- **Calibrate…** → writes `DetectorCalibration` (2b); 2b shows the current calibration as a
  read-only summary + the button.
- **Refine geometry…** → the control-point `refine_goniometer` fit → updates the `Diffractometer`
  (2a/2b). Surfaces the fitted scales/offsets back into 2a.
- **Make mask…** → the detector mask (2b).

---

## Section 3 — Processing options (the plan)

Backed by the mode's `*Plan`. Four groups:

- **Ranges** — radial/azimuth range (Int), `radial_range`/`azimuth_range` (Stitch), `q_bounds`
  (RSM, or auto-scout).
- **Bins / points** — `npt` (Int/Stitch 1D), `npt_rad`/`npt_azim` (2D), `bins` (RSM 3D).
- **Axes** — the output coordinate: q / 2θ / r / χ (standard), q_ip / q_oop / exit-angle / χ_GI
  (GI), hkl / qx,qy,qz (RSM). For Stitch: the **merge backend** (`multigeometry` / `pyfai_hist` /
  `xu_hist`) lives here too.
- **Corrections** — `CorrectionStack` (solid-angle, polarization factor, air absorption) and, in GI
  mode, the `GICorrectionStack` toggles (footprint / Fresnel / absorption / refraction).

### Section 3 is *reactive* to section 2c — the key interaction

The available **axes** and **corrections** are **driven by the measurement mode (2c)**:
- standard → axes {q, 2θ, r, χ}; corrections {solid-angle, polarization}.
- GI → axes {q, q_ip, q_oop, exit-angle, χ_GI}; corrections **add** {footprint, Fresnel,
  absorption, refraction}.
- RSM → axes {hkl, qx/qy/qz}; the 3D grid; corrections as above.

So section 3 re-renders when 2c changes. And a section-3 correction can **read section-2 state**:
the `GICorrectionStack` needs the sample **material** + **energy** (both section 2). Design the
corrections panel to pull material/energy from section 2 rather than duplicate them.

---

## Contract fix before wiring — unify the GI knobs (`GISettings`)

Today the GI config is **duplicated** as loose fields on both `StitchPlan` and `RSMPlan`
(`gi: GICorrectionStack`, `gi_incident_angle_deg`, `gi_sample_orientation`, `gi_tilt_deg`). Before
the GUI wires to them, consolidate into **one shared object** — call it `GISettings` — bundling the
corrections + the fiber sample geometry:

```python
@dataclass(frozen=True)
class GISettings:
    corrections: GICorrectionStack | None = None   # footprint/Fresnel/absorption/refraction
    incident_angle_deg: float | None = None        # fixed αi (else from the Diffractometer)
    sample_orientation: int = 1                     # pyFAI fiber EXIF orientation
    tilt_deg: float = 0.0
```

Then `StitchPlan.gi: GISettings | None` and `RSMPlan.gi: GISettings | None`. This (a) removes the
duplication, (b) gives section 2c **one object to bind to**, and (c) keeps the reduction-side
`GIMode` (which carries reduction-output baggage: `mode_1d/2d`, `method`, `npt_oop`,
`incidence_motor`) separate for now — a future convergence of `GIMode`↔`GISettings` is noted but
not in scope. (See the take-stock review for the final recommended shape.)

---

## Generalization to Int-1D/2D — a MODERATE migration, noted as a direction (not now)

The 3-section structure generalizes: Int / Stitch / RSM / Fitting all fit (1) data, (2) instrument,
(3) plan. The new Stitch/RSM tools adopt this layout as the **reference**; Int migrates later. A
concrete migration assessment (read the current widgets, `aad2247d`) sized it as **MODERATE
(~6.5–13 h)** — *not straightforward* (so not folded in now), but a re-organisation of already-
separated pieces, not a redesign.

**What's already separable (the good news):** the wrangler and the integrator own **separate**
pyqtgraph param trees (no single mixed tree); the **1D and 2D blocks are fully independent**
(`frame1D`/`frame2D`, `bai_1d_args`/`bai_2d_args`, separate handlers) so the re-slice applies to
each identically; Tools (`toolsFrame`) + `StaticControls` are already the mode-agnostic panes
(`staticUI.py`). Only the two middle panes (`wranglerFrame` + `integratorFrame`) get re-sliced.

**What makes it MODERATE (the entanglement at the section-2↔3 boundary):**
- **`set_image_units()`** (`integrator.py`, the GI↔standard reactivity) straddles the boundary: a
  section-2c GI toggle must keep re-rendering the section-3 axis/range combos (force unit=Q, disable
  `chi_offset`, swap to GI axes). Exactly **one** signal crosses today —
  `integrator.sigUpdateGI → staticWidget.update_scattering_geometry → scan.gi → set_image_units()` —
  and it must stay wired across the new container boundary (the same state→render flow §3 above).
- **GI geometry is integrator-owned but conceptually section-2** (`gi_enable`/`gi_motor`/
  `gi_sample_orientation`/`gi_tilt` live in the integrator; the wrangler keeps a hidden carrier for
  persistence). Extracting them to section-2 means re-homing those widgets + their session keys.
- **PONI relocation → progressive disclosure**: moving the PONI param out of the wrangler into
  section-2b means section-3 visibility should gate on `scan.poni is not None`.

**Migration phases (when there's a reason to do it):** (1) container re-slice — reparent
`frame1D`/`frame2D` into a section-3 frame, the GI frame into section-2; (2) extract the GI-geometry
widgets to section-2, keep `sigUpdateGI` routing intact; (3) move the PONI param + gate section-3 on
PONI-loaded; (4) verify `set_image_units()` still fires on the section-2 GI toggle; (5) session
backward-compat (old keys fallback). **Signals that must survive:** `sigUpdateGI`, the range-default
repopulation, the advanced-corrections `sigTreeStateChanged`, the reintegrate dispatch, and
`get_gi_config()` reading from the new location. **Main risk:** the `set_image_units()`
disconnect/reconnect + the PONI progressive-disclosure gate.

**Verdict:** worth doing once the Stitch/RSM 3-section layout is proven (so Int inherits a validated
pattern), but it's real GUI work — **not bundled with the new-tool wiring**.

---

## The notebook is the GUI spec

The headless example notebooks should be refreshed against the current API (Diffractometer /
CorrectionStack / `*Plan` / `assemble_circle_angles` / the two-gridder accumulator / the
persistence). Beyond removing stale APIs, the refresh is a **design forcing-function**: a notebook
reads in exactly the section order —

```
load data            # section 1
→ configure the instrument (diffractometer / detector / sample+measurement / beam)   # section 2
→ build a Plan (ranges / bins / axes / corrections)                                  # section 3
→ run → display → persist
```

If the updated notebook reads cleanly top-to-bottom, the sections are validated; where a step feels
awkward in the notebook, that section will feel awkward in the GUI. So **refresh the notebooks as
step 1 of the GUI design**, not a chore after it.

---

## Open questions (for Vivek)

- **Energy single-source (decision needed).** The take-stock found energy triple-sourced
  (`RSMPlan.energy` / `GICorrectionStack.energy_eV` / calibration wavelength). Recommend: the
  **calibration wavelength** is the source of truth (it persists under `/entry/diffractometer`);
  `RSMPlan.energy` + `GICorrectionStack.energy_eV` *derive/validate* against it. Confirm, then it's
  a small headless fix.
- **GISettings self-contained vs section-2-global energy/material?** `GISettings` is built (new
  object, separate from `GIMode` — resolved). Open: should it carry its own energy/material, or
  read them from section-2-global (since the non-GI `CorrectionStack` + q-conversion need energy
  too)? Lean: section-2-global, decide during the notebook refresh.
- Read-only-after-load vs editable section 2 (e.g. override an auto-inferred motor mapping?).
- Multi-scan section-1 selector UX for Stitch / multi-scan RSM.

## Take-stock review refinements (`wbalkxzey`, Jun 2026)

A multi-agent take-stock confirmed the 3-section split is "a presentation re-grouping
over an already-separated core" — low-risk. Refinements + corrections:

**Already done (don't redo):** the GI-knob consolidation into `GISettings` (`grazing.py`);
`stitched_1d/2d` + `rsm` ARE schema-registered + capability-gated (P5/P6.7); `pyfai_q_frames`
computes the correction weight **per-frame** (the stitch is not geometry-static — only the
RSM weight is, by design).

**The real pre-GUI contract fixes (small, headless, do first):**
1. **Energy single-source (HIGH).** Energy enters three ways — `RSMPlan.energy`,
   `GICorrectionStack.energy_eV` (eV), and the calibration **wavelength** (`/entry/diffractometer`).
   Pick one source of truth (the calibration wavelength — it persists), and make the others
   *derive/validate* against it, before a section-2 energy widget binds to it. **Decision needed**
   (see Open questions).
2. **`StitchPlan.from_provenance` / `RSMPlan.from_provenance` (MEDIUM-HIGH).** The plans have
   `provenance()` (write) but no inverse — section-3 reload needs the rebuild path. Add it during
   the notebook refresh (a persist→read→rebuild cell forces it).
3. `DetectorHeader` is mm while PONI is SI — gate with a unit assertion (LOW footgun).

**Two silent assumptions → become GUI rules (the review surfaced these):**
- **`multigeometry` ignores `corrections`/`gi`** — it uses pyFAI's *own* correctSolidAngle/
  polarization. `run_stitch` now **warns**; the GUI must **disable/relabel the section-3
  CorrectionStack + GI toggles when backend = multigeometry** (else users believe they applied
  the shared pre-weight when they didn't).
- **Stitch-2D `pyfai_hist` χ axis is provisional** (`stitch_hist.py` χ = `atan2(qz,qy)`, P3c-gated
  vs pyFAI `chiArray`). Until P3c clears, **default Stitch-2D to `multigeometry`** or surface a
  "provisional azimuth" note. (1D `|q|` is convention-free and safe.)

**Concrete current-widget anchors (for the re-slice):**
- The **PONI param** currently lives in the *wrangler* (`image_wrangler.py`) — **move it to
  section-2b detector config**, carrying its run-precondition gate.
- The section-2→3 **reactivity already exists** as `set_image_units` (`integrator.py`): GI vs
  standard repopulates the axis combos, forces unit=Q, disables `chi_offset`. Wire the new
  layout's 2c→3 coupling as the same **state→render** one-direction flow the display layer uses.
- **Tools** (Calibrate/Refine/Mask — the *producers* of section-2 state) and **`StaticControls`**
  (mode + Start/Pause) stay as the two mode-agnostic panes; re-slice **only the two middle panes**.
- The 1D/2D blocks are parallel + dimension-neutral — the re-slice maps onto each independently,
  preserving every integrate/reintegrate signal. No 1D/2D rewrite.

**UB / sample is its own block (2d), not folded into diff/detector** — per ADR-0007 the UB +
energy "come from the scan, not the goniometer"; UB is a separate persisted group. Add a distinct
**2d sample/UB sub-block** (RSM-relevant, sourced from the scan, editable for override).

## Sequenced next steps

1. **Contract cleanup** — consolidate the GI knobs into `GISettings` (approved); fix any other
   contract inconsistencies the take-stock review surfaces.
2. **Notebook refresh** (the forcing function) — Int / Stitch / RSM / GI end-to-end on the current
   API; this doubles as the GUI section spec.
3. **GUI scaffold** — the 3-section layout for the Stitch/RSM tools, section 2 auto-populated from
   loaded data + restorable from the `.nxs`, section 3 reactive to 2c.
4. **Live-gated tail** — the real-data convention validations (P3c circle order, P4/P6 GI signs,
   χ azimuth, GI refraction) are isolated and batched; the GUI exposes them behind the GI mode.
