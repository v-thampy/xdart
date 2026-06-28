# Int-1D/2D → three-section layout — detailed migration design

**Status:** detailed design (Jun 2026), ready for UI mockups (Claude design) → Qt implementation.
Parent: `design_gui_three_section_layout_jun2026.md` (the principle + the Stitch/RSM reference).
Sizing: **MODERATE, ~6.5–13 h** (migration analysis `aad2247d`) — a re-organisation of already-
separated widgets, *not* a redesign. **Do this once the Stitch/RSM 3-section layout is proven**, so
Int inherits a validated pattern; not bundled with the new-tool wiring.

This is the **structural** spec (which control lives where + the re-wiring). The visual arrangement
of section-2's sub-groups (tabs vs stacked group-boxes vs a tree) is deliberately left to the
mockup pass — see *Open UX questions*.

---

## 1. Current vs proposed pane structure

The right column is a vertical `QSplitter` (`rightSplitter`, `staticUI.py:88`). Today, top→bottom:

```
TOOLS      (toolsFrame, 50px)      Calibrate | Make Mask         ── mode-agnostic, KEEP
WRANGLER   (wranglerFrame)         imageWrangler param tree      ── re-slice
INTEGRATOR (integratorFrame, ≤360) integratorTree                ── re-slice
CONTROLS   (controlsFrame, fixed)  StaticControls (mode/Start)   ── mode-agnostic, KEEP
```

Proposed — the two middle panes become **three** (Tools + Controls untouched):

```
TOOLS                              Calibrate | Make Mask | Refine…
SECTION 1: DATA                    (the wrangler, minus the PONI param)
SECTION 2: EXPERIMENTAL CONFIG     (NEW pane — the instrument)
SECTION 3: PROCESSING OPTIONS      (the integrator panel, minus the GI frame)
CONTROLS                           mode | Start/Pause/Stop
```

**The re-slice touches only `wranglerFrame` + `integratorFrame`.** `toolsFrame`/`controlsFrame`,
`StaticControls`, and the 4-pane host stay as-is. The 1D and 2D blocks are fully independent
(separate frames, param trees `bai_1d_args`/`bai_2d_args`, handlers) so everything below applies to
each dimension identically — **no 1D/2D rewrite**.

---

## 2. Widget-move table (current location → new section)

**→ SECTION 2 (experimental config / the instrument):**

| Control | Current location | New home |
|---|---|---|
| **PONI file** + its run-gate | wrangler Calibration group (`image_wrangler.py:60,293`; gate `:1090`) | **2b** detector |
| Detector type / orientation / wavelength | derived from PONI/calibration (no widget today) | **2b** detector + **2d** beam |
| **GI on/off** | `integrator.gi_enable` (`integrator.py:277`) — live source | **2c** measurement |
| GI incidence motor | `integrator.gi_motor` (`:286`) | **2c** |
| GI sample_orientation | `integrator.gi_sample_orientation` (`:303`) | **2c** |
| GI tilt | `integrator.gi_tilt` (`:310`) | **2c** |
| GI material/density/film (NEW — needed for `GICorrectionStack`) | — | **2c** |
| Diffractometer convention (NEW — psic/fourc…) | — (inferred today) | **2a** |

The wrangler's existing hidden GI carrier group (`image_wrangler.py:86-109`) was a persistence
shim; section-2c becomes the **live** home and the carrier is retired (or repurposed as the
session-restore target).

**→ SECTION 3 (processing options / the plan) — stays in the integrator, minus GI:**

| Control | Location |
|---|---|
| 1D axis/unit, npts (+ `npts_oop` GI), radial/azim ranges + auto | `integratorUI.py:85/102/152` |
| 2D axis/unit, npt_rad/npt_azim, ranges + auto | `integratorUI.py:264/282/334` |
| Corrections (Advanced tree): correctSolidAngle, polarization_factor, method, chi_offset | `integrator.py:59-96` |
| Pixel reject: threshold enable/min/max, Mask Saturated | `integratorUI.py:488-520` |
| Reintegrate 1D / 2D + Advanced | `integratorUI.py:447` |

**Energy/wavelength (single-sourced):** bind the **2d** energy widget to the **calibration
wavelength** (the canonical source, `core/energy.py`); `GICorrectionStack.energy_eV` (2c) and any
plan energy derive from it. A divergence already warns (`check_energy_consistency`). **GI material**
(2c) feeds the same `GICorrectionStack` — both stay section-2-global, not duplicated (Vivek's call).

---

## 3. The section-2 panel (NEW) — sub-groups

A new `experimentalConfig` widget (parallels `imageWrangler`), four sub-groups, each binding to a
headless object; **persisted form = the `/entry/diffractometer` blob + `DetectorCalibration` + the
GI/sample state**, so reload re-hydrates section 2 from a loaded `.nxs`:

- **2a Diffractometer** → `Diffractometer.preset` (psic/fourc/sixc/two_circle dropdown) + the circle
  stack; per-circle motor map (`circle_motors`, feeding `assemble_circle_angles`). *Refine…* (a
  Tools action) writes back the fitted scales here.
- **2b Detector & calibration** → `DetectorCalibration` (PONI dist/poni/rot + wavelength) +
  `Detector_config` orientation + the detector mask. The relocated **PONI… button** + a read-only
  summary of the loaded calibration.
- **2c Sample & measurement** → one **`GISettings`** object: GI on/off, incidence (motor or fixed
  `incident_angle_deg`), `sample_orientation`, `tilt_deg`, + the `GICorrectionStack`
  material/density/film. Plus **UB / sample orientation** (RSM-relevant; sourced from the scan).
- **2d Beam** → energy/wavelength (bound to 2b's calibration wavelength) + the polarization plane.

**Field provenance (the §design-doc table):** most 2b/2a/2d fields are *derived* (PONI file → 2b;
SPEC motor names → 2a; `.nxs` blob → all), so the panel shows each field's source (inferred /
loaded / user-set / restored) and is largely read-after-load with explicit override.

---

## 4. Signal re-wiring (the load-bearing part)

Only **one** signal crosses the section-2↔3 boundary today, and it must survive the move:

```
GI toggle (gi_enable)  →  integrator.sigUpdateGI  →  staticWidget.update_scattering_geometry
   →  scan.gi = gi  →  integrator.set_image_units()   [re-render section-3 axes/ranges]
```

After migration, the GI toggle lives in **section-2c**, so:
- **2c's GI toggle emits `sigUpdateGI`** (re-home the emit; keep the same `staticWidget` slot).
- `staticWidget.update_scattering_geometry` still writes `scan.gi` and still calls
  `integrator.set_image_units()` (`integrator.py:1808`) — section 3 re-renders unchanged
  (GI↔standard axis swap, unit=Q force, `chi_offset` read-only). **This is the §state→render flow**:
  section-2 mutates `scan`, section-3 re-renders from it.
- `chi_offset` read-only (`integrator.py:1827`) is driven by the same `sigUpdateGI` — re-use it.

**PONI progressive disclosure:** section 3 should be **disabled/greyed until `scan.poni is not
None`** (you can't pick npts/ranges without a calibration). Wire the relocated PONI load/clear (2b)
to enable/disable the section-3 pane (re-use the existing `_inputs_valid` gate, `image_wrangler.py:1090`).

**Signals that must keep working** (re-home the widget, keep the connection): `sigUpdateGI`; the
range-default repopulation (`_update_gi_mode_1d/2d` → `_set_range_defaults_*`); the per-control
`textChanged`/`toggled`/`currentIndexChanged` → `_save_to_session`; the advanced-corrections
`sigTreeStateChanged` → `get_args`; the reintegrate dispatch; and **`get_gi_config()`**
(`integrator.py:1483`) — its reader must follow the GI widgets to 2c.

---

## 5. Proposed wireframe (for the mockup pass)

```
┌ TOOLS ─────────────────────────────────────────────────┐
│  [Calibrate]  [Make Mask]  [Refine…]                    │
├ 1 · DATA ───────────────────────────────────────────────┤
│  Project Folder […]   Save Path […]                     │
│  Source: [Image series ▾]  glob [*.tif]  frames [1-9]   │
│  Monitor/i0 [____]                                      │
├ 2 · EXPERIMENTAL CONFIG ─────────────────────────────────┤
│  2a Diffractometer  [psic ▾]   circles: x+,z-,y+,z- / x+,z-│
│  2b Detector        [PONI…]  Pilatus 100k · dist 0.20 m  │   ← greys 3 until loaded
│                     orientation [3 ▾]   [Mask…]          │
│  2c Sample/Measure  ☐ Grazing incidence                  │
│                     incidence: ● motor [eta ▾]  ○ fixed […]°│
│                     orient [1] tilt [0.0]° material [Si]  │
│                     UB […]                               │
│  2d Beam            energy 10.00 keV  (λ 1.2398 Å, from PONI)│
├ 3 · PROCESSING OPTIONS  (greyed until PONI loaded) ──────┤
│  1D  axis [Q ▾]  npts [1000]   radial [..][..] ☑auto     │
│  2D  axis [Q ▾]  npt_rad [1000] npt_azim [360]  ranges…  │
│  ▸ Corrections   ☑ solid-angle  pol [0.99]  method [..]  │
│  Pixel reject    ☐ threshold [..][..]   ☑ Mask saturated │
│  [Reintegrate 1D] [Reintegrate 2D]   ▸ Advanced          │
├ CONTROLS ───────────────────────────────────────────────┤
│  Mode [Int 1D ▾]  ☐ Batch  ☐ Live   [Start] [Pause] [Stop]│
└─────────────────────────────────────────────────────────┘
```

When **2c Grazing** is ticked, section 3's axis combos swap to {Q, Qip, Qoop, exit-angle, χ_GI},
unit forces to Q, `chi_offset` greys out, and `npts_oop` appears — all via the existing
`set_image_units()` (the only behavioural coupling).

---

## 6. Phased implementation (each phase shippable)

1. **Containers** — add a section-2 `experimentalConfig` frame + relabel the integrator frame as
   section-3 in `staticUI.py`/`static_scan_widget.py`; reparent `frame1D`/`frame2D`/pixreject/
   reintegrate into section-3. (No behaviour change yet — just the split.)
2. **GI extraction** — move `gi_enable`/`gi_motor`/`gi_sample_orientation`/`gi_tilt` (+ a new
   material field) into 2c; re-home the `sigUpdateGI` emit; point `get_gi_config()` at 2c. Verify
   `set_image_units()` still fires + the axis swap works.
3. **PONI relocation** — move the PONI param + run-gate to 2b; wire load/clear to enable/disable
   section 3 (progressive disclosure).
4. **2a/2b/2d binding** — add the Diffractometer-preset, detector-orientation, and energy widgets;
   bind energy to the calibration wavelength; make reload hydrate section 2 from the
   `/entry/diffractometer` blob (the diff convention, which today is re-derived each load).
5. **Session backward-compat** — old sessions lack the new 2a/2c/PONI keys; fall back to the old
   keys when the new ones are absent.
6. **Regression pass** — GI on↔off axis/range swap; PONI load → section-3 unhide; reintegrate reads
   GI from 2c; 1D/2D independence; session round-trip; live/batch/reload equivalence unchanged.

**Main risks:** the `set_image_units()` disconnect/reconnect (`integrator.py:1835/1914` — block
signals during restore, as today); the PONI progressive-disclosure gate; session backward-compat.

---

## 7. Open UX questions (for the mockup pass)

- Section-2 sub-groups: **tabs** (2a|2b|2c|2d) vs **stacked group-boxes** vs a **param tree**? (The
  wrangler uses a param tree today; the integrator uses widgets.)
- Should 2c GI collapse to a single "Grazing incidence" expander (hidden until ticked), matching the
  current integrator GI frame's "More…" popup?
- Where does the read-only "loaded calibration / loaded-from-file" summary live — inline in 2b, or a
  status strip?
- Does section 3 *disable* (grey) or *hide* until PONI loaded? (Disable keeps the layout stable.)

---

## Tier-B note: redo the wrangler grouping + disclosure here (added Jun 2026, post live-test)

A **tactical** regroup landed on the live ParameterTree wrangler (`image_wrangler.py`, commit
`8704678`): PROJECT = Folder + Save Path, DATA leads with **Poni**, the standalone CALIBRATION group
is gone, and the N1 progressive disclosure was **collapsed from three stages to two** (Project always
→ whole DATA reveals once a folder is set) *because the PONI picker now lives inside DATA and so can
no longer be its own reveal stage*. That staging conflict is a symptom of the ParameterTree forcing
group-granular show/hide. **In the custom-card migration, redo this properly:**

- Cards are `QFrame#card`s, so grouping is free layout — **Poni is just the first field row of the
  DATA card**, no group/path gymnastics, and Save Path is a Project-card row.
- Disclosure becomes **per-field/per-card enable or reveal** driven by the value model, not
  group-tree show/hide — so a guided "folder → calibration → data" flow (or any other) is expressible
  *without* the deadlock that forced the 2-stage collapse. Decide the flow at design time, not around
  a tree constraint.
- **Write Mode moves to the Controls strip** (run/output property, not a data input); the DATA card
  holds only data inputs.
- Mirror all of this for the **NeXus wrangler** (still on the old Calibration/Output grouping).
