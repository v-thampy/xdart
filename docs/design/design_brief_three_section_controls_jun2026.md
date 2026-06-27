# Design brief — XRD reduction tool, three-section controls panel

**For:** claude.ai/design (UI mockups). **Self-contained** — no code references; every term below
is user-facing. Produce clean, dense, scientific-desktop mockups of the **controls panel** (the right
column) of an X-ray diffraction data-reduction tool, in the layout and states described here.

---

## 1. What this is

A desktop app that turns raw area-detector images from a synchrotron X-ray experiment into reduced
1D/2D curves and 3D reciprocal-space maps. The window has two columns:

- **Left (~70%): the display** — plots and images (specced separately; in mockups, show it as a
  large placeholder canvas labelled "Display" so the controls have context).
- **Right (~30%, ~360 px): the controls panel** — the subject of this brief.

There are several **tools** (modes) that share this controls layout: **Integrate** (1D / 2D),
**Stitch** (combine many detector frames into one curve/image), and **RSM** (3D reciprocal-space
map). Mock the **Stitch** and **RSM** tools.

The organising idea — and the thing the mockup must make legible — is that the controls read
**top-to-bottom as the scientific workflow**: *get data → describe the instrument → choose how to
process → run.* Three numbered sections plus a tools row and a run bar.

---

## 2. The controls panel — overall structure

A vertical stack of **collapsible group boxes**, in this order:

```
┌─ Controls ───────────────────────────┐
│ [ Stitch ▾ ]            (tool picker) │   ← which tool; compact dropdown or tabs
├───────────────────────────────────────┤
│ ▾ 1 · DATA                            │
│ ▾ 2 · EXPERIMENT  (the instrument)    │
│     2a Diffractometer                 │
│     2b Detector                       │
│     2c Sample & measurement           │
│     2d Beam                           │
│ ▾ 3 · PROCESSING  (the plan)          │
├───────────────────────────────────────┤
│  Tools:  [Calibrate…][Refine…][Mask…] │   ← produce section-2 state
├───────────────────────────────────────┤
│  [  Run  ]   [ Pause ]   [ Stop ]      │   ← run bar, pinned to the bottom
└───────────────────────────────────────┘
```

Numbered section headers (**1 · DATA**, **2 · EXPERIMENT**, **3 · PROCESSING**) are the load-bearing
visual hierarchy — make them prominent (e.g. a number chip + caps label + subtitle). Section 2 has
four labelled sub-groups. Group boxes collapse/expand; show them expanded in the hero mockup.

---

## 3. The field-state badge system (a key visual)

Most fields in **section 2 are not typed by the user — they are derived from the loaded data.** The
panel must show *where each value came from* with a small colored badge/chip on the field's right.
Four states:

| Badge | Meaning | Suggested color |
|---|---|---|
| `AUTO` | inferred from the data (e.g. detector type from image shape) | blue |
| `FILE` | loaded from a calibration/parameter file | green |
| `SET` | typed by the user (the sample *is* the experiment) | amber |
| `SAVED` | restored from a previously-saved result file | grey |

This badge column is a signature element — render it consistently down section 2. A field can be
edited to override its auto/loaded value (the badge then flips to `SET`).

---

## 4. Section 1 — DATA

The data source. Compact.

| Field | Control | Example |
|---|---|---|
| Source kind | dropdown | `SPEC scan` / `Image series` / `NeXus` |
| File / folder | file-picker row (path + "Browse…") | `…/sample_A/scan_0005` |
| **Scan group** | multi-select chips OR a small list with +/− | `5, 7, 8` (Stitch & RSM combine several scans) |
| Project folder | file-picker row | `…/beamtime_2026/` |
| Save path | file-picker row | `…/results/scan_0005.nxs` |

The **scan group** control is important for Stitch/RSM — several scans combine into one result. Show
it as a row of removable chips (`5 ✕  7 ✕  8 ✕  ＋`). Loading data here is what fills in section 2
(badges go `AUTO`/`FILE`).

---

## 5. Section 2 — EXPERIMENT (the instrument)

The geometry of the apparatus — configured once, then reused/saved. Four sub-groups. Most fields
carry a state badge (§3).

### 2a · Diffractometer
| Field | Control | Example | Badge |
|---|---|---|---|
| Geometry preset | dropdown | `psic` (others: fourc, sixc, twoc, custom) | `SET` |
| Circle → motor map | small 2-col table (circle ▸ motor name) | `μ→mu, η→eta, χ→chi, φ→phi, ν→nu, δ→del` | `AUTO` |

Show the circle→motor map as a compact read-only table with an "Edit…" affordance.

### 2b · Detector
| Field | Control | Example | Badge |
|---|---|---|---|
| Calibration | **read-only summary card** + `Calibrate…` button | "Eiger 1M · dist 200.4 mm · λ 0.974 Å" | `FILE` |
| Detector type / shape | text (read-only) | `Eiger1M · 1062×1028` | `AUTO` |
| Orientation / mount | dropdown | `Top-left origin` | `AUTO` |
| Mask | summary + `Mask…` button | "12,840 px masked" | `SET` |

The calibration is a **summary card** (a few read-only key/values) with the action button beside it —
not an editable form. Same pattern for the mask.

### 2c · Sample & measurement  ← *drives section 3 (see §7)*
| Field | Control | Example | Badge |
|---|---|---|---|
| **Measurement mode** | **segmented control** | `Standard` ▏`Grazing incidence` ▏`Transmission` | `SET` |
| Sample material | text | `Si` / `HfO2` | `SET` |
| UB matrix | 3×3 read-only grid (compact) + "Edit…" | identity-ish 3×3 | `SAVED` |
| Incidence-angle source | dropdown (appears only in Grazing mode) | `Fixed 0.20°` / `From motor: eta` | `SET` |

The **measurement-mode segmented control is the single most interactive element** — switching it
re-renders section 3 (§7). Render the **Grazing incidence** variant in one mockup to show the extra
"Incidence angle" field appearing.

### 2d · Beam
| Field | Control | Example | Badge |
|---|---|---|---|
| Energy / wavelength | linked numeric pair (eV ↔ Å) | `12700 eV  ·  0.9762 Å` | `AUTO` |
| Polarization plane | dropdown | `Horizontal` | `SET` |

Energy and wavelength are two views of one value — render as a linked pair (editing one updates the
other).

---

## 6. Section 3 — PROCESSING (the plan)

The per-run reduction choices. Four groups. **Its contents depend on section-2c measurement mode and
on the tool** (§7).

### Stitch (Standard mode)
| Group | Field | Control | Example |
|---|---|---|---|
| Ranges | Radial range | min/max numeric pair (+ "auto") | `0.5 – 6.0 Å⁻¹` |
| | Azimuth range (2D) | min/max pair | `−180 – 180°` |
| Bins | Points (1D) | numeric | `2000` |
| | Radial × azimuth (2D) | two numerics | `1000 × 360` |
| Axes | Output axis | dropdown | `q (Å⁻¹)` (also 2θ, r, χ) |
| | Merge method | dropdown | `Multi-geometry` / `Histogram` |
| Corrections | toggles | checkbox group | ☑ Solid-angle ☑ Polarization ☐ Air absorption |

### RSM
| Group | Field | Control | Example |
|---|---|---|---|
| Bounds | Q bounds (H/K/L) | three min/max pairs (+ "auto-scout") | auto |
| Bins | Grid (H×K×L) | three numerics | `101 × 101 × 101` |
| Axes | Coordinates | dropdown | `hkl` / `qx,qy,qz` |
| Corrections | toggles | checkbox group | ☑ Solid-angle ☑ Polarization |

---

## 7. The one reactive rule to depict

**Section 3 re-renders when the section-2c measurement mode changes.** Show this as two variants of
the Stitch mockup:

- **Standard:** axes dropdown = {q, 2θ, r, χ}; corrections = {solid-angle, polarization, air
  absorption}.
- **Grazing incidence:** axes dropdown **gains** {q-in-plane, q-out-of-plane, exit-angle, χ (GI)};
  corrections **gain** {footprint, Fresnel, absorption, refraction}; and section 2c shows the
  **Incidence angle** field (`0.20°`).

A subtle "merge method = Histogram required" hint can disable the GI corrections when the method is
Multi-geometry (a greyed-out group with a tooltip) — nice-to-have, shows the cross-field logic.

---

## 8. Mockups to produce

1. **Hero — Stitch, Standard mode, fully populated.** All sections expanded, realistic values,
   state badges visible, run bar at the bottom showing **Run** (green).
2. **Stitch, Grazing-incidence mode.** Same panel; section 2c shows the incidence-angle field;
   section 3 shows the GI axes + the extra corrections. (Demonstrates §7 reactivity.)
3. **RSM tool.** Section 3 swapped to the RSM groups (Q bounds / grid / coordinates); section 1 shows
   a 3-scan group (`5, 7, 8`).
4. **Run states (small).** The run bar in three states: **Run** (green) → **Pause** (amber, mid-run)
   → **Resume** + **Stop**. A thin progress indication is welcome.

---

## 9. Visual style

- Scientific desktop, **information-dense but calm**; think a well-organised instrument-control panel.
- Clear numbered-section hierarchy; sub-groups indented under section 2.
- **Read-only summary cards + action buttons** for calibration/mask (not sprawling forms).
- The **state badges** (§3) and the **measurement-mode segmented control** (§5/2c) are the two
  signature interactions — make them legible.
- Monospace for numeric values and motor names; sentence-case labels.
- Light theme primary; a dark variant is welcome as a second pass.
- Keep the left "Display" area as a simple placeholder — the focus is the right controls column.
