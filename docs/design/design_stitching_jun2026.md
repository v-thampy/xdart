# Design: stitching — reconciled to the xrd-tools monorepo / arch-v2

**Status:** draft for discussion · 2026-06-14 · planning only (no code)
**Supersedes:** `docs/gui/stitching_design.md` and
`docs/gui/nexus_stitch_refactor_plan.md` (both pre-monorepo / pre-arch-v2; kept for
provenance). This doc is the current source of truth for stitching.
**Gated on:** 3e+Phase-5 (one store / `FrameRecord`) done + tested.
**Depends on:** the shared geometry object —
[`design_diffractometer_geometry_jun2026.md`] — for per-frame rotations; N1 raw
resolution (✓); schema-as-code (✓); the display-controller registry (✓). The Stitch
*viewer* does **not** require render-extensibility #69 (see §5.3).

---

## 1. Why the old docs need reconciling

The two existing docs predate everything structural about the current repo. What they
assume vs what is now true:

| Old doc assumption | Now true |
|---|---|
| Two repos (`ssrl_xrd_tools` + `xdart`) | **One monorepo** `xrd-tools` (`src/xrd_tools` + `src/xdart`) |
| "create `core/geometry.py` with `DiffractometerGeometry`" | **Already exists** (`core/geometry/diffractometer.py`); being consolidated into a shared `Diffractometer` (companion doc) |
| Target `0.36.0`; **no backcompat, no `schema_version`** | Past `0.41/0.40`; **schema-as-code** with `PROCESSED_SCHEMA_VERSION`, `CapabilityAttr`s, `ACCEPTED_SCHEMA_NAMES` back-compat (`io/schema.py`) |
| Writer = xdart `sphere.py::_save_to_nexus` | Writer is **core** (`io/nexus.py` via a `ReductionSink`); xdart sink is Qt-signal-only |
| New `read_sphere`/`write_stitched`/`read_stitched` | Readers are `get_1d`/`get_2d`/`get_metadata`/`read_scan`/`ProcessedScan`; **`write_stitched`/`read_stitched` already exist** in `io/nexus.py` (but are NOT yet schema-registered — §4) |
| Mode dropdown "gains Stitch 1D/2D" in a wrangler | Display is the **`Mode` enum + `PanelRole` + controller registry** (`display_logic.py`); a Stitch viewer **registers** a controller |
| `StitchSource` Protocol + per-format backends | The **`FrameSource` + `open_source(spec)`** abstraction already exists; `StitchSource` collapses into it + a thin grouping layer |

**What the old docs got RIGHT and we keep:**
- The hard seam: **(images, per-image PONIs) → `MultiGeometry` → stitched pattern.**
  The bottom box never knows where the PONIs came from. This is exactly the
  one-source-layer reframing (memory `stitching_design_reframed`): **one scan/group
  source layer feeds BOTH integration and stitch/RSM; metadata is OPTIONAL for plain
  Int 1D/2D but MANDATORY for stitch/RSM.**
- The `stitch_ponis(images, ponis)` idea (a complete PONI per image, generalizing
  "base PONI + rotation offsets") — handles the different-detector-*position* case for
  free.
- Scan grouping via range syntax (`1-3, 5, 7-9`) as xdart UX.

---

## 2. What already exists (cited)

- **Headless plan seam:** `StitchPlan` + `run_stitch(plan, source, *, frame_indices=)`
  in `analysis/plans.py`. Materializes images eagerly with a `max_eager_bytes` guard
  that raises `MemoryError` and **names the future streaming backend**; pulls
  `rot1_key`/`rot2_key`/`monitor_key` series via `_metadata_series`; `base_poni` from
  the plan or `source.poni`; calls `stitch_images(...)`; returns
  `AnalysisResult(kind="stitch", payload, provenance)`.
- **Stitch primitive:** `integrate/multi.py` — `create_multigeometry_integrators`
  (l.33), `stitch_1d` (l.86), `stitch_2d` (l.145), `stitch_images` (l.264). Handles
  2-circle + psic via the optional `rot2_angles`.
- **Persistence (partial):** `io/nexus.py::write_stitched` (l.1519) writes
  `/entry/stitched_1d` + `/entry/stitched_2d` (NXdata, scan-level); `read_stitched`
  (l.2639) reads them into an xarray. **Gap:** these are **not** registered in
  `io/schema.py` `SCHEMA` (no `stitched_1d`/`stitched_2d` `GroupSchema`, no
  `CapabilityAttr`) — so they bypass schema-as-code (no capability detection, no
  validator coverage). Reconciling that is step one of the persistence work.
- **Display reservations:** `display_logic.py` already reserves
  `PanelRole.STITCH_2D` and `Mode` is open for a `STITCH_VIEWER`. No controller/layout
  yet.

---

## 2.5 Validated against real data (Jun 2026) — gaps the stitching notebook exposed

`examples/.../Stitching/stitch_simplified.ipynb` was built + executed end-to-end on
real beamline data (SSRL, **Pilatus 300k-w on a `psic` arm**: LaB6 at 17 keV, a
287-frame `mesh del 5 45 / nu -10 29`; `.raw` int32 `(195, 1475)`; `del`/`nu` from the
SPEC file). It runs the headless readers (`io.spec.get_angles`, `io.image.read_image`,
the result/`PONI` containers) but had to fall back to **pyFAI's own goniometer** for
the stitch — which surfaced concrete gaps in the current stitch modules:

- **The real-world geometry is a pyFAI `GoniometerRefinement`** (`MG_gonio_object.json`):
  params `{dist, poni1, poni2, rot1_offset, rot1_scale, rot2_offset, rot2_scale,
  rot3_offset}` + a `GeometryTransformation` — `rot1 = rot1_scale·nu + rot1_offset`,
  `rot2 = rot2_scale·del + rot2_offset`, `rot3 = rot3_offset (≈0)` — over a detector
  carrying `Detector_config {"orientation": 3}` (the 90° panel mount). This is the
  declarative model `DiffractometerGeometry` already approximates, *fitted*.

- **GAP A — `stitch_images` / `create_multigeometry_integrators` hardwire the scale to
  `deg2rad`.** `create_multigeometry_integrators` (`multi.py:33`) builds each AI as
  `rot1 = base.rot1 + deg2rad(motor)` / `rot2 = base.rot2 + deg2rad(motor)`. It cannot
  use the **fitted** `rot1_scale`/`rot2_scale` (≈0.0172/0.0168 ≈ 0.96–0.99·deg2rad
  here), so per-frame geometry drifts from the calibration by `(deg2rad − scale)·angle`
  — **~1.7° at del=45, growing with angle**. The notebook confirms the stitched pattern
  diverges from the calibrated goniometer.

- **GAP B — `PONI` carries only the detector *name*, not `Detector_config`.**
  `core/containers.PONI.detector` is a `str`; `poni_to_integrator` (`calibration.py:110`)
  rebuilds it with `detector_factory(name)` (`:164`) at **default** config. A non-default
  orientation / custom mask / binning is silently lost. It only works here because
  `Pilatus300kw`'s pyFAI default *is* `orientation 3`; on a detector whose needed
  orientation isn't the default, the PONI base would be geometrically wrong.

- **GAP C — no per-frame *full-geometry* input path.** `run_stitch`/`stitch_images`
  vary only `rot1`/`rot2` from one fixed base; there is no way to hand the stitch a list
  of complete per-frame geometries. The `stitch_ponis(images, ponis)` primitive (§3.2)
  does not exist yet, so neither a calibrated goniometer **nor** a moving-detector source
  can flow through the headless API.

- **GAP D — no goniometer/calibration model in core, and no importer.** The real
  artifact is a pyFAI gonio JSON. `DiffractometerGeometry` is close but (a) produces only
  `rot1/2/3` (no `dist`/`poni`/`detector`), (b) has no detector config, and (c) has no
  `from_pyfai_goniometer` constructor. So a calibration computed in pyFAI can't be loaded
  into the xrd-tools model.

- **GAP E — image-array orientation is a required, beamline-specific input (the biggest
  *practical* bug; validated).** The raw detector array must be transformed (rot/flip/
  transpose) to match the orientation the calibration used — here a **180° rotation**.
  With identity, every frame's q-map is wrong and the stitched rings wash out to a flat
  line. The detector's `Detector_config orientation` alone is **not** sufficient (identity
  stays broken). **This is exactly what the old `rot3=-90` was wrongly compensating for.**
  So the source/geometry spec needs an explicit **image-orientation transform** field
  (0/90/180/270 + flip/transpose; 90/270 also swap the detector dims → a transposed
  detector). The wrangler must collect it per detector mount.
- **GAP F — the detector mask is mandatory + build the MultiGeometry with `unit=`.**
  Without the module-gap mask (`detector.calc_mask()`, applied per-image via `lst_mask`)
  the gaps dominate the histogram and flatten the pattern. And `gonio.get_mg()` returns a
  `(2th_deg, chi_deg)` MultiGeometry — setting `.unit` afterward is **silently ignored**,
  so the headless stitch must construct `MultiGeometry(ais, unit=…, radial_range=…)`
  explicitly. Both must be baked into the headless stitch path (the notebook hit all of
  GAP E/F before it produced LaB6).

**Net:** for this psic/Pilatus case the calibrated goniometer is the production path; the
current headless stitch needs the geometry carrier + bridge + an explicit image-orientation
transform + mandatory masking. Closing A–F = a **per-frame full-geometry carrier**
(§3.1–3.2) + a **pyFAI-goniometer bridge** (companion doc) + an **image-orientation input**
+ **always-on detector masking**. These are the load-bearing additions for a real
stitching module.

---

## 2.6 THREE stitch backends — the merge is selectable (decision 2026-06-23)

**Stitching factors into two independent choices — the q-GEOMETRY engine and the MERGE
engine — and we ship THREE of the meaningful combinations** (Vivek, 2026-06-23). Validation
on real del/nu data (the `Multi120_*` notebooks, §8) converged on **xrayutilities** for the
geometry (it fits the `del`/`nu` motor offsets directly + evaluates the stacked pose exactly,
where pyFAI `MultiGeometry` hard-wires `deg2rad` per axis = GAP A), while pyFAI still owns the
per-pixel **intensity corrections** — but the **histogram** merge (vs pyFAI's azimuthal
`MultiGeometry`) is independently valuable because it **streams** (stitch-on-the-fly). So:

```python
StitchPlan.backend: Literal["xu_hist", "pyfai_hist", "multigeometry"] = "xu_hist"
```

|  | q-geometry | merge | 1D + 2D | streams | role |
|---|---|---|---|---|---|
| **`"multigeometry"`** | pyFAI AIs (`to_pyfai_per_frame`) | pyFAI `MultiGeometry` | **both via MG** | no | the validated pyFAI azimuthal path |
| **`"pyfai_hist"`** | pyFAI q/χ maps | per-pixel **histogram** | both via hist | **yes** | pyFAI geometry, streaming merge |
| **`"xu_hist"`** (default) | xu `Ang2Q.area` (`to_qconversion`) | per-pixel **histogram** | both via hist | **yes** | the converged path |

- **`"multigeometry"`** — per-frame pyFAI `AzimuthalIntegrator`s → pyFAI `MultiGeometry`, which
  does geometry + corrections + azimuthal-integration + merge in one. **Both the 1D and the 2D
  (q, χ) stitch go through MultiGeometry** (`stitch_1d`/`stitch_2d`, §3.2) — the validated
  pyFAI azimuthal path. *(Gate: compare its 1D + 2D against the `reduce_pyFAI_multigeometry`
  notebook to confirm parity — §7.)*
- **`"pyfai_hist"` / `"xu_hist"`** — share ONE per-pixel **histogram** merge fed by a per-frame
  **q-provider** `(|q| per pixel, χ per pixel, weight)`; they differ ONLY in the q-provider:
  `pyfai_hist` from pyFAI's q/χ maps, `xu_hist` from `Diffractometer.to_qconversion()` → xu
  `HXRD.Ang2Q.area(...)`. Per-pixel **corrections** (solid-angle, polarization, GI stack)
  applied as weights, reused from pyFAI arrays (`design_intensity_corrections_jun2026.md`). The
  histogram is the notebooks' shared `stitch(provider)` and the SAME accumulator shape as RSM's
  `rsm.gridding.StreamingGridder` (`design_rsm_jun2026.md`) — so it **streams**
  (stitch-on-the-fly, §3.4).

**The shared seam.** `run_stitch(plan, source)` dispatches on `plan.backend`. `stitch_ponis`
(§3.2) is the `multigeometry` feeder; a sibling `stitch_q_grid(provider, corrections)` is the
**shared histogram** feeder for BOTH `pyfai_hist` and `xu_hist` (only the q-provider closure
swaps). `xu_hist` is the default (the converged, geometry-exact, streaming path) — *flagged for
the maintainer*.

**Why this does NOT touch geometry (§3.1):** the shared `Diffractometer` already produces both
adapters — `to_pyfai_per_frame` (MultiGeometry + the pyFAI q-provider) and
`to_qconversion`/`to_hxrd(energy)` (the xu q-provider, the same adapter RSM consumes). All three
backends consume the SAME per-frame `DetectorCalibration`; only the q-provider + merge differ.
Corrections are a shared pre-weight feeding all three (and RSM). *(Validated: on del-only the
backends overlay in ring position — notebook §6 confirms pyFAI's corrections reweight intensity,
not peak position; the del/nu "edges-off" was high-`nu` extrapolation, not a missing engine. The
3-way `pyfai_hist` vs `multigeometry` vs `xu_hist` comparison also cross-checks histogram-vs-MG
parity for the pyFAI geometry.)*

---

## 3. Headless design (the one source layer)

### 3.1 Geometry input = a base `DetectorCalibration` + the shared `Diffractometer`

**Refinement is OPTIONAL — two construction routes, ONE stitch path** (Vivek, Jun 2026):
- **Uncalibrated** (no goniometer fit): build the `Diffractometer` from a **preset + a
  base `DetectorCalibration`** (a single-position `.poni` + its `detector_config` **and
  image-orientation transform**, GAP E). Per-frame `rot = base.rot + deg2rad(motor)`
  (scale = 1). Approximate but **usable across the full nu/del range** — a single low-nu
  row is sharpest, wide nu broadens mildly (no cliff; med|Δq| 0.027→0.048 Å⁻¹). This is
  today's `stitch_images`/`create_multigeometry_integrators`, **fixed to carry
  `detector_config` + the image transform** (GAP B/E). Matches the old notebook's
  `MultiGeometry([AzInt(...rot1=nu,rot2=del...)])` path.
- **Calibrated** (goniometer fit done): `Diffractometer.from_pyfai_goniometer(gonio_json)`
  — fitted per-axis scale+offset (§3.4). Sharpest over the full range. (Build the
  `MultiGeometry` explicitly with `unit=` — `gonio.get_mg()` returns `2th_deg` and a
  post-hoc `.unit` is silently ignored, GAP F.)

Both routes produce per-frame `DetectorCalibration`s that feed the **same**
`stitch_ponis(images, geometries)` → `MultiGeometry` (§3.2). The stitch primitive is
geometry-source-agnostic — "refinement optional" just means the `Diffractometer` was
**preset-built** or **gonio-fitted**. The Refine button (§5.4) is what turns the
uncalibrated inputs into a calibrated goniometer.

A stitch frame's geometry has two parts, and the current API only carries one:

- **Constant per scan:** `dist, poni1, poni2`, the *detector* and its `Detector_config`
  (orientation/mask/binning), wavelength. Today this is a `PONI`, which **drops the
  detector config** (GAP B). Fix: a **`DetectorCalibration`** = `PONI` fields **plus**
  `detector_config` (or extend `PONI` with a `detector_config: dict`), so the base
  carries the 90° mount and any custom detector setup. Both stitch (pyFAI AI) and RSM
  (the `DetectorHeader`/camera mount) read it.
- **Per frame:** `rot1/rot2/rot3` from `Diffractometer.to_pyfai_per_frame(motors)`
  (companion doc) — the **same** object RSM consumes. Crucially the `Diffractometer` must
  carry the **fitted** per-axis scale+offset (not a hardwired `deg2rad`, GAP A): its
  `AngleMapping(sign=scale, offset=…)` already expresses `rot = scale·motor + offset`, so
  a fitted goniometer model fits the object exactly.

So: `per_frame_geometry = base DetectorCalibration ⊕ Diffractometer.to_pyfai_per_frame`.
`run_stitch` today takes `rot1_key`/`rot2_key` and **adds `deg2rad(motor)`** — reconcile
it to derive rotations from the scan's `Diffractometer` (fitted scales preserved), with
the explicit-key path kept only as a no-calibration fallback.

### 3.2 `stitch_ponis` must accept *full* per-frame geometries (closes GAP C)
Add the primitive the old doc proposed, generalized: **`stitch_ponis(images,
geometries)`** where each `geometry` is a complete per-frame `DetectorCalibration`
(dist/poni/rot1/rot2/rot3 + detector_config) — built either from the
`base ⊕ Diffractometer` path above, or supplied directly when a source genuinely *moves*
the detector (per-position PONI files, a translation motor). `MultiGeometry` consumes the
resulting per-image AIs identically. `create_multigeometry_integrators` becomes the
thin "base + Diffractometer rotations → per-frame geometries" helper feeding it.

> **Backend-aware (§2.6):** `stitch_ponis` is the **`"multigeometry"`** feeder (per-frame
> `DetectorCalibration`s → AIs → `MultiGeometry`, both 1D + 2D). The histogram backends
> **`"pyfai_hist"`** and **`"xu_hist"`** take the SAME per-frame `DetectorCalibration`s through
> a sibling `stitch_q_grid(provider, corrections)` — a per-frame q-provider `(|q|, χ, weight)`
> → a per-pixel histogram merge into the (q, χ) grid, pyFAI correction arrays as weights —
> differing ONLY in the q-provider (pyFAI q/χ maps vs `to_qconversion` → `Ang2Q.area`). All
> dispatched by `run_stitch(plan, source)` on `plan.backend`; the geometry carrier is shared.

### 3.2a Bridge: load a pyFAI goniometer calibration (closes GAP D)
The real-world calibration artifact is a pyFAI `GoniometerRefinement` JSON. Add a
**`Diffractometer.from_pyfai_goniometer(json)`** importer (companion doc) that parses the
`GeometryTransformation` expressions into `AngleMapping`s (scale→`sign`, offset→`offset`)
and the base params into a `DetectorCalibration` (incl. `detector_config`). This lets a
beamline calibrate in pyFAI and stitch/RSM headlessly in xrd-tools with no pyFAI
`Goniometer` at runtime — and is the single most useful interop step (the notebook
currently calls pyFAI's `Goniometer.sload` directly only because this bridge is missing).

### 3.3 Stitch is a SCAN-LEVEL finalize-stage product — NOT a `FrameRecord`
Critical reconciliation to arch-v2: ADR-0003/0005 are about **per-frame** records
(`FrameRecord`, `core/frame_view.py:476`, holds per-mode `FrameView`s). A stitched
pattern is **one result for the whole scan/group** — it does not fit the per-frame
store and must not be forced into it. ADR-0006 already classified stitch as a
**finalize-stage** post-pass ("all frames already in memory"). So:

- Stitch output is a **scan-level `AnalysisResult`** (already is), persisted as a
  **scan-level schema group** (`stitched_1d`/`stitched_2d`), displayed by a controller
  that reads the whole-scan product. It **does not touch** `FrameEvent`/`FrameRecord`
  cardinality. This keeps stitching cleanly orthogonal to the store collapse.
- It still produces per-image `integrated_1d`/`integrated_2d` (the existing per-frame
  stack) **in addition** to the merged stitch, so the viewer can show per-image QA.

### 3.4 Streaming (forward seam, not v1)
`run_stitch` eagerly materializes all images (guarded). `MultiGeometry` itself is not
streaming-friendly, but the **scout-bounds → bin → finalize** shape mirrors RSM's
streaming gridder. Flag a future `StitchPlan` streaming backend (the `MemoryError`
message already points at it); v1 stays eager with the guard. If a `prepare_*` pass is
ever needed for stitch bounds, it follows the **ADR-0006 pattern** (a concrete
function + the `scan_manifest()` capability), *not* a speculative framework — ADR-0006
explicitly classified stitch as finalize-stage with no prepass today, so do not build
one until a real need lands.

---

## 4. Persistence (schema-as-code)

1. **Register the existing stitched groups.** Add `stitched_1d`/`stitched_2d`
   `GroupSchema` entries + `CapabilityAttr`s to `io/schema.py` so `write_stitched`/
   `read_stitched` are covered by capability detection + validators (they exist but
   are unregistered today — the one real correctness gap). Preserve the **as-is**
   `stitched_2d` orientation `(n_q, n_chi)` (`write_stitched` docstring + xdart files
   stay interchangeable) — do **not** "fix" it to the per-frame `(chi, q)` convention.
2. **Write through the sink/schema path**, not a bespoke xdart writer (the
   "complete-v2-record orchestration into core" arc): the headless run produces a
   complete file; xdart only triggers it.
3. **Add a convenience reader** `get_stitched_1d`/`get_stitched_2d` mirroring
   `get_1d`/`get_2d` (notebook-friendly), alongside the existing xarray `read_stitched`.
4. Persist the stitch provenance (plan + `Diffractometer` + group list) so a reloaded
   scan's stitch is reproducible — and so the **mandatory** geometry metadata for
   stitch is present offline.

---

## 5. xdart (thin): viewer + association UX

### 5.1 Producing a stitch
- Stitching is **batch/finalize only** (not live/append — stitch is whole-scan by
  definition). xdart builds a `FrameSource` (or a grouped set) and calls
  `run_stitch` — it does **not** reimplement the stitch.
- **Source = the existing `FrameSource`/`open_source(spec)` abstraction.** The old
  `StitchSource` Protocol + per-format backends collapse into this: SPEC/NeXus/Tiled/
  image sources already exist as `FrameSource`s; stitching needs only a thin
  **grouping** layer on top (which frames form one output). The wrangler's Source panel
  is the shared `ScanSourceWidget`
  ([`design_shared_source_panel_jun2026.md`](design_shared_source_panel_jun2026.md),
  approved 2026-06-23) — SPEC scan-number / NeXus entry / Eiger / TIFF / Tiled-future, File
  or Directory entry, embedded with `mode="stitch"`.
- **Grouping** keeps the range syntax (`1-3, 5, 7-9` → group, single, group); each group
  **combines** its scans into one output via a `CompositeFrameSource` (shared-panel doc §2)
  → one `run_stitch` call → one stitched output.

### 5.2 Display layout — "Int-minus-raw, optional raw popup"
(Memory `display_modules_layouts_jun2026`.) The Stitch viewer shows the **integrated/
stitched** result and **drops the inline raw panel**; raw is reachable via an
**optional popup**:

- **Stitch 2D** → `PanelRole.STITCH_2D` (the stitched cake) + `PanelRole.PLOT_1D` (a
  line-cut / the 1D projection). No inline `RAW_2D`.
- **Stitch 1D** → `PanelRole.PLOT_1D` only.
- **Optional raw popup** → a standalone QDialog showing a chosen frame's raw —
  **reuse the same dialog infrastructure as the ROI / `scan_data` popup**
  ([`design_roi_stats_plotting_jun2026.md`]), not a registered panel.

### 5.3 Registration (the seam, with exact API)
Register through the existing controller registry (`display_logic.py` /
`display_controllers.py`):

1. Add `Mode.STITCH_VIEWER` to the `Mode` enum.
2. Add a `PANEL_LAYOUT[Mode.STITCH_VIEWER]` `PanelLayout` (stitched panel on top, 1D
   below; raw hidden).
3. Implement a `StitchViewerController(_BaseController)` with `compute_state(widget,
   mode) -> DisplayState` (panels = `[(PanelKey(PanelRole.STITCH_2D), ...),
   (PanelKey(PanelRole.PLOT_1D), ...)]`) and `build_payload(widget, state) ->
   DisplayPayload` (reads the stitched result from the file/session).
4. `register_controller(Mode.STITCH_VIEWER, StitchViewerController())`.

**#69 / WS-X2 is NOT required for stitching.** `STITCH_2D` and `PLOT_1D` are
**non-repeating** roles, and the render dispatch is role-level today
(`display_frame_widget.render_display`); the WS-X2 TODO
(`display_logic.py:924`) only matters for **repeated** roles (RSM's `SLICE_2D`/
`PROJ_1D` instances). So stitching slots into the current dispatch as-is — correcting
the memory note that lumped stitch + RSM as both needing #69. (RSM does; stitch
doesn't.)

### 5.4 Inputs the wrangler widget must collect (from the notebook)

> **Organization** (how these inputs are grouped into mode-gated panels, plus the RSM-only
> additions — DiffractometerConfig + UB capture, image-orientation transform, scan_data
> round-trip, live-vs-batch) is specced in
> [`design_wrangler_organization_jun2026.md`](design_wrangler_organization_jun2026.md). This
> section stays the authoritative *input list*; that doc is the *layout*.

Everything the notebook hard-codes becomes a widget field. Grouped by concern, with the
`StitchPlan`/source mapping (★ = **new**, not on `StitchPlan` today):

**Source / data**
- SPEC (or NeXus/Tiled) master file path → builds the `FrameSource`.
- Scan selection **+ grouping** with range syntax `1-3, 5, 7-9` → one `run_stitch` per
  group (`frame_indices` / a grouped source).
- Image directory + filename pattern/prefix (`{prefix}{spec}_scan{N}_{frame:04d}.raw`),
  or auto-derived → resolved by the source.
- ★ Raw-image read params for headerless formats: `detector_shape`, `raw_dtype`,
  `header_skip` (or inferred from the detector) → `read_image` args on the source.

**Geometry / calibration** (the load-bearing new inputs)
- ★ **Calibration source**: a pyFAI **goniometer JSON** (preferred) *or* a base
  `.poni` + a `Diffractometer` preset → a `DetectorCalibration` + `Diffractometer`
  (§3.1–3.2a). Replaces today's bare `base_poni`.
- ★ **"Refine" button next to the calibration control** (Vivek, Jun 2026): when no
  goniometer JSON exists yet, the user supplies a **single-position base `.poni`** (from
  pyFAI-calib on one low-angle image) + **calibration images + their `(del,nu)`
  metadata** + the calibrant (LaB6); the button runs the headless
  `refine_goniometer(base_poni, images, angles, calibrant, …)` (companion doc §3.4) and
  stores the resulting `Diffractometer`. The base `.poni` is the *starting point* for
  the fit, not the final geometry. Thin button; refinement is the headless function.
- ★ Detector name **+ `Detector_config`** (orientation, mask, binning) — from the gonio/
  poni or set explicitly (GAP B).
- ★ Convention/preset (`two_circle`/`psic`/…) **+ motor-name mapping**: which SPEC
  columns are the detector-arm angles (`del`,`nu`) and the incidence motor (GI). The
  notebook assumes `del`/`nu`; the widget must let the user map/override.

**Stitch parameters** (mostly exist on `StitchPlan`)
- `mode` (1D/2D); `npt_1d` / `npt_rad_2d` / `npt_azim_2d`; `unit`; `method`;
  `radial_range` / `azimuth_range` (or auto from the angle span).
- `mask` (detector mask) ★ + a **hot-pixel/saturation threshold** (`threshold`→NaN in
  `read_image`; the notebook used `8e5`).
- `monitor_key` for per-frame normalization (i0/i1; optional).

**Frame selection** (optional — NOT an accuracy gate)
- Angle-range filters (nu/del) to drop genuinely bad frames only. **No default guard:**
  with the geometry correct, stitching is good across the full nu/del range (measured:
  full mesh == narrow band), so default to **ALL frames**. The `nu·del` cross-term of
  pyFAI's fixed order is a graceful residual the *fitted* path shares too — restrict only
  if you actually see broadening at extreme angles. Applied as `frame_indices`.

**Energy / wavelength**: from the SPEC `energy` motor, or an override field.
**Output**: stitched-file name/location (one `.nxs` per group; §6.5).

> Most "stitch parameters" already exist on `StitchPlan`; the **new** surface is the
> calibration/Diffractometer carrier, `detector_config`, raw-read params, and the
> angle-range frame filters — i.e. exactly GAPs A–D plus the validity window.

---

## 6. Open questions — resolved or flagged

1. **Multi / cross-file combine** (old §5.1). **Resolved:** grouping within a source is
   the range syntax; "Multi" is specifically *combine across different files/formats*
   into one output → model it as a **`CompositeFrameSource`** (a list of sources
   presented as one frame stream) handed to `run_stitch`. Not a separate code path —
   just a source that concatenates. Specced + built in
   [`design_shared_source_panel_jun2026.md`](design_shared_source_panel_jun2026.md) §2
   (`parse_scan_groups` + `CompositeFrameSource`), shared with RSM + the ROI plotter.
2. **Per-source motor→angle mapping** (old §5.2). **Resolved:** the shared
   `Diffractometer` preset supplies defaults; auto-detect candidate motor names from
   the source (SPEC `#O/#P`, NeXus positioners, Tiled metadata); user overrides names
   in the geometry panel. Convention static-per-instrument; values from `scan_data`.
3. **Different detector-position UX** (old §5.3). **Resolved headless** (`stitch_ponis`
   per-image PONIs); xdart input UX (multiple PONI files vs a translation motor)
   **flagged** for the implementing branch.
4. **Live directory stitching** (old §5.4). **Deferred:** stitch is batch/finalize;
   a directory watcher that stitches each new scan reuses the existing watch
   machinery + the grouping rule — a later step, not v1.
5. **Output naming/location** (old §5.5). **Flagged:** one `.nxs` per stitched output;
   convention e.g. `combi_scans_1-3_stitched.nxs` — implementing-branch detail.
6. **`stitched_2d` orientation.** **Resolved:** keep the existing as-is `(n_q, n_chi)`
   layout for file interchange; just register it in the schema (§4.1).

---

## 7. Gated step sequence (each step independently testable; gates front-loaded)

> Land **after** 3e+Phase-5 is done + tested. Stitch is orthogonal to the store
> collapse (scan-level, not `FrameRecord`), but reuses the post-collapse display
> read-paths, so sequence it after.

0. **(prereq) Shared `Diffractometer`** through step 4 of the companion doc, so stitch
   and RSM share one geometry input. **Gate:** companion doc's gates.
1. **Schema-register stitched output.** Add `stitched_1d`/`stitched_2d` `GroupSchema`
   + `CapabilityAttr` to `io/schema.py`; route `write_stitched` through the sink/schema
   path; add `get_stitched_1d/2d` readers. **Gate:** write→read round-trip; capability
   feature-detect; existing `read_stitched` xarray still passes; orientation preserved.
2. **Geometry carrier + goniometer bridge (closes GAPs A–D).** Add the
   `DetectorCalibration` (PONI + `detector_config`), make `Diffractometer` carry fitted
   per-axis scale+offset (companion doc), add `stitch_ponis(images, geometries)` over
   *full* per-frame geometries, and add `Diffractometer.from_pyfai_goniometer(json)`.
   **Gate (real data):** load `MG_gonio_object.json` → stitch the LaB6 17 keV mesh →
   assert the stitched pattern matches the pyFAI-`Goniometer` reference within the
   radial bin width (the notebook is the fixture); a non-default `Detector_config`
   round-trips; the fitted scale (not `deg2rad`) is used.
3. **Reconcile `run_stitch` to the one source layer.** Derive per-frame rotations from
   `scan.Diffractometer` (explicit `rot*_key` as a no-calibration fallback); persist
   stitch provenance (Diffractometer + DetectorCalibration). **Gate:** synthetic
   multi-frame source → stitched 1D + 2D; no-raw on a moved tree fails loud (never
   stitch off thumbnails).
3b. **Histogram stitch backends `"pyfai_hist"` + `"xu_hist"` (§2.6).** Add
   `StitchPlan.backend` (3 values); implement the shared `stitch_q_grid(provider, corrections)`
   — a per-frame q-provider `(|q|, χ, weight)` → a per-pixel **histogram merge** into the
   common (q, χ) grid (the `Multi120_Compare_xu_vs_pyFAI_del_only` notebook's `stitch(provider)`
   and the same accumulator shape as RSM's `StreamingGridder`, so it streams); the two histogram
   backends differ only in the q-provider (pyFAI q/χ maps vs `to_qconversion` → `Ang2Q.area`);
   per-pixel
   corrections (solid-angle / polarization) applied as weights from pyFAI arrays. **Gate (real
   data):** the 3-way `multigeometry` (1D+2D via MG, parity vs `reduce_pyFAI_multigeometry`) vs
   `pyfai_hist` vs `xu_hist` comparison on the del-only LaB6 scan — all overlay in ring position
   within the radial bin width (the `Multi120_Compare` notebook is the fixture); per-pixel χ
   matches pyFAI `chiArray` ≤ 0.03°.
4. **xdart grouping over `FrameSource`.** Range-syntax grouping; one `run_stitch` per
   group; composite source for cross-file "Multi"; collect the §5.4 inputs. **Gate:**
   grouping parser test; end-to-end on the real SPEC mesh (the notebook data).
5. **Stitch viewer registration.** `Mode.STITCH_VIEWER` + `PANEL_LAYOUT` +
   `StitchViewerController` + `register_controller`; "Int-minus-raw" layout; optional
   raw popup (shared dialog). **Gate:** offscreen `display_logic` test asserts the
   panels/layout for the mode; controller `build_payload` renders a stitched result.
6. **(deferred) Live directory stitching + streaming `StitchPlan` backend.** Only when
   a real need lands; streaming follows the RSM gridder shape; any bounds prepass
   follows ADR-0006 (concrete function + `scan_manifest()`), not a framework.

---

## 8. References
- Code: `analysis/plans.py` (`StitchPlan`/`run_stitch`), `integrate/multi.py`
  (`stitch_images`/`stitch_1d`/`stitch_2d`/`create_multigeometry_integrators`),
  `io/nexus.py` (`write_stitched` l.1519 / `read_stitched` l.2639), `io/schema.py`
  (where stitched groups must be registered), `core/scan.py` (`FrameSource`,
  `Scan.geometry`), `core/frame_view.py:476` (`FrameRecord` — why stitch is NOT
  per-frame), `display_logic.py` (`Mode`/`PanelRole.STITCH_2D`/`PanelKey`/`PanelLayout`/
  `register_controller`, WS-X2 TODO l.924), `display_controllers.py` (`_BaseController`).
- Code (gaps): `integrate/multi.py:33` (`create_multigeometry_integrators`, hardwired
  `deg2rad`), `integrate/calibration.py:110,164` (`poni_to_integrator` → `detector_factory(name)`,
  drops `Detector_config`), `core/containers.py` (`PONI.detector: str`).
- **Reference notebooks (NOT in-repo — `~/repos/example_notebooks/Stitching/`):** the
  dual-backend + calibration validation fixtures on real SSRL data (LaB6, Pilatus 300k-w,
  `del`/`nu` mesh):
  - `Multi120_Compare_xu_vs_pyFAI_del_only.ipynb` — the **canonical dual-backend** head-to-head
    (§2.6): one shared `stitch(provider)` histogram merge fed by a `pf_provider` and an
    `xu_provider` (`HXRD.Ang2Q.area`); §6/§7 = the "why xu" conclusion + the corrections answer.
  - `Multi120_Diagnose_xu_pyFAI_intensity_discrepancy.ipynb` — the per-pixel correction diff
    (solid-angle/polarization reweight intensity, not ring position).
  - `Multi120_GI_Corrections_Explorer.ipynb` — the GI correction stack (footprint/refraction/
    Fresnel), pairs with `design_intensity_corrections_jun2026.md`.
  - `Multi120_Calibration_Pilatus300kw_del_nu*.ipynb` + `MG_gonio_del_nu_*.json` /
    `xu_geometry_del_nu.json` — the goniometer/xu control-point calibrations (the geometry doc's
    fixtures); `integration_xru.ipynb` / `reduce_pyFAI_multigeometry.ipynb` — the two engines
    standalone; `stitch_simplified.ipynb` — the simplified end-to-end.
- Superseded: `docs/gui/stitching_design.md`, `docs/gui/nexus_stitch_refactor_plan.md`.
- Decisions: ADR-0002 (capability attrs), ADR-0003 (per-frame cardinality — stitch is
  out of scope of it), ADR-0005 (store ownership), ADR-0006 (finalize-stage
  classification + the prepare/capability pattern).
- Memory: `stitching_design_reframed`, `display_modules_layouts_jun2026`,
  `keep_xdart_thin`, `planned_features_roi_and_stitching_jun2026`.
