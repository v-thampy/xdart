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

## 3. Headless design (the one source layer)

### 3.1 Geometry input = the shared `Diffractometer`
Stitch's per-frame detector rotations come from
`Diffractometer.to_pyfai_per_frame(motors)` (companion doc), the **same** object RSM
consumes. `run_stitch` today takes `rot1_key`/`rot2_key` and reads raw motor columns;
reconcile it to **derive rotations from the scan's `Diffractometer`** (already on
`Scan.geometry` / persisted + reloadable), with the explicit-key path kept as an
override. This is the concrete "one geometry object, both consumers" payoff.

### 3.2 Per-image PONIs (the different-position case)
Keep `run_stitch`'s common path (shared `base_poni` + per-frame rotations) and add the
general path the old doc proposed: a `stitch_ponis(images, ponis)` primitive (a
complete `PONI` per image) so a source that genuinely moves the detector (per-position
PONI files, or a translation motor) feeds a per-image PONI list and skips rotation
derivation. `MultiGeometry` consumes either identically.

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
  **grouping** layer on top (which frames form one output).
- **Grouping** keeps the range syntax (`1-3, 5, 7-9` → group, single, group) as the
  one expressive field; each group → one `run_stitch` call → one stitched output.

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

---

## 6. Open questions — resolved or flagged

1. **Multi / cross-file combine** (old §5.1). **Resolved:** grouping within a source is
   the range syntax; "Multi" is specifically *combine across different files/formats*
   into one output → model it as a **composite `FrameSource`** (a list of sources
   presented as one frame stream) handed to `run_stitch`. Not a separate code path —
   just a source that concatenates.
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
2. **Reconcile `run_stitch` to the one source layer.** Derive per-frame rotations from
   `scan.Diffractometer` (explicit `rot*_key` as override); add the `stitch_ponis`
   per-image-PONI path; persist stitch provenance. **Gate:** synthetic multi-frame
   source → stitched 1D + 2D; per-image-PONI path; no-raw on a moved tree fails loud
   with a clear message (never stitch off thumbnails).
3. **xdart grouping over `FrameSource`.** Range-syntax grouping; one `run_stitch` per
   group; composite source for cross-file "Multi". **Gate:** grouping parser test;
   end-to-end on real SPEC data (the old §11 "1000-frame psic" check).
4. **Stitch viewer registration.** `Mode.STITCH_VIEWER` + `PANEL_LAYOUT` +
   `StitchViewerController` + `register_controller`; "Int-minus-raw" layout; optional
   raw popup (shared dialog). **Gate:** offscreen `display_logic` test asserts the
   panels/layout for the mode; controller `build_payload` renders a stitched result.
5. **(deferred) Live directory stitching + streaming `StitchPlan` backend.** Only when
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
- Superseded: `docs/gui/stitching_design.md`, `docs/gui/nexus_stitch_refactor_plan.md`.
- Decisions: ADR-0002 (capability attrs), ADR-0003 (per-frame cardinality — stitch is
  out of scope of it), ADR-0005 (store ownership), ADR-0006 (finalize-stage
  classification + the prepare/capability pattern).
- Memory: `stitching_design_reframed`, `display_modules_layouts_jun2026`,
  `keep_xdart_thin`, `planned_features_roi_and_stitching_jun2026`.
