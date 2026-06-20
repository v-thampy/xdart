# Design: wrangler widget organization (Int / Stitch / RSM)

**Status:** draft for discussion · 2026-06-20 · planning only (no code)
**Companion to:** [`design_stitching_jun2026.md`](design_stitching_jun2026.md) §5.4 (the
authoritative *input inventory*) and
[`design_diffractometer_geometry_jun2026.md`](design_diffractometer_geometry_jun2026.md)
(the calibration object the Refine button produces). This doc is about **organization** —
how those inputs are grouped into panels, gated by mode, and surfaced live vs batch — not a
new input list.
**Purpose:** pull every wrangler requirement Vivek has stated (some only in project memory)
into one place, organized for the eventual Claude Code handoff.
**Gated on:** the headless seams (`Diffractometer`, `DetectorCalibration`, `refine_goniometer`,
streaming sinks) land first; the wrangler is the thin Qt layer over them. Sequence after
3e+Phase-5 (memory `planned_features_roi_and_stitching_jun2026`).

---

## 1. North-star: one source layer, mode-gated; the wrangler is thin

- **One scan/group source layer feeds BOTH integration and stitch/RSM** (memory
  `stitching_design_reframed`). "Give me images grouped into scans, with whatever metadata is
  available." Do **not** fork into separate wranglers by source type or by mode.
- **Metadata is OPTIONAL for plain Int 1D/2D, MANDATORY for Stitch/RSM** (same memory). A bare
  image stack integrates with no motor info; stitch/RSM cannot build per-frame PONIs / Q-maps
  without per-frame detector angles. **This asymmetry is the single biggest driver of the UI
  gating** (§3).
- **Thin per-source adapters, source-agnostic analysis** (memory `source_agnostic_ingestion`,
  `keep_xdart_thin`). Source quirks (file layout, column names, sidecar `.pdi`, directory
  watch) live in the wrangler adapter; everything downstream consumes a uniform
  `motor_name → per-frame values` stream + image stream via the ssrl `FrameSource` seam.
  Anything in the wrangler that doesn't touch Qt belongs in the headless core.
- **During a run the wrangler tree is HARD-disabled** (`tree.setEnabled(False)`), not
  read-only (memory `feedback_wrangler_hard_disable`); the Grazing-checkbox repaint (#56) is an
  accepted cosmetic — don't revert to the readonly approach. **Batch mode stays silent**
  (per-frame `sigUpdate` suppressed, single end-of-run refresh — memory
  `feedback_batch_mode_silent`).

---

## 2. The mode axis drives the layout

A single **Mode** selector — `Integrate 1D` / `Integrate 2D` / `Stitch` / `RSM` — is the top
control; it gates which panels are shown/required and which display layout the controller
picks afterward (memory `display_modules_layouts`; stitch §5.3). Everything below is organized
as **panels whose visibility/required-ness is a function of Mode**.

| Panel | Integrate 1D/2D | Stitch | RSM |
|---|---|---|---|
| **Source / data** | required | required | required |
| **Geometry / calibration** (poni or gonio + preset) | optional¹ | **required** | **required** |
| **Refine button** (calibration refinement) | hidden | shown | shown |
| **DiffractometerConfig** (convention + camera mount + motor map) | hidden² | **required** | **required** |
| **UB matrix** | hidden | hidden | **required** |
| **Grazing-incidence** group | optional | optional | optional |
| **Reduction params** (npt, unit, method, ranges) | required | required | required (grid bins/ranges) |
| **Frame selection** (angle filters) | n/a | optional³ | optional³ |
| **Normalization / monitor** | optional | optional | optional |
| **scan_data capture** (positioners table) | optional | **required**⁴ | **required**⁴ |
| **Output** (file name/location, one per group) | required | required | required |

¹ A bare `.poni` is enough for plain integration; no goniometer needed.
² Plain Int needs only a single PONI, not a circle stack.
³ Frame selection is for dropping *bad* frames only — **not an accuracy gate**; with correct
geometry, stitching is good across the full nu/del range (stitch §5.4; measured full mesh ==
narrow band). No default angle guard.
⁴ Mandatory only when the user wants the stitched/gridded result correlated with experimental
variables (T, stress, time…) — see §7. Always *collected*; the requirement is that it
*round-trips* to the output file.

---

## 3. Panel-by-panel organization

### 3.1 Source / data (stitch §5.4)
- Master file path (SPEC / NeXus / Tiled) → builds the `FrameSource`.
- Scan selection **+ grouping** with range syntax `1-3, 5, 7-9` → one reduction per group.
- Image directory + filename pattern/prefix (headerless formats).
- Raw-image read params: `detector_shape`, `raw_dtype`, `header_skip`, hot-pixel/saturation
  `threshold` → `read_image` args.
- **Source-agnostic mapping:** auto-detect candidate motor columns (SPEC `#O/#P`, NeXus
  positioners, Tiled metadata) and let the user confirm/override which are the detector-arm
  angles and the incidence motor. The mapping is `motor_name → role`, consumed by
  `Diffractometer.derive_per_frame(motors)`.

### 3.2 Geometry / calibration — the load-bearing panel
This is where stitch/RSM live or die (stitch §2.5 GAPs; §3.1). Two calibration routes, both
producing a `Diffractometer` (+ `DetectorCalibration`) that feeds the same downstream path:

1. **Uncalibrated:** base `.poni` + a `Diffractometer` **preset** (`two_circle`/`psic`/…).
   Approximate but usable (scale = `deg2rad`); mild broadening, no cliff.
2. **Calibrated:** a pyFAI **goniometer JSON** → `Diffractometer.from_pyfai_goniometer(json)`.
   Sharpest over the full range (fitted per-axis scale + offset, incl. motor-zero offsets).

Plus the **detector** side (closes stitch GAP B/E):
- Detector name **+ `Detector_config`** (orientation / mask / binning) — from the gonio/poni
  or set explicitly. A non-default panel mount silently lost today → wrong geometry.
- **Image-orientation transform** (0/90/180/270 + flip/transpose; stitch GAP E) — the
  beamline-specific raw-array transform that makes the frame match the calibration. *This is
  in stitch §2.5 but NOT yet in the §5.4 input list — add it.* Surface it as a detector-mount
  dropdown; 90/270 also transpose the detector dims.

### 3.3 The "Refine" sub-panel (Vivek's idea; diffractometer §3.4–3.5)
Next to the calibration control, a **Refine** button that takes:
`(base .poni seed, calibration images, their (del, nu) metadata, calibrant)` → runs headless
`refine_goniometer` (control-point `least_squares`, **not** `refine3` — diffractometer §3.5)
→ stores the fitted `Diffractometer`. Thin button; refinement is the headless function.

UI affordances the validated recipe (diffractometer §3.5) calls for:
- **Show the fitted beam-centre next to the picker's seed** so a large drift is visible —
  the fit is only as well-conditioned as the angular spread of the control points (short-axis
  `cch1` drifted 28 px on a nu-starved mesh; pinned to <1 px of pyFAI on del-only). The
  beamline's "direct-beam pixel" is only a seed; the fit may move it ~100 px.
- **Report the recovered motor-zero offsets** (`del`/`nu`) — they were the missing ingredient
  and a non-zero value is expected, not a bug.
- **Auto-pick the detector mount by RMS** (sweep the camera-orientation combos, keep lowest
  control-point RMS) rather than asking the user to guess.
- Hint to add high-angle control frames if the conditioning is poor.

### 3.4 DiffractometerConfig capture (RSM; memory `xdart_rsm_wrangler_extensions` — GAP)
For RSM the wrangler must capture the xu convention, today **not** in either design doc's
wrangler list:
- Convention preset (`psic`/`sixc`/…) selects the sample/detector circle stack + axis
  directions; user override for the camera orientation (`init_area_detrot`,
  `init_area_tiltazimuth`) and `r_i`.
- This is the *same* `Diffractometer` object as stitch uses — RSM just additionally consumes
  the `to_qconversion()` view. The panel is shared; RSM unhides the convention/camera fields.

### 3.5 UB matrix (RSM only; memory `xdart_rsm_wrangler_extensions` — GAP)
- **SPEC:** parse from `#G3` (existing `io.spec.get_energy_and_UB`) → auto-fill, user confirms.
- **NeXus / Tiled / live:** no canonical source — let the user (a) load a UB file, (b) type a
  3×3 matrix, or (c) [later] infer from a reference reflection. v1 = (a)/(b).
- Persists to `/entry/sample/UB` (v2 schema; diffractometer §4). UB is the *sample* half — it
  stays out of `Diffractometer` (changes per sample/alignment, not per instrument).

### 3.6 Reduction params / frame selection / normalization / output
- Stitch: `mode` (1D/2D), `npt_1d`/`npt_rad_2d`/`npt_azim_2d`, `unit`, `method`,
  `radial_range`/`azimuth_range` (or auto from angle span), `mask`, `monitor_key`.
- RSM: grid bins + q/HKL ranges (or auto-scout — §6), UB, energy.
- Frame selection: optional angle filters → `frame_indices`; not an accuracy gate (§2 note ³).
- Energy/wavelength: from the energy motor or an override field. *(Watch the per-beamline unit:
  `get_from_spec_file` returns energy already in eV on the psic data — memory
  `psic_del_nu_calibration_solved`.)*
- Output: stitched/gridded file name + location, one `.nxs` per group.

---

## 4. Grazing-incidence as a click-to-expand group (memory `feature_grazing_group_toggle_idea`)
Future (own branch, not v1): replace the `Grazing` bool checkbox with the **"Grazing
Incidence" group header acting as a toggle**; child params (Theta Motor, Sample Orientation,
Tilt Angle) shown only when expanded (`child.setOpts(visible=…)`). Kills the disabled-checkbox
repaint cosmetic (#56) by removing the bool widget entirely.

---

## 5. scan_data / positioners must round-trip (memory `nexussink_scan_data_persistence`)
The per-frame motor/counter table is **collected during source reading** but the headless write
path persists only integrated stacks today. For stitch/RSM to be correlated with experimental
conditions (temperature, stress, time, field) — the whole point of variable-correlated analysis
— `NexusSink.write` must persist `scan_data` + positioners (`upsert_scan_metadata` /
`upsert_positioners` exist, just aren't called headlessly). The wrangler's responsibility is to
*surface* the available positioner columns and *guarantee* they reach the output file; ROI-stats
and any "plot vs scanned variable" view then read them back (memory
`planned_features_roi_and_stitching`). Also handle **non-numeric** metadata (string tags/status)
which is currently dropped (memory `source_agnostic_ingestion`).

---

## 6. Live vs batch — the streaming dimension (NEW, code-grounded Jun 2026)

Vivek's question: *the stitch/RSM machinery streams — could it stitch live as frames arrive,
not wait for scan end?* The architecture is **half there**, and the two families differ:

- **RSM is genuinely stream-ready.** `rsm/gridding.py::StreamingGridder` wraps
  `xu.Gridder3D(KeepData=True)` with fixed bounds and an `add(chunk)` that folds each frame's
  voxels into a running grid; `to_volume()` yields a valid **partial** volume at any time
  (`combine_grids` proves grids are mergeable). The only look-ahead is a cheap **angle-only**
  `scout()` to fix the grid bounds before the first `add()` — and that bounds-then-accumulate
  shape already exists in the reduction core as the GI `gi_freeze_mode="scout_union"` pre-pass
  (scout first+last frames, then stream). Bounds can also be supplied explicitly (`q_bounds`),
  removing the look-ahead entirely → live gridding with zero wait.
- **Powder stitch is batch-only.** `integrate/multi.py` calls pyFAI `MultiGeometry.integrate1d`
  over a Python list of *all* images + *all* per-image integrators; `analysis/plans.py::
  run_stitch` materializes every image first and even guards against a too-large eager stack,
  with a comment naming an unbuilt *"future streaming StitchPlan backend."* The per-q-bin
  (signal, counts) histogram *is* associative in principle, but pyFAI exposes no incremental
  accumulator — going live needs either a per-bin accumulator beneath `MultiGeometry` or
  re-integrating a growing list.
- **The streaming spine is mature but not wired to either.** `ScanSession`/`ReductionSession`
  stream **single-frame integration** results to a `ReductionSink` (`begin`/`write`/`finish`,
  pause/drain/finalize); there is no stitch/RSM event type and no sink hook calling
  `StreamingGridder.add()`. Both `run_stitch` and `run_rsm` run as **scan-level finalize**
  steps today; `run_rsm` is not imported anywhere in `xdart` yet.

**Wrangler implications (design choices for Vivek):**
- A **"live"** affordance is *architecturally cheap for RSM* (add an RSM-aware sink that calls
  `sg.add()` per frame/chunk on the writer thread and emits a partial-volume event; pre-fix
  bounds via scout or explicit `q_bounds`) and *not yet available for stitch* (needs the
  streaming-MultiGeometry replacement first). The wrangler could expose **Live RSM** before
  **Live stitch**.
- If a live toggle is shown, it must surface the **bounds decision** (scout pre-pass vs explicit
  q-range) since that's the one piece of look-ahead.
- Until then, present stitch/RSM honestly as **finalize-stage**: configured in the wrangler,
  produced once the scan's frames are in, displayed by the post-run controller.

---

## 7. Open questions for Vivek
1. **Live RSM v1?** Worth wiring an RSM accumulator sink into the streaming path now (cheap), or
   keep RSM finalize-only until the GUI display layout lands? (Stitch stays batch regardless.)
2. **DiffractometerConfig editing depth.** Preset-select only, or a full editor for
   circles/camera/`r_i`? (Recommend preset + camera-mount override; full editor later.)
3. **One wrangler, mode-gated panels, vs a mode-specific sub-widget swapped in.** The
   one-source-layer principle says shared source/geometry panels; RSM/stitch only *add* panels.
   Confirm we don't fork the widget.
4. **Where does scan_data surface** — a positioner picker in the wrangler, or always-persist-all
   and pick the x-axis at plot time? (Recommend persist-all; pick at plot time.)
5. **Group → output fan-out.** One `.nxs` per scan-group is assumed; confirm naming/threading
   under the hard-disable-during-run constraint.

---

## 8. References
- Docs: `design_stitching_jun2026.md` §5.3–5.4 (registration + input inventory), §2.5 (GAPs),
  `design_diffractometer_geometry_jun2026.md` §3.2–3.5 + §4 (the object, Refine, persistence).
- Code: `analysis/plans.py` (`run_stitch`/`StitchPlan`, `run_rsm`/`RSMPlan`),
  `rsm/gridding.py` (`StreamingGridder.add`/`to_volume`/`combine_grids`),
  `integrate/multi.py` (`MultiGeometry` stitch), `reduction/core.py` +
  `session/scan_session.py` (streaming sink/executor, `gi_freeze_mode="scout_union"`),
  `xdart/modules/ewald/stitch.py` (batch-only wrapper).
- Memory: `stitching_design_reframed` (one source layer; metadata optional-vs-mandatory),
  `source_agnostic_ingestion` + `keep_xdart_thin` (thin per-source adapters),
  `xdart_rsm_wrangler_extensions` (DiffractometerConfig + UB capture; v2 schema),
  `nexussink_scan_data_persistence` (positioner round-trip),
  `feature_grazing_group_toggle_idea`, `feedback_wrangler_hard_disable`,
  `feedback_batch_mode_silent`, `display_modules_layouts`,
  `planned_features_roi_and_stitching`, `psic_del_nu_calibration_solved`.
