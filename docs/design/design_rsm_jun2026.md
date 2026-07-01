# Design: Reciprocal Space Mapping (RSM) — consolidated

**Status:** PARTIAL · reconciled 2026-06-27. Headless RSM is implemented through the
shared `Diffractometer`, weighted streaming gridder, NeXus persistence, and correction
seams. Production angle reroute, GI refraction/convention validation, and the RSM GUI
viewer/wrangler remain deferred/P7. Current status authority:
[`stitching_rsm_build_plan.md`](stitching_rsm_build_plan.md).
**Depends on:** the shared `Diffractometer`
([`design_diffractometer_geometry_jun2026.md`](design_diffractometer_geometry_jun2026.md) —
the single geometry input, incl. `to_qconversion`); the per-pixel correction stack
([`design_intensity_corrections_jun2026.md`](design_intensity_corrections_jun2026.md)); the
shared source panel ([`design_shared_source_panel_jun2026.md`](design_shared_source_panel_jun2026.md))
+ `CompositeFrameSource`; the wrangler layout
([`design_wrangler_organization_jun2026.md`](design_wrangler_organization_jun2026.md) §3.4–3.6, §6).
**Shares with stitching:** the **histogram/`StreamingGridder` accumulator IS the histogram-stitch
merge** ([`design_stitching_jun2026.md`](design_stitching_jun2026.md) §2.6) — only the bin space
differs (RSM grids (qx, qy, qz); stitch grids (q, χ)). Same corrections, same geometry, same
source layer.
**Reference notebooks (NOT in-repo — `~/repos/example_notebooks/RSM/`):** `RSM_process.ipynb`,
`RSM.ipynb`; cross-beamline `~/repos/example_notebooks/Stitching/` calibration fixtures (the
del/nu pose + xu control-point fit). These are the validation gates (headless, no live test).

---

## 1. What RSM is (and the one-line seam)

A reciprocal-space map grids per-pixel scattering intensity from a rocking/mesh scan into a
3-D **(qx, qy, qz)** volume (or 2-D slices / 1-D projections of it), using the diffractometer
pose per frame to map each detector pixel to **q**. The hard seam, identical in shape to the
histogram stitch:

> **(images, per-frame `Diffractometer` pose, UB, energy, corrections) → q per pixel → grid →
> RSM volume.** The accumulator never knows where the angles came from.

RSM differs from stitching only in the **bin space** ((qx,qy,qz) vs (q,χ)) and in needing a
**UB matrix** (the sample orientation, to put q in HKL/sample frame). Everything upstream — the
source layer, the `Diffractometer`, the corrections — is shared.

## 2. What already exists (cited — read before acting)

The headless RSM spine is **largely built**; the remaining gaps are production convention
reroute/validation and xdart wiring.

- **Plan seam:** `analysis/plans.py` `RSMPlan(diff_motors, bins, UB, energy, q_bounds, roi, …)`
  + `run_rsm(plan, source | [sources])` (l.177/194) → `AnalysisResult(kind="rsm", payload=RSMVolume)`.
  Accepts one source OR a list (multi-scan) via `ScanInput(scan, energy, UB, roi)`.
- **Streaming gridder:** `rsm/gridding.py` `StreamingGridder(mapper: PixelQMap, bins)` (l.192)
  wraps `xu.Gridder3D` in `KeepData(True)` with a FIXED `dataRange` — `scout()` (l.249, fix the
  bounds from a cheap angle-only pass), `add(chunk)` (l.310, fold each chunk's voxels into the
  running grid), `to_volume() -> RSMVolume` (l.399, a valid PARTIAL volume at any time),
  `combine_grids(...)` (l.585, grids are mergeable). Bounds can be supplied explicitly (`q_bounds`)
  → zero look-ahead. This **is** the histogram-stitch accumulator (stitching §2.6).
- **Pipeline:** `rsm/pipeline.py` `ScanInfo(spec_path, img_dir, h5_path)` (the spec+images scan
  definition — now generalized by the shared source panel), `load_images`, `process_scan_data`,
  `process_scan_from_nexus`, `grid_scans_streaming`. `volume.py` `RSMVolume`.
- **Geometry today:** the pipeline has the shared `Diffractometer` drop-in path and
  `assemble_circle_angles`; production reroute is deferred until convention validation.
- **NOT wired into xdart:** `run_rsm` is imported nowhere in `src/xdart`. No RSM wrangler, no RSM
  viewer controller. Display reserves the seam (§6).

## 3. Headless design — reconcile onto the shared seams

### 3.1 Geometry = the shared `Diffractometer.to_qconversion()` (implemented drop-in)
RSM's q-mapping must derive from the **one** `Diffractometer`
(`design_diffractometer_geometry_jun2026.md` §3, ADR-0007), not a parallel qconversion. The
geometry object already plans `to_qconversion()`/`to_hxrd(energy)` — the SAME adapter the
`xu_hist` stitch backend uses. So:
- `PixelQMap` is built from `Diffractometer.to_qconversion()` + the per-frame angles from the
  source's `scan_data` (motor→role mapping from the shared `Diffractometer` preset).
- The **camera mount / image-orientation transform** (stitch GAP E) is the same field; RSM reads
  it as the xu camera tuple, stitch as the pyFAI `Detector_config`.
- **Gate:** the reconciled q-map reproduces the pipeline's current voxel positions on the
  `RSM_process` notebook fixture.

### 3.2 UB matrix — the sample half (RSM-only)
The **UB** orients q into the sample/HKL frame; it stays OUT of `Diffractometer` (it changes per
sample/alignment, not per instrument — `design_wrangler_organization_jun2026.md` §3.5).
- **SPEC:** parse from `#G3` via `io.spec.get_energy_and_UB` → auto-fill, user confirms.
- **NeXus/Tiled/live:** no canonical source — load a UB file or type a 3×3 (v1); infer from a
  reference reflection later.
- Persisted to `/entry/sample/UB` (v2 schema, capability-gated).

### 3.3 Corrections — the shared per-pixel weight stack (implemented headless)
RSM consumes the **same** correction stack as stitch — applied as a per-pixel **weight** into
`StreamingGridder.add` (the accumulator seam). Solid-angle/polarization and GI intensity
weights are implemented; GI refraction/convention validation remains a real-data gate.

### 3.4 Streaming + multi-scan (the gridder is already stream-ready)
- **Live RSM is architecturally cheap** (`design_wrangler_organization_jun2026.md` §6): the
  `StreamingGridder` already `add()`s chunks into a running grid and `to_volume()` yields a valid
  partial at any time. The only look-ahead is the cheap angle-only `scout()` to fix bounds (or
  pass `q_bounds` explicitly → zero wait). Wire an RSM-aware sink that calls `sg.add()` per
  chunk on the writer thread and emits a partial-volume event — the same streaming spine the
  reduction uses. (Stitch's histogram merge gets live-streaming for free from the same shape.)
- **Multi-scan** combine: `run_rsm` already takes a list of sources; a scan **group** (the
  shared source panel's `CompositeFrameSource`, stitching §2.6 / source-panel §2) grids into ONE
  volume — `combine_grids` proves grids merge.

## 4. Persistence (schema-as-code, additive, capability-gated)
Persist the **RSMVolume** (qx/qy/qz axes + intensity + counts) as a scan-level schema group
(mirroring stitched_1d/2d, stitching §4) with a `rsm` `CapabilityAttr`; persist provenance
(`Diffractometer` + UB + bins + q_bounds + corrections). **Implemented:** `write_rsm` /
`read_rsm` and schema/capability registration. **Deferred:** convenience `get_rsm` reader.

## 5. xdart (thin) — wrangler + viewer

### 5.1 Wrangler (mode = RSM)
Reuse the shared `ScanSourceWidget(mode="rsm")` (source-panel §3, grouping on) +
`design_wrangler_organization_jun2026.md`:
- **§3.4 DiffractometerConfig:** convention preset (`psic`/`sixc`) + camera mount + motor→role
  map — the SAME `Diffractometer` panel stitch uses; RSM unhides the convention/camera fields.
- **§3.5 UB capture** (§3.2 above).
- **Reduction params:** grid `bins` + q/HKL ranges (or auto from `scout`), `energy`.
- **Live toggle** (§3.4): surfaces the bounds decision (scout pre-pass vs explicit `q_bounds`).

### 5.2 Viewer — needs repeated display roles (#69 / WS-X2)
Unlike stitch, the RSM viewer shows **repeated** panels — a 2×3 of `SLICE_2D` (qx-qy, qx-qz,
qy-qz cuts) + `PROJ_1D` projections — so it **requires** the display registry's repeated-role
support (the WS-X2 TODO at `display_logic.py:924`; stitching §5.3 correctly notes stitch does
NOT need it but RSM does). P7/deferred: register `Mode.RSM_VIEWER` + a `PANEL_LAYOUT` of repeated
`PanelKey(SLICE_2D)/PanelKey(PROJ_1D)` instances + an `RSMViewerController` (`design` §10 seam
in the static-scan display layer). Slice axis + index are view state.

## 6. Gated step sequence (each independently testable; headless gates = the RSM notebooks)
0. **DONE: shared `Diffractometer`** (geometry doc) lands first.
1. **DONE headless: reconcile q-mapping** onto `Diffractometer.to_qconversion()` (§3.1). *Gate:* voxel
   positions match the `RSM_process` fixture; old qconversion retired/aliased.
2. **DONE headless: corrections into the gridder weight** (§3.3). *Gate:* corrected voxel intensities match the
   GI-corrections notebook.
3. **PARTIAL: UB capture + persistence** (§3.2, §4). *Gate:* `#G3` parse; write→read round-trip; UB-less
   path warns.
4. **P7/deferred: xdart RSM wrangler** over the shared source panel (§5.1) — DiffractometerConfig + UB + bins.
   *Gate:* end-to-end `run_rsm` on the notebook scan → a volume; grouping → one volume.
5. **P7/deferred: RSM viewer** (§5.2) — requires WS-X2 repeated roles. *Gate:* offscreen `display_logic` test
   for the 2×3 layout; controller renders slices/projections.
6. **(cheap, optional) Live RSM** — an RSM sink calling `StreamingGridder.add` per chunk;
   partial-volume events; bounds via `scout` or explicit `q_bounds` (§3.4). *Gate:* a
   live≡finalize mini-spine — the streamed partial == `run_rsm` over the same frames.

## 7. References
- Code: `analysis/plans.py` (`RSMPlan`/`run_rsm`), `rsm/gridding.py`
  (`StreamingGridder.scout/add/to_volume/combine_grids`), `rsm/pipeline.py`
  (`ScanInfo`/`process_scan_*`/`grid_scans_streaming`), `rsm/volume.py` (`RSMVolume`),
  `core/geometry/pixel_q.py` (`PixelQMap`), `core/geometry/diffractometer.py`,
  `io/spec.py` (`get_angles`/`get_energy_and_UB`), `display_logic.py` (WS-X2 repeated-role TODO
  l.924, `PanelRole`).
- Docs: `design_diffractometer_geometry_jun2026.md` (geometry + `to_qconversion`),
  `design_intensity_corrections_jun2026.md` (the shared correction stack),
  `design_stitching_jun2026.md` §2.6 (the shared histogram/`StreamingGridder`
  merge), `design_shared_source_panel_jun2026.md` (`ScanSourceWidget`/`CompositeFrameSource`),
  `design_wrangler_organization_jun2026.md` §3.4–3.6 + §6 (DiffractometerConfig/UB/live RSM).
- Notebooks (`~/repos/example_notebooks/`): `RSM/RSM_process.ipynb`, `RSM/RSM.ipynb`;
  `Stitching/Multi120_Calibration_*` + `xu_geometry_del_nu.json` (the xu pose/calibration),
  `Stitching/Multi120_GI_Corrections_Explorer.ipynb` (GI corrections).
- Memory: `stitching_dual_backend_decision` (the shared gridder), `keep_xdart_thin`,
  `xdart_rsm_wrangler_extensions` (DiffractometerConfig + UB), `nexussink_scan_data_persistence`.
