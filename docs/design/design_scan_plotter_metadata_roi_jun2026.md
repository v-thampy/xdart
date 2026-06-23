# Design: unified Scan Plotter (metadata + ROI stats)

**Status:** draft for discussion · 2026-06-22 · planning only (no code)
**Supersedes the *framing* of** [`design_roi_stats_plotting_jun2026.md`] — that doc's
headless core (`RoiSpec` / `RoiStatsPlan` / `run_roi_stats`, background math, masking,
persistence, live monitor) **still stands and is the source of truth for the headless
layer**; this doc folds it into one GUI tool and records the refinements from the
2026-06-22 discussion (generic source picker, subtract-**or-divide** background,
normalization axis, multi-ROI overlay, and reuse of the now-shipped fitting-tool
machinery).
**Reuses (now exist, didn't when the ROI doc was written):**
- `xdart/.../param_trend.py` `ParamTrendMixin` — the per-frame accumulator + column
  selector + overlay + CSV (built for the Peak/Phase fitters).
- the region↔fields two-way sync pattern (Peak Fitter fit-range `LinearRegionItem` ↔
  numeric fields) → the 2-D `RectROI` ↔ row/col-center/width fields.
- `BatchAnalysisWorker` pattern (off-thread, per-frame `sigFrameFit`, progress,
  cancel) → the ROI-stat worker.
- the **reachable-raw** gating already used to enable/disable Reintegrate
  (`_apply_integration_control_state`) → enable/disable "Plot ROI".

---

## 1. The idea (one tool, not two)

A single **Scan Plot** tool (Tools card, a standalone non-modal `QDialog` — NOT a
display Mode) that plots **a per-frame scalar vs frame (or vs any column)** for a
"scan". Metadata plotting and ROI-stats plotting are the **same operation**; they
differ only in where the plotted column comes from:

- a **metadata column** — read for free from the scan source (motors / counters);
- an **ROI stat** — *computed* by loading each frame's raw image and reducing over a
  rectangle (sum/mean/max/…).

**Key unification:** an ROI stat is just a **computed column** appended to the same
per-frame table. So there is one plot, one column selector, one overlay, one CSV — the
ROI plotter is "add a computed column", not a second tool. (This resolves the
fold-or-not question: fold.)

## 2. The "scan" is the existing `FrameSource`

> **Update 2026-06-23:** SPEC is the **primary** scan source and is now wired
> (`xrd_tools.sources.SpecSource`: extensionless content detection, all `#L`+`#O/#P` motors,
> scan-number selection, optional images). The ad-hoc source row described below is being
> promoted to the **shared `ScanSourceWidget`**
> ([`design_shared_source_panel_jun2026.md`](design_shared_source_panel_jun2026.md), approved)
> — kind-general (SPEC/NeXus/Eiger/TIFF/Tiled-future), File or Directory entry, with the same
> picker reused by the stitch/RSM wrangler. ROI-on-SPEC (point at the image folder) lands when
> that widget is adopted here.

Not tied to the loaded `.nxs`. On open, start with the **currently loaded `.nxs`** (or
blank). A **source picker** at the top chooses a "scan", classified by the EXISTING
machinery (`xrd_tools.sources.ensure_frame_source` — the same classifier the Wranglers
+ reduction use), one of:

- a **NeXus** file (processed scan),
- an **Eiger** master/data file,
- the **first image** of a TIFF / RAW sequence (like the image Wrangler),
- a **SPEC** file (metadata only — no per-frame images).

A `FrameSource` exposes per-frame **metadata** (`motors` / `metadata_for(idx)`) and,
when present, per-frame **images** (`frame_for(idx)` / `load_frame`). Metadata is read
wherever available (NeXus `scan_data`, SPEC, `.txt`/`.pdi` sidecars beside TIFFs, image
headers — `io/metadata.py`: `read_txt_metadata` / `read_pdi_metadata` /
`read_image_metadata` / `_read_spec_metadata`). So "load metadata wherever there's
enough info" is largely existing code; the picker selects a classifier, it does not
write a new loader.

## 3. Plot model — selectable axes + normalization + overlay

The popup is XRD-agnostic: it holds a per-frame table `{frame -> {column: value}}` and
plots selected columns. Axes:

- **x axis** — a column selector. **Default = the positioner** (the scanned motor).
  Selectable to **any** column (incl. `frame_index`).
- **y axis** — a column selector. **Default = intensity** (a sensible default counter /
  the first ROI when ROIs exist). Selectable to **any** column.
- **normalization axis** — a third column selector, **default `None`**. When set to any
  column, the plotted y is **y / norm** (per frame). (Display-time; applies to whatever
  y series are shown.)
- **overlay** — plot **multiple** y columns/ROIs **together to compare** (the general
  overlay the user wants, not restricted to a single family). Reuses the
  `ParamTrendMixin` overlay; the single-left-axis ships first, the **2nd/3rd right-hand
  axis** (already a planned round-2 item for the fitting trend) lets incommensurable
  columns overlay cleanly.

CSV export of the assembled table (reuses `accumulator_to_table`).

## 4. ROIs — gating, selection, multiple, background op

### 4.1 "Plot ROI" gating = reachable raw (not just "images exist")
The button is enabled **only when raw frames are reachable**:

- Eiger / TIFF / RAW sequence → **yes** (the images ARE the source).
- SPEC → **no** (metadata only; button disabled).
- processed **NeXus** → **only if** it links back to its raw source and that raw is
  reachable on disk (the arch-v2 `relative_source_path` + the existing reachable-raw
  check). If the `.nxs` doesn't store raw and can't reach it → **disable** (the user's
  call). **Future feature flagged:** *optionally store raw frames in the `.nxs`* so a
  self-contained file can support ROI without the original tree — noted, not built.
- Never compute stats off a **thumbnail** (downsampled/quantized → wrong). Strict raw.

### 4.2 Selection (mouse + numeric, two-way synced)
"Plot ROI" pops the **first image** of the scan with ROI controls:

- a draggable pyqtgraph **`RectROI`** on the image, two-way synced to **numeric fields**
  (`center_row, center_col, width_row, width_col`) — the same value-object↔widget sync
  the Peak Fitter uses for the fit-range region.
- a **stat selector** per ROI: **sum / mean / max** (+ min / std as cheap extras).

### 4.3 Multiple ROIs + general overlay
**Several ROIs at once** (an ROI list UI like the Phase Fitter's CIF list). Each ROI
becomes its own computed column → its own overlaid curve. This is the user's general
"overlay multiple things to compare" applied to ROIs.

### 4.4 Background ROI — subtract OR divide
Each signal ROI may have an optional paired **background ROI** + an **operation**:

```
plotted = stat(signal_roi)  -  bkg          # subtract
plotted = stat(signal_roi)  /  bkg          # divide
```

- background reduced over **valid pixels** with the mask-correct `bkg_density` rule from
  the ROI doc §6.2 (so `sum` stays area-scaled and masked/dead/saturated pixels never
  distort signal or background).
- **OPEN (resolve before build):** is the background **global** (one bg ROI applied to
  all signal ROIs, as the existing ROI doc assumes) or **per-ROI** (each signal ROI
  pairs with its own bg)? The user said "for *an* ROI … a background ROI that is
  subtracted or divided from the chosen signal ROI", which reads **per-ROI**. Lean
  per-ROI (more flexible; a shared bg is the special case of pointing several ROIs at
  the same bg), but confirm.

## 5. Headless vs xdart (keep xdart thin)

The ROI math stays **headless** (`xrd_tools`), per the existing ROI doc — extend it,
don't move it into the GUI:

- `core/roi.py` `RoiSpec` + reducer math (existing doc §3).
- `analysis/plans.py` `RoiStatsPlan` + `run_roi_stats(plan, source)` →
  `AnalysisResult` (existing doc §3). **Extensions for this design:** reducer gains
  `max`/`min`/`std`; the background op gains `divide` (not just subtract); per-ROI
  background if §4.4 resolves that way.
- The **normalization axis** is a **plot-time** division (y / norm column) — it lives in
  the xdart popup, not the headless plan (it's display, and the norm column is already
  in the table).

xdart owns only: the source picker, the popup (axis selectors + overlay + CSV — reusing
`ParamTrendMixin`), the ROI image popup (`RectROI` ↔ fields + stat/bg selectors), and
the **ROI-stat worker** (the `BatchAnalysisWorker` pattern: load each raw frame off the
GUI thread, call the headless reducer, emit per-frame, fill the plot incrementally with
progress + cancel). I/O (loading every frame) is the real cost; the stat is cheap.

## 6. Subtleties / explicitly-deferred

- **Fixed pixel ROI assumes the feature stays put** on the detector across the scan; a
  drifting ring would be sampled by a static box. v1 = static ROI; per-frame ROI
  tracking is much later.
- **Same-detector assumption** — a pixel ROI isn't meaningful across a multi-geometry
  scan. Assume one detector for v1.
- **q-space / cake ROIs** (ROI on the cake or q-χ image, not raw pixels) — future; the
  `RoiSpec` would gain a `space` discriminator (existing ROI doc §5).
- **Store raw in the `.nxs`** — the future feature that would let a processed file
  support ROI standalone (§4.1).
- **Mask/saturation awareness** — the existing ROI doc §6.3 prerequisite (R3-C:
  invalid-pixel policy into core) still applies so headless ROI stats match the
  reducer/display. Ship with static-mask + NaN exclusion first, full saturation parity
  after R3-C.

## 7. Staged plan (each step independently testable)

The headless steps are the existing ROI doc §7 steps 0–2 (`RoiSpec`+math →
[R3-C prereq] → `RoiStatsPlan`/`run_roi_stats`), **extended** with the `divide` op,
extra reducers, and (if chosen) per-ROI background. The GUI steps, in order:

1. **Scan source picker + metadata table.** Source picker (loaded `.nxs` / blank →
   pick NeXus/Eiger/TIFF-seq/SPEC via `ensure_frame_source`); read metadata into the
   per-frame table. *Gate:* offscreen test — each source kind populates the expected
   columns; SPEC yields metadata + no images.
2. **The plot popup (metadata only).** x/y/normalization column selectors (defaults:
   positioner / intensity / none) + overlay + CSV, over the table from step 1, reusing
   `ParamTrendMixin`. *Gate:* synthetic table renders the expected traces; normalization
   divides; overlay shows multiple.
3. **"Plot ROI" gating + the ROI image popup.** Enable only on reachable raw; first
   image + `RectROI` ↔ numeric fields (2-way sync round-trip), multiple ROIs, stat + bg
   ROI + subtract/divide. *Gate:* sync round-trip (drag → spec → fields and back);
   gating truth-table per source kind.
4. **ROI-stat worker → columns.** Off-thread per-frame raw load + `run_roi_stats`;
   each ROI series appended as a column; plot fills incrementally with progress/cancel.
   *Gate:* end-to-end on a small real/synthetic image-series source; the incremental
   series equals a direct `run_roi_stats` (a mini live≡batch spine).
5. **(optional) persistence + live monitor** — existing ROI doc §7 steps 5–6.

## 8. Decisions captured from the 2026-06-22 discussion

- **Fold metadata + ROI into one tool** (ROI = computed column). ✓
- **Generic source picker** (NeXus/Eiger/TIFF-seq/SPEC), not nxs-bound; start on the
  loaded nxs or blank. ✓
- **Several ROIs at once** + a **general overlay** to compare multiple columns/ROIs. ✓
- **Background ROI** per signal ROI, **subtract OR divide** (op selectable). ✓ (global
  vs per-ROI bg: lean per-ROI — confirm.)
- **Disable Plot ROI** when raw isn't stored in / reachable from the source (esp. a
  processed nxs without stored raw — and *storing raw in the nxs* is a future feature). ✓
- **Selectable axes:** x default = positioner, y default = intensity, both any column;
  **+ a normalization axis** (default none, any column → y/norm). ✓

## 9. References
- [`design_roi_stats_plotting_jun2026.md`] — the headless core (RoiSpec / RoiStatsPlan /
  run_roi_stats), background math (§6.2), masking prereq (§6.3), persistence (§6.4),
  live monitor (§6.1), full step sequence (§7). **Read it for the headless contract.**
- Code to reuse: `xrd_tools/sources/__init__.py` (`ensure_frame_source`),
  `xrd_tools/io/metadata.py` (readers), `xrd_tools/io/read.py`
  (`get_metadata["scan_data"]` / `_scan_data_for_frames` / raw resolution),
  `xdart/.../param_trend.py` (`ParamTrendMixin`), `xdart/.../analysis_worker.py`
  (`BatchAnalysisWorker`), `xdart/.../static_scan_widget.py`
  (`_apply_integration_control_state` reachable-raw gating; the dialog-parameterized
  batch wiring `_on_batch_clicked(dialog)`).
- Memory: `planned_features_roi_and_stitching_jun2026`, `keep_xdart_thin`,
  `gui_redesign_direction_a` (the fitting tools + `ParamTrendMixin` + the batch worker).
