# Design: unified Scan Plotter (metadata + ROI stats)

> **2026-06-29 — planned split: ROI Statistics gets its OWN Tools button.** The
> maintainer wants **ROI Statistics surfaced as a separate tool button** in the
> Tools card (`static_scan_widget._build_tools_card`), distinct from the current
> **"Plot Metadata"** button — today ROI stats are reached *through* Plot Metadata
> (the "add a computed column" fold below). This is a **GUI-surface** change only,
> NOT a re-split of the headless layer: the unified per-frame-table /
> `ParamTrendMixin` / `BatchAnalysisWorker` machinery stays shared. Likely shape:
> "Plot Metadata" opens the plotter focused on metadata columns; a new
> **"ROI Statistics"** button opens the same dialog (or a sibling) focused on the
> ROI picker + computed ROI columns. Details TBD with the maintainer ("more on that
> later"). Keep the headless `RoiSpec`/`RoiSignal`/`run_roi_signals` contract intact.

**Status:** PARTIAL / implemented · reconciled 2026-06-27. The Scan Plotter,
metadata plotting, ROI picker, ROI worker, per-ROI reducers/background operations, and
source picker are implemented. Optional ROI persistence/live monitor remain deferred.
**Supersedes the *framing* of** [`design_roi_stats_plotting_jun2026.md`] — that doc's
headless core (`RoiSpec` / `RoiSignal` / `run_roi_signals`, plus `RoiStatsPlan` /
`run_roi_stats` compatibility, background math, masking,
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
  Selectable to **any** column (incl. `frame_index`). *(default refined by §10.3.)*
- **y axis** — a column selector. **Default = intensity** (a sensible default counter /
  the first ROI when ROIs exist). Selectable to **any** column.
  *(default refined by §10.1–10.2 — counter priority list, never an ROI column.)*
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
- **Resolved/shipped:** background is **per-ROI** via `RoiSignal.background` and
  `RoiSignal.background_op`. A shared/global background is represented by assigning the
  same background ROI to multiple signals.

## 5. Headless vs xdart (keep xdart thin)

The ROI math stays **headless** (`xrd_tools`), per the existing ROI doc — extend it,
don't move it into the GUI:

- `core/roi.py` `RoiSpec` + reducer math (existing doc §3).
- `analysis/plans.py` `RoiSignal` + `run_roi_signals(signals, source)` →
  `AnalysisResult`; `RoiStatsPlan` + `run_roi_stats(plan, source)` remain the
  shared-config compatibility wrapper. Reducers include `mean`/`sum`/`max`/`min`/`std`,
  `background_op` includes `subtract`/`divide`, and background is per-ROI.
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
- **Mask/saturation awareness** — implemented via static masks, non-finite exclusion, and
  `mask_saturation` in the shared core ROI/reduction helpers so headless ROI stats match
  the reducer/display when the toggle is enabled.

## 7. Staged plan (each step independently testable)

The headless steps in the ROI doc §7 are implemented (`RoiSpec`+math →
`RoiSignal`/`run_roi_signals`, `RoiStatsPlan` compatibility, static/saturation masks).
The GUI steps, in order:

1. **DONE: Scan source picker + metadata table.** Source picker (loaded `.nxs` / blank →
   pick NeXus/Eiger/TIFF-seq/SPEC via `ensure_frame_source`); read metadata into the
   per-frame table. *Gate:* offscreen test — each source kind populates the expected
   columns; SPEC yields metadata + no images.
2. **DONE: The plot popup (metadata only).** x/y/normalization column selectors (defaults:
   positioner / intensity / none) + overlay + CSV, over the table from step 1, reusing
   `ParamTrendMixin`. *Gate:* synthetic table renders the expected traces; normalization
   divides; overlay shows multiple.
3. **DONE: "Plot ROI" gating + the ROI image popup.** Enable only on reachable raw; first
   image + `RectROI` ↔ numeric fields (2-way sync round-trip), multiple ROIs, stat + bg
   ROI + subtract/divide. *Gate:* sync round-trip (drag → spec → fields and back);
   gating truth-table per source kind.
4. **DONE: ROI-stat worker → columns.** Off-thread per-frame raw load + `run_roi_signals`;
   each ROI series appended as a column; plot fills incrementally with progress/cancel.
   *Gate:* end-to-end on a small real/synthetic image-series source; the incremental
   series equals a direct `run_roi_stats` (a mini live≡batch spine).
5. **(optional) persistence + live monitor** — existing ROI doc §7 steps 5–6.

## 8. Decisions captured from the 2026-06-22 discussion

- **Fold metadata + ROI into one tool** (ROI = computed column). ✓
- **Generic source picker** (NeXus/Eiger/TIFF-seq/SPEC), not nxs-bound; start on the
  loaded nxs or blank. ✓
- **Several ROIs at once** + a **general overlay** to compare multiple columns/ROIs. ✓
- **Background ROI** per signal ROI, **subtract OR divide** (op selectable). ✓
- **Disable Plot ROI** when raw isn't stored in / reachable from the source (esp. a
  processed nxs without stored raw — and *storing raw in the nxs* is a future feature). ✓
- **Selectable axes:** x default = positioner, y default = intensity, both any column;
  **+ a normalization axis** (default none, any column → y/norm). ✓

## 9. References
- [`design_roi_stats_plotting_jun2026.md`] — the headless core (RoiSpec / RoiSignal /
  run_roi_signals, plus RoiStatsPlan compatibility), background math (§6.2), masking (§6.3), persistence (§6.4),
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

## 10. Refinement backlog — 2026-06-30 GUI review

Captured from a live GUI pass (reloading `Combi4_Angledependence_samz_4p9_03271002.nxs`).
These refine the *shipped* Scan Plotter; **items 10.1–10.3 supersede the default-axis
wording in §3.** Source files:
`xdart/gui/tabs/static_scan/scan_plot_dialog.py` (the Scan Plot popup),
`roi_select_dialog.py` (the "Select ROIs" image popup),
`plot_axes.py` / `param_trend.py` (shared trend plotting),
`xdart/gui/widgets/image_widget.py` (`ImageWidget` — histogram + log),
`display_plot.py` (the Int1D/2D display — the styling + log reference).

**STATUS — all of §10 implemented on `feature/geometry` @ `8492b16` (2026-06-30).**
10.1 resolved: the `ROI1..6` are confirmed real `scan_data` counters (not pre-seeded);
per the maintainer they stay **listed/selectable**, just **never auto-defaulted**.
10.4 done as a **graft** (ceiling-safe autoscale + `ColorBarItem` + Default/Log onto the
existing viewer), NOT a `pgImageWidget` swap — that would have broken the RectROI↔field
coordinate sync.  10.6 = markers 4→7 / lines 2→3 (tunable).  A 3-lens adversarial review
caught + fixed one P2 (10.5 right-axis log mislabel).  Visual-only aspects (colorbar
render, viridis, marker size, the Folder-clip width) need an eyeball pass.

### Axis defaults + column list (refine §3)

**10.1 — ROI columns must not appear (or be defaulted) before they are computed.**
On launch the y/overlay list shows `ROI1`–`ROI6` and even defaults y to `ROI1`, before
any ROI has been computed via *Plot ROI*. No ROI-stat column should be offered in the
x / y / normalize selectors until it has actually been computed, and the default-y
heuristic must never pick an ROI column. *Open question for implementation:* are the
`ROI1..6` in the list **pre-seeded ROI slots**, or **`scan_data` counters literally
named `ROI*`** (beamline detector ROI counters carried in from the file)? If the latter,
the real defect is the §3 "first ROI when ROIs exist" name-heuristic matching metadata
columns — fixed anyway by 10.2; the question is only whether such counters should still
be *plottable* (just not the default). Resolve during implementation; either way nothing
ROI-named is offered/defaulted pre-compute. (`scan_plot_dialog.py` — column-model
population + default selection.)

**10.2 — Default y-axis = first present of a fixed counter priority list.**
On first launch pick y = the first column that exists, in order:
`Photod`, `bs`, `mon`, `i2`, `i1`, `i0`; if none present → `frame_index`.
Replaces §3's "a sensible default counter / the first ROI when ROIs exist." Explicitly
**never** an ROI column (ties to 10.1). Match the file's actual column spelling
(`scan_data` keys are case-sensitive — confirm `Photod` casing against real data).

**10.3 — Default x-axis = the NeXus positioner (scanned motor) if present, else
`frame_index`.** §3 already says "default = positioner"; pin it to the positioner
recorded in the processed `.nxs` (the scanned-motor entry / `NXdata @axes`), with
`frame_index` as the fallback when no positioner is recorded. (`io/read.py` metadata
exposes the positioner; `scan_plot_dialog.py` consumes it for the x default.)

### Viewer parity + styling

**10.4 — ROI-selection image viewer = full Image-Viewer controls.** The "Select ROIs"
popup (`roi_select_dialog.py`) must reuse the same controls as the Image Viewer display
mode: identical **autoscaling**, the **intensity (histogram) bar**, and the
**Default / Log** buttons. Today it's a bare image without them. Reuse
`xdart/gui/widgets/image_widget.py` (`ImageWidget` — histogram + `setLogMode`, ~lines
221–258) rather than a plain `ImageView`, so ROI picking sees the same levels/log the
main viewer does.

**10.5 — Log toggle for the 1D Scan Plot.** Add a **Log** button (y-axis log scale) to
the Scan Plot 1D, matching the Default/Log pattern. Reference:
`display_plot.py:757–770` (`getAxis("left").setLogMode(True/False)`).

**10.6 — Bigger markers + thicker connecting lines.** Match the marker size and line
width the Int1D 1D plots use. Scan Plot currently uses `symbolSize=4` with a thin pen
(`scan_plot_dialog.py:434`); raise to the Int1D values (`display_plot.py:1024`,
`plot_axes.py:63`, `param_trend.py:131`, and the Int1D `mkPen` width).

**10.7 — Legend font ~50% larger.** Increase the Scan Plot legend font size by ~50%
(`scan_plot_dialog.py` legend setup — `LegendItem` label text size / per-item style).

### UX

**10.8 — Esc must not close the Plot Metadata windows.** Esc currently dismisses the
Scan Plot (and the ROI-select) popup via the default `QDialog` Esc→reject. Suppress it
(override `keyPressEvent` to ignore `Qt.Key_Escape`, or don't route Esc to `reject`) so
an accidental Esc doesn't discard the assembled table / ROI setup. Applies to both
`scan_plot_dialog.py` and `roi_select_dialog.py`.

## 11. Source-panel fixes — 2026-06-30 GUI review (round 2)

`scan_source_widget.py`.  **Implemented on `feature/geometry` @ `da22887`.**

**11.1 — A picked single TIFF loaded the WHOLE folder (BUG, fixed).** `_file_candidates`
turned a picked `.tif` into `SourceSpec(parent_dir, TIFF_SERIES)` with no filter, and
`TiffSeriesSource.from_directory` globs `*.tif*` — so a folder holding several scans
(`…_03271002_*` + `…_03271005_*`) was concatenated into one bogus 22-frame series (the
metadata "from nowhere" + the frame-5→6 discontinuity in the `bs` plot were the two scan
boundaries).  Fix: filter the glob to the picked file's **scan stem** (name minus a
trailing `_<frame number>`) via the `pattern` option the registry already forwards to
`from_directory` — GUI-only, no headless change.  No frame-number suffix → fall back to
the whole folder.  The kind label now shows the **resolved** kind (`tiff_series`) so it's
clear one file pulls in the scan's frames.

**11.2 — "Folder" checkbox clipped to "Folde".** Reserve min width (font-metrics based).

**11.3 — Combine the source rows.** The "Raw params" toggle now shares the images/Repoint
row instead of taking its own (its expandable params still drop below).
