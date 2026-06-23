# Next-session prompt — finish the Scan Plotter (ROI GUI + right axis)

Paste the **PROMPT** block below into a fresh session to continue. Everything
above it is the state summary it refers to.

---

## State (branch `feature/gui-redesign-direction-a`, NOT pushed; all green)

A long session built, on top of the analyzer framework + fitting tools:
- **Scan Plot tool** (`Tools ▸ Scan Plot`, `scan_plot_dialog.py`) — step 1 done:
  source picker (NeXus/Eiger/TIFF-or-RAW/SPEC via `xrd_tools.sources`), X/Y(overlay)/
  Normalize column selectors, CSV. Metadata from `io.read_scan_data` (processed
  NeXus) or per-frame `source.metadata_for` sidecars (image series). `7b8a724`,
  `fcbb4f9`.
- **ROI-stats headless core** done (`65abee0`): `xrd_tools.core.roi`
  (`RoiSpec`, `roi_reduce`, `invalid_pixel_mask`) + `analysis/plans.py`
  (`RoiStatsPlan`/`RoiStatsResult`/`run_roi_stats`) — per-frame ROI series, per-ROI
  background **subtract** (mask-correct density; sum stays area-scaled) **or
  divide** (same-reducer ratio), mean/sum/max/min/std, R3-C masking, no-raw→NaN.
  Exported from `xrd_tools.analysis`. Tested in `tests/core/test_roi.py`.
- Design: `docs/design/design_scan_plotter_metadata_roi_jun2026.md` (GUI) +
  `design_roi_stats_plotting_jun2026.md` (headless). R3-C is **already done**
  (`ReductionPlan.mask_saturation`), so no saturation-parity blocker.

Reusable pieces already shipped: `param_trend.py ParamTrendMixin`; the
region↔fields 2-way sync in `peak_fit_dialog.py` (LinearRegionItem ↔ numeric
fields) → the 2-D `pg.RectROI` ↔ row/col fields; `analysis_worker.py`
`BatchAnalysisWorker` (off-thread, `sigFrameFit`, progress, cancel) → the
ROI-stat worker; the dialog-parameterized batch wiring in `static_scan_widget.py`
(`_on_batch_clicked(dialog)` / `_run_batch_fit(dialog)` / `_batch_dialog`); the
reachable-raw gating in `_apply_integration_control_state` (the Reintegrate gate).

Env: `conda activate xrd_test` (has lmfit; **needs `pip install -e ".[all]"` for
pymatgen** — required by the Phase Fitter). Tests:
`QT_QPA_PLATFORM=offscreen python -m pytest tests/xdart` (offscreen GUI; an
occasional pyqtgraph teardown SIGSEGV is a known flake — just rerun) and
`python -m pytest tests/core`. Launch the app with the `xdart` console command;
quit + relaunch to pick up edits.

---

## PROMPT

> Continue the Scan Plotter on branch `feature/gui-redesign-direction-a` (xrd-tools
> monorepo, `~/repos/xrd-tools`; activate `xrd_test`). The metadata-plotting half
> and the ROI-stats **headless** core (`xrd_tools.core.roi` + `run_roi_stats`) are
> done and tested. Build the remaining two features, committing + testing each
> (offscreen GUI tests; rerun on the known pyqtgraph teardown segfault):
>
> 1. **ROI GUI in `ScanPlotDialog`.** Add a "Plot ROI…" button, **enabled only
>    when the source's raw frames are reachable** (Eiger/TIFF series: yes; SPEC or
>    a processed NeXus without stored/reachable raw: disabled — reuse the
>    reachable-raw logic the Reintegrate gate uses). Clicking it opens an image
>    popup on the scan's first frame with: one or more draggable `pg.RectROI`s,
>    each two-way synced to row/col center+width numeric fields (mirror the
>    LinearRegionItem↔fields sync in `peak_fit_dialog.py`); a reducer selector
>    (sum/mean/max/min/std); and an optional per-ROI background ROI + op
>    (subtract/divide). On "compute", run an **off-thread worker** (mirror
>    `BatchAnalysisWorker`) that calls `xrd_tools.analysis.run_roi_stats(plan,
>    source)` and streams per-frame results; **append each ROI's series as a
>    column** in the ScanPlotDialog table so it plots/overlays/normalizes/exports
>    exactly like a metadata column. Keep the ROI math headless (it already is);
>    xdart is the thin popup + worker. Add GUI tests (gating truth-table per
>    source kind; RectROI↔fields round-trip; an end-to-end that the worker fills
>    ROI columns — drive it with a synthetic `MemoryFrameSource`-style source).
>
> 2. **Cross-family right-hand axis.** Give `ScanPlotDialog`'s overlay a second
>    (and ideally third) right-hand Y axis so columns of very different magnitude
>    (e.g. peak center ~2 vs amplitude ~1e5, or temperature vs an ROI sum) read
>    cleanly — the standard pyqtgraph linked-`ViewBox` pattern. Add a per-series
>    "axis" choice (left/right). Then bring the same right-axis option to the
>    fitting trend (`ParamTrendMixin` / the peak+phase dialogs); note the trend's
>    Y selector is currently single-family, so extend it to allow a second family
>    on the right axis.
>
> Before starting, skim `docs/design/design_scan_plotter_metadata_roi_jun2026.md`
> (§4 ROI, §3 axes) and `scan_plotter_next_session.md`. Lower-priority tracked
> debt you can fold in if convenient: extract a `BaseAnalysisDialog` for the
> peak/phase/scan dialogs' shared scaffolding (the health review flagged ~150 dup
> lines); refine `_guess_axes` positioner detection.
