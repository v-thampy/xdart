# Design: ROI statistics plotting + general `scan_data` popup plotter

> **2026-06-22 update — folded into the unified Scan Plotter.** The GUI framing
> (one combined metadata + ROI tool, generic source picker, multi-ROI overlay,
> subtract-**or-divide** background, a **normalization axis**, and reuse of the
> now-shipped `ParamTrendMixin` / `BatchAnalysisWorker` / reachable-raw gating)
> lives in [`design_scan_plotter_metadata_roi_jun2026.md`]. **This doc remains the
> source of truth for the HEADLESS layer** (`RoiSpec` / `RoiStatsPlan` /
> `run_roi_stats`, background math §6.2, masking prereq §6.3, persistence §6.4,
> live monitor §6.1, step sequence §7). Refinements to apply here when built:
> the reducer gains `max`/`min`/`std`; the background op gains **`divide`** (not
> only subtract); per-ROI background (vs the single global background assumed
> below) is an open call (lean per-ROI). The normalization axis (y / column) is a
> plot-time concern in the GUI, not a `RoiStatsPlan` field.

**Status:** draft for discussion · 2026-06-14 (updated 2026-06-22) · planning only (no code)
**Gated on:** 3e+Phase-5 (one store / `FrameRecord`) done + tested. The live ROI
monitor reads the *same* raw projection the 2D panel maintains, so it must not be
designed against a moving store.
**Depends on:** N1 raw resolution (✓ shipped), the analysis-plan seam (✓
`analysis/plans.py`), `scan_data` (✓ persisted + readable), and — for mask-aware
stats — **R3-C "invalid-pixel policy into core"** (OPEN; see §6.3).
**Pairs with:** [`design_diffractometer_geometry_jun2026.md`] (unrelated geometry,
but same "headless-first, thin xdart" discipline).

---

## 1. What the user asked for

Plot statistics of rectangular ROIs selected on a scan's image, as a **series over
the scan** (memory `planned_features_roi_and_stitching_jun2026`):

- Up to **5 ROIs**, one of which defaults to the **entire frame**. Plus an optional
  **background ROI** whose stat is **subtracted** from each ROI's stat.
- ROIs set via a **numeric popup** (`center_x, center_y, width_x, width_y`) **and** by
  **mouse** on the image (pyqtgraph ROI items), **two-way synced**.
- For a scan, plot the **mean OR sum** of each ROI **vs the scanned motor — or any
  other `scan_data` column** (x-axis user-selectable), background-subtracted.
- **Generalize (if clean):** a reusable **"plot `scan_data` in a separate popup
  window"** — y = any `scan_data` column OR a derived ROI series; x = any `scan_data`
  column. ROI stats become **one producer** feeding the general popup.

---

## 2. Decomposition (headless-first / thin xdart)

```
HEADLESS  (xrd_tools, fully testable with synthetic images + scan_data)
  core/roi.py            RoiSpec  (rect in detector-pixel coords) + reducer math
  analysis/plans.py      RoiStatsPlan + run_roi_stats(plan, source) -> AnalysisResult
                         (mirrors run_stitch / run_rsm — operates on a FrameSource)
THIN xdart (Qt only)
  scan_data popup        a standalone QDialog plot window (NOT a display Mode)
  ROI items              pyqtgraph RectROI on the image, two-way synced to the popup
  live ROI monitor       OPTIONAL consumer of the live raw projection (additive)
```

**Key boundary call:** the popup plotter is a **standalone dialog/window, NOT a main
display Mode** — so it is **not** a `register_controller` `PanelKey`/`Mode` (unlike the
Stitch/RSM viewers, which *are* display modes). It reuses the existing display
machinery for nothing; it is a plain dialog over headless arrays. (Memory pins this
distinction.)

---

## 3. Headless core API (the seam to mirror `run_stitch`/`run_rsm`)

```python
# core/roi.py
@dataclass(frozen=True, slots=True)
class RoiSpec:
    center_x: float            # detector pixel coords on the RAW frame
    center_y: float
    width_x: float
    width_y: float
    name: str = ""             # series label; "" -> auto ("roi0", ...)

    @classmethod
    def full_frame(cls, name="full") -> "RoiSpec": ...   # the default ROI
    def pixel_slice(self, image_shape) -> tuple[slice, slice]: ...   # clamped to bounds

# analysis/plans.py  (alongside StitchPlan/RSMPlan/PeakFitPlan)
@dataclass(frozen=True, slots=True)
class RoiStatsPlan:
    rois: tuple[RoiSpec, ...] = ()          # <=5; if empty, run inserts RoiSpec.full_frame()
    background: RoiSpec | None = None
    reducer: str = "mean"                   # "mean" | "sum"
    x_key: str | None = None                # scan_data column for x; None -> frame label
    mask: MaskSpec | None = None            # static detector mask (core.scan.MaskSpec)
    invalid_policy: InvalidPixelPolicy | None = None   # saturation/dead px (R3-C; §6.3)
    require_raw: bool = True                 # strict raw; never compute stats off a thumbnail
    frame_indices: tuple[int, ...] | None = None

@dataclass(frozen=True, slots=True)
class RoiStatsResult:
    x: np.ndarray                            # aligned x values (or frame labels)
    x_label: str
    series: Mapping[str, np.ndarray]         # roi name -> bkg-subtracted stat per frame
    frames: np.ndarray                       # frame labels
    valid_counts: Mapping[str, np.ndarray]   # per-roi valid-pixel count per frame
    diagnostics: Mapping[str, Any]           # e.g. frames where raw was unresolvable

def run_roi_stats(plan: RoiStatsPlan, source: FrameSource, *,
                  frame_indices=None) -> AnalysisResult:
    """For each frame: load RAW (strict), apply mask + invalid-pixel policy, reduce
    each ROI over VALID pixels, subtract the background density, append to the series.
    x = scan_data[plan.x_key] aligned to frame labels (else the labels themselves).
    Returns AnalysisResult(kind="roi_stats", payload=RoiStatsResult, provenance=...)."""
```

- Operates on any `FrameSource` (`core.scan.FrameSource`): a reloaded `ProcessedScan`,
  a `Scan`, an Eiger/Tiff source, or a just-finished batch's source. `load_frame` gives
  raw; `_scan_data_for_frames` / `get_metadata["scan_data"]` gives x.
- Reuses `analysis/plans.py` helpers: `_metadata_series`-style alignment + the
  `AnalysisResult`/`_json_safe` envelope already there.
- **Fully headless-testable:** synthetic `(N, H, W)` image stack + a synthetic
  `scan_data` dict → assert per-ROI series, background subtraction, mask exclusion,
  and `x` alignment, with no GUI and no monkeypatching.

---

## 4. The general `scan_data` popup plotter (the generalization)

Build the general plotter as the **primary** deliverable; ROI stats is one producer.

- **Model:** the popup takes `x: {label -> value}` and one or more
  `y: {label -> value}` named series, plus axis labels. Nothing XRD-specific.
- **`scan_data` is already fully available:** `get_metadata` returns `scan_data` (ALL
  per-frame columns: motors AND counters) + `positioners` (geometry motors);
  `_scan_data_for_frames` aligns to frame labels (`io/read.py`). So a "plot column Y vs
  column X" needs no new headless code — the popup reads `scan.scan_data`.
- **ROI series plug in identically:** `run_roi_stats(...).series[name]` is a
  `{label -> value}`-shaped derived column. The popup treats it like any `scan_data`
  column. So the seam is: **x = any `scan_data` column; y = any `scan_data` column OR a
  named ROI series.**
- **UI:** x-axis dropdown (every `scan_data`/positioner column + `frame_index`); y
  multi-select (columns and/or ROI series); reducer + background toggles when an ROI
  series is selected; live-update when ROIs move (re-call `run_roi_stats`).
- **One window serves both** (open question 5, resolved): the ROI-specific plot is the
  general popup with its y pre-bound to ROI series; there is no second window.

---

## 5. Where ROIs are defined (coordinate space)

- **v1 = detector-pixel ROIs on the RAW frame** (`center_x/y, width_x/y` in pixels),
  matching "rectangular ROIs selected on a scan's image." `RoiSpec.pixel_slice(shape)`
  clamps to bounds; the full-frame default is the whole detector.
- **Flagged future extension:** ROIs on the cake / q-χ image (q-space ROIs) — out of
  scope for v1; the `RoiSpec` would gain a `space` discriminator and `run_roi_stats`
  would read `results_2d` instead of raw. Note in the doc, don't build it.
- **Two-way sync (xdart):** the numeric popup fields and the pyqtgraph `RectROI` item
  serialize to/from the *same* `RoiSpec` value object — that shared dataclass is what
  makes the sync trivial and keeps the GUI thin.

---

## 6. Open design questions — resolved or flagged

### 6.1 Raw dependency: live vs reload — **resolved by splitting two clean paths**
ROI stats need the raw frame per index. ADR-0003 `FrameEvent` carries integration
*results*, not raw; `FrameRecord` (`core/frame_view.py:476`) carries per-mode
`FrameView`s, not a raw image. So ROI compute cannot ride the event/record.

- **(i) On-demand over a completed/reloaded scan — the PRIMARY path.**
  `run_roi_stats(plan, source)` loads raw via `FrameSource.load_frame` (N1 resolution
  for a reloaded `ProcessedScan`). **Strict raw** (`allow_thumbnail=False`): a
  thumbnail is downsampled + quantized, so ROI stats off it would be wrong. If a
  frame's raw is unresolvable (moved tree, no `source_root`), record **NaN** for that
  frame + a `diagnostics` entry — **warn, don't crash** (the same "design the no-raw
  case" rule round-8 set for mode-switch on a reloaded file). This path is fully
  headless and fully tested.
- **(ii) Live ROI monitor — OPTIONAL, additive, deferred to a later step.** Very
  useful at the beamline. It is **not** worker-side precompute (ROIs are interactive
  and *retroactive* — the user draws an ROI at frame 50 and wants frames 1–49 too, so
  you cannot precompute at integration time without bloating every event). Instead it
  is a **consumer of the live raw projection** that ADR-0005 already keeps GUI-side
  (the "bounded raw-image window for the 2D panel"): on each `FrameEvent`, read that
  frame's raw from the bounded window, reduce the ROIs, append to an in-memory series;
  frames older than the window are filled on demand by calling `run_roi_stats` over the
  persisted prefix. **The headless function is the source of truth; the monitor is an
  incremental cache that must agree with it** (a small live≡reload spine for ROI
  series).
- **Forward constraint (record/store design):** keep **raw-per-frame reachable through
  the finalized record/store** — ROI + the live monitor need `map_raw`; do not
  architect raw away while collapsing the store (memory note + round-8 forward
  watch-item).

### 6.2 Background subtraction normalization — **resolved (one mask-correct rule)**
Define subtraction in terms of **background mean-per-VALID-pixel** (`bkg_density`),
which is correct for both reducers and for masked pixels:

```
bkg_density   = nansum(bkg_valid) / count(bkg_valid)
mean reducer: stat = nanmean(roi_valid)            - bkg_density
sum  reducer: stat = nansum(roi_valid)             - bkg_density * count(roi_valid)
```

This makes `sum` area-scaled automatically (it subtracts background *per signal
pixel*, not the raw background sum), and uses valid-pixel counts everywhere so masked/
dead/saturated pixels never distort either the signal or the background. `mean` is
area-independent as expected.

### 6.3 Mask / saturation awareness — **resolved, with a hard prerequisite**
ROI mean/sum must exclude masked, dead, and saturated pixels. The invalid-pixel
policy must be the **same** one the reducer uses, or headless ROI stats and xdart's
display would disagree (a live≡reload violation). Today that policy is **xdart-only**:
memory's R3-C item ("expose invalid-pixel/saturation policy on `ReductionPlan`/
`MaskSpec` in core; the headless reducer doesn't call `core/invalid.py`") is **still
open**. Therefore:

- **R3-C-into-core is a prerequisite** for mask-aware ROI stats. `RoiStatsPlan` carries
  a `MaskSpec` (static detector mask, `core.scan.MaskSpec`) + an `invalid_policy`
  (saturation/NaN), and `run_roi_stats` applies the **same core helper** the reducer
  applies. Sequence R3-C first (it is orthogonal, small, and benefits reduction too).
- Until R3-C lands, ship ROI stats with **static-mask + NaN exclusion only** and a
  documented caveat that saturation masking matches the reducer **only after** R3-C.

### 6.4 Persistence — **resolved: on-demand by default, optional capability-gated save**
- **Default: compute on demand.** ROIs are interactive/exploratory; persisting every
  ROI a user drags would pollute the file. Recompute is cheap (one rectangular
  reduction per frame).
- **Optional persist** of a *named* ROI-series set as an additive
  `roi_stats/<name>/` group (schema-as-code: `GroupSchema` + a `roi_stats`
  `CapabilityAttr`, presence-detected, never rename), for archival + headless reuse —
  important when **raw is later unavailable** (moved tree) and recompute is impossible.
  Persisted series store the `RoiSpec` set + reducer + background as provenance, so
  they are reproducible, plus an inputs hash so a reader can flag staleness vs current
  raw.

### 6.5 One window for both — **resolved** (see §4): the ROI plot *is* the general
popup with y pre-bound to ROI series.

---

## 7. Gated step sequence (each step independently testable; gates front-loaded)

> All steps land **after** 3e+Phase-5 is done + tested.

0. **`core/roi.py`: `RoiSpec` + reducer math.** Rect → clamped pixel slice; mean/sum
   over valid pixels; `bkg_density` subtraction (§6.2). **Gate:** unit tests on
   synthetic arrays incl. clamping at edges, full-frame default, masked-pixel
   exclusion, area-scaled sum.
1. **(prereq) R3-C invalid-pixel policy into core.** Expose `invalid_policy` on
   `ReductionPlan`/`MaskSpec`; headless reducer calls the core helper; xdart delegates.
   **Gate:** headless saturation-masking test; xdart and headless agree on the same
   frame. *(Orthogonal — can proceed in parallel.)*
2. **`analysis/plans.py`: `RoiStatsPlan` + `run_roi_stats`.** Reload/completed-scan
   path; strict raw; NaN + diagnostic on no-raw; x alignment from `scan_data`. **Gate:**
   synthetic stack + synthetic `scan_data` → assert series, background subtraction,
   mask exclusion, x alignment, and the no-raw NaN path; `AnalysisResult` JSON-safe.
3. **xdart: general `scan_data` popup dialog.** Standalone QDialog; x dropdown + y
   multi-select over `scan.scan_data`/positioners/`frame_index`; pyqtgraph plot. **Gate:**
   offscreen test feeding a synthetic `scan_data` dict renders the expected traces.
4. **xdart: ROI items + two-way numeric/mouse sync.** pyqtgraph `RectROI` ↔ numeric
   fields ↔ `RoiSpec`; ≤5 ROIs + a background ROI; "plot" calls `run_roi_stats` and
   feeds the popup as derived y-series. **Gate:** sync round-trip test (mouse-drag →
   `RoiSpec` → fields and back); end-to-end on a reloaded scan.
5. **(optional) Persistence.** `roi_stats/<name>/` group + `roi_stats` capability;
   write/read; provenance + staleness hash. **Gate:** write→read round-trip; capability
   feature-detect; back-compat (absent → recompute path).
6. **(optional) Live ROI monitor.** Consumer of the bounded raw projection; on-demand
   backfill via `run_roi_stats` over the persisted prefix. **Gate:** a live≡on-demand
   mini-spine — the monitor's incremental series equals `run_roi_stats` over the same
   frames.

---

## 8. References
- Code: `analysis/plans.py` (the `run_*` seam + `AnalysisResult`/`_metadata_series`/
  `_json_safe`), `io/read.py` (`get_metadata["scan_data"]`, `_scan_data_for_frames`,
  `get_raw_frame`, `ProcessedScan.load_frame`/`scan_data`), `core/scan.py`
  (`FrameSource`, `MaskSpec`), `core/frame_view.py:476` (`FrameRecord` — carries no
  raw, why ROI can't ride it), `io/schema.py` (`GroupSchema`/`CapabilityAttr` for
  optional persistence).
- Decisions: ADR-0003 (event carries results not raw), ADR-0004 (event/thread),
  ADR-0005 (bounded raw projection is GUI-side, derived), ADR-0006 (the prepare/
  capability pattern, for reference).
- Memory: `planned_features_roi_and_stitching_jun2026`, `keep_xdart_thin`,
  `display_modules_layouts_jun2026`, and the round-8 "design the no-raw case" note in
  `xrd_tools_monorepo_push_jun2026`.
