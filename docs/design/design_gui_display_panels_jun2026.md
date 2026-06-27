# Stitch / RSM display panels — design

**Status:** design (Jun 2026), ready for UI mockups → Qt implementation. The third UI-spec doc:
- `design_gui_three_section_layout_jun2026.md` — the CONTROLS (right column: data / experimental / processing).
- `design_gui_int_migration_jun2026.md` — migrating Int-1D/2D onto that layout.
- **this** — the DISPLAY panels (main view) for the new Stitch + RSM tools.

Recon: `ws0zt0lvg` (display-layer API + Int-2D panels + persistence/frames + the RSM notebook + 3D options).

## The principle: the display layer was built for this

`display_logic.py` already anticipates these three cases — **arrangement is data**, not render-branching:
- `DisplayState` carries `panels` (keyed `(PanelKey, PanelPlan)`) + `layout` (rows of `PanelKey`s)
  (`display_logic.py:554-592`); the docstring names *"Stitch-2D: cake / plot"* and *"RSM: a 2×3 grid
  of repeated SLICE_2D/PROJ_1D roles"* as target arrangements.
- `PanelRole` (`:236-251`) already declares `STITCH_2D`, `SLICE_2D` ("repeats"), `PROJ_1D` ("repeats"),
  `RESULTS` as *reserved*; `PanelKey.instance` (`:275`) exists to disambiguate repeated roles
  (RSM's HK/HL/KL slices, H/K/L projections).
- `render_keys_for_state` + `RenderPlan.draw_keys/clear_keys` (`:1383, 1488`) already plan at
  **PanelKey granularity without role-dedupe** — explicitly "so a repeated-role layout (a future
  RSM/Stitch viewer) keeps every instance."
- New modules add a controller via `register_controller` (`:623`); the core never imports them (§10 seam 3).

So both tools are **new layout descriptors + new controllers**, not a new framework. The **only**
deferred gap is the widget *execution* level (the WS-X2 TODO, `:1359`): the renderer + widget have
draw delegates for `RAW_2D`/`CAKE_2D`/`PLOT_1D` only.

---

## 1. Stitch display — pure reuse

**Layouts (data):** Stitch-2D = `((stitch_key,), (plot_key,))` — cake on top, 1D below; Stitch-1D =
`((plot_key,),)`. Reuse the Int **cake** panel (`binned_widget`, `display_frame_widget.py:731`) +
the **1D plot** (`self.plot`, `:741`). **DROP the raw inline panel** (`image_widget`, `:724`) — a
merged stitch has no single raw frame; `render_plan` already clears panels the state omits.

**Controls reuse the existing split** (`_set_2d_controls_visible` / `_set_middle_1d_controls_visible`,
`display_frame_widget.py:520-532`): keep scale/Log/cmap, `plotUnit`/`plotMethod`
(Single/Overlay/Waterfall/Sum/Avg)/legend; the 2D-only controls (`imageUnit`, `shareAxis`, slice
trio) show for Stitch-2D, hide for Stitch-1D. **Show the `Raw` button** (`_showImageBtn`, `:835`) in
both.

**`StitchDisplayController`** — subclass `_BaseController` (`display_controllers.py:222`), register
`register_controller(Mode.STITCH_VIEWER, StitchDisplayController())` next to
`register_default_controllers` (`:583`). It reads the in-memory/reloaded stitch result, sets
`has_2d = stitched_2d is not None`, emits the 2 panels + the collapsing layout, is generation-stamped,
and **does NOT consult `scan.frames` or the integration-unit combo** (it's a result viewer). The 1D
curve flows through the layered `Trace` model (overlay across runs = append `data` traces, no render
change). One **Mode** with a `has_2d` collapse (reuse the existing `skip_2d` 1D-collapse path,
`display_frame_widget.py:2372`) is preferred over two Modes — see Open Q1.

### 1a. The raw-image popup (a *contributing-frame* picker)

**Semantics:** a stitched q-point is the merge of *many* contributing frames, so the popup is **not**
"the raw for this q" — it's **"pick one contributing frame and view its raw detector image."** The
**h5viewer Frames panel** is the picker.

**Selection — zero new plumbing:** `H5Viewer.listData` (the Frames panel, `h5viewer.py:2109`) already
rewrites the **shared** `frame_ids` → `sigUpdate` → `staticWidget.set_data` → `displayframe.update()`.
**Trigger:** reuse `_showImageBtn` + `_show_image_preview` (`display_frame_widget.py:835, 3628`).
**Load:** route through `ImageViewerController`'s ssrl loaders (`classify_image_source` /
`load_image_frame`, `display_controllers.py:305-325`) so it shows the genuine per-frame raw image
resolved from the source ref (falling back to the thumbnail).

### 1b. Persistence enabler — **the one real gap (T1)**

`write_stitched` (`nexus.py:1889`) writes only `/entry/stitched_*` + the provenance blob — **no
frame records, no `@source_base`**; `StitchPlan.provenance()` records only integer frame labels (no
source paths). So a reloaded stitch `.nxs` has nothing to resolve a contributing raw frame today.

**The read side needs nothing** (`get_raw_frame`/`classify_image_source` are source-ref-driven). **The
write side** must, per contributing label, call `stamp_source_base` + `ensure_frames_container` +
`write_frame_record` (`nexus_record.py:127/174/207`) — harvesting `source_path`/`source_frame_index`
from the `Frame`s `run_stitch` iterates (`core/scan.py:151`); template = `NexusSink._write_frame_record`
(`reduction/core.py:559`). **Caveat:** the `frame_key` must equal the label the Frames panel requests
(`get_raw_frame` formats `frame_{idx:04d}`, `read.py:492`).

---

## 2. RSM display — the WS-X2 trigger + the new slider state

**Layout (2×3, data):**
```
((SLICE_2D 'HK', SLICE_2D 'HL', SLICE_2D 'KL'),   # 3 orthogonal 2D slices
 (PROJ_1D  'H',  PROJ_1D  'K',  PROJ_1D  'L'))     # 3 1D projections, x-shared per column
```
Roles + `PanelKey.instance` exist; this is **the case that REQUIRES the WS-X2 promotion**.

### 2a. WS-X2 render-core work (the genuinely-new pure-core edit)

The decision half is done (`render_keys_for_state`/`draw_keys`/`clear_keys` plan per-PanelKey). The
execution half is not (`render_display` consumes role-level `plan.draw`/`clear`; the widget has one
cake + one plot). To finish (`display_logic.py:1359` TODO):
1. Drive the `render_display` loop from `plan.draw_keys`/`clear_keys` (additive; keep role-level as
   the legacy fallback).
2. Make the delegates **per-PanelKey** (`(role, instance)`), `display_frame_widget.py:1350-1564`.
3. A **standalone `rsm_panel_widget.py`** (3 `pgImageWidget` + 3 `pg.PlotItem` keyed by instance +
   the slider row) swapped into `twoDWindow`/`plotWindow` in `RSM_VIEWER` mode — *not* shoehorned
   into the Int splitter (keeps the Int path untouched).
4. `DisplayPayload` (3 slots today) gains an **optional `panels: dict[PanelKey, ImagePayload|PlotPayload]`**
   field; Int/Stitch keep the 3 legacy slots, repeated-role viewers use the keyed map.

### 2b. NEW slider display-state (`RSMViewState`)

The notebook's three sliders are **integration bands, not slice indices** (`mask_data` → crop to an
HKL box → integrate). Headless math already exists: `RSMVolume.get_slice(axis, band)` /
`line_cut(axis, fixed_ranges)` / `get_bounds()` (`volume.py:95/136/69`). Model the bands as
**immutable, generation-stamped view-state**:
```
@dataclass(frozen=True)
class RSMViewState:
    h_band, k_band, l_band: tuple[float, float]   # the 3 FloatRangeSliders
    log: bool; clim_pct: tuple[float, float]; cmap: str
```
Slider change → bump `display_generation` → `update()` → `RSMDisplayController.compute_state` carries
the new `RSMViewState` → `build_payload` calls `get_slice`/`line_cut` → 6 payloads → render. **The
generation stamp is the safety guard** (`render_plan` drops a payload whose generation ≠ the state's,
`:1500`) — a slow recompute from a superseded band can't paint over a newer one (reuse the Int load
worker's staleness mechanism verbatim).

> **Round-trip nuance:** the notebook integrates with `nanmean`; headless `extract_2d_slice`/
> `extract_line_cut` use `nansum` (`volume.py:317`). Pick **`nansum`** for the GUI (matches the
> headless round-trip; label the colorbar accordingly) so reload-equivalence holds — Open Q2.

**`RSMDisplayController`** copies the read-only `NexusViewerController` pattern; holds the loaded
`RSMVolume` + `RSMViewState`; emits the 6 `PanelPlan`s + the 2×3 layout; never consults
`scan.frames`/the unit combo; `build_payload` returns the keyed `panels` map.

### 2c. The raw-image popup (RSM has it too)

RSM also gets the **contributing-frame raw popup** — the *same* mechanism as Stitch §1a (the
h5viewer Frames panel picks a contributing frame → `ImageViewerController` ssrl loaders → a raw
image dialog). RSM thus has **two** popups: the raw-frame viewer (this) and the 3D view (§2d). The
enabler is the same frame-record persistence (now standard for both writers — §1b / §4-T1/T2) with
the multi-scan naming below.

### 2d. The 3D popup (the other genuinely-new piece)

A **standalone, view-only** rotatable/zoomable window — **not** in the display registry (no Mode, no
controller, no generation state). **`PyOpenGL` is NOT installed in `xrd_test`** → add it as an
optional extra (`[project.optional-dependencies] gl = ["PyOpenGL"]`); **disable the 3D button with a
tooltip when GL import fails** (the 2D/1D viewer degrades gracefully).

**Recommended v1 backend: `GLScatterPlotItem`** on thresholded finite voxels (a ~101³ ≈ 1M-voxel
volume; scattering only the bright N points is trivially interactive). `GLViewWidget` gives
rotate/zoom/pan free. Alternatives: `pg.isosurface`→`GLMeshItem` (a Bragg iso-shell; needs an
iso-level control); `GLVolumeItem` (best-looking but GPU-fill-bound; downsample to ~64-100³).
**Minimal controls:** colormap, threshold(scatter)/iso-level slider, log, downsample factor. No
slicing (that's the 2D/1D panels). Several details still need planning — see Open Qs 4-5.

---

## 2e. Multi-scan grouping — frame records + the Frames-panel naming

A single Stitch/RSM can **group several scans** (RSM already does via `grid_scans_streaming` over a
list of `ScanInput`; Stitch needs the same multi-source path). The Frames panel must then
**disambiguate which scan** each contributing frame came from — Vivek's convention: prefix the scan
number, so grouping scans (5, 7, 8) lists `5-1, 5-2, … 7-1, 7-2, … 8-1, …`.

**Is this consistent with the current storage? — Not yet; it's a small extension.** Today the frame
records are **flat + per-scan**: `NexusSink._write_frame_record` writes `/entry/frames/frame_{index:04d}`
(`reduction/core.py:559`), and `get_raw_frame(frame: int)` reads by that **int** index (`read.py:492`).
Grouped scans **collide** (every scan has a `frame_0001`). But `write_frame_record(frames_grp,
frame_key, …)` takes a free **string** key (`nexus_record.py:207`) and the scan number is available
(the source `scan_id`/`name`, or `get_scan_path_info`), so it's an additive extension:

- **Storage:** nested per-scan groups for grouped runs — `/entry/frames/scan_5/frame_0001`,
  `…/scan_7/frame_0001` — keeping the flat `/entry/frames/frame_NNNN` for **single-scan**
  (backward-compatible). (Alternative: flat encoded keys `frame_5-0001`; nesting is cleaner + groups
  naturally.) Each record still carries its `source/{path,frame_index}` (the raw-resolution pointer).
- **Frames-panel display:** flatten to `"<scan>-<frame>"` labels (`5-1`, `5-2`, …) — a presentation
  layer over the stored structure (a flat list, or a scan→frames tree).
- **Read path:** generalize `get_raw_frame` from a bare `int` to a `(scan, frame)` address (stays
  int-keyed for single-scan); the multi-scan writer tags each frame with its scan number.

This is the shared enabler for **both** raw popups (Stitch §1a, RSM §2c). **Open Q7:** nested groups
(recommended) vs flat encoded keys; and confirm the multi-scan Stitch path exists/lands (RSM's does).

---

## 3. Shared + "free vs new"

Both popups are **standalone `QDialog`s outside the display registry** (no Mode/controller/render-plan).
Each new controller self-registers alongside `register_default_controllers`; the dispatch core never
branches on mode or imports these modules (§10 seam 3 holds).

| | Stitch | RSM |
|---|---|---|
| New `PanelRole`? | No (`STITCH_2D`+`PLOT_1D` exist) | No (`SLICE_2D`/`PROJ_1D` exist) |
| New Mode + layout? | Yes (`STITCH_VIEWER`, +`imageFrame_w` field) | Yes (`RSM_VIEWER`, 2×3) |
| New controller? | Yes | Yes |
| Slider/view state? | No | **Yes** (`RSMViewState`) |
| Render-core change? | No (role-level suffices) | **Yes** (WS-X2: draw_keys/clear_keys, per-key delegates, 6-panel scaffold, keyed payload) |
| Raw-image popup? | **Yes** (contributing-frame picker) | **Yes** (same mechanism) + the 3D view |
| 3D viewer? | n/a | **Yes** (standalone GL dialog + PyOpenGL extra) |
| Frame records on `.nxs`? | **Yes — standard** (scan-tagged for multi-scan) | **Yes — standard** (scan-tagged) |
| Free from the layer | panels/layout/PanelKey, compute_display_state, render_plan, registry, `_BaseController`, Image/Plot/Trace payloads, the raw-popup pattern, the shared `frame_ids`→`sigUpdate` selection | the same data/decision spine + per-PanelKey *planning* + the `get_slice`/`line_cut`/`get_bounds` headless math |

**One line:** Stitch = pure reuse (new Mode + controller + 2 trivial edits); RSM reuses the spine
fully but is the trigger to finish the deferred WS-X2 render-execution half + add the slider state +
the 3D dialog.

---

## 4. Open questions, persistence TODOs, sequenced plan

**Open questions (decide before/at the mockup):**
1. Stitch: one `STITCH_VIEWER` Mode + `has_2d` collapse (recommended) vs two Modes.
2. RSM integration op: **`nansum`** (recommended, round-trips) vs `nanmean` (notebook).
3. `DisplayPayload` for 6 panels: optional keyed `panels` field (recommended) vs controller bypass.
4. 3D v1 backend: **scatter** (recommended) vs iso vs volume.
5. 3D honors current bands (`crop`) vs always full volume (recommended: full, decoupled, v1).
6. RSM 6-panel: standalone `rsm_panel_widget` (recommended) vs extend the splitter.
7. Multi-scan frame storage: nested `scan_<N>/frame_NNNN` (recommended) vs flat `frame_<N>-<idx>`
   keys; + confirm/land the multi-scan **Stitch** path (RSM's `grid_scans_streaming` exists).

**Persistence TODOs (write side) — STANDARD for BOTH writers (Vivek, Jun 2026):**
- **T1 (Stitch) + T2 (RSM):** `write_stitched` AND `write_rsm` write per-frame records (the
  raw-popup enabler — RSM has the raw popup too) — `stamp_source_base` + `ensure_frames_container`
  + `write_frame_record`, harvesting `source_path`/`source_frame_index` from the `Frame`s the run
  iterates (template `NexusSink._write_frame_record`).
- **T-multiscan:** for grouped scans the `frame_key` is **scan-tagged** (`scan_<N>/frame_NNNN`),
  and `get_raw_frame` is generalized to a `(scan, frame)` address; the Frames panel shows
  `"<scan>-<frame>"`. The writer carries the scan number (source `scan_id` / `get_scan_path_info`).
- **T3:** the `ssrl_xrd_tools>=` floor caveat (CLAUDE.md north-star) applies if these primitives move.

**Sequenced plan (each live-gated in `xrd_test`):**
1. **Stitch-1D** — smallest: `STITCH_VIEWER` + an INT_1D-shaped layout + `StitchDisplayController`
   (PLOT_1D only) + register. No render-core change.
2. **Stitch-2D** — add `imageFrame_w` to `PanelLayout`; `STITCH_2D→binned_widget` delegate; emit
   `STITCH_2D` in the `cake_image` slot.
3. **Frame records (both writers) + the multi-scan scheme** — add frame-record writing to
   `write_stitched` AND `write_rsm`, scan-tagged (`scan_<N>/frame_NNNN`); generalize `get_raw_frame`
   to `(scan, frame)`. The headless persistence enabler for both raw popups; round-trip tested.
4. **Stitch raw popup** — reuse `_show_image_preview`, reroute load to the ssrl loaders; Frames panel
   shows `"<scan>-<frame>"`; verify it resolves a contributing frame from a freshly-saved `.nxs`.
5. **WS-X2 render-core promotion** — drive `render_display` from `draw_keys`/`clear_keys`;
   per-PanelKey delegates; keyed `DisplayPayload`. Additive (keep the role-level path); unit-test
   headlessly (`pytest -m display_logic`). **Keep the live≡batch≡reload equivalence spine green.**
6. **RSM 6-panel viewer** — `RSM_VIEWER` + 2×3 layout + `RSMDisplayController` + `RSMViewState`
   sliders + the standalone 6-panel widget; wire `get_slice`/`line_cut`/`get_bounds`. + the **raw
   popup** (reuses step 3/4).
7. **RSM 3D popup** — add the `PyOpenGL` extra; a standalone `GLViewWidget` + `GLScatterPlotItem`
   dialog with the 4 controls; disable-with-tooltip when GL is unavailable.

**Risk:** step 3 (frame records + multi-scan addressing) is headless + testable; steps 1-2/4 are
low-risk reuse; **step 5 touches the pure render core** — land it additively (behind the existing
role-level path) before flipping the renderer, keep the equivalence spine green. Steps 6-7 are
net-new and live-gated.

## Key anchors
`display_logic.py`: `:236` roles, `:275` PanelKey, `:554` DisplayState, `:595` DisplayPayload (+ keyed
`panels`), `:1359` WS-X2 TODO, `:1383` key-planning, `:1492` render_plan. `display_controllers.py`:
`:222` `_BaseController`, `:297` ImageViewerController, `:583` register. `display_frame_widget.py`:
`:520-532` control split, `:724/731/741` raw/cake/plot widgets, `:1350-1564` delegates, `:2329`
`_apply_layout`, `:3628` `_show_image_preview`. `h5viewer.py:2109` Frames `listData`;
`static_scan_widget.py:858` `sigUpdate`→`set_data`. `volume.py:69/95/136` RSMVolume math.
`nexus.py:1889/1952` write_stitched/write_rsm (frame-record gap); `nexus_record.py:127/174/207`;
`read.py:492` get_raw_frame.
