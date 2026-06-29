# Persistent stitch display + mask + GI wiring — design (Jun 2026)

**Status:** IMPLEMENTED (landed as of HEAD, 2026-06-28 — `StitchDisplayController`
registered for `Mode.STITCH_1D/STITCH_2D` in `display_controllers.py`; `_live_mode()`
returns those when `scan.stitched_*` exists in `display_frame_widget.py`;
`render_stitch_result` is now the legacy one-shot path). Builds on the Phase-1a/1b
stitch wiring + the accumulator reconciliation (`8addcdd`). Remaining (P7): the Refine
button + GI-stitch panels (reconcile against `stitching_rsm_build_plan.md` P7, which
still lists the Stitch viewer as LIVE-GATED). This design doc stays local/uncommitted
per the no-docs-in-commits rule.

## Problem
`displayFrameWidget.render_stitch_result` is a **one-shot** direct draw called from
`stitch_thread_finished` after `update_all()`. The *next* `update()` (a frame
click, a timer tick, autorange) recomputes the display via
`controller_for(_live_mode())` — and `_live_mode()` has no stitch awareness, so it
returns `INT_1D`/`INT_2D` → the `ScanDisplayController` redraws the per-frame
integration view **over** the stitch. The stitch must become a first-class display
source that re-renders every tick.

## Model (chosen)
The stitch display **follows the wrangler Mode dropdown**, mirroring the existing
`viewer_mode` pattern, gated by result-existence:
- Shown while a **Stitch 1D/2D** mode is selected in the dropdown **AND** the
  matching `scan.stitched_1d`/`stitched_2d` result exists.
- Selecting **Stitch 1D** before running keeps the per-frame view (no result yet →
  no premature blank).
- Switching the dropdown back to a per-frame mode returns to the per-frame view.
- A new scan clears `scan.stitched_*` → automatically returns to per-frame.
- Frame selection does **not** dismiss the stitch (that's the whole point).

Storage: the result already lives on the scan (`scan.stitched_1d`/`stitched_2d`,
`IntegrationResult1D/2D`). The controller reads it from `widget.scan` — **no new
store**. A `PublicationStore`-backed store was rejected: it is keyed by frame
label with per-frame eviction/generation/rehydration that a whole-scan synthetic
does not fit (see the Understand-workflow gotchas).

## Implementation

### display_logic.py (pure, Qt-free)
- `Mode.STITCH_1D = "stitch_1d"`, `Mode.STITCH_2D = "stitch_2d"`.
- `PANEL_LAYOUT[STITCH_1D]` = the INT_1D geometry (plot-only); `PANEL_LAYOUT[STITCH_2D]`
  = a cake-focused geometry (2D pane visible, 1D plot collapsed).
- `stitch_display_state(mode, generation, *, has_1d, has_2d, title="")` → a
  `DisplayState`: STITCH_1D → one `PLOT_1D` panel (`has_data=has_1d`),
  `layout=((PanelKey(PLOT_1D),),)`; STITCH_2D → one `CAKE_2D` panel
  (`has_data=has_2d`), `layout=((PanelKey(CAKE_2D),),)`; `load_status` READY iff the
  relevant result exists, else EMPTY. `render_roles_for_state` appends the other
  legacy roles as cleanup → raw/plot are blanked.
- Reuse the existing `stitch_plot_payload` / `stitch_image_payload`.

### display_controllers.py
- `StitchDisplayController(_BaseController)`:
  - `compute_state(widget, mode)`: read `scan.stitched_1d/2d` presence, return
    `stitch_display_state(mode, widget.display_generation, has_1d=…, has_2d=…)`.
  - `build_payload(widget, state)`: `DisplayPayload(generation=state.generation,
    plot=stitch_plot_payload(scan.stitched_1d) if STITCH_1D else None,
    cake_image=stitch_image_payload(scan.stitched_2d) if STITCH_2D else None,
    raw_image=None)`. Generation always matches (built in one synchronous pass).
  - Register for STITCH_1D + STITCH_2D in `register_default_controllers`.

### display_frame_widget.py
- `__init__`: `self.stitch_display_mode = None`.
- `_active_stitch_mode()`: returns `'1d'`/`'2d'` iff `stitch_display_mode` matches
  **and** the corresponding `scan.stitched_*` exists, else None.
- `_live_mode()`: after the viewer checks, `'1d'→STITCH_1D`, `'2d'→STITCH_2D`.
- `_updated()`: `return True` early when `_active_stitch_mode()` is set (stitch is
  independent of per-frame cache readiness).
- `render_display`: for STITCH modes call `self._apply_layout(mode)` (the
  INT branch already does share-axis + 1d-only). `_draw_payload(PLOT_1D, …, state)`:
  when `state.mode` is a STITCH mode, draw the single trace directly onto
  `self.plot` (the proven one-shot 1D path — avoids the per-frame `update_plot_view`
  machinery); CAKE_2D already routes through `_draw_image_payload`.

### static_scan_widget.py + image_wrangler.py
- Wrangler: add `sigStitchModeChanged = Signal(str)` emitting `'1d'`/`'2d'`/`''`
  from `_on_mode_changed` (prev-tracked like `sigViewerModeChanged`).
- staticWidget: connect it to set `self.displayframe.stitch_display_mode = (s or None)`
  + bump `display_generation` + `update_all()`.
- `stitch_thread_finished`: on success set `displayframe.stitch_display_mode =
  stitch_thread.mode`, bump generation, and `update_all()` (drop the one-shot
  `render_stitch_result` call — the controller now owns the paint).
- `new_scan`: clear `scan.stitched_1d/2d` + `displayframe.stitch_display_mode`.

## Tests
- Headless (`pytest -m display_logic`): `stitch_display_state` shapes + the
  STITCH `_live_mode`/`render_plan` draw/clear decisions.
- Offscreen GUI: a stitch result survives a subsequent `update()`; switching the
  mode dropdown back returns to per-frame; new scan clears it.

## Mask (task #18)
`_build_stitch_params`: convert the wrangler flat-index mask → 2D bool (True=exclude)
via `scan.detector_shape`; verify the wrangler convention (1=bad). pyfai_q_frames
applies `w = where(mask, 0, w)`.

## GI wiring (task #19) — GATED
Add `backend`/`gi`/`corrections` to `ewald.stitch.run_stitch` + the pyfai_hist+GI
path. Real-data GIXSGUI absolute-convention validation is still outstanding
(guardrail: don't guess geometry), so GI stays OFF by default in the GUI until that
lands — plumb it but keep the default `multigeometry`.
