# Design — GUI live-display responsiveness (liveness push)

**Status:** IN PROGRESS (branch `feature/gui-liveness`, off `f2e99a4a`, 2026-07-01). Living doc —
update the §Status checklist after each step (this is the convention; keep it current).

## Problem
During a live scan the display updates "in bunches": the screen jumps several frames at once
every ~200 ms rather than scrolling smoothly. Root cause (measured/traced, not producer batching):
**one GUI flush quantum** — `static_scan_widget._update_timer = Coalescer(200, "throttle")` (:3474)
fires `_flush_pending_update` (:3991) at most every 200 ms, which in one go runs the heavy
`_drain_pending_frames` (publication build + upsert + `scan_data` rebuild) AND the frame-list
refresh AND the cursor advance AND the render (`data_changed` → `setImage`). All frames that piled
up in the window surface together = the chunkiness. The producer emits one `sigUpdate(idx)` per
frame; `update_data(idx)` (:3799) is already O(1) (index append + stash + trigger).

**Floor:** producer-paced. When frame inter-arrival > flush window each frame already gets its own
paint; smoothing only helps the fast-scan case where frames pile up per window.

## Guardrails
- **Measure-first** (`XDART_PERF=1`, the existing `[PERF] flush: drain= list= render= total=` at
  :4053 + `[PERF] drain …` at :3983) on the 651-frame eiger baseline in `xrd_test`, before/after
  every change. Keep the flush window ≥ median flush total or the event loop re-saturates (the exact
  freeze the 200 ms was tuned against).
- **Processing-perf tradeoff bar (maintainer, 2026-07-01):** these changes must NOT significantly
  regress *processing* throughput. Also measure the **total reduction time** (not just flush legs) —
  the 651-frame eiger baselines are ~**1D Live 26 s / Batch 35 s, 2D Live 29 s / Batch 41 s**
  (`perf_baselines_jun2026`). Acceptance: a **≤5%** processing-time regression is worth it **only if**
  it buys a **≥50%** smoothness/responsiveness improvement; otherwise back it out or tune the knob
  (the fast-timer interval). Better-with-no-regression is always preferred; the ≤5%/≥50% band is the
  fallback the tradeoff is allowed to occupy. The list leg is GUI-thread-only (no worker-thread work),
  so any processing hit is pure GIL contention — expected << 5%, but confirm.
- Content-preserving → **spine-safe**: verify `test_gi_batch_real_data.py` (68) + offscreen after each change.
- Branch isolation: this touches `static_scan_widget.py`'s live path — the same file Phase 5
  rewrites — so it lives on its own branch; reconcile at merge. Low-risk wins are cherry-pickable to v1.0.

## Plan
| # | Change | Win | Effort/Risk |
|---|---|---|---|
| 1 | Instrument the new-scan boundary (log `live_run_active` + index length) to pin the reported frames-panel-persist bug on a real run; apply the *consume-the-flag* fix in `h5viewer.update_data` empty-index branch ONLY if a non-empty-index boundary reproduces | Fixes the actual bug once pinned | S (after repro) |
| 2 | **Split the light list/cursor update from the heavy render**: a fast `Coalescer(~70 ms)` runs only `h5viewer.update_data` (O(new)) + auto-last cursor (`latest_frame(emit_update=False)`, no render); keep drain+render on the 200 ms timer | **Biggest perceived-liveness win** — Frames list + selection + status scroll continuously for near-zero cost | M / low |
| 3 | Shorten the heavy flush quantum 200→100 (→80) ms | High on fast non-GI scans | S / measure-gated |
| 4 | Raise waterfall `setImage` throttle ~2→3–4 Hz (`display_plot.py` `_wf_last_draw_t`) | Smoother Overlay/Waterfall on long scans | tiny / measured |
| 5 | (opt) Decouple the drain onto its own steady timer independent of paint | Removes build cost from the visible step | M / low |

## Status checklist (update after each step)
| # | Status | Commit | Note |
|---|---|---|---|
| 1 | **DONE — FRAME-DRIVEN + SIGNAL-DEFERRED** (LIVE-VALIDATED) | this commit | Root cause (adversarial workflow `wf_22468eef`): `new_scan` is emitted from the wrangler **run** thread while per-frame `sigUpdate` comes from the sink **writer** thread (no Qt cross-thread ordering) + Eiger prefetch read-ahead → `new_scan(scanB)` can land AFTER scanB frames 1..N. **Fix:** the boundary is now FRAME-DRIVEN — `update_data` derives each frame's scan from its `source_file` (`_scan_key_from_source`, matching the wrangler's naming) and calls `_rescope_frame_panel_to()` BEFORE appending, so the triggering frame is #1 of the new scan and its own frames can't be dropped. A late `new_scan` is **identity-guarded** by a **consumed flag** `_frame_driven_rescoped_pending` (set when a frame rescopes, reset at run start via `_enter_run_state` + consumed on each `new_scan`) **AND** a key match — NOT a bare name/stamp test. This closes BOTH loss modes: a late `new_scan` for already-arrived frames skips the FULL destructive set (index + `_pending_frames`/`publication_store`/`_scan_info_rows`/overlay), while a **same-name RE-RUN still clears** (flag False → no frame rescoped this run). Tests `test_multi_scan_frame_boundary.py`: `test_frame_driven_boundary_survives_late_new_scan` (reverted-fix regression) + `test_same_scan_rerun_clears_panel` (the 2nd bug) + well-timed-clear + batch-inert + helper. **Directory-mode finish:** Image Directory bursts new_scan signals wildly out of order, and `set_file`'s async `set_datafile` RENAMES scan.name — both would flip `cur` mid-scan (the "restart-at-9" + "flickers to LaB6 then reverts" symptoms). FINAL fix: `new_scan` acts on the panel/cursor ONLY when IN SYNC with the frame stream (`name == current scan`), else DEFERS entirely; viewer wiring (`set_file`) moved into `_rescope` (frame-driven, fed by a per-scan fname stash) so the async rename can't hijack `cur`. Contract change: a genuinely-new (different-name) `new_scan` no longer clears — the frame stream does (tests `test_out_of_sync_new_scan_defers_to_frame_stream`, updated `test_live_new_scan_invalidates_publication_store`). **LIVE-VALIDATED: directory + frame numbering fixed, no freeze, no name flicker; perf back to baseline after removing the per-frame diagnostic log.** Follow-up: the Detector/Experiment **config** panel can still show a different scan's settings (same burst) — needs a config-stash-and-restore, not yet done. |
| 2 | **DONE + LIVE-VALIDATED** | `cf72fca3` | `_list_timer` (`Coalescer(70, throttle)`) → `_flush_frame_list` runs only the O(new) list refresh + signal-blocked auto-last cursor; heavy drain+render stays on `_update_timer` (200 ms). Timers stopped together at teardown; mocks updated (+`_list_timer`); unit test; offscreen + spine green. **Live (651-frame eiger, Int 2D): NO regression — total 27.25 s → 24.98 s, dispatch 24.97 → 22.58 s, per-frame 38.3 → 34.7 ms, flush render 51–83 → 40–68 ms; Frames list now scrolls continuously.** Well inside the ≤5%/≥50% bar (on the good side). |
| 3 | **DONE + LIVE-VALIDATED** | this commit | live heavy-render quantum + fast list timer, terminal-tunable and **now defaulting to flush=150 / list=60 ms** (the maintainer's validated sweet spot; was 100/70). Measure-gated: keep flush ≥ median flush total (~70–90 ms) or the loop re-saturates. Also disabled the **Append/Replace** (writeModeButton) mid-run via `set_mode_row_enabled` — can't switch write mode during a scan. |
| 4 | **SKIPPED** | — | maintainer: waterfall throttle is not an issue |
| 5 | **DONE — fixed-unverified** | this commit | render-leg smoothing: Linear scale now uses the raw array view instead of copying, `_ceiling_safe_levels` no longer makes a second full float copy, large autoscale populations are sampled at stride 4, and levels are reused for ~1 s unless scale/cmap/shape/dtype/pct changes. `XDART_PERF=1` now logs copy/transform/levels/setImage/hist/total for live A6 validation. |

## Terminal-tunable knobs (sweep live, no rebuild)
- `XDART_FLUSH_MS` (default **150**, floor **110**) — the heavy image-update quantum (`_update_timer`).
  Values below 110 ms are clamped with a warning. Keep ≥ the 100 ms user-selection debounce and the
  median flush total (~70–90 ms) or the event loop re-saturates (freeze). Smaller = smoother image,
  riskier.
- `XDART_LIST_MS` (default **60**) — the fast Frames-list/cursor refresh (`_list_timer`).
- `XDART_PERF` — `[PERF]` logging: per-flush legs, per-scan-boundary, startup timer values (always
  available; per-frame boundary logging was removed after the directory-mode fix landed).
- Both timers logged at startup under `XDART_PERF` (`[PERF] live timers: flush=… list=…`). Example sweep:
  `XDART_PERF=1 XDART_FLUSH_MS=110 XDART_LIST_MS=40 PYTHONPATH=$PWD/src xdart`.

## Measurement protocol (live — the maintainer runs)
`conda activate xrd_test && XDART_PERF=1 xdart`, point at the 651-frame eiger dir (Image Directory,
live), run Int 1D + Int 2D, and capture the `[PERF] flush: drain= list= render= total=` lines
before vs after. Success = smoother perceived scrolling (list/status) with `render`/`drain` legs
unchanged and no `total` exceeding the (shortened) window.
**Also record the TOTAL reduction time** for Live and Batch (Int 1D + Int 2D) before vs after —
compare against the ~26/35/29/41 s baselines. Per the tradeoff bar: keep any processing-time
regression **≤5%** and only accept it for a **≥50%** smoothness gain; if the list-leg frequency
shows up as a processing hit, raise the fast-timer interval (70→100 ms) and re-measure.
