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
| 1 | **RE-INVESTIGATING** (clear-on-signal reverted — it regressed) | instrument `d66671c6`; fix+revert follow | The clear-on-identity fix **made it worse** and was reverted. Live symptom: the `new_scan` signal arrives **LATE** (mid-2nd-scan), so clearing on it wiped the 2nd scan's already-accumulated frames (1–8) and restarted the panel at 9. **Lesson: the clear cannot key off the `new_scan` SIGNAL — its timing is unreliable.** The correct fix must attribute frames to their scan/file (per-frame identity), so the panel scopes to the current scan without depending on a mis-timed boundary signal. Re-investigation in progress. Instrumentation (the `[PERF] new_scan boundary` + `h5viewer.update_data` logs) stays. |
| 2 | **DONE + LIVE-VALIDATED** | `cf72fca3` | `_list_timer` (`Coalescer(70, throttle)`) → `_flush_frame_list` runs only the O(new) list refresh + signal-blocked auto-last cursor; heavy drain+render stays on `_update_timer` (200 ms). Timers stopped together at teardown; mocks updated (+`_list_timer`); unit test; offscreen + spine green. **Live (651-frame eiger, Int 2D): NO regression — total 27.25 s → 24.98 s, dispatch 24.97 → 22.58 s, per-frame 38.3 → 34.7 ms, flush render 51–83 → 40–68 ms; Frames list now scrolls continuously.** Well inside the ≤5%/≥50% bar (on the good side). |
| 3 | **DONE** (live-perf pending) | this commit | live heavy-render quantum 200 → **100 ms** (2× image update rate). Measure-gated: keep window ≥ median flush total (~70–90 ms); raise back if `total` creeps over. |
| 4 | **SKIPPED** | — | maintainer: waterfall throttle is not an issue |
| 5 | proposed | — | next smoothing lever = cut the render leg (40–68 ms, dominated by re-drawing + re-leveling the raw Eiger frame): throttle/subsample the autoscale re-level. Needs finer render-leg instrumentation first (measure-first). |

## Measurement protocol (live — the maintainer runs)
`conda activate xrd_test && XDART_PERF=1 xdart`, point at the 651-frame eiger dir (Image Directory,
live), run Int 1D + Int 2D, and capture the `[PERF] flush: drain= list= render= total=` lines
before vs after. Success = smoother perceived scrolling (list/status) with `render`/`drain` legs
unchanged and no `total` exceeding the (shortened) window.
**Also record the TOTAL reduction time** for Live and Batch (Int 1D + Int 2D) before vs after —
compare against the ~26/35/29/41 s baselines. Per the tradeoff bar: keep any processing-time
regression **≤5%** and only accept it for a **≥50%** smoothness gain; if the list-leg frequency
shows up as a processing hit, raise the fast-timer interval (70→100 ms) and re-measure.
