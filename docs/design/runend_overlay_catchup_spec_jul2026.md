# Run-end overlay catch-up (PERF-3 tail) + live overlay cost — diagnosis & handoff spec

**Date:** 2026-07-05 · **Status:** diagnosis complete, fix spec'd, hand-off ready.
Sites cite committed HEAD `d8eca202`; the working tree carries an uncommitted XDART_PERF
heartbeat patch (+58 lines in static_scan_widget.py) — anchor by SYMBOL, not line.
`SSW` = static_scan_widget.py, `H5V` = h5viewer.py, `DFW` = display_frame_widget.py.

> **Verification (2026-07-07, orchestrator — 6-way adversarial pass vs. real code): SOUND WITH
> CORRECTIONS.** Part 1 (tail gap + Item-2 autopsy + one-shot catch-up fix) is confirmed and
> implementable — the false "cheap in-memory reselect" premise is even still enshrined in the code
> comment at `wrangler_finished` (the old Item-2 site). ONE blocking correctness fix applied to
> guard-3 below (the bare `(skey, fidx)` unpack ValueErrors on slice-mode 3-tuples). Part 2 is a
> PLAUSIBLE-BUT-UNPROVEN hypothesis: the fixed ≤2 Hz cadence is confirmed, but the paint is
> decimation-capped at 256 rows (`MAX_WF_ROWS`/`MAX_WATERFALL_PAYLOAD_ROWS`), so only accumulator
> bookkeeping is truly O(N) — the 55 s attribution needs the instrumented run this spec prescribes.
> Two Part-2 factual errors corrected inline (reduction workers are ThreadPoolExecutor THREADS, not
> processes; the paint does not grow unboundedly with N).

## Part 1 — the ~120-frame run-end tail gap

### Root cause (pinned)

The tail's display payloads are dropped mid-run by the cap-128 drop-oldest handoff whenever the
GUI lags; those frames are on disk but were NEVER published into the store, so no
resident-rebuild can ever paint them (Item-2's "cheap in-memory reselect" premise was false —
Item-1's 1 GiB budget keeps *published* frames, not never-published ones). Recovery requires a
real async disk load — exactly what a manual Show-All does.

### Why Item-2 (8fdc81e1) failed — autopsy

1. Tail non-resident (above) — `_render_overlay_full_scan` could only rebuild the ~3476 resident
   rows.
2. With missing frames, `_data_changed_now(show_all=True)` does NOT paint synchronously — it
   schedules a bulk load on a 100 ms debounce and defers the render behind
   `_browse_one_shot_pending_render` (H5V:3062-3075).
3. Two run-end 100 ms one-shots were already armed when the hook ran: `set_run_writing(False)`'s
   falling-edge `data_changed()` (H5V:2540-2543) and `update_all → latest_frame → data_changed()`
   (auto-last). Both fire AFTER `wrangler_finished` returns, rewrite `frame_ids=[latest]`, and
   reset `_browse_one_shot_pending_render` (H5V:3091-3093) → the deferred full render dies before
   the tail hydration completes; tail publications reach the store but nothing appends them.
4. `integrator_thread_finished`'s `clear_overlay()` (SSW:5751) had just wiped the accumulator AND
   the pending-append queue (DFW:4412-4419).
5. NOT the freeze guards — `_processing_active`/`_run_writing` were already False. Sequencing was
   the killer.
6. The test asserted "a render fires", not "accumulator reaches N" — mechanism-not-outcome trap.

### Key facts for the fix

- `h5viewer.show_all()` (H5V:775/1312) is an ordinary slot, programmatically callable,
  byte-identical to the click: selects all rows (which keeps the completion render's selection
  full — the thing Item-2 lacked), splits resident/missing, bulk-loads missing via ONE
  `_LoadFramesWorker` (RN-2 chunked), generation-gated absorb, completion render appends into the
  carried accumulator (OV-3 semantics — append-only, dedup by scan-qualified row id).
- The `FrameHydrationWorker` is app-lifetime; the store, scheduler, and load machinery are all
  alive at run end. Everything needed is true from the end of the `integrator_thread_finished`
  delegate onward; only the two 100 ms selection-collapse echoes are still in flight.

### Fix spec — one-shot post-quiescence auto-Show-All (size S)

Hook: end of `wrangler_finished`, LIVE saw-frames branch (same guard cluster as the reconcile),
REPLACING the failed Item-2 call. Arm `self._runend_catchup_token = (scan.name,
displayframe.display_generation)`, `_tries = 0`, then `QTimer.singleShot(250,
self._runend_overlay_catchup)`. Clear the token in `_enter_run_state` and in
`set_file`/`data_reset`.

Callback guards, in order:
1. **Cancellation (generation-gated):** abort if token stale — new run active, `scan.name`
   changed, `displayframe.display_generation` changed (any user gesture bumps it), or
   `auto_last` is False (user clicked a frame).
2. **Quiescence:** if `_selection_coalesce_timer` / `_load_coalesce_timer` /
   `_update_coalesce_timer` active, or `h5viewer._load_worker is not None` → re-arm
   `singleShot(250)`, bounded ≤8 tries (~2 s), then give up silently (degrades to today's
   manual-Show-All behavior).
3. **Applicability:** method in (Overlay, Waterfall) and auto_last; missing set computed from the
   accumulator ids WITHOUT destructuring the row-id tuple directly — in slice-active 2D waterfall
   `overlay_identity_for_widget` (display_overlay_utils.py:378) emits a **3-tuple**
   `(scan_key, frame_idx, projection_id)`, so a literal `for (skey, fidx) in …ids` raises
   `ValueError`. Use the length-tolerant decoders:
   `missing = set(scan.frames.index) − {frame_index_from_qualified_id(r) for r in
   _waterfall_history.ids if scan_key_from_qualified_id(r) == current}`
   (`frame_index_from_qualified_id`/`scan_key_from_qualified_id`, display_logic.py:484/491, handle
   len≥2 tuples). None/empty history with non-empty index = all-missing. Empty → done (idempotent).
4. **Fire once:** consume the token, one INFO log, call `self.h5viewer.show_all()`. No new
   machinery — the click path does the rest, bounded by RL-1 resident-skip/success-set/backoff
   and BR-3b's anchor budget.

Must-not-happen (design already covers): no treadmill (fires at most once per run; never re-arms
on "still missing after firing"); no interference with S-14/OV-6 resets (runs strictly after the
run-end clear, appends through the normal payload path, never calls clear_overlay); no firing
after a user gesture (guard 1); no mid-run firing (token armed only in wrangler_finished).

**Outcome-asserting test** (offscreen, production-wired accumulate seam): widget in Overlay +
auto_last; `frames.index = 1..N` (N≈40); store resident only for `1..N−k`; tail served ONLY by
the disk/hydrator path (real small .nxs preferred) — reproducing the non-resident-tail trap.
Drive the real `wrangler_finished` LIVE branch, pump the event loop until token consumed + load
worker done. ASSERT: `{fidx for (_, fidx) in _waterfall_history.ids} == set(scan.frames.index)`,
and the count never decreased after arming. Negative cases: user click before callback → no
show_all; method=Single → no-op; second run → token cleared; already-complete → not called;
quiescence never reached → gives up after ≤8 tries.

Rejected alternatives: raising the handoff cap (drops are continuous mid-run, not a final burst;
big cap reintroduces the MEM-1a OOM); reconcile-enqueues-into-1D-hydrator (viable,
selection-neutral, but needs the SAME deferral scaffolding with a weaker per-label driver and
bypasses the tested one-shot render contract — fallback only).

**Estimate:** ~50-70 lines production + ~120-180 lines test; half a day to a day.
**Risk:** low — display-only, guarded no-op by default, reuses the manually-exercised machinery;
failure degrades to current behavior. Safe pre-tag ONLY with a live 3600-frame verification slot
(ledger rule); otherwise first v1.1 item. Recommendation: implement now, gate on the live repro.

## Part 2 — overlay/waterfall live processing overhead (125 s → 180 s, progressive)

> **MEASURED 2026-07-07 — Part 2 hypothesis REFUTED; adaptive cadence NOT worth building.**
> Instrumented `update_wf` (`[PERF] overlay render: render_ms=… rows=… drawn=…`) on a live 3621-frame
> Overlay run: `render_ms` is FLAT ~2–3.5 ms independent of `rows`, and `rows`/`drawn` plateau at
> ~248–253 (the payload/`MAX_WF_ROWS` decimation), never approaching 3621. So the overlay render is
> NOT the slowdown and lever-1 (adaptive cadence) would save ~nothing — dropped. **The real issue the
> heartbeat surfaced instead:** recurring ~3 s MAIN-THREAD stalls in the LAST ~15 s of the run
> (`[PERF] main-thread stall: gap≈3000ms at t+143…159s`, max 6132 ms), worsening as N grows,
> UNINSTRUMENTED (the flushes around them are only ~100–180 ms). NOT the run-end explicit gc.collect
> (70 ms, 0 collected). A GC timing probe (`[PERF] gc:` in `_gui_main.run`, XDART_PERF) + a faulthandler
> SIGUSR1 dump during a stall will pin it (GC vs an O(N) per-flush op at N≈3400). This end-of-run freeze
> is now the tracked PERF item; it likely also drives the Part-1 tail gap (a frozen GUI drops hand-offs).


Mechanism (HYPOTHESIS — must be confirmed with one instrumented run; NOT statically proven): the
live overlay render is throttled to a FIXED cadence (≤2 Hz while `_processing_active`, confirmed:
`display_plot.py:1195` + the SSW flush throttle, no row-count scaling). **Correction:** the render
is NOT an unbounded O(N) redraw — both the payload projection (`MAX_WATERFALL_PAYLOAD_ROWS=256`,
display_publication.py) and the `setImage` draw (`MAX_WF_ROWS=256`, display_plot.py:47/1209)
decimate to a strided ≤256-row view, so the DRAWN image plateaus at 256 rows. Only the
accumulator/payload BOOKKEEPING (out_ids/out_names/out_meta lists + index_by_key dict in
`accumulate_waterfall`; name_by_id/overlaid_ids in `_history_to_payload`) is genuinely O(N). That
bookkeeping runs on the GUI thread and competes for the GIL with the wrangler's collect/dispatch
loop, the writer, and every publish — **all of which are Python THREADS in the same process,
including the reduction workers (ThreadPoolExecutor, reduction/core.py:1068 — the ProcessPoolExecutor
path was removed), which release the GIL only inside pyFAI/numpy/h5py C sections.** WHETHER that
residual O(N) bookkeeping (plus GIL contention) actually makes `render_ms ∝ N` and accounts for the
125→180 s regression is UNVERIFIABLE by reading code — it is exactly what the instrumented run below
must decide (does `render_ms` track N, or plateau at ~256 rows?). If it plateaus, the slowdown is
elsewhere. The tail-feeding link to Part 1 holds only if the render IS the growing GUI cost.

What is NOT the cause (already fixed): the O(N²) full-stack rebuild (BW-A6 geometric buffer) and
the hydration treadmill (RL-1).

Confirm with: the uncommitted XDART_PERF heartbeat + a per-render timing slice (log
`render_ms` and row count every 50 renders) — expect render_ms ∝ N.

Easy wins, in order of payback/risk:
1. **Adaptive live render cadence (S):** scale the throttle interval with accumulated row count
   (e.g. ≤500 rows → 2 Hz; ≤1500 → 1 Hz; ≤3000 → 0.5 Hz; above → 0.25 Hz). Bounded GUI cost,
   ~zero risk (batch mode is already fully silent; this is a middle ground). Expect most of the
   55 s back on a 3621-frame run.
2. **Bounded live draw window (S/M) — DOWN-RANKED, gate behind the instrumented run:** the DRAW is
   already 256-row decimation-bounded (above) and this spec keeps the accumulator complete, so a
   draw window cuts NEITHER the already-capped paint NOR the O(N) bookkeeping — likely low value as
   written. Only pursue if the instrumented run shows the *paint* (not bookkeeping/GIL) is the
   bottleneck. (Original idea: while `_processing_active`, draw only the last M rows with a "showing
   last M of N" annotation; full paint at run end via the Part-1 catch-up.)
3. **Waterfall image path check (S, diagnostic):** for >15 curves the auto-waterfall uses the
   pmesh/image path — verify the per-render image build reuses the BW-A6 geometric buffer as a
   view (no full copy per paint) and consider `setImage(..., autoLevels=False)` with the S-15
   token levels; per-trace overlay below 15 curves is the expensive Python-object path and the
   cadence lever (1) covers it.

Progressive slowdown is NOT unavoidable; only a small constant overhead is. Lever 1 alone
probably makes overlay-live ≈ single-mode + a few percent. All three are v1.1-compatible;
lever 1 is small enough to ride pre-tag if a live slot exists.
