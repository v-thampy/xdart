# Architecture V2 Notes

This branch explores xdart as a thin Qt shell around the headless spine in
`ssrl_xrd_tools`. Do not tag or publish from here until the prerelease gates in
the review runbook pass cleanly.

## Boundary

- xdart owns Qt widgets, worker orchestration, user preferences, display state,
  and `FramePublication` caching.
- `ssrl_xrd_tools` owns source opening, reduction, GI math, persistence, typed
  analysis plans, and headless notebook APIs.
- New xdart code should convert GUI/live state into a headless source + plan,
  run the ssrl spine, then publish display-ready frame records.

## New Seams

- `xdart.modules.reduction.frame_from_live_frame` / `scan_from_live_scan`
  adapt a `LiveFrame`/`LiveScan` to the headless core contracts
  (`xrd_tools.core.scan`) without importing Qt — the single LiveScan->core
  adapter (the duplicate `LiveScanFrameSource` was removed in the monorepo
  migration; the core `Scan` itself satisfies the chunk-iteration boundary
  RSM/stitching consume).
- **Streaming, sink-driven write is the DEFAULT (WS-X1 endpoint).** The wrangler
  opens one persistent `ssrl_xrd_tools.reduction.ReductionSession` per scan
  (`execution="streaming"`, via `open_live_reduction_session`) with an
  `xdart…QtNexusSink` as its `ReductionSink`. Worker threads integrate frames in
  parallel; a single writer thread drains completions (out-of-order ok) and owns
  the `.nxs`/XYE write — so the per-thread pyFAI integrators are built once per
  scan and I/O pipelines with compute. This is the default for **batch** and for
  a **non-batch reprocess** (`XDART_LIVE_EXECUTION`/`XDART_BATCH_EXECUTION`,
  default `streaming`; `serial`/`chunked` kept one cycle as fallbacks). The same
  engine is exposed headlessly via `run_reduction(execution="streaming")`.
  - **Live display contract:** the writer/worker threads do ZERO Qt work — they
    stash the hydrated frame into `host._published_frames` and emit one
    lightweight `sigUpdate`; the GUI-thread `static_scan_widget.update_data`
    (coalesced) owns the display caches + the publication the cake renders from.
  - **Two deliberate write paths:** true-live *watching* (Phase 3, detector-rate,
    one frame at a time) intentionally keeps the serial `_process_one` + direct
    `_save_to_nexus` write — parallelism is moot there. So "one write path" means
    batch + reprocess; true-live is a second, intentional path, not a gap.
  - **GI common grid:** a batch-streaming run freezes the q/χ grid from the WHOLE
    scan's incidence range (a cheap metadata pre-pass over the lowest/highest-
    incidence frames) before the session opens, so multi-chunk angle-dependence
    scans don't clip later frames to the chunk-1 grid.
  - **Fail-loud writes:** `ReductionSession.finish()` re-raises a sink/write
    failure by default; xdart surfaces it (no silent "saved").
- `display_logic.render_roles_for_state(...)` derives render roles from the
  `DisplayState.layout` descriptor and appends legacy panel roles only for stale
  panel cleanup.

## Current Limitations (backlog)

- Render planning is still role-level rather than exact `PanelKey` level.
  `RESTRUCTURE-TODO(WS-X2)` marks the step needed for repeated RSM panels and
  future fitting/result panes; it is intentionally deferred to land with the RSM
  viewer.
- A handful of GI-scout helpers in `image_wrangler_thread.py` are now test-only
  fixtures (GI freezing moved into the ssrl session); `RESTRUCTURE-TODO(B2)`
  marks them for relocation to a `tests/` helper module together with the
  GI-equivalence test refactor.
- Backward compatibility is deliberately secondary on this branch. Correctness,
  performance on slow computers, and simple headless reuse are the priorities.

## Acceptance Gate

Before considering this branch for stabilization, the strict real-data spine
must compare live, batch, and reload outputs for standard and GI scans without
loosening writer validators or numerical tolerances. GUI tests should cover
mode switches, viewer modes, publication-store eviction, and long-scan memory
behavior.
