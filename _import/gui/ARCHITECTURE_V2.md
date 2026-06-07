# Architecture V2 Spike

This branch explores xdart as a thin Qt shell around the headless spine in
`ssrl_xrd_tools`. It is not a release branch: do not merge, tag, bump, push, or
publish from here without a separate stabilization pass.

## Boundary

- xdart owns Qt widgets, worker orchestration, user preferences, display state,
  and `FramePublication` caching.
- `ssrl_xrd_tools` owns source opening, reduction, GI math, persistence, typed
  analysis plans, and headless notebook APIs.
- New xdart code should convert GUI/live state into a headless source + plan,
  run the ssrl spine, then publish display-ready frame records.

## New Seams

- `xdart.modules.sources.LiveScanFrameSource` adapts a `LiveScan` to the ssrl
  `FrameSource` protocol without importing Qt.
- `ReductionJob` packages source, plan, sinks, chunk policy, and run options so
  Qt workers can become orchestration wrappers.
- `PublicationSink` converts `FrameReduction` output into `FramePublication`
  records for the GUI store.
- `display_logic.render_roles_for_state(...)` now derives render roles from the
  `DisplayState.layout` descriptor and appends legacy panel roles only for stale
  panel cleanup.

## Current Limitations

- Existing wrangler threads still own much of the production execution path.
  `RESTRUCTURE-TODO(WS-X1)` marks the seam where they should switch to
  `ReductionJob`.
- Render planning is still role-level rather than exact `PanelKey` level.
  `RESTRUCTURE-TODO(WS-X2)` marks the step needed for repeated RSM panels and
  future fitting/result panes.
- Backward compatibility is deliberately secondary on this branch. Correctness,
  performance on slow computers, and simple headless reuse are the priorities.

## Acceptance Gate

Before considering this branch for stabilization, the strict real-data spine
must compare live, batch, and reload outputs for standard and GI scans without
loosening writer validators or numerical tolerances. GUI tests should cover
mode switches, viewer modes, publication-store eviction, and long-scan memory
behavior.
