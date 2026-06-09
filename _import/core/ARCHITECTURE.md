# Architecture V2 Spike

This branch explores the next architecture for `ssrl_xrd_tools` and `xdart`.
The goal is one headless ingestion, reduction, persistence, and result spine in
`ssrl_xrd_tools`; `xdart` should become a Qt shell around that spine.

## Contracts

- `FrameSource` is the input seam. Sources expose frame labels, lazy frame
  loading, chunk iteration, per-frame metadata, geometry, and capabilities.
- `ScanFrame` / `Scan` are the canonical headless frame and scan containers.
  Legacy import paths may re-export these names while callers migrate.
- `FrameView` is the reduced-frame/result contract. It is immutable,
  GUI-free, and carries axes, 1D/2D intensities, raw/thumbnail references,
  heterogeneous metadata, source provenance, and GI identity.
- `ReductionSession` is the core reduction primitive: it owns the executor,
  per-thread pyFAI integrators, sink lifecycle, the GI-freeze pre-pass,
  progress, and cancellation for a scan's lifetime, so callers feed chunks
  without rebuilding CSR-LUTs or reopening sinks. Two execution modes:
  `"chunked"` (`process()`) integrates frames in chunks; `"streaming"`
  (`submit()` + a bounded in-flight window drained by one writer thread,
  out-of-order completion ok, single-writer sink) is what xdart's GUI runs by
  default. `run_reduction(..., execution=...)` is the one-shot convenience
  wrapper over both. `finish()` is **fail-loud**: a sink/write failure re-raises
  by default (preserving the original exception; `raise_on_failure=False` opts
  out) so a data-writing run can't silently report success. Live, batch,
  notebook, XYE, and NeXus workflows differ only by source, sink, executor, and
  policy.
- Sinks are the output seam — `MemorySink`, `XYESink`, `NexusSink`, and the
  `CompositeSink` fanout. Re-feeding an already-processed frame index is a
  `replace` (idempotent), not a second write/count.
- NeXus is the persistence contract. Writer validators remain strict; fixes
  must make data faithful rather than relaxing validation.

## Boundaries

- `ssrl_xrd_tools` must not import `xdart`.
- Headless contracts must import without Qt, napari, or GUI dependencies.
- `xdart` may own Qt widgets, user interaction, worker scheduling, and
  display publications. It should not own scientific algorithms.

## Spike Rules

This branch may leave explicit `RESTRUCTURE-TODO(<workstream>)` markers at
unfinished seams. Silent invariant breaks are not acceptable. The final review
gate is strict live/batch/reload equivalence plus full headless and GUI suites.

## Implemented In This Spike

- `ssrl_xrd_tools.core.scan` defines the canonical source/frame/scan
  contracts and keeps NeXus loading lazy to avoid import cycles.
- `ssrl_xrd_tools.sources` provides `open_source(...)` and adapters for memory,
  live streams, image files, TIFF series, NeXus stacks, and processed xdart
  NeXus files.
- `ReductionSession` / `run_reduction(...)` accept a `Scan` or `FrameSource`,
  reuse one executor + per-thread integrator set across the whole scan (the
  per-scan CSR-LUT build is asserted once by the perf gate), support multiple
  sinks, chunking, cancellation, progress callbacks, XYE output, and NeXus
  output.
- GI reduction modes are represented in the headless plan via typed 1D/2D mode
  enums and incident-angle resolution; the GI output-range freeze runs as a
  pre-pass inside the session (`first_frame` for live, `scout_union` for batch).
  A blank/degenerate scout raises the typed `GIFreezeError`.
- Processed NeXus files are stamped as schema v2 and preserve string metadata
  alongside numeric metadata.
- Typed notebook/headless analysis entry points exist for stitching, RSM, peak
  fitting, phase fitting, and sin2psi. These are intentionally thin wrappers
  over existing engines until each workflow receives its own equivalence gate.

## Open Seams (backlog)

- The typed analysis entry points (stitching, RSM, peak/phase fitting, sin2psi)
  are public API scaffolding over the existing engines, not a replacement for
  per-workflow validation, persistence, or GUI tools. A generic
  `AnalysisStep`/registry abstraction is deliberately deferred until at least
  two of these typed plans converge (`RESTRUCTURE-TODO`: generic analysis
  protocol) — premature unification would lock in the wrong shape.
