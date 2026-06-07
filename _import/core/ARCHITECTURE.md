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
- `run_reduction` is the only reduction spine. Live, batch, notebook, XYE, and
  NeXus workflows differ by source, sink, executor, and policy.
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
