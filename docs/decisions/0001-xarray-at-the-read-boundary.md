# ADR-0001: xarray stays at the read boundary

**Status:** accepted · 2026-06-12 · (greenfield Difference 7 follow-up)

## Context

The greenfield design floated "xarray as the in-memory currency throughout
the result path" — reduction emitting an `xr.Dataset` per frame, NeXus
written from it — with custom containers earning their place only on the
hot integration path.

The June 2026 gap scout costed the change against the real codebase:
~850 LOC touched across the writer (`write_nexus_frame` re-unpacking), the
display layers (attribute access migration), the GI freeze readers, and
~50 test fixtures; an estimated 5–10% suite slowdown (per-frame Dataset
construction is slower than the slotted containers for this access
pattern); and the publication/validation path re-plumbed.  The benefit —
notebook-native indexing and coord alignment — already exists where
notebooks actually enter: `read_scan` / `read_scan_metadata` /
`open_scan` return xarray.

## Decision

`IntegrationResult1D/2D` and `FrameReduction` remain the result-path
currency.  xarray is used **only** at the read boundary (file → analysis).
The reduction hot path, sinks, and display layers consume the containers
directly.

## Consequences

- No result-path churn; the equivalence spine and byte-compat gate are
  untouched.
- Notebook users get xarray by reading the file (or `open_scan`), which is
  also the honest data path — what you analyze is what was persisted.
- If a future consumer genuinely needs per-frame Datasets in-process, the
  right seam is a thin adapter over `FrameReduction` (or the Phase-5
  `FrameRecord`), not a rewrite of the path.
