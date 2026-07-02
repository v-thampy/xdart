# xrd-tools architecture

One distribution, two import packages, one direction of dependency:

```
xdart  (Qt GUI: widgets, wranglers, display)        ── view / commands
  │  observes events, sends commands, never owns data
  ▼
xrd_tools.session*  (headless acquisition/reduction service)
  │  owns the live state machine, the writer, cadence, eviction
  ▼
xrd_tools  (pure compute + schema + I/O: core / io / reduction /
            integrate / sources / rsm / analysis / viz)
```

\* the session layer.  The headless
**`xrd_tools.session.ScanSession`** EXISTS (4f-headless: commands in /
immutable `FrameEvent` out; ADR-0003/0004) and `reduction.ReductionSession`
is its streaming engine.  The **4f-bridge** is now BUILT — xdart's
`ScanSessionAdapter` (`src/xdart/gui/tabs/static_scan/wranglers/scan_session.py`)
wires over the public session so the GUI is an event→signal view (gated on live
testing).  The static-scan display decision core also lives under
`xrd_tools.session.display_logic`, with xdart retaining only a compatibility
shim.  What remains: moving the GI whole-scan freeze into core.

## North star

(Full version: `design/roadmap_2026-06-10.md`; the plan that executed the
from-scratch framing: `design/CC_greenfield_implementation_plan_2026-06-12.md`.)

1. **Headless-first** — everything below the GUI is fully usable with no
   Qt installed; a notebook, CLI batch job, or autonomous agent drives the
   same code path the GUI does.
2. **Thin xdart** — the GUI renders immutable frame events and sends
   commands.  The boundary is *data ownership*, not "Qt vs not-Qt": xdart
   should have no API through which to own data.
3. **Robustness** — fail-loud writes, strict schema validators, and the
   live≡batch≡reload equivalence spine as an executable acceptance gate.
4. **Performance** — streaming reduction (parallel pyFAI workers, single
   writer thread, bounded in-flight window), bounded memory
   (persist-before-evict, hydration LRUs).
5. **Expandability** — three frozen seams: `FrameSource` (ingestion),
   `ReductionPlan` (declarative work), `ReductionSink` (injected
   persistence).  New sources/sinks (Tiled, zarr) implement the seam and
   self-verify against the contract tests.

## The contracts

- **`FrameSource`** (`xrd_tools.core.scan`): `frame_indices`,
  `capabilities`, `load_frame`, `iter_chunks`.  `Scan` (the reduction
  input), the `sources/` implementations, and `io.read.ProcessedScan`
  (the file handle) all satisfy it.
- **`ReductionPlan`** (`xrd_tools.reduction`): declarative 1D/2D/GI
  settings; execution policy (chunking, image clearing) lives on the
  runner, not the plan.
- **`ReductionSink`** (`xrd_tools.reduction`): `begin`/`write`/`finish`,
  plus optional `replace`/`abort`/`worker_process`.  Thread discipline is
  part of the contract: `write`/`replace` only ever on the single writer
  thread; `worker_process` on pool workers; `begin`/`finish`/`abort` on
  the caller.

## The record and the schema

- The processed-scan NeXus layout is declared in **`xrd_tools.io.schema`**
  (schema-as-code): attribute keys, row-aligned dataset sets, axis names,
  capability attributes.  Writers, validators, readers, and test fixtures
  consume it — the schema is data, not prose + discipline.
- The on-disk format is **frozen + additive-only**.  Attribute keys keep
  their historical `ssrl_` prefixes; the byte-compat gate
  (`tests/core/test_v2_record_compat.py`) pins the written record.
- One frame, one record: `FrameView` (today) → `FrameRecord` is
  the immutable, round-trippable unit — integration results + source ref
  + geometry + diagnostics.  Phase 5 A-Steps A/B/C have LANDED: the
  `FrameRecordStore` is wired into the live path, owns eviction + worker-thread
  hydration, and reads are store-first (`record_store → publication →
  data_1d/2d` fallback), with `LiveFrameSeries` demoted to write-side staging.
  N1 portability: raw-source paths are stored
  relative to `entry/@source_base` (design:
  `design/design_project_root_paths_jun2026.md`).

## The display model

Background threads write data → the GUI computes *what to show* as
immutable state → a thin renderer draws it, all generation-stamped
(`display_logic.py` is the pure, Qt-free decision core; a purity guard
enforces it).  The `PublicationStore` is becoming the sole display
contract (Phase 3); `data_1d`/`data_2d` are interim hydration mirrors.

## Acceptance gates (non-negotiable)

- **live≡batch≡reload equivalence**
  (`tests/xdart/test_gi_batch_real_data.py::test_*_equivalence`) — the
  same scan processed live, in batch, and reloaded from disk must produce
  identical publications.  A failure is a bug, never a tolerance to widen.
- **Byte-compat gate** (`tests/core/test_v2_record_compat.py`) — the
  written record's content signature is pinned; a diff means the on-disk
  format changed.
- Architecture guards (`tests/core/test_architecture_guards.py`) and the
  display purity guard keep the layering honest mechanically.

## Decisions on file

See `decisions/` (ADRs).  Highlights: xarray lives at the read boundary
only (ADR-0001); schema evolution is integer version + per-feature
capability attributes (ADR-0002).

## Document map

- `design/` — living design docs (roadmap, greenfield design, deep
  review, the deferred-items register, the current implementation plan).
- `decisions/` — ADRs.
- `history/` — completed-effort records (monorepo migration plan/handoff,
  pre-release fix review) + an index of the pre-monorepo review-cycle
  archive.
- `legacy/` — the two repos' pre-monorepo docs, kept verbatim for
  provenance.
- `core/`, `gui/` — topic notes (some predate the monorepo; headers say
  so).
