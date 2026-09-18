# xrd-tools architecture

One distribution, two import packages, one direction of dependency:

```
xdart  (Qt GUI: Scattering Workspace + widgets)     ── view / commands
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
is its streaming engine. The sole built-in page is the current Scattering
Workspace (`src/xdart/gui/tabs/scattering/`), whose adapters drive the public
session/source APIs while the GUI remains an event-to-view consumer. Display
decisions live under `xrd_tools.session.display_logic`; there is no second
Static Scan page or xdart compatibility shim. Whole-scan GI preparation is
owned by the headless core.

## North star

(Full version: `design/roadmap_2026-06-10.md`; the plan that executed the
from-scratch framing was a local-only CC_ note, not published.)

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

Integrity complexity is proportional to demonstrated risk.  Finite operations
keep raw/source artifacts immutable and publish closed, scientifically validated
results into stable operation slots.  Ordinary streamed Run keeps its current
high-throughput output transaction, Overwrite, and STOP semantics; this policy
changes only its target naming.  Derived finite operations atomically replace
their processed slots while leaving the prior slot untouched until publication.
Scientific correctness remains independently non-negotiable, and finite
operations do not gain a general crash-recovery state machine (ADR-0010).

The public slots for one `<family>` are `_int1d.nexus`, `_int2d.nexus`,
`_average.nexus`, `_reintegrate1d.nexus`, `_reintegrate2d.nexus`,
`_stitch1d.nexus`, `_stitch2d.nexus`, and `_rsm.nexus`.  Public filenames
never expose a version or hash; exact identities remain in provenance.  An
automatically named grazing-incidence result starts the family `<scan>_gi`
(`sample_gi_int2d.nexus`), decided once where the family is first derived and
then persisted, so later operations consume it verbatim; an explicit filename
is used as written and the scientific mode is never read back from a name.
Finite Average, Reintegration, Stitch, and RSM publish a hidden same-directory
candidate only after close and scientific validation.  STOP discards a partial
candidate.  Run is not converted to that protocol: a hard crash may leave its
public result or hidden backup unresolved, and automatic recovery is not
guaranteed.  The raw/source artifact remains safe.  Append and Live follow their
separate continuation contract.

Stable-slot publication belongs outside the hot integration and writer loop.
It must leave Run's output-transaction loop unchanged and preserve the accepted
promotion performance floors.

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
  hydration, and reads are store-only: `record_store → bounded publication
  projection → disk hydration` (the Role-A `data_1d`/`data_2d` mirrors are
  retired; viewer rows stay viewer-scoped), with acquisition staging bounded
  by the source and session resource owners.
  N1 portability: raw-source paths are stored
  relative to `entry/@source_base` (design:
  `design/design_project_root_paths_jun2026.md`).

## The display model

Background threads write data → the GUI computes *what to show* as
immutable state → a thin renderer draws it, all generation-stamped
(`xrd_tools.session.display_logic` is the pure, Qt-free decision core; a
purity guard enforces it). The session `FrameRecordStore` is authoritative;
`PublicationStore` is a bounded derived projection (H8), and the legacy
`data_1d`/`data_2d` mirrors are deleted (H9).

## Acceptance gates (non-negotiable)

- **scientific run/reload equivalence** — the current core reduction spine,
  multimode record round-trip, and Scattering Workspace GI-axis parity gates
  must agree on canonical results. A failure is a bug, never a tolerance to
  widen.
- **authenticated real-data promotion** — promotion names its beamline corpus,
  fails if that corpus is unavailable, and runs the applicable pyFAI, Stitch,
  RSM, and persisted-reload scientific/performance gates. Synthetic contract
  fixtures do not substitute for this evidence (ADR-0010).
- **Byte-compat gate** (`tests/core/test_v2_record_compat.py`) — the
  written record's content signature is pinned; a diff means the on-disk
  format changed.
- Architecture guards (`tests/core/test_architecture_guards.py`) and the
  display purity guard keep the layering honest mechanically.

## Decisions on file

See `decisions/` (ADRs).  Highlights: xarray lives at the read boundary
only (ADR-0001); schema evolution is integer version + per-feature
capability attributes (ADR-0002); integrity complexity is proportional to
demonstrated risk and promotion requires real-data evidence (ADR-0010).

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
