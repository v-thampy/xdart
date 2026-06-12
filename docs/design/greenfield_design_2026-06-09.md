# Greenfield Design — what I'd do differently from scratch

**Date:** 2026-06-09 · Companion to `deep_review_2026-06-09.md`

This is a thought experiment, not a rewrite proposal. The current codebase converged — through a lot of
review-and-fix cycles — onto a design that is mostly what a careful from-scratch design would produce anyway.
The value of asking "what would be different?" is that each genuine difference points at where to *steer* the
existing code, and each non-difference is a decision you can stop revisiting. Each section ends with the
incremental path for the real codebase.

---

## What I would keep (and stop second-guessing)

- **The three contracts.** `FrameSource` / `ReductionPlan` / `ReductionSink` is the right factoring of this
  domain. Source-agnostic ingestion, declarative plans, injected persistence — a greenfield design lands here.
- **NeXus/HDF5 as the archival format.** Ecosystem interop (pyFAI, silx, nexpy, beamline conventions) and the
  one-file deliverable users value make this non-negotiable. The *role* of the file changes below, not the format.
- **The pure display-decision core.** `display_logic` + immutable state + generation stamps is the
  model-view-update pattern; it killed the stale-draw bug class. From scratch I'd just adopt it on day one.
- **Equivalence as the acceptance gate.** live≡batch≡reload as an executable property is the single best idea
  in the project's test strategy.
- **pyFAI as the integration engine, xarray at the read boundary, headless-core-plus-GUI as a product shape.**

---

## Difference 1 — One repo (workspace), two packages

The deepest *process* problem isn't in either repo; it's between them: version floors that the editable-install
workflow bypasses, coordinated ordered releases, schema written in one repo and read in the other, contract
drift only detectable by manually running two suites, and review docs living in a third directory that neither
repo versions.

From scratch: a single monorepo with two (or three, see below) packages — `packages/xrd-core`,
`packages/xdart`, one lockfile, one test invocation, one CI matrix, atomic cross-package commits. Publishing to
PyPI as separate wheels is a release-tool concern, not a repo-structure concern; ordered publishing becomes a
script, not a runbook. The `review/` folder becomes `docs/` inside the repo and is versioned with the code it
describes.

This single choice deletes entire finding categories from the deep review: C2 (schema split across repos) can't
happen because the writer and its reader tests live in one tree; C4 (floor bypassed) has no floor to bypass;
C7's cross-repo CI is just CI.

*Incremental path:* hard to retrofit fully, but cheap approximations exist — a thin "workspace" repo with both
as submodules + shared CI; or at minimum mirrored GitHub Actions in both repos where xdart's CI checks out ssrl
at head and ssrl's CI runs xdart's seam-touching test subset. Move `review/` into the repos.

## Difference 2 — Draw the boundary at data ownership, not at Qt

The current split is "Qt vs not-Qt". That rule produced a GUI that *owns the canonical write path* for live
data (nexus_writer orchestration, save cadence, append cursors, source refs) — because live acquisition state
had nowhere else to live. Most of the deep review's structural findings (incomplete headless NexusSink, schema
knowledge in xdart, GUI-thread save logic, the file-lock/h5pool pause-resume machinery) descend from that.

From scratch: three layers, with the boundary drawn at *who owns the data*:

1. **`xrd-core`** — pure compute + schema + readers (today's ssrl, minus any orchestration gaps).
2. **`xrd-session`** — a headless *acquisition/reduction service*: owns the live scan state machine
   (start/pause/resume/stop), the single writer, save cadence, eviction, the publication store, and emits a
   stream of immutable frame events. No Qt. Runs identically under a GUI, a notebook, a CLI batch job, or —
   the long-term stretch goal — an autonomous-experiment agent.
3. **`xdart`** — a *view* over an `xrd-session`: renders events, sends commands. Never writes data. Qt-specific
   code shrinks to widgets + a small event→signal bridge.

Two big consequences. First, live≡batch≡reload stops being a property you *test* and becomes a property of the
construction — there is only one path, owned by the session, regardless of front end. Second, the
"thin xdart" goal stops being a constant audit ("does this belong in ssrl?") and becomes structural: xdart has
no API through which to own data even if a future contributor tries.

*Incremental path:* this is exactly where the codebase is already drifting — QtNexusSink + ReductionSession is
two-thirds of an `xrd-session`. The remaining move is lifting the orchestration that still lives in xdart
(nexus_writer's frame-record assembly, save cadence, LiveFrameSeries eviction policy, PublicationStore) into a
headless `ScanSession` object in ssrl that xdart instantiates and observes. Deep-review item S3/C2 ("move the
complete-v2-record orchestration into ssrl") is step one of this path; do it with this destination in mind
rather than as an isolated fix.

## Difference 3 — One frame record, one store

Today a frame is representable as: `ScanFrame` (core), `LiveFrame` (Qt-coupled mirror), `FrameView` (round-trip
record), `FramePublication` (GUI envelope), plus entries in `data_1d`, `data_2d`, and the `PublicationStore` —
four classes and three caches with three eviction regimes, kept consistent by convention at every write site.
Each layer was added for a sound historical reason, and the team is already collapsing them (publications as
sole display contract), but a greenfield design starts there:

- **One immutable `FrameRecord`** (≈ today's `FrameView`, plus source ref, geometry, diagnostics) emitted by the
  session, persisted by the writer, consumed by the display. The GUI envelope (`FramePublication`'s validation
  verdict) is a field on it, not a wrapper class.
- **One store** with one eviction policy (heavyweight payloads dropped beyond a bound, metadata retained,
  transparent rehydration from disk through the readers). `data_1d`/`data_2d` don't exist; "hydration mirrors"
  aren't a concept.
- `LiveFrame`/`LiveScan` reduce to thin command-side handles, or disappear into the session.

*Incremental path:* already underway — finishing X1 (sole display contract) and X2 (retire the parallel
`LiveFrame.integrate_*`) gets most of the way. The end state to aim at: grep for "keep both stores in sync"
comments returning nothing.

## Difference 4 — Separate the live working store from the archival file

A large fraction of the hardest-won machinery exists because the *same HDF5 file* is simultaneously the live
append target, the GUI's browse source, and the archival artifact: file locks shared across threads, h5pool
pause/resume bracketing, single-writer-thread discipline, NFS retry loops, SWMR ambitions (currently broken,
per deep review S6), append cursors, atomic-tmp-rename dances.

From scratch: the live working set is a write-friendly store the session owns exclusively — zarr (chunked,
concurrent-reader-safe by design, append-trivial), or even per-frame files + an index. Readers (GUI panels,
notebooks attached mid-run) read it without any coordination with the writer. The NeXus `.nxs` is produced as
the *export artifact* — at `finish()`, and optionally incrementally in the background for crash safety. The
archival format stops constraining the hot path.

Honest trade-offs: a second on-disk representation; "open the .nxs of a still-running scan" becomes "attach to
the session / read the working store" (arguably the better UX anyway); crash recovery needs the working store
to be durable (zarr is). Bluesky/tiled ecosystems made the same separation for the same reasons — and the tiled
`FrameSource` already on the roadmap would slot into exactly this seam.

*Incremental path:* lowest priority of the differences — the current locking machinery now *works* and is
well-tested. Revisit only if SWMR pressure returns (e.g., users wanting external tools to read mid-run) or when
Tiled/Bluesky integration becomes the active feature; don't build it speculatively.

## Difference 5 — Schema as code, not as prose + discipline

The on-disk schema today is defined by the writer's implementation, guarded by validators, documented in
docstrings, and partially re-encoded in xdart (`_drop_integrated_rows`' hardcoded row-aligned set, axis
signatures) and in hand-built test fixtures. Version `2` is stamped but never read, and "version 2" now covers
several generations of additive features.

From scratch: a single declarative schema module in core — datasets, dtypes, shapes, which datasets are
row-aligned to the frame axis, per-feature capability flags, version — from which the writer layout, the reader
expectations, the validators, and the test fixtures are all *derived*. Readers check the version and the
capability flags; "what is row-aligned" is a query, not tribal knowledge. This eliminates the C2/C3 finding
class by construction and makes additive evolution self-describing (`2.1`-style capability attrs instead of an
overloaded `2`).

*Incremental path:* very retrofittable, high leverage. Start with a `SCHEMA` table in `io/nexus.py` listing the
row-aligned datasets + capability attrs; make `_drop_integrated_rows` (moved into ssrl) and the validators read
it; add the reader-side version check (deep review C1). Grow it opportunistically.

## Difference 6 — CI and contract tests from day one

Not a design insight, but the cheapest counterfactual in this list: both repos have exemplary test suites and
no CI. Most of the review-round archaeology in `review/` (seven+ lettered review cycles) was the manual
equivalent of what a CI matrix with the architecture-guard tests, the equivalence spine, and a sink-contract
test would have caught continuously. From scratch: CI before the second module exists; the sink/source duck
contracts frozen by tests in core (including *which thread calls which hook*); a release script that enforces
publish order and floor bumps mechanically.

*Incremental path:* deep review C4/C7 — a GitHub Actions skeleton is an afternoon; the sink-contract test is a
day. Do both this cycle.

## Difference 7 — Smaller things I'd set on day one

- **Domain naming from the start** (`Frame`/`Scan`, not `EwaldArch`/`EwaldSphere`): the rename consumed multiple
  sessions and still leaves two public `Scan` classes and `Live*` prefixes. Name by domain meaning, not
  implementation metaphor.
- **xarray as the in-memory currency** throughout the result path (reduction emits `xr.Dataset` per frame;
  NeXus written from it), not only at the read boundary. Custom containers earn their place only on the hot
  integration path.
- **One coalescing idiom** (throttle) as a tiny shared utility, instead of per-site timers with divergent
  semantics (the debounce-vs-throttle confusion already bit once and survives in `_absorb_chunk`).
- **`None` as the only missing-value sentinel** in APIs (the NaN-vs-None energy split, the 1.0 Å wavelength
  sentinel — both deep-review findings — are day-one decisions gone latent).
- **Strictness flags on every silent-degradation path** (thumbnail fallback, monitor-skip): default loud for
  headless/analysis use, with the GUI explicitly opting into graceful degradation — never the reverse.
- **Qt choice:** PySide6 via the pyqtgraph shim is right and already done. The session layer (Difference 2) is
  what keeps the GUI replaceable, so no need to relitigate the framework.

---

## Summary table

| # | From scratch | Today's equivalent | Retrofit verdict |
|---|---|---|---|
| 1 | Monorepo/workspace, shared CI | Two repos + floors + runbooks | Approximate with shared CI + submodule workspace |
| 2 | Boundary at data ownership (`xrd-session` layer) | Boundary at Qt; GUI owns live writes | **Steer here** — S3/C2 move is step one |
| 3 | One `FrameRecord`, one store | 4 frame classes, 3 caches | Already converging (X1, X2) — finish it |
| 4 | Working store (zarr) + NeXus export | One HDF5 file does everything | Defer; revisit with Tiled/Bluesky work |
| 5 | Schema as code, versioned + capability-flagged | Writer-defined schema, validators, prose | **High leverage, cheap** — start with row-aligned table + version check |
| 6 | CI + contract tests day one | Manual two-suite runs | Do this cycle |
| 7 | Domain naming, xarray currency, None sentinel, loud-by-default | Mixed, post-hoc | Adopt as policies for new code |

The honest overall answer: a from-scratch design would differ less in *architecture* (the contracts, the pure
display core, and the equivalence spine would all reappear) than in *infrastructure* — where the code lives,
who owns the data, and what is enforced mechanically versus by discipline. The codebase paid for its current
shape in review cycles; the differences above are mostly ways to stop paying interest.
