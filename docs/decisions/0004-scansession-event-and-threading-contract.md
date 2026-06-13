# ADR-0004: ScanSession event, threading, and generation contract

**Status:** accepted · 2026-06-13 · (greenfield Difference 2 — Phase 4f)
**Builds on:** ADR-0003 (frame result cardinality).

## Context

Phase 4f extracts the public headless `xrd_tools.session.ScanSession`: a thin
facade over a streaming `ReductionSession` + a `ReductionSink` that emits
immutable events, so xdart can become a thin Qt event→signal bridge with no
API to own data (Difference 2, "the biggest steer"). The design doc
(`phase4_scansession_design.md` §4f) requires an ADR resolving three open
questions before the event shape freezes. ADR-0003 already fixed the *cardinality*
(single-result events). This ADR fixes the rest.

## Decision

### 1. Which thread fires each event

- **`on_frame_completed(FrameEvent)` fires on the session's single WRITER
  thread**, synchronously *after* the wrapped sink's `write`/`replace` returns.
  This is the only place a completion is observable, and it preserves the HDF5
  single-writer invariant (the sink is still touched by exactly one thread). A
  callback registered here MUST be thread-safe; the **Qt bridge MUST marshal it
  onto the GUI thread with a `QueuedConnection`** (never touch widgets from the
  writer thread).
- **`on_state_change(StateChangeEvent)` fires on the orchestrating (caller)
  thread** — inside `start`/`pause`/`resume`/`stop`/`finish`, after the state
  transition is durable.
- **`on_progress(ProgressEvent)`** fires from *both*: the submit side (caller
  thread, on `submit`) and the completion side (writer thread, after a
  completion). Consumers must treat it as possibly-concurrent and idempotent
  (it carries absolute counts, not deltas).

Errors raised by a callback are caught and logged by the session — a listener
must never be able to kill the writer thread (the T0-7/S1 trap: an escape from
the writer loop deadlocks `submit` and reports false success).

### 2. The `generation` stamp

`FrameEvent.generation` is a **session-level integer the CALLER owns**, set via
`ScanSession.set_generation(n)` and stamped onto every subsequent event. The
session **never auto-advances it**, and in particular **pause/resume/stop do NOT
bump it** — a freeze is not a new view. It exists solely so a display layer can
drop a render computed against a superseded *parameter/selection* generation
(the stale-render guard); a pure-headless caller leaves it at 0 and ignores it.
Param/selection-change generation bumping lives entirely in the GUI
(`displayFrameWidget`), which calls `set_generation` — the headless session has
no concept of "selection."

### 3. The validation verdict stays out of the session

Per ADR-0003: `ScanSession` emits raw results; the publication **validation
verdict is a GUI layer that consumes `FrameEvent`** (it is intrinsically coupled
to the xdart display/publication envelope). The headless session never imports
`FramePublication` / `PublicationStore`.

### 4. `flush` is part of the sink/session contract; cadence is NOT

- `ScanSession.flush(force=False)` delegates to the sink's **optional `flush`
  hook** (documented on the `ReductionSink` protocol). This is the
  promote-`flush`-into-the-contract fix (deep-review P2): consumers call a named
  contract method, not a private `_flush`. The interim `QtNexusSink._flush`
  exposes a public `flush` when the xdart bridge lands.
- The **save *cadence*** (`FlushPolicy`, persist-before-evict, the
  `LiveFrameSeries` eviction bound) stays an **xdart-adapter concern**, NOT a
  headless-session one. A headless notebook sink (`NexusSink`) flushes on its own
  `flush_every`; the eviction pressure that `FlushPolicy` serves only exists in
  the GUI's in-memory frame cache. So `ScanSession` does **not** own a
  `FlushPolicy` (a deliberate deviation from the design doc's §2.1 sketch, which
  predated this separation); the xdart bridge owns it and calls `flush` when due.

## Consequences

- `FrameEvent` shape is frozen for 4f: `frame_index, mode_key, result_1d?,
  result_2d?, metadata, generation, timestamp` (single-result; ADR-0003).
- The session wraps the user's sink in an internal event-emitting decorator that
  forwards every probed hook (`begin/write/replace/finish/abort/worker_process/
  flush`) so wrapping never disables the engine's parallel thumbnail / re-feed /
  abort paths.
- Contract tests assert: events fire on the writer thread (`get_ident` ≠ caller),
  a listener exception cannot kill the run, pause/resume does not bump
  `generation`, and `FrameEvent` is single-result + immutable. The no-Qt example
  proves the headless path runs Qt-free.
- The xdart `ScanSessionAdapter` becomes a thin bridge: it registers a
  `QueuedConnection`-marshalled `on_frame_completed`, owns the `FlushPolicy` +
  h5pool bracket, and maps `on_state_change → sigPaused/sigResuming`. That
  rewiring touches the live acquisition path and is gated on the manual live
  checkpoint (§4); the headless module here is offscreen-provable and additive.
