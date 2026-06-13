# ADR-0003: a frame has ONE result per completion, MANY over its lifetime — single-result events, multi-result records

**Status:** accepted · 2026-06-13 · (greenfield Difference 2 + Difference 3; the
keystone for Phase 4f and Phase 5)

## Context

Three deferred-but-coupled moves are all blocked on one unanswered question, and
two independent deep reviews + a codex pass converged on naming it the keystone:

> **Does a frame have ONE reduction result, or potentially MANY?**

It bites because of grazing-incidence (GI) sub-modes. A GI frame integrated under
the active `(gi_mode_1d, gi_mode_2d)` selection yields one 1D + one 2D result;
switching the sub-mode (e.g. `q_total → q_ip`, `qip_qoop → qz_qxy`) yields a
*different* result for the *same frame*. Today the GUI keeps that switch instant
with per-mode caches (`data_1d` / `data_2d` and the per-mode products `LiveFrame`
carries), which is exactly the substrate Difference 3 ("one record, one store")
wants to delete.

The three blocked moves:

1. **Difference 3 / Phase 3 (X1): collapse to one store.** Cannot delete
   `data_1d` / `data_2d` until their per-mode role has a home.
2. **Phase 4f: the public `xrd_tools.session.ScanSession` emits `FrameEvent`s.**
   Freezing the event shape commits to single- or multi-result *now*.
3. **Phase 5: `FrameRecord`** is the record `LiveFrame` / `FramePublication` /
   `data_1d` / `data_2d` collapse into. Its cardinality must match the events.

The current working decision (commit `e616c3a`) is "publications stay
single-result." Both reviews accept that as defensible **but warn it is unmade as
a contract**: if 4f freezes `FrameEvent` single-result while Phase 5's
`FrameRecord` must be multi-result-capable, that is a seam mismatch being locked
in, and the dicts can never be deleted in one move. Hence this ADR — *before*
4f freezes anything.

## Decision

**Cardinality is split by lifecycle stage:**

- **A reduction *completion* is single-result.** One frame, integrated under one
  active mode key, produces one `result_1d` and/or one `result_2d`. The
  `FrameEvent` emitted by `ScanSession.on_frame_completed` carries exactly that,
  plus the **mode key it was computed under**:

  ```
  FrameEvent: frame_index, mode_key, result_1d?, result_2d?, metadata,
              generation, timestamp        # immutable, single-result
  ```

  This matches physical reality (the writer thread finalizes one reduction at a
  time) and keeps the event small, immutable, and safe to freeze in 4f.

- **A frame *record* is multi-result.** `FrameRecord` accumulates results as
  modes are computed, keyed by mode:

  ```
  FrameRecord: frame_index, source_ref, geometry,
               results_1d: Mapping[Mode1dKey, IntegrationResult1D],
               results_2d: Mapping[Mode2dKey, IntegrationResult2D],
               active_mode_key, diagnostics, publication_verdict
  ```

  The single store is `Mapping[frame_index, FrameRecord]`. A completion event
  **upserts** into the matching record: `record.results_1d[event.mode_key] =
  event.result_1d` (and/or 2D) and updates `active_mode_key`. The per-mode caches
  `data_1d` / `data_2d` become the `results_1d` / `results_2d` fields of the
  record — deleted in **one move**, not piecemeal.

- **The mode key.** A hashable identifying the integration variant: for GI it is
  the `gi_mode_1d` (1D) / `gi_mode_2d` (2D) sub-mode; for standard scans it is a
  single trivial key. Per-dimension maps mirror today's two-dict split and let a
  1D-only (`skip_2d`) run carry only `results_1d`.

- **The validation verdict stays a GUI layer ON the record/events** — never in
  `ScanSession` (preserves the Phase-3 decision; the headless session does not
  own the publication contract).

So: **events single-result, records multi-result, store keyed by frame index.**

## Rationale

- **Events model "what just happened"** — one integration finished. A
  multi-result event would force the producer to either eagerly compute every
  mode (it does not — modes are computed lazily on user switch via a fresh
  submit) or ship a one-entry map (awkward, and dishonest about what occurred).
  Single-result events are the truthful shape of a streaming completion.
- **Records model "what we know about this frame"** — the union of every mode
  computed so far. Multi-result records are the truthful shape of the cache the
  GUI already maintains; making it a named field of one record is the *whole
  point* of "one store."
- **It unblocks all three moves without a seam mismatch.** 4f freezes a
  single-result `FrameEvent` safely; Phase 5 builds a multi-result `FrameRecord`
  that absorbs `data_1d` / `data_2d` in one deletion; the store is genuinely one
  map, with per-mode results a field rather than a parallel structure.
- **Mode-switch stays instant when cached, lazy when not:** the GUI asks the
  record for `mode_key`; a hit renders immediately, a miss triggers a re-submit
  whose completion event upserts the new mode.

## Alternatives considered

- **Multi-result events (`results: Mapping[Mode, Result]`).** Rejected: a
  completion never produces more than one mode, so the map is always a singleton
  at emit time; it would only ever be filled by the *consumer* merging events —
  which is exactly what the record does. Pushing the map into the event conflates
  "what happened" with "what is accumulated."
- **A separate headless per-mode cache object the session owns** (Finding 1's
  option b, as a distinct object). Rejected as a *separate* object: it
  reintroduces a second store next to the record. Realizing the per-mode map as a
  **field of `FrameRecord`** keeps the single-store promise while serving the same
  need.
- **Keep single-result everywhere, re-integrate on every switch (no cache).**
  Rejected: a mode toggle on a large scan would re-reduce every visible frame —
  the latency regression the per-mode cache exists to prevent.

## Consequences

- **Phase 4f may freeze `FrameEvent` as single-result + `mode_key`.** Unblocked.
- **Phase 5 `FrameRecord`** carries `results_1d` / `results_2d` maps + active
  mode; `data_1d` / `data_2d` are deleted in one move (their per-mode role *is*
  those maps); `LiveFrame` thins to a frame-build/source shell; `FramePublication`
  becomes a verdict-carrying view over a record.
- **The store collapse (Difference 3 / X1) is a Phase-5 single-move**, not a
  piecemeal cache retirement — which is why it was correctly deferred out of
  Phase 3 once this question surfaced (it was the *unmade decision*, now made,
  that blocked it).
- **Standard (non-GI) scans** carry a one-entry map — no special-casing; the
  record shape is uniform.
- The `live ≡ batch ≡ reload` spine is unaffected: equivalence is asserted
  per (frame, mode), which a single-result event/record-entry expresses directly.

## Status note for the maintainer

This is the single most consequential architectural commitment of the remaining
cycle (it shapes both the public event API and the Phase-5 record). Both deep
reviews state "either answer is fine — the *unmade* answer is the problem." It is
recorded here so 4f does not freeze the event shape against an undecided record.
If you prefer multi-result events, the only code committed against this so far is
`FrameEvent` in Phase 4f — reversible until Phase 5 builds `FrameRecord` on it.
