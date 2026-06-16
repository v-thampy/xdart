# Design checkpoint — Steps 7+8: one authoritative store (ADR-0005 store→session) + retire data_1d/data_2d

**Date:** 2026-06-14 · **Status:** design checkpoint, needs maintainer sign-off on the open
decisions before build. **Builds on:** ADR-0003 (multi-result record), ADR-0004 (event/threading;
§4 cadence interim), ADR-0005 (store ownership → session). **Trigger:** the Round-12 P1 finding
(silent >64-frame aggregation truncation) is the direct symptom of the two-store half-migration that
7+8 exist to remove.

## Why 7 and 8 are one push
- **Step 8** = delete the legacy integration caches (`data_1d`/`data_2d`/`hydrated_raw`) so the
  payload+record are the sole 1D/2D source (keep h5viewer `_ViewerRows`).
- **Step 7** = move the authoritative `Mapping[frame_index → FrameRecord]` store into
  `xrd_tools.session`; `PublicationStore` becomes a derived projection (ADR-0005).
- **Coupling:** deleting `data_1d` (8) requires whole-scan aggregation to source ALL frames WITHOUT
  holding them all in RAM — `data_1d = FixSizeOrderedDict(max=0)` was the unbounded backstop. That
  capability (store-owns-aggregation + persist-before-evict) IS the ADR-0005 session store (7). So 8
  cannot land cleanly without 7's capability; do them as one coordinated structural push.

## Current state (grounded in code)
- `xrd_tools.session.ScanSession` holds the streaming `ReductionSession` + event contract
  (`submit`/`on_frame_completed`); **no record store yet** — 7 adds it.
- `PublicationStore` (xdart): bounded per-mode `FrameRecord`s (`max_heavy_items=64`); `get_or_hydrate`
  exists but is a per-frame path.
- `data_1d = FixSizeOrderedDict(max=0)` — unbounded, the de-facto aggregation backstop; `data_2d` cap 40.
- Aggregation (`get_frames_int_1d`): reads `data_1d`, falls back to per-frame `_hydrate_frame_from_disk`
  → O(N) disk opens.
- **`io.read.get_1d(file, frame=None)` returns the FULL stacked 1D `(n_frames, n_q)` in one read**
  (and `get_2d` likewise) — the clean aggregation source.
- Multi-mode persistence: per-mode nested subgroups are writable + readable (Step-1/2 gate), but **no
  production write path feeds the accumulated record yet** (Round-11 in-memory-only gap) → non-primary
  GI modes live only in the in-memory store.

## Target architecture (ADR-0005)
- **Authoritative store → `xrd_tools.session`** (headless): bounded `Mapping[idx → FrameRecord]`,
  persist-before-evict, store-owns-hydration.
- **`PublicationStore` → derived projection** over the session store + GUI-only display artifacts
  (thumbnail tier, raw-2D window, publication verdict). xdart holds no authoritative data.
- **Cadence/eviction policy → session** (refines ADR-0004 §4, which kept it xdart-side as interim);
  the **h5pool-bracketed flush MECHANISM + Qt marshaling stay xdart** (session decides WHEN; xdart
  executes the write on its single writer thread + file lock).

## The aggregation design (the crux — what replaces data_1d)
Whole-scan Sum/Average/Overall must source ALL frames without holding them all in RAM. Recommended:
- Aggregate from the **on-disk stacked dataset** (`io.read.get_1d/get_2d(file, frame=None)` →
  `(n_frames, n_q)`) in one bounded read, `nanmean`/`nansum` **off the GUI thread**.
- **Persist-before-evict** guarantees every evicted frame is already on disk → for a LIVE in-progress
  scan: aggregate = on-disk flushed prefix (stacked read) ⊕ in-memory unflushed tail (session store).
  No `data_1d`.
- O(one read), not O(N hydrations) — strictly better than the legacy per-frame path; chunk the stacked
  read if `n_frames×n_q` is large (streaming aggregate).
- This is also the correct fix-forward for the Round-12 P1: aggregation no longer depends on the
  bounded store membership OR on `data_1d`.

## Open decisions (need sign-off before build)
1. **Aggregation source:** stacked-disk-read + in-memory tail **(RECOMMENDED)** vs per-frame
   `get_or_hydrate`. Recommend stacked-read.
2. **Multi-mode aggregation source:** a whole-scan aggregate in a NON-primary GI mode needs that
   mode's stack on disk. (a) **persist ALL per-mode stacks** — wire the multi-mode write (closes the
   Round-11 in-memory-only gap; disk-read then works for any mode) **(RECOMMENDED)**; (b) primary
   from disk + re-integrate non-primary on demand; (c) restrict whole-scan aggregate to the primary
   mode. Recommend (a): the multi-mode write is owed anyway for durable multi-mode, and it makes
   aggregation-from-disk uniform.
3. **Cadence/eviction location:** policy into the session per ADR-0005 **(RECOMMENDED)**, flush
   mechanism stays xdart. Confirm.
4. **Off-GUI-thread aggregation:** the stacked read + aggregate is I/O → must be off the GUI thread
   (reuse the D2 hydration-worker pattern), not a blocking read. Confirm.

## Gated sub-step sequence (each: spine + byte-compat green; commit separately)
- **7a (headless, additive):** bounded `FrameRecord` store + persist-before-evict + store-owns-hydration
  in `xrd_tools.session`; headless-tested; dormant (not yet the GUI's source).
- **W (write wiring):** feed the accumulated record's per-mode subgroups to the writer (closes the
  Round-11 in-memory-only gap) → the on-disk stack carries all modes. byte-compat gated.
- **7b (aggregation):** route whole-scan Sum/Average/Overall through the stacked-disk-read + in-memory
  tail, off the GUI thread. Still additive (`data_1d` still present). This is the capability that
  replaces `data_1d`.
- **8a (flip):** `PublicationStore` becomes a derived projection over the session store; live display
  + aggregation read the session store / stacked-read; remove the legacy `update_plot` fallback (the
  payload/session-store is the sole source).
- **8b (delete):** remove `data_1d`/`data_2d`/`hydrated_raw`; keep h5viewer `_ViewerRows`.
  **Done-test:** those identifiers gone from the display layer (the greenfield "done").
- **7c (cadence):** move `FlushPolicy` + eviction policy into the session (mechanism stays xdart).

## Gates / risks
- Spine (live≡batch≡reload) + byte-compat at every commit. **NEW gate:** whole-scan Sum/Average over a
  >64-frame scan correct over ALL frames (the Round-12 case) AND no GUI freeze (off-thread
  aggregation) — add a >64-frame aggregation test that models the production data flow (the Round-12
  test gap).
- Risk: a huge stacked read (`n_frames×n_q`) is a big transient alloc → chunk/stream it if needed.
- Risk: biggest structural change + live-display + live-acquisition path → **live checkpoint required**
  for 8a/8b/7c. This doc is the design-checkpoint-first gate.

## Relationship to features
After 7+8 the ROI/stitching features build on the headless session store as proper headless analysis
plans — no rework. Step 9 (strictness/sentinel, offscreen-gatable) comes after.

## Decisions resolved (Vivek, 2026-06-14)
1. Aggregation source — **CONFIRMED stacked-disk-read + in-memory tail** (not per-frame hydration).
2. Multi-mode aggregation — **RESOLVED: Option 3, primary-mode-scoped whole-scan aggregates.**
   Whole-scan Sum/Average/Overall operate ONLY on the scan's primary (configured) GI mode, whose stack
   is complete on disk; non-primary modes are lazy/partial (ADR-0003) and a whole-scan aggregate is NOT
   served from a partial stack. Whole-scan stats in a non-primary GI mode are deemed low-value and are
   **deferred** — if ever needed, add an explicit, raw-gated "re-integrate mode across the scan" action
   as its own sub-step (NOT a silent instant aggregate). STILL DO: persist the accumulated per-mode
   results that exist (durability + instant mode-switch after reload, closing the Round-11 in-memory-only
   gap) — this is the "W" sub-step, but it is for durability/instant-switch, NOT an aggregation
   prerequisite. NEVER serve a non-primary whole-scan aggregate from the partial accumulated stack
   (that would reintroduce the Round-12 P1 silent-truncation).
3. Cadence/eviction — **CONFIRMED policy → session; flush mechanism stays xdart.**
4. Off-GUI-thread aggregation — **CONFIRMED worker (D2 hydration-worker pattern), never a blocking read.**

Net effect on the sequence: 7b aggregation reads the PRIMARY mode's complete on-disk stack only; the W
write-wiring still lands (durable multi-mode for instant-switch-after-reload) but is no longer on the
aggregation critical path. Everything else in the sub-step sequence stands.
