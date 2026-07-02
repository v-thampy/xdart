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
  (`submit`/`on_frame_completed`) and an optional `FrameRecordStore`. Phase 5 A-Step-A wires the
  xdart GUI streaming path to create a per-scan, unbounded store and mark it persisted after
  `QtNexusSink.flush()` completes the Nexus save. **A-Step-B DONE (`ca61215b`): display reads now
  consult the session store first, with `PublicationStore` and `data_1d`/`data_2d` retained as
  fallbacks. A-Step-C DONE (this commit; offscreen, live checkpoint PENDING): the session
  `FrameRecordStore` owns the heavy-array bound and worker-thread hydration; `LiveFrameSeries`
  is demoted to write-side staging, with persist-before-evict still marked at the Nexus flush
  boundary.**
- `PublicationStore` (xdart): bounded per-mode `FrameRecord`s (`max_heavy_items=64`); `get_or_hydrate`
  exists but is a per-frame path.
- `data_1d = FixSizeOrderedDict(max=0)` — unbounded, the de-facto aggregation backstop; `data_2d` cap 40.
- Aggregation (`get_frames_int_1d`): reads `data_1d`, falls back to per-frame `_hydrate_frame_from_disk`
  → O(N) disk opens.
- **`io.read.get_1d(file, frame=None)` returns the FULL stacked 1D `(n_frames, n_q)` in one read**
  (and `get_2d` likewise) — the clean aggregation source.
- Multi-mode persistence: per-mode nested subgroups are writable + readable, and **W DONE (this
  commit)** wires both production writers through the shared `xrd_tools.io` record/stack path. The
  accumulated per-mode record now survives crash/reload; non-primary GI modes are durable for
  instant mode-switch after reload, but whole-scan aggregation remains primary-mode-scoped.

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
  in `xrd_tools.session`; headless-tested.
- **A-Step-A (live wiring, additive):** GUI streaming sessions now pass a dormant per-scan
  `FrameRecordStore(max_heavy_items=None)` into `ScanSession`; `QtNexusSink` marks published labels
  persisted only after the durable `.nxs` save. Display and aggregation still read the legacy mirrors.
- **A-Step-B DONE (`ca61215b`, store-first reads):** display access now resolves
  `record_store → PublicationStore → data_1d/data_2d`, projecting the active `(mode_1d, mode_2d)`
  before adapting to the existing renderer. Mirror writes remain intact for fallback and rollback.
- **A-Step-C DONE (this commit; offscreen, live checkpoint PENDING):** `FrameRecordStore` now uses
  the 64-frame heavy-array cap, registers a `read_frame_view` disk hydrator, and hydrates through
  the existing worker thread rather than the GUI thread. `LiveFrameSeries._in_memory_cap` and the
  save-before-evict margin remain as write-side staging inputs to `FlushPolicy`; `QtNexusSink.flush`
  marks both Nexus and store persistence at the durable save boundary.
- **W DONE (this commit, H6):** feed the accumulated record's per-mode subgroups to both production
  writers through one shared `xrd_tools.io` path. The on-disk stack carries every written mode;
  byte-compat remains additive.
- **7b (aggregation):** route whole-scan Sum/Average/Overall through the stacked-disk-read + in-memory
  tail, off the GUI thread. Still additive (`data_1d` still present). This is the capability that
  replaces `data_1d`.
- **8a (flip):** `PublicationStore` becomes a derived projection over the session store; live display
  + aggregation read the session store / stacked-read; remove the legacy `update_plot` fallback (the
  payload/session-store is the sole source). **Unblocked by W/H6.**
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
write-wiring is now done (durable multi-mode for instant-switch-after-reload) and 8a is unblocked, but
W is still not an aggregation source for non-primary partial stacks. Everything else in the sub-step
sequence stands.

## DECOUPLING UPDATE (2026-06-15, after grounding) — supersedes the "7+8 coupled" framing above
**Correction:** the earlier claim "Step 8 can't land without Step 7's session store" is WRONG. Grounding
showed the aggregation capability ALREADY EXISTS — `io.read.get_1d/2d(frame=None)` returns the full
`(n_frames, n_q)` stack in one read, and persist-before-evict is real (`LiveFrameSeries._persisted` =
on-disk boundary; in-memory tail = `_in_memory` − `_persisted`). So "on-disk flushed prefix ⊕ in-memory
unflushed tail" is buildable on existing primitives WITHOUT a new headless session store. The two efforts
are SEPARABLE:

- **Phase A — `data_1d` retirement (the correctness goal + greenfield "done" test). Low risk, existing
  primitives.** Steps: (1) **7b aggregation-from-disk** (stacked read ⊕ in-memory tail; aggregate in
  label space, reconcile the stored `(n_chi,n_q)` vs display `(q,chi)` transpose, coordinate the
  off-thread read with `file_lock`); (2) **fix `get_or_hydrate` thumbnail-as-heavy** rehydration FIRST
  (prerequisite — tier-1 semilight never rehydrates; `data_1d` unbounded masks it today; deleting
  `data_1d` exposes it on scroll-back); (3) **retarget Role-A reads** to the store — NOTE `data_1d/data_2d`
  serve TWO roles: **Role-A** = scan-integration cache (TIFF export, raw-panel mask, image preview,
  `_data_snapshot` availability) → retarget+delete; **Role-B** = viewer-mode arrays (Image/XYE/NeXus) →
  KEEP (the doc's earlier "keep `_ViewerRows`" understated this); (4) **8a** flip scan-1D draw to
  payload-only; (5) **8b** delete Role-A `data_1d/data_2d`, keep Role-B.

  **Phase A blockers/traps (ported from `review_2026-06-15_followup_plan.md` §0.4, now superseded here):**
  - **`update_plot` None-payload blocker.** Retiring the legacy `update_plot` requires full PLOT_1D payload
    coverage for Single/Sum/Average FIRST — `update_plot` is still the None-payload fallback for those roles,
    so it can only be deleted once every 1D role supplies a payload (or the None path resolves to a clean
    clear). Do not delete it before that coverage exists.
  - **C1 — wrangler read-back (silent data loss).** Before deleting Role-A `data_1d`/`data_2d` from
    `staticWidget` + all constructors, verify no writer/reader pair breaks: silent data loss occurs if a
    wrangler reads back what it wrote through these caches. Classify every writer/reader pair and confirm the
    store covers it before removal.
  - **reset_key-not-generation trap.** The payload accumulator (Overlay/Waterfall history and any store-backed
    aggregate) MUST key its reset on a STABLE scan/source `reset_key`, **NOT** `state.generation`. The
    generation bumps every tick as live auto-last grows the selection; keying reset on it would rebuild each
    tick from only the un-evicted frames (`intensity_1d` is heavy → tier-1 eviction drops `has_1d`) = the exact
    cap-truncation the accumulator exists to prevent.
  - **`ImagePayload`/`PlotPayload` immutability.** Make the payload array fields read-only when payloads become
    the SOLE display contract (this A4 boundary) — not partially now, which is inconsistent (the `image` field
    is already mutable). Do the immutability flip as part of the final A4 deletion, not before.
- **Phase B — ADR-0005 store→session relocation (7a + 7c). The architectural "thin xdart" move; biggest/
  riskiest headless change; NOT required to retire `data_1d`.** Build the authoritative `FrameRecord`
  store in `xrd_tools.session`, make `PublicationStore` a projection, move cadence/eviction (FlushPolicy)
  into the session.

**Sequencing decision (Vivek, leaning option 3):** do **Phase A now** (no-regret; all options start here),
then **continue into Phase B this cycle**, with an explicit **go/defer checkpoint at the A→B boundary**
(weigh the YAGNI case: B is the foundation for ROI/stitching, so building it WITH the first feature could
design it against real needs — ADR-0006 lesson). Phase A is independently gated + live-validated before B
starts, and is independently shippable, so 3 degrades cleanly to "ship A, defer B" (= option 1) if B
proves gnarly. Option 2 (coupled, 7a-first) is OBSOLETE — it front-loads the risky store for no reason.
Each phase: spine + byte-compat green per step; A's 8a/scroll-back and all of B are live-checkpointed.
