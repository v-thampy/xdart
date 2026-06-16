# Fix: restore read‖reduce overlap in xdart Batch (recommendation A) — 2026-06-15

**Status:** queued, do AFTER Phase A+B of the store→session work (`design_store_session_steps7_8_jun2026.md`);
folds into the "collapse dispatch paths" goal. **Gate:** live≡batch≡reload spine + byte-compat + a live
checkpoint. **Baseline to beat:** `docs/perf_baseline_2026-06-15.md`.

## Symptom
xdart **Batch** is ~50% slower than **Live** for the same scan (651-frame Eiger): 1D 28.3s vs 18.9s; 2D
36.3s vs 24.8s.

## Diagnosis (confirmed by the perf table — NOT a reduction problem)
The reduction work is **identical** in both modes — for 2D, `dispatch (reduce+write)` is 34.1 ms/fr Live
vs 35.1 ms/fr Batch. The gap is entirely **lost read‖reduce overlap**: `read` ms/fr goes from 3.3
(hidden, Live) to 19.2 (exposed, Batch). The reads didn't get slower; they stopped hiding behind
reduction.

Three-part interaction in `image_wrangler_thread.py`:
1. **Batch accumulates before dispatching.** The collect loop reads `_PENDING_FLUSH_SIZE` frames
   (64 for 2D, 256 for 1D; ~`:292`/`:821`) into a pending list before handing the batch to the reduction
   session. Live sets flush_size = 1 and dispatches every frame immediately.
2. **The prefetch queue is only 4 deep** (`XDART_PREFETCH_QUEUE_SIZE = 4`, `~:318`, pushed one-at-a-time
   `~:2289`) → the background Eiger prefetcher can only run ~4 frames ahead before it stalls on a full
   queue.
3. **Result: read and reduce run in phases.** Batch spends a long stretch reading 64–256 frames with no
   reduction happening (prefetcher fills 4 slots and stalls), then a long stretch reducing — reads
   exposed. Live's per-frame dispatch (13–34 ms of reduction each) gives the prefetcher exactly the
   window it needs to keep the queue full, so reads hide.

(Deeper standing limit, see §Deeper: xdart reads single-threaded — one prefetch thread doing the decode,
handing the already-decoded array to the worker pool, so the workers never decode. That caps xdart at the
single-thread decode ceiling regardless of this fix.)

## The fix (recommendation A)
In `_dispatch_batch_streaming`, **submit each frame the instant it is read** instead of accumulating
`_PENDING_FLUSH_SIZE` frames first. The reduction session is already persistent and bounded by
`inflight_max`, so pre-accumulating buys nothing — it only destroys the read‖reduce overlap. This
restores live-like overlap AND unifies batch+live onto one submit path (the "collapse dispatch paths"
direction).

## Caveats — MUST hold (these are the traps)
1. **Preserve the batch-silent invariant.** Per-frame *submit* (into the reduction pipeline) must NOT
   become per-frame *display*. Batch deliberately suppresses per-frame `sigUpdate` and does a single
   end-of-run refresh — that is a recorded design decision and a perf feature. Don't let per-frame submit
   re-enable per-frame GUI updates in batch.
2. **Decouple submit-cadence from save-cadence.** Submit per-frame, but KEEP the save/flush cadence
   (`FlushPolicy` / persist-before-evict; the 256/64 save batching). Do NOT turn "stop accumulating before
   dispatch" into "save every frame." They are two different knobs that `_PENDING_FLUSH_SIZE` currently
   conflates — separate them.
3. **GI batch still freezes the whole-scan grid first.** The per-frame submit must run AFTER the GI
   whole-scan freeze/scout (the frozen grid is set before dispatch); don't reorder so a per-frame submit
   pre-empts the freeze.
4. **Gate:** live≡batch≡reload spine + byte-compat green; a live checkpoint confirming batch output is
   byte-identical to before AND batch is still silent (no per-frame display) AND memory stays bounded.

## Expected payoff (set expectations by mode)
- **2D-batch → ~2D-live** (~36s → ~25s): the big win — 2D's slow cake reduction (34 ms/fr) gives the
  reads a window to hide in.
- **1D-batch: smaller** — 1D is read-bound even in Live (reduction ≈ decode, ~13–15 ms/fr each), so there
  is little reduction to hide the reads behind. A closes the overlap gap; it does not change the 1D
  read-bound floor.

## Do NOT do (rejected alternative)
**B — bumping `XDART_PREFETCH_QUEUE_SIZE` ≥ flush size** is a *confirmation experiment only*, not a fix:
it costs RAM (64×~18 MB ≈ 1.15 GB buffered) and doesn't create real overlap (reduction still doesn't
start until the batch dispatches). Use it only to lock the diagnosis if desired; don't ship it as the fix.

## Deeper win (separate, bigger, LATER — folds into the 7/8 session push)
As of the N1 follow-up, **headless `run_reduction(..., sink=NexusSink/XYESink/...)` auto-selects the
streaming `ReductionSession` path by default**. Memory/no-sink callers still default to chunked because
`result.frames` is their product channel; durable sinks write the product externally and can stream safely.

That headless streaming path calls `session.submit(frame)` with **no image**, and each worker reads+decodes
its own frame lazily via `frame.load_image()` inside `_reduce_frame` (`core.py ~:2055`). So reads
parallelize across N workers and, because HDF5/fabio decode releases the GIL, decode actually parallelizes
— **breaking the single-thread decode ceiling** the xdart pipeline hits. Consequences:
- The xdart "Batch" number does NOT predict notebook throughput — benchmark the headless path separately;
  a simple durable-sink notebook call now uses the streaming path unless `execution="chunked"` is supplied
  explicitly.
- The sub-ceiling xdart win (beyond A) is to adopt that **submit-without-image / read-in-workers** pattern
  in xdart so the worker pool does the decode in parallel. That is the same "parallel decode" idea from
  the earlier perf rounds and folds naturally into the Step 7/8 session work (the session machinery
  already reads in the workers). **A now** (batch→live parity, moderate effort); **read-in-workers later**
  (with the session push) for sub-ceiling.
