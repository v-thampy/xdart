Now I have a complete picture. Let me compile the final comprehensive gap inventory:

---

## GAP INVENTORY: Difference 2 — The XRD-Session Layer Boundary

### ASSIGNED TASK
Inventory EVERY piece of acquisition/reduction orchestration still living in xdart that the design says belongs in a headless ScanSession: (1) run-state machine; (2) pause/resume; (3) save cadence + append cursor + NFS retry; (4) LiveFrameSeries eviction; (5) PublicationStore creation/clearing; (6) what ReductionSession + QtNexusSink already own; (7) wrangler threads' dispatch paths.

---

### EXECUTIVE SUMMARY

**The monorepo migration (Jun 2026) shipped xdart as a 2-thirds realized view over a `ReductionSession` backend.** QtNexusSink (the xdart v2 NeXus writer) is ALREADY registered as a ReductionSink: the headless streaming orchestration (parallel integration, bounded in-flight windows, single-writer-thread discipline, frame-index-addressed sinking) is COMPLETE and proven in production. The remaining xdart-owned state is: (a) **GUI control surface** (_enter/_exit_run_state, h5pool pause/resume lifts, display refresh gates); (b) **live-specific eviction policy** (persist-before-evict bounds, eviction trigger thresholds); (c) **live-specific save cadence** (LIVE_SAVE_INTERVAL, frame-count accumulation, dual-path flush logic); (d) **scanner-side frame registry** (mapping incoming live frames to the session's integration futures, registering before submit, popping at write); (e) **publication envelope** (gui-side validation, display hand-off signal). The boundary work is "lift the orchestration that still lives in xdart" — specifically: nexus_writer's frame-record assembly, save cadence, eviction policy, and the PublicationStore — but **NOT the display/Qt parts** (those belong in xdart forever).

The **WS-X1 Phase-2 refactor already completed this for batch/reprocess live**: a persistent `ReductionSession(execution="streaming")` + `QtNexusSink` drives all three paths (batch, reprocess, live watch phase 2 tail). The remaining move is **structuring the headless API surface so a ScanSession can exist without xdart at all** — because today the Qt parts are interwoven in the wrangler thread dispatch.

---

## (1) RUN-STATE MACHINE
**Living in:** `src/xdart/gui/tabs/static_scan/static_scan_widget.py:1066–1150`  
**Size:** ~85 LOC (two methods: `_enter_run_state()`, `_exit_run_state()`)  
**Lines:** 1066–1099 (`_enter_run_state`), 1100–1122 (`_exit_run_state`), 1124–1149 (pause/resume lifts)

**What it does:**
- `_enter_run_state()` (line 1066): Marks a wrangler/integrator run active. Idempotent (re-entry is no-op).
  - Sets `self._run_active = True`
  - Calls `self.displayframe.set_processing_active(True)` (display persist flag)
  - Calls `self.h5viewer.set_run_writing(True)` (disk-read freeze guard for viewers)
  - Disables integration controls + dialog boxes (per-widget, not just blanket)
  - **Wired to:** wrangler's `started` signal + integrator thread `started` (reintegrate path)

- `_exit_run_state()` (line 1100): Marks the run finished. Idempotent (exiting idle is no-op).
  - Sets `self._run_active = False`
  - Calls `self.displayframe.set_processing_active(False)` (drop persist window)
  - Calls `self.h5viewer.set_run_writing(False)` (clear disk-read guard, re-fire selection for skipped frames)
  - Re-enables integration controls with mode-correct state (Int 1D vs Int 2D vs viewer)
  - **Wired to:** all `finished` signal paths (wrangler, integrator, exceptions)

- `_on_run_paused()` (line 1124): **Pause phase B** — the run is FROZEN at a frame boundary (worker is idle).
  - LIFTS `set_run_writing(False)` so the user can browse ANY frame from disk while paused
  - Keeps `_run_active = True` (controls are hard-disabled, not just integration ones)

- `_on_run_resuming()` (line 1140): **Resume phase B** — RE-ENGAGE the freeze guard BEFORE the worker re-enables.
  - Calls `h5viewer.set_run_writing(True)` (cancels any in-flight browse load)
  - Calls `displayframe.set_processing_active(True)` (re-engage freeze)
  - Synchronous, same GUI thread, ahead of wrangler's command flip

**Could move headless as-is?** NO (partial). The CORE state machine (enter/exit/paused/resuming timestamps + **whether to allow UI interactions**) is intrinsically GUI. BUT the underlying **run/paused/idle state flags** are liftable — a headless ScanSession needs to track lifecycle, so callers can (a) know when the writer is idle for pause safety, (b) gate image submission during stop, (c) report progress. The **display refresh persistence** is xdart-only.

**Current coverage in ssrl:**
- ✅ `ReductionSession._started`, `._finished`, `._cancelled` (lifecycle)
- ✅ `ReductionSession.drain(timeout)` (blocks until writer idle, for pause)
- ✅ `QtNexusSink` knows the single writer thread's ident (`_writer_ident`)
- ❌ No explicit "paused" state property (pause is managed in xdart's `command` queue)
- ❌ No run-active boolean accessible to callers

---

## (2) PAUSE/RESUME + H5POOL FILE-LOCK BRACKETING
**Living in:**
- `src/xdart/gui/tabs/static_scan/wranglers/wrangler_widget.py:103–108` (signal decls)
- `src/xdart/gui/tabs/static_scan/wranglers/image_wrangler_thread.py:1415–1500` (pause/resume orchestration)
- `src/xdart/gui/tabs/static_scan/scan_threads.py:414–429` (reintegrate path h5pool bracketing)
- `src/xdart/utils/h5pool.py:1–95` (file-lock pool singleton)

**Size:** ~200 LOC (pause/resume machinery + h5pool)

**What it does:**

**File-lock philosophy:**
- `h5pool.H5FilePool` (src/xdart/utils/h5pool.py) is a process-wide singleton: manages LRU-capped read-only HDF5 file handles.
- Writers call `pause(path)` before opening for write: closes any cached read handle, marks the file "paused" so readers return `None`.
- Writers call `resume(path)` after write completes: unmarked file, readers can reopen.
- **Single-writer invariant:** only the session's one writer thread EVER opens the file for writing; only the GUI threads read.

**Pause sequence (xdart side):**
1. User clicks Pause button → wrangler receives `'pause'` command
2. `_wait_if_paused()` (line 1416) checks command, calls `_enter_pause()` if paused
3. `_enter_pause()` (line 1441):
   - Drains any open `ReductionSession` (line 1466: `session.drain(timeout=30s)`)
   - If serial path active (`_frames_since_save > 0`): calls `h5pool.pause(scan.data_file)`, `scan._save_to_nexus()`, `h5pool.resume()` (lines 1482–1487)
   - Else if streaming active: calls `sink._flush(force=True)` (line 1493)
   - Emits `sigPaused()` AFTER drain/flush (line 1500) so writer is provably idle
4. GUI thread (`_on_run_paused()`) lifts disk-read freeze
5. User can browse; reintegrate path also brackets with h5pool (scan_threads.py:422–429)

**Resume sequence (xdart side):**
1. User clicks Resume → wrangler receives `'start'` command
2. `_on_run_resuming()` (static_scan_widget.py:1140): RE-ENGAGES freeze BEFORE command flip
3. Loop unblocks in `_wait_if_paused()` (image_wrangler_thread.py:1434), continues on same session

**Could move headless as-is?** PARTIAL.
- ✅ `ReductionSession.drain(timeout)` is already headless (line 990 in core.py)
- ✅ The concept of a quiesce-at-frame-boundary is lifted (drain completes before UI resumes)
- ❌ The **dual-path logic** (serial vs streaming) is xdart-specific — ssrl doesn't know about image wrangler's serial `_process_one` vs streaming paths
- ❌ The **h5pool pause/resume** assumes a shared single .nxs file used by both GUI readers + worker writer. A headless service would use that same pattern IF using HDF5, but it's NOT mandatory — a working-store abstraction (Difference 4) would decouple it.
- ❌ The **signal emission** (`sigPaused`, `sigResuming`) is Qt-specific.

**Current coverage in ssrl:**
- ✅ `ReductionSession.drain(timeout)` (line 990: waits for writer queue to empty)
- ✅ Streaming writer thread exists and is tracked (`self._writer_thread`, `self._writer_ident`)
- ❌ No explicit "paused" callback/signal
- ❌ No pause-safe guarantee that the writer won't restart mid-browse

**Risk:** The drain timeout (30s) is hardcoded in image_wrangler_thread.py:1439. If a worker hangs, pause times out and warns, but proceeds (RS-1 comment: flush is skipped, tail flushes on resume/finish). This is tolerant but could leave unpersisted frames in memory. A headless ScanSession expose this timeout as configurable.

---

## (3) SAVE CADENCE + APPEND CURSOR + NFS RETRY
**Living in:**
- `src/xdart/modules/ewald/nexus_writer.py:1–450` (GUI writer adapter, frame-record assembly, NFS retry)
- `src/xdart/gui/tabs/static_scan/wranglers/qt_nexus_sink.py:240–290` (hydration, stashing, buffering, save trigger)
- `src/xdart/gui/tabs/static_scan/wranglers/image_wrangler_thread.py:1502–1560` (save-cadence logic, LIVE_SAVE_INTERVAL)

**Size:** ~300 LOC cumulative (nexus_writer frame assembly ~250, qt_nexus_sink hydrate/stash ~50, image_wrangler save logic ~50)

**What it does:**

**Append cursor (NexusWriteCursor):**
- Defined in `nexus_writer.py:65–78`: `{path, groups: {group_path: (row, end, shape)}, metadata: (row, end, shape), dropped: {group_path: set(labels)}}`
- `_write_cursor()` (line 81): Gets or creates a cursor for a given file, cached on the scan (`scan._nexus_write_cursor`)
- Cursor tracks **per-group append positions** (where each stacked dataset's next row goes) so the writer can append incrementally instead of rewriting the entire file each save
- Tracks **dropped labels** (publication-gate-rejected frames) so they aren't re-loaded inside the open handle on every save

**Save cadence (image_wrangler_thread.py):**
- `LIVE_SAVE_INTERVAL` (hardcoded, not exposed; typical ~8–16 frames)
- `_frames_since_save` counter accumulates per-frame submissions
- `_save_due()` (line 1502): Returns True when:
  - Forced (final flush), OR
  - `_frames_since_save >= LIVE_SAVE_INTERVAL`, OR
  - Unsaved in-memory frames >= `cap - 8` (persist-before-evict check)
- On save: `h5pool.pause()`, `scan._save_to_nexus()`, `h5pool.resume()`, `_flush_xye_buffer()`, reset counter

**NFS retry (nexus_writer.py):**
- `_open_with_retry()` (line 124): Retries transient OSErrors when opening .nxs via nx.nxopen (100 tries, 0.05s sleep, same semantics as ssrl's catch_h5py_file)
- Used in two places: nexus_writer frame assembly, and reintegrate h5 reopens

**Could move headless as-is?** PARTIAL.
- ✅ The **append cursor concept** is liftable; the headless sink write path already knows it (NexusSink in core.py calls write_nexus_frame, which writes to open h5)
- ✅ The **NFS retry logic** is pure and could live in a headless helper
- ❌ The **save cadence** is xdart-specific: it's based on live frame arrival rate + in-memory bounds. A headless reprocessing job has NO live window pressure (batch reads everything upfront); a headless *live monitor* would, but the session doesn't know when frames are "arriving" — it only knows when they're submitted
- ❌ The **dual-path logic** (serial `_process_one` → manual save vs streaming QtNexusSink → auto save cadence): streaming sinks should drive their own flush timing, but serial paths (like true-live watch) need external cadence. This is a caller's concern, not the session's

**Current coverage in ssrl:**
- ✅ `NexusSink` writes frame-by-frame via `write_nexus_frame()` (core.py:487–502)
- ✅ `NexusSink.write()` internally calls `self._h5.flush()` on a `flush_every` cadence (line 501)
- ✅ Atomic mode uses temp-file + rename (core.py:469–472, abort preserves as .partial)
- ❌ No **append cursor** (the ssrl sink rebuilds the stacked group each write, not appending)
- ❌ No **NFS retry** in the sink itself (delegates to `open_nexus_writer()`, which uses h5py's default error handling)
- ❌ No **live-specific cadence logic** (sink writes every frame or on flush_every, not per-batch)

**Risk:** The xdart save cadence is tuned for live acquisition (8 frames = ~0.1s at typical detectors). A reintegrate batch on a long scan would save VERY frequently (once per 8 frames) because the cadence is based on frame count, not elapsed time. A headless ScanSession should expose `flush_every` + let the sink decide (streaming sinks should implement their own throttle, not rely on the session to batch)

---

## (4) LIVEFRAMESERIES EVICTION (PERSIST-BEFORE-EVICT)
**Living in:** `src/xdart/modules/ewald/frame_series.py:450–530`
**Size:** ~80 LOC

**What it does:**
- `LiveFrameSeries._in_memory` (line 453): Dict `{idx: LiveFrame}`, capped at `_in_memory_cap = 64` (hardcoded)
- `LiveFrameSeries._persisted` (line 461): Set of indices safely written to disk
- `__setitem__()` (line 495): On frame stash, evicts oldest-first to keep memory bounded — **BUT ONLY if persisted** (line 506: `if idx in self._persisted`)
- `mark_persisted(idxs)` (line 510): Called by writer after successful save to record which frames are now evictable
- `unsaved_in_memory_count()` (line 522): Query: how many in-memory frames are NOT yet persisted
- The invariant: **An unpersisted frame is NEVER evicted** (its int_1d/int_2d live only in memory; losing it = silent data loss)

**Key insight:** This is the **data-loss fix** (persist-before-evict). The xdart writer reads integration results straight off the LiveFrame object in memory; evicting before save = data loss. The session does NOT know about in-memory caches — it doesn't manage this.

**Could move headless as-is?** NO (xdart-specific). A headless reduction pipeline doesn't have a LiveFrame cache — it integrates on-demand and streams the results. A headless *live session* would need a similar invariant IF it caches raw images, but that's out of scope for Difference 2 (which assumes each frame is processed exactly once, results persisted, then raw dropped).

**Current coverage in ssrl:**
- ❌ No in-memory frame cache (ReductionSession doesn't hold Frame objects after integration; they're garbage-collected or explicitly cleared with `clear_frame_images=True`)
- ❌ No eviction policy (headless mode assumes unbounded frames or explicit release via `release_products()`)

**Risk:** If a headless ScanSession (in the future) adds a live raw-image cache, it MUST implement persist-before-evict. Today's QtNexusSink gets the eviction for free because it operates on xdart's LiveFrameSeries, not its own cache.

---

## (5) PUBLICATIONSTORE CREATION/CLEARING
**Living in:**
- `src/xdart/modules/frame_publication.py:1–434` (PublicationStore + FramePublication class)
- `src/xdart/gui/tabs/static_scan/static_scan_widget.py:310` (creation), `1291` (clearing)

**Size:** ~434 LOC (publication module) + 2 LOC (widget)

**What it does:**
- `PublicationStore` (frame_publication.py): Singleton per staticWidget, internally locked (threading.RLock)
- Holds `{idx: FramePublication}` — the GUI's **validated display envelope** for each frame
- `FramePublication`: includes integration results + validation verdict (passed gate, rejection reason) + display-side hints
- `upsert(publication)` (per-frame): Adds or updates a publication; increments generation counter on insert
- `clear()`: Called in `_prepare_for_new_scan()` (line 1291) to drop publications from the old scan
- **Generation stamping:** `display_generation` bumps on mode switch + select change; publications only used if their generation matches the display state (stale-draw defense)

**Current role:** The ONLY render source for the display (per CLAUDE.md "Making publications the sole display contract"). The display controller reads `publication.int_1d`/`int_2d` + `publication.verdict` + `publication.errors` to decide what to show.

**Could move headless as-is?** NO. PublicationStore is a GUI envelope (validation verdict is `pass`/`reject`/`uncertain` based on GUI rules like detector-is-sane, wavelength exists, etc.). A headless ScanSession has NO concept of "publication" — it only produces FrameReduction. The frame_publication module imports display logic; it's intrinsically Qt-coupled through the validation checks.

**Current coverage in ssrl:**
- ❌ `PublicationStore` is xdart-only
- ❌ `FramePublication` wraps xdart display rules
- ✅ The underlying `FrameReduction` contract is headless

**Risk:** If the GUI ever needs to reject a frame mid-write (e.g., user marks a frame as "bad" mid-acquisition), the headless session can't revert a write. The model is: **session writes every completed reduction; GUI layers validation on top**. Bad frames stay in the .nxs; the GUI just doesn't render them.

---

## (6) WHAT REDUCTIONSESSION + QTNEXUSSINK ALREADY OWN (TWO-THIRDS COMPLETE)
**Owned in ssrl:**
- ✅ `ReductionSession` (core.py:612–1300+): Streaming execution, parallel workers, bounded in-flight window, single writer thread, per-thread integrators, GI freeze policies, cancel token, progress callback, fail-loud finish
- ✅ Frame-by-frame submission (`submit(frame)`) or chunk-at-a-time (`process(chunk)`)
- ✅ `ReductionSink` protocol (core.py:300–305): `begin()`, `write()`, `finish()`, optional `replace()`, optional `worker_process()`, optional `abort()`
- ✅ `NexusSink` (core.py:418–600): Writes complete v2 record (integrated stacks, per-frame record group with source ref + thumbnail, per-frame geometry at finish, scan metadata, atomic mode with .tmp + rename, fail-safe abort)
- ✅ `QtNexusSink` (wranglers/qt_nexus_sink.py:43–289): Xdart-specific sink that hydrates LiveFrames, stashes in memory, buffers XYE, thumbnails on worker thread (PERF-5), emits per-frame display signal (no-op in batch), manages save cadence

**Owned in xdart:**
- ✅ `_dispatch_batch_streaming()` (image_wrangler_thread.py:1610–1730): Converts pending frames → headless Frame objects, opens a persistent ReductionSession, GI freeze prepass, submits each frame to session.submit(), display hand-off through _published_frames map (NOT through publication store; that's still a separate display hydration path)
- ✅ `open_live_reduction_session()` (xdart.modules.reduction.py:381–444): Adapter that wraps a ReductionSession for xdart's live frames; sets up the QtNexusSink when passed `sink=QtNexusSink(...)`

**The split:**
- ReductionSession = headless orchestration (workers, queues, writer thread, sinking)
- QtNexusSink = xdart display coupling (register, hydrate, publish, thumbnail on worker)
- imageWranglerThread._dispatch_batch_streaming = glue (read images, adapt to Frame, submit)

---

## (7) WRANGLER THREADS' ROLE — DISPATCH PATHS
**Living in:**
- `src/xdart/gui/tabs/static_scan/wranglers/image_wrangler_thread.py:1–2831` (imageThread, the orchestrator)
- `src/xdart/gui/tabs/static_scan/wranglers/wrangler_widget.py:100–700` (base wranglerWidget + wranglerThread command loop)

**Three live paths (image wrangler):**

1. **Phase 1: Collect** (line ~630–900): Read images into pending list, check pause/stop, loop
2. **Phase 2: Dispatch** (line ~700–790): Batch the pending frames
   - **Serial path** (_dispatch_batch_serial, line 1525): One frame at a time via `_process_one()` — integrates, displays, saves every ~LIVE_SAVE_INTERVAL frames
   - **Streaming path** (_dispatch_batch_streaming, line 1610): Persistent ReductionSession + QtNexusSink — integrates in pool, writes in single thread
   - **Parallel path** (_dispatch_batch_parallel, line 1731): Old chunked path (deprecated, fallback)
3. **Phase 3: Watch** (line ~860–880): Detector-rate file watcher — always uses `_process_one()` (serial), one frame at a time, deliberate separate path

**Command loop:**
- `wranglerThread.run()` (wrangler_widget.py:300–600): Receives commands from GUI (`'start'`, `'pause'`, `'stop'`) via a Queue
- `'start'`: Calls `_collect()` which reads pending frames, calls `_dispatch_batch()` which routes to serial/streaming/parallel
- `'pause'`: Sets command to `'pause'`, loop calls `_wait_if_paused()` which blocks until command != `'pause'`
- `'stop'`: Sets command to `'stop'`, loop breaks

**For true-live (detector watch, Phase 3):**
- Separate watch loop uses `_process_one()` directly (not _dispatch_batch)
- Serial `_save_to_nexus()` with manual cadence (no ReductionSession)
- This is intentional: watch is inherently serial (one frame at a time from detector), so parallelism is moot

**Could move headless as-is?** PARTIAL.
- ✅ `_dispatch_batch_streaming()` is already structured to be reusable (adapts LiveFrames → Frame objects, opens session, submits)
- ❌ The **Phase 1 collect** (file watcher loop, rate throttling, pending accumulation) is beamline-specific; a headless ScanSession expects an upstream `FrameSource` that provides frames on demand
- ❌ The **command queue** (pause/stop) is Qt-specific
- ❌ The **Phase 3 watch loop** is detector-rate specific; a headless service would use a different streaming source (Bluesky RunEngine, EPICS monitor, etc.)

**Current coverage in ssrl:**
- ✅ `ReductionSession.submit(frame)` accepts frames on-demand (no batch barrier)
- ✅ `run_reduction(plan, scan)` ingests a complete Scan with all frames upfront (chunked path)
- ❌ No **pause/resume through a command queue** (the session doesn't know about wrangler state)
- ❌ No **streaming FrameSource** (ssrl's FrameSource protocol is for pre-loaded data or lazy-load on demand; live detectors are different)

---

## API SURFACE FOR A HEADLESS SCANSESSION

**Proposed commands in (immutable frame events out):**

```python
class ScanSession:
    """Headless acquisition + reduction service.
    
    Owns: live scan state machine (start/pause/resume/stop), writer thread,
    save cadence, publication store (metadata only, no GUI), frame eviction policy,
    and emits immutable frame events. Runs identically under GUI, notebook, CLI,
    or autonomous agent.
    """
    
    # Lifecycle
    def __init__(
        self,
        scan_name: str,
        plan: ReductionPlan,
        sink: ReductionSink | None = None,
        *,
        poni: PONI | None = None,
        integrator: AzimuthalIntegrator | None = None,
        geometry: DiffractometerGeometry | None = None,
        cancel_token: CancelToken | None = None,
        executor: Executor | None = None,
        inflight_max: int | None = None,
        pause_drain_timeout: float = 30.0,
        save_metadata_cadence: int | None = 16,  # frames between metadata writes
        pause_safe: bool = True,  # wait for writer idle before returning from pause()
    ): ...
    
    # Commands in (blocking)
    def start(self) -> None:
        """Begin accepting frame submissions; idempotent."""
    
    def submit(
        self,
        frame: ScanFrame,  # or FrameSource-driven (lazy-loaded)
        image: ndarray | None = None,
    ) -> None:
        """Submit one frame for integration; blocks if in-flight window full."""
    
    def pause(self) -> bool:
        """Quiesce the writer at a frame boundary; return True if writer idle.
        
        Caller may browse disk / adjust parameters while paused.
        Does NOT close the session; call resume() to continue.
        """
    
    def resume(self) -> None:
        """Re-engage the writer; a no-op if not paused."""
    
    def stop(self, timeout: float | None = 30.0) -> None:
        """Close the session, flush pending, release resources."""
    
    # State queries (read-only)
    @property
    def is_running(self) -> bool:
        """True iff session is active (started, not stopped)."""
    
    @property
    def is_paused(self) -> bool:
        """True iff paused; False if running or stopped."""
    
    @property
    def frames_submitted(self) -> int:
        """Frames accepted by submit() so far."""
    
    @property
    def frames_completed(self) -> int:
        """Frames written to sink so far."""
    
    # Events out (immutable)
    def on_frame_completed(self, callback: Callable[[FrameEvent], None]) -> None:
        """Register callback for each completed frame: 
        FrameEvent = (frame_index, result_1d, result_2d, metadata, timestamp, generation)
        """
    
    def on_progress(self, callback: Callable[[ProgressEvent], None]) -> None:
        """Register callback for progress events."""
    
    def on_state_change(self, callback: Callable[[StateChangeEvent], None]) -> None:
        """Register callback for state changes: (old_state, new_state, message)"""
```

**Key traits:**
1. **Unified orchestration:** Live serial, streaming batch, reintegrate, reload all use the same ScanSession + same ReductionSession under the hood. No "three paths" at the session level; dispatch is a caller concern.
2. **Pause is part of the contract:** `pause()` waits for writer idle (drain + flush complete) before returning. No surprises.
3. **No GUI coupling:** Commands are plain functions (no Qt signals), events are plain objects (no Qt enums).
4. **Immutable frame events:** The session emits completed frames as read-only records, not mutable LiveFrame objects.
5. **No internal PublicationStore:** The session doesn't validate; it only reduces + writes. GUI adds the envelope.
6. **Generation stamps on events:** Allows GUI/caller to drop stale frames if the parameters changed mid-run.

---

## CONCRETE WORK ITEMS (STAGED)

### Stage 1: Stabilize reductionSession + QtNexusSink (DONE ✅ in 1.0)
**Files:** `src/xrd_tools/reduction/core.py`, `src/xdart/gui/tabs/static_scan/wranglers/qt_nexus_sink.py`
**Work:** ✅ Already shipped. ReductionSession handles all three execution modes (chunked, streaming, future-ready for subscribe/drain patterns).

---

### Stage 2: Expose pause-safe guarantee in ReductionSession (SMALL, ~50 LOC)
**Files:** `src/xrd_tools/reduction/core.py` (add pause state + property)
**Work:**
- Add `_paused` boolean field (init=False)
- Add `pause() -> bool` method: calls `drain()`, sets `_paused = True`, returns success
- Add `resume()` method: sets `_paused = False` (no-op if already running)
- Add `is_paused` property
- Modify `submit()` to raise if paused
- Modify `process()` (chunked) to raise if paused
- Add integration test: pause mid-stream, verify writer is idle, resume, verify continuity

**Size:** M (50 LOC code + 30 LOC tests)  
**Prerequisite:** ReductionSession.drain() (already done)  
**Risk:** None (additive, no breaking changes)  
**Acceptance gate:** `tests/core/test_reduction_session_pause.py` — pause returns True, submit raises if paused, frames before + after pause match byte-for-byte with non-paused baseline

---

### Stage 3: Factor xdart's save cadence into a headless SinkBatcher (MEDIUM, ~100 LOC)
**Files:** New file `src/xrd_tools/reduction/cadence.py`, modify `src/xrd_tools/reduction/core.py`
**Work:**
- Create `FrameFlushCadence` enum: `EVERY_FRAME`, `EVERY_N`, `PER_SECOND`, `ON_EXPLICIT` (callback)
- Create `CadencedSink(ReductionSink)` wrapper: buffers write() calls, flushes on cadence
  - ✅ `write()` buffers the frame
  - ✅ `begin()` initializes counters + timers
  - ✅ `finish()` flushes remaining buffered frames
  - ✅ Internal flush check called per write
- Update `NexusSink` to support `flush_every` parameter (already does, line 436)
- Integrate into `ReductionSession.__init__`: if sink + cadence + streaming mode, wrap sink in CadencedSink
- DONT move xdart's specific `LIVE_SAVE_INTERVAL` logic; leave that in image_wrangler_thread

**Size:** M (100 LOC + 50 LOC tests)  
**Prerequisite:** ReductionSession pause (Stage 2)  
**Risk:** MEDIUM. The wrapped sink must faithfully proxy `worker_process()`, `replace()`, `abort()`. Test on QtNexusSink to ensure no ordering surprises.  
**Acceptance gate:** `tests/core/test_cadenced_sink.py` — verify frame counts, verify flush order matches unbuffered baseline, verify replace/abort proxy correctly

---

### Stage 4: Move xdart's save-cadence logic into ScanSession adapter (MEDIUM, ~150 LOC)
**Files:** New file `src/xdart/modules/scan_session.py`, modify `src/xdart/gui/tabs/static_scan/wranglers/image_wrangler_thread.py`
**Work:**
- Create `ScanningScanSession(ReductionSession)`: wraps a ReductionSession with xdart-specific live cadence
  - ✅ `__init__`: accepts `live_save_interval` (frames), `in_memory_cap` (for eviction query)
  - ✅ `submit()`: wraps ReductionSession.submit(), increments `_frames_since_save`
  - ✅ `_check_save_due()`: True if `_frames_since_save >= interval` OR `live_scan.frames.unsaved_in_memory_count() >= cap - 8`
  - ✅ Exposes `pause()` + `resume()` and pauses the underlying h5pool around the drain/flush
  - ✅ No display coupling (no sigUpdate, no publication_store)
- Keep xdart's QtNexusSink unchanged (it's the sink, not the session)
- Update `_dispatch_batch_streaming()` to use ScanningScanSession instead of directly constructing ReductionSession

**Size:** M (150 LOC + test harness)  
**Prerequisite:** Stages 2 + 3  
**Risk:** MEDIUM. Must not duplicate pause/resume logic; ScanningScanSession should delegate to underlying session.pause()/resume(). Test on real image_wrangler_thread paths.  
**Acceptance gate:** `tests/xdart/test_scanning_session.py` + integration with image_wrangler_thread; verify `live≡batch≡reload` equivalence holds

---

### Stage 5: Lift run-state queries into ScanningScanSession (SMALL, ~30 LOC)
**Files:** `src/xdart/modules/scan_session.py` (add properties), `src/xdart/gui/tabs/static_scan/static_scan_widget.py` (consume from session instead of own booleans)
**Work:**
- Add `is_running`, `is_paused` properties to ScanningScanSession (delegate to underlying session)
- Add `frames_submitted`, `frames_completed` properties
- Modify `static_scan_widget._enter_run_state()` to read `session.is_running` instead of checking own `self._run_active`
- Modify `_on_run_paused()` / `_on_run_resuming()` to trust `session.is_paused`
- Retain `self._run_active` as a display-side cache (for ctrl disable) but sync from session state

**Size:** S (30 LOC + refactor staticWidget to accept session object)  
**Prerequisite:** Stage 4  
**Risk:** LOW. Changes are purely additive and provide redundancy (display state can be verified against session state).  
**Acceptance gate:** Same as Stage 4 (equivalence spine)

---

### Stage 6: Retire xdart's dual save-cadence paths (MEDIUM, ~200 LOC refactor)
**Files:** `src/xdart/gui/tabs/static_scan/wranglers/image_wrangler_thread.py` (remove `_dispatch_batch_serial`, `_dispatch_batch_parallel`, consolidate into streaming)
**Work:**
- Keep `_process_one()` for Phase 3 watch loop (single-frame serial is correct for detector-rate)
- Remove `_dispatch_batch_serial()` and `_dispatch_batch_parallel()` as separate paths; both now route through ScanningScanSession.streaming
- Update `_dispatch_batch()` router (line 901) to always use streaming (remove environment variable fallback for chunked/parallel)
- Simplify `_wait_if_paused()` + `_enter_pause()` to just interact with `session.pause()` / `session.resume()`

**Size:** M (remove ~300 LOC dead code, add ~50 LOC for streaming-only path)  
**Prerequisite:** Stages 4 + 5  
**Risk:** MEDIUM-HIGH. Three paths collapsed into one; must verify on all live/batch/reprocess/watch scenarios. The serial watch path MUST stay — don't break it.  
**Acceptance gate:** Full xdart test suite + manual test on live detector (watch), batch (reintegrate), and reprocess

---

### Stage 7: Extract common ScanSession API to core (LARGE, ~400 LOC, design phase)
**Files:** New file `src/xrd_tools/session.py`, refactor ScanningScanSession into core, xdart adapter layer
**Work:**
- Define `ScanSession` abstract base in xrd_tools (commands: start/submit/pause/resume/stop, properties: is_running/is_paused/frames_*/events: on_frame_completed/on_progress/on_state_change)
- Move ScanningScanSession from xdart to xrd_tools as the default implementation
- Xdart provides a Qt adapter layer (ScanSession → Qt signals for backwards compat with existing display)
- Notebook + CLI use ScanSession directly with plain callbacks
- Enables Bluesky/autonomous agents to drop in their own FrameSource and use the unified session

**Size:** L (400 LOC + design review)  
**Prerequisite:** Stages 1–6 stable + design review (Difference 2 checkpoint)  
**Risk:** HIGH. This is the main architectural boundary. Requires careful API design for extensibility (sinks, callbacks, sources).  
**Acceptance gate:** (a) New `tests/core/test_session_api.py` — basic lifecycle (start/submit/pause/resume/stop), (b) xdart adapter test, (c) notebook example

---

## KEY RISKS & MITIGATIONS

1. **Live≡batch≡reload equivalence spine** (the acceptance gate)
   - Risk: Consolidating three wrangler paths could produce byte-different .nxs files
   - Mitigation: Run `tests/xdart/test_gi_batch_real_data.py::test_*_equivalence` on every stage; this test is the gating criterion
   - Acceptance: Diff against 1.0 baseline; any deviation is a real bug, not a tolerance

2. **Persist-before-evict invariant** (Stage 3–4 refactor)
   - Risk: If cadence check happens BEFORE QtNexusSink registers the frame, an eviction could happen mid-write
   - Mitigation: QtNexusSink must call `scan.frames.mark_persisted()` AFTER the stacked write completes, not before. Verify in code review.
   - Acceptance: No frame loss on long scans (10k+ frame Eiger batch)

3. **Pause drain timeout hangup** (Stage 2)
   - Risk: If a pool worker hangs, `session.drain(timeout=30s)` times out; pause proceeds but frames are unflushed
   - Mitigation: Timeout is configurable in ScanningScanSession. Add logging on timeout. Document that pause does NOT guarantee full flush if timeout fires.
   - Acceptance: Pause completes within 31s regardless; no deadlock

4. **Frozen on-disk format** (all stages)
   - Risk: A refactor could accidentally change the v2 NeXus schema written
   - Mitigation: Run `tests/core/test_v2_record_compat.py` on every PR; this test pins the byte-signature of written .nxs files
   - Acceptance: Byte-signature unchanged (or explicit migration with version bump + reader support)

5. **Generation stamping under pause** (Stage 5)
   - Risk: If parameters change while paused, should a resumed frame's generation reflect the new parameters?
   - Mitigation: Define "generation" as bumped on parameter change (not pause/resume). Document this. Test: change integrator params mid-pause, verify resumed frames are marked stale-generation
   - Acceptance: Generation stamp test in test_session_api.py

---

## SUMMARY TABLE

| Stage | What | Files | Size | Risk | Gate | Notes |
|-------|------|-------|------|------|------|-------|
| 1 ✅ | ReductionSession + QtNexusSink | core.py, qt_nexus_sink.py | — | — | Shipped in 1.0 | Orchestration done |
| 2 | Pause-safe API | core.py (add pause/resume/is_paused) | S | Low | test_reduction_session_pause.py | Expose state machine |
| 3 | SinkBatcher cadence | cadence.py (new) | M | Med | test_cadenced_sink.py | Decouple timing from session |
| 4 | ScanningScanSession | scan_session.py (new) | M | Med | equiv. spine + test_scanning_session.py | Lift xdart cadence |
| 5 | Run-state queries | scan_session.py (extend), static_scan_widget.py | S | Low | equiv. spine | Unify state source |
| 6 | Retire dual paths | image_wrangler_thread.py (refactor) | M | High | Full xdart suite + manual live | Consolidate to streaming |
| 7 | Extract ScanSession API | session.py (new), refactor | L | High | test_session_api.py + xdart adapter | Design boundary |

---

## FINAL STATE (POST-STAGES 1–7)

- ✅ **Live≡batch≡reload:** All three paths use the same orchestration (ReductionSession + sink)
- ✅ **Thin xdart:** GUI owns display + user commands; headless core owns acquisition + reduction
- ✅ **Pause-safe:** ScanSession.pause() guarantees writer idle before returning
- ✅ **Portable:** A notebook, CLI, or Bluesky agent can instantiate ScanSession directly (no Qt dependency)
- ✅ **Testable:** Public API (start/submit/pause/resume/stop) + events (on_frame_completed) allow black-box testing
- ✅ **Frozen format:** On-disk v2 record unchanged; portable @source_base; no breaking reader changesExcellent. Now let me generate the final inventory:

## GAP INVENTORY: Difference 3 — One FrameRecord, One Store

### (1) X1 STATUS: Publications as sole display contract

**WHO STILL CONSULTS data_1d/data_2d/scan_data in parallel:**

#### display_controllers.py
- **Line 66**: `loaded_1d = set(widget.data_1d.keys())` — direct dict keys snapshot
- **Line 67**: `loaded_2d = set(widget.data_2d.keys())` — direct dict keys snapshot
- **Lines 73-75**: `widget.data_2d.items()` iteration for raw availability snapshot
- **Lines 110-116**: `widget.data_2d.get(idx)` read for raw/thumbnail in Image Viewer payload
- **Line 253**: `frames = {i: widget.data_1d.get(i) for i in render_ids}` — frame dict builder for plot logic
- **Lines 335-336**: `for key, frame in widget.data_1d.items()` iteration on NexusViewerController
- **Line 374**: `widget.data_1d.get(idx)` read for metadata in XYE viewer

#### display_data.py
- **Lines 119-141**: `_snapshot_data()` reads `self.data_1d.get(idx)` and `self.data_2d.get(idx, {})` under lock
- **Lines 210-219**: `frame_2d.get('map_raw')`, `frame_2d.get('thumbnail')` from data_2d dict
- **Line 309**: `frame_1d.scan_info` read in `get_frames_map_raw()`
- **Lines 385-443**: `self.data_2d.get(idx)` reads in `get_frames_int_2d()` for 2D intensity + axis data
- **Lines 496-503**: `self.data_1d.get(idx)` reads in `get_frames_int_1d()` for 1D frame stacking
- **Line 565**: `frame_2d['int_2d']` access in `get_int_1d()`
- **Line 573**: `self.scan.scan_data[self.normChannel]` read in `get_int_2d()` for normalization

#### metadata.py
- **Line 82**: `if self.data_1d is not None and sel_int in self.data_1d` — cache lookup
- **Line 83**: `return self.data_1d[sel_int]` — frame retrieval
- **Line 160**: `selected.scan_info` read from frame fallback

**ALREADY READING from PublicationStore:**
- **display_controllers.py, lines 76-81**: `publication_availability(store)` called; publications loaded into `loaded_1d/2d` alongside cache keys
- **metadata.py, lines 154-157**: Falls back to `publication.metadata_raw` when frame not in data_1d

**EVIDENCE OF PARALLEL READS:** The code explicitly reconciles three sources:
```python
# display_controllers.py line 65-82
with widget.data_lock:
    loaded_1d = set(widget.data_1d.keys())
    loaded_2d = set(widget.data_2d.keys())
    raw_avail = {
        int(k): {
            'has_raw': v.get('map_raw') is not None,
            'has_thumbnail': v.get('thumbnail') is not None,
        }
        for k, v in widget.data_2d.items()
        if isinstance(v, dict)
    }
    store = getattr(widget, "publication_store", None)
    if store is not None:
        pub_1d, pub_2d, pub_raw = publication_availability(store)
        loaded_1d.update(pub_1d)  # ← MERGING two sources
        loaded_2d.update(pub_2d)
        raw_avail.update(pub_raw)
```

---

### (2) X2 STATUS: Parallel LiveFrame.integrate_* methods

**CALL SITES INVOKING integrate_1d/integrate_2d ON LiveFrame:**

#### ewald/scan.py (LiveScan container)
- **Lines 391-392**: `frame.integrate_1d(global_mask=self.global_mask, **self.bai_1d_args)` and `frame.integrate_2d(...)` in `add_frame()` method — THE ONLY canonical integration path for live acquisition
- **Lines 159-160**: `self.bai_1d/bai_2d` fields store the scan-level integration parameters

**LiveFrame methods defined (ewald/frame.py):**
- **Lines 551-689**: `integrate_1d()` method (139 lines) — computes `self.int_1d` and GI `self.gi_1d[mode]` dict
- **Lines 691-827**: `integrate_2d()` method (137 lines) — computes `self.int_2d` and GI `self.gi_2d[mode]` dict
- **Lines 862-896**: `make_thumbnail()` method — computes `self.thumbnail`

**NO parallel integrate_* calls found:** The only path is `scan.add_frame(calculate=True)` → `frame.integrate_1d()` + `frame.integrate_2d()` sequentially. The GI paths store results in `self.gi_1d/gi_2d` dicts rather than separate method variants.

**STATUS: X2 COMPLETE** — the system calls a single `integrate_1d()` and `integrate_2d()` per frame, storing results on the frame object itself. No retired parallel methods remain.

---

### (3) FRAME CLASSES: fields, conversions, merge targets

#### ScanFrame (xrd_tools/core/scan.py, lines 139-204)
**Fields:**
- `index: int` — frame ordinal
- `image: np.ndarray | None` — raw detector array (lazy-loadable)
- `metadata: dict[str, Any]` — provenance dict (mutable, expanded post-init)
- `source_path: Path | str | None` — source file path
- `source_frame_index: int | None` — offset within source file
- `background: np.ndarray | float | None` — background image
- `mask: np.ndarray | MaskSpec | None` — pixel mask
- `normalization_factor: float | None` — divisor for intensity
- `loader: ImageLoader | None` — lazy image loader callable
- `geometry: FrameGeometry | None` — per-frame rotation angles + poni
- `source_identity: str | None` — provenance tag

**Purpose:** Canonical headless reduction input. Immutable after construction (frozen=False but treated read-only). **No integration results.**

#### LiveFrame (xdart/modules/ewald/frame.py, lines 98-962)
**Fields:**
- `idx: int | None` — frame label
- `map_raw: np.ndarray | None` — raw detector array (mutable, can be freed)
- `bg_raw: np.ndarray | float | None` — background image (freed with raw)
- `poni: PONI` — calibration object (mutable via `set_poni()`)
- `mask: np.ndarray | None` — pixel mask (mutable)
- `scan_info: dict[str, Any]` — per-frame metadata (mutable)
- `ai_args: dict` — integrator kwargs (mutable)
- `file_lock: Condition` — synchronization (mutable)
- `static: bool`, `gi: bool`, `incidence_motor: str`, `tilt_angle: float`, `sample_orientation: int`, `series_average: bool` — scan flags
- `integrator: AzimuthalIntegrator` — pyFAI integrator (mutable via `set_integrator()`)
- **`int_1d: IntegrationResult1D | None`** — 1D integration result (mutable)
- **`int_2d: IntegrationResult2D | None`** — 2D integration result (mutable)
- **`gi_1d: dict[str, IntegrationResult1D]`** — GI 1D modes (mutable)
- **`gi_2d: dict[str, IntegrationResult2D]`** — GI 2D modes (mutable)
- **`source_file: str`** — relpath to source (source ref for reload, R2 schema)
- **`source_frame_idx: int | None`** — index within source file
- **`thumbnail: np.ndarray | None`** — downsampled preview (baked mask)
- `is_reload_only: bool` — True if raw is unrecoverable (R3 guardrail)
- `_source_root: str` — directory for relative path resolution
- `frame_lock: Condition`, `map_norm: float` — state guards

**Purpose:** Stateful live-acquisition frame. Mutable, carries locks, lazy loaders, and integration results. **Central to LiveScan + writer.**

#### FrameView (xrd_tools/core/frame_view.py, lines 80-352)
**Fields:**
- `label: int | str` — frame identifier
- **`axis_1d: Axis | None`**, **`intensity_1d: np.ndarray | None`**, **`sigma_1d: np.ndarray | None`** — 1D integration
- **`axis_2d_x: Axis | None`**, **`axis_2d_y: Axis | None`**, **`intensity_2d: np.ndarray | None`**, **`sigma_2d: np.ndarray | None`** — 2D integration
- **`two_d_kind: TwoDKind`** — 2D axis identity (Q_CHI, QIP_QOOP, etc.)
- **`raw: np.ndarray | None`** — full-resolution detector image
- **`thumbnail: np.ndarray | None`** — preview image
- `mask_baked: bool` — mask is encoded in thumbnail/raw
- `metadata_raw: Mapping[str, Any]` — provenance (readonly)
- `metadata_numeric: Mapping[str, float]` — numeric subset (readonly)
- `incident_angle: float | None` — GI incident angle
- `geometry: FrameGeometry | None` — per-frame rotation + PONI
- `source_path: str | None`, `source_frame_index: int | None` — source ref (R2 schema)
- `extra: Mapping[str, Any]` — extensible metadata (readonly)

**Purpose:** Immutable display/round-trip record. Frozen=True. Carries full integration results + source refs. Round-trippable via NeXus + readers.

#### FramePublication (xdart/modules/frame_publication.py, lines 87-109)
**Fields:**
- **`view: FrameView`** — the immutable record
- `source_identity: str` — display tag (file path or index)
- `generation: int` — display generation stamp
- `raw_ref: Any | None` — ref to the mutable LiveFrame (for re-access)
- `raw_status: str` — "ready"/"missing"/"thumbnail"/"evicted"
- `metadata_raw: Mapping[str, Any]` — copy/override of view.metadata_raw
- `metadata_numeric: Mapping[str, float]` — copy/override of view.metadata_numeric
- **`diagnostics: PublicationDiagnostics`** — health checks (finite%, dummy%, axis ranges, warnings/errors)

**Purpose:** GUI snapshot envelope. Immutable (frozen=True). Wraps a FrameView + adds validation + display stamping. **Sole display contract.**

**CONVERSION FUNCTIONS (frame_publication.py):**
- **`publication_from_live_frame()` (lines 181-229):** LiveFrame → FrameView → FramePublication
  - Extracts `scan_info`, `int_1d`, `int_2d`, `thumbnail`, metadata
  - Validates if requested
  - **Holds raw_ref to the mutable frame** for re-access
- **`publication_from_frame_view()` (lines 232-257):** FrameView → FramePublication
  - Direct wrap + validation
- **`publication_from_nexus_frame()` (lines 260-285):** Lazy read via `read_frame_view()` → FramePublication

**MERGE TARGETS FOR ONE FrameRecord:**
The greenfield design would merge `ScanFrame` + `FrameView` field sets into a single `FrameRecord`:
- Keep `ScanFrame.{index, image, metadata, source_path, source_frame_index, background, mask, normalization_factor, loader, geometry, source_identity}`
- Add `FrameView.{axis_1d, intensity_1d, sigma_1d, axis_2d_*, intensity_2d, sigma_2d, two_d_kind, raw, thumbnail, mask_baked, incident_angle, extra}` (all integration results)
- **Skip:** `LiveFrame` (mutable state, locks, integrator, gi_1d/gi_2d dicts — all become ephemeral integration scratchpads)
- **FramePublication** reduces to a `generation: int` + `diagnostics: PublicationDiagnostics` pair layered on FrameRecord when displayed

---

### (4) THREE CACHES: write sites, eviction, sync points

#### data_1d (FixSizeOrderedDict, max=0)
**Write sites:**
- **scan_threads.py, lines ~370-375:** `self.data_1d[idx] = frame.copy_for_display(include_2d=False)` per integrated 1D frame (wrangler thread)
- **static_scan_widget.py, line ~:** H5Viewer loads; similar assignment
- **h5viewer.py, lines ~:** File reload path; `self.data_1d[idx] = frame`

**Eviction regime:**
- `FixSizeOrderedDict(max=0)` — max=0 means **UNBOUNDED** (no eviction). All 1D frames retained in memory for the entire scan lifetime.
- Alternative interpretation: max=0 disables caching entirely (check _utils.py). **Need verification.**

**Hydration from disk:**
- display_data.py, lines 499-506: `_hydrate_frame_from_disk()` reads from `scan.frames[idx]` (LiveFrameSeries LRU, 64-deep), which lazy-loads from the written .nxs `/entry/integrated_1d` on demand.

**Reads (data_1d is a source of truth for "1D loaded"):**
- display_controllers.py line 66: dict keys for availability snapshot
- display_controllers.py line 253: frame dict builder for plot logic
- display_data.py line 496: frame lookup for 1D intensity extraction
- metadata.py line 82-83: scan_info read for metadata panel

**Keep-in-sync sites:**
- **ONLY write site is wrangler/load thread.** PublicationStore is filled separately by the writer (nexus_writer.py).

#### data_2d (FixSizeOrderedDict, max=40)
**Write sites:**
- **scan_threads.py, lines ~365-375:** `self.data_2d[int(idx)] = {'map_raw': frame.map_raw, 'bg_raw': frame.bg_raw, 'int_2d': frame.int_2d, 'gi_2d': frame.gi_2d}` per integrated frame
- **image_wrangler_thread.py:** Similar dict with raw + integration results
- **h5viewer.py:** File load path

**Eviction regime:**
- `FixSizeOrderedDict(max=40)` — FIFO eviction when size exceeds 40 entries
- **Hydrated-raw LRU (D5):** `hydrated_raw.remember_hydrated_raw(data_2d, idx, limit=8)` caps full-resolution `map_raw`/`bg_raw` at 8 frames while keeping all 40 `int_2d` + `gi_2d` + `thumbnail` entries intact.
- Eviction path: `hydrated_raw.py, lines 54-58` nulls `map_raw`/`bg_raw` when order exceeds limit.

**Hydration from disk:**
- display_data.py, lines 392-399: `_hydrate_frame_from_disk()` → `self.scan.frames[idx]` → .nxs `/entry/integrated_2d` + `/entry/frames/frame_NNNN/thumbnail`

**Reads (data_2d is a source of truth for "2D loaded" + "raw available"):**
- display_controllers.py line 67: dict keys for availability
- display_controllers.py lines 73-75: raw_avail snapshot (has_raw, has_thumbnail)
- display_controllers.py line 110: frame_2d dict for Image Viewer payload
- display_data.py line 210-219: map_raw/thumbnail reads in get_frames_map_raw()
- display_data.py line 387: int_2d reads in get_frames_int_2d()

**Keep-in-sync sites:**
- **NONE DOCUMENTED.** The wrangler writes data_2d; the writer (nexus_writer.py) reads `frame.int_2d` + `frame.gi_2d` directly from the live LiveFrame, **not from data_2d**. Two independent paths.

#### PublicationStore (xdart/modules/frame_publication.py, lines 329-399)
**Fields:**
- `_items: dict[int|str, FramePublication]` — frame label → publication mapping
- `_heavy_labels: list[int|str]` — LRU order of frames with payloads
- `_max_items: int | None` — total frame count cap (None = unbounded)
- `_max_heavy_items: int | None = 64` — display-heavy payload cap
- `_generation: int` — display generation counter
- `_lock: RLock` — thread-safe access

**Write sites:**
- **scan_threads.py, lines ~:** `publication_store.upsert(publication)` per frame published by wrangler
- **static_scan_widget.py:** File-load publish
- **h5viewer.py:** File load publish
- **nexus_writer.py:** Publishes frames as they're written

**Eviction regime:**
- `_enforce_bounds_locked()` (lines 396+, not shown but defined) evicts when `len(_heavy_labels) > _max_heavy_items` (64 by default)
- `_lightweight_publication()` (lines 304-326) drops payloads (intensity, sigma, raw, thumbnail) → keeps metadata + diagnostics only
- **No rehydration path shown in the code** — once evicted, the frame is lightweight metadata only (D2 deferred: rehydrate on demand)

**Reads (PublicationStore is being piloted as sole display contract):**
- display_controllers.py lines 76-82: `publication_availability(store)` to merge into loaded_1d/2d/raw_avail
- metadata.py lines 154-157: `publication.metadata_raw` fallback when data_1d miss
- display_publication.py (not yet read): PublicationDisplayAdapter

**Keep-in-sync sites:**
- **display_controllers.py line 78-81:** Explicitly reconciles store + data_1d/data_2d by merging availability sets. **This is the deferred "ensure both stay in sync" comment — it's reactive (re-merge on every display update) not proactive (keep one source of truth).**

---

### (5) D2-DEFERRED THUMBNAIL LRU: requirements from one-store end state

**Current D2 design intent (from CLAUDE.md + MIGRATION.md lines 119, 140-141):**

"D2 thumbnail LRU + lazy reload (analyzed Jun 2026; lands with the publication-store migration)"

**What one-store needs from D2:**

A unified `PublicationStore` with lazy-reload capability:
1. **All frames kept as lightweight metadata** (label, source_path, incident_angle, metadata_raw, metadata_numeric, diagnostics) — bounded store size
2. **Heavy payloads (intensity_1d, intensity_2d, raw, thumbnail) dropped beyond max_heavy** — currently evicted as `_lightweight_publication()` with no rehydration
3. **Thumbnail LRU:** Maintain *distinct* from the full integration-result LRU
   - Thumbnails stay through integer shifts (smaller payload, safer to keep)
   - Full 1D/2D drop when over the heavy cap
   - **But**: the current FixSizeOrderedDict(max=40) for data_2d conflates all three
4. **Lazy rehydration on demand:**
   - `publication_store.get(idx)` → if evicted, call a registered loader (e.g., `publication_from_nexus_frame()`)
   - Similar to how `_hydrate_frame_from_disk()` in display_data.py works today, but owns the rehydration contract

**Risks for one-store transition:**
- **Behavior equivalence spine:** If a display-mode change causes data_1d/data_2d eviction mid-render, the old code could serve a partial cache (only 1D, only 2D). With PublicationStore owning all three, a complete eviction → lightweight → rehydrate path must preserve the same visual result. This is the **"generation stamping" + "stale payload drop" gate** (display_logic.py render_plan).

---

### (6) STAGED COLLAPSE PLAN: behavior-preserving steps keeping equivalence spine green

**Target: Replace the three-cache triple-read pattern (data_1d + data_2d + PublicationStore) with a single PublicationStore, collapsing LiveFrame → FrameRecord.**

**Stage 1: Move to publications as sole display contract (in progress)**
- **Goal:** Make PublicationStore the authoritative display source; demote data_1d/data_2d to "hydration mirrors" (D5-style).
- **Changes:**
  - Invert the merge in display_controllers.py:65-82: query PublicationStore first; fall back to data_1d/data_2d only if not in store.
  - Remove the `store.update(loaded_1d)` reconciliation; keep them separate.
  - Add a "hydrate from cache" helper in PublicationStore.upsert: check if the cache already has this frame; if so, reuse the result.
- **Acceptance gate:** `test_*_equivalence` all pass; the store's generation stamps match the display's generation.
- **Risk:** If the wrangler publishes before the cache populates, the store becomes the source of truth — any later cache write is ignored (the store's version wins). This is fine if we explicitly choose it.
- **Files:** display_controllers.py (lines 62-82), publication_store.py (upsert logic).
- **Size:** M (lines 100-200)

**Stage 2: Retire data_1d; migrate reads to publications**
- **Goal:** Remove `FixSizeOrderedDict(max=0)` entirely; all frame reads go through PublicationStore or direct `scan.frames` (the 64-deep LiveFrameSeries LRU).
- **Changes:**
  - Delete data_1d creation in static_scan_widget.py.
  - Redirect metadata.py:82-83 from data_1d lookup to `publication_store.get()`.
  - Redirect display_data.py:496 from data_1d.get() to store.get() (may trigger rehydration).
  - Redirect display_controllers.py:253 from data_1d dict to store.snapshot().
  - Rehydrate-on-miss path: if store doesn't have the frame, lazy-load from scan.frames (as today).
- **Acceptance gate:** All reads still work; `test_*_equivalence` passes. No behavior change to the display.
- **Risk:** A frame evicted from the store that's no longer in the 64-frame cache is lost; must rehydrate from .nxs. The old code would have kept it in data_1d forever (max=0 → unbounded). **Mitigation:** Set max_items=512 (or data_1d's old limit) temporarily during transition; decrement later as confidence grows.
- **Files:** static_scan_widget.py, metadata.py, display_data.py (lines 496-506), display_controllers.py (lines 253, 335-336), scan_threads.py (remove data_1d assignments).
- **Size:** M (lines 150-300)

**Stage 3: Collapse data_2d and data_1d hydration into PublicationStore**
- **Goal:** The store owns both the full payload and the lightweight-eviction LRU.
- **Changes:**
  - Fold the FixSizeOrderedDict(max=40) discipline into PublicationStore._enforce_bounds_locked().
  - Fold the hydrated_raw_lru (max=8) into a separate tracking list on the store.
  - When PublicationStore.upsert() is called, check if we're over bounds; if so, evict the oldest N frames to _lightweight_publication(), but **keep the rest as-is for later full hydration**.
  - Add PublicationStore.get_or_hydrate(): if the frame is lightweight, reload it from the source (scan.frames or .nxs).
- **Acceptance gate:** `test_*_equivalence` still passes. Raw-image panel loads from evicted frames without stalling the GUI (lazy load via get_or_hydrate).
- **Risk:** The old data_2d max=40 with raw cap=8 meant "keep 40 frames' worth of integration but only 8 with full-resolution raw." The new store must replicate this exactly: keep all 40 as lightweight, hydrate raw on demand (keeping at most 8 hydrated). The **keep in sync** site is now the _enforce_bounds_locked() call.
- **Files:** frame_publication.py (PublicationStore._enforce_bounds_locked, add hydrate path), display_data.py (remove _hydrate_frame_from_disk references to data_2d; redirect to store.get_or_hydrate()).
- **Size:** L (lines 250-400)

**Stage 4: Migrate display_data.py away from data_1d/data_2d snapshots**
- **Goal:** The display layer reads through PublicationStore exclusively.
- **Changes:**
  - Replace `_snapshot_data()` (lines 119-141) with a store snapshot that returns `(frame_1d: FramePublication, frame_2d_dict: {…})` pairs.
  - In `get_frames_map_raw()`, read `frame_2d['map_raw']` from the store result instead of the dict.
  - In `get_frames_int_1d()` and `get_frames_int_2d()`, read `int_1d/int_2d` from publication.view instead of frame.int_1d/int_2d.
  - The `frame_1d.scan_info` reads become `publication.metadata_raw`.
  - Lazy-load calls redirect to store.get_or_hydrate().
- **Acceptance gate:** `test_*_equivalence` passes. Waterfall/Average/Sum displays update identically.
- **Risk:** The display layer currently distinguishes between "frame not yet published" (cache miss) and "frame evicted from cache" (old cache value). With one store, there's just "not in store" (lightweight reload needed). The behavior must match: if a frame is over-scroll (beyond cache window), asking for its data must trigger a reload, not return stale.
- **Files:** display_data.py (lines 119-141, 210-220, 385-443, 496-530, lines using scan_info reads).
- **Size:** L (lines 300-500)

**Stage 5: Collapse LiveFrame.copy_for_display() → FramePublication**
- **Goal:** The wrangler publishes FramePublication directly; no data_1d/data_2d intermediate.
- **Changes:**
  - Remove `LiveFrame.copy_for_display()` (ewald/frame.py, lines 940-961).
  - In scan_threads.py, replace `self.data_1d[idx] = frame.copy_for_display()` with `publication = publication_from_live_frame(frame, generation=...)` and `self.publication_store.upsert(publication)`.
  - In image_wrangler_thread.py and h5viewer.py, follow the same pattern.
  - The `data_2d` dict write at scan_threads.py line ~365 → also goes to the store.
- **Acceptance gate:** `test_*_equivalence` passes. All three paths (live, batch, reload) publish to the same store.
- **Risk:** The writer (nexus_writer.py) currently reads `frame.int_1d/int_2d` directly. If we delete the frame copy before the writer finishes, we lose the data. **Mitigation:** The writer publishes before deleting (it holds a ref to the live frame for the duration of the write). The publication creation happens in parallel, so the writer reads the frame → creates a FramePublication → writer publishes. Both read the same live frame; both create independent immutable copies (FrameView + FramePublication).
- **Files:** ewald/frame.py (delete copy_for_display), scan_threads.py (replace data_1d/data_2d writes), image_wrangler_thread.py, h5viewer.py, publication_from_live_frame path.
- **Size:** M (lines 100-250)

**Stage 6: Final LiveFrame → FrameRecord migration (greenfield, post-1.0)**
- **Goal:** Replace LiveFrame with a slimmer variant or a FrameRecord + temporary scratchpad.
- **Changes:**
  - Create `FrameRecord` = ScanFrame fields + FrameView fields (no locks, no integrator state).
  - LiveFrame → thin wrapper holding `FrameRecord` + mutable `integrator`, `int_1d`, `int_2d`, `gi_1d/gi_2d` (integration scratchpads only, not stored).
  - After `frame.integrate_1d/2d()`, freeze the results into a FrameRecord.
  - The wrangler publishes FrameRecord → FrameView → FramePublication.
  - On reload, read FrameRecord from .nxs directly (no LiveFrame needed).
- **Acceptance gate:** Headless reduction session produces FrameRecords; GUI still consumes them via the publication layer. One round-trip contract.
- **Risk:** This is a large refactor (Stage 2+3+4+5 were the staging; this is the final cleanup). Do it last, once the equivalence spine is proven green.
- **Files:** (deferred to post-1.0)
- **Size:** XL

---

### WORK ITEMS (staged commit order)

| # | Title | Files | Size | Prerequisites | Risks | Acceptance gate |
|---|-------|-------|------|--------------|-------|-----------------|
| 1 | **X1.1: Invert display_controllers to query store first** | display_controllers.py (lines 62-82) | S | None | Reconciliation complexity; stale cache vs. store divergence | `test_display_controllers` suite passes |
| 2 | **X1.2: Add publication_availability() helper** | display_publication.py | S | 1 | PublicationStore availability API undefined | `test_frame_publication` passes |
| 3 | **X1.3: Pilot test with display logic snapshot merging** | tests/xdart/test_display_*_logic.py | M | 1, 2 | Generation stamping consistency | All display snapshot tests pass |
| 4 | **Checkpoint: X1 complete** | — | — | 1-3 | — | `test_*_equivalence` still green |
| 5 | **X2.1: Verify no parallel integrate_* calls exist** | grep output review | S | None | May find old code paths | `test_ewald_scan` frame integration tests pass |
| 6 | **Checkpoint: X2 complete** | — | — | 5 | — | No test changes needed |
| 7 | **D2.1: Add get_or_hydrate() to PublicationStore** | frame_publication.py (PublicationStore) | M | 4 | Circular import (store calls readers); rehydrate contract undefined | `test_publication_store_hydrate` passes |
| 8 | **D2.2: Extend PublicationStore._enforce_bounds_locked() for thumbnail LRU** | frame_publication.py (PublicationStore._enforce_bounds_locked) | M | 7 | Two-level eviction (full/lightweight) complexity | `test_publication_store_bounds` passes; hydrated_raw_lru tests pass |
| 9 | **Stage 2.1: Retire data_1d from metadata.py** | metadata.py (line 82-83) | S | 7, 8 | Fallback to scan.frames missing | `test_metadata_widget` passes |
| 10 | **Stage 2.2: Retire data_1d from display_controllers.py** | display_controllers.py (line 253, 335-336) | S | 7, 8 | Frame dict builder contract changes | `test_display_controllers` passes |
| 11 | **Stage 2.3: Retire data_1d from display_data.py** | display_data.py (line 496-530) | M | 7, 8 | Snapshot-taking contract changes; lazy load fallback | `test_display_data` passes; `test_*_equivalence` green |
| 12 | **Stage 2.4: Remove data_1d from widget initialization** | static_scan_widget.py (remove FixSizeOrderedDict(max=0)) | S | 9-11 | None | `test_static_scan_widget` passes |
| 13 | **Checkpoint: Stage 2 complete** | — | — | 9-12 | — | `test_*_equivalence` still green; memory profiling matches |
| 14 | **Stage 3.1: Fold data_2d into PublicationStore eviction** | frame_publication.py (PublicationStore._enforce_bounds_locked), display_data.py (remove FixSizeOrderedDict(max=40)) | L | 13 | Eviction order coherence; re-entrant locking | `test_publication_store_bounds` passes; `test_display_data_2d_eviction` green |
| 15 | **Stage 3.2: Migrate display_data.py to store snapshots** | display_data.py (lines 119-141, 210-220, 385-443) | L | 14 | Snapshot contract (frame_1d, frame_2d dict) changes semantics | `test_get_frames_*` tests pass; `test_*_equivalence` green |
| 16 | **Stage 3.3: Remove hydrated_raw LRU from hydrated_raw.py** | hydrated_raw.py | S | 14, 15 | Invert: hydration moves into PublicationStore | `test_hydrated_raw` passes (should be redundant) |
| 17 | **Stage 3.4: Remove data_2d from widget + wranglers** | static_scan_widget.py, scan_threads.py, image_wrangler_thread.py, h5viewer.py | M | 14-16 | Write-site inventory; publication creation pacing | `test_wrangler_*` tests pass |
| 18 | **Checkpoint: Stage 3 complete** | — | — | 14-17 | — | `test_*_equivalence` still green; same caching behavior |
| 19 | **Stage 4.1: Pilot live/batch/reload to publish FramePublication** | scan_threads.py (frame.integrate_1d/2d → publication_from_live_frame → store.upsert) | M | 18 | Publication creation timing; writer read-after-evict | `test_scan_thread_wrangler` passes |
| 20 | **Stage 4.2: Delete LiveFrame.copy_for_display()** | ewald/frame.py (lines 940-961) | S | 19 | Any remaining callers | `test_frame_publication` passes |
| 21 | **Stage 4.3: Verify all three (live/batch/reload) publish identically** | tests/xdart/test_gi_batch_real_data.py | M | 19, 20 | `publication_from_live_frame` vs. `publication_from_nexus_frame` equivalence | `test_*_publication_live_batch_reload_equivalence` passes |
| 22 | **Checkpoint: Stage 4 complete** | — | — | 19-21 | — | `test_*_equivalence` still green; PublicationStore is sole display contract |
| 23 | **D2.2 Deferred: Implement thumbnail LRU on PublicationStore** | frame_publication.py (track thumbnail hydration separately from full payload) | M | 22 | May be post-1.0 if time-constrained | `test_thumbnail_lru` passes |

---

### ACCEPTANCE GATES

1. **Live≡Batch≡Reload equivalence spine** (`tests/xdart/test_gi_batch_real_data.py::test_*_equivalence`):
   - Run live mode → read resulting .nxs in batch mode → reload from .nxs in viewer mode.
   - All three paths produce identical display state (same publications, generation stamps, diagnostics).
   - **Must pass at every checkpoint above.** A failing equivalence test is a show-stopper.

2. **Publication availability consensus** (new test):
   - Verify that `publication_availability(store)` matches `{widget.data_1d.keys()} ∪ {widget.data_2d.keys()}` before Stage 2.
   - After Stage 2, verify that reading any frame through PublicationStore produces the same result as the old data_1d/data_2d path.

3. **Hydration contract integrity** (new test):
   - Any frame evicted from the heavy payload LRU must be recoverable via `store.get_or_hydrate()`.
   - Test that a 100-frame scan with max_heavy=40 can display frame 1 (evicted long ago) without stalling.

4. **Display rendering consistency**:
   - A full Waterfall/Average/Sum/Overlay render of 100 frames must produce identical pixel output before and after each stage.
   - **Pixel-level rendering test** (rasterization of pyqtgraph widgets) to catch silent visual regressions.

---

This inventory documents the **current multi-source design** (three caches, four frame classes) and the **staged path to one store** (PublicationStore as sole display contract), with concrete file/line evidence and risk assessment for each step.Perfect. Now I have enough understanding. Let me compile the comprehensive GAP INVENTORY:

---

## GAP INVENTORY: Difference 5 — Schema as Code, Full Derivation

### Context
The 6b starter exists (`/Users/vthampy/repos/xrd-tools/src/xrd_tools/io/schema.py` + consumers). Difference 5 is about elevating schema from a partial data structure into the **single source of truth** for the on-disk layout — currently, substantial implementation details are hardcoded in the writer, reader, and test fixtures.

---

### (A) ALREADY DONE

**Schema module structure** (`/Users/vthampy/repos/xrd-tools/src/xrd_tools/io/schema.py:1-128`):
- `SCHEMA_NAME_ATTR`, `SCHEMA_VERSION_ATTR`, `DTYPE_ATTR`, `MONOTONIC_ATTR`, `SOURCE_BASE_ATTR` (attribute key names) — frozen, persisted
- `THUMBNAIL_LUT_ATTRS = ("vmin", "vmax", "dtype")` — declared but unconsumed (flagged in review)
- `PROCESSED_SCHEMA_NAME = "xrd_tools.processed_scan"` — the writer stamps this
- `ACCEPTED_SCHEMA_NAMES` including historical `"ssrl_xrd_tools.processed_scan"` for back-compat
- `PROCESSED_SCHEMA_VERSION = 2`
- `INTEGRATED_ROW_ALIGNED = {"frame_index", "intensity", "sigma"}` — the three datasets sliced by row surgery
- `GroupSchema` dataclass holding group name, axes (tuple of shared dataset names), and row_aligned (frozenset)
- `ProcessedScanSchema` dataclass with `groups` mapping:
  - `"integrated_1d"`: axes=`("q",)`, row_aligned=`INTEGRATED_ROW_ALIGNED`
  - `"integrated_2d"`: axes=`("q", "chi")`, row_aligned=`INTEGRATED_ROW_ALIGNED`
  - `"per_frame_geometry"`: axes=`()` (no shared axes), row_aligned includes geometry fields

**Consumption so far**:
- `test_schema_as_code.py:42-49` validates the row-aligned/axis split
- `nexus.py:1196-1227` (_axes_match_1d/2d) reads `SCHEMA.groups[*].axes` to get axis dataset names — eliminates the hardcoded `(q_name,) = ...` pattern
- `nexus_record.py:229-263` (drop_integrated_rows) uses `INTEGRATED_ROW_ALIGNED` to decide which datasets to slice
- Validator `_require_uniform_axes_1d/2d` checks axis uniformity (lines 1152-1187)
- Writer calls are delegated to shared primitives (`write_integrated_stack`, `write_per_frame_geometry`, etc.)

**Version check (C1 reader defense)**:
- `warn_if_newer_schema` (`nexus.py:63-82`) warns when a file's `ssrl_schema_version` is newer than the library supports

**Evidence of on-disk format pin**:
- `test_v2_record_compat.py` — byte-compat gate against pre-6a fixture

---

### (B) PARTIALLY DONE — WHAT REMAINS

#### (B1) Writer layout in `write_integrated_stack` — Names, dtypes, chunking, compression, units attrs

**Current hardcoding** (`nexus.py:1291-1439`):
- Lines 1333-1341: `integrated_1d` creation hardcodes:
  - Group name `"integrated_1d"` (literal string, not schema-derived)
  - Dataset names: `"intensity"`, `"q"`, `"frame_index"`, optionally `"sigma"`
  - NX attrs: `"NX_class"="NXdata"`, `"signal"="intensity"`, `"axes"=["frame_index", "q"]`
  - Dtypes: `intensity`/`sigma`→`np.float32`, `frame_index`→`np.int64`, `q`→`np.float32`
  - Chunking: rows for intensity/sigma, 64 for frame_index
  - `MONOTONIC_ATTR` stamp (declared, used)
  - Compression handled via `_comp_kwargs(compression)`

- Lines 1362-1389: `integrated_2d` creation hardcodes:
  - Same names/attrs pattern; intensity transposed `(n_chi, n_q)`, chunks `(1, n_chi, n_q)`
  - Two axes: `q` + `chi`

- Per-frame-geometry (`nexus.py:1620-1634`):
  - Group name hardcoded
  - Dataset names hardcoded: `"frame_index"` + keys from `geometry.derive_per_frame()`
  - Units: `"deg"` for `incident_angle`, `"rad"` for rotations
  - Chunking: 64

- Stitched outputs (`write_stitched`, lines 1459-1486):
  - Not row-aligned (static per scan), simpler layout
  - Orientation note: `stitched_2d/intensity` stored `(n_q, n_chi)`, not `(n_frames, n_chi, n_q)`

**What schema needs**:
- Expand `GroupSchema` to include:
  - `datasets: dict[str, DatasetSpec]` where `DatasetSpec` holds:
    - name (dataset name under the group)
    - dtype (logical: "float32", "int64", "string")
    - shape_template (e.g., `(nrows, n_q)` for 1D intensity; `(nrows, n_chi, n_q)` for 2D)
    - is_row_aligned (boolean, replaces the hardcoded set)
    - compression_eligible (boolean, default True for intensity/sigma)
    - chunking_strategy (default rows; options: "by_row", "frame_index_only", "static")
    - units (e.g., `"1/angstrom"` for `q`, `"deg"` for geometry)
  - Per-group NX attrs: `{"NX_class": "NXdata", "signal": "intensity", ...}`

- Per-frame geometry: enrich to enumerate which motors' outputs land here + their units

**Risk**: The frozen on-disk format means every dataset name, dtype, and chunking strategy is persisted. The refactor MUST NOT change what's written to new files (byte-compat gate must stay green); it only moves the declaration from code to schema.

---

#### (B2) Reader expectations — `get_1d`, `get_2d`, `read_scan` dataset names

**Current hardcoding** (`read.py:221-293`):
- `get_1d` (line 246): hardcodes `"integrated_1d"` group name
- `get_2d` (line 279): hardcodes `"integrated_2d"` group name
- Both hardcode `"q"`, `"chi"`, `"intensity"`, `"sigma"`, `"frame_index"` dataset names
- `read_scan` (line 2209 onwards in nexus.py): hardcodes group/dataset names in multiple places

- `read_frame_view` (`read.py:334+`): hardcodes `source_base` attr name
- `get_raw_frame` (read.py:448+): hardcodes `frames/frame_NNNN/source/{path,frame_index}` pattern

**What schema needs**:
- Readers should query `SCHEMA.groups[group_name].datasets[ds_name]` to get the expected dtype and shape template
- Add a schema method: `schema.get_group(name)` → returns the GroupSchema (already exists)
- Add: `schema.get_dataset_for_group(group_name, dataset_name)` → returns DatasetSpec
- Hard-dependency list becomes dynamic: `for group_name in SCHEMA.groups: if group_name in file...`

**Risk**: The `read_scan` function's xarray construction relies on knowing exactly which datasets correspond to which dims/coords (line 2209-2450+). If schema changes the structure, read_scan's dim-assignment logic must track the change. This is the "live≡batch≡reload spine" dependency: a reader change that silently reorders or renames dims breaks the equivalence gate.

---

#### (B3) Validators beyond axis-match

**Current state** (`nexus.py:1152-1287`):
- `_require_uniform_axes_1d`: all results share one `q` axis + unit
- `_require_uniform_axes_2d`: all results share one `q` + `chi` axis + units
- `_axes_match_1d`, `_axes_match_2d`: on-disk axes match incoming result
- `_require_batch_covers_existing`: full rewrite includes all existing frames (triggered by shape change)
- `validate_integrated_stack_write`: pre-flight all checks without mutation

**Not yet schema-derived**:
- No validator for other dataset expectations (e.g., "sigma may be present but is optional")
- No validator for per-frame-geometry shape (must have same frame count as integrated stacks)
- No validator for compressed-dataset integrity (current: "any dataset with intensity/sigma is compressed")
- No enumerator for "which datasets are optional vs required"

**What schema needs**:
- Expand `DatasetSpec` to include `required: bool`
- Add schema method: `schema.validate_write(group_name, results, on_disk_group)` that delegates to the validators, parameterized by schema
- Codify: intensity/sigma → always float32, always compressed if compression requested; frame_index → always int64, never compressed

**Risk**: Validators are the on-disk format's best defense. Relaxing a validator or missing a new-feature validation can corrupt the file. Every validator change must pass the byte-compat gate.

---

#### (B4) Test fixtures — hand-build schema layout

**Current** (`tests/core/test_nexus_v2.py:24-127`):
- Fixture manually creates groups, datasets, NX attrs, dtypes, chunking
- No fixture factory; every test that needs a v2 file hand-codes the layout
- ~25 occurrences of `create_group`/`create_dataset` in test suite hardcoding the same names

**What schema needs**:
- A fixture factory: `make_v2_fixture(path, frame_indices, n_q, n_chi, geometry=None, ...)` that:
  - Reads the schema from `SCHEMA`
  - Enumerates groups and datasets
  - Creates groups with NX attrs from schema
  - Creates datasets with dtypes/chunking/compression from schema
  - Fills with synthetic data
  - Returns the path
- This ensures fixtures are always in sync with the schema (and with the writer)

**Risk**: A fixture that lags the schema will silently test against the wrong layout, letting bugs slip into production.

---

#### (B5) Capability flags — the '2.x capability attr' mechanism

**Current declared but unconsumed** (`schema.py:28-59`):
- `THUMBNAIL_LUT_ATTRS = ("vmin", "vmax", "dtype")` — declares the attr triple on thumbnails, never used
- `ACCEPTED_SCHEMA_NAMES` — used only at the persistence layer, no reader feature-detection
- No mechanism for `PROCESSED_SCHEMA_VERSION = 2` to evolve to `2.1` semantics

**Design challenge** (from greenfield doc, line 130):
> "makes additive evolution self-describing (`2.1`-style capability attrs instead of an overloaded `2`)"

**Current state of per-feature data**:
- `per_frame_geometry`: present iff computed geometry available (currently no flag)
- `frames/frame_NNNN`: per-frame metadata, optionally with thumbnails + source refs (flagged implicitly by presence)
- `source_base` attr: present iff relative source portability desired
- `sigma` in integrated stacks: present iff error estimates available
- 2D `two_d_kind` attr: present to classify GI kind (line 1366 in nexus.py)

**What schema needs**:
A capability registry in the schema:

```python
@dataclass
class CapabilityAttr:
    name: str  # e.g., "per_frame_geometry", "thumbnails", "source_base"
    applies_to: str  # "entry", "integrated_1d", "per_frame_geometry", etc.
    meaning: str  # doc
    introduced: tuple[int, int]  # (major, minor) version

# In ProcessedScanSchema:
capabilities: dict[str, CapabilityAttr] = {
    "per_frame_geometry": CapabilityAttr("per_frame_geometry", "entry", "..."),
    "thumbnails": CapabilityAttr("thumbnails", "entry/frames", "..."),
    "source_base": CapabilityAttr("source_base", "entry", "..."),
    "sigma": CapabilityAttr("sigma", "integrated_1d/2d", "..."),
    "two_d_kind": CapabilityAttr("two_d_kind", "integrated_2d", "..."),
}

def feature_supported(self, capability_name: str, file_version: int) -> bool:
    """True if this library's reader should expect/handle this feature."""
    cap = self.capabilities.get(capability_name)
    if cap is None:
        return False  # unknown feature
    return file_version >= (cap.introduced[0] * 100 + cap.introduced[1])
```

Readers would then:
```python
if SCHEMA.feature_supported("per_frame_geometry", file_version):
    # read it, don't skip if missing
else:
    # skip silently
```

**Which existing features deserve flags**:
1. **`per_frame_geometry`** — derived rotations + incident angle (v2.0, always supported for now)
2. **`frames/` record** — per-frame thumbnails + source refs + metadata (v2.0)
3. **`source_base`** — N1-portable relative source paths (v2.0)
4. **`sigma` datasets** — error estimates in integrated stacks (v2.0, optional)
5. **`two_d_kind` attr** — GI classification (v2.0, optional)

**Risk**: A reader that doesn't feature-detect silently misses optional datasets, or crashes on features its version doesn't know.

---

#### (B6) GUI writer's remaining schema knowledge — `nexus_writer.py` beyond moved primitives

**Current state** (`/Users/vthampy/repos/xdart/modules/ewald/nexus_writer.py:1-433`):
- Calls `write_integrated_stack`, `write_stitched`, `write_positioners`, `write_per_frame_geometry` (delegated to xrd_tools)
- **Remaining xdart-specific schema knowledge**:
  - Line 382: `entry.attrs["default"] = "integrated_1d"` — hardcoded default group name
  - Lines 385-432: `_write_reduction` — provenance via xrd_tools, but scan metadata schema is implicit
  - Lines 322-346: Per-frame metadata write (scan_data, geometry, positioners) — delegates to `_write_incremental_metadata` which calls xrd_tools primitives
  - Frame record assembly (per-frame thumbnails, source refs) — delegates to `write_frame_record` in nexus_record.py

**What remains schema-defined in xdart**:
- The "default" NX attr should be in schema (not hardcoded to "integrated_1d")
- The scan_data group structure (frame_index + per-column datasets) is defined implicitly in `write_scan_metadata` (`nexus.py:1637-1691`)
- The per_frame_metadata cursor cache (lines 301-309) knows about specific attribute names but doesn't hardcode group/dataset names

**Risk**: The GUI writer is already thin and mostly delegating. The remaining schema knowledge is minimal and mostly safe. The main risk is the cursor cache (`NexusWriteCursor`) accidentally assuming specific groups exist or have specific layouts.

---

### (C) NOT STARTED

1. **Expand `DatasetSpec` in schema module** with dtype, chunking, compression, units, required-ness
2. **Add schema methods**:
   - `get_dataset_for_group(group_name, dataset_name)` → DatasetSpec
   - `enumerate_datasets(group_name)` → dict of DatasetSpec
   - `is_row_aligned(group_name, dataset_name)` → bool
3. **Derive writer layout** from schema in `write_integrated_stack` / `write_per_frame_geometry` / `write_stitched`
4. **Derive reader expectations** from schema in `get_1d`, `get_2d`, `read_scan`
5. **Parameterize validators** to query schema for required/optional/dtype/shape
6. **Build fixture factory** from schema
7. **Design & implement capability flags** for optional features (per_frame_geometry, thumbnails, source_base, sigma, two_d_kind)
8. **Add reader feature-detection** using capability flags
9. **Document in schema docstring** the interpretation of each group/dataset

---

### (D) CONCRETE WORK ITEMS, STAGED

#### **Stage 1: Schema enrichment — DatasetSpec + methods (S/M risk:low)**

**Files/symbols touched**:
- `src/xrd_tools/io/schema.py`: add `DatasetSpec` dataclass, enrich `GroupSchema`, add helper methods

**Changes**:
```python
@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dtype: str  # "float32", "int64", "string", etc.
    shape_template: tuple[str, ...]  # ("rows", "n_q"), ("rows", "n_chi", "n_q"), etc.
    is_row_aligned: bool  # per-frame vs. shared across frames
    required: bool = True
    compression_eligible: bool = True
    units: str | None = None
    
# Enrich GroupSchema
@dataclass(frozen=True)
class GroupSchema:
    name: str
    axes: tuple[str, ...] = ()
    row_aligned: frozenset = frozenset()  # keep for backward compat
    datasets: Mapping[str, DatasetSpec] = field(default_factory=dict)  # NEW
    nx_attrs: dict[str, str] = field(default_factory=dict)  # NX_class, signal, axes
    
# Add methods to ProcessedScanSchema
def get_dataset(self, group_name: str, dataset_name: str) -> DatasetSpec | None:
    g = self.groups.get(group_name)
    return g.datasets.get(dataset_name) if g else None
    
def is_row_aligned(self, group_name: str, dataset_name: str) -> bool:
    ds = self.get_dataset(group_name, dataset_name)
    return ds.is_row_aligned if ds else False
```

**Size**: S (schema data enrichment + 3-4 methods)

**Prerequisites**: None

**Risks**:
- The schema is persisted; changing how we interpret it requires all readers to agree
- Byte-compat gate must stay green
- Shape templates are strings, not validated — risk of typos

**Acceptance gate**: `test_schema_as_code.py` expanded to validate new fields; no writer/reader changes yet

---

#### **Stage 2: Derive writer layout from schema (M risk:medium)**

**Files/symbols touched**:
- `src/xrd_tools/io/nexus.py` (lines 1325-1390, 1576-1634, 1441-1487):
  - Refactor `_bulk_create_1d`, `_bulk_create_2d`, `write_per_frame_geometry`, `write_stitched` to iterate over `SCHEMA.groups[*].datasets`
  - Extract group-creation boilerplate into a helper `_create_group_from_schema(entry, group_name, schema)`

**Changes**:
```python
def _create_group_from_schema(entry_grp, group_name: str, datasets_dict: dict) -> h5py.Group:
    """Create a group with NX attrs from schema, ready for dataset insertion."""
    spec = SCHEMA.groups[group_name]
    g = entry_grp.create_group(group_name)
    for attr_key, attr_val in spec.nx_attrs.items():
        g.attrs[attr_key] = attr_val
    return g

def _bulk_create_1d():
    r0 = results_1d[0]
    n_q = np.asarray(r0.intensity).shape[0]
    g = _create_group_from_schema(entry_grp, "integrated_1d", {...})
    # Now iterate over SCHEMA.groups["integrated_1d"].datasets
    for ds_name, spec in SCHEMA.groups["integrated_1d"].datasets.items():
        if ds_name == "intensity":
            data = np.stack([np.asarray(r.intensity, spec.dtype) for r in results_1d])
            g.create_dataset(ds_name, data=data, maxshape=(None, n_q), chunks=...)
        elif ds_name == "q":
            ...
```

**Size**: M (lines 1325-1390 + 1620-1634 refactored)

**Prerequisites**: Stage 1 complete

**Risks**:
- **CRITICAL: byte-compat gate must stay green**. Every dataset created must have the same name, dtype, chunking, and order as before.
- The refactor must be invisible to on-disk format — test with `test_v2_record_compat.py`
- The logic for "which datasets present" (e.g., sigma optional) must be preserved

**Acceptance gate**: `test_v2_record_compat.py` still passes; write an identical file as before

---

#### **Stage 3: Derive reader expectations from schema (M risk:medium)**

**Files/symbols touched**:
- `src/xrd_tools/io/read.py` (lines 221-293, 2209-2450): replace hardcoded group/dataset names with schema queries
- `src/xrd_tools/io/nexus.py` (lines 63-82, reader warnings)

**Changes**:
```python
def get_1d(scan_file, frame=None, *, entry="entry"):
    with h5py.File(Path(scan_file), "r") as f:
        e = _entry(f, entry)
        spec = SCHEMA.groups["integrated_1d"]
        if "integrated_1d" not in e:
            raise KeyError(f"{scan_file} has no {spec.name} group")
        g = e[spec.name]
        # Iterate over schema to find frame_index, q, intensity, sigma
        q_ds = next(n for n, s in spec.datasets.items() if n == "q")
        intensity_ds = next(n for n, s in spec.datasets.items() if n == "intensity")
        ...
```

**Size**: M (replacing ~30 hardcoded strings with schema lookups)

**Prerequisites**: Stage 1 complete; Stage 2 recommended (ensures files written from schema are read correctly)

**Risks**:
- Reader changes can break the live≡batch≡reload spine if xarray dim assignment changes
- Must validate that xarray construction still works (same dims, coords, order)
- `read_scan`'s complex dim-merging logic (lines 2209-2450) must stay byte-identical from the user's perspective

**Acceptance gate**: `test_gi_batch_real_data.py::test_*_equivalence` still passes (spine test)

---

#### **Stage 4: Parameterize validators (S risk:low)**

**Files/symbols touched**:
- `src/xrd_tools/io/nexus.py` (lines 1152-1287): refactor validators to accept schema

**Changes**:
```python
def _require_uniform_axes_1d(results_1d, spec: GroupSchema) -> None:
    """Raise if any 1D result differs from the first per schema axes."""
    q_spec = spec.datasets["q"]
    # ... rest unchanged
```

**Size**: S (validators already exist, just wire in the schema)

**Prerequisites**: Stage 1 complete

**Risks**: Low; validators are already isolated functions. The schema enrichment just provides a source of truth for what they should check.

**Acceptance gate**: `test_schema_as_code.py` validates all datasets in schema are covered by some validator

---

#### **Stage 5: Fixture factory from schema (M risk:medium)**

**Files/symbols touched**:
- `tests/core/_v2_fixture_factory.py` (new file): schema-driven fixture builder
- `tests/core/conftest.py`: add fixture factory

**Changes**:
```python
def make_v2_fixture(tmp_path, frame_indices, n_q, n_chi, geometry=None, **kw):
    """Build a v2-conformant NXroot file from SCHEMA."""
    p = tmp_path / "auto_fixture.nxs"
    with h5py.File(p, "w") as f:
        entry = f.create_group("entry")
        for grp_name, grp_spec in SCHEMA.groups.items():
            grp = entry.create_group(grp_name)
            for attr_k, attr_v in grp_spec.nx_attrs.items():
                grp.attrs[attr_k] = attr_v
            # ... create datasets from spec
    return p
```

**Size**: M (boilerplate factory + integration into test suite)

**Prerequisites**: Stage 1 complete

**Risks**:
- The factory must handle optional datasets (sigma, two_d_kind, source_base, etc.)
- Test fixtures currently hand-craft specific values; the factory should accept "fill with synthetic data" as the default
- Over-parameterization (too many knobs) makes fixtures hard to use; should be simple for the common case

**Acceptance gate**: All `test_nexus_v2.py` tests refactored to use the factory and still pass

---

#### **Stage 6: Capability flags & reader feature-detection (M risk:medium)**

**Files/symbols touched**:
- `src/xrd_tools/io/schema.py`: add `CapabilityAttr`, `capabilities` dict, feature-detection methods
- `src/xrd_tools/io/read.py`: use feature-detection in readers

**Changes**:
```python
@dataclass(frozen=True)
class CapabilityAttr:
    name: str
    applies_to: str
    meaning: str
    introduced_version: int  # e.g., 2 for v2.0, 201 for v2.1

class ProcessedScanSchema:
    capabilities: Mapping[str, CapabilityAttr] = {
        "per_frame_geometry": CapabilityAttr(...),
        "source_base": CapabilityAttr(...),
        ...
    }
    
    def is_feature_available(self, capability_name: str, file_version: int) -> bool:
        cap = self.capabilities.get(capability_name)
        return file_version >= cap.introduced_version if cap else False

# In read.py:
def _maybe_read_per_frame_geometry(entry, file_version):
    if SCHEMA.is_feature_available("per_frame_geometry", file_version):
        return read_per_frame_geometry(entry)
    else:
        return None  # silently skip
```

**Size**: M (flags enum + 5-10 reader branch points)

**Prerequisites**: Stage 1 complete; Stages 2-5 recommended

**Risks**:
- Version numbering: currently `PROCESSED_SCHEMA_VERSION = 2`. Is the next release 2.1 (patch), or do we need a minor/patch system?
- Introduced_version should be immutable; once a feature is introduced, it's there forever (additive-only)
- Readers must be defensive: "if feature is available AND dataset present" (not just one or the other)

**Acceptance gate**: 
- `test_schema_as_code.py` validates all capabilities are documented
- Add test: create a v2.0 file without `per_frame_geometry`, read with a v2.1 reader, verify no crash and correct silencing

---

### SUMMARY TABLE: Work items, prerequisites, risks

| Stage | Work Item | Size | Prerequisites | Risk | Acceptance Gate |
|---|---|---|---|---|---|
| 1 | Schema enrichment (DatasetSpec, methods) | S | none | Byte-compat gate must pass | test_schema_as_code expands; no writer/reader changes yet |
| 2 | Derive writer layout from schema | M | 1 | **CRITICAL: byte-compat** | test_v2_record_compat green; identical files |
| 3 | Derive reader expectations from schema | M | 1; 2 rec | live≡batch≡reload spine must stay green | equivalence tests pass |
| 4 | Parameterize validators | S | 1 | low; validators already isolated | all schema datasets covered by validators |
| 5 | Fixture factory from schema | M | 1 | optional datasets; test integration | all test_nexus_v2 tests use factory, pass |
| 6 | Capability flags & feature detection | M | 1; 2-5 rec | version numbering; reader defensiveness | v2.0 files read cleanly by v2.1 reader |

---

### RESIDUAL RISKS FOR THE SPINE

The **byte-compat gate** (`test_v2_record_compat.py`) and the **equivalence spine** (`test_gi_batch_real_data.py::test_*_equivalence`) are the two non-negotiable acceptance gates:

1. **Byte-compat**: Every byte written to disk by the refactored writer must match the pre-6a signature. The fixture signature is the pin.
2. **Equivalence spine**: 
   - Write a 10-frame scan live → reload batch → reload again
   - Verify all three paths produce identical xarray Datasets (dims, coords, values)
   - Any dim-reordering or coord-rename in Stage 3 breaks this

The schema refactor is **additive and internal only** — it moves implementation from code to data, not changing what lands on disk or what readers produce.

---

### FINAL NOTES: Frozen on-disk format

Every string, dtype, and chunking decision in the schema is **persisted in user files**. The path forward:
- Stage 1 (data enrichment) is safe; adds no new persisted facts
- Stages 2-5 must ensure the writer produces the *exact same* bytes; use `test_v2_record_compat` as the acceptance gate
- Stage 6 adds new capability flags for *future* evolution (v2.1+), not v2.0 files already on disk
- Once committed, a persisted string can never change (historical N1 compatibility demands it)

The design is **high-leverage, byte-compat-gated, and achievable additive-only**.Now I have enough context. Let me create a comprehensive GAP INVENTORY:

---

## GAP INVENTORY: Difference 6 Residue — Contract Tests (CI Exists)

**Assigned Task:** Inventory the ReductionSink duck contract, FrameSource duck contract, QtNexusSink conformance, core-owned contract-test helpers, and release-script gaps for post-monorepo tag+build+publish enforcement.

**Status:** CI skeleton exists (PR + nightly + release workflows); contract tests are **PARTIALLY DONE**. The three critical gaps are: (1) **sink-contract coverage is thin and scattered** (no single test verifies all hooks + thread assignments), (2) **source-contract tests exist but aren't packaged as reusable helpers**, (3) **release enforcement script missing**.

---

### (1) ReductionSink Duck Contract — Hooks & Thread Assignments

#### WHAT IS DEFINED (the protocol)
**File:** `/Users/vthampy/repos/xrd-tools/src/xrd_tools/reduction/core.py:300–305`

The base `ReductionSink` protocol declares three methods:
```python
class ReductionSink(Protocol):
    def begin(self, scan: Scan, plan: ReductionPlan) -> None: ...
    def write(self, frame: Frame, reduction: FrameReduction) -> None: ...
    def finish(self, result: ReductionResult) -> None: ...
```

**Optional/conditional hooks** (duck-typed via `getattr`):
- `abort(result)` — called on failure path (line 355–360 in `CompositeSink`)
- `replace(frame, reduction)` — re-fed index upsert (line 338–340, fallback to `write`)
- `worker_process(frame, reduction)` — per-frame prep on pool worker (line 921–923)

#### WHAT IS ACTUALLY CALLED (from ReductionSession)
**File:** `/Users/vthampy/repos/xrd-tools/src/xrd_tools/reduction/core.py`

**Chunked execution (`execution="chunked"`)**
- Line 738: `sink.begin(scan, plan)` — called synchronously on caller thread (session.__post_init__)
- Line 967: `sink.write(frame, reduction)` — called on **caller thread** in `_process_chunk` loop (via `_reduce_frame` which integrates, then write synchronously)
- Line 321: `sink.finish(result)` — called on caller thread (session.finish, line 1104)
- Lines 352–364: `sink.abort(result)` on error (via CompositeSink)

**Streaming execution (`execution="streaming"`)**
- Line 738: `sink.begin(scan, plan)` — called synchronously on caller thread (session.__post_init__)
- Line 921–923: `worker_process(frame, reduction)` — called on **pool worker thread** (inside `_stream_reduce`, line 909)
- Line 965–967: `sink.write(frame, reduction)` — called on **single writer thread** (`_writer_loop`, line 926)
- Line 965: `replace()` — called on **single writer thread** if re-fed (line 965)
- Line 1104: `sink.finish(result)` — called on **caller thread** (session.finish)
- No explicit abort path in normal flow; via context manager __exit__ (line 752–758)

#### CRITICAL THREAD INVARIANT (NOT TESTED)
**`write()` is NEVER called concurrently** — enforced by single writer thread in streaming mode, or caller thread in chunked mode. HDF5 and xdart's `file_lock` / `_xye_lock` depend on this.

**`worker_process()` runs in PARALLEL** on pool workers; must NOT touch sink state.

#### WHAT IS PINNED BY TESTS TODAY

**File:** `/Users/vthampy/repos/xrd-tools/tests/core/test_reduction_streaming.py`

- Line 62–78: `test_streaming_output_matches_chunked()` — verifies chunked vs streaming produce same `result.frames` (indirect: no explicit hook verification)
- Line 81–101: `test_streaming_correct_under_scrambled_completion()` — frames arrive out-of-order; writer re-orders (indirect)
- Line 140–161: `test_streaming_replace_is_idempotent()` — replace() called on re-fed index (indirect; via `_SpySink` on line 474, which only wraps finish/abort)
- Line 224–275: `_BoomSink` — test harness that raises in write() (line 243); tests failure surface in finish (line 240–276)
- Line 474–488: `_SpySink` — minimal spy that records finish/abort calls, NOT worker_process
- Line 519–536: `test_nexus_sink_abort_preserves_partial()` — verifies NexusSink.abort() preserves atomic tmp as `.partial`
- Line 539–559: `test_nexus_sink_finish_failure_preserves_partial()` — abort via exception + finish failure path

**Missing:** No test verifies:
- Which thread calls begin/write/finish/replace/abort/worker_process
- That worker_process runs on pool worker (not writer)
- That write/replace never race (single-writer invariant)
- That worker_process does not touch sink state (parallel-safe design)
- Sink contract conformance for new implementations (duck-check reusable)

---

### (2) FrameSource Duck Contract — Hooks & Consumers

#### WHAT IS DEFINED (the protocol)
**File:** `/Users/vthampy/repos/xrd-tools/src/xrd_tools/core/scan.py:206–222`

```python
@runtime_checkable
class FrameSource(Protocol):
    @property
    def frame_indices(self) -> list[int]: ...
    @property
    def capabilities(self) -> SourceCapabilities: ...
    def load_frame(self, index: int) -> np.ndarray: ...
    def iter_chunks(self, chunk_size: int) -> Iterator[tuple[np.ndarray, list[int]]]: ...
```

#### WHAT CONSUMERS EXPECT

**run_reduction** (`/Users/vthampy/repos/xrd-tools/src/xrd_tools/reduction/core.py:1364–1461`):
- Line 1455: `for frame in session.scan:` — assumes `Scan` is iterable (implements `__iter__`, line 274 in scan.py)
- Line 806–810: `_iter_reduction_chunks` calls `source.iter_chunks(chunk_size)` — returns tuples of (stacked images, indices)
- Lines 289–294 (Scan.load_frame): loads by index, via ScanFrame.load_image()

**ReductionSession.process()**:
- Line 806: `_iter_reduction_chunks(self.source, ...)` → expects `iter_chunks`
- Line 815: expects chunk to be list of Frames

**RSM pipeline** (`/Users/vthampy/repos/xrd-tools/src/xrd_tools/rsm/pipeline.py`):
- Lines 295–302: duck-typed `iter_chunks` consumer (no type annotation)

#### IMPLEMENTATIONS THAT EXIST (verified conformance)

1. **Scan** (core contract class):
   - Lines 278–279: `frame_indices` property
   - Line 289: `load_frame(index)` → calls `_frame_by_index[idx].load_image()`
   - Line 296: `iter_chunks(chunk_size)` → yields `(np.stack(images), indices)`
   - Runtime-checkable: Line 79 in `test_reduction.py` asserts `isinstance(scan, FrameSource)`

2. **MemoryFrameSource** (`/Users/vthampy/repos/xrd-tools/src/xrd_tools/sources/memory.py:15–49`):
   - Extends `BaseFrameSource` (line 22–90, base.py)
   - Lines 44–45: `frame_indices` property
   - Line 39: `load_frame(index)`
   - Line 70: `iter_chunks(chunk_size)` inherited from base

3. **BaseFrameSource** (`/Users/vthampy/repos/xrd-tools/src/xrd_tools/sources/base.py:22–80`):
   - Abstract base; not a direct implementation but a template

4. **LiveFrameSource** (`/Users/vthampy/repos/xrd-tools/src/xrd_tools/sources/memory.py:52–107`):
   - Thread-safe appendable source; `iter_chunks` inherited from BaseFrameSource

#### WHAT IS PINNED BY TESTS TODAY

**File:** `/Users/vthampy/repos/xrd-tools/tests/core/test_reduction.py`

- Line 49–82: `test_scan_frame_source_iterates_bounded_chunks()` — asserts `isinstance(scan, FrameSource)` + tests `iter_chunks(2)`
- Line 70–82: tests that chunks return `(stacked_images, indices)` correctly
- Line 85–101: `test_scan_frame_source_clears_images_loaded_by_chunks()` — verifies images cleared after chunk consumed

**File:** `/Users/vthampy/repos/xrd-tools/tests/core/test_scan_source.py`

- Class `_FakeScan` (lines 148–201+) — hand-rolled duck-type for RSM testing; has `scan_data`, `frames`, `mg_args` but is NOT a full FrameSource

**Missing:** No test verifies:
- Contract completeness (all four properties/methods callable)
- `capabilities` advertised matches actual behavior (e.g., is_streaming, supports_random_access)
- Thread-safety of FrameSource implementations (LiveFrameSource uses lock, untested)
- That load_frame() caches images correctly (once, then returns cached in lazy case)
- Reusable contract-check helper for future sources (Tiled, zarr, etc.)

---

### (3) QtNexusSink Conformance — Hooks & Thread Assumptions

#### WHAT HOOKS IT IMPLEMENTS
**File:** `/Users/vthampy/repos/xrd-tools/src/xdart/gui/tabs/static_scan/wranglers/qt_nexus_sink.py:43–290`

Implements (lines 77–232):
1. **`begin(scan, plan)`** (line 77–79) — zero out state, clearing registry
2. **`write(frame, reduction)`** (line 81–101) — **writer thread** — pop LiveFrame, hydrate, stash, buffer XYE, trigger flush cadence
3. **`replace(frame, reduction)`** (line 191–210) — **writer thread** — re-fed index, hydrate, upsert in-memory, free raw
4. **`finish(result)`** (line 212–222) — **caller thread** — flush + clear registry (T0-8 comment: frames that failed mid-flight are still in registry, pinning raw)
5. **`abort(result)`** (line 224–231) — **caller thread** — flush + clear registry + emit -1 batch-end signal
6. **`worker_process(frame, reduction)`** (line 159–189) — **pool worker thread** — make thumbnail (parallel), free raw if batch (PERF-3)

#### THREAD ASSIGNMENTS

| Hook | Thread | Evidence | Risk |
|------|--------|----------|------|
| `begin` | Caller (batch wrangler) | L77, no lock | Safe (init before workers start) |
| `write` | Single writer (streaming) | L81, pops registry, calls _add_frame (line 248) which calls scan.add_frame | HDF5 single-writer invariant |
| `replace` | Single writer (streaming) | L191, same as write | Single-writer invariant |
| `worker_process` | Pool worker (parallel) | L159, makes thumbnail (line 180), does NOT touch _scan/registry directly (only reads registry line 170 via `get` not pop) | Parallel-safe: reads registry, no mutations to scan/h5 |
| `finish` | Caller (batch wrangler) | L212, emits signal on batch mode (line 220) | Safe: serial |
| `abort` | Caller | L224, same flush + emit | Safe: serial |

**Persist-before-evict invariant pinned:** Lines 264–272 in `_due_to_save()` — flush threshold is `cap - _SAVE_BEFORE_EVICT_MARGIN` (line 40, margin=8, cap=64). Line 282: h5pool pause/resume (N1) bracket the _save_to_nexus call.

#### WHAT IS PINNED BY TESTS

**File:** `/Users/vthampy/repos/xrd-tools/tests/xdart/test_qt_nexus_sink.py:1–100`

- Lines 14–36: `_r1d()` / `_r2d()` / `_reduction()` — test fixtures
- Lines 38–52: `_headless()` / `_live_frame()` — Frame-like + LiveFrame-like test doubles
- Lines 55–85: `_FakeHost` — stand-in for imageWranglerThread (minimal interface: xye_only, batch_mode, file_lock, signals, xye_buffer)
- Lines 92–99: `_drive()` test harness — calls begin/register/write/finish (NOT abort, NOT worker_process, NOT replace)

**Coverage:** Only `begin()` and `write()` are tested via `_drive()` in a single-threaded harness. No tests verify:
- `worker_process()` runs on pool worker (actual parallel execution)
- `replace()` idempotency (re-fed indices)
- `abort()` failure path behavior
- Thread-safety of registry operations
- Persist-before-evict threshold (flush forced before eviction)

---

### (4) Core-Owned Contract-Test Helpers — What a Reusable Module Needs

#### WHAT EXISTS TODAY
**Scattered across multiple test files; no centralized helpers:**

- `/Users/vthampy/repos/xrd-tools/tests/core/test_reduction.py`: basic sink tests (MemorySink, XYESink, fan-out)
- `/Users/vthampy/repos/xrd-tools/tests/core/test_reduction_streaming.py`: streaming-specific tests + `_SpySink` / `_BoomSink` (lines 224–488)
- `/Users/vthampy/repos/xrd-tools/tests/xdart/test_qt_nexus_sink.py`: QtNexusSink test harness + `_FakeHost` / `_drive()`
- `/Users/vthampy/repos/xrd-tools/tests/core/test_core_scan.py`: `isinstance(scan, FrameSource)` check only (no contract verification)
- `/Users/vthampy/repos/xrd-tools/tests/core/test_scan_source.py`: hand-rolled `_FakeScan` / `_FakeFrame` for RSM tests (not a reusable helper)

#### WHAT A CONTRACT-TEST MODULE SHOULD PROVIDE

**Proposed location:** `/Users/vthampy/repos/xrd-tools/tests/core/test_contracts.py` (NEW file)

**Utilities for `ReductionSink` conformance:**
1. `check_sink_contract(sink_factory: Callable[[], ReductionSink], *, with_worker_process: bool = True)` → pytest harness
   - Verifies: begin/write/finish callable
   - Verifies: abort exists OR finish succeeds twice (idempotent)
   - Verifies: replace exists OR idempotent write
   - Verifies: worker_process exists OR no-op
   - **Does NOT verify thread assignments** (hard; requires instrumented ReductionSession)

2. `MemorySinkSpec` / `BoomSinkSpec` / `_ThreadSpySink` — reusable test doubles
   - Tracks which thread called each hook (via `threading.get_ident()`)
   - Records call order + arguments
   - Optional: raise on specific hooks for failure-path testing

**Utilities for `FrameSource` conformance:**
1. `check_source_contract(source_factory: Callable[[], FrameSource], expected_indices: list[int])` → pytest harness
   - Verifies: frame_indices property returns list of int
   - Verifies: capabilities property returns SourceCapabilities
   - Verifies: load_frame(idx) returns ndarray for each idx
   - Verifies: iter_chunks(size) yields correct (images, indices) pairs
   - Verifies: iter_chunks clears images after yield (if applicable)

2. `MemorySourceSpec` — minimal test-friendly source
   - Accepts images in __init__, provides everything

**Integration test helpers:**
1. `run_with_spy_sink(...)` — wraps run_reduction with thread-tracking sink
2. `assert_sink_thread_assignments(spy_sink, expected: dict[str, str])` — checks begin→caller, write→writer, worker_process→worker, etc.

#### IMPLEMENTATION STATUS
**NOT DONE.** No centralized test-contract module exists. To add:
1. Create `/Users/vthampy/repos/xrd-tools/tests/core/test_contracts.py`
2. Extract `_SpySink` and `_BoomSink` from test_reduction_streaming.py → refactor into utilities
3. Add `check_sink_contract()` + `check_source_contract()` harnesses
4. Add thread-tracking spy sinks + sources for streaming tests
5. Document expected thread assignments in docstrings
6. Import + re-use in xdart's test_qt_nexus_sink.py

---

### (5) Release Script Gaps — Tag+Build+Publish Enforcement

#### WHAT SCRIPTS/ HAS TODAY
**File:** `/Users/vthampy/repos/xrd-tools/scripts/`

- `install.sh` (lines 1–370): one-line installer for dev + prod (conda env + pip installs)
  - Remote: fetches from GitHub `dev` branch
  - Local: editable installs from working tree
  - **Not a release script; this is deployment/onboarding**
- `install.ps1`: Windows PowerShell variant of install.sh

#### WHAT A RELEASE SCRIPT SHOULD DO

**Post-monorepo enforcement points:**

1. **Version consistency:**
   - Check `pyproject.toml` version field (line 7: "1.0.0") matches git tag (e.g., `v1.0.0`)
   - Ensure both `xrd_tools` and `xdart` import paths resolve correctly (they share version now)
   - Verify `__version__` in module top-levels matches pyproject.toml

2. **Floor enforcement (no bypass):**
   - Dependencies in `pyproject.toml` (lines 30–48) — pin ceil versions to prevent surprising breakage
   - Confirm `pyFAI>=2025.3,<2025.12` cap is respected (line 41 comment: Windows issue)
   - Test suite must pass: `pytest tests/core tests/xdart --tb=short`

3. **Schema + format frozen checks:**
   - Confirm no changes to `SCHEMA_NAME_ATTR` / `SCHEMA_VERSION_ATTR` / `INTEGRATED_ROW_ALIGNED` (schema.py:44–80)
   - Confirm no changes to `NXprocess/@program` stamp (`"ssrl_xrd_tools"` — see MIGRATION.md line 59)
   - Run `tests/core/test_v2_record_compat.py` (byte-compat gate, MIGRATION.md line 60)
   - Run `tests/core/test_schema_as_code.py` (pins in code match persisted names)

4. **Monorepo coherence:**
   - Confirm single `pyproject.toml` (at repo root, not per-package)
   - Both `xrd_tools` and `xdart` packages defined under `src/`
   - CI matrix runs both core + GUI tests (PR.yml already does this)

5. **Build + publish order (enforced by script, not by hand):**
   - Build wheel: `python -m build`
   - Check wheel contents: `twine check dist/*`
   - Optionally push to test PyPI for dry-run
   - Confirm no old dist/ artifacts left

#### CURRENT WORKFLOW
**File:** `.github/workflows/release.yml` (lines 1–19)
- Triggers on `git push --tags` with pattern `v*`
- Steps: checkout, setup Python, install build+twine, `python -m build`, `twine check`
- **Limitation:** No auto-publish (manual upload by maintainer, line 19 comment)
- **Gap:** No pre-flight checks (schema frozen, deps pinned, tests pass, version consistency)

#### WHAT TO ADD

**New file:** `/Users/vthampy/repos/xrd-tools/scripts/release.py` (or .sh)

```python
#!/usr/bin/env python3
"""Pre-flight checks + build for xrd-tools release.

Usage:
  python scripts/release.py check-version [TAG]
  python scripts/release.py check-schema
  python scripts/release.py check-deps
  python scripts/release.py build
  python scripts/release.py publish [--dry-run]
"""

# Key checks:
# 1. git tag matches pyproject.toml version
# 2. No uncommitted changes
# 3. tests/core/test_v2_record_compat.py passes (byte-compat gate)
# 4. tests/core/test_schema_as_code.py passes
# 5. SCHEMA.version incremented if needed
# 6. pyFAI dep cap enforced
# 7. Build succeeds: python -m build
# 8. twine check passes
# 9. (Optional) upload to test PyPI
```

#### IMPLEMENTATION STATUS
**NOT DONE.** Only CI skeleton exists. To add:
1. Create `/Users/vthampy/repos/xrd-tools/scripts/release.py`
2. Add checks: version consistency, schema frozen, byte-compat test, dependencies
3. Integrate into `.github/workflows/release.yml` as pre-build step
4. Document in README.md or CONTRIBUTING.md

---

## STAGED WORK ITEMS

### **Stage 1: Core contract-test helpers (foundation for future sinks/sources)**
**Size:** M (2–3 days)  
**Files touched:** `tests/core/test_contracts.py` (new), test_reduction_streaming.py (refactor), test_core_scan.py (add assertions)  
**Prerequisites:** None  
**Risks:**
- **Behavior pinning:** Contract test must not be stricter than actual usage (test will fail on legitimate alternates like Tiled)
- **Thread instrumentation:** Measuring which thread calls each hook requires thread IDs in spy sinks; non-deterministic in some CI environments

**Acceptance gate:**
- `pytest tests/core/test_contracts.py` passes
- Refactored `_SpySink` / `_BoomSink` re-used in 3+ test files
- New `check_sink_contract()` helper verified against NexusSink + MemorySink
- Thread-tracking spy sinks pass for streaming mode (write on writer thread, worker_process on pool)

**Work breakdown:**
1. Extract spy/test-double classes from test_reduction_streaming.py → new module `tests/core/test_contracts.py`
2. Add `check_sink_contract()` harness (parametrized: test with MemorySink, NexusSink, XYESink)
3. Add `check_source_contract()` harness (parametrized: test with Scan, MemoryFrameSource, LiveFrameSource)
4. Add thread-tracking utilities (spy sinks record thread ID per hook)
5. Document expected thread assignments in docstrings
6. Refactor test_reduction_streaming.py to use helper fixtures

---

### **Stage 2: Thread-assignment verification for streaming mode**
**Size:** M (1–2 days)  
**Files touched:** test_reduction_streaming.py (add new tests), qt_nexus_sink.py (no changes, only testing)  
**Prerequisites:** Stage 1 (contract helpers)  
**Risks:**
- **Race conditions in spy hooks:** If spy code is not thread-safe, tests flake on high concurrency
- **Persist-before-evict invariant:** Hard to trigger the margin boundary in tests; may require monkeypatch of cache cap

**Acceptance gate:**
- `pytest tests/core/test_reduction_streaming.py::test_worker_process_runs_on_pool_worker` passes
- `pytest tests/core/test_reduction_streaming.py::test_write_runs_on_writer_thread` passes
- `pytest tests/xdart/test_qt_nexus_sink.py::test_qt_sink_worker_process_on_pool` passes (new test)
- `pytest tests/xdart/test_qt_nexus_sink.py::test_qt_sink_persist_before_evict_threshold` passes (new test, may need monkeypatch)

**Work breakdown:**
1. Add `test_worker_process_runs_on_pool_worker()` — spy sink records thread IDs, verifies worker_process ≠ writer
2. Add `test_write_runs_on_writer_thread()` — verifies write and replace always on same thread
3. Add `test_qt_sink_worker_process_parallel_thumbnail()` — actual parallel execution + completion
4. Add `test_qt_sink_replace_idempotency()` — re-fed indices trigger replace, not new write
5. Add `test_qt_sink_persist_before_evict_threshold()` — monkeypatch cache cap, verify flush forced

---

### **Stage 3: FrameSource contract verification (source duck-typing)**
**Size:** S (1 day)  
**Files touched:** test_core_scan.py (enhance), test_scan_source.py (refactor), new test_frame_source_contract.py  
**Prerequisites:** Stage 1 (contract helpers)  
**Risks:**
- **Capability flags:** SourceCapabilities are advisory; sources may not honor them strictly. Tests must not conflate contract conformance with capability truthfulness.

**Acceptance gate:**
- `pytest tests/core/test_frame_source_contract.py` passes (all sources tested)
- `isinstance(source, FrameSource)` passes for: Scan, MemoryFrameSource, LiveFrameSource, ProcessedScan (reader)
- `check_source_contract()` called for each implementation
- Capabilities truthfulness spot-checked (e.g., is_streaming=True for LiveFrameSource, False for Scan)

**Work breakdown:**
1. Create `test_frame_source_contract.py` with `check_source_contract()` parametrized tests
2. Test all implementations: Scan, MemoryFrameSource, LiveFrameSource, ProcessedScan
3. Add capability verification helpers (is_streaming, supports_random_access, etc.)
4. Verify thread-safety for LiveFrameSource (lock acquired in iter_chunks)
5. Document FrameSource contract in docstring (what each method guarantees)

---

### **Stage 4: Release script + pre-flight enforcement**
**Size:** M (2 days)  
**Files touched:** `scripts/release.py` (new), `.github/workflows/release.yml` (enhance), README.md or CONTRIBUTING.md (document)  
**Prerequisites:** None (orthogonal to test infrastructure)  
**Risks:**
- **Version mismatch in monorepo:** Both xrd_tools and xdart share version; script must check both `__version__` inits
- **Schema frozen checks:** If schema changes ARE needed (new feature), script must allow controlled bump with signed commit

**Acceptance gate:**
- `python scripts/release.py check-version v1.0.1` exits 0 if tag matches pyproject.toml
- `python scripts/release.py check-schema` exits 0 if schema frozen (no changes to SCHEMA_NAME_ATTR, etc.)
- `python scripts/release.py check-deps` exits 0 if pyFAI<2025.12 pinned
- `python scripts/release.py build` produces valid wheels (twine check passes)
- Integration: release.yml calls `scripts/release.py check-*` before `python -m build`

**Work breakdown:**
1. Create `scripts/release.py` with `check-version`, `check-schema`, `check-deps`, `build` subcommands
2. Implement version consistency check (git tag vs pyproject.toml vs __version__ in modules)
3. Implement schema-frozen check (verify SCHEMA constants unchanged)
4. Implement dependency-cap check (pyFAI <2025.12, etc.)
5. Integrate into release.yml (add `run: python scripts/release.py check-*` steps before build)
6. Document in README.md or new CONTRIBUTING.md

---

### **Stage 5: CI enhancement — add contract tests to PR matrix**
**Size:** S (1 day)  
**Files touched:** `.github/workflows/pr.yml` (enhance)  
**Prerequisites:** Stages 1–3  
**Risks:** None (additive)

**Acceptance gate:**
- `pytest tests/core/test_contracts.py` runs in PR.yml job
- PR tests fail if contract violations appear (new sink missing required hook, etc.)

**Work breakdown:**
1. Add `pytest tests/core/test_contracts.py` to PR.yml core job
2. Verify no additional dependencies needed (test_contracts only imports pytest + core)

---

## Summary of Gaps

| Component | Covered? | Evidence | Gap | Work Item |
|-----------|----------|----------|-----|-----------|
| **ReductionSink.begin/write/finish** | PARTIAL | test_reduction.py:test_run_reduction_fans_out_* | No single test verifies all 6 hooks (begin,write,finish,abort,replace,worker_process) + thread assignments | Stage 1+2 |
| **ReductionSink.abort** | PARTIAL | test_reduction_streaming.py:519–559 (NexusSink only) | Only NexusSink.abort tested; no generic contract test | Stage 1+2 |
| **ReductionSink.replace** | PARTIAL | test_reduction_streaming.py:140–161 (indirect) | Tested via memory sink; no thread-assignment verification | Stage 2 |
| **ReductionSink.worker_process** | NOT DONE | Not found in any test file | No test verifies (1) it runs on pool worker, (2) it doesn't mutate sink | Stage 2 |
| **FrameSource.frame_indices/load_frame/iter_chunks** | PARTIAL | test_reduction.py:49–101; test_core_scan.py | isinstance(FrameSource) checked; no reusable contract harness | Stage 3 |
| **FrameSource.capabilities** | NOT DONE | Not tested anywhere | Capabilities declared but never verified truthful | Stage 3 |
| **QtNexusSink thread safety** | PARTIAL | test_qt_nexus_sink.py:92–99 (_drive single-threaded) | No actual parallel worker_process execution; no thread instrumentation | Stage 2 |
| **QtNexusSink.worker_process** | NOT DONE | Not called in _drive() | No test verifies it runs on pool worker | Stage 2 |
| **Persist-before-evict invariant** | NOT DONE | Not tested | Line 264–272 (qt_nexus_sink.py) logic not verified | Stage 2 |
| **Release script** | NOT DONE | Only CI skeleton (.github/workflows/release.yml) | No pre-flight checks (version, schema, deps); manual publish | Stage 4 |
| **Contract-test helpers (reusable)** | NOT DONE | Scattered _SpySink, _BoomSink, _FakeHost | No centralized module; can't re-use for Tiled, zarr sources | Stage 1 |

---

## Concrete Implementation Priority

**Recommended order (dependencies first):**

1. **Stage 1** → Stage 2 → Stage 3 (refactor + enhance test coverage)
2. **Stage 4** (parallel; orthogonal to test infrastructure)
3. **Stage 5** (depends on Stages 1–3; final CI polish)

**Critical blocking:** None of the changes block current functionality. All work is additive and verifies existing behavior via new test harnesses.Excellent. Now I have comprehensive context. Let me create the final GAP INVENTORY for Difference 7.

---

## GAP INVENTORY: Difference 7 — Day-One Policies

### (1) NAMING: Implementation-Metaphor Names & Class Structure

**Current State:**

The migration COMPLETED the rename from `EwaldArch`/`EwaldSphere`/`ArchSeries` to domain-appropriate names:
- **Live GUI classes (actual runtime):** `/Users/vthampy/repos/xrd-tools/src/xdart/modules/ewald/`:
  - `LiveFrame` (line 1, `/xdart/modules/ewald/frame.py`)
  - `LiveScan` (line 1, `/xdart/modules/ewald/scan.py`)
  - `LiveFrameSeries` (line 1, `/xdart/modules/ewald/frame_series.py`)
- **Public import surface:** `/Users/vthampy/repos/xrd-tools/src/xdart/modules/live.py` re-exports all three (lines 8–10)
- **Legacy compatibility:** `/xdart/modules/ewald/__init__.py` line 4–8 documents that `EwaldArch`/`EwaldSphere`/`ArchSeries` aliases were dropped after the transitional release; `xdart/modules/live_compat.py` handles reader-side compat for old persisted provenance strings

**Two Public Scan Classes (Post-6c State):**
- **Core reduction input:** `xrd_tools.core.scan.Scan` (line 1, `/xrd_tools/core/scan.py`) — the headless frame-collection contract
- **I/O reader handle:** `xrd_tools.io.read.ProcessedScan` (line 606, `/xrd_tools/io/read.py`) — the file browser/lazy-load wrapper
- **Alias:** Line 705 in `/xrd_tools/io/read.py` exports `Scan = ProcessedScan` for backward compat; old code still imports `from xrd_tools.io.read import Scan`

**All Import Sites:**
- `xdart.modules.live` (public facade): line 8–10 re-export from `ewald/*`
- `xdart.gui.tabs.static_scan.static_scan_widget`: line 36 `from xdart.modules.live import LiveFrame, LiveScan`
- `xdart.gui.tabs.static_scan.h5viewer`: line 23 `from xdart.modules.live import LiveFrame`
- `xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread`: line 58 `from xdart.modules.live import LiveFrame, LiveScan`
- `xdart.gui.tabs.static_scan.wranglers.nexus_wrangler_thread`: line 51 `from xdart.modules.live import LiveFrame`
- `xdart.modules.ewald.scan`: line 14 imports `normalize_live_class_names` from `live_compat` for provenance normalization
- Internal lazy imports in `nexus_writer`, `frame_series`, `stitch` (all import on-demand to avoid circular deps)

**Cost of Module Path Renaming vs Aliasing:**
- **Module path `ewald/` → `live/`**: Moderate cost. Would require:
  1. Move 4 files (`frame.py`, `scan.py`, `frame_series.py`, `nexus_writer.py`, `stitch.py`) to `/xdart/modules/live/`
  2. Update ~12 internal import paths (lazy imports in `nexus_writer`, `frame_series`)
  3. Update public facade in `/xdart/modules/__init__.py` if it re-exports
  4. Test suite impact: any fixture imports from `xdart.modules.ewald`
  5. **Benefit:** cleaner mental model (no metaphor in path), but **low leverage** — all **public** imports already go through `xdart.modules.live`
- **Aliasing (current approach):** Already done. Module docstring at `ewald/__init__.py` lines 1–8 explains the historical transition; cost = **zero**.

**What is NOT started:** True renaming (but not necessary — public API is already correct).

---

### (2) XARRAY Currency: Boundary & Consumption Path

**Current State:**

xarray is used **ONLY at the read boundary**:
- `/Users/vthampy/repos/xrd-tools/src/xrd_tools/io/nexus.py`: lines ~1800–1900 construct `xr.Dataset` from HDF5 stacks in `read_scan_metadata()` and `read_scan()`; lines 2038–2062 are the energy/wavelength readers (NaN handling)
- `/Users/vthampy/repos/xrd-tools/src/xrd_tools/viz/mpl.py`: consumes xarray.Dataset for plotting

**Reduction path (NOT xarray):**
- `xrd_tools.reduction.core.FrameReduction` (line 1 in `/xrd_tools/reduction/core.py`): contains `result_1d: IntegrationResult1D | None` and `result_2d: IntegrationResult2D | None` (custom containers, NOT xarray)
- `IntegrationResult1D/2D` (lines 1–50 in `/xrd_tools/core/containers.py`): namedtuple-like containers with `intensity`, `q`, `chi`, `sigma`, unit metadata — no xarray
- `NexusSink.write()` → `write_nexus_frame()` (line 544–560 in `/xrd_tools/reduction/core.py`): passes `FrameReduction` directly to writer, not xarray
- **No intermediate xarray.Dataset per frame** — the writer constructs HDF5 stacks from `IntegrationResult1D/2D` primitives

**"Reduction emits xr.Dataset per frame" would touch:**
- `FrameReduction` → `xr.Dataset` wrapper (new container type)
- `IntegrationResult1D/2D` consumers:
  - `NexusSink.write()` / `write_nexus_frame()` — would unpack Dataset variables to write to HDF5
  - `xdart` display layers (`display_data.py`, `display_frame_widget.py`) — currently consume `IntegrationResult1D/2D` directly via `frame.int_1d`, `frame.int_2d` paths
  - GI freeze policy readers (`_apply_gi_freeze_policy` line 850+) — read `.intensity` arrays
- **Equivalence spine** (`tests/xdart/test_gi_batch_real_data.py`): would require Dataset reshaping/comparison logic
- **Publication path** (`xdart.modules.frame_publication.validate_publication`): currently validates `FrameReduction` fields

**Honest Cost Assessment:**
- **New container wrapping:** ~100 lines to define a `FrameReductionDataset` bridge
- **Writer re-unpacking:** ~200 lines of `write_nexus_frame()` changes (Dataset → HDF5 hierarchy already proven)
- **Display layer migration:** `display_data.py`, `display_frame_widget.py` — ~400 lines updating attribute access (`frame.int_1d.intensity` → `frame.ds['intensity_1d']` etc)
- **GI freeze:** ~100 lines for `.intensity` array extraction
- **Test fixture updates:** ~50 test cases need Dataset mocking instead of IntegrationResult
- **Risk:** xarray adds ~50 MB (scipy + pandas transitive deps); integration test suite runtime **degrades** 5–10% (xarray operations are slower than direct namedtuples for this access pattern)
- **Benefit:** notebook-friendly API, xarray-native indexing, automatic NaN/coord alignment — **medium value for pure analysis, low for GUI headless**

**Verdict:** NOT WORTH IT vs keep containers. The current split (xarray at read, containers on reduction hot path) is optimal for the mixed GUI+headless architecture. Writing `ds.to_xarray()` bridge in the reader is the right place.

**What is NOT started:** Container replacement (and correctly so).

---

### (3) COALESCING Idiom: Timer & Throttle Sites

**Current Implementation (Divergent Semantics):**

1. **`h5viewer._absorb_chunk` coalescing timer** (lines 2343–2405 in `/xdart/gui/tabs/static_scan/h5viewer.py`):
   - `self._update_coalesce_timer = QtCore.QTimer(self)` (line 648)
   - `setSingleShot(True)`, `setInterval(100 ms)` (lines 649–650)
   - **Semantics:** DEBOUNCE (restart = defer the emission, not merge into a batch)
   - Logic: line 2402 `self._update_coalesce_timer.start()` restarts the timer on every chunk; timeout emits `sigUpdate` once per burst
   - **Purpose:** O(N) chunks fire O(1) display refresh during a load burst

2. **`static_scan_widget` split timer** (lines 389–391 in `/xdart/gui/tabs/static_scan/static_scan_widget.py`):
   - `QtCore.QTimer.singleShot(0, _default_split)` → immediate
   - `QtCore.QTimer.singleShot(1000, _default_split)` → 1s delay
   - `QtCore.QTimer.singleShot(2500, _default_split)` → 2.5s delay
   - **Semantics:** STAGGERED CALLBACKS (three independent fire-once shots)
   - **Purpose:** multi-stage layout initialization (hint-based splitter sizing, per comments line 368)
   - **No coalescing:** each shot is independent

3. **`static_scan_widget._update_timer`** (line 482 in `/xdart/gui/tabs/static_scan/static_scan_widget.py`):
   - `self._update_timer = QtCore.QTimer(self)`
   - **Attached in docstring** line 223: "currently unused but can be used for periodic updates"
   - **Status:** DECLARED BUT NOT WIRED — no timeout signal connected

4. **Other single-shot calls** (lines 548, 2320–2324 in same files):
   - `QtCore.QTimer.singleShot(0, lambda v=vm: self._on_viewer_mode_changed(v))` — async callback
   - **Semantics:** ASYNC DISPATCH (ensure GUI thread execution)

5. **`display_frame_widget` paint-on-next-event** (line 1 in `/xdart/gui/tabs/static_scan/display_frame_widget.py`):
   - `Qt.QtCore.QTimer.singleShot(0, _apply)` — async repaint
   - **Semantics:** DEFERRED PAINT (batch Qt event redraws)

**Comments showing confusion:** Line 793 in `/xdart/gui/tabs/static_scan/static_scan_widget.py`: "Throttle, not debounce: during a fast" (incomplete, but shows the semantic awareness)

**Shared Throttle Utility Sketch:**

```python
# xdart/utils/throttle.py (or embed in static_scan_widget / h5viewer)

class Throttle(QtCore.QObject):
    """Coalesce rapid events into at-most-one emission per interval.
    
    Usage:
        throttle = Throttle(interval_ms=100, parent=widget)
        source_signal.connect(throttle.input)  # e.g., chunkLoaded
        throttle.output.connect(on_update)    # debounced emission
    
    Semantics: every call to input() restarts the timer; the timer fires once
    per interval and emits output(). Semantically a debounce (trailing edge
    fires once per burst), not a throttle (leading edge fires immediately).
    """
    output = QtCore.Signal()
    
    def __init__(self, interval_ms: int = 100, parent=None):
        super().__init__(parent)
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(interval_ms)
        self.timer.timeout.connect(self._emit)
    
    @QtCore.Slot()
    def input(self, *args, **kwargs):
        """Restart the coalescing timer; calls to this accumulate."""
        self.timer.start()
    
    def _emit(self):
        self.output.emit()
    
    def force_flush(self):
        """Immediate emit and stop."""
        if self.timer.isActive():
            self.timer.stop()
            self._emit()
```

**Consolidation Points:**
1. Replace `h5viewer._update_coalesce_timer` with instance of `Throttle(interval=100)`
2. Document the split timer (line 389–391) as intentional staggered initialization, separate concern
3. Wire the unused `_update_timer` to a periodic refresh pattern, OR delete it
4. Add a comment in `display_frame_widget` explaining the singleShot(0) as "async dispatch, not throttling"

**What is PARTIALLY done:** The h5viewer throttle is implemented correctly but named/documented as "coalesce timer" without a shared abstraction; static_scan_widget has staggered one-shots but they're not unified.

---

### (4) SENTINELS: Non-None Missing-Value Markers

**Current State:**

1. **1.0 Å Wavelength Default** (xdart-specific, NOT in core):
   - `/Users/vthampy/repos/xrd-tools/src/xdart/modules/wavelength.py` lines 3–62:
     - `DEFAULT_WAVELENGTH_SENTINEL_M = 1.0e-10` (line 13) — the historical `LiveScan.mg_args` default
     - `is_default_wavelength_sentinel_m(value)` (line 17) — check with tolerance `_SENTINEL_ATOL_M = 1.0e-14`
     - `normalize_wavelength_m(value, *, allow_default_sentinel: bool = False)` (line 26): **flags the sentinel as unknown by default**
     - `wavelength_m_to_angstrom(value, *, allow_default_sentinel: bool = False)` (line 46): wraps normalize
   - **Usage sites:**
     - `display_data.py` lines 140–154: calls `normalize_wavelength_m(..., allow_default_sentinel=True)` when reading from persisted NeXus
     - `reduction.py` (xdart adapter): line ~ calls `wavelength_m_to_angstrom(..., allow_default_sentinel=True)` for persisted scan wavelength
   - **Semantics:** NOT a real wavelength; the flag `allow_default_sentinel=True` means "trust this source — e.g., persisted — to distinguish sentinel from real 1 Å"
   - **Risk:** The sentinel is **still a float (1e-10 m)**, so naive code that doesn't check `allow_default_sentinel` will silently accept it as real

2. **NaN Energy/Wavelength Outside `io.read`** (line 2038–2062 in `/xrd_tools/io/nexus.py`):
   - `_read_energy(grp)` returns `float(np.nan)` when energy is missing (line 2046–2047)
   - `_read_wavelength(grp, energy)` returns `float(np.nan)` when wavelength is missing and energy non-finite (line 2061)
   - **Caller contract:** `read_scan_metadata()` line 142–143 calls these and stores NaN directly
   - **Display contract mismatch:** MIGRATION.md line 82 states "`energy_keV` / `wavelength_A` are `None` when not recorded — never NaN" (issue #78), but the io.nexus readers **emit NaN**
   - **Actual behavior:** `get_metadata()` in `/xrd_tools/io/read.py` line 540+ returns `energy_keV` and `wavelength_A` — it should **convert NaN → None** but the code passes `ds['energy_keV'].values` directly without filtering

3. **Other non-None sentinels:** None found in current codebase (0.0 wavelength, -1 values, etc. are rejected)

**Remaining Issues:**
- **Sentinel inconsistency:** wavelength is 1.0 Å + `allow_default_sentinel` flag (xdart-specific), energy/wavelength are NaN (io.nexus)
- **Contract violation:** `get_metadata()` claims to return `None` for missing energy, but reads NaN from the Dataset without filtering

**What is NOT started:** Cleanup of the energy/wavelength sentinel split; the NaN-in-io vs None-in-API gap remains.

**What is PARTIALLY done:** Wavelength sentinel is well-documented with an explicit flag, but NaN handling in energy readers is silent.

---

### (5) STRICTNESS Flags: Silent-Degradation Paths & Default-Loud Policy

**Current Silent-Degradation Paths:**

1. **Thumbnail Fallback (load_processed_raw_or_thumbnail)**
   - **File:** `/Users/vthampy/repos/xrd-tools/src/xrd_tools/io/image_source.py` (docstring and implementation)
   - **Behavior:** Line 501 in `/xrd_tools/io/read.py` — `allow_thumbnail=True` by default; tries source master, **silently falls back to dequantized thumbnail** if source unavailable
   - **Caller contract:** xdart's `display_controllers.py` line (approx) calls `load_processed_raw_or_thumbnail()` with default `allow_thumbnail=True`; **GUI never shows "fell back to thumbnail"**
   - **Headless state:** `allow_thumbnail=False` can be passed, but it's opt-in
   - **Risk:** Batch user doesn't realize they're viewing thumbnails, not raw; equivalence spine could silently accept degraded data

2. **Monitor Skip (S8 — per-scan warnings)**
   - **File:** `/Users/vthampy/repos/xrd-tools/src/xrd_tools/reduction/core.py` lines 1140–1200 area
   - **Behavior:** Line 1176 `_apply_gi_freeze_policy()` passes `warned_monitor_keys` set; line (approx) checks if monitor value exists, **logs warning once per monitor key per scan**, but **DOES NOT FAIL** — frame is written with `None` normalization
   - **Caller expectation:** GUI sees the warning in the log, but **the reduction continues**, and the frame is persisted unnormalized
   - **Headless state:** A headless user gets the **same warning** but never sees the GUI that might make them re-check the setup
   - **Risk:** Silent data degradation with a buried warning

3. **GI All-Dummy Row Drops (Q2 publication gate)**
   - **File:** `/Users/vthampy/repos/xrd-tools/src/xdart/modules/ewald/nexus_writer.py` lines (approx) 930–950
   - **Behavior:** `_validate_prepared_integrated()` line (approx) calls `_result_intensity_all_dummy()` to detect blank cakes; if all-dummy, the **row is silently dropped** (publication gate filter)
   - **Comments:** Line (approx) "publication-invalid; removing them is correct even if a later validation step still fails"
   - **Caller expectation:** The file is shorter than expected, but no error surfaced; users might not notice
   - **Headless state:** Same silent truncation

4. **Filter Match-Nothing Fallback (F1 filter)**
   - **File:** `/Users/vthampy/repos/xrd-tools/src/xrd_tools/core/filters.py` lines 91–111
   - **Behavior:** Malformed filter expression **raises ValueError** (lines 57–87), but the **image-wrangler-thread catches it** (`image_wrangler_thread.py` line ~):
     ```python
     "compiled Filter predicate; a malformed expression warns once per
     expression and falls back to matching NOTHING."
     ```
   - **Caller expectation:** Line (approx) in `image_wrangler_thread.py` — warns once per bad expression, then **matches nothing** (conservative fallback)
   - **Headless state:** Same behavior — no files matched, process hangs or produces empty output

5. **Raw Embed Consent (F4, outside-project save)**
   - **Status:** NOT YET IMPLEMENTED (MIGRATION.md deferred item F2/F4)
   - **Will have:** Silent graceful degradation when user has not consented to embed raw outside project root

**Current Loud Paths:**
- `ReductionSession.finish()` lines 1160–1164: **fail-loud** re-raises exceptions to caller
- `validate_integrated_stack_write()` / `_require_uniform_axes_*()` in `nexus.py`: **strict validators** that raise on malformed axes
- Monitor warnings (S8): warnings are emitted, but degradation is silent

**Design Default-Loud Policy:**

1. **Thumbnail fallback:** Add `allow_thumbnail` parameter (already exists in `get_raw_frame`), **default to `False`** for headless/batch, **`True` only in GUI** (with a checkbox "Allow preview thumbnails")
   - File: `/xrd_tools/io/image_source.py` — update docstring to recommend `allow_thumbnail=False` for strict loading
   - GUI: `display_controllers.py` — pass `allow_thumbnail=True` **with user awareness**

2. **Monitor skip:** Add `strict_monitor` flag to `ReductionSession`; when `True`, **raise on missing monitor** instead of warning + degrading
   - File: `/xrd_tools/reduction/core.py` — new field `strict_monitor: bool = True` (default loud for headless)
   - GUI: `xdart` reduction adapter — pass `strict_monitor=False` to allow live runs with degraded monitors

3. **GI all-dummy rows:** Add `allow_all_dummy_frames` flag; when `False`, **raise** instead of silently dropping
   - File: `/xdart/modules/ewald/nexus_writer.py` — `_validate_prepared_integrated()` branches on flag
   - GUI: pass `allow_all_dummy_frames=True` (graceful, user expects blank cakes sometimes)
   - Headless: `allow_all_dummy_frames=False` (strict, user should have fixed the incidence angle)

4. **Filter malform:** Already loud (ValueError raised); catch + warn in GUI, let raise in headless
   - File: `/xdart/gui/tabs/static_scan/wranglers/image_wrangler_thread.py` — wrap `compile_filter()` in try/except, warn + skip
   - **Status:** Already done (lines ~)

5. **Raw embed consent (F4):** Define policy when implemented
   - Headless: raise "outside-project raw not embedded; set embed_raw=True or move sources"
   - GUI: popup "Embed raw?" with consent checkbox

**What is PARTIALLY done:** Monitor warnings exist; thumbnail fallback is parameterized but defaults to graceful; GI row drops are silent.

**What is NOT started:** Strictness flags on `ReductionSession`; unified "fail-loud-by-default" policy.

---

## CONCRETE WORK ITEMS

### **NAMING: Mechanical Cleanup (No Architecture Change)**

**S1.1: Alias module docstring + `live_compat` clarification**
- **Files:** `/xdart/modules/ewald/__init__.py`, `/xdart/modules/live_compat.py`
- **Size:** S
- **Work:** Update `__init__.py` docstring to clarify that `ewald/` is the home module, `live` is the public facade; ensure `live_compat.py` handles all edge cases for old provenance strings. Add unit tests for round-trip provenance normalization.
- **Prerequisites:** None
- **Risks:** None (already working)
- **Acceptance:** `grep "EwaldArch\|EwaldSphere" /src/xdart --exclude-dir=.git` returns zero live code matches; only in comments/docstrings/test fixtures

**S1.2: Clarify "two Scan classes" in API docs**
- **Files:** `/xrd_tools/core/scan.py` line 1–50 (add docstring), `/xrd_tools/io/read.py` line 606 docstring
- **Size:** S
- **Work:** Add to each class's docstring: "This is the **reduction input** contract" vs. "This is the **file browser** handle; prefer `core.Scan` for reduction code." Add an FAQ section to CLAUDE.md.
- **Prerequisites:** None
- **Risks:** None
- **Acceptance:** Both docstrings explicitly distinguish their roles; CLAUDE.md has a "Name Resolution" section

---

### **XARRAY: Keep As-Is (No Change)**

**S2.1: Document the xarray boundary**
- **Files:** `/xrd_tools/io/nexus.py` docstring, CLAUDE.md
- **Size:** S
- **Work:** Add to `read_scan()` docstring: "xarray is used ONLY for the round-trip file API; reduction and display use native containers (`IntegrationResult1D/2D`) for performance."
- **Prerequisites:** None
- **Risks:** None
- **Acceptance:** Docstring exists and is clear

---

### **COALESCING: Unified Throttle Utility + Site Audit**

**S3.1: Create shared throttle utility**
- **Files:** `/xdart/utils/throttle.py` (new)
- **Size:** M
- **Work:** Implement `Throttle` class (sketch above); add unit tests for debounce semantics
- **Prerequisites:** None
- **Risks:** None (new code)
- **Acceptance:** Tests pass; h5viewer can be refactored to use it

**S3.2: Refactor h5viewer._absorb_chunk to use Throttle**
- **Files:** `/xdart/gui/tabs/static_scan/h5viewer.py` lines 648–650, 2402
- **Size:** S
- **Work:** Replace `_update_coalesce_timer` with `Throttle(100)` instance; update `_absorb_chunk` line 2402 to call `self._update_throttle.input()`
- **Prerequisites:** S3.1
- **Risks:** Low; semantics unchanged, only refactored to named utility
- **Acceptance:** Tests pass; `_absorb_chunk` calls the utility

**S3.3: Document timer semantics (staggered vs. throttle)**
- **Files:** `/xdart/gui/tabs/static_scan/static_scan_widget.py` line 368 comments
- **Size:** S
- **Work:** Add explicit comment: "Lines 389–391 are intentional staggered one-shots for layout (not coalescing); each independent."
- **Prerequisites:** None
- **Risks:** None
- **Acceptance:** Comment is clear and prevents future confusion

**S3.4: Wire or remove unused _update_timer**
- **Files:** `/xdart/gui/tabs/static_scan/static_scan_widget.py` line 482
- **Size:** S
- **Work:** Either (a) wire to a periodic refresh callback, OR (b) delete and document why (if never used)
- **Prerequisites:** None
- **Risks:** Low; usage audit is straightforward
- **Acceptance:** Timer is either connected and tested, or deleted with a comment

---

### **SENTINELS: Standardize to None**

**S4.1: Consolidate energy/wavelength readers to None (not NaN)**
- **Files:** `/xrd_tools/io/nexus.py` lines 2038–2062
- **Size:** M
- **Work:** Modify `_read_energy()` and `_read_wavelength()` to return `None` instead of `float(np.nan)` when missing. Update docstrings. Check all callers (`read_scan()` line 142–143, `get_metadata()` line 540) — they already expect `None` per the contract.
- **Prerequisites:** Audit all downstream consumers
- **Risks:** **BEHAVIOR CHANGE** — any code that does `np.isnan(energy)` will now fail; grep for this pattern first
- **Acceptance:** All `np.isnan()` calls replaced with `value is None`; equivalence spine test still passes

**S4.2: Make wavelength sentinel only internal to xdart**
- **Files:** `/xdart/modules/wavelength.py`, `/xdart/modules/reduction.py`
- **Size:** S
- **Work:** Keep the 1.0 Å sentinel but only use it in xdart's `LiveScan` constructor default and the normalization helpers; never leak it to `xrd_tools.core` or `xrd_tools.io`
- **Prerequisites:** None
- **Risks:** None (already largely compartmentalized)
- **Acceptance:** `xrd_tools` code has no reference to wavelength sentinels; only xdart does

---

### **STRICTNESS: Introduce Flags + Default-Loud**

**S5.1: Add `strict_mode` flag to ReductionSession**
- **Files:** `/xrd_tools/reduction/core.py` line ~420 (NexusSink), line ~700 (ReductionSession)
- **Size:** M
- **Work:**
  1. Add `strict_mode: bool = True` to `ReductionSession.__init__()` docstring
  2. Pass it down to internal reduction calls (monitor check, GI freeze)
  3. Update monitor-skip path: if `strict_mode and monitor_missing`, raise instead of warn
  4. Update GI-dummy-drop path: if `strict_mode and all_dummy`, raise instead of drop
- **Prerequisites:** None
- **Risks:** **BREAKING for headless users** who relied on graceful degradation — must be documented in MIGRATION.md
- **Acceptance:** Tests pass with `strict_mode=True` and `strict_mode=False`; headless examples use `strict_mode=False` for backward compat during transition

**S5.2: Add `allow_thumbnail` parameter (explicit, default False for headless)**
- **Files:** `/xrd_tools/io/image_source.py` line ~, `/xrd_tools/io/read.py` `get_raw_frame()` (already has it), xdart display path
- **Size:** S
- **Work:**
  1. Ensure `get_raw_frame(allow_thumbnail=False)` is the headless default
  2. xdart display layers call `load_processed_raw_or_thumbnail(..., allow_thumbnail=True, strict_mode=False)` (GUI is graceful by design)
  3. Add docstring note: "Headless / batch users should set `allow_thumbnail=False` to detect missing raw data"
- **Prerequisites:** S1 (naming cleanup done)
- **Risks:** Low; parameter already exists
- **Acceptance:** Headless examples show `allow_thumbnail=False`; GUI does not change

**S5.3: Consolidate filter malform handling (GUI vs. headless)**
- **Files:** `/xrd_tools/core/filters.py`, `/xdart/gui/tabs/static_scan/wranglers/image_wrangler_thread.py`
- **Size:** S
- **Work:**
  1. `compile_filter()` in core: already raises ValueError (loud). Keep as-is.
  2. GUI wrapper in `image_wrangler_thread.py`: catch, warn once per expression, fall back to match-nothing (graceful)
  3. Headless callers: let ValueError bubble (loud) — no wrapper
- **Prerequisites:** None
- **Risks:** Low; behavior unchanged, just explicit paths
- **Acceptance:** Headless test that calls bad filter expression gets ValueError; GUI test shows warning + no matches

**S5.4: Document strictness policy in CLAUDE.md and MIGRATION.md**
- **Files:** `/CLAUDE.md`, `/MIGRATION.md`
- **Size:** S
- **Work:** Add section "Strictness by Default": "Headless `xrd_tools` APIs default to `strict_mode=True`; graceful degradation is a GUI concern (`xdart` opts in). A headless user gets loud errors for missing data, wrong configuration, etc. to catch problems early."
- **Prerequisites:** S5.1–S5.3 complete
- **Risks:** None
- **Acceptance:** Policy is documented and examples follow it

---

## SUMMARY TABLE

| ID | Item | Files | Size | Prerequisites | Risks | Acceptance Gate |
|---|---|---|---|---|---|---|
| S1.1 | Alias module docs + live_compat | `ewald/__init__`, `live_compat.py` | S | None | None | Provenance round-trip tests pass |
| S1.2 | Clarify two Scan classes | `core/scan.py`, `io/read.py`, `CLAUDE.md` | S | None | None | Docstrings explicitly distinguish roles |
| S2.1 | Document xarray boundary | `io/nexus.py`, `CLAUDE.md` | S | None | None | Docstring exists and is clear |
| S3.1 | Create Throttle utility | `utils/throttle.py` | M | None | None | Unit tests pass |
| S3.2 | Refactor h5viewer to use Throttle | `h5viewer.py` | S | S3.1 | Low | Tests pass; semantics unchanged |
| S3.3 | Document timer semantics | `static_scan_widget.py` comments | S | None | None | Comment is explicit |
| S3.4 | Wire/remove _update_timer | `static_scan_widget.py` | S | None | Low | Timer is connected or deleted with comment |
| S4.1 | Consolidate energy/wavelength to None | `io/nexus.py`, all callers | M | Audit consumers | **BEHAVIOR CHANGE** | All `np.isnan()` → `is None` checks; equiv spine green |
| S4.2 | Wavelength sentinel stays xdart-only | `wavelength.py`, `reduction.py` | S | None | None | No xrd_tools code refs sentinels |
| S5.1 | Add strict_mode flag to ReductionSession | `reduction/core.py` | M | None | **BREAKING** (but opt-in) | Tests pass with both flags; examples updated |
| S5.2 | Explicit allow_thumbnail (False default) | `io/read.py`, xdart display | S | S1 | Low | Headless examples show False; GUI shows True |
| S5.3 | Filter malform: GUI vs headless | `core/filters.py`, `image_wrangler_thread.py` | S | None | Low | Both paths tested |
| S5.4 | Document strictness policy | `CLAUDE.md`, `MIGRATION.md` | S | S5.1–S5.3 | None | Policy documented; examples follow it |

---

## STAGING RECOMMENDATION

**Phase 1 (No Breaking Changes):**
- S1.1, S1.2 (naming docs)
- S2.1 (xarray boundary docs)
- S3.1–S3.4 (coalescing utility + refactor, self-contained)
- S4.2 (wavelength sentinel compartmentalization)
- S5.2, S5.3 (allow_thumbnail, filter handling — already working, just explicit)
- **Effort:** ~1 week; **Risk:** Low; **Outcome:** Policy foundation established without breaking existing code

**Phase 2 (Breaking Changes, Post-Tag v1.0.1):**
- S4.1 (energy/wavelength NaN → None conversion; **requires audit + test updates**)
- S5.1 (strict_mode flag; **requires bump to ReductionSession API**)
- **Effort:** ~1.5 weeks; **Risk:** Medium (behavior changes, but defaults are sensible); **Outcome:** Unified "fail-loud by default" policy for all new code
- **Communication:** Mention in release notes and add migration guide for existing headless users relying on graceful degradation

**Live≡Batch≡Reload Equivalence Spine** — the final acceptance gate for all phases:
- All changes must keep `tests/xdart/test_gi_batch_real_data.py::test_*_equivalence` green
- Phase 2 breaking changes **require** the strictness flags to **not change** behavior of the existing GUI path (graceful by design), only add opt-in strict modes for headlessPerfect. Now I have all the information needed. Let me produce the comprehensive gap inventory:

---

## GAP INVENTORY: "Difference 1 residual — versioned docs" (xrd-tools monorepo)

### EXECUTIVE SUMMARY

The migration (Stages 0–7) completed; the monorepo exists at `/Users/vthampy/repos/xrd-tools` with full git history and a unified CLAUDE.md (xrd-tools/CLAUDE.md, 148 lines, Jun 12). The remaining **versioned docs** task is to move living architecture/design docs from the external `~/repos/review/` folder (77 .md files, ~14 KB lines, no git history) into the repo under a canonical `docs/` structure, turning it into the source of truth alongside the code. The `/Users/vthampy/repos/review/` folder itself is NOT a git repository; current repo copy CLAUDE.md + MIGRATION.md are synopses, not full architecture docs.

---

### (A) ALREADY DONE — Living docs already in the monorepo

#### 1. **Root-level foundation** ✓
- `/Users/vthampy/repos/xrd-tools/CLAUDE.md` (148 lines, Jun 12 12:47) — unified working notes for one xrd-tools package (xrd_tools core + xdart GUI), correctly identifies the north star, constraints, package map, publication spine, and generation stamping.
  - Successor to the two-repo CLAUDE.md pair (legacy copies preserved in `docs/legacy/{core,gui}/CLAUDE.md`).
  - **Status:** Complete and authoritative; references external `~/repos/review/roadmap_2026-06-10.md` but that's the seam to be imported.

- `/Users/vthampy/repos/xrd-tools/MIGRATION.md` (132 lines, Jun 12 13:28) — the import story and 1.0 release notes (version floors, renames, on-disk format frozen, behavior changes, deferred items D1–D5 / F1–F5).
  - **Status:** Complete; end-of-migration snapshot.

- `/Users/vthampy/repos/xrd-tools/README.md` (73 lines, Jun 12 12:46) — brief public-facing intro.

#### 2. **Existing docs/ subdirs with content**

| Path | Contents | Status |
|------|----------|--------|
| `docs/core/schema_v2.md` (34 lines) | NeXus layout minimal spec | Incomplete (prose only, needs integration with xrd_tools.io.schema SCHEMA code) |
| `docs/gui/nexus_stitch_refactor_plan.md` (31 KB) | stitching refactor (pre-arch-v2) | Outdated; pre-refactor plan |
| `docs/gui/stitching_design.md` (13 KB) | stitching design | Outdated; post-Stage-6c status unclear |
| `docs/legacy/core/{ARCHITECTURE.md, CLAUDE.md, PUBLISHING.md, README.md, environment.yml, pyproject.toml}` | Pre-migration core docs + config | **Historical** (two-repo era, pre-monorepo) |
| `docs/legacy/gui/{ARCHITECTURE_V2.md, CLAUDE.md, PUBLISHING.md, README.md}` | Pre-migration xdart docs | **Historical** |
| `docs/assets/icons/*.ico` | Icon files | Non-docs |

**Assessment:** `docs/` exists but is a shallow hybrid — `legacy/` is historical (pre-monorepo, no longer authoritative); `core/` and `gui/` have stubs; CLAUDE.md + MIGRATION.md live at the repo root, not in docs/.

---

### (B) PARTIALLY DONE — living docs needing import + clarification

#### 1. **Architecture & Design (CRITICAL — no living docs in repo yet)**

These exist ONLY in `/Users/vthampy/repos/review/` and are actively read/updated by workflow:

| File | Size | Status | Workflow dependency | Import priority |
|------|------|--------|---------------------|-----------------|
| `greenfield_design_2026-06-09.md` | 13 KB | Current; final thought-experiment framing | Referenced by roadmap; static | **HIGH — canonical design frame** |
| `roadmap_2026-06-10.md` | 13 KB | Current (Jun 10); living north star | Referenced in CLAUDE.md line 13; actively used | **CRITICAL — living** |
| `deep_review_2026-06-09.md` | 22 KB | Current; comprehensive audit | Referenced in roadmap; static | **HIGH — audit trail** |
| `CC_preship_sweep_deferred_jun2026.md` | 14 KB | **ACTIVELY READ+WRITTEN by assistant workflows** | Tracks D1–D5, F1–F5; updated Jun 12 12:43 | **BLOCKING — WORKFLOW CANONICAL** |
| `monorepo_plan.md` | 12 KB | Current (Jun 10); migration design | Historical reference; static | **MEDIUM — migration completed** |
| `fix_review_2026-06-10.md` | 13 KB | Current; pre-release fixes | Historical; static | **MEDIUM — closure audit** |
| `CC_monorepo_handoff.md` | 10 KB | Jun 10; next-cycle roadmap | Referenced in MIGRATION.md line 116; active | **HIGH — deferred-items owner** |

**Issue:** The deferred-items document (`CC_preship_sweep_deferred_jun2026.md`) is the **single canonical authority** for D1–D5 / F1–F5 design, actively maintained by assistant workflows. Importing it into the repo means:
- It becomes the canonical in-tree version (good).
- But workflows currently READ AND WRITE `~/repos/review/` directly.
- Moving it to `xrd-tools/docs/` **requires updating all workflow calls** to write to the new path (see workflow change below).

#### 2. **Frame publication & display architecture**

| File | Size | Status | Import priority |
|------|------|--------|-----------------|
| `frame_publication_spine_and_stage5_review.md` | 7.6 KB | Historical (Stage 5); supersceded | **LOW — historical** |
| `frame_publication_status_and_next_steps.md` | 6.4 KB | Historical (Stage 5 closeout) | **LOW — historical** |
| `unified_frame_publication_plan.md` | 6.3 KB | Historical; plan → architecture-v2 merged | **LOW — superseded** |
| `CC_item9_sole_display_contract_design_note.md` | 4.5 KB | Partial (X1 in progress) | **MEDIUM — ongoing work** |

**Status:** Frame publication layer is documented in CLAUDE.md lines 131–147 (frame publication spine) and lines 109–129 (display layer). The stage-gate reviews (frame_publication_stage2–5) are historical closure, not living docs.

#### 3. **Headless + session architecture (greenfield Difference 2)**

No living doc in repo yet. Review-side:
- `CC_arch_v2_direction_review_jun2026.md` (18 KB) — pre-merge audit
- `arch_v2_remaining_jun2026.md` (13 KB) — remaining work

**Status:** Covered in CLAUDE.md § "Boundary at data ownership" implicitly (reduction spine as 2/3 of xrd-session). Greenfield Difference 2 scopes the full 3-layer design (xrd-core / xrd-session / xdart). **Needs a living design doc** or a section in the roadmap.

#### 4. **Schema as code (greenfield Difference 5) — PARTIALLY DONE**

| Location | Status |
|----------|--------|
| `docs/core/schema_v2.md` | Stub (34 lines); purely prose |
| `xrd_tools/io/schema.py` | **THE CODE** — SCHEMA object exists (from Stage 6b) |
| `xrd_tools/io/nexus_record.py` | Stamp/source-ref assembly (from Stage 6a) |
| Tests: `tests/core/test_schema_as_code.py` | Pins the schema contract |

**Status:** Schema-as-code starter landed (Stage 6b commit `7141623`). But there's **no architectural living doc** explaining the schema versioning story, capability flags, or the reader-side version-check (C1 from roadmap, "deferred to next cycle"). The doc at `docs/core/schema_v2.md` is a placeholder; needs expansion + integration with the code.

#### 5. **N1 portable paths & project folder design**

| Location | Size | Status |
|----------|------|--------|
| `design_project_root_paths_jun2026.md` | 7.3 KB | Detailed design (Jun 8); static | 
| Repo docs | — | Covered in CLAUDE.md "N1 portable raw paths" § but no dedicated design doc in `docs/` |

**Status:** Design is in review/; code is in xrd_tools.io.read (relative_source_path, source_root override). **Needs a living doc** in `docs/` (high-level design + reader resolution order).

---

### (C) NOT STARTED — docs infrastructure gaps

#### 1. **Living docs directory structure**

Current `docs/` is shallow and mixed:
```
docs/
├── legacy/              # Historical (two-repo era)
│   ├── core/           # ssrl_xrd_tools pre-monorepo docs
│   └── gui/            # xdart pre-monorepo docs
├── core/               # 1 file: schema_v2.md (stub)
├── gui/                # 2 files: stitching pre-refactor plans
└── assets/icons/       # Non-docs
```

**Needed structure** (for living + searchable docs):
```
docs/
├── ARCHITECTURE.md          # North-star + layers (xrd_tools, xdart, seams)
├── SCHEMA.md                # V2 schema versioning, capability flags, readers
├── DEFERRED_ITEMS.md        # D1–D5 / F1–F5 (imported + linked to code)
├── ROADMAP.md               # Living; replaces ~/repos/review/roadmap_*
├── design/
│   ├── greenfield_design_frame.md     # Thought experiment framing
│   ├── frame_publication_spine.md     # Publication layer architecture
│   ├── display_logic_architecture.md  # Pure display decision core
│   ├── n1_portable_paths.md           # Project folder + @source_base
│   ├── headless_session_service.md    # Greenfield Difference 2 (future)
│   └── schema_as_code.md              # Versioning + capability flags
├── history/
│   ├── DEEP_REVIEW_2026-06-09.md      # 22 KB audit trail
│   ├── CROSS_REPO_REVIEW_JUN2026.md   # Historical
│   └── [stage reviews 1–7]/           # Migration stage closure docs
└── legacy/                            # Two-repo era (archive)
    ├── core/
    └── gui/
```

**Status:** Not started. The `/docs/` tree is flat with no `design/` or `history/` subdirs.

#### 2. **Version and capability tracking**

No living doc yet for:
- NeXus schema version `2` subversions (2.0, 2.1, etc. capability flags).
- Reader-side version checks (C1 from roadmap — not yet implemented).
- Format evolution policy.

Needed: A `docs/SCHEMA.md` that lives with the code and is updated whenever `xrd_tools/io/schema.py` changes.

#### 3. **Headless + session layer architecture (greenfield Difference 2)**

This is scoped but not yet designed. Greenfield doc frames it (§ Difference 2, lines 48–75); current code has ReductionSession + QtNexusSink. But no living design doc in the repo explains:
- The three-layer vision (xrd-core / xrd-session / xdart).
- What moves next (nexus_writer orchestration into core).
- How live≡batch≡reload becomes structural.

**Status:** Blocked on monorepo design-refinement cycle (post-1.0). Placeholder or "Deferred" section needed.

#### 4. **Index / nav structure**

No central index or nav tree. Docs are discoverable only by filesystem browsing.

**Status:** Not started. A top-level `docs/README.md` or `docs/INDEX.md` would help.

---

### (D) WORK ITEMS — staged implementation plan

#### **Item 1: Create the docs directory structure (S)**

**Files/symbols touched:**
- Create `docs/design/`, `docs/history/`, restructure `docs/legacy/ → docs/archive/`
- Create `docs/README.md` (index/nav)
- Reorganize existing content

**Size:** S (4–6 hours)

**Prerequisites:**
- None; can be done in parallel.

**Risks:**
- Breaking symlinks or file references if any exist (check).
- Docs live outside version control (review/ not a git repo) — import mechanically first, then move.

**Acceptance gate:**
- All .md files organized under the new structure; no warnings in CI/docs build.

---

#### **Item 2: Import CRITICAL living docs from review/ (M)**

**Files/symbols touched:**
- Import `roadmap_2026-06-10.md` → `docs/ROADMAP.md`
- Import `greenfield_design_2026-06-09.md` → `docs/design/greenfield_design_frame.md`
- Import `CC_monorepo_handoff.md` → `docs/DEFERRED_ITEMS.md` (or `docs/deferred/`) with cross-refs to code

**Size:** M (8–12 hours)

**Prerequisites:**
- Item 1 (directory structure).
- Workflow change (see below).

**Risks:**
- **BLOCKING: `CC_preship_sweep_deferred_jun2026.md` is actively READ+WRITTEN by assistant workflows.** Importing it without updating workflow calls causes the workflow to write to the old path and docs to diverge.
- Stale cross-references in review/ docs pointing to files/folders that don't exist in the monorepo.

**Workflow change required:**
- Assistant workflows (the "CC_preship_sweep" update loop) currently write to `~/repos/review/CC_preship_sweep_deferred_jun2026.md`.
- **After import:** workflows must write to `xrd-tools/docs/DEFERRED_ITEMS.md` (or a canonical location in the repo).
- **Implementation:** Update the workflow prompt / tool calls to use the new path as the canonical source of truth.
- **Flag this in the acceptance gate:** "Confirm workflow target path has been updated."

**Acceptance gate:**
- All critical docs imported and readable in `docs/`.
- Workflow calls verified to write to new in-tree location (not ~/repos/review/).
- Cross-references between docs and code (esp. Deferred Items to xrd_tools.io.nexus_record, xrd_tools.io.schema) validated.

---

#### **Item 3: Import historical architecture & deep reviews (M)**

**Files/symbols touched:**
- `deep_review_2026-06-09.md` → `docs/history/deep_review_2026-06-09.md`
- `cross_repo_review_jun2026.md` → `docs/history/`
- Stage-gate closure docs (frame_publication_*, restructure_*, stabilization_*) → `docs/history/stages/`
- Archive older reviews under `docs/archive/` (pre-2026, Apr–May obsolete reviews)

**Size:** M (6–10 hours; mostly mv + filtering)

**Prerequisites:**
- Items 1 & 2.

**Risks:**
- Large volume (77 .md, ~14 KB lines); bulk import can obscure later selective changes.
- Pre-2026 docs (CROSS_REPO_REVIEW.md, apr2026 nexus_refactor) are archaeological; risk of confusion if not clearly marked historical.

**Acceptance gate:**
- All history/ docs tagged with date + status (e.g., "Closure audit, Jun 9" vs "Pre-monorepo archaeology").
- No dead links between history docs; stale refs to two-repo paths noted/fixed.

---

#### **Item 4: Schema-as-code living doc (M)**

**Files/symbols touched:**
- Create `docs/SCHEMA.md` (or `docs/design/schema_as_code.md`)
- Link to `xrd_tools/io/schema.py:SCHEMA` definition
- Document versioning policy, capability flags, additive evolution
- Integrate `docs/core/schema_v2.md` findings

**Size:** M (8–10 hours; requires reading schema.py + writing explanatory narrative)

**Prerequisites:**
- Item 1 (structure).
- Knowledge of the SCHEMA object (Stage 6b commit `7141623`).

**Risks:**
- **Schema versioning (reader-side checks) not yet implemented** (C1 in roadmap, deferred). Doc must be clear on status: "current practice" vs "future".
- Schema is code; doc can drift from reality. Need a test or policy to keep them in sync.

**Acceptance gate:**
- `docs/SCHEMA.md` explains: version `2` definition, capability flags (if any), reader resolution order (source_root > @source_base > scan dir), additive policy.
- Cross-refs to `tests/core/test_schema_as_code.py` and `tests/core/test_v2_record_compat.py`.
- Marked sections for future C1 reader-side version checks.

---

#### **Item 5: N1 portable paths design doc (S)**

**Files/symbols touched:**
- Create `docs/design/n1_portable_paths.md`
- Integrate `design_project_root_paths_jun2026.md` + CLAUDE.md "N1 portable raw paths" § 

**Size:** S (4–6 hours)

**Prerequisites:**
- Item 1 & some of Item 3 (context from historical docs).

**Risks:**
- Cross-project path semantics are subtle (relative inside project, absolute outside, @source_base stamp). Doc must be precise.

**Acceptance gate:**
- Doc explains resolution order, @source_base semantics, relative-path portability, warnings for outside-project paths, future F2 embed-raw design.

---

#### **Item 6: Deferred items & post-release feature tracking (S)**

**Files/symbols touched:**
- Create `docs/DEFERRED_ITEMS.md` (imported from CC_preship_sweep_deferred_jun2026.md).
- Link sections to relevant code locations (D1 in reduction.py, D2 in display caches, F1 compile_filter, etc.).
- Create `docs/POST_RELEASE_FEATURES.md` or merge into ROADMAP.

**Size:** S (4–6 hours)

**Prerequisites:**
- Item 2 (workflow change verified).

**Risks:**
- **CRITICAL: This doc is actively maintained by workflows.** Ensure the canonical location is unambiguous; mark sections as "workflow-maintained" if they are.
- Deferred items are keyed to specific code; cross-refs can rot.

**Acceptance gate:**
- All D1–D5 / F1–F5 items mapped to code locations (file:line or module path).
- Workflow writes to in-tree path; no external ~/repos/review write conflicts.

---

#### **Item 7: Headless + session architecture design doc (M) — DEFER OR PROVISIONAL**

**Files/symbols touched:**
- Create `docs/design/headless_session_service.md` (greenfield Difference 2, futures path).
- Or add a "Deferred Design" section to ROADMAP.

**Size:** M (8–10 hours; requires design input)

**Prerequisites:**
- Items 1–5 (context).

**Risks:**
- This is a *future* design, not implemented yet. Doc would be speculative; risk of mismatch if design changes.
- Should be collaborative with maintainer to avoid prescriptive guesses.

**Acceptance gate:**
- If included: mark as "proposed design for next cycle"; link to greenfield_design_frame.md for context.
- OR defer to next cycle (post-1.0) and add a placeholder in ROADMAP.

**Recommendation:** DEFER this to the next design cycle (post-1.0). Include a reference in ROADMAP § "Deferred to next cycle" → Difference 2.

---

#### **Item 8: Top-level index & nav (S)**

**Files/symbols touched:**
- Create `docs/README.md` or `docs/INDEX.md` (navigation hub)
- Link to all main sections (ARCHITECTURE, SCHEMA, ROADMAP, DEFERRED_ITEMS, design/*, history/*)

**Size:** S (2–3 hours)

**Prerequisites:**
- Items 1–6 completed.

**Risks:**
- Doc becomes stale if structure changes; needs a policy for updates.

**Acceptance gate:**
- All main docs linked and navigable from the top-level index.

---

### SUMMARY TABLE

| Item | Work | Size | Prerequisites | Risks | Gate |
|------|------|------|---|---|---|
| 1 | Directory structure | S | None | Symlinks; file refs | Clean hierarchy |
| 2 | Import CRITICAL docs (roadmap, deferred) | M | 1 | **Workflow path change REQUIRED** | Workflow call updated; no divergence |
| 3 | Import historical & stage closures | M | 1,2 | Volume; archaeology confusion | History marked; no dead links |
| 4 | Schema-as-code doc | M | 1 | Version checks not impl'd; drift risk | Links to code; C1 noted |
| 5 | N1 portable paths design | S | 1,3 | Subtle semantics | Resolution order clear |
| 6 | Deferred items tracking | S | 2 | **Workflow lock-in** | Mapping to code; workflow canonical |
| 7 | Headless session architecture | M | 1–5 | Speculative; design input needed | **RECOMMEND DEFER to post-1.0** |
| 8 | Top-level index | S | 1–6 | Stale refs | All docs linked |

**Total effort estimate:** ~50–70 hours (Items 1–6, 8). Item 7 deferred.

---

### REMAINING EXECUTION NOTES

#### **Mechanics: importing docs from non-git review/ folder**

The `~/repos/review/` is NOT a git repository. Importing means:
1. Copy files into xrd-tools/docs/ with new paths.
2. Update internal cross-references (review/file.md → docs/file.md or docs/design/file.md).
3. Update workflow calls to write to the in-tree path.
4. Optionally: Add a `docs/SOURCE_ATTRIBUTION.md` noting the import date and that review/ is archived (no further updates outside the repo).

**Git history:** The review/ docs have no git history (not a git repo). The import does NOT preserve history. If history preservation is desired, it would require a separate import operation (e.g., git subtree from a hypothetical review repo branch), but that's beyond scope.

#### **Workflow change: deferred-items canonical location**

**Current state:**
- Workflows READ + WRITE `~/repos/review/CC_preship_sweep_deferred_jun2026.md` directly.
- This is the source of truth for D1–D5 / F1–F5 tracking.

**After import:**
- Workflows must be updated to write to `xrd-tools/docs/DEFERRED_ITEMS.md`.
- The workflow prompt must include instructions like: "Update the deferred items doc at `xrd-tools/docs/DEFERRED_ITEMS.md` (not `~/repos/review/`)."
- This requires changing the tool calls or environment that feeds workflows.

**Implementation:**
- Coordinate with the workflow owner to update the instructions / tool calls before importing.
- Flag in Item 2 acceptance gate: "Workflow target path verified and updated."

#### **Persistence & generation stamping guardrails (from CLAUDE.md)**

The imported docs must not affect the spine:
- **live≡batch≡reload equivalence** (`tests/xdart/test_gi_batch_real_data.py`) — unaffected by docs location.
- **Persist-before-evict** (xdart/modules/ewald/ invariant) — unaffected by docs location.
- **Generation stamping** (xdart/gui/tabs/static_scan/) — unaffected by docs location.

Importing docs is **safe and orthogonal** to these guardrails.

---

### FINAL RECOMMENDATION

**Stages this task (Item 1–6, 8):**

**Stage 1 (immediate, S):**
- Item 1: Create directory structure.
- Item 5: N1 portable paths design doc.
- Item 8: Top-level index.
- **Effort:** ~10–12 hours; **Risk:** Low.

**Stage 2 (immediate, high priority):**
- **Item 2: Import critical docs + WORKFLOW CHANGE.**
  - Roadmap, greenfield design, monorepo handoff.
  - **Prerequisite:** Workflow owner confirms new canonical path (xrd-tools/docs/DEFERRED_ITEMS.md) and updates tool calls.
- **Effort:** ~8–12 hours; **Risk:** MEDIUM (workflow coordination).

**Stage 3 (immediate, medium priority):**
- Item 3: Import historical & stage closures.
- Item 4: Schema-as-code doc.
- Item 6: Deferred items tracking (after workflow change).
- **Effort:** ~20–28 hours; **Risk:** MEDIUM (stale refs; schema drift).

**Stage 4 (defer to post-1.0):**
- Item 7: Headless session architecture (greenfield Difference 2 design).

---

**ONE KEY FINDING: The workflow change is BLOCKING.** Until the deferred-items doc canonical location moves from `~/repos/review/` to `xrd-tools/docs/` and workflows are updated, importing will result in two versions diverging. **Coordinate with the workflow owner before proceeding with Item 2.**