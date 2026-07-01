# Phase 4b–4f — The Headless Session Seam (ScanSession)

> **DEFERRED / DESIGN-PHASE (post-flip):** keep this document as the accepted
> design map, but do not start the ScanSession lift until the panel-v2 manual
> live checkpoint and the store-collapse sequencing are cleared.

**Status:** deferred design, ready to implement after the post-flip/store-collapse gates.
**Branch base:** `main` @ `875d46d` (Phase 4a done).
**Scope:** lift xdart's live-acquisition orchestration (save cadence, h5pool bracketing,
run-state, dispatch) onto a headless session seam in `xrd_tools`, ending with a public
`xrd_tools.session.ScanSession` that emits immutable frame events. xdart becomes a thin
event→Qt bridge with no API to own data.

This doc supersedes the gap-inventory's 7-stage "D2 session-layer" sketch (which predates
Phases 1–3) and the generic `CadencedSink` write-buffering wrapper (rejected — §1).

---

## 0. Ground truth (verified against the working tree, not the scout maps)

Three claims that reorder the plan, each verified in code:

1. **The equivalence spine's batch leg is `_dispatch_batch_parallel` (chunked), NOT streaming.**
   `tests/xdart/test_gi_batch_real_data.py::_assert_live_batch_reload_equivalent` (line 400)
   drives three legs: live = `_run_live_single`→`_process_one` (439), batch =
   `_run_batch_parallel`→`_dispatch_batch_parallel` (443/292), reload =
   `_write_publication_reload` (454). Streaming is covered only by the *separate*
   `test_streaming_batch_xye_matches_chunked` (765) and the GI streaming tests (1050+).
   **Consequence:** retiring `_dispatch_batch_parallel` in 4e deletes the spine's own batch
   reference implementation. 4e is therefore gated on **re-pointing the spine onto streaming
   in a standalone green commit FIRST** — this is the single highest-risk reordering vs the
   scout maps, all of which assumed the spine already ran streaming.

2. **The two cadence predicates are NOT byte-identical transcriptions of each other.**
   - Serial: `image_wrangler_thread._save_due` (1502–1523) =
     `force OR _frames_since_save >= LIVE_SAVE_INTERVAL OR unsaved >= max(1, cap-8)`, where
     `unsaved = scan.frames.unsaved_in_memory_count()` (a *separate query*, not the local counter).
   - Streaming: `QtNexusSink._due_to_save` (264–272) = `_since_save >= threshold`, where
     `threshold = max(1, cap-8)` clamped to `min(threshold, LIVE_SAVE_INTERVAL)` only when
     `not batch_mode`. It uses the sink's own `_since_save`, **never** `unsaved_in_memory_count()`.

   They coincide in common runs but differ in their pressure input (live query vs local count)
   and in interval placement (independent branch vs clamp). The unified policy (§1) must take
   **both** `frames_since_flush` and an optional `unsaved_in_memory` and reproduce each caller's
   truth table — this is why a careless "byte-for-byte" merge would be a real bug.

3. **`ReductionSession.is_running` does not exist yet.** Only `is_paused` (1079) plus private
   `_started`/`_finished`/`_cancelled`/`_stream_started` fields. 4d/4f add `is_running`.

4. **The contract harness is ready today.** `tests/core/contracts.py::check_sink_contract`
   (186) already asserts the HDF5 single-writer discipline executably: `write`/`replace` run on
   exactly one writer thread, never the caller (214–217); `worker_process` runs on pool threads
   disjoint from the writer (218–221). This is the 4f acceptance gate, in place now.

Also confirmed load-bearing:
- The h5pool-bracketed serial flush (`pause → file_lock → _save_to_nexus → resume → reset
  _frames_since_save`) is duplicated at **eight** serial sites (717, 797, 870, 884, 1482, 1549,
  1907, 2686) plus once in `QtNexusSink._flush` (274). 4c consolidates the serial copies.
- `LIVE_SAVE_INTERVAL` is a mode-aware property (`wrangler_widget.py` 318): `_LIVE_SAVE_INTERVAL=8`
  (2D), `_LIVE_SAVE_INTERVAL_1D=1000` (1D, `skip_2d`); settable per-instance. A lift must keep
  the property, not freeze a constant.
- The QThread-teardown trap is real and guarded: `_dispatch_batch_streaming` wraps
  `session.submit(...)` in `except BaseException` (1648–1677) that translates a recorded
  writer/sink failure into `command='stop'` so no raise escapes `run()`.
- `sigPaused` is emitted in `_enter_pause`'s `finally` (1500) **after** drain+flush — the
  strict ordering the disk-read-during-pause race depends on.

---

## 1. The cadence seam (decisive: a `FlushPolicy` decision object, not a buffering wrapper)

**Rejected: a generic `CadencedSink(ReductionSink)` write-buffering wrapper** (the gap-inventory
Stage 3 sketch). It is structurally wrong for `QtNexusSink`:

- `QtNexusSink.write()` (qt_nexus_sink.py 81–101) does not buffer writes to coalesce HDF5 I/O.
  It **immediately** hydrates the `LiveFrame`, stashes it into `LiveFrameSeries._in_memory`
  (cap 64) via `scan.add_frame(...)`, and buffers only the XYE *row*. The expensive
  `_save_to_nexus` fires on `_due_to_save()`.
- The flush trigger is **eviction pressure**, not write-amplification:
  `unsaved_in_memory_count() >= cap − margin` exists so `LiveFrameSeries.stash` never has to
  evict an **unsaved** frame (persist-before-evict). A buffering wrapper that held reductions
  back from `write()` would *delay* `_save_to_nexus`, letting the unsaved set exceed the cap —
  directly breaking the invariant it was meant to serve. It would also be a redundant third
  reference pinning every cake.

**Adopted: lift the decision, not the action.** A pure, headless, Qt-free policy answers
"*is a flush due now?*"; the flush *mechanism* (h5pool bracket + `_save_to_nexus` + XYE drain)
stays in the sink, on the writer thread, where the single-writer + bracketing invariants live.

```python
# src/xrd_tools/reduction/cadence.py   (pure: no Qt, no h5py, no numpy)
@dataclass(frozen=True, slots=True)
class FlushPolicy:
    interval: int = 8        # upper bound on save spacing (frames)
    cap: int = 64            # mirrors LiveFrameSeries._in_memory_cap
    margin: int = 8          # forces a flush before the unsaved set hits the cap

    def hard_threshold(self) -> int:
        return max(1, self.cap - self.margin)

    def should_flush(self, *, frames_since_flush: int,
                     unsaved_in_memory: int | None = None,
                     force: bool = False) -> bool:
        # short-circuit: nothing pending
        if frames_since_flush <= 0:
            return False
        if force:
            return True
        # interval branch (upper bound on spacing)
        if frames_since_flush >= self.interval:
            return True
        # pressure branch (hard bound; caller supplies the live unsaved count,
        # or None -> fall back to the local counter, exactly today's _save_due)
        pressure = frames_since_flush if unsaved_in_memory is None else unsaved_in_memory
        return pressure >= self.hard_threshold()
```

This is the **single source of truth** the two divergent predicates collapse into. Each caller
keeps passing what it actually has:

- `_save_due` (serial) → `should_flush(frames_since_flush=self._frames_since_save,
  unsaved_in_memory=scan.frames.unsaved_in_memory_count(), force=force)` with
  `interval=self.LIVE_SAVE_INTERVAL`. Reproduces all three branches exactly.
- `QtNexusSink._due_to_save` (streaming) → `should_flush(frames_since_flush=self._since_save,
  unsaved_in_memory=None)` with `interval = self.LIVE_SAVE_INTERVAL if not batch_mode else <large>`.
  The non-batch clamp becomes `interval=LIVE_SAVE_INTERVAL` (which already wins in live, since
  `8 < cap-8 = 56`); in batch the interval is effectively the cap bound. Net: the streaming
  predicate's `_since_save >= min(cap-8, LIVE_SAVE_INTERVAL_when_live)` is reproduced by the
  interval + pressure branches with `unsaved_in_memory=None`.

The headless core never imports `LiveFrameSeries`; the `unsaved_in_memory=None` fallback is the
pure-headless path (a notebook sink with no in-memory cache uses the count only). This keeps the
policy in `xrd_tools` while the eviction coupling stays an xdart concern injected at the call site.

---

## 2. The `ScanSession` API surface (final signatures)

Two objects: the **headless** `xrd_tools.session.ScanSession` (4f) and the **xdart adapter**
`ScanSessionAdapter` (4c→4f) that wraps it with the Qt/LiveFrameSeries/h5pool concerns.

### 2.1 Headless `xrd_tools.session.ScanSession` (4f)

Constructs + arms a streaming `ReductionSession` over (`plan`, `source`, `sink`); commands in /
immutable events out. The save cadence (`FlushPolicy` / persist-before-evict) is deliberately
**NOT** owned here — it is an **adapter** concern (the GUI owns flush timing; `flush()` is a
contract pass-through). No Qt, no `PublicationStore`, no `LiveFrameSeries`, no h5pool. **(As-built
signatures — verified against `src/xrd_tools/session/scan_session.py`.)**

```python
class ScanSession:
    def __init__(self, plan: ReductionPlan, source: Any, sink: Any = None, *,
                 executor: Any | None = None, inflight_max: int | None = None,
                 gi_freeze_mode: str | None = None,
                 cancel_token: Any | None = None) -> None: ...
                 # arms the streaming ReductionSession at construction
                 # (writer thread starts + sink.begin runs)

    # commands in
    def start(self) -> None: ...                       # idempotent; writer already armed
    def submit(self, frame: Frame, image: np.ndarray | None = None) -> bool: ...
                 # True=accepted, False=DROPPED (cancelled/writer-dead); contract
                 # violations (post-finish, while paused) RAISE
    def pause(self, timeout: float | None = None) -> bool: ...   # delegates to ReductionSession.pause
    def resume(self) -> None: ...
    def stop(self) -> None: ...                        # cooperative cancel (sets cancel_token)
    def finish(self, *, raise_on_failure: bool = True,
               join_timeout: float | None = None) -> ReductionResult: ...
    def flush(self, *, force: bool = False) -> None: ...   # cadence pass-through (adapter owns timing)
    def set_generation(self, generation: int) -> None: ...  # caller-owned stale-render stamp

    # state out (read-only properties)
    @property
    def is_running(self) -> bool: ...
    @property
    def is_paused(self) -> bool: ...                   # delegates to ReductionSession.is_paused
    @property
    def frames_submitted(self) -> int: ...
    @property
    def frames_completed(self) -> int: ...

    # events out (plain callbacks; no Qt); each returns an unsubscribe handle
    def on_frame_completed(self, cb: Callable[[FrameEvent], None]) -> Callable[[], None]: ...
    def on_progress(self, cb: Callable[[ProgressEvent], None]) -> Callable[[], None]: ...
    def on_state_change(self, cb: Callable[[StateChangeEvent], None]) -> Callable[[], None]: ...
```

```python
@dataclass(frozen=True, slots=True)
class FrameEvent:                      # built from the existing immutable FrameReduction
    frame_index: int
    mode_key: Any                      # GI (mode_1d, mode_2d) value tuple, or None (single-result)
    result_1d: IntegrationResult1D | None
    result_2d: IntegrationResult2D | None
    metadata: Mapping[str, Any]        # read-only (MappingProxyType) view of FrameReduction.metadata
    generation: int                    # caller-owned stale-render stamp (param-change only; never pause/resume)
    timestamp: float                   # wall-clock completion (time.time())

@dataclass(frozen=True, slots=True)
class StateChangeEvent:
    is_running: bool
    is_paused: bool

@dataclass(frozen=True, slots=True)
class ProgressEvent:
    submitted: int
    completed: int
    total: int | None
```

**Threading contract (pin in 4f design checkpoint):** `on_frame_completed` fires from the
**writer thread** (inside the writer loop, after `sink.write` succeeds) — the Qt bridge MUST use
a `QueuedConnection` to marshal onto the GUI thread. `on_state_change` fires from the
orchestrating (caller) thread. The publication **validation verdict is NOT here** — it stays a
GUI layer that consumes `FrameEvent` (Phase-3 decision: publications stay single-result, no GI
mode dict).

### 2.2 xdart `ScanSessionAdapter` (4c, becomes a thin bridge in 4f)

Lives in `src/xdart/gui/tabs/static_scan/wranglers/scan_session.py`. Owns the three things that
are liftable-as-pattern but irreducibly xdart-bound: the live `unsaved_in_memory` feed, the
h5pool-bracketed flush *action*, and the Qt-signal bridge.

```python
class ScanSessionAdapter:
    def __init__(self, host, scan, reduction_session, sink, *,
                 flush_policy: FlushPolicy) -> None: ...

    # streaming write path
    def submit(self, live: LiveFrame) -> None:          # sink.register(live); session.submit(frame_from_live_frame(live))
                                                        # owns the except-BaseException → command='stop' translation
    def pause(self, timeout: float | None = None) -> bool:   # session.pause(timeout); on drained -> sink._flush(force=True)
    def resume(self) -> None: ...
    def finish(self, *, join_timeout: float = 60.0): ...

    # serial/watch flush action (consolidates the 8 duplicated h5pool brackets)
    def note_written(self, n: int = 1) -> None: ...
    def flush_serial_tail(self, scan, *, force: bool = False,
                          published_idxs: set[int] | None = None) -> None:
        # if not policy.should_flush(...): return
        # h5pool.pause(scan.data_file); try: with file_lock: scan._save_to_nexus()
        # finally: h5pool.resume(scan.data_file); host._flush_xye_buffer(scan, ...)
        # self._frames_since_save = 0

    @property
    def is_running(self) -> bool: ...      # session.is_running
    @property
    def is_paused(self) -> bool: ...       # session.is_paused
```

The adapter **strictly delegates** pause/resume to the session — it never reimplements drain.
The h5pool bracket stays inside the adapter's `flush_serial_tail` / inside `QtNexusSink._flush`
for streaming; it never moves into the headless `ScanSession`.

---

## 3. Staged order 4b → 4f

Sizes: S ≤ ~80 LOC, M ~80–250, L > 250. Each stage commits with its named gate green; full
suites (`tests/core` + offscreen `tests/xdart`) at each phase boundary. The non-negotiable
spine (`test_gi_batch_real_data.py::test_*_equivalence`) and byte-compat
(`test_v2_record_compat.py`) gates run at **every** commit.

### Stage 4b-1 — Pure `FlushPolicy` in core — **S, risk: none**
- **Files:** new `src/xrd_tools/reduction/cadence.py` (`FlushPolicy`); export from
  `xrd_tools.reduction.__init__`.
- **Change:** the §1 dataclass. Nothing consumes it yet (dead code).
- **Gate:** new `tests/core/test_flush_policy.py` — exhaustive truth table over
  `interval ∈ {8, 1000}`, `frames_since ∈ {0, 1, 7, 8, 55, 56, 57}`,
  `unsaved ∈ {None, 55, 56}`, `force ∈ {T, F}`; golden values copied from today's inlined
  `_save_due` / `_due_to_save` expressions. `tests/core` green.
- **Adversarial check:** purity (no Qt/h5py/numpy import) enforced by an import-guard assertion
  in the test. Nothing wired → spine untouched. **Prereq:** none.

### Stage 4b-2 — `QtNexusSink` consumes the policy (behavior-preserving) — **S, risk: low**
- **Files:** `qt_nexus_sink.py` `_due_to_save` (264–272) → delegates to a `FlushPolicy` built
  from `host.LIVE_SAVE_INTERVAL`, `scan.frames._in_memory_cap`, `_SAVE_BEFORE_EVICT_MARGIN`.
  `_flush` / h5pool bracket / writer thread **untouched** (single-writer + bracket unmoved).
- **Adversarial check (persist-before-evict, R2):** the policy reproduces the same threshold;
  the flush *action* and its timing (inside `write()`, on the writer thread) are unchanged, so
  `_in_memory` still gets a `_save_to_nexus`→`mark_persisted` before the cap. Pinned by the
  Phase-1 monkeypatched-cap threshold test.
- **Gate:** `tests/xdart/test_qt_nexus_sink.py` (incl. the monkeypatched-cap persist-before-evict
  threshold test) + `test_streaming_batch_xye_matches_chunked` (765) + byte-compat. Spine green.
- **Prereq:** 4b-1.

### Stage 4b-3 — Serial/watch path consumes the SAME policy (closes the divergence) — **S, risk: low**
- **Files:** `image_wrangler_thread._save_due` (1502–1523) → delegates to the identical
  `FlushPolicy`, passing `unsaved_in_memory=scan.frames.unsaved_in_memory_count()`. The watch
  loop (864–877) and `_dispatch_batch_serial` (1545) keep calling `_save_due` (now a thin wrapper).
- **Adversarial check (R9 divergence):** a new test asserts both the serial `_save_due` and the
  streaming `_due_to_save` resolve through one `FlushPolicy` class. The §0-#2 input difference
  (serial passes the live `unsaved` count; streaming passes `None`) is *preserved by design* —
  each caller passes what it has; the policy handles both via the `None` fallback.
- **Gate:** new `tests/xdart/test_cadence_unified.py` — drive a >cap serial run with a small
  monkeypatched cap; assert no unsaved frame is ever evicted (persist-before-evict) and the flush
  count matches the streaming path on the same frames. Spine green.
- **Prereq:** 4b-1.

> After 4b: cadence math is unified, headless, and pure-tested. The flush *action* and the
> h5pool bracket are byte-for-byte unchanged. **Fully offscreen-gated; no live checkpoint.**

### Stage 4c-1 — `ScanSessionAdapter` skeleton (streaming) — **M, risk: medium**
- **Files:** new `scan_session.py` (`ScanSessionAdapter`, §2.2 streaming methods).
  `_get_streaming_session` (1681) returns the adapter; `_dispatch_batch_streaming` (1647–1678)
  calls `adapter.submit(live)` (register+submit move inside); `_enter_pause` streaming branch
  (1490–1493) calls `adapter.pause(timeout)`. The `except BaseException`→`command='stop'`
  translation (1648–1677) moves into `adapter.submit` and **still returns cleanly** into the
  dispatch loop.
- **Adversarial checks:**
  - *Single-writer (R3):* the adapter never writes; it delegates to `session.submit` (worker
    pool) and `sink._flush` (writer thread). Still exactly one `_writer_loop`.
  - *QThread-teardown trap (R4):* `adapter.submit` re-raises nothing — it catches the recorded
    failure and returns; the wrangler's dispatch loop sees `command='stop'`. No raise escapes
    `run()`. (`ReductionSession.submit`'s own cancel/writer-dead paths already return cleanly.)
  - *h5pool bracket (R5):* the streaming flush stays inside `QtNexusSink._flush`'s try/finally —
    unmoved.
  - *Pause ordering (R7):* `adapter.pause` = `session.pause(timeout)` (drain) **then** flush
    only on `drained` — the post-drain flush guarantee is preserved; `sigPaused` still fires in
    `_enter_pause`'s `finally`, after the adapter returns.
- **Gate:** `check_sink_contract` against the adapter's sink (single-writer/worker-thread);
  `test_streaming_batch_xye_matches_chunked` + GI streaming tests (1050+); new
  `tests/xdart/test_scan_session_adapter.py` — drive on a duck `host` (no Qt loop): submit N,
  `pause()` returns True with writer idle, h5pool `pause`/`resume` balanced (incl. on exception),
  resume+finish, output frame count == N. Spine green.
- **Prereq:** 4b complete.

### Stage 4c-2 — Adapter owns the serial-tail flush + h5pool bracket — **M, risk: medium**
- **Files:** consolidate the eight duplicated serial flush blocks (717, 797, 870, 884, 1482,
  1549, 1907, 2686) into one `adapter.flush_serial_tail(scan, ...)`; serial dispatch + watch loop
  + `_enter_pause` serial branch call it.
- **Adversarial checks:** *R5* — the `finally: resume` survives the consolidation (one bracket,
  proven symmetric on exception). *Watch sanctity (R8)* — the watch loop keeps calling
  `_process_one`; only its *flush* routes through the adapter; it **never** calls
  `adapter.submit`. *Persist-before-evict* — `flush_serial_tail` calls `_save_to_nexus`→
  `mark_persisted` before resetting `_frames_since_save`, and `should_flush` fires at the
  `cap−margin` bound, so the unsaved set never reaches the cap.
- **Gate:** test asserting the watch loop never calls `adapter.submit` (R8); symmetric-bracket-
  on-exception test (R5); spine + streaming tests.
- **Prereq:** 4c-1.

### Stage 4d — Run-state reads from the session — **S, risk: low-medium**
- **Files:** add `ReductionSession.is_running` in `core.py`
  (`_stream_started and not (_finished or _cancelled)`). In `static_scan_widget.py`, `_run_active`
  (477) and the paused booleans become **reads** of `wrangler.scan_session.is_running/is_paused`
  where an adapter exists; `_enter/_exit_run_state` (1066/1100) and `_on_run_paused/
  _on_run_resuming` (1124/1140) keep their Qt view-side effects (`set_processing_active`,
  `set_run_writing`, control disable). `_run_active` stays as the display-side cache for the
  control-disable path (the reintegrate-via-`integratorThread` path has no adapter), synced from
  session state when present.
- **Adversarial checks:** *R7* — `_on_run_paused` stays wired to the post-drain `sigPaused`
  signal (514), **NOT** to the `is_paused` flag (which flips before drain). The `is_*` reads are
  for *display correctness*, never for the disk-read-guard timing. *Generation stamping* — a test
  asserts a pause/resume cycle does not bump `displayframe.display_generation` (generation bumps
  on param/selection change only).
- **Gate:** existing run-state offscreen tests; new assertion that `_on_run_paused` is reachable
  only after `sigPaused`; pause/resume-doesn't-bump-generation test. Spine green.
  **No live checkpoint** (no acquisition-path change). **Prereq:** 4c.

### Stage 4e — Collapse dispatch paths — **M, risk: HIGH care**
- **Pre-step (separate commit, REQUIRED first):** re-point the spine's `_run_batch_parallel`
  (test 244–292) onto `_dispatch_batch_streaming` so the spine's batch leg runs streaming;
  land it with the spine green on streaming. **Only then** is chunked safe to delete (§0-#1).
- **Files:** delete `_dispatch_batch_parallel` (1731–1898), `_dispatch_batch_parallel_phase2`
  (1900–1940), `_dispatch_batch_serial` (1525–1571) for the live-dispatch case, and the
  `_BATCH_EXECUTION`/`_LIVE_EXECUTION` env flags + `XDART_BATCH_EXECUTION`. `_dispatch_batch`
  router (901–918) simplifies to: batch/live → streaming. **KEEP** the first-frame serial
  bootstrap inside `_dispatch_batch_streaming` (1620–1623, reached when `_cached_integrator is
  None`) and **KEEP** `_process_one` (1942+) + the watch loop (825–892), routing the watch save
  cadence through the adapter (4c-2).
- **Adversarial checks:** *R1* — the spine now references streaming (pre-step); keep
  `test_streaming_batch_xye_matches_chunked` as a streaming-vs-direct-integration check.
  *R8* — watch loop stays serial `_process_one`; explicit test pins it never submits to the pool.
  *QThread trap* — the `except BaseException`→`command='stop'` (now in `adapter.submit`) is the
  only escape path; `run()` stays raise-free.
- **Gate:** full `tests/` (core + xdart) + spine + byte-compat green. **MANUAL LIVE CHECKPOINT**
  (§4). **Prereq:** 4c, 4d, and the spine-re-point commit.

### Stage 4f — Extract public `xrd_tools.session.ScanSession` — **L, risk: HIGH (design checkpoint first)**
- **Pre-step:** write an ADR in `docs/decisions/` resolving the three open questions:
  (1) the writer-thread→Qt `QueuedConnection` threading contract for `on_frame_completed`;
  (2) generation stamped on param-change only, never pause/resume; (3) the publication validation
  verdict stays a GUI layer ON the events (never in `ScanSession`).
- **Files:** new `src/xrd_tools/session/__init__.py` + `scan_session.py` (§2.1 surface, immutable
  `FrameEvent`/`ProgressEvent`/`StateChangeEvent` built from the existing `FrameReduction`).
  `ScanSessionAdapter` becomes a thin subclass/bridge: injects the LiveFrameSeries-backed
  `unsaved_in_memory` into the `FlushPolicy`, owns h5pool bracketing, maps
  `on_frame_completed → sigUpdate` and `on_state_change → sigPaused/sigResuming`. New
  `examples/headless_scan_session.py` proving the no-Qt path.
- **Adversarial checks:** *R4* — the new public surface raises only on caller-contract violations
  (mirrors `ReductionSession.submit`); the xdart bridge's `except BaseException` still translates
  any escape to `command='stop'`. *R3* — `check_sink_contract`/`check_source_contract` (Phase 1)
  run against the `ScanSession`-driven sink. *Single-result GI (Phase-3 decision)* — `FrameEvent`
  carries one `result_1d`/`result_2d`; GI mode-switch keeps the `data_1d`/`data_2d` path.
- **Gate:** new `tests/core/test_session_api.py` (lifecycle, event immutability,
  generation-not-bumped-on-pause); Phase-1 contract tests against `ScanSession`; the no-Qt example
  runs Qt-free in CI; spine + full suites. **MANUAL LIVE CHECKPOINT** (§4). **Prereq:** 4b–4e + ADR.

| Stage | What | Size | Risk | Gate | Live? |
|---|---|---|---|---|---|
| 4b-1 | `FlushPolicy` in core | S | none | `test_flush_policy.py` truth table | no |
| 4b-2 | `QtNexusSink` uses policy | S | low | qt_nexus_sink + streaming-matches-chunked + byte-compat | no |
| 4b-3 | serial/watch uses same policy (kills divergence) | S | low | `test_cadence_unified.py` (>cap, persist-before-evict) | no |
| 4c-1 | `ScanSessionAdapter` (streaming) | M | med | `check_sink_contract` + adapter pause/bracket test | no |
| 4c-2 | adapter owns serial-tail + h5pool bracket | M | med | symmetric-bracket + watch-never-submits | no |
| 4d | run-state reads session | S | low-med | run-state tests + post-drain ordering + no-gen-bump | no |
| 4e | collapse dispatch (re-point spine FIRST) | M | **high** | spine on streaming, then full suite | **YES** |
| 4f | public `xrd_tools.session.ScanSession` | L | **high** | contract tests + no-Qt example + spine | **YES** |

---

## 4. Offscreen-gatable vs Vivek's manual live checkpoints

**Fully offscreen-provable (no live session needed): 4b-1 through 4d.** These are the
high-value, low-residual-risk core — cadence unification (4b) and the adapter (4c) are gated
entirely by existing or small new offscreen tests (truth table, `check_sink_contract`,
monkeypatched-cap persist-before-evict, symmetric-bracket, watch-never-submits, post-drain
ordering). Offscreen-provable invariants: equivalence spine (with 4e's pre-step re-point),
persist-before-evict, single-writer, h5pool bracketing, frozen format, watch-loop sanctity,
cadence-divergence closure, and the *ordering* parts of the pause races.

**Require Vivek's manual live checkpoint: 4e and 4f only.** What no offscreen test substitutes:
the actual QThread teardown under a real Qt event loop (R4); the disk-read-during-pause and
human-Pause-mid-flush races against a real churning `.nxs` (R7/R10); and the end-to-end command
choreography.

**The live test (both 4e and 4f must exercise):**
1. **True-live detector watch** (serial `_process_one`): start, let frames arrive at detector
   rate, confirm in-order low-latency display and a correct incremental `.nxs`.
2. **Streaming batch reintegrate**: run a >64-frame scan through the streaming batch path;
   confirm no frame loss (persist-before-evict under real eviction) and a correct stacked `.nxs`.
3. **Pause → browse → resume**: pause mid-run, confirm the writer is idle and the GUI can browse
   already-saved frames from disk (no read race), resume, confirm the tail flushes and the run
   continues without duplicate/missing frames.
4. **Stop mid-run**: confirm a clean stop, a 'Save FAILED' path does not tear down the QThread,
   and the partial `.nxs` is valid + reloadable.

Acceptance: each produces a correct `.nxs`, and the spine equivalence holds on the batch output.

---

## 5. What defers to Phase 5

- **Full cache deletion (`data_1d`/`data_2d` retirement).** Phase-3 recorded decision:
  publications stay single-result; GI mode-switching keeps the `data_1d`/`data_2d` frame path.
  4f's `FrameEvent` is the bridge — Phase 5 folds `FrameEvent` → `FrameRecord` and retires the
  parallel caches. Lifting them now would mean lifting code Phase 5 deletes.
- **`PublicationStore` / validation envelope into core.** Stays an xdart GUI layer ON the events
  (gap-inventory: intrinsically Qt-coupled). Phase 5's `FrameRecord` collapse is where the
  publication record and the frame event converge.
- **`LiveFrameSeries` eviction into headless core.** **UPDATED by ADR-0005:** the authoritative
  `FrameRecord` store (bounded, persist-before-evict, disk-hydration) moves INTO the session in
  Phase 5 — it does NOT stay in xdart. `FlushPolicy` + the eviction bound follow the store into
  the session (refining ADR-0004 §4); only the h5pool bracket + Qt marshaling + display-only
  projections (thumbnails, raw window) stay GUI-side. A future headless live monitor reuses the
  store, not just the policy.
- **D1 (re-integrate re-expose, replace-aware sink + its RAM fix).** Per the deferred doc, ships
  TOGETHER through the session machinery as the FIRST post-v1 feature-queue item — it does not
  block Phase 5 and Phase 5 does not block it.
- **`ewald/` → `live/` path rename.** Decision 2: path stays; optional ride-along only if 4c/4f
  move those files anyway.

### Phase-5 acceptance criteria (record now, so they aren't afterthoughts)
- **Store ownership = session** (ADR-0005): Phase 5 builds the bounded `FrameRecord` store in
  `xrd_tools.session` and collapses `data_1d`/`data_2d`/`LiveFrameSeries`/`PublicationStore` into
  it (authoritative) + GUI projections (derived) — the one-store move in ONE step on the ADR-0003
  record shape.
- **MULTI-MODE reload equivalence (reviewer note 2 — NEW gate):** the multi-result persistence
  path (per-mode `results_1d`/`results_2d` maps written + read back) is brand new in Phase 5, and
  the spine today only exercises the single active mode. Phase 5 MUST add a spine case: **store ≥2
  GI sub-modes for a frame → reload → assert each mode comes back byte-identical** (per `(frame,
  mode)`, per ADR-0003). Without it the most schema-touching change of the cycle ships without the
  equivalence gate that protects everything else.
- **Two types, not one (reviewer push-back):** keep a transient reduction *handle* (the
  `LiveFrame` successor: integration scratch, lazy raw, fiber-cache) that PRODUCES the immutable
  `FrameRecord` (the event payload = sink unit = reader return). Do not merge them into a
  god-object that is both the value and the working buffer.

---

## 6. Constraint ledger (every hard gate, where it survives)

| Hard gate | Survives via |
|---|---|
| live≡batch≡reload spine | green every commit; 4e re-points the batch leg onto streaming in a **separate green commit before** any deletion (§0-#1) |
| persist-before-evict | `FlushPolicy.should_flush` IS the trigger at `cap−margin`, fed by `unsaved_in_memory_count` (serial) / `_since_save` (streaming); `frame_series.stash` eviction guard untouched; pinned by the monkeypatched-cap >cap test (4b-2/4b-3/4c-2) |
| HDF5 single-writer | exactly one `ReductionSession._writer_loop`; adapter + session delegate, never write; `check_sink_contract` proves it (4c-1, 4f) |
| QThread-teardown trap | the `except BaseException`→`command='stop'` translation moves into `adapter.submit`/the bridge but still RETURNS cleanly into the dispatch loop; `run()` stays raise-free |
| h5pool pause/resume bracketing | moved intact (try/finally) into `adapter.flush_serial_tail` (4c-2) and unmoved in `QtNexusSink._flush` (streaming); never into headless `ScanSession`; symmetric-on-exception test |
| frozen on-disk v2 format | no writer/format change in 4b–4f — only *who decides when to flush* moves, never *what is written*; byte-compat gate every commit |
| GI single-result publication (Phase-3) | `FrameEvent` carries one `result_1d`/`result_2d`; GI mode-switch keeps the `data_1d`/`data_2d` path; no GI dict |

---

## 7. Sequencing summary

`4b-1 → 4b-2 → 4b-3 → 4c-1 → 4c-2 → 4d → [4e pre-step: re-point spine] → 4e → [4f ADR] → 4f`.

The one decision taken now: **lift the cadence *decision* (`FlushPolicy`), not the cadence
*action*** — the generic `CadencedSink` write-buffering wrapper double-buffers what
`LiveFrameSeries` already holds and inverts the eviction-pressure trigger it was meant to serve;
a pure `FlushPolicy.should_flush(...)` the sink consults inside its existing `write()` unifies
the two divergent persist-before-evict predicates while leaving the h5pool-bracketed
`_save_to_nexus` exactly where the single-writer + bracketing invariants require it.
