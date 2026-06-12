# Greenfield-design implementation plan — pre-v1 cycle

**Date:** 2026-06-12 · Executes the remaining differences from
`greenfield_design_2026-06-09.md` against the completed monorepo
(`~/repos/xrd-tools`, main @ `f4ddb6d`).  Evidence base: the six gap
inventories in `CC_greenfield_gap_inventories_2026-06-12.md` (file:line
citations live there; this plan stays at the decision level).

**Vivek's directive (Jun 12):** implement all the greenfield design changes
BEFORE shipping v1; features (F3 ROI etc.) come after.

## Where each Difference stands today

| # | Difference | Status after the 1.0 migration |
|---|---|---|
| 1 | Monorepo + shared CI | **DONE** beyond the doc's "approximation" (full monorepo, one dist, CI). Residual: `review/` docs are still unversioned outside the repo → Phase 0 |
| 2 | `xrd-session` layer (boundary at data ownership) | **~2/3 by construction**: ReductionSession+QtNexusSink own streaming orchestration; 6a moved record assembly into core. Remaining: pause/cadence/run-state/eviction still in xdart; no headless session object → Phase 4 |
| 3 | One FrameRecord, one store | **Half-converged**: X2 (parallel `LiveFrame.integrate_*`) is already complete — the scout verified one canonical path; X1 (publications as sole contract) still has triple-source reads (display_controllers/display_data/metadata merge data_1d + data_2d + store) → Phase 3 |
| 4 | Working store (zarr) + NeXus export | **DEFERRED by design** — the doc itself says revisit with Tiled/Bluesky. Not in this cycle. |
| 5 | Schema as code | **Starter shipped (6b)**: SCHEMA + key consumers. Remaining: writer/reader/validator/fixture derivation + capability flags → Phase 2 |
| 6 | CI + contract tests | **CI done**. Remaining: sink/source contract-test module (incl. which-thread-calls-which-hook), release pre-flight script → Phase 1 |
| 7 | Day-one policies | **Mixed**: energy None-sentinel done at io.read; naming done at the public surface (`xdart.modules.live` facade; `ProcessedScan`). Remaining: throttle idiom, strictness flags, sentinel compartmentalization, docs → Phases 1 & 5. **xarray-as-currency: rejected** (see Decisions) |

## Decisions taken now (stop revisiting)

1. **xarray stays at the read boundary.** The scout costed "reduction emits
   `xr.Dataset` per frame" at ~850 LOC across writer/display/GI-freeze plus
   a 5–10% suite slowdown, for value that only materializes in notebooks —
   which `read_scan`/`open_scan` already serve.  Record as an ADR in Phase 0;
   keep `IntegrationResult1D/2D` on the hot path.
2. **`ewald/` module path stays; `xdart.modules.live` is the public facade.**
   Every public import already goes through the facade; a path rename is
   churn without leverage.  Optional ride-along only if Phase 4 moves those
   files anyway.
3. **NaN stays inside xarray Datasets; `None` at scalar API boundaries.**
   `np.nan` is the idiomatic missing-value inside a float Dataset
   (`read_scan_metadata`); converting those to None would force object
   dtypes.  The #78 contract ("None, never NaN") applies to scalar returns
   (`get_metadata`, `ProcessedScan.energy*`) — already true.  Phase 5 audits
   the remaining scalar consumers; nothing else changes.
4. **Schema versioning: keep integer `2` + per-feature capability attrs.**
   No fractional versions; a reader feature-detects capabilities, the
   integer only bumps on a breaking layout change (ideally never).
5. **Phase 3 (one store) goes BEFORE Phase 4 (session layer).**  The session
   must emit frame events into ONE store; lifting today's cadence/eviction
   machinery first would mean lifting code Phase 3 deletes.

## Working agreements (every phase)

- The **live≡batch≡reload equivalence spine**
  (`tests/xdart/test_gi_batch_real_data.py::test_*_equivalence`) and the
  **byte-compat gate** (`tests/core/test_v2_record_compat.py`) are
  non-negotiable acceptance gates at every commit.  A failing spine is a
  bug; the gate failing means the on-disk format changed — stop.
- Persisted format stays frozen + additive-only; validators stay strict;
  persist-before-evict and generation stamping are invariants.
- Commit per work item with the relevant suite green; full suites
  (`tests/core` + offscreen `tests/xdart`) at every phase boundary.
- No push / no publish / no tags — maintainer only.
- Manual live testing checkpoints (Vivek) at the end of Phases 3, 4, 5 —
  these phases touch the live acquisition path.

---

## Phase 0 — Docs into the repo (Difference 1 residual) — S, ~½ day

The review folder (`~/repos/review/`, 77 files, no git) holds the living
design docs.  Import the living set; leave scratch behind.

1. Create the structure: `docs/ARCHITECTURE.md` (north star + layer map,
   distilled from CLAUDE.md + roadmap), `docs/design/`, `docs/decisions/`
   (ADRs), `docs/history/`.
2. Import living docs into `docs/design/`: `roadmap_2026-06-10.md`,
   `greenfield_design_2026-06-09.md`, `deep_review_2026-06-09.md`,
   `CC_preship_sweep_deferred_jun2026.md` (**becomes canonical in-repo** —
   all future deferred-item updates happen there; the review-folder copy
   gets a tombstone pointer), `design_project_root_paths_jun2026.md` (N1),
   this plan + its gap-inventory appendix.
3. Import migration-era docs into `docs/history/`:
   `CC_monorepo_handoff.md`, `monorepo_plan.md`, `fix_review_2026-06-10.md`.
   (The rest of the review-cycle archaeology can come later or never —
   list it in a docs/history/INDEX.md without importing.)
4. Write the first two ADRs: xarray boundary (Decision 1), schema
   versioning (Decision 4).
5. Update CLAUDE.md/MIGRATION.md pointers from `~/repos/review/...` to
   in-repo paths; refresh the stale stubs (`docs/core/schema_v2.md` points
   at `xrd_tools/io/schema.py` as the source of truth).

*Gate:* every doc link in CLAUDE.md/MIGRATION.md/README resolves in-repo.

## Phase 1 — Contract tests, release script, policy groundwork
*(Differences 6 + 7-phase-1)* — M, ~3–5 days

**1a. Contract-test module** (`tests/core/contracts.py` + tests).
Extract `_SpySink`/`_BoomSink` into reusable, thread-tracking test doubles
(record `threading.get_ident()` per hook).  `check_sink_contract(factory)`
and `check_source_contract(factory)` harnesses, parametrized over
MemorySink/XYESink/NexusSink and Scan/MemoryFrameSource/LiveFrameSource/
ProcessedScan.  New pinned invariants (none are tested today):
`worker_process` runs on a pool worker and never on the writer thread;
`write`/`replace` always on the single writer thread (streaming) — the
HDF5 single-writer discipline as an executable contract;
`capabilities` truthfulness spot-checks.  Extend
`tests/xdart/test_qt_nexus_sink.py`: worker_process under real parallel
execution, replace idempotency, abort path, and the persist-before-evict
flush threshold (monkeypatched cap).  These harnesses are what future
Tiled/zarr sources and sinks will self-verify against.

**1b. Release pre-flight script** (`scripts/release.py` + release.yml step).
Checks: tag ↔ `pyproject.toml` ↔ `xrd_tools.__version__`/`xdart.__version__`
consistency; clean tree; byte-compat gate + schema-pin tests pass; dep caps
(pyFAI `<2025.12`) intact; build + twine check.  No auto-publish.

**1c. Policy groundwork (low-risk D7 items).**
- `xdart/utils/throttle.py`: ONE coalescing utility (trailing-edge
  debounce with `force_flush()`); refactor `h5viewer._update_coalesce_timer`
  onto it; document the staticWidget staggered one-shots as deliberate;
  wire-or-delete the dead `_update_timer`.
- Naming docs: `core.Scan` vs `io.ProcessedScan` role docstrings + a
  CLAUDE.md name-resolution note; `ewald/__init__` facade docstring.
- Wavelength sentinel: assert (test) that the 1.0 Å sentinel never leaks
  out of xdart (`xrd_tools` grep-clean), keep the explicit
  `allow_default_sentinel` flag as the only crossing point.

*Gate:* new contract tests green in CI; full suites green.

## Phase 2 — Schema derivation (Difference 5 completion) — M/L, ~1 week

All additive, byte-compat-gated at every commit.

- **2a. Enrich the schema** (S): `DatasetSpec` (name, dtype, shape template,
  row-aligned, required, compression-eligible, units) + per-group NX attrs
  + lookup methods.  Pure data; gate-neutral.
- **2b. Writer derives from SCHEMA** (M, **byte-compat critical**):
  `_bulk_create_1d/2d`, `write_per_frame_geometry`, `write_stitched`
  iterate the schema instead of hard-coding names/dtypes/chunking.  The
  refactor must be invisible on disk.
- **2c. Readers consume SCHEMA** (M, spine-critical): `get_1d/get_2d/
  read_scan` group/dataset names from schema; `read_scan`'s xarray dim
  assembly must stay output-identical.
- **2d. Validators parameterized by schema** (S): required/optional/dtype
  knowledge queried, never relaxed.
- **2e. Fixture factory** (M): `make_v2_fixture(...)` builds test files
  FROM the schema; refactor `test_nexus_v2.py` hand-built layouts onto it.
  Fixtures can no longer lag the writer.
- **2f. Capability registry + feature detection** (M): per-feature
  capability attrs (`per_frame_geometry`, `frames` record, `source_base`,
  `sigma`, `two_d_kind`) with introduced-version metadata; readers
  feature-detect (presence AND capability).  This finally CONSUMES
  `ACCEPTED_SCHEMA_NAMES` and `THUMBNAIL_LUT_ATTRS` (declared-but-unused
  since 6b).  New-file stamps are additive attrs only — old readers ignore
  them.

*Gate:* byte-compat + spine at every commit; `test_schema_as_code.py`
extended to assert every group/dataset in the writer appears in SCHEMA.

## Phase 3 — One record, one store (Difference 3 / X1 completion) — L, ~1.5–2 wks

X2 is already done (verified: `scan.add_frame` → one `integrate_1d/2d`
path).  This phase finishes X1 and collapses the caches.  Each step is
behavior-preserving with the spine green; the D2-deferred thumbnail LRU
lands here as part of the store, not as a bolt-on.

- **3a. Store-first availability** (S/M): invert the merge in
  `display_controllers` — PublicationStore is queried first, the dict
  caches only fill gaps.  (Today three sources are reconciled per render.)
- **3b. Store owns hydration — this IS the D2 implementation** (M):
  `PublicationStore.get_or_hydrate()` (lazy reload from `scan.frames`/.nxs
  via the readers — background-queue for heavy loads, never blocking h5py
  reads on the GUI thread) + two-level bounds (`max_items` lightweight
  metadata, `max_heavy_items` payloads, with a thumbnail tier kept longer
  than full raw).  The Jun-12 deferral said "don't bolt D2 onto the old
  caches; wait for the publication-store migration" — Phase 3 *is* that
  migration, so D2 lands here as store policy, not as a bolt-on.  Done
  test: scroll-back to a long-evicted frame shows its thumbnail without a
  GUI stall, then rehydrates full raw lazily.
- **3c. Retire `data_1d`** (M): redirect metadata.py / display_controllers /
  display_data reads to the store; delete the `FixSizeOrderedDict(max=0)`.
- **3d. Retire `data_2d` + interim hydrated-raw LRU** (L): fold the
  max=40 / raw-cap-8 discipline into the store's bounds; delete
  `hydrated_raw.py` (the D5 fix dissolves into the one store);
  wranglers publish to the store only.
- **3e. Publish directly** (M): wrangler/load paths build
  `FramePublication` straight from frames (`publication_from_live_frame`),
  delete `LiveFrame.copy_for_display()`; verify live/batch/reload publish
  identically (the spine's publication variant).
- **Done test (the greenfield's own):** grep for "keep both stores in
  sync"-style comments returns nothing; `data_1d`/`data_2d` identifiers
  gone from the display layer.

*Gate per step:* equivalence spine + display snapshot tests; a
100-frame scroll-back test (evicted frame rehydrates without GUI stall).
*Checkpoint:* manual live session (Vivek) before Phase 4 starts.

## Phase 4 — The session layer (Difference 2) — L, ~2–3 weeks

The centerpiece.  End state: a headless `ScanSession` in `xrd_tools`
owning start/pause/resume/stop, the writer, save cadence, and eviction —
emitting immutable frame events; xdart renders events and sends commands,
and **has no API through which to own data**.

- **4a. Pause in the core** (S): `ReductionSession.pause()/resume()/
  is_paused` — pause = drain (configurable timeout) + reject submits;
  exposes the quiesce-at-frame-boundary guarantee the GUI's
  `_enter_pause()` currently hand-rolls around `drain()`.
- **4b. Save cadence in the core** (M): a cadence policy on the sink seam
  (flush every N frames / seconds / explicit), replacing the
  `LIVE_SAVE_INTERVAL` + `_frames_since_save` + `_save_due()` logic in
  `image_wrangler_thread`.  Must preserve the persist-before-evict
  coupling: flush forced when unsaved-in-memory approaches the cap.
- **4c. xdart `ScanSession` adapter** (M): one object lifting save-due,
  h5pool pause/resume bracketing, and the eviction trigger out of the
  wrangler thread; `_dispatch_batch_streaming` constructs it instead of a
  bare ReductionSession.
- **4d. Run-state from the session** (S): staticWidget's
  `_run_active`/paused booleans become reads of `session.is_running/
  is_paused`; `_enter/_exit_run_state` stay as the Qt view-side effects.
- **4e. Collapse dispatch paths** (M, **high care**): retire
  `_dispatch_batch_serial` and `_dispatch_batch_parallel` (streaming is
  the only batch path; the env-var fallbacks have had their one cycle).
  The Phase-3 watch loop (`_process_one`, detector-rate serial) STAYS —
  it is deliberately serial; route its save cadence through the session.
- **4f. Extract `xrd_tools.session.ScanSession`** (L, design checkpoint
  first): commands in (start/submit/pause/resume/stop), immutable frame
  events out (frame_completed/progress/state_change callbacks),
  generation-stamped; xdart provides the event→Qt-signal bridge; a
  notebook example proves the no-Qt path.  The publication envelope
  (validation verdict) stays a GUI layer ON the events.  Contract tests
  from Phase 1 run against it.

*Gates:* spine + full suites per step; 4e and 4f each get a manual live
checkpoint (live serial, streaming batch, pause/browse/resume, stop).
*Note:* D1 (re-integrate re-expose, replace-aware sink) becomes natural
here — per the deferred doc it ships TOGETHER with its RAM fix through the
session machinery.  Schedule it as the FIRST item of the post-v1 feature
queue (it does not block Phase 5 and Phase 5 does not block it; first-in-
queue simply because Phase 4 just built its foundations).

## Phase 5 — FrameRecord + strictness + sentinel closure
*(Differences 3-final + 7-phase-2)* — M/L, ~1 week

- **5a. `FrameRecord`** (M): merge FrameView + source ref + geometry +
  diagnostics into the one immutable record the session emits; the
  publication verdict becomes a field, `FramePublication` thins to (or
  becomes) it; `LiveFrame` reduces toward a command-side handle with
  integration scratchpads.  (The big rename/collapse is mechanical once
  Phases 3+4 own the flow.)
- **5b. Strictness flags, default-loud** (M): `strict` mode on the session
  /readers — missing monitor, GI all-dummy row drops, thumbnail-fallback:
  loud (raise/flagged) for headless callers by default, the GUI explicitly
  opts into graceful degradation.  Never the reverse.  Document the policy.
- **5c. Sentinel audit closure** (S): per Decision 3 — scalar boundaries
  None-clean (test-pinned), NaN documented as Dataset-internal.

*Gate:* spine + suites + manual checkpoint; MIGRATION.md gains the v1
behavior-change notes (strict modes, retired dispatch paths).

---

## Sequencing & effort summary

| Phase | What | Size | Depends on |
|---|---|---|---|
| 0 | Docs into repo + ADRs | ~½ day | — |
| 1 | Contract tests, release script, throttle/policies | 3–5 days | — (parallel with 0) |
| 2 | Schema derivation + capability flags | ~1 week | 0/1 not required, gate-driven |
| 3 | One store (X1 completion, D2-thumbnails) | 1.5–2 weeks | 1 (contract tests helpful) |
| 4 | Session layer | 2–3 weeks | 3 (one store), 1 (contracts) |
| 5 | FrameRecord + strictness + sentinels | ~1 week | 3, 4 |

Total ≈ 6–8 working weeks of focused effort.  Phases 0–2 are independently
shippable and low-risk; 3→4→5 are ordered by dependency.

**If a v1 cut is wanted earlier:** the defensible early cut-line is after
Phase 2 (+1a contract tests) — everything beyond is the data-ownership
re-architecture, which is exactly what Vivek asked to finish first, so the
default is: v1 ships after Phase 5.

**Feature queue after v1 (unchanged priorities):** F3 ROI statistics
(lands cleanly on the session events + one store), D1 re-integrate
re-expose with the replace-aware sink (Phase 4 makes it natural), F4
embed-raw flag + consent popup, F5 Set Bkg in all modes.  Difference 4
(zarr working store) stays parked until Tiled/Bluesky work begins.

## Standing risks to watch

1. **Pause-drain timeout** (Phase 4a): a hung worker means pause times out
   with unflushed frames — keep the timeout configurable + logged; the
   tail flushes on resume/finish (current RS-1 behavior preserved).
2. **Persist-before-evict across the cadence lift** (4b/4c): the flush
   threshold (`cap − margin`) must move WITH the eviction trigger; the
   Phase-1 threshold test pins it.
3. **Store rehydration latency** (3b): scroll-back to an evicted frame
   must lazy-load off the GUI thread; pin with a stall test.
4. **`read_scan` dim assembly** (2c): the spine catches reorder/rename
   regressions — trust it, run it per commit.
5. **Watch-path sanctity** (4e): the detector-rate serial loop is
   intentionally serial; collapsing it into streaming is NOT in scope.
