# xrd-tools — Codebase State

**Updated:** 2026-07-03 (pre-v1.0.0 merge) · **Convention:** this is the ALTITUDE view — health,
direction, features. Update at every release and major merge. Chunk-level execution lives in the
MASTER TABLE (`handoff_chunks_jul2026.md`); bug-level truth in `live_findings_ledger.md`.

## What this codebase is
One monorepo, two packages, five north-star goals (roadmap_2026-06-10):
- **`xrd_tools`** — the headless analysis/IO core: usable from notebooks/scripts/services, zero Qt.
- **`xdart`** — the Qt live-experiment GUI, a shell over the core.
- Goals: headless-first · thin GUI · robustness (fail-loud, live≡batch≡reload) · performance
  (streaming, bounded memory, no freezes) · expandability (additive seams).

## Health dashboard (v1.0.0 candidate)
| Signal | State |
|---|---|
| Core suite | ~1467 passed (grows weekly; includes byte-compat + schema + purity + architecture guards) |
| Offscreen GUI suite | ~1300+ across files; known: run `test_gui_modes_end_to_end.py` with `-p no:faulthandler` (152 ✓); exit 133/139 at teardown is a known PySide artifact — read the `N passed` line |
| Equivalence spine | `test_gi_batch_real_data.py` — 71 incl. production multi-mode; the live≡batch≡reload guarantee |
| Perf baseline | 651-frame Eiger Int-2D live ≈ 25 s at the measured 4-worker knee (worker sweep 2026-07-03: speed saturates at 4; ~1 GB RSS per extra worker). Making that knee the default cap is MEM-3, still in flight. |
| Memory | Peak RSS ≈ 9 GB, FULLY ACCOUNTED (pyFAI ~2 GB + ~4 integrator copies ~3-4 GB + transient high-water; see `review_2026-07-02_memory_load.md` + addendum). Plateaus — not a leak. Steady-state observed ~3.5 GB (macOS compressed-memory accounting differs from RSS). Windows are RAM-aware (16–64 frames, `XDART_HEAVY_WINDOW`) |
| Structural guards | One-store done-test grep-guard · monotonic overlay-accumulator test · memory plateau gate · readiness purity · placement/import guards · MS-1 run-end reconciliation |
| Live tunables | `XDART_FLUSH_MS` (150, floor 110) · `XDART_LIST_MS` (60) · `XDART_PERF=1` per-leg timings · `kill -USR1` all-thread stack dump |

## Architecture state (what an agent must know)
- **One store.** The D3 collapse is COMPLETE: `FrameRecordStore` (headless, session-owned) is the
  sole scan-display source; `data_1d`/`data_2d` mirrors are DELETED with a guard test. Eviction
  is persist-gated with THREE states: persisted / owed (pins memory, never lost) / consciously
  dropped (evictable, never falsely promised).
- **One read contract.** `resolve_frame_data` → typed `RESIDENT | EVICTED_HYDRATING | ABSENT` +
  one policy table (`xrd_tools.session.display_logic` — headless since H22). Hydration is
  purpose-scoped (1D bulk reads for overlay; full only when 2D/raw is needed), generation-
  cancelled, retry-capped.
- **One render authority.** Selection-generation scheduler; completions request
  current-selection repaints, never paint their own frame. Accumulator lifecycle: exactly four
  sanctioned resets (Clear · incompatible grid · real norm change · reintegrate-finish).
- **Lock discipline.** `file_lock` OUTERMOST everywhere; H5 pool inside; every writer holds it
  for the full save; every reader is locked or pool-bracketed; the writer's axis-consistency
  guard is the last-line backstop.
- **Headless keystones.** `session.readiness` (run gating), `session.display_logic` (display
  decisions), `sources.probe/readiness`, `core.metadata` resolvers, `core.staging` (RAM budgets),
  shared per-mode NeXus record builders. GUI consumes; notebooks get the same APIs.
- **Known shape debt** (tracked, post-1.0): xdart LOC still exceeds the core (H24 ewald-model
  migration is the big one); reserved scaffolds `analysis.refinement`, `analysis.texture`,
  `gui.main` intentionally raise.

## Current features (v1.0.0, user-facing)
- **Processing:** Int 1D/2D, Standard + Grazing; streaming live/batch with parallel reduction
  (measured 4-worker knee; default cap pending MEM-3) + single fail-loud writer; Append with instant already-done skip;
  series-average; Reintegrate 1D/2D; per-mode durable GI results (survive reload).
- **Display:** Single / Overlay / Waterfall with CROSS-SCAN comparison (grid-keyed identity),
  PINNED SLICE CUTS (Pin button / Cmd+P — χ-cuts at multiple q per frame; texture workflow),
  auto-waterfall >15 traces, share-axis, live-during-run browsing incl. evicted-frame hydration.
- **Sources & metadata:** TIFF/EDF/CBF series, Eiger master, HDF5/NeXus, SPEC; `-`/`_` frame
  indices; Meta Type **auto** (SSRL `.txt`, QXRD `.tif.metadata`, generic name=value sidecars).
- **Controls V2:** native Int panel, readiness row with named blockers, config-mismatch
  **Yes/No overwrite modal** on Append, data-derived display units (panel toggles never relabel
  loaded data).
- **Shortcuts:** Cmd+R run/pause · Cmd+Shift+C stop · Cmd+O/Cmd+S load/save · Cmd+Shift+A
  append/replace · Cmd+P pin.
- **Robustness UX:** run-end frame reconciliation warnings; zero-frames-processed reasons;
  cross-process file-conflict hints; one-modal error dialog (app stays up).

## In flight (pre-tag)
MEM-3 worker-pool guardrail (honest Cores knob, warn >knee, RAM floor) → RC-FV final
verification → maintainer live Session-1 (closes every fixed-unverified ledger row) → RC-8
merge → tag → PyPI.

## Direction (post-1.0, from the master table — priority order)
1. **7c + ADR decision** (H10): cadence/eviction policy → session; enables second sinks
   (Tiled/zarr) + detector-aware byte budgets.
2. **Controls V2 completion** (H15→H21): ToolDescriptor seam → native Stitch/RSM pages →
   Stage-6 readiness delegation → Source card → Experiment authority → legacy retirement.
3. **Reintegrate-All returns** (H29/D1) with chunked replace-save (no OOM).
4. **Texture lane:** auto-pin χ-cuts at Peak-Fit ring positions (one click from fit to
   pole-figure-style overlay); F-3 GI corrections for Int (H26).
5. **Thin-GUI mass:** ewald live model → session (H24); nexus_writer convergence (H23);
  placement ratchet (H25).
6. **Scheduler unification** (H30, after a live soak) · viewer seam (H11) · registry seams for
   Tiled/Bluesky (H17) · ADR-0006 STEP 2 (H28).
7. **Memory (declined-for-now, priced):** shared read-only pyFAI geometry (−3-4 GB at the knee;
   thread-safety/spine/pyFAI-coupling risks — build only if the 9 GB peak demonstrably hurts);
   native-dtype raw ([5]); `frame.py` npt=10000 default landmine.

## Doc map
`handoff_chunks_jul2026.md` (MASTER TABLE — chunk tracker) · `live_findings_ledger.md` (bug
truth; rows close only on maintainer live verification) · `live_checkpoint_session1_jul2026.md`
(the live gate) · `deferred_ledger.md` (parked items) · `review_2026-07-02_memory_load.md`
(memory model) · plan docs per feature (steps7_8 = store, controls_panel_v2, gui_liveness,
headless_contracts) · ADRs in `docs/decisions/` · handoff briefs in `~/repos/codex_tasks/`
(outside the repo).

## Process (why this worked)
Handoff briefs with file-ownership + gates per chunk · one agent per file-set at a time ·
ledger updates in the same commit · orchestrator reviews every commit with independent gates ·
live verification is the ONLY thing that closes a bug row · design checkpoints need maintainer
sign-off · refuted findings stay refuted.
