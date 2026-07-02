# Orchestration — the remainder: greenfield store collapse + controls-panel-v2 + liveness residue

**Date:** 2026-07-01 · **Status:** PROPOSED (synthesized from a 24-agent deep review: 12 doc readers,
6 code auditors, 6 per-goal adversarial critics; run against INT @ `30ecf58a`, since advanced to
`292eb14c`). Needs maintainer sign-off on the AMENDMENTS below before build — the tranche sequence
is executable as written once signed off.

**Supersession scope:** this doc orders the REMAINING work across `design_store_session_steps7_8`
(greenfield), `design_controls_panel_v2` (§14 post-v2 list), `design_gui_liveness_jul2026` (step 5 +
follow-ups), and `design_headless_contracts_migration_jul2026` (Stages 6–8). It does not restate
their content; per-step detail stays in those docs.

---

## 1. Where things stand (verified against code, not just doc ledgers)

- **Branches:** `main == feature/controls-panel-v2 == f2e99a4a` (v1.0.0 staged, UNTAGGED, and main
  is 1 ahead of origin/main — unpushed). `feature/remediation` (INT) = main + 30 commits: all of
  gui-liveness (patch-id proves `59d1f8f1 ≡ 30ecf58a`), headless-contracts Stages 1–5, Phase-5
  A-Steps A/B/C, and the 2026-07-01 evening freeze fixes. `feature/gui-liveness` has NOTHING unique.
  `feature/remediation-floor` is an ancestor of main (75 behind) — stale, but its worktree holds an
  UNCOMMITTED scan_session.py ADR-0005 docstring fix + MIGRATION.md D7 note worth salvaging.
- **Greenfield (steps 7+8):** Phase A steps 1–2 done (7b aggregation-from-disk `91df4096`+,
  `get_or_hydrate` thumbnail fix `8712302e`); Phase-5 A-Steps A (`a12c8273`), B (`ca61215b`),
  C (`0b6ebb4e`) done offscreen. **Remaining: W, 8a, 8b, 7c** + the A-Step-C live checkpoint.
  The greenfield done-test currently FAILS by ~76 `data_1d/data_2d` references across the display
  layer (mirrors still allocated at `static_scan_widget.py:402-403`, bounded 512/40).
- **Controls-panel-v2:** Int flip shipped, default-on, native-authoritative; only flip gate open is
  the **human live checkpoint** (rerun with the four post-flip fixes). Remaining: Stage 6 rewire,
  Phase 3 (Source card), Phase 5 (native Stitch/RSM pages), Phase 4 completion (Experiment
  authority), Phase 8 (retire legacy), F-3, §13 polish.
- **Headless contracts:** Stages 1–5 DONE on remediation only (readiness core → `xrd_tools.session.
  readiness`, purity-guarded). The headless builders (`describe_source_readiness`,
  `capabilities_for_processed`) have **zero GUI consumers** — `_controls_v2_state` still computes
  caps inline (`static_scan_widget.py:2466-2560`).
- **Liveness:** steps 1–3 done + live-validated (flush=150/list=60); step 4 skipped; step 5
  (render leg, 40–68 ms) proposed with no design; follow-ups open (config-panel stash-restore,
  viewer-mode sync I/O). The 21:15 `c89fa406` debounced the normal-mode blocking disk load —
  verify its scope against the I2(a) list below before re-doing.

## 2. AMENDMENTS (the review's plan changes — need sign-off)

**A1 (BLOCKING, S) — mark_persisted must be mode-scoped before 8a.** `QtNexusSink.flush` marks whole
labels persisted (`qt_nexus_sink.py:343-345`) and `FrameRecordStore.mark_persisted` expands to ALL
mode keys (`frame_record_store.py:222-238`), but the production writer emits only the active mode
(`nexus_record.py:212-230`, the W gap). Persist-before-evict is therefore a **false guarantee for
non-primary GI modes**: eviction thins heavy arrays that exist nowhere on disk, unrecoverable by the
hydrator. Latent today; becomes active silent data loss the moment 8a re-points accumulation at the
session store. Fix: flush passes the mode keys actually written; test that an unwritten heavy mode
blocks thinning (bounded memory pressure) or fails loud. This alone makes 8a safe even if W slips.

**A2 (M) — re-sequence W BEFORE 8a** (the steps7_8 doc demotes W to "durability, off the critical
path" — that framing predates A1's finding). W is headless-only, offscreen-provable, closes the
Round-11 in-memory-only gap, and must precede native Stitch/RSM persistence anyway. Gate: a
**production-path** multi-mode spine case (GI run ≥2 sub-modes → flush → reload → per-(frame,mode)
equivalence) in `test_gi_batch_real_data.py` — the existing `test_multimode_record_roundtrip.py` is
pure-io and never sees the production writer. Wire BOTH the GUI writer and headless NexusSink
through the same `mode_subgroup_name` path.

**A3 (M) — typed store-read result before 8a.** The empty-non-blocking-read regression class has
fired twice (`59d1f8f1`/`30ecf58a` overlay-clear), and the fix duplicates preserve-vs-clear policy
in two layers (`display_logic.py:895-917` and `:1254-1268`). Introduce
`RESIDENT | EVICTED_HYDRATING | ABSENT` (or one mandatory choke-point helper) as the FIRST commit of
8a, plus an offscreen contract test enumerating every consumer. 8a/8b deletes the mirror fallback
that currently masks this class.

**A4 (S/M) — memory items steps7_8 never had** (Phase 5 exists to deliver bounded memory; as written
it doesn't):
- Bound the FrameRecordStore thin tail: `max_items=None` + per-frame Axis copies
  (`frame_record_store.py:60-70,137`, `frame_view.py:43-48`, live path passes only
  `max_heavy_items` at `image_wrangler_thread.py:1982`) ⇒ store is O(n_frames), ~340–700 MB at 10k
  frames **after** 8b. Intern/share per-scan axis arrays or strip axes in `_thin_view`; set
  `max_items`. Headless file, NOT contested — can land now.
- 8a spec: PublicationStore-as-projection must set a bound or stop storing independently
  (`frame_publication.py:617` max_items=None otherwise ports the unbounded tail forward).
- Waterfall accumulator: geometric preallocation + row cap/decimation
  (`display_logic.py:477-479` is O(N) vstack per tick, O(N²) per scan). Bundle with liveness step 5.
- Memory-plateau acceptance gate: synthetic ~5k-frame offscreen run through
  ScanSession + QtNexusSink stub asserting bounded store sizes + KB/frame slope budget — in place
  BEFORE 8a/8b (the steps that claim to improve exactly this property).
- Close the `_set_1d_cache_limit(None)`→unbounded footgun now (`static_scan_widget.py:405-419`,
  `_utils.py:271-288`).
- (rides on 7c) optional byte-based heavy budget (`max_heavy_bytes` from detector frame size,
  mirroring stitch's `max_stack_bytes`): the 64/64/40 caps encode an 18 MB/frame assumption; an
  Eiger 16M multiplies the ~2.5–4.5 GB plateau ~4–8× silently.

**A5 (S) — N-2 lands before W.** `h5sig.py` pins shape/dtype/value/attrs only; W is the next
intentional format-adjacent change. Pin compression/chunks/maxshape and re-pin fixtures ONCE, so the
byte-gate isn't blind exactly when the writer changes.

**A6 (M) — liveness "Step 6" (the unowned freeze class).** Promote into the liveness doc + roadmap:
(a) I2(a) viewer-mode off-thread loads + coalesced emits — image (`h5viewer.py:2253-2272`), XYE
(O(N²) sync re-read on selection, `:1288-1330`), NeXus preview (`:2280-2282`) — reusing the
`_LoadFramesWorker`/generation idiom (CHECK `c89fa406` scope first);
(b) floor `XDART_FLUSH_MS` at the 100 ms selection debounce or convert the terminal emit to
throttle — the doc's own sweep example (80 ms) starves the normal-mode render
(`throttle.py:13-17`, `h5viewer.py:731,2344`); sync stale comments (`static_scan_widget.py:3695`,
`h5viewer.py:2340`);
(c) `_teardown_load_worker` non-blocking retirement (GUI-thread `wait(2000)` per re-selection,
`h5viewer.py:2774-2795`), keep blocking wait for shutdown only;
(d) `_get_wavelength` per-scan cache incl. negative result; skip the fallback-3 h5py open while
`_run_writing` (`display_data.py:1156-1169` — contends with the writer lock, the errno-35 class);
(e) memoize/offload the scan-boundary `update_scans` directory rebuild (`h5viewer.py:800-828`);
(f) config-panel stash-restore (step-1 follow-up).
Step 5 itself: instrument the render legs (copy/levels/setImage), then reuse levels across flushes
unless the percentile population shifts; subsample the nanpercentile (`image_widget.py:208-272`).

**A7 (S) — extensibility seams while the files are already open:**
- ToolDescriptor/PageDescriptor table as the FIRST commit of CP2 Phase 5 (the scattered
  `state.tool ==` branches in `readiness.py:1510-1922` get touched for Stitch/RSM anyway;
  acceptance: a synthetic tool = one descriptor + one page class, no other core edits).
- Extract the xdart dotted-path entry points out of headless `readiness.py:1587-1663` into the
  GUI-side AnalysisContext binding — before Phase 6 enables Strain/Texture (layering inversion the
  purity guard can't see).
- Source-registry seam tests (`register_source` has ZERO callers/tests; `registry.py:59-86`) + a
  SourceKind extension-policy note (Bluesky cannot even register today) — before any Tiled work.
- ViewerModeHandler protocol rides INSIDE 8a/8b (same files being rewritten; roadmap's RSM/NeXus
  viewers otherwise shotgun-edit h5viewer a 4th time).
- 7c ADR decision (recommit with milestone, or formally descope) before any second sink.

**A8 (S) — ledger/doc hygiene (cheap, corrects what every future agent reads):**
- **Re-home the deferred ledger:** `CC_preship_sweep_deferred_jun2026.md` is GITIGNORED
  (`xrd-tools/.gitignore:24`), exists only on disk in the MAIN worktree, yet three tracked docs cite
  it as canonical. Commit as `docs/design/deferred_ledger.md`; fix citations
  (followup:306, gi_panel_move:74, azimuthal:268).
- Flip S7 → DONE in roadmap (implemented+tested: `reduction/core.py:671-699`,
  `test_reduction_streaming.py:644-680`); salvage FLOOR's stranded docstring/D7 edits; refresh
  ARCHITECTURE.md (4f-bridge built; A-Steps A–C landed); supersession banner reconciling followup
  §0.4 vs steps7_8 Phase A (port §0.4's unique blockers: update_plot None-payload, C1 wrangler
  read-back, reset_key-not-generation trap, ImagePayload immutability); add ADR-0006 STEP 2 to the
  roadmap; Mask-File validation note into MIGRATION.md before the tag; M4-chunked-replace-save as an
  explicit precondition inside D1.
- Stage-6 parity test NOW: `_controls_v2_state` caps ≡ `describe_source_readiness` over a fixture
  matrix — freezes the two run-gating truth sources together for the deferral window.
- Surface `record_store.upsert` failures (`scan_session.py:430-436` swallows them) in the readiness
  row before 8b removes the mirror that hides blank frames.

**A9 (M) — consolidated live-checkpoint checklist.** Six-plus live gates are scattered across eight
docs for ONE scarce maintainer resource. Two sessions total:
**Session 1 (release):** panel-v2 §14 flip re-pass + A-Step-C store items (scroll-back to evicted
frame, overlay preserve, >64-frame run) + N2 batch cadence + serial-XYE + Share-Axis/ROI visual +
the 21:15 freeze fixes → tag v1.0.0 → merge remediation→main → retire gui-liveness + floor.
**Session 2 (post-W/8a/8b):** 8a scroll-back latency + 8b greenfield done-test + 7c + CP2 Phase 8
retirement + ADR-0006 STEP 2 (if beamline time allows).

**A10 (post-8b, next cycle) — the thin-GUI mass plan** (today: xdart 49.9k LOC > core 40.0k; the
contracts plan moved ~1–2k of ~25k misplaced LOC):
Stage 5.5 MOVE `display_logic.py` (1,659 LOC, already Qt/h5py/pyFAI-free by its own guardrail) →
`xrd_tools` with shim + purity guard — same recipe as Stage 1, offscreen-provable (sequence after
8b to avoid double-churning files 8a/8b rewrite); nexus_writer schema-ownership convergence onto
shared `xrd_tools.io` code (AFTER W — don't converge a moving surface); ewald live model
(scan/frame/frame_series, 2,703 LOC) → session two-type shape (L, post-8b); placement-ratchet test
(xdart/gui h5py/pyFAI import whitelist, shrink-only); `xrd_tools/gui` disposition note
(keep-as-extra vs extract).

## 3. Tranche sequence (the executable order)

**T0 — hygiene (now; no live gate, no contested files):** push main; release step-5 fresh-checkout
check; A8 ledger/doc commits; A5 N-2 re-pin; A9 checklist doc; A4 footgun + plateau gate + thin-tail
bound; A1 mark_persisted fix; Stage-6 parity test. All offscreen-provable; most are S.

**T1 — Session 1 live checkpoint → tag v1.0.0 → merge remediation→main → retire gui-liveness/floor.**
One integration line afterward. (Everything below happens on it.)

**T2 — W** (per A2, with the production multi-mode spine gate + post-N-2 single fixture re-pin).
In parallel (file-disjoint): liveness step 5 instrumentation + A6 items not touching
static_scan_widget; CP2 Phase 5 ToolDescriptor refactor + native Stitch/RSM pages (need W for
persistence); source-registry tests + entry-point extraction.

**T3 — 8a flip** (opening commit = A3 typed read result; PublicationStore projection WITH bound;
resolve §0.4 blockers; explicit-subset hydrate-all-or-refuse gate) **→ 8b delete** (Role-A only,
keep `_ViewerRows`; done-test = identifiers gone) **→ A→B go/defer checkpoint → 7c** per the ADR
decision (+ byte-budget). ViewerModeHandler seam rides inside 8a/8b. Config stash-restore slots
before 8a or after 8b, never concurrent.

**T4 — post-collapse:** Stage 6 rewire (8b IS the "stable static_scan_widget checkpoint" the
contracts doc gates on) → Stages 7–8 → CP2 Phase 3 Source-card mount (R-2 rides here) → Phase 4
completion → **Session 2 live gate** → CP2 Phase 8 retirement → F-3 + §13 polish → A10 mass plan →
D1 (with M4 precondition) → Tiled/Bluesky on the hardened registry seam.

## 4. Standing rules
- Every step: spine (live≡batch≡reload) + byte-compat green per commit; 8a/8b/7c/Phase-8
  live-gated via the consolidated checklist only (no implicit discharge by unrelated live runs).
- Keep the `controls_logic` shim until A-Step-B's `run_target_readiness_note` import
  (`static_scan_widget.py:83`) is re-pointed at `xrd_tools.session.readiness`; delete shim in that
  same commit; add an import-resolution guard test.
- `static_scan_widget.py` contention: at most ONE of {liveness follow-ups, 8a/8b, Stage 6, Phase 8}
  edits it at a time.
- Refuted findings stay refuted (do not re-fix): per-run wrangler-thread leak, set_wrangler
  accumulation, H5FilePool race, GI-2D freeze hole.
