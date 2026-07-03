# Handoff chunks — release wave + structural remainder

**Date:** 2026-07-01 · **Status:** ACTIVE ledger. Companion to
`orchestration_remainder_jul2026.md` (the sequence + amendment rationale). This doc is the
dispatch surface: each chunk below is a self-contained brief for a hand-off agent.

**Conventions (every chunk):**
- Branch: `feature/remediation` unless stated. One commit (or a short gated series) per chunk.
- Gates per commit: `pytest tests/core -q` green; offscreen xdart chunks relevant to touched files
  (chunked, `QT_QPA_PLATFORM=offscreen`, exit-139 retry is a known flake); spine
  (`test_gi_batch_real_data.py -k equivalence`) + byte-compat when touching writer/reduction paths.
- Each agent updates: (1) this ledger's Status column, (2) the owning design doc's own status
  ledger (the repo convention), in the same commit.
- Do NOT touch `static_scan_widget.py` unless the chunk says so — one chunk at a time owns it.
- Refuted findings (do not "fix"): per-run wrangler-thread leak, set_wrangler accumulation,
  GI-2D freeze hole.

---

## MASTER STATUS TABLE (2026-07-02) — the status authority; agents update THIS table
*(The per-wave tables further below are frozen history; briefs for each ID live below them and in
`codex_tasks/`. Bug-level status lives in `live_findings_ledger.md`. Effort: S≤2h · M≤1day · L=days.)*

### Pre-tag: release wave + fix series
| ID | Chunk | Effort | Risk | Payback | Prereqs | State |
|----|-------|--------|------|---------|---------|-------|
| RC-1 | MIGRATION mask-file + old-package notes | S | none | v1.0 promise kept | — | **DONE** `ddacbc0e` |
| RC-2 | Flush floor + comment sync | S | low | live-checkpoint protection | — | **DONE** `4d2f3957` |
| RC-3 | N-2 h5sig filter/chunk pinning + one re-pin | S | low | byte-gate sees filter drift before W | — | **DONE** `843b9f84` |
| RC-4 | Deferred ledger re-homed + citations | S | none | planning survives git clean | — | **DONE** `ddacbc0e` |
| RC-5 | Packaging: license fix, LICENSE, README, stubs, install scripts | S | low | correct PyPI metadata | — | **DONE** `c0535d21` |
| RC-6 | Doc-ledger truth fixes + FLOOR salvage | S | none | agents read true state | — | **DONE** `ddacbc0e` |
| RC-7 | Session-1 live checklist doc | S | none | one artifact discharges all live gates | — | **DONE** `bec431b1` (session PENDING) |
| Fix series | Freeze/overlay/race/popup/bounce (FS/OV/RN/LD/BR rows) | — | — | see `live_findings_ledger.md` | — | code DONE through Wave 2; live-verify at Session-1 |
| RN-2 | Bulk 1D overlay hydration locks/chunks `get_1d` reads | S | low-med | 3600-frame overlay no longer holds `.nxs` read-only across writer flush retries | H-hyd, H30 | **DONE** this commit; live-scale overlay verification pending |
| WM-1 | Window resize state diagnostic action | S | none | captures main/splitter/cursor/top-level state when resize handles stop appearing | — | **DONE** diagnostic this commit; fix deferred until maintainer capture |
| UI-2 | Fresh-launch χ (c/w) slice toggle crash | S | low | slice controls stay disabled until a cake exists; programmatic no-data toggles no-op instead of unpacking `binned_data=None` | — | fixed-unverified; verify at Session-1 F5-adjacent |
| H1 | Mode-scoped `mark_persisted` | S | low | no silent multi-mode loss on eviction | — | **DONE** `4d3e2b3c` |
| H4 | `_set_1d_cache_limit(None)` guard | S | none | unbounded footgun closed | — | **DONE** `4d2f3957` |
| H7a | Typed read contract + policy table | M | low | one read policy | — | **DONE** `03492440` |
| H7-gate | Entry gate + accumulator lifecycle split (OV-5) | S+M | med | wipe class structurally closed | H7a | **DONE** `5a7736ee`+`031ab9bf` |
| H14 | Waterfall accumulator bounding | S/M | low | O(N²) repaint gone | — | **DONE** `084f3410` |
| MS-1 | Run-end count reconciliation + counter fix + shadow-drop lock | S | low | silent in-flight drop now visible | — | **DONE** `4d3e2b3c` |
| H12 | Liveness step-5 render leg (level-reuse, subsampled percentile) | M | low | last big live-smoothness lever | — | **DONE** `7fb20748` (A6 live-verify pending) |
| H13(a-c) | Timer safety: LD-1 load-debounce cancel, alias fix, reint-timer stop, pool race, non-blocking teardown, XYE cache, **BR-1** browser-select no-clear | S×7 | low-med | wrong-file pollute + last stalls + BR-1 | Wave 1 | **DONE** `b8e62542` (LD-1/BR-1 live-verify pending) |
| H-hyd | Hydration purpose-scoping residuals (full→1d leaks, dedupe key, Sum/Avg churn) | S×4 | low | no 60MB/frame churn loops | H7-gate | **DONE** `cba5995c` |
| RC-FV | Final verification (full suite, strict-tree, build, staging) | S | none | certifies the tag commit | Waves 1-3 | **QUEUED** (`release_final_verification.md`) |
| RC-7s | Maintainer live Session-1 | — | — | closes every fixed-unverified row | RC-FV | **PENDING** (maintainer) |
| RC-8 | Merge → tag v1.0.0 → publish + stubs + branch cleanup | S | low | **v1.0.0 on PyPI** | RC-7s, repo-public decision | **PENDING** (maintainer) |

### Greenfield (store/session collapse) — H6/H8/H9/H22 pulled PRE-tag (2026-07-02); only H10/H11/H29/H24 remain post-tag
| ID | Chunk | Effort | Risk | Payback | Prereqs | State |
|----|-------|--------|------|---------|---------|-------|
| H6 (W) | Per-mode subgroup writes, GUI+headless via one path; production multi-mode spine gate | M | med | Round-11 durability closed; multi-mode seam real | RC-3✓, H1✓ | **DONE** `c0b80b02` |
| H8 (8a) | PublicationStore → bounded projection; delete legacy fallback; explicit-subset gate | M/L | med (was high; H7✓+H1✓+H6✓ de-risked) | triple-store divergence class impossible | H6✓, H7✓, Session-1 §E live gate | **DONE** this commit + `e509094c` (H2/H3 commit-0; Wave-3 retry-counter carry-in) |
| H9 (8b) | Delete Role-A `data_1d/data_2d/hydrated_raw` (~76 refs; keep Role-B) | M | med | **greenfield done-test**; ~0.5-1GB mirrors gone; H7b completes | H8 | **DONE** this commit |
| H10 (7c) | Cadence/eviction policy → session + `max_heavy_bytes` | M | low-med | second-sink recipe; detector-aware caps | H9, ADR-0005 reaffirm | open (ADR decision owed) |
| H2 | Thin-tail axis-interning / max_items | S | low | long-scan memory bounded | — | **DONE** `e509094c` |
| H3 | Memory-plateau acceptance gate | S | low | boundedness regression-detected | — | **DONE** `e509094c` |
| H11 | ViewerModeHandler seam | M | low | new viewers register, not shotgun-edit | rides H8/H9 | open |
| H29 (D1) | Reintegrate-All + chunked replace-save (M4) | M/L | med | top missing feature, no 10GB OOM | H1✓; session machinery | open |
| H24 | ewald live model → session | L | high | thin-GUI mass; "xdart owns no data" true | H9; do last | open |

### Post-tag: controls-panel-v2
| ID | Chunk | Effort | Risk | Payback | Prereqs | State |
|----|-------|--------|------|---------|---------|-------|
| CP-live | Int-flip live re-pass (§14 gate) | — | — | closes the only open flip gate | = Session-1 items B1-B7 | **PENDING** (in RC-7s) |
| H5 | Stage-6 parity test (inline caps ≡ headless readiness) | S | low | freezes two truth sources until H18 | — | open |
| H15 | Phase 5: ToolDescriptor refactor → native Stitch/RSM pages | L | med | Stitch/RSM first-class; tool #4 = one descriptor | H6 (persistence) | open |
| H18 | Stage 6: readiness delegation to headless core | M | low-med | one gating truth source; Tiled prereq | H9 (stable widget), H5 | open |
| H19 | Phase 3: ScanSourceWidget as Source card + R-2 async open | M | med | no probe stalls; real grouped-scan chips | H18 | open |
| H20 | Phase 4: Experiment card authoritative (editors, badges, Refine) | M/L | med | provenance UX; Refine usable | H18 | open |
| H21 | Phase 8: retire ParameterTree + embedded Int + escape hatch | M | med | ~2k LOC gone; sync-bug class ends | H15,H18,H19,H20 + live gate | open |
| H16 | Entry-point extraction from `readiness.py` | S | low | layering fixed; analyses GUI-side | before Strain/Texture | open |
| H26 (F-3) | GI corrections for Int 1D/2D | M | low-med | GIWAXS physics parity | validated notebook exists | open |
| H27 | §13 visual polish (items 1-4, 8-13) | M | low | the mockup look | H21 | open |

### Next cycle: enablers + thin-GUI mass
| ID | Chunk | Effort | Risk | Payback | Prereqs | State |
|----|-------|--------|------|---------|---------|-------|
| H17 | Source-registry seam tests + SourceKind policy | S | low | Tiled/Bluesky additive | — | open |
| H22 | Stage 5.5: `display_logic` → xrd_tools | M | low | 1.9k LOC decision core headless | H9 | **DONE** this commit |
| PF-1/PF-2 | Append pre-read skip + dash-index filenames + zero-processed warning | S | low | completed appends skip raw I/O; APS dash-index series enumerates | — | **DONE + HOTFIXED** this commit (skip snapshot read-only at run start; per-frame path never opens `.nxs`) |
| MD-1 | APS/QXRD name=value sidecar autodetect | M | low | `.tif.metadata` works headless; auto sidecar convention cache | — | **DONE** this commit |
| MD-2 | Meta Type auto default + explicit none/off boundary | S | low | fresh sessions discover sidecars by default; disabled metadata never calls the reader | MD-1 | **DONE** this commit |
| UX-1 | Menu-backed keyboard shortcuts for run/stop/load/save/write-mode toggle | S | low | fast operator loop without bypassing button locks | — | **DONE** this commit |
| GI-1 | Auto χ/χGI full-range determinism for 1D integrations | S | med | Auto writes match explicit `-180..180`; χGI boundary scans stable | — | **DONE** this commit |
| H23 | nexus_writer schema convergence | M/L | med | one on-disk-contract owner | H6 | open |
| H25 | Placement ratchet + xrd_tools/gui disposition | S | low | shell can't re-thicken silently | — | open |
| H28 | ADR-0006 STEP 2 prepass deletion | M | med | dual-prepass risk retired | live gate slot | open |
| H30 | Unify ALL paint triggers under the generation scheduler (live flush `_flush_pending_update`, reintegrate idx==-1 path — completions already unified in 8d35a37c) | M | med | implemented: `staticWidget._request_render` delegates to the display-generation scheduler; stale heavy flushes re-arm without draining/dropping pending frames; fast list/cursor timer remains outside the heavy painter. Hotfix folded in: writer flushes lock before pausing the H5 read pool, and RN-2 bulk 1D hydration now reads via `DisplayDataMixin._locked_scan_read()` in <=256-row chunks. Session-1 A4/F6 + XDART_PERF live bar still certify | pre-tag; live/perf verification pending | fixed-unverified |
| OV-7 | Pinned slice cuts: slice projection identity + Pin action | M | med | texture workflow can overlay multiple χ/q cuts from the same frame; live cut mutates in place while pinned cuts survive norm/BG/unit rebuilds through recipes | OV-6, H30 | fixed-unverified; verify Session-1 texture-cut workflow |
| H13(rest) | Config stash-restore; click-latency shed; reintegrate debounce-stack; update_scans memoize | S ea | med/low | UX niceties | live repro in hand | open |

**Sequencing spine:** RC-FV → Session-1 → RC-8 (tag) → H6 → H8 → H9 (Session-2) → {H10, H18} → {H15, H19, H20} ∥ lanes → H21 (live gate) → next-cycle table.

---

## Wave R — before the v1.0.0 tag (all S unless noted; parallel-safe with each other)

| ID | Chunk | Status |
|----|-------|--------|
| RC-1 | MIGRATION.md Mask-File validation note | open |
| RC-2 | XDART_FLUSH_MS floor + stale-comment sync | open |
| RC-3 | N-2: h5sig pins compression/chunks/maxshape (+ one fixture re-pin) | **DONE** this commit |
| RC-4 | Re-home the gitignored deferred ledger + fix citations | open |
| RC-5 | Packaging/install-script/README fixes (finalized from release audit) | open |
| RC-6 | Doc-ledger hygiene commit | open |
| RC-7 | Consolidated live Session-1 checklist (orchestrator writes; maintainer runs) | open |
| RC-8 | Merge → tag v1.0.0 → publish (maintainer; verified recipe below) | open |

**RC-0 (maintainer, blocks the merge):** resolve the in-flight waterfall-decimation WIP (7 files,
+328 lines incl. `MAX_WF_ROWS=256` + 216 test lines — this is chunk **H14**, being done by the
maintainer directly): land as a final remediation commit or stash. `release.py check --strict-tree`
fails on a dirty tree.

**Pre-tag verification (recorded 2026-07-01, INT @ 292eb14c + WIP):** `release.py check v1.0.0` →
OK; `tests/core` → **1424 passed, 4 skipped** (all 4 environmental); offscreen chunks
(controls_panel_v2 90, live_refresh 199, display_logic+hydration 81) green, zero exit-139.
Wheel+sdist build clean, `twine check` PASS, no junk, headless install verified Qt-free, shim
DeprecationWarning + friendly `xdart` no-gui error verified. Still owed at the FINAL merge commit:
full `tests/xdart` offscreen run (S-3 was recorded green at f2e99a4a, 31 commits back) + one
`--strict-tree` preflight from an env with the release content actually installed (the xrd_test
editable install points at the MAIN worktree — use `PYTHONPATH` override or a scratch venv).

**RC-1** — Add the Mask-File validation behavior release note to `MIGRATION.md` (promised to ship
WITH v1.0, `review_2026-06-15_followup_plan.md` §0.3; currently absent — only Mask Saturated is
mentioned). Source the behavior description from §0.3 and the mask validation code/tests. Doc-only.

**RC-2** — Floor the live flush quantum: `XDART_FLUSH_MS` values below the h5viewer selection
debounce (100 ms) starve the normal-mode render (`h5viewer.py:731` debounce restarted by every
flush, `throttle.py:13-17`; `static_scan_widget.py` `_ms()` floors at 10). Either clamp flush ≥
~110 ms (log a warning when clamped) or convert the terminal emit at `h5viewer.py:2344` to
throttle mode — read `c89fa406` first (it touched this path 2026-07-01 21:15) and reconcile.
Sync stale comments: `static_scan_widget.py:3695-3699` (says 100/70; defaults are 150/60),
`h5viewer.py:2340` (says 200 ms). Add a unit test pinning the floor/no-starvation. Update the
liveness doc knobs section. MUST land before the live session (its sweep protocol suggests 80 ms).

**RC-3** — N-2: extend `tests/core/h5sig.py` signatures with compression/compression_opts/chunks/
maxshape; re-pin byte-compat fixtures ONCE. Rationale: the byte-gate is blind to filter/chunking
drift exactly where W (per-mode subgroup writes) will change the writer next. Gate: full
tests/core green after re-pin; fixture diff reviewed (expected: signature-file-only changes).

**RC-4** — The canonical deferred ledger `CC_preship_sweep_deferred_jun2026.md` is GITIGNORED
(`.gitignore:24` ignores `docs/design/CC_*.md`) and exists only untracked in the MAIN worktree
(`~/repos/xrd-tools`). Copy it into the tracked tree as `docs/design/deferred_ledger.md` (rename
escapes the ignore rule), add a header noting the rename, and fix the three dangling citations:
`review_2026-06-15_followup_plan.md:306`, `design_gi_panel_move_and_2way_sync_jun2026.md:74`,
`azimuthal_1d_profiles_design_2026-06-18.md:268`. Also fix roadmap:74,114 + `docs/history/INDEX.md:
80-81` references to the two deleted CC_* docs (mark dangling or restore from git history).

**RC-5** — Packaging fixes (scope finalized by the 2026-07-01 release audit; all verified at
`292eb14c`):
- **RC-5a (metadata/license):** README.md:710 links a nonexistent top-level `LICENSE` — add a
  canonical top-level MIT `LICENSE` and fix the link. PEP-639 modernization: `license = "MIT AND
  BSD-3-Clause"` (licenses/LICENSE-ssrl_xrd_tools is BSD-3-Clause — current `MIT`-only metadata is
  WRONG), move `license-files` from `[tool.setuptools]` to `[project]`, drop the MIT classifier,
  bump `setuptools>=77`. Add `Changelog` (→ MIGRATION.md) to `[project.urls]`. Make README.md:23's
  MIGRATION.md link absolute so it renders on PyPI. Fix the wrong comment at pyproject.toml:45
  (only pyFAI 2025.3.0 exists in the `>=2025.3,<2025.12` window — "2025.3-2025.11" versions don't
  exist). README.md:689 `git clone <this repo>` placeholder → real URL. Rebuild + `twine check`.
- **RC-5b (old-package migration):** add an "Upgrading from the old `xdart` / `ssrl_xrd_tools`
  PyPI packages" section to README + MIGRATION.md — both legacy packages are the maintainer's own
  (latest 2026-06-12), and `pip install xdart` over an xrd-tools env silently overwrites
  `site-packages/xdart/` (empirically confirmed: uninstalling then leaves a gutted hybrid).
  Instruction: `pip uninstall -y xdart ssrl_xrd_tools` first. Prepare (do NOT upload yet) two stub
  dists in scratch: `xdart 0.41.0` with `dependencies=["xrd-tools[gui]>=1.0.0"]` and
  `ssrl_xrd_tools 0.42.0` with `dependencies=["xrd-tools>=1.0.0"]`, no top-level packages, README
  pointing at xrd-tools. Uploaded post-publish (RC-8), then the old projects get archived (not
  yanked).
- **RC-5c (install scripts):** delete `scripts/install.sh` + `install.ps1` or gut to a 3-line
  `pip install xrd-tools[gui]` pointer — they curl the old ssrl_xrd_tools repo's dev branch and a
  nonexistent environment.yml. Nothing references them; not shipped in artifacts.
- **Maintainer decision embedded here:** the GitHub repo `v-thampy/xrd-tools` is PRIVATE while
  pyproject URLs + README links bake it into the PyPI page — make it public at/before upload, or
  strip the URLs.

**RC-6** — One doc-hygiene commit: flip roadmap S7 → DONE (cite `reduction/core.py:671-699`,
`test_reduction_streaming.py:644-680`); apply FLOOR's stranded edits (`scan_session.py` ADR-0005
docstring + MIGRATION D7 note — salvage from `~/repos/xrd-tools-floor` uncommitted diff, then the
branch dies); refresh `ARCHITECTURE.md` (4f-bridge built at `wranglers/scan_session.py`; Phase-5
A-Steps A–C landed; ADR-0005 "Phase 5 not started" note obsolete); supersession banner on
`review_2026-06-15_followup_plan.md` §0.4 pointing at `design_store_session_steps7_8` Phase A and
porting §0.4's unique blockers (update_plot None-payload, C1 wrangler read-back, reset_key trap,
ImagePayload immutability) into steps7_8; add ADR-0006 STEP 2 to the roadmap next-cycle list; add
the M4-chunked-replace-save precondition line inside the D1 item; record the 7c decision once the
maintainer makes it (recommit-with-milestone vs descope, see orchestration doc A-series).

---

## Wave H — structural remainder (post-tag, on the single merged line)

Dependency lanes (∥ = parallel-safe):

```
Lane A (store spine):   H1 → H6 → H7 → H8 → H9 → {H10, H18, H22, H24}
Lane B (liveness):      {H12+H14} ∥ Lane A;  H13 BEFORE H8 (file contention)
Lane C (controls/ext):  {H15, H16, H17} ∥ Lane A;  H18 → H19 → H20 → H21 (H21 after Session 2)
Lane D (tests/bounds):  {H2, H3, H4, H5} — anytime, no dependencies
```

| ID | Chunk | Size | Depends on | Status |
|----|-------|------|-----------|--------|
| H1 | mark_persisted mode-scoping (A1) | S | — | open |
| H2 | FrameRecordStore thin-tail bound / axis interning (A4) | S/M | — | done `e509094c` |
| H3 | Memory-plateau acceptance gate (A4) | S | — | done `e509094c` |
| H4 | `_set_1d_cache_limit(None)` footgun (A4) | S | — | open |
| H5 | Stage-6 parity test (A8) | S | — | open |
| H6 | W: per-mode subgroup write wiring + production multi-mode spine gate (A2) | M | H1, RC-3 | done (this commit) |
| H7a | Typed read result + policy table + render authority (display-side, fallbacks KEPT) (A3) | M | round-2 fix — **PULLED FORWARD pre-release** after 3 live bugs of this class; `codex_tasks/h7_typed_reads_render_authority.md` | done |
| H7b | Remove the fallback tiers behind the H7a accessor | S | H7a; rides 8a | **DONE** this commit |
| H8 | 8a flip: PublicationStore→bounded projection; remove update_plot fallback; §0.4 blockers; explicit-subset hydrate-all-or-refuse gate | M/L | H6, H7, H13; live-gated (Session 1 §E) | done this commit |
| H9 | 8b delete Role-A `data_1d/data_2d/hydrated_raw` (keep Role-B `_ViewerRows`); greenfield done-test | M | H8 | **DONE** this commit |
| H10 | 7c cadence→session per ADR decision + optional `max_heavy_bytes` budget | M | H9, ADR decision | open |
| H11 | ViewerModeHandler seam (rides inside H8/H9 commits) | M | with H8/H9 | open |
| H12 | Liveness step 5: render-leg instrumentation, then level-reuse/subsampled percentiles (`image_widget.py`) | M | — | open |
| H13 | Liveness step 6 bundle: viewer-mode off-thread loads + coalesced emits (verify `c89fa406` scope first), `_teardown_load_worker` non-blocking retirement, `update_scans` memoize/offload, config-panel stash-restore. *(Item (d) `_get_wavelength` cache moved PRE-release into `codex_tasks/fix_fastselect_sweep_beachball_and_popup.md`.)* | M | before H8 | open |
| H14 | Waterfall accumulator bounding (geometric growth + cap/decimation) | S/M | bundle with H12 | **in progress (maintainer, uncommitted WIP 2026-07-01)** |
| H15 | CP2 Phase 5: ToolDescriptor/PageDescriptor refactor (FIRST commit) + native Stitch/RSM pages | L | H6 (persistence) | open |
| H16 | Extract xdart dotted-path entry points from `readiness.py:1587-1663` → GUI-side AnalysisContext binding | S | before Phase 6 launchers | open |
| H17 | Source-registry seam tests + SourceKind extension-policy note (Bluesky) | S | — | open |
| H18 | Contracts Stage 6: `_controls_v2_state` → `describe_source_readiness` delegation | M | H9 (stable checkpoint), H5 | open |
| H19 | CP2 Phase 3: mount ScanSourceWidget as authoritative Source card (+R-2 async open) | M | H18 | open |
| H20 | CP2 Phase 4 completion: Experiment producers/badges authoritative, native editors | M/L | H18 | open |
| H21 | CP2 Phase 8: retire ParameterTree primary + embedded Int widget + escape hatch + dead keys | M | H15,H18,H19,H20 + Session 2 | open |
| H22 | Stage 5.5: MOVE `display_logic.py` → xrd_tools (shim + purity guard, Stage-1 recipe) | M | H9 | **DONE** this commit |
| PF-1/PF-2 | Append-mode pre-read skips, dash-index filenames, zero-frame warning | S | — | **DONE + HOTFIXED** this commit (read-only run-start snapshot; all-skipped Append reloads/selects output) |
| MD-1 | APS/QXRD `.tif.metadata` structured sidecar parser + auto-detect | M | — | **DONE** this commit |
| MD-2 | Meta Type `auto` default; `none` remains metadata-off | S | MD-1 | **DONE** this commit |
| UX-1 | Keyboard shortcuts (Run/Pause, Stop, Load/Save settings, Append/Replace) | S | — | **DONE** this commit |
| GI-1 | χ/χGI Auto range equals explicit full range for 1D writes | S | — | **DONE** this commit |
| H23 | nexus_writer schema-ownership convergence onto shared xrd_tools.io | M/L | H6 | open |
| H24 | ewald live model (scan/frame/frame_series) → session two-type shape | L | H9 | open |
| H25 | Placement-ratchet test (xdart/gui h5py/pyFAI import whitelist) + xrd_tools/gui disposition note | S | — | open |
| H26 | F-3: GI correction stack for Int 1D/2D (footprint/absorption/Fresnel first; refraction needs histogram path) | M | post-v2 | open |
| H27 | §13 visual polish pass (items 1–4, 8–13) | M | H21 | open |
| H28 | ADR-0006 STEP 2: prepass deletion + `_frame_source_for` factory (live-gated) | M | Session 2 slot | open |
| H29 | D1: re-expose Reintegrate-All WITH M4 chunked replace-save + per-batch mark_persisted | M/L | H1; first post-v1 feature | open |

### Briefs (Wave H)

**H1** — Make persist-marking mode-scoped. `QtNexusSink.flush` (`qt_nexus_sink.py:343-345`) passes
the mode keys `_save_to_nexus` actually wrote; `FrameRecordStore.mark_persisted`
(`frame_record_store.py:222-238`) marks only those. Add tests: a heavy mode never written to disk
must block thinning (`_label_heavy_payload_persisted_locked`, `:331-338`) or fail loud; bounded
memory pressure is the acceptable degraded mode. This converts silent multi-mode data loss on
eviction into fail-safe behavior and de-risks 8a independently of W. Files: headless store +
qt_nexus_sink (contested file — coordinate; small surgical diff).

**H2** — Bound the thin-record tail. Options (pick after measuring): intern/share per-scan axis
arrays (identical across frames; the copy is `frame_view.py:43-48`), strip axes in `_thin_view`
(`frame_record_store.py:60-70`) and let the hydrator restore, and/or pass `max_items` at the live
instantiation (`image_wrangler_thread.py:1981-1982`). Headless file — NOT contested. Acceptance:
thin-tail growth ≤ ~2 KB/frame/mode in the H3 gate.

**H3** — Synthetic long-run plateau gate: offscreen test driving ~5k frames through
`ScanSession` + a stub sink; assert store/publication sizes bounded and per-frame growth slope
below budget (tracemalloc or explicit object-size accounting). Lands in tests/core (headless) +
one xdart offscreen variant.

**H4** — `_set_1d_cache_limit(None)` maps to `_max=0` = never-evict (`static_scan_widget.py:
405-419`, `_utils.py:271-288`). Raise or clamp to a positive floor + test. (Touches
static_scan_widget trivially — schedule in a quiet window.)

**H5** — Parity test: `_controls_v2_state`'s inline SourceCaps/ResultCaps
(`static_scan_widget.py:2466-2560`) ≡ `describe_source_readiness`/`capabilities_for_processed`
(`xrd_tools/sources/readiness.py:39,97`) over a fixture matrix (SPEC, Eiger, processed±reachable
raw, live/unknown-length, unreachable). Read-only test; freezes the two truth sources together
until H18. Known reconcile point: the inline `has_frames=has_raw=raw_reachable=source_ready`
simplification vs the headless true-live escape hatch — the test documents which wins where.

**H6 — DONE this commit** — W: production writer emits the accumulated record's per-mode subgroups. Wire BOTH the GUI
writer (`nexus_writer.py`) and headless `NexusSink` through the same `mode_subgroup_name` path
(`nexus_record.py:212-230` currently metadata+thumbnail only). Gate: NEW production-path spine
case in `test_gi_batch_real_data.py` — GI run persisting ≥2 sub-modes → flush → reload →
per-(frame,mode) equivalence (the pure-io `test_multimode_record_roundtrip.py` stays but does not
count). Byte-compat: additive groups only; fixture re-pin rides the RC-3 pinning (single re-pin).
Closes the Round-11 in-memory-only durability gap.

**H7** — Typed store-read result `RESIDENT | EVICTED_HYDRATING | ABSENT` (or one mandatory
choke-point helper). Single home for preserve-vs-clear policy (today duplicated:
`display_logic.py:895-917` + `:1254-1268` — the twice-fired overlay-clear class). Offscreen
contract test enumerating every consumer of empty non-blocking reads. First commit of the 8a
series; changes the read-chain signature 8a consumes.

**H8/H9/H10** — Execute per `design_store_session_steps7_8_jun2026.md` Phase A steps (4)-(5) + 7c,
as amended: 8a resolves §0.4's ported blockers and specifies the PublicationStore projection
BOUND; 8b's done-test = `data_1d/data_2d/hydrated_raw` identifiers gone from the display layer
(Role-B `_ViewerRows` stays); A→B go/defer checkpoint after 8b; 7c only after the ADR-0005
amendment records the decision. Live gates in Session 2. H11's ViewerModeHandler protocol replaces
the stringly viewer-mode `if/elif` (`h5viewer.py:816-822,1239-1245,2253-2282`) inside these same
commits — acceptance: a new viewer mode registers a handler without editing existing branch sites.

**H12–H14** — Per `design_gui_liveness_jul2026.md` step 5 + the audit list; H13 MUST verify what
`c89fa406` (2026-07-01 21:15 "Freeze fix part 2") already covers before implementing. All H13
sub-items carry their own [PERF] before/after measurement per the doc's protocol; the ≤5%/≥50%
tradeoff bar applies.

**H15–H21** — Per `design_controls_panel_v2_jun2026.md` §8 Phases 3/4/5/8 + §14 post-v2 list, as
amended (descriptor table first; entry-point extraction before Phase 6). H21 (retirement) is the
LAST structural chunk and needs its own live gate.

**H22–H25** — The thin-GUI mass workstream (orchestration doc A10). H22 uses the exact Stage-1
recipe: MOVE + re-export shim + purity-guard test, byte-identical behavior.

---

## Live sessions (maintainer; the scarce resource)

**Session 1 (pre-tag):** panel-v2 §14 flip re-pass · A-Step-C store items (scroll-back to evicted
frame, overlay preserve, >64-frame Overall) · N2 batch cadence · serial-XYE flush · Share-Axis +
ROI §10 visual · the 2026-07-01 freeze fixes (`c89fa406`, `5c10e190`) + waterfall decimation (H14)
· RC-2 flush-floor sanity. PASS → RC-8.

**RC-8 verified recipe** (dry-run-verified in a scratch clone; merge is conflict-free, main is a
strict ancestor of remediation, 61 files +5864/−2521; GitHub has ONLY `main` @ `a2211ef4` and zero
tags; worktrees pin their branches so remove worktrees before `branch -d`):
```
# 0) salvage FLOOR docs (also in RC-6): git -C ~/repos/xrd-tools-floor diff > /tmp/floor.patch
#    apply+commit in INT; then checkout -- the floor files
# 1) optional tie-off (makes branch -d honest; tree-identical, verified):
git -C ~/repos/xrd-tools-integrate merge --no-ff feature/gui-liveness -m "Tie off gui-liveness"
# 2) release merge + tag (MAIN worktree; main not checked out anywhere, same tip as cp2):
git -C ~/repos/xrd-tools checkout main
git -C ~/repos/xrd-tools merge --no-ff feature/remediation -m "Merge feature/remediation: v1.0.0"
#    → final gates HERE: full offscreen tests/xdart (pr.yml exit-139-retry pattern) +
#      release.py check v1.0.0 --strict-tree (after tagging) from a venv with this content installed
git -C ~/repos/xrd-tools tag -a v1.0.0 -m "xrd-tools v1.0.0"
git -C ~/repos/xrd-tools push origin main --follow-tags     # release.yml builds + checks on tag
# 3) publish (manual upload is the designed path): python -m build && twine check dist/* &&
#    twine upload dist/*   (TestPyPI dry-run first if desired)
# 4) post-publish: upload the RC-5b stub dists (xdart 0.41.0, ssrl_xrd_tools 0.42.0), archive the
#    old PyPI projects; cleanup: git worktree remove xrd-tools-{integrate,gui,floor} →
#    branch -d feature/{remediation,remediation-floor,controls-panel-v2,gui-liveness} →
#    git fetch --prune
```

**Session 2 (post-H9):** 8a scroll-back latency budget · 8b done-test spot checks · 7c ·
Phase 8 retirement gate · ADR-0006 STEP 2 confirm (if beamline time allows).
