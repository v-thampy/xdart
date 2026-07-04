# Deep review — feature/remediation pre-release (2026-07-03)

**Scope:** the full `main..feature/remediation` diff (77 commits @ `031c134c`, re-verified where
noted at tip `59f9c6c0` after MEM-3 landed mid-review), reviewed by six parallel adversarial
agents (headless core / memory+store / GUI display+liveness / run-path+wranglers /
controls-panel-v2 / release-readiness+test-run) with top findings re-verified against source by
the orchestrator. Convention severity: BLOCKER / SERIOUS / MINOR / NOTE.

**Verdict: NOT tag-ready yet — but close.** The architecture work (H6/H8/H9/H22, headless
contracts Stages 1–5, MEM-1/2) is demonstrably faithful and well-gated. What blocks the tag is a
short list: two Append data-destruction paths, the auto-sidecar junk latch (worsened by the new
`auto` default), one thread-race crash, a red test suite at tip, and the OV-6 cross-scan misgrid.
Most fixes are small (several are one-liners). Ledger rows CF-1, CF-2, GI-1, PF-2, MS-1, MEM-1b,
OV-6 should NOT be closed as-is.

Cross-confirmation note: four findings were found independently by two agents each (GI mode-switch
range re-key; pool-pause-outside-file_lock; MEM-2 publication-store wiring gap; sidecar junk
latch) — treat those as high-confidence.

---

## BLOCKERS (fix before tag)

### BL-1. Append truncates the whole existing file when the target's metadata read degrades
`image_wrangler_thread.py:3215-3221`: Append does `scan.load_from_h5(replace=False, mode='r')`
then `if len(scan.frames.index) == 0: scan.save_to_nexus(replace=True)`. `_load_from_nexus_v2`
(`scan.py` ~792-802, ~994-999) swallows read failures (`logger.exception; return`) leaving the
index empty — a transient read hiccup on a 651-frame file makes Append conclude "empty file" and
atomically rewrite it as an empty skeleton. Silent destruction of all prior data in the flagship
workflow. **Fix:** distinguish "load failed" from "genuinely frame-less" — probe
`entry/integrated_1d` row count on disk before `save_to_nexus(replace=True)`; abort loudly on
load failure. *(Verified by orchestrator.)*

### BL-2. First flush of an Append run silently full-rewrites the file
`qt_nexus_sink.py:316-337` (`_needs_atomic_first_batch_flush`): predicate is `not persisted`, but
an Append-loaded scan has `frames.index` populated while `LiveFrameSeries._persisted` stays empty
(lazy read marks persisted only on access). First flush takes `mode="w"`: stitched groups
(finalize-only), nested multi-mode GI subgroups from prior runs, and any foreign groups are absent
from the rewrite; the append-axis backstop is bypassed (tmp file is empty). **Fix:** tighten the
predicate to "no integrated rows on disk", or `frames.mark_persisted(frames.index)` right after
the Append load (which is truthful — those frames ARE on disk). *(Verified by orchestrator.)*

### BL-3. MD-1 auto-sidecar discovery can latch a junk companion and poison scan_data — and `auto` is now the default
`xrd_tools/io/metadata.py:269-294` (`_iter_auto_sidecar_candidates`), `:235-239`, `:320-327`
(min-pairs=3 at `:17`). Candidates are ANY non-image `image.name.*`/`image.stem.*` file, tried in
alphabetical order, accepted on ≥3 loose `key=value`/`key: value` pairs. A per-frame `.poni`
(colon-separated — a real artifact of this project's own stitching workflow) sorts before `.txt`
and WINS; `img.tif.log` beats `img.tif.txt`; pretty-printed JSON passes. The winning convention is
cached per `(dir, image_suffix)` and applied to every frame → wrong motors/counters (incl. GI
incidence) persisted into the `.nxs`. Commit `3e11ed82` makes `auto` the out-of-box default.
**Fix:** extension allow-list for auto candidates (`txt`, `pdi`, `metadata`, `inf`); rank known
metadata extensions first; harden plausibility (reject `�`/control chars); cap candidate file
size; log which convention auto locked onto. Related SERIOUS: auto's `.txt` route is
SSRL-format-only (generic `name=value` .txt yields `{}` → a worse candidate latches); explicit
`metadata` format is gated by the auto min-pairs threshold (1–2-pair sidecars silently dropped
with a misleading "unknown format" warning).

### BL-4. `_BoundedFrameHandoff` eviction races the GUI pop → KeyError kills the run
`wrangler_widget.py:56-63`: drop-oldest snapshots `list(self.keys())[:over]` then
`super().__delitem__(stale)`. Writer thread inserts (`_publish_display`), GUI thread pops
(`update_data` `pop(idx, None)`) — and eviction fires exactly when the GUI is ≥128 behind, i.e.
when its queued pops target the same oldest keys. A pop between snapshot and delete raises
KeyError out of `QtNexusSink.write()` → run recorded as failed. MEM-1a exists to make overload
survivable; this converts it into a crash. **Fix (one line):** `super().pop(stale, None)`.
*(Verified by orchestrator.)*

### BL-5. Test suite red at tip (must be green at the tag SHA)
- `test_filter_sites::test_eiger_master_queue_applies_filter_to_stem` fails deterministically:
  `3e11ed82` added an `img_ext` early-return to `_eiger_refill_master_queue`
  (`image_wrangler_thread.py:2700-2702`); the test's `SimpleNamespace` holder lacks `img_ext`.
  Product path fine; add `img_ext="h5"` to the holder.
- `test_live_refresh::test_streaming_dispatch_series_average_submits_one_mean_frame` is
  RAM-dependent: binds the real `_heavy_staging_window` with no detector shape and asserts
  `_max_heavy_items == 64`, but MEM-2's shape-unknown tier returns 16/32 below 32 GiB — fails on
  16 GB CI runners. Pin `XDART_HEAVY_WINDOW=64` in the test.
- Full state at tip `59f9c6c0`: core 1468 pass / 3 env-skips; xdart offscreen ~1552 pass / 2 fail
  (above) / 6 skip; guards + purity green; one known exit-139 teardown flake. GI spine 71/71 at
  tip (was 57/71 red at `031c134c`; MEM-3 `59f9c6c0` fixed it, matching its message).
- Re-run the full gate at the frozen tag SHA — the tree moved twice during verification.

### BL-6. OV-6 compatible cross-scan append plots traces at the WRONG x
`display_publication.py:982-987` seeds `ref_x` from the current render (new scan's grid), never
from `prior.x`; `xrd_tools/session/display_logic.py:633-636` appends against `history.x` (OLD
scan's grid) and reinterps ONLY if sample-count differs. Two scans with same axis kind + npt but
different `radial_range` (recalibration, edited range — the very cross-scan-comparison workflow
OV-6 exists for) are "compatible" by the grid key (range deliberately excluded), so scan B's
intensities render at scan A's x positions. Silent misgridded overlay; the legacy accumulator
grid-merged via `np.union1d`. **Fix:** seed `ref_x = prior.x` when prior is compatible and unit
unchanged, and/or in `accumulate_waterfall` reinterp when sizes match but
`not np.allclose(x, base_x)`. Surfaces at Session-1 F7/F8. *(Verified by orchestrator.)*

---

## SERIOUS

### Data correctness (written .nxs)
- **S-1. Dropped-modes report wired dead (one-line fix).** `scan.py::_save_to_nexus` never
  `return`s the `save_scan_to_nexus(...)` result; the sink does
  `dropped = self._scan._save_to_nexus(mode=mode)` (`qt_nexus_sink.py:450`) → always `None` →
  gate-dropped GI 2D modes are marked persisted → wrong `is_persisted`, evict-then-fail-hydrate
  loops. The MEM-1b `mark_dropped` fix is dead code in production; tests pass via a monkeypatched
  fake. **Fix:** propagate the return through `_save_to_nexus`/`save_to_nexus` + one
  non-monkeypatched test. *(Verified by orchestrator.)*
- **S-2. CF-1's headless blocker is wired dead; the modal's scoping leaves ungated Append paths.**
  `static_scan_widget.py:2557` passes `processed_config=None`, so
  `append_config_mismatch_check` never fires in production — only the Run-click modal enforces.
  The `14efca96` scoping returns "ok" when the predicted target ≠ loaded file — which includes (a)
  Append onto an existing-but-not-loaded `.nxs`, and (b) live runs where `img_file` is empty at
  Run click. There the only guard is the writer backstop, which compares AXIS VALUES — a config
  change producing an identical grid appends silently mixed data. **Fix:** thread-side check in
  `initialize_scan`'s Append branch comparing the target file's stored reduction-config signature
  vs run args (fail-loud); populate `processed_config` from the loaded scan.
- **S-3. CF-1 signature omits value-affecting, grid-preserving params:** mask file, PONI /
  calibration, `chi_offset`, polarization, monitor normalization, error model, manual GI `th_val`
  (`readiness.py:40-63`). Changing any passes both the modal and the axis backstop → mixed
  provenance rows under a `/entry/reduction` that claims the first run's config. Extend the
  signature (mask path+mtime, poni signature, chi_offset, polarization, monitor, th_val) or ADR +
  MIGRATION-disclose the acceptance.
- **S-4. GI-1 "auto ≡ explicit −180..180" fails in STANDARD mode through the real GUI pipeline
  because of `chi_offset` (default 90°).** Auto injects (−180,180) in pyFAI's raw χ frame
  (`integrate/single.py:154-158`) but GUI-explicit ranges are shifted by −chi_offset first
  (`readiness.py:1154-1156`); the written 1D χ axis is 90° out of frame with the 2D cake χ (the 2D
  branch re-adds the offset, the 1D branch never does). GI mode unaffected — which is why the live
  χGI repro passed. The new byte-equality tests call the integrate functions directly and miss it.
- **S-5. GI mode-switch range re-key does NOT exist** (two agents independently; the
  commit/ledger description doesn't match code). `q_oop`/`exit_angle`/`chi_gi` share the
  `azimuth_range` key in different units; a frozen/hydrated explicit range survives
  `_controls_v2_axis_to_native` (`static_scan_widget.py:1376-1400`, only `gi_mode_1d` changes) →
  next run silently clips χGI to a ~4° wedge and WRITES it. **Fix:** clear (or re-freeze) the
  output-axis range keys on `gi_mode_*`/unit change.
- **S-6. Two divergent monitor-norm "canonicals" post-Stage-5.** The reduction spine's
  `_normalization_for` (`reduction/core.py:2837-2883`; exact/upper/lower lookup, accepts negative)
  vs the new `core.metadata.resolve_monitor_norm` (case-insensitive scan, rejects ≤0) now used by
  the GUI mirror — mixed-case keys → spine writes UN-normalized data while the file's `map_norm`
  claims normalization. **Fix:** make `_normalization_for` delegate lookup/guard to the canonical
  resolver.
- **S-7. Stage-4 "byte-identical shim" is false:** every GUI-written file gains
  `/entry/reduction/config/gi` (fixture re-pin admits it: the only content change in
  `v2_record_signature_pre6a.json` is that dataset); the parity test uses a `.gi`-less
  SimpleNamespace so it can't notice. Reload stays compatible. **Fix:** gi-bearing parity fixture +
  MIGRATION disclosure (or gate off the GUI path).
- **S-8. PF-2: lexicographic `>= first_img` filter drops series members**
  (`image_wrangler_thread.py:3075-3076`): `'-' < '_'` so mixed dash/underscore series (which the
  new `[_-]` parser deliberately unifies) silently exclude members; unpadded series (`frame_2` vs
  `frame_10`) drop numerically-later frames. Dropped files never reach discovery, so the new
  warning can't fire. **Fix:** filter by parsed `(scan_name, index)`. Companion: the
  zero-processed warning early-returns on `discovered <= 0` (`:991-994`) — the original PF-2
  symptom shape still ends with only "Total Files Processed: 0"; warn with directory/pattern/ext.
- **S-9. Live whole-scan Overall Sum/Average silently omits the writer's unflushed tail.**
  `scan_aggregate.py` reads disk ⊕ `_unflushed_tail(scan)` — but the scan passed is the GUI-side
  `LiveScan` whose staging is never populated during live; the tail is always empty → up to
  cap−margin newest frames missing from the rendered aggregate during a run (self-heals at flush).
  The exact "silent subset" P1 class H8 set out to kill; the module docstring's invariant is false
  for the object passed. **Fix:** fetch the tail from the wrangler thread's scan, fold in the
  resident publication tail, or annotate the trace as partial during runs.
- **S-10. Live Overall+Average triggers continuous full-stack HDF5 re-reads under the shared
  `file_lock`** (generation signature bumps per drain tick → cache invalidated ~5/s; each
  aggregation holds `file_lock` for the whole chunked read; the writer's flush blocks behind it →
  backpressure → handoff drops). **Fix:** recompute at flush boundaries only during
  `_processing_active` (disk only changes at flush), and/or chunk the lock hold.

### MEM-2 wiring
- **S-11. The frame-precise heavy window is never applied to the GUI `PublicationStore`** (built
  once at app start with the coarse RAM-tier default; the code comment claiming it "reads
  `self._heavy_window`" is false), and **the cached `_heavy_window` is never invalidated on
  detector switch** (`image_wrangler_thread.py:2301-2310`: compute with Pilatus → window 64; switch
  to Eiger-16M same session → 64 × ~144 MB ≈ 9 GB on a 16 GB box — the exact OOM MEM-2 prevents).
  All four wiring tests pin `XDART_HEAVY_WINDOW`, forcing the caps equal by construction. **Fix:**
  key the cache by detector shape; resize the publication store at `_get_streaming_session` time;
  one test asserting the real topology without the env override. Note: MEM-3 (`59f9c6c0`) landed
  after review start — re-check whether it addresses any of this before fixing.

### Locking / lifetime (GUI)
- **S-12. Five writer sites pause the H5 pool BEFORE taking `file_lock`,** violating the
  documented H30 invariant (`image_wrangler_thread.py:2021-2023`): `scan_threads.py:442/456/513/559`
  (reintegrate-shadow writers) + `nexus_wrangler_thread.py:409` (Stop tail flush). `pause()` closes
  handles unconditionally → can close a handle under a mid-read load worker → silently lost/partial
  loads (RN-1's failure family). Two agents independently. **Fix:** swap the nesting (file_lock
  outer) or reuse `_h5pool_bracket` under the lock at all five sites.
- **S-13. Close-during-run qFatal, two paths:** (a) `_teardown_load_worker` timeout branch
  (`h5viewer.py:2894-2906`) returns without `setParent(None)`/retain — likely during live close
  (final flush holds `file_lock` seconds; worker blocks; `wait(2000)` times out; parented running
  QThread destroyed → process abort). Mirror the retire path (3 lines). (b) `stitch_thread` never
  stopped/waited in `staticWidget.close()` (created `:565`, started `:5409`; close waits for
  everything else). Same class.
- **S-14. Same-name re-run: overlay permanently shows the PREVIOUS run's curves.** Row identity is
  `(scan.name, frame_idx)`; compatible-grid rescope keeps history; first-occurrence dedupe drops
  every new frame → stale intensities under live labels for the whole run. **Fix:** per-rescope
  nonce in `current_scan_key` or clear on same-name re-run.
- **S-15. H12 level-reuse cache never invalidated** (`image_widget.py:253-271`): key omits scan
  identity; `_clear_image_widget` leaves it → scan A's contrast applied to scan B within the TTL,
  and a cache-hit as last render leaves wrong levels indefinitely. **Fix:** null in
  `_clear_image_widget` + scan token in key.
- **S-16. Norm-channel wipe detection desyncs from application** (`display_frame_widget.py:3698`
  keys on `_last_applied_norm_channel`; per-row normalization reads the live combo;
  `refresh_norm_channels` can silently reset the combo cross-scan) → permanently mixed
  normalized/unnormalized accumulator with no reset. Record the channel actually applied on the
  history; any difference ⇒ reset.
- **S-17. One empty-grid publication wipes the accumulator (first row) or raises uncaught
  `np.interp` ValueError (later row)** (`display_publication.py:964-967`,
  `display_logic.py:604-615`). Skip `x.size == 0` rows.
- **S-18. Pinned slice-cut recipes are scan-unqualified at rematerialization** — pins survive
  boundaries/norm resets; recipe stores a bare frame int → rebuilt from the CURRENT scan's frame N
  under the old legend. Stamp `current_scan_key` into the recipe at pin time; prune on mismatch.

### Perf (headless, silent O(N)/O(N²))
- **S-19. `provenance_config._inputs_from_scan` frame-walk hydrates every evicted frame from HDF5
  for a guaranteed-empty result** (xdart `LiveFrame` has no `source_path`) — on every replace-mode
  save and reintegrate swap: O(N) full-frame reads per batch, against MEM-1/2's goals. Gate the
  walk to spine `Scan` objects or read `source_file` without hydration.
- **S-20. Per-frame O(directory) sidecar lookups** (`metadata.py:179-196` case-insensitive probe
  does up to two full `iterdir()` sweeps per frame; auto negatives never cached; GUI seed probe
  makes O(N²·logN) per parameter edit). Beamline-NFS-hostile. Stat-first + per-dir listing cache +
  negative-cache TTL.

### Release packaging/docs
- **S-21. MIGRATION.md gaps:** missing CF-1/CF-2 (Append now blocked with modal — was silent
  no-op), MEM-1c (series-average Append refusal — data-loss-class fix), PF-2 dash-index convention
  (also changes scan grouping: `LaB6-1..60.tif` = 60 scans before, ONE scan now), the
  `/entry/reduction/config/gi` dataset (S-7), explicit out-of-domain χ range clamping. And the
  "Post-v1.0 — Plan B item 3" header is false — those notes ship IN 1.0.0; reword before tag.

---

## MINOR (fix opportunistically; none tag-blocking alone)

Core/metadata: `read_image_metadata(meta_format=None)` now means auto-discover, not off (public
API trap — keep `None` → `{}`); string metadata values crash the series-average worker
(`image_wrangler_thread.py:3113-3117` catches only TypeError — add ValueError/KeyError, average
numerics only); BOM/UTF-16 sidecars (`utf-8-sig`); auto-vs-explicit sidecar precedence inverted;
stale `== "None"` vs lowercase `'none'` at `static_scan_widget.py:3018` (probe-only); SPEC reader
clamps out-of-range frame numbers to last scan point silently.

Writer/store: H6 stale extra-mode subgroups survive a mode-set-changing reintegrate and stay
listed in `multi_result_modes`; `mark_dropped` frames permanently exempt from the `max_items`
bound (thin-record growth on long GI scans); `get_or_hydrate` persisted-set TOCTOU (stale
`prev_persisted` re-applied after a concurrent flush); cross-scan publication ghost re-stamped to
current generation; H3 plateau gate runs `record_store_persisted_on_write=True` while production
wires False (flush-marked path outside the gate); `catch_h5py_file` erases errno/exception
subclass (public API); `setBkg` guards 2D/raw with `require_all` but not `bkg_1d`.

GUI: stale `_browser_scan_reset_pending` fires deferred reset mid-reintegrate (clear in
`_enter_run_state`); `open_folder` misses the LD-1 `cancel_pending_loads`; planned-npt drift wipes
pins on a `numpoints` edit; overlay seed double-subtracts background at Overlay entry with Set-Bkg
active; all-NaN image → NaN autoLevels + stale colorbar (`update_wf_pmesh` misses all H12 guards);
Pin button enabled-but-inert outside Overlay/Waterfall; CF-2 modal's "re-integrate all N frames"
overstates (Replace uses the `>= first_img` filter); duplicate tracebacks for one writer failure
+ short-run abort never prints the preservation line; `x_0001.tif` vs `x-0001.tif` collide on one
label (silent last-wins); `.nxs` directory sweep can ingest xdart's own processed outputs as raw
with Include-Subdir (skip files carrying `entry/reduction`); streaming `replace` never passes
`replace_frame_indices` to a flush (unreachable today; landmine); bulk 1D hydration: one missing
label poisons its ≤256-frame chunk; skipped single-image emits phantom `sigUpdate`.

Release: stale `build/` dir in worktree (add `rm -rf build/` to RC-8); README license line says
MIT only (metadata is `MIT AND BSD-3-Clause`); dangling `release_final_verification.md` citation;
`_gui_main.run()` discards `app.exec()` return (always exit 0); `test_image_widget.py` is a
manual script with a `test_` name; env knobs undocumented outside design docs; a few ledger rows
say "DONE this commit" without SHA.

GI residuals: fully out-of-domain explicit range clamps to an inverted pair (validate lo<hi,
raise); `integrate_radial` doesn't clamp explicit ranges (inconsistent with `gid.py`); no test
pins that the default radial grid spans ±180.

---

## Controls-panel-v2 — small pre-tag robustness wins (ranked, ≈1 day total)

1. **Run-lock hole (S, ~1-2h):** V2 `ActionButton`s + GI `…` popup + Source-energy `…` stay LIVE
   during a run — `CHOOSE_SOURCE/CHOOSE_PROJECT/CHOOSE_OUTPUT` call wrangler browse methods
   directly with no run guards; GI popup rows are built from pre-run snapshots with
   `enabled=True` and write `scan.gi_config` mid-run. Extend
   `_set_controls_v2_current_fields_enabled` to ActionButtons + More-buttons; early-return in
   `_on_controls_v2_action`/`_on_controls_v2_field_changed` when run-active. (Calibrate/Make
   Mask/Reintegrate are accidentally safe — they delegate to disabled legacy buttons.)
2. **Advanced dialog escapes the run lock via reparenting (S, ~30min):** the combined dialog
   reparents `advancedWidget1D/2D.tree` into itself, so disabling the advancedWidgets no longer
   reaches the trees; edits mid-run silently stick for the next run. Disable
   `_integ_adv_combined_dlg` in `_enter_run_state` (4 lines).
3. **Stale frame-count cache can hold Run disabled forever (S, ~1h):**
   `_v2_frame_count_cache` never invalidates on filesystem change — an initially-empty directory
   stays "Choose a frame source" as frames arrive. Fold `st_mtime_ns` into the cache key (same
   for the metadata probe cache).
4. **H5 parity test (S/M, test-only):** inline GUI `SourceCaps`
   (`static_scan_widget.py:2466-2475`, collapses `has_frames=has_raw=raw_reachable=source_ready`)
   vs headless `describe_source_readiness` — zero tests compare them today. Fixture-matrix
   equality test with the two deliberate divergences documented. The pre-tag substitute for H18.
5. **Append-mismatch readiness note (S, optional):** readiness bar says "Ready" until the CF-2
   modal. Add a non-blocking "Append target config differs — Run will prompt" tooltip using the
   already-written-but-DEAD `_controls_v2_append_target_matches_displayed_scan()` (`:2681` — wired
   to nobody; if not wiring it, delete it). Keep Run clickable — the pinned CF-2 tests require it.
6. **Hygiene (S each):** browse-cancel clobbers `img_dir`/`mask_file` outside the `if path != ''`
   guard (`image_wrangler.py:1836-1838, 2074-2076`); duplicate `_controls_v2_positive_float`
   definition (`static_scan_widget.py:2866` and `:2945` — second silently overrides);
   refresh-failure `except Exception: logger.debug` → warn-once (a persistent failure freezes the
   readiness row at its last value, possibly "Ready"); clamp npt ≥1 at the four leaves.

Already solid (verified): signal-loop guards throughout; session-blob restore fully defensive;
run-gating single-sourced from field statuses; StaticControls run bar hard-locks; PONI gating;
live-source escape hatch both sides; refresh perf discipline.

---

## Explicitly verified CLEAN (high-value assurances)

- **Move fidelity:** Stage 1 `controls_logic` → `session/readiness.py` = exactly one changed line;
  H22 `display_logic` move byte-identical (1963 lines); masks byte-identical; shims cover every
  name imported anywhere in xdart/tests; zero Qt/xdart imports in `xrd_tools`; purity guards real.
- **H6 writer:** non-GI single-mode path structurally unchanged; fixture re-pin adds only the
  storage-metadata fields + the S-7 `gi` dataset (no numeric drift); compat gate drives the real
  GUI writer end-to-end.
- **H9 deletion complete:** only 3 explanatory comments reference data_1d/data_2d/hydrated_raw;
  Role-B `_ViewerRows` survives as designed; no dead imports.
- **H8 aggregation guards:** explicit Sum/Average subsets hydrate-or-refuse (never silent
  resident-subset averages); disk⊕tail dedup by label prevents double-count — the P1 truncation
  class is closed except S-9's live-tail gap.
- **Lock graph:** no deadlock cycle found; `file_lock → pool/scan_lock` hierarchy holds at all
  wrangler write sites; stores release locks across hydrator I/O; persist-before-evict invariant
  holds under any MEM-2 window; RN-2 chunking correct and worker-thread-only.
- **No cross-thread QTimer starts** (the 0.37.1 class); generation scheduler drops no frames at
  run end; LD-1 covers all real file-swap paths except `open_folder`.
- **CF-1 core compare** robust to float roundtrip/aliases/ordering; Replace truly
  truncates-then-reintegrates (no mixing); modal Cancel doesn't run; provenance written after
  validation.
- **PF-1:** snapshot mode='r' primed once; per-frame checks in-memory; MEM-1c blocker scoped
  correctly (no false blocking of legit partial appends); all-skipped reload works.
- **GI-1 core:** auto injection only when range is None (explicit q/2θ data unchanged); freeze
  clamp makes scout==per-frame==explicit in GI; MIGRATION discloses the auto change.
- **UX-1 shortcuts** gate on `isEnabled()` and route through `button.click()`.
- **Packaging:** version 1.0.0; PEP-639 `MIT AND BSD-3-Clause` + license-files in wheel METADATA;
  wheel+sdist build, twine check PASSED; base `import xrd_tools` pulls zero Qt AND zero heavy
  modules; `ssrl_xrd_tools` shim one DeprecationWarning, true module identity; all versions
  dynamic from dist metadata; `.ui` files packaged.
- **Refuted findings honored** (per-run wrangler-thread leak, set_wrangler accumulation, GI-2D
  freeze hole — not re-raised).

---

## Suggested sequencing

1. **One-liners first:** BL-4 (`pop(stale, None)`), S-1 (return the drop report), BL-5 test fixes.
2. **Append integrity commit:** BL-1 + BL-2 (+ mark_persisted-after-load), M-item duplicate
   tracebacks if convenient. These are the tag-gating data-destruction paths.
3. **Sidecar hardening commit:** BL-3 + the `.txt` route + explicit-format threshold (S-5-adjacent)
   + perf S-20 stat-first.
4. **Overlay/x-grid commit:** BL-6 + S-14 (+ S-17 empty-row skip) — same code region.
5. **CF/GI truth commit:** S-2 (thread-side Append signature check), S-3 (extend signature or
   ADR+disclose), S-4/S-5 (chi_offset 1D consistency + range re-key on mode switch).
6. **Lock/lifetime commit:** S-12 (five pause sites), S-13 (two teardown paths).
7. **MEM-2 follow-up:** S-11 (check MEM-3 first), S-9/S-10 (live aggregate tail + throttle).
8. **MIGRATION pass:** S-21 + README license + "Post-v1.0" header.
9. **CP-v2 batch:** wins 1+2 (run-lock story), 3, 6 (hygiene), 4 (test-only), 5 optional.
10. Re-run the full gate at the frozen tag SHA; then RC-7s Session-1 (add BL-6/S-14 overlay
    scenarios and an Append-degraded-load drill to the checklist), then RC-8.

Ledger rows to reopen/annotate: CF-1, CF-2 (S-2/S-3), GI-1 (S-4/S-5), PF-2 (S-8), MS-1
(reconciliation is log-only; batch runs get no indexed-vs-processed check), MEM-1b (S-1), MEM-2
(S-11), OV-6 (BL-6), OV-7 (S-18).

---

# ADDENDUM — Fix-wave re-review (2026-07-04, tip `1da29875`)

Re-review of the blocker wave: Lane A (MEM-3 `e75c1a80`, BW-A1 `41cd5f11`, BW-A2 `769360e3`,
BW-A1b `d7d87051`, BW-A3 `947750b6`+`9a7cdf82`) and the Lane-B port (`db2b5d16`..`1da29875`).
Two verification agents + orchestrator spot-checks of the material claims against source.

## Verdicts per original finding

| Finding | Verdict | Notes |
|---|---|---|
| BL-1 Append truncate on degraded load | **CLOSED** (`769360e3`) | `_nexus_integrated_frame_count` disk probe; non-zero or probe-failure → loud RuntimeError, file untouched; legit empty skeleton still allowed; partial-degraded load fail-loud via the append-axis guard |
| BL-2 first-flush atomic `"w"` | **CLOSED** (`769360e3`) | `mark_persisted` after Append load (truthful); predicate unchanged; fresh Overwrite still gets atomic first flush; stitched/multi-mode/foreign groups survive |
| BL-3 sidecar junk latch | **CLOSED** (`db2b5d16`) with 2 residuals below | allow-list + ranking + binary rejection + 1 MiB cap + convention log + `.txt` generic fallback + explicit-format bypass, all empirically confirmed |
| BL-4 handoff KeyError race | **CLOSED** (`41cd5f11`) | `super().pop(stale, None)` |
| BL-5 red suite | **LANDED** (`41cd5f11`) | `img_ext="h5"` stub + `XDART_HEAVY_WINDOW=64` pin; NOT executed in sandbox — certify at frozen SHA on the Mac |
| BL-6 + S-17 overlay x-grid | **CLOSED** (`552365eb`) | prior.x seeding + value-mismatch reinterp + empty-row skip; unit-relabel path verified uncorrupted |
| S-1 dropped-report dead | **CLOSED** (`41cd5f11`) | return propagated both levels; test now production-wired (monkeypatch deleted, asserts on-disk 2D absence) |
| S-2 CF-1 gate dead / scoping | **CLOSED** (`947750b6`+`9a7cdf82`) | worker-side guard in `initialize_scan` reads stored `/entry/reduction` config, fail-loud pre-run; covers unloaded target, empty img_file, Eiger `_master`; canonicalization survived 13 adversarial probes (no false negatives OR positives found); Append-only scoping correct |
| S-5 range re-key | **CLOSED** (`67b103a3`) | real mode/unit change only; hydration-safe; over-clears in a few safe-direction cases (MINOR) |
| S-6 monitor-norm canonicals | **CLOSED** (`1c32db6a`) | true delegation; strict/warn-once preserved; MIGRATION disclosure present and honest |
| S-14 same-name re-run | **PARTIAL** (`7ac91c77`) | immediate A→A closed both directions; **A→B→A residual open (SERIOUS, below)** |
| S-16 norm-channel reset | **CLOSED as prescribed** (`1da29875`) | 2 scoped residuals below |
| S-18 pin scan-qualify | **CLOSED** (`7ac91c77`) | pruning + legacy-None tolerance verified |
| OV-7b Pin absorbs current | **SANE** (`67e115e9`) | 6-digit round both sides (no tolerance hole); palette iterator not advanced for current |
| MEM-3 | **SANE** (`e75c1a80`) | stdlib-only cap, no written-data impact; explains the 57/71 spine fix; silently bundles the CF-3 cold-launch modal + duplicate-traceback suppression (correct, but un-mentioned in message) |

## NEW / residual findings (orchestrator-verified where noted)

- **[SERIOUS] S-20 is NOT fixed and the BL-3 commit message claims it is.**
  `_existing_path_case_insensitive` (metadata.py:213-227) still leads with TWO full `iterdir()`
  sweeps per probe; auto probes 8 candidates → measured 16 full directory listings PER FRAME in a
  sidecar-less directory (old code: 1 sorted listing) — a ~16× regression on the default path.
  *(Orchestrator-verified in source.)* **Fix:** exact-case `candidate.is_file()` fast path FIRST,
  then a per-directory listing cache for the case-insensitive fallback. The declined negative
  cache stays declined (live-sidecar pickup test is right).
- **[SERIOUS, low likelihood] junk `.pdi` latch bypasses every BL-3 gate.**
  `_read_auto_candidate_metadata` (metadata.py:297-299) returns `read_pdi_metadata(path) or None`
  — no plausibility gate, and the pdi reader's last-resort fallback FABRICATES
  `{'TwoTheta': 0.0, 'Theta': 0.0}` from garbage (empirically confirmed) → cached convention,
  fake motors persisted for every frame. **Fix:** reject the fabricated-default result in the
  auto path (require a real parse).
- **[SERIOUS] S-14 residual: A→B→A re-run still shows run-1's curves.** The clear fires only on
  `prev_scan_key == name` (static_scan_widget.py:5609); after an intervening compatible scan B,
  re-running A collides `(A, idx)` row-ids with run-1 rows → first-occurrence dedup silently
  drops every new frame. *(Orchestrator-verified in source.)* **Fix:** track scan keys seen since
  the last accumulator reset and clear when a name RE-ENTERS the set (or the per-rescope nonce
  from the original review). Same residual applies to run-1 pins. Add to the ledger's
  acceptance-contract reset list (the contract text wasn't updated for the S-14 path).
- **[MINOR-SERIOUS] S-16 scope gap: per-row demotion still mixes silently** — `normalize()`
  skips division when a frame's metadata lacks the key or value ≤0 (display_data.py:1383); a
  dead/zero monitor mid-scan mixes rows with no reset (the widget-level channel record can't see
  row-level demotion). Record the applied channel per-row on the history if this matters.
- **[MINOR] boundary norm-combo race:** `refresh_norm_channels` with scan_data cleared at rescope
  can clear the combo and never re-select → silent de-normalization + (with S-16) a spurious
  wipe; narrow window (user repaint between rescope and first drain).
- **[MINOR] BW-A2 probe blind spot:** an Append target whose every prior frame was
  publication-dropped (per-frame groups present, integrated groups absent) still counts 0 →
  skeleton rewrite discards those groups. Cheap hardening: probe per-frame payload too, or warn.
- **[MINOR] new pre-run RuntimeErrors (BW-A2/A3) escape `imageThread.run()`** with no
  showLabel/status emit — surfaced only via the global excepthook dialog. Catch around
  `initialize_scan` → `showLabel.emit` + log, mirroring the MEM-1c blocker pattern.
- **[MINOR] `accumulate_waterfall` empty-x history append** (display_logic.py:630-631): adopt the
  incoming x when `history.x.size == 0`. Also hoist the loop-invariant `allclose(x, base_x)` out
  of the per-row loop.
- **[RESOLVED] v2 Append mismatch uses the shared Run-click modal again.**
  The temporary hard-block was reverted: readiness reports
  `ControlProfile.append_confirm_reason`, keeps Run clickable, and the real v2
  Run button reaches the same Yes/No modal path as the legacy panel.
- **[NOTE] legacy files without stored reduction config pass the worker guard silently**
  (`processed None → ok`) — only the axis backstop guards those appends; add a MIGRATION line.
- **[NOTE] ledger rows cite pre-port lane SHAs** (`63a248d9`/`74818d9b`) instead of the ported
  `db2b5d16`/`552365eb`; INFO re-log of run caps per batch chunk on long runs (cosmetic); the
  false PublicationStore comment at image_wrangler_thread.py:2384 survives as the S-11 marker.

## Decisions requested — recommendations

- **S-3 (CF signature gaps): take (A), scoped.** Extend the signature with the silent-wrong-data
  fields (mask path+mtime, PONI content signature, chi_offset, polarization, monitor key,
  th_val) — (B) documents the hole the CF work exists to close. Two constraints: per-field
  FAIL-OPEN when the stored config predates the field (else every legacy append bounces), and a
  MIGRATION line for the stricter gate. The canonicalization layer from 9a7cdf82 already gives
  the comparison machinery.
- **S-7 (config/gi byte drift): take (A)** — gi-bearing parity fixture + MIGRATION note. The
  field is useful provenance (the reload heuristic already prefers it) and (B) would re-fork the
  GUI vs headless schema, against the H6/H23 one-writer direction. The byte gate is already
  re-pinned; just make the fixture honest and disclose.
- **S-4 (1D χ chi_offset): endorse the proposed direction** — `azimuth_offset` on
  `Integration1DPlan`, drop the readiness-layer range shift, re-add the offset on the 1D χ output
  mirroring the 2D path (core.py:~2440), so 1D≡2D is headless-testable. Spine-touching: own
  commit, byte-compat + spine gates, and real-data validation of the absolute χ frame against the
  team reference exactly like gi_real_data_validation. Do NOT ship blind.

## Remaining before tag (updated)

1. S-20 stat-first (real fix this time) + `.pdi` plausibility gate — same file, one commit.
2. S-14 A→B→A (nonce or seen-set) + contract-text update.
3. S-12 five pause-before-lock sites (template: wrangler_widget.py:877-878).
4. S-8 parsed `(scan,index)` first_img filter (+ the zero-discovered warning).
5. S-11 heavy-window → PublicationStore wiring + detector-switch cache key (MEM-3 did NOT change
   this picture — confirmed) + delete the false comment.
6. S-4 with validation harness; S-3(A); S-7(A).
7. MIGRATION sweep (review S-21: CF-1/CF-2 modal+hard-block, MEM-1c, PF-2 grouping change,
   config/gi, legacy-no-config append note, "Post-v1.0" header).
8. Optional small: BW-A2 blind spot, run()-level catch, S-16 residuals, CP-v2 batch from the
   original review (run-lock hole, Advanced-dialog escape, frame-count mtime, H5 parity test).
9. Full gate (core + full offscreen xdart + spine 71 + byte-compat) at the frozen candidate SHA
   on the Mac — sandbox cannot execute the GUI suites.
10. Session-1 live checklist (now G1–G16) → tag.

---

# ADDENDUM 2 — Round-3 verification (2026-07-04, tip `c79a7434`)

Commits verified: `6f0b726f` (BW-A4 bundle: S-9/S-10/S-12 + unbilled S-8/S-11), `0c378f46` (S-3),
`457dcf53` (S-7), `c79a7434` (S-4).

## Verdicts
- **S-12 CLOSES** — all five sites lock-outer (order-asserting tests added); full pause-caller
  audit clean, no inversion remains.
- **S-8 CLOSES** — parsed `(scan_name, index)` filter + zero-discovered warning, trap-pinning
  tests (`'-'<'_'`, `frame_2`/`frame_10`).
- **S-11 CLOSES for the image wrangler** — `set_max_heavy_items` wired at session create AND
  reuse; heavy-window cache keyed by `(detector_shape, frame_bytes, env)` (Pilatus→Eiger closed);
  env-override-free test. Verify the nexusThread session path also resizes the store.
- **S-10 CLOSES starvation** — recompute keyed to the persisted-set signature (fires per flush,
  not per drain tick). Residual [MINOR-SERIOUS]: the `file_lock` hold is still monolithic for the
  whole full-stack read (`scan_aggregate.py:243-267`) — multi-second writer stall once per flush
  per key on huge 2D scans; chunk the lock if Session-1 shows stutter.
- **S-9 PARTIAL** — image wrangler closed (`_active_scan` handoff, `_cache_lock`-snapshotted
  tail, data_file guard, cleared on run end). **[SERIOUS] NeXus wrangler still open AND now
  frozen:** `nexusThread` never sets `_active_scan` → tail falls back to the GUI scan (empty) AND
  the new persisted-set generation signature is CONSTANT on the GUI scan → a live NeXus Overall
  aggregate computes once and never refreshes for the whole run (pre-commit it at least tracked
  flushes). **Fix (1 line):** set `self._active_scan = scan` in nexusThread's `initialize_scan`.
  [MINOR] GUI-side `_aggregate_data_signature` reads `frames._persisted` without `_cache_lock`.
- **S-3 CLOSES as scoped** — chi_offset/monitor/polarization/error_model ×(1d,2d) + gi_incidence,
  per-field `_UNSET` fail-open both sides, float round-12, monitor case-insensitive; written
  config carries the fields (not permanently fail-open); no byte-compat impact. Gaps: mask+PONI
  omission is rationalized only in the commit message — add a ledger row for the follow-up; the
  required MIGRATION line for the stricter gate is NOT yet in MIGRATION.md (riding S-21).
- **S-7 CLOSES** — gi-bearing scan through the real writer, `config/gi` asserted both ways,
  honest MIGRATION disclosure. [NOTE] assertion test, not a byte-parity pin of gi_config content.
- **S-4 PARTIAL — one NEW BLOCKER-class finding.**
  The output half is right (Integration1DPlan.azimuth_offset; readiness stops shifting; reduction
  relabels `r1d.radial` for chi_deg only; GI zeroed; G18 ship-gate honestly tracked). But the
  input half of the 2D mirror is MISSING: the 2D shifts an explicit panel-frame `azimuth_range`
  by −offset at input (`_integration_azimuth_range`, core.py:2524-2533) and re-adds at output;
  the new 1D Mode-A branch passes `p1.azimuth_range` RAW into `integrate_radial`
  (core.py:2390-2391) then relabels output +offset. With default chi_offset=90 an explicit χ
  range (0,90) read off the offset-labeled axes now integrates raw (0,90) = panel (90,180) — 90°
  from the 2D, from pre-fix behavior, and from the user's intent. Auto and full-domain are
  unaffected — which is exactly why every committed test AND an auto-only G18 validation would
  miss it. *(Orchestrator-verified in source.)* **Fix:** 1D analogue of
  `_integration_azimuth_range` in the Mode-A branch + a mirror of the 2D test
  (test_reduction.py:387/409) + include an EXPLICIT PARTIAL range in the G18 real-data harness.
  **[SERIOUS] legacy plan builder diverges:** `plan_from_live_scan`
  (xdart/modules/reduction.py:303-310) still pre-shifts the 1D input and never sets
  azimuth_offset — flipping `XDART_CONTROLS_PANEL_V2`/`XDART_CONTROLS_V2_NATIVE_RUN_PLAN` to "0"
  silently changes written 1D χ data; `test_reduction_adapters.py:358` pins the stale behavior.
  Port the fix or hard-deprecate the fallback. [SERIOUS-disclosure] changed written χ axes +
  Append-onto-pre-S-4-χ-file backstop bounce undisclosed — MIGRATION sweep. [NOTE]
  `_native_offset_range` (readiness.py:1193) now dead.

## Remaining before tag (round-3 update)
1. S-4 input-shift half + explicit-partial-range test + G18 harness amendment (blocker-class).
2. Port S-4 to `plan_from_live_scan` (or kill the env fallback) + fix the stale adapter pin.
3. nexusThread `_active_scan = scan` (1 line).
4. S-20 stat-first + `.pdi` plausibility gate (still open from Addendum 1).
5. S-14 A→B→A nonce/seen-set + contract text (still open).
6. MIGRATION sweep: S-21 set + S-3 gate line + S-4 χ disclosure + legacy-no-config note + the
   false "Post-v1.0" header.
7. Optional: chunk the aggregate lock hold; `_cache_lock` on the signature read; mask/PONI ledger
   row; nexusThread S-11 resize check; delete `_native_offset_range`; Addendum-1 minors
   (BW-A2 blind spot, run()-catch, S-16 residuals); CP-v2 small-win batch.
8. Full gate on the Mac at the frozen SHA → Session-1 (G1–G18, G18 amended) → tag.
