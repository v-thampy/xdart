> **Note:** renamed from `CC_preship_sweep_deferred_jun2026.md` (which was
> gitignored under `docs/design/CC_*.md`) to `deferred_ledger.md` so the
> canonical deferred ledger is tracked and published with the release.

# Pre-ship sweep — deferred findings & refuted non-bugs (Jun 10, 2026)

Outcome of the final pre-merge multi-agent sweep of `refactor/architecture-v2`
(29 agents, every finding adversarially verified: 20 confirmed / 4 refuted).
All autonomously-fixable findings landed before the merge (xdart `f8748e3`,
`ad77873`, `2894071`, `5d5a3f5`, plus the serial-XYE flush fix).  The items
below are **deliberately deferred**: each is a behavior trade-off, and the
proper fixes belong to the post-release design refinements that start with the
monorepo migration (`CC_monorepo_handoff.md`).

## v1.0 inclusion summary (Jun 20, 2026 — final wrap-up)

Audited every item below against the shipped code this round.  **Now shipping in
v1.0** (done/fixed, verified):
- **D1** Reintegrate-All RAM — FIXED (streaming reintegrate writes each batch to
  shadow integrated groups, then atomically swaps the completed stack);
  Re-Integrate 1D/2D buttons **re-exposed** (D1 addendum corrected).
- **D5** hydrated-raw LRU — FIXED (shared order on `data_2d` under `data_lock`).
- **D6** chunked error-cleanup — FIXED (`_wait_pending_futures` drains in-flight
  futures before clearing `frame.image`).
- **F1** boolean filter grammar — DONE (`compile_filter`, all three sites).
- **F5** Set Bkg in viewer modes — DONE (display-only, shared row).
- **F7** "Auto Mask Saturated" authoritative toggle — DONE (integrator panel);
  the "More" popup stays deferred.
- **F8** viewer intensity slider + Autoscale — DONE.
- **F6** metadata all-motors — DONE (reader **and** the nexus GI-motor
  **dropdown** wiring: `nexus_wrangler._emit_gi_motor_options` →
  `sigGIMotorOptions`, fired on browse + setup).

**Still deferred** (post-v1.0 / monorepo cycle): D2, D3, D4, F2, F3, F4, plus the
F7 "More" popup follow-up.  Stage-4 `update_plot` deletion + Stage-5 Role-A
`data_1d`/`data_2d` + `hydrated_raw.py` removal are also post-v1.0 (live-gated;
the dicts remain as bounded transitional mirrors for the viewer/fallback paths).

**✅ FIXED (Jun 20):** the F6 `scan_data`-harvest reader regression —
`xrd_tools/io/nexus.py` `_harvest` + the NXpositioner cast now **pre-filter
`dtype.kind in "fiub"`** (skip `|S`/`|U`/object string columns without reading
them) and add `OSError` to the catch.  `frame_index` (a bookkeeping index our own
writer emits into `scan_data`) is skipped via `_NON_MOTOR_COLUMNS`.  Fixtures
(`|S` + vlen-string + `frame_index`) in `tests/core/test_nexus.py`.

**✅ BOTH former v1.0-TAG BLOCKERS ARE FIXED (Round-38 → resolved on
`fix/reintegrate-publication-drop-coverage`):**
1. **D1 Re-Integrate RAM — FIXED via shadow-stack streaming reintegrate**
   (`scan_threads._reintegrate_all` + `nexus_writer` shadow helpers): each batch
   streams to a `…__reint` shadow group, marks the frames persisted + evicts, so
   peak memory is bounded to ~cap (no longer the single end-of-run save that
   pinned all N).  The 1D-only pass drops the stale 2D slab; raw is dropped
   per frame.  Remaining reintegrate work is tuning, not correctness: keep an eye
   on per-batch RAW residency on very small-memory machines.  Interrupted shadows
   are recovered by `cleanup_reintegrate_shadow_groups` on the next writer pass
   and resolved read-only by `schema.resolve_integrated_group` on read.
2. **Azimuthal Mode A (non-GI χ) — FIXED**: `reduction/core._reduce_frame` now
   dispatches `unit=='chi_deg'` to `integrate_radial` (q-band as `radial_range`,
   `radial_unit='q_A^-1'`, `npt`=χ bins, tunable `npt_rad` band sampling) and the
   writer records/validates the 1D `axis_kind` (declared in schema CAPABILITIES).
   (Mode B / CHI_GI was already correct.)

## Resolved from the original deferred list

### D1 — Reintegrate-All peak RAM (M4) — FIXED + SHIPPING
Historical failure: reintegration used to hold all recomputed frames in memory
until one end-of-run `save_to_nexus(replace_frame_indices=…)`, which pinned
large raw/cake payloads for the full scan.

Current fix: reintegration now streams batches to shadow integrated groups,
marks the written frames persisted, evicts old frame payloads, and atomically
swaps the completed shadow stack into place at the end.  Stop/error drops the
shadow and leaves the canonical integrated stack unchanged.  Re-Integrate 1D/2D
buttons are re-exposed (`integratorUI.py` `frame_reint`, `integrator.py` wiring
to `bai_1d`/`bai_2d`).

Remaining work: only low-memory tuning, not a v1.0 blocker.

## Deferred (fix during the monorepo design-refinement cycle)

### D2 — `data_1d` thumbnail copies — MED
`copy_for_display(include_2d=False)` keeps a fresh ≤256×256 float32 thumbnail
(~262 KB) on every "1D-only" cache entry — ~10× the documented per-entry
budget; ~3–4 GB retained on a 10k-frame run (per-scan, cleared on scan
change).  The thumbnails are **functional** (the Raw preview reads them), so
they can't just be dropped.
**Design direction:** keep thumbnails for the most recent ~64 entries; older
previews lazy-load from the `.nxs` on demand (`ssrl_xrd_tools.io
.read_thumbnail` already exists).  Natural fit for the FrameRecord collapse.

### D3 — PublicationStore metadata duplication — LOW
`max_items=None` keeps one lightweight publication per frame for the scan's
life, each holding **4 independent dict copies** of `scan_info`
(FrameView.__post_init__ ×2, FramePublication.__post_init__ ×2) — ~0.1–0.5 GB
on a 10k-frame SPEC-rich run.  Capping `_items` is not safe today: the store
is the cake's only render source, so eviction would blank re-display of old
frames.
**Design direction:** share immutable metadata mappings instead of copying
(or collapse the FrameView/FramePublication pair) — part of the FrameRecord
collapse in the monorepo plan.

### D4 — `LiveFrameSeries` post-run residency — LOW (observation)
Even with persist-before-evict satisfied, nothing evicts after the *final*
save of a run; the last `_in_memory_cap` window stays resident between runs.
Bounded (cap 64), so harmless — but the monorepo session-layer redesign
(`xrd-session`) should give eviction an explicit owner.  Cross-ref: ADR-0005's
store→session ownership flip already homes the persist-before-evict bookkeeping
in `ScanSession`; the still-unowned piece is just the *post-final-save* sweep,
which `xrd-session` should pick up.

## Refuted — scary-sounding, traced unreachable (do NOT "fix")

1. **"imageWrangler leaks one QThread per run"** — the thread is built once in
   `__init__`; `setup()` only syncs attributes.  At most ONE scan is pinned
   between runs (`_active_scan`, overwritten on the next run).
2. **"set_wrangler accumulates filters/connections per wrangler switch"** —
   `set_wrangler` runs exactly once per session: `wranglerStack` has no
   selector UI and nothing ever changes the page.  Latent hazard only if a
   wrangler selector is reintroduced (note it then).
3. **"H5FilePool eviction closes handles other threads borrow"** — exactly one
   `pool.get()` call site exists (`_LoadFramesWorker.run`), and the previous
   worker is torn down before a new one spawns; the evicting thread IS the
   only borrower.  Revisit only if a second pool consumer is added.
4. **"GI 2D freeze can silently leave a needed range unfrozen"** — 2D axes are
   content-independent pyFAI bin-center grids (finite, increasing for any
   npt ≥ 2); the data-degenerate cases are all-dummy and hit the existing
   fail-loud GIFreezeError path.  Only an absurd npt=1 config reaches the
   gap, and it still fails loudly at the writer.

## Post-release feature requests (Vivek, Jun 10)

### F1 — Boolean filter expressions in Image Directory mode
**Today:** the Filter field whitespace-splits into ONE glob:
`"abc def"` → `*abc*def*` (image_wrangler_thread.py — built identically at the
Image Directory glob, the Eiger `_master.h5` queue glob, and the BG Match
filter).  That is an *ordered* AND — terms must appear in the filename in the
typed order; reversed order does not match.  No OR / NOT / XOR.

**Requested:** combine multiple filters with OR / AND / NOT (and possibly XOR).

**Design sketch:**
- Syntax: keep bare space-separated terms but make them an UNORDERED AND
  (matches user intuition; strictly widens current matches); `|` or `OR` for
  union; leading `-term` (or `NOT`) for exclusion; parenthesized groups only
  if demand appears.  XOR falls out of `(a | b) -(a b)` if ever needed.
- Implementation: stop encoding the filter into the glob.  Glob only
  `*.{ext}` / `*_master.h5`, then apply a compiled predicate
  (case-insensitive substring per term) to the names in Python — one shared
  `compile_filter(expr) -> Callable[[str], bool]` in ssrl (headless,
  unit-testable) used by all three sites.  Eiger/BG sites reuse it.
- Compatibility: a plain one-term filter behaves identically; multi-term
  changes from ordered to unordered AND (call it out in the release notes).
- Belongs with the monorepo design refinements (CC_monorepo_handoff.md);
  natural first user of the shared headless-helpers layout.

**STATUS (Jun 20): DONE + SHIPPING** — `compile_filter`
(`xrd_tools/core/filters.py`) implements unordered AND / OR (`|`) / NOT (`-`) as
a headless predicate, used at all three sites (image-dir glob, Eiger `_master.h5`
queue, BG Match); `"abc def"` now matches `def_abc`.  Tests in
`tests/core/test_filters.py`.

### D1 addendum — Re-Integrate status & decision (Vivek, Jun 10)
**SUPERSEDED (Jun 20):** the Re-Integrate 1D/2D buttons are now **VISIBLE and
wired** — `frame_reint` has no `setVisible(False)` (only the internal frame1D/2D
button sub-rows remain hidden).  The D1 RAM issue is live-triggerable again and
was FIXED (see the D1 STATUS line above).  *Historical (v2):* the buttons were
deliberately hidden (`integratorUI: setVisible(False)`); the interim reprocessing
story was Start + Write Mode=Overwrite, with D1 dormant until the buttons returned.

Why reintegrate stays in the design (indispensable cases):
- redo ONE output (1D or 2D) without losing the other — Start+Overwrite
  rewrites the whole file (an Int 1D rerun silently discards the old 2D
  stack: known footgun of the interim story);
- reprocess after SPEC/meta files are gone — the .nxs carries metadata;
  Start re-reads meta from disk;
- exact frame-set reproducibility — Start re-derives membership from the
  panel glob/filter, which can drift as the directory grows.
Note both paths still need the RAW files (integration needs pixels;
reintegrate lazily reloads from recorded source paths, R3-guarded).

Plan for the monorepo cycle: re-expose Re-Integrate TOGETHER with the D1
RAM fix, and route it through the same streaming-session machinery as
Start with a replace-aware sink ("one write path") — the Jun 10 writer
replace-path hardening (stale-dropped clearing, drop-before-validate) is
the foundation for exactly this.

### F2 — Save Path outside the project: allow-with-consequences design (post-release)
**Shipped behavior (conservative):** the Save Path box is editable and the
scans browser follows it, but an outside-project path is REJECTED with a
status message (previous valid value kept).  Rationale: today the .nxs never
embeds raw data — sources are stored as references (relative inside the
project per N1, absolute + warning outside) — so an outside save location's
only real consequence is the output leaving the portable project tree.

**Design questions for the monorepo cycle (Vivek, Jun 10):**
- Should outside paths be ALLOWED with an informed-consent warning instead of
  rejected?  What exactly does the warning promise (portability loss only)?
- Vivek's idea: optionally EMBED the raw frames in the .nxs when the output
  can't reference project-relative sources — a fully self-contained file.
  This is a new writer capability (size warning needed: ~18 MB/frame Eiger;
  651 frames ≈ 12 GB), interacts with @source_base semantics, free_raw /
  lazy-reload (raw could reload from the .nxs itself — would also unlock
  reintegration without the original source files, cf. D1 addendum), and the
  schema.  Needs design; natural companion to the Tiled-source work where
  raws may not exist as files at all.

### D5 — hydrated-raw LRU not applied on thread-side insert paths (LOW, Jun 11)
scan_threads.py reintegrate-display publish (~:236) and full-reload load_frames
(~:627) insert full map_raw into data_2d from worker threads without the
`_remember_hydrated_raw` trim (the LRU order list is GUI-thread-owned).
Bounded by FixSizeOrderedDict size (~40 x 18 MB worst case), not a true leak.
Fix in the monorepo cycle: make the hydrated-raw LRU state live with the
shared dict under data_lock so all writers can trim safely.
**STATUS (Jun 20): FIXED + SHIPPING** — shared order rides on `data_2d` under
`data_lock` (`hydrated_raw.py`); thread-side inserts (`scan_threads.py` reint
display, `h5viewer.py` full-reload) synchronize and trim the same cap.

### D6 — chunked error-cleanup vs running worker (LOW, Jun 11, ssrl core.py:1540)
The except-BaseException cleanup in process()'s drain loop nulls
frame.image/background for all pending frames, but a future already RUNNING
can re-pin frame.image (core.py:1960) after the clear -- one frame's raw
(~18 MB) retained until session close, on an already-failing path.  Strictly
better than pre-fix (whole chunk leaked).  Airtight fix belongs with the
monorepo session-layer rework (e.g. clear inside _reduce_frame's finally).
**STATUS (Jun 20): FIXED** — `_wait_pending_futures` drains in-flight futures
before the clear (the fix lives in THIS repo at `xrd_tools/reduction/core.py`,
not an external `ssrl` core.py — the heading path reference is historical), so a
running worker can no longer re-pin raw after the clear.

### D2 — status update (monorepo cycle, Jun 12): ANALYZED, DEFERRED AGAIN
Scouted during the Stage-6e deferred-items pass.  Decision: do NOT bolt a
thumbnail LRU + lazy reload onto the current display caches.  Reasons:
(1) lazy thumbnail reload in the display path would do h5py reads on the
GUI thread (latency hazard) — it wants the background-queue pattern of
_LoadFramesWorker; (2) the right owner is the PublicationStore once
publications become the SOLE display contract (that migration is already
in progress — display_controllers/display_data/metadata still read the
legacy dicts in parallel); (3) eviction policy should be unified with the
D5 hydrated-raw LRU (D5 itself was FIXED in this pass: shared order rides
on data_2d under data_lock — see hydrated_raw.py — so the thumbnail LRU
can adopt the same model when it lands in PublicationStore).
Revisit when the publication-store migration completes.

### F3 — ROI selection + per-scan ROI statistics (Vivek, Jun 12 — LIKELY FIRST
### POST-DESIGN PRIORITY)
Select one or more ROIs interactively (Image Viewer or the Int-2D raw panel,
e.g. pyqtgraph RectROI/PolyROI) and plot ROI statistics (sum / mean — later
max/min/std) as a function of frame across a scan.

Design notes for the monorepo cycle:
- Headless first: an `xrd_tools` ROI-stats primitive
  (`roi_stats(frames|source, rois, stats=("sum","mean")) -> per-frame table`)
  that iterates raw frames via the source layer (iter_chunks / get_raw_frame)
  -- usable from notebooks without the GUI, GUI is a thin view over it.
- ROI definitions should serialize (store with/next to the .nxs so a scan's
  ROIs reload; candidate: /entry/analysis/rois as schema-versioned records).
- The per-frame stats output is exactly the kind of derived per-scan series
  the metadata table already displays -- plot pane can reuse the 1D plot
  machinery (frame # / motor position on x).
- Fits the xrd-session layer (D1/D2 companions): ROI evaluation wants the
  same bounded raw-frame iteration the reintegration rework needs.

### F4 — Embed-full-raw flag in the .nxs + outside-project consent popup
### (Vivek, Jun 12 — makes the F2 embed idea explicit)
A user-visible flag that makes the writer store the FULL raw image data
inside the processed .nxs (today it stores only references: relative under
@source_base per N1, absolute + warning outside; thumbnails are the only
embedded pixels).  Already implicit in the F2/D1 design notes; recorded
explicitly per Vivek.

- **GUI placement:** expose the flag in the wrangler parameter tree directly
  BELOW the Save Path item (both image and nexus wranglers; session-persisted
  like project_folder/save_path).
- **Consent popup:** when the raw data path is NOT inside the project folder
  (i.e. relative source paths cannot resolve later — the file would otherwise
  carry absolute, non-portable refs), pop up a box asking whether to embed
  the raw data in the .nxs instead.
- **Popup frequency:** per-process vs once-per-session is UNDECIDED — Vivek
  expects this to be a trivially changeable knob; implement the prompt behind
  a small policy helper (e.g. `_should_prompt_embed(scope)` with the answer
  cached on the wrangler/session object) so flipping scope is a one-line
  change.  (Confirmed: easy to change later.)
- **Writer/schema side:** new capability on the v2 record — raw stack or
  per-frame raw datasets under the existing frames/ group; size warning
  needed (~18 MB/frame Eiger; 651-frame scan ≈ 12 GB); interacts with
  @source_base semantics, free_raw / lazy reload (raw could reload from the
  .nxs itself → unlocks D1 reintegration without source files), PERF-3, and
  the 6b schema-as-code declarations (additive group + capability attr).
- Companion to F2 (outside-project Save Path) and the Tiled-source work
  where raws may not exist as files at all.

### F5 — "Set Bkg" button in ALL display modes (Vivek, Jun 12)
Today the Set Bkg button exists only in Int 1D / Int 2D modes.  Expose it in
the viewer modes too (Image Viewer, XYE Viewer, 1D (XYE), NeXus Viewer where
sensible) — quick visual comparison of raw data against a chosen background
is useful even when nothing is being integrated.

Notes for implementation:
- Display-layer concern: per-mode panel button sets live in the controller/
  display_frame_widget layer; the Set Bkg toggle should become part of the
  shared button row (like the Log / colormap controls made common in the
  pre-release UI pass) rather than per-mode special cases.
- Background subtraction in viewer modes is DISPLAY-ONLY (never touches the
  publication/persisted record): subtract for the on-screen image/trace,
  consistent with the existing Int-mode display behavior.
- Mind mode-switch state: the chosen background should survive mode
  switches (store on the shared display state, not the mode controller),
  and the generation-stamp rules apply (background change = effective
  selection change -> bump display_generation).

**STATUS (Jun 20): DONE + SHIPPING** — Set Bkg exposed in Image/XYE viewers,
display-only (never touches the persisted record), on the shared display-state
row so it survives mode switches.

### F6 — metadata readers should record ALL motor positions, not just the scanned motor (Vivek, Jun 19)
**STATUS (Jun 20): DONE + SHIPPING — reader (all source types), the nexus
GI-motor dropdown wiring, AND the `|S`-column crash fix all landed (details below).**
- **SPEC** — already records all motors: `_read_spec_metadata` returns every
  `scan.motor_names` (the `#O`/`#P` header positioners) + every `scan.labels`
  (the `#L` per-point columns).
- **txt / pdi** — record whatever the file's Motors section lists (already all).
- **NeXus** — FIXED: `read_nexus`/`_read_data_group` now harvests
  `entry/data/` + `entry/scan_data/` (the SPEC-style all-motors per-point table)
  **and** `entry/sample/positioners/<motor>/value` (the scanned NXpositioner).
  So all motors reach the per-frame `scan_info` → GI incidence resolution +
  provenance work for nexus.  Test: `test_nexus.py::...harvests_scan_data_and_positioners`.
- **DONE (F6 dropdown):** the nexus wrangler widget now feeds those motor names
  to the integrator's GI-motor dropdown — `nexus_wrangler._emit_gi_motor_options`
  calls `read_nexus(...).angles` and emits `sigGIMotorOptions` on browse + setup
  (best-effort; a bad/locked file never crashes the GUI), mirroring the image
  wrangler.  So a non-standard nexus incidence motor is selectable.
- **FIXED (was a v1.0 blocker):** the `_read_data_group` harvest `|S`-column crash
  — both float casts (`_harvest` and the NXpositioner cast) now catch `OSError`
  and pre-filter `dtype.kind` (`xrd_tools/io/nexus.py` ~:2580/:2620, commit
  `8bbea58`), so a SPEC-style `scan_data` table with `|S` timestamp/label columns
  no longer crashes `read_nexus`.  Test: the `|S`/vlen/`frame_index` cases in
  `tests/core/test_nexus.py`.

Original context:
In a SPEC file the **scanned** motors/positioners are recorded per-point in the
**table data** (`#L` columns), while the **non-scanned** motors are recorded
**once at the scan header** (`#P`/`#O` positioner lines).  The current readers
likely capture only the scanned column(s); they should record **all** motor
positions — per-frame scanned + static header — and the same applies to future
NeXus / other source readers.

**Why it matters:** the GI incidence motor can be a non-standard / non-scanned
motor (desired default search order, case-insensitive: `th`, `theta`, `eta`,
`halpha`, `gth`, `gonth`).  If only the scanned motor is kept, a GI scan whose
incidence motor wasn't the scanned axis loses its angle source.  Also general
provenance/reprocessing value.

Notes for implementation:
- Look at `xrd_tools.io.metadata` (`read_image_metadata`, `_extract_scan_info`)
  + the SPEC parser; compare what it extracts vs the full `#P`/`#O` header vs
  the `#L` data columns.  Estimate effort first — depth unknown.
- Dependency of the GI-panel-move + 2-way-sync feature
  (`design_gi_panel_move_and_2way_sync_jun2026.md`): the incidence-motor
  dropdown's options come from this metadata.

### F7 — "Auto Mask Saturated" toggle → "More" filter-options popup (Vivek, Jun 20)
The integrator's **"Auto Mask Saturated"** toggle (authoritative on/off for
saturated-pixel masking — OFF masks nothing, keeping genuinely-saturated Bragg
peaks) should eventually become a **"More" button → filter-options popup** (same
pattern as the GI "More" popup), exposing additional per-pixel / per-stack image
filters.  First requested: **median filtering across a stack of images**; room for
others (e.g. outlier rejection, custom masks).

**Notes:**
- Keep it DISTINCT from the **"Threshold"** toggle (intensity-band filter,
  `apply_threshold`/`threshold_min/max`): they are independent controls.
- Mask/saturation masking is the per-frame `compute_bad_pixel_mask`
  (`xdart.modules.reduction`); a stack-median filter is a *cross-frame* operation
  → needs a different seam (likely `xrd_tools.corrections`/a provider stage), not
  the per-frame bad-pixel mask.  Estimate where it belongs before building.
- See memory `mask-saturated-toggle-authoritative`.

**STATUS (Jun 20): toggle DONE + SHIPPING** — the authoritative on/off "Auto Mask
Saturated" toggle ships in the integrator panel (gating the whole mask via
`compute_bad_pixel_mask`).  The "More" button → filter-options popup (stack-median
etc.) stays DEFERRED (a future cross-frame seam in `xrd_tools.corrections`).

### F8 — Image/XYE Viewer intensity-range slider + Autoscale toggle (Vivek, Jun 20)
The file viewers (Image Viewer, XYE Viewer) should grow a **second top row** —
mirroring the integration view's Q/Single/Options/Clear row for layout
consistency, but WITHOUT those boxes.  Instead, **right-justified** toward the
edge: an **intensity-range slider** + an **Autoscale toggle** next to it.

Behaviour:
- **Autoscale ON by default** (current behaviour — levels auto-fit the data).
- When Autoscale is OFF, the **slider sets the intensity scale**, ranging from the
  data **min → max** (image levels for the Image Viewer; y-axis range for the XYE
  Viewer).

**Why deferred (not a 5-10 min change):** a new slider+toggle widget added to the
viewer top row for BOTH viewer modes, wired to the display's levels (pyqtgraph
image LUT levels for the image; y-range for the 1D plot), with data-min/max
derivation and Autoscale-state persistence.  A real feature, not a tweak.

**Notes:**
- The viewer top row is owned by the display layer (`display_frame_widget` /
  the controllers); the Q/Single/Options/Clear row is the integration view's —
  the viewer variant shows only the slider+toggle, right-justified.
- Image Viewer "intensity" = pyqtgraph image levels (vmin/vmax); XYE Viewer
  "intensity" = the 1D plot's y-range.  A `Range` slider (two handles) fits both.
- Persist the Autoscale state + manual levels per viewer mode (session), like the
  other viewer controls.

**STATUS (Jun 20): DONE + SHIPPING** — intensity-range slider + Autoscale toggle
implemented for Image/XYE viewers, wired to the display levels (image LUT levels /
1D y-range); Autoscale ON by default, state persisted per viewer mode.

### UI — uniform control-panel text scaling (Vivek, Jun 30)
The **Small / Default / Large** "Control Panel Font Size" preset should scale
**all** control-panel text uniformly and predictably.  Today the scale is
fragmented across independent tokens (`control_panel_font`, `_status_font`,
`_run_font`, `_tick_font`, `_browse_font`) and — worse — several labels don't
even receive the token they nominally target, because a higher-specificity
ancestor rule wins:
- `QWidget#staticRunControls QLabel` (run font, 13px) overrode
  `QLabel#runReadinessLabel` (status font) — the readiness bar was silently
  pinned to the RUN font until the rule was qualified with the
  `#staticRunControls` ancestor (fixed point-wise Jun 30).
- `QWidget#controlsPanelV2 QLabel` (body font, 12px) overrides
  `#controlsV2SectionStatus` / `#controlsV2SubsectionStatus` (status font) — the
  section-header status text ("21 frames · Image Directory") is actually the
  BODY font, not the status font, so status-font edits never touch it.
- Pill rounding is height- (hence font-) coupled on macOS (QMacStyle only draws
  the rounded bezel above ~26px), so text size and pill shape are entangled.

**Want:** one coherent, documented font scale for the whole panel where the
preset moves everything together, the per-role tokens derive from a single base
(or are consolidated), and the QSS specificity is audited so every label
actually gets the token it names — letting per-role sizes be tuned in one place.

**Why deferred:** a theme-wide refactor + specificity audit across the controls
panel QSS, not a one-line tweak; touches `dark.py` tokens + many selectors and
wants a visual regression pass at all three presets.

**STATUS (Jun 30): DEFERRED** — logged from the pill/status-font session; the
readiness-bar specificity + the pill rounding were fixed point-wise, the uniform
scale is the follow-up.
