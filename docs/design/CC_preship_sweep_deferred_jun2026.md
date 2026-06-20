# Pre-ship sweep — deferred findings & refuted non-bugs (Jun 10, 2026)

Outcome of the final pre-merge multi-agent sweep of `refactor/architecture-v2`
(29 agents, every finding adversarially verified: 20 confirmed / 4 refuted).
All autonomously-fixable findings landed before the merge (xdart `f8748e3`,
`ad77873`, `2894071`, `5d5a3f5`, plus the serial-XYE flush fix).  The items
below are **deliberately deferred**: each is a behavior trade-off, and the
proper fixes belong to the post-release design refinements that start with the
monorepo migration (`CC_monorepo_handoff.md`).

## Deferred (fix during the monorepo design-refinement cycle)

### D1 — Reintegrate-All peak RAM (M4) — HIGH
`scan_threads._reintegrate_all` publishes every recomputed frame via
`frames[idx] = frame` → `stash()` marks it UNSAVED, and the only save is the
single end-of-run `save_to_nexus(replace_frame_indices=…)`.  Persist-before-
evict therefore pins **all N frames** (each ~8–16 MB with the float64 2D slab
`_load_frame_v2` loads even for 1D-only reintegration) — ~10 GB on a 651-frame
Eiger scan; OOM territory at 10k.  After the save they stay resident (eviction
only runs inside `stash`).
**Why deferred:** the obvious fix (periodic replace-saves inside the loop)
collides with the writer validator *by design* — a reintegration that changes
npts/ranges (the common case) must save all frames together
(`_select_frames_to_write` raises on partial shape-changed replaces).
**Design directions:** (a) stage shape-changed stacks in a temp group and swap
at the end so per-batch saves become legal; (b) teach the lazy loader to skip
the 2D slab for 1D-only reintegration (~100 KB/frame instead of 8–16 MB);
(c) explicit eviction sweep after the final save.
**Until then:** avoid reintegrate-all on multi-thousand-frame scans in one go.

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
(`xrd-session`) should give eviction an explicit owner.

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

### D1 addendum — Re-Integrate status & decision (Vivek, Jun 10)
The Re-Integrate 1D/2D buttons are **deliberately hidden** in v2
(`integratorUI: setVisible(False)`); the interim reprocessing story is
Start + Write Mode=Overwrite.  The D1 RAM issue is therefore **dormant**
(no GUI trigger) until the buttons return.

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

### D6 — chunked error-cleanup vs running worker (LOW, Jun 11, ssrl core.py:1540)
The except-BaseException cleanup in process()'s drain loop nulls
frame.image/background for all pending frames, but a future already RUNNING
can re-pin frame.image (core.py:1960) after the clear -- one frame's raw
(~18 MB) retained until session close, on an already-failing path.  Strictly
better than pre-fix (whole chunk leaked).  Airtight fix belongs with the
monorepo session-layer rework (e.g. clear inside _reduce_frame's finally).

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

### F6 — metadata readers should record ALL motor positions, not just the scanned motor (Vivek, Jun 19)
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
