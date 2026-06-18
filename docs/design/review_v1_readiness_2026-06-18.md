# xrd-tools v1.0-readiness review — 2026-06-18

**Scope:** whole repo as v1.0 approaches, with the requested lenses — **performance,
memory, robustness, expandability** — plus a regression audit of the last few
sessions' churn, read against the design docs (`phase4_scansession_design.md`, the
ADRs, the three new feature design docs, the azimuthal doc). **Review only — no code
changed.**

**Baseline & surface:** the prior whole-repo review `review_2026-06-15.md` is the
catalogue; this round verifies its deltas and hunts **new** issues. `display-payload-unification`
(stable) is merged into `main` (`2ce388d`). The working branch
`overlay-waterfall-payload-flip` (`6cdfa0e`) is **in-flight WIP**, 10 commits ahead of
main — the overlay/waterfall "payload flip" (stages 1-5) + today's azimuthal-1D
headless wrapper. That delta (`git diff 7dbe292..HEAD`) is the freshest, least-reviewed
surface and got the most scrutiny.

**Method:** four parallel deep-review agents (perf/memory, core robustness+regressions,
xdart GUI+thin-xdart, expandability), every headline finding then hand-verified against
the cited code.

---

## 0. Verdict

**Genuinely close to v1.0. No P0. No silent-wrong-*data*.** The equivalence spine
(live ≡ batch ≡ reload) still holds — the recent churn did not diverge the three paths.
The three items the project memory tracked as open are **all now fixed in code**
(verified). Memory hygiene has clearly improved since 2026-06-15 (the headline unbounded
caches are gone). The architecture is well-prepared for the queued features — four of
the five seams are "fill a documented slot," and the FrameRecord store is already built.

**What stands between here and a confident ship is small and mostly defensive:**

- **One real correctness bug** (display-only, not data): the viewer background
  subtraction is scoped to the *mode*, not the *file* — switching frames within a viewer
  keeps a stale/cross-scan background, and the XYE path subtracts across mismatched
  grids/units. **§2 P1-a.**
- **One cheap robustness gap:** xdart is PySide6 with **no `sys.excepthook`**, so a stray
  read error escaping a slot aborts the process instead of logging. **§2 P1-b.**
- A handful of P2 memory/responsiveness items, all bounded and most carryover.

Before piling the queued features on, three **surgical** architecture gaps (§4) return
disproportionate value: a keyed-panel display payload (RSM/stitch viewers), a reader
`mode=` param (multi-mode reload + fitting-over-modes), and `ProcessedScan` metadata
completeness (stitch/ROI on reloaded files).

---

## 1. Carryovers & prior-review P1s — status (all good news)

**The three memory-tracked carryovers are RESOLVED (verified in code):**

| Carryover | Status | Evidence |
|---|---|---|
| Set-Bkg whole-scan raw read no-ops (bkg=0) on evicted frames | ✅ fixed | `display_frame_widget.py:2576` `get_frames_map_raw(idxs, require_all=True, allow_blocking_read=True)`; `display_data.py:502-504` forces sync hydration; refuses (returns None) if any frame missing rather than silently zeroing |
| R3-C headless saturation policy still xdart-only | ✅ fixed | `reduction/core.py:2132` `_apply_saturation_mask(...)` → `:2562` `from xrd_tools.core.invalid import integer_saturation_ceiling, saturation_pixels`; gated by `plan.mask_saturation`, uses the pre-float integer image so the exact-ceiling test is intact |
| phase-fit `nan_policy` docstring-only → headless GI-1D ValueErrors | ✅ fixed | `analysis/fitting/phase_fitting.py:1939` `nan_policy: str = "omit"` default, passed to lmfit at `:2010` |

**Prior-review (2026-06-15) P1 memory/perf items now fixed (verified):** PublicationStore
is bounded (`frame_publication.py:606`, `max_heavy_items=64`/`max_thumbnail_items=512`);
`bai_1d`/`bai_2d` whole-slab accumulators deleted (`ewald/scan.py`); RSM `combine_grids`
meshgrid eliminated (`rsm/gridding.py:606-623`); batch phase-fit lightweight default
(`analysis/fitting/batch.py:378`, `keep_results="auto"`); chunked aggregation reader
exists (`io/aggregate.py`) + `_slice_stack` skips the no-op reorder copy (`io/read.py:691`);
H5FilePool pause/resume is refcounted (`utils/h5pool.py`); scalar-zero background fast
path added (`core.py:2223`).

---

## 2. Regression audit — did bugs creep into the recent churn?

Mostly **no** — the heavy new work is sound (see "verified-sound" below). A few new
issues, only one a real correctness bug.

### P1-a — Viewer background subtraction is mode-scoped, not file-scoped (display correctness; **no data impact**) — NEW
`display_controllers.py:209-211` (image: subtract whenever `_bkg.shape == data.shape`)
and `:436-437` (XYE: `intensity - np.interp(radial, bkg_x, bkg_y)`). `_clear_bkg`
(`display_frame_widget.py:2394`) is called **only** on manual toggle (`:2519`) and a
**mode change** (`:2776`) — the docstring (`:2511-2514`) explicitly scopes background to
the mode. There is **no clear on file/scan selection within a mode**. So: Set BG on file
A → select file B of the same detector dims (image) or any grid (XYE) → B is silently
background-subtracted with A's frame. The XYE path is worse — `np.interp` resamples
across mismatched grids with endpoint clamping and **no units/x-overlap check** (a
2θ-grid background subtracts from a q-grid trace). Display-only (never persisted, never
fed to integration), and the button does read "Clear BG" (a visible tell), but it
violates "no silent-wrong-data" on the display. **Fix:** call `_clear_bkg()` on viewer
file/scan selection (mirror the existing `clear_overlay()` in `new_scan`), and add a
units/identity guard to the XYE subtraction (not just a shape check). *(verified)*

### P1-b — No global exception hook under PySide6 → a stray slot error aborts the app — NEW/latent
xdart is PySide6 (`src/xdart/__init__.py:1`) and installs **no `sys.excepthook`** (grep
confirms none). In PySide6 an exception escaping a slot terminates the process. The new
`setBkg` viewer reads (`display_frame_widget.py:2548-2577`) and `update_data`
(`static_scan_widget.py:543`, catches only `AttributeError`) are unguarded; an h5py
error on a reloaded scan would crash rather than log. Cheap, high-value hardening for a
research-preview ship: install a top-level `sys.excepthook` (log + non-fatal dialog) or
guard those slots. *(verified: no excepthook; trigger is latent)*

### P2 — worth fixing, not blockers (all NEW unless noted)
- **Set-Bkg block-reads the whole evicted scan on the GUI thread** (`display_frame_widget.py:2576`,
  `require_all=True, allow_blocking_read=True`) — a multi-second freeze on a 600+-frame
  reloaded scan (the exact sync `.nxs` read the overlay path was just fixed to avoid).
  Correct (no partial-bkg bug) but route through the async path or accept as a one-shot.
- **Overlay/Waterfall double-accumulator** — `display_frame_widget.py:1224` writes the
  legacy `self.plot_data` AND `:1238` carries the new `self._waterfall_history`; both
  retain every row for the whole scan in overlay mode → duplicated (opt-in) memory.
  **Complete the flip:** retire `plot_data`/`update_wf` once the payload path is trusted
  — a memory win *and* removes the dual-path hazard. *(verified)*
- **Overlay/Waterfall unit-lag after Q↔2θ toggle (suspected).** `plotUnit.activated`
  (`display_frame_widget.py:584-586`) is wired to the legacy `update_plot` but **not** to
  `self.update` (the payload path), unlike `imageUnit.activated` (`:578`). If the next
  payload render reaches the accumulator with no resident frames (idle/end-of-scan) it
  re-emits the prior history in the pre-toggle unit. Fix: wire `plotUnit.activated` to
  `self.update`. *(wiring verified; stale window suspected)*
- **Carryover memory (still open, all bounded-but-grow-to-N):** persistent streaming
  session retains every `Frame` for the whole scan (`reduction/core.py:1372`, no evict on
  `_frame_by_index`); per-frame raw `astype(float)` float64 (`core.py:2112`) + 2D cake
  built float64 before float32 downcast (`integrate/single.py:249`, `multi.py:212`);
  `io/image.py` `read_images_parallel`/`read_image_stack(reduce=None)` list-then-stack the
  whole stack (`:268,354`); `ProcessedScan` reopens HDF5 per `get_*` access (`io/read.py:251`
  etc., O(N) for a per-frame loop); in-memory display thumbnail is float32 vs uint8 on disk.
- **`update_wf` throttle lowers repaint frequency but each repaint still re-uploads the
  full N×M stack** (`display_plot.py:919-940`) — cost relocated, not eliminated; fine for
  v1.0, incremental-append is the real fix.

### P3
- `integrate_radial` missing from `integrate/single.py:__all__` (`:364`). *(verified)*
- Standard non-GI `integrate_1d` does not NaN-mask `count==0` while `integrate_2d`/
  `integrate_radial` do (`single.py:99` vs `:158,249`); low risk, doesn't break the spine.
- `WaterfallHistory.reset_key` collides if a new live scan reuses the name before
  `data_file` is set (`display_publication.py:790`) — backstopped by `clear_overlay()` in
  `new_scan`.
- `_wf_y_axis` residual row-id/slice-window length coupling edge (`display_plot.py`).

### Verified-SOUND new churn (no regression — checked, not assumed)
- **count==0 → NaN cake masking** (GI 5526841, std 20bdf50, azimuthal 6cdfa0e): keyed on
  `result.count`, **not** the pyFAI dummy/`-1`; genuine zero-count-but-real bins preserved;
  masks before the 2D transpose; byte-compat signature digests decompressed values with
  `equal_nan` → gate unaffected. Shared `_nan_empty_1d/_2d`, no fork.
- **lz4+shuffle default**: graceful gzip fallback if `hdf5plugin` absent
  (`io/nexus.py:937`); filter id `32004` **is recorded** in the dataset pipeline → a reader
  without the filter fails **loud** (h5py error), never silent-wrong; round-trips
  byte-identically; byte-compat signature is filter-invariant.
- **detector_shape/gaps persisted**: conditional write + `is None`-guarded reads
  (`scan.py:733`, `display_publication.py:338`); old files (absent attr) handled (af64c7f).
- **`IntegrationResult1D` on a `chi_deg` ±180° axis**: `__post_init__` only shape-checks
  (negatives/NaN fine); `to_unit` correctly *raises* for a chi axis (never converted).
- **Universal raw-display "prefer thumbnail"**: display-only; Set-Bkg (Int) pulls full-res
  raw via `get_or_hydrate(include_raw=True)`; no ROI/line-cut/export reads the displayed
  array (ROI feature not built yet).
- **Qt threading bridge (ADR-0004)**: `on_frame_completed`/`_publish_display` on the single
  writer thread; `sigUpdate(int)` set-only + AutoConnection→QueuedConnection across
  threads; no `DirectConnection`; hydration/aggregation workers generation-gate + join.
- **Overlay accumulator invalidation matrix**: new-scan reset, id-set-change REBUILD
  rejection, never-wipe-on-failed-read, row-id alignment — all present and guarded.
- **`display_logic.py` purity** intact after +139 lines (subprocess import-guard test).

**Spine & data integrity:** the live≡batch≡reload spine holds; the only silent-wrong is
the **display** background (P1-a), not any persisted/integrated data.

---

## 3. Performance & memory — bottom line

Markedly improved since 2026-06-15. Remaining open items are mostly *bounded-but-grow-to-N*
(streaming session frame inventory, the doubled overlay accumulator, the O(N×M) waterfall
repaint) plus the long-standing float64 working-set and whole-stack `image.py` loads —
none new-correctness, none blocking for typical scan sizes. Highest-value post-v1 perf
items (from 2026-06-15, still valid): flip `run_reduction`'s default to streaming when a
durable sink is supplied (the biggest notebook win — default is currently serial
`chunk_size=1`); xdart batch submit-per-read (Phase 4e); stream `integrate/batch.py` +
`read_image_stack`.

---

## 4. Expandability / architecture fit (the forward-looking ask)

Assessed against `phase4_scansession_design.md` §5, the ADRs, and the three new design
docs. **Strong overall** — four of five seams are "fill a documented slot."

| Seam | Verdict | The gap (file:line) | Blocks |
|---|---|---|---|
| **1. Analysis-plan** (`analysis/plans.py`) | READY | `run_stitch` eager-materializes w/ MemoryError guard (`:120`); fit plans take arrays not a `source` (`:249,276,322`) → no source-driven "fit over modes" plan | nothing hard (streaming-stitch + fit-over-modes are additive) |
| **2. Source abstraction** (`core/scan.py`, `sources/registry.py`) | MINOR-FRICTION | **`ProcessedScan` lacks `metadata_for`/`frame_for`/`motors`** (`io/read.py:707`), but `_metadata_series` (`plans.py:354`) needs one of them → `run_stitch`/`run_roi_stats` with a `rot1_key`/`x_key` raise on a reloaded `ProcessedScan` (works only via `open_source`→`ProcessedNexusSource`) | stitch + ROI x-axis on reloaded files |
| **3. Display payload model** (`display_logic.py`, `display_publication.py`) | NEEDS-WORK (RSM/ROI panels); READY (stitch single-role, azimuthal) | `DisplayPayload` has only `raw_image`/`cake_image`/`plot` (`display_logic.py:595`); `STITCH_2D`/`SLICE_2D`/`PROJ_1D`/`RESULTS` are reserved enums with no payload field/`build_payload` branch/store resolver. WS-X2 render dispatch is role-level (`:1353`) | RSM viewer (payload + repeated-role render); stitch viewer needs the payload field only |
| **4. FrameRecord / store** (`core/frame_view.py`, `session/`) | READY — **store already built** (phase4 §2-F is stale) | `FrameRecordStore` exists (`session/frame_record_store.py:124`, bounded + hydrator), `ScanSession` accepts `record_store=` (dormant). Real gap: **`get_1d`/`get_2d` have no `mode=` param** (`io/read.py:226,269`) though schema + writer support per-mode subgroups | multi-mode reload equivalence (ADR-0005 gate), fitting-over-modes |
| **5. Schema-as-code** (`io/schema.py`) | MINOR-FRICTION | `stitched_1d/2d` are **written** (`io/nexus.py:1519`) but **not registered** in `SCHEMA`/`CAPABILITIES`; no `diffractometer`/UB group | stitch persistence capability-gating; offline RSM/stitch geometry |

**Top architecture gaps to close before piling on features (ranked):**
1. **Keyed-panel display payload (Seam 3a).** Grow `DisplayPayload` to keyed panels + add
   per-role store resolvers. Deepest gap; unblocks **both** the RSM and stitch viewers.
   The pure-logic half (`PanelKey`, `render_keys_for_state`, `draw_keys`) is already built.
2. **Reader `mode=` param (Seam 4).** Small, high-leverage; unblocks the multi-mode reload
   equivalence test (the recorded Phase-5 gate) and fitting-over-modes. Hard prerequisite
   for the Phase-5 store collapse to be *testable*.
3. **`ProcessedScan` metadata completeness (Seam 2).** Add `metadata_for`/`frame_for` (or a
   `scan_data` fallback in `_metadata_series`). One-method fix; removes a silent
   two-reload-class divergence that affects the ROI + stitching designs.
4. **WS-X2 renderer key-dispatch (Seam 3b).** Promote render from role-level to per-`PanelKey`;
   unblocks RSM's repeated `SLICE_2D`/`PROJ_1D`. Sequence with #1 (both touch the renderer).
   Stitch does **not** need it.
5. **Schema-register stitched + add `diffractometer`/UB group (Seam 5).**

---

## 5. Design-doc reconciliations (update these — the review changed two premises)

- **ROI doc §6.3** stated R3-C (invalid-pixel policy into core) is a *hard prerequisite,
  still open*. **It has since landed** (`reduction/core.py:2132`). Mask-aware ROI stats is
  no longer blocked on it — update the doc to "available; reuse `core.invalid` +
  `plan.mask_saturation`."
- **ROI doc §8 + Stitching doc §2/§5** assume a reloaded **`ProcessedScan`** is the
  FrameSource that `run_roi_stats`/`run_stitch` consume. Today that works for the x-axis/
  motor series only via `open_source`→`ProcessedNexusSource` (Seam 2). Either note that as
  the reload path, or — better — close gap #3 so `ProcessedScan` works directly.
- **phase4_scansession_design.md §2-F/§5** says the headless `FrameRecord` store is net-new
  in Phase 5 (7a). **Stale:** `FrameRecordStore` + `ScanSession(record_store=)` are already
  built and tested (dormant). Phase 5 is now the *collapse/wiring* (delete `data_1d`/`data_2d`/
  `LiveFrameSeries`/`PublicationStore` in favor of the store), not building the store.

---

## 6. Recommended sequencing for v1.0

**Before ship (small):** P1-a (clear background on viewer file-switch + XYE units guard);
P1-b (global `sys.excepthook`). Optionally route Set-Bkg's evicted read async (P2) and wire
`plotUnit.activated → self.update` (P2).

**Complete the in-flight branch:** retire the legacy `plot_data`/`update_wf` accumulator
once the payload flip is live-validated (removes the double-accumulator + the dual-path
unit-lag in one move).

**Architecture, before the queued features pile on (each surgical, high-leverage):** gaps
#2 (reader `mode=`) and #3 (`ProcessedScan` metadata) first — small and they unblock
disproportionate value and the Phase-5 testability; then #1/#4 (keyed-panel payload + WS-X2)
together when the RSM/stitch viewers start; #5 with the stitching feature.

**Post-v1 perf (unchanged from 2026-06-15):** `run_reduction` streaming default; xdart batch
submit-per-read; stream `integrate/batch.py` + `read_image_stack`.

---

*Method note: four parallel subagent reviews (perf/memory, core robustness+regressions,
xdart GUI+thin-xdart, expandability), headline findings hand-verified against cited code.
"Verified" tags above mean I read the cited lines firsthand this round.*
