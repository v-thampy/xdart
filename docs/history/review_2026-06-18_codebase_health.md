# Codebase health + feature-readiness review (2026-06-18)

**Status:** HISTORICAL REVIEW. Several findings below have since landed or been
superseded; use `stitching_rsm_build_plan.md` and the individual design docs for
current implementation status.

**Method:** 10-agent review workflow (4 regression audits of the last few sessions' major changes + 5 architecture-suitability assessments + a lead synthesis), each agent reading source and citing file:line.  Branch `overlay-waterfall-payload-flip` @ `6cdfa0e`.  Full per-agent JSON in the session transcript.

> Note: the 3 viewer Set-BG "HIGH" findings predate the maintainer's explicit "shape-match-only" decision — see the resolution notes in the follow-up plan; the shape-mismatch no-subtract is intended, the normChannel-restore + reset_key(None) are the genuinely actionable ones.

---

All claims verified. `fitting/` exists (plans.py imports it; the listing is suppressed by the shell quirk but `run_peak_fit` is real). There's a `tests/xdart/test_stitch.py` already (relevant to stitching readiness), a `test_display_logic.py` (purity), and 96 test files total. I have enough verified ground truth to synthesize the report.

---

# Lead Review: xrd-tools post-session health + feature-readiness

Branch `overlay-waterfall-payload-flip` @ `6cdfa0e`. Synthesis of 4 regression audits + 5 architecture assessments, cross-checked against source.

## 1. Executive summary

The codebase is **healthy**. Three of four regression audits returned clean (overlay/waterfall flip, 2D-cake NaN-empty fix, the 15-commit branch sweep with 1064 core tests passing); the only audit with real bugs is the **Raw-Display Background / Set-BG viewer feature**, which carries 3 high-severity but *narrow* defects — all in viewer Set-BG edge paths, none in the reduction spine or the live≡batch≡reload equivalence gate. The architecture is **very well-suited** to the planned features: the §10 display seams (reserved `STITCH_2D`/`SLICE_2D`/`PROJ_1D`/`RESIDUAL_1D` roles, the controller registry, the layout descriptor) plus the publication store and the now-landed headless `session/` layer mean azimuthal/stitching/fitting/phase4 are **additive** (Mode enum + a pure `*_logic.py` + a registered controller), with no core refactor required. The two features that need real new headless authoring are **geometry** (no canonical `Diffractometer` class yet — two parallel encodings must be unified) and **RSM** (needs the deferred WS-X2 multi-instance-panel promotion to render its 2×3 grid). **Fix the 3 Set-BG highs first** (they ship a misleading "Clear BG" button that silently does nothing), then proceed to geometry → stitching as planned.

## 2. Regression findings

**3 clean audits, 1 with real bugs.** The overlay/waterfall flip, the GI + non-GI 2D-cake NaN-empty fix (47 tests green, `count==0` discriminator correct, shared `_nan_empty_2d` helper, equivalence spine uses `equal_nan=True`), and the full 15-commit branch sweep are all **clean** — I verified the count-based masking and shared-helper claims hold. Only the Set-BG feature has real findings.

| Sev | Finding | file:line | One-line fix |
|---|---|---|---|
| **HIGH** | **Shape-mismatched BG silently skips subtraction — "Clear BG" button lies.** The publication path got `_bg_for_image_shape` resize (7b23f5c); the Image-Viewer path never did. | `display_controllers.py:210-211` (vs. fixed path `display_publication.py:318`) | Apply the `_bg_for_image_shape` resize logic to the viewer path, **or** at minimum log a shape-mismatch warning so the no-op is visible. |
| **HIGH** | **`normChannel` not fully restored on Image-Viewer→Int return.** Else branch re-shows the widget but never calls `_apply_layout`, so stale `frame_4` geometry can leave it hidden. *(Verified: lines 2857-2879 toggle visibility but do not rebuild layout.)* | `display_frame_widget.py:2857-2879` | Call `_apply_layout(layout_mode)` in the normal-mode else branch. |
| **HIGH** | **XYE BG interpolated onto mismatched grids may distort.** `_set_bkg_xye_viewer` stores BG at the *first* file's grid; coarse/fine mismatches smear or leave stale background. | `display_frame_widget.py:2463-2486`, render at `display_controllers.py:436` | Store BG on a common grid, warn on mismatch, or compute per-file BG. |
| MED | Overlay `reset_key=(scan_id, needs_2d)` collides when `scan is None` → accumulator not reset across different scan-less loads. *(Verified: scan_id is None when scan is None, lines 789-792.)* | `display_publication.py:789-792` | Add finer scan identity (file path / dataset id) or explicit source-change reset. |
| MED | Set-BG with empty selection silently no-ops (no dialog), unlike the partial-2D path which warns. | `display_frame_widget.py:2515-2516` | Add a "select frames first" status/dialog. |
| MED | Publication evict/hydrate not synchronized with Set-BG capture → BG can mismatch displayed frame after scroll-away/back. | `display_frame_widget.py:2418-2437` | Capture+revalidate frame label, or defer Set-BG until selected frames are resident. |

Lower-severity (button-width magic `scale=1.177`, redundant NeXus double-hide) are real but cosmetic/defensive — fix opportunistically. The remaining ~14 "nit" entries in the cake audit are *confirmations of correctness*, not defects.

**LOUD callout:** the three HIGHs all live in the viewer Set-BG flow and converge on one UX lie — a visible **"Clear BG"** state while no subtraction is happening. They do not touch the reduction core, the writer validators, or the equivalence spine, so they are not release-blocking for the reduction path, but they should be fixed before the Set-BG feature is presented as done.

## 3. Architecture readiness (per feature)

Verified ground truth: `Mode` enum has only the 5 shipped values (`display_logic.py:102-107`); roles `STITCH_2D`/`SLICE_2D`/`PROJ_1D`/`RESIDUAL_1D` are reserved (247-250); `register_controller`/`controller_for` exist (623, 630); headless `run_stitch`/`run_rsm`/`run_peak_fit` + `StitchPlan`/`PeakFitPlan` all exist in `analysis/plans.py`; `write_stitched`/`read_stitched` exist (`nexus.py:1832/2952`) but are **not** in `schema.py`; `integrate_radial` landed (`single.py:105`); `session/` has `scan_session.py` + `frame_record_store.py`.

| Feature | Rating | Single most important gap | Recommended first step |
|---|---|---|---|
| **Azimuthal (I vs χ)** | **Ready / largely landed** | Headless `integrate_radial` shipped (6cdfa0e, `single.py:105`); only display wiring of Mode-A profile remains. | Wire the χ-profile into a `PlotPayload` trace; smallest of the set. |
| **Geometry (unified Diffractometer)** | **Minor extension, but the heaviest authoring** | **No canonical `Diffractometer` class** — two parallel encodings (`DiffractometerConfig` line 31, `DiffractometerGeometry` line 119) risk drift. | Author `Diffractometer` + presets + `to_pyfai_per_frame`/`to_qconversion` **and the consistency test as a merge gate** (design §5.1/§6 step 0). |
| **Stitching** | **Ready (headless foundations landed)** | At review time, `stitched_1d/2d` were not registered in `schema.py`; this is now fixed. Update (2026-06-28): the persistent Stitch display has since landed — `StitchDisplayController` is registered for `Mode.STITCH_1D/STITCH_2D` and `_live_mode()` returns those when `scan.stitched_*` exists, so the "deferred GUI viewer" caveat is now partly down (Refine button + GI-stitch panels remain P7). | See `stitching_rsm_build_plan.md` for current backend/schema status. |
| **Fitting** | **Ready** | No `Mode.PEAK_FIT` + no `fit_logic.py`; everything else (lmfit `run_peak_fit`, `Trace.kind='fit'/'component'/'background'/'residual'`, reserved `RESIDUAL_1D`, `ResultsView` stub) is in place. | Add `Mode.PEAK_FIT` + a pure `fit_logic.py` emitting fit/residual traces; UI later. |
| **RSM** | **Ready (headless), moderate display work** | **WS-X2 multi-instance panel promotion (#69) still deferred** — the 2×3 repeated-role grid needs it (a role-level fallback loop exists as a bridge). | Add `Mode.RSM_VIEWER` + `PANEL_LAYOUT`, verify the 2×3 layout via a *headless* `compute_display_state` test **before** wiring reduction. |
| **Phase 4 session** | **Complete minus 1 ADR + live checkpoint** | Three already-implemented Phase-4f decisions need an ADR; QThread-teardown / disk-read-during-pause need a **manual live checkpoint** (not offscreen-simulable). | Write the ADR (~20 min), then schedule Vivek's live checkpoint before v1.0. |

**Seams that already support the work well:** the controller registry + reserved roles + layout descriptor make fitting and stitching nearly pure-additive; the publication store cleanly carries multi-mode `FrameRecord` results; the headless `session/` layer (`ScanSession`, `FrameEvent`, `FlushPolicy`) is landed and offscreen-green. **What needs real work:** geometry (new canonical class + cross-adapter consistency test), RSM display (WS-X2 promotion), and stitching schema registration.

## 4. Cross-cutting observations

- **The Set-BG feature is the lone soft spot** and it has a single root cause repeated three ways: the viewer Set-BG paths were added *after* the publication path got its robustness work (resize, shape handling) and never caught up. They diverged. A shared `_subtract_if_shape_matches` / `_bg_for_image_shape` helper used by *both* paths would collapse all three highs.
- **Store/session duality is deliberate and acknowledged.** `data_1d`/`data_2d` are explicitly "internal hydration mirrors" pending the live-gated Phase-5 projection flip; `PublicationStore` is generation-aware and store-first. This is debt-by-design, not rot — but the medium Set-BG evict/hydrate finding shows the duality leaking into feature code, so new features should read store-first and avoid capturing from the mutable mirrors.
- **Purity guard is healthy.** `display_logic.py` stays Qt-free with reserved seams; a purity test exists (`tests/xdart/test_display_logic.py`); the §10 docstrings keep the core module-agnostic.
- **Live-gated debt (A3/A4 Role-A deletion, Phase-B projection flip) remains parked but not blocking.** 43 items "landed & verified" offscreen; the flip awaits a live checkpoint. Don't let geometry/stitching work entangle with it.
- **Test coverage is strong on headless cores** (1064 core tests, 47 cake tests, 24 session + 17 flush + 13 record-store) **but thin on the Set-BG viewer UX** — none of the three highs was caught by a test. Add offscreen controller tests for Set-BG shape-mismatch + mode-return layout.
- **Recurring discipline that's working:** count-based empty-bin masking (not value-based), `equal_nan=True` equivalence comparisons, unidirectional `single.py→gid.py` imports, generation-keyed accumulators — all verified correct.

## 5. Recommended sequencing

**Fix-first (before new branch):** the **3 Set-BG highs**, ideally via one shared shape-handling helper used by both the publication and viewer paths (`display_controllers.py:210-211`, `display_frame_widget.py:2857-2879`, `:2463-2486`), plus a regression test. Low effort, removes a user-visible lie. Optionally fold in the two cheap mediums (empty-selection warning, scan-None reset_key).

**Then, geometry + stitching on the new branch (max reuse, min risk):**
1. **Geometry first.** It is the shared dependency: `run_stitch` should derive per-frame rotations from `scan.geometry.to_pyfai_per_frame(motors)`, and RSM's `to_qconversion` needs the same canonical object. Land `Diffractometer` + presets + both adapters **gated on the consistency test** (design §6 step 0). Doing this first means stitching and RSM consume the finished adapter instead of re-deriving geometry three times.
2. **Stitching, in parallel where possible.** Historical note: schema registration has since landed. Remaining GUI work is `Mode.STITCH_VIEWER` + `PANEL_LAYOUT` + `StitchViewerController` (reads pre-loaded result first), plus xdart range-syntax grouping. Single `STITCH_2D` role means **no WS-X2 dependency** — stitching ships before RSM.

**Later (fitting / RSM):**
3. **Fitting** is the lowest-risk addition (headless `run_peak_fit` is prod-grade, all traces/roles reserved) — slot it whenever convenient; it doesn't gate anything.
4. **RSM last** of the new modes: it depends on geometry (`to_qconversion`) *and* needs WS-X2 promotion (#69) for its repeated-role 2×3 grid. Do the headless layout test before wiring reduction.
5. **Phase 4**: write the ADR now (20 min); schedule the live checkpoint independently before v1.0. Keep the Phase-B store→session projection flip parked post-v1.0.

**Risk-minimizing rule:** geometry's consistency test is the linchpin — make it a hard merge gate so the two adapter views can never silently drift, which is the exact failure mode the unification exists to kill.

---

## Maintainer resolution (verified post-review, 2026-06-18)

The 3 viewer Set-BG "HIGH" findings were re-checked against source + the maintainer's
explicit decisions this session; **none is a genuine must-fix bug:**

- **normChannel-restore → FALSE POSITIVE.** `set_viewer_display_mode` calls
  `_apply_layout(layout_mode)` at `display_frame_widget.py:2808` (BEFORE the mode
  branches) — for the Int return `layout_mode` is INT_1D/INT_2D with
  `frame_4_vis=True` — and the normal-mode else branch explicitly re-shows
  `normChannel` at `:2859`. Verified; viewer→Int round-trips restore it. (The audit
  missed both.)
- **Shape-mismatch silently skips → INTENDED.** The maintainer explicitly chose
  "check shape and only subtract if shape matches" (no resize of a possibly
  incompatible background). In the Image Viewer the bg + the displayed frame use the
  SAME source preference, so they match in practice; the only mismatch is full-res-bg
  vs thumbnail-display (same detector, different resolution) — a deliberate trade-off,
  not a defect. Re-adding the resize would contradict the decision.
- **XYE interp on mismatched grids → ACCEPTED approximation.** Per-file interpolation
  of the stored bg is the chosen design; XYE files from one instrument share a grid in
  the common case.

**Genuinely-open (minor, deferred — not blocking):** the `reset_key=(None, …)`
collision when `scan is None` (`display_publication.py:789-792`) is theoretical
(Overlay/Waterfall accumulation only runs in Int modes, which carry a scan); the
empty-selection silent no-op + the evict/hydrate store-first hygiene are minor UX.
Logged here for a later pass.

**Net:** the last few sessions' work is clean; proceed per §5 sequencing
(**geometry first** as the shared dependency, then stitching, then fitting/RSM).
