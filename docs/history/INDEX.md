# History index

Completed-effort records imported into this folder:

- `CC_monorepo_handoff.md` — the stage-by-stage monorepo migration spec
  (executed Jun 2026; outcome in `/MIGRATION.md`).
- `monorepo_plan.md` — migration mechanics (filter-repo, pyproject,
  dependency lists).
- `fix_review_2026-06-10.md` — pre-release fix review (C/S/X item
  verdicts feeding the 1.0 work).

## Archived 2026-06-28 (executed plans / dated reviews / handoff notes)

Moved out of `../design/` after a verified KEEP/ARCHIVE/DELETE sweep (0 deleted —
all preserved here for provenance; the still-relevant content lives in the named
successors):

- `review_2026-06-18_codebase_health.md` — point-in-time codebase-health review
  (branch overlay-waterfall-payload-flip @6cdfa0e); findings landed/superseded;
  current status in `../design/stitching_rsm_build_plan.md` + the feature design docs.
- `review_2026-06-28_gui_streamlining.md` — controls-panel streamlining review;
  fully absorbed by `../design/design_controls_panel_v2_jun2026.md` (CANONICAL).
- `review_2026-06-28_stitch_rsm_corrections.md` — headless stitch/RSM/corrections
  audit (@8addcdd); findings closed in code (`multi.py` / `discover.py` /
  `test_stitch_geometry.py`); verification ledger preserved here.
- `design_streaming_reintegrate_jun2026.md` — D1 reintegrate-memory fix
  (shadow-group + atomic swap); IMPLEMENTED + SHIPPING; record of the design approach.
- `design_stitch_persistent_display_jun2026.md` — stitch as a persistent display
  mode (`Mode.STITCH_1D/2D`); IMPLEMENTED @HEAD; residual P7 panels tracked in
  `../design/stitching_rsm_build_plan.md`.
- `geometry_next_session.md` — geometry session handoff (steps 0-4 done; archived
  2026-06-27); current status in ADR-0007 + `../design/stitching_rsm_build_plan.md`.
- `geometry_step4_next_session.md` — geometry step-4 handoff snapshot; steps 4/4b/5
  now DONE in ADR-0007; live check in `../design/stitching_rsm_build_plan.md`.
- `scan_plotter_next_session.md` — Scan Plotter / ROI session handoff; prompted work
  now DONE in `../design/design_scan_plotter_metadata_roi_jun2026.md`.
- `nexus_stitch_refactor_plan.md` — pre-arch-v2 NeXus/stitch branch plan
  (2026-04-13); SUPERSEDED 2026-06-14 by `../design/design_stitching_jun2026.md`;
  §2 schema landed in code.
- `stitching_design_gui.md` — pre-monorepo xdart stitching GUI design note
  (May 2026; renamed from `stitching_design.md` on archive); SUPERSEDED 2026-06-14
  by `../design/design_stitching_jun2026.md`.

## UI / visual design — `ui/` (Direction A, superseded)

Pure-visual UI artifacts kept separate from the architecture/plan records.  This is
**Direction A** of the xdart redesign — superseded by the chosen path
(`../design/design_controls_panel_v2_jun2026.md` + the external `xdart_controls_handoff`
mockup).  Preserved for visual provenance and for the exact Qt values if the v2 panel
is ever built:

- `ui/XDART_REDESIGN.md` — the Direction-A UI redesign spec.
- `ui/CLAUDE_CODE_FIXES.md` — round-2 Qt/QSS corrective (exact card radius / padding /
  font / token values) for Direction A.
- `ui/xdart_reference_all_modes_dark.html`, `ui/xdart_reference_all_modes_light.html` —
  flat static HTML mockups (all modes) that `CLAUDE_CODE_FIXES.md` targets.

## Pre-monorepo review-cycle archive (not imported)

The two-repo era accumulated ~70 review/plan documents in the
maintainer's `~/repos/review/` folder.  They are working artifacts of
completed cycles — superseded by the docs in `../design/` and by
`/MIGRATION.md` — and are listed here for provenance rather than
imported.  If one turns out to be load-bearing, import it then.

Notable clusters (filenames as in the review folder):

- **Architecture-v2 cycle:** `CC_arch_v2_direction_review_jun2026.md`,
  `arch_v2_remaining_jun2026.md`, `CC_restructure_closeout_jun2026.md`,
  `restructure_plan.md`, `restructuring_plan_jun2026.md`,
  `restructure_codex_oneshot_jun2026.md`.
- **Frame-publication spine (display refactor stages):**
  `unified_frame_publication_plan.md`,
  `frame_publication_stage2..5_review.md`,
  `frame_publication_spine_and_stage5_review.md`,
  `CC_item9_sole_display_contract_design_note.md` (the X1 design note —
  Phase 3 of the current plan executes it),
  `CC_step1..4_*.md` (renderer unification steps).
- **Pre-release stabilization:** `CC_stabilization_fixes_jun2026.md`,
  `stabilization_test_plan_jun2026.md`, `CC_prerelease_gate_jun2026.md`
  (dangling — local-only CC_ note, not published),
  `CC_postreview_plan_jun2026.md` (dangling — local-only CC_ note, not published),
  `CC_codex_residuals_jun2026.md`,
  `stable_release_immediate_fixes.md`.
- **Performance pushes:** `CC_perf_round2_jun2026.md`,
  `CC_perf_4b_sharded_executor.md`, `xdart_speedup.md`.
- **Feature/task specs (executed):** `CC_pause_button_spec.md`,
  `CC_gui_sweep_jun2026.md`, `CC_gui_polish_and_controlstate.md`,
  `CC_data_loss_save_vs_evict_jun2026.md`,
  `CC_freeze_frame_click_live_run.md`, `gi_*_task.md`,
  `nexussink_scan_data_task.md`, `wrangler_disable_during_run_task.md`.
- **Earlier-era reviews (Apr–Jun 2026):** `deep_review_vs_goals_jun2026.md`,
  `cross_repo_review_jun2026.md`, `CROSS_REPO_REVIEW*.md`,
  `repo_review_jun2026.md`, `xdart_code_review.md`,
  `xdart_refactoring_patterns.md`, `data_source_unification_plan.md`,
  `project_goals.md`, `release_runbook.md`, `ROADMAP.md` (superseded by
  `../design/roadmap_2026-06-10.md`).
