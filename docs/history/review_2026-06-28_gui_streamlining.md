# Review 2026-06-28 — GUI design: streamlining the controls panel

**Scope:** how the controls-panel design (the `xdart_controls_handoff/` mockup + the GUI design
docs in `docs/design/`) can be **simplified and unified**. Read-only review; no code/doc edits.
Grounded against the current `feature/geometry` GUI (today = a 5-pane `QSplitter` with
`ParameterTree` wranglers; the 4-section card panel is **mockup-only, unbuilt**).

**Verdict.** The design is strong and its *pieces* are code-aligned (the PROJECT/DATA grouping,
the `sigUpdateGI→set_image_units` reactivity, shared `StaticControls`, `ScanSourceWidget`, the
Stitch display controller all exist). The streamlining opportunity is **(A) collapse the design
to ONE canonical model** — several artifacts have drifted — and **(B) a handful of UX
simplifications** that reduce clicks and "why is this greyed?" friction. The single biggest win
is **one control surface for all tools**, not a legacy-Int panel beside a new Stitch/RSM panel.

---

## A. Reconcile to one canonical model (kill the drift first)

The mockup (Jun 28 19:19) is the newest artifact and is code-aligned; treat it as canonical and
fix the docs to match. Four concrete drifts:

1. **Section count — 4, not 3 (RESOLVE to 4).** The brief + three-section doc say *3* (Project
   folded into Data); the mockup + the live ParameterTree say *4* (**PROJECT** separate from
   **DATA**). PROJECT (folder/save-path = workspace, persists across runs) genuinely has a
   different lifetime than DATA (per-run input), so 4 is the better model *and* it's already in
   code (`image_wrangler.py` PROJECT/DATA groups). **Action:** renumber the docs' "section
   2 = instrument / section 3 = plan" language to the 4-section map (EXPERIMENT = §3,
   PROCESSING = §4) so cross-references stop pointing at the wrong number.

2. **Transmission is dropped (RESOLVE to Standard/Grazing only).** Only the two oldest docs
   mention a Transmission measurement mode; ADR-0008, the mockup enum, and the code (a single
   Grazing `bool`) all have just Standard/GI. **Action:** strike "transmission" from
   `design_brief` §5 and `design_gui_three_section_layout` §2c.

3. **Mask belongs to the Detector (3b), not DATA.** The mockup puts Mask in DATA; the headless
   model (`DetectorCalibration`/`Detector_config`) and the docs put it in the detector. Mask is
   *instrument state persisted with the geometry*, so DATA breaks the "sections mirror the data
   model" invariant that makes reload symmetry free. **Action:** Mask → 3b Detector; drop the
   duplicate (it's also in the Tools row). Keep one home.

4. **Two redesign directions exist in the repo** — `XDART_REDESIGN.md`/`Redesign A.dc.html`
   (conservative, reorganizes *today's Int* UI into 4 cards) vs `xdart_controls_handoff/` (the
   reactive Stitch/RSM panel). They share a visual language but are *different specs*.
   **Action:** declare the handoff the target; archive Direction A or merge its Int-card mapping
   into the handoff as the "Int page" (see B1).

---

## B. Streamlining recommendations (highest leverage first)

### B1. One control surface: `Tool = {Int 1D, Int 2D, Stitch, RSM}` — *the* win
Today Int lives in the legacy Mode combo and Stitch/RSM are display-only; the mockup adds a
**separate** Stitch/RSM Tool dropdown. That yields **two parallel control idioms** in one app —
the opposite of streamlined. `design_wrangler_organization` §2 already calls for a *single*
`Integrate 1D / Integrate 2D / Stitch / RSM` selector. **Make the 4-section panel the one panel
for every tool**, with PROCESSING swapping per tool (Int's PROCESSING page = today's integrator
params). The Int migration is already designed + sized MODERATE (~6.5–13 h,
`design_gui_int_migration`). Doing it means **one mental model, one reload path, one place to
learn** — and it retires the legacy wrangler+integrator split. *If* Int can't migrate
immediately, at least put all four tools in the *same dropdown* now (Int pages = the existing
widgets embedded) so users never meet two different control surfaces.

### B2. Inline the load-bearing *status*; popup the editing
The mockup minimizes 3a/3b/3c to one inline control (a *name*) + a "More…" popup holding
everything else — including the **calibration summary** (dist / beam-centre / fitted-vs-preset).
But geometry "is where stitch/RSM live or die" (wrangler-org §3.2): the calibration summary is
exactly the at-a-glance trust signal users want *before* they hit Run. **Recommendation:** the
inline primary should be a **status glance**, not just a name — e.g. 3b inline shows
`Eiger1M · 200.4 mm · fitted ✓` with the badge, and "More…" is for *editing*. Same for 3a
(show the preset *and* whether the circle→motor map auto-resolved). This keeps the popups for
decluttering without burying the one thing a calibration-heavy flow needs to see.

### B3. Make cross-field dependencies *actionable*, not puzzles
The Stitch-GI page greys the GI-corrections group with a "⚠ requires Merge = Histogram
(currently Multi-geometry)" hint. A greyed group + a cross-field rule is a puzzle the user must
solve. **Streamline:** enabling any GI correction (or any shared `CorrectionStack` toggle)
should **auto-switch Merge to Histogram** (or offer a one-click "switch & enable"), because the
headless truth is that `multigeometry` *ignores* the shared pre-weight. Make the correct path
the default path. Likewise, a GI correction needs section-3↔section-2 data (material, energy):
if material is unset, surface an inline "set sample material (3c) to enable" affordance rather
than a silent/greyed control.

### B4. Badge economy — show provenance only when it carries signal
Four badge states (AUTO/FILE/SET/SAVED) on *every* field is visual noise on a dense panel.
**Recommendation:** badge only when provenance *deviates from the expected default* — FILE (a
loaded calibration), SET (a user override of an AUTO value), SAVED (restored from `.nxs`). A
field that auto-inferred as expected can stay quiet (or carry a single subtle dot). The point of
the badge is to flag "this isn't the obvious source," not to label every row.

### B5. Flatten the disclosure depth
The mockup stacks four disclosure layers: section collapse → sub-group (3a–3d) → "More…" popup →
dialog fields. For an expert tool that's a lot of hide-and-seek. **Recommendation:** since
EXPERIMENT's sub-groups are *already* minimal (one inline + popup), they don't *also* need to be
independently collapsible — let the EXPERIMENT section collapse as a whole and keep 3a–3d as
fixed rows. Reserve collapse for the four top sections only.

### B6. One coherent mode model, not two interacting controls
There are two "mode" controls: the top **Tool** dropdown and the **Measurement-mode** segmented
control in 3c — and the mockup quietly couples them (`setGrazing` forces `tool=stitch`; RSM greys
Measurement-mode). The real state space is just **{Int, Stitch-Standard, Stitch-GI, RSM}**.
**Recommendation:** keep the two controls *only if* the orthogonality is real and visible;
otherwise model the four states directly and avoid one control silently flipping the other (a
hidden side-effect is harder to learn than an honest 4-way selector). At minimum, never let
`setGrazing` mutate the Tool without showing it.

### B7. Don't drop the shipped run controls
The mockup's run bar is `Run / Pause / Stop` only; the live `StaticControls` also has
**Batch / Cores / Live / Append (write-mode)**. These are real, in the controls strip (correct
home). **Action:** the implementation must keep them — the mockup is a *styling* reference, not a
control inventory; flag so they aren't lost.

---

## C. Process (these make the design self-correcting)
- **Notebook-as-spec (already in the three-section doc — keep it).** A refreshed notebook that
  reads `load data → configure instrument → build plan → run → display → persist` *is* the
  section order; where the notebook feels awkward, the panel will too. Refresh notebooks as
  **step 1** of the GUI build, not after.
- **Reuse, don't rebuild.** DATA should embed the existing `ScanSourceWidget` (today used only by
  the ROI plotter), not a new widget. The Tools row already has Calibrate + Make-Mask; **Refine
  is unbuilt** — it's the `refine_goniometer` wrapper (diffractometer §3.4) and is the one new
  Tools-row launcher to add.

---

## D. Priority
1. **B1** — unify the tool surface (one panel, `Tool ∈ {Int1D,Int2D,Stitch,RSM}`). Biggest
   simplification; everything else is detail on top of it.
2. **A1–A3** — reconcile section-count / transmission / mask so the docs stop contradicting the
   mockup + the headless model.
3. **B2 + B3** — inline calibration status; auto-switch backend for corrections. The two changes
   that most reduce mistrust + friction in the stitch/RSM flow.
4. **B4–B6** — badge economy, disclosure depth, mode-model coherence (polish).
5. **C** — notebook-as-spec + ScanSourceWidget reuse + the Refine launcher, alongside the build.

*(Open decisions genuinely needing Vivek: B1 — does Int join the dropdown now or stay legacy
until its migration; B6 — two mode controls vs one 4-way. The rest are mechanical reconciliations
to the newest/code-aligned artifact.)*
