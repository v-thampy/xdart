# Design: Controls Panel v2 — Project / Source / Experiment / Processing

**Status:** CANONICAL (2026-06-28) · the single source of truth for the xdart controls panel.
Synthesizes the `xdart_controls_handoff` mockup (visual + interaction direction), the Codex
architecture review (the typed reactive layer + retire `ParameterTree`), and the two streamlining
reviews (`../history/review_2026-06-28_gui_streamlining.md` + the wrangler-org doc). Supersedes — and
absorbs the content of — the earlier 3-section controls design + brief (now retired): their
vocabulary/section model and the Transmission mode are replaced here.
**Gated on:** the headless seams (`Diffractometer`, `CorrectionStack`/GI, the stitch/RSM plans,
`ScanSourceWidget`, `discover_scans`) — all shipped. P7 work; the Qt panel must not *expose* the
GI / `xu_hist` knobs until their real-data convention gates land (build plan LIVE CHECKLIST).
**North star:** thin GUI over reusable headless logic; the panel is a *renderer* of a Qt-free
state model, so stitching/RSM/fitting expansion doesn't multiply state-sync bugs.

---

## 1. Canonical vocabulary — four sections, each = a headless object

Rename the mockup's overloaded "Data" → **Source** (maps 1:1 onto `FrameSource`). Tools and Run
controls live **outside** the sections.

| # | Section | Headless object | Lifetime | Holds |
|---|---|---|---|---|
| 1 | **PROJECT** | session / project-root (N1) | workspace, across runs | project root · save path · output naming |
| 2 | **SOURCE** | `FrameSource` / `CompositeFrameSource` | per-run input | source kind · file/folder (+`discover_scans`) · scan group (chips, combined via `CompositeFrameSource`) · raw/mask **reachability status** · motor→role map |
| 3 | **EXPERIMENT** | `Diffractometer` + `DetectorCalibration` + `GISettings`(sample facts) + beam/UB | the instrument/sample — **persisted once, reload-restored** (`/entry/diffractometer`) | 3a Diffractometer · 3b Detector (PONI/orientation/**mask value**) · 3c Sample & measurement (mode, material, αi, UB) · 3d Beam (energy↔λ, polarization *plane*) |
| 4 | **PROCESSING** | the `*Plan` (`ReductionPlan`/`StitchPlan`/`RSMPlan`) + GI run choices | per-run provenance | ranges · bins · axes/merge-backend · corrections (incl. GI toggles) |
| — | *Tools* (instrument producers) | write §3 state | actions | Calibrate→`DetectorCalibration` · **Refine**→`Diffractometer` · Make-mask→detector mask |
| — | *Analysis* (post-reduction popups) | read the reduced/loaded scan | launchers | **Peak Fit** · **Phase Fit** · **Plot Metadata** (metadata + image-ROI stats vs frame) · *(future)* **sin²ψ strain** |
| — | *Run controls* | run control | — | Run/Pause/Resume/Stop · **Batch · Cores · Live · Write-mode** (keep all four) |

This is the load-bearing invariant: **§3 is the editable view of the persisted instrument
record**, so load/save + the `headless ≡ reload ≡ live` equivalence fall out for free. Any field
placed against its headless home (not for visual convenience) keeps that invariant intact.

### 1.1 Tool taxonomy — three kinds, plus the popups

The panel hosts **three** distinct tool categories; only the first lives *inside* the four sections:
- **Reduction tools** (the Processing Tool selector, §4): Int 1D/2D · Stitch · RSM — the reactive
  four-section pages.
- **Instrument producers** (a Tools row): Calibrate · Refine · Make-mask — they *write* §3 state
  (results land in the 3a/3b summaries with a `FILE`/`SET` badge), not transient dialogs.
- **Analysis popups** (a separate launcher row): **Peak Fitting** · **Phase Fitting** (the "Plot Fit"
  family, `peak_fit_dialog`/`phase_fit_dialog`) · **Plot Metadata** (`scan_plot_dialog` — scan
  metadata + image-ROI statistics vs frame) · *(future)* **sin²ψ strain** (`analysis/strain.py`
  exists headless; no GUI yet). These are **lazy, single-instance, non-modal** dialogs that consume
  the *reduced / loaded* scan (a 1D/2D stack, or raw frames for ROI) — **not** the raw-reduction
  config — so they stay popups, not four-section pages. Each is a **thin GUI over a headless
  primitive** (`fit_peaks`/`PhaseFitter`, `roi_stats`/`run_roi_signals`, `sin2psi`/`strain`); Peak +
  Phase already share one batch worker + the vs-frame trend plot.

**Contract (the thin-GUI seam — enforceable).** Each analysis popup must delegate its computation to
a named `xrd_tools.analysis` primitive — **no reduction / fit / ROI math in the dialog itself**. The
per-tool mapping (dialog → exact function + signature) is **owned by that tool's own design doc**
(`design_roi_stats_plotting`, `fit_advanced_options_catalog`, `design_scan_plotter_metadata_roi`) and
is deliberately *not* restated here, so it stays a single source of truth as those APIs churn. The
illustrative names above are pointers, not the authoritative list. This same rule already holds for
the reduction tools (Int/Stitch/RSM → the `*Plan` + `run_*` primitives) and is what keeps xdart thin.

**Their enable-state is part of the keystone, not ad-hoc.** Add
`ControlProfile.analysis_launchers` (§2): Peak/Phase Fit live only with a 1D result; Plot Metadata
with a loaded scan; ROI stats with raw **reachable**; sin²ψ with its strain inputs present. A greyed
launcher carries the reason (the `run_blockers` tooltip pattern) — **no analysis button is ever dead.**

**Edit / inspector popups fold into the new model** (no new top-level surface):
- the GI-options floating popup → the **3c "More…"**; the detector-options popup → the **3b "More…"**.
- the **Advanced integration settings** dialog (`_show_integration_advanced`) → stays as the
  `ParameterTree` **inspector** — the one place the tree survives (§7).
- the frame-metadata popup (`_open_metadata_dialog`) → a **Frames/metadata readout** off SOURCE.
- the **DISPLAY popups** — the contributing-frame raw picker and the RSM 3D scatter — are owned by
  `design_gui_display_panels_jun2026.md`; referenced here, not duplicated.

Keeping analysis as popups (not crammed into the four sections) is itself a decluttering win, and the
headless-primitive split keeps them on the thin-GUI north star.

---

## 2. The keystone — a Qt-free `ControlState → ControlProfile` layer

Do **not** implement this as a styled `ParameterTree`. Build a typed, reactive model — the same
pattern as the shipped display refactor (`display_logic.py`: a Qt-free profile + a controller
registry that Qt renders). One pure function turns the typed inputs into a render description:

```python
build_profile(state: ControlState) -> ControlProfile
```

```python
@dataclass(frozen=True)
class ControlState:                       # all typed inputs the user/data set
    tool: Tool                            # INT_1D | INT_2D | STITCH | RSM
    measurement_mode: MeasMode            # STANDARD | GRAZING
    source_caps: SourceCaps               # probed: has_motors, is_multiscan, n_frames, kind,
                                          #         raw_reachable, mask_reachable
    result_caps: ResultCaps               # loaded/reduced outputs: has_1d, has_2d, has_raw,
                                          #         has_scan_data, has_rsm, unit/kind hints
    geometry: GeomState                   # calibrated? fitted? detector known? motor map resolved?
    fields: Mapping[FieldId, FieldValue]  # current values + their provenance

@dataclass(frozen=True)
class FieldStatus:
    provenance: Provenance                # AUTO | FILE | SET | SAVED
    state: FieldState                     # OK | MISSING | CONFLICT

@dataclass(frozen=True)
class ControlProfile:                     # the render description (Qt reads this, nothing else)
    sections: Mapping[Section, SectionVis]        # visible? collapsed-default?
    fields: Mapping[FieldId, FieldStatus]         # per-field provenance + OK/MISSING/CONFLICT
    processing_page: ProcessingPage               # int1d | int2d | stitch_std | stitch_gi | rsm
    valid_modes: frozenset[MeasMode]              # RSM → ∅ (grey the mode control)
    backend_required: str | None                  # GI corrections → "pyfai_hist" (auto-select/flag)
    can_run: bool                                 # no unresolved MISSING/CONFLICT
    run_blockers: tuple[str, ...]                 # the reasons (Run-button tooltip)
    analysis_launchers: tuple[AnalysisLauncherSpec, ...]  # post-reduction popups (§2.1)
```

**Why this is the keystone:** every streamlining rule below becomes a *declarative* output of
`build_profile`, not hand-wired Qt signals — and it's **unit-testable now**, Qt-free, dead-but-
proven, decoupled from both the rendering and the still-pending headless GI gates.

### 2.1 Analysis launcher specs — make popups first-class without making them pages

The fitting/plotting tools are not Processing pages because they do not create raw reductions; they
consume the loaded/reduced scan. But they still need one typed surface so new popup tools do not grow
one-off enable logic.

```python
@dataclass(frozen=True)
class AnalysisLauncherSpec:
    tool: AnalysisTool                 # PEAK_FIT | PHASE_FIT | SCAN_PLOT | ROI_STATS |
                                       # SIN2PSI_STRAIN | TEXTURE | USER_PLUGIN
    label: str
    enabled: bool
    reason: str | None                 # tooltip when disabled
    entry_point: str                   # importable Qt dialog factory, lazy-imported
    required_caps: frozenset[ResultCap]
    optional_deps: frozenset[str]      # "fitting", "rsm", "viz", ...
    singleton_key: str                 # one non-modal instance per scan/window
```

`build_profile` owns launcher availability:

| Launcher | Enable when | Consumes | Headless primitive |
|---|---|---|---|
| **Peak Fit** | active/loaded 1D pattern | current 1D trace or frame stack | `fit_peaks`, `PeakFitPlan`, `PeakFitAnalyzer` |
| **Phase Fit** | 1D pattern + fitting deps | 1D trace/stack + CIFs | `PhaseFitter`, `PhaseFitPlan`, `PhaseFitAnalyzer` |
| **Scan Plot / Metadata** | `scan_data` or metadata table exists | per-frame columns | `ProcessedScan`/`FrameSource` metadata readers |
| **ROI Stats** | raw frames reachable, never thumbnail-only | raw frames + ROI specs | `RoiSignal`, `run_roi_signals` |
| **sin²ψ Strain** | 1D peaks + psi/tilt metadata | peak positions vs psi | `Sin2PsiPlan`, `run_sin2psi`, `strain` helpers |
| **Texture / Preferred Orientation** | 1D/2D result + phase/texture config | fit/phase or cake-derived signals | `analysis.texture` / future texture plan |

This table is the controls-panel inventory only. Each dialog's detailed fields remain in the tool's
own design doc (`fit_advanced_options_catalog`, `design_scan_plotter_metadata_roi`,
`design_roi_stats_plotting`, future strain/texture docs). The controls panel must never duplicate
their parameters; it only answers "can this tool launch, and why not?"

**Popup lifecycle contract.** All analysis popups are lazy, single-instance, non-modal dialogs. They
receive a small immutable `AnalysisContext` (scan handle, selected frame labels, current 1D/2D
payload accessors, source/raw reachability, cancellation hooks) and do their compute through a shared
worker contract (`BatchAnalysisWorker` / future `AnalysisRunner`). No dialog reads private wrangler
state directly, and no dialog performs domain math inline.

---

## 3. The four sections (inline-status / popup-edit; drift reconciled)

Each EXPERIMENT sub-group shows **one inline summary line** + a **More…** popup for editing — but
the inline line is a **status glance, not a bare name** (geometry is where stitch/RSM live or die;
don't bury it). Examples:
- **3a Diffractometer** — inline: `psic · motors auto ✓` (preset + did the circle→motor map
  resolve). More…: circle→motor map, sign convention, reference frame.
- **3b Detector** — inline: `Eiger1M · 200.4 mm · fitted ✓` (name + distance + calibrated/fitted
  state). More…: calibration summary card + **Calibrate**, type/shape, orientation, **mask value**.
- **3c Sample & measurement** — inline: the **Standard / Grazing segmented control** (a proper
  segmented toggle, *not* a group-header checkbox — kills the #56 repaint class). More…: sample
  name, material, **Incidence** (grazing only), **UB matrix** (RSM only).
- **3d Beam** — fully inline (energy↔λ linked, polarization **plane**). No popup.

**Drift reconciled (the "one source of truth"):**
- **Transmission is dropped** — modes are Standard/Grazing only (ADR-0008 + code agree; the old
  three-mode docs are historical).
- **Mask:** the **value** lives in 3b Detector (it's `DetectorCalibration` state, reload-restored);
  **mask/raw reachability** is reported as a **status** in SOURCE. One editable home (Detector),
  one status readout (Source) — no duplicate field. (The mockup's "mask in Data" is superseded.)
- **Polarization** has two homes by design: the *plane* is 3d Beam (§3, instrument), the *factor*
  is a §4 Processing correction. Same word, correctly two fields.
- **Section count is 4** (Project separate from Source) — matches the lifetime split and the code.

**Progressive-disclosure & single-source rules (the profile drives these, not the widgets):**
- **Energy is ONE field.** The three entry points — `RSMPlan.energy`, `GICorrectionStack.energy_eV`,
  and the calibration **wavelength** — collapse to the single 3d Beam widget bound to the calibration
  wavelength (ADR-0009). Divergence surfaces as a `CONFLICT` badge (§5), never a second user input.
- **Auto-collapse the hydrated instrument.** When EXPERIMENT hydrates entirely from a `.nxs` (all
  fields `SAVED`), `build_profile` sets its `collapsed-default` — a restored instrument is rarely
  re-edited. Progressive disclosure driven by the **value model**, not a tree constraint.
- **Headerless-`.raw` params** (`detector_shape` / `dtype` / `header_skip`) live behind an
  *Advanced* disclosure in SOURCE — TIFF/EDF/CBF/Eiger auto-detect, so most users never see them.
- **Tools are producers, not dialogs.** Calibrate / Refine / Make-mask write their results back into
  the 3a/3b summary lines (with the right `FILE`/`SET` badge) — the user sees the calibration appear
  in 3b after Calibrate, not vanish into a transient dialog.

---

## 4. Reactive PROCESSING — a `QStackedWidget` keyed by the profile

`processing_page` selects the page; nothing in Qt branches on tool/mode directly.

| Tool | Mode | Page | Shows |
|---|---|---|---|
| Int 1D/2D | Standard | `int1d`/`int2d` | ranges · npt(s) · unit · method · corrections |
| Stitch | Standard | `stitch_std` | ranges · bins(1D/2D) · output axis + **merge backend** · corrections (SA/pol/air) |
| Stitch | Grazing | `stitch_gi` | ranges · GI axes (q_oop / q_ip / exit-angle / χ_GI) · corrections **+ GI group** |
| RSM | — | `rsm` | Q bounds (H/K/L) + auto-scout · grid (H×K×L) · axes=hkl · corrections |

`valid_modes` greys the mode control for RSM (∅). `backend_required="pyfai_hist"` is set whenever a
GI or shared `CorrectionStack` toggle is on — because **`multigeometry` silently ignores the shared
pre-weight**. The renderer then **auto-selects Histogram** (or, if the user pinned `multigeometry`,
marks the correction field `CONFLICT` with the reason) — so the old "⚠ requires Merge = Histogram"
greyed puzzle becomes a one-click/automatic resolution, not a riddle. Corrections read **material +
energy from §3** (don't duplicate); a missing material surfaces as a `MISSING` badge on the GI
toggle with the blocker "set sample material (3c)".

---

## 5. Provenance + status badges — typed, functional, economical

Badges are a *view of `FieldStatus`*, not styling. Six states:

- `AUTO` (inferred) · `FILE` (loaded) · `SET` (user) · `SAVED` (restored from `.nxs`) — provenance.
- **`MISSING`** (required input absent) · **`CONFLICT`** (values disagree, e.g. corrections on +
  `multigeometry`, or energy ≠ calibration wavelength) — these **gate `can_run`** and populate
  `run_blockers` (the Run-button tooltip). No more silent failures.

**Economy:** show a provenance badge only when it carries signal — `FILE`/`SET`/`SAVED`, or `AUTO`
that the user is about to override. A field that auto-inferred as expected stays quiet.
`MISSING`/`CONFLICT` always show. (This resolves the "4 badges on every row = noise" problem.)

---

## 6. Performance / reliability — async Source, lazy metadata

`ScanSourceWidget` is the right abstraction, but **probing must move off the GUI thread before it
becomes central** — synchronous open + first-frame probe will stall on large Eiger/NeXus inputs.
Reuse the exact discipline the display loaders earned: a worker + **generation token + cancellation**
(supersede a stale probe when the selection changes), and the Source card shows `probing… / cached /
N frames / raw ✓`. **Metadata previews are lazy** — never build a full metadata/positioner table
synchronously during source selection; populate on demand.

---

## 7. Retire `ParameterTree` as the primary panel (keep it as the inspector)

The recurring GUI fragility — the #56 grazing-checkbox repaint, label clipping, hidden carrier
fields, the hard-disable hack, state-sync edge cases — is `ParameterTree` being forced to act like
a custom control panel. The typed-card panel fixes it at the root. **Keep `ParameterTree` only for
the advanced / raw-inspector dialog** (its real strength).

**Caveat (the real cost):** the risk is **behavior** parity, not **visual** parity. The wrangler
has accumulated non-obvious correctness behavior — two-stage progressive disclosure, hard-disable-
during-run, session persistence, async motor-options population, the reload-hydration 2-way sync
(`design_gi_panel_move_and_2way_sync`). Reproducing the look is quick; re-achieving those twenty
small behaviors without regressions is the work. So retire it **last**, behind a flag, only after
the card panel reaches behavior parity — with the inspector as the escape hatch throughout.

---

## 8. Detailed implementation plan

Implement this as a new branch and land it in small, gated commits. The first half is deliberately
Qt-free or hidden behind a flag; the live visible flip comes only after profile parity is proven.

**Current implementation note (2026-06-28):** the foundation slices have landed on
`feature/controls-panel-v2`:
`xdart.gui.tabs.static_scan.controls_logic` provides the Qt-free `ControlState →
ControlProfile` / field-status / analysis-launcher gate,
`xdart.gui.tabs.static_scan.ui.controls_panel_v2.ControlsPanelV2` renders a hidden
card scaffold behind `XDART_CONTROLS_PANEL_V2=1`, and
`xdart.gui.tabs.static_scan.analysis_context.AnalysisContext` is now the seam used by
Peak Fit, Phase Fit, and Scan Plot launch. This is intentionally behavior-preserving:
the visible controls panel is still legacy-backed, and live fitting still follows the
latest processed frame through the same latest-wins worker. The hidden V2 panel is
observational only: it shows source/experiment/processing/output/analysis status and opens
the existing analysis popups. It also renders hidden producer/inspector action intents for
Choose Source, Calibrate, Make Mask, Refine, and Advanced; the enabled ones route through
the existing production hooks and do not yet replace legacy processing controls.

### Phase 0 — doc/API reconcile, no GUI behavior change

- Make this document the controls-panel status authority; update older docs to point here rather
  than restating the section model.
- Add a short ADR or design note for the launcher taxonomy if it grows beyond this doc:
  Reduction tools vs Instrument producers vs Analysis popups.
- Inventory every current wrangler field and map it to a `FieldId`, section, headless owner, and
  existing session key. Keep this table in the implementation PR, even if not in the final docs.

**Gate:** markdown link check; no code change.

### Phase 1 — Qt-free profile core

- Add `xdart.gui.tabs.static_scan.controls_logic` (or equivalent) with:
  `ControlState`, `ControlProfile`, `FieldId`, `FieldStatus`, `SourceCaps`, `ResultCaps`,
  `AnalysisLauncherSpec`, and `build_profile`.
- Keep it import-clean: no PySide, no pyqtgraph, no h5py open, no source probing.
- Encode all gating here:
  PONI/calibration required for processing; Stitch/RSM require metadata/motors; RSM disables
  measurement-mode; corrections require compatible backend/material/energy; analysis launchers
  expose disabled reasons.
- Treat the existing Int controls as a legacy-backed Processing page at first. This gives users one
  Tool selector immediately without moving all Int widgets on day one.

**Tests:** pure unit tests for Standard/GI Int, Stitch standard/GI, RSM, missing PONI, missing motors,
raw-unreachable ROI, missing 1D fit inputs, optional dependency missing, energy conflict, and
`multigeometry`+corrections conflict.

**Status:** FOUNDATION IMPLEMENTED. The current module covers `ControlState`,
`ControlProfile`, `FieldId`, `FieldSpec`, `FieldStatus`, source/result/geometry caps,
analysis launchers, viewer run suppression, legacy mode-text mapping, and
GI Stitch/RSM/xu_hist real-data gates. The current `FieldStatus` implementation is a
first-pass status/provenance surface; richer conflict/provenance detail remains Phase 4
work as the Experiment card becomes authoritative.

### Phase 2 — hidden `ControlsPanelV2` scaffold

- Add a hidden/feature-flagged card panel that renders `ControlProfile` but does not yet drive runs.
- Build reusable widgets:
  `SectionCard`, `FieldRow`, `StatusBadge`, `MoreButton`, `LauncherButton`,
  `SegmentedControl`, `ProcessingStack`.
- Reuse theme tokens; do not style a `ParameterTree` to look like cards.
- Make hydration explicit: `set_state(state, *, block_signals=True)` followed by one profile render.

**Tests:** offscreen render tests for card visibility, badge text/classes, disabled tooltips, and
signal blocking during profile swaps.

**Status:** HIDDEN PREVIEW IMPLEMENTED. `ControlsPanelV2` renders Run Readiness,
Source, Experiment, Processing, Output, and Analysis cards from `ControlProfile`.
`staticWidget` mounts it only when `XDART_CONTROLS_PANEL_V2=1`; the legacy panel remains
the production surface. The preview refreshes on wrangler attach, mode changes, new scans,
display data changes, viewer mode changes, and stitch-mode changes. `MoreButton`,
`SegmentedControl`, and the real Processing stack remain future phases.

### Phase 3 — Source card over `ScanSourceWidget`

- Embed the existing `ScanSourceWidget` as the Source card engine.
- Move source probing to an async worker with generation cancellation. The card displays
  `probing`, `cached`, frame count, metadata status, raw reachability, mask reachability, and
  multi-scan chips.
- Keep metadata-table construction lazy. The Source card should show summaries; the full table belongs
  in Scan Plot / metadata popups.
- Ensure scan grouping uses real `CompositeFrameSource` output, not purely visual chips.

**Tests:** source-kind probes with stale-generation cancellation, natural scan order, raw-reachable
truth table, grouped scans producing one composite source, no GUI-thread blocking regression for a
slow fake source.

### Phase 4 — Experiment card and instrument producer tools

- Build Experiment summaries:
  3a Diffractometer, 3b Detector/PONI/mask, 3c Sample + Standard/Grazing measurement, 3d Beam.
- Wire Calibrate, Refine, and Make Mask as producers that update Experiment state and provenance.
- Move the GI facts to Experiment in the model first; keep the legacy widgets as the active
  producers until the V2 panel is flipped.
- Use ADR-0009 for energy: one Beam wavelength/energy field, conflicts marked in profile, no duplicate
  plan-energy inputs.
- Add the "More..." dialogs for detector, diffractometer, sample/GI, and UB. These can still reuse
  small inspector widgets internally; the primary panel stays card-based.

**Tests:** PONI load hydrates detector+beam; `.nxs` reload marks saved fields; GI toggle changes valid
processing axes without spurious reintegrate; conflicting energy blocks run; Refine result updates
diffractometer summary; mask value and mask reachability do not drift.

**Status:** PREVIEW ACTIONS IMPLEMENTED. The hidden V2 panel now exposes typed
`ControlActionSpec` producer/inspector buttons. Choose Source delegates to the current wrangler
browser, Calibrate and Make Mask click the existing integrator buttons, and Advanced opens the
current combined advanced integration dialog. Refine is visible but disabled with a real-data-gate
reason. The Experiment card is still a read-only status preview; the legacy panel remains the
authoritative editor until field hydration/provenance parity is complete.

### Phase 5 — Processing stack for Int/Stitch/RSM

- Build Processing pages keyed by `ControlProfile.processing_page`.
- Start with Stitch/RSM pages because they have no legacy primary panel yet.
- Keep Int pages legacy-backed until Phase 7, but route all mode/tool validity through the new
  profile.
- Add backend-aware correction behavior:
  `multigeometry` default for Stitch-2D, `pyfai_hist` required for shared corrections/GI, `xu_hist`
  hidden/disabled until the convention gate lands.
- Persist processing page values per tool/mode so page swaps never drop user input.

**Tests:** page swap preserves values, invalid backend is blocked or auto-resolved as specified,
Stitch/RSM plan construction matches the headless `StitchPlan`/`RSMPlan`, no hidden controls can mutate
the plan during session restore.

### Phase 6 — Analysis launcher rail + popup standardization

- Add a small Analysis launcher row/card separate from the Processing page:
  Peak Fit, Phase Fit, Scan Plot / Metadata, ROI Stats, and future Strain / Texture.
- Introduce `AnalysisContext` as the only object passed to dialogs. It exposes:
  selected labels, active 1D trace provider, loaded scan/source handle, raw reachability,
  metadata table access, cancellation/progress hooks, and optional display/current-frame hints.
- Refactor existing Peak/Phase/Scan Plot dialogs only enough to use `AnalysisLauncherSpec` and
  `AnalysisContext`; do not redesign their interiors in this pass.
- Add lazy optional-dependency messaging through the launcher spec so the button can explain
  "install fitting dependencies" before opening.
- Future tools:
  - **sin²ψ / strain:** launcher enabled when a 1D result and psi/tilt metadata or a compatible
    scan grouping exist; dialog delegates to `run_sin2psi` / strain helpers.
  - **texture:** initially a Phase-Fit advanced option or standalone popup over phase-fit results;
    graduate to its own launcher once `analysis.texture` has a stable plan/result object.
  - **RSM/ROI-derived tools:** consume persisted/loaded `RSMVolume` or ROI computed columns through
    the same AnalysisContext, not private display widgets.

**Tests:** launcher availability matrix; disabled-reason tooltips; Peak/Phase still fit active 1D;
Scan Plot opens from loaded metadata; ROI disabled when raw is unreachable; optional deps missing yields
friendly message without crashing xdart.

**Status:** PARTIAL/PREVIEW IMPLEMENTED. `AnalysisContext` exists and existing Peak/Phase/Scan
Plot entry points use it instead of direct dialog-to-widget internals. The hidden V2 Analysis
card renders launcher buttons with disabled reasons, entry-point metadata, required result
capabilities, optional dependency names, and singleton keys. It opens the existing Peak Fit,
Phase Fit, Plot Metadata/Scan Plot, and ROI entry points through the tab owner. Future
Strain/Texture launchers are present but disabled until their headless result contracts and
real-data gates are ready.

### Phase 7 — Int 1D/2D migration

- Move Int controls from the legacy wrangler/integrator trees into the V2 Processing pages following
  `design_gui_int_migration_jun2026.md`.
- Re-home PONI, GI facts, beam, and sample fields to Experiment; keep Int output/ranges/method/corrections
  in Processing.
- Keep reintegrate buttons and advanced integration settings, but make "Advanced" the surviving
  `ParameterTree` inspector.
- Use the same `build_reduction_plan` path as today; this is a UI relocation, not a reduction rewrite.

**Tests:** Int 1D/2D live, batch, reintegrate 1D, reintegrate 2D, GI submodes, XYE, Image Viewer,
session restore, `.nxs` reload hydration, and `live≡batch≡reload`.

### Phase 8 — flip, retire, and clean

- Feature-flag flip the V2 panel on by default after parity.
- Keep a short emergency flag for the legacy panel during live testing only.
- Retire the primary `ParameterTree` wrangler once tests and live checkpoint pass; keep the advanced
  inspector dialog.
- Remove dead session keys, hidden carrier fields, duplicate GI/PONI state, and obsolete docs.

**Tests:** full offscreen suite, GI equivalence spine, byte-compat, real-data manual checklist:
standard TIFF/Eiger Int, GI Int, Stitch grouped source, RSM grouped source, Peak/Phase, Scan Plot,
ROI raw-reachable/unreachable, session restore.

### Sequencing and acceptance guards

- The panel must not expose GI Stitch, `xu_hist`, or RSM GUI controls as production-ready until their
  real-data convention gates land; the profile may compute them earlier, the renderer hides or disables
  them.
- Field relocations — PONI→3b, GI facts→3c, energy→3d — cross the persistence/reload boundary. Every
  phase that moves one must keep `tests/xdart/test_gi_batch_real_data.py::test_*_equivalence` green.
- DISPLAY panels (`design_gui_display_panels_jun2026.md`) are a separate preserved workstream. The
  controls refactor may call display APIs, but must not fork display state, bypass generation stamping,
  or revive role-level special cases.

---

## 9. Open decisions (Vivek)
- **Int in the unified panel now vs after its migration.** Recommendation: one Tool selector
  immediately (step 6 refinement); Int *page* migrates last.
- **Two mode controls (Tool + Measurement-mode) vs one 4-way selector** `{Int, Stitch-std,
  Stitch-GI, RSM}`. Either is fine *if* the `(tool, mode)` validity lives in `ControlProfile.valid_modes`
  and no control silently mutates another (today `setGrazing` forces Tool=Stitch — make that explicit).
- **Analysis launcher density.** Recommendation: one compact "Analysis" rail with icon+text buttons and
  disabled reasons, not a large fifth section. If the rail gets crowded, put only the top four launchers
  inline and move future tools into an "More analysis..." menu backed by the same `AnalysisLauncherSpec`.
- **Whether Texture is a Phase-Fit option or a standalone launcher.** Recommendation: keep texture inside
  Phase Fit until the headless `analysis.texture` API has a stable plan/result and at least one non-fit
  workflow. Do not create a standalone texture popup just because a module exists.

## 10. Risks / implementation traps

Mostly low-risk (several devices have working precedents), but these are specific to *this* refactor:

- **A badge that lies is worse than none.** `FieldStatus` must be recomputed on every hydrate from
  the *same* provenance the value carries, and badge updates **blocked during hydration** (mirror the
  existing `sigUpdateGI` signal-block) so a half-hydrated state never flashes a wrong `AUTO`/`SAVED`.
  One source of truth per field — never a hand-set label.
- **Reactive-stack signal storms (the #56 / spurious-reintegrate family).** Swapping the Processing
  `QStackedWidget` page must not drop user-entered §4 values nor re-fire the GI render mid-restore.
  **Block signals during page swap *and* during session restore**, exactly as the GI reload-hydration
  does today — a page swap that re-emits the GI signal during restore is the classic spurious run.
- **Greying is a hint, not enforcement.** The `multigeometry`-ignores-corrections fact is enforced
  **headless** (`run_stitch` warns/skips, `plans.py`). The §4 grey / `CONFLICT` / auto-select-Histogram
  is UX on top — keep the headless warn as the authoritative guard (the
  `[[mask-saturated-toggle-authoritative]]` principle: the GUI gates, core policy is separate).
- **Stitch-2D χ is provisional.** The `pyfai_hist` Stitch-2D χ axis is convention-provisional until
  P3c clears — **default Stitch-2D to `multigeometry`** (or surface a "provisional azimuth" note) in
  the reactive page until then. (1D `|q|` is convention-free and safe.)
- **Multi-scan chips must be backed by `ScanSourceWidget`.** The chips pull in `CompositeFrameSource`
  grouping, `discover_scans`, raw-reachability, and metadata-optional sources — chips without the real
  widget underneath are a fake control that doesn't produce grouped runs.
- **Don't re-skin the `ParameterTree`.** A re-styled tree re-creates the group-granular show/hide that
  forced the 2-stage disclosure collapse; per-field disclosure *requires* the typed cards (§7).
  Re-skinning is not the streamlining this doc describes.
- **Analysis dialogs must not become hidden wranglers.** Peak/Phase/Scan Plot/ROI/Strain popups are
  consumers of `AnalysisContext`; they do not read private `imageWrangler`, `integrator`, or display-widget
  internals. If a popup needs a new input, add it to the context or a headless plan, not to a dialog-side
  shortcut.
- **Batch analysis needs the same cancel/progress discipline as reduction.** Fitting and ROI workers can be
  expensive on long scans. They must use bounded workers, progress/cancel, and latest-generation result
  guards; never compute a full scan synchronously from a launcher click.
- **Optional dependencies are UX state.** Missing `lmfit`, `pymatgen`, `xrayutilities`, or future GL/viz
  extras should disable or soften a launcher with a clear reason. They must not make xdart startup fail.

---

## 11. Doc status (the reconcile)
- **Canonical:** this doc (vocabulary + the `ControlProfile` contract + the staged plan).
- **Still authoritative (feed into this):** `design_wrangler_organization_jun2026.md` (the input
  inventory + mode-gating table — re-label "Data"→"Source"); `design_gui_int_migration_jun2026.md`
  (step 6 detail); `design_gi_panel_move_and_2way_sync_jun2026.md` (the 2-way hydration);
  `design_shared_source_panel_jun2026.md` (the `ScanSourceWidget` spec); `design_gui_display_panels_jun2026.md`
  (the DISPLAY side); ADR-0008 (GI ownership), ADR-0009 (energy single-source).
- **Retired (content absorbed here):** the earlier 3-section controls design + brief — their
  3-section "Data/Experiment/Processing" naming and the Transmission mode are superseded by
  §1–4 above; the source docs were removed to avoid confusion.
