# xdart — working notes for Claude Code

## North star + Architecture V2 (Jun 2026)
Long-term goals (full roadmap: `review/roadmap_2026-06-10.md`): headless-first ssrl APIs, **thin xdart**
over them, robustness, performance, expandability.

**Architecture V2 — merged to `refactor/architecture-v2` (Jun 2026), not yet released.** xdart now drives
the ssrl streaming reduction spine through a Qt `QtNexusSink` (**one write path**, fail-loud); live + batch
both stream by default (live 2D ~27.5 s vs 59–76 s serial; true-live stays serial by design). Also: the
Pause/Resume rework (Live/Batch are pure **mode toggles**, single **Start** morphs green→orange Pause/Resume,
Live stays purple) + UI-1 group-header toggles, and **N1 portable Project-Folder / relative-path** storage.
Staging on `architecture-v2`; pending live testing (Stabilization C + B3/B4) → `dev` → coordinated release.
**At release, bump the `ssrl_xrd_tools>=` floor** to the version that adds `relative_source_path` — xdart's
writer hard-imports it, so old-ssrl + new-xdart crashes on every write.

## Environment & tests
- Env: `conda activate xrd_edit` (has Qt + pyFAI). Always activate before running tests/GUI.
- Tests: `python -m pytest tests/` (full; needs Qt/pyFAI, runs offscreen via `QT_QPA_PLATFORM=offscreen`).
- Headless display-logic only: `python -m pytest -m display_logic` (pure, no Qt/pyFAI needed).
- Sibling lib: `ssrl_xrd_tools` at `../ssrl_xrd_tools` (editable install). xdart stays thin; non-Qt
  logic belongs in ssrl. xdart pins it via `ssrl_xrd_tools>=` in `pyproject.toml`.

## Guardrails (do not violate)
- **Do NOT `git push`, publish, or bump versions** — leave releases to the maintainer.
- **Do NOT loosen the writer correctness checks** (`_select_frames_to_write`,
  `_require_uniform_axes_1d/2d`, `validate_integrated_stack_write`). GI per-frame-axis issues are
  fixed by **freezing a common grid**, not by relaxing validators (see `gi_axes_uniform`).
- Batch mode is a perf switch: keep it silent during a run (no per-frame display refresh; the GUI
  reloads from the `.nxs` at end-of-batch).
- **Publication gate** (`validate_publication`) *complements* the ssrl writer validators — it never
  replaces or relaxes them. Reject/skip bad output **per frame** (e.g. drop one all-dummy GI 2D
  row), never abort a whole-scan save. Carry metadata as both `metadata_raw` + `metadata_numeric`
  (a stray text field must never blank the panel).
- The **live≡batch≡reload equivalence spine**
  (`tests/test_gi_batch_real_data.py::test_*_publication_live_batch_reload_equivalence`) is the
  acceptance gate for the publication layer — keep it green; a failing equivalence is a real bug to
  fix, not a tolerance to widen.
- When fixing a bug, label it **regression** (caused by recent work) vs **pre-existing**.

## Display layer (`xdart/gui/tabs/static_scan/`)
One direction of flow: background threads write data → the GUI computes *what to show* as immutable
state → a thin renderer draws it. "What to show" is decided once, explicitly, with a generation
stamp, before anything is drawn — this kills the stale / clear-vs-draw class of display bugs.

- **`display_logic.py`** — the pure, **Qt-free** decision core (selection, raw-source, sentinel,
  axes, overlay, GI uniformity, `compute_display_state`, `build_payload`, `render_plan`, the
  `DisplayState`/`DisplayPayload`/`PanelKey`/`Axis`/`Trace` shapes, and the controller registry).
  It must import **no** Qt/pyqtgraph/h5py/pyFAI (numpy is allowed). A purity guard test enforces this.
  Pure decisions live here so they're unit-tested headlessly (`pytest -m display_logic`).
- **`display_controllers.py`** — one controller per mode (`ScanDisplayController`,
  `ImageViewerController`, `XYEViewerController`, `NexusViewerController`), registered into the open
  registry (`register_controller`/`controller_for`). Each owns its mode's selection rules + loading
  lifecycle. Viewer controllers never consult `scan.frames` or the integration-unit combo.
  `ImageViewerController` resolves images through the headless `ssrl_xrd_tools.io` API
  (`classify_image_source` / `load_processed_raw_or_thumbnail` / `load_image_frame`);
  `NexusViewerController` is a read-only schema/dataset inspector backed by
  `ssrl_xrd_tools.io.inspect_nexus` / `preview_nexus_dataset` — xdart never opens HDF5 to guess.
- **`display_frame_widget.py`** — the Qt widget. `update()` is mode-agnostic:
  `get_idxs → _live_display_state (controller.compute_state) → controller.build_payload →
  render_display`. `render_display` executes the pure `render_plan`: drop a stale-generation payload,
  then draw the panels the state wants and **clear the panels it doesn't** (so a panel left from a
  previous mode/selection is always blanked). Each panel role first tries the **payload** path
  (`_draw_payload`: `PlotPayload` → 1D, `ImagePayload` → raw/cake), falling back to the legacy draw
  methods (`update_image`/`update_binned`/`update_plot`/`_update_*_viewer`) when the controller did
  not supply a payload for that role.

### Designing for future modules (§10 seams)
The integration view is the *first* set of registered panels. `DisplayState` carries a keyed
`panels` collection + a `layout` descriptor (rows of `PanelKey`s) so arrangement is data, not
mode-branching — Stitch-2D (`cake/plot`), Stitch-1D (`plot`), and RSM (2×3 of repeated
`SLICE_2D`/`PROJ_1D` roles) all express without touching `render`. Plot payloads are layered
`Trace`s (`data`/`fit`/`component`/`background`/`residual`); non-array output flows through the
`results` channel. Future modules add their own `*_logic.py` + register a controller; the core never
imports them.

## Frame publication layer (the result-publication contract)
Built on top of the display layer to give live / batch / reload / viewers **one** path from "a frame
was processed" → "validated, displayed, persisted." See `frame_publication_plan.md` (review folder).
- ssrl owns the headless, round-trippable record: `ssrl_xrd_tools.core.frame_view`
  (`Axis`, `TwoDKind`, `FrameGeometry`, `FrameView`, `assert_frameview_equivalent`) and the readers
  `ssrl_xrd_tools.io.frame_view` (`read_frame_view` / `iter_frame_views`). NeXus structure/dataset
  inspection lives in `ssrl_xrd_tools.io.nexus_inspect`.
- xdart owns the GUI envelope (Qt-free contract): `xdart/modules/frame_publication.py`
  (`FramePublication`, `PublicationDiagnostics`, `PublicationStore`, `validate_publication`,
  `publication_from_live_frame` / `_from_nexus_frame` / `_from_frame_view`) and the display adapter
  `gui/tabs/static_scan/display_publication.py` (publications → `DisplayPayload`).
- The single shared `PublicationStore` is created in `staticWidget` and passed to `H5Viewer`,
  `displayFrameWidget`, `metadataWidget`. It is generation-aware + internally locked; cleared
  alongside `data_1d`/`data_2d` on reset (incl. the synchronous live `new_scan`).
- **In progress:** making publications the *sole* display contract — `display_controllers`,
  `display_data`, and `metadata` still also read `data_1d`/`data_2d`/`scan_data` in parallel. Treat
  those caches as internal hydration mirrors; do this only behavior-preserving with the spine green.

## Generation stamping
`displayFrameWidget.display_generation` bumps on mode switch + effective-selection change. The
load worker publishes only through generation-checked snapshots (`_absorb_chunk` drops a chunk whose
generation no longer matches), and `render_plan` drops a payload whose generation ≠ the state's.
