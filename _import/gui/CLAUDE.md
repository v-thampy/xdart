# xdart — working notes for Claude Code

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
  `ImageViewerController`, `XYEViewerController`), registered into the open registry
  (`register_controller`/`controller_for`). Each owns its mode's selection rules + loading lifecycle.
  Viewer controllers never consult `scan.frames` or the integration-unit combo. `ImageViewerController`
  resolves images through the headless `ssrl_xrd_tools.io` API (`classify_image_source` /
  `load_processed_raw_or_thumbnail` / `load_image_frame`) — xdart never opens HDF5 to guess.
- **`display_frame_widget.py`** — the Qt widget. `update()` is mode-agnostic:
  `get_idxs → _live_display_state (controller.compute_state) → controller.build_payload →
  render_display`. `render_display` executes the pure `render_plan`: drop a stale-generation payload,
  then draw the panels the state wants and **clear the panels it doesn't** (so a panel left from a
  previous mode/selection is always blanked). It currently delegates the pixel push to the legacy
  draw methods (`update_image`/`update_binned`/`update_plot`/`_update_*_viewer`); collapsing those
  into direct payload rendering is the data-source-unification work, not done here.

### Designing for future modules (§10 seams)
The integration view is the *first* set of registered panels. `DisplayState` carries a keyed
`panels` collection + a `layout` descriptor (rows of `PanelKey`s) so arrangement is data, not
mode-branching — Stitch-2D (`cake/plot`), Stitch-1D (`plot`), and RSM (2×3 of repeated
`SLICE_2D`/`PROJ_1D` roles) all express without touching `render`. Plot payloads are layered
`Trace`s (`data`/`fit`/`component`/`background`/`residual`); non-array output flows through the
`results` channel. Future modules add their own `*_logic.py` + register a controller; the core never
imports them.

## Generation stamping
`displayFrameWidget.display_generation` bumps on mode switch + effective-selection change. The
load worker publishes only through generation-checked snapshots (`_absorb_chunk` drops a chunk whose
generation no longer matches), and `render_plan` drops a payload whose generation ≠ the state's.
