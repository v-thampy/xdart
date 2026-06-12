# xrd-tools — working notes for Claude Code

ONE distribution (`xrd-tools`), TWO import packages under `src/`:
**`xrd_tools`** — the headless XRD reduction + I/O core (imports **no
Qt/pyqtgraph**), and **`xdart`** — the Qt GUI, a thin consumer of it.
Anything that does not need Qt belongs in `xrd_tools` ("keep xdart thin").
Merged from the former `ssrl_xrd_tools` + `xdart` repos (Jun 2026, full
histories — `git log --follow` works); see `MIGRATION.md`.  The
`ssrl_xrd_tools` import name is a deprecation shim returning the real
`xrd_tools` modules.

## North star
Roadmap: `~/repos/review/roadmap_2026-06-10.md`.  (1) **headless-first
APIs** — `xrd_tools` fully usable with no GUI; (2) **thin xdart** over it;
(3) **robustness** (fail-loud writes, strict schema, live≡batch≡reload
spine); (4) **performance** (streaming, parallel, bounded memory);
(5) **expandability** (clean `FrameSource`/`ReductionPlan`/`ReductionSink`
seams).  Post-1.0 priority: F3 ROI statistics, then the deferred items in
`~/repos/review/CC_preship_sweep_deferred_jun2026.md` (D1 reintegrate, D2
thumbnail LRU, F2/F4 save-path + embed-raw designs, F5 Set Bkg everywhere).

## Environment & tests
- Dev venv: `~/.venvs/xrd-tools-dev` (editable install, all extras).  The
  conda env `xrd_edit` is the user's SHIPPED install — do not touch it.
- Core suite: `pytest tests/core` (self-contained synthetic fixtures;
  `-m "not slow"` skips full-size detector cases).
- GUI suite: `QT_QPA_PLATFORM=offscreen pytest tests/xdart` (needs Qt +
  pyFAI; real-data GI tests auto-skip if `test_data/` is absent).
- Pure display logic: `pytest -m display_logic` (no Qt/pyFAI needed).
- `tests/__init__.py` and `tests/xdart/__init__.py` anchor the package
  chain — without them pytest puts `tests/` on sys.path and `tests/xdart`
  shadows the real `xdart` package.

## Guardrails (do not violate)
- **Do NOT `git push`, publish, bump versions, or tag** — maintainer only.
- **Keep the NeXus writer/reader correctness checks strict**:
  `validate_integrated_stack_write`, `_require_uniform_axes_1d/2d`
  (`xrd_tools/io/nexus.py`) and xdart's `_select_frames_to_write`.  Fix GI
  per-frame-axis issues by **freezing a common grid**, never by relaxing a
  validator.
- **Persisted format is frozen + additive-only.**  Attribute keys keep the
  historical `ssrl_` prefixes; the GUI writer's NXprocess `@program` stays
  `"ssrl_xrd_tools"`.  Layout facts live in `xrd_tools/io/schema.py`
  (schema-as-code) — change code to match the pins in
  `tests/core/test_schema_as_code.py`, never the pins.  The byte-compat
  gate `tests/core/test_v2_record_compat.py` pins the written record
  against a committed pre-migration signature.
- **The live≡batch≡reload equivalence spine**
  (`tests/xdart/test_gi_batch_real_data.py::test_*_equivalence`) is the
  acceptance gate for the publication layer.  A failing equivalence is a
  real bug, not a tolerance to widen.
- **2D orientation convention** (classic bug source):
  `IntegrationResult2D.intensity` is `(radial, azimuthal)`; the saved stack
  and `get_2d` are `(chi, q)` = `(y, x)`; `FrameView` stores
  `(axis_2d_y, axis_2d_x)`.  `FrameView.from_results` transposes the pyFAI
  result; `read_frame_view` does not.  Don't "tidy" that asymmetry.
- **`xrd_tools.core` stays import-light** (no Qt/h5py/fabio/pyFAI at import
  — the h5py codec re-exports are lazy).  `display_logic.py` imports it
  top-level under the purity guard.
- Batch mode is a perf switch: silent during a run (no per-frame display
  refresh; the GUI reloads from the `.nxs` at end-of-batch).
- The **publication gate** (`validate_publication`) complements the writer
  validators, never replaces them.  Reject/skip bad output **per frame**,
  never abort a whole-scan save.  Carry `metadata_raw` + `metadata_numeric`.
- When fixing a bug, label it **regression** vs **pre-existing**.

## Package map — `src/xrd_tools/`
- **`core/`** — pure contracts, import-light.  `containers` (`PONI`,
  `IntegrationResult1D/2D`), `frame_view` (`Axis`, `TwoDKind`, `FrameView`,
  `two_d_kind_from_units` — THE GI-kind classifier, lenient for legacy
  spellings), `scan` (`ScanFrame`/`Scan`/`FrameSource` — the reduction
  input contracts; `Scan.geometry` drives finish-time per-frame geometry),
  `filters` (`compile_filter` — the boolean Filter grammar), `geometry/`,
  `metadata`, `hdf5` (codec; lazily re-exported), `provenance`, `config`.
- **`io/`** — persistence + readers.  `schema.py` (schema-as-code:
  `SCHEMA`, attr keys, row-aligned sets); `nexus.py` (stacked v2
  writer/reader + validators); `nexus_record.py` (per-frame record
  primitives: `write_frame_record`, `stamp_source_base`, thumbnails,
  `drop_integrated_rows` — shared by the headless sink AND xdart's
  writer); `read.py` (`get_1d/2d/thumbnail/metadata`, `get_raw_frame`,
  `open_scan`/`ProcessedScan`, `relative_source_path` — N1 portable paths,
  resolution precedence `source_root` > `@source_base` > scan dir);
  `frame_view.py` (`read_frame_view`/`iter_frame_views` — the reload half
  of the equivalence spine); `image_source.py`, `nexus_inspect.py`, etc.
- **`reduction/`** — the streaming spine.  `ReductionSession` (parallel
  workers + single writer thread, bounded in-flight, fail-loud
  `finish()`), `run_reduction`, sinks (`NexusSink` writes the COMPLETE v2
  record by default — `complete_record=True`, `source_base=`), GI freeze
  policies.  Monitor warnings are per scan (session-owned warn set).
- **`integrate/`** (pyFAI + GI/FiberIntegrator), **`sources/`**
  (`open_source(spec)`), **`rsm/`**, **`transforms/`**, **`corrections/`**,
  **`analysis/`**, **`viz/`** (matplotlib only).

## Package map — `src/xdart/`
- `xdart_main.py` — thin Qt-probing entry (`xdart` console script); real
  startup in `_gui_main.py`.
- `modules/reduction.py` — THE LiveScan→core adapter
  (`frame_from_live_frame` / `scan_from_live_scan` /
  `open_live_reduction_session` / `plan_from_live_scan`).  One copy of the
  scan-data-row/path/wavelength extraction lives here.
- `modules/frame_publication.py` — the Qt-free GUI envelope
  (`FramePublication`, `PublicationStore`, `validate_publication`).
- `modules/ewald/` — `LiveScan`/`LiveFrame`, `nexus_writer.py` (GUI
  writer: append cursor, NFS retry, LiveFrame thumbnail policy — record
  itself via `xrd_tools.io.nexus_record`), `frame_series.py`
  (persist-before-evict invariant: `_in_memory` never evicts unpersisted
  frames).

## Display layer (`xdart/gui/tabs/static_scan/`)
One direction of flow: background threads write data → the GUI computes
*what to show* as immutable state → a thin renderer draws it, all
generation-stamped.
- **`display_logic.py`** — pure, Qt-free decision core (purity-guard
  enforced: no Qt/pyqtgraph/h5py/pyFAI/fabio; numpy and the import-light
  `xrd_tools.core` are allowed).  `compute_display_state`, `build_payload`,
  `render_plan`, controller registry.
- **`display_controllers.py`** — one controller per mode, registered into
  the open registry.  Viewer controllers resolve files through the headless
  `xrd_tools.io` APIs; xdart never opens HDF5 to guess.
- **`display_frame_widget.py`** — mode-agnostic `update()`:
  `get_idxs → controller.compute_state → controller.build_payload →
  render_display` (draw wanted panels, clear unwanted ones, drop
  stale-generation payloads).
- **`hydrated_raw.py`** — the shared hydrated-raw LRU (D5): order rides on
  the shared `data_2d` under `data_lock`; ALL writers (GUI + scan/load
  threads) trim the same cap.
- §10 seams: `DisplayState` carries keyed `panels` + a `layout` descriptor
  so Stitch-2D/Stitch-1D/RSM arrangements are data, not mode branching.
  Future modules add their own `*_logic.py` + register a controller.

## Frame publication layer
One path from "a frame was processed" → "validated, displayed, persisted"
for live / batch / reload / viewers.  `xrd_tools` owns the round-trippable
record (`core.frame_view` + `io.frame_view` readers); xdart owns the GUI
envelope and the single shared `PublicationStore` (created in
`staticWidget`, generation-aware, internally locked).  **In progress:**
making publications the *sole* display contract — `display_controllers`,
`display_data`, `metadata` still read `data_1d`/`data_2d`/`scan_data` in
parallel; treat those as internal hydration mirrors and migrate only
behavior-preserving with the spine green.  (D2 thumbnail LRU lands with
this migration.)

## Generation stamping
`displayFrameWidget.display_generation` bumps on mode switch + effective-
selection change.  The load worker publishes only through generation-
checked snapshots (`_absorb_chunk` drops stale chunks); `render_plan`
drops payloads whose generation ≠ the state's.
