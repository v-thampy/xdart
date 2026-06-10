# ssrl_xrd_tools — working notes for Claude Code

The headless XRD reduction + I/O library. **xdart (the Qt GUI) is the consumer** and pins this via
`ssrl_xrd_tools>=` in its `pyproject.toml`. Anything that does not need Qt belongs here, not in
xdart ("keep xdart thin"). This package must import **no Qt/pyqtgraph**.

## North star + Architecture V2 (Jun 2026)
Long-term goals (full roadmap: `review/roadmap_2026-06-10.md`): **(1) headless-first ssrl APIs** — this
package is fully usable with no GUI; **(2) thin xdart** over it; **(3) robustness** (fail-loud writes,
strict schema, live≡batch≡reload spine); **(4) performance** (streaming, parallel, bounded memory);
**(5) expandability** (clean `FrameSource` / `ReductionPlan` / `ReductionSink` seams).

**Architecture V2 — merged to `refactor/architecture-v2` (Jun 2026), not yet released.** Realized here:
one streaming `reduction.ReductionSession` spine (parallel workers + single writer thread, bounded
in-flight, **fail-loud `finish()`** — a failed write raises, never silently succeeds), source abstraction
(`sources/` + `open_source(spec)`), sink-driven writes (`ReductionSink`; xdart injects its Qt sink so ssrl
never imports xdart), and **N1 portable raw paths** (`io.read.relative_source_path` + `entry/@source_base`
resolution, with a `source_root=` override; precedence override > `@source_base` > `.nxs` dir). Staging on
`architecture-v2`; pending live testing → `dev` → coordinated release (**ssrl first**). Next-cycle items
(chunked-executor retirement, RSM real tests + energy sentinel, FIFO-writer perf) are in the roadmap;
metadata models stay **separate by design** (`ScanMetadata` scan-level vs `HeterogeneousMetadata`
per-frame — orthogonal, not a collapse).

## Environment & tests
- Env: `conda activate xrd_edit` (shared with xdart; has pyFAI/h5py/numpy). Editable install.
- Tests: `python -m pytest tests/` (verbose by default). Fixtures are synthetic (see
  `tests/conftest.py`: `poni_fixture`, `ai_fixture`); the suite is self-contained — no external data
  env var required. Heavy/full-size cases are marked `slow` (`pytest -m "not slow"` to skip).

## Guardrails (do not violate)
- **Do NOT `git push`, publish, or bump versions** — leave releases to the maintainer.
- **Keep the NeXus writer/reader correctness checks strict.** `validate_integrated_stack_write`,
  `_require_uniform_axes_1d/2d`, and `_select_frames_to_write` (in `io/nexus.py`) exist to keep the
  stacked schema consistent. Fix GI per-frame-axis issues by **freezing a common grid**, never by
  relaxing a validator.
- **NeXus schema changes are additive + back-compatible only.** Old files must still read; missing
  axis identity defaults to `TwoDKind.Q_CHI`. Persisted wavelength comes from the active
  PONI/integrator, never a default `1.0`.
- **2D orientation convention (classic bug source).** `core.IntegrationResult2D.intensity` is
  `(radial, azimuthal)`; the saved stack and `io.read.get_2d` are `(chi, q)` = `(y, x)`; `FrameView`
  stores `(axis_2d_y, axis_2d_x)` = `(y, x)`. So `FrameView.from_results` transposes the pyFAI
  result, `read_frame_view` does not transpose the already-`(y,x)` saved data. Don't "tidy" that
  asymmetry into a bug.
- When fixing a bug, label it **regression** (caused by recent work) vs **pre-existing**.

## Package map (`ssrl_xrd_tools/`)
- **`core/`** — pure data contracts (no I/O). `containers` (`PONI`, `IntegrationResult1D/2D`,
  `ScanMetadata`), `frame_view` (`Axis`, `TwoDKind`, `FrameGeometry`, `FrameView`,
  `assert_frameview_equivalent`, `numeric_metadata`, `two_d_kind_from_units`), `geometry/`,
  `hdf5` (codec), `provenance`, `metadata`, `config`.
- **`io/`** — persistence + readers.
  - `nexus.py` — the stacked v2 writer/reader (`write_integrated_stack`, `read_scan`, the validators
    above). Canonical source for the headless integrated-stack schema; xdart still owns a few
    GUI/source-reference compatibility fields around that core.
  - `read.py` — convenience readers: `read_scan`-adjacent `get_1d` / `get_2d` / `get_thumbnail` /
    `get_metadata`, `get_raw_frame`, `open_scan` / `Scan`.
  - `frame_view.py` — `read_frame_view` / `iter_frame_views` (per-frame `FrameView` reconstruction;
    the headless half of xdart's live≡batch≡reload equivalence spine). `iter_frame_views` should
    stream frame-by-frame (don't materialize the whole scan).
  - `image_source.py` — `classify_image_source` / `load_image_frame` /
    `load_processed_raw_or_thumbnail` (what kind of file is this + how to get a displayable frame).
    A bare native `entry/frames` is NOT a processed marker (eiger masters carry one); raw dataset /
    eiger-master wins.
  - `nexus_inspect.py` — read-only structure/dataset inspection for the NeXus viewer
    (`inspect_nexus`, `preview_nexus_dataset` bounded for GUI, `read_nexus_dataset` full for
    headless). `is_processed` must match `classify_image_source` semantics (don't reintroduce the
    bare-`entry/frames` heuristic).
  - also: `image.py`, `export.py`, `spec.py`, `metadata.py`, `tiled.py`, `chunk_size.py`.
- **`integrate/`** — pyFAI integration + GI (`calibration`, `gid`/FiberIntegrator, pixel-q maps).
- **`reduction/`**, **`rsm/`** (`RSMVolume`, streaming gridder), **`transforms/`**, **`corrections/`**,
  **`analysis/`** (phase/strain/sin2psi fitting), **`viz/`** (matplotlib only).

## Relationship to xdart
`FrameView` (here) is wrapped by xdart's `FramePublication`. The validation/display gate lives in
xdart; the compute, persistence, and round-trip fidelity live here. The live≡batch≡reload
equivalence spine is an xdart test, but `read_frame_view` is its reload half — keep it faithful.
