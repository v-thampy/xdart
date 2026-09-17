# xrd_tools Notebook Examples

These notebooks are small interactive tools built from public, headless
`xrd_tools` APIs. They use Jupyter, ipywidgets, and Plotly for exploration;
scientific operations remain in the library so the same calls work in scripts
and later GUI front ends.

Every notebook has one configuration cell. Its default is deterministic smoke
mode, which needs no network or private data. Set
`XDART_NOTEBOOK_SMOKE=0` and `XDART_TEST_DATA=/path/to/data` to enable its
explicit real-data configuration. Notebook execution never writes into source
or test-data directories; temporary smoke artifacts use `/tmp`.

| # | Notebook | Purpose | Primary public APIs |
| --- | --- | --- | --- |
| 01 | [Batch integration](01_batch_integration.ipynb) | Low-level 1-D/2-D integration reference | `integrate_1d`, `integrate_2d` |
| 02 | [Multi-geometry stitching](02_multigeometry_stitching.ipynb) | Calibrated multi-angle stitching and monitor normalization | `stitch_1d`, `stitch_2d` |
| 03 | [Phase and peak fitting](03_phase_and_peak_fitting.ipynb) | Pilot peak fitting and the separate phase-fit path | `PeakFitPlan`, `PeakFitAnalyzer`, `PatternViewer` |
| 04 | [Batch phase fitting](04_batch_phase_fitting.ipynb) | Aligned, failure-preserving batch analysis | `run_batch`, `batch_params_table` |
| 05 | [sin2psi strain](05_sin2psi_analysis.ipynb) | GI polar sectors and strain regression | `Sin2PsiPlan`, `run_sin2psi` |
| 06 | [Headless reduction](06_headless_reduction_pipeline.ipynb) | Plan, executor, and sink ownership | `ReductionPlan`, `run_reduction` |
| 07 | [Reading processed NeXus](07_reading_processed_nxs.ipynb) | Frame-label reads, thumbnail/raw distinction, and viewers | `open_scan`, `get_1d`, `get_2d` |
| 08 | [Time-resolved XRD](08_time_resolved_xrd_analysis.ipynb) | Lazy processed-series analysis, fitting, lattice, and thermal trends | `load_time_resolved_series`, `fit_peak_series`, Plotly helpers |
| 09 | [Reciprocal-space mapping](09_reciprocal_space_mapping.ipynb) | RSM plan/result separation and interactive slice review | `RSMPlan`, `run_rsm`, `RSMVolume` |

## Interactive behavior

Matplotlib defaults to the interactive **widget** (`ipympl`) backend, equivalent
to `%matplotlib widget` or `%matplotlib ipympl`. The notebook extra includes
`ipympl`. The first import cell selects it before importing `pyplot`; restart
the kernel and run all cells after updating an older notebook. For automated
execution without a frontend, set `XDART_NOTEBOOK_BACKEND=inline`; the smoke
runner does this explicitly. Plotly figures remain independently interactive.

Existing `ImageViewer`, `PatternViewer`, `PeakFitControls`, and related
notebook-safe widgets are reused where their contracts fit. Sliders use
`continuous_update=False`; integration and fitting actions are explicit
buttons. Changing a display control does not trigger a reduction or a batch
fit.

For a waterfall, choose any two one-dimensional coordinates on the intensity
Dataset:

```python
from xrd_tools.viz import plot_waterfall

fig = plot_waterfall(dataset, x_coord="q", y_coord="temperature")
fig.show()
```

The default coordinates are `q` and `frame`; select `y_coord="time"` for a time
series. Coordinate `long_name` and `units` attributes set axis and hover labels.
The helper preserves row order and intensity values; logarithmic colors and
percentile limits affect only the display. The former time-specific function
name has been replaced without an alias.

## Generate and verify

Do not hand-edit generated notebook JSON. Edit `_build_notebooks.py`, then run:

```bash
pixi run python examples/notebooks/_build_notebooks.py
pixi run python examples/notebooks/_smoke_notebooks.py
git diff --exit-code -- examples/notebooks
```

The smoke gate validates each notebook with `nbformat`, compiles every code
cell, rejects Qt/legacy imports and private paths, requires clean cells, and
executes all nine notebooks in a temporary directory.

## Install

For installed use, include the notebook extra:

```bash
pip install "xdart[notebook,fitting,rsm]"
```

Notebook 08 can optionally use the local processed Pt fixtures when
`XDART_TEST_DATA` is set. Its linear thermal calibration is illustrative only:
mechanical strain, texture, and reflection disagreement can bias inferred
temperature.
