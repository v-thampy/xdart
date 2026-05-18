# `ssrl_xrd_tools` example notebooks

End-to-end demonstrations of the headless `ssrl_xrd_tools` API.

Every notebook follows the same convention:

1. **Imports** cell (no edits needed).
2. **✏️ Configuration** cell — *the only cell you need to edit*.
   Sectioned into REQUIRED, OPTIONAL, and tuning groups; REPLACE
   markers flag the lines that always need attention.
3. **Validation** cell — runs `assert` checks on the paths and
   reports `OK` if everything is in place.  Catches typos before
   any integration / fitting starts.
4. The rest of the notebook runs top-to-bottom unchanged.

| #   | Notebook                                | What it shows                                                                  | Modules exercised                                       |
| --- | --------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------- |
| 01  | [`01_batch_integration.ipynb`](01_batch_integration.ipynb)            | Per-frame `integrate_1d` / `integrate_2d` loop over an image directory; NeXus round-trip.  **Low-level reference** — see 06 for the recommended high-level path. | `io`, `integrate.single`                                |
| 02  | [`02_multigeometry_stitching.ipynb`](02_multigeometry_stitching.ipynb) | `MultiGeometry` 1D + 2D stitching across detector-angle scans                  | `integrate.multi` (`create_multigeometry_integrators`, `stitch_1d`, `stitch_2d`) |
| 03  | [`03_phase_and_peak_fitting.ipynb`](03_phase_and_peak_fitting.ipynb)   | Structure-informed (`PhaseFitter`) **and** structure-agnostic (`fit_peaks`) on the same pattern — side-by-side comparison | `analysis.phase`, `analysis.fitting`                    |
| 04  | [`04_batch_phase_fitting.ipynb`](04_batch_phase_fitting.ipynb)         | `FitConfig` + `fit_sequence` + `FitResultStore` over a sequence of patterns; phase fractions + lattice trends as a DataFrame | `analysis.fitting.batch`                                |
| 05  | [`05_sin2psi_analysis.ipynb`](05_sin2psi_analysis.ipynb)               | GI polar integration → χ-sector peak fits → sin²ψ regression → strain / stress | `integrate.gid`, `analysis.strain`                      |
| 06  | [`06_headless_reduction_pipeline.ipynb`](06_headless_reduction_pipeline.ipynb) | **Canonical headless reduction**: `ReductionPlan` + `Scan` + `Frame` + `MemorySink` / `NexusSink` + `run_reduction`.  Same workflow xdart's wranglers use internally; the recommended path for new code. | `reduction` (`Frame`, `Scan`, `ReductionPlan`, `GIMode`, `MaskSpec`, `MemorySink`, `NexusSink`, `run_reduction`) |

## Prerequisites

Every notebook needs at minimum:

- A pyFAI **PONI** calibration file.
- One or more detector **images** (TIFF/EDF/CBF/HDF5/NeXus — fabio
  handles them all).
- Optionally a detector **mask** (EDF/NPY) — without it the
  notebooks still work, just with no pixel masking.

Notebooks 03, 04 also need:

- **CIF files** for each phase you want to fit (a few KB each;
  download from [Materials Project](https://next-gen.materialsproject.org/)
  or your favourite database).

Notebook 05 also needs:

- A **metadata sidecar** (`.txt`) or some other source for the
  GI incidence angle.

## Install

The notebooks use the optional extras that match the modules they
exercise:

```bash
# Minimal headless install — sufficient for notebooks 01, 02:
pip install ssrl_xrd_tools

# For fitting (notebooks 03, 04, 05):
pip install "ssrl_xrd_tools[fitting]"

# Full kitchen sink (you'll thank yourself later):
pip install "ssrl_xrd_tools[all]"
```

> `uv pip install` is **10–100× faster** if you have `uv` on hand —
> see the project README for install instructions.

## How they were built

Cells are authored in a single Python script (`_build_notebooks.py`)
and emitted as JSON.  After editing cell content there, re-run:

```bash
python examples/notebooks/_build_notebooks.py
```

This is a maintainer-only script — not packaged or required to run
the notebooks themselves.
