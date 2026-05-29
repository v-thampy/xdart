# ssrl_xrd_tools — SSRL X-ray Diffraction Processing Toolkit

A Python library for processing synchrotron X-ray diffraction data, providing a complete pipeline from raw detector images to publication-ready results. Built on [pyFAI](https://pyfai.readthedocs.io/) for azimuthal integration with added support for grazing incidence, multi-geometry stitching, reciprocal space mapping, and advanced analysis.

## Overview

`ssrl_xrd_tools` is a standalone Python library developed at SSRL (SLAC National Accelerator Laboratory) for comprehensive X-ray diffraction (XRD) data processing. It serves as the computational core for the [xdart](https://github.com/v-thampy/xdart) interactive GUI but is equally suited for scripting, Jupyter notebooks, and automated batch pipelines at beamlines or in the lab.

The library handles the complete workflow from raw detector images through calibration, integration, unit conversion, and advanced analysis including peak fitting, multi-phase pattern fitting, and sin²(ψ) strain analysis.

## Features

- **1D/2D Azimuthal Integration** — Fast azimuthal integration via pyFAI with full detector geometry support
- **Grazing Incidence Diffraction (GID)** — Specialized integration for surface-sensitive and thin-film measurements
- **Multi-Geometry Stitching** — Seamlessly combine multiple detector angles for extended Q-range coverage
- **Reciprocal Space Mapping (RSM)** — Generate 3D reciprocal space volumes with HKL gridding
- **NeXus/HDF5 I/O** — Standards-compliant data storage and retrieval; full NumPy/pandas object serialization
- **SPEC File Parsing** — Extract metadata, scan commands, and counter data from SPEC files
- **Peak Fitting** — lmfit-based single-peak and multi-peak fitting with selectable backgrounds (linear, constant, Chebyshev, polynomial, SNIP)
- **Phase Pattern Fitting** — Multi-phase pseudo-Voigt fitting with pymatgen-derived peak positions and template intensities (`PhaseFitter`)
- **Strain Analysis** — sin²(ψ) method for biaxial stress/strain calculation, with optional (E, ν) inputs for direct stress output
- **Batch Processing** — Directory watching and automated pipeline execution
- **Bluesky/Tiled Integration** — Access data from Bluesky-collected runs via Tiled

> **Roadmap (not yet implemented):** texture analysis (pole figures, ODF), Rietveld/LeBail refinement, automated phase matching against structure databases, flat-field / air-scatter / polarization correction modules. The corresponding `analysis/texture.py`, `analysis/refinement.py`, and `corrections/` entries are placeholders reserved for these features.

## Installation

Most users should install [**xdart**](https://github.com/v-thampy/xdart) — the desktop GUI — which pulls `ssrl_xrd_tools` in as a dependency automatically. If you only want the library (e.g. for headless scripting or Jupyter), follow the same four-step pattern below and install `ssrl_xrd_tools` directly.

### 🚀 Strongly recommended: install [`uv`](https://docs.astral.sh/uv/) first

All install commands below use **[`uv pip`](https://docs.astral.sh/uv/)** instead of plain `pip`. `uv` is a drop-in pip replacement from Astral that's typically **10–100× faster** on cold installs, ships an aggressive resolver, and has a binary cache. With the scientific-stack dependency tree (pyFAI, h5py, silx, xrayutilities, …) the difference is often the gap between a fresh-env install finishing in a few seconds versus several minutes.

**Install `uv` once** — then every install / update command below works:

```bash
# macOS:
brew install uv
# Linux / WSL:
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell):
winget install astral-sh.uv
# Cross-platform fallback (works anywhere Python is installed):
pip install uv
```

> **If you'd rather not install uv**, every command below works with plain `pip` — just drop the `uv ` prefix and replace `uv pip install ...` with `pip install ...`. The result is the same, only slower. The full command lookup table at the end of this section shows both forms side-by-side.

### Quick install (4 steps)

**1. Install conda** (skip if you already have `mamba`, `conda`, or an Anaconda Prompt installed).

Pick one:

- [**Miniforge**](https://github.com/conda-forge/miniforge/releases/latest) — recommended; conda-forge first, smaller install, ships with `mamba`.
- [**Miniconda**](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html) — Anaconda's minimal distribution.

**2. Create a Python 3.12 environment.**

```bash
mamba create -n xrd python=3.12 -y
mamba activate xrd
```

**3. Install [`uv`](https://docs.astral.sh/uv/)** (see the callout above — skip if already installed).

**4. Install xdart** (recommended — brings `ssrl_xrd_tools` in automatically):

```bash
uv pip install xdart
```

Or, library only:

```bash
# Minimum headless install (12 deps; sufficient for io / integrate / viz):
uv pip install ssrl_xrd_tools

# Common notebook user (RSM + phase fitting + Jupyter widgets):
uv pip install "ssrl_xrd_tools[fitting,rsm,gui]"

# Everything except dev tooling:
uv pip install "ssrl_xrd_tools[all]"
```

#### Plain-pip equivalents

If you skipped the `uv` install step, use these instead:

| With `uv` (recommended)                                       | Plain `pip` (slower)                                       |
| ------------------------------------------------------------- | ---------------------------------------------------------- |
| `uv pip install xdart`                                        | `pip install xdart`                                        |
| `uv pip install ssrl_xrd_tools`                               | `pip install ssrl_xrd_tools`                               |
| `uv pip install "ssrl_xrd_tools[fitting,rsm,gui]"`            | `pip install "ssrl_xrd_tools[fitting,rsm,gui]"`            |
| `uv pip install "ssrl_xrd_tools[all]"`                        | `pip install "ssrl_xrd_tools[all]"`                        |
| `uv pip install -U xdart`                                     | `pip install -U xdart`                                     |
| `uv pip install -e ./ssrl_xrd_tools`                          | `pip install -e ./ssrl_xrd_tools`                          |

#### What each extra enables

The base install is now **headless / scriptable**: just `core`, `io`,
`integrate`, `viz`.  Domain-specific features live under
[PEP 621 extras](https://peps.python.org/pep-0621/) so the dependency
footprint stays modest for batch / pipeline / CI use cases.

| Extra        | What it enables                                              | Packages                                                |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------- |
| *(base)*     | `core`, `io`, `integrate`, `viz` — headless / batch / scripts | numpy, scipy, xarray, h5py, nexusformat, fabio, silx, pyFAI, joblib, natsort, matplotlib, plotly |
| `[fitting]`  | `analysis.fitting.*` — peak / phase / strain fitting          | lmfit, pymatgen                                         |
| `[rsm]`      | `rsm.*` — Ang2Q, Gridder3D, VTK export                        | xrayutilities, pyevtk                                   |
| `[gui]`      | `gui.widgets.*` — Jupyter UI on top of plotly                 | ipywidgets, anywidget, ipyfilechooser                   |
| `[notebook]` | self-contained Jupyter environment                            | ipykernel, jupyterlab                                   |
| `[napari]`   | napari 3D viewer for RSM volumes                              | napari                                                  |
| `[tiled]`    | Tiled client for Bluesky-collected data                       | tiled[client]                                           |
| `[vtk]`      | back-compat alias for `pyevtk` (now also in `[rsm]`)          | pyevtk                                                  |
| `[all]`      | everything except dev                                         | `ssrl_xrd_tools[fitting,rsm,gui,notebook,napari,tiled]` |
| `[dev]`      | test / build / release                                        | pytest, build, twine                                    |

Extras compose: `uv pip install "ssrl_xrd_tools[fitting,rsm,gui,napari]"`
gives you the full interactive notebook stack plus napari viewing
without the heavy `[notebook]` jupyterlab install (assumes you have
your own Jupyter environment already).

### Updating

```bash
mamba activate xrd
uv pip install -U xdart                # or:  pip install -U xdart
# library only:
uv pip install -U ssrl_xrd_tools       # or:  pip install -U ssrl_xrd_tools
```

### Editable / developer install

```bash
git clone -b dev https://github.com/v-thampy/ssrl_xrd_tools.git
git clone -b dev https://github.com/v-thampy/xdart.git         # optional
mamba activate xrd
uv pip install -e ./ssrl_xrd_tools     # or:  pip install -e ./ssrl_xrd_tools
uv pip install -e ./xdart              # or:  pip install -e ./xdart   (optional)
```

### Conda-forge (in progress)

A `conda-forge` recipe is in submission. Once it merges:

```bash
mamba create -n xrd -c conda-forge ssrl_xrd_tools xdart
```

Until then, the three-step pip install above is the recommended route.

### Bulk one-line installers (experimental)

For users who want a single command that builds the env, installs the scientific stack, and installs both packages, we ship bash and PowerShell installer scripts. **These are not fully shaken out yet** — in particular, the Windows PowerShell variant currently runs reliably only inside Git Bash on at least one test machine, not from a stock Anaconda Prompt or PowerShell. If the installer trips, fall back to the three-step `pip install` above.

**Linux / macOS (bash):**

```bash
curl -sSL https://raw.githubusercontent.com/v-thampy/ssrl_xrd_tools/dev/scripts/install.sh | bash
```

**Windows (PowerShell):**

```powershell
iex "& { $(iwr -useb https://raw.githubusercontent.com/v-thampy/ssrl_xrd_tools/dev/scripts/install.ps1) }"
```

Options accepted by both: `-n <name>` (env name, default `xrd`), `-p <ver>` / `--python <ver>` (Python version), `--bootstrap` / `-Bootstrap` (install Miniforge if no conda is found), `--force` / `-Force` (replace existing env), `--no-xdart` / `-NoXdart` (skip the GUI), `--dev` / `-Dev` (editable install from a local clone), `--branch <name>` / `-Branch <name>` (non-default git branch).

## Quick Start

### Basic Integration: Load, Integrate, Save

```python
from ssrl_xrd_tools.io import read_image
from ssrl_xrd_tools.integrate import load_poni, integrate_1d
from ssrl_xrd_tools.io import write_nexus

# Load calibration
poni = load_poni("path/to/calibration.poni")

# Read detector image
img = read_image("path/to/image.h5")

# Perform azimuthal integration
result_1d = integrate_1d(img, poni)

# Save to NeXus file
write_nexus("output.h5", result_1d)
```

### Unit Conversion

```python
from ssrl_xrd_tools.transforms import tth_to_q, tth_to_dspacing

# Convert scattering angle to Q-vector magnitude
q = tth_to_q(two_theta, energy=15.0)  # energy in keV

# Or to d-spacing
d = tth_to_dspacing(two_theta)
```

### Batch Processing

```python
from ssrl_xrd_tools.integrate import process_series

# Process multiple scans with consistent settings
results = process_series(
    scan_path="./raw_data",
    poni_file="calibration.poni",
    output_dir="./processed"
)
```

### Grazing Incidence Integration

```python
from ssrl_xrd_tools.integrate import integrate_gi_1d

# Perform grazing incidence integration
result = integrate_gi_1d(
    img,
    poni=poni,
    alpha_f=0.5,  # exit angle in degrees
)
```

### Reciprocal Space Mapping

```python
from ssrl_xrd_tools.rsm import ExperimentConfig, RSMVolume

config = ExperimentConfig(
    energy=15.0,
    lattice_params={'a': 3.85, 'b': 3.85, 'c': 12.69},
    crystal_system='tetragonal'
)

volume = RSMVolume(
    scans=scan_list,
    config=config,
    output_shape=(512, 512, 512)
)

# Map to reciprocal space
rsm_data = volume.to_hkl_grid()
```

## Module Architecture

The library is organized in layered modules, each with a distinct responsibility:

### Core (`core`)

Foundational data structures and HDF5 serialization:

- `IntegrationResult1D`, `IntegrationResult2D` — Containers for integration results (Q/2θ/d-spacing axes, intensity, error, metadata)
- `PONI` — Detector calibration (distance, center, rotation, detector model)
- `ScanMetadata` — Experimental metadata (energy, temperature, scan command, timestamps)
- **HDF5 Codec** — Universal encoder/decoder for NumPy, pandas, and native Python types; enables storing arbitrary Python objects in HDF5 with automatic type recovery

### I/O (`io`)

Reading and writing detector images, metadata, and standard formats:

- **Image Reading** — fabio-based support for TIFF, HDF5, ADSC, Pilatus, Mar, Rigaku, and other formats; parallel reading
- **SPEC Parsing** — Extract scan geometries, counter data, and metadata from SPEC files
- **NeXus I/O** — Full read/write support for NeXus HDF5 with proper group structure and attribute encoding
- **Export Formats** — XYE (simple ASCII), HDF5 (with codec), NeXus
- **Metadata Loading** — Image headers, SPEC file metadata, PDI (proprietary beamline format) parsing
- **Tiled Integration** — Query and fetch runs from Bluesky/Tiled data servers

### Transforms (`transforms`)

Unit conversions and angular calculations:

- Q-vector magnitude (Å⁻¹) ↔ scattering angle (2θ, degrees)
- d-spacing (Å) ↔ 2θ
- Energy conversions
- HKL to Miller index and back
- Polarization and geometric corrections

### Integration (`integrate`)

Azimuthal integration and calibration:

- **Single Geometry** — 1D and 2D azimuthal integration via pyFAI integrators
- **Grazing Incidence (GID)** — Fiber integration for surface-sensitive diffraction; exit-angle and polar integrations
- **Multi-Geometry Stitching** — Combine multiple detector angles into seamless 1D/2D patterns
- **Calibration** — Load/save PONI files; pyFAI detector database; manual refinement support
- **Batch Processing** — Process entire directories; directory watching for automated pipeline; parallel integration

### Corrections (`corrections`)

Experimental correction modules (stub stage):

- Detector sensitivity (planned: flat-field correction)
- Beam profile normalization
- Air scattering subtraction
- Polarization correction

### RSM (`rsm`)

Reciprocal space mapping for crystallographic analysis:

- **ExperimentConfig** — Define crystal structure, reciprocal lattice parameters, and Q-space geometry
- **RSMVolume** — Build 3D reciprocal space grids from series of scans; interpolation and rebinning
- **HKL Mapping** — Convert detector/Q-space coordinates to crystallographic indices
- **Volume Export** — Save to HDF5 or VTK for visualization

### Analysis (`analysis`)

Advanced data analysis and characterization:

- **Fitting** (`fitting`) — lmfit-based peak fitting (`fit_peaks`), multi-phase pattern fitting (`PhaseFitter`), custom models including a Chebyshev background, and the `background` submodule (SNIP etc.)
- **Phase Models** (`phase`) — `PhaseModel` wrapping pymatgen structures, peak position / template-intensity generation from CIF, used by `PhaseFitter`
- **Strain Analysis** (`strain`) — sin²(ψ) method: χ-sector extraction, per-sector peak fitting, linear regression; optional `(E, ν)` for stress output
- **Texture Analysis** (`texture`) — *placeholder* — planned pole figures and ODF
- **Refinement** (`refinement`) — *placeholder* — planned GSAS-II / FullProf / lmfit-Rietveld backends

### GUI

`ssrl_xrd_tools` is a headless library and does not ship its own GUI. The
primary graphical front end is [**xdart**](https://github.com/v-thampy/xdart),
a standalone PySide6 desktop app that imports `ssrl_xrd_tools` as a
dependency.

## API Examples

### Reading and Writing Data

```python
from ssrl_xrd_tools.io import read_image, write_h5, read_nexus
from ssrl_xrd_tools.core import IntegrationResult1D

# Read various image formats
img = read_image("image.h5")          # HDF5
img = read_image("image.tiff")        # TIFF
img = read_image("image_001.cbf")     # CBF/MAR

# Write integration results
result_1d = IntegrationResult1D(...)
write_h5("result.h5", result_1d)

# Read NeXus
data = read_nexus("data.nxs", entry="/entry_0")
```

### Reading Processed Scan Files

Once a scan has been reduced (by xdart or the headless pipeline) the
results live in a v2 NeXus `.nxs` file. The `get_*` convenience readers
pull 1D / 2D patterns, thumbnails, and metadata back out in one line —
no xarray knowledge required. A processed file is a **scan**; each
reader takes a **frame** label or `None` for all frames.

```python
from ssrl_xrd_tools.io import (
    get_frames, get_metadata, get_1d, get_2d, get_thumbnail, open_scan,
)

get_frames("scan.nxs")              # -> array of frame labels, e.g. [1, 2, 3]
meta = get_metadata("scan.nxs")     # sample, energy, wavelength, axes, motors

r = get_1d("scan.nxs", frame=2)     # one frame:  r.q, r.intensity, r.sigma
allr = get_1d("scan.nxs")           # all frames: (n_frames, n_q)
cake = get_2d("scan.nxs", frame=2)  # cake.q, cake.chi, cake.intensity (n_chi, n_q)

# object-style sugar
scan = open_scan("scan.nxs")
scan.frames, len(scan), scan.get_1d(2), scan.metadata
```

For the full `xarray.Dataset` (lazy per-frame access over very large
scans) use `read_sphere` / `read_sphere_metadata`. A runnable walkthrough
is in [`examples/notebooks/07_reading_processed_nxs.ipynb`](examples/notebooks/).

### Working with SPEC Files

```python
from ssrl_xrd_tools.io import get_angles, get_scan_path_info, get_from_spec_file

# Get scan geometry from SPEC file
scan_type = get_spec_scan_type("data.spec", scan_number=1)
angles = get_angles("data.spec", scan_number=1)

# Extract metadata
energy, UB = get_energy_and_UB("data.spec")
```

### Performing Integrations

```python
from ssrl_xrd_tools.integrate import (
    integrate_1d, integrate_2d,
    integrate_gi_1d, integrate_gi_2d,
    stitch_1d, stitch_2d
)

# Standard azimuthal integration
result_1d = integrate_1d(img, poni, unit="Q_A^-1")
result_2d = integrate_2d(img, poni, bins=(256, 256))

# Grazing incidence
result_gi = integrate_gi_1d(img, poni, alpha_f=0.5)

# Multi-geometry stitching
results = [integrate_1d(img1, poni1), integrate_1d(img2, poni2)]
stitched = stitch_1d(results)
```

### Peak Fitting

```python
from ssrl_xrd_tools.analysis.fitting import fit_line_cut

# Fit peaks in 1D data
peaks = fit_line_cut(
    x=result_1d.x,
    y=result_1d.y,
    model="lorentzian",
    n_peaks=3
)

for peak in peaks:
    print(f"Center: {peak['center']:.3f}, FWHM: {peak['fwhm']:.3f}")
```

### Multi-Phase Pattern Fitting

```python
from ssrl_xrd_tools.analysis.phase import PhaseModel
from ssrl_xrd_tools.analysis.fitting.phase_fitting import PhaseFitter

# Build a phase from a CIF and compute its peaks
phase = PhaseModel.from_cif("Fe.cif", name="alpha-Fe")
phase.calculate_peaks(q_range=(1.0, 7.0))

# Fit a measured 1D pattern with one or more phases
fitter = PhaseFitter(q, intensity, background='snip')
fitter.add_phase(phase)
params = fitter.build_params()
result = fitter.fit(params)
```

## Example Notebooks

End-to-end, runnable demonstrations live in
[`examples/notebooks/`](examples/notebooks/) (each has a single
`✏️ Configuration` cell to edit, then runs top-to-bottom):

| #  | Notebook | What it shows |
| -- | -------- | ------------- |
| 01 | Batch integration | Per-frame `integrate_1d` / `integrate_2d` over an image directory + NeXus round-trip |
| 02 | MultiGeometry stitching | 1D + 2D stitching across detector-angle scans |
| 03 | Phase + peak fitting | `PhaseFitter` vs structure-agnostic `fit_peaks` side by side |
| 04 | Batch phase fitting | `fit_sequence` over many patterns → phase fractions + lattice trends |
| 05 | sin²ψ analysis | GI integration → χ-sector fits → strain / stress |
| 06 | Headless reduction pipeline | Canonical `ReductionPlan` + `Scan` + `Frame` + `NexusSink` (the path xdart uses) |
| 07 | **Reading processed `.nxs`** | Pull 1D / 2D / thumbnails / metadata back out with one-line `get_1d` / `get_2d` / `get_metadata` |

See [`examples/notebooks/README.md`](examples/notebooks/README.md) for the
full table, prerequisites, and install hints.

## Dependencies

The full extras matrix is documented in
[**Installation → What each extra enables**](#what-each-extra-enables).
Summary:

### Base (always installed)

- `numpy` / `scipy` — numerical arrays + scientific computing
- `xarray` — labelled n-D arrays (used in NeXus I/O)
- `h5py` / `nexusformat` — HDF5 + NeXus file access
- `fabio` — detector image I/O (TIFF, CBF, EDF, Eiger HDF5, …)
- `silx` — SPEC-file parsing + HDF5 utilities
- `pyFAI` — azimuthal integration engine
- `joblib` — parallel image loading
- `natsort` — natural file sorting
- `matplotlib` / `plotly` — `viz.mpl` + `viz.plotly` headless plotting

### Optional (install via extras)

| Extra        | Why you'd add it                                                        |
| ------------ | ----------------------------------------------------------------------- |
| `[fitting]`  | peak / phase / strain fitting via `analysis.fitting.*` (lmfit, pymatgen) |
| `[rsm]`      | reciprocal-space mapping via `rsm.*` (xrayutilities, pyevtk)            |
| `[gui]`      | Jupyter notebook widgets on top of plotly (ipywidgets, anywidget, ipyfilechooser) |
| `[notebook]` | self-contained Jupyter environment (ipykernel, jupyterlab)              |
| `[napari]`   | napari 3D viewer for RSM volumes                                        |
| `[tiled]`    | Tiled client for Bluesky-collected runs                                 |
| `[vtk]`      | back-compat alias for `pyevtk` (now also in `[rsm]`)                    |
| `[all]`      | everything except dev                                                    |
| `[dev]`      | test / build / release tooling (pytest, build, twine)                    |

Combine extras as needed: `pip install "ssrl_xrd_tools[fitting,rsm,gui]"`
is the typical "Jupyter notebook + RSM + phase fitting" combination.

## Python Version

Requires **Python ≥ 3.10**.

## Contributing

Contributions are welcome! Please submit pull requests or open issues on the [GitHub repository](https://github.com/v-thampy/ssrl_xrd_tools).

## License

License information coming soon. Please see the repository for details.

## Citation

Citation information coming soon. If you use `ssrl_xrd_tools` in your research, please check back for citation guidelines.

## Acknowledgments

`ssrl_xrd_tools` is developed at the [Stanford Synchrotron Radiation Lightsource (SSRL)](https://www-ssrl.slac.stanford.edu/), SLAC National Accelerator Laboratory. The project builds on the excellent `pyFAI` library and benefits from the broader scientific Python ecosystem including NumPy, SciPy, and lmfit.
