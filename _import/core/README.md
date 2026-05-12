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

Most users should install [**xdart**](https://github.com/v-thampy/xdart) — the desktop GUI — which pulls `ssrl_xrd_tools` in as a dependency automatically. If you only want the library (e.g. for headless scripting or Jupyter), follow the same three-step pattern below and install `ssrl_xrd_tools` directly.

### Quick install (3 steps)

**1. Install conda** (skip if you already have `mamba`, `conda`, or an Anaconda Prompt installed).

Pick one:

- [**Miniforge**](https://github.com/conda-forge/miniforge/releases/latest) — recommended; conda-forge first, smaller install, ships with `mamba`.
- [**Miniconda**](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html) — Anaconda's minimal distribution.

**2. Create a Python 3.12 environment.**

```bash
mamba create -n xrd python=3.12 -y
mamba activate xrd
```

**3. Install xdart (recommended — brings `ssrl_xrd_tools` in automatically):**

```bash
pip install xdart
```

Or, library only:

```bash
pip install ssrl_xrd_tools
```

### Updating

```bash
mamba activate xrd
pip install -U xdart    # or: pip install -U ssrl_xrd_tools
```

### Editable / developer install

```bash
git clone -b dev https://github.com/v-thampy/ssrl_xrd_tools.git
git clone -b dev https://github.com/v-thampy/xdart.git         # optional
mamba activate xrd
pip install -e ./ssrl_xrd_tools
pip install -e ./xdart                                          # optional
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

## Dependencies

### Core

- `numpy` — Numerical arrays
- `scipy` — Scientific computing (interpolation, optimization)
- `h5py` — HDF5 file access
- `fabio` — Detector image I/O
- `silx` — Data visualization toolkit (used for NeXus and HDF5 utilities)
- `pyFAI` — Azimuthal integration engine
- `xrayutilities` — Crystallography calculations
- `joblib` — Parallel processing
- `natsort` — Human-friendly file sorting
- `lmfit` — Nonlinear least-squares fitting and peak modeling

### Optional extras

- **Napari** (`[napari]`) — `napari` viewer integration for 2D/3D image browsing
- **Tiled** (`[tiled]`) — `tiled[client]` for Bluesky data access
- **VTK** (`[vtk]`) — `pyevtk` for exporting reciprocal-space volumes to VTK
- **Development** (`[dev]`) — `pytest`, `build`, `twine`

Install with `pip install 'ssrl-xrd-tools[tiled,vtk]'` or via the installer script.

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
