# ssrl_xrd_tools — SSRL X-ray Diffraction Processing Toolkit

A Python library for processing synchrotron X-ray diffraction data, providing a complete pipeline from raw detector images to publication-ready results. Built on [pyFAI](https://pyfai.readthedocs.io/) for azimuthal integration with added support for grazing incidence, multi-geometry stitching, reciprocal space mapping, and advanced analysis.

## Overview

`ssrl_xrd_tools` is a standalone Python library developed at SSRL (SLAC National Accelerator Laboratory) for comprehensive X-ray diffraction (XRD) data processing. It serves as the computational core for the [xdart](https://github.com/v-thampy/xdart) interactive GUI but is equally suited for scripting, Jupyter notebooks, and automated batch pipelines at beamlines or in the lab.

The library handles the complete workflow from raw detector images through calibration, integration, unit conversion, and advanced analysis including peak fitting, phase identification, texture analysis, and strain calculation.

## Features

- **1D/2D Azimuthal Integration** — Fast azimuthal integration via pyFAI with full detector geometry support
- **Grazing Incidence Diffraction (GID)** — Specialized integration for surface-sensitive and thin-film measurements
- **Multi-Geometry Stitching** — Seamlessly combine multiple detector angles for extended Q-range coverage
- **Reciprocal Space Mapping (RSM)** — Generate 3D reciprocal space volumes with HKL gridding
- **NeXus/HDF5 I/O** — Standards-compliant data storage and retrieval; full NumPy/pandas object serialization
- **SPEC File Parsing** — Extract metadata, scan commands, and counter data from SPEC files
- **Peak Fitting & Analysis** — lmfit-based peak models with background subtraction and residual analysis
- **Phase Identification** — Automatic phase matching against structure databases
- **Texture Analysis** — Pole figures, orientation distribution functions (ODF)
- **Strain Analysis** — sin²χ method for stress/strain calculation from d-spacing maps
- **Batch Processing** — Directory watching and automated pipeline execution
- **Bluesky/Tiled Integration** — Access data from Bluesky-collected runs via Tiled

## Installation

Most users should install [**xdart**](https://github.com/v-thampy/xdart) — the desktop GUI — which pulls `ssrl_xrd_tools` in as a dependency automatically. A single installer script handles everything: it creates a dedicated conda environment, installs the heavy scientific stack from conda-forge (`pyFAI`, `h5py`, `pymatgen`, Qt, HDF5 libraries), and installs both `xdart` and `ssrl_xrd_tools` on top. This conda-for-native + pip-for-Python split avoids the binary mismatches that can occur when mixing sources for native-backed packages.

### One-line install (recommended)

No clone required — just run:

```bash
curl -sSL https://raw.githubusercontent.com/v-thampy/ssrl_xrd_tools/dev/scripts/install.sh | bash
```

This creates a new conda environment called `xrd` containing Python 3.12, the full scientific stack, `xdart`, and `ssrl_xrd_tools`. After it finishes:

```bash
conda activate xrd
xdart          # launch the GUI
```

> **Why `xrd`?** The script's default conda environment name is `xrd` — a short, memorable alias for "X-Ray Diffraction". This is the environment you'll activate whenever you want to use `ssrl_xrd_tools` or `xdart`. If you prefer a different name (e.g. to keep multiple versions side-by-side), pass `-n <name>`:
>
> ```bash
> curl -sSL https://raw.githubusercontent.com/v-thampy/ssrl_xrd_tools/dev/scripts/install.sh | bash -s -- -n myenv
> ```

### If you don't have conda/mamba

Pass `--bootstrap` to have the script download and install [miniforge](https://github.com/conda-forge/miniforge) automatically into `~/miniforge3`:

```bash
curl -sSL https://raw.githubusercontent.com/v-thampy/ssrl_xrd_tools/dev/scripts/install.sh | bash -s -- --bootstrap
```

### Installer options

```
-n, --name NAME       Conda environment name (default: xrd)
-p, --python VERSION  Python version (default: 3.12)
--bootstrap           Install miniforge to ~/miniforge3 if conda is missing
--branch BRANCH       Git branch to install from (default: dev)
--force               Replace an existing env of the same name
--no-xdart            Install ssrl_xrd_tools only, skip xdart
--dev                 Editable install (requires a local clone — see below)
```

### Updating

Re-run the installer with `--force`:

```bash
curl -sSL https://raw.githubusercontent.com/v-thampy/ssrl_xrd_tools/dev/scripts/install.sh | bash -s -- --force
```

### Development setup

Developers who want an editable install with immediate reloads should clone the repos and run the installer locally with `--dev`:

```bash
git clone -b dev https://github.com/v-thampy/ssrl_xrd_tools.git
git clone -b dev https://github.com/v-thampy/xdart.git         # optional, auto-detected if sibling
cd ssrl_xrd_tools
./scripts/install.sh --dev
```

The script will auto-detect a sibling `xdart` clone and install both in editable mode.

### Release branch

The installer currently points at the `dev` branch (both repos) while the APIs stabilize. Once the packages are more mature, the default will switch to `main`. To pin to a specific branch or tag at any time, pass `--branch <name>`.

### Conda-forge (coming eventually)

A conda-forge recipe is planned once the API stabilizes, which will make installation as simple as:

```bash
mamba create -n xrd -c conda-forge ssrl-xrd-tools xdart
```

Until then, the installer script above is the recommended path.

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

- **Fitting** (`fitting`) — lmfit-based peak fitting, background models, residual analysis
- **Phase Identification** (`phase`) — Phase matching, CIF database loading, structure factor calculation
- **Texture Analysis** (`texture`) — Pole figure generation, ODF calculation, texture strength metrics
- **Strain Analysis** (`strain`) — sin²χ method for biaxial/triaxial stress calculation; d-spacing mapping
- **Refinement** (`refinement`) — Stubs for future integration with GSAS-II, FullProf

### GUI (Optional)

Interactive visualization and data exploration (requires `gui` install):

- `xdart` — Standalone desktop application (see [xdart repository](https://github.com/v-thampy/xdart))

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

### Phase Matching

```python
from ssrl_xrd_tools.analysis.phase import match_phases

phases = match_phases(
    d_spacing_list=d_vals,
    database="ICSD"
)
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

### Optional

- **GUI** (`[gui]`) — `panel`, `holoviews`, `bokeh`, `napari`
- **Tiled** (`[tiled]`) — `tiled[client]` for Bluesky data access
- **Development** (`[dev]`) — `jupyterlab`, `matplotlib`, `pytest`, `ipykernel`, `ipywidgets`

## Python Version

Requires **Python >= 3.12**

## Contributing

Contributions are welcome! Please submit pull requests or open issues on the [GitHub repository](https://github.com/v-thampy/ssrl_xrd_tools).

## License

License information coming soon. Please see the repository for details.

## Citation

Citation information coming soon. If you use `ssrl_xrd_tools` in your research, please check back for citation guidelines.

## Acknowledgments

`ssrl_xrd_tools` is developed at the [Stanford Synchrotron Radiation Lightsource (SSRL)](https://www-ssrl.slac.stanford.edu/), SLAC National Accelerator Laboratory. The project builds on the excellent `pyFAI` library and benefits from the broader scientific Python ecosystem including NumPy, SciPy, and lmfit.
