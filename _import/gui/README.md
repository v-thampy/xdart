# xdart — X-ray Diffraction Analysis in Real Time

A pyFAI-based desktop GUI for real-time azimuthal integration and visualization of synchrotron X-ray diffraction data. Built with PySide6 and pyqtgraph for high-performance interactive plotting.

## Overview

xdart enables fast, intuitive analysis of X-ray diffraction (XRD) data from synchrotron sources. Whether you're monitoring a live experiment or processing batch data offline, xdart integrates seamlessly with pyFAI's proven azimuthal integration algorithms while providing a responsive graphical interface optimized for detector images and reciprocal space visualization.

### Key Capabilities

- **Real-time 1D/2D azimuthal integration** using pyFAI
- **Batch processing** of image series with parallel multicore support
- **Grazing incidence diffraction (GID)** integration using pyFAI FiberIntegrator
- **Live monitoring** of ongoing experiments with directory-watched file ingestion
- **NeXus/HDF5 data format** with full metadata preservation
- **Interactive 2D detector image visualization** with zoom, pan, and masking tools
- **Unit conversion** between Q (Å⁻¹) and 2θ (°) with wavelength awareness
- **Background subtraction** (single file, series average, or directory-matched)
- **Masking tools** for bad pixels, beamstop shadows, and detector edges
- **Calibration management** via PONI files with visual feedback
- **Raw image preview thumbnails** for quick file browsing

## Installation

A single installer script handles everything: it creates a dedicated conda environment, installs the heavy scientific stack from conda-forge (`pyFAI`, `h5py`, `pymatgen`, Qt, HDF5 libraries), and installs both `xdart` and its computational core [`ssrl_xrd_tools`](https://github.com/v-thampy/ssrl_xrd_tools) on top. This conda-for-native + pip-for-Python split avoids the binary mismatches that can occur when mixing sources for native-backed packages.

**TL;DR — one-line install** (requires conda or mamba; see [below](#if-you-dont-have-condamamba) if you don't have either):

**Linux / macOS (bash)**

```bash
curl -sSL https://raw.githubusercontent.com/v-thampy/ssrl_xrd_tools/dev/scripts/install.sh | bash
```

**Windows (PowerShell)**

```powershell
iex "& { $(iwr -useb https://raw.githubusercontent.com/v-thampy/ssrl_xrd_tools/dev/scripts/install.ps1) }"
```

Then `conda activate xrd` and run `xdart`. Read on for prerequisites, options, and the developer workflow.

### Windows notes

The PowerShell installer (`install.ps1`) is the native Windows path and should be preferred. It has two advantages over piping the bash installer through Git Bash:

1. **It does not require Git Bash.** The script runs in a stock PowerShell or Windows Terminal session.
2. **It finds your existing conda install.** If you already have Anaconda, Miniconda, or Miniforge installed, the script locates `conda.exe` by inspecting the standard install directories (`%USERPROFILE%\miniforge3`, `%USERPROFILE%\anaconda3`, `C:\ProgramData\Miniconda3`, etc.) — even when `conda init powershell` has not been run. That means you don't need to re-install conda just because `conda` isn't on PowerShell's `PATH`.

If you prefer the bash installer, you'll still need [Git for Windows](https://git-scm.com/download/win) (which provides Git Bash) and conda must be activated inside that shell. The PowerShell path skips both requirements.

### One-line install (recommended)

No clone required — just run:

**Linux / macOS**

```bash
curl -sSL https://raw.githubusercontent.com/v-thampy/ssrl_xrd_tools/dev/scripts/install.sh | bash
```

**Windows**

```powershell
iex "& { $(iwr -useb https://raw.githubusercontent.com/v-thampy/ssrl_xrd_tools/dev/scripts/install.ps1) }"
```

This creates a new conda environment called `xrd` containing Python 3.12, the full scientific stack, `xdart`, and `ssrl_xrd_tools`. After it finishes:

```bash
conda activate xrd
xdart          # launch the GUI
```

> **Why `xrd`?** The script's default conda environment name is `xrd` — a short, memorable alias for "X-Ray Diffraction". This is the environment you'll activate whenever you want to use `xdart` or `ssrl_xrd_tools`. If you prefer a different name (e.g. to keep multiple versions side-by-side), pass `-n <name>`:
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
git clone -b dev https://github.com/v-thampy/xdart.git
git clone -b dev https://github.com/v-thampy/ssrl_xrd_tools.git
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

### Launch the GUI

```bash
conda activate xrd
xdart
```

Or from Python:

```python
from xdart.xdart_main import main
main()
```

### Basic Workflow

1. **Launch xdart** and wait for the main window to open
2. **Set calibration**: Browse and select your PONI calibration file in the right panel
3. **Select data**: Choose an image file or directory containing images
4. **Choose processing mode**: Pick from Batch 1D, Live 1D, Batch 2D, Live 2D, or Viewer
5. **Configure parameters** (optional): Set background subtraction, masking, or advanced integration options
6. **Click Start**: Processing begins; monitor progress and view results in real time

## Usage Guide

### Loading and Processing Data

#### Batch Mode (Batch 1D, Batch 2D)
Process all frames in a selected image file or directory at once. Ideal for complete datasets acquired in a single experiment.

1. Select your PONI file
2. Select your image file or directory
3. Choose "Batch 1D" or "Batch 2D" from the mode dropdown
4. Adjust processing parameters as needed
5. Click Start

Results are saved as NeXus HDF5 files (.nxs) in the output directory.

#### Live Mode (Live 1D, Live 2D)
Monitor a directory for new image files and integrate them as they arrive. Perfect for real-time feedback during an active beamline experiment.

1. Set PONI file and image directory
2. Choose "Live 1D" or "Live 2D"
3. Click Start
4. xdart watches the directory and processes new files automatically

#### Viewer Mode
Browse and display previously saved NeXus files without re-integrating.

### Integration Axes and Units

The 1D integration panel's axis dropdown lets you choose between:

- **Q (Å⁻¹)**: Scattering vector magnitude; independent of wavelength
- **2θ (°)**: Scattering angle; depends on the wavelength in your PONI file

The 2D integration panel offers:

- **Q-χ**: Radial-azimuthal in reciprocal space
- **2θ-χ**: Radial-azimuthal in angle space

Unit conversion respects your calibration file's wavelength automatically.

### Grazing Incidence Diffraction (GID)

For surface-sensitive measurements, xdart supports grazing incidence geometry using pyFAI's FiberIntegrator:

1. In the **Grazing Incidence** section of the parameter tree, enable GI mode
2. Specify the incident angle and sample normal direction (if using advanced geometry)
3. The integrator panel switches to GI-specific modes:
   - **1D modes**: Qip (in-plane), Qoop (out-of-plane), Q-total
   - **2D modes**: Qip-Qoop, Q-χ
4. Process as normal; output will reflect the rotated reciprocal space axes

### Advanced Integration Settings

Click **"Advanced..."** next to the processing mode dropdown to access detailed pyFAI parameters:

- **Solid angle correction**: Account for detector solid angle variations
- **Dummy values**: Mark pixels to ignore in integration
- **Polarization factor**: Apply polarization correction for synchrotron radiation
- **Integration method**: Choose the algorithm (e.g., histogram, csr, full-split)
- **Radial range**: Manually clip Q or 2θ range (overrides auto-detection)
- **Azimuthal range**: Select only certain χ sectors

### Background Subtraction

Configure in the **Background** section of the parameter tree:

- **No background**: Raw data only
- **Single file**: Subtract a single dark image
- **Series average**: Average all images in a background directory, then subtract
- **Directory-matched**: Match each sample image to a background image by filename pattern

Background frames are integrated using the same parameters as sample frames for consistency.

### Calibration and Masking

The integrator panel includes **Calibrate** and **Make Mask** buttons at the bottom:

- **Calibrate** launches the pyFAI-calib2 module for interactive detector calibration. Use a calibration standard (e.g., LaB6, CeO2) image to refine detector geometry and generate a PONI file.
- **Make Mask** launches the pyFAI mask drawing tool, where you can interactively draw regions on a detector image to create or edit a bad-pixel mask file.

Additional masking options are available in the parameter tree:

- **Preset masks**: Load a mask file from disk
- **Threshold masking**: Automatically mask pixels above or below intensity thresholds
- **Detector masks**: Apply detector-specific dead-pixel maps

Masks are saved with your results in the NeXus file for reproducibility.

### Data Export and Saving

- **Automatic export**: Processed 1D/2D data is saved as NeXus HDF5 (.nxs) files during batch processing
- **Manual export**: Use the **Save** button in the display area to export the currently displayed pattern
- **Metadata**: All integration parameters, calibration info, and masking are stored in the HDF5 structure

## Architecture

xdart is built on top of **ssrl_xrd_tools**, a standalone library for X-ray diffraction data processing. The GUI provides interactive access to the library's integration, I/O, and analysis capabilities, while maintaining tight integration with pyFAI for proven, well-tested algorithms.

## Configuration and Calibration

### PONI Files

xdart uses pyFAI PONI (PyFAI Object Containing Necessary Information) files for detector calibration. A PONI file contains:

- Detector geometry (pixel size, shape, name)
- Incident beam center location
- Sample-to-detector distance
- Wavelength
- Detector rotation (if any)

You can generate a PONI file using the **Calibrate** button in xdart's integrator panel (which launches pyFAI-calib2), or from the command line:

```bash
pyFAI-calib2
```

Refer to the pyFAI documentation for detailed calibration procedures.

## Dependencies

### Core Libraries

- **pyFAI**: Fast azimuthal integration and geometry handling
- **PySide6**: Modern Qt6 bindings for the GUI
- **pyqtgraph**: High-performance plotting and image visualization
- **ssrl_xrd_tools**: XRD data I/O and integration utilities
- **h5py & silx**: NeXus/HDF5 file support
- **fabio**: Image file I/O (CBF, EDF, TIFF, etc.)
- **numpy, scipy, scikit-image**: Numerical and image processing
- **pandas**: Data manipulation and analysis
- **matplotlib**: Publication-quality plotting

### Optional for Advanced Use

- **lmfit**: Peak fitting and parameterization
- **xrayutilities**: Additional X-ray geometry utilities
- **hvplot, holoviews, panel**: Interactive Jupyter visualization

See `pyproject.toml` for the complete dependency list.

## Troubleshooting

### Common Issues

**pyFAI installation fails on macOS/Windows:**
Try installing via conda instead of pip; conda packages include pre-built binaries.

```bash
conda install -c conda-forge pyfai
```

**GUI doesn't appear or crashes on startup:**
Check that PySide6 is properly installed and your Qt plugins are accessible:

```bash
python -c "from PySide6 import QtWidgets; print('PySide6 OK')"
```

**Slow integration or freezing:**
Ensure multicore processing is enabled in advanced settings, and reduce the image resolution or integration radial resolution if working with very large detectors.

**PONI file not recognized:**
Verify the PONI file is in the correct format (text-based key=value pairs) and paths are absolute or relative to the working directory.

## Contributing

Contributions are welcome! Please:

1. Fork the repository on GitHub
2. Create a feature branch (`git checkout -b my-feature`)
3. Make your changes and add tests
4. Submit a pull request with a clear description

For bug reports or feature requests, open an issue on [GitHub](https://github.com/v-thampy/xdart/issues).

## License

xdart is released under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## Citation

If you use xdart in your research, please cite:

```
xdart: X-ray Diffraction Analysis in Real Time
https://github.com/v-thampy/xdart
```

(Formal publication citation coming soon)

## Acknowledgments

Developed at the [Stanford Synchrotron Radiation Lightsource (SSRL)](https://www-ssrl.slac.stanford.edu/), SLAC National Accelerator Laboratory. Grateful thanks to the pyFAI community and all collaborators and users who provide feedback and improvements.

## Contact

For questions, feedback, or collaboration inquiries, reach out to:

**Vivek Thampy**
vthampy@stanford.edu

---

**xdart** | Python ≥ 3.10 | [Repository](https://github.com/v-thampy/xdart) | [Manual](https://github.com/v-thampy/xdart/blob/master/xdart_manual.pdf)
