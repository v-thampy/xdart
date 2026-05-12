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

The recommended path is a `pip install` into a fresh conda environment. This pulls the wheel from PyPI and brings in `ssrl_xrd_tools` (the analysis library) plus the full scientific stack automatically.

### Quick install (3 steps)

**1. Install conda** (skip if you already have `mamba`, `conda`, or an Anaconda Prompt installed).

Pick one:

- [**Miniforge**](https://github.com/conda-forge/miniforge/releases/latest) — recommended; conda-forge first, smaller install, ships with `mamba`.
- [**Miniconda**](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html) — Anaconda's minimal distribution.

On Windows the installer creates a "Miniforge Prompt" / "Anaconda Prompt" shortcut — open that for the next steps.
On macOS / Linux, open a regular terminal; the installer adds `mamba`/`conda` to your shell.

**2. Create a Python 3.12 environment.**

```bash
mamba create -n xrd python=3.12 -y
mamba activate xrd
```

(Substitute `conda` for `mamba` if you prefer — mamba is just the faster solver.)

**3. Install xdart.**

```bash
pip install xdart
```

This pulls `xdart`, `ssrl_xrd_tools`, and all dependencies (pyFAI, h5py, fabio, silx, PySide6, pyqtgraph, …) from PyPI. After it finishes:

```bash
xdart          # launch the GUI
```

### Updating

```bash
mamba activate xrd
pip install -U xdart
```

### Editable / developer install

Clone the repos and install in editable mode into the same environment:

```bash
git clone -b dev https://github.com/v-thampy/ssrl_xrd_tools.git
git clone -b dev https://github.com/v-thampy/xdart.git
mamba activate xrd
pip install -e ./ssrl_xrd_tools
pip install -e ./xdart
```

### Conda-forge (in progress)

A `conda-forge` recipe is in submission. Once it merges, installation becomes:

```bash
mamba create -n xrd -c conda-forge ssrl_xrd_tools xdart
mamba activate xrd
xdart
```

Until then, the three-step pip install above is the recommended route.

### Bulk one-line installers (experimental)

For users who want a single command that creates the env, installs the scientific stack, and installs both packages, we ship bash and PowerShell installer scripts. **These are not fully shaken out yet** — in particular, on Windows the current PowerShell script ran cleanly only when invoked from within Git Bash on at least one test machine, rather than from a stock Anaconda Prompt or PowerShell. If the installer trips, fall back to the three-step `pip install` above.

**Linux / macOS (bash):**

```bash
curl -sSL https://raw.githubusercontent.com/v-thampy/ssrl_xrd_tools/dev/scripts/install.sh | bash
```

**Windows (PowerShell):**

```powershell
iex "& { $(iwr -useb https://raw.githubusercontent.com/v-thampy/ssrl_xrd_tools/dev/scripts/install.ps1) }"
```

Both installers create a conda env called `xrd` with Python 3.12. Options accepted by both: `-n <name>` (env name), `-p <ver>` / `--python <ver>` (Python version), `--bootstrap` / `-Bootstrap` (install Miniforge if no conda is found), `--force` / `-Force` (replace existing env), `--no-xdart` / `-NoXdart` (skip the GUI), `--dev` / `-Dev` (editable install from a local clone), `--branch <name>` / `-Branch <name>` (pick a non-default git branch).

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
