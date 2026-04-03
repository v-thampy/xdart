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

### Prerequisites

- Python ≥ 3.10
- A working pyFAI installation (installation can be platform-dependent; conda is recommended)

### Recommended: Conda

```bash
conda create -n xdart python=3.12
conda activate xdart
pip install xdart --upgrade
```

### pip

```bash
pip install xdart
```

### Development Installation

```bash
git clone https://github.com/v-thampy/xdart.git
cd xdart
pip install -e ".[interactive]"
```

## Quick Start

### Launch the GUI

```bash
conda activate xdart
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

**xdart v0.15** | Python ≥ 3.10 | [Repository](https://github.com/v-thampy/xdart) | [Manual](https://github.com/v-thampy/xdart/blob/master/xdart_manual.pdf)
