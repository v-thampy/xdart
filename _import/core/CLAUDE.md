# CLAUDE.md — Project Context for AI Assistants

## What This Project Is

`ssrl_xrd_tools` is a Python library for synchrotron X-ray diffraction (XRD) data processing, built for workflows at the Stanford Synchrotron Radiation Lightsource (SSRL). It provides headless-first tools for image I/O, reciprocal space mapping (RSM), azimuthal integration, corrections, analysis (peak fitting, strain, texture), and an optional GUI layer.

The companion repo `xdart` (at `../xdart/`) is a PyQt5-based GUI for XRD integration using pyFAI. It is being refactored so that xdart becomes a thin GUI consumer of `ssrl_xrd_tools` — all shared logic (integration, I/O, corrections, calibration, data containers) lives here, and xdart imports from this package.

## Architecture & Module Map

```
ssrl_xrd_tools/
├── core/           # ✅ Shared types, containers, serialization
│   ├── containers.py   # XRDImage, PONI, IntegrationResult, nzarray
│   ├── metadata.py     # ScanMetadata (source-agnostic scan metadata)
│   └── hdf5.py         # HDF5 serialization codec (extracted from xdart)
├── io/             # File I/O — reads data, returns core types
│   ├── image.py        # ✅ detector-agnostic image I/O via fabio + HDF5
│   ├── spec.py         # ✅ SPEC file parsing via silx
│   ├── nexus.py        # ✅ NeXus/HDF5 reader + writer (Bluesky input; processed-result output)
│   ├── tiled.py        # ✅ Bluesky/Tiled catalog reader
│   ├── export.py       # ✅ write_xye, write_csv, write_h5
│   └── metadata.py     # ✅ txt/pdi metadata readers + unified read_image_metadata
├── corrections/    # ✅ Detector & beam corrections
│   ├── detector.py     # ✅ subtract_dark, apply_flatfield, apply_threshold, apply_mask, combine_masks, correct_image
│   ├── beam.py         # ✅ polarization_correction, solid_angle_correction, absorption_correction
│   └── normalization.py # ✅ normalize_monitor, normalize_time, normalize_stack, scale_to_range
├── transforms/     # Unit conversions
│   └── __init__.py     # ✅ q_to_tth, tth_to_q, d_to_q, energy_to_wavelength, etc.
├── integrate/      # pyFAI azimuthal integration wrappers
│   ├── calibration.py  # ✅ load_poni, save_poni, poni_to_integrator, get_detector
│   ├── single.py       # ✅ integrate_1d, integrate_2d, integrate_scan
│   ├── multi.py        # ✅ MultiGeometry stitching (create_multigeometry_integrators, stitch_1d/2d)
│   ├── gid.py          # ✅ GIXRD via pyFAI FiberIntegrator: integrate_gi_1d/2d/polar/exitangles + polar_1d/exitangles_1d line cuts
│   └── batch.py        # ✅ process_scan, process_series, DirectoryWatcher
├── rsm/            # Reciprocal space mapping via xrayutilities
│   ├── volume.py       # ✅ RSMVolume, extract_line_cut, extract_2d_slice, mask_data, save_vtk
│   ├── geometry.py     # ✅ DiffractometerConfig
│   ├── gridding.py     # ✅ grid_img_data, combine_grids, get_common_grid
│   └── pipeline.py     # ✅ ExperimentConfig, ScanInfo, process_scan, load_images
├── analysis/       # Data analysis
│   ├── fitting/
│   │   ├── models.py   # ✅ 1D/2D peak models (Gaussian, Lorentzian², PseudoVoigt, etc.)
│   │   ├── fit.py      # ✅ fit_line_cut, fit_2d_slice (lmfit-based)
│   │   ├── peaks.py    # STUB: find_peaks, peak_table
│   │   └── background.py # STUB: snip, rolling_ball, fit_background
│   ├── phase.py        # STUB: match_phase, load_cif
│   ├── texture.py      # STUB: chi_series, pole_figure, odf_analysis
│   ├── strain.py       # STUB: sin2chi_analysis, d_spacing_map
│   └── refinement.py   # STUB: lebail_fit, rietveld_fit (future: GSAS-II wrapper)
└── gui/            # Thin GUI layer — imports from all above modules
    ├── main.py         # Entry point (registered as `xdart` console script)
    ├── widgets/        # Reusable GUI widgets
    ├── rsm_viewer.py
    ├── napari_viewer.py
    ├── powder_1d_viewer.py
    └── powder_2d_viewer.py
```

## Key Design Principles

### 1. Dependency direction: always inward
```
gui → analysis/integrate/rsm → corrections → io → core
```
Never import from a higher layer into a lower one. `core/` has zero internal dependencies. `io/` depends only on `core/`. Everything else builds on these.

### 2. Headless first
Every function must work without a GUI. The GUI is just one consumer. This enables: batch processing, HPC jobs, Jupyter notebooks, Bluesky integration, and scripting.

### 3. Source-agnostic metadata
We are migrating from SPEC files to Bluesky/Tiled. With Bluesky, scan data and
metadata will be saved as NeXus-formatted HDF5 files (NXentry/NXdata/NXinstrument
hierarchy) and/or stored in a Tiled database. The `ScanMetadata` dataclass (in `core/metadata.py`) carries the actual data (angles, energy, UB
matrix, counters) rather than pointers to source files. Separate reader functions
produce `ScanMetadata` objects:
- `io/spec.py` — reads from SPEC files (current, via silx)
- `io/nexus.py` — reads from NeXus/HDF5 files (Bluesky output)
- `io/tiled.py` — reads from Tiled catalog (Bluesky database)
Processing code never knows the data source.

### 4. Stateless functions with config dataclasses
Prefer pure functions that take a config dataclass + data and return results. See `process_scan_data()` and `grid_img_data()` as examples. Avoid classes with mutable state for processing logic. Dataclasses with `slots=True` are used for all data containers.

### 5. NaN-based masking
Masked/bad pixels are set to `np.nan` throughout the pipeline. This is consistent across image I/O, integration, and analysis.

## Code Conventions

- **Python >=3.12**, use modern type hints (`X | Y` union syntax, not `Optional[X]`)
- **`from __future__ import annotations`** at the top of every module
- **`@dataclass(slots=True)`** for all data containers
- **`Path | str`** for file path arguments, convert internally with `Path(path)`
- **Logging**: use `logger = logging.getLogger(__name__)`, never `print()`
- **Docstrings**: NumPy-style with Parameters/Returns sections
- **Error handling**: log + return None for recoverable errors; raise for programming errors. Use `strict=True` parameter pattern to optionally re-raise.
- **Imports**: absolute imports within the package (`from ssrl_xrd_tools.io.image import ...`)

## Key Dependencies & Their Roles

| Package | Role | Used in |
|---------|------|---------|
| numpy, scipy | Core numerics | Everywhere |
| fabio | Image format detection/reading | `io/image.py` |
| h5py | HDF5 file I/O | `io/image.py`, `io/nexus.py`, `core/hdf5.py` |
| silx | SPEC file parsing | `io/spec.py` |
| pyFAI | Azimuthal integration, detector registry, calibration | `io/image.py`, `integrate/` |
| xrayutilities | Diffractometer geometry, HKL conversion, gridding | `rsm/` |
| lmfit | Peak fitting engine | `analysis/fitting/` |
| joblib | Parallel image loading | `io/image.py` |
| natsort | Natural file sorting | `io/image.py` |

**Optional**: `panel`/`holoviews`/`bokeh`/`napari` (GUI), `pyevtk` (VTK export), `watchdog` (filesystem-event-based directory watching — `DirectoryWatcher` falls back to polling when not installed)

## The xdart Relationship

xdart (`../xdart/`) is a PyQt5 + pyqtgraph application for interactive XRD integration. Key architecture:

- **EwaldArch**: Single detector image + pyFAI AzimuthalIntegrator + integration results
- **EwaldSphere**: Collection of EwaldArch objects + MultiGeometry integration
- **GUI**: Tab-based (static_scan, ttheta_scan), with H5Viewer, IntegratorTree, DisplayFrame
- **Storage**: HDF5 with custom serialization codec in `_utils.py` (extracted to
  `core/hdf5.py`). For **new workflows**, use `io/nexus.write_nexus` /
  `open_nexus_writer` / `write_nexus_frame` instead — they produce self-describing
  NeXus files and support SWMR for live beamline use.

**Refactoring plan**: Extract from xdart into ssrl_xrd_tools:
1. HDF5 serialization codec (`_utils.py`) → `core/hdf5.py` ✅
2. PONI calibration container (`containers/poni.py`) → `core/containers.py`
3. Integration result containers (`containers/int_data.py`) → `core/containers.py`
4. Sparse arrays (`containers/nzarrays.py`) → `core/containers.py`
5. Image loading logic → already replaced by `io/image.py`
6. SPEC metadata parsing → already replaced by `io/spec.py`
10. Per-image metadata (txt/pdi sidecar files) → `io/metadata.py` ✅
7. Integration logic (from EwaldArch) → `integrate/single.py`, `integrate/multi.py` ✅ (Phase 3)
8. 2D fitting models (`utils/lmfit_models.py`) → already in `analysis/fitting/models.py`
9. Processed-result HDF5 output → `io/nexus.write_nexus` ✅ (replaces xdart codec for new workflows)

After extraction, xdart imports from `ssrl_xrd_tools` and becomes a thin GUI shell.

## What's Implemented vs. Stub

**Working code** (test/use these, don't rewrite from scratch), sorted by module path:
- `analysis/fitting/fit.py` — fit_line_cut, fit_2d_slice (172 lines)
- `analysis/fitting/models.py` — 1D/2D peak models for lmfit (335 lines)
- `core/containers.py` — PONI, IntegrationResult1D, IntegrationResult2D dataclasses
- `core/metadata.py` — ScanMetadata dataclass
- `corrections/beam.py` — polarization_correction, solid_angle_correction, absorption_correction
- `corrections/detector.py` — subtract_dark, apply_flatfield, apply_threshold, apply_mask, combine_masks, correct_image
- `corrections/normalization.py` — normalize_monitor, normalize_time, normalize_stack, scale_to_range
- `integrate/batch.py` — process_scan, process_series, DirectoryWatcher (watchdog or polling fallback)
- `integrate/calibration.py` — load_poni, save_poni, poni_to_integrator, get_detector, get_detector_mask
- `integrate/gid.py` — GIXRD via pyFAI FiberIntegrator: create_fiber_integrator, integrate_gi_1d/2d/polar/exitangles, integrate_gi_polar_1d/exitangles_1d (1D line cuts)
- `integrate/multi.py` — create_multigeometry_integrators, stitch_1d, stitch_2d (MultiGeometry stitching)
- `integrate/single.py` — integrate_1d, integrate_2d, integrate_scan; explicit `polarization_factor` and `normalization_factor` params (see note below)
- `io/export.py` — write_xye, write_csv, write_h5
- `io/metadata.py` — read_txt_metadata, read_pdi_metadata, read_image_metadata
- `io/image.py` — detector-agnostic image I/O via fabio + HDF5 (220 lines)
- `io/nexus.py` — read_nexus, find_nexus_image_dataset, list_entries; write_nexus, open_nexus_writer, write_nexus_frame (NeXus/HDF5 reader + writer; write_nexus replaces the custom xdart HDF5 codec for new workflows — LZF compression, SWMR support for live beamline reduction)
- `io/spec.py` — SPEC file parsing via silx (97 lines)
- `io/tiled.py` — read_tiled_run, connect_tiled, list_scans (Bluesky/Tiled reader, optional dep)
- `rsm/` — ✅ split into volume.py, geometry.py, gridding.py, pipeline.py (tests in test_rsm.py)
- `transforms/__init__.py` — q↔tth, d↔q, energy↔wavelength conversions

> **Polarization & normalization — two approaches, pick one per pipeline:**
> `integrate_1d` / `integrate_2d` accept `polarization_factor` and `normalization_factor`
> as explicit keyword arguments and pass them directly to pyFAI (applied *during* binning,
> before any output is written). Alternatively, `corrections/beam.py::polarization_correction`
> and `corrections/normalization.py::normalize_monitor` apply the same corrections to the
> raw pixel array *before* integration. Do not apply both; pick whichever fits your pipeline.

**Stubs** (docstrings describe intended API, implement these):
- `analysis/fitting/peaks.py`, `analysis/fitting/background.py`
- `analysis/phase.py`, `analysis/texture.py`, `analysis/strain.py`, `analysis/refinement.py`

## `core/` Module

Implemented shared primitives:

```python
# core/metadata.py — Source-agnostic scan metadata
@dataclass(slots=True)
class ScanMetadata:
    scan_id: str
    energy: float               # keV
    wavelength: float           # Angstroms
    angles: dict[str, np.ndarray]   # motor_name -> values per point
    counters: dict[str, np.ndarray] # counter_name -> values per point
    ub_matrix: np.ndarray | None = None
    sample_name: str = ""
    scan_type: str = ""
    source: str = ""            # "spec", "tiled", "hdf5"
    image_paths: list[Path] = field(default_factory=list)
    h5_path: Path | None = None
    extra: dict = field(default_factory=dict)
```

```python
# core/containers.py — Shared data containers
@dataclass(slots=True)
class PONI:
    """pyFAI calibration geometry."""
    dist: float
    poni1: float
    poni2: float
    rot1: float = 0.0
    rot2: float = 0.0
    rot3: float = 0.0
    wavelength: float = 0.0
    detector: str = ""

@dataclass(slots=True)
class IntegrationResult1D:
    """Result of 1D azimuthal integration."""
    radial: np.ndarray          # 2theta or q axis
    intensity: np.ndarray
    sigma: np.ndarray | None = None
    unit: str = "2th_deg"
    # ... pixel counts, raw signal for proper normalization

@dataclass(slots=True)
class IntegrationResult2D:
    """Result of 2D (cake) integration."""
    radial: np.ndarray          # 2theta or q axis
    azimuthal: np.ndarray       # chi axis
    intensity: np.ndarray       # 2D array
    sigma: np.ndarray | None = None
    unit: str = "2th_deg"
```

## Completed: RSM Split

`rsm/core.py` (776 lines) has been split into:
- `rsm/volume.py` — RSMVolume class with slicing, cropping, VTK export, line cuts, 2D slices
- `rsm/geometry.py` — DiffractometerConfig with make_hxrd
- `rsm/gridding.py` — grid_img_data, combine_grids, get_common_grid
- `rsm/pipeline.py` — ExperimentConfig, ScanInfo, process_scan, process_scan_data, load_images

`rsm/__init__.py` re-exports everything for backward compatibility.

## Real Workflow Patterns (from experimental notebooks)

These patterns come from actual beamline notebooks and define the workflows that
`ssrl_xrd_tools` must support cleanly. Any new API should make these workflows
simpler than the current notebook code.

### Beamlines & Detectors

Detectors are NOT fixed to beamlines — the same beamline may use different
detectors depending on the experiment. All beamlines can use any of the
available detectors. Design accordingly: never hardcode detector-beamline pairs.

Available detectors (all in pyFAI registry):
- Pilatus 100k (195×487, 172µm pixels) — raw binary files
- Pilatus 300k (619×487, 172µm pixels) — EDF or raw files
- Eiger 1M — HDF5 master files (`*_master.h5`)
- Eiger 4M (2162×2068, 75µm pixels) — HDF5 master files
- Pilatus 4M (planned future)

Detector mounting varies: landscape or portrait orientation, so images may need
`rotation` (multiples of 90°) or `flip`. This is specified at runtime.

Beamline motor configurations (typical, but can vary):
- **BL2-1**: `del` (detector 2theta), `nu` (out-of-plane), `eta`, `phi`.
  Normalization by `i1` (monitor) counter.
- **BL7-2**: RSM motors: `VTH`, `Chi`, `Phi`, `VTTH`.
- **BL17-2**: `eta`, `phi`. Calibrated via `.poni` files.

### Workflow 1: Single-image pyFAI integration (1D + 2D cake)

Used for powder XRD at BL17-2 with Eiger 4M. Pattern:
1. Load `.poni` calibration → `ai = pyFAI.load(poni_file)`
2. Get detector mask from the integrator
3. For each scan, for each frame in the Eiger HDF5:
   - Read single frame via fabio
   - Apply mask + threshold (typically 1e9)
   - `ai.integrate1d(image, npt=4000, mask=mask, method='csr', unit='q_A^-1')`
   - `ai.integrate2d(image, npt_rad=1000, npt_azim=1000, azimuth_range=(-125, -5), ...)`
4. Save per-frame results to HDF5: `{scan}_processed.h5` with groups per frame
   containing `q`, `I`, `IQChi`, `Q`, `Chi` datasets
5. Often runs in a continuous `while True` loop for live data reduction at the beamline

**Key parameters**: `npt=4000` (1D), `npt_rad=1000, npt_azim=1000` (2D),
`method='csr'`, `unit='q_A^-1'`, threshold `1e7`–`1e9`

### Workflow 2: MultiGeometry stitching

Used when the detector is scanned across angular positions during a measurement.
Both `del` (in-plane 2theta) and `nu` (out-of-plane) can be scanned, either
individually or simultaneously. These motor angles map to pyFAI's `rot1` and
`rot2` parameters on the AzimuthalIntegrator. Pattern:
1. Manual calibration: scan direct beam across detector → fit peak positions →
   compute sample-detector distance in pixels → create `AzimuthalIntegrator` with
   computed `dist`, `poni1`, `poni2`, `wavelength`
2. For each scan:
   - Read SPEC file to get `del`, `nu`, `i0`, `i1` arrays
   - For each image point: read raw file, divide by `i1`, create a separate
     `AzimuthalIntegrator` with `rot1=deg2rad(del[idx])`, `rot2=deg2rad(nu[idx])`
   - `mg = MultiGeometry(ais, unit='q_A^-1', radial_range=...)`
   - `int1d = mg.integrate1d(imgs, npts, lst_mask=mask, method='BBox')`
3. Save to `.xye` and `.csv` files; cache in pickle dict
4. Also runs in continuous loop for live reduction

**Key details**:
- Each image gets its OWN AzimuthalIntegrator with the detector angle baked in
  via `rot1`/`rot2`. The mask is a full-detector mask, not per-pixel.
- Integration method (`'csr'`, `'BBox'`, etc.), number of points, radial range,
  azimuthal range, and unit are all specified at runtime — never hardcoded.
- Can be used with any detector (Pilatus 100k/300k, Eiger, etc.).

### Workflow 3: Reciprocal space mapping (RSM)

Used at BL7-2 for single-crystal thin films. Pattern:
1. Configure diffractometer geometry:
   - `sampleAxes` (e.g., `['z-', 'y+', 'z-']` for VTH, Chi, Phi)
   - `detAxes` (e.g., `['z-']` for VTTH)
   - `idir` (incident beam, typically `[0, 1, 0]`)
   - `ndir` (surface normal, typically `[0, 0, 1]`)
   - `camera_or` (detector pixel orientation, e.g., `('x-', 'z-')`)
   - Direct beam pixel position and pixel size in mm
2. Read energy and UB matrix from SPEC scan header (`G3` header line)
3. Get diffractometer angles from SPEC (scanning + fixed motors)
4. Load image stack (parallel via joblib)
5. `xu.QConversion(sampleAxes, detAxes, idir)` → `xu.HXRD(..., qconv=qconv)`
6. `hxrd.Ang2Q.init_area(*camera_or, **header)` where header has `cch1`, `cch2`,
   `pwidth1`, `pwidth2`, `distance`, `Nch1`, `Nch2`
7. `qx, qy, qz = hxrd.Ang2Q.area(*angles, UB=UB)`
8. `xu.Gridder3D(*bins)` → `gridder(qx, qy, qz, img_stack)`
9. Result: (h_axis, k_axis, l_axis, intensity_3D) → pickle + VTK export
10. Multi-scan: concatenate raw qx/qy/qz/data from all scans, grid once

**Key detail**: The `header` dict uses detector coordinates in mm (not meters)
with keys `cch1/cch2/pwidth1/pwidth2/distance/Nch1/Nch2`. The `camera_or`
orientation depends on how the detector is physically mounted and varies between
setups. ROI cropping requires adjusting `cch1/cch2`.

### Workflow 4: Grazing incidence diffraction (GIXRD)

Used for thin film characterization. The sample is at a fixed shallow incidence
angle, and the scattering is analyzed in the plane of the sample (in-plane, Qxy)
and perpendicular to it (out-of-plane, Qz).

**pyFAI FiberIntegrator** (pyFAI >= 2025.01) handles all GIXRD integration
without any external dependency beyond pyFAI itself — no `pygix` required.
`integrate/gid.py` wraps it with a stable, convention-consistent API:

```python
from ssrl_xrd_tools.integrate.gid import create_fiber_integrator, integrate_gi_1d, integrate_gi_2d

fi = create_fiber_integrator(poni, incident_angle=0.2, angle_unit="deg")
r1d = integrate_gi_1d(image, fi, npt=1000, unit="qoop_A^-1")
r2d = integrate_gi_2d(image, fi, npt_rad=500, npt_azim=500, unit="qip_A^-1")
rp  = integrate_gi_polar(image, fi, npt_rad=500, npt_azim=500)      # (Q, Chi)
rex = integrate_gi_exitangles(image, fi, npt_rad=500, npt_azim=500) # (Qxy, Qz)
```

Under the hood, `create_fiber_integrator` promotes an `AzimuthalIntegrator` via
`ai.promote("FiberIntegrator")` and caches the incident/tilt angles on the
instance (pyFAI resets its internal state after each call, so the wrapper
re-injects them on every integration call).

Available integration modes (all return `IntegrationResult1D` or `IntegrationResult2D`):
- `integrate_gi_1d` — 1D radial profile (`integrate1d_grazing_incidence`)
- `integrate_gi_2d` — 2D cake in (Qip, Qoop) (`integrate2d_grazing_incidence`)
- `integrate_gi_polar` — 2D (Q, Chi) map (`integrate2d_polar`)
- `integrate_gi_exitangles` — 2D (Qxy, Qz) reciprocal space map (`integrate2d_exitangles`)
- `integrate_gi_polar_1d` — 1D line cut from (Q, Chi) map (`integrate1d_polar`) → `IntegrationResult1D`
- `integrate_gi_exitangles_1d` — 1D line cut from exit-angle space (`integrate1d_exitangles`) → `IntegrationResult1D`

Key parameters:
- `incident_angle`, `tilt_angle`: accepted in **degrees** by `create_fiber_integrator`
  when `angle_unit="deg"` (converted to radians internally)
- `sample_orientation`: 1–8 (EXIF convention); default 1 = horizontal detector
- **IMPORTANT**: GI integration should be used WITHOUT pixel-splitting!
  Use `method="no"` or `method="nosplit_csr"` (not the default CSR).
- `MultiGeometryFiber` is available for multi-position GI experiments (not yet wrapped).

### Workflow 5: Viewing integrated data

Interactive Jupyter widgets for exploring pre-reduced data:
- Toggle between `q` and `2theta` x-axis: `tth = 2 * rad2deg(arcsin(12.398 / (4*pi*energy) * q))`
- Log/linear y-scale
- Overlay multiple scans
- For 2D: pcolormesh of I(Q, Chi) with adjustable colormap percentile clipping
- For RSM: 3-panel view (HK, KL, HL projections) + 1D line cuts below each
- HKL range sliders for cropping the 3D volume interactively

### Workflow 6: Folder watching for live data reduction

Critical for beamline operations. Both scripted notebooks and xdart need to
monitor a directory (including subdirectories) for new data files and
automatically process them as they arrive.

`integrate/batch.py` provides `DirectoryWatcher` (and `process_scan` /
`process_series` for one-shot use). `DirectoryWatcher`:
- Watches a directory tree for new image files (HDF5 master files, raw, EDF, etc.)
- Tracks which files have already been processed via an in-memory set; per-frame
  skip logic is handled inside `process_scan` via the HDF5 output file.
- Supports graceful stop via `KeyboardInterrupt` (blocking) or `DirectoryWatcher.stop()` (background thread).
- Works both as a blocking call (`dw.start()`) and as a daemon thread (`dw.start_background()`).
- Uses `watchdog` for filesystem events when installed; falls back to polling
  (configurable via `poll_interval`) for network-mounted filesystems (common at SSRL).

```python
from ssrl_xrd_tools.integrate.batch import DirectoryWatcher, process_scan, process_series
```

### Unit conversion formula used throughout

```python
tth = 2 * np.rad2deg(np.arcsin(12.398 / (4 * np.pi * energy_keV) * q))
# where 12.398 ≈ hc in keV·Å
```

This should go into `transforms/` as `q_to_tth(q, energy)` and `tth_to_q(tth, energy)`.

### Export formats

- `.xye`: 3-column whitespace-separated (x, y, error) — standard powder diffraction format
- `.csv`: comma-separated version of same
- `.pkl`: pickle of dict or tuple (used for caching processed data)
- `.h5`: HDF5 with per-frame groups containing q, I, IQChi arrays
- `.vtk`: VTK rectilinear grid for 3D RSM visualization (ParaView/VESTA)



- Use `pytest` (in dev dependencies)
- Test data: typically .edf/.tif images + .spec files from SSRL beamlines
- RSM tests need: SPEC file with UB matrix, detector images, xrayutilities
- Fitting tests can use synthetic data (Gaussian peaks + noise)

## Common Tasks

### Adding a new module (e.g., implementing a stub)
1. Read the stub's docstring for the intended API
2. Follow the conventions above (type hints, dataclasses, NumPy docstrings)
3. Add exports to the subpackage `__init__.py`
4. If the module introduces new shared types, put them in `core/`

### Extracting code from xdart
1. Find the relevant code in `../xdart/xdart/` (usually in `utils/` or `modules/ewald/`)
2. Refactor to remove PyQt/GUI dependencies
3. Replace xdart-specific containers with `ssrl_xrd_tools` dataclasses
4. Ensure the function works headless (no Qt imports at module level)
