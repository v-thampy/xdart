# Cursor Prompts for `integrate/` Module

Use these prompts one at a time, in order. After each one, review the output,
accept/edit as needed, then move to the next.

---

## Prompt 1: `integrate/calibration.py`

```
Implement integrate/calibration.py. This module bridges our PONI dataclass with
pyFAI's AzimuthalIntegrator.

Functions to implement:

1. load_poni(path: Path | str) -> PONI
   - Parse a .poni file (INI-like format) and return our PONI dataclass
   - pyFAI poni files have keys: Distance, Poni1, Poni2, Rot1, Rot2, Rot3,
     Wavelength, Detector
   - Use pyFAI.load() internally to parse, then extract fields into our PONI

2. save_poni(poni: PONI, path: Path | str) -> None
   - Write our PONI dataclass to a .poni file on disk
   - Create an AzimuthalIntegrator from the PONI, then call its .save() method

3. poni_to_integrator(poni: PONI) -> pyFAI.AzimuthalIntegrator
   - Create a pyFAI AzimuthalIntegrator from our PONI dataclass
   - Set dist, poni1, poni2, rot1, rot2, rot3, wavelength
   - If poni.detector is set, use pyFAI.detectors.detector_factory() to get it
   - Return the configured integrator

4. get_detector(name: str) -> pyFAI.detectors.Detector
   - Thin wrapper around pyFAI's detector_factory
   - Raise ValueError with a helpful message if detector not found

5. get_detector_mask(name: str) -> np.ndarray | None
   - Get the bad-pixel mask from the pyFAI detector registry
   - Return None if detector not found (log warning)
   - NOTE: This function already exists in io/image.py — consider whether to
     re-export it from here or import from io. Prefer importing from io to avoid
     duplication.

Reference our PONI dataclass in core/containers.py. Also reference
../xdart/xdart/utils/containers/poni.py for how xdart does this (the
create_ai_from_dict function).

For grazing incidence support: add a function:
6. poni_to_fiber_integrator(
       poni: PONI,
       incident_angle: float,
       tilt_angle: float = 0.0,
       sample_orientation: int = 1,
       angle_unit: str = "deg",
   ) -> "FiberIntegrator"
   - Create a regular AzimuthalIntegrator via poni_to_integrator(poni)
   - Promote it: fi = ai.promote("FiberIntegrator")
   - If angle_unit == "deg", convert incident_angle and tilt_angle to radians
   - Call fi.reset_integrator(incident_angle, tilt_angle, sample_orientation)
   - Return the configured FiberIntegrator
   - If promote() fails (pyFAI < 2025.01), raise ImportError with helpful message
   - This is used by integrate/gid.py
   - No pygix dependency — FiberIntegrator is built into pyFAI

Follow conventions in CLAUDE.md. Export all public functions in __all__.
```

---

## Prompt 2: `integrate/single.py`

```
Implement integrate/single.py. This wraps pyFAI's AzimuthalIntegrator for
single-image integration.

Functions to implement:

1. integrate_1d(
       image: np.ndarray,
       ai: pyFAI.AzimuthalIntegrator,
       npt: int = 1000,
       unit: str = "q_A^-1",
       method: str = "csr",
       mask: np.ndarray | None = None,
       radial_range: tuple[float, float] | None = None,
       azimuth_range: tuple[float, float] | None = None,
       error_model: str | None = None,
       **kwargs,
   ) -> IntegrationResult1D
   - Call ai.integrate1d() with the given parameters
   - Convert pyFAI's result to our IntegrationResult1D dataclass
   - pyFAI returns (radial, intensity) or (radial, intensity, sigma) depending
     on error_model
   - Pass through **kwargs to pyFAI for any additional parameters

2. integrate_2d(
       image: np.ndarray,
       ai: pyFAI.AzimuthalIntegrator,
       npt_rad: int = 1000,
       npt_azim: int = 1000,
       unit: str = "q_A^-1",
       method: str = "csr",
       mask: np.ndarray | None = None,
       radial_range: tuple[float, float] | None = None,
       azimuth_range: tuple[float, float] | None = None,
       error_model: str | None = None,
       **kwargs,
   ) -> IntegrationResult2D
   - Call ai.integrate2d() with the given parameters
   - pyFAI's integrate2d returns a Integrate2dResult namedtuple with fields:
     intensity, radial, azimuthal (and sigma if error_model set)
   - Convert to our IntegrationResult2D dataclass
   - IMPORTANT: pyFAI returns intensity with shape (npt_azim, npt_rad).
     Our IntegrationResult2D expects shape (npt_rad, npt_azim), so transpose
     the intensity (and sigma if present).

3. integrate_scan(
       images: np.ndarray,
       ai: pyFAI.AzimuthalIntegrator,
       npt: int = 1000,
       unit: str = "q_A^-1",
       method: str = "csr",
       mask: np.ndarray | None = None,
       reduce: str = "sum",
       **kwargs,
   ) -> IntegrationResult1D
   - Integrate each frame in a 3D image stack, then combine
   - reduce='sum': sum all 1D patterns; reduce='mean': average them
   - Return a single IntegrationResult1D

CRITICAL: Never hardcode default values for npt, method, unit, radial_range, or
azimuth_range in function signatures beyond the reasonable defaults shown above.
These are ALWAYS user-configurable at runtime.

Import IntegrationResult1D, IntegrationResult2D from ssrl_xrd_tools.core.
Follow all conventions in CLAUDE.md.
```

---

## Prompt 3: `integrate/multi.py`

```
Implement integrate/multi.py. This wraps pyFAI's MultiGeometry for stitching
together images taken at different detector angles.

The key pattern: when the detector is scanned across angular positions (both
in-plane "del" and out-of-plane "nu"), each image gets its OWN
AzimuthalIntegrator with the detector angle encoded via rot1 and rot2.

Functions to implement:

1. create_multigeometry_integrators(
       base_poni: PONI,
       rot1_angles: np.ndarray | Sequence[float],
       rot2_angles: np.ndarray | Sequence[float] | None = None,
   ) -> list[pyFAI.AzimuthalIntegrator]
   - Takes a base PONI (the zero-angle calibration) and arrays of detector
     angles (in DEGREES — convert to radians internally)
   - For each image index, create a new AzimuthalIntegrator from base_poni
     with rot1 = base_rot1 + deg2rad(rot1_angles[i])
     and rot2 = base_rot2 + deg2rad(rot2_angles[i]) if provided
   - If rot2_angles is None, only rot1 varies (single-axis scan)
   - Use poni_to_integrator from calibration.py to create the base, then
     clone and modify rot1/rot2 for each image
   - Return list of AzimuthalIntegrators, one per image

2. stitch_1d(
       images: list[np.ndarray] | np.ndarray,
       integrators: list[pyFAI.AzimuthalIntegrator],
       npt: int = 1000,
       unit: str = "q_A^-1",
       method: str = "BBox",
       radial_range: tuple[float, float] | None = None,
       mask: np.ndarray | None = None,
       normalization: np.ndarray | None = None,
       **kwargs,
   ) -> IntegrationResult1D
   - Create pyFAI MultiGeometry from the list of integrators
   - If normalization is provided (e.g., monitor counts i1 per image),
     divide each image by its normalization value before integration
   - Call mg.integrate1d(images, npt, lst_mask=..., ...)
   - lst_mask should be [mask] * len(images) if mask is provided
   - Convert result to IntegrationResult1D
   - method default is 'BBox' (not 'csr') because MultiGeometry works best
     with BBox

3. stitch_2d(
       images: list[np.ndarray] | np.ndarray,
       integrators: list[pyFAI.AzimuthalIntegrator],
       npt_rad: int = 1000,
       npt_azim: int = 1000,
       unit: str = "q_A^-1",
       method: str = "BBox",
       radial_range: tuple[float, float] | None = None,
       azimuth_range: tuple[float, float] | None = None,
       mask: np.ndarray | None = None,
       **kwargs,
   ) -> IntegrationResult2D
   - Same pattern as stitch_1d but using mg.integrate2d()
   - Convert to IntegrationResult2D (handle transpose as in single.py)

Import PONI from core, poni_to_integrator from integrate.calibration.
Import IntegrationResult1D, IntegrationResult2D from core.
Follow conventions in CLAUDE.md.
```

---

## Prompt 4: `integrate/gid.py`

```
Implement integrate/gid.py. This provides grazing incidence X-ray diffraction
(GIXRD) integration using pyFAI's FiberIntegrator — NO pygix dependency.

pyFAI >= 2025.01 includes pyFAI.integrator.fiber.FiberIntegrator which fully
replaces pygix for GIXRD work. A FiberIntegrator is created by "promoting"
a regular AzimuthalIntegrator:

    from ssrl_xrd_tools.integrate.calibration import poni_to_integrator
    ai = poni_to_integrator(poni)
    fi = ai.promote("FiberIntegrator")

FiberIntegrator inherits from AzimuthalIntegrator and adds grazing-incidence
methods. Key API details (from pyFAI source):
- incident_angle, tilt_angle: in RADIANS by default (angle_unit="rad")
  Pass angle_unit="deg" to use degrees instead.
- sample_orientation: int 1–8 (EXIF convention). Default 1 = detector
  horizontal, beam from left. SSRL typically uses 1 (check at runtime).
- reset_integrator(incident_angle, tilt_angle, sample_orientation) to
  update geometry parameters between frames.
- IMPORTANT: GI integration should be used WITHOUT pixel-splitting!
  Use method="no" or method="nosplit_csr" (NOT the default "csr").

Available integration methods on FiberIntegrator:
- integrate_fiber(data, npt, ...) → 1D radial profile
  Alias: integrate1d_grazing_incidence()
- integrate2d_fiber(data, npt_rad, npt_azim, ...) → 2D cake
  Alias: integrate2d_grazing_incidence()
- integrate1d_polar(data, npt, ...) → 1D polar profile
- integrate2d_polar(data, npt_rad, npt_azim, ...) → 2D (Q, Chi)
- integrate1d_exitangles(data, npt, ...) → 1D vs exit angles
- integrate2d_exitangles(data, npt_rad, npt_azim, ...) → 2D (Qxy, Qz)

Functions to implement:

1. create_fiber_integrator(
       poni: PONI,
       incident_angle: float,
       tilt_angle: float = 0.0,
       sample_orientation: int = 1,
       angle_unit: str = "deg",
   ) -> "FiberIntegrator"
   - Create an AzimuthalIntegrator from poni using poni_to_integrator()
   - Promote it: fi = ai.promote("FiberIntegrator")
   - If angle_unit == "deg", convert incident_angle and tilt_angle to
     radians internally (fi expects radians by default)
   - Call fi.reset_integrator(incident_angle, tilt_angle, sample_orientation)
   - Return the configured FiberIntegrator
   - If promote() fails (old pyFAI), raise ImportError with message:
     "FiberIntegrator requires pyFAI >= 2025.01. Upgrade with: pip install -U pyFAI"

2. integrate_gi_1d(
       image: np.ndarray,
       fi: "FiberIntegrator",
       npt: int = 1000,
       unit: str = "q_A^-1",
       method: str = "no",
       mask: np.ndarray | None = None,
       radial_range: tuple[float, float] | None = None,
       azimuth_range: tuple[float, float] | None = None,
       incident_angle: float | None = None,
       tilt_angle: float | None = None,
       sample_orientation: int | None = None,
       **kwargs,
   ) -> IntegrationResult1D
   - If incident_angle or tilt_angle or sample_orientation is provided,
     call fi.reset_integrator() to update (convert degrees to radians)
   - Call fi.integrate_fiber(image, npt, unit=unit, method=method,
     mask=mask, radial_range=radial_range, azimuth_range=azimuth_range,
     **kwargs)
   - Convert result to IntegrationResult1D
   - Default method="no" because pixel-splitting is discouraged for GI

3. integrate_gi_2d(
       image: np.ndarray,
       fi: "FiberIntegrator",
       npt_rad: int = 500,
       npt_azim: int = 500,
       unit: str = "q_A^-1",
       method: str = "no",
       mask: np.ndarray | None = None,
       radial_range: tuple[float, float] | None = None,
       azimuth_range: tuple[float, float] | None = None,
       incident_angle: float | None = None,
       tilt_angle: float | None = None,
       sample_orientation: int | None = None,
       **kwargs,
   ) -> IntegrationResult2D
   - Same reset_integrator pattern as above
   - Call fi.integrate2d_fiber(image, npt_rad, npt_azim, ...)
   - Convert to IntegrationResult2D (handle transpose as in single.py)

4. integrate_gi_polar(
       image: np.ndarray,
       fi: "FiberIntegrator",
       npt_rad: int = 500,
       npt_azim: int = 500,
       unit: str = "q_A^-1",
       method: str = "no",
       mask: np.ndarray | None = None,
       incident_angle: float | None = None,
       tilt_angle: float | None = None,
       sample_orientation: int | None = None,
       **kwargs,
   ) -> IntegrationResult2D
   - Call fi.integrate2d_polar(image, npt_rad, npt_azim, ...)
   - Returns (Q, Chi) 2D map
   - Convert to IntegrationResult2D

5. integrate_gi_exitangles(
       image: np.ndarray,
       fi: "FiberIntegrator",
       npt_rad: int = 500,
       npt_azim: int = 500,
       unit: str = "q_A^-1",
       method: str = "no",
       mask: np.ndarray | None = None,
       incident_angle: float | None = None,
       tilt_angle: float | None = None,
       sample_orientation: int | None = None,
       **kwargs,
   ) -> IntegrationResult2D
   - Call fi.integrate2d_exitangles(image, npt_rad, npt_azim, ...)
   - Returns (Qxy, Qz) 2D map — the reciprocal space representation
   - Convert to IntegrationResult2D

Type hints: use TYPE_CHECKING guard for FiberIntegrator type annotation:
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from pyFAI.integrator.fiber import FiberIntegrator

Import poni_to_integrator from integrate.calibration.
Import IntegrationResult1D, IntegrationResult2D from core.
Follow conventions in CLAUDE.md.
```

---

## Prompt 5: `integrate/batch.py`

```
Implement integrate/batch.py. This provides batch processing of scans and
directory watching for live data reduction at the beamline.

Functions/classes to implement:

1. process_scan(
       scan_dir: Path | str,
       ai: pyFAI.AzimuthalIntegrator,
       output_path: Path | str,
       npt: int = 4000,
       npt_rad: int = 1000,
       npt_azim: int = 1000,
       unit: str = "q_A^-1",
       method: str = "csr",
       mask: np.ndarray | None = None,
       azimuth_range: tuple[float, float] | None = None,
       radial_range: tuple[float, float] | None = None,
       threshold: float = 1e9,
       detector: str = "",
       rotation: int = 0,
       reprocess: bool = False,
       **kwargs,
   ) -> Path
   - Process all frames in a scan directory or HDF5 file
   - For each frame: read image, apply threshold mask, integrate 1D + 2D
   - Save results to HDF5 using io.export.write_h5()
   - Skip already-processed frames unless reprocess=True (check if frame
     group already exists in the output HDF5)
   - Return path to the output HDF5 file
   - Use ssrl_xrd_tools.io.image for reading, ssrl_xrd_tools.integrate.single
     for integration, ssrl_xrd_tools.io.export for writing

2. process_series(
       scan_paths: Sequence[Path | str],
       ai: pyFAI.AzimuthalIntegrator,
       output_dir: Path | str,
       reprocess: bool = False,
       **kwargs,
   ) -> list[Path]
   - Process multiple scans in sequence
   - Pass through all kwargs to process_scan
   - Return list of output paths
   - Log progress for each scan

3. class DirectoryWatcher:
   """Watch a directory tree for new data files and process them automatically.

   This replaces the common beamline pattern of:
       while True:
           scan_h5s = find_new_scans(base_path)
           for scan in scan_h5s:
               process_scan(scan, ...)
           time.sleep(30)
   """

   def __init__(
       self,
       watch_dir: Path | str,
       ai: pyFAI.AzimuthalIntegrator,
       output_dir: Path | str,
       patterns: Sequence[str] = ("*_master.h5", "*.edf", "*.raw"),
       recursive: bool = True,
       poll_interval: float = 10.0,
       **process_kwargs,
   ):
       - Store configuration
       - Initialize set of already-processed files
       - Try to import watchdog; fall back to polling if unavailable

   def start(self) -> None:
       - Start watching (blocking). Use watchdog.observers.Observer if available,
         otherwise use a polling loop with time.sleep(poll_interval)
       - On new file detected: call process_scan()
       - Handle KeyboardInterrupt gracefully (log "Stopping..." and clean up)

   def start_background(self) -> threading.Thread:
       - Start watching in a daemon thread (for GUI integration with xdart)
       - Return the thread object so caller can join/stop it

   def stop(self) -> None:
       - Signal the watcher to stop (set a threading.Event)

   @property
   def processed_files(self) -> set[Path]:
       - Return the set of files that have been processed

NOTE: watchdog is an OPTIONAL dependency. The polling fallback must work
without it. Add watchdog to optional dependencies in the docstring but don't
add it to pyproject.toml — we'll do that separately.

Import from ssrl_xrd_tools.io.image (find_image_files, read_image, etc.)
Import from ssrl_xrd_tools.integrate.single (integrate_1d, integrate_2d)
Import from ssrl_xrd_tools.io.export (write_h5)
Follow conventions in CLAUDE.md.
```

---

## Prompt 6: `integrate/__init__.py`

```
Update integrate/__init__.py to export the key functions from all submodules.

Export from calibration: load_poni, save_poni, poni_to_integrator, poni_to_fiber_integrator, get_detector
Export from single: integrate_1d, integrate_2d, integrate_scan
Export from multi: create_multigeometry_integrators, stitch_1d, stitch_2d
Export from gid: create_fiber_integrator, integrate_gi_1d, integrate_gi_2d, integrate_gi_polar, integrate_gi_exitangles
Export from batch: process_scan, process_series, DirectoryWatcher

Follow conventions in CLAUDE.md. Include __all__.
```

---

## Prompt 7: Update CLAUDE.md

After all modules are implemented, ask Cursor:

```
Update CLAUDE.md to reflect that the integrate/ module is now implemented.
Change the status markers from STUB to implemented for:
- integrate/calibration.py
- integrate/single.py
- integrate/multi.py
- integrate/gid.py
- integrate/batch.py

Also note that watchdog is an optional dependency for directory watching.
GIXRD uses pyFAI's built-in FiberIntegrator (no pygix dependency).
```

---

## Order of execution

1. calibration.py (foundation — other modules import from it)
2. single.py (depends on calibration + core)
3. multi.py (depends on calibration + core)
4. gid.py (depends on calibration + core, pyFAI FiberIntegrator)
5. batch.py (depends on single + io)
6. __init__.py (wires everything together)
7. Update CLAUDE.md

Review each one before moving to the next. If Cursor makes mistakes on
conventions, correct it once and it will adjust for subsequent prompts.
