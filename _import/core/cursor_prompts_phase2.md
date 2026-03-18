# Cursor Prompts — Phase 1 (Tests) + Phase 2 (Corrections) + API Expansion

Use these prompts one at a time, in order.  After each one, review the output,
accept/edit as needed, then move to the next.

---

## Phase 0: pyFAI API Expansion

These two prompts add commonly-needed parameters and missing 1D GI wrappers
before writing tests, so the tests cover the final API surface.

---

### Prompt 0A: Add explicit pyFAI parameters to `integrate/single.py`

```
Update integrate/single.py to add two commonly-used pyFAI parameters as
explicit named arguments on integrate_1d and integrate_2d (they already work
via **kwargs, but making them explicit gives autocomplete and documentation):

For both integrate_1d and integrate_2d, add these OPTIONAL parameters
AFTER error_model and BEFORE **kwargs:

    polarization_factor: float | None = None,
    normalization_factor: float | None = None,

- polarization_factor: Synchrotron X-rays are horizontally polarized.
  Typical value is ~0.99 for bending-magnet beamlines, ~1.0 for undulators.
  Passed directly to ai.integrate1d / ai.integrate2d as
  polarization_factor=polarization_factor.
  Only pass it if not None (so pyFAI's default is preserved).
- normalization_factor: Scales the result by 1/normalization_factor.
  Useful for monitor normalization (e.g., dividing by i1 counts).
  Passed directly to pyFAI as normalization_factor=normalization_factor.
  Only pass it if not None.

Implementation: build a dict of extra kwargs, update with **kwargs, then
unpack when calling ai.integrate1d / ai.integrate2d.  Example:

    extra = dict(**kwargs)
    if polarization_factor is not None:
        extra["polarization_factor"] = polarization_factor
    if normalization_factor is not None:
        extra["normalization_factor"] = normalization_factor
    result = ai.integrate1d(image, npt, ..., **extra)

Also update integrate_scan to accept and forward these two parameters.

Update docstrings (NumPy style) for all three functions.
Do NOT change any existing behavior — only add the new optional params.
Follow conventions in CLAUDE.md.
```

---

### Prompt 0B: Add 1D GI line-cut wrappers to `integrate/gid.py`

```
Add two new public functions to integrate/gid.py for 1D line-cut extraction
from GIWAXS data.  These wrap pyFAI FiberIntegrator's integrate1d_polar and
integrate1d_exitangles, which are commonly used to extract intensity vs Q or
intensity vs exit angle from a 2D reciprocal space map.

1. integrate_gi_polar_1d(
       image: np.ndarray,
       fi: "FiberIntegrator",
       npt: int = 1000,
       unit: str = "q_A^-1",
       method: str = "no",
       mask: np.ndarray | None = None,
       incident_angle: float | None = None,
       tilt_angle: float | None = None,
       sample_orientation: int | None = None,
       **kwargs,
   ) -> IntegrationResult1D
   - Same _effective_gi_params pattern as integrate_gi_1d
   - Determine radial_unit from unit: "A^-1" → "A^-1", else "nm^-1"
   - Call fi.integrate1d_polar(
         polar_degrees=True,
         radial_unit=radial_unit,
         data=image,
         npt_ip=npt,
         npt_oop=npt,
         sample_orientation=orient,
         method=method,
         mask=mask,
         incident_angle=inc,
         tilt_angle=tilt,
         **kwargs,
     )
   - Convert result to IntegrationResult1D
   - This gives a 1D profile: intensity vs Q_total, integrated over all
     polar angles (chi).

2. integrate_gi_exitangles_1d(
       image: np.ndarray,
       fi: "FiberIntegrator",
       npt: int = 1000,
       method: str = "no",
       mask: np.ndarray | None = None,
       incident_angle: float | None = None,
       tilt_angle: float | None = None,
       sample_orientation: int | None = None,
       **kwargs,
   ) -> IntegrationResult1D
   - Call fi.integrate1d_exitangles(
         angle_degrees=kwargs.pop("angle_degrees", True),
         data=image,
         npt_ip=npt,
         npt_oop=npt,
         sample_orientation=orient,
         method=method,
         mask=mask,
         incident_angle=inc,
         tilt_angle=tilt,
         **kwargs,
     )
   - Convert result to IntegrationResult1D
   - This gives intensity vs horizontal exit angle, integrated over all
     vertical exit angles.

Add both to __all__.
Also add both to integrate/__init__.py exports.
Follow conventions in CLAUDE.md.
```

---

## Phase 1: Tests

Create tests/ directory with a conftest.py and per-module test files.  All
tests use synthetic data (no real beamline images needed).

---

### Prompt 1A: `tests/conftest.py` — shared fixtures

```
Create tests/conftest.py with shared pytest fixtures for the test suite.

Fixtures to define:

1. poni_fixture -> PONI
   - Return a realistic PONI for an Eiger 4M at ~200mm distance:
     PONI(dist=0.2, poni1=0.081, poni2=0.0775, rot1=0.0, rot2=0.0,
          rot3=0.0, wavelength=1.0e-10, detector="eiger4m")
   - Scope: session (reused across all tests)

2. ai_fixture(poni_fixture) -> AzimuthalIntegrator
   - Use poni_to_integrator(poni_fixture)
   - Scope: session

3. synthetic_image() -> np.ndarray
   - 2D float64 array of shape (2162, 2068) (Eiger 4M dimensions)
   - Fill with a radial Gaussian ring pattern centered at the PONI to
     simulate a powder ring:
       y, x = np.mgrid[:2162, :2068]
       r = np.sqrt((y - 1081)**2 + (x - 1034)**2)
       image = 1000 * np.exp(-((r - 500) / 50)**2) + np.random.poisson(10, (2162, 2068))
   - Scope: session (expensive to generate)

4. synthetic_image_small() -> np.ndarray
   - Smaller 100x100 image for fast tests that don't need realistic geometry
   - Gaussian peak + Poisson noise
   - Scope: session

5. tmp_poni_file(poni_fixture, tmp_path_factory) -> Path
   - Save the poni_fixture to a temporary .poni file using save_poni
   - Return the path
   - Scope: session

6. synthetic_mask() -> np.ndarray
   - Boolean mask matching Eiger 4M shape, True for a 10-pixel border
   - Scope: session

Import from:
- ssrl_xrd_tools.core.containers (PONI)
- ssrl_xrd_tools.integrate.calibration (poni_to_integrator, save_poni)

Follow conventions in CLAUDE.md.
```

---

### Prompt 1B: `tests/test_core.py` — containers and metadata tests

```
Create tests/test_core.py with pytest tests for the core/ module.

Tests to write:

1. test_poni_creation():
   - Create a PONI with known values, verify all fields accessible
   - Verify slots=True works (no __dict__)

2. test_integration_result_1d_shapes():
   - Create IntegrationResult1D with matching arrays, verify no error
   - Create with mismatched shapes, verify ValueError

3. test_integration_result_1d_sigma():
   - Create with sigma=None, verify sigma is None
   - Create with matching sigma, verify it's stored
   - Create with wrong-shaped sigma, verify ValueError

4. test_integration_result_2d_shapes():
   - Create IntegrationResult2D with radial(100,), azimuthal(50,),
     intensity(100, 50), verify no error
   - Create with wrong intensity shape, verify ValueError

5. test_integration_result_2d_transpose_needed():
   - Create with intensity shape (50, 100) when radial=100, azimuthal=50
   - Should raise ValueError — caller must transpose

6. test_scan_metadata_creation():
   - Create a ScanMetadata with realistic values, verify all fields

7. test_scan_metadata_optional_fields():
   - Verify defaults: ub_matrix=None, sample_name="", etc.

Import from ssrl_xrd_tools.core.containers and ssrl_xrd_tools.core.metadata.
```

---

### Prompt 1C: `tests/test_transforms.py`

```
Create tests/test_transforms.py with pytest tests for ssrl_xrd_tools.transforms.

Tests to write:

1. test_energy_wavelength_roundtrip():
   - energy_to_wavelength(12.398) should be ~1.0 Å
   - wavelength_to_energy(energy_to_wavelength(10.0)) ≈ 10.0

2. test_q_tth_roundtrip():
   - For energy=12.0 keV, q=3.0 Å^-1:
     tth_to_q(q_to_tth(3.0, 12.0), 12.0) ≈ 3.0
   - Test with arrays too

3. test_d_q_roundtrip():
   - q_to_d(d_to_q(3.5)) ≈ 3.5
   - d_to_q(1.0) ≈ 2*pi

4. test_q_to_tth_known_value():
   - At 12.398 keV (λ = 1.0 Å), q = 4*pi*sin(θ)/λ
   - For q = 2*pi (d = 1 Å), tth should be 2*arcsin(1/(4*pi) * 2*pi * 12.398/(12.398))
     = 2*arcsin(0.5) = 60°

5. test_array_inputs():
   - All functions should accept and return numpy arrays
   - Test q_to_tth with array input

Use np.testing.assert_allclose with atol=1e-10 or rtol=1e-6 as appropriate.
```

---

### Prompt 1D: `tests/test_calibration.py`

```
Create tests/test_calibration.py with pytest tests for
ssrl_xrd_tools.integrate.calibration.

Use the fixtures from conftest.py (poni_fixture, ai_fixture, tmp_poni_file).

Tests to write:

1. test_load_poni_roundtrip(poni_fixture, tmp_poni_file):
   - Load the temporary .poni file with load_poni
   - Verify dist, poni1, poni2, wavelength match poni_fixture
   - Use pytest.approx for float comparison

2. test_save_poni_creates_file(poni_fixture, tmp_path):
   - Save to tmp_path / "test.poni", verify file exists

3. test_poni_to_integrator(poni_fixture):
   - Create AI from poni_fixture
   - Verify ai.dist, ai.poni1, etc. match
   - Verify ai.detector is not None when poni.detector is set

4. test_get_detector_known():
   - get_detector("Pilatus300k") should return a detector with correct shape

5. test_get_detector_unknown():
   - get_detector("NotADetector") should raise ValueError

6. test_get_detector_mask_known():
   - get_detector_mask("eiger4m") should return an ndarray or None
   - If ndarray, verify it's boolean-like and has correct shape

7. test_poni_to_fiber_integrator(poni_fixture):
   - Try to create a FiberIntegrator with incident_angle=0.5
   - If pyFAI >= 2025.01: verify it returns without error
   - If old pyFAI: verify ImportError is raised
   - Use pytest.importorskip or try/except to handle both cases
   - Mark with @pytest.mark.skipif if FiberIntegrator unavailable
```

---

### Prompt 1E: `tests/test_single.py` — integration tests

```
Create tests/test_single.py with pytest tests for
ssrl_xrd_tools.integrate.single.

Use fixtures from conftest.py (ai_fixture, synthetic_image, synthetic_mask).

Tests to write:

1. test_integrate_1d_returns_correct_type(ai_fixture, synthetic_image):
   - Call integrate_1d(synthetic_image, ai_fixture, npt=500)
   - Verify result is IntegrationResult1D
   - Verify result.radial.shape == (500,)
   - Verify result.intensity.shape == (500,)
   - Verify result.unit contains "q" or matches the unit arg

2. test_integrate_1d_with_mask(ai_fixture, synthetic_image, synthetic_mask):
   - Call with mask=synthetic_mask
   - Should return without error
   - Intensity should differ from unmasked result

3. test_integrate_2d_returns_correct_type(ai_fixture, synthetic_image):
   - Call integrate_2d(synthetic_image, ai_fixture, npt_rad=200, npt_azim=100)
   - Verify result is IntegrationResult2D
   - Verify result.radial.shape == (200,)
   - Verify result.azimuthal.shape == (100,)
   - Verify result.intensity.shape == (200, 100)  # our convention!

4. test_integrate_2d_transpose_convention(ai_fixture, synthetic_image):
   - Verify that intensity.shape == (npt_rad, npt_azim), NOT (npt_azim, npt_rad)
   - This is the most important convention test

5. test_integrate_scan_sum(ai_fixture, synthetic_image):
   - Stack 3 copies: images = np.stack([synthetic_image]*3)
   - Call integrate_scan(images, ai_fixture, npt=500, reduce="sum")
   - Verify result.intensity is roughly 3x a single integration
   - Use atol/rtol appropriate for integration noise

6. test_integrate_scan_mean(ai_fixture, synthetic_image):
   - Same but reduce="mean"
   - Verify result.intensity is roughly equal to a single integration

7. test_integrate_scan_invalid_reduce(ai_fixture, synthetic_image):
   - Stack 2 copies, call with reduce="invalid"
   - Verify ValueError

8. test_integrate_1d_polarization_factor(ai_fixture, synthetic_image):
   - Call with polarization_factor=0.99
   - Should not raise an error
   - Result should differ slightly from no polarization

9. test_integrate_1d_normalization_factor(ai_fixture, synthetic_image):
   - Call with normalization_factor=2.0
   - Intensity should be roughly half compared to normalization_factor=1.0

NOTE: These tests will be somewhat slow because they use full-size Eiger 4M
images.  Mark the slow ones with @pytest.mark.slow if you want.  For CI,
consider using synthetic_image_small with a generic integrator.
```

---

### Prompt 1F: `tests/test_multi.py`

```
Create tests/test_multi.py with pytest tests for
ssrl_xrd_tools.integrate.multi.

Use poni_fixture from conftest.py.

Tests to write:

1. test_create_integrators_count(poni_fixture):
   - rot1_angles = [0.0, 5.0, 10.0]
   - Result should have len == 3

2. test_create_integrators_rot1_offsets(poni_fixture):
   - rot1_angles = [0.0, 10.0]
   - Verify integrators[1].rot1 - integrators[0].rot1 ≈ deg2rad(10)

3. test_create_integrators_rot2(poni_fixture):
   - rot1_angles = [0.0, 5.0], rot2_angles = [0.0, 3.0]
   - Verify integrators[1].rot2 - integrators[0].rot2 ≈ deg2rad(3)

4. test_create_integrators_mismatched_lengths(poni_fixture):
   - rot1_angles = [0, 1, 2], rot2_angles = [0, 1]
   - Should raise ValueError

5. test_stitch_1d_runs(poni_fixture):
   - Create 3 integrators with rot1 = [0, 5, 10]
   - Create 3 small synthetic images (100x100 each)
   - Create a simple AI from poni with a small detector override
     (use generic detector to avoid shape mismatch)
   - Call stitch_1d with npt=200
   - Verify result is IntegrationResult1D with radial.shape == (200,)

6. test_stitch_1d_normalization(poni_fixture):
   - Same setup but with normalization=[1.0, 2.0, 0.5]
   - Should not raise
   - Result should differ from un-normalized

7. test_stitch_2d_runs(poni_fixture):
   - Same 3-integrator setup
   - Call stitch_2d with npt_rad=100, npt_azim=50
   - Verify result.intensity.shape == (100, 50)

NOTE: MultiGeometry tests are tricky because the integrators need to have a
detector that matches the image shape.  For these tests, override the PONI
detector field to "" and set the detector directly on each AI:
    from pyFAI.detectors import Detector
    det = Detector(pixel1=75e-6, pixel2=75e-6, max_shape=(100, 100))
    ai.detector = det
This avoids needing real Eiger 4M images for MultiGeometry tests.
```

---

### Prompt 1G: `tests/test_gid.py`

```
Create tests/test_gid.py with pytest tests for
ssrl_xrd_tools.integrate.gid.

ALL tests in this file should be marked with:
    pytestmark = pytest.mark.skipif(
        not _HAS_FIBER,
        reason="FiberIntegrator requires pyFAI >= 2025.01",
    )

At the top of the file, probe for FiberIntegrator:
    try:
        from pyFAI.integrator.fiber import FiberIntegrator
        _HAS_FIBER = True
    except ImportError:
        _HAS_FIBER = False

Use poni_fixture from conftest.py.

Tests to write:

1. test_create_fiber_integrator(poni_fixture):
   - Call create_fiber_integrator(poni_fixture, incident_angle=0.3)
   - Verify result has _gi_incident_angle attribute ≈ deg2rad(0.3)
   - Verify type name contains "FiberIntegrator"

2. test_create_fiber_integrator_radians(poni_fixture):
   - Call with angle_unit="rad", incident_angle=0.005
   - Verify _gi_incident_angle ≈ 0.005

3. test_integrate_gi_1d(poni_fixture, synthetic_image):
   - fi = create_fiber_integrator(poni_fixture, incident_angle=0.2)
   - result = integrate_gi_1d(synthetic_image, fi, npt=500)
   - Verify IntegrationResult1D, radial.shape == (500,)

4. test_integrate_gi_2d(poni_fixture, synthetic_image):
   - fi = create_fiber_integrator(poni_fixture, incident_angle=0.2)
   - result = integrate_gi_2d(synthetic_image, fi, npt_rad=200, npt_azim=100)
   - Verify IntegrationResult2D
   - Verify intensity.shape == (200, 100)

5. test_integrate_gi_polar(poni_fixture, synthetic_image):
   - result = integrate_gi_polar(synthetic_image, fi, npt_rad=200, npt_azim=100)
   - Verify IntegrationResult2D with correct shapes

6. test_integrate_gi_exitangles(poni_fixture, synthetic_image):
   - result = integrate_gi_exitangles(synthetic_image, fi, npt_rad=200, npt_azim=100)
   - Verify IntegrationResult2D with correct shapes

7. test_integrate_gi_1d_angle_override(poni_fixture, synthetic_image):
   - Create fi with incident_angle=0.2
   - Call integrate_gi_1d with incident_angle=0.5 (override)
   - Verify _gi_incident_angle was updated to deg2rad(0.5)

8. test_integrate_gi_polar_1d(poni_fixture, synthetic_image):
   - fi = create_fiber_integrator(poni_fixture, incident_angle=0.2)
   - result = integrate_gi_polar_1d(synthetic_image, fi, npt=500)
   - Verify IntegrationResult1D, radial.shape == (500,)

9. test_integrate_gi_exitangles_1d(poni_fixture, synthetic_image):
   - result = integrate_gi_exitangles_1d(synthetic_image, fi, npt=500)
   - Verify IntegrationResult1D, radial.shape == (500,)
```

---

### Prompt 1H: `tests/test_batch.py`

```
Create tests/test_batch.py with pytest tests for
ssrl_xrd_tools.integrate.batch.

Use poni_fixture and ai_fixture from conftest.py.

Tests to write:

1. test_process_scan_directory(ai_fixture, tmp_path):
   - Create a temp directory with 3 small synthetic images saved as .tif
     files (use fabio.edfimage.EdfImage or just np.save — actually use
     fabio to write EDF files so io.image can read them):
       import fabio
       for i in range(3):
           img = np.random.poisson(100, (100, 100)).astype(np.float32)
           edf = fabio.edfimage.EdfImage(data=img)
           edf.write(str(tmp_path / "images" / f"frame_{i:04d}.edf"))
   - Create a generic AzimuthalIntegrator for 100x100 images:
       from pyFAI.detectors import Detector
       det = Detector(pixel1=75e-6, pixel2=75e-6, max_shape=(100, 100))
       ai = AzimuthalIntegrator(dist=0.2, poni1=0.00375, poni2=0.00375,
                                 wavelength=1e-10, detector=det)
   - Call process_scan(tmp_path / "images", ai, tmp_path / "out.h5", npt=100,
                        npt_rad=50, npt_azim=50)
   - Verify output file exists
   - Open with h5py, verify groups "0", "1", "2" exist
   - Verify each group has "q", "I", "IQChi", "Q", "Chi" datasets

2. test_process_scan_skip_existing(ai_fixture, tmp_path):
   - Run process_scan once, note output
   - Run again with reprocess=False
   - Verify frames were skipped (check log or just verify no error)
   - Run with reprocess=True, verify no error

3. test_process_series(tmp_path):
   - Create 2 scan directories with images
   - Call process_series([dir1, dir2], ai, output_dir)
   - Verify 2 output files created

4. test_directory_watcher_lifecycle(tmp_path):
   - Create a DirectoryWatcher with poll_interval=0.5
   - Start in background thread
   - Verify watcher.processed_files is a set
   - Call watcher.stop()
   - Join thread with timeout=5
   - Verify thread is dead

5. test_directory_watcher_detects_new_file(tmp_path):
   - Create watcher, start in background
   - Write a new .edf file to the watched directory
   - Wait up to 5 seconds, check if file appears in processed_files
   - Stop watcher
   - (This test may be flaky on slow CI — mark with @pytest.mark.slow)

NOTE: Create a small generic AI in each test that matches the 100x100 test
image size.  Don't use the full Eiger 4M fixtures here — batch tests should
be fast.
```

---

### Prompt 1I: `tests/test_export.py`

```
Create tests/test_export.py with pytest tests for ssrl_xrd_tools.io.export.

Tests to write:

1. test_write_xye(tmp_path):
   - write_xye(tmp_path / "test.xye", [1,2,3], [4,5,6])
   - Read back with np.loadtxt, verify 3 columns, 3 rows
   - Verify x and y values match

2. test_write_xye_with_variance(tmp_path):
   - write_xye(tmp_path / "test.xye", [1,2,3], [4,5,6], variance=[0.1, 0.2, 0.3])
   - Read back, verify third column matches variance

3. test_write_csv(tmp_path):
   - write_csv(tmp_path / "test.csv", [1,2,3], [4,5,6])
   - Read back with np.loadtxt(delimiter=","), verify values

4. test_write_h5(tmp_path):
   - write_h5(tmp_path / "test.h5", frame=0,
              q=np.linspace(0, 5, 100), intensity=np.random.rand(100),
              iqchi=np.random.rand(50, 60), q_2d=np.linspace(0, 5, 50),
              chi=np.linspace(-180, 180, 60))
   - Open with h5py, verify group "0" exists
   - Verify datasets q, I, IQChi, Q, Chi have correct shapes

5. test_write_h5_multiple_frames(tmp_path):
   - Write frames 0, 1, 2 to the same file
   - Verify all three groups exist

6. test_write_h5_overwrite_frame(tmp_path):
   - Write frame 0, then write frame 0 again with different data
   - Verify the data was overwritten (not duplicated)

7. test_write_xye_shape_mismatch(tmp_path):
   - write_xye with mismatched x and y shapes
   - Verify ValueError
```

---

## Phase 2: Corrections Module

These prompts implement the corrections/ subpackage.  Corrections are applied
to raw detector images BEFORE integration.  They can also be passed through
to pyFAI via the polarization_factor / flat / dark parameters if preferred.

The design principle: every function takes an image array and returns a
corrected image array.  Pure functions, no state.

---

### Prompt 2A: `corrections/detector.py`

```
Implement corrections/detector.py.  These correct detector-specific artifacts
on raw images before integration.

Functions to implement:

1. subtract_dark(
       image: np.ndarray,
       dark: np.ndarray,
   ) -> np.ndarray
   - Subtract dark current image: result = image - dark
   - Preserve NaN masking: where image is NaN, result is NaN
   - Return float64 array

2. apply_flatfield(
       image: np.ndarray,
       flat: np.ndarray,
       min_flat: float = 0.1,
   ) -> np.ndarray
   - Divide by flat-field: result = image / flat
   - Where flat < min_flat, set result to NaN (avoid division by ~zero)
   - Preserve existing NaN pixels

3. apply_threshold(
       image: np.ndarray,
       threshold: float,
       low: float | None = None,
   ) -> np.ndarray
   - Set pixels above threshold to NaN
   - If low is provided, also set pixels below low to NaN
   - Return a COPY (don't modify in-place)

4. apply_mask(
       image: np.ndarray,
       mask: np.ndarray,
   ) -> np.ndarray
   - Set pixels where mask is True to NaN
   - mask is boolean, same shape as image
   - Return a copy

5. combine_masks(
       *masks: np.ndarray | None,
   ) -> np.ndarray | None
   - OR together all non-None boolean masks
   - Return None if all inputs are None
   - Raise ValueError if shapes don't match

6. correct_image(
       image: np.ndarray,
       dark: np.ndarray | None = None,
       flat: np.ndarray | None = None,
       mask: np.ndarray | None = None,
       threshold: float | None = None,
       low_threshold: float | None = None,
   ) -> np.ndarray
   - Convenience pipeline: applies dark, flat, mask, threshold in order
   - Each step is skipped if its parameter is None
   - Returns corrected image

All functions: accept float or int arrays, return float64.
Follow conventions in CLAUDE.md.  Export all in __all__.
```

---

### Prompt 2B: `corrections/beam.py`

```
Implement corrections/beam.py.  These correct for beam and geometry effects.

Functions to implement:

1. polarization_correction(
       image: np.ndarray,
       ai: "AzimuthalIntegrator",
       polarization_factor: float = 0.99,
   ) -> np.ndarray
   - Compute the pyFAI polarization correction array:
     pol = ai.polarization(shape=image.shape, factor=polarization_factor)
   - result = image / pol
   - Where pol < 1e-10, set result to NaN
   - Return corrected image
   - NOTE: This is an ALTERNATIVE to passing polarization_factor to
     integrate_1d.  Use one or the other, not both.

2. solid_angle_correction(
       image: np.ndarray,
       ai: "AzimuthalIntegrator",
   ) -> np.ndarray
   - Compute solid angle array: sa = ai.solidAngleArray(shape=image.shape)
   - result = image / sa
   - Where sa < 1e-10, set result to NaN
   - NOTE: pyFAI already applies solid angle correction by default in
     integrate1d/integrate2d (correctSolidAngle=True).  This function is
     for cases where you need the correction applied pre-integration, e.g.,
     for direct image analysis or RSM processing.

3. absorption_correction(
       image: np.ndarray,
       mu_t: float,
       ai: "AzimuthalIntegrator",
   ) -> np.ndarray
   - For each pixel, compute the path length through the sample based on
     the scattering angle: tth = ai.twoThetaArray(shape=image.shape)
   - Absorption factor: abs_factor = 1 / (1 - exp(-mu_t / cos(tth)))
     (simplified for transmission geometry)
   - result = image * abs_factor
   - This is a STUB-like implementation — the formula depends on geometry
     (transmission vs reflection).  Document that in the docstring and note
     that users should verify the formula for their specific geometry.

Type hint ai using TYPE_CHECKING guard.
Follow conventions in CLAUDE.md.  Export all in __all__.
```

---

### Prompt 2C: `corrections/normalization.py`

```
Implement corrections/normalization.py.  These handle monitor normalization
and intensity scaling.

Functions to implement:

1. normalize_monitor(
       image: np.ndarray,
       monitor: float,
       reference: float = 1.0,
   ) -> np.ndarray
   - Scale image by reference / monitor
   - If monitor <= 0, log warning and return image unchanged
   - Common pattern: image was measured with monitor count i1; normalize
     to a standard monitor value (e.g., reference=100000)

2. normalize_time(
       image: np.ndarray,
       exposure_time: float,
       reference_time: float = 1.0,
   ) -> np.ndarray
   - Scale image by reference_time / exposure_time
   - Used to normalize images with different exposure times

3. normalize_stack(
       images: np.ndarray | list[np.ndarray],
       monitors: np.ndarray | Sequence[float],
       reference: float | None = None,
   ) -> list[np.ndarray]
   - Normalize each image in a stack by its corresponding monitor value
   - If reference is None, use the mean of monitors as reference
   - Returns list of normalized images
   - Useful before MultiGeometry stitching where each image has different
     i1 monitor counts

4. scale_to_range(
       intensity: np.ndarray,
       vmin: float = 0.0,
       vmax: float = 1.0,
   ) -> np.ndarray
   - Linearly scale intensity to [vmin, vmax] range
   - Ignoring NaN values for min/max determination
   - Useful for visualization

Follow conventions in CLAUDE.md.  Export all in __all__.
```

---

### Prompt 2D: `corrections/__init__.py`

```
Update corrections/__init__.py to export key functions from all submodules.

Export from detector: subtract_dark, apply_flatfield, apply_threshold,
    apply_mask, combine_masks, correct_image
Export from beam: polarization_correction, solid_angle_correction,
    absorption_correction
Export from normalization: normalize_monitor, normalize_time,
    normalize_stack, scale_to_range

Include __all__.  Follow conventions in CLAUDE.md.
```

---

### Prompt 2E: `tests/test_corrections.py`

```
Create tests/test_corrections.py with pytest tests for the corrections/ module.

Tests to write:

1. test_subtract_dark():
   - image = np.full((10, 10), 100.0); dark = np.full((10, 10), 10.0)
   - result = subtract_dark(image, dark)
   - Verify all values ≈ 90.0
   - Add NaN to image[0, 0], verify result[0, 0] is NaN

2. test_apply_flatfield():
   - image = np.full((10, 10), 100.0); flat = np.full((10, 10), 2.0)
   - result = apply_flatfield(image, flat)
   - Verify all values ≈ 50.0
   - Set flat[5, 5] = 0.01 (below min_flat=0.1), verify result[5, 5] is NaN

3. test_apply_threshold():
   - image with values [0, 50, 100, 200]
   - apply_threshold(image, threshold=150)
   - Verify 200 → NaN, others unchanged
   - With low=25: verify 0 → NaN too

4. test_apply_mask():
   - image 10x10 filled with 1.0
   - mask: True at (0,0) and (5,5)
   - Verify result has NaN at those positions

5. test_combine_masks():
   - mask1: True at (0,0); mask2: True at (1,1)
   - combined = combine_masks(mask1, mask2)
   - Verify True at both positions
   - combine_masks(None, None) should return None
   - combine_masks(mask1, None) should return mask1

6. test_correct_image_pipeline():
   - Create image, dark, flat, mask, threshold
   - Call correct_image with all parameters
   - Verify result has dark subtracted, flat divided, masked, thresholded

7. test_polarization_correction():
   - Create a small AI (generic detector 100x100)
   - image = np.ones((100, 100))
   - result = polarization_correction(image, ai, polarization_factor=0.99)
   - Verify result is not all ones (correction was applied)
   - Verify no NaN in center region

8. test_solid_angle_correction():
   - Same small AI
   - Verify correction changes values (corners differ from center)

9. test_normalize_monitor():
   - image = np.ones((10, 10)) * 1000
   - normalize_monitor(image, monitor=50000, reference=100000)
   - Verify values ≈ 2000 (scaled by 100000/50000)

10. test_normalize_monitor_zero():
    - monitor=0 should return image unchanged (with warning)

11. test_normalize_stack():
    - 3 images, monitors=[100, 200, 50]
    - Verify each image is scaled appropriately

12. test_scale_to_range():
    - intensity from 10 to 100
    - scale_to_range → [0, 1]
    - Verify min ≈ 0, max ≈ 1
```

---

### Prompt 2F: Update CLAUDE.md

```
Update CLAUDE.md to reflect the new state:

1. In the module map, change corrections/ status markers from STUBS to ✅
2. Add corrections functions to the "What's Implemented" list
3. Note that polarization_factor and normalization_factor are available as
   explicit parameters on integrate_1d/integrate_2d, AND as standalone
   correction functions in corrections/ (use one approach, not both)
4. Note that integrate/gid.py now also has integrate_gi_polar_1d and
   integrate_gi_exitangles_1d for 1D line-cut extraction
5. In the dependencies table or optional deps, note watchdog is optional
   for DirectoryWatcher

Also update the "What's Implemented" section ordering to be alphabetical
by module path.
```

---

## Order of execution

Phase 0 (API expansion — do first so tests cover final API):
1. Prompt 0A: Add polarization_factor / normalization_factor to single.py
2. Prompt 0B: Add 1D GI line-cut wrappers to gid.py

Phase 1 (Tests):
3. Prompt 1A: tests/conftest.py
4. Prompt 1B: tests/test_core.py
5. Prompt 1C: tests/test_transforms.py
6. Prompt 1D: tests/test_calibration.py
7. Prompt 1E: tests/test_single.py
8. Prompt 1F: tests/test_multi.py
9. Prompt 1G: tests/test_gid.py
10. Prompt 1H: tests/test_batch.py
11. Prompt 1I: tests/test_export.py

Phase 2 (Corrections):
12. Prompt 2A: corrections/detector.py
13. Prompt 2B: corrections/beam.py
14. Prompt 2C: corrections/normalization.py
15. Prompt 2D: corrections/__init__.py
16. Prompt 2E: tests/test_corrections.py
17. Prompt 2F: Update CLAUDE.md

After all prompts: run `pytest tests/ -v` and fix any failures.
