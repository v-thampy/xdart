# Direct-beam distance calibration

`xrd_tools.integrate.calibrate_direct_beam` accepts degree angles and an ordered
iterable of arrays or image references. `calibrate_direct_beam_scan` reads one
SPEC scan and calls the same numerical owner. Both return `DirectBeamResult`;
neither writes files, plots, changes application state or constructs a PONI.

## Geometry and coordinates

The default `model="tangent"` requires an **explicit known** `angle_zero_deg`:

```
centre_px = intercept_px + signed_scale_px * tan(radians(angle_deg - angle_zero_deg))
distance_m = abs(signed_scale_px) * pixel_pitch_m
```

The detector is a uniform flat plane rotating on a rigid arm about the sample.
At the supplied zero, its normal is parallel to the incoming beam. The fitted
positive distance is the perpendicular sample-to-plane distance, equal to the
beam-path distance at that zero. It is not the beam-path length at every scan
angle. Mapping this number into later geometry requires those same assumptions;
no detector tilt, angular zero, wavelength or full PONI is inferred. Only the
intercept and signed scale are fitted. Curved, distorted and nonuniform pixels
are explicitly unsupported. Multiple axes or motion with a varying arm radius
are outside this model.

Public beam coordinates are zero-based `(x,y) = (column,row)` pixel centres;
arrays have `(row,column)` shape. **`beam_pixel` is an approximate strip locator,
not a known reference-angle constraint.** A horizontal scan sums rows around its
y coordinate; vertical sums columns around its x coordinate. The motion-axis
coordinate only checks the seed lies inside the selected image/ROI. The Gaussian
plus linear-background fit locates the strongest peak in that strip. Choose an
ROI if another bright feature is present. `strip_half_width=2` gives up to five
rows/columns, clipped to the ROI/image without wrapping at an edge.

Use the existing `ImageOrientation` (CCW rotation, transpose and flips) to
transform raw frames and the detector's native mask once. Supply `beam_pixel`,
user `mask`, ROI and `motion` **in the resulting oriented frame**. ROI is
`(x0,y0,x1,y1)`, with exclusive upper bounds. Pixel pitches swap on an axis swap;
horizontal uses the column pitch, vertical the row pitch. The pyFAI detector's
`orientation` config is its physical-coordinate convention, not another array
transform. See [pyFAI detector conventions](https://pyfai.readthedocs.io/en/stable/api/detectors/).

## Explicit arrays

```python
from xrd_tools.integrate import calibrate_direct_beam

result = calibrate_direct_beam(
    angles_deg, images,  # same order and length; None keeps a missing frame's row
    detector="Detector",
    detector_config={"pixel1": 120e-6, "pixel2": 80e-6, "max_shape": (45, 161)},
    beam_pixel=(80, 22), motion="horizontal", angle_zero_deg=7.0,
    frame_ids=frame_names,  # optional; arrays default to original row numbers
)
print(result.distance_mm, result.distance_std_m * 1000)
print(result.centres_px, result.residuals_px, result.rejection_reasons)
```

Alternatively pass a registered detector name such as `"Pilatus100k"` or a
configured pyFAI detector. Detector objects are copied. Effective full-panel
integer binning is resolved from the first readable image's dimensions;
subsequent frames must have that shape. A smaller cropped image must not be
presented as a binned full panel. For an already-binned generic detector, supply
its effective pitches and actual shape, or configure its `.binning` explicitly.
When an existing detector mask cannot match the resolved shape, the scan fails
rather than silently losing that mask.

## SPEC and file references

```python
from pathlib import Path
from xrd_tools.integrate import calibrate_direct_beam_scan

root = Path("/path/to/data")
result = calibrate_direct_beam_scan(
    root / "direct_beam", "1.1",  # scan NUMBER.repetition, never a list index
    image_dir=root / "images",
    filename_template="b_thampy_direct_beam_scan{scan}_{point:04d}.raw",
    point_index_origin=0, motor="del",
    detector="Pilatus100k", beam_pixel=(243, 99), angle_zero_deg=0.0,
    reader_options={"detector_shape": (195, 487), "raw_dtype": "<i4",
                    "raw_header_skip": 0},
)
```

`scan=1` selects `"1.1"`; repeated scans use an explicit key such as `"23.2"`.
The motor must be a per-point SPEC column, not a constant starting motor.
A template can use `{point}`, `{scan}` and `{repetition}`. There is no filename
discovery. Alternatively pass `images=ordered_references` to the SPEC adapter.

References can be paths, `(path, frame)` or `(path, frame, dataset_path)`;
for example `(Path("beam.h5"), 17, "/entry/data/data")`. Exact reader frame
selection prevents fallback to another frame. `reader_options` allows `frame`,
`dataset_path`, `detector_shape`, `raw_dtype`, `raw_header_skip`. References
override shared frame/dataset options. TIFF/EDF/CBF/NPY/HDF5/NeXus/raw handling
belongs to `read_image`; do not pass pyFAI objects as its `detector` argument.
Masks, rotation and thresholds belong to calibration's explicit arguments.
Only one frame is processed at a time; results retain scalar summaries, not images.

## Rejections, fit quality and uncertainty

`angles_deg`, `frame_ids`, `point_ids`, `centres_px`, `centre_std_px`, `used`,
`rejection_reasons`, `predictions_px` and `residuals_px` all preserve original row
order. The SPEC adapter's point IDs are `(scan_key, original_point_index)`.
Rejected centres/residuals are NaN; predictions remain available at those angles.
`np.flatnonzero(result.used)` and `np.flatnonzero(~result.used)` give original
used/rejected row indices. No skipped image renumbers the angles. Pixel zero is valid.

Static/user masks, detector dummy values, negative and nonfinite pixels exclude
whole profile bins rather than reducing their summed intensity. A missing bin
within two fitted Gaussian sigmas rejects that frame. Supply `saturation` as an
inclusive acquisition threshold; saturated pixels anywhere in the selected strip
reject the frame. Integer dtype maxima are also rejected. Clipping below those
limits, count-rate nonlinearity and detector-specific corrections require caller
knowledge. This routine does not infer that an apparently smooth peak is unsaturated.
Fewer than three usable rows/distinct angles, unresolved fits or insignificant
motion raise `ValueError` with reasons. Gaussian covariance warnings reject only
that frame; unrelated warnings and programming errors are not suppressed.

`distance_std_m` is one standard error of the unweighted two-parameter regression,
conditional on exact angles and pixel pitch. `centre_std_px` is the Gaussian-fit
centre error. These are fit uncertainties, **not total calibration accuracy**;
unknown tilt/zero/pitch/motor/saturation errors are not included. Inspect residuals.
`distance_mm == 1000 * distance_m`; `motion_sign` and `slope_deg_per_pixel` retain
motion direction. The latter is the tangent model's local angle/position slope at
zero. `intercept_px` is the predicted centre at the supplied zero (tangent), or at
motor angle zero (linear).

`model="linear"` preserves the notebook regression `angle = slope*pixel + intercept`
and its distance `pitch / tan(abs(slope)*pi/180)`. It does not require known angular
zero, but is only a local small-angle estimate near normal incidence. With noisy
centres, inverse regression also differs from fitting positions against angles.
The tangent model is exact only under the stated planar geometry; it is not an
unconstrained tilt/zero fit. The related [xrayutilities tangent model](https://xrayutilities.sourceforge.io/_modules/xrayutilities/analysis/sample_align.html)
is a useful reference; this implementation needs only existing NumPy/SciPy/pyFAI.

Run the [standalone usage notebook](../../examples/direct_beam_distance_calibration.ipynb)
for synthetic arrays, the SPEC convenience call and marked diagnostic plots.
