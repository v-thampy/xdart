"""Finite offline scientific instructions; no GUI or scientific imports.

Topic IDs are presentation identifiers, independent of settings or readiness.
Diagnostic prose supplements the original strings; it never classifies a result.
"""
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True, slots=True)
class HelpTopic:
    identifier: str
    title: str
    text: str
    figures: tuple[str, ...] = ()
    run_key: str | None = None
    guide: str | None = None


#: The multi-step calibration's seven steps, declared once as (label, text), each text starting with its label:
#: Calibrate's checklist lists them and the "calibration.multi-step" topic numbers them under Steps.
CALIBRATION_STEPS = tuple((label, label + rest) for label, rest in (
    ("Describe the instrument first", ": geometry, motor names, signed axes and zeros, detector, camera mounting "
     "and first-pixel corner."),
    ("Start from a rough geometry", ": direct-beam scan, typed distance, or Skip. The guess predicts rings; it does "
     "not replace ring calibration."),
    ("Read calibrant-scan metadata, inspect the suggested frames and prepare the selected images",
     ". Prefer varied arm angles and useful ring coverage."),
    ("Calibrate each selected image in pyFAI, saving its PONI and control points",
     ". Import existing files only when they match that exact frame and image."),
    ("Fit at least three Complete frames together", ". Shared detector geometry is fitted across the arm positions; "
     "enabled free parameters move within their bounds, while fixed parameters stay fixed. Read model-selection "
     "reasons."),
    ("Inspect the summary and Show the fit", ". Refine with further scan observations when appropriate; check "
     "residuals and the held-out comparison rather than accepting more points merely because there are more of "
     "them."),
    ("Validate the informative frames, then Save geometry and use it in Stitch or RSM", ". Read the existing "
     "overwrite confirmation before replacing a file."),
))


def _numbered(steps):
    """A topic's numbered list, wrapped as the hand-written topics are."""
    import textwrap
    return "\n".join(textwrap.fill(text, 79, initial_indent=f"{number}. ", subsequent_indent="   ",
                                    break_on_hyphens=False, break_long_words=False)
                      for number, (_label, text) in enumerate(steps, 1))


TOPICS = (
    HelpTopic("calibration.multi-step", "Multi-step detector calibration", """Purpose
Build one detector geometry from powder-ring control points across a calibrant
scan, then save and use it in Stitch or RSM. Int's Calibrate action instead opens
the external single-image pyFAI calibration tool.

Required inputs
Detector and instrument settings; calibrant images with detector-arm motor
angles; the matching calibrant; recorded energy/wavelength or an Energy override;
matching PONI and control-point files for each calibrated frame.

Steps
""" + _numbered(CALIBRATION_STEPS) + """

Units and conventions
Motor angles are degrees; displayed sample–detector distance is millimetres;
Energy is keV; wavelength in geometry records is metres. Ring residuals are
reciprocal-space magnitude errors in Å⁻¹. Keep detector configuration and image
orientation consistent across frames. A single arm holds the other detector
circle at zero; two moving arms require both matching motor columns.

Assumptions
Frames share a detector configuration, image shape, calibrant and wavelength.
The global model describes the selected arm. A successful fit is not an
independent validation of sample orientation or azimuthal direction.

Expected observations
Rows become Complete after matched PONI and control points are available. The
fit summary reports residuals, conditioning, parameters and bounds. Saving
selects the geometry for the calling page. A refused xrayutilities partner is
reported separately; read the message rather than assuming both assets exist.

Diagnostics
Missing energy, mismatched files, insufficient Complete frames, convergence,
physical plausibility, poor conditioning and excessive RMS can prevent fitting.
Use Calibration diagnostics for interpreting the actual summary and errors.

Common mistakes and corrective actions
Do not pair files from neighbouring frames. Reprepare/refit after changes to
source, instrument, camera or fit settings. Nearly identical frames may add
little information; seek broader angles and rings. Never compensate a pixel
flip by changing a physical motor sign.

Worked example
Conceptual workflow: a prepared frame reads Needs PONI + points. Its own matched
files make it Complete. Three to five Complete frames spanning useful angles
supply a global fit. Inspect its summary and an informative frame before saving;
this example supplies no measured geometry or guaranteed fit outcome.

References
xdart/gui/tabs/calibration/dialog.py — guided controls, fit summary and save.
xrd_tools/integrate/stitch_calibration.py — frame matching and global fit.
xrd_tools/integrate/stitch_autocalibration.py — refinement and validation.
"""),
    HelpTopic("calibration.frame-choice", "Choosing calibrant frames", """Purpose
Choose observations that constrain the detector geometry through angular and
ring coverage, rather than through frame count alone.

Required inputs
A readable calibrant scan, its motor-angle metadata, accessible raw images,
correct calibrant and energy, and matching per-image calibration files.

Steps
1. Read metadata and inspect the suggested frames. Three to five is a starting
   suggestion, not proof of sufficiency.
2. Inspect Delta and Nu values and choose coverage across the fitted motors.
3. Prepare the selected images and inspect whether usable rings are present.
4. Calibrate each image or import its matched PONI and control points.
5. Fit the Complete frames; use conditioning, correlations and residuals to
   decide whether more angular or ring coverage is needed.

Units and conventions
Delta and Nu are degrees. Frame identity follows the source records, not an
assumed consecutive filename sequence. Control points belong to their matching
image and must lie within its shape.

Assumptions
Three Complete frames are the minimum for the global fit. Metadata readiness,
completed preparation and scientific identifiability are separate questions.
An extended mount/sample-offset model needs appropriate Nu coverage; single-arm
scans use the simpler model. Read the model's stated coverage requirements.

Expected observations
Suggestions spread across normalized angular coverage. Complete means the
matched files are present, not that the geometry is uniquely determined.
Automatic refinement reports its selected/available rows and exclusions.

Diagnostics
Poor conditioning calls for observations spanning the fitted motors. Large or
patterned per-frame/ring residuals call for checking matching, angles and ring
assignment. Additional nearly identical frames do not establish identifiability.

Common mistakes and corrective actions
Do not select only neighbouring images because they look clean. Compare their
angles and include wider useful coverage. Do not borrow another frame's PONI or
control-point file merely because its rings look similar.

Worked example
Conceptual case: several Complete frames occupy one narrow angular region. The
fit may still be poorly conditioned. Frame completion answers whether files are
available; angle coverage answers whether they constrain the model.

References
xrd_tools/integrate/stitch_calibration.py — suggestions, matching and requirements.
xrd_tools/integrate/stitch_autocalibration.py — automatic coverage validation.
xdart/gui/tabs/calibration/dialog.py — Frames and fit controls.
"""),
    HelpTopic("geometry.orientation", "Geometry and detector orientation", """Purpose
Keep physical instrument rotations, detector-camera mounting, array transforms
and the first-pixel corner consistent with the geometry used for reduction.

Required inputs
Actual circle/motor names, signed rotation axes and known zero offsets; detector
dimensions and mounting; matching calibration; an explicitly known beam pixel
when a rough/direct-beam method requires it.

Steps
1. Describe the physical instrument. For fourc/twoc, vertical 2θ plays psic del;
   horizontal 2θ plays nu. Check the actual motor column names and angle zeros.
2. Set the camera mounting rotation and flips from the real detector mounting.
   Reset returns the upright setting; it cannot discover an unknown mounting.
3. Keep array coordinates (row, column) distinct from public image coordinates
   (x, y): x is column and y is row, with zero-based pixel indices.
4. Apply ImageOrientation once on the scientific path. Its quarter-turns or
   transpose swap dimensions; do not rotate both the data and its interpretation
   a second time to make a display look familiar.
5. Set the first-pixel corner separately. pyFAI's corners 1/2/3/4 are top-left,
   top-right, bottom-right, bottom-left, viewed from behind towards the sample.
   Use the corner recorded for the calibration and detector configuration.
6. Check beam position, image/mask shape, rings and azimuthal direction on an
   informative frame before saving or reducing other arm positions.

Units and conventions
Angles/zeros are degrees: the physical angle is motor readback minus its zero.
Positive rotation is right-handed about the signed axis. The psic lab frame is
x up, y along the beam, z left looking downstream. Raw rows/columns follow the
camera's signed x/z axes; pyFAI's native camera is rows x+, columns z−.
ImageOrientation rotation is counterclockwise in array space; the mounting
editor describes physical mounting. These are related settings, not synonyms.
Detector-frame q/χ and sample/beam coordinates are different. Backend χ
conventions require matching camera, first-pixel corner and geometry; an
apparently similar radial profile does not establish the same azimuth.

Assumptions
The saved calibration describes the same mounted detector and transformed image
frame. Powder rings constrain radial geometry but do not determine sample
orientation or the sign of every azimuthal coordinate.

Expected observations
Array shape, beam pixel, static mask and predicted rings align consistently at
different arm positions. A quarter-turn of a non-square image changes its shape.

Diagnostics
Displaced/mirrored rings or a wrong azimuth can indicate a camera/corner mismatch,
wrong motor sign/name or wrong zero. Read the exact refusal before changing fields.

Common mistakes and corrective actions
Do not fix a pixel flip with a motor sign. Restore the calibration's camera or
recalibrate with an explicitly chosen mounting. Fix malformed angle zeros rather
than clearing them to hide the error. Do not use radial agreement to certify χ.

Worked example
Conceptual inputs: readback 12° and zero 2° give physical angle 10°. Independently,
a 180° array rotation changes [[1,2,3],[4,5,6]] to [[6,5,4],[3,2,1]]. The angle
offset and pixel transform solve different parts of the geometry.

References
xrd_tools/core/geometry/camera.py — raw camera and pyFAI frame mapping.
xrd_tools/core/geometry/diffractometer.py — ImageOrientation and physical geometry.
xrd_tools/session/instrument.py — circles, motor names and signed axes.
"""),
    HelpTopic("calibration.diagnostics", "Interpreting calibration diagnostics", """Purpose
Understand why a Stitch/RSM calibration is usable, refused or insufficiently
constrained; preserve the actual summary and uncoded messages as the evidence.

Required inputs
A fit attempt and its calibrated frames, or a whole-scan refinement result.

Steps
1. Read the fit message, RMS |q|, control-point count and Jacobian condition.
2. Compare fitted parameters with starting values and allowed bounds.
3. Open Show the fit: inspect residuals against motor angle and expected rings,
   including an informative frame/arm/ring rather than only the pooled RMS.
4. Inspect uncertainty, strong parameter correlation, model-selection reasons,
   rejected points and excluded frames. Read the held-out comparison if present.
5. Correct mismatched inputs or improve coverage before another Fit/Refine.

Units and conventions
RMS and ring residuals measure reciprocal-space magnitude errors in Å⁻¹;
residual plots display 10⁻³ Å⁻¹. Condition number is dimensionless. Parameter
uncertainty uses the parameter's displayed units.

Assumptions
Small RMS alone does not establish a usable geometry. Convergence, physical
plausibility and conditioning are also checked. Uncertainty belongs to the
selected observations and fitted model; readiness/success is not scientific
quality or independent validation of sample/azimuth orientation.

Expected observations
The summary identifies parameters at bounds and strong correlations. Refinement
explains its model choice and held-out comparison. The fit view marks rejected
control points. Original messages and summary/provenance remain readable.

Diagnostics
Unconverged, nonphysical, poorly conditioned or excessive-RMS fits are refused.
Source or calibration-file changes can invalidate prepared inputs. A refused
xrayutilities transfer is reported separately from a saved pyFAI geometry.
No diagnostic code is inferred from the words in an uncoded failure message.

Common mistakes and corrective actions
Do not interpret low RMS as proof that all parameters are determined. Do not
repeatedly widen bounds without checking angles, detector settings and matched
files. Add useful coverage when conditioning/correlation exposes ambiguity.

Worked example
Conceptual case: small ring residuals coexist with two strongly correlated
parameters. The summary exposes uncertainty that a visually close ring fit can
conceal. No numerical fit result is claimed by this example.

References
xdart/gui/tabs/calibration/dialog.py — summary, bounds and literal messages.
xdart/gui/tabs/calibration/fit_windows.py — residual display.
xrd_tools/integrate/refine.py — fit success requirements and refusals.
"""),
    HelpTopic("corrections.existing", "Existing intensity and detector corrections", """Purpose
Apply the admitted intensity corrections and normalization with explicit inputs.
Keep sensor parallax, a position model, separate from intensity scaling.

Required inputs
Detector/wavelength and calibration; detector-shaped 2D dark/flat images when
selected; normalization metadata; recorded sensor material/thickness for an
enabled, supported PONI v3 parallax path.

Steps
1. Choose appropriate dark and flat files and inspect their file-check messages.
   Dark is subtracted as given; flat divides as given and is not rescaled.
2. Choose the flat cutoff. The default masks values below 10% of the positive,
   finite flat median; zero cutoff still masks nonpositive/nonfinite flat pixels.
3. Set solid angle and polarization where exposed for the selected backend.
   A polarization factor is dimensionless and ranges from −1 to 1.
4. Select the beam divisor that matches acquisition. A monitor after foils
   already includes attenuation. A monitor before foils uses monitor × foil
   transmission; counting time without a monitor uses exposure × transmission.
   Never divide by both exposure and a monitor that already counts over it.
5. Keep the selected PONI sensor authoritative unless the exposed override is
   explicitly enabled. Check material, thickness and positive wavelength before
   enabling a supported parallax path. Read backend refusals; support is not universal.
6. Inspect corrected profiles and masks against the acquisition assumptions.

Units and conventions
Exposure is seconds; monitor values are counts. Signal is divided by the finite,
positive per-frame divisor. Foil transmission depends on energy and absorption
lengths. Sensor thickness is metres: 450 µm = 0.00045 m. Dark/flat ordering is
subtraction then flat-field division, with geometric corrections and the
normalization denominator handled by the selected reduction owner.

Assumptions
Detector geometric masks (gaps/nonpixels) are independent of opt-in value or
saturation masks. Choosing corrections does not prove their scientific validity.
Sample optical and refraction controls are unavailable in these GUI settings.
Existing headless GI correction paths require their own scientific qualification.

Expected observations
Dark shifts intensity; flat, solid angle, polarization and normalization scale
intensity. A flat median far from one changes the overall scale. Parallax models
positions, rather than correcting an intensity denominator or camera mismatch.

Diagnostics
Not usable file checks, shape mismatch and nonfinite/nonpositive divisors require
matching input files/metadata before recomputing. Read the original backend
refusal for unsupported sensor/parallax settings.

Common mistakes and corrective actions
Avoid double exposure normalization. Do not treat counts as seconds. Do not
assume a flat is normalized automatically. Correct positional misalignment with
the geometry/camera model rather than by changing intensity corrections.

Worked example
Conceptual intensity arithmetic: raw 100 minus dark 10 gives 90; dividing by flat
2 gives 45; dividing by beam divisor 5 gives 9 before solid angle/polarization.
These explicit inputs illustrate ordering, not a calibration or validation result.

References
xrd_tools/corrections/detector.py — dark/flat preparation.
xrd_tools/corrections/normalization.py — beam divisors and foil transmission.
xrd_tools/integrate/calibration.py — admitted PONI sensor/parallax settings.
"""),
    HelpTopic("analysis.plot-roi", "Combined Plot ROI and scan statistics", """Purpose
Plot rectangular raw-image statistics against scan metadata and inspect the exact
corresponding frame in the combined image/statistics window.

Required inputs
A scan with accessible full raw frames, numeric metadata for plot axes, and one
to five rectangular signal ROIs with independent optional background definitions.

Steps
1. Open the scan plot and choose X. Open Plot ROI and position the rectangles.
2. For each signal choose mean, sum, max, min or std. For mean/sum, optionally
   add its own background rectangle and choose subtract or divide.
3. Choose the existing saturation-mask policy, then explicitly Compute/Update.
   Edits latch the retained result as stale until a new computation is adopted.
4. Select ROI columns on the left or right axis; optionally divide Y by a metadata
   column. Left Log affects the left plot axis; image Log affects the image only.
5. Click an unambiguous plotted point to inspect its raw frame. Repeated X values
   retain original row identities; ambiguous points require an explicit choice.
6. Cancel to stop current computation; Close retires this tool and its context.

Units and conventions
Arrays use (row, column); displayed image X is column and Y is row. ROI values
follow raw signal units; background division is a ratio. Mean uses background
density directly; Sum scales density by the signal ROI's valid-pixel count.
Signal/background rectangles may have different support and valid counts.

Assumptions
ROI statistics use raw pixels, not inherited dark/flat/solid angle/polarization
or normalization corrections from reduction. This combined path does not supply
the reduction's static mask. Nonfinite pixels are excluded; recorded detector
saturation masking is optional. Preview admission charges the whole frame;
choosing a smaller rectangle does not reduce the existing 64 MiB preview cap.

Expected observations
Each signal contributes a series with valid-pixel counts retained in its result.
Missing raw frames produce unavailable values, not covered zeros. Cursor movement
does not recompute statistics. Stale definitions retain the old series and its
warnings, labelled as belonging to the retained result.

Diagnostics
ROI_IDENTITY_MISMATCH requires matching source/table/labels/series identities.
Unavailable/non-2D raw data or preview/result caps can stop calculation. Zero or
nonfinite plot normalization gives unavailable points; Log excludes nonpositive
left-axis values. The help action's ! cue exposes literal result/refusal codes
and details even after the status text has been redrawn.

Common mistakes and corrective actions
Background is supported only for mean/sum. With no usable background pixels,
subtraction is a no-op and division unavailable; zero background density also
makes division unavailable. Choose suitable support and inspect valid counts.
Do not confuse selecting a new frame with recomputing a changed ROI definition.
Check raw access for missing points; do not shrink the rectangle to evade a
whole-frame preview limit.

Worked example
Conceptual Sum subtraction: signal valid count 4, raw sum 40; background valid
count 2, raw sum 6. Background density is 3, so adjusted sum is 40 − 3 × 4 = 28.
No equal-area assumption is needed. This example does not execute a scan.

References
xdart/gui/tabs/scattering/roi_selection_view.py — shared ROI controls.
xrd_tools/core/roi.py — valid support and reducer behavior.
xrd_tools/analysis/plans.py — raw scan reduction and background handling.
xrd_tools/analysis/scan_operations.py — identity, normalization and admission.
"""),
    HelpTopic("phase-analysis.phases", "Phase Analysis: the phase list", """Purpose
One list of phases serves both the reference overlays on the plots and the fit. A
checked (include) phase is drawn as a reference and is fitted; an unchecked one is
neither. Built-in calibrants are reference lines only: they are drawn but never fitted.

Required inputs
CIF files, lattice parameters (cell lengths, angles and space group), a built-in
calibrant, or an entry already kept in the phase library. Fitting also needs one
displayed radial trace in q (Int) or Q/2θ with a recorded wavelength (Stitch).

Steps
1. Add CIF… parses the files on a worker and keeps their bytes in the phase library.
2. Add lattice… predicts positions only; Add calibrant… and Add from library… list
   built-in calibrants and kept entries. Remove takes only the selected row.
3. Check a row to include it: it is drawn and, unless it is a calibrant, fitted.
4. Same minimum for all phases applies one minimum intensity to every CIF phase; turn
   it off to give each CIF row its own minimum.

Units and conventions
Minimum intensity is a percent of a phase's strongest reachable reflection. Each phase
keeps the colour slot it joined with, and its legend entry; clicking a legend entry
flips the include check. The top-bar Phases toggle hides the overlays only.

Assumptions
At most eight phases. Lattice and calibrant phases carry no structural intensities, so
no minimum applies to them and their positions are all listed.

Expected observations
The list is saved with the session's plot options; closing and reopening the window
lists the same rows. After a fit, a CIF row's tooltip shows the receipt of the bytes
that were fitted.

Diagnostics
The rings notice under the list says why detector rings are unavailable. A missing
library entry is named in the status line. A list holding only calibrants refuses Fit.

Common mistakes and corrective actions
An unchecked phase is not fitted: check it. A calibrant is never fitted: to fit an
internal standard, add its CIF. A legend click also changes what the next Fit uses.

Worked example
List LaB6 (calibrant), HfO2 (CIF) and a TiN lattice; uncheck TiN. Fit uses HfO2 only,
LaB6 stays a reference line, and TiN is neither drawn nor fitted.

References
xdart/gui/analysis/phase_source.py — the shared phase list.
xrd_tools/analysis/phase_references.py — the phase library and calibrant entries.
"""),
    HelpTopic("phase-analysis.peak-shape", "Phase Analysis: peak shape", """Purpose
Choose the profile every reflection is fitted with, how its width varies with q, and
how relative intensities within a phase may depart from the structure (texture).

Required inputs
At least one included CIF or lattice phase and a displayed radial trace.

Steps
1. Profile: pseudo-Voigt, Gaussian, Lorentzian, Voigt, Lorentzian², Pearson VII,
   Pearson IV, split Lorentzian, Moffat, Student's t or skewed Gaussian.
2. Width model: Caglioti U, V, W (σ² = U·q² + V·q + W), Scherrer size and strain
   (crystallite size D and microstrain ε), or one fixed width per phase.
3. Texture: None, March-Dollase (one ratio r along the axis h₀ k₀ l₀ in Advanced), or
   free per-peak heights within the free-height range in Advanced.
4. σ min and σ max in Advanced bound the width; blank uses automatic limits.

Units and conventions
q in Å⁻¹; σ in Å⁻¹; D in Å; ε dimensionless. Extra shape parameters (fraction,
exponent, skew) are fitted for the profiles that have them.

Assumptions
Texture applies to CIF phases. A lattice phase always fits free heights, because it
has no structural intensities.

Expected observations
The table lists each phase's lattice and fraction; Scherrer adds D and ε, Caglioti U,
V and W. A March-Dollase r near 1 means little preferred orientation.

Diagnostics
FIT_PARAMETER_LIMIT_EXCEEDED: free heights over many reflections can exceed the
512-parameter limit; raise the minimum intensity or narrow the fit range.

Common mistakes and corrective actions
Free texture with fractions: free heights trade off with the phase scale, so fractions
become less reliable. Prefer March-Dollase when one orientation dominates.

Worked example
A thin film with (111) preferred orientation: Texture March-Dollase, axis 1 1 1. If r
fits near 0.7, intensities along (111) are enhanced.

References
xrd_tools/analysis/fitting/phase_fitting.py — profiles, width models and texture.
"""),
    HelpTopic("phase-analysis.background", "Phase Analysis: background", """Purpose
Model what is not crystalline: a baseline removed before the fit, a background fitted
with the phases, a measured substrate template, and an amorphous peak.

Required inputs
A displayed radial trace. A substrate template also needs the bare substrate's pattern,
processed with the same calibration and displayed (for example an Average result).

Steps
1. Pre-fit background: None, SNIP (width in points in Advanced; blank is 5 % of the
   trace) or Chebyshev; it is subtracted before the fit and has no free parameters.
2. Fitted background: Linear, Polynomial, Chebyshev or Spline of the chosen degree
   (Spline uses at least four knots), refined with the phases.
3. Substrate: in the template row, Use displayed captures the displayed pattern
   (named by its file, frame and mode once a fit confirms that file holds it), or
   Choose file… takes a NeXus result or an .xye file. Then choose Substrate (one free
   amplitude) or Substrate plus linear, polynomial or Chebyshev. Clear forgets it.
4. Amorphous peak: Gaussian, pseudo-Voigt, Lorentzian, Voigt or Lorentzian², with a
   starting centre and σ.

Units and conventions
A captured substrate is in q (Å⁻¹), through the same input path as the fit; a file's
unit must convert to Å⁻¹ (nm⁻¹ does, 2θ does not). Centre and σ are in Å⁻¹. Outside
the template's q range its value is held flat.

Assumptions
The substrate pattern scales as a whole; its shape does not change between samples.

Expected observations
The fitted background and the amorphous peak are drawn as their own curves. A template
amplitude near the substrate's share of the scattering is expected.

Diagnostics
A substrate choice without a template refuses Fit by name. A template needs at least
two distinct finite points; BACKGROUND_TEMPLATE_ codes name a file that is missing,
changed or in another unit.

Common mistakes and corrective actions
An amorphous peak can absorb broad crystalline intensity: start its centre on the
hump and keep σ broad. Do not combine SNIP with a fitted background of high degree.

Worked example
Fused silica under a thin film: an amorphous Gaussian near 1.55 Å⁻¹ with σ 0.3 Å⁻¹, or
the bare substrate captured as a template with one free amplitude.

References
xrd_tools/analysis/fitting/phase_fitting.py — template, fitted background, amorphous.
xrd_tools/analysis/fitting/background.py — SNIP and Chebyshev baselines.
"""),
    HelpTopic("phase-analysis.constraints", "Phase Analysis: constraints and solver", """Purpose
Bound how far the fit may move each lattice and the whole pattern, keep lattice axes in
order, choose the fitted q interval and set the solver's budget and weighting.

Required inputs
Included CIF or lattice phases with starting cells.

Steps
1. Lattice ±%: each cell length may move within this percent of its start.
2. q-shift bound: one global shift of every reflection, within ± this value.
3. Cross-phase lock keeps the same axis letter ordered across phases.
4. Fit range: drag the range on the plot or type it; only these points enter the
   solve, while the whole trace is drawn.
5. In Advanced, Max iterations caps solver evaluations (blank is automatic) and Use
   uncertainties weights points by their measured σ.

Units and conventions
The q-shift bound is in Å⁻¹ and must be positive. The fit range is in the displayed
axis; a 2θ range is converted with the recorded wavelength.

Assumptions
Cell angles are not refined; symmetry-equal axes stay tied.

Expected observations
A refined lattice parameter well inside its bound. A q-shift pinned at its bound means
the bound is too tight or the calibration is off.

Diagnostics
INSUFFICIENT_FIT_POINTS: fewer than ten finite points inside the fit range.

Common mistakes and corrective actions
A tight lattice tolerance stops a strained film from fitting: widen it. Use
uncertainties only when the trace carries measured uncertainties.

Worked example
Lattice ±2 %, q-shift bound 0.02 Å⁻¹, fit range 1.0 to 5.8 Å⁻¹, 4000 iterations, as
in the HZO batch fitting notebook.

References
xrd_tools/analysis/fitting/phase_fitting.py — bounds, ordering and q-shift.
xrd_tools/analysis/display_fit_operations.py — the displayed fit's validation.
"""),
    HelpTopic("phase-analysis.batch", "Phase Analysis: batch, settings and export", """Purpose
Fit many results with the same settings, seed each fit from the previous one, save and
reload the window's settings, and export a notebook that repeats the fit.

Required inputs
Stitch results for Batch (Int's Batch is not available here), and a writable folder
for saved settings.

Steps
1. Batch fits the selected results, or all listed results, and plots the trend.
2. Sequential seeds each fit from the previous fit's values; a fit whose reflections
   differ starts fresh, and a refused fit leaves the seed as it was.
3. The Trend column plots a scan column instead of the result order.
4. Save settings… writes every option, the phase list and the substrate's file once a
   fit has named it; Load settings… restores them. A substrate captured but not named
   by its file, and Sequential's seeds, are not saved.

Units and conventions
Saved settings are a versioned JSON document. The trend's x is the result order or
the chosen scan column.

Assumptions
Neighbouring results are similar when Sequential is on.

Expected observations
With Sequential on, each fit starts near its neighbour's answer and converges faster.

Diagnostics
A foreign or newer settings file is refused and nothing changes; missing phases are
named. BACKGROUND_TEMPLATE_NOTEBOOK_UNAVAILABLE: export waits for a substrate named by
its file and recorded by a fit.

Common mistakes and corrective actions
Sequential across a phase change carries the old values: turn it off there. After
Load settings, a substrate whose file has moved says so: choose it again.

Worked example
A composition series: fit the first result, check it, then Batch with Sequential on;
save the settings to repeat the series later or in a notebook.

References
xdart/gui/analysis/snapshot_fit_host.py — Batch and the trend.
xrd_tools/notebooks/recipe.py — exported fit recipes.
"""),
    HelpTopic("peak-fit.model", "Peak Fitter: peak model", """Purpose
Choose the profile fitted to every peak: Gaussian, Lorentzian, pseudo-Voigt, Voigt,
Pearson VII or split pseudo-Voigt.

Required inputs
A displayed 1-D trace with resolved peaks.

Steps
1. Model: Pseudo-Voigt (the default), Gaussian, Lorentzian, Voigt, Pearson VII or
   Split pseudo-Voigt.
2. PV fraction (Advanced) starts the Lorentzian fraction of a pseudo-Voigt or split
   pseudo-Voigt, from 0 to 1; it is fitted within those bounds.
3. Fit, then read each peak's centre, width and area in the table.

Units and conventions
Centres and widths are in the displayed axis unit. A split pseudo-Voigt fits a left
width σ and a right width σ_r.

Assumptions
One profile describes every peak in the fit range.

Expected observations
Pearson VII and split pseudo-Voigt follow heavy-tailed or asymmetric peaks that a
Gaussian or Lorentzian cannot; without a σ start they start from the data's half
maximum.

Diagnostics
A fit that does not converge says so in the status line; its values are the best
effort reached.

Common mistakes and corrective actions
A Lorentzian on a Gaussian peak overestimates the tails and the area: compare the
residuals. Use split pseudo-Voigt only for visibly asymmetric peaks.

Worked example
A strained film reflection with a low-q tail: Split pseudo-Voigt, then compare σ
with σ_r.

References
xrd_tools/analysis/fitting/fit.py — fit_peaks and its profiles.
xrd_tools/analysis/fitting/models.py — the split pseudo-Voigt model.
"""),
    HelpTopic("peak-fit.background", "Peak Fitter: background and templates", """Purpose
Model the intensity under the peaks: a constant, linear or Chebyshev background, or a
measured template (a substrate's or an empty holder's pattern) scaled by one
amplitude, alone or plus a linear or Chebyshev correction.

Required inputs
For a template: the bare background's displayed trace, or its file: a NeXus result
(a frame and a 1-D mode) or an .xye file with its x unit.

Steps
1. Background: Linear, Constant, None or Chebyshev (3); or Template, Template plus
   linear, or Template plus Chebyshev (3).
2. Template row: Use displayed captures the displayed trace and names it by its
   file, frame and mode; Choose file… takes a NeXus file (enter the frame; a single
   result's mode is default) or an .xye file (confirm its unit). Clear forgets it.
3. Fit. The first fit reads the file and records its content; the row then names
   the template, which is saved with the settings and exported to notebooks.

Units and conventions
A template's x must be in the fit's unit or convert to it without other data: Å⁻¹
and nm⁻¹ of the same axis, or degrees and radians; 2θ never converts to q here.
Outside its range the template is held flat. Its amplitude bg_A is at least 0, and a
correction is fitted under bgx_ names.

Assumptions
The template has the background's shape; only its scale changes. Uncertainties are
conditional on the template: its own noise (an .xye σ column) is not propagated.

Expected observations
bg_A near the background's share of the scattering, and a flat residual away from
the peaks.

Diagnostics
BACKGROUND_TEMPLATE_UNIT_UNSUPPORTED: the template's unit does not convert to the
fit's. BACKGROUND_TEMPLATE_CHANGED: the file changed since its content was recorded;
choose it again. BACKGROUND_TEMPLATE_UNREPRODUCIBLE: the displayed trace's file does
not hold that curve, so the capture stays unnamed (usable, not saved or exported).
BACKGROUND_TEMPLATE_NOTEBOOK_UNAVAILABLE: export waits for a named, recorded file.

Common mistakes and corrective actions
A template processed with another calibration or mask does not match the sample:
process it with the same settings. A missing .xye unit is never guessed: choose it.

Worked example
A film on silicon: process the bare wafer, choose its NeXus result (frame 0, mode
default), choose Template plus linear, Fit, then export a notebook that reads the
same file.

References
xrd_tools/analysis/background_template.py — sources, reading and units.
xrd_tools/analysis/fitting/models.py — the scaled template model.
"""),
    HelpTopic("peak-fit.peaks", "Peak Fitter: peaks and starting values", """Purpose
Choose which peaks are fitted and where each fit starts: automatic detection, a peak
count or typed centers, with optional width starts and bounds.

Required inputs
A displayed 1-D trace and a fit range that holds the peaks.

Steps
1. Auto finds up to twelve peaks in the fit range; turn it off to fit a count of 1
   to 12 peaks spread across the range.
2. Peak centers (Advanced), comma-separated, seed the fit and override Auto and the
   count.
3. σ initial starts every width; σ min / max bound it.
4. Center ± delta keeps each centre within that distance of its start.
5. Max iterations caps the solver's evaluations; blank or 0 is lmfit's default.

Units and conventions
Centers, σ and the delta are in the displayed axis unit.

Assumptions
Each detected or typed centre is one peak of the chosen profile.

Expected observations
One fitted marker per peak; with a delta, each centre stays near its start.

Diagnostics
PEAKS_NOT_FOUND: Auto found no peak in the range. INSUFFICIENT_FIT_POINTS: fewer
than five points, or three per peak, in the range.

Common mistakes and corrective actions
For overlapping peaks, type both centers rather than relying on Auto. A tight σ
bound can pin a width at its bound: widen it.

Worked example
Two overlapping reflections near 2.10 and 2.15 Å⁻¹: centers 2.10, 2.15, σ initial
0.01 and center ± 0.02.

References
xrd_tools/analysis/display_fit_operations.py — peak selection and validation.
xrd_tools/analysis/fitting/fit.py — starting values and bounds.
"""),
    HelpTopic("peak-fit.range", "Peak Fitter: fit range", """Purpose
Choose the interval of the displayed trace that is fitted.

Required inputs
A displayed 1-D trace.

Steps
1. Drag the shaded range on the plot, or type its low and high values.
2. Full resets the range to the whole trace.
3. The range is kept across Reload and clamped to each new trace.

Units and conventions
The range is in the displayed axis unit; only finite points inside it are fitted.

Assumptions
The range holds every peak and enough background on both sides to define it.

Expected observations
The fit, background and residual are drawn over the range; the data outside it stay
visible.

Diagnostics
INVALID_PEAK_PLAN (invalid fit bounds): the range lies outside the trace or its low
is not below its high.

Common mistakes and corrective actions
A range that cuts a peak's tail biases its width and area: include the tails. A
Chebyshev background over a wide range can follow broad peaks: narrow the range.

Worked example
One reflection near 2.2 Å⁻¹: drag from 2.0 to 2.4 Å⁻¹, then Fit.

References
xdart/gui/analysis/peak_fit_dialog.py — the range row and the dragged range.
xrd_tools/analysis/display_fit_operations.py — the fitted points.
"""),
    HelpTopic("peak-fit.weighting", "Peak Fitter: weighting", """Purpose
Choose whether points are weighted by their measured uncertainties.

Required inputs
A trace that carries measured uncertainties (σ); otherwise the option is off.

Steps
1. In Advanced, Use uncertainties weights each point by 1/σ.
2. Leave it off for an unweighted least-squares fit.

Units and conventions
σ is in intensity units and the weights are 1/σ. No uncertainties are inferred from
the counts.

Assumptions
The measured σ are correct and independent between points.

Expected observations
A weighted fit follows the low-noise points more closely, and its parameter errors
scale with σ.

Diagnostics
FIT_UNCERTAINTIES_REQUIRED: the trace has no measured σ. INVALID_FIT_UNCERTAINTIES:
a σ in the range is not finite and positive.

Common mistakes and corrective actions
Turn it on only when σ was measured or propagated; an assumed σ makes the weights
and the errors meaningless.

Worked example
A trace processed with error propagation: Use uncertainties on, Fit, then compare
the errors with an unweighted fit.

References
xrd_tools/analysis/display_fit_operations.py — measured σ admission.
"""),
    HelpTopic("strain.sin2psi", "Strain Analysis: the sin²ψ method", """Purpose
Measure the in-plane residual stress of a film or surface layer from how one
reflection's plane spacing d changes with the tilt ψ of its planes, in a
grazing-incidence fiber map. The illustrated guide explains the physics.

Required inputs
A GI q–χ (fiber) 2-D result from Int; one reflection's q range; its stress-free
spacing d0 (typed, a phase-library line, a stress-free reference result, or ψ*);
the reflection's diffraction elastic constants s1 and ½s2, or E and ν for the
isotropic assumption. Refraction needs the incidence angle, energy and material.

Steps
1. Cut the map into χ sectors (centres and width) on one or both χ branches.
2. Fit the reflection in each sector (pseudo-Voigt, Pearson VII or split
   pseudo-Voigt) for its d; a sector without enough finite data is left out.
3. Fit d against sin²ψ, with ψ = |χ|: weighted by 1/σ² once three points carry
   an error, otherwise unweighted.
4. With d0 and the elastic constants, read the equibiaxial stress from the
   slope m: σ = m/(½s2·d0).
In a notebook, run_strain(StrainPlan(...), result2d) runs these steps.

Units and conventions
ψ and χ in degrees, χ measured from the surface normal (0°) toward the in-plane
direction (90°); d and d0 in Å; stress in Pa; s1 and ½s2 in 1/Pa. A sector's ψ
is the mean |χ| of its bins. Positive stress is tensile.

Assumptions
Equibiaxial in-plane stress with none normal to the free surface; one
reflection's elastic constants; d linear in sin²ψ. Uniaxial stress and shear
are refused: they need each point's in-plane direction, and the +χ and −χ
branches of a fiber map are mirror images, not two in-plane directions.

Expected observations
A straight line of d against sin²ψ: a positive slope is tensile, a negative
one compressive. The line crosses d0 at the strain-free direction ψ*, where
sin²ψ* = −2s1/½s2 (2ν/(1+ν) for isotropic constants). Each branch's own fit
is descriptive only.

Diagnostics
A fit needs three usable points at two distinct ψ. A weighted χ²_red above 1 is
named in the notice: the points scatter more than their errors, and the formal
error leaves that scatter out. Refraction is estimated, never corrected: near
the critical angle, or with no material chosen, the stress is apparent.

Common mistakes and corrective actions
Taking the intercept as d0: it is d at ψ = 0, which the strain already shifts.
Treating a phase-library d0 as stress-free: it is a chosen reference, so the
absolute stress inherits its error. Reading the branch difference as shear.

Worked example
Pt 111 under 1 GPa of equibiaxial tension (E = 168 GPa, ν = 0.38,
d0 = 2.2650 Å): the slope is d0·½s2·σ ≈ 0.0186 Å and ψ* ≈ 47.9°, as the
illustrated guide's figure shows.

References
xrd_tools/analysis/strain.py — sectors, the sin²ψ regression, d0 and elastic
constants, the refraction estimate.
xrd_tools/analysis/plans.py — StrainPlan and run_strain.
"""),
)


# Finite presentation references, resolved lazily by the help consumer. The
# original scientific topic text remains unchanged and import-light.
_FIGURES = {
    'geometry.orientation': ('coordinates', 'mountings', 'first-pixel', 'off-centre',
                             'centred', 'arm-motion', 'mirror-tie', 'tilt', 'psic', 'fourc'),
    'calibration.multi-step': ('psic', 'off-centre'),
    'calibration.frame-choice': ('arm-motion', 'centred'),
    'calibration.diagnostics': ('residuals',),
}
#: Each tool's illustrated physics guide (NEXT-128-HELP-GUIDES), by topic family: an offline page under assets/help
#: that the tool window's top "?" and the Help window open in the system browser. Its figures are drawn by
#: xrd_tools.guide_figures from xdart's own models.
_GUIDES = {"peak-fit": "peak-fit.html", "phase-analysis": "phase-analysis.html", "strain": "strain.html"}
GUIDE_DIR = Path(__file__).resolve().parent / "assets" / "help"
TOPICS = tuple(replace(t, figures=_FIGURES.get(t.identifier, ()),
                      run_key='orientation-check' if t.identifier == 'geometry.orientation' else None,
                      guide=_GUIDES.get(t.identifier.split(".", 1)[0]))
               for t in TOPICS)


def get_topic(identifier: str) -> HelpTopic | None:
    return next((topic for topic in TOPICS if topic.identifier == identifier), None)


def guide_path(identifier: str) -> Path | None:
    """Where the topic's guide ships, or None when the topic has no guide; the caller reports a missing file."""
    topic = get_topic(identifier)
    return None if topic is None or topic.guide is None else GUIDE_DIR / topic.guide


def search_topics(query: str) -> tuple[HelpTopic, ...]:
    terms = str(query).casefold().split()
    return tuple(topic for topic in TOPICS if all(term in
        f"{topic.identifier} {topic.title} {topic.text}".casefold() for term in terms))


_GUIDANCE = {
    "SOURCE_CANDIDATE_LIMIT_EXCEEDED": "At most 256 source candidates and 1 MiB of canonical candidate storage.\nNext action: select an exact source or narrow the discovery request.",
    "INVALID_SOURCE_SELECTION": "The source-selection options do not form a supported, consistent request.\nNext action: check the source, selection mode and accompanying options, then request metadata again.",
    "SOURCE_NOT_FOUND": "Required source content could not be found or accessed.\nNext action: check the source and member paths, file availability and access permissions, then reopen the source.",
    "SOURCE_SELECTION_REQUIRED": "One exact source, scan or entry must be selected.\nNext action: choose an available source, scan or entry, then request metadata again.",
    "SOURCE_KIND_UNSUPPORTED": "This analysis path does not accept the selected source kind.\nNext action: choose a supported file-based source.",
    "SOURCE_IDENTITY_MISMATCH": "The requested or retained source or frame identity could not be confirmed.\nNext action: reload the intended source and metadata, then recompute the current analysis.",
    "INVALID_CANONICAL_VALUE": "The input or metadata does not satisfy the operation’s supported value format.\nNext action: check the source-selection values and metadata for malformed or unsupported values, then reload.",
    "METADATA_COLUMN_PROJECTION_UNSUPPORTED": "This source cannot provide the requested occurrence-aware metadata-column projection.\nNext action: remove the explicit projection or choose a source that supports it.",
    "METADATA_COLUMN_PROJECTION_INVALID": "A requested column name or occurrence is absent from the available metadata columns.\nNext action: select existing column names and occurrences, then reload metadata.",
    "METADATA_COLUMN_LIMIT_EXCEEDED": "At most 256 table columns, including the generated frame-index column.\nNext action: request fewer metadata columns or choose a source with a smaller table.",
    "METADATA_TABLE_LIMIT_EXCEEDED": "The canonical metadata-table payload must fit within 64 MiB.\nNext action: reduce the requested table or source payload, or choose a smaller source.",
    "SOURCE_REVISION_CHANGED": "A source or dependency no longer matches the file state recorded for this operation.\nNext action: let file changes finish, then reload metadata and repeat the analysis.",
    "METADATA_ROW_LIMIT_EXCEEDED": "At most 100,000 metadata rows.\nNext action: choose a source with fewer frames or rows.",
    "CANCELLED": "The operation ended with cancellation rather than a completed result.\nNext action: start the analysis again when ready.",
    "SCAN_PLOT_NONNUMERIC_COLUMN": "A requested plot column is absent, ambiguous, or lacks numeric data.\nNext action: choose a uniquely available numeric column from the current metadata table.",
    "METADATA_TABLE_REQUIRED": "Plotting requires a completed metadata table.\nNext action: load metadata successfully, then request the plot again.",
    "SCAN_PLOT_Y_REQUIRED": "No Y series could be selected for the plot.\nNext action: select an available numeric Y column or compute an ROI series.",
    "SCAN_PLOT_TRACE_LIMIT_EXCEEDED": "At most 16 requested or emitted plot traces.\nNext action: select fewer Y series and account for names that produce both metadata and ROI traces.",
    "SCAN_PLOT_COLUMN_NOT_FOUND": "A requested Y column is available in neither the metadata table nor the supplied ROI result.\nNext action: select a current available column or recompute the intended ROI series.",
    "SCAN_PLOT_LIMIT_EXCEEDED": "The canonical plot payload must fit within 64 MiB.\nNext action: plot fewer series or choose a source with fewer rows.",
    "P3_7_LIVE_FITTING_UNAVAILABLE": "Live fitting is unavailable in this analysis path.\nNext action: use an available fitting action; this live action remains disabled.",
    "P3_7_BATCH_DISPLAY_PROJECTION_UNAVAILABLE": "Batch fitting through this display-projection path is unavailable.\nNext action: use an available fitting action; this batch action remains disabled.",
    "P3_7_PHASE_TEXTURE_UNAVAILABLE": "Texture selection is unavailable in this phase-fitting path.\nNext action: continue with the available phase-fitting settings.",
    "P3_7_EXPORT_UNAVAILABLE": "This export action is unavailable in the current analysis path.\nNext action: continue inspecting the current result; this export action remains disabled.",
    "P3_7_ANALYSIS_TERMINAL_INVALID": "No valid terminal update is available for adoption.\nNext action: wait for the operation to finish or request it again; report the literal code if it repeats.",
    "P3_7_ANALYSIS_STALE": "The result no longer belongs to the current analysis request.\nNext action: request analysis again for the current source and settings.",
    "P3_7_ANALYSIS_REFUSED": "The outcome is not eligible for adoption as a completed result.\nNext action: review the original code and details, then retry a supported request.",
    "P3_7_GUI_RETENTION_LIMIT": "Admission requires a valid nonnegative integer byte charge and no more than 128 MiB of combined retained analysis data.\nNext action: reduce requested data and retry; report this code if a small valid request still refuses.",
    "P3_7_ANALYSIS_RESULT_IDENTITY_MISMATCH": "The returned result or its qualification does not match the current request.\nNext action: reload the current source and metadata, then request analysis again.",
    "ROI_RAW_UNAVAILABLE": "The selected full raw preview could not load.\nNext action: check source/raw accessibility and request preview again. No discarded exception detail is inferred.",
    "ROI_RAW_SHAPE_INVALID": "The preview is not one 2D detector image.\nNext action: select a source providing a 2D detector frame and request preview again.",
    "ROI_PREVIEW_LIMIT_EXCEEDED": "The whole-frame preview exceeds the existing 64 MiB canonical admission cap.\nNext action: select a smaller source frame; a smaller rectangle cannot lower the whole-frame charge.",
    "ROI_IDENTITY_MISMATCH": "ROI/table/source/labels or plotted series do not agree.\nNext action: reload the current table and recompute current ROI statistics; the failed comparison is not inferred.",
    "ROI_SIGNAL_COUNT_INVALID": "The existing reducer accepts 1–5 signal ROIs.\nNext action: keep one to five signal definitions, then Compute again.",
    "ROI_RAW_UNAVAILABLE_ALL": "Every completed raw frame was unavailable in the completed result.\nNext action: check raw access before recomputing; absent points are not covered zeros.",
    "ROI_INVALID_MASK_IGNORED": "Static-mask resolution was attempted and failed; that mask was not applied.\nNext action: supply a mask resolving to detector shape and recompute through the path that accepts it.",
    "ROI_RESULT_LIMIT_EXCEEDED": "The result exceeds the existing 64 MiB canonical admission cap.\nNext action: reduce requested frames or signal count; the cap is not a pixel-dimension limit.",
}


def diagnostic_text(code: str, details: tuple[str, ...] = ()) -> str:
    """Keep every literal string first, then append guidance for exact known codes."""
    originals = (code, *details)
    guidance = tuple(f"{value}\n{_GUIDANCE[value]}" for value in dict.fromkeys(originals) if value in _GUIDANCE)
    return "\n".join(originals) + ("\n\nGuidance\n" + "\n\n".join(guidance) if guidance else "")
