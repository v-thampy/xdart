"""Finite offline scientific instructions; no GUI or scientific imports.

Topic IDs are presentation identifiers, independent of settings or readiness.
Diagnostic prose supplements the original strings; it never classifies a result.
"""
from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class HelpTopic:
    identifier: str
    title: str
    text: str
    figures: tuple[str, ...] = ()
    run_key: str | None = None


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
TOPICS = tuple(replace(t, figures=_FIGURES.get(t.identifier, ()),
                      run_key='orientation-check' if t.identifier == 'geometry.orientation' else None)
               for t in TOPICS)


def get_topic(identifier: str) -> HelpTopic | None:
    return next((topic for topic in TOPICS if topic.identifier == identifier), None)


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
