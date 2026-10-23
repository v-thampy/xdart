# Release notes

## 1.2.5 (unreleased)

### Changed

- **Append adds new frames to the existing output in place:** xdart no longer
  copies a scan's output, or reads it in full, before and after adding the new
  frames; it opens the output and writes only the new frames. The same holds
  between a Live run's batches. The trade-off: if xdart crashes, or the run
  fails, while it is adding frames, that scan's output is left incomplete. The
  next Append skips it and says to use Replace, which rebuilds it from the
  images; if the output can no longer be opened at all, move or delete it, then
  run again. Stop is not a failure: the frames added before it are kept.

### Fixes

- **A directory Append skips scans it has already processed:** a scan whose output
  already holds every frame, from the same image files at the same sizes and times
  and with the same settings, is now skipped at once. xdart no longer reads such a
  scan again, and no longer warns "same-extent source identity changed" when only
  its metadata files or the file server's bookkeeping changed. A scan with new
  frames still checks its earlier files before adding the new ones, and a changed
  image still warns. To reprocess a scan whose metadata you changed, use Replace.
- **Append to an output on a network folder:** a run could stop with "writer begin
  incomplete: stream checkpoint failed identity verification" when the output's
  modification time moved a moment after xdart wrote it. Windows can set a file's
  final write time when the program closes it, and the reported failure fits that;
  xdart now records the file's state after that close. Not yet confirmed on the
  beamline share.

## 1.2.4 (2026-10-22)

### Added

- **Config ▸ Debug ▸ Debug Logging:** writes detailed diagnostics to the log file
  (Help ▸ Open Log Location) from the moment you tick it, with no restart, and is
  remembered between sessions. Turn it on while reproducing a problem.
- **Show log:** a button below Reintegrate/Advanced (Int, Stitch and RSM) opens the
  log under the controls: the messages written to the log file, newest last. It
  takes the free space above the Run controls, or a few lines where the controls
  fill the column; drag its top edge to make it taller. Scroll up to read earlier
  lines; new ones wait until you return to the bottom or press Latest. Select text
  to copy it. Hidden, the panel does no work, and the log file is unaffected
  either way. Detailed diagnostics appear only while Debug Logging is on.

### Fixes

- **Live runs that Append to several existing results:** a Live directory run
  stopped with "persisted work ordinal must advance high-water" when a second
  scan's output already held frames. Live counts only new frames in its progress,
  so that output's saved frames were numbered after the new frames rather than
  after every frame already shown. They now take the next free numbers; frames
  already shown keep theirs.
- **Overlay keeps the curves when you narrow a selection:** in 1.2.2 and 1.2.3,
  choosing fewer frames of the same processed file in Overlay could leave the plot
  empty once a later repaint arrived. The remaining curves now stay drawn.
- **A directory Append no longer stops at one changed scan:** when Append refuses
  a scan (for example "same-extent source identity changed"), xdart now skips that
  scan, writes the reason to the log, processes the remaining scans, and lists the
  skipped scans in a message at the end of the run. The skipped scan's saved
  result is left untouched; use Replace to reprocess it.
- **Append refusal says what changed:** "same-extent source identity changed" now
  names the source details that differ from the processed file's record (for
  example a content digest or a modification time, with the difference). The
  refusal itself is unchanged.

## 1.2.3 (2026-10-22)

### Fixes

- **Make Mask on Windows network folders:** publish the checked mask by renaming
  it into place, without hard-link probes that left undeletable names on the
  reported share. Cleanup continues past a file it cannot delete and reports
  the reason. A changed temporary-folder identity prevents rollback writes.
- **Mask source-change diagnostics:** name the source or folder details that
  changed, including modification-time differences, while preserving the checks.

## 1.2.2 (2026-10-22)

### Fixes

- **Opening processed results through an alias:** avoid a path-identity mismatch
  that could refuse Browse adoption when a mapped drive (for example `X:`) resolves
  to its network path, or a symlinked folder resolves to another path. Browse now
  uses the same normalized path throughout loading and adoption. Adoption failures
  also log their reason and traceback instead of disappearing silently.
- **Overlay across processed files in Int:** keep compatible selected curves when
  opening another processed file, while its raw image and 2D map follow the current
  selection. Incompatible sampled axes or the detached display memory limit start
  a new overlay with a clear notice. Clear and Single remove earlier comparisons.

## 1.2.1 (2026-10-22)

### Fixes

- **TIFF scans after a packaged install:** 1.2.0 reads each TIFF's header with
  `tifffile` before a Run, but declared `tifffile` only for development, so
  `pixi global install xdart` and `pip install xdart` refused TIFF scans with
  "No module named 'tifffile'". It is now a runtime dependency. On 1.2.0,
  `pixi global add --environment xdart tifffile` works around it.
- **Make Mask on Windows network folders:** after saving and checking the mask,
  Make Mask reported a failure when Windows kept xdart from deleting its
  temporary folder, and the mask was not selected. xdart now retries the delete
  for up to 2 seconds. If the folder still stays behind, the saved mask is
  selected anyway and a warning names the folder and the Windows error; you can
  review the remaining temporary files. Other save or validation failures still
  report failure.

## 1.2.0 (2026-10-22)

This release brings Stitch 1D, Stitch 2D and reciprocal-space mapping into the
Scattering Workspace alongside integration and the viewers.

### Processing and analysis

- **Stitch 1D and Stitch 2D:** process raw detector frames with scan-angle
  metadata using pyFAI MultiGeometry or xrayutilities histogram. Standard and
  grazing-incidence workflows share calibration, masking, result browsing and
  analysis tools. Stitch 2D saves both a map and an independent 1D profile.
- **RSM:** build HKL or Cartesian-Q volumes from raw scans with a six-circle
  PSIC calibration. Browse saved volumes, open the 3D viewer, make custom slices,
  and use the native lineout, 2D and 3D fitting tools.
- **Saved results and notebooks:** reopen Stitch/RSM `.nexus` results without
  reprocessing. Export processing inputs through **Help → Export Run notebook…**
  and export supported fit workflows from their fit windows to Jupyter or
  marimo. Raw-image inspection and processing replay still need their sources.
- **Stitch text companions:** saved profiles have `iq_` or `itth_` XYE companions;
  saved grazing-incidence cuts also have `iqip_` and `iqoop_` companions. The GI
  cut header identifies its third column as an estimated uncertainty.

See the README's [Stitch](README.md#stitch-1d-and-stitch-2d) and
[RSM](README.md#reciprocal-space-mapping-rsm) workflows for input and calibration
requirements.

### Corrections and behavior changes

- Standard radial q/2θ integration preserves the requested detector directions
  for the corrected seam-crossing azimuth windows. Exact 360° Standard radial
  ranges use Auto admission. **Regenerate affected older outputs with
  Replace/Overwrite; Append does not repair old frames.**
- Active Stitch/RSM result folders use natural name order immediately, including
  results not yet read by the folder catalog. Rows keep that order when the
  catalog arrives. The newest result remains current but need not be the last
  row. No-selection fit batches follow the visible order; selected batches use
  the existing selection order.
- **Each page lists its own results.** The Data Browser's filename filter starts
  with the current mode's operation suffixes: `_int1d, _int2d, _reintegrate1d,
  _average` in Int 1D, `_int2d, _reintegrate2d, _average` in Int 2D, `_stitch1d,
  _stitch2d` in Stitch 1D, `_stitch2d` in Stitch 2D and `_rsm` in RSM. Folders and
  the results being viewed always stay listed. Typed text matches anywhere in a
  filename (separate alternatives with commas); clear the field to list every
  result, including older or hand-named files without an operation suffix. Stitch
  and RSM also offer Int's Time order.
- **Average results name their dimension:** a new Average writes
  `<family>_int1d_average.nexus` or `<family>_int2d_average.nexus`. A 1.1.5
  `<family>_average.nexus` stays readable, but a new Average does not replace it.
- On Linux and macOS, Stitch, RSM, Average and Reintegrate saves no longer delete
  or publish a file that another process put in their place.
- Calibration dialogs release their windows and pending restore callbacks with
  their owners, reducing retained GUI memory.
- Live input discovery reuses unchanged identities and avoids repeated checks
  while retaining strict admission and publication checks. A newly seen file is
  processed once a second directory listing confirms it; network directory
  visibility remains relevant.

### Dependencies and plot exports

- Update silx to 3.1.3, fixing the Python 3.13 HDF5 tree crash in **silx view**;
  update SciPy to 1.18.1, NumExpr to 2.14.2 and hdf5plugin to 7.1.0.
- Restore SVG plot export under Qt 6.11 with the upstream close-path correction
  for the pinned pyqtgraph 0.14.0. PNG and the shared export guards remain available.
- Keep NumPy 2.5.1, pyFAI 2026.9.x and xrayutilities 1.7.12. The supported native
  installers also keep FabIO 2025.10.0 and PySide6 6.11.x from conda-forge;
  the planned FabIO 2026.06 and PySide6 6.12 updates await conda-forge packages.

### Upgrading and installation

Use the [migration checklist](MIGRATION.md#upgrading-from-115-to-120) before
continuing existing outputs. The processed-file format is not bulk-migrated, and
installing the update does not recalculate saved data.

**Pixi and conda environments using conda-forge's compiled dependencies are the
supported GUI installation routes.** A pip-only GUI install is best-effort;
headless `pip install xdart` remains available. See
[installation instructions](docs/INSTALLATION.md).

### Known limitations

Read the [limitations and workarounds](docs/LIMITATIONS.md), especially the
explicit full-turn I(χ)/cake limitation, the Auto workaround, additional memory
for seam-window integrators, and the scope of full-turn Auto behavior. Release
qualification, native operation and package publication remain separate steps.
