# Release notes

## 1.2.0 (unreleased)

This release brings Stitch 1D, Stitch 2D and reciprocal-space mapping into the
Scattering Workspace alongside integration and the viewers. This entry describes
the release being prepared; it is not an announcement that packages have been
published or that release qualification is complete.

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
- Finite-result saves on POSIX preserve file identity through validation and
  cleanup, protecting files substituted outside the operation.
- Calibration dialogs release their windows and pending restore callbacks with
  their owners, reducing retained GUI memory.
- Live input discovery reuses unchanged identities and avoids repeated checks
  while retaining strict admission and publication checks. First-seen files
  still require confirmation; network directory visibility remains relevant.

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
