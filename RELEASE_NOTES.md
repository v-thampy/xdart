# Release notes

## 1.2.7 (2026-10-28)

### Added

- **Updates show what changed, and wait for you:** at startup and from
  **Help → Check for Updates…**, xdart shows the installed and available
  versions and a scrollable summary of the changes in each newer release, with
  **Update** and **Cancel**; **Open release notes** opens the full notes.
  Nothing installs until you choose Update, and there is no countdown. Cancel
  skips that version until xdart next starts, and a Run starting while the
  offer is open postpones it. **Help → Check for Updates at Startup** turns the
  check off. An install pinned to a version, a pip or conda install, or a
  source checkout is never offered an update at startup; **Help → Check for
  Updates…** says how to update it.
- **Updates in their own window on macOS and Windows:** xdart closes and a
  progress window shows the update as it runs, without opening Terminal or a
  console, and it cannot be closed while the update runs. When the update
  finishes, **Restart xdart** opens the new version, which says "xdart was
  updated from <old> to <new>." A failed update keeps its details on screen
  and offers no restart. On Linux, xdart closes, updates with nothing on
  screen, and reopens itself. The output is also kept in `update.log` beside
  the installation.
- **Updates on Windows:** **Help → Check for Updates…** now updates a
  `pixi global` install or the one-line installer's workspace on Windows too,
  where 1.2.6 gave a command to copy. The update waits up to 10 s for programs
  still using xdart's environment, such as Pixi's launcher; if they stay, it
  names them, changes nothing, and you start xdart again yourself.
- **A failed update is offered again:** the next start says "The update to
  xdart <version> did not complete; see <log>." **Help → Check for Updates…**
  can retry it at once, and the start after that offers it again. An update
  still running after 20 min is stopped.
- **A second start during an update says so:** opening xdart while it is being
  updated shows "xdart is being updated. Use Restart xdart in the update
  progress window when it finishes." (on Linux, "xdart is being updated and
  reopens when the update finishes.") instead of starting.
- **Pixi updates check their own channel:** a `pixi global` install looks for a
  newer xdart in the conda channel it updates from rather than on PyPI.

These updates work from 1.2.7 on. Updating from 1.2.6 to 1.2.7 uses 1.2.6's
updater: on macOS and Linux it updates in the background once you accept, and
on Windows it gives the command to run.

### Fixes

- **A scan clicked while the shown scan is still loading opens:** clicking
  another scan in the Data Browser while the shown scan's reads were finishing
  was refused ("Browse refused: previous Browse cleanup is pending"). xdart now
  shows "Opening <file> once the current scan finishes loading." and opens it
  then. A newer click replaces it; starting a Run, Reintegrate, a Viewer,
  Refresh, another folder or closing drops it; after 10 s it shows the old
  message.
- **Overlay keeps the first file's trace:** in Overlay, the trace kept from the
  first file could be lost when the second file opened while the first file's
  reads were still finishing.
- **Slices stay shown while another scan waits to open:** the shown scan's
  slice plot and its pinned slices could empty when another scan was clicked
  while the shown scan's reads were still finishing.

## 1.2.6 (2026-10-27)

### Added

- **The footer names the output file:** under Int's plots, the image shown is
  followed by the NeXus file it is saved in, by file name only, for example
  "LaB6_0001.tif [Output: LaB6_int2d.nexus]". Int 1D (XYE), which writes no
  NeXus file, shows the image alone. A footer too long for the window is cut at
  its right edge, and its tooltip shows it whole.
- **A failed Run's cleanup that keeps failing is given up:** when the same
  cleanup failure repeats for about 10 s, Run and the Data Browser's scans
  unlock, and a notice names the output still held, for example "Run unlocked:
  the failed Run's cleanup did not finish in 10 s. scan_int1d.nexus stays as
  the failure left it (Replace rebuilds it); until xdart restarts, a new Run
  skips it." Until xdart restarts, a Live or Folder Run skips that one output,
  saying why, and processes the others; a Run of that scan alone fails.
  Quitting waits at most about 10 s for a cleanup still pending, then closes
  xdart.

### Changed

- **A compact Run status:** a Live or Folder Run shows its processed files out
  of those discovered, its skipped files only when there are some, and an
  Append's new frames, for example "Running · 40/65 processed (10 new) ·
  2 skipped". A running Live Run that has caught up reads "Running · Waiting
  for new images · 65/65 processed (15 new)", where 1.2.5 read "Running · Up to
  date, waiting for new images · 65 processed (15 new) · 0 skipped ·
  65 discovered". An image waiting for its metadata is still named in the
  status, and the status at the Run's end is unchanged.
- **Skipped and reprocessed scans say why in plain words:** for example
  "scan_int1d.nexus stops growing: it was made with different integration
  settings; its frames so far are kept and its later frames are skipped in this
  Run (Replace reprocesses it)", and at the Run's end "Append skipped
  scan_int1d.nexus: it was made with different integration settings". Other
  reasons read as plainly: different results (1-D, 2-D or GI maps), another
  Project folder, a different source (another folder, path spelling or file
  type), a reintegrated output, changed images (naming the first, as "image 2,
  scan_0001.tif"), fewer images than the output holds, an earlier Append that
  did not finish, another program holding the output open, a file that cannot
  be read, or an output Append cannot continue. The "Scans reprocessed" list
  uses the same words. The technical reason is a debug line (Config ▸ Debug ▸
  Debug Logging).
- **A quieter log:** a Live image's short wait for its .txt sidecar is a debug
  line. An image still waiting after 30 s is told once ("… has waited over 30 s
  for the metadata of <image>; it is processed once its metadata arrives"), and
  so is the end of its wait. The lz4 compression note at each session's first
  write is a debug line too; disabled or unavailable compression still warns.

### Fixes

- **A failed Live Append no longer leaves Run disabled:** on the beamline's
  network share, a Live Append failed when the share reported an older time for
  its open output, and every cleanup retry then failed too ("Standard cleanup
  remains pending"): Run stayed disabled, and closing xdart retried until it
  was killed. Such a Run still fails, but its cleanup now ends and Run unlocks.
  Its output is left as any interrupted Append leaves it, so the next Append
  skips it and says to use Replace.
- **The status counts frames as they are shown:** a scan already on disk read
  0 processed until all its frames were saved; it now counts up frame by frame.
- **Windows mapped drives:** a network folder picked in a dialog
  (`\\server\share\...`) and the same folder pasted from Explorer with its drive
  letter (`X:\...`) are now one place. Dialogs give the drive letter, as
  Explorer does. A source on the share inside a Project folder spelled the other
  way no longer warns that it "is outside the project root", and its outputs
  record its path relative to the Project. The Save Path follows the Project
  folder whichever spelling either uses.
- **The skip of an Append onto an output written before 1.2.5** (no
  `epoch_local` column) says "Replace reprocesses it", as the button reads,
  where it said Overwrite.

## 1.2.5 (2026-10-26)

### Added

- **Live says when it has caught up:** while a running Live Run has processed
  everything it found, its status leads with "Up to date, waiting for new
  images", for example "Running · Up to date, waiting for new images · 561
  processed · 0 skipped · 561 discovered". A new image brings back the usual
  counts.
- **An Append says how many frames are new:** its status reads, for example,
  "Running · 35 processed (5 new)" and, once it ends, "Complete · 35 Frames
  (5 new)". Frames its outputs already held count as processed from the first
  status.
- **Warnings:** a button beside Show log (Int, Stitch and RSM) filters the log
  to warnings, errors and each Run's start and end, with a count. Pressed while
  the log is hidden, it shows the log.
- **A clearer log:** each Run opens and closes with a dated `==== … ====` line,
  and each scan with a `**** … ****` line and an empty line after it. A scan's
  end gives its frames and processing time, and skipped, stopped and rebuilt
  scans say plainly what happened. Timing and diagnostic lines appear only with
  Config ▸ Debug ▸ Debug Logging or the new Config ▸ Debug ▸ Performance
  Monitoring (remembered between sessions), and Debug Logging's lines no longer
  count as warnings. The log panel is shown at start, and its Show log /
  Warnings / Latest buttons sit at its bottom edge.
- **Data Browser filter:** it takes the control panel's filename filter and
  tooltip: space-separated terms must all match, `|` or OR gives alternatives,
  `-term` or NOT excludes, and `*` lists every result. Left empty it is Auto,
  the current mode's own results, shown faded ("Auto: _int2d"); Reintegrate and
  Average results are no longer listed by default. A **Dirs** button shows or
  hides folders in the scan list.
- **Run controls:** Pause is yellow (❚❚ Pause) and Resume orange (❚▶ Resume).
  While a Run cannot be browsed, the Data Browser's scan list is grayed out and
  its tooltip says why; the running scan's frames stay available. Switching Live
  on says to pause or stop the Run to browse the scans it has processed, and the
  Metadata window says its table is available once the Run stops.

### Changed

- **A Folder Run writes its outputs flat into the Save Path:** it no longer
  recreates the source's subfolders there. A scan name found in more than one
  folder is saved as `<folder>_<scan>` for each of them; names are compared
  ignoring case, and a name already taken gets `_2`, `_3`, and so on. A Live Run
  never renames a scan it has written. Notebook replay names and places outputs
  the same way. Outputs that 1.2.4 wrote in subfolders are not continued: an
  Append writes those scans again in the Save Path.
- **The Save Path is never read as input:** a Folder Run with Subdirs leaves out
  a Save Path inside its source folder, where 1.2.4 refused to run.
- **A Save Path folder name may not contain a dot:** choosing or typing an
  existing folder whose name has one is refused, saying why, since such a name
  reads as an output file name.
- **One metadata format per Run:** the first sidecar format a Run finds, or the
  one you chose, applies to every image in every folder of that Run.
  - In a Live Run, an image waits for its own sidecar while the other series
    and folders go on, and it is processed when the sidecar lands. The status
    and the log name an image that waits. An image whose series has moved on (a
    later image already has its sidecar) is left out and counted as skipped.
    The Run's end lists the images left out and those still waiting.
  - Under Auto, when no image of the Run has a sidecar and one has waited 10 s,
    xdart asks once whether this Run's images have sidecars. Yes: each image
    waits for its own. No, or no answer within 150 s: images are processed
    without metadata.
  - A Folder Run with sidecars leaves out images without their own. The last
    image of a series waits up to 150 s for its own only if it was written
    within the last 150 s, so reprocessing old data never waits.
  - In Live, a sidecar still changing 150 s after it appears, or a SPEC row
    missing for 30 s, skips its scan from that frame on.
- **A Live or Folder Run no longer stops at one scan:** a scan that cannot be
  read, admitted or processed (an unreadable image, a grazing-incidence scan
  without its incidence motor, a refused Append, and so on) is named in the log,
  skipped and listed at the end, and the other scans go on; in Live only that
  scan stops growing. A failure to write an output still fails the Run.
- **Changed images are reprocessed:** when an image that a scan's output already
  holds is replaced on disk by a new acquisition, an Append reduces that scan
  again whole and replaces its output, and the end of the Run lists it under
  "Scans reprocessed". For outputs written by 1.2.5, the start time in the
  image's .txt sidecar tells a new acquisition from the same image copied
  again, which is left alone; older outputs reprocess a re-dated image too. A
  Live Run rebuilds a finished scan whose images changed on disk the same way;
  the scan it is growing stops growing instead, and the next Run rebuilds it.
  Stitch and RSM Append also treat an output as out of date when one of its
  images changed size or modification time.
- **An Append onto an output written by an earlier xdart skips that scan** when
  its new frames carry the start-time column that output lacks (`epoch_local`,
  see Fixes). The scan is named and the Run goes on; Replace reprocesses it.
- **Overlay keeps its curves:** through a Run's scans, its end-of-Run Browse and
  the next Run, in Live and Folder Runs alike. Only Clear, a change to an axis
  the curves cannot share (q to q_ip, say) or a reintegration starts a new
  overlay. A fast Run's curves are all drawn by its end: the Run keeps its
  finished scans' 1-D curves in memory, within the display's memory budget, and
  ends once they are drawn. Curves excluded in Overlay stay out of Sum and
  Average.
- **A Norm Channel stays on across a Run's scans,** each scan's frames
  normalized by their own scan's values. A Run reads an already processed
  output's saved values only while a Norm Channel is chosen.
- **Legends name the scan:** curves are labelled by their scan (`b_1`, `b_2`),
  never by the output's suffix (`b_int1d_1`), during a Run and in Browse.
- **An Append ends on the last frame it processed,** even when already processed
  scans follow it in the folder.
- **Append adds new frames to the existing output in place:** xdart no longer
  copies a scan's output, or reads it in full, before and after adding the new
  frames; it opens the output and writes only the new frames. The same holds
  between a Live run's batches. The trade-off: if xdart crashes, or the run
  fails, while it is adding frames, that scan's output is left incomplete. The
  next Append skips it and says to use Replace, which rebuilds it from the
  images; if the output can no longer be opened at all, move or delete it, then
  run again. Stop is not a failure: the frames added before it are kept.
- **A directory Append over a scan that grew reads only the new frames:** the
  frames its output already holds (the same image names, sizes and times as its
  saved record) are not read or checked again.
- **A Run of many short series no longer rebuilds pyFAI's integration tables for
  each one:** xdart builds them once per Run and reuses them for every series
  with the same calibration, energy, mask and integration settings, in directory
  and Live runs alike. Each series used to rebuild them, which for a large
  detector took far longer than integrating the frame itself (a 4096 x 4096
  Rayonix frame: 2.7 s to build, 0.13 s to integrate, on a fast Mac). Results
  are identical. Grazing incidence keeps a set per incident angle, so positions
  measured at a few repeated angles build each angle once per Run (Rayonix, three
  angles: 1.8 s per series before, 0.7 s once every angle is cached); the sensor
  parallax correction is reused too. Kept sets stay within the memory the Run is
  planned for, and at most 8 per kind. Energy scans and calibrations with a
  spline or a detector flat field still build them for each series. The tables
  also stay built from one Run to the next while the settings are unchanged, so
  a Run's first frame no longer waits for them, and on Windows the memory plan
  now counts the PC's physical memory, so every worker's tables stay built. In
  Standard mode a short series on a large detector (tables of 1 GiB or more)
  starts only the workers whose tables are built, plus one per 16 frames, since
  each new worker builds its own; long series, energy scans and Live runs use
  every worker the Run is given.
- **Live over many folders on a network share starts and polls faster:** it
  reads a scan's images only once their metadata is ready, reuses directory
  listings, and checks a complete TIFF from its header instead of decoding it.
- Live rechecks settled input prefixes every ten seconds to reduce polling cost.
  A change to an already processed input can remain unnoticed within that window.
- Directory and Live outputs now use buffered writes by default, avoiding
  repeated disk flushes over a share. Checkpoint and lineage checks remain;
  a power loss can leave the newest output incomplete and require Replace.
  An explicitly enabled durable flush runs at each checkpoint and once more after
  the output closes, instead of three times per epoch.
- **Stitch and RSM:** a queue goes on to its next prepared output after one
  fails, while the form is unchanged, and its summary names the failures.
  Stitch's default Save Path follows the Project folder as Int's does, and
  Stitch and RSM Runs open and close with the same log lines as Int.

### Fixes

- **A directory Append skips scans it has already processed:** a scan whose output
  already holds every frame, from the same image files at the same sizes and times
  and with the same settings, is now skipped at once. xdart no longer reads such a
  scan again, and no longer warns "same-extent source identity changed" when only
  its metadata files or the file server's bookkeeping changed. A Live Run does the
  same: it shows a scan processed earlier from its saved output instead of reading
  it again. To reprocess a scan whose metadata you changed, use Replace.
- **Windows network shares:** Windows can set a file's creation time a moment
  after the file is written. Live and Append no longer take that for a changed
  image, and Append recognizes outputs saved while it still counted.
- **Append to an output on a network folder:** a run could stop with "writer begin
  incomplete: stream checkpoint failed identity verification" when the output's
  modification time moved a moment after xdart wrote it. Windows can set a file's
  final write time when the program closes it, and the reported failure fits that;
  xdart now records the file's state after that close. Not yet confirmed on the
  beamline share.
- **A brief file error on a share** (a permission error, or WinError 64, while a
  file is renamed into place) is retried at the next look instead of failing the
  Run.
- **An energy scan in q whose new frames reach beyond its grid:** a directory
  Append reduces it again whole on the wider grid, where it failed the whole Run;
  a Live Run keeps growing it on its first grid and drops the points outside it.
- **Saved start times are exact:** an output stored each frame's start time
  rounded to 128 s. It now keeps the full value, beside the clock time the .txt
  records (`epoch_local`).
- **The Save Path follows the Project folder** on Windows however the folder
  dialog spells it (slashes, letter case).
- **Browse keeps a scan's spelling on Windows:** its legend could show the name
  in lower case.
- **A refused Append says what differs:** the first image that differs, by name,
  size or modification time, and what to do about it.
- **The controls are no longer cut off at their right edge:** since 1.2.4's Show
  log, a window narrow enough to leave the controls column at its minimum width
  cut two pixels off the controls' right edge (Int, Stitch and RSM), and Int showed
  a thin horizontal scrollbar under them.

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
