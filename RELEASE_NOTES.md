# Release notes

## 1.2.21 (2026-11-27)

### Fixes

- **RSM and Stitch Rows/Columns follow the calibration's detector:** when
  Advanced Rows/Columns were left from another detector (an Eiger, say),
  headerless RAW frames were refused ("raw detector layout does not match
  the exact file size"), and changing the detector in Calibrate did not
  change them. Rows/Columns now take the shape the calibration was made
  for when a calibration is chosen, saved or restored, and again before
  Preview, Run, Live or Make Mask. Eiger and NeXus readers use the shape
  recorded in the file. RSM also shows that shape after finding the scans.
  Preview still checks the image shape against the calibration. The RAW
  fields stay fixed during a Run or Live, and keep their current values
  when the calibration provides no shape, as with Stitch's PONI inputs.

## 1.2.20 (2026-11-27)

### Changes

- **Per-scan Live in RSM and Stitch:** with Live (◉) on, Run watches the
  selected SPEC file, NeXus files or folder and processes each scan once it
  has finished, through the page's normal Run. Results are saved exactly as
  an ordinary Run saves them. A SPEC scan counts as finished when it has its
  planned points or a complete later scan header follows it, and its
  detector files are all present and unchanged. A Bluesky/NXWriter
  container is taken once every entry in it is marked complete. A scan
  whose output is already current is skipped, and the view stays. Stop
  cancels the current scan; closing the page stops Live and waits for a
  running job to end.
- **Live preview while a scan records:** when a SPEC scan writes one
  detector file per point (RAW, TIFF or CBF), the RSM map or Stitch pattern
  fills in as the points arrive, titled "Provisional · scan N". The grid
  grows when a new point falls outside it. RSM applies a provisional
  all-frame hot-pixel mask from the second point, and the preview is capped
  at 1,000,000 voxels.
  Nothing is written during the scan; when it finishes, the saved result
  replaces the preview. The newest scan stays on screen and an older scan's
  saved result goes to the results list. The preview is on by default;
  "Preview frames" in Advanced (also the "Preview on" button next to Run)
  turns it off. Scans whose detector files appear only at scan end (Eiger
  HDF5) and NXWriter containers get per-scan Live without a preview.
- **Hidden files stay out of the browsers:** the Data Browser and the
  results lists no longer show files or folders whose names start with a
  dot. That includes the private file a save writes before it takes the
  result's name, macOS "._" copies on network shares and ".DS_Store". A
  folder's Date comes from its newest visible item; a folder with none
  shows its own date. Which files a Run processes is unchanged.

### Not yet tested

- Per-scan Live and the preview have not been run on a real beamline
  recording, on SMB shares while a detector writes, or under memory
  pressure.

## 1.2.19 (2026-11-24)

### Changes

- **RSM and Stitch run while the next scan is recording:** a Run on a
  finished scan was refused ("SOURCE_IDENTITY_MISMATCH: source metadata no
  longer matches its receipt") as soon as another scan wrote a row into the
  same SPEC file, which at a beamline is every count time. A job now binds
  only its own scan's lines and the file header in effect for it: later
  scans being written no longer refuse it, and a change to its own scan
  (a row, a UB line, an earlier scan renumbered) still does. Preparation
  and execution each parse a SPEC file once and share that view across the
  job's scans.
- **HKL maps with a UB taken from a SPEC scan stay current for Append**
  while that SPEC file grows. A map saved before 1.2.19 is gridded once
  more the first time its UB's SPEC file has grown, then stays current.
- **ROI Statistics and Metadata open the file browsed in the 2D Viewer:** an
  Eiger master, a NeXus stack or a processed NeXus selected in the 2D
  Viewer's Data Browser now opens in both windows, as a TIFF did. Before,
  ROI Statistics opened empty ("Choose a source, then Plot ROI.") and
  Metadata said "Select a retained frame first.".
- **A stack without metadata opens straight in the ROI picker:** when the
  loaded table holds only the frame index (an Eiger master without a SPEC
  file, a TIFF without a sidecar), ROI Statistics goes to the Plot ROI
  picker without a second click, whether the file was browsed or picked in
  the window. A first frame missing on disk is reported there in words. A
  file with metadata shows its table first, as before.
- **ROI Statistics reads metadata sidecars:** an image file or TIFF folder
  opened in the ROI Statistics window now shows its sidecar metadata
  columns (for example a TIFF's .txt or .pdi), as the Metadata window
  does. Before, the window read only the frame index for it.

## 1.2.18 (2026-11-22)

### Changes

- **Set BG follows every frame, scan and file:** in the 2D Viewer, Int 1D,
  Int 2D, the 1D Viewer, Stitch and RSM, a display background set on a
  selection is subtracted from every frame shown after it, in any scan or
  file, until Clear BG. Before, it was cleared on the next timer tick or
  frame choice ("immediately reset"). A curve on other bins of the same axis
  takes the background interpolated onto its bins (blank outside the
  background's range). A map of another shape or other map axes, or a curve
  on another axis quantity or unit, is drawn as it is, with a note in the
  button's tooltip. In Int the background goes at Clear BG, Clear 1D, a
  processing-mode change, a normalization edit (1-D), an image-axis edit
  (2-D), a detector-mode change (a raw background), and when the page
  closes.
- **Int 1D background with a normalization channel:** each curve is now its
  own intensity over its own monitor, minus the background's mean of each
  contributor over its own monitor (I/m − ⟨I_bg/m_bg⟩). 1.2.17 drew
  (I − B)/m with B in raw counts, so the numbers change when the monitors
  differ. A curve normalized by another channel than the one the background
  was set under is drawn as it is.
- **Overlay curves kept from earlier scans are corrected too**, and Clear BG
  brings every curve back as it was (it used to remove the kept curves).
- **Average and Sum include the curves kept from earlier scans:** switching
  an Overlay of several scans to Average or Sum used to drop the earlier
  scans' curves. The folded curve's name says how many it holds, for example
  "Average · 3 curves, 1 kept".
- **Browse Average and Sum:** they used to draw nothing ("Browse cache
  display is unavailable for Average/Sum or this operation."). They now fold
  exactly the selected frames, up to 256; more is refused with a notice, and
  nothing is drawn. Average or Sum of χ cuts stays unavailable in Browse.
- **Set BG on several frames in Browse (Int 1D):** it was refused ("Select at
  least one display frame") when more than the current frame was selected.
  Set 2D BG in Browse still takes one frame's map.
- **RSM 3D Viewer: axis arrows.** Three arrows from the volume's corner show
  the directions of its axes, each labelled, and the default view fits them.

### Fixes

- **Int 1D: opening a frame's raw image during a Run no longer clears the
  background.**
- **RSM: the background stays when its own result is read back** after
  other results were shown.

## 1.2.17 (2026-11-20)

### Fixes

- **ROI Statistics opens Eiger files:** an Eiger master picked as a file, or
  from a Folder as a raw NeXus stack, was refused (SOURCE_IDENTITY_MISMATCH),
  and so was any NeXus file picked from a Folder. They now open, preview and
  compute. The window's refusals read as sentences instead of codes.
- **ROI Statistics opens on the browsed scan:** the Tools button opens the
  window on the scan selected in the Data Browser. Another file or folder
  can still be chosen in the window, and it stays when the window is raised
  again.
- **ROI Statistics leaves out Eiger module gaps:** with Mask saturated off
  (its default), the gaps' invalid value (4294967295) was summed. A 200x200
  ROI across a gap read about 3.9e13 per frame instead of about 7e5. That
  value is now always left out, as calibration already does; Mask saturated
  still controls saturated counts, and Int's masking is unchanged. The ROI
  picker no longer draws the frame black under the gaps.
- **RSM with SPEC + Eiger takes the detector shape from the Eiger file:** it
  took the shape from Advanced Rows/Columns (the raw-file decoder's settings),
  so a 512x1028 Eiger was refused with RSM_MEMBER_IDENTITY_MISMATCH unless
  they were typed in. Rows/Columns now follow the file when Preview or Run
  finds the scans. RAW and TIFF images still use them, and a real mismatch
  with the calibration still refuses, naming both shapes.
- **RSM names the scans without images:** the Source header showed a check
  mark for a scan with no detector images. It now shows "no images: scan N"
  with the reason in its tooltip, and the other scans still run.
- **RSM: a queued Run on Windows always asked to run again:** with All
  Scans, or several RSMs, "The output changed while the scans were found: Run
  again" came back on every Run, because the check compared two spellings of
  the same output path. Both sides now use the same normalized path. Found by
  reading the code and reproduced on macOS; Windows CI now runs the test.

## 1.2.16 (2026-11-19)

### Fixes

- **Int: Folder Runs spend less time grouping scans:** Append, Overwrite and
  Live share the faster scan grouping. Finite Folder admission also indexes
  listed paths once, and Append reuses one saved-coverage science fingerprint
  per Run (the beamline PC spent about 6 s on this for 327 saved scans). In
  local testing with 300 saved scans, finite Append planning outside the
  saved-record and collision checks fell from about 1.6 s to 0.02 s, and Live
  counted all 900 saved images in about 1.3 s instead of 6.9 s.
- **Int: Append and Live Append reuse verified saved records within one
  session:** when an eligible output's observed file revision is unchanged,
  the next Run skips opening it in HDF5; a changed or unverifiable revision
  means a fresh read. Windows still opens a file handle to verify the
  revision, and the first Run after starting xdart still reads every record.
  In local testing with 300 saved scans, the saved-record checks on a repeat
  Append took about 0.005 s instead of 0.7 s.
- **Int: Folder output collision checks make fewer file-system calls:** one
  check of all outputs reuses successful path and stat answers, and on
  Windows ordinary raw files are resolved through their folder while links
  and unusual names are still resolved one by one. Outputs are still checked
  against raw files and protected inputs before writing. The saving on a
  network share is not measured.

## 1.2.15 (2026-11-19)

### Fixes

- **Int: a Folder Run's admission asked the network share about each raw
  file several times:** before a Run started, the folder listing checked
  every listed file one by one, sidecars included; it now takes each file's
  type from the listing itself. On macOS and Linux the raw-file check also
  matches raw files by their identity instead of resolving each path; on
  Windows it still resolves each path. Outputs are still refused when they
  would land on a raw file. In testing on macOS, admission of 120 saved
  images went from 720 file-system calls to 120. On Windows the saving is
  only the listing's, and it is estimated, not measured.
- **Debug log: where a Folder Run's admission spends its time:** with Debug
  Logging on, `[DIRECTORY-LIST]` times the folder listing, and
  `[APPEND-ADMIT]` splits its time into reading the saved outputs' records,
  the raw-file guard and the scans still to process.

## 1.2.14 (2026-11-19)

### Fixes

- **Int: a Folder Append Run could take minutes to start over many saved
  scans:** before it showed the first saved scan, the Run looked up the path
  of every listed file once for each scan that was already processed, so the
  wait grew with the square of the folder's size (about 2 minutes for 229
  saved scans on a network share). Each path is now looked up once per Run.
  In testing, a 30-scan folder took 720 lookups instead of 2,820; the time
  this saves on a share is estimated, not measured.

## 1.2.13 (2026-11-19)

### Fixes

- **Int: a frame read could still start in a scan the Run was releasing:**
  between scans, a Full Raw view, or a frame read queued behind another one,
  could start reading the scan the Run was releasing. Such a read is now
  refused while that scan is being released. This follows up the 1.2.12 fix
  and was found in its review.

## 1.2.12 (2026-11-18)

### Fixes

- **Int: a Run over several scans could stop between scans with "prior
  light-1D lineage cleanup remains pending":** when the Run moved on to the
  next scan while a frame image was still being read for the display (a frame
  you picked, or one the display read itself), the Run failed. Slow reads on
  a network share make that overlap more likely. The Run now goes on: it no
  longer waits for reads of other scans, and it waits up to 15 s for a read of
  the scan it is finishing.

## 1.2.11 (2026-11-16)

### Fixes

- **macOS: an in-app update could stall while rebuilding the app shortcut:**
  Check for Updates installed the new version and could then stall while pixi
  rebuilt the xdart app shortcut, and the shortcut did not open xdart while it
  was stalled. The stalled step was pixi running the environment's activation
  script for the shortcut; the macOS shortcut no longer runs it, and in testing
  the update then finished in seconds. The shortcut is rebuilt from the new
  package, so this also covers updates started from older versions. Opened from
  Finder, xdart also finds a pixi installed with Homebrew, so Check for Updates
  can still update it.

## 1.2.10 (2026-11-13)

### New

- **Remember an images-per-point grouping for a SPEC file:** the "Multiple
  images per scan point" question has **Apply to all scans in this file,
  including new ones**. Later scans of that file whose image count fits are
  grouped the same way without asking, and the choice is saved with the
  window's settings. A scan it cannot apply to (Difference on three images per
  point) still asks. The **Images per point** card lists the remembered files;
  untick one there to be asked again.
- **The grouping question confirms itself:** left unanswered, it confirms the
  grouping it shows after 30 s while it is the active window, for the rest of
  that Run's scans. Any change in it stops the countdown. RSM asks about all
  the scans that need a grouping together (scans whose grouping cannot be
  shared may still each ask).
- **The Save Path takes focus after a Project change:** after you change the
  Project (typed or with Browse), focus moves to the Save Path, shown in full
  while focused, in Int, Stitch and RSM.

### Changed

- **One Save Path rule in every tool:** Int, Stitch and RSM default to the
  Project's `xdart_processed_data` folder and keep a folder you choose. A
  relative Save Path is inside the Project.
- **The Data Browser follows the Save Path:** a Project or Save Path change
  moves it to the Save Path folder in Int, Stitch and RSM, replacing a folder
  opened with Open Folder, and a Run brings it back so each result is listed
  as it finishes and the newest is shown.
- **RSM's scan search is much faster, shows its progress and can be
  stopped:** a search of a SPEC file's scans lists each image folder once and
  shows "Finding the source's scans… N of M"; Stop works between scans. In our
  test of 100 scans and 20,700 images on a local disk, it fell from about
  264 s to 2 s.
- **All Scans and Directory skip the scans they cannot grid:** in Stitch and
  RSM, a scan with no detector images, no data points (an aborted scan), an
  unreadable data table, an image count different from its points or a gap in
  its image numbering is skipped and listed with its reason. A scan you select
  explicitly is still refused, with the reason.
- **A Live Run's saved Append history stores each image once:** it grew with
  the square of the frames (an estimated 100 MB at 1,000 single-frame epochs);
  it now grows with the frames (636 KB measured). A file whose history has two
  or more epochs, written by 1.2.10, is refused by 1.2.9's Append and
  Reintegrate.
- **Faster saves and fresh Appends:** Stitch, RSM and analysis saves check the
  new file once rather than several times, and an Append to a new output uses
  the fast writer (a failed write reprocesses that scan on the next Append and
  leaves no partial file). Config ▸ Debug ▸ Performance Monitoring shows the
  readback and hash timings.
- **Memory-aware limits on Windows use the PC's installed memory:** they used
  fixed fallbacks. A restored session keeps its saved Cores.
- **Performance Diagnostics, metadata and authored-asset dialogs use the shared
  forms and file choosers**, with tooltips.

### Fixes

- **Stop returns at once while a save is in progress:** it no longer waits for
  the writer, and the window stays responsive.
- **Opening a calibration session no longer freezes the window:** its file
  checks run in the background with Cancel; Cancel or closing the window
  applies nothing.
- **A damaged thumbnail no longer blocks a good raw image:** the thumbnail is
  read only when it stands in for a missing raw image.
- **Performance Diagnostics Apply no longer breaks later Runs:** a Live,
  Batch, Append or XYE Run ignores a pipeline choice it cannot take and says
  so; sessions and profiles from 1.2.9 that carried the setting start clean.
- **A Calibrate or Import replaces the old fit before the session is saved.**
- **`~` in calibration's Image Path and the Refine card's Geometry file means
  your home folder** (a 1.2.9 known limitation).
- **The Browse list updates in place as files arrive**, keeping its rows and
  selection.
- **The refinement toggle says its state.**
- **Calibration's "Save geometry and use" offers a place inside the Project.**
- **A grouping confirmed while a scan was still recording no longer sticks:**
  once the scan reads one image per point it is processed ungrouped; a count
  still settling is skipped with its reason and recovers.
- **Pillow is declared for the GUI**, and the GUI utilities no longer load
  pandas when imported.

### Known limitations

- Grouped scans inside an Int GI Folder Run are refused (select the scan as
  the Source), and Stitch's Directory mode refuses on the first scan that
  needs a grouping instead of asking.
- A full Reintegrate of a folder-mode Live Run later extended with Append is
  refused, as in 1.2.9.
- Stitch's Stop still waits for its own source search to finish.
- RSM settings saved by 1.2.10 do not load in 1.2.9, which starts RSM on its
  defaults.
- When the Data Browser moves to a new Save Path folder, Stitch's plot can
  clear if the result it shows is not in that folder.
- An Int Save Path typed as a relative path is now inside the Project, so an
  output an earlier version wrote under the working folder is not continued by
  Append; a new output starts in the Project.

## 1.2.9 (2026-11-09)

### Fixes

- **Stitch calibration Refine is faster:** pyFAI ran two full garbage
  collections on the calibration worker for every frame the ring search set
  up, about 12 s of each Refine in our tests. The ring search now sets up its
  frames without collecting.
- **A fit made while a calibration session restores is kept:** pressing
  **Fit** before a saved calibration session finished restoring could put the
  session's older fit back over the new one. The new fit now stays, and the
  status says "Calibration session restored; the fit made while it restored is
  kept."
- **The calibrant preview keeps its own summary:** after **Stitch calibrant**,
  "Calibrant preview complete." replaced the preview's summary of its frames,
  solid angle and normalization; the summary now stays. A preview stitched for
  a fit or inputs that changed meanwhile is discarded, and now says so: "The
  fit or its inputs changed while stitching, so this preview was discarded.
  Stitch calibrant again to preview the current inputs."
- **A typed `~` in a calibration scan's file path means your home folder:** a
  path typed as `~/…` was read inside the current folder. It, and the Auto
  image folder found from it, now resolve under your home folder.
- **One missing-SPEC warning per Run:** a Run of an Auto series named
  `<name>_scanN_M` whose SPEC file is missing logged "SPEC file … not found"
  several times; it now logs it once per Run, Live Runs included.
- **Average with a missing or unreadable PONI says so:** it stopped without a
  notice; it now says "Average needs a PONI calibration." or "Average needs a
  readable, parseable PONI calibration."
- **A folder Run keeps its file count:** a notice shown as a folder Run
  started, such as its energy notice, hid the Run's file count until its next
  progress update.
- **Peak Fitting and Phase Analysis open with their first option focused:**
  keyboard focus started on the options panel.
- **Help in Peak Fitting and Phase Analysis offers the window's illustrated
  guide:** F1 with no option focused opens the help index, which now offers
  **Illustrated guide** for that window. The Source card's **?** offers it
  again too; 1.2.8 had lost it there.
- **The install hint names the right package:** starting xdart without its
  GUI packages now suggests `pip install "xdart[gui]"` instead of
  `xrd-tools[gui]`.

### Changed

- **Faster per-frame metadata from NeXus files with many motors:** reading
  each frame's motor and counter values from a NeXus entry, as calibration's
  held batches and Average do, now plans the entry once rather than for every
  frame: 1.1 ms a frame instead of 4.5 ms with 20 positioners.

### Known limitations

- A `~` typed in calibration's **Image Path** or the Refine card's
  **Geometry file** is not expanded yet; browse to it or type the full path.

## 1.2.8 (2026-11-04)

### Added

- **Phase Analysis:** one window replaces Phase Fitting and Reference Phases.
  - Its phase list (CIF files, lattice parameters and the phase library) feeds
    both the reference overlays and the fit. Built-in calibrants are listed
    apart, under **Reference lines (not fitted)**: drawn, never fitted.
  - The fit has the options of the HZO batch-fitting notebook:
    - eleven peak profiles;
    - Caglioti, Scherrer or fixed widths;
    - March-Dollase or free-height texture;
    - a pre-fit and a fitted background, a measured substrate and an amorphous
      peak;
    - lattice and q-shift bounds.
  - **Fit all** fits every frame of a processed result in Int, or Stitch
    results, with **Sequential** seeding.
  - **Save settings…** and **Load settings…** keep every option, and every
    option has a **?**.
- **Peak Fitter:**
  - Pearson VII and split pseudo-Voigt profiles.
  - A background template from a file (a NeXus result's 1-D frame, or an .xye
    file whose unit you choose) or from the displayed trace.
  - **Save settings…** and **Load settings…**, and a **?** for every option.
  - Both fit windows now use Int's axis labels, top bar and colour maps, and
    **Components** draws each peak or phase on its own.
- **The fit windows read a source of their own:** a **Source** card, first in
  Int's Peak Fitter and Phase Analysis, fits the trace Int shows (the default)
  or a file you choose: a processed NeXus result's frame and 1-D mode, or an
  .xye file (Peak Fitter).
  - **Fit all** now works in Int: it fits every frame of the source's NeXus
    result from each frame's stored 1-D curve, then plots the results against
    the frame number. Int's χ slice, normalization and Set BG are display-only
    and are not applied; remove a substrate with the template background.
  - A notebook exported with a NeXus source names its file and frames and
    reads them again, instead of copying them.
  - The Source card's **Frame** row is the main window's `<` / frame / `>`.
    Following Int's display, it lists Int's frames: a step moves the main
    window to that frame, and the fit window then shows it. With a chosen
    NeXus file, it steps that file's frames and Int stays where it is.
- **Fit current and Fit all, with Pause and Stop:** the fit windows' Fit and
  Batch buttons are now **Fit current** and **Fit all**.
  - A running Fit all counts its frames ("Fitting all frames… 312/651"), shows
    each fitted frame in the plot and the table, and fills the trend.
  - **Pause** holds it after the frame being fitted and **Resume** goes on;
    **Stop** ends it there and keeps every frame fitted so far.
  - While it runs, its options, the fit range on the plot and Reload are
    locked. At its end it says how many frames were fitted.
  - Stepping through the frames afterwards keeps the trend, and Fit current
    shows its own fit without changing the run's rows.
  - A run stays on the file it started on: browsing another result in Int
    keeps its rows until Fit current, Reload or Fit all follows Int.
  - Save CSV during a run saves the frames fitted so far, in every window.
- **Frames are numbered as Int numbers them, from 1,** in the fit windows' plot
  titles, trend, Frame row, Source card and frame popups. The frames' stored
  labels are unchanged in files, settings and notebooks.
- **Track lists every fitted parameter:** each peak's FWHM, height, amplitude,
  σ and shape (Lorentzian fraction, Pearson VII exponent, Voigt γ), and the
  background's own parameters, which no peak choice hides.
- **Polynomial and Chebyshev backgrounds of any degree:** in the Peak Fitter,
  Polynomial replaces Linear and Constant. Polynomial, Chebyshev, Template +
  polynomial and Template + Chebyshev take a degree in a box in the Background
  row, shown only for those choices. Phase Analysis's fitted background works
  the same way, its substrate included, and keeps Spline (its knots in the same
  box).
- **Notebooks from the fit windows:** a fit exports to Jupyter and marimo
  notebooks that repeat it, with its background template once a fit has
  recorded the template's file. Phase Analysis exports fits of CIF phases.
- **Illustrated physics guides:**
  - The Help window's **Illustrated guide** opens a guide to the Peak Fitter,
    Phase Analysis or Strain Analysis in your browser. It explains the physics
    first, then what each fitted number measures. Help opens from F1 or any
    option's **?**.
  - The guides work offline, and xdart's own code draws their plots.
- **All frames:** Int Browse's **Show All** is now a persistent **All frames**
  toggle, and it works in Single too.
- **Strain analysis in notebooks:** the sin²ψ method for grazing-incidence
  fiber maps (`run_strain`), with four kinds of stress-free spacing and a
  refraction check. Its window comes in a later release.

### Changed

- **Phase Analysis's Pearson VII and Moffat profiles:** each peak's amplitude
  is now its area, as for the other profiles, so their scales and phase
  fractions change. With Moffat and a Caglioti or Scherrer width, refitted
  widths and other parameters change too.
- **Browse opens results faster:** it reads a file's frame facts when they are
  needed, not when the catalogue loads. Clicking the open result reloads it
  only if its file changed.
- **Less memory for 2-D displays:**
  - Images are held in single precision.
  - The 2D Viewer's budget is a tenth of the RAM.
  - On Windows, both viewers read the PC's own RAM, so a 16 GB PC opens
    4096 × 4096 and Eiger2 16M frames.
- **Return in the fit windows keeps the value:** in Phase Analysis and the Peak
  Fitter, Return commits the field without pressing a button or opening Help.
- **The Peak Fitter's layout:** its options are on the left and its plots on the
  right, as in Phase Analysis.
  - The rarely changed options sit in a **Peak parameters** card, not a separate
    window: Center ± delta follows Peak centers, under a Solver heading.
  - **No. of peaks** and **Auto** share one row.
  - Every option's **?** ends its row, so the **?**s form one column.
- **Fit ranges start at Full:** in both fit windows **Full** is on by default,
  and the range shows the trace's ends greyed. Turn it off to type a range, or
  drag one on the plot.
- **Linear is now Polynomial of degree 1, and Constant degree 0:** the fit is
  the same, but new fits name the background's parameters `bg_c0` and `bg_c1`
  (`bgx_c0`, `bgx_c1` after a template), not `bg_slope`, `bg_intercept` or
  `bg_c`, in the results table, the trend and saved CSV files. Settings files
  from earlier versions load as the same fit (Linear as Polynomial 1, Chebyshev
  (3) as Chebyshev 3), and notebooks exported earlier still run as written.
- **Darker data points and clearer fit lines** in both fit windows: the fit,
  its background, the residual, the fit range's edges, the peak markers and the
  trend's curves use colours that read on the plots' light panel, and the fit
  draws over its components.
- **Phase Analysis:** its options column fits every **?**, and the Phases list is
  half as tall.
- **The fit windows' options line up:** every box in the options column starts
  at one x, after the column's longest label. The **Track** boxes are as wide
  as their longest name, and **Components** has room for its bold checked text.
- **The template line shows only while a template is held:** its "No
  template" hint, and each change, go to the window's status line.
- **The trend row** reads **Track** and **Track 2**, with one menu choosing the
  peaks or phases drawn (all by default).
- **Template controls wait for a template background:**
  - Use displayed, Choose file… and Clear are enabled only with a template (Peak
    Fitter) or Substrate (Phase Analysis) background.
  - An .xye file's unit and a NeXus file's frame are asked in a popup.
  - A message about the file opens in a popup.

### Fixes

- **The end of a long Fit all no longer stalls the window:** the trend takes all
  of a run's rows in one redraw (3621 frames took 27 s, one redraw per row).
- **Fit current on Int's display draws the frame it fits:** after a run or a
  step in Int's footer, it used to draw the new fit over the trace shown before.
- **A file's frame has a readable plot title** (it showed the frame's internal
  selection).
- **A standard result after a grazing-incidence one draws its trace** on its
  own axes in Int Browse.
- **The angular coverage strip after a Run** is rebuilt for the result just
  written.
- **Browse slices follow the loaded file's revision.**
- **Scientific help no longer crashes on macOS 27.2** when it first draws a
  topic's figures.

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
