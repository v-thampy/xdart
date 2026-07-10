# Migration: `ssrl_xrd_tools` + `xdart` → `xrd-tools` 1.0 (monorepo)

June 2026.  The two repositories are now ONE distribution — `xrd-tools` —
holding the headless reduction core (import package `xrd_tools`) and the Qt
GUI (import package `xdart`).  Both histories were imported intact:
`git log --follow src/xrd_tools/...` / `src/xdart/...` reaches every
pre-migration commit.

| imported repo | branch @ SHA | version at import |
|---|---|---|
| `ssrl_xrd_tools` | `dev` @ `b1235a5` | 0.41.0 |
| `xdart` | `dev` @ `9b5997c` | 0.40.0 |

## Installing

```bash
pip install xrd-tools            # headless core only (no Qt anywhere)
pip install "xrd-tools[gui]"     # + the xdart GUI (PySide6/pyqtgraph)
# extras: [fitting] (pymatgen/lmfit), [rsm], [dev]
```

The `xdart` console command probes for Qt and prints a friendly
`pip install "xrd-tools[gui]"` hint (exit 1) when only the base package is
installed.  Core and GUI can no longer version-skew — the runtime
version-guard machinery from the two-repo era was deleted.

### Upgrading from the old PyPI packages

Before installing `xrd-tools`, uninstall the two old distributions:

```bash
pip uninstall -y xdart ssrl_xrd_tools
```

The old `xdart` / `ssrl_xrd_tools` installs collide file-for-file with the new
`xrd-tools` wheel (they ship the same `xdart` / `xrd_tools` import packages), so
a leftover old install can shadow the new one.  Uninstall them first, then
install `xrd-tools`.

## Imports

```python
import xrd_tools           # was: import ssrl_xrd_tools
from xrd_tools.reduction import ReductionPlan, ReductionSession, NexusSink
from xrd_tools.io import read_scan, get_1d, read_frame_view, open_scan
```

`import ssrl_xrd_tools` still works through a deprecation shim that returns
the REAL `xrd_tools` modules (true module identity — `isinstance` and
monkeypatching across the alias are safe) and emits one `DeprecationWarning`.
The shim is scheduled for removal; update imports.

## Renames / API changes (the 1.0 window)

| old | new | notes |
|---|---|---|
| `ssrl_xrd_tools.*` | `xrd_tools.*` | mechanical rename, shimmed |
| `io.read.Scan` | `io.read.ProcessedScan` | `Scan` alias kept (deprecated); collided with the reduction-input `Scan` |
| `reduction/core.py` legacy `Frame/MaskSpec/FrameSource/Scan` block | deleted | the names remain as aliases to the `xrd_tools.core.scan` contracts (same runtime classes) |
| `NexusSink.swmr` | removed | was dead: `open_nexus_writer(swmr=True)` has refused since 0.41 |
| — | `Scan.geometry` (new field) | `DiffractometerGeometry`; lets the headless `NexusSink` derive `/entry/per_frame_geometry` at finish |
| — | `xrd_tools.io.schema` (new) | schema-as-code: the processed-scan layout declared once (`SCHEMA`), consumed by writers/validators/readers |
| — | `xrd_tools.io.nexus_record` (new) | per-frame record primitives (source refs, thumbnails, `@source_base`, row surgery) shared by the headless sink and the GUI writer |
| — | `xrd_tools.core.filters.compile_filter` (new) | the boolean Filter grammar (see behavior changes) |

`xrd_tools.core` is now import-light: importing the core contracts pulls no
Qt/pyqtgraph/h5py/pyFAI/fabio (the h5py codec re-exports are lazy).

## On-disk format: unchanged

No persisted-format changes.  Attribute keys keep their historical `ssrl_`
prefixes (`ssrl_schema`, `ssrl_schema_version`, `ssrl_dtype`); the NXprocess
`@program` of the GUI writer stays `"ssrl_xrd_tools"`.  A byte-compat gate
(`tests/core/test_v2_record_compat.py`) pins the written record against a
pre-migration reference signature.  Two additive notes:

* the `ssrl_schema` *value* on newly written files is now
  `"xrd_tools.processed_scan"`; no reader ever compares it, and both names
  are declared in `SCHEMA.accepted_names`;
* a purely headless `NexusSink` run now writes the COMPLETE v2 record by
  default (`complete_record=True`): per-frame raw-source pointers (relative
  to `@source_base` when `source_base=` is given — N1-portable), thumbnails,
  and finish-time per-frame geometry.  Pass `complete_record=False` for the
  minimal pre-1.0 output;
* **(S-7 disclosure)** newly written records carry the reduction PROVENANCE
  under `/entry/reduction/config` — including a `gi` flag (and, for grazing-
  incidence scans, a `gi_config` group with the mode + frozen-range diagnostic).
  This block is **additive** and did not exist in the earliest pre-6a writer, so
  a strict byte-for-byte diff against a pre-6a capture would show it; the
  committed `v2_record_signature_pre6a.json` fixture was re-pinned to include it.
  Reload is unaffected (readers that don't know the block ignore it), and CF-3's
  target-config read + the data-derived units depend on it — so it is disclosed,
  not removed.  `test_v2_record_gi_scan_writes_gi_provenance` exercises the GI
  branch the non-GI parity fixture cannot;
* **(S-4 disclosure — changed χ axis)** for a STANDARD-mode Mode-A (I-vs-χ,
  `unit=chi_deg`) reduction with a non-zero `chi_offset` (default 90°), the
  WRITTEN 1D χ axis now matches the 2D cake χ (offset-applied) instead of the raw
  pyFAI frame it landed in before — the same frame the input range is interpreted
  in, so an explicit partial χ range integrates the intended bins.  Intensities
  are unchanged (axis relabel only); auto / full-domain reductions are unaffected;
  GI is unchanged.  **Consequence for Append:** appending a NEW run onto a
  **pre-S-4 χ file** whose stored 1D χ axis is in the old raw frame will BOUNCE at
  the writer's axis-value backstop (the axes no longer match) — switch that scan
  to **Replace**, or re-reduce the old file.  This is a deliberate refuse-not-mix,
  not a regression.

## Behavior changes to know about


* **Auto χ / χGI ranges are now deterministic (v1.0):** with the 1-D axis set to χ or χGI and
  Auto range enabled, integrations previously used pyFAI's implicit geometry-derived azimuth
  extent — the written 1D axis/intensities could differ from an explicit `-180..180` range.
  Auto now always uses/clamps to the full `-180..180` convention and is byte-equal to the
  explicit full range. Files processed with Auto χ/χGI by OLDER builds should be re-integrated
  before comparing against new outputs.
* `resolve_monitor_norm` now treats zero or negative monitor values as
  no-normalization.
* `resolve_incident_angle` now falls back to metadata when the GI motor field
  is blank.

> **Strictness (D7): the headless REDUCTION is loud by default.**
> `run_reduction` / `ReductionSession` now take a `strict: StrictPolicy`
> (default `StrictPolicy.loud()`): a scripted/batch run now **raises**
> (`MissingNormalizationError` / `GIAllDummyError` — both
> `StrictnessError(ValueError)`) on a missing monitor normalization or an
> all-dummy 2D frame, instead of silently writing degraded data.  Pass
> `StrictPolicy.graceful()` for the old never-abort behavior — the xdart GUI
> does this (it must never abort a whole-scan save).  The display reader
> `io.image_source.load_processed_raw_or_thumbnail` keeps its raw→thumbnail
> fallback **graceful by default** (a display helper); pass `StrictPolicy.loud()`
> to make it raise on a missing full-res raw.  (The headless FrameSource raw
> path stays strict by a separate, unchanged mechanism:
> `get_raw_frame(allow_thumbnail=False)`.)  Import:
> `from xrd_tools.reduction import StrictPolicy` (defined in
> `xrd_tools.core.strictness`).

1. **Filter fields (Image Directory / Eiger queue / BG Match)** use the new
   boolean grammar: space-separated terms are an **unordered** AND
   (`abc def` now also matches `def_abc`; the old glob `*abc*def*` was
   order-sensitive), `|`/`OR` for union, leading `-term`/`NOT` for
   exclusion.  Single-term filters behave exactly as before; a
   malformed expression warns and matches NOTHING until corrected.
2. **`get_metadata` energy sentinel (#78):** `energy_keV` / `wavelength_A`
   are `None` when not recorded — never NaN.  `ProcessedScan.energy*`
   hints now match reality.
3. **Streaming sink-driven sessions** (`execution="streaming"` + a sink,
   the GUI batch/live path) return `result.frames == {}` — products are
   consumed through the sink and released (S2; ~14 GB saved on a 10k-frame
   2D batch).  Chunked sessions keep retention.
4. **Monitor warnings are per scan** (S8): a dead monitor warns once per
   scan, and warns again on the next scan (was: once per process, with
   cross-session clears).
5. **GI-kind classification from units** is unified in
   `xrd_tools.core.frame_view.two_d_kind_from_units` and is leniently
   substring-matched — legacy persisted spellings (`horiz_exit`/`vert_exit`)
   are now correctly recognized by the core readers too (previously
   misclassified as `Q_CHI` outside the GUI).
6. **Chunked error-path cleanup (D6)** waits out the already-running worker
   tail before releasing image refs, so an error can no longer leave one
   frame's raw pinned until session close.
7. **Viewer raw-display LRU (D5/H9)** is scoped to Image/XYE/NeXus viewer rows,
   capping full-resolution `map_raw` payloads without making those rows a scan
   display authority.  Normal scan display now reads from `FrameRecordStore` /
   `PublicationStore` and disk hydration; the former `data_1d`/`data_2d`
   scan-display mirrors are retired.
8. **Provenance version stamps** follow the new distribution:
   `xdart.__version__` and `entry/reduction/version` report the
   `xrd-tools` version (a clean two-repo-era install recorded the old
   `xdart` dist version; a monorepo install without this fix recorded
   `''`/`0.0.0+unknown`).
9. **Detector-saturation masking (the Auto toggle in the pixel-rejection row,
   "Auto Mask Saturated", default ON).** Pixels at
   the integer detector ceiling (`np.iinfo(dtype).max` — 65535 for uint16,
   derived from the raw dtype) are masked from BOTH the raw display and the
   INTEGRATION, when at least `1e-4` of the frame sits exactly at the ceiling
   (so a handful of legitimately-saturated Bragg pixels are NOT masked — only a
   dead/overflowed block is).  This is a behavior change for uint16 detectors
   that emit 65535 as an overflow sentinel: their integrated patterns no longer
   carry that contamination by default.  Saturated counts are clipped/unreliable
   anyway, but if you need them included, untick the **Auto** toggle in the
   pixel-rejection row ("Auto Mask Saturated") — available for both image-series
   and NeXus sources (R3-B).
   In the headless/core readers the uint32 ceiling (Eiger dummy) and non-finite
   values stay always-masked; in the xdart GUI the toggle is fully authoritative
   — OFF integrates the raw frame as-is, including the uint32 sentinel. Negatives
   are masked on the RAW frame before background subtraction only. Applied
   identically across live/batch/reload (one `_resolve_frame_mask`,
   spine-verified).
10. **GI 1D empty bins are NaN** (not 0).  The GI output-axis freeze keeps a
   small coverage pad; the empty bins it (or a masked gap) creates are now
   NaN-filled — so they don't plot/aggregate as a spurious flat line at the
   low/high edge.  Aggregations are NaN-aware (`nanmean`/`nansum`).
11. **Default-loud reduction strictness (D7).**  `run_reduction` /
   `ReductionSession` / `ScanSession` now take a `StrictPolicy` (default
   `StrictPolicy.loud()`): a scripted/headless run **RAISES** on a degraded
   frame — a missing normalization or an all-dummy 2D frame — instead of
   silently writing bad data.  Errors are the `StrictnessError(ValueError)`
   family (`MissingNormalizationError` / `GIAllDummyError`) in
   `xrd_tools.core.strictness`.  **The xdart GUI is unaffected** — it passes
   `StrictPolicy.graceful()` (records + skips the bad frame per-frame, re-raises
   at `finish()`, never aborts a whole-scan save).  Scripted callers wanting the
   old never-raise behavior pass `strict=StrictPolicy.graceful()`.  No on-disk
   format change.
12. **Custom Mask File shape validation (no-built-in-mask detectors).** A custom
   Mask File whose shape does not match the detector frame is now **rejected**
   on detectors that have NO built-in mask (Rayonix-type), where it was
   previously applied unchecked.  This is correct — a wrong-shape mask cannot
   index the frame — but is a user-visible behavior change: a mismatched mask
   that silently no-op'd (or corrupted the index) before now surfaces as a
   validation error so you can supply a correctly-shaped mask.
13. **Fresh xdart sessions default Meta Type to `auto`.** Auto discovers `.txt`,
   `.metadata`, and structured name=value sidecars; choose `none`/blank to
   disable metadata, or `spec` for SPEC files.
14. **Append config-mismatch is now a Run-click MODAL (CF-1/CF-2), not a silent
   no-op.** Appending into an existing `.nxs` with a reduction config that
   differs from the stored one now prompts (Yes = switch to Replace and re-integrate, No = leave untouched) instead
   of silently mixing differently-reduced rows under a `/entry/reduction` that
   claims the first run's config.
15. **Series-average Append onto an existing averaged output is REFUSED
   (MEM-1c), not a silent no-op.**  Every source frame collapsed to output #1,
   which the skip-before-read path skipped, so the whole run wrote NOTHING with
   only a benign INFO line.  The run now refuses at start with an actionable
   message — a data-loss-class change from silent to loud.
16. **Dash/underscore filename index convention unified (`[_-]`), which CHANGES
   SCAN GROUPING (PF-2).**  A dash-separated frame index is now parsed like an
   underscore one, so `LaB6-1.tif … LaB6-60.tif` is ONE 60-frame scan (`LaB6`)
   where it was previously 60 separate single-frame scans.  Re-index any pipeline
   that relied on the old per-file grouping.
17. **Append blocks on value-affecting, grid-preserving config changes (S-3).**
   Changing `chi_offset`, monitor normalization, polarization, error model, or
   the manual GI incidence — which change the written NUMBERS while leaving the
   axis/npt/range grid identical — now trips the Append modal (they used to pass
   both the modal and the axis backstop).  The check is BACKWARD-TOLERANT: a
   field absent from a pre-upgrade stored config is treated as unknown and never
   false-triggers.
18. **Explicit out-of-domain χ ranges are passed through after the S-4 input
   shift.**  With the standard-mode Mode-A χ input-shift (item below /
   On-disk §S-4), an explicit panel-frame χ range is shifted by
   −`chi_offset` before it reaches pyFAI.  v1.0 does **not** add a second
   xrd-tools clamp for ranges that shift partly outside pyFAI's raw
   −180..180 χ domain; use Auto/full-range, or keep explicit edge ranges inside
   the shifted domain, when byte-stable edge behavior matters.  Session-1's G18
   validator pins the observed behavior; explicit clamping is deferred rather
   than silently promised here.
19. **Legacy no-config env fallback.**  `XDART_CONTROLS_PANEL_V2=0` /
   `XDART_CONTROLS_V2_NATIVE_RUN_PLAN=0` route runs through the legacy
   `plan_from_live_scan` builder, which does NOT persist the full
   `/entry/reduction/config` block the default v2 path writes (and predates the
   S-4 χ fix, now ported to it).  Prefer the default (flags unset); the fallback
   is retained only for parity debugging.

## Stage-6 redesign items: done vs deferred

Done in 1.0: 6a complete-v2-record orchestration into core (incl. headless
sink record + byte-compat gate), 6b schema-as-code starter, 6c API renames
(list above), 6d single LiveScan→core adapter + single TwoDKind classifier
(+ import-light `xrd_tools.core`), 6e cleanups + S8 + D1 + D6 + D5 + F1.

Deferred to a later release:
D2 thumbnail LRU + lazy-thumbnail reload (distinct from the landed
publication-store hydration worker), F2 outside-project Save Path consent
design, F3 ROI selection + per-scan ROI statistics (**likely first post-design
priority**), F4 embed-full-raw flag + outside-project consent popup, F5 Set Bkg
button in all display modes.

## For maintainers

* Tag `v1.0.0` when ready — the migration deliberately ships untagged.
* CI: `.github/workflows/pr.yml` (core + guards + offscreen xdart),
  `nightly.yml` (full suites), `release.yml` (build + twine check, no
  auto-publish).
* The old repos should be archived with a pointer here once the release is
  cut.

## Shipped in v1.0.0 — Plan B item 3 (headless contracts)

Originally slated post-tag, these landed on `feature/remediation` and ship in
v1.0.0 (headless source/capability/provenance extraction). Two behavior notes
for downstream users:

* **Headless runs now emit `/entry/reduction/`.** A purely headless `run_reduction` /
  `NexusSink` now writes the same `NXprocess` reduction-provenance group the GUI writer
  already produced (config + inputs), via `xrd_tools.reduction.provenance_config
  .build_reduction_config`. Additive to the frozen format; the GUI writer's bytes are
  unchanged.
* **Monitor-normalization is now guarded + case-insensitive everywhere.** The per-frame
  monitor-norm resolver was unified into `xrd_tools.core.metadata.resolve_monitor_norm`
  (canonical: case-insensitive key lookup; only finite, positive values normalize;
  0/negative/inf/NaN → no normalization, factor 1.0). Two of these are strictly safer
  (removing a div-by-zero / sign-flip / zeroing). One is a **silent numeric change**: a
  monitor-counter key whose *case* didn't exactly match the configured key was previously
  left un-normalized (factor 1.0) and now resolves and normalizes — so a dataset with a
  case-mismatched monitor key will integrate to different (correct) numbers. Re-reduce such
  datasets if exact reproduction of the old (un-normalized) values matters.

## Fixes in v1.0.0

* **Live mode no longer stops after idle gaps.** In ≤0.40.0 the live watch loop
  ended via a hidden `Timeout` parameter (default 1 second, not shown in the GUI):
  any pause longer than that with no new image — a detector-readout pause, a motor
  move between scan points, the gap between scans — quietly ended live processing
  (the only trace was a brief "Timeout occurred" in the status bar). The v1.0 watch
  loop polls indefinitely with adaptive backoff (0.1→2 s) and exits only on Stop
  (or Pause→Stop), so live keeps processing across arbitrary idle gaps.
* **Live mode no longer crashes on a partially-written detector image.** On a
  network share (SMB/NFS) the detector often creates an image file before its bytes
  are flushed; the glob would pick it up and fabio would raise on the truncated file
  (`Could not interpret magic string`). That exception was unhandled and escaped the
  worker thread's `run()`, silently killing live processing (it looked like a
  timeout). The frame read now tolerates this: a file that is present but not yet
  readable is re-polled on later sweeps (never dropped) until it finishes writing, or
  skipped after a bounded deadline (`XDART_FRAME_READ_DEADLINE`, default 30 s) so a
  genuinely corrupt file can't wedge the watch — the read never crashes the thread.
  The same tolerance covers Eiger: the master declares `nimages` up front but the
  per-frame data files stream in after, so a frame whose data has not landed yet is
  now waited for (refreshing the handle) instead of being read as end-of-stream and
  stalling the live acquisition.
* **Auto peak-detection ignores peakless/diffuse scatter.** Fitting a 1D with no
  real peak in range (e.g. a grazing-incidence cut off the rings) used to latch onto
  noise spikes, then grind a doomed many-component fit ending in a bare "Fit failed:".
  Detection now applies a robust noise-floor + width gate (light smoothing + a peak
  must rise several noise-sigma over a multi-sample span), so peakless data detects
  nothing and shows the clean "no peaks auto-detected" hint immediately.
