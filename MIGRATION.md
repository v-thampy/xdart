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
  minimal pre-1.0 output.

## Behavior changes to know about

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
7. **Hydrated-raw display LRU (D5)** is shared across all writers (GUI +
   worker threads), capping full-resolution `map_raw` payloads in `data_2d`
   regardless of which thread hydrated them.
8. **Provenance version stamps** follow the new distribution:
   `xdart.__version__` and `entry/reduction/version` report the
   `xrd-tools` version (a clean two-repo-era install recorded the old
   `xdart` dist version; a monorepo install without this fix recorded
   `''`/`0.0.0+unknown`).
9. **Detector-saturation masking ("Mask Saturated", default ON).** Pixels at
   the integer detector ceiling (`np.iinfo(dtype).max` — 65535 for uint16,
   derived from the raw dtype) are masked from BOTH the raw display and the
   INTEGRATION, when at least `1e-4` of the frame sits exactly at the ceiling
   (so a handful of legitimately-saturated Bragg pixels are NOT masked — only a
   dead/overflowed block is).  This is a behavior change for uint16 detectors
   that emit 65535 as an overflow sentinel: their integrated patterns no longer
   carry that contamination by default.  Saturated counts are clipped/unreliable
   anyway, but if you need them included, untick **Mask Saturated** in the
   Intensity-Threshold area.  Non-finite and the uint32 ceiling (Eiger dummy)
   stay always-masked; negatives are masked on the RAW frame before background
   subtraction only.  Applied identically across live/batch/reload (one
   `_resolve_frame_mask`, spine-verified).
10. **GI 1D empty bins are NaN** (not 0).  The GI output-axis freeze keeps a
   small coverage pad; the empty bins it (or a masked gap) creates are now
   NaN-filled — so they don't plot/aggregate as a spurious flat line at the
   low/high edge.  Aggregations are NaN-aware (`nanmean`/`nansum`).

## Stage-6 redesign items: done vs deferred

Done in 1.0: 6a complete-v2-record orchestration into core (incl. headless
sink record + byte-compat gate), 6b schema-as-code starter, 6c API renames
(list above), 6d single LiveScan→core adapter + single TwoDKind classifier
(+ import-light `xrd_tools.core`), 6e cleanups + S8 + D6 + D5 + F1.

Deferred (tracked in
`docs/design/CC_preship_sweep_deferred_jun2026.md`):
D1 re-integrate RAM rework (re-expose the buttons with a replace-aware
sink), D2 thumbnail LRU + lazy reload (analyzed Jun 2026; lands with the
publication-store migration), F2 outside-project Save Path consent design,
F3 ROI selection + per-scan ROI statistics (**likely first post-design
priority**), F4 embed-full-raw flag + outside-project consent popup,
F5 Set Bkg button in all display modes.

## For maintainers

* Tag `v1.0.0` when ready — the migration deliberately ships untagged.
* CI: `.github/workflows/pr.yml` (core + guards + offscreen xdart),
  `nightly.yml` (full suites), `release.yml` (build + twine check, no
  auto-publish).
* The old repos should be archived with a pointer here once the release is
  cut.
