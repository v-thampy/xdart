# Design: the shared scan-source panel (`ScanSourceWidget`)

**Status:** IMPLEMENTED + ADOPTED (ROI) · reconciled 2026-07-01. The shared
source seam, SPEC/NeXus sources, grouped/composite sources, directory-mode
discovery, the `ScanSourceWidget` itself (with async-probe support), and its
adoption in the ROI Scan Plotter are all shipped and tested (staged-plan steps
0–3 complete; `scan_source_widget.py` 544 lines, `scan_plot_dialog.py:159-161`,
`tests/xdart/test_scan_source_widget.py` 9 tests). Stitch/RSM wrangler embedding
(step 4) and Tiled (step 5) remain deferred.
**Realizes:** [`design_wrangler_organization_jun2026.md`](design_wrangler_organization_jun2026.md)
§3.1 ("Source / data") as a concrete, reusable Qt widget — and
[`design_stitching_jun2026.md`](design_stitching_jun2026.md) §5.4's source/data input
inventory.
**Driven by:** SPEC is the **primary** scan-definition source today (Vivek, 2026-06-23) —
extensionless, scan-number-addressed, with image files paired by directory + filename root —
**but the panel must be general across source kinds from the start**: SPEC now, processed/raw
**NeXus** (+ Eiger) now, **Tiled** later. Designed that way, the one source panel is "truly
general" — the ROI Scan Plotter, stitch, and RSM all consume any kind through it (Vivek,
2026-06-23). Build it **once**, shared.
**Depends on (already shipped):** the `FrameSource` + `open_source(uri, **opts)` /
`SourceSpec` seam (`xrd_tools.sources`); `SpecSource` (extensionless content detection, all
`#L`+`#O/#P` motors, scan selection, optional images via `io.image.read_image`);
`io.spec.is_spec_file/list_spec_scans/read_spec_scan_table`; `io.image.find_image_files`.

---

## 1. Why one widget

Three consumers need the *identical* answer to "**which images, grouped into scan(s), with
what metadata?**":

- **ROI Scan Plotter** (`scan_plot_dialog.py`) — one scan → metadata table + (if raw is
  reachable) ROI stats.
- **Stitch wrangler** — one *or grouped* scans → `run_stitch` per group.
- **RSM wrangler** — one *or grouped* scans → `run_rsm` per group.

The one-source-layer principle (memory `stitching_design_reframed`, wrangler-org §1) says: do
**not** fork by source type or by consumer. The ROI dialog today has an *ad-hoc* picker
(`Choose…` + a SPEC scan combo) — that was the seed; this doc promotes it to a reusable
`ScanSourceWidget` the wrangler reuses verbatim, so SPEC/image-pairing logic is written and
tested **once**.

**Thin-xdart discipline:** the widget only *assembles a `SourceSpec`* (and opens it). All
parsing/IO already lives headless (`io.spec`, `io.image`, `sources`). The widget contributes
Qt + a tiny amount of "assemble options from fields" glue.

---

## 2. The headless seam the widget targets (already exists)

```python
SourceSpec(uri, kind, *, metadata_uri=None, entry=None, options={...})
open_source(uri, **options) -> FrameSource          # or open_source(spec)
```

Everything the widget collects becomes `SourceSpec.options`, which is the **serialization +
reproducibility seam** (it round-trips; a saved scan definition re-opens the same source):

| Widget field | `options` key (consumed by) | Notes |
|---|---|---|
| Scan number | `scan` (`SpecSource`) | `"5"` / `"5.1"`; default = first scan |
| Image folder | `image_dir` (`SpecSource`) | omitted ⇒ metadata only ⇒ ROI disabled |
| Filename root | `image_stem` (`SpecSource`) | default `{spec}_scan{N}_` (trailing `_`) |
| Raw read params | `read_image_kwargs` (`SpecSource`→`read_image`) | `detector_shape`/`raw_dtype`/`raw_header_skip`/`threshold`/`detector` |
| (NeXus) entry | `SourceSpec.entry` | existing |
| (Processed NeXus) moved raw | `source_root` (`ProcessedNexusSource`) | existing (N1) |
| (Tiled, future) catalog / node | `catalog` / `node` (`TiledSource`) | reserved; no factory yet |

**Headless additions now shipped** (pure, testable, no Qt) — grouping + the composite source
that makes "combine several scans into one output" seamless (stitch/RSM design §5.1, §6 Q1):

```python
# xrd_tools/sources/grouping.py
def parse_scan_groups("1-3, 5, 7-9") -> [[1,2,3], [5], [7,8,9]]   # pure: GROUPS, not a flat list
# xrd_tools/sources/composite.py
class CompositeFrameSource(FrameSource):     # concatenates member sources into one frame stream
    """frames = s0.frames ++ s1.frames ++ …; load_frame/metadata_for dispatch to the owning
    member; motors/scan_data concatenate.  One scan-group => one source => ONE run."""
```

A **group** (e.g. `1-3`) *combines* scans 1+2+3 into a single `CompositeFrameSource` → ONE
`run_stitch`/`run_rsm`; `5` is a singleton; so `"1-3, 5, 7-9"` → **three** outputs. ROI uses a
single scan (no grouping shown); stitch/RSM turn grouping on. Keeping group-parsing + the
concatenation headless means the widget's grouping field is trivial and the "process several
scans together" capability is the same `FrameSource` everything else already consumes (it is
also how stitch's cross-file "Multi" is modelled — a composite over different files/kinds).

## 2.1 One widget, every source kind (the generality that makes it reusable)

Downstream is **already source-agnostic**: ROI/stitch/RSM only ever see a `FrameSource`
(`frame_indices` / `load_frame` / `metadata_for` / `motors`). So the widget is the *only*
kind-aware layer, and "general" means **two fields adapt by kind** — the **scan-id selector**
and the **image affordance** — while everything below is identical. The unifying abstraction:

> Every source has a **scan-id space** (what you pick) and an **image origin** (where raw comes
> from). SPEC: scan number + an external image folder. NeXus: an entry + images *inside* the
> file (raw stack) or *linked* from it (processed). Tiled: a run/node UID + images *via the
> catalog*. All three resolve to `open_source(uri, **options) → FrameSource`.

| Kind | `uri` | Scan-id selector | Image origin | Metadata | Status |
|---|---|---|---|---|---|
| **SPEC** | the (extensionless) spec file | **scan number** (`list_spec_scans` → combo) | **external** folder + stem (`image_dir`/`image_stem`) — or none ⇒ metadata-only | `#L` + all `#O/#P` (`read_spec_scan_table`) | ✅ shipped |
| **Processed NeXus** | the `.nxs` | **entry** (usually one) | **linked raw**, auto-resolved (N1); a *Repoint raw* field = `source_root` for a moved tree | `read_scan_data` (full `scan_data`) | ✅ exists (`ProcessedNexusSource`) |
| **Raw NeXus / Eiger** | the `.h5`/`.nxs` master | **entry** (+ optional dataset path) | **inline** frames in the file | positioners in-file | ✅ exists (`NexusStackSource`) |
| **TIFF / RAW series** | the first image (or its folder) | n/a (the folder *is* the scan) | the folder itself | `.txt`/`.pdi` sidecars | ✅ exists (`TiffSeriesSource`) |
| **Tiled** | a catalog URI + node path | **run/node UID** picked from a **catalog browser** | **via the catalog** (lazy) | catalog metadata | 🔜 `SourceKind.TILED` reserved; no factory yet |

So the widget's "Scan" control is a **kind-adaptive selector** (SPEC scan combo / NeXus entry
combo / Tiled catalog browser), and its "Images" control is a **kind-adaptive affordance**
(external folder for SPEC; hidden+auto for NeXus/Tiled; a *Repoint raw* field for a moved
processed-NeXus tree). The grouping range (§2) is over **whatever the scan-id space is** — SPEC
scan numbers, NeXus entries, Tiled runs — so it generalizes unchanged.

**Tiled forward seam (no speculative code now):** a `TiledSource(FrameSource)` +
`open_source` factory + a catalog-browser sub-widget land when Tiled does; the `uri`/`options`
contract (`catalog`, `node`/`uid`) is reserved so today's widget + every consumer accept it
with no downstream change. Until then the kind is simply absent from the picker.

## 2.2 Two entry modes: a single master file, or a directory + kind

The widget's source can be entered two ways (this is how the wrangler works today and how Vivek
wants to keep it):

- **File mode** — pick one master (a SPEC file, a `.nxs`, an Eiger master). Its scan-id space
  (§2.1) is enumerated *from the file* (SPEC scans / NeXus entries).
- **Directory mode** — pick a **folder** + a **Scan kind** selector (SPEC / TIFF / NeXus /
  Eiger / …). The headless layer then **walks the directory (and, optionally, subdirectories)**,
  finds the files matching that kind, and **uses `FrameSource` to identify the scan(s)** inside
  it — e.g. group a folder of `*_scanN_*` images into per-`N` scans, or list every SPEC/`.nxs`
  found. Directory mode can therefore surface **many** scans; the user then selects (and, for
  stitch/RSM, groups) among them.

Headless seam (one discovery function per kind, returning openable specs — Qt-free, testable):

```python
# xrd_tools/sources/discover.py
def discover_scans(directory, kind, *, recursive=False, **opts) -> list[SourceSpec]
```

`TiffSeriesSource.from_directory` is today's only instance; `discover_scans` generalizes it
(SPEC: each spec file × its scans, or images grouped by `_scanN_`; NeXus/Eiger: each master).
**Directory mode is available to the ROI plotter** (which wants one scan) — but because it
produces the same `SourceSpec`s, adding it is just another entry path into the same machinery,
so the widget supports it where it's cheap and the consumer can hide it.

## 2.3 Metadata is OPTIONAL — images-only sources (Eiger/raw burst mode)

Some scans have **no associated metadata** — notably **Eiger burst-mode** acquisition (and some
raw collections): a stack of frames, no motors/counters. This must stay first-class
(wrangler-org §1: metadata optional for Int; extend to ROI-vs-frame):

- An images-only `FrameSource` exposes `frame_indices` + `load_frame` with `metadata_for ≈ {}`.
  `_table_from_source` then yields **just `frame_index`**.
- **ROI stats still work** — plotted **vs frame number** (the dialog's default x). The ROI path
  must NOT require any metadata column (it already synthesizes `frame_index` from
  `source.frame_indices`); the only requirement is reachable raw, which an image stack has.
- **Integration still works** — a bare stack already integrates through the pyFAI Int 1D/2D path
  (wrangler-org §1; `ImageFileSource`/`NexusStackSource`). The source the widget emits feeds the
  *same* integrators, so an Eiger burst can be both ROI-vs-frame **and** Int 1D/2D'd.

So the **raw-reachable dot is independent of metadata**: an Eiger burst shows the dot green
(images present) with an empty metadata table, and Plot ROI + integration are both available.
*(Gap to verify in the build: the integration path supports metadata-less stacks today; the ROI
path should be exercised on one too — see §5 step 3 gate.)*

---

## 3. The widget — fields + behavior

`ScanSourceWidget(QWidget)` with one public signal: **`sigSourceChanged(ScanSelection | None)`**
(emitted when a complete, openable source is defined, or `None` when cleared/invalid). The widget
itself calls `open_source(spec)` + `probe_first_frame(source)` and emits a `ScanSelection`
dataclass (`spec`, opened `source`, `label`, `reachable`, `first_image`); consumers use the
already-opened source and reuse the probed first frame rather than re-opening. (There is also an
internal `sigProbeDone` channel used by the optional async-probe path.)

```
┌ Source ─────────────────────────────────────────────────────────┐
│ File   [ /data/run42/myscan            ] [Choose…]   kind: SPEC  │
│ Scan   [ 5.1 ▼ ]   (shown for multi-scan SPEC)                   │
│ Images [ /data/run42/images            ] [Folder…]  ● raw ok     │   ← raw-reachable dot
│   ▸ Advanced (raw read params)                                   │
│       Detector [Pilatus300k ▼] or shape [195]×[1475]            │
│       dtype [int32 ▼]  header skip [0]  threshold [   ]          │
│       orientation [0°/flip… ▼]   (stitch/RSM only — GAP E)       │
│ ▸ Grouping  [ 1-3, 5, 7-9 ]   (stitch/RSM only)                  │
└──────────────────────────────────────────────────────────────────┘
```

**File / source** — a **File ⟷ Directory** entry-mode toggle (§2.2):
- *File*: one picker, "All files" (SPEC is extensionless). `guess_source_kind` classifies
  (content-sniff for SPEC; suffix/inspect for NeXus/Eiger/TIFF); the `kind:` label reflects it.
- *Directory*: a folder picker **+ a Scan-kind dropdown** (SPEC / TIFF / NeXus / Eiger) **+ a
  "include subfolders" check** → `discover_scans(dir, kind, recursive)` lists the scans found;
  the Scan selector below becomes a list of *those* scans.

*(Tiled later adds a third entry: a catalog-URI field + a Browse… catalog dialog — same
`sigSourceChanged` contract.)*

**Scan — kind-adaptive selector** (§2.1): SPEC → a **scan-number** combo
(`SpecSource(uri).available_scans`); NeXus → an **entry** combo (usually one); Tiled → a
**catalog browser** (future). Shown only when there is a real choice (>1 scan/entry); hidden
for single-scan / single-entry / folder-series sources. (The SPEC case is already prototyped in
the ROI dialog.)

**Images — kind-adaptive affordance** (§2.1):
- **SPEC** → an external image **folder** + **filename root** (default `{spec}_scan{N}_`).
  Auto-default: when blank, try the spec file's own directory with the default stem; if
  `find_image_files` hits, pre-fill + show the **raw-reachable dot green**, else grey
  (metadata-only). Omitting it ⇒ metadata-only ⇒ raw features off.
- **Raw NeXus / Eiger** → images are **inline**; no folder field (the dot is green when the
  file has a readable image stack). An optional dataset-path override for non-standard files.
- **Processed NeXus** → raw is **linked**; no folder field, but a **Repoint raw** path
  (`source_root`) for a moved tree, with the dot reflecting whether the raw resolves.
- **Tiled** → images come **via the catalog**; no folder field.

The single invariant the consumers rely on: **the dot = "raw reachable"** (`probe_first_frame`
succeeds), regardless of kind — that's what gates Plot ROI / stitch / RSM.

**Advanced (raw read params)** — collapsed by default; only needed for **headerless raw
binaries** (`.raw`): `detector_shape` *or* a detector name (→ pyFAI factory), `raw_dtype`,
`raw_header_skip`, hot-pixel `threshold`. TIFF/EDF/CBF/Eiger-h5 auto-detect, so this stays
hidden for them. (These are stitch §5.4's "★ raw-image read params".) The **image-orientation
transform** (GAP E) belongs here too but is **stitch/RSM-only** (irrelevant to ROI box stats on
the raw array) — gate it off for the ROI consumer.

**Grouping** — range syntax `1-3, 5, 7-9` (`parse_scan_groups` → combine 1+2+3, then 5, then
7+8+9 via `CompositeFrameSource`, §2); **stitch/RSM only**, hidden for ROI (single scan). Emits
one source **per group**.

### 3.1 Mode/consumer gating (one widget, fields toggle)
Mirrors wrangler-org §2's mode table, but at the *field* level inside this widget:

| Field | ROI Scan Plot | Stitch / RSM |
|---|---|---|
| File + kind | ✓ | ✓ |
| Scan (single) | ✓ | ✓ (per group member) |
| Images + root | ✓ (optional → ROI on/off) | **required** |
| Advanced raw params | ✓ (no orientation) | ✓ (+ orientation) |
| Grouping range | hidden | ✓ |

The consumer constructs the widget with a `mode`/feature flags (`allow_grouping`,
`show_orientation`, `images_required`). Everything else is shared.

---

## 4. How each consumer uses it

**ROI Scan Plotter (now).** Replace the dialog's ad-hoc source row with an embedded
`ScanSourceWidget(mode="roi")`. On `sigSourceChanged(spec)`: `open_source(spec)` → build the
metadata table (existing `open_scan`/`_table_from_source` path) + run `probe_first_frame` →
enable Plot ROI iff raw reachable. **This is what unlocks ROI-on-SPEC** (the deferred piece):
the user points Images at the raw folder and Plot ROI lights up. The dialog's current
`Choose…`/scan-combo/`open_scan(uri, scan=)` code is the migration target — it already does a
subset.

**Stitch / RSM wrangler (later).** Drop the **same** widget into the wrangler-org §3.1 slot
with `mode="stitch"|"rsm"` (grouping + orientation on, images required). The wrangler adds the
geometry/calibration, DiffractometerConfig, UB, reduction, and output panels *around* it
(wrangler-org §3.2–3.6); the source half is done + tested from the ROI work.

---

## 5. Staged plan (each step independently testable)

0. **(done)** Headless `SpecSource` (metadata + optional images) + content detection + scan
   selection. *(Shipped `3a9d18c`.)*
1. **Headless grouping + composite + discovery.** `parse_scan_groups("1-3,5") → [[1,2,3],[5]]`;
   `CompositeFrameSource` (concatenate members → one frame stream, metadata dispatched per
   member); `discover_scans(dir, kind, recursive)` (generalize `TiffSeriesSource.from_directory`).
   *Gate:* pure unit tests — group parsing; a composite's frames/metadata equal the concatenation;
   `discover_scans` finds the scans in a synthetic folder; image_dir/options thread through.
2. **`ScanSourceWidget` — kind-adaptive, SPEC + NeXus.** The widget above with the §2.1
   kind-adaptive Scan selector (SPEC scan combo / NeXus entry combo) + Images affordance
   (external folder for SPEC; inline for raw NeXus/Eiger; `source_root` repoint for processed
   NeXus); emits `SourceSpec`; the raw-reachable **dot** via `probe_first_frame`; Advanced raw
   params. *Gate:* offscreen — each kind emits a spec with the right options; SPEC images folder
   flips the dot; a processed-NeXus + reachable raw shows the dot green; the scan/entry selector
   reloads. (Both source families already exist headless, so this is GUI-only.)
3. **Adopt in the ROI Scan Plotter.** Replace the ad-hoc source row; ROI works end-to-end on a
   SPEC scan (point at images → Plot ROI), a processed NeXus (linked raw), AND a **metadata-less
   image stack** (Eiger/raw burst → ROI vs frame_index). *Gate:* the existing `test_scan_plot_roi`
   SPEC/NeXus tests + "SPEC + image dir → ROI column fills" + **"images-only source → ROI plots
   vs frame_index with an empty metadata table"** (the §2.3 case).
4. **(later) Wrangler reuse.** Embed `mode="stitch"|"rsm"`; grouping + orientation on. *Gate:*
   wrangler-org's gates.
5. **(future) Tiled.** Add `TiledSource` + an `open_source` factory + a catalog-browser
   sub-widget; the widget gains a "Tiled" path with no change to the `sigSourceChanged` contract
   or any consumer. *Gate:* catalog browse → spec → `FrameSource` round-trip.

ROI gets the full SPEC + NeXus story at step 3; stitch/RSM inherit the source half for free at
step 4; Tiled slots in at step 5 behind the same seam.

---

## 6. Open questions for Vivek
1. **Image auto-derivation aggressiveness.** Auto-fill Images from the SPEC file's directory +
   default stem when a match is found (proposed), or always require an explicit folder pick?
2. **Raw read params per-source memory.** Persist the last `detector_shape`/`dtype`/`header_skip`
   per beamline/detector (a small preset), so the user sets them once? (Recommend yes — a
   `detector` preset dropdown that fills shape/dtype.)
3. **Where the widget lives.** `xdart/gui/.../scan_source_widget.py` (shared), imported by both
   the static-scan ROI dialog and the future wrangler — confirm the single-home placement.
4. **Composite / cross-file "Multi"** (stitch §6 Q1) — the `CompositeFrameSource` (§2) makes
   combining scans (even across different files/kinds) into one output first-class. v1 supports
   grouping within one source-entry; cross-file Multi (a group whose members come from different
   master files) is the same composite over specs from several widget instances — flag the UX
   (one widget vs a small list of source rows) for the implementing branch.
5. **NeXus "scan id".** A processed `.nxs` is usually one entry; multi-entry/multi-scan NeXus —
   treat entries as the scan-id space (so grouping a range of entries works like SPEC scan
   numbers), or one file = one scan? (Recommend: entries are the scan-id space, symmetric with
   SPEC, so grouping generalizes.)
6. **Tiled catalog UX.** When Tiled lands, is the catalog browser embedded in this widget or a
   separate dialog that returns a `(catalog, node)` spec? (Recommend a separate Browse… dialog
   feeding the same `sigSourceChanged`, to keep this widget compact.)

---

## 7. References
- Docs: `design_wrangler_organization_jun2026.md` §1–3.1 (the one-source-layer + the Source
  panel inventory this realizes); `design_stitching_jun2026.md` §5.4 (input list), §2.5 GAP E
  (image orientation); `design_scan_plotter_metadata_roi_jun2026.md` §2 (the source picker the
  ROI tool started from).
- Code: `xrd_tools/sources/registry.py` (`open_source`/`guess_source_kind`),
  `xrd_tools/sources/spec.py` (`SpecSource`), `xrd_tools/io/spec.py`
  (`is_spec_file`/`list_spec_scans`/`read_spec_scan_table`/`get_*`),
  `xrd_tools/io/image.py` (`find_image_files`/`read_image`),
  `xrd_tools/rsm/pipeline.py` (`ScanInfo(spec_path, img_dir)` — the pattern this generalizes),
  `xdart/.../scan_plot_dialog.py` (the ad-hoc picker to promote).
- Memory: `stitching_design_reframed`, `keep_xdart_thin`, `source_agnostic_ingestion`,
  `scan_taxonomy_gi_grid_policy`, `metadata_readers_record_all_motors`,
  `scan_plotter_metadata_roi_design`.
```
