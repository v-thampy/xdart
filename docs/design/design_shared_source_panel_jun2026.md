# Design: the shared scan-source panel (`ScanSourceWidget`)

**Status:** draft for discussion · 2026-06-23 · planning only (no code)
**Realizes:** [`design_wrangler_organization_jun2026.md`](design_wrangler_organization_jun2026.md)
§3.1 ("Source / data") as a concrete, reusable Qt widget — and
[`design_stitching_jun2026.md`](design_stitching_jun2026.md) §5.4's source/data input
inventory.
**Driven by:** SPEC is the **primary** scan-definition source today (Vivek, 2026-06-23) —
extensionless, scan-number-addressed, with image files paired by directory + filename root.
The ROI Scan Plotter needs it now; stitch/RSM need the **same** definition. Build the source
panel **once**, shared.
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

**One small headless addition** (pure, testable, no Qt): a grouping helper for stitch/RSM —

```python
# xrd_tools/sources/grouping.py
def parse_scan_range("1-3, 5, 7-9") -> [1, 2, 3, 5, 7, 9]      # pure
def spec_scan_specs(spec_uri, scans, *, image_dir=None, **read_kw) -> list[SourceSpec]
```

ROI uses a single scan; stitch/RSM pass a range → a list of `SourceSpec`s → one
`run_stitch`/`run_rsm` per group. Keeping range-parsing headless means the widget's grouping
field is trivial.

---

## 3. The widget — fields + behavior

`ScanSourceWidget(QWidget)` with one signal: **`sigSourceChanged(SourceSpec | None)`** (emitted
when a complete, openable source is defined, or `None` when cleared/invalid). Consumers connect
to it and call `open_source(spec)` themselves (so the widget never holds analysis state).

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

**File** — one picker, "All files" (SPEC is extensionless). On pick, `guess_source_kind`
classifies (content-sniff for SPEC); the `kind:` label reflects it. NeXus/Eiger/TIFF-seq paths
are unchanged.

**Scan** — for a SPEC file, populated from `SpecSource(uri).available_scans`; shown only when
>1 scan. Hidden for single-scan / non-SPEC sources. (Already prototyped in the ROI dialog.)

**Images** — the directory of detector files. **Auto-default:** when blank, try the SPEC
file's own directory with the default stem; if `find_image_files` hits, pre-fill + show the
**raw-reachable dot green**; else grey (metadata only). The user can point elsewhere. Omitting
it entirely ⇒ metadata-only source ⇒ the consumer disables raw features (ROI/stitch/RSM).
**Filename root** override lives next to it (default `{spec}_scan{N}_`).

**Advanced (raw read params)** — collapsed by default; only needed for **headerless raw
binaries** (`.raw`): `detector_shape` *or* a detector name (→ pyFAI factory), `raw_dtype`,
`raw_header_skip`, hot-pixel `threshold`. TIFF/EDF/CBF/Eiger-h5 auto-detect, so this stays
hidden for them. (These are stitch §5.4's "★ raw-image read params".) The **image-orientation
transform** (GAP E) belongs here too but is **stitch/RSM-only** (irrelevant to ROI box stats on
the raw array) — gate it off for the ROI consumer.

**Grouping** — range syntax `1-3, 5, 7-9` (`parse_scan_range`); **stitch/RSM only**, hidden for
ROI (single scan). Emits a list of specs instead of one.

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
1. **Headless grouping helper.** `parse_scan_range` + `spec_scan_specs`. *Gate:* pure unit
   tests (`"1-3,5"` → specs; image_dir threads into each spec's options).
2. **`ScanSourceWidget`.** The widget above, `mode="roi"` first; emits `SourceSpec`; auto-derive
   image dir/stem + the raw-reachable dot; Advanced raw params. *Gate:* offscreen — picking a
   SPEC file emits a spec with the right options; pointing Images at a raw folder flips the
   reachable dot; the scan combo reloads.
3. **Adopt in the ROI Scan Plotter.** Replace the ad-hoc source row; ROI-on-SPEC works
   end-to-end (point at images → Plot ROI). *Gate:* the existing `test_scan_plot_roi` SPEC
   tests + a new "SPEC + image dir → ROI column fills" end-to-end.
4. **(later) Wrangler reuse.** Embed `mode="stitch"|"rsm"`; grouping + orientation on. *Gate:*
   wrangler-org's gates.

ROI gets the full SPEC story at step 3; stitch/RSM inherit the source half for free at step 4.

---

## 6. Open questions for Vivek
1. **Image auto-derivation aggressiveness.** Auto-fill Images from the SPEC file's directory +
   default stem when a match is found (proposed), or always require an explicit folder pick?
2. **Raw read params per-source memory.** Persist the last `detector_shape`/`dtype`/`header_skip`
   per beamline/detector (a small preset), so the user sets them once? (Recommend yes — a
   `detector` preset dropdown that fills shape/dtype.)
3. **Where the widget lives.** `xdart/gui/.../scan_source_widget.py` (shared), imported by both
   the static-scan ROI dialog and the future wrangler — confirm the single-home placement.
4. **Composite / cross-file sources** (stitch "Multi", stitching §6 Q1) — out of scope for this
   widget v1 (one master file per definition); a later composite layer concatenates specs.

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
