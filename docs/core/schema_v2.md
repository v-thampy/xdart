# Processed Scan NeXus Schema (v3, with v2 reading)

> **Source of truth:** `src/xrd_tools/io/schema.py` (schema-as-code: the
> `SCHEMA` object — attribute keys, row-aligned dataset sets, axis names,
> capability attrs), pinned by `tests/core/test_schema_as_code.py` and the
> byte-compat gate `tests/core/test_v2_record_compat.py`.  Versioning
> policy: `docs/decisions/0002-schema-version-and-capability-attrs.md`.
> This page is a prose overview only.

Version 3 uses neutral integrated axis dataset names. Version 2 files with the
same `xrd_tools.processed_scan` identity and a valid `q`/`chi` layout remain
readable without modification. This does not admit unstamped or arbitrary
historical files. Unknown versions and mismatched layouts are refused.

New writes use v3. Append requires v3; use a new output for a v2 predecessor.
Ordinary Replace may replace a positively recognized v2 result with a new v3
result. Immutable Reintegration converts only its private output copy, including
the preserved dimension and named GI modes; it never migrates the original.
The older in-place Reintegration API cannot write v2 files.

The root entry group written by `xrd_tools.io.nexus` carries:

- `NX_class = "NXentry"`
- `ssrl_schema = "xrd_tools.processed_scan"`
- `ssrl_schema_version = 3`

The main processed groups remain:

- `/entry/integrated_1d`
- `/entry/integrated_2d`
- `/entry/frames`
- `/entry/per_frame_geometry`
- `/entry/scan_data`
- `/entry/reduction`

`/entry/scan_data` is an `NXcollection` indexed by `frame_index`. Each metadata
column is one appendable dataset. Numeric columns are stored as `float32`.
Non-numeric columns are stored as UTF-8 variable-length string datasets with:

- `ssrl_dtype = "string"`
- `encoding = "utf-8"`
- `missing_value = ""`
- `description = "Per-frame scan metadata column"`

Integrated coordinates use `axis_x` (1-D) and `axis_x`, `axis_y` (2-D), including
named GI mode groups. Their NXdata `axes` attributes are respectively
`[frame_index, axis_x]` and `[frame_index, axis_y, axis_x]`. Intensity orientation
is unchanged: `(frame, y, x)`. Existing Python `q`/`chi` result fields and xarray
coordinate names remain unchanged; these changes concern on-disk names only.

Readers should use dataset units and descriptions where present. Integrated
axes carry the existing scientific unit tokens (for example `qip_A^-1`) and
`long_name` labels including human-readable units (for example `Q_ip (Å⁻¹)`).
Version 2 stores the same coordinates under `q`/`chi`. `integrated_2d` may also
carry `two_d_kind` to distinguish standard `q/chi`, GI `qip/qoop`, GI
`qtotal/chigi`, and exit-angle maps.

Stitched analysis groups and RSM retain their separate existing layouts; this
version change does not rename H/K/L or Cartesian-Q coordinates.
