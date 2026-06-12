# Processed Scan NeXus Schema v2

Architecture-v2 keeps the processed scan layout stable while adding explicit
schema stamps and lossless per-frame metadata.

The root entry group written by `xrd_tools.io.nexus` carries:

- `NX_class = "NXentry"`
- `ssrl_schema = "xrd_tools.processed_scan"`
- `ssrl_schema_version = 2`

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

Readers should use dataset units and descriptions where present. For integrated
data, axis units live on the `q` and `chi` datasets and `integrated_2d` may also
carry `two_d_kind` to distinguish standard `q/chi`, GI `qip/qoop`, GI
`qtotal/chigi`, and exit-angle maps.
