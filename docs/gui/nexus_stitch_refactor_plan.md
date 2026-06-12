# Nexus + Stitching Refactor — Branch Plan & TODO Spec

**Status:** Ready to start. This document is the complete specification; a fresh
conversation should be able to pick it up and execute without needing
clarification. If anything is ambiguous, prefer the "clean / final schema"
interpretation — backward compatibility with v1 xdart HDF5 files is
**explicitly out of scope**.

Target end state: xdart `0.36.0` + ssrl_xrd_tools `0.36.0`, both with a single
unified NeXus-compliant schema, flexible diffractometer geometry, and new
"Stitch 1D" / "Stitch 2D" processing modes that call pyFAI MultiGeometry.

---

## 0. Preconditions — Do Before Branching

These must be green before creating the nexus branches. Verify each:

1. `xdart/dev` has commit `973a4c1` (multi-frame NeXus perf + stop fix) pushed
   to origin.
2. `ssrl_xrd_tools/dev` has commit `8d0cfd0` (importlib.metadata version
   cherry-pick) pushed to origin.
3. `ssrl_xrd_tools/feat/phase-fitter-rewrite` has been merged into
   `ssrl_xrd_tools/dev`. The nexus reader work depends on the rewritten
   `BatchPhaseFitter`.
4. (Optional but recommended) xdart `0.35.2` and ssrl_xrd_tools `0.35.2` are
   tagged and published to PyPI so users have a stable v1-schema baseline to
   pin to before the schema break.

Verify:

```bash
cd ~/repos/xdart && git log --oneline origin/dev -3
cd ~/repos/ssrl_xrd_tools && git log --oneline origin/dev -5
cd ~/repos/ssrl_xrd_tools && git branch --list feat/phase-fitter-rewrite
```

---

## 1. Create the Branches

```bash
# xdart
cd ~/repos/xdart
git checkout dev && git pull
git checkout -b nexus
git push -u origin nexus

# ssrl_xrd_tools — only after phase-fitter-rewrite has merged into dev
cd ~/repos/ssrl_xrd_tools
git checkout dev && git pull
git checkout -b nexus
git push -u origin nexus
```

Both branches track `origin/nexus`. Do **not** branch off `main`; branch off
`dev` so provenance metadata picks up the current dev version strings.

Work order: land ssrl_xrd_tools reader first (it's testable against
hand-crafted HDF5 files), then xdart writer, then viewer migration, then a
coordinated 0.36.0 release.

---

## 2. The New NeXus Schema (No Backcompat)

Single file-format, produced by xdart's wrangler, consumed by ssrl_xrd_tools
and the xdart viewer. Follows NeXus NXroot / NXentry conventions where
practical but is pragmatic where NeXus doesn't fit well (e.g. stacked
integrated arrays instead of per-frame NXdata groups).

### 2.1 Top-level structure

```
<file>.nxs                                            (NXroot)
└── entry/                                            (NXentry, default)
    ├── @NX_class = "NXentry"
    ├── @default = "integrated_1d"
    ├── title                                         (str)
    ├── start_time / end_time                         (ISO 8601 str)
    ├── definition = "NXxdart" (or similar)           (str)
    │
    ├── integrated_1d/                                (NXdata, stacked)
    │   ├── @NX_class = "NXdata"
    │   ├── @signal = "intensity"
    │   ├── @axes = ["frame_index", "q"]
    │   ├── @frame_index_indices = 0
    │   ├── @q_indices = 1
    │   ├── intensity   shape (N, nq)   float32
    │   ├── sigma       shape (N, nq)   float32 (optional)
    │   ├── q           shape (nq,)     float32   @units = "1/angstrom"
    │   └── frame_index shape (N,)      int32
    │
    ├── integrated_2d/                                (NXdata, stacked)
    │   ├── @NX_class = "NXdata"
    │   ├── @signal = "intensity"
    │   ├── @axes = ["frame_index", "chi", "q"]
    │   ├── intensity shape (N, nchi, nq) float32
    │   ├── q         shape (nq,)         float32  @units = "1/angstrom"
    │   ├── chi       shape (nchi,)       float32  @units = "deg"
    │   └── frame_index shape (N,)        int32
    │
    ├── stitched_1d/           (NXdata, optional — only when Stitch 1D ran)
    │   ├── intensity shape (nq,)   float32
    │   ├── sigma     shape (nq,)   float32 (optional)
    │   └── q         shape (nq,)   float32  @units = "1/angstrom"
    │
    ├── stitched_2d/           (NXdata, optional — only when Stitch 2D ran)
    │   ├── intensity shape (nq, nchi)  float32    # xdart convention (rad, azim)
    │   ├── q         shape (nq,)       float32    @units = "1/angstrom"
    │   └── chi       shape (nchi,)     float32    @units = "deg"
    │
    ├── frames/                                       (NXcollection)
    │   ├── @NX_class = "NXcollection"
    │   └── frame_NNNN/                               (NXcollection, N entries)
    │       ├── thumbnail      shape (H, W) uint8 or uint16   # uncompressed
    │       ├── thumbnail_lut  (tiny vmin/vmax metadata)
    │       ├── map_raw        (2D reduced-res heatmap image, float32)
    │       ├── timestamp      (ISO 8601)
    │       └── source_ref/
    │           ├── path   (original raw file path, relative preferred)
    │           └── index  (frame index inside that file)
    │
    ├── instrument/                                   (NXinstrument)
    │   ├── @NX_class = "NXinstrument"
    │   ├── source/                                   (NXsource)
    │   │   ├── energy_keV
    │   │   ├── wavelength_A
    │   │   └── name (e.g. "SSRL BL 11-3")
    │   ├── detector/                                 (NXdetector)
    │   │   ├── name / model
    │   │   ├── pixel_size_x / pixel_size_y (m)
    │   │   ├── shape (H, W)
    │   │   ├── distance (m)    # = poni dist
    │   │   ├── mask (optional, uint8)
    │   │   └── positioners/                          (NXcollection)
    │   │       ├── @NX_class = "NXcollection"
    │   │       └── <motor_name>/                     (NXpositioner, N entries)
    │   │           ├── value  shape (N,)  float32
    │   │           └── @units (usually "deg")
    │   └── calibration/                              (NXcollection)
    │       ├── poni_json    (str, full pyFAI PONI serialized)
    │       ├── wavelength   (float, meters)
    │       └── dist / poni1 / poni2 / rot1 / rot2 / rot3 (scalars)
    │
    ├── sample/                                       (NXsample)
    │   ├── @NX_class = "NXsample"
    │   ├── name  (str, optional)
    │   └── positioners/                              (NXcollection)
    │       ├── @NX_class = "NXcollection"
    │       └── <motor_name>/                         (NXpositioner, N entries)
    │           ├── value  shape (N,)  float32
    │           └── @units
    │
    ├── per_frame_geometry/                           (NXdata)
    │   # Derived pyFAI geometry per frame, populated from raw positioners
    │   # via the DiffractometerGeometry config in /reduction/config/geometry.
    │   # Writer-derived; re-deriving it is a pure function of the config blob.
    │   ├── rot1           shape (N,) float32  @units = "rad"
    │   ├── rot2           shape (N,) float32  @units = "rad"
    │   ├── rot3           shape (N,) float32  @units = "rad"
    │   ├── incident_angle shape (N,) float32  @units = "deg"
    │   └── frame_index    shape (N,) int32
    │
    └── reduction/                                    (NXprocess)
        ├── @NX_class = "NXprocess"
        ├── program = "xdart"
        ├── version = <xdart.__version__>
        ├── date    = ISO 8601 timestamp of reduction start
        ├── versions/                                 (NXcollection)
        │   ├── xdart           (str)
        │   ├── ssrl_xrd_tools  (str)
        │   ├── pyFAI           (str)
        │   ├── h5py            (str)
        │   ├── numpy           (str)
        │   ├── pymatgen        (str)
        │   └── python          (str)
        ├── host (optional)
        ├── cli_args / gui_state (optional JSON)
        ├── config/
        │   ├── bai_1d_args  (JSON str: npt, unit, method, mask_path, …)
        │   ├── bai_2d_args  (JSON str: npt_rad, npt_azim, unit, method, …)
        │   ├── mg_1d_args   (JSON str, optional — Stitch 1D params)
        │   ├── mg_2d_args   (JSON str, optional — Stitch 2D params)
        │   ├── gi_config    (JSON str: flavor, incidence angle source, …)
        │   └── geometry/                             (NXcollection)
        │       # Full DiffractometerGeometry config — see §3
        │       ├── convention   = "psic" | "two_circle" | "custom"
        │       ├── mapping_json = (str) JSON blob of AngleMappings
        │       └── motor_sources = (str) JSON dict mapping logical
        │                                 motor → meta-file column name
        └── inputs/
            ├── raw_files (N,) str       (original per-frame source paths)
            └── meta_file str            (original SPEC / tsv / csv file path)
```

### 2.2 Key schema rules

- **Stacked 1D / 2D are the single source of truth** for integrated data.
  `frames/frame_NNNN/` holds *only* per-frame non-array metadata + thumbnail.
  Do not duplicate integrated arrays into per-frame groups.
- **Thumbnails are uncompressed uint8 or uint16**; gzip on thumbnails was a
  measured hot spot in the v1 pipeline. The encoding rule:
  compute vmin/vmax across frame once, then linearly quantize. Store
  `thumbnail_lut = (vmin, vmax, dtype)` so viewers can invert.
- **Stacked 1D/2D writes one h5py slice assignment per batch flush** — no
  per-frame resize-append. Pre-allocate with `shape=(N, nq)` at scan start
  when N is known; otherwise allocate with `maxshape=(None, nq)` and resize
  in power-of-two chunks.
- **Raw motor positioners live in `instrument/detector/positioners/` and
  `sample/positioners/`**, split by physical host (sample motors vs
  detector-arm motors). This is where the user's meta-file columns land
  verbatim.
- **Derived pyFAI rotations live in `per_frame_geometry/`** and are recomputed
  from the raw positioners + `/reduction/config/geometry` mapping. If a
  downstream tool doesn't trust the writer's derivation, it can always
  re-derive.
- **NXprocess/versions is populated with `importlib.metadata.version(...)`**
  and must not hard-code any string. This is the *reason* the
  `__version__ = importlib.metadata.version(...)` pattern exists on both
  repos; keep it that way.

### 2.3 What gets deleted

Remove these from the xdart v2 (nexus) codebase outright:

- `_write_scan_data_nexus` in `modules/ewald/sphere.py` (old non-stacked
  writer).
- `save_bai_1d`, `save_bai_2d` on `EwaldArch` — merged into the stacked
  writer.
- The `batch_save=True` flag on `EwaldSphere.add_arch` — batch is the
  *only* save path now, so the flag is implicit.
- `flush_batch_state` in its current form — replaced by `_save_to_nexus(h5f)`
  which does everything in one pass.
- Any migration shims or v1-schema reader in xdart — if someone needs to read
  a v1 file, they pin to xdart ≤ 0.35.x.
- Old per-frame `scan_data/` groups replication.

---

## 3. Flexible Diffractometer Geometry

The one big design change beyond "write a cleaner file": xdart must stop
hard-coding the GI-only `th_motor` incidence-angle selector and instead support
arbitrary diffractometer geometries with arbitrary motor naming conventions.

### 3.1 Problem recap

Three real cases we must support:

| Convention | Sample motors | Detector motors | Incidence angle source |
|---|---|---|---|
| 2-circle  | `th`, (`gonchi`) | `tth` | `th` |
| psic 6-circle | `eta`, `chi`, `phi`, `mu` | `del`, `nu` | `eta` |
| psic-halpha | `halpha`, `chi`, `phi`, `mu` | `del`, `nu` | `halpha` |

Per-image pyFAI per-arch rotations required for MultiGeometry:

- **2-circle**: `rot1` or `rot2` gets the per-image `tth`/`del`. Only one
  non-zero.
- **psic**: `rot2 = del`, `rot1 = nu` (both per-image).

The mapping from logical motor name (what the user typed in the SPEC file)
to pyFAI rotation (`rot1`/`rot2`/`rot3`) must be configurable and stored
in the NeXus file.

### 3.2 New module: `ssrl_xrd_tools/core/geometry.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

@dataclass(frozen=True)
class AngleMapping:
    """How a logical pyFAI angle is derived from a raw motor column.

    derived_angle_rad = sign * deg2rad(motor_value_deg) + offset_rad
    """
    source_motor: str      # e.g. "del", "nu", "tth", or "" for unused
    sign: float = 1.0      # +1 or -1
    offset_rad: float = 0.0

    @property
    def is_active(self) -> bool:
        return bool(self.source_motor)


@dataclass(frozen=True)
class DiffractometerGeometry:
    """
    Maps raw scan-file motor columns to pyFAI geometry + GI incidence angle.

    Convention presets:

    - "two_circle":   rot1 <- tth,           rot2 unused,     incidence <- th
    - "psic":         rot1 <- nu,            rot2 <- del,     incidence <- eta
    - "custom":       user-supplied AngleMappings
    """
    convention: Literal["two_circle", "psic", "custom"] = "two_circle"
    rot1: AngleMapping = field(default_factory=AngleMapping)
    rot2: AngleMapping = field(default_factory=AngleMapping)
    rot3: AngleMapping = field(default_factory=AngleMapping)
    incident_angle: AngleMapping = field(default_factory=AngleMapping)

    # All logical sample motors to persist into /entry/sample/positioners/
    sample_motors: tuple[str, ...] = ()
    # All logical detector motors to persist into
    # /entry/instrument/detector/positioners/
    detector_motors: tuple[str, ...] = ()

    @classmethod
    def two_circle(cls, tth: str = "tth", th: str = "th",
                   gonchi: str | None = None) -> "DiffractometerGeometry": ...

    @classmethod
    def psic(cls, del_: str = "del", nu: str = "nu",
             eta: str = "eta", chi: str = "chi",
             phi: str = "phi", mu: str = "mu") -> "DiffractometerGeometry": ...

    @classmethod
    def psic_halpha(cls, del_: str = "del", nu: str = "nu",
                    halpha: str = "halpha", chi: str = "chi",
                    phi: str = "phi", mu: str = "mu") -> "DiffractometerGeometry":
        # same as psic but incidence_angle.source_motor = "halpha"
        ...

    def derive_per_frame(self, motors: dict[str, np.ndarray]
                         ) -> dict[str, np.ndarray]:
        """Return {'rot1','rot2','rot3','incident_angle'} arrays shape (N,).

        Applies sign + offset + deg2rad (except incident_angle, which stays
        in degrees since that's what downstream GI code expects).
        """

    def to_json(self) -> str: ...
    @classmethod
    def from_json(cls, s: str) -> "DiffractometerGeometry": ...
```

This lives in ssrl_xrd_tools (not xdart) because both the xdart writer and
downstream analysis tools need it. Keep it dataclass-clean and serializable.

### 3.3 Storage in NeXus

The `DiffractometerGeometry` instance round-trips through
`/entry/reduction/config/geometry/`:

- `convention` — str attribute
- `mapping_json` — the full dataclass as JSON
- `motor_sources` — a flat dict mapping each logical motor to the column name
  that actually appeared in the meta file (they may differ if the user
  aliases them in the GUI)

The derived per-frame arrays (`rot1`, `rot2`, `rot3`, `incident_angle`) are
computed once at write time and stored in `/entry/per_frame_geometry/`.

---

## 4. xdart Side — Writer, Wrangler, UI

### 4.1 Writer (`xdart/modules/ewald/sphere.py`)

Single method replaces all prior save paths:

```python
def _save_to_nexus(self, h5f: h5py.File) -> None:
    """Write the entire NXentry in one coherent pass.

    Called once from the wrangler after all frames have been added
    (or on periodic flush / final stop). Uses accumulated in-memory
    state — no per-frame HDF5 rewrites.
    """
```

Responsibilities:

1. Create / reopen `entry/` group with all NX_class attrs.
2. Pre-allocate or resize stacked datasets `integrated_1d/intensity`,
   `integrated_2d/intensity`, plus their axes.
3. Slice-assign the current batch into the preallocated datasets.
4. Write per-frame `frames/frame_NNNN/{thumbnail, map_raw, timestamp,
   source_ref}` for new frames only.
5. Write / update `instrument/`, `sample/` positioner arrays (append the
   new batch slice).
6. Write / update `per_frame_geometry/` arrays (append).
7. On first call only, write `reduction/` NXprocess with
   `importlib.metadata` versions + `config/` blobs.
8. On final call (end-of-scan), optionally write `stitched_1d/` /
   `stitched_2d/` if stitching ran.

Concurrency: one h5py handle, held by the wrangler thread. Lock around
open/close boundaries the same way v1 did.

### 4.2 EwaldArch changes

- `make_thumbnail()` already exists — keep.
- Replace compressed gzip writes with uint8/uint16 quantized writes. Keep the
  LUT as a small attribute dict on the frame group.
- Drop `save_bai_1d/2d` entirely; these are only used to push into the
  stacked array now, and that push happens in `_save_to_nexus`.

### 4.3 Wrangler thread
(`xdart/gui/tabs/static_scan/wranglers/spec_wrangler_thread.py`)

Changes:

1. On scan start, construct the `DiffractometerGeometry` from the GUI
   convention + motor-name dropdowns.
2. Resolve motor columns from the meta file once; cache N-length arrays for
   every logical motor in `self._motor_arrays`.
3. Call `self._sphere.geometry = diff_geom` so per-frame rotations are
   derived cleanly at write time.
4. Add a **batch-only processing mode switch** that disables per-frame save
   in favor of end-of-scan `_save_to_nexus`. (Actually, all modes go through
   `_save_to_nexus`; the difference is *when* the stacked slice-assign
   happens.)
5. Add a **Stitch 1D / Stitch 2D** execution path — see §4.5.
6. Keep the prefetcher / bulk-read path from v1 (already shipped in
   `973a4c1`). That is NOT a nexus-branch change; it just continues to work.

### 4.4 UI — Wrangler widget (`spec_wrangler.py`)

Dropdown currently contains `Int 1D`, `Int 2D`, plus live-view. Extend to:

```
Int 1D
Int 2D
Int 1D+2D       (if not already)
Stitch 1D       (new, batch-only)
Stitch 2D       (new, batch-only)
```

When **Stitch 1D** or **Stitch 2D** is selected:

- Auto-lock "batch mode" = True, grey out the per-frame / live option.
- Reveal a **Geometry panel** (collapsible, below the mode selector) with:
  - Convention dropdown: `two_circle | psic | psic_halpha | custom`
  - Motor-name text boxes populated from convention preset but editable:
    - For `two_circle`: `tth`, `th`, optional `gonchi`
    - For `psic`: `del`, `nu`, `eta`, `chi`, `phi`, `mu`
    - For `psic_halpha`: same as psic but `halpha` replaces `eta` label
    - For `custom`: empty, user fills in each `rot1 / rot2 / rot3 /
      incidence_angle` source + sign + offset directly.
  - Validate at panel-close time that each named column exists in the
    loaded meta file; if not, flag it red.

When **any GI mode** is active (1D or 2D, not just stitch):

- The existing `self.th_motor` dropdown becomes **incidence-angle motor
  dropdown** (rename label too). Populated by default with whatever
  `DiffractometerGeometry.incident_angle.source_motor` resolves to. Options
  are every column in the meta file. Typical choices: `th` (2-circle),
  `eta` (psic), `halpha` (psic-halpha).
- `gi_config` in `reduction/config/` records the choice.

Existing `self.th_motor` attribute:

- Rename internally to `self.incidence_motor` (alias `th_motor` for backward
  compat inside a single release). Update every reference.
- The value stored is just the logical motor name string; the actual
  per-frame incidence angles come from the `DiffractometerGeometry` derivation.

### 4.5 Stitching execution path

New method on the wrangler (or a helper module):

```python
def _run_stitch(self, mode: Literal["1d", "2d"]) -> None:
    """Batch-only stitch using pyFAI MultiGeometry via ssrl_xrd_tools.

    1. Ensure all frames have been loaded (blocking; not live).
    2. Pull base_poni, per-frame images, per-frame masks, per-frame
       normalization (i0/i1 columns if configured).
    3. Use self._sphere.geometry (DiffractometerGeometry) to derive
       rot1/rot2/rot3 arrays.
    4. Call ssrl_xrd_tools.integrate.multi.create_multigeometry_integrators
       with rot1_angles (degrees — the raw motor values) and optionally
       rot2_angles.
    5. Call stitch_1d(...) or stitch_2d(...) with the image stack.
    6. Store the result into self._sphere.stitched_1d / stitched_2d.
    7. Trigger _save_to_nexus — writes integrated_1d/2d (per-image) AND
       stitched_1d/2d (single merged pattern).
    """
```

Key points:

- Stitch mode still produces per-image `integrated_1d`/`integrated_2d` (so the
  viewer can still show per-image patterns) **in addition** to the merged
  `stitched_1d`/`stitched_2d`. This gives the user both a per-image QA view
  and the final merged pattern.
- Per-image normalization: if the user's geometry config names an i0/i1
  column, divide each image by i1 before passing to stitch_1d (per the
  notebook reference).
- Radial range: determined from per-image geometry + image size. Default:
  let pyFAI auto-size it; expose override in `mg_1d_args` / `mg_2d_args`.
- Method: default `BBox` (matches the notebook). Expose in config.
- Masks: `mask=global_mask` (one per detector, not per image). Per-image
  masks are possible but not worth the UI complexity in v1 of stitch.

### 4.6 Thumbnails encoding change

In `EwaldArch.make_thumbnail`, after computing the downsampled float array:

```python
# Measured at session start across full scan (once), fall back to per-frame:
if global_vmin is None:
    vmin, vmax = np.percentile(downsampled, [1, 99])
else:
    vmin, vmax = global_vmin, global_vmax
thumb = np.clip((downsampled - vmin) / max(vmax - vmin, 1e-12), 0, 1)
# Quantize — default uint8, use uint16 when scan requests higher dynamic range:
thumb_q = (thumb * 255).astype(np.uint8)   # or 65535 / uint16
```

HDF5: store uncompressed. No gzip filter. Store `(vmin, vmax, dtype)` as
attrs on the frame's `thumbnail` dataset.

---

## 5. ssrl_xrd_tools Side — Reader, Provenance, Viz

### 5.1 New module: `ssrl_xrd_tools/io/nexus.py`

```python
import xarray as xr

def read_sphere(path: str, *, groups: tuple[str, ...] = ("1d", "2d")
                ) -> xr.Dataset:
    """Read an xdart nexus file into a single coherent xarray.Dataset.

    Dimensions:
      - frame: N
      - q: nq
      - chi: nchi (if 2d loaded)

    Data variables:
      - intensity_1d  dims (frame, q)
      - sigma_1d      dims (frame, q)         (if present)
      - intensity_2d  dims (frame, chi, q)    (if 2d loaded)
      - thumbnail     dims (frame, thumb_y, thumb_x)  (optional)
      - rot1, rot2, rot3, incident_angle    dims (frame,)
      - <each sample motor>                 dims (frame,)
      - <each detector motor>               dims (frame,)

    Coords:
      - frame (frame_index)
      - q (1/angstrom)
      - chi (deg)
      - reduction metadata attrs on the Dataset

    Always lazy where practical (use h5py + xarray.open_dataset with the
    h5netcdf engine when possible); fall back to eager loads for small
    arrays.
    """

def read_stitched(path: str) -> xr.Dataset:
    """Read stitched_1d / stitched_2d if present. Raises if absent."""

def write_stitched(path: str, stitched: xr.Dataset) -> None:
    """Append a stitched result to an existing file (rare — usually xdart
    writes this, but handy for ad-hoc re-stitches from notebooks)."""
```

Rules:

- Return xarray.Dataset, never raw h5py handles.
- Coordinates use the q/chi arrays actually stored in the file.
- Attach `ds.attrs['reduction']` = provenance dict (see §5.2).
- Keep the `h5py.File` handle open only inside the function scope.
- This replaces every per-frame h5py call in `BatchPhaseFitter`.

### 5.2 New module: `ssrl_xrd_tools/core/provenance.py`

```python
def read_provenance(path: str) -> dict:
    """Return flat dict: versions, config JSONs, host, date, inputs."""

def write_provenance(h5f: h5py.File, *,
                     xdart_version: str,
                     extra: dict | None = None) -> None:
    """Used by xdart at write time. Populates /entry/reduction/ with
    importlib.metadata versions for all known deps."""
```

Canonical version targets to record: `xdart`, `ssrl_xrd_tools`, `pyFAI`,
`h5py`, `numpy`, `pymatgen`, `python`.

### 5.3 BatchPhaseFitter update

Currently per-frame h5py loops are the bottleneck. Replace with:

```python
from ssrl_xrd_tools.io.nexus import read_sphere

def fit_scan(path: str, phases: list[Phase], ...) -> xr.Dataset:
    ds = read_sphere(path, groups=("1d",))
    # Now vectorized np operations over ds.intensity_1d.values
    # No more per-frame h5py.open/close round trips
```

Goal: fitting a 1000-frame scan should be dominated by the fit arithmetic,
not by I/O.

### 5.4 viz helpers

`ssrl_xrd_tools/viz/mpl.py` (and any plotly versions) must accept
`xarray.Dataset` (or `xarray.DataArray` for single-panel) as the canonical
input. Remove or deprecate any viz function that takes raw numpy + metadata
tuples. Keeps the API coherent with the reader.

### 5.5 core/geometry.py

See §3.2. This module is **new** on the nexus branch and is the authoritative
home for the `DiffractometerGeometry` dataclass.

### 5.6 tests

Add `tests/test_nexus_io.py`:

- Hand-craft a small NeXus file in the fixture (pure h5py, no xdart dep)
  that matches the v2 schema. Include a 5-frame 1D + 2D stack, one
  stitched_1d, psic geometry config.
- Assert `read_sphere` returns a Dataset with the expected dims, coords,
  motor variables, and that `read_provenance` parses back the same dict
  that was written.
- Run a BatchPhaseFitter fit on the fixture end-to-end and assert the
  returned Dataset shape / coordinate names.

---

## 6. Version Bumps & Release

### 6.1 xdart

- `pyproject.toml` → `version = "0.36.0"`
- Tag `v0.36.0` on merge to `main`.
- CHANGELOG/README entry calls out:
  - Schema break — v1 files not readable
  - New stitching modes
  - Flexible diffractometer geometry

### 6.2 ssrl_xrd_tools

- `pyproject.toml` → `version = "0.36.0"`
- Tag `v0.36.0` on merge to `main`, **coordinated with xdart** so a single
  PyPI drop for users.
- CHANGELOG/README entry calls out:
  - New `io.nexus.read_sphere` reader
  - New `core.geometry.DiffractometerGeometry`
  - New `core.provenance`
  - BatchPhaseFitter now Dataset-based

### 6.3 PyPI publishing order

1. `ssrl_xrd_tools==0.36.0` first (xdart's install pulls from PyPI if not
   editable).
2. `xdart==0.36.0` second.

### 6.4 Install provenance check

After publishing, spot-check the curl install produces:

```python
import xdart, ssrl_xrd_tools
print(xdart.__version__, ssrl_xrd_tools.__version__)
# 0.36.0 0.36.0
```

---

## 7. Execution Sequence

Rough but concrete ordering. Each step should be one or two commits max to
keep bisect sane.

1. **ssrl_xrd_tools/nexus**: create `core/geometry.py`, tests pass.
2. **ssrl_xrd_tools/nexus**: create `core/provenance.py`, tests pass.
3. **ssrl_xrd_tools/nexus**: create `io/nexus.py` reader against a hand-crafted
   HDF5 fixture, tests pass.
4. **ssrl_xrd_tools/nexus**: port `BatchPhaseFitter` to use `read_sphere`;
   regression-test against captured-golden fit results on a known scan.
5. **ssrl_xrd_tools/nexus**: update viz helpers to accept Dataset; bump
   version to `0.36.0-dev0`.
6. **xdart/nexus**: add `_save_to_nexus` alongside v1 writer (branch only,
   behind a feature flag for step 6 only).
7. **xdart/nexus**: delete v1 writers, v1 migration shims, `batch_save`
   flag; `_save_to_nexus` is the only path.
8. **xdart/nexus**: stacked preallocation + slice-assign batches.
9. **xdart/nexus**: uint8/uint16 uncompressed thumbnails.
10. **xdart/nexus**: wire `DiffractometerGeometry` through wrangler; rename
    `th_motor` → `incidence_motor`; generalize GI dropdown.
11. **xdart/nexus**: Stitch 1D / Stitch 2D processing modes + geometry UI
    panel.
12. **xdart/nexus**: viewer migration — read v2 schema only. Delete v1
    viewer code paths.
13. **xdart/nexus**: bump version to `0.36.0-dev0`; end-to-end test on a real
    captured scan (1000-frame psic stitch).
14. **ssrl_xrd_tools**: merge `nexus` → `dev` → `main`, tag `v0.36.0`,
    publish to PyPI.
15. **xdart**: merge `nexus` → `dev` → `main`, tag `v0.36.0`, publish to
    PyPI.

---

## 8. Explicit Non-Goals

- No v1-schema reader in xdart/nexus. Users on v1 pin to xdart≤0.35.x.
- No migration tool from v1 → v2 files. Re-reduce if needed.
- No runtime schema versioning (no `schema_version` attr) — if a future
  break happens, it'll be another minor bump with a clear break.
- No per-image mask stacking in Stitch v1 — one global mask.
- No stitch modes for live/append — stitch is batch-only by definition.
- No changes to the calibration / PONI input workflow — still
  pyFAI-compatible, still single PONI per scan.

---

## 9. Open Questions (resolve before step 6)

1. **Sample motor column list default** for `psic`: the dataclass factory
   pre-fills `chi, phi, mu` — is that what 17-2 actually records? Confirm
   from a recent SPEC log.
2. **Units on `incident_angle`**: degrees throughout xdart/GI code today.
   Keeping that. Will not normalize to radians at the NeXus layer even
   though NeXus prefers SI.
3. **Thumbnail global vmin/vmax**: computed on first ~32 frames then locked
   for the rest of the scan? Or recomputed per frame? Leaning "locked" for
   stability of the viewer's color scale. Make it a config option.
4. **Stitch output pixel size**: default `npt = 2000` for stitched_1d,
   `npt_rad = 1500, npt_azim = 720` for stitched_2d. Revisit if it's too
   coarse for high-Q.

---

## 10. Reference Material

- Current xdart dev HEAD: `973a4c1` (perf + stop fix)
- Current ssrl_xrd_tools dev HEAD: `8d0cfd0` (version cherry-pick)
- Stitching reference notebook:
  `~/repos/example_notebooks/reduce_pyFAI_multigeometry.ipynb`
- Existing stitching implementation:
  `ssrl_xrd_tools/integrate/multi.py` — exports
  `create_multigeometry_integrators`, `stitch_1d`, `stitch_2d`. Already
  handles both 2-circle and psic cases via the optional `rot2_angles` arg.
- NeXus spec for reference:
  - NXentry: https://manual.nexusformat.org/classes/base_classes/NXentry.html
  - NXprocess: https://manual.nexusformat.org/classes/base_classes/NXprocess.html
  - NXpositioner: https://manual.nexusformat.org/classes/base_classes/NXpositioner.html

---

## 11. Final Push / Release Checklist

Before tagging 0.36.0 on either repo:

- [ ] Both nexus branches rebased cleanly on current dev
- [ ] All tests in both repos green
- [ ] End-to-end: run a real 1000-frame psic scan end-to-end; confirm
      stitched_1d matches the notebook-produced reference within numerical
      tolerance
- [ ] `importlib.metadata.version("xdart")` returns `0.36.0` in both a
      `pip install .` and an editable install
- [ ] `importlib.metadata.version("ssrl_xrd_tools")` returns `0.36.0` same
- [ ] `/entry/reduction/versions/` in a freshly written file matches the
      installed package versions
- [ ] PyPI upload dry-run (`twine check dist/*`) passes on both repos
- [ ] README install instructions reference `>=0.36.0` (or unpinned —
      consistent with current style)

---

*Written 2026-04-13. If months have passed, re-verify git HEADs and PyPI
state before starting — the preconditions in §0 may have shifted.*
