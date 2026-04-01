# xdart + ssrl_xrd_tools Refactoring Plan

## Goal

Make xdart a **thin desktop GUI shell** that imports all scientific functionality from `ssrl_xrd_tools`. Separately, build **Jupyter notebook widgets** in `ssrl_xrd_tools/gui/` for advanced scripting users.

## Current State (March 2026)

xdart already imports from ssrl_xrd_tools in several key places:

| xdart module | ssrl_xrd_tools import | Status |
|---|---|---|
| `EwaldArch` | `integrate.single`, `integrate.gid`, `integrate.calibration` | Working |
| `EwaldSphere` | `integrate.multi` (stitch_1d/2d) | Working |
| `_utils.py` | `core.hdf5` (entire codec) | Working |
| `spec_wrangler` | `io.image`, `io.export`, `io.metadata`, `integrate.gid` | Working |

What **hasn't** migrated yet:

- **Data containers**: xdart's `PONI`, `int_1d_data_static`, `int_2d_data_static` are still xdart-local (ssrl_xrd_tools equivalents exist but aren't used)
- **EwaldArch/EwaldSphere**: Still live in xdart as the bridge between GUI and ssrl_xrd_tools functions
- **GUI-specific utils**: session.py, h5pool.py, pyFAI_binaries.py, pgOverrides — these stay in xdart
- **lmfit_models**: Duplicated (xdart has its own copy; ssrl_xrd_tools has `analysis.fitting.models`)

---

## Phase 1: Unify Data Containers

**Goal**: Eliminate duplicate container classes. xdart uses ssrl_xrd_tools containers directly.

### 1a. Align PONI container

xdart's `utils/containers/poni.py::PONI` vs ssrl_xrd_tools' `core/containers.py::PONI`.

**Current difference**: xdart's PONI stores `detector` as a pyFAI Detector object; ssrl_xrd_tools stores it as a string. xdart PONI has `from_yaml`, `from_ponifile`, `to_dict`/`from_dict` methods.

**Action**:
1. Enhance ssrl_xrd_tools `PONI` to support the I/O methods xdart needs (`from_ponifile`, `to_dict`/`from_dict`)
2. Keep `detector` as string in ssrl_xrd_tools (resolve to pyFAI object in `integrate/calibration.py` where needed)
3. Replace all `from xdart.utils.containers import PONI` with `from ssrl_xrd_tools.core.containers import PONI`
4. Add a thin compatibility shim in xdart if needed during transition

### 1b. Align integration result containers

xdart's `int_1d_data_static` / `int_2d_data_static` vs ssrl_xrd_tools' `IntegrationResult1D` / `IntegrationResult2D`.

**Current differences**:
- xdart containers dual-store both 2-theta AND Q arrays, with interpolation for conversion
- xdart containers have `__add__`/`__sub__` for summing results
- xdart containers have HDF5 serialization (`to_hdf5`/`from_hdf5`)
- xdart containers handle GI-specific fields (`i_qxy`, `qxy`, `i_qz`, `qz`, `i_QxyQz`)
- ssrl_xrd_tools containers are simpler dataclasses with `radial`/`intensity`/`sigma`/`unit`

**Action**:
1. Extend ssrl_xrd_tools `IntegrationResult1D`/`2D` to support:
   - Dual 2theta/Q storage with lazy conversion (using `transforms/`)
   - `__add__`/`__sub__` operators
   - GI-specific result fields (as optional attributes or a subclass)
   - HDF5 serialization methods
2. Add a `from_pyfai_result()` classmethod (porting xdart's `from_result()` logic)
3. Replace xdart's containers with imports from ssrl_xrd_tools
4. Update `EwaldArch` and `EwaldSphere` to use the new containers

### 1c. Remove xdart's lmfit_models.py

**Action**: Delete `xdart/utils/lmfit_models.py`, update any imports to use `ssrl_xrd_tools.analysis.fitting.models`.

---

## Phase 2: Simplify EwaldArch / EwaldSphere + NeXus Output + GI Integration

**Goal**: Replace legacy HDF5 output with NeXus-formatted HDF5. Simplify EwaldArch and EwaldSphere. Fix display bugs. Restructure GI mode selection and integration panel.

### Status: COMPLETED (March 2026)

**NeXus output**:
- NeXus output format for all EwaldArch/EwaldSphere/ArchSeries writes
  - `entry/frames/<idx>` (1D), `<idx>_2d` (2D), `<idx>_thumb` (thumbnail)
  - `entry/integrated_1d`, `entry/integrated_2d` (sphere-level summed results)
  - `entry/calibration` (PONI), `entry/scan_data` (DataFrame columns)
  - Uses gzip compression only (lzf crashes on ARM64 macOS)
- `save_to_nexus()` / `load_from_nexus()` on EwaldArch
- ArchSeries rewritten — NeXus-only, no legacy `arches/` group
- EwaldSphere rewritten (~270 lines vs ~580) — NeXus file structure
- Thumbnail feature: downsampled raw image with mask baked in (NaN before zoom)
- `source_file` stored as relative path from HDF5 directory to raw image
- `global_mask` propagation: sphere → arch_series → save_to_nexus → thumbnail

**GI integration restructuring**:
- GI 1D/2D mode selection moved from display_frame_widget to spec_wrangler parameter tree
- Integrator panel dynamically updates labels/ranges per mode (q_ip, q_oop, q_total, qip_qoop, q_chi)
- npt_oop parameter added for fiber integrator control
- GI-specific kwargs (`gi_mode_1d`, `gi_mode_2d`, `npt_oop`, `sample_orientation`, `tilt_angle`) filtered before passing to standard pyFAI
- FiberIntegrator cached between integrate_1d and integrate_2d calls
- GI integration optimization: only compute the selected mode (was computing all 4 1D + all 3 2D modes)

**2D integration fixes**:
- Fixed qip_qoop double-transpose (result used directly, no extra `.T`)
- Fixed 2D normalization in GI path (`(map_raw - bg_raw) / map_norm`)
- Removed all `[:, ::-1]` azimuthal flips from display code — FiberIntegrator results handle axis direction correctly
- Chi axis auto-centred around 0 for non-GI mode (pyFAI geometry offset corrected post-integration)
- Chi range set to auto for non-GI and polar GI modes (no forced -90/90 cutoff)

**Mask & performance**:
- `USE_LEGACY_MASK_NORMALIZATION = False` on both AzimuthalIntegrator and FiberIntegrator
- `get_mask()` optimized: uses `dtype=bool` array, handles shape mismatches
- Mask+threshold adds ~0.3s per image (unavoidable due to pyFAI engine rebuild)

**GUI label cleanup**:
- Shortened all GI labels: Q_ip, Q_oop, Q, Chi (Greek), etc.
- "IPxOOP" → "Pts", "Points" → "Pts"
- Default sample_orientation changed to 4

**Persistence**:
- Mask threshold parameters (apply_threshold, min, max) saved/restored in session
- Data directory file dialog starts from last-used directory
- `set_image_units()` called on `sigUpdateGI` and `new_scan()` for immediate UI updates

**Known remaining items** (deferred to Phase 4):
- Display not scrolling to last processed image (existing bug)
- GUI update lag — display repaints skip every other frame during fast processing
- Legacy file backward compatibility: `load_from_h5()` still reads old format

### Files Modified (xdart)

| File | Changes |
|------|---------|
| `modules/ewald/arch.py` | `save_to_nexus`, `load_from_nexus`, `_make_thumbnail`, GI integration (only selected mode), FiberIntegrator caching, `_gi_only_keys` filter, chi axis centering, `USE_LEGACY_MASK_NORMALIZATION`, mask dtype=bool |
| `modules/ewald/arch_series.py` | Complete rewrite — NeXus-only, `global_mask` forwarding |
| `modules/ewald/sphere.py` | Complete rewrite — NeXus file structure, `global_mask` forwarding |
| `gui/tabs/static_scan/display_frame_widget.py` | 2D rotation fix, removed azimuthal flip, thumbnail fallback, GI axis labels, simplified GI data access, label shortening |
| `gui/tabs/static_scan/integrator.py` | GI mode combos, npts_oop widget, dynamic label/range updates, `_set_range_defaults_1d/2d` with auto chi, `_update_gi_mode_1d/2d` |
| `gui/tabs/static_scan/h5viewer.py` | Reads from `entry/frames` via `load_from_nexus`, data dir dialog starts from last dir |
| `gui/tabs/static_scan/static_scan_widget.py` | `set_image_units()` called in `new_scan()` |
| `gui/tabs/static_scan/sphere_threads.py` | Reads from `entry/frames` via `load_from_nexus` |
| `gui/tabs/static_scan/wranglers/spec_wrangler.py` | GI mode params in tree, sample_orientation=4, label shortening, mask threshold persistence, relative `source_file` path |

---

## Phase 3: Clean Up xdart Utils

**Goal**: Remove code from xdart that's now redundant with ssrl_xrd_tools.

### Status: COMPLETED (March 2026)

### 3a. _utils.py cleanup

**Removed** (unused functions — ~200 lines):
- `find_between()`, `find_between_r()` — string utilities, never called
- `get_scan_name()`, `get_img_number()` — superseded by `get_sname_img_number()`
- `get_motor_val()` — unused, ssrl_xrd_tools has `read_pdi_metadata`
- `get_norm_fac()`, `get_normChannel()` — unused (display_frame_widget has its own method)
- `smooth_img()` — unused image filter
- `launch()` — unused OS launcher
- `get_mask_array()` — unused, ssrl_xrd_tools has equivalent

**Replaced with re-exports**:
- `write_xye()`, `write_csv()` — now re-exported from `ssrl_xrd_tools.io.export`. All call sites (`ut.write_xye()` in display_frame_widget.py) continue to work unchanged.

**Kept** (GUI-specific, no ssrl_xrd_tools equivalent):
- `get_fname_dir()` — xdart temp directory management
- `split_file_name()`, `get_sname_img_number()` — file name parsing
- `get_series_avg()` — multi-image averaging with metadata
- `match_img_detector()` + `detector_file_sizes` — detector validation by file size
- `get_img_meta()` — thin wrapper around ssrl_xrd_tools (used by `get_series_avg`)
- `get_img_data()` — image loading with GUI-specific transforms (flip, transpose)
- `FixSizeOrderedDict` — LRU cache for GUI
- HDF5 codec re-exports from `ssrl_xrd_tools.core.hdf5`

### 3b. Remove xdart/modules/spec/

**DONE** — Entire `modules/spec/` directory removed. No imports found anywhere in xdart. ssrl_xrd_tools.io.spec (silx-based) fully replaces it.

Files removed: `__init__.py`, `spec_utils.py`, `LoadSpecFile.py`, `MakePONI.py`

### 3c. Audit remaining xdart.utils

Kept in xdart (GUI-specific, no duplication):
- `session.py` — app session persistence (`~/.xdart/session.json`)
- `h5pool.py` — HDF5 file handle caching
- `pyFAI_binaries.py` — external GUI tool wrappers (calib2, drawmask)
- `pgOverrides/` — pyqtgraph customizations
- `lmfit_models.py` — fitting models (ssrl_xrd_tools has its own copy in `analysis.fitting.models`; deduplication deferred to Phase 5)
- `containers/` — thin re-export of ssrl_xrd_tools containers

---

## Phase 4: Modernize xdart GUI

**Goal**: Clean up the GUI code itself, independent of the ssrl_xrd_tools extraction.

### Status: COMPLETED (March 2026)

### 4a. Framework decision: Stay with PySide6 + pyqtgraph
Evaluated alternatives (vispy/napari, Dear PyGui, Tkinter). Staying with current stack — proven for large detector images, mature ecosystem.

### 4b. Spec wrangler split — DONE
Split `spec_wrangler.py` (1721→929 lines) into:
- `wranglers/spec_wrangler.py` — UI widget (specWrangler class, parameter tree, session persistence)
- `wranglers/spec_wrangler_thread.py` (850 lines) — Worker thread (specThread class, utility functions: `_is_eiger_master`, `_get_scan_info`, natural sort helpers)

### 4c. Display frame partial split — DONE
Extracted axis labels, unicode constants, GI label arrays, and `_downsample_for_display` to `display_constants.py`. A deeper 3-way split (1D/2D/controller) was assessed but deferred: the single `displayFrameWidget` class is tightly coupled to the `Ui_Form`, and splitting requires rearchitecting the Qt Designer form. Safe to revisit once Qt test infrastructure is available.

### 4d. GUI update lag fix — DONE
Root cause: each `sigUpdate` emission from the processing thread triggers a full `displayframe.update()` on the main thread. When the thread processes faster than the GUI renders, signals queue up and every frame triggers a redundant redraw.

Fix: Added a **coalescing QTimer** (200 ms single-shot) in `static_scan_widget`:
- `update_data(idx)` now only updates in-memory data structures (fast, under `file_lock`), then starts/restarts the timer
- `_flush_pending_update()` fires once after the burst settles, rendering only the latest frame
- `wrangler_finished()` flushes immediately so the final frame is always shown
- Result: ~5 fps display refresh during processing, no frame skipping, no GUI lag

### 4e. NeXus/Tiled wrangler — DONE
Added new wrangler for NeXus/HDF5 image stacks:
- `wranglers/nexus_wrangler.py` — Widget with parameter tree for NeXus file, PONI, mask, GI params, output dir. Simpler than specWrangler (no SPEC file parsing, no background matching). Session persistence via `_SESSION_PARAMS` pattern.
- `wranglers/nexus_wrangler_thread.py` — Worker thread using `ssrl_xrd_tools.io.nexus.find_nexus_image_dataset` to locate the 3D image dataset, reads frames via h5py, integrates each through EwaldArch pipeline.
- Registered as `'NeXus'` in `static_scan_widget.wranglers` dict — appears as a tab alongside `'SPEC'`.

**Files changed in Phase 4**:

| File | Changes |
|------|---------|
| `wranglers/spec_wrangler.py` | Removed specThread + utilities (~800 lines), simplified imports |
| `wranglers/spec_wrangler_thread.py` | **NEW** — specThread class + utility functions |
| `wranglers/nexus_wrangler.py` | **NEW** — nexusWrangler widget |
| `wranglers/nexus_wrangler_thread.py` | **NEW** — nexusThread worker |
| `wranglers/__init__.py` | Added nexusWrangler import |
| `display_constants.py` | **NEW** — axis labels, unicode constants, downsample helper |
| `display_frame_widget.py` | Imports from display_constants (~50 lines removed) |
| `static_scan_widget.py` | Coalescing QTimer for update lag fix, NeXus wrangler registration |

---

## Phase 5: Build ssrl_xrd_tools Jupyter Widgets

**Goal**: Create interactive Jupyter notebook widgets in `ssrl_xrd_tools/gui/` for advanced users.

### 5a. Architecture

Use `panel` + `bokeh` for interactive widgets (already listed as optional deps). These provide:
- Interactive plots embeddable in Jupyter
- Parameter widgets (sliders, dropdowns)
- Layout management
- No Qt dependency

### 5b. Planned widgets

1. **powder_1d_viewer** — Interactive 1D diffraction pattern viewer
   - Q/2theta toggle, log/linear scale, peak markers
   - Overlay multiple scans
   - Integration parameter adjustment

2. **powder_2d_viewer** — Interactive 2D (cake) map viewer
   - I(Q, chi) pcolormesh with colormap controls
   - Line cut extraction (horizontal/vertical)

3. **rsm_viewer** — RSM volume explorer
   - 3-panel HK/KL/HL projections
   - Interactive HKL range sliders
   - 1D line cut extraction

4. **napari_viewer** — 3D volumetric viewer (uses napari)
   - For RSM volumes and image stacks

5. **integration_widget** — Interactive integration parameter tuning
   - Load image + PONI, adjust parameters, see live preview
   - Export integration config for batch processing

### 5c. Pattern

Each widget:
- Takes ssrl_xrd_tools data types as input (IntegrationResult1D/2D, RSMVolume, etc.)
- Pure visualization/interaction — no data processing logic
- Returns user selections/parameters as ssrl_xrd_tools config objects
- Works in JupyterLab, VS Code notebooks, and Google Colab

---

## Execution Order

```
Phase 1 (Containers)     ✅ DONE
  ├── 1a: PONI
  ├── 1b: Integration results
  └── 1c: lmfit models
      ↓
Phase 2 (EwaldArch/Sphere + GI) ✅ DONE
  ├── 2a: EwaldArch — NeXus, GI restructuring, mask fixes
  ├── 2b: EwaldSphere — NeXus rewrite
  └── 2c: Location decision — keep in xdart (Option A)
      ↓
Phase 3 (Cleanup)         ✅ DONE
  ├── 3a: _utils.py — removed ~200 lines, re-exported write_xye/csv
  ├── 3b: modules/spec — removed entirely
  └── 3c: Audit remaining — kept GUI-specific utils
      ↓
Phase 4 (GUI Modernize)   ✅ DONE
  ├── 4a: Framework decision (stay PySide6)
  ├── 4b: Split spec_wrangler → spec_wrangler.py + spec_wrangler_thread.py
  ├── 4c: Extract display_constants.py (deeper split deferred)
  ├── 4d: Coalescing QTimer for GUI update lag fix
  └── 4e: NeXus/Tiled wrangler (nexus_wrangler.py + nexus_wrangler_thread.py)
      ↓
Phase 4.5 (xdart Hardening) ✅ DONE
  ├── Eiger HDF5: persistent h5py handle (was open/close per frame)
  ├── Dead code removal (unused imports, functions, commented blocks)
  ├── 1D-only mode audit: confirmed correct (skip_2d propagation)
  ├── Live mode audit: functional, glob+sleep(2s) polling
  ├── ssrl_xrd_tools delegation audit: all heavy computation delegated
  └── Threading audit: h5pool, coalescing timer, file_lock all verified
      ↓
Phase 5 (Jupyter Widgets + Analysis) ← NEXT
  ├── 5a: Architecture (panel + bokeh)
  ├── 5b: Jupyter viewers (powder_1d, powder_2d, rsm, integration)
  └── 5c: Analysis tools (fitting, strain, texture)
```

## Testing Strategy

Each phase should maintain passing tests:

- **Phase 1**: Write adapter tests — create xdart containers from ssrl_xrd_tools types and vice versa
- **Phase 2**: Integration tests — full pipeline from raw image through EwaldArch to plotable results
- **Phase 3**: Ensure no xdart test regressions after removing code
- **Phase 4**: Manual GUI testing at minimum; consider pytest-qt for widget tests
- **Phase 5**: Notebook-based test examples

## Future xdart Features (Post-Phase 5)

### MultiGeometry stitching
pyFAI's `MultiGeometry` class (in `pyFAI.multi`) can stitch together images from multiple detector positions into a single pattern with extended Q/2θ range. `ssrl_xrd_tools.integrate.multi` already wraps this with `stitch_1d()` and `stitch_2d()`. xdart's `EwaldSphere` already has stub methods `multigeometry_integrate_1d()` and `multigeometry_integrate_2d()` that delegate to these. The GUI side needs a UI for selecting multiple PONI files and associating them with detector positions.

### Watchdog for live mode
Current live mode uses filesystem polling (glob every 2 seconds). The `watchdog` library would provide event-driven file detection with lower latency. This would be an **optional dependency** — add to `pyproject.toml` as:
```
watchdog = {version = "*", optional = true}
```
The wrangler thread could detect its availability at runtime and fall back to glob polling if not installed.

## Risk Mitigation

- **HDF5 backward compatibility**: Existing .h5 files saved by xdart must remain loadable. Add a version check / migration path in `load_from_h5()`.
- **Beamline disruption**: xdart is used at active beamlines. Each phase should produce a working xdart. Use feature branches and test at the beamline before merging.
- **Container mismatch**: The biggest risk is subtle differences between xdart's containers and ssrl_xrd_tools equivalents (especially the dual 2theta/Q storage and GI fields). Phase 1b needs careful testing with real beamline data.
