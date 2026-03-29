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

## Phase 2: Simplify EwaldArch / EwaldSphere

**Goal**: Reduce EwaldArch and EwaldSphere to thin GUI-oriented wrappers over ssrl_xrd_tools functions. Eventually, consider whether they should move into ssrl_xrd_tools as "workflow" classes or remain in xdart as GUI state containers.

### 2a. EwaldArch simplification

**Current role**: Holds raw image + PONI + mask + integration results + threading lock. Calls ssrl_xrd_tools for actual integration.

**Action**:
1. Replace internal `poni` attribute type with ssrl_xrd_tools `PONI`
2. Replace `int_1d`/`int_2d` attribute types with ssrl_xrd_tools `IntegrationResult1D`/`2D`
3. Remove any integration logic that duplicates ssrl_xrd_tools (already mostly done)
4. Keep threading lock and HDF5 persistence (these are GUI-specific concerns)
5. Consider: should `EwaldArch` become a dataclass in ssrl_xrd_tools that xdart wraps with threading?

### 2b. EwaldSphere simplification

**Current role**: Collection of EwaldArch objects + multi-geometry integration + summed results. Already uses ssrl_xrd_tools for stitching.

**Action**:
1. Use ssrl_xrd_tools containers for `bai_1d`/`bai_2d`
2. Keep `ArchSeries` (lazy HDF5-backed storage) in xdart — it's a GUI/performance concern
3. Consider: should the "sum individual arch results" logic move to ssrl_xrd_tools as a utility?

### 2c. Decide EwaldArch/Sphere location

**Option A — Keep in xdart**: They remain GUI state containers. ssrl_xrd_tools stays purely functional (stateless functions + data containers). xdart's `EwaldArch` wraps ssrl_xrd_tools types with threading, HDF5 caching, and GUI state.

**Option B — Move to ssrl_xrd_tools**: Create `ssrl_xrd_tools.workflow.EwaldArch` and `EwaldSphere` as non-GUI workflow classes (no threading, no Qt). xdart subclasses them to add GUI concerns.

**Recommendation**: Option A for now. EwaldArch/Sphere are tightly coupled to xdart's HDF5 persistence and threading model. Moving them would require abstracting those concerns first, which adds complexity without clear benefit.

---

## Phase 3: Clean Up xdart Utils

**Goal**: Remove code from xdart that's now redundant with ssrl_xrd_tools.

### 3a. _utils.py cleanup

Much of `_utils.py` re-exports from ssrl_xrd_tools.core.hdf5 already. Remaining functions:

- `get_fname_dir()`, `split_file_name()`, `get_sname_img_number()` — file path utilities, keep in xdart (GUI-specific)
- `write_xye()`, `write_csv()` — already in ssrl_xrd_tools.io.export, remove xdart copies
- `FixSizeOrderedDict` — keep in xdart (GUI cache utility)
- `match_img_detector()`, `get_series_avg()` — evaluate whether ssrl_xrd_tools.io or integrate modules already cover these

### 3b. Remove xdart/modules/spec/

If `ssrl_xrd_tools.io.spec` fully replaces this, remove it.

### 3c. Audit remaining xdart.utils

Keep in xdart (GUI-specific):
- `session.py` — app session persistence
- `h5pool.py` — HDF5 file handle caching
- `pyFAI_binaries.py` — external GUI tool wrappers (calib2, drawmask)
- `pgOverrides/` — pyqtgraph customizations

---

## Phase 4: Modernize xdart GUI

**Goal**: Clean up the GUI code itself, independent of the ssrl_xrd_tools extraction.

### 4a. GUI framework evaluation

Current: PySide6 + pyqtgraph. Options to consider:

| Framework | Pros | Cons |
|---|---|---|
| **PySide6 + pyqtgraph** (current) | Proven, fast rendering for large images, mature ecosystem, works well at beamlines | Complex signal/slot wiring, platform-specific UI issues |
| **PySide6 + vispy/napari** | Better 3D support, modern rendering | Migration effort, napari is heavier |
| **Dear PyGui** | Very fast rendering, modern API, simpler than Qt | Less mature, smaller ecosystem |
| **Tkinter + matplotlib** | Simple, no external deps | Slower for large images, less interactive |

**Recommendation**: Stay with PySide6 + pyqtgraph. It's proven for this use case, pyqtgraph handles large detector images well, and switching frameworks would be high effort with unclear benefit. Focus engineering time on architecture, not framework migration.

### 4b. Spec wrangler refactoring

`spec_wrangler.py` is 1721 lines — the largest file. It mixes:
- Parameter tree UI
- SPEC file parsing orchestration
- Image loading/stacking logic
- Session save/load
- Integration parameter management

**Action**: Split into:
1. `wranglers/spec_wrangler_ui.py` — Pure UI (parameter tree, signals)
2. `wranglers/spec_wrangler_logic.py` — Orchestration that calls ssrl_xrd_tools
3. Consider a `wranglers/nexus_wrangler.py` for NeXus/Bluesky data sources

### 4c. Display frame modernization

`display_frame_widget.py` (1492 lines) handles both 1D plots and 2D images. Consider splitting:
1. `display/plot_1d_widget.py` — 1D line plots with overlays
2. `display/image_2d_widget.py` — 2D image display with colormap/histogram
3. `display/display_controller.py` — Coordination between views

### 4d. Threading cleanup

Current threading uses a mix of `QThread`, `ProcessPoolExecutor`, and `multiprocessing.Process` with `Condition` locks.

**Action**:
1. Standardize on `QThread` + `QThreadPool` for GUI-bound work
2. Use `concurrent.futures.ProcessPoolExecutor` for CPU-heavy integration (already partially done)
3. Remove raw `multiprocessing.Process` usage in wranglers
4. Consider `asyncio` integration for I/O-bound operations (Tiled catalog queries, network file access)

### 4e. Add NeXus/Tiled wrangler

For the Bluesky migration, add a new wrangler that reads from NeXus files or Tiled catalogs using `ssrl_xrd_tools.io.nexus` and `ssrl_xrd_tools.io.tiled`.

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

Recommended sequence, with each phase building on the previous:

```
Phase 1 (Containers)     ← Foundation, unblocks everything else
  ├── 1a: PONI
  ├── 1b: Integration results
  └── 1c: lmfit models
      ↓
Phase 2 (EwaldArch/Sphere) ← Uses new containers
  ├── 2a: EwaldArch
  ├── 2b: EwaldSphere
  └── 2c: Location decision
      ↓
Phase 3 (Cleanup)         ← Remove now-redundant code
  ├── 3a: _utils.py
  ├── 3b: modules/spec
  └── 3c: Audit remaining
      ↓
Phase 4 (GUI Modernize)   ← Can partially overlap with Phase 3
  ├── 4a: Framework decision (stay PySide6)
  ├── 4b: Split spec_wrangler
  ├── 4c: Split display_frame
  ├── 4d: Threading cleanup
  └── 4e: NeXus/Tiled wrangler
      ↓
Phase 5 (Jupyter Widgets)  ← Independent, can start in parallel
  ├── 5a: Architecture
  ├── 5b: powder_1d_viewer (first)
  ├── 5c: powder_2d_viewer
  └── 5d: rsm_viewer, integration_widget
```

## Testing Strategy

Each phase should maintain passing tests:

- **Phase 1**: Write adapter tests — create xdart containers from ssrl_xrd_tools types and vice versa
- **Phase 2**: Integration tests — full pipeline from raw image through EwaldArch to plotable results
- **Phase 3**: Ensure no xdart test regressions after removing code
- **Phase 4**: Manual GUI testing at minimum; consider pytest-qt for widget tests
- **Phase 5**: Notebook-based test examples

## Risk Mitigation

- **HDF5 backward compatibility**: Existing .h5 files saved by xdart must remain loadable. Add a version check / migration path in `load_from_h5()`.
- **Beamline disruption**: xdart is used at active beamlines. Each phase should produce a working xdart. Use feature branches and test at the beamline before merging.
- **Container mismatch**: The biggest risk is subtle differences between xdart's containers and ssrl_xrd_tools equivalents (especially the dual 2theta/Q storage and GI fields). Phase 1b needs careful testing with real beamline data.
