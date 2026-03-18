# Phase 3: RSM Module Split

Split `rsm/core.py` (776 lines) into four focused modules while preserving
every function, every signature, and every docstring. After the split,
`rsm/__init__.py` must re-export **all** public names so that existing
`from ssrl_xrd_tools.rsm import …` statements keep working with zero changes.

Execute prompts **in order** (3A → 3B → 3C → 3D → 3E → 3F → 3G).

---

## Prompt 3A — `rsm/volume.py` (data container + utilities)

Create `ssrl_xrd_tools/rsm/volume.py`.

Move these items **verbatim** from `core.py`:

1. **RSMVolume** dataclass (lines 74-248)
   - `__post_init__`, `shape`, `get_bounds`, `save_vtk`, `get_slice`,
     `line_cut`, `crop`
   - RSMVolume methods call `extract_2d_slice`, `extract_line_cut`,
     `mask_data`, and `save_vtk` — import those from the same file.

2. **mask_data()** function (lines 625-638)

3. **save_vtk()** free function (lines 641-663)
   - Keep the `_VTK_AVAILABLE` / `gridToVTK` optional-import guard at module
     level.

4. **extract_line_cut()** function (lines 666-718)

5. **extract_2d_slice()** function (lines 721-776)

Imports this file needs:
```python
from __future__ import annotations
import logging
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any
```

Plus the optional pyevtk import (same pattern as current core.py).

Do **not** import anything from other rsm submodules.

---

## Prompt 3B — `rsm/geometry.py` (diffractometer config)

Create `ssrl_xrd_tools/rsm/geometry.py`.

Move these items **verbatim** from `core.py`:

1. **DiffractometerConfig** dataclass (lines 258-291)
   - Imports `xrayutilities as xu` for `xu.QConversion` and `xu.HXRD`
     inside `make_hxrd`.

Imports this file needs:
```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import xrayutilities as xu
```

Do **not** import anything from other rsm submodules.

---

## Prompt 3C — `rsm/gridding.py` (HKL gridding + volume combination)

Create `ssrl_xrd_tools/rsm/gridding.py`.

Move these items **verbatim** from `core.py`:

1. **grid_img_data()** (lines 408-462)
   - Calls `diff_config.make_hxrd(energy)` — type-hint the parameter as
     `DiffractometerConfig` via a `TYPE_CHECKING` guard import from
     `ssrl_xrd_tools.rsm.geometry`.
   - Returns `RSMVolume` — import from `ssrl_xrd_tools.rsm.volume` at
     runtime (not behind TYPE_CHECKING since it's used at runtime).

2. **get_common_grid()** (lines 572-589)

3. **combine_grids()** (lines 592-618)
   - Uses `RegularGridInterpolator` from scipy.
   - Returns `RSMVolume` — same runtime import as above.

Imports this file needs:
```python
from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any
import numpy as np
import xrayutilities as xu
from scipy.interpolate import RegularGridInterpolator
from ssrl_xrd_tools.rsm.volume import RSMVolume

if TYPE_CHECKING:
    from ssrl_xrd_tools.rsm.geometry import DiffractometerConfig
```

---

## Prompt 3D — `rsm/pipeline.py` (experiment config + processing)

Create `ssrl_xrd_tools/rsm/pipeline.py`.

Move these items **verbatim** from `core.py`:

1. **_as_path()** helper (line 51-52)

2. **_load_pickle()** helper (lines 55-61)

3. **_save_pickle()** helper (lines 64-67)

4. **ScanInfo** dataclass (lines 251-255)

5. **ExperimentConfig** dataclass (lines 294-361)
   - Its `process()` method calls `process_scan()` which is defined in the
     same file.
   - Uses `DiffractometerConfig` — import from
     `ssrl_xrd_tools.rsm.geometry`.
   - Uses `ScanInfo` (same file, no cross-module import needed).

6. **load_images()** (lines 368-401)
   - Imports from `ssrl_xrd_tools.io.image` and `ssrl_xrd_tools.io.spec`
     — keep these exactly as they are in core.py.

7. **process_scan_data()** (lines 465-512)
   - Calls `grid_img_data` — import from `ssrl_xrd_tools.rsm.gridding`.
   - Calls `load_images` (same file).
   - Returns `RSMVolume | None` — import from `ssrl_xrd_tools.rsm.volume`.

8. **process_scan()** (lines 515-565)
   - Calls `process_scan_data` (same file).
   - Uses `_load_pickle` / `_save_pickle` (same file).
   - Returns `RSMVolume | None` — import from `ssrl_xrd_tools.rsm.volume`.

Imports this file needs:
```python
from __future__ import annotations
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
from ssrl_xrd_tools.io.image import (
    read_image,
    read_image_stack,
    read_images_parallel,
    find_image_files,
    get_detector_mask,
    apply_rotation,
)
from ssrl_xrd_tools.io.spec import (
    get_scan_path_info,
    get_energy_and_UB,
    get_angles,
)
from ssrl_xrd_tools.rsm.volume import RSMVolume
from ssrl_xrd_tools.rsm.geometry import DiffractometerConfig
from ssrl_xrd_tools.rsm.gridding import grid_img_data
```

---

## Prompt 3E — Update `rsm/__init__.py` (re-export everything)

Replace the contents of `ssrl_xrd_tools/rsm/__init__.py` with:

```python
"""
Reciprocal Space Mapping (RSM) utilities for x-ray diffraction data.

Processes SPEC scans and detector images into 3D HKL volumes, with I/O,
gridding, combination, line cuts, slices. Image I/O and SPEC parsing
live in ssrl_xrd_tools.io; fitting in ssrl_xrd_tools.analysis.fitting.
"""

from ssrl_xrd_tools.rsm.volume import (
    RSMVolume,
    extract_2d_slice,
    extract_line_cut,
    mask_data,
    save_vtk,
)
from ssrl_xrd_tools.rsm.geometry import DiffractometerConfig
from ssrl_xrd_tools.rsm.gridding import (
    combine_grids,
    get_common_grid,
    grid_img_data,
)
from ssrl_xrd_tools.rsm.pipeline import (
    ExperimentConfig,
    ScanInfo,
    load_images,
    process_scan,
    process_scan_data,
)

# Backward-compatible alias
extract_slice = extract_2d_slice
```

Then **delete** `rsm/core.py`. It must no longer exist.

Verify: `from ssrl_xrd_tools.rsm import RSMVolume, ExperimentConfig` must
still work.

---

## Prompt 3F — Tests for RSM modules

Create `tests/test_rsm.py`.

### Fixtures (in conftest.py or at top of file)

```python
@pytest.fixture
def rsm_volume():
    """Small 10×10×10 RSMVolume for testing."""
    h = np.linspace(-1, 1, 10)
    k = np.linspace(-1, 1, 10)
    l = np.linspace(0, 2, 10)
    intensity = np.random.default_rng(42).random((10, 10, 10))
    return RSMVolume(h=h, k=k, l=l, intensity=intensity)
```

### Test classes

1. **TestRSMVolume**
   - `test_shape` — verify `.shape` property returns (10, 10, 10)
   - `test_get_bounds` — verify 3 elements, each with [min, max, step]
   - `test_crop` — crop to half range, verify smaller shape and axis bounds
   - `test_crop_no_change` — crop with (-inf, inf) returns same shape
   - `test_line_cut_h` — `line_cut("h")` returns (h, 1d_array) with
     len(h)==10
   - `test_line_cut_with_ranges` — pass fixed_ranges, verify shapes
   - `test_get_slice_k` — `get_slice("k")` returns 4-tuple, slice is 2D
   - `test_get_slice_with_range` — pass val_range, verify integrated axis
     shrinks
   - `test_invalid_axis` — ValueError for axis="x"
   - `test_intensity_shape_mismatch` — ValueError when axes don't match
     intensity

2. **TestMaskData**
   - `test_crop_range` — verify cropped array shape
   - `test_full_range` — (-inf, inf) returns original

3. **TestExtractLineCut**
   - `test_axis_0` — extract along axis=0, verify shape
   - `test_axis_invalid` — ValueError for axis=3

4. **TestExtract2dSlice**
   - `test_integrate_axis_0` — verify returned 2D shape and axis values
   - `test_with_range` — restrict axis_range, verify fewer integrated vals

5. **TestSaveVtk**
   - `test_save_vtk_not_available` — mock `_VTK_AVAILABLE=False`, expect
     ImportError
   - (Skip real VTK test if pyevtk not installed)

6. **TestDiffractometerConfig**
   - `test_defaults` — verify default sample_rot, detector_rot, r_i
   - `test_make_hxrd` — call `make_hxrd(12000.0)`, verify returns xu.HXRD
     instance (skipif xrayutilities not installed)

7. **TestGridImgData** (skipif xrayutilities not installed)
   - `test_returns_rsm_volume` — pass tiny synthetic 3-frame stack with
     mock energy/UB/angles, verify returns RSMVolume
   - `test_roi_crop` — pass roi, verify resulting image dimensions change

8. **TestCombineGrids**
   - `test_combine_two_volumes` — combine 2 RSMVolumes, verify returned
     shape matches bins
   - `test_empty_list` — ValueError

9. **TestGetCommonGrid**
   - `test_common_grid_bounds` — verify grid spans full range of both volumes
   - `test_empty_list` — ValueError

10. **TestScanInfo**
    - `test_fields` — verify spec_path, img_dir, h5_path fields

11. **TestExperimentConfig**
    - `test_post_init_creates_dir` — verify pickle_dir is created
    - `test_find_h5` — mock directory with matching file, verify returns Path

12. **TestProcessScanData** / **TestProcessScan**
    - These depend on SPEC files and real detector images. Add
      `@pytest.mark.skipif` with reason "requires SPEC data" and leave as
      integration test stubs:
    ```python
    @pytest.mark.skip(reason="requires SPEC data and detector images")
    def test_process_scan_data(self):
        pass
    ```

### Conventions
- `from __future__ import annotations` at top
- Use `np.testing.assert_allclose` for float comparisons
- `pytest.raises` for expected errors
- Mark xrayutilities-dependent tests with
  `@pytest.mark.skipif(not _HAS_XU, reason="xrayutilities not installed")`

---

## Prompt 3G — Update CLAUDE.md

In `CLAUDE.md`:

1. Update the module map to show the split:
```
├── rsm/            # Reciprocal space mapping via xrayutilities
│   ├── volume.py       # ✅ RSMVolume, extract_line_cut, extract_2d_slice, mask_data, save_vtk
│   ├── geometry.py     # ✅ DiffractometerConfig
│   ├── gridding.py     # ✅ grid_img_data, combine_grids, get_common_grid
│   └── pipeline.py     # ✅ ExperimentConfig, ScanInfo, process_scan, load_images
```

2. Replace the "Planned: RSM Split" section with:
```
## Completed: RSM Split

`rsm/core.py` (776 lines) has been split into:
- `rsm/volume.py` — RSMVolume class with slicing, cropping, VTK export, line cuts, 2D slices
- `rsm/geometry.py` — DiffractometerConfig with make_hxrd
- `rsm/gridding.py` — grid_img_data, combine_grids, get_common_grid
- `rsm/pipeline.py` — ExperimentConfig, ScanInfo, process_scan, process_scan_data, load_images

`rsm/__init__.py` re-exports everything for backward compatibility.
```

3. Add to the status line for `rsm/`:
```
- `rsm/` — ✅ split into volume.py, geometry.py, gridding.py, pipeline.py (tests in test_rsm.py)
```
