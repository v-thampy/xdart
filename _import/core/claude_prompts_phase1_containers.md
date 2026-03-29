# Phase 1: Container Unification — Claude Code Prompts

## Prerequisites

Set up the branch structure on both repos.

**Existing state** (as of March 2026):
- ssrl_xrd_tools is on `dev` (2 commits ahead of origin/dev)
- xdart is on `refactor/ssrl-xrd-tools` (34 commits of prior refactoring)

```bash
# ssrl_xrd_tools: commit any unstaged work, then create phase branch off dev
cd ~/repos/ssrl_xrd_tools
git checkout dev
git add ssrl_xrd_tools/io/__init__.py ssrl_xrd_tools/io/image.py
git commit -m "WIP: io module changes"
git add REFACTORING_PLAN.md claude_prompts_phase1_containers.md
git commit -m "Add refactoring plan and Phase 1 prompts"
git checkout -b refactor/container-unification

# xdart: commit any unstaged work on existing branch (Phase 1c will branch off this)
cd ~/repos/xdart
git checkout refactor/ssrl-xrd-tools
git add xdart/gui/tabs/static_scan/h5viewer.py xdart/gui/tabs/static_scan/static_scan_widget.py xdart/gui/tabs/static_scan/wranglers/spec_wrangler.py xdart/modules/ewald/arch_series.py xdart/modules/ewald/sphere.py xdart/utils/h5pool.py
git commit -m "WIP: in-progress refactoring changes"
# Later, for Phase 1c:
# git checkout -b refactor/ssrl-integration
```

**Branch strategy**:
- ssrl_xrd_tools: phase branches merge into `dev`. `dev` merges into `main` at the end.
- xdart: phase branches merge into `refactor/ssrl-xrd-tools`. That merges into `master` at the end.

---

## Phase 1a: Enhance ssrl_xrd_tools PONI Container

### Prompt for Claude Code

```
## Task: Enhance the PONI dataclass in ssrl_xrd_tools

### FIRST: Ensure you're on the correct branch
Before making any code changes, verify you're on the phase branch:
```bash
git checkout refactor/container-unification 2>/dev/null || git checkout -b refactor/container-unification dev
```
All work in this session goes on `refactor/container-unification`.
When done, it will be merged into `dev` (not main). Main is only updated
after the entire multi-phase refactor is complete.

### Context
I'm refactoring xdart to import all scientific functionality from ssrl_xrd_tools.
xdart has its own PONI class (in `xdart/utils/containers/poni.py`) with I/O methods
that ssrl_xrd_tools' PONI (in `ssrl_xrd_tools/core/containers.py`) currently lacks.
I need to enhance the ssrl_xrd_tools PONI so xdart can stop using its local copy.

### Current ssrl_xrd_tools PONI (core/containers.py)
- Simple @dataclass(slots=True) with: dist, poni1, poni2, rot1, rot2, rot3, wavelength, detector (str)
- No I/O methods

### xdart PONI features to port (xdart/utils/containers/poni.py)
- `to_dict()` → returns dict with pyFAI keys: Distance, Poni1, Poni2, Rot1, Rot2, Rot3, Wavelength
- `from_dict(d)` → classmethod that creates PONI from dict with those keys
- `from_yamdict(d)` → classmethod parsing YAML-format dict, handles detector instantiation
- `from_ponifile(file)` → classmethod that reads a .poni file (YAML format)
- `from_yaml(stream)` → classmethod parsing YAML stream
- Detector stored as pyFAI Detector object (not string)

### Requirements

1. **Keep `detector` as `str`** in the dataclass (store the detector name, not the object).
   Resolving to a pyFAI Detector object is already handled by `integrate/calibration.py::poni_to_integrator`.

2. **Add these methods to the ssrl_xrd_tools PONI dataclass**:
   - `to_dict() -> dict` — returns pyFAI-style dict with keys: dist, poni1, poni2, rot1, rot2, rot3, wavelength, detector
   - `from_dict(d: dict) -> PONI` — classmethod, accepts both pyFAI-style keys (Distance/Poni1/Poni2) and lowercase keys
   - `from_poni_file(path: Path | str) -> PONI` — classmethod, reads a .poni file (YAML format since pyFAI 0.21+). Extract dist, poni1, poni2, rot1-3, wavelength, and detector name from the file.
   - `to_poni_file(path: Path | str) -> None` — writes a .poni file in pyFAI YAML format

3. **Do NOT add pyFAI as a required import** — use lazy import or just parse the YAML directly.
   The .poni file format is simple YAML with keys like `Detector`, `Distance`, `Poni1`, etc.

4. **Keep @dataclass(slots=True)** — add methods using the pattern:
   ```python
   @dataclass(slots=True)
   class PONI:
       ...fields...

       def to_dict(self) -> dict:
           ...
   ```

5. **Follow project conventions**: `from __future__ import annotations`, NumPy-style docstrings,
   `Path | str` for file args, logging not print.

6. **Add/update tests** in `tests/` for the new methods.
   Test round-trip: create PONI → to_dict → from_dict → compare.
   Test from_poni_file with a sample .poni file (create a fixture).

### Files to modify
- `ssrl_xrd_tools/core/containers.py` — enhance PONI class
- `tests/test_containers.py` (create if doesn't exist) — add PONI tests

### Files to reference (read-only)
- `../xdart/xdart/utils/containers/poni.py` — xdart's PONI for feature parity
- `ssrl_xrd_tools/integrate/calibration.py` — see how PONI is currently used
```

---

## Phase 1b: Enhance Integration Result Containers

### Prompt for Claude Code

```
## Task: Redesign IntegrationResult1D and IntegrationResult2D containers

### Context
I'm unifying the data containers between ssrl_xrd_tools and xdart. The current
ssrl_xrd_tools containers are minimal dataclasses. xdart's containers
(`int_1d_data_static`, `int_2d_data_static`) have features I need: HDF5 I/O,
arithmetic operators, and unit-aware conversions. Additionally, the containers
must support ALL output types from pyFAI's FiberIntegrator for grazing incidence
XRD — not just standard (q, 2theta, chi) results.

### Design Decisions (already made, follow these exactly)

1. **Unit-aware, not axis-specific**: The containers are generic. The `unit` string
   tells you what each axis represents. Do NOT create separate container classes
   for GI vs standard results.

2. **Single representation, not dual storage**: xdart's approach of storing both
   q AND 2theta arrays simultaneously is being DROPPED. Store one representation.
   Provide conversion methods that compute on the fly using `transforms/`.

3. **Two unit fields for 2D**: `IntegrationResult2D` needs both `unit` (radial)
   and `azimuthal_unit` (azimuthal axis) because GI results have non-trivial
   azimuthal axes (qoop, exit angles, chi_gi).

### Supported unit strings (from pyFAI)

These are the pyFAI unit strings the containers must handle:

**Standard integration:**
- Radial: `"2th_deg"`, `"2th_rad"`, `"q_A^-1"`, `"q_nm^-1"`, `"d*2_A^-2"`, `"r_mm"`
- Azimuthal: `"chi_deg"`, `"chi_rad"`

**GI integration (FiberIntegrator):**
- `"qip_A^-1"`, `"qip_nm^-1"` — in-plane q
- `"qoop_A^-1"`, `"qoop_nm^-1"` — out-of-plane q
- `"qtot_A^-1"`, `"qtot_nm^-1"` — total q (polar mode)  [NOTE: pyFAI may use "q_A^-1" here too]
- `"chigi_deg"`, `"chigi_rad"` — GI chi angle (polar mode)
- Exit angle units (degrees) for exit-angle integration

The containers don't need to validate units — just store whatever string pyFAI gives us.

### Requirements for IntegrationResult1D

```python
@dataclass(slots=True)
class IntegrationResult1D:
    """Result of 1D azimuthal/radial integration.

    Works for standard integration (intensity vs q or 2theta) and GI integration
    (intensity vs qip, qoop, qtotal, exit angle, etc). The `unit` field identifies
    what the radial axis represents.
    """
    radial: np.ndarray          # x-axis values
    intensity: np.ndarray       # integrated intensity
    sigma: np.ndarray | None = None   # uncertainty per bin (optional)
    unit: str = "2th_deg"       # pyFAI unit string for radial axis

    # Conversion methods
    def to_unit(self, target_unit: str, wavelength: float | None = None) -> IntegrationResult1D:
        """Convert radial axis to a different unit. Returns a NEW IntegrationResult1D.

        Supported conversions (requires wavelength in Angstroms):
        - 2th_deg <-> q_A^-1  (via transforms.tth_to_q / q_to_tth)
        - 2th_rad <-> q_A^-1
        - q_A^-1 <-> q_nm^-1  (factor of 10)

        For GI units (qip, qoop, etc), conversion between _A^-1 and _nm^-1 is
        supported. Cross-axis conversions (qip -> qoop) raise ValueError.

        If conversion is not possible, raises ValueError with clear message.
        """

    # Arithmetic
    def __add__(self, other: IntegrationResult1D) -> IntegrationResult1D:
        """Add intensities. Axes must match. Propagates sigma."""

    def __sub__(self, other: IntegrationResult1D) -> IntegrationResult1D:
        """Subtract intensities. Axes must match. Propagates sigma."""

    def __mul__(self, scalar: float) -> IntegrationResult1D:
        """Scale intensity by scalar. Propagates sigma."""

    # Factory
    @classmethod
    def from_pyfai(cls, result, unit: str | None = None) -> IntegrationResult1D:
        """Create from a pyFAI integrate1d result (namedtuple with .radial, .intensity, .sigma)."""

    # I/O
    def to_hdf5(self, grp: h5py.Group, compression: str = "lzf") -> None:
        """Write to HDF5 group: datasets 'radial', 'intensity', 'sigma'; attrs 'unit'."""

    @classmethod
    def from_hdf5(cls, grp: h5py.Group) -> IntegrationResult1D:
        """Read from HDF5 group."""

    def to_nexus(self, grp: h5py.Group, signal_name: str = "intensity") -> None:
        """Write as NXdata group with proper NeXus attributes (@signal, @axes, @units)."""
```

### Requirements for IntegrationResult2D

```python
@dataclass(slots=True)
class IntegrationResult2D:
    """Result of 2D integration (cake, GI reciprocal space map, polar map, etc).

    Axis combinations this must handle:
    - Standard cake:     radial=q or 2theta,  azimuthal=chi
    - GI qip/qoop map:  radial=qip,          azimuthal=qoop
    - GI polar map:      radial=qtotal,       azimuthal=chi_gi
    - GI exit angles:    radial=horiz_exit,   azimuthal=vert_exit
    """
    radial: np.ndarray          # 1D axis
    azimuthal: np.ndarray       # 1D axis
    intensity: np.ndarray       # 2D: shape (len(radial), len(azimuthal))
    sigma: np.ndarray | None = None
    unit: str = "2th_deg"               # unit for radial axis
    azimuthal_unit: str = "chi_deg"     # unit for azimuthal axis

    # Same methods as 1D: to_unit, arithmetic, from_pyfai, to_hdf5, from_hdf5, to_nexus
    # to_unit should convert the radial axis (same as 1D)

    def to_azimuthal_unit(self, target_unit: str) -> IntegrationResult2D:
        """Convert azimuthal axis unit (e.g., chi_deg <-> chi_rad)."""

    @classmethod
    def from_pyfai(cls, result, unit: str | None = None,
                   azimuthal_unit: str | None = None) -> IntegrationResult2D:
        """Create from pyFAI integrate2d result. Note: pyFAI returns shape
        (npt_azim, npt_rad) — this constructor should transpose to (npt_rad, npt_azim)
        to match our convention. CHECK: the integrate functions in integrate/single.py
        and integrate/gid.py already handle this transpose — make sure from_pyfai
        does NOT double-transpose. If the data is already transposed, accept as-is."""

    def extract_1d(self, axis: str = "radial", index: int | None = None,
                   range_: tuple[float, float] | None = None) -> IntegrationResult1D:
        """Extract a 1D line cut from the 2D result.

        axis="radial": sum/slice along azimuthal to get I(radial)
        axis="azimuthal": sum/slice along radial to get I(azimuthal)

        If index is given, extract single row/column.
        If range_ is given, sum rows/columns in that value range.
        If neither, sum over the full axis.
        """
```

### NeXus output format (to_nexus)

The `to_nexus` method should write proper NXdata groups:

```
grp/  (NXdata)
  @signal = "intensity"
  @axes = ["radial", "azimuthal"]  (or just ["radial"] for 1D)
  @radial_indices = [0]
  @azimuthal_indices = [1]
  radial (dataset)
    @units = "angstrom^-1"  (mapped from pyFAI unit string)
    @long_name = "Q"  (human-readable label)
  azimuthal (dataset)
    @units = "degrees"
    @long_name = "Chi"
  intensity (dataset)
    @long_name = "Intensity"
  sigma (dataset, optional)
    @long_name = "Uncertainty"
```

Provide a helper `_pyfai_unit_to_nexus(unit_str) -> tuple[str, str]` that maps
pyFAI unit strings to (nexus_units, long_name) pairs:
- "q_A^-1" -> ("angstrom^-1", "Q")
- "2th_deg" -> ("degrees", "2Theta")
- "chi_deg" -> ("degrees", "Chi")
- "qip_A^-1" -> ("angstrom^-1", "Q_ip")
- "qoop_A^-1" -> ("angstrom^-1", "Q_oop")
- etc.

### Validation

The `__post_init__` should validate:
- radial and intensity have matching shapes (1D) or intensity.shape[0] == len(radial) (2D)
- sigma shape matches intensity if provided
- intensity is 2D for IntegrationResult2D

### Implementation notes

- Keep `@dataclass(slots=True)`
- `from __future__ import annotations` at top
- `import h5py` should be lazy (inside methods) since h5py is not needed for basic usage
- Use `ssrl_xrd_tools.transforms` for unit conversions in `to_unit()`
- NumPy-style docstrings
- For arithmetic: check that units match, raise ValueError if not
- For sigma propagation: use standard error propagation
  - add/sub: sigma = sqrt(sigma1² + sigma2²)
  - mul by scalar: sigma = |scalar| * sigma

### Files to modify
- `ssrl_xrd_tools/core/containers.py` — rewrite IntegrationResult1D and IntegrationResult2D
- `tests/test_containers.py` — comprehensive tests

### Files to reference (read-only)
- `../xdart/xdart/utils/containers/int_data_static.py` — xdart's containers (for feature parity)
- `ssrl_xrd_tools/transforms/__init__.py` — unit conversion functions
- `ssrl_xrd_tools/integrate/single.py` — see how results are currently created
- `ssrl_xrd_tools/integrate/gid.py` — see GI result creation and unit strings
- `ssrl_xrd_tools/io/nexus.py` — see existing NeXus write patterns

### Tests to write

1. Create IntegrationResult1D with various units, check validation
2. Round-trip: create → to_hdf5 → from_hdf5 → compare all fields
3. Round-trip: create → to_nexus → verify NXdata attributes
4. Unit conversion: create with q_A^-1, convert to 2th_deg, verify values
5. Arithmetic: add two results, check intensity and sigma propagation
6. from_pyfai: mock a pyFAI result namedtuple, verify correct parsing
7. IntegrationResult2D: all of the above plus extract_1d line cuts
8. GI-specific: create results with qip/qoop units, verify round-trip and to_nexus
```

---

## Phase 1c: Replace xdart Containers with ssrl_xrd_tools Imports

### Prompt for Claude Code

```
## Task: Replace xdart's local containers with ssrl_xrd_tools imports

### FIRST: Ensure you're on the correct branch
```bash
git checkout refactor/ssrl-integration 2>/dev/null || git checkout -b refactor/ssrl-integration refactor/ssrl-xrd-tools
```
All work goes on `refactor/ssrl-integration`, which merges into
`refactor/ssrl-xrd-tools` (not master). Master is only updated after the
entire refactor is complete.

### Context
Phase 1a and 1b enhanced the ssrl_xrd_tools containers (PONI, IntegrationResult1D,
IntegrationResult2D) to cover all features xdart needs. Now we need to update xdart
to use them instead of its local copies.

### IMPORTANT: Backward compatibility
Existing HDF5 files saved by xdart must remain loadable. The old format used
xdart's int_1d_data_static/int_2d_data_static serialization. We need a migration
path — either a compatibility reader or a conversion utility.

### Step 1: Create adapter module in xdart

Create `xdart/utils/containers/compat.py` with:

```python
"""Backward-compatible readers for legacy xdart HDF5 files.

These functions read the old int_1d_data_static / int_2d_data_static format
and return ssrl_xrd_tools IntegrationResult1D / IntegrationResult2D objects.
"""
from ssrl_xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D

def read_legacy_1d(grp) -> IntegrationResult1D:
    """Read old int_1d_data_static HDF5 format, return IntegrationResult1D.
    The old format has: norm, ttheta, q (all as datasets).
    Prefer q if available, fall back to ttheta."""

def read_legacy_2d(grp) -> IntegrationResult2D:
    """Read old int_2d_data_static HDF5 format, return IntegrationResult2D.
    The old format has: i_qChi (or i_tthChi), chi, q (or ttheta).
    Also handles GI data: i_QxyQz, qz, qxy."""
```

### Step 2: Update xdart/utils/containers/__init__.py

Change from:
```python
from .poni import PONI, get_poni_dict
from .int_data_static import int_1d_data_static, int_2d_data_static
```

To:
```python
from ssrl_xrd_tools.core.containers import PONI, IntegrationResult1D, IntegrationResult2D
from .compat import read_legacy_1d, read_legacy_2d

# Backward compat aliases (deprecated, will be removed)
int_1d_data_static = IntegrationResult1D
int_2d_data_static = IntegrationResult2D
```

### Step 3: Update EwaldArch (xdart/modules/ewald/arch.py)

Replace all usage of int_1d_data_static / int_2d_data_static with the new types:
- `self.int_1d` type: IntegrationResult1D
- `self.int_2d` type: IntegrationResult2D
- Update `integrate_1d()` / `integrate_2d()` to store results directly
  (the ssrl_xrd_tools integrate functions already return the right types)
- Update `save_to_h5()` to use the new `to_hdf5()` method
- Update `load_from_h5()` to try new format first, fall back to `read_legacy_1d/2d`
- Replace xdart PONI usage with ssrl_xrd_tools PONI
- Remove `_make_lib_poni()` conversion since both sides now use the same PONI

### Step 4: Update EwaldSphere (xdart/modules/ewald/sphere.py)

Same container type updates as EwaldArch:
- `self.bai_1d` → IntegrationResult1D
- `self.bai_2d` → IntegrationResult2D
- Update `by_arch_integrate_*` and `multigeometry_integrate_*` methods
- The stitching functions already return ssrl_xrd_tools types

### Step 5: Update sphere_threads.py

Replace container imports and update any code that constructs or accesses
the old container attributes (e.g., `.norm` becomes `.intensity`,
`.ttheta`/`.q` becomes `.radial`).

### Step 6: Update display_frame_widget.py and h5viewer.py

These access container attributes for plotting. Key renames:
- `.norm` → `.intensity`
- `.ttheta` or `.q` → `.radial` (use `.unit` to determine what it is)
- `.chi` → `.azimuthal`
- `.i_qChi` or `.i_tthChi` → `.intensity` (it's the 2D intensity array)
- `.i_QxyQz` → `.intensity` (for GI results, unit tells you the axes)
- `.qz` / `.qxy` → `.azimuthal` / `.radial` (depending on convention)

The GUI should use `.unit` and `.azimuthal_unit` to determine axis labels.

### Step 7: Update spec_wrangler.py

Replace container construction with new types. This file already uses
ssrl_xrd_tools for I/O, so the changes should be straightforward.

### Step 8: Delete deprecated files

After all references are updated:
- Delete `xdart/utils/containers/int_data_static.py`
- Delete `xdart/utils/containers/poni.py`
- Keep `xdart/utils/containers/compat.py` (still needed for legacy file reading)

### Step 9: Delete xdart's lmfit_models.py

- Delete `xdart/utils/lmfit_models.py`
- Search for any imports of it in xdart and replace with
  `from ssrl_xrd_tools.analysis.fitting.models import ...`

### Testing

1. Run all existing ssrl_xrd_tools tests: `cd ssrl_xrd_tools && pytest`
2. Run all existing xdart tests: `cd xdart && pytest`
3. If you have a legacy .h5 file from xdart, verify it loads correctly
   with the new code (test the compat readers)
4. Verify integration round-trip: create EwaldArch with image + PONI,
   integrate, check that int_1d and int_2d have the right types and data

### Attribute mapping reference

| Old (xdart) | New (ssrl_xrd_tools) | Notes |
|---|---|---|
| int_1d_data_static | IntegrationResult1D | |
| int_2d_data_static | IntegrationResult2D | |
| .norm | .intensity | Primary signal |
| .ttheta | .radial (unit="2th_deg") | Check unit field |
| .q | .radial (unit="q_A^-1") | Check unit field |
| .chi | .azimuthal | |
| .i_tthChi | .intensity (unit="2th_deg") | 2D intensity |
| .i_qChi | .intensity (unit="q_A^-1") | 2D intensity |
| .i_QxyQz | .intensity (unit="qip_A^-1", azimuthal_unit="qoop_A^-1") | GI map |
| .qxy | .radial (GI) | With qip unit |
| .qz | .azimuthal (GI) | With qoop unit |
| .i_qz / .i_qxy | 1D: .intensity with qoop/qip unit | GI 1D results |
| PONI.detector (object) | PONI.detector (str) | Just the name now |
```

---

## Execution Notes

### Order
Run these prompts in order: 1a → 1b → 1c. Each builds on the previous.

### Branch strategy
- ssrl_xrd_tools: work on `refactor/container-unification`, merge into `dev` when Phase 1 passes tests
- xdart: work on `refactor/ssrl-integration`, merge into `refactor/ssrl-xrd-tools` when Phase 1 passes tests
- ssrl_xrd_tools `dev` → `main` only after the ENTIRE refactor is complete
- xdart `refactor/ssrl-xrd-tools` → `master` only after the ENTIRE refactor is complete
- For Phase 2, create new phase branches off the integration branches (e.g., `refactor/ewald-simplification`)

### Between prompts
After each prompt completes, verify:
1. `pytest` passes in the modified repo
2. No broken imports (run `python -c "import ssrl_xrd_tools"` or `python -c "import xdart"`)
3. Git commit the changes on the `dev` branch

### GUI testing
Phase 1c changes the GUI attribute names. After completing it, manually test:
1. Open xdart, load a scan, verify 1D/2D plots display correctly
2. Load a legacy .h5 file, verify backward compatibility
3. Test GI mode if you have GI data
