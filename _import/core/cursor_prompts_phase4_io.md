# Phase 4-IO: NeXus Reader, Tiled Reader, and Tests

Create two new reader modules (`io/nexus.py`, `io/tiled.py`) that convert
external data sources into `ScanMetadata` objects. Also update `io/__init__.py`
and add tests. Execute prompts **in order** (4A → 4B → 4C → 4D → 4E).

The key design principle: **all readers produce `ScanMetadata`** from
`ssrl_xrd_tools.core.metadata`. Processing code never knows the data source.

---

## Prompt 4A — `io/nexus.py` (NeXus/HDF5 reader)

Create `ssrl_xrd_tools/io/nexus.py`.

This module reads **NeXus-formatted HDF5 files** produced by Bluesky at SSRL
beamlines. Each file represents one scan and follows the NeXus/NXentry
hierarchy. The reader extracts metadata and returns a `ScanMetadata` instance.

### NeXus HDF5 Structure (Bluesky output)

Bluesky saves one HDF5 file per scan. The typical structure is:

```
/entry/                         (NXentry)
  @NX_class = "NXentry"
  instrument/                   (NXinstrument)
    @NX_class = "NXinstrument"
    source/                     (NXsource)
      energy                    # ring energy, not beam energy
    monochromator/              (NXmonochromator)
      energy                    # beam energy in keV (float or array)
      wavelength                # wavelength in Angstroms
    detector/                   (NXdetector)
      data                      # image stack: shape (nframes, ny, nx)
      <detector_name>/          # sometimes nested by detector name
        data
  sample/                       (NXsample)
    name                        # sample name string
    ub_matrix                   # 3x3 UB matrix (optional)
    orientation_matrix          # alternative name for UB (optional)
  data/                         (NXdata)
    <motor_name>                # 1D array per scanned motor
    <counter_name>              # 1D array per counter (i0, i1, etc.)
```

Not all fields will be present in every file. The reader must handle missing
fields gracefully.

### Functions to implement

```python
def read_nexus(
    path: Path | str,
    entry: str = "entry",
    motor_names: list[str] | None = None,
    counter_names: list[str] | None = None,
) -> ScanMetadata:
    """
    Read a NeXus/HDF5 file and return a ScanMetadata instance.

    Parameters
    ----------
    path : Path or str
        Path to the NeXus HDF5 file.
    entry : str, optional
        Name of the NXentry group (default ``"entry"``).
    motor_names : list of str, optional
        Motor names to extract from the data group. If None, extracts
        all 1D float datasets in ``/{entry}/data/`` that are not in
        counter_names.
    counter_names : list of str, optional
        Counter names to extract. If None, tries common names:
        ``["i0", "i1", "monitor", "det", "seconds"]``.

    Returns
    -------
    ScanMetadata

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If the entry group is not found in the file.
    """
```

```python
def find_nexus_image_dataset(
    path: Path | str,
    entry: str = "entry",
) -> str | None:
    """
    Return the HDF5 internal path to the image dataset, or None.

    Search order:
    1. /{entry}/instrument/detector/data
    2. /{entry}/data/data
    3. /{entry}/instrument/*/data  (any detector subgroup)
    4. Largest 3D dataset under /{entry}/
    """
```

```python
def list_entries(path: Path | str) -> list[str]:
    """
    List all NXentry group names in a NeXus file.

    Useful when a file contains multiple entries (e.g., multiple scans
    saved to one file).
    """
```

### Implementation details

- Import `h5py` at module level (it's already a core dependency).
- Import `ScanMetadata` from `ssrl_xrd_tools.core.metadata`.
- For `energy` extraction: look in `/{entry}/instrument/monochromator/energy`.
  If it's an array, take the first element. If not found, log a warning and
  use `np.nan`.
- For `wavelength`: look in `/{entry}/instrument/monochromator/wavelength`.
  If not found but energy is available, compute via `12.398 / energy`.
  Import `energy_to_wavelength` from `ssrl_xrd_tools.transforms` for this.
- For `ub_matrix`: check `/{entry}/sample/ub_matrix` then
  `/{entry}/sample/orientation_matrix`. Reshape to (3, 3) if found as flat
  array. Set to `None` if not found.
- For `sample_name`: check `/{entry}/sample/name`. Decode bytes to str if
  needed.
- For `scan_id`: use the file stem (e.g., `"scan_042"` from
  `"/path/to/scan_042.h5"`).
- Set `source = "nexus"`.
- Set `h5_path = Path(path)`.
- For angles/counters: iterate over datasets in `/{entry}/data/`. Classify
  as motor or counter based on the provided name lists.
- Auto-detection when name lists are None: any 1D float dataset in
  `/{entry}/data/` whose name appears in a known set of counter names goes
  to `counters`; everything else goes to `angles`.
- Known counter names (default): `{"i0", "i1", "i2", "monitor", "mon",
  "det", "diode", "seconds", "epoch", "time"}`.

### Conventions
- `from __future__ import annotations` at top
- `logger = logging.getLogger(__name__)`
- NumPy-style docstrings on all public functions
- `@dataclass(slots=True)` not needed here (no new dataclasses)
- Accept `Path | str`, convert internally

---

## Prompt 4B — `io/tiled.py` (Bluesky/Tiled catalog reader)

Create `ssrl_xrd_tools/io/tiled.py`.

This module reads scan data from a **Tiled server** (the Bluesky data access
layer). Tiled is an optional dependency — all imports must be guarded.

### Background

Tiled is a data access service for Bluesky. It provides a Python client
(`tiled.client`) that returns catalog-like objects. Each "run" in Tiled
corresponds to one scan and can be accessed by scan_id or uid.

### Functions to implement

```python
def read_tiled_run(
    client: Any,
    scan_id: str | int,
    motor_names: list[str] | None = None,
    counter_names: list[str] | None = None,
    stream: str = "primary",
) -> ScanMetadata:
    """
    Read a Bluesky run from a Tiled catalog and return ScanMetadata.

    Parameters
    ----------
    client : tiled.client.CatalogClient or similar
        Connected Tiled catalog client. Caller is responsible for
        authentication and connection setup.
    scan_id : str or int
        Scan identifier (uid string or integer scan_id).
    motor_names : list of str, optional
        Motor column names to extract. If None, uses the run's
        ``start.motors`` metadata field.
    counter_names : list of str, optional
        Counter names to extract. If None, uses the run's
        ``start.detectors`` metadata field.
    stream : str, optional
        Data stream name (default ``"primary"``).

    Returns
    -------
    ScanMetadata

    Raises
    ------
    ImportError
        If tiled is not installed.
    KeyError
        If the scan_id is not found in the catalog.
    """
```

```python
def connect_tiled(
    uri: str,
    api_key: str | None = None,
    **kwargs: Any,
) -> Any:
    """
    Create and return a Tiled catalog client.

    Thin wrapper around ``tiled.client.from_uri`` so callers don't need
    to import tiled directly.

    Parameters
    ----------
    uri : str
        Tiled server URI (e.g., ``"https://tiled.ssrl.slac.stanford.edu"``).
    api_key : str, optional
        API key for authentication.
    **kwargs
        Additional keyword arguments passed to ``tiled.client.from_uri``.

    Returns
    -------
    tiled.client.CatalogClient

    Raises
    ------
    ImportError
        If tiled is not installed.
    """
```

```python
def list_scans(
    client: Any,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """
    List recent scans in a Tiled catalog.

    Returns a list of dicts with keys: scan_id, uid, plan_name,
    sample_name (if available), and num_points.

    Parameters
    ----------
    client : tiled.client.CatalogClient
        Connected Tiled catalog.
    limit : int, optional
        Maximum number of scans to return (default 50).

    Returns
    -------
    list of dict
    """
```

### Implementation details

- Guard all tiled imports:
  ```python
  _HAS_TILED = False
  try:
      from tiled.client import from_uri as _tiled_from_uri
      _HAS_TILED = True
  except ImportError:
      pass
  ```
- Each public function that needs tiled should check `_HAS_TILED` and raise
  `ImportError("tiled is required: pip install tiled[client]")` if not
  available.
- `read_tiled_run` implementation:
  1. Look up the run: `run = client[scan_id]`
  2. Extract start document: `start = run.metadata["start"]`
  3. Energy: `start.get("energy")` or look in the data stream
  4. Wavelength: `start.get("wavelength")` or compute from energy
  5. UB matrix: `start.get("ub_matrix")` — reshape (3,3) if present
  6. Sample name: `start.get("sample_name", "")`
  7. Motor names (if None): `start.get("motors", [])`
  8. Counter names (if None): `start.get("detectors", [])`
  9. Read data stream: `ds = run[stream].read()` — returns an xarray Dataset
  10. Extract arrays: `ds[name].values` for each motor/counter
  11. Set `source = "tiled"`, `scan_id = str(scan_id)`
  12. Set `h5_path = None` (data comes from network, not a local file)
- `connect_tiled`: just wraps `_tiled_from_uri(uri, api_key=api_key, **kwargs)`
- `list_scans`: iterate over `client.values()` (or `client.items()`),
  extract metadata from each run's start document, return list of dicts.

### Conventions
- Same as 4A: `from __future__ import annotations`, logging, NumPy docstrings
- Type-hint `client` as `Any` since tiled may not be installed
- Import `ScanMetadata` from `ssrl_xrd_tools.core.metadata`
- Import `energy_to_wavelength` from `ssrl_xrd_tools.transforms` when needed

---

## Prompt 4C — Update `io/__init__.py` (add new exports)

Update `ssrl_xrd_tools/io/__init__.py` to add exports from the new modules.

Add these imports:

```python
from ssrl_xrd_tools.io.nexus import (
    find_nexus_image_dataset,
    list_entries,
    read_nexus,
)
```

For tiled, use a conditional import since tiled is optional:

```python
try:
    from ssrl_xrd_tools.io.tiled import (
        connect_tiled,
        list_scans,
        read_tiled_run,
    )
except ImportError:
    pass
```

Keep all existing exports intact. Don't remove anything.

---

## Prompt 4D — Tests

Create `tests/test_nexus.py` and `tests/test_tiled.py`.

### `tests/test_nexus.py`

```python
import pytest
import numpy as np
from pathlib import Path
import h5py
from ssrl_xrd_tools.io.nexus import read_nexus, find_nexus_image_dataset, list_entries
from ssrl_xrd_tools.core.metadata import ScanMetadata
```

**Fixtures:**

```python
@pytest.fixture
def nexus_file(tmp_path):
    """Create a minimal NeXus HDF5 file for testing."""
    p = tmp_path / "scan_001.h5"
    with h5py.File(p, "w") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"

        inst = entry.create_group("instrument")
        mono = inst.create_group("monochromator")
        mono.create_dataset("energy", data=12.0)
        mono.create_dataset("wavelength", data=1.033)

        det = inst.create_group("detector")
        det.create_dataset("data", data=np.random.default_rng(0).random((5, 20, 30)))

        sample = entry.create_group("sample")
        sample.create_dataset("name", data="test_sample")
        sample.create_dataset("ub_matrix", data=np.eye(3).flatten())

        data = entry.create_group("data")
        data.create_dataset("th", data=np.linspace(10, 20, 5))
        data.create_dataset("tth", data=np.linspace(20, 40, 5))
        data.create_dataset("i0", data=np.ones(5) * 1e5)
        data.create_dataset("i1", data=np.ones(5) * 500)
        data.create_dataset("seconds", data=np.ones(5) * 1.0)
    return p
```

**Test classes:**

1. **TestReadNexus**
   - `test_returns_scan_metadata(nexus_file)` — verify returns ScanMetadata
   - `test_energy(nexus_file)` — verify `energy == 12.0`
   - `test_wavelength(nexus_file)` — verify `wavelength ≈ 1.033`
   - `test_ub_matrix(nexus_file)` — verify shape (3,3), values match eye(3)
   - `test_sample_name(nexus_file)` — verify `"test_sample"`
   - `test_source(nexus_file)` — verify `source == "nexus"`
   - `test_scan_id(nexus_file)` — verify derived from filename stem
   - `test_h5_path(nexus_file)` — verify `h5_path == nexus_file`
   - `test_angles(nexus_file)` — verify `"th"` and `"tth"` in angles dict
     with correct shapes
   - `test_counters(nexus_file)` — verify `"i0"` in counters dict
   - `test_custom_motor_names(nexus_file)` — pass `motor_names=["th"]`,
     verify only "th" in angles
   - `test_missing_entry` — try reading with `entry="nonexistent"`, expect
     KeyError
   - `test_file_not_found` — expect FileNotFoundError

2. **TestFindNexusImageDataset**
   - `test_finds_detector_data(nexus_file)` — verify returns path string
     containing "detector/data"
   - `test_returns_none_no_images(tmp_path)` — create HDF5 with no image
     data, verify returns None

3. **TestListEntries**
   - `test_single_entry(nexus_file)` — verify returns `["entry"]`
   - `test_multiple_entries(tmp_path)` — create file with "entry_1" and
     "entry_2" groups with NX_class attrs, verify both returned

4. **TestReadNexusMissingFields**
   - `test_missing_energy(tmp_path)` — create minimal file without energy,
     verify returns ScanMetadata with `energy == np.nan` (or computed from
     wavelength if wavelength is present)
   - `test_missing_ub_matrix(tmp_path)` — create file without UB, verify
     `ub_matrix is None`
   - `test_missing_sample_name(tmp_path)` — verify `sample_name == ""`

### `tests/test_tiled.py`

Since tiled is an optional dependency that likely won't be installed in CI,
all tests should use mocks.

```python
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from ssrl_xrd_tools.core.metadata import ScanMetadata

try:
    from ssrl_xrd_tools.io.tiled import read_tiled_run, connect_tiled, list_scans, _HAS_TILED
except ImportError:
    _HAS_TILED = False
    pytestmark = pytest.mark.skip(reason="tiled module not available")
```

**Fixtures:**

```python
@pytest.fixture
def mock_run():
    """Create a mock Tiled run object."""
    run = MagicMock()
    run.metadata = {
        "start": {
            "scan_id": 42,
            "energy": 12.0,
            "wavelength": 1.033,
            "sample_name": "test_sample",
            "motors": ["th", "tth"],
            "detectors": ["i0", "i1"],
            "plan_name": "scan",
        }
    }
    # Mock the data stream (xarray-like Dataset)
    ds = MagicMock()
    ds.__getitem__ = lambda self, key: MagicMock(
        values=np.linspace(0, 1, 10)
    )
    run.__getitem__ = lambda self, key: MagicMock(read=lambda: ds)
    return run


@pytest.fixture
def mock_client(mock_run):
    """Create a mock Tiled catalog client."""
    client = MagicMock()
    client.__getitem__ = lambda self, key: mock_run
    client.values = lambda: [mock_run]
    client.items = lambda: [("uid_abc", mock_run)]
    return client
```

**Test classes:**

1. **TestReadTiledRun**
   - `test_returns_scan_metadata(mock_client)` — verify returns ScanMetadata
   - `test_energy(mock_client)` — verify energy extracted from start doc
   - `test_source(mock_client)` — verify `source == "tiled"`
   - `test_custom_motor_names(mock_client)` — override motor_names, verify
     only those appear in angles

2. **TestConnectTiled**
   - `test_raises_without_tiled` — monkeypatch `_HAS_TILED = False`, verify
     ImportError

3. **TestListScans**
   - `test_returns_list(mock_client)` — verify returns list of dicts with
     expected keys

---

## Prompt 4E — Update CLAUDE.md and pyproject.toml

1. In `CLAUDE.md`, update the module map:
```
├── io/
│   ├── image.py         # ✅ detector-agnostic image I/O via fabio + HDF5
│   ├── spec.py          # ✅ SPEC file parsing via silx
│   ├── nexus.py         # ✅ NeXus/HDF5 reader (Bluesky output files)
│   ├── tiled.py         # ✅ Bluesky/Tiled catalog reader
│   ├── export.py        # ✅ write_xye, write_csv, write_h5
│   └── metadata.py      # STUB: txt/pdi/log metadata readers
```

2. Update the "Working code" list to add:
```
- `io/nexus.py` — read_nexus, find_nexus_image_dataset, list_entries (NeXus/HDF5 reader)
- `io/tiled.py` — read_tiled_run, connect_tiled, list_scans (Bluesky/Tiled reader, optional dep)
```

3. Remove `io/nexus.py` and `io/tiled.py` from the "Planned" references
   in section 3 ("Source-agnostic metadata"). Update the bullets to show
   they are implemented:
```
- `io/spec.py` — reads from SPEC files (current, via silx)
- `io/nexus.py` — reads from NeXus/HDF5 files (Bluesky output)
- `io/tiled.py` — reads from Tiled catalog (Bluesky database)
```
   (Remove the "planned" annotations.)

4. In `pyproject.toml`, add tiled as an optional dependency:
```toml
[project.optional-dependencies]
tiled = ["tiled[client]"]
```
   Keep `h5py` in the core dependencies (it's already there). Do NOT add
   tiled to core dependencies.
