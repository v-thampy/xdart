"""
NeXus/HDF5 reader and writer for SSRL beamline scan files.

**Reading** (``read_nexus``, ``find_nexus_image_dataset``, ``list_entries``):
Each file represents one scan and follows the NXentry hierarchy written by
Bluesky's ``suitcase-nexus`` or equivalent ophyd/databroker exporter.

**Writing** (``write_nexus``, ``open_nexus_writer``, ``write_nexus_frame``):
Produces self-describing NeXus-formatted HDF5 files from processed integration
results.  Designed to replace the custom HDF5 codec in ``xdart`` for new
workflows.  Layout::

    /{entry}/                       NXentry
        scan_id, source             (attrs)
        instrument/monochromator/   NXmonochromator — energy, wavelength
        sample/                     NXsample — name, ub_matrix
        data/                       NXdata   — per-point motor & counter arrays
        reduction/                  NXprocess
            {frame}/
                int_1d/             NXdata — radial, intensity, [sigma]
                int_2d/             NXdata — radial, azimuthal, intensity, [sigma]

Performance features:
- Chunked datasets for efficient frame-by-frame writes
- Optional SWMR mode for live beamline reduction (GUI reads while processing
  writes)
- Single file open/close per scan via ``write_nexus_frame`` for hot loops
- LZF compression by default (fast, suitable for live use)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ssrl_xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from ssrl_xrd_tools.core.metadata import ScanMetadata
from ssrl_xrd_tools.transforms import energy_to_wavelength

logger = logging.getLogger(__name__)

# Datasets that are counters/scalers rather than motor angles.
_DEFAULT_COUNTER_NAMES: frozenset[str] = frozenset(
    {"i0", "i1", "i2", "monitor", "mon", "det", "diode", "seconds", "epoch", "time"}
)


# ---------------------------------------------------------------------------
# Read API
# ---------------------------------------------------------------------------

def read_nexus(
    path: Path | str,
    entry: str = "entry",
    motor_names: list[str] | None = None,
    counter_names: list[str] | None = None,
) -> ScanMetadata:
    """
    Read a NeXus/HDF5 file and return a ``ScanMetadata`` instance.

    Parameters
    ----------
    path : Path or str
        Path to the NeXus HDF5 file.
    entry : str, optional
        Name of the NXentry group (default ``"entry"``).
    motor_names : list of str, optional
        Motor names to extract from ``/{entry}/data/``.  If *None*, all 1D
        float datasets whose names are **not** in ``counter_names`` are
        treated as motors.
    counter_names : list of str, optional
        Counter names to extract.  If *None*, tries the built-in set:
        ``{"i0", "i1", "i2", "monitor", "mon", "det", "diode", "seconds",
        "epoch", "time"}``.

    Returns
    -------
    ScanMetadata

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    KeyError
        If the ``entry`` group is not found in the file.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"NeXus file not found: {p}")

    with h5py.File(p, "r") as f:
        if entry not in f:
            raise KeyError(
                f"Entry group {entry!r} not found in {p}. "
                f"Available top-level keys: {list(f.keys())}"
            )
        grp = f[entry]

        scan_id = p.stem
        energy = _read_energy(grp)
        wavelength = _read_wavelength(grp, energy)
        ub_matrix = _read_ub_matrix(grp)
        sample_name = _read_sample_name(grp)
        angles, counters = _read_data_group(grp, motor_names, counter_names)

    return ScanMetadata(
        scan_id=scan_id,
        energy=energy,
        wavelength=wavelength,
        angles=angles,
        counters=counters,
        ub_matrix=ub_matrix,
        sample_name=sample_name,
        source="nexus",
        h5_path=p,
    )


def find_nexus_image_dataset(
    path: Path | str,
    entry: str = "entry",
) -> str | None:
    """
    Return the HDF5 internal path to the image dataset, or *None*.

    Search order:

    1. ``/{entry}/instrument/detector/data``
    2. ``/{entry}/data/data``
    3. ``/{entry}/instrument/*/data``  (any detector sub-group)
    4. Largest 3D dataset anywhere under ``/{entry}/``

    Parameters
    ----------
    path : Path or str
        Path to the NeXus HDF5 file.
    entry : str, optional
        Name of the NXentry group (default ``"entry"``).

    Returns
    -------
    str or None
        HDF5 internal path (e.g. ``"/entry/instrument/detector/data"``) or
        *None* if no suitable dataset is found.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"NeXus file not found: {p}")

    with h5py.File(p, "r") as f:
        if entry not in f:
            logger.warning("Entry %r not found in %s", entry, p)
            return None
        grp = f[entry]

        # 1. Canonical detector location
        candidate = f"{entry}/instrument/detector/data"
        if candidate in f and isinstance(f[candidate], h5py.Dataset) and f[candidate].ndim == 3:
            return f"/{candidate}"

        # 2. /data/data
        candidate = f"{entry}/data/data"
        if candidate in f and isinstance(f[candidate], h5py.Dataset) and f[candidate].ndim == 3:
            return f"/{candidate}"

        # 3. Any detector sub-group under instrument/
        if "instrument" in grp:
            instr = grp["instrument"]
            for subname in instr:
                sub = instr[subname]
                if not isinstance(sub, h5py.Group):
                    continue
                inner = f"{entry}/instrument/{subname}/data"
                if inner in f and isinstance(f[inner], h5py.Dataset) and f[inner].ndim == 3:
                    return f"/{inner}"

        # 4. Fallback: largest 3D dataset anywhere under the entry
        best_path: str | None = None
        best_size = 0

        def _visit(name: str, obj: Any) -> None:
            nonlocal best_path, best_size
            if not isinstance(obj, h5py.Dataset) or obj.ndim != 3:
                return
            size = int(np.prod(obj.shape))
            if size > best_size:
                best_size = size
                best_path = f"/{entry}/{name}"

        grp.visititems(_visit)
        if best_path:
            logger.debug("Found image dataset by fallback scan: %s", best_path)
        return best_path


def list_entries(path: Path | str) -> list[str]:
    """
    List all NXentry group names in a NeXus file.

    Useful when a file contains multiple entries (e.g., multiple scans saved
    to one file).

    Parameters
    ----------
    path : Path or str
        Path to the NeXus HDF5 file.

    Returns
    -------
    list of str
        Names of all top-level groups whose ``NX_class`` attribute equals
        ``"NXentry"``, or — as a fallback — all top-level groups.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"NeXus file not found: {p}")

    with h5py.File(p, "r") as f:
        entries = [
            name
            for name, obj in f.items()
            if isinstance(obj, h5py.Group) and _nx_class(obj) == "NXentry"
        ]
        if not entries:
            # Graceful fallback: return all top-level groups
            entries = [name for name, obj in f.items() if isinstance(obj, h5py.Group)]
    return entries


# ---------------------------------------------------------------------------
# Write API — high-level: write a complete scan result in one call
# ---------------------------------------------------------------------------

def write_nexus(
    path: Path | str,
    metadata: ScanMetadata | None = None,
    results_1d: dict[int | str, IntegrationResult1D] | None = None,
    results_2d: dict[int | str, IntegrationResult2D] | None = None,
    entry: str = "entry",
    compression: str | None = "lzf",
    overwrite: bool = False,
) -> Path:
    """
    Write processed integration results to a NeXus-formatted HDF5 file.

    Creates a self-describing file with optional raw metadata (NXentry/
    NXinstrument) and per-frame processed results (NXprocess/NXdata).

    Parameters
    ----------
    path : Path or str
        Output file path.
    metadata : ScanMetadata, optional
        Scan metadata to write into the NXentry header. If None, only
        processed results are written.
    results_1d : dict of {frame: IntegrationResult1D}, optional
        Per-frame 1D integration results. Keys are frame indices or labels.
    results_2d : dict of {frame: IntegrationResult2D}, optional
        Per-frame 2D (cake) integration results.
    entry : str, optional
        NXentry group name (default ``"entry"``).
    compression : str or None, optional
        HDF5 compression filter. ``"lzf"`` (default) is fast for live use;
        ``"gzip"`` gives better compression for archival. None disables
        compression.
    overwrite : bool, optional
        If True, overwrite existing file. If False (default), append/update.

    Returns
    -------
    Path
        The output file path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "a"

    comp_kwargs = _comp_kwargs(compression)

    with h5py.File(p, mode) as f:
        grp = f.require_group(entry)
        grp.attrs["NX_class"] = "NXentry"

        if metadata is not None:
            _write_metadata(grp, metadata, comp_kwargs)

        proc = grp.require_group("reduction")
        proc.attrs["NX_class"] = "NXprocess"
        proc.attrs.setdefault("program", "ssrl_xrd_tools")

        if results_1d:
            for frame_key, r1d in results_1d.items():
                _write_result_1d(proc, str(frame_key), r1d, comp_kwargs)

        if results_2d:
            for frame_key, r2d in results_2d.items():
                _write_result_2d(proc, str(frame_key), r2d, comp_kwargs)

    logger.debug("Wrote NeXus file: %s", p)
    return p


# ---------------------------------------------------------------------------
# Write API — frame-by-frame: for live reduction hot loops
# ---------------------------------------------------------------------------

def open_nexus_writer(
    path: Path | str,
    metadata: ScanMetadata | None = None,
    entry: str = "entry",
    compression: str | None = "lzf",
    swmr: bool = False,
    overwrite: bool = False,
) -> h5py.File:
    """
    Open an HDF5 file for incremental NeXus writing.

    Use this instead of :func:`write_nexus` when writing frame-by-frame
    in a live reduction loop. Returns an open ``h5py.File`` that the
    caller must close (or use as a context manager).

    Parameters
    ----------
    path : Path or str
        Output file path.
    metadata : ScanMetadata, optional
        Scan metadata to write into the header on first open.
    entry : str, optional
        NXentry group name.
    compression : str or None, optional
        Compression filter.
    swmr : bool, optional
        Enable single-writer-multiple-reader mode. When True, readers
        (e.g. the GUI) can open the file while the writer is active.
        Requires ``libver='latest'``.
    overwrite : bool, optional
        If True, overwrite existing file.

    Returns
    -------
    h5py.File
        Open file handle. Caller is responsible for closing.

    Examples
    --------
    ::

        h5 = open_nexus_writer("scan_001.h5", metadata=meta, swmr=True)
        try:
            for i, (r1d, r2d) in enumerate(process_frames(...)):
                write_nexus_frame(h5, i, result_1d=r1d, result_2d=r2d)
                h5.flush()  # makes data visible to SWMR readers
        finally:
            h5.close()
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    file_kwargs: dict[str, Any] = {}
    if swmr:
        file_kwargs["libver"] = "latest"

    mode = "w" if overwrite else "a"
    f = h5py.File(p, mode, **file_kwargs)

    ck = _comp_kwargs(compression)

    grp = f.require_group(entry)
    grp.attrs["NX_class"] = "NXentry"

    if metadata is not None:
        _write_metadata(grp, metadata, ck)

    proc = grp.require_group("reduction")
    proc.attrs["NX_class"] = "NXprocess"
    proc.attrs.setdefault("program", "ssrl_xrd_tools")

    if swmr:
        f.swmr_mode = True

    return f


def write_nexus_frame(
    h5: h5py.File,
    frame: int | str,
    result_1d: IntegrationResult1D | None = None,
    result_2d: IntegrationResult2D | None = None,
    entry: str = "entry",
    compression: str | None = "lzf",
) -> None:
    """
    Write a single frame's results to an already-open NeXus file.

    Designed for use with :func:`open_nexus_writer` in live reduction
    loops. Call ``h5.flush()`` after each frame to make data visible
    to SWMR readers.

    Parameters
    ----------
    h5 : h5py.File
        Open file handle from :func:`open_nexus_writer`.
    frame : int or str
        Frame index or label.
    result_1d : IntegrationResult1D, optional
    result_2d : IntegrationResult2D, optional
    entry : str, optional
        NXentry group name.
    compression : str or None, optional
        Compression filter.
    """
    ck = _comp_kwargs(compression)
    proc = h5[entry].require_group("reduction")
    frame_str = str(frame)

    if result_1d is not None:
        _write_result_1d(proc, frame_str, result_1d, ck)

    if result_2d is not None:
        _write_result_2d(proc, frame_str, result_2d, ck)


# ---------------------------------------------------------------------------
# Private helpers — compression
# ---------------------------------------------------------------------------

def _comp_kwargs(compression: str | None) -> dict[str, Any]:
    """Build h5py dataset kwargs for a given compression filter."""
    if compression is None:
        return {}
    ck: dict[str, Any] = {"compression": compression}
    if compression == "gzip":
        ck["shuffle"] = True
    return ck


# ---------------------------------------------------------------------------
# Private helpers — metadata
# ---------------------------------------------------------------------------

def _write_metadata(
    entry_grp: h5py.Group,
    meta: ScanMetadata,
    comp_kwargs: dict[str, Any],
) -> None:
    """Write ScanMetadata into NXentry subgroups."""
    inst = entry_grp.require_group("instrument")
    inst.attrs["NX_class"] = "NXinstrument"
    mono = inst.require_group("monochromator")
    mono.attrs["NX_class"] = "NXmonochromator"
    _replace(mono, "energy", np.float64(meta.energy))
    _replace(mono, "wavelength", np.float64(meta.wavelength))

    sample = entry_grp.require_group("sample")
    sample.attrs["NX_class"] = "NXsample"
    if meta.sample_name:
        _replace(sample, "name", meta.sample_name)
    if meta.ub_matrix is not None:
        _replace(sample, "ub_matrix", meta.ub_matrix, **comp_kwargs)

    data = entry_grp.require_group("data")
    data.attrs["NX_class"] = "NXdata"
    for name, arr in meta.angles.items():
        _replace(data, name, arr, **comp_kwargs)
    for name, arr in meta.counters.items():
        _replace(data, name, arr, **comp_kwargs)

    entry_grp.attrs["scan_id"] = meta.scan_id
    if meta.source:
        entry_grp.attrs["source"] = meta.source


# ---------------------------------------------------------------------------
# Private helpers — results
# ---------------------------------------------------------------------------

def _write_result_1d(
    proc_grp: h5py.Group,
    frame_key: str,
    r: IntegrationResult1D,
    comp_kwargs: dict[str, Any],
) -> None:
    """Write one IntegrationResult1D into /{entry}/reduction/{frame}/int_1d/."""
    frame_grp = proc_grp.require_group(frame_key)
    grp = frame_grp.require_group("int_1d")
    grp.attrs["NX_class"] = "NXdata"
    grp.attrs["signal"] = "intensity"
    grp.attrs["axes"] = ["radial"]

    _replace(grp, "radial", r.radial, **comp_kwargs)
    _replace(grp, "intensity", r.intensity, **comp_kwargs)
    if r.sigma is not None:
        _replace(grp, "sigma", r.sigma, **comp_kwargs)
    grp.attrs["unit"] = r.unit


def _write_result_2d(
    proc_grp: h5py.Group,
    frame_key: str,
    r: IntegrationResult2D,
    comp_kwargs: dict[str, Any],
) -> None:
    """Write one IntegrationResult2D into /{entry}/reduction/{frame}/int_2d/."""
    frame_grp = proc_grp.require_group(frame_key)
    grp = frame_grp.require_group("int_2d")
    grp.attrs["NX_class"] = "NXdata"
    grp.attrs["signal"] = "intensity"
    grp.attrs["axes"] = ["radial", "azimuthal"]

    _replace(grp, "radial", r.radial, **comp_kwargs)
    _replace(grp, "azimuthal", r.azimuthal, **comp_kwargs)

    chunks = (r.intensity.shape[0], min(r.intensity.shape[1], 64))
    _replace(grp, "intensity", r.intensity, chunks=chunks, **comp_kwargs)
    if r.sigma is not None:
        _replace(grp, "sigma", r.sigma, chunks=chunks, **comp_kwargs)
    grp.attrs["unit"] = r.unit


# ---------------------------------------------------------------------------
# Private helpers — HDF5 utilities
# ---------------------------------------------------------------------------

def _replace(group: h5py.Group, name: str, data: Any, **kwargs: Any) -> None:
    """Delete-and-recreate a dataset to avoid shape/type conflicts on update."""
    if name in group:
        del group[name]
    group.create_dataset(name, data=data, **kwargs)


def _nx_class(obj: h5py.Group) -> str:
    """Return the NX_class attribute of an HDF5 group as a plain str."""
    raw = obj.attrs.get("NX_class", "")
    if isinstance(raw, (bytes, np.bytes_)):
        return raw.decode("utf-8", errors="replace")
    return str(raw) if raw else ""


def _scalar_or_first(ds: h5py.Dataset) -> float:
    """Return a scalar float from a scalar or 1-D dataset."""
    arr = ds[()]
    if np.ndim(arr) == 0:
        return float(arr)
    return float(np.asarray(arr).ravel()[0])


def _read_energy(grp: h5py.Group) -> float:
    """Extract beam energy (keV) from the monochromator group."""
    path = "instrument/monochromator/energy"
    if path in grp:
        try:
            return _scalar_or_first(grp[path])
        except Exception:
            logger.warning("Could not read energy from %s", path, exc_info=True)
    logger.warning("Energy not found in NeXus file; using NaN")
    return float(np.nan)


def _read_wavelength(grp: h5py.Group, energy: float) -> float:
    """Extract wavelength (Å) or derive from energy."""
    path = "instrument/monochromator/wavelength"
    if path in grp:
        try:
            return _scalar_or_first(grp[path])
        except Exception:
            logger.warning("Could not read wavelength from %s", path, exc_info=True)
    if np.isfinite(energy) and energy > 0:
        return float(energy_to_wavelength(energy))
    logger.warning("Wavelength not derivable; using NaN")
    return float(np.nan)


def _read_ub_matrix(grp: h5py.Group) -> np.ndarray | None:
    """Extract the 3×3 UB matrix from the sample group, or None."""
    for path in ("sample/ub_matrix", "sample/orientation_matrix"):
        if path in grp:
            try:
                arr = np.asarray(grp[path], dtype=float)
                return arr.reshape(3, 3)
            except Exception:
                logger.warning("Could not parse UB matrix from %s", path, exc_info=True)
    return None


def _read_sample_name(grp: h5py.Group) -> str:
    """Extract the sample name string, decoding bytes if necessary."""
    path = "sample/name"
    if path not in grp:
        return ""
    try:
        raw = grp[path][()]
        if isinstance(raw, (bytes, np.bytes_)):
            return raw.decode("utf-8", errors="replace")
        if isinstance(raw, np.ndarray):
            item = raw.ravel()[0]
            return item.decode("utf-8", errors="replace") if isinstance(item, (bytes, np.bytes_)) else str(item)
        return str(raw)
    except Exception:
        logger.warning("Could not read sample name", exc_info=True)
        return ""


def _read_data_group(
    grp: h5py.Group,
    motor_names: list[str] | None,
    counter_names: list[str] | None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Extract per-point arrays from ``/{entry}/data/``.

    Returns
    -------
    angles : dict[str, np.ndarray]
    counters : dict[str, np.ndarray]
    """
    angles: dict[str, np.ndarray] = {}
    counters: dict[str, np.ndarray] = {}

    if "data" not in grp:
        return angles, counters

    data_grp = grp["data"]
    if not isinstance(data_grp, h5py.Group):
        return angles, counters

    counter_set: frozenset[str]
    if counter_names is not None:
        counter_set = frozenset(counter_names)
    else:
        counter_set = _DEFAULT_COUNTER_NAMES

    motor_set: frozenset[str] | None = frozenset(motor_names) if motor_names is not None else None

    for name, obj in data_grp.items():
        if not isinstance(obj, h5py.Dataset):
            continue
        arr = np.asarray(obj, dtype=float)
        if arr.ndim != 1:
            continue

        if motor_set is not None:
            if name in motor_set:
                angles[name] = arr
            elif name in counter_set:
                counters[name] = arr
        else:
            if name in counter_set:
                counters[name] = arr
            else:
                angles[name] = arr

    return angles, counters


__all__ = [
    # Read
    "find_nexus_image_dataset",
    "list_entries",
    "read_nexus",
    # Write
    "open_nexus_writer",
    "write_nexus",
    "write_nexus_frame",
]
