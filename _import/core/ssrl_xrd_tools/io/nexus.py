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

        # 4. Eiger external-link pattern: /entry/data/data_NNNNNN
        #    Eiger master files store external links named data_000001,
        #    data_000002, … that point to _data_*.h5 files.  Each link
        #    resolves to a 3D dataset.  Return the *first* link path so
        #    the caller can enumerate siblings to build the full stack.
        if "data" in grp and isinstance(grp["data"], h5py.Group):
            data_grp = grp["data"]
            ext_keys = sorted(
                k for k in data_grp
                if data_grp.get(k, getlink=True).__class__.__name__ == "ExternalLink"
            )
            if ext_keys:
                first_key = ext_keys[0]
                try:
                    ds = data_grp[first_key]
                    if isinstance(ds, h5py.Dataset) and ds.ndim == 3:
                        logger.debug(
                            "Found Eiger external-link dataset: /%s/data/%s",
                            entry, first_key,
                        )
                        return f"/{entry}/data/{first_key}"
                except Exception:
                    # External target file may be missing
                    pass

        # 5. Fallback: largest 3D dataset anywhere under the entry
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

        h5 = open_nexus_writer("scan_001.nxs", metadata=meta, swmr=True)
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
    grp.attrs["azimuthal_unit"] = r.azimuthal_unit


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


# ===========================================================================
# v2 schema reader (xdart 0.37+) + v1 backcompat
# ---------------------------------------------------------------------------
# See xdart/docs/nexus_stitch_refactor_plan.md §2 for the v2 layout.
# read_sphere() auto-detects v1 vs v2 and returns a single canonical
# xarray.Dataset shape in either case, so downstream analysis code
# (BatchPhaseFitter, viewer, notebooks) never has to branch on schema.
#
# Canonical Dataset shape:
#   dims:   frame, q, chi          (chi/q only when corresponding stack loaded)
#   vars:   intensity_1d   (frame, q)
#           sigma_1d       (frame, q)         optional
#           intensity_2d   (frame, chi, q)
#           rot1, rot2, rot3, incident_angle  (frame,)   v2 only
#           <each scan motor>                 (frame,)
#   coords: frame, q, chi
#   attrs:  reduction (provenance dict — empty {} for v1 files)
#           schema_version  "v1" or "v2"
# ===========================================================================

def _v2_decode_str(v):
    return v.decode("utf-8") if isinstance(v, bytes) else v


def _read_v1_or_v2(entry: h5py.Group) -> str:
    """Detect schema version from an open NXentry group.

    Decision rules, in order:

    1. If ``entry.attrs["type"]`` == ``"EwaldSphere"`` → v1 (xdart's v1
       writer stamps that attribute).
    2. If ``entry/integrated_1d/intensity.ndim`` == 2 → v2 (stacked
       (N, nq) tensor); if 1 → v1 (single summed pattern).
    3. If ``entry/frames`` contains a child matching the v1 naming
       pattern ``NNNN`` (digit-only) → v1; ``frame_NNNN`` (prefixed) → v2.
    4. Default to v2.
    """
    type_attr = entry.attrs.get("type", b"")
    if isinstance(type_attr, bytes):
        type_attr = type_attr.decode("utf-8", errors="replace")
    if type_attr == "EwaldSphere":
        return "v1"
    if "integrated_1d" in entry and "intensity" in entry["integrated_1d"]:
        ndim = entry["integrated_1d"]["intensity"].ndim
        if ndim == 2:
            return "v2"
        if ndim == 1:
            return "v1"
    if "frames" in entry:
        for name in entry["frames"]:
            if name.startswith("frame_"):
                return "v2"
            if name.isdigit():
                return "v1"
    return "v2"


def _read_positioners(grp: h5py.Group) -> dict[str, np.ndarray]:
    """Read NXpositioner children of an NXcollection into a dict (v2)."""
    out: dict[str, np.ndarray] = {}
    for k, item in grp.items():
        if isinstance(item, h5py.Group):
            if "value" in item:
                out[k] = np.asarray(item["value"][()])
        elif isinstance(item, h5py.Dataset):
            out[k] = np.asarray(item[()])
    return out


def _read_v1_scan_data(entry: h5py.Group) -> dict[str, np.ndarray]:
    """Read v1's ``entry/scan_data/<column>`` datasets into a dict."""
    out: dict[str, np.ndarray] = {}
    if "scan_data" not in entry:
        return out
    sd = entry["scan_data"]
    for k, v in sd.items():
        if isinstance(v, h5py.Dataset):
            arr = v[()]
            out[k] = np.asarray(arr)
    return out


def _read_sphere_v1(path: Path, entry: str, groups: tuple[str, ...],
                   include_thumbnails: bool):
    """v1-schema reader.  See public ``read_sphere`` for the contract."""
    import xarray as xr

    data_vars: dict[str, tuple] = {}
    coords: dict[str, np.ndarray] = {}
    attrs_per_coord: dict[str, dict] = {}

    with h5py.File(path, "r") as f:
        if entry not in f:
            raise KeyError(f"No {entry!r} group in {path}")
        e = f[entry]

        # ── enumerate per-frame groups ────────────────────────────
        if "frames" not in e:
            raise KeyError(f"No frames/ in {path}:{entry}; not an xdart v1 file")
        frames_grp = e["frames"]

        frame_indices: list[int] = sorted(
            int(name) for name in frames_grp
            if name.isdigit() and name in frames_grp
            and isinstance(frames_grp[name], h5py.Group)
        )
        if not frame_indices:
            raise KeyError(
                f"v1 file {path} has no digit-named frame groups under frames/"
            )

        # ── 1D stack ──────────────────────────────────────────────
        if "1d" in groups:
            i1: list[np.ndarray] = []
            s1: list[np.ndarray] = []
            radial_1d: np.ndarray | None = None
            unit_1d = ""
            for idx in frame_indices:
                key = f"{idx:04d}"
                fg = frames_grp[key]
                if "intensity" not in fg:
                    continue
                i1.append(np.asarray(fg["intensity"][()], dtype=np.float32))
                if radial_1d is None and "radial" in fg:
                    radial_1d = np.asarray(fg["radial"][()], dtype=np.float32)
                    u = fg["radial"].attrs.get("units", b"")
                    unit_1d = _v2_decode_str(u) if u else ""
                if "sigma" in fg:
                    s1.append(np.asarray(fg["sigma"][()], dtype=np.float32))
            if i1:
                data_vars["intensity_1d"] = (
                    ("frame", "q"), np.stack(i1, axis=0)
                )
                if radial_1d is not None:
                    coords["q"] = radial_1d
                    if unit_1d:
                        attrs_per_coord["q"] = {"units": unit_1d}
                if s1 and len(s1) == len(i1):
                    data_vars["sigma_1d"] = (
                        ("frame", "q"), np.stack(s1, axis=0)
                    )

        # ── 2D stack ──────────────────────────────────────────────
        if "2d" in groups:
            i2: list[np.ndarray] = []
            radial_2d: np.ndarray | None = None
            azim_2d: np.ndarray | None = None
            azim_unit = ""
            for idx in frame_indices:
                key2 = f"{idx:04d}_2d"
                if key2 not in frames_grp:
                    continue
                fg2 = frames_grp[key2]
                if "intensity" not in fg2:
                    continue
                arr = np.asarray(fg2["intensity"][()], dtype=np.float32)
                # v1 xdart convention: shape (nq, nchi). v2 canonical: (nchi, nq).
                if arr.ndim == 2:
                    arr = arr.T
                i2.append(arr)
                if radial_2d is None and "radial" in fg2:
                    radial_2d = np.asarray(fg2["radial"][()], dtype=np.float32)
                if azim_2d is None and "azimuthal" in fg2:
                    azim_2d = np.asarray(fg2["azimuthal"][()], dtype=np.float32)
                    u = fg2["azimuthal"].attrs.get("units", b"")
                    azim_unit = _v2_decode_str(u) if u else "deg"
            if i2:
                data_vars["intensity_2d"] = (
                    ("frame", "chi", "q"), np.stack(i2, axis=0)
                )
                if "q" not in coords and radial_2d is not None:
                    coords["q"] = radial_2d
                if azim_2d is not None:
                    coords["chi"] = azim_2d
                    attrs_per_coord["chi"] = {"units": azim_unit or "deg"}

        # ── motor positioners from scan_data/ ─────────────────────
        scan_data = _read_v1_scan_data(e)
        N = len(frame_indices)
        reserved = {"q", "chi", "frame"}
        for k, arr in scan_data.items():
            if arr.ndim != 1 or arr.shape[0] != N:
                # Skip wrong-shape columns (e.g. scalar metadata snuck in)
                continue
            var_name = k if k not in reserved else f"sample_{k}"
            if var_name not in data_vars:
                data_vars[var_name] = (("frame",), arr)

        # ── thumbnails (optional, v1 stores them as NNNN_thumb datasets) ──
        if include_thumbnails:
            thumbs: list[np.ndarray] = []
            for idx in frame_indices:
                tkey = f"{idx:04d}_thumb"
                if tkey in frames_grp:
                    thumbs.append(np.asarray(frames_grp[tkey][()]))
            if thumbs and all(t.shape == thumbs[0].shape for t in thumbs):
                data_vars["thumbnail"] = (
                    ("frame", "thumb_y", "thumb_x"),
                    np.stack(thumbs, axis=0),
                )

        # ── frame coordinate ─────────────────────────────────────
        coords["frame"] = np.asarray(frame_indices, dtype=np.int32)

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    for var, attrs in attrs_per_coord.items():
        if var in ds.coords:
            ds[var].attrs.update(attrs)
    ds.attrs["reduction"] = {}    # v1 files have no NXprocess block
    ds.attrs["schema_version"] = "v1"
    return ds


def _read_sphere_v2(path: Path, entry: str, groups: tuple[str, ...],
                   include_thumbnails: bool):
    """v2-schema reader. See public ``read_sphere`` for the contract."""
    import xarray as xr

    from ssrl_xrd_tools.core.provenance import read_provenance

    data_vars: dict[str, tuple] = {}
    coords: dict[str, np.ndarray] = {}
    attrs_per_coord: dict[str, dict] = {}

    with h5py.File(path, "r") as f:
        if entry not in f:
            raise KeyError(f"No {entry!r} group in {path}")
        e = f[entry]

        if "1d" in groups and "integrated_1d" in e:
            g1 = e["integrated_1d"]
            data_vars["intensity_1d"] = (
                ("frame", "q"),
                np.asarray(g1["intensity"][()]),
            )
            coords["q"] = np.asarray(g1["q"][()])
            u = g1["q"].attrs.get("units", None)
            if u is not None:
                attrs_per_coord["q"] = {"units": _v2_decode_str(u)}
            if "sigma" in g1:
                data_vars["sigma_1d"] = (
                    ("frame", "q"),
                    np.asarray(g1["sigma"][()]),
                )
            if "frame_index" in g1:
                coords["frame"] = np.asarray(g1["frame_index"][()])

        if "2d" in groups and "integrated_2d" in e:
            g2 = e["integrated_2d"]
            data_vars["intensity_2d"] = (
                ("frame", "chi", "q"),
                np.asarray(g2["intensity"][()]),
            )
            coords.setdefault("q", np.asarray(g2["q"][()]))
            coords["chi"] = np.asarray(g2["chi"][()])
            u = g2["chi"].attrs.get("units", None)
            if u is not None:
                attrs_per_coord["chi"] = {"units": _v2_decode_str(u)}
            if "frame_index" in g2 and "frame" not in coords:
                coords["frame"] = np.asarray(g2["frame_index"][()])

        if "per_frame_geometry" in e:
            gg = e["per_frame_geometry"]
            for key in ("rot1", "rot2", "rot3", "incident_angle"):
                if key in gg:
                    data_vars[key] = (("frame",), np.asarray(gg[key][()]))
            if "frame_index" in gg and "frame" not in coords:
                coords["frame"] = np.asarray(gg["frame_index"][()])

        for category, path_in in [
            ("sample", "sample/positioners"),
            ("detector", "instrument/detector/positioners"),
        ]:
            if path_in in e:
                pos = _read_positioners(e[path_in])
                for k, arr in pos.items():
                    reserved = {"q", "chi", "frame"}
                    if k in reserved or k in data_vars:
                        var_name = f"{category}_{k}"
                    else:
                        var_name = k
                    data_vars[var_name] = (("frame",), arr)

        if include_thumbnails and "frames" in e:
            thumbs: list[np.ndarray] = []
            for name in sorted(e["frames"].keys()):
                fg = e[f"frames/{name}"]
                if "thumbnail" in fg:
                    thumbs.append(np.asarray(fg["thumbnail"][()]))
            if thumbs:
                data_vars["thumbnail"] = (
                    ("frame", "thumb_y", "thumb_x"),
                    np.stack(thumbs, axis=0),
                )

        if "frame" not in coords:
            N = None
            for _, (_, arr) in data_vars.items():
                if isinstance(arr, np.ndarray) and arr.ndim >= 1:
                    N = arr.shape[0]
                    break
            if N is not None:
                coords["frame"] = np.arange(N, dtype=np.int32)

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    for var, attrs in attrs_per_coord.items():
        if var in ds.coords:
            ds[var].attrs.update(attrs)

    try:
        ds.attrs["reduction"] = read_provenance(str(path), entry=entry)
    except Exception:
        ds.attrs["reduction"] = {}
    ds.attrs["schema_version"] = "v2"
    return ds


def read_sphere(
    path: Path | str,
    *,
    entry: str = "entry",
    groups: tuple[str, ...] = ("1d", "2d"),
    include_thumbnails: bool = False,
    schema: str | None = None,
):
    """Read an xdart NeXus file into an :class:`xarray.Dataset`.

    Auto-detects the schema version (v1 = xdart ≤ 0.36.x; v2 = xdart
    ≥ 0.37) and returns a canonical Dataset shape that downstream
    analysis code doesn't have to branch on.

    Parameters
    ----------
    path
        Path to the ``.nxs`` file.
    entry
        NXentry group name (default ``"entry"``).
    groups
        Which integrated stacks to load: subset of ``("1d", "2d")``.
    include_thumbnails
        If ``True``, load per-frame thumbnails as a
        ``thumbnail`` data variable.
    schema
        Force a specific schema version (``"v1"`` or ``"v2"``).  Default
        ``None`` auto-detects.

    Returns
    -------
    xarray.Dataset
        See module-level docstring for the canonical shape.
        ``ds.attrs["schema_version"]`` is ``"v1"`` or ``"v2"``.
    """
    path = Path(path)
    if schema is None:
        with h5py.File(path, "r") as f:
            if entry not in f:
                raise KeyError(f"No {entry!r} group in {path}")
            schema = _read_v1_or_v2(f[entry])
    if schema == "v1":
        return _read_sphere_v1(path, entry, groups, include_thumbnails)
    if schema == "v2":
        return _read_sphere_v2(path, entry, groups, include_thumbnails)
    raise ValueError(f"Unknown schema {schema!r}; expected 'v1' or 'v2'")


def read_stitched(
    path: Path | str,
    *,
    entry: str = "entry",
):
    """Read ``stitched_1d`` / ``stitched_2d`` if present (v2 only).

    v1 schema doesn't have stitched outputs — :class:`KeyError` is
    raised on v1 files regardless of which entry is requested.
    """
    import xarray as xr

    path = Path(path)
    data_vars: dict[str, tuple] = {}
    coords: dict[str, np.ndarray] = {}

    with h5py.File(path, "r") as f:
        if entry not in f:
            raise KeyError(f"No {entry!r} group in {path}")
        e = f[entry]
        has_1d = "stitched_1d" in e
        has_2d = "stitched_2d" in e
        if not (has_1d or has_2d):
            raise KeyError(f"No stitched_1d/2d in {path}:{entry}")
        if has_1d:
            g = e["stitched_1d"]
            coords["q"] = np.asarray(g["q"][()])
            data_vars["stitched_1d"] = (("q",), np.asarray(g["intensity"][()]))
            if "sigma" in g:
                data_vars["stitched_1d_sigma"] = (
                    ("q",), np.asarray(g["sigma"][()])
                )
        if has_2d:
            g = e["stitched_2d"]
            coords.setdefault("q", np.asarray(g["q"][()]))
            coords["chi"] = np.asarray(g["chi"][()])
            data_vars["stitched_2d"] = (
                ("q", "chi"), np.asarray(g["intensity"][()])
            )

    return xr.Dataset(data_vars=data_vars, coords=coords)


__all__ = [
    # v1 (legacy beamline files)
    "find_nexus_image_dataset",
    "list_entries",
    "read_nexus",
    "open_nexus_writer",
    "write_nexus",
    "write_nexus_frame",
    # v2 (xdart 0.37+)
    "read_sphere",
    "read_stitched",
]
