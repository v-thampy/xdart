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


def _find_eiger_external_link_paths(
    h5f: h5py.File, entry: str
) -> list[str]:
    """Return sorted ``/{entry}/data/data_NNNNNN`` external-link paths.

    Eiger master files store the full image stack as a sequence of
    sibling external links named ``data_000001``, ``data_000002``, ….
    Each link resolves to a 3D ``(n_in_file, H, W)`` dataset; the full
    scan is the concatenation along axis 0.

    Returns an empty list if there are no external links under
    ``/{entry}/data``.
    """
    if entry not in h5f:
        return []
    grp = h5f[entry]
    if "data" not in grp or not isinstance(grp["data"], h5py.Group):
        return []
    data_grp = grp["data"]
    ext_keys: list[str] = []
    for k in data_grp:
        try:
            link = data_grp.get(k, getlink=True)
        except KeyError:
            continue
        if link.__class__.__name__ == "ExternalLink":
            ext_keys.append(k)
    if not ext_keys:
        return []
    # Eiger names are zero-padded, so lexical sort = numeric order.
    ext_keys.sort()
    return [f"/{entry}/data/{k}" for k in ext_keys]


class NexusImageStack:
    """Read-only 3D image stack spanning one or more h5py datasets.

    Wraps either:

    * a single 3D dataset (e.g. ``/entry/instrument/detector/data``),
      in which case this proxy is functionally equivalent to the
      dataset itself; or
    * a sorted sequence of Eiger external-link datasets
      ``/entry/data/data_000001``, ``data_000002``, …, concatenated
      logically along axis 0.

    Provides the subset of the h5py.Dataset interface the wranglers
    actually use: :attr:`shape`, :attr:`dtype`, :attr:`ndim`,
    :func:`len`, ``__iter__``, and ``__getitem__`` with int or slice
    indexing along axis 0.

    Slicing is segment-aware: ``stack[a:b]`` reads each underlying
    dataset only over its intersection with ``[a, b)``, so a chunked
    bulk read crossing a file boundary still costs one HDF5 read per
    touched file rather than per frame.

    Must be used as a context manager so the owned ``h5py.File`` is
    closed::

        with open_nexus_image_stack(path) as stack:
            nframes = stack.shape[0]
            block = np.asarray(stack[0:16], dtype=np.float32)
    """

    __slots__ = ("_h5", "_paths", "_dsets", "_offsets", "shape",
                 "dtype", "ndim")

    def __init__(self, h5f: h5py.File, paths: list[str]):
        if not paths:
            raise ValueError("NexusImageStack requires at least one path")
        dsets = []
        for p in paths:
            obj = h5f[p]
            if not isinstance(obj, h5py.Dataset):
                raise TypeError(f"{p} is not a Dataset (got {type(obj).__name__})")
            if obj.ndim != 3:
                raise ValueError(
                    f"{p} is {obj.ndim}-D; NexusImageStack expects 3-D"
                )
            dsets.append(obj)

        per_frame_shapes = {d.shape[1:] for d in dsets}
        if len(per_frame_shapes) > 1:
            raise ValueError(
                f"Inconsistent per-frame shapes across segments: "
                f"{per_frame_shapes}"
            )
        # Offsets[i] = global index where segment i starts.
        # Offsets[-1] = total number of frames.
        lengths = [int(d.shape[0]) for d in dsets]
        offsets = [0]
        for n in lengths:
            offsets.append(offsets[-1] + n)

        self._h5 = h5f
        self._paths = list(paths)
        self._dsets = dsets
        self._offsets = offsets
        self.shape = (offsets[-1],) + dsets[0].shape[1:]
        self.dtype = dsets[0].dtype
        self.ndim = 3

    # ── Public introspection ────────────────────────────────────────────
    @property
    def n_segments(self) -> int:
        """Number of underlying datasets concatenated by this stack."""
        return len(self._dsets)

    @property
    def paths(self) -> tuple[str, ...]:
        """Internal HDF5 paths of the underlying datasets, in stack order."""
        return tuple(self._paths)

    # ── Container protocol ──────────────────────────────────────────────
    def __len__(self) -> int:
        return self.shape[0]

    def __iter__(self):
        for i in range(self.shape[0]):
            yield self[i]

    def __getitem__(self, key):
        n = self.shape[0]
        if isinstance(key, (int, np.integer)):
            i = int(key)
            if i < 0:
                i += n
            if not 0 <= i < n:
                raise IndexError(f"frame index {key} out of range [0, {n})")
            seg, local = self._locate(i)
            return self._dsets[seg][local]
        if isinstance(key, slice):
            start, stop, step = key.indices(n)
            if step == 1:
                return self._read_range(start, stop)
            # Rare path: non-unit step; fall back to per-frame reads
            # but stay vectorised at the end.
            if start >= stop:
                return np.empty((0,) + self.shape[1:], dtype=self.dtype)
            return np.stack(
                [self[i] for i in range(start, stop, step)],
                axis=0,
            )
        raise TypeError(
            f"NexusImageStack supports int and slice indexing along axis 0, "
            f"got {type(key).__name__}"
        )

    # ── Lifecycle / context manager ─────────────────────────────────────
    def close(self) -> None:
        """Close the underlying ``h5py.File`` if still open."""
        h5 = getattr(self, "_h5", None)
        if h5 is not None:
            try:
                h5.close()
            except Exception:
                pass
            self._h5 = None

    def __enter__(self) -> "NexusImageStack":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ── Internals ───────────────────────────────────────────────────────
    def _locate(self, i: int) -> tuple[int, int]:
        """Return ``(segment_index, local_index)`` for global frame ``i``."""
        # Linear scan: there are typically <20 Eiger files in a scan,
        # so a Python-level loop is faster than bisect's overhead.
        offs = self._offsets
        for seg in range(len(self._dsets)):
            if i < offs[seg + 1]:
                return seg, i - offs[seg]
        # Unreachable if bounds-checked at the caller.
        raise IndexError(i)

    def _read_range(self, start: int, stop: int) -> np.ndarray:
        if start >= stop:
            return np.empty((0,) + self.shape[1:], dtype=self.dtype)
        offs = self._offsets
        # Locate first and last segments touched by [start, stop).
        first_seg, _ = self._locate(start)
        last_seg, _ = self._locate(stop - 1)

        if first_seg == last_seg:
            base = offs[first_seg]
            return np.asarray(
                self._dsets[first_seg][start - base:stop - base]
            )
        chunks = []
        for seg in range(first_seg, last_seg + 1):
            seg_start = max(start, offs[seg]) - offs[seg]
            seg_stop = min(stop, offs[seg + 1]) - offs[seg]
            chunks.append(np.asarray(self._dsets[seg][seg_start:seg_stop]))
        return np.concatenate(chunks, axis=0)


def open_nexus_image_stack(
    path: Path | str,
    entry: str = "entry",
) -> NexusImageStack:
    """Open a NeXus file and return a :class:`NexusImageStack` proxy.

    Resolves the image dataset(s) under ``/{entry}/`` and wraps them
    in a single logical 3D stack.  The proxy owns the underlying
    ``h5py.File``; use it as a context manager.

    Resolution order:

    1. **External-link Eiger pattern**: if ``/{entry}/data/`` contains
       one or more external-link siblings (``data_000001``, …), they
       are concatenated along axis 0 in lexical (= numeric) order.
       Preferred over the single-dataset paths because real Eiger
       master files always carry external links here.
    2. **Single dataset**: the path returned by
       :func:`find_nexus_image_dataset`.

    Parameters
    ----------
    path : Path or str
        Path to the NeXus HDF5 file.
    entry : str, optional
        Name of the NXentry group (default ``"entry"``).

    Returns
    -------
    NexusImageStack
        Proxy over the resolved dataset(s).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If no 3D image dataset is found.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"NeXus file not found: {p}")

    h5f = h5py.File(p, "r")
    try:
        ext_paths = _find_eiger_external_link_paths(h5f, entry)
        if ext_paths:
            return NexusImageStack(h5f, ext_paths)

        single = find_nexus_image_dataset_in_open_file(h5f, entry)
        if single is None:
            raise KeyError(
                f"No 3-D image dataset found in {p}:{entry}"
            )
        return NexusImageStack(h5f, [single])
    except Exception:
        h5f.close()
        raise


def find_nexus_image_dataset_in_open_file(
    h5f: h5py.File, entry: str = "entry"
) -> str | None:
    """Variant of :func:`find_nexus_image_dataset` that reuses an open file.

    Used internally by :func:`open_nexus_image_stack` to avoid opening
    the file twice.  Same search order, minus the Eiger external-link
    branch (which the caller handles explicitly so it can grab *all*
    sibling links, not just the first).
    """
    if entry not in h5f:
        return None
    grp = h5f[entry]

    candidate = f"{entry}/instrument/detector/data"
    if candidate in h5f and isinstance(h5f[candidate], h5py.Dataset) and h5f[candidate].ndim == 3:
        return f"/{candidate}"

    candidate = f"{entry}/data/data"
    if candidate in h5f and isinstance(h5f[candidate], h5py.Dataset) and h5f[candidate].ndim == 3:
        return f"/{candidate}"

    if "instrument" in grp:
        instr = grp["instrument"]
        for subname in instr:
            sub = instr[subname]
            if not isinstance(sub, h5py.Group):
                continue
            inner = f"{entry}/instrument/{subname}/data"
            if inner in h5f and isinstance(h5f[inner], h5py.Dataset) and h5f[inner].ndim == 3:
                return f"/{inner}"

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
# v2 schema reader (xdart 0.37+)
# ---------------------------------------------------------------------------
# See xdart/docs/nexus_stitch_refactor_plan.md §2 for the layout.
# v1 (xdart ≤ 0.36.x) is intentionally not supported — re-reduce old
# data with current xdart if you need to view it.
#
# Dataset shape produced by ``read_scan``:
#   dims:   frame, q, q_2d, chi
#   vars:   intensity_1d   (frame, q)
#           sigma_1d       (frame, q)         optional
#           intensity_2d   (frame, chi, q_2d)
#           rot1, rot2, rot3, incident_angle  (frame,)
#           <each scan motor>                 (frame,)
#   coords: frame, q, q_2d, chi
#   attrs:  reduction (provenance dict)
# ===========================================================================

def _v2_decode_str(v):
    return v.decode("utf-8") if isinstance(v, bytes) else v


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


def _read_scan_v2(path: Path, entry: str, groups: tuple[str, ...],
                  include_thumbnails: bool):
    """v2-schema reader.  Body of public ``read_scan``."""
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
            # 1D and 2D radial axes are the same physical quantity (q
            # magnitude) but sampled at independent resolutions
            # (npt=2000 for 1D, npt_rad=500 for 2D is typical).  xarray
            # requires distinct dim names for differently-sized axes,
            # so 2D's radial axis lives under ``q_2d``.
            data_vars["intensity_2d"] = (
                ("frame", "chi", "q_2d"),
                np.asarray(g2["intensity"][()]),
            )
            coords["q_2d"] = np.asarray(g2["q"][()])
            u_q2 = g2["q"].attrs.get("units", None)
            if u_q2 is not None:
                attrs_per_coord["q_2d"] = {"units": _v2_decode_str(u_q2)}
            coords["chi"] = np.asarray(g2["chi"][()])
            u = g2["chi"].attrs.get("units", None)
            if u is not None:
                attrs_per_coord["chi"] = {"units": _v2_decode_str(u)}
            if "frame_index" in g2 and "frame" not in coords:
                coords["frame"] = np.asarray(g2["frame_index"][()])

        # Reject per-frame columns whose length disagrees with the frame
        # coord (malformed/partial file) so one bad column doesn't make the
        # whole xr.Dataset construction raise and the viewer come up empty.
        n_frames = len(coords["frame"]) if "frame" in coords else None

        def _add_frame_var(name, arr):
            arr = np.asarray(arr)
            if n_frames is not None and arr.ndim >= 1 and arr.shape[0] != n_frames:
                logger.warning(
                    "Skipping per-frame column %r in %s: length %d != %d frames",
                    name, path, arr.shape[0], n_frames,
                )
                return
            data_vars[name] = (("frame",), arr)

        if "per_frame_geometry" in e:
            gg = e["per_frame_geometry"]
            for key in ("rot1", "rot2", "rot3", "incident_angle"):
                if key in gg:
                    _add_frame_var(key, gg[key][()])
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
                    _add_frame_var(var_name, arr)

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
    return ds


def read_scan_metadata(
    path: Path | str,
    *,
    entry: str = "entry",
):
    """Read everything *except* the heavy integrated stacks.

    Returns an :class:`xarray.Dataset` shaped like :func:`read_scan`
    but with ``intensity_1d``, ``intensity_2d``, ``sigma_1d``, and
    thumbnails omitted.  Still includes:

    * ``frame`` coord (from ``integrated_1d/frame_index`` or
      ``integrated_2d/frame_index``, falling back to
      ``per_frame_geometry/frame_index``)
    * ``q`` and ``chi`` coords (cheap, ~few KB)
    * Per-frame motor positioners and derived geometry
    * ``ds.attrs["reduction"]`` (provenance, bai_*_args, geometry config)

    Intended for the GUI open path where a viewer needs to know what
    frames exist + what axes they're sampled on, but doesn't yet need
    the full (frame, chi, q) intensity tensor.  ``ArchSeries`` lazy-
    loads each frame's slices on demand.

    Reading just frame_index + coords + positioners is O(few KB) vs
    O(N * nchi * nq * 4 B) for the full ``intensity_2d`` materialisation
    — opening a 10k-frame Eiger scan goes from ~seconds to ~tens of ms.
    """
    import xarray as xr
    from ssrl_xrd_tools.core.provenance import read_provenance

    path = Path(path)
    data_vars: dict[str, tuple] = {}
    coords: dict[str, np.ndarray] = {}
    attrs_per_coord: dict[str, dict] = {}

    with h5py.File(path, "r") as f:
        if entry not in f:
            raise KeyError(f"No {entry!r} group in {path}")
        e = f[entry]

        # frame_index — try integrated_1d first, then integrated_2d,
        # then per_frame_geometry.  Cheap; no intensity is loaded.
        for grp_name in ("integrated_1d", "integrated_2d",
                         "per_frame_geometry"):
            if grp_name in e and "frame_index" in e[grp_name]:
                coords["frame"] = np.asarray(
                    e[grp_name]["frame_index"][()]
                )
                break

        # q / chi axes (small).
        if "integrated_1d" in e and "q" in e["integrated_1d"]:
            coords["q"] = np.asarray(e["integrated_1d/q"][()])
            u = e["integrated_1d/q"].attrs.get("units", None)
            if u is not None:
                attrs_per_coord["q"] = {"units": _v2_decode_str(u)}
        if "integrated_2d" in e:
            g2 = e["integrated_2d"]
            if "q" in g2:
                coords["q_2d"] = np.asarray(g2["q"][()])
                u_q2 = g2["q"].attrs.get("units", None)
                if u_q2 is not None:
                    attrs_per_coord["q_2d"] = {"units": _v2_decode_str(u_q2)}
            if "chi" in g2:
                coords["chi"] = np.asarray(g2["chi"][()])
                u = g2["chi"].attrs.get("units", None)
                if u is not None:
                    attrs_per_coord["chi"] = {"units": _v2_decode_str(u)}

        # Number of frames implied by the frame coord, used to reject
        # per-frame columns whose length disagrees (a malformed/partial
        # file — e.g. 16 integrated frames but only 8 ``th`` positions).
        # Without this guard a single mismatched column makes the whole
        # xr.Dataset construction raise and the viewer comes up empty.
        n_frames = len(coords["frame"]) if "frame" in coords else None

        def _add_frame_var(name, arr):
            arr = np.asarray(arr)
            if n_frames is not None and arr.ndim >= 1 and arr.shape[0] != n_frames:
                logger.warning(
                    "Skipping per-frame column %r in %s: length %d != %d frames",
                    name, path, arr.shape[0], n_frames,
                )
                return
            data_vars[name] = (("frame",), arr)

        # Derived per-frame geometry (rot1/2/3, incident_angle).
        if "per_frame_geometry" in e:
            gg = e["per_frame_geometry"]
            for key in ("rot1", "rot2", "rot3", "incident_angle"):
                if key in gg:
                    _add_frame_var(key, gg[key][()])

        # Positioners.
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
                    _add_frame_var(var_name, arr)

        # Fallback frame coord if neither integrated_* nor
        # per_frame_geometry carried frame_index but positioners
        # did contribute (frame,) arrays.
        if "frame" not in coords:
            for _, (_, arr) in data_vars.items():
                if isinstance(arr, np.ndarray) and arr.ndim >= 1:
                    coords["frame"] = np.arange(arr.shape[0], dtype=np.int32)
                    break

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    for var, attrs in attrs_per_coord.items():
        if var in ds.coords:
            ds[var].attrs.update(attrs)
    try:
        ds.attrs["reduction"] = read_provenance(str(path), entry=entry)
    except Exception:
        ds.attrs["reduction"] = {}
    return ds


def read_scan(
    path: Path | str,
    *,
    entry: str = "entry",
    groups: tuple[str, ...] = ("1d", "2d"),
    include_thumbnails: bool = False,
):
    """Read an xdart v2 NeXus scan file into an :class:`xarray.Dataset`.

    v1 (xdart ≤ 0.36.x) is intentionally not supported.  Re-reduce
    older data with current xdart if you need to open it.

    Parameters
    ----------
    path
        Path to the ``.nxs`` file.
    entry
        NXentry group name (default ``"entry"``).
    groups
        Which integrated stacks to load: subset of ``("1d", "2d")``.
    include_thumbnails
        If ``True``, load per-frame thumbnails as a ``thumbnail`` data
        variable.

    Returns
    -------
    xarray.Dataset
        See module-level docstring for the canonical shape.
    """
    return _read_scan_v2(Path(path), entry, groups, include_thumbnails)


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
    "NexusImageStack",
    "open_nexus_image_stack",
    "list_entries",
    "read_nexus",
    "open_nexus_writer",
    "write_nexus",
    "write_nexus_frame",
    # v2 (xdart 0.37+)
    "read_scan",
    "read_scan_metadata",
    "read_stitched",
]
