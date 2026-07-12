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
- gzip+shuffle compression by default: portable (in every HDF5 build, stock-h5py
  readable with no plugin, no ARM64-macOS native bus error). ``lzf`` is accepted
  only as a backward-compat alias and normalized to gzip on EVERY platform; it is
  never emitted.
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

# Best-effort: register hdf5plugin's HDF5 dynamic filters (lz4 etc.) for both
# writing AND reading.  gzip (the portable default) needs no plugin; lz4 needs it
# on the reader too, so importing here lets a headless reader round-trip an
# lz4-compressed stack when hdf5plugin is installed.  Absent -> lz4 unavailable
# (resolve_stack_compression / _comp_kwargs fall back to gzip).
try:
    import hdf5plugin  # noqa: F401  (registers HDF5 dynamic filters)
    _HAS_HDF5PLUGIN = True
except Exception:
    _HAS_HDF5PLUGIN = False

from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from xrd_tools.core.frame_view import two_d_kind_from_units
from xrd_tools.core.metadata import ScanMetadata
# Layout facts (stamp attrs, version, capability attrs) are declared once in
# xrd_tools.io.schema; this module is a consumer.  PROCESSED_SCHEMA_NAME /
# _VERSION stay importable from here (public API since the C1 pull-ins).
from xrd_tools.io.schema import (
    DEFAULT_MODE_KEY,
    DTYPE_ATTR,
    MONOTONIC_ATTR,
    MULTI_RESULT_MODES_ATTR,
    PRIMARY_MODE_ATTR,
    PROCESSED_SCHEMA_NAME,
    PROCESSED_SCHEMA_VERSION,
    SCHEMA,
    SCHEMA_NAME_ATTR,
    SCHEMA_VERSION_ATTR,
    mode_subgroup_name,
)
from xrd_tools.transforms import energy_to_wavelength

logger = logging.getLogger(__name__)

_UTF8_DTYPE = h5py.string_dtype(encoding="utf-8")


def warn_if_newer_schema(entry_grp, path="") -> None:
    """C1: warn when a file's ``ssrl_schema_version`` is NEWER than this
    library supports.

    The writer stamps the version on every file; until now no reader ever
    looked at it, so a v3 file hit today's readers with opaque downstream
    KeyErrors or silently missing features.  Absent/old stamps pass silently
    (back-compat); only a newer stamp warns."""
    try:
        ver = int(entry_grp.attrs.get(
            SCHEMA_VERSION_ATTR, PROCESSED_SCHEMA_VERSION))
    except (TypeError, ValueError):
        return
    if ver > PROCESSED_SCHEMA_VERSION:
        warnings.warn(
            f"{path or 'file'} has ssrl_schema_version={ver}, newer than the "
            f"supported {PROCESSED_SCHEMA_VERSION} — upgrade xrd_tools; "
            f"some datasets/features may be missing or misread.",
            RuntimeWarning, stacklevel=3,
        )

# Datasets that are counters/scalers rather than motor angles.
_DEFAULT_COUNTER_NAMES: frozenset[str] = frozenset(
    {"i0", "i1", "i2", "monitor", "mon", "det", "diode", "seconds", "epoch", "time"}
)

# Per-point columns that are neither motors nor counters — bookkeeping indices
# that must not pollute the motor list.  xdart's own writer emits ``frame_index``
# into ``/entry/scan_data`` (and under sample/positioners), so a re-ingested file
# would otherwise classify it as a motor angle.
_NON_MOTOR_COLUMNS: frozenset[str] = frozenset({"frame_index"})


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
        Motor names to extract.  If *None*, all 1D float datasets whose names
        are **not** in ``counter_names`` are treated as motors.  Sources scanned
        (see :func:`_read_data_group`): ``/{entry}/data/`` + ``/{entry}/scan_data/``
        (per-point motor/counter tables) and ``/{entry}/sample/positioners/*``
        (NXpositioner ``value`` arrays — the scanned motors).
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

    Search order (the Eiger external-link arm first, then ONE shared
    implementation with :func:`find_nexus_image_dataset_in_open_file` — the two
    finders drifted once (F6/wf_3614041c: only the open-file variant learned the
    single-2-D-frame acceptance, so the live watch silently dropped one-exposure
    ``.nxs`` files the headless seam opened fine); delegating kills that class):

    1. Eiger external-link pattern ``/{entry}/data/data_NNNNNN`` (the first
       link path is returned so the caller can enumerate siblings — the same
       precedence :func:`open_nexus_image_stack` applies)
    2. ``/{entry}/instrument/detector/data`` (3-D)
    3. ``/{entry}/data/data`` (3-D)
    4. ``/{entry}/instrument/*/data``  (any detector sub-group, 3-D)
    5. A dataset flagged ``@signal_type='detector'`` (Bluesky/NXWriter)
    6. Largest 3-D dataset anywhere under ``/{entry}/``
    7. Last resort: a single 2-D detector frame at a canonical location
       (a one-exposure acquisition; normalized to ``(1, H, W)`` downstream)

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

        # Eiger external-link pattern: /entry/data/data_NNNNNN.  Return the
        # *first* link path so the caller can enumerate siblings to build the
        # full stack.  (External target files may be missing mid-transfer, so
        # the resolution is guarded.)
        ext_paths = _find_eiger_external_link_paths(f, entry)
        if ext_paths:
            try:
                ds = f[ext_paths[0]]
                if isinstance(ds, h5py.Dataset) and ds.ndim == 3:
                    logger.debug("Found Eiger external-link dataset: %s",
                                 ext_paths[0])
                    return ext_paths[0]
            except Exception:
                pass

        return find_nexus_image_dataset_in_open_file(f, entry)


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
      dataset itself;
    * a single 2-D detector dataset (a one-frame acquisition, e.g. a
      Bluesky/NXWriter count with one exposure), exposed as a logical
      ``(1, H, W)`` stack — the same normalization the display readers
      (``classify_image_source`` / ``read_image``) already apply; or
    * a sorted sequence of Eiger external-link datasets
      ``/entry/data/data_000001``, ``data_000002``, …, concatenated
      logically along axis 0 (always strict-3D).

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
                 "dtype", "ndim", "_squeeze2d")

    def __init__(self, h5f: h5py.File, paths: list[str]):
        if not paths:
            raise ValueError("NexusImageStack requires at least one path")
        # F6: a SINGLE 2-D detector dataset is a one-frame stack (the finder /
        # classifier already accept ndim >= 2); normalize it to (1, H, W) here
        # instead of raising.  Multi-segment (Eiger links) stays strict-3D.
        squeeze2d = False
        dsets = []
        for p in paths:
            obj = h5f[p]
            if not isinstance(obj, h5py.Dataset):
                raise TypeError(f"{p} is not a Dataset (got {type(obj).__name__})")
            if obj.ndim == 2 and len(paths) == 1:
                squeeze2d = True
            elif obj.ndim != 3:
                raise ValueError(
                    f"{p} is {obj.ndim}-D; NexusImageStack expects 3-D "
                    f"(or a single 2-D detector frame)"
                )
            dsets.append(obj)

        per_frame_shapes = ({dsets[0].shape} if squeeze2d
                            else {d.shape[1:] for d in dsets})
        if len(per_frame_shapes) > 1:
            raise ValueError(
                f"Inconsistent per-frame shapes across segments: "
                f"{per_frame_shapes}"
            )
        # Offsets[i] = global index where segment i starts.
        # Offsets[-1] = total number of frames.
        lengths = [1] if squeeze2d else [int(d.shape[0]) for d in dsets]
        offsets = [0]
        for n in lengths:
            offsets.append(offsets[-1] + n)

        self._h5 = h5f
        self._paths = list(paths)
        self._dsets = dsets
        self._offsets = offsets
        self._squeeze2d = squeeze2d
        self.shape = (offsets[-1],) + per_frame_shapes.pop()
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
            if self._squeeze2d:
                # The lone 2-D dataset IS frame 0.
                return self._dsets[0][()]
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
        if self._squeeze2d:
            # Single logical frame: any non-empty range is exactly [0, 1).
            return np.asarray(self._dsets[0])[None]
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
    in a single logical 3D stack (a lone 2-D detector frame becomes a
    ``(1, H, W)`` stack).  The proxy owns the underlying
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
        If no image dataset is found.
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
                f"No image dataset found in {p}:{entry}"
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

    # 3-D resolution keeps its FULL pre-F6 precedence: every arm below runs
    # before any 2-D acceptance, so a lone 2-D non-image dataset (an MCA/spectra
    # table at entry/data/data, a small per-frame array under instrument/*) can
    # never shadow a real 3-D stack that used to resolve (review-caught
    # regression, wf_3614041c).  A single 2-D detector FRAME (F6: one-exposure
    # acquisition) is accepted by the canonical-location LAST-RESORT pass at the
    # bottom, only when nothing else resolves.
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

    # Bluesky / apstools NXWriter: the NXdata ``@signal`` points at a scalar
    # counter, so the detector pixels are flagged ``@signal_type='detector'``
    # instead (on ``entry/data/eiger_image`` etc.).  Prefer that explicit marker
    # over the weak largest-3D fallback below — the same convention the viewer
    # resolver in ``image.py`` uses.
    from xrd_tools.io.bluesky_nexus import find_detector_signal_dataset
    det = find_detector_signal_dataset(grp)
    if det is not None:
        return det.name

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
    if best_path is not None:
        return best_path

    # F6 LAST RESORT: a single 2-D detector frame at a canonical location (a
    # one-exposure acquisition; classify_image_source / read_image accept it,
    # and NexusImageStack normalizes it to (1, H, W)).  Runs only when NOTHING
    # 3-D or detector-flagged resolved above, so a 2-D table can never shadow a
    # real stack.  The largest-dataset fallback stays strict-3D — a lone 2-D
    # array of unknown provenance is more likely a table than a frame.
    for cand in (f"{entry}/instrument/detector/data", f"{entry}/data/data"):
        if cand in h5f and isinstance(h5f[cand], h5py.Dataset) and h5f[cand].ndim == 2:
            return f"/{cand}"
    if "instrument" in grp:
        instr = grp["instrument"]
        for subname in instr:
            sub = instr[subname]
            if not isinstance(sub, h5py.Group):
                continue
            inner = f"{entry}/instrument/{subname}/data"
            if inner in h5f and isinstance(h5f[inner], h5py.Dataset) and h5f[inner].ndim == 2:
                return f"/{inner}"
    return None


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
    compression: str | None = "gzip",
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
        HDF5 compression filter. ``"gzip"`` (default) is the portable policy
        (gzip+shuffle, ``compression_opts=1``): in every HDF5 build, stock-h5py
        readable, no ARM64-macOS bus error. ``"lzf"`` is accepted only as a
        backward-compat alias and normalized to gzip on every platform (never
        emitted). ``None`` disables compression.
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
    sorted_1d = _sorted_result_items(results_1d, "results_1d")
    sorted_2d = _sorted_result_items(results_2d, "results_2d")
    if sorted_1d:
        _require_uniform_axes_1d([result for _, result in sorted_1d])
    if sorted_2d:
        _require_uniform_axes_2d([result for _, result in sorted_2d])

    with h5py.File(p, mode) as f:
        grp = f.require_group(entry)
        grp.attrs["NX_class"] = "NXentry"
        _stamp_processed_schema(grp)

        if sorted_1d:
            validate_integrated_stack_write(
                grp,
                frame_indices=[frame for frame, _ in sorted_1d],
                results_1d=[result for _, result in sorted_1d],
            )
        if sorted_2d:
            validate_integrated_stack_write(
                grp,
                frame_indices=[frame for frame, _ in sorted_2d],
                results_2d=[result for _, result in sorted_2d],
            )

        if metadata is not None:
            _write_metadata(grp, metadata, comp_kwargs)

        proc = grp.require_group("reduction")
        proc.attrs["NX_class"] = "NXprocess"
        # Persisted-format literal: files have always stamped this program
        # name; keep it stable across the monorepo rename (readers and the
        # 6a byte-compat gate rely on unchanged output).
        proc.attrs.setdefault("program", "ssrl_xrd_tools")

        # Stacked v2 layout (read_scan-compatible).  Iterate in ascending
        # frame order so frame_index is monotonic on disk.
        if sorted_1d:
            write_integrated_stack(
                grp,
                frame_indices=[frame for frame, _ in sorted_1d],
                results_1d=[result for _, result in sorted_1d],
                compression=compression,
            )

        if sorted_2d:
            write_integrated_stack(
                grp,
                frame_indices=[frame for frame, _ in sorted_2d],
                results_2d=[result for _, result in sorted_2d],
                compression=compression,
            )

    logger.debug("Wrote NeXus file: %s", p)
    return p


def _sorted_result_items(results, name: str) -> list[tuple[int, Any]]:
    """Normalize result labels once and reject collisions before mutation."""
    if not results:
        return []
    items = [(int(frame), result) for frame, result in results.items()]
    labels = [frame for frame, _ in items]
    if len(labels) != len(set(labels)):
        raise ValueError(f"{name} contains duplicate normalized frame labels: {labels}")
    return sorted(items, key=lambda item: item[0])


# ---------------------------------------------------------------------------
# Write API — frame-by-frame: for live reduction hot loops
# ---------------------------------------------------------------------------

def open_nexus_writer(
    path: Path | str,
    metadata: ScanMetadata | None = None,
    entry: str = "entry",
    compression: str | None = "gzip",
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
        INTENTIONALLY UNAVAILABLE — raises :class:`NotImplementedError`.
        SWMR-write requires every dataset to exist before the mode is
        enabled, but this writer creates the integrated stacks on the
        first frame append (HDF5 forbids object creation in SWMR-write
        mode).  Concurrent readers are served instead by the retrying
        open helpers (``catch_h5py_file``-style); leave ``swmr=False``.
    overwrite : bool, optional
        If True, overwrite existing file.

    Returns
    -------
    h5py.File
        Open file handle. Caller is responsible for closing.

    Examples
    --------
    ::

        h5 = open_nexus_writer("scan_001.nxs", metadata=meta)
        try:
            for i, (r1d, r2d) in enumerate(process_frames(...)):
                write_nexus_frame(h5, i, result_1d=r1d, result_2d=r2d)
                h5.flush()
        finally:
            h5.close()
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # S6: SWMR-write is advertised but not functional — HDF5 forbids object
    # creation in SWMR-write mode, and this writer creates the integrated
    # stacks on the FIRST frame append, so enabling it here guaranteed a
    # failure on the first write.  Refuse loudly until the writer pre-creates
    # every dataset before flipping swmr_mode.
    if swmr:
        raise NotImplementedError(
            "open_nexus_writer(swmr=True) is not functional: the integrated "
            "stacks are created on the first frame append, which HDF5 forbids "
            "in SWMR-write mode.  Open without swmr (readers already tolerate "
            "concurrent reads via the retrying open helpers)."
        )

    file_kwargs: dict[str, Any] = {}

    mode = "w" if overwrite else "a"
    f = h5py.File(p, mode, **file_kwargs)
    try:
        return _open_nexus_writer_body(f, entry, metadata, compression)
    except BaseException:
        # Close-on-construction-failure (same guard as FrameViewReader /
        # open_nexus_image_stack): a header/metadata error otherwise orphans
        # the open handle -- which LOCKS the file on Windows.
        f.close()
        raise


def _open_nexus_writer_body(f, entry, metadata, compression):
    ck = _comp_kwargs(compression)

    grp = f.require_group(entry)
    grp.attrs["NX_class"] = "NXentry"
    _stamp_processed_schema(grp)

    if metadata is not None:
        _write_metadata(grp, metadata, ck)

    proc = grp.require_group("reduction")
    proc.attrs["NX_class"] = "NXprocess"
    # Persisted-format literal: files have always stamped this program
    # name; keep it stable across the monorepo rename (readers and the
    # 6a byte-compat gate rely on unchanged output).
    proc.attrs.setdefault("program", "ssrl_xrd_tools")

    return f


def write_nexus_frame(
    h5: h5py.File,
    frame: int | str,
    result_1d: IntegrationResult1D | None = None,
    result_2d: IntegrationResult2D | None = None,
    entry: str = "entry",
    compression: str | None = "gzip",
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
    e = h5[entry]

    if result_1d is not None:
        _append_stacked_1d(e, frame, result_1d, ck)

    if result_2d is not None:
        _append_stacked_2d(e, frame, result_2d, ck)


# ---------------------------------------------------------------------------
# Private helpers — compression
# ---------------------------------------------------------------------------

_GZIP_KWARGS = {"compression": "gzip", "compression_opts": 1, "shuffle": True}


def _native_filter_unsafe() -> bool:
    """True when the hdf5plugin LZ4 filter must NOT be used -- i.e. hdf5plugin is
    not importable (it is needed on BOTH the writer and the reader).

    There is NO platform guard: the former ARM64-macOS bus error was specific to
    h5py's BUNDLED LZF filter (a separate C implementation), NOT the hdf5plugin
    LZ4 filter.  A write+read spot-check on arm64-macOS (h5py 3.16 / hdf5 1.14)
    confirms LZ4 (filter 32004) round-trips cleanly there, and xdart already
    DECODES lz4 on arm64-macOS every time it opens a Dectris Eiger file -- so the
    only real requirement is plugin availability.  Callers fall back to
    gzip+shuffle when this is True."""
    return not _HAS_HDF5PLUGIN


_lz4_default_warned = False


def resolve_stack_compression(default: "str | None" = "lz4") -> "str | None":
    """Resolve the integrated-stack compression filter, overridable for BOTH the
    GUI and the headless reduction via the ``XDART_INTEGRATED_COMPRESSION`` env var
    (read per call -- set it in the shell before launching xdart, or before a
    headless run).  Case-insensitive values:

    * ``none`` / ``off`` / ``0`` / ``false`` / ``no`` -> ``None`` (uncompressed); when
      it comes from the env var this WARNS (opting into ~4x larger files).  An EMPTY
      value is treated as UNSET -> ``default`` (a stale empty export must not
      silently disable compression).
    * ``lz4`` -> fast hdf5plugin LZ4+shuffle (THE DEFAULT): fast writes, gzip-class
      ratio; the reader needs hdf5plugin (a base dep of xrd-tools).  Falls back to
      gzip only when hdf5plugin is not importable.
    * ``gzip`` (``lzf`` is an alias) -> portable gzip+shuffle (no hdf5plugin needed
      on the reader; the right choice for stock-h5py interoperability)
    * any UNRECOGNIZED value -> warn + fall back to gzip (a typo'd codec must not
      crash every integrated-stack write at ``create_dataset``)

    An unset env var uses ``default`` (lz4).  Headless callers that pass an
    explicit ``compression=`` to the sink/writer bypass this entirely.
    """
    raw = os.environ.get("XDART_INTEGRATED_COMPRESSION")
    # An empty value (e.g. a stale ``export XDART_INTEGRATED_COMPRESSION=``) means
    # UNSET -> default, NOT uncompressed: a leftover empty export must never
    # silently disable compression (~4x larger integrated .nxs -- observed live).
    if raw is not None and raw.strip() == "":
        raw = None
    # An EXPLICIT disable from the env var is LOUD (WARNING): the user is opting
    # into uncompressed output and should see it.  (A ``default=None`` from a
    # headless caller is intentional and stays silent below.)
    if raw is not None and raw.strip().lower() in (
            "none", "off", "0", "false", "no"):
        logger.warning(
            "integrated-stack compression DISABLED via XDART_INTEGRATED_COMPRESSION"
            "=%r -- integrated .nxs will be UNCOMPRESSED (~4x larger). Unset the "
            "variable to restore the lz4 default.", raw)
        return None
    val = raw if raw is not None else default
    if val is None:
        return None
    v = str(val).strip().lower()
    if v in ("none", "off", "0", "false", "no"):
        return None
    if v in ("gzip", "lzf"):
        return "gzip"
    if v == "lz4":
        if _native_filter_unsafe():
            logger.warning(
                "integrated-stack compression 'lz4' is unavailable here "
                "(hdf5plugin not importable); using gzip+shuffle instead")
            return "gzip"
        global _lz4_default_warned
        if not _lz4_default_warned:
            _lz4_default_warned = True
            logger.warning(
                "integrated stacks are lz4-compressed (fast; reader needs "
                "hdf5plugin, a base dep of xrd-tools).  Reading them OUTSIDE "
                "xrd-tools requires hdf5plugin; set XDART_INTEGRATED_COMPRESSION="
                "gzip for stock-h5py-portable files, or =none to disable.")
        return "lz4"
    # Unrecognized / typo'd value: degrade LOUDLY to the portable default rather
    # than honor it verbatim -- a verbatim unknown filter crashes every write at
    # create_dataset (ValueError: Compression filter "X" is unavailable), and the
    # env var is a documented A/B knob where a typo is plausible.
    logger.warning(
        "unrecognized XDART_INTEGRATED_COMPRESSION=%r; using gzip+shuffle "
        "(valid: none, lz4, gzip)", val)
    return "gzip"


def _comp_kwargs(compression: str | None) -> dict[str, Any]:
    """Build h5py dataset kwargs for a given compression filter.

    ``"lz4"`` is the project default: hdf5plugin's LZ4 (filter 32004) + shuffle --
    fast writes with a gzip-class ratio on the smooth integrated arrays.  The
    READER needs hdf5plugin (a base dep of xrd-tools); it round-trips on every
    platform including arm64-macOS (the old bus error was h5py's bundled LZF, a
    different filter).  ``"gzip"`` (DEFLATE) is the portable fallback: part of
    every HDF5 build, readable with a stock h5py (no hdf5plugin on the reader);
    ``"lzf"`` is a backward-compatible alias normalized to gzip+shuffle and is
    NEVER emitted (h5py-only filter, ARM64-macOS bus error).  All of these are
    lossless, so the byte-compat signature -- which digests DECOMPRESSED values --
    is invariant under the filter choice.  ``None`` stores uncompressed.

    Any value the resolver couldn't map (it degrades unknowns to gzip) is treated
    here as a last-resort verbatim filter so a deliberate caller-supplied codec
    still works; the resolver already shields the env path from typos.
    """
    if compression is None:
        return {}
    if compression in ("gzip", "lzf"):
        # level 1 = fastest gzip; shuffle markedly improves the ratio on the
        # smooth integrated-intensity arrays.  One policy on every platform.
        return dict(_GZIP_KWARGS)
    if compression == "lz4":
        # hdf5plugin LZ4 (filter 32004) + HDF5 shuffle: fast, gzip-class ratio.
        # The READER needs hdf5plugin.  Defensive fallback to gzip when the plugin
        # is missing so we never emit a filter that can't be read back here.
        if _native_filter_unsafe():
            logger.warning("lz4 compression unavailable here; using gzip+shuffle")
            return dict(_GZIP_KWARGS)
        return dict(shuffle=True, **hdf5plugin.LZ4())
    return {"compression": compression}


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

def _append_stacked_1d(
    entry_grp: h5py.Group,
    frame_idx: int | str,
    r: IntegrationResult1D,
    comp_kwargs: dict[str, Any],
    *,
    group_name: str = "integrated_1d",
) -> None:
    """Append one IntegrationResult1D as a row of the stacked
    ``/{entry}/integrated_1d`` NXdata group.

    This is the canonical v2 layout that :func:`read_scan` consumes (and
    that the xdart GUI writer produces): resizable ``intensity`` of shape
    ``(n_frames, n_q)``, a shared ``q`` axis, optional stacked ``sigma``,
    and a ``frame_index`` vector.

    Upsert semantics (matches the "append/update" contract): a frame whose
    label is already on disk replaces that row in place; a new label is
    appended.  This keeps reruns / partial reprocessing idempotent instead
    of producing duplicate ``frame_index`` entries.
    """
    intensity = np.asarray(r.intensity, dtype=np.float32)
    n_q = intensity.shape[0]
    idx = int(frame_idx)

    if group_name not in entry_grp:
        g = _create_group_from_schema(
            entry_grp, "integrated_1d", disk_name=group_name,
        )
        g.create_dataset("intensity", data=intensity[None, :],
                         maxshape=(None, n_q), chunks=(1, n_q), **comp_kwargs)
        qd = g.create_dataset("q", data=np.asarray(r.radial, dtype=np.float32))
        qd.attrs["units"] = r.unit
        g.create_dataset("frame_index", data=np.asarray([idx], dtype=np.int64),
                         maxshape=(None,), chunks=(64,))
        axis_kind = _axis_kind_1d(r.unit)
        if axis_kind != "radial":
            g.attrs["axis_kind"] = axis_kind
        g.attrs[MONOTONIC_ATTR] = True
        if r.sigma is not None:
            g.create_dataset("sigma", data=np.asarray(r.sigma, dtype=np.float32)[None, :],
                             maxshape=(None, n_q), chunks=(1, n_q), **comp_kwargs)
        return

    g = entry_grp[group_name]
    di = g["intensity"]
    if di.shape[1] != n_q:
        raise ValueError(
            f"{group_name} row size {n_q} != on-disk {di.shape[1]}; "
            "all frames in a scan must share the same npt."
        )
    # Per-frame appends can't refresh the shared q axis / units, so a frame
    # whose radial axis differs from what's on disk would be stored under a
    # stale axis (silent corruption).  The bulk ``write_integrated_stack``
    # path detects this earlier and rebuilds; the live / NexusSink path
    # reaches here directly, so reject it loudly.
    if not _axes_match_1d(g, r):
        raise ValueError(
            f"{group_name} radial axis/unit differs from what's on disk; "
            "a frame's q axis must match the rest of the scan (use "
            "write_integrated_stack with all frames to re-axis a reintegration)."
        )
    fi = g["frame_index"]
    n = di.shape[0]
    monotonic = bool(g.attrs.get(MONOTONIC_ATTR, False))
    last_idx = int(fi[n - 1]) if n else None
    match = (
        np.empty(0, dtype=int)
        if monotonic and (last_idx is None or idx > last_idx)
        else np.where(np.asarray(fi[()]) == idx)[0]
    )
    if match.size:
        pos = int(match[0])  # upsert: replace existing row for this label
        di[pos] = intensity
        _upsert_sigma_1d(g, pos, r, n_q, comp_kwargs)
        return
    di.resize(n + 1, axis=0)
    di[n] = intensity
    fi.resize(n + 1, axis=0)
    fi[n] = idx
    g.attrs[MONOTONIC_ATTR] = (
        monotonic and (last_idx is None or idx > last_idx)
    )
    _append_sigma_row(g, n, (None if r.sigma is None
                             else np.asarray(r.sigma, np.float32)),
                      (n_q,), comp_kwargs)


def _append_sigma_row(g, new_row_pos, sigma_row, row_shape, comp_kwargs):
    """Append a frame's sigma at position ``new_row_pos`` (= the index of the
    just-appended intensity row), keeping sigma row-aligned with intensity.

    Handles the three sigma cases the per-frame appenders must get right:

    * sigma exists, frame has sigma → extend + write the value;
    * sigma exists, frame has none → extend + NaN-pad (don't desync);
    * sigma does NOT exist yet but THIS frame brings sigma → create the
      dataset and NaN-backfill the earlier rows, then write this row (so a
      sigma introduced mid-scan isn't silently discarded).
    """
    if "sigma" in g:
        ds = g["sigma"]
        ds.resize(new_row_pos + 1, axis=0)
        ds[new_row_pos] = (sigma_row if sigma_row is not None
                           else np.full(row_shape, np.nan, np.float32))
    elif sigma_row is not None:
        # First sigma in the scan arrives after some sigma-less frames —
        # create the stack NaN-backfilled for the prior rows.
        n_rows = new_row_pos + 1
        chunks = (max(1, min(n_rows, 32)),) + tuple(row_shape)
        data = np.full((n_rows,) + tuple(row_shape), np.nan, np.float32)
        data[new_row_pos] = sigma_row
        g.create_dataset("sigma", data=data,
                         maxshape=(None,) + tuple(row_shape),
                         chunks=chunks, **comp_kwargs)


def _upsert_sigma_1d(g, pos, r, n_q, comp_kwargs=None):
    """Update sigma for an upserted (replaced) 1D row.

    * frame has sigma → write it (creating the dataset NaN-backfilled if it
      didn't exist yet — a sigma introduced on reintegration);
    * frame has NO sigma but a sigma dataset exists → write NaN, so a
      reintegration that dropped sigma doesn't leave a STALE uncertainty.
    """
    if r.sigma is not None:
        row = np.asarray(r.sigma, np.float32)
        if "sigma" in g:
            g["sigma"][pos] = row
        else:
            n_rows = g["intensity"].shape[0]
            data = np.full((n_rows, n_q), np.nan, np.float32)
            data[pos] = row
            g.create_dataset("sigma", data=data, maxshape=(None, n_q),
                             chunks=(max(1, min(n_rows, 32)), n_q),
                             **(comp_kwargs or {}))
    elif "sigma" in g:
        g["sigma"][pos] = np.full(n_q, np.nan, np.float32)


def _upsert_sigma_2d(g, pos, r, n_chi, n_q, comp_kwargs=None):
    """2D analogue of :func:`_upsert_sigma_1d` (sigma stored transposed)."""
    if r.sigma is not None:
        row = np.asarray(r.sigma, np.float32).T
        if "sigma" in g:
            g["sigma"][pos] = row
        else:
            n_rows = g["intensity"].shape[0]
            data = np.full((n_rows, n_chi, n_q), np.nan, np.float32)
            data[pos] = row
            g.create_dataset("sigma", data=data, maxshape=(None, n_chi, n_q),
                             chunks=(1, n_chi, n_q), **(comp_kwargs or {}))
    elif "sigma" in g:
        g["sigma"][pos] = np.full((n_chi, n_q), np.nan, np.float32)


def _append_stacked_2d(
    entry_grp: h5py.Group,
    frame_idx: int | str,
    r: IntegrationResult2D,
    comp_kwargs: dict[str, Any],
    *,
    group_name: str = "integrated_2d",
) -> None:
    """Append one IntegrationResult2D as a slice of the stacked
    ``/{entry}/integrated_2d`` NXdata group ``(n_frames, n_chi, n_q)``.

    ``IntegrationResult2D.intensity`` is ``(n_q, n_chi)`` (the integrate_2d
    convention); :func:`read_scan` reads ``(frame, chi, q)``, so each frame
    is transposed to ``(n_chi, n_q)`` on write.
    """
    intensity = np.asarray(r.intensity, dtype=np.float32).T  # (n_chi, n_q)
    n_chi, n_q = intensity.shape
    idx = int(frame_idx)

    if group_name not in entry_grp:
        g = _create_group_from_schema(
            entry_grp, "integrated_2d", disk_name=group_name,
        )
        g.attrs["two_d_kind"] = two_d_kind_from_units(r.unit, r.azimuthal_unit).value
        g.create_dataset("intensity", data=intensity[None],
                         maxshape=(None, n_chi, n_q),
                         chunks=(1, n_chi, n_q), **comp_kwargs)
        qd = g.create_dataset("q", data=np.asarray(r.radial, dtype=np.float32))
        qd.attrs["units"] = r.unit
        cd = g.create_dataset("chi", data=np.asarray(r.azimuthal, dtype=np.float32))
        cd.attrs["units"] = r.azimuthal_unit
        g.create_dataset("frame_index", data=np.asarray([idx], dtype=np.int64),
                         maxshape=(None,), chunks=(64,))
        g.attrs[MONOTONIC_ATTR] = True
        if r.sigma is not None:
            g.create_dataset("sigma",
                             data=np.asarray(r.sigma, dtype=np.float32).T[None],
                             maxshape=(None, n_chi, n_q),
                             chunks=(1, n_chi, n_q), **comp_kwargs)
        return

    g = entry_grp[group_name]
    if "two_d_kind" not in g.attrs:
        g.attrs["two_d_kind"] = two_d_kind_from_units(r.unit, r.azimuthal_unit).value
    di = g["intensity"]
    if di.shape[1:] != (n_chi, n_q):
        raise ValueError(
            f"{group_name} slice {(n_chi, n_q)} != on-disk {tuple(di.shape[1:])}; "
            "all frames in a scan must share the same (npt_azim, npt_rad)."
        )
    # Reject a frame whose q/chi axis differs from disk (see _append_stacked_1d).
    if not _axes_match_2d(g, r):
        raise ValueError(
            f"{group_name} q/chi axis or unit differs from what's on disk; "
            "a frame's axes must match the rest of the scan (use "
            "write_integrated_stack with all frames to re-axis a reintegration)."
        )
    fi = g["frame_index"]
    n = di.shape[0]
    monotonic = bool(g.attrs.get(MONOTONIC_ATTR, False))
    last_idx = int(fi[n - 1]) if n else None
    match = (
        np.empty(0, dtype=int)
        if monotonic and (last_idx is None or idx > last_idx)
        else np.where(np.asarray(fi[()]) == idx)[0]
    )
    if match.size:
        pos = int(match[0])  # upsert: replace existing row for this label
        di[pos] = intensity
        _upsert_sigma_2d(g, pos, r, n_chi, n_q, comp_kwargs)
        return
    di.resize(n + 1, axis=0)
    di[n] = intensity
    fi.resize(n + 1, axis=0)
    fi[n] = idx
    g.attrs[MONOTONIC_ATTR] = (
        monotonic and (last_idx is None or idx > last_idx)
    )
    _append_sigma_row(g, n, (None if r.sigma is None
                             else np.asarray(r.sigma, np.float32).T),
                      (n_chi, n_q), comp_kwargs)


def _require_uniform_axes_1d(results_1d) -> None:
    """Raise if any 1D result in the batch has a radial axis/unit differing
    from the first — they all share one stored ``q`` axis, so a divergent
    row would be silently mislabeled."""
    r0 = results_1d[0]
    q0 = np.asarray(r0.radial, np.float32)
    u0 = r0.unit or ""
    kind0 = _axis_kind_1d(u0)
    for i, r in enumerate(results_1d[1:], start=1):
        q = np.asarray(r.radial, np.float32)
        if q.shape != q0.shape or not np.allclose(q, q0, rtol=1e-5, atol=1e-8) \
                or (r.unit or "") != u0 or _axis_kind_1d(r.unit or "") != kind0:
            raise ValueError(
                f"results_1d[{i}] has a different radial axis/unit than "
                "results_1d[0]; all frames in a batch must share one q axis."
            )


_AZIMUTHAL_1D_UNITS = frozenset({
    "chi_deg",
    "chi_rad",
    "chigi_deg",
    "chigi_rad",
})


def _axis_kind_1d(unit: str | None) -> str:
    """Return the logical identity of a stacked 1D axis."""
    return "azimuthal" if (unit or "").lower() in _AZIMUTHAL_1D_UNITS else "radial"


def _axis_kind_from_group_1d(g: h5py.Group) -> str:
    """Read or infer the stacked 1D axis kind for back-compatible validation."""
    attr = _v2_decode_str(g.attrs.get("axis_kind", ""))
    if attr in ("radial", "azimuthal"):
        return attr
    (q_name,) = SCHEMA.groups["integrated_1d"].axes
    unit = _v2_decode_str(g[q_name].attrs.get("units", "")) if q_name in g else ""
    return _axis_kind_1d(unit)


def _require_uniform_axes_2d(results_2d) -> None:
    """Raise if any 2D result's q/chi axis or unit differs from the first."""
    r0 = results_2d[0]
    q0 = np.asarray(r0.radial, np.float32)
    c0 = np.asarray(r0.azimuthal, np.float32)
    u0 = r0.unit or ""
    au0 = getattr(r0, "azimuthal_unit", "") or ""
    for i, r in enumerate(results_2d[1:], start=1):
        q = np.asarray(r.radial, np.float32)
        c = np.asarray(r.azimuthal, np.float32)
        if (q.shape != q0.shape or not np.allclose(q, q0, rtol=1e-5, atol=1e-8)
                or c.shape != c0.shape
                or not np.allclose(c, c0, rtol=1e-5, atol=1e-8)
                or (r.unit or "") != u0
                or (getattr(r, "azimuthal_unit", "") or "") != au0):
            raise ValueError(
                f"results_2d[{i}] has a different q/chi axis or unit than "
                "results_2d[0]; all frames in a batch must share one axis set."
            )


def _axes_match_1d(g, r0) -> bool:
    """True if the on-disk integrated_1d q axis + units match ``r0``.

    Lets a same-bin-count reintegration upsert in place (axes unchanged)
    vs. rebuild (e.g. q_A^-1→2th_deg, or a different radial range at the
    same npt).  Missing q dataset → treated as a match."""
    (q_name,) = SCHEMA.groups["integrated_1d"].axes
    if q_name not in g:
        return True
    q = np.asarray(g[q_name][()])
    new_q = np.asarray(r0.radial, np.float32)
    if q.shape != new_q.shape or not np.allclose(q, new_q, rtol=1e-5, atol=1e-8):
        return False
    unit = r0.unit or ""
    return (
        _v2_decode_str(g[q_name].attrs.get("units", "")) == unit
        and _axis_kind_from_group_1d(g) == _axis_kind_1d(unit)
    )


def _axes_match_2d(g, r0) -> bool:
    """True if the on-disk integrated_2d q + chi axes + units match ``r0``."""
    q_name, chi_name = SCHEMA.groups["integrated_2d"].axes
    new_q = np.asarray(r0.radial, np.float32)
    new_chi = np.asarray(r0.azimuthal, np.float32)
    if q_name in g:
        q = np.asarray(g[q_name][()])
        if q.shape != new_q.shape or not np.allclose(q, new_q, rtol=1e-5, atol=1e-8):
            return False
        if _v2_decode_str(g[q_name].attrs.get("units", "")) != (r0.unit or ""):
            return False
    if chi_name in g:
        chi = np.asarray(g[chi_name][()])
        if chi.shape != new_chi.shape or not np.allclose(
            chi, new_chi, rtol=1e-5, atol=1e-8
        ):
            return False
        if _v2_decode_str(g[chi_name].attrs.get("units", "")) != (
            getattr(r0, "azimuthal_unit", "") or ""
        ):
            return False
    return True


def _require_batch_covers_existing(
    group: h5py.Group,
    name: str,
    frame_indices: Sequence[int],
) -> None:
    """Ensure a full-rewrite batch includes every persisted frame label."""
    existing = {int(x) for x in np.asarray(group["frame_index"][()]).ravel()}
    missing = existing - set(frame_indices)
    if missing:
        raise ValueError(
            f"write_integrated_stack: {name} row size changed but the "
            f"incoming batch is missing frame(s) {sorted(missing)} already "
            "on disk. A reintegration that changes the bin count must pass "
            "all frames (full rewrite), not a subset — otherwise the "
            "omitted rows would be dropped."
        )


def validate_integrated_stack_write(
    entry_grp: h5py.Group,
    *,
    frame_indices: Sequence[int],
    results_1d: Sequence[IntegrationResult1D] | None = None,
    results_2d: Sequence[IntegrationResult2D] | None = None,
    group_name_1d: str = "integrated_1d",
    group_name_2d: str = "integrated_2d",
) -> list[int]:
    """Validate a stacked integrated write without mutating ``entry_grp``.

    This mirrors the compatibility checks in :func:`write_integrated_stack`
    so callers that update multiple outputs can preflight every affected
    group before any one output is committed.
    """
    fis = [int(x) for x in frame_indices]
    if len(set(fis)) != len(fis):
        raise ValueError(f"frame_indices contains duplicate labels: {fis}")

    if results_1d is not None and len(results_1d):
        if len(results_1d) != len(fis):
            raise ValueError("results_1d length must match frame_indices")
        _require_uniform_axes_1d(results_1d)
        g = entry_grp.get(group_name_1d)
        if g is not None and (
            g["intensity"].shape[1] != np.asarray(results_1d[0].intensity).shape[0]
            or not _axes_match_1d(g, results_1d[0])
        ):
            _require_batch_covers_existing(g, group_name_1d, fis)

    if results_2d is not None and len(results_2d):
        if len(results_2d) != len(fis):
            raise ValueError("results_2d length must match frame_indices")
        _require_uniform_axes_2d(results_2d)
        g = entry_grp.get(group_name_2d)
        new_2d_shape = np.asarray(results_2d[0].intensity).T.shape
        if g is not None and (
            tuple(g["intensity"].shape[1:]) != new_2d_shape
            or not _axes_match_2d(g, results_2d[0])
        ):
            _require_batch_covers_existing(g, group_name_2d, fis)

    return fis


_NP_DTYPES = {"float32": np.float32, "int64": np.int64}


def _create_group_from_schema(
    entry_grp: h5py.Group, group_name: str, *, disk_name: str | None = None
) -> h5py.Group:
    """2b: create + stamp a group from its SCHEMA declaration.  Static NX
    attrs only — runtime-valued attrs (two_d_kind, the monotonic flag) are
    stamped by the caller.

    ``group_name`` selects the SCHEMA spec (datasets + nx_attrs); ``disk_name``
    overrides the on-disk child name so a nested per-mode subgroup (e.g.
    ``q_oop`` under ``integrated_1d``) is created with the SAME NXdata contract
    as its parent group.  ``disk_name=None`` ⇒ the canonical name ⇒ the
    existing top-level write is byte-identical."""
    spec = SCHEMA.groups[group_name]
    g = entry_grp.create_group(disk_name or group_name)
    for key, value in spec.nx_attrs.items():
        g.attrs[key] = list(value) if isinstance(value, tuple) else value
    return g


def _schema_dataset(g: h5py.Group, group_name: str, name: str, data,
                    *, ck: dict, row_chunk: tuple | None = None) -> h5py.Dataset:
    """2b: create one dataset per its DatasetSpec — dtype, compression and
    chunk strategy derive from the schema; SHAPES are runtime (row_chunk
    carries the rows/frame chunk tuple for row-aligned stacks)."""
    spec = SCHEMA.groups[group_name].datasets[name]
    data = np.asarray(data, _NP_DTYPES[spec.dtype])
    kwargs = {}
    if spec.chunk_style == "labels":
        kwargs.update(maxshape=(None,), chunks=(64,))
    elif spec.chunk_style in ("rows", "frame"):
        assert row_chunk is not None, (group_name, name)
        kwargs.update(maxshape=(None,) + data.shape[1:], chunks=row_chunk)
    if spec.compressed:
        kwargs.update(ck)
    return g.create_dataset(spec.name, data=data, **kwargs)


def _norm_attr(value):
    """Normalize an attr value (schema-declared or h5py-read) for comparison.

    numpy arrays / tuples / lists → list, bytes → str — exactly the shape
    ``_create_group_from_schema`` writes (it stores a declared tuple as a list,
    and h5py reads a string attr back as ``str`` or ``bytes`` by build), so a
    declared ``("q", "chi")`` compares equal to the on-disk ``["q", "chi"]``."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, (list, tuple)):
        return [_norm_attr(v) for v in value]
    return value


def validate_group_against_schema(group: h5py.Group,
                                  group_name: str) -> list[str]:
    """2d: check an on-disk group against its SCHEMA declaration.

    Returns a list of problem strings (empty = conformant).  Checks the
    DECLARED facts only: required datasets present, dtypes match, no
    row-aligned dataset disagrees with the row-label length, axes are
    1-D.  Presence of EXTRA datasets is allowed (additive-only format).
    This complements — never replaces — the strict write-time validators.
    """
    spec = SCHEMA.groups[group_name]
    problems: list[str] = []
    for name, ds_spec in spec.datasets.items():
        if name not in group:
            if ds_spec.required:
                problems.append(f"{group_name}/{name}: required, missing")
            continue
        node = group[name]
        if not isinstance(node, h5py.Dataset):
            problems.append(f"{group_name}/{name}: not a dataset")
            continue
        if node.dtype != np.dtype(_NP_DTYPES[ds_spec.dtype]):
            problems.append(
                f"{group_name}/{name}: dtype {node.dtype} != {ds_spec.dtype}"
            )
        if ds_spec.role == "axis" and node.ndim != 1:
            problems.append(f"{group_name}/{name}: axis must be 1-D")
    label = spec.datasets.get("frame_index")
    if label is not None and label.name in group:
        n = group[label.name].shape[0]
        for name, ds_spec in spec.datasets.items():
            if ds_spec.row_aligned and name in group and                     group[name].shape[0] != n:
                problems.append(
                    f"{group_name}/{name}: row count {group[name].shape[0]} "
                    f"!= frame_index length {n}"
                )

    # nx_attrs pass (Q-C2): every group writer is schema-routed via
    # _create_group_from_schema, which stamps these static NX attrs — so check
    # them by strict equality, normalized exactly as that writer stores them
    # (tuple→list, bytes→str), flagging absence too.  Runtime capability attrs
    # (two_d_kind, the monotonic flag, primary_mode) are deliberately NOT in
    # spec.nx_attrs, so they are never required here.
    for key, value in spec.nx_attrs.items():
        if key not in group.attrs:
            problems.append(f"{group_name}: nx_attr {key!r} missing")
            continue
        want = _norm_attr(value)
        got = _norm_attr(group.attrs[key])
        if got != want:
            problems.append(
                f"{group_name}: nx_attr {key!r} = {got!r} != {want!r}")
    return problems


def _stamp_mode_attrs(entry_grp, group_name, primary_mode, extra_modes) -> None:
    """Stamp the per-scan multi-result mode attrs on a top-level integrated
    group.  No-op for a ``DEFAULT_MODE_KEY`` / ``None`` primary (a standard or
    unnamed scan ⇒ byte-identical, no new attrs).  A named GI primary records
    ``primary_mode`` + ``multi_result_modes = [primary, *extras]`` — primary
    FIRST, stored as an order-preserving h5py vlen-string array (never a set;
    the reader and ``modes_*()[0] == primary`` depend on the order)."""
    if primary_mode is None or primary_mode == DEFAULT_MODE_KEY:
        return
    g = entry_grp.get(group_name)
    if g is None:
        return
    g.attrs[PRIMARY_MODE_ATTR] = primary_mode
    existing = _norm_attr(g.attrs.get(MULTI_RESULT_MODES_ATTR, []))
    if isinstance(existing, str):
        existing = [existing]
    modes: list[str] = [str(primary_mode)]
    for mode in [*(existing or []), *((extra_modes or {}).keys())]:
        mode = str(mode)
        if mode != primary_mode and mode not in modes:
            modes.append(mode)
    g.attrs[MULTI_RESULT_MODES_ATTR] = modes


def write_integrated_stack(
    entry_grp: h5py.Group,
    *,
    frame_indices: Sequence[int],
    results_1d: Sequence[IntegrationResult1D] | None = None,
    results_2d: Sequence[IntegrationResult2D] | None = None,
    extra_modes_1d: "Mapping[str, Sequence[IntegrationResult1D]] | None" = None,
    extra_modes_2d: "Mapping[str, Sequence[IntegrationResult2D]] | None" = None,
    extra_mode_indices_1d: "Mapping[str, Sequence[int]] | None" = None,
    extra_mode_indices_2d: "Mapping[str, Sequence[int]] | None" = None,
    primary_mode_1d: str | None = None,
    primary_mode_2d: str | None = None,
    group_name_1d: str = "integrated_1d",
    group_name_2d: str = "integrated_2d",
    compression: str | None = None,
) -> None:
    """Write/extend the stacked ``integrated_1d`` / ``integrated_2d`` NXdata
    groups from aligned lists of IntegrationResult + their frame labels.

    Multi-result GI (ADR-0003): ``results_1d``/``results_2d`` are the PRIMARY
    mode → the top-level group (meaning unchanged); ``extra_modes_*`` maps each
    NON-primary GI ``mode_key`` to its aligned results list → a nested
    ``integrated_<dim>/<mode>/`` NXdata subgroup (same contract as the parent).
    ``primary_mode_*`` names the per-scan top-level slot: when it is a named GI
    mode the group is stamped with ``primary_mode`` + ``multi_result_modes``
    (primary first); a ``DEFAULT_MODE_KEY``/``None`` primary stamps nothing, so
    a standard (non-GI) scan stays byte-identical.  Each group — top-level and
    every nested child — is validated independently by the frozen uniform-axes
    validators; nested children never affect top-level validation.

    Nested modes may be partial: ``extra_mode_indices_*[mode]`` supplies the
    labels for that subgroup, defaulting to ``frame_indices`` for the original
    all-modes-aligned call shape.  Existing nested groups are upserted/appended
    with the same axis/shape guards as the top-level stacks.

    The canonical stacked-write primitive — the headless path and
    (eventually, #18) the xdart GUI writer both target this so a single
    implementation owns the on-disk layout ``read_scan`` consumes.

    First save (group absent) creates the whole stack in one write — O(N),
    with multi-frame chunks.  Subsequent calls fall back to the per-frame
    upsert appenders (:func:`_append_stacked_1d` / ``_2d``), so re-saving a
    frame label replaces its row rather than duplicating it.  ``int64``
    frame_index; ``compression`` applies to the intensity/sigma stacks.

    Reintegration shape change (C3): if the incoming row size differs from
    what's on disk (a different ``npt`` / ``npt_rad`` / ``npt_azim``), the
    existing group is dropped and rewritten from this batch so the q/chi
    axes refresh — pass *all* frames in that case, not a subset.
    """
    fis = validate_integrated_stack_write(
        entry_grp,
        frame_indices=frame_indices,
        results_1d=results_1d,
        results_2d=results_2d,
        group_name_1d=group_name_1d,
        group_name_2d=group_name_2d,
    )
    ck = _comp_kwargs(compression)

    def _bulk_create_1d(parent, results, fis_, *, disk_name=None):
        # 2b: names/dtypes/chunk-strategy/compression derive from SCHEMA;
        # only the runtime shapes and values are computed here.  ``parent`` +
        # ``disk_name`` let the SAME body write either the top-level group
        # (disk_name=None ⇒ byte-identical) or a nested per-mode subgroup.
        r0 = results[0]
        n_q = np.asarray(r0.intensity).shape[0]
        rows = max(1, min(len(fis_), 32))
        g = _create_group_from_schema(parent, "integrated_1d", disk_name=disk_name)
        _schema_dataset(
            g, "integrated_1d", "intensity",
            np.stack([np.asarray(r.intensity, np.float32) for r in results]),
            ck=ck, row_chunk=(rows, n_q),
        )
        qd = _schema_dataset(g, "integrated_1d", "q", r0.radial, ck=ck)
        qd.attrs["units"] = r0.unit            # units_from="radial_unit"
        _schema_dataset(g, "integrated_1d", "frame_index", fis_, ck=ck)
        axis_kind = _axis_kind_1d(r0.unit)
        if axis_kind != "radial":
            g.attrs["axis_kind"] = axis_kind
        g.attrs[MONOTONIC_ATTR] = bool(
            len(fis_) < 2 or np.all(np.diff(fis_) > 0)
        )
        # Write sigma if ANY frame has it (NaN-pad the frames that don't) —
        # an all-or-nothing test silently dropped sigma for a mixed batch.
        if any(r.sigma is not None for r in results):
            sig = np.stack([
                (np.asarray(r.sigma, np.float32) if r.sigma is not None
                 else np.full(n_q, np.nan, np.float32))
                for r in results
            ])
            _schema_dataset(g, "integrated_1d", "sigma", sig,
                            ck=ck, row_chunk=(rows, n_q))

    def _bulk_create_2d(parent, results, fis_, *, disk_name=None):
        # 2b: schema-derived like _bulk_create_1d; two_d_kind and the
        # monotonic flag are runtime-valued capability attrs.
        r0 = results[0]
        stacked = np.stack(
            [np.asarray(r.intensity, np.float32).T for r in results]
        )  # (N, n_chi, n_q)
        n_chi, n_q = stacked.shape[1], stacked.shape[2]
        # Block the leading (frame) axis so a streaming reintegrate's per-row
        # appends after batch 1 reuse a chunk instead of allocating one per
        # frame (F2).  Byte-budgeted (~2 MB) so a large cake stays at one frame
        # per chunk (unchanged) while small cakes block up to 32 frames.
        rows_2d = max(1, min(len(fis_), 32,
                             (2 << 20) // max(1, n_chi * n_q * 4)))
        g = _create_group_from_schema(parent, "integrated_2d", disk_name=disk_name)
        g.attrs["two_d_kind"] = two_d_kind_from_units(
            r0.unit, r0.azimuthal_unit
        ).value
        _schema_dataset(g, "integrated_2d", "intensity", stacked,
                        ck=ck, row_chunk=(rows_2d, n_chi, n_q))
        qd = _schema_dataset(g, "integrated_2d", "q", r0.radial, ck=ck)
        qd.attrs["units"] = r0.unit            # units_from="radial_unit"
        cd = _schema_dataset(g, "integrated_2d", "chi", r0.azimuthal, ck=ck)
        cd.attrs["units"] = r0.azimuthal_unit  # units_from="azimuthal_unit"
        _schema_dataset(g, "integrated_2d", "frame_index", fis_, ck=ck)
        g.attrs[MONOTONIC_ATTR] = bool(
            len(fis_) < 2 or np.all(np.diff(fis_) > 0)
        )
        # Sigma if ANY frame has it (NaN-pad the rest) — see _bulk_create_1d.
        if any(r.sigma is not None for r in results):
            sig = np.stack([
                (np.asarray(r.sigma, np.float32).T if r.sigma is not None
                 else np.full((n_chi, n_q), np.nan, np.float32))
                for r in results
            ])
            _schema_dataset(g, "integrated_2d", "sigma", sig,
                            ck=ck, row_chunk=(rows_2d, n_chi, n_q))

    if results_1d is not None and len(results_1d):
        if len(results_1d) != len(fis):
            raise ValueError("results_1d length must match frame_indices")
        # Every row must share one radial axis + unit — they all land under
        # the single stored ``q`` axis, so a row with a different grid would
        # be silently mislabeled.  Validate the whole batch BEFORE touching
        # the file (checking only results[0] missed later divergent rows).
        _require_uniform_axes_1d(results_1d)
        g = entry_grp.get(group_name_1d)
        # Rebuild trigger: reintegration that changes the npt (row size) OR
        # the radial axis / unit (e.g. q_A^-1 → 2th_deg, or a different
        # radial range at the same bin count).  The per-frame upsert path
        # only rewrites intensity, leaving the stored q axis + units stale,
        # so any axis change must drop the group and rewrite from this
        # (full) batch.  Same-axis re-saves take the upsert path.
        if g is not None and (
            g["intensity"].shape[1] != np.asarray(results_1d[0].intensity).shape[0]
            or not _axes_match_1d(g, results_1d[0])
        ):
            _require_batch_covers_existing(g, group_name_1d, fis)
            del entry_grp[group_name_1d]
            g = None
        if g is None:
            _bulk_create_1d(entry_grp, results_1d, fis, disk_name=group_name_1d)
        else:
            for fi, r in zip(fis, results_1d):
                _append_stacked_1d(entry_grp, fi, r, ck, group_name=group_name_1d)

    if results_2d is not None and len(results_2d):
        if len(results_2d) != len(fis):
            raise ValueError("results_2d length must match frame_indices")
        _require_uniform_axes_2d(results_2d)
        g = entry_grp.get(group_name_2d)
        new_2d_shape = np.asarray(results_2d[0].intensity).T.shape  # (n_chi, n_q)
        # Rebuild on a row-shape change OR a q/chi axis / unit change (see
        # the 1D block) — the upsert path can't refresh the stored axes.
        if g is not None and (
            tuple(g["intensity"].shape[1:]) != new_2d_shape
            or not _axes_match_2d(g, results_2d[0])
        ):
            _require_batch_covers_existing(g, group_name_2d, fis)
            del entry_grp[group_name_2d]
            g = None
        if g is None:
            _bulk_create_2d(entry_grp, results_2d, fis, disk_name=group_name_2d)
        else:
            for fi, r in zip(fis, results_2d):
                _append_stacked_2d(entry_grp, fi, r, ck, group_name=group_name_2d)

    def _mode_fis(mode_indices, mode_key, default_fis):
        fis_ = [int(x) for x in (
            mode_indices.get(mode_key) if mode_indices and mode_key in mode_indices
            else default_fis
        )]
        if len(set(fis_)) != len(fis_):
            raise ValueError(
                f"extra mode {mode_key!r} frame_indices contains duplicates: {fis_}"
            )
        return fis_

    def _write_extra_1d(parent, mode_key, results, fis_):
        sub = mode_subgroup_name(mode_key)  # canonical; raises on default/unknown
        if not results or len(results) != len(fis_):
            raise ValueError(
                f"extra_modes_1d[{mode_key!r}] length must match its frame_indices"
            )
        _require_uniform_axes_1d(results)
        g = parent.get(sub)
        if g is not None and (
            g["intensity"].shape[1] != np.asarray(results[0].intensity).shape[0]
            or not _axes_match_1d(g, results[0])
        ):
            _require_batch_covers_existing(g, f"{group_name_1d}/{sub}", fis_)
            del parent[sub]
            g = None
        if g is None:
            _bulk_create_1d(parent, results, fis_, disk_name=sub)
        else:
            for fi, r in zip(fis_, results):
                _append_stacked_1d(parent, fi, r, ck, group_name=sub)

    def _write_extra_2d(parent, mode_key, results, fis_):
        sub = mode_subgroup_name(mode_key)
        if not results or len(results) != len(fis_):
            raise ValueError(
                f"extra_modes_2d[{mode_key!r}] length must match its frame_indices"
            )
        _require_uniform_axes_2d(results)
        g = parent.get(sub)
        new_shape = np.asarray(results[0].intensity).T.shape
        if g is not None and (
            tuple(g["intensity"].shape[1:]) != new_shape
            or not _axes_match_2d(g, results[0])
        ):
            _require_batch_covers_existing(g, f"{group_name_2d}/{sub}", fis_)
            del parent[sub]
            g = None
        if g is None:
            _bulk_create_2d(parent, results, fis_, disk_name=sub)
        else:
            for fi, r in zip(fis_, results):
                _append_stacked_2d(parent, fi, r, ck, group_name=sub)

    # ── nested per-mode GI subgroups (ADR-0003) ─────────────────────────────
    # Each non-primary mode is its own NXdata child of the top-level group,
    # validated INDEPENDENTLY by the frozen uniform-axes validators (a child's
    # axis grid legitimately differs from the primary's).  First-save only.
    if extra_modes_1d:
        parent = entry_grp.get(group_name_1d)
        if parent is None:
            raise ValueError(
                "extra_modes_1d requires results_1d (the primary 1D mode at "
                "the top-level group)"
            )
        if primary_mode_1d is None or primary_mode_1d == DEFAULT_MODE_KEY:
            raise ValueError(
                "extra_modes_1d requires a named primary_mode_1d so the file "
                "records its multi-mode capability (got "
                f"{primary_mode_1d!r})"
            )
        if primary_mode_1d in extra_modes_1d:
            raise ValueError(
                f"primary_mode_1d {primary_mode_1d!r} must not also appear in "
                "extra_modes_1d (the primary lives at the top-level group, not "
                "a subgroup)"
            )
        for mode_key, results in extra_modes_1d.items():
            _write_extra_1d(
                parent, mode_key, results,
                _mode_fis(extra_mode_indices_1d, mode_key, fis),
            )

    if extra_modes_2d:
        parent = entry_grp.get(group_name_2d)
        if parent is None:
            raise ValueError(
                "extra_modes_2d requires results_2d (the primary 2D mode at "
                "the top-level group)"
            )
        if primary_mode_2d is None or primary_mode_2d == DEFAULT_MODE_KEY:
            raise ValueError(
                "extra_modes_2d requires a named primary_mode_2d so the file "
                "records its multi-mode capability (got "
                f"{primary_mode_2d!r})"
            )
        if primary_mode_2d in extra_modes_2d:
            raise ValueError(
                f"primary_mode_2d {primary_mode_2d!r} must not also appear in "
                "extra_modes_2d (the primary lives at the top-level group, not "
                "a subgroup)"
            )
        for mode_key, results in extra_modes_2d.items():
            _write_extra_2d(
                parent, mode_key, results,
                _mode_fis(extra_mode_indices_2d, mode_key, fis),
            )

    # Stamp the per-scan mode attrs LAST, per dimension (no-op for a DEFAULT/
    # unnamed primary ⇒ standard scans stay byte-identical).
    _stamp_mode_attrs(entry_grp, group_name_1d, primary_mode_1d, extra_modes_1d)
    _stamp_mode_attrs(entry_grp, group_name_2d, primary_mode_2d, extra_modes_2d)


def frame_record_write_parts(
    records, *, include_1d: bool = True, include_2d: bool = True
) -> dict[str, Any]:
    """Convert multi-result FrameRecords into ``write_integrated_stack`` kwargs.

    The top-level primary mode remains aligned to the full record list; nested
    extra modes carry their own per-mode frame indices so lazy/partial mode
    accumulation persists only the rows that actually exist.
    """
    from xrd_tools.core.frame_view import view_to_result_1d, view_to_result_2d

    records = list(records)
    fis = [int(r.label) for r in records]

    prim_1d = {r.active_mode_1d for r in records if include_1d and r.results_1d}
    prim_2d = {r.active_mode_2d for r in records if include_2d and r.results_2d}
    if len(prim_1d) > 1 or len(prim_2d) > 1:
        raise ValueError(
            f"FrameRecords disagree on active mode (1D={sorted(prim_1d)}, "
            f"2D={sorted(prim_2d)}); one scan = one primary per dimension"
        )
    primary_1d = next(iter(prim_1d), DEFAULT_MODE_KEY)
    primary_2d = next(iter(prim_2d), DEFAULT_MODE_KEY)

    # The stacked writer needs one aligned row per frame.  Partial /
    # per-frame-varying mode sets (some frames have a dimension, others don't,
    # or a record omits the agreed primary) aren't supported yet; fail with a
    # precise message rather than a downstream length-mismatch.
    has_1d = [bool(include_1d and r.results_1d) for r in records]
    has_2d = [bool(include_2d and r.results_2d) for r in records]
    if any(has_1d) and not all(has_1d):
        miss = [r.label for r, h in zip(records, has_1d) if not h]
        raise ValueError(
            f"frames {miss} have no 1D results while others do; per-frame-"
            "varying mode sets are not yet supported"
        )
    if any(has_2d) and not all(has_2d):
        miss = [r.label for r, h in zip(records, has_2d) if not h]
        raise ValueError(
            f"frames {miss} have no 2D results while others do; per-frame-"
            "varying mode sets are not yet supported"
        )
    for r in records:  # defensive: every record carries the agreed primary
        if include_1d and r.results_1d and primary_1d not in r.results_1d:
            raise ValueError(f"frame {r.label} lacks the primary 1D mode {primary_1d!r}")
        if include_2d and r.results_2d and primary_2d not in r.results_2d:
            raise ValueError(f"frame {r.label} lacks the primary 2D mode {primary_2d!r}")

    top_1d: list = []
    extra_1d: dict = {}
    extra_idx_1d: dict = {}
    for r in (records if include_1d else ()):
        for mode, view in r.results_1d.items():
            res = view_to_result_1d(view)
            if mode == primary_1d:
                top_1d.append(res)
            else:
                extra_1d.setdefault(mode, []).append(res)
                extra_idx_1d.setdefault(mode, []).append(int(r.label))

    top_2d: list = []
    extra_2d: dict = {}
    extra_idx_2d: dict = {}
    for r in (records if include_2d else ()):
        for mode, view in r.results_2d.items():
            res = view_to_result_2d(view)
            if mode == primary_2d:
                top_2d.append(res)
            else:
                extra_2d.setdefault(mode, []).append(res)
                extra_idx_2d.setdefault(mode, []).append(int(r.label))

    return {
        "frame_indices": fis,
        "results_1d": top_1d or None,
        "results_2d": top_2d or None,
        "extra_modes_1d": extra_1d or None,
        "extra_modes_2d": extra_2d or None,
        "extra_mode_indices_1d": extra_idx_1d or None,
        "extra_mode_indices_2d": extra_idx_2d or None,
        "primary_mode_1d": primary_1d,
        "primary_mode_2d": primary_2d,
    }


def write_frame_records(entry_grp: h5py.Group, records, *, compression=None) -> None:
    """Persist a stack of multi-result :class:`FrameRecord`s through the single
    stacked-write path (the ergonomic seam over :func:`write_integrated_stack`).

    All records must agree on their per-dimension active mode (= the per-scan
    primary); a diverging active mode is a caller bug → raise.  Each record's
    active-mode view goes to the top-level group; every other ``mode_key`` →
    its nested ``integrated_<dim>/<mode>/`` subgroup.  ONE
    ``write_integrated_stack`` call.

    Byte-compat collapse: a standard stack (every record carries only the
    ``DEFAULT_MODE_KEY`` mode) passes ``primary_mode_*=DEFAULT_MODE_KEY`` +
    no ``extra_modes_*`` ⇒ the legacy path ⇒ no new attr/group ⇒ byte-identical.
    A single NAMED GI mode persists ``primary_mode`` (additive on GI files only)
    so the active mode round-trips.
    """
    parts = frame_record_write_parts(records)

    write_integrated_stack(
        entry_grp,
        **parts,
        compression=compression,
    )


def write_stitched(
    entry_grp: h5py.Group,
    *,
    stitched_1d: IntegrationResult1D | None = None,
    stitched_2d: IntegrationResult2D | None = None,
    provenance: "Mapping[str, object] | str | None" = None,
    frame_records=None,
    source_base=None,
    compression: str | None = None,
) -> None:
    """Write ``/entry/stitched_1d`` / ``/entry/stitched_2d`` — the symmetric
    counterpart to :func:`read_stitched`.

    Note the orientation: unlike the per-frame ``integrated_2d`` stack
    (stored ``(frame, chi, q)``), ``stitched_2d/intensity`` is stored
    **as-is** ``(n_q, n_chi)`` and read back with dims ``(q, chi)`` — this
    matches the existing xdart writer + ``read_stitched`` so files stay
    interchangeable.  Each group is replaced atomically (idempotent).

    ``provenance`` (the StitchPlan + applied CorrectionStack — typically
    ``StitchPlan.provenance()``) is stamped into each written group as a
    ``provenance_json`` vlen-UTF8 blob, the same idiom as the diffractometer
    ``config_json``; ``None`` writes no blob.  A dict is JSON-encoded; a str is
    written verbatim (assumed already-JSON).
    """
    ck = _comp_kwargs(compression)
    prov_json: str | None = None
    if provenance is not None:
        prov_json = provenance if isinstance(provenance, str) else json.dumps(
            provenance, default=str)

    if stitched_1d is not None:
        if "stitched_1d" in entry_grp:
            del entry_grp["stitched_1d"]
        # 2b: group + NX attrs + dataset dtypes/compression derive from SCHEMA
        # (byte-identical to the hand-written form); only the per-axis units and
        # the provenance_json blob (an additive extra) are stamped by hand.
        g = _create_group_from_schema(entry_grp, "stitched_1d")
        _schema_dataset(g, "stitched_1d", "intensity", stitched_1d.intensity, ck=ck)
        qd = _schema_dataset(g, "stitched_1d", "q", stitched_1d.radial, ck=ck)
        qd.attrs["units"] = stitched_1d.unit            # units_from="radial_unit"
        if stitched_1d.sigma is not None:
            _schema_dataset(g, "stitched_1d", "sigma", stitched_1d.sigma, ck=ck)
        if prov_json is not None:
            g.create_dataset("provenance_json", data=prov_json, dtype=_UTF8_DTYPE)

    if stitched_2d is not None:
        # Fail loud on a transposed cake: read_stitched blindly applies (q, chi)
        # to the stored array and the stitched validator skips the row-count
        # block, so a transposed (n_chi, n_q) array round-trips as silently
        # wrong axes for a square cake.  Enforce the (n_q, n_chi) contract at the
        # write boundary (complements, never relaxes, the validators).
        _i2d = np.asarray(stitched_2d.intensity, np.float32)
        _exp = (len(stitched_2d.radial), len(stitched_2d.azimuthal))
        if _i2d.shape != _exp:
            raise ValueError(
                f"stitched_2d.intensity shape {_i2d.shape} != (len(radial), "
                f"len(azimuthal)) = {_exp}; the stored cake must be (n_q, n_chi).")
        if "stitched_2d" in entry_grp:
            del entry_grp["stitched_2d"]
        # 2b: schema-routed group + datasets (byte-identical); the (n_q, n_chi)
        # intensity is stored as-is — see docstring.  Per-axis units + the
        # provenance_json blob stay hand-stamped.
        g = _create_group_from_schema(entry_grp, "stitched_2d")
        _schema_dataset(g, "stitched_2d", "intensity", _i2d, ck=ck)
        qd = _schema_dataset(g, "stitched_2d", "q", stitched_2d.radial, ck=ck)
        qd.attrs["units"] = stitched_2d.unit            # units_from="radial_unit"
        cd = _schema_dataset(g, "stitched_2d", "chi", stitched_2d.azimuthal, ck=ck)
        cd.attrs["units"] = stitched_2d.azimuthal_unit  # units_from="azimuthal_unit"
        if prov_json is not None:
            g.create_dataset("provenance_json", data=prov_json, dtype=_UTF8_DTYPE)

    if frame_records:
        from xrd_tools.io.nexus_record import write_contributing_frames  # noqa: PLC0415
        write_contributing_frames(entry_grp, frame_records, source_base=source_base)


def write_rsm(
    entry_grp: h5py.Group,
    volume: Any,
    *,
    provenance: "Mapping[str, object] | str | None" = None,
    frame_records=None,
    source_base=None,
    compression: str | None = None,
) -> None:
    """Write ``/entry/rsm`` — a gridded :class:`~xrd_tools.rsm.RSMVolume` as an
    NXdata group (``h``/``k``/``l`` axes + the 3D ``intensity``), plus an optional
    ``provenance_json`` blob (the RSMPlan + applied CorrectionStack), the same
    idiom as :func:`write_stitched` / the diffractometer ``config_json``.  The
    group is replaced atomically (idempotent).
    """
    ck = _comp_kwargs(compression)
    intensity = np.asarray(volume.intensity, np.float32)
    expected = (len(volume.h), len(volume.k), len(volume.l))
    if intensity.shape != expected:
        raise ValueError(
            f"rsm.intensity shape {intensity.shape} != "
            f"(len(h), len(k), len(l)) = {expected}; the stored volume must be "
            "(n_h, n_k, n_l)."
        )
    if "rsm" in entry_grp:
        del entry_grp["rsm"]
    # 2b: schema-routed group + datasets (byte-identical).  h/k/l carry NO units
    # (no units_from in the schema) — do not add any, it would change the bytes.
    g = _create_group_from_schema(entry_grp, "rsm")
    _schema_dataset(g, "rsm", "intensity", intensity, ck=ck)
    for name, axis in (("h", volume.h), ("k", volume.k), ("l", volume.l)):
        _schema_dataset(g, "rsm", name, axis, ck=ck)
    if provenance is not None:
        prov = provenance if isinstance(provenance, str) else json.dumps(
            provenance, default=str)
        g.create_dataset("provenance_json", data=prov, dtype=_UTF8_DTYPE)

    if frame_records:
        from xrd_tools.io.nexus_record import write_contributing_frames  # noqa: PLC0415
        write_contributing_frames(entry_grp, frame_records, source_base=source_base)


def read_rsm(path: Path | str, *, entry: str = "entry"):
    """Read ``/entry/rsm`` into an :class:`~xrd_tools.rsm.RSMVolume`.

    The ``provenance_json`` blob (if present) is parsed onto
    :attr:`RSMVolume.provenance`.  Raises :class:`KeyError` when the entry or the
    ``rsm`` group is absent.
    """
    from xrd_tools.rsm.volume import RSMVolume  # noqa: PLC0415

    path = Path(path)
    with h5py.File(path, "r") as f:
        if entry not in f:
            raise KeyError(f"No {entry!r} group in {path}")
        e = f[entry]
        if "rsm" not in e:
            raise KeyError(f"No rsm group in {path}:{entry}")
        g = e["rsm"]
        prov = None
        if "provenance_json" in g:
            raw = g["provenance_json"][()]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            try:
                prov = json.loads(raw)
            except (ValueError, TypeError):
                prov = str(raw)
        return RSMVolume(
            h=np.asarray(g["h"][()]), k=np.asarray(g["k"][()]),
            l=np.asarray(g["l"][()]), intensity=np.asarray(g["intensity"][()]),
            provenance=prov,
        )


def _reindex_scan_data_to_frames(scan_data, frame_indices):
    """Return ``scan_data`` rows aligned 1:1 to ``frame_indices``.

    Reindexes when the scan_data index isn't already exactly the frame ids
    (different order, gaps, or a shorter batch — missing rows become NaN),
    so positioners / per-frame geometry always share the integrated-frame
    dimension and can't attach a motor value to the wrong frame.
    """
    fis = [int(x) for x in frame_indices]
    if len(fis) != len(set(fis)):
        raise ValueError(f"frame_indices contains duplicate labels: {fis}")
    try:
        labels = [int(x) for x in scan_data.index]
    except (TypeError, ValueError):
        labels = []
    if labels and len(labels) != len(set(labels)):
        raise ValueError(f"scan_data index contains duplicate labels: {labels}")
    if fis and list(scan_data.index) != fis:
        scan_data = scan_data.reindex(fis)
    return scan_data, fis


def write_positioners(
    entry_grp: h5py.Group,
    scan_data,
    frame_indices: Sequence[int],
    geometry,
    *,
    compression: str | None = None,
) -> None:
    """Write motor positioners under ``/entry/sample`` and
    ``/entry/instrument/detector`` (NXcollection of NXpositioner/value).

    Sample- vs detector-axis split comes from ``geometry``; no geometry → no-op.
    ``scan_data`` is aligned to ``frame_indices`` first (see
    :func:`_reindex_scan_data_to_frames`).  Mirrors what
    :func:`read_scan_metadata` reads back.
    """
    def _write_coll(parent_path: str, motors, nx_class: str) -> None:
        parent = entry_grp.get(parent_path)
        if parent is not None and "positioners" in parent:
            del parent["positioners"]
        present = [m for m in motors if m in scan_data.columns]
        if not present:
            return
        parent = entry_grp.require_group(parent_path)
        parent.attrs["NX_class"] = nx_class
        coll = parent.create_group("positioners")
        coll.attrs["NX_class"] = "NXcollection"
        coll.attrs[MONOTONIC_ATTR] = bool(
            len(fis) < 2 or np.all(np.diff(fis) > 0)
        )
        coll.create_dataset(
            "frame_index",
            data=np.asarray(fis, dtype=np.int64),
            maxshape=(None,),
            chunks=(64,),
        )
        for m in present:
            pg = coll.create_group(m)
            pg.attrs["NX_class"] = "NXpositioner"
            ds = pg.create_dataset(
                "value",
                data=np.asarray(scan_data[m].values, dtype=np.float32),
                maxshape=(None,),
                chunks=(64,),
                **ck,
            )
            ds.attrs["units"] = "deg"

    sample_motors = tuple(geometry.sample_motors) if geometry is not None else ()
    detector_motors = tuple(geometry.detector_motors) if geometry is not None else ()
    if geometry is None or scan_data is None or len(scan_data) == 0:
        for path in ("sample", "instrument/detector"):
            parent = entry_grp.get(path)
            if parent is not None and "positioners" in parent:
                del parent["positioners"]
        return
    scan_data, fis = _reindex_scan_data_to_frames(scan_data, frame_indices)
    ck = _comp_kwargs(compression)

    if detector_motors:
        entry_grp.require_group("instrument").attrs["NX_class"] = "NXinstrument"
    _write_coll("sample", sample_motors, "NXsample")
    _write_coll("instrument/detector", detector_motors, "NXdetector")


def write_per_frame_geometry(
    entry_grp: h5py.Group,
    scan_data,
    frame_indices: Sequence[int],
    geometry,
    *,
    compression: str | None = None,
) -> None:
    """Write ``/entry/per_frame_geometry`` (rot1/2/3 + incident_angle +
    frame_index) derived from ``scan_data`` motor columns via
    ``geometry.derive_per_frame``.

    ``scan_data`` is aligned to ``frame_indices`` first so the per-frame
    dimension matches ``integrated_1d``/``integrated_2d`` (a missing row →
    NaN-padded → NaN derived geometry, honestly flagged rather than
    misaligned).  No geometry / no usable motor columns → no-op.
    """
    if geometry is None or scan_data is None or len(scan_data) == 0:
        if "per_frame_geometry" in entry_grp:
            del entry_grp["per_frame_geometry"]
        return
    # Validate/align BEFORE deleting the authoritative group: a malformed
    # frame_indices (e.g. duplicate labels) makes _reindex raise, and we must
    # not have already destroyed the existing on-disk group at that point.
    scan_data, fis = _reindex_scan_data_to_frames(scan_data, frame_indices)
    # Past validation — every exit below clears the stale group (matching the
    # original always-rewrite behaviour) but only now that nothing can raise.
    if "per_frame_geometry" in entry_grp:
        del entry_grp["per_frame_geometry"]
    motors = {
        m: np.asarray(scan_data[m].values, dtype=float)
        for m in geometry.all_referenced_motors()
        if m in scan_data.columns
    }
    if not motors:
        return
    try:
        derived = geometry.derive_per_frame(motors)
    except Exception:
        logger.debug("per_frame_geometry: derive_per_frame failed", exc_info=True)
        return

    frame_idx_arr = np.asarray(fis if fis else range(len(scan_data)), dtype=np.int64)
    ck = _comp_kwargs(compression)
    g = _create_group_from_schema(entry_grp, "per_frame_geometry")
    g.attrs[MONOTONIC_ATTR] = bool(
        len(frame_idx_arr) < 2 or np.all(np.diff(frame_idx_arr) > 0)
    )
    _schema_dataset(g, "per_frame_geometry", "frame_index", frame_idx_arr,
                    ck=ck)
    geo_specs = SCHEMA.groups["per_frame_geometry"].datasets
    for key, arr in derived.items():
        spec = geo_specs.get(key)
        if spec is not None:
            ds = _schema_dataset(g, "per_frame_geometry", key, arr, ck=ck)
            ds.attrs["units"] = spec.units_from
        else:
            # future derive_per_frame outputs not yet declared: identical
            # legacy fallback (float32, compressed, rad)
            ds = g.create_dataset(key, data=np.asarray(arr, dtype=np.float32),
                                  maxshape=(None,), chunks=(64,), **ck)
            ds.attrs["units"] = "rad"


def write_diffractometer(entry_grp: h5py.Group, diffractometer) -> None:
    """Write ``/entry/diffractometer`` — the canonical :class:`Diffractometer`
    serialized to a single ``config_json`` vlen-UTF8 string blob.

    Scan-level + capability-gated (``diffractometer``): a reloaded scan
    reconstructs the full instrument geometry (both adapter views + the fitted
    ``DetectorCalibration`` + preset + motor map) for offline stitch/RSM with
    no GUI.  ``None`` (or a geometry without ``to_json``) → no-op / clears any
    stale group, so an old file simply lacks the group (reader → ``None``).
    """
    to_json = getattr(diffractometer, "to_json", None)
    if diffractometer is None or not callable(to_json):
        if "diffractometer" in entry_grp:
            del entry_grp["diffractometer"]
        return
    blob = to_json()
    if "diffractometer" in entry_grp:
        del entry_grp["diffractometer"]
    g = _create_group_from_schema(entry_grp, "diffractometer")
    g.create_dataset("config_json", data=blob, dtype=_UTF8_DTYPE)


def write_scan_metadata(
    entry_grp: h5py.Group,
    scan_data,
    frame_indices: Sequence[int],
    *,
    compression: str | None = None,
) -> None:
    """Persist the **full** per-frame scan metadata table under
    ``/entry/scan_data`` (NXcollection: ``frame_index`` + one typed dataset
    per column).

    The ``positioners`` group only carries the geometry-referenced motors;
    this stores the complete ``scan_data`` DataFrame — every counter and
    motor the wrangler recorded (monitor, i0, temperature, …) — so that a
    reload restores the same metadata the live in-memory scan had, not just
    the geometry motors.  ``read_scan`` / ``read_scan_metadata`` surface
    these columns as per-frame variables (preferred over positioners, which
    they then read only for any motor not already present here).

    Numeric columns are stored as float32. Non-numeric columns are stored as
    UTF-8 variable-length strings so notebook/headless readers preserve
    labels, modes, source file names, operator notes, and similar context.
    """
    if scan_data is None or len(scan_data) == 0 or not len(scan_data.columns):
        if "scan_data" in entry_grp:
            del entry_grp["scan_data"]
        return
    # Validate/align BEFORE deleting the authoritative group so a malformed
    # frame_indices (duplicate labels) raises without losing existing data.
    scan_data, fis = _reindex_scan_data_to_frames(scan_data, frame_indices)
    if "scan_data" in entry_grp:
        del entry_grp["scan_data"]
    ck = _comp_kwargs(compression)
    g = entry_grp.create_group("scan_data")
    g.attrs["NX_class"] = "NXcollection"
    g.attrs[MONOTONIC_ATTR] = bool(
        len(fis) < 2 or np.all(np.diff(fis) > 0)
    )
    g.create_dataset(
        "frame_index",
        data=np.asarray(fis if fis else range(len(scan_data)), dtype=np.int64),
        maxshape=(None,),
        chunks=(64,),
    )
    for col in scan_data.columns:
        arr, create_kwargs, attrs = _scan_data_column_payload(scan_data[col].values)
        ds = g.create_dataset(
            str(col),
            data=arr,
            maxshape=(None,),
            chunks=(64,),
            **create_kwargs,
            **({} if attrs[DTYPE_ATTR] == "string" else ck),
        )
        ds.attrs.update(attrs)


def _upsert_indexed_group(
    group: h5py.Group,
    *,
    frame_indices: Sequence[int],
    values: dict[str, np.ndarray],
) -> None:
    """Update or append rows in an indexed metadata group.

    The group must already use the appendable layout created by the
    replacement writers above. Callers should fall back to a full replacement
    when this raises, which keeps upgrades from older files honest.
    """
    fis = [int(x) for x in frame_indices]
    if len(fis) != len(set(fis)):
        raise ValueError(f"frame_indices contains duplicate labels: {fis}")
    if "frame_index" not in group:
        raise ValueError(f"{group.name} has no appendable frame_index")
    existing_cols = {name for name, item in group.items()
                     if name != "frame_index" and isinstance(item, h5py.Dataset)}
    if existing_cols != set(values):
        raise ValueError(
            f"{group.name} metadata columns changed: "
            f"disk={sorted(existing_cols)}, incoming={sorted(values)}"
        )
    labels_ds = group["frame_index"]
    if labels_ds.maxshape is None or labels_ds.maxshape[0] is not None:
        raise ValueError(f"{group.name}/frame_index is not appendable")
    for col, arr in values.items():
        if len(arr) != len(fis):
            raise ValueError(f"{group.name}/{col} row count does not match frame_indices")
        if group[col].maxshape is None or group[col].maxshape[0] is not None:
            raise ValueError(f"{group.name}/{col} is not appendable")
    n = int(labels_ds.shape[0])
    last_label = int(labels_ds[n - 1]) if n else None
    monotonic = bool(group.attrs.get(MONOTONIC_ATTR, False))
    if (fis and monotonic and _strictly_increasing(fis)
            and (last_label is None or fis[0] > last_label)):
        labels_ds.resize((n + len(fis),))
        labels_ds[n:] = fis
        for col, arr in values.items():
            ds = group[col]
            ds.resize((n + len(fis),))
            ds[n:] = arr
        return
    labels = [int(x) for x in np.asarray(labels_ds[()]).ravel()]
    if len(labels) != len(set(labels)):
        raise ValueError(f"{group.name}/frame_index contains duplicate labels")
    row_of = {label: row for row, label in enumerate(labels)}
    for label_pos, label in enumerate(fis):
        row = row_of.get(label)
        if row is None:
            row = int(labels_ds.shape[0])
            labels_ds.resize((row + 1,))
            labels_ds[row] = label
            for col, arr in values.items():
                ds = group[col]
                ds.resize((row + 1,))
                ds[row] = arr[label_pos]
            row_of[label] = row
        else:
            for col, arr in values.items():
                group[col][row] = arr[label_pos]
    group.attrs[MONOTONIC_ATTR] = bool(
        _strictly_increasing(list(row_of))
    )


def _strictly_increasing(labels: Sequence[int]) -> bool:
    return all(a < b for a, b in zip(labels, labels[1:]))


def _stamp_processed_schema(entry_grp: h5py.Group) -> None:
    entry_grp.attrs[SCHEMA_NAME_ATTR] = PROCESSED_SCHEMA_NAME
    entry_grp.attrs[SCHEMA_VERSION_ATTR] = PROCESSED_SCHEMA_VERSION


def _stringify_scan_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if np.asarray(value).shape == () and isinstance(float(value), float):
            if not np.isfinite(float(value)):
                return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _scan_data_column_payload(values: Any) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    try:
        arr = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError):
        arr = np.asarray([_stringify_scan_value(v) for v in values], dtype=object)
        return arr, {"dtype": _UTF8_DTYPE}, {
            DTYPE_ATTR: "string",
            "description": "Per-frame scan metadata column",
            "missing_value": "",
            "encoding": "utf-8",
        }
    return arr, {}, {
        DTYPE_ATTR: "float32",
        "description": "Per-frame scan metadata column",
    }


def upsert_scan_metadata(
    entry_grp: h5py.Group,
    scan_data,
    frame_indices: Sequence[int],
) -> None:
    """Incrementally append or replace rows in ``/entry/scan_data``."""
    if scan_data is None or len(scan_data) == 0:
        return
    scan_data, fis = _reindex_scan_data_to_frames(scan_data, frame_indices)
    values: dict[str, np.ndarray] = {}
    attrs_by_col: dict[str, dict[str, Any]] = {}
    for col in scan_data.columns:
        arr, _create_kwargs, attrs = _scan_data_column_payload(scan_data[col].values)
        values[str(col)] = arr
        attrs_by_col[str(col)] = attrs
    if not values:
        return
    if "scan_data" not in entry_grp:
        write_scan_metadata(entry_grp, scan_data, fis)
        return
    _upsert_indexed_group(entry_grp["scan_data"], frame_indices=fis, values=values)
    for col, attrs in attrs_by_col.items():
        if col in entry_grp["scan_data"]:
            entry_grp["scan_data"][col].attrs.update(attrs)


def upsert_per_frame_geometry(
    entry_grp: h5py.Group,
    scan_data,
    frame_indices: Sequence[int],
    geometry,
    *,
    allow_create: bool = True,
) -> None:
    """Incrementally append or replace derived geometry rows."""
    if geometry is None or scan_data is None or len(scan_data) == 0:
        return
    scan_data, fis = _reindex_scan_data_to_frames(scan_data, frame_indices)
    motors = {
        m: np.asarray(scan_data[m].values, dtype=float)
        for m in geometry.all_referenced_motors()
        if m in scan_data.columns
    }
    if not motors:
        return
    derived = {
        key: np.asarray(arr, dtype=np.float32)
        for key, arr in geometry.derive_per_frame(motors).items()
    }
    if "per_frame_geometry" not in entry_grp:
        if not allow_create:
            raise ValueError("/entry/per_frame_geometry is missing; full replacement required")
        write_per_frame_geometry(entry_grp, scan_data, fis, geometry)
        return
    _upsert_indexed_group(
        entry_grp["per_frame_geometry"], frame_indices=fis, values=derived,
    )


def upsert_positioners(
    entry_grp: h5py.Group,
    scan_data,
    frame_indices: Sequence[int],
    geometry,
    *,
    allow_create: bool = True,
) -> None:
    """Incrementally append or replace NXpositioner rows."""
    if geometry is None or scan_data is None or len(scan_data) == 0:
        return
    scan_data, fis = _reindex_scan_data_to_frames(scan_data, frame_indices)
    for parent_path, motors in (
        ("sample", tuple(geometry.sample_motors)),
        ("instrument/detector", tuple(geometry.detector_motors)),
    ):
        present = [m for m in motors if m in scan_data.columns]
        if not present:
            continue
        coll_path = f"{parent_path}/positioners"
        if coll_path not in entry_grp:
            if not allow_create:
                raise ValueError(f"/entry/{coll_path} is missing; full replacement required")
            write_positioners(entry_grp, scan_data, fis, geometry)
            return
        coll = entry_grp[coll_path]
        values = {
            m: np.asarray(scan_data[m].values, dtype=np.float32)
            for m in present
        }
        flat = {name: coll[name]["value"] for name in present if name in coll}
        if set(flat) != set(values):
            raise ValueError(f"{coll.name} positioner columns changed")
        labels_ds = coll.get("frame_index")
        if labels_ds is None:
            raise ValueError(f"{coll.name} has no appendable frame_index")
        if labels_ds.maxshape is None or labels_ds.maxshape[0] is not None:
            raise ValueError(f"{coll.name}/frame_index is not appendable")
        for name, arr in values.items():
            if len(arr) != len(fis):
                raise ValueError(f"{coll.name}/{name} row count does not match frame_indices")
            ds = flat[name]
            if ds.maxshape is None or ds.maxshape[0] is not None:
                raise ValueError(f"{ds.name} is not appendable")
        n = int(labels_ds.shape[0])
        last_label = int(labels_ds[n - 1]) if n else None
        monotonic = bool(coll.attrs.get(MONOTONIC_ATTR, False))
        if (fis and monotonic and _strictly_increasing(fis)
                and (last_label is None or fis[0] > last_label)):
            labels_ds.resize((n + len(fis),))
            labels_ds[n:] = fis
            for name, arr in values.items():
                ds = flat[name]
                ds.resize((n + len(fis),))
                ds[n:] = arr
            continue
        labels = [int(x) for x in np.asarray(labels_ds[()]).ravel()]
        if len(labels) != len(set(labels)):
            raise ValueError(f"{coll.name}/frame_index contains duplicate labels")
        row_of = {label: row for row, label in enumerate(labels)}
        for pos, label in enumerate(fis):
            row = row_of.get(label)
            if row is None:
                row = int(labels_ds.shape[0])
                labels_ds.resize((row + 1,))
                labels_ds[row] = label
                for name, arr in values.items():
                    ds = flat[name]
                    ds.resize((row + 1,))
                    ds[row] = arr[pos]
                row_of[label] = row
            else:
                for name, arr in values.items():
                    flat[name][row] = arr[pos]
        coll.attrs[MONOTONIC_ATTR] = bool(
            _strictly_increasing(list(row_of))
        )


def _read_scan_data_group(e, add_frame_var, data_vars, coords) -> set:
    """Read ``/entry/scan_data`` (full metadata table) into per-frame vars.

    Returns the set of column keys loaded so the positioner reader can skip
    duplicates.  No-op (empty set) when the group is absent — old files fall
    back to the positioners.
    """
    loaded: set = set()
    if "scan_data" not in e:
        return loaded
    sd = e["scan_data"]

    # Align scan_data rows to the established frame coordinate BY LABEL.
    # scan_data carries its own frame_index; if a ``frame`` coord already
    # exists (from integrated_1d/2d) and scan_data's labels differ — even
    # at equal length, e.g. integrated [0, 2] but scan_data stored for
    # [0, 1] — attaching the columns positionally would assign metadata to
    # the wrong frames.  Build a row-permutation that maps each coord
    # label to its scan_data row (NaN where absent) so columns land on the
    # right frame.  When labels already match (or no coord yet) this is a
    # no-op identity.
    sd_labels = (np.asarray(sd["frame_index"][()]) if "frame_index" in sd
                 else None)
    if sd_labels is not None and len(sd_labels) != len(set(map(int, sd_labels))):
        raise ValueError("scan_data/frame_index contains duplicate labels")
    coord = coords.get("frame")
    perm = None          # row-permutation: coord position → scan_data row
    fully_covers = True  # does scan_data cover every coord label?
    if sd_labels is not None and coord is not None:
        if not (len(sd_labels) == len(coord)
                and np.array_equal(sd_labels, coord)):
            row_of = {int(lbl): i for i, lbl in enumerate(sd_labels)}
            perm = [row_of.get(int(c), -1) for c in coord]  # -1 → missing
            fully_covers = all(src >= 0 for src in perm)

    def _aligned(arr):
        if perm is None:
            return arr
        arr = np.asarray(arr)
        if arr.dtype.kind in {"O", "U", "S"}:
            out = np.full((len(perm),) + arr.shape[1:], "", dtype=object)
        else:
            out = np.full((len(perm),) + arr.shape[1:], np.nan, dtype=float)
        for i, src in enumerate(perm):
            if src >= 0:
                out[i] = arr[src]
        return out

    # If scan_data is a reorder of the same labels, align by label.  If it
    # only covers a SUBSET of the integrated frames (a stale/partial table,
    # the batch-misalign case), skip its columns entirely so a complete
    # same-named NXpositioner provides the data instead of a NaN-gapped
    # column — and so we never mis-assign metadata to the wrong frame.
    if not fully_covers:
        return loaded  # nothing loaded; positioners fill in

    for k, item in sd.items():
        if k == "frame_index" or not isinstance(item, h5py.Dataset):
            continue
        name = f"meta_{k}" if (k in {"q", "chi", "frame"} or k in data_vars) else k
        add_frame_var(name, _aligned(_dataset_to_native_array(item)))
        # Only suppress the positioner fallback for this key if the column
        # was actually accepted — ``add_frame_var`` skips length-mismatched
        # columns, and a valid same-named NXpositioner should still load.
        if name in data_vars:
            loaded.add(k)
    # Establish the frame coord from scan_data only when nothing else has.
    if sd_labels is not None and "frame" not in coords:
        coords["frame"] = sd_labels
    return loaded


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
    from xrd_tools.io.bluesky_nexus import bluesky_energy_kev, is_bluesky_nxwriter
    if is_bluesky_nxwriter(grp):
        # Bluesky/NXWriter: energy lives in the eiger detector config (in eV).
        ev = bluesky_energy_kev(grp)
        if np.isfinite(ev):
            return ev
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
    from xrd_tools.io.bluesky_nexus import bluesky_wavelength, is_bluesky_nxwriter
    if is_bluesky_nxwriter(grp):
        # Bluesky/NXWriter: eiger config records the wavelength directly (Å).
        wl = bluesky_wavelength(grp)
        if np.isfinite(wl):
            return wl
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
    Extract per-point motor/counter arrays from a scan's data groups.

    Harvests 1-D float datasets from BOTH ``/{entry}/data/`` and
    ``/{entry}/scan_data/`` (the SPEC-style "all motors + counters" per-point
    table some facilities write — the NeXus equivalent of reading every SPEC
    motor column), classifying each as a motor *angle* or a *counter*; PLUS the
    NXpositioner ``value`` arrays under ``/{entry}/sample/positioners/<motor>``
    (the actual scanned motor(s)), which are ALWAYS treated as angles and win on
    a name clash.  The GI incidence motor can be a non-scanned positioner, so all
    three sources are needed.  (NXpositioner stores ``value`` as a rank-1 ``[n]``
    array over the scan — NeXus manual.)

    Returns
    -------
    angles : dict[str, np.ndarray]
    counters : dict[str, np.ndarray]
    """
    angles: dict[str, np.ndarray] = {}
    counters: dict[str, np.ndarray] = {}

    # Bluesky / apstools NXWriter branch: motors are the authoritative
    # ``instrument/positioners`` group names (NOT the 80 flat ``entry/data``
    # columns), counters are the ion-chamber/photodiode channels, both read as
    # per-frame arrays from ``entry/data/<name>``.  Keeps the dtype.kind guard +
    # frame_index skip inside the harvesters.  Without this, ``meta.angles`` is
    # polluted with EPICS/stats columns and the scan motor (``hy``) is buried.
    from xrd_tools.io.bluesky_nexus import (
        bluesky_angles,
        bluesky_counters,
        is_bluesky_nxwriter,
    )
    if is_bluesky_nxwriter(grp):
        angles = bluesky_angles(grp, motor_names)
        counters = bluesky_counters(grp, counter_names)
        return angles, counters

    counter_set: frozenset[str] = (
        frozenset(counter_names) if counter_names is not None
        else _DEFAULT_COUNTER_NAMES)
    motor_set: frozenset[str] | None = (
        frozenset(motor_names) if motor_names is not None else None)

    def _harvest(group: h5py.Group) -> None:
        for name, obj in group.items():
            if not isinstance(obj, h5py.Dataset):
                continue
            if name in _NON_MOTOR_COLUMNS:   # frame_index etc. — not a motor
                continue
            # Skip non-numeric columns WITHOUT reading them: SPEC-style
            # ``scan_data`` tables routinely carry fixed-length-string (|S),
            # unicode (|U), or variable-length-string (object) timestamp/label
            # columns.  Casting those to float raises ``OSError`` from h5py
            # (not just Type/ValueError), which would crash read_nexus on a
            # user-selected foreign file.  ``dtype.kind`` is metadata — no read.
            if getattr(obj.dtype, "kind", "O") not in "fiub":
                continue
            try:
                arr = np.asarray(obj, dtype=float)
            except (TypeError, ValueError, OSError):
                continue
            if arr.ndim != 1:
                continue
            if motor_set is not None:
                if name in motor_set:
                    angles.setdefault(name, arr)
                elif name in counter_set:
                    counters.setdefault(name, arr)
            elif name in counter_set:
                counters.setdefault(name, arr)
            else:
                angles.setdefault(name, arr)

    for gname in ("data", "scan_data"):
        sub = grp.get(gname)
        if isinstance(sub, h5py.Group):
            _harvest(sub)

    # NXpositioner ``value`` arrays under sample/positioners — the actual scanned
    # motor(s).  Each positioner is a GROUP (NXpositioner) with a ``value`` field;
    # non-group children (e.g. a stray ``frame_index`` dataset) are NOT motors and
    # are skipped.  These are authoritative motors -> force into angles.
    sample = grp.get("sample")
    if isinstance(sample, h5py.Group):
        positioners = sample.get("positioners")
        if isinstance(positioners, h5py.Group):
            for name, obj in positioners.items():
                if not isinstance(obj, h5py.Group):
                    continue
                value = obj.get("value")
                if not isinstance(value, h5py.Dataset):
                    continue
                if getattr(value.dtype, "kind", "O") not in "fiub":
                    continue
                try:
                    arr = np.asarray(value, dtype=float)
                except (TypeError, ValueError, OSError):
                    continue
                if arr.ndim != 1:
                    continue
                angles[name] = arr          # authoritative scanned motor
                counters.pop(name, None)

    return angles, counters


# ===========================================================================
# v2 schema reader (xdart 0.37+)
# ---------------------------------------------------------------------------
# The v2 schema layout is defined in ``xrd_tools.io.schema``.
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


def _dataset_to_native_array(ds: h5py.Dataset) -> np.ndarray:
    if h5py.check_string_dtype(ds.dtype) is not None:
        return np.asarray(ds.asstr()[()])
    arr = np.asarray(ds[()])
    if arr.dtype.kind == "S":
        return arr.astype(str)
    return arr


def _read_positioners(grp: h5py.Group) -> dict[str, np.ndarray]:
    """Read NXpositioner children of an NXcollection into a dict (v2).

    Handles two ``value`` shapes: xdart's flat ``<motor>/value`` dataset, and
    Bluesky/NXWriter's nested ``<motor>/value`` NXdata group whose own ``value``
    child holds the per-frame array."""
    out: dict[str, np.ndarray] = {}
    for k, item in grp.items():
        if k == "frame_index":
            continue
        if isinstance(item, h5py.Group):
            value = item.get("value")
            if isinstance(value, h5py.Dataset):
                out[k] = np.asarray(value[()])
            elif isinstance(value, h5py.Group):
                # Bluesky positioner: value is an NXdata group -> its 'value' array.
                inner = value.get("value")
                if isinstance(inner, h5py.Dataset):
                    out[k] = np.asarray(inner[()])
        elif isinstance(item, h5py.Dataset):
            out[k] = np.asarray(item[()])
    return out


def _read_scan_v2(path: Path, entry: str, groups: tuple[str, ...],
                  include_thumbnails: bool):
    """v2-schema reader.  Body of public ``read_scan``."""
    import xarray as xr

    from xrd_tools.core.provenance import read_provenance

    data_vars: dict[str, tuple] = {}
    coords: dict[str, np.ndarray] = {}
    attrs_per_coord: dict[str, dict] = {}
    attrs_per_var: dict[str, dict] = {}

    with h5py.File(path, "r") as f:
        if entry not in f:
            raise KeyError(f"No {entry!r} group in {path}")
        e = f[entry]
        warn_if_newer_schema(e, str(path))

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
            fidx_2d = (
                np.asarray(g2["frame_index"][()])
                if "frame_index" in g2 else None
            )
            # Decide which frame dimension the 2D stack rides on.  Normally
            # 1D and 2D were reduced over the same frames and share labels →
            # shared ``frame``.  If they differ (partial / separately
            # re-reduced outputs), forcing 2D onto the 1D ``frame`` coord
            # would silently mislabel rows (or, if lengths differ, make the
            # Dataset construction raise).  In that case give the 2D stack
            # its own ``frame_2d`` dimension so both load correctly and the
            # mismatch is explicit rather than silent.
            frame_dim = "frame"
            if "frame" in coords and fidx_2d is not None:
                f1 = coords["frame"]
                if f1.shape != fidx_2d.shape or not np.array_equal(f1, fidx_2d):
                    frame_dim = "frame_2d"
                    coords["frame_2d"] = fidx_2d
                    logger.warning(
                        "integrated_1d and integrated_2d carry different "
                        "frame_index in %s; loading 2D on a separate "
                        "'frame_2d' dimension to avoid mislabeling.", path,
                    )
            elif fidx_2d is not None:
                coords["frame"] = fidx_2d

            # 1D and 2D radial axes are the same physical quantity (q
            # magnitude) but sampled at independent resolutions
            # (npt=2000 for 1D, npt_rad=500 for 2D is typical).  xarray
            # requires distinct dim names for differently-sized axes,
            # so 2D's radial axis lives under ``q_2d``.
            data_vars["intensity_2d"] = (
                (frame_dim, "chi", "q_2d"),
                np.asarray(g2["intensity"][()]),
            )
            attrs_per_var["intensity_2d"] = {
                "two_d_kind": _v2_decode_str(
                    g2.attrs.get(
                        "two_d_kind",
                        two_d_kind_from_units(
                            _v2_decode_str(g2["q"].attrs.get("units", ""))
                            if "q" in g2 else "",
                            _v2_decode_str(g2["chi"].attrs.get("units", ""))
                            if "chi" in g2 else "",
                        ).value,
                    )
                )
            }
            if "sigma" in g2:
                data_vars["sigma_2d"] = (
                    (frame_dim, "chi", "q_2d"),
                    np.asarray(g2["sigma"][()]),
                )
            coords["q_2d"] = np.asarray(g2["q"][()])
            u_q2 = g2["q"].attrs.get("units", None)
            if u_q2 is not None:
                attrs_per_coord["q_2d"] = {"units": _v2_decode_str(u_q2)}
            coords["chi"] = np.asarray(g2["chi"][()])
            u = g2["chi"].attrs.get("units", None)
            if u is not None:
                attrs_per_coord["chi"] = {"units": _v2_decode_str(u)}

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

        # Full per-frame metadata table (preferred source); positioners
        # below only fill in geometry motors it didn't already carry.
        _sd_loaded = _read_scan_data_group(e, _add_frame_var, data_vars, coords)

        for category, path_in in [
            ("sample", "sample/positioners"),
            ("detector", "instrument/detector/positioners"),
            ("instrument", "instrument/positioners"),
        ]:
            if path_in in e:
                pos = _read_positioners(e[path_in])
                for k, arr in pos.items():
                    if k in _sd_loaded:
                        continue  # already loaded from /entry/scan_data
                    reserved = {"q", "chi", "frame"}
                    if k in reserved or k in data_vars:
                        var_name = f"{category}_{k}"
                    else:
                        var_name = k
                    _add_frame_var(var_name, arr)

        if include_thumbnails and "frames" in e:
            thumbs: list[np.ndarray] = []
            thumb_labels: list[int] = []
            for name in sorted(e["frames"].keys()):
                fg = e[f"frames/{name}"]
                if "thumbnail" in fg:
                    thumbs.append(np.asarray(fg["thumbnail"][()]))
                    try:
                        thumb_labels.append(int(name.removeprefix("frame_")))
                    except ValueError:
                        logger.debug("Skipping thumbnail with unparseable group label %r", name)
                        thumbs.pop()
            if thumbs:
                data_vars["thumbnail"] = (
                    ("thumbnail_frame", "thumb_y", "thumb_x"),
                    np.stack(thumbs, axis=0),
                )
                coords["thumbnail_frame"] = np.asarray(thumb_labels, dtype=np.int64)

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
    for var, attrs in attrs_per_var.items():
        if var in ds.data_vars:
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
    from xrd_tools.core.provenance import read_provenance

    path = Path(path)
    data_vars: dict[str, tuple] = {}
    coords: dict[str, np.ndarray] = {}
    attrs_per_coord: dict[str, dict] = {}

    with h5py.File(path, "r") as f:
        if entry not in f:
            raise KeyError(f"No {entry!r} group in {path}")
        e = f[entry]
        warn_if_newer_schema(e, str(path))

        # frame_index — try integrated_1d first, then integrated_2d,
        # then per_frame_geometry.  Cheap; no intensity is loaded.
        for grp_name in ("integrated_1d", "integrated_2d",
                         "per_frame_geometry"):
            if grp_name in e and "frame_index" in e[grp_name]:
                coords["frame"] = np.asarray(
                    e[grp_name]["frame_index"][()]
                )
                break

        # Consistency with read_scan: if 1D and 2D were reduced over
        # different frame labels, surface the 2D labels on a separate
        # ``frame_2d`` coord too, so the lightweight metadata path doesn't
        # silently report only the 1D labels (the get_metadata "frames"
        # would otherwise disagree with read_scan / get_2d).
        if (
            "integrated_1d" in e and "frame_index" in e["integrated_1d"]
            and "integrated_2d" in e and "frame_index" in e["integrated_2d"]
        ):
            f1 = np.asarray(e["integrated_1d"]["frame_index"][()])
            f2 = np.asarray(e["integrated_2d"]["frame_index"][()])
            if f1.shape != f2.shape or not np.array_equal(f1, f2):
                coords["frame_2d"] = f2

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

        # Full per-frame metadata table (preferred source); positioners
        # below only fill in geometry motors it didn't already carry.
        _sd_loaded = _read_scan_data_group(e, _add_frame_var, data_vars, coords)

        # Positioners.
        for category, path_in in [
            ("sample", "sample/positioners"),
            ("detector", "instrument/detector/positioners"),
            ("instrument", "instrument/positioners"),
        ]:
            if path_in in e:
                pos = _read_positioners(e[path_in])
                for k, arr in pos.items():
                    if k in _sd_loaded:
                        continue  # already loaded from /entry/scan_data
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
        If ``True``, load available thumbnails as a ``thumbnail`` data
        variable indexed by the independent ``thumbnail_frame`` coordinate.

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
    attrs: dict[str, object] = {}

    def _read_provenance(g):
        """The provenance_json blob → a parsed dict (or the raw string if it
        isn't JSON; absent → None)."""
        if "provenance_json" not in g:
            return None
        raw = g["provenance_json"][()]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return str(raw)

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
            prov = _read_provenance(g)
            if prov is not None:
                attrs["stitched_1d_provenance"] = prov
        if has_2d:
            g = e["stitched_2d"]
            coords.setdefault("q", np.asarray(g["q"][()]))
            coords["chi"] = np.asarray(g["chi"][()])
            data_vars["stitched_2d"] = (
                ("q", "chi"), np.asarray(g["intensity"][()])
            )
            prov = _read_provenance(g)
            if prov is not None:
                attrs["stitched_2d_provenance"] = prov

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


__all__ = [
    "PROCESSED_SCHEMA_NAME",
    "PROCESSED_SCHEMA_VERSION",
    "validate_group_against_schema",
    # v1 (legacy beamline files)
    "find_nexus_image_dataset",
    "NexusImageStack",
    "open_nexus_image_stack",
    "list_entries",
    "read_nexus",
    "open_nexus_writer",
    "write_nexus",
    "write_nexus_frame",
    "validate_integrated_stack_write",
    "frame_record_write_parts",
    "write_integrated_stack",
    "write_frame_records",
    "write_stitched",
    "write_rsm",
    "read_rsm",
    "upsert_scan_metadata",
    "upsert_positioners",
    "upsert_per_frame_geometry",
    "write_positioners",
    "write_per_frame_geometry",
    "write_scan_metadata",
    # v2 (xdart 0.37+)
    "read_scan",
    "read_scan_metadata",
    "read_stitched",
]
