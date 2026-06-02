"""Simple, notebook-friendly readers for processed xdart v2 NeXus scan files.

A processed ``.nxs`` file is a **scan**: a stack of integrated **frames**.
These helpers pull 1D / 2D integrated patterns, thumbnails, and scan
metadata out of a scan file with a single function call and no xarray
knowledge required — the intent is "open a file, get arrays I can plot."

For the full :class:`xarray.Dataset` (every frame, every motor column,
provenance) use :func:`ssrl_xrd_tools.io.read_scan` /
:func:`read_scan_metadata`.  These ``get_*`` helpers sit on top of the
same v2 layout but slice **one frame at a time straight from h5py**, so
``get_2d(scan, frame=k)`` does not materialise the full
``(n_frames, chi, q)`` tensor — important for 10k-frame Eiger scans.

Frame addressing
----------------
``frame`` arguments refer to the frame **label** (the value stored in the
file's ``frame_index`` — 1-based for SPEC scans, 0-based for many
detectors, possibly gapped after a partial re-reduction), *not* the row
position.  Pass ``frame=None`` (the default where allowed) to get every
frame stacked.  Use :func:`get_frames` to see which labels exist.

Examples
--------
>>> from ssrl_xrd_tools.io import get_1d, get_2d, get_frames, open_scan
>>> get_frames("scan_42.nxs")
array([1, 2, 3, 4, 5])
>>> r = get_1d("scan_42.nxs", frame=3)
>>> r.q.shape, r.intensity.shape
((2000,), (2000,))
>>> # object-style sugar:
>>> scan = open_scan("scan_42.nxs")
>>> len(scan)
5
>>> all_1d = scan.get_1d()          # (n_frames, q)
"""

from __future__ import annotations

import logging
from collections import namedtuple
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "Integrated1D",
    "Integrated2D",
    "get_frames",
    "get_1d",
    "get_2d",
    "get_thumbnail",
    "get_raw_frame",
    "get_metadata",
    "open_scan",
    "Scan",
]


# ``frames`` is the resolved frame label(s): an int for a single-frame
# read, or an np.ndarray of labels when multiple frames were returned.
Integrated1D = namedtuple("Integrated1D", ["q", "intensity", "sigma", "q_unit", "frames"])
Integrated2D = namedtuple(
    "Integrated2D", ["q", "chi", "intensity", "q_unit", "chi_unit", "frames"]
)


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _entry(f: h5py.File, entry: str) -> h5py.Group:
    if entry not in f:
        raise KeyError(f"No {entry!r} group in {f.filename}")
    return f[entry]


def _decode(v):
    return v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else v


def _frame_index(grp: h5py.Group, prefer: str | None = None) -> np.ndarray:
    """Return the frame-label array for an entry.

    ``prefer`` names the group whose ``frame_index`` to use first — pass
    ``"integrated_2d"`` from :func:`get_2d` so frames resolve against the
    2D output's own labels.  This matters when 1D and 2D were reduced over
    different frame subsets (or re-reduced with different labels): a
    ``frame=`` request must index the group it's reading from, not the
    other one.  Falls back to the usual order if the preferred group has
    no ``frame_index``.
    """
    order = ("integrated_1d", "integrated_2d", "per_frame_geometry")
    if prefer is not None:
        order = (prefer,) + tuple(n for n in order if n != prefer)
    for name in order:
        if name in grp and "frame_index" in grp[name]:
            return np.asarray(grp[name]["frame_index"][()])
    raise KeyError("No frame_index found (integrated_1d/2d/per_frame_geometry)")


def _all_frame_index(grp: h5py.Group) -> np.ndarray:
    """Return the union of labels with reduced data or raw-source groups."""
    labels: set[int] = set()
    if "frames" in grp:
        for name in grp["frames"]:
            if not name.startswith("frame_"):
                continue
            try:
                labels.add(int(name.removeprefix("frame_")))
            except ValueError:
                continue
    for name in ("integrated_1d", "integrated_2d"):
        if name in grp and "frame_index" in grp[name]:
            labels.update(int(x) for x in np.asarray(grp[name]["frame_index"][()]).ravel())
    if labels:
        return np.asarray(sorted(labels), dtype=np.int64)
    return _frame_index(grp)


def _scan_data_for_frames(
    scan_file: str | Path,
    frames: Sequence[int],
    *,
    entry: str = "entry",
) -> dict[str, np.ndarray]:
    """Read /entry/scan_data aligned to explicit frame labels."""
    frames = [int(frame) for frame in frames]
    out: dict[str, np.ndarray] = {}
    if not frames:
        return out
    with h5py.File(Path(scan_file), "r") as f:
        e = _entry(f, entry)
        if "scan_data" not in e or "frame_index" not in e["scan_data"]:
            return out
        sd = e["scan_data"]
        labels = [int(x) for x in np.asarray(sd["frame_index"][()]).ravel()]
        if len(labels) != len(set(labels)):
            raise ValueError("scan_data/frame_index contains duplicate labels")
        row_of = {label: row for row, label in enumerate(labels)}
        rows = [row_of.get(frame, -1) for frame in frames]
        for key, item in sd.items():
            if key == "frame_index" or not isinstance(item, h5py.Dataset):
                continue
            try:
                arr = np.asarray(item[()], dtype=float)
            except (TypeError, ValueError):
                continue
            aligned = np.full((len(frames),) + arr.shape[1:], np.nan, dtype=float)
            for dst, src in enumerate(rows):
                if src >= 0:
                    aligned[dst] = arr[src]
            out[str(key)] = aligned
    return out


def _resolve_positions(frame_index: np.ndarray, frame):
    """Map requested frame label(s) to row position(s) in ``frame_index``.

    Returns ``(positions, frames, single)`` where ``positions`` is an
    index array into the stacked datasets, ``frames`` is the matching
    label(s), and ``single`` is True iff a scalar ``frame`` was given.
    """
    if frame is None:
        return np.arange(len(frame_index)), frame_index, False

    single = np.isscalar(frame)
    wanted = [frame] if single else list(frame)
    label_to_pos = {int(lbl): pos for pos, lbl in enumerate(frame_index)}
    positions = []
    for lbl in wanted:
        if int(lbl) not in label_to_pos:
            raise KeyError(
                f"Frame {lbl} not in this scan. Available frames: "
                f"{frame_index.tolist()}"
            )
        positions.append(label_to_pos[int(lbl)])
    positions = np.asarray(positions, dtype=int)
    frames = frame_index[positions]
    if single:
        return positions, int(frames[0]), True
    return positions, frames, False


# ---------------------------------------------------------------------------
# public readers
# ---------------------------------------------------------------------------

def get_frames(
    scan_file: str | Path,
    *,
    entry: str = "entry",
    union: bool = False,
) -> np.ndarray:
    """Return the array of frame labels present in ``scan_file``."""
    with h5py.File(Path(scan_file), "r") as f:
        e = _entry(f, entry)
        return _all_frame_index(e) if union else _frame_index(e)


def get_1d(
    scan_file: str | Path,
    frame=None,
    *,
    entry: str = "entry",
) -> Integrated1D:
    """Read 1D integrated intensity from a processed scan file.

    Parameters
    ----------
    scan_file
        Path to the processed ``.nxs`` file.
    frame
        A single frame label, an iterable of labels, or ``None`` for all
        frames.  See module docstring on frame addressing.

    Returns
    -------
    Integrated1D
        Named tuple ``(q, intensity, sigma, q_unit, frames)``.  ``intensity``
        is ``(n_q,)`` for a single frame, else ``(n_frames, n_q)``.
        ``sigma`` is ``None`` when the file stored no error estimate.
    """
    with h5py.File(Path(scan_file), "r") as f:
        e = _entry(f, entry)
        if "integrated_1d" not in e:
            raise KeyError(f"{scan_file} has no integrated_1d group")
        g = e["integrated_1d"]
        positions, frames, single = _resolve_positions(
            _frame_index(e, prefer="integrated_1d"), frame)

        q = np.asarray(g["q"][()])
        q_unit = _decode(g["q"].attrs.get("units")) if "units" in g["q"].attrs else None
        intensity = _slice_stack(g["intensity"], positions, single)
        sigma = (
            _slice_stack(g["sigma"], positions, single) if "sigma" in g else None
        )
    return Integrated1D(q=q, intensity=intensity, sigma=sigma, q_unit=q_unit, frames=frames)


def get_2d(
    scan_file: str | Path,
    frame=None,
    *,
    entry: str = "entry",
) -> Integrated2D:
    """Read 2D (cake / q-chi) integrated intensity from a processed scan file.

    Returns
    -------
    Integrated2D
        Named tuple ``(q, chi, intensity, q_unit, chi_unit, frames)``.
        ``intensity`` is ``(n_chi, n_q)`` for a single frame, else
        ``(n_frames, n_chi, n_q)``.
    """
    with h5py.File(Path(scan_file), "r") as f:
        e = _entry(f, entry)
        if "integrated_2d" not in e:
            raise KeyError(f"{scan_file} has no integrated_2d group")
        g = e["integrated_2d"]
        positions, frames, single = _resolve_positions(
            _frame_index(e, prefer="integrated_2d"), frame)

        q = np.asarray(g["q"][()])
        chi = np.asarray(g["chi"][()])
        q_unit = _decode(g["q"].attrs.get("units")) if "units" in g["q"].attrs else None
        chi_unit = (
            _decode(g["chi"].attrs.get("units")) if "units" in g["chi"].attrs else None
        )
        intensity = _slice_stack(g["intensity"], positions, single)
    return Integrated2D(
        q=q, chi=chi, intensity=intensity, q_unit=q_unit, chi_unit=chi_unit, frames=frames
    )


def get_thumbnail(
    scan_file: str | Path,
    frame: int,
    *,
    entry: str = "entry",
) -> np.ndarray:
    """Return the stored thumbnail image for a single ``frame`` label.

    Raises ``KeyError`` if the file stored no thumbnail for that frame.
    """
    with h5py.File(Path(scan_file), "r") as f:
        e = _entry(f, entry)
        if "frames" not in e:
            raise KeyError(f"{scan_file} has no per-frame thumbnails")
        key = f"frames/frame_{int(frame):04d}/thumbnail"
        if key not in e:
            raise KeyError(f"No thumbnail for frame {frame} in {scan_file}")
        return np.asarray(e[key][()])


def _dequantize_thumbnail(ds: h5py.Dataset) -> np.ndarray:
    """Invert the uint8/uint16 thumbnail quantization back to intensities.

    Thumbnails are stored as ``clip((x-vmin)/(vmax-vmin),0,1)*scale`` (scale
    255 for uint8, 65535 for uint16) with ``vmin``/``vmax``/``dtype`` attrs.
    """
    arr = np.asarray(ds[()], dtype=float)
    vmin = float(ds.attrs.get("vmin", 0.0))
    vmax = float(ds.attrs.get("vmax", 1.0))
    dt = _decode(ds.attrs.get("dtype", "uint8"))
    scale = 65535.0 if str(dt) == "uint16" else 255.0
    return vmin + (arr / scale) * (vmax - vmin)


def get_raw_frame(
    scan_file: str | Path,
    frame: int,
    *,
    entry: str = "entry",
    allow_thumbnail: bool = True,
) -> np.ndarray:
    """Return the raw detector image for one ``frame`` of a processed scan.

    A processed v2 ``.nxs`` stores integrated patterns, not raw detector
    images — but each frame carries a *source pointer*
    (``frames/frame_NNNN/source/{path,frame_index}``) back to the original
    detector master plus a quantized *thumbnail*.  This resolves the source
    pointer (``path`` is relative to the scan file's directory) and reads the
    full-resolution raw image from the master via
    :func:`ssrl_xrd_tools.io.image.read_image`.  If the master can't be
    located or read, it falls back to the stored thumbnail (dequantized to
    its original intensity range) unless ``allow_thumbnail=False``.

    ``frame`` is the frame **label** (the ``frame_index`` value), matching
    the other ``get_*`` readers.  Raises ``KeyError`` when neither a usable
    source pointer nor a thumbnail is present.
    """
    from ssrl_xrd_tools.io.image import read_image

    scan_file = Path(scan_file)
    master: Path | None = None
    src_frame_idx = 0
    thumb: np.ndarray | None = None

    with h5py.File(scan_file, "r") as f:
        e = _entry(f, entry)
        fg = e.get(f"frames/frame_{int(frame):04d}")
        if fg is None:
            raise KeyError(f"No frame group for frame {frame} in {scan_file}")
        src = fg.get("source")
        if src is not None and "path" in src:
            rel = _decode(src["path"][()])
            if "frame_index" in src:
                src_frame_idx = int(np.asarray(src["frame_index"][()]).ravel()[0])
            if rel:
                rel_path = Path(str(rel)).expanduser()
                candidates = []
                if rel_path.is_absolute():
                    candidates.append(rel_path)
                candidates.extend([
                    scan_file.parent / rel_path,
                    scan_file.parent / rel_path.name,
                    rel_path,
                ])
                seen: set[Path] = set()
                for cand in candidates:
                    try:
                        resolved = cand.resolve()
                    except OSError:
                        resolved = cand
                    if resolved in seen:
                        continue
                    seen.add(resolved)
                    if resolved.exists():
                        master = resolved
                        break
        thumb_ds = fg.get("thumbnail")
        if thumb_ds is not None:
            thumb = _dequantize_thumbnail(thumb_ds)

    if master is not None:
        try:
            return np.asarray(read_image(master, frame=src_frame_idx), dtype=float)
        except Exception:
            logger.debug("get_raw_frame: failed reading master %s frame %d; "
                         "%s thumbnail", master, src_frame_idx,
                         "falling back to" if allow_thumbnail else "not falling back to",
                         exc_info=True)
    if allow_thumbnail and thumb is not None:
        return thumb
    raise KeyError(
        f"frame {frame}: source master file not found/readable"
        + (
            f" and no thumbnail stored in {scan_file}"
            if allow_thumbnail else
            "; thumbnail fallback disabled for strict raw loading"
        )
    )


def get_metadata(scan_file: str | Path, *, entry: str = "entry") -> dict:
    """Return a flat dict of scan-level metadata (no heavy intensity arrays).

    Keys: ``frames``, ``n_frames``, ``has_1d``, ``has_2d``, ``q``, ``q_2d``,
    ``chi`` (axes, when present), ``sample_name``, ``energy_keV``,
    ``wavelength_A``, ``ub_matrix`` (or ``None``), ``positioners`` (dict of
    per-frame **geometry-motor** arrays only), ``scan_data`` (dict of *all*
    per-frame columns — motors AND counters), and ``reduction`` (provenance).

    ``positioners`` is intentionally narrow (just the diffractometer motors
    from the ``sample``/``detector`` positioner groups) so geometry/
    normalization APIs that consume it stay unambiguous; the complete
    per-frame metadata table (i0, monitor, temperature, …) is in
    ``scan_data``.
    """
    # Reuse the canonical metadata-only reader for axes / positioners /
    # provenance, then add the instrument/sample scalars it doesn't carry.
    from ssrl_xrd_tools.io.nexus import (
        read_scan_metadata,
        _read_positioners,
        _read_energy,
        _read_wavelength,
        _read_ub_matrix,
        _read_sample_name,
    )

    ds = read_scan_metadata(scan_file, entry=entry)
    # Full per-frame table: every (frame,) data_var except the derived
    # geometry rotations (those are computed, not recorded metadata).
    reserved = {"rot1", "rot2", "rot3", "incident_angle"}
    scan_data = {
        name: np.asarray(ds[name].values)
        for name in ds.data_vars
        if name not in reserved and ds[name].dims == ("frame",)
    }
    # Narrow positioners: ONLY the motors stored in the sample/detector
    # NXpositioner groups (the diffractometer geometry motors), not the
    # counters folded into the full scan_data table.

    with h5py.File(Path(scan_file), "r") as f:
        e = _entry(f, entry)
        positioners: dict = {}
        for path_in in ("sample/positioners",
                        "instrument/detector/positioners"):
            if path_in in e:
                positioners.update(_read_positioners(e[path_in]))
        energy = _read_energy(e)
        meta = {
            "frames": np.asarray(ds["frame"].values) if "frame" in ds.coords else np.array([]),
            "has_1d": "integrated_1d" in e,
            "has_2d": "integrated_2d" in e,
            "sample_name": _read_sample_name(e),
            "energy_keV": energy,
            "wavelength_A": _read_wavelength(e, energy),
            "ub_matrix": _read_ub_matrix(e),
        }
    meta["n_frames"] = int(meta["frames"].size)
    # When 1D and 2D were reduced over different frame labels, read_scan_metadata
    # exposes the 2D labels separately; surface them so this matches read_scan.
    if "frame_2d" in ds.coords:
        meta["frames_2d"] = np.asarray(ds["frame_2d"].values)
    for axis in ("q", "q_2d", "chi"):
        if axis in ds.coords:
            meta[axis] = np.asarray(ds[axis].values)
    meta["positioners"] = positioners
    meta["scan_data"] = scan_data
    meta["reduction"] = ds.attrs.get("reduction", {})
    return meta


def _slice_stack(dset: h5py.Dataset, positions: np.ndarray, single: bool) -> np.ndarray:
    """Read ``positions`` rows from a stacked dataset, dropping the frame
    axis when a single frame was requested."""
    # h5py fancy indexing needs strictly increasing indices; frame_index is
    # already sorted in v2 files, but sort defensively for arbitrary input.
    order = np.argsort(positions, kind="stable")
    out = np.asarray(dset[positions[order]])
    # restore caller-requested order
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    out = out[inv]
    return out[0] if single else out


# ---------------------------------------------------------------------------
# object-style sugar
# ---------------------------------------------------------------------------

class Scan:
    """Lightweight handle to a processed scan file.

    Thin sugar over the module-level ``get_*`` functions so notebook code
    can read ``scan.get_1d(3)`` instead of repeating the path.  Holds no
    open file handle and caches only lightweight metadata; image and
    integration slices are always read on demand.
    """

    def __init__(self, scan_file: str | Path, *, entry: str = "entry"):
        self.path = Path(scan_file)
        self.entry = entry
        self._metadata_cache: dict | None = None
        self._scan_data_cache: dict[str, np.ndarray] | None = None

    @property
    def frames(self) -> np.ndarray:
        return get_frames(self.path, entry=self.entry, union=True)

    @property
    def frame_indices(self) -> list[int]:
        return [int(frame) for frame in self.frames]

    @property
    def metadata(self) -> dict:
        if self._metadata_cache is None:
            self._metadata_cache = get_metadata(self.path, entry=self.entry)
        return self._metadata_cache

    @property
    def scan_data(self) -> dict[str, np.ndarray]:
        if self._scan_data_cache is None:
            self._scan_data_cache = _scan_data_for_frames(
                self.path, self.frame_indices, entry=self.entry,
            )
        return self._scan_data_cache

    @property
    def energy(self) -> float | None:
        return self.energy_keV

    @property
    def energy_keV(self) -> float | None:
        return self.metadata.get("energy_keV")

    @property
    def energy_eV(self) -> float | None:
        energy = self.energy_keV
        return None if energy is None else float(energy) * 1000.0

    def refresh_metadata(self) -> dict:
        """Discard the lightweight cache and read the latest file metadata."""
        self._metadata_cache = None
        self._scan_data_cache = None
        return self.metadata

    def get_1d(self, frame=None) -> Integrated1D:
        return get_1d(self.path, frame, entry=self.entry)

    def get_2d(self, frame=None) -> Integrated2D:
        return get_2d(self.path, frame, entry=self.entry)

    def get_thumbnail(self, frame: int) -> np.ndarray:
        return get_thumbnail(self.path, frame, entry=self.entry)

    def load_frame(self, index: int) -> np.ndarray:
        """Load one raw detector frame through its stored source pointer."""
        return get_raw_frame(
            self.path, int(index), entry=self.entry, allow_thumbnail=False,
        )

    def iter_chunks(self, chunk_size: int):
        """Yield bounded raw-image chunks for RSM and other streaming consumers."""
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0; got {chunk_size}")
        indices = self.frame_indices
        for start in range(0, len(indices), chunk_size):
            chunk_indices = indices[start:start + chunk_size]
            yield np.stack([self.load_frame(idx) for idx in chunk_indices]), chunk_indices

    def __len__(self) -> int:
        try:
            return int(self.frames.size)
        except KeyError:
            return 0

    def __repr__(self) -> str:
        return f"Scan({self.path.name!r}, n_frames={len(self)})"


def open_scan(scan_file: str | Path, *, entry: str = "entry") -> Scan:
    """Return a :class:`Scan` handle for ``scan_file`` (notebook sugar)."""
    return Scan(scan_file, entry=entry)
