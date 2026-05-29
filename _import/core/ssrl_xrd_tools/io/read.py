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

from collections import namedtuple
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

__all__ = [
    "Integrated1D",
    "Integrated2D",
    "get_frames",
    "get_1d",
    "get_2d",
    "get_thumbnail",
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


def _frame_index(grp: h5py.Group) -> np.ndarray:
    """Return the frame-label array, trying each group that may carry it."""
    for name in ("integrated_1d", "integrated_2d", "per_frame_geometry"):
        if name in grp and "frame_index" in grp[name]:
            return np.asarray(grp[name]["frame_index"][()])
    raise KeyError("No frame_index found (integrated_1d/2d/per_frame_geometry)")


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

def get_frames(scan_file: str | Path, *, entry: str = "entry") -> np.ndarray:
    """Return the array of frame labels present in ``scan_file``."""
    with h5py.File(Path(scan_file), "r") as f:
        return _frame_index(_entry(f, entry))


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
        positions, frames, single = _resolve_positions(_frame_index(e), frame)

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
        positions, frames, single = _resolve_positions(_frame_index(e), frame)

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


def get_metadata(scan_file: str | Path, *, entry: str = "entry") -> dict:
    """Return a flat dict of scan-level metadata (no heavy intensity arrays).

    Keys: ``frames``, ``n_frames``, ``has_1d``, ``has_2d``, ``q``, ``q_2d``,
    ``chi`` (axes, when present), ``sample_name``, ``energy_keV``,
    ``wavelength_A``, ``ub_matrix`` (or ``None``), ``positioners`` (dict of
    per-frame motor arrays), and ``reduction`` (provenance dict).
    """
    # Reuse the canonical metadata-only reader for axes / positioners /
    # provenance, then add the instrument/sample scalars it doesn't carry.
    from ssrl_xrd_tools.io.nexus import (
        read_scan_metadata,
        _read_energy,
        _read_wavelength,
        _read_ub_matrix,
        _read_sample_name,
    )

    ds = read_scan_metadata(scan_file, entry=entry)
    reserved = {"rot1", "rot2", "rot3", "incident_angle"}
    positioners = {
        name: np.asarray(ds[name].values)
        for name in ds.data_vars
        if name not in reserved and ds[name].dims == ("frame",)
    }

    with h5py.File(Path(scan_file), "r") as f:
        e = _entry(f, entry)
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
    for axis in ("q", "q_2d", "chi"):
        if axis in ds.coords:
            meta[axis] = np.asarray(ds[axis].values)
    meta["positioners"] = positioners
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
    open file handle and caches nothing heavy — every call re-reads from
    disk (cheap, single-frame slices).
    """

    def __init__(self, scan_file: str | Path, *, entry: str = "entry"):
        self.path = Path(scan_file)
        self.entry = entry

    @property
    def frames(self) -> np.ndarray:
        return get_frames(self.path, entry=self.entry)

    @property
    def metadata(self) -> dict:
        return get_metadata(self.path, entry=self.entry)

    def get_1d(self, frame=None) -> Integrated1D:
        return get_1d(self.path, frame, entry=self.entry)

    def get_2d(self, frame=None) -> Integrated2D:
        return get_2d(self.path, frame, entry=self.entry)

    def get_thumbnail(self, frame: int) -> np.ndarray:
        return get_thumbnail(self.path, frame, entry=self.entry)

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
