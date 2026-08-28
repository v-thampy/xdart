"""Simple, notebook-friendly readers for processed xdart v2 NeXus scan files.

A processed ``.nexus`` file is a **scan**: a stack of integrated **frames**.
These helpers pull 1D / 2D integrated patterns, thumbnails, and scan
metadata out of a scan file with a single function call and no xarray
knowledge required — the intent is "open a file, get arrays I can plot."

For the full :class:`xarray.Dataset` (every frame, every motor column,
provenance) use :func:`xrd_tools.io.read_scan` /
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
>>> from xrd_tools.io import get_1d, get_2d, get_frames, open_scan
>>> get_frames("scan_42.nexus")
array([1, 2, 3, 4, 5])
>>> r = get_1d("scan_42.nexus", frame=3)
>>> r.q.shape, r.intensity.shape
((2000,), (2000,))
>>> # object-style sugar:
>>> scan = open_scan("scan_42.nexus")
>>> len(scan)
5
>>> all_1d = scan.get_1d()          # (n_frames, q)
"""

from __future__ import annotations

import logging
import hashlib
import os
import struct
from collections import namedtuple
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

import h5py
import numpy as np

from xrd_tools.core.scan import ScanFrame

# 2c: the convenience readers consume the declared layout — group and
# dataset names come from the schema, so writer/reader drift is
# impossible by construction.
from xrd_tools.io.schema import SCHEMA, resolve_integrated_group


def _group_or_shadow(grp, name: str):
    """Resolve ``name`` under ``grp``, adopting an orphan ``__reint`` shadow for
    the integrated groups when a crash left the file mid-swap (read-only)."""
    if name in ("integrated_1d", "integrated_2d"):
        g, _ = resolve_integrated_group(grp, name)
        return g
    return grp.get(name)

logger = logging.getLogger(__name__)

__all__ = [
    "Integrated1D",
    "Integrated2D",
    "get_frames",
    "get_average_finite_counts",
    "get_1d",
    "get_2d",
    "get_thumbnail",
    "get_raw_frame",
    "get_metadata",
    "read_scan_data",
    "open_scan",
    "ProcessedScan",
    "RawSourceObservation",
    "observe_resolved_raw_source",
    "resolved_raw_source",
    "resolve_project_source_path",
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
    from xrd_tools.io.processed_scan_id import require_current_processed_groups
    return require_current_processed_groups(f, entry).entry


def _decode(v):
    return v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else v


def _dataset_values(ds: h5py.Dataset) -> np.ndarray:
    if h5py.check_string_dtype(ds.dtype) is not None:
        return np.asarray(ds.asstr()[()])
    arr = np.asarray(ds[()])
    if arr.dtype.kind == "S":
        return arr.astype(str)
    return arr


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
        g = _group_or_shadow(grp, name)
        if g is not None and "frame_index" in g:
            return np.asarray(g["frame_index"][()])
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
        g = _group_or_shadow(grp, name)
        if g is not None and "frame_index" in g:
            labels.update(int(x) for x in np.asarray(g["frame_index"][()]).ravel())
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
        from xrd_tools.io.processed_scan_id import require_current_processed_groups
        processed = require_current_processed_groups(
            f,
            entry,
            container=Path(scan_file),
        )
        e = processed.entry
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
            arr = _dataset_values(item)
            if arr.dtype.kind in {"O", "U", "S"}:
                aligned = np.full((len(frames),) + arr.shape[1:], "", dtype=object)
            else:
                aligned = np.full((len(frames),) + arr.shape[1:], np.nan, dtype=float)
            for dst, src in enumerate(rows):
                if src >= 0:
                    aligned[dst] = arr[src]
            out[str(key)] = aligned
    return out


def read_scan_data(
    scan_file: str | Path,
    frames: Sequence[int] | None = None,
    *,
    entry: str = "entry",
) -> dict[str, np.ndarray]:
    """Read ``/entry/scan_data`` as ``{column: array}`` — the per-frame metadata
    columns (motors + counters) a processed scan persisted.

    With ``frames`` given, each column is aligned to those frame labels (missing
    labels → NaN / "" rows); with ``frames=None``, every column is returned in
    natural ``scan_data`` order, **including** ``frame_index``.  This is the
    public, frame-aligned companion to :func:`get_metadata` — the basis for
    plotting any column vs frame (or vs another column).  Returns ``{}`` when the
    file has no ``scan_data``.
    """
    if frames is not None:
        return _scan_data_for_frames(scan_file, frames, entry=entry)
    out: dict[str, np.ndarray] = {}
    with h5py.File(Path(scan_file), "r") as f:
        e = _entry(f, entry)
        if "scan_data" not in e:
            return out
        for key, item in e["scan_data"].items():
            if isinstance(item, h5py.Dataset):
                out[str(key)] = _dataset_values(item)
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


def get_average_finite_counts(
    scan_file: str | Path,
    *,
    entry: str = "entry",
    max_bytes: int = 268_435_456,
):
    """Read and authenticate the optional Average per-pixel count map.
    The caller opts into the full-resolution allocation explicitly.  Layout
    and the complete modeled Python-owned peak are checked before that owner
    is allocated; values are then read and authenticated in bounded row slabs.
    """
    from xrd_tools.io.nexus_record import _average_count_chunks
    from xrd_tools.io.schema import detect_capabilities
    from xrd_tools.reduction.average import (
        AverageFiniteCounts,
        AverageFiniteCountsEvidence,
    )
    if type(entry) is not str or not entry or "/" in entry or entry in {".", ".."}:
        raise ValueError("Average count entry must be one exact group component")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive non-Boolean integer")
    with h5py.File(Path(scan_file), "r") as handle:
        entry_group = _entry(handle, entry)
        if "average_finite_counts" not in detect_capabilities(entry_group):
            raise ValueError(
                f"selected entry {entry!r} has no Average count capability",
            )
        frames_link = entry_group.get("frames", getlink=True)
        frames = entry_group.get("frames")
        frame_link = (
            None if not isinstance(frames, h5py.Group)
            else frames.get("frame_0001", getlink=True)
        )
        frame = (
            None if not isinstance(frames, h5py.Group)
            else frames.get("frame_0001")
        )
        if (
            type(frames_link) is not h5py.HardLink
            or not isinstance(frames, h5py.Group)
            or set(frames) != {"frame_0001"}
            or type(frame_link) is not h5py.HardLink
            or not isinstance(frame, h5py.Group)
            or "source" in frame
        ):
            raise ValueError(
                "Average count capability requires sole source-free frame 1",
            )
        dataset_link = frame.get("finite_counts", getlink=True)
        dataset = frame.get("finite_counts")
        attr_names = {
            "average_scan_policy",
            "contributor_extent",
            "finite_counts_sha256",
            "finite_counts_min",
            "finite_counts_max",
            "finite_counts_zero_count",
        }
        if (
            type(dataset_link) is not h5py.HardLink
            or not isinstance(dataset, h5py.Dataset)
            or dataset.dtype != np.dtype("<u4")
            or dataset.ndim != 2
            or any(int(value) <= 0 for value in dataset.shape)
            or dataset.is_virtual
            or dataset.external is not None
            or dataset.compression != "gzip"
            or dataset.compression_opts != 1
            or dataset.shuffle is not True
            or dataset.fletcher32 is not False
            or set(dataset.attrs) != attr_names
        ):
            raise ValueError("Average finite-count dataset schema or codec is invalid")
        shape = tuple(int(value) for value in dataset.shape)
        chunks = _average_count_chunks(shape)
        if dataset.chunks != chunks:
            raise ValueError("Average finite-count chunk layout is invalid")
        def scalar(name: str, dtype: str):
            value = dataset.attrs[name]
            if (
                np.asarray(value).shape != ()
                or dataset.attrs.get_id(name).dtype != np.dtype(dtype)
            ):
                raise ValueError(f"Average finite-count attribute {name} is invalid")
            return value
        policy_raw = scalar("average_scan_policy", "S15")
        extent_raw = scalar("contributor_extent", "<u4")
        digest_raw = scalar("finite_counts_sha256", "S64")
        minimum_raw = scalar("finite_counts_min", "<u4")
        maximum_raw = scalar("finite_counts_max", "<u4")
        zero_raw = scalar("finite_counts_zero_count", "<u8")
        try:
            policy = bytes(policy_raw).decode("ascii", errors="strict")
            expected_digest = bytes(digest_raw).decode("ascii", errors="strict")
        except (TypeError, UnicodeDecodeError) as error:
            raise ValueError("Average finite-count text attributes are invalid") from error
        extent = int(extent_raw)
        minimum = int(minimum_raw)
        maximum = int(maximum_raw)
        zero_count = int(zero_raw)
        evidence = AverageFiniteCountsEvidence(
            policy,
            extent,
            shape,
            "<u4",
            expected_digest,
            minimum,
            maximum,
            zero_count,
            chunks,
            "gzip",
            1,
            True,
            False,
        )
        count_bytes = 4 * shape[0] * shape[1]
        slab_bytes = 4 * chunks[0] * shape[1]
        if count_bytes + 4 * slab_bytes > max_bytes:
            raise ValueError("Average finite-count read exceeds the byte budget")
        owner = np.empty(shape, dtype="<u4", order="C")
        digest = hashlib.sha256(b"xdart.average-finite-counts.v1\0")
        digest.update(struct.pack("<QQQ", shape[0], shape[1], extent))
        observed_min = extent
        observed_max = 0
        observed_zero = 0
        for start in range(0, shape[0], chunks[0]):
            stop = min(shape[0], start + chunks[0])
            slab = np.asarray(dataset[start:stop])
            if slab.dtype != np.dtype("<u4") or slab.shape != (stop - start, shape[1]):
                raise ValueError("Average finite-count slab is invalid")
            owner[start:stop] = slab
            digest.update(slab.tobytes(order="C"))
            observed_min = min(observed_min, int(slab.min()))
            observed_max = max(observed_max, int(slab.max()))
            observed_zero += int(np.count_nonzero(slab == 0))
        if digest.hexdigest() != expected_digest:
            raise ValueError("Average finite-count digest is invalid")
        if (
            observed_min != minimum
            or observed_max != maximum
            or observed_zero != zero_count
            or observed_max > extent
            or observed_zero >= shape[0] * shape[1]
        ):
            raise ValueError("Average finite-count census or extent is invalid")
        return AverageFiniteCounts._from_owned_array(evidence, owner)
def _read_average_lineage(scan_file: str | Path, *, entry: str = "entry"):
    """Return one detached, authenticated committed source-lineage snapshot."""
    from xrd_tools.io.append import _source_from_dict, decode_replacement_lineage
    with h5py.File(Path(scan_file), "r") as handle:
        source_base, _raw, lineage = decode_replacement_lineage(handle, entry=entry)
    if lineage is None or lineage.get("state") != "committed":
        raise ValueError("Average lineage is absent or not committed")
    epochs = lineage.get("epochs")
    if type(epochs) is not list or len(epochs) != 1:
        raise ValueError("Average lineage requires one committed epoch")
    epoch = epochs[0]
    if type(epoch) is not dict or epoch.get("labels") != [1] or type(epoch.get("source")) is not dict:
        raise ValueError("Average lineage has invalid labels or source")
    return source_base, _source_from_dict(epoch["source"])
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
        Path to the current processed ``.nexus`` file.
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
    spec = SCHEMA.groups["integrated_1d"]
    (q_name,) = spec.axes
    with h5py.File(Path(scan_file), "r") as f:
        e = _entry(f, entry)
        g = _group_or_shadow(e, spec.name)
        if g is None:
            raise KeyError(f"{scan_file} has no {spec.name} group")
        positions, frames, single = _resolve_positions(
            _frame_index(e, prefer=spec.name), frame)

        q = np.asarray(g[q_name][()])
        q_unit = (_decode(g[q_name].attrs.get("units"))
                  if "units" in g[q_name].attrs else None)
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
    spec = SCHEMA.groups["integrated_2d"]
    q_name, chi_name = spec.axes
    with h5py.File(Path(scan_file), "r") as f:
        e = _entry(f, entry)
        g = _group_or_shadow(e, spec.name)
        if g is None:
            raise KeyError(f"{scan_file} has no {spec.name} group")
        positions, frames, single = _resolve_positions(
            _frame_index(e, prefer=spec.name), frame)

        q = np.asarray(g[q_name][()])
        chi = np.asarray(g[chi_name][()])
        q_unit = (_decode(g[q_name].attrs.get("units"))
                  if "units" in g[q_name].attrs else None)
        chi_unit = (
            _decode(g[chi_name].attrs.get("units"))
            if "units" in g[chi_name].attrs else None
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
    from xrd_tools.io.schema import THUMBNAIL_LUT_ATTRS

    vmin_key, vmax_key, dtype_key = THUMBNAIL_LUT_ATTRS
    arr = np.asarray(ds[()], dtype=float)
    vmin = float(ds.attrs.get(vmin_key, 0.0))
    vmax = float(ds.attrs.get(vmax_key, 1.0))
    dt = _decode(ds.attrs.get(dtype_key, "uint8"))
    scale = 65535.0 if str(dt) == "uint16" else 255.0
    result = vmin + (arr / scale) * (vmax - vmin)
    mask = ds.parent.get("thumbnail_mask")
    if mask is not None:
        invalid = np.asarray(mask[()], dtype=bool)
        if invalid.shape == result.shape:
            result[invalid] = np.nan
    return result


def resolve_project_source_path(
    locator: str,
    source_root: str | Path,
    *,
    must_exist: bool,
) -> Path:
    """Resolve one relative POSIX locator beneath an authenticated Project root.

    Both the root and candidate are resolved before containment is checked, so
    a symlink inside the Project cannot redirect a relative locator outside it.
    """
    if type(locator) is not str or not locator or "\\" in locator:
        raise ValueError("source locator is not canonical relative POSIX text")
    if type(must_exist) is not bool:
        raise TypeError("must_exist must be an exact bool")
    shown = PurePosixPath(locator)
    if (
        shown.is_absolute()
        or str(shown) != locator
        or locator == "."
        or "." in shown.parts
        or ".." in shown.parts
    ):
        raise ValueError("source locator is not canonical relative POSIX text")
    root_text = os.fspath(source_root)
    root = Path(root_text)
    normalized_root = os.path.normcase(
        os.path.abspath(os.path.normpath(root_text))
    )
    if not root.is_absolute() or normalized_root != root_text:
        raise ValueError("source root must be normalized absolute text")
    try:
        resolved_root = root.resolve(strict=False)
        resolved = root.joinpath(*shown.parts).resolve(strict=must_exist)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as error:
        raise ValueError("relative source locator escapes Project root") from error
    return resolved


def resolve_source_master(
    stored_path,
    *,
    scan_file: str | Path,
    source_base: str | None = None,
    source_root: str | Path | None = None,
) -> "Path | None":
    """Resolve a stored frame ``source/path`` to an EXISTING raw master (N1).

    The stored path may be **absolute** (old files, or a raw browsed OUTSIDE
    the project root -> used as-is) or **relative** (the portable form: a POSIX
    relpath against a project root).  Relative paths are joined against each
    root in PRECEDENCE order and the first existing candidate wins:

        explicit ``source_root``  >  the file's stored ``@source_base``.

    POSIX-stored relatives convert cross-OS via :class:`PurePosixPath`.  No
    basename, output-directory, parent-directory, or CWD guess is allowed: a
    moved tree must supply the newly selected Project root as ``source_root``.
    Returns ``None`` if neither explicit root resolves the exact stored path.

    This is the single source of N1 path resolution; every reader
    (``get_raw_frame``, ``image_source``, the frame-view reader) routes through
    it so they agree on the explicit relocation contract.
    """
    if not stored_path:
        return None
    raw = str(stored_path)
    if not raw or "\\" in raw:
        return None
    shown = PurePosixPath(raw)
    if shown.is_absolute():
        if str(shown) != raw or "." in shown.parts or ".." in shown.parts:
            return None
        candidate = Path(str(shown))
    else:
        selected_root = source_root if source_root is not None else source_base
        if selected_root is None:
            return None
        try:
            return resolve_project_source_path(
                raw, selected_root, must_exist=True,
            )
        except (TypeError, ValueError):
            return None
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        return None
    return resolved if resolved.exists() else None


_OUTSIDE_ROOT_WARNED: set = set()    # (source_dir, root) pairs already warned


@dataclass(frozen=True)
class RawSourceObservation:
    """Typed, pixel-free processed-record provenance observation."""

    source: Path | None
    definitive: bool
    detail: str = ""


def observe_resolved_raw_source(
    scan_file,
    frame: int | None = None,
    *,
    entry: str = "entry",
    source_root=None,
) -> RawSourceObservation:
    """Resolve one frame's stored raw-source provenance WITHOUT loading pixels.

    H18-R9: record identity for capability decisions — returns the absolute
    existing raw master ``Path`` the record's ``frames/frame_NNNN/source``
    pointer resolves to (N1 precedence via :func:`resolve_source_master`),
    or ``None`` when the record has no usable pointer or nothing resolves.
    One tiny metadata read; never decodes an image.

    H18-R13: the DEFAULT (``frame=None``) means the first ACTUAL stored
    source-bearing frame — real Eiger records are commonly one-based, so an
    invented label ``0`` resolved nothing for them.  An explicit ``frame``
    label stays an exact lookup and fails closed when that label has no
    stored source.
    """
    from xrd_tools.sources.probe import TRANSIENT_PROBE_ERRORS

    scan_file = Path(scan_file)
    try:
        with h5py.File(scan_file, "r") as f:
            entry_grp = _entry(f, entry)
            if frame is None:
                frame_group = _first_source_frame_group(entry_grp)
            else:
                frame_group = _frame_group_for_label(entry_grp, int(frame))
            if frame_group is None:
                return RawSourceObservation(None, True, "no source frame")
            master, _idx, _thumb = _raw_frame_parts_from_group(
                scan_file,
                entry_grp,
                frame_group,
                source_root=source_root,
            )
    except TRANSIENT_PROBE_ERRORS as exc:
        return RawSourceObservation(
            None, False, f"transient provenance read error: {exc}")
    except Exception as exc:
        return RawSourceObservation(None, True, f"provenance unavailable: {exc}")
    return RawSourceObservation(master, True, "resolved provenance")


def resolved_raw_source(
    scan_file,
    frame: int | None = None,
    *,
    entry: str = "entry",
    source_root=None,
):
    """Return strict stored raw provenance, or ``None``.

    Uses the same explicit Project-root/source-base contract as pixel loading.
    Use :func:`observe_resolved_raw_source` when transient sharing failures
    must be retried instead of collapsed to ``None``.
    """
    return observe_resolved_raw_source(
        scan_file,
        frame,
        entry=entry,
        source_root=source_root,
    ).source


def relative_source_path(src, root=None) -> str:
    """N1 WRITE-side counterpart of :func:`resolve_source_master`: the portable
    string to store in ``source/path``.

    When ``src`` is INSIDE ``root`` (the project folder), return the **POSIX
    relpath** against it (depth-robust + cross-OS).  When ``src`` is OUTSIDE the
    root, or no root is given, return the **absolute POSIX path** -- and warn for
    the outside-root case (out-of-tree raw is loadable but not portable).  Pair
    this with ``@source_base = root`` written on the entry.
    """
    import os
    src_abs = os.path.abspath(os.path.expanduser(str(src)))
    if root:
        root_abs = os.path.abspath(os.path.expanduser(str(root)))
        try:
            inside = os.path.commonpath([src_abs, root_abs]) == root_abs
        except ValueError:          # different drives (Windows) -> not inside
            inside = False
        if inside:
            return Path(os.path.relpath(src_abs, root_abs)).as_posix()
        # Warn ONCE per (source directory, root): every frame of a scan
        # shares the source dir (one Eiger master / one TIFF series dir),
        # so the per-frame call otherwise repeated this for the whole scan.
        _key = (os.path.dirname(src_abs), root_abs)
        if _key in _OUTSIDE_ROOT_WARNED:
            return Path(src_abs).as_posix()
        _OUTSIDE_ROOT_WARNED.add(_key)
        logger.warning("source %s is outside the project root %s; storing an "
                       "absolute (non-portable) path", src_abs, root_abs)
    return Path(src_abs).as_posix()


def get_raw_frame(
    scan_file: str | Path,
    frame: int,
    *,
    scan: str | int | None = None,
    entry: str = "entry",
    allow_thumbnail: bool = True,
    source_root: str | Path | None = None,
    preserve_dtype: bool = False,
) -> np.ndarray:
    """Return the raw detector image for one ``frame`` of a processed scan.

    A processed v2 ``.nexus`` stores integrated patterns, not raw detector
    images — but each frame carries a *source pointer*
    (``frames/frame_NNNN/source/{path,frame_index}``) back to the original
    detector master plus a quantized *thumbnail*.  This resolves the source
    pointer via :func:`resolve_source_master` (N1: a relative ``path`` joins
    against the selected ``source_root`` when supplied, otherwise the file's
    ``@source_base``; an absolute ``path`` is used as-is) and reads the full-resolution
    raw image via :func:`xrd_tools.io.image.read_image`.  If the master
    can't be located or read, it falls back to the stored thumbnail
    (dequantized) unless ``allow_thumbnail=False``.

    ``source_root`` (N1) repoints relative source paths at a moved data tree,
    overriding the stored ``@source_base`` — pass it when the raw was relocated
    after processing.

    ``frame`` is the frame **label** (the ``frame_index`` value), matching
    the other ``get_*`` readers; ``scan`` selects a contributing scan for a
    grouped Stitch/RSM result (``frames/scan_<scan>/frame_NNNN``), ``None`` is
    the flat single-scan record.  Raises ``KeyError`` when neither a usable
    source pointer nor a thumbnail is present.
    ``preserve_dtype=True`` is reserved for detector-aware presentation
    callers; the default retains the historical floating-point result.
    """
    scan_file = Path(scan_file)
    with h5py.File(scan_file, "r") as f:
        master, src_frame_idx, thumb = _raw_frame_parts_from_entry(
            scan_file,
            _entry(f, entry),
            int(frame),
            scan=scan,
            source_root=source_root,
        )
    return _raw_frame_or_thumbnail(
        scan_file,
        int(frame),
        master,
        src_frame_idx,
        thumb,
        allow_thumbnail=allow_thumbnail,
        preserve_dtype=preserve_dtype,
    )


def _raw_frame_from_entry(
    scan_file: Path,
    entry_group: h5py.Group,
    frame: int,
    *,
    allow_thumbnail: bool = True,
    source_root: str | Path | None = None,
) -> np.ndarray:
    """Internal open-file version of :func:`get_raw_frame`.

    ``ProcessedScan.iter_chunks`` uses this to avoid reopening the processed
    NeXus file for every frame while preserving the public reader behavior.
    """
    master, src_frame_idx, thumb = _raw_frame_parts_from_entry(
        scan_file,
        entry_group,
        frame,
        source_root=source_root,
    )
    return _raw_frame_or_thumbnail(
        scan_file,
        frame,
        master,
        src_frame_idx,
        thumb,
        allow_thumbnail=allow_thumbnail,
    )


def _raw_frame_parts_from_entry(
    scan_file: Path,
    entry_group: h5py.Group,
    frame: int,
    *,
    scan: str | int | None = None,
    source_root: str | Path | None = None,
) -> tuple[Path | None, int, np.ndarray | None]:
    """Resolve one processed frame's raw source pointer and thumbnail.

    ``scan`` selects a grouped-scan record (``frames/scan_<scan>/frame_NNNN``);
    ``None`` is the flat single-scan record (``frames/frame_NNNN``).
    """

    frame_group = _frame_group_for_label(entry_group, frame, scan=scan)
    if frame_group is None:
        raise KeyError(
            f"No frame group for frame {frame}"
            f"{f' (scan {scan})' if scan is not None else ''} in {scan_file}")
    return _raw_frame_parts_from_group(
        scan_file,
        entry_group,
        frame_group,
        source_root=source_root,
    )


def _frame_group_for_label(
    entry_group: h5py.Group,
    frame: int,
    *,
    scan: str | int | None = None,
) -> h5py.Group | None:
    """Resolve a stored frame label without assuming zero-padding width."""
    from xrd_tools.io.nexus_record import frame_record_key  # noqa: PLC0415

    frames = entry_group.get("frames")
    if not isinstance(frames, h5py.Group):
        return None
    parent = frames
    if scan is not None:
        parent = frames.get(f"scan_{scan}")
        if not isinstance(parent, h5py.Group):
            return None
    expected = frame_record_key(None, frame)
    direct = parent.get(expected)
    if isinstance(direct, h5py.Group):
        return direct
    matches = []
    for name in parent:
        if not name.startswith("frame_"):
            continue
        try:
            if int(name.removeprefix("frame_")) == int(frame):
                matches.append(name)
        except ValueError:
            continue
    if len(matches) != 1:
        return None
    group = parent.get(matches[0])
    return group if isinstance(group, h5py.Group) else None


def _first_source_frame_group(entry_group: h5py.Group) -> h5py.Group | None:
    """First source-bearing flat frame group in numeric label order."""
    frames = entry_group.get("frames")
    if not isinstance(frames, h5py.Group):
        return None
    candidates = []
    for name in frames:
        if not name.startswith("frame_"):
            continue
        group = frames.get(name)
        if not isinstance(group, h5py.Group) or "source" not in group:
            continue
        try:
            label = int(name.removeprefix("frame_"))
        except ValueError:
            continue
        candidates.append((label, name, group))
    return min(candidates, key=lambda item: (item[0], item[1]))[2] \
        if candidates else None


def _raw_frame_parts_from_group(
    scan_file: Path,
    entry_group: h5py.Group,
    frame_group: h5py.Group,
    *,
    source_root: str | Path | None = None,
) -> tuple[Path | None, int, np.ndarray | None]:
    """Resolve raw parts from an already-selected stored frame group."""
    source_base = (
        _decode(entry_group.attrs["source_base"])
        if "source_base" in entry_group.attrs
        else None
    )

    master: Path | None = None
    src_frame_idx = 0
    src = frame_group.get("source")
    if src is not None and "path" in src:
        rel = _decode(src["path"][()])
        if "frame_index" in src:
            src_frame_idx = int(np.asarray(src["frame_index"][()]).ravel()[0])
        master = resolve_source_master(
            rel,
            scan_file=scan_file,
            source_base=source_base,
            source_root=source_root,
        )

    thumb: np.ndarray | None = None
    thumb_ds = frame_group.get("thumbnail")
    if thumb_ds is not None:
        thumb = _dequantize_thumbnail(thumb_ds)

    return master, src_frame_idx, thumb


def _raw_frame_or_thumbnail(
    scan_file: Path,
    frame: int,
    master: Path | None,
    src_frame_idx: int,
    thumb: np.ndarray | None,
    *,
    allow_thumbnail: bool = True,
    preserve_dtype: bool = False,
) -> np.ndarray:
    from xrd_tools.io.image import read_image

    if master is not None:
        try:
            image = read_image(
                master,
                frame=src_frame_idx,
                preserve_dtype=preserve_dtype,
            )
            return (
                np.asarray(image)
                if preserve_dtype
                else np.asarray(image, dtype=float)
            )
        except Exception:
            logger.debug(
                "get_raw_frame: failed reading master %s frame %d; %s thumbnail",
                master,
                src_frame_idx,
                "falling back to" if allow_thumbnail else "not falling back to",
                exc_info=True,
            )
    if allow_thumbnail and thumb is not None:
        return thumb
    raise KeyError(
        f"frame {frame}: source master file not found/readable"
        + (
            f" and no thumbnail stored in {scan_file}"
            if allow_thumbnail
            else "; thumbnail fallback disabled for strict raw loading"
        )
    )


def _bluesky_metadata(entry) -> dict:
    """Build the :func:`get_metadata` dict for a Bluesky/NXWriter file directly
    from the Bluesky harvesters (motors + counters + eiger config)."""
    from xrd_tools.io.bluesky_nexus import (
        bluesky_angles,
        bluesky_constant_metadata,
        bluesky_energy_kev,
        bluesky_per_frame_table,
        bluesky_scalar_metadata,
        bluesky_wavelength,
    )

    angles = bluesky_angles(entry)
    scan_data = bluesky_per_frame_table(entry)
    scalars = bluesky_scalar_metadata(entry)
    wl = bluesky_wavelength(entry)
    en = bluesky_energy_kev(entry)

    n = 0
    for arr in scan_data.values():
        n = int(np.asarray(arr).shape[0])
        break
    if n == 0 and "num_points" in scalars:
        n = int(scalars["num_points"])

    # Held-fixed motors + eiger counting time are per-scan constants: broadcast
    # each across all n frames so Plot Metadata / scan_data shows them as columns.
    if n:
        for name, val in bluesky_constant_metadata(
                entry, exclude=scan_data.keys()).items():
            scan_data[name] = np.full(n, float(val), dtype=float)

    return {
        "frames": np.arange(n, dtype=np.int64),
        "n_frames": n,
        "has_1d": False,
        "has_2d": False,
        "sample_name": str(scalars.get("title", "") or ""),
        "energy_keV": float(en) if np.isfinite(en) else None,
        "wavelength_A": float(wl) if np.isfinite(wl) else None,
        "ub_matrix": None,
        "capabilities": [],
        "positioners": angles,
        "scan_data": scan_data,
        "reduction": {"bluesky": scalars},
    }


def get_metadata(scan_file: str | Path, *, entry: str = "entry") -> dict:
    """Return a flat dict of scan-level metadata (no heavy intensity arrays).

    Keys: ``frames``, ``n_frames``, ``has_1d``, ``has_2d``, ``q``, ``q_2d``,
    ``chi`` (axes, when present), ``sample_name``, ``energy_keV`` and
    ``wavelength_A`` (``None`` when not recorded — never NaN, #78),
    ``ub_matrix`` (or ``None``), ``positioners`` (dict of
    per-frame **geometry-motor** arrays only), ``scan_data`` (dict of *all*
    per-frame columns — motors AND counters), and ``reduction`` (provenance).

    ``positioners`` is intentionally narrow (just the diffractometer motors
    from the ``sample``/``detector`` positioner groups) so geometry/
    normalization APIs that consume it stay unambiguous; the complete
    per-frame metadata table (i0, monitor, temperature, …) is in
    ``scan_data``.
    """
    # Bluesky / apstools NXWriter acquisition files are NOT xdart-processed
    # (no integrated stacks / scan_data group).  Build the metadata straight
    # from the Bluesky harvesters so Plot Metadata sees the real per-frame
    # columns (hy, i0..pd, EPOCH) instead of an empty table.
    from xrd_tools.io.bluesky_nexus import is_bluesky_nxwriter, resolve_nxentry
    with h5py.File(Path(scan_file), "r") as _bf:
        _be = resolve_nxentry(_bf, entry)
        if _be is not None and is_bluesky_nxwriter(_be):
            return _bluesky_metadata(_be)
        from xrd_tools.io.processed_scan_id import require_current_processed
        require_current_processed(_bf, entry)

    # Reuse the canonical metadata-only reader for axes / positioners /
    # provenance, then add the instrument/sample scalars it doesn't carry.
    from xrd_tools.io.nexus import (
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
        from xrd_tools.io.processed_scan_id import (
            require_current_processed_groups,
        )
        processed = require_current_processed_groups(
            f,
            entry,
            container=Path(scan_file),
        )
        e = processed.entry
        positioners: dict = {}
        for path_in in ("sample/positioners",
                        "instrument/detector/positioners",
                        "instrument/positioners"):
            if path_in in e:
                positioners.update(_read_positioners(e[path_in]))
        # #78: the io.read boundary reports "absent" as None, never NaN —
        # the internal nexus readers keep their NaN sentinel.
        energy = _read_energy(e)
        wavelength = _read_wavelength(e, energy)
        from xrd_tools.io.schema import detect_capabilities
        capabilities = sorted(detect_capabilities(e))
        meta = {
            "frames": np.asarray(ds["frame"].values) if "frame" in ds.coords else np.array([]),
            "has_1d": processed.integrated_1d is not None,
            "has_2d": processed.integrated_2d is not None,
            "sample_name": _read_sample_name(e),
            "energy_keV": float(energy) if np.isfinite(energy) else None,
            "wavelength_A": (float(wavelength) if np.isfinite(wavelength)
                             else None),
            "ub_matrix": _read_ub_matrix(e),
            # 2f: which optional v2 features this file carries
            # (feature-detected per the schema capability registry)
            "capabilities": capabilities,
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
    positions = np.asarray(positions)
    # h5py fancy indexing needs strictly increasing indices.  When the request
    # is already strictly increasing (the common case: frame=None gives a
    # contiguous arange, and v2 frame_index is sorted) read directly — the
    # argsort + inverse-gather below is an identity that, unconditional, costs a
    # full extra copy of the WHOLE stack (~1.9 GB transient for a 651-frame 2D
    # cake at frame=None).  Only reorder for genuinely unsorted/duplicate input.
    if positions.size > 1 and not np.all(np.diff(positions) > 0):
        order = np.argsort(positions, kind="stable")
        out = np.asarray(dset[positions[order]])
        # restore caller-requested order
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        out = out[inv]
    else:
        out = np.asarray(dset[positions])
    return out[0] if single else out


def get_diffractometer(scan_file: str | Path, *, entry: str = "entry"):
    """Read the persisted canonical :class:`Diffractometer`, or ``None``.

    Returns ``None`` when the current record has no ``diffractometer`` group;
    never synthesizes a default geometry.  Reconstructs the full instrument (both
    adapter views + the fitted ``DetectorCalibration`` + preset + motor map)
    from the ``config_json`` blob for offline stitch/RSM.
    """
    from xrd_tools.core.geometry import Diffractometer

    with h5py.File(Path(scan_file), "r") as f:
        grp = _entry(f, entry)
        ds = grp.get("diffractometer/config_json")
        if ds is None:
            return None
        blob = ds.asstr()[()] if h5py.check_string_dtype(ds.dtype) else ds[()]
        if isinstance(blob, (bytes, np.bytes_)):
            blob = blob.decode("utf-8")
    try:
        return Diffractometer.from_json(str(blob))
    except Exception:
        logger.warning("Could not parse persisted diffractometer blob in %s",
                       scan_file, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# object-style sugar
# ---------------------------------------------------------------------------

class ProcessedScan:
    """Lightweight handle to a processed scan file.

    THE READ-SIDE handle: thin sugar over the module-level ``get_*``
    functions so notebook code can read ``scan.get_1d(3)`` instead of
    repeating the path.  Holds no open file handle and caches only
    lightweight metadata; image and integration slices are always read
    on demand.  Satisfies the :class:`~xrd_tools.core.scan.FrameSource`
    contract (contract-pinned), so it also feeds RSM/stitching/
    reduction directly.  Not to be confused with
    :class:`xrd_tools.core.scan.Scan`, the reduction INPUT — see the
    name-resolution note in CLAUDE.md.
    """

    def __init__(self, scan_file: str | Path, *, entry: str = "entry",
                 source_root: str | Path | None = None):
        self.path = Path(scan_file)
        self.entry = entry
        from xrd_tools.io.processed_scan_id import require_current_processed
        require_current_processed(self.path, entry)
        # N1: repoint a moved raw tree for load_frame/iter_chunks (overrides the
        # stored @source_base).
        self.source_root = source_root
        self._metadata_cache: dict | None = None
        self._scan_data_cache: dict[str, np.ndarray] | None = None
        self._frames_cache: np.ndarray | None = None
        #: ``None`` is a valid value (absent group), so cache via a flag.
        self._diffractometer_cache = None
        self._diffractometer_loaded = False

    @property
    def frames(self) -> np.ndarray:
        if self._frames_cache is None:
            self._frames_cache = get_frames(self.path, entry=self.entry, union=True)
        return np.array(self._frames_cache, copy=True)

    @property
    def frame_indices(self) -> list[int]:
        return [int(frame) for frame in self.frames]

    @property
    def capabilities(self):
        """FrameSource capability advertisement (completes the duck
        contract the RSM/stitch boundary consumes — pinned by the
        contract tests)."""
        from xrd_tools.core.scan import SourceCapabilities

        return SourceCapabilities(
            is_streaming=False,
            supports_random_access=True,
            supports_chunks=True,
            has_metadata=True,
            has_geometry=True,
            has_raw_references=True,
            has_thumbnails=True,
        )

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
    def diffractometer(self):
        """The persisted canonical :class:`Diffractometer`, or ``None``.

        Lets offline stitch/RSM run from the file with no GUI (the "metadata
        mandatory for stitch/RSM" contract).  ``None`` on any file written
        before the group existed.
        """
        if not self._diffractometer_loaded:
            self._diffractometer_cache = get_diffractometer(
                self.path, entry=self.entry)
            self._diffractometer_loaded = True
        return self._diffractometer_cache

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
        self._frames_cache = None
        self._diffractometer_cache = None
        self._diffractometer_loaded = False
        return self.metadata

    def get_1d(self, frame=None) -> Integrated1D:
        return get_1d(self.path, frame, entry=self.entry)

    def get_2d(self, frame=None) -> Integrated2D:
        return get_2d(self.path, frame, entry=self.entry)

    def get_thumbnail(self, frame: int) -> np.ndarray:
        return get_thumbnail(self.path, frame, entry=self.entry)

    def metadata_for(self, index: int) -> Mapping[str, Any]:
        """Return per-frame metadata/scan-data for ``index``.

        ``ProcessedScan`` is the notebook-friendly reader, but stitch/ROI paths
        also consume it as a lightweight FrameSource.  Surface the persisted
        scan-data columns aligned to the stored frame labels; whole-scan scalar
        metadata remains available through ``metadata``.
        """
        label = int(index)
        labels = self.frame_indices
        try:
            pos = labels.index(label)
        except ValueError as exc:
            raise KeyError(f"frame {label!r} not present in {self.path}") from exc

        result: dict[str, Any] = {"frame_index": label}
        for key, values in self.scan_data.items():
            arr = np.asarray(values)
            try:
                if arr.shape == ():
                    result[key] = arr.item()
                elif len(arr) == len(labels):
                    value = arr[pos]
                    result[key] = value.item() if hasattr(value, "item") else value
            except Exception:
                continue
        return result

    def frame_for(self, index: int) -> ScanFrame:
        label = int(index)
        return ScanFrame(
            index=label,
            metadata=dict(self.metadata_for(label)),
            loader=lambda frame: self.load_frame(frame.index),
            source_identity=str(self.path),
        )

    def load_frame(self, index: int) -> np.ndarray:
        """Load one raw detector frame through its stored source pointer.

        ``source_root`` (N1, from the constructor) repoints a moved raw tree."""
        return get_raw_frame(
            self.path, int(index), entry=self.entry, allow_thumbnail=False,
            source_root=self.source_root,
        )

    def iter_chunks(self, chunk_size: int):
        """Yield bounded raw-image chunks for RSM and other streaming consumers."""
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0; got {chunk_size}")
        indices = self.frame_indices
        with h5py.File(self.path, "r") as f:
            entry_group = _entry(f, self.entry)
            for start in range(0, len(indices), chunk_size):
                chunk_indices = indices[start:start + chunk_size]
                yield np.stack([
                    _raw_frame_from_entry(
                        self.path,
                        entry_group,
                        idx,
                        allow_thumbnail=False,
                        source_root=self.source_root,
                    )
                    for idx in chunk_indices
                ]), chunk_indices

    def __len__(self) -> int:
        try:
            return int(self.frames.size)
        except KeyError:
            return 0

    def __repr__(self) -> str:
        return f"ProcessedScan({self.path.name!r}, n_frames={len(self)})"


def open_scan(scan_file: str | Path, *, entry: str = "entry",
              source_root: str | Path | None = None) -> ProcessedScan:
    """Return a :class:`ProcessedScan` handle for ``scan_file`` (notebook
    sugar).

    ``source_root`` (N1) repoints relative raw-source paths at a moved data
    tree for ``ProcessedScan.load_frame`` / ``iter_chunks`` (overrides the
    stored ``@source_base``)."""
    return ProcessedScan(scan_file, entry=entry, source_root=source_root)
