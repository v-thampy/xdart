# -*- coding: utf-8 -*-
"""Complete-v2-record primitives — per-frame source refs, thumbnails,
``@source_base``, and integrated-stack row surgery.

This module is the first concrete piece of the *xrd-session* data-ownership
layer (greenfield design, Difference 2): the components of the processed-NeXus
v2 record that used to live only in xdart's GUI writer are public, headless
primitives here.  Both writers orchestrate THESE functions:

* the headless :class:`~xrd_tools.reduction.NexusSink` — so a purely
  headless run writes the **complete** v2 record (raw-source pointers that
  ``get_raw_frame`` can resolve, thumbnails, geometry, provenance);
* xdart's GUI writer — which keeps only the GUI-side concerns (append
  cursor, NFS retry, Qt signals) and calls down into this module for the
  record itself.

Layout written (per the v2 schema)::

    /entry/@source_base            POSIX project root (N1 portability)
    /entry/frames/                 NXcollection
        frame_NNNN/                NXcollection
            thumbnail              uint8/uint16 (@vmin, @vmax, @dtype)
            timestamp              str, optional
            source/                NXcollection, optional
                path               str (POSIX; relative to @source_base)
                frame_index        int (index within the source file)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)

#: default maximum thumbnail edge, matches the GUI's preview budget
THUMBNAIL_MAX = 256

__all__ = [
    "THUMBNAIL_MAX",
    "make_thumbnail_array",
    "quantize_thumbnail",
    "stamp_source_base",
    "ensure_frames_container",
    "write_frame_record",
    "write_frame_source_ref",
    "write_thumbnail",
    "drop_integrated_rows",
]


# ---------------------------------------------------------------------------
# Thumbnails
# ---------------------------------------------------------------------------

def make_thumbnail_array(image, *, mask_flat=None, global_mask_flat=None,
                         max_size: int = THUMBNAIL_MAX):
    """Downsample a 2D image to at most ``(max_size, max_size)``.

    Masked pixels (flat indices) become NaN *before* downsampling so the
    mask is baked into the preview — viewers need no full-resolution mask.
    Returns float32, or ``None`` for invalid input.
    """
    if image is None:
        return None
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 2:
        return None

    all_mask = []
    if mask_flat is not None and len(mask_flat) > 0:
        all_mask.append(np.asarray(mask_flat, dtype=int))
    if global_mask_flat is not None and len(global_mask_flat) > 0:
        all_mask.append(np.asarray(global_mask_flat, dtype=int))
    if all_mask:
        flat_mask = np.unique(np.concatenate(all_mask))
        flat_mask = flat_mask[flat_mask < arr.size]
        arr_flat = arr.ravel().copy()
        arr_flat[flat_mask] = np.nan
        arr = arr_flat.reshape(arr.shape)

    h, w = arr.shape
    if h <= max_size and w <= max_size:
        return arr
    from scipy.ndimage import zoom as ndimage_zoom
    factor = min(max_size / h, max_size / w)
    return ndimage_zoom(arr, factor, order=1).astype(np.float32)


def quantize_thumbnail(arr, dtype: str = "uint8"):
    """Linear-quantize a 2-D thumbnail to uint8/uint16.

    Returns ``(quantized, (vmin, vmax, dtype))`` — the LUT triple is stored
    as attributes so viewers can invert.
    """
    finite = np.isfinite(arr)
    if not finite.any():
        quant = np.zeros(
            arr.shape, dtype=np.uint8 if dtype == "uint8" else np.uint16
        )
        return quant, (0.0, 1.0, dtype)
    vmin, vmax = np.percentile(arr[finite], [1, 99])
    if vmax <= vmin:
        vmax = vmin + 1e-12
    # NaN/inf (masked pixels) -> vmin BEFORE the clip so they don't
    # propagate to (NaN * 255).astype(uint8) ("invalid value in cast").
    arr_clean = np.where(finite, arr, vmin)
    norm = np.clip((arr_clean - vmin) / (vmax - vmin), 0, 1)
    if dtype == "uint16":
        return (norm * 65535).astype(np.uint16), (float(vmin), float(vmax), "uint16")
    return (norm * 255).astype(np.uint8), (float(vmin), float(vmax), "uint8")


# ---------------------------------------------------------------------------
# @source_base (N1 portability root)
# ---------------------------------------------------------------------------

def stamp_source_base(entry_grp: h5py.Group, source_base) -> str | None:
    """Normalize + stamp the project root on ``entry/@source_base``.

    ONE scan-level root governs ALL frames' relative source paths: appending
    to a file written under a DIFFERENT root would silently rebase the
    earlier frames' pointers, so a mismatch raises rather than corrupting
    resolution.  Returns the normalized absolute base (native separators)
    for use with :func:`write_frame_source_ref`, or ``None`` when no base
    was given (absolute-path back-compat mode).
    """
    if not source_base:
        return None
    base = os.path.abspath(os.path.expanduser(str(source_base)))
    posix_base = Path(base).as_posix()
    existing = entry_grp.attrs.get("source_base")
    if existing is not None:
        if isinstance(existing, bytes):
            existing = existing.decode("utf-8", errors="replace")
        if str(existing) != posix_base:
            raise ValueError(
                f"cannot append to {os.fspath(entry_grp.file.filename)!r}: its "
                f"Project Folder (@source_base={str(existing)!r}) differs from "
                f"the current ({posix_base!r}).  Earlier frames' relative source "
                f"paths are stored against the old root; start a NEW output "
                f"file for the new Project Folder."
            )
    try:
        entry_grp.attrs["source_base"] = posix_base
    except Exception as exc:
        raise RuntimeError(
            f"failed to stamp @source_base={posix_base!r} on "
            f"{entry_grp.name!r}; relative raw source paths would be "
            "unresolvable"
        ) from exc
    return base


# ---------------------------------------------------------------------------
# Per-frame record groups
# ---------------------------------------------------------------------------

def _nxcollection(parent: h5py.Group, name: str) -> h5py.Group:
    grp = parent.require_group(name)
    grp.attrs.setdefault("NX_class", "NXcollection")
    return grp


def ensure_frames_container(entry_grp: h5py.Group) -> h5py.Group:
    """``entry/frames`` as an NXcollection (create-if-missing — re-creating
    would clobber per-frame groups from previous batches)."""
    return _nxcollection(entry_grp, "frames")


def write_thumbnail(frame_grp: h5py.Group, thumbnail,
                    dtype: str = "uint8") -> None:
    """Quantize + store ``thumbnail`` with its inversion LUT attributes."""
    arr, lut = quantize_thumbnail(np.asarray(thumbnail), dtype=dtype)
    ds = frame_grp.create_dataset("thumbnail", data=arr)
    ds.attrs["vmin"] = lut[0]
    ds.attrs["vmax"] = lut[1]
    ds.attrs["dtype"] = lut[2]


def write_frame_source_ref(frame_grp: h5py.Group, source_path,
                           frame_index: int, *,
                           source_base=None) -> None:
    """``source/{path,frame_index}`` — the raw-source pointer.

    ``path`` is stored RELATIVE to ``source_base`` (POSIX, portable) when
    the source sits inside it, else absolute POSIX (with a warning) — the
    N1 contract, via :func:`xrd_tools.io.read.relative_source_path`.
    """
    from xrd_tools.io.read import relative_source_path

    if not source_path:
        return
    sub = _nxcollection(frame_grp, "source")
    sub["path"] = relative_source_path(str(source_path), source_base)
    sub["frame_index"] = int(frame_index)


def write_frame_record(frames_grp: h5py.Group, frame_key: str, *,
                       thumbnail=None, thumbnail_dtype: str = "uint8",
                       source_path=None, source_frame_index: int = 0,
                       timestamp=None, source_base=None) -> h5py.Group:
    """Write one complete per-frame record group (idempotent per key).

    Per the v2 schema, per-frame groups carry *only* metadata + thumbnail —
    never the full raw image (an early writer dumped 18 MB per Eiger frame
    here; don't bring that back).
    """
    fg = _nxcollection(frames_grp, frame_key)
    if thumbnail is not None and "thumbnail" not in fg:
        write_thumbnail(fg, thumbnail, dtype=thumbnail_dtype)
    if source_path and "source" not in fg:
        write_frame_source_ref(fg, source_path, source_frame_index,
                               source_base=source_base)
    if timestamp is not None and "timestamp" not in fg:
        fg["timestamp"] = str(timestamp)
    return fg


# ---------------------------------------------------------------------------
# Integrated-stack row surgery
# ---------------------------------------------------------------------------

def drop_integrated_rows(h5f, group_path: str, frame_indices) -> None:
    """Remove stale rows from an existing ``integrated_*`` stack by frame
    label (rebuilds the group preserving compression/chunking/attrs)."""
    if group_path not in h5f or "frame_index" not in h5f[group_path]:
        return
    group = h5f[group_path]
    labels = np.asarray(group["frame_index"][()], dtype=np.int64)
    drop = {int(idx) for idx in frame_indices}
    keep_mask = np.asarray(
        [int(label) not in drop for label in labels], dtype=bool
    )
    if bool(np.all(keep_mask)):
        return

    parent_path, name = group_path.rsplit("/", 1)
    parent = h5f[parent_path]
    if not bool(np.any(keep_mask)):
        del parent[name]
        return

    group_attrs = dict(group.attrs.items())
    datasets = []
    for key, obj in group.items():
        if not isinstance(obj, h5py.Dataset):
            continue
        data = obj[()]
        row_aligned = data.shape[:1] == labels.shape and key in {
            "frame_index", "intensity", "sigma",
        }
        if row_aligned:
            data = data[keep_mask]
        datasets.append((
            key, data, dict(obj.attrs.items()), obj.compression,
            obj.compression_opts, obj.shuffle, obj.fletcher32,
            row_aligned, obj.chunks,
        ))

    del parent[name]
    new_group = parent.create_group(name)
    for key, value in group_attrs.items():
        new_group.attrs[key] = value
    for (key, data, attrs, compression, compression_opts, shuffle,
         fletcher32, row_aligned, chunks) in datasets:
        kwargs = {}
        if row_aligned:
            kwargs["maxshape"] = (None,) + tuple(np.asarray(data).shape[1:])
            if chunks is not None:
                kwargs["chunks"] = chunks
        if compression is not None:
            kwargs["compression"] = compression
            if compression_opts is not None:
                kwargs["compression_opts"] = compression_opts
            kwargs["shuffle"] = shuffle
            kwargs["fletcher32"] = fletcher32
        ds = new_group.create_dataset(key, data=data, **kwargs)
        for attr_key, value in attrs.items():
            ds.attrs[attr_key] = value
