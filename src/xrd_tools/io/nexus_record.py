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

from xrd_tools.io.schema import (
    INTEGRATED_ROW_ALIGNED,
    SOURCE_BASE_ATTR,
    THUMBNAIL_LUT_ATTRS,
)

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
    "frame_record_key",
    "write_contributing_frames",
    "iter_frame_record_groups",
    "frame_record_labels",
    "harvest_frame_records",
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
    existing = entry_grp.attrs.get(SOURCE_BASE_ATTR)
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
        entry_grp.attrs[SOURCE_BASE_ATTR] = posix_base
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
    for key, value in zip(THUMBNAIL_LUT_ATTRS, lut):
        ds.attrs[key] = value


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


def frame_record_key(scan_label, frame_index: int) -> str:
    """The ``/entry/frames/<key>`` group name for one contributing frame.

    ``scan_label is None`` → flat ``frame_NNNN`` (single-scan, the reduction
    convention — backward-compatible).  A scan label → nested
    ``scan_<label>/frame_NNNN`` so frames from **grouped** scans (a Stitch/RSM over
    several scans) don't collide on the flat index.  The h5viewer Frames panel
    surfaces these as ``"<label>-<frame>"``.
    """
    base = f"frame_{int(frame_index):04d}"
    return base if scan_label is None else f"scan_{scan_label}/{base}"


def write_contributing_frames(entry_grp: h5py.Group, records, *,
                              source_base=None) -> int:
    """Write the per-frame **source records** for a Stitch/RSM result — the
    enabler for the raw-image popup (resolve a contributing frame from the saved
    ``.nxs``).  ``records`` is an iterable of mappings with ``frame_index`` and
    optionally ``scan_label`` / ``source_path`` / ``source_frame_index`` /
    ``thumbnail``.  Multi-scan records (a ``scan_label``) nest under
    ``scan_<label>/``; single-scan stays flat.  Returns the count written.
    """
    records = list(records)
    if not records:
        return 0
    if source_base is not None:
        stamp_source_base(entry_grp, source_base)
    frames = ensure_frames_container(entry_grp)
    # pre-create the scan subgroups so they carry NX_class (require_group via a
    # nested key would leave the intermediate group bare).
    for sl in sorted({r.get("scan_label") for r in records
                      if r.get("scan_label") is not None}, key=str):
        _nxcollection(frames, f"scan_{sl}")
    for r in records:
        write_frame_record(
            frames, frame_record_key(r.get("scan_label"), r["frame_index"]),
            source_path=r.get("source_path"),
            source_frame_index=int(r.get("source_frame_index") or 0),
            thumbnail=r.get("thumbnail"), source_base=source_base)
    return len(records)


def _frame_label_int(name: str):
    """``frame_0007`` → ``7`` (the int label), or ``None`` if it doesn't parse."""
    try:
        return int(name.removeprefix("frame_"))
    except ValueError:
        return None


def iter_frame_record_groups(frames_grp):
    """Yield ``(scan_label, frame_label, group)`` for every contributing-frame
    record, descending **one level** into nested ``scan_<N>/`` subgroups.

    ``scan_label`` is ``None`` for flat single-scan records (``frame_NNNN``) and
    the ``<N>`` string (as written) for grouped-scan records
    (``scan_<N>/frame_NNNN``).  Intermediate ``scan_<N>`` containers are not
    yielded themselves — only the leaf frame groups.  This is the grouped-aware
    counterpart to the flat ``frames/frame_NNNN`` iteration the integrated-frame
    readers do; use it wherever a Stitch/RSM result's contributing frames must be
    surfaced (the Frames-panel raw popup).
    """
    if frames_grp is None:
        return
    for name, obj in frames_grp.items():
        if not isinstance(obj, h5py.Group):
            continue
        if name.startswith("frame_"):
            yield None, _frame_label_int(name), obj
        elif name.startswith("scan_"):
            scan_label = name.removeprefix("scan_")
            for sub_name, sub_obj in obj.items():
                if isinstance(sub_obj, h5py.Group) and sub_name.startswith("frame_"):
                    yield scan_label, _frame_label_int(sub_name), sub_obj


def frame_record_labels(frames_grp):
    """The Frames-panel display list for a result's contributing frames.

    Returns ``[(label_text, scan_label, frame_label), ...]`` — only records with
    something loadable (a thumbnail or a source pointer).  Flat records sort
    first as bare ``"<frame>"``; grouped records follow as ``"<scan>-<frame>"``
    (Vivek's convention for grouped Stitch/RSM, e.g. ``5-1, 5-2, … 7-1``).
    ``scan_label``/``frame_label`` are the resolution address — pass them to
    :func:`~xrd_tools.io.read.get_raw_frame` (``scan=scan_label, frame=…``).
    """
    flat: list[int] = []
    grouped: list[tuple[str, int]] = []
    for scan_label, frame_label, grp in iter_frame_record_groups(frames_grp):
        if frame_label is None:
            continue
        if "thumbnail" not in grp and "source" not in grp:
            continue   # nothing the raw popup could resolve
        if scan_label is None:
            flat.append(frame_label)
        else:
            grouped.append((scan_label, frame_label))
    flat.sort()
    grouped.sort(key=lambda t: (str(t[0]), t[1]))
    out = [(str(f), None, f) for f in flat]
    out += [(f"{s}-{f}", s, f) for s, f in grouped]
    return out


def harvest_frame_records(source, *, scan_labels=None, selected_labels=None,
                          with_thumbnails: bool = False):
    """Build the ``frame_records`` list for a Stitch/RSM result from its source(s).

    Mirrors what :class:`~xrd_tools.reduction.NexusSink` writes per frame, but for
    a *whole-result* writer (:func:`write_stitched` / :func:`write_rsm` via
    :func:`write_contributing_frames`).  ``source`` may be a single FrameSource, a
    sequence of them (the ``run_rsm`` grouping), or a
    :class:`~xrd_tools.sources.composite.CompositeFrameSource` (the ``run_stitch``
    grouping) — the members are expanded either way.

    * one source                  → flat records (``scan_label=None``);
    * several sources/members      → one ``scan_label`` per member.  ``scan_labels``
      overrides (e.g. the real scan numbers ``[5, 7, 8]``); the default is
      ``1..N``.

    Pointer-only by default (cheap — no image load): each record carries the
    member's ``source_path`` / ``source_frame_index``, which the raw popup
    resolves to the original master.  ``with_thumbnails=True`` additionally bakes a
    downsampled preview (loads each frame).

    ``selected_labels`` restricts the records to the frames that **actually
    contributed** — pass ``run_stitch``'s reduced ``frame_indices`` here (in the
    *source*'s own label space: the composite's GLOBAL index for a group, the
    source's own labels otherwise).  ``None`` (the default) records every frame.
    Without this, a subselected stitch would persist records for frames that were
    never merged.
    """
    members = _expand_frame_sources(source)
    if scan_labels is None:
        labels = [None] if len(members) == 1 else list(range(1, len(members) + 1))
    else:
        labels = list(scan_labels)
        if len(labels) != len(members):
            raise ValueError(
                f"scan_labels has {len(labels)} entries but the source expands to "
                f"{len(members)} member(s)")
    wanted = _partition_selected_local_labels(source, members, selected_labels)
    records: list[dict] = []
    for mi, (scan_label, member) in enumerate(zip(labels, members)):
        want = wanted[mi] if wanted is not None else None
        for idx in member.frame_indices:
            if want is not None and int(idx) not in want:
                continue
            sf = member.frame_for(idx)
            sp = getattr(sf, "source_path", None)
            rec = {
                "scan_label": scan_label,
                "frame_index": int(idx),
                "source_path": str(sp) if sp is not None else None,
                "source_frame_index": int(getattr(sf, "source_frame_index", None) or 0),
            }
            if with_thumbnails:
                try:
                    rec["thumbnail"] = make_thumbnail_array(
                        np.asarray(member.load_frame(idx), dtype=np.float32))
                except Exception:  # noqa: BLE001 — a preview is best-effort
                    logger.debug("harvest_frame_records: thumbnail failed for "
                                 "%s frame %s", scan_label, idx, exc_info=True)
            records.append(rec)
    return records


def _partition_selected_local_labels(source, members, selected_labels):
    """Map ``selected_labels`` (in ``source``'s own label space) to a per-member
    set of LOCAL frame labels — ``None`` means "no restriction".

    For a :class:`~xrd_tools.sources.composite.CompositeFrameSource` the selection
    is in the GLOBAL ``0..N-1`` index, so it's resolved through the composite's
    ``_map`` (``global → (member_pos, local_label)``).  For a single source the
    selection is already that source's own labels.  A bare multi-member sequence
    with a selection is ambiguous — there is no global→local map to honour the
    selection, so silently widening to ALL frames would be a label-space footgun;
    raise instead (no caller hits this: run_stitch wraps groups in a
    CompositeFrameSource before harvest)."""
    if selected_labels is None:
        return None
    sel = {int(s) for s in selected_labels}
    gmap = getattr(source, "_map", None)   # composite: list[(member_pos, local)]
    if gmap is not None:
        per: list[set] = [set() for _ in members]
        for g in sel:
            member_pos, local = gmap[int(g)]
            per[int(member_pos)].add(int(local))
        return per
    if len(members) == 1:
        return [sel]
    raise ValueError(
        f"selected_labels={sorted(sel)} given for a bare {len(members)}-member "
        "sequence with no global→local index map; wrap the group in a "
        "CompositeFrameSource (or omit selected_labels) — silently recording all "
        "frames would ignore the selection.")


def _expand_frame_sources(source) -> list:
    """A composite → its members; a sequence → its items; a single source → ``[it]``.

    Duck-typed (no import of CompositeFrameSource) to keep this io module free of a
    ``sources`` dependency.  A CompositeFrameSource re-indexes its frames to a
    global ``0..N-1``; harvesting from its **members** instead recovers each
    contributing scan's own per-frame labels (so the records are scan-tagged, not
    flattened)."""
    members = getattr(source, "members", None)
    if members is not None:
        return list(members)
    if isinstance(source, (list, tuple)):
        return list(source)
    return [source]


# ---------------------------------------------------------------------------
# Integrated-stack row surgery
# ---------------------------------------------------------------------------

def _recreate_filter_kwargs(obj) -> dict:
    """``create_dataset`` kwargs that reproduce ``obj``'s filter pipeline.

    h5py's high-level ``.compression`` reports an hdf5plugin codec (e.g. LZ4,
    filter id 32004) as the string ``'unknown'`` with no ``compression_opts`` --
    replaying that crashes ``create_dataset``.  So inspect the filter ids directly
    and re-apply the known codecs: LZ4 -> hdf5plugin LZ4 (gzip if the plugin is
    missing), gzip/lzf -> portable gzip (never re-emit raw lzf).  ``shuffle`` /
    ``fletcher32`` carry through.  An unrecognized filter degrades to uncompressed
    rather than crash."""
    filters = dict(getattr(obj, "_filters", {}) or {})
    kw: dict = {}
    if "32004" in filters:                       # hdf5plugin LZ4
        try:
            import hdf5plugin
            kw.update(hdf5plugin.LZ4())
        except Exception:
            kw["compression"] = "gzip"
            kw["compression_opts"] = 1
    elif obj.compression in ("gzip", "lzf"):     # never re-emit raw lzf
        kw["compression"] = "gzip"
        kw["compression_opts"] = obj.compression_opts or 1
    if kw:                                        # a compressor is present
        if obj.shuffle:
            kw["shuffle"] = True
        if obj.fletcher32:
            kw["fletcher32"] = True
    return kw


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
        row_aligned = (data.shape[:1] == labels.shape
                       and key in INTEGRATED_ROW_ALIGNED)
        if row_aligned:
            data = data[keep_mask]
        datasets.append((
            key, data, dict(obj.attrs.items()),
            _recreate_filter_kwargs(obj), row_aligned, obj.chunks,
        ))

    del parent[name]
    new_group = parent.create_group(name)
    for key, value in group_attrs.items():
        new_group.attrs[key] = value
    for (key, data, attrs, filter_kwargs, row_aligned, chunks) in datasets:
        kwargs = {}
        if row_aligned:
            kwargs["maxshape"] = (None,) + tuple(np.asarray(data).shape[1:])
            if chunks is not None:
                kwargs["chunks"] = chunks
        kwargs.update(filter_kwargs)
        # A filter requires chunking; restore the source chunks if the recreate
        # path above didn't already set them (non-row-aligned filtered datasets).
        if filter_kwargs and "chunks" not in kwargs and chunks is not None:
            kwargs["chunks"] = chunks
        ds = new_group.create_dataset(key, data=data, **kwargs)
        for attr_key, value in attrs.items():
            ds.attrs[attr_key] = value
