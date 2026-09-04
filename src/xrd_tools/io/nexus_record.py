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

import logging, hashlib, json, re, struct
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import uuid

import h5py
import numpy as np

from xrd_tools.core import (
    DEFAULT_MODE_KEY,
    FrameRecord,
    FrameView,
    numeric_metadata,
)
from xrd_tools.io.schema import (
    INTEGRATED_ROW_ALIGNED,
    ARTIFACT_FAMILY_ATTR,
    SOURCE_BASE_ATTR,
    THUMBNAIL_LUT_ATTRS,
    canonical_gi_mode_key,
)

logger = logging.getLogger(__name__)

#: default maximum thumbnail edge, matches the GUI's preview budget
THUMBNAIL_MAX = 256
_MAX_BACKGROUND_DESCRIPTOR_BYTES = 262_144

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
    "write_background_dependency", "read_background_dependency",
    "write_average_finite_counts",
    "drop_integrated_rows",
    "frame_record_from_live_frame",
    "frame_record_from_reduction",
    "merge_frame_records",
]


_AVERAGE_COUNT_PREFIX = b"xdart.average-finite-counts.v1\0"
def _average_count_chunks(shape: tuple[int, int]) -> tuple[int, int]:
    height, width = shape; count_bytes = 4 * height * width
    cap = min(count_bytes, max(4 * width, 4 << 20))
    return min(height, max(1, cap // (4 * width))), width
def _average_count_digest(values: np.ndarray, contributor_extent: int, *, rows: int) -> str:
    height, width = values.shape
    digest = hashlib.sha256(_AVERAGE_COUNT_PREFIX)
    digest.update(struct.pack("<QQQ", height, width, contributor_extent))
    for start in range(0, height, rows):
        slab = np.ascontiguousarray(values[start:start + rows], dtype="<u4")
        digest.update(slab.tobytes(order="C"))
    return digest.hexdigest()
def write_average_finite_counts(entry_grp: h5py.Group, counts) -> h5py.Dataset:
    """Write the sole Average count map under the already-written frame 1."""
    from xrd_tools.reduction.average import AverageFiniteCounts, AverageFiniteCountsEvidence
    if type(counts) is not AverageFiniteCounts: raise ValueError("average finite-count finalization value is invalid")
    evidence = counts.evidence
    if type(evidence) is not AverageFiniteCountsEvidence: raise ValueError("average finite-count evidence is invalid")
    values = counts.values
    expected_chunks = _average_count_chunks(evidence.shape)
    if evidence.chunks != expected_chunks: raise ValueError("average finite-count chunk evidence is invalid")
    observed_digest = _average_count_digest(values, evidence.contributor_extent, rows=expected_chunks[0])
    if observed_digest != evidence.sha256: raise ValueError("average finite-count digest is invalid")
    frames = entry_grp.get("frames")
    frame = None if not isinstance(frames, h5py.Group) else frames.get("frame_0001")
    if not isinstance(frame, h5py.Group): raise ValueError("Average finalization requires exactly frame 1")
    if "source" in frame: raise ValueError("Average frame 1 must not have a singular source")
    if "finite_counts" in frame: del frame["finite_counts"]
    dataset = frame.create_dataset(
        "finite_counts", data=values, dtype="<u4", chunks=expected_chunks,
        compression="gzip", compression_opts=1, shuffle=True, fletcher32=False,
    )
    dataset.attrs.create(
        "average_scan_policy", np.bytes_(b"average_scan_v1"), dtype="S15",
    )
    dataset.attrs.create(
        "contributor_extent", np.uint32(evidence.contributor_extent),
        dtype="<u4",
    )
    dataset.attrs.create(
        "finite_counts_sha256", np.bytes_(evidence.sha256.encode("ascii")),
        dtype="S64",
    )
    dataset.attrs.create(
        "finite_counts_min", np.uint32(evidence.minimum), dtype="<u4",
    )
    dataset.attrs.create(
        "finite_counts_max", np.uint32(evidence.maximum), dtype="<u4",
    )
    dataset.attrs.create(
        "finite_counts_zero_count", np.uint64(evidence.zero_count), dtype="<u8",
    )
    return dataset
def _mode_or_default(mode, dimension: str) -> str:
    if mode is None:
        return DEFAULT_MODE_KEY
    return canonical_gi_mode_key(mode, dimension)


def _owned_result_modes(raw_modes, active_mode, active_result, dimension: str):
    """Bind result names only from the producer's canonical result mapping."""
    if not isinstance(raw_modes, Mapping):
        raise ValueError(f"{dimension} result modes require an exact mapping")
    modes: dict[str, object] = {}
    for raw_mode, result in raw_modes.items():
        mode = canonical_gi_mode_key(raw_mode, dimension, allow_default=False)
        if mode in modes:
            raise ValueError(f"duplicate canonical {dimension} result mode {mode!r}")
        modes[mode] = result

    selected = None if active_mode is None else canonical_gi_mode_key(
        active_mode,
        dimension,
    )
    if not modes:
        if selected not in (None, DEFAULT_MODE_KEY):
            raise ValueError(
                f"active {dimension} selector {selected!r} has no owned result"
            )
        return (
            {} if active_result is None else {DEFAULT_MODE_KEY: active_result},
            DEFAULT_MODE_KEY,
        )
    if selected is not None and selected not in modes:
        raise ValueError(
            f"active {dimension} selector {selected!r} has no owned result"
        )
    owners = [mode for mode, result in modes.items() if result is active_result]
    if active_result is not None and not owners:
        raise ValueError(f"active {dimension} result is absent from its mode mapping")
    if len(owners) > 1:
        raise ValueError(f"active {dimension} result has multiple mode owners")
    active = owners[0] if owners else (selected or next(iter(modes)))
    return modes, active


def frame_record_from_live_frame(
    frame,
    *,
    active_mode_1d: str | None = None,
    active_mode_2d: str | None = None,
    include_raw: bool = False,
    include_2d: bool = True,
    include_thumbnail: bool = True,
) -> FrameRecord:
    """Build the durable multi-mode :class:`FrameRecord` for a LiveFrame-like
    object.

    This is the shared GUI writer/display adapter for current output.  GI maps
    must already use canonical keys; the maps own result names.  An empty map
    collapses an unnamed result to ``default`` and a named selector cannot
    manufacture an otherwise-unowned result.
    """
    metadata_raw = dict(getattr(frame, "scan_info", None) or {})
    metadata_num = numeric_metadata(metadata_raw)
    incident_angle = None
    if getattr(frame, "gi", False):
        try:
            incident_angle = float(frame._get_incident_angle())
        except Exception:
            incident_angle = None
    result_1d = getattr(frame, "int_1d", None)
    result_2d = getattr(frame, "int_2d", None) if include_2d else None
    thumbnail = getattr(frame, "thumbnail", None) if include_thumbnail else None
    common = dict(
        metadata_raw=metadata_raw,
        metadata_numeric=metadata_num,
        incident_angle=incident_angle,
        source_path=getattr(frame, "source_file", None) or None,
        source_frame_index=getattr(frame, "source_frame_idx", None),
    )
    view = FrameView.from_results(
        label=getattr(frame, "idx", ""),
        result_1d=result_1d,
        result_2d=result_2d,
        raw=(getattr(frame, "map_raw", None) if include_raw else None),
        thumbnail=thumbnail,
        mask_baked=thumbnail is not None,
        **common,
    )

    gi_1d = getattr(frame, "gi_1d", None) or {}
    gi_2d = (getattr(frame, "gi_2d", None) or {}) if include_2d else {}
    modes_1d, active_1d = _owned_result_modes(
        gi_1d,
        active_mode_1d,
        result_1d,
        "1d",
    )
    modes_2d, active_2d = _owned_result_modes(
        gi_2d,
        active_mode_2d,
        result_2d,
        "2d",
    )
    results_1d = {
        m: FrameView.from_results(
            label=view.label,
            result_1d=r,
            thumbnail=thumbnail,
            mask_baked=thumbnail is not None,
            **common,
        )
        for m, r in modes_1d.items()
    }
    results_2d = {
        m: FrameView.from_results(
            label=view.label,
            result_2d=r,
            thumbnail=thumbnail,
            mask_baked=thumbnail is not None,
            **common,
        )
        for m, r in modes_2d.items()
    }
    return FrameRecord(
        label=view.label,
        results_1d=results_1d,
        results_2d=results_2d,
        active_mode_1d=active_1d if results_1d else DEFAULT_MODE_KEY,
        active_mode_2d=active_2d if results_2d else DEFAULT_MODE_KEY,
    )


def frame_record_from_reduction(
    frame,
    reduction,
    *,
    mode_1d: str | None = None,
    mode_2d: str | None = None,
) -> FrameRecord:
    """Build a single-completion record from a headless FrameReduction."""
    metadata_raw = dict(getattr(reduction, "metadata", None)
                        or getattr(frame, "metadata", None) or {})
    view = FrameView.from_results(
        label=int(getattr(reduction, "frame_index", getattr(frame, "index", -1))),
        result_1d=getattr(reduction, "result_1d", None),
        result_2d=getattr(reduction, "result_2d", None),
        metadata_raw=metadata_raw,
        metadata_numeric=getattr(frame, "metadata_numeric", None),
        incident_angle=getattr(getattr(frame, "geometry", None), "incident_angle", None),
        source_path=getattr(frame, "source_path", None),
        source_frame_index=getattr(frame, "source_frame_index", None),
    )
    return FrameRecord.from_view(
        view,
        mode_1d=_mode_or_default(
            mode_1d if mode_1d is not None else getattr(reduction, "mode_1d", None),
            "1d",
        ),
        mode_2d=_mode_or_default(
            mode_2d if mode_2d is not None else getattr(reduction, "mode_2d", None),
            "2d",
        ),
    )


def merge_frame_records(existing: FrameRecord, incoming: FrameRecord) -> FrameRecord:
    if existing.label != incoming.label:
        raise ValueError(
            f"cannot merge FrameRecords with labels {existing.label!r} and "
            f"{incoming.label!r}"
        )
    merged = existing
    for mode, view in incoming.results_1d.items():
        merged = merged.with_result_1d(
            mode, view, make_active=(mode == incoming.active_mode_1d)
        )
    for mode, view in incoming.results_2d.items():
        merged = merged.with_result_2d(
            mode, view, make_active=(mode == incoming.active_mode_2d)
        )
    return merged


# ---------------------------------------------------------------------------
# Thumbnails
# ---------------------------------------------------------------------------

def make_thumbnail_array(image, *, mask=None, mask_flat=None,
                         global_mask_flat=None, max_size: int = THUMBNAIL_MAX,
                         _owned: bool = False):
    """Downsample a 2D image to at most ``(max_size, max_size)``.

    Masked pixels become NaN *before* downsampling so the mask is baked into
    the preview — viewers need no full-resolution mask.  The legacy flat-index
    inputs remain copy-safe.  ``_owned=True`` is an internal fast path for one
    private, writable float32 scratch that may be consumed in place.  Returns
    float32, or ``None`` for invalid input.
    """
    if image is None:
        return None
    source = np.asarray(image)
    if source.ndim != 2:
        return None
    if type(_owned) is not bool:
        raise TypeError("owned thumbnail input flag must be an exact bool")
    if _owned and (
        not isinstance(image, np.ndarray)
        or source is not image
        or source.dtype != np.dtype(np.float32)
        or not source.flags.owndata
        or not source.flags.writeable
    ):
        raise ValueError(
            "owned thumbnail input must be a writable owning float32 ndarray"
        )
    if mask is not None and (mask_flat is not None or global_mask_flat is not None):
        raise ValueError("boolean and flat thumbnail masks are mutually exclusive")
    if _owned and (mask_flat is not None or global_mask_flat is not None):
        raise ValueError("owned thumbnail input requires a boolean mask")

    if mask is not None:
        bool_mask = np.asarray(mask)
        if bool_mask.dtype != np.dtype(bool) or bool_mask.shape != source.shape:
            raise ValueError(
                "boolean thumbnail mask must match the 2D image shape"
            )
        arr = source if _owned else np.array(source, dtype=np.float32, copy=True)
        arr[bool_mask] = np.nan
    else:
        all_mask = []
        if mask_flat is not None and len(mask_flat) > 0:
            all_mask.append(np.asarray(mask_flat, dtype=np.intp).ravel())
        if global_mask_flat is not None and len(global_mask_flat) > 0:
            all_mask.append(np.asarray(global_mask_flat, dtype=np.intp).ravel())
    if mask is None and all_mask:
        # Flat-index assignment is already idempotent, so sorting and
        # uniquing detector-sized masks only adds work to every frame.
        flat_mask = (
            all_mask[0] if len(all_mask) == 1 else np.concatenate(all_mask)
        )
        flat_mask = flat_mask[
            (flat_mask >= 0) & (flat_mask < source.size)
        ]
        arr = np.array(source, dtype=np.float32, copy=True)
        arr.ravel()[flat_mask] = np.nan
    elif mask is None:
        arr = source if _owned else np.asarray(source, dtype=np.float32)

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


@dataclass(frozen=True, slots=True)
class _PreparedThumbnail:
    """Owned immutable thumbnail bytes shared by write and verification."""

    array: np.ndarray
    lut: tuple[float, float, str]
    mask: np.ndarray | None


def _prepare_thumbnail(
    thumbnail,
    dtype: str = "uint8",
    *,
    thumbnail_mask=None,
) -> _PreparedThumbnail:
    source = np.asarray(thumbnail)
    array, raw_lut = quantize_thumbnail(source, dtype=dtype)
    array = np.asarray(array)
    if not array.flags.owndata:
        array = np.array(array, copy=True)
    array.setflags(write=False)
    invalid = (
        np.asarray(thumbnail_mask, dtype=bool)
        if thumbnail_mask is not None
        else ~np.isfinite(source)
    )
    if thumbnail_mask is not None and invalid.shape != source.shape:
        raise ValueError("thumbnail_mask shape must equal thumbnail shape")
    mask = None
    if invalid.any() or thumbnail_mask is not None:
        mask = np.array(invalid, dtype=bool, copy=True)
        mask.setflags(write=False)
    return _PreparedThumbnail(
        array=array,
        lut=(float(raw_lut[0]), float(raw_lut[1]), str(raw_lut[2])),
        mask=mask,
    )


def _write_prepared_thumbnail(
    frame_grp: h5py.Group,
    prepared: _PreparedThumbnail,
    *,
    mask_baked: bool,
) -> None:
    ds = frame_grp.create_dataset("thumbnail", data=prepared.array)
    for key, value in zip(THUMBNAIL_LUT_ATTRS, prepared.lut):
        ds.attrs[key] = value
    ds.attrs["mask_baked"] = bool(mask_baked)
    if prepared.mask is not None:
        frame_grp.create_dataset(
            "thumbnail_mask",
            data=prepared.mask,
            compression="gzip",
        )


# ---------------------------------------------------------------------------
# @source_base (N1 portability root)
# ---------------------------------------------------------------------------

def validate_source_base(entry_grp: h5py.Group, source_base) -> str | None:
    """Normalize and validate the project root without mutating the entry."""
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
                "paths are stored against the old root; start a NEW output "
                "file for the new Project Folder."
            )
    return base


def validate_artifact_family(entry_grp: h5py.Group, artifact_family) -> str | None:
    """Validate the persisted root family without mutating the entry.

    Mirrors :func:`validate_source_base`, including its LEGACY TOLERANCE: a
    record written before this attribute existed has none, and appending to it
    must keep working, so an ABSENT stored value is not a mismatch.  Only a
    stored value that DIFFERS raises.
    """
    if not artifact_family:
        return None
    family = str(artifact_family)
    existing = entry_grp.attrs.get(ARTIFACT_FAMILY_ATTR)
    if existing is not None:
        if isinstance(existing, bytes):
            existing = existing.decode("utf-8", errors="replace")
        if str(existing) != family:
            raise ValueError(
                f"cannot append to {os.fspath(entry_grp.file.filename)!r}: its "
                f"root family (@{ARTIFACT_FAMILY_ATTR}={str(existing)!r}) "
                f"differs from the current ({family!r}).  Every operation on "
                "this artifact publishes into that family's stable slots; "
                "start a NEW output file for a different family."
            )
    return family


def stamp_artifact_family(entry_grp: h5py.Group, artifact_family) -> str | None:
    """Stamp the root family on ``entry/@artifact_family_v1``.

    Written ONCE per file, at the same seam as ``@source_base`` -- never in the
    per-frame loop, so the accepted integration/writer floors are untouched.

    This is what makes the stable-slot policy's anti-chaining rule real. A later
    operation CONSUMES this value; without it, deriving a family from the
    artifact's own stem turns `sample_int2d.nexus` into
    `sample_int2d_average.nexus`.
    """
    family = validate_artifact_family(entry_grp, artifact_family)
    if family is None:
        return None
    try:
        entry_grp.attrs[ARTIFACT_FAMILY_ATTR] = family
    except Exception as exc:
        raise RuntimeError(
            f"failed to stamp @{ARTIFACT_FAMILY_ATTR}={family!r} on "
            f"{entry_grp.name!r}; a later operation would derive a chained "
            "family from this artifact's own stem"
        ) from exc
    return family


def stamp_source_base(entry_grp: h5py.Group, source_base) -> str | None:
    """Normalize + stamp the project root on ``entry/@source_base``.

    ONE scan-level root governs ALL frames' relative source paths: appending
    to a file written under a DIFFERENT root would silently rebase the
    earlier frames' pointers, so a mismatch raises rather than corrupting
    resolution.  Returns the normalized absolute base (native separators)
    for use with :func:`write_frame_source_ref`, or ``None`` when no base
    was given (absolute-path back-compat mode).
    """
    base = validate_source_base(entry_grp, source_base)
    if base is None:
        return None
    posix_base = Path(base).as_posix()
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


def _background_require(condition, message): return condition or (_ for _ in ()).throw(ValueError(message))
def _background_pair(descriptor_bytes, fingerprint) -> tuple[bytes, str]:
    _background_require(type(descriptor_bytes) is bytes and 0 < len(descriptor_bytes) <= _MAX_BACKGROUND_DESCRIPTOR_BYTES and type(fingerprint) is str and len(fingerprint) == 64 and fingerprint == fingerprint.lower() and all(value in "0123456789abcdef" for value in fingerprint), "background dependency pair is malformed")
    text = descriptor_bytes.decode("utf-8", errors="replace"); _background_require(text.encode() == descriptor_bytes, "background descriptor is not UTF-8"); value = json.loads(text); canonical = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    _background_require(type(value) is dict and canonical == descriptor_bytes and hashlib.sha256(descriptor_bytes).hexdigest() == fingerprint, "background descriptor fingerprint/canonical form differs")
    from xrd_tools.reduction.background import FrameBackgroundPlan, _MEMBER, _frame, _tag, _untag, compile_filter
    hex64 = lambda item: type(item) is str and len(item) == 64 and item == item.lower() and all(char in "0123456789abcdef" for char in item); path = lambda item: type(item) is str and bool(item) and not item.startswith("//") and len(item.encode()) <= 4096 and os.path.abspath(os.path.normpath(item)) == item; state = lambda item: type(item) is list and len(item) == 6 and all(type(part) is int for part in item) and item[2] & 0o170000 == 0o100000 and 0 <= item[3] <= 256 * 1024 ** 2
    plan = FrameBackgroundPlan.from_mapping(value.get("policy")); _background_require(plan.mode != "None", "background descriptor policy is inactive"); policy = plan.to_mapping(); mode = plan.mode; raw_fact = value.get("frame_fact")
    _background_require(type(raw_fact) is list and len(raw_fact) == 6 and type(raw_fact[4]) is list and type(raw_fact[5]) is list and all(type(row) is list and len(row) == 2 and type(row[1]) is list for row in raw_fact[5]), "background frame_fact schema is invalid")
    fact = (raw_fact[0], raw_fact[1], raw_fact[2], raw_fact[3], tuple(raw_fact[4]), tuple((row[0], tuple(row[1])) for row in raw_fact[5])); _frame(fact, persisted=True)
    expected_names = tuple(sorted(set(key for key in (policy["metadata_key"], policy["normalization_key"]) if key))); _background_require(tuple(row[0] for row in fact[5]) == expected_names, "background frame_fact policy projection differs")
    def items(raw): _background_require(type(raw) is list and all(type(row) is list and len(row) == 2 and type(row[1]) is list for row in raw), "background metadata_items schema is invalid"); return _frame((0, "/x", None, 0, (1, 1), tuple((row[0], tuple(row[1])) for row in raw)), persisted=True)[5]
    def decoded(raw, shape): _background_require(type(raw) is dict and set(raw) == {"shape", "dtype", "sha256"} and type(raw["shape"]) is list and all(type(part) is int for part in raw["shape"]) and tuple(raw["shape"]) == shape and hex64(raw["sha256"]) and type(raw["dtype"]) is str and len(raw["dtype"]) == 3 and raw["dtype"][0] in "<>=|" and raw["dtype"][1] in "iuf" and raw["dtype"][2] in "1248", "background decoded fact is malformed"); dtype = np.dtype(raw["dtype"]); _background_require(dtype.str == raw["dtype"] and 1 <= dtype.itemsize <= 8, "background decoded dtype is malformed")
    def metadata_source(raw): _background_require(raw is None or type(raw) is dict and set(raw) == {"locator", "state", "sha256"} and path(raw["locator"]) and state(raw["state"]) and hex64(raw["sha256"]), "background metadata source is malformed")
    positive = lambda tagged: type(number := _untag(tagged)) is float and np.isfinite(number) and number > 0 and _tag(number) == tagged
    keys = {"Single BG File": {"version", "mode", "policy", "frame_fact", "source", "metadata_items", "metadata_source", "decoded", "result_sha256"}, "Series Average": {"version", "mode", "policy", "frame_fact", "manifest", "selected", "normalization", "result_sha256"}, "BG Directory": {"version", "mode", "policy", "frame_fact", "manifest", "selected", "metadata_items", "metadata_source", "decoded", "result_sha256"}}; _background_require(value.get("mode") == mode and type(value.get("version")) is int and value["version"] == 1 and set(value) == keys[mode] and hex64(value["result_sha256"]), "background descriptor schema is invalid"); shape = fact[4]
    if mode == "Single BG File":
        source = value["source"]; _background_require(type(source) is dict and set(source) == {"locator", "state", "sha256", "hdf", "decoded"} and path(source["locator"]) and source["locator"] == policy["locator"] and state(source["state"]) and hex64(source["sha256"]), "background source fact is malformed"); decoded(source["decoded"], shape); hdf = source["hdf"]
        _background_require((hdf is None) == (policy["dataset_path"] is None) and (hdf is None or type(hdf) is dict and set(hdf) == {"dataset_path", "frame_index", "shape", "dtype"} and hdf["dataset_path"] == policy["dataset_path"] and type(hdf["frame_index"]) is int and hdf["frame_index"] == policy["frame_index"] and type(hdf["shape"]) is list and len(hdf["shape"]) in {2, 3} and all(type(part) is int and part > 0 for part in hdf["shape"]) and tuple(hdf["shape"][-2:]) == shape and (len(hdf["shape"]) == 2 and hdf["frame_index"] == 0 or len(hdf["shape"]) == 3 and 0 <= hdf["frame_index"] < hdf["shape"][0]) and type(hdf["dtype"]) is str and decoded({"shape": hdf["shape"][-2:], "dtype": hdf["dtype"], "sha256": "0" * 64}, shape) is None), "background HDF proof is malformed"); projected = items(value["metadata_items"]); norm = policy["normalization_key"]; _background_require(value["decoded"] == source["decoded"] and tuple(row[0] for row in projected) == tuple(key for key in (norm,) if key) and bool(projected) == (value["metadata_source"] is not None) and (norm is None or positive(dict(fact[5])[norm]) and positive(dict(projected)[norm])), "background Single projection differs"); metadata_source(value["metadata_source"])
    else:
        manifest = value["manifest"]; _background_require(len(descriptor_bytes) <= 16_384 and type(manifest) is dict and set(manifest) == {"version", "count", "receipt_bytes", "sha256"} and type(manifest["version"]) is int and manifest["version"] == 1 and type(manifest["count"]) is int and 1 <= manifest["count"] <= 10_000 and type(manifest["receipt_bytes"]) is int and manifest["count"] <= manifest["receipt_bytes"] <= min(64 * 1024 ** 2, manifest["count"] * 16_384) and hex64(manifest["sha256"]) and path(value["selected"]), "background manifest is invalid")
        if mode == "Series Average":
            normalization = value["normalization"]; _background_require(value["selected"] == policy["locator"] and type(normalization) is dict and set(normalization) == {"target", "denominator"} and all(type(normalization[key]) is list and len(normalization[key]) == 2 for key in normalization), "background Series normalization is malformed"); numbers = tuple(_untag(tuple(normalization[key])) for key in ("target", "denominator")); _background_require(all(_tag(number) == tuple(normalization[key]) and type(number) is float and np.isfinite(number) and number > 0 for key, number in zip(("target", "denominator"), numbers)) and tuple(normalization["target"]) == dict(fact[5]).get(policy["normalization_key"], _tag(1.0)) and (policy["normalization_key"] is not None or tuple(normalization["denominator"]) == _tag(1.0)), "background Series normalization is malformed")
        else:
            decoded(value["decoded"], shape); projected = items(value["metadata_items"]); metadata_source(value["metadata_source"]); allowed = expected_names if policy["match_rule"] == "Metadata Key" else tuple(key for key in (policy["normalization_key"],) if key); selected = Path(value["selected"]); target_match = _MEMBER.match(Path(fact[1]).stem); selected_match = _MEMBER.match(selected.stem); norm = policy["normalization_key"]; _background_require(selected.parent == Path(policy["locator"]) and value["selected"] != fact[1] and selected.suffix.casefold() in {".cbf", ".edf", ".img", ".mar3450", ".raw", ".tif", ".tiff"} and compile_filter(policy["filename_filter"])(selected.name) and tuple(row[0] for row in projected) == allowed and bool(projected) == (value["metadata_source"] is not None) and (norm is None or positive(dict(fact[5])[norm]) and positive(dict(projected)[norm])) and (policy["match_rule"] != "Metadata Key" or dict(projected)[policy["metadata_key"]] == dict(fact[5])[policy["metadata_key"]]) and (policy["match_rule"] != "Scan Root + Frame Number" or target_match and selected_match and int(selected_match.group(2)) == int(target_match.group(2)) and re.search(rf"(?:^|[_-]){re.escape(target_match.group(1))}(?:$|[_-])", selected_match.group(1), re.IGNORECASE)), "background Directory projection differs")
    return descriptor_bytes, fingerprint
def read_background_dependency(frame_group: h5py.Group) -> tuple[bytes, str] | None:
    if "background_dependency" not in frame_group: return None
    link = frame_group.get("background_dependency", getlink=True); _background_require(isinstance(link, h5py.HardLink), "background dependency link is not direct"); child = frame_group["background_dependency"]; nx_class = child.attrs.get("NX_class") if isinstance(child, h5py.Group) else None; nx_class = nx_class.decode("utf-8", errors="strict") if isinstance(nx_class, bytes) else nx_class
    _background_require(isinstance(child, h5py.Group) and nx_class == "NXcollection", "background dependency collection is malformed"); names = iter(child); observed = (next(names, None), next(names, None), next(names, None)); _background_require(set(observed[:2]) == {"descriptor_json", "fingerprint"} and observed[2] is None, "background dependency collection is malformed")
    def scalar(name, limit): link = child.get(name, getlink=True); dataset = child[name] if isinstance(link, h5py.HardLink) else None; encoding = h5py.check_string_dtype(dataset.dtype) if isinstance(dataset, h5py.Dataset) else None; _background_require(isinstance(dataset, h5py.Dataset) and dataset.shape == () and not dataset.is_virtual and dataset.id.get_create_plist().get_external_count() == 0 and dataset.dtype.kind == "S" and encoding is not None and encoding.encoding == "utf-8" and 0 < dataset.dtype.itemsize <= limit and dataset.id.get_storage_size() == dataset.dtype.itemsize, "background dependency scalar is malformed"); value = dataset[()]; _background_require(isinstance(value, bytes), "background dependency scalar is not bounded UTF-8"); return value.decode("utf-8", errors="strict"), dataset.dtype.itemsize
    values = (scalar("descriptor_json", _MAX_BACKGROUND_DESCRIPTOR_BYTES), scalar("fingerprint", 64)); _background_require(values[0][1] == len(values[0][0].encode()) and values[1][1] == 64, "background dependency scalar length differs")
    return _background_pair(values[0][0].encode("utf-8"), values[1][0])
def write_background_dependency(frame_group: h5py.Group, descriptor_bytes: bytes | None, fingerprint: str | None) -> None:
    if descriptor_bytes is None and fingerprint is None: _background_require("background_dependency" not in frame_group, "background dependency is unexpectedly present"); return
    pair = _background_pair(descriptor_bytes, fingerprint)
    if "background_dependency" in frame_group:
        _background_require(read_background_dependency(frame_group) == pair, "background dependency changed for an existing frame"); return
    child = _nxcollection(frame_group, "background_dependency")
    child.create_dataset("descriptor_json", data=pair[0], dtype=h5py.string_dtype(encoding="utf-8", length=len(pair[0]))); child.create_dataset("fingerprint", data=pair[1].encode(), dtype=h5py.string_dtype(encoding="utf-8", length=64))
def ensure_frames_container(entry_grp: h5py.Group) -> h5py.Group:
    """``entry/frames`` as an NXcollection (create-if-missing — re-creating
    would clobber per-frame groups from previous batches)."""
    return _nxcollection(entry_grp, "frames")


def write_thumbnail(
    frame_grp: h5py.Group,
    thumbnail,
    dtype: str = "uint8",
    *,
    mask_baked: bool = True,
    thumbnail_mask=None,
) -> None:
    """Quantize + store ``thumbnail`` with its inversion LUT attributes."""
    prepared = _prepare_thumbnail(
        thumbnail,
        dtype=dtype,
        thumbnail_mask=thumbnail_mask,
    )
    _write_prepared_thumbnail(frame_grp, prepared, mask_baked=mask_baked)


def write_frame_source_ref(
    frame_grp: h5py.Group,
    source_path,
    frame_index: int,
    *,
    source_base=None,
    source_snapshot=None,
) -> None:
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
    snapshot = dict(source_snapshot or {})
    attr_names = {
        "adapter_id": "adapter_id",
        "size": "file_size",
        "mtime_ns": "file_mtime_ns",
        "frame_count": "frame_count",
        "dataset_path": "dataset_path",
        "self_contained": "self_contained",
    }
    for key, attr_name in attr_names.items():
        value = snapshot.get(key)
        if value is None:
            continue
        try:
            if key in {"size", "mtime_ns", "frame_count"}:
                value = int(value)
            elif key == "self_contained":
                value = bool(value)
            else:
                value = str(value)
            sub.attrs[attr_name] = value
        except (TypeError, ValueError, OverflowError):
            logger.debug(
                "invalid source snapshot field %s=%r for %s",
                key,
                value,
                source_path,
            )


def write_frame_record(frames_grp: h5py.Group, frame_key: str, *,
                       thumbnail=None, thumbnail_dtype: str = "uint8",
                       thumbnail_mask_baked: bool = True,
                       mask_baked: bool = True, thumbnail_mask=None,
                       source_path=None, source_frame_index: int = 0,
                       timestamp=None, source_base=None,
                       source_snapshot=None, background_dependency_bytes=None,
                       background_dependency_fingerprint=None) -> h5py.Group:
    """Write one complete per-frame record group (idempotent per key).

    Per the v2 schema, per-frame groups carry *only* metadata + thumbnail —
    never the full raw image (an early writer dumped 18 MB per Eiger frame
    here; don't bring that back).
    """
    fg = _nxcollection(frames_grp, frame_key)
    fg.attrs["mask_baked"] = bool(mask_baked)
    if thumbnail is not None and "thumbnail" not in fg:
        write_thumbnail(
            fg,
            thumbnail,
            dtype=thumbnail_dtype,
            mask_baked=thumbnail_mask_baked,
            thumbnail_mask=thumbnail_mask,
        )
    if source_path and "source" not in fg:
        write_frame_source_ref(fg, source_path, source_frame_index,
                               source_base=source_base,
                               source_snapshot=source_snapshot)
    if timestamp is not None and "timestamp" not in fg:
        fg["timestamp"] = str(timestamp)
    write_background_dependency(fg, background_dependency_bytes, background_dependency_fingerprint)
    return fg


def replace_frame_record(frames_grp: h5py.Group, frame_key: str, *,
                         thumbnail=None, thumbnail_dtype: str = "uint8",
                         thumbnail_mask_baked: bool = True,
                         mask_baked: bool = True, thumbnail_mask=None,
                         source_path=None, source_frame_index: int = 0,
                         timestamp=None, source_base=None,
                         source_snapshot=None, background_dependency_bytes=None,
                         background_dependency_fingerprint=None,
                         _prepared_thumbnail: _PreparedThumbnail | None = None,
                         ) -> h5py.Group:
    """Replace one direct per-frame record through an atomic sibling stage."""
    if not frame_key or "/" in frame_key:
        raise ValueError(
            f"frame_key must be one direct child name; got {frame_key!r}"
        )
    token = uuid.uuid4().hex
    staged = f".{frame_key}.{token}.pending"
    backup = f".{frame_key}.{token}.prior"
    installed = False
    prior_moved = False
    try:
        frame = write_frame_record(
            frames_grp,
            staged,
            thumbnail=(thumbnail if _prepared_thumbnail is None else None),
            thumbnail_dtype=thumbnail_dtype,
            thumbnail_mask_baked=thumbnail_mask_baked,
            mask_baked=mask_baked,
            thumbnail_mask=thumbnail_mask,
            source_path=source_path,
            source_frame_index=source_frame_index,
            timestamp=timestamp,
            source_base=source_base,
            source_snapshot=source_snapshot,
            background_dependency_bytes=background_dependency_bytes,
            background_dependency_fingerprint=background_dependency_fingerprint,
        )
        if _prepared_thumbnail is not None:
            if _prepared_thumbnail.lut[2] != thumbnail_dtype:
                raise ValueError("prepared thumbnail dtype differs")
            _write_prepared_thumbnail(
                frame,
                _prepared_thumbnail,
                mask_baked=thumbnail_mask_baked,
            )
        if frame_key in frames_grp:
            frames_grp.move(frame_key, backup)
            prior_moved = True
        frames_grp.move(staged, frame_key)
        installed = True
        if prior_moved:
            del frames_grp[backup]
        return frames_grp[frame_key]
    except BaseException:
        if (
            installed
            and frame_key in frames_grp
            and prior_moved
            and backup in frames_grp
        ):
            del frames_grp[frame_key]
        if prior_moved and backup in frames_grp and frame_key not in frames_grp:
            frames_grp.move(backup, frame_key)
        if staged in frames_grp:
            del frames_grp[staged]
        raise


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
    ``.nexus``).  ``records`` is an iterable of mappings with ``frame_index`` and
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
