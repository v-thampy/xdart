"""Bounded source records and companion coordinates for native Stitch artifacts.

The frame writer, thumbnail codec, raw reader and Project path resolver remain
the shared processed-scan owners. This module only admits their finite graph.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np

_MAX_FRAMES = 100_000
_MAX_MANIFEST_BYTES = 64 << 20
_MAX_TEXT = 4096
_SCREEN_KEYS = frozenset({"screen", "excluded"})
_SCREEN_STATUSES = frozenset({"ok", "empty", "partial", "unscreened"})
_SNAPSHOT_ATTRS = {
    "adapter_id": "adapter_id", "size": "file_size", "mtime_ns": "file_mtime_ns",
    "frame_count": "frame_count", "dataset_path": "dataset_path",
    "self_contained": "self_contained", "raw_shape": "raw_shape",
    "raw_dtype": "raw_dtype", "raw_header_bytes": "raw_header_bytes",
    "rotation": "rotation",
}


def _text(value, name):
    if type(value) is not str or not value or "\x00" in value or len(value.encode()) > _MAX_TEXT:
        raise ValueError(f"invalid contributing frame {name}")
    return value


def _index(value, name):
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError(f"invalid contributing frame {name}")
    return value


def _member(record, source_base):
    from xrd_tools.io.nexus_record import frame_record_key
    from xrd_tools.io.read import relative_source_path
    frame = _index(record["frame_index"], "frame index")
    scan = record.get("scan_label")
    if scan is not None:
        scan = _text(str(scan), "scan label")
        if "/" in scan or scan in {".", ".."}:
            raise ValueError("contributing scan label must be one path component")
    identity = dict(record["identity"])
    if set(identity) != {"source_id", "scan_id", "frame_index", "contribution_index"}:
        raise ValueError("contributing frame identity fields differ")
    for key in ("source_id", "scan_id"):
        _text(identity[key], key)
    if _index(identity["frame_index"], "original frame") != frame:
        raise ValueError("contributing frame original identity differs")
    _index(identity["contribution_index"], "contribution index")
    source_path = _text(str(record["source_path"]), "source path")
    snapshot = dict(record.get("source_snapshot") or {})
    if set(snapshot) - set(_SNAPSHOT_ATTRS):
        raise ValueError("unknown contributing frame source snapshot")
    for key, value in tuple(snapshot.items()):
        if key in {"adapter_id", "dataset_path", "raw_dtype"}:
            _text(value, key)
        elif key == "self_contained":
            if type(value) is not bool:
                raise ValueError("invalid self-contained source flag")
        elif key == "raw_shape":
            if not isinstance(value, (list, tuple)) or len(value) != 2 or any(type(n) is not int or n < 1 for n in value):
                raise ValueError("invalid raw detector shape")
            snapshot[key] = list(value)
        else:
            _index(value, key)
    if "raw_dtype" in snapshot:
        dtype = np.dtype(snapshot["raw_dtype"])
        if dtype.kind not in "iuf" or dtype.itemsize not in {1, 2, 4, 8}:
            raise ValueError("invalid raw detector dtype")
    if snapshot.get("rotation", 0) not in {0, 90, 180, 270}:
        raise ValueError("invalid raw detector rotation")
    return {"key": frame_record_key(scan, frame), "identity": identity,
            "source_path": relative_source_path(source_path, source_base),
            "source_frame_index": _index(record.get("source_frame_index", 0), "source frame index"),
            "source_snapshot": snapshot}


def analysis_contributing_frame_members(records, *, source_base):
    """Project source identities at Preview without loading any detector pixels."""
    members = [_member(record, source_base) for record in records]
    if not 1 <= len(members) <= _MAX_FRAMES:
        raise ValueError("contributing frame count is outside the bound")
    if len({m["key"] for m in members}) != len(members) or len({m["identity"]["contribution_index"] for m in members}) != len(members):
        raise ValueError("contributing frame identities collide")
    return members


def _thumbnail_manifest(array, lut, mask, mask_baked):
    if array.ndim != 2 or min(array.shape) < 1 or max(array.shape) > 256 or array.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        raise ValueError("thumbnail must use the shared bounded representation")
    if not all(math.isfinite(float(n)) for n in lut[:2]) or lut[1] <= lut[0] or lut[2] != array.dtype.name:
        raise ValueError("thumbnail lookup table is invalid")
    return {"shape": list(array.shape), "dtype": array.dtype.name,
            "vmin": float(lut[0]), "vmax": float(lut[1]), "mask_baked": bool(mask_baked),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            "mask_sha256": None if mask is None else hashlib.sha256(np.asarray(mask, dtype=bool).tobytes()).hexdigest()}


def _screen_fields(value):
    """The dropped-frame check's verdict for one frame; empty when it was not recorded."""
    present = _SCREEN_KEYS & set(value)
    if not present:
        return {}
    screen, excluded = value.get("screen"), value.get("excluded")
    if (present != _SCREEN_KEYS or screen not in _SCREEN_STATUSES or type(excluded) is not bool
            or (excluded and screen not in {"empty", "partial"})):
        raise ValueError("contributing frame screen verdict is invalid")
    return {"screen": screen, "excluded": excluded}


def analysis_contributing_frames_manifest(records, *, source_base):
    """Bind shared encoded thumbnails captured during the qualified science read."""
    from xrd_tools.io.nexus_record import _prepare_thumbnail
    records = tuple(records)
    members = analysis_contributing_frame_members(records, source_base=source_base)
    result = []
    for record, member in zip(records, members, strict=True):
        prepared = _prepare_thumbnail(record["thumbnail"], dtype=record.get("thumbnail_dtype", "uint8"), thumbnail_mask=record.get("thumbnail_mask"))
        result.append({**member, "mask_baked": bool(record.get("mask_baked", True)),
                       "thumbnail": _thumbnail_manifest(prepared.array, prepared.lut, prepared.mask, record.get("thumbnail_mask_baked", True)),
                       **_screen_fields(record)})
    return {"source_base": str(Path(source_base).expanduser().absolute()), "records": result}


def canonical_frame_manifest(value):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    if len(text.encode()) > _MAX_MANIFEST_BYTES:
        raise ValueError("contributing frame manifest exceeds its bound")
    return text


def analysis_contributing_frame_binding(members):
    """Bind the ordered, canonical source-member projection without duplicating it."""
    if type(members) is not list or not 1 <= len(members) <= _MAX_FRAMES:
        raise ValueError("contributing frame count is outside the bound")
    return {"version": 1, "count": len(members),
            "sha256": hashlib.sha256(canonical_frame_manifest(members).encode("utf-8")).hexdigest()}


def analysis_contributing_frame_binding_matches(binding, members):
    """Accept the original exact list or the strict versioned compact binding."""
    if type(binding) is list:
        return binding == members
    return (type(binding) is dict and set(binding) == {"version", "count", "sha256"}
            and type(binding["version"]) is int and binding["version"] == 1
            and type(binding["count"]) is int and 1 <= binding["count"] <= _MAX_FRAMES
            and type(binding["sha256"]) is str
            and binding == analysis_contributing_frame_binding(members))


def native_stitch_companion(radial, unit, wavelength_A):
    from xrd_tools.core.containers import convert_radial_axis
    if type(wavelength_A) not in {float, int} or not math.isfinite(wavelength_A) or wavelength_A <= 0:
        raise ValueError("Stitch wavelength must be positive finite angstroms")
    if unit not in {"q_A^-1", "2th_deg"}:
        raise ValueError("native Stitch axis must be Q or two-theta")
    values = np.asarray(radial)
    limit = 4 * np.pi / wavelength_A if unit == "q_A^-1" else 180.0
    if np.any(values < 0) or np.any(values > limit):
        raise ValueError("Stitch radial axis is outside the wavelength conversion domain")
    target = "2th_deg" if unit == "q_A^-1" else "q_A^-1"
    converted = np.asarray(convert_radial_axis(values, unit, target, wavelength_A), dtype="<f4")
    if not np.isfinite(converted).all() or (len(converted) > 1 and not np.all(np.diff(converted.astype(float)) > 0)):
        raise ValueError("converted Stitch axis is not finite and increasing")
    converted.setflags(write=False)
    return target, converted


def inspect_contributing_frames(entry, provenance):
    """Admit the exact contributing-frames graph of ``entry`` (a stitched record's ``stitch`` group, or
    an older artifact's entry) against the frozen selected members in ``provenance``, without following
    raw source paths. Returns the manifest JSON."""
    from xrd_tools.io.analysis_artifact import (
        _direct, _bounded_attr_text, _require_bounded_chunk, AnalysisArtifactInvalid,
    )
    def fail(message):
        raise AnalysisArtifactInvalid(message)
    def text(node, cap=_MAX_TEXT):
        if node.shape != () or node.dtype.kind != "S" or not 0 < node.dtype.itemsize <= cap or len(node.attrs) or node.id.get_storage_size() != node.dtype.itemsize:
            fail("contributing frame text is not bounded fixed storage")
        return bytes(node[()]).decode("utf-8", errors="strict")
    def number(owner, name, *, shape=(), kinds="biuf"):
        attr = owner.attrs.get_id(name)
        if attr.shape != shape or attr.dtype.kind not in kinds or attr.dtype.itemsize > 8:
            fail("contributing frame numeric attribute exceeds its bound")
        return np.asarray(owner.attrs[name]).tolist()
    manifest_ds = _direct(entry, "contributing_frames_json", h5py.Dataset)
    raw = text(manifest_ds, _MAX_MANIFEST_BYTES)
    manifest = json.loads(raw)
    if canonical_frame_manifest(manifest) != raw or set(manifest) != {"source_base", "records"}:
        fail("contributing frame manifest differs")
    rows = manifest["records"]
    if type(rows) is not list or not 1 <= len(rows) <= _MAX_FRAMES:
        fail("contributing frame count exceeds bound")
    if _bounded_attr_text(entry, "source_base") != manifest["source_base"]:
        fail("contributing frame Project binding differs")
    base = _text(manifest["source_base"], "Project root")
    if str(Path(base).expanduser().absolute()) != base:
        fail("contributing frame Project root is not absolute")
    members = [{k: v for k, v in row.items() if k not in {"thumbnail", "mask_baked", *_SCREEN_KEYS}} for row in rows]
    screened = {bool(_SCREEN_KEYS & set(row)) for row in rows}
    if len(screened) != 1:
        fail("contributing frame screen verdicts are partial")
    originals = []
    for row in rows:
        if set(row) - _SCREEN_KEYS != {"key", "identity", "source_path", "source_frame_index", "source_snapshot", "mask_baked", "thumbnail"}:
            fail("contributing manifest fields differ")
        try:
            _screen_fields(row)
        except ValueError:
            fail("contributing frame screen verdict is invalid")
        parts = row["key"].split("/")
        if len(parts) not in {1, 2} or (len(parts) == 2 and not parts[0].startswith("scan_")):
            fail("contributing record address is invalid")
        originals.append(dict(frame_index=row["identity"]["frame_index"],
            scan_label=parts[0][5:] if len(parts) == 2 else None,
            identity=row["identity"], source_path=str(Path(base) / row["source_path"]),
            source_frame_index=row["source_frame_index"], source_snapshot=row["source_snapshot"]))
    if analysis_contributing_frame_members(originals, source_base=base) != members:
        fail("contributing source identities are not canonical")
    if not analysis_contributing_frame_binding_matches(provenance.get("frame_members"), members):
        fail("contributing frames differ from the frozen selected members")
    frames = _direct(entry, "frames", h5py.Group)
    expected_parents = {row["key"].split("/")[0] for row in rows}
    if set(frames) != expected_parents:
        fail("contributing frame membership differs")
    def collection(group, extra=()):
        if set(group.attrs) != {"NX_class", *extra} or _bounded_attr_text(group, "NX_class") != "NXcollection":
            fail("contributing frame collection attributes differ")
    collection(frames)
    for parent in expected_parents:
        if parent.startswith("scan_"):
            subgroup = _direct(frames, parent, h5py.Group)
            collection(subgroup)
            if set(subgroup) != {r["key"].split("/")[1] for r in rows if r["key"].startswith(parent + "/")}:
                fail("contributing scan membership differs")
    for row in rows:
        parts = row["key"].split("/")
        if len(parts) not in {1, 2}:
            fail("contributing frame path is invalid")
        frame = frames
        for part in parts:
            frame = _direct(frame, part, h5py.Group)
        collection(frame, ("mask_baked",))
        if number(frame, "mask_baked", kinds="b") != row["mask_baked"]:
            fail("contributing frame mask flag differs")
        expected = {"source", "identity_json", "thumbnail"}
        if row["thumbnail"]["mask_sha256"] is not None:
            expected.add("thumbnail_mask")
        if set(frame) != expected or json.loads(text(_direct(frame, "identity_json", h5py.Dataset))) != row["identity"]:
            fail("contributing frame identity or graph differs")
        source = _direct(frame, "source", h5py.Group)
        collection(source, tuple(_SNAPSHOT_ATTRS[k] for k in row["source_snapshot"]))
        if set(source) != {"path", "frame_index"} or text(_direct(source, "path", h5py.Dataset)) != row["source_path"]:
            fail("contributing source locator differs")
        index = _direct(source, "frame_index", h5py.Dataset)
        if index.shape != () or index.dtype.kind not in "iu" or index.dtype.itemsize > 8 or len(index.attrs) or int(index[()]) != row["source_frame_index"]:
            fail("contributing source frame index differs")
        for key, expected_value in row["source_snapshot"].items():
            attr = _SNAPSHOT_ATTRS[key]
            if type(expected_value) is str:
                actual = _bounded_attr_text(source, attr)
            else:
                actual = number(source, attr, shape=(2,) if key == "raw_shape" else (), kinds="biu")
            if actual != expected_value:
                fail("contributing source descriptor differs")
        thumb = _direct(frame, "thumbnail", h5py.Dataset)
        if thumb.ndim != 2 or min(thumb.shape) < 1 or max(thumb.shape) > 256 or thumb.dtype not in (np.dtype("uint8"), np.dtype("uint16")) or set(thumb.attrs) != {"vmin", "vmax", "dtype", "mask_baked"}:
            fail("contributing thumbnail exceeds its bounded schema")
        _require_bounded_chunk(thumb)
        mask = None
        if "thumbnail_mask" in frame:
            mask_ds = _direct(frame, "thumbnail_mask", h5py.Dataset)
            if mask_ds.shape != thumb.shape or mask_ds.dtype != np.dtype(bool) or len(mask_ds.attrs):
                fail("contributing thumbnail mask differs")
            _require_bounded_chunk(mask_ds)
            mask = mask_ds[()]
        observed = _thumbnail_manifest(thumb[()], (number(thumb, "vmin"), number(thumb, "vmax"), _bounded_attr_text(thumb, "dtype")), mask, number(thumb, "mask_baked", kinds="b"))
        if observed != row["thumbnail"]:
            fail("contributing thumbnail differs from its stored projection")
    return raw
