"""Saved angular support, format v1: the one codec for a native 1-D profile's label-keyed support table.

The four datasets (:data:`xrd_tools.io.schema.ANGULAR_SUPPORT_DATASETS`) are direct, hard-linked children of the
profile's integrated_1d mode group (Int top level; Stitch label 1):

``angular_support_deg``
    float32 (K, n): one row per distinct saved profile, the supported degrees of each native bin, NaN throughout for
    a profile that carries a reason instead. Attributes: ``ssrl_angular_support_version`` = 1,
    ``ssrl_angular_support_metric``, ``units`` = "deg" and ``ssrl_angular_support_axis_sha256`` (the group's float32
    ``axis_1`` bytes the bins describe).
``angular_support_meta``
    UTF-8 JSON (K,), fixed-length 4096 bytes (NUL-padded), one row's facts each (:meth:`AngularSupportProfile.metadata`).
``angular_support_rows``
    int64 (M, 2): [frame label, table row].
``angular_support_row_digest``
    uint8 (M, 16): BLAKE2b-128 of that label's stored float32 intensity row.

An entry is bound to its label and to the content of its intensity row, never to a position: rows moved by surgery
keep their support, a row rewritten by a path that does not maintain support reads stale, and a removed label's entry
is ignored. A group without the datasets has no saved support; a group with any of them malformed raises
:class:`AngularSupportFormatError`.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable, Mapping

import numpy as np

from xrd_tools.core.angular_support import (
    SUPPORT_CONVENTIONS,
    SUPPORT_METRIC,
    AngularSupportProfile,
    SupportOrigin,
    SupportReason,
)
from xrd_tools.io.schema import ANGULAR_SUPPORT_DATASETS, ANGULAR_SUPPORT_VERSION_ATTR

__all__ = [
    "AngularSupportFormatError",
    "FORMAT_VERSION",
    "METRIC_ATTR",
    "AXIS_SHA256_ATTR",
    "MAX_META_BYTES",
    "axis_sha256",
    "has_angular_support",
    "intensity_digest",
    "read_angular_support",
    "remove_angular_support",
    "SupportTableReader",
    "SupportTableWriter",
    "write_angular_support",
]

FORMAT_VERSION = 1
METRIC_ATTR = "ssrl_angular_support_metric"
AXIS_SHA256_ATTR = "ssrl_angular_support_axis_sha256"
#: Largest UTF-8 JSON a table row's facts may take.
MAX_META_BYTES = 4096
_DEG, _META, _ROWS, _DIGEST = ANGULAR_SUPPORT_DATASETS
#: Facts kept when a profile's provenance would push its JSON past MAX_META_BYTES.
_ESSENTIAL_PROVENANCE = ("pyfai_version", "wavelength_m", "oversample", "unique_poses", "poses_sha256", "backend",
                         "included_contributors", "mask_sha256")


class AngularSupportFormatError(ValueError):
    """Saved angular support is present but malformed (never legacy absence)."""


def intensity_digest(row: Any) -> bytes:
    """BLAKE2b-128 of an intensity row's canonical float32 bytes (what the writer stores)."""
    values = np.ascontiguousarray(np.asarray(row), dtype=np.float32)
    return hashlib.blake2b(values.tobytes(), digest_size=16).digest()


def axis_sha256(axis: Any) -> str:
    """SHA-256 of a native 1-D axis's canonical float32 bytes."""
    return hashlib.sha256(np.ascontiguousarray(np.asarray(axis), dtype=np.float32).tobytes()).hexdigest()


def has_angular_support(group: Any) -> bool:
    """Whether any of the four support names exists in ``group`` (present means it must be well formed)."""
    return any(name in group for name in ANGULAR_SUPPORT_DATASETS)


def _meta_text(profile: AngularSupportProfile) -> str:
    facts = profile.metadata()
    text = json.dumps(facts, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(text.encode("utf-8")) > MAX_META_BYTES:
        provenance = facts.get("provenance") or {}
        facts["provenance"] = {key: provenance[key] for key in _ESSENTIAL_PROVENANCE if key in provenance}
        facts["provenance"]["provenance_trimmed"] = True
        text = json.dumps(facts, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(text.encode("utf-8")) > MAX_META_BYTES:
        raise ValueError("angular support facts exceed the saved format's 4096-byte row limit")
    return text


def _row_values(profile: AngularSupportProfile, points: int) -> np.ndarray:
    if profile.supported_deg is None:
        return np.full(points, np.nan, np.float32)
    values = np.asarray(profile.supported_deg, dtype=np.float32)
    if values.shape != (points,):
        raise ValueError("angular support has a different bin count from the saved axis")
    return values


def _hard_dataset(group: Any, name: str):
    import h5py
    link = group.get(name, getlink=True)
    if link is None:
        return None
    if type(link) is not h5py.HardLink:
        raise AngularSupportFormatError(f"{group.name}/{name} is a link, not a dataset")
    node = group[name]
    if not isinstance(node, h5py.Dataset) or node.is_virtual or node.external:
        raise AngularSupportFormatError(f"{group.name}/{name} is not a local dataset")
    return node


def _datasets(group: Any) -> tuple | None:
    """The four datasets, validated for kind, rank, dtype and version; None when all are absent."""
    import h5py
    present = [name in group for name in ANGULAR_SUPPORT_DATASETS]
    if not any(present):
        return None
    if not all(present):
        raise AngularSupportFormatError(f"{group.name} holds only part of the angular support table")
    deg, meta, rows, digest = (_hard_dataset(group, name) for name in ANGULAR_SUPPORT_DATASETS)
    if deg.dtype != np.dtype("<f4") or deg.ndim != 2:
        raise AngularSupportFormatError("angular_support_deg must be a float32 (K, n) table")
    encoding = h5py.check_string_dtype(meta.dtype)
    if encoding is None or meta.ndim != 1 or meta.shape[0] != deg.shape[0]:
        raise AngularSupportFormatError("angular_support_meta must hold one string per table row")
    if rows.dtype != np.dtype("<i8") or rows.ndim != 2 or rows.shape[1] != 2:
        raise AngularSupportFormatError("angular_support_rows must be int64 (M, 2)")
    if digest.dtype != np.dtype("u1") or digest.ndim != 2 or digest.shape != (rows.shape[0], 16):
        raise AngularSupportFormatError("angular_support_row_digest must be uint8 (M, 16), one per entry")
    version = deg.attrs.get(ANGULAR_SUPPORT_VERSION_ATTR)
    if isinstance(version, (bool, np.bool_)) or not isinstance(version, (int, np.integer)) \
            or int(version) != FORMAT_VERSION:
        raise AngularSupportFormatError(f"unknown angular support format version {version!r}")
    if _attr_text(deg, METRIC_ATTR) != SUPPORT_METRIC or _attr_text(deg, "units") != "deg":
        raise AngularSupportFormatError("angular support metric or units are not this format's")
    if len(_attr_text(deg, AXIS_SHA256_ATTR)) != 64:
        raise AngularSupportFormatError("angular support has no axis binding")
    return deg, meta, rows, digest


def _attr_text(node: Any, name: str) -> str:
    value = node.attrs.get(name)
    if isinstance(value, bytes):
        value = value.decode("utf-8", "strict")
    if not isinstance(value, str):
        raise AngularSupportFormatError(f"{node.name} attribute {name} must be text")
    return value


def _decode_meta(raw: Any) -> Mapping[str, Any]:
    text = raw.decode("utf-8", "strict") if isinstance(raw, bytes) else raw
    if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_META_BYTES:
        raise AngularSupportFormatError("an angular support row's facts exceed 4096 bytes")
    try:
        facts = json.loads(text)
    except ValueError as error:
        raise AngularSupportFormatError("an angular support row's facts are not JSON") from error
    if not isinstance(facts, dict) or facts.get("status") not in ("available", "unavailable") \
            or facts.get("metric") != SUPPORT_METRIC:
        raise AngularSupportFormatError("an angular support row's facts are incomplete")
    return facts


def _pair(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if not (isinstance(value, list) and len(value) == 2
            and all(type(item) in (int, float) and math.isfinite(item) for item in value)):
        raise AngularSupportFormatError("an angular support interval is malformed")
    return float(value[0]), float(value[1])


def _profile(facts: Mapping[str, Any], values: np.ndarray, points: int) -> AngularSupportProfile:
    """The saved profile of one table row (typed origin ``saved``)."""
    try:
        common = dict(
            origin=SupportOrigin.SAVED,
            requested_window_deg=_pair(facts.get("requested_window_deg")),
            effective_window_deg=tuple(_pair(item) for item in facts.get("effective_window_deg") or ()),
            convention=str(facts.get("convention") or ""),
            radial_unit=str(facts.get("radial_unit") or ""),
            bin_edges=_pair(facts.get("bin_edges")),
            detail=str(facts.get("detail") or ""),
            provenance=dict(facts.get("provenance") or {}),
        )
        if facts["status"] == "unavailable":
            if np.any(np.isfinite(values)):
                raise AngularSupportFormatError("an unavailable angular support row holds values")
            window = facts.get("window_deg")
            return AngularSupportProfile.unavailable(
                SupportReason(facts.get("reason")), window_deg=math.nan if window is None else float(window),
                **common)
        if facts.get("reason") is not None or common["convention"] not in SUPPORT_CONVENTIONS \
                or values.shape != (points,) or not np.all(np.isfinite(values)):
            raise AngularSupportFormatError("an available angular support row is incomplete")
        return AngularSupportProfile(supported_deg=values.astype(np.float64), window_deg=float(facts["window_deg"]),
                                     **common)
    except AngularSupportFormatError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise AngularSupportFormatError(f"an angular support row cannot be read: {error}") from error


class SupportTableReader:
    """One group's validated support table, decoded lazily per label (FrameViewReader, Browse, Stitch readers).

    Construction validates the four datasets' links, kinds, dtypes, ranks, version and size bounds before any data
    is read, and raises :class:`AngularSupportFormatError` when present but malformed. ``present`` is False when
    the group has no table (every label then has ``no_saved_support``).
    """

    def __init__(self, group: Any, *, frame_count: int | None = None) -> None:
        datasets = _datasets(group)
        self.present = datasets is not None
        self._cache: dict[int, AngularSupportProfile] = {}
        if datasets is None:
            return
        deg, meta, rows_node, digest_node = datasets
        axis = group.get("axis_1")
        if axis is None or getattr(axis, "ndim", None) != 1 or axis.shape[0] != deg.shape[1]:
            raise AngularSupportFormatError("angular support does not match the group's axis")
        if frame_count is None:
            index = group.get("frame_index")
            frame_count = 0 if index is None else int(index.shape[0])
        if rows_node.shape[0] > 4 * int(frame_count) + 64 or deg.shape[0] > 2 * rows_node.shape[0] + 64:
            raise AngularSupportFormatError("the angular support table is larger than its output")
        entries = np.asarray(rows_node[()], dtype=np.int64)
        digests = np.asarray(digest_node[()], dtype=np.uint8)
        if entries.size and (np.any(entries[:, 1] < 0) or np.any(entries[:, 1] >= deg.shape[0])):
            raise AngularSupportFormatError("an angular support entry names a missing table row")
        if len(set(entries[:, 0].tolist())) != entries.shape[0]:
            raise AngularSupportFormatError("a frame label has two angular support entries")
        self._deg, self._meta, self._points = deg, meta, int(deg.shape[1])
        self._stale_axis = _attr_text(deg, AXIS_SHA256_ATTR) != axis_sha256(axis[()])
        self._entries = {int(label): (int(row), digests[number].tobytes())
                         for number, (label, row) in enumerate(entries)}

    def table_profile(self, row: int) -> AngularSupportProfile:
        """The saved profile of one table row (strict)."""
        if row not in self._cache:
            self._cache[row] = _profile(_decode_meta(self._meta[row]), np.asarray(self._deg[row], np.float32),
                                        self._points)
        return self._cache[row]

    def profile(self, label: int, intensity_row: Any) -> AngularSupportProfile:
        """``label``'s saved support given its stored intensity row, or the reason it has none."""
        entry = self._entries.get(int(label)) if self.present else None
        if entry is None:
            return AngularSupportProfile.unavailable(SupportReason.NO_SAVED_SUPPORT, origin=SupportOrigin.SAVED)
        row, saved_digest = entry
        saved = self.table_profile(row)
        if self._stale_axis or intensity_row is None or intensity_digest(intensity_row) != saved_digest:
            return AngularSupportProfile.unavailable(SupportReason.STALE_SAVED_SUPPORT, origin=SupportOrigin.SAVED)
        return saved


def read_angular_support(group: Any, *, labels: Iterable[int], intensity_rows: Any,
                         frame_index: Any = None) -> dict[int, AngularSupportProfile]:
    """Each requested label's saved support, or the reason it has none (strict: malformed raises).

    ``intensity_rows`` are the group's stored intensity rows in storage order (any float dtype; compared through
    their float32 bytes); ``frame_index`` defaults to the group's own. A label without an entry, or a group without
    the table, has ``no_saved_support``; an entry whose intensity row changed reads ``stale_saved_support``.
    """
    stored = np.asarray(group["frame_index"][()] if frame_index is None else frame_index, dtype=np.int64)
    reader = SupportTableReader(group, frame_count=int(stored.size))
    position = {int(label): index for index, label in enumerate(stored.tolist())}
    out: dict[int, AngularSupportProfile] = {}
    for label in (int(item) for item in labels):
        row = intensity_rows[position[label]] if label in position else None
        out[label] = (reader.profile(label, row) if row is not None else
                      AngularSupportProfile.unavailable(SupportReason.NO_SAVED_SUPPORT, origin=SupportOrigin.SAVED))
    return out


class SupportTableWriter:
    """Incremental writer of one group's support table, owned by one record writer for its whole life.

    The large table only grows: a profile identical to a stored one reuses its row, a new one is appended, and rows
    no longer referenced are compacted away once they outnumber the live ones. Entries are small and are updated in
    place: a replaced label's table row is written before its digest, so an interruption can only leave an entry
    whose digest no longer matches (stale), never another profile marked current.
    """

    def __init__(self, compression: Mapping[str, Any] | None = None) -> None:
        self._compression = dict(compression or {})
        self._group: str | None = None
        self._points = 0
        self._dedup: dict[tuple[bytes, str], int] = {}
        self._table: list[tuple[bytes, str]] = []
        self._entries: dict[int, tuple[int, bytes]] = {}
        self._order: list[int] = []

    def _load(self, group: Any) -> None:
        self._group, self._dedup, self._table, self._entries, self._order = group.name, {}, [], {}, []
        axis = np.asarray(group["axis_1"][()], dtype=np.float32)
        self._points = int(axis.size)
        existing = _datasets(group)
        if existing is None:
            return
        deg, meta, rows_node, digest_node = existing
        if deg.shape[1] != self._points or _attr_text(deg, AXIS_SHA256_ATTR) != axis_sha256(axis):
            remove_angular_support(group)  # a rewritten axis invalidates every earlier entry
            return
        for row in range(deg.shape[0]):
            raw = meta[row]
            key = (np.asarray(deg[row], dtype=np.float32).tobytes(),
                   raw.decode("utf-8") if isinstance(raw, bytes) else str(raw))
            self._dedup.setdefault(key, row)
            self._table.append(key)
        digests = np.asarray(digest_node[()], dtype=np.uint8)
        for number, (label, row) in enumerate(np.asarray(rows_node[()], dtype=np.int64)):
            self._entries[int(label)] = (int(row), digests[number].tobytes())
            self._order.append(int(label))

    def write(self, group: Any, entries: Mapping[int, AngularSupportProfile | None], *,
              intensity_rows: Mapping[int, Any]) -> None:
        """Merge ``entries`` (label → profile, or None to forget the label) into ``group``'s table."""
        if self._group != group.name or (self._entries and _DEG not in group):
            self._load(group)
        new_rows: list[tuple[bytes, str]] = []
        updated: list[int] = []
        removed = False
        for label, profile in entries.items():
            label = int(label)
            if profile is None:
                removed |= self._entries.pop(label, None) is not None
                continue
            key = (_row_values(profile, self._points).tobytes(), _meta_text(profile))
            if key not in self._dedup:
                self._dedup[key] = len(self._table) + len(new_rows)
                new_rows.append(key)
            if label not in self._entries:
                self._order.append(label)
            else:
                updated.append(label)
            self._entries[label] = (self._dedup[key], intensity_digest(intensity_rows[label]))
        if removed:
            self._order = [label for label in self._order if label in self._entries]
        if not self._entries:
            remove_angular_support(group)
            self._dedup, self._table, self._order = {}, [], []
            return
        live = {row for row, _digest in self._entries.values()}
        if len(self._table) + len(new_rows) - len(live) > max(64, len(live)):
            self._compact(group, new_rows)
            return
        self._append_rows(group, new_rows)
        self._write_entries(group, rewrite=removed or _ROWS not in group, updated=updated)

    def _append_rows(self, group: Any, new_rows: list[tuple[bytes, str]]) -> None:
        import h5py
        if not new_rows:
            return
        values = np.stack([np.frombuffer(row, np.float32) for row, _text in new_rows])
        if _DEG not in group:
            deg = group.create_dataset(_DEG, shape=(0, self._points), maxshape=(None, self._points),
                                       chunks=(1, self._points), dtype=np.float32, **self._compression)
            deg.attrs[ANGULAR_SUPPORT_VERSION_ATTR] = np.int64(FORMAT_VERSION)
            deg.attrs[METRIC_ATTR] = SUPPORT_METRIC
            deg.attrs["units"] = "deg"
            deg.attrs[AXIS_SHA256_ATTR] = axis_sha256(group["axis_1"][()])
            # fixed-length UTF-8, not variable-length: the finite replacement seal hashes every result dataset as
            # bounded numeric slabs and refuses object dtypes (record_writer._finite_result_dataset_blocks); one row
            # per chunk, as the degrees, so the NUL padding compresses away
            group.create_dataset(_META, shape=(0,), maxshape=(None,), chunks=(1,),
                                 dtype=h5py.string_dtype("utf-8", MAX_META_BYTES), **self._compression)
        deg, meta = group[_DEG], group[_META]
        start = deg.shape[0]
        deg.resize(start + len(new_rows), axis=0)
        deg[start:] = values
        meta.resize(start + len(new_rows), axis=0)
        meta[start:] = np.array([text.encode("utf-8") for _row, text in new_rows], dtype=meta.dtype)
        self._table.extend(new_rows)

    def _write_entries(self, group: Any, *, rewrite: bool, updated: list[int]) -> None:
        if rewrite:
            for name in (_ROWS, _DIGEST):
                if name in group:
                    del group[name]
            group.create_dataset(_ROWS, data=np.array([[label, self._entries[label][0]] for label in self._order],
                                                      np.int64), maxshape=(None, 2), chunks=(256, 2))
            group.create_dataset(_DIGEST, data=np.frombuffer(
                b"".join(self._entries[label][1] for label in self._order), np.uint8).reshape(-1, 16),
                maxshape=(None, 16), chunks=(256, 16))
            return
        rows, digests = group[_ROWS], group[_DIGEST]
        position = {label: index for index, label in enumerate(self._order)} if updated else {}
        for label in updated:  # table row before digest: an interruption reads stale, never wrong
            rows[position[label]] = (label, self._entries[label][0])
            digests[position[label]] = np.frombuffer(self._entries[label][1], np.uint8)
        start = rows.shape[0]
        added = self._order[start:]
        if added:
            rows.resize(start + len(added), axis=0)
            rows[start:] = np.array([[label, self._entries[label][0]] for label in added], np.int64)
            digests.resize(start + len(added), axis=0)
            digests[start:] = np.frombuffer(b"".join(self._entries[label][1] for label in added),
                                            np.uint8).reshape(-1, 16)

    def _compact(self, group: Any, new_rows: list[tuple[bytes, str]]) -> None:
        everything = self._table + new_rows
        keep = sorted({row for row, _digest in self._entries.values()})
        remap = {old: new for new, old in enumerate(keep)}
        self._entries = {label: (remap[row], digest) for label, (row, digest) in self._entries.items()}
        kept = [everything[row] for row in keep]
        remove_angular_support(group)
        self._table, self._dedup = [], {}
        for number, key in enumerate(kept):
            self._dedup[key] = number
        self._append_rows(group, kept)
        self._write_entries(group, rewrite=True, updated=[])


def write_angular_support(group: Any, entries: Mapping[int, AngularSupportProfile | None], *,
                          intensity_rows: Mapping[int, Any], compression: Mapping[str, Any] | None = None) -> None:
    """Merge ``entries`` (label → profile, or None to forget the label) into ``group``'s support table in one call.

    ``intensity_rows`` holds each written label's intensity row exactly as stored. A record writer keeps one
    :class:`SupportTableWriter` across its batches instead; the caller writes in the same transaction as the rows.
    """
    SupportTableWriter(compression).write(group, entries, intensity_rows=intensity_rows)


def remove_angular_support(group: Any) -> None:
    """Drop every support dataset (the older-output form)."""
    for name in ANGULAR_SUPPORT_DATASETS:
        if name in group:
            del group[name]
