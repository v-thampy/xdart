# -*- coding: utf-8 -*-
"""Headless bounded store for :class:`~xrd_tools.core.FrameRecord`.

This is the ADR-0005 foundation: the durable, GUI-free place where a scan
session can accumulate multi-result frame records while bounding heavy arrays.
It is intentionally small and dormant-friendly.  A caller may use it directly
from notebooks today; xdart can later project this store into its GUI-local
``PublicationStore`` without moving the display flip and the ownership move in
one risky step.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from threading import RLock
from types import MappingProxyType

from xrd_tools.core import FrameRecord, FrameView

_ModeKey = tuple[str, str]


def _record_views(record: FrameRecord) -> tuple[FrameView, ...]:
    return tuple(record.results_1d.values()) + tuple(record.results_2d.values())


def _view_has_heavy_payload(view: FrameView) -> bool:
    return (
        view.intensity_1d is not None
        or view.sigma_1d is not None
        or view.intensity_2d is not None
        or view.sigma_2d is not None
        or view.raw is not None
        or view.thumbnail is not None
    )


def _record_mode_keys(record: FrameRecord) -> set[_ModeKey]:
    keys: set[_ModeKey] = set()
    keys.update(("1d", mode) for mode in record.results_1d)
    keys.update(("2d", mode) for mode in record.results_2d)
    return keys


def _normalize_mode_keys(modes: Iterable[_ModeKey] | _ModeKey) -> set[_ModeKey]:
    if (
        isinstance(modes, tuple)
        and len(modes) == 2
        and isinstance(modes[0], str)
    ):
        iterable = (modes,)
    else:
        iterable = tuple(modes)  # type: ignore[arg-type]
    keys: set[_ModeKey] = set()
    for key in iterable:
        dim, mode = key
        keys.add((str(dim), str(mode)))
    return keys


def _heavy_mode_keys(record: FrameRecord) -> set[_ModeKey]:
    keys: set[_ModeKey] = set()
    for mode, view in record.results_1d.items():
        if _view_has_heavy_payload(view):
            keys.add(("1d", mode))
    for mode, view in record.results_2d.items():
        if _view_has_heavy_payload(view):
            keys.add(("2d", mode))
    return keys


def _has_heavy_payload(record: FrameRecord) -> bool:
    return bool(_heavy_mode_keys(record))


def _thin_view(view: FrameView) -> FrameView:
    """Drop array payloads but keep labels, axes, metadata, source, and modes."""
    return replace(
        view,
        intensity_1d=None,
        sigma_1d=None,
        intensity_2d=None,
        sigma_2d=None,
        raw=None,
        thumbnail=None,
    )


def _thin_record(record: FrameRecord) -> FrameRecord:
    return FrameRecord(
        label=record.label,
        results_1d={mode: _thin_view(view) for mode, view in record.results_1d.items()},
        results_2d={mode: _thin_view(view) for mode, view in record.results_2d.items()},
        active_mode_1d=record.active_mode_1d,
        active_mode_2d=record.active_mode_2d,
    )


def _thin_modes(record: FrameRecord, modes: set[_ModeKey]) -> FrameRecord:
    """Drop the array payloads of ONLY ``modes``; every other mode is kept as-is."""
    return FrameRecord(
        label=record.label,
        results_1d={
            mode: (_thin_view(view) if ("1d", mode) in modes else view)
            for mode, view in record.results_1d.items()
        },
        results_2d={
            mode: (_thin_view(view) if ("2d", mode) in modes else view)
            for mode, view in record.results_2d.items()
        },
        active_mode_1d=record.active_mode_1d,
        active_mode_2d=record.active_mode_2d,
    )


def _merge_records(existing: FrameRecord, incoming: FrameRecord) -> FrameRecord:
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


def _source_identity_from_record(record: FrameRecord) -> str:
    ids: set[str] = set()
    for view in _record_views(record):
        if view.source_path is None and view.source_frame_index is None:
            continue
        path = "" if view.source_path is None else str(view.source_path)
        frame = "" if view.source_frame_index is None else str(int(view.source_frame_index))
        ids.add(f"{path}#{frame}")
    if len(ids) > 1:
        raise ValueError(
            f"FrameRecord {record.label!r} carries conflicting source identities: "
            f"{sorted(ids)!r}"
        )
    return next(iter(ids), "")


def _same_source_id(a: str, b: str) -> bool:
    """Strict headless merge rule: exact non-empty identity, or both missing."""
    if not a or not b:
        return not a and not b
    return a == b


class FrameRecordStore:
    """Thread-safe, bounded store for one scan's :class:`FrameRecord` objects.

    Heavy arrays are evicted only after every heavy result mode on that frame is
    marked persisted, unless ``require_persisted_for_eviction=False`` is
    requested.  This preserves the important "persist before evict" invariant
    while still bounding memory during long scans once durable sinks have
    flushed frames.
    """

    def __init__(
        self,
        *,
        max_items: int | None = None,
        max_heavy_items: int | None = 64,
        require_persisted_for_eviction: bool = True,
    ) -> None:
        if max_items is not None and max_items < 1:
            raise ValueError("max_items must be positive or None")
        if max_heavy_items is not None and max_heavy_items < 0:
            raise ValueError("max_heavy_items must be non-negative or None")
        self._lock = RLock()
        self._records: dict[int | str, FrameRecord] = {}
        self._source_ids: dict[int | str, str] = {}
        self._persisted_modes: dict[int | str, set[_ModeKey]] = {}
        self._heavy_labels: list[int | str] = []
        self._max_items = max_items
        self._max_heavy_items = max_heavy_items
        self._require_persisted_for_eviction = bool(require_persisted_for_eviction)
        self._hydrator: Callable[[int | str], FrameRecord | None] | None = None

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._source_ids.clear()
            self._persisted_modes.clear()
            self._heavy_labels.clear()

    def set_hydrator(
        self, hydrator: Callable[[int | str], FrameRecord | None] | None
    ) -> None:
        """Register a synchronous hydrator for thinned records.

        Disk-backed hydrators must be called from a worker thread, never a GUI
        render thread.  The store deliberately does not hide I/O behind an
        implicit thread; UI integrations should register a disk reader here and
        invoke :meth:`get_or_hydrate` from their existing hydration worker.
        """
        with self._lock:
            self._hydrator = hydrator

    def upsert(
        self,
        record: FrameRecord,
        *,
        source_identity: str | None = None,
        persisted: bool = False,
        persisted_modes: Iterable[_ModeKey] | None = None,
    ) -> FrameRecord:
        source_id = (
            str(source_identity)
            if source_identity is not None
            else _source_identity_from_record(record)
        )
        label = record.label
        incoming_mode_keys = _record_mode_keys(record)
        with self._lock:
            existing = self._records.get(label)
            if existing is not None and _same_source_id(self._source_ids.get(label, ""), source_id):
                record = _merge_records(existing, record)
                persisted_set = set(self._persisted_modes.get(label, set()))
                persisted_set.difference_update(incoming_mode_keys)
            else:
                persisted_set = set()

            self._records.pop(label, None)
            self._drop_heavy_label_locked(label)
            self._records[label] = record
            self._source_ids[label] = source_id
            # Per-mode persistence (``persisted_modes``) takes precedence over the
            # blanket ``persisted`` flag: ``get_or_hydrate`` uses it so a hydrator
            # that returns an EXTRA freshly-computed (unsaved) mode does NOT get
            # that mode marked persisted — which would let it be evicted before it
            # is written (the persist-before-evict bug 748fcac fixed).
            if persisted_modes is not None:
                valid = _record_mode_keys(record)
                persisted_set.update(key for key in persisted_modes if key in valid)
            elif persisted:
                persisted_set.update(incoming_mode_keys)
            if persisted_set:
                self._persisted_modes[label] = persisted_set
            else:
                self._persisted_modes.pop(label, None)
            if _has_heavy_payload(record):
                self._heavy_labels.append(label)
            self._enforce_bounds_locked()
            return self._records[label]

    def mark_persisted(
        self,
        labels: Iterable[int | str] | int | str,
        *,
        modes: Iterable[_ModeKey] | _ModeKey | None = None,
    ) -> None:
        if isinstance(labels, (str, bytes)):
            iterable = (labels,)
        else:
            try:
                iterable = tuple(labels)  # type: ignore[arg-type]
            except TypeError:
                iterable = (labels,)  # type: ignore[assignment]
        requested_modes = None if modes is None else _normalize_mode_keys(modes)
        with self._lock:
            for label in iterable:
                record = self._records.get(label)
                if record is None:
                    continue
                valid_modes = _record_mode_keys(record)
                persisted = self._persisted_modes.setdefault(label, set())
                if requested_modes is None:
                    persisted.update(valid_modes)
                else:
                    persisted.update(requested_modes.intersection(valid_modes))
                if not persisted:
                    self._persisted_modes.pop(label, None)
            self._enforce_bounds_locked()

    def mark_dropped(
        self,
        labels: Iterable[int | str] | int | str,
        *,
        modes: Iterable[_ModeKey] | _ModeKey,
    ) -> None:
        """Mark ``modes`` on ``labels`` as CONSCIOUSLY DISCARDED at write (MEM-1b).

        Unlike :meth:`mark_persisted`, this makes NO promise that the mode is on
        disk — it was intentionally not written (e.g. an all-dummy GI 2D cake
        below the critical angle), so hydration must never be attempted for it
        and it is NEVER added to ``_persisted_modes`` (``is_persisted`` stays
        honest).  It drops that mode's heavy array payload from the record right
        away: otherwise the cake would pin forever, because
        ``_label_heavy_payload_persisted_locked`` can never clear a mode that is
        not — and must not be — persisted (the leak).  Light labels/axes/
        metadata and every other mode on the frame are left intact.
        """
        if isinstance(labels, (str, bytes)):
            iterable = (labels,)
        else:
            try:
                iterable = tuple(labels)  # type: ignore[arg-type]
            except TypeError:
                iterable = (labels,)  # type: ignore[assignment]
        requested = _normalize_mode_keys(modes)
        if not requested:
            return
        with self._lock:
            for label in iterable:
                record = self._records.get(label)
                if record is None:
                    continue
                thinned = _thin_modes(record, requested)
                self._records[label] = thinned
                if not _has_heavy_payload(thinned):
                    self._drop_heavy_label_locked(label)
            self._enforce_bounds_locked()

    def get(self, label: int | str) -> FrameRecord | None:
        with self._lock:
            return self._records.get(label)

    def get_many(self, labels: Iterable[int | str]) -> dict[int | str, FrameRecord]:
        with self._lock:
            return {
                label: record
                for label in labels
                if (record := self._records.get(label)) is not None
            }

    def get_or_hydrate(self, label: int | str) -> FrameRecord | None:
        with self._lock:
            record = self._records.get(label)
            if record is None or _has_heavy_payload(record):
                return record
            hydrator = self._hydrator
            # Capture the per-mode persisted set BEFORE hydration: only these
            # modes stay persisted afterward.  A hydrator that returns an EXTRA
            # freshly-computed mode (e.g. a lazy non-primary GI mode not on disk)
            # must NOT inherit persisted status — else it could be thinned before
            # it is written (persist-before-evict, the 748fcac bug).
            prev_persisted = set(self._persisted_modes.get(label, set()))
        if hydrator is None:
            return record
        fresh = hydrator(label)
        if fresh is None:
            return record
        current_source_identity = self.source_identity(label)
        fresh_source_identity = _source_identity_from_record(fresh)
        source_identity = (
            current_source_identity
            if current_source_identity and not fresh_source_identity
            else None
        )
        return self.upsert(
            fresh,
            source_identity=source_identity,
            persisted_modes=prev_persisted,
        )

    def is_persisted(self, label: int | str) -> bool:
        with self._lock:
            return self._label_persisted_locked(label)

    def has_heavy_payload(self, label: int | str) -> bool:
        with self._lock:
            record = self._records.get(label)
            return bool(record is not None and _has_heavy_payload(record))

    def source_identity(self, label: int | str) -> str:
        with self._lock:
            return self._source_ids.get(label, "")

    def labels(self) -> tuple[int | str, ...]:
        with self._lock:
            return tuple(self._records)

    def snapshot(self) -> Mapping[int | str, FrameRecord]:
        with self._lock:
            return MappingProxyType(dict(self._records))

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def _drop_heavy_label_locked(self, label: int | str) -> None:
        try:
            self._heavy_labels.remove(label)
        except ValueError:
            pass

    def _find_evictable_heavy_label_locked(self) -> int | str | None:
        for label in self._heavy_labels:
            if (
                not self._require_persisted_for_eviction
                or self._label_heavy_payload_persisted_locked(label)
            ):
                return label
        return None

    def _label_persisted_locked(self, label: int | str) -> bool:
        record = self._records.get(label)
        if record is None:
            return False
        mode_keys = _record_mode_keys(record)
        return bool(mode_keys) and mode_keys.issubset(
            self._persisted_modes.get(label, set())
        )

    def _label_heavy_payload_persisted_locked(self, label: int | str) -> bool:
        record = self._records.get(label)
        if record is None:
            return False
        heavy_keys = _heavy_mode_keys(record)
        return bool(heavy_keys) and heavy_keys.issubset(
            self._persisted_modes.get(label, set())
        )

    def _enforce_bounds_locked(self) -> None:
        if self._max_heavy_items is not None:
            while len(self._heavy_labels) > self._max_heavy_items:
                label = self._find_evictable_heavy_label_locked()
                if label is None:
                    break
                record = self._records.get(label)
                self._drop_heavy_label_locked(label)
                if record is not None:
                    self._records[label] = _thin_record(record)

        if self._max_items is not None:
            while len(self._records) > self._max_items:
                label = next(
                    (
                        candidate
                        for candidate in self._records
                        if not self._require_persisted_for_eviction
                        or self._label_persisted_locked(candidate)
                    ),
                    None,
                )
                if label is None:
                    break
                self._records.pop(label, None)
                self._source_ids.pop(label, None)
                self._persisted_modes.pop(label, None)
                self._drop_heavy_label_locked(label)


__all__ = ["FrameRecordStore"]
