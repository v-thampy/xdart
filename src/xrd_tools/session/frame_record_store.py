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
from dataclasses import dataclass, replace
from threading import RLock
from types import MappingProxyType

from xrd_tools.core import FrameRecord, FrameView

_ModeKey = tuple[str, str]


@dataclass(frozen=True, slots=True, eq=False)
class FrameHydrationRequest:
    label: int | str
    source_identity: str
    revision: int
    generation: int
    persisted_modes: frozenset[_ModeKey] = frozenset()
    durable_modes: frozenset[_ModeKey] = frozenset()
    dropped_modes: frozenset[_ModeKey] = frozenset()
    projected: bool = False
    commit_epoch: object | None = None


@dataclass(frozen=True, slots=True)
class FrameHydrationResult:
    request: FrameHydrationRequest
    record: FrameRecord


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

    # The registered disk hydrator reads integrated results from the processed
    # container. Detector raw pixels belong to PublicationStore/source fallback.
    hydration_purposes = frozenset({"1d", "2d", "record"})

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
        self._revisions: dict[int | str, int] = {}
        self._generation = 0
        # ``_projected`` = labels a ScanSession owns; else the pre-C2-A rule.
        self._persisted_modes: dict[int | str, set[_ModeKey]] = {}
        self._durable_modes: dict[int | str, set[_ModeKey]] = {}
        self._dropped_modes: dict[int | str, set[_ModeKey]] = {}
        self._projected: set[int | str] = set()
        self._heavy_labels: list[int | str] = []
        self._max_items = max_items
        self._max_heavy_items = max_heavy_items
        self._require_persisted_for_eviction = bool(require_persisted_for_eviction)
        self._hydrator: Callable[[int | str], FrameRecord | None] | None = None
        self._hydrator_revision_qualified = False

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._source_ids.clear()
            self._revisions.clear()
            self._generation += 1
            self._persisted_modes.clear()
            self._durable_modes.clear()
            self._dropped_modes.clear()
            self._projected.clear()
            self._heavy_labels.clear()

    def set_hydrator(
        self, hydrator: Callable | None, *, revision_qualified: bool = False,
    ) -> None:
        """Register a synchronous hydrator for thinned records.

        Disk-backed hydrators must be called from a worker thread, never a GUI
        render thread.  The store deliberately does not hide I/O behind an
        implicit thread; UI integrations should register a disk reader here and
        invoke :meth:`get_or_hydrate` from their existing hydration worker.
        """
        with self._lock:
            self._hydrator = hydrator
            self._hydrator_revision_qualified = bool(revision_qualified)

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
                for registry in (self._durable_modes, self._dropped_modes):
                    stale = registry.get(label)
                    if stale is not None:
                        self._assign_locked(registry, label,
                                            stale - incoming_mode_keys)
            else:
                persisted_set = set()
                self._durable_modes.pop(label, None)
                self._dropped_modes.pop(label, None)

            self._records.pop(label, None)
            self._drop_heavy_label_locked(label)
            self._records[label] = record
            self._source_ids[label] = source_id
            self._revisions[label] = self._revisions.get(label, 0) + 1
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
                self._dropped_modes.setdefault(label, set()).update(requested)
                thinned = _thin_modes(record, requested)
                self._records[label] = thinned
                if not _has_heavy_payload(thinned):
                    self._drop_heavy_label_locked(label)
            self._enforce_bounds_locked()

    def replace_projection(
        self,
        label: int | str,
        *,
        hydratable: Iterable[_ModeKey] = (),
        durable: Iterable[_ModeKey] = (),
        dropped: Iterable[_ModeKey] = (),
    ) -> None:
        """Publish this label's COMPLETE projection in ONE atomic swap: no
        half-published mixture, and no stale mark survives a replacement."""
        with self._lock:
            record = self._records.get(label)
            if record is None:
                return
            valid = _record_mode_keys(record)
            hydratable_set = _normalize_mode_keys(hydratable) & valid
            durable_set = _normalize_mode_keys(durable) & valid
            dropped_set = _normalize_mode_keys(dropped) & valid
            self._projected.add(label)
            self._assign_locked(self._persisted_modes, label, hydratable_set)
            self._assign_locked(self._durable_modes, label, durable_set)
            self._assign_locked(self._dropped_modes, label, dropped_set)
            if dropped_set:
                thinned = _thin_modes(record, dropped_set)
                self._records[label] = thinned
                if not _has_heavy_payload(thinned):
                    self._drop_heavy_label_locked(label)
            self._enforce_bounds_locked()

    @staticmethod
    def _assign_locked(registry: dict, label, value: set[_ModeKey]) -> None:
        if value:
            registry[label] = value
        else:
            registry.pop(label, None)

    def release_heavy(self, label: int | str) -> bool:
        """Release only the heavy arrays this label is licensed to drop."""
        with self._lock:
            record = self._records.get(label)
            if record is None:
                return False
            releasable = (self._releasable_modes_locked(label)
                          & _heavy_mode_keys(record))
            if not releasable:
                return False
            thinned = _thin_modes(record, releasable)
            self._records[label] = thinned
            if not _has_heavy_payload(thinned):
                self._drop_heavy_label_locked(label)
            return True

    def can_release_record(self, label: int | str) -> bool:
        """Return whether the exact row is qualified for complete release."""
        with self._lock:
            return self._label_deletable_locked(label)

    def release_record(self, label: int | str) -> bool:
        """Release one recoverable row without weakening projection truth."""
        with self._lock:
            if not self._label_deletable_locked(label):
                return False
            self._forget_label_locked(label)
            return True

    def exchange_releasable_record(
        self,
        label: int | str,
        *,
        expected: FrameRecord | None,
        replacement: FrameRecord | None,
        source_identity: str | None = None,
        persisted: bool = False,
    ) -> bool:
        """Atomically exchange one exact releasable row without merging."""
        if replacement is not None and replacement.label != label:
            raise ValueError(
                "replacement record label differs from exchange label"
            )
        resolved_source = ""
        replacement_is_heavy = False
        if replacement is not None:
            resolved_source = (
                str(source_identity)
                if source_identity is not None
                else _source_identity_from_record(replacement)
            )
            replacement_is_heavy = _has_heavy_payload(replacement)
        with self._lock:
            current = self._records.get(label)
            if current is not expected:
                return False
            if current is not None and not self._label_deletable_locked(label):
                return False
            if replacement is not None:
                next_items = len(self._records) + (current is None)
                next_heavy = (
                    len(self._heavy_labels)
                    - int(label in self._heavy_labels)
                    + int(replacement_is_heavy)
                )
                if (
                    self._max_items is not None
                    and next_items > self._max_items
                ) or (
                    self._max_heavy_items is not None
                    and next_heavy > self._max_heavy_items
                ):
                    return False
            if current is not None:
                self._forget_label_locked(label)
            if replacement is None:
                return True
            self._records[label] = replacement
            self._source_ids[label] = resolved_source
            self._revisions[label] = self._revisions.get(label, 0) + 1
            if persisted:
                self._persisted_modes[label] = _record_mode_keys(replacement)
            if replacement_is_heavy:
                self._heavy_labels.append(label)
            return True

    def durable_modes(self, label: int | str) -> frozenset[_ModeKey]:
        """Modes durable on EVERY applicable target; the only releasable ones."""
        with self._lock:
            return frozenset(self._durable_modes.get(label, set()))

    def dropped_modes(self, label: int | str) -> frozenset[_ModeKey]:
        with self._lock:
            return frozenset(self._dropped_modes.get(label, set()))

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

    def get_or_hydrate(self, label: int | str, *, commit_gate=None,
                       commit_epoch=None) -> FrameRecord | None:
        """Hydrate outside locks, then commit only through the live exact gate.

        The captured record/source/projection facts are rechecked after the
        read and after gate entry.  Losing either authority inserts nothing.
        """
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
            # Stale-read fence: merge back only if every captured fact is current.
            captured = self._capture_locked(label, record, prev_persisted)
            if label in self._projected and not prev_persisted:
                return record
        if hydrator is None:
            return record
        request = FrameHydrationRequest(
            label=label,
            source_identity=captured[1],
            revision=captured[2],
            generation=captured[3],
            persisted_modes=captured[4],
            durable_modes=captured[5],
            dropped_modes=captured[6],
            projected=captured[7],
            commit_epoch=commit_epoch,
        )
        returned = hydrator(request if self._hydrator_revision_qualified else label)
        if returned is None:
            return record
        certified_return = isinstance(returned, FrameHydrationResult)
        if certified_return:
            if returned.request is not request:
                return record
            fresh = returned.record
        else:
            # Documented compatibility only for an unqualified legacy row.
            # It can never certify a source-qualified stored revision.
            if captured[1] or self._hydrator_revision_qualified:
                return record
            fresh = returned
        # Hydration is a read of the exact captured logical row.  A hydrator
        # may fill payloads, but it may not redirect that read to a different
        # label or source identity.
        if fresh.label != label:
            return record
        captured_source_identity = captured[1]
        fresh_source_identity = _source_identity_from_record(fresh)
        if captured_source_identity and not fresh_source_identity:
            return record
        if fresh_source_identity and not _same_source_id(
                captured_source_identity, fresh_source_identity):
            return record
        with self._lock:
            current = self._records.get(label)
            if current is None or self._capture_locked(
                    label, current, self._persisted_modes.get(label, set())
            ) != captured:
                return current
        if commit_gate is not None and not commit_gate.enter(request.commit_epoch):
            return record
        try:
            with self._lock:
                current = self._records.get(label)
                if current is None or self._capture_locked(
                        label, current, self._persisted_modes.get(label, set())
                ) != captured:
                    return current
                durable = set(self._durable_modes.get(label, set()))
                dropped = set(self._dropped_modes.get(label, set()))
                projected = label in self._projected
                merged = self.upsert(
                    fresh,
                    source_identity=captured_source_identity,
                    persisted_modes=prev_persisted,
                )
                # Hydration RE-ARMS an existing revision: restore its projection.
                valid = _record_mode_keys(merged)
                self._assign_locked(self._durable_modes, label, durable & valid)
                self._assign_locked(self._dropped_modes, label, dropped & valid)
                if projected:
                    self._projected.add(label)
                return merged
        finally:
            if commit_gate is not None:
                commit_gate.leave()

    def _capture_locked(self, label: int | str, record: FrameRecord,
                        persisted: set[_ModeKey]) -> tuple:
        return (id(record), self._source_ids.get(label, ""),
                self._revisions.get(label, 0), self._generation,
                frozenset(persisted),
                frozenset(self._durable_modes.get(label, set())),
                frozenset(self._dropped_modes.get(label, set())),
                label in self._projected)

    def is_persisted(self, label: int | str) -> bool:
        with self._lock:
            return self._label_persisted_locked(label)

    def persisted_modes(self, label: int | str) -> frozenset[_ModeKey]:
        """The ``(dim, mode)`` keys of ``label`` confirmed on disk.

        A mode consciously discarded at write (:meth:`mark_dropped`) is
        intentionally absent here (it was never written).  Persistence alone
        does not imply current hydratability: a store also needs a registered
        hydrator.  Use :meth:`hydratable_modes` for that combined fact.
        Read-only; does not touch retention/eviction policy."""
        with self._lock:
            return frozenset(self._persisted_modes.get(label, set()))

    def hydratable_modes(self, label: int | str) -> frozenset[_ModeKey]:
        """Persisted modes recoverable through this store right now."""
        with self._lock:
            if self._hydrator is None:
                return frozenset()
            return frozenset(self._persisted_modes.get(label, set()))

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

    def _releasable_modes_locked(self, label: int | str) -> set[_ModeKey]:
        """Under a projection exactly the DURABLE set; persistence never evicts."""
        if label in self._projected:
            return set(self._durable_modes.get(label, set()))
        return set(self._persisted_modes.get(label, set()))

    def _label_persisted_locked(self, label: int | str) -> bool:
        record = self._records.get(label)
        if record is None:
            return False
        mode_keys = _record_mode_keys(record)
        return bool(mode_keys) and mode_keys.issubset(
            self._persisted_modes.get(label, set())
        )

    def _label_deletable_locked(self, label: int | str) -> bool:
        """NON-DROPPED modes recoverable here, non-empty, none awaiting durability."""
        record = self._records.get(label)
        if record is None:
            return False
        remaining = _record_mode_keys(record) - self._dropped_modes.get(label, set())
        if not remaining or not remaining.issubset(
                self._persisted_modes.get(label, set())):
            return False
        return _heavy_mode_keys(record).issubset(
            self._releasable_modes_locked(label))

    def _label_heavy_payload_persisted_locked(self, label: int | str) -> bool:
        record = self._records.get(label)
        if record is None:
            return False
        heavy_keys = _heavy_mode_keys(record)
        return bool(heavy_keys) and heavy_keys.issubset(
            self._releasable_modes_locked(label)
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
                        or self._label_deletable_locked(candidate)
                    ),
                    None,
                )
                if label is None:
                    break
                self._forget_label_locked(label)

    def _forget_label_locked(self, label: int | str) -> None:
        self._records.pop(label, None)
        self._source_ids.pop(label, None)
        self._persisted_modes.pop(label, None)
        self._durable_modes.pop(label, None)
        self._dropped_modes.pop(label, None)
        self._projected.discard(label)
        self._drop_heavy_label_locked(label)


__all__ = ["FrameRecordStore"]
