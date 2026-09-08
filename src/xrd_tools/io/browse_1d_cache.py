"""Bounded disposable cache for complete Browse 1-D rows."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import heapq
import os
from threading import Lock, RLock
from types import MappingProxyType
from typing import TYPE_CHECKING, Callable, Mapping

import numpy as np

from xrd_tools.core.physical_memory import (
    PhysicalRootAuthority, PhysicalRootFact, PhysicalRootLease,
    PhysicalRootReservation, physical_root_fact,
)

if TYPE_CHECKING:
    from xrd_tools.io.frame_view import Frame1DRows, FrameScalarCatalog

_GIB = 1 << 30
_MAX_ROW_NAME_BYTES = 1 << 10
_MAX_BROWSE_1D_RESIDENT_ROWS = 4096
_ROW_NAME_PREFIX = "browse-1d-row-v1:"
_ROW_COMPONENTS = frozenset({"axis", "intensity", "sigma"})


def browse_1d_row_name(mode: str, component: str) -> str:
    if type(mode) is not str or not mode:
        raise TypeError("Browse 1-D mode must be a nonempty exact string")
    if type(component) is not str:
        raise TypeError("Browse 1-D row component must be an exact string")
    if component not in _ROW_COMPONENTS:
        raise ValueError("Browse 1-D row component must be axis, intensity, or sigma")
    name = f"{_ROW_NAME_PREFIX}{len(mode.encode('utf-8'))}:{mode}:{component}"
    if len(name.encode("utf-8")) > _MAX_ROW_NAME_BYTES:
        raise ValueError("Browse 1-D row name exceeds the encoded-byte limit")
    return name


def _detect_physical_ram_bytes() -> int | None:
    try:
        pages, page_size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    if type(pages) is not int or type(page_size) is not int or pages <= 0 or page_size <= 0:
        return None
    return pages * page_size


def default_browse_1d_cache_budget(physical_ram_bytes: int | None = None) -> int:
    available = _detect_physical_ram_bytes() if physical_ram_bytes is None else physical_ram_bytes
    if type(available) is not int or available <= 0:
        return _GIB
    return min(_GIB, available // 20)


class Browse1DCachePhase(Enum):
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class Browse1DRowKey:
    frame: int
    label: int
    name: str
    def __post_init__(self) -> None:
        if type(self.frame) is not int or type(self.label) is not int:
            raise TypeError("Browse 1-D frame and label must be exact integers")
        if type(self.name) is not str or not self.name:
            raise TypeError("Browse 1-D row name must be a nonempty exact string")
        if len(self.name.encode("utf-8")) > _MAX_ROW_NAME_BYTES:
            raise ValueError("Browse 1-D row name exceeds the encoded-byte limit")


class _Linear:
    __slots__ = ()
    def __copy__(self): raise TypeError("Browse 1-D linear objects cannot be copied")
    def __deepcopy__(self, memo): raise TypeError("Browse 1-D linear objects cannot be copied")
    def __reduce__(self): raise TypeError("Browse 1-D linear objects cannot be pickled")
    def __reduce_ex__(self, protocol): raise TypeError("Browse 1-D linear objects cannot be pickled")


@dataclass(frozen=True, slots=True, eq=False)
class _Semantic: identity: object
@dataclass(frozen=True, slots=True, eq=False)
class _PendingRow:
    key: Browse1DRowKey; array: np.ndarray; fact: PhysicalRootFact; identity: object; semantic: _Semantic; touch: int
@dataclass(frozen=True, slots=True, eq=False)
class _ResidentRow:
    key: Browse1DRowKey; array: np.ndarray; fact: PhysicalRootFact; identity: object; semantic: _Semantic; lease: PhysicalRootLease; touch: int
@dataclass(frozen=True, slots=True, eq=False)
class _BorrowBinding: token: object; identity: object
@dataclass(frozen=True, slots=True, eq=False)
class _CatalogRecord: frame: int; label: int; scalars: tuple[tuple[str, object], ...]
@dataclass(slots=True)
class _ReleasePending: leases: tuple[PhysicalRootLease, ...]
@dataclass(slots=True)
class _ReservationPending: reservation: PhysicalRootReservation


class Browse1DBorrow(_Linear):
    __slots__ = ("_cache", "_token", "_identity", "_key", "_array", "_released")
    def __init__(self, cache: "Browse1DCache", token: object, row: _ResidentRow) -> None:
        self._cache, self._token, self._identity = cache, token, row.identity
        self._key, self._array, self._released = row.key, row.array, False
    @property
    def key(self) -> Browse1DRowKey: return self._key
    @property
    def array(self) -> np.ndarray: return self._array
    @property
    def released(self) -> bool: return self._released
    def release(self) -> None:
        if not self._released:
            self._cache._release_borrow(self._token, self._identity); self._released = True
    def __enter__(self) -> "Browse1DBorrow":
        if self._released: raise RuntimeError("Browse 1-D borrow is released")
        return self
    def __exit__(self, exc_type, exc, tb) -> None: self.release()


class Browse1DCache(_Linear):
    """Membership owns complete rows; pending custody owns only real charges."""
    def __init__(self, budget_bytes: int | None = None, *, physical_ram_bytes: int | None = None) -> None:
        if budget_bytes is None: budget_bytes = default_browse_1d_cache_budget(physical_ram_bytes)
        if type(budget_bytes) is not int or budget_bytes < 0: raise TypeError("Browse 1-D cache budget must be nonnegative")
        self._budget, self._authority = budget_bytes, PhysicalRootAuthority(budget_bytes)
        self._lock, self._operation_lock = Lock(), RLock()
        self._phase = Browse1DCachePhase.OPEN
        self._rows: tuple[_ResidentRow, ...] = ()
        self._catalog: tuple[_CatalogRecord, ...] = ()
        self._borrows: tuple[_BorrowBinding, ...] = ()
        self._clock = 0
        self._generation = 0
        self._pending: _ReleasePending | _ReservationPending | None = None

    @property
    def budget_bytes(self) -> int: return self._budget
    @property
    def phase(self) -> Browse1DCachePhase:
        with self._lock: return self._phase
    def _require_open(self) -> None:
        if self._phase is Browse1DCachePhase.CLOSED: raise RuntimeError("Browse 1-D cache is closed")
        if self._phase is not Browse1DCachePhase.OPEN or self._pending is not None: raise RuntimeError("Browse 1-D cache reads are gated")
    @property
    def resident_bytes(self) -> int:
        with self._lock: self._require_open()
        return self._authority.retained_bytes
    @property
    def resident_root_count(self) -> int:
        with self._lock: self._require_open()
        return self._authority.retained_root_count
    @property
    def resident_keys(self) -> tuple[Browse1DRowKey, ...]:
        with self._lock: self._require_open(); return tuple(row.key for row in self._rows)
    @property
    def outstanding_borrows(self) -> int:
        with self._lock: self._require_open(); return len(self._borrows)

    @staticmethod
    def _normalize_scalars(scalars: tuple[tuple[str, object], ...]) -> tuple[tuple[str, object], ...]:
        if type(scalars) is not tuple: raise TypeError("Browse 1-D scalar catalog must be an exact tuple")
        names: set[str] = set(); normalized: list[tuple[str, object]] = []
        for item in scalars:
            if type(item) is not tuple or len(item) != 2: raise TypeError("Browse 1-D scalar item must be a pair")
            name, value = item
            if type(name) is not str or not name or len(name.encode("utf-8")) > _MAX_ROW_NAME_BYTES: raise TypeError("Browse 1-D scalar name must be nonempty")
            if name in names: raise ValueError("Browse 1-D scalar name is duplicated")
            if type(value) not in {type(None), bool, int, float, str}: raise TypeError("Browse 1-D catalog values must be scalar")
            names.add(name); normalized.append((name, value))
        return tuple(normalized)
    def record_scalars(self, frame: int, label: int, scalars: tuple[tuple[str, object], ...]) -> None:
        if type(frame) is not int or type(label) is not int: raise TypeError("Browse 1-D frame and label must be exact integers")
        record = _CatalogRecord(frame, label, self._normalize_scalars(scalars))
        with self._lock:
            self._require_open()
            self._catalog = tuple(
                item for item in self._catalog
                if (item.frame, item.label) != (frame, label)
            ) + (record,)
    def scalars(self, frame: int, label: int) -> Mapping[str, object]:
        with self._lock:
            self._require_open()
            record = next((item for item in self._catalog if item.frame == frame and item.label == label), None)
        return MappingProxyType({} if record is None else dict(record.scalars))

    @staticmethod
    def _normalize_rows(frame: int, label: int, rows: tuple[tuple[str, np.ndarray], ...], touch: int) -> tuple[_PendingRow, ...]:
        if type(frame) is not int or type(label) is not int: raise TypeError("Browse 1-D frame and label must be exact integers")
        if type(rows) is not tuple or not rows: raise TypeError("Browse 1-D rows must be a nonempty exact tuple")
        names: set[str] = set(); result: list[_PendingRow] = []
        for offset, item in enumerate(rows, 1):
            if type(item) is not tuple or len(item) != 2: raise TypeError("Browse 1-D row item must be a pair")
            name, array = item; key = Browse1DRowKey(frame, label, name)
            if name in names: raise ValueError("Browse 1-D row name is duplicated")
            if type(array) is not np.ndarray or array.ndim != 1 or array.dtype.kind not in {"i", "u", "f", "c"} or not array.flags.c_contiguous or array.flags.writeable: raise ValueError("Browse 1-D row must be read-only contiguous numeric 1-D")
            fact = physical_root_fact(array)
            if isinstance(fact.root, np.ndarray):
                if fact.root.ndim != 1 or fact.root.flags.writeable: raise ValueError("Browse 1-D physical root must be read-only and 1-D")
            elif type(fact.root) is not bytes: raise ValueError("Browse 1-D physical root must be immutable")
            identity = object(); result.append(_PendingRow(key, array, fact, identity, _Semantic(identity), touch + offset)); names.add(name)
        return tuple(result)

    def _plan_rows(self, current: tuple[_ResidentRow, ...], borrows: tuple[_BorrowBinding, ...], pending: tuple[_PendingRow, ...], protected: tuple[_ResidentRow, ...] = ()) -> tuple[tuple[_ResidentRow, ...], tuple[_ResidentRow, ...]]:
        incoming = {row.key for row in pending}; borrowed = {id(item.identity) for item in borrows}; protected_ids = {id(row) for row in protected}
        if len(protected_ids) != len(protected) or any(not any(row is old for old in current) for row in protected): raise RuntimeError("protected Browse 1-D rows changed")
        mandatory = tuple(row for row in current if row.key in incoming)
        if any(id(row) in protected_ids for row in mandatory): raise ValueError("protected Browse 1-D row cannot be replaced")
        if any(id(row.identity) in borrowed for row in mandatory): raise RuntimeError("borrowed Browse 1-D row cannot be replaced")
        survivors = [row for row in current if row.key not in incoming]; victims = list(mandatory)
        def bytes_for(rows: tuple[object, ...]) -> int:
            roots: dict[int, PhysicalRootFact] = {}
            for row in rows:
                prior = roots.get(id(row.fact.root))
                if prior is not None and (prior.root is not row.fact.root or prior.nbytes != row.fact.nbytes): raise ValueError("Browse 1-D physical-root identity changed")
                roots[id(row.fact.root)] = row.fact
            return sum(item.nbytes for item in roots.values())
        projected_bytes = bytes_for(tuple(survivors) + pending)
        while (
            len(survivors) + len(pending) > _MAX_BROWSE_1D_RESIDENT_ROWS
            or projected_bytes > self._budget
        ):
            eligible = [row for row in survivors if id(row.identity) not in borrowed and id(row) not in protected_ids]
            if not eligible:
                if len(survivors) + len(pending) > _MAX_BROWSE_1D_RESIDENT_ROWS: raise ValueError("Browse 1-D rows exceed the resident row limit")
                raise ValueError("Browse 1-D rows exceed the cache budget")
            oldest = min(eligible, key=lambda row: row.touch); survivors.remove(oldest); victims.append(oldest)
            projected_bytes = bytes_for(tuple(survivors) + pending)
        return tuple(survivors), tuple(victims)

    @staticmethod
    def _release(leases: tuple[PhysicalRootLease, ...]) -> tuple[PhysicalRootLease, ...]:
        failed: list[PhysicalRootLease] = []
        first: BaseException | None = None
        for lease in leases:
            try: lease.release()
            except BaseException as error:
                failed.append(lease)
                if first is None: first = error
        if first is not None: raise first
        return tuple(failed)

    def _settle_pending(self) -> None:
        with self._lock: pending = self._pending
        if pending is None: return
        try:
            if type(pending) is _ReleasePending: self._release(pending.leases)
            else: pending.reservation.rollback()
        except BaseException: raise
        with self._lock:
            if self._pending is pending: self._pending = None

    def store_rows(self, frame: int, label: int, rows: tuple[tuple[str, np.ndarray], ...], *, _protected_rows: tuple[_ResidentRow, ...] = (), cancelled: Callable[[], bool] | None = None) -> str:
        with self._operation_lock:
            while True:
                if cancelled is not None and cancelled():
                    raise InterruptedError("Browse 1-D cache store is stale")
                with self._lock:
                    self._require_open()
                    current, borrows = self._rows, self._borrows
                    generation, touch = self._generation, self._clock
                normalized = self._normalize_rows(frame, label, rows, touch)
                survivors, victims = self._plan_rows(
                    current, borrows, normalized, _protected_rows,
                )
                leases = tuple(row.lease for row in victims)
                with self._lock:
                    self._require_open()
                    if self._generation != generation:
                        continue
                    self._rows = survivors
                    self._clock += len(normalized)
                    self._generation += 1
                    self._pending = _ReleasePending(leases)
                    break
            del victims
            try: self._settle_pending()
            except BaseException: raise
            reservation = self._authority.reserve()
            with self._lock: self._pending = _ReservationPending(reservation)
            try:
                for row in normalized: reservation.reserve(row.array, row.semantic)
                leases_by_semantic = reservation.commit()
                incoming = tuple(_ResidentRow(row.key, row.array, row.fact, row.identity, row.semantic, leases_by_semantic[row.semantic], row.touch) for row in normalized)
                if cancelled is not None and cancelled():
                    raise InterruptedError("Browse 1-D cache store is stale")
            except BaseException:
                try:
                    reservation.rollback()
                except BaseException:
                    # The reservation itself is the one pending real charge;
                    # retain it for close/retry rather than pretending release.
                    raise
                else:
                    with self._lock:
                        if type(self._pending) is _ReservationPending and self._pending.reservation is reservation:
                            self._pending = None
                raise
            with self._lock:
                if self._phase is not Browse1DCachePhase.OPEN or self._pending is None: raise RuntimeError("Browse 1-D cache closed during store")
                self._rows += incoming; self._pending = None; self._generation += 1
            return "accepted"

    @staticmethod
    def _validated_label_rows(catalog: "FrameScalarCatalog", rows: "Frame1DRows", ordinal: int, label: int) -> tuple[tuple[str, np.ndarray], ...]:
        from xrd_tools.io.frame_view import Frame1DModeRows, Frame1DRows, FrameScalarCatalog, FrameScalarRow
        if type(catalog) is not FrameScalarCatalog or type(rows) is not Frame1DRows: raise TypeError("catalog and rows must be exact FrameView values")
        if type(ordinal) is not int or ordinal < 1 or type(label) is not int or label < 0: raise TypeError("ordinal and label must be exact nonnegative integers")
        if ordinal > len(catalog.labels) or catalog.labels[ordinal - 1] != label: raise ValueError("catalog ordinal and label are not exact")
        if rows.artifact_path != catalog.artifact_path or rows.entry != catalog.entry or label not in rows.labels: raise ValueError("catalog and 1-D rows identify different artifacts")
        scalar = catalog.rows[ordinal - 1]
        if type(scalar) is not FrameScalarRow: raise ValueError("catalog row identity is not exact")
        if tuple(mode.mode for mode in rows.modes) != tuple(item[0] for item in catalog.axes_1d): raise ValueError("catalog and row mode ordering differs")
        named: list[tuple[str, np.ndarray]] = []
        for descriptor, mode in zip(catalog.axes_1d, rows.modes, strict=True):
            if type(mode) is not Frame1DModeRows: raise TypeError("1-D modes must be exact Frame1DModeRows")
            name, axis_label, axis_unit, axis_log = descriptor
            if mode.mode != name or mode.axis.label != axis_label or mode.axis.unit != axis_unit or mode.axis.log is not axis_log: raise ValueError("1-D mode axis does not match its catalog descriptor")
            result = mode.row(label); expected = name in scalar.modes_1d
            if (result is None) is expected: raise ValueError("1-D rows do not match catalog label modes")
            if not expected: continue
            intensity, sigma = result
            values = mode.axis.values
            components = (("axis", values), ("intensity", intensity)) + (() if sigma is None else (("sigma", sigma),))
            for component, array in components:
                if type(array) is not np.ndarray or array.dtype != np.dtype(np.float64) or array.ndim != 1 or not array.flags.c_contiguous or array.flags.writeable or array.shape != values.shape: raise ValueError(f"1-D {component} row is not exactly aligned")
                named.append((browse_1d_row_name(name, component), array))
        return tuple(named)

    def store_1d_label(self, catalog: "FrameScalarCatalog", rows: "Frame1DRows", ordinal: int, label: int, *, cancelled: Callable[[], bool] | None = None) -> tuple[Browse1DRowKey, ...]:
        named = self._validated_label_rows(catalog, rows, ordinal, label); keys = tuple(Browse1DRowKey(ordinal, label, name) for name, _ in named)
        with self._operation_lock:
            with self._lock:
                self._require_open(); existing = tuple(row for row in self._rows if row.key.frame == ordinal and row.key.label == label)
                current = {row.key.name: row for row in existing}; incoming = dict(named)
                if set(current).difference(incoming): raise ValueError("Browse 1-D resident label inventory has extra rows")
                for name, row in current.items():
                    fact = physical_root_fact(incoming[name])
                    if row.array is not incoming[name] or row.fact.root is not fact.root or row.fact.nbytes != fact.nbytes or row.lease.released: raise ValueError("Browse 1-D resident row conflicts with projection")
                missing = tuple(item for item in named if item[0] not in current)
            if missing:
                self.store_rows(ordinal, label, missing, _protected_rows=existing, cancelled=cancelled)
            return keys

    def borrow(self, frame: int, label: int, name: str) -> Browse1DBorrow:
        key = Browse1DRowKey(frame, label, name); token = object()
        with self._lock:
            self._require_open(); row = next((item for item in self._rows if item.key == key), None)
            if row is None: raise KeyError(key)
            self._clock += 1
            touched = _ResidentRow(row.key, row.array, row.fact, row.identity, row.semantic, row.lease, self._clock)
            self._rows = tuple(touched if item is row else item for item in self._rows)
            self._generation += 1
            row = touched
            self._borrows += (_BorrowBinding(token, row.identity),)
            self._generation += 1
        return Browse1DBorrow(self, token, row)
    def _release_borrow(self, token: object, identity: object) -> None:
        with self._lock:
            bindings = [item for item in self._borrows if item.token is token and item.identity is identity]
            if len(bindings) != 1: raise RuntimeError("Browse 1-D borrow is not owned")
            self._borrows = tuple(item for item in self._borrows if item is not bindings[0])
            self._generation += 1

    def recover(self) -> str:
        with self._operation_lock:
            self._settle_pending()
            with self._lock: phase = self._phase
            if phase is Browse1DCachePhase.CLOSING: self.close()
            return self.phase.value
    def close(self) -> None:
        with self._operation_lock:
            self._settle_pending()
            with self._lock:
                if self._phase is Browse1DCachePhase.CLOSED: return
                if self._borrows: raise RuntimeError("Browse 1-D cache has outstanding borrows")
                self._phase = Browse1DCachePhase.CLOSING
                rows, self._rows = self._rows, ()
                self._catalog = ()
                if self._pending is None: self._pending = _ReleasePending(tuple(row.lease for row in rows))
            del rows
            self._settle_pending()
            self._authority.close()
            with self._lock: self._phase = Browse1DCachePhase.CLOSED


__all__ = ["Browse1DBorrow", "Browse1DCache", "Browse1DCachePhase", "Browse1DRowKey", "browse_1d_row_name", "default_browse_1d_cache_budget"]
