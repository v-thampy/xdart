"""Bounded disposable cache for complete Browse 1-D rows.

Membership owns arrays; incomplete cleanup retains only its actual reservation
or lease handles. Failed fills may discard unborrowed, rereadable old entries.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import heapq
import os
from threading import Lock
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
    """Return one collision-free versioned cache name for a 1-D row role.

    The decimal length field is the exact UTF-8 byte length of ``mode``.
    Consumers therefore never depend on a separator being absent from an
    arbitrary Unicode mode name.  The complete encoded key remains within the
    cache's existing row-name limit.
    """

    if type(mode) is not str or not mode:
        raise TypeError("Browse 1-D mode must be a nonempty exact string")
    if type(component) is not str:
        raise TypeError("Browse 1-D row component must be an exact string")
    if component not in _ROW_COMPONENTS:
        raise ValueError(
            "Browse 1-D row component must be axis, intensity, or sigma"
        )
    mode_bytes = mode.encode("utf-8")
    name = f"{_ROW_NAME_PREFIX}{len(mode_bytes)}:{mode}:{component}"
    if len(name.encode("utf-8")) > _MAX_ROW_NAME_BYTES:
        raise ValueError("Browse 1-D row name exceeds the encoded-byte limit")
    return name


def _detect_physical_ram_bytes() -> int | None:
    """Return installed RAM without adding an optional runtime dependency."""

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    if type(pages) is not int or type(page_size) is not int:
        return None
    if pages <= 0 or page_size <= 0:
        return None
    return pages * page_size


def default_browse_1d_cache_budget(
    physical_ram_bytes: int | None = None,
) -> int:
    """Return ``min(1 GiB, physical RAM / 20)`` or the 1-GiB fallback."""

    detected = (
        _detect_physical_ram_bytes()
        if physical_ram_bytes is None
        else physical_ram_bytes
    )
    if type(detected) is not int or detected <= 0:
        return _GIB
    return min(_GIB, detected // 20)


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


class _LinearObject:
    """Reject aliasing of stateful cache owners and linear handles."""

    __slots__ = ()

    def __copy__(self):
        raise TypeError("Browse 1-D linear objects cannot be copied")

    def __deepcopy__(self, memo):
        raise TypeError("Browse 1-D linear objects cannot be copied")

    def __reduce__(self):
        raise TypeError("Browse 1-D linear objects cannot be pickled")

    def __reduce_ex__(self, protocol):
        raise TypeError("Browse 1-D linear objects cannot be pickled")


@dataclass(frozen=True, slots=True, eq=False)
class _RowSemantic:
    row_identity: object


@dataclass(frozen=True, slots=True, eq=False)
class _BorrowToken:
    row_identity: object


@dataclass(frozen=True, slots=True, eq=False)
class _PendingRow:
    key: Browse1DRowKey
    array: np.ndarray
    fact: PhysicalRootFact
    row_identity: object
    semantic: _RowSemantic
    touch: int


@dataclass(frozen=True, slots=True, eq=False)
class _ResidentRow:
    key: Browse1DRowKey
    array: np.ndarray
    fact: PhysicalRootFact
    row_identity: object
    semantic: _RowSemantic
    lease: PhysicalRootLease
    touch: int


@dataclass(frozen=True, slots=True, eq=False)
class _BorrowBinding:
    token: _BorrowToken
    row_identity: object


@dataclass(frozen=True, slots=True, eq=False)
class _CatalogRecord:
    frame: int
    label: int
    scalars: tuple[tuple[str, object], ...]


@dataclass(frozen=True, slots=True, eq=False)
class _CacheState:
    rows: tuple[_ResidentRow, ...] = ()
    catalog: tuple[_CatalogRecord, ...] = ()
    borrows: tuple[_BorrowBinding, ...] = ()
    clock: int = 0
    phase: Browse1DCachePhase = Browse1DCachePhase.OPEN


class Browse1DBorrow(_LinearObject):
    """One exact non-evictable borrow of one resident row object."""

    __slots__ = (
        "_cache", "_token", "_row_identity", "_released", "_key", "_array",
    )

    def __init__(
        self,
        cache: "Browse1DCache",
        token: _BorrowToken,
        row: _ResidentRow,
    ) -> None:
        self._cache = cache
        self._token = token
        self._row_identity = row.row_identity
        self._released = False
        self._key = row.key
        self._array = row.array

    @property
    def key(self) -> Browse1DRowKey:
        return self._key

    @property
    def array(self) -> np.ndarray:
        return self._array

    @property
    def released(self) -> bool:
        return self._released

    def release(self) -> None:
        if self._released:
            return
        self._cache._release_borrow(self._token, self._row_identity)
        self._released = True

    def __enter__(self) -> "Browse1DBorrow":
        if self._released:
            raise RuntimeError("Browse 1-D borrow is released")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


class Browse1DCache(_LinearObject):
    """One membership graph and at most one real pending cleanup bundle."""

    def __init__(self, budget_bytes: int | None = None, *,
                 physical_ram_bytes: int | None = None) -> None:
        if budget_bytes is None:
            budget_bytes = default_browse_1d_cache_budget(physical_ram_bytes)
        if type(budget_bytes) is not int or budget_bytes < 0:
            raise TypeError("Browse 1-D cache budget must be nonnegative")
        self._budget = budget_bytes
        self._authority = PhysicalRootAuthority(budget_bytes)
        self._lock = Lock()
        self._state = _CacheState()
        self._busy = False
        self._pending: PhysicalRootReservation | tuple[PhysicalRootLease, ...] | None = None

    def _snapshot_state(self) -> _CacheState:
        with self._lock:
            return self._state

    def _cas_state(self, expected: _CacheState, replacement: _CacheState) -> bool:
        # Both graphs remain held by the caller: swapping never drops arrays
        # or runs a destructor while the membership lock is held.
        with self._lock:
            if self._state is not expected:
                return False
            self._state = replacement
            return True

    @staticmethod
    def _require_open(state: _CacheState) -> None:
        if state.phase is not Browse1DCachePhase.OPEN:
            raise RuntimeError("Browse 1-D cache is closed or closing")

    def _read_state(self) -> _CacheState:
        with self._lock:
            self._require_open(self._state)
            if self._busy or self._pending is not None:
                raise RuntimeError("Browse 1-D cache reads are gated")
            return self._state

    def _begin_work(self, *, closing: bool = False) -> None:
        with self._lock:
            if self._busy:
                raise RuntimeError("Browse 1-D cache operation is in progress")
            if not closing:
                self._require_open(self._state)
            self._busy = True

    def _end_work(self) -> None:
        with self._lock:
            self._busy = False

    @property
    def budget_bytes(self) -> int:
        return self._budget

    @property
    def phase(self) -> Browse1DCachePhase:
        return self._snapshot_state().phase

    @property
    def resident_bytes(self) -> int:
        self._read_state()
        return self._authority.retained_bytes

    @property
    def resident_root_count(self) -> int:
        self._read_state()
        return self._authority.retained_root_count

    @property
    def resident_keys(self) -> tuple[Browse1DRowKey, ...]:
        return tuple(row.key for row in self._read_state().rows)

    @property
    def outstanding_borrows(self) -> int:
        return len(self._read_state().borrows)

    @staticmethod
    def _validate_frame_label(frame: int, label: int) -> None:
        if type(frame) is not int or type(label) is not int:
            raise TypeError("Browse 1-D frame and label must be exact integers")


    @staticmethod
    def _normalize_scalars(
        scalars: tuple[tuple[str, object], ...],
    ) -> tuple[tuple[str, object], ...]:
        if type(scalars) is not tuple:
            raise TypeError("Browse 1-D scalar catalog must be an exact tuple")
        names: set[str] = set()
        normalized: list[tuple[str, object]] = []
        for item in scalars:
            if type(item) is not tuple or len(item) != 2:
                raise TypeError("Browse 1-D scalar item must be a pair")
            name, value = item
            if type(name) is not str or not name:
                raise TypeError("Browse 1-D scalar name must be nonempty")
            if len(name.encode("utf-8")) > _MAX_ROW_NAME_BYTES:
                raise ValueError("Browse 1-D scalar name exceeds the limit")
            if name in names:
                raise ValueError("Browse 1-D scalar name is duplicated")
            if type(value) not in {type(None), bool, int, float, str}:
                raise TypeError("Browse 1-D catalog values must be scalar")
            names.add(name)
            normalized.append((name, value))
        return tuple(normalized)


    @staticmethod
    def _normalize_rows(
        frame: int,
        label: int,
        rows: tuple[tuple[str, np.ndarray], ...],
        first_touch: int,
    ) -> tuple[_PendingRow, ...]:
        Browse1DCache._validate_frame_label(frame, label)
        if type(rows) is not tuple or not rows:
            raise TypeError("Browse 1-D rows must be a nonempty exact tuple")
        names: set[str] = set()
        pending: list[_PendingRow] = []
        for offset, item in enumerate(rows, start=1):
            if type(item) is not tuple or len(item) != 2:
                raise TypeError("Browse 1-D row item must be a pair")
            name, array = item
            key = Browse1DRowKey(frame, label, name)
            if name in names:
                raise ValueError("Browse 1-D row name is duplicated")
            if type(array) is not np.ndarray:
                raise TypeError("Browse 1-D row must be an exact ndarray")
            if array.ndim != 1 or array.dtype.kind not in {"i", "u", "f", "c"}:
                raise ValueError("Browse 1-D row must be numeric and 1-D")
            if not array.flags.c_contiguous:
                raise ValueError("Browse 1-D row must be C-contiguous")
            if array.flags.writeable:
                raise ValueError("Browse 1-D row must already be read-only")
            fact = physical_root_fact(array)
            if isinstance(fact.root, np.ndarray):
                if fact.root.ndim != 1:
                    raise ValueError("Browse 1-D row cannot retain a wide root")
                if fact.root.flags.writeable:
                    raise ValueError(
                        "Browse 1-D physical root must be read-only"
                    )
            elif type(fact.root) is not bytes:
                raise ValueError("Browse 1-D physical root must be immutable")
            row_identity = object()
            pending.append(
                _PendingRow(
                    key,
                    array,
                    fact,
                    row_identity,
                    _RowSemantic(row_identity),
                    first_touch + offset,
                )
            )
            names.add(name)
        return tuple(pending)


    @staticmethod
    def _projected_unique_bytes(
        residents: tuple[_ResidentRow, ...],
        pending: tuple[_PendingRow, ...],
    ) -> int:
        unique: dict[int, tuple[object, int]] = {}
        for row in (*residents, *pending):
            identity = id(row.fact.root)
            prior = unique.get(identity)
            if prior is not None and (
                prior[0] is not row.fact.root or prior[1] != row.fact.nbytes
            ):
                raise ValueError("Browse 1-D physical-root identity changed")
            unique[identity] = (row.fact.root, row.fact.nbytes)
        return sum(item[1] for item in unique.values())


    @staticmethod
    def _borrowed_identities(state: _CacheState) -> set[int]:
        return {id(binding.row_identity) for binding in state.borrows}


    def _plan_rows(
        self,
        state: _CacheState,
        pending: tuple[_PendingRow, ...],
        protected_rows: tuple[_ResidentRow, ...] = (),
    ) -> tuple[tuple[_ResidentRow, ...], tuple[_ResidentRow, ...]]:
        incoming_keys = {row.key for row in pending}
        borrowed = self._borrowed_identities(state)
        protected: set[int] = set()
        if protected_rows:
            protected = {id(row) for row in protected_rows}
            resident_identities = {id(row): row for row in state.rows}
            if len(protected) != len(protected_rows) or any(
                resident_identities.get(id(row)) is not row
                for row in protected_rows
            ):
                raise RuntimeError("protected Browse 1-D rows changed")
        mandatory_rows: list[_ResidentRow] = []
        survivor_rows: list[_ResidentRow] = []
        replaces_protected = False
        replaces_borrowed = False
        for row in state.rows:
            if row.key not in incoming_keys:
                survivor_rows.append(row)
                continue
            if id(row) in protected:
                replaces_protected = True
            if id(row.row_identity) in borrowed:
                replaces_borrowed = True
            mandatory_rows.append(row)
        if replaces_protected:
            raise ValueError("protected Browse 1-D row cannot be replaced")
        if replaces_borrowed:
            raise RuntimeError("borrowed Browse 1-D row cannot be replaced")
        mandatory = tuple(mandatory_rows)
        survivors = tuple(survivor_rows)
        row_count = len(survivors) + len(pending)
        eligible: list[tuple[int, int, _ResidentRow]] | None = None
        evicted: list[_ResidentRow] = []
        evicted_identities: set[int] = set()
        if row_count > _MAX_BROWSE_1D_RESIDENT_ROWS:
            eligible = [
                (row.touch, position, row)
                for position, row in enumerate(survivors)
                if id(row.row_identity) not in borrowed
                and id(row) not in protected
            ]
            heapq.heapify(eligible)
            while row_count > _MAX_BROWSE_1D_RESIDENT_ROWS:
                if not eligible:
                    raise ValueError(
                        "Browse 1-D rows exceed the resident row limit"
                    )
                _touch, _position, oldest = heapq.heappop(eligible)
                evicted.append(oldest)
                evicted_identities.add(id(oldest))
                row_count -= 1

        remaining = tuple(
            row for row in survivors if id(row) not in evicted_identities
        )
        roots: dict[int, tuple[object, int, int]] = {}
        projected_bytes = 0
        for rows in (remaining, pending):
            for row in rows:
                identity = id(row.fact.root)
                prior = roots.get(identity)
                if prior is None:
                    roots[identity] = (row.fact.root, row.fact.nbytes, 1)
                    projected_bytes += row.fact.nbytes
                else:
                    root, nbytes, references = prior
                    if root is not row.fact.root or nbytes != row.fact.nbytes:
                        raise ValueError(
                            "Browse 1-D physical-root identity changed"
                        )
                    roots[identity] = (root, nbytes, references + 1)

        if projected_bytes <= self._budget:
            return remaining, mandatory + tuple(evicted)

        if eligible is None:
            eligible = [
                (row.touch, position, row)
                for position, row in enumerate(survivors)
                if id(row.row_identity) not in borrowed
                and id(row) not in protected
            ]
            heapq.heapify(eligible)
        while projected_bytes > self._budget:
            if not eligible:
                raise ValueError("Browse 1-D rows exceed the cache budget")
            _touch, _position, oldest = heapq.heappop(eligible)
            evicted.append(oldest)
            evicted_identities.add(id(oldest))
            identity = id(oldest.fact.root)
            root, nbytes, references = roots[identity]
            if references == 1:
                del roots[identity]
                projected_bytes -= nbytes
            else:
                roots[identity] = (root, nbytes, references - 1)
        return (
            tuple(row for row in remaining if id(row) not in evicted_identities),
            mandatory + tuple(evicted),
        )


    @staticmethod
    def _validated_label_rows(
        catalog: "FrameScalarCatalog",
        rows: "Frame1DRows",
        ordinal: int,
        label: int,
    ) -> tuple[tuple[str, np.ndarray], ...]:
        """Qualify one scalar/array projection before reserving cache capacity."""

        # Local import keeps the cache nucleus independent of HDF5 until this
        # FrameView-specific admission boundary is actually used.
        from xrd_tools.io.frame_view import (
            Frame1DModeRows,
            Frame1DRows,
            FrameScalarCatalog,
            FrameScalarRow,
        )

        if type(catalog) is not FrameScalarCatalog:
            raise TypeError("catalog must be an exact FrameScalarCatalog")
        if type(rows) is not Frame1DRows:
            raise TypeError("rows must be exact Frame1DRows")
        if type(ordinal) is not int or ordinal < 1:
            raise TypeError("ordinal must be a positive exact integer")
        if type(label) is not int or label < 0:
            raise TypeError("label must be a nonnegative exact integer")
        if ordinal > len(catalog.labels) or catalog.labels[ordinal - 1] != label:
            raise ValueError("catalog ordinal and label are not exact")
        if (
            rows.artifact_path != catalog.artifact_path
            or rows.entry != catalog.entry
        ):
            raise ValueError("catalog and 1-D rows identify different artifacts")
        if label not in rows.labels:
            raise ValueError("1-D rows do not contain the requested label")

        scalar_row = catalog.rows[ordinal - 1]
        if type(scalar_row) is not FrameScalarRow or scalar_row.label != label:
            raise ValueError("catalog row identity is not exact")
        descriptors = catalog.axes_1d
        result_modes = rows.modes
        if (
            type(descriptors) is not tuple
            or type(result_modes) is not tuple
            or len(descriptors) != len(result_modes)
        ):
            raise ValueError("catalog and row mode inventories differ")
        if any(
            type(descriptor) is not tuple
            or len(descriptor) != 4
            or type(descriptor[0]) is not str
            or not descriptor[0]
            or type(descriptor[1]) is not str
            or not descriptor[1]
            or type(descriptor[2]) is not str
            or type(descriptor[3]) is not bool
            for descriptor in descriptors
        ):
            raise TypeError("catalog axis descriptors are invalid")
        if any(type(mode_rows) is not Frame1DModeRows for mode_rows in result_modes):
            raise TypeError("1-D modes must be exact Frame1DModeRows")
        descriptor_modes = tuple(descriptor[0] for descriptor in descriptors)
        if tuple(mode_rows.mode for mode_rows in result_modes) != descriptor_modes:
            raise ValueError("catalog and row mode ordering differs")
        expected_modes = tuple(
            mode for mode in descriptor_modes if mode in scalar_row.modes_1d
        )
        if expected_modes != scalar_row.modes_1d:
            raise ValueError("catalog label mode ordering is not exact")

        named: list[tuple[str, np.ndarray]] = []
        for descriptor, mode_rows in zip(
            descriptors, result_modes, strict=True,
        ):
            mode, axis_label, axis_unit, axis_log = descriptor
            axis = mode_rows.axis
            axis_values = axis.values
            if (
                mode_rows.mode != mode
                or type(axis.label) is not str
                or axis.label != axis_label
                or type(axis.unit) is not str
                or axis.unit != axis_unit
                or type(axis.log) is not bool
                or axis.log is not axis_log
                or type(axis_values) is not np.ndarray
                or axis_values.dtype != np.dtype(np.float64)
                or axis_values.ndim != 1
                or not axis_values.flags.c_contiguous
                or axis_values.flags.writeable
            ):
                raise ValueError("1-D mode axis does not match its catalog descriptor")

            result = mode_rows.row(label)
            expected = mode in scalar_row.modes_1d
            if not expected:
                if result is not None:
                    raise ValueError("1-D rows contain an unexpected label mode")
                continue
            if result is None:
                raise ValueError("1-D rows omit a catalog-present label mode")
            intensity, sigma = result
            points = axis_values.shape
            components = (("intensity", intensity),)
            if sigma is not None:
                components += (("sigma", sigma),)
            for component, array in components:
                if (
                    type(array) is not np.ndarray
                    or array.dtype != np.dtype(np.float64)
                    or array.ndim != 1
                    or array.shape != points
                    or not array.flags.c_contiguous
                    or array.flags.writeable
                ):
                    raise ValueError(
                        f"1-D {component} row is not exactly aligned"
                    )
            named.append((browse_1d_row_name(mode, "axis"), axis_values))
            named.append((browse_1d_row_name(mode, "intensity"), intensity))
            if sigma is not None:
                named.append((browse_1d_row_name(mode, "sigma"), sigma))
        return tuple(named)


    def record_scalars(self, frame: int, label: int,
                       scalars: tuple[tuple[str, object], ...]) -> None:
        self._validate_frame_label(frame, label)
        record = _CatalogRecord(frame, label, self._normalize_scalars(scalars))
        self._begin_work()
        try:
            self._settle_pending()
            while True:
                state = self._snapshot_state()
                catalog = tuple(item for item in state.catalog
                                if (item.frame, item.label) != (frame, label)) + (record,)
                if self._cas_state(state, replace(state, catalog=catalog)):
                    return
        finally:
            self._end_work()

    def scalars(self, frame: int, label: int) -> Mapping[str, object]:
        self._validate_frame_label(frame, label)
        for record in self._read_state().catalog:
            if (record.frame, record.label) == (frame, label):
                return MappingProxyType(dict(record.scalars))
        return MappingProxyType({})

    def _settle_pending(self) -> None:
        pending = self._pending
        if pending is None:
            return
        if type(pending) is PhysicalRootReservation:
            pending.rollback()
        else:
            failed = []
            first_error = None
            for lease in pending:
                try:
                    lease.release()
                except BaseException as error:
                    failed.append(lease)
                    if first_error is None:
                        first_error = error
            self._pending = tuple(failed) or None
            if first_error is not None:
                raise first_error
        self._pending = None

    def _store_rows(self, frame: int, label: int,
                    rows: tuple[tuple[str, np.ndarray], ...], *,
                    protected: tuple[_ResidentRow, ...] = (),
                    cancelled: Callable[[], bool] | None = None) -> str:
        if cancelled is not None and cancelled():
            raise InterruptedError("Browse 1-D cache store is stale")
        state = self._snapshot_state()
        normalized = self._normalize_rows(frame, label, rows, state.clock)
        survivors, victims = self._plan_rows(state, normalized, protected)
        self._pending = tuple(row.lease for row in victims)
        next_clock = state.clock + len(normalized)
        # New borrows/edits are gated while work is active. Borrow releases may
        # still advance membership, so keep their latest token set at the swap.
        while True:
            current = self._snapshot_state()
            replacement = replace(current, rows=survivors, clock=next_clock)
            if self._cas_state(current, replacement):
                break
        # D2: evicted rows are gone before their charge is freed. No prior
        # membership snapshot is kept for restoration or error history.
        del state, current, victims
        self._settle_pending()
        reservation = self._authority.reserve()
        self._pending = reservation
        incoming = ()
        row = None
        try:
            for row in normalized:
                reservation.reserve(row.array, row.semantic)
            leases = reservation.commit()
            incoming = tuple(
                _ResidentRow(row.key, row.array, row.fact, row.row_identity,
                             row.semantic, leases[row.semantic], row.touch)
                for row in normalized
            )
            if cancelled is not None and cancelled():
                raise InterruptedError("Browse 1-D cache store is stale")
            while True:
                current = self._snapshot_state()
                replacement = replace(current, rows=survivors + incoming)
                if self._cas_state(current, replacement):
                    break
            self._pending = None
            return "accepted"
        except BaseException as primary:
            # Caller-owned input arrays are outside this cache's membership.
            # Drop every private incoming array reference before lease cleanup.
            rows = normalized = incoming = ()
            row = None
            try:
                self._settle_pending()
            except BaseException as cleanup:
                primary.add_note(f"Browse 1-D cleanup remains pending: {cleanup}")
            raise

    def store_rows(self, frame: int, label: int,
                   rows: tuple[tuple[str, np.ndarray], ...], *,
                   _protected_rows: tuple[_ResidentRow, ...] = (),
                   cancelled: Callable[[], bool] | None = None) -> str:
        self._begin_work()
        try:
            self._settle_pending()
            return self._store_rows(frame, label, rows,
                                    protected=_protected_rows, cancelled=cancelled)
        finally:
            rows = ()
            self._end_work()

    def store_1d_label(self, catalog: "FrameScalarCatalog", rows: "Frame1DRows",
                       ordinal: int, label: int, *,
                       cancelled: Callable[[], bool] | None = None) -> tuple[Browse1DRowKey, ...]:
        named = self._validated_label_rows(catalog, rows, ordinal, label)
        keys = tuple(Browse1DRowKey(ordinal, label, name) for name, _ in named)
        self._begin_work()
        try:
            self._settle_pending()
            if cancelled is not None and cancelled():
                raise InterruptedError("Browse 1-D cache store is stale")
            state = self._snapshot_state()
            existing = tuple(row for row in state.rows
                             if row.key.frame == ordinal and row.key.label == label)
            names = frozenset(row.key.name for row in existing)
            incoming = dict(named)
            if names.difference(incoming):
                raise ValueError("Browse 1-D resident label inventory has extra rows")
            for resident in existing:
                expected = incoming[resident.key.name]
                fact = physical_root_fact(expected)
                if (resident.array is not expected or resident.fact.root is not fact.root
                        or resident.fact.nbytes != fact.nbytes or resident.lease.released):
                    raise ValueError("Browse 1-D resident row conflicts with projection")
            missing = tuple(item for item in named if item[0] not in names)
            del state
            if missing:
                self._store_rows(ordinal, label, missing,
                                 protected=existing, cancelled=cancelled)
            return keys
        finally:
            rows = None
            named = ()
            self._end_work()

    def borrow(self, frame: int, label: int, name: str) -> Browse1DBorrow:
        key = Browse1DRowKey(frame, label, name)
        while True:
            state = self._read_state()
            row = next((row for row in state.rows if row.key == key), None)
            if row is None:
                raise KeyError(key)
            token = _BorrowToken(row.row_identity)
            touched = replace(row, touch=state.clock + 1)
            rows = tuple(touched if item is row else item for item in state.rows)
            borrows = state.borrows + (_BorrowBinding(token, row.row_identity),)
            borrowed = Browse1DBorrow(self, token, row)
            replacement = replace(state, rows=rows, borrows=borrows, clock=state.clock + 1)
            with self._lock:
                if self._busy or self._pending is not None:
                    raise RuntimeError("Browse 1-D cache reads are gated")
                if self._state is state:
                    self._state = replacement
                    return borrowed

    def _release_borrow(self, token: _BorrowToken, row_identity: object) -> None:
        while True:
            state = self._snapshot_state()
            binding = next((item for item in state.borrows
                            if item.token is token and item.row_identity is row_identity), None)
            if binding is None:
                raise RuntimeError("Browse 1-D borrow is not owned")
            borrows = tuple(item for item in state.borrows if item is not binding)
            if self._cas_state(state, replace(state, borrows=borrows)):
                return

    def retry_cleanup(self) -> None:
        self._begin_work(closing=True)
        try:
            self._settle_pending()
        finally:
            self._end_work()
        if self.phase is Browse1DCachePhase.CLOSING:
            self.close()

    def close(self) -> None:
        self._begin_work(closing=True)
        try:
            state = self._snapshot_state()
            if state.phase is Browse1DCachePhase.CLOSED:
                return
            if state.borrows:
                raise RuntimeError("Browse 1-D cache has outstanding borrows")
            # Settle any actual failed fill before detaching surviving rows.
            # A failure keeps those rows and their real leases owned for retry.
            self._settle_pending()
            while True:
                state = self._snapshot_state()
                closing = replace(state, phase=Browse1DCachePhase.CLOSING,
                                  rows=(), catalog=())
                leases = tuple(row.lease for row in state.rows)
                if self._cas_state(state, closing):
                    break
            self._pending = leases
            del state
            self._settle_pending()
            self._authority.close()
            closed = replace(closing, phase=Browse1DCachePhase.CLOSED)
            self._cas_state(closing, closed)
        finally:
            self._end_work()


__all__ = [
    "Browse1DBorrow", "Browse1DCache", "Browse1DCachePhase", "Browse1DRowKey",
    "browse_1d_row_name", "default_browse_1d_cache_budget",
]
