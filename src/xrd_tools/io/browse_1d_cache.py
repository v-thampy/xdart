"""Bounded, failure-atomic cache nucleus for Browse 1-D rows.

The cache owns only complete named one-dimensional rows.  It deliberately has
no HDF5, Qt, renderer, record-publication, or wide-array fallback dependency.
All cache coordination is one immutable state-pointer CAS.  The cache lock is
used only to snapshot or swap that pointer; physical-memory exchanges and all
object callbacks/decrefs run outside it.

The configured byte limit applies to the final unique physical roots retained
by :class:`PhysicalRootAuthority`.  Exchange staging can transiently retain an
old and incoming graph, as documented by the authority itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
from threading import Lock
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping
from weakref import ReferenceType, ref

import numpy as np

from xrd_tools.core.physical_memory import (
    PhysicalRootAuthority,
    PhysicalRootExchange,
    PhysicalRootExchangePhase,
    PhysicalRootFact,
    PhysicalRootLease,
    physical_root_fact,
)


_GIB = 1 << 30
_MAX_ROW_NAME_BYTES = 1 << 10
_MAX_BROWSE_1D_RESIDENT_ROWS = 4096
_ROW_NAME_PREFIX = "browse-1d-row-v1:"
_ROW_COMPONENTS = frozenset({"axis", "intensity", "sigma"})

if TYPE_CHECKING:
    from xrd_tools.io.frame_view import Frame1DRows, FrameScalarCatalog


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
    PREPARING = "preparing"
    ROLLBACK_PENDING = "rollback-pending"
    STAGED_PENDING = "staged-pending"
    COMMITTING = "committing"
    ACCEPT_PENDING = "accept-pending"
    ACCEPTING = "accepting"
    ROLLBACK_READY = "rollback-ready"
    ROLLING_BACK = "rolling-back"
    CLOSING_AUTHORITY = "closing-authority"
    BLOCKED = "blocked"
    CLOSED = "closed"


class _TerminalDirection(Enum):
    ACCEPTED = "accepted"
    ROLLED_BACK = "rolled-back"
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
class _BorrowToken:
    row_identity: object


@dataclass(frozen=True, slots=True, eq=False)
class _BorrowBinding:
    token: _BorrowToken
    row_identity: object


@dataclass(frozen=True, slots=True, eq=False)
class _CatalogRecord:
    frame: int
    label: int
    scalars: tuple[tuple[str, object], ...]


@dataclass(frozen=True, slots=True, eq=False, weakref_slot=True)
class _OperationMarker:
    pass


@dataclass(frozen=True, slots=True, eq=False, weakref_slot=True)
class _DriverOwner:
    pass


@dataclass(frozen=True, slots=True, eq=False)
class _Driver:
    token: object
    marker: _OperationMarker
    owner: ReferenceType[_DriverOwner]


class _DriverClaim:
    """Stack-owned strong liveness for one immutable driver state."""

    __slots__ = ("state", "driver", "_owner")

    def __init__(
        self,
        state: "_CacheState",
        driver: _Driver,
        owner: _DriverOwner,
    ) -> None:
        self.state = state
        self.driver = driver
        self._owner: _DriverOwner | None = owner

    def release(self) -> None:
        self._owner = None


@dataclass(frozen=True, slots=True, eq=False)
class _CacheJournal:
    marker: _OperationMarker
    exchange: PhysicalRootExchange
    prior_rows: tuple[_ResidentRow, ...]
    survivor_rows: tuple[_ResidentRow, ...]
    pending_rows: tuple[_PendingRow, ...]
    staged_rows: tuple[_ResidentRow, ...] | None
    closing: bool


@dataclass(frozen=True, slots=True, eq=False)
class _CacheState:
    generation: int
    phase: Browse1DCachePhase
    rows: tuple[_ResidentRow, ...]
    catalog: tuple[_CatalogRecord, ...]
    borrows: tuple[_BorrowBinding, ...]
    clock: int
    journal: _CacheJournal | None
    driver: _Driver | None
    terminal_evidence: tuple[ReferenceType[_OperationMarker], ...]
    terminal_directions: tuple[_TerminalDirection, ...]


def _marker_is_present(state: _CacheState, marker: _OperationMarker) -> bool:
    return _marker_direction(state, marker) is not None


def _marker_direction(
    state: _CacheState,
    marker: _OperationMarker,
) -> _TerminalDirection | None:
    if len(state.terminal_evidence) != len(state.terminal_directions):
        raise RuntimeError("Browse 1-D terminal evidence is inconsistent")
    for marker_ref, direction in zip(
        state.terminal_evidence,
        state.terminal_directions,
        strict=True,
    ):
        if marker_ref() is marker:
            return direction
    return None


def _terminal_evidence_with(
    state: _CacheState,
    marker: _OperationMarker,
    direction: _TerminalDirection,
) -> tuple[
    tuple[ReferenceType[_OperationMarker], ...],
    tuple[_TerminalDirection, ...],
]:
    """Prune dead markers and append one live marker by identity only."""

    refs: list[ReferenceType[_OperationMarker]] = []
    live: list[_OperationMarker] = []
    directions: list[_TerminalDirection] = []
    found_direction: _TerminalDirection | None = None
    if len(state.terminal_evidence) != len(state.terminal_directions):
        raise RuntimeError("Browse 1-D terminal evidence is inconsistent")
    for marker_ref, prior_direction in zip(
        state.terminal_evidence,
        state.terminal_directions,
        strict=True,
    ):
        candidate = marker_ref()
        if candidate is None:
            continue
        if any(candidate is previous for previous in live):
            continue
        live.append(candidate)
        refs.append(marker_ref)
        directions.append(prior_direction)
        if candidate is marker:
            found_direction = prior_direction
    if found_direction is not None and found_direction is not direction:
        raise RuntimeError("Browse 1-D terminal direction changed")
    if found_direction is None:
        refs.append(ref(marker))
        directions.append(direction)
    return tuple(refs), tuple(directions)


def _state_with(
    state: _CacheState,
    *,
    phase: Browse1DCachePhase | None = None,
    rows: tuple[_ResidentRow, ...] | None = None,
    catalog: tuple[_CatalogRecord, ...] | None = None,
    borrows: tuple[_BorrowBinding, ...] | None = None,
    clock: int | None = None,
    journal: _CacheJournal | None = None,
    replace_journal: bool = False,
    driver: _Driver | None = None,
    replace_driver: bool = False,
    terminal_evidence: tuple[ReferenceType[_OperationMarker], ...] | None = None,
    terminal_directions: tuple[_TerminalDirection, ...] | None = None,
) -> _CacheState:
    return _CacheState(
        state.generation + 1,
        state.phase if phase is None else phase,
        state.rows if rows is None else rows,
        state.catalog if catalog is None else catalog,
        state.borrows if borrows is None else borrows,
        state.clock if clock is None else clock,
        journal if replace_journal else state.journal,
        driver if replace_driver else state.driver,
        (
            state.terminal_evidence
            if terminal_evidence is None
            else terminal_evidence
        ),
        (
            state.terminal_directions
            if terminal_directions is None
            else terminal_directions
        ),
    )


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


class Browse1DCacheOperation(_LinearObject):
    """Recoverable handle for one cache exchange journal."""

    __slots__ = ("_cache", "_marker", "_intent", "_terminal")

    def __init__(
        self, cache: "Browse1DCache", marker: _OperationMarker,
    ) -> None:
        self._cache = cache
        self._marker: _OperationMarker | None = marker
        self._intent: _TerminalDirection | None = None
        self._terminal: _TerminalDirection | None = None

    @property
    def terminal_direction(self) -> str | None:
        if self._terminal is not None:
            return self._terminal.value
        marker = self._marker
        if marker is None:
            return None
        state = self._cache._snapshot_state()
        direction = _marker_direction(state, marker)
        return None if direction is None else direction.value

    def run(self) -> str:
        return self._cache._drive_operation(self)

    def recover(self) -> str:
        return self.run()

    def rollback(self) -> str:
        return self._cache._request_rollback(self)

    def _classify(self, direction: _TerminalDirection) -> str:
        if self._terminal is not None and self._terminal is not direction:
            raise RuntimeError("Browse 1-D operation terminal direction changed")
        self._terminal = direction
        self._marker = None
        return direction.value


class Browse1DLabelStoreCustodyError(RuntimeError):
    """Receipt-publication failure retaining its exact unsettled operation."""

    __slots__ = ("_operation", "_receipt_error", "_rollback_error")

    def __init__(
        self,
        operation: Browse1DCacheOperation,
        receipt_error: BaseException,
        rollback_error: BaseException,
    ) -> None:
        if type(operation) is not Browse1DCacheOperation:
            raise TypeError("Browse 1-D custody requires an exact operation")
        super().__init__(
            "Browse 1-D receipt failed and exact operation cleanup is pending"
        )
        self._operation = operation
        self._receipt_error = receipt_error
        self._rollback_error = rollback_error

    @property
    def operation(self) -> Browse1DCacheOperation:
        return self._operation

    @property
    def receipt_error(self) -> BaseException:
        return self._receipt_error

    @property
    def rollback_error(self) -> BaseException:
        return self._rollback_error


@dataclass(frozen=True, slots=True, eq=False)
class Browse1DLabelStoreReceipt:
    """Immutable admission receipt for one persisted frame-label row bundle.

    A nonempty receipt exposes the exact linear cache operation; callers own
    its explicit ``run``/``recover``/``rollback`` settlement.  An empty label
    is already complete and deliberately owns no cache journal.
    """

    ordinal: int
    label: int
    keys: tuple[Browse1DRowKey, ...]
    operation: Browse1DCacheOperation | None
    complete_empty: bool = False
    already_complete: bool = False

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal < 1:
            raise TypeError("Browse 1-D ordinal must be a positive exact integer")
        if type(self.label) is not int or self.label < 0:
            raise TypeError("Browse 1-D label must be a nonnegative exact integer")
        if (
            type(self.keys) is not tuple
            or any(
                type(key) is not Browse1DRowKey
                or key.frame != self.ordinal
                or key.label != self.label
                for key in self.keys
            )
        ):
            raise TypeError("Browse 1-D receipt keys are invalid")
        if (
            type(self.complete_empty) is not bool
            or type(self.already_complete) is not bool
        ):
            raise TypeError("Browse 1-D empty completion must be exact")
        if self.complete_empty and self.already_complete:
            raise ValueError("Browse 1-D receipt has conflicting completion")
        if self.complete_empty:
            if self.keys or self.operation is not None:
                raise ValueError("Empty Browse 1-D receipt cannot own a journal")
        elif self.already_complete:
            if not self.keys or self.operation is not None:
                raise ValueError(
                    "Complete Browse 1-D receipt cannot own a journal"
                )
        elif not self.keys or type(self.operation) is not Browse1DCacheOperation:
            raise ValueError("Nonempty Browse 1-D receipt requires one operation")


class Browse1DCache(_LinearObject):
    """Failure-atomic bounded cache of indivisible named 1-D rows."""

    def __init__(
        self,
        budget_bytes: int | None = None,
        *,
        physical_ram_bytes: int | None = None,
    ) -> None:
        if budget_bytes is None:
            budget_bytes = default_browse_1d_cache_budget(physical_ram_bytes)
        if type(budget_bytes) is not int or budget_bytes < 0:
            raise TypeError("Browse 1-D cache budget must be nonnegative")
        self._budget = budget_bytes
        self._authority = PhysicalRootAuthority(budget_bytes)
        self._lock = Lock()
        self._state = _CacheState(
            0,
            Browse1DCachePhase.OPEN,
            (),
            (),
            (),
            0,
            None,
            None,
            (),
            (),
        )

    @property
    def budget_bytes(self) -> int:
        return self._budget

    @property
    def phase(self) -> Browse1DCachePhase:
        return self._snapshot_state().phase

    @property
    def resident_bytes(self) -> int:
        self._require_open(self._snapshot_state())
        return self._authority.retained_bytes

    @property
    def resident_root_count(self) -> int:
        self._require_open(self._snapshot_state())
        return self._authority.retained_root_count

    @property
    def resident_keys(self) -> tuple[Browse1DRowKey, ...]:
        state = self._snapshot_state()
        self._require_open(state)
        return tuple(row.key for row in state.rows)

    @property
    def outstanding_borrows(self) -> int:
        state = self._snapshot_state()
        self._require_open(state)
        return len(state.borrows)

    def _snapshot_state(self) -> _CacheState:
        with self._lock:
            return self._state

    def _cas_state(self, expected: _CacheState, replacement: _CacheState) -> bool:
        """Exact pointer CAS with no callbacks and no overlapping owner lock."""

        with self._lock:
            if self._state is not expected:
                return False
            self._state = replacement
            return True

    def _transition(
        self, expected: _CacheState, replacement: _CacheState,
    ) -> bool:
        try:
            swapped = self._cas_state(expected, replacement)
        except BaseException as error:
            current = self._snapshot_state()
            if current is expected or current is replacement:
                raise
            raise RuntimeError("Browse 1-D cache state drifted") from error
        if swapped:
            return True
        current = self._snapshot_state()
        if current is replacement:
            return True
        if current is expected:
            return False
        if (
            expected.journal is not None
            and _marker_is_present(current, expected.journal.marker)
        ):
            # The same operation reached a durable terminal state first.
            return False
        if (
            expected.journal is not None
            and current.journal is not None
            and current.journal.marker is expected.journal.marker
        ):
            # A same-operation helper won a documented state arbitration.
            return False
        if (
            expected.phase is Browse1DCachePhase.OPEN
            and current.phase is Browse1DCachePhase.OPEN
            and expected.journal is None
            and current.journal is None
        ):
            # A concurrent scalar/borrow COW won; the caller may recompute.
            return False
        raise RuntimeError("Browse 1-D cache state drifted")

    def _install_open_cow(
        self,
        expected: _CacheState,
        replacement: _CacheState,
        token: _BorrowToken,
        row_identity: object,
        *,
        installing: bool,
    ) -> bool:
        """Install a handle-bearing OPEN COW without orphaning its receipt.

        Borrow admission/release cannot publish a binding and then propagate a
        post-swap interruption: admission would lose the only release token,
        while release would leave its public handle falsely live.  Therefore
        an exact observed replacement, or any later cache state retaining its
        exact token effect, self-heals as success.  Pre-swap interruptions
        still propagate unchanged.
        """

        def effect_is_present(state: _CacheState) -> bool:
            present = any(
                binding.token is token
                and binding.row_identity is row_identity
                for binding in state.borrows
            )
            return present if installing else not present

        try:
            swapped = self._cas_state(expected, replacement)
        except BaseException as error:
            current = self._snapshot_state()
            if current is replacement or effect_is_present(current):
                return True
            if current is expected:
                raise
            raise RuntimeError("Browse 1-D cache state drifted") from error
        if swapped:
            return True
        current = self._snapshot_state()
        if current is replacement or effect_is_present(current):
            return True
        if current is expected:
            return False
        if (
            current.phase is Browse1DCachePhase.OPEN
            and current.journal is None
            and current.driver is None
        ):
            return False
        raise RuntimeError("Browse 1-D cache state drifted")

    @staticmethod
    def _require_open(state: _CacheState) -> None:
        if state.phase is Browse1DCachePhase.CLOSED:
            raise RuntimeError("Browse 1-D cache is closed")
        if state.phase is not Browse1DCachePhase.OPEN:
            raise RuntimeError("Browse 1-D cache reads are gated")

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

    def record_scalars(
        self,
        frame: int,
        label: int,
        scalars: tuple[tuple[str, object], ...],
    ) -> None:
        self._validate_frame_label(frame, label)
        normalized = self._normalize_scalars(scalars)
        while True:
            state = self._snapshot_state()
            self._require_open(state)
            catalog = tuple(
                record
                for record in state.catalog
                if (record.frame, record.label) != (frame, label)
            ) + (_CatalogRecord(frame, label, normalized),)
            replacement = _state_with(state, catalog=catalog)
            if self._transition(state, replacement):
                return

    def scalars(self, frame: int, label: int) -> Mapping[str, object]:
        self._validate_frame_label(frame, label)
        state = self._snapshot_state()
        self._require_open(state)
        for record in state.catalog:
            if record.frame == frame and record.label == label:
                return MappingProxyType(dict(record.scalars))
        return MappingProxyType({})

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
        protected = {id(row) for row in protected_rows}
        if len(protected) != len(protected_rows) or any(
            not any(row is resident for resident in state.rows)
            for row in protected_rows
        ):
            raise RuntimeError("protected Browse 1-D rows changed")
        mandatory = tuple(row for row in state.rows if row.key in incoming_keys)
        if any(id(row) in protected for row in mandatory):
            raise ValueError("protected Browse 1-D row cannot be replaced")
        if any(id(row.row_identity) in borrowed for row in mandatory):
            raise RuntimeError("borrowed Browse 1-D row cannot be replaced")
        survivors = [row for row in state.rows if row.key not in incoming_keys]
        victims = list(mandatory)
        while (
            len(survivors) + len(pending)
            > _MAX_BROWSE_1D_RESIDENT_ROWS
            or self._projected_unique_bytes(tuple(survivors), pending)
            > self._budget
        ):
            row_limit_exceeded = (
                len(survivors) + len(pending)
                > _MAX_BROWSE_1D_RESIDENT_ROWS
            )
            eligible = [
                row
                for row in survivors
                if id(row.row_identity) not in borrowed
                and id(row) not in protected
            ]
            if not eligible:
                if row_limit_exceeded:
                    raise ValueError(
                        "Browse 1-D rows exceed the resident row limit"
                    )
                raise ValueError("Browse 1-D rows exceed the cache budget")
            oldest = min(eligible, key=lambda item: item.touch)
            survivors.remove(oldest)
            victims.append(oldest)
        return tuple(survivors), tuple(victims)

    def begin_store(
        self,
        frame: int,
        label: int,
        rows: tuple[tuple[str, np.ndarray], ...],
        *,
        _expected_state: _CacheState | None = None,
        _protected_rows: tuple[_ResidentRow, ...] = (),
    ) -> Browse1DCacheOperation:
        state = (
            self._snapshot_state()
            if _expected_state is None else _expected_state
        )
        if _expected_state is not None and self._snapshot_state() is not state:
            raise RuntimeError("Browse 1-D cache changed before store admission")
        self._require_open(state)
        pending = self._normalize_rows(frame, label, rows, state.clock)
        survivors, victims = self._plan_rows(
            state, pending, _protected_rows,
        )
        exchange = self._authority.exchange(tuple(row.lease for row in victims))
        try:
            for row in pending:
                exchange.reserve(row.array, row.semantic)
        except BaseException:
            exchange.rollback()
            raise
        marker = _OperationMarker()
        journal = _CacheJournal(
            marker,
            exchange,
            state.rows,
            survivors,
            pending,
            None,
            False,
        )
        operation = Browse1DCacheOperation(self, marker)
        preparing = _state_with(
            state,
            phase=Browse1DCachePhase.PREPARING,
            clock=state.clock + len(pending),
            journal=journal,
            replace_journal=True,
            driver=None,
            replace_driver=True,
        )
        try:
            installed = self._transition(state, preparing)
        except BaseException:
            if self._snapshot_state() is not preparing:
                exchange.rollback()
                operation._classify(_TerminalDirection.ROLLED_BACK)
            raise
        if not installed:
            exchange.rollback()
            operation._classify(_TerminalDirection.ROLLED_BACK)
            raise RuntimeError("Browse 1-D prepare transition did not complete")
        return operation

    @staticmethod
    def _validated_label_rows(
        catalog: "FrameScalarCatalog",
        rows: "Frame1DRows",
        ordinal: int,
        label: int,
    ) -> tuple[tuple[str, np.ndarray], ...]:
        """Qualify one scalar/array projection before opening a cache journal."""

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

    def begin_store_1d_label(
        self,
        catalog: "FrameScalarCatalog",
        rows: "Frame1DRows",
        ordinal: int,
        label: int,
    ) -> Browse1DLabelStoreReceipt:
        """Admit all named 1-D rows for one exact catalog ordinal/label.

        Validation is complete before the sole ``begin_store`` call.  The
        returned receipt never drives its operation implicitly, so a failed
        settlement remains available to the caller by exact object identity.
        A compatible resident subset is protected and the one operation owns
        only missing components; resident extras or conflicting components are
        refused before a journal is opened.
        """

        named = self._validated_label_rows(catalog, rows, ordinal, label)
        keys = tuple(
            Browse1DRowKey(ordinal, label, name) for name, _array in named
        )
        incoming = {name: array for name, array in named}
        incoming_names = frozenset(incoming)
        while True:
            state = self._snapshot_state()
            self._require_open(state)
            resident_rows = tuple(
                resident
                for resident in state.rows
                if resident.key.frame == ordinal
                and resident.key.label == label
            )
            resident_names = frozenset(row.key.name for row in resident_rows)
            if len(resident_names) != len(resident_rows):
                raise RuntimeError(
                    "Browse 1-D resident label inventory is duplicated"
                )
            extras = resident_names.difference(incoming_names)
            if extras:
                raise ValueError(
                    "Browse 1-D resident label inventory has extra rows"
                )
            for resident in resident_rows:
                expected = incoming[resident.key.name]
                fact = physical_root_fact(expected)
                if (
                    resident.array is not expected
                    or resident.fact.root is not fact.root
                    or resident.fact.nbytes != fact.nbytes
                    or resident.lease.released
                ):
                    raise ValueError(
                        "Browse 1-D resident row conflicts with projection"
                    )
            missing = tuple(
                item for item in named if item[0] not in resident_names
            )
            if not named:
                if self._snapshot_state() is not state:
                    continue
                return Browse1DLabelStoreReceipt(
                    ordinal, label, (), None, complete_empty=True,
                )
            if not missing:
                if self._snapshot_state() is not state:
                    continue
                return Browse1DLabelStoreReceipt(
                    ordinal, label, keys, None, already_complete=True,
                )
            break

        operation = self.begin_store(
            ordinal,
            label,
            missing,
            _expected_state=state,
            _protected_rows=resident_rows,
        )
        try:
            return Browse1DLabelStoreReceipt(
                ordinal, label, keys, operation,
            )
        except BaseException as receipt_error:
            # No callback follows journal admission except the receipt
            # allocation itself.  If even that fails, settle the exact owned
            # operation instead of leaving a hidden PREPARING journal.
            rollback_error: BaseException | None = None
            try:
                direction = operation.rollback()
            except BaseException as caught:
                rollback_error = caught
            else:
                if direction != _TerminalDirection.ROLLED_BACK.value:
                    rollback_error = RuntimeError(
                        "Browse 1-D receipt cleanup did not roll back"
                    )
            if rollback_error is not None:
                raise Browse1DLabelStoreCustodyError(
                    operation, receipt_error, rollback_error,
                ) from receipt_error
            raise

    def store_rows(
        self,
        frame: int,
        label: int,
        rows: tuple[tuple[str, np.ndarray], ...],
    ) -> str:
        return self.begin_store(frame, label, rows).run()

    def _journal_for(
        self, state: _CacheState, marker: _OperationMarker,
    ) -> _CacheJournal:
        journal = state.journal
        if journal is None or journal.marker is not marker:
            raise RuntimeError("Browse 1-D operation no longer owns the cache")
        return journal

    def _blocked(self, state: _CacheState) -> None:
        replacement = _state_with(
            state,
            phase=Browse1DCachePhase.BLOCKED,
            driver=None,
            replace_driver=True,
        )
        self._transition(state, replacement)

    def _claim_driver(
        self,
        state: _CacheState,
        phase: Browse1DCachePhase,
    ) -> _DriverClaim | None:
        journal = state.journal
        if journal is None:
            raise RuntimeError("Browse 1-D cache journal is missing")
        if state.driver is not None and state.driver.owner() is not None:
            raise RuntimeError("Browse 1-D cache helper is busy")
        owner = _DriverOwner()
        driver = _Driver(object(), journal.marker, ref(owner))
        replacement = _state_with(
            state,
            phase=phase,
            driver=driver,
            replace_driver=True,
        )
        claim = _DriverClaim(replacement, driver, owner)
        try:
            installed = self._transition(state, replacement)
        except BaseException:
            claim.release()
            raise
        if not installed:
            claim.release()
            return None
        return claim

    def _settle_prepare(
        self,
        driven: _CacheState,
        journal: _CacheJournal,
        exchange_phase: PhysicalRootExchangePhase,
    ) -> _CacheState:
        current = self._snapshot_state()
        if current is not driven and not (
            current.phase is Browse1DCachePhase.ROLLBACK_PENDING
            and current.journal is journal
            and current.driver is driven.driver
        ):
            raise RuntimeError("Browse 1-D prepare state drifted")
        rollback_requested = (
            current.phase is Browse1DCachePhase.ROLLBACK_PENDING
        )
        if exchange_phase is PhysicalRootExchangePhase.STAGED:
            leases = journal.exchange.prepared_leases
            incoming = tuple(
                _ResidentRow(
                    row.key,
                    row.array,
                    row.fact,
                    row.row_identity,
                    row.semantic,
                    leases[row.semantic],
                    row.touch,
                )
                for row in journal.pending_rows
            )
            # PREPARING rows are the exact admitted survivors.
            staged_rows = journal.survivor_rows + incoming
            next_journal = _CacheJournal(
                journal.marker,
                journal.exchange,
                journal.prior_rows,
                journal.survivor_rows,
                journal.pending_rows,
                staged_rows,
                journal.closing,
            )
            replacement = _state_with(
                current,
                phase=(
                    Browse1DCachePhase.ROLLBACK_READY
                    if rollback_requested
                    else Browse1DCachePhase.STAGED_PENDING
                ),
                journal=next_journal,
                replace_journal=True,
                driver=None,
                replace_driver=True,
            )
        elif exchange_phase in {
            PhysicalRootExchangePhase.OPEN,
            PhysicalRootExchangePhase.PREPARED,
        }:
            replacement = _state_with(
                current,
                phase=Browse1DCachePhase.ROLLBACK_READY,
                driver=None,
                replace_driver=True,
            )
        else:
            replacement = _state_with(
                current,
                phase=Browse1DCachePhase.BLOCKED,
                driver=None,
                replace_driver=True,
            )
        if not self._transition(current, replacement):
            raise RuntimeError("Browse 1-D prepare settlement did not complete")
        return replacement

    def _drive_prepare(
        self, state: _CacheState, operation: Browse1DCacheOperation,
    ) -> None:
        claimed = self._claim_driver(state, Browse1DCachePhase.PREPARING)
        if claimed is None:
            return
        driven = claimed.state
        journal = self._journal_for(driven, operation._marker)  # type: ignore[arg-type]
        error: BaseException | None = None
        try:
            try:
                journal.exchange.prepare()
            except BaseException as caught:
                error = caught
            phase = journal.exchange.phase
            self._settle_prepare(driven, journal, phase)
        finally:
            claimed.release()
        if error is not None:
            raise error

    def _settle_commit(
        self,
        driven: _CacheState,
        journal: _CacheJournal,
        exchange_phase: PhysicalRootExchangePhase,
    ) -> None:
        current = self._snapshot_state()
        if current is not driven:
            raise RuntimeError("Browse 1-D commit state drifted")
        if exchange_phase in {
            PhysicalRootExchangePhase.COMMIT_PENDING,
            PhysicalRootExchangePhase.ACCEPTED,
        }:
            phase = Browse1DCachePhase.ACCEPT_PENDING
        elif exchange_phase is PhysicalRootExchangePhase.STAGED:
            phase = Browse1DCachePhase.STAGED_PENDING
        else:
            phase = Browse1DCachePhase.BLOCKED
        replacement = _state_with(
            current,
            phase=phase,
            driver=None,
            replace_driver=True,
        )
        if not self._transition(current, replacement):
            raise RuntimeError("Browse 1-D commit settlement did not complete")

    def _drive_commit(
        self, state: _CacheState, operation: Browse1DCacheOperation,
    ) -> None:
        claimed = self._claim_driver(state, Browse1DCachePhase.COMMITTING)
        if claimed is None:
            return
        driven = claimed.state
        journal = self._journal_for(driven, operation._marker)  # type: ignore[arg-type]
        error: BaseException | None = None
        try:
            try:
                journal.exchange.commit()
            except BaseException as caught:
                error = caught
            phase = journal.exchange.phase
            self._settle_commit(driven, journal, phase)
        finally:
            claimed.release()
        if error is not None:
            raise error

    def _terminal_state(
        self,
        state: _CacheState,
        journal: _CacheJournal,
        direction: _TerminalDirection,
    ) -> _CacheState:
        if direction is _TerminalDirection.ACCEPTED:
            rows = journal.staged_rows
            phase = Browse1DCachePhase.OPEN
        elif direction is _TerminalDirection.ROLLED_BACK:
            rows = journal.prior_rows
            phase = Browse1DCachePhase.OPEN
        elif direction is _TerminalDirection.CLOSED:
            rows = ()
            phase = Browse1DCachePhase.CLOSED
        else:  # pragma: no cover - exact enum contract
            raise RuntimeError("Browse 1-D terminal direction is invalid")
        if rows is None:
            raise RuntimeError("Browse 1-D staged rows are unavailable")
        evidence, directions = _terminal_evidence_with(
            state, journal.marker, direction,
        )
        return _state_with(
            state,
            phase=phase,
            rows=rows,
            catalog=(
                ()
                if direction is _TerminalDirection.CLOSED
                else state.catalog
            ),
            borrows=(
                ()
                if direction is _TerminalDirection.CLOSED
                else state.borrows
            ),
            journal=None,
            replace_journal=True,
            driver=None,
            replace_driver=True,
            terminal_evidence=evidence,
            terminal_directions=directions,
        )

    def _install_terminal_state(
        self,
        state: _CacheState,
        journal: _CacheJournal,
        operation: Browse1DCacheOperation,
        direction: _TerminalDirection,
    ) -> None:
        replacement = self._terminal_state(state, journal, direction)
        if not self._transition(state, replacement):
            raise RuntimeError("Browse 1-D terminal transition did not complete")
        operation._classify(direction)

    def _drive_accept(
        self, state: _CacheState, operation: Browse1DCacheOperation,
    ) -> None:
        claimed = self._claim_driver(state, Browse1DCachePhase.ACCEPTING)
        if claimed is None:
            return
        driven = claimed.state
        journal = self._journal_for(driven, operation._marker)  # type: ignore[arg-type]
        error: BaseException | None = None
        try:
            try:
                journal.exchange.accept()
            except BaseException as caught:
                error = caught
            phase = journal.exchange.phase
            current = self._snapshot_state()
            if current is not driven:
                raise RuntimeError("Browse 1-D accept state drifted")
            if phase is PhysicalRootExchangePhase.ACCEPTED:
                if journal.closing:
                    replacement = _state_with(
                        current,
                        phase=Browse1DCachePhase.CLOSING_AUTHORITY,
                        rows=(),
                        driver=None,
                        replace_driver=True,
                    )
                    if not self._transition(current, replacement):
                        raise RuntimeError(
                            "Browse 1-D close transition did not complete"
                        )
                else:
                    self._install_terminal_state(
                        current,
                        journal,
                        operation,
                        _TerminalDirection.ACCEPTED,
                    )
            elif phase is PhysicalRootExchangePhase.COMMIT_PENDING:
                replacement = _state_with(
                    current,
                    phase=Browse1DCachePhase.ACCEPT_PENDING,
                    driver=None,
                    replace_driver=True,
                )
                if not self._transition(current, replacement):
                    raise RuntimeError(
                        "Browse 1-D accept settlement did not complete"
                    )
            else:
                self._blocked(current)
        finally:
            claimed.release()
        if error is not None:
            raise error

    def _drive_rollback(
        self, state: _CacheState, operation: Browse1DCacheOperation,
    ) -> None:
        claimed = self._claim_driver(state, Browse1DCachePhase.ROLLING_BACK)
        if claimed is None:
            return
        driven = claimed.state
        journal = self._journal_for(driven, operation._marker)  # type: ignore[arg-type]
        error: BaseException | None = None
        try:
            try:
                journal.exchange.rollback()
            except BaseException as caught:
                error = caught
            phase = journal.exchange.phase
            current = self._snapshot_state()
            if current is not driven:
                raise RuntimeError("Browse 1-D rollback state drifted")
            if phase is PhysicalRootExchangePhase.ROLLED_BACK:
                self._install_terminal_state(
                    current,
                    journal,
                    operation,
                    _TerminalDirection.ROLLED_BACK,
                )
            elif phase in {
                PhysicalRootExchangePhase.OPEN,
                PhysicalRootExchangePhase.PREPARED,
                PhysicalRootExchangePhase.STAGED,
                PhysicalRootExchangePhase.COMMIT_PENDING,
            }:
                replacement = _state_with(
                    current,
                    phase=Browse1DCachePhase.ROLLBACK_READY,
                    driver=None,
                    replace_driver=True,
                )
                if not self._transition(current, replacement):
                    raise RuntimeError(
                        "Browse 1-D rollback settlement did not complete"
                    )
            else:
                self._blocked(current)
        finally:
            claimed.release()
        if error is not None:
            raise error

    def _drive_close_authority(
        self, state: _CacheState, operation: Browse1DCacheOperation,
    ) -> None:
        claimed = self._claim_driver(
            state, Browse1DCachePhase.CLOSING_AUTHORITY,
        )
        if claimed is None:
            return
        driven = claimed.state
        journal = self._journal_for(driven, operation._marker)  # type: ignore[arg-type]
        try:
            try:
                self._authority.close()
            except BaseException:
                current = self._snapshot_state()
                if current is driven:
                    replacement = _state_with(
                        current,
                        phase=Browse1DCachePhase.CLOSING_AUTHORITY,
                        driver=None,
                        replace_driver=True,
                    )
                    self._transition(current, replacement)
                raise
            current = self._snapshot_state()
            if current is not driven:
                raise RuntimeError("Browse 1-D close state drifted")
            self._install_terminal_state(
                current, journal, operation, _TerminalDirection.CLOSED,
            )
        finally:
            claimed.release()

    def _drive_operation(self, operation: Browse1DCacheOperation) -> str:
        marker = operation._marker
        if marker is None:
            assert operation._terminal is not None
            return operation._terminal.value
        while True:
            state = self._snapshot_state()
            direction = _marker_direction(state, marker)
            if direction is not None:
                return operation._classify(direction)
            journal = self._journal_for(state, marker)
            if state.phase is Browse1DCachePhase.PREPARING:
                self._drive_prepare(state, operation)
            elif state.phase is Browse1DCachePhase.ROLLBACK_PENDING:
                if state.driver is not None and state.driver.owner() is not None:
                    raise RuntimeError("Browse 1-D prepare rollback is pending")
                replacement = _state_with(
                    state,
                    phase=Browse1DCachePhase.ROLLBACK_READY,
                    driver=None,
                    replace_driver=True,
                )
                self._transition(state, replacement)
            elif state.phase is Browse1DCachePhase.STAGED_PENDING:
                self._drive_commit(state, operation)
            elif state.phase is Browse1DCachePhase.ACCEPT_PENDING:
                operation._intent = _TerminalDirection.ACCEPTED
                self._drive_accept(state, operation)
            elif state.phase is Browse1DCachePhase.ROLLBACK_READY:
                operation._intent = _TerminalDirection.ROLLED_BACK
                self._drive_rollback(state, operation)
            elif state.phase is Browse1DCachePhase.CLOSING_AUTHORITY:
                operation._intent = _TerminalDirection.CLOSED
                self._drive_close_authority(state, operation)
            elif state.phase is Browse1DCachePhase.COMMITTING:
                self._drive_commit(state, operation)
            elif state.phase is Browse1DCachePhase.ACCEPTING:
                self._drive_accept(state, operation)
            elif state.phase is Browse1DCachePhase.ROLLING_BACK:
                self._drive_rollback(state, operation)
            elif state.phase is Browse1DCachePhase.BLOCKED:
                raise RuntimeError("Browse 1-D cache journal is blocked")
            else:
                raise RuntimeError("Browse 1-D operation state drifted")

    def _request_rollback(self, operation: Browse1DCacheOperation) -> str:
        marker = operation._marker
        if marker is None:
            assert operation._terminal is not None
            if operation._terminal is not _TerminalDirection.ROLLED_BACK:
                raise RuntimeError("Browse 1-D operation was accepted")
            return operation._terminal.value
        while True:
            state = self._snapshot_state()
            direction = _marker_direction(state, marker)
            if direction is not None:
                if direction is not _TerminalDirection.ROLLED_BACK:
                    raise RuntimeError("Browse 1-D operation was accepted")
                return operation._classify(_TerminalDirection.ROLLED_BACK)
            journal = self._journal_for(state, marker)
            if state.phase is Browse1DCachePhase.PREPARING:
                replacement = _state_with(
                    state,
                    phase=Browse1DCachePhase.ROLLBACK_PENDING,
                )
                if not self._transition(state, replacement):
                    continue
                if state.driver is not None and state.driver.owner() is not None:
                    return Browse1DCachePhase.ROLLBACK_PENDING.value
                ready = _state_with(
                    replacement,
                    phase=Browse1DCachePhase.ROLLBACK_READY,
                    driver=None,
                    replace_driver=True,
                )
                if not self._transition(replacement, ready):
                    continue
                operation._intent = _TerminalDirection.ROLLED_BACK
                return self._drive_operation(operation)
            if state.phase is Browse1DCachePhase.STAGED_PENDING:
                replacement = _state_with(
                    state,
                    phase=Browse1DCachePhase.ROLLBACK_READY,
                )
                if not self._transition(state, replacement):
                    continue
                operation._intent = _TerminalDirection.ROLLED_BACK
                return self._drive_operation(operation)
            if state.phase in {
                Browse1DCachePhase.ROLLBACK_PENDING,
                Browse1DCachePhase.ROLLBACK_READY,
                Browse1DCachePhase.ROLLING_BACK,
            }:
                operation._intent = _TerminalDirection.ROLLED_BACK
                return self._drive_operation(operation)
            if state.phase in {
                Browse1DCachePhase.COMMITTING,
                Browse1DCachePhase.ACCEPT_PENDING,
                Browse1DCachePhase.ACCEPTING,
                Browse1DCachePhase.CLOSING_AUTHORITY,
            }:
                # Commit won the STAGED_PENDING arbitration; follow it.
                operation._intent = _TerminalDirection.ACCEPTED
                return self._drive_operation(operation)
            raise RuntimeError("Browse 1-D rollback state drifted")

    def recover(self) -> str:
        state = self._snapshot_state()
        if state.phase is Browse1DCachePhase.OPEN:
            return Browse1DCachePhase.OPEN.value
        if state.phase is Browse1DCachePhase.CLOSED:
            return Browse1DCachePhase.CLOSED.value
        journal = state.journal
        if journal is None:
            raise RuntimeError("Browse 1-D cache journal is missing")
        operation = Browse1DCacheOperation(self, journal.marker)
        if state.phase in {
            Browse1DCachePhase.PREPARING,
            Browse1DCachePhase.ROLLBACK_PENDING,
        }:
            if journal.closing and state.phase is Browse1DCachePhase.PREPARING:
                operation._intent = _TerminalDirection.CLOSED
                return operation.run()
            return operation.rollback()
        if state.phase in {
            Browse1DCachePhase.ROLLBACK_READY,
            Browse1DCachePhase.ROLLING_BACK,
        }:
            operation._intent = _TerminalDirection.ROLLED_BACK
        elif state.phase is Browse1DCachePhase.CLOSING_AUTHORITY:
            operation._intent = _TerminalDirection.CLOSED
        else:
            operation._intent = _TerminalDirection.ACCEPTED
        return operation.run()

    def borrow(
        self, frame: int, label: int, name: str,
    ) -> Browse1DBorrow:
        key = Browse1DRowKey(frame, label, name)
        while True:
            state = self._snapshot_state()
            self._require_open(state)
            row = next((item for item in state.rows if item.key == key), None)
            if row is None:
                raise KeyError(key)
            token = _BorrowToken(row.row_identity)
            binding = _BorrowBinding(token, row.row_identity)
            clock = state.clock + 1
            rows = tuple(
                (
                    _ResidentRow(
                        item.key,
                        item.array,
                        item.fact,
                        item.row_identity,
                        item.semantic,
                        item.lease,
                        clock,
                    )
                    if item.row_identity is row.row_identity
                    else item
                )
                for item in state.rows
            )
            replacement = _state_with(
                state,
                rows=rows,
                borrows=state.borrows + (binding,),
                clock=clock,
            )
            if self._install_open_cow(
                state,
                replacement,
                token,
                row.row_identity,
                installing=True,
            ):
                rebound = next(
                    item
                    for item in rows
                    if item.row_identity is row.row_identity
                )
                return Browse1DBorrow(self, token, rebound)

    def _release_borrow(
        self, token: _BorrowToken, row_identity: object,
    ) -> None:
        if type(token) is not _BorrowToken or token.row_identity is not row_identity:
            raise ValueError("Browse 1-D borrow token is invalid")
        while True:
            state = self._snapshot_state()
            self._require_open(state)
            found = next(
                (
                    binding
                    for binding in state.borrows
                    if binding.token is token
                    and binding.row_identity is row_identity
                ),
                None,
            )
            if found is None:
                raise ValueError("Browse 1-D borrow token is invalid")
            replacement = _state_with(
                state,
                borrows=tuple(
                    binding
                    for binding in state.borrows
                    if binding is not found
                ),
            )
            if self._install_open_cow(
                state,
                replacement,
                token,
                row_identity,
                installing=False,
            ):
                return

    def close(self) -> None:
        state = self._snapshot_state()
        if state.phase is Browse1DCachePhase.CLOSED:
            return
        self._require_open(state)
        if state.borrows:
            raise RuntimeError("Browse 1-D cache has outstanding borrows")
        exchange = self._authority.exchange(
            tuple(row.lease for row in state.rows),
        )
        marker = _OperationMarker()
        journal = _CacheJournal(
            marker,
            exchange,
            state.rows,
            (),
            (),
            (),
            True,
        )
        operation = Browse1DCacheOperation(self, marker)
        preparing = _state_with(
            state,
            phase=Browse1DCachePhase.PREPARING,
            rows=(),
            journal=journal,
            replace_journal=True,
            driver=None,
            replace_driver=True,
        )
        try:
            installed = self._transition(state, preparing)
        except BaseException:
            if self._snapshot_state() is not preparing:
                exchange.rollback()
            raise
        if not installed:
            exchange.rollback()
            raise RuntimeError("Browse 1-D close transition did not complete")
        operation.run()


__all__ = [
    "Browse1DBorrow",
    "Browse1DCache",
    "Browse1DLabelStoreCustodyError",
    "Browse1DLabelStoreReceipt",
    "Browse1DCacheOperation",
    "Browse1DCachePhase",
    "Browse1DRowKey",
    "browse_1d_row_name",
    "default_browse_1d_cache_budget",
]
