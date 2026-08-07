# -*- coding: utf-8 -*-
"""Qt- and NumPy-import-free byte owner for canonical light 1-D records."""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping as MappingABC
import threading
import sys
import weakref
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Callable, Hashable, Mapping

from ._closed_values import (
    freeze_identity,
    freeze_metadata_mapping,
    require_exact_string,
)

__all__ = [
    "GUIThreadHydrationRefused",
    "Light1DBorrow",
    "Light1DBufferLayout",
    "Light1DCleanupHooks",
    "Light1DCleanupPending",
    "Light1DCleanupReceipt",
    "Light1DCleanupToken",
    "Light1DHydrationToken",
    "Light1DLayout",
    "Light1DLeaseState",
    "Light1DModeData",
    "Light1DModeLayout",
    "Light1DRecord",
    "Light1DReleaseReceipt",
    "Light1DRetentionLease",
    "Light1DStaleGeneration",
    "Light1DUnavailable",
    "SessionResourceAuthority",
    "SessionResourceAuthoritySnapshot",
    "acquire_light_1d_retention",
]


def _nonnegative_int(name: str, value: int, *, positive: bool = False) -> int:
    if isinstance(value, bool) or int(value) != value:
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < int(positive):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _freeze_light_identity(value, name: str):
    """Own one small immutable identity without accepting hidden payloads."""
    def extension(candidate, recurse):
        stage_module = sys.modules.get("xrd_tools.session.stage_accounting")
        result_mode_type = getattr(stage_module, "ResultMode", None)
        if result_mode_type is not None and type(candidate) is result_mode_type:
            return result_mode_type(
                recurse(candidate.kind), recurse(candidate.key),
            )
        dynamic_module = sys.modules.get("xrd_tools.session.dynamic_accounting")
        frame_identity_type = getattr(
            dynamic_module, "DynamicFrameIdentity", None,
        )
        if frame_identity_type is not None and type(candidate) is frame_identity_type:
            return frame_identity_type(
                recurse(candidate.source_identity),
                recurse(candidate.logical_frame_identity),
            )
        return NotImplemented

    return freeze_identity(
        value,
        f"{name} identity",
        extension=extension,
        description=(
            "exact int/string/bytes/tuple values, ResultMode, or "
            "DynamicFrameIdentity"
        ),
    )


@dataclass(frozen=True, slots=True)
class Light1DBufferLayout:
    length: int
    itemsize: int
    owner_key: str
    dtype: str
    shared: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "length", _nonnegative_int(
            "buffer length", self.length, positive=True,
        ))
        object.__setattr__(self, "itemsize", _nonnegative_int(
            "buffer itemsize", self.itemsize, positive=True,
        ))
        require_exact_string(self.owner_key, "buffer owner_key")
        require_exact_string(self.dtype, "buffer dtype")
        if type(self.shared) is not bool:
            raise TypeError("shared must be an exact bool")

    @property
    def nbytes(self) -> int:
        return self.length * self.itemsize


@dataclass(frozen=True, slots=True)
class Light1DModeLayout:
    mode: Hashable
    coordinate: Light1DBufferLayout
    intensity: Light1DBufferLayout
    uncertainty: Light1DBufferLayout | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "mode", _freeze_light_identity(self.mode, "light-1D mode"),
        )
        if type(self.coordinate) is not Light1DBufferLayout:
            raise TypeError("coordinate must be Light1DBufferLayout")
        if type(self.intensity) is not Light1DBufferLayout:
            raise TypeError("intensity must be Light1DBufferLayout")
        if self.uncertainty is not None and type(
            self.uncertainty,
        ) is not Light1DBufferLayout:
            raise TypeError("uncertainty must be Light1DBufferLayout or None")


@dataclass(frozen=True, slots=True)
class Light1DLayout:
    modes: tuple[Light1DModeLayout, ...]
    active_mode: Hashable
    _groups: Mapping[str, Light1DBufferLayout] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        modes = tuple(self.modes)
        if not modes:
            raise ValueError("light-1D layout requires at least one mode")
        if any(type(mode) is not Light1DModeLayout for mode in modes):
            raise TypeError("modes must contain Light1DModeLayout")
        names = [mode.mode for mode in modes]
        if len(set(names)) != len(names):
            raise ValueError("light-1D layout mode identities must be unique")
        active_mode = _freeze_light_identity(
            self.active_mode, "light-1D active mode",
        )
        if active_mode not in names:
            raise ValueError("active_mode must name one declared 1-D mode")
        groups: dict[str, Light1DBufferLayout] = {}
        for mode in modes:
            if mode.intensity.shared or (
                mode.uncertainty is not None and mode.uncertainty.shared
            ):
                raise ValueError("only coordinate buffers may be declared shared")
            for spec in (mode.coordinate, mode.intensity, mode.uncertainty):
                if spec is None:
                    continue
                prior = groups.get(spec.owner_key)
                if prior is not None and (
                    prior.length != spec.length
                    or prior.itemsize != spec.itemsize
                    or prior.dtype != spec.dtype
                    or prior.shared != spec.shared
                ):
                    raise ValueError(
                        f"owner group {spec.owner_key!r} has contradictory layouts"
                    )
                groups[spec.owner_key] = spec
        object.__setattr__(self, "modes", modes)
        object.__setattr__(self, "active_mode", active_mode)
        object.__setattr__(self, "_groups", MappingProxyType(groups))

    @property
    def shared_bytes(self) -> int:
        return sum(spec.nbytes for spec in self._groups.values() if spec.shared)

    @property
    def per_row_unique_ndarray_bytes(self) -> int:
        return sum(spec.nbytes for spec in self._groups.values() if not spec.shared)

    @property
    def requested_mode_count(self) -> int:
        return len(self.modes)


@dataclass(frozen=True, slots=True)
class Light1DModeData:
    coordinate: object
    intensity: object
    uncertainty: object | None = None


@dataclass(frozen=True, slots=True)
class Light1DRecord:
    row_identity: Hashable
    generation: int
    active_mode: Hashable
    modes: Mapping[Hashable, Light1DModeData]
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        row_identity = _freeze_light_identity(
            self.row_identity, "light record row",
        )
        active_mode = _freeze_light_identity(
            self.active_mode, "light record active mode",
        )
        if isinstance(self.generation, bool) or int(self.generation) != self.generation:
            raise ValueError("light record generation must be an integer")
        frozen_modes = {}
        for mode, value in dict(self.modes).items():
            frozen_mode = _freeze_light_identity(mode, "light record mode")
            if type(value) is not Light1DModeData:
                raise TypeError("light record modes must contain Light1DModeData")
            if frozen_mode in frozen_modes:
                raise ValueError("light record mode identities must be unique")
            frozen_modes[frozen_mode] = value
        frozen_provenance = _freeze_provenance_mapping(self.provenance)
        object.__setattr__(self, "row_identity", row_identity)
        object.__setattr__(self, "generation", int(self.generation))
        object.__setattr__(self, "active_mode", active_mode)
        object.__setattr__(self, "modes", MappingProxyType(frozen_modes))
        object.__setattr__(self, "provenance", frozen_provenance)


def _freeze_provenance_mapping(value) -> Mapping[str, object]:
    return freeze_metadata_mapping(value, "light-1D provenance")


@dataclass(frozen=True, slots=True)
class SessionResourceAuthoritySnapshot:
    capacity_bytes: int
    reserved_bytes: int
    available_bytes: int
    committed_bytes: Mapping[str, int]
    categories: Mapping[str, int]
    reservation_count: int


_ALLOCATION_AUTHORITY_LOCK = threading.Lock()
_ALLOCATION_AUTHORITIES: dict[int, weakref.ReferenceType] = {}
_ALLOCATION_FACTORY_TOKEN = object()


@dataclass(slots=True)
class _ResourceReservation:
    amount: int
    owner: str
    generation: int
    category: str
    claim: object
    prior_generation: int | None
    claimed: bool = False


class SessionResourceAuthority:
    """One explicit byte authority; it never reads physical RAM itself.

    An authority derived from a session allocation preserves that allocation's
    category map as committed parent truth.  A later integration adapter must
    retire any old record/publication 1-D owner before reserving the same array
    buffers here; this authority grants only genuinely uncommitted bytes.
    """

    def __init__(
        self,
        *,
        capacity_bytes: int,
        committed_bytes: Mapping[str, int] | None = None,
        parent_allocation=None,
        _factory_token=None,
    ) -> None:
        if parent_allocation is not None and _factory_token is not _ALLOCATION_FACTORY_TOKEN:
            raise TypeError(
                "parent_allocation authority must be created by from_allocation"
            )
        self._capacity_bytes = _nonnegative_int(
            "capacity_bytes", capacity_bytes,
        )
        committed = {
            str(name): _nonnegative_int(f"committed {name}", value)
            for name, value in dict(committed_bytes or {}).items()
        }
        if sum(committed.values()) > self.capacity_bytes:
            raise ValueError("committed parent bytes exceed authority capacity")
        self._committed = MappingProxyType(committed)
        self._parent_allocation = parent_allocation
        self._lock = threading.Lock()
        self._next_grant = 1
        self._reservations: dict[str, _ResourceReservation] = {}
        self._last_generation: dict[str, int] = {}

    def __copy__(self):
        raise TypeError("SessionResourceAuthority is a unique byte owner")

    def __deepcopy__(self, _memo):
        raise TypeError("SessionResourceAuthority is a unique byte owner")

    @property
    def capacity_bytes(self) -> int:
        return self._capacity_bytes

    @property
    def parent_allocation(self):
        return self._parent_allocation

    @classmethod
    def from_allocation(cls, allocation) -> "SessionResourceAuthority":
        if cls is not SessionResourceAuthority:
            raise TypeError(
                "allocation authority factory requires exact SessionResourceAuthority"
            )
        policy_module = sys.modules.get("xrd_tools.session.policy")
        allocation_type = getattr(
            policy_module, "SessionResourceAllocation", None,
        )
        if allocation_type is None or type(allocation) is not allocation_type:
            raise TypeError(
                "authority requires an exact H10 SessionResourceAllocation"
            )
        resolver = getattr(policy_module, "resolve_session_policy", None)
        if not callable(resolver) or resolver(
            allocation.requirements,
            allocation=allocation,
            env={},
        ).allocation is not allocation:
            raise ValueError("session allocation evidence did not survive validation")
        categories = dict(getattr(allocation, "categories", {}))
        assigned = _nonnegative_int(
            "allocation assigned_bytes", getattr(allocation, "assigned_bytes", -1),
        )
        committed = {
            str(name): _nonnegative_int(f"allocation category {name}", value)
            for name, value in categories.items()
        }
        if sum(committed.values()) != assigned:
            raise ValueError(
                "allocation categories must sum exactly to assigned_bytes"
            )
        envelope = _nonnegative_int(
            "allocation envelope_bytes", getattr(allocation, "envelope_bytes", -1),
        )
        oversize = _nonnegative_int(
            "allocation oversize_excess_bytes",
            getattr(allocation, "oversize_excess_bytes", 0),
        )
        authorized = envelope + oversize
        if assigned > authorized:
            raise ValueError("allocation exceeds its effective authorized capacity")
        allocation_id = id(allocation)
        with _ALLOCATION_AUTHORITY_LOCK:
            observed_ref = _ALLOCATION_AUTHORITIES.get(allocation_id)
            observed = observed_ref() if observed_ref is not None else None
            if observed is not None:
                if observed.parent_allocation is not allocation:
                    raise RuntimeError("allocation authority identity collision")
                return observed
            authority = cls(
                capacity_bytes=authorized,
                committed_bytes=committed,
                parent_allocation=allocation,
                _factory_token=_ALLOCATION_FACTORY_TOKEN,
            )

            def clear(ref, *, key=allocation_id):
                with _ALLOCATION_AUTHORITY_LOCK:
                    if _ALLOCATION_AUTHORITIES.get(key) is ref:
                        _ALLOCATION_AUTHORITIES.pop(key, None)

            _ALLOCATION_AUTHORITIES[allocation_id] = weakref.ref(authority, clear)
            return authority

    def _reserve_light(
        self,
        *,
        owner: str,
        generation: int,
        layout: Light1DLayout,
        requested_rows: int,
        compatibility_byte_ceiling: int,
    ) -> tuple[str, int, int, object]:
        requested_rows = _nonnegative_int("requested_rows", requested_rows)
        ceiling = _nonnegative_int(
            "compatibility_byte_ceiling", compatibility_byte_ceiling,
        )
        owner, generation = str(owner), int(generation)
        with self._lock:
            if any(
                reservation.owner == owner and reservation.category == "light_1d"
                for reservation in self._reservations.values()
            ):
                raise RuntimeError(
                    "owner already has an active light-1D reservation"
                )
            prior_generation = self._last_generation.get(owner)
            if prior_generation is not None and generation <= prior_generation:
                raise ValueError(
                    "replacement light-1D generation must be strictly newer"
                )
            used = sum(self._committed.values()) + sum(
                reservation.amount for reservation in self._reservations.values()
            )
            available = max(0, self.capacity_bytes - used)
            per_row = layout.per_row_unique_ndarray_bytes
            shared = layout.shared_bytes
            requested_bytes = (
                0 if requested_rows == 0
                else shared + requested_rows * per_row
            )
            limit = min(requested_bytes, ceiling, available)
            if requested_rows == 0 or limit < shared + per_row:
                row_cap = reserved = 0
            else:
                row_cap = min(requested_rows, (limit - shared) // per_row)
                reserved = shared + row_cap * per_row
            grant_id = f"light-1d-{self._next_grant}"
            self._next_grant += 1
            claim = object()
            self._reservations[grant_id] = _ResourceReservation(
                reserved, owner, generation, "light_1d", claim, prior_generation,
            )
            self._last_generation[owner] = generation
            return grant_id, row_cap, reserved, claim

    def _claim_light(
        self, grant_id: str, *, owner: str, generation: int, claim: object,
    ) -> None:
        with self._lock:
            reservation = self._reservations.get(grant_id)
            if (
                reservation is None
                or reservation.owner != owner
                or reservation.generation != int(generation)
                or reservation.claim is not claim
            ):
                raise RuntimeError(
                    "light-1D lease must be created through exact acquire authority"
                )
            if reservation.claimed:
                raise RuntimeError("light-1D reservation already has a lease owner")
            reservation.claimed = True

    def _cancel_light_construction(
        self, grant_id: str, *, owner: str, generation: int, claim: object,
    ) -> None:
        with self._lock:
            reservation = self._reservations.get(grant_id)
            if reservation is None:
                return
            if (
                reservation.owner != owner
                or reservation.generation != int(generation)
                or reservation.claim is not claim
            ):
                raise RuntimeError("cannot cancel a foreign light-1D reservation")
            del self._reservations[grant_id]
            if reservation.prior_generation is None:
                self._last_generation.pop(owner, None)
            else:
                self._last_generation[owner] = reservation.prior_generation

    def _release(self, grant_id: str, *, owner: str, generation: int) -> int:
        with self._lock:
            reservation = self._reservations.get(grant_id)
            if reservation is None:
                raise RuntimeError("resource release grant is absent")
            if reservation.owner != owner or reservation.generation != int(generation):
                raise RuntimeError("resource release names a foreign grant owner")
            if not reservation.claimed:
                raise RuntimeError("resource release names an unclaimed reservation")
            del self._reservations[grant_id]
            return reservation.amount

    def snapshot(self) -> SessionResourceAuthoritySnapshot:
        with self._lock:
            categories = dict(self._committed)
            for reservation in self._reservations.values():
                categories[reservation.category] = (
                    categories.get(reservation.category, 0) + reservation.amount
                )
            reserved = sum(categories.values())
            return SessionResourceAuthoritySnapshot(
                capacity_bytes=self.capacity_bytes,
                reserved_bytes=reserved,
                available_bytes=self.capacity_bytes - reserved,
                committed_bytes=self._committed,
                categories=MappingProxyType(categories),
                reservation_count=len(self._reservations),
            )


class Light1DLeaseState(str, Enum):
    ACTIVE = "active"
    FENCED = "fenced"
    CLEANUP_PENDING = "cleanup-pending"
    RELEASED = "released"


class Light1DStaleGeneration(RuntimeError):
    pass


class GUIThreadHydrationRefused(RuntimeError):
    pass


class Light1DUnavailable(RuntimeError):
    pass


class _WeakRootMapping(MappingABC):
    """Immutable keys that let a worker borrow, but never retain, roots."""

    __slots__ = ("_refs",)

    def __init__(self, roots: Mapping[str, object]) -> None:
        self._refs = MappingProxyType({
            key: weakref.ref(value) for key, value in roots.items()
        })

    def __getitem__(self, key: str) -> object:
        value = self._refs[key]()
        if value is None:
            raise Light1DStaleGeneration(
                "hydration shared-root borrow is no longer live"
            )
        return value

    def __iter__(self):
        return iter(self._refs)

    def __len__(self) -> int:
        return len(self._refs)


@dataclass(frozen=True, eq=False, slots=True)
class Light1DHydrationToken:
    grant_id: str
    generation: int
    row_identity: Hashable
    ordinal: int
    shared_roots: Mapping[str, object] = field(default_factory=dict)
    may_create_shared: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "row_identity",
            _freeze_light_identity(self.row_identity, "hydration row"),
        )
        roots = dict(self.shared_roots)
        if any(not isinstance(key, str) or not key for key in roots):
            raise ValueError("hydration shared-root keys must be non-empty strings")
        if roots and self.may_create_shared:
            raise ValueError(
                "hydration token cannot borrow and create shared roots"
            )
        object.__setattr__(self, "shared_roots", _WeakRootMapping(roots))
        object.__setattr__(self, "may_create_shared", bool(self.may_create_shared))


@dataclass(frozen=True, slots=True)
class Light1DCleanupToken:
    grant_id: str
    generation: int
    ordinal: int


@dataclass(frozen=True, slots=True)
class Light1DCleanupReceipt:
    """Immutable evidence that cleanup is incomplete, never a release."""

    grant_id: str
    owner: str
    generation: int
    reserved_bytes: int
    reason: str
    retry_token: Light1DCleanupToken
    state: str
    completed_steps: tuple[str, ...]
    failed_step: str
    error: str


@dataclass(frozen=True, slots=True)
class Light1DReleaseReceipt:
    grant_id: str
    owner: str
    generation: int
    released_bytes: int
    reason: str
    retry_token: Light1DCleanupToken


@dataclass(frozen=True, slots=True)
class Light1DCleanupHooks:
    cancel: Callable[[], None] | None = None
    drain: Callable[[], None] | None = None
    clear: Callable[[], None] | None = None
    detach: Callable[[], None] | None = None
    verify: Callable[[], None] | None = None
    release: Callable[[], None] | None = None


class Light1DCleanupPending(RuntimeError):
    def __init__(
        self, receipt: Light1DCleanupReceipt, cause: BaseException,
    ) -> None:
        super().__init__(f"light-1D cleanup remains pending: {cause}")
        self.receipt = receipt
        self.token = receipt.retry_token


_BORROW_FACTORY_TOKEN = object()


class Light1DBorrow:
    """Tracked shell/display borrow that must close before byte regrant."""

    __slots__ = (
        "_lease_ref", "_ordinal", "_record", "_closed", "__weakref__",
    )

    def __init__(self, lease, ordinal: int, record: Light1DRecord, *, _claim=None):
        if _claim is not _BORROW_FACTORY_TOKEN:
            raise TypeError("light-1D borrows are minted only by their lease")
        self._lease_ref = weakref.ref(lease)
        self._ordinal = int(ordinal)
        self._record = record
        self._closed = False

    def __copy__(self):
        raise TypeError("Light1DBorrow is an exact tracked owner")

    def __deepcopy__(self, _memo):
        raise TypeError("Light1DBorrow is an exact tracked owner")

    @property
    def closed(self) -> bool:
        return self._closed

    def __getattr__(self, name: str):
        record = self._record
        if record is None or self._closed:
            raise Light1DStaleGeneration("light-1D shell borrow is closed")
        return getattr(record, name)

    def close(self) -> None:
        if self._closed:
            return
        lease = self._lease_ref()
        if lease is not None:
            lease._close_borrow(self._ordinal, self)
            return
        self._record = None
        self._closed = True

    def __enter__(self) -> "Light1DBorrow":
        if self._closed:
            raise Light1DStaleGeneration("light-1D shell borrow is closed")
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()


def _array_chain(array) -> tuple[object, ...]:
    chain = []
    value = array
    seen = set()
    while id(value) not in seen:
        seen.add(id(value))
        chain.append(value)
        base = getattr(value, "base", None)
        if base is None:
            break
        value = base
    return tuple(chain)


def _array_root(array):
    return _array_chain(array)[-1]


def _array_nbytes(array) -> int:
    value = getattr(array, "nbytes", None)
    if isinstance(value, bool) or value is None or int(value) != value:
        raise ValueError("light-1D layout requires ndarray-like nbytes")
    return int(value)


def _canonical_payload(root, expected_nbytes: int) -> bytes:
    """Return the exact immutable payload behind one canonical root."""
    if type(root) is not memoryview or not root.readonly:
        raise RuntimeError("light-1D canonical root is not an immutable memoryview")
    payload = root.obj
    if type(payload) is not bytes or len(payload) != int(expected_nbytes):
        raise RuntimeError("light-1D canonical root has no exact bytes payload")
    return payload


def _payload_group_detached(payloads: tuple[bytes, ...]) -> bool:
    """Positively prove that only this receipt owns each immutable payload.

    NumPy and memoryview aliases can outlive their tracked shell and even the
    exact memoryview root.  Exact CPython reference counts are the remaining
    liveness receipt for the non-weak-referenceable ``bytes`` owner.  Runtimes
    without that receipt fail closed instead of regranting uncertain bytes.
    """
    getrefcount = getattr(sys, "getrefcount", None)
    if not callable(getrefcount):
        return False
    for payload in payloads:
        # One reference from ``payloads``, one loop-local reference, and the
        # temporary reference taken by getrefcount itself.
        if getrefcount(payload) != 3:
            return False
    return True


def _exact_numpy_ndarray_type():
    """Resolve NumPy only if the consumer already loaded it; never import it."""
    module = sys.modules.get("numpy")
    ndarray_type = getattr(module, "ndarray", None)
    if not isinstance(ndarray_type, type):
        raise ValueError(
            "light-1D records require NumPy to provide the exact ndarray type"
        )
    return ndarray_type


def _freeze_array(array) -> None:
    if type(array) is memoryview:
        if not array.readonly:
            raise ValueError("light-1D buffer owner is not immutable")
        return
    setter = getattr(array, "setflags", None)
    if not callable(setter):
        raise ValueError("light-1D array has no enforceable write fence")
    try:
        setter(write=False)
    except BaseException as exc:
        raise ValueError("light-1D array write fence failed") from exc
    try:
        flags = getattr(array, "flags", None)
        writeable = getattr(flags, "writeable", None)
    except BaseException as exc:
        raise ValueError("light-1D array write fence cannot be observed") from exc
    if writeable is None or bool(writeable):
        raise ValueError("light-1D array write fence was not established")


@dataclass(frozen=True, slots=True)
class _Light1DGrant:
    """Immutable byte grant; the lease owns only mutable residency state."""

    authority: SessionResourceAuthority
    grant_id: str
    owner: str
    generation: int
    layout: Light1DLayout
    requested_rows: int
    row_cap: int
    reserved_ndarray_bytes: int
    shared_bytes: int
    per_row_unique_ndarray_bytes: int
    gui_thread_id: int


class Light1DRetentionLease:
    """One reservation and one canonical record index for one generation.

    The byte grant is exact for declared unique ndarray roots. Python object
    overhead is outside that byte envelope, so the same granted row cap also
    bounds resident rows, hydration owners, and eviction diagnostics.
    """

    def __init__(
        self,
        authority: SessionResourceAuthority,
        *,
        grant_id: str,
        owner: str,
        generation: int,
        layout: Light1DLayout,
        requested_rows: int,
        row_cap: int,
        reserved_ndarray_bytes: int,
        gui_thread_id: int,
        _reservation_claim=None,
    ) -> None:
        if type(authority) is not SessionResourceAuthority:
            raise TypeError("lease requires exact SessionResourceAuthority")
        if type(layout) is not Light1DLayout:
            raise TypeError("lease requires exact Light1DLayout")
        if isinstance(gui_thread_id, bool) or int(gui_thread_id) != gui_thread_id:
            raise ValueError("gui_thread_id must be an integer")
        authority._claim_light(
            grant_id,
            owner=str(owner),
            generation=int(generation),
            claim=_reservation_claim,
        )
        self._grant = _Light1DGrant(
            authority=authority,
            grant_id=str(grant_id),
            owner=str(owner),
            generation=int(generation),
            layout=layout,
            requested_rows=int(requested_rows),
            row_cap=int(row_cap),
            reserved_ndarray_bytes=int(reserved_ndarray_bytes),
            shared_bytes=layout.shared_bytes,
            per_row_unique_ndarray_bytes=layout.per_row_unique_ndarray_bytes,
            gui_thread_id=int(gui_thread_id),
        )
        self._lock = threading.RLock()
        self._cleanup_lock = threading.Lock()
        self._state = Light1DLeaseState.ACTIVE
        self._records: dict[Hashable, Light1DRecord] = {}
        self._roots: dict[Hashable, dict[str, object]] = {}
        self._shared_roots: dict[str, object] = {}
        self._shared_views: dict[str, object] = {}
        self._borrow_ordinal = 0
        self._borrows: dict[
            int, tuple[Hashable, Light1DBorrow]
        ] = {}
        self._evicted: deque[Hashable] = deque(maxlen=self.row_cap)
        self._hydration_ordinal = 0
        self._hydration_tokens: dict[int, Light1DHydrationToken] = {}
        self._hydration_rows: dict[Hashable, Light1DHydrationToken] = {}
        self._hydration_authorizations = 0
        self._cleanup_token: Light1DCleanupToken | None = None
        self._cleanup_step = 0
        self._cleanup_reason: str | None = None
        self._cleanup_hooks: Light1DCleanupHooks | None = None
        self._cleanup_callback_call = None
        self._retired_payload_groups: deque[tuple[bytes, ...]] = deque()
        self._cleanup_receipt: Light1DCleanupReceipt | None = None
        self._release_receipt: Light1DReleaseReceipt | None = None

    def __copy__(self):
        raise TypeError("Light1DRetentionLease is a unique generation owner")

    def __deepcopy__(self, _memo):
        raise TypeError("Light1DRetentionLease is a unique generation owner")

    @property
    def authority(self) -> SessionResourceAuthority:
        return self._grant.authority

    @property
    def grant_id(self) -> str:
        return self._grant.grant_id

    @property
    def owner(self) -> str:
        return self._grant.owner

    @property
    def generation(self) -> int:
        return self._grant.generation

    @property
    def layout(self) -> Light1DLayout:
        return self._grant.layout

    @property
    def requested_rows(self) -> int:
        return self._grant.requested_rows

    @property
    def row_cap(self) -> int:
        return self._grant.row_cap

    @property
    def reserved_ndarray_bytes(self) -> int:
        return self._grant.reserved_ndarray_bytes

    @property
    def shared_bytes(self) -> int:
        return self._grant.shared_bytes

    @property
    def per_row_unique_ndarray_bytes(self) -> int:
        return self._grant.per_row_unique_ndarray_bytes

    @property
    def state(self) -> Light1DLeaseState:
        return self._state

    @property
    def evicted(self) -> tuple[Hashable, ...]:
        with self._lock:
            return tuple(self._evicted)

    @property
    def hydration_authorizations(self) -> int:
        with self._lock:
            return self._hydration_authorizations

    @property
    def pending_hydration_count(self) -> int:
        with self._lock:
            return len(self._hydration_tokens)

    @property
    def active_borrow_count(self) -> int:
        with self._lock:
            self._prune_borrows()
            return len(self._borrows)

    @property
    def cleanup_receipt(self) -> Light1DCleanupReceipt | None:
        with self._lock:
            return self._cleanup_receipt

    @property
    def owned_buffer_ids(self) -> frozenset[int]:
        with self._lock:
            self._prune_retired_payloads()
            return frozenset(
                id(root)
                for root in (
                    *self._shared_roots.values(),
                    *(root for roots in self._roots.values() for root in roots.values()),
                )
            ) | frozenset(
                id(payload)
                for group in self._retired_payload_groups
                for payload in group
            )

    @property
    def unique_owned_ndarray_bytes(self) -> int:
        with self._lock:
            self._prune_retired_payloads()
            roots = {
                id(root): root
                for root in (
                    *self._shared_roots.values(),
                    *(root for row_roots in self._roots.values()
                      for root in row_roots.values()),
                )
            }
            return (
                sum(_array_nbytes(root) for root in roots.values())
                + sum(
                    len(payload)
                    for group in self._retired_payload_groups
                    for payload in group
                )
            )

    def _check_generation(self, grant_id: str, generation: int) -> None:
        if grant_id != self.grant_id or int(generation) != self.generation:
            raise Light1DStaleGeneration("light-1D grant/generation is stale")
        if self._state is not Light1DLeaseState.ACTIVE:
            raise Light1DStaleGeneration("light-1D generation is fenced")

    def _specs(self):
        for mode in self.layout.modes:
            yield mode.mode, "coordinate", mode.coordinate
            yield mode.mode, "intensity", mode.intensity
            if mode.uncertainty is not None:
                yield mode.mode, "uncertainty", mode.uncertainty

    @staticmethod
    def _validate_dtype_graph(dtype, role: str) -> None:
        if getattr(dtype, "metadata", None) is not None:
            raise ValueError(
                f"light-1D layout for {role} cannot account dtype metadata"
            )
        if (
            getattr(dtype, "fields", None) is not None
            or getattr(dtype, "subdtype", None) is not None
            or getattr(dtype, "kind", None) not in frozenset("biufc")
        ):
            raise ValueError(
                f"light-1D layout for {role} requires a plain numeric/bool dtype"
            )

    @staticmethod
    def _validate_array(array, spec: Light1DBufferLayout, role: str) -> object:
        if array is None:
            raise ValueError(f"light-1D layout requires {role}")
        ndarray_type = _exact_numpy_ndarray_type()
        if type(array) is not ndarray_type:
            raise ValueError(
                f"light-1D layout for {role} requires an exact NumPy ndarray"
            )
        ndim = getattr(array, "ndim", None)
        shape = getattr(array, "shape", None)
        dtype = getattr(array, "dtype", None)
        if bool(getattr(dtype, "hasobject", False)):
            raise ValueError(
                f"light-1D layout for {role} cannot account object-containing dtype"
            )
        Light1DRetentionLease._validate_dtype_graph(dtype, role)
        itemsize = getattr(dtype, "itemsize", None)
        dtype_descriptor = getattr(dtype, "str", None)
        if (
            ndim != 1
            or not shape
            or int(shape[0]) != spec.length
            or itemsize is None
            or int(itemsize) != spec.itemsize
            or dtype_descriptor != spec.dtype
            or _array_nbytes(array) != spec.nbytes
        ):
            raise ValueError(
                f"light-1D record does not match layout dtype/shape for {role}"
            )
        root = _array_root(array)
        if type(root) is ndarray_type:
            root_dtype = getattr(root, "dtype", None)
            if bool(getattr(root_dtype, "hasobject", False)):
                raise ValueError(
                    f"light-1D layout for {role} hides an object-containing root"
                )
            Light1DRetentionLease._validate_dtype_graph(
                root_dtype, f"{role} root",
            )
        elif type(root) is memoryview:
            owner = getattr(root, "obj", None)
            if (
                not root.readonly
                or type(owner) is not bytes
                or len(owner) != spec.nbytes
            ):
                raise ValueError(
                    f"light-1D layout for {role} requires an exact immutable "
                    "buffer root"
                )
        else:
            raise ValueError(
                f"light-1D layout for {role} requires an exact NumPy ndarray root "
                "or immutable memoryview root"
            )
        if _array_nbytes(root) != spec.nbytes:
            raise ValueError(
                f"light-1D layout for {role} hides a larger owned base buffer"
            )
        return root

    def _validate_record(
        self, record: Light1DRecord,
    ) -> tuple[dict[str, object], tuple[object, ...]]:
        if type(record) is not Light1DRecord:
            raise TypeError("retain requires exact Light1DRecord")
        if record.generation != self.generation:
            raise Light1DStaleGeneration("light record belongs to another generation")
        expected_modes = {mode.mode for mode in self.layout.modes}
        if set(record.modes) != expected_modes or record.active_mode != self.layout.active_mode:
            raise ValueError("light-1D record mode layout does not match grant layout")
        roots: dict[str, object] = {}
        supplied: list[object] = []
        for mode_layout in self.layout.modes:
            values = record.modes[mode_layout.mode]
            triples = (
                ("coordinate", values.coordinate, mode_layout.coordinate),
                ("intensity", values.intensity, mode_layout.intensity),
                ("uncertainty", values.uncertainty, mode_layout.uncertainty),
            )
            for role, array, spec in triples:
                if spec is None:
                    if array is not None:
                        raise ValueError(
                            f"light-1D record does not match layout: unexpected {role}"
                        )
                    continue
                root = self._validate_array(
                    array, spec, f"{mode_layout.mode!r}.{role}",
                )
                supplied.extend(_array_chain(array))
                prior = roots.get(spec.owner_key)
                if prior is not None and prior is not root:
                    raise ValueError(
                        f"light-1D layout owner {spec.owner_key!r} is not aliased"
                    )
                roots[spec.owner_key] = root
        reverse: dict[int, str] = {}
        for owner_key, root in roots.items():
            prior = reverse.get(id(root))
            if prior is not None and prior != owner_key:
                raise ValueError(
                    "light-1D record aliases distinct declared owner groups"
                )
            reverse[id(root)] = owner_key
        return roots, tuple(supplied)

    @staticmethod
    def _same_array_values(left, right) -> bool:
        try:
            return left.tobytes(order="C") == right.tobytes(order="C")
        except BaseException as exc:
            raise ValueError("light-1D canonical buffer comparison failed") from exc

    def _private_array_copy(
        self, array, spec: Light1DBufferLayout, role: str,
    ) -> tuple[object, object]:
        try:
            source_payload = array.tobytes(order="C")
            numpy = sys.modules.get("numpy")
            frombuffer = getattr(numpy, "frombuffer", None)
            if type(source_payload) is not bytes or not callable(frombuffer):
                raise TypeError("exact immutable NumPy buffer factory unavailable")
            # Force a fresh, non-interned identity even for one-byte values so
            # its exact CPython reference count remains a usable receipt.
            payload = bytes(memoryview(source_payload))
            view = frombuffer(
                memoryview(payload), dtype=spec.dtype, count=spec.length,
            )
        except BaseException as exc:
            raise ValueError(
                f"light-1D canonical copy failed for {role}"
            ) from exc
        root = self._validate_array(view, spec, role)
        if type(root) is not memoryview:
            raise ValueError("light-1D canonical copy has no immutable buffer owner")
        _freeze_array(root)
        _freeze_array(view)
        return view, root

    def _canonicalize_record(
        self, record: Light1DRecord,
    ) -> tuple[
        Light1DRecord,
        dict[str, object],
        tuple[object, ...],
        dict[str, object],
    ]:
        self._validate_record(record)
        canonical_views: dict[str, object] = {}
        canonical_roots: dict[str, object] = {}
        source_values: dict[str, object] = {}
        modes = {}
        for mode_layout in self.layout.modes:
            values = record.modes[mode_layout.mode]
            fields = {}
            for role, array, spec in (
                ("coordinate", values.coordinate, mode_layout.coordinate),
                ("intensity", values.intensity, mode_layout.intensity),
                ("uncertainty", values.uncertainty, mode_layout.uncertainty),
            ):
                if spec is None:
                    fields[role] = None
                    continue
                prior_source = source_values.get(spec.owner_key)
                if prior_source is not None and not self._same_array_values(
                    array, prior_source,
                ):
                    raise ValueError(
                        f"light-1D owner {spec.owner_key!r} has contradictory views"
                    )
                source_values.setdefault(spec.owner_key, array)
                view = canonical_views.get(spec.owner_key)
                root = canonical_roots.get(spec.owner_key)
                if view is None and spec.shared:
                    view = self._shared_views.get(spec.owner_key)
                    root = self._shared_roots.get(spec.owner_key)
                    if view is not None and not self._same_array_values(array, view):
                        raise ValueError(
                            f"light-1D shared owner {spec.owner_key!r} changed values"
                        )
                    if (view is None) != (root is None):
                        raise RuntimeError(
                            "light-1D shared view/root authority is incomplete"
                        )
                if view is None:
                    view, root = self._private_array_copy(
                        array, spec, f"{mode_layout.mode!r}.{role}",
                    )
                canonical_views[spec.owner_key] = view
                canonical_roots[spec.owner_key] = root
                fields[role] = view
            modes[mode_layout.mode] = Light1DModeData(**fields)
        canonical_record = Light1DRecord(
            row_identity=record.row_identity,
            generation=record.generation,
            active_mode=record.active_mode,
            modes=modes,
            provenance=record.provenance,
        )
        supplied = tuple((
            *canonical_views.values(),
            *canonical_roots.values(),
        ))
        return canonical_record, canonical_roots, supplied, canonical_views

    def _remove(self, key: Hashable, *, evicted: bool = False) -> None:
        if key not in self._records:
            return
        if self._row_has_borrow(key):
            raise Light1DUnavailable(
                "tracked light-1D shell borrow must close before eviction"
            )
        self._retire_row(key)
        if evicted:
            self._evicted.append(key)

    def _retire_row(self, key: Hashable) -> None:
        """Move one resident row into its still-charged alias receipt."""
        roots = self._roots.pop(key)
        self._records.pop(key)
        payload_by_identity: dict[int, bytes] = {}
        for owner_key, root in roots.items():
            spec = self.layout._groups[owner_key]
            if spec.shared:
                continue
            payload = _canonical_payload(root, spec.nbytes)
            payload_by_identity.setdefault(id(payload), payload)
        payloads = tuple(payload_by_identity.values())
        if not payloads:
            raise RuntimeError("light-1D row has no charged unique payload")
        self._retired_payload_groups.append(payloads)

    def _prune_retired_payloads(self) -> None:
        if not self._retired_payload_groups:
            return
        retained: deque[tuple[bytes, ...]] = deque()
        while self._retired_payload_groups:
            payloads = self._retired_payload_groups.popleft()
            if not _payload_group_detached(payloads):
                retained.append(payloads)
        self._retired_payload_groups = retained

    def _occupied_row_slots(self, pending_count: int | None = None) -> int:
        self._prune_retired_payloads()
        if pending_count is None:
            pending_count = len(self._hydration_tokens)
        return len(self._records) + int(pending_count) + len(
            self._retired_payload_groups
        )

    def _raise_slot_unavailable(self, message: str) -> None:
        if self._retired_payload_groups:
            raise Light1DUnavailable(
                f"{message}; an externally retained immutable alias still "
                "owns its granted row slot"
            )
        raise Light1DUnavailable(message)

    def _prune_borrows(self) -> None:
        # Deliberately keep a strong owner until explicit close.  Otherwise a
        # temporary expression such as ``lease.borrow(row).modes[...]`` could
        # drop its handle while an extracted ndarray remained live, allowing
        # cleanup to regrant bytes without positive borrower detachment.
        return None

    def _row_has_borrow(self, row_identity: Hashable) -> bool:
        self._prune_borrows()
        return any(
            row == row_identity for row, _handle in self._borrows.values()
        )

    def _close_borrow(self, ordinal: int, handle: Light1DBorrow) -> None:
        with self._lock:
            if handle._closed:
                return
            observed = self._borrows.get(int(ordinal))
            if observed is None or observed[1] is not handle:
                raise Light1DStaleGeneration(
                    "light-1D shell borrow is stale or foreign"
                )
            handle._record = None
            handle._closed = True
            self._borrows.pop(int(ordinal), None)

    def _validate_canonical_root_ownership(
        self,
        row_identity: Hashable,
        roots: Mapping[str, object],
    ) -> None:
        """Validate aliases in a short-lived frame before any row retires."""
        other_roots = tuple(
            (owner_key, root)
            for key, row_roots in self._roots.items()
            if key != row_identity
            for owner_key, root in row_roots.items()
        )
        for owner_key, root in roots.items():
            spec = self.layout._groups[owner_key]
            prior_shared = self._shared_roots.get(owner_key)
            if spec.shared and prior_shared is not None and prior_shared is not root:
                raise ValueError(
                    f"light-1D shared owner {owner_key!r} changed identity"
                )
            for prior_owner, prior_root in other_roots:
                if prior_root is root and (
                    not spec.shared or prior_owner != owner_key
                ):
                    raise ValueError(
                        "light-1D row reuses a buffer charged to another owner"
                    )

    def _retain_validated(
        self,
        record: Light1DRecord,
        roots: dict[str, object],
        supplied: tuple[object, ...],
        views: dict[str, object],
        *,
        hydration_token: Light1DHydrationToken | None = None,
    ) -> None:
        pending = self._hydration_rows.get(record.row_identity)
        if pending is not None and pending is not hydration_token:
            raise Light1DUnavailable(
                "row has a pending hydration owner; complete its exact token"
            )
        initializers = tuple(
            token for token in self._hydration_tokens.values()
            if token.may_create_shared
        )
        if (
            self.shared_bytes
            and not self._shared_roots
            and not self._shared_views
            and initializers
            and hydration_token not in initializers
        ):
            raise Light1DUnavailable(
                "shared light-1D roots have an exact hydration initializer"
            )
        self._validate_canonical_root_ownership(record.row_identity, roots)
        # Establish immutability before changing the resident owner map.
        # A failed/no-op fence therefore cannot evict the exact prior row.
        frozen = set()
        for array in (*supplied, *roots.values()):
            if id(array) not in frozen:
                _freeze_array(array)
                frozen.add(id(array))
        # Reserve capacity before insertion. Hydration tokens already occupy a
        # row slot; borrowed resident rows cannot be evicted out from under a
        # shell and therefore keep that slot charged until their handle closes.
        replacing = record.row_identity in self._records
        if replacing:
            self._remove(record.row_identity)
        pending_count = len(self._hydration_tokens) - int(
            hydration_token is not None
            and self._hydration_tokens.get(hydration_token.ordinal)
            is hydration_token
        )
        while self._occupied_row_slots(pending_count) >= self.row_cap:
            if not self._records:
                self._raise_slot_unavailable(
                    "light-1D byte grant has no unborrowed retained slot"
                )
            oldest = next(iter(self._records))
            self._remove(oldest, evicted=True)
        self._records[record.row_identity] = record
        self._roots[record.row_identity] = roots
        for owner_key, root in roots.items():
            if self.layout._groups[owner_key].shared:
                self._shared_roots[owner_key] = root
                self._shared_views[owner_key] = views[owner_key]

    def retain(
        self,
        record: Light1DRecord,
        *,
        grant_id: str,
        generation: int,
    ) -> None:
        with self._lock:
            self._check_generation(grant_id, generation)
            if self.row_cap <= 0:
                raise Light1DUnavailable("light-1D byte grant holds zero rows")
            canonical, roots, supplied, views = self._canonicalize_record(record)
            self._retain_validated(canonical, roots, supplied, views)

    def keys(self) -> tuple[Hashable, ...]:
        with self._lock:
            return tuple(self._records)

    def retire(
        self, row_identity: Hashable, *, grant_id: str, generation: int,
    ) -> bool:
        """Retire one active row without classifying it as capacity eviction."""
        row_identity = _freeze_light_identity(row_identity, "retire row")
        with self._lock:
            self._check_generation(grant_id, generation)
            if row_identity not in self._records:
                return False
            self._remove(row_identity)
            return True

    def get(self, row_identity: Hashable) -> Light1DBorrow | None:
        return self.borrow(row_identity)

    def borrow(self, row_identity: Hashable) -> Light1DBorrow | None:
        row_identity = _freeze_light_identity(row_identity, "borrow row")
        with self._lock:
            self._check_generation(self.grant_id, self.generation)
            record = self._records.get(row_identity)
            if record is None:
                return None
            self._borrow_ordinal += 1
            ordinal = self._borrow_ordinal
            handle = Light1DBorrow(
                self, ordinal, record, _claim=_BORROW_FACTORY_TOKEN,
            )
            self._borrows[ordinal] = (row_identity, handle)
            return handle

    def fence(self) -> None:
        with self._lock:
            if self._state is Light1DLeaseState.ACTIVE:
                self._state = Light1DLeaseState.FENCED

    def issue_hydration_token(self, row_identity: Hashable) -> Light1DHydrationToken:
        row_identity = _freeze_light_identity(row_identity, "hydration row")
        with self._lock:
            self._check_generation(self.grant_id, self.generation)
            if row_identity in self._records:
                raise Light1DUnavailable(
                    "row is already resident and needs no hydration owner"
                )
            if threading.get_ident() == self._grant.gui_thread_id:
                raise GUIThreadHydrationRefused(
                    "disk hydration is prohibited on the GUI thread"
                )
            if row_identity in self._hydration_rows:
                raise Light1DUnavailable(
                    "row already owns a pending hydration token"
                )
            shared_keys = frozenset(
                owner_key for owner_key, spec in self.layout._groups.items()
                if spec.shared
            )
            if self._shared_roots or self._shared_views:
                if (
                    set(self._shared_roots) != set(shared_keys)
                    or set(self._shared_views) != set(shared_keys)
                ):
                    raise RuntimeError(
                        "light-1D shared-view/root authority is incomplete"
                    )
                shared_roots = dict(self._shared_views)
                may_create_shared = False
            else:
                if any(
                    token.may_create_shared
                    for token in self._hydration_tokens.values()
                ):
                    raise Light1DUnavailable(
                        "shared light-1D root initialization is already pending"
                    )
                shared_roots = {}
                may_create_shared = bool(shared_keys)
            # A token reserves one of the granted retained-row slots before a
            # worker allocates the replacement arrays.  Otherwise a full cache
            # plus ``row_cap`` concurrent hydrations could transiently own
            # twice the byte grant.  Pending tokens are never evicted; resident
            # rows leave in the same deterministic oldest-insertion order used
            # by retain().
            while self._occupied_row_slots() >= self.row_cap:
                if not self._records:
                    self._raise_slot_unavailable(
                        "pending hydration token cap is exhausted"
                    )
                oldest = next(iter(self._records))
                self._remove(oldest, evicted=True)
            self._hydration_ordinal += 1
            token = Light1DHydrationToken(
                self.grant_id, self.generation, row_identity,
                self._hydration_ordinal,
                shared_roots,
                may_create_shared,
            )
            self._hydration_tokens[token.ordinal] = token
            self._hydration_rows[row_identity] = token
            self._hydration_authorizations += 1
            return token

    def complete_hydration(
        self, token: Light1DHydrationToken, record: Light1DRecord,
    ) -> None:
        with self._lock:
            if type(token) is not Light1DHydrationToken:
                raise TypeError("hydration completion requires Light1DHydrationToken")
            observed = self._hydration_tokens.get(token.ordinal)
            if observed is not token:
                raise Light1DStaleGeneration("hydration token is stale or foreign")
            if (
                token.grant_id != self.grant_id
                or token.generation != self.generation
            ):
                raise Light1DStaleGeneration(
                    "hydration token belongs to another grant/generation"
                )
            if self._state is not Light1DLeaseState.ACTIVE:
                self._hydration_tokens.pop(token.ordinal)
                self._hydration_rows.pop(token.row_identity, None)
                raise Light1DStaleGeneration(
                    "hydration completed after its generation was fenced"
                )
            self._check_generation(token.grant_id, token.generation)
            if record.row_identity != token.row_identity:
                raise ValueError("hydration completion remapped its row identity")
            canonical, roots, supplied, views = self._canonicalize_record(record)
            for owner_key, shared_view in token.shared_roots.items():
                if views.get(owner_key) is not shared_view:
                    raise ValueError(
                        "hydrated record did not borrow its token's shared root"
                    )
            self._retain_validated(
                canonical, roots, supplied, views, hydration_token=token,
            )
            self._hydration_tokens.pop(token.ordinal)
            self._hydration_rows.pop(token.row_identity)

    def abandon_hydration(self, token: Light1DHydrationToken) -> None:
        """Release one exact active worker authorization after failed I/O."""
        with self._lock:
            if type(token) is not Light1DHydrationToken:
                raise TypeError("hydration abandonment requires Light1DHydrationToken")
            observed = self._hydration_tokens.get(token.ordinal)
            if observed is not token:
                raise Light1DStaleGeneration("hydration token is stale or foreign")
            if (
                token.grant_id != self.grant_id
                or token.generation != self.generation
            ):
                raise Light1DStaleGeneration(
                    "hydration token belongs to another grant/generation"
                )
            self._hydration_tokens.pop(token.ordinal)
            self._hydration_rows.pop(token.row_identity, None)

    @staticmethod
    def _call(hook: Callable[[], None] | None) -> None:
        if hook is not None:
            hook()

    def _assert_cleanup_callback(
        self, hooks: Light1DCleanupHooks, step_name: str,
        callback: Callable[[], None], expected_step: int,
    ) -> None:
        with self._lock:
            if (
                self._cleanup_hooks is not hooks
                or getattr(hooks, step_name) is not callback
                or self._cleanup_token is None
                or self._cleanup_step != expected_step
                or self._state not in {
                    Light1DLeaseState.FENCED,
                    Light1DLeaseState.CLEANUP_PENDING,
                }
                or self._cleanup_callback_call != (
                    threading.get_ident(), step_name, callback,
                )
            ):
                raise RuntimeError("light-1D cleanup callback lacks exact authority")

    def _clear_owned_buffers(self) -> None:
        """Retire every canonical payload without laundering live aliases."""
        for key in tuple(self._records):
            self._retire_row(key)
        shared_payloads: dict[int, bytes] = {}
        for owner_key, root in self._shared_roots.items():
            spec = self.layout._groups[owner_key]
            payload = _canonical_payload(root, spec.nbytes)
            shared_payloads.setdefault(id(payload), payload)
        if shared_payloads:
            self._retired_payload_groups.append(
                tuple(shared_payloads.values())
            )
        self._shared_roots.clear()
        self._shared_views.clear()
        self._evicted.clear()

    def _owned_buffers_detached(self) -> bool:
        self._prune_retired_payloads()
        return not self._retired_payload_groups

    def _run_cleanup(self) -> Light1DReleaseReceipt:
        with self._lock:
            hooks = self._cleanup_hooks
        if hooks is None:
            raise RuntimeError("light-1D cleanup hooks were not frozen")
        names = ("cancel", "drain", "clear", "detach", "verify", "release")
        hook_by_name = {
            "cancel": hooks.cancel,
            "drain": hooks.drain,
            "clear": hooks.clear,
            "detach": hooks.detach,
            "verify": hooks.verify,
            "release": hooks.release,
        }
        try:
            while True:
                with self._lock:
                    step = self._cleanup_step
                if step >= len(names):
                    break
                name = names[step]
                # Hooks may join workers that need the state lock to observe
                # the generation fence, so no external call runs under it.
                callback = hook_by_name[name]
                if callback is not None:
                    armed = (threading.get_ident(), name, callback)
                    with self._lock:
                        self._cleanup_callback_call = armed
                    try:
                        callback()
                    finally:
                        with self._lock:
                            if self._cleanup_callback_call == armed:
                                self._cleanup_callback_call = None
                with self._lock:
                    if name == "clear":
                        if self._hydration_tokens or self._hydration_rows:
                            raise RuntimeError(
                                "light-1D hydration authorizations remain active"
                            )
                        self._clear_owned_buffers()
                    elif name == "verify":
                        self._prune_borrows()
                        if (
                            self._records or self._roots or self._shared_roots
                            or self._shared_views or self._hydration_tokens
                            or self._hydration_rows or self._borrows
                            or self._evicted or not self._owned_buffers_detached()
                        ):
                            raise RuntimeError(
                                "light-1D buffers remain owned or externally borrowed"
                            )
                    self._cleanup_step += 1
            released = self.authority._release(
                self.grant_id, owner=self.owner, generation=self.generation,
            )
        except BaseException as exc:
            with self._lock:
                self._state = Light1DLeaseState.CLEANUP_PENDING
                receipt = Light1DCleanupReceipt(
                    self.grant_id,
                    self.owner,
                    self.generation,
                    self.reserved_ndarray_bytes,
                    self._cleanup_reason,
                    self._cleanup_token,
                    "cleanup-pending",
                    names[:self._cleanup_step],
                    names[self._cleanup_step] if self._cleanup_step < len(names)
                    else "authority-release",
                    f"{type(exc).__name__}: {exc}",
                )
                self._cleanup_receipt = receipt
            raise Light1DCleanupPending(receipt, exc) from exc
        with self._lock:
            receipt = Light1DReleaseReceipt(
                self.grant_id,
                self.owner,
                self.generation,
                released,
                self._cleanup_reason,
                self._cleanup_token,
            )
            self._cleanup_hooks = None
            self._release_receipt = receipt
            self._state = Light1DLeaseState.RELEASED
            return receipt

    def release(
        self,
        *,
        reason: str,
        hooks: Light1DCleanupHooks | None = None,
    ) -> Light1DReleaseReceipt:
        with self._cleanup_lock:
            with self._lock:
                if self._release_receipt is not None:
                    if str(reason) != self._release_receipt.reason:
                        raise ValueError("released light-1D lease reason cannot change")
                    return self._release_receipt
                if self._state is Light1DLeaseState.CLEANUP_PENDING:
                    raise RuntimeError(
                        "cleanup-pending release requires its retry token"
                    )
                if self._cleanup_token is None:
                    self._cleanup_token = Light1DCleanupToken(
                        self.grant_id, self.generation, 1,
                    )
                    self._cleanup_reason = str(reason)
                    self._cleanup_hooks = hooks or Light1DCleanupHooks()
                self._state = Light1DLeaseState.FENCED
            return self._run_cleanup()

    def retry_cleanup(
        self,
        token: Light1DCleanupToken,
        *,
        hooks: Light1DCleanupHooks | None = None,
    ) -> Light1DReleaseReceipt:
        with self._cleanup_lock:
            with self._lock:
                if token is not self._cleanup_token:
                    raise RuntimeError(
                        "light-1D cleanup requires the exact retry token"
                    )
                if self._release_receipt is not None:
                    return self._release_receipt
                if self._state is not Light1DLeaseState.CLEANUP_PENDING:
                    raise RuntimeError("light-1D lease has no cleanup-pending work")
                if hooks is not None and hooks is not self._cleanup_hooks:
                    raise RuntimeError("light-1D cleanup hooks are frozen")
            return self._run_cleanup()

    def retry_release(
        self,
        token: Light1DCleanupToken,
        *,
        hooks: Light1DCleanupHooks | None = None,
    ) -> Light1DReleaseReceipt:
        """Compatibility alias for the exact frozen cleanup retry."""
        return self.retry_cleanup(token, hooks=hooks)


def acquire_light_1d_retention(
    authority: SessionResourceAuthority,
    *,
    owner: str,
    generation: int,
    layout: Light1DLayout,
    requested_rows: int,
    compatibility_byte_ceiling: int,
    gui_thread_id: int,
) -> Light1DRetentionLease:
    if type(authority) is not SessionResourceAuthority:
        raise TypeError(
            "light-1D acquisition requires exact SessionResourceAuthority"
        )
    if type(layout) is not Light1DLayout:
        raise TypeError("layout must be exact Light1DLayout")
    requested_rows = _nonnegative_int("requested_rows", requested_rows)
    try:
        if isinstance(generation, bool) or int(generation) != generation:
            raise ValueError
        generation = int(generation)
    except (TypeError, ValueError):
        raise ValueError("generation must be an integer") from None
    try:
        if isinstance(gui_thread_id, bool) or int(gui_thread_id) != gui_thread_id:
            raise ValueError
        gui_thread_id = int(gui_thread_id)
    except (TypeError, ValueError):
        raise ValueError("gui_thread_id must be an integer") from None
    if not isinstance(owner, str) or not owner:
        raise ValueError("light-1D owner must be a non-empty string")
    if layout.per_row_unique_ndarray_bytes <= 0:
        raise ValueError("light-1D layout must charge per-row owned buffers")
    grant_id, row_cap, reserved, reservation_claim = authority._reserve_light(
        owner=owner,
        generation=generation,
        layout=layout,
        requested_rows=requested_rows,
        compatibility_byte_ceiling=compatibility_byte_ceiling,
    )
    try:
        return Light1DRetentionLease(
            authority,
            grant_id=grant_id,
            owner=owner,
            generation=generation,
            layout=layout,
            requested_rows=requested_rows,
            row_cap=row_cap,
            reserved_ndarray_bytes=reserved,
            gui_thread_id=gui_thread_id,
            _reservation_claim=reservation_claim,
        )
    except BaseException:
        authority._cancel_light_construction(
            grant_id,
            owner=owner,
            generation=generation,
            claim=reservation_claim,
        )
        raise
