"""Exact physical-root accounting for bounded in-process array owners.

The utility is deliberately independent of HDF5 and GUI packages. It counts
the ultimate buffer owner, not the size of an arbitrarily small NumPy view,
and keeps that owner alive for every committed semantic reference.

Reservations admit the final unique-root set against the byte limit. Callers
must release evicted roots before reserving replacement capacity.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import RLock
from types import MappingProxyType
from typing import Hashable, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class PhysicalRootFact:
    root: object
    nbytes: int


def physical_root_fact(value: object) -> PhysicalRootFact:
    """Return the ultimate supported buffer owner and its exact byte size."""

    current = value
    seen: set[int] = set()
    while True:
        identity = id(current)
        if identity in seen:
            raise ValueError("physical buffer ownership contains a cycle")
        seen.add(identity)
        if isinstance(current, np.ndarray):
            base = current.base
            if base is None:
                return PhysicalRootFact(current, int(current.nbytes))
            current = base
            continue
        if type(current) is memoryview:
            try:
                owner = current.obj
            except (AttributeError, ValueError) as error:
                raise ValueError("memoryview ownership is unavailable") from error
            if owner is None:
                raise ValueError("memoryview has no exact physical owner")
            current = owner
            continue
        if type(current) in {bytes, bytearray}:
            return PhysicalRootFact(current, len(current))
        raise ValueError(
            "buffer root must be an ndarray, memoryview, bytes, or bytearray"
        )


class _GatePhase(Enum):
    BASE = "base"
    LEGACY = "legacy"


@dataclass(frozen=True, slots=True, eq=False)
class _LeaseToken:
    semantic: Hashable


@dataclass(frozen=True, slots=True, eq=False)
class _RootEntry:
    identity: int
    root: object
    nbytes: int
    references: int


@dataclass(frozen=True, slots=True, eq=False)
class _SemanticBinding:
    semantic: Hashable
    semantic_hash: int
    root_identity: int
    token: _LeaseToken


@dataclass(frozen=True, slots=True, eq=False)
class _AuthorityState:
    roots: tuple[_RootEntry, ...]
    bindings: tuple[_SemanticBinding, ...]
    root_index: Mapping[int, _RootEntry]
    token_index: Mapping[int, _SemanticBinding]
    semantic_index: Mapping[int, tuple[_SemanticBinding, ...]]
    retained_bytes: int
    gate: object | None
    phase: _GatePhase
    closed: bool


def _indexed_state(
    roots: tuple[_RootEntry, ...],
    bindings: tuple[_SemanticBinding, ...],
    gate: object | None,
    phase: _GatePhase,
    closed: bool,
) -> _AuthorityState:
    """Build one callback-free immutable index for an authority graph."""

    root_index: dict[int, _RootEntry] = {}
    retained_bytes = 0
    for entry in roots:
        if entry.identity != id(entry.root) or entry.identity in root_index:
            raise RuntimeError("physical-root authority state is inconsistent")
        root_index[entry.identity] = entry
        retained_bytes += entry.nbytes
    token_index: dict[int, _SemanticBinding] = {}
    semantic_index: dict[int, list[_SemanticBinding]] = {}
    references: dict[int, int] = {}
    for binding in bindings:
        token_identity = id(binding.token)
        if token_identity in token_index:
            raise RuntimeError("physical-root authority state is inconsistent")
        token_index[token_identity] = binding
        semantic_index.setdefault(binding.semantic_hash, []).append(binding)
        references[binding.root_identity] = (
            references.get(binding.root_identity, 0) + 1
        )
    if any(
        references.get(entry.identity, 0) != entry.references
        for entry in roots
    ) or any(identity not in root_index for identity in references):
        raise RuntimeError("physical-root authority state is inconsistent")
    return _AuthorityState(
        roots,
        bindings,
        MappingProxyType(root_index),
        MappingProxyType(token_index),
        MappingProxyType({
            value: tuple(bucket) for value, bucket in semantic_index.items()
        }),
        retained_bytes,
        gate,
        phase,
        closed,
    )


def _state_with_graph(
    state: _AuthorityState,
    *,
    gate: object | None,
    phase: _GatePhase,
    closed: bool | None = None,
) -> _AuthorityState:
    """Reuse one exact immutable graph while changing only shell state."""

    return _AuthorityState(
        state.roots,
        state.bindings,
        state.root_index,
        state.token_index,
        state.semantic_index,
        state.retained_bytes,
        gate,
        phase,
        state.closed if closed is None else closed,
    )








def _root_entry(
    state: _AuthorityState, identity: int,
) -> _RootEntry | None:
    entry = state.root_index.get(identity)
    if entry is not None and entry.identity == identity:
        return entry
    return None


def _binding_for_token(
    state: _AuthorityState, token: _LeaseToken,
) -> _SemanticBinding | None:
    binding = state.token_index.get(id(token))
    if binding is not None and binding.token is token:
        return binding
    return None


def _validate_semantic(semantic: Hashable) -> int:
    try:
        return hash(semantic)
    except TypeError as error:
        raise TypeError("physical-root semantic must be hashable") from error


def _binding_for_semantic(
    state: _AuthorityState,
    semantic: Hashable,
) -> _SemanticBinding | None:
    semantic_hash = _validate_semantic(semantic)
    for binding in state.semantic_index.get(semantic_hash, ()):
        if binding.semantic is semantic or binding.semantic == semantic:
            return binding
    return None


def _validate_projection(
    state: _AuthorityState,
    victim_tokens: tuple[_LeaseToken, ...],
    claims: dict[object, int],
    incoming: dict[Hashable, PhysicalRootFact],
    limit: int,
) -> int:
    """Validate a final projection outside the authority swap lock."""

    victim_ids: set[int] = set()
    victim_roots: dict[int, int] = {}
    for token in victim_tokens:
        token_identity = id(token)
        if token_identity in victim_ids:
            raise RuntimeError("physical-root authority state is inconsistent")
        victim_ids.add(token_identity)
        binding = _binding_for_token(state, token)
        if binding is None:
            raise RuntimeError("physical-root authority state is inconsistent")
        victim_roots[binding.root_identity] = (
            victim_roots.get(binding.root_identity, 0) + 1
        )
    projected = state.retained_bytes
    for identity, count in victim_roots.items():
        entry = _root_entry(state, identity)
        if entry is None or count > entry.references:
            raise RuntimeError("physical-root authority state is inconsistent")
        if count == entry.references:
            projected -= entry.nbytes
    unique: dict[int, PhysicalRootFact] = {}
    for semantic, fact in incoming.items():
        existing_binding = _binding_for_semantic(state, semantic)
        if (
            existing_binding is not None
            and id(existing_binding.token) not in victim_ids
        ):
            raise ValueError("physical-root semantic is already retained")
        identity = id(fact.root)
        existing = _root_entry(state, identity)
        if existing is not None and (
            existing.root is not fact.root or existing.nbytes != fact.nbytes
        ):
            raise ValueError("physical-root identity changed")
        prior = unique.get(identity)
        if prior is not None and (
            prior.root is not fact.root or prior.nbytes != fact.nbytes
        ):
            raise ValueError("physical-root byte size changed")
        if prior is None:
            unique[identity] = fact
            if (
                existing is None
                or victim_roots.get(identity, 0) == existing.references
            ):
                projected += fact.nbytes
    projected += sum(claims.values())
    if projected > limit:
        raise ValueError("retained physical-root bytes exceed limit")
    return projected


def _replacement_state(
    state: _AuthorityState,
    victim_tokens: tuple[_LeaseToken, ...],
    incoming: tuple[tuple[Hashable, PhysicalRootFact, "PhysicalRootLease"], ...],
    projected_bytes: int,
    *,
    gate: object | None,
    phase: _GatePhase,
) -> _AuthorityState:
    """Apply one validated graph delta without re-walking retained bindings."""

    victim_ids: set[int] = set()
    root_deltas: dict[int, int] = {}
    for token in victim_tokens:
        token_identity = id(token)
        if token_identity in victim_ids:
            raise RuntimeError("physical-root authority state is inconsistent")
        victim_ids.add(token_identity)
        binding = _binding_for_token(state, token)
        if binding is None:
            raise RuntimeError("physical-root authority state is inconsistent")
        root_deltas[binding.root_identity] = (
            root_deltas.get(binding.root_identity, 0) - 1
        )

    incoming_bindings: list[_SemanticBinding] = []
    incoming_facts: dict[int, PhysicalRootFact] = {}
    for semantic, fact, lease in incoming:
        identity = id(fact.root)
        prior = incoming_facts.get(identity)
        if prior is not None and (
            prior.root is not fact.root or prior.nbytes != fact.nbytes
        ):
            raise RuntimeError("physical-root authority state is inconsistent")
        incoming_facts[identity] = fact
        root_deltas[identity] = root_deltas.get(identity, 0) + 1
        incoming_bindings.append(
            _SemanticBinding(
                semantic,
                _validate_semantic(semantic),
                identity,
                lease._token,
            )
        )

    root_replacements: dict[int, _RootEntry | None] = {}
    new_roots: list[_RootEntry] = []
    retained_bytes = state.retained_bytes
    for identity, delta in root_deltas.items():
        existing = _root_entry(state, identity)
        if existing is None:
            fact = incoming_facts.get(identity)
            if fact is None or delta <= 0:
                raise RuntimeError("physical-root authority state is inconsistent")
            entry = _RootEntry(identity, fact.root, fact.nbytes, delta)
            root_replacements[identity] = entry
            new_roots.append(entry)
            retained_bytes += fact.nbytes
            continue
        fact = incoming_facts.get(identity)
        if fact is not None and (
            existing.root is not fact.root or existing.nbytes != fact.nbytes
        ):
            raise RuntimeError("physical-root authority state is inconsistent")
        references = existing.references + delta
        if references < 0:
            raise RuntimeError("physical-root authority state is inconsistent")
        if references == 0:
            root_replacements[identity] = None
            retained_bytes -= existing.nbytes
        elif references != existing.references:
            root_replacements[identity] = _RootEntry(
                identity, existing.root, existing.nbytes, references,
            )

    if retained_bytes != projected_bytes:
        raise RuntimeError("physical-root authority projection is inconsistent")
    if root_replacements:
        roots = tuple(
            replacement
            for entry in state.roots
            if (
                replacement := root_replacements.get(entry.identity, entry)
            ) is not None
        ) + tuple(new_roots)
    else:
        roots = state.roots
    bindings = (
        tuple(
            binding
            for binding in state.bindings
            if id(binding.token) not in victim_ids
        )
        if victim_ids
        else state.bindings
    ) + tuple(incoming_bindings)

    root_index = dict(state.root_index)
    for identity, replacement in root_replacements.items():
        if replacement is None:
            root_index.pop(identity, None)
        else:
            root_index[identity] = replacement
    token_index = dict(state.token_index)
    semantic_index = dict(state.semantic_index)
    for token in victim_tokens:
        binding = token_index.pop(id(token), None)
        if binding is None or binding.token is not token:
            raise RuntimeError("physical-root authority state is inconsistent")
        bucket = semantic_index.get(binding.semantic_hash)
        if bucket is None:
            raise RuntimeError("physical-root authority state is inconsistent")
        replacement_bucket = tuple(
            item for item in bucket if item.token is not token
        )
        if len(replacement_bucket) + 1 != len(bucket):
            raise RuntimeError("physical-root authority state is inconsistent")
        if replacement_bucket:
            semantic_index[binding.semantic_hash] = replacement_bucket
        else:
            del semantic_index[binding.semantic_hash]
    for binding in incoming_bindings:
        token_identity = id(binding.token)
        if token_identity in token_index:
            raise RuntimeError("physical-root authority state is inconsistent")
        token_index[token_identity] = binding
        semantic_index[binding.semantic_hash] = (
            semantic_index.get(binding.semantic_hash, ()) + (binding,)
        )
    return _AuthorityState(
        roots,
        bindings,
        MappingProxyType(root_index),
        MappingProxyType(token_index),
        MappingProxyType(semantic_index),
        retained_bytes,
        gate,
        phase,
        state.closed,
    )


class PhysicalRootLease:
    __slots__ = ("_authority", "_semantic", "_token", "_released")

    def __init__(
        self, authority: "PhysicalRootAuthority", semantic: Hashable,
    ) -> None:
        self._authority = authority
        self._semantic = semantic
        self._token = _LeaseToken(semantic)
        self._released = False

    @property
    def released(self) -> bool:
        return self._released

    def release(self) -> None:
        if self._released:
            return
        # Preserve the accepted one-positional private fault surface while
        # making retirement token-primary and semantic-secondary.
        self._authority._release(self._token)
        self._released = True


class PhysicalRootReservation:
    """One all-or-none set of provisional capacity and root claims."""

    __slots__ = (
        "_authority", "_operation", "_claims", "_roots", "_active",
        "_committed", "_committed_leases",
    )

    def __init__(
        self,
        authority: "PhysicalRootAuthority",
        operation: object | None = None,
    ) -> None:
        self._authority = authority
        self._operation = object() if operation is None else operation
        self._claims: dict[object, int] = {}
        self._roots: dict[Hashable, PhysicalRootFact] = {}
        self._active = True
        self._committed = False
        self._committed_leases: dict[Hashable, PhysicalRootLease] = {}

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("physical-root reservation is no longer active")

    def claim(self, nbytes: int) -> object:
        self._require_active()
        if type(nbytes) is not int or nbytes < 0:
            raise TypeError("physical-root capacity must be a nonnegative integer")
        token = object()
        self._claims[token] = nbytes
        try:
            self._authority._qualify(
                self._claims, self._roots, self._operation,
            )
        except BaseException:
            self._claims.pop(token, None)
            raise
        return token

    def bind(
        self, token: object, value: object, semantic: Hashable,
    ) -> PhysicalRootFact:
        self._require_active()
        if token not in self._claims:
            raise ValueError("physical-root capacity token is not owned")
        _validate_semantic(semantic)
        if semantic in self._roots:
            raise ValueError("physical-root semantic is duplicated")
        fact = physical_root_fact(value)
        expected = self._claims[token]
        if fact.nbytes != expected:
            raise ValueError("allocated physical root differs from reserved bytes")
        del self._claims[token]
        self._roots[semantic] = fact
        try:
            self._authority._qualify(
                self._claims, self._roots, self._operation,
            )
        except BaseException:
            self._roots.pop(semantic, None)
            self._claims[token] = expected
            raise
        return fact

    def reserve(
        self, value: object, semantic: Hashable,
    ) -> PhysicalRootFact:
        self._require_active()
        _validate_semantic(semantic)
        if semantic in self._roots:
            raise ValueError("physical-root semantic is duplicated")
        fact = physical_root_fact(value)
        self._roots[semantic] = fact
        try:
            self._authority._qualify(
                self._claims, self._roots, self._operation,
            )
        except BaseException:
            self._roots.pop(semantic, None)
            raise
        return fact

    @property
    def projected_bytes(self) -> int:
        self._require_active()
        return self._authority._qualify(
            self._claims, self._roots, self._operation,
        )

    def commit(self) -> dict[Hashable, PhysicalRootLease]:
        self._require_active()
        if self._claims:
            raise RuntimeError("unbound physical-root capacity remains")
        leases = self._authority._commit(
            self._roots, self._committed_leases, self._operation,
        )
        self._active = False
        self._committed = True
        self._roots = {}
        return leases

    def rollback(self) -> None:
        if not self._active and not self._committed_leases:
            return
        was_active = self._active
        first_error: BaseException | None = None
        if was_active:
            # If a receipt was prepared but never published, clearing the
            # legacy gate first makes its incoming leases safely stale.
            self._authority._cancel_reservation(self._operation)
        pending = dict(self._committed_leases)
        for _attempt in range(2):
            failed: dict[Hashable, PhysicalRootLease] = {}
            for semantic, lease in pending.items():
                try:
                    lease.release()
                except BaseException as error:
                    if first_error is None:
                        first_error = error
                    failed[semantic] = lease
            pending = failed
            if not pending:
                break
        self._committed_leases.clear()
        self._committed_leases.update(pending)
        self._claims.clear()
        self._roots.clear()
        self._active = False
        self._committed = bool(pending)
        if pending:
            assert first_error is not None
            raise first_error

    def __enter__(self) -> "PhysicalRootReservation":
        self._require_active()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._active:
            self.rollback()


class PhysicalRootAuthority:
    """Bounded, reference-counted authority over unique physical roots."""

    def __init__(self, limit_bytes: int) -> None:
        if type(limit_bytes) is not int or limit_bytes < 0:
            raise TypeError("physical-root limit must be a nonnegative integer")
        self._limit = limit_bytes
        self._state = _indexed_state(
            (), (), None, _GatePhase.BASE, False,
        )
        self._lock = RLock()

    @property
    def limit_bytes(self) -> int:
        return self._limit

    @property
    def retained_bytes(self) -> int:
        state = self._accounting_state()
        return state.retained_bytes

    @property
    def retained_roots(self) -> tuple[object, ...]:
        state = self._accounting_state()
        return tuple(entry.root for entry in state.roots)

    @property
    def retained_root_count(self) -> int:
        return len(self._accounting_state().roots)

    @property
    def semantic_references(self) -> int:
        return len(self._accounting_state().bindings)

    def _accounting_state(self) -> _AuthorityState:
        state = self._snapshot_state()
        return state

    def _snapshot_state(self) -> _AuthorityState:
        with self._lock:
            return self._state


    def reserve(self) -> PhysicalRootReservation:
        operation = object()
        reservation = PhysicalRootReservation(self, operation)
        with self._lock:
            state = self._state
            if state.closed:
                raise RuntimeError("physical-root authority is closed")
            if state.gate is not None:
                raise RuntimeError("physical-root reservation is already active")
            self._state = _state_with_graph(
                state, gate=operation, phase=_GatePhase.LEGACY,
            )
        return reservation

    def _cancel_reservation(self, operation: object | None = None) -> None:
        with self._lock:
            state = self._state
            if state.gate is None:
                return
            if state.phase is not _GatePhase.LEGACY:
                raise RuntimeError("physical-root authority is busy")
            if operation is not None and state.gate is not operation:
                raise RuntimeError("physical-root reservation is not active")
            self._state = _state_with_graph(
                state, gate=None, phase=_GatePhase.BASE,
            )

    def _qualify(
        self,
        claims: dict[object, int],
        roots: dict[Hashable, PhysicalRootFact],
        operation: object | None = None,
    ) -> int:
        while True:
            with self._lock:
                state = self._state
                if state.closed:
                    raise RuntimeError("physical-root authority is closed")
                if state.phase is not _GatePhase.LEGACY:
                    raise RuntimeError("physical-root reservation is not active")
                if operation is not None and state.gate is not operation:
                    raise RuntimeError("physical-root reservation is not active")
            projected = _validate_projection(
                state, (), claims, roots, self._limit,
            )
            with self._lock:
                if self._state is state:
                    return projected

    def _commit(
        self,
        roots: dict[Hashable, PhysicalRootFact],
        receipt: dict[Hashable, PhysicalRootLease],
        operation: object | None = None,
    ) -> dict[Hashable, PhysicalRootLease]:
        while True:
            with self._lock:
                state = self._state
                if state.phase is not _GatePhase.LEGACY:
                    raise RuntimeError("physical-root reservation is not active")
                if operation is not None and state.gate is not operation:
                    raise RuntimeError("physical-root reservation is not active")
            if receipt:
                raise RuntimeError("physical-root commit receipt is not empty")
            projected_bytes = _validate_projection(
                state, (), {}, roots, self._limit,
            )
            incoming = tuple(
                (semantic, fact, PhysicalRootLease(self, semantic))
                for semantic, fact in roots.items()
            )
            target = _replacement_state(
                state,
                (),
                incoming,
                projected_bytes,
                gate=None,
                phase=_GatePhase.BASE,
            )
            leases = {
                semantic: lease for semantic, _fact, lease in incoming
            }
            # Keep user hashing/equality outside the pointer-swap lock.
            receipt.update(leases)
            with self._lock:
                if self._state is state:
                    self._state = target
                    return leases
            receipt.clear()

    def _release(self, token: object) -> None:
        """Retire one exact opaque lease token."""

        if type(token) is not _LeaseToken:
            raise TypeError("physical-root release requires an opaque lease token")
        semantic = token.semantic
        while True:
            with self._lock:
                state = self._state
            binding: _SemanticBinding | None = None
            candidate = _binding_for_token(state, token)
            if candidate is not None and candidate.semantic is semantic:
                binding = candidate
            if binding is None:
                return
            entry = _root_entry(state, binding.root_identity)
            if entry is None:
                raise RuntimeError("physical-root authority state is inconsistent")
            bindings = tuple(
                item for item in state.bindings if item.token is not binding.token
            )
            roots: list[_RootEntry] = []
            for item in state.roots:
                if item.identity != entry.identity:
                    roots.append(item)
                elif item.references > 1:
                    roots.append(
                        _RootEntry(
                            item.identity, item.root, item.nbytes,
                            item.references - 1,
                        )
                    )
            replacement = _indexed_state(
                tuple(roots), bindings, state.gate, state.phase, state.closed,
            )
            with self._lock:
                if self._state is state:
                    self._state = replacement
                    return

    def close(self) -> None:
        old_state: _AuthorityState
        with self._lock:
            old_state = self._state
            if old_state.closed:
                return
            self._state = _indexed_state(
                (), (), None, _GatePhase.BASE, True,
            )
        # Keep the losing state strong through lock exit so arbitrary root or
        # semantic destructors cannot run in the pointer-swap critical section.
        del old_state


__all__ = [
    "PhysicalRootAuthority",
    "PhysicalRootFact",
    "PhysicalRootLease",
    "PhysicalRootReservation",
    "physical_root_fact",
]
