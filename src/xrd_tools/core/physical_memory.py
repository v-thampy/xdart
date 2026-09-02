"""Exact physical-root accounting for bounded in-process array owners.

The utility is deliberately independent of HDF5 and GUI packages. It counts
the ultimate buffer owner, not the size of an arbitrarily small NumPy view,
and keeps that owner alive for every committed semantic reference.

The byte limit applies to the final committed set of unique physical roots.
An exchange stages incoming allocations outside the authority while its old
roots remain live, so transient process ownership can approach old plus new
(roughly twice the configured limit). Authority accounting describes only the
published graph and never claims that staging overlap is a hard-memory cap.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import RLock, local
from types import MappingProxyType
from typing import Hashable, Mapping
from weakref import ReferenceType, ref

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
    STAGED = "staged"
    COMMIT_PENDING = "commit-pending"


class PhysicalRootExchangePhase(Enum):
    """Externally observable exact phase of one exchange journal."""

    OPEN = "open"
    PREPARED = "prepared"
    STAGED = "staged"
    COMMIT_PENDING = "commit-pending"
    ACCEPTED = "accepted"
    ROLLED_BACK = "rolled-back"
    FAILED = "failed"
    DRIFTED = "drifted"


@dataclass(frozen=True, slots=True, eq=False)
class _LeaseToken:
    semantic: Hashable


@dataclass(frozen=True, slots=True, eq=False, weakref_slot=True)
class _TerminalMarker:
    direction: PhysicalRootExchangePhase


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
    terminal_evidence: tuple[ReferenceType[_TerminalMarker], ...]


def _indexed_state(
    roots: tuple[_RootEntry, ...],
    bindings: tuple[_SemanticBinding, ...],
    gate: object | None,
    phase: _GatePhase,
    closed: bool,
    terminal_evidence: tuple[ReferenceType[_TerminalMarker], ...],
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
        terminal_evidence,
    )


def _state_with_graph(
    state: _AuthorityState,
    *,
    gate: object | None,
    phase: _GatePhase,
    closed: bool | None = None,
    terminal_evidence: tuple[ReferenceType[_TerminalMarker], ...] | None = None,
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
        (
            state.terminal_evidence
            if terminal_evidence is None
            else terminal_evidence
        ),
    )


def _is_exchange_gate(state: _AuthorityState) -> bool:
    return state.phase in {_GatePhase.STAGED, _GatePhase.COMMIT_PENDING}


def _has_terminal_marker(
    state: _AuthorityState, marker: _TerminalMarker,
) -> bool:
    return any(marker_ref() is marker for marker_ref in state.terminal_evidence)


def _terminal_evidence_with(
    state: _AuthorityState, marker: _TerminalMarker,
) -> tuple[ReferenceType[_TerminalMarker], ...]:
    """Prune dead evidence and append ``marker`` using identity only."""

    refs: list[ReferenceType[_TerminalMarker]] = []
    live: list[_TerminalMarker] = []
    marker_present = False
    for marker_ref in state.terminal_evidence:
        candidate = marker_ref()
        if candidate is None:
            continue
        if any(candidate is prior for prior in live):
            continue
        live.append(candidate)
        refs.append(marker_ref)
        if candidate is marker:
            marker_present = True
    if not marker_present:
        refs.append(ref(marker))
    return tuple(refs)


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
        state.terminal_evidence,
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


@dataclass(frozen=True, slots=True, eq=False)
class _ExchangeReceipt:
    leases: Mapping[Hashable, PhysicalRootLease]
    projected_bytes: int
    staged: _AuthorityState
    pending: _AuthorityState
    accepted_marker: _TerminalMarker
    rolled_back_marker: _TerminalMarker


@dataclass(frozen=True, slots=True, eq=False, weakref_slot=True)
class _ExchangeJournal:
    """One immutable generation of an open exchange journal."""

    generation: int
    base: _AuthorityState
    victims: tuple[PhysicalRootLease, ...]
    victim_tokens: tuple[_LeaseToken, ...]
    operation: object
    claims: Mapping[object, int]
    roots: Mapping[Hashable, PhysicalRootFact]
    receipt: _ExchangeReceipt | None
    terminal_intent: PhysicalRootExchangePhase | None


@dataclass(frozen=True, slots=True, eq=False)
class _TerminalRecord:
    """Compact local terminal result, independent of authority history."""

    phase: PhysicalRootExchangePhase
    leases: Mapping[Hashable, PhysicalRootLease] | None
    projected_bytes: int | None


_FAILED_TERMINAL_RECORD = _TerminalRecord(
    PhysicalRootExchangePhase.FAILED, None, None,
)


@dataclass(slots=True, eq=False)
class _ExchangeOperation:
    token: object
    journal: _ExchangeJournal | _TerminalRecord
    held: list[object]


_EXCHANGE_TLS = local()


class PhysicalRootExchange:
    """Failure-atomic replacement of an exact tuple of retained leases.

    Once :meth:`prepare` publishes a receipt this object is a linear journal:
    exactly one terminal direction can win and the same direction may be
    retried after any fault cut.  Staging may transiently retain both the old
    graph and incoming roots; ``final_projected_bytes`` describes only the
    final committed graph.

    Deliberately, there is no destructor or authority-driven auto-rollback.
    The eventual reader/cache owner must durably settle the journal itself.
    """

    __slots__ = ("_authority", "_lock", "_active_operation", "_journal")

    def __init__(
        self,
        authority: "PhysicalRootAuthority",
        base: _AuthorityState,
        victims: tuple[PhysicalRootLease, ...],
    ) -> None:
        self._authority = authority
        self._lock = RLock()
        self._active_operation: object | None = None
        self._journal: _ExchangeJournal | _TerminalRecord = _ExchangeJournal(
            0,
            base,
            victims,
            tuple(lease._token for lease in victims),
            object(),
            MappingProxyType({}),
            MappingProxyType({}),
            None,
            None,
        )

    def __copy__(self) -> "PhysicalRootExchange":
        raise TypeError("physical-root exchange cannot be copied")

    def __deepcopy__(self, _memo: object) -> "PhysicalRootExchange":
        raise TypeError("physical-root exchange cannot be copied")

    def __reduce__(self) -> object:
        raise TypeError("physical-root exchange cannot be serialized")

    def __reduce_ex__(self, _protocol: int) -> object:
        raise TypeError("physical-root exchange cannot be serialized")

    @staticmethod
    def _advanced(
        journal: _ExchangeJournal,
        *,
        claims: Mapping[object, int] | None = None,
        roots: Mapping[Hashable, PhysicalRootFact] | None = None,
        receipt: _ExchangeReceipt | None = None,
        replace_receipt: bool = False,
        terminal_intent: PhysicalRootExchangePhase | None = None,
        replace_intent: bool = False,
    ) -> _ExchangeJournal:
        return _ExchangeJournal(
            journal.generation + 1,
            journal.base,
            journal.victims,
            journal.victim_tokens,
            journal.operation,
            journal.claims if claims is None else claims,
            journal.roots if roots is None else roots,
            receipt if replace_receipt else journal.receipt,
            terminal_intent if replace_intent else journal.terminal_intent,
        )

    @staticmethod
    def _terminal_error(record: _TerminalRecord) -> RuntimeError:
        if record.phase is PhysicalRootExchangePhase.ACCEPTED:
            return RuntimeError("physical-root exchange was accepted")
        if record.phase is PhysicalRootExchangePhase.ROLLED_BACK:
            return RuntimeError("physical-root exchange was rolled back")
        return RuntimeError("physical-root exchange failed")

    def _begin_operation(
        self,
        terminal_intent: PhysicalRootExchangePhase | None = None,
        *,
        install_terminal_intent: bool = True,
    ) -> _ExchangeOperation:
        """Claim this exchange after installing the module-wide TLS guard."""

        if getattr(_EXCHANGE_TLS, "token", None) is not None:
            raise RuntimeError("physical-root exchange callback reentry")
        token = object()
        # Do not snapshot a callback-bearing journal before the lock. A racing
        # compaction could otherwise make overwriting that speculative ref run
        # its finalizers inside the exchange lock.
        operation = _ExchangeOperation(token, _FAILED_TERMINAL_RECORD, [])
        _EXCHANGE_TLS.token = token
        try:
            acquired = self._lock.acquire(blocking=False)
        except BaseException:
            del _EXCHANGE_TLS.token
            raise
        if not acquired:
            del _EXCHANGE_TLS.token
            raise RuntimeError("physical-root exchange is busy")
        succeeded = False
        try:
            current = self._journal
            operation.journal = current
            if isinstance(current, _ExchangeJournal) and terminal_intent is not None:
                prior = current.terminal_intent
                if prior is not None and prior is not terminal_intent:
                    if prior is PhysicalRootExchangePhase.ACCEPTED:
                        raise RuntimeError(
                            "physical-root exchange acceptance is pending"
                        )
                    raise RuntimeError(
                        "physical-root exchange rollback is pending"
                    )
            if self._active_operation is not None:
                raise RuntimeError("physical-root exchange is busy")
            if (
                isinstance(current, _ExchangeJournal)
                and terminal_intent is not None
                and install_terminal_intent
                and current.terminal_intent is None
                and (
                    terminal_intent is PhysicalRootExchangePhase.ROLLED_BACK
                    or current.receipt is not None
                )
            ):
                replacement = self._advanced(
                    current,
                    terminal_intent=terminal_intent,
                    replace_intent=True,
                )
                operation.held.append(current)
                self._journal = replacement
                current = replacement
                operation.journal = replacement
            self._active_operation = token
            succeeded = True
            return operation
        finally:
            self._lock.release()
            if not succeeded:
                del _EXCHANGE_TLS.token
                operation.held.clear()

    def _finish_operation(self, operation: _ExchangeOperation) -> None:
        """Clear ownership before dropping any callback-bearing graph refs."""

        with self._lock:
            if self._active_operation is operation.token:
                self._active_operation = None
        if getattr(_EXCHANGE_TLS, "token", None) is operation.token:
            del _EXCHANGE_TLS.token
        operation.journal = _FAILED_TERMINAL_RECORD
        operation.held.clear()

    def _check_operation(
        self,
        operation: _ExchangeOperation,
        journal: _ExchangeJournal | _TerminalRecord,
    ) -> None:
        with self._lock:
            if (
                self._active_operation is not operation.token
                or self._journal is not journal
            ):
                raise RuntimeError("physical-root exchange journal drifted")

    def _check_open_base(
        self,
        operation: _ExchangeOperation,
        journal: _ExchangeJournal,
    ) -> None:
        self._check_operation(operation, journal)
        if self._authority._snapshot_state() is not journal.base:
            raise RuntimeError("physical-root exchange base state drifted")
        self._check_operation(operation, journal)

    def _install_open_journal(
        self,
        operation: _ExchangeOperation,
        expected: _ExchangeJournal,
        replacement: _ExchangeJournal,
    ) -> None:
        """Install one COW generation against the exact authority base."""

        with self._authority._lock:
            if self._authority._state is not expected.base:
                raise RuntimeError("physical-root exchange base state drifted")
            with self._lock:
                if (
                    self._active_operation is not operation.token
                    or self._journal is not expected
                ):
                    raise RuntimeError("physical-root exchange journal drifted")
                operation.held.append(expected)
                self._journal = replacement
                operation.journal = replacement

    def _install_terminal_intent(
        self,
        operation: _ExchangeOperation,
        expected: _ExchangeJournal,
        direction: PhysicalRootExchangePhase,
    ) -> _ExchangeJournal:
        """Install one terminal winner after scientific state qualification."""

        prior = expected.terminal_intent
        if prior is direction:
            return expected
        if prior is PhysicalRootExchangePhase.ACCEPTED:
            raise RuntimeError("physical-root exchange acceptance is pending")
        if prior is PhysicalRootExchangePhase.ROLLED_BACK:
            raise RuntimeError("physical-root exchange rollback is pending")
        replacement = self._advanced(
            expected,
            terminal_intent=direction,
            replace_intent=True,
        )
        with self._lock:
            if (
                self._active_operation is not operation.token
                or self._journal is not expected
            ):
                raise RuntimeError("physical-root exchange journal drifted")
            operation.held.append(expected)
            self._journal = replacement
            operation.journal = replacement
        return replacement

    def _install_terminal(
        self,
        operation: _ExchangeOperation,
        expected: _ExchangeJournal,
        record: _TerminalRecord,
    ) -> _TerminalRecord:
        """Publish terminal locally before the receipt markers can be dropped."""

        with self._lock:
            if (
                self._active_operation is not operation.token
                or self._journal is not expected
            ):
                raise RuntimeError("physical-root exchange journal drifted")
            operation.held.append(expected)
            self._journal = record
            operation.journal = record
        return record

    @staticmethod
    def _require_open(journal: _ExchangeJournal | _TerminalRecord) -> _ExchangeJournal:
        if not isinstance(journal, _ExchangeJournal) or journal.receipt is not None:
            raise RuntimeError("physical-root exchange is no longer open")
        return journal

    @staticmethod
    def _require_receipt(journal: _ExchangeJournal) -> _ExchangeReceipt:
        receipt = journal.receipt
        if receipt is None:
            raise RuntimeError("physical-root exchange is not prepared")
        return receipt

    @staticmethod
    def _recognized_terminal(
        state: _AuthorityState,
        receipt: _ExchangeReceipt,
    ) -> PhysicalRootExchangePhase | None:
        if _has_terminal_marker(state, receipt.accepted_marker):
            return PhysicalRootExchangePhase.ACCEPTED
        if _has_terminal_marker(state, receipt.rolled_back_marker):
            return PhysicalRootExchangePhase.ROLLED_BACK
        return None

    @staticmethod
    def _terminal_replacement(
        journal: _ExchangeJournal,
        receipt: _ExchangeReceipt,
        state: _AuthorityState,
        direction: PhysicalRootExchangePhase,
    ) -> _AuthorityState:
        """Build terminal evidence from the exact current gated state."""

        if direction is PhysicalRootExchangePhase.ACCEPTED:
            graph = state
            marker = receipt.accepted_marker
        elif direction is PhysicalRootExchangePhase.ROLLED_BACK:
            graph = journal.base
            marker = receipt.rolled_back_marker
        else:  # pragma: no cover - internal exact-enum contract
            raise RuntimeError("physical-root terminal direction is invalid")
        return _state_with_graph(
            graph,
            gate=None,
            phase=_GatePhase.BASE,
            closed=state.closed,
            terminal_evidence=_terminal_evidence_with(state, marker),
        )

    def _compact_accepted(
        self,
        operation: _ExchangeOperation,
        journal: _ExchangeJournal,
    ) -> _TerminalRecord:
        receipt = self._require_receipt(journal)
        return self._install_terminal(
            operation,
            journal,
            _TerminalRecord(
                PhysicalRootExchangePhase.ACCEPTED,
                receipt.leases,
                receipt.projected_bytes,
            ),
        )

    def _compact_rolled_back(
        self,
        operation: _ExchangeOperation,
        journal: _ExchangeJournal,
    ) -> _TerminalRecord:
        return self._install_terminal(
            operation,
            journal,
            _TerminalRecord(PhysicalRootExchangePhase.ROLLED_BACK, None, None),
        )

    def _cas_transition(
        self,
        expected: _AuthorityState,
        replacement: _AuthorityState,
        *,
        terminal_marker: _TerminalMarker | None = None,
    ) -> bool:
        """Run one faultable CAS and classify the exact observed state."""

        try:
            swapped = self._authority._swap_state(expected, replacement)
        except BaseException as error:
            state = self._authority._snapshot_state()
            if (
                state is expected
                or state is replacement
                or (
                    terminal_marker is not None
                    and _has_terminal_marker(state, terminal_marker)
                )
            ):
                raise
            raise RuntimeError(
                "physical-root exchange authority state drifted"
            ) from error
        if swapped:
            return True
        state = self._authority._snapshot_state()
        if (
            state is replacement
            or (
                terminal_marker is not None
                and _has_terminal_marker(state, terminal_marker)
            )
        ):
            return True
        if state is expected:
            return False
        raise RuntimeError("physical-root exchange authority state drifted")

    @property
    def phase(self) -> PhysicalRootExchangePhase:
        """Return the journal phase using only exact state identities."""

        operation = self._begin_operation()
        try:
            journal = operation.journal
            if isinstance(journal, _TerminalRecord):
                return journal.phase
            receipt = journal.receipt
            if receipt is None:
                return PhysicalRootExchangePhase.OPEN
            state = self._authority._snapshot_state()
            self._check_operation(operation, journal)
            recognized = self._recognized_terminal(state, receipt)
            if recognized is not None:
                return recognized
            if state is journal.base:
                return PhysicalRootExchangePhase.PREPARED
            if state is receipt.staged:
                return PhysicalRootExchangePhase.STAGED
            if state is receipt.pending:
                return PhysicalRootExchangePhase.COMMIT_PENDING
            return PhysicalRootExchangePhase.DRIFTED
        finally:
            self._finish_operation(operation)

    @property
    def prepared_leases(self) -> Mapping[Hashable, PhysicalRootLease]:
        """Return the immutable durable incoming-lease receipt."""

        operation = self._begin_operation()
        try:
            journal = operation.journal
            if isinstance(journal, _TerminalRecord):
                if journal.phase is not PhysicalRootExchangePhase.ACCEPTED:
                    raise self._terminal_error(journal)
                assert journal.leases is not None
                return journal.leases
            receipt = self._require_receipt(journal)
            state = self._authority._snapshot_state()
            self._check_operation(operation, journal)
            recognized = self._recognized_terminal(state, receipt)
            if recognized is PhysicalRootExchangePhase.ACCEPTED:
                record = self._compact_accepted(operation, journal)
                assert record.leases is not None
                return record.leases
            if recognized is PhysicalRootExchangePhase.ROLLED_BACK:
                self._compact_rolled_back(operation, journal)
                raise RuntimeError("physical-root exchange was rolled back")
            return receipt.leases
        finally:
            self._finish_operation(operation)

    @property
    def final_projected_bytes(self) -> int:
        """Return the admitted final committed unique-root projection."""

        operation = self._begin_operation()
        try:
            journal = operation.journal
            if isinstance(journal, _TerminalRecord):
                if journal.phase is not PhysicalRootExchangePhase.ACCEPTED:
                    raise self._terminal_error(journal)
                assert journal.projected_bytes is not None
                return journal.projected_bytes
            receipt = self._require_receipt(journal)
            state = self._authority._snapshot_state()
            self._check_operation(operation, journal)
            recognized = self._recognized_terminal(state, receipt)
            if recognized is PhysicalRootExchangePhase.ACCEPTED:
                record = self._compact_accepted(operation, journal)
                assert record.projected_bytes is not None
                return record.projected_bytes
            if recognized is PhysicalRootExchangePhase.ROLLED_BACK:
                self._compact_rolled_back(operation, journal)
                raise RuntimeError("physical-root exchange was rolled back")
            return receipt.projected_bytes
        finally:
            self._finish_operation(operation)

    def claim(self, nbytes: int) -> object:
        operation = self._begin_operation()
        try:
            journal = self._require_open(operation.journal)
            self._check_open_base(operation, journal)
            if type(nbytes) is not int or nbytes < 0:
                raise TypeError(
                    "physical-root capacity must be a nonnegative integer"
                )
            token = object()
            claims = dict(journal.claims)
            claims[token] = nbytes
            roots = dict(journal.roots)
            self._check_open_base(operation, journal)
            _validate_projection(
                journal.base,
                journal.victim_tokens,
                claims,
                roots,
                self._authority.limit_bytes,
            )
            self._check_open_base(operation, journal)
            replacement = self._advanced(
                journal, claims=MappingProxyType(claims),
            )
            self._install_open_journal(operation, journal, replacement)
            return token
        finally:
            self._finish_operation(operation)

    def bind(
        self, token: object, value: object, semantic: Hashable,
    ) -> PhysicalRootFact:
        operation = self._begin_operation()
        try:
            journal = self._require_open(operation.journal)
            self._check_open_base(operation, journal)
            owned_token: object | None = None
            expected: int | None = None
            for candidate, nbytes in journal.claims.items():
                if candidate is token:
                    owned_token = candidate
                    expected = nbytes
                    break
            if owned_token is None or expected is None:
                raise ValueError("physical-root capacity token is not owned")
            claims = dict(journal.claims)
            _validate_semantic(semantic)
            self._check_open_base(operation, journal)
            roots = dict(journal.roots)
            self._check_open_base(operation, journal)
            duplicated = semantic in roots
            self._check_open_base(operation, journal)
            if duplicated:
                raise ValueError("physical-root semantic is duplicated")
            fact = physical_root_fact(value)
            self._check_open_base(operation, journal)
            if fact.nbytes != expected:
                raise ValueError(
                    "allocated physical root differs from reserved bytes"
                )
            del claims[owned_token]
            roots[semantic] = fact
            self._check_open_base(operation, journal)
            _validate_projection(
                journal.base,
                journal.victim_tokens,
                claims,
                roots,
                self._authority.limit_bytes,
            )
            self._check_open_base(operation, journal)
            replacement = self._advanced(
                journal,
                claims=MappingProxyType(claims),
                roots=MappingProxyType(roots),
            )
            self._install_open_journal(operation, journal, replacement)
            return fact
        finally:
            self._finish_operation(operation)

    def reserve(
        self, value: object, semantic: Hashable,
    ) -> PhysicalRootFact:
        operation = self._begin_operation()
        try:
            journal = self._require_open(operation.journal)
            self._check_open_base(operation, journal)
            _validate_semantic(semantic)
            self._check_open_base(operation, journal)
            roots = dict(journal.roots)
            self._check_open_base(operation, journal)
            duplicated = semantic in roots
            self._check_open_base(operation, journal)
            if duplicated:
                raise ValueError("physical-root semantic is duplicated")
            fact = physical_root_fact(value)
            self._check_open_base(operation, journal)
            roots[semantic] = fact
            self._check_open_base(operation, journal)
            _validate_projection(
                journal.base,
                journal.victim_tokens,
                dict(journal.claims),
                roots,
                self._authority.limit_bytes,
            )
            self._check_open_base(operation, journal)
            replacement = self._advanced(
                journal, roots=MappingProxyType(roots),
            )
            self._install_open_journal(operation, journal, replacement)
            return fact
        finally:
            self._finish_operation(operation)

    @property
    def projected_bytes(self) -> int:
        operation = self._begin_operation()
        try:
            journal = self._require_open(operation.journal)
            self._check_open_base(operation, journal)
            claims = dict(journal.claims)
            self._check_open_base(operation, journal)
            roots = dict(journal.roots)
            self._check_open_base(operation, journal)
            projected = _validate_projection(
                journal.base,
                journal.victim_tokens,
                claims,
                roots,
                self._authority.limit_bytes,
            )
            self._check_open_base(operation, journal)
            return projected
        finally:
            self._finish_operation(operation)

    def prepare(self) -> Mapping[Hashable, PhysicalRootLease]:
        """Install STAGED and return its precreated incoming leases."""

        operation = self._begin_operation()
        try:
            current = operation.journal
            if isinstance(current, _TerminalRecord):
                raise self._terminal_error(current)
            journal = current
            if journal.terminal_intent is PhysicalRootExchangePhase.ROLLED_BACK:
                raise RuntimeError("physical-root exchange rollback is pending")
            receipt = journal.receipt
            if receipt is None:
                if journal.claims:
                    raise RuntimeError("unbound physical-root capacity remains")
                self._authority._validate_exchange_base(
                    journal.base, journal.victims,
                )
                roots = dict(journal.roots)
                self._check_open_base(operation, journal)
                projected_bytes = _validate_projection(
                    journal.base,
                    journal.victim_tokens,
                    {},
                    roots,
                    self._authority.limit_bytes,
                )
                self._check_open_base(operation, journal)
                incoming = tuple(
                    (
                        semantic,
                        fact,
                        PhysicalRootLease(self._authority, semantic),
                    )
                    for semantic, fact in journal.roots.items()
                )
                self._check_open_base(operation, journal)
                pending = _replacement_state(
                    journal.base,
                    journal.victim_tokens,
                    incoming,
                    projected_bytes,
                    gate=journal.operation,
                    phase=_GatePhase.COMMIT_PENDING,
                )
                accepted_marker = _TerminalMarker(
                    PhysicalRootExchangePhase.ACCEPTED,
                )
                rolled_back_marker = _TerminalMarker(
                    PhysicalRootExchangePhase.ROLLED_BACK,
                )
                staged = _state_with_graph(
                    journal.base,
                    gate=journal.operation,
                    phase=_GatePhase.STAGED,
                )
                lease_dict = {
                    semantic: lease
                    for semantic, _fact, lease in incoming
                }
                self._check_open_base(operation, journal)
                receipt = _ExchangeReceipt(
                    MappingProxyType(lease_dict),
                    projected_bytes,
                    staged,
                    pending,
                    accepted_marker,
                    rolled_back_marker,
                )
                replacement = self._advanced(
                    journal, receipt=receipt, replace_receipt=True,
                )
                self._install_open_journal(
                    operation, journal, replacement,
                )
                journal = replacement
            state = self._authority._snapshot_state()
            self._check_operation(operation, journal)
            recognized = self._recognized_terminal(state, receipt)
            if recognized is PhysicalRootExchangePhase.ACCEPTED:
                self._compact_accepted(operation, journal)
                raise RuntimeError("physical-root exchange was accepted")
            if recognized is PhysicalRootExchangePhase.ROLLED_BACK:
                self._compact_rolled_back(operation, journal)
                raise RuntimeError("physical-root exchange was rolled back")
            if state is receipt.staged or state is receipt.pending:
                return receipt.leases
            if state is journal.base:
                if not self._cas_transition(journal.base, receipt.staged):
                    raise RuntimeError(
                        "physical-root exchange transition did not complete"
                    )
                return receipt.leases
            raise RuntimeError("physical-root exchange authority state drifted")
        finally:
            self._finish_operation(operation)

    def commit(self) -> Mapping[Hashable, PhysicalRootLease]:
        operation = self._begin_operation()
        try:
            current = operation.journal
            if isinstance(current, _TerminalRecord):
                if current.phase is PhysicalRootExchangePhase.ACCEPTED:
                    assert current.leases is not None
                    return current.leases
                raise self._terminal_error(current)
            journal = current
            if journal.terminal_intent is PhysicalRootExchangePhase.ROLLED_BACK:
                raise RuntimeError("physical-root exchange rollback is pending")
            receipt = self._require_receipt(journal)
            state = self._authority._snapshot_state()
            self._check_operation(operation, journal)
            recognized = self._recognized_terminal(state, receipt)
            if recognized is PhysicalRootExchangePhase.ACCEPTED:
                record = self._compact_accepted(operation, journal)
                assert record.leases is not None
                return record.leases
            if recognized is PhysicalRootExchangePhase.ROLLED_BACK:
                self._compact_rolled_back(operation, journal)
                raise RuntimeError("physical-root exchange was rolled back")
            if state is receipt.staged:
                if not self._cas_transition(receipt.staged, receipt.pending):
                    raise RuntimeError(
                        "physical-root exchange transition did not complete"
                    )
            elif state is not receipt.pending:
                raise RuntimeError(
                    "physical-root exchange authority state drifted"
                )
            return receipt.leases
        finally:
            self._finish_operation(operation)

    def accept(self) -> Mapping[Hashable, PhysicalRootLease]:
        operation = self._begin_operation(
            PhysicalRootExchangePhase.ACCEPTED,
            install_terminal_intent=False,
        )
        try:
            current = operation.journal
            if isinstance(current, _TerminalRecord):
                if current.phase is not PhysicalRootExchangePhase.ACCEPTED:
                    raise self._terminal_error(current)
                assert current.leases is not None
                return current.leases
            journal = current
            receipt = self._require_receipt(journal)
            state = self._authority._snapshot_state()
            self._check_operation(operation, journal)
            recognized = self._recognized_terminal(state, receipt)
            if recognized is PhysicalRootExchangePhase.ACCEPTED:
                record = self._compact_accepted(operation, journal)
                assert record.leases is not None
                return record.leases
            if recognized is PhysicalRootExchangePhase.ROLLED_BACK:
                self._compact_rolled_back(operation, journal)
                raise RuntimeError("physical-root exchange was rolled back")
            if journal.terminal_intent is PhysicalRootExchangePhase.ROLLED_BACK:
                raise RuntimeError("physical-root exchange rollback is pending")
            if state is receipt.staged:
                raise RuntimeError("physical-root exchange is not committed")
            if state is not receipt.pending:
                raise RuntimeError(
                    "physical-root exchange authority state drifted"
                )
            journal = self._install_terminal_intent(
                operation,
                journal,
                PhysicalRootExchangePhase.ACCEPTED,
            )
            receipt = self._require_receipt(journal)
            state = self._authority._snapshot_state()
            self._check_operation(operation, journal)
            if state is not receipt.pending:
                recognized = self._recognized_terminal(state, receipt)
                if recognized is PhysicalRootExchangePhase.ACCEPTED:
                    record = self._compact_accepted(operation, journal)
                    assert record.leases is not None
                    return record.leases
                if recognized is PhysicalRootExchangePhase.ROLLED_BACK:
                    self._compact_rolled_back(operation, journal)
                    raise RuntimeError("physical-root exchange was rolled back")
                raise RuntimeError(
                    "physical-root exchange authority state drifted"
                )
            accepted = self._terminal_replacement(
                journal,
                receipt,
                state,
                PhysicalRootExchangePhase.ACCEPTED,
            )
            if not self._cas_transition(
                receipt.pending,
                accepted,
                terminal_marker=receipt.accepted_marker,
            ):
                raise RuntimeError(
                    "physical-root exchange transition did not complete"
                )
            record = self._compact_accepted(operation, journal)
            assert record.leases is not None
            return record.leases
        finally:
            self._finish_operation(operation)

    def rollback(self) -> None:
        operation = self._begin_operation(PhysicalRootExchangePhase.ROLLED_BACK)
        try:
            current = operation.journal
            if isinstance(current, _TerminalRecord):
                if current.phase is PhysicalRootExchangePhase.ROLLED_BACK:
                    return
                raise self._terminal_error(current)
            journal = current
            receipt = journal.receipt
            if receipt is None:
                self._compact_rolled_back(operation, journal)
                return
            state = self._authority._snapshot_state()
            self._check_operation(operation, journal)
            recognized = self._recognized_terminal(state, receipt)
            if recognized is PhysicalRootExchangePhase.ACCEPTED:
                self._compact_accepted(operation, journal)
                raise RuntimeError("physical-root exchange was accepted")
            if recognized is PhysicalRootExchangePhase.ROLLED_BACK:
                self._compact_rolled_back(operation, journal)
                return
            if state is journal.base or state.gate is not journal.operation:
                # PREPARED never installed an owned gate. Legal ungated drift
                # cannot prevent local staged-root retirement.
                self._compact_rolled_back(operation, journal)
                return
            if state is not receipt.staged and state is not receipt.pending:
                raise RuntimeError(
                    "physical-root exchange authority state drifted"
                )
            rolled_back = self._terminal_replacement(
                journal,
                receipt,
                state,
                PhysicalRootExchangePhase.ROLLED_BACK,
            )
            if not self._cas_transition(
                state,
                rolled_back,
                terminal_marker=receipt.rolled_back_marker,
            ):
                raise RuntimeError(
                    "physical-root exchange transition did not complete"
                )
            self._compact_rolled_back(operation, journal)
        finally:
            self._finish_operation(operation)


class PhysicalRootAuthority:
    """Bounded, reference-counted authority over unique physical roots."""

    def __init__(self, limit_bytes: int) -> None:
        if type(limit_bytes) is not int or limit_bytes < 0:
            raise TypeError("physical-root limit must be a nonnegative integer")
        self._limit = limit_bytes
        self._state = _indexed_state(
            (), (), None, _GatePhase.BASE, False, (),
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
        if _is_exchange_gate(state):
            raise RuntimeError("physical-root authority accounting is busy")
        return state

    def _snapshot_state(self) -> _AuthorityState:
        with self._lock:
            return self._state

    def _swap_state(
        self, expected: _AuthorityState, replacement: _AuthorityState,
    ) -> bool:
        """Exact identity CAS containing no user callbacks."""

        with self._lock:
            if self._state is not expected:
                return False
            self._state = replacement
            return True

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

    def exchange(
        self, victims: tuple[PhysicalRootLease, ...],
    ) -> PhysicalRootExchange:
        """Create an ungated exact replacement operation for ``victims``."""

        if type(victims) is not tuple:
            raise TypeError("physical-root exchange victims must be an exact tuple")
        while True:
            with self._lock:
                state = self._state
                if state.closed:
                    raise RuntimeError("physical-root authority is closed")
                if state.gate is not None:
                    raise RuntimeError("physical-root authority is busy")
            token_ids: set[int] = set()
            for lease in victims:
                if type(lease) is not PhysicalRootLease:
                    raise TypeError("physical-root exchange victims must be leases")
                if lease._authority is not self:
                    raise ValueError("physical-root exchange victim is foreign")
                if lease._released:
                    raise ValueError("physical-root exchange victim is released")
                token_id = id(lease._token)
                if token_id in token_ids:
                    raise ValueError("physical-root exchange victim is duplicated")
                token_ids.add(token_id)
                binding = _binding_for_token(state, lease._token)
                if binding is None or binding.semantic is not lease._semantic:
                    raise ValueError("physical-root exchange victim is stale")
            with self._lock:
                if self._state is state:
                    return PhysicalRootExchange(self, state, victims)

    def _validate_exchange_base(
        self,
        base: _AuthorityState,
        victims: tuple[PhysicalRootLease, ...],
    ) -> None:
        with self._lock:
            state = self._state
            if state.closed:
                raise RuntimeError("physical-root authority is closed")
            if _is_exchange_gate(state):
                raise RuntimeError("physical-root authority is busy")
            if state.gate is not None:
                raise RuntimeError("physical-root authority is busy")
            if state is not base:
                raise RuntimeError("physical-root exchange base state drifted")
        for lease in victims:
            if lease._released:
                raise ValueError("physical-root exchange victim is released")
            binding = _binding_for_token(base, lease._token)
            if binding is None or binding.semantic is not lease._semantic:
                raise ValueError("physical-root exchange victim is stale")
        with self._lock:
            if self._state is not base:
                raise RuntimeError("physical-root exchange base state drifted")

    def _cancel_reservation(self, operation: object | None = None) -> None:
        with self._lock:
            state = self._state
            if _is_exchange_gate(state):
                raise RuntimeError("physical-root authority is busy")
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
                if _is_exchange_gate(state):
                    raise RuntimeError("physical-root authority is busy")
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
                if _is_exchange_gate(state):
                    raise RuntimeError("physical-root authority is busy")
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
                # Busy is checked before token liveness.
                if _is_exchange_gate(state):
                    raise RuntimeError("physical-root authority is busy")
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
                state.terminal_evidence,
            )
            with self._lock:
                if self._state is state:
                    self._state = replacement
                    return

    def close(self) -> None:
        old_state: _AuthorityState
        with self._lock:
            old_state = self._state
            if _is_exchange_gate(old_state):
                raise RuntimeError("physical-root authority is busy")
            if old_state.closed:
                return
            self._state = _indexed_state(
                (), (), None, _GatePhase.BASE, True,
                old_state.terminal_evidence,
            )
        # Keep the losing state strong through lock exit so arbitrary root or
        # semantic destructors cannot run in the pointer-swap critical section.
        del old_state


__all__ = [
    "PhysicalRootAuthority",
    "PhysicalRootExchange",
    "PhysicalRootExchangePhase",
    "PhysicalRootFact",
    "PhysicalRootLease",
    "PhysicalRootReservation",
    "physical_root_fact",
]
