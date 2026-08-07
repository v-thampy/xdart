# -*- coding: utf-8 -*-
"""Qt-free dynamic identity overlay borrowing the one H10 stage ledger.

Accepted attempts, outcomes, and top-level writes are recorded immediately in
the borrowed ledger and are never rolled back.  Only target-qualified
persisted/durable receipts and publication drops wait behind the current H23
lineage epoch.  The exact bound live-session owner is the only object allowed
to promote or discard that staged target truth.
"""
from __future__ import annotations

import threading
import sys
import weakref
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Hashable, Iterable, Mapping

from ._closed_values import freeze_identity
from .stage_accounting import ItemDisposition, ResultMode, StageReceipt


_LEDGER_OVERLAY_LOCK = threading.Lock()
_LEDGER_OVERLAYS = weakref.WeakKeyDictionary()

__all__ = [
    "DynamicAccountingLimits",
    "DynamicAttemptState",
    "DynamicAttemptToken",
    "DynamicCleanupReceipt",
    "DynamicFrameIdentity",
    "DynamicGroupHighWater",
    "DynamicRunAccounting",
    "DynamicRunSnapshot",
    "DynamicRunState",
]


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True, slots=True)
class DynamicAccountingLimits:
    max_groups: int
    max_attempts_per_frame: int
    max_outstanding: int

    def __post_init__(self) -> None:
        for name in ("max_groups", "max_attempts_per_frame", "max_outstanding"):
            object.__setattr__(self, name, _positive_int(name, getattr(self, name)))


@dataclass(frozen=True, slots=True)
class DynamicFrameIdentity:
    source_identity: Hashable
    logical_frame_identity: Hashable

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_identity",
            freeze_identity(self.source_identity, "dynamic source identity"),
        )
        object.__setattr__(
            self,
            "logical_frame_identity",
            freeze_identity(
                self.logical_frame_identity, "dynamic logical frame identity",
            ),
        )


@dataclass(frozen=True, eq=False, slots=True)
class DynamicAttemptToken:
    """Immutable exact identity for one provisional source attempt."""

    key: DynamicFrameIdentity
    run_generation: int
    revision: int
    source_revision: int


class DynamicAttemptState(str, Enum):
    PROVISIONAL = "provisional"
    ENQUEUED = "enqueued"
    ACCEPTED = "accepted"
    COMPLETED = "completed"
    FAILED_RETRYABLE = "failed-retryable"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DynamicRunState(str, Enum):
    ACTIVE = "active"
    STOPPED = "stopped"
    CLEANUP_PENDING = "cleanup-pending"
    FINISHED = "finished"
    ABORTED = "aborted"


@dataclass(frozen=True, slots=True)
class DynamicGroupHighWater:
    accepted: int = -1
    completed: int = -1
    written: int = -1
    persisted: int = -1
    durable: int = -1


@dataclass(frozen=True, slots=True)
class DynamicCleanupReceipt:
    terminal_state: DynamicRunState
    light_cleanup_receipt: object | None
    retry_token: object | None
    complete: bool


@dataclass(frozen=True, slots=True)
class DynamicRunSnapshot:
    run_generation: int
    state: DynamicRunState
    discovered: frozenset[DynamicFrameIdentity]
    enqueued: frozenset[DynamicFrameIdentity]
    accepted: frozenset[DynamicFrameIdentity]
    completed: frozenset[DynamicFrameIdentity]
    written: frozenset[DynamicFrameIdentity]
    persisted: frozenset[tuple[DynamicFrameIdentity, ResultMode, str]]
    durable: frozenset[tuple[DynamicFrameIdentity, ResultMode, str]]
    publication_dropped: frozenset[tuple[DynamicFrameIdentity, ResultMode]]
    pending_persisted: frozenset[tuple[DynamicFrameIdentity, ResultMode, str]]
    pending_durable: frozenset[tuple[DynamicFrameIdentity, ResultMode, str]]
    pending_publication_dropped: frozenset[tuple[DynamicFrameIdentity, ResultMode]]
    in_flight: frozenset[DynamicFrameIdentity]
    retry_owned: frozenset[DynamicFrameIdentity]
    attempts: Mapping[DynamicFrameIdentity, tuple[DynamicAttemptToken, ...]]
    attempt_states: Mapping[DynamicAttemptToken, DynamicAttemptState]
    ledger_attempts: Mapping[DynamicAttemptToken, int]
    aborted_epoch_attempts: frozenset[DynamicAttemptToken]
    errors: Mapping[DynamicAttemptToken, str]
    completed_attempts: Mapping[DynamicFrameIdentity, DynamicAttemptToken]
    written_attempts: Mapping[tuple[DynamicFrameIdentity, ResultMode], DynamicAttemptToken]
    persisted_attempts: Mapping[
        tuple[DynamicFrameIdentity, ResultMode, str], DynamicAttemptToken
    ]
    durable_attempts: Mapping[
        tuple[DynamicFrameIdentity, ResultMode, str], DynamicAttemptToken
    ]
    publication_dropped_attempts: Mapping[
        tuple[DynamicFrameIdentity, ResultMode], DynamicAttemptToken
    ]
    high_water: Mapping[Hashable, DynamicGroupHighWater]
    epoch_revision: int
    cleanup_receipt: DynamicCleanupReceipt | None
    owner_count: int


@dataclass(frozen=True, slots=True)
class _FrameRegistration:
    group: Hashable
    ordinal: int
    output_label: int


@dataclass(slots=True)
class _AttemptRecord:
    token: DynamicAttemptToken
    state: DynamicAttemptState = DynamicAttemptState.PROVISIONAL
    error: str | None = None
    retryable: bool = False
    enqueued: bool = False
    accepted: bool = False
    ledger_attempt: int | None = None
    completed: bool = False
    produced: tuple[ResultMode, ...] = ()
    result_revisions: dict[ResultMode, int] = field(default_factory=dict)
    written: set[ResultMode] = field(default_factory=set)
    epoch_finalized: bool = False
    epoch_aborted: bool = False


@dataclass(frozen=True, eq=False, slots=True)
class _EpochSeal:
    kind: str
    ordinal: int
    epoch_revision: int
    terminal_state: DynamicRunState | None = None


class _DynamicWriterBoundary:
    """The only H23 receipt and epoch-owner surface."""

    __slots__ = ("_accounting",)
    defer_epoch_durability = True

    def __init__(self, accounting: "DynamicRunAccounting") -> None:
        self._accounting = accounting

    def __copy__(self):
        raise TypeError("dynamic writer boundary is a unique deferred authority")

    def __deepcopy__(self, _memo):
        raise TypeError("dynamic writer boundary is a unique deferred authority")

    @property
    def accounting(self) -> "DynamicRunAccounting":
        return self._accounting

    def targets_for(self, mode):
        return self._accounting.ledger.targets_by_mode.get(mode, frozenset())

    def capture_receipt(self, label, mode, target) -> StageReceipt:
        return self._accounting._capture_receipt(label, mode, target)

    def commit_durable(self, receipts: Iterable[StageReceipt]) -> None:
        self._accounting._stage_durable(receipts)

    def commit_publication_drop(self, label, mode, expected_revision) -> None:
        self._accounting._stage_publication_drop(label, mode, expected_revision)

    def bind_live_session(self, session, owner_token) -> None:
        self._accounting._bind_live_session(session, owner_token)

    def prepare_epoch_commit(self, session, owner_token):
        return self._accounting._prepare_bound_epoch(
            session, owner_token, kind="epoch",
        )

    def prepare_session_finish(self, session, owner_token, *, stopped=False):
        return self._accounting._prepare_bound_epoch(
            session,
            owner_token,
            kind="finish",
            terminal_state=(
                DynamicRunState.STOPPED if stopped else DynamicRunState.FINISHED
            ),
        )

    def epoch_prepare_failed(self, session, owner_token, seal) -> None:
        self._accounting._unseal_bound_epoch(session, owner_token, seal)

    def epoch_committed(self, session, owner_token, seal, anchor) -> None:
        self._accounting._commit_bound_epoch(session, owner_token, seal, anchor)

    def session_finished(self, session, owner_token, seal, identity) -> None:
        self._accounting._finish_bound(session, owner_token, seal, identity)

    def session_stopped(self, session, owner_token, seal, reason) -> None:
        self._accounting._stop_bound(session, owner_token, seal, str(reason))

    def epoch_aborted(self, session, owner_token, reason) -> None:
        self._accounting._abort_bound(session, owner_token, str(reason))


class DynamicRunAccounting:
    """Dynamic run state borrowing an exact existing ledger object."""

    def __init__(self, ledger, *, run_generation: int,
                 limits: DynamicAccountingLimits) -> None:
        required = (
            "record_discovered", "record_enqueued", "record_accepted",
            "record_outcome", "record_written", "record_persisted",
            "record_durable", "record_publication_dropped", "receipt",
            "current_revision", "snapshot", "required_modes", "targets_by_mode",
        )
        owner_module = sys.modules.get("xrd_tools.session.stage_accounting")
        owner_type = getattr(owner_module, "StageLedger", None)
        if owner_type is None or type(ledger) is not owner_type:
            raise TypeError("dynamic accounting requires the existing H10 StageLedger")
        if any(not hasattr(ledger, name) for name in required):
            raise TypeError("dynamic accounting requires the existing H10 stage ledger")
        if isinstance(run_generation, bool) or int(run_generation) != run_generation:
            raise ValueError("run_generation must be an integer")
        if type(limits) is not DynamicAccountingLimits:
            raise TypeError("limits must be DynamicAccountingLimits")
        self._ledger = ledger
        self._generation = int(run_generation)
        self._limits = limits
        self._lock = threading.RLock()
        self._state = DynamicRunState.ACTIVE
        self._frames: dict[DynamicFrameIdentity, _FrameRegistration] = {}
        self._logical_enqueued: set[DynamicFrameIdentity] = set()
        self._ordinals: dict[Hashable, dict[int, DynamicFrameIdentity]] = {}
        self._labels: dict[int, DynamicFrameIdentity] = {}
        self._attempts: dict[DynamicFrameIdentity, list[DynamicAttemptToken]] = {}
        self._records: dict[DynamicAttemptToken, _AttemptRecord] = {}
        self._latest_source_revision: dict[DynamicFrameIdentity, int] = {}
        # This maps the ledger's exact result revision to the dynamic attempt;
        # it owns no counter and mints no receipt.
        self._receipt_suppliers: dict[
            tuple[int, ResultMode, int], DynamicAttemptToken
        ] = {}
        self._epoch_tokens: list[DynamicAttemptToken] = []
        self._pending_persisted: dict[
            tuple[DynamicFrameIdentity, ResultMode, str],
            tuple[DynamicAttemptToken, StageReceipt],
        ] = {}
        self._pending_durable: dict[
            tuple[DynamicFrameIdentity, ResultMode, str],
            tuple[DynamicAttemptToken, StageReceipt],
        ] = {}
        self._pending_dropped: dict[
            tuple[DynamicFrameIdentity, ResultMode],
            tuple[DynamicAttemptToken, int],
        ] = {}
        self._persisted: dict[
            tuple[DynamicFrameIdentity, ResultMode, str], DynamicAttemptToken
        ] = {}
        self._durable: dict[
            tuple[DynamicFrameIdentity, ResultMode, str], DynamicAttemptToken
        ] = {}
        self._dropped: dict[
            tuple[DynamicFrameIdentity, ResultMode], DynamicAttemptToken
        ] = {}
        self._epoch_revision = 0
        self._last_epoch_identity = None
        self._seal_ordinal = 0
        self._active_seal: _EpochSeal | None = None
        self._sealed_promoted = False
        self._writer_boundary = _DynamicWriterBoundary(self)
        self._live_session_ref: weakref.ReferenceType | None = None
        self._live_owner_token = None
        self._light_lease = None
        self._light_cleanup_hooks = None
        self._light_retry_token = None
        self._cleanup_receipt: DynamicCleanupReceipt | None = None
        self._terminal_intent: DynamicRunState | None = None
        with _LEDGER_OVERLAY_LOCK:
            if ledger in _LEDGER_OVERLAYS:
                raise RuntimeError(
                    "StageLedger already has a dynamic accounting owner"
                )
            # The weak key keeps this marker for exactly the ledger lifetime,
            # even if a caller drops the overlay while retaining mutated ledger
            # revisions.  The value owns neither object.
            _LEDGER_OVERLAYS[ledger] = True

    def __copy__(self):
        raise TypeError("DynamicRunAccounting is a unique run owner")

    def __deepcopy__(self, _memo):
        raise TypeError("DynamicRunAccounting is a unique run owner")

    @property
    def ledger(self):
        return self._ledger

    @property
    def writer_boundary(self) -> _DynamicWriterBoundary:
        return self._writer_boundary

    def _require_open_frontier(self) -> None:
        if self._state is not DynamicRunState.ACTIVE:
            raise RuntimeError("dynamic accepted frontier is frozen")

    def _require_preterminal(self) -> None:
        if self._terminal_intent is not None:
            raise RuntimeError("dynamic run terminal transition has started")

    def _require_unsealed(self) -> None:
        self._require_preterminal()
        if self._active_seal is not None:
            raise RuntimeError("dynamic lineage epoch is sealed for H23 commit")

    def discover(self, key: DynamicFrameIdentity, *, group: Hashable,
                 ordinal: int, output_label: int) -> DynamicFrameIdentity:
        if type(key) is not DynamicFrameIdentity:
            raise TypeError("key must be DynamicFrameIdentity")
        group = freeze_identity(group, "dynamic group identity")
        if isinstance(ordinal, bool) or int(ordinal) != ordinal or int(ordinal) < 0:
            raise ValueError("group ordinal must be a non-negative integer")
        if isinstance(output_label, bool) or int(output_label) != output_label:
            raise ValueError("output label must be an integer")
        ordinal, output_label = int(ordinal), int(output_label)
        with self._lock:
            self._require_unsealed()
            self._require_open_frontier()
            if key in self._frames:
                prior = self._frames[key]
                if prior == _FrameRegistration(group, ordinal, output_label):
                    raise ValueError(f"duplicate discovery for {key!r}")
                raise ValueError(f"dynamic discovery remap for {key!r}: {prior!r}")
            if group not in self._ordinals and len(self._ordinals) >= self._limits.max_groups:
                raise ValueError("dynamic group limit exceeded")
            if ordinal in self._ordinals.get(group, {}):
                raise ValueError(f"group ordinal {ordinal} is already owned")
            if output_label in self._labels:
                raise ValueError(f"output label {output_label} is already mapped")
            # Publish discovery to the borrowed ledger while this frontier is
            # still locked.  A concurrent begin_attempt must never observe a
            # dynamic registration before its one H10 discovery exists.
            self._ledger.record_discovered(key)
            self._frames[key] = _FrameRegistration(group, ordinal, output_label)
            self._ordinals.setdefault(group, {})[ordinal] = key
            self._labels[output_label] = key
            self._attempts[key] = []
        return key

    def _key_is_canonical(
        self,
        key: DynamicFrameIdentity,
        *,
        durable=None,
        dropped=None,
    ) -> bool:
        durable = self._durable if durable is None else durable
        dropped = self._dropped if dropped is None else dropped
        modes = tuple(self._ledger.required_modes)
        if not modes or any(not self._ledger.targets_by_mode.get(mode) for mode in modes):
            return False
        for mode in modes:
            current = self._current_result_supplier(key, mode)
            if current is None:
                return False
            if dropped.get((key, mode)) is current:
                continue
            if not self._mode_has_one_supplier(
                key, mode, durable, expected=current,
            ):
                return False
        return True

    def _mode_has_one_supplier(
        self, key, mode, registry, *, expected=None,
    ) -> bool:
        suppliers = tuple(
            registry.get((key, mode, target))
            for target in self._ledger.targets_by_mode[mode]
        )
        return bool(suppliers) and all(
            supplier is not None for supplier in suppliers
        ) and len(set(suppliers)) == 1 and (
            expected is None or suppliers[0] is expected
        )

    def _key_has_complete_targets(self, key, registry) -> bool:
        modes = tuple(self._ledger.required_modes)
        return bool(modes) and all(
            self._ledger.targets_by_mode.get(mode)
            and self._mode_has_one_supplier(key, mode, registry)
            for mode in modes
        )

    def _key_is_unresolved(
        self,
        key: DynamicFrameIdentity,
        *,
        durable=None,
        dropped=None,
    ) -> bool:
        tokens = self._attempts.get(key, ())
        if not tokens:
            return False
        record = self._records[tokens[-1]]
        if record.state in {
            DynamicAttemptState.PROVISIONAL,
            DynamicAttemptState.ENQUEUED,
            DynamicAttemptState.ACCEPTED,
            DynamicAttemptState.FAILED_RETRYABLE,
        }:
            return True
        if record.state is DynamicAttemptState.COMPLETED:
            return not self._key_is_canonical(
                key, durable=durable, dropped=dropped,
            )
        return False

    def _outstanding_keys(self) -> set[DynamicFrameIdentity]:
        return {key for key in self._attempts if self._key_is_unresolved(key)}

    def _prospective_outstanding_keys(self) -> set[DynamicFrameIdentity]:
        """Resolve only exact suppliers in the frozen pending H23 batch.

        This is validation evidence, not publication: the canonical registries
        and borrowed StageLedger remain untouched until H23 consumes the seal.
        """
        durable = dict(self._durable)
        dropped = dict(self._dropped)
        for key, (token, _receipt) in self._pending_durable.items():
            durable[key] = token
        for key, (token, _revision) in self._pending_dropped.items():
            dropped[key] = token
        return {
            key for key in self._attempts
            if self._key_is_unresolved(
                key, durable=durable, dropped=dropped,
            )
        }

    def begin_attempt(self, key: DynamicFrameIdentity, *, source_revision: int
                      ) -> DynamicAttemptToken:
        if isinstance(source_revision, bool) or int(source_revision) != source_revision:
            raise ValueError("source revision must be an integer")
        source_revision = int(source_revision)
        with self._lock:
            self._require_unsealed()
            self._require_open_frontier()
            if key not in self._frames:
                raise ValueError("attempt names a foreign dynamic frame identity")
            attempts = self._attempts[key]
            if len(attempts) >= self._limits.max_attempts_per_frame:
                raise ValueError("dynamic attempt limit exceeded")
            prior_record = self._records[attempts[-1]] if attempts else None
            if prior_record is not None and prior_record.state in {
                DynamicAttemptState.PROVISIONAL,
                DynamicAttemptState.ENQUEUED,
                DynamicAttemptState.ACCEPTED,
            }:
                raise ValueError(
                    "dynamic live attempt must be failed or cancelled before retry"
                )
            outstanding = self._outstanding_keys()
            if key not in outstanding and len(outstanding) >= self._limits.max_outstanding:
                raise ValueError("dynamic outstanding attempt limit exceeded")
            prior = self._latest_source_revision.get(key)
            if prior is not None and source_revision < prior:
                raise ValueError(
                    f"source revision regressed from {prior} to {source_revision}"
                )
            token = DynamicAttemptToken(
                key, self._generation, len(attempts) + 1, source_revision,
            )
            if (
                prior_record is not None
                and prior_record.state is DynamicAttemptState.FAILED_RETRYABLE
            ):
                prior_record.retryable = False
                prior_record.state = DynamicAttemptState.FAILED
            attempts.append(token)
            self._records[token] = _AttemptRecord(token)
            self._latest_source_revision[key] = source_revision
            return token

    def _record(self, token: DynamicAttemptToken) -> _AttemptRecord:
        if type(token) is not DynamicAttemptToken:
            raise TypeError("attempt must be DynamicAttemptToken")
        if token.run_generation != self._generation:
            raise ValueError("attempt belongs to another run generation")
        record = self._records.get(token)
        if record is None or record.token is not token:
            raise ValueError("foreign attempt token")
        return record

    def _require_latest_positive(self, token: DynamicAttemptToken) -> None:
        attempts = self._attempts[token.key]
        if not attempts or attempts[-1] is not token:
            raise ValueError(
                "superseded dynamic attempt cannot make a first positive transition"
            )

    def record_enqueued(self, token: DynamicAttemptToken) -> None:
        with self._lock:
            self._require_unsealed()
            record = self._record(token)
            if record.enqueued:
                return
            self._require_latest_positive(token)
            if record.state is not DynamicAttemptState.PROVISIONAL:
                raise ValueError("only a provisional attempt can be enqueued")
            # Source-axis admission is a logical-frame identity set.  A retry
            # remains visible in attempt_states without inflating discovery.
            self._ledger.record_enqueued(token.key)
            self._logical_enqueued.add(token.key)
            record.enqueued = True
            record.state = DynamicAttemptState.ENQUEUED

    def record_accepted(self, token: DynamicAttemptToken) -> int:
        with self._lock:
            self._require_unsealed()
            record = self._record(token)
            if record.accepted:
                return int(record.ledger_attempt)
            self._require_latest_positive(token)
            self._require_open_frontier()
            if record.state not in {
                DynamicAttemptState.PROVISIONAL,
                DynamicAttemptState.ENQUEUED,
            }:
                raise ValueError("only a live provisional attempt can be accepted")
            label = self._frames[token.key].output_label
            ledger_attempt = self._ledger.record_accepted(label)
            record.ledger_attempt = int(ledger_attempt)
            record.accepted = True
            record.state = DynamicAttemptState.ACCEPTED
            self._epoch_tokens.append(token)
            return int(ledger_attempt)

    def record_completed(self, token: DynamicAttemptToken, *,
                         produced: Iterable[ResultMode]) -> None:
        modes = tuple(dict.fromkeys(produced))
        with self._lock:
            self._require_unsealed()
            record = self._record(token)
            if record.completed:
                if record.produced == modes:
                    return
                raise ValueError("contradictory completed-attempt replay")
            self._require_latest_positive(token)
            if not record.accepted or record.state is not DynamicAttemptState.ACCEPTED:
                raise ValueError("completion requires the exact accepted attempt")
            if not modes or set(modes) - set(self._ledger.required_modes):
                raise ValueError("completion must name declared produced modes")
            label = self._frames[token.key].output_label
            self._ledger.record_outcome(
                label, ItemDisposition.COMPLETED, produced=modes,
                attempt=record.ledger_attempt,
            )
            for mode in modes:
                revision = self._ledger.current_revision(label, mode)
                record.result_revisions[mode] = revision
                self._receipt_suppliers[(label, mode, revision)] = token
            record.completed = True
            record.produced = modes
            record.state = DynamicAttemptState.COMPLETED

    def record_failed(self, token: DynamicAttemptToken, *, error: str,
                      retryable: bool) -> None:
        with self._lock:
            self._require_unsealed()
            record = self._record(token)
            if record.state in {
                DynamicAttemptState.FAILED, DynamicAttemptState.FAILED_RETRYABLE,
            }:
                if record.error == str(error) and record.retryable is bool(retryable):
                    return
                raise ValueError("contradictory failed-attempt replay")
            if record.completed or record.state is DynamicAttemptState.CANCELLED:
                raise ValueError("attempt is already terminal")
            if record.accepted:
                label = self._frames[token.key].output_label
                self._ledger.record_outcome(
                    label, ItemDisposition.FAILED, error=str(error),
                    attempt=record.ledger_attempt,
                )
            record.error = str(error)
            record.retryable = bool(retryable)
            record.state = (
                DynamicAttemptState.FAILED_RETRYABLE
                if retryable else DynamicAttemptState.FAILED
            )

    def record_cancelled(self, token: DynamicAttemptToken, *, reason: str) -> None:
        with self._lock:
            self._require_unsealed()
            record = self._record(token)
            if record.state is DynamicAttemptState.CANCELLED:
                if record.error == str(reason):
                    return
                raise ValueError("contradictory cancelled-attempt replay")
            if record.completed or record.state in {
                DynamicAttemptState.FAILED, DynamicAttemptState.FAILED_RETRYABLE,
            }:
                raise ValueError("attempt is already terminal")
            if record.accepted:
                label = self._frames[token.key].output_label
                self._ledger.record_outcome(
                    label, ItemDisposition.CANCELLED_BEFORE_COMPLETION,
                    error=str(reason), attempt=record.ledger_attempt,
                )
            record.error = str(reason)
            record.state = DynamicAttemptState.CANCELLED

    def record_written(self, token: DynamicAttemptToken, *,
                       modes: Iterable[ResultMode]) -> None:
        modes = tuple(dict.fromkeys(modes))
        with self._lock:
            self._require_unsealed()
            record = self._record(token)
            if not record.completed or not set(modes) <= set(record.produced):
                raise ValueError("written transition requires produced completed modes")
            fresh = tuple(mode for mode in modes if mode not in record.written)
            if not fresh:
                return
            label = self._frames[token.key].output_label
            stale = tuple(
                mode for mode in fresh
                if record.result_revisions.get(mode)
                != self._ledger.current_revision(label, mode)
            )
            if stale:
                raise ValueError(
                    "stale/out-of-order dynamic write cannot certify the current "
                    "result revision"
                )
            # StageLedger certifies the current revision, so only token-owned
            # fresh pairs may reach it.  The complete batch was checked first.
            self._ledger.record_written(label, fresh)
            record.written.update(fresh)

    def _capture_receipt(self, label, mode, target) -> StageReceipt:
        label, target = int(label), str(target)
        with self._lock:
            self._require_preterminal()
            if label not in self._labels:
                raise ValueError("writer receipt names a foreign output label")
            receipt = self._ledger.receipt(label, mode, target)
            if (label, mode, receipt.revision) not in self._receipt_suppliers:
                raise ValueError("writer receipt has no exact dynamic attempt supplier")
            return receipt

    def _validate_receipt(self, receipt: StageReceipt):
        if not isinstance(receipt, StageReceipt):
            raise TypeError("dynamic receipt must be StageReceipt")
        observed = self._ledger.receipt(receipt.label, receipt.mode, receipt.target)
        if observed != receipt:
            raise ValueError("stale dynamic writer receipt")
        token = self._receipt_suppliers.get(
            (receipt.label, receipt.mode, receipt.revision),
        )
        if token is None:
            raise ValueError("dynamic writer receipt has no exact attempt")
        if receipt.mode not in self._records[token].written:
            raise ValueError(
                "dynamic target truth requires the exact supplier mode to be written"
            )
        return token, (token.key, receipt.mode, receipt.target)

    @staticmethod
    def _strictly_newer(
        current: DynamicAttemptToken, prior: DynamicAttemptToken,
    ) -> bool:
        return (
            current.key == prior.key
            and current.run_generation == prior.run_generation
            and current.revision > prior.revision
        )

    def _pending_drop_replacement(
        self, token: DynamicAttemptToken, pair,
    ) -> bool:
        pending = self._pending_dropped.get(pair)
        if pending is None:
            return False
        prior, _revision = pending
        if prior is token:
            raise ValueError("dynamic result is publication-dropped")
        if not self._strictly_newer(token, prior):
            raise ValueError("stale dynamic receipt cannot replace a pending drop")
        return True

    def record_persisted(self, receipts: Iterable[StageReceipt]) -> None:
        with self._lock:
            self._require_preterminal()
        batch = tuple(receipts)
        with self._lock:
            self._require_preterminal()
            values = [self._validate_receipt(receipt) + (receipt,) for receipt in batch]
            fresh = []
            retire_drops = set()
            for token, key, receipt in values:
                if self._dropped.get(key[:2]) is token:
                    raise ValueError("dynamic result is publication-dropped")
                if self._pending_drop_replacement(token, key[:2]):
                    retire_drops.add(key[:2])
                if self._persisted.get(key) is token:
                    continue
                fresh.append((token, key, receipt))
            if self._active_seal is not None:
                if not retire_drops and all(
                    self._pending_persisted.get(key) == (token, receipt)
                    for token, key, receipt in fresh
                ):
                    return
                raise RuntimeError("dynamic lineage epoch is sealed for H23 commit")
            for pair in retire_drops:
                self._pending_dropped.pop(pair, None)
            for token, key, receipt in fresh:
                self._pending_persisted[key] = (token, receipt)

    def _stage_durable(self, receipts: Iterable[StageReceipt]) -> None:
        with self._lock:
            self._require_preterminal()
        batch = tuple(receipts)
        with self._lock:
            self._require_preterminal()
            values = [self._validate_receipt(receipt) + (receipt,) for receipt in batch]
            fresh = []
            retire_drops = set()
            for token, key, receipt in values:
                if self._dropped.get(key[:2]) is token:
                    raise ValueError("dynamic result is publication-dropped")
                if self._pending_drop_replacement(token, key[:2]):
                    retire_drops.add(key[:2])
                if self._durable.get(key) is token:
                    continue
                fresh.append((token, key, receipt))
            if self._active_seal is not None:
                if not retire_drops and all(
                    self._pending_persisted.get(key) == (token, receipt)
                    and self._pending_durable.get(key) == (token, receipt)
                    for token, key, receipt in fresh
                ):
                    return
                raise RuntimeError("dynamic lineage epoch is sealed for H23 commit")
            for pair in retire_drops:
                self._pending_dropped.pop(pair, None)
            for token, key, receipt in fresh:
                self._pending_persisted[key] = (token, receipt)
                self._pending_durable[key] = (token, receipt)

    def _stage_publication_drop(self, label, mode, expected_revision) -> None:
        label, expected = int(label), int(expected_revision)
        with self._lock:
            self._require_preterminal()
            revision = self._ledger.current_revision(label, mode)
            token = self._receipt_suppliers.get((label, mode, revision))
            if token is None or expected != revision:
                raise ValueError("stale dynamic publication-drop revision")
            if mode not in self._records[token].written:
                raise ValueError(
                    "dynamic publication drop requires the exact supplier mode "
                    "to be written"
                )
            key = (token.key, mode)
            if self._dropped.get(key) is token:
                return
            if any(
                triple[:2] == key and supplier is token
                for triple, supplier in {
                    **self._persisted, **self._durable,
                }.items()
            ):
                raise ValueError("dynamic result is already persisted/durable")
            if self._active_seal is not None:
                if self._pending_dropped.get(key) == (token, revision):
                    return
                raise RuntimeError("dynamic lineage epoch is sealed for H23 commit")
            retire = set()
            for triple, (prior, _receipt) in {
                **self._pending_persisted, **self._pending_durable,
            }.items():
                if triple[:2] != key:
                    continue
                if prior is token:
                    raise ValueError("dynamic result is already persisted/durable")
                if not self._strictly_newer(token, prior):
                    raise ValueError(
                        "stale dynamic drop cannot replace a pending receipt"
                    )
                retire.add(triple)
            for triple in retire:
                self._pending_persisted.pop(triple, None)
                self._pending_durable.pop(triple, None)
            self._pending_dropped[key] = (token, revision)

    def _bind_live_session(self, session, owner_token) -> None:
        if owner_token is None:
            raise ValueError("live session must supply an exact owner token")
        with self._lock:
            self._require_preterminal()
            current = self._live_session_ref() if self._live_session_ref else None
            if current is session and self._live_owner_token is owner_token:
                return
            if current is not None or self._live_owner_token is not None:
                raise RuntimeError("dynamic accounting already has a bound live session")
            self._live_session_ref = weakref.ref(session)
            self._live_owner_token = owner_token

    def _require_bound(self, session, owner_token) -> None:
        current = self._live_session_ref() if self._live_session_ref else None
        if current is not session or self._live_owner_token is not owner_token:
            raise RuntimeError("epoch transition requires the exact bound live session")

    def _validate_pending_current(self) -> None:
        for _key, (_token, receipt) in {
            **self._pending_persisted, **self._pending_durable,
        }.items():
            if self._ledger.receipt(
                receipt.label, receipt.mode, receipt.target,
            ) != receipt:
                raise ValueError("epoch contains a stale target receipt")
        for (key, mode), (token, revision) in self._pending_dropped.items():
            label = self._frames[key].output_label
            if (
                self._ledger.current_revision(label, mode) != revision
                or self._receipt_suppliers.get((label, mode, revision)) is not token
            ):
                raise ValueError("epoch contains a stale publication drop")

    def _validate_epoch_terminality(
        self, *, terminal_state: DynamicRunState | None,
    ) -> None:
        if (
            terminal_state is DynamicRunState.FINISHED
            and self._prospective_outstanding_keys()
        ):
            raise RuntimeError(
                "session finish cannot abandon unresolved dynamic work"
            )

    def _prepare_bound_epoch(
        self,
        session,
        owner_token,
        *,
        kind: str,
        terminal_state: DynamicRunState | None = None,
    ):
        if kind not in {"epoch", "finish"}:
            raise ValueError("unknown dynamic epoch seal kind")
        with self._lock:
            self._require_preterminal()
            self._require_bound(session, owner_token)
            if self._active_seal is not None:
                raise RuntimeError("dynamic lineage epoch already has an active seal")
            self._validate_pending_current()
            if kind == "finish" and self._state is DynamicRunState.STOPPED:
                terminal_state = DynamicRunState.STOPPED
            self._validate_epoch_terminality(terminal_state=terminal_state)
            self._seal_ordinal += 1
            if kind == "epoch" and terminal_state is not None:
                raise ValueError("epoch commit seal cannot name a terminal state")
            if kind == "finish" and terminal_state not in {
                DynamicRunState.FINISHED, DynamicRunState.STOPPED,
            }:
                raise ValueError("finish seal requires an exact terminal state")
            seal = _EpochSeal(
                kind, self._seal_ordinal, self._epoch_revision, terminal_state,
            )
            self._active_seal = seal
            self._sealed_promoted = False
            return seal

    def _require_seal(self, session, owner_token, seal, *, kind: str) -> None:
        self._require_bound(session, owner_token)
        if self._active_seal is not seal or not isinstance(seal, _EpochSeal):
            raise RuntimeError("epoch transition requires the exact prepare seal")
        revision_matches = (
            seal.epoch_revision == self._epoch_revision
            or (
                self._sealed_promoted
                and seal.kind == "finish"
                and seal.epoch_revision + 1 == self._epoch_revision
            )
        )
        if seal.kind != kind or not revision_matches:
            raise RuntimeError("epoch prepare seal does not match this transition")

    def _unseal_bound_epoch(self, session, owner_token, seal) -> None:
        with self._lock:
            self._require_preterminal()
            if not isinstance(seal, _EpochSeal):
                raise RuntimeError("epoch transition requires the exact prepare seal")
            self._require_seal(session, owner_token, seal, kind=seal.kind)
            if self._sealed_promoted:
                raise RuntimeError("a promoted epoch seal cannot be abandoned")
            self._active_seal = None

    def _promote_epoch(self, epoch_identity, *, prevalidated: bool = False) -> None:
        if (
            epoch_identity == self._last_epoch_identity
            and not self._pending_persisted
            and not self._pending_durable
            and not self._pending_dropped
        ):
            return
        if not prevalidated:
            self._validate_pending_current()
        committed_tokens = {
            token
            for token, _receipt in self._pending_persisted.values()
        } | {
            token
            for token, _receipt in self._pending_durable.values()
        } | {
            token
            for token, _revision in self._pending_dropped.values()
        }
        persisted = tuple(value[1] for value in self._pending_persisted.values())
        durable = tuple(value[1] for value in self._pending_durable.values())
        if persisted:
            self._ledger.record_persisted(persisted)
        if durable:
            self._ledger.record_durable(durable)
        for (key, mode), (_token, revision) in self._pending_dropped.items():
            label = self._frames[key].output_label
            self._ledger.record_publication_dropped(
                label, mode, expected_revision=revision,
            )
        for key, (token, _receipt) in self._pending_persisted.items():
            self._persisted[key] = token
            self._dropped.pop(key[:2], None)
        for key, (token, _receipt) in self._pending_durable.items():
            self._durable[key] = token
            self._persisted[key] = token
            self._dropped.pop(key[:2], None)
        for key, (token, _revision) in self._pending_dropped.items():
            self._dropped[key] = token
            for triple in tuple(self._persisted):
                if triple[:2] == key:
                    self._persisted.pop(triple, None)
                    self._durable.pop(triple, None)
        for token in self._epoch_tokens:
            record = self._records[token]
            if (
                token in committed_tokens
                or self._key_is_canonical(token.key)
                or record.state in {
                    DynamicAttemptState.FAILED,
                    DynamicAttemptState.CANCELLED,
                }
            ):
                record.epoch_finalized = True
        self._clear_epoch()
        self._epoch_revision += 1
        self._last_epoch_identity = epoch_identity

    def _discard_epoch(self, reason: str) -> None:
        for token in self._epoch_tokens:
            record = self._records[token]
            if record.state is DynamicAttemptState.ACCEPTED:
                label = self._frames[token.key].output_label
                self._ledger.record_outcome(
                    label,
                    ItemDisposition.CANCELLED_BEFORE_COMPLETION,
                    error=str(reason),
                    attempt=record.ledger_attempt,
                )
                record.error = str(reason)
                record.state = DynamicAttemptState.CANCELLED
            elif record.state is DynamicAttemptState.FAILED_RETRYABLE:
                record.retryable = False
                record.state = DynamicAttemptState.FAILED
            record.epoch_finalized = True
            record.epoch_aborted = True
        for record in self._records.values():
            if record.state in {
                DynamicAttemptState.PROVISIONAL,
                DynamicAttemptState.ENQUEUED,
            }:
                record.error = str(reason)
                record.state = DynamicAttemptState.CANCELLED
            elif record.state is DynamicAttemptState.FAILED_RETRYABLE:
                record.retryable = False
                record.state = DynamicAttemptState.FAILED
        self._clear_epoch(discard_all=True)

    def _clear_epoch(self, *, discard_all: bool = False) -> None:
        if discard_all:
            self._epoch_tokens.clear()
        else:
            self._epoch_tokens[:] = [
                token for token in self._epoch_tokens
                if not self._records[token].epoch_finalized
            ]
        self._pending_persisted.clear()
        self._pending_durable.clear()
        self._pending_dropped.clear()

    def _commit_bound_epoch(self, session, owner_token, seal, identity) -> None:
        with self._lock:
            self._require_preterminal()
            self._require_seal(session, owner_token, seal, kind="epoch")
            self._promote_epoch(identity, prevalidated=True)
            self._active_seal = None
            self._sealed_promoted = False

    def stop(self) -> DynamicRunSnapshot:
        with self._lock:
            self._require_unsealed()
            if self._state is DynamicRunState.ACTIVE:
                self._state = DynamicRunState.STOPPED
            return self.snapshot()

    def bind_light_1d(self, lease, *, cleanup_hooks) -> None:
        with self._lock:
            self._require_preterminal()
            light_module = sys.modules.get(
                "xrd_tools.session.light_1d_retention",
            )
            lease_type = getattr(light_module, "Light1DRetentionLease", None)
            hooks_type = getattr(light_module, "Light1DCleanupHooks", None)
            if lease_type is None or type(lease) is not lease_type:
                raise TypeError("dynamic run requires the exact light-1D lease")
            if hooks_type is None or type(cleanup_hooks) is not hooks_type:
                raise TypeError("dynamic run requires exact light-1D cleanup hooks")
            if self._state is not DynamicRunState.ACTIVE:
                raise RuntimeError("a non-active dynamic run cannot bind light-1D")
            if getattr(getattr(lease, "state", None), "value", None) != "active":
                raise RuntimeError("dynamic run requires an active light-1D lease")
            if self._light_lease is lease:
                if cleanup_hooks is not self._light_cleanup_hooks:
                    raise RuntimeError("light-1D cleanup hooks cannot change")
                return
            if self._light_lease is not None:
                raise RuntimeError("dynamic run already owns a light-1D lease")
            if int(getattr(lease, "generation", -1)) != self._generation:
                raise ValueError("light-1D lease belongs to another generation")
            self._light_lease = lease
            self._light_cleanup_hooks = cleanup_hooks

    def _release_light(self, terminal: DynamicRunState) -> None:
        with self._lock:
            lease = self._light_lease
            retry_token = self._light_retry_token
            hooks = self._light_cleanup_hooks
        if lease is None:
            with self._lock:
                self._cleanup_receipt = DynamicCleanupReceipt(
                    terminal, None, None, True,
                )
                self._state = terminal
            return
        try:
            if retry_token is None:
                receipt = lease.release(
                    reason=terminal.value, hooks=hooks,
                )
            else:
                receipt = lease.retry_cleanup(
                    retry_token, hooks=hooks,
                )
        except BaseException as exc:
            token = getattr(exc, "token", None)
            pending = getattr(exc, "receipt", None)
            if token is None or pending is None:
                raise
            with self._lock:
                self._light_retry_token = token
                self._cleanup_receipt = DynamicCleanupReceipt(
                    terminal, pending, token, False,
                )
                self._state = DynamicRunState.CLEANUP_PENDING
            raise
        with self._lock:
            self._cleanup_receipt = DynamicCleanupReceipt(
                terminal, receipt, retry_token, True,
            )
            self._state = terminal

    def _drop_terminal_owners(self) -> None:
        self._live_session_ref = None
        self._live_owner_token = None
        self._light_lease = None
        self._light_cleanup_hooks = None
        self._light_retry_token = None
        self._active_seal = None
        self._sealed_promoted = False

    def _finish_bound(self, session, owner_token, seal, identity) -> None:
        with self._lock:
            self._require_seal(session, owner_token, seal, kind="finish")
            terminal = seal.terminal_state
            if terminal not in {DynamicRunState.FINISHED, DynamicRunState.STOPPED}:
                raise RuntimeError("finish seal has no exact terminal intent")
            if not self._sealed_promoted:
                if self._terminal_intent not in {None, terminal}:
                    raise RuntimeError("terminal cleanup retry changed its intent")
                self._terminal_intent = terminal
                self._promote_epoch(identity, prevalidated=True)
                if terminal is DynamicRunState.STOPPED:
                    self._discard_epoch("LiveScan stopped after canonical prefix")
                self._sealed_promoted = True
            elif self._terminal_intent is not terminal:
                raise RuntimeError("terminal cleanup retry changed its intent")
        self._release_light(terminal)
        with self._lock:
            self._drop_terminal_owners()

    def _stop_bound(self, session, owner_token, seal, reason: str) -> None:
        with self._lock:
            self._require_seal(session, owner_token, seal, kind="finish")
            if seal.terminal_state is not DynamicRunState.STOPPED:
                raise RuntimeError("only a stopped finish seal may discard an epoch")
            if not self._sealed_promoted:
                if self._terminal_intent not in {None, DynamicRunState.STOPPED}:
                    raise RuntimeError("terminal cleanup retry changed its intent")
                self._terminal_intent = DynamicRunState.STOPPED
                self._discard_epoch(reason)
                self._sealed_promoted = True
            elif self._terminal_intent is not DynamicRunState.STOPPED:
                raise RuntimeError("terminal cleanup retry changed its intent")
        self._release_light(DynamicRunState.STOPPED)
        with self._lock:
            self._drop_terminal_owners()

    def _abort_bound(self, session, owner_token, reason: str) -> None:
        with self._lock:
            self._require_bound(session, owner_token)
            if self._state is DynamicRunState.CLEANUP_PENDING:
                terminal = self._terminal_intent
                if terminal is None:
                    raise RuntimeError("cleanup-pending run lost its terminal intent")
            else:
                if self._active_seal is not None and self._sealed_promoted:
                    raise RuntimeError(
                        "a canonical terminal seal cannot be changed to abort"
                    )
                if self._terminal_intent not in {None, DynamicRunState.ABORTED}:
                    raise RuntimeError("terminal cleanup retry changed its intent")
                self._terminal_intent = DynamicRunState.ABORTED
                self._discard_epoch(reason)
                self._sealed_promoted = self._active_seal is not None
                terminal = DynamicRunState.ABORTED
        self._release_light(terminal)
        with self._lock:
            self._drop_terminal_owners()

    def _current_result_supplier(self, key, mode):
        label = self._frames[key].output_label
        revision = self._ledger.current_revision(label, mode)
        return self._receipt_suppliers.get((label, mode, revision))

    def _stage_sets(self):
        accepted, completed, written = set(), set(), set()
        completed_attempts, written_attempts = {}, {}
        for key, tokens in self._attempts.items():
            records = [self._records[token] for token in tokens]
            if any(record.accepted for record in records):
                accepted.add(key)
            latest = next((
                record for record in reversed(records)
                if record.accepted and record.state in {
                    DynamicAttemptState.COMPLETED,
                    DynamicAttemptState.FAILED,
                    DynamicAttemptState.FAILED_RETRYABLE,
                    DynamicAttemptState.CANCELLED,
                }
            ), None)
            if latest is not None:
                completed.add(key)
                completed_attempts[key] = latest.token
            mode_written = bool(self._ledger.required_modes)
            for mode in self._ledger.required_modes:
                token = self._current_result_supplier(key, mode)
                if token is not None and mode in self._records[token].written:
                    written_attempts[(key, mode)] = token
                else:
                    mode_written = False
            if mode_written:
                written.add(key)
        return accepted, completed, written, completed_attempts, written_attempts

    @staticmethod
    def _contiguous(ordinals, members) -> int:
        value = -1
        while value + 1 in ordinals and ordinals[value + 1] in members:
            value += 1
        return value

    def snapshot(self) -> DynamicRunSnapshot:
        with self._lock:
            (accepted, completed, written, completed_attempts,
             written_attempts) = self._stage_sets()
            persisted, durable, dropped = (
                frozenset(self._persisted), frozenset(self._durable),
                frozenset(self._dropped),
            )
            persisted_frames = {
                key for key in self._frames
                if self._key_has_complete_targets(key, self._persisted)
            }
            durable_frames = {
                key for key in self._frames
                if self._key_has_complete_targets(key, self._durable)
            }
            retry_owned = {
                key for key, tokens in self._attempts.items()
                if tokens and self._records[tokens[-1]].state in {
                    DynamicAttemptState.PROVISIONAL, DynamicAttemptState.ENQUEUED,
                    DynamicAttemptState.FAILED_RETRYABLE,
                }
            }
            in_flight = set()
            for key, tokens in self._attempts.items():
                if not tokens:
                    continue
                latest = self._records[tokens[-1]]
                if latest.state is DynamicAttemptState.ACCEPTED:
                    in_flight.add(key)
                elif (
                    latest.state is DynamicAttemptState.COMPLETED
                    and not latest.epoch_aborted
                    and not self._key_is_canonical(key)
                ):
                    in_flight.add(key)
            high = {
                group: DynamicGroupHighWater(
                    self._contiguous(ordinals, accepted),
                    self._contiguous(ordinals, completed),
                    self._contiguous(ordinals, written),
                    self._contiguous(ordinals, persisted_frames),
                    self._contiguous(ordinals, durable_frames),
                )
                for group, ordinals in self._ordinals.items()
            }
            return DynamicRunSnapshot(
                self._generation, self._state, frozenset(self._frames),
                frozenset(self._logical_enqueued),
                frozenset(accepted), frozenset(completed), frozenset(written),
                persisted, durable, dropped,
                frozenset(self._pending_persisted),
                frozenset(self._pending_durable),
                frozenset(self._pending_dropped),
                frozenset(in_flight), frozenset(retry_owned),
                MappingProxyType({k: tuple(v) for k, v in self._attempts.items()}),
                MappingProxyType({t: r.state for t, r in self._records.items()}),
                MappingProxyType({t: int(r.ledger_attempt) for t, r in self._records.items()
                                  if r.accepted}),
                frozenset(t for t, r in self._records.items() if r.epoch_aborted),
                MappingProxyType({t: r.error for t, r in self._records.items()
                                  if r.error is not None}),
                MappingProxyType(completed_attempts),
                MappingProxyType(written_attempts),
                MappingProxyType(dict(self._persisted)),
                MappingProxyType(dict(self._durable)),
                MappingProxyType(dict(self._dropped)),
                MappingProxyType(high), self._epoch_revision,
                self._cleanup_receipt, len(self.owner_census()),
            )

    def owner_census(self) -> tuple[object, ...]:
        with self._lock:
            values = [self._ledger, self._writer_boundary]
            if self._light_lease is not None:
                values.append(self._light_lease)
            session = self._live_session_ref() if self._live_session_ref else None
            if session is not None:
                values.append(session)
                sink = getattr(session, "sink", None)
                values.extend(value for value in (
                    getattr(sink, "_lease", None), getattr(sink, "_writer", None),
                ) if value is not None)
            result, seen = [], set()
            for value in values:
                if id(value) not in seen:
                    seen.add(id(value))
                    result.append(value)
            return tuple(result)
