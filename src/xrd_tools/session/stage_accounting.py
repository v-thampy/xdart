# -*- coding: utf-8 -*-
"""The Qt-free typed stage-accounting contract (H10-C1, handoff §4.1).

One :class:`StageLedger` per run is the accounting authority for the five
frozen processing stages — *accepted*, *completed*, *written*, *persisted*,
*durable* — plus the separate source-observation axis (*discovered*,
*enqueued*, *skipped*).  Everything is an identity set or an identity-keyed
mapping; there is no increment-only counter anywhere, so replays and
replace/re-feeds are idempotent by construction.

Frozen semantics:

* **accepted** — the session accepted the item for processing.  Distinct from
  source-queue admission (``enqueued``), which lives on the source axis.
* **completed** — reduction reached a typed result or a typed terminal
  disposition; :class:`ItemDisposition` distinguishes a result from
  ``FAILED`` and ``CANCELLED_BEFORE_COMPLETION``.
* **written** — the TOP-LEVEL run sink hook returned successfully (including
  a buffering sink whose write only stashes in memory).  Deliberately NOT
  target-qualified: when one child of a :class:`CompositeSink` succeeds and a
  later child raises, the aggregate pair is *not* written even though the
  earlier child's own target receipt stays truthful.  No child success may
  widen this set, and target persistence is therefore not required to be a
  subset of it before H23 supplies cross-target transaction semantics.
* **persisted** — exact ``(label, mode)`` pairs passed through the
  persistence layer into the logical on-disk representation of one declared
  target.
* **durable** — the sink's successful flush/close/publish boundary reports
  those exact pairs recoverable by the application.  Application-level
  recoverability only — no power-loss/``fsync`` guarantee is claimed.

Processing-attempt identity is the admitted item plus a monotonic per-label
``attempt_revision``: every accepted submit advances it and makes the new
attempt's ``PENDING`` state visible *before* any compute outcome, sink write
or public completion callback for that item can publish.  ``accepted`` is the
distinct-label projection, while ``pending``, ``completed``, ``dispositions``
and ``errors`` are LATEST-attempt label projections — an older overlapping
attempt still mints the ``result_revision`` for whatever it produced, but it
can never replace a newer attempt's disposition or supply that attempt's
error, and the latest attempt's error is absent unless that same attempt
supplied one.  Every recorded outcome must name its exact accepted attempt
(§13.2.2) — no result, disposition or error identity can originate from an
unaccepted label or attempt — and exactly one terminal outcome identity exists
per ``(label, attempt_revision)``: an exact replay is a no-op (it cannot
re-mint a revision or invalidate a durable receipt), a contradictory replay
raises atomically, and each produced mode mints at most one revision per
outcome.  A replacement awaiting compute keeps the prior typed result and
its certifications until a new typed result actually exists, so ``written`` /
``persisted`` / ``durable`` / ``mode_complete`` may coexist truthfully with a
latest-attempt ``PENDING``.

Compute identity is the admitted item plus result mode plus a monotonic
per-``(label, mode)`` ``result_revision`` minted on every completion that
produces the mode — replacement makes a previously durable mode dirty again
for exactly that mode.  Persistence identity additionally carries the frozen
output obligation/target identity (e.g. NeXus versus XYE).  A receipt
certifies exactly one ``(label, mode, revision, target)``; it can never widen
to a whole frame, and a failed operation must emit **no** receipt for the
pairs it failed to make recoverable.  ``FrameEvent.generation`` is a
caller-owned render-staleness stamp and is deliberately absent from every
identity here.

A frame is *mode-complete* only when every required result mode is durable at
its current revision for **every** declared obligation; with no declared
obligations nothing is ever reported durable (no vacuous recoverability).

The public compatibility projections are derived identity facts, never
counters: the submitted total is ``len(accepted)`` and the completed total is
``len(written_labels)`` — the labels whose top-level sink hook returned
successfully at least once during the run.  That history is deliberately
independent of the current-revision ``written`` projection, so a re-feed
cannot inflate the totals and a later replacement whose write fails cannot
decrement them.

Publication-dropped is a per-``(label, mode, revision)`` terminal state:
the pair can never be persisted or durable at that revision (receipts for it
raise), while a later re-feed mints a new revision that may publish cleanly.

Threading: mutators may be called from the session's writer thread and the
orchestrating caller thread concurrently; every method takes the internal
lock and :meth:`StageLedger.snapshot` returns an immutable value object.
Ordering across threads is not required — sets commute — and
:meth:`StageSnapshot.verify_conservation` checks cross-stage containment at
quiescent points.

This module is Qt-free and import-light (stdlib + the core mode-key constant
only).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Callable, Hashable, Iterable, Mapping

from xrd_tools.core import DEFAULT_MODE_KEY

__all__ = [
    "freeze_target_map",
    "ItemDisposition",
    "ResultMode",
    "StageLedger",
    "StageReceipt",
    "StageSnapshot",
]

_KINDS = ("1d", "2d")


@dataclass(frozen=True, slots=True)
class ResultMode:
    """One result mode: the dimension kind plus the (GI) mode key."""

    kind: str                    # "1d" | "2d"
    key: str                     # DEFAULT_MODE_KEY or a GI mode value

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"ResultMode.kind must be one of {_KINDS}; "
                             f"got {self.kind!r}")

    @classmethod
    def one_d(cls, key: str = DEFAULT_MODE_KEY) -> "ResultMode":
        return cls("1d", str(key))

    @classmethod
    def two_d(cls, key: str = DEFAULT_MODE_KEY) -> "ResultMode":
        return cls("2d", str(key))


class ItemDisposition(Enum):
    """Typed per-item processing state of the LATEST accepted attempt.
    PENDING means that attempt has no typed outcome yet; the other three are
    the typed result / typed terminal dispositions of §4.1."""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED_BEFORE_COMPLETION = "cancelled_before_completion"


@dataclass(frozen=True, slots=True)
class _AttemptState:
    attempt: int
    disposition: ItemDisposition
    error: str | None


@dataclass(frozen=True, slots=True)
class StageReceipt:
    """One persistence/durability certification: exactly one
    ``(label, mode, revision, target)`` — never a frame, never unqualified."""

    label: int
    mode: ResultMode
    revision: int
    target: str


@dataclass(frozen=True, slots=True)
class _StageTargetProjection:
    """Current target truth for one label at one ledger instant."""

    persisted: frozenset[tuple[ResultMode, str]]
    durable: frozenset[tuple[ResultMode, str]]
    publication_dropped: frozenset[ResultMode]


@dataclass(frozen=True, slots=True)
class StageSnapshot:
    """Immutable public accounting snapshot (the §4.1 exposure contract)."""

    required_modes: tuple[ResultMode, ...]
    obligations: frozenset[str]
    accepted: frozenset[int]
    refused: frozenset[int]
    pending: frozenset[int]
    completed: frozenset[int]
    dispositions: Mapping[int, ItemDisposition]
    errors: Mapping[int, str]
    attempt_revisions: Mapping[int, int]
    revisions: Mapping[tuple[int, ResultMode], int]
    written: frozenset[tuple[int, ResultMode]]
    # Historical distinct-label write identity: every label whose TOP-LEVEL
    # sink hook returned successfully at least once, independent of later
    # replacement revisions.  This — not the current-revision ``written``
    # projection — backs the public completed-progress compatibility values.
    written_labels: frozenset[int]
    persisted: frozenset[tuple[int, ResultMode, str]]
    durable: frozenset[tuple[int, ResultMode, str]]
    publication_dropped: frozenset[tuple[int, ResultMode]]
    mode_complete: frozenset[int]
    discovered: frozenset[Hashable]
    enqueued: frozenset[Hashable]
    skipped: Mapping[Hashable, str]
    targets_by_mode: Mapping[ResultMode, frozenset[str]] = field(
        default_factory=lambda: MappingProxyType({}))

    def verify_conservation(self) -> None:
        """Cross-stage containment at a quiescent point; raises ``ValueError``
        on the first violation.

        Deliberately NOT checked: that target-qualified ``persisted``/
        ``durable`` pairs are a subset of the aggregate ``written`` set.  §4.1
        fixes the partial-``CompositeSink`` case — an earlier child's target
        can be genuinely recoverable while the aggregate hook raised — and
        cross-target atomicity belongs to H23, not H10.
        """
        if not self.durable <= self.persisted:
            raise ValueError("conservation: durable must be a subset of persisted")
        if not self.written_labels <= self.accepted:
            raise ValueError("conservation: a written label was never accepted")
        # §13.2.2 extension: no result/disposition/error/attempt identity can
        # originate from a label that was never accepted.
        for label in self.dispositions:
            if label not in self.accepted:
                raise ValueError(
                    f"conservation: disposition for never-accepted label {label!r}")
        for label in self.errors:
            if label not in self.accepted:
                raise ValueError(
                    f"conservation: error for never-accepted label {label!r}")
        for label in self.attempt_revisions:
            if label not in self.accepted:
                raise ValueError(
                    f"conservation: attempt_revision for never-accepted label "
                    f"{label!r}")
        for label, _mode in self.revisions:
            if label not in self.accepted:
                raise ValueError(
                    f"conservation: result revision for never-accepted label "
                    f"{label!r}")
        for label, _mode in self.written:
            if label not in self.accepted:
                raise ValueError(f"conservation: written label {label!r} was never accepted")
        for label, _mode, _target in self.persisted:
            if label not in self.accepted:
                raise ValueError(f"conservation: persisted label {label!r} was never accepted")
        if not self.completed <= self.accepted:
            raise ValueError("conservation: completed must be a subset of accepted")
        if not self.pending <= self.accepted:
            raise ValueError("conservation: pending must be a subset of accepted")
        if self.pending & self.completed:
            raise ValueError("conservation: pending and completed overlap")
        if not self.mode_complete <= self.accepted:
            raise ValueError("conservation: mode_complete must be a subset of accepted")
        if self.enqueued & set(self.skipped):
            raise ValueError("conservation: a source identity is both enqueued and skipped")


def freeze_target_map(name: str, supplied, required: Iterable[ResultMode], *,
                      allow_empty: bool = False, exactly_one: bool = False,
                      within: Mapping[ResultMode, frozenset[str]] | None = None,
                      ) -> Mapping[ResultMode, frozenset[str]]:
    """Validate + freeze ONE per-mode target map; copies the containers."""
    frozen: dict[ResultMode, frozenset[str]] = {}
    for mode, targets in dict(supplied).items():
        if not isinstance(mode, ResultMode):
            raise TypeError(f"{name} keys must be ResultMode; got {mode!r}")
        values = tuple(targets)
        for target in values:
            if not isinstance(target, str) or not target:
                raise ValueError(
                    f"{name}[{mode!r}] target {target!r} must be a non-empty string")
        if not values and not allow_empty:
            raise ValueError(f"{name}[{mode!r}] must declare a target")
        if exactly_one and len(set(values)) != 1:
            raise ValueError(
                f"{name}[{mode!r}] must declare EXACTLY ONE applicable target; "
                f"got {list(values)!r}")
        if within is not None and not set(values) <= set(within.get(mode, ())):
            raise ValueError(
                f"{name}[{mode!r}] {sorted(set(values))} must be a subset of that "
                f"mode's applicable targets {sorted(within.get(mode, ()))}")
        frozen[mode] = frozenset(values)
    if set(frozen) != set(required):
        raise ValueError(
            f"{name} keyset {sorted(map(repr, frozen))} must equal the "
            f"required-mode set {sorted(map(repr, required))}")
    return MappingProxyType(frozen)


class StageLedger:
    """The one per-run stage-accounting owner (identity sets, typed receipts).

    ``required_modes`` and ``obligations`` are frozen at construction — they
    are the run's declared result modes and output obligations from which the
    frame-level durable total (``mode_complete``) is derived.
    """

    def __init__(self, *, required_modes: Iterable[ResultMode],
                 obligations: Iterable[str] = (),
                 targets_by_mode: Mapping[ResultMode, Iterable[str]] | None = None,
                 ) -> None:
        self._required_modes = tuple(required_modes)
        for mode in self._required_modes:
            if not isinstance(mode, ResultMode):
                raise TypeError(f"required_modes entries must be ResultMode; got {mode!r}")
        obligations = frozenset(str(t) for t in obligations)
        self._targets_by_mode = self._freeze_targets(
            self._required_modes, obligations, targets_by_mode)
        self._obligations = (
            obligations if targets_by_mode is None
            else frozenset().union(*self._targets_by_mode.values())
            if self._targets_by_mode else frozenset())
        self._lock = threading.Lock()
        self._refused: set[int] = set()
        self._attempts: dict[int, _AttemptState] = {}
        # One terminal outcome identity per (label, attempt_revision): the
        # exact (outcome, produced-mode set, error) recorded for that attempt.
        # Exact replay is a no-op; a contradictory replay raises (§13.2.2).
        self._outcomes: dict[tuple[int, int],
                             tuple[ItemDisposition, frozenset[ResultMode],
                                   str | None]] = {}
        self._revisions: dict[tuple[int, ResultMode], int] = {}
        self._written: dict[tuple[int, ResultMode], int] = {}
        self._written_labels: set[int] = set()
        self._persisted: dict[tuple[int, ResultMode, str], int] = {}
        self._durable: dict[tuple[int, ResultMode, str], int] = {}
        self._dropped: dict[tuple[int, ResultMode], int] = {}
        self._discovered: set[Hashable] = set()
        self._enqueued: set[Hashable] = set()
        self._skipped: dict[Hashable, str] = {}

    @staticmethod
    def _freeze_targets(required, obligations, supplied):
        if supplied is None:
            return MappingProxyType({mode: obligations for mode in required})
        if obligations:
            raise ValueError(
                "a non-empty global obligations set cannot be combined with an "
                "explicit targets_by_mode map; declare applicability once")
        return freeze_target_map("targets_by_mode", supplied, required)

    # -- frozen run configuration ------------------------------------------
    @property
    def required_modes(self) -> tuple[ResultMode, ...]:
        return self._required_modes

    @property
    def obligations(self) -> frozenset[str]:
        return self._obligations

    @property
    def targets_by_mode(self) -> Mapping[ResultMode, frozenset[str]]:
        return self._targets_by_mode

    # -- processing axis ----------------------------------------------------
    def record_accepted(self, label: int, *, publish_acceptance:
                        Callable[[int], None] | None = None) -> int:
        """The session accepted *label* for processing; returns the new
        monotonic per-label ``attempt_revision``.

        The new attempt is ``PENDING`` from this instant — a prior attempt's
        disposition or error may never masquerade as the latest attempt's — but
        the prior typed result and its certifications are untouched until a new
        typed result actually exists.  ``accepted`` itself is an identity set,
        so re-feed cannot inflate it; post-assignment exceptions complete the
        outside-lock publisher before propagating.
        """
        label = int(label)
        state: _AttemptState | None = None
        try:
            with self._lock:
                prior = self._attempts.get(label)
                attempt = (prior.attempt if prior is not None else 0) + 1
                state = _AttemptState(attempt, ItemDisposition.PENDING, None)
                self._attempts[label] = state
            if publish_acceptance is not None:
                publish_acceptance(state.attempt)
        except BaseException:
            if state is not None and publish_acceptance is not None:
                with self._lock:
                    committed = self._attempts.get(label) is state
                if committed:
                    publish_acceptance(state.attempt)
            raise
        return state.attempt

    def record_refused(self, label: int) -> None:
        """The session refused/dropped an offered item without accepting it."""
        with self._lock:
            self._refused.add(int(label))

    def current_attempt(self, label: int) -> int:
        """The latest accepted ``attempt_revision`` for *label* (0 if never
        accepted)."""
        with self._lock:
            state = self._attempts.get(int(label))
            return state.attempt if state is not None else 0

    def record_outcome(self, label: int, outcome: ItemDisposition, *,
                       produced: Iterable[ResultMode] = (),
                       error: str | None = None,
                       attempt: int | None = None) -> None:
        """One typed per-item compute outcome (from the engine's receipt).

        ``COMPLETED`` mints a new ``result_revision`` for exactly the DISTINCT
        modes in *produced*; ``FAILED``/``CANCELLED_BEFORE_COMPLETION`` mint
        nothing (no typed result exists, so prior revisions stay
        authoritative).

        *attempt* is REQUIRED to name the exact accepted ``attempt_revision``
        this outcome belongs to (§13.2.2): an outcome for an unaccepted label,
        a never-accepted attempt (forged/future/zero), or a missing attempt
        identity raises ``ValueError`` and mutates nothing.  Exactly one
        terminal outcome identity is retained per ``(label, attempt)`` — an
        exact replay is a no-op (it cannot re-mint a revision or invalidate a
        durable receipt), a contradictory replay raises.  A legal OLDER
        accepted attempt still owns whatever it produced — it mints its result
        revisions once — but it can neither replace the latest attempt's
        disposition nor supply that attempt's error.  The complete batch is
        validated before any mutation, so an invalid batch is atomic.
        """
        label = int(label)
        batch = tuple(produced)
        if outcome is ItemDisposition.PENDING:
            raise ValueError("record_outcome requires a typed outcome, not PENDING")
        if batch and outcome is not ItemDisposition.COMPLETED:
            raise ValueError(f"{outcome.name} outcome cannot carry produced modes")
        for mode in batch:
            if not isinstance(mode, ResultMode):
                raise TypeError(f"produced entries must be ResultMode; got {mode!r}")
        # Each produced mode mints at most one revision per outcome: dedup the
        # batch (order-preserving) before it is applied.
        modes = tuple(dict.fromkeys(batch))
        err = str(error) if error else None
        identity = (outcome, frozenset(modes), err)
        with self._lock:
            latest_state = self._attempts.get(label)
            if latest_state is None:
                raise ValueError(
                    f"outcome for label {label!r}, which has no accepted attempt")
            latest = latest_state.attempt
            if attempt is None:
                raise ValueError(
                    f"outcome for label {label!r} carries no attempt_revision; "
                    "the exact accepted attempt is required")
            if attempt != int(attempt):
                raise ValueError(
                    f"outcome for label {label!r} names a non-integral "
                    f"attempt {attempt!r}")
            attempt = int(attempt)
            if not 1 <= attempt <= latest:
                raise ValueError(
                    f"outcome for label {label!r} names attempt {attempt}, "
                    f"which was never accepted (latest accepted attempt is "
                    f"{latest})")
            recorded = self._outcomes.get((label, attempt))
            if recorded is not None:
                if recorded == identity:
                    return              # exact replay: a no-op by identity
                raise ValueError(
                    f"contradictory outcome replay for label {label!r} attempt "
                    f"{attempt}: {recorded!r} is already terminal")
            # Every validation passed — apply atomically under the lock.
            self._outcomes[(label, attempt)] = identity
            for mode in modes:
                pair = (label, mode)
                self._revisions[pair] = self._revisions.get(pair, 0) + 1
            if attempt < latest:
                return                  # a superseded attempt: view unchanged
            self._attempts[label] = _AttemptState(attempt, outcome, err)

    def record_written(self, label: int, modes: Iterable[ResultMode]) -> None:
        """The TOP-LEVEL run sink hook returned successfully for *label*.

        Certifies the exact ``(label, mode)`` pairs at their current revision
        and records the label in the historical distinct-label write identity
        behind the public completed-progress projections.  A write receipt for
        a never-produced pair is a conservation violation and raises; the batch
        is validated before any of it applies.
        """
        label = int(label)
        modes = tuple(modes)            # caller code never runs under the lock
        with self._lock:
            for mode in modes:
                if self._revisions.get((label, mode), 0) < 1:
                    raise ValueError(
                        f"written receipt for never-produced pair "
                        f"{(label, mode)!r}")
            for mode in modes:
                self._written[(label, mode)] = self._revisions[(label, mode)]
            self._written_labels.add(label)

    # -- derived public projections (identity facts, never counters) ---------
    def accepted_label_count(self) -> int:
        """Distinct accepted label identities — the identity behind the public
        submitted projection."""
        with self._lock:
            return len(self._attempts)

    def written_label_count(self) -> int:
        """Distinct labels whose top-level sink hook returned successfully at
        least once during the run — the identity behind the public completed
        projection.  Monotonic: a failed replacement cannot decrement it."""
        with self._lock:
            return len(self._written_labels)

    # -- persistence/durability receipts -------------------------------------
    def current_revision(self, label: int, mode: ResultMode) -> int:
        with self._lock:
            return self._revisions.get((int(label), mode), 0)

    def receipt(self, label: int, mode: ResultMode, target: str) -> StageReceipt:
        """Mint a receipt certifying the CURRENT revision of one pair for one
        declared target — what a flush/close boundary emits after success."""
        label = int(label)
        target = str(target)
        with self._lock:
            self._require_applicable_target(mode, target)
            revision = self._revisions.get((label, mode), 0)
            if revision < 1:
                raise ValueError(
                    f"no typed result exists for ({label!r}, {mode!r}); "
                    "nothing can be certified")
        return StageReceipt(label=label, mode=mode, revision=revision, target=target)

    def record_persisted(self, receipts: Iterable[StageReceipt]) -> None:
        """Exact pairs passed through the persistence layer (per target).
        The batch is validated first and applied atomically — an invalid
        receipt certifies nothing from its batch.

        The declared ``Iterable`` is materialized BEFORE the lock is taken: a
        generator legally mints its receipts through :meth:`receipt`, which
        takes the same non-reentrant lock.
        """
        batch = tuple(receipts)
        with self._lock:
            for receipt in batch:
                self._validate_receipt(receipt)
            for receipt in batch:
                self._apply_receipt(receipt, self._persisted)

    def record_durable(self, receipts: Iterable[StageReceipt]) -> None:
        """Exact pairs reported recoverable at the flush/close/publish
        boundary.  Durable implies persisted for the same identity.  The
        batch is materialized before the lock (see :meth:`record_persisted`),
        then validated first and applied atomically."""
        batch = tuple(receipts)
        with self._lock:
            for receipt in batch:
                self._validate_receipt(receipt)
            for receipt in batch:
                self._apply_receipt(receipt, self._persisted)
                self._apply_receipt(receipt, self._durable)

    def record_publication_dropped(self, label: int, mode: ResultMode, *,
                                   expected_revision: int) -> None:
        """REVISION-QUALIFIED: stale is a no-op, above-current is rejected."""
        label = int(label)
        expected = int(expected_revision)
        pair = (label, mode)
        with self._lock:
            revision = self._revisions.get(pair, 0)
            if revision < 1:
                raise ValueError(
                    f"publication drop for never-produced pair {pair!r}")
            if expected > revision:
                raise ValueError(
                    f"publication drop for pair {pair!r} names revision "
                    f"{expected}, above the current revision {revision}")
            if expected < revision:
                return
            if self._dropped.get(pair) == revision:
                return
            for key, certified in tuple(self._persisted.items()) + tuple(self._durable.items()):
                if key[0] == label and key[1] == mode and certified == revision:
                    raise ValueError(
                        f"pair {pair!r} revision {revision} is already "
                        "persisted/durable; dropping it now is incoherent")
            self._dropped[pair] = revision

    def _target_projections(
        self, labels: Iterable[int]
    ) -> Mapping[int, _StageTargetProjection]:
        """Return current target truth for *labels* from one ledger instant.

        This deliberately avoids building the public whole-run snapshot for
        the session's label-scoped record-store reconciliation.  Materialize
        before taking the non-reentrant lock so caller iterators cannot run
        ledger code while the lock is held.
        """
        requested = tuple(dict.fromkeys(int(label) for label in labels))
        with self._lock:
            result: dict[int, _StageTargetProjection] = {}
            for label in requested:
                persisted: set[tuple[ResultMode, str]] = set()
                durable: set[tuple[ResultMode, str]] = set()
                dropped: set[ResultMode] = set()
                for mode in self._required_modes:
                    pair = (label, mode)
                    revision = self._revisions.get(pair, 0)
                    if revision < 1:
                        continue
                    if self._dropped.get(pair) == revision:
                        dropped.add(mode)
                        continue
                    for target in self._targets_by_mode.get(mode, ()):
                        key = (label, mode, target)
                        target_pair = (mode, target)
                        if self._persisted.get(key) == revision:
                            persisted.add(target_pair)
                        if self._durable.get(key) == revision:
                            durable.add(target_pair)
                result[label] = _StageTargetProjection(
                    frozenset(persisted), frozenset(durable),
                    frozenset(dropped))
            return MappingProxyType(result)

    def _require_applicable_target(self, mode: ResultMode, target: str) -> None:
        """Against THAT MODE's targets, never the union."""
        applicable = self._targets_by_mode.get(mode)
        if applicable is None:
            raise ValueError(
                f"mode {mode!r} is not one of this run's required modes "
                f"{sorted(map(repr, self._required_modes))}")
        if target not in applicable:
            raise ValueError(
                f"target {target!r} does not apply to mode {mode!r} "
                f"{sorted(applicable)}")

    def _validate_receipt(self, receipt: StageReceipt) -> None:
        pair = (receipt.label, receipt.mode)
        self._require_applicable_target(receipt.mode, receipt.target)
        current = self._revisions.get(pair, 0)
        if receipt.revision < 1 or receipt.revision > current:
            raise ValueError(
                f"receipt revision {receipt.revision} is invalid for pair "
                f"{pair!r} at current revision {current}")
        if self._dropped.get(pair) == receipt.revision:
            raise ValueError(
                f"pair {pair!r} revision {receipt.revision} is "
                "publication-dropped and can never be persisted or durable")

    @staticmethod
    def _apply_receipt(receipt: StageReceipt,
                       registry: dict[tuple[int, ResultMode, str], int]) -> None:
        key = (receipt.label, receipt.mode, receipt.target)
        # Idempotent replay; a stale (older-revision) receipt never downgrades
        # a newer certification and never certifies the current revision.
        registry[key] = max(registry.get(key, 0), receipt.revision)

    # -- source-observation axis ---------------------------------------------
    def record_discovered(self, identity: Hashable) -> None:
        with self._lock:
            self._discovered.add(identity)

    def record_enqueued(self, identity: Hashable) -> None:
        """The source item was successfully admitted to the consumer/prefetch
        queue — earlier than, and distinct from, processing acceptance."""
        with self._lock:
            if identity in self._skipped:
                raise ValueError(
                    f"source identity {identity!r} was already skipped "
                    f"({self._skipped[identity]!r}); it cannot also be enqueued")
            self._enqueued.add(identity)

    def record_skipped(self, identity: Hashable, *, reason: str) -> None:
        reason = str(reason)
        with self._lock:
            if identity in self._enqueued:
                raise ValueError(
                    f"source identity {identity!r} was already enqueued; "
                    "it cannot also be skipped")
            existing = self._skipped.get(identity)
            if existing is not None and existing != reason:
                raise ValueError(
                    f"source identity {identity!r} was already skipped for "
                    f"{existing!r}; conflicting reason {reason!r}")
            self._skipped[identity] = reason

    # -- exposure -------------------------------------------------------------
    def snapshot(self) -> StageSnapshot:
        with self._lock:
            attempts = dict(self._attempts)
            dispositions = {label: state.disposition for label, state in attempts.items()}
            errors = {label: state.error for label, state in attempts.items()
                      if state.error is not None}
            revisions = dict(self._revisions)
            written = frozenset(
                pair for pair, revision in self._written.items()
                if revision == revisions.get(pair, 0)
            )
            dropped_now = frozenset(
                pair for pair, revision in self._dropped.items()
                if revision == revisions.get(pair, 0)
            )
            persisted = self._certified(self._persisted, revisions, dropped_now)
            durable = self._certified(self._durable, revisions, dropped_now)
            accepted = frozenset(attempts)
            mode_complete = self._mode_complete(accepted, durable)
            return StageSnapshot(
                required_modes=self._required_modes,
                obligations=self._obligations,
                accepted=accepted,
                refused=frozenset(self._refused),
                pending=frozenset(
                    label for label, disposition in dispositions.items()
                    if disposition is ItemDisposition.PENDING
                ),
                completed=frozenset(
                    label for label, disposition in dispositions.items()
                    if disposition is not ItemDisposition.PENDING
                ),
                dispositions=MappingProxyType(dispositions),
                errors=MappingProxyType(errors),
                attempt_revisions=MappingProxyType(
                    {label: state.attempt for label, state in attempts.items()}),
                revisions=MappingProxyType(revisions),
                written=written,
                written_labels=frozenset(self._written_labels),
                persisted=persisted,
                durable=durable,
                publication_dropped=dropped_now,
                mode_complete=mode_complete,
                discovered=frozenset(self._discovered),
                enqueued=frozenset(self._enqueued),
                skipped=MappingProxyType(dict(self._skipped)),
                targets_by_mode=self._targets_by_mode,
            )

    @staticmethod
    def _certified(registry: dict[tuple[int, ResultMode, str], int],
                   revisions: dict[tuple[int, ResultMode], int],
                   dropped_now: frozenset) -> frozenset:
        return frozenset(
            key for key, certified in registry.items()
            if certified == revisions.get((key[0], key[1]), 0)
            and (key[0], key[1]) not in dropped_now
        )

    def _mode_complete(self, accepted: frozenset[int],
                       durable: frozenset) -> frozenset[int]:
        """Durable at current revision for every target APPLICABLE to the mode."""
        if not self._required_modes:
            return frozenset()
        if any(not self._targets_by_mode.get(mode) for mode in self._required_modes):
            return frozenset()
        return frozenset(
            label for label in accepted
            if all((label, mode, target) in durable
                   for mode in self._required_modes
                   for target in self._targets_by_mode[mode])
        )
