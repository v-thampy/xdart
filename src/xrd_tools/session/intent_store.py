"""Revision-qualified, Qt-free ownership for mutable run intent.

``RunIntentStore`` is the canonical boundary between editable next-run state
and a frozen run attempt.  It intentionally owns both edit revision and run
generation without introducing GUI, source, or writer dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock

from .run_configuration import FrozenRunConfiguration, RunIntent


def _validate_revision(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("expected_revision must be an integer")
    if value < 0:
        raise ValueError("expected_revision cannot be negative")
    return value


def _copy_intent(value: RunIntent) -> RunIntent:
    """Copy through the canonical detached RunIntent owner."""

    return value.clone_candidate()


@dataclass(frozen=True, slots=True)
class RunIntentSnapshot:
    """A deep, revision-qualified view of editable run intent."""

    revision: int
    _intent: RunIntent = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_intent", _copy_intent(self._intent))

    @classmethod
    def _from_owned_intent(
        cls,
        *,
        revision: int,
        intent: RunIntent,
    ) -> "RunIntentSnapshot":
        """Construct from the store's already-independent private copy."""

        snapshot = object.__new__(cls)
        object.__setattr__(snapshot, "revision", revision)
        object.__setattr__(snapshot, "_intent", intent)
        return snapshot

    def thaw(self) -> RunIntent:
        """Return an independent mutable copy of the captured intent."""

        return _copy_intent(self._intent)


@dataclass(frozen=True, slots=True)
class IntentCommitAccepted:
    revision: int
    snapshot: RunIntentSnapshot


@dataclass(frozen=True, slots=True)
class IntentFreezeAccepted:
    revision: int
    configuration: FrozenRunConfiguration


@dataclass(frozen=True, slots=True)
class IntentRecaptureRequired:
    expected_revision: int
    snapshot: RunIntentSnapshot


class RunIntentStore:
    """One locked owner of editable intent, revision, and run generation."""

    def __init__(self, initial: RunIntent | None = None) -> None:
        if initial is not None and not isinstance(initial, RunIntent):
            raise TypeError("initial must be RunIntent or None")
        self._lock = RLock()
        self._intent = _copy_intent(initial) if initial is not None else RunIntent()
        self._revision = 0

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    def _snapshot_locked(self) -> RunIntentSnapshot:
        return RunIntentSnapshot._from_owned_intent(
            revision=self._revision,
            intent=_copy_intent(self._intent),
        )

    def snapshot(self) -> RunIntentSnapshot:
        with self._lock:
            return self._snapshot_locked()

    def _recapture_locked(self, expected_revision: int) -> IntentRecaptureRequired:
        return IntentRecaptureRequired(
            expected_revision=expected_revision,
            snapshot=self._snapshot_locked(),
        )

    def commit(
        self,
        candidate: RunIntent,
        *,
        expected_revision: int,
    ) -> IntentCommitAccepted | IntentRecaptureRequired:
        """Install one whole-value edit only if it matches the current revision."""

        expected_revision = _validate_revision(expected_revision)
        if not isinstance(candidate, RunIntent):
            raise TypeError("candidate must be RunIntent")
        with self._lock:
            if expected_revision != self._revision:
                return self._recapture_locked(expected_revision)
            installed = _copy_intent(candidate)
            installed.generation = self._intent.generation
            self._intent = installed
            self._revision += 1
            return IntentCommitAccepted(
                revision=self._revision,
                snapshot=self._snapshot_locked(),
            )

    def freeze(
        self,
        *,
        expected_revision: int,
        gi_motor_choices: list[str] | tuple[str, ...] | None = None,
    ) -> IntentFreezeAccepted | IntentRecaptureRequired:
        """Freeze the owned intent exactly once after a locked revision check."""

        expected_revision = _validate_revision(expected_revision)
        with self._lock:
            if expected_revision != self._revision:
                return self._recapture_locked(expected_revision)
            configuration = self._intent.freeze(
                gi_motor_choices=gi_motor_choices,
            )
            return IntentFreezeAccepted(
                revision=self._revision,
                configuration=configuration,
            )


__all__ = [
    "IntentCommitAccepted",
    "IntentFreezeAccepted",
    "IntentRecaptureRequired",
    "RunIntentSnapshot",
    "RunIntentStore",
]
