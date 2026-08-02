"""Immutable value outcomes for the revision-qualified Start pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from xrd_tools.core.scan import SourceSpec
from xrd_tools.session.intent_store import (IntentCommitAccepted, IntentFreezeAccepted,
                                            IntentRecaptureRequired, RunIntentSnapshot)
from xrd_tools.session.run_configuration import FrozenRunConfiguration, RunIntent
from xrd_tools.sources.selection import DirectorySourceSpec

from .contracts import SourceCapture, SourceSelection, StartCapture
from .events import (CleanupStatus, DetachedDiagnostic, ExecutorAccepted, ExecutorClosed,
                     ExecutorStartFailed, LifecycleError, LifecycleResult, LifecycleStatus,
                     RequestId, RunIdentity, detached_diagnostic_is_valid,
                     detached_exception_strings)
from .state_machine import RunPhase


def invalid_result(operation: str) -> RecoveryFailure:
    return RecoveryFailure(
        operation,
        exception_detail(TypeError(f"{operation} returned an invalid result")),
    )


def inert_lifecycle_rejection() -> LifecycleResult:
    return LifecycleResult(
        LifecycleStatus.REJECTED, RunPhase.FAILED,
        error=LifecycleError.ILLEGAL_TRANSITION,
    )


class StartRecaptureCause(str, Enum):
    COMMIT_RACE = "commit_race"
    FREEZE_RACE = "freeze_race"
    SOURCE_OBSERVATION = "source_observation"
    RECAPTURE_ADVANCED = "recapture_advanced"
class StartRejection(str, Enum):
    LIFECYCLE = "lifecycle"
    STALE_CAPTURE = "stale_capture"
    STALE_RECAPTURE = "stale_recapture"
    SOURCE_OBSERVATION = "source_observation"
class StartRefusal(str, Enum):
    MISSING_SOURCE = "missing_source"
    SOURCE_CAPTURE_INVALID = "source_capture_invalid"
    SOURCE_CAPTURE_EXCEPTION = "source_capture_exception"
    FREEZE_VALIDATION = "freeze_validation"
    OUTPUT_PREFLIGHT = "output_preflight"
class StartFailureKind(str, Enum):
    CANONICAL_INVARIANT = "canonical_invariant"
    LIFECYCLE_INVARIANT = "lifecycle_invariant"
    EXECUTOR_INVARIANT = "executor_invariant"
    EXECUTOR_START_FAILED = "executor_start_failed"
@dataclass(frozen=True, slots=True)
class ExceptionDetail:
    type_module: str
    type_qualname: str
    message: str
@dataclass(frozen=True, slots=True)
class RecoveryFailure:
    operation: str
    exception: ExceptionDetail
@dataclass(frozen=True, slots=True)
class StartRecaptureRequired:
    request_id: RequestId
    superseded_capture_sequence: int
    cause: StartRecaptureCause
    current_snapshot: RunIntentSnapshot
    minimum_source_epoch: int = 0
@dataclass(frozen=True, slots=True)
class StartRejected:
    reason: StartRejection
    lifecycle_result: LifecycleResult | None = None
@dataclass(frozen=True, slots=True)
class StartRefused:
    request_id: RequestId
    reason: StartRefusal
    lifecycle_result: LifecycleResult
    detail: str = ""
    exception: ExceptionDetail | None = None
    recovery_failures: tuple[RecoveryFailure, ...] = ()
@dataclass(frozen=True, slots=True)
class StartLaunched:
    configuration: FrozenRunConfiguration
    source_capture: SourceCapture
    run_identity: RunIdentity
    lifecycle_result: LifecycleResult
    recovery_failures: tuple[RecoveryFailure, ...] = ()
@dataclass(frozen=True, slots=True)
class StartFailed:
    reason: StartFailureKind
    configuration: FrozenRunConfiguration | None
    run_identity: RunIdentity | None
    lifecycle_result: LifecycleResult
    cleanup_status: CleanupStatus
    detail: str = ""
    exception: ExceptionDetail | None = None
    recovery_failures: tuple[RecoveryFailure, ...] = ()


@dataclass(frozen=True, slots=True)
class StartClosed:
    lifecycle_result: LifecycleResult
    recovery_failures: tuple[RecoveryFailure, ...] = ()
    cleanup_status: CleanupStatus = CleanupStatus.CLEANED
    primary: DetachedDiagnostic | None = None
    cleanup_failures: tuple[DetachedDiagnostic, ...] = ()
    cleanup_identity: RunIdentity | None = None


def exception_detail(value: Exception) -> ExceptionDetail:
    """Project an exception into an immutable, value-only diagnostic."""

    return ExceptionDetail(*detached_exception_strings(value))


def snapshot_is_valid(value: object) -> bool:
    """Validate a store snapshot before any pipeline code reads from it."""

    try:
        return type(value) is RunIntentSnapshot and _nonnegative_int(value.revision) and type(value._intent) is RunIntent
    except Exception:
        return False


def commit_result_is_valid(value: object, expected_revision: int) -> bool:
    """Validate one canonical accepted-commit result before dereferencing it."""

    try:
        return (type(value) is IntentCommitAccepted and _nonnegative_int(value.revision)
                and value.revision == expected_revision + 1 and snapshot_is_valid(value.snapshot)
                and value.snapshot.revision == value.revision)
    except Exception:
        return False


def freeze_result_is_valid(value: object, expected_revision: int) -> bool:
    """Validate one canonical accepted-freeze result before using its config."""

    try:
        return (freeze_configuration(value) is not None and _nonnegative_int(value.revision)
                and value.revision == expected_revision)
    except Exception:
        return False


def freeze_configuration(value: object) -> FrozenRunConfiguration | None:
    try:
        return value.configuration if type(value) is IntentFreezeAccepted and type(value.configuration) is FrozenRunConfiguration else None
    except Exception:
        return None


def recapture_result_is_valid(value: object, expected_revision: int) -> bool:
    """Validate a stale-store result against the operation that produced it."""

    try:
        return (type(value) is IntentRecaptureRequired and _nonnegative_int(value.expected_revision)
                and value.expected_revision == expected_revision and snapshot_is_valid(value.snapshot)
                and value.snapshot.revision > expected_revision)
    except Exception:
        return False


def lifecycle_result_is_valid(value: object) -> bool:
    """Return whether an external lifecycle value has only scalar kernel fields."""

    try:
        return (type(value) is LifecycleResult and type(value.status) is LifecycleStatus
                and type(value.phase) is RunPhase
                and (value.request_id is None or request_id_is_valid(value.request_id))
                and (value.run_identity is None or run_identity_is_valid(value.run_identity))
                and (value.error is None or type(value.error) is LifecycleError))
    except Exception:
        return False


def request_id_is_valid(value: object) -> bool:
    try:
        return type(value) is RequestId and type(value.value) is int and value.value > 0
    except Exception:
        return False


def run_identity_is_valid(value: object) -> bool:
    try:
        return type(value) is RunIdentity and _nonnegative_int(value.generation) and type(value.fingerprint) is str
    except Exception:
        return False


def executor_closed_is_valid(value: object, expected: RunIdentity) -> bool:
    """Validate one exact cleanup receipt before it can release lifecycle owners."""

    try:
        return (type(value) is ExecutorClosed and value.run_identity is expected
                and type(value.cleanup_status) is CleanupStatus and value.cleanup_status is not CleanupStatus.NOT_STARTED
                and (value.primary is None or detached_diagnostic_is_valid(value.primary))
                and type(value.cleanup_failures) is tuple
                and all(detached_diagnostic_is_valid(item) and bool(item.operation) for item in value.cleanup_failures))
    except Exception:
        return False


def _nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0

def _applied(value: object, phase: RunPhase) -> bool:
    return lifecycle_result_is_valid(value) and value.status is LifecycleStatus.APPLIED and value.phase is phase and value.error is None

_ANY = object()


def lifecycle_is_applied(value: object, phase: RunPhase, request_id: object = _ANY,
                         run_identity: object = _ANY) -> bool:
    return _applied(value, phase) and (request_id is _ANY or value.request_id is request_id) and (run_identity is _ANY or value.run_identity is run_identity)


def lifecycle_promotion_is_valid(value: object, request_id: RequestId, expected: RunIdentity) -> bool:
    return (_applied(value, RunPhase.STARTING) and value.request_id is request_id
            and run_identity_is_valid(value.run_identity)
            and value.run_identity.generation == expected.generation
            and value.run_identity.fingerprint == expected.fingerprint)


def executor_event_is_valid(value: object, expected: RunIdentity) -> bool:
    """Validate all executor event fields before lifecycle mutation."""

    try:
        if type(value) is ExecutorAccepted:
            return value.run_identity is expected
        return (type(value) is ExecutorStartFailed and value.run_identity is expected
                and type(value.cleanup_status) is CleanupStatus and type(value.error) is LifecycleError)
    except Exception:
        return False

def source_capture_error(value: object, request_id: RequestId, source: SourceSelection,
                         previous: SourceCapture | None, minimum_source_epoch: int) -> str | None:
    """Return a value-only source-capture contract violation, if any."""

    if type(value) is not SourceCapture:
        return "SourcePort.capture() returned an unknown result"
    try:
        captured_request, captured_epoch, captured_source, choices = (
            value.request_id, value.source_epoch, value.source, value.gi_motor_choices)
    except Exception:
        return "SourcePort.capture() returned an invalid result"
    if not request_id_is_valid(captured_request) or captured_request is not request_id:
        return "source capture does not match the request-qualified intent source"
    if not _nonnegative_int(captured_epoch):
        return "source capture epoch must be a non-negative integer"
    if not _nonnegative_int(minimum_source_epoch):
        return "source capture minimum epoch is invalid"
    if not _source_matches(captured_source, source):
        return "source capture does not match the request-qualified intent source"
    if not _choices_are_frozen(choices):
        return "source capture GI motor choices must be None or a tuple of strings"
    if captured_epoch < minimum_source_epoch:
        return "source capture precedes the required observation epoch"
    if previous is not None:
        if type(previous) is not SourceCapture or not _nonnegative_int(previous.source_epoch):
            return "previous source capture is invalid"
        if captured_epoch < previous.source_epoch:
            return "source capture epoch regressed within the request"
        if not _source_matches(previous.source, source):
            return "source capture does not match the request-qualified intent source"
    return None

def _source_matches(left: object, right: object) -> bool:
    if type(left) not in {SourceSpec, DirectorySourceSpec} or type(right) is not type(left):
        return False
    try:
        return (left == right) is True
    except Exception:
        return False

def _choices_are_frozen(value: object) -> bool:
    return value is None or (type(value) is tuple and all(type(choice) is str and bool(choice) for choice in value))

__all__ = ["ExceptionDetail", "RecoveryFailure", "StartCapture", "StartClosed", "StartFailed",
           "StartFailureKind", "StartLaunched", "StartRecaptureCause", "StartRecaptureRequired",
           "StartRefusal", "StartRefused", "StartRejected", "StartRejection", "commit_result_is_valid",
           "exception_detail", "executor_closed_is_valid", "executor_event_is_valid", "freeze_configuration", "freeze_result_is_valid", "lifecycle_is_applied",
           "inert_lifecycle_rejection", "invalid_result",
           "lifecycle_promotion_is_valid", "lifecycle_result_is_valid", "recapture_result_is_valid",
           "request_id_is_valid", "run_identity_is_valid", "snapshot_is_valid", "source_capture_error"]
