"""Scalar lifecycle identities, events, and outcomes for E0a."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from xrd_tools.session.run_configuration import FrozenRunConfiguration

from .state_machine import RunPhase


@dataclass(frozen=True, slots=True, order=True)
class RequestId:
    value: int

    def __post_init__(self) -> None:
        if self.value < 1:
            raise ValueError("request ids start at 1")


@dataclass(frozen=True, slots=True)
class RunIdentity:
    generation: int
    fingerprint: str

    @classmethod
    def from_configuration(cls, value: FrozenRunConfiguration) -> "RunIdentity":
        generation, fingerprint = value.identity
        return cls(generation=generation, fingerprint=fingerprint)


class LifecycleError(str, Enum):
    POLICY_REFUSED = "policy_refused"
    EXECUTOR_START_FAILED = "executor_start_failed"
    ILLEGAL_TRANSITION = "illegal_transition"
    SUPERSEDED = "superseded"


class CleanupStatus(str, Enum):
    NOT_STARTED = "not_started"
    CLEANED = "cleaned"
    CLEANUP_PENDING = "cleanup_pending"


@dataclass(frozen=True, slots=True)
class DetachedDiagnostic:
    """A total, Qt-free projection of one executor or cleanup exception."""

    type_module: str
    type_qualname: str
    message: str
    operation: str = ""


def detached_diagnostic_is_valid(value: object) -> bool:
    try:
        return (type(value) is DetachedDiagnostic and type(value.type_module) is str
                and type(value.type_qualname) is str and type(value.message) is str
                and type(value.operation) is str)
    except Exception:
        return False


def detached_exception_strings(value: BaseException) -> tuple[str, str, str]:
    """Detach module, qualified name, and message independently and totally."""

    kind = type(value)
    return (_exception_text(kind, "__module__", "unknown"),
            _exception_text(kind, "__qualname__", "Exception"),
            _exception_text(value, None, "<unprintable>"))


def detach_exception(
    value: BaseException, operation: str = "",
) -> DetachedDiagnostic:
    return DetachedDiagnostic(*detached_exception_strings(value), operation)


def _exception_text(value: object, attribute: str | None, fallback: str) -> str:
    try:
        raw = str(value) if attribute is None else getattr(value, attribute)
        return raw if type(raw) is str else str(raw)
    except BaseException:
        return fallback


@dataclass(frozen=True, slots=True)
class PreflightRefused:
    request_id: RequestId
    error: LifecycleError = LifecycleError.POLICY_REFUSED


@dataclass(frozen=True, slots=True)
class PreflightAccepted:
    request_id: RequestId
    configuration: FrozenRunConfiguration


@dataclass(frozen=True, slots=True)
class ExecutorAccepted:
    run_identity: RunIdentity


@dataclass(frozen=True, slots=True)
class ExecutorStartFailed:
    """A failed start consumes exactly this frozen attempt identity."""

    run_identity: RunIdentity
    cleanup_status: CleanupStatus
    error: LifecycleError = LifecycleError.EXECUTOR_START_FAILED
    primary: DetachedDiagnostic | None = None
    cleanup_failures: tuple[DetachedDiagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class ExecutorClosed:
    """Exact cleanup truth for one executor-owned run shell."""

    run_identity: RunIdentity
    cleanup_status: CleanupStatus
    primary: DetachedDiagnostic | None = None
    cleanup_failures: tuple[DetachedDiagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class OwnersClosed:
    run_identity: RunIdentity


@dataclass(frozen=True, slots=True)
class StopRequested:
    run_identity: RunIdentity


@dataclass(frozen=True, slots=True)
class PauseRequested:
    run_identity: RunIdentity


@dataclass(frozen=True, slots=True)
class DurablePaused:
    run_identity: RunIdentity
    durable_generation: int


@dataclass(frozen=True, slots=True)
class PauseFailed:
    run_identity: RunIdentity
    diagnostic: DetachedDiagnostic
    compensation: DetachedDiagnostic | None = None


@dataclass(frozen=True, slots=True)
class ResumeRequested:
    run_identity: RunIdentity


@dataclass(frozen=True, slots=True)
class ResumeFailed:
    run_identity: RunIdentity
    diagnostic: DetachedDiagnostic
    compensation: DetachedDiagnostic | None = None


@dataclass(frozen=True, slots=True)
class Resumed:
    run_identity: RunIdentity


@dataclass(frozen=True, slots=True)
class ExecutionEnded:
    run_identity: RunIdentity


@dataclass(frozen=True, slots=True)
class DurableFinal:
    run_identity: RunIdentity


@dataclass(frozen=True, slots=True)
class FatalExecution:
    run_identity: RunIdentity


class LifecycleStatus(str, Enum):
    APPLIED = "applied"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


@dataclass(frozen=True, slots=True)
class LifecycleResult:
    status: LifecycleStatus
    phase: RunPhase
    request_id: RequestId | None = None
    run_identity: RunIdentity | None = None
    error: LifecycleError | None = None


__all__ = [
    "CleanupStatus",
    "DetachedDiagnostic",
    "detached_diagnostic_is_valid",
    "detached_exception_strings",
    "detach_exception",
    "ExecutorClosed",
    "DurableFinal",
    "ExecutorAccepted",
    "ExecutorStartFailed",
    "ExecutionEnded",
    "FatalExecution",
    "LifecycleError",
    "LifecycleResult",
    "LifecycleStatus",
    "OwnersClosed",
    "PauseRequested",
    "PauseFailed",
    "PreflightAccepted",
    "PreflightRefused",
    "RequestId",
    "ResumeRequested",
    "ResumeFailed",
    "Resumed",
    "RunIdentity",
    "StopRequested",
    "DurablePaused",
]
