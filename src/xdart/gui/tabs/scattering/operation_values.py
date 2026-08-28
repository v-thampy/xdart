"""Detached immutable values for the page-owned operation slot."""

from __future__ import annotations

from dataclasses import dataclass, is_dataclass
from enum import Enum

from .events import CleanupStatus

def _invalid(condition: bool, message: str) -> None:
    if condition: raise ValueError(message)

@dataclass(frozen=True, slots=True, order=True)
class OperationIdentity:
    serial: int

    def __post_init__(self) -> None:
        _invalid(
            type(self.serial) is not int or self.serial < 1,
            "operation identity is invalid",
        )
@dataclass(frozen=True, slots=True)
class OperationContextStamp:
    intent_revision: int
    context_token: str | None = None
    display_generation: int | None = None

    def __post_init__(self) -> None:
        selected = self.context_token is not None
        _invalid(
            type(self.intent_revision) is not int
            or self.intent_revision < 0
            or selected != (self.display_generation is not None)
            or (
                selected
                and (
                    type(self.context_token) is not str
                    or not self.context_token
                    or type(self.display_generation) is not int
                    or self.display_generation < 1
                )
            ),
            "operation context stamp is invalid",
        )
@dataclass(frozen=True, slots=True)
class OperationProgress:
    identity: OperationIdentity
    stage: str
    completed: int
    total: int
    revision: int

    def __post_init__(self) -> None:
        _invalid(
            type(self.identity) is not OperationIdentity
            or type(self.stage) is not str
            or not self.stage
            or type(self.completed) is not int
            or type(self.total) is not int
            or not 0 <= self.completed <= self.total
            or type(self.revision) is not int
            or self.revision < 1,
            "operation progress is invalid",
        )
        self.identity.__post_init__()


@dataclass(frozen=True, slots=True)
class OperationPending:
    identity: OperationIdentity
    revision: int
    phase: str
    diagnostic: str

    def __post_init__(self) -> None:
        _invalid(
            type(self.identity) is not OperationIdentity
            or type(self.revision) is not int or self.revision < 1
            or type(self.phase) is not str or not self.phase
            or type(self.diagnostic) is not str or not self.diagnostic,
            "operation pending value is invalid",
        )
        self.identity.__post_init__()
class OperationTerminalStatus(str, Enum):
    RETURNED = "returned"
    CANCELLED = "cancelled"
    FAILED = "failed"
@dataclass(frozen=True, slots=True)
class OperationTerminal:
    identity: OperationIdentity
    status: OperationTerminalStatus
    diagnostic: str = ""
    payload: object | None = None

    def __post_init__(self) -> None:
        _invalid(
            type(self.identity) is not OperationIdentity
            or type(self.status) is not OperationTerminalStatus
            or type(self.diagnostic) is not str
            or (
                self.status is OperationTerminalStatus.FAILED
                and not self.diagnostic
            )
            or (
                self.status is not OperationTerminalStatus.FAILED
                and bool(self.diagnostic)
            )
            or (self.payload is not None and (not is_dataclass(self.payload) or isinstance(self.payload, type) or not vars(type(self.payload))["__dataclass_params__"].frozen)),
            "operation terminal is invalid",
        )
        self.identity.__post_init__()
@dataclass(frozen=True, slots=True)
class OperationUpdate:
    identity: OperationIdentity
    progress: OperationProgress | None = None
    pending: OperationPending | None = None
    terminal: OperationTerminal | None = None
    stale: bool = False

    def __post_init__(self) -> None:
        _invalid(
            type(self.identity) is not OperationIdentity
            or (self.progress is None and self.pending is None
                and self.terminal is None)
            or (self.pending is not None and self.terminal is not None)
            or type(self.stale) is not bool
            or (
                self.progress is not None
                and (
                    type(self.progress) is not OperationProgress
                    or self.progress.identity is not self.identity
                )
            )
            or (
                self.pending is not None
                and (
                    type(self.pending) is not OperationPending
                    or self.pending.identity is not self.identity
                )
            )
            or (
                self.terminal is not None
                and (
                    type(self.terminal) is not OperationTerminal
                    or self.terminal.identity is not self.identity
                )
            ),
            "operation update is invalid",
        )
        self.identity.__post_init__()
        if self.progress is not None:
            self.progress.__post_init__()
        if self.pending is not None:
            self.pending.__post_init__()
        if self.terminal is not None:
            self.terminal.__post_init__()
@dataclass(frozen=True, slots=True)
class OperationCleanupReceipt:
    identity: OperationIdentity | None
    cleanup_status: CleanupStatus
    cancel_accepted: bool = False
    worker_identity: int | None = None
    terminal: OperationTerminal | None = None
    stale: bool = False

    def __post_init__(self) -> None:
        active = self.identity is not None
        pending = self.cleanup_status is CleanupStatus.CLEANUP_PENDING
        _invalid(
            (active and type(self.identity) is not OperationIdentity)
            or type(self.cleanup_status) is not CleanupStatus
            or type(self.cancel_accepted) is not bool
            or type(self.stale) is not bool
            or (
                self.worker_identity is not None
                and (
                    type(self.worker_identity) is not int
                    or self.worker_identity < 1
                )
            )
            or (
                self.terminal is not None
                and (
                    type(self.terminal) is not OperationTerminal
                    or self.terminal.identity is not self.identity
                )
            )
            or self.cleanup_status not in (
                CleanupStatus.CLEANED, CleanupStatus.CLEANUP_PENDING
            )
            or (pending and (not active or self.worker_identity is None))
            or (pending and self.terminal is not None)
            or (
                not pending
                and active
                and (self.worker_identity is None or self.terminal is None)
            )
            or (
                not active
                and (
                    self.worker_identity is not None
                    or self.terminal is not None
                    or self.cancel_accepted
                    or self.stale
                )
            ),
            "operation cleanup receipt is invalid",
        )
        if self.identity is not None:
            self.identity.__post_init__()
        if self.terminal is not None:
            self.terminal.__post_init__()
__all__ = [
    "OperationCleanupReceipt", "OperationContextStamp", "OperationIdentity",
    "OperationPending", "OperationProgress", "OperationTerminal", "OperationTerminalStatus",
    "OperationUpdate",
]
