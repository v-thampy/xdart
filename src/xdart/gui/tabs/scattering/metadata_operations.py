"""Qt-free identity owner for deferred metadata operations.

The composition page owns dialogs and paints.  This owner keeps only opaque
dialog identities plus the exact request, context, candidate, and worker
identity needed to decide whether a metadata result is still current.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from xrd_tools.analysis.scan_operations import (
    MetadataTablePlan,
    MetadataTableRequalificationPlan,
    MetadataTableResult,
)

from .events import CleanupStatus
from .operation_values import (
    OperationCleanupReceipt,
    OperationContextStamp,
    OperationIdentity,
)


_TARGETS = frozenset({"metadata", "scan_roi"})


class MetadataDisposition(str, Enum):
    """Admission state for the current latest-only metadata request."""

    READY = "ready"
    TRANSIENT = "transient"
    PERMANENT = "permanent"


class MetadataLifecycle(str, Enum):
    """Absorbing workspace lifetime for metadata identity state."""

    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class MetadataDialogIdentity:
    """Opaque identity for one page-owned dialog lifetime."""

    serial: int
    target: str
    generation: int

    def __post_init__(self) -> None:
        if (
            type(self.serial) is not int
            or self.serial < 1
            or self.target not in _TARGETS
            or type(self.generation) is not int
            or self.generation < 0
        ):
            raise ValueError("metadata dialog identity is invalid")


@dataclass(frozen=True, slots=True)
class MetadataRequest:
    """One exact metadata or requalification request."""

    serial: int
    plan: MetadataTablePlan | MetadataTableRequalificationPlan
    request: tuple[object, ...]
    dialog: MetadataDialogIdentity
    candidate: MetadataTableResult | None = None

    def __post_init__(self) -> None:
        requalification = self.candidate is not None
        if (
            type(self.serial) is not int
            or self.serial < 1
            or type(self.plan) not in {
                MetadataTablePlan,
                MetadataTableRequalificationPlan,
            }
            or type(self.request) is not tuple
            or type(self.dialog) is not MetadataDialogIdentity
            or (
                requalification
                != (type(self.plan) is MetadataTableRequalificationPlan)
            )
            or (
                self.candidate is not None
                and type(self.candidate) is not MetadataTableResult
            )
        ):
            raise ValueError("metadata request is invalid")
        self.dialog.__post_init__()

    @property
    def target(self) -> str:
        return self.dialog.target

    @property
    def generation(self) -> int:
        return self.dialog.generation

    @property
    def kind(self) -> str:
        return (
            "metadata_requalification"
            if self.candidate is not None
            else "metadata"
        )


@dataclass(frozen=True, slots=True)
class MetadataProcess:
    """Active worker identity bound to the request that launched it."""

    identity: OperationIdentity
    request: MetadataRequest
    context: OperationContextStamp

    def __post_init__(self) -> None:
        if (
            type(self.identity) is not OperationIdentity
            or type(self.request) is not MetadataRequest
            or type(self.context) is not OperationContextStamp
        ):
            raise ValueError("metadata process is invalid")
        self.identity.__post_init__()
        self.request.__post_init__()
        self.context.__post_init__()


class MetadataOperationOwner:
    """Own latest-only metadata request and process identity without Qt."""

    def __init__(self) -> None:
        self._dialog_serial = 0
        self._request_serial = 0
        self._generations = {target: -1 for target in _TARGETS}
        self._dialogs: dict[str, MetadataDialogIdentity] = {}
        self._deferred: MetadataRequest | None = None
        self._active: MetadataProcess | None = None
        self._lifecycle = MetadataLifecycle.OPEN
        self._cleanup_identity: OperationIdentity | None = None
        self._cleanup_pending = False

    @property
    def lifecycle(self) -> MetadataLifecycle:
        return self._lifecycle

    @property
    def cleanup_identity(self) -> OperationIdentity | None:
        return self._cleanup_identity

    @property
    def deferred(self) -> MetadataRequest | None:
        return self._deferred

    @property
    def active(self) -> MetadataProcess | None:
        return self._active

    @property
    def active_identity(self) -> OperationIdentity | None:
        active = self._active
        return None if active is None else active.identity

    @property
    def busy(self) -> bool:
        return self._active is not None

    @property
    def polling_needed(self) -> bool:
        return self._deferred is not None

    def dialog_identity(self, target: str) -> MetadataDialogIdentity | None:
        return self._dialogs.get(target) if target in _TARGETS else None

    def open_dialog(self, target: str) -> MetadataDialogIdentity | None:
        if (
            self._lifecycle is not MetadataLifecycle.OPEN
            or target not in _TARGETS
        ):
            return None
        current = self._dialogs.get(target)
        if current is not None:
            return current
        self._dialog_serial += 1
        generation = self._generations[target] + 1
        self._generations[target] = generation
        identity = MetadataDialogIdentity(
            self._dialog_serial, target, generation,
        )
        self._dialogs[target] = identity
        return identity

    def close_dialog(
        self, identity: object,
    ) -> OperationIdentity | None:
        if type(identity) is not MetadataDialogIdentity:
            return None
        current = self._dialogs.get(identity.target)
        if current is not identity:
            return None
        del self._dialogs[identity.target]
        if (
            self._deferred is not None
            and self._deferred.dialog is identity
        ):
            self._deferred = None
        active = self._active
        return (
            active.identity
            if active is not None and active.request.dialog is identity
            else None
        )

    def capture(
        self,
        plan: MetadataTablePlan | MetadataTableRequalificationPlan,
        request: tuple[object, ...],
        *,
        target: str,
        dialog: object,
        candidate: MetadataTableResult | None = None,
    ) -> MetadataRequest | None:
        if (
            self._lifecycle is not MetadataLifecycle.OPEN
            or target not in _TARGETS
            or type(dialog) is not MetadataDialogIdentity
            or self._dialogs.get(target) is not dialog
            or dialog.target != target
            or type(plan) not in {
                MetadataTablePlan,
                MetadataTableRequalificationPlan,
            }
            or ((candidate is not None)
                != (type(plan) is MetadataTableRequalificationPlan))
            or (
                candidate is not None
                and type(candidate) is not MetadataTableResult
            )
            or type(request) is not tuple
        ):
            return None
        self._request_serial += 1
        captured = MetadataRequest(
            self._request_serial,
            plan,
            request,
            dialog,
            candidate,
        )
        self._deferred = captured
        return captured

    def classify(
        self,
        captured: object,
        *,
        current_request: object,
        blocked: bool,
        start_allowed: bool,
        closing: bool,
    ) -> MetadataDisposition:
        if (
            type(captured) is not MetadataRequest
            or self._deferred is not captured
            or self._lifecycle is not MetadataLifecycle.OPEN
            or closing is not False
            or self._dialogs.get(captured.target) is not captured.dialog
            or current_request != captured.request
            or type(blocked) is not bool
            or type(start_allowed) is not bool
        ):
            return MetadataDisposition.PERMANENT
        if blocked:
            return MetadataDisposition.TRANSIENT
        return (
            MetadataDisposition.READY
            if start_allowed
            else MetadataDisposition.PERMANENT
        )

    def drop(self, captured: object) -> bool:
        if self._deferred is not captured:
            return False
        self._deferred = None
        return True

    def start(
        self,
        captured: object,
        identity: object,
        context: object,
    ) -> MetadataProcess | None:
        if (
            type(captured) is not MetadataRequest
            or self._deferred is not captured
            or self._active is not None
            or type(identity) is not OperationIdentity
            or type(context) is not OperationContextStamp
            or self._dialogs.get(captured.target) is not captured.dialog
        ):
            return None
        try:
            context.__post_init__()
        except (TypeError, ValueError):
            return None
        process = MetadataProcess(identity, captured, context)
        self._active = process
        self._deferred = None
        return process

    def finish(self, identity: object) -> MetadataProcess | None:
        active = self._active
        if active is None or active.identity is not identity:
            return None
        self._active = None
        return active

    def lost(self, identity: object) -> MetadataProcess | None:
        process = self.finish(identity)
        if (
            process is not None
            and self._lifecycle is MetadataLifecycle.CLOSING
            and self._cleanup_identity is identity
        ):
            self._cleanup_pending = False
            self._lifecycle = MetadataLifecycle.CLOSED
        return process

    def has_newer_request(self, process: object) -> bool:
        return bool(
            type(process) is MetadataProcess
            and self._active is process
            and self._deferred is not None
        )

    def process_is_current(
        self,
        process: object,
        *,
        current_request: object,
        current_context: object,
    ) -> bool:
        return bool(
            type(process) is MetadataProcess
            and self._active is process
            and self._dialogs.get(process.request.target)
            is process.request.dialog
            and current_request == process.request.request
            and type(current_context) is OperationContextStamp
            and current_context == process.context
        )

    def begin_close(self) -> OperationIdentity | None:
        if self._lifecycle is not MetadataLifecycle.OPEN:
            return self._cleanup_identity
        self._lifecycle = MetadataLifecycle.CLOSING
        self._deferred = None
        self._dialogs.clear()
        self._cleanup_identity = self.active_identity
        if self._cleanup_identity is None:
            self._lifecycle = MetadataLifecycle.CLOSED
        return self._cleanup_identity

    def consume_close_receipt(self, receipt: object) -> bool:
        if type(receipt) is not OperationCleanupReceipt:
            return False
        try:
            receipt.__post_init__()
        except (TypeError, ValueError):
            return False
        identity = self._cleanup_identity
        if (
            self._lifecycle is not MetadataLifecycle.CLOSING
            or identity is None
            or receipt.identity is not identity
        ):
            return False
        if receipt.cleanup_status is CleanupStatus.CLEANUP_PENDING:
            self._cleanup_pending = True
            return True
        if receipt.cleanup_status is not CleanupStatus.CLEANED:
            return False
        self._cleanup_pending = False
        self._active = None
        self._lifecycle = MetadataLifecycle.CLOSED
        return True


__all__ = [
    "MetadataDialogIdentity",
    "MetadataDisposition",
    "MetadataLifecycle",
    "MetadataOperationOwner",
    "MetadataProcess",
    "MetadataRequest",
]
