"""Small, standard-library-only owner for one page operation worker.

Science adapters inject the runner.  This module owns only worker lifetime,
exact request identity, cooperative cancellation, bounded progress, terminal
delivery, and retryable page close.  It deliberately imports no Qt or science
package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from threading import Event, Lock, Thread, TIMEOUT_MAX
from typing import Callable

from .values import CloseReceipt, PageCleanup


class OperationTerminalStatus(str, Enum):
    RETURNED = "returned"
    CANCELLED = "cancelled"
    FAILED = "failed"


class OperationCancelled(InterruptedError):
    """Cooperative runner signal that cancellation won settlement."""


@dataclass(eq=False, frozen=True, slots=True)
class OperationIdentity:
    """Opaque authority for one exact request and worker lifetime."""

    serial: int
    request: object = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.serial) is not int or self.serial < 1:
            raise TypeError("operation serial must be a positive exact integer")


@dataclass(frozen=True, slots=True)
class OperationProgress:
    identity: OperationIdentity
    revision: int
    stage: str
    completed: int
    total: int

    def __post_init__(self) -> None:
        if (
            type(self.identity) is not OperationIdentity
            or type(self.revision) is not int
            or self.revision < 1
            or type(self.stage) is not str
            or not self.stage
            or type(self.completed) is not int
            or type(self.total) is not int
            or self.total < 1
            or self.completed < 0
            or self.completed > self.total
        ):
            raise TypeError("operation progress is invalid")


@dataclass(frozen=True, slots=True)
class OperationTerminal:
    identity: OperationIdentity
    status: OperationTerminalStatus
    payload: object | None = field(default=None, repr=False)
    failure_module: str = ""
    failure_type: str = ""
    failure_message: str = ""

    def __post_init__(self) -> None:
        if (
            type(self.identity) is not OperationIdentity
            or type(self.status) is not OperationTerminalStatus
            or any(
                type(value) is not str
                for value in (
                    self.failure_module,
                    self.failure_type,
                    self.failure_message,
                )
            )
            or (
                self.status is OperationTerminalStatus.FAILED
                and (not self.failure_type or self.payload is not None)
            )
            or (
                self.status is not OperationTerminalStatus.FAILED
                and any(
                    (self.failure_module, self.failure_type, self.failure_message)
                )
            )
        ):
            raise TypeError("operation terminal is invalid")


@dataclass(frozen=True, slots=True)
class OperationUpdate:
    identity: OperationIdentity
    progress: OperationProgress | None = None
    terminal: OperationTerminal | None = None

    def __post_init__(self) -> None:
        if (
            type(self.identity) is not OperationIdentity
            or (self.progress is None) == (self.terminal is None)
            or (
                self.progress is not None
                and (
                    type(self.progress) is not OperationProgress
                    or self.progress.identity is not self.identity
                )
            )
            or (
                self.terminal is not None
                and (
                    type(self.terminal) is not OperationTerminal
                    or self.terminal.identity is not self.identity
                )
            )
        ):
            raise TypeError("operation update is invalid")


Runner = Callable[[object, Event, Callable[[str, int, int], bool]], object]


def _failure_facts(error: BaseException) -> tuple[str, str, str]:
    kind = type(error)
    try:
        message = str(error)
    except BaseException:
        message = "exception message unavailable"
    return kind.__module__, kind.__qualname__, message[:4096]


class SingleWorkerOwner:
    """Own one non-daemon, joinable worker and latest-only progress."""

    def __init__(self, runner: Runner, *, join_timeout: float = 0.0) -> None:
        if not callable(runner):
            raise TypeError("runner must be callable")
        if (
            type(join_timeout) not in {int, float}
            or isinstance(join_timeout, bool)
            or join_timeout < 0
            or not math.isfinite(float(join_timeout))
            or float(join_timeout) > TIMEOUT_MAX
        ):
            raise TypeError("join_timeout must be a nonnegative number")
        self._runner = runner
        self._join_timeout = float(join_timeout)
        self._lock = Lock()
        self._next_serial = 1
        self._identity: OperationIdentity | None = None
        self._worker: Thread | None = None
        self._starting = False
        self._cancel: Event | None = None
        self._cancel_commanded = False
        self._progress: OperationProgress | None = None
        self._progress_revision = 0
        self._progress_delivered_revision = 0
        self._progress_total: int | None = None
        self._terminal: OperationTerminal | None = None
        self._closing = False
        self._clean_receipt: CloseReceipt | None = None

    def __copy__(self):
        raise TypeError("single-worker operation owner is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("single-worker operation owner is not copyable")

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._identity is not None

    @property
    def current_identity(self) -> OperationIdentity | None:
        with self._lock:
            return self._identity

    def begin(self, request: object) -> OperationIdentity | None:
        with self._lock:
            if self._closing or self._identity is not None:
                return None
            identity = OperationIdentity(self._next_serial, request)
            self._next_serial += 1
            cancel = Event()
            worker = Thread(
                target=self._run,
                args=(identity, request, cancel),
                name=f"xdart-page-operation-{identity.serial}",
                daemon=False,
            )
            self._identity = identity
            self._worker = worker
            self._starting = True
            self._cancel = cancel
            self._cancel_commanded = False
            self._progress = None
            self._progress_revision = 0
            self._progress_delivered_revision = 0
            self._progress_total = None
            self._terminal = None
        try:
            worker.start()
        except BaseException as error:
            facts = _failure_facts(error)
            with self._lock:
                if self._identity is identity:
                    self._starting = False
                    if worker.ident is None:
                        self._worker = None
                        self._terminal = OperationTerminal(
                            identity,
                            OperationTerminalStatus.FAILED,
                            failure_module=facts[0],
                            failure_type=facts[1],
                            failure_message=facts[2],
                        )
            return identity
        with self._lock:
            if self._identity is identity:
                self._starting = False
        return identity

    def _publish(
        self,
        identity: OperationIdentity,
        stage: str,
        completed: int,
        total: int,
    ) -> bool:
        if (
            type(stage) is not str
            or not stage
            or type(completed) is not int
            or type(total) is not int
            or total < 1
            or completed < 0
            or completed > total
        ):
            return False
        with self._lock:
            if (
                self._identity is not identity
                or self._closing
                or self._terminal is not None
                or (
                    self._progress_total is not None
                    and self._progress_total != total
                )
                or (
                    self._progress is not None
                    and completed < self._progress.completed
                )
            ):
                return False
            self._progress_total = total
            self._progress_revision += 1
            self._progress = OperationProgress(
                identity,
                self._progress_revision,
                stage,
                completed,
                total,
            )
            return True

    def _run(
        self,
        identity: OperationIdentity,
        request: object,
        cancel: Event,
    ) -> None:
        try:
            result = self._runner(
                request,
                cancel,
                lambda stage, completed, total: self._publish(
                    identity, stage, completed, total
                ),
            )
        except OperationCancelled:
            terminal = OperationTerminal(
                identity, OperationTerminalStatus.CANCELLED
            )
        except BaseException as error:
            facts = _failure_facts(error)
            terminal = OperationTerminal(
                identity,
                OperationTerminalStatus.FAILED,
                failure_module=facts[0],
                failure_type=facts[1],
                failure_message=facts[2],
            )
        else:
            terminal = OperationTerminal(
                identity,
                OperationTerminalStatus.RETURNED,
                payload=result,
            )
        with self._lock:
            if self._identity is identity:
                self._terminal = terminal

    def poll(self, identity: object) -> OperationUpdate | None:
        worker: Thread | None = None
        with self._lock:
            if self._closing or self._identity is not identity:
                return None
            if self._terminal is not None:
                worker = self._worker
                if worker is not None and worker.is_alive():
                    return None
                terminal = self._terminal
                self._retire_locked()
                update = OperationUpdate(identity, terminal=terminal)
            elif (
                self._progress is not None
                and self._progress_revision > self._progress_delivered_revision
            ):
                self._progress_delivered_revision = self._progress_revision
                return OperationUpdate(identity, progress=self._progress)
            else:
                return None
        if worker is not None:
            worker.join(0)
        return update

    def cancel(self, identity: object) -> bool:
        with self._lock:
            if (
                self._closing
                or self._identity is not identity
                or self._cancel is None
                or self._cancel_commanded
                or self._terminal is not None
            ):
                return False
            self._cancel_commanded = True
            self._cancel.set()
            return True

    def _retire_locked(self) -> None:
        self._identity = None
        self._worker = None
        self._starting = False
        self._cancel = None
        self._cancel_commanded = False
        self._progress = None
        self._progress_revision = 0
        self._progress_delivered_revision = 0
        self._progress_total = None
        self._terminal = None

    def close(self) -> CloseReceipt:
        with self._lock:
            if self._clean_receipt is not None:
                return self._clean_receipt
            self._closing = True
            worker = self._worker
            cancel = self._cancel
            if cancel is not None and not self._cancel_commanded:
                self._cancel_commanded = True
                cancel.set()
            if self._starting:
                serial = 0 if self._identity is None else self._identity.serial
                return CloseReceipt(
                    PageCleanup.PENDING,
                    f"operation-worker-{serial}",
                )
        if worker is not None:
            try:
                worker.join(self._join_timeout)
            except RuntimeError:
                # ``Thread.start`` itself failed; the stored terminal is already
                # sufficient and there is no live worker to join.
                pass
        with self._lock:
            if self._clean_receipt is not None:
                return self._clean_receipt
            worker = self._worker
            if worker is not None and worker.is_alive():
                serial = 0 if self._identity is None else self._identity.serial
                return CloseReceipt(
                    PageCleanup.PENDING,
                    f"operation-worker-{serial}",
                )
            self._retire_locked()
            self._clean_receipt = CloseReceipt(PageCleanup.CLEAN)
            return self._clean_receipt


__all__ = [
    "OperationCancelled",
    "OperationIdentity",
    "OperationProgress",
    "OperationTerminal",
    "OperationTerminalStatus",
    "OperationUpdate",
    "SingleWorkerOwner",
]
