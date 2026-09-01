"""Qt-free asynchronous owner for standalone RSM preflight and execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from threading import Lock

from xdart.gui.pages.operation_owner import (
    OperationCancelled,
    OperationIdentity,
    OperationProgress,
    OperationTerminalStatus,
    SingleWorkerOwner,
)
from xdart.gui.pages.values import CloseReceipt, PageCleanup
from xrd_tools.analysis.rsm_operation import (
    RSMOperationCleanupPending,
    RSMOperationExecution,
    RSMOperationResult,
    RSMOperationVerificationError,
)

from .rsm_values import (
    RSMToolForm,
    RSMToolPreflight,
    RSMToolPreflightRefused,
    prepare_rsm_tool,
)


class RSMOwnerAction(str, Enum):
    PREFLIGHT = "preflight"
    RUN = "run"
    RETRY_CLEANUP = "retry-cleanup"
    RETRY_VERIFICATION = "retry-verification"


class RSMOwnerOutcomeKind(str, Enum):
    PREFLIGHT_READY = "preflight-ready"
    PREFLIGHT_REFUSED = "preflight-refused"
    RESULT = "result"
    CLEANUP_PENDING = "cleanup-pending"
    VERIFICATION_PENDING = "verification-pending"


class RSMOwnerFinalization(str, Enum):
    NONE = "none"
    CLEANUP_PENDING = "cleanup-pending"
    VERIFICATION_PENDING = "verification-pending"


@dataclass(frozen=True, slots=True)
class RSMOwnerOutcome:
    kind: RSMOwnerOutcomeKind
    preflight: RSMToolPreflight | None = None
    result: RSMOperationResult | None = None
    refusal_code: str = ""
    refusal_message: str = ""
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.kind) is not RSMOwnerOutcomeKind
            or (
                self.preflight is not None
                and type(self.preflight) is not RSMToolPreflight
            )
            or (
                self.result is not None
                and type(self.result) is not RSMOperationResult
            )
            or type(self.refusal_code) is not str
            or type(self.refusal_message) is not str
            or type(self.diagnostics) is not tuple
            or any(type(item) is not str for item in self.diagnostics)
            or (
                self.kind is RSMOwnerOutcomeKind.PREFLIGHT_READY
                and (
                    self.preflight is None
                    or self.result is not None
                    or self.refusal_code
                    or self.refusal_message
                    or self.diagnostics
                )
            )
            or (
                self.kind is RSMOwnerOutcomeKind.PREFLIGHT_REFUSED
                and (
                    self.preflight is not None
                    or self.result is not None
                    or not self.refusal_code
                )
            )
            or (
                self.kind is RSMOwnerOutcomeKind.RESULT
                and (
                    self.preflight is not None
                    or self.result is None
                    or self.refusal_code
                    or self.refusal_message
                    or self.diagnostics
                )
            )
            or (
                self.kind
                in {
                    RSMOwnerOutcomeKind.CLEANUP_PENDING,
                    RSMOwnerOutcomeKind.VERIFICATION_PENDING,
                }
                and (
                    self.preflight is not None
                    or self.result is not None
                    or self.refusal_code
                    or self.refusal_message
                    or self.diagnostics
                )
            )
        ):
            raise TypeError("RSM owner outcome is invalid")


@dataclass(frozen=True, slots=True)
class RSMOwnerUpdate:
    identity: OperationIdentity
    action: RSMOwnerAction
    progress: OperationProgress | None = None
    terminal_status: OperationTerminalStatus | None = None
    outcome: RSMOwnerOutcome | None = None
    failure_module: str = ""
    failure_type: str = ""
    failure_message: str = ""
    stale: bool = False

    def __post_init__(self) -> None:
        terminal = self.terminal_status is not None
        if (
            type(self.identity) is not OperationIdentity
            or type(self.action) is not RSMOwnerAction
            or (self.progress is None) == (not terminal)
            or (
                self.progress is not None
                and (
                    type(self.progress) is not OperationProgress
                    or self.progress.identity is not self.identity
                )
            )
            or (
                terminal
                and type(self.terminal_status) is not OperationTerminalStatus
            )
            or (
                self.outcome is not None
                and type(self.outcome) is not RSMOwnerOutcome
            )
            or (
                self.outcome is not None
                and self.terminal_status is not OperationTerminalStatus.RETURNED
            )
            or (
                self.terminal_status is OperationTerminalStatus.RETURNED
                and self.outcome is None
            )
            or any(
                type(value) is not str
                for value in (
                    self.failure_module,
                    self.failure_type,
                    self.failure_message,
                )
            )
            or (
                self.terminal_status is OperationTerminalStatus.FAILED
                and not self.failure_type
            )
            or (
                self.terminal_status is not OperationTerminalStatus.FAILED
                and any(
                    (
                        self.failure_module,
                        self.failure_type,
                        self.failure_message,
                    )
                )
            )
            or type(self.stale) is not bool
        ):
            raise TypeError("RSM owner update is invalid")

    @property
    def terminal(self) -> bool:
        return self.terminal_status is not None


@dataclass(frozen=True, slots=True)
class _WorkerCommand:
    action: RSMOwnerAction
    form_fingerprint: str
    form_revision: int
    form: RSMToolForm | None = None
    preflight: RSMToolPreflight | None = None
    execution: RSMOperationExecution | None = None

    def __post_init__(self) -> None:
        if (
            type(self.action) is not RSMOwnerAction
            or type(self.form_fingerprint) is not str
            or not self.form_fingerprint
            or type(self.form_revision) is not int
            or self.form_revision < 1
            or (self.form is not None and type(self.form) is not RSMToolForm)
            or (
                self.preflight is not None
                and type(self.preflight) is not RSMToolPreflight
            )
            or (
                self.execution is not None
                and type(self.execution) is not RSMOperationExecution
            )
        ):
            raise TypeError("RSM worker command is invalid")
        if self.action is RSMOwnerAction.PREFLIGHT:
            valid = (
                self.form is not None
                and self.preflight is None
                and self.execution is None
            )
        else:
            valid = (
                self.form is None
                and self.preflight is not None
                and self.execution is not None
                and self.execution.request is self.preflight.request
                and self.form_fingerprint == self.preflight.form.fingerprint
            )
        if not valid:
            raise TypeError("RSM worker command fields disagree")


PreflightRunner = Callable[..., RSMToolPreflight]
ExecutionFactory = Callable[..., RSMOperationExecution]


class RSMToolOwner:
    """Own one worker, one exact preflight, and retained finalization retries."""

    def __init__(
        self,
        *,
        coordinator=None,
        preflight_runner: PreflightRunner = prepare_rsm_tool,
        execution_factory: ExecutionFactory = RSMOperationExecution,
        join_timeout: float = 0.0,
    ) -> None:
        if not callable(preflight_runner) or not callable(execution_factory):
            raise TypeError("RSM owner adapters must be callable")
        self._coordinator = coordinator
        self._preflight_runner = preflight_runner
        self._execution_factory = execution_factory
        self._worker = SingleWorkerOwner(self._run_command, join_timeout=join_timeout)
        self._lock = Lock()
        self._form: RSMToolForm | None = None
        self._form_revision = 0
        self._prepared: RSMToolPreflight | None = None
        self._prepared_revision: int | None = None
        self._execution: RSMOperationExecution | None = None
        self._execution_preflight: RSMToolPreflight | None = None
        self._execution_form_revision: int | None = None
        self._last_result: RSMOperationResult | None = None
        self._finalization = RSMOwnerFinalization.NONE
        self._active_identity: OperationIdentity | None = None
        self._active_action: RSMOwnerAction | None = None
        self._closing = False
        self._clean_receipt: CloseReceipt | None = None

    def __copy__(self):
        raise TypeError("RSM tool owner is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("RSM tool owner is not copyable")

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._active_identity is not None

    @property
    def form(self) -> RSMToolForm | None:
        with self._lock:
            return self._form

    @property
    def prepared(self) -> RSMToolPreflight | None:
        with self._lock:
            return self._prepared

    @property
    def prepared_stale(self) -> bool:
        with self._lock:
            return self._prepared is not None and not self._prepared.is_current(
                self._form
            )

    @property
    def finalization(self) -> RSMOwnerFinalization:
        with self._lock:
            return self._finalization

    @property
    def last_result(self) -> RSMOperationResult | None:
        with self._lock:
            return self._last_result

    @property
    def current_identity(self) -> OperationIdentity | None:
        with self._lock:
            return self._active_identity

    def set_form(self, form: RSMToolForm) -> bool:
        if type(form) is not RSMToolForm:
            raise TypeError("RSM owner form must be exact RSMToolForm")
        with self._lock:
            if self._closing:
                return False
            changed = self._form is None or self._form.fingerprint != form.fingerprint
            revoked = changed and self._prepared is not None
            self._form = form
            if changed:
                self._form_revision += 1
                self._prepared = None
                self._prepared_revision = None
            return revoked

    def _begin_locked(
        self,
        command: _WorkerCommand,
        *,
        allow_closing: bool = False,
    ) -> OperationIdentity | None:
        if (self._closing and not allow_closing) or self._active_identity is not None:
            return None
        identity = self._worker.begin(command)
        if identity is None:
            return None
        self._active_identity = identity
        self._active_action = command.action
        return identity

    def begin_preflight(self) -> OperationIdentity | None:
        with self._lock:
            if (
                self._form is None
                or self._finalization is not RSMOwnerFinalization.NONE
            ):
                return None
            command = _WorkerCommand(
                RSMOwnerAction.PREFLIGHT,
                self._form.fingerprint,
                self._form_revision,
                form=self._form,
            )
            identity = self._begin_locked(command)
            if identity is not None:
                self._prepared = None
                self._prepared_revision = None
            return identity

    def begin_run(self) -> OperationIdentity | None:
        with self._lock:
            preflight = self._prepared
            if (
                preflight is None
                or not preflight.is_current(self._form)
                or self._prepared_revision != self._form_revision
                or self._finalization is not RSMOwnerFinalization.NONE
            ):
                return None
            execution = self._execution_factory(
                preflight.request,
                coordinator=self._coordinator,
            )
            if (
                type(execution) is not RSMOperationExecution
                or execution.request is not preflight.request
            ):
                raise TypeError(
                    "execution factory must return the exact prepared RSM execution"
                )
            command = _WorkerCommand(
                RSMOwnerAction.RUN,
                preflight.form.fingerprint,
                self._prepared_revision,
                preflight=preflight,
                execution=execution,
            )
            identity = self._begin_locked(command)
            if identity is not None:
                self._execution = execution
                self._execution_preflight = preflight
                self._execution_form_revision = self._prepared_revision
                self._last_result = None
            return identity

    def _retry_command_locked(
        self,
        action: RSMOwnerAction,
        finalization: RSMOwnerFinalization,
        *,
        allow_closing: bool = False,
    ) -> OperationIdentity | None:
        preflight = self._execution_preflight
        execution = self._execution
        form_revision = self._execution_form_revision
        if (
            self._finalization is not finalization
            or preflight is None
            or execution is None
            or form_revision is None
            or execution.request is not preflight.request
        ):
            return None
        return self._begin_locked(
            _WorkerCommand(
                action,
                preflight.form.fingerprint,
                form_revision,
                preflight=preflight,
                execution=execution,
            ),
            allow_closing=allow_closing,
        )

    def begin_retry_cleanup(self) -> OperationIdentity | None:
        with self._lock:
            return self._retry_command_locked(
                RSMOwnerAction.RETRY_CLEANUP,
                RSMOwnerFinalization.CLEANUP_PENDING,
            )

    def begin_retry_verification(self) -> OperationIdentity | None:
        with self._lock:
            return self._retry_command_locked(
                RSMOwnerAction.RETRY_VERIFICATION,
                RSMOwnerFinalization.VERIFICATION_PENDING,
            )

    def cancel(self) -> bool:
        with self._lock:
            identity = self._active_identity
            action = self._active_action
            if action not in {RSMOwnerAction.PREFLIGHT, RSMOwnerAction.RUN}:
                return False
        return identity is not None and self._worker.cancel(identity)

    def _run_command(self, request, cancel, publish):
        if type(request) is not _WorkerCommand:
            raise TypeError("RSM worker requires one exact command")
        if request.action is RSMOwnerAction.PREFLIGHT:
            publish("preflight", 0, 1)
            try:
                preflight = self._preflight_runner(
                    request.form,
                    cancel_token=cancel,
                )
            except RSMToolPreflightRefused as error:
                if error.code == "CANCELLED":
                    raise OperationCancelled from error
                return RSMOwnerOutcome(
                    RSMOwnerOutcomeKind.PREFLIGHT_REFUSED,
                    refusal_code=error.code,
                    refusal_message=str(error),
                    diagnostics=error.diagnostics,
                )
            if type(preflight) is not RSMToolPreflight:
                raise TypeError("preflight adapter returned an invalid value")
            if preflight.form.fingerprint != request.form_fingerprint:
                raise ValueError("preflight adapter changed the form identity")
            publish("preflight", 1, 1)
            return RSMOwnerOutcome(
                RSMOwnerOutcomeKind.PREFLIGHT_READY,
                preflight=preflight,
            )
        execution = request.execution
        if execution is None:
            raise TypeError("RSM execution command lost its owner")
        try:
            if request.action is RSMOwnerAction.RUN:
                result = execution.run(
                    cancel_token=cancel,
                    progress_callback=lambda progress: publish(
                        progress.stage,
                        progress.completed,
                        progress.total,
                    ),
                )
            elif request.action is RSMOwnerAction.RETRY_CLEANUP:
                publish("cleanup", 0, 1)
                result = execution.retry_cleanup()
                publish("cleanup", 1, 1)
            elif request.action is RSMOwnerAction.RETRY_VERIFICATION:
                publish("verification", 0, 1)
                result = execution.retry_verification()
                publish("verification", 1, 1)
            else:  # pragma: no cover
                raise RuntimeError("unknown RSM owner action")
        except RSMOperationCleanupPending as error:
            if error.execution is not execution:
                raise RuntimeError("cleanup exception changed RSM execution owner")
            return RSMOwnerOutcome(RSMOwnerOutcomeKind.CLEANUP_PENDING)
        except RSMOperationVerificationError as error:
            if error.execution is not execution:
                raise RuntimeError("verification changed RSM execution owner")
            return RSMOwnerOutcome(RSMOwnerOutcomeKind.VERIFICATION_PENDING)
        if (
            type(result) is not RSMOperationResult
            or result.request is not execution.request
        ):
            raise TypeError("RSM execution returned an invalid result")
        return RSMOwnerOutcome(RSMOwnerOutcomeKind.RESULT, result=result)

    def _adopt(self, raw_update) -> RSMOwnerUpdate:
        identity = raw_update.identity
        command = identity.request
        if type(command) is not _WorkerCommand:
            raise RuntimeError("RSM worker identity lost its exact command")
        with self._lock:
            stale = (
                self._form is None
                or self._form.fingerprint != command.form_fingerprint
                or self._form_revision != command.form_revision
            )
            if raw_update.progress is not None:
                return RSMOwnerUpdate(
                    identity,
                    command.action,
                    progress=raw_update.progress,
                    stale=stale,
                )
            terminal = raw_update.terminal
            if terminal is None:
                raise RuntimeError("RSM worker update has no terminal")
            outcome = terminal.payload
            if terminal.status is OperationTerminalStatus.RETURNED:
                if type(outcome) is not RSMOwnerOutcome:
                    raise RuntimeError("RSM worker terminal payload is invalid")
                if outcome.kind is RSMOwnerOutcomeKind.PREFLIGHT_READY:
                    if not stale:
                        self._prepared = outcome.preflight
                        self._prepared_revision = command.form_revision
                elif outcome.kind is RSMOwnerOutcomeKind.PREFLIGHT_REFUSED:
                    self._prepared = None
                    self._prepared_revision = None
                elif outcome.kind is RSMOwnerOutcomeKind.RESULT:
                    self._last_result = outcome.result
                    self._execution = None
                    self._execution_preflight = None
                    self._execution_form_revision = None
                    self._finalization = RSMOwnerFinalization.NONE
                elif outcome.kind is RSMOwnerOutcomeKind.CLEANUP_PENDING:
                    self._finalization = RSMOwnerFinalization.CLEANUP_PENDING
                elif outcome.kind is RSMOwnerOutcomeKind.VERIFICATION_PENDING:
                    self._finalization = RSMOwnerFinalization.VERIFICATION_PENDING
            elif command.action is RSMOwnerAction.RETRY_CLEANUP:
                self._finalization = RSMOwnerFinalization.CLEANUP_PENDING
            elif command.action is RSMOwnerAction.RETRY_VERIFICATION:
                self._finalization = RSMOwnerFinalization.VERIFICATION_PENDING
            else:
                self._execution = None
                self._execution_preflight = None
                self._execution_form_revision = None
                self._finalization = RSMOwnerFinalization.NONE
            self._active_identity = None
            self._active_action = None
            return RSMOwnerUpdate(
                identity,
                command.action,
                terminal_status=terminal.status,
                outcome=(
                    outcome
                    if terminal.status is OperationTerminalStatus.RETURNED
                    else None
                ),
                failure_module=terminal.failure_module,
                failure_type=terminal.failure_type,
                failure_message=terminal.failure_message,
                stale=stale,
            )

    def _poll_active(self) -> RSMOwnerUpdate | None:
        with self._lock:
            identity = self._active_identity
        if identity is None:
            return None
        raw = self._worker.poll(identity)
        return None if raw is None else self._adopt(raw)

    def poll(self) -> RSMOwnerUpdate | None:
        with self._lock:
            if self._closing:
                return None
        return self._poll_active()

    def close(self) -> CloseReceipt:
        with self._lock:
            if self._clean_receipt is not None:
                return self._clean_receipt
            self._closing = True
            identity = self._active_identity
            action = self._active_action
        if identity is not None and action in {
            RSMOwnerAction.PREFLIGHT,
            RSMOwnerAction.RUN,
        }:
            self._worker.cancel(identity)
        self._poll_active()
        with self._lock:
            if self._active_identity is not None:
                return CloseReceipt(PageCleanup.PENDING, "rsm-operation")
            if self._finalization is RSMOwnerFinalization.CLEANUP_PENDING:
                started = self._retry_command_locked(
                    RSMOwnerAction.RETRY_CLEANUP,
                    RSMOwnerFinalization.CLEANUP_PENDING,
                    allow_closing=True,
                )
                if started is None:
                    return CloseReceipt(PageCleanup.PENDING, "rsm-cleanup-retry")
                return CloseReceipt(PageCleanup.PENDING, "rsm-cleanup-retry")
            if self._finalization is RSMOwnerFinalization.VERIFICATION_PENDING:
                started = self._retry_command_locked(
                    RSMOwnerAction.RETRY_VERIFICATION,
                    RSMOwnerFinalization.VERIFICATION_PENDING,
                    allow_closing=True,
                )
                if started is None:
                    return CloseReceipt(
                        PageCleanup.PENDING, "rsm-verification-retry"
                    )
                return CloseReceipt(PageCleanup.PENDING, "rsm-verification-retry")
        receipt = self._worker.close()
        if receipt.status is PageCleanup.PENDING:
            return receipt
        with self._lock:
            if self._clean_receipt is None:
                self._clean_receipt = receipt
            return self._clean_receipt


__all__ = [
    "RSMOwnerAction",
    "RSMOwnerFinalization",
    "RSMOwnerOutcome",
    "RSMOwnerOutcomeKind",
    "RSMOwnerUpdate",
    "RSMToolOwner",
]
