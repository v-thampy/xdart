"""Qt-free asynchronous owner for standalone Stitch preflight and execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from threading import Lock

from xdart.gui.pages.operation_owner import (
    OperationIdentity,
    OperationProgress,
    OperationTerminalStatus,
    SingleWorkerOwner,
)
from xdart.gui.pages.values import CloseReceipt, PageCleanup
from xrd_tools.analysis.stitch_operation import (
    StitchOperationCleanupPending,
    StitchOperationExecution,
    StitchOperationResult,
    StitchOperationVerificationError,
)

from .stitch_values import (
    StitchToolForm,
    StitchToolPreflight,
    prepare_stitch_tool,
)


class StitchOwnerAction(str, Enum):
    PREFLIGHT = "preflight"
    RUN = "run"
    RETRY_CLEANUP = "retry-cleanup"
    RETRY_VERIFICATION = "retry-verification"


class StitchOwnerOutcomeKind(str, Enum):
    PREFLIGHT_READY = "preflight-ready"
    RESULT = "result"
    CLEANUP_PENDING = "cleanup-pending"
    VERIFICATION_PENDING = "verification-pending"


class StitchOwnerFinalization(str, Enum):
    NONE = "none"
    CLEANUP_PENDING = "cleanup-pending"
    VERIFICATION_PENDING = "verification-pending"


@dataclass(frozen=True, slots=True)
class StitchOwnerOutcome:
    """One returned worker value; pending outcomes retain execution in owner."""

    kind: StitchOwnerOutcomeKind
    preflight: StitchToolPreflight | None = None
    result: StitchOperationResult | None = None
    finalization_message: str = ""

    def __post_init__(self) -> None:
        if (
            type(self.kind) is not StitchOwnerOutcomeKind
            or type(self.finalization_message) is not str
            or (self.preflight is not None and type(self.preflight) is not StitchToolPreflight)
            or (self.result is not None and type(self.result) is not StitchOperationResult)
            or (
                self.kind is StitchOwnerOutcomeKind.PREFLIGHT_READY
                and (self.preflight is None or self.result is not None)
            )
            or (
                self.kind is StitchOwnerOutcomeKind.RESULT
                and (self.preflight is not None or self.result is None)
            )
            or (
                self.kind
                in {
                    StitchOwnerOutcomeKind.CLEANUP_PENDING,
                    StitchOwnerOutcomeKind.VERIFICATION_PENDING,
                }
                and (self.preflight is not None or self.result is not None)
            )
        ):
            raise TypeError("Stitch owner outcome is invalid")


@dataclass(frozen=True, slots=True)
class StitchOwnerUpdate:
    """Latest-only progress or one terminal suitable for a thin dialog."""

    identity: OperationIdentity
    action: StitchOwnerAction
    progress: OperationProgress | None = None
    terminal_status: OperationTerminalStatus | None = None
    outcome: StitchOwnerOutcome | None = None
    failure_module: str = ""
    failure_type: str = ""
    failure_message: str = ""
    stale: bool = False

    def __post_init__(self) -> None:
        terminal = self.terminal_status is not None
        if (
            type(self.identity) is not OperationIdentity
            or type(self.action) is not StitchOwnerAction
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
            or (self.outcome is not None and type(self.outcome) is not StitchOwnerOutcome)
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
            raise TypeError("Stitch owner update is invalid")

    @property
    def terminal(self) -> bool:
        return self.terminal_status is not None


@dataclass(frozen=True, slots=True)
class _WorkerCommand:
    action: StitchOwnerAction
    form_fingerprint: str
    form_revision: int
    form: StitchToolForm | None = None
    preflight: StitchToolPreflight | None = None
    execution: StitchOperationExecution | None = None

    def __post_init__(self) -> None:
        if (
            type(self.action) is not StitchOwnerAction
            or type(self.form_fingerprint) is not str
            or not self.form_fingerprint
            or type(self.form_revision) is not int
            or self.form_revision < 1
            or (self.form is not None and type(self.form) is not StitchToolForm)
            or (
                self.preflight is not None
                and type(self.preflight) is not StitchToolPreflight
            )
            or (
                self.execution is not None
                and type(self.execution) is not StitchOperationExecution
            )
        ):
            raise TypeError("Stitch worker command is invalid")
        if self.action is StitchOwnerAction.PREFLIGHT:
            valid = self.form is not None and self.preflight is None and self.execution is None
        else:
            valid = (
                self.form is None
                and self.preflight is not None
                and self.execution is not None
                and self.execution.request is self.preflight.request
                and self.form_fingerprint == self.preflight.form.fingerprint
            )
        if not valid:
            raise TypeError("Stitch worker command fields disagree")


PreflightRunner = Callable[[StitchToolForm], StitchToolPreflight]
ExecutionFactory = Callable[..., StitchOperationExecution]


class StitchToolOwner:
    """Own one worker, one prepared request, and exact finalization retries.

    Editing a form never mutates or silently replaces an accepted preflight.
    It only makes that preflight stale, so :meth:`begin_run` refuses until a new
    preflight is delivered.  Cleanup and verification retries always target the
    retained :class:`StitchOperationExecution`; they never reconstruct a request
    or replay science.
    """

    def __init__(
        self,
        *,
        coordinator=None,
        preflight_runner: PreflightRunner = prepare_stitch_tool,
        execution_factory: ExecutionFactory = StitchOperationExecution,
        join_timeout: float = 0.0,
    ) -> None:
        if not callable(preflight_runner) or not callable(execution_factory):
            raise TypeError("Stitch owner adapters must be callable")
        self._coordinator = coordinator
        self._preflight_runner = preflight_runner
        self._execution_factory = execution_factory
        self._worker = SingleWorkerOwner(self._run_command, join_timeout=join_timeout)
        self._lock = Lock()
        self._form: StitchToolForm | None = None
        self._form_revision = 0
        self._prepared: StitchToolPreflight | None = None
        self._prepared_revision: int | None = None
        self._execution: StitchOperationExecution | None = None
        self._execution_preflight: StitchToolPreflight | None = None
        self._execution_form_revision: int | None = None
        self._last_result: StitchOperationResult | None = None
        self._finalization = StitchOwnerFinalization.NONE
        self._active_identity: OperationIdentity | None = None
        self._active_action: StitchOwnerAction | None = None
        self._closing = False
        self._clean_receipt: CloseReceipt | None = None

    def __copy__(self):
        raise TypeError("Stitch tool owner is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("Stitch tool owner is not copyable")

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._active_identity is not None

    @property
    def form(self) -> StitchToolForm | None:
        with self._lock:
            return self._form

    @property
    def prepared(self) -> StitchToolPreflight | None:
        with self._lock:
            return self._prepared

    @property
    def prepared_stale(self) -> bool:
        with self._lock:
            return self._prepared is not None and not self._prepared.is_current(
                self._form
            )

    @property
    def finalization(self) -> StitchOwnerFinalization:
        with self._lock:
            return self._finalization

    @property
    def last_result(self) -> StitchOperationResult | None:
        with self._lock:
            return self._last_result

    @property
    def current_identity(self) -> OperationIdentity | None:
        with self._lock:
            return self._active_identity

    def set_form(self, form: StitchToolForm) -> bool:
        """Set current form and report whether an accepted preflight is stale."""

        if type(form) is not StitchToolForm:
            raise TypeError("Stitch owner form must be exact StitchToolForm")
        with self._lock:
            if self._closing:
                return False
            changed = (
                self._form is None
                or self._form.fingerprint != form.fingerprint
            )
            revoked = changed and self._prepared is not None
            self._form = form
            if changed:
                self._form_revision += 1
                # A transition permanently revokes prior run authority.  A
                # later A -> B -> A bounce must require a fresh preflight even
                # though A's content fingerprint is identical.
                self._prepared = None
                self._prepared_revision = None
            return revoked

    def _begin_locked(
        self,
        command: _WorkerCommand,
        *,
        allow_closing: bool = False,
    ) -> OperationIdentity | None:
        if (
            (self._closing and not allow_closing)
            or self._active_identity is not None
        ):
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
                or self._finalization is not StitchOwnerFinalization.NONE
            ):
                return None
            command = _WorkerCommand(
                StitchOwnerAction.PREFLIGHT,
                self._form.fingerprint,
                self._form_revision,
                form=self._form,
            )
            identity = self._begin_locked(command)
            if identity is not None:
                # A re-preflight revokes the previous request immediately.  A
                # failed source/geometry recapture for the same form fingerprint
                # must never leave the older exact-object request runnable.
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
                or self._finalization is not StitchOwnerFinalization.NONE
            ):
                return None
            execution = self._execution_factory(
                preflight.request, coordinator=self._coordinator
            )
            if (
                type(execution) is not StitchOperationExecution
                or execution.request is not preflight.request
            ):
                raise TypeError(
                    "execution factory must return the exact prepared Stitch execution"
                )
            command = _WorkerCommand(
                StitchOwnerAction.RUN,
                preflight.form.fingerprint,
                self._prepared_revision,
                preflight=preflight,
                execution=execution,
            )
            identity = self._begin_locked(command)
            if identity is not None:
                # Install before the worker can deliver a pending-finalization
                # terminal.  The execution, not an exception wrapper, is owner.
                self._execution = execution
                self._execution_preflight = preflight
                self._execution_form_revision = self._prepared_revision
                self._last_result = None
            return identity

    def _retry_command_locked(
        self,
        action: StitchOwnerAction,
        finalization: StitchOwnerFinalization,
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
        command = _WorkerCommand(
            action,
            preflight.form.fingerprint,
            form_revision,
            preflight=preflight,
            execution=execution,
        )
        return self._begin_locked(command, allow_closing=allow_closing)

    def begin_retry_cleanup(self) -> OperationIdentity | None:
        with self._lock:
            return self._retry_command_locked(
                StitchOwnerAction.RETRY_CLEANUP,
                StitchOwnerFinalization.CLEANUP_PENDING,
            )

    def begin_retry_verification(self) -> OperationIdentity | None:
        with self._lock:
            return self._retry_command_locked(
                StitchOwnerAction.RETRY_VERIFICATION,
                StitchOwnerFinalization.VERIFICATION_PENDING,
            )

    def cancel(self) -> bool:
        """Request cooperative cancellation of science exactly once."""

        with self._lock:
            identity = self._active_identity
            if self._active_action is not StitchOwnerAction.RUN:
                return False
        return identity is not None and self._worker.cancel(identity)

    def _run_command(self, request, cancel, publish):
        if type(request) is not _WorkerCommand:
            raise TypeError("Stitch worker requires one exact command")
        if request.action is StitchOwnerAction.PREFLIGHT:
            publish("preflight", 0, 1)
            preflight = self._preflight_runner(request.form)
            if type(preflight) is not StitchToolPreflight:
                raise TypeError("preflight adapter returned an invalid value")
            if preflight.form.fingerprint != request.form_fingerprint:
                raise ValueError("preflight adapter changed the form identity")
            publish("preflight", 1, 1)
            return StitchOwnerOutcome(
                StitchOwnerOutcomeKind.PREFLIGHT_READY,
                preflight=preflight,
            )
        execution = request.execution
        if execution is None:
            raise TypeError("Stitch execution command lost its owner")
        try:
            if request.action is StitchOwnerAction.RUN:
                result = execution.run(
                    cancel_token=cancel,
                    progress_callback=lambda progress: publish(
                        progress.stage, progress.completed, progress.total
                    ),
                )
            elif request.action is StitchOwnerAction.RETRY_CLEANUP:
                publish("cleanup", 0, 1)
                result = execution.retry_cleanup()
                publish("cleanup", 1, 1)
            elif request.action is StitchOwnerAction.RETRY_VERIFICATION:
                publish("verification", 0, 1)
                result = execution.retry_verification()
                publish("verification", 1, 1)
            else:  # pragma: no cover - exact enum exhaustiveness fence
                raise RuntimeError("unknown Stitch owner action")
        except StitchOperationCleanupPending as error:
            if error.execution is not execution:
                raise RuntimeError("cleanup exception changed Stitch execution owner")
            return StitchOwnerOutcome(
                StitchOwnerOutcomeKind.CLEANUP_PENDING,
                finalization_message=str(error),
            )
        except StitchOperationVerificationError as error:
            if error.execution is not execution:
                raise RuntimeError(
                    "verification exception changed Stitch execution owner"
                )
            return StitchOwnerOutcome(StitchOwnerOutcomeKind.VERIFICATION_PENDING)
        if type(result) is not StitchOperationResult or result.request is not execution.request:
            raise TypeError("Stitch execution returned an invalid result")
        return StitchOwnerOutcome(StitchOwnerOutcomeKind.RESULT, result=result)

    def _adopt(self, raw_update) -> StitchOwnerUpdate:
        identity = raw_update.identity
        command = identity.request
        if type(command) is not _WorkerCommand:
            raise RuntimeError("Stitch worker identity lost its exact command")
        with self._lock:
            stale = (
                self._form is None
                or self._form.fingerprint != command.form_fingerprint
                or self._form_revision != command.form_revision
            )
            if raw_update.progress is not None:
                return StitchOwnerUpdate(
                    identity,
                    command.action,
                    progress=raw_update.progress,
                    stale=stale,
                )
            terminal = raw_update.terminal
            if terminal is None:
                raise RuntimeError("Stitch worker update has no terminal")
            outcome = terminal.payload
            if terminal.status is OperationTerminalStatus.RETURNED:
                if type(outcome) is not StitchOwnerOutcome:
                    raise RuntimeError("Stitch worker terminal payload is invalid")
                if outcome.kind is StitchOwnerOutcomeKind.PREFLIGHT_READY:
                    if not stale:
                        self._prepared = outcome.preflight
                        self._prepared_revision = command.form_revision
                elif outcome.kind is StitchOwnerOutcomeKind.RESULT:
                    self._last_result = outcome.result
                    self._execution = None
                    self._execution_preflight = None
                    self._execution_form_revision = None
                    self._finalization = StitchOwnerFinalization.NONE
                elif outcome.kind is StitchOwnerOutcomeKind.CLEANUP_PENDING:
                    self._finalization = StitchOwnerFinalization.CLEANUP_PENDING
                elif outcome.kind is StitchOwnerOutcomeKind.VERIFICATION_PENDING:
                    self._finalization = StitchOwnerFinalization.VERIFICATION_PENDING
            elif command.action is StitchOwnerAction.RETRY_CLEANUP:
                # An unexpected adapter failure does not transfer or abandon
                # exact cleanup authority.  A later retry still targets the
                # same execution and never replays science.
                self._finalization = StitchOwnerFinalization.CLEANUP_PENDING
            elif command.action is StitchOwnerAction.RETRY_VERIFICATION:
                self._finalization = StitchOwnerFinalization.VERIFICATION_PENDING
            else:
                self._execution = None
                self._execution_preflight = None
                self._execution_form_revision = None
                self._finalization = StitchOwnerFinalization.NONE
            self._active_identity = None
            self._active_action = None
            return StitchOwnerUpdate(
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

    def _poll_active(self) -> StitchOwnerUpdate | None:
        with self._lock:
            identity = self._active_identity
        if identity is None:
            return None
        raw = self._worker.poll(identity)
        if raw is None:
            return None
        return self._adopt(raw)

    def poll(self) -> StitchOwnerUpdate | None:
        """Deliver latest progress or one terminal for the active command."""

        with self._lock:
            if self._closing:
                return None
        return self._poll_active()

    def close(self) -> CloseReceipt:
        """Cancel, drain/adopt the terminal, and retry owned cleanup before seal."""

        with self._lock:
            if self._clean_receipt is not None:
                return self._clean_receipt
            self._closing = True
            identity = self._active_identity
            action = self._active_action
        if identity is not None and action is StitchOwnerAction.RUN:
            self._worker.cancel(identity)
        # Crucially poll before SingleWorkerOwner.close().  Its close seals
        # polling and retires terminal payloads; doing that first could discard
        # the only signal that the exact execution still owns retryable cleanup.
        self._poll_active()
        with self._lock:
            if self._active_identity is not None:
                return CloseReceipt(PageCleanup.PENDING, "stitch-operation")
            if self._finalization is StitchOwnerFinalization.CLEANUP_PENDING:
                started = self._retry_command_locked(
                    StitchOwnerAction.RETRY_CLEANUP,
                    StitchOwnerFinalization.CLEANUP_PENDING,
                    allow_closing=True,
                )
                if started is None:
                    return CloseReceipt(
                        PageCleanup.PENDING, "stitch-cleanup-retry"
                    )
                return CloseReceipt(PageCleanup.PENDING, "stitch-cleanup-retry")
            if self._finalization is StitchOwnerFinalization.VERIFICATION_PENDING:
                started = self._retry_command_locked(
                    StitchOwnerAction.RETRY_VERIFICATION,
                    StitchOwnerFinalization.VERIFICATION_PENDING,
                    allow_closing=True,
                )
                if started is None:
                    return CloseReceipt(
                        PageCleanup.PENDING, "stitch-verification-retry"
                    )
                return CloseReceipt(
                    PageCleanup.PENDING, "stitch-verification-retry"
                )
        receipt = self._worker.close()
        if receipt.status is PageCleanup.PENDING:
            return receipt
        with self._lock:
            if self._clean_receipt is None:
                self._clean_receipt = receipt
            return self._clean_receipt


__all__ = [
    "StitchOwnerAction",
    "StitchOwnerFinalization",
    "StitchOwnerOutcome",
    "StitchOwnerOutcomeKind",
    "StitchOwnerUpdate",
    "StitchToolOwner",
]
