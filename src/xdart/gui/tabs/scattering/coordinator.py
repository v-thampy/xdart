"""Pure lifecycle coordinator for the phase-ready scattering workspace vNext kernel."""

from __future__ import annotations

from .events import (
    DurablePaused,
    DurableFinal,
    ExecutorAccepted,
    ExecutorStartFailed,
    ExecutionEnded,
    FatalExecution,
    LifecycleError,
    LifecycleResult,
    LifecycleStatus,
    OwnersClosed,
    PauseFailed,
    PauseRequested,
    PreflightAccepted,
    PreflightRefused,
    RequestId,
    ResumeFailed,
    ResumeRequested,
    Resumed,
    RunIdentity,
    StopRequested,
)
from .state_machine import IllegalTransition, RunPhase, RunSignal, transition


class ScatteringCoordinator:
    """Own only lifecycle facts; source, intent, writer, and Qt remain external."""

    def __init__(self) -> None:
        self._phase = RunPhase.IDLE
        self._next_request_value = 0
        self._request_id: RequestId | None = None
        self._attempt_run_identity: RunIdentity | None = None
        self._active_run_identity: RunIdentity | None = None
        self._cleanup_run_identity: RunIdentity | None = None
        self._event_sequence = 0
        self._closed = False
        self._invalidation_epoch = 0
        self._owners_closed = False

    @property
    def phase(self) -> RunPhase:
        return self._phase

    @property
    def request_id(self) -> RequestId | None:
        return self._request_id

    @property
    def attempt_run_identity(self) -> RunIdentity | None:
        return self._attempt_run_identity

    @property
    def active_run_identity(self) -> RunIdentity | None:
        return self._active_run_identity

    @property
    def event_sequence(self) -> int:
        return self._event_sequence

    @property
    def invalidation_epoch(self) -> int:
        return self._invalidation_epoch

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def reset_permitted(self) -> bool:
        """Whether FAILED cleanup has reached the reset transition."""

        return (
            not self._closed
            and self._phase is RunPhase.FAILED
            and self._owners_closed
        )

    def _result(
        self,
        status: LifecycleStatus,
        *,
        error: LifecycleError | None = None,
    ) -> LifecycleResult:
        return LifecycleResult(
            status=status,
            phase=self._phase,
            request_id=self._request_id,
            run_identity=self._active_run_identity or self._attempt_run_identity,
            error=error,
        )

    def _apply(self, signal: RunSignal) -> LifecycleResult:
        try:
            self._phase = transition(self._phase, signal)
        except IllegalTransition:
            return self._result(
                LifecycleStatus.REJECTED,
                error=LifecycleError.ILLEGAL_TRANSITION,
            )
        self._event_sequence += 1
        return self._result(LifecycleStatus.APPLIED)

    def _superseded(self) -> LifecycleResult:
        return self._result(
            LifecycleStatus.SUPERSEDED,
            error=LifecycleError.SUPERSEDED,
        )

    def begin_start(self) -> LifecycleResult:
        """Open a pre-freeze request.  No intent or source value is owned here."""

        result = self._apply(RunSignal.START)
        if result.status is not LifecycleStatus.APPLIED:
            return result
        self._next_request_value += 1
        self._request_id = RequestId(self._next_request_value)
        self._owners_closed = False
        return self._result(LifecycleStatus.APPLIED)

    def preflight_refused(self, event: PreflightRefused) -> LifecycleResult:
        if self._closed or event.request_id is not self._request_id:
            return self._superseded()
        result = self._apply(RunSignal.PREFLIGHT_REFUSED)
        if result.status is not LifecycleStatus.APPLIED:
            return result
        self._request_id = None
        return LifecycleResult(
            status=LifecycleStatus.APPLIED,
            phase=self._phase,
            request_id=event.request_id,
        )

    def preflight_accepted(self, event: PreflightAccepted) -> LifecycleResult:
        if self._closed or event.request_id is not self._request_id:
            return self._superseded()
        result = self._apply(RunSignal.PREFLIGHT_ACCEPTED)
        if result.status is LifecycleStatus.APPLIED:
            self._attempt_run_identity = RunIdentity.from_configuration(event.configuration)
            self._active_run_identity = None
            self._request_id = None
            return LifecycleResult(
                status=LifecycleStatus.APPLIED,
                phase=self._phase,
                request_id=event.request_id,
                run_identity=self._attempt_run_identity,
            )
        return result

    def executor_accepted(self, event: ExecutorAccepted) -> LifecycleResult:
        if self._closed or event.run_identity is not self._attempt_run_identity:
            return self._superseded()
        result = self._apply(RunSignal.EXECUTOR_ACCEPTED)
        if result.status is LifecycleStatus.APPLIED:
            self._active_run_identity = event.run_identity
            self._attempt_run_identity = None
        return self._result(result.status, error=result.error)

    def executor_start_failed(self, event: ExecutorStartFailed) -> LifecycleResult:
        if self._closed or event.run_identity is not self._attempt_run_identity:
            return self._superseded()
        return self._apply(RunSignal.EXECUTOR_START_FAILED)

    def contain_executor_failure(self, run_identity: RunIdentity) -> LifecycleResult:
        """Retain one invoked executor attempt while normalizing it to ``FAILED``."""

        if self._closed or (
            run_identity is not self._attempt_run_identity
            and run_identity is not self._active_run_identity
        ):
            return self._superseded()
        if self._phase is not RunPhase.FAILED:
            result = self._apply(RunSignal.FATAL)
            if result.status is not LifecycleStatus.APPLIED:
                return result
        else:
            self._event_sequence += 1
        self._active_run_identity = None
        self._attempt_run_identity = run_identity
        self._owners_closed = False
        return self._result(LifecycleStatus.APPLIED)

    def stop_requested(self, event: StopRequested) -> LifecycleResult:
        if (
            self._closed
            or type(event) is not StopRequested
            or event.run_identity is not self._active_run_identity
            or self._phase not in (
                RunPhase.RUNNING,
                RunPhase.PAUSING,
                RunPhase.PAUSED,
                RunPhase.RESUMING,
            )
        ):
            return self._superseded()
        return self._apply(RunSignal.STOP)

    def pause_requested(self, event: PauseRequested) -> LifecycleResult:
        if (
            self._closed
            or type(event) is not PauseRequested
            or event.run_identity is not self._active_run_identity
            or self._phase is not RunPhase.RUNNING
        ):
            return self._superseded()
        return self._apply(RunSignal.PAUSE)

    def durable_paused(self, event: DurablePaused) -> LifecycleResult:
        if (
            self._closed
            or type(event) is not DurablePaused
            or event.run_identity is not self._active_run_identity
            or type(event.durable_generation) is not int
            or event.durable_generation < 1
            or self._phase is not RunPhase.PAUSING
        ):
            return self._superseded()
        return self._apply(RunSignal.DURABLE_PAUSED)

    def pause_failed(self, event: PauseFailed) -> LifecycleResult:
        if (
            self._closed
            or type(event) is not PauseFailed
            or event.run_identity is not self._active_run_identity
            or self._phase is not RunPhase.PAUSING
        ):
            return self._superseded()
        return self._apply(RunSignal.PAUSE_FAILED)

    def resume_requested(self, event: ResumeRequested) -> LifecycleResult:
        if (
            self._closed
            or type(event) is not ResumeRequested
            or event.run_identity is not self._active_run_identity
            or self._phase is not RunPhase.PAUSED
        ):
            return self._superseded()
        return self._apply(RunSignal.RESUME)

    def resumed(self, event: Resumed) -> LifecycleResult:
        if (
            self._closed
            or type(event) is not Resumed
            or event.run_identity is not self._active_run_identity
            or self._phase is not RunPhase.RESUMING
        ):
            return self._superseded()
        return self._apply(RunSignal.RESUMED)

    def resume_failed(self, event: ResumeFailed) -> LifecycleResult:
        if (
            self._closed
            or type(event) is not ResumeFailed
            or event.run_identity is not self._active_run_identity
            or self._phase is not RunPhase.RESUMING
        ):
            return self._superseded()
        return self._apply(RunSignal.RESUME_FAILED)

    def execution_ended(self, event: ExecutionEnded) -> LifecycleResult:
        if (
            self._closed
            or type(event) is not ExecutionEnded
            or event.run_identity is not self._active_run_identity
            or self._phase not in (RunPhase.RUNNING, RunPhase.PAUSED, RunPhase.STOPPING)
        ):
            return self._superseded()
        return self._apply(RunSignal.NORMAL_END)

    def durable_final(self, event: DurableFinal) -> LifecycleResult:
        if type(event) is not DurableFinal or event.run_identity is not self._active_run_identity:
            return self._superseded()
        result = self._apply(RunSignal.DURABLE_FINAL)
        if result.status is not LifecycleStatus.APPLIED:
            return result
        self._active_run_identity = None
        self._attempt_run_identity = None
        self._owners_closed = False
        return LifecycleResult(
            LifecycleStatus.APPLIED,
            self._phase,
            run_identity=event.run_identity,
        )

    def fatal(self, event: FatalExecution) -> LifecycleResult:
        if (
            self._closed
            or type(event) is not FatalExecution
            or event.run_identity is not self._active_run_identity
        ):
            return self._superseded()
        result = self._apply(RunSignal.FATAL)
        if result.status is not LifecycleStatus.APPLIED:
            return result
        self._active_run_identity = None
        self._attempt_run_identity = event.run_identity
        self._owners_closed = False
        return self._result(LifecycleStatus.APPLIED)

    def owners_closed(self, event: OwnersClosed) -> LifecycleResult:
        if type(event) is not OwnersClosed:
            return self._superseded()
        if self._closed:
            if event.run_identity is not self._cleanup_run_identity:
                return self._superseded()
            result = self._apply(RunSignal.OWNERS_CLOSED)
            if result.status is not LifecycleStatus.APPLIED:
                return result
            self._cleanup_run_identity = None
            self._owners_closed = True
            return LifecycleResult(
                LifecycleStatus.APPLIED,
                self._phase,
                run_identity=event.run_identity,
            )
        if (
            self._phase is not RunPhase.FAILED
            or self._owners_closed
            or event.run_identity is not self._attempt_run_identity
        ):
            return self._superseded()
        result = self._apply(RunSignal.OWNERS_CLOSED)
        if result.status is LifecycleStatus.APPLIED:
            self._owners_closed = True
        return result

    def reset(self) -> LifecycleResult:
        if self._phase is RunPhase.FAILED and not self._owners_closed:
            return self._result(
                LifecycleStatus.REJECTED,
                error=LifecycleError.ILLEGAL_TRANSITION,
            )
        result = self._apply(RunSignal.RESET)
        if result.status is LifecycleStatus.APPLIED:
            self._attempt_run_identity = None
            self._owners_closed = False
        return self._result(result.status, error=result.error)

    def close(self) -> LifecycleResult:
        """Invalidate all identities before external owners receive close callbacks."""

        if self._closed:
            return self._superseded()
        signal = (
            RunSignal.CLOSE_AFTER_CLEANUP
            if self._phase is RunPhase.FAILED and self._owners_closed
            else RunSignal.CLOSE
        )
        result = self._apply(signal)
        if result.status is not LifecycleStatus.APPLIED:
            return result
        self._request_id = None
        self._attempt_run_identity = None
        self._active_run_identity = None
        self._cleanup_run_identity = None
        self._invalidation_epoch += 1
        self._closed = True
        cleanup_identity = result.run_identity
        if self._phase is RunPhase.STOPPING:
            self._cleanup_run_identity = cleanup_identity
        return LifecycleResult(
            LifecycleStatus.APPLIED,
            self._phase,
            run_identity=cleanup_identity if self._phase is RunPhase.STOPPING else None,
        )


__all__ = ["ScatteringCoordinator"]
