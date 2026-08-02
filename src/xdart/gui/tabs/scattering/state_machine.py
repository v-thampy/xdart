"""Pure run-lifecycle transition table for the scattering workspace vNext kernel."""

from __future__ import annotations

from enum import Enum


class RunPhase(str, Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    STARTING = "starting"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    RESUMING = "resuming"
    STOPPING = "stopping"
    FINALIZING = "finalizing"
    FAILED = "failed"
    CLOSED = "closed"


class RunSignal(str, Enum):
    START = "start"
    PREFLIGHT_REFUSED = "preflight_refused"
    PREFLIGHT_ACCEPTED = "preflight_accepted"
    EXECUTOR_ACCEPTED = "executor_accepted"
    EXECUTOR_START_FAILED = "executor_start_failed"
    PAUSE = "pause"
    DURABLE_PAUSED = "durable_paused"
    PAUSE_FAILED = "pause_failed"
    RESUME = "resume"
    RESUMED = "resumed"
    RESUME_FAILED = "resume_failed"
    STOP = "stop"
    NORMAL_END = "normal_end"
    DURABLE_FINAL = "durable_final"
    FATAL = "fatal"
    OWNERS_CLOSED = "owners_closed"
    RESET = "reset"
    CLOSE = "close"
    CLOSE_AFTER_CLEANUP = "close_after_cleanup"


class IllegalTransition(ValueError):
    def __init__(self, phase: RunPhase, signal: RunSignal) -> None:
        super().__init__(f"illegal lifecycle transition: {phase.value} + {signal.value}")
        self.phase = phase
        self.signal = signal


_TRANSITIONS: dict[tuple[RunPhase, RunSignal], RunPhase] = {
    (RunPhase.IDLE, RunSignal.START): RunPhase.PREPARING,
    (RunPhase.PREPARING, RunSignal.PREFLIGHT_REFUSED): RunPhase.IDLE,
    (RunPhase.PREPARING, RunSignal.PREFLIGHT_ACCEPTED): RunPhase.STARTING,
    (RunPhase.STARTING, RunSignal.EXECUTOR_ACCEPTED): RunPhase.RUNNING,
    (RunPhase.STARTING, RunSignal.EXECUTOR_START_FAILED): RunPhase.FAILED,
    (RunPhase.RUNNING, RunSignal.PAUSE): RunPhase.PAUSING,
    (RunPhase.PAUSING, RunSignal.DURABLE_PAUSED): RunPhase.PAUSED,
    (RunPhase.PAUSING, RunSignal.PAUSE_FAILED): RunPhase.RUNNING,
    (RunPhase.PAUSED, RunSignal.RESUME): RunPhase.RESUMING,
    (RunPhase.RESUMING, RunSignal.RESUMED): RunPhase.RUNNING,
    (RunPhase.RESUMING, RunSignal.RESUME_FAILED): RunPhase.PAUSED,
    (RunPhase.FINALIZING, RunSignal.DURABLE_FINAL): RunPhase.IDLE,
    (RunPhase.FAILED, RunSignal.OWNERS_CLOSED): RunPhase.FAILED,
    (RunPhase.FAILED, RunSignal.RESET): RunPhase.IDLE,
    (RunPhase.IDLE, RunSignal.CLOSE): RunPhase.CLOSED,
    (RunPhase.PREPARING, RunSignal.CLOSE): RunPhase.CLOSED,
    (RunPhase.FAILED, RunSignal.CLOSE): RunPhase.STOPPING,
    (RunPhase.FAILED, RunSignal.CLOSE_AFTER_CLEANUP): RunPhase.CLOSED,
    (RunPhase.STOPPING, RunSignal.OWNERS_CLOSED): RunPhase.CLOSED,
    (RunPhase.STOPPING, RunSignal.CLOSE): RunPhase.STOPPING,
    (RunPhase.STARTING, RunSignal.CLOSE): RunPhase.CLOSED,
}
for _phase in (
    RunPhase.STARTING,
    RunPhase.RUNNING,
    RunPhase.PAUSING,
    RunPhase.PAUSED,
    RunPhase.RESUMING,
):
    _TRANSITIONS[_phase, RunSignal.STOP] = RunPhase.STOPPING
    _TRANSITIONS[_phase, RunSignal.NORMAL_END] = RunPhase.FINALIZING
for _phase in (
    RunPhase.STARTING,
    RunPhase.RUNNING,
    RunPhase.PAUSING,
    RunPhase.PAUSED,
    RunPhase.RESUMING,
    RunPhase.STOPPING,
    RunPhase.FINALIZING,
):
    _TRANSITIONS[_phase, RunSignal.FATAL] = RunPhase.FAILED
for _phase in (
    RunPhase.RUNNING,
    RunPhase.PAUSING,
    RunPhase.PAUSED,
    RunPhase.RESUMING,
    RunPhase.FINALIZING,
):
    _TRANSITIONS[_phase, RunSignal.CLOSE] = RunPhase.STOPPING
_TRANSITIONS[RunPhase.STOPPING, RunSignal.NORMAL_END] = RunPhase.FINALIZING


def transition(phase: RunPhase, signal: RunSignal) -> RunPhase:
    """Return a next phase without mutating either input."""

    try:
        return _TRANSITIONS[phase, signal]
    except KeyError:
        raise IllegalTransition(phase, signal) from None


__all__ = ["IllegalTransition", "RunPhase", "RunSignal", "transition"]
