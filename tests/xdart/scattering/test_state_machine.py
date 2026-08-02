from __future__ import annotations

import pytest

from xdart.gui.tabs.scattering.state_machine import IllegalTransition, RunPhase, RunSignal, transition


LEGAL = {
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
for phase in (RunPhase.STARTING, RunPhase.RUNNING, RunPhase.PAUSING, RunPhase.PAUSED, RunPhase.RESUMING):
    LEGAL[phase, RunSignal.STOP] = RunPhase.STOPPING
    LEGAL[phase, RunSignal.NORMAL_END] = RunPhase.FINALIZING
for phase in (RunPhase.STARTING, RunPhase.RUNNING, RunPhase.PAUSING, RunPhase.PAUSED, RunPhase.RESUMING, RunPhase.STOPPING, RunPhase.FINALIZING):
    LEGAL[phase, RunSignal.FATAL] = RunPhase.FAILED
for phase in (RunPhase.RUNNING, RunPhase.PAUSING, RunPhase.PAUSED, RunPhase.RESUMING, RunPhase.FINALIZING):
    LEGAL[phase, RunSignal.CLOSE] = RunPhase.STOPPING
LEGAL[RunPhase.STOPPING, RunSignal.NORMAL_END] = RunPhase.FINALIZING


@pytest.mark.parametrize(("phase", "signal"), [(phase, signal) for phase in RunPhase for signal in RunSignal])
def test_transition_table_is_exhaustive_and_rejects_every_illegal_pair(phase, signal):
    expected = LEGAL.get((phase, signal))
    if expected is None:
        with pytest.raises(IllegalTransition) as exc:
            transition(phase, signal)
        assert exc.value.phase is phase
        assert exc.value.signal is signal
    else:
        assert transition(phase, signal) is expected


def test_transition_is_pure_and_start_failure_is_not_a_refusal():
    phase = RunPhase.STARTING
    assert transition(phase, RunSignal.EXECUTOR_START_FAILED) is RunPhase.FAILED
    assert phase is RunPhase.STARTING
