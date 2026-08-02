"""Exact-object adversaries for the E1b-0 active-Close contract."""

from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.events import (
    ExecutorAccepted,
    FatalExecution,
    LifecycleStatus,
    OwnersClosed,
    PreflightAccepted,
    StopRequested,
)
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xrd_tools.session.run_configuration import RunIntent


def _running():
    coordinator = ScatteringCoordinator()
    request = coordinator.begin_start().request_id
    accepted = coordinator.preflight_accepted(
        PreflightAccepted(request, RunIntent().freeze())
    )
    identity = accepted.run_identity
    assert identity is not None
    assert (
        coordinator.executor_accepted(ExecutorAccepted(identity)).status
        is LifecycleStatus.APPLIED
    )
    return coordinator, identity


def test_close_after_stop_retains_cleanup_identity_until_exact_ack():
    coordinator, identity = _running()
    assert (
        coordinator.stop_requested(StopRequested(identity)).phase
        is RunPhase.STOPPING
    )

    closed = coordinator.close()

    assert closed.status is LifecycleStatus.APPLIED
    assert closed.phase is RunPhase.STOPPING
    assert closed.run_identity is identity
    assert coordinator.closed is True
    assert coordinator.active_run_identity is None
    assert coordinator.attempt_run_identity is None
    acknowledged = coordinator.owners_closed(OwnersClosed(identity))
    assert acknowledged.status is LifecycleStatus.APPLIED
    assert acknowledged.phase is RunPhase.CLOSED


def test_close_while_failed_cleanup_pending_waits_for_exact_ack():
    coordinator, identity = _running()
    assert coordinator.fatal(FatalExecution(identity)).phase is RunPhase.FAILED

    closed = coordinator.close()

    assert closed.status is LifecycleStatus.APPLIED
    assert closed.phase is RunPhase.STOPPING
    assert closed.run_identity is identity
    assert coordinator.closed is True
    assert coordinator.active_run_identity is None
    assert coordinator.attempt_run_identity is None
    acknowledged = coordinator.owners_closed(OwnersClosed(identity))
    assert acknowledged.status is LifecycleStatus.APPLIED
    assert acknowledged.phase is RunPhase.CLOSED
