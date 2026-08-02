from __future__ import annotations

from pathlib import Path

from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.session.run_configuration import RunIntent
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.events import (
    DurableFinal,
    ExecutionEnded,
    ExecutorAccepted,
    FatalExecution,
    LifecycleStatus,
    OwnersClosed,
    PreflightAccepted,
    RunIdentity,
    StopRequested,
)
from xdart.gui.tabs.scattering.state_machine import RunPhase


def _running() -> tuple[ScatteringCoordinator, RunIdentity]:
    coordinator = ScatteringCoordinator()
    request = coordinator.begin_start().request_id
    assert request is not None
    frozen = RunIntent(
        source_spec=SourceSpec(
            Path("/data/frame_0001.tif"),
            SourceKind.IMAGE_FILE,
        ),
    ).freeze()
    accepted = coordinator.preflight_accepted(
        PreflightAccepted(request, frozen),
    )
    identity = accepted.run_identity
    assert identity is not None
    assert (
        coordinator.executor_accepted(ExecutorAccepted(identity)).status
        is LifecycleStatus.APPLIED
    )
    return coordinator, identity


def test_duplicate_fatal_owners_closed_is_inert() -> None:
    coordinator, identity = _running()
    assert (
        coordinator.fatal(FatalExecution(identity)).status
        is LifecycleStatus.APPLIED
    )
    assert (
        coordinator.owners_closed(OwnersClosed(identity)).status
        is LifecycleStatus.APPLIED
    )
    before = (
        coordinator.phase,
        coordinator.event_sequence,
        coordinator.attempt_run_identity,
    )

    duplicate = coordinator.owners_closed(OwnersClosed(identity))

    assert duplicate.status is LifecycleStatus.SUPERSEDED
    assert (
        coordinator.phase,
        coordinator.event_sequence,
        coordinator.attempt_run_identity,
    ) == before


def test_workspace_close_during_user_stop_retains_cleanup_identity() -> None:
    coordinator, identity = _running()
    assert (
        coordinator.stop_requested(StopRequested(identity)).status
        is LifecycleStatus.APPLIED
    )
    before = coordinator.event_sequence

    closed = coordinator.close()

    assert closed.status is LifecycleStatus.APPLIED
    assert closed.phase is RunPhase.STOPPING
    assert closed.run_identity is identity
    assert coordinator.closed is True
    assert coordinator.active_run_identity is None
    assert coordinator.event_sequence == before + 1
    acknowledged = coordinator.owners_closed(OwnersClosed(identity))
    assert acknowledged.status is LifecycleStatus.APPLIED
    assert acknowledged.phase is RunPhase.CLOSED


def test_all_terminal_malformed_objects_are_total_and_inert() -> None:
    coordinator, identity = _running()
    operations = (
        coordinator.stop_requested,
        coordinator.execution_ended,
        coordinator.durable_final,
        coordinator.fatal,
        coordinator.owners_closed,
    )
    before = (
        coordinator.phase,
        coordinator.event_sequence,
        coordinator.active_run_identity,
    )

    for operation in operations:
        result = operation(object())
        assert result.status is LifecycleStatus.SUPERSEDED
        assert (
            coordinator.phase,
            coordinator.event_sequence,
            coordinator.active_run_identity,
        ) == before

    assert (
        coordinator.execution_ended(ExecutionEnded(identity)).status
        is LifecycleStatus.APPLIED
    )
    assert (
        coordinator.durable_final(DurableFinal(identity)).status
        is LifecycleStatus.APPLIED
    )
