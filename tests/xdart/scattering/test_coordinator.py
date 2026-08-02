from __future__ import annotations

from pathlib import Path

from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.session.run_configuration import RunIntent
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.events import (
    CleanupStatus, ExecutorAccepted, ExecutorStartFailed, LifecycleError,
    LifecycleStatus, OwnersClosed, PreflightAccepted, PreflightRefused,
    RequestId, RunIdentity,
)
from xdart.gui.tabs.scattering.state_machine import RunPhase


def _frozen():
    return RunIntent(
        source_spec=SourceSpec(Path("/data/frame_0001.tif"), SourceKind.IMAGE_FILE),
    ).freeze()


def _start_to_starting() -> tuple[ScatteringCoordinator, RequestId, RunIdentity]:
    coordinator = ScatteringCoordinator()
    request = coordinator.begin_start().request_id
    assert request is not None
    frozen = _frozen()
    accepted = coordinator.preflight_accepted(PreflightAccepted(request, frozen))
    assert accepted.status is LifecycleStatus.APPLIED
    identity = accepted.run_identity
    assert identity is not None
    return coordinator, request, identity


def test_preflight_refusal_consumes_request_and_makes_late_events_inert():
    coordinator = ScatteringCoordinator()
    request = coordinator.begin_start().request_id
    assert request is not None

    result = coordinator.preflight_refused(PreflightRefused(request))

    assert result.status is LifecycleStatus.APPLIED
    assert result.request_id == request
    assert coordinator.phase is RunPhase.IDLE
    assert coordinator.request_id is None
    assert coordinator.attempt_run_identity is None
    assert coordinator.active_run_identity is None
    before = (coordinator.phase, coordinator.request_id, coordinator.event_sequence)

    late_refusal = coordinator.preflight_refused(PreflightRefused(request))
    late_acceptance = coordinator.preflight_accepted(
        PreflightAccepted(request, _frozen()),
    )

    assert late_refusal.status is LifecycleStatus.SUPERSEDED
    assert late_acceptance.status is LifecycleStatus.SUPERSEDED
    assert (coordinator.phase, coordinator.request_id, coordinator.event_sequence) == before


def test_executor_start_failure_records_one_consumed_attempt_and_enters_failed():
    coordinator, _, identity = _start_to_starting()

    result = coordinator.executor_start_failed(
        ExecutorStartFailed(identity, CleanupStatus.CLEANUP_PENDING),
    )

    assert result.status is LifecycleStatus.APPLIED
    assert coordinator.phase is RunPhase.FAILED
    assert coordinator.attempt_run_identity == identity
    assert coordinator.active_run_identity is None


def test_executor_acceptance_promotes_exact_attempt_identity():
    coordinator, _, identity = _start_to_starting()

    result = coordinator.executor_accepted(ExecutorAccepted(identity))

    assert result.status is LifecycleStatus.APPLIED
    assert coordinator.phase is RunPhase.RUNNING
    assert coordinator.attempt_run_identity is None
    assert coordinator.active_run_identity == identity


def test_preflight_acceptance_consumes_request_and_makes_late_events_inert():
    coordinator = ScatteringCoordinator()
    request = coordinator.begin_start().request_id
    assert request is not None
    frozen = _frozen()

    accepted = coordinator.preflight_accepted(PreflightAccepted(request, frozen))
    identity = RunIdentity.from_configuration(frozen)
    before = (
        coordinator.phase,
        coordinator.request_id,
        coordinator.attempt_run_identity,
        coordinator.event_sequence,
    )
    late = coordinator.preflight_refused(PreflightRefused(request))
    late_second = coordinator.preflight_accepted(PreflightAccepted(request, frozen))

    assert accepted.request_id == request
    assert accepted.run_identity == identity
    assert coordinator.request_id is None
    assert late.status is LifecycleStatus.SUPERSEDED
    assert late_second.status is LifecycleStatus.SUPERSEDED
    assert (
        coordinator.phase,
        coordinator.request_id,
        coordinator.attempt_run_identity,
        coordinator.event_sequence,
    ) == before


def test_later_request_is_monotonic_after_preflight_request_is_consumed():
    coordinator = ScatteringCoordinator()
    first = coordinator.begin_start().request_id
    assert first is not None
    coordinator.preflight_refused(PreflightRefused(first))

    second = coordinator.begin_start().request_id

    assert second is not None
    assert second.value == first.value + 1


def test_stale_request_event_is_inert():
    coordinator = ScatteringCoordinator()
    request = coordinator.begin_start().request_id
    assert request is not None
    before = (coordinator.phase, coordinator.request_id, coordinator.event_sequence)

    result = coordinator.preflight_refused(PreflightRefused(RequestId(request.value + 1)))

    assert result.status is LifecycleStatus.SUPERSEDED
    assert result.error is LifecycleError.SUPERSEDED
    assert (coordinator.phase, coordinator.request_id, coordinator.event_sequence) == before


def test_stale_run_event_is_inert():
    coordinator, _, identity = _start_to_starting()
    stale = RunIdentity(identity.generation + 1, identity.fingerprint)
    before = (coordinator.phase, coordinator.attempt_run_identity, coordinator.event_sequence)

    result = coordinator.executor_accepted(ExecutorAccepted(stale))

    assert result.status is LifecycleStatus.SUPERSEDED
    assert (coordinator.phase, coordinator.attempt_run_identity, coordinator.event_sequence) == before


def test_reset_is_rejected_until_owners_closed_is_acknowledged():
    coordinator, _, identity = _start_to_starting()
    coordinator.executor_start_failed(ExecutorStartFailed(identity, CleanupStatus.CLEANUP_PENDING))

    rejected = coordinator.reset()
    acknowledged = coordinator.owners_closed(OwnersClosed(identity))
    reset = coordinator.reset()

    assert rejected.status is LifecycleStatus.REJECTED
    assert coordinator.phase is RunPhase.IDLE
    assert acknowledged.status is LifecycleStatus.APPLIED
    assert reset.status is LifecycleStatus.APPLIED


def test_close_invalidates_identities_before_close_callbacks_run():
    coordinator, request, identity = _start_to_starting()
    coordinator.executor_accepted(ExecutorAccepted(identity))

    closed = coordinator.close()

    def external_close_callback() -> None:
        assert coordinator.closed is True
        assert coordinator.request_id is None
        assert coordinator.attempt_run_identity is None
        assert coordinator.active_run_identity is None
        assert coordinator.invalidation_epoch == 1

    external_close_callback()
    stale = coordinator.executor_accepted(ExecutorAccepted(identity))
    acknowledged = coordinator.owners_closed(OwnersClosed(identity))

    assert closed.status is LifecycleStatus.APPLIED
    assert closed.phase is RunPhase.STOPPING
    assert closed.run_identity is identity
    assert stale.status is LifecycleStatus.SUPERSEDED
    assert acknowledged.status is LifecycleStatus.APPLIED
    assert coordinator.phase is RunPhase.CLOSED
    assert request.value == 1
