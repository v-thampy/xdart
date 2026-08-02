"""Production-wired E1b-0.2 Close idempotence oracle."""

from pathlib import Path

from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import GIIntent, RunIntent
from xdart.gui.tabs.scattering.contracts import SourceCapture
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.events import (
    ExecutorAccepted,
    LifecycleError,
    LifecycleResult,
    LifecycleStatus,
    OwnersClosed,
    RequestId,
    RunIdentity,
)
from xdart.gui.tabs.scattering.start_pipeline import StartPipeline
from tests.xdart.scattering._admission import admission_for
from xdart.gui.tabs.scattering.state_machine import RunPhase


class Source:
    def __init__(self) -> None:
        self.cancelled: list[RequestId] = []

    def capture(self, source: SourceSpec, request_id: RequestId) -> SourceCapture:
        return SourceCapture(request_id, 0, source, ("th",))

    def cancel(self, request_id: RequestId) -> None:
        self.cancelled.append(request_id)


class Executor:
    def __init__(self, lifecycle: ScatteringCoordinator) -> None:
        self.lifecycle = lifecycle
        self.closed: list[RunIdentity] = []
        self.observed_public_identities: list[tuple[object, object]] = []

    def start(self, configuration, source, run_identity, admission):
        return ExecutorAccepted(run_identity)

    def close(self, run_identity: RunIdentity) -> None:
        self.closed.append(run_identity)
        self.observed_public_identities.append(
            (self.lifecycle.active_run_identity, self.lifecycle.attempt_run_identity)
        )

    def pause(self, run_identity: RunIdentity) -> None:
        pass

    def resume(self, run_identity: RunIdentity) -> None:
        pass

    def stop(self, run_identity: RunIdentity) -> None:
        pass


class RejectedDuplicateClose:
    """Delegate the first Close, then return a valid but unqualified rejection."""

    def __init__(self, coordinator: ScatteringCoordinator) -> None:
        self._coordinator = coordinator
        self._close_count = 0

    def __getattr__(self, name: str):
        return getattr(self._coordinator, name)

    def close(self) -> LifecycleResult:
        self._close_count += 1
        if self._close_count == 1:
            return self._coordinator.close()
        return LifecycleResult(
            LifecycleStatus.REJECTED,
            RunPhase.STOPPING,
            error=LifecycleError.ILLEGAL_TRANSITION,
        )


def _pipeline(*, lifecycle: ScatteringCoordinator | None = None):
    coordinator = lifecycle or ScatteringCoordinator()
    source_spec = SourceSpec(Path("/data/frame_0001.tif"), SourceKind.IMAGE_FILE)
    intent = RunIntent(
        source_spec=source_spec,
        gi=GIIntent(enabled=True, incidence_motor="th", th_val=0.2),
    )
    source = Source()
    executor = Executor(coordinator)
    return (
        StartPipeline(
            intents=RunIntentStore(intent),
            lifecycle=coordinator,
            sources=source,
            executor=executor,
        ),
        coordinator,
        source,
        executor,
    )


def test_duplicate_composed_close_returns_exact_inert_supersession() -> None:
    pipeline, lifecycle, source, executor = _pipeline()
    launched = pipeline.start(admission_for(pipeline.begin()))
    identity = launched.run_identity

    first = pipeline.close()
    before = (lifecycle.phase, lifecycle.event_sequence, lifecycle.closed)
    second = pipeline.close()

    assert first.lifecycle_result.status is LifecycleStatus.APPLIED
    assert first.lifecycle_result.phase is RunPhase.STOPPING
    assert first.lifecycle_result.run_identity is identity
    assert executor.observed_public_identities == [(None, None)]
    assert second.lifecycle_result == LifecycleResult(
        LifecycleStatus.SUPERSEDED,
        RunPhase.STOPPING,
        error=LifecycleError.SUPERSEDED,
    )
    assert second.recovery_failures == ()
    assert (lifecycle.phase, lifecycle.event_sequence, lifecycle.closed) == before
    assert source.cancelled == [RequestId(1)]
    assert executor.closed == [identity]
    foreign = RunIdentity(identity.generation, identity.fingerprint)
    assert pipeline.owners_closed(OwnersClosed(foreign)).status is LifecycleStatus.REJECTED
    assert (lifecycle.phase, lifecycle.event_sequence, lifecycle.closed) == before
    assert pipeline.owners_closed(OwnersClosed(identity)).phase is RunPhase.CLOSED


def test_duplicate_immediate_close_returns_exact_inert_supersession() -> None:
    pipeline, lifecycle, source, executor = _pipeline()

    first = pipeline.close()
    before = (lifecycle.phase, lifecycle.event_sequence, lifecycle.closed)
    second = pipeline.close()

    assert first.lifecycle_result == LifecycleResult(LifecycleStatus.APPLIED, RunPhase.CLOSED)
    assert second.lifecycle_result == LifecycleResult(
        LifecycleStatus.SUPERSEDED,
        RunPhase.CLOSED,
        error=LifecycleError.SUPERSEDED,
    )
    assert second.recovery_failures == ()
    assert (lifecycle.phase, lifecycle.event_sequence, lifecycle.closed) == before
    assert source.cancelled == []
    assert executor.closed == []


def test_rejected_duplicate_result_is_not_accepted_as_inert_close() -> None:
    coordinator = ScatteringCoordinator()
    lifecycle = RejectedDuplicateClose(coordinator)
    pipeline, _, _, executor = _pipeline(lifecycle=lifecycle)
    launched = pipeline.start(admission_for(pipeline.begin()))

    pipeline.close()
    second = pipeline.close()

    assert second.lifecycle_result == LifecycleResult(LifecycleStatus.REJECTED, RunPhase.CLOSED)
    assert [failure.operation for failure in second.recovery_failures] == ["lifecycle.close"]
    assert executor.closed == [launched.run_identity]
