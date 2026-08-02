"""Adversarial containment oracles for the revision-qualified Start owner."""

from __future__ import annotations

from pathlib import Path

import pytest

from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import GIIntent, RunIntent
from xdart.gui.tabs.scattering.contracts import SourceCapture
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.events import (
    CleanupStatus,
    ExecutorAccepted,
    ExecutorClosed,
    ExecutorStartFailed,
    LifecycleError,
    LifecycleResult,
    LifecycleStatus,
    OwnersClosed,
    RequestId,
    RunIdentity,
)
from xdart.gui.tabs.scattering.start_outcomes import (
    StartCapture,
    StartFailed,
    StartRejection,
    StartRejected,
)
from xdart.gui.tabs.scattering.start_pipeline import StartPipeline
from xdart.gui.tabs.scattering.state_machine import RunPhase
from tests.xdart.scattering._admission import admission_for


def _intent(*, source: bool = True) -> RunIntent:
    return RunIntent(
        source_spec=(
            SourceSpec(Path("/data/frame.tif"), SourceKind.IMAGE_FILE)
            if source else None
        ),
        gi=GIIntent(enabled=True, incidence_motor="th", th_val=0.2),
    )


class Source:
    def __init__(self) -> None:
        self.cancelled: list[RequestId] = []
        self.callback = None

    def capture(self, source: SourceSpec, request_id: RequestId) -> SourceCapture:
        return SourceCapture(request_id, 0, source, ("th",))

    def cancel(self, request_id: RequestId) -> None:
        self.cancelled.append(request_id)
        if self.callback is not None:
            self.callback()


class Executor:
    def __init__(self, outcome: object = "accepted") -> None:
        self.outcome = outcome
        self.calls: list[RunIdentity] = []
        self.configurations: list[object] = []
        self.closed: list[RunIdentity] = []
        self.callback = None

    def start(self, configuration, source, run_identity, admission):
        self.calls.append(run_identity)
        self.configurations.append(configuration)
        if self.outcome == "raise":
            raise RuntimeError("executor primary")
        if self.outcome == "failed":
            return ExecutorStartFailed(run_identity, CleanupStatus.CLEANUP_PENDING)
        if self.outcome == "cleaned":
            return ExecutorStartFailed(run_identity, CleanupStatus.CLEANED)
        if self.outcome == "foreign":
            return ExecutorAccepted(RunIdentity(run_identity.generation + 1, run_identity.fingerprint))
        if self.outcome == "equal":
            return ExecutorAccepted(RunIdentity(run_identity.generation, run_identity.fingerprint))
        if self.outcome == "bomb":
            return ExecutorAccepted(EqualityBomb())
        if self.outcome == "unknown":
            return object()
        return ExecutorAccepted(run_identity)

    def close(self, run_identity: RunIdentity) -> ExecutorClosed:
        self.closed.append(run_identity)
        if self.callback is not None:
            self.callback()
        return ExecutorClosed(run_identity, CleanupStatus.CLEANED)

    def pause(self, run_identity: RunIdentity) -> None: ...
    def resume(self, run_identity: RunIdentity) -> None: ...
    def stop(self, run_identity: RunIdentity) -> None: ...


class EqualityBomb:
    def __eq__(self, other: object) -> bool:
        raise RuntimeError("untrusted equality")


class DispatchFault(ScatteringCoordinator):
    def __init__(self, operation: str, timing: str) -> None:
        super().__init__()
        self.operation = operation
        self.timing = timing

    def _fault(self, operation: str, call, event):
        if operation != self.operation:
            return call(event)
        if self.timing == "before":
            raise RuntimeError(f"{operation} before")
        result = call(event)
        raise RuntimeError(f"{operation} after")

    def executor_accepted(self, event):
        return self._fault("accepted", super().executor_accepted, event)

    def executor_start_failed(self, event):
        return self._fault("failed", super().executor_start_failed, event)


class RejectedFailure(ScatteringCoordinator):
    def executor_start_failed(self, event):
        return LifecycleResult(
            LifecycleStatus.REJECTED,
            self.phase,
            run_identity=self.attempt_run_identity,
            error=LifecycleError.ILLEGAL_TRANSITION,
        )


class CountingDispatch(ScatteringCoordinator):
    def __init__(self) -> None:
        super().__init__()
        self.accepted_events = 0

    def executor_accepted(self, event):
        self.accepted_events += 1
        return super().executor_accepted(event)


class OrderedExecutor(Executor):
    def __init__(self, lifecycle: ScatteringCoordinator) -> None:
        super().__init__()
        self.lifecycle = lifecycle
        self.observed_phases: list[RunPhase] = []

    def start(self, configuration, source, run_identity, admission):
        self.observed_phases.append(self.lifecycle.phase)
        return super().start(configuration, source, run_identity, admission)


class OwnersClosedFault(ScatteringCoordinator):
    def __init__(self, timing: str) -> None:
        super().__init__()
        self.timing = timing

    def owners_closed(self, event):
        if self.timing == "before":
            raise RuntimeError("owners closed before")
        result = super().owners_closed(event)
        raise RuntimeError("owners closed after")


class MalformedLifecycle(ScatteringCoordinator):
    def __init__(self, operation: str) -> None:
        super().__init__()
        self.operation = operation

    def begin_start(self):
        return object() if self.operation == "begin" else super().begin_start()

    def preflight_refused(self, event):
        return object() if self.operation == "refusal" else super().preflight_refused(event)

    def preflight_accepted(self, event):
        return object() if self.operation == "promotion" else super().preflight_accepted(event)

    def executor_accepted(self, event):
        return object() if self.operation == "accepted" else super().executor_accepted(event)

    def executor_start_failed(self, event):
        return object() if self.operation == "failed" else super().executor_start_failed(event)

    def owners_closed(self, event):
        return object() if self.operation == "owners_closed" else super().owners_closed(event)

    def close(self):
        return object() if self.operation == "close" else super().close()


class MalformedBeginRequest(ScatteringCoordinator):
    def begin_start(self):
        return LifecycleResult(LifecycleStatus.APPLIED, RunPhase.PREPARING, request_id="bad")


def _pipeline(
    lifecycle: ScatteringCoordinator | None = None,
    *,
    source_present: bool = True,
    executor: Executor | None = None,
) -> tuple[StartPipeline, ScatteringCoordinator, Source, Executor]:
    source = Source()
    port = executor or Executor()
    coordinator = lifecycle or ScatteringCoordinator()
    return (
        StartPipeline(
            intents=RunIntentStore(_intent(source=source_present)),
            lifecycle=coordinator,
            sources=source,
            executor=port,
        ),
        coordinator,
        source,
        port,
    )


def _capture(pipeline: StartPipeline) -> StartCapture:
    capture = pipeline.begin()
    assert isinstance(capture, StartCapture)
    return capture


@pytest.mark.parametrize("outcome", ["raise", "foreign", "equal", "bomb", "unknown"])
def test_untrusted_executor_results_preserve_one_expected_failed_attempt(outcome):
    pipeline, lifecycle, _, executor = _pipeline(executor=Executor(outcome))

    failed = pipeline.start(admission_for(_capture(pipeline)))

    assert isinstance(failed, StartFailed)
    assert failed.configuration is executor.configurations[0]
    assert failed.run_identity is executor.calls[0]
    assert failed.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert lifecycle.phase is RunPhase.FAILED
    assert lifecycle.closed is False
    assert lifecycle.attempt_run_identity is failed.run_identity
    assert pipeline.owners_closed(OwnersClosed(failed.run_identity)).status is LifecycleStatus.APPLIED


@pytest.mark.parametrize("operation", ["accepted", "failed"])
@pytest.mark.parametrize("timing", ["before", "after"])
def test_dispatch_faults_after_executor_keep_cleanup_identity(operation, timing):
    executor = Executor("accepted" if operation == "accepted" else "failed")
    pipeline, lifecycle, _, _ = _pipeline(DispatchFault(operation, timing), executor=executor)

    failed = pipeline.start(admission_for(_capture(pipeline)))

    assert isinstance(failed, StartFailed)
    assert failed.run_identity is executor.calls[0]
    assert failed.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert lifecycle.phase is RunPhase.FAILED
    assert lifecycle.closed is False
    assert lifecycle.reset().status is LifecycleStatus.REJECTED
    assert pipeline.owners_closed(OwnersClosed(failed.run_identity)).status is LifecycleStatus.APPLIED


def test_rejected_synthetic_failure_dispatch_uses_canonical_post_executor_containment():
    pipeline, lifecycle, _, executor = _pipeline(RejectedFailure(), executor=Executor("raise"))

    failed = pipeline.start(admission_for(_capture(pipeline)))

    assert isinstance(failed, StartFailed)
    assert failed.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert failed.run_identity is executor.calls[0]
    assert lifecycle.phase is RunPhase.FAILED
    assert lifecycle.closed is False
    assert pipeline.owners_closed(OwnersClosed(failed.run_identity)).status is LifecycleStatus.APPLIED


def test_foreign_executor_identity_is_rejected_before_lifecycle_dispatch():
    lifecycle = CountingDispatch()
    pipeline, _, _, _ = _pipeline(lifecycle, executor=Executor("foreign"))

    failed = pipeline.start(admission_for(_capture(pipeline)))

    assert isinstance(failed, StartFailed)
    assert lifecycle.accepted_events == 0


def test_executor_is_called_once_and_only_after_preflight_promotion():
    lifecycle = ScatteringCoordinator()
    executor = OrderedExecutor(lifecycle)
    pipeline, _, _, _ = _pipeline(lifecycle, executor=executor)

    launched = pipeline.start(admission_for(_capture(pipeline)))

    assert not isinstance(launched, StartFailed)
    assert executor.observed_phases == [RunPhase.STARTING]


@pytest.mark.parametrize("timing", ["before", "after"])
def test_immediate_owners_closed_fault_keeps_original_failure_and_pending_cleanup(timing):
    pipeline, lifecycle, source, executor = _pipeline(OwnersClosedFault(timing), executor=Executor("cleaned"))
    source.cancel = lambda request_id: (_ for _ in ()).throw(RuntimeError("source cleanup"))

    failed = pipeline.start(admission_for(_capture(pipeline)))

    assert isinstance(failed, StartFailed)
    assert failed.reason.value == "executor_start_failed"
    assert failed.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert failed.run_identity is executor.calls[0]
    assert lifecycle.phase is RunPhase.FAILED
    assert lifecycle.closed is False
    assert [item.operation for item in failed.recovery_failures] == ["source.cancel", "lifecycle.owners_closed"]


@pytest.mark.parametrize(
    ("operation", "source_present", "outcome"),
    [
        ("begin", True, "accepted"),
        ("refusal", False, "accepted"),
        ("promotion", True, "accepted"),
        ("accepted", True, "accepted"),
        ("failed", True, "failed"),
        ("owners_closed", True, "cleaned"),
        ("close", True, "accepted"),
    ],
)
def test_malformed_lifecycle_values_are_diagnostic_only(operation, source_present, outcome):
    pipeline, _, _, _ = _pipeline(MalformedLifecycle(operation), source_present=source_present, executor=Executor(outcome))

    if operation == "begin":
        result = pipeline.begin()
    elif operation == "close":
        _capture(pipeline)
        result = pipeline.close()
    else:
        result = pipeline.start(admission_for(_capture(pipeline))) if source_present else pipeline.begin()

    assert isinstance(result.lifecycle_result, LifecycleResult)


def test_malformed_begin_request_and_source_observation_inputs_are_inert_or_contained():
    pipeline, _, source, _ = _pipeline(MalformedBeginRequest())
    failed = pipeline.begin()
    assert isinstance(failed, StartFailed)
    assert isinstance(failed.lifecycle_result, LifecycleResult)
    assert source.cancelled == []

    pipeline, lifecycle, source, _ = _pipeline()
    capture = _capture(pipeline)
    for epoch in ("bad", -1, False):
        result = pipeline.source_observation_changed(capture.request_id, epoch)
        assert isinstance(result, StartRejected)
        assert result.reason is StartRejection.SOURCE_OBSERVATION
        assert lifecycle.phase is RunPhase.PREPARING
        assert source.cancelled == []

    same_epoch = pipeline.source_observation_changed(capture.request_id, 0)
    assert isinstance(same_epoch, StartRejected)
    assert same_epoch.reason is StartRejection.SOURCE_OBSERVATION
    assert lifecycle.phase is RunPhase.PREPARING


def test_owners_closed_rejects_malformed_boundary_values_without_lifecycle_mutation():
    pipeline, lifecycle, _, _ = _pipeline()
    capture = _capture(pipeline)

    result = pipeline.owners_closed(object())

    assert isinstance(result, LifecycleResult)
    assert result.status is LifecycleStatus.REJECTED
    assert lifecycle.phase is RunPhase.PREPARING
    assert capture.request_id is lifecycle.request_id


def test_close_callbacks_observe_invalidated_start_tokens_before_external_cleanup():
    pipeline, lifecycle, source, executor = _pipeline()
    capture = _capture(pipeline)
    observed = []
    tokens = []
    source.callback = lambda: (tokens.append((pipeline._capture, pipeline._recapture)), observed.append(pipeline.start(capture)))

    pipeline.close()

    assert lifecycle.closed is True
    assert observed and observed[0].reason is StartRejection.STALE_CAPTURE
    assert tokens == [(None, None)]
    assert executor.calls == []

    pipeline, lifecycle, _, executor = _pipeline()
    launched = pipeline.start(admission_for(_capture(pipeline)))
    assert not isinstance(launched, StartFailed)
    executor.callback = lambda: observed.append(pipeline.begin())

    pipeline.close()

    assert lifecycle.closed is True
    assert observed[-1].reason is StartRejection.LIFECYCLE
