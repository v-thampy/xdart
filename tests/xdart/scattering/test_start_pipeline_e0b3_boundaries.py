"""Total-boundary adversaries for the E0b Start owner."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.session.intent_store import (
    IntentCommitAccepted,
    IntentRecaptureRequired,
    RunIntentStore,
)
from xrd_tools.session.run_configuration import GIIntent, RunIntent
from xdart.gui.tabs.scattering.contracts import SourceCapture
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.events import (
    ExecutorAccepted,
    LifecycleResult,
    LifecycleStatus,
    OwnersClosed,
    RequestId,
    RunIdentity,
)
from xdart.gui.tabs.scattering.start_outcomes import (
    StartCapture,
    StartFailed,
    StartLaunched,
    StartRecaptureRequired,
    StartRefusal,
    StartRefused,
)
from xdart.gui.tabs.scattering.start_pipeline import StartPipeline
from xdart.gui.tabs.scattering.state_machine import RunPhase
from tests.xdart.scattering._admission import admission_for


def _intent() -> RunIntent:
    return RunIntent(
        source_spec=SourceSpec(Path("/data/frame.tif"), SourceKind.IMAGE_FILE),
        gi=GIIntent(enabled=True, incidence_motor="th", th_val=0.2),
    )


class Source:
    def __init__(self, *, choices: object = ("th",)) -> None:
        self.choices = choices
        self.cancelled = []

    def capture(self, source, request_id):
        return SourceCapture(request_id, 0, source, self.choices)

    def cancel(self, request_id):
        self.cancelled.append(request_id)


class Executor:
    def __init__(self, *, error: Exception | None = None, lifecycle=None) -> None:
        self.error = error
        self.lifecycle = lifecycle
        self.calls = []
        self.closed = []
        self.coordinator_owned = []

    def start(self, configuration, source, run_identity, admission):
        self.calls.append(run_identity)
        if self.lifecycle is not None:
            self.coordinator_owned.append(run_identity is self.lifecycle.attempt_run_identity)
        if self.error is not None:
            raise self.error
        return ExecutorAccepted(run_identity)

    def close(self, run_identity):
        self.closed.append(run_identity)

    def pause(self, run_identity): ...
    def resume(self, run_identity): ...
    def stop(self, run_identity): ...


def _pipeline(*, store=None, lifecycle=None, source=None, executor=None):
    lifecycle = lifecycle or ScatteringCoordinator()
    source = source or Source()
    executor = executor or Executor()
    return (
        StartPipeline(
            intents=store or RunIntentStore(_intent()),
            lifecycle=lifecycle,
            sources=source,
            executor=executor,
        ),
        lifecycle,
        source,
        executor,
    )


def _capture(pipeline) -> StartCapture:
    result = pipeline.begin()
    assert type(result) is StartCapture
    return result


class BadCommitSnapshotStore(RunIntentStore):
    def commit(self, candidate, *, expected_revision):
        return IntentCommitAccepted(1, object())


class BadRecaptureStore(RunIntentStore):
    def commit(self, candidate, *, expected_revision):
        return IntentRecaptureRequired(999, object())


@pytest.mark.parametrize("store", [BadCommitSnapshotStore(_intent()), BadRecaptureStore(_intent())])
def test_malformed_commit_results_are_contained_before_their_fields_are_used(store):
    pipeline, lifecycle, source, _ = _pipeline(store=store)
    capture = _capture(pipeline)

    result = pipeline.apply_operator_decision(capture, capture.intent_snapshot.thaw())

    assert type(result) is StartFailed
    assert not isinstance(result, StartRecaptureRequired)
    assert lifecycle.closed is True
    assert source.cancelled == [capture.request_id]


class MismatchedRecaptureStore(RunIntentStore):
    def commit(self, candidate, *, expected_revision):
        return IntentRecaptureRequired(expected_revision + 1, self.snapshot())


class UnadvancedRecaptureStore(RunIntentStore):
    def commit(self, candidate, *, expected_revision):
        return IntentRecaptureRequired(expected_revision, self.snapshot())


@pytest.mark.parametrize("store", [MismatchedRecaptureStore(_intent()), UnadvancedRecaptureStore(_intent())])
def test_recapture_result_must_preserve_its_expected_revision_and_advance_snapshot(store):
    pipeline, lifecycle, source, _ = _pipeline(store=store)
    capture = _capture(pipeline)

    result = pipeline.apply_operator_decision(capture, capture.intent_snapshot.thaw())

    assert type(result) is StartFailed
    assert lifecycle.closed is True
    assert source.cancelled == [capture.request_id]


class InvalidContainment(ScatteringCoordinator):
    def contain_executor_failure(self, run_identity):
        return object()


class RaisingContainment(ScatteringCoordinator):
    def contain_executor_failure(self, run_identity):
        raise StringBomb()


@pytest.mark.parametrize("lifecycle", [InvalidContainment(), RaisingContainment()])
def test_invalid_post_executor_containment_is_diagnostic_and_still_releases_owners(lifecycle):
    pipeline, lifecycle, source, executor = _pipeline(
        lifecycle=lifecycle,
        executor=Executor(error=RuntimeError("primary executor fault")),
    )
    capture = _capture(pipeline)

    failed = pipeline.start(admission_for(capture))

    assert type(failed) is StartFailed
    assert type(failed.lifecycle_result) is LifecycleResult
    assert failed.lifecycle_result.status is LifecycleStatus.REJECTED
    assert lifecycle.phase is RunPhase.FAILED
    assert source.cancelled == [capture.request_id]
    assert executor.closed == executor.calls
    assert pipeline.start(capture).reason.value == "stale_capture"


class StringBomb(RuntimeError):
    def __str__(self):
        raise RuntimeError("diagnostic formatting escaped")


class RaisingSource(Source):
    def capture(self, source, request_id):
        raise StringBomb()


class RaisingCancelSource(Source):
    def cancel(self, request_id):
        super().cancel(request_id)
        raise StringBomb()


@pytest.mark.parametrize("source", [RaisingSource(), RaisingCancelSource()])
def test_unprintable_primary_or_cleanup_exceptions_never_escape_or_skip_cleanup(source):
    executor = Executor(error=StringBomb()) if type(source) is RaisingCancelSource else Executor()
    pipeline, lifecycle, _, _ = _pipeline(source=source, executor=executor)

    result = (
        pipeline.begin()
        if type(source) is RaisingSource
        else pipeline.start(admission_for(_capture(pipeline)))
    )

    assert not isinstance(result, StartCapture)
    assert lifecycle.phase in {RunPhase.IDLE, RunPhase.FAILED}
    if type(source) is RaisingCancelSource:
        assert executor.closed == executor.calls


class EqualSourceImpostor:
    def __eq__(self, other):
        return True


class ImpostorSource(Source):
    def capture(self, source, request_id):
        return SourceCapture(request_id, 0, EqualSourceImpostor(), ("th",))


class TupleSubclass(tuple):
    pass


class StringSubclass(str):
    pass


@pytest.mark.parametrize(
    "source",
    [
        ImpostorSource(),
        Source(choices=TupleSubclass(("th",))),
        Source(choices=(StringSubclass("th"),)),
    ],
)
def test_source_capture_requires_exact_source_and_builtin_choice_values(source):
    pipeline, lifecycle, _, executor = _pipeline(source=source)

    result = pipeline.begin()

    assert type(result) is StartRefused
    assert result.reason is StartRefusal.SOURCE_CAPTURE_INVALID
    assert lifecycle.phase is RunPhase.IDLE
    assert executor.calls == []


class ClonePromotionIdentity(ScatteringCoordinator):
    def preflight_accepted(self, event):
        result = super().preflight_accepted(event)
        identity = result.run_identity
        assert identity is not None
        return replace(result, run_identity=RunIdentity(identity.generation, identity.fingerprint))


def test_equal_but_distinct_promotion_identity_is_rejected_before_executor_dispatch():
    lifecycle = ClonePromotionIdentity()
    executor = Executor(lifecycle=lifecycle)
    pipeline, _, _, _ = _pipeline(lifecycle=lifecycle, executor=executor)

    result = pipeline.start(admission_for(_capture(pipeline)))

    assert not isinstance(result, StartLaunched)
    assert executor.calls == []
    assert executor.coordinator_owned == []


def test_executor_callback_receives_the_exact_coordinator_owned_identity():
    lifecycle = ScatteringCoordinator()
    executor = Executor(lifecycle=lifecycle)
    pipeline, _, _, _ = _pipeline(lifecycle=lifecycle, executor=executor)

    result = pipeline.start(admission_for(_capture(pipeline)))

    assert type(result) is StartLaunched
    assert executor.coordinator_owned == [True]


class CountingOwnersClosed(ScatteringCoordinator):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def owners_closed(self, event):
        self.calls += 1
        return super().owners_closed(event)


@pytest.mark.parametrize("event", [OwnersClosed(None), object()])
def test_invalid_owners_closed_values_are_rejected_before_lifecycle_dispatch(event):
    lifecycle = CountingOwnersClosed()
    pipeline, _, _, _ = _pipeline(lifecycle=lifecycle)
    _capture(pipeline)

    result = pipeline.owners_closed(event)

    assert result.status is LifecycleStatus.REJECTED
    assert lifecycle.calls == 0


class BooleanRequestLifecycle(ScatteringCoordinator):
    def begin_start(self):
        return LifecycleResult(LifecycleStatus.APPLIED, RunPhase.PREPARING, RequestId(True))


def test_boolean_request_ids_are_not_accepted_as_lifecycle_values():
    pipeline, lifecycle, source, _ = _pipeline(lifecycle=BooleanRequestLifecycle())

    failed = pipeline.begin()

    assert type(failed) is StartFailed
    assert lifecycle.closed is True
    assert source.cancelled == []
