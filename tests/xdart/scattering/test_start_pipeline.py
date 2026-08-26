from __future__ import annotations

from pathlib import Path
import inspect
from typing import get_type_hints

import pytest

from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.session.intent_store import IntentCommitAccepted, RunIntentStore
from xrd_tools.session.run_configuration import GIIntent, RunIntent
from xdart.gui.tabs.scattering.contracts import AdmissionReceipt, SourceCapture
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.events import (
    CleanupStatus,
    ExecutorAccepted,
    ExecutorStartFailed,
    OwnersClosed,
    RequestId,
    RunIdentity,
)
from xdart.gui.tabs.scattering.start_outcomes import (
    ExceptionDetail,
    RecoveryFailure,
    StartCapture,
    StartFailureKind,
    StartLaunched,
    StartRecaptureCause,
    StartRecaptureRequired,
    StartRefusal,
    StartRefused,
    StartRejected,
)
from xdart.gui.tabs.scattering.start_pipeline import StartPipeline
from xdart.gui.tabs.scattering.state_machine import RunPhase
from tests.xdart.scattering._admission import admission_for


def _source() -> SourceSpec:
    return SourceSpec(Path("/data/frame_0001.tif"), SourceKind.IMAGE_FILE)


def _intent(**changes: object) -> RunIntent:
    values: dict[str, object] = {
        "source_spec": _source(),
        "gi": GIIntent(enabled=True, incidence_motor="th", th_val=0.2),
    }
    values.update(changes)
    return RunIntent(**values)


class RecordingSource:
    def __init__(self, *, choices: tuple[str, ...] | None = ("th",)) -> None:
        self.choices = choices
        self.epoch = 0
        self.captures: list[SourceCapture] = []
        self.cancelled: list[RequestId] = []
        self.error: Exception | None = None
        self.callback: object | None = None
        self.returned_request: RequestId | None = None
        self.returned_source: SourceSpec | None = None
        self.returned_value: object | None = None
        self.cancel_error: Exception | None = None

    def capture(self, source: SourceSpec, request_id: RequestId) -> SourceCapture:
        if self.error is not None:
            raise self.error
        if self.returned_value is not None:
            return self.returned_value
        capture = SourceCapture(
            self.returned_request or request_id,
            self.epoch,
            self.returned_source or source,
            self.choices,
        )
        self.captures.append(capture)
        return capture

    def cancel(self, request_id: RequestId) -> None:
        self.cancelled.append(request_id)
        if self.callback is not None:
            self.callback()
        if self.cancel_error is not None:
            raise self.cancel_error


class RecordingExecutor:
    def __init__(self, outcome: str = "accepted") -> None:
        self.outcome = outcome
        self.calls: list[tuple[object, SourceCapture, RunIdentity]] = []
        self.closed: list[RunIdentity] = []
        self.callback: object | None = None
        self.close_error: Exception | None = None

    def start(self, configuration, source, run_identity, admission):
        self.calls.append((configuration, source, run_identity))
        if self.outcome == "raise":
            raise RuntimeError("executor boom")
        if self.outcome == "foreign":
            return ExecutorAccepted(RunIdentity(run_identity.generation + 1, run_identity.fingerprint))
        if self.outcome == "pending":
            return ExecutorStartFailed(run_identity, CleanupStatus.CLEANUP_PENDING)
        if self.outcome == "malformed_failure":
            return ExecutorStartFailed(run_identity, "bad-cleanup", object())
        if self.outcome == "cleaned":
            return ExecutorStartFailed(run_identity, CleanupStatus.CLEANED)
        return ExecutorAccepted(run_identity)

    def pause(self, run_identity: RunIdentity) -> None:
        pass

    def resume(self, run_identity: RunIdentity) -> None:
        pass

    def stop(self, run_identity: RunIdentity) -> None:
        pass

    def close(self, run_identity: RunIdentity) -> None:
        self.closed.append(run_identity)
        if self.callback is not None:
            self.callback()
        if self.close_error is not None:
            raise self.close_error


class TracingStore(RunIntentStore):
    def __init__(self, initial: RunIntent) -> None:
        super().__init__(initial)
        self.freeze_inputs: list[tuple[int, tuple[str, ...] | None]] = []

    def freeze(self, *, expected_revision: int, gi_motor_choices=None):
        self.freeze_inputs.append((expected_revision, gi_motor_choices))
        return super().freeze(
            expected_revision=expected_revision,
            gi_motor_choices=gi_motor_choices,
        )


def _pipeline(
    *,
    intent: RunIntent | None = None,
    source: RecordingSource | None = None,
    executor: RecordingExecutor | None = None,
    store: RunIntentStore | None = None,
) -> tuple[StartPipeline, RunIntentStore, ScatteringCoordinator, RecordingSource, RecordingExecutor]:
    real_store = store or RunIntentStore(intent or _intent())
    real_source = source or RecordingSource()
    real_executor = executor or RecordingExecutor()
    lifecycle = ScatteringCoordinator()
    return (
        StartPipeline(
            intents=real_store,
            lifecycle=lifecycle,
            sources=real_source,
            executor=real_executor,
        ),
        real_store,
        lifecycle,
        real_source,
        real_executor,
    )


def _admit(capture: StartCapture) -> AdmissionReceipt:
    return admission_for(capture)


def test_straight_start_uses_real_store_and_exact_frozen_configuration_identity():
    pipeline, store, lifecycle, source, executor = _pipeline()

    capture = pipeline.begin()
    assert isinstance(capture, StartCapture)
    launched = pipeline.start(_admit(capture))

    assert isinstance(launched, StartLaunched)
    assert len(executor.calls) == 1
    configuration, passed_capture, identity = executor.calls[0]
    assert configuration is launched.configuration
    assert passed_capture is capture.source_capture is launched.source_capture
    assert identity is launched.run_identity
    assert identity is lifecycle.active_run_identity
    assert identity == RunIdentity.from_configuration(launched.configuration)
    assert lifecycle.phase is RunPhase.RUNNING
    assert store.revision == 0
    assert store.snapshot().thaw().generation == 1
    assert len(source.captures) == len(source.cancelled) == 1


def test_accepted_output_decision_preserves_qualified_unchanged_source():
    pipeline, store, _, source, _ = _pipeline()
    capture = pipeline.begin()
    assert isinstance(capture, StartCapture)
    candidate = capture.intent_snapshot.thaw()
    candidate.output_mode = "Overwrite"

    replacement = pipeline.apply_operator_decision(capture, candidate)

    assert isinstance(replacement, StartCapture)
    assert replacement.intent_snapshot.revision == store.revision == 1
    assert replacement.capture_sequence == capture.capture_sequence + 1
    assert len(source.captures) == 1
    assert replacement.source_capture is capture.source_capture
    assert isinstance(pipeline.start(_admit(capture)), StartRejected)


def test_r1_capture_does_not_silently_adopt_unrelated_r2_before_freeze():
    pipeline, store, _, _, executor = _pipeline()
    capture = pipeline.begin()
    assert isinstance(capture, StartCapture)
    candidate = capture.intent_snapshot.thaw()
    candidate.output_mode = "Overwrite"
    r1_capture = pipeline.apply_operator_decision(capture, candidate)
    assert isinstance(r1_capture, StartCapture)
    unrelated = store.snapshot().thaw()
    unrelated.processing_mode = "Int 1D"
    store.commit(unrelated, expected_revision=1)

    ticket = pipeline.start(_admit(r1_capture))

    assert isinstance(ticket, StartRecaptureRequired)
    assert ticket.current_snapshot.revision == 2
    assert store.snapshot().thaw().generation == 0
    assert executor.calls == []


def test_commit_race_preserves_intervening_edit_and_returns_canonical_recapture_ticket():
    pipeline, store, lifecycle, _, executor = _pipeline()
    capture = pipeline.begin()
    assert isinstance(capture, StartCapture)
    intervening = capture.intent_snapshot.thaw()
    intervening.output_mode = "Overwrite"
    store.commit(intervening, expected_revision=0)

    result = pipeline.apply_operator_decision(capture, capture.intent_snapshot.thaw())

    assert isinstance(result, StartRecaptureRequired)
    assert result.cause is StartRecaptureCause.COMMIT_RACE
    assert result.current_snapshot.revision == 1
    assert store.snapshot().thaw().output_mode == "Overwrite"
    assert lifecycle.phase is RunPhase.PREPARING
    assert store.snapshot().thaw().generation == 0
    assert executor.calls == []


def test_operator_canonicalization_conflict_surfaces_the_race_without_adopting():
    """Bounded rework 2026-08-04: an unrelated edit landing BETWEEN the
    accepted operator commit (r1) and its canonicalization commit must
    surface a COMMIT_RACE carrying the exact current snapshot — never be
    consumed into a canonical r3 replacement capture or a source
    recapture (the no-silent-adoption contract)."""

    class InterleavingStore(RunIntentStore):
        # A REAL store whose next accepted commit is immediately followed
        # by an unrelated writer's commit — the exact single-threaded
        # window between apply_operator_decision's operator commit and
        # the canonicalization commit.
        def arm(self) -> None:
            self._armed = True

        def commit(self, candidate, *, expected_revision):
            result = super().commit(candidate, expected_revision=expected_revision)
            if getattr(self, "_armed", False) and isinstance(result, IntentCommitAccepted):
                self._armed = False
                unrelated = super().snapshot().thaw()
                unrelated.processing_mode = "Int 1D"
                injected = super().commit(
                    unrelated, expected_revision=result.snapshot.revision
                )
                assert isinstance(injected, IntentCommitAccepted)
            return result

    pipeline, store, lifecycle, source, executor = _pipeline(
        store=InterleavingStore(_intent())
    )
    capture = pipeline.begin()
    assert isinstance(capture, StartCapture)
    candidate = capture.intent_snapshot.thaw()
    candidate.threshold.apply_threshold = True
    candidate.threshold.threshold_min = None
    # The displayed manual lower bound must materialize in a second commit.
    store.arm()

    result = pipeline.apply_operator_decision(capture, candidate)

    assert isinstance(result, StartRecaptureRequired)
    assert result.cause is StartRecaptureCause.COMMIT_RACE
    # The ticket carries the EXACT current snapshot: the unrelated
    # writer's r2, not a canonicalized r3.
    assert result.current_snapshot.revision == 2
    assert store.revision == 2
    assert result.current_snapshot.thaw().processing_mode == "Int 1D"
    # No replacement capture and no source recapture were created.
    assert len(source.captures) == 1
    assert executor.calls == []
    assert lifecycle.phase is RunPhase.PREPARING


def test_freeze_race_uses_ticket_snapshot_revision_and_requires_explicit_recapture():
    pipeline, store, lifecycle, source, executor = _pipeline()
    capture = pipeline.begin()
    assert isinstance(capture, StartCapture)
    edit = capture.intent_snapshot.thaw()
    edit.output_mode = "Overwrite"
    store.commit(edit, expected_revision=0)

    ticket = pipeline.start(_admit(capture))
    assert isinstance(ticket, StartRecaptureRequired)
    assert ticket.cause is StartRecaptureCause.FREEZE_RACE
    assert ticket.current_snapshot.revision == 1
    assert ticket.superseded_capture_sequence == capture.capture_sequence
    assert store.snapshot().thaw().generation == 0
    assert executor.calls == []

    retry = pipeline.recapture(ticket)
    assert isinstance(retry, StartCapture)
    assert retry.intent_snapshot is ticket.current_snapshot
    assert len(source.captures) == 2
    assert isinstance(pipeline.start(_admit(retry)), StartLaunched)
    assert lifecycle.phase is RunPhase.RUNNING


@pytest.mark.parametrize("choices", [None, (), ("eta",)])
def test_source_choices_are_preserved_without_normalization(choices):
    source = RecordingSource(choices=choices)
    store = TracingStore(_intent())
    pipeline, _, lifecycle, _, executor = _pipeline(source=source, store=store)

    capture = pipeline.begin()
    assert isinstance(capture, StartCapture)
    outcome = pipeline.start(_admit(capture))

    if choices is None:
        # No admitted catalog: the explicit motor passes through unvalidated.
        assert isinstance(outcome, StartLaunched)
    else:
        # Strict GI admission refuses: 'th' is absent from the catalog.
        assert isinstance(outcome, StartRefused)
        assert outcome.reason is StartRefusal.FREEZE_VALIDATION
        assert "admitted motor catalog" in outcome.detail
        assert lifecycle.phase is RunPhase.IDLE
        assert executor.calls == []

    assert store.freeze_inputs == [(0, choices)]


@pytest.mark.parametrize("choices", [["th"], ("th", 1)])
def test_malformed_source_choices_are_visible_preflight_refusals(choices):
    pipeline, store, lifecycle, source, executor = _pipeline(source=RecordingSource(choices=choices))

    refused = pipeline.begin()

    assert isinstance(refused, StartRefused)
    assert refused.reason is StartRefusal.SOURCE_CAPTURE_INVALID
    assert lifecycle.phase is RunPhase.IDLE
    assert store.snapshot().thaw().generation == 0
    assert executor.calls == []
    assert source.cancelled == [refused.request_id]


def test_invalid_freeze_is_typed_refusal_with_structured_exception_and_contiguous_retry():
    pipeline, store, lifecycle, source, executor = _pipeline(intent=_intent(live_mode=True, batch_mode=True))
    capture = pipeline.begin()
    assert isinstance(capture, StartCapture)

    refused = pipeline.start(_admit(capture))

    assert isinstance(refused, StartRefused)
    assert refused.reason is StartRefusal.FREEZE_VALIDATION
    assert refused.exception == ExceptionDetail("builtins", "ValueError", refused.detail)
    assert lifecycle.phase is RunPhase.IDLE
    assert store.snapshot().thaw().generation == 0
    assert executor.calls == []
    assert source.cancelled == [refused.request_id]


def test_source_capture_value_error_is_not_mapped_as_freeze_validation():
    source = RecordingSource()
    source.error = ValueError("source unavailable")
    pipeline, _, lifecycle, _, executor = _pipeline(source=source)

    refused = pipeline.begin()

    assert isinstance(refused, StartRefused)
    assert refused.reason is StartRefusal.SOURCE_CAPTURE_EXCEPTION
    assert refused.exception == ExceptionDetail("builtins", "ValueError", "source unavailable")
    assert lifecycle.phase is RunPhase.IDLE
    assert executor.calls == []


def test_unknown_source_capture_result_is_a_visible_preflight_refusal():
    source = RecordingSource()
    source.returned_value = object()
    pipeline, _, lifecycle, _, executor = _pipeline(source=source)

    refused = pipeline.begin()

    assert isinstance(refused, StartRefused)
    assert refused.reason is StartRefusal.SOURCE_CAPTURE_INVALID
    assert lifecycle.phase is RunPhase.IDLE
    assert executor.calls == []


@pytest.mark.parametrize("fault", ["request", "source"])
def test_request_or_source_mismatched_capture_is_refused_without_freeze(fault):
    source = RecordingSource()
    if fault == "request":
        source.returned_request = RequestId(99)
    else:
        source.returned_source = SourceSpec(Path("/other/frame.tif"), SourceKind.IMAGE_FILE)
    pipeline, store, lifecycle, _, executor = _pipeline(source=source)

    refused = pipeline.begin()

    assert isinstance(refused, StartRefused)
    assert refused.reason is StartRefusal.SOURCE_CAPTURE_INVALID
    assert lifecycle.phase is RunPhase.IDLE
    assert store.snapshot().thaw().generation == 0
    assert executor.calls == []


def test_changed_source_requires_a_later_observation_epoch():
    source = RecordingSource()
    pipeline, _, lifecycle, _, executor = _pipeline(source=source)
    capture = pipeline.begin()
    assert isinstance(capture, StartCapture)
    candidate = capture.intent_snapshot.thaw()
    candidate.source_spec = SourceSpec(Path("/data/frame_0002.tif"), SourceKind.IMAGE_FILE)

    refused = pipeline.apply_operator_decision(capture, candidate)

    assert isinstance(refused, StartRefused)
    assert refused.reason is StartRefusal.SOURCE_CAPTURE_INVALID
    assert lifecycle.phase is RunPhase.IDLE
    assert executor.calls == []


@pytest.mark.parametrize(
    ("outcome", "cleanup"),
    [("cleaned", CleanupStatus.CLEANED), ("pending", CleanupStatus.CLEANUP_PENDING)],
)
def test_executor_start_failure_preserves_exact_configuration_and_cleanup(outcome, cleanup):
    executor = RecordingExecutor(outcome)
    pipeline, _, lifecycle, _, _ = _pipeline(executor=executor)
    capture = pipeline.begin()
    assert isinstance(capture, StartCapture)

    failed = pipeline.start(_admit(capture))

    assert failed.configuration is executor.calls[0][0]
    assert failed.run_identity == executor.calls[0][2]
    assert failed.cleanup_status is cleanup
    assert failed.reason is StartFailureKind.EXECUTOR_START_FAILED
    assert lifecycle.phase is RunPhase.FAILED
    reset = lifecycle.reset()
    assert (reset.status.value == "applied") is (cleanup is CleanupStatus.CLEANED)


@pytest.mark.parametrize("outcome", ["foreign", "raise"])
def test_executor_invariant_failure_keeps_expected_attempt_failed_until_owners_close(outcome):
    executor = RecordingExecutor(outcome)
    pipeline, _, lifecycle, _, _ = _pipeline(executor=executor)
    capture = pipeline.begin()
    assert isinstance(capture, StartCapture)

    failed = pipeline.start(_admit(capture))

    assert failed.reason is StartFailureKind.EXECUTOR_INVARIANT
    assert failed.configuration is executor.calls[0][0]
    assert failed.run_identity == executor.calls[0][2]
    assert failed.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert executor.closed == [failed.run_identity]
    assert lifecycle.closed is False
    assert lifecycle.phase is RunPhase.FAILED
    assert lifecycle.attempt_run_identity == failed.run_identity
    assert lifecycle.reset().status.value == "rejected"
    assert pipeline.owners_closed(OwnersClosed(failed.run_identity)).status.value == "applied"
    assert lifecycle.reset().status.value == "applied"


def test_source_observation_advance_invalidates_old_choices_and_recaptures():
    source = RecordingSource(choices=("th",))
    store = TracingStore(_intent())
    pipeline, _, _, _, _ = _pipeline(source=source, store=store)
    capture = pipeline.begin()
    assert isinstance(capture, StartCapture)
    source.epoch = 1
    source.choices = ("eta",)

    ticket = pipeline.source_observation_changed(capture.request_id, 1)

    assert isinstance(ticket, StartRecaptureRequired)
    assert isinstance(pipeline.start(_admit(capture)), StartRejected)
    retry = pipeline.recapture(ticket)
    assert isinstance(retry, StartCapture)
    assert retry.source_capture.gi_motor_choices == ("eta",)
    # Strict GI admission refuses the recapture: 'th' is absent from it.
    outcome = pipeline.start(_admit(retry))
    assert isinstance(outcome, StartRefused)
    assert outcome.reason is StartRefusal.FREEZE_VALIDATION
    assert "admitted motor catalog" in outcome.detail
    assert store.freeze_inputs == [(0, ("eta",))]


def test_source_observation_ticket_refuses_a_recapture_below_its_announced_epoch():
    source = RecordingSource(choices=("old",))
    pipeline, store, lifecycle, _, executor = _pipeline(source=source)
    capture = pipeline.begin()
    assert isinstance(capture, StartCapture)

    ticket = pipeline.source_observation_changed(capture.request_id, 5)
    assert isinstance(ticket, StartRecaptureRequired)
    assert ticket.minimum_source_epoch == 5
    refused = pipeline.recapture(ticket)

    assert isinstance(refused, StartRefused)
    assert refused.reason is StartRefusal.SOURCE_CAPTURE_INVALID
    assert lifecycle.phase is RunPhase.IDLE
    assert store.snapshot().thaw().generation == 0
    assert executor.calls == []


def test_recapture_replaces_ticket_when_revision_advances_again_without_source_capture():
    pipeline, store, _, source, _ = _pipeline()
    capture = pipeline.begin()
    assert isinstance(capture, StartCapture)
    changed = capture.intent_snapshot.thaw()
    changed.output_mode = "Overwrite"
    store.commit(changed, expected_revision=0)
    ticket = pipeline.start(_admit(capture))
    assert isinstance(ticket, StartRecaptureRequired)
    changed_again = store.snapshot().thaw()
    changed_again.processing_mode = "Int 1D"
    store.commit(changed_again, expected_revision=1)

    replacement = pipeline.recapture(ticket)

    assert isinstance(replacement, StartRecaptureRequired)
    assert replacement.current_snapshot.revision == 2
    assert len(source.captures) == 1
    assert isinstance(pipeline.recapture(ticket), StartRejected)


def test_non_intent_decision_candidate_is_closed_invariant_not_validation_refusal():
    pipeline, _, lifecycle, _, executor = _pipeline()
    capture = pipeline.begin()
    assert isinstance(capture, StartCapture)

    failed = pipeline.apply_operator_decision(capture, "not an intent")

    assert failed.reason is StartFailureKind.CANONICAL_INVARIANT
    assert failed.configuration is None
    assert failed.run_identity is None
    assert failed.cleanup_status is CleanupStatus.NOT_STARTED
    assert lifecycle.closed is True
    assert executor.calls == []


def test_close_invalidates_before_reentrant_source_cancel_callback():
    source = RecordingSource()
    pipeline, _, lifecycle, _, _ = _pipeline(source=source)
    capture = pipeline.begin()
    assert isinstance(capture, StartCapture)

    observed: list[StartRejected] = []
    source.callback = lambda: observed.append(pipeline.begin())
    closed = pipeline.close()

    assert closed.lifecycle_result.phase is RunPhase.CLOSED
    assert lifecycle.closed is True
    assert len(observed) == 1
    assert observed[0].lifecycle_result is not None
    assert observed[0].lifecycle_result.phase is RunPhase.CLOSED


def test_close_invalidates_running_attempt_before_reentrant_executor_close_callback():
    executor = RecordingExecutor()
    pipeline, _, lifecycle, _, _ = _pipeline(executor=executor)
    capture = pipeline.begin()
    assert isinstance(capture, StartCapture)
    launched = pipeline.start(_admit(capture))
    assert isinstance(launched, StartLaunched)
    observed: list[StartRejected] = []
    executor.callback = lambda: observed.append(pipeline.begin())

    closed = pipeline.close()
    acknowledged = pipeline.owners_closed(OwnersClosed(launched.run_identity))

    assert closed.lifecycle_result.phase is RunPhase.STOPPING
    assert closed.lifecycle_result.run_identity is launched.run_identity
    assert lifecycle.closed is True
    assert executor.closed == [launched.run_identity]
    assert len(observed) == 1
    assert observed[0].lifecycle_result is not None
    assert observed[0].lifecycle_result.phase is RunPhase.STOPPING
    assert acknowledged.phase is RunPhase.CLOSED


def test_capture_sequence_restarts_at_one_for_each_new_preparing_request():
    pipeline, _, _, _, _ = _pipeline()
    first = pipeline.begin()
    assert isinstance(first, StartCapture)
    assert first.capture_sequence == 1
    assert isinstance(pipeline.refuse(first, StartRefusal.MISSING_SOURCE), StartRefused)

    second = pipeline.begin()

    assert isinstance(second, StartCapture)
    assert second.capture_sequence == 1
    assert second.request_id.value == first.request_id.value + 1


def test_consecutive_start_owners_keep_edit_revision_and_advance_generation_once_each():
    store = RunIntentStore(_intent())
    first_pipeline, _, _, _, _ = _pipeline(store=store)
    first_capture = first_pipeline.begin()
    assert isinstance(first_capture, StartCapture)
    first = first_pipeline.start(_admit(first_capture))
    assert isinstance(first, StartLaunched)

    second_pipeline, _, _, _, _ = _pipeline(store=store)
    second_capture = second_pipeline.begin()
    assert isinstance(second_capture, StartCapture)
    second = second_pipeline.start(_admit(second_capture))

    assert isinstance(second, StartLaunched)
    assert store.revision == 0
    assert first.configuration.generation == 1
    assert second.configuration.generation == 2
    assert first.run_identity != second.run_identity
    assert first_capture is not second_capture


def test_executor_invariant_recovery_preserves_secondary_cleanup_order():
    source = RecordingSource()
    source.cancel_error = RuntimeError("source close")
    executor = RecordingExecutor("raise")
    executor.close_error = RuntimeError("executor close")
    pipeline, _, _, _, _ = _pipeline(source=source, executor=executor)
    capture = pipeline.begin()
    assert isinstance(capture, StartCapture)

    failed = pipeline.start(_admit(capture))

    assert [failure.operation for failure in failed.recovery_failures] == [
        "source.cancel",
        "executor.close",
    ]


def test_malformed_typed_executor_failure_is_contained_before_lifecycle_dispatch():
    executor = RecordingExecutor("malformed_failure")
    pipeline, _, lifecycle, _, _ = _pipeline(executor=executor)
    capture = pipeline.begin()
    assert isinstance(capture, StartCapture)

    failed = pipeline.start(_admit(capture))

    assert failed.reason is StartFailureKind.EXECUTOR_INVARIANT
    assert failed.configuration is executor.calls[0][0]
    assert failed.run_identity is executor.calls[0][2]
    assert failed.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert lifecycle.phase is RunPhase.FAILED
    assert lifecycle.attempt_run_identity is failed.run_identity


def test_public_start_values_and_methods_have_runtime_resolvable_hints():
    import xdart.gui.tabs.scattering.start_outcomes as module

    for name in module.__all__:
        value = getattr(module, name)
        if getattr(value, "__annotations__", None):
            assert get_type_hints(value), name
    for name, method in inspect.getmembers(StartPipeline, predicate=inspect.isfunction):
        if name.startswith("_"):
            continue
        assert get_type_hints(method)
    assert RecoveryFailure("source.cancel", ExceptionDetail("builtins", "ValueError", "x"))
    assert OwnersClosed(RunIdentity(1, "x"))
