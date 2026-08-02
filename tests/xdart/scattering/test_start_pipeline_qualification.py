"""Canonical-store qualification adversaries for ``StartPipeline``."""

from __future__ import annotations

from pathlib import Path

from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.session.intent_store import IntentCommitAccepted, IntentFreezeAccepted, RunIntentStore
from xrd_tools.session.run_configuration import GIIntent, RunIntent
from xdart.gui.tabs.scattering.contracts import SourceCapture
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.events import ExecutorAccepted, RunIdentity
from xdart.gui.tabs.scattering.start_outcomes import StartCapture, StartFailed, StartLaunched
from xdart.gui.tabs.scattering.start_pipeline import StartPipeline
from tests.xdart.scattering._admission import admission_for


def _intent() -> RunIntent:
    return RunIntent(
        source_spec=SourceSpec(Path("/data/frame.tif"), SourceKind.IMAGE_FILE),
        gi=GIIntent(enabled=True, incidence_motor="th", th_val=0.2),
    )


class Source:
    def __init__(self) -> None:
        self.captures: list[SourceCapture] = []

    def capture(self, source: SourceSpec, request_id) -> SourceCapture:
        capture = SourceCapture(request_id, 0, source, ("th",))
        self.captures.append(capture)
        return capture

    def cancel(self, request_id) -> None: ...


class Executor:
    def __init__(self) -> None:
        self.calls: list[tuple[object, RunIdentity]] = []

    def start(self, configuration, source, run_identity, admission):
        self.calls.append((configuration, run_identity))
        return ExecutorAccepted(run_identity)

    def close(self, run_identity) -> None: ...
    def pause(self, run_identity) -> None: ...
    def resume(self, run_identity) -> None: ...
    def stop(self, run_identity) -> None: ...


class MismatchedCommitStore(RunIntentStore):
    def commit(self, candidate, *, expected_revision):
        result = super().commit(candidate, expected_revision=expected_revision)
        assert isinstance(result, IntentCommitAccepted)
        return IntentCommitAccepted(result.revision + 1, result.snapshot)


class MismatchedFreezeStore(RunIntentStore):
    def freeze(self, *, expected_revision, gi_motor_choices=None):
        result = super().freeze(expected_revision=expected_revision, gi_motor_choices=gi_motor_choices)
        assert isinstance(result, IntentFreezeAccepted)
        return IntentFreezeAccepted(result.revision + 1, result.configuration)


class EditAfterAcceptedCommitStore(RunIntentStore):
    def __init__(self, initial: RunIntent) -> None:
        super().__init__(initial)
        self.accepted_snapshot = None

    def commit(self, candidate, *, expected_revision):
        accepted = super().commit(candidate, expected_revision=expected_revision)
        assert isinstance(accepted, IntentCommitAccepted)
        self.accepted_snapshot = accepted.snapshot
        later = self.snapshot().thaw()
        later.processing_mode = "Int 1D"
        super().commit(later, expected_revision=accepted.revision)
        return accepted


class UnexpectedFreezeStore(RunIntentStore):
    def freeze(self, *, expected_revision, gi_motor_choices=None):
        raise RuntimeError("unexpected freeze fault")


def _pipeline(store: RunIntentStore):
    source = Source()
    executor = Executor()
    return StartPipeline(
        intents=store,
        lifecycle=ScatteringCoordinator(),
        sources=source,
        executor=executor,
    ), source, executor


def _capture(pipeline: StartPipeline) -> StartCapture:
    capture = pipeline.begin()
    assert isinstance(capture, StartCapture)
    return capture


def test_mismatched_accepted_commit_result_is_canonical_invariant():
    pipeline, _, executor = _pipeline(MismatchedCommitStore(_intent()))
    capture = _capture(pipeline)

    failed = pipeline.apply_operator_decision(capture, capture.intent_snapshot.thaw())

    assert isinstance(failed, StartFailed)
    assert executor.calls == []


def test_mismatched_accepted_freeze_result_is_canonical_invariant():
    pipeline, _, executor = _pipeline(MismatchedFreezeStore(_intent()))

    failed = pipeline.start(admission_for(_capture(pipeline)))

    assert isinstance(failed, StartFailed)
    assert failed.configuration is not None
    assert executor.calls == []


def test_accepted_commit_capture_uses_the_exact_returned_snapshot_not_a_later_store_read():
    store = EditAfterAcceptedCommitStore(_intent())
    pipeline, _, executor = _pipeline(store)
    capture = _capture(pipeline)
    candidate = capture.intent_snapshot.thaw()
    candidate.output_mode = "Overwrite"

    replacement = pipeline.apply_operator_decision(capture, candidate)

    assert isinstance(replacement, StartCapture)
    assert replacement.intent_snapshot is store.accepted_snapshot
    assert replacement.intent_snapshot.revision == 1
    assert store.snapshot().revision == 2
    launched = pipeline.start(admission_for(replacement))
    assert not isinstance(launched, StartLaunched)
    assert executor.calls == []


def test_unexpected_freeze_exception_is_closed_invariant_not_a_validation_refusal():
    pipeline, _, executor = _pipeline(UnexpectedFreezeStore(_intent()))

    failed = pipeline.start(admission_for(_capture(pipeline)))

    assert isinstance(failed, StartFailed)
    assert failed.reason.value == "canonical_invariant"
    assert executor.calls == []
