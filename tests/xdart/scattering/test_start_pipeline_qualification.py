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


def test_degenerate_threshold_pair_is_canonical_from_capture_through_provenance():
    """End-to-end LV-UI-11 identity oracle (Codex P1, 2026-08-04): for BOTH
    degenerate intent pairs, the start capture canonicalizes THROUGH the
    store, and capture, admission signature, frozen configuration, execution
    values, fingerprint and writer provenance all describe the SAME canonical
    pair.  The displayed Auto fact (mask_saturation) wins."""
    from xdart.gui.tabs.scattering.contracts import threshold_pair_is_canonical
    from xdart.gui.tabs.scattering.output_preflight import (
        OutputCandidate,
        execution_plan_values,
    )

    for apply_flag, mask_flag in ((True, True), (False, False)):
        intent = _intent()
        intent.output_mode = "Overwrite"
        intent.threshold.apply_threshold = apply_flag
        intent.threshold.mask_saturation = mask_flag
        intent.threshold.threshold_min = 1.0
        intent.threshold.threshold_max = 2.0
        store = RunIntentStore(intent)
        pipeline, _, _ = _pipeline(store)

        capture = _capture(pipeline)

        # 1. capture: canonicalized through the store, not a local copy.
        captured = capture.intent_snapshot.thaw().threshold
        assert threshold_pair_is_canonical(captured)
        assert captured.mask_saturation is mask_flag
        assert captured.apply_threshold is (not mask_flag)
        stored = store.snapshot().thaw().threshold
        assert stored.apply_threshold is (not mask_flag)
        assert stored.mask_saturation is mask_flag

        canonical_dict = {
            "apply_threshold": not mask_flag,
            "threshold_min": 1.0,
            "threshold_max": 2.0,
            "mask_saturation": mask_flag,
        }

        # 2. admission signature.
        candidate = OutputCandidate.from_start_capture(capture)
        assert candidate.processing_mapping()["threshold"] == canonical_dict

        # 3. frozen configuration (the store freeze start() performs).
        result = store.freeze(
            expected_revision=capture.intent_snapshot.revision
        )
        assert isinstance(result, IntentFreezeAccepted)
        configuration = result.configuration
        assert configuration.threshold.apply_threshold is (not mask_flag)
        assert configuration.threshold.mask_saturation is mask_flag

        # 4. fingerprint: the signed candidate and the executed
        # configuration carry one identity.
        assert candidate.fingerprint == configuration.fingerprint

        # 5. execution values agree with the canonical identity.
        _, _, values = execution_plan_values(configuration)
        if mask_flag:
            assert values["threshold_min"] is None
            assert values["threshold_max"] is None
            assert values["mask_saturation"] is True
        else:
            assert values["threshold_min"] == 1.0
            assert values["threshold_max"] == 2.0
            assert values["mask_saturation"] is False

        # 6. writer provenance records that same canonical mapping.
        assert configuration.as_provenance()["threshold"] == canonical_dict


def test_boundaries_refuse_a_degenerate_pair_that_bypassed_capture():
    """Admission and execution REJECT a degenerate pair instead of silently
    reinterpreting it — canonicalization has exactly one owner (capture)."""
    import dataclasses

    import pytest

    from xdart.gui.tabs.scattering.output_preflight import (
        OutputCandidate,
        execution_plan_values,
    )

    degenerate = _intent()
    degenerate.output_mode = "Overwrite"
    degenerate.threshold.apply_threshold = True
    degenerate.threshold.mask_saturation = True

    pipeline, _, _ = _pipeline(RunIntentStore(_intent()))
    capture = _capture(pipeline)
    bypassed = dataclasses.replace(
        capture,
        intent_snapshot=RunIntentStore(degenerate).snapshot(),
    )
    with pytest.raises(ValueError, match="degenerate threshold identity"):
        OutputCandidate.from_start_capture(bypassed)

    with pytest.raises(ValueError, match="degenerate threshold identity"):
        execution_plan_values(degenerate.freeze())
