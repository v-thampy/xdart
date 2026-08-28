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


def test_all_threshold_pairs_remain_exact_through_execution_and_provenance():
    """Manual thresholding and saturated masking are independent run facts."""
    from xdart.gui.tabs.scattering.output_preflight import (
        OutputCandidate,
        execution_plan_values,
    )

    for apply_flag, mask_flag in (
        (False, False), (False, True), (True, False), (True, True),
    ):
        intent = _intent()
        intent.output_mode = "Overwrite"
        intent.threshold.apply_threshold = apply_flag
        intent.threshold.mask_saturation = mask_flag
        intent.threshold.threshold_min = 1.0
        intent.threshold.threshold_max = 2.0
        store = RunIntentStore(intent)
        pipeline, _, _ = _pipeline(store)

        capture = _capture(pipeline)

        # Capture and store preserve the exact independently chosen pair.
        captured = capture.intent_snapshot.thaw().threshold
        assert captured.mask_saturation is mask_flag
        assert captured.apply_threshold is apply_flag
        stored = store.snapshot().thaw().threshold
        assert stored.apply_threshold is apply_flag
        assert stored.mask_saturation is mask_flag

        exact_dict = {
            "apply_threshold": apply_flag,
            "threshold_min": 1.0,
            "threshold_max": 2.0,
            "mask_saturation": mask_flag,
        }

        # Admission signature.
        candidate = OutputCandidate.from_start_capture(capture)
        assert candidate.processing_mapping()["threshold"] == exact_dict

        # Frozen configuration (the store freeze start() performs).
        result = store.freeze(
            expected_revision=capture.intent_snapshot.revision
        )
        assert isinstance(result, IntentFreezeAccepted)
        configuration = result.configuration
        assert configuration.threshold.apply_threshold is apply_flag
        assert configuration.threshold.mask_saturation is mask_flag

        # The signed candidate and executed configuration carry one identity.
        assert candidate.fingerprint == configuration.fingerprint

        # Execution uses the band only when requested, and masks saturation
        # independently.
        _, _, values = execution_plan_values(configuration)
        if apply_flag:
            assert values["threshold_min"] == 1.0
            assert values["threshold_max"] == 2.0
        else:
            assert values["threshold_min"] is None
            assert values["threshold_max"] is None
        assert values["mask_saturation"] is mask_flag

        # Writer provenance records that same exact mapping.
        assert configuration.as_provenance()["threshold"] == exact_dict


def test_direct_admission_and_execution_accept_each_threshold_pair():
    """No later boundary silently rejects or rewrites either independent fact."""
    import dataclasses

    from xdart.gui.tabs.scattering.output_preflight import (
        OutputCandidate,
        execution_plan_values,
    )

    pipeline, _, _ = _pipeline(RunIntentStore(_intent()))
    capture = _capture(pipeline)
    for apply_flag, mask_flag in (
        (False, False), (False, True), (True, False), (True, True),
    ):
        direct = _intent()
        direct.output_mode = "Overwrite"
        direct.threshold.apply_threshold = apply_flag
        direct.threshold.mask_saturation = mask_flag
        direct.threshold.threshold_min = 1.0
        direct.threshold.threshold_max = 2.0
        bypassed = dataclasses.replace(
            capture,
            intent_snapshot=RunIntentStore(direct).snapshot(),
        )
        candidate = OutputCandidate.from_start_capture(bypassed)
        assert candidate.processing_mapping()["threshold"][
            "apply_threshold"
        ] is apply_flag
        _, _, values = execution_plan_values(direct.freeze())
        assert values["mask_saturation"] is mask_flag


def _eiger_poni(tmp_path: Path) -> str:
    poni = tmp_path / "eiger.poni"
    poni.write_text(
        "Poni_version: 2.1\n"
        "Detector: Eiger1M\n"
        'Detector_config: {"orientation": 1}\n'
        "Distance: 0.2\n"
        "Poni1: 0.1\n"
        "Poni2: 0.2\n"
        "Rot1: 0.0\n"
        "Rot2: 0.0\n"
        "Rot3: 0.0\n"
        "Wavelength: 1e-10\n"
    )
    return str(poni)


def test_defaulted_bounds_materialize_to_exactly_the_displayed_band(tmp_path):
    """DESIGN_STOP oracle (2026-08-04): manual mode with absent/cleared bounds
    DISPLAYS a substituted band; the start capture must MATERIALIZE that exact
    band into the one run identity.  Known detector -> the finite displayed
    band everywhere; unknown detector -> blank max = open-ended, EVERYWHERE
    (display and identity agree in both directions).  Spans rendered values,
    capture/store, admission signature, fingerprint, execution values and
    writer provenance."""
    from xdart.gui.tabs.scattering.controls_inventory import (
        THRESHOLD_MAX,
        THRESHOLD_MIN,
    )
    from xdart.gui.tabs.scattering.controls_projection import project_controls
    from xdart.gui.tabs.scattering.output_preflight import (
        OutputCandidate,
        execution_plan_values,
    )
    from xdart.gui.tabs.scattering.state_machine import RunPhase

    for poni_file, want_max in ((_eiger_poni(tmp_path), 4294967295.0), ("", None)):
        intent = _intent()
        intent.output_mode = "Overwrite"
        intent.poni_file = poni_file
        intent.threshold.apply_threshold = True
        intent.threshold.mask_saturation = False
        assert intent.threshold.threshold_min is None   # cleared/absent bounds
        assert intent.threshold.threshold_max is None
        store = RunIntentStore(intent)

        # 1. rendered values: what the panel actually shows pre-run.
        shown = {
            field.path: field.value
            for field in project_controls(
                store.snapshot(), None, RunPhase.IDLE
            ).fields
        }
        assert shown[THRESHOLD_MIN] == 0.0
        assert shown[THRESHOLD_MAX] == want_max

        # 2. capture + store: the displayed manual band is MATERIALIZED
        # through the store without changing either independent switch.
        pipeline, _, _ = _pipeline(store)
        capture = _capture(pipeline)
        captured = capture.intent_snapshot.thaw().threshold
        assert captured.threshold_min == 0.0
        assert captured.threshold_max == want_max
        stored = store.snapshot().thaw().threshold
        assert stored.threshold_min == 0.0
        assert stored.threshold_max == want_max

        canonical = {
            "apply_threshold": True,
            "threshold_min": 0.0,
            "threshold_max": want_max,
            "mask_saturation": False,
        }

        # 3. admission signature.
        candidate = OutputCandidate.from_start_capture(capture)
        assert candidate.processing_mapping()["threshold"] == canonical

        # 4. frozen configuration + fingerprint agreement.
        result = store.freeze(
            expected_revision=capture.intent_snapshot.revision
        )
        assert isinstance(result, IntentFreezeAccepted)
        configuration = result.configuration
        assert candidate.fingerprint == configuration.fingerprint

        # 5. execution values are the displayed band, not a silent no-op.
        _, _, values = execution_plan_values(configuration)
        assert values["threshold_min"] == 0.0
        assert values["threshold_max"] == want_max
        assert values["mask_saturation"] is False

        # 6. writer provenance records the same canonical mapping.
        assert configuration.as_provenance()["threshold"] == canonical
