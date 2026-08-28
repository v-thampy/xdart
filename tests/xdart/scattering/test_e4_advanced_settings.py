"""Native vNext Advanced integration-settings ownership."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering.advanced_editor import (
    AdvancedSettingsDialog,
)
from xdart.gui.tabs.scattering.contracts import (
    SourceCapture,
    SourceObservation,
    SourceObservationRequest,
    SourceObservationStatus,
)
from xdart.gui.tabs.scattering.controls_editing import (
    AdvancedSettingsValues,
    EditNoChange,
    EditRefusal,
    advanced_settings_values,
    reduce_advanced_settings,
)
from xdart.gui.tabs.scattering.controls_projection import project_controls
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.shell_values import (
    ShellCommand,
    ShellCommandKind,
)
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xdart.gui.widgets.controls_panel import ActionButton
from xrd_tools.session.intent_store import (
    IntentFreezeAccepted,
    RunIntentStore,
)
from xrd_tools.session.readiness import (
    ControlAction,
    SectionId,
    build_native_int_reduction_plan_from_args,
)
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import image_series_spec


@pytest.fixture
def qapp() -> QtWidgets.QApplication:
    return (
        QtWidgets.QApplication.instance()
        or QtWidgets.QApplication([])
    )


class _Sources:
    def __init__(self) -> None:
        self.epoch = 0

    def capture(self, source, request_id):
        self.epoch += 1
        return SourceCapture(request_id, self.epoch, source)

    def cancel(self, _request_id) -> None:
        return None

    def observe(
        self,
        request: SourceObservationRequest,
    ) -> SourceObservation:
        return SourceObservation(
            request.observation_id,
            request.intent_revision,
            request.source,
            SourceObservationStatus.AVAILABLE,
            "frame.tif",
            True,
            False,
        )

    def cancel_observation(self, _observation_id: int) -> None:
        return None

    def publish_motor_knowledge(self, _observation) -> None:
        return None

    def project_motor_knowledge(self, _source, _fingerprint=None):
        return None


def _intent(tmp_path: Path) -> RunIntent:
    return RunIntent(
        source_spec=image_series_spec(tmp_path / "frame_0001.tif"),
        poni_file=str(tmp_path / "calibration.poni"),
        save_path=str(tmp_path / "output.nxs"),
        output_mode="Overwrite",
        bai_1d_args={
            "npt": 128,
            "method": "csr",
            "correctSolidAngle": True,
            "polarization_factor": None,
        },
        bai_2d_args={
            "npt_rad": 128,
            "npt_azim": 64,
            "method": "csr",
            "azimuth_offset": 7.0,
        },
    )


def _action(state):
    return next(
        action
        for action in state.actions_for(SectionId.PROCESSING)
        if action.action is ControlAction.ADVANCED_PROCESSING
    )


def _advanced_command() -> ShellCommand:
    return ShellCommand(
        ShellCommandKind.CONTROL_ACTION,
        "advanced_processing",
    )


def _close(page: ScatteringWorkspace) -> None:
    # Per-test disposal intentionally does not drain DeferredDelete.
    page.close_workspace()
    page.close()


def test_advanced_action_requires_a_mounted_editor_and_unlocked_controls(
    tmp_path: Path,
) -> None:
    snapshot = RunIntentStore(_intent(tmp_path)).snapshot()

    absent = _action(
        project_controls(
            snapshot,
            None,
            RunPhase.IDLE,
            advanced_editor_available=False,
        )
    )
    mounted = _action(
        project_controls(
            snapshot,
            None,
            RunPhase.IDLE,
            advanced_editor_available=True,
        )
    )
    running = _action(
        project_controls(
            snapshot,
            None,
            RunPhase.RUNNING,
            advanced_editor_available=True,
        )
    )

    assert absent.enabled is False
    assert "native vNext editor" in absent.reason
    assert mounted.enabled is True
    assert mounted.production_ready is True
    assert running.enabled is False
    assert "locked" in running.reason


@pytest.mark.parametrize(
    ("dimension", "changes", "reason"),
    (
        ("one_d", {"method": "not-a-method"}, "method"),
        ("one_d", {"polarization_factor": 2.0}, "from -1 through 1"),
        ("two_d", {"chi_offset": float("nan")}, "finite"),
        ("two_d", {"safe": "yes"}, "true or false"),
        ("one_d", {"dummy": "not-a-number"}, "finite or blank"),
    ),
)
def test_advanced_reducer_rejects_invalid_values_without_mutating_snapshot(
    tmp_path: Path,
    dimension: str,
    changes: dict[str, object],
    reason: str,
) -> None:
    store = RunIntentStore(_intent(tmp_path))
    snapshot = store.snapshot()
    values = advanced_settings_values(snapshot)
    edited = replace(getattr(values, dimension), **changes)
    submission = replace(values, **{dimension: edited})

    result = reduce_advanced_settings(snapshot, submission)

    assert isinstance(result, EditRefusal)
    assert reason in result.reason
    assert store.revision == 0
    assert store.snapshot().thaw().bai_1d_args == (
        snapshot.thaw().bai_1d_args
    )


def test_value_dialog_round_trip_is_no_change_and_has_no_model_authority(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    snapshot = RunIntentStore(_intent(tmp_path)).snapshot()
    dialog = AdvancedSettingsDialog()
    try:
        dialog.load_values(advanced_settings_values(snapshot))

        assert dialog.one_d.polarization_factor.isEnabled() is False
        assert dialog.gi_group.isHidden() is True
        assert dialog.one_d.method.isEnabled() is True
        assert dialog.two_d.method.isEnabled() is True
        assert isinstance(
            reduce_advanced_settings(snapshot, dialog.values()),
            EditNoChange,
        )
        assert dialog.findChildren(QtWidgets.QTreeView) == []
    finally:
        dialog.close()


def test_gi_advanced_defaults_to_cython_and_commits_python_to_both_dimensions(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    intent = _intent(tmp_path)
    intent.gi.enabled = True
    store = RunIntentStore(intent)
    snapshot = store.snapshot()
    projected = advanced_settings_values(snapshot)

    assert projected.gi_enabled is True
    assert projected.gi_method == "cython"

    dialog = AdvancedSettingsDialog()
    try:
        dialog.load_values(projected)

        assert dialog.gi_group.isHidden() is False
        assert dialog.gi_method.currentData() == "cython"
        assert dialog.one_d.method.isEnabled() is False
        assert dialog.two_d.method.isEnabled() is False

        python_index = dialog.gi_method.findData("python")
        assert python_index >= 0
        dialog.gi_method.setCurrentIndex(python_index)
        candidate = reduce_advanced_settings(snapshot, dialog.values())

        assert isinstance(candidate, RunIntent)
        assert candidate.bai_1d_args["gi_method_1d"] == "python"
        assert candidate.bai_2d_args["gi_method_2d"] == "python"
        store.commit(candidate, expected_revision=snapshot.revision)
        frozen = store.freeze(expected_revision=1)
        assert type(frozen) is IntentFreezeAccepted
        assert frozen.configuration.bai_1d_args["gi_method_1d"] == (
            "python"
        )
        assert frozen.configuration.bai_2d_args["gi_method_2d"] == (
            "python"
        )
        plan = build_native_int_reduction_plan_from_args(
            frozen.configuration.bai_1d_args,
            frozen.configuration.bai_2d_args,
            gi_enabled=True,
            gi_incident_angle=0.1,
        )
        assert plan.gi is not None
        assert plan.gi.method == "python"
    finally:
        dialog.close()


def test_legacy_no_backend_displays_as_python_without_implicit_rewrite(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    intent = _intent(tmp_path)
    intent.gi.enabled = True
    intent.bai_1d_args["gi_method_1d"] = "no"
    snapshot = RunIntentStore(intent).snapshot()
    projected = advanced_settings_values(snapshot)

    assert projected.gi_method == "python"
    dialog = AdvancedSettingsDialog()
    try:
        dialog.load_values(projected)

        assert dialog.gi_method.currentData() == "python"
        assert isinstance(
            reduce_advanced_settings(snapshot, dialog.values()),
            EditNoChange,
        )

        unrelated = replace(
            projected,
            one_d=replace(projected.one_d, safe=False),
        )
        candidate = reduce_advanced_settings(snapshot, unrelated)
        assert isinstance(candidate, RunIntent)
        assert candidate.bai_1d_args["gi_method_1d"] == "no"
        assert "gi_method_2d" not in candidate.bai_2d_args

        switched = reduce_advanced_settings(
            snapshot,
            replace(projected, gi_method="cython"),
        )
        assert isinstance(switched, RunIntent)
        assert switched.bai_1d_args["gi_method_1d"] == "cython"
        assert switched.bai_2d_args["gi_method_2d"] == "cython"
    finally:
        dialog.close()


@pytest.mark.parametrize(
    ("changes", "reason"),
    (
        ({"gi_enabled": 1}, "true or false"),
        ({"gi_enabled": False}, "outside"),
        ({"gi_method": None}, "invalid"),
        ({"gi_method": "no"}, "unsupported"),
    ),
)
def test_gi_advanced_rejects_invalid_exact_schema(
    tmp_path: Path,
    changes: dict[str, object],
    reason: str,
) -> None:
    intent = _intent(tmp_path)
    intent.gi.enabled = True
    snapshot = RunIntentStore(intent).snapshot()
    submission = replace(advanced_settings_values(snapshot), **changes)

    result = reduce_advanced_settings(snapshot, submission)

    assert isinstance(result, EditRefusal)
    assert reason in result.reason


def test_standard_advanced_cannot_stage_a_hidden_gi_backend(
    tmp_path: Path,
) -> None:
    snapshot = RunIntentStore(_intent(tmp_path)).snapshot()
    current = advanced_settings_values(snapshot)

    result = reduce_advanced_settings(
        snapshot,
        replace(current, gi_method="python"),
    )

    assert isinstance(result, EditRefusal)
    assert "only in Grazing mode" in result.reason


def test_cancel_and_semantic_no_change_do_not_advance_revision(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    store = RunIntentStore(_intent(tmp_path))
    replies: list[AdvancedSettingsValues | None] = [None]

    def editor(snapshot):
        if replies:
            return replies.pop(0)
        return advanced_settings_values(snapshot)

    page = ScatteringWorkspace(
        intents=store,
        lifecycle=ScatteringCoordinator(),
        sources=_Sources(),
        advanced_settings_editor=editor,
    )
    try:
        button = next(
            candidate
            for candidate in page._shell.controls.findChildren(ActionButton)
            if candidate.spec.action
            is ControlAction.ADVANCED_PROCESSING
        )
        assert button.isEnabled()
        button.click()
        assert store.revision == 0

        page._shell.commandRequested.emit(_advanced_command())
        assert store.revision == 0
    finally:
        _close(page)


def test_accepted_advanced_values_commit_by_revision_and_freeze_for_start(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunIntentStore(_intent(tmp_path))

    def editor(snapshot):
        current = advanced_settings_values(snapshot)
        return AdvancedSettingsValues(
            replace(
                current.one_d,
                correct_solid_angle=False,
                apply_polarization=True,
                polarization_factor=0.73,
                method="BBox",
                dummy=-2.0,
                delta_dummy=0.25,
                chi_offset=5.0,
                safe=False,
            ),
            replace(
                current.two_d,
                apply_polarization=True,
                polarization_factor=0.81,
                method="cython",
                dummy=-3.0,
                delta_dummy=0.5,
                chi_offset=12.0,
                safe=False,
            ),
        )

    page = ScatteringWorkspace(
        intents=store,
        lifecycle=ScatteringCoordinator(),
        sources=_Sources(),
        executor=object(),
        advanced_settings_editor=editor,
    )
    captures = []
    try:
        page._shell.commandRequested.emit(_advanced_command())

        assert store.revision == 1
        current = store.snapshot()
        one = current.thaw().bai_1d_args
        two = current.thaw().bai_2d_args
        assert one["npt"] == 128
        assert one["correctSolidAngle"] is False
        assert one["polarization_factor"] == pytest.approx(0.73)
        assert one["method"] == "BBox"
        assert one["dummy"] == pytest.approx(-2.0)
        assert one["delta_dummy"] == pytest.approx(0.25)
        assert one["chi_offset"] == pytest.approx(5.0)
        assert one["safe"] is False
        assert two["npt_rad"] == 128
        assert two["npt_azim"] == 64
        assert "azimuth_offset" not in two
        assert two["chi_offset"] == pytest.approx(12.0)

        monkeypatch.setattr(page, "_begin_admission", captures.append)
        page._shell.commandRequested.emit(
            ShellCommand(ShellCommandKind.RUN_ACTION)
        )
        assert len(captures) == 1
        assert captures[0].intent_snapshot.revision == 1

        frozen = store.freeze(expected_revision=1)
        assert type(frozen) is IntentFreezeAccepted
        configuration = frozen.configuration
        assert configuration.bai_1d_args["polarization_factor"] == (
            pytest.approx(0.73)
        )
        assert configuration.bai_2d_args["chi_offset"] == (
            pytest.approx(12.0)
        )
        provenance = configuration.as_provenance()
        assert provenance["bai_1d_args"]["method"] == "BBox"
        assert provenance["bai_2d_args"]["safe"] is False
    finally:
        _close(page)


def test_advanced_cas_does_not_overwrite_a_concurrent_intent_revision(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    store = RunIntentStore(_intent(tmp_path))

    def editor(snapshot):
        concurrent = store.snapshot().thaw()
        concurrent.project_root = "/concurrent"
        store.commit(concurrent, expected_revision=store.revision)
        current = advanced_settings_values(snapshot)
        return replace(
            current,
            one_d=replace(current.one_d, method="BBox"),
        )

    page = ScatteringWorkspace(
        intents=store,
        lifecycle=ScatteringCoordinator(),
        sources=_Sources(),
        advanced_settings_editor=editor,
    )
    try:
        page._shell.commandRequested.emit(_advanced_command())

        assert store.revision == 1
        assert store.snapshot().thaw().project_root == "/concurrent"
        assert store.snapshot().thaw().bai_1d_args["method"] == "csr"
        assert "superseded" in page._shell.scientific.status.text()
    finally:
        _close(page)
