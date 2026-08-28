from __future__ import annotations

import time

from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering.contracts import (
    SourceObservation,
    SourceObservationRequest,
    SourceObservationStatus,
)
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.controls_projection import GI_MOTOR, SAVE_PATH
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell
from xdart.gui.widgets.controls_panel import FormRow
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import GIIntent, RunIntent
from xrd_tools.sources.selection import DirectorySourceSpec


class MotorSource:
    def __init__(self) -> None:
        self.knowledge: SourceObservation | None = None

    def observe(self, request: SourceObservationRequest) -> SourceObservation:
        return SourceObservation(
            request.observation_id,
            request.intent_revision,
            request.source,
            SourceObservationStatus.AVAILABLE,
            "raw",
            True,
            True,
            direct_child_count=1,
            candidate_fingerprint="one-candidate",
        )

    def preview_motors(
        self, request: SourceObservationRequest
    ) -> SourceObservation:
        return SourceObservation(
            request.observation_id,
            request.intent_revision,
            request.source,
            SourceObservationStatus.AVAILABLE,
            "raw",
            True,
            True,
            direct_child_count=1,
            gi_motor_choices=("halpha",),
            candidate_fingerprint="one-candidate",
        )

    def cancel_observation(self, _observation_id: int) -> None:
        return

    def publish_motor_knowledge(
        self, observation: SourceObservation
    ) -> None:
        self.knowledge = observation

    def project_motor_knowledge(
        self, source, candidate_fingerprint=None
    ):
        knowledge = self.knowledge
        if knowledge is None or knowledge.source != source:
            return None
        if (
            candidate_fingerprint is not None
            and knowledge.candidate_fingerprint != candidate_fingerprint
        ):
            return None
        return knowledge


def _shell(page: ScatteringWorkspace) -> ScatteringWorkspaceShell:
    shell = page.findChild(ScatteringWorkspaceShell)
    assert shell is not None
    return shell


def _choices(page: ScatteringWorkspace) -> tuple[str, ...]:
    state = _shell(page).controls.projection
    assert state is not None
    return next(field.choices for field in state.fields if field.path == GI_MOTOR)


def _visible_choices(page: ScatteringWorkspace) -> tuple[str, ...]:
    row = next(
        row
        for row in _shell(page).controls.findChildren(FormRow)
        if row.path == GI_MOTOR
    )
    return tuple(row.editor.itemText(index) for index in range(row.editor.count()))


def _wait(app: QtWidgets.QApplication, predicate) -> None:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("page operation did not settle")


def test_motor_preview_survives_unrelated_valid_edit(tmp_path) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    source = DirectorySourceSpec(tmp_path / "raw", suffixes=(".nxs",))
    run_store = RunIntentStore(
        RunIntent(
            source_spec=source,
            save_path=str(tmp_path / "processed"),
            gi=GIIntent(enabled=True),
        )
    )
    page = ScatteringWorkspace(
        intents=run_store,
        lifecycle=ScatteringCoordinator(),
        sources=MotorSource(),
    )
    try:
        _wait(
            app,
            lambda: not page._source_selection.observing
            and _choices(page) == ("Manual", "halpha"),
        )
        # The mounted editor must reconcile to the projected source vocabulary;
        # keeping construction-time choices would leave a discovered motor
        # selectable in state but absent from the actual dropdown.
        assert _visible_choices(page) == ("Manual", "halpha")
        _shell(page).controls.fieldValueChanged.emit(
            SAVE_PATH, str(tmp_path / "processed-two")
        )
        assert _choices(page) == ("Manual", "halpha")
        assert _visible_choices(page) == ("Manual", "halpha")
    finally:
        page.close_workspace()


def test_source_change_synchronously_drops_prior_motor_knowledge(
    tmp_path,
) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    first = DirectorySourceSpec(tmp_path / "first", suffixes=(".nxs",))
    second = DirectorySourceSpec(tmp_path / "second", suffixes=(".nxs",))
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(
            source_spec=first,
            gi=GIIntent(enabled=True),
        )),
        lifecycle=ScatteringCoordinator(),
        sources=MotorSource(),
    )
    try:
        _wait(
            app,
            lambda: not page._source_selection.observing
            and _choices(page) == ("Manual", "halpha"),
        )
        page.select_source(second)
        assert _choices(page) == ("Manual",)
        assert _visible_choices(page) == ("Manual",)
        assert page._intents.snapshot().thaw().gi.incidence_motor == "Manual"
    finally:
        page.close_workspace()


def test_source_owner_never_projects_knowledge_for_another_source(
    tmp_path,
) -> None:
    first = DirectorySourceSpec(tmp_path / "first", suffixes=(".nxs",))
    second = DirectorySourceSpec(tmp_path / "second", suffixes=(".nxs",))
    adapter = FilesystemSourceAdapter()
    adapter.publish_motor_knowledge(
        SourceObservation(
            1,
            0,
            first,
            SourceObservationStatus.AVAILABLE,
            "first",
            True,
            True,
            gi_motor_choices=("halpha",),
            candidate_fingerprint="first-fingerprint",
        )
    )

    assert adapter.project_motor_knowledge(second) is None
