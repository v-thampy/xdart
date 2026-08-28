from __future__ import annotations

import time

import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering.contracts import (
    SourceObservation,
    SourceObservationRequest,
    SourceObservationStatus,
)
from xdart.gui.tabs.scattering.controls_projection import GI_MOTOR
from xdart.gui.tabs.scattering.controls_projection import OUTPUT_MODE, project_controls
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell
from xdart.gui.widgets.controls_panel import FormRow
from xrd_tools.session.readiness import ControlFieldKind, ControlFormField, SectionId
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import GIIntent, RunIntent
from xrd_tools.sources.selection import DirectorySourceSpec


class _ChangingMotorSource:
    def __init__(self, first, second) -> None:
        self.first = first
        self.second = second
        self.knowledge = None

    def _choices(self, source):
        return ("halpha",) if source == self.first else ("eta",)

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
            candidate_fingerprint=(
                "first-fingerprint"
                if request.source == self.first
                else "second-fingerprint"
            ),
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
            gi_motor_choices=self._choices(request.source),
            candidate_fingerprint=(
                "first-fingerprint"
                if request.source == self.first
                else "second-fingerprint"
            ),
        )

    def cancel_observation(self, _observation_id: int) -> None:
        return

    def publish_motor_knowledge(self, observation: SourceObservation) -> None:
        self.knowledge = observation

    def project_motor_knowledge(self, source, candidate_fingerprint=None):
        knowledge = self.knowledge
        if knowledge is None or knowledge.source != source:
            return None
        if (
            candidate_fingerprint is not None
            and knowledge.candidate_fingerprint != candidate_fingerprint
        ):
            return None
        return knowledge


def _wait(app, predicate) -> None:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("page operation did not settle")


def _shell(page: ScatteringWorkspace) -> ScatteringWorkspaceShell:
    shell = page.findChild(ScatteringWorkspaceShell)
    assert shell is not None
    return shell


def _field(page):
    return next(
        field
        for field in _shell(page).controls._bound_state.fields
        if field.path == GI_MOTOR
    )


def _row(page):
    return next(
        row
        for row in _shell(page).controls.findChildren(FormRow)
        if row.path == GI_MOTOR
    )


def test_known_new_source_keeps_stale_motor_visible_until_real_edit(tmp_path) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    first = DirectorySourceSpec(tmp_path / "first", suffixes=(".nxs",))
    second = DirectorySourceSpec(tmp_path / "second", suffixes=(".nxs",))
    store = RunIntentStore(
        RunIntent(
            source_spec=first,
            gi=GIIntent(enabled=True, incidence_motor="halpha"),
        )
    )
    page = ScatteringWorkspace(
        intents=store,
        lifecycle=ScatteringCoordinator(),
        sources=_ChangingMotorSource(first, second),
    )
    try:
        _wait(
            app,
            lambda: page._observation is None
            and _field(page).choices == ("Manual", "halpha"),
        )
        assert _row(page).editor.currentText() == "halpha"

        page.select_source(second)
        _wait(
            app,
            lambda: page._observation is None
            and _field(page).choices == ("Manual", "eta"),
        )

        visible = tuple(
            _row(page).editor.itemText(index)
            for index in range(_row(page).editor.count())
        )
        with pytest.raises(ValueError, match="GI metadata motor 'halpha'"):
            store.snapshot().thaw().freeze(gi_motor_choices=("eta",))
        assert store.snapshot().thaw().gi.incidence_motor == "halpha"
        assert _row(page).editor.currentText() == "halpha"
        assert "unavailable in this preview" in _field(page).reason
        # The construction token proves projection did not clear-and-rebuild;
        # only the missing accepted value is appended for representability.
        assert visible == ("Manual", "eta", "halpha")
        assert _row(page).editor.currentText() in visible
        _row(page).editor.setCurrentText("eta")
        _row(page).editor.textActivated.emit("eta")
        _wait(
            app,
            lambda: store.snapshot().thaw().gi.incidence_motor == "eta",
        )
        frozen = store.snapshot().thaw().freeze(gi_motor_choices=("eta",))
        assert frozen.gi.effective_motor == "eta"
    finally:
        page.close_workspace()


def _projected_field(store: RunIntentStore, path):
    return next(
        field
        for field in project_controls(
            store.snapshot(), None, RunPhase.IDLE
        ).bound_controls.fields
        if field.path == path
    )


def test_unknown_motor_knowledge_preserves_explicit_raw_projection() -> None:
    store = RunIntentStore(
        RunIntent(gi=GIIntent(enabled=True, incidence_motor="halpha"))
    )

    field = _projected_field(store, GI_MOTOR)

    assert field.value == "halpha"
    assert field.choices == ("Manual", "halpha")


def test_current_append_projects_the_deferred_reason() -> None:
    field = _projected_field(
        RunIntentStore(RunIntent(output_mode="Append")), OUTPUT_MODE
    )

    assert field.value == "Append"
    assert field.choices == ("Overwrite", "Append")
    assert field.reason == ""


def test_combo_reconciliation_is_signal_blocked() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    row = FormRow(
        label="Motor",
        path=GI_MOTOR,
        value="halpha",
        kind="combo",
        choices=("Manual", "halpha"),
    )
    emitted = []
    row.valueChanged.connect(lambda *values: emitted.append(values))
    try:
        assert row.apply_field(
            ControlFormField(
                SectionId.EXPERIMENT,
                "Motor",
                GI_MOTOR,
                "eta",
                ControlFieldKind.COMBO,
                ("Manual", "eta"),
            )
        )
        app.processEvents()
        assert emitted == []
        assert row.editor.currentText() == "eta"
    finally:
        row.close()
        row.deleteLater()


def test_form_row_never_invents_a_value_absent_from_typed_choices() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    row = FormRow(
        label="Motor",
        path=GI_MOTOR,
        value="Manual",
        kind="combo",
        choices=("Manual",),
    )
    try:
        assert row.apply_field(
            ControlFormField(
                SectionId.EXPERIMENT,
                "Motor",
                GI_MOTOR,
                "halpha",
                ControlFieldKind.COMBO,
                ("Manual", "eta"),
            )
        )
        app.processEvents()
        visible = tuple(
            row.editor.itemText(index)
            for index in range(row.editor.count())
        )
        # Typed choices plus the explicit current appended last — never more.
        assert visible == ("Manual", "eta", "halpha")
        assert row.editor.currentText() == "halpha"
    finally:
        row.close()
        row.deleteLater()
