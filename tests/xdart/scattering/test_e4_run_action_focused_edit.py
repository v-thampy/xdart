"""Action-time ownership for the one focused vNext Controls edit."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering.contracts import (
    SourceCapture,
    SourceObservation,
    SourceObservationRequest,
    SourceObservationStatus,
)
from xdart.gui.tabs.scattering.controls_inventory import (
    INT_1D_POINTS,
    INT_2D_RADIAL_POINTS,
    SOURCE_DIRECTORY,
)
from xdart.gui.tabs.scattering.controls_inventory import integration_values
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.shell_values import (
    ShellCommand,
    ShellCommandKind,
)
from xdart.gui.widgets.controls_panel import FormRow
from xrd_tools.session.intent_store import (
    IntentFreezeAccepted,
    RunIntentStore,
)
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import (
    DirectorySourceSpec,
    image_series_spec,
)


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
        self, request: SourceObservationRequest
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


class _CountingStore(RunIntentStore):
    def __init__(self, initial: RunIntent) -> None:
        super().__init__(initial)
        self.commit_calls = 0
        self.recapture_next_commit = False

    def commit(self, candidate, *, expected_revision: int):
        self.commit_calls += 1
        if self.recapture_next_commit:
            self.recapture_next_commit = False
            concurrent = self.snapshot().thaw()
            concurrent.project_root = "/concurrent"
            RunIntentStore.commit(
                self,
                concurrent,
                expected_revision=self.revision,
            )
        return RunIntentStore.commit(
            self,
            candidate,
            expected_revision=expected_revision,
        )


def _page(
    tmp_path: Path,
) -> tuple[ScatteringWorkspace, _CountingStore]:
    store = _CountingStore(
        RunIntent(
            source_spec=image_series_spec(tmp_path / "frame_0001.tif"),
            poni_file=str(tmp_path / "calibration.poni"),
            save_path=str(tmp_path / "output.nxs"),
            output_mode="Overwrite",
        )
    )
    page = ScatteringWorkspace(
        intents=store,
        lifecycle=ScatteringCoordinator(),
        sources=_Sources(),
        executor=object(),  # Admission is intercepted at the capture boundary.
    )
    return page, store


def _row(
    page: ScatteringWorkspace, path: tuple[str, ...]
) -> FormRow:
    return next(
        candidate
        for candidate in page._shell.controls.findChildren(FormRow)
        if candidate.path == path
    )


def _focus(
    page: ScatteringWorkspace,
    editor: QtWidgets.QWidget,
    qapp: QtWidgets.QApplication,
) -> None:
    page.show()
    qapp.processEvents()
    editor.setFocus()
    qapp.processEvents()
    assert editor.hasFocus()


def _run(page: ScatteringWorkspace) -> None:
    page._shell.commandRequested.emit(
        ShellCommand(ShellCommandKind.RUN_ACTION)
    )


def _close(page: ScatteringWorkspace) -> None:
    # Do not post or drain DeferredDelete during per-test disposal.
    page.close_workspace()
    page.close()


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (INT_1D_POINTS, "444"),
        (INT_2D_RADIAL_POINTS, "555"),
    ),
)
def test_run_cas_commits_one_focused_point_edit_before_capture(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: tuple[str, ...],
    value: str,
) -> None:
    page, store = _page(tmp_path)
    captures = []
    try:
        row = _row(page, path)
        other_path = (
            INT_2D_RADIAL_POINTS
            if path == INT_1D_POINTS
            else INT_1D_POINTS
        )
        other = _row(page, other_path)
        _focus(page, row.editor, qapp)
        row.editor.setText(value)
        other.editor.setText("777")
        monkeypatch.setattr(
            page._shell.controls,
            "current_form_edits",
            lambda: pytest.fail(
                "diagnostic full-form snapshot must not be polled"
            ),
        )
        monkeypatch.setattr(page, "_begin_admission", captures.append)

        _run(page)

        assert store.commit_calls == 1
        assert len(captures) == 1
        captured = captures[0].intent_snapshot
        assert captured.revision == 1
        values = integration_values(captured.thaw())
        assert values[path] == int(value)
        assert values[other_path] != 777
        frozen = store.freeze(expected_revision=captured.revision)
        assert type(frozen) is IntentFreezeAccepted
        frozen_value = (
            frozen.configuration.bai_1d_args["npt"]
            if path == INT_1D_POINTS
            else frozen.configuration.bai_2d_args["npt_rad"]
        )
        assert frozen_value == int(value)
    finally:
        _close(page)


def test_invalid_focused_edit_visibly_refuses_without_capture(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page, store = _page(tmp_path)
    captures = []
    try:
        row = _row(page, INT_1D_POINTS)
        _focus(page, row.editor, qapp)
        row.editor.setText("not-an-integer")
        monkeypatch.setattr(page, "_begin_admission", captures.append)

        _run(page)

        assert store.commit_calls == 0
        assert captures == []
        assert page._lifecycle.phase.value == "idle"
        assert "positive integer" in page._shell.scientific.status.text()
    finally:
        _close(page)


def test_focused_edit_cas_recapture_visibly_refuses_without_capture(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page, store = _page(tmp_path)
    captures = []
    try:
        row = _row(page, INT_1D_POINTS)
        _focus(page, row.editor, qapp)
        row.editor.setText("444")
        store.recapture_next_commit = True
        monkeypatch.setattr(page, "_begin_admission", captures.append)

        _run(page)

        assert store.commit_calls == 1
        assert captures == []
        assert page._lifecycle.phase.value == "idle"
        assert store.snapshot().thaw().project_root == "/concurrent"
        assert (
            integration_values(store.snapshot().thaw())[INT_1D_POINTS]
            == 1000
        )
        assert "superseded" in page._shell.scientific.status.text()
        assert "Run not started" in page._shell.scientific.status.text()
    finally:
        _close(page)


def test_run_cas_resets_an_automatic_motor_with_the_focused_source_edit(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page, store = _page(tmp_path)
    captures = []
    try:
        source = DirectorySourceSpec(
            tmp_path / "raw",
            suffixes=(".nxs",),
        )
        page.select_source(source)
        page._maybe_default_gi_motor(
            SimpleNamespace(gi_motor_choices=("exposure", "halpha"))
        )
        assert store.snapshot().thaw().gi.incidence_motor == "halpha"

        row = _row(page, SOURCE_DIRECTORY)
        _focus(page, row.editor, qapp)
        replacement = str(tmp_path / "replacement")
        row.editor.setText(replacement)
        # QLineEdit.setText is deliberately not user intent.  Emit the native
        # user-edit signal so the basename-only path row marks this focused
        # value dirty and action-time capture sees the full typed path.
        row.editor.textEdited.emit(replacement)
        qapp.processEvents()
        store.commit_calls = 0
        monkeypatch.setattr(page, "_begin_admission", captures.append)

        _run(page)

        assert store.commit_calls == 1
        assert len(captures) == 1
        captured = captures[0].intent_snapshot.thaw()
        assert captured.source_spec.root == tmp_path / "replacement"
        assert captured.gi.incidence_motor == "Manual"
    finally:
        _close(page)


def test_unfocused_programmatic_text_is_not_harvested_for_run(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page, store = _page(tmp_path)
    captures = []
    try:
        row = _row(page, INT_1D_POINTS)
        page.show()
        qapp.processEvents()
        page._shell.run_controls.modeCombo.setFocus()
        qapp.processEvents()
        assert not row.editor.hasFocus()
        row.editor.setText("777")
        monkeypatch.setattr(
            page._shell.controls,
            "current_form_edits",
            lambda: pytest.fail(
                "diagnostic full-form snapshot must not be polled"
            ),
        )
        monkeypatch.setattr(page, "_begin_admission", captures.append)

        _run(page)

        assert store.commit_calls == 0
        assert len(captures) == 1
        captured = captures[0].intent_snapshot
        assert captured.revision == 0
        assert (
            integration_values(captured.thaw())[INT_1D_POINTS]
            == 1000
        )
    finally:
        _close(page)
