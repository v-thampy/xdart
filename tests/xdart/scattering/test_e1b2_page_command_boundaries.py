from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from pyqtgraph.Qt import QtWidgets

from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import image_series_spec
from xdart.gui.tabs.scattering.contracts import (
    SourceCapture, SourceObservation, SourceObservationRequest, SourceObservationStatus,
)
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.display_values import (
    DisplayFrameKey,
    DisplayNavigationDelta,
    StandardEventKind,
    StandardRunEvent,
)
from xdart.gui.tabs.scattering.events import (
    CleanupStatus, ExecutorAccepted, ExecutorClosed, PreflightAccepted, RunIdentity,
)
from xdart.gui.tabs.scattering.display_retirement import (
    DisplayRetirementReceipt,
)
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.shell_values import (
    FrameNavigationProjection,
    ShellCommand,
    ShellCommandKind,
)
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell
from tests.xdart.scattering._admission import (
    ImmediateAdmission,
    admission_for,
)


@pytest.fixture
def qapp() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _Sources:
    def __init__(self) -> None:
        self.epoch = 0

    def capture(self, source, request_id):
        self.epoch += 1
        return SourceCapture(request_id, self.epoch, source)

    def cancel(self, _request_id) -> None:
        return None

    def observe(self, request: SourceObservationRequest) -> SourceObservation:
        return SourceObservation(request.observation_id, request.intent_revision, request.source,
                                 SourceObservationStatus.AVAILABLE, "frame.tif", True, False)

    def cancel_observation(self, _observation_id: int) -> None:
        return None

    def publish_motor_knowledge(self, _observation) -> None:
        return None

    def project_motor_knowledge(self, _source, _fingerprint=None):
        return None


class _Executor(ImmediateAdmission):
    def __init__(self) -> None:
        self.events: list[object] = []
        self.stop_error: Exception | None = None
        self.start_calls = 0
        self.last_identity: RunIdentity | None = None

    def begin_admission(self, capture):
        token = super().begin_admission(capture)
        identity = self.last_identity
        if identity is not None:
            self._test_admission = (
                token,
                replace(
                    admission_for(capture),
                    display_retirement=DisplayRetirementReceipt(
                        identity, CleanupStatus.CLEANED
                    ),
                ),
            )
        return token

    def start(self, _configuration, _source, run_identity, _admission):
        self.start_calls += 1
        self.last_identity = run_identity
        return ExecutorAccepted(run_identity)

    def stop(self, _run_identity) -> None:
        if self.stop_error is not None:
            raise self.stop_error

    def close(self, run_identity):
        return ExecutorClosed(run_identity, CleanupStatus.CLEANED)

    def pause(self, _run_identity) -> None:
        return None

    def resume(self, _run_identity) -> None:
        return None

    def drain_events(self):
        events, self.events = tuple(self.events), []
        return events

def _active_page(executor: _Executor) -> tuple[ScatteringWorkspace, ScatteringCoordinator, RunIdentity]:
    lifecycle = ScatteringCoordinator()
    request = lifecycle.begin_start().request_id
    assert request is not None
    configuration = RunIntent().freeze()
    identity = lifecycle.preflight_accepted(PreflightAccepted(request, configuration)).run_identity
    assert identity is not None
    assert lifecycle.executor_accepted(ExecutorAccepted(identity)).phase is RunPhase.RUNNING
    page = ScatteringWorkspace(
        intents=RunIntentStore(
            RunIntent(
                source_spec=image_series_spec(Path("frame_0001.tif")),
                poni_file="calibration.poni",
                save_path="output.nxs",
                output_mode="Overwrite",
            )
        ),
        lifecycle=lifecycle,
        sources=_Sources(),
        executor=executor,
    )
    return page, lifecycle, identity


def _dispose(page: ScatteringWorkspace, qapp: QtWidgets.QApplication) -> None:
    page.close_workspace()
    page.deleteLater()
    qapp.processEvents()


def _shell(page: ScatteringWorkspace) -> ScatteringWorkspaceShell:
    shell = page.findChild(ScatteringWorkspaceShell)
    assert shell is not None
    return shell


def test_pending_failure_cannot_reset_or_launch(qapp: QtWidgets.QApplication) -> None:
    executor = _Executor()
    page, lifecycle, identity = _active_page(executor)
    executor.events.append(StandardRunEvent(identity, StandardEventKind.FAILED,
                                             cleanup_status=CleanupStatus.CLEANUP_PENDING))
    try:
        page._drain_executor()
        shell = _shell(page)
        shell.commandRequested.emit(
            ShellCommand(ShellCommandKind.RUN_ACTION)
        )
        page._drain_executor()
        assert lifecycle.phase is RunPhase.FAILED
        assert executor.start_calls == 0
        assert not shell.run_controls.startButton.isEnabled()
        assert (
            shell.run_controls.readinessLabel.text()
            == "Standard cleanup remains pending"
        )
    finally:
        _dispose(page, qapp)


def test_executor_stop_failure_is_contained_at_qt_command_boundary(qapp: QtWidgets.QApplication) -> None:
    executor = _Executor()
    executor.stop_error = RuntimeError("stop dispatch failed")
    page, lifecycle, identity = _active_page(executor)
    try:
        shell = _shell(page)
        shell.commandRequested.emit(ShellCommand(ShellCommandKind.STOP))
        assert lifecycle.active_run_identity is identity
        assert lifecycle.phase is RunPhase.STOPPING
        assert "stop" in shell.scientific.status.text().lower()
    finally:
        executor.stop_error = None
        _dispose(page, qapp)


def test_executor_drain_timer_tracks_only_launched_run_lifetime(qapp: QtWidgets.QApplication,
                                                                 tmp_path: Path) -> None:
    executor = _Executor()
    source = image_series_spec(tmp_path / "frame_0001.tif")
    lifecycle = ScatteringCoordinator()
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(source_spec=source, poni_file=str(tmp_path / "calibration.poni"),
                                          save_path=str(tmp_path / "output.nxs"),
                                          output_mode="Overwrite")),
        lifecycle=lifecycle, sources=_Sources(), executor=executor,
    )
    try:
        assert lifecycle.phase is RunPhase.IDLE
        assert not page._run_timer.isActive()
        assert page._run_timer.interval() == 125
        shell = _shell(page)
        shell.commandRequested.emit(
            ShellCommand(ShellCommandKind.RUN_ACTION)
        )
        page._drain_executor()
        identity = lifecycle.active_run_identity
        assert identity is not None
        assert lifecycle.phase is RunPhase.RUNNING
        assert page._run_timer.isActive()
        executor.events.append(StandardRunEvent(identity, StandardEventKind.FINISHED,
                                                 artifact=str(tmp_path / "output.nxs"),
                                                 cleanup_status=CleanupStatus.CLEANED))
        page._drain_executor()
        assert lifecycle.phase is RunPhase.IDLE
        assert not page._run_timer.isActive()
        assert executor.start_calls == 1
    finally:
        _dispose(page, qapp)


def test_run_click_preserves_outgoing_paint_until_a_frame_arrives(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
    monkeypatch,
) -> None:
    executor = _Executor()
    page = ScatteringWorkspace(
        intents=RunIntentStore(
            RunIntent(
                source_spec=image_series_spec(tmp_path / "frame_0001.tif"),
                poni_file=str(tmp_path / "calibration.poni"),
                save_path=str(tmp_path / "output.nxs"),
                output_mode="Overwrite",
            )
        ),
        lifecycle=ScatteringCoordinator(),
        sources=_Sources(),
        executor=executor,
    )
    refreshes: list[bool] = []
    monkeypatch.setattr(
        page,
        "_refresh_shell",
        lambda *, preserve_display=False: refreshes.append(
            preserve_display
        ),
    )
    try:
        _shell(page).commandRequested.emit(
            ShellCommand(ShellCommandKind.RUN_ACTION)
        )
        assert refreshes == [True]

        page._drain_executor()

        assert executor.start_calls == 1
        assert page._lifecycle.phase is RunPhase.RUNNING
        assert refreshes == [True, True]
    finally:
        _dispose(page, qapp)


def test_historical_frame_disables_auto_last_and_reenable_selects_latest(
    qapp: QtWidgets.QApplication,
    monkeypatch,
) -> None:
    executor = _Executor()
    page, _, identity = _active_page(executor)
    first = DisplayFrameKey(identity, "scan", "output.nxs", 1, 1)
    latest = DisplayFrameKey(identity, "scan", "output.nxs", 2, 2)
    runtime = page._context_controller._runtime
    runtime._acquisition_navigation = FrameNavigationProjection(
        (first, latest),
        latest,
        (latest,),
    )

    def select_navigation(frame, frames):
        runtime._acquisition_navigation = FrameNavigationProjection(
            (first, latest),
            frame,
            frames,
        )
        return True

    selected_plot_modes: list[str] = []

    def select_latest_navigation(*, plot_mode="Single"):
        selected_plot_modes.append(plot_mode)
        runtime._acquisition_navigation = FrameNavigationProjection(
            (first, latest),
            latest,
            (latest,),
        )
        return True

    monkeypatch.setattr(
        page._context_controller,
        "owns_frame",
        lambda frame: frame in {first, latest},
    )
    monkeypatch.setattr(
        page._context_controller,
        "select_navigation",
        select_navigation,
    )
    monkeypatch.setattr(
        page._context_controller,
        "select_latest_navigation",
        select_latest_navigation,
    )
    try:
        page._handle_shell_command(
            ShellCommand(
                ShellCommandKind.SELECT_FRAME,
                frame=first,
                frames=(first,),
            )
        )
        assert page._auto_last is False
        assert page._context_controller.navigation.current is first

        page._handle_shell_command(
            ShellCommand(ShellCommandKind.SET_AUTO_LAST, True)
        )
        assert page._auto_last is True
        assert selected_plot_modes == ["Single"]
        assert page._context_controller.navigation.current is latest
        assert page._context_controller.navigation.selected == (latest,)
    finally:
        _dispose(page, qapp)


def test_cold_frame_refresh_catches_up_distinct_acquisition_owner_before_paint(
    qapp: QtWidgets.QApplication,
    monkeypatch,
) -> None:
    from tests.xdart.scattering.test_e3_context_contract import _acquisition

    executor = _Executor()
    page, _, identity = _active_page(executor)
    configuration = RunIntent().freeze()
    _, acquisition = _acquisition(
        configuration=configuration,
        identity=identity,
    )
    executor.acquisition_context = lambda candidate: (
        acquisition if candidate is identity else None
    )
    controller = page._context_controller
    controller.adopt_acquisition(identity)
    first = controller.navigation.current
    assert first is not None
    acquisition.rescope_to("run.b", "/data/b_0001.tif")
    assert controller.project_navigation() == ()
    page._retain_outgoing_display = True
    monkeypatch.setattr(page, "_follow_processed_artifact", lambda _frame: None)

    try:
        executor.events.append(
            StandardRunEvent(
                identity,
                StandardEventKind.FRAME_READY,
                completed=1,
                total=2,
                artifact=first.artifact,
                frame_key=first,
                navigation_delta=DisplayNavigationDelta(first),
            )
        )
        page._drain_executor()

        selection = controller.selection
        assert selection is not None
        assert selection.owner == acquisition.hydration_owner
        assert controller.navigation.current is first
        assert page._retain_outgoing_display is False
        shell = _shell(page)
        assert shell.scientific.title.text() != "Current"
        assert shell.scientific.raw.image.image is not None
        assert shell.scientific.cake.image.image is not None
        assert shell.scientific.curve.listDataItems()
    finally:
        _dispose(page, qapp)


def test_batch_run_defers_frame_paints_and_follows_latest_at_terminal(
    qapp: QtWidgets.QApplication,
    monkeypatch,
) -> None:
    executor = _Executor()
    page, lifecycle, identity = _active_page(executor)
    frame = DisplayFrameKey(identity, "scan", "output.nxs", 7, 1)
    delta = DisplayNavigationDelta(frame)
    refreshes: list[None] = []
    followed: list[DisplayFrameKey] = []
    monkeypatch.setattr(
        page._context_controller,
        "accept_navigation",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(page, "_follow_processed_artifact", followed.append)
    monkeypatch.setattr(
        page, "_refresh_shell", lambda: refreshes.append(None)
    )
    page._active_batch_mode = True
    try:
        executor.events.append(
            StandardRunEvent(
                identity,
                StandardEventKind.FRAME_READY,
                completed=1,
                total=1,
                artifact=frame.artifact,
                frame_key=frame,
                navigation_delta=delta,
            )
        )
        page._drain_executor()

        assert followed == []
        assert refreshes == []
        assert lifecycle.phase is RunPhase.RUNNING

        executor.events.append(
            StandardRunEvent(
                identity,
                StandardEventKind.FINISHED,
                completed=1,
                total=1,
                artifact=frame.artifact,
                cleanup_status=CleanupStatus.CLEANED,
            )
        )
        page._drain_executor()

        assert followed == [frame]
        assert refreshes == [None]
        assert lifecycle.phase is RunPhase.IDLE
        assert page._active_batch_mode is False
    finally:
        _dispose(page, qapp)


def test_batch_frame_survives_unrelated_refresh_until_terminal(
    qapp: QtWidgets.QApplication,
    monkeypatch,
) -> None:
    executor = _Executor()
    page, lifecycle, identity = _active_page(executor)
    shell = _shell(page)
    frame = DisplayFrameKey(identity, "scan", "output.nxs", 7, 1)
    applied: list[bool] = []
    apply_state = shell.apply_state

    def record_apply(state, *, preserve_display=False):
        apply_state(state, preserve_display=preserve_display)
        applied.append(preserve_display)

    monkeypatch.setattr(shell, "apply_state", record_apply)
    monkeypatch.setattr(
        page._context_controller,
        "accept_navigation",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(page, "_follow_processed_artifact", lambda _frame: None)
    page._active_batch_mode = True
    page._retain_outgoing_display = True
    try:
        executor.events.append(StandardRunEvent(
            identity,
            StandardEventKind.FRAME_READY,
            completed=1,
            total=1,
            artifact=frame.artifact,
            frame_key=frame,
            navigation_delta=DisplayNavigationDelta(frame),
        ))
        page._drain_executor()
        assert applied == []

        # A readiness/browser callback may refresh controls while the batch is
        # running; it must not expose the deferred frame or clear old paint.
        page._refresh_shell()
        assert applied == [True]

        executor.events.append(StandardRunEvent(
            identity,
            StandardEventKind.FINISHED,
            completed=1,
            total=1,
            artifact=frame.artifact,
            cleanup_status=CleanupStatus.CLEANED,
        ))
        page._drain_executor()

        assert lifecycle.phase is RunPhase.IDLE
        assert page._retain_outgoing_display is False
        assert applied[-1] is False
    finally:
        _dispose(page, qapp)


def test_run_retries_transient_prior_display_cleanup_without_second_click(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
    monkeypatch,
) -> None:
    executor = _Executor()
    source = image_series_spec(tmp_path / "frame_0001.tif")
    page = ScatteringWorkspace(
        intents=RunIntentStore(
            RunIntent(
                source_spec=source,
                poni_file=str(tmp_path / "calibration.poni"),
                save_path=str(tmp_path / "output.nxs"),
                output_mode="Overwrite",
            )
        ),
        lifecycle=ScatteringCoordinator(),
        sources=_Sources(),
        executor=executor,
    )
    attempts: list[object] = []

    def fail_once(receipt):
        attempts.append(receipt)
        return len(attempts) > 1

    monkeypatch.setattr(
        page._context_controller,
        "apply_display_retirement",
        fail_once,
    )
    try:
        _shell(page).commandRequested.emit(
            ShellCommand(ShellCommandKind.RUN_ACTION)
        )
        page._drain_executor()
        assert executor.start_calls == 0
        assert page._admission is not None
        assert "Prior display cleanup" not in page._notice_text

        page._drain_executor()
        assert executor.start_calls == 1
        assert page._admission is None
        assert attempts[0] is attempts[1]
    finally:
        _dispose(page, qapp)
