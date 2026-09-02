from __future__ import annotations

from pathlib import Path

import pytest
from pyqtgraph.Qt import QtWidgets

from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import image_series_spec
from xdart.gui.tabs.scattering.contracts import (
    SourceCapture,
    SourceObservation,
    SourceObservationRequest,
    SourceObservationStatus,
)
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.display_values import (
    StandardEventKind,
    StandardRunEvent,
)
from xdart.gui.tabs.scattering.events import (
    CleanupStatus,
    DetachedDiagnostic,
    ExecutorAccepted,
    ExecutorClosed,
    PreflightAccepted,
    RunIdentity,
)
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xdart.gui.tabs.scattering.start_outcomes import executor_closed_is_valid
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell
from tests.xdart.scattering._admission import ImmediateAdmission


class _UnprintableError(RuntimeError):
    def __str__(self) -> str:
        raise RuntimeError("string projection also failed")


class _Sources:
    def __init__(self) -> None:
        self.epoch = 0

    def capture(self, source, request_id):
        self.epoch += 1
        return SourceCapture(request_id, self.epoch, source)

    def cancel(self, _request_id) -> None:
        return None

    def observe(self, request: SourceObservationRequest) -> SourceObservation:
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


class _Executor(ImmediateAdmission):
    def __init__(self) -> None:
        self.events: list[object] = []
        self.start_calls = 0
        self.stop_error: Exception | None = None
        self.drain_error: Exception | None = None

    def start(self, _configuration, _source, identity, _admission):
        self.start_calls += 1
        return ExecutorAccepted(identity)

    def stop(self, _identity) -> None:
        if self.stop_error is not None:
            raise self.stop_error

    def close(self, identity):
        return ExecutorClosed(identity, CleanupStatus.CLEANED)

    def pause(self, _identity) -> None:
        return None

    def resume(self, _identity) -> None:
        return None

    def drain_events(self):
        if self.drain_error is not None:
            raise self.drain_error
        events, self.events = tuple(self.events), []
        return events

@pytest.fixture
def qapp() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _active_page(
    executor: _Executor,
    *,
    intents: RunIntentStore | None = None,
) -> tuple[ScatteringWorkspace, ScatteringCoordinator, RunIdentity]:
    lifecycle = ScatteringCoordinator()
    configuration = RunIntent().freeze()
    begun = lifecycle.begin_start()
    promoted = lifecycle.preflight_accepted(
        PreflightAccepted(begun.request_id, configuration)
    )
    identity = promoted.run_identity
    assert identity is not None
    assert lifecycle.executor_accepted(ExecutorAccepted(identity)).phase is RunPhase.RUNNING
    page = ScatteringWorkspace(
        intents=intents or RunIntentStore(),
        lifecycle=lifecycle,
        sources=_Sources(),
        executor=executor,
    )
    return page, lifecycle, identity


def _dispose(page: ScatteringWorkspace, qapp: QtWidgets.QApplication) -> None:
    page.close_workspace()
    page.deleteLater()
    qapp.processEvents()


def test_throwing_drain_exception_text_is_contained_at_page_boundary(
    qapp: QtWidgets.QApplication,
    caplog: pytest.LogCaptureFixture,
) -> None:
    executor = _Executor()
    page, lifecycle, _identity = _active_page(executor)
    shell = page.findChild(ScatteringWorkspaceShell)
    assert shell is not None
    try:
        executor.drain_error = _UnprintableError()
        page._drain_executor()
        assert lifecycle.phase is RunPhase.RUNNING
        assert shell.scientific.status.text()
        records = [
            record for record in caplog.records
            if record.name == "xdart.gui.tabs.scattering.page"
            and record.getMessage() == "Standard event drain failed"
        ]
        assert len(records) == 1
        assert records[0].exc_info is not None
        assert records[0].exc_info[0] is _UnprintableError
    finally:
        executor.drain_error = None
        _dispose(page, qapp)


def test_throwing_shell_apply_is_contained_at_page_boundary(
    qapp: QtWidgets.QApplication,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    executor = _Executor()
    page, lifecycle, _identity = _active_page(executor)
    shell = page.findChild(ScatteringWorkspaceShell)
    assert shell is not None

    commits: list[tuple[object, ...]] = []

    def apply_state(_projection, *, preserve_display=False) -> None:
        raise _UnprintableError()

    monkeypatch.setattr(shell, "apply_state", apply_state)
    monkeypatch.setattr(
        page._context_controller,
        "commit_navigation_projection",
        lambda frames: commits.append(frames),
    )
    try:
        page._refresh_shell()
        assert lifecycle.phase is RunPhase.RUNNING
        assert page._notice_text == (
            "Passive shell render failed: <unprintable>"
        )
        assert commits == []
        records = [
            record for record in caplog.records
            if record.name == "xdart.gui.tabs.scattering.page"
            and record.getMessage() == "Passive shell render failed"
        ]
        assert len(records) == 1
        assert records[0].exc_info is not None
        assert records[0].exc_info[0] is _UnprintableError
    finally:
        monkeypatch.undo()
        _dispose(page, qapp)


def test_successful_shell_apply_commits_exact_live_projection(
    qapp: QtWidgets.QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = _Executor()
    page, lifecycle, _identity = _active_page(executor)
    shell = page.findChild(ScatteringWorkspaceShell)
    assert shell is not None
    projected: list[dict[str, object]] = []
    commits: list[tuple[object, ...]] = []
    presented = (object(), object())

    def project_navigation(**kwargs):
        projected.append(kwargs)
        return ()

    def apply_state(_projection, *, preserve_display=False) -> None:
        shell.scientific._trace_history_keys = presented

    monkeypatch.setattr(
        page._context_controller,
        "project_navigation",
        project_navigation,
    )
    monkeypatch.setattr(
        page._context_controller,
        "commit_navigation_projection",
        lambda frames: commits.append(frames),
    )
    monkeypatch.setattr(shell, "apply_state", apply_state)
    try:
        page._refresh_shell()

        assert lifecycle.phase is RunPhase.RUNNING
        assert projected == [{
            "preferences": page._preferences,
            "processing_mode": (
                page._intents.snapshot().thaw().processing_mode
            ),
            "live_update": True,
        }]
        assert commits == [presented]
    finally:
        monkeypatch.undo()
        _dispose(page, qapp)


def test_cleanup_failure_operation_must_be_nonempty() -> None:
    identity = RunIdentity(1, "f" * 64)
    unnamed = DetachedDiagnostic("builtins", "RuntimeError", "abort failed", "")
    receipt = ExecutorClosed(
        identity,
        CleanupStatus.CLEANUP_PENDING,
        cleanup_failures=(unnamed,),
    )
    assert not executor_closed_is_valid(receipt, identity)


def test_cleanup_status_must_be_exact_enum() -> None:
    identity = RunIdentity(1, "f" * 64)
    receipt = ExecutorClosed(identity, "cleaned")  # type: ignore[arg-type]
    assert not executor_closed_is_valid(receipt, identity)


def test_cleaned_failure_can_launch_a_real_next_attempt(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    executor = _Executor()
    source = image_series_spec(tmp_path / "frame_0001.tif")
    intents = RunIntentStore(
        RunIntent(
            source_spec=source,
            poni_file=str(tmp_path / "calibration.poni"),
            save_path=str(tmp_path / "output.nxs"),
            output_mode="Overwrite",
        )
    )
    page, lifecycle, identity = _active_page(executor, intents=intents)
    executor.events.append(
        StandardRunEvent(
            identity,
            StandardEventKind.FAILED,
            cleanup_status=CleanupStatus.CLEANED,
            detail="contained failure",
        )
    )
    try:
        page._drain_executor()
        assert lifecycle.phase is RunPhase.FAILED
        shell = page.findChild(ScatteringWorkspaceShell)
        assert shell is not None
        shell.commandRequested.emit(
            ShellCommand(ShellCommandKind.RUN_ACTION)
        )
        page._drain_executor()
        assert lifecycle.phase is RunPhase.RUNNING
        assert executor.start_calls == 1
        assert page._run_timer.isActive()
    finally:
        _dispose(page, qapp)
