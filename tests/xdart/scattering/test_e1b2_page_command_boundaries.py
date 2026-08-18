from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from xrd_tools.core import FrameView
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
    StandardDisplayPayload,
    StandardEventKind,
    StandardRunEvent,
    StandardTerminalTiming,
)
from xdart.gui.tabs.scattering.events import (
    CleanupStatus, ExecutorAccepted, ExecutorClosed, PreflightAccepted, RunIdentity,
)
from xdart.gui.tabs.scattering.display_retirement import (
    DisplayRetirementReceipt,
)
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
import xdart.gui.tabs.scattering.page as page_module
from xdart.gui.tabs.scattering.performance_diagnostics import (
    PerformanceDiagnosticsValues,
)
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

def _active_page(
    executor: _Executor,
    *,
    output_mode: str = "Overwrite",
) -> tuple[ScatteringWorkspace, ScatteringCoordinator, RunIdentity]:
    lifecycle = ScatteringCoordinator()
    request = lifecycle.begin_start().request_id
    assert request is not None
    configuration = RunIntent(output_mode=output_mode).freeze()
    identity = lifecycle.preflight_accepted(PreflightAccepted(request, configuration)).run_identity
    assert identity is not None
    assert lifecycle.executor_accepted(ExecutorAccepted(identity)).phase is RunPhase.RUNNING
    page = ScatteringWorkspace(
        intents=RunIntentStore(
            RunIntent(
                source_spec=image_series_spec(Path("frame_0001.tif")),
                poni_file="calibration.poni",
                save_path="output.nxs",
                output_mode=output_mode,
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


def _paced_frame_events(page, executor, identity, count):
    from tests.xdart.scattering.test_e3_context_contract import _acquisition

    _, acquisition = _acquisition(
        configuration=RunIntent(output_mode="Overwrite").freeze(), identity=identity)
    executor.acquisition_context = lambda candidate: acquisition if candidate is identity else None
    acquisition.publication_store.catalog.resize(max(16, count))
    page._context_controller.adopt_acquisition(identity)
    display = acquisition.publication_store
    deltas = [DisplayNavigationDelta(display.catalog_snapshot().entries[0])]
    deltas.extend(display.append_navigation("run.a", "/out/a.nxs", label)
                  for label in range(2, count + 1))
    return tuple(
        StandardRunEvent(
            identity,
            StandardEventKind.FRAME_READY,
            completed=index,
            total=count,
            artifact=delta.appended.artifact,
            frame_key=delta.appended,
            navigation_delta=delta,
            artifact_completed=index,
            artifact_total=count,
        )
        for index, delta in enumerate(deltas, start=1)
    )


def test_performance_diagnostics_are_explicit_and_apply_next_run_values(
        qapp: QtWidgets.QApplication, monkeypatch) -> None:
    monkeypatch.delenv("XDART_PERF", raising=False)
    monkeypatch.delenv(
        "XDART_UNSAFE_UNFUNDED_STAGING_DIAGNOSTIC", raising=False,
    )
    executor = _Executor()
    page, _, _ = _active_page(executor)
    shell = _shell(page)
    try:
        config = shell.browser.findChild(
            QtWidgets.QToolButton, "configMenuButton",
        )
        assert config is not None
        assert "Performance Diagnostics…" in {
            action.text() for action in config.menu().actions()
        }
        initial = page._intents.snapshot()
        assert "_post_g2_pipeline_v2" not in initial.thaw().run_options
        assert "XDART_PERF" not in os.environ

        page._performance_diagnostics_editor = lambda *_args: (
            PerformanceDiagnosticsValues(
                1, 8, 16, 56, 375,
                save_xye=False,
                durable_fsync=False,
                staging_frame_cap=64,
            )
        )
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.MENU, "Config:Performance Diagnostics…",
        ))

        applied = page._intents.snapshot()
        assert applied.revision == initial.revision + 1
        assert applied.thaw().run_options["_post_g2_pipeline_v2"] == {
            "writer_settlement_batch_size": 1,
            "nexus_record_batch_size": 8,
            "reduction_inflight": 16,
            "semantic_checkpoint_frame_cap": 56,
            "staging_frame_cap": 64,
        }
        assert applied.thaw().run_options[
            "_post_g2_output_diagnostics_v1"
        ] == {
            "save_xye": False,
            "durable_fsync": False,
        }
        assert page._live_plot_interval_ms == 375
        assert os.environ["XDART_PERF"] == "1"
        assert "next run" in page._notice_text.lower()
        assert "crash" in page._notice_text.lower()

        monkeypatch.setenv(
            "XDART_UNSAFE_UNFUNDED_STAGING_DIAGNOSTIC", "1",
        )
        page._performance_diagnostics_editor = lambda *_args: (
            PerformanceDiagnosticsValues(
                1, 8, 8, 10_000, 375,
                staging_frame_cap=10_008,
            )
        )
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.MENU, "Config:Performance Diagnostics…",
        ))
        unsafe = page._intents.snapshot()
        assert unsafe.revision == applied.revision + 1
        assert unsafe.thaw().run_options[
            "_post_g2_unfunded_staging_diagnostic_v1"
        ] == {
            "mode": "UNSAFE_UNFUNDED",
            "checkpoint": 10_000,
            "staging_frame_cap": 10_008,
            "max_frames": 3_621,
        }
        assert "unsafe unfunded" in page._notice_text.lower()

        page._performance_diagnostics_editor = lambda *_args: (
            PerformanceDiagnosticsValues(
                1, 8, 8, 56, 375,
                staging_frame_cap=64,
            )
        )
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.MENU, "Config:Performance Diagnostics…",
        ))
        ordinary = page._intents.snapshot()
        assert ordinary.revision == unsafe.revision + 1
        assert (
            "_post_g2_unfunded_staging_diagnostic_v1"
            not in ordinary.thaw().run_options
        )

        page._performance_diagnostics_editor = lambda *_args: None
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.MENU, "Config:Performance Diagnostics…",
        ))
        assert page._intents.snapshot().revision == ordinary.revision
        assert page._live_plot_interval_ms == 375
    finally:
        _dispose(page, qapp)


def test_performance_diagnostics_reject_invalid_coupled_values(
        qapp: QtWidgets.QApplication, monkeypatch) -> None:
    monkeypatch.delenv("XDART_PERF", raising=False)
    page, _, _ = _active_page(_Executor())
    initial = page._intents.snapshot()
    try:
        page._performance_diagnostics_editor = lambda *_args: None
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.MENU, "Config:Performance Diagnostics…",
        ))
        assert page._intents.snapshot().revision == initial.revision
        assert "XDART_PERF" not in os.environ

        page._performance_diagnostics_editor = lambda *_args: (
            PerformanceDiagnosticsValues(5, 8, 4, 56, 125)
        )
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.MENU, "Config:Performance Diagnostics…",
        ))
        assert page._intents.snapshot().revision == initial.revision
        assert "batching bounds" in page._notice_text.lower()
        assert page._live_plot_interval_ms == 125
        assert "XDART_PERF" not in os.environ

        page._performance_diagnostics_editor = lambda *_args: (
            PerformanceDiagnosticsValues(
                1, 8, 8, 1_000, 125,
                staging_frame_cap=64,
            )
        )
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.MENU, "Config:Performance Diagnostics…",
        ))
        assert page._intents.snapshot().revision == initial.revision
        assert "staging" in page._notice_text.lower()
        assert "checkpoint" in page._notice_text.lower()
    finally:
        _dispose(page, qapp)


def test_terminal_elapsed_and_split_remain_in_run_status(
        qapp: QtWidgets.QApplication) -> None:
    executor = _Executor()
    page, _, identity = _active_page(executor)
    timing = StandardTerminalTiming(
        23.414,
        20.0,
        3.414,
        (
            ("source_read", 5.25),
            ("submit_wait", 12.5),
            ("writer_batch", 1.25),
            ("writer_flush", 0.5),
            ("xye", 2.0),
            ("finish_wait", 3.0),
            ("display", 0.75),
        ),
    )
    executor.events.append(StandardRunEvent(
        identity,
        StandardEventKind.FINISHED,
        completed=651,
        total=651,
        terminal_timing=timing,
    ))
    try:
        page._drain_executor()
        label = _shell(page).run_controls.readinessLabel
        assert label.text() == "Complete · 23.41 s"
        assert label.toolTip() == (
            "Total: 23.41 s\n"
            "Work: 20.00 s\n"
            "Cleanup: 3.41 s\n"
            "Source read: 5.25 s\n"
            "Submit/backpressure: 12.50 s\n"
            "NeXus write/checkpoint: 1.25 s\n"
            "Checkpoint flush/fsync: 0.50 s\n"
            "XYE: 2.00 s\n"
            "Finish/drain (includes terminal seal): 3.00 s\n"
            "Display: 0.75 s\n"
            "Parallel detail timers may overlap."
        )
        page._refresh_shell()
        assert label.text() == "Complete · 23.41 s"
    finally:
        _dispose(page, qapp)


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


@pytest.mark.parametrize(
    ("plot_interval_ms", "expected_light_refreshes"),
    ((125, (False, False, False)), (375, (False, True, False))),
)
def test_single_auto_last_paces_eight_frame_burst_to_same_drain_latest(
        qapp: QtWidgets.QApplication, monkeypatch,
        plot_interval_ms: int,
        expected_light_refreshes: tuple[bool, bool, bool]) -> None:
    monkeypatch.setenv(
        "XDART_LIVE_PLOT_INTERVAL_MS", str(plot_interval_ms),
    )
    executor = _Executor()
    page, _, identity = _active_page(executor)
    events = _paced_frame_events(page, executor, identity, 24)
    now = [0.0]
    monkeypatch.setattr(
        page_module, "time", SimpleNamespace(monotonic=lambda: now[0]),
        raising=False,
    )
    assert page._live_plot_interval_ms == plot_interval_ms
    page._last_live_plot_at = None
    accepted_follow_latest: list[bool] = []
    accept_navigation = page._context_controller.accept_navigation

    def accept(delta, *, plot_mode="Single", follow_latest=True):
        accepted_follow_latest.append(follow_latest)
        return accept_navigation(
            delta,
            plot_mode=plot_mode,
            follow_latest=follow_latest,
        )

    monkeypatch.setattr(page._context_controller, "accept_navigation", accept)
    monkeypatch.setattr(page, "_follow_processed_artifact", lambda _frame: None)
    paints: list[tuple[int | None, bool]] = []
    def record_paint(*, preserve_scientific=False, **_kwargs):
        current = page._context_controller.navigation.current
        paints.append((
            None if current is None else current.local_frame_label,
            preserve_scientific,
        ))
        if not preserve_scientific:
            page._last_live_plot_at = now[0]

    monkeypatch.setattr(page, "_refresh_shell", record_paint)
    try:
        executor.events.extend(events[:8])
        page._drain_executor()

        assert tuple(
            frame.local_frame_label
            for frame in page._context_controller.navigation.frames
        ) == (
            1, 2, 3, 4, 5, 6, 7, 8,
        )
        assert page._progress.completed == 8
        assert page._artifact_progress["/out/a.nxs"].published == 8
        assert accepted_follow_latest == [False] * 8
        assert paints == [(8, expected_light_refreshes[0])]
        assert tuple(page._presentation_targets) == ()

        for _ in range(3):
            page._drain_executor()

        assert paints == [(8, expected_light_refreshes[0])]
        assert tuple(page._presentation_targets) == ()

        now[0] = 0.125
        executor.events.extend(events[8:16])
        page._drain_executor()

        assert tuple(
            frame.local_frame_label
            for frame in page._context_controller.navigation.frames
        ) == tuple(range(1, 17))
        assert page._progress.completed == 16
        assert page._artifact_progress["/out/a.nxs"].published == 16
        assert accepted_follow_latest == [False] * 16
        assert paints == [
            (8, expected_light_refreshes[0]),
            (16, expected_light_refreshes[1]),
        ]
        assert tuple(page._presentation_targets) == ()

        for _ in range(3):
            page._drain_executor()

        assert paints == [
            (8, expected_light_refreshes[0]),
            (16, expected_light_refreshes[1]),
        ]
        assert tuple(page._presentation_targets) == ()

        now[0] = 0.375
        executor.events.extend(events[16:])
        page._drain_executor()

        assert tuple(
            frame.local_frame_label
            for frame in page._context_controller.navigation.frames
        ) == tuple(range(1, 25))
        assert page._progress.completed == 24
        assert accepted_follow_latest == [False] * 24
        assert paints == [
            (8, expected_light_refreshes[0]),
            (16, expected_light_refreshes[1]),
            (24, expected_light_refreshes[2]),
        ]
    finally:
        _dispose(page, qapp)


def test_pacer_boundaries_clear_stale_work_and_flush_exact_latest(
        qapp: QtWidgets.QApplication, monkeypatch) -> None:
    executor = _Executor()
    page, lifecycle, identity = _active_page(executor)
    events = _paced_frame_events(page, executor, identity, 5)
    monkeypatch.setattr(page, "_follow_processed_artifact", lambda _frame: None)
    monkeypatch.setattr(page, "_refresh_shell", lambda **_kwargs: None)
    try:
        executor.events.extend(events)
        page._drain_executor()
        assert tuple(page._presentation_targets) == ()
        assert page._context_controller.navigation.current is events[-1].frame_key

        first = events[0].frame_key
        assert first is not None
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.SELECT_FRAME,
            frame=first,
            frames=(first,),
        ))
        assert tuple(page._presentation_targets) == ()
        assert page._auto_last is False
        assert page._context_controller.navigation.current is first

        latest = events[-1].frame_key
        assert latest is not None

        def refill(*targets):
            page._presentation_targets.extend(targets or (latest,))
            page._presentation_run_identity = identity

        refill(*(
            event.frame_key for event in events[-3:]
            if event.frame_key is not None
        ))
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.SET_AUTO_LAST, True,
        ))
        assert tuple(page._presentation_targets) == ()
        assert page._presentation_run_identity is None
        assert page._context_controller.navigation.current is latest

        refill()
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.SET_PLOT_MODE, "Overlay",
        ))
        assert tuple(page._presentation_targets) == ()
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.SET_PLOT_MODE, "Single",
        ))

        dispatched: list[tuple[str, int]] = []

        def pause():
            current = page._context_controller.navigation.current
            dispatched.append(("pause", current.local_frame_label))

        refill()
        monkeypatch.setattr(page._context_controller, "pause", pause)
        page._handle_shell_command(ShellCommand(ShellCommandKind.RUN_ACTION))
        assert dispatched == [("pause", 5)]
        assert tuple(page._presentation_targets) == ()

        def stop():
            current = page._context_controller.navigation.current
            dispatched.append(("stop", current.local_frame_label))

        refill()
        monkeypatch.setattr(page._context_controller, "stop", stop)
        page._handle_shell_command(ShellCommand(ShellCommandKind.STOP))
        assert dispatched[-1] == ("stop", 5)
        assert tuple(page._presentation_targets) == ()

        foreign_identity = RunIdentity(
            identity.generation + 1, f"{identity.fingerprint}-stale",
        )
        stale = DisplayFrameKey(
            foreign_identity, "run.a", "/out/a.nxs", 99, 99,
        )
        refill(stale, latest)
        assert page._context_controller.select_navigation(
            events[0].frame_key, (events[0].frame_key,),
        )
        page._drain_executor()
        assert page._context_controller.navigation.current is latest
        assert tuple(page._presentation_targets) == ()

        refill()
        executor.events.append(StandardRunEvent(
            identity, StandardEventKind.FINISHED,
            completed=5, total=5, artifact="/out/a.nxs",
            cleanup_status=CleanupStatus.CLEANED,
        ))
        page._drain_executor()
        assert page._context_controller.navigation.current is latest
        assert tuple(page._presentation_targets) == ()
        assert page._presentation_run_identity is None
        assert lifecycle.phase is RunPhase.IDLE
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


def test_fresh_run_projects_output_checking_not_cleanup(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    executor = _Executor()
    lifecycle = ScatteringCoordinator()
    page = ScatteringWorkspace(
        intents=RunIntentStore(
            RunIntent(
                source_spec=image_series_spec(tmp_path / "frame_0001.tif"),
                poni_file=str(tmp_path / "calibration.poni"),
                save_path=str(tmp_path / "output.nxs"),
                output_mode="Overwrite",
            )
        ),
        lifecycle=lifecycle,
        sources=_Sources(),
        executor=executor,
    )
    try:
        shell = _shell(page)
        shell.commandRequested.emit(
            ShellCommand(ShellCommandKind.RUN_ACTION)
        )

        assert lifecycle.phase is RunPhase.PREPARING
        assert executor.start_calls == 0
        assert page._admission is not None
        assert not shell.run_controls.startButton.isEnabled()
        assert (
            shell.run_controls.readinessLabel.text()
            == "Checking output targets…"
        )
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


def test_noop_append_display_ready_replaces_outgoing_paint_while_running(
    qapp: QtWidgets.QApplication,
    monkeypatch,
) -> None:
    executor = _Executor()
    page, lifecycle, identity = _active_page(
        executor, output_mode="Append",
    )
    frames = tuple(
        DisplayFrameKey(
            identity, "run.append", "/out/noop.nxs", label, label,
        )
        for label in range(1, 6)
    )
    runtime = page._context_controller._runtime
    runtime._acquisition_navigation = FrameNavigationProjection(
        frames, frames[-1], (frames[-1],),
    )
    payloads: dict[DisplayFrameKey, StandardDisplayPayload] = {}
    monkeypatch.setattr(
        runtime,
        "resident_frame_keys",
        lambda *_args: frozenset(payloads),
    )
    monkeypatch.setattr(
        page._context_controller,
        "project_navigation",
        lambda **_kwargs: tuple(
            payloads[frame]
            for frame in runtime.navigation.selected
            if frame in payloads
        ),
    )
    monkeypatch.setattr(
        page._context_controller,
        "owns_frame",
        lambda frame: any(frame is candidate for candidate in frames),
    )

    def select_navigation(frame, selected):
        runtime._acquisition_navigation = FrameNavigationProjection(
            frames, frame, selected,
        )
        return True

    monkeypatch.setattr(
        page._context_controller, "select_navigation", select_navigation,
    )

    def qualify(event):
        if (
            event.run_identity is identity
            and any(event.frame_key is frame for frame in frames)
            and event.selection_generation == 0
        ):
            return payloads.get(event.frame_key)
        return None

    monkeypatch.setattr(
        page._context_controller, "qualify_display_event", qualify,
    )
    shell = _shell(page)
    page._retain_outgoing_display = True
    events: list[StandardRunEvent] = []

    def install_hydrated(label: int) -> DisplayFrameKey:
        key = frames[label - 1]
        raw = np.full((2, 3), float(label))
        view = FrameView(
            label,
            raw=raw,
            thumbnail=raw,
            source_path=f"/data/scan_{label}.tif",
            source_frame_index=label,
        )
        payloads[key] = StandardDisplayPayload(
            0, key, f"frame {label}", view,
        )
        event = StandardRunEvent(
            identity,
            StandardEventKind.DISPLAY_READY,
            artifact=key.artifact,
            frame_key=key,
            selection_generation=0,
        )
        events.append(event)
        executor.events.append(event)
        return key

    try:
        page._refresh_shell(preserve_display=True)
        controller = page._context_controller
        assert tuple(
            frame.local_frame_label for frame in controller.navigation.frames
        ) == (1, 2, 3, 4, 5)
        assert controller.navigation.current is frames[-1]
        assert page._retain_outgoing_display is True
        assert page._run_frame_seen is False

        foreign = (
            StandardRunEvent(
                RunIdentity(
                    identity.generation + 1,
                    f"{identity.fingerprint}-stale",
                ),
                StandardEventKind.DISPLAY_READY,
                artifact=frames[-1].artifact,
                selection_generation=0,
            ),
            StandardRunEvent(
                identity,
                StandardEventKind.DISPLAY_READY,
                artifact="foreign.nxs",
                frame_key=DisplayFrameKey(
                    identity, "foreign", "foreign.nxs", 1, 1,
                ),
                selection_generation=0,
            ),
        )
        events.extend(foreign)
        executor.events.extend(foreign)
        page._drain_executor()
        assert page._retain_outgoing_display is True

        page._active_batch_mode = True
        latest = install_hydrated(5)
        page._drain_executor()
        assert page._retain_outgoing_display is True

        page._active_batch_mode = False
        executor.events.append(events[-1])
        page._drain_executor()
        assert lifecycle.phase is RunPhase.RUNNING
        assert page._run_frame_seen is False
        assert page._retain_outgoing_display is False
        assert not any(
            event.kind is StandardEventKind.FRAME_READY for event in events
        )
        assert shell.scientific.frame_selector.currentData() is latest
        assert shell.scientific.progress.text() == "5/5"
        assert shell.scientific.title.text() == "scan_5.tif"

        first = frames[0]
        shell.scientific.frame_selector.setCurrentIndex(0)
        assert controller.navigation.current is first
        install_hydrated(1)
        page._drain_executor()

        assert lifecycle.phase is RunPhase.RUNNING
        assert page._run_frame_seen is False
        assert not any(
            event.kind is StandardEventKind.FRAME_READY for event in events
        )
        assert shell.scientific.frame_selector.currentData() is first
        assert shell.scientific.progress.text() == "1/5"
        assert shell.scientific.title.text() == "scan_1.tif"
        assert shell.scientific.raw.image.image is not None
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
        # Batch FRAME_READY refreshes scan-local progress while preserving the
        # outgoing scientific paint; follow/paint remains terminal-owned.
        assert refreshes == [None]
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
        assert refreshes == [None, None]
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
        assert applied == [True]

        # A readiness/browser callback may refresh controls while the batch is
        # running; it must not expose the deferred frame or clear old paint.
        page._refresh_shell()
        assert applied == [True, True]

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
        assert (
            _shell(page).run_controls.readinessLabel.text()
            == "Finishing prior display cleanup…"
        )

        page._drain_executor()
        assert executor.start_calls == 1
        assert page._admission is None
        assert attempts[0] is attempts[1]
    finally:
        _dispose(page, qapp)


def test_clean_output_release_projects_pending_display_retirement(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
    monkeypatch,
) -> None:
    executor = _Executor()
    lifecycle = ScatteringCoordinator()
    page = ScatteringWorkspace(
        intents=RunIntentStore(
            RunIntent(
                source_spec=image_series_spec(tmp_path / "frame_0001.tif"),
                poni_file=str(tmp_path / "calibration.poni"),
                save_path=str(tmp_path / "output.nxs"),
                output_mode="Overwrite",
            )
        ),
        lifecycle=lifecycle,
        sources=_Sources(),
        executor=executor,
    )
    allow_retirement = False

    def apply_retirement(_receipt) -> bool:
        return allow_retirement

    monkeypatch.setattr(
        page._context_controller,
        "apply_display_retirement",
        apply_retirement,
    )
    try:
        shell = _shell(page)
        shell.commandRequested.emit(
            ShellCommand(ShellCommandKind.RUN_ACTION)
        )
        page._drain_executor()
        assert page._admission is not None

        shell.commandRequested.emit(ShellCommand(ShellCommandKind.STOP))

        state = page._admission_state
        assert state is not None
        assert state.releasing
        assert state.release_receipt is not None
        assert state.release_receipt.cleanup_status is CleanupStatus.CLEANED
        assert lifecycle.phase is RunPhase.IDLE
        assert (
            shell.run_controls.readinessLabel.text()
            == "Finishing prior display cleanup…"
        )

        allow_retirement = True
        page._drain_executor()
        assert page._admission is None
    finally:
        _dispose(page, qapp)
