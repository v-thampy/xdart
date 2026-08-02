from __future__ import annotations

import builtins
from dataclasses import replace
import time

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.tabs.scattering.display_values import DisplayFrameKey
from xdart.gui.tabs.scattering.shell_values import (
    ProgressProjection,
    ShellCommandKind,
    ShellPhase,
    TraceProjection,
)
from xdart.gui.tabs.scattering.workspace_shell import (
    ScatteringWorkspaceShell,
)

from tests.xdart.scattering.e3_shell_support import make_shell_projection


@pytest.fixture
def qapp() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _dispose(
    shell: ScatteringWorkspaceShell,
    qapp: QtWidgets.QApplication,
) -> None:
    shell.close()
    shell.deleteLater()
    qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


@pytest.mark.parametrize("frame_count", [5, 25, 651])
def test_e3_ui3_navigation_is_full_history_not_heavy_window(
    qapp: QtWidgets.QApplication,
    frame_count: int,
) -> None:
    shell = ScatteringWorkspaceShell()
    state = make_shell_projection(
        frame_count=frame_count,
        heavy_indices=(0, frame_count - 1),
        plot_mode="Single",
    )
    commands = []
    shell.commandRequested.connect(commands.append)
    try:
        shell.apply_state(state)
        assert shell.scientific.frame_selector.count() == frame_count
        assert len(shell.scientific._heavy_available) == 2
        assert all(
            shell.scientific.frame_selector.itemData(index)
            is state.navigation.frames[index]
            for index in range(frame_count)
        )
        shell.scientific.frame_selector.setCurrentIndex(frame_count - 1)
        assert commands[-1].kind is ShellCommandKind.SELECT_FRAME
        assert commands[-1].frame is state.navigation.frames[-1]
    finally:
        _dispose(shell, qapp)


def test_e3_ui3_selector_append_is_constant_work_with_new_duplicates(
    qapp: QtWidgets.QApplication,
) -> None:
    shell = ScatteringWorkspaceShell()
    state = make_shell_projection(frame_count=5, plot_mode="Single")
    try:
        shell.apply_state(state)
        before = shell.scientific._selector_operations
        frames = state.navigation.frames
        appended = DisplayFrameKey(
            frames[0].run_identity,
            "scan-c",
            "result.nxs",
            frames[0].local_frame_label,
            6,
        )
        axis = state.scientific.traces[0].axis
        trace = TraceProjection(
            appended,
            axis,
            state.scientific.traces[0].intensity,
            "scan-c:1",
        )
        scientific = replace(
            state.scientific,
            traces=(*state.scientific.traces, trace),
        )
        navigation = replace(
            state.navigation,
            frames=(*frames, appended),
            current=appended,
            selected=(appended,),
        )
        shell.apply_state(
            replace(
                state,
                revision=2,
                scientific=scientific,
                navigation=navigation,
            )
        )

        assert shell.scientific.frame_selector.count() == 6
        assert shell.scientific._selector_operations - before == 1
        assert shell.scientific.frame_selector.itemData(5) is appended
    finally:
        _dispose(shell, qapp)


def test_e3_ui3_terminal_progress_does_not_follow_historical_selection(
    qapp: QtWidgets.QApplication,
) -> None:
    shell = ScatteringWorkspaceShell()
    state = make_shell_projection(frame_count=25, selected_index=0)
    state = replace(
        state,
        progress=ProgressProjection(25, 25, "Finished"),
    )
    try:
        shell.apply_state(state)
        assert shell.scientific.frame_selector.currentData() is (
            state.navigation.frames[0]
        )
        assert shell.scientific.progress.text() == "25/25"
        assert shell.scientific.status.text() == "Ready"
    finally:
        _dispose(shell, qapp)


@pytest.mark.parametrize(
    ("phase", "button_text", "run_enabled", "stop_enabled"),
    [
        (ShellPhase.IDLE, "Run", True, False),
        (ShellPhase.PREPARING, "Run", False, True),
        (ShellPhase.RUNNING, "Pause", False, True),
        (ShellPhase.PAUSED, "Resume", False, True),
        (ShellPhase.STOPPING, "Run", False, True),
        (ShellPhase.FAILED, "Run", True, False),
    ],
)
def test_e3_ui3_run_state_projection_is_visual_only(
    qapp: QtWidgets.QApplication,
    phase: ShellPhase,
    button_text: str,
    run_enabled: bool,
    stop_enabled: bool,
) -> None:
    shell = ScatteringWorkspaceShell()
    state = make_shell_projection(phase=phase)
    try:
        shell.apply_state(state)
        assert button_text in shell.run_controls.startButton.text()
        assert shell.run_controls.startButton.isEnabled() is run_enabled
        assert shell.run_controls.stopButton.isEnabled() is stop_enabled
    finally:
        _dispose(shell, qapp)


def test_e3_ui3_browsed_and_finished_presentations_keep_same_shell(
    qapp: QtWidgets.QApplication,
) -> None:
    shell = ScatteringWorkspaceShell()
    state = make_shell_projection()
    try:
        browsed = replace(
            state,
            scientific=replace(
                state.scientific,
                status="Browsing loaded result",
            ),
        )
        shell.apply_state(browsed)
        assert shell.scientific.status.text() == "Browsing loaded result"

        finished = replace(
            state,
            revision=2,
            scientific=replace(
                state.scientific,
                status="Finished",
            ),
            progress=ProgressProjection(5, 5, "Finished"),
        )
        shell.apply_state(finished)
        assert shell.scientific.status.text() == "Finished"
        assert shell.scientific.progress.text() == "5/5"
        assert shell.splitter.count() == 3
    finally:
        _dispose(shell, qapp)


def test_e3_ui3_view_callbacks_never_open_a_file(
    qapp: QtWidgets.QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shell = ScatteringWorkspaceShell()
    shell.apply_state(make_shell_projection())
    commands = []
    shell.commandRequested.connect(commands.append)

    def forbidden_open(*_args, **_kwargs):
        raise AssertionError("visual-shell callback opened a file")

    monkeypatch.setattr(builtins, "open", forbidden_open)
    try:
        shell.browser.refresh.click()
        shell.browser.show_all.click()
        shell.tools.findChildren(QtWidgets.QPushButton)[0].click()
        shell.scientific.background.click()
        shell.scientific.frame_selector.setCurrentIndex(1)
        assert len(commands) == 5
    finally:
        _dispose(shell, qapp)


def test_e3_ui3_651_reconciliation_preserves_gui_heartbeat(
    qapp: QtWidgets.QApplication,
) -> None:
    shell = ScatteringWorkspaceShell()
    state = make_shell_projection(
        frame_count=651,
        heavy_indices=(0, 650),
        plot_mode="Single",
    )
    ticks: list[float] = []
    timer = QtCore.QTimer(shell)
    timer.setInterval(10)
    timer.timeout.connect(lambda: ticks.append(time.perf_counter()))
    try:
        shell.apply_state(state)
        operations = shell.scientific._selector_operations
        timer.start()
        for revision, index in enumerate(range(1, 81), 2):
            scientific = replace(
                state.scientific,
                title=f"frame {index}",
            )
            current = state.navigation.frames[index]
            navigation = replace(
                state.navigation,
                current=current,
                selected=(current,),
            )
            shell.apply_state(
                replace(
                    state,
                    revision=revision,
                    scientific=scientific,
                    navigation=navigation,
                )
            )
            qapp.processEvents()
        timer.stop()

        gaps = np.diff(ticks)
        assert len(ticks) >= 5
        assert float(gaps.max(initial=0.0)) < 0.25
        assert shell.scientific._selector_operations == operations
        assert shell.scientific.frame_selector.count() == 651
    finally:
        timer.stop()
        _dispose(shell, qapp)
