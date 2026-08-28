from __future__ import annotations

import builtins
from dataclasses import replace
import time

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.tabs.scattering.display_values import DisplayFrameKey
from xdart.gui.tabs.scattering.shell_values import (
    ArtifactProgress,
    ProgressProjection,
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
        current = state.navigation.current
        assert current is not None
        artifact_frames = tuple(
            frame
            for frame in state.navigation.frames
            if frame.artifact == current.artifact
        )
        assert shell.browser.frame_model.rowCount() == frame_count
        assert all(
            shell.browser.frame_model.index(index, 0).data(
                QtCore.Qt.ItemDataRole.UserRole
            ) is state.navigation.frames[index]
            for index in range(frame_count)
        )
        assert shell.scientific.frame_selector.count() == len(artifact_frames)
        assert len(shell.scientific._heavy_available) == 2
        assert all(
            shell.scientific.frame_selector.itemData(index)
            is artifact_frames[index]
            for index in range(len(artifact_frames))
        )
        shell.scientific.frame_selector.setCurrentIndex(len(artifact_frames) - 1)
        assert commands[-1].frame is artifact_frames[-1]
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
            frames[0].source_scan,
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
        assert shell.scientific.frame_selector.itemData(
            5,
            QtCore.Qt.ItemDataRole.ToolTipRole,
        ) == f"{appended.source_scan}:{appended.local_frame_label}"
    finally:
        _dispose(shell, qapp)


def test_e3_ui3_terminal_footer_follows_current_artifact_selection(
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
        assert shell.scientific.progress.text() == "1/25"
        assert shell.scientific.status.text() == "Ready"
    finally:
        _dispose(shell, qapp)


def test_e3_ui3_footer_uses_typed_current_artifact_progress(
    qapp: QtWidgets.QApplication,
) -> None:
    shell = ScatteringWorkspaceShell()
    state = make_shell_projection(frame_count=52, selected_index=51)
    current = state.navigation.current
    assert current is not None
    state = replace(
        state,
        progress=ProgressProjection(
            52,
            2961,
            "Running",
            (ArtifactProgress(current.artifact, 52, 1000, published=52),),
        ),
    )
    try:
        shell.apply_state(state)
        # Two logical source scans deliberately share this retained artifact.
        # The scientific footer and typed progress therefore follow their
        # shared artifact as one 52/1000 catalog.
        assert shell.scientific.progress.text() == "52/1000"
        assert shell.scientific.frame_selector.count() == 52
        assert all(
            shell.scientific.frame_selector.itemData(index).artifact
            == current.artifact
            for index in range(shell.scientific.frame_selector.count())
        )
    finally:
        _dispose(shell, qapp)


@pytest.mark.parametrize("label_start", [0, 1])
def test_e3_ui3_footer_keeps_published_prefix_position_when_durable_ahead(
    qapp: QtWidgets.QApplication,
    label_start: int,
) -> None:
    shell = ScatteringWorkspaceShell()
    state = make_shell_projection(
        frame_count=10,
        selected_index=0,
        heavy_indices=(),
        plot_mode="Single",
    )
    frames = tuple(
        replace(
            frame,
            source_scan="scan-a",
            local_frame_label=label_start + index,
        )
        for index, frame in enumerate(state.navigation.frames)
    )
    artifact = frames[0].artifact
    state = replace(
        state,
        browser=replace(state.browser, frames=frames),
        scientific=replace(
            state.scientific,
            traces=tuple(
                replace(trace, frame=frames[index])
                for index, trace in enumerate(state.scientific.traces)
            ),
            heavy_available=frozenset(),
            heavy=None,
        ),
        navigation=replace(
            state.navigation,
            frames=frames,
            current=frames[0],
            selected=(frames[0],),
        ),
        progress=ProgressProjection(
            1000,
            1000,
            "Display projection failed",
            (ArtifactProgress(artifact, 1000, 1000, published=10),),
            terminal=True,
        ),
    )
    try:
        shell.apply_state(state)
        assert shell.scientific.progress.text() == "1/1000"

        shell.apply_state(replace(
            state,
            revision=2,
            navigation=replace(
                state.navigation,
                current=frames[-1],
                selected=(frames[-1],),
            ),
        ))
        assert shell.scientific.progress.text() == "10/1000"

        retained = frames[2:]
        shell.apply_state(replace(
            state,
            revision=3,
            navigation=replace(
                state.navigation,
                frames=retained,
                current=retained[0],
                selected=(retained[0],),
            ),
        ))
        assert shell.scientific.progress.text() == "3/1000"
    finally:
        _dispose(shell, qapp)


def test_e3_ui3_footer_keeps_absolute_scan_position_after_catalog_eviction(
    qapp: QtWidgets.QApplication,
) -> None:
    shell = ScatteringWorkspaceShell()
    state = make_shell_projection(
        frame_count=5,
        selected_index=2,
        plot_mode="Single",
    )
    retained = state.navigation.frames[2:]
    artifact = retained[0].artifact
    progress = ProgressProjection(
        5,
        5,
        "Finished",
        (ArtifactProgress(artifact, 5, 5, published=5),),
    )
    try:
        shell.apply_state(replace(
            state,
            navigation=replace(
                state.navigation,
                frames=retained,
                current=retained[0],
                selected=(retained[0],),
            ),
            progress=progress,
        ))
        assert shell.scientific.frame_selector.count() == 3
        assert shell.scientific.progress.text() == "3/5"

        shell.apply_state(replace(
            state,
            revision=2,
            navigation=replace(
                state.navigation,
                frames=retained,
                current=retained[-1],
                selected=(retained[-1],),
            ),
            progress=progress,
        ))
        assert shell.scientific.progress.text() == "5/5"
    finally:
        _dispose(shell, qapp)


@pytest.mark.parametrize(
    ("phase", "button_text", "run_enabled", "stop_enabled"),
    [
        (ShellPhase.IDLE, "Run", True, False),
        # The pending Run has already been consumed.  PREPARING previews the
        # eventual Pause affordance but keeps it disabled until acceptance.
        (ShellPhase.PREPARING, "Pause", False, True),
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
        assert shell.scientific.progress.text() == "1/5"
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


def test_shell_light_refresh_skips_only_scientific_reconcile(
        qapp: QtWidgets.QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    shell = ScatteringWorkspaceShell()
    state = make_shell_projection(frame_count=2, heavy_indices=(0, 1))
    browser_calls: list[int] = []
    scientific_calls: list[int] = []
    monkeypatch.setattr(
        shell.browser, "reconcile",
        lambda *_args, **_kwargs: browser_calls.append(1),
    )
    monkeypatch.setattr(
        shell.scientific, "reconcile",
        lambda *_args, **_kwargs: scientific_calls.append(1),
    )
    try:
        shell.apply_state(state, preserve_scientific=True)
        assert browser_calls == [1]
        assert scientific_calls == []
        assert shell._revision == state.revision

        shell.apply_state(replace(state, revision=state.revision + 1))
        assert browser_calls == [1, 1]
        assert scientific_calls == [1]

        shell.apply_state(
            replace(state, revision=state.revision + 2),
            preserve_display=True,
            preserve_scientific=True,
        )
        assert browser_calls == [1, 1]
        assert scientific_calls == [1]
    finally:
        _dispose(shell, qapp)
