from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtGui, QtTest, QtWidgets

from xdart.modules.display_context import ContextKind
from xdart.gui.tabs.scattering.display_values import StandardDisplayPayload
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.shell_projection import (
    ScientificPreferences,
    build_scientific_projection,
)
from xdart.gui.tabs.scattering.shell_values import (
    FrameNavigationProjection,
    ShellCommand,
    ShellCommandKind,
    SlicePin,
)
from xdart.gui.tabs.scattering.workspace_shell import (
    ScatteringWorkspaceShell,
)
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xrd_tools.core import Axis, FrameView, TwoDKind

from tests.xdart.scattering.e3_shell_support import make_shell_projection


@pytest.fixture
def qapp() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _selected_browser_rows(
    shell: ScatteringWorkspaceShell,
) -> tuple[int, ...]:
    return tuple(
        sorted(
            index.row()
            for index in shell.browser.frames.selectionModel().selectedRows()
        )
    )


def _click_frame(
    qapp: QtWidgets.QApplication,
    shell: ScatteringWorkspaceShell,
    row: int,
    *,
    modifiers: QtCore.Qt.KeyboardModifier = (
        QtCore.Qt.KeyboardModifier.NoModifier
    ),
) -> None:
    index = shell.browser.frame_model.index(row, 0)
    shell.browser.frames.scrollTo(index)
    qapp.processEvents()
    QtTest.QTest.mouseClick(
        shell.browser.frames.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        modifiers,
        shell.browser.frames.visualRect(index).center(),
    )
    qapp.processEvents()
    # Leave scheduler headroom beyond the production 100 ms quiet window;
    # loaded full packets can otherwise sample the timer before delivery.
    QtTest.QTest.qWait(200)


def _set_current_row(
    shell: ScatteringWorkspaceShell,
    row: int,
    flags: QtCore.QItemSelectionModel.SelectionFlag,
) -> None:
    shell.browser.frames.selectionModel().setCurrentIndex(
        shell.browser.frame_model.index(row, 0),
        flags | QtCore.QItemSelectionModel.SelectionFlag.Rows,
    )


def _send_key(
    qapp: QtWidgets.QApplication,
    shell: ScatteringWorkspaceShell,
    event_type: QtCore.QEvent.Type,
    key: QtCore.Qt.Key,
    *,
    modifiers: QtCore.Qt.KeyboardModifier = (
        QtCore.Qt.KeyboardModifier.NoModifier
    ),
    auto_repeat: bool = False,
) -> None:
    event = QtGui.QKeyEvent(
        event_type,
        key,
        modifiers,
        "",
        auto_repeat,
        1,
    )
    qapp.sendEvent(shell.browser.frames, event)
    qapp.processEvents()


@pytest.mark.parametrize(
    "plot_mode",
    ("Overlay", "Waterfall"),
)
def test_accumulating_click_carries_exact_toggle_membership(
    qapp: QtWidgets.QApplication,
    plot_mode: str,
) -> None:
    state = make_shell_projection(plot_mode=plot_mode)
    shell = ScatteringWorkspaceShell()
    commands = []
    shell.commandRequested.connect(commands.append)
    shell.resize(1200, 800)
    shell.show()
    try:
        shell.apply_state(state)
        qapp.processEvents()
        assert _selected_browser_rows(shell) == (0,)

        commands.clear()
        _click_frame(qapp, shell, 2)

        assert _selected_browser_rows(shell) == (0, 2)
        assert len(commands) == 1
        assert commands[0].kind is ShellCommandKind.SELECT_BROWSER_FRAMES
        assert commands[0].frames == (
            state.navigation.frames[0],
            state.navigation.frames[2],
        )
        assert commands[0].frame is state.navigation.frames[2]

        commands.clear()
        _click_frame(qapp, shell, 2)

        assert _selected_browser_rows(shell) == (0,)
        assert len(commands) == 1
        assert commands[0].frames == (state.navigation.frames[0],)
        assert commands[0].frame is state.navigation.frames[2]
        assert (
            shell.browser.frames.currentIndex().data(
                QtCore.Qt.ItemDataRole.UserRole
            )
            is state.navigation.frames[2]
        )
    finally:
        shell.close()


@pytest.mark.parametrize(
    "plot_mode",
    ("Overlay", "Waterfall"),
)
def test_idle_accumulating_selection_carries_exact_highlighted_rows(
    qapp: QtWidgets.QApplication,
    plot_mode: str,
) -> None:
    """Historical Browse selection is operator-owned once the run is idle."""

    state = make_shell_projection(plot_mode=plot_mode)
    first = state.navigation.frames[0]
    state = replace(
        state,
        navigation=replace(
            state.navigation,
            current=first,
            selected=(first,),
        ),
    )
    shell = ScatteringWorkspaceShell()
    commands = []
    shell.commandRequested.connect(commands.append)
    shell.resize(1200, 800)
    shell.show()
    try:
        shell.apply_state(state)
        qapp.processEvents()
        commands.clear()

        _click_frame(qapp, shell, 2)

        assert _selected_browser_rows(shell) == (0, 2)
        assert len(commands) == 1
        assert commands[0].kind is ShellCommandKind.SELECT_BROWSER_FRAMES
        assert commands[0].frame is state.navigation.frames[2]
        assert commands[0].frames == (
            state.navigation.frames[0],
            state.navigation.frames[2],
        )
    finally:
        shell.close()


@pytest.mark.parametrize("plot_mode", ("Overlay", "Waterfall"))
def test_accumulating_sole_row_toggle_preserves_visible_membership(
    qapp: QtWidgets.QApplication,
    plot_mode: str,
) -> None:
    state = make_shell_projection(
        frame_count=1,
        heavy_indices=(0,),
        plot_mode=plot_mode,
    )
    shell = ScatteringWorkspaceShell()
    commands = []
    shell.commandRequested.connect(commands.append)
    shell.resize(1200, 800)
    shell.show()
    try:
        shell.apply_state(state)
        qapp.processEvents()
        commands.clear()

        _click_frame(qapp, shell, 0)

        assert len(commands) == 1
        assert commands[0].kind is ShellCommandKind.SELECT_BROWSER_FRAMES
        assert commands[0].frame is state.navigation.frames[0]
        assert commands[0].frames == (state.navigation.frames[0],)
    finally:
        shell.close()


def test_plain_click_in_single_mode_still_replaces_the_exact_row(
    qapp: QtWidgets.QApplication,
) -> None:
    state = make_shell_projection(plot_mode="Single")
    shell = ScatteringWorkspaceShell()
    commands = []
    shell.commandRequested.connect(commands.append)
    shell.resize(1200, 800)
    shell.show()
    try:
        shell.apply_state(state)
        qapp.processEvents()
        assert _selected_browser_rows(shell) == (0,)

        commands.clear()
        _click_frame(qapp, shell, 3)

        assert _selected_browser_rows(shell) == (3,)
        assert len(commands) == 1
        assert commands[0].kind is ShellCommandKind.SELECT_BROWSER_FRAMES
        assert commands[0].frames == (state.navigation.frames[3],)
        assert commands[0].frame is state.navigation.frames[3]
    finally:
        shell.close()


@pytest.mark.parametrize(
    "modifier",
    (
        QtCore.Qt.KeyboardModifier.ControlModifier,
        QtCore.Qt.KeyboardModifier.MetaModifier,
    ),
)
def test_modified_single_click_toggles_without_collapsing_membership(
    qapp: QtWidgets.QApplication,
    modifier: QtCore.Qt.KeyboardModifier,
) -> None:
    state = make_shell_projection(plot_mode="Single")
    shell = ScatteringWorkspaceShell()
    commands = []
    shell.commandRequested.connect(commands.append)
    shell.resize(1200, 800)
    shell.show()
    try:
        shell.apply_state(state)
        qapp.processEvents()
        commands.clear()

        _click_frame(qapp, shell, 2, modifiers=modifier)

        assert _selected_browser_rows(shell) == (0, 2)
        assert len(commands) == 1
        assert commands[0].frames == (
            state.navigation.frames[0],
            state.navigation.frames[2],
        )
        assert commands[0].frame is state.navigation.frames[2]

        commands.clear()
        _click_frame(qapp, shell, 0, modifiers=modifier)
        assert _selected_browser_rows(shell) == (2,)
        assert len(commands) == 1
        assert commands[0].frames == (state.navigation.frames[2],)
        assert commands[0].frame is state.navigation.frames[2]
    finally:
        shell.close()


def test_rapid_single_selection_updates_highlight_continuously_but_emits_once(
    qapp: QtWidgets.QApplication,
) -> None:
    state = make_shell_projection(plot_mode="Single")
    shell = ScatteringWorkspaceShell()
    commands = []
    shell.commandRequested.connect(commands.append)
    shell.show()
    try:
        shell.apply_state(state)
        qapp.processEvents()
        commands.clear()

        for row in (1, 2, 3, 4):
            _set_current_row(
                shell,
                row,
                QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect,
            )
            qapp.processEvents()
            assert _selected_browser_rows(shell) == (row,)
            assert commands == []

        QtTest.QTest.qWait(130)
        assert len(commands) == 1
        assert commands[0].kind is ShellCommandKind.SELECT_BROWSER_FRAMES
        assert commands[0].frame is state.navigation.frames[4]
        assert commands[0].frames == (state.navigation.frames[4],)
    finally:
        shell.close()


def test_three_hundred_selection_changes_parse_and_emit_only_once(
    qapp: QtWidgets.QApplication,
) -> None:
    state = make_shell_projection(plot_mode="Single")
    shell = ScatteringWorkspaceShell()
    commands = []
    shell.commandRequested.connect(commands.append)
    shell.show()
    try:
        shell.apply_state(state)
        qapp.processEvents()
        commands.clear()

        for offset in range(300):
            row = offset % 4 + 1
            _set_current_row(
                shell,
                row,
                QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect,
            )
        assert _selected_browser_rows(shell) == (4,)
        assert commands == []

        QtTest.QTest.qWait(130)
        assert len(commands) == 1
        assert commands[0].frame is state.navigation.frames[4]
        assert commands[0].frames == (state.navigation.frames[4],)
    finally:
        shell.close()


def test_held_arrow_is_one_gesture_with_no_intermediate_command(
    qapp: QtWidgets.QApplication,
) -> None:
    state = make_shell_projection(plot_mode="Single")
    shell = ScatteringWorkspaceShell()
    commands = []
    shell.commandRequested.connect(commands.append)
    shell.show()
    try:
        shell.apply_state(state)
        shell.browser.frames.setFocus()
        qapp.processEvents()
        commands.clear()

        _send_key(
            qapp,
            shell,
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_Down,
        )
        for _ in range(2):
            _send_key(
                qapp,
                shell,
                QtCore.QEvent.Type.KeyRelease,
                QtCore.Qt.Key.Key_Down,
                auto_repeat=True,
            )
            _send_key(
                qapp,
                shell,
                QtCore.QEvent.Type.KeyPress,
                QtCore.Qt.Key.Key_Down,
                auto_repeat=True,
            )
        assert commands == []

        _send_key(
            qapp,
            shell,
            QtCore.QEvent.Type.KeyRelease,
            QtCore.Qt.Key.Key_Down,
        )
        assert commands == []
        QtTest.QTest.qWait(130)

        assert len(commands) == 1
        assert commands[0].frame is state.navigation.frames[3]
        assert commands[0].frames == (state.navigation.frames[3],)
    finally:
        shell.close()


def test_held_overlay_arrow_carries_exact_new_ui_membership(
    qapp: QtWidgets.QApplication,
) -> None:
    base = make_shell_projection(plot_mode="Overlay")
    initial = base.navigation.frames[0]
    state = replace(
        base,
        navigation=FrameNavigationProjection(
            base.navigation.frames,
            initial,
            (initial,),
        ),
    )
    shell = ScatteringWorkspaceShell()
    commands = []
    shell.commandRequested.connect(commands.append)
    shell.show()
    try:
        shell.apply_state(state)
        shell.browser.frames.setFocus()
        qapp.processEvents()
        commands.clear()

        _send_key(
            qapp,
            shell,
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_Down,
        )
        for _ in range(2):
            _send_key(
                qapp,
                shell,
                QtCore.QEvent.Type.KeyRelease,
                QtCore.Qt.Key.Key_Down,
                auto_repeat=True,
            )
            _send_key(
                qapp,
                shell,
                QtCore.QEvent.Type.KeyPress,
                QtCore.Qt.Key.Key_Down,
                auto_repeat=True,
            )
        _send_key(
            qapp,
            shell,
            QtCore.QEvent.Type.KeyRelease,
            QtCore.Qt.Key.Key_Down,
        )
        assert commands == []
        QtTest.QTest.qWait(130)

        assert len(commands) == 1
        assert commands[0].frame is state.navigation.frames[3]
        assert commands[0].frames == (state.navigation.frames[3],)
    finally:
        shell.close()


def test_live_append_rebases_pending_overlay_anchor_on_new_membership(
    qapp: QtWidgets.QApplication,
) -> None:
    base = make_shell_projection(plot_mode="Overlay")
    frames = base.navigation.frames
    state = replace(
        base,
        navigation=FrameNavigationProjection(
            frames,
            frames[0],
            frames[:2],
        ),
    )
    shell = ScatteringWorkspaceShell()
    commands = []
    shell.commandRequested.connect(commands.append)
    shell.show()
    try:
        shell.apply_state(state)
        qapp.processEvents()
        commands.clear()

        _set_current_row(
            shell,
            2,
            QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect,
        )
        qapp.processEvents()
        assert shell.browser.frame_selection_pending
        assert commands == []

        appended = type(frames[0])(
            frames[0].run_identity,
            "scan-c",
            frames[0].artifact,
            9,
            99,
        )
        appended_frames = (*frames, appended)
        latest_membership = (*frames[:2], appended)
        shell.apply_state(
            replace(
                state,
                revision=state.revision + 1,
                browser=replace(
                    state.browser,
                    frames=appended_frames,
                ),
                navigation=FrameNavigationProjection(
                    appended_frames,
                    appended,
                    latest_membership,
                ),
            )
        )
        qapp.processEvents()
        assert shell.browser.frame_selection_pending
        assert commands == []

        QtTest.QTest.qWait(130)
        assert len(commands) == 1
        assert commands[0].frame is frames[2]
        assert commands[0].frames == latest_membership
    finally:
        shell.close()


def test_overlay_footer_moves_anchor_without_mutating_membership(
    qapp: QtWidgets.QApplication,
) -> None:
    base = make_shell_projection(
        frame_count=5,
        selected_index=0,
        heavy_indices=(0, 2),
        plot_mode="Overlay",
        source_scan="scan-a",
    )
    frames = base.navigation.frames
    state = replace(
        base,
        navigation=FrameNavigationProjection(
            frames,
            frames[0],
            frames[:2],
        ),
    )
    shell = ScatteringWorkspaceShell()
    commands = []
    shell.commandRequested.connect(commands.append)
    shell.show()
    try:
        shell.apply_state(state)
        qapp.processEvents()
        commands.clear()

        shell.scientific.frame_selector.setCurrentIndex(2)

        assert len(commands) == 1
        assert commands[0].kind is ShellCommandKind.SELECT_FRAME
        assert commands[0].frame is frames[2]
        assert commands[0].frames == state.navigation.selected
    finally:
        shell.close()


def test_single_shift_range_survives_as_one_multi_frame_command(
    qapp: QtWidgets.QApplication,
) -> None:
    state = make_shell_projection(plot_mode="Single")
    shell = ScatteringWorkspaceShell()
    commands = []
    shell.commandRequested.connect(commands.append)
    shell.show()
    try:
        shell.apply_state(state)
        shell.browser.frames.setFocus()
        qapp.processEvents()
        commands.clear()

        _send_key(
            qapp,
            shell,
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_Shift,
            modifiers=QtCore.Qt.KeyboardModifier.ShiftModifier,
        )
        for _ in range(3):
            _send_key(
                qapp,
                shell,
                QtCore.QEvent.Type.KeyPress,
                QtCore.Qt.Key.Key_Down,
                modifiers=QtCore.Qt.KeyboardModifier.ShiftModifier,
            )
            _send_key(
                qapp,
                shell,
                QtCore.QEvent.Type.KeyRelease,
                QtCore.Qt.Key.Key_Down,
                modifiers=QtCore.Qt.KeyboardModifier.ShiftModifier,
                auto_repeat=True,
            )
        _send_key(
            qapp,
            shell,
            QtCore.QEvent.Type.KeyRelease,
            QtCore.Qt.Key.Key_Shift,
        )
        assert commands == []
        QtTest.QTest.qWait(130)

        assert _selected_browser_rows(shell) == (0, 1, 2, 3)
        assert len(commands) == 1
        assert commands[0].frames == state.navigation.frames[:4]
        assert commands[0].frame is state.navigation.frames[3]
    finally:
        shell.close()


def test_stale_shell_reconcile_does_not_erase_pending_highlight(
    qapp: QtWidgets.QApplication,
) -> None:
    state = make_shell_projection(plot_mode="Single")
    shell = ScatteringWorkspaceShell()
    commands = []
    shell.commandRequested.connect(commands.append)
    shell.show()
    try:
        shell.apply_state(state)
        qapp.processEvents()
        commands.clear()

        _set_current_row(
            shell,
            3,
            QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect,
        )
        qapp.processEvents()
        assert _selected_browser_rows(shell) == (3,)
        assert commands == []

        stale_completion = replace(
            state,
            revision=state.revision + 1,
            scientific=replace(
                state.scientific,
                title="stale pre-gesture completion",
            ),
        )
        shell.apply_state(stale_completion)
        qapp.processEvents()
        assert _selected_browser_rows(shell) == (3,)
        assert (
            shell.scientific.title.text()
            != "stale pre-gesture completion"
        )
        assert commands == []

        QtTest.QTest.qWait(130)
        assert len(commands) == 1
        assert commands[0].frames == (state.navigation.frames[3],)
    finally:
        shell.close()


def test_live_append_during_held_key_keeps_newer_browser_intent(
    qapp: QtWidgets.QApplication,
) -> None:
    state = make_shell_projection(plot_mode="Single")
    frames = state.navigation.frames
    shell = ScatteringWorkspaceShell()
    commands = []
    shell.commandRequested.connect(commands.append)
    shell.show()
    try:
        shell.apply_state(state)
        shell.browser.frames.setFocus()
        qapp.processEvents()
        commands.clear()

        _send_key(
            qapp,
            shell,
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_Down,
        )
        assert shell.browser.frame_selection_pending
        assert _selected_browser_rows(shell) == (1,)

        appended = type(frames[0])(
            frames[0].run_identity,
            "scan-c",
            frames[0].artifact,
            9,
            99,
        )
        appended_frames = (*frames, appended)
        live_navigation = FrameNavigationProjection(
            appended_frames,
            appended,
            (appended,),
        )
        shell.apply_state(
            replace(
                state,
                revision=state.revision + 1,
                browser=replace(
                    state.browser,
                    frames=appended_frames,
                ),
                navigation=live_navigation,
                scientific=replace(
                    state.scientific,
                    title="async append must not repaint",
                ),
            )
        )
        qapp.processEvents()
        assert shell.browser.frame_selection_pending
        assert _selected_browser_rows(shell) == (1,)
        assert (
            shell.scientific.title.text()
            != "async append must not repaint"
        )
        assert commands == []

        _send_key(
            qapp,
            shell,
            QtCore.QEvent.Type.KeyRelease,
            QtCore.Qt.Key.Key_Down,
        )
        QtTest.QTest.qWait(130)
        assert len(commands) == 1
        assert commands[0].frame is frames[1]
        assert commands[0].frames == (frames[1],)
    finally:
        shell.close()


def test_catalog_replacement_cancels_stale_debounced_selection(
    qapp: QtWidgets.QApplication,
) -> None:
    state = make_shell_projection(plot_mode="Single")
    shell = ScatteringWorkspaceShell()
    commands = []
    shell.commandRequested.connect(commands.append)
    shell.show()
    try:
        shell.apply_state(state)
        qapp.processEvents()
        commands.clear()

        _set_current_row(
            shell,
            3,
            QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect,
        )
        qapp.processEvents()
        assert shell.browser.frame_selection_pending

        replacement = make_shell_projection(
            revision=state.revision + 1,
            plot_mode="Single",
        )
        shell.apply_state(replacement)
        qapp.processEvents()
        assert not shell.browser.frame_selection_pending
        QtTest.QTest.qWait(130)
        assert commands == []
        assert _selected_browser_rows(shell) == (0,)
    finally:
        shell.close()


@pytest.mark.parametrize("plot_mode", ("Single", "Overlay", "Waterfall"))
def test_production_parity_keeps_raw_and_cake_on_current_frame(
    plot_mode: str,
) -> None:
    base = make_shell_projection(
        frame_count=3,
        selected_index=1,
        heavy_indices=(0, 1, 2),
        plot_mode=plot_mode,
    )
    frames = base.navigation.frames
    current = frames[1]
    selected = frames
    navigation = FrameNavigationProjection(frames, current, selected)
    radial = np.linspace(0.1, 1.0, 4)
    azimuth = np.linspace(-10.0, 10.0, 3)
    payloads = tuple(
        StandardDisplayPayload(
            0,
            frame,
            f"frame {index}",
            FrameView(
                frame.local_frame_label,
                raw=np.full((2, 3), float(index + 1)),
                axis_1d=Axis("radial", "q_A^-1", values=radial),
                intensity_1d=np.full((4,), float(index + 10)),
                axis_2d_x=Axis("radial", "q_A^-1", values=radial),
                axis_2d_y=Axis("chi", "chi_deg", values=azimuth),
                intensity_2d=np.full((3, 4), float(index + 20)),
                two_d_kind=TwoDKind.Q_CHI,
            ),
        )
        for index, frame in enumerate(frames)
        if any(frame is item for item in selected)
    )

    projection = build_scientific_projection(
        payloads,
        navigation,
        frozenset(selected),
        ScientificPreferences(plot_mode=plot_mode),
        "",
    )

    assert projection.heavy is not None
    assert projection.heavy.frame is current
    np.testing.assert_array_equal(
        projection.heavy.raw,
        np.full((2, 3), 2.0),
    )
    np.testing.assert_array_equal(
        projection.heavy.cake,
        np.full((3, 4), 21.0),
    )
    assert tuple(trace.frame for trace in projection.traces) == selected


def test_current_heavy_anchor_may_live_outside_accumulated_trace_membership(
) -> None:
    base = make_shell_projection(
        frame_count=3,
        selected_index=0,
        heavy_indices=(0, 1, 2),
        plot_mode="Overlay",
    )
    frames = base.navigation.frames
    navigation = FrameNavigationProjection(
        frames,
        frames[2],
        frames[:2],
    )
    radial = np.linspace(0.1, 1.0, 4)
    azimuth = np.linspace(-10.0, 10.0, 3)
    payloads = tuple(
        StandardDisplayPayload(
            0,
            frame,
            f"frame {index}",
            FrameView(
                frame.local_frame_label,
                raw=np.full((2, 3), float(index + 1)),
                axis_1d=Axis("radial", "q_A^-1", values=radial),
                intensity_1d=np.full((4,), float(index + 10)),
                axis_2d_x=Axis("radial", "q_A^-1", values=radial),
                axis_2d_y=Axis("chi", "chi_deg", values=azimuth),
                intensity_2d=np.full((3, 4), float(index + 20)),
                two_d_kind=TwoDKind.Q_CHI,
            ),
        )
        for index, frame in enumerate(frames)
    )

    projection = build_scientific_projection(
        payloads,
        navigation,
        frozenset(frames),
        ScientificPreferences(plot_mode="Overlay"),
        "",
    )

    assert projection.heavy is not None
    assert projection.heavy.frame is frames[2]
    assert tuple(trace.frame for trace in projection.traces) == frames[:2]


def test_page_preserves_explicit_single_multiselection_but_mode_entry_collapses(
) -> None:
    state = make_shell_projection(plot_mode="Single")
    frames = state.navigation.frames[:3]
    events: list[str] = []
    calls: list[
        tuple[
            object,
            tuple[object, ...],
        ]
    ] = []

    class Controller:
        navigation = FrameNavigationProjection(
            frames,
            frames[-1],
            frames,
        )

        @staticmethod
        def owns_frame(frame: object) -> bool:
            return any(frame is candidate for candidate in frames)

        @staticmethod
        def select_navigation(
            current: object,
            selected: tuple[object, ...],
        ) -> bool:
            events.append("select")
            calls.append((current, selected))
            return True

    owner = SimpleNamespace(
        _shell=SimpleNamespace(
            browser=SimpleNamespace(
                cancel_pending_frame_selection=lambda: events.append(
                    "cancel"
                )
            )
        ),
        _context_controller=Controller(),
        _preferences=ScientificPreferences(plot_mode="Single"),
        _refresh_shell=lambda: None,
        _ensure_timer=lambda: None,
    )
    ScatteringWorkspace._select_frames(
        owner,
        ShellCommand(
            ShellCommandKind.SELECT_BROWSER_FRAMES,
            frame=frames[-1],
            frames=frames,
        ),
    )
    assert calls == [(frames[-1], frames)]
    assert events == ["select"]

    events.clear()
    owner._preferences = ScientificPreferences(plot_mode="Overlay")
    assert ScatteringWorkspace._edit_scientific_preference(
        owner,
        ShellCommand(ShellCommandKind.SET_PLOT_MODE, "Single"),
    )
    assert owner._preferences.plot_mode == "Single"
    assert calls == [
        (frames[-1], frames),
        (frames[-1], (frames[-1],)),
    ]
    assert events == ["cancel", "select"]


def test_show_all_page_dispatch_preserves_current_order_and_skips_empty_noop(
) -> None:
    frames = make_shell_projection().navigation.frames[:3]
    events: list[tuple[object, ...]] = []
    outcome = [True]

    def select_navigation(current, selected) -> bool:
        events.append(("select", current, selected))
        return outcome[0]

    controller = SimpleNamespace(
        navigation=FrameNavigationProjection(frames, frames[1], (frames[1],)),
        select_navigation=select_navigation,
    )
    owner = SimpleNamespace(
        _closing=False, _closed=False,
        _shell=SimpleNamespace(
            browser=SimpleNamespace(
                cancel_pending_frame_selection=lambda: events.append(("cancel",))
            )
        ),
        _context_controller=controller,
        _refresh_shell=lambda: events.append(("refresh",)),
        _edit_scientific_preference=lambda command: False,
    )
    command = ShellCommand(ShellCommandKind.SHOW_ALL)
    ScatteringWorkspace._handle_shell_command(owner, command)
    assert events == [("cancel",), ("select", frames[1], frames), ("refresh",)]

    events.clear()
    outcome[0] = False
    ScatteringWorkspace._handle_shell_command(owner, command)
    assert events == [("cancel",), ("select", frames[1], frames)]

    events.clear()
    controller.navigation = FrameNavigationProjection((), None, ())
    ScatteringWorkspace._handle_shell_command(owner, command)
    assert events == [("cancel",)]


def test_active_acquisition_overlay_visit_preserves_arrival_membership() -> None:
    state = make_shell_projection(plot_mode="Overlay")
    frames = state.navigation.frames
    arrival_membership = frames[:2]
    calls: list[tuple[object, tuple[object, ...]]] = []

    class Controller:
        navigation = FrameNavigationProjection(
            frames,
            frames[1],
            arrival_membership,
        )
        selection = SimpleNamespace(kind=ContextKind.ACQUISITION)

        @staticmethod
        def owns_frame(frame: object) -> bool:
            return any(frame is candidate for candidate in frames)

        @staticmethod
        def select_navigation(current, selected) -> bool:
            calls.append((current, selected))
            return True

    owner = SimpleNamespace(
        _context_controller=Controller(),
        _preferences=ScientificPreferences(plot_mode="Overlay"),
        _lifecycle=SimpleNamespace(phase=RunPhase.RUNNING),
        _auto_last=True,
        _refresh_shell=lambda: None,
        _ensure_timer=lambda: None,
    )
    ScatteringWorkspace._select_frames(
        owner,
        ShellCommand(
            ShellCommandKind.SELECT_BROWSER_FRAMES,
            frame=frames[3],
            frames=(frames[0], frames[3]),
        ),
    )

    assert calls == [(frames[3], arrival_membership)]


@pytest.mark.parametrize(
    ("selection_kind", "phase"),
    (
        (ContextKind.ACQUISITION, RunPhase.IDLE),
        (ContextKind.BROWSE, RunPhase.RUNNING),
        (ContextKind.BROWSE, RunPhase.PAUSED),
    ),
)
def test_nonactive_or_browse_overlay_adopts_exact_ui_membership(
    selection_kind: ContextKind,
    phase: RunPhase,
) -> None:
    state = make_shell_projection(plot_mode="Overlay")
    frames = state.navigation.frames
    requested = (frames[0], frames[3])
    calls: list[tuple[object, tuple[object, ...]]] = []

    class Controller:
        navigation = FrameNavigationProjection(
            frames,
            frames[1],
            frames[:2],
        )
        selection = SimpleNamespace(kind=selection_kind)

        @staticmethod
        def owns_frame(frame: object) -> bool:
            return any(frame is candidate for candidate in frames)

        @staticmethod
        def select_navigation(current, selected) -> bool:
            calls.append((current, selected))
            return True

    owner = SimpleNamespace(
        _context_controller=Controller(),
        _preferences=ScientificPreferences(plot_mode="Overlay"),
        _lifecycle=SimpleNamespace(phase=phase),
        _auto_last=True,
        _refresh_shell=lambda: None,
        _ensure_timer=lambda: None,
    )
    ScatteringWorkspace._select_frames(
        owner,
        ShellCommand(
            ShellCommandKind.SELECT_BROWSER_FRAMES,
            frame=frames[3],
            frames=requested,
        ),
    )

    assert calls == [(frames[3], requested)]


def test_pin_slice_command_is_owned_instead_of_silently_dropped() -> None:
    state = make_shell_projection(
        frame_count=1,
        heavy_indices=(0,),
        plot_mode="Overlay",
    )
    scientific = replace(
        state.scientific,
        slice_enabled=True,
        slice_center=0.0,
        slice_width=5.0,
    )
    owner = SimpleNamespace(
        _preferences=ScientificPreferences(
            plot_mode="Overlay",
            slice_enabled=True,
            slice_center=0.0,
            slice_width=5.0,
        ),
        _last_scientific_projection=scientific,
        _context_controller=SimpleNamespace(
            navigation=state.navigation,
            owns_frame=lambda frame: frame is state.navigation.current,
        ),
    )

    assert ScatteringWorkspace._edit_scientific_preference(
        owner,
        ShellCommand(ShellCommandKind.PIN_SLICE),
    )
    assert len(owner._preferences.slice_pins) == 1
    assert not ScatteringWorkspace._edit_scientific_preference(
        owner,
        ShellCommand(ShellCommandKind.PIN_SLICE),
    )


def test_pin_slice_uses_only_the_singular_accepted_heavy_frame() -> None:
    state = make_shell_projection(
        frame_count=4,
        selected_index=3,
        heavy_indices=(3,),
        plot_mode="Overlay",
    )
    navigation = replace(
        state.navigation,
        selected=state.navigation.frames,
    )
    scientific = replace(
        state.scientific,
        slice_enabled=True,
        slice_center=0.5,
        slice_width=2.0,
    )
    assert scientific.heavy is not None
    assert scientific.heavy.frame is navigation.current
    owner = SimpleNamespace(
        _preferences=ScientificPreferences(
            plot_mode="Overlay",
            slice_enabled=True,
            slice_center=0.5,
            slice_width=2.0,
        ),
        _last_scientific_projection=scientific,
        _context_controller=SimpleNamespace(
            navigation=navigation,
            owns_frame=lambda frame: any(
                frame is candidate for candidate in navigation.frames
            ),
        ),
    )

    assert ScatteringWorkspace._edit_scientific_preference(
        owner,
        ShellCommand(ShellCommandKind.PIN_SLICE),
    )
    assert len(owner._preferences.slice_pins) == 1
    assert (
        owner._preferences.slice_pins[0].frame
        is scientific.heavy.frame
    )


def test_slice_pins_survive_accumulating_mode_switch_and_clear_together() -> None:
    state = make_shell_projection(
        frame_count=1,
        heavy_indices=(0,),
        plot_mode="Overlay",
    )
    frame = state.navigation.frames[0]
    pin = SlicePin(frame, "Q", 0.0, 5.0)
    selected: list[tuple[object, tuple[object, ...]]] = []
    owner = SimpleNamespace(
        _preferences=ScientificPreferences(
            plot_mode="Overlay",
            slice_enabled=True,
            slice_pins=(pin,),
        ),
        _shell=SimpleNamespace(
            browser=SimpleNamespace(
                cancel_pending_frame_selection=lambda: None,
            )
        ),
        _context_controller=SimpleNamespace(
            navigation=state.navigation,
            select_navigation=lambda current, frames: selected.append(
                (current, frames)
            ),
        ),
    )

    assert ScatteringWorkspace._edit_scientific_preference(
        owner,
        ShellCommand(ShellCommandKind.SET_PLOT_MODE, "Waterfall"),
    )
    assert owner._preferences.slice_pins == (pin,)
    assert ScatteringWorkspace._edit_scientific_preference(
        owner,
        ShellCommand(ShellCommandKind.CLEAR_1D),
    )
    assert owner._preferences.slice_pins == ()
    assert selected == [(frame, (frame,))]


def test_direct_plot_axis_change_retargets_compatible_pins_and_clears_others() -> None:
    state = make_shell_projection(frame_count=1, heavy_indices=(0,))
    pin = SlicePin(state.navigation.frames[0], "Q", 0.5, 2.0)
    owner = SimpleNamespace(
        _preferences=ScientificPreferences(
            plot_axis="Q",
            plot_mode="Overlay",
            slice_pins=(pin,),
        )
    )

    assert ScatteringWorkspace._edit_scientific_preference(
        owner,
        ShellCommand(ShellCommandKind.SET_PLOT_AXIS, "2theta"),
    )
    assert len(owner._preferences.slice_pins) == 1
    retargeted = owner._preferences.slice_pins[0]
    assert retargeted.frame is pin.frame
    assert retargeted.plot_axis == "2theta"
    assert (retargeted.center, retargeted.width) == (0.5, 2.0)

    assert ScatteringWorkspace._edit_scientific_preference(
        owner,
        ShellCommand(ShellCommandKind.SET_PLOT_AXIS, "chi"),
    )
    assert owner._preferences.slice_pins == ()


def test_shared_image_axis_changes_apply_the_same_slice_pin_policy() -> None:
    state = make_shell_projection(frame_count=1, heavy_indices=(0,))
    pin = SlicePin(state.navigation.frames[0], "Q", 0.5, 2.0)
    owner = SimpleNamespace(
        _preferences=ScientificPreferences(
            image_axis="Q-Chi",
            plot_axis="Q",
            plot_mode="Overlay",
            share_axis=True,
            slice_pins=(pin,),
        )
    )

    assert ScatteringWorkspace._edit_scientific_preference(
        owner,
        ShellCommand(ShellCommandKind.SET_IMAGE_AXIS, "2Th-Chi"),
    )
    assert owner._preferences.plot_axis == "2theta"
    assert len(owner._preferences.slice_pins) == 1
    assert owner._preferences.slice_pins[0].plot_axis == "2theta"

    assert ScatteringWorkspace._edit_scientific_preference(
        owner,
        ShellCommand(ShellCommandKind.SET_IMAGE_AXIS, "qip_qoop"),
    )
    assert owner._preferences.plot_axis == "q_ip"
    assert owner._preferences.slice_pins == ()


def test_enabling_share_axis_retargets_compatible_slice_pins() -> None:
    state = make_shell_projection(frame_count=1, heavy_indices=(0,))
    pin = SlicePin(state.navigation.frames[0], "Q", 0.5, 2.0)
    owner = SimpleNamespace(
        _preferences=ScientificPreferences(
            image_axis="2Th-Chi",
            plot_axis="Q",
            plot_mode="Overlay",
            slice_pins=(pin,),
        )
    )

    assert ScatteringWorkspace._edit_scientific_preference(
        owner,
        ShellCommand(ShellCommandKind.SET_SHARE_AXIS, True),
    )
    assert owner._preferences.share_axis
    assert owner._preferences.plot_axis == "2theta"
    assert len(owner._preferences.slice_pins) == 1
    assert owner._preferences.slice_pins[0].plot_axis == "2theta"


def test_explicit_footer_selection_cancels_older_browser_debounce() -> None:
    state = make_shell_projection(plot_mode="Single")
    frame = state.navigation.frames[2]
    events: list[str] = []
    owner = SimpleNamespace(
        _closing=False,
        _closed=False,
        _shell=SimpleNamespace(
            browser=SimpleNamespace(
                cancel_pending_frame_selection=lambda: events.append(
                    "cancel"
                )
            )
        ),
        _select_frames=lambda command: events.append(
            f"select:{command.kind.value}"
        ),
    )

    ScatteringWorkspace._handle_shell_command(
        owner,
        ShellCommand(
            ShellCommandKind.SELECT_FRAME,
            frame=frame,
            frames=(frame,),
        ),
    )

    assert events == ["cancel", "select:select_frame"]
