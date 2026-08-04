from __future__ import annotations

from dataclasses import replace

import pytest
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets

from xdart.gui.tabs.scattering.display_values import DisplayFrameKey
from xdart.gui.tabs.scattering.shell_values import (
    FrameNavigationProjection,
    ShellCommandKind,
)
from xdart.gui.tabs.scattering.tools_view import ToolsView
from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell

from tests.xdart.scattering.e3_shell_support import make_shell_projection


@pytest.fixture
def qapp() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _dispose(
    widget: QtWidgets.QWidget,
    qapp: QtWidgets.QApplication,
) -> None:
    widget.close()


def _browser_selected(
    shell: ScatteringWorkspaceShell,
) -> tuple[DisplayFrameKey, ...]:
    return tuple(
        index.data(QtCore.Qt.ItemDataRole.UserRole)
        for index in shell.browser.frames.selectionModel().selectedRows()
    )


def _row_of(
    shell: ScatteringWorkspaceShell,
    frame: DisplayFrameKey,
) -> int | None:
    """Locate a catalog row the way production does: by object identity."""

    model = shell.browser.frames.model()
    return next(
        (
            row
            for row in range(model.rowCount())
            if model.index(row, 0).data(QtCore.Qt.ItemDataRole.UserRole)
            is frame
        ),
        None,
    )


def test_e3_ui61_user_clear_emits_empty_membership_then_projects_total(
    qapp: QtWidgets.QApplication,
) -> None:
    state = make_shell_projection()
    shell = ScatteringWorkspaceShell()
    commands = []
    browser_commands = []
    shell.commandRequested.connect(commands.append)
    shell.browser.commandRequested.connect(browser_commands.append)
    try:
        shell.apply_state(state)
        # A fresh Overlay catalog collapses membership to the current frame.
        assert len(_browser_selected(shell)) == 1
        assert _browser_selected(shell)[0] is state.navigation.current

        shell.browser.frames.selectionModel().clearSelection()
        QtTest.QTest.qWait(130)
        assert len(commands) == 1
        assert commands[0].kind is (
            ShellCommandKind.SELECT_BROWSER_FRAMES
        )
        # Accepted split: a user clear reports the still-current frame and
        # the committed membership.  The browser no longer synthesises an
        # empty membership — the page owns accumulation and projects the
        # empty total below.
        assert commands[0].frame is state.navigation.current
        assert all(
            actual is expected
            for actual, expected in zip(
                commands[0].frames,
                state.navigation.selected,
                strict=True,
            )
        )
        assert browser_commands == commands

        empty = replace(
            state.navigation,
            current=None,
            selected=(),
        )
        commands.clear()
        shell.apply_state(
            replace(state, revision=2, navigation=empty)
        )
        assert commands == []
        assert _browser_selected(shell) == ()
        assert not shell.browser.frames.currentIndex().isValid()
        assert shell.scientific.frame_selector.count() == 0
        assert shell.scientific.frame_selector.currentIndex() == -1

        shell.apply_state(
            replace(state, revision=3, navigation=empty)
        )
        shell.browser.frames.selectionModel().clearSelection()
        assert commands == []

        shell.apply_state(replace(state, revision=4))
        assert len(_browser_selected(shell)) == 1
        assert _browser_selected(shell)[0] is state.navigation.current
        commands.clear()
        browser_commands.clear()
        shell.apply_state(
            replace(state, revision=5, navigation=empty)
        )
        # An empty projection retracts focus but not an accumulated visit:
        # membership is retired by the user or by a new catalog, never by a
        # projection alone.
        assert _browser_selected(shell)[0] is state.navigation.current
        assert not shell.browser.frames.currentIndex().isValid()
        assert shell.scientific.frame_selector.count() == 0
        assert commands == []
        assert browser_commands == []
    finally:
        _dispose(shell, qapp)


def test_e3_ui61_browser_sort_control_is_named_time(
    qapp: QtWidgets.QApplication,
) -> None:
    shell = ScatteringWorkspaceShell()
    try:
        assert shell.browser.date_sort.text() == "Time"
    finally:
        _dispose(shell, qapp)


def test_e3_ui61_empty_single_catalog_has_no_implicit_current(
    qapp: QtWidgets.QApplication,
) -> None:
    state = make_shell_projection(plot_mode="Single")
    shell = ScatteringWorkspaceShell()
    commands = []
    shell.commandRequested.connect(commands.append)
    try:
        shell.apply_state(state)
        empty = FrameNavigationProjection(
            state.navigation.frames,
            None,
            (),
        )
        scientific = replace(
            state.scientific,
            title="No current frame",
            heavy=None,
        )
        shell.apply_state(
            replace(
                state,
                revision=2,
                navigation=empty,
                scientific=scientific,
            )
        )

        assert commands == []
        assert _browser_selected(shell) == ()
        assert not shell.browser.frames.currentIndex().isValid()
        assert shell.scientific.frame_selector.count() == 5
        assert shell.scientific.frame_selector.currentIndex() == -1
        assert shell.scientific.frame_selector.currentData() is None
        assert shell.scientific.title.text() == "No current frame"
        assert shell.scientific.raw.image.image is None
        assert shell.scientific.cake.image.image is None
        assert shell.scientific.curve.listDataItems() == []
    finally:
        _dispose(shell, qapp)


def test_e3_ui61_equal_distinct_membership_and_current_move_by_identity(
    qapp: QtWidgets.QApplication,
) -> None:
    state = make_shell_projection()
    frame_a = state.navigation.frames[0]
    frame_b = DisplayFrameKey(
        frame_a.run_identity,
        frame_a.source_scan,
        frame_a.artifact,
        frame_a.local_frame_label,
        frame_a.work_ordinal,
    )
    assert frame_a == frame_b
    assert frame_a is not frame_b
    frames = (frame_a, frame_b)
    first = FrameNavigationProjection(frames, frame_a, (frame_a,))
    second = FrameNavigationProjection(frames, frame_b, (frame_b,))
    state = replace(
        state,
        browser=replace(state.browser, frames=frames),
    )
    shell = ScatteringWorkspaceShell()
    commands = []
    shell.commandRequested.connect(commands.append)
    try:
        shell.apply_state(
            replace(state, navigation=first)
        )
        assert _browser_selected(shell) == (frame_a,)
        assert _browser_selected(shell)[0] is frame_a

        shell.apply_state(
            replace(state, revision=2, navigation=second)
        )
        assert commands == []
        # Accepted split: focus moves to the equal-but-distinct key while
        # accumulated membership stays on the frame the user actually
        # visited.  Both resolve by object identity, never by value — the
        # two keys compare equal, so a value match would collapse them.
        assert _row_of(shell, frame_a) == 0
        assert _row_of(shell, frame_b) == 1
        assert _browser_selected(shell)[0] is frame_a
        assert shell.browser.frames.currentIndex().row() == _row_of(
            shell, frame_b
        )
        assert shell.browser.frames.currentIndex().data(
            QtCore.Qt.ItemDataRole.UserRole
        ) is frame_b
        assert shell.scientific.frame_selector.currentData() is frame_b
    finally:
        _dispose(shell, qapp)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/data/processed", "[/data/processed]"),
        ("", ""),
    ],
)
def test_e3_ui61_browser_path_brackets_and_unbracketed_tooltip(
    qapp: QtWidgets.QApplication,
    path: str,
    expected: str,
) -> None:
    state = make_shell_projection()
    state = replace(
        state,
        browser=replace(state.browser, directory=path),
    )
    shell = ScatteringWorkspaceShell()
    try:
        shell.apply_state(state)
        assert shell.browser.scans_label.text() == "Scans"
        assert shell.browser.directory_label.text() == expected
        assert shell.browser.directory_label.toolTip() == path
    finally:
        _dispose(shell, qapp)


def test_e3_ui61_long_path_elides_inside_brackets_with_exact_tooltip(
    qapp: QtWidgets.QApplication,
) -> None:
    path = (
        "/facility/beamline/very-long-experiment-name/"
        "sample-series/processed-results"
    )
    state = make_shell_projection()
    state = replace(
        state,
        browser=replace(state.browser, directory=path),
    )
    shell = ScatteringWorkspaceShell()
    try:
        shell.apply_state(state)
        shell.resize(1024, 900)
        shell.show()
        qapp.processEvents()
        shown = shell.browser.directory_label.text()
        assert shown.startswith("[")
        assert shown.endswith("]")
        assert "…" in shown[1:-1]
        assert len(shown[1:-1]) <= 30
        assert shell.browser.directory_label.toolTip() == path
    finally:
        _dispose(shell, qapp)


def test_e3_ui61_tools_have_symmetric_padding_and_added_tool_is_reachable(
    qapp: QtWidgets.QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ToolsView,
        "_TOOLS",
        tuple((f"Tool {index}", f"tool_{index}") for index in range(10)),
    )
    tools = ToolsView()
    tools.resize(255, 120)
    tools.show()
    try:
        qapp.processEvents()
        margins = tools.tool_content.layout().contentsMargins()
        assert margins.top() == margins.bottom()
        assert margins.top() >= 8

        buttons = tools.findChildren(QtWidgets.QPushButton)
        assert len(buttons) == 10
        scroll = tools.tool_scroll.verticalScrollBar()
        assert scroll.maximum() > 0
        scroll.setValue(scroll.maximum())
        qapp.processEvents()
        last_center = buttons[-1].mapTo(
            tools.tool_scroll.viewport(),
            buttons[-1].rect().center(),
        )
        assert tools.tool_scroll.viewport().rect().contains(last_center)
    finally:
        _dispose(tools, qapp)
