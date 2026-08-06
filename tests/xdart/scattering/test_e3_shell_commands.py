from __future__ import annotations

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets

from xdart.gui.tabs.scattering.shell_values import (
    ShellCommand,
    ShellCommandKind,
)
from xdart.gui.tabs.scattering.workspace_shell import (
    ScatteringWorkspaceShell,
)

from tests.xdart.scattering.e3_shell_support import make_shell_projection


@pytest.fixture
def qapp() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def shell(qapp: QtWidgets.QApplication):
    widget = ScatteringWorkspaceShell()
    widget.apply_state(make_shell_projection())
    yield widget
    widget.close()


def test_e3_ui2_user_actions_emit_one_exact_typed_command(
    shell: ScatteringWorkspaceShell,
) -> None:
    commands: list[ShellCommand] = []
    shell.commandRequested.connect(commands.append)

    shell.browser.refresh.click()
    assert [item.kind for item in commands] == [
        ShellCommandKind.REFRESH_BROWSER
    ]

    commands.clear()
    shell.scientific.plot_mode.setCurrentText("Waterfall")
    assert commands == [
        ShellCommand(ShellCommandKind.SET_PLOT_MODE, "Waterfall")
    ]

    commands.clear()
    shell.run_controls.startButton.click()
    assert commands == [ShellCommand(ShellCommandKind.RUN_ACTION)]

    commands.clear()
    shell.controls.fieldValueChanged.emit(
        ("Project", "project_folder"), "/new-project"
    )
    assert commands == [
        ShellCommand(
            ShellCommandKind.CONTROL_EDIT,
            "/new-project",
            ("Project", "project_folder"),
        )
    ]


def test_e3_ui2_browser_multi_selection_carries_exact_frame_keys_once(
    shell: ScatteringWorkspaceShell,
) -> None:
    commands: list[ShellCommand] = []
    shell.commandRequested.connect(commands.append)
    emitted = QtTest.QSignalSpy(shell.commandRequested)
    selection = shell.browser.frames.selectionModel()
    # A fresh Overlay catalog collapses accumulated membership to the current
    # frame; membership grows from user visits, not from the projection.
    assert len(selection.selectedRows()) == 1
    assert selection.selectedRows()[0].row() == 0
    catalog = tuple(
        shell.browser.frames.model().index(row, 0).data(
            QtCore.Qt.ItemDataRole.UserRole
        )
        for row in range(shell.browser.frames.model().rowCount())
    )

    chosen = QtCore.QItemSelection()
    chosen.select(
        shell.browser.frames.model().index(0, 0),
        shell.browser.frames.model().index(0, 0),
    )
    chosen.select(
        shell.browser.frames.model().index(1, 0),
        shell.browser.frames.model().index(1, 0),
    )
    commands.clear()
    selection.select(
        chosen,
        QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
        | QtCore.QItemSelectionModel.SelectionFlag.Rows,
    )
    # The contract is one 100 ms quiet-window command, not a hard wall-clock
    # deadline 30 ms later.  Under a loaded CI event loop the coarse Qt timer
    # can legally wake after 130 ms, so wait boundedly for the actual signal.
    assert emitted.wait(500)

    assert len(commands) == 1
    assert commands[0].kind is ShellCommandKind.SELECT_BROWSER_FRAMES
    # Overlay decouples focus from membership: the browser reports the newly
    # focused key plus the literal highlighted membership, each as the exact
    # catalog object.  Accumulating visits into plot membership is the page's
    # job.
    assert commands[0].frame is catalog[1]
    highlighted = tuple(
        index.data(QtCore.Qt.ItemDataRole.UserRole)
        for index in selection.selectedRows()
    )
    assert len(commands[0].frames) == len(highlighted)
    assert all(
        actual is expected
        for actual, expected in zip(
            commands[0].frames, highlighted, strict=True
        )
    )


def test_e3_ui2_evicted_selection_requests_hydration_without_blank(
    shell: ScatteringWorkspaceShell,
    qapp: QtWidgets.QApplication,
) -> None:
    commands: list[ShellCommand] = []
    shell.commandRequested.connect(commands.append)
    before = np.array(shell.scientific.raw.image.image, copy=True)

    shell.scientific.frame_selector.setCurrentIndex(1)
    assert len(commands) == 1
    assert commands[0].kind is ShellCommandKind.HYDRATE_FRAME
    assert commands[0].frame is shell.scientific._frame_keys[1]
    np.testing.assert_array_equal(shell.scientific.raw.image.image, before)

    commands.clear()
    shell.apply_state(
        make_shell_projection(
            revision=2,
            selected_index=1,
            heavy_indices=(0, 4),
        )
    )
    qapp.processEvents()
    assert commands == []
    assert shell.scientific.title.text() == "scan-a:2"
    np.testing.assert_array_equal(shell.scientific.raw.image.image, before)

    shell.apply_state(
        make_shell_projection(
            revision=3,
            selected_index=4,
            heavy_indices=(0, 4),
        )
    )
    assert not np.array_equal(shell.scientific.raw.image.image, before)


def test_e3_ui2_commands_reject_scientific_arrays() -> None:
    with pytest.raises(TypeError, match="detached scalars"):
        ShellCommand(
            ShellCommandKind.SET_BACKGROUND,
            np.ones((2, 2)),
        )
