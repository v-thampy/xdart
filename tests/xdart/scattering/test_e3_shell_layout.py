from __future__ import annotations

import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.tabs.scattering.workspace_shell import (
    ScatteringWorkspaceShell,
)
from xdart.gui.tabs.static_scan.ui.controls_panel_v2 import ControlsPanelV2
from xdart.gui.tabs.static_scan.ui.static_controls import StaticControls

from tests.xdart.scattering.e3_shell_support import make_shell_projection


@pytest.fixture
def qapp() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_e3_ui1_composes_exact_three_column_shell(
    qapp: QtWidgets.QApplication,
) -> None:
    shell = ScatteringWorkspaceShell()
    try:
        assert shell.splitter.count() == 3
        assert [
            shell.splitter.widget(index).objectName()
            for index in range(3)
        ] == [
            "e3BrowserColumn",
            "e3ScientificView",
            "e3ControlsColumn",
        ]
        assert shell.left.minimumWidth() == 255
        assert shell.scientific.minimumWidth() == 300
        # LV-UI-7: the controls column carries NO hand-pinned minimum — its
        # floor derives from the embedded panel through the width-hugging
        # scroll area, so the constants can never disagree again.
        assert shell.right.minimumWidth() == 0
        assert shell.control_scroll.minimumSizeHint().width() >= (
            shell.controls.minimumSizeHint().width()
        )
        assert type(shell.controls) is ControlsPanelV2
        assert type(shell.run_controls) is StaticControls
    finally:
        shell.close()
        shell.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def test_e3_ui4_default_column_and_scientific_row_geometry(
    qapp: QtWidgets.QApplication,
) -> None:
    shell = ScatteringWorkspaceShell()
    shell.resize(1488, 1080)
    shell.show()
    try:
        qapp.processEvents()
        assert shell.splitter.sizes() == [289, 868, 323]
        row_sizes = shell.scientific.vertical_splitter.sizes()
        assert abs(row_sizes[0] - row_sizes[1]) <= 1
    finally:
        shell.close()
        shell.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def test_e3_ui1_has_one_reconciliation_boundary_and_emits_no_command(
    qapp: QtWidgets.QApplication,
) -> None:
    shell = ScatteringWorkspaceShell()
    commands = []
    shell.commandRequested.connect(commands.append)
    try:
        shell.apply_state(make_shell_projection())
        qapp.processEvents()
        assert commands == []
        assert shell._revision == 1
        assert shell.scientific.progress.text() == "1/5"
        assert shell.run_controls.current_mode() == "Int 2D"
        assert shell.controls.profile is not None
    finally:
        shell.close()
        shell.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def test_e3_ui1_does_not_construct_legacy_authority_widgets(
    qapp: QtWidgets.QApplication,
) -> None:
    shell = ScatteringWorkspaceShell()
    try:
        types = {
            f"{type(widget).__module__}.{type(widget).__name__}"
            for widget in shell.findChildren(QtWidgets.QWidget)
        }
        assert not any(
            name.endswith(
                (
                    ".staticWidget",
                    ".H5Viewer",
                    ".displayFrameWidget",
                    ".ParameterTree",
                )
            )
            for name in types
        )
        assert not {
            "scan",
            "store",
            "source",
            "executor",
            "context",
            "worker",
            "writer",
            "lifecycle",
        } & set(vars(shell))
    finally:
        shell.close()
        shell.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
