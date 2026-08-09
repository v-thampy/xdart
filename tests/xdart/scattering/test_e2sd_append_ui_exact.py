from __future__ import annotations

from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.controls_projection import OUTPUT_MODE
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.shell_values import (
    ShellCommand,
    ShellCommandKind,
)
from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell
from xdart.gui.tabs.static_scan.ui.controls_panel_v2 import FormRow
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent


def _shell(page: ScatteringWorkspace) -> ScatteringWorkspaceShell:
    shell = page.findChild(ScatteringWorkspaceShell)
    assert shell is not None
    return shell


def test_unsupported_append_cannot_remain_a_selectable_visual_mode() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    run_store = RunIntentStore(RunIntent(output_mode="Append"))
    page = ScatteringWorkspace(
        intents=run_store,
        lifecycle=ScatteringCoordinator(),
        sources=object(),
    )
    try:
        shell = _shell(page)
        button = shell.run_controls.writeModeButton
        assert shell.run_controls.write_mode() == "Append"
        button.click()
        app.processEvents()
        assert run_store.snapshot().thaw().output_mode == "Overwrite"
        assert shell.run_controls.write_mode() == "Overwrite"
        assert button.text() == "Replace ⇄"

        button.click()
        app.processEvents()
        assert run_store.snapshot().thaw().output_mode == "Append"
        assert shell.run_controls.write_mode() == "Append"
        assert button.text() == "Append ⇄"
    finally:
        page.close_workspace()


def test_refused_append_edit_cannot_diverge_view_from_intent() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    run_store = RunIntentStore(RunIntent(output_mode="Overwrite"))
    page = ScatteringWorkspace(
        intents=run_store,
        lifecycle=ScatteringCoordinator(),
        sources=object(),
    )
    try:
        shell = _shell(page)
        button = shell.run_controls.writeModeButton
        prior_revision = run_store.snapshot().revision
        assert shell.run_controls.write_mode() == "Overwrite"
        button.click()
        app.processEvents()
        assert run_store.snapshot().thaw().output_mode == "Append"
        assert run_store.snapshot().revision == prior_revision + 1
        assert shell.run_controls.write_mode() == "Append"
    finally:
        page.close_workspace()


def test_real_controls_projects_append_reason_then_exact_overwrite() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    run_store = RunIntentStore(RunIntent(output_mode="Append"))
    page = ScatteringWorkspace(
        intents=run_store,
        lifecycle=ScatteringCoordinator(),
        sources=object(),
        executor=StandardRunExecutor(),
    )
    commands: list[ShellCommand] = []
    try:
        shell = _shell(page)
        shell.commandRequested.connect(commands.append)
        button = shell.run_controls.writeModeButton
        assert run_store.snapshot().thaw().output_mode == "Append"
        assert shell.run_controls.write_mode() == "Append"
        assert button.text() == "Append ⇄"
        assert (
            shell.run_controls.readinessLabel.text()
            == "Needs source, PONI, output"
        )
        assert all(
            row.path != OUTPUT_MODE
            for row in shell.controls.findChildren(FormRow)
        )
        prior_revision = run_store.snapshot().revision

        button.click()
        app.processEvents()

        current = run_store.snapshot()
        assert current.revision == prior_revision + 1
        assert current.thaw().output_mode == "Overwrite"
        assert shell.run_controls.write_mode() == "Overwrite"
        assert button.text() == "Replace ⇄"
        assert commands == [
            ShellCommand(
                ShellCommandKind.SET_OUTPUT_POLICY,
                "Overwrite",
            )
        ]
    finally:
        page.close_workspace()
