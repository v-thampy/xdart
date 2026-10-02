"""Native pages place analysis tools in Int's shared browser/tools column."""
import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.pages.services import empty_host_services
from xdart.gui.pages.values import PageCleanup
from xdart.gui.tabs.scattering.tools_view import ToolsView
from tests.xdart.pages.test_stitch_page import _Status, qapp  # noqa: F401


@pytest.mark.parametrize("mode, names", (
    ("stitch", ("peak_fit", "phase_fit")),
    ("rsm", ("lineout", "fit2d", "fit3d")),
))
def test_native_fit_tools_mount_ints_column_and_leave_display_controls_in_place(qapp, mode, names):
    if mode == "stitch":
        from xdart.gui.tabs.stitch.page import StitchPage
        page = StitchPage(empty_host_services(_Status()))
        view = page.scientific
    else:
        from xdart.gui.tabs.rsm.page import RSMPage
        page = RSMPage(empty_host_services(_Status()))
        view = page.view
    try:
        assert type(page.tools) is ToolsView
        column = page.splitter.widget(0)
        assert column is page.browser.parentWidget() is page.tools.parentWidget()
        assert column.layout().indexOf(page.browser) == 0
        assert column.layout().indexOf(page.tools) == 1
        assert tuple(page.tools._tool_buttons) == names
        for name in names:
            assert page.tools.page_tool(name).parentWidget() is page.tools.tool_content
            assert view.findChild(QtWidgets.QPushButton, f"e7PageTool_{name}") is None
        assert view.findChild(QtWidgets.QPushButton, "e7PageTool_export-run") is None
        assert "Export Run notebook…" in [action.text() for action in page.browser.menus["Help"].actions()]
        if mode == "rsm":
            assert [view.page_tool(name).text() for name in ("3d", "custom")] == ["3D Viewer", "Custom Slice"]
    finally:
        receipt = page.shutdown() if mode == "stitch" else page.close_page()
        assert receipt.status is PageCleanup.CLEAN
        page.deleteLater()
        qapp.processEvents()


def test_int_tools_keep_their_existing_buttons_and_command_kind(qapp):
    from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
    from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell
    shell = ScatteringWorkspaceShell()
    commands = []
    shell.commandRequested.connect(commands.append)
    try:
        assert tuple(shell.tools._tool_buttons) == (
            "peak_fitting", "phase_fitting", "plot_metadata", "roi_statistics",
            "nexpy_selected", "silx_h5viewer")
        shell.tools._tool_buttons["peak_fitting"].click()
        assert commands == [ShellCommand(ShellCommandKind.LAUNCH_TOOL, "peak_fitting")]
    finally:
        shell.close()
        shell.deleteLater()
        qapp.processEvents()
