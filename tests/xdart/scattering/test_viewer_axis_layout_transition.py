"""A real viewer-to-integration mode edit restores native axis controls."""

import pytest

from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from tests.xdart.scattering.test_viewer_1d_selection import _settle, viewer


@pytest.mark.parametrize("destination", ("Int 1D", "Int 2D"))
def test_loaded_1d_viewer_return_restores_plot_axis(viewer, destination):
    app, page, _paths = viewer
    view = page._shell.scientific
    assert view._processing_mode == "1D Viewer"
    assert view.plot_axis.isHidden()
    page._handle_shell_command(ShellCommand(
        ShellCommandKind.SET_PROCESSING_MODE, destination,
    ))
    _settle(app)
    assert page._intents.snapshot().thaw().processing_mode == destination
    assert view._processing_mode == destination
    assert not view.plot_axis.isHidden()
    assert view.plot_axis.isVisible()
    assert view.plot_axis.count() > 0
    assert view.image_axis.isVisible() is (destination == "Int 2D")
