"""BROWSE-1D-128: a standard result opened after a GI one draws its 1-D trace on its own axes (Claude, 2026-10-30).

The maintainer's live report: "some processed nexus files are not showing the 1d traces with and without all frames
toggled. i checked with nexpy and the integrated 1d data is there", then "i think it has to do with the integration
axis units". Opening a GI result moves the plot axis to Qᵢₚ and the map to Qᵢₚ-Qₒₒₚ. A standard result opened next
offered Q and Q-χ in its selectors while the page still asked it for Qᵢₚ, which it does not have, so nothing was drawn.

Real pages and real processed outputs; each result is opened as a click on its row does.
"""
from __future__ import annotations

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from tests.xdart.scattering.test_e4_preview_transport import _write_processed
from tests.xdart.scattering.test_gi_browse_preview_modes import _write_named_gi
from tests.xdart.scattering.test_p3_experiment_operation_composition import _page
from tests.xdart.scattering.test_p34_reintegrate_operation import _wait
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(params=("Int 2D", "Int 1D"))
def workspace(request, qapp, tmp_path, monkeypatch):
    page, _ = _page(tmp_path, monkeypatch, store=RunIntentStore(RunIntent(
        project_root=str(tmp_path), processing_mode=request.param)))
    page.resize(1200, 800)
    page.show()
    try:
        yield page
    finally:
        _wait(lambda: True if page.close_workspace().cleanup_status is CleanupStatus.CLEANED else None)
        page.deleteLater()
        qapp.processEvents()


def _open(page, path):
    """Open ``path`` as a click on its row does; return the projection once its trace is painted."""
    page._handle_shell_command(ShellCommand(ShellCommandKind.SELECT_SCAN, str(path), path=("artifact",)))
    controller, view = page._context_controller, page._shell.scientific

    def painted():
        frame = controller.navigation.current
        state = page._last_scientific_projection
        idle = (not controller.browse_pending and not page._scientific_repaint_pending
                and not page._shell.browser.frame_selection_pending)
        return True if (frame is not None and frame.artifact == str(path) and idle and state is not None
                        and state.traces and frame in view.trace_history_keys) else None
    _wait(painted, timeout=10)
    return page._last_scientific_projection


def test_a_standard_result_after_a_gi_one_draws_its_trace_on_its_own_axes(workspace, tmp_path):
    page, view = workspace, workspace._shell.scientific
    standard, _ = _write_processed(tmp_path / "standard", labels=(1, 2, 3))
    gi = tmp_path / "gi" / "scan_gi_int2d.nexus"
    gi.parent.mkdir()
    _write_named_gi(gi)

    _open(page, standard)
    assert page._preferences.plot_axis == "Q", "control: the standard result alone draws on Q"
    _open(page, gi)
    assert page._preferences.plot_axis == "q_ip", "the GI result moves the plot to its own Qᵢₚ"

    state = _open(page, standard)
    trace = state.traces[0]
    assert trace.axis.unit == "q_A^-1" and np.isfinite(trace.intensity).all(), "its own Q trace is drawn"
    assert page._preferences.plot_axis == "Q" and view.plot_axis.currentData() == "Q"
    assert page._preferences.image_axis == "Q-Chi" and view.image_axis.currentData() == "Q-Chi", (
        "the map selection the GI result left falls back to this result's own")
