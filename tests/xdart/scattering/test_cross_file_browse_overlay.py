"""Cross-file comparison through real processed Browse and shared widgets."""

from __future__ import annotations

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from tests.xdart.scattering.test_e4_accumulation_selection_parity import _click_frame
from tests.xdart.scattering.test_e4_preview_transport import _write_processed
from tests.xdart.scattering.test_p3_experiment_operation_composition import _page
from tests.xdart.scattering.test_p34_reintegrate_operation import _wait
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xrd_tools.io.read import get_1d
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_int2d_overlay_retains_selected_point_when_browsing_another_file(
    tmp_path, monkeypatch, qapp,
):
    first, _ = _write_processed(tmp_path / "first", labels=(1, 2))
    second, _ = _write_processed(tmp_path / "second", labels=(1, 2))
    persisted = (get_1d(first, 2), get_1d(second, 1))
    store = RunIntentStore(RunIntent(
        project_root=str(tmp_path), processing_mode="Int 2D",
    ))
    page, _ = _page(tmp_path, monkeypatch, store=store)
    controller, shell = page._context_controller, page._shell
    view = shell.scientific

    def settled(path, label):
        frame = controller.navigation.current
        state = page._last_scientific_projection
        return state if (
            frame is not None and frame.artifact == str(path)
            and frame.local_frame_label == label
            and not controller.browse_pending
            and not page._scientific_repaint_pending
            and not shell.browser.frame_selection_pending
            and state is not None and state.heavy is not None
            and state.heavy.frame is frame
            and any(trace.frame is frame for trace in state.traces)
            and frame in view.trace_history_keys
            and view.raw.canvas.displayed_image.size
            and view.cake.canvas.displayed_image.size
        ) else None

    def open_file(path):
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.SELECT_SCAN, str(path), path=("artifact",),
        ))
        return _wait(lambda: settled(path, 1))

    try:
        page.resize(1200, 800)
        page.show()
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.SET_PLOT_OPTION, 0.0, ("overlay", "offset"),
        ))
        open_file(first)
        _click_frame(qapp, shell, 1)
        _wait(lambda: settled(first, 2))
        selected_first = controller.navigation.current
        view.plot_mode.setCurrentText("Overlay")
        _wait(lambda: settled(first, 2))
        assert view.trace_history_keys == (selected_first,)

        open_file(second)
        _click_frame(qapp, shell, 0)
        _wait(lambda: settled(second, 1))
        selected_second = controller.navigation.current
        assert view.plot_mode.currentText() == "Overlay"
        assert selected_second.artifact == str(second)
        assert all(frame.artifact == str(second) for frame in controller.navigation.frames)
        assert view.trace_history_keys == (selected_first, selected_second)
        items = tuple(view.curve.listDataItems())
        assert len(items) == 2
        for item, expected in zip(items, persisted, strict=True):
            np.testing.assert_array_equal(item.xData, expected.q)
            np.testing.assert_array_equal(item.yData, expected.intensity)
    finally:
        _wait(lambda: True if page.close_workspace().cleanup_status is CleanupStatus.CLEANED else None)
        page.deleteLater()
        qapp.processEvents()
