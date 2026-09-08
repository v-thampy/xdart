"""Real Qt gestures retain exact Single-mode Browse trace membership."""

from __future__ import annotations

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from tests.core.reintegrate_support import _seed_existing
from tests.xdart.scattering.test_e4_accumulation_selection_parity import (
    _click_frame,
)
from tests.xdart.scattering.test_p3_experiment_operation_composition import (
    _close,
    _page,
)
from tests.xdart.scattering.test_p34_reintegrate_operation import (
    _loaded_page,
    _wait,
)
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xrd_tools.io.read import get_1d


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.mark.parametrize(
    "modifier,selected_rows",
    (
        (QtCore.Qt.KeyboardModifier.ControlModifier, (0, 2)),
        (QtCore.Qt.KeyboardModifier.MetaModifier, (0, 2)),
        (QtCore.Qt.KeyboardModifier.ShiftModifier, (0, 1, 2)),
    ),
    ids=("control", "command", "shift-range"),
)
def test_single_browse_modifier_selection_paints_every_selected_trace(
    tmp_path, monkeypatch, qapp, modifier, selected_rows,
):
    # The writer, Browse admission, sparse hydration, page command path and
    # renderer are real. Only the input scientific values are fixture data.
    seeded = _seed_existing(tmp_path, labels=(2, 5, 9, 12))
    persisted = {label: get_1d(seeded.target, label) for label in seeded.labels}
    page, _store, _seeded, _context = _loaded_page(
        tmp_path, monkeypatch, qapp, seed=seeded,
    )
    controller = page._context_controller
    shell = page._shell
    frames = controller.navigation.frames

    def settled_current(frame):
        page._drain_executor()
        state = page._last_scientific_projection
        if (
            state is not None
            and state.browse_trace_snapshot is not None
            and not page._scientific_repaint_pending
            and not shell.browser.frame_selection_pending
            and controller.navigation.current is frame
            and any(trace.frame is frame for trace in state.traces)
        ):
            return state
        return None

    def assert_painted(state, expected):
        assert state.plot_mode == "Single"
        assert state.browse_trace_snapshot.logical_frames == expected
        assert tuple(trace.frame for trace in state.traces) == expected
        assert shell.scientific.trace_history_keys == expected
        items = tuple(shell.scientific.curve.listDataItems())
        assert len(items) == len(expected)
        for item, frame in zip(items, expected, strict=True):
            result = persisted[frame.local_frame_label]
            np.testing.assert_array_equal(item.xData, result.q)
            np.testing.assert_array_equal(item.yData, result.intensity)
        assert page._browse_snapshot_is_exact_current(
            state, state.browse_science_contract,
        )

    try:
        # Disable the existing visual stacking offset so mounted data can be
        # compared byte-for-byte with the independently read persisted curves.
        assert page._edit_scientific_preference(ShellCommand(
            ShellCommandKind.SET_PLOT_OPTION, 0.0, ("overlay", "offset"),
        ))
        page.resize(1200, 800)
        page.show()
        page._set_browser_directory(str(seeded.target.parent), explicit=True)
        page._refresh_shell()
        _wait(lambda: (
            True if shell.browser.frame_model.rowCount() == len(frames)
            else None
        ))
        assert shell.browser.frames.isVisible()
        assert shell.scientific.plot_mode.currentText() == "Single"
        _click_frame(qapp, shell, 0)
        assert_painted(_wait(lambda: settled_current(frames[0])), (frames[0],))

        _click_frame(qapp, shell, 2, modifiers=modifier)
        selected = tuple(frames[row] for row in selected_rows)
        assert controller.navigation.selected == selected
        current = controller.navigation.current
        assert current in selected
        assert_painted(_wait(lambda: settled_current(current)), selected)

        # Single is still exact selection, not Overlay's visit accumulation.
        _click_frame(qapp, shell, 1)
        assert controller.navigation.selected == (frames[1],)
        assert_painted(_wait(lambda: settled_current(frames[1])), (frames[1],))
    finally:
        _close(page, qapp)


def test_standard_page_rejects_stale_qz_qxy_image_command(
    tmp_path, monkeypatch, qapp,
):
    page, _store = _page(tmp_path, monkeypatch)
    try:
        before = page._preferences
        assert not page._edit_scientific_preference(
            ShellCommand(ShellCommandKind.SET_IMAGE_AXIS, "Qz-Qxy"),
        )
        assert page._preferences is before
    finally:
        _close(page, qapp)
