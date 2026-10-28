"""Cross-file comparison through real processed Browse and shared widgets."""

from __future__ import annotations

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from tests.xdart.scattering.test_e4_accumulation_selection_parity import _click_frame
from tests.xdart.scattering.test_e4_preview_transport import _write_processed
from tests.xdart.scattering.test_p3_experiment_operation_composition import _page
from tests.xdart.scattering.test_p34_reintegrate_operation import _wait
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering.hydration_transport import HydrationTransport
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xrd_tools.io.read import get_1d
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.mark.parametrize("waits", (False, True), ids=("opens", "waits"))
@pytest.mark.parametrize("mode", ("Int 1D", "Int 2D"))
def test_int_overlay_retains_selected_point_when_browsing_another_file(
    tmp_path, monkeypatch, qapp, mode, waits,
):
    first, _ = _write_processed(tmp_path / "first", labels=(1, 2))
    second, _ = _write_processed(tmp_path / "second", labels=(1, 2))
    persisted = (get_1d(first, 2), get_1d(second, 1))
    store = RunIntentStore(RunIntent(
        project_root=str(tmp_path), processing_mode=mode,
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
            and state is not None
            and any(trace.frame is frame for trace in state.traces)
            and frame in view.trace_history_keys
            and (mode == "Int 1D" or (
                state.heavy is not None and state.heavy.frame is frame
                and view.raw.canvas.displayed_image.size
                and view.cake.canvas.displayed_image.size
            ))
        ) else None

    def open_file(path):
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.SELECT_SCAN, str(path), path=("artifact",),
        ))
        kept = page._deferred_browse_reopen is not None
        _wait(lambda: settled(path, 1))
        return kept

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

        if waits:
            # The first file's preview reads still hold its gate, so the open waits and is replayed (hosted run
            # 40636668726); the wait must keep the point the overlay carries into the second file.
            real, asked = HydrationTransport.retains_gate, []

            def retains_gate(self, gate):
                asked.append(gate)
                return len(asked) == 1 or real(self, gate)
            monkeypatch.setattr(HydrationTransport, "retains_gate", retains_gate)
        assert open_file(second) is waits
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


@pytest.fixture
def comparison(tmp_path, monkeypatch, qapp, request):
    paths = tuple(_write_processed(tmp_path / name, labels=(1, 2))[0]
                  for name in ("left", "right"))
    if getattr(request, "param", False):
        import h5py
        from xrd_tools.core.angular_support import AngularSupportProfile, SupportOrigin
        from xrd_tools.io.angular_support import write_angular_support

        profile = AngularSupportProfile(
            origin=SupportOrigin.RUN, supported_deg=np.array([90., 180., 270.]),
            window_deg=360., convention="detector_chi", radial_unit="q_A^-1",
            bin_edges=(0.05, 0.35),
        )
        for path in paths:
            with h5py.File(path, "r+") as handle:
                group = handle["entry/integrated_1d"]
                write_angular_support(group, {1: profile, 2: profile},
                                      intensity_rows=dict(zip((1, 2), group["intensity"][()], strict=True)))
    store = RunIntentStore(RunIntent(project_root=str(tmp_path), processing_mode="Int 2D"))
    page, _ = _page(tmp_path, monkeypatch, store=store)
    controller, view = page._context_controller, page._shell.scientific

    def ready():
        state = page._last_scientific_projection
        frame = controller.navigation.current
        return state if (state is not None and state.heavy is not None
                         and state.heavy.frame is frame and state.traces
                         and not controller.browse_pending and not page._scientific_repaint_pending
                         and not page._shell.browser.frame_selection_pending) else None

    def open_file(path):
        page._handle_shell_command(ShellCommand(ShellCommandKind.SELECT_SCAN, str(path), path=("artifact",)))
        _wait(lambda: ready() if controller.navigation.current is not None
              and controller.navigation.current.artifact == str(path) else None)
        return controller.navigation.current

    page.resize(1200, 800)
    page.show()
    try:
        first = open_file(paths[0])
        view.plot_mode.setCurrentText("Overlay")
        _wait(ready)
        yield page, paths, first, open_file, ready
    finally:
        _wait(lambda: True if page.close_workspace().cleanup_status is CleanupStatus.CLEANED else None)
        page.deleteLater()
        qapp.processEvents()


def test_cross_file_overlay_removal_clear_and_single_keep_current_authority(comparison, qapp):
    page, paths, first, open_file, ready = comparison
    controller, shell = page._context_controller, page._shell
    view = shell.scientific
    second = open_file(paths[1])
    assert view.trace_history_keys == (first, second)
    _click_frame(qapp, shell, 1)
    _wait(ready)
    last = controller.navigation.current
    assert view.trace_history_keys == (first, second, last)
    _click_frame(qapp, shell, 0, modifiers=QtCore.Qt.KeyboardModifier.ControlModifier)
    _wait(ready)
    assert controller.navigation.current is second
    assert controller.navigation.selected == (last,)
    assert view.trace_history_keys == (first, last)
    assert page._last_scientific_projection.heavy.frame is second
    page._handle_shell_command(ShellCommand(ShellCommandKind.CLEAR_1D))
    _wait(ready)
    assert view.trace_history_keys == (second,)
    open_file(paths[0])
    assert any(frame.artifact == str(paths[1]) for frame in view.trace_history_keys)
    view.plot_mode.setCurrentText("Single")
    _wait(ready)
    assert view.trace_history_keys == controller.navigation.selected
    view.plot_mode.setCurrentText("Overlay")
    _wait(ready)
    assert view.trace_history_keys == controller.navigation.selected


def test_cross_file_overlay_semantic_change_discards_prior_science(comparison):
    page, paths, first, open_file, ready = comparison
    second = open_file(paths[1])
    view = page._shell.scientific
    assert view.trace_history_keys == (first, second)
    view.slice.click()
    _wait(ready)
    assert view.trace_history_keys == (second,)
    assert page._last_scientific_projection.slice_enabled


def test_cross_file_overlay_different_sampled_grid_starts_new_scope_with_notice(comparison):
    import h5py

    page, paths, first, open_file, ready = comparison
    # Mutate the second real saved artifact before it is admitted, preserving its schema.
    with h5py.File(paths[1], "r+") as handle:
        axis = handle["entry/integrated_1d/axis_1"]
        axis[...] = axis[...] * 2
    second = open_file(paths[1])
    assert page._shell.scientific.trace_history_keys == (second,)
    assert "new overlay" in page._notice_text.lower()
    assert "axes" in page._notice_text.lower()


def test_cross_file_overlay_counts_shared_roots_against_one_global_budget(comparison, monkeypatch):
    from xdart.gui.tabs.scattering import browse_1d_display

    page, paths, first, open_file, ready = comparison
    view = page._shell.scientific

    def root_bytes():
        roots = {}
        for trace in view._trace_history_by_identity.values():
            arrays = [trace.axis.values, trace.intensity]
            if trace.angular_support is not None and trace.angular_support.supported_deg is not None:
                arrays.append(trace.angular_support.supported_deg)
            for array in arrays:
                while isinstance(array.base, np.ndarray):
                    array = array.base
                roots[id(array)] = array
        return sum(array.nbytes for array in roots.values())

    limit = root_bytes() * 3 // 2
    monkeypatch.setattr(browse_1d_display, "MAX_BROWSE_1D_DETACHED_BYTES", limit)
    second = open_file(paths[1])
    assert root_bytes() <= limit
    assert view.trace_history_keys == (second,)
    assert "new overlay" in page._notice_text.lower()
    assert "budget" in page._notice_text.lower()


@pytest.mark.parametrize("comparison", [True], indirect=True)
def test_cross_file_overlay_detaches_and_budgets_saved_angular_support(comparison, monkeypatch):
    from xdart.gui.tabs.scattering import browse_1d_display

    page, paths, first, open_file, ready = comparison
    view = page._shell.scientific
    admitted = page._last_scientific_projection.traces[0].angular_support
    retained = view._trace_history_by_identity[id(first)].angular_support
    assert admitted.available and retained.available
    assert retained.supported_deg is not admitted.supported_deg
    assert not np.shares_memory(retained.supported_deg, admitted.supported_deg)
    assert not retained.supported_deg.flags.writeable
    np.testing.assert_array_equal(retained.supported_deg, admitted.supported_deg)

    roots = {}
    for array in (view._trace_history_by_identity[id(first)].axis.values,
                  view._trace_history_by_identity[id(first)].intensity):
        while isinstance(array.base, np.ndarray):
            array = array.base
        roots[id(array)] = array
    # Two x/y allocations fit; adding both support roots exceeds the same budget.
    limit = 2 * sum(array.nbytes for array in roots.values()) + retained.supported_deg.nbytes
    monkeypatch.setattr(browse_1d_display, "MAX_BROWSE_1D_DETACHED_BYTES", limit)
    second = open_file(paths[1])
    assert view.trace_history_keys == (second,)
    assert "budget" in page._notice_text.lower()
