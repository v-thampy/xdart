"""Real persisted cakes through Browse controls, worker and renderer."""
from __future__ import annotations

import time

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from tests.xdart.scattering.test_e4_preview_transport import _write_processed
from tests.xdart.scattering.test_p3_experiment_operation_composition import _page
from xdart.gui.tabs.scattering.browse_values import BrowseLoadStatus
from xdart.gui.tabs.scattering.events import CleanupStatus


def _wait(app, function, page):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        app.processEvents()
        value = function()
        if value:
            return value
        time.sleep(0.005)
    raise AssertionError(page._notice_text)


def _ready(page, count, *, pins=0):
    page._drain_executor()
    page._refresh_shell()
    state = page._last_scientific_projection
    view = page._shell.scientific
    return state if (state is not None and len(state.traces) == count
                     and len(state.pinned_traces) == pins
                     and state.heavy is not None
                     and state.heavy.frame is page._context_controller.navigation.current
                     and view.raw.canvas.displayed_image.size
                     and view.cake.canvas.displayed_image.size
                     and not page._scientific_repaint_pending) else None


def _close(page, app):
    _wait(app, lambda: page.close_workspace().cleanup_status is CleanupStatus.CLEANED, page)
    page.deleteLater()
    app.processEvents()


@pytest.mark.parametrize("mode", ["Single", "Overlay", "Waterfall"])
def test_selected_saved_cakes_slice_in_each_plot_mode(tmp_path, monkeypatch, mode):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    path, _ = _write_processed(tmp_path / "source", schema_version=3,
                               labels=(1, 2, 3, 4))
    page, _ = _page(tmp_path, monkeypatch)
    controller = page._context_controller
    view = page._shell.scientific
    try:
        request = controller.begin_browse(str(path.resolve()))
        outcome = _wait(app, controller.poll_browse, page)
        assert outcome.request is request and outcome.status is BrowseLoadStatus.READY
        page._refresh_shell()
        view.plot_mode.setCurrentText(mode)
        frames = controller.navigation.frames
        assert controller.select_navigation(frames[-1], frames)
        _wait(app, lambda: _ready(page, 4), page)
        view.slice_center.setValue(-1)
        view.slice_width.setValue(0.1)
        view.slice.click()
        for center, base in ((-1, np.array([0., 2., 4.])),
                             (1, np.array([1., 3., 5.]))):
            view.slice_center.setValue(center)
            state = _wait(app, lambda: _ready(page, 4), page)
            assert tuple(trace.frame for trace in state.traces) == frames
            for trace, frame in zip(state.traces, frames, strict=True):
                np.testing.assert_array_equal(trace.intensity, base + frame.local_frame_label)
                assert "±0.1" in trace.title
            assert controller.navigation.selected == frames
            assert view.raw.canvas.displayed_image.size
            assert view.cake.canvas.displayed_image.size
            if mode == "Waterfall":
                assert view.bottom_waterfall_active
        view.slice.click()
        state = _wait(app, lambda: _ready(page, 4), page)
        for trace in state.traces:
            np.testing.assert_array_equal(trace.intensity,
                                          np.array([1., 2., 3.]) + trace.frame.local_frame_label)
    finally:
        _close(page, app)


def test_browse_pin_keeps_its_cut_across_settings_and_selection(tmp_path, monkeypatch):
    from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    path, _ = _write_processed(tmp_path / "source", schema_version=3,
                               labels=(1, 2, 3, 4))
    page, _ = _page(tmp_path, monkeypatch)
    controller, view = page._context_controller, page._shell.scientific
    try:
        controller.begin_browse(str(path.resolve()))
        assert _wait(app, controller.poll_browse, page).status is BrowseLoadStatus.READY
        page._refresh_shell()
        view.plot_mode.setCurrentText("Overlay")
        frames = controller.navigation.frames
        assert controller.select_navigation(frames[-1], frames)
        _wait(app, lambda: _ready(page, 4), page)
        view.slice_center.setValue(-1)
        view.slice_width.setValue(0.1)
        view.slice.click()
        _wait(app, lambda: _ready(page, 4), page)
        assert page._edit_scientific_preference(ShellCommand(ShellCommandKind.PIN_SLICE))
        state = _wait(app, lambda: _ready(page, 3, pins=1), page)
        frozen = state.pinned_traces[0].trace
        np.testing.assert_array_equal(frozen.intensity, [4., 6., 8.])
        assert frozen.frame is frames[-1]
        view.slice_center.setValue(1)
        state = _wait(app, lambda: _ready(page, 4, pins=1), page)
        assert state.pinned_traces[0].trace.title == frozen.title
        np.testing.assert_array_equal(state.pinned_traces[0].trace.intensity, frozen.intensity)
        assert controller.select_navigation(frames[0], frames[:2])
        state = _wait(app, lambda: _ready(page, 2, pins=1), page)
        np.testing.assert_array_equal(state.pinned_traces[0].trace.intensity, frozen.intensity)
        assert page._edit_scientific_preference(ShellCommand(ShellCommandKind.PIN_SLICE))
        state = _wait(app, lambda: _ready(page, 1, pins=2), page)
        assert [item.pin.frame for item in state.pinned_traces] == [frames[-1], frames[0]]
        view.slice.click()
        state = _wait(app, lambda: _ready(page, 2, pins=2), page)
        np.testing.assert_array_equal(state.traces[0].intensity, [2., 3., 4.])
        np.testing.assert_array_equal(state.pinned_traces[0].trace.intensity, frozen.intensity)
        view.plot_mode.setCurrentText("Single")
        _wait(app, lambda: _ready(page, 1), page)
        assert not page._preferences.slice_pins
    finally:
        _close(page, app)


def test_slice_without_saved_cake_never_returns_native_trace(tmp_path):
    from xdart.gui.tabs.scattering.scientific_axes import trace_projection
    from tests.xdart.scattering.e3_shell_support import make_shell_projection
    from xdart.gui.tabs.scattering.display_values import StandardDisplayPayload
    from xrd_tools.core import Axis, FrameView

    frame = make_shell_projection(frame_count=1).navigation.current
    payload = StandardDisplayPayload(1, frame, "native only", FrameView(
        label=frame.local_frame_label,
        axis_1d=Axis("Q", "q_A^-1", values=np.array([1., 2., 3.])),
        intensity_1d=np.array([2., 3., 4.]),
    ))
    assert trace_projection(payload, requested_axis="Q", allow_cake=True,
                            slice_enabled=False, slice_center=0., slice_width=1.) is not None
    assert trace_projection(payload, requested_axis="Q", allow_cake=True,
                            slice_enabled=True, slice_center=0., slice_width=1.) is None


@pytest.mark.parametrize("mode,count", [("Single", 4), ("Overlay", 4),
                                        ("Single", 17), ("Waterfall", 4)])
def test_full_chi_selection_never_presents_native_q(tmp_path, monkeypatch, mode, count):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    path, _ = _write_processed(tmp_path / "source", schema_version=3,
                               labels=tuple(range(1, count + 1)))
    page, _ = _page(tmp_path, monkeypatch)
    controller, view = page._context_controller, page._shell.scientific
    presented = []
    reconcile = view.reconcile

    def observe(state, *args, **kwargs):
        if page._preferences.plot_axis == "chi":
            presented.append((state.plot_axis, tuple(trace.axis.unit for trace in state.traces)))
        return reconcile(state, *args, **kwargs)

    monkeypatch.setattr(view, "reconcile", observe)
    try:
        controller.begin_browse(str(path.resolve()))
        assert _wait(app, controller.poll_browse, page).status is BrowseLoadStatus.READY
        page._refresh_shell()
        view.plot_mode.setCurrentText(mode)
        frames = controller.navigation.frames
        assert controller.select_navigation(frames[-1], frames)
        _wait(app, lambda: _ready(page, count), page)
        view.plot_axis.setCurrentIndex(view.plot_axis.findData("chi"))
        assert page._preferences.plot_axis == "chi"
        assert not page._preferences.slice_enabled
        for current in (frames[-1], frames[0]):
            assert controller.select_navigation(current, frames)
            state = _wait(app, lambda: _ready(page, count), page)
            assert presented and all(axis == "chi" and all(unit == "chi_deg" for unit in units)
                                     for axis, units in presented), presented
            assert tuple(trace.frame for trace in state.traces) == frames
            for trace in state.traces:
                np.testing.assert_array_equal(trace.axis.values, [-1., 1.])
                np.testing.assert_array_equal(trace.intensity,
                                              np.array([2., 3.]) + trace.frame.local_frame_label)
                assert not trace.intensity.flags.writeable
            if mode == "Waterfall" or count > 15:
                assert view.bottom_waterfall_active
        view.plot_axis.setCurrentIndex(view.plot_axis.findData("Q"))
        state = _wait(app, lambda: _ready(page, count), page)
        for trace in state.traces:
            np.testing.assert_array_equal(trace.intensity,
                                          np.array([1., 2., 3.]) + trace.frame.local_frame_label)
    finally:
        _close(page, app)


def test_full_chi_without_saved_cake_never_substitutes_native_q():
    from xdart.gui.tabs.scattering.browse_slice_hydration import BrowseSliceLane
    from xdart.gui.tabs.scattering.display_values import StandardDisplayPayload
    from tests.xdart.scattering.e3_shell_support import make_shell_projection
    from xrd_tools.core import Axis, FrameView

    frame = make_shell_projection(frame_count=1).navigation.current
    payload = StandardDisplayPayload(1, frame, "native only", FrameView(
        label=frame.local_frame_label,
        axis_1d=Axis("Q", "q_A^-1", values=np.array([1., 2., 3.])),
        intensity_1d=np.array([2., 3., 4.]),
    ))
    with pytest.raises(RuntimeError, match="no saved 2-D data"):
        BrowseSliceLane._cut(payload, "chi", False, 0., 1., "")
