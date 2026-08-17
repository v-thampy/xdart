from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets

from xdart.gui.tabs.scattering.scientific_view import ScientificView
from xdart.gui.tabs.scattering.shell_values import (
    FrameNavigationProjection,
    ShellCommandKind,
)

from tests.xdart.scattering.e3_shell_support import make_shell_projection


def _reconcile(
    view: ScientificView,
    scientific,
    navigation: FrameNavigationProjection,
) -> None:
    view.reconcile(
        scientific,
        navigation,
        completed=1,
        total=len(navigation.frames),
        detail="Ready",
    )


def _x_range(view_box) -> tuple[float, float]:
    return tuple(float(value) for value in view_box.viewRange()[0])


def _global_data_x(widget, view_box, value: float) -> int:
    scene_point = view_box.mapViewToScene(QtCore.QPointF(value, 0.0))
    return widget.mapToGlobal(widget.mapFromScene(scene_point)).x()


def _assert_shared_pixels_align(
    view: ScientificView,
    *values: float,
) -> None:
    cake_view = view.cake.canvas.imageViewBox
    bottom_widget, bottom_view = view._active_bottom_plot()
    for value in values:
        cake_x = _global_data_x(
            view.cake.canvas.image_win,
            cake_view,
            value,
        )
        curve_x = _global_data_x(bottom_widget, bottom_view, value)
        assert abs(cake_x - curve_x) <= 1


def _dispose(view: ScientificView) -> None:
    view.close()


@pytest.mark.parametrize("available", (True, False))
def test_int2d_detector_control_is_exclusive_and_acquisition_gated(
    available: bool,
) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    view = ScientificView()
    projection = make_shell_projection(plot_mode="Single")
    scientific = replace(
        projection.scientific,
        processing_mode="Int 2D",
        detector_mode="thumbnail",
        detector_available=available,
        detector_diagnostic=(
            "" if available
            else "Full Raw is available only for an exact acquisition frame."
        ),
    )
    commands = []
    view.commandRequested.connect(commands.append)
    try:
        _reconcile(view, scientific, projection.navigation)
        assert view.detector_mode_group.exclusive()
        assert view.detector_thumbnail.isChecked()
        assert view.detector_full.isEnabled() is available
        assert not commands
        view.detector_full.click()
        if available:
            assert commands[-1].kind is ShellCommandKind.SET_DETECTOR_MODE
            assert commands[-1].value == "full"
        else:
            assert not commands
            assert view.detector_full.toolTip() == scientific.detector_diagnostic
    finally:
        _dispose(view)


def test_int1d_raw_popup_tracks_exact_current_and_releases_full_references() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    view = ScientificView()
    projection = make_shell_projection(frame_count=2, plot_mode="Single")
    frames = projection.navigation.frames
    thumbnail = np.arange(6, dtype=np.float32).reshape(2, 3)
    thumbnail.flags.writeable = False
    first = replace(
        projection.scientific.heavy,
        frame=frames[0], raw=thumbnail, detector_shape=(8, 12),
        detector_source="thumbnail",
    )
    scientific = replace(
        projection.scientific, processing_mode="Int 1D", heavy=first,
        detector_mode="thumbnail", detector_available=True,
    )
    commands = []
    def collect(command):
        commands.append(command)
        if command.value == "thumbnail" and view.raw_popup_dialog.isVisible():
            _reconcile(view, scientific, projection.navigation)
    view.commandRequested.connect(collect)
    try:
        _reconcile(view, scientific, projection.navigation)
        assert view.raw.image.image is None
        view.raw_popup_button.click()
        assert view.raw_popup_dialog.isVisible()
        np.testing.assert_array_equal(
            view.raw_popup_image.image.image, thumbnail.T[:, ::-1]
        )
        assert commands[-1].value == "full" and commands[-1].path == ("popup",)

        full = np.arange(20, dtype=np.float32).reshape(4, 5)
        full.flags.writeable = False
        second = replace(
            first, frame=frames[1], raw=full, detector_shape=None,
            detector_source="full",
        )
        navigation = FrameNavigationProjection(frames, frames[1], (frames[1],))
        _reconcile(
            view,
            replace(scientific, heavy=second, detector_mode="full"),
            navigation,
        )
        assert view.raw.image.image is None
        np.testing.assert_array_equal(
            view.raw_popup_image.image.image, full.T[:, ::-1]
        )
        view.raw_popup_dialog.close()
        app.processEvents()
        assert commands[-1].value == "thumbnail"
        assert view.raw_popup_image.image.image is None
        assert view.raw_popup_image.canvas.raw_image.size == 0
    finally:
        _dispose(view)


def test_overlay_footer_moves_anchor_without_mutating_accumulator() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    view = ScientificView()
    projection = make_shell_projection(
        frame_count=3,
        heavy_indices=(0, 1, 2),
        plot_mode="Overlay",
        source_scan="scan-a",
    )
    frames = projection.navigation.frames
    commands = []
    view.commandRequested.connect(commands.append)
    try:
        first_navigation = FrameNavigationProjection(
            frames,
            frames[0],
            (frames[0],),
        )
        _reconcile(view, projection.scientific, first_navigation)

        assert view.frame_selector.count() == len(frames)
        assert len(view.curve.listDataItems()) == 1
        first_curve = view.curve.listDataItems()[0]
        cursor = view.cursor_position

        view.frame_selector.setCurrentIndex(1)
        command = commands[-1]
        assert command.kind is ShellCommandKind.SELECT_FRAME
        assert command.frame is frames[1]
        assert command.frames == (frames[0],)

        second_navigation = FrameNavigationProjection(
            frames,
            frames[1],
            (frames[0],),
        )
        second_heavy = replace(
            projection.scientific.heavy,
            frame=frames[1],
        )
        _reconcile(
            view,
            replace(projection.scientific, heavy=second_heavy),
            second_navigation,
        )

        items = view.curve.listDataItems()
        assert len(items) == 1
        assert items[0] is first_curve
        assert len(view.legend.items) == 1
        assert all(item.opts["symbol"] == "o" for item in items)
        assert all(item.opts["pen"].widthF() == 1.4 for item in items)
        assert view.cursor_position is cursor

        view.frame_selector.setCurrentIndex(2)
        command = commands[-1]
        assert command.frame is frames[2]
        assert command.frames == (frames[0],)

        third_navigation = FrameNavigationProjection(
            frames,
            frames[2],
            (frames[0],),
        )
        third_heavy = replace(
            projection.scientific.heavy,
            frame=frames[2],
        )
        _reconcile(
            view,
            replace(projection.scientific, heavy=third_heavy),
            third_navigation,
        )
        assert len(view.curve.listDataItems()) == 1
        assert len(view.legend.items) == 1

        single = replace(projection.scientific, plot_mode="Single")
        single_navigation = FrameNavigationProjection(
            frames,
            frames[2],
            (frames[2],),
        )
        _reconcile(view, single, single_navigation)

        items = view.curve.listDataItems()
        assert len(items) == 1
        assert len(view.legend.items) == 1
        assert all(item.opts["symbol"] == "o" for item in items)
        assert all(item.opts["pen"].widthF() == 1.4 for item in items)
        assert view.cursor_position is cursor

        view.frame_selector.setCurrentIndex(0)
        command = commands[-1]
        assert command.frame is frames[0]
        assert command.frames == (frames[0],)
    finally:
        _dispose(view)


def test_share_axis_links_both_directions_survives_repaint_and_unlinks() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    view = ScientificView()
    projection = make_shell_projection(
        frame_count=5,
        selected_index=0,
        heavy_indices=(0, 4),
        plot_mode="Single",
    )
    frames = projection.navigation.frames
    scientific = replace(projection.scientific, share_axis=True)
    view.resize(1200, 800)
    view.show()
    try:
        _reconcile(view, scientific, projection.navigation)
        app.processEvents()

        cake_view = view.cake.canvas.imageViewBox
        curve_view = view.curve.getPlotItem().getViewBox()
        assert view.share_axis.isEnabled()
        assert view.share_axis.isChecked()
        QtTest.QTest.qWait(80)
        _assert_shared_pixels_align(view, 0.5, 1.5, 2.5)

        range_events: list[str] = []
        cake_view.sigXRangeChanged.connect(
            lambda *_args: range_events.append("cake")
        )
        curve_view.sigXRangeChanged.connect(
            lambda *_args: range_events.append("curve")
        )
        cake_view.setXRange(0.45, 2.15, padding=0.0)
        QtTest.QTest.qWait(80)
        _assert_shared_pixels_align(view, 0.55, 1.25, 2.05)
        settled_event_count = len(range_events)
        for _ in range(3):
            app.processEvents()
        assert len(range_events) == settled_event_count
        assert not view._share_axis_syncing

        curve_view.setXRange(0.75, 1.85, padding=0.0)
        QtTest.QTest.qWait(80)
        _assert_shared_pixels_align(view, 0.8, 1.3, 1.8)

        next_navigation = FrameNavigationProjection(
            frames,
            frames[4],
            (frames[4],),
        )
        next_heavy = replace(
            scientific.heavy,
            frame=frames[4],
            raw=np.array(scientific.heavy.raw, copy=True),
            cake=np.array(scientific.heavy.cake, copy=True),
        )
        next_heavy.raw.flags.writeable = False
        next_heavy.cake.flags.writeable = False
        _reconcile(
            view,
            replace(scientific, heavy=next_heavy),
            next_navigation,
        )
        QtTest.QTest.qWait(80)

        assert view.share_axis.isChecked()
        _assert_shared_pixels_align(view, 0.8, 1.3, 1.8)

        view.resize(1460, 720)
        QtTest.QTest.qWait(80)
        _assert_shared_pixels_align(view, 0.8, 1.3, 1.8)

        _reconcile(
            view,
            replace(scientific, share_axis=False),
            next_navigation,
        )
        app.processEvents()
        assert not view.share_axis.isChecked()
        curve_before = _x_range(curve_view)
        cake_view.setXRange(0.25, 1.25, padding=0.0)
        app.processEvents()
        np.testing.assert_allclose(_x_range(curve_view), curve_before)
        assert not np.allclose(_x_range(curve_view), _x_range(cake_view))
    finally:
        _dispose(view)


def test_share_axis_geometry_coalescing_cannot_starve() -> None:
    """A sustained resize stream still admits a bounded alignment callback."""

    view = ScientificView()
    calls: list[None] = []
    view._align_curve_under_cake = lambda: calls.append(None)
    try:
        for _ in range(8):
            view._schedule_curve_under_cake()
            QtTest.QTest.qWait(10)

        assert calls
    finally:
        _dispose(view)


def test_share_axis_stays_available_for_derivable_radial_unit_mismatch() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    view = ScientificView()
    projection = make_shell_projection(plot_mode="Single")
    trace = projection.scientific.traces[0]
    mismatched_trace = replace(
        trace,
        axis=replace(trace.axis, label="2θ", unit="°"),
    )
    scientific = replace(
        projection.scientific,
        traces=(mismatched_trace,),
        share_axis=True,
    )
    try:
        _reconcile(view, scientific, projection.navigation)

        assert view.share_axis.isEnabled()
        assert not view.share_axis.isChecked()
        assert not view._share_link_on
    finally:
        _dispose(view)


def test_share_axis_refuses_nonradial_rendered_unit_mismatch() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    view = ScientificView()
    projection = make_shell_projection(plot_mode="Single")
    trace = projection.scientific.traces[0]
    mismatched_trace = replace(
        trace,
        axis=replace(trace.axis, label="χ", unit="°"),
    )
    scientific = replace(
        projection.scientific,
        traces=(mismatched_trace,),
        share_axis=True,
    )
    try:
        _reconcile(view, scientific, projection.navigation)

        assert not view.share_axis.isEnabled()
        assert not view.share_axis.isChecked()
    finally:
        _dispose(view)


@pytest.mark.parametrize(
    ("plot_mode", "frame_count", "expect_waterfall"),
    (
        ("Single", 1, False),
        ("Overlay", 4, False),
        ("Overlay", 16, True),
        ("Waterfall", 4, True),
        ("Sum", 4, False),
        ("Average", 4, False),
    ),
)
def test_share_axis_uses_the_active_curve_or_waterfall_in_every_plot_mode(
    plot_mode: str,
    frame_count: int,
    expect_waterfall: bool,
) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    projection = make_shell_projection(
        frame_count=frame_count,
        selected_index=frame_count - 1,
        heavy_indices=(frame_count - 1,),
        plot_mode=plot_mode,
    )
    view = ScientificView()
    scientific = replace(projection.scientific, share_axis=True)
    view.resize(1200, 800)
    view.show()
    try:
        _reconcile(view, scientific, projection.navigation)
        QtTest.QTest.qWait(80)

        assert view.share_axis.isEnabled()
        assert view.share_axis.isChecked()
        assert view._share_link_on
        assert (
            view.bottom_stack.currentWidget() is view.waterfall
        ) is expect_waterfall
        _assert_shared_pixels_align(view, 0.5, 1.5, 2.5)

        _widget, bottom_view = view._active_bottom_plot()
        bottom_view.setXRange(0.75, 1.85, padding=0.0)
        QtTest.QTest.qWait(80)
        _assert_shared_pixels_align(view, 0.8, 1.3, 1.8)
    finally:
        _dispose(view)
