from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from xrd_tools.core import Axis, FrameView, TwoDKind
from xdart.gui.tabs.scattering.display_values import (
    StandardDisplayPayload,
    display_payload_is_valid,
)
from xdart.gui.tabs.scattering.shell_projection import (
    ScientificPreferences,
    build_scientific_projection,
)
from xdart.gui.tabs.scattering.shell_values import (
    AxisProjection,
    FrameNavigationProjection,
    HeavyProjection,
    PinnedTraceProjection,
    SlicePin,
)
from xdart.gui.tabs.scattering.workspace_shell import (
    ScatteringWorkspaceShell,
)

from tests.xdart.scattering.e3_shell_support import make_shell_projection


@pytest.fixture
def qapp() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_e3_ui2_renders_raw_cake_axes_and_all_retained_overlay_rows(
    qapp: QtWidgets.QApplication,
) -> None:
    shell = ScatteringWorkspaceShell()
    state = make_shell_projection()
    try:
        shell.apply_state(state)

        assert shell.scientific.raw.image.image.shape == (64, 48)
        assert shell.scientific.cake.image.image.shape == (32, 24)
        raw_levels = tuple(np.nanpercentile(np.arange(1.0, 3073.0), (2, 98)))
        cake_levels = tuple(
            np.nanpercentile(np.arange(1.0, 769.0), (0.5, 99.5))
        )
        assert shell.scientific.raw.color_scale.levels() == pytest.approx(
            raw_levels
        )
        assert shell.scientific.cake.color_scale.levels() == pytest.approx(
            cake_levels
        )
        np.testing.assert_allclose(
            shell.scientific.raw.image.getLevels(), raw_levels
        )
        np.testing.assert_allclose(
            shell.scientific.cake.image.getLevels(), cake_levels
        )
        assert (
            shell.scientific.cake.plot.getAxis("bottom").labelText
            == "Q"
        )
        assert (
            shell.scientific.cake.plot.getAxis("left").labelText
                == "χ"
        )
        assert len(shell.scientific.curve.listDataItems()) == 5
        current = state.navigation.current
        assert current is not None
        footer = tuple(
            frame
            for frame in state.navigation.frames
            if frame.artifact == current.artifact
        )
        assert shell.browser.frame_model.frames is state.navigation.frames
        assert tuple(
            shell.scientific.frame_selector.itemData(index)
            for index in range(shell.scientific.frame_selector.count())
        ) == footer
        assert [
            shell.scientific.frame_selector.itemText(index)
            for index in range(shell.scientific.frame_selector.count())
        ] == [
            str(position) for position in range(1, len(footer) + 1)
        ]
    finally:
        shell.close()
        shell.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def test_slice_projection_draws_exact_cake_extent_boundaries(
    qapp: QtWidgets.QApplication,
) -> None:
    shell = ScatteringWorkspaceShell()
    state = make_shell_projection(
        frame_count=1,
        heavy_indices=(0,),
        plot_mode="Single",
    )
    state = replace(
        state,
        scientific=replace(
            state.scientific,
            slice_enabled=True,
            slice_center=12.0,
            slice_width=3.0,
        ),
    )
    try:
        shell.apply_state(state)
        qapp.processEvents()

        lines = shell.scientific._slice_extent_lines
        assert len(lines) == 2
        assert tuple(line.value() for line in lines) == pytest.approx(
            (9.0, 15.0)
        )
        assert tuple(float(line.angle) for line in lines) == (0.0, 0.0)

        shell.apply_state(
            replace(
                state,
                revision=state.revision + 1,
                scientific=replace(
                    state.scientific,
                    slice_enabled=False,
                ),
            )
        )
        assert shell.scientific._slice_extent_lines == ()
    finally:
        shell.close()
        shell.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


@pytest.mark.parametrize(
    ("measurement_mode", "plot_axis", "cake_x", "cake_y", "angle"),
    (
        (
            "Standard",
            "chi",
            AxisProjection(np.linspace(0.0, 4.0, 32), "Q", "Å⁻¹"),
            AxisProjection(np.linspace(-90.0, 90.0, 24), "χ", "°"),
            90.0,
        ),
        (
            "GI",
            "q_ip",
            AxisProjection(np.linspace(-2.0, 2.0, 32), "Qip", "Å⁻¹"),
            AxisProjection(np.linspace(0.0, 4.0, 24), "Qoop", "Å⁻¹"),
            0.0,
        ),
    ),
)
def test_slice_extent_orientation_matches_standard_and_gi_projection(
    qapp: QtWidgets.QApplication,
    measurement_mode: str,
    plot_axis: str,
    cake_x: AxisProjection,
    cake_y: AxisProjection,
    angle: float,
) -> None:
    shell = ScatteringWorkspaceShell()
    state = make_shell_projection(
        frame_count=1,
        heavy_indices=(0,),
        plot_mode="Single",
    )
    assert state.scientific.heavy is not None
    scientific = replace(
        state.scientific,
        measurement_mode=measurement_mode,
        plot_axis=plot_axis,
        slice_enabled=True,
        slice_center=1.0,
        slice_width=0.25,
        heavy=replace(
            state.scientific.heavy,
            cake_x=cake_x,
            cake_y=cake_y,
        ),
    )
    try:
        shell.apply_state(replace(state, scientific=scientific))

        lines = shell.scientific._slice_extent_lines
        assert len(lines) == 2
        assert tuple(line.value() for line in lines) == pytest.approx(
            (0.75, 1.25)
        )
        assert tuple(float(line.angle) for line in lines) == (angle, angle)
    finally:
        shell.close()
        shell.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def test_slice_projection_change_refits_and_rearms_one_d_autoscale(
    qapp: QtWidgets.QApplication,
) -> None:
    shell = ScatteringWorkspaceShell()
    state = make_shell_projection(
        frame_count=1,
        heavy_indices=(0,),
        plot_mode="Single",
    )
    state = replace(
        state,
        scientific=replace(
            state.scientific,
            slice_enabled=True,
            slice_center=0.0,
            slice_width=2.0,
        ),
    )
    try:
        shell.apply_state(state)
        qapp.processEvents()
        view = shell.scientific.curve.getPlotItem().getViewBox()
        view.setYRange(5000.0, 6000.0, padding=0.0)
        assert view.autoRangeEnabled()[1] is False

        shell.apply_state(
            replace(
                state,
                revision=state.revision + 1,
                scientific=replace(
                    state.scientific,
                    slice_center=10.0,
                ),
            )
        )
        qapp.processEvents()

        y_range = view.viewRange()[1]
        assert view.autoRangeEnabled()[1] is not False
        assert y_range[0] < 10.0 and y_range[1] < 100.0
    finally:
        shell.close()
        shell.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def test_retained_hydration_keeps_coherent_slice_marker_and_range(
    qapp: QtWidgets.QApplication,
) -> None:
    shell = ScatteringWorkspaceShell()
    state = make_shell_projection(
        frame_count=1,
        heavy_indices=(0,),
        plot_mode="Overlay",
    )
    scientific = replace(
        state.scientific,
        slice_enabled=True,
        slice_center=0.0,
        slice_width=2.0,
    )
    try:
        shell.apply_state(replace(state, scientific=scientific))
        view = shell.scientific.curve.getPlotItem().getViewBox()
        view.setYRange(5000.0, 6000.0, padding=0.0)
        prior_lines = shell.scientific._slice_extent_lines
        prior_values = tuple(line.value() for line in prior_lines)
        prior_contract = shell.scientific._rendered_slice_contract

        shell.apply_state(replace(
            state,
            revision=state.revision + 1,
            scientific=replace(
                scientific,
                heavy=None,
                traces=(),
                slice_center=20.0,
                retain_display=True,
            ),
        ))

        assert shell.scientific._slice_extent_lines == prior_lines
        assert tuple(line.value() for line in prior_lines) == prior_values
        assert shell.scientific._rendered_slice_contract == prior_contract
        assert view.viewRange()[1] == pytest.approx((5000.0, 6000.0))
        assert not shell.scientific.pin.isEnabled()
    finally:
        shell.close()
        shell.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def test_slice_autorange_preserves_effective_shared_x_range(
    qapp: QtWidgets.QApplication,
) -> None:
    shell = ScatteringWorkspaceShell()
    state = make_shell_projection(
        frame_count=1,
        heavy_indices=(0,),
        plot_mode="Single",
    )
    scientific = replace(
        state.scientific,
        image_axis="Q-Chi",
        plot_axis="Q",
        share_axis=True,
        slice_enabled=True,
        slice_center=0.0,
        slice_width=2.0,
    )
    try:
        shell.apply_state(replace(state, scientific=scientific))
        qapp.processEvents()
        assert shell.scientific._share_link_on
        view = shell.scientific.curve.getPlotItem().getViewBox()
        view.setXRange(0.75, 1.25, padding=0.0)
        qapp.processEvents()
        before = tuple(view.viewRange()[0])

        shell.apply_state(replace(
            state,
            revision=state.revision + 1,
            scientific=replace(scientific, slice_center=10.0),
        ))
        qapp.processEvents()

        assert tuple(view.viewRange()[0]) == pytest.approx(before)
        assert view.autoRangeEnabled()[0] is False
        assert view.autoRangeEnabled()[1] is not False
    finally:
        shell.close()
        shell.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def test_overlay_pin_retains_old_slice_and_absorbs_matching_live_cut(
    qapp: QtWidgets.QApplication,
) -> None:
    shell = ScatteringWorkspaceShell()
    state = make_shell_projection(
        frame_count=1,
        heavy_indices=(0,),
        plot_mode="Overlay",
    )
    frame = state.navigation.current
    assert frame is not None
    q = np.linspace(0.1, 1.0, 4)
    chi = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
    cake = np.repeat(np.arange(1.0, 6.0)[:, None], q.size, axis=1)
    payload = StandardDisplayPayload(
        0,
        frame,
        "slice pin",
        FrameView(
            frame.local_frame_label,
            axis_1d=Axis("Q", "q_A^-1", values=q),
            intensity_1d=np.full(q.shape, 99.0),
            axis_2d_x=Axis("Q", "q_A^-1", values=q),
            axis_2d_y=Axis("chi", "chi_deg", values=chi),
            intensity_2d=cake,
            two_d_kind=TwoDKind.Q_CHI,
        ),
    )
    old_pin = SlicePin(frame, "Q", -5.0, 0.1)
    old_preferences = ScientificPreferences(
        plot_mode="Overlay",
        plot_axis="Q",
        slice_enabled=True,
        slice_center=-5.0,
        slice_width=0.1,
        slice_pins=(old_pin,),
    )
    old_projection = build_scientific_projection(
        (payload,),
        state.navigation,
        frozenset({frame}),
        old_preferences,
        "",
    )
    assert old_projection.traces == ()
    assert len(old_projection.pinned_traces) == 1
    np.testing.assert_array_equal(
        old_projection.pinned_traces[0].trace.intensity,
        np.full(q.shape, 2.0),
    )

    moved_preferences = replace(
        old_preferences,
        slice_center=5.0,
    )
    moved_projection = build_scientific_projection(
        (payload,),
        state.navigation,
        frozenset({frame}),
        moved_preferences,
        "",
    )
    assert len(moved_projection.traces) == 1
    assert len(moved_projection.pinned_traces) == 1
    np.testing.assert_array_equal(
        moved_projection.traces[0].intensity,
        np.full(q.shape, 4.0),
    )

    new_pin = SlicePin(frame, "Q", 5.0, 0.1)
    absorbed_projection = build_scientific_projection(
        (payload,),
        state.navigation,
        frozenset({frame}),
        replace(moved_preferences, slice_pins=(old_pin, new_pin)),
        "",
    )
    assert absorbed_projection.traces == ()
    assert tuple(
        pinned.pin.projection_id
        for pinned in absorbed_projection.pinned_traces
    ) == (old_pin.projection_id, new_pin.projection_id)

    try:
        shell.apply_state(replace(state, scientific=old_projection))
        assert len(shell.scientific.curve.listDataItems()) == 1
        shell.apply_state(replace(
            state,
            revision=state.revision + 1,
            scientific=moved_projection,
        ))
        assert len(shell.scientific.curve.listDataItems()) == 2
        shell.apply_state(replace(
            state,
            revision=state.revision + 2,
            scientific=absorbed_projection,
        ))
        assert len(shell.scientific.curve.listDataItems()) == 2
        assert len(set(shell.scientific._rendered_trace_keys)) == 2
        assert shell.scientific.trace_history_keys == (frame,)
    finally:
        shell.close()
        shell.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def test_retained_pin_keeps_live_suffix_projection_incremental(
    qapp: QtWidgets.QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shell = ScatteringWorkspaceShell()
    full = make_shell_projection(
        frame_count=3,
        heavy_indices=(0,),
        plot_mode="Overlay",
    )
    frames = full.navigation.frames
    base = full.scientific.traces[0].intensity
    pinned_trace = replace(full.scientific.traces[0], intensity=base)
    second_trace = replace(full.scientific.traces[1], intensity=base)
    third_trace = replace(full.scientific.traces[2], intensity=base)
    pin = SlicePin(frames[0], "Q", 0.0, 1.0)
    initial = replace(
        full,
        navigation=FrameNavigationProjection(
            frames[:2],
            frames[1],
            frames[:2],
        ),
        scientific=replace(
            full.scientific,
            traces=(second_trace,),
            slice_enabled=True,
            slice_center=0.0,
            slice_width=1.0,
            slice_pins=(pin,),
            pinned_traces=(PinnedTraceProjection(pin, pinned_trace),),
        ),
    )
    try:
        shell.apply_state(initial)
        assert len(shell.scientific.curve.listDataItems()) == 2

        calls = {"clear": 0, "plot": 0}
        original_clear = shell.scientific.curve.clear
        original_plot = shell.scientific.curve.plot

        def counted_clear(*args, **kwargs):
            calls["clear"] += 1
            return original_clear(*args, **kwargs)

        def counted_plot(*args, **kwargs):
            calls["plot"] += 1
            return original_plot(*args, **kwargs)

        monkeypatch.setattr(shell.scientific.curve, "clear", counted_clear)
        monkeypatch.setattr(shell.scientific.curve, "plot", counted_plot)
        shell.apply_state(replace(
            full,
            revision=full.revision + 1,
            scientific=replace(
                initial.scientific,
                traces=(third_trace,),
                pinned_traces=(),
            ),
        ))

        assert calls == {"clear": 0, "plot": 1}
        assert len(shell.scientific.curve.listDataItems()) == 3
        assert shell.scientific.trace_history_keys == frames
    finally:
        shell.close()
        shell.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


@pytest.mark.parametrize("plot_mode", ("Overlay", "Waterfall"))
def test_slice_pin_preserves_bounded_waterfall_rendering(
    qapp: QtWidgets.QApplication,
    plot_mode: str,
) -> None:
    shell = ScatteringWorkspaceShell()
    state = make_shell_projection(
        frame_count=17,
        heavy_indices=(0,),
        plot_mode=plot_mode,
    )
    frame = state.navigation.frames[0]
    pin = SlicePin(frame, "Q", 0.0, 1.0)
    scientific = replace(
        state.scientific,
        traces=state.scientific.traces[1:],
        slice_enabled=True,
        slice_pins=(pin,),
        pinned_traces=(
            PinnedTraceProjection(pin, state.scientific.traces[0]),
        ),
    )
    try:
        shell.apply_state(replace(state, scientific=scientific))

        assert shell.scientific.bottom_stack.currentWidget() \
            is shell.scientific.waterfall
        assert shell.scientific.waterfall.image.image.shape == (64, 17)
        assert len(shell.scientific._rendered_trace_keys) == 17
    finally:
        shell.close()
        shell.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


@pytest.mark.parametrize(
    ("mode", "line_count", "waterfall_rows"),
    [
        ("Single", 1, 0),
        ("Overlay", 5, 0),
        ("Waterfall", 0, 5),
    ],
)
def test_e3_ui2_one_d_modes_preserve_expected_history_scope(
    qapp: QtWidgets.QApplication,
    mode: str,
    line_count: int,
    waterfall_rows: int,
) -> None:
    shell = ScatteringWorkspaceShell()
    state = make_shell_projection(plot_mode=mode)
    try:
        shell.apply_state(state)
        assert len(shell.scientific.curve.listDataItems()) == line_count
        assert len(shell.scientific._rendered_trace_keys) == (
            waterfall_rows or line_count
        )
        if waterfall_rows:
            assert (
                shell.scientific.bottom_stack.currentWidget()
                is shell.scientific.waterfall
            )
            assert shell.scientific.waterfall.image.image.shape == (
                64,
                waterfall_rows,
            )
        else:
            assert (
                shell.scientific.bottom_stack.currentWidget()
                is shell.scientific.curve
            )
    finally:
        shell.close()
        shell.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def test_e3_ui2_restores_control_inventory_and_ranges(
    qapp: QtWidgets.QApplication,
) -> None:
    shell = ScatteringWorkspaceShell()
    shell.resize(1920, 1080)
    shell.show()
    try:
        shell.apply_state(make_shell_projection())
        qapp.processEvents()
        labels = {
            widget.text()
            for widget in shell.findChildren(QtWidgets.QLabel)
            if widget.text()
        }
        buttons = {
            widget.text()
            for widget in shell.findChildren(QtWidgets.QAbstractButton)
            if widget.text()
        }
        assert {"PROJECT", "EXPERIMENT", "SOURCE", "PROCESSING"} <= labels
        assert {
            "Poni",
            "Mask File",
            "Grazing",
            "Source",
            "Threshold",
        } <= labels | buttons
        assert {
            "File",
            "Config",
            "Help",
            "Show All",
            "Metadata",
            "Auto Last",
            "Peak Fitting",
            "Phase Fitting",
            "Plot Metadata",
            "Run",
        } <= {
            text.removeprefix("▶ ")
            .removeprefix("∧ ")
            .removeprefix("≈ ")
            .removeprefix("▤ ")
            for text in buttons
        }
        options = shell.scientific.plot_options_dialog
        assert options.waterfall_y_axis.currentText() == "Frame #"
        assert options.waterfall_start.value() == 1
        assert options.waterfall_stop.value() == 0
        assert options.waterfall_step.value() == 1
        assert options.overlay_offset.value() == pytest.approx(5.0)
        assert options.show_legend.isChecked()
        assert options.intensity_scale.currentText() == "Linear"
    finally:
        shell.close()
        shell.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def test_e3_ui2_view_objects_do_not_cache_projection_ndarrays(
    qapp: QtWidgets.QApplication,
) -> None:
    shell = ScatteringWorkspaceShell()
    try:
        shell.apply_state(make_shell_projection())
        for owner in (shell, shell.browser, shell.scientific):
            assert not any(
                type(value) is np.ndarray for value in vars(owner).values()
            )
    finally:
        shell.close()
        shell.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def test_e3_ui2_scientific_images_transpose_once_and_keep_axis_geometry(
    qapp: QtWidgets.QApplication,
) -> None:
    shell = ScatteringWorkspaceShell()
    state = make_shell_projection(
        frame_count=1,
        heavy_indices=(0,),
        plot_mode="Single",
    )
    frame = state.navigation.current
    assert frame is not None
    raw_yx = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, np.nan, 7.0, 4294967295.0],
        ]
    )
    cake_yx = np.zeros((3, 4))
    cake_yx[2, 3] = 17.0
    x_axis = AxisProjection(
        np.array([-4.0, -1.0, 2.0, 5.0]),
        "Q",
        "q_A^-1",
    )
    y_axis = AxisProjection(
        np.array([-6.0, 0.0, 9.0]),
        "chi",
        "chi_deg",
    )
    scientific = replace(
        state.scientific,
        heavy=HeavyProjection(
            frame,
            raw_yx,
            cake_yx,
            x_axis,
            y_axis,
        ),
    )
    try:
        shell.apply_state(replace(state, scientific=scientific))

        raw_item = shell.scientific.raw.image
        cake_item = shell.scientific.cake.image
        np.testing.assert_allclose(
            raw_item.image,
            raw_yx.T[:, ::-1],
            equal_nan=True,
        )
        np.testing.assert_array_equal(cake_item.image, cake_yx.T)
        assert raw_item.axisOrder == "col-major"
        assert cake_item.axisOrder == "col-major"
        raw_levels = raw_item.getLevels()
        assert all(np.isfinite(raw_levels))
        assert raw_levels[1] < 4294967295.0
        assert (
            shell.scientific.raw.color_scale.levels()[1]
            < 4294967295.0
        )
        rect = cake_item.mapRectToParent(QtCore.QRectF(0, 0, 4, 3))
        marker = cake_item.mapToParent(QtCore.QPointF(3.5, 2.5))
        raw_range = shell.scientific.raw.plot.viewRange()
        cake_range = shell.scientific.cake.plot.viewRange()
        plot = shell.scientific.cake.plot
        assert rect == QtCore.QRectF(-4.0, -6.0, 9.0, 15.0)
        assert marker.x() > 0 and marker.y() > 0
        assert raw_range[0][0] <= 0.0 and raw_range[0][1] >= 4.0
        assert raw_range[1][0] <= 0.0 and raw_range[1][1] >= 2.0
        assert cake_range[0][0] <= -4.0 and cake_range[0][1] >= 5.0
        assert cake_range[1][0] <= -6.0 and cake_range[1][1] >= 9.0
        assert (
            shell.scientific.raw.plot.getViewBox().state["aspectLocked"]
            is not False
        )
        assert (
            shell.scientific.cake.plot.getViewBox().state["aspectLocked"]
            is False
        )
        assert plot.getAxis("bottom").labelText == "Q"
        assert plot.getAxis("bottom").labelUnits == "Å⁻¹"
        assert plot.getAxis("left").labelText == "χ"
        assert plot.getAxis("left").labelUnits == "°"
        assert plot.getViewBox().state["yInverted"] is False
        raw_bottom = shell.scientific.raw.plot.getAxis("bottom")
        raw_left = shell.scientific.raw.plot.getAxis("left")
        assert raw_bottom.labelText == "x (Pixels)"
        assert raw_bottom.labelUnits == ""
        assert raw_left.labelText == "y (Pixels)"
        assert raw_left.labelUnits == ""
        assert raw_bottom.autoSIPrefix is False
        assert raw_left.autoSIPrefix is False
    finally:
        shell.close()
        shell.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def test_e3_ui2_rapid_distinct_frames_do_not_share_percentile_levels(
    qapp: QtWidgets.QApplication,
) -> None:
    shell = ScatteringWorkspaceShell()
    state = make_shell_projection(
        frame_count=2,
        heavy_indices=(0, 1),
        plot_mode="Single",
    )
    first = state.navigation.frames[0]
    second = replace(first)
    assert second == first
    assert second is not first
    state = replace(
        state,
        navigation=replace(
            state.navigation,
            frames=(first, second),
            current=first,
            selected=(first,),
        ),
    )
    uniform = np.linspace(0.0, 100.0, 10_000).reshape(100, 100)
    # Keep the widget cache's sampled min/max identical across both frames.
    # The frame identity, rather than an incidental range difference, must
    # invalidate the percentile result.
    uniform[0, 4] = 100.0
    sparse = np.zeros((100, 100))
    sparse[0, 4] = 100.0
    replacement = np.zeros((100, 100))
    replacement[50:, :] = 100.0
    try:
        shell.apply_state(
            replace(
                state,
                scientific=replace(
                    state.scientific,
                    heavy=HeavyProjection(first, uniform, None),
                ),
            )
        )
        first_levels = shell.scientific.raw.image.getLevels()

        shell.apply_state(
            replace(
                state,
                revision=state.revision + 1,
                navigation=replace(
                    state.navigation,
                    current=second,
                    selected=(second,),
                ),
                scientific=replace(
                    state.scientific,
                    heavy=HeavyProjection(second, sparse, None),
                    title="scan-a:2",
                ),
            )
        )
        second_levels = shell.scientific.raw.image.getLevels()

        assert first_levels != pytest.approx(second_levels)
        assert second_levels == pytest.approx(
            tuple(np.nanpercentile(sparse, (2.0, 98.0)))
        )

        shell.apply_state(
            replace(
                state,
                revision=state.revision + 2,
                navigation=replace(
                    state.navigation,
                    current=second,
                    selected=(second,),
                ),
                scientific=replace(
                    state.scientific,
                    heavy=HeavyProjection(second, replacement, None),
                    title="scan-a:2 replaced",
                ),
            )
        )
        replacement_levels = shell.scientific.raw.image.getLevels()
        assert replacement_levels != pytest.approx(second_levels)
        assert replacement_levels == pytest.approx(
            tuple(np.nanpercentile(replacement, (2.0, 98.0)))
        )
    finally:
        shell.close()
        shell.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def test_e3_ui2_image_adapter_prewarms_default_colormap_at_construction(
    qapp: QtWidgets.QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    resolve = pg.colormap.getFromMatplotlib

    def observed(name: str):
        calls.append(name)
        return resolve(name)

    monkeypatch.setattr(pg.colormap, "getFromMatplotlib", observed)
    from xdart.gui.tabs.scattering.shell_widgets import ScientificImagePane

    pane = ScientificImagePane(lock_aspect=True)
    try:
        assert calls == ["viridis"]
    finally:
        pane.close()
        pane.deleteLater()
        qapp.processEvents()


def test_plot_mode_reconcile_keeps_identical_detector_and_cake_paint(
    qapp: QtWidgets.QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shell = ScatteringWorkspaceShell()
    state = make_shell_projection(plot_mode="Overlay")
    raw_updates = []
    cake_updates = []
    raw_set_image = shell.scientific.raw.canvas.setImage
    cake_set_image = shell.scientific.cake.canvas.setImage

    def update_raw(*args, **kwargs):
        raw_updates.append(True)
        return raw_set_image(*args, **kwargs)

    def update_cake(*args, **kwargs):
        cake_updates.append(True)
        return cake_set_image(*args, **kwargs)

    monkeypatch.setattr(shell.scientific.raw.canvas, "setImage", update_raw)
    monkeypatch.setattr(shell.scientific.cake.canvas, "setImage", update_cake)
    try:
        shell.apply_state(state)
        assert raw_updates == [True]
        assert cake_updates == [True]

        shell.apply_state(replace(
            state,
            revision=state.revision + 1,
            scientific=replace(state.scientific, plot_mode="Waterfall"),
        ))

        assert raw_updates == [True]
        assert cake_updates == [True]
        assert shell.scientific.bottom_stack.currentWidget() is (
            shell.scientific.waterfall
        )
    finally:
        shell.close()
        shell.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def test_mutable_image_render_never_seeds_identity_reuse_cache(
    qapp: QtWidgets.QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xdart.gui.tabs.scattering.shell_widgets import ScientificImagePane

    pane = ScientificImagePane(lock_aspect=True)
    source = np.arange(16.0).reshape(4, 4)
    updates = []
    set_image = pane.canvas.setImage

    def update(*args, **kwargs):
        updates.append(np.array(args[0], copy=True))
        return set_image(*args, **kwargs)

    monkeypatch.setattr(pane.canvas, "setImage", update)
    try:
        pane.render(source)
        source[0, 0] = 999.0
        source.flags.writeable = False
        pane.render(source)

        assert len(updates) == 2
        assert updates[0][0, -1] != updates[1][0, -1]
        assert pane.render_matches(source)
    finally:
        pane.close()
        pane.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def test_failed_image_render_retires_contract_before_canvas_mutation(
    qapp: QtWidgets.QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xdart.gui.tabs.scattering.shell_widgets import ScientificImagePane

    pane = ScientificImagePane(lock_aspect=True)
    first = np.arange(16.0).reshape(4, 4)
    second = np.arange(15.0).reshape(3, 5) + 100.0
    first.flags.writeable = False
    second.flags.writeable = False
    pane.render(first)
    set_image = pane.canvas.setImage
    set_range = pane.canvas.imageViewBox.setRange
    updates = []

    def update(*args, **kwargs):
        updates.append(np.array(args[0], copy=True))
        return set_image(*args, **kwargs)

    def fail_range(*_args, **_kwargs):
        raise RuntimeError("injected range failure")

    monkeypatch.setattr(pane.canvas, "setImage", update)
    monkeypatch.setattr(pane.canvas.imageViewBox, "setRange", fail_range)
    try:
        with pytest.raises(RuntimeError, match="injected range failure"):
            pane.render(second)
        assert pane._render_contract is None
        assert len(updates) == 1

        monkeypatch.setattr(pane.canvas.imageViewBox, "setRange", set_range)
        pane.render(first)

        assert len(updates) == 2
        assert pane.render_matches(first)
        np.testing.assert_array_equal(
            pane.image.image,
            first.T[:, ::-1],
        )
    finally:
        pane.close()
        pane.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def test_distinct_immutable_images_keep_zoom_for_unchanged_geometry(
    qapp: QtWidgets.QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """New pixels need not reset an operator's view of the same geometry."""
    from xdart.gui.tabs.scattering.shell_widgets import ScientificImagePane

    pane = ScientificImagePane(lock_aspect=False)
    first = np.arange(12.0).reshape(3, 4)
    second = first + 100.0
    for source in (first, second):
        source.flags.writeable = False
    x_values = np.array((-4.0, -1.0, 2.0, 5.0))
    y_values = np.array((-6.0, 0.0, 9.0))
    same_extent_x = np.array((-4.0, 0.5, 3.0, 5.0))
    for values in (x_values, y_values, same_extent_x):
        values.flags.writeable = False
    x_axis = AxisProjection(x_values, "Q", "q_A^-1")
    y_axis = AxisProjection(y_values, "chi", "chi_deg")
    same_extent_axis = AxisProjection(same_extent_x, "Q", "q_A^-1")
    calls: list[tuple[str, object]] = []
    set_range = pane.canvas.imageViewBox.setRange
    set_label = pane.plot.setLabel

    def observed_range(*args, **kwargs):
        calls.append(("range", args[0] if args else kwargs.get("rect")))
        return set_range(*args, **kwargs)

    def observed_label(*args, **kwargs):
        calls.append(("label", args[0] if args else kwargs.get("axis")))
        return set_label(*args, **kwargs)

    monkeypatch.setattr(pane.canvas.imageViewBox, "setRange", observed_range)
    monkeypatch.setattr(pane.plot, "setLabel", observed_label)
    try:
        pane.render(first, x_axis=x_axis, y_axis=y_axis)
        pane.canvas.imageViewBox.setRange(
            QtCore.QRectF(-2.0, -3.0, 3.0, 6.0), padding=0.0,
        )
        retained = pane.canvas.imageViewBox.targetRect()
        calls.clear()

        pane.render(second, x_axis=same_extent_axis, y_axis=y_axis)

        assert pane.canvas.imageViewBox.targetRect() == retained
        assert calls == []

        explicit = QtCore.QRectF(-1.0, -2.0, 2.0, 4.0)
        pane.render(second, x_axis=same_extent_axis, y_axis=y_axis,
                    view_range=explicit)
        assert pane.canvas.imageViewBox.targetRect() == explicit
        assert [kind for kind, _value in calls].count("range") == 1
        assert [kind for kind, _value in calls].count("label") == 0

        calls.clear()
        changed_units = AxisProjection(same_extent_x, "Q", "2th_deg")
        pane.render(second + 1.0, x_axis=changed_units, y_axis=y_axis)

        assert [kind for kind, _value in calls].count("range") == 1
        assert [kind for kind, _value in calls].count("label") == 2
    finally:
        pane.close()
        pane.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def test_viewer_2d_same_source_reconcile_reuses_image_until_exact_scrub(
    qapp: QtWidgets.QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xdart.gui.tabs.scattering.scientific_view import ScientificView

    shell = ScatteringWorkspaceShell()
    state = make_shell_projection(frame_count=1, heavy_indices=(0,))
    frame = state.navigation.current
    assert frame is not None and state.scientific.heavy is not None
    raw = state.scientific.heavy.raw
    assert raw is not None and not raw.flags.writeable
    heavy = HeavyProjection(
        frame,
        raw,
        None,
        detector_shape=raw.shape,
        detector_source="full",
    )
    viewer = replace(
        state,
        scientific=replace(
            state.scientific,
            processing_mode="2D Viewer",
            traces=(),
            heavy=heavy,
        ),
    )
    clears = []
    updates = []
    clear_viewer = ScientificView.clear_viewer_2d
    set_image = shell.scientific.raw.canvas.setImage

    def clear(*args, **kwargs):
        clears.append(True)
        return clear_viewer(*args, **kwargs)

    def update(*args, **kwargs):
        updates.append(True)
        return set_image(*args, **kwargs)

    monkeypatch.setattr(ScientificView, "clear_viewer_2d", clear)
    monkeypatch.setattr(shell.scientific.raw.canvas, "setImage", update)
    try:
        shell.apply_state(viewer)
        assert clears == [True]
        assert updates == [True]

        shell.apply_state(replace(viewer, revision=viewer.revision + 1))
        assert clears == [True]
        assert updates == [True]

        replacement = np.array(raw, copy=True)
        replacement.flags.writeable = False
        shell.apply_state(replace(
            viewer,
            revision=viewer.revision + 2,
            scientific=replace(
                viewer.scientific,
                heavy=replace(heavy, raw=replacement),
            ),
        ))
        assert clears == [True]
        assert updates == [True, True]

        shell.apply_state(replace(
            viewer,
            revision=viewer.revision + 3,
            scientific=replace(
                viewer.scientific,
                heavy=replace(
                    heavy,
                    raw=replacement,
                    detector_source="thumbnail",
                ),
            ),
        ))
        assert clears == [True, True]
        assert updates == [True, True, True]

        shell.apply_state(replace(
            viewer,
            revision=viewer.revision + 4,
            scientific=replace(viewer.scientific, heavy=None),
        ))
        assert clears == [True, True, True]
        assert shell.scientific._viewer_2d_known_empty
    finally:
        shell.close()
        shell.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def test_waterfall_to_curve_switch_waits_for_replacement_curve(
    qapp: QtWidgets.QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shell = ScatteringWorkspaceShell()
    state = make_shell_projection(plot_mode="Waterfall")
    try:
        shell.apply_state(state)
        view = shell.scientific
        assert view.bottom_stack.currentWidget() is view.waterfall
        active_during_plot = []
        clear_calls = []
        clear_curve = view.curve.clear
        plot_curve = view.curve.plot

        def clear_replacement_curve():
            clear_calls.append(view.bottom_stack.currentWidget())
            return clear_curve()

        def plot_replacement_curve(*args, **kwargs):
            active_during_plot.append(view.bottom_stack.currentWidget())
            return plot_curve(*args, **kwargs)

        monkeypatch.setattr(view.curve, "clear", clear_replacement_curve)
        monkeypatch.setattr(view.curve, "plot", plot_replacement_curve)
        shell.apply_state(replace(
            state,
            revision=state.revision + 1,
            scientific=replace(state.scientific, plot_mode="Overlay"),
        ))

        assert clear_calls == []
        assert active_during_plot
        assert all(active is view.waterfall for active in active_during_plot)
        assert view.bottom_stack.currentWidget() is view.curve
        assert len(view.curve.listDataItems()) == len(state.scientific.traces)
    finally:
        shell.close()
        shell.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def test_e3_ui2_processed_cake_keeps_valid_65535_contrast(
    qapp: QtWidgets.QApplication,
) -> None:
    shell = ScatteringWorkspaceShell()
    state = make_shell_projection(
        frame_count=1,
        heavy_indices=(0,),
        plot_mode="Single",
    )
    frame = state.navigation.current
    assert frame is not None
    cake = np.zeros((20, 20), dtype=float)
    cake[10, 10] = 65535.0
    x_axis = AxisProjection(np.arange(20.0), "Q", "q_A^-1")
    y_axis = AxisProjection(np.arange(20.0), "chi", "chi_deg")
    try:
        shell.apply_state(
            replace(
                state,
                scientific=replace(
                    state.scientific,
                    heavy=HeavyProjection(
                        frame,
                        None,
                        cake,
                        x_axis,
                        y_axis,
                    ),
                ),
            )
        )

        assert shell.scientific.cake.image.getLevels() == pytest.approx(
            (0.0, 65535.0)
        )
        assert shell.scientific.cake.color_scale.levels() == pytest.approx(
            (0.0, 65535.0)
        )
    finally:
        shell.close()
        shell.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def test_e3_ui2_gi_trace_and_cake_use_canonical_axis_presentation(
    qapp: QtWidgets.QApplication,
) -> None:
    shell = ScatteringWorkspaceShell()
    state = make_shell_projection(
        frame_count=1,
        heavy_indices=(0,),
        plot_mode="Single",
    )
    frame = state.navigation.current
    assert frame is not None
    trace = state.scientific.traces[0]
    cake = np.arange(12.0).reshape(3, 4)
    cake_x = AxisProjection(
        np.arange(4.0),
        "Q_ip",
        "qip_A^-1",
    )
    cake_y = AxisProjection(
        np.arange(3.0),
        "Q_oop",
        "qoop_A^-1",
    )
    trace_axis = AxisProjection(
        trace.axis.values,
        "Q_total",
        "qtot_A^-1",
    )
    try:
        shell.apply_state(
            replace(
                state,
                scientific=replace(
                    state.scientific,
                    heavy=HeavyProjection(
                        frame,
                        None,
                        cake,
                        cake_x,
                        cake_y,
                    ),
                    traces=(replace(trace, axis=trace_axis),),
                ),
            )
        )

        trace_plot = shell.scientific.curve.getPlotItem()
        cake_plot = shell.scientific.cake.plot
        assert trace_plot.getAxis("bottom").labelText == "Q<sub>total</sub>"
        assert trace_plot.getAxis("bottom").labelUnits == "Å⁻¹"
        assert cake_plot.getAxis("bottom").labelText == "Q<sub>ip</sub>"
        assert cake_plot.getAxis("bottom").labelUnits == "Å⁻¹"
        assert cake_plot.getAxis("left").labelText == "Q<sub>oop</sub>"
        assert cake_plot.getAxis("left").labelUnits == "Å⁻¹"
    finally:
        shell.close()
        shell.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def test_e3_ui2_one_d_axis_follows_trace_and_clears_with_exact_absence(
    qapp: QtWidgets.QApplication,
) -> None:
    shell = ScatteringWorkspaceShell()
    state = make_shell_projection(
        frame_count=1,
        heavy_indices=(0,),
        plot_mode="Single",
    )
    try:
        shell.apply_state(state)
        axis = shell.scientific.curve.getPlotItem().getAxis("bottom")
        assert axis.labelText == "Q"
        assert axis.labelUnits == "Å⁻¹"

        shell.apply_state(
            replace(
                state,
                revision=state.revision + 1,
                scientific=replace(
                    state.scientific,
                    traces=(),
                    title="no 1-D",
                    retain_display=False,
                ),
            )
        )
        assert shell.scientific.curve.listDataItems() == []
        assert axis.labelText == ""
        assert axis.labelUnits in ("", None)
    finally:
        shell.close()
        shell.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def test_e3_ui2_exact_heavy_wins_over_retain_for_complete_presentation(
    qapp: QtWidgets.QApplication,
) -> None:
    shell = ScatteringWorkspaceShell()
    state = make_shell_projection(
        frame_count=2,
        heavy_indices=(0,),
        plot_mode="Single",
    )
    qualified = state.navigation.frames[1]
    expected_trace = state.scientific.traces[1]
    try:
        shell.apply_state(state)
        shell.apply_state(
            replace(
                state,
                revision=state.revision + 1,
                navigation=replace(
                    state.navigation,
                    current=qualified,
                    selected=(qualified,),
                ),
                scientific=replace(
                    state.scientific,
                    heavy=HeavyProjection(qualified, None, None),
                    title="qualified scan-a:2",
                    retain_display=True,
                ),
            )
        )

        assert shell.scientific.title.text() == "qualified scan-a:2"
        assert shell.scientific.raw.image.image is None
        assert shell.scientific.cake.image.image is None
        rendered = shell.scientific.curve.listDataItems()
        assert len(rendered) == 1
        np.testing.assert_array_equal(rendered[0].xData, expected_trace.axis.values)
        np.testing.assert_array_equal(rendered[0].yData, expected_trace.intensity)
    finally:
        shell.close()
        shell.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


@pytest.mark.parametrize(
    ("label", "unit", "expected_label", "expected_unit"),
    [
        ("Q", "q_A^-1", "Q", "Å⁻¹"),
        ("2theta", "2th_deg", "2θ", "°"),
        ("chi", "chi_deg", "χ", "°"),
    ],
)
def test_e3_ui2_one_d_axis_uses_shared_scientific_presentation_vocabulary(
    qapp: QtWidgets.QApplication,
    label: str,
    unit: str,
    expected_label: str,
    expected_unit: str,
) -> None:
    shell = ScatteringWorkspaceShell()
    state = make_shell_projection(
        frame_count=1,
        heavy_indices=(0,),
        plot_mode="Single",
    )
    trace = state.scientific.traces[0]
    raw_axis = AxisProjection(trace.axis.values, label, unit)
    try:
        shell.apply_state(
            replace(
                state,
                scientific=replace(
                    state.scientific,
                    traces=(replace(trace, axis=raw_axis),),
                ),
            )
        )
        axis = shell.scientific.curve.getPlotItem().getAxis("bottom")
        assert axis.labelText == expected_label
        assert axis.labelUnits == expected_unit
    finally:
        shell.close()
        shell.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def test_e3_ui2_status_fallback_and_non_string_rejection_are_exact(
    qapp: QtWidgets.QApplication,
) -> None:
    shell = ScatteringWorkspaceShell()
    state = make_shell_projection(
        frame_count=1,
        heavy_indices=(0,),
        plot_mode="Single",
    )
    frame = state.navigation.current
    assert frame is not None
    payload = StandardDisplayPayload(
        1,
        frame,
        "finished",
        FrameView(
            frame.local_frame_label,
            raw=np.ones((2, 3)),
        ),
        status="finished",
    )
    scientific = build_scientific_projection(
        (payload,),
        state.navigation,
        frozenset({frame}),
        ScientificPreferences(),
        "",
    )
    try:
        shell.apply_state(replace(state, scientific=scientific))
        assert shell.scientific.status.text() == "finished"
        assert shell.scientific.title.text() == "finished"

        malformed = replace(payload, status=object())
        assert (
            display_payload_is_valid(
                malformed,
                frame.run_identity,
                frame,
                payload.selection_generation,
            )
            is False
        )
        rejected = build_scientific_projection(
            (malformed,),
            state.navigation,
            frozenset({frame}),
            ScientificPreferences(),
            "",
        )
        assert rejected.heavy is None
        assert rejected.traces == ()
        assert rejected.retain_display is True
        # Projection and residency are separate bounded reads.  If a live
        # publication lands between them, the current frame can appear
        # resident before its qualified payload is in this projection.  That
        # transient must keep the last coherent three-panel presentation.
        raw_before = np.array(shell.scientific.raw.image.image, copy=True)
        shell.apply_state(replace(
            state,
            revision=state.revision + 1,
            scientific=rejected,
        ))
        assert shell.scientific.title.text() == "finished"
        np.testing.assert_array_equal(
            shell.scientific.raw.image.image,
            raw_before,
        )
    finally:
        shell.close()
        shell.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
