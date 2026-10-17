from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering.scientific_plot_options import (
    INTENSITY_SCALE_CHOICES,
    WATERFALL_Y_AXIS_CHOICES,
    waterfall_should_be_active,
)
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.scientific_view import ScientificView
from xdart.gui.tabs.scattering.shell_projection import (
    ScientificPreferences,
    build_scientific_projection,
)
from xdart.gui.tabs.scattering.shell_values import (
    AxisProjection,
    FrameNavigationProjection,
    ScientificPlotOptions,
    ShellCommand,
    ShellCommandKind,
)

from tests.xdart.scattering.e3_shell_support import make_shell_projection


@pytest.fixture
def qapp() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _reconcile(
    view: ScientificView,
    scientific,
    navigation: FrameNavigationProjection,
) -> None:
    view.reconcile(
        scientific,
        navigation,
        completed=len(navigation.frames),
        total=len(navigation.frames),
        detail="Ready",
    )


def _preference_owner(
    preferences: ScientificPreferences | None = None,
    **fields: object,
) -> SimpleNamespace:
    return SimpleNamespace(
        _preferences=(
            ScientificPreferences()
            if preferences is None
            else preferences
        ),
        _background_owner=SimpleNamespace(projection=lambda: None),
        **fields,
    )


def test_plot_options_are_frozen_and_projected_by_identity() -> None:
    options = ScientificPlotOptions(
        waterfall_y_axis="Time (s)",
        waterfall_start=2,
        waterfall_stop=9,
        waterfall_step=3,
        overlay_offset=12.5,
        show_legend=False,
        intensity_scale="Sqrt",
    )
    preferences = ScientificPreferences(plot_options=options)

    projection = build_scientific_projection(
        (),
        FrameNavigationProjection(),
        frozenset(),
        preferences,
        "",
    )

    assert projection.plot_options is options
    with pytest.raises(FrozenInstanceError):
        options.waterfall_step = 4


@pytest.mark.parametrize(
    ("path", "value", "field", "expected"),
    (
        (("waterfall", "y_axis"), "Time (s)", "waterfall_y_axis", "Time (s)"),
        (("waterfall", "start"), 2, "waterfall_start", 2),
        (("waterfall", "stop"), 9, "waterfall_stop", 9),
        (("waterfall", "step"), 3, "waterfall_step", 3),
        (("overlay", "offset"), 12, "overlay_offset", 12.0),
        (("other", "legend"), False, "show_legend", False),
        (("other", "intensity_scale"), "Sqrt", "intensity_scale", "Sqrt"),
    ),
)
def test_page_owns_each_typed_plot_option_edit(
    path: tuple[str, ...],
    value: str | int | bool,
    field: str,
    expected: object,
) -> None:
    owner = _preference_owner()
    accepted = ScatteringWorkspace._edit_scientific_preference(
        owner,
        ShellCommand(ShellCommandKind.SET_PLOT_OPTION, value, path),
    )

    assert accepted
    assert getattr(owner._preferences.plot_options, field) == expected


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("waterfall", "y_axis"), ""),
        (("waterfall", "start"), 0),
        (("waterfall", "start"), True),
        (("waterfall", "stop"), -1),
        (("waterfall", "step"), 0),
        (("overlay", "offset"), float("nan")),
        (("other", "legend"), 1),
        (("other", "intensity_scale"), "Cubic"),
        (("unknown",), 1),
    ),
)
def test_page_rejects_invalid_plot_option_edits_without_mutation(
    path: tuple[str, ...],
    value: object,
) -> None:
    owner = _preference_owner()
    before = owner._preferences
    command = ShellCommand(ShellCommandKind.SET_PLOT_OPTION, value, path)

    assert not ScatteringWorkspace._edit_scientific_preference(owner, command)
    assert owner._preferences is before


@pytest.mark.parametrize(
    ("image_axis", "expected_plot_axis"),
    (
        ("Q-Chi", "Q"),
        ("2Th-Chi", "2theta"),
        ("qip_qoop", "q_ip"),
        ("q_chi", "Q"),
        ("exit_angles", "exit_angle"),
    ),
)
def test_share_axis_intent_repoints_plot_axis_to_rendered_cake_identity(
    image_axis: str,
    expected_plot_axis: str,
) -> None:
    owner = _preference_owner(
        ScientificPreferences(
            image_axis=image_axis,
            plot_axis=("2theta" if expected_plot_axis == "Q" else "Q"),
        )
    )

    assert ScatteringWorkspace._edit_scientific_preference(
        owner,
        ShellCommand(ShellCommandKind.SET_SHARE_AXIS, True),
    )
    assert owner._preferences.share_axis
    assert owner._preferences.plot_axis == expected_plot_axis


def test_share_axis_intent_is_refused_for_nonsharing_qz_qxy_image() -> None:
    owner = _preference_owner(
        ScientificPreferences(
            image_axis="Qz-Qxy",
            plot_axis="Q",
        )
    )

    assert ScatteringWorkspace._edit_scientific_preference(
        owner,
        ShellCommand(ShellCommandKind.SET_SHARE_AXIS, True),
    )
    assert not owner._preferences.share_axis
    assert owner._preferences.plot_axis == "Q"


def test_share_axis_intent_uses_rendered_gi_cake_not_stale_preference() -> None:
    owner = _preference_owner(
        ScientificPreferences(
            image_axis="Q-Chi",
            plot_axis="Q",
        ),
        _rendered_image_axis="qip_qoop",
    )

    assert ScatteringWorkspace._edit_scientific_preference(
        owner,
        ShellCommand(ShellCommandKind.SET_SHARE_AXIS, True),
    )
    assert owner._preferences.share_axis
    assert owner._preferences.plot_axis == "q_ip"


def test_live_image_axis_flip_keeps_shared_plot_axis_converged() -> None:
    owner = _preference_owner(
        ScientificPreferences(
            image_axis="Q-Chi",
            plot_axis="Q",
            share_axis=True,
        )
    )

    assert ScatteringWorkspace._edit_scientific_preference(
        owner,
        ShellCommand(ShellCommandKind.SET_IMAGE_AXIS, "2Th-Chi"),
    )
    assert owner._preferences.image_axis == "2Th-Chi"
    assert owner._preferences.plot_axis == "2theta"


def test_share_axis_in_options_mirrors_the_existing_view_command(qapp):
    view = ScientificView()
    state = make_shell_projection(plot_mode="Single")
    commands = []
    view.commandRequested.connect(commands.append)
    try:
        _reconcile(view, replace(state.scientific, share_axis=True), state.navigation)
        assert view.share_axis.isChecked() and not view.share_axis.isHidden()
        view.options.click()
        dialog = view.plot_options_dialog
        assert dialog.share_axis.isChecked() and dialog.share_axis.isEnabled()
        assert dialog.share_axis.toolTip()
        commands.clear()
        dialog.share_axis.click()
        assert view.share_axis.isChecked()  # a draft until Okay
        dialog.cancel_button.click()
        assert commands == [] and view.share_axis.isChecked()
        view.options.click()
        assert dialog.share_axis.isChecked()
        commands.clear()
        dialog.share_axis.click()
        dialog.accept_button.click()
        assert commands == [ShellCommand(ShellCommandKind.SET_SHARE_AXIS, False)]
        assert not view.share_axis.isChecked()
        view.share_axis.click()
        assert dialog.share_axis.isChecked()  # the existing main control still mirrors back
        _reconcile(view, replace(state.scientific, processing_mode="Int 1D", share_axis=False), state.navigation)
        view.options.click()
        assert not dialog.share_axis.isEnabled()
        assert "coordinate" in dialog.share_axis.toolTip()
    finally:
        view.close()


def test_plot_options_dialog_has_production_sections_and_emits_typed_edits(
    qapp: QtWidgets.QApplication,
) -> None:
    view = ScientificView()
    state = make_shell_projection(plot_mode="Overlay")
    commands: list[ShellCommand] = []
    view.commandRequested.connect(commands.append)
    try:
        _reconcile(view, state.scientific, state.navigation)
        view.options.click()

        dialog = view.plot_options_dialog
        assert dialog.isVisible()
        assert commands == [
            ShellCommand(ShellCommandKind.SHOW_WATERFALL_OPTIONS)
        ]
        assert [
            dialog.waterfall_y_axis.itemText(index)
            for index in range(dialog.waterfall_y_axis.count())
        ] == list(WATERFALL_Y_AXIS_CHOICES)
        assert [
            dialog.intensity_scale.itemText(index)
            for index in range(dialog.intensity_scale.count())
        ] == list(INTENSITY_SCALE_CHOICES)
        assert dialog.waterfall_start.minimum() == 1
        assert dialog.waterfall_stop.minimum() == 0
        assert dialog.waterfall_stop.specialValueText() == "End"
        assert dialog.waterfall_step.minimum() == 1

        commands.clear()
        dialog.waterfall_y_axis.setCurrentText("Time (s)")
        dialog.waterfall_start.setValue(2)
        dialog.waterfall_stop.setValue(9)
        dialog.waterfall_step.setValue(3)
        dialog.overlay_offset.setValue(12.5)
        dialog.show_legend.setChecked(False)
        dialog.intensity_scale.setCurrentText("Sqrt")
        dialog.accept_button.click()

        assert commands == [
            ShellCommand(
                ShellCommandKind.SET_PLOT_OPTION,
                "Time (s)",
                ("waterfall", "y_axis"),
            ),
            ShellCommand(
                ShellCommandKind.SET_PLOT_OPTION,
                2,
                ("waterfall", "start"),
            ),
            ShellCommand(
                ShellCommandKind.SET_PLOT_OPTION,
                9,
                ("waterfall", "stop"),
            ),
            ShellCommand(
                ShellCommandKind.SET_PLOT_OPTION,
                3,
                ("waterfall", "step"),
            ),
            ShellCommand(
                ShellCommandKind.SET_PLOT_OPTION,
                12.5,
                ("overlay", "offset"),
            ),
            ShellCommand(
                ShellCommandKind.SET_PLOT_OPTION,
                False,
                ("other", "legend"),
            ),
            ShellCommand(
                ShellCommandKind.SET_PLOT_OPTION,
                "Sqrt",
                ("other", "intensity_scale"),
            ),
        ]

        commands.clear()
        view.options.click()
        commands.clear()
        dialog.waterfall_start.setValue(8)
        dialog.cancel_button.click()
        assert commands == []
        assert dialog.waterfall_start.value() == 1
    finally:
        view.close()


@pytest.mark.parametrize(
    ("plot_mode", "trace_count", "was_active", "expected"),
    (
        ("Waterfall", 3, False, False),
        ("Waterfall", 4, False, True),
        ("Waterfall", 3, True, False),
        ("Overlay", 15, False, False),
        ("Overlay", 16, False, True),
        ("Overlay", 8, True, True),
        ("Overlay", 7, True, False),
        ("Single", 16, False, True),
        ("Single", 8, True, True),
        ("Single", 7, True, False),
        ("Average", 100, True, False),
        ("Sum", 100, True, False),
    ),
)
def test_waterfall_activation_matches_production_thresholds(
    plot_mode: str,
    trace_count: int,
    was_active: bool,
    expected: bool,
) -> None:
    assert waterfall_should_be_active(
        plot_mode,
        trace_count,
        was_active=was_active,
    ) is expected


def test_view_retains_overlay_waterfall_through_eight_and_drops_at_seven(
    qapp: QtWidgets.QApplication,
) -> None:
    view = ScientificView()
    state = make_shell_projection(
        frame_count=16,
        selected_index=15,
        heavy_indices=(0, 15),
        plot_mode="Overlay",
    )
    frames = state.navigation.frames
    try:
        _reconcile(view, state.scientific, state.navigation)
        assert view._bottom_waterfall_active
        assert view.bottom_stack.currentWidget() is view.waterfall
        assert view.waterfall.image.image.shape == (64, 16)

        eight = FrameNavigationProjection(frames, frames[7], frames[:8])
        _reconcile(view, state.scientific, eight)
        assert view._bottom_waterfall_active
        assert view.bottom_stack.currentWidget() is view.waterfall
        assert view.waterfall.image.image.shape == (64, 8)

        seven = FrameNavigationProjection(frames, frames[6], frames[:7])
        _reconcile(view, state.scientific, seven)
        assert not view._bottom_waterfall_active
        assert view.bottom_stack.currentWidget() is view.curve

        _reconcile(
            view,
            replace(state.scientific, plot_mode="Average"),
            state.navigation,
        )
        assert not view._bottom_waterfall_active
        assert view.bottom_stack.currentWidget() is view.curve
    finally:
        view.close()


def test_overlay_waterfall_retains_651_but_paints_bounded_terminal_extent(
    qapp: QtWidgets.QApplication,
) -> None:
    view = ScientificView()
    state = make_shell_projection(
        frame_count=651,
        selected_index=650,
        heavy_indices=(0, 650),
        plot_mode="Overlay",
    )
    try:
        _reconcile(view, state.scientific, state.navigation)
        assert view._bottom_waterfall_active
        assert view.bottom_stack.currentWidget() is view.waterfall
        assert view.trace_history_keys == state.navigation.selected
        assert len(view.trace_history_keys) == 651
        assert view.waterfall.image.image.shape == (64, 256)
        assert len(view._waterfall_y_values) == 256
        assert view._waterfall_y_values[0] == 1.0
        assert view._waterfall_y_values[-1] == 651.0
    finally:
        view.close()


def test_single_waterfall_accepts_numerically_equivalent_live_axes(
    qapp: QtWidgets.QApplication,
) -> None:
    view = ScientificView()
    state = make_shell_projection(
        frame_count=16,
        selected_index=15,
        heavy_indices=(0, 15),
        plot_mode="Single",
    )
    frames = state.navigation.frames
    traces = tuple(
        replace(
            trace,
            axis=AxisProjection(
                np.asarray(
                    trace.axis.values + index * np.finfo(float).eps,
                    dtype=float,
                ),
                trace.axis.label,
                trace.axis.unit,
            ),
        )
        for index, trace in enumerate(state.scientific.traces)
    )
    navigation = FrameNavigationProjection(frames, frames[-1], frames)
    try:
        _reconcile(
            view,
            replace(state.scientific, traces=traces),
            navigation,
        )

        assert view._bottom_waterfall_active
        assert view.bottom_stack.currentWidget() is view.waterfall
        assert view.waterfall.image.image.shape == (64, 16)
    finally:
        view.close()


def test_explicit_waterfall_mounts_an_image_only_at_four_traces(
    qapp: QtWidgets.QApplication,
) -> None:
    view = ScientificView()
    state = make_shell_projection(
        frame_count=4,
        selected_index=3,
        heavy_indices=(0, 3),
        plot_mode="Waterfall",
    )
    try:
        three = FrameNavigationProjection(
            state.navigation.frames,
            state.navigation.frames[2],
            state.navigation.frames[:3],
        )
        _reconcile(view, state.scientific, three)
        assert not view._bottom_waterfall_active
        assert view.bottom_stack.currentWidget() is view.curve

        _reconcile(view, state.scientific, state.navigation)
        assert view._bottom_waterfall_active
        assert view.bottom_stack.currentWidget() is view.waterfall
        assert view.waterfall.image.image.shape == (64, 4)
        np.testing.assert_allclose(
            view._waterfall_y_values,
            np.arange(1.0, 5.0),
        )
    finally:
        view.close()


def test_waterfall_time_axis_is_relative_to_the_full_selected_history(
    qapp: QtWidgets.QApplication,
) -> None:
    view = ScientificView()
    state = make_shell_projection(
        frame_count=5,
        selected_index=4,
        heavy_indices=(0, 4),
        plot_mode="Waterfall",
    )
    traces = tuple(
        replace(trace, epoch=100.0 + index * 60.0)
        for index, trace in enumerate(state.scientific.traces)
    )
    options = ScientificPlotOptions(
        waterfall_y_axis="Time (minutes)",
        waterfall_start=2,
        waterfall_step=2,
    )
    scientific = replace(
        state.scientific,
        traces=traces,
        plot_options=options,
    )
    try:
        _reconcile(view, scientific, state.navigation)

        assert view.bottom_stack.currentWidget() is view.waterfall
        assert view._waterfall_y_label == "Time (minutes)"
        np.testing.assert_allclose(view._waterfall_y_values, (1.0, 3.0))
        assert view.waterfall.image.image.shape == (64, 2)
    finally:
        view.close()


def test_reconciled_plot_options_slice_scale_and_hide_line_legend(
    qapp: QtWidgets.QApplication,
) -> None:
    view = ScientificView()
    state = make_shell_projection(
        frame_count=4,
        selected_index=3,
        heavy_indices=(0, 3),
        plot_mode="Overlay",
    )
    options = ScientificPlotOptions(
        waterfall_start=2,
        waterfall_stop=4,
        waterfall_step=2,
        overlay_offset=0.0,
        show_legend=False,
        intensity_scale="Sqrt",
    )
    scientific = replace(state.scientific, plot_options=options)
    try:
        _reconcile(view, scientific, state.navigation)

        rendered = view.curve.listDataItems()
        assert len(rendered) == 2
        np.testing.assert_allclose(
            rendered[0].yData,
            np.sign(scientific.traces[1].intensity)
            * np.sqrt(np.abs(scientific.traces[1].intensity)),
        )
        np.testing.assert_allclose(
            rendered[1].yData,
            np.sign(scientific.traces[3].intensity)
            * np.sqrt(np.abs(scientific.traces[3].intensity)),
        )
        assert not view.legend.isVisible()
    finally:
        view.close()


def test_left_drag_zooms_the_1d_plot_into_a_box_and_shift_drag_pans(qapp):
    """Live: a left-drag panned the 1-D plot (Int 1-D and Stitch) while it drew a zoom box on the
    2-D views, as the retired Static Scan page's 1-D plot also did. The 1-D curve now uses the
    2-D views' RectViewBox: a real left-drag zooms to the box (a pan keeps the width), and
    Shift+left-drag pans (a trackpad has no middle button)."""
    from pyqtgraph.Qt import QtCore, QtGui
    from xdart.gui.gui_utils import RectViewBox

    view = ScientificView()
    view.resize(1000, 800)
    view.show()
    try:
        box = view.curve.getPlotItem().getViewBox()
        assert isinstance(box, RectViewBox) and box.state["mouseMode"] == box.RectMode
        x = np.linspace(0., 10., 101)
        view.curve.getPlotItem().plot(x, x ** 2)
        viewport = view.curve.viewport()

        def at(fx, fy, item=box):
            """A point in viewport pixels: fractions of the plot area, or with ``item=None`` a scene
            coordinate on the axis whose fraction is not given (fx or fy > 1 is a scene position)."""
            area = box.sceneBoundingRect()
            if item is None:
                sx = fx if fx > 1 else area.left() + fx * area.width()
                sy = fy if fy > 1 else area.top() + fy * area.height()
                return view.curve.mapFromScene(QtCore.QPointF(sx, sy))
            corner = item.boundingRect()
            local = QtCore.QPointF(corner.left() + fx * corner.width(), corner.top() + fy * corner.height())
            return view.curve.mapFromScene(item.mapToScene(local))

        def drag(start, end, modifiers=QtCore.Qt.KeyboardModifier.NoModifier, item=box, both=False):
            """A real left-button press, ten moves and a release; the x range after it."""
            box.setRange(xRange=(0., 10.), yRange=(0., 100.), padding=0)
            qapp.processEvents()

            def send(kind, point, button, buttons):
                QtWidgets.QApplication.sendEvent(viewport, QtGui.QMouseEvent(
                    kind, QtCore.QPointF(point), QtCore.QPointF(viewport.mapToGlobal(point)),
                    button, buttons, modifiers))
            left, none = QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.NoButton
            start, end = at(*start, item=item), at(*end, item=item)
            send(QtCore.QEvent.Type.MouseButtonPress, start, left, left)
            for step in range(1, 11):
                send(QtCore.QEvent.Type.MouseMove, start + (end - start) * (step / 10), none, left)
                qapp.processEvents()
            send(QtCore.QEvent.Type.MouseButtonRelease, end, left, none)
            qapp.processEvents()
            return box.viewRange() if both else box.viewRange()[0]

        a, b = drag((.3, .3), (.6, .7))
        assert 0. < a < b < 10. and (b - a) < 5., (a, b)  # zoomed to the box
        a, b = drag((.6, .5), (.3, .5), QtCore.Qt.KeyboardModifier.ShiftModifier)
        assert (b - a) == pytest.approx(10., rel=1e-6) and a > 1., (a, b)  # panned right by ~3
        # a drag on an axis's tick labels pans that axis only, as the plain view did (Codex R20-F1);
        # the axes draw the grid across the plot, so their labels are the strip outside the plot area
        plot = view.curve.getPlotItem()
        area = box.sceneBoundingRect()
        bottom, left = plot.getAxis("bottom").sceneBoundingRect(), plot.getAxis("left").sceneBoundingRect()
        labels_y, labels_x = (area.bottom() + bottom.bottom()) / 2, (left.left() + area.left()) / 2
        assert labels_y > area.bottom() and labels_x < area.left()
        (a, b), y = drag((.3, labels_y), (.6, labels_y), item=None, both=True)
        assert (b - a) == pytest.approx(10., rel=1e-6) and a < -1. and y == pytest.approx([0., 100.]), (a, b, y)
        x, (c, d) = drag((labels_x, .3), (labels_x, .6), item=None, both=True)
        assert (d - c) == pytest.approx(100., rel=1e-6) and c > 10. and x == pytest.approx([0., 10.]), (x, c, d)
    finally:
        view.close()
        view.deleteLater()


def test_a_hosts_plot_option_is_a_choice_the_dialog_builds() -> None:
    """Codex's Stitch/Int reuse audit R3: a host's own plot option (Stitch's Diagnostic) is a choice the
    plot-options dialog builds, above its buttons, from the host's choices and value. A host hands in no
    widget of its own (``add_row`` is gone)."""
    from xdart.gui.tabs.scattering.scientific_plot_options import PlotOptionsDialog
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dialog = PlotOptionsDialog()
    try:
        assert not hasattr(dialog, "add_row")
        editor = dialog.add_choice("Diagnostic", ("Intensity", "Coverage", "Normalization"), "Coverage")
        assert isinstance(editor, QtWidgets.QComboBox) and dialog.isAncestorOf(editor)
        assert [editor.itemText(i) for i in range(editor.count())] == ["Intensity", "Coverage", "Normalization"]
        assert editor.currentText() == "Coverage" and editor.objectName() == "e4PlotOptionsHostChoice"
        layout = dialog.layout()
        assert layout.itemAt(layout.count() - 2).widget().isAncestorOf(editor)  # just above the buttons
        with pytest.raises(ValueError):
            dialog.add_choice("Diagnostic", ("Intensity",), "Coverage")
    finally:
        dialog.close()
        app.processEvents()
