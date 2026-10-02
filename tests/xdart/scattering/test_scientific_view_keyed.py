"""ScientificView's keyed layout (AUD-9 stage 2b G7): display_logic's repeated PanelKey roles, the H11/WS-X2
seam, drawn under Int's own top bar. The RSM page's 2×3 is its first consumer; Int's and Stitch's views never
build it."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyqtgraph")

from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering.scientific_view import ScientificView
from xdart.gui.tabs.scattering.shell_values import (
    AxisProjection, KeyedImageProjection, KeyedPanelsProjection, KeyedTraceProjection, ShellCommandKind,
)
from xrd_tools.session.display_logic import PanelKey, PanelRole


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


_HK, _HL = PanelKey(PanelRole.SLICE_2D, "HK"), PanelKey(PanelRole.SLICE_2D, "HL")
_H, _K = PanelKey(PanelRole.PROJ_1D, "H"), PanelKey(PanelRole.PROJ_1D, "K")


def _state(offset=0.0, *, color_map="Default", log_scale=False, index=1):
    x, y = np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0])

    def image(key):
        return KeyedImageProjection(key, f"{key.instance} slice", np.arange(6.0).reshape(2, 3) + offset,
                                    AxisProjection(x, "H", "", edges=(0.5, 3.5)),
                                    AxisProjection(y, "K", "", edges=(5.0, 25.0)), index, ("0", "1", "2"))

    def trace(key):
        return KeyedTraceProjection(key, f"{key.instance} mean", AxisProjection(x, key.instance, ""),
                                    np.array([1.0, 2.0, 4.0]) + offset, "Mean I")
    return KeyedPanelsProjection(((_HK, _HL), (_H, _K)), (image(_HK), image(_HL), trace(_H), trace(_K)),
                                 title="result", status="facts", color_map=color_map, log_scale=log_scale)


def test_a_keyed_layout_draws_its_panels_under_ints_top_bar_and_gives_ints_layout_back(qapp):
    view = ScientificView()
    try:
        assert view._keyed_area is None  # Int's and Stitch's views never build it
        view.reconcile_keyed(_state(color_map="magma", log_scale=True))
        assert view.keyed_projection is not None and not view._keyed_area.isHidden()
        assert view.vertical_splitter.isHidden() and view.frame_selector.isHidden()
        np.testing.assert_array_equal(view.keyed_panel(_HK).pane.canvas.raw_image, np.arange(6.0).reshape(2, 3).T)
        x, y = view.keyed_panel(_K).curve.getData()
        np.testing.assert_array_equal(y, [1.0, 2.0, 4.0])
        assert view.color_map.currentText() == "magma" and view.log_scale.isChecked()
        assert view.title.text() == "result" and view.status.text() == "facts"
        view._apply_processing_layout("Int 1D")  # an Int mode: the keyed area goes, Int's layout comes back
        assert view._keyed_area.isHidden() and not view.vertical_splitter.isHidden()
        assert not view.frame_selector.isHidden()
    finally:
        view.deleteLater()
        qapp.processEvents()


def test_a_slice_steps_along_its_normal_through_the_views_own_navigation(qapp):
    view = ScientificView()
    commands = []
    view.commandRequested.connect(commands.append)
    try:
        view.reconcile_keyed(_state(index=1))
        panel = view.keyed_panel(_HL)
        assert panel.selector.currentIndex() == 1 and panel.previous.isEnabled() and panel.next.isEnabled()
        panel.next.click()
        panel.selector.setCurrentIndex(0)
        assert [(command.kind, command.value, command.path) for command in commands] == [
            (ShellCommandKind.SET_PANEL_INDEX, 2, ("HL",)), (ShellCommandKind.SET_PANEL_INDEX, 0, ("HL",))]
        commands.clear()
        view.reconcile_keyed(_state(index=2))  # the host's draw moves the selector without echoing a command
        assert view.keyed_panel(_HL).selector.currentIndex() == 2 and not commands
        assert not view.keyed_panel(_HL).next.isEnabled()
    finally:
        view.deleteLater()
        qapp.processEvents()


def test_a_keyed_draw_is_all_or_nothing_and_refuses_roles_it_cannot_draw(qapp, monkeypatch):
    view = ScientificView()
    try:
        view.reconcile_keyed(_state())
        prior = view.keyed_projection
        before = np.array(view.keyed_panel(_HK).pane.canvas.raw_image)
        original = view.keyed_panel(_HL).pane.render
        calls = []

        def failing(*args, **kwargs):
            calls.append(True)
            if len(calls) == 1:
                raise RuntimeError("pane failed")
            return original(*args, **kwargs)
        monkeypatch.setattr(view.keyed_panel(_HL).pane, "render", failing)
        with pytest.raises(RuntimeError, match="pane failed"):
            view.reconcile_keyed(_state(offset=5.0))
        assert view.keyed_projection is prior
        np.testing.assert_array_equal(view.keyed_panel(_HK).pane.canvas.raw_image, before)
        cake = PanelKey(PanelRole.CAKE_2D, "")
        with pytest.raises(TypeError):  # a keyed image panel is a SLICE_2D
            KeyedImageProjection(cake, "", np.zeros((2, 3)), AxisProjection(np.arange(3.0), "x"),
                                 AxisProjection(np.arange(2.0), "y"), 0, ("0",))
    finally:
        view.deleteLater()
        qapp.processEvents()


def test_a_host_pages_tools_sit_in_the_top_bar_and_the_slice_tool_can_be_left_out(qapp):
    """``set_page_tools`` puts a host page's own buttons where Int's Raw pop-up button sits, each emitting
    PAGE_TOOL with its name; ``set_hidden_tools("slice")`` leaves out the χ (c/w) slice and its Pin, also
    through Int's own mode layouts."""
    view = ScientificView()
    commands = []
    view.commandRequested.connect(commands.append)
    try:
        view.set_page_tools(("3d", "3D Viewer", "tip"), ("custom", "Custom Slice", "tip"))
        row = view._top_bar_row
        assert [row.indexOf(view.page_tool(name)) for name in ("3d", "custom")] == [
            row.indexOf(view.raw_popup_button) + 1, row.indexOf(view.raw_popup_button) + 2]
        view.page_tool("custom").click()
        assert [(command.kind, command.value) for command in commands] == [(ShellCommandKind.PAGE_TOOL, "custom")]
        view.set_page_tools(("3d", "3D Viewer", "tip"))  # replaced
        with pytest.raises(KeyError):
            view.page_tool("custom")
        with pytest.raises(TypeError):
            view.set_page_tools(("3d", "", "tip"))
        view.set_hidden_tools("slice")
        view._apply_processing_layout("Int 2D")
        assert all(widget.isHidden() for widget in (view.slice, view.slice_center, view.slice_width, view.pin))
    finally:
        view.deleteLater()
        qapp.processEvents()


def test_the_colour_map_and_log_controls_are_built_once_for_every_intensity_view(qapp):
    """Int's colour-map list and Log toggle come from one shared builder (``color_map_combo``, ``log_toggle``),
    which ``ScientificView``'s top bar and the 3D Viewer's ``VolumeView`` both mount (AUD-9 stage 2b G9; hard
    rule 14: reuse by extraction, never a re-declared list)."""
    from xdart.gui.tabs.scattering.shell_widgets import color_map_choices, color_map_combo, log_toggle
    view = ScientificView()
    try:
        items = tuple(view.color_map.itemText(index) for index in range(view.color_map.count()))
        assert items == color_map_choices() and items[0] == "Default" and "viridis" in items
        assert view.log_scale.isCheckable() and view.log_scale.text() == "Log"
        combo, log = color_map_combo(), log_toggle()
        assert tuple(combo.itemText(index) for index in range(combo.count())) == items
        assert type(combo) is type(view.color_map) and log.isCheckable() and log.text() == "Log"
    finally:
        view.deleteLater()
        qapp.processEvents()


def test_a_plane_with_no_index_to_step_spans_its_row_and_draws_no_navigation(qapp):
    """Custom Slice's layout (AUD-9 stage 2b G9): one plane, ``navigation=False`` (no index to step along a
    normal), alone in its row so it spans the widest row, over its two projections. The RSM 2×3 is unchanged:
    every panel of a row of several takes one column."""
    plane, along_t, along_s = (PanelKey(PanelRole.SLICE_2D, "custom"), PanelKey(PanelRole.PROJ_1D, "t"),
                               PanelKey(PanelRole.PROJ_1D, "s"))
    x, y = np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0])
    image = KeyedImageProjection(plane, "HH-L plane", np.arange(6.0).reshape(2, 3), AxisProjection(x, "HH", ""),
                                 AxisProjection(y, "L", ""), 0, (), navigation=False)
    traces = tuple(KeyedTraceProjection(key, key.instance, AxisProjection(axis, key.instance, ""),
                                        np.ones(axis.size), "Mean I") for key, axis in ((along_t, x), (along_s, y)))
    state = KeyedPanelsProjection(((plane,), (along_t, along_s)), (image, *traces))
    with pytest.raises(TypeError):  # a plane without navigation names no captions
        KeyedImageProjection(plane, "", np.zeros((2, 3)), AxisProjection(x, "HH", ""), AxisProjection(y, "L", ""),
                             0, ("0",), navigation=False)
    view = ScientificView()
    try:
        view.reconcile_keyed(state)
        grid = view._keyed_area.layout()
        panel = view.keyed_panel(plane)
        assert grid.getItemPosition(grid.indexOf(panel.container)) == (0, 0, 1, 2)
        assert all(widget.isHidden() for widget in (panel.previous, panel.selector, panel.next))
        np.testing.assert_array_equal(panel.pane.canvas.raw_image, np.arange(6.0).reshape(2, 3).T)
        view.reconcile_keyed(_state())  # the 2×3-style layout: each panel one column, navigation drawn
        grid = view._keyed_area.layout()
        assert grid.getItemPosition(grid.indexOf(view.keyed_panel(_HK).container)) == (0, 0, 1, 1)
        assert not view.keyed_panel(_HK).selector.isHidden()
    finally:
        view.deleteLater()
        qapp.processEvents()


_KL, _L = PanelKey(PanelRole.SLICE_2D, "KL"), PanelKey(PanelRole.PROJ_1D, "L")
_Q_LABELS = ("Qx (Å⁻¹)", "Qy (Å⁻¹)", "Qz (Å⁻¹)")


def _rsm_state(*, share_axis=False, log_scale=False, labels=("H", "K", "L"), scales=(1.0, 10.0, 100.0)):
    """The RSM page's 2×3: the HK, HL and KL slices, each on its own intensity scale (``scales``), over the H, K
    and L mean projections; ``labels`` name the three coordinates (HKL, or the Q frame's)."""
    values = (np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0, 40.0]), np.array([-1.0, 0.0, 1.0, 2.0, 3.0]))
    edges = ((0.5, 3.5), (5.0, 45.0), (-1.5, 3.5))

    def axis(index):
        return AxisProjection(values[index], labels[index], "", edges=edges[index])

    def image(key, x, y, scale):
        data = (np.arange(values[y].size * values[x].size, dtype=float).reshape(values[y].size, values[x].size)
                + 1.0) * scale
        return KeyedImageProjection(key, key.instance, data, axis(x), axis(y), 0, ("0", "1"))

    traces = tuple(KeyedTraceProjection(key, key.instance, AxisProjection(values[index], labels[index], ""),
                                        np.ones(values[index].size), "Mean I")
                   for key, index in ((_H, 0), (_K, 1), (_L, 2)))
    slices = (image(_HK, 0, 1, scales[0]), image(_HL, 0, 2, scales[1]), image(_KL, 1, 2, scales[2]))
    return KeyedPanelsProjection(((_HK, _HL, _KL), (_H, _K, _L)), (*slices, *traces), share_axis=share_axis,
                                 log_scale=log_scale)


def _own_levels(state):
    """Each slice's own automatic levels and finite range, drawn alone on Int's image pane (Int's rule)."""
    from xdart.gui.tabs.scattering.shell_widgets import ScientificImagePane
    found = []
    for panel in state.panels:
        if type(panel) is KeyedImageProjection:
            pane = ScientificImagePane(lock_aspect=False)
            pane.render(panel.image, x_axis=panel.x_axis, y_axis=panel.y_axis, log_scale=state.log_scale)
            histogram = pane.canvas.histogram
            found.append((tuple(histogram.levels()), (histogram.lo_lim, histogram.hi_lim)))
            pane.deleteLater()
    return found


def _slice_levels(view):
    return [tuple(view.keyed_panel(key).pane.canvas.histogram.levels()) for key in (_HK, _HL, _KL)]


@pytest.mark.parametrize("log_scale", (False, True))
def test_the_intensity_controls_set_one_level_for_every_keyed_slice_and_autoscale_spans_them(qapp, log_scale):
    """G13 unit 1a (decision 1's toolbar, G7's choice 2 undone): Int's intensity controls, on the keyed layout's
    own row, drive the three slices together as one set of levels. Autoscale spans the slices' own automatic levels
    (each Int's rule over its finite values) and the auto bar shows those active levels. Typed or dragged levels
    apply to every slice; Autoscale again brings the span back. With Log the levels are in what each pane draws,
    log10, as Int's viewer's are."""
    state = _rsm_state(log_scale=log_scale)
    own = _own_levels(state)
    span = (min(levels[0] for levels, _range in own), max(levels[1] for levels, _range in own))
    values = (min(found[0] for _levels, found in own), max(found[1] for _levels, found in own))
    assert all(levels != span for levels, _range in own)  # no one slice's levels are the span
    view = ScientificView()
    try:
        view.reconcile_keyed(state)
        controls = view.viewer_intensity
        assert view.viewer_intensity_row.isAncestorOf(controls) and not view.viewer_intensity_row.isHidden()
        assert not controls.isHidden() and controls.autoscale.isChecked() and controls.slider.isEnabled()
        np.testing.assert_allclose(_slice_levels(view), [span] * 3)
        np.testing.assert_allclose(controls.values(), span)
        np.testing.assert_allclose(controls.slider.domain(), span)
        typed = (0.8, 2.5) if log_scale else (30.0, 600.0)
        controls._open_entry()  # typed, as Int's viewer takes them
        low, high = controls._entry_editors()
        low.setText(f"{typed[0]:g}")
        high.setText(f"{typed[1]:g}")
        controls._apply_entry()
        assert not controls.autoscale.isChecked()
        np.testing.assert_allclose(controls.slider.domain(),
                                   (min(values[0], typed[0]), max(values[1], typed[1])))
        np.testing.assert_allclose(_slice_levels(view), [typed] * 3)
        dragged = (1.2, 2.0) if log_scale else (50.0, 900.0)
        controls.slider.setValues(*dragged, emit=True)  # dragged
        np.testing.assert_allclose(_slice_levels(view), [dragged] * 3)
        view.reconcile_keyed(state)  # a draw keeps the levels typed
        np.testing.assert_allclose(_slice_levels(view), [dragged] * 3)
        controls.autoscale.setChecked(True)
        np.testing.assert_allclose(_slice_levels(view), [span] * 3)
        np.testing.assert_allclose(controls.values(), span)
    finally:
        view.deleteLater()
        qapp.processEvents()


def test_log_moves_the_keyed_levels_into_what_the_panes_draw(qapp):
    """G13 unit 1a: Log composes with the keyed levels as with Int's viewer: Autoscale's span follows the panes into
    log10 and back."""
    view = ScientificView()
    try:
        view.reconcile_keyed(_rsm_state())
        linear = _slice_levels(view)
        view.reconcile_keyed(_rsm_state(log_scale=True))
        logged = _rsm_state(log_scale=True)
        own = _own_levels(logged)
        span = (min(levels[0] for levels, _range in own), max(levels[1] for levels, _range in own))
        np.testing.assert_allclose(_slice_levels(view), [span] * 3)
        assert span[1] < 4.0 < linear[0][1]  # log10 of at most 2000
        view.reconcile_keyed(_rsm_state())
        np.testing.assert_allclose(_slice_levels(view), linear)
    finally:
        view.deleteLater()
        qapp.processEvents()


def _ranges(view):
    """Every keyed axis's view range: each slice's x and y, each projection's x."""
    found = {}
    for key in (_HK, _HL, _KL):
        x, y = view.keyed_panel(key).pane.canvas.imageViewBox.viewRange()
        found[f"{key.instance}.x"], found[f"{key.instance}.y"] = tuple(x), tuple(y)
    for key in (_H, _K, _L):
        found[f"{key.instance}"] = tuple(view.keyed_panel(key).plot.getViewBox().viewRange()[0])
    return found


@pytest.mark.parametrize("labels", (("H", "K", "L"), _Q_LABELS))
def test_share_axis_links_the_panels_that_draw_one_coordinate_and_unlinks_them(qapp, labels):
    """G13 unit 1b: Share Axis over the keyed 2×3 links each coordinate across the panels that draw it: H on HK's x,
    HL's x and the H projection; K on HK's y, KL's x and the K projection; L on HL's y, KL's y and the L
    projection. Pan or zoom one, and the others follow along that coordinate only. The coordinate is the axis's
    label and unit, so the Q frame's Qx, Qy and Qz stay apart (Int's canonical axis key takes all three as Q).
    Off, the panels are independent and the projections follow their data again."""
    commands = []
    view = ScientificView()
    view.commandRequested.connect(commands.append)
    links = {"HK.x": ("HL.x", "H"), "HK.y": ("KL.x", "K"), "HL.y": ("KL.y", "L")}
    try:
        view.reconcile_keyed(_rsm_state(labels=labels))
        assert view.share_axis.isEnabled() and not view.share_axis.isChecked()
        assert view.viewer_intensity_row.isAncestorOf(view.share_axis) and not view.share_axis.isHidden()
        view.share_axis.click()  # the host's command; the host draws it back
        assert [(command.kind, command.value) for command in commands] == [(ShellCommandKind.SET_SHARE_AXIS, True)]
        view.reconcile_keyed(_rsm_state(labels=labels, share_axis=True))
        assert view.share_axis.isChecked()
        for axis_name, followers in links.items():  # aligned from the first panel that draws the coordinate
            assert {_ranges(view)[name] for name in (axis_name, *followers)} == {_ranges(view)[axis_name]}
        for moved, (low, high) in (("HK.x", (1.2, 2.4)), ("K", (12.0, 31.0)), ("KL.y", (0.25, 1.75))):
            before = _ranges(view)
            panel = {"HK.x": _HK, "K": _K, "KL.y": _KL}[moved]
            box = (view.keyed_panel(panel).plot.getViewBox() if panel is _K
                   else view.keyed_panel(panel).pane.canvas.imageViewBox)
            (box.setYRange if moved == "KL.y" else box.setXRange)(low, high, padding=0)
            after = _ranges(view)
            linked = next({axis, *followers} for axis, followers in links.items() if moved in {axis, *followers})
            for name, value in after.items():
                if name in linked:
                    np.testing.assert_allclose(value, (low, high), err_msg=name)
                else:
                    assert value == before[name], name
        view.reconcile_keyed(_rsm_state(labels=labels))  # off
        assert not view.share_axis.isChecked()
        before = _ranges(view)
        view.keyed_panel(_HK).pane.canvas.imageViewBox.setXRange(1.5, 2.5, padding=0)
        after = _ranges(view)
        assert after["HL.x"] == before["HL.x"] and after["H"] == before["H"]
        assert view.keyed_panel(_H).plot.getViewBox().autoRangeEnabled()[0]  # follows its data again
    finally:
        view.deleteLater()
        qapp.processEvents()


def test_leaving_the_keyed_layout_gives_share_axis_and_the_intensity_controls_back_to_ints_layout(qapp):
    """G13 unit 1: the keyed layout borrows Int's Share Axis and intensity controls for its own row; an Int mode
    puts Share Axis back in Int's plot bar where it was, and the intensity controls where that mode keeps them."""
    view = ScientificView()
    try:
        home = view.plot_bar.indexOf(view.share_axis)
        view.reconcile_keyed(_rsm_state(share_axis=True))
        assert view.plot_bar.indexOf(view.share_axis) == -1
        view._apply_processing_layout("Int 2D")
        assert view.plot_bar.indexOf(view.share_axis) == home and view.viewer_intensity_row.isHidden()
        assert view.viewer_intensity.isHidden()
        view.reconcile_keyed(_rsm_state())
        view._apply_processing_layout("2D Viewer")
        assert view.viewer_intensity_row_layout.indexOf(view.viewer_intensity) >= 0
        assert not view.viewer_intensity_row.isHidden() and view.plot_bar.indexOf(view.share_axis) == home
    finally:
        view.deleteLater()
        qapp.processEvents()
