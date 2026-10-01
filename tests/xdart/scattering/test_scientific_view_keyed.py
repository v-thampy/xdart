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
