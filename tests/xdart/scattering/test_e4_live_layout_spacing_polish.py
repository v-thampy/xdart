from __future__ import annotations

from dataclasses import astuple
from pathlib import Path

import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.tabs.scattering.controls_projection import project_controls
from xdart.gui.tabs.scattering.scientific_view import (
    ONE_D_PLOT_BOTTOM_MARGIN,
    PLOT_TOOLBAR_INTER_GROUP_GAP,
    SCIENTIFIC_TOOLBAR_EDGE_INSET,
)
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell
from xdart.gui.widgets.controls_panel import FormRow, RangeRow
from xdart.gui.themes import apply_theme
from xdart.gui.themes.spacing import spacing_tokens
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import DirectorySourceSpec

from tests.xdart.scattering.e3_shell_support import make_shell_projection


@pytest.fixture
def qapp() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _close(widget: QtWidgets.QWidget, qapp: QtWidgets.QApplication) -> None:
    widget.close()
    widget.deleteLater()
    qapp.processEvents()


def _center_y_in(
    widget: QtWidgets.QWidget,
    owner: QtWidgets.QWidget,
) -> int:
    return widget.mapTo(owner, widget.rect().center()).y()


def _intent() -> RunIntent:
    return RunIntent(
        source_spec=DirectorySourceSpec(
            Path("/raw/eiger"),
            suffixes=(".h5",),
        ),
        poni_file="/project/detector.poni",
        project_root="/project",
        save_path="/processed",
        bai_1d_args={
            "radial_range": (0.0, 5.0),
            "azimuth_range": (-180.0, 180.0),
        },
        bai_2d_args={
            "radial_range": (0.0, 5.0),
            "azimuth_range": (-180.0, 180.0),
        },
    )


def test_scientific_toolbar_surfaces_are_vertically_centered(
    qapp: QtWidgets.QApplication,
) -> None:
    apply_theme(qapp, "dark", spacing="normal")
    shell = ScatteringWorkspaceShell()
    try:
        shell.apply_state(make_shell_projection())
        shell.resize(1488, 1000)
        shell.show()
        qapp.processEvents()

        top_content = shell.scientific.top_bar.parentWidget()
        top_viewport = top_content.parentWidget()
        top_center = top_viewport.rect().center().y()
        top_margins = top_content.layout().contentsMargins()
        assert top_margins.left() == SCIENTIFIC_TOOLBAR_EDGE_INSET
        assert top_margins.right() == SCIENTIFIC_TOOLBAR_EDGE_INSET
        for widget in (
            shell.scientific.norm,
            shell.scientific.background,
            shell.scientific.title,
            shell.scientific.color_map,
            shell.scientific.log_scale,
        ):
            assert abs(_center_y_in(widget, top_viewport) - top_center) <= 1

        plot_content = shell.scientific.plot_bar.parentWidget()
        plot_viewport = plot_content.parentWidget()
        plot_center = plot_viewport.rect().center().y()
        plot_margins = plot_content.layout().contentsMargins()
        assert plot_margins.left() == SCIENTIFIC_TOOLBAR_EDGE_INSET
        assert plot_margins.right() == SCIENTIFIC_TOOLBAR_EDGE_INSET
        for widget in (
            shell.scientific.plot_axis,
            shell.scientific.slice,
            shell.scientific.slice_center,
            shell.scientific.slice_width,
            shell.scientific.pin,
            shell.scientific.plot_mode,
            shell.scientific.options,
            shell.scientific.clear,
            shell.scientific.share_axis,
            shell.scientific.image_axis,
        ):
            assert abs(_center_y_in(widget, plot_viewport) - plot_center) <= 1

        assert shell.scientific.curve.getPlotItem().layout.getContentsMargins()[
            3
        ] == ONE_D_PLOT_BOTTOM_MARGIN
    finally:
        _close(shell, qapp)


def test_scientific_footer_has_balanced_navigation_and_tunable_gaps(
    qapp: QtWidgets.QApplication,
) -> None:
    apply_theme(qapp, "dark", spacing="normal")
    shell = ScatteringWorkspaceShell()
    try:
        shell.apply_state(make_shell_projection())
        shell.resize(1488, 1000)
        shell.show()
        qapp.processEvents()

        scientific = shell.scientific
        navigation = (
            scientific.previous_frame,
            scientific.frame_selector,
            scientific.next_frame,
        )
        assert len({widget.height() for widget in navigation}) == 1
        assert len({widget.geometry().center().y() for widget in navigation}) == 1
        assert (
            getattr(scientific.frame_selector, "display_alignment", None)
            == QtCore.Qt.AlignmentFlag.AlignCenter
        )
        assert scientific.status.contentsMargins().left() >= 8

        assert (
            PLOT_TOOLBAR_INTER_GROUP_GAP
            >= scientific.pin.sizeHint().width() * 3 // 4
        )
        assert (
            PLOT_TOOLBAR_INTER_GROUP_GAP
            <= scientific.pin.sizeHint().width() * 5 // 4
        )
    finally:
        _close(shell, qapp)


def test_spacing_tiers_use_the_maintainer_requested_interpolation() -> None:
    # The prior Tight and Spacious values become the new endpoints.  The
    # intermediate tiers are exact integer midpoints around unchanged Normal.
    old_tight = (
        3, 9, 2, 7, 1, 3, 2, 5, 4, 8, 3, 6, 3, 10, 3, 5,
        1, 5, 1, 3, 1, 8, 5, 6, 6, 6, 9,
    )
    normal = astuple(spacing_tokens("normal"))
    old_spacious = (
        6, 15, 5, 12, 4, 6, 5, 9, 8, 13, 7, 11, 6, 16, 6, 10,
        3, 9, 3, 6, 3, 12, 11, 12, 11, 11, 15,
    )

    assert astuple(spacing_tokens("extra_tight")) == old_tight
    assert astuple(spacing_tokens("tight")) == tuple(
        (low + high) // 2
        for low, high in zip(old_tight, normal, strict=True)
    )
    assert astuple(spacing_tokens("spacious")) == tuple(
        (low + high) // 2
        for low, high in zip(normal, old_spacious, strict=True)
    )
    assert astuple(spacing_tokens("extra_spacious")) == old_spacious


def test_processing_axis_combos_align_with_range_editors(
    qapp: QtWidgets.QApplication,
) -> None:
    apply_theme(qapp, "dark", spacing="normal")
    shell = ScatteringWorkspaceShell()
    try:
        shell.controls.reconcile(
            project_controls(
                RunIntentStore(_intent()).snapshot(),
                None,
                RunPhase.IDLE,
            )
        )
        shell.resize(1488, 1000)
        shell.show()
        qapp.processEvents()

        axis_rows = {
            row.path[0]: row
            for row in shell.controls.findChildren(FormRow)
            if row.path in {("Int1D", "axis"), ("Int2D", "axis")}
        }
        first_ranges = {
            row._low_path[0]: row
            for row in shell.controls.findChildren(RangeRow)
            if row._low_path[0] in {"Int1D", "Int2D"}
        }
        assert set(axis_rows) == set(first_ranges) == {"Int1D", "Int2D"}
        for root in ("Int1D", "Int2D"):
            axis_left = axis_rows[root].editor.mapTo(
                shell.controls,
                QtCore.QPoint(0, 0),
            ).x()
            range_left = first_ranges[root]._low.mapTo(
                shell.controls,
                QtCore.QPoint(0, 0),
            ).x()
            assert axis_left == range_left
    finally:
        _close(shell, qapp)


def test_default_tools_panel_fits_every_tool_without_inner_scroll(
    qapp: QtWidgets.QApplication,
) -> None:
    apply_theme(qapp, "dark", spacing="normal")
    shell = ScatteringWorkspaceShell()
    try:
        shell.resize(1488, 1000)
        shell.show()
        for tier in (
            "extra_tight",
            "tight",
            "normal",
            "spacious",
            "extra_spacious",
        ):
            apply_theme(qapp, "dark", spacing=tier)
            qapp.processEvents()
            qapp.processEvents()

            tools = shell.tools
            buttons = tools.tool_content.findChildren(QtWidgets.QPushButton)
            assert len(buttons) == len(tools._TOOLS)
            assert tools.tool_scroll.verticalScrollBar().maximum() == 0
            viewport = tools.tool_scroll.viewport()
            for button in buttons:
                center = button.mapTo(viewport, button.rect().center())
                assert viewport.rect().contains(center)
    finally:
        _close(shell, qapp)
