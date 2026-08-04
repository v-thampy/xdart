from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest
from pyqtgraph.Qt import QtCore, QtWidgets
from shiboken6 import isValid

from xdart.gui.tabs.scattering.workspace_shell import (
    ScatteringWorkspaceShell,
)
from xdart.gui.tabs.static_scan.ui.controls_panel_v2 import (
    FormRow,
    PillRow,
    RangeRow,
)

from tests.xdart.scattering.e3_shell_support import make_shell_projection


_SOURCE_DIR = (
    Path(__file__).parents[3]
    / "src"
    / "xdart"
    / "gui"
    / "tabs"
    / "scattering"
)
_VISUAL_FILES = (
    "browser_model.py",
    "browser_view.py",
    "scientific_view.py",
    "shell_values.py",
    "shell_widgets.py",
    "tools_view.py",
    "workspace_shell.py",
)


@pytest.fixture
def qapp() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.mark.parametrize(
    ("width", "height"),
    [(1440, 900), (1920, 1080), (1024, 900)],
)
def test_e3_ui4_viewport_structure_is_exact(
    qapp: QtWidgets.QApplication,
    width: int,
    height: int,
) -> None:
    shell = ScatteringWorkspaceShell()
    shell.apply_state(make_shell_projection())
    shell.resize(width, height)
    shell.show()
    try:
        qapp.processEvents()
        geometries = [
            shell.splitter.widget(index).geometry()
            for index in range(3)
        ]
        assert shell.size() == QtCore.QSize(width, height)
        assert [item.y() for item in geometries] == [0, 0, 0]
        assert len({item.height() for item in geometries}) == 1
        assert geometries[0].right() < geometries[1].left()
        assert geometries[1].right() < geometries[2].left()
        assert geometries[0].width() >= 255
        # LV-UI-7 replacement oracle: the controls column must receive at
        # least the panel-derived floor, and the panel must fit its viewport
        # with ZERO horizontal overflow — clipping is a failure even when a
        # scrollbar could reach it.
        controls_floor = shell.control_scroll.minimumSizeHint().width()
        assert controls_floor >= shell.controls.minimumSizeHint().width()
        assert geometries[2].width() >= controls_floor
        assert shell.control_scroll.horizontalScrollBar().maximum() == 0
        viewport = shell.control_scroll.viewport()
        rows = [
            row for row in shell.controls.findChildren(RangeRow)
            if row._toggle is not None and row.isVisibleTo(shell.controls)
        ]
        assert rows, "populated shell must expose range rows"
        for row in rows:
            btn = row._toggle[1]
            right_edge = btn.mapTo(
                viewport, btn.rect().topRight()
            ).x()
            assert right_edge <= viewport.width(), (
                f"rightmost control clipped: {row._display_label!r} "
                f"{right_edge} > {viewport.width()}"
            )

        rows = shell.scientific.vertical_splitter.sizes()
        assert abs(rows[0] - rows[1]) <= 1
        for pane in (shell.scientific.raw, shell.scientific.cake):
            assert pane.color_scale.isVisible()
            assert pane.layout().count() == 1
            assert pane.layout().itemAt(0).widget() is pane.canvas

        top = (
            shell.scientific.norm,
            shell.scientific.background,
            shell.scientific.title,
            shell.scientific.color_map,
            shell.scientific.log_scale,
        )
        assert all(
            left.geometry().right() <= right.geometry().left()
            for left, right in zip(top, top[1:])
        )
        assert shell.scientific.footer.parentWidget() is shell.scientific
        assert not shell.findChildren(
            QtWidgets.QAbstractButton, "e3PaneRoi"
        )
        assert not shell.findChildren(
            QtWidgets.QAbstractButton, "e3PaneMenu"
        )
        assert len(
            [
                widget
                for widget in shell.findChildren(QtWidgets.QAbstractButton)
                if widget.text() == "Metadata"
            ]
        ) == 1
    finally:
        shell.close()
        shell.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


@pytest.mark.parametrize(
    ("width", "height"),
    [(1440, 900), (1920, 1080), (1024, 900)],
)
def test_e3_ui5_maintainer_controls_are_exact_at_review_viewports(
    qapp: QtWidgets.QApplication,
    width: int,
    height: int,
) -> None:
    state = make_shell_projection()
    shell = ScatteringWorkspaceShell()
    shell.apply_state(state)
    shell.resize(width, height)
    shell.show()
    try:
        qapp.processEvents()
        rows = {
            tuple(row.path): row
            for row in shell.controls.findChildren(FormRow)
        }
        expected_labels = {
            ("Signal", "poni_file"): "Poni",
            ("Signal", "mask_file"): "Mask File",
            ("Signal", "inp_type"): "Source",
            ("Signal", "include_subdir"): "Subdirs",
            ("Signal", "img_dir"): "Directory",
            ("Signal", "img_ext"): "File Type",
            ("Signal", "meta_ext"): "Meta Type",
            ("Signal", "Filter"): "Filter",
        }
        assert {
            path: rows[path].label.text()
            for path in expected_labels
        } == expected_labels
        assert sum(
            "Pts" in {
                label.text()
                for label in rows[path].findChildren(QtWidgets.QLabel)
            }
            for path in (("Int1D", "axis"), ("Int2D", "axis"))
        ) == 2

        ranges = shell.controls.findChildren(RangeRow)
        assert [row.label.text() for row in ranges] == [
            "Q (Å⁻¹)",
            "χ (°)",
            "Q (Å⁻¹)",
            "χ (°)",
            "Threshold",
        ]
        assert [
            button.text()
            for row in shell.controls.findChildren(PillRow)
            for _path, button in row._pills
        ] == ["Mask Saturated", "Average Scan"]

        expected_frames = [
            str(frame.local_frame_label)
            for frame in state.navigation.frames
        ]
        browser_model = shell.browser.frames.model()
        assert [
            browser_model.index(index, 0).data(
                QtCore.Qt.ItemDataRole.DisplayRole
            )
            for index in range(browser_model.rowCount())
        ] == expected_frames
        assert [
            shell.scientific.frame_selector.itemText(index)
            for index in range(shell.scientific.frame_selector.count())
        ] == expected_frames
        assert type(shell.browser.metadata) is QtWidgets.QPushButton
        assert shell.browser.metadata.menu() is None
    finally:
        shell.close()
        shell.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


@pytest.mark.parametrize(
    ("width", "height"),
    [(1440, 900), (1920, 1080), (1024, 900)],
)
def test_e3_ui6_compact_shell_continuity_at_review_viewports(
    qapp: QtWidgets.QApplication,
    width: int,
    height: int,
) -> None:
    full_path = (
        "/facility/beamline/very-long-experiment-name/"
        "sample-series/processed-results"
    )
    state = make_shell_projection()
    state = replace(
        state,
        browser=replace(state.browser, directory=full_path),
    )
    shell = ScatteringWorkspaceShell()
    shell.apply_state(state)
    shell.resize(width, height)
    shell.show()
    try:
        qapp.processEvents()
        for pane in (shell.scientific.raw, shell.scientific.cake):
            assert pane.layout().count() == 1
            assert pane.canvas.geometry().top() == 0
            assert pane.canvas.geometry().height() == pane.geometry().height()

        assert type(shell.browser.refresh) is QtWidgets.QToolButton
        assert not shell.browser.refresh.icon().isNull()
        assert shell.browser.refresh.accessibleName() == "Refresh"
        assert shell.browser.scans_label.text() == "Scans"
        assert shell.browser.directory_label.text() != full_path
        assert shell.browser.directory_label.toolTip() == full_path

        assert not shell.tools.findChildren(
            QtWidgets.QLabel, "toolsHeader"
        )
        tool_buttons = shell.tools.findChildren(QtWidgets.QPushButton)
        assert [button.text() for button in tool_buttons] == [
            "∧ Peak Fitting",
            "≈ Phase Fitting",
            "▤ Plot Metadata",
        ]
        assert all(
            shell.tools.tool_content.rect().contains(
                button.mapTo(
                    shell.tools.tool_content,
                    button.rect().center(),
                )
            )
            for button in tool_buttons
        )

        chips = shell.controls.findChildren(
            QtWidgets.QLabel, "controlsV2SectionChip"
        )
        assert all(chip.isHidden() for chip in chips)

        navigation = state.navigation
        browser_model = shell.browser.frame_model
        assert browser_model.frames is navigation.frames
        assert tuple(
            shell.scientific.frame_selector.itemData(index)
            for index in range(shell.scientific.frame_selector.count())
        ) == navigation.selected
        assert shell.browser.frames.currentIndex().data(
            QtCore.Qt.ItemDataRole.UserRole
        ) is navigation.current
    finally:
        shell.close()
        shell.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def test_e3_ui3_narrow_focus_and_scroll_containment(
    qapp: QtWidgets.QApplication,
) -> None:
    shell = ScatteringWorkspaceShell()
    shell.apply_state(make_shell_projection())
    shell.resize(1024, 900)
    shell.show()
    try:
        qapp.processEvents()
        shell.browser.scans.setFocus()
        qapp.processEvents()
        assert shell.browser.scans.hasFocus()
        shell.scientific.frame_selector.setFocus()
        qapp.processEvents()
        assert shell.scientific.frame_selector.hasFocus()
        assert shell.control_scroll.verticalScrollBar().maximum() > 0
        toolbars = shell.findChildren(
            QtWidgets.QScrollArea, "e3ScrollableToolbar"
        )
        assert len(toolbars) == 2
        assert any(
            area.horizontalScrollBar().maximum() > 0
            for area in toolbars
        )
    finally:
        shell.close()
        shell.deleteLater()
        qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def test_e3_ui3_source_has_no_forbidden_authority_imports() -> None:
    forbidden = {
        "context_controller",
        "context_projection",
        "run_executor",
        "static_scan_widget",
        "h5viewer",
        "display_frame_widget",
        "parameter_tree",
    }
    for name in _VISUAL_FILES:
        source = (_SOURCE_DIR / name).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {
            alias.name.lower()
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            (node.module or "").lower()
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        assert not any(
            token in imported
            for token in forbidden
            for imported in imports
        )
        assert "ParameterTree" not in source


def test_e3_ui3_direct_deferred_delete_releases_entire_shell(
    qapp: QtWidgets.QApplication,
) -> None:
    shell = ScatteringWorkspaceShell()
    child = shell.scientific
    shell.apply_state(make_shell_projection())
    shell.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(
        None, QtCore.QEvent.Type.DeferredDelete
    )
    qapp.processEvents()

    assert not isValid(child)
    assert not isValid(shell)
