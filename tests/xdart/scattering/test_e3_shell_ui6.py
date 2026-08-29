from __future__ import annotations

from dataclasses import fields, replace
import json
from pathlib import Path

import pytest
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets

from xdart.gui.tabs.scattering import shell_values
from xdart.gui.tabs.scattering.display_values import DisplayFrameKey
from xdart.gui.tabs.scattering.shell_values import (
    BrowserProjection,
    ScientificProjection,
    ShellCommandKind,
)
from xdart.gui.tabs.scattering.tools_view import ToolsView
from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell
from xdart.gui.widgets.controls_panel import ControlsPanel

from tests.xdart.scattering.e3_shell_support import make_shell_projection


_OVERRIDES = (
    Path(__file__).with_name("fixtures")
    / "e3_shell_ui6_overrides_ddc96a97.json"
)


@pytest.fixture
def qapp() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _dispose(
    widget: QtWidgets.QWidget,
    qapp: QtWidgets.QApplication,
) -> None:
    widget.close()


def _navigation_type():
    projection = getattr(shell_values, "FrameNavigationProjection", None)
    assert projection is not None
    return projection


def _navigation(state):
    navigation = getattr(state, "navigation", None)
    assert navigation is not None
    return navigation


def _selected_browser_keys(
    shell: ScatteringWorkspaceShell,
) -> tuple[DisplayFrameKey, ...]:
    return tuple(
        index.data(QtCore.Qt.ItemDataRole.UserRole)
        for index in shell.browser.frames.selectionModel().selectedRows()
    )


def _footer_frames(navigation) -> tuple[DisplayFrameKey, ...]:
    current = navigation.current
    assert current is not None
    return tuple(
        frame
        for frame in navigation.frames
        if frame.artifact == current.artifact
    )


def test_e3_ui6_override_inventory_is_anchored_to_exact_ui5_tip() -> None:
    accepted = json.loads(_OVERRIDES.read_text(encoding="utf-8"))

    assert accepted["parent_commit"] == (
        "ddc96a97e11e19bd6261f6eeeed2716683d1fbb9"
    )
    assert accepted["canonical_inventory"] == (
        "e3_shell_inventory_ff5380a7.json"
    )
    assert accepted["navigation"]["owner"].startswith(
        "one exact FrameNavigationProjection"
    )


def test_e3_ui6_image_panes_have_no_dead_toolbar_or_blank_row(
    qapp: QtWidgets.QApplication,
) -> None:
    shell = ScatteringWorkspaceShell()
    shell.apply_state(make_shell_projection())
    shell.resize(1440, 900)
    shell.show()
    try:
        qapp.processEvents()
        for pane in (shell.scientific.raw, shell.scientific.cake):
            assert not hasattr(pane, "tool_bar")
            assert pane.layout().count() == 1
            assert pane.layout().itemAt(0).widget() is pane.canvas
            assert pane.canvas.geometry().top() == 0
        assert not shell.findChildren(
            QtWidgets.QAbstractButton, "e3PaneRoi"
        )
        assert not shell.findChildren(
            QtWidgets.QAbstractButton, "e3PaneMenu"
        )
        assert not hasattr(ShellCommandKind, "IMAGE_TOOL")
    finally:
        _dispose(shell, qapp)


def test_e3_ui6_refresh_and_directory_are_compact_and_accessible(
    qapp: QtWidgets.QApplication,
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
    commands = []
    shell.commandRequested.connect(commands.append)
    try:
        shell.apply_state(state)
        shell.resize(1024, 900)
        shell.show()
        qapp.processEvents()

        assert type(shell.browser.refresh) is QtWidgets.QToolButton
        assert not shell.browser.refresh.icon().isNull()
        assert shell.browser.refresh.toolTip() == "Refresh"
        assert shell.browser.refresh.accessibleName() == "Refresh"
        assert shell.browser.refresh.focusPolicy() != (
            QtCore.Qt.FocusPolicy.NoFocus
        )
        shell.browser.refresh.click()
        assert [command.kind for command in commands] == [
            ShellCommandKind.REFRESH_BROWSER
        ]

        assert shell.browser.scans_label.text() == "Scans"
        shown_path = shell.browser.directory_label.text()
        assert shown_path != f"[{full_path}]"
        assert shown_path.startswith("[")
        assert shown_path.endswith("]")
        assert "…" in shown_path[1:-1]
        assert len(shown_path[1:-1]) <= 30
        assert shell.browser.directory_label.toolTip() == full_path
        before = shell.browser.directory_label.text()
        font = shell.browser.directory_label.font()
        font.setPointSize(font.pointSize() + 2)
        shell.browser.directory_label.setFont(font)
        shell.resize(1440, 900)
        qapp.processEvents()
        assert shell.browser.directory_label.text()
        assert shell.browser.directory_label.toolTip() == full_path
        assert before != full_path
    finally:
        _dispose(shell, qapp)


def test_e3_ui6_tools_are_heading_free_dynamic_and_scrollable(
    qapp: QtWidgets.QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ToolsView,
        "_TOOLS",
        tuple((f"Tool {index}", f"tool_{index}") for index in range(12)),
    )
    tools = ToolsView()
    tools.resize(255, 120)
    tools.show()
    try:
        qapp.processEvents()
        assert not tools.findChildren(QtWidgets.QLabel, "toolsHeader")
        buttons = tools.findChildren(QtWidgets.QPushButton)
        assert len(buttons) == 12 + len(tools._EXTERNAL_VIEWERS)
        assert all(button.parentWidget() is tools.tool_content for button in buttons)
        scroll = tools.tool_scroll.verticalScrollBar()
        assert tools.tool_scroll.verticalScrollBarPolicy() == (
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        assert scroll.maximum() > 0
        scroll.setValue(scroll.maximum())
        qapp.processEvents()
        last_center = buttons[-1].mapTo(
            tools.tool_scroll.viewport(),
            buttons[-1].rect().center(),
        )
        assert tools.tool_scroll.viewport().rect().contains(last_center)
    finally:
        _dispose(tools, qapp)


def test_e3_ui6_section_numbers_are_hidden_only_in_vnext_shell(
    qapp: QtWidgets.QApplication,
) -> None:
    canonical = ControlsPanel()
    shell = ScatteringWorkspaceShell()
    try:
        canonical_chips = canonical.findChildren(
            QtWidgets.QLabel, "controlsSectionChip"
        )
        shell_chips = shell.controls.findChildren(
            QtWidgets.QLabel, "controlsSectionChip"
        )
        assert any(chip.text() == "1" and not chip.isHidden() for chip in canonical_chips)
        assert all(chip.isHidden() for chip in shell_chips)
    finally:
        _dispose(canonical, qapp)
        _dispose(shell, qapp)


def test_e3_ui6_navigation_projection_preserves_identity_and_anchor_split() -> None:
    projection = _navigation_type()
    state = make_shell_projection()
    frame = state.navigation.frames[0]
    clone = DisplayFrameKey(
        frame.run_identity,
        frame.source_scan,
        frame.artifact,
        frame.local_frame_label,
        frame.work_ordinal,
    )

    with pytest.raises(ValueError, match="identity"):
        projection((frame,), clone, (clone,))
    with pytest.raises(ValueError, match="unique"):
        projection((frame,), frame, (frame, frame))
    anchor_only = projection((frame,), frame, ())
    assert anchor_only.current is frame
    assert anchor_only.selected == ()


def test_e3_ui6_shell_has_one_frame_navigation_owner() -> None:
    state = make_shell_projection()
    assert _navigation(state) is state.navigation
    assert "navigation" in {field.name for field in fields(type(state))}
    assert "frames" in {
        field.name for field in fields(BrowserProjection)
    }
    assert all(
        actual is expected
        for actual, expected in zip(
            state.browser.frames,
            state.navigation.frames,
            strict=True,
        )
    )
    assert not {
        "frames",
        "selected_frame",
    } & {field.name for field in fields(ScientificProjection)}


def test_e3_ui6_single_footer_catalog_replaces_singleton_selection(
    qapp: QtWidgets.QApplication,
) -> None:
    state = make_shell_projection(plot_mode="Single")
    navigation = _navigation(state)
    selected = (
        navigation.frames[0],
        navigation.frames[2],
        navigation.frames[4],
    )
    state = replace(
        state,
        navigation=type(navigation)(
            navigation.frames,
            navigation.frames[2],
            selected,
        ),
    )
    shell = ScatteringWorkspaceShell()
    commands = []
    shell.commandRequested.connect(commands.append)
    try:
        shell.apply_state(state)
        footer = _footer_frames(state.navigation)
        assert tuple(
            shell.scientific.frame_selector.itemData(index)
            for index in range(shell.scientific.frame_selector.count())
        ) == footer
        assert shell.browser.frame_model.frames is navigation.frames
        assert _selected_browser_keys(shell) == selected
        assert len(shell.scientific.curve.listDataItems()) == 3

        shell.scientific.frame_selector.setCurrentIndex(1)
        assert len(commands) == 1
        assert commands[0].frame is navigation.frames[1]
        assert commands[0].frames == (navigation.frames[1],)

        next_navigation = type(navigation)(
            navigation.frames,
            navigation.frames[1],
            (navigation.frames[1],),
        )
        shell.apply_state(
            replace(state, revision=2, navigation=next_navigation)
        )
        assert shell.browser.frames.currentIndex().data(
            QtCore.Qt.ItemDataRole.UserRole
        ) is navigation.frames[1]
        assert _selected_browser_keys(shell) == (navigation.frames[1],)
    finally:
        _dispose(shell, qapp)


@pytest.mark.parametrize(
    "mode",
    ["Overlay", "Waterfall", "Average", "Sum"],
)
def test_e3_ui6_multi_mode_footer_preserves_membership_and_mirrors_focus(
    qapp: QtWidgets.QApplication,
    mode: str,
) -> None:
    state = make_shell_projection()
    navigation = _navigation(state)
    selected = (
        navigation.frames[0],
        navigation.frames[3],
        navigation.frames[4],
    )
    navigation = type(navigation)(
        navigation.frames,
        navigation.frames[3],
        selected,
    )
    state = replace(
        state,
        scientific=replace(state.scientific, plot_mode=mode),
        navigation=navigation,
    )
    shell = ScatteringWorkspaceShell()
    commands = []
    shell.commandRequested.connect(commands.append)
    try:
        shell.apply_state(state)
        footer = _footer_frames(navigation)
        assert tuple(
            shell.scientific.frame_selector.itemData(index)
            for index in range(shell.scientific.frame_selector.count())
        ) == footer
        assert shell.browser.frame_model.frames is navigation.frames
        expected_browser_selection = (
            (selected[1],)
            if mode in {"Overlay", "Waterfall"}
            else selected
        )
        assert _selected_browser_keys(shell) == expected_browser_selection
        assert shell.browser.frames.currentIndex().data(
            QtCore.Qt.ItemDataRole.UserRole
        ) is selected[1]

        shell.scientific.frame_selector.setCurrentIndex(1)
        assert len(commands) == 1
        assert commands[0].frame is navigation.frames[1]
        expected_membership = (
            selected
            if mode in {"Overlay", "Waterfall"}
            else (*selected, navigation.frames[1])
        )
        assert commands[0].frames == expected_membership

        next_navigation = type(navigation)(
            navigation.frames,
            navigation.frames[1],
            expected_membership,
        )
        shell.apply_state(
            replace(state, revision=2, navigation=next_navigation)
        )
        expected_after_selection = (
            expected_browser_selection
            if mode in {"Overlay", "Waterfall"}
            else expected_membership
        )
        assert _selected_browser_keys(shell) == expected_after_selection
        assert shell.browser.frames.currentIndex().data(
            QtCore.Qt.ItemDataRole.UserRole
        ) is navigation.frames[1]
    finally:
        _dispose(shell, qapp)


def test_e3_ui6_prefix_append_and_retirement_update_both_widgets_atomically(
    qapp: QtWidgets.QApplication,
) -> None:
    state = make_shell_projection()
    navigation = state.navigation
    shell = ScatteringWorkspaceShell()
    commands = []
    shell.commandRequested.connect(commands.append)
    try:
        shell.apply_state(state)
        first = navigation.frames[0]
        appended = DisplayFrameKey(
            first.run_identity,
            "scan-c",
            first.artifact,
            10,
            6,
        )
        appended_frames = (*navigation.frames, appended)
        appended_navigation = type(navigation)(
            appended_frames,
            appended,
            appended_frames,
        )
        shell.apply_state(
            replace(
                state,
                revision=2,
                navigation=appended_navigation,
                browser=replace(
                    state.browser,
                    frames=appended_frames,
                ),
            )
        )
        assert shell.browser.frame_model.frames is appended_frames
        assert tuple(
            shell.scientific.frame_selector.itemData(index)
            for index in range(shell.scientific.frame_selector.count())
        ) == appended_frames

        retired_frames = appended_frames[1:]
        retired_navigation = type(navigation)(
            retired_frames,
            appended,
            retired_frames,
        )
        shell.apply_state(
            replace(
                state,
                revision=3,
                navigation=retired_navigation,
                browser=replace(
                    state.browser,
                    frames=retired_frames,
                ),
            )
        )
        assert shell.browser.frame_model.frames is retired_frames
        assert tuple(
            shell.scientific.frame_selector.itemData(index)
            for index in range(shell.scientific.frame_selector.count())
        ) == retired_frames
        assert shell.browser.frames.currentIndex().data(
            QtCore.Qt.ItemDataRole.UserRole
        ) is appended
        assert shell.scientific.frame_selector.currentData() is appended
        assert commands == []
    finally:
        _dispose(shell, qapp)


def test_e3_ui6_overlay_browser_selection_moves_focus_not_membership(
    qapp: QtWidgets.QApplication,
) -> None:
    state = make_shell_projection()
    navigation = _navigation(state)
    state = replace(
        state,
        scientific=replace(state.scientific, plot_mode="Overlay"),
    )
    shell = ScatteringWorkspaceShell()
    commands = []
    shell.commandRequested.connect(commands.append)
    try:
        shell.apply_state(state)
        selection = shell.browser.frames.selectionModel()
        chosen = QtCore.QItemSelection()
        chosen.select(
            shell.browser.frame_model.index(1, 0),
            shell.browser.frame_model.index(1, 0),
        )
        chosen.select(
            shell.browser.frame_model.index(3, 0),
            shell.browser.frame_model.index(3, 0),
        )
        commands.clear()
        selection.select(
            chosen,
            QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
            | QtCore.QItemSelectionModel.SelectionFlag.Rows,
        )
        QtTest.QTest.qWait(130)

        assert len(commands) == 1
        assert commands[0].kind is (
            ShellCommandKind.SELECT_BROWSER_FRAMES
        )
        assert len(commands[0].frames) == 2
        assert commands[0].frames[0] is navigation.frames[1]
        assert commands[0].frames[1] is navigation.frames[3]
        assert commands[0].frame is navigation.frames[3]
    finally:
        _dispose(shell, qapp)


def test_e3_ui6_frame_selector_width_uses_five_digit_style_metrics(
    qapp: QtWidgets.QApplication,
) -> None:
    shell = ScatteringWorkspaceShell()
    try:
        shell.apply_state(make_shell_projection())
        selector = shell.scientific.frame_selector
        option = QtWidgets.QStyleOptionComboBox()
        selector.initStyleOption(option)
        content = QtCore.QSize(
            selector.fontMetrics().horizontalAdvance("00000"),
            selector.fontMetrics().height(),
        )
        expected = selector.style().sizeFromContents(
            QtWidgets.QStyle.ContentsType.CT_ComboBox,
            option,
            content,
            selector,
        ).width()
        assert selector.minimumWidth() == expected
        assert selector.maximumWidth() == expected
        assert expected != 115
    finally:
        _dispose(shell, qapp)
