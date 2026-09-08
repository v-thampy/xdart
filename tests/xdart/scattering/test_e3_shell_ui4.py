from __future__ import annotations

from dataclasses import replace
import time

import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.tabs.scattering.browser_view import BrowserView
import xdart.gui.tabs.scattering.browser_view as browser_view_module
from xdart.gui.tabs.scattering.display_values import DisplayFrameKey
from xdart.gui.tabs.scattering.events import RunIdentity
from xdart.gui.tabs.scattering.scientific_view import (
    PLOT_TOOLBAR_INTER_GROUP_GAP,
    PLOT_TOOLBAR_INTRA_GROUP_GAP,
    PLOT_TOOLBAR_MENU_INDICATOR,
)
from xdart.gui.tabs.scattering.shell_values import (
    BrowserProjection,
    FrameNavigationProjection,
    HeavyProjection,
)
from xdart.gui.tabs.scattering.shell_widgets import repeated_labels
from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell
from xdart.gui.widgets.controls_panel import ControlsPanel, FormRow

from tests.xdart.scattering.e3_shell_support import make_shell_projection


@pytest.fixture
def qapp() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _dispose(widget: QtWidgets.QWidget, qapp: QtWidgets.QApplication) -> None:
    widget.close()


def _layout_widgets(layout: QtWidgets.QLayout) -> tuple[QtWidgets.QWidget, ...]:
    return tuple(
        item.widget()
        for index in range(layout.count())
        if (item := layout.itemAt(index)).widget() is not None
    )


def _labels(widget: QtWidgets.QWidget) -> set[str]:
    return {
        label.text()
        for label in widget.findChildren(QtWidgets.QLabel)
        if label.text()
    }


def _large_browser_state(
    frame_count: int,
    *,
    frames: tuple[DisplayFrameKey, ...] | None = None,
) -> FrameNavigationProjection:
    if frames is None:
        identity = RunIdentity(29, "e3-ui4-browser")
        frames = tuple(
            DisplayFrameKey(
                identity,
                f"scan-{index // 1000}",
                "large.nxs",
                index % 997,
                index + 1,
            )
            for index in range(frame_count)
        )
    return FrameNavigationProjection(frames, frames[-1], frames[-1:])


def test_e3_ui4_pending_heavy_persists_but_exact_absent_tiers_clear(
    qapp: QtWidgets.QApplication,
) -> None:
    shell = ScatteringWorkspaceShell()
    initial = make_shell_projection()
    try:
        shell.apply_state(initial)
        raw = shell.scientific.raw.image.image.copy()
        cake = shell.scientific.cake.image.image.copy()
        title = shell.scientific.title.text()
        traces = tuple(
            (item.xData.copy(), item.yData.copy())
            for item in shell.scientific.curve.listDataItems()
        )
        selected = initial.navigation.frames[1]

        pending = replace(
            initial,
            revision=2,
            scientific=replace(
                initial.scientific,
                heavy=None,
                title="scan-a:2",
                retain_display=True,
            ),
            navigation=replace(
                initial.navigation,
                current=selected,
                selected=(selected,),
            ),
        )
        shell.apply_state(pending)
        assert shell.scientific.title.text() == title
        assert (shell.scientific.raw.image.image == raw).all()
        assert (shell.scientific.cake.image.image == cake).all()
        rendered = shell.scientific.curve.listDataItems()
        assert len(rendered) == len(traces)
        for item, (x_data, y_data) in zip(rendered, traces, strict=True):
            assert (item.xData == x_data).all()
            assert (item.yData == y_data).all()

        exact_absent = replace(
            pending,
            revision=3,
            scientific=replace(
                pending.scientific,
                heavy=HeavyProjection(selected, None, None),
            ),
        )
        shell.apply_state(exact_absent)
        assert shell.scientific.raw.image.image is None
        assert shell.scientific.cake.image.image is None

        exact_cake = replace(
            initial.scientific.heavy,
            frame=selected,
            raw=None,
        )
        shell.apply_state(
            replace(
                exact_absent,
                revision=4,
                scientific=replace(exact_absent.scientific, heavy=exact_cake),
            )
        )
        assert shell.scientific.raw.image.image is None
        assert shell.scientific.cake.image.image is not None
    finally:
        _dispose(shell, qapp)


def test_e3_ui4_scientific_controls_have_exact_local_and_global_order(
    qapp: QtWidgets.QApplication,
) -> None:
    shell = ScatteringWorkspaceShell()
    try:
        assert _layout_widgets(shell.scientific.top_bar) == (
            shell.scientific.norm,
            shell.scientific.background,
            shell.scientific.title,
            shell.scientific.color_map,
            shell.scientific.log_scale,
        )
        assert shell.scientific.raw.color_scale.isVisible()
        assert shell.scientific.raw.layout().count() == 1
        assert shell.scientific.cake.color_scale.isVisible()
        assert shell.scientific.cake.layout().count() == 1

        assert _layout_widgets(shell.scientific.plot_bar) == (
            shell.scientific.axis_display_group,
            shell.scientific.plot_action_group,
            shell.scientific.share_axis,
            shell.scientific.image_axis,
        )
        assert _layout_widgets(
            shell.scientific.axis_display_group.layout()
        ) == (
            shell.scientific.plot_axis,
            shell.scientific.slice,
            shell.scientific.slice_center,
            shell.scientific.slice_width,
            shell.scientific.pin,
        )
        assert _layout_widgets(
            shell.scientific.plot_action_group.layout()
        ) == (
            shell.scientific.plot_mode,
            shell.scientific.options,
            shell.scientific.clear,
        )
        assert _layout_widgets(shell.scientific.footer) == (
            shell.scientific.status,
            shell.scientific.previous_frame,
            shell.scientific.frame_selector,
            shell.scientific.next_frame,
            shell.scientific.progress,
        )
        assert shell.scientific.options.menu() is None
        assert shell.scientific.options.text() == (
            f"Options {PLOT_TOOLBAR_MENU_INDICATOR}"
        )
        assert shell.scientific.options.accessibleName() == "Options"
        assert not (
            shell.scientific.options._paint_option().features
            & QtWidgets.QStyleOptionButton.ButtonFeature.HasMenu
        )
        assert (
            shell.scientific.axis_display_group.layout().spacing()
            == PLOT_TOOLBAR_INTRA_GROUP_GAP
        )
        assert (
            shell.scientific.plot_action_group.layout().spacing()
            == PLOT_TOOLBAR_INTRA_GROUP_GAP
        )
        assert (
            shell.scientific.plot_bar.itemAt(1).spacerItem().sizeHint().width()
            == PLOT_TOOLBAR_INTER_GROUP_GAP
        )
        assert (
            shell.scientific.plot_bar.itemAt(3)
            .spacerItem()
            .expandingDirections()
            & QtCore.Qt.Orientation.Horizontal
        )
        assert shell.scientific.plot_toolbar.objectName() == (
            "e3ScrollableToolbar"
        )
        assert (
            shell.scientific.plot_toolbar.horizontalScrollBarPolicy()
            is QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        assert all(
            PLOT_TOOLBAR_MENU_INDICATOR not in button.text()
            for button in (
                shell.scientific.slice,
                shell.scientific.share_axis,
            )
        )
        dialog = shell.scientific.plot_options_dialog
        assert [
            group.title()
            for group in dialog.findChildren(QtWidgets.QGroupBox)
        ] == ["Waterfall", "Overlay", "Other"]
        assert dialog.waterfall_stop.specialValueText() == "End"
        assert dialog.show_legend.isCheckable()
    finally:
        _dispose(shell, qapp)


def test_e3_ui4_shell_has_one_output_control_and_overrides_title_only(
    qapp: QtWidgets.QApplication,
) -> None:
    state = make_shell_projection()
    shell = ScatteringWorkspaceShell()
    standalone = ControlsPanel()
    try:
        shell.apply_state(state)
        shell_paths = {
            row.path for row in shell.controls.findChildren(FormRow)
        }
        original_paths = {
            field.path for field in state.controls.fields
        }
        assert ("Project", "output_mode") not in shell_paths
        assert ("Project", "output_mode") not in original_paths
        assert not shell.run_controls.writeModeButton.isHidden()
        assert "Configuration" in _labels(shell.controls)
        assert "Sample & measurement" not in _labels(shell.controls)

        standalone.reconcile(state.controls)
        assert "Sample & measurement" in _labels(standalone)
        assert "Configuration" not in _labels(standalone)
    finally:
        _dispose(shell, qapp)
        _dispose(standalone, qapp)


def test_e3_ui4_exact_scientific_labels_and_metadata_action(
    qapp: QtWidgets.QApplication,
) -> None:
    shell = ScatteringWorkspaceShell()
    try:
        shell.apply_state(make_shell_projection())
        assert [
            shell.scientific.plot_axis.itemText(index)
            for index in range(shell.scientific.plot_axis.count())
        ] == ["Q (Å⁻¹)", "2θ (°)", "χ (°)"]
        assert [
            shell.scientific.image_axis.itemText(index)
            for index in range(shell.scientific.image_axis.count())
        ] == ["Q-χ", "2θ-χ"]
        assert shell.scientific.slice.text() == "χ (c/w)"
        assert shell.scientific.cake.plot.getAxis("bottom").labelText == "Q"
        assert shell.scientific.cake.plot.getAxis("bottom").labelUnits == "Å⁻¹"
        assert shell.scientific.cake.plot.getAxis("left").labelText == "χ"
        assert shell.scientific.cake.plot.getAxis("left").labelUnits == "°"
        assert type(shell.browser.metadata) is QtWidgets.QPushButton
        assert shell.browser.metadata.menu() is None
    finally:
        _dispose(shell, qapp)


def test_e3_ui4_browser_15000_unchanged_and_prefix_delta_are_bounded(
    qapp: QtWidgets.QApplication,
) -> None:
    browser = BrowserView()
    state = _large_browser_state(15_000)
    browser_state = BrowserProjection("/data/processed")
    try:
        browser_state = replace(browser_state, frames=state.frames)
        browser.reconcile(browser_state, state, plot_mode="Overlay")

        started = time.perf_counter()
        browser.reconcile(browser_state, state, plot_mode="Overlay")
        unchanged_s = time.perf_counter() - started

        frames = state.frames
        appended = DisplayFrameKey(
            frames[0].run_identity,
            "scan-15",
            "large.nxs",
            frames[0].local_frame_label,
            15_001,
        )
        prefix = _large_browser_state(
            15_001,
            frames=(*frames, appended),
        )
        started = time.perf_counter()
        browser.reconcile(
            replace(browser_state, frames=prefix.frames),
            prefix,
            plot_mode="Overlay",
        )
        prefix_s = time.perf_counter() - started

        assert browser.frames.model().rowCount() == 15_001
        assert unchanged_s < 0.25
        assert prefix_s < 0.25
    finally:
        _dispose(browser, qapp)


def test_e3_ui4_overlay_membership_never_rescans_catalog_per_trace(
    qapp: QtWidgets.QApplication,
    monkeypatch,
) -> None:
    browser = BrowserView()
    navigation = _large_browser_state(3_621)
    navigation = replace(navigation, selected=navigation.frames)
    browser_state = replace(
        BrowserProjection("/data/processed"),
        frames=navigation.frames,
    )

    def _forbid_prefix_scan(_items) -> bool:
        raise AssertionError("overlay membership rescanned the frame catalog")

    monkeypatch.setattr(
        browser_view_module, "any", _forbid_prefix_scan, raising=False,
    )
    try:
        browser.reconcile(browser_state, navigation, plot_mode="Overlay")
        assert browser._trace_frames == navigation.frames
    finally:
        _dispose(browser, qapp)


def test_e3_ui4_repeated_labels_is_linear_at_15000_rows() -> None:
    frames = _large_browser_state(15_000).frames
    started = time.perf_counter()
    repeated = repeated_labels(frames)
    elapsed = time.perf_counter() - started

    assert repeated == frozenset(range(997))
    assert elapsed < 0.25
