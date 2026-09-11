from __future__ import annotations

from pathlib import Path

import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering.controls_projection import project_controls
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xdart.gui.widgets.controls_panel import (
    ControlsPanel,
    FormRow,
    RangeRow,
    SubsectionCard,
)
from xdart.gui.widgets.run_controls import RunControlsBar
from xdart.gui.themes import apply_theme
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import DirectorySourceSpec


@pytest.fixture
def qapp() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _intent() -> RunIntent:
    return RunIntent(
        source_spec=DirectorySourceSpec(
            Path("/raw/eiger"),
            suffixes=(".h5",),
        ),
        poni_file="/project/detector.poni",
        mask_file="/project/mask.npy",
        project_root="/project",
        save_path="/processed",
        output_mode="Overwrite",
        bai_1d_args={
            "radial_range": (0.0, 5.0),
            "azimuth_range": (-180.0, 180.0),
        },
        bai_2d_args={
            "radial_range": (0.0, 5.0),
            "azimuth_range": (-180.0, 180.0),
        },
    )


_SURFACE_NAMES = (
    "controlsLineEdit",
    "controlsComboBox",
    "controlsToggleButton",
    "controlsPillButton",
    "controlsSegmentButton",
    "controlsBrowseButton",
    "controlsMoreButton",
    "controlsAutoButton",
    "controlsActionButton",
)


def _is_square(widget: QtWidgets.QWidget) -> bool:
    image = widget.grab().toImage()
    if image.isNull():
        return False
    corner = image.pixelColor(0, 0)
    top_edge = image.pixelColor(image.width() // 2, 0)
    left_edge = image.pixelColor(0, image.height() // 2)
    return corner == top_edge == left_edge


def _right_interior_colors(
    widget: QtWidgets.QWidget,
    *,
    width: int,
) -> set[str]:
    image = widget.grab().toImage()
    assert not image.isNull()
    return {
        image.pixelColor(x, y).name()
        for x in range(max(1, image.width() - width), image.width() - 1)
        for y in range(1, image.height() - 1)
    }


@pytest.mark.parametrize(
    ("theme", "font_scale", "spacing", "expected_height"),
    (
        ("dark", "default", "normal", 28),
        ("light", "default", "normal", 28),
        ("dark", "extra_large", "extra_spacious", 34),
        ("light", "extra_large", "extra_spacious", 34),
    ),
)
def test_controls_interactive_surfaces_share_square_scaled_geometry(
    qapp: QtWidgets.QApplication,
    theme: str,
    font_scale: str,
    spacing: str,
    expected_height: int,
) -> None:
    apply_theme(
        qapp,
        theme,
        font_scale=font_scale,
        spacing=spacing,
    )
    panel = ControlsPanel()
    run_controls = RunControlsBar()
    try:
        intent = _intent()
        intent.gi.enabled = True
        panel.reconcile(
            project_controls(
                RunIntentStore(intent).snapshot(),
                None,
                RunPhase.IDLE,
            )
        )
        # Reproduce the mounted Controls redraw that used to leave the native
        # Tight popup container at approximately one closed-control height.
        panel.reconcile(
            project_controls(
                RunIntentStore(intent).snapshot(),
                None,
                RunPhase.IDLE,
            )
        )
        panel.resize(520, 1800)
        run_controls.resize(520, run_controls.sizeHint().height())
        panel.show()
        run_controls.show()
        qapp.processEvents()

        surfaces = {
            name: panel.findChildren(QtWidgets.QWidget, name)
            for name in _SURFACE_NAMES
        }
        assert all(surfaces.values())
        for widgets in surfaces.values():
            for widget in widgets:
                assert widget.height() == expected_height
                assert widget.minimumHeight() == expected_height
                assert widget.maximumHeight() == expected_height
                assert _is_square(widget)
        for combo in surfaces["controlsComboBox"]:
            # The compact caret adds one theme-coloured interior pixel family;
            # a blank drop-down well would be a single flat field colour.
            assert len(_right_interior_colors(combo, width=16)) > 1

        # Directory/file fields and coalesced range bounds use the same exact
        # surface contract rather than a separate path/range exception.
        path_rows = {
            row.path: row
            for row in panel.findChildren(FormRow)
            if row.browse_button is not None
        }
        assert {
            ("Project", "project_folder"),
            ("Signal", "img_dir"),
            ("Signal", "poni_file"),
            ("Signal", "mask_file"),
        } <= set(path_rows)
        for row in path_rows.values():
            assert row.editor.height() == row.browse_button.height() == (
                expected_height
            )
        ranges = panel.findChildren(RangeRow)
        assert ranges
        for row in ranges:
            assert row._low.height() == row._high.height() == expected_height
            if row._toggle is not None:
                assert row._toggle[1].height() == expected_height

        # The bottom Controls strip's combo, spin input and toggle/action
        # buttons share the same app-wide appearance owner.
        run_surfaces = (
            run_controls.modeCombo,
            run_controls.batchButton,
            run_controls.coresSpin,
            run_controls.liveButton,
            run_controls.startButton,
            run_controls.stopButton,
            run_controls.writeModeButton,
        )
        for widget in run_surfaces:
            assert widget.height() == expected_height
            assert widget.minimumHeight() == expected_height
            assert widget.maximumHeight() == expected_height
            assert _is_square(widget)
        assert len(
            _right_interior_colors(run_controls.modeCombo, width=16)
        ) > 1

        # Only actual menu-bearing tool buttons receive the compact indicator.
        # A plain/checkable tool button with the same empty content has no
        # indicator pixels in its interior.
        menu_button = QtWidgets.QToolButton()
        menu_button.setObjectName("fileMenuButton")
        menu_button.setPopupMode(
            QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup
        )
        menu = QtWidgets.QMenu(menu_button)
        menu.addAction("Open")
        menu_button.setMenu(menu)
        menu_button.resize(36, expected_height)
        plain_toggle = QtWidgets.QToolButton()
        plain_toggle.setCheckable(True)
        plain_toggle.resize(36, expected_height)
        menu_button.show()
        plain_toggle.show()
        qapp.processEvents()
        assert len(_right_interior_colors(menu_button, width=12)) > 1
        assert len(_right_interior_colors(plain_toggle, width=12)) == 1
        plain_toggle.close()
        menu_button.close()

        # Enclosing workflow cards remain subtly rounded.
        subsection = panel.findChild(SubsectionCard)
        assert subsection is not None
        card_image = subsection.grab().toImage()
        assert not card_image.isNull()
        assert card_image.pixelColor(0, 0) != card_image.pixelColor(
            card_image.width() // 2,
            0,
        )
        apply_theme(
            qapp,
            theme,
            font_scale=font_scale,
            spacing=spacing,
            controls_card_corners=False,
        )
        qapp.processEvents()
        assert _is_square(subsection)
        for widgets in surfaces.values():
            for widget in widgets:
                assert widget.height() == expected_height
                assert _is_square(widget)
    finally:
        panel.close()
        run_controls.close()
        apply_theme(
            qapp,
            "dark",
            font_scale="default",
            spacing="normal",
            controls_card_corners=True,
        )


@pytest.mark.parametrize("path", (
    ("Int1D", "axis"), ("Signal", "img_ext"), ("Signal", "meta_ext"),
))
def test_tight_combo_popup_rows_keep_independent_readable_height(
    qapp: QtWidgets.QApplication,
    path: tuple[str, ...],
) -> None:
    apply_theme(
        qapp,
        "dark",
        font_scale="default",
        spacing="tight",
    )
    panel = ControlsPanel()
    combo = None
    try:
        panel.reconcile(
            project_controls(
                RunIntentStore(_intent()).snapshot(),
                None,
                RunPhase.IDLE,
            )
        )
        panel.resize(520, 1800)
        panel.show()
        qapp.processEvents()
        combo = next(
            row.editor
            for row in panel.findChildren(FormRow)
            if row.path == path
        )
        combo.showPopup()
        qapp.processEvents()
        first = combo.model().index(0, 0)
        control_height = max(combo.height(), combo.sizeHint().height())

        assert control_height > 0
        assert combo.view().sizeHintForRow(0) >= control_height
        assert combo.view().visualRect(first).height() >= control_height
        visible = min(combo.count(), combo.maxVisibleItems())
        expected_viewport_height = sum(
            combo.view().sizeHintForRow(row)
            for row in range(visible)
        )
        assert combo.view().height() >= expected_viewport_height
        widest_text = max(
            combo.fontMetrics().horizontalAdvance(combo.itemText(index))
            for index in range(combo.count())
        )
        assert combo.view().viewport().width() >= widest_text
        if path == ("Signal", "meta_ext"):
            row = combo.parentWidget()
            text_width = row.label.fontMetrics().horizontalAdvance(row.label.text())
            gap = combo.x() - (row.label.x() + text_width)
            # QLabel rounds its glyph bounds up to whole widget pixels.
            assert gap <= row.layout().spacing() + 1
    finally:
        if combo is not None:
            combo.hidePopup()
        panel.close()
        apply_theme(
            qapp,
            "dark",
            font_scale="default",
            spacing="normal",
            controls_card_corners=True,
        )
