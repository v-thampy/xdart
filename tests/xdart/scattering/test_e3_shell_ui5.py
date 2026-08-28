from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets

from xdart.gui.tabs.scattering.shell_values import ShellCommandKind
from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell
from xdart.gui.widgets.controls_panel import (
    FormRow,
    PillRow,
    RangeRow,
)

from tests.xdart.scattering.e3_shell_support import make_shell_projection


_OVERRIDES = (
    Path(__file__).with_name("fixtures")
    / "e3_shell_ui5_overrides_2e712ced.json"
)


@pytest.fixture
def qapp() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _dispose(
    shell: ScatteringWorkspaceShell,
    qapp: QtWidgets.QApplication,
) -> None:
    shell.close()


def _form_rows(widget: QtWidgets.QWidget) -> dict[tuple[str, ...], FormRow]:
    return {
        tuple(row.path): row
        for row in widget.findChildren(FormRow)
    }


def _field_map(state) -> dict[tuple[str, ...], object]:
    return {
        tuple(field.path): field
        for field in state.controls.fields
    }


def test_e3_ui5_override_inventory_is_anchored_without_rewriting_canonical() -> None:
    accepted = json.loads(_OVERRIDES.read_text(encoding="utf-8"))

    assert accepted["parent_commit"] == (
        "2e712cedd17bea4feb6d4c6894721f15345be32f"
    )
    assert accepted["canonical_inventory"] == (
        "e3_shell_inventory_ff5380a7.json"
    )
    assert accepted["frame_caption"].startswith("local_frame_label only")
    assert accepted["metadata"] == (
        "one plain push action with no attached menu"
    )


def test_e3_ui5_detector_and_directory_source_use_accepted_paths_and_order(
    qapp: QtWidgets.QApplication,
) -> None:
    state = make_shell_projection()
    shell = ScatteringWorkspaceShell()
    shell.apply_state(state)
    shell.resize(1920, 1080)
    shell.show()
    try:
        qapp.processEvents()
        fields = _field_map(state)
        assert fields[("Signal", "poni_file")].label == "Poni"
        assert fields[("Signal", "mask_file")].label == "Mask File"

        source_paths = tuple(
            field.path
            for field in state.controls.fields
            if field.section.value == "source"
        )
        assert source_paths == (
            ("Signal", "inp_type"),
            ("Signal", "img_dir"),
            ("Signal", "img_ext"),
            ("Signal", "include_subdir"),
            ("Signal", "Filter"),
            ("Signal", "meta_ext"),
            ("Source", "energy_preference"),
        )
        assert ("Signal", "File") not in fields

        experiment = _form_rows(shell.controls.experiment_card.body)
        assert experiment[("Signal", "poni_file")].label.text() == "Poni"
        assert experiment[("Signal", "mask_file")].label.text() == "Mask File"

        source = _form_rows(shell.controls.source_card.body)
        expected_labels = {
            ("Signal", "inp_type"): "Source",
            ("Signal", "include_subdir"): "Subdirs",
            ("Signal", "img_dir"): "Directory",
            ("Signal", "img_ext"): "File Type",
            ("Signal", "meta_ext"): "Meta Type",
            ("Signal", "Filter"): "Filter",
        }
        assert {
            path: source[path].label.text()
            for path in expected_labels
        } == expected_labels
        assert not {
            "Source Type",
            "Image File",
            "Mode",
        } & {
            row.label.text()
            for row in source.values()
            if not row.label.isHidden()
        }

        body = shell.controls.source_card.body
        y = {
            path: source[path].mapTo(body, QtCore.QPoint()).y()
            for path in expected_labels
        }
        assert y[("Signal", "inp_type")] == y[("Signal", "include_subdir")]
        assert y[("Signal", "inp_type")] < y[("Signal", "img_dir")]
        assert y[("Signal", "img_dir")] < y[("Signal", "img_ext")]
        assert y[("Signal", "img_ext")] == y[("Signal", "meta_ext")]
        assert y[("Signal", "meta_ext")] < y[("Signal", "Filter")]

        more = body.findChildren(
            QtWidgets.QToolButton, "controlsMoreButton"
        )
        assert len(more) == 1
        assert more[0].property("role") == "sourceEnergy"
        assert more[0].mapTo(body, QtCore.QPoint()).y() == (
            y[("Signal", "meta_ext")]
        )
    finally:
        _dispose(shell, qapp)


def test_e3_ui5_processing_rows_use_axis_paths_ranges_units_and_pills(
    qapp: QtWidgets.QApplication,
) -> None:
    state = make_shell_projection()
    shell = ScatteringWorkspaceShell()
    try:
        shell.apply_state(state)
        fields = _field_map(state)
        assert fields[("Int1D", "axis")].choices == (
            "Q (Å⁻¹)",
            "2θ (°)",
            "χ (°)",
        )
        assert fields[("Int2D", "axis")].choices == ("Q-χ", "2θ-χ")
        assert ("Int1D", "unit") in fields
        assert ("Int2D", "unit") in fields

        processing = shell.controls.processing_card.body
        rows = _form_rows(processing)
        for axis_path, point_paths in (
            (("Int1D", "axis"), (("Int1D", "points"),)),
            (
                ("Int2D", "axis"),
                (
                    ("Int2D", "radial_points"),
                    ("Int2D", "azim_points"),
                ),
            ),
        ):
            axis = rows[axis_path]
            assert axis.label.text() == "Axis"
            assert "Pts" in {
                label.text()
                for label in axis.findChildren(QtWidgets.QLabel)
            }
            for path in point_paths:
                assert rows[path].parentWidget() is axis
                assert rows[path].label.isHidden()

        ranges = {
            tuple(row._low_path): row
            for row in processing.findChildren(RangeRow)
        }
        expected_ranges = {
            ("Int1D", "radial_low"): "Q (Å⁻¹)",
            ("Int1D", "azim_low"): "χ (°)",
            ("Int2D", "radial_low"): "Q (Å⁻¹)",
            ("Int2D", "azim_low"): "χ (°)",
            ("Mask", "min"): "Threshold",
        }
        assert {
            path: ranges[path].label.text()
            for path in expected_ranges
        } == expected_ranges

        pills = processing.findChildren(PillRow)
        matching = [
            row
            for row in pills
            if {
                path for path, _button in row._pills
            } == {
                ("MaskSat", "mask_sentinel"),
                ("Signal", "series_average"),
            }
        ]
        assert len(matching) == 1
        assert {
            path: button.text()
            for path, button in matching[0]._pills
        } == {
            ("MaskSat", "mask_sentinel"): "Mask Saturated",
            ("Signal", "series_average"): "Average Scan",
        }
    finally:
        _dispose(shell, qapp)


def test_e3_ui5_one_based_visible_captions_keep_exact_key_commands(
    qapp: QtWidgets.QApplication,
) -> None:
    state = make_shell_projection()
    frames = state.navigation.frames
    shell = ScatteringWorkspaceShell()
    browser_commands = []
    scientific_commands = []
    shell.browser.commandRequested.connect(browser_commands.append)
    shell.scientific.commandRequested.connect(scientific_commands.append)
    try:
        shell.apply_state(state)
        expected = [str(position) for position in range(1, len(frames) + 1)]
        footer = tuple(
            frame
            for frame in frames
            if frame.artifact == state.navigation.current.artifact
        )
        model = shell.browser.frames.model()
        assert [
            model.index(index, 0).data(QtCore.Qt.ItemDataRole.DisplayRole)
            for index in range(model.rowCount())
        ] == expected
        assert all(
            model.index(index, 0).data(QtCore.Qt.ItemDataRole.UserRole)
            is frames[index]
            for index in range(model.rowCount())
        )
        assert [
            shell.scientific.frame_selector.itemText(index)
            for index in range(shell.scientific.frame_selector.count())
        ] == [
            str(position) for position in range(1, len(footer) + 1)
        ]
        assert all(
            shell.scientific.frame_selector.itemData(index)
            is footer[index]
            for index in range(shell.scientific.frame_selector.count())
        )

        selection = shell.browser.frames.selectionModel()
        selection.clearSelection()
        browser_commands.clear()
        selection.select(
            model.index(3, 0),
            QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
            | QtCore.QItemSelectionModel.SelectionFlag.Rows,
        )
        QtTest.QTest.qWait(130)
        assert len(browser_commands) == 1
        assert browser_commands[0].kind is (
            ShellCommandKind.SELECT_BROWSER_FRAMES
        )
        assert len(browser_commands[0].frames) == 1
        assert browser_commands[0].frames[0] is frames[3]
        assert browser_commands[0].frame is frames[3]

        shell.apply_state(
            replace(
                state,
                revision=2,
                navigation=replace(
                    state.navigation,
                    current=frames[2],
                ),
            )
        )
        shell.scientific.frame_selector.setCurrentIndex(1)
        assert len(scientific_commands) == 1
        assert scientific_commands[0].kind is ShellCommandKind.HYDRATE_FRAME
        assert scientific_commands[0].frame is frames[1]
        assert scientific_commands[0].frames == frames
        assert frames[0].local_frame_label == (
            frames[3].local_frame_label
        ) == 1
        assert expected[0] == "1"
        assert expected[3] == "4"
    finally:
        _dispose(shell, qapp)


def test_e3_ui5_metadata_is_one_plain_popup_action(
    qapp: QtWidgets.QApplication,
) -> None:
    shell = ScatteringWorkspaceShell()
    commands = []
    shell.commandRequested.connect(commands.append)
    try:
        shell.apply_state(make_shell_projection())
        assert type(shell.browser.metadata) is QtWidgets.QPushButton
        assert shell.browser.metadata.text() == "Metadata"
        assert shell.browser.metadata.menu() is None

        shell.browser.metadata.click()
        assert len(commands) == 1
        assert commands[0].kind is ShellCommandKind.SHOW_METADATA
    finally:
        _dispose(shell, qapp)
