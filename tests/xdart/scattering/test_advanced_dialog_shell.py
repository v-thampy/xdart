"""Int's Advanced dialog as the owner of every page's Advanced settings (AUD-9 stage 2b G2; Codex's reuse
audits, 2026-10-01: Stitch's and RSM's own Advanced dialogs are copies of it)."""

from __future__ import annotations

import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering.advanced_editor import AdvancedDialog, AdvancedSettingsDialog
from xdart.gui.widgets.controls_panel import FormRow, SubsectionCard


@pytest.fixture
def qapp(_xdart_qt_harness):
    """Use the repository-owned QApplication instead of pytest-qt."""

    return _xdart_qt_harness.app


def test_the_advanced_dialog_holds_a_pages_groups_of_shared_rows_and_closes(
    qapp: QtWidgets.QApplication,
) -> None:
    dialog = AdvancedDialog(title="Advanced — Page", object_name="pageAdvancedDialog")
    try:
        rows = (FormRow(label="Rows", path=("Page", "rows"), value="195", kind="int", minimum=2),
                FormRow(label="Data type", path=("Page", "dtype"), value="int32", kind="combo",
                        choices=("int32", "uint16")))
        card = dialog.add_group("Source", rows)
        chunk = FormRow(label="Chunk", path=("Page", "chunk"), value="8", kind="int", minimum=1)
        card.add_row(chunk)
        other = dialog.add_group("Memory")
        assert isinstance(card, SubsectionCard) and isinstance(other, SubsectionCard)
        assert all(card.isAncestorOf(row) for row in (*rows, chunk)) and dialog.isAncestorOf(other)
        assert dialog.windowTitle() == "Advanced — Page" and dialog.objectName() == "pageAdvancedDialog"
        # a page's rows commit as they are edited: no values to accept, a non-modal Close
        assert not dialog.isModal()
        assert dialog.buttons.standardButtons() == QtWidgets.QDialogButtonBox.StandardButton.Close
        dialog.show()
        qapp.processEvents()
        assert dialog.isVisible()
        dialog.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Close).click()
        qapp.processEvents()
        assert not dialog.isVisible()
    finally:
        dialog.deleteLater()


def test_ints_integration_editor_is_the_same_dialog_modal_with_ok_and_cancel(
    qapp: QtWidgets.QApplication,
) -> None:
    dialog = AdvancedSettingsDialog()
    try:
        assert isinstance(dialog, AdvancedDialog) and dialog.isModal()
        assert dialog.buttons.standardButtons() == (QtWidgets.QDialogButtonBox.StandardButton.Ok
                                                    | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        assert dialog.windowTitle() == "Integration — Advanced Settings"
        assert dialog.objectName() == "vnextAdvancedSettingsDialog"
        # Int's own groups keep their editors, in Int's order, above the buttons
        for group in (dialog.gi_group, dialog.poni_v3_group, dialog.corrections_group):
            assert dialog.isAncestorOf(group)
    finally:
        dialog.deleteLater()
