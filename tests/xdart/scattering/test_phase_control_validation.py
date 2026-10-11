"""Direct shared-owner checks for disabled phase help and refused floor drafts."""

import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.analysis.phase_source import PhaseSourceWidget
from xdart.gui.tabs.scattering.scientific_plot_options import PlotOptionsDialog
from xrd_tools.analysis.phase_references import PhaseLibrary


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_disabled_phase_reason_survives_resolution_on_every_shared_control(qapp, tmp_path):
    source = PhaseSourceWidget(library=PhaseLibrary(tmp_path / "library"))
    controls = [source, source.browser, source.browser.scans, source.browser.scans_label,
                source.browser.show_all, source.browser.metadata, source.browser.auto_last,
                source.browser.date_sort, source.browser.refresh, source.add_cif_btn,
                source.add_lattice_btn, source.remove_btn, source.lattice_form,
                source.add_lattice_commit, source.status]
    for row in source.lattice_rows.values():
        controls.extend((row, row.label, row.editor))
    purposes = {control: control.toolTip() for control in controls}
    reason = "Phase sources need pymatgen; install xdart[fitting]."
    try:
        source.set_available(False, reason)
        assert source.selected_entries() == ()
        for control in controls:
            assert reason in control.toolTip(), (control.objectName(), control.toolTip())
            assert purposes[control] in control.toolTip()
        assert not source.isEnabled()
        source.set_available(True)
        assert source.selected_entries() == ()
        for control in controls:
            assert control.toolTip() == purposes[control]
    finally:
        source.close()
        source.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("draft", ("", "nan", "inf", "-1", "101"))
def test_plot_options_keep_invalid_phase_floor_open_with_reason_then_accept_valid_edit(qapp, draft):
    dialog = PlotOptionsDialog()
    accepted = []
    dialog.accepted.connect(lambda: accepted.append(dialog.edited_options()))
    original = dialog.projection
    try:
        dialog.show()
        dialog.phase_floor.editor.setText(draft)
        dialog.accept_button.click()
        assert dialog.isVisible() and accepted == []
        assert dialog.projection is original
        assert dialog.phase_floor.editor.text() == draft
        reason = dialog.phase_floor.editor.toolTip().lower()
        assert "finite" in reason and "0" in reason and "100" in reason
        assert dialog.phase_floor.label.toolTip() == dialog.phase_floor.editor.toolTip()
        dialog.phase_floor.editor.setText("12.5")
        dialog.accept_button.click()
        assert not dialog.isVisible() and len(accepted) == 1
        assert accepted[0].phase_min_intensity == 12.5
        assert "finite" not in dialog.phase_floor.editor.toolTip().lower()
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("floor", ("0", "100"))
def test_plot_options_accept_both_phase_floor_boundaries(qapp, floor):
    dialog = PlotOptionsDialog()
    accepted = []
    dialog.accepted.connect(lambda: accepted.append(dialog.edited_options()))
    try:
        dialog.show()
        dialog.phase_floor.editor.setText(floor)
        dialog.accept_button.click()
        assert not dialog.isVisible() and len(accepted) == 1
        assert accepted[0].phase_min_intensity == float(floor)
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()
