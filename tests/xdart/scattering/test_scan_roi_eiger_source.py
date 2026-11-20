"""NEXT-1217-EIGER-SOURCES through Int's ROI Statistics window (maintainer, 2026-11-20: "I'm trying to open an Eiger h5
file the to plot ROI stats for that and it doesn't work"). A DECTRIS Eiger master picked as a file, or from a Folder
as a raw NeXus stack, opens, previews and computes; real page, real operation slot, real file."""
import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.analysis.scan_source_widget import _DIR_KINDS
from tests.core.test_container_descriptor import _vendor_eiger_master
from tests.xdart.scattering.test_p37_analysis_mount import _close_page, _page
from tests.xdart.scattering.test_roi_combined_window import _settle


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _master(folder):
    return _vendor_eiger_master(folder / "scan_master.h5", [folder / "scan_data_000001.h5"])


def _opens_previews_and_computes(page, dialog, qapp):
    table = dialog._vnext_table_result
    assert table is not None, dialog.status.text()
    assert table.receipt.resolved_kind.value == "eiger_master" and len(table.labels) == 5
    assert dialog.roi_btn.isEnabled()
    dialog.roi_btn.click(); _settle(page, qapp)
    picker = dialog._roi_dialog
    assert picker is not None and picker._image is not None and picker._image.shape == (8, 8)
    picker.compute_btn.click(); _settle(page, qapp)
    result = dialog._vnext_roi_result
    assert result is not None and result.completed_labels == table.labels
    assert "roi1" in dialog._selector_columns


def test_an_eiger_master_picked_as_a_file_opens_previews_and_computes(tmp_path, qapp):
    master = _master(tmp_path)
    page = _page()
    try:
        for _opening in range(2):  # and again after the window is closed
            page._open_analysis_mount("scan_roi")
            dialog = page._scan_roi_dialog
            dialog.load_uri(str(master)); _settle(page, qapp)
            _opens_previews_and_computes(page, dialog, qapp)
            dialog.close(); qapp.processEvents()
            assert page._scan_roi_dialog is None
    finally:
        _close_page(page, qapp)


def test_an_eiger_master_picked_from_a_folder_as_a_raw_stack_opens_previews_and_computes(tmp_path, qapp):
    _master(tmp_path)
    page = _page()
    try:
        page._open_analysis_mount("scan_roi")
        dialog = page._scan_roi_dialog
        widget = dialog.source_widget
        widget.dir_check.setChecked(True)
        widget.dir_kind_combo.setCurrentIndex([kind for _label, kind in _DIR_KINDS].index("nexus_stack"))
        widget.path_edit.setText(str(tmp_path)); widget._refresh_candidates(); _settle(page, qapp)  # as Choose does
        assert widget.scan_combo.count() == 2, dialog.status.text()  # "Choose a scan…" and the master
        widget.scan_combo.setCurrentIndex(1); _settle(page, qapp)
        _opens_previews_and_computes(page, dialog, qapp)
    finally:
        _close_page(page, qapp)
