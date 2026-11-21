"""NEXT-1217-EIGER-SOURCES through Int's ROI Statistics window (maintainer, 2026-11-20: "I'm trying to open an Eiger h5
file the to plot ROI stats for that and it doesn't work"; his picks "Always exclude gap value (Recommended)" and "ROI
window shows words"). A DECTRIS Eiger master picked as a file, or a NeXus container from a Folder, opens, previews and
computes; Eiger module gaps never enter the statistics; refusals read as words. Real page, slot and files."""
import h5py
import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from xrd_tools.core.invalid import UINT32_CEILING
from xrd_tools.core.scan import SourceKind, SourceSpec
from xdart.gui.analysis.scan_source_widget import _DIR_KINDS
from tests.core.test_container_descriptor import _chunked_stack, _vendor_eiger_master
from tests.xdart.scattering.test_p37_analysis_mount import _close_page, _page
from tests.xdart.scattering.test_roi_combined_window import _settle


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _master(folder, **kwargs):
    return _vendor_eiger_master(folder / "scan_master.h5", [folder / "scan_data_000001.h5"], **kwargs)


def _opens_previews_and_computes(page, dialog, qapp, kind="eiger_master", frames=5, shape=(8, 8)):
    table = dialog._vnext_table_result
    assert table is not None, dialog.status.text()
    assert table.receipt.resolved_kind.value == kind and len(table.labels) == frames
    assert dialog.roi_btn.isEnabled()
    dialog.roi_btn.click(); _settle(page, qapp)
    picker = dialog._roi_dialog
    assert picker is not None and picker._image is not None and picker._image.shape == shape
    picker.compute_btn.click(); _settle(page, qapp)
    result = dialog._vnext_roi_result
    assert result is not None and result.completed_labels == table.labels
    assert "roi1" in dialog._selector_columns
    return result


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


@pytest.mark.parametrize("container, folder_kind", [("eiger", "nexus_stack"), ("eiger", "eiger_master"),
                                                     ("plain", "nexus_stack")])
def test_a_container_picked_from_a_folder_opens_previews_and_computes(tmp_path, qapp, container, folder_kind):
    """A Folder's candidates leave the NeXus entry blank; the receipt names the entry (and an Eiger's kind)."""
    if container == "eiger":
        _master(tmp_path)
        expected = dict(kind="eiger_master", frames=5, shape=(8, 8))
    else:
        _chunked_stack(tmp_path / "plain.nxs")
        expected = dict(kind="nexus_stack", frames=8, shape=(16, 16))
    page = _page()
    try:
        page._open_analysis_mount("scan_roi")
        dialog = page._scan_roi_dialog
        widget = dialog.source_widget
        widget.dir_check.setChecked(True)
        widget.dir_kind_combo.setCurrentIndex([kind for _label, kind in _DIR_KINDS].index(folder_kind))
        widget.path_edit.setText(str(tmp_path)); widget._refresh_candidates(); _settle(page, qapp)  # as Choose does
        assert widget.scan_combo.count() == 2, dialog.status.text()  # "Choose a scan…" and the container
        widget.scan_combo.setCurrentIndex(1); _settle(page, qapp)
        _opens_previews_and_computes(page, dialog, qapp, **expected)
    finally:
        _close_page(page, qapp)


def test_eiger_module_gaps_never_enter_the_roi_statistics(tmp_path, qapp):
    """Mask saturated off (its default): the uint32 invalid value of the module gaps is still left out."""
    master = _master(tmp_path, dtype=np.uint32)
    with h5py.File(tmp_path / "scan_data_000001.h5", "r+") as handle:
        frames = np.ones((5, 8, 8), dtype=np.uint32)
        frames[:, 4, :] = frames[:, :, 4] = int(UINT32_CEILING)  # a horizontal and a vertical gap
        handle["entry/data/data"][...] = frames
    page = _page()
    try:
        page._open_analysis_mount("scan_roi")
        dialog = page._scan_roi_dialog
        dialog.load_uri(str(master)); _settle(page, qapp)
        dialog.roi_btn.click(); _settle(page, qapp)
        assert not dialog._roi_dialog.mask_saturated()
        result = _opens_previews_and_computes(page, dialog, qapp)
        np.testing.assert_array_equal(result.signal_values[0], np.ones(5))  # the whole-frame mean of real pixels
        np.testing.assert_array_equal(result.valid_counts[0], np.full(5, 64 - 15))
        assert "always" in dialog._roi_dialog.mask_sat_check.toolTip()
    finally:
        _close_page(page, qapp)


def test_a_refused_source_reads_as_words_in_the_roi_window(tmp_path, qapp):
    page = _page()
    try:
        page._open_analysis_mount("scan_roi")
        dialog = page._scan_roi_dialog
        dialog.load_uri(str(tmp_path / "missing_master.h5")); _settle(page, qapp)
        assert dialog._vnext_table_result is None
        assert dialog.status.text().startswith("Required source content could not be found or accessed.")
        assert "SOURCE_NOT_FOUND" not in dialog.status.text()
        assert dialog._help_current_diagnostic is not None  # the code stays in the window's Help diagnostics
    finally:
        _close_page(page, qapp)


@pytest.mark.parametrize("with_spec", [False, True], ids=["direct-master", "spec"])
def test_the_roi_picker_scales_the_image_to_real_pixels_not_eiger_gaps(tmp_path, qapp, with_spec):
    """The picker's image levels come from real pixels: an Eiger's gap value (4294967295) shows as missing, so the
    frame is not drawn black under it (display only; the statistics read the raw frame)."""
    import pyqtgraph as pg
    master = _master(tmp_path, dtype=np.uint32)
    with h5py.File(tmp_path / "scan_data_000001.h5", "r+") as handle:
        frames = np.tile(np.arange(64, dtype=np.uint32).reshape(8, 8), (5, 1, 1))
        frames[:, 4, :] = int(UINT32_CEILING)
        handle["entry/data/data"][...] = frames
    page = _page()
    try:
        page._open_analysis_mount("scan_roi")
        dialog = page._scan_roi_dialog
        if with_spec:
            from tests.core.test_rsm_sources import _pair
            from xrd_tools.analysis.scan_operations import _candidate
            spec = _pair(tmp_path, with_images=(), scans=(1,))
            candidate = _candidate(SourceSpec(
                spec, SourceKind.SPEC,
                options={"scan": "1.1", "image_dir": str(tmp_path), "image_stem": master.stem},
            ))
            dialog.source_widget.set_external_candidates((candidate,))
            dialog.source_widget.scan_combo.setCurrentIndex(1)
        else:
            dialog.load_uri(str(master))
        _settle(page, qapp)
        assert dialog.roi_btn.isEnabled(), dialog.status.text()
        dialog.roi_btn.click(); _settle(page, qapp)
        picker = dialog._roi_dialog
        assert picker._image.dtype == (np.float64 if with_spec else np.uint32)
        assert (picker._image == int(UINT32_CEILING)).sum() == 8
        images = [item for view in picker.findChildren(pg.GraphicsView) for item in view.items()
                  if isinstance(item, pg.ImageItem) and item.image is not None]
        assert images and all(float(item.getLevels()[1]) <= 63 for item in images), [i.getLevels() for i in images]
    finally:
        _close_page(page, qapp)


def _browse(page, qapp, path):
    import time
    page._select_scan(str(path), reopen=True)
    deadline = time.monotonic() + 30
    while page._context_controller.navigation.current is None and time.monotonic() < deadline:
        qapp.processEvents(); page._drain_executor(); time.sleep(0.01)
    assert page._context_controller.navigation.current is not None


def test_roi_statistics_opens_on_the_scan_selected_in_the_data_browser(tmp_path, qapp):
    """Maintainer, 2026-11-20: "how quick and easy is it for the ROI statistics window to open with the currently
    selected file in the data browser if the selection is a valid one?"; picks "Fold into 1.2.17" and "On open only
    (Recommended)". The Tools button opens the window on the browsed scan; a file picked in the window stays."""
    from tests.xdart.scattering.test_p37_analysis_mount import _roi_navigation_table
    other = tmp_path / "other"
    other.mkdir()
    _roi_navigation_table(tmp_path)  # processed scans whose frame records reach their raw images
    _roi_navigation_table(other)
    page = _page()
    try:
        _browse(page, qapp, tmp_path / "navigation.nexus")
        button = page.findChild(QtWidgets.QPushButton, "e3Tool_roi_statistics")
        assert "Data Browser" in button.toolTip()
        button.click(); _settle(page, qapp)
        dialog = page._scan_roi_dialog
        assert dialog.source_widget.path_edit.text() == str(tmp_path / "navigation.nexus")
        table = dialog._vnext_table_result
        assert table is not None and table.labels == (3, 7, 11), dialog.status.text()
        assert dialog.roi_btn.isEnabled()
        dialog.load_uri(str(other / "navigation.nexus")); _settle(page, qapp)
        button.click(); _settle(page, qapp)  # raised again: on open only
        assert page._scan_roi_dialog is dialog
        assert dialog.source_widget.path_edit.text() == str(other / "navigation.nexus")
    finally:
        _close_page(page, qapp)


def test_roi_statistics_opens_empty_without_a_browsed_scan(qapp):
    page = _page()
    try:
        page.findChild(QtWidgets.QPushButton, "e3Tool_roi_statistics").click(); _settle(page, qapp)
        dialog = page._scan_roi_dialog
        assert dialog._vnext_table_result is None and dialog.source_widget.path_edit.text() == ""
    finally:
        _close_page(page, qapp)
