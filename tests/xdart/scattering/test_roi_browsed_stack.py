"""NEXT-1219-ROI-BROWSED-SOURCE: ROI Statistics opens on the browsed stack in the 2D Viewer, and a stack without
metadata goes straight to the Plot ROI picker.

His words (2026-11-23): "the ROI statistics button doesn't currently select the selected eiger files automatically when
clicked. does that only apply to selected nexus files? I'm thinking for selected image stacks like an eiger file without
metadata, it makes more sense to open the plot roi window directly"; "it doesn't automatically load the selected nexus
file either". His picks: "Both windows follow" (the Metadata window too) and "Any stack w/o metadata (Recommended)".
Real page, owners, worker threads and files; until-loops, never a fixed settle.
"""
from __future__ import annotations

import time

import h5py
import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from tests.core.test_container_descriptor import _vendor_eiger_master
from tests.xdart.scattering.test_p37_analysis_mount import _roi_navigation_table
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def open_page(tmp_path, qapp):
    pages = []

    def make(mode):
        page = ScatteringWorkspace(
            intents=RunIntentStore(RunIntent(
                processing_mode=mode, project_root=str(tmp_path), save_path=str(tmp_path))),
            lifecycle=ScatteringCoordinator(), sources=FilesystemSourceAdapter())
        page.resize(1200, 800)
        page.show()
        page._set_browser_directory(str(tmp_path), explicit=True)
        pages.append(page)
        return page

    yield make
    for page in pages:
        page.close_workspace()
        qapp.processEvents()
        page.deleteLater()
        qapp.processEvents()


def _until(page, qapp, predicate, what, seconds=30):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        qapp.processEvents()
        page._drain_executor()
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError(f"{what} did not happen; status: {getattr(page._scan_roi_dialog, 'status', None) and page._scan_roi_dialog.status.text()!r}")


def _idle(page):
    return (not page._metadata_operations.busy and page._metadata_operations.deferred is None
            and page._analysis_identity is None and not page._analysis_slot.owned)


def _settled(page, qapp, what):
    """Idle, and still idle after the zero-delay events a finished table may post."""
    _until(page, qapp, lambda: _idle(page), what)
    for _ in range(5):
        qapp.processEvents()
        page._drain_executor()
    _until(page, qapp, lambda: _idle(page), what)


def _browse(page, qapp, path, *, viewer):
    page._handle_shell_command(ShellCommand(ShellCommandKind.SELECT_SCAN, str(path)))
    controller = page._context_controller
    if viewer:
        _until(page, qapp, lambda: (
            controller.viewer_2d_context is not None
            and controller.viewer_2d_context.original_path == str(path)
            and controller.viewer_2d_frame is not None
            and controller.navigation.current is not None), f"browsing {path.name}")
    else:
        _until(page, qapp, lambda: controller.navigation.current is not None, f"browsing {path.name}")


def _roi_statistics(page, qapp):
    button = page.findChild(QtWidgets.QPushButton, "e3Tool_roi_statistics")
    button.click()
    qapp.processEvents()
    dialog = page._scan_roi_dialog
    assert dialog is not None and dialog.isVisible()
    return dialog


def _picker_shown(dialog):
    picker = dialog._roi_dialog
    return picker is not None and picker._image is not None


def _eiger(folder):
    return _vendor_eiger_master(folder / "scan_master.h5", [folder / "scan_data_000001.h5"])


def _processed(folder, *, metadata):
    """A processed NeXus whose frame records reach their raw TIFFs, labels (3, 7, 11); optionally a scan_data column."""
    _roi_navigation_table(folder)
    path = folder / "navigation.nexus"
    if metadata:
        with h5py.File(path, "a") as handle:
            group = handle["entry"].create_group("scan_data")
            group.create_dataset("frame_index", data=np.array([3, 7, 11]))
            group.create_dataset("i0", data=np.array([1.0, 2.0, 3.0]))
    return path


def test_the_2d_viewer_opens_a_browsed_eiger_master_straight_in_the_roi_picker(tmp_path, qapp, open_page):
    master = _eiger(tmp_path)
    page = open_page("2D Viewer")
    _browse(page, qapp, master, viewer=True)
    dialog = _roi_statistics(page, qapp)
    _until(page, qapp, lambda: _picker_shown(dialog) and _idle(page), "the ROI picker opening on the Eiger master")
    assert dialog.source_widget.path_edit.text() == str(master)
    table = dialog._vnext_table_result
    assert table.receipt.resolved_kind.value == "eiger_master" and table.labels == (0, 1, 2, 3, 4)
    assert dialog._roi_dialog._image.shape == (8, 8)
    dialog._roi_dialog.compute_btn.click()
    _until(page, qapp, lambda: dialog._vnext_roi_result is not None and _idle(page), "the ROI statistics")
    assert dialog._vnext_roi_result.completed_labels == table.labels


@pytest.mark.parametrize("kind", ["hdf5", "tiff"])
def test_the_2d_viewer_opens_a_browsed_stack_without_metadata_in_the_roi_picker(tmp_path, qapp, open_page, kind):
    rng = np.random.default_rng(1219)
    if kind == "hdf5":
        path = tmp_path / "stack.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("entry/data/data", data=rng.integers(0, 1000, size=(3, 8, 8), dtype=np.uint16))
    else:
        import tifffile
        path = tmp_path / "frame.tif"
        tifffile.imwrite(path, rng.integers(0, 1000, size=(8, 8), dtype=np.uint16))
    page = open_page("2D Viewer")
    _browse(page, qapp, path, viewer=True)
    dialog = _roi_statistics(page, qapp)
    _until(page, qapp, lambda: _picker_shown(dialog) and _idle(page), f"the ROI picker opening on {path.name}")
    assert dialog.source_widget.path_edit.text() == str(path)
    assert dialog._roi_dialog._image.shape == (8, 8)
    assert set(dialog._columns) <= {"frame_index"}


@pytest.mark.parametrize("metadata", [True, False])
def test_the_2d_viewer_opens_a_browsed_processed_nexus(tmp_path, qapp, open_page, metadata):
    """A processed NeXus browsed in the 2D Viewer ("Processed raw"): with metadata the table shows first, as in Int;
    without, the picker opens."""
    path = _processed(tmp_path, metadata=metadata)
    page = open_page("2D Viewer")
    _browse(page, qapp, path, viewer=True)
    dialog = _roi_statistics(page, qapp)
    if metadata:
        _until(page, qapp, lambda: dialog._vnext_table_result is not None, "the processed table")
        _settled(page, qapp, "the processed table")
        assert dialog.source_widget.path_edit.text() == str(path)
        assert dialog._vnext_table_result.labels == (3, 7, 11) and "i0" in dialog._columns
        assert dialog._roi_dialog is None and dialog.roi_btn.isEnabled()
        dialog.roi_btn.click()
    _until(page, qapp, lambda: _picker_shown(dialog) and _idle(page), "the ROI picker on the processed NeXus")
    assert dialog.source_widget.path_edit.text() == str(path)
    assert dialog._roi_dialog._image.shape == (4, 4)


def test_the_metadata_window_follows_the_browsed_eiger_in_the_2d_viewer(tmp_path, qapp, open_page):
    """His pick "Both windows follow": the Metadata window reads the 2D Viewer's browsed Eiger master too."""
    master = _eiger(tmp_path)
    page = open_page("2D Viewer")
    _browse(page, qapp, master, viewer=True)
    page._shell.browser.metadata.click()
    _until(page, qapp, lambda: page._metadata_result is not None and _idle(page), "the Metadata window's table")
    result = page._metadata_result
    assert result.receipt.resolved_kind.value == "eiger_master" and result.labels == (0, 1, 2, 3, 4)
    assert page._metadata_dialog is not None and page._metadata_dialog.isVisible()


def test_a_file_picked_in_the_window_goes_to_the_picker_only_without_metadata(tmp_path, qapp, open_page):
    """His pick "Any stack w/o metadata (Recommended)": a file picked in the window follows the same rule."""
    with_metadata = _processed(tmp_path, metadata=True)
    master = _eiger(tmp_path)
    page = open_page("Int 1D")
    dialog = _roi_statistics(page, qapp)
    assert dialog.source_widget.path_edit.text() == ""
    dialog.load_uri(str(with_metadata))
    _until(page, qapp, lambda: dialog._vnext_table_result is not None, "the picked processed table")
    _settled(page, qapp, "the picked processed table")
    assert "i0" in dialog._columns and dialog._roi_dialog is None
    dialog.load_uri(str(master))
    _until(page, qapp, lambda: _picker_shown(dialog) and _idle(page), "the ROI picker on the picked Eiger master")
    assert dialog._vnext_table_result.receipt.resolved_kind.value == "eiger_master"


def test_a_file_picked_before_the_browsed_table_arrives_keeps_its_own_table(tmp_path, qapp, open_page):
    """The browsed Eiger's picker step is dropped when another file is picked in the window first."""
    with_metadata = _processed(tmp_path, metadata=True)
    master = _eiger(tmp_path)
    page = open_page("2D Viewer")
    _browse(page, qapp, master, viewer=True)
    dialog = _roi_statistics(page, qapp)
    dialog.load_uri(str(with_metadata))
    _until(page, qapp, lambda: dialog._vnext_table_result is not None, "the picked processed table")
    _settled(page, qapp, "the picked processed table")
    assert dialog.source_widget.path_edit.text() == str(with_metadata)
    assert dialog._vnext_table_result.labels == (3, 7, 11) and "i0" in dialog._columns
    assert dialog._roi_dialog is None
