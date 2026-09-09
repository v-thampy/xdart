"""Real Viewer page selection agrees with the resident scientific traces."""

import time

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets

from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent


def _settle(app, seconds=0.15):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)


def _ready(app, page):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        app.processEvents()
        context = page._context_controller.viewer_1d_context
        if context is not None and context.state.value == "ready":
            _settle(app)
            return
        time.sleep(0.005)
    pytest.fail("real Viewer loading did not settle")


@pytest.fixture
def viewer(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    paths = tuple(str(tmp_path / f"curve_{index}.xye") for index in range(7))
    for index, path in enumerate(paths):
        np.savetxt(path, np.column_stack((np.arange(3),
            np.arange(3) + 10 * (index + 1), np.ones(3))))
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(processing_mode="1D Viewer",
                                       project_root=str(tmp_path))),
        lifecycle=ScatteringCoordinator(), sources=FilesystemSourceAdapter(),
    )
    page.resize(1400, 1000)
    page.show()
    page._set_browser_directory(str(tmp_path), explicit=True)
    _settle(app)
    page._open_viewer_1d_paths(paths[:6], current_path=paths[5])
    _ready(app, page)
    yield app, page, paths
    for _ in range(100):
        receipt = page.close_workspace()
        if receipt.cleanup_status is CleanupStatus.CLEANED:
            break
        _settle(app, 0.01)
    assert receipt.cleanup_status is CleanupStatus.CLEANED
    page.deleteLater()
    _settle(app, 0.01)


def _mode(app, page, mode):
    page._handle_shell_command(ShellCommand(ShellCommandKind.SET_PLOT_MODE, mode))
    _settle(app)


def _mounted(page):
    view = page._shell.scientific
    return {key: view._curve_items_by_key[key] for key in view._curve_mounted_keys}


def _assert_curves(page, count):
    navigation = page._context_controller.navigation
    assert len(navigation.selected) == count
    assert tuple(trace.frame for trace in page._last_scientific_projection.traces) == navigation.selected
    assert len(_mounted(page)) == count
    assert len(page._shell.scientific.curve.getPlotItem().listDataItems()) == count
    for trace, item in zip(page._last_scientific_projection.traces,
                           _mounted(page).values(), strict=True):
        x, y = item.getData()
        np.testing.assert_array_equal(x, trace.axis.values)
        # Presentation offset may differ between modes; the distinct per-file
        # scientific values and within-curve differences must remain exact.
        np.testing.assert_allclose(np.diff(y), np.diff(trace.intensity))
        index = next(index for index, frame in enumerate(navigation.frames)
                     if frame is trace.frame)
        path = page._context_controller.viewer_1d_context.paths[index]
        np.testing.assert_array_equal(trace.intensity, np.loadtxt(path)[:, 1])


def test_real_single_explicit_selection_survives_mode_switch(viewer):
    app, page, _paths = viewer
    _assert_curves(page, 6)
    original_items = _mounted(page)
    browser = page._shell.browser
    first = browser.frame_model.index(0, 0)
    QtTest.QTest.mouseClick(browser.frames.viewport(), QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.ControlModifier, browser.frames.visualRect(first).center())
    _settle(app)
    _assert_curves(page, 5)
    for mode in ("Overlay", "Single"):
        _mode(app, page, mode)
        _assert_curves(page, 5)
        assert all(original_items[key] is item for key, item in _mounted(page).items())


def test_real_overlay_select_all_commits_all_highlighted_rows(viewer):
    app, page, _paths = viewer
    browser = page._shell.browser
    last = browser.frame_model.index(5, 0)
    QtTest.QTest.mouseClick(browser.frames.viewport(), QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier, browser.frames.visualRect(last).center())
    _settle(app)
    _mode(app, page, "Overlay")
    _assert_curves(page, 1)
    browser.frames.setFocus()
    QtTest.QTest.keyClick(browser.frames, QtCore.Qt.Key.Key_A,
                        QtCore.Qt.KeyboardModifier.ControlModifier)
    _settle(app)
    assert len(browser.frames.selectionModel().selectedRows()) == 6
    _assert_curves(page, 6)


def test_resident_artifact_subset_reuses_batch_and_new_file_keeps_clear_fence(viewer, monkeypatch):
    app, page, paths = viewer
    _mode(app, page, "Overlay")
    owner = page._context_controller._viewer_1d
    batch, holder, borrow_id = owner.batch_identity, owner.holder, id(owner.holder.borrow)
    reads = owner.provider.counters()
    original_frames = page._context_controller.navigation.frames
    original_item_ids = {key: id(item) for key, item in _mounted(page).items()}
    observed = []
    clear = page._shell.scientific.clear_viewer_1d
    acknowledge = page._context_controller.acknowledge_viewer_1d_renderer_clear

    def observed_clear(request, **kwargs):
        assert owner.holder is holder and id(holder.borrow) == borrow_id
        observed.append("clear")
        return clear(request, **kwargs)

    def observed_acknowledge(receipt):
        assert owner.holder is holder and id(holder.borrow) == borrow_id
        assert not page._shell.scientific.trace_history_keys
        observed.append("acknowledge")
        return acknowledge(receipt)

    monkeypatch.setattr(page._shell.scientific, "clear_viewer_1d", observed_clear)
    monkeypatch.setattr(page._context_controller, "acknowledge_viewer_1d_renderer_clear",
                        observed_acknowledge)
    page._handle_shell_command(ShellCommand(ShellCommandKind.SELECT_SCAN, paths[4],
        path=("artifact",), artifacts=paths[:5]))
    _assert_curves(page, 5)
    assert observed == []
    assert owner.batch_identity is batch and owner.holder is holder
    assert id(holder.borrow) == borrow_id and owner.provider.counters() == reads
    assert page._context_controller.navigation.frames is original_frames
    assert all(original_item_ids[key] == id(item) for key, item in _mounted(page).items())
    page._handle_shell_command(ShellCommand(ShellCommandKind.SELECT_SCAN, paths[6],
        path=("artifact",), artifacts=paths))
    _ready(app, page)
    assert observed == ["clear", "acknowledge"]
    assert holder.borrow is None and owner.batch_identity is not batch
    _assert_curves(page, 7)
    monkeypatch.undo()


def test_overlay_file_visits_keep_previous_curves(viewer):
    app, page, paths = viewer
    scans = page._shell.browser.scans

    def click(path, modifiers=QtCore.Qt.KeyboardModifier.NoModifier):
        item = next(scans.item(row) for row in range(scans.count())
                    if scans.item(row).data(QtCore.Qt.ItemDataRole.UserRole) == path)
        scans.scrollToItem(item)
        app.processEvents()
        QtTest.QTest.mouseClick(scans.viewport(), QtCore.Qt.MouseButton.LeftButton,
            modifiers, scans.visualItemRect(item).center())
        _ready(app, page)

    click(paths[0])
    _assert_curves(page, 1)
    _mode(app, page, "Overlay")
    click(paths[1])
    _assert_curves(page, 2)
    # A previously unopened file uses the real reader/clear fence too.
    click(paths[6])
    assert len(page._context_controller.navigation.selected) == 3
    assert {trace.title for trace in page._last_scientific_projection.traces} == {
        f"curve_{index}.xye" for index in (0, 1, 6)}
    assert len(_mounted(page)) == 3
    click(paths[0], QtCore.Qt.KeyboardModifier.ControlModifier)
    _assert_curves(page, 2)
    assert {trace.title for trace in page._last_scientific_projection.traces} == {
        "curve_1.xye", "curve_6.xye"}


@pytest.mark.parametrize("mode", ("Single", "Overlay"))
def test_viewer_dense_selection_uses_shared_waterfall_threshold(viewer, tmp_path, mode):
    app, page, paths = viewer
    extra = tuple(str(tmp_path / f"dense_{index}.xye") for index in range(9))
    for index, path in enumerate(extra):
        np.savetxt(path, np.column_stack((np.arange(3),
            np.arange(3) + 100 + index, np.ones(3))))
    paths = paths + extra
    _mode(app, page, mode)
    page._open_viewer_1d_paths(paths[:15])
    _ready(app, page)
    assert not page._shell.scientific.bottom_waterfall_active
    page._open_viewer_1d_paths(paths)
    _ready(app, page)
    assert len(page._last_scientific_projection.traces) == 16
    assert page._shell.scientific.bottom_waterfall_active
    assert page._shell.scientific.bottom_stack.currentWidget() is page._shell.scientific.waterfall
    page._open_viewer_1d_paths(paths[:7])
    _ready(app, page)
    assert not page._shell.scientific.bottom_waterfall_active
    assert len(_mounted(page)) == 7


@pytest.mark.parametrize("mode", ("Single", "Overlay"))
def test_selected_viewer_files_keep_conflicting_unit_refusal(viewer, tmp_path, mode):
    app, page, paths = viewer
    mixed = tmp_path / "degrees.npz"
    np.savez(mixed, x=np.arange(3.0), y=np.arange(3.0),
             x_unit=np.asarray("degrees"))
    _mode(app, page, mode)
    page._open_viewer_1d_paths((paths[0], str(mixed)), current_path=paths[0])
    _ready(app, page)
    assert len(page._context_controller.navigation.selected) == 2
    assert page._last_scientific_projection.traces == ()
    assert "conflicting units" in page._last_scientific_projection.status
    assert not _mounted(page)
    page._open_viewer_1d_paths((paths[0],), current_path=paths[0])
    _assert_curves(page, 1)
