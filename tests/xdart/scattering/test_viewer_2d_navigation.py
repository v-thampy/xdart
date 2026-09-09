"""Real Qt 2-D viewer navigation and bounded same-artifact retirement."""

from __future__ import annotations

import time
from threading import Event

import h5py
import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets

from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent


def _wait(page, qapp, predicate):
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline:
        qapp.processEvents()
        page._drain_executor()
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("real viewer did not settle")


@pytest.fixture
def viewer(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    values = np.arange(8 * 12 * 16, dtype=np.uint16).reshape(8, 12, 16)
    paths = (tmp_path / "a.h5", tmp_path / "c.h5")
    for path in paths:
        with h5py.File(path, "w") as handle:
            handle.create_dataset("entry/data/data", data=values)
    (tmp_path / "b_folder").mkdir()
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(
            processing_mode="2D Viewer", project_root=str(tmp_path),
            save_path=str(tmp_path),
        )),
        lifecycle=ScatteringCoordinator(), sources=FilesystemSourceAdapter(),
    )
    page.resize(1200, 800)
    page.show()
    page._set_browser_directory(str(tmp_path), explicit=True)
    page._open_viewer_2d_path(str(paths[0]))
    _wait(page, app, lambda: (
        page._context_controller.viewer_2d_frame is not None
        and page._shell.scientific._viewer_2d_payload is not None
        and page._shell.browser.scans.count() == 4
    ))
    try:
        yield page, app, paths, values
    finally:
        page.close_workspace()
        app.processEvents()
        page.deleteLater()
        app.processEvents()


def _row(browser, path):
    return next(index for index in range(browser.scans.count())
                if browser.scans.item(index).data(QtCore.Qt.UserRole) == str(path))


def test_hdf_viewer_left_frames_and_footer_share_current_file_and_labels(viewer):
    page, app, paths, values = viewer
    browser, view = page._shell.browser, page._shell.scientific
    controller = page._context_controller
    assert browser.frame_model.rowCount() == len(values)
    assert view.frame_selector.count() == len(values)
    assert browser.scans.currentItem().data(QtCore.Qt.UserRole) == str(paths[0])
    assert tuple(browser.frame_model.frames) == controller.navigation.frames
    QtTest.QTest.mouseClick(view.next_frame, QtCore.Qt.LeftButton)
    _wait(page, app, lambda: controller.viewer_2d_frame is not None
          and controller.viewer_2d_frame.label == 1
          and view._viewer_2d_payload is controller.viewer_2d_frame.array)
    assert browser.frames.currentIndex().data(QtCore.Qt.UserRole) is controller.navigation.current
    assert view.frame_selector.currentData() is controller.navigation.current
    assert browser.frame_model.rowCount() == len(values)
    np.testing.assert_array_equal(controller.viewer_2d_frame.array, values[1])


def test_file_arrow_keys_skip_directories_without_entering_them(viewer):
    page, app, paths, _values = viewer
    browser = page._shell.browser
    # Explicitly establish the initial row so this oracle isolates keyboard
    # navigation even while the separate current-artifact projection is broken.
    blocker = QtCore.QSignalBlocker(browser.scans)
    browser.scans.setCurrentRow(_row(browser, paths[0]))
    del blocker
    browser.scans.setFocus()
    QtTest.QTest.keyClick(browser.scans, QtCore.Qt.Key_Down)
    _wait(page, app, lambda: not page._context_controller.viewer_2d_loading)
    assert page._processed_browser.projection().directory == str(paths[0].parent)
    assert page._context_controller.viewer_2d_context.original_path == str(paths[1])
    _wait(page, app, lambda: browser.scans.currentItem() is not None
          and browser.scans.currentItem().data(QtCore.Qt.UserRole) == str(paths[1]))
    QtTest.QTest.keyClick(browser.scans, QtCore.Qt.Key_Up)
    _wait(page, app, lambda: not page._context_controller.viewer_2d_loading)
    assert page._context_controller.viewer_2d_context.original_path == str(paths[0])
    assert page._processed_browser.projection().directory == str(paths[0].parent)


@pytest.mark.parametrize("gesture", ["click", "double-click", "enter"])
def test_directory_activation_remains_explicit_and_works(viewer, gesture):
    page, app, paths, _values = viewer
    browser = page._shell.browser
    folder = paths[0].parent / "b_folder"
    row = _row(browser, folder)
    if gesture == "enter":
        blocker = QtCore.QSignalBlocker(browser.scans)
        browser.scans.setCurrentRow(row)
        del blocker
        QtTest.QTest.keyClick(browser.scans, QtCore.Qt.Key_Return)
    else:
        point = browser.scans.visualItemRect(browser.scans.item(row)).center()
        click = (QtTest.QTest.mouseClick if gesture == "click"
                 else QtTest.QTest.mouseDClick)
        click(browser.scans.viewport(), QtCore.Qt.LeftButton, pos=point)
    _wait(page, app, lambda: page._processed_browser.projection().directory == str(folder))


def test_same_hdf_frame_step_retires_arrays_but_keeps_viewer_chrome(viewer, monkeypatch):
    import xdart.gui.tabs.scattering.hydration_transport as hydration

    page, app, _paths, values = viewer
    view = page._shell.scientific
    controller = page._context_controller
    entered, release = Event(), Event()
    read = hydration.read_viewer_2d_frame

    def gated_read(catalog, label, **kwargs):
        if label == 1:
            entered.set()
            assert release.wait(6)
        return read(catalog, label, **kwargs)

    monkeypatch.setattr(hydration, "read_viewer_2d_frame", gated_read)
    image_item = view.raw.image
    frame_items = tuple(view.frame_selector.itemData(i) for i in range(view.frame_selector.count()))
    selector_operations = view._selector_operations
    view.raw.canvas.imageViewBox.setRange(xRange=(3, 10), yRange=(2, 8), padding=0)
    # Keep the operator's requested rectangle. Locked aspect ratio expands
    # the visible range when histogram tick labels change the viewport width.
    prior_target = view.raw.canvas.imageViewBox.targetRect()
    try:
        QtTest.QTest.mouseClick(view.next_frame, QtCore.Qt.LeftButton)
        _wait(page, app, entered.is_set)
        assert controller.viewer_2d_frame is None
        assert view._viewer_2d_payload is None
        assert view.raw.canvas.raw_image.size == view.raw.canvas.displayed_image.size == 0
        assert image_item.image is None
        assert image_item.qimage is None
        assert view.raw.isVisible() and view.image_splitter.isVisible()
        assert tuple(view.frame_selector.itemData(i) for i in range(view.frame_selector.count())) == frame_items
        assert view._selector_operations == selector_operations
        assert view.raw.canvas.imageViewBox.targetRect() == prior_target
        release.set()
        _wait(page, app, lambda: controller.viewer_2d_frame is not None
              and controller.viewer_2d_frame.label == 1
              and view._viewer_2d_payload is controller.viewer_2d_frame.array)
        assert view.raw.image is image_item
        assert view._selector_operations == selector_operations
        assert view.raw.canvas.imageViewBox.targetRect() == prior_target
        np.testing.assert_array_equal(controller.viewer_2d_frame.array, values[1])
    finally:
        release.set()


def test_cancelled_frame_read_releases_before_subsequent_viewer_load(
        viewer, monkeypatch):
    import xdart.gui.tabs.scattering.hydration_transport as hydration

    page, app, paths, values = viewer
    controller = page._context_controller
    entered, release = Event(), Event()
    read = hydration.read_viewer_2d_frame

    def gated_read(catalog, label, **kwargs):
        if label == 1:
            entered.set()
            assert release.wait(6)
        return read(catalog, label, **kwargs)

    monkeypatch.setattr(hydration, "read_viewer_2d_frame", gated_read)
    try:
        QtTest.QTest.mouseClick(
            page._shell.scientific.next_frame, QtCore.Qt.LeftButton,
        )
        _wait(page, app, entered.is_set)
        prior_gate = controller.viewer_2d_context.commit_gate
        assert controller.viewer_2d_frame is None
        assert not controller.close_viewer_2d()
        assert prior_gate.cancelled
        release.set()
        _wait(page, app, controller.close_viewer_2d)

        page._open_viewer_2d_path(str(paths[1]))
        _wait(page, app, lambda: (
            controller.viewer_2d_frame is not None
            and controller.viewer_2d_context.original_path == str(paths[1])
            and page._shell.scientific._viewer_2d_payload
            is controller.viewer_2d_frame.array
        ))
        np.testing.assert_array_equal(controller.viewer_2d_frame.array, values[0])
    finally:
        release.set()


def test_cancelled_prepared_frame_releases_before_subsequent_viewer_load(
        viewer, monkeypatch):
    page, app, paths, values = viewer
    controller = page._context_controller
    entered, release = Event(), Event()
    commit = controller._viewer_2d.commit

    def gated_commit(owner, prepared):
        assert owner is controller._viewer_2d
        entered.set()
        assert release.wait(6)
        return commit(prepared)

    monkeypatch.setattr(type(controller._viewer_2d), "commit", gated_commit)
    try:
        QtTest.QTest.mouseClick(
            page._shell.scientific.next_frame, QtCore.Qt.LeftButton,
        )
        _wait(page, app, entered.is_set)
        prior_gate = controller.viewer_2d_context.commit_gate
        assert controller.viewer_2d_frame is None
        assert not controller.close_viewer_2d()
        assert prior_gate.cancelled
        release.set()
        _wait(page, app, controller.close_viewer_2d)

        page._open_viewer_2d_path(str(paths[1]))
        _wait(page, app, lambda: (
            controller.viewer_2d_frame is not None
            and controller.viewer_2d_context.original_path == str(paths[1])
            and page._shell.scientific._viewer_2d_payload
            is controller.viewer_2d_frame.array
        ))
        np.testing.assert_array_equal(controller.viewer_2d_frame.array, values[0])
    finally:
        release.set()
