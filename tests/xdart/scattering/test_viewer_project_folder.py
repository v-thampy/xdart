"""Project Folder remains a workspace edit in standalone viewers."""

import time

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets
import tifffile

from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.controls_projection import PROJECT_ROOT
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.events import (
    CleanupStatus, ExecutorAccepted, FatalExecution, OwnersClosed, PreflightAccepted,
    PreflightRefused,
)
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.widgets.controls_panel import FormRow
from xdart.utils.browse import browse_start_dir
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent


def _wait(app, predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.005)
    assert predicate()


def _row(page):
    return next(row for row in page._shell.controls.findChildren(FormRow)
                if row.path == PROJECT_ROOT)


@pytest.fixture(params=("1D Viewer", "2D Viewer"))
def viewer(request, tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    original = tmp_path / "original"
    picked = tmp_path / "picked"
    original.mkdir()
    picked.mkdir()
    calls = []

    def choose(path, current, start):
        calls.append((path, current, start))
        return str(picked)

    store = RunIntentStore(RunIntent(processing_mode=request.param,
        project_root=str(original), save_path=str(original / "xdart_processed_data")))
    lifecycle = ScatteringCoordinator()
    page = ScatteringWorkspace(intents=store, lifecycle=lifecycle,
        sources=FilesystemSourceAdapter(), control_path_chooser=choose)
    page.resize(1400, 1000)
    page.show()
    app.processEvents()
    yield app, page, store, lifecycle, calls, picked
    _wait(app, lambda: page.close_workspace().cleanup_status is CleanupStatus.CLEANED)
    page.deleteLater()
    app.processEvents()


def _load(app, page, tmp_path):
    if page._intents.snapshot().thaw().processing_mode == "1D Viewer":
        path = tmp_path / "tiny.xye"
        np.savetxt(path, np.column_stack((np.arange(3), np.arange(3) + 10, np.ones(3))))
        page._open_viewer_1d_paths((str(path),))
        _wait(app, lambda: page._context_controller.viewer_1d_context is not None
              and page._context_controller.viewer_1d_context.state.value == "ready")
    else:
        path = tmp_path / "tiny.tif"
        tifffile.imwrite(path, np.arange(64, dtype=np.uint16).reshape(8, 8))
        page._open_viewer_2d_path(str(path))
        _wait(app, lambda: page._context_controller.viewer_2d_frame is not None)
    return page._context_controller.navigation.frames


@pytest.mark.parametrize("loaded", (False, True))
def test_real_viewer_project_folder_edit_and_picker(viewer, tmp_path, loaded):
    app, page, store, _lifecycle, calls, picked = viewer
    frames = _load(app, page, tmp_path) if loaded else ()
    viewer_action = (page._shell.run_controls.startButton.text(),
                     page._shell.run_controls.startButton.isEnabled())
    row = _row(page)
    assert row.editor.isEnabled() and row.browse_button.isEnabled()
    assert not any(field.enabled for field in page._shell.controls.projection.fields
                   if field.path != PROJECT_ROOT)
    typed = tmp_path / "typed project"
    typed.mkdir()
    row.editor.setFocus()
    row.editor.selectAll()
    QtTest.QTest.keyClicks(row.editor, str(typed))
    QtTest.QTest.keyClick(row.editor, QtCore.Qt.Key.Key_Return)
    app.processEvents()
    assert store.snapshot().thaw().project_root == str(typed)
    assert store.snapshot().thaw().save_path == str(typed / "xdart_processed_data")
    assert _row(page).editor.text() == str(typed)
    # The existing reducer keeps a separately chosen output directory.
    custom = tmp_path / "explicit_output"
    custom_intent = store.snapshot().thaw()
    custom_intent.save_path = str(custom)
    store.commit(custom_intent, expected_revision=store.revision)
    page._refresh_shell()
    expected_start = browse_start_dir(str(typed), fallback=str(typed))
    QtTest.QTest.mouseClick(_row(page).browse_button, QtCore.Qt.MouseButton.LeftButton)
    app.processEvents()
    assert calls == [(PROJECT_ROOT, str(typed), expected_start)]
    assert store.snapshot().thaw().project_root == str(picked)
    assert store.snapshot().thaw().save_path == str(custom)
    assert _row(page).editor.text() == str(picked)
    assert _row(page).editor.toolTip() == str(picked)
    assert page._context_controller.navigation.frames == frames
    assert (page._shell.run_controls.startButton.text(),
            page._shell.run_controls.startButton.isEnabled()) == viewer_action


def test_project_edit_rechecks_real_run_cleanup_and_picker_return(viewer, tmp_path):
    app, page, store, lifecycle, calls, picked = viewer
    request = lifecycle.begin_start().request_id
    identity = lifecycle.preflight_accepted(PreflightAccepted(request, RunIntent().freeze())).run_identity
    lifecycle.executor_accepted(ExecutorAccepted(identity))
    before = store.snapshot()
    try:
        for failed in (False, True):
            if failed:
                lifecycle.fatal(FatalExecution(identity))
            page._refresh_shell()
            assert not _row(page).editor.isEnabled()
            assert not _row(page).browse_button.isEnabled()
            page._on_field_value(PROJECT_ROOT, str(picked))
            page._choose_control_path(PROJECT_ROOT)
            assert store.revision == before.revision and calls == []
    finally:
        if lifecycle.active_run_identity is identity:
            lifecycle.fatal(FatalExecution(identity))
        lifecycle.owners_closed(OwnersClosed(identity))
        lifecycle.reset()
    page._refresh_shell()
    assert _row(page).editor.isEnabled()

    def start_while_picker_open(_path, _current, _start):
        lifecycle.begin_start()
        return str(picked)

    page._control_path_chooser = start_while_picker_open
    try:
        page._choose_control_path(PROJECT_ROOT)
        assert store.revision == before.revision
    finally:
        lifecycle.preflight_refused(PreflightRefused(lifecycle.request_id))


def test_project_folder_stays_locked_until_viewer_clear_is_acknowledged(viewer, tmp_path):
    app, page, store, _lifecycle, calls, picked = viewer
    _load(app, page, tmp_path)
    controller = page._context_controller
    is_1d = store.snapshot().thaw().processing_mode == "1D Viewer"
    request = (controller.begin_viewer_1d_renderer_clear() if is_1d
               else controller.begin_viewer_2d_renderer_clear())
    assert request is not None
    before = store.revision
    try:
        page._refresh_shell(preserve_scientific=True)
        assert not _row(page).editor.isEnabled()
        assert not _row(page).browse_button.isEnabled()
        page._on_field_value(PROJECT_ROOT, str(picked))
        page._choose_control_path(PROJECT_ROOT)
        assert store.revision == before and calls == []
    finally:
        assert (page._clear_viewer_1d_renderer(close=True) if is_1d
                else page._clear_viewer_2d_renderer(close=True))
