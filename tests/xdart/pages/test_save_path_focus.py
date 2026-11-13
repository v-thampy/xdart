"""C-11 of NEXT-1210-RESPONSIVENESS: the Save Path takes focus after the user changes the Project.

The maintainer, 2026-11-12 (verbatim), asked whether the Save Path should show its full path at rest: "If the project
path is changed, move fo cus to save path and show the full path. When focus moves away from there only show the last
part of the path like now. The same behavior for all windows int, stitch, RSM etc".

Through the real pages and their shared rows (``FormRow``): a typed Project commit or a Browse pick that changes the
Project moves focus to the Save Path, which shows its full path while focused and its last part once focus leaves. A
Project the user did not change (set by code, or a commit of the same text) moves no focus.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

import pytest

pytest.importorskip("pyqtgraph")

from pyqtgraph.Qt import QtCore, QtWidgets
from PySide6 import QtTest

from xdart.gui.widgets.controls_panel import FormRow

PROJECT = ("Project", "project_folder")
SAVE_PATHS = {"int": ("Project", "h5_dir"), "stitch": ("Stitch", "output_path"), "rsm": ("RSM", "output_path")}


def _pump(app, done, seconds=5.0):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        app.processEvents()
        if done():
            return True
        time.sleep(.005)
    return done()


class _Tool:
    def __init__(self, name, page, root, set_project, picks):
        self.name, self.page, self.root, self._set_project, self.picks = name, page, root, set_project, picks

    def row(self, path):
        return next(row for row in self.root.findChildren(FormRow) if row.path == path and row.isVisible())

    def project(self):
        return self.row(PROJECT)

    def save_path(self):
        return self.row(SAVE_PATHS[self.name])

    def set_project_by_code(self, value):
        self._set_project(value)


@contextmanager
def _tool(name, tmp_path, monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setenv("XDART_SESSION_FILE", str(tmp_path / "session.json"))
    first = tmp_path / "first"
    first.mkdir()
    picks = []
    from xdart.gui.widgets import file_dialogs
    monkeypatch.setattr(file_dialogs, "browse", lambda *args, **kwargs: picks[-1:])  # Stitch's and RSM's chooser
    if name == "int":
        from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
        from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
        from xdart.gui.tabs.scattering.events import CleanupStatus
        from xdart.gui.tabs.scattering.page import ScatteringWorkspace
        from xrd_tools.session.intent_store import RunIntentStore
        from xrd_tools.session.run_configuration import RunIntent
        store = RunIntentStore(RunIntent(project_root=str(first), save_path=str(first / "xdart_processed_data")))
        page = ScatteringWorkspace(intents=store, lifecycle=ScatteringCoordinator(),
                                   sources=FilesystemSourceAdapter(),
                                   control_path_chooser=lambda path, current, start: picks[-1] if picks else None)

        def set_project(value):
            intent = store.snapshot().thaw()
            intent.project_root, intent.save_path = value, str(Path(value) / "xdart_processed_data")
            store.commit(intent, expected_revision=store.revision)
            page._refresh_shell()
        root = page._shell.controls
    elif name == "stitch":
        from xdart.gui.pages.services import empty_host_services
        from xdart.gui.tabs.stitch.page import StitchPage
        from tests.xdart.pages.test_stitch_page import _Status
        page = StitchPage(empty_host_services(_Status()))
        set_project, root = (lambda value: page.set_field("project_root", value)), page
    else:
        from tests.xdart.pages.test_rsm_page import _page
        page = _page(app)
        set_project, root = (lambda value: page.set_field("project_root", value)), page
    page.resize(1400, 1000)
    page.show()
    page.activateWindow()
    try:
        tool = _Tool(name, page, root, set_project, picks)
        if name != "int":
            tool.set_project_by_code(str(first))
        app.processEvents()
        yield app, tool, first
    finally:
        if name == "int":
            assert _pump(app, lambda: page.close_workspace().cleanup_status is CleanupStatus.CLEANED)
            page.deleteLater()
        elif name == "stitch":
            from tests.xdart.pages.test_stitch_fit_host import _dispose
            _dispose(page, app)
        else:
            from tests.xdart.pages.test_rsm_page import _close
            _close(page, app)
        app.processEvents()


def _type_project(app, tool, value):
    editor = tool.project().editor
    editor.setFocus()
    app.processEvents()
    editor.selectAll()
    QtTest.QTest.keyClicks(editor, value)
    QtTest.QTest.keyClick(editor, QtCore.Qt.Key.Key_Return)


@pytest.mark.parametrize("name", tuple(SAVE_PATHS))
def test_a_typed_project_change_moves_focus_to_the_full_save_path(tmp_path, monkeypatch, name):
    with _tool(name, tmp_path, monkeypatch) as (app, tool, _first):
        second = tmp_path / "second"
        second.mkdir()
        _type_project(app, tool, str(second))
        save = tool.save_path()
        full = str(second / "xdart_processed_data")
        assert _pump(app, lambda: save.editor.hasFocus()), QtWidgets.QApplication.focusWidget()
        assert save.editor.text() == full  # the full path while focused
        tool.project().editor.setFocus()  # focus moves away
        assert _pump(app, lambda: save.editor.text() == "xdart_processed_data"), save.editor.text()


@pytest.mark.parametrize("name", tuple(SAVE_PATHS))
def test_a_browsed_project_change_moves_focus_to_the_full_save_path(tmp_path, monkeypatch, name):
    with _tool(name, tmp_path, monkeypatch) as (app, tool, _first):
        picked = tmp_path / "picked"
        picked.mkdir()
        tool.picks.append(str(picked))
        QtTest.QTest.mouseClick(tool.project().browse_button, QtCore.Qt.MouseButton.LeftButton)
        save = tool.save_path()
        assert _pump(app, lambda: save.editor.hasFocus()), QtWidgets.QApplication.focusWidget()
        assert save.editor.text() == str(picked / "xdart_processed_data")


@pytest.mark.parametrize("name", tuple(SAVE_PATHS))
def test_a_project_the_user_did_not_change_moves_no_focus(tmp_path, monkeypatch, name):
    with _tool(name, tmp_path, monkeypatch) as (app, tool, first):
        project = tool.project().editor
        _type_project(app, tool, str(first))  # the same Project committed again
        _pump(app, lambda: False, .2)
        assert project.hasFocus()
        other = tmp_path / "other"
        other.mkdir()
        tool.set_project_by_code(str(other))  # a restore or another tool's Project, not his edit
        _pump(app, lambda: False, .2)
        assert project.hasFocus() and not tool.save_path().editor.hasFocus()


@pytest.mark.parametrize("name", tuple(SAVE_PATHS))
def test_each_project_commit_is_judged_against_the_last_one(tmp_path, monkeypatch, name):
    """Changed back, the Project moves focus again; Return on the Project just set moves none (Stitch and RSM read
    their editors and never project the typed Project back into the row)."""
    with _tool(name, tmp_path, monkeypatch) as (app, tool, first):
        second = tmp_path / "second"
        second.mkdir()
        save = tool.save_path()
        _type_project(app, tool, str(second))
        assert _pump(app, lambda: save.editor.hasFocus())
        project = tool.project().editor
        project.setFocus()
        app.processEvents()
        QtTest.QTest.keyClick(project, QtCore.Qt.Key.Key_Return)  # the Project just set, committed again
        _pump(app, lambda: False, .2)
        assert project.hasFocus()
        _type_project(app, tool, str(first))  # and back
        assert _pump(app, lambda: save.editor.hasFocus()), QtWidgets.QApplication.focusWidget()
        assert save.editor.text() == str(first / "xdart_processed_data")


def test_ints_first_project_moves_focus_to_its_save_path(tmp_path, monkeypatch):
    """Int opened without a Project: naming the first one, as he does in a fresh session, focuses the Save Path."""
    from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
    from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
    from xdart.gui.tabs.scattering.events import CleanupStatus
    from xdart.gui.tabs.scattering.page import ScatteringWorkspace
    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.session.run_configuration import RunIntent
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setenv("XDART_SESSION_FILE", str(tmp_path / "session.json"))
    project = tmp_path / "project"
    project.mkdir()
    page = ScatteringWorkspace(intents=RunIntentStore(RunIntent()), lifecycle=ScatteringCoordinator(),
                               sources=FilesystemSourceAdapter())
    page.resize(1400, 1000)
    page.show()
    page.activateWindow()
    try:
        app.processEvents()
        tool = _Tool("int", page, page._shell.controls, None, [])
        _type_project(app, tool, str(project))
        assert _pump(app, lambda: tool.save_path().editor.hasFocus()), QtWidgets.QApplication.focusWidget()
        assert tool.save_path().editor.text() == str(project / "xdart_processed_data")
    finally:
        assert _pump(app, lambda: page.close_workspace().cleanup_status is CleanupStatus.CLEANED)
        page.deleteLater()
        app.processEvents()
