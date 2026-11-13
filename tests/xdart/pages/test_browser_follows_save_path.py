"""C-14 and C-15 of NEXT-1210-RESPONSIVENESS: the Data Browser follows the Save Path, and a Run lists its results.

The maintainer, 2026-11-13 (verbatim): "when i changed the project folder, the save path changed but the data browser
path didn't - this should follow the save path"; "i started processing all scan numbers in the LaB6 file. it seems the
scans were processed but the display is not updated as the scans are updated"; "above should be how stitch and int
work as well"; his pick "Always follow (Recommended)": any Project or Save Path change moves the Data Browser to the
Save Path folder in Int, Stitch and RSM, replacing a folder he had opened, and a Run start brings it back to the Save
Path so each result shows as it finishes. (His RSM Run had finished none before his Stop: the scan search, C-16.)
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from tests.xdart.pages.test_rsm_page import (  # noqa: F401
    _close, _commit_instead_of_gridding, _page, _synthetic_source, _values, _wait, _with_source, qapp,
)
from tests.xdart.pages.test_save_path_rule import _tool


def test_ints_browser_follows_a_save_path_change_over_an_opened_folder(tmp_path):
    from xdart.gui.tabs.scattering.processed_browser import ProcessedBrowserOwner
    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.session.run_configuration import RunIntent
    first, second = (str(tmp_path / name / "xdart_processed_data") for name in ("A", "B"))
    owner = ProcessedBrowserOwner(save_path=first, processing_mode="Int 1D", deliver=lambda _wake: None,
                                  catalog_reader=lambda _directory, **_kwargs: ())
    try:
        owner.set_directory(str(tmp_path / "opened"), explicit=True)  # File > Open Folder
        owner.reconcile_intent(RunIntentStore(RunIntent(save_path=first)).snapshot(),
                               RunIntentStore(RunIntent(save_path=second)).snapshot())
        assert owner.directory == second and not owner.explicit_directory
        project = str(tmp_path / "C")  # a relative Save Path is inside the Project, not the working directory
        owner.reconcile_intent(RunIntentStore(RunIntent(save_path=second)).snapshot(),
                               RunIntentStore(RunIntent(project_root=project,
                                                        save_path="xdart_processed_data")).snapshot())
        assert owner.directory == str(Path(project) / "xdart_processed_data")
        next_project = str(tmp_path / "D")
        owner.set_directory(str(tmp_path / "opened"), explicit=True)
        owner.reconcile_intent(RunIntentStore(RunIntent(project_root=project, save_path="results")).snapshot(),
                               RunIntentStore(RunIntent(project_root=next_project, save_path="results")).snapshot())
        assert owner.directory == str(Path(next_project) / "results") and not owner.explicit_directory
    finally:
        owner.begin_close()
        until = time.monotonic() + 10
        while not owner.retry_close() and time.monotonic() < until:
            time.sleep(.01)


@pytest.mark.parametrize("tool_name", ("stitch", "rsm"))
def test_the_browser_follows_the_projects_and_the_save_paths_changes(qapp, tmp_path, tool_name):
    first, second, chosen, opened = (tmp_path / name for name in ("A", "B", "chosen", "opened"))
    first.mkdir()
    second.mkdir()  # its processed-data folder does not exist yet
    with _tool(qapp, tool_name) as tool:
        browser = tool.page._result_browser
        tool.set_project(str(first))
        assert browser.directory == str(first / "xdart_processed_data")
        browser.set_directory(str(opened))  # File > Open Folder, a folder he chose
        notices = []
        browser.notice.connect(notices.append)
        tool.set_project(str(second))
        assert browser.directory == str(second / "xdart_processed_data")
        until = time.monotonic() + 10  # its folder does not exist yet: the read ends quietly
        while browser.busy and time.monotonic() < until:
            qapp.processEvents()
            time.sleep(.01)
        qapp.processEvents()
        assert not browser.busy and notices == [], notices
        tool.set_save_path(str(chosen))
        assert browser.directory == str(chosen)
        browser.set_directory(str(opened))
        tool.set_project(str(first))  # a custom absolute Save Path survives the Project change
        assert browser.directory == str(chosen)
        tool.set_project(str(second))
        tool.set_save_path("xdart_processed_data")  # a relative Save Path is inside the Project
        assert browser.directory == str(second / "xdart_processed_data")


def test_int_folder_run_returns_to_save_path_before_publishing(qapp, tmp_path):
    from xdart.gui.pages.scattering_workspace import _WorkspaceOpenFolder
    from xdart.gui.tabs.scattering.adapters.run_executor import StandardEventKind
    from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell
    from tests.xdart.scattering.test_browse_reference_rings_run import _page as int_page, _run, _pump
    from tests.xdart.scattering.test_p1b_output_graph import _live_directory_intent, _write_tiff, write_poni

    source, output = tmp_path / "raw", tmp_path / "processed"
    source.mkdir()
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    _write_tiff(source / "first_0001.tif", 1)
    _write_tiff(source / "second_0001.tif", 2)
    intent = _live_directory_intent(source, output, poni, processing_mode="Int 2D")
    intent.live_mode = False
    with int_page(qapp, intent) as (page, _view, observed):
        _WorkspaceOpenFolder(page).set_directory(str(tmp_path / "inherited"))
        followed, ready = [], set()
        original_follow = page._processed_browser.begin_follow
        original_select = page._select_presentation_target

        def begin_follow(*args, **kwargs):
            result = original_follow(*args, **kwargs)
            followed.append(page._processed_browser.directory)
            return result

        def select(identity, frame):
            selected = original_select(identity, frame)
            if selected:
                ready.add(page._context_controller.navigation.current.artifact)
            return selected

        page._processed_browser.begin_follow = begin_follow
        page._select_presentation_target = select
        assert _pump(qapp, page.findChild(ScatteringWorkspaceShell).run_controls.startButton.isEnabled)
        terminal = _run(qapp, page, observed)
        assert terminal.kind is StandardEventKind.FINISHED and terminal.files_processed == 2
        assert followed == [str(output)]
        paths = {str(path) for path in output.glob("*.nexus")}
        assert len(paths) == 2
        assert ready == paths
        assert _pump(qapp, lambda: paths.issubset(
            {entry.artifact for entry in page._processed_browser.catalog}), seconds=10)


def test_int_a_pending_browse_open_does_not_move_back_from_a_new_save_path(qapp, tmp_path):
    from xdart.gui.tabs.scattering.controls_inventory import SAVE_PATH
    from tests.xdart.scattering.test_browse_reference_rings_run import _page as int_page, _run, _browsing
    from tests.xdart.scattering.test_p1b_output_graph import _intent, _write_tiff, write_poni

    raw, output, new_output = (tmp_path / name for name in ("scan_0001.tif", "old", "new"))
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    _write_tiff(raw, 1)
    with int_page(qapp, _intent(raw, output, poni)) as (page, _view, observed):
        _run(qapp, page, observed)
        _browsing(qapp, page)
        path, = output.glob("*.nexus")
        page._select_scan(str(path), reopen=True)
        assert page._context_controller.browse_pending
        page._on_field_value(SAVE_PATH, str(new_output))
        assert page._processed_browser.directory == str(new_output)
        _browsing(qapp, page)
        assert page._processed_browser.directory == str(new_output)
        assert page._context_controller.navigation.current.artifact == str(path)


def test_int_relative_save_path_and_actual_output_share_the_project_folder(qapp, tmp_path, monkeypatch):
    from tests.xdart.scattering.test_browse_reference_rings_run import _page as int_page, _run
    from tests.xdart.scattering.test_p1b_output_graph import _intent, _write_tiff, write_poni

    project, cwd = tmp_path / "project", tmp_path / "elsewhere"
    project.mkdir()
    cwd.mkdir()
    raw, poni = project / "scan_0001.tif", project / "cal.poni"
    _write_tiff(raw, 1)
    write_poni(poni)
    monkeypatch.chdir(cwd)
    intent = _intent(raw, Path("processed"), poni)
    with int_page(qapp, intent) as (page, _view, observed):
        _run(qapp, page, observed)
        assert len(tuple((project / "processed").glob("*.nexus"))) == 1
        assert page._processed_browser.directory == str(project / "processed")
        assert not (cwd / "processed").exists()


def test_stitch_typing_a_save_path_does_not_move_the_browser_before_commit(qapp, tmp_path):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    with _tool(qapp, "stitch") as tool:
        tool.page.show()
        tool.set_project(str(tmp_path))
        old = str(tmp_path / "xdart_processed_data")
        editor = tool.page._rows["output_path"].editor
        editor.setFocus()
        qapp.processEvents()
        editor.selectAll()
        QTest.keyClicks(editor, "next")
        assert tool.page._result_browser.directory == old
        QTest.keyClick(editor, Qt.Key.Key_Return)
        assert tool.page._result_browser.directory == str(tmp_path / "next")


def test_an_rsm_run_brings_the_browser_back_and_lists_its_result(qapp, tmp_path, monkeypatch):
    _commit_instead_of_gridding(monkeypatch, _values())
    page = _page(qapp)
    try:
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path)
        output = Path(page.build_form().output_path)
        page._result_browser.set_directory(str(tmp_path / "opened"))  # he had opened another folder
        page.begin_run()
        _wait(page, qapp, lambda: page.shown is not None and not page.active())
        assert page._result_browser.directory == str(output.parent)
        assert str(output) in page.browser.listed_results(), page.browser.listed_results()
        assert page.browser.current_result() == str(output)
    finally:
        _close(page, qapp)


def test_a_stitch_run_brings_the_browser_back_and_shows_its_result(qapp, tmp_path):
    from xdart.gui.pages.services import empty_host_services
    from xdart.gui.tabs.stitch.page import StitchPage
    from tests.xdart.pages.test_stitch_page import _Status, _dispose, _wait as wait_stitch
    from tests.xdart.tools.test_stitch_native_values import _native_form
    form = _native_form(tmp_path, "multigeometry", scans="1")
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.load_form(form)
        page._result_browser.set_directory(str(tmp_path / "opened"))  # he had opened another folder
        page.preview_button.click()
        wait_stitch(page, qapp, catalog=True)
        target = Path(page._owner.prepared.request.module.output.target)
        page.run_bar.startButton.click()
        wait_stitch(page, qapp, catalog=True)
        assert target.is_file()
        assert page._result_browser.directory == str(target.parent)
        assert page.browser.current_result() == str(target), page.browser.listed_results()
        assert page.curve.listDataItems(), "the finished result is drawn"
    finally:
        _dispose(page, qapp)


def test_a_stitch_run_with_a_blank_save_path_lists_its_result_from_the_projects_folder(qapp, tmp_path):
    """Stitch reads a blank Save Path as the Project's processed-data folder; its Run brings the browser there."""
    from xdart.gui.pages.services import empty_host_services
    from xdart.gui.tabs.stitch.page import StitchPage
    from tests.xdart.pages.test_stitch_page import _Status, _dispose, _wait as wait_stitch
    from tests.xdart.tools.test_stitch_native_values import _native_form
    form = _native_form(tmp_path, "multigeometry", scans="1")
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.load_form(form)
        page.set_field("output_path", "")
        page._result_browser.set_directory(str(tmp_path / "opened"))  # he had opened another folder
        page.preview_button.click()
        wait_stitch(page, qapp, catalog=True)
        target = Path(page._owner.prepared.request.module.output.target)
        assert target.parent == Path(page._value("project_root")) / "xdart_processed_data"
        page.run_bar.startButton.click()
        wait_stitch(page, qapp, catalog=True)
        assert page._result_browser.directory == str(target.parent)
        assert page.browser.current_result() == str(target), page.browser.listed_results()
    finally:
        _dispose(page, qapp)
