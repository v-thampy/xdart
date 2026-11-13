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
        tool.set_project(str(second))
        assert browser.directory == str(second / "xdart_processed_data")
        tool.set_save_path(str(chosen))
        assert browser.directory == str(chosen)
        tool.set_save_path("xdart_processed_data")  # a relative Save Path is inside the Project
        assert browser.directory == str(second / "xdart_processed_data")


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
