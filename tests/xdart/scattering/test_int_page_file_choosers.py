"""Int's own file choosers open through the shared file dialog (INT-PAGE-FILE-DIALOGS-1210; his pick "Move them in
1.2.10 (Recommended)").

The widened reuse fence found Int's page calling QFileDialog directly at eight sites. Int keeps the one Browse rule
itself (it seeds each start folder from browse_start_dir and remembers a pick only once it is accepted), so each site
calls file_dialogs.ask, the one place a dialog opens, with the start folder and file types it already chose; ask then
spells a mapped share's pick with its drive letter and returns focus to the window. The OS picker is answered at the
QFileDialog boundary; ask is spied on and called through.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pyqtgraph.Qt import QtWidgets

from tests.xdart.scattering.test_e1b2_page_command_boundaries import _Executor, _Sources, _dispose
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.widgets import file_dialogs
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


_CASES = {
    "profile save": ("_choose_profile_path_dialog", ("save",), "getSaveFileName", "profile.json", False),
    "profile load": ("_choose_profile_path_dialog", ("load",), "getOpenFileName", "profile.json", False),
    "notebook export": ("_choose_results_notebook_dialog", (), "getSaveFileName", "results.ipynb", False),
    "1D Viewer": ("_choose_viewer_1d_dialog", (), "getOpenFileNames", "pattern.xye", True),
    "2D Viewer": ("_choose_viewer_2d_dialog", (), "getOpenFileName", "image.tif", False),
    "processed folder": ("_choose_directory_dialog", ("",), "getExistingDirectory", "processed", False),
    "authoring source": ("_choose_authoring_source_dialog", ("poni",), "getOpenFileName", "calibrant.tif", False),
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_int_page_choosers_open_through_the_shared_file_dialog(qapp, tmp_path, monkeypatch, case):
    method, leading, static, name, many = _CASES[case]
    start = str(tmp_path / "start")
    Path(start).mkdir()
    picked = str(tmp_path / name)
    opened, asked = [], []

    def os_picker(parent, title, directory, *rest):
        opened.append((title, directory, rest))
        if static == "getExistingDirectory":
            return picked
        return ([picked], rest[0] if rest else "") if many else (picked, rest[0] if rest else "")

    real_ask = file_dialogs.ask
    monkeypatch.setattr(QtWidgets.QFileDialog, static, staticmethod(os_picker))
    monkeypatch.setattr(file_dialogs, "ask", lambda *args, **kwargs: asked.append(kwargs) or real_ask(*args, **kwargs))
    page = ScatteringWorkspace(intents=RunIntentStore(RunIntent(project_root=str(tmp_path))),
                               lifecycle=ScatteringCoordinator(), sources=_Sources(), executor=_Executor())
    try:
        result = getattr(page, method)(*leading, start)
        assert len(asked) == 1, f"{case}: the chooser did not open through file_dialogs.ask"
        assert asked[0]["start"] == start and len(opened) == 1 and opened[0][1] == start, (asked, opened)
        if static != "getExistingDirectory":
            assert asked[0]["file_filter"] == opened[0][2][0] and asked[0]["file_filter"], asked
        assert result == ((picked,) if many else picked), result
    finally:
        _dispose(page, qapp)
