"""One Save Path rule for Int 1D/2D, Stitch and RSM (NEXT-1210-RESPONSIVENESS C-6).

The maintainer, 2026-11-12 (verbatim): "yes no placeholder. we don't have a placeholder in int1d or 2d either. this is
supposed to be just the path where the processed data is saved", then "the save path should be the same and derived
from the project folder in all cases int1d/2d/stitch/rsm", and "Same rule, still editable (Recommended)".

Every tool's Save Path is a folder: ``<Project>/xdart_processed_data`` by default, moved with the Project while it is
that default (or blank), and kept once the user chooses another folder. RSM names its result inside it, from its
scans (``<family>_rsm.nexus``); RSM's Save Path no longer holds a file name or a ``rsm_rsm.nexus`` placeholder.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

pytest.importorskip("pyqtgraph")

from xdart.gui.pages.services import empty_host_services
from xdart.gui.tabs.scattering.controls_editing import reduce_control_edit
from xdart.gui.tabs.scattering.controls_inventory import PROJECT_ROOT, SAVE_PATH
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent
from tests.xdart.pages.test_rsm_page import (  # noqa: F401
    _Owner, _Status, _close, _page, _synthetic_source, _with_source, qapp,
)

TOOLS = ("int", "stitch", "rsm")


class _Int:
    """Int 1D/2D's Save Path through its real edit owner (``reduce_control_edit``)."""

    def __init__(self):
        self.intent = RunIntent()

    def _edit(self, path, value):
        candidate = reduce_control_edit(RunIntentStore(self.intent).snapshot(), path, value)
        if isinstance(candidate, RunIntent):
            self.intent = candidate

    def set_project(self, value):
        self._edit(PROJECT_ROOT, value)

    def set_save_path(self, value):
        self._edit(SAVE_PATH, value)

    def project(self):
        return str(self.intent.project_root or "")

    def save_path(self):
        return str(self.intent.save_path or "")


class _Page:
    """Stitch's or RSM's Save Path through the real page."""

    def __init__(self, page):
        self.page = page

    def set_project(self, value):
        self.page.set_field("project_root", value)

    def set_save_path(self, value):
        self.page.set_field("output_path", value)

    def project(self):
        return str(self.page._value("project_root") or "")

    def save_path(self):
        return str(self.page._value("output_path") or "")


@contextmanager
def _tool(qapp, tool):
    if tool == "int":
        yield _Int()
        return
    if tool == "stitch":
        from xdart.gui.tabs.stitch.page import StitchPage
        from tests.xdart.pages.test_stitch_page import _Status as StitchStatus
        from tests.xdart.pages.test_stitch_fit_host import _dispose
        page = StitchPage(empty_host_services(StitchStatus()))
        try:
            yield _Page(page)
        finally:
            _dispose(page, qapp)
        return
    page = _page(qapp)
    try:
        yield _Page(page)
    finally:
        _close(page, qapp)


def _folder(tool):
    """The folder a Save Path names: a relative one is inside the Project."""
    path = Path(tool.save_path()).expanduser()
    return path if path.is_absolute() else Path(tool.project()) / path


@pytest.mark.parametrize("tool_name", TOOLS)
def test_every_tool_saves_into_the_projects_processed_data_folder_and_follows_the_project(
        qapp, tmp_path, tool_name):
    first, second, third, chosen = (tmp_path / name for name in ("A", "B", "C", "elsewhere"))
    for folder in (first, second, third):
        folder.mkdir()
    with _tool(qapp, tool_name) as tool:
        tool.set_project(str(first))
        assert tool.save_path() == str(first / "xdart_processed_data")
        tool.set_project(str(second))
        assert tool.save_path() == str(second / "xdart_processed_data")
        tool.set_save_path("xdart_processed_data")  # the default, spelled inside the Project
        tool.set_project(str(third))
        assert _folder(tool) == third / "xdart_processed_data"
        tool.set_save_path(str(chosen))  # still editable: a folder the user chooses is kept
        tool.set_project(str(first))
        assert tool.save_path() == str(chosen)


def test_rsm_save_path_stays_the_folder_and_its_result_is_named_inside_it(qapp, tmp_path):
    from xdart.gui.tabs.rsm.page import suggested_family

    page = _page(qapp)
    try:
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path)
        folder = tmp_path / "xdart_processed_data"
        assert page._value("output_path") == str(folder)  # no file name, no placeholder, with the scans found
        family = suggested_family(tmp_path, page._group_members())
        form = page.build_form()
        assert Path(form.output_path) == folder / f"{family}_rsm.nexus"
        assert form.output_family == family
        chosen = tmp_path / "mine"
        page.set_field("output_path", str(chosen))
        form = page.build_form()
        assert Path(form.output_path) == chosen / f"{family}_rsm.nexus" and form.output_family == family
    finally:
        _close(page, qapp)


def test_several_rsms_write_into_the_save_path_folder_itself(qapp, tmp_path):
    page = _page(qapp)
    try:
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path, scans="1,2")
        forms = page.build_forms()
        assert len(forms) == 2
        assert {Path(form.output_path).parent for form in forms} == {Path(page._value("output_path"))}
        assert all(Path(form.output_path).name == f"{form.output_family}_rsm.nexus" for form in forms)
    finally:
        _close(page, qapp)


def test_rsm_readiness_asks_for_a_save_path_folder_not_a_nexus_file(qapp, tmp_path):
    page = _page(qapp)
    try:
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path)
        assert page._readiness[1], page._readiness
        page.set_field("output_path", str(tmp_path / "named_rsm.nexus"))
        assert not page._readiness[1]
        assert "a Save Path folder, not a .nexus file" in page.run_bar.readinessLabel.full_text()
    finally:
        _close(page, qapp)


def test_a_saved_rsm_output_file_loads_as_its_folder(qapp, tmp_path):
    """A 1.2.9 settings document names an output file; it loads as that file's folder, and the result is named in it
    from the scans."""
    page, restored = _page(qapp, owner=_Owner()), _page(qapp, owner=_Owner())
    try:
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path)
        document = page.settings()
        legacy = tmp_path / "older"
        document["fields"]["output_path"] = str(legacy / "named_rsm.nexus")
        restored.apply_settings(document)
        restored.resolve_source()
        assert restored._value("output_path") == str(legacy)
        form = restored.build_form()
        assert Path(form.output_path).parent == legacy and form.output_family
    finally:
        _close(page, qapp)
        _close(restored, qapp)


def test_rsm_save_path_browses_for_a_folder(qapp, tmp_path, monkeypatch):
    from xdart.gui.widgets import file_dialogs

    calls = []
    monkeypatch.setattr(file_dialogs, "browse", lambda widget, title, **kwargs: calls.append((title, kwargs)) or [])
    page = _page(qapp)
    try:
        page.set_field("project_root", str(tmp_path))
        page._browse("output_path")
        ((title, kwargs),) = calls
        assert title == "Choose processed-data folder" and kwargs.get("folder") is True
        assert not kwargs.get("save")
    finally:
        _close(page, qapp)


def test_rsm_takes_the_shared_project_and_save_path_folder(qapp, tmp_path):
    """A workspace opened without a Project takes another tool's Project and Save Path (ProjectDefaults); RSM keeps the
    folder as its Save Path, as Stitch does."""
    from xdart.gui.tabs.rsm.page import build_rsm_page

    handle = build_rsm_page(empty_host_services(_Status()), None)
    page = handle.widget
    try:
        page.set_field("project_root", "")
        page.set_field("output_path", "")
        chosen = tmp_path / "shared"
        handle.project_location.inherit((str(tmp_path), str(chosen)))
        assert page._value("project_root") == str(tmp_path)
        assert page._value("output_path") == str(chosen)
        assert handle.project_location.location() == (str(tmp_path), str(chosen))
    finally:
        _close(page, qapp)


def test_rsm_shows_a_following_save_path_in_full_and_moves_it_with_the_project(qapp, tmp_path):
    """The maintainer, 2026-11-12: "when i changed the project folder the save path didn't change to follow the
    project path" (his RSM Save Path read the relative "xdart_processed_data"). A following Save Path shows the
    Project's folder in full, as Int and Stitch show it, so a Project change visibly moves it."""
    first, second = tmp_path / "A", tmp_path / "B"
    for folder in (first, second):
        folder.mkdir()
    page = _page(qapp)
    try:
        page.set_field("project_root", str(first))
        page.set_field("output_path", "xdart_processed_data")
        page.set_field("project_root", str(second))
        assert page._value("output_path") == str(second / "xdart_processed_data")
    finally:
        _close(page, qapp)
