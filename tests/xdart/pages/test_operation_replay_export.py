"""Native pages mount shared replay export and capture only current operator intent."""
import json

import pytest

from xdart.gui.pages.services import empty_host_services
from xdart.gui.pages.values import PageCleanup
from tests.xdart.pages.test_stitch_page import _Status, qapp  # noqa: F401


@pytest.mark.parametrize("mode", ("stitch", "rsm"))
def test_page_exports_current_run_recipe_without_processing(qapp, tmp_path, mode, monkeypatch):
    from xdart.gui.analysis.workflow_export import WorkflowExportActions
    from xrd_tools.notebooks.workflow import WorkflowRecipe
    if mode == "stitch":
        from xdart.gui.tabs.stitch.page import StitchPage
        from tests.xdart.tools.test_stitch_native_values import _native_form
        page = StitchPage(empty_host_services(_Status()))
        page.load_form(_native_form(tmp_path, "multigeometry", scans="1"))
        view = page.scientific
    else:
        from xdart.gui.tabs.rsm.page import RSMPage
        from tests.xdart.pages.test_rsm_page import _synthetic_source, _with_source, _wait as rsm_wait
        _synthetic_source(tmp_path)
        page = RSMPage(empty_host_services(_Status()))
        _with_source(page, tmp_path)
        rsm_wait(page, qapp, lambda: not page._owner.busy)
        view = page.view
    try:
        assert type(page._workflow_export) is WorkflowExportActions
        action = next(action for action in page.browser.menus["Help"].actions()
                      if action.text() == "Export Run notebook…")
        assert view.findChild(type(page.browser.open_result), "e7PageTool_export-run") is None
        recipe = page._capture_run_workflow()
        assert type(recipe) is WorkflowRecipe and recipe.mode == mode
        assert recipe.configuration["forms"]
        before = {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
        notebook = tmp_path / "replay.ipynb"
        from xdart.gui.widgets import file_dialogs
        monkeypatch.setattr(file_dialogs, "browse", lambda *args, **kwargs: (str(notebook),))
        action.trigger()
        assert notebook.is_file()
        document = json.loads(notebook.read_text())
        assert "run_workflow" in "".join("".join(cell.get("source", ())) for cell in document["cells"])
        assert not page._owner.busy
        assert not (tmp_path / "replay_outputs").exists()
        assert not list(tmp_path.rglob("*_stitch1d.nexus")) and not list(tmp_path.rglob("*_rsm.nexus"))
        for path, content in before.items():
            assert (tmp_path / path).read_bytes() == content
    finally:
        receipt = page.shutdown() if mode == "stitch" else page.close_page()
        assert receipt.status is PageCleanup.CLEAN
        page.deleteLater()
        qapp.processEvents()
