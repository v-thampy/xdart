"""The real Int page captures its committed intent and mounts only shared export/display owners."""
import json


def test_int_page_exports_current_intent_without_run_or_generation_change(tmp_path):
    from pyqtgraph.Qt import QtWidgets
    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.session.run_configuration import RunIntent
    from xrd_tools.session.run_intent_profile import dump_run_intent_profile
    from xdart.gui.tabs.scattering.page import ScatteringWorkspace
    from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
    from xdart.gui.analysis.workflow_export import WorkflowExportActions
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    intent = RunIntent(project_root=str(tmp_path), save_path=str(tmp_path / "prior-results"),
                       processing_mode="Int 2D", max_cores=2, batch_mode=True)
    store = RunIntentStore(intent)
    page = ScatteringWorkspace(intents=store, lifecycle=ScatteringCoordinator(), sources=None, executor=None)
    before = store.snapshot()
    try:
        assert isinstance(page._workflow_export_actions, WorkflowExportActions)
        assert page._shell.scientific.page_tool("export-run").text()
        expected = json.loads(dump_run_intent_profile(before.thaw()))
        captured = page._capture_workflow_recipe()
        assert captured.mode == "int" and captured.configuration == expected
        path = tmp_path / "replay.ipynb"
        page._workflow_export_actions.export_to(path)
        assert path.is_file() and "RUN_PROCESSING = False" in path.read_text()
        assert not (tmp_path / "replay_outputs").exists()
        after = store.snapshot()
        assert after.revision == before.revision
        assert after.thaw().generation == before.thaw().generation
        assert dump_run_intent_profile(after.thaw()) == dump_run_intent_profile(before.thaw())
    finally:
        page.close_workspace()
        page.deleteLater()
        app.processEvents()
