"""GUI composition checks for ordinary runs and Average ownership."""

from pathlib import Path
import sys


def test_nonaverage_paths_have_zero_average_reachability_and_allocations(
    tmp_path, monkeypatch, _xdart_qt_harness,
):
    from xrd_tools.io import nexus_record, read as read_module, record_writer
    from xrd_tools.reduction import average
    from xrd_tools.sources import execution_graph
    from xdart.gui.tabs.scattering.adapters import external_operation
    from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
    from xdart.gui.tabs.scattering.browse_values import BrowseLoadStatus
    from xdart.gui.tabs.scattering.display_values import StandardEventKind
    from xdart.gui.tabs.scattering.state_machine import RunPhase
    from xdart.gui.tabs.scattering.page import ScatteringWorkspace
    from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
    from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
    from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell
    from xrd_tools.session.intent_store import RunIntentStore
    from tests.xdart.scattering.test_e3_join_oracle import _wait
    from tests.xdart.scattering.test_p1b_output_graph import (_TERMINAL, _drain_until,
        _intent, _live_directory_intent, _run_to_terminal, _start, _write_tiff)
    from tests.xdart.scattering._e2sd_support import write_poni
    hits = []
    def forbidden(*_a, _name="average", **_k):
        hits.append(_name); raise AssertionError(f"ordinary path reached {_name}")
    for owner, names in (
        (average, ("AverageScanRecipe", "AverageScanRunner",
                   "requirements_from", "resolve_session_policy")),
        (execution_graph, ("read_detector_image_layout",)),
        (read_module, ("get_average_finite_counts",)),
        (nexus_record, ("write_average_finite_counts",)),
        (record_writer, ("write_average_finite_counts",)),
        (external_operation, ("AverageScanRecipe", "AverageScanRunner")),
    ):
        for name in names:
            monkeypatch.setattr(owner, name, lambda *a, _name=name, **k:
                                forbidden(*a, _name=_name, **k), raising=False)
    app = _xdart_qt_harness.app
    page_root = tmp_path / "page"
    page_root.mkdir()
    raw = page_root / "scan_0001.tif"
    poni = page_root / "cal.poni"
    output_root = page_root / "processed"
    _write_tiff(raw, 1)
    write_poni(poni)
    lifecycle = ScatteringCoordinator()
    page = ScatteringWorkspace(
        intents=RunIntentStore(_intent(raw, output_root, poni)),
        lifecycle=lifecycle,
        sources=FilesystemSourceAdapter(),
        executor=StandardRunExecutor(),
    )
    page.show()
    controller = page._context_controller
    shell = page.findChild(ScatteringWorkspaceShell)
    assert shell is not None
    monkeypatch.setattr(
        type(page._workspace_operations._slot), "begin_average", forbidden,
        raising=False,
    )
    try:
        _wait(app, lambda: shell.run_controls.startButton.isEnabled())
        shell.run_controls.startButton.click()
        _wait(app, lambda: controller.acquisition_context is not None)
        _wait(app, lambda: lifecycle.phase is RunPhase.IDLE,
              diagnostic=lambda: lifecycle.phase.value)
        output = output_root / "scan_int1d.nexus"
        assert output.is_file()
        request = controller.begin_browse(str(output))
        outcomes = []

        def browse_finished():
            outcome = controller.poll_browse()
            if outcome is None:
                return False
            outcomes.append(outcome)
            return True

        _wait(app, browse_finished)
        assert len(outcomes) == 1
        assert outcomes[0].request is request
        assert outcomes[0].status is BrowseLoadStatus.READY
        assert controller.browse_context is not None
        assert controller.browse_context.load_request is request
    finally:
        cleanup = []
        def close_page():
            receipt = page.close_workspace()
            cleanup.append(receipt)
            return receipt.cleanup_status.value == "cleaned"
        _wait(app, close_page)
        assert cleanup[-1].cleanup_status.value == "cleaned"
    raw = tmp_path / "raw_0001.tif"; poni = tmp_path / "cal.poni"
    target = tmp_path / "ordinary"; _write_tiff(raw, 1); write_poni(poni)
    for ordinal, mode in enumerate(("Overwrite", "Append"), 1):
        executor, identity, events = _run_to_terminal(
            _intent(raw, target, poni, output_mode=mode), request_value=36_000 + ordinal)
        assert next(event for event in events if event.kind in _TERMINAL).kind is StandardEventKind.FINISHED
        assert executor.close(identity).cleanup_status.value == "cleaned"
    live = tmp_path / "live"; live.mkdir(); _write_tiff(live / "live_0001.tif", 2)
    executor = StandardRunExecutor(join_timeout=2.0)
    identity = _start(executor, _live_directory_intent(
        live, tmp_path / "live-output", poni, processing_mode="Int 1D"),
        request_value=36_003)
    events = _drain_until(executor, lambda rows: any(
        event.kind in {StandardEventKind.FRAME_READY, *_TERMINAL} for event in rows))
    assert any(event.kind is StandardEventKind.FRAME_READY for event in events)
    executor.stop(identity)
    terminal = _drain_until(executor, lambda rows: any(event.kind in _TERMINAL for event in rows))
    assert next(event for event in terminal if event.kind in _TERMINAL).kind is StandardEventKind.STOPPED
    assert executor.close(identity).cleanup_status.value == "cleaned" and hits == []
def test_average_import_owner_and_alternate_writer_census():
    import ast, subprocess
    from xrd_tools.io import record_writer
    from xrd_tools.reduction import average
    public = {"AverageCommand", "AveragePendingPhase", "AverageRunnerPhase", "AverageScanPending", "AverageScanRecipe", "AverageScanPlan", "AverageScanProgress", "AverageScanResult", "AverageScanRunner", "AverageContributor", "AverageFiniteCounts", "AverageFiniteCountsEvidence"}; functions = {"iter_average_contributors"}
    assert public | functions <= set(vars(average)) and all(getattr(average, name).__module__ == average.__name__ for name in public)
    assert "AverageCleanupRequired" not in vars(average)
    assert "prepare_average_scan" not in vars(average)
    assert "average_finite_counts" not in record_writer.RecordWrite.__dataclass_fields__ and "average_finite_counts" in record_writer.WriterFinalization.__dataclass_fields__
    average_path = Path(average.__file__); root = average_path.parents[2]
    probe = subprocess.run([sys.executable, "-c", "import xdart.gui.tabs.scattering.page, xdart.gui.tabs.scattering.adapters.run_executor, xdart.gui.tabs.scattering.adapters.browse_loader, sys; assert 'xrd_tools.reduction.average' not in sys.modules"], cwd=root, check=False)
    assert probe.returncode == 0; owners, nodes, calls, edges = {}, {}, {}, []
    class Census(ast.NodeVisitor):
        def __init__(self, path): self.path, self.scope = path, []
        def visit_FunctionDef(self, node): owners.setdefault(node.name, []).append(self.path); nodes[self.path, node.name] = node; self.scope.append(node.name); self.generic_visit(node); self.scope.pop()
        visit_AsyncFunctionDef = visit_FunctionDef
        def visit_ClassDef(self, node): owners.setdefault(node.name, []).append(self.path); self.generic_visit(node)
        def visit_Call(self, node):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
            calls.setdefault(name, []).append(self.path); edges.append((self.path, self.scope[-1] if self.scope else None, name)); self.generic_visit(node)
    for path in root.rglob("*.py"): Census(path).visit(ast.parse(path.read_text()))
    assert all(owners[name] == [average_path] for name in public | functions)
    read_path, nexus_path = root / "xrd_tools/io/read.py", root / "xrd_tools/io/nexus_record.py"; writer_path = root / "xrd_tools/io/record_writer.py"; request_path = root / "xdart/gui/tabs/scattering/adapters/external_operation.py"; run_path = root / "xdart/gui/tabs/scattering/adapters/run_executor.py"; graph_path = root / "xrd_tools/sources/execution_graph.py"
    assert owners["get_average_finite_counts"] == [read_path] and owners["write_average_finite_counts"] == [nexus_path]
    assert [(path, scope) for path, scope, name in edges if name == "write_average_finite_counts"] == [(writer_path, "_write_finalization")] and owners["_AverageRequest"] == [request_path]
    assert calls.get("prepare_average_scan", ()) == ()
    assert calls["AverageScanRunner"] == [request_path]
    assert calls["for_finite_document"].count(average_path) == 1
    assert average_path not in calls.get("NexusSink", ())
    assert average_path not in calls.get("NexusRecordWriter", ())
    assert average_path not in calls.get("OutputTransaction", ())
    run_scope = next(scope for path, scope, name in edges if path == request_path and name == "AverageScanRunner")
    assert "average" in run_scope.lower() and sum(path == request_path and scope == run_scope and name == "_seal_publication" for path, scope, name in edges) == 1
    page_path = root / "xdart/gui/tabs/scattering/page.py"
    page_edges = [(scope, name) for path, scope, name in edges if path == page_path and name == "begin_average"]
    assert len(page_edges) == 1 and "average" in page_edges[0][0].lower()
    forbidden = {"AverageScanRecipe", "single_image_spec", "image_series_spec", "qualify_source_execution_graph", "read_image_metadata", "stat", "open", "iterdir"}
    assert forbidden.isdisjoint(name for path, scope, name in edges if
        (path, scope) in {(request_path, "begin_average"), (page_path, page_edges[0][0])})
    trees = tuple(ast.parse(path.read_text()) for path in (average_path, graph_path)); imports = {name for tree in trees for node in ast.walk(tree) for name in (([alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []))}
    assert not any(name == "xdart" or name.startswith(("xdart.", "qtpy", "pyqtgraph", "PyQt", "PySide")) for name in imports) and not any(name == "h5py" or name.startswith("h5py.") for node in ast.walk(trees[0]) for name in (([alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []))) and not ({average_path, request_path} & set(calls.get("create_dataset", ()))) and not ({average_path, request_path} & set(calls.get("File", ()))) and not ({"read_image", "openimage", "data"} & {node.attr if isinstance(node, ast.Attribute) else node.id for node in ast.walk(nodes[root / "xrd_tools/io/image.py", "read_detector_image_layout"]) if isinstance(node, (ast.Attribute, ast.Name))}) and all(not ({average_path, graph_path, request_path} & set(calls.get(name, ()))) for name in {"open_source", "source_for_kind", "NexusStackSource", "metadata_for", "scan_table", "motors", "write_contributing_frames"})
    for path, scope in ((writer_path, "write"), (writer_path, "write_batch"),
        (root / "xdart/gui/tabs/scattering/adapters/dynamic_output.py", "submit"),
        (run_path, "_execute_current"), (run_path, "_background_ready"), (run_path, "_submit_container_source")):
        assert "average_finite_counts" not in {node.id if isinstance(node, ast.Name) else node.attr for node in ast.walk(nodes[path, scope]) if isinstance(node, (ast.Name, ast.Attribute))}
