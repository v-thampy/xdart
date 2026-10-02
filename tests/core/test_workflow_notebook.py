"""Direct page-Run notebook contract: literal configuration and an explicit processing gate."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys

import pytest


def test_workflow_codec_preserves_literal_configuration_without_optional_imports():
    from xrd_tools.notebooks.workflow import WorkflowRecipe, workflow_to_json, workflow_from_json
    configuration = {"source": "folder/'α'\\frames", "note": "__import__('os').system('false')", "nested": {"on": True, "values": [1, 2.5, None]}}
    recipe = WorkflowRecipe("int", configuration)
    restored = workflow_from_json(workflow_to_json(recipe))
    assert restored == recipe and restored.configuration == configuration
    configuration["nested"]["values"].append(4)
    assert restored.configuration != configuration
    with pytest.raises((TypeError, ValueError)):
        WorkflowRecipe("int", {"not_plain": object()})
    with pytest.raises((TypeError, ValueError)):
        WorkflowRecipe("int", {"nonfinite": float("nan")})
    probe = subprocess.run([sys.executable, "-c", "import sys; import xrd_tools.notebooks.workflow; assert not any(n.startswith(('PyQt', 'PySide', 'pyqtgraph', 'ipywidgets', 'marimo')) for n in sys.modules)"],
                           capture_output=True, text=True)
    assert probe.returncode == 0, probe.stderr


@pytest.mark.parametrize("format", ["jupyter", "marimo"])
def test_workflow_notebook_open_and_run_all_never_process_without_activation(tmp_path, format):
    from xrd_tools.notebooks.workflow import WorkflowRecipe
    from xrd_tools.notebooks.workflow_render import render_workflow_notebook
    data_root = tmp_path / "project"
    data_root.mkdir()
    recipe = WorkflowRecipe("int", {"missing source": "never-open-this.tif", "literal": "'; raise AssertionError #"})
    path = tmp_path / ("Run α.ipynb" if format == "jupyter" else "Run α.py")
    document = render_workflow_notebook(recipe, format=format, data_root=str(data_root), name=path.stem)
    assert not (data_root / (path.stem + "_outputs")).exists()
    if format == "jupyter":
        notebook = json.loads(document)
        namespace = {"NOTEBOOK_HEADLESS": True}
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                exec(cell["source"], namespace)
        assert namespace["RUN_PROCESSING"] is False and namespace["outputs"] == ()
        assert namespace["CONFIGURATION"] == recipe.configuration
        namespace["RUN_PROCESSING"] = "yes"
        processing = next(cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code" and "RUN_GATE_BOOLEAN_REQUIRED" in cell["source"])
        with pytest.raises(TypeError, match="RUN_GATE_BOOLEAN_REQUIRED"):
            exec(processing, namespace)
        namespace["RUN_PROCESSING"] = True
        namespace["OUTPUT_ROOT"] = data_root
        with pytest.raises(FileExistsError, match="OUTPUT_EXISTS"):
            exec(processing, namespace)  # explicit activation reaches the real processing entry point
    else:
        path.write_text(document)
        spec = importlib.util.spec_from_file_location("native_workflow_export", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _, definitions = module.app.run()
        assert definitions["run_processing"].value is False
        assert definitions["get_outputs"]() == ()
        assert definitions["recipe"].configuration == recipe.configuration
    assert not (data_root / (path.stem + "_outputs")).exists()
    assert list(data_root.iterdir()) == []


def test_workflow_existing_output_refuses_before_dispatch(tmp_path):
    from xrd_tools.notebooks.workflow import WorkflowRecipe, run_workflow
    target = tmp_path / "existing"
    target.mkdir()
    with pytest.raises(FileExistsError, match="OUTPUT_EXISTS"):
        run_workflow(WorkflowRecipe("int", {}), data_root=tmp_path, output_root=target)


def test_workflow_result_viewer_reuses_public_reader_and_display_owners(tmp_path):
    import numpy as np
    from xrd_tools.io import write_xye
    from xrd_tools.gui.widgets import PatternViewer, PatternSelector, FitRangeControls
    from xrd_tools.notebooks.workflow_viewer import WorkflowResultViewer
    x = np.linspace(1., 3., 21)
    paths = (tmp_path / "one.xye", tmp_path / "two.xye")
    for i, path in enumerate(paths):
        write_xye(path, x, 1. + i + x, variance=np.ones(x.shape))
    viewer = WorkflowResultViewer(paths, "int")
    try:
        assert isinstance(viewer.pattern, PatternViewer)
        assert isinstance(viewer.selector, PatternSelector)
        assert isinstance(viewer.ranges, FitRangeControls)
        assert len(viewer.ranges.widget.children) == 1  # presentation controls only, no unused fit editor
        viewer.selector.overlays.value = (1,)
        assert len(viewer.pattern.figure_widget.data) == 2
        np.testing.assert_allclose(viewer.pattern.figure_widget.data[1].y, 2. + x)
        viewer.selector.selection.value = 1
        assert viewer.snapshot.key.path == str(paths[1])
        viewer.ranges.display_min.value = 1.5
        assert tuple(viewer.pattern.figure_widget.layout.xaxis.range) == (1.5, 3.)
    finally:
        viewer.dispose()
        viewer.dispose()
