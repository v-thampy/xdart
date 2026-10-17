"""One fresh Jupyter kernel executes the emitted ordinary call and mounts the shared controls."""
from __future__ import annotations

import json
import sys

import numpy as np


def test_export_runs_in_fresh_jupyter_kernel(tmp_path, monkeypatch):
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from xrd_tools.analysis.fit_snapshot import FitAxis, FitInputKey, FitInputSnapshot
    from xrd_tools.notebooks import FitNotebookRecipe, capture_snapshot, render_notebook
    x = np.linspace(1., 4., 101)
    snapshot = FitInputSnapshot(FitInputKey("fresh kernel member 7", "derived", "receipt", 1),
                                (FitAxis("q", "A^-1", x),), 3.+10.*np.exp(-.5*((x-2.5)/.08)**2))
    captured = capture_snapshot(snapshot, "input.npz")
    (tmp_path / "input.npz").write_bytes(captured.content)
    recipe = FitNotebookRecipe("peak", captured.reference, settings={"model": "gaussian", "selection_mode": "count", "n_peaks": 1})
    notebook = nbformat.reads(render_notebook(recipe, format="jupyter", bundle_root=str(tmp_path)), as_version=4)
    # Use this task's Python, not a user's registered kernel; no kernelspec is installed in their profile.
    kernel_directory = tmp_path / "kernels" / "fit-notebook"
    kernel_directory.mkdir(parents=True)
    (kernel_directory / "kernel.json").write_text(json.dumps({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                                                             "display_name": "Task notebook kernel", "language": "python"}))
    kernel = KernelManager(kernel_name="fit-notebook", kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(tmp_path / "kernels")]))
    client = NotebookClient(notebook, km=kernel, timeout=60)
    client.owns_km = True  # the standard client's context owns normal kernel shutdown and connection cleanup
    monkeypatch.setenv("JUPYTER_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("JUPYTER_CONFIG_DIR", str(tmp_path / "config"))
    notebook.cells.append(nbformat.v4.new_code_cell("assert outcome.ok, outcome.message\nassert fit_viewer.outcome is outcome\nassert abs(outcome.params['center_0'] - 2.5) < .01\nfit_viewer.dispose()"))
    executed = client.execute(cwd=str(tmp_path.parent))
    assert len(executed.cells) == 5
    assert not any(output.output_type == "error" for cell in executed.cells if cell.cell_type == "code" for output in cell.outputs)
