"""Results-notebook export stays tied to real selected result files."""

from __future__ import annotations

import time
from pathlib import Path

import nbformat
import numpy as np
from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.results_notebook import results_notebook_text
from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from xrd_tools.io.export import write_xye
from xrd_tools.reduction.core import Frame, FrameReduction, Scan
from xrd_tools.reduction import NexusSink
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent
from tests.core.reintegrate_support import _seed_existing


def _notebook_namespace(text: str) -> tuple[nbformat.NotebookNode, dict]:
    notebook = nbformat.reads(text, as_version=4)
    nbformat.validate(notebook)
    namespace = {}
    for cell in notebook.cells:
        if cell.cell_type == "code":
            exec(cell.source, namespace)
    return notebook, namespace


def _small_nexus(path):
    axis = np.array([0.1, 0.2, 0.3])
    frame = Frame(index=7, image=np.zeros((2, 2)))
    sink = NexusSink(path=path, overwrite=True)
    sink.begin(Scan(name="selected", frames=[frame]), plan=None)
    sink.write(
        frame,
        FrameReduction(
            frame_index=7,
            result_1d=IntegrationResult1D(axis, np.array([2., 3., 5.])),
            result_2d=IntegrationResult2D(
                axis, np.array([-1., 1.]), np.arange(6.).reshape(3, 2),
            ),
        ),
    )
    sink.finish(result=None)


def test_results_notebook_uses_public_readers_for_real_nexus_and_xye(tmp_path):
    nexus_path = tmp_path / "selected.nexus"
    xye_path = tmp_path / "selected.xye"
    _small_nexus(nexus_path)
    write_xye(xye_path, [1., 2., 3.], [4., 5., 6.])

    for kind, paths in (("nexus", (str(nexus_path),)), ("xye", (str(xye_path),))):
        notebook, namespace = _notebook_namespace(
            results_notebook_text(kind=kind, paths=paths)
        )
        assert notebook.nbformat == 4
        assert all(cell.execution_count is None and not cell.outputs
                   for cell in notebook.cells if cell.cell_type == "code")
        assert "xdart" not in "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code"
        )
        assert namespace["RESULT_PATHS"] == tuple(Path(path) for path in paths)
        if kind == "nexus":
            assert namespace["pattern"].intensity.tolist() == [2., 3., 5.]
            assert namespace["read_selected_cake"]().intensity.shape == (2, 3)
        else:
            assert namespace["xye_results"][0]["intensity"].tolist() == [4., 5., 6.]


def _wait_for_viewer(app, page) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        app.processEvents()
        context = page._context_controller.viewer_1d_context
        if context is not None and context.state.value == "ready":
            return
        time.sleep(0.005)
    raise AssertionError("real Viewer1D did not become ready")


def _help_action(page) -> None:
    help_button = page.findChild(QtWidgets.QToolButton, "helpMenuButton")
    action = next(
        item for item in help_button.menu().actions()
        if item.text() == "Export Analyze Results Notebook"
    )
    action.trigger()


def _close_workspace(app, page) -> None:
    for _ in range(100):
        receipt = page.close_workspace()
        if receipt.cleanup_status is CleanupStatus.CLEANED:
            break
        app.processEvents()
        time.sleep(0.01)
    assert receipt.cleanup_status is CleanupStatus.CLEANED
    page.deleteLater()


def test_help_export_notebook_uses_real_selected_viewer_xye_paths(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    paths = tuple(str(tmp_path / f"curve_{index}.xye") for index in range(2))
    for index, path in enumerate(paths):
        write_xye(path, [1., 2., 3.], [index + 1., index + 2., index + 3.])
    original = Path(paths[0]).read_bytes()
    requested = Path(paths[0])
    destination = requested.with_name(f"{requested.name}.ipynb")
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(
            processing_mode="1D Viewer", project_root=str(tmp_path),
        )),
        lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
        results_notebook_chooser=lambda _start: str(requested),
    )
    page.resize(1400, 1000)
    page.show()
    try:
        page._open_viewer_1d_paths(paths)
        _wait_for_viewer(app, page)
        _help_action(page)
        app.processEvents()
        notebook, namespace = _notebook_namespace(destination.read_text())
        assert notebook.nbformat == 4
        assert namespace["RESULT_KIND"] == "xye"
        assert namespace["RESULT_PATHS"] == tuple(Path(path) for path in paths)
        assert Path(paths[0]).read_bytes() == original
    finally:
        _close_workspace(app, page)


def test_help_export_notebook_uses_real_loaded_browse_nexus(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    seeded = _seed_existing(tmp_path)
    destination = tmp_path / "browse-results.ipynb"
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(project_root=str(tmp_path))),
        lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
        results_notebook_chooser=lambda _start: str(destination),
    )
    page.show()
    try:
        page._context_controller.begin_browse(str(seeded.target.resolve()))
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            app.processEvents()
            page._drain_executor()
            if page._capture_current_loaded_browse() is not None:
                break
            time.sleep(0.005)
        assert page._capture_current_loaded_browse() is not None
        _help_action(page)
        app.processEvents()
        _notebook, namespace = _notebook_namespace(destination.read_text())
        assert namespace["RESULT_KIND"] == "nexus"
        assert namespace["RESULT_PATHS"] == (seeded.target.resolve(),)
    finally:
        _close_workspace(app, page)
