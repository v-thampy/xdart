"""Real Qt/fixture acceptance cases for the corrected E1b Standard slice."""

from __future__ import annotations

import os
from pathlib import Path
import threading
import time

import pytest
from pyqtgraph.Qt import QtWidgets

from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import image_series_spec
from xdart.gui.tabs.scattering.adapters import run_executor as executor_module
from xdart.gui.tabs.scattering import output_preflight as preflight_module
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell


@pytest.fixture
def qapp() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _wait(app: QtWidgets.QApplication, predicate, timeout: float = 180.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _page(tmp_path: Path) -> tuple[ScatteringWorkspace, ScatteringCoordinator, Path]:
    corpus = os.environ.get("XDART_TEST_DATA")
    if not corpus:
        pytest.skip("XDART_TEST_DATA is required for the E1b Standard fixture")
    root = Path(corpus) / "Tiff"
    selected = root / "Combi4_Angledependence_samz_4p9_03271002_0001.tif"
    poni = root / "LaB6_detz190_dety72_th5_03261554_0001.poni"
    output = tmp_path / "standard.nxs"
    lifecycle = ScatteringCoordinator()
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(
            source_spec=image_series_spec(selected), poni_file=str(poni),
            project_root=str(root), save_path=str(output),
            output_mode="Overwrite", max_cores=1,
            bai_1d_args={"npt": 128}, bai_2d_args={"npt_rad": 128, "npt_azim": 64},
        )),
        lifecycle=lifecycle,
        sources=FilesystemSourceAdapter(),
        executor=StandardRunExecutor(),
    )
    return page, lifecycle, output


def test_real_standard_construction_is_off_the_qt_thread(
    qapp: QtWidgets.QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, threading.Thread]] = []
    real_open, real_poni = executor_module.open_source, preflight_module.load_poni

    def traced_open(source):
        calls.append(("source", threading.current_thread()))
        return real_open(source)

    def traced_poni(path):
        calls.append(("poni", threading.current_thread()))
        return real_poni(path)

    monkeypatch.setattr(executor_module, "open_source", traced_open)
    monkeypatch.setattr(preflight_module, "load_poni", traced_poni)
    page, lifecycle, output = _page(tmp_path)
    gui_thread = threading.current_thread()
    try:
        shell = page.findChild(ScatteringWorkspaceShell)
        assert shell is not None
        assert _wait(qapp, shell.run_controls.startButton.isEnabled)
        shell.run_controls.startButton.click()
        assert _wait(qapp, lambda: lifecycle.phase is RunPhase.IDLE)
        assert [name for name, _thread in calls] == ["poni", "source"]
        assert all(thread is not gui_thread for _name, thread in calls)
        assert output.is_file()
    finally:
        page.close_workspace()
