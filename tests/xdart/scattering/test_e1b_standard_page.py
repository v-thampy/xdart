"""Real Qt composition oracle for the one E1b Standard vertical slice."""

from __future__ import annotations

import os
from pathlib import Path
import time

import pytest
from pyqtgraph.Qt import QtWidgets

from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import image_series_spec
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell


def _wait(qapp: QtWidgets.QApplication, predicate, timeout: float = 180.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _shell(page: ScatteringWorkspace) -> ScatteringWorkspaceShell:
    shell = page.findChild(ScatteringWorkspaceShell)
    assert shell is not None
    return shell


@pytest.fixture
def qapp() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_real_run_click_projects_standard_values_without_page_io(
    qapp: QtWidgets.QApplication, tmp_path: Path,
) -> None:
    root = os.environ.get("XDART_TEST_DATA")
    if not root:
        pytest.skip("XDART_TEST_DATA is required for the E1b Standard fixture")
    tiff = Path(root) / "Tiff"
    selected = tiff / "Combi4_Angledependence_samz_4p9_03271002_0001.tif"
    poni = tiff / "LaB6_detz190_dety72_th5_03261554_0001.poni"
    if not selected.is_file() or not poni.is_file():
        pytest.skip("the frozen E1b Standard fixture is unavailable")
    intent = RunIntent(
        source_spec=image_series_spec(selected), poni_file=str(poni),
        project_root=str(tiff), save_path=str(tmp_path / "page-standard.nxs"),
        output_mode="Overwrite",
        max_cores=1, bai_1d_args={"npt": 128},
        bai_2d_args={"npt_rad": 128, "npt_azim": 64},
    )
    lifecycle = ScatteringCoordinator()
    page = ScatteringWorkspace(
        intents=RunIntentStore(intent), lifecycle=lifecycle,
        sources=FilesystemSourceAdapter(), executor=StandardRunExecutor(),
    )
    try:
        shell = _shell(page)
        shell.run_controls.startButton.click()
        assert _wait(qapp, lambda: lifecycle.phase is RunPhase.RUNNING)
        assert _wait(qapp, lambda: lifecycle.phase is RunPhase.IDLE)
        assert shell.run_controls.stopButton.isEnabled() is False
        assert shell.scientific.raw.image.image is not None
        assert shell.scientific.cake.image.image is not None
        assert shell.scientific.curve.listDataItems()
        member = shell.scientific.status.text()
        assert member == shell.scientific.title.text()
        assert Path(member).name == member
        assert member.lower().endswith((".tif", ".tiff"))
        assert "page-standard.nxs" not in member
    finally:
        page.close_workspace()


def test_stop_click_reports_stopped_not_false_normal_finish(
    qapp: QtWidgets.QApplication, tmp_path: Path,
) -> None:
    root = os.environ.get("XDART_TEST_DATA")
    if not root:
        pytest.skip("XDART_TEST_DATA is required for the E1b Standard fixture")
    tiff = Path(root) / "Tiff"
    selected = tiff / "Combi4_Angledependence_samz_4p9_03271002_0001.tif"
    poni = tiff / "LaB6_detz190_dety72_th5_03261554_0001.poni"
    if not selected.is_file() or not poni.is_file():
        pytest.skip("the frozen E1b Standard fixture is unavailable")
    lifecycle = ScatteringCoordinator()
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(
            source_spec=image_series_spec(selected), poni_file=str(poni),
            project_root=str(tiff), save_path=str(tmp_path / "stopped.nxs"),
            output_mode="Overwrite", max_cores=1,
            bai_1d_args={"npt": 128}, bai_2d_args={"npt_rad": 128, "npt_azim": 64},
        )),
        lifecycle=lifecycle, sources=FilesystemSourceAdapter(), executor=StandardRunExecutor(),
    )
    try:
        shell = _shell(page)
        shell.run_controls.startButton.click()
        assert _wait(qapp, lambda: lifecycle.phase is RunPhase.RUNNING)
        shell.run_controls.stopButton.click()
        assert _wait(qapp, lambda: lifecycle.phase is RunPhase.IDLE)
        terminal_status = shell.scientific.status.text().lower()
        assert "stopped" in terminal_status
        assert "finished" not in terminal_status
    finally:
        page.close_workspace()


def test_close_invalidates_run_before_executor_completion(
    qapp: QtWidgets.QApplication, tmp_path: Path,
) -> None:
    root = os.environ.get("XDART_TEST_DATA")
    if not root:
        pytest.skip("XDART_TEST_DATA is required for the E1b Standard fixture")
    tiff = Path(root) / "Tiff"
    selected = tiff / "Combi4_Angledependence_samz_4p9_03271002_0001.tif"
    poni = tiff / "LaB6_detz190_dety72_th5_03261554_0001.poni"
    if not selected.is_file() or not poni.is_file():
        pytest.skip("the frozen E1b Standard fixture is unavailable")
    lifecycle = ScatteringCoordinator()
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(
            source_spec=image_series_spec(selected), poni_file=str(poni),
            project_root=str(tiff), save_path=str(tmp_path / "close.nxs"),
            output_mode="Overwrite", max_cores=1,
            bai_1d_args={"npt": 128}, bai_2d_args={"npt_rad": 128, "npt_azim": 64},
        )),
        lifecycle=lifecycle, sources=FilesystemSourceAdapter(), executor=StandardRunExecutor(),
    )
    _shell(page).run_controls.startButton.click()
    assert _wait(qapp, lambda: lifecycle.phase is RunPhase.RUNNING)
    page.close_workspace()
    qapp.processEvents()

    assert lifecycle.closed is True
    assert lifecycle.phase is RunPhase.CLOSED
