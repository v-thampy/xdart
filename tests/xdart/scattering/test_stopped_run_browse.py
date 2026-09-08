"""A real saved partial Stop reaches authenticated Browse and external tools."""

from pathlib import Path
from threading import Event
import time

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets
import tifffile

from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xrd_tools.io import FrameViewReader
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.image import TiffSeriesSource
from xrd_tools.sources.selection import image_series_spec


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _wait(qapp, predicate, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return False


@pytest.mark.parametrize("stop_before", (65, 1), ids=("saved-prefix", "no-output"))
def test_real_partial_stop_qualifies_nexpy_only_after_saved_browse(
    qapp, tmp_path, monkeypatch, stop_before,
):
    """Only raw-source pacing is controlled; Run, sink, Stop and Browse are real."""

    image = np.arange(195 * 487, dtype=np.uint16).reshape(195, 487)
    for label in range(1, 81):
        tifffile.imwrite(tmp_path / f"raw_{label:04d}.tif", image)
    poni = tmp_path / "calibration.poni"
    poni.write_text(
        "poni_version: 2\nDetector: Pilatus100k\nDetector_config: {}\n"
        "Distance: 0.1234\nPoni1: 0.01\nPoni2: 0.01\n"
        "Rot1: 0.0\nRot2: 0.0\nRot3: 0.0\nWavelength: 1.0e-10\n",
        encoding="utf-8",
    )
    entered, release = Event(), Event()
    read_path = TiffSeriesSource._read_path

    def paced_read(source, path):
        if Path(path).stem == f"raw_{stop_before:04d}":
            entered.set()
            assert release.wait(20.0), "raw-source pacing was not released"
        return read_path(source, path)

    monkeypatch.setattr(TiffSeriesSource, "_read_path", paced_read)
    executor = StandardRunExecutor(join_timeout=2.0)
    observed = []
    drain_events = executor.drain_events

    def observe_events():
        events = drain_events()
        observed.extend(events)
        return events

    monkeypatch.setattr(executor, "drain_events", observe_events)
    output = tmp_path / "partial_int2d.nexus"
    lifecycle = ScatteringCoordinator()
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(
            source_spec=image_series_spec(tmp_path / "raw_0001.tif"),
            poni_file=str(poni), project_root=str(tmp_path),
            save_path=str(tmp_path / "partial.nexus"), output_mode="Overwrite", max_cores=1,
            processing_mode="Int 2D", bai_1d_args={"npt": 16},
            bai_2d_args={"npt_rad": 16, "npt_azim": 8},
        )),
        lifecycle=lifecycle, sources=FilesystemSourceAdapter(), executor=executor,
    )
    try:
        shell = page._shell
        button = shell.tools.findChild(QtWidgets.QPushButton, "e3Tool_nexpy_selected")
        assert button is not None and not button.isEnabled()
        shell.run_controls.startButton.click()
        assert _wait(qapp, entered.is_set), page._notice_text
        assert lifecycle.phase is RunPhase.RUNNING
        assert not button.isEnabled()
        assert page._qualify_external_nexus(validate_disk=True).target is None
        chosen = None
        if stop_before > 1:
            assert _wait(qapp, lambda: len(page._context_controller.frame_keys) >= 8)
            chosen = page._context_controller.frame_keys[3]
            page._handle_shell_command(ShellCommand(ShellCommandKind.SET_AUTO_LAST, False))
            page._handle_shell_command(ShellCommand(
                ShellCommandKind.SELECT_FRAME, frame=chosen, frames=(chosen,),
            ))
            assert page._context_controller.navigation.current is chosen
        shell.run_controls.stopButton.click()
        release.set()
        assert _wait(qapp, lambda: any(
            event.kind in {StandardEventKind.STOPPED, StandardEventKind.FAILED}
            for event in observed
        )), page._notice_text
        terminal = next(event for event in observed if event.kind in {
            StandardEventKind.STOPPED, StandardEventKind.FAILED,
        })
        assert terminal.kind is StandardEventKind.STOPPED, terminal
        assert terminal.cleanup_status is CleanupStatus.CLEANED
        assert terminal.terminal_commit_identity is None
        assert lifecycle.phase is RunPhase.IDLE
        if chosen is None:
            assert terminal.artifact_completed == 0
            assert not output.exists()
            assert page._capture_current_loaded_browse() is None
            assert not button.isEnabled()
            assert shell.run_controls.startButton.isEnabled()
            return
        assert 0 < terminal.artifact_completed < terminal.artifact_total
        assert terminal.artifact == str(output)
        assert terminal.artifact in terminal.artifacts
        with FrameViewReader(output) as reader:
            expected = reader.read(chosen.local_frame_label)
        assert _wait(qapp, lambda: page._capture_current_loaded_browse() is not None), (
            terminal, page._notice_text,
        )
        capture = page._capture_current_loaded_browse()
        assert capture.target == str(output)
        assert capture.request.terminal_commit_identity is None
        assert _wait(qapp, button.isEnabled), button.toolTip()
        assert _wait(qapp, shell.run_controls.startButton.isEnabled)
        assert page._qualify_external_nexus(validate_disk=True).target == str(output)
        navigation = page._context_controller.navigation
        assert navigation.current.local_frame_label == chosen.local_frame_label
        assert tuple(key.local_frame_label for key in navigation.selected) == (
            chosen.local_frame_label,
        )
        assert _wait(qapp, lambda: bool(shell.scientific.curve.listDataItems()))
        np.testing.assert_allclose(
            shell.scientific.curve.listDataItems()[0].getData()[1],
            expected.intensity_1d,
        )
        assert shell.scientific.raw.image.image is not None
        assert shell.scientific.cake.image.image is not None
    finally:
        release.set()
        assert _wait(qapp, lambda: page.close_workspace().cleanup_status is CleanupStatus.CLEANED)
        page.deleteLater()
        qapp.processEvents()
