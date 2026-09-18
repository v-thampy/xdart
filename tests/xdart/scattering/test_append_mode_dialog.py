"""A real stopped output needs explicit consent before incompatible replacement."""

from threading import Event
import time

import h5py
import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets
import tifffile

from tests.xdart.scattering._e2sd_support import write_poni
from tests.xdart.scattering.test_p1b_output_graph import _intent, _written
from xdart.gui.tabs.scattering.adapters.dynamic_output import DynamicOutputAdapter
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.controls_projection import GI_ENABLED
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xrd_tools.core.provenance import read_provenance
from xrd_tools.session.intent_store import RunIntentStore


@pytest.mark.parametrize("initial_gi,reopen", ((False, False), (True, False), (False, True)))
def test_stopped_mode_change_requires_cancel_or_overwrite(tmp_path, monkeypatch, initial_gi, reopen):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    raw = tmp_path / "scan_0001.tif"
    for label in range(1, 17):
        tifffile.imwrite(tmp_path / f"scan_{label:04d}.tif",
                         np.full((195, 487), label * 10, np.uint16))
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    target = tmp_path / "result.nexus"
    intent = _intent(raw, target, poni, processing_mode="Int 2D")
    intent.gi.enabled = initial_gi
    page = ScatteringWorkspace(
        intents=RunIntentStore(intent), lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(), executor=StandardRunExecutor(join_timeout=2.0),
    )
    entered, release = Event(), Event()
    original_submit = DynamicOutputAdapter.submit

    def paced_submit(adapter, frame, *args, **kwargs):
        if int(frame.index) == 9:
            entered.set()
            assert release.wait(30), "source pacing was not released"
        return original_submit(adapter, frame, *args, **kwargs)

    monkeypatch.setattr(DynamicOutputAdapter, "submit", paced_submit)

    def wait(predicate):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                return
            time.sleep(0.005)
        pytest.fail(page._notice_text or "Run/dialog did not settle")

    def dialog():
        return next((box for box in page.findChildren(QtWidgets.QMessageBox)
                     if box.isVisible()), None)

    try:
        page._shell.run_controls.startButton.click()
        wait(entered.is_set)
        page._shell.run_controls.stopButton.click()
        release.set()
        wait(lambda: page._capture_current_loaded_browse() is not None
             and page._start_permitted()[0])
        artifact = _written(target, "Int 2D")
        with h5py.File(artifact, "r") as handle:
            labels = handle["entry/integrated_1d/frame_index"][()]
            assert 0 < len(labels) < 16
        before = artifact.read_bytes()
        page._shell.run_controls.writeModeButton.click()
        page._on_field_value(GI_ENABLED, not initial_gi)
        wait(page._shell.run_controls.startButton.isEnabled)
        page._shell.run_controls.startButton.click()
        wait(lambda: dialog() is not None or (
            page._admission is None and page._notice_text.startswith("Output admission failed:")
        ))
        box = dialog()
        assert box is not None, page._notice_text
        assert "incompatible" in box.text().lower()
        cancel = box.button(QtWidgets.QMessageBox.StandardButton.Cancel)
        assert box.defaultButton() is cancel
        assert artifact.read_bytes() == before
        cancel.click()
        app.processEvents()
        assert page._intents.snapshot().thaw().output_mode == "Append"
        assert page._admission is None
        assert artifact.read_bytes() == before

        if reopen:
            current = page._intents.snapshot().thaw()
            wait(lambda: page.close_workspace().cleanup_status is CleanupStatus.CLEANED)
            page.deleteLater()
            app.processEvents()
            page = ScatteringWorkspace(
                intents=RunIntentStore(current), lifecycle=ScatteringCoordinator(),
                sources=FilesystemSourceAdapter(), executor=StandardRunExecutor(join_timeout=2.0),
            )
            wait(page._shell.run_controls.startButton.isEnabled)
        page._shell.run_controls.startButton.click()
        wait(lambda: dialog() is not None)
        box = dialog()
        overwrite = next(button for button in box.buttons() if button.text() == "Overwrite")
        overwrite.click()
        wait(lambda: page._capture_current_loaded_browse() is not None
             and len(page._capture_current_loaded_browse().labels) == 16
             and page._start_permitted()[0])
        assert page._intents.snapshot().thaw().output_mode == "Overwrite"
        with h5py.File(artifact, "r") as handle:
            np.testing.assert_array_equal(handle["entry/integrated_1d/frame_index"][()],
                                          np.arange(1, 17))
            np.testing.assert_array_equal(handle["entry/integrated_2d/frame_index"][()],
                                          np.arange(1, 17))
            assert np.isfinite(handle["entry/integrated_1d/intensity"][()]).all()
        saved = read_provenance(artifact)["config"]["run_configuration"]
        assert saved["gi"]["enabled"] is (not initial_gi)
    finally:
        release.set()
        for box in page.findChildren(QtWidgets.QMessageBox):
            box.reject()
        wait(lambda: page.close_workspace().cleanup_status is CleanupStatus.CLEANED)
        page.deleteLater()
        app.processEvents()
