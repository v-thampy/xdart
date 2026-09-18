"""GI-COMPANION-20260918 — Stop and Append with two direct 2-D maps.

A real stopped run from the real page.  Continuing with the SAME map set
appends every remaining frame to both maps; a CHANGED direct map set takes the
existing incompatible-output dialog and leaves the file untouched on Cancel.
"""

from __future__ import annotations

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
from xdart.gui.tabs.scattering.controls_inventory import INT_2D_AXIS
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xrd_tools.session.intent_store import RunIntentStore

FRAMES = 16
COMBINED = "Q-χ + Qip-Qoop"


@pytest.mark.parametrize(("first", "second", "compatible"), (
    (COMBINED, COMBINED, True),
    ("Qip-Qoop", COMBINED, False),
    (COMBINED, "Qip-Qoop", False),
))
def test_append_follows_the_direct_map_set(tmp_path, monkeypatch, first, second, compatible):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    for label in range(1, FRAMES + 1):
        tifffile.imwrite(tmp_path / f"scan_{label:04d}.tif",
                         np.full((195, 487), label * 10, np.uint16))
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    target = tmp_path / "result.nexus"
    intent = _intent(tmp_path / "scan_0001.tif", target, poni, processing_mode="Int 2D")
    intent.gi.enabled = True
    page = ScatteringWorkspace(
        intents=RunIntentStore(intent), lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(), executor=StandardRunExecutor(join_timeout=2.0),
    )
    page._on_field_value(INT_2D_AXIS, first)
    entered, release = Event(), Event()
    original_submit = DynamicOutputAdapter.submit

    def paced_submit(adapter, frame, *args, **kwargs):
        if int(frame.index) == 9 and not entered.is_set():
            entered.set()
            assert release.wait(30), "source pacing was not released"
        return original_submit(adapter, frame, *args, **kwargs)

    monkeypatch.setattr(DynamicOutputAdapter, "submit", paced_submit)

    def wait(predicate):
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                return
            time.sleep(0.005)
        pytest.fail(page._notice_text or "Run/dialog did not settle")

    def dialog():
        return next((box for box in page.findChildren(QtWidgets.QMessageBox)
                     if box.isVisible()), None)

    def maps(handle):
        top = handle["entry/integrated_2d"]
        return {"": list(top["frame_index"][()]), **{
            name: list(top[name]["frame_index"][()])
            for name in top if isinstance(top[name], h5py.Group)
        }}

    try:
        wait(page._shell.run_controls.startButton.isEnabled)
        page._shell.run_controls.startButton.click()
        wait(entered.is_set)
        page._shell.run_controls.stopButton.click()
        release.set()
        wait(lambda: page._capture_current_loaded_browse() is not None
             and page._start_permitted()[0])
        artifact = _written(target, "Int 2D")
        with h5py.File(artifact, "r") as handle:
            stopped = maps(handle)
        assert set(stopped) == ({"", "q_chi"} if first == COMBINED else {""})
        # Stop kept a durable prefix, and every map holds the SAME frames.
        assert 0 < len(stopped[""]) < FRAMES
        assert all(labels == stopped[""] for labels in stopped.values())
        before = artifact.read_bytes()

        page._shell.run_controls.writeModeButton.click()
        assert page._intents.snapshot().thaw().output_mode == "Append"
        if second != first:
            page._on_field_value(INT_2D_AXIS, second)
        wait(page._shell.run_controls.startButton.isEnabled)
        page._shell.run_controls.startButton.click()

        if compatible:
            wait(lambda: page._capture_current_loaded_browse() is not None
                 and len(page._capture_current_loaded_browse().labels) == FRAMES
                 and page._start_permitted()[0])
            assert dialog() is None
            with h5py.File(artifact, "r") as handle:
                finished = maps(handle)
                assert list(handle["entry/integrated_1d/frame_index"][()]) == list(range(1, FRAMES + 1))
                assert len(handle["entry/frames"]) == FRAMES
            assert set(finished) == {"", "q_chi"}
            for labels in finished.values():
                # One physical frame, once, in each map -- across the Append seam.
                assert labels == list(range(1, FRAMES + 1))
        else:
            wait(lambda: dialog() is not None or (
                page._admission is None
                and page._notice_text.startswith("Output admission failed:")
            ))
            box = dialog()
            assert box is not None, page._notice_text
            assert "incompatible" in box.text().lower()
            cancel = box.button(QtWidgets.QMessageBox.StandardButton.Cancel)
            assert box.defaultButton() is cancel
            cancel.click()
            app.processEvents()
            assert page._admission is None
            assert artifact.read_bytes() == before
    finally:
        release.set()
        for box in page.findChildren(QtWidgets.QMessageBox):
            box.reject()
        wait(lambda: page.close_workspace().cleanup_status is CleanupStatus.CLEANED)
        page.deleteLater()
        app.processEvents()
