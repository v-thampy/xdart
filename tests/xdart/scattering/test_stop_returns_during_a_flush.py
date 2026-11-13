"""C-1 of NEXT-1210-RESPONSIVENESS: Stop returns at once while the writer holds its file lock.

The plan (handoffs/review_fixes_1210_2026-11-11.md, C-1, from the 1.2.9 review's A2): Stop reached
``DynamicOutputAdapter.stop``, which took ``_command_lock``, the writer's ``file_lock``, so a Stop pressed during a
flush waited on the GUI thread for the flush. The window shows "Stopping run…" at once; the stop flag is set without
waiting for the writer; the writer's serialization, receipts and the page's pending states stay. Row: a real Int Run
whose writer is held inside its locked boundary while Stop is pressed; the click returns within a bound, and the Run
still ends STOPPED with its frames kept.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from threading import Event, Timer
import time

import numpy as np
from pyqtgraph.Qt import QtWidgets
import tifffile

from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xrd_tools.io.record_writer import NexusRecordWriter
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import image_series_spec

_TERMINAL = {StandardEventKind.FINISHED, StandardEventKind.STOPPED, StandardEventKind.FAILED}
_HOLD_AT = 4  # the writer's fourth boundary: frames are written before it, and it holds the file lock
_STOP_BOUND_S = 1.0
_PARENT_RELEASE_S = 5.0  # where Stop waits for the writer, the writer is let go after this, so the row ends


def _wait(app, predicate, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_stop_returns_at_once_while_the_writer_holds_its_lock(tmp_path, monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setenv("XDART_SESSION_FILE", str(tmp_path / "session.json"))
    total = 40
    for label in range(1, total + 1):
        tifffile.imwrite(tmp_path / f"raw_{label:04d}.tif", np.full((195, 487), label + 1, dtype=np.uint16))
    poni = tmp_path / "calibration.poni"
    poni.write_text("poni_version: 2\nDetector: Pilatus100k\nDetector_config: {}\nDistance: 0.1234\nPoni1: 0.01\n"
                    "Poni2: 0.01\nRot1: 0.0\nRot2: 0.0\nRot3: 0.0\nWavelength: 1.0e-10\n", encoding="utf-8")
    entered, release, boundaries = Event(), Event(), [0]
    original = NexusRecordWriter._boundary

    @contextmanager
    def held_boundary(writer):
        with original(writer):  # the file lock is held from here
            boundaries[0] += 1
            if boundaries[0] == _HOLD_AT:
                entered.set()
                assert release.wait(30.0), "writer boundary not released"
            yield

    monkeypatch.setattr(NexusRecordWriter, "_boundary", held_boundary)
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(
            source_spec=image_series_spec(tmp_path / "raw_0001.tif"), poni_file=str(poni),
            project_root=str(tmp_path), save_path=str(tmp_path / "run.nexus"), output_mode="Overwrite",
            processing_mode="Int 1D", max_cores=1, bai_1d_args={"npt": 16},
        )),
        lifecycle=ScatteringCoordinator(), sources=FilesystemSourceAdapter(),
        executor=StandardRunExecutor(join_timeout=2.0),
    )
    events = []
    original_drain = page._run_executor.drain_events

    def observe_events():
        update = original_drain()
        events.extend(update)
        return update

    monkeypatch.setattr(page._run_executor, "drain_events", observe_events)
    let_go = Timer(_PARENT_RELEASE_S, release.set)
    try:
        page._shell.run_controls.startButton.click()
        assert _wait(app, entered.is_set), page._notice_text
        let_go.start()
        started = time.monotonic()
        page._shell.run_controls.stopButton.click()
        took = time.monotonic() - started
        shown = page._progress.detail
        stalls = []  # the window stays live while the writer is still held: no other GUI path waits on its lock
        until = time.monotonic() + .5
        while time.monotonic() < until and not release.is_set():
            pumped = time.monotonic()
            app.processEvents()
            stalls.append(time.monotonic() - pumped)
            time.sleep(.01)
        writer_still_held = not release.is_set()
        release.set()
        assert writer_still_held and took < _STOP_BOUND_S, f"Stop took {took:.2f} s while the writer held its lock"
        assert stalls and max(stalls) < .25, f"the window stalled {max(stalls, default=0):.2f} s after Stop"
        assert shown == "Stopping run…"
        assert _wait(app, lambda: any(event.kind in _TERMINAL for event in events)), page._notice_text
        terminal = next(event for event in events if event.kind in _TERMINAL)
        assert terminal.kind is StandardEventKind.STOPPED, terminal
        assert terminal.cleanup_status is CleanupStatus.CLEANED
        assert 0 < terminal.completed < total, terminal
        assert _wait(app, lambda: page._capture_current_loaded_browse() is not None)
        capture = page._capture_current_loaded_browse()
        assert len(capture.labels) == terminal.completed  # the frames written before Stop are kept
        assert Path(capture.target).is_file()
    finally:
        let_go.cancel()
        release.set()
        assert _wait(app, lambda: page.close_workspace().cleanup_status is CleanupStatus.CLEANED)
        page.deleteLater()
        app.processEvents()
