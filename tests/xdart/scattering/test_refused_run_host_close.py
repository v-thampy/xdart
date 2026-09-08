"""Direct host-close discriminator for an untouched or refused first Run."""

import time

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets
import tifffile

from xdart import _gui_main
from xdart.gui.pages.catalog import SCATTERING_WORKSPACE_PAGE
from xdart.gui.pages.values import PageCleanup
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xrd_tools.sources.selection import image_series_spec


@pytest.mark.parametrize("refuse_run", [False, True], ids=["idle", "refused-first-run"])
def test_real_host_quit_immediately_after_first_run_refusal(tmp_path, monkeypatch, refuse_run):
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setenv("XDART_SESSION_FILE", str(tmp_path / "host-session.json"))
    monkeypatch.setenv("XDART_SETTINGS_FILE", str(tmp_path / "host-settings.ini"))
    window = _gui_main.Main(
        page_descriptors=(SCATTERING_WORKSPACE_PAGE,),
        selected_page_key=SCATTERING_WORKSPACE_PAGE.key,
    )
    page = window.main_widget
    terminal_events, closed, cleared, terminated = [], [], [], []
    real_drain = page._run_executor.drain_events
    real_close = page.close_workspace
    real_clear = page._shell.scientific.clear_workspace

    def observe_events():
        events = real_drain()
        terminal_events.extend(events)
        return events

    def observe_close():
        receipt = real_close()
        closed.append(receipt)
        return receipt

    def observe_clear():
        result = real_clear()
        cleared.append(result)
        return result

    monkeypatch.setattr(page._run_executor, "drain_events", observe_events)
    monkeypatch.setattr(page, "close_workspace", observe_close)
    monkeypatch.setattr(page._shell.scientific, "clear_workspace", observe_clear)
    # The real Exit action, closeEvent, page closer and cleanup all run; only
    # the final process-termination side effect is observed instead of killing pytest.
    monkeypatch.setattr(window, "_terminate_process", lambda: terminated.append(True))
    try:
        if refuse_run:
            raw = tmp_path / "raw_0001.tif"
            tifffile.imwrite(raw, np.arange(195 * 487, dtype=np.uint16).reshape(195, 487))
            poni = tmp_path / "calibration.poni"
            poni.write_text(
                "poni_version: 2\nDetector: Pilatus100k\nDetector_config: {}\n"
                "Distance: 0.1234\nPoni1: 0.01\nPoni2: 0.01\n"
                "Rot1: 0.0\nRot2: 0.0\nRot3: 0.0\nWavelength: 1.0e-10\n",
                encoding="utf-8",
            )
            target = tmp_path / "failed_int2d.nexus"
            original = b"not a processed result; must not be replaced"
            target.write_bytes(original)
            prior = page._intents.snapshot()
            candidate = prior.thaw()
            candidate.source_spec = image_series_spec(raw)
            candidate.poni_file = str(poni)
            candidate.project_root = str(tmp_path)
            candidate.save_path = str(tmp_path / "failed.nexus")
            candidate.output_mode = "Overwrite"
            candidate.processing_mode = "Int 2D"
            candidate.max_cores = 1
            candidate.bai_1d_args = {"npt": 16}
            candidate.bai_2d_args = {"npt_rad": 16, "npt_azim": 8}
            accepted = page._intents.commit(candidate, expected_revision=prior.revision)
            page._reconcile_snapshot(prior, accepted.snapshot)
            page._shell.run_controls.startButton.click()
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                qapp.processEvents()
                if any(event.kind is StandardEventKind.FAILED for event in terminal_events):
                    break
                time.sleep(0.005)
            terminal = next((event for event in terminal_events
                             if event.kind is StandardEventKind.FAILED), None)
            assert terminal is not None, (page._lifecycle.phase, page._notice_text)
            assert terminal.completed == 0
            assert terminal.cleanup_status is CleanupStatus.CLEANED, terminal
            assert "not a current xdart processed result" in terminal.primary.message.casefold()
            assert target.read_bytes() == original
            assert page._lifecycle.phase is RunPhase.FAILED
            assert page._context_controller.acquisition_context is None

        window.ui.actionExit.trigger()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not terminated:
            qapp.processEvents()
            time.sleep(0.005)
        print("HOST_CLOSE", refuse_run, "cleared", cleared,
              "receipts", [(item.cleanup_status, item.cleanup_failures) for item in closed],
              "terminated", terminated, "phase", page._lifecycle.phase,
              "retry_scheduled", window._close_retry_scheduled)
        assert terminated == [True], (cleared, closed, page._notice_text)
        assert closed[-1].cleanup_status is CleanupStatus.CLEANED
        assert window.page_handle.close().status is PageCleanup.CLEAN
        assert not window._close_retry_scheduled
        assert not window.isVisible()
        assert page._context_controller.retained_contexts == ()
    finally:
        window._process_exit_requested = False
        window.close()
        window.deleteLater()
        qapp.processEvents()
