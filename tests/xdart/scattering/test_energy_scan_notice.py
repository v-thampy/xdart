"""PRE-P3-BATCH-20261010: an energy scan Int refuses at run start opens a notification with the advice.

Maintainer, October 10, on the parallax refusal: "there should be a notification for that with advice to turn it off
to allow processing". Each run is real: the page's Run button, the production executor and its energy binder.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import PoniV3OverrideIntent
from xrd_tools.sources import image_series_spec
from tests.xdart.scattering.test_energy_scan_run import _series
from tests.xdart.scattering.test_p1b_output_graph import _intent, write_poni
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell

_TITLE = "Energy scan not processed"


@pytest.fixture
def qapp() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _failed_run(qapp, monkeypatch, root: Path, energies_keV, *, parallax: bool,
                mask_shape=None, poni_detector=None, attempts=1, stop=False):
    """Click Run on a txt-energy TIFF series; return the failed event and the page's notification boxes."""
    root.mkdir(parents=True, exist_ok=True)
    raw = _series(root, energies_keV)
    poni = root / "cal.poni"
    write_poni(poni)
    if poni_detector is not None:
        poni.write_text(poni.read_text().replace("Pilatus100k", poni_detector))
    intent = _intent(raw, root / "escan.nexus", poni)
    intent.live_mode = stop
    intent.source_spec = image_series_spec(raw, metadata_format="txt")
    intent.bai_1d_args["unit"] = "q_A^-1"
    if parallax:
        intent.poni_v3_override = PoniV3OverrideIntent("Si", 450e-6, True)
    if mask_shape is not None:
        mask = root / "wrong-mask.npy"
        np.save(mask, np.zeros(mask_shape, dtype=bool))
        intent.mask_file = str(mask)
    executor = StandardRunExecutor(join_timeout=2.0)
    page = ScatteringWorkspace(intents=RunIntentStore(intent), lifecycle=ScatteringCoordinator(),
                               sources=FilesystemSourceAdapter(), executor=executor)
    observed = []
    real_drain = executor.drain_events

    def observe_events():
        events = real_drain()
        observed.extend(events)
        return events

    monkeypatch.setattr(executor, "drain_events", observe_events)
    try:
        shell = page.findChild(ScatteringWorkspaceShell)
        assert shell is not None
        results = []
        terminal_kinds = {StandardEventKind.FAILED, StandardEventKind.STOPPED, StandardEventKind.FINISHED}
        for _attempt in range(attempts):
            observed.clear()
            shell.run_controls.startButton.click()
            deadline = time.monotonic() + 60
            stop_clicked = False
            while time.monotonic() < deadline and not any(event.kind in terminal_kinds for event in observed):
                qapp.processEvents()
                if stop and not stop_clicked and any(event.kind is StandardEventKind.FRAME_READY for event in observed):
                    shell.run_controls.stopButton.click()
                    stop_clicked = True
                time.sleep(0.01)
            terminal = [event for event in observed if event.kind in terminal_kinds]
            assert len(terminal) == 1, (page._notice_text, tuple((event.kind, event.detail) for event in observed))
            event = terminal[0]
            assert event.cleanup_status is CleanupStatus.CLEANED
            for _ in range(5):
                qapp.processEvents()
            # The box's text, not its window title: Qt leaves message-box titles empty on macOS.
            boxes = [box for box in page.findChildren(QtWidgets.QMessageBox) if box.isVisible()]
            results.append((event, [(box.text(), box.informativeText()) for box in boxes]))
            if not stop:
                run = executor._active
                assert run is not None
                executor._terminal_event(run, event.kind, executor._receipt(run), event.completed, event.total)
                assert real_drain() == ()  # the same real run cannot emit/log another terminal
                executor._events.put(event)
                page._drain_executor()
                assert [box for box in page.findChildren(QtWidgets.QMessageBox) if box.isVisible()] == boxes
                for box in boxes:
                    box.close()
                qapp.processEvents()
                executor._events.put(event)
                page._drain_executor()
                assert not any(box.isVisible() for box in page.findChildren(QtWidgets.QMessageBox))
        return results
    finally:
        for box in page.findChildren(QtWidgets.QMessageBox):
            box.close()
        page.close()
        page.deleteLater()
        qapp.processEvents()


def test_a_parallax_energy_scan_opens_the_notification_with_the_advice(qapp, monkeypatch, tmp_path):
    (failed, boxes), = _failed_run(qapp, monkeypatch, tmp_path / "parallax", (16, 17), parallax=True)
    primary = failed.primary
    assert (primary.type_module, primary.type_qualname) == ("xrd_tools.session.readiness", "EnergyScanRefused")
    assert len(boxes) == 1
    text, advice = boxes[0]
    assert text == _TITLE
    assert "Detector sensor / parallax" in advice and "Enable parallax" in advice
    assert not (tmp_path / "parallax" / "escan.nexus").exists()  # refused before any frame was reduced


def test_an_ordinary_failed_run_opens_no_energy_scan_notification(qapp, monkeypatch, tmp_path):
    (failed, boxes), = _failed_run(qapp, monkeypatch, tmp_path / "mask", (16, 16), parallax=False, mask_shape=(2, 2))
    assert failed.primary is not None and failed.primary.type_qualname != "EnergyScanRefused"
    assert all(title != _TITLE for title, _detail in boxes)


@pytest.mark.parametrize("mismatch", ["explicit-mask", "poni-detector"])
def test_run_failure_reason_is_visible_logged_once_and_rearmed_for_the_next_run(
    qapp, monkeypatch, tmp_path, caplog, mismatch,
):
    options = {"mask_shape": (2, 2)} if mismatch == "explicit-mask" else {"poni_detector": "Eiger4M"}
    results = _failed_run(qapp, monkeypatch, tmp_path / mismatch, (16, 16),
                          parallax=False, attempts=2, **options)
    assert results[0][0].run_identity is not results[1][0].run_identity
    for failed, boxes in results:
        assert failed.kind is StandardEventKind.FAILED and failed.completed == 0
        assert failed.primary is not None
        assert "does not match image shape (195, 487)" in failed.detail
        assert "PONI calibration detector" in failed.detail and "selected mask" in failed.detail
        assert boxes == [("Run failed", failed.detail)]
    errors = [record for record in caplog.records
              if record.name.endswith("adapters.run_executor") and record.levelname == "ERROR"]
    assert len(errors) == 2
    assert all(failed.detail in record.getMessage() and failed.primary.type_qualname in record.getMessage()
               for (failed, _boxes), record in zip(results, errors, strict=True))


def test_stopped_run_has_no_failure_notification_or_error_log_and_keeps_its_prefix(
    qapp, monkeypatch, tmp_path, caplog,
):
    from xrd_tools.io import FrameViewReader
    (stopped, boxes), = _failed_run(qapp, monkeypatch, tmp_path / "stop", (16, 16), parallax=False, stop=True)
    assert stopped.kind is StandardEventKind.STOPPED and stopped.completed > 0
    assert boxes == []
    assert not [record for record in caplog.records
                if record.name.endswith("adapters.run_executor") and record.levelname == "ERROR"]
    with FrameViewReader(stopped.artifact) as reader:
        assert len(reader.labels()) == stopped.completed
