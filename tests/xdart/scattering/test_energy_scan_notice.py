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
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell

_TITLE = "Energy scan not processed"


@pytest.fixture
def qapp() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _failed_run(qapp, monkeypatch, root: Path, energies_keV, *, parallax: bool, mask_shape=None):
    """Click Run on a txt-energy TIFF series; return the failed event and the page's notification boxes."""
    root.mkdir(parents=True, exist_ok=True)
    raw = _series(root, energies_keV)
    poni = root / "cal.poni"
    write_poni(poni)
    intent = _intent(raw, root / "escan.nexus", poni)
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
        shell.run_controls.startButton.click()
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not any(event.kind is StandardEventKind.FAILED for event in observed):
            qapp.processEvents()
            time.sleep(0.01)
        failed = [event for event in observed if event.kind is StandardEventKind.FAILED]
        assert failed, (page._notice_text, tuple((event.kind, event.detail) for event in observed))
        for _ in range(5):
            qapp.processEvents()
        boxes = [box for box in page.findChildren(QtWidgets.QMessageBox)
                 if box.windowTitle() == _TITLE and box.isVisible()]
        return failed[0], [(box.text(), box.informativeText()) for box in boxes]
    finally:
        for box in page.findChildren(QtWidgets.QMessageBox):
            box.close()
        page.close()
        page.deleteLater()
        qapp.processEvents()


def test_a_parallax_energy_scan_opens_the_notification_with_the_advice(qapp, monkeypatch, tmp_path):
    failed, boxes = _failed_run(qapp, monkeypatch, tmp_path / "parallax", (16, 17), parallax=True)
    primary = failed.primary
    assert (primary.type_module, primary.type_qualname) == ("xrd_tools.session.readiness", "EnergyScanRefused")
    assert len(boxes) == 1
    text, advice = boxes[0]
    assert text == _TITLE
    assert "Detector sensor / parallax" in advice and "Enable parallax" in advice
    assert not (tmp_path / "parallax" / "escan.nexus").exists()  # refused before any frame was reduced


def test_an_ordinary_failed_run_opens_no_energy_scan_notification(qapp, monkeypatch, tmp_path):
    failed, boxes = _failed_run(qapp, monkeypatch, tmp_path / "mask", (16, 16), parallax=False, mask_shape=(2, 2))
    assert failed.primary is not None and failed.primary.type_qualname != "EnergyScanRefused"
    assert boxes == []
