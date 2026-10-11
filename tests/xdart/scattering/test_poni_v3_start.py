"""PRE-P3-BATCH-20261010, found while adding the parallax notification: an Int run with a PONI v3 calibration (sensor
and parallax), or with the Advanced parallax override, never started (pre-existing since e90c3adb).

Admission signs the accepted PONI projection into its candidate configuration; the start must freeze the same one, or
the executor refuses the start ("executor_start_failed") before any frame is read. Each run below is real: the page's
Run button and the production executor.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from pyqtgraph.Qt import QtWidgets

from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import PoniV3OverrideIntent
from xrd_tools.sources import image_series_spec
from tests.xdart.scattering._output_slots import written
from tests.xdart.scattering.test_energy_scan_run import _series
from tests.xdart.scattering.test_p1b_output_graph import _intent, write_poni
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell

_V3_PONI = ('poni_version: 3.0\nDetector: Pilatus100k\n'
            'Detector_config: {"orientation":3,"sensor":{"material":"Si","thickness":0.00045}}\n'
            "Distance: 0.1234\nPoni1: 0.05\nPoni2: 0.06\nRot1: 0.0\nRot2: 0.0\nRot3: 0.0\n"
            "Wavelength: 1.0e-10\nParallax: False\n")


@pytest.fixture
def qapp() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _constant_energy_intent(root: Path, poni_text: str | None):
    root.mkdir(parents=True, exist_ok=True)
    raw = _series(root, (16, 16))
    poni = root / "cal.poni"
    if poni_text is None:
        write_poni(poni)
    else:
        poni.write_text(poni_text, encoding="utf-8")
    intent = _intent(raw, root / "run.nexus", poni)
    intent.source_spec = image_series_spec(raw, metadata_format="txt")
    return intent


def _page_run(qapp, monkeypatch, intent):
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
    terminal = {StandardEventKind.FINISHED, StandardEventKind.FAILED}
    try:
        page.findChild(ScatteringWorkspaceShell).run_controls.startButton.click()
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not any(event.kind in terminal for event in observed):
            qapp.processEvents()
            time.sleep(0.01)
        return page._notice_text, [event for event in observed if event.kind in terminal]
    finally:
        page.close()
        page.deleteLater()
        qapp.processEvents()


def test_the_start_freeze_signs_the_calibration_identity_admission_signed(tmp_path):
    from xdart.gui.tabs.scattering.contracts import SourceCapture, StartCapture
    from xdart.gui.tabs.scattering.events import RequestId
    from xdart.gui.tabs.scattering.output_preflight import OutputCandidate, _load_scientific_assets
    for name, text in (("v3", _V3_PONI), ("v2", None)):
        intent = _constant_energy_intent(tmp_path / name, text)
        assets = _load_scientific_assets(intent)
        store = RunIntentStore(intent)
        snapshot = store.snapshot()
        start = StartCapture(RequestId(20), 1, snapshot, SourceCapture(RequestId(20), 1, intent.source_spec))
        candidate = OutputCandidate.from_start_capture(start, assets)
        frozen = store.freeze(expected_revision=snapshot.revision, poni_values=assets.signed_poni_values)
        assert candidate.matches(frozen.configuration), name
        assert (assets.signed_poni_values is None) is (name == "v2")


def test_a_run_with_a_poni_v3_calibration_starts_and_writes_its_result(qapp, monkeypatch, tmp_path):
    intent = _constant_energy_intent(tmp_path / "v3", _V3_PONI)
    notice, terminal = _page_run(qapp, monkeypatch, intent)
    assert [event.kind for event in terminal] == [StandardEventKind.FINISHED], (notice, terminal)
    assert written(tmp_path / "v3" / "run.nexus", "Int 1D").is_file()


def test_a_run_with_the_parallax_override_starts_and_writes_its_result(qapp, monkeypatch, tmp_path):
    intent = _constant_energy_intent(tmp_path / "override", None)
    intent.poni_v3_override = PoniV3OverrideIntent("Si", 450e-6, True)
    notice, terminal = _page_run(qapp, monkeypatch, intent)
    assert [event.kind for event in terminal] == [StandardEventKind.FINISHED], (notice, terminal)
    assert written(tmp_path / "override" / "run.nexus", "Int 1D").is_file()
