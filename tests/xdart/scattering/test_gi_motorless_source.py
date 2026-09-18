"""Motorless containers must expose Manual before a GI Run is captured."""

import logging
import time

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from tests.xdart.scattering._e2sd_support import write_motor_container, write_poni
from tests.xdart.scattering.test_p1b_comprehensive_live import _write_eiger
from tests.xdart.scattering.test_p1b_output_graph import _intent, _written
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.contracts import SourceObservationRequest
from xdart.gui.tabs.scattering.controls_projection import GI_MOTOR, GI_THETA
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xrd_tools.core.provenance import read_provenance
from xrd_tools.io import FrameViewReader
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.sources.selection import image_series_spec


def _wait(app, predicate):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.005)
    pytest.fail("GI Run/observation did not settle")


def _page(tmp_path, *, broken_calibration=False):
    raw = tmp_path / "scan_master.h5"
    _write_eiger(raw, tmp_path / "scan_data_000001.h5", 2)
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    if broken_calibration:
        poni.write_text("not a calibration\n")
    target = tmp_path / "result.nexus"
    intent = _intent(raw, target, poni, processing_mode="Int 2D")
    intent.gi.enabled = True
    intent.gi.incidence_motor = "th"
    intent.gi.th_val = 0.37
    return ScatteringWorkspace(
        intents=RunIntentStore(intent), lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(), executor=StandardRunExecutor(join_timeout=2.0),
    ), _written(target, "Int 2D")


def _close(page, app):
    for box in page.findChildren(QtWidgets.QMessageBox):
        box.reject()
    _wait(app, lambda: page.close_workspace().cleanup_status is CleanupStatus.CLEANED)
    page.deleteLater()
    app.processEvents()


@pytest.mark.parametrize("motors", ((), ("eta",)))
def test_container_observation_exposes_known_motor_catalog(tmp_path, motors):
    raw = tmp_path / "scan_master.h5"
    if motors:
        write_motor_container(raw, motors[0])
    else:
        _write_eiger(raw, tmp_path / "scan_data_000001.h5", 2)
    observed = FilesystemSourceAdapter().observe(
        SourceObservationRequest(1, 0, image_series_spec(raw)))
    assert observed.gi_motor_choices == motors


def test_eiger_gi_run_falls_back_to_manual_and_keeps_angle(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    page, artifact = _page(tmp_path)
    try:
        _wait(app, lambda: page._source_selection.observation is not None)
        page._shell.run_controls.startButton.click()
        _wait(app, lambda: page._capture_current_loaded_browse() is not None
              or page._admission is None and page._notice_text.startswith("Output admission failed:"))
        assert page._capture_current_loaded_browse() is not None, page._notice_text
        current = page._intents.snapshot().thaw()
        assert current.gi.incidence_motor == "Manual"
        assert current.gi.th_val == 0.37
        fields = {field.path: field for field in page._shell.controls.projection.fields}
        assert fields[GI_MOTOR].value == "Manual"
        assert fields[GI_THETA].value == 0.37
        saved = read_provenance(artifact)["config"]["run_configuration"]
        assert saved["gi"]["incidence_motor"] == "Manual"
        assert saved["gi"]["th_val"] == 0.37
        with FrameViewReader(artifact) as reader:
            for label in (0, 1):
                view = reader.read(label)
                assert view.has_2d
                assert np.isfinite(view.intensity_1d).all()
    finally:
        _close(page, app)


def test_admission_error_is_visible_and_logged(tmp_path, caplog):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    page, artifact = _page(tmp_path, broken_calibration=True)
    try:
        _wait(app, lambda: page._source_selection.observation is not None)
        with caplog.at_level(logging.WARNING):
            page._shell.run_controls.startButton.click()
            _wait(app, lambda: page._admission is None and
                  page._notice_text.startswith("Output admission failed:"))
        boxes = [box for box in page.findChildren(QtWidgets.QMessageBox) if box.isVisible()]
        assert len(boxes) == 1, page._notice_text
        assert boxes[0].text() == "Run not started"
        assert "malformed noncomment PONI line" in boxes[0].informativeText()
        assert "Output admission failed:" in caplog.text
        assert not artifact.exists()
    finally:
        _close(page, app)
