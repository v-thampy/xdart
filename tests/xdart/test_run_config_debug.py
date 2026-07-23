"""R4-D run-configuration ownership diagnostics and captured reproducer."""

import json
import logging
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("pyqtgraph")
from pyqtgraph import QtWidgets

from xdart.gui.tabs.static_scan.run_config_debug import run_config_debug_log
from xdart.gui.tabs.static_scan.ui.controls_panel_v2 import SegmentedControl


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _click_grazing(widget):
    widget._refresh_controls_v2_profile_now()
    segments = [
        row
        for row in widget.controls_v2.experiment_card.body.findChildren(
            SegmentedControl
        )
        if row.path == ("GI", "Grazing")
    ]
    assert len(segments) == 1
    button = next(
        button
        for value, button in segments[0]._segments
        if value is True
    )
    button.click()


def _prime_stale_hidden_carriers(widget):
    """Match the live first-Run ordering without replacing production QObjects."""
    parameters = widget.wrangler.parameters
    hidden_gi = parameters.child("GI").child("Grazing")
    hidden_threshold = parameters.child("Mask")

    # Make every threshold carrier except max already agree.  The run-boundary
    # max push is therefore the one value edit that invokes the production
    # imageWrangler._setup_on_value_change -> setup path.
    hidden_gi.setValue(False)
    hidden_threshold.child("Threshold").setValue(True)
    hidden_threshold.child("min").setValue(0)
    hidden_threshold.child("max").setValue(0)
    widget._on_controls_v2_field_changed(("Mask", "max"), "5000")
    _click_grazing(widget)

    cfg = widget._controls_v2_threshold_config()
    assert cfg.apply_threshold is True
    assert cfg.threshold_max == 5000
    assert hidden_threshold.child("max").value() == 0
    assert hidden_gi.value() is False
    assert widget.scan.gi is True
    assert widget.scan.gi_config


def test_run_config_debug_is_a_true_noop_when_disabled(monkeypatch):
    class ExplodingWidget:
        def __getattribute__(self, _name):
            raise AssertionError("disabled diagnostics inspected application state")

    class RecordingLogger:
        def __init__(self):
            self.messages = []

        def info(self, message):
            self.messages.append(message)

    monkeypatch.delenv("XDART_RUN_CONFIG_DEBUG", raising=False)
    logger = RecordingLogger()

    run_config_debug_log(
        logger,
        "disabled_probe",
        widget=ExplodingWidget(),
        origin="test",
    )

    assert logger.messages == []


def test_run_config_debug_records_all_run_ownership_carriers(
    qapp, monkeypatch, tmp_path, caplog
):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    monkeypatch.setenv("XDART_SESSION_FRESH", "1")
    monkeypatch.setenv("XDART_SESSION_FILE", str(tmp_path / "session.json"))
    monkeypatch.setenv("XDART_RUN_CONFIG_DEBUG", "1")
    caplog.set_level(
        logging.INFO,
        logger="xdart.gui.tabs.static_scan.static_scan_widget",
    )
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        _prime_stale_hidden_carriers(widget)
        run_config_debug_log(
            logging.getLogger(
                "xdart.gui.tabs.static_scan.static_scan_widget"
            ),
            "diagnostic_probe",
            widget=widget,
            origin="test",
        )

        message = next(
            record.getMessage()
            for record in reversed(caplog.records)
            if '"event":"diagnostic_probe"' in record.getMessage()
        )
        payload = json.loads(message.removeprefix("RUN_CONFIG_DEBUG "))

        assert payload["origin"] == "test"
        assert payload["controls"]["gi_visible_values"] == [True]
        assert payload["shared_scan"]["gi"] is True
        assert payload["wrangler"]["hidden_parameters"]["gi"] is False
        assert payload["wrangler"]["worker"]["gi"] is False
        assert payload["controls"]["threshold"]["threshold_max"] == 5000.0
        assert (
            payload["wrangler"]["hidden_parameters"]["threshold"]["max"] == 0
        )
        assert "diagnostic_config" in payload["generations"]
        assert "poni_file" in payload["controls"]
        assert "global_mask" in payload["shared_scan"]
        assert "mask_file" in payload["wrangler"]["worker"]
    finally:
        widget.close()
        widget.deleteLater()
        qapp.processEvents()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "R4-D captured live failure: first Run lets a stale hidden threshold "
        "carrier invoke setup and clobber Controls V2 Grazing"
    ),
)
def test_r4d_first_run_preserves_grazing_across_stale_hidden_carriers(
    qapp, monkeypatch, tmp_path
):
    """Desired ownership contract; captured red until the structural fix lands."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    monkeypatch.setenv("XDART_SESSION_FRESH", "1")
    monkeypatch.setenv("XDART_SESSION_FILE", str(tmp_path / "session.json"))
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        _prime_stale_hidden_carriers(widget)

        # Production Run commit path (no monkeypatched setup/thread owners).
        widget._apply_controls_v2_run_state()

        # The one run snapshot must win over every stale compatibility carrier.
        # At 41292079 this is (False, False, False, False, False): the hidden
        # threshold max write synchronously invokes full setup(), whose stale GI
        # false is emitted back into the shared scan before the GI push derives
        # its value.
        assert (
            widget.scan.gi,
            bool(widget.scan.gi_config),
            widget.wrangler.parameters.child("GI").child("Grazing").value(),
            widget.wrangler.gi,
            widget.wrangler.thread.gi,
        ) == (True, True, True, True, True)
    finally:
        widget.close()
        widget.deleteLater()
        qapp.processEvents()
