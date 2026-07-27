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
        # O-1b §54.4 S1: D3 DELETED the worker's `gi` policy mirror -- the
        # worker consumes the exact FrozenRunConfiguration now.  The diagnostic
        # must therefore report that carrier as ABSENT.  Asserting a stale
        # `False` here would only pass again if someone re-added the mirror,
        # which is the regression this row now guards against.
        assert payload["wrangler"]["worker"]["present"] is True
        assert payload["wrangler"]["worker"]["gi"] is None, (
            "the retired worker GI carrier is back; the worker takes its policy "
            "as the frozen configuration argument, not as a mirror")
        assert not hasattr(widget.wrangler.thread, "gi"), (
            "imageThread must not carry a `gi` attribute at all")
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


def test_r4d_first_run_preserves_grazing_across_stale_hidden_carriers(
    qapp, monkeypatch, tmp_path
):
    """The first frozen run wins over every stale compatibility carrier."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    monkeypatch.setenv("XDART_SESSION_FRESH", "1")
    monkeypatch.setenv("XDART_SESSION_FILE", str(tmp_path / "session.json"))
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    from xrd_tools.session.run_configuration import FrozenRunConfiguration

    widget = staticWidget()
    try:
        _prime_stale_hidden_carriers(widget)

        # Production Run commit path (no monkeypatched setup/thread owners).
        run_configuration = widget._prepare_controls_v2_run_configuration()
        assert isinstance(run_configuration, FrozenRunConfiguration)
        assert run_configuration.gi.enabled is True
        assert run_configuration.threshold.threshold_max == 5000
        widget._apply_controls_v2_run_state(run_configuration)

        # The one run snapshot must win over every stale compatibility carrier
        # that still EXISTS.  The worker's `gi` mirror is not one of them: D3
        # deleted it, so first-run truth for the worker is asserted below
        # through the frozen object it actually executes, and reading a mirror
        # here would only re-legitimise the carrier the deletion removed.
        assert (
            widget.scan.gi,
            bool(widget.scan.gi_config),
            widget.wrangler.parameters.child("GI").child("Grazing").value(),
            widget.wrangler.gi,
        ) == (True, True, True, True)
        assert not hasattr(widget.wrangler.thread, "gi"), (
            "the retired worker GI carrier is back")
        assert (
            widget.wrangler.parameters.child("Mask").child("max").value()
            == 5000
        )
        assert widget.wrangler.run_configuration is run_configuration
        # §54.4 S1 item 2: the worker's first-run GI truth, stated through the
        # exact object it executes rather than a mirror of it.
        assert widget.wrangler.thread.run_configuration is run_configuration
        assert run_configuration.gi.enabled is True
    finally:
        widget.close()
        widget.deleteLater()
        qapp.processEvents()


def _adopt_direct_nxs_metadata(widget, source_dir):
    """Drive the real preview -> wrangler signal -> integrator/Controls chain."""
    from xrd_tools.sources import DirectorySourceSpec

    wrangler = widget.wrangler
    wrangler.source_spec = DirectorySourceSpec(
        root=source_dir,
        recursive=True,
        suffixes=(".nxs",),
    )
    signal = wrangler.parameters.child("Signal")
    previous = wrangler.parameters.blockSignals(True)
    try:
        signal.child("inp_type").setValue("Image Directory")
        signal.child("img_dir").setValue(str(source_dir))
        signal.child("img_ext").setValue("nxs")
        signal.child("include_subdir").setValue(True)
    finally:
        wrangler.parameters.blockSignals(previous)
    wrangler.inp_type = "Image Directory"
    wrangler.get_img_fname()


def test_r4b_directory_metadata_motor_is_offered_and_frozen_for_run(
    qapp, monkeypatch, tmp_path
):
    """Lazy directory discovery still supplies and freezes the GI motor."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    monkeypatch.setenv("XDART_SESSION_FRESH", "1")
    monkeypatch.setenv("XDART_SESSION_FILE", str(tmp_path / "session.json"))
    from tests.core.test_bluesky_nexus import (
        _write_bluesky_baseline_only_motors,
    )
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    source = tmp_path / "source"
    source.mkdir()
    _write_bluesky_baseline_only_motors(source / "direct_00001.nxs")

    widget = staticWidget()
    try:
        _adopt_direct_nxs_metadata(widget, source)
        qapp.processEvents()

        hidden = widget.wrangler.parameters.child("GI").child("th_motor")
        hidden_choices = tuple(hidden.opts["limits"])
        combo = widget.integratorTree.ui.gi_motor
        combo_choices = tuple(
            combo.itemText(index) for index in range(combo.count())
        )
        controls_choices = widget._controls_v2_native_int_choices()[
            ("GI", "th_motor")
        ]

        assert "halpha" in hidden_choices
        assert "halpha" in combo_choices
        assert "halpha" in controls_choices
        assert hidden.value() == "halpha"
        assert combo.currentText() == "halpha"

        _click_grazing(widget)
        run_configuration = widget._prepare_controls_v2_run_configuration()
        widget._apply_controls_v2_run_state(run_configuration)

        assert run_configuration.gi.enabled is True
        assert run_configuration.gi.incidence_motor == "halpha"
        assert widget.wrangler.run_configuration is run_configuration
        assert widget.wrangler.thread.run_configuration is run_configuration
    finally:
        widget.close()
        widget.deleteLater()
        qapp.processEvents()


def test_r4b_display_scan_mutation_cannot_change_next_frozen_gi_intent(
    qapp, monkeypatch, tmp_path
):
    """Browsing a Standard result cannot overwrite the next GI run intent."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    monkeypatch.setenv("XDART_SESSION_FRESH", "1")
    monkeypatch.setenv("XDART_SESSION_FILE", str(tmp_path / "session.json"))
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget.integratorTree.set_gi_motor_options(["halpha", "detx"])
        widget._on_gi_motor_options_changed(["halpha", "detx"])
        _click_grazing(widget)
        widget._on_controls_v2_field_changed(
            ("GI", "th_motor"), "halpha"
        )

        # The display LiveScan is reused when a different processed result is
        # browsed.  Model that projection directly: it is display state, not
        # authorization to rewrite the Controls-owned next-run intent.
        widget.scan.gi = False
        widget.scan.gi_config = {}
        widget.scan.incidence_motor = "display_only_motor"
        widget.scan.sample_orientation = 8
        widget.scan.tilt_angle = 12.5
        widget._refresh_controls_v2_profile_now()

        run_configuration = widget._prepare_controls_v2_run_configuration()
        widget._apply_controls_v2_run_state(run_configuration)

        assert run_configuration.gi.enabled is True
        assert run_configuration.gi.incidence_motor == "halpha"
        assert run_configuration.gi.sample_orientation == 4
        assert run_configuration.gi.tilt_angle == 0.0
        assert widget.scan.gi is True
        assert widget.scan.gi_config["incidence_motor"] == "halpha"
        assert widget.wrangler.run_configuration is run_configuration
        assert widget.wrangler.thread.run_configuration is run_configuration
    finally:
        widget.close()
        widget.deleteLater()
        qapp.processEvents()
