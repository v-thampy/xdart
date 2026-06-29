"""Offscreen tests for the hidden Controls Panel V2 scaffold."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("pyqtgraph")
from pyqtgraph import QtWidgets

from xdart.gui.tabs.static_scan.controls_logic import (
    AnalysisLauncherSpec,
    AnalysisTool,
    ControlAction,
    ControlState,
    ControlProfile,
    ProcessingPage,
    ResultCaps,
    SourceCaps,
    Tool,
    build_control_profile,
)
from xdart.gui.tabs.static_scan.ui.controls_panel_v2 import ControlsPanelV2, FieldRow


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_controls_panel_v2_renders_blockers_and_launchers(qapp):
    profile = ControlProfile(
        processing_page=ProcessingPage.RSM,
        run_enabled=False,
        run_blockers=("RSM GUI awaits real-data gate.",),
        analysis_launchers=(
            AnalysisLauncherSpec(
                AnalysisTool.PEAK_FIT, "Peak Fitting", enabled=True,
                live_capable=True),
            AnalysisLauncherSpec(
                AnalysisTool.SIN2PSI, "Strain / sin²ψ", enabled=False,
                reason="Needs ψ metadata.", production_ready=False),
        ),
    )

    panel = ControlsPanelV2()
    panel.set_profile(profile)

    badges = panel.summary_card.body.findChildren(QtWidgets.QLabel)
    assert [b.text() for b in badges] == ["RSM GUI awaits real-data gate."]

    buttons = panel.analysis_card.body.findChildren(QtWidgets.QPushButton)
    assert [b.text() for b in buttons] == ["Peak Fitting", "Strain / sin²ψ"]
    assert buttons[0].isEnabled()
    assert not buttons[1].isEnabled()
    assert buttons[1].toolTip() == "Needs ψ metadata."


def test_controls_panel_v2_emits_launcher_intent(qapp):
    profile = ControlProfile(
        processing_page=ProcessingPage.INT_1D,
        run_enabled=True,
        analysis_launchers=(
            AnalysisLauncherSpec(AnalysisTool.SCAN_PLOT, "Plot Metadata"),),
    )
    panel = ControlsPanelV2()
    panel.set_profile(profile)
    got = []
    panel.analysisLaunchRequested.connect(got.append)
    panel.analysis_card.body.findChildren(QtWidgets.QPushButton)[0].click()
    assert got == [AnalysisTool.SCAN_PLOT]


def test_controls_panel_v2_emits_action_intent(qapp):
    profile = build_control_profile(
        ControlState(
            tool=Tool.INT_2D,
            source_caps=SourceCaps(has_frames=True),
        )
    )
    panel = ControlsPanelV2()
    panel.set_profile(profile)
    got = []
    panel.controlActionRequested.connect(got.append)

    buttons = panel.source_card.body.findChildren(QtWidgets.QPushButton)
    buttons[0].click()

    assert got == [ControlAction.CHOOSE_SOURCE]


def test_controls_panel_v2_renders_typed_field_cards(qapp):
    profile = build_control_profile(
        ControlState(
            source_label="/tmp/scan.nxs",
            save_path="/tmp/out",
            frame_count=5,
            processing_mode="Int 1D",
            source_caps=SourceCaps(
                has_frames=True, has_raw=True, raw_reachable=True,
                has_metadata=True),
            result_caps=ResultCaps(has_1d=True),
        )
    )

    panel = ControlsPanelV2()
    panel.set_profile(profile)

    source_rows = panel.source_card.body.findChildren(FieldRow)
    assert [row.status.label for row in source_rows][:2] == ["Source", "Frames"]
    assert source_rows[0].status.value == "/tmp/scan.nxs"
    assert source_rows[1].status.value == "5"

    analysis_rows = panel.analysis_card.body.findChildren(FieldRow)
    assert [row.status.label for row in analysis_rows] == [
        "1D result", "2D result", "RSM result"]


def test_controls_panel_v2_mounts_behind_feature_flag(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        assert widget.controls_v2 is not None
        assert widget.controls_v2.profile is not None
        assert widget.controls_v2.source_card.body.findChildren(FieldRow)
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_static_widget_routes_actions(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    calls = []
    try:
        monkeypatch.setattr(
            widget,
            "_controls_v2_choose_source",
            lambda: calls.append(("source", None)),
        )
        monkeypatch.setattr(
            widget,
            "_controls_v2_click_integrator_button",
            lambda name: calls.append(("button", name)),
        )
        monkeypatch.setattr(
            widget,
            "_show_integration_advanced",
            lambda: calls.append(("advanced", None)),
        )

        widget._on_controls_v2_action(ControlAction.CHOOSE_SOURCE)
        widget._on_controls_v2_action(ControlAction.CALIBRATE)
        widget._on_controls_v2_action(ControlAction.MAKE_MASK)
        widget._on_controls_v2_action(ControlAction.ADVANCED_PROCESSING)

        assert calls == [
            ("source", None),
            ("button", "pyfai_calib"),
            ("button", "get_mask"),
            ("advanced", None),
        ]
    finally:
        widget.close()
        widget.deleteLater()
