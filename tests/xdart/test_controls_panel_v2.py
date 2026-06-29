"""Offscreen tests for the hidden Controls Panel V2 scaffold."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("pyqtgraph")
from pyqtgraph import QtWidgets

from xdart.gui.tabs.static_scan.controls_logic import (
    AnalysisLauncherSpec,
    AnalysisTool,
    BoundControlState,
    ControlAction,
    ControlFieldKind,
    ControlFormField,
    ControlPanelRenderState,
    ControlState,
    ControlProfile,
    ProcessingPage,
    ResultCaps,
    SectionId,
    SourceCaps,
    Tool,
    build_control_profile,
)
from xdart.gui.tabs.static_scan.ui.controls_panel_v2 import (
    ControlsPanelV2,
    FieldRow,
    FormRow,
    PillRow,
    RangeRow,
)


def _find_pill(widget, path):
    """Return the pill toggle button for ``path`` in the processing card, or None."""
    for pill_row in widget.controls_v2.processing_card.body.findChildren(PillRow):
        for p, btn in pill_row._pills:
            if p == path:
                return btn
    return None


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
            project_root="/tmp/project",
            source_caps=SourceCaps(has_frames=True),
        )
    )
    panel = ControlsPanelV2()
    panel.set_profile(profile)
    got = []
    panel.controlActionRequested.connect(got.append)

    buttons = panel.project_card.body.findChildren(QtWidgets.QPushButton)
    buttons[0].click()

    assert got == [ControlAction.CHOOSE_PROJECT]


def test_controls_panel_v2_renders_typed_field_cards(qapp):
    profile = build_control_profile(
        ControlState(
            source_label="/tmp/scan.nxs",
            project_root="/tmp/project",
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

    project_rows = panel.project_card.body.findChildren(FieldRow)
    assert [row.status.label for row in project_rows] == ["Project folder"]
    assert project_rows[0].status.value == "/tmp/project"

    source_rows = panel.source_card.body.findChildren(FieldRow)
    assert [row.status.label for row in source_rows][:2] == ["Source", "Frames"]
    assert source_rows[0].status.value == "/tmp/scan.nxs"
    assert source_rows[1].status.value == "5"

    analysis_rows = panel.analysis_card.body.findChildren(FieldRow)
    assert [row.status.label for row in analysis_rows] == [
        "1D result", "2D result", "RSM result"]


def test_controls_panel_v2_renders_bound_render_state_directly(qapp):
    profile = build_control_profile(
        ControlState(
            source_caps=SourceCaps(has_frames=True),
            result_caps=ResultCaps(has_1d=True),
        )
    )
    state = ControlPanelRenderState(
        profile=profile,
        bound_controls=BoundControlState(fields=(
            ControlFormField(
                section=SectionId.PROJECT,
                label="Folder",
                path=("Project", "project_folder"),
                value="/data",
                browse=True,
            ),
            ControlFormField(
                section=SectionId.SOURCE,
                label="Source",
                path=("Signal", "inp_type"),
                value="Image Series",
                kind=ControlFieldKind.COMBO,
                choices=("Image Series", "Image Directory"),
                enabled=False,
                reason="locked",
            ),
        )),
    )

    panel = ControlsPanelV2()
    panel.set_state(state)

    project_rows = panel.project_card.body.findChildren(FormRow)
    source_rows = panel.source_card.body.findChildren(FormRow)
    assert [row.label.text() for row in project_rows] == ["Folder"]
    assert [row.label.text() for row in source_rows] == ["Source"]
    assert not source_rows[0].editor.isEnabled()
    assert source_rows[0].toolTip() == "locked"
    assert panel.analysis_card.isHidden()


def test_make_mask_updates_mask_file_box(qapp, monkeypatch):
    """_on_mask_created writes the mask path to the wrangler param AND the V2
    Mask File box reflects it — the handler refreshes the panel, which re-reads
    the params and re-renders the row.  (Same refresh path the Calibrate→Poni
    autofill uses, so this covers both boxes.)"""
    monkeypatch.delenv("XDART_CONTROLS_PANEL_V2", raising=False)
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        mask_path = "/tmp/example-mask.edf"
        widget._on_mask_created(mask_path)
        # single source of truth: the wrangler param is updated
        assert widget.wrangler.parameters.child(
            "Signal", "mask_file").value() == mask_path
        # ...and the rendered Mask File box shows it
        rows = [
            r for r in widget.controls_v2.experiment_card.body.findChildren(FormRow)
            if getattr(r, "path", None) == ("Signal", "mask_file")
        ]
        assert rows, "Mask File row not rendered"
        assert rows[0].current_value() == mask_path
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_mounts_by_default(qapp, monkeypatch):
    monkeypatch.delenv("XDART_CONTROLS_PANEL_V2", raising=False)
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        assert widget.controls_v2 is not None
        assert widget.controls_v2.profile is not None
        assert widget.controls_v2.source_card.body.findChildren(FormRow)
        assert widget.ui.wranglerStack.isHidden()
        assert widget.controls_v2.analysis_card.isHidden()
        tool_labels = {
            btn.text() for btn in widget.ui.metaFrame.findChildren(
                QtWidgets.QPushButton)
        }
        # Buttons carry a leading glyph (e.g. "∧   Peak Fitting"); match the label.
        assert all(any(name in t for t in tool_labels)
                   for name in ("Peak Fitting", "Phase Fitting", "Plot Metadata"))
        assert widget.controls_v2.processing_card.isAncestorOf(
            widget.ui.integratorFrame)
        assert widget.ui.integratorFrame.isHidden()
        # Producers render inside the Experiment section, not a top bar.  Refine
        # is hidden in the Int 1D/2D modes (the default), so only Calibrate +
        # Make Mask show here.
        exp_labels = {
            btn.text()
            for btn in widget.controls_v2.experiment_card.body.findChildren(
                QtWidgets.QPushButton)
        }
        assert {"⌖ Calibrate", "▦ Make Mask"} <= exp_labels
        assert "◎ Refine" not in exp_labels
        assert not widget.controls_v2.top_action_bar.isVisible()
        action_labels = {
            btn.text() for btn in widget.controls_v2.processing_card.body.findChildren(
                QtWidgets.QPushButton)
        }
        assert {"Reintegrate 1D", "Reintegrate 2D", "Advanced"} <= action_labels
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_bound_mode_uses_inline_browse_and_top_actions(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget._refresh_controls_v2_profile_now()
        # Producers render inside the Experiment section, not a top bar.  Refine
        # is hidden in the Int 1D/2D modes (the default), so only Calibrate +
        # Make Mask show here.
        exp_labels = {
            btn.text()
            for btn in widget.controls_v2.experiment_card.body.findChildren(
                QtWidgets.QPushButton)
        }
        assert {"⌖ Calibrate", "▦ Make Mask"} <= exp_labels
        assert "◎ Refine" not in exp_labels
        assert not widget.controls_v2.top_action_bar.isVisible()

        project_labels = {
            btn.text()
            for btn in widget.controls_v2.project_card.body.findChildren(
                QtWidgets.QPushButton)
        }
        source_labels = {
            btn.text()
            for btn in widget.controls_v2.source_card.body.findChildren(
                QtWidgets.QPushButton)
        }
        assert "Choose Project" not in project_labels
        assert "Save Folder" not in project_labels
        assert "Choose Source" not in source_labels
        assert {
            btn.text()
            for btn in widget.controls_v2.project_card.body.findChildren(
                QtWidgets.QToolButton)
            if btn.objectName() == "controlsV2BrowseButton"
        } <= {"📁"}
        assert {
            btn.text()
            for btn in widget.controls_v2.source_card.body.findChildren(
                QtWidgets.QToolButton)
            if btn.objectName() == "controlsV2BrowseButton"
        } <= {"📁"}
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_can_be_hidden_by_env(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "0")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        assert widget.controls_v2 is None
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_field_edits_update_legacy_parameters(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        path = ("Project", "project_folder")
        widget._on_controls_v2_field_changed(path, "/tmp/controls-v2-project")
        assert widget.wrangler.parameters.child(*path).value() == \
            "/tmp/controls-v2-project"
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_integration_edits_write_through_immediately(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget._on_controls_v2_field_changed(("Int1D", "points"), "1234")
        assert widget.integratorTree.ui.npts_1D.text() == "1234"
        assert widget.scan.bai_1d_args["numpoints"] == 1234

        widget._on_controls_v2_field_changed(("Int2D", "radial_points"), "321")
        assert widget.integratorTree.ui.npts_radial_2D.text() == "321"
        assert widget.scan.bai_2d_args["npt_rad"] == 321

        widget._on_controls_v2_field_changed(("Int1D", "radial_auto"), False)
        assert not widget.integratorTree.ui.radial_autoRange_1D.isChecked()
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_gi_edits_update_integrator_and_carrier(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget._on_controls_v2_field_changed(("GI", "Grazing"), True)
        assert widget.integratorTree.ui.gi_enable.isChecked()
        assert widget.wrangler.parameters.child("GI", "Grazing").value() is True

        widget._on_controls_v2_field_changed(("GI", "sample_orientation"), "5")
        assert widget.integratorTree.ui.gi_sample_orientation.value() == 5
        assert widget.wrangler.parameters.child("GI", "sample_orientation").value() == 5
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_threshold_edits_update_integrator_and_carrier(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget._on_controls_v2_field_changed(("Mask", "Threshold"), True)
        widget._on_controls_v2_field_changed(("Mask", "min"), "10")
        widget._on_controls_v2_field_changed(("Mask", "max"), "900")
        widget._on_controls_v2_field_changed(("MaskSat", "mask_sentinel"), False)

        assert widget.integratorTree.ui.threshold_enable.isChecked()
        assert widget.integratorTree.ui.threshold_min.text() == "10"
        assert widget.integratorTree.ui.threshold_max.text() == "900"
        assert not widget.integratorTree.ui.mask_saturated.isChecked()
        assert widget.wrangler.parameters.child("Mask", "Threshold").value() is True
        assert widget.wrangler.parameters.child("Mask", "min").value() == 10
        assert widget.wrangler.parameters.child("Mask", "max").value() == 900
        assert widget.wrangler.parameters.child(
            "MaskSat", "mask_sentinel").value() is False
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_renders_integration_fields_from_live_widgets(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget._refresh_controls_v2_profile_now()
        rows = widget.controls_v2.processing_card.body.findChildren(FormRow)
        # Axis rows drop the redundant "1D"/"2D" prefix (the subsection title says it).
        visible_labels = [r.label.text() for r in rows if not r.label.isHidden()]
        assert "Axis" in visible_labels
        assert "1D Axis" not in visible_labels
        assert "2D Axis" not in visible_labels
        # Points now ride on the Axis row, right of the dropdown (hidden-label
        # FormRows), not their own body rows — still exist + route through.
        point_rows = [
            r for r in rows
            if r.path and r.path[-1] in ("points", "radial_points", "azim_points")
        ]
        assert point_rows
        assert all(r.label.isHidden() for r in point_rows)
        if widget.controls.current_mode() != "Int 1D":
            # Int 2D shows both groups → a 1-D and a 2-D Axis row.
            assert visible_labels.count("Axis") >= 2
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_mask_saturated_survives_run_state(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget._on_controls_v2_field_changed(("MaskSat", "mask_sentinel"), True)
        assert widget.integratorTree.ui.mask_saturated.isChecked()
        assert widget.wrangler.parameters.child(
            "MaskSat", "mask_sentinel").value() is True

        widget._enter_run_state()
        widget._refresh_controls_v2_profile_now()

        # Mask Saturated is now a compact pill toggle (in a PillRow), not a
        # full-width row.  It must survive the run-state lock checked + disabled.
        btn = _find_pill(widget, ("MaskSat", "mask_sentinel"))
        assert btn is not None
        assert btn.isCheckable()
        assert btn.isChecked()
        assert not btn.isEnabled()
        assert widget.integratorTree.ui.mask_saturated.isChecked()
        assert widget.wrangler.parameters.child(
            "MaskSat", "mask_sentinel").value() is True
    finally:
        widget._exit_run_state()
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_bool_rows_render_as_pill_toggles(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget._refresh_controls_v2_profile_now()
        # Bool toggles render as compact pills in a PillRow (mockup), not as
        # full-width rows.
        btn = _find_pill(widget, ("MaskSat", "mask_sentinel"))
        assert btn is not None
        assert isinstance(btn, QtWidgets.QPushButton)
        assert btn.isCheckable()
        assert btn.text() == "Mask Saturated"
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_run_commits_focused_integration_edit(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    class FakeThread:
        batch_mode = False
        xye_only = False

        def start(self):
            pass

    widget = staticWidget()
    try:
        widget._refresh_controls_v2_profile_now()
        rows = [
            row
            for row in widget.controls_v2.processing_card.body.findChildren(FormRow)
            if row.path == ("Int1D", "points")
        ]
        assert rows
        rows[0].editor.setText("777")
        assert widget.integratorTree.ui.npts_1D.text() != "777"

        widget.wrangler.thread = FakeThread()
        monkeypatch.setattr(widget.wrangler, "setup", lambda: None)

        widget.start_wrangler()

        assert widget.integratorTree.ui.npts_1D.text() == "777"
        assert widget.scan.bai_1d_args["numpoints"] == 777
        assert widget.wrangler.scan_args["bai_1d_args"]["numpoints"] == 777
    finally:
        widget._exit_run_state()
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_run_commits_focused_2d_points(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    class FakeThread:
        batch_mode = False
        xye_only = False

        def start(self):
            pass

    widget = staticWidget()
    try:
        idx = widget.controls.modeCombo.findText("Int 2D")
        assert idx >= 0
        widget.controls.modeCombo.setCurrentIndex(idx)
        widget._refresh_controls_v2_profile_now()
        rows = {
            row.path: row
            for row in widget.controls_v2.processing_card.body.findChildren(FormRow)
        }
        rows[("Int2D", "radial_points")].editor.setText("123")
        rows[("Int2D", "azim_points")].editor.setText("456")
        assert widget.integratorTree.ui.npts_radial_2D.text() != "123"

        widget.wrangler.thread = FakeThread()
        monkeypatch.setattr(widget.wrangler, "setup", lambda: None)

        widget.start_wrangler()

        assert widget.integratorTree.ui.npts_radial_2D.text() == "123"
        assert widget.integratorTree.ui.npts_azim_2D.text() == "456"
        assert widget.wrangler.scan_args["bai_2d_args"]["npt_rad"] == 123
        assert widget.wrangler.scan_args["bai_2d_args"]["npt_azim"] == 456
    finally:
        widget._exit_run_state()
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_range_labels_follow_legacy_integrator_labels(
        qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget.integratorTree.ui.gi_radial_label_1D.setText("LEGACY Q")
        widget.integratorTree.ui.label_azim_1D.setText("LEGACY CHI")
        widget._refresh_controls_v2_profile_now()

        # Ranges are coalesced into one compact RangeRow each; the row label is
        # the axis stem (the " Low"/" High" suffixes are dropped).
        labels = [
            row.label.text()
            for row in widget.controls_v2.processing_card.body.findChildren(RangeRow)
        ]
        assert "LEGACY Q" in labels
        assert "LEGACY CHI" in labels
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_auto_rows_disable_range_edits(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget._on_controls_v2_field_changed(("Int1D", "radial_auto"), True)
        widget._on_controls_v2_field_changed(("Mask", "Threshold"), False)
        widget._refresh_controls_v2_profile_now()

        # Range low/high now live inside a coalesced RangeRow, keyed by low path.
        ranges = {
            row._low_path: row
            for row in widget.controls_v2.processing_card.body.findChildren(RangeRow)
        }
        assert not ranges[("Int1D", "radial_low")]._low.isEnabled()
        assert not ranges[("Int1D", "radial_low")]._high.isEnabled()
        assert not ranges[("Mask", "min")]._low.isEnabled()

        widget._on_controls_v2_field_changed(("Int1D", "radial_auto"), False)
        widget._on_controls_v2_field_changed(("Mask", "Threshold"), True)
        widget._refresh_controls_v2_profile_now()

        ranges = {
            row._low_path: row
            for row in widget.controls_v2.processing_card.body.findChildren(RangeRow)
        }
        assert ranges[("Int1D", "radial_low")]._low.isEnabled()
        assert ranges[("Int1D", "radial_low")]._high.isEnabled()
        assert ranges[("Mask", "min")]._low.isEnabled()
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_mask_saturated_is_pushed_before_run_lock(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    class FakeThread:
        batch_mode = False
        xye_only = False

        def start(self):
            pass

    widget = staticWidget()
    seen_at_disable = []
    try:
        widget._on_controls_v2_field_changed(("MaskSat", "mask_sentinel"), True)
        original_enabled = widget.wrangler.enabled

        def enabled(enable):
            if enable is False:
                seen_at_disable.append(widget.wrangler.parameters.child(
                    "MaskSat", "mask_sentinel").value())
            return original_enabled(enable)

        widget.wrangler.thread = FakeThread()
        monkeypatch.setattr(widget.wrangler, "enabled", enabled)
        monkeypatch.setattr(widget.wrangler, "setup", lambda: None)

        widget.start_wrangler()

        assert seen_at_disable == [True]
        assert widget.integratorTree.ui.mask_saturated.isChecked()
    finally:
        widget._exit_run_state()
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_renders_nexus_wrangler_fields(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget.ui.wranglerStack.setCurrentIndex(1)
        widget._refresh_controls_v2_profile_now()
        labels = [
            row.label.text()
            for row in widget.controls_v2.source_card.body.findChildren(FormRow)
        ]
        assert "NeXus File" in labels
        assert "Entry" in labels
        assert widget.ui.wranglerStack.isHidden()
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
            "_controls_v2_choose_project",
            lambda: calls.append(("project", None)),
        )
        monkeypatch.setattr(
            widget,
            "_controls_v2_choose_output",
            lambda: calls.append(("output", None)),
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
        widget._on_controls_v2_action(ControlAction.CHOOSE_PROJECT)
        widget._on_controls_v2_action(ControlAction.CHOOSE_OUTPUT)
        widget._on_controls_v2_action(ControlAction.CALIBRATE)
        widget._on_controls_v2_action(ControlAction.MAKE_MASK)
        widget._on_controls_v2_action(ControlAction.REINTEGRATE_1D)
        widget._on_controls_v2_action(ControlAction.REINTEGRATE_2D)
        widget._on_controls_v2_action(ControlAction.ADVANCED_PROCESSING)

        assert calls == [
            ("source", None),
            ("project", None),
            ("output", None),
            ("button", "pyfai_calib"),
            ("button", "get_mask"),
            ("button", "reintegrate1D"),
            ("button", "reintegrate2D"),
            ("advanced", None),
        ]
    finally:
        widget.close()
        widget.deleteLater()
