"""Offscreen tests for the hidden Controls Panel V2 scaffold."""

import os
from pathlib import Path

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
    FieldId,
    INTEGRATOR_BACKED_CONTROL_SPECS,
    ProcessingPage,
    ResultCaps,
    SectionId,
    SourceCaps,
    StatusKind,
    Tool,
    build_control_panel_state,
    build_control_profile,
)
from xdart.gui.tabs.static_scan.ui.controls_panel_v2 import (
    ControlsPanelV2,
    FieldRow,
    FormRow,
    PillRow,
    RangeRow,
    SegmentedControl,
)


@pytest.fixture(autouse=True)
def _controls_panel_session_isolation():
    """Keep saved integrator state from leaking between tests in this module.

    ``staticWidget.close()`` persists the integrator session, including GI mode.
    Individual GI tests still reset explicitly before close, but this guard makes
    the file resilient to test reordering and future tests that forget cleanup.
    """

    path = os.environ.get("XDART_SESSION_FILE")

    def _unlink_session():
        if not path:
            return
        try:
            Path(path).unlink()
        except FileNotFoundError:
            pass

    _unlink_session()
    yield
    _unlink_session()


def _find_pill(widget, path):
    """Return the pill toggle button for ``path`` in the processing card, or None."""
    for pill_row in widget.controls_v2.processing_card.body.findChildren(PillRow):
        for p, btn in pill_row._pills:
            if p == path:
                return btn
    return None


def _find_segmented(widget, path):
    """Return the SegmentedControl for ``path`` in the experiment card, or None."""
    for seg in widget.controls_v2.experiment_card.body.findChildren(SegmentedControl):
        if seg.path == tuple(path):
            return seg
    return None


def _gi_detail_rows(widget):
    """Paths of the inline GI detail FormRows in Experiment (θ motor + the
    manual θ value; Orientation/Tilt now live behind the '…' popup)."""
    gi_detail = {("GI", "th_motor"), ("GI", "th_val")}
    return {
        row.path
        for row in widget.controls_v2.experiment_card.body.findChildren(FormRow)
        if row.path in gi_detail
    }


def _find_more_button(widget):
    """The '…' GI-options button in the Experiment card, or None."""
    for btn in widget.controls_v2.experiment_card.body.findChildren(
            QtWidgets.QToolButton):
        if btn.objectName() == "controlsV2MoreButton":
            return btn
    return None


def _plain(value):
    """Small, stable representation for reduction-plan equivalence tests."""
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return tuple(_plain(v) for v in value)
    return value


def _plan_snapshot(plan):
    def _snap(obj, attrs):
        if obj is None:
            return None
        return {name: _plain(getattr(obj, name)) for name in attrs}

    return {
        "integration_1d": _snap(plan.integration_1d, (
            "npt",
            "npt_rad",
            "unit",
            "method",
            "radial_range",
            "azimuth_range",
            "monitor_key",
            "error_model",
            "polarization_factor",
            "extra",
        )),
        "integration_2d": _snap(plan.integration_2d, (
            "npt_rad",
            "npt_azim",
            "unit",
            "method",
            "radial_range",
            "azimuth_range",
            "azimuth_offset",
            "monitor_key",
            "error_model",
            "polarization_factor",
            "extra",
        )),
        "gi": _snap(plan.gi, (
            "incident_angle",
            "incidence_motor",
            "tilt_angle",
            "sample_orientation",
            "method",
            "mode_1d",
            "mode_2d",
            "npt_oop",
        )),
        "mask_kind": type(plan.mask).__name__ if plan.mask is not None else None,
        "threshold_min": _plain(plan.threshold_min),
        "threshold_max": _plain(plan.threshold_max),
        "mask_saturation": _plain(plan.mask_saturation),
    }


def _threshold_snapshot(widget):
    cfg = widget.integratorTree.get_threshold_config()
    return {
        "apply_threshold": cfg.apply_threshold,
        "threshold_min": cfg.threshold_min,
        "threshold_max": cfg.threshold_max,
        "mask_saturation": cfg.mask_saturation,
    }


def _set_legacy_integrator_field(widget, path, value):
    """Set a hidden legacy integrator widget directly, then run the legacy parser.

    V2 uses the same widgets while it is still migrating.  This helper bypasses
    the V2 field-change signal so the tests compare V2 write-through against the
    old parser surface instead of simply testing V2 against itself.
    """
    from xdart.gui.tabs.static_scan.controls_logic import (
        coerce_control_edit_value,
    )

    spec = next(s for s in INTEGRATOR_BACKED_CONTROL_SPECS if s.path == path)
    ui = widget.integratorTree.ui
    editor = getattr(ui, spec.widget_name)
    if spec.value_role == "current_text":
        idx = editor.findText(str(value))
        assert idx >= 0, f"{value!r} not found for {path}"
        editor.setCurrentIndex(idx)
    elif spec.value_role == "checked":
        editor.setChecked(bool(coerce_control_edit_value(editor.isChecked(), value)))
    elif spec.value_role == "value":
        editor.setValue(coerce_control_edit_value(editor.value(), value))
    else:
        editor.setText("" if value is None else str(value))
    widget._sync_controls_v2_integrator_path(path)


def _apply_v2_edits(widget, edits):
    for path, value in edits:
        widget._on_controls_v2_field_changed(path, value)


def _apply_legacy_edits(widget, edits):
    for path, value in edits:
        _set_legacy_integrator_field(widget, path, value)


def _current_plan_snapshot(widget, *, include_threshold=True,
                           commit_pending=True):
    from xdart.modules.reduction import (
        apply_threshold_saturation_to_plan,
        plan_from_live_scan,
    )

    if commit_pending:
        widget._commit_controls_v2_pending_edits()
    widget._sync_controls_v2_integrator_args()
    widget.integratorTree._apply_gi_config_to_scan()
    plan = plan_from_live_scan(widget.scan, integrate_2d=True)
    if include_threshold:
        plan = apply_threshold_saturation_to_plan(
            plan,
            widget.integratorTree.get_threshold_config(),
        )
    return _plan_snapshot(plan)


def _combo_text(widget, name, predicate, *, fallback_current=True):
    combo = getattr(widget.integratorTree.ui, name)
    for i in range(combo.count()):
        text = combo.itemText(i)
        if predicate(text):
            return text
    if fallback_current:
        return combo.currentText()
    raise AssertionError(f"No matching choice in {name}")


def _visible_control_value(widget, path):
    path = tuple(path)
    cards = (
        widget.controls_v2.project_card,
        widget.controls_v2.source_card,
        widget.controls_v2.experiment_card,
        widget.controls_v2.processing_card,
    )
    for card in cards:
        for seg in card.body.findChildren(SegmentedControl):
            if seg.path == path:
                return seg.current_value()
        for row in card.body.findChildren(FormRow):
            if row.path == path:
                return row.current_value()
        for row in card.body.findChildren(RangeRow):
            for row_path, value in row.current_edits():
                if row_path == path:
                    return value
        for row in card.body.findChildren(PillRow):
            for row_path, value in row.current_edits():
                if row_path == path:
                    return value
    raise AssertionError(f"No visible V2 control for {path!r}")


def _reset_controls_v2_gi(*widgets):
    """Leave GI tests in Standard mode even if an assertion fails midway."""
    for widget in widgets:
        try:
            widget._on_controls_v2_field_changed(("GI", "Grazing"), False)
        except Exception:
            pass


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
        _reset_controls_v2_gi(widget)
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_grazing_renders_as_segmented_control(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget._on_controls_v2_field_changed(("GI", "Grazing"), False)
        widget._refresh_controls_v2_profile_now()

        seg = _find_segmented(widget, ("GI", "Grazing"))
        assert seg is not None
        # Two mutually-exclusive segments: Standard | Grazing.
        labels = [btn.text() for _v, btn in seg._segments]
        assert labels == ["Standard", "Grazing"]
        assert seg._group.exclusive()
        assert seg.current_value() is False  # Standard active

        # Clicking Grazing flips the integrator + wrangler carrier (same
        # (path, bool) signal the old toggle emitted).
        grazing_btn = next(btn for v, btn in seg._segments if v is True)
        grazing_btn.click()
        assert widget.integratorTree.ui.gi_enable.isChecked()
        assert widget.wrangler.parameters.child("GI", "Grazing").value() is True
    finally:
        _reset_controls_v2_gi(widget)
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_gi_detail_fields_inline_only_in_grazing(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        # Standard mode -> only the segmented control, no GI detail rows / popup.
        widget._on_controls_v2_field_changed(("GI", "Grazing"), False)
        widget._refresh_controls_v2_profile_now()
        assert _find_segmented(widget, ("GI", "Grazing")) is not None
        assert _gi_detail_rows(widget) == set()
        assert _find_more_button(widget) is None

        # Grazing mode -> θ motor inline + the '…' GI-options button (progressive
        # disclosure, gated in controls_logic on the Grazing state).
        widget._on_controls_v2_field_changed(("GI", "Grazing"), True)
        widget._refresh_controls_v2_profile_now()
        assert _gi_detail_rows(widget) == {("GI", "th_motor")}
        assert _find_more_button(widget) is not None
    finally:
        _reset_controls_v2_gi(widget)
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_grazing_roundtrips_scan_gi_and_config(qapp, monkeypatch):
    """P2: the V2 Grazing path must flip scan.gi AND land sample facts in
    get_gi_config() (the reintegrate source), independent of the legacy toggle —
    the GI-inline rework re-routes exactly this signal."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget._on_controls_v2_field_changed(("GI", "Grazing"), True)
        assert widget.scan.gi is True

        widget._on_controls_v2_field_changed(("GI", "sample_orientation"), "5")
        assert widget.integratorTree.get_gi_config()["sample_orientation"] == 5

        widget._on_controls_v2_field_changed(("GI", "Grazing"), False)
        assert widget.scan.gi is False
    finally:
        _reset_controls_v2_gi(widget)
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_refresh_does_not_refire_gi_signal(qapp, monkeypatch):
    """P2/#56: a profile refresh or programmatic GI re-sync must NOT re-emit
    sigUpdateGI when the user didn't toggle.  Removing the GI popup makes the old
    re-open-on-refresh bug impossible; this pins that no spurious GI signal fires
    across refreshes, and exactly one fires per real value-changing toggle."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget._on_controls_v2_field_changed(("GI", "Grazing"), True)
        emissions = []
        widget.integratorTree.sigUpdateGI.connect(lambda v: emissions.append(v))

        # Forced rebuilds + a programmatic GI re-sync with GI already on -> no
        # spurious GI emissions.
        for _ in range(5):
            widget._refresh_controls_v2_profile_now()
        widget._sync_controls_v2_integrator_path(("GI", "Grazing"))
        assert emissions == []

        # A genuine value-changing toggle -> exactly one emission; re-applying the
        # same value is idempotent (equality-guarded -> no emission).
        widget._on_controls_v2_field_changed(("GI", "Grazing"), False)
        assert emissions == [False]
        widget._on_controls_v2_field_changed(("GI", "Grazing"), False)
        assert emissions == [False]
    finally:
        _reset_controls_v2_gi(widget)
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_refresh_defers_while_line_editor_focused(qapp, monkeypatch):
    """P2 (dropped input): a background rebuild (set_state -> clear_rows) must NOT
    destroy a line editor the user is mid-edit in and drop the uncommitted text.
    When a QLineEdit is focused, the refresh defers by arming a one-shot on the
    editor's editingFinished (NO throttle re-arm / spin); the rebuild is scheduled
    once the editor commits."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget._refresh_controls_v2_profile_now()
        rows = [
            row
            for row in widget.controls_v2.processing_card.body.findChildren(FormRow)
            if row.path == ("Int1D", "points")
        ]
        assert rows
        editor = rows[0].editor
        editor.setText("999")  # typed, NOT committed (no editingFinished)

        # Simulate the editor holding focus and a signature-changing background
        # event (e.g. a load completing).
        monkeypatch.setattr(widget.controls_v2, "focusWidget", lambda: editor)
        widget._controls_v2_last_signature = None
        triggered = []
        monkeypatch.setattr(
            widget._controls_v2_refresh_timer, "trigger",
            lambda: triggered.append(True),
        )

        widget._refresh_controls_v2_profile_now()

        # Rebuild deferred WITHOUT re-arming the throttle (no spin): the editor is
        # the SAME object, its uncommitted text survived, and a one-shot is armed
        # on the editor instead of waking a timer.
        assert triggered == []                              # no throttle spin
        assert widget._controls_v2_pending_editor is editor
        same = [
            row
            for row in widget.controls_v2.processing_card.body.findChildren(FormRow)
            if row.path == ("Int1D", "points")
        ]
        assert same and same[0].editor is editor
        assert editor.text() == "999"

        # When the editor commits, the one-shot SCHEDULES the deferred rebuild
        # through the throttle (immediate=False) and clears itself — it does not
        # run synchronously (would delete the editor mid-emission) nor re-arm a
        # spinning timer.  (Invoke the handler directly: emitting editingFinished
        # on a real FormRow editor would also fire the row's own field-change.)
        monkeypatch.setattr(widget.controls_v2, "focusWidget", lambda: None)
        widget._on_controls_v2_pending_editor_done()
        assert widget._controls_v2_pending_editor is None
        assert triggered == [True]                           # scheduled once, on commit
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_batch_refreshes_once_after_run(qapp, monkeypatch):
    """Batch runs should not rebuild the V2 controls panel every progress tick.

    The profile is refreshed once the run exits, so the final state still
    appears without adding GUI churn during long batch reductions.
    """
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    calls = []
    try:
        monkeypatch.setattr(
            widget.controls_v2, "set_state",
            lambda state: calls.append(state),
        )
        widget.controls.batchButton.setChecked(True)
        widget._run_active = True

        widget._refresh_controls_v2_profile_now()

        assert calls == []
        assert widget._controls_v2_batch_refresh_deferred is True

        widget._exit_run_state()

        assert len(calls) == 1
        assert widget._controls_v2_batch_refresh_deferred is False
        assert calls[0].profile is not None
        assert calls[0].profile.fields
    finally:
        widget._run_active = False
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_energy_conflict_reaches_widget_profile(
        qapp, monkeypatch):
    """The real widget state now carries both calibration and source energy.

    A mismatch is rendered as a conflict and becomes a run blocker before the
    legacy gate is retired.
    """
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    from xrd_tools.core.energy import wavelength_m_to_energy_eV

    widget = staticWidget()
    try:
        widget.scan._persisted_wavelength_m = 1.0e-10
        widget.scan.source_energy_eV = 15_000.0

        state = widget._controls_v2_state()
        render_state = build_control_panel_state(
            state,
            widget._controls_v2_field_values(),
            widget._controls_v2_field_choices(),
        )
        energy = render_state.profile.fields[FieldId.BEAM_ENERGY]

        assert energy.status is StatusKind.CONFLICT
        assert "disagree" in energy.reason.lower()
        assert render_state.profile.can_run is False
        assert any("energy" in blocker.lower()
                   for blocker in render_state.profile.run_blockers)
        assert state.geom.calibration_energy_eV == pytest.approx(
            wavelength_m_to_energy_eV(1.0e-10), rel=1e-12)
        assert state.geom.source_energy_eV == pytest.approx(15_000.0)
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_integrator_bridge_uses_binding_table(
        qapp, monkeypatch):
    """All V2 integration fields are harvested and written through the single
    binding table, so adding a future control is a one-row change."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    synced = []
    try:
        widget._refresh_controls_v2_profile_now()
        values = widget._controls_v2_integrator_values()
        ui = widget.integratorTree.ui

        expected_paths = {
            spec.path
            for spec in INTEGRATOR_BACKED_CONTROL_SPECS
            if getattr(ui, spec.widget_name, None) is not None
            if not (
                spec.visible_when == "widget_visible"
                and not widget._controls_v2_integrator_widget_exposed(
                    getattr(ui, spec.widget_name)
                )
            )
        }
        assert expected_paths <= set(values)

        monkeypatch.setattr(
            widget,
            "_sync_controls_v2_integrator_path",
            lambda path: synced.append(tuple(path)),
        )
        for spec in INTEGRATOR_BACKED_CONTROL_SPECS:
            source_widget = getattr(ui, spec.widget_name, None)
            if source_widget is None:
                continue
            if (
                spec.visible_when == "widget_visible"
                and not widget._controls_v2_integrator_widget_exposed(source_widget)
            ):
                continue
            assert widget._set_controls_v2_integrator_field(
                spec.path,
                values[spec.path],
            )

        assert set(synced) == expected_paths
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_hidden_legacy_parent_keeps_gi_oop_points(
        qapp, monkeypatch):
    """The V2 bridge uses the hidden v1 integrator as a parser.

    ``npts_oop_1D`` is intentionally shown by the legacy GI mode logic, but its
    parent panel is hidden while V2 owns the surface.  The bridge must key off
    the child widget's explicit show/hide state so GI fiber OOP points remain
    visible to V2 and reach the plan for the very next run.
    """
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        _apply_v2_edits(widget, ((("GI", "Grazing"), True),))
        qip = _combo_text(
            widget,
            "axis1D",
            lambda text: "ip" in text.lower(),
            fallback_current=False,
        )
        _apply_v2_edits(
            widget,
            (
                (("Int1D", "axis"), qip),
                (("Int1D", "points"), "234"),
                (("Int1D", "points_oop"), "345"),
            ),
        )
        widget._refresh_controls_v2_profile_now()

        ui = widget.integratorTree.ui
        assert widget.ui.integratorFrame.isHidden()
        assert not ui.npts_oop_1D.isHidden()
        assert not ui.npts_oop_1D.isVisible()

        values = widget._controls_v2_integrator_values()
        assert values[("Int1D", "points_oop")] == "345"
        assert _visible_control_value(widget, ("Int1D", "points_oop")) == "345"

        widget._apply_controls_v2_run_state()
        snapshot = _current_plan_snapshot(widget, commit_pending=False)
        assert snapshot["integration_1d"]["npt"] == 234
        assert snapshot["gi"]["mode_1d"] == "q_ip"
        assert snapshot["gi"]["npt_oop"] == 345
    finally:
        _reset_controls_v2_gi(widget)
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_standard_edits_match_legacy_reduction_plan(
        qapp, monkeypatch):
    """V2 edits must produce the same standard ReductionPlan as the v1 parser.

    This is the migration gate for making the V2 controls authoritative: the
    hidden integrator may disappear later, but first the resulting plan surface
    must be indistinguishable for the settings users edit before pressing Run.
    """
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    v2_widget = staticWidget()
    legacy_widget = staticWidget()
    try:
        axis_1d = _combo_text(
            v2_widget,
            "axis1D",
            lambda text: "2" in text and "θ" in text,
        )
        axis_2d = _combo_text(
            v2_widget,
            "axis2D",
            lambda text: "2" in text and "θ" in text,
        )
        edits = (
            (("GI", "Grazing"), False),
            (("Int1D", "axis"), axis_1d),
            (("Int1D", "points"), "321"),
            (("Int1D", "radial_auto"), False),
            (("Int1D", "radial_low"), "0.25"),
            (("Int1D", "radial_high"), "4.5"),
            (("Int1D", "azim_auto"), False),
            (("Int1D", "azim_low"), "-90"),
            (("Int1D", "azim_high"), "90"),
            (("Int2D", "axis"), axis_2d),
            (("Int2D", "radial_points"), "111"),
            (("Int2D", "azim_points"), "77"),
            (("Int2D", "radial_auto"), False),
            (("Int2D", "radial_low"), "0.5"),
            (("Int2D", "radial_high"), "5.0"),
            (("Int2D", "azim_auto"), False),
            (("Int2D", "azim_low"), "-120"),
            (("Int2D", "azim_high"), "120"),
        )

        _apply_v2_edits(v2_widget, edits)
        _apply_legacy_edits(legacy_widget, edits)

        assert _current_plan_snapshot(v2_widget) == _current_plan_snapshot(
            legacy_widget,
            commit_pending=False,
        )
    finally:
        v2_widget.close()
        legacy_widget.close()
        v2_widget.deleteLater()
        legacy_widget.deleteLater()


def test_controls_panel_v2_gi_edits_match_legacy_reduction_plan(
        qapp, monkeypatch):
    """GI is especially fragile: keep V2 labels/ranges/modes on legacy parity."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    v2_widget = staticWidget()
    legacy_widget = staticWidget()
    try:
        # First switch both widgets to GI so the old integrator repopulates its
        # axis labels and modes.  The subsequent labels come from those old combo
        # boxes, not duplicated test knowledge.
        _apply_v2_edits(v2_widget, ((("GI", "Grazing"), True),))
        _apply_legacy_edits(legacy_widget, ((("GI", "Grazing"), True),))
        q_total_or_current = _combo_text(
            v2_widget,
            "axis1D",
            lambda text: "total" in text.lower(),
        )
        qip_qoop_or_current = _combo_text(
            v2_widget,
            "axis2D",
            lambda text: "ip" in text.lower() and "oop" in text.lower(),
        )
        edits = (
            (("GI", "Grazing"), True),
            (("GI", "th_motor"), "Manual"),
            (("GI", "th_val"), "0.17"),
            (("GI", "sample_orientation"), "4"),
            (("GI", "tilt_angle"), "0.25"),
            (("Int1D", "axis"), q_total_or_current),
            (("Int1D", "points"), "222"),
            (("Int1D", "points_oop"), "33"),
            (("Int1D", "radial_auto"), False),
            (("Int1D", "radial_low"), "0.1"),
            (("Int1D", "radial_high"), "5.4"),
            (("Int1D", "azim_auto"), False),
            (("Int1D", "azim_low"), "-45"),
            (("Int1D", "azim_high"), "35"),
            (("Int2D", "axis"), qip_qoop_or_current),
            (("Int2D", "radial_points"), "64"),
            (("Int2D", "azim_points"), "48"),
            (("Int2D", "radial_auto"), False),
            (("Int2D", "radial_low"), "-3.0"),
            (("Int2D", "radial_high"), "3.0"),
            (("Int2D", "azim_auto"), False),
            (("Int2D", "azim_low"), "0.0"),
            (("Int2D", "azim_high"), "4.0"),
        )

        _apply_v2_edits(v2_widget, edits)
        _apply_legacy_edits(legacy_widget, edits)

        assert _current_plan_snapshot(v2_widget) == _current_plan_snapshot(
            legacy_widget,
            commit_pending=False,
        )
    finally:
        _reset_controls_v2_gi(v2_widget, legacy_widget)
        v2_widget.close()
        legacy_widget.close()
        v2_widget.deleteLater()
        legacy_widget.deleteLater()


def test_controls_panel_v2_threshold_edits_match_legacy_plan_overlay(
        qapp, monkeypatch):
    """Threshold + saturation are a separate plan overlay; keep it in parity."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    v2_widget = staticWidget()
    legacy_widget = staticWidget()
    try:
        edits = (
            (("Mask", "Threshold"), True),
            (("Mask", "min"), "12.5"),
            (("Mask", "max"), "987.5"),
            (("MaskSat", "mask_sentinel"), True),
        )

        _apply_v2_edits(v2_widget, edits)
        _apply_legacy_edits(legacy_widget, edits)

        assert _threshold_snapshot(v2_widget) == _threshold_snapshot(
            legacy_widget)
        assert _current_plan_snapshot(v2_widget) == _current_plan_snapshot(
            legacy_widget,
            commit_pending=False,
        )
    finally:
        v2_widget.close()
        legacy_widget.close()
        v2_widget.deleteLater()
        legacy_widget.deleteLater()


def test_controls_panel_v2_integrator_session_roundtrip_hydrates_visible_rows(
        qapp, monkeypatch):
    """V2-edited integration state must survive the close/open session path.

    The visible V2 panel currently writes through to the hidden legacy
    integrator.  Until the legacy widgets are retired, the close-time
    ``integrator.session_state()`` blob is the migration bridge, so it must
    preserve both the hidden parser widgets and the rows users see.
    """
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    monkeypatch.setenv("XDART_SESSION_FRESH", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    edited = staticWidget()
    restored = staticWidget()
    try:
        _apply_v2_edits(
            edited,
            (
                (("GI", "Grazing"), True),
                (("GI", "th_motor"), "Manual"),
                (("GI", "th_val"), "0.23"),
                (("GI", "sample_orientation"), "5"),
                (("GI", "tilt_angle"), "0.75"),
                (("Int1D", "points"), "432"),
                (("Int1D", "radial_auto"), False),
                (("Int1D", "radial_low"), "0.2"),
                (("Int1D", "radial_high"), "4.2"),
                (("Int2D", "radial_points"), "96"),
                (("Int2D", "azim_points"), "84"),
                (("Mask", "Threshold"), True),
                (("Mask", "min"), "7"),
                (("Mask", "max"), "777"),
                (("MaskSat", "mask_sentinel"), True),
            ),
        )

        blob = edited.integratorTree.session_state()
        restored.integratorTree.restore_session_state(blob)
        restored._refresh_controls_v2_profile_now()

        ui = restored.integratorTree.ui
        assert ui.gi_enable.isChecked()
        assert ui.gi_sample_orientation.value() == 5
        assert ui.gi_tilt.text() == "0.75"
        assert ui.gi_motor.currentText() == "Manual"
        assert ui.gi_motor_value.text() == "0.23"
        assert ui.npts_1D.text() == "432"
        assert not ui.radial_autoRange_1D.isChecked()
        assert ui.radial_low_1D.text() == "0.2"
        assert ui.radial_high_1D.text() == "4.2"
        assert ui.npts_radial_2D.text() == "96"
        assert ui.npts_azim_2D.text() == "84"
        assert ui.threshold_enable.isChecked()
        assert ui.threshold_min.text() == "7"
        assert ui.threshold_max.text() == "777"
        assert ui.mask_saturated.isChecked()

        assert _visible_control_value(restored, ("GI", "Grazing")) is True
        assert _visible_control_value(restored, ("GI", "th_motor")) == "Manual"
        assert _visible_control_value(restored, ("GI", "th_val")) == "0.23"
        assert _visible_control_value(restored, ("Int1D", "points")) == "432"
        assert _visible_control_value(restored, ("Int1D", "radial_low")) == "0.2"
        assert _visible_control_value(restored, ("Int1D", "radial_high")) == "4.2"
        assert _visible_control_value(restored, ("Int2D", "radial_points")) == "96"
        assert _visible_control_value(restored, ("Int2D", "azim_points")) == "84"
        assert _visible_control_value(restored, ("Mask", "Threshold")) is True
        assert _visible_control_value(restored, ("Mask", "min")) == "7"
        assert _visible_control_value(restored, ("Mask", "max")) == "777"
        assert _visible_control_value(restored, ("MaskSat", "mask_sentinel")) is True
    finally:
        _reset_controls_v2_gi(edited, restored)
        edited.close()
        restored.close()
        edited.deleteLater()
        restored.deleteLater()


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


def test_controls_panel_v2_average_scan_renders_as_conditioning_pill(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        # Image Series source -> Average Scan is offered (a frame-averaging
        # *processing* choice, re-homed from SOURCE to PROCESSING).
        widget._on_controls_v2_field_changed(("Signal", "inp_type"), "Image Series")
        widget._refresh_controls_v2_profile_now()

        avg = _find_pill(widget, ("Signal", "series_average"))
        assert avg is not None
        assert isinstance(avg, QtWidgets.QPushButton)
        assert avg.isCheckable()
        assert avg.text() == "Average Scan"

        # It coalesces into the same Conditioning PillRow as Mask Saturated.
        sat = _find_pill(widget, ("MaskSat", "mask_sentinel"))
        assert sat is not None
        assert avg.parent() is sat.parent()

        # The pill writes through to the wrangler param (no integrator widget
        # backs series_average), flipping the legacy carrier immediately.
        widget._on_controls_v2_field_changed(("Signal", "series_average"), True)
        assert (
            widget.wrangler.parameters.child("Signal", "series_average").value()
            is True
        )
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_source_layout_coalesces_rows(qapp, monkeypatch):
    """SOURCE layout: in Image Directory mode the mode combo + Subdirs toggle
    share a row, and File Type + Meta Type share a row.  In Image Series mode
    Subdirs is absent and Meta Type stands alone."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    def _src_row(widget, path):
        for row in widget.controls_v2.source_card.body.findChildren(FormRow):
            if row.path == path:
                return row
        return None

    widget = staticWidget()
    try:
        widget._on_controls_v2_field_changed(("Signal", "inp_type"), "Image Directory")
        widget._refresh_controls_v2_profile_now()

        src = _src_row(widget, ("Signal", "inp_type"))
        subdirs = _src_row(widget, ("Signal", "include_subdir"))
        ftype = _src_row(widget, ("Signal", "img_ext"))
        mtype = _src_row(widget, ("Signal", "meta_ext"))
        assert src is not None and subdirs is not None
        assert ftype is not None and mtype is not None
        # Mode combo + Subdirs share a row; File Type + Meta Type share a row;
        # the two rows are distinct.
        assert src.parent() is subdirs.parent()
        assert ftype.parent() is mtype.parent()
        assert src.parent() is not ftype.parent()

        # Image Series: no Subdirs (directory-only), Meta Type still renders.
        widget._on_controls_v2_field_changed(("Signal", "inp_type"), "Image Series")
        widget._refresh_controls_v2_profile_now()
        assert _src_row(widget, ("Signal", "include_subdir")) is None
        assert _src_row(widget, ("Signal", "meta_ext")) is not None
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_gi_options_popup_holds_orient_and_tilt(qapp, monkeypatch):
    """In Grazing mode: θ motor renders inline (compact label, descriptive
    tooltip); Orientation + Tilt Angle live behind a '…' button that opens a
    small popup, whose rows still write through."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget._on_controls_v2_field_changed(("GI", "Grazing"), True)
        widget._refresh_controls_v2_profile_now()

        exp = widget.controls_v2.experiment_card

        def _row(card, path):
            for r in card.body.findChildren(FormRow):
                if r.path == path:
                    return r
            return None

        motor = _row(exp, ("GI", "th_motor"))
        assert motor is not None
        assert motor.label.text() == "θ motor"
        assert "incidence" in motor.editor.toolTip().lower()
        # Orient + Tilt are NOT inline -- they live behind the '…' button.
        assert _row(exp, ("GI", "sample_orientation")) is None
        assert _row(exp, ("GI", "tilt_angle")) is None
        more = _find_more_button(widget)
        assert more is not None

        # Clicking '…' opens a popup containing the Orientation + Tilt rows.
        more.click()
        popup = widget.controls_v2._gi_options_popup
        popup_paths = {r.path for r in popup.findChildren(FormRow)}
        assert ("GI", "sample_orientation") in popup_paths
        assert ("GI", "tilt_angle") in popup_paths

        # A popup row writes through to the integrator carrier.
        for r in popup.findChildren(FormRow):
            if r.path == ("GI", "sample_orientation"):
                r.editor.setText("6")
                r.editor.editingFinished.emit()
        assert widget.integratorTree.get_gi_config()["sample_orientation"] == 6
    finally:
        _reset_controls_v2_gi(widget)
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_path_fields_show_full_path_tooltip(qapp, monkeypatch):
    """Path/file fields (browse) show their FULL value on hover, since the editor
    truncates it in the narrow panel (a description would be less useful)."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget._on_controls_v2_field_changed(("Signal", "inp_type"), "Image Series")
        widget._on_controls_v2_field_changed(
            ("Signal", "File"), "/data/very/long/path/scan_0001.tif")
        widget._refresh_controls_v2_profile_now()

        rows = [
            r for r in widget.controls_v2.source_card.body.findChildren(FormRow)
            if r.path == ("Signal", "File")
        ]
        assert rows
        editor = rows[0].editor
        # The tooltip is the full path (the editor's own text), not empty/a blurb.
        assert editor.toolTip()
        assert editor.toolTip() == editor.text()
        assert "/" in editor.toolTip()
    finally:
        widget.close()
        widget.deleteLater()


def test_integrator_gi_motor_autoselects_preferred_over_manual(qapp, monkeypatch):
    """A new motor list with a preferred motor (th) auto-selects it instead of
    staying on the 'Manual' fallback; a deliberate real-motor choice is kept."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        it = widget.integratorTree
        # On 'Manual' (no motors), then a source whose metadata offers 'th'.
        it.set_gi_motor_options([])
        assert it.ui.gi_motor.currentText() == "Manual"
        it.set_gi_motor_options(["th", "i0", "eta"])
        assert it.ui.gi_motor.currentText() == "th"   # auto-selected, not Manual

        # A deliberate real-motor choice survives a same-list refresh.
        it.ui.gi_motor.setCurrentText("eta")
        it.set_gi_motor_options(["th", "i0", "eta"])
        assert it.ui.gi_motor.currentText() == "eta"
    finally:
        widget.close()
        widget.deleteLater()


def test_integrator_gi_motor_keeps_deliberate_manual_across_repopulation(qapp, monkeypatch):
    """F3: a DELIBERATELY chosen 'Manual' incidence motor stays Manual when the
    motor list repopulates on a source/format switch — the user's manual θ must
    not be silently swapped for a file motor.  The INITIAL DEFAULT Manual still
    yields to the preference order (th)."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        it = widget.integratorTree
        it._gi_motor_user_choice = None  # deterministic: no prior deliberate pick

        # Default (non-deliberate) state -> a source offering th auto-selects it.
        it.set_gi_motor_options(["th", "eta"])
        assert it.ui.gi_motor.currentText() == "th"

        # The user DELIBERATELY picks 'Manual' (fires the activated user-pick
        # signal that records the deliberate choice).
        it.ui.gi_motor.setCurrentText("Manual")
        it.ui.gi_motor.activated.emit(it.ui.gi_motor.currentIndex())
        assert it.ui.gi_motor.currentText() == "Manual"

        # A source/format switch repopulates the motor list -> Manual persists.
        it.set_gi_motor_options(["th", "eta", "gonth"])
        assert it.ui.gi_motor.currentText() == "Manual"
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_gi_popup_torn_down_on_rebuild_no_stale_clobber(qapp, monkeypatch):
    """F1/F2: the GI '…' popup is torn down on a panel rebuild, so a stale popup
    row can't be harvested by current_form_edits and clobber a fresher
    sample_orientation on the next pending-edit commit — and it leaves no orphan
    window."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget._on_controls_v2_field_changed(("GI", "Grazing"), True)
        widget._refresh_controls_v2_profile_now()

        # Open the '…' popup (Orientation + Tilt).
        more = _find_more_button(widget)
        more.click()
        assert widget.controls_v2._gi_options_popup is not None

        # An out-of-band orientation change + a panel rebuild.
        widget._on_controls_v2_field_changed(("GI", "sample_orientation"), "7")
        widget._refresh_controls_v2_profile_now()

        # Popup torn down: no orphan, nothing stale for current_form_edits.
        assert widget.controls_v2._gi_options_popup is None
        edit_paths = {e.path for e in widget.controls_v2.current_form_edits()}
        assert ("GI", "sample_orientation") not in edit_paths

        # Committing pending edits must NOT revert the fresh value.
        widget._commit_controls_v2_pending_edits()
        assert widget.integratorTree.get_gi_config()["sample_orientation"] == 7
    finally:
        _reset_controls_v2_gi(widget)
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_gi_popup_edit_commits_on_run(qapp, monkeypatch):
    """Finding 2: typing a new Orientation in the GI '…' popup and immediately
    committing pending edits (what Run does at run start) applies the value to
    THIS run — the popup is a transient widget, so the run-commit must harvest its
    in-progress edit, not let it land only on the *next* run."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget._on_controls_v2_field_changed(("GI", "Grazing"), True)
        widget._refresh_controls_v2_profile_now()

        more = _find_more_button(widget)
        more.click()
        popup = widget.controls_v2._gi_options_popup
        assert popup is not None

        # Type a new Orientation but do NOT commit it (no editingFinished) — as if
        # the user types and clicks Run immediately.
        for r in popup.findChildren(FormRow):
            if r.path == ("GI", "sample_orientation"):
                r.editor.setText("4")

        # Run's pending-edit commit harvests the in-progress popup value, so the
        # reduction config sees it for THIS run.
        widget._commit_controls_v2_pending_edits()
        assert widget.integratorTree.get_gi_config()["sample_orientation"] == 4
    finally:
        _reset_controls_v2_gi(widget)
        widget.close()
        widget.deleteLater()


def test_enter_run_state_resets_frame_count_snapshot(qapp, monkeypatch):
    """F6: a new run re-snapshots the frozen frame count from scratch, so it can't
    freeze at the PREVIOUS run's count if no inter-run refresh cleared it."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget._controls_v2_run_frame_count = 999  # stale leftover from a prior run
        widget._run_active = False
        widget._enter_run_state()
        # The stale snapshot is discarded at run start: it's reset to None and the
        # freeze logic re-snapshots the CURRENT frame count (never the old 999).
        assert widget._controls_v2_run_frame_count != 999
    finally:
        widget._exit_run_state()
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


def test_controls_panel_v2_run_state_harvests_and_deep_copies_snapshot(
        qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget._refresh_controls_v2_profile_now()
        rows = {
            row.path: row
            for row in widget.controls_v2.processing_card.body.findChildren(FormRow)
        }
        assert ("Int1D", "points") in rows

        # Simulate the user typing and immediately pressing Run: no
        # editingFinished has fired yet, so only the run-boundary harvest can
        # make this value part of the current run.
        rows[("Int1D", "points")].editor.setText("246")
        assert widget.integratorTree.ui.npts_1D.text() != "246"

        args = widget._apply_controls_v2_run_state()

        assert args is widget.wrangler.scan_args
        assert widget.integratorTree.ui.npts_1D.text() == "246"
        assert widget.scan.bai_1d_args["numpoints"] == 246
        assert args["bai_1d_args"]["numpoints"] == 246

        widget.scan.bai_1d_args["numpoints"] = 999
        assert args["bai_1d_args"]["numpoints"] == 246
        assert widget.wrangler.scan_args["bai_1d_args"]["numpoints"] == 246
    finally:
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


def test_controls_panel_v2_reintegrate_commits_focused_edit(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    class FakeButton:
        def __init__(self):
            self.clicked = False

        def click(self):
            self.clicked = True

    widget = staticWidget()
    fake = FakeButton()
    try:
        widget._refresh_controls_v2_profile_now()
        rows = {
            row.path: row
            for row in widget.controls_v2.processing_card.body.findChildren(FormRow)
        }
        rows[("Int1D", "points")].editor.setText("432")
        assert widget.integratorTree.ui.npts_1D.text() != "432"

        monkeypatch.setattr(widget.integratorTree.ui, "reintegrate1D", fake)
        widget._controls_v2_click_integrator_button("reintegrate1D")

        assert fake.clicked is True
        assert widget.integratorTree.ui.npts_1D.text() == "432"
        assert widget.scan.bai_1d_args["numpoints"] == 432
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_advanced_commits_focused_edit(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    calls = []
    try:
        widget._refresh_controls_v2_profile_now()
        rows = {
            row.path: row
            for row in widget.controls_v2.processing_card.body.findChildren(FormRow)
        }
        rows[("Int1D", "points")].editor.setText("543")
        assert widget.integratorTree.ui.npts_1D.text() != "543"

        def _fake_advanced():
            calls.append("advanced")
            assert widget.integratorTree.ui.npts_1D.text() == "543"
            assert widget.scan.bai_1d_args["numpoints"] == 543

        monkeypatch.setattr(widget, "_show_integration_advanced", _fake_advanced)
        widget._on_controls_v2_action(ControlAction.ADVANCED_PROCESSING)

        assert calls == ["advanced"]
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_range_labels_follow_legacy_integrator_labels(
        qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        # Standard mode: the live legacy label is authoritative here (in GI mode
        # gi_mode wins -- see _range_axis_labels_1d).  Force it explicitly so the
        # test is independent of any GI state leaked via session from a prior test.
        widget._on_controls_v2_field_changed(("GI", "Grazing"), False)
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
