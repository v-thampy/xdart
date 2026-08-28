"""Offscreen tests for the hidden Controls Panel V2 scaffold."""

import copy
import gc
import json
import os
import time
from pathlib import Path
from types import MethodType, SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("pyqtgraph")
from pyqtgraph import QtCore, QtWidgets
from pyqtgraph.Qt import QtTest


def _user_types(qapp, widget, editor, text):
    """Drive a REAL user edit into *editor*.

    §19.9: a programmatic ``editor.setText(...)`` on an unfocused row is a
    PROJECTION, not user intent — the no-focus full-form sweep that used to
    convert it into a journal entry is deleted.  The production input
    authorities are the keystroke draft signal (``textEdited`` →
    ``draftChanged``) and the focused-editor flush, and real focus + real
    ``keyClicks`` drives both, exactly as the amended §12 harvest tests do.
    """
    widget.show()
    qapp.processEvents()
    editor.setFocus()
    qapp.processEvents()
    assert editor.hasFocus()
    editor.selectAll()
    QtTest.QTest.keyClicks(editor, text)
    qapp.processEvents()

from xrd_tools.session.readiness import (
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
    GeomState,
    MeasMode,
    ProcessingPage,
    ResultCaps,
    RunTarget,
    SectionId,
    SourceCaps,
    StatusKind,
    Tool,
    build_control_profile,
    build_native_int_reduction_plan_from_args,
    build_native_int_reduction_plan_from_scan,
)
from xdart.gui.widgets.controls_panel import (
    ControlsPanel,
    FieldRow,
    FormRow,
    PillRow,
    RangeRow,
    SegmentedControl,
    SubsectionCard,
)


def _wait_until(qapp, predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture(autouse=True)
def _controls_panel_session_isolation():
    """Keep saved integrator state from leaking between tests in this module.

    Keep any saved control session from leaking between standalone panel tests.
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


def _find_form_row(widget, path):
    path = tuple(path)
    for card in (
        widget.controls_v2.project_card,
        widget.controls_v2.source_card,
        widget.controls_v2.experiment_card,
        widget.controls_v2.processing_card,
    ):
        for row in card.body.findChildren(FormRow):
            if getattr(row, "path", None) == path:
                return row
    return None


def _find_more_button(widget):
    """The '…' GI-options button in the Experiment card, or None."""
    for btn in widget.controls_v2.experiment_card.body.findChildren(
            QtWidgets.QToolButton):
        if btn.objectName() == "controlsV2MoreButton":
            return btn
    return None


def _find_source_energy_button(widget):
    """The Source-card energy-preference '…' button, or None."""
    for btn in widget.controls_v2.source_card.body.findChildren(
            QtWidgets.QToolButton):
        if (
            btn.objectName() == "controlsV2MoreButton"
            and btn.property("role") == "sourceEnergy"
        ):
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

    def _mask(mask):
        if mask is None:
            return None
        values = getattr(mask, "values", mask)
        arr = np.asarray(values)
        return {
            "kind": type(mask).__name__,
            "shape": tuple(arr.shape),
            "dtype": str(arr.dtype),
            "values": tuple(arr.ravel().tolist()) if arr.size <= 20 else None,
            "true_count": (
                int(arr.astype(bool, copy=False).sum())
                if arr.dtype == bool
                else None
            ),
        }

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
        "mask": _mask(plan.mask),
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


def _apply_v2_edits(widget, edits):
    for path, value in edits:
        widget._on_controls_v2_field_changed(path, value)


def _apply_prepared_run_state(widget):
    """Drive the post-admission owner with the one staged frozen identity.

    W-1R-D1 split preparation from admission.  Focused projection tests use
    this seam explicitly instead of relying on the deleted second-freeze
    fallback in ``_apply_controls_v2_run_state()``.
    """
    frozen = widget._prepare_controls_v2_run_configuration()
    assert frozen is not None
    return widget._apply_controls_v2_run_state(frozen), frozen


def _current_plan_snapshot(widget, *, include_threshold=True,
                           integrate_1d=True, integrate_2d=True,
                           commit_pending=True):
    from xdart.modules.reduction import (
        apply_threshold_saturation_to_plan,
        plan_from_live_scan,
    )

    if commit_pending:
        widget._commit_controls_v2_pending_edits()
    widget._controls_v2_ensure_native_int_defaults()
    widget._controls_v2_apply_gi_config_to_scan()
    plan = plan_from_live_scan(
        widget.scan,
        integrate_1d=integrate_1d,
        integrate_2d=integrate_2d,
    )
    if include_threshold:
        plan = apply_threshold_saturation_to_plan(
            plan,
            widget._controls_v2_threshold_config(),
        )
    return _plan_snapshot(plan)


def _native_plan_snapshot(widget, *, include_threshold=True,
                          integrate_1d=True, integrate_2d=True,
                          commit_pending=True):
    plan = widget._controls_v2_native_reduction_plan(
        include_threshold=include_threshold,
        integrate_1d=integrate_1d,
        integrate_2d=integrate_2d,
        commit_pending=commit_pending,
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


def _field_choice_text(widget, path, predicate, *, fallback_current=True):
    choices = widget._controls_v2_field_choices().get(tuple(path), ())
    for text in choices:
        if predicate(str(text)):
            return str(text)
    if fallback_current:
        return str(widget._controls_v2_field_values().get(tuple(path), ""))
    raise AssertionError(f"No matching choice for {path}")


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


@pytest.fixture(autouse=True)
def _drain_qt_events_after_test(qapp):
    yield
    for _ in range(3):
        qapp.processEvents()
    gc.collect()
    for _ in range(2):
        qapp.processEvents()


def test_controls_panel_renders_blockers_and_launchers(qapp):
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

    panel = ControlsPanel()
    panel.set_profile(profile)

    badges = panel.summary_card.body.findChildren(QtWidgets.QLabel)
    assert [b.text() for b in badges] == ["RSM GUI awaits real-data gate."]

    buttons = panel.analysis_card.body.findChildren(QtWidgets.QPushButton)
    assert [b.text() for b in buttons] == ["Peak Fitting", "Strain / sin²ψ"]
    assert buttons[0].isEnabled()
    assert not buttons[1].isEnabled()
    assert buttons[1].toolTip() == "Needs ψ metadata."


def test_controls_panel_emits_launcher_intent(qapp):
    profile = ControlProfile(
        processing_page=ProcessingPage.INT_1D,
        run_enabled=True,
        analysis_launchers=(
            AnalysisLauncherSpec(AnalysisTool.SCAN_PLOT, "Plot Metadata"),),
    )
    panel = ControlsPanel()
    panel.set_profile(profile)
    got = []
    panel.analysisLaunchRequested.connect(got.append)
    panel.analysis_card.body.findChildren(QtWidgets.QPushButton)[0].click()
    assert got == [AnalysisTool.SCAN_PLOT]


def test_controls_panel_emits_action_intent(qapp):
    profile = build_control_profile(
        ControlState(
            tool=Tool.INT_2D,
            project_root="/tmp/project",
            source_caps=SourceCaps(has_frames=True),
        )
    )
    panel = ControlsPanel()
    panel.set_profile(profile)
    got = []
    panel.controlActionRequested.connect(got.append)

    buttons = panel.project_card.body.findChildren(QtWidgets.QPushButton)
    buttons[0].click()

    assert got == [ControlAction.CHOOSE_PROJECT]


def test_controls_panel_backend_conflict_gates_run():
    profile = build_control_profile(
        ControlState(
            tool=Tool.STITCH,
            mode=MeasMode.GI,
            backend="multigeometry",
            source_caps=SourceCaps(has_frames=True, has_energy=True),
            geom=GeomState(
                calibrated=True,
                energy_known=True,
                gi_enabled=True,
                sample_orientation_known=True,
            ),
            real_data_gates=frozenset({"gi_stitch_real_data"}),
        )
    )

    backend = profile.fields[FieldId.PROCESSING_BACKEND]

    assert backend.status is StatusKind.CONFLICT
    assert "pyfai_hist" in backend.reason
    assert profile.can_run is False
    assert profile.run_blockers == (backend.reason,)


def test_controls_panel_renders_typed_field_cards(qapp):
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

    panel = ControlsPanel()
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


def test_controls_panel_renders_bound_render_state_directly(qapp):
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

    panel = ControlsPanel()
    panel.set_state(state)

    project_rows = panel.project_card.body.findChildren(FormRow)
    source_rows = panel.source_card.body.findChildren(FormRow)
    assert [row.label.text() for row in project_rows] == ["Folder"]
    assert [row.label.text() for row in source_rows] == ["Source"]
    assert not source_rows[0].editor.isEnabled()
    assert source_rows[0].toolTip() == "locked"
    assert panel.analysis_card.isHidden()


def test_controls_panel_detector_status_uses_poni_summary(qapp):
    profile = build_control_profile(
        ControlState(
            source_caps=SourceCaps(has_frames=True),
            detector_summary="Eiger 1M · 200.4mm · fitted",
        )
    )
    state = ControlPanelRenderState(
        profile=profile,
        bound_controls=BoundControlState(fields=(
            ControlFormField(
                section=SectionId.EXPERIMENT,
                label="Poni",
                path=("Signal", "poni_file"),
                value="/tmp/example.poni",
                browse=True,
            ),
        )),
    )

    panel = ControlsPanel()
    panel.set_state(state)

    detector = next(
        card for card in panel.experiment_card.body.findChildren(SubsectionCard)
        if card.title.text() == "Detector"
    )
    assert detector.status.text() == "Eiger 1M · 200.4mm · fitted"


def test_controls_panel_section_ticks_and_source_synopsis(qapp):
    profile = build_control_profile(
        ControlState(
            tool=Tool.INT_2D,
            project_root_required=True,
            project_root="/tmp/project",
            project_root_valid=True,
            source_label="/tmp/project/raw/scan_0001.tif",
            processing_mode="Int 2D",
            source_caps=SourceCaps(
                has_raw=True,
                raw_reachable=True,
                has_energy=True,
            ),
            geom=GeomState(calibrated=True, energy_known=True),
        )
    )
    state = ControlPanelRenderState(
        profile=profile,
        bound_controls=BoundControlState(fields=(
            ControlFormField(
                section=SectionId.PROJECT,
                label="Folder",
                path=("Project", "project_folder"),
                value="/tmp/project",
                browse=True,
            ),
            ControlFormField(
                section=SectionId.SOURCE,
                label="Source",
                path=("Signal", "inp_type"),
                value="Image Series",
                kind=ControlFieldKind.COMBO,
            ),
            ControlFormField(
                section=SectionId.EXPERIMENT,
                label="Poni",
                path=("Signal", "poni_file"),
                value="/tmp/project/cal.poni",
                browse=True,
            ),
        )),
    )

    panel = ControlsPanel()
    panel.set_state(state)

    assert panel.source_card.status.text() == "Image Series"
    assert not panel.project_card.valid_marker.isHidden()
    assert not panel.source_card.valid_marker.isHidden()
    assert not panel.experiment_card.valid_marker.isHidden()


def test_controls_panel_viewer_mode_shows_only_project(qapp):
    profile = build_control_profile(
        ControlState(tool=Tool.IMAGE_VIEWER, processing_mode="Image Viewer")
    )
    state = ControlPanelRenderState(
        profile=profile,
        bound_controls=BoundControlState(fields=(
            ControlFormField(
                section=SectionId.PROJECT,
                label="Folder",
                path=("Project", "project_folder"),
                value="",
                browse=True,
            ),
            ControlFormField(
                section=SectionId.SOURCE,
                label="Source",
                path=("Signal", "inp_type"),
                value="Image Series",
                kind=ControlFieldKind.COMBO,
            ),
            ControlFormField(
                section=SectionId.EXPERIMENT,
                label="Poni",
                path=("Signal", "poni_file"),
                value="",
                browse=True,
            ),
            ControlFormField(
                section=SectionId.PROCESSING,
                label="Background",
                path=("BG", "bg_type"),
                value="None",
                kind=ControlFieldKind.COMBO,
            ),
        )),
    )

    panel = ControlsPanel()
    panel.set_state(state)

    assert not panel.project_card.isHidden()
    assert panel.source_card.isHidden()
    assert panel.experiment_card.isHidden()
    assert panel.processing_card.isHidden()


def test_controls_panel_requires_valid_project_before_setup_cards(qapp):
    fields = (
        ControlFormField(
            section=SectionId.PROJECT,
            label="Folder",
            path=("Project", "project_folder"),
            value="",
            browse=True,
        ),
        ControlFormField(
            section=SectionId.SOURCE,
            label="Source",
            path=("Signal", "inp_type"),
            value="Image Series",
            kind=ControlFieldKind.COMBO,
        ),
        ControlFormField(
            section=SectionId.EXPERIMENT,
            label="Poni",
            path=("Signal", "poni_file"),
            value="",
            browse=True,
        ),
        ControlFormField(
            section=SectionId.PROCESSING,
            label="Background",
            path=("BG", "bg_type"),
            value="None",
            kind=ControlFieldKind.COMBO,
        ),
    )
    profile = build_control_profile(
        ControlState(
            project_root_required=True,
            project_root="",
            project_root_valid=False,
            processing_mode="Int 2D",
        )
    )

    panel = ControlsPanel()
    panel.set_state(ControlPanelRenderState(
        profile=profile,
        bound_controls=BoundControlState(fields=fields),
    ))

    assert not panel.project_card.isHidden()
    assert panel.source_card.isHidden()
    assert panel.experiment_card.isHidden()
    assert panel.processing_card.isHidden()


def test_run_readiness_label_elides_without_widening_controls(qapp):
    from xdart.gui.widgets.run_controls import RunControlsBar

    controls = RunControlsBar()
    try:
        controls.set_readiness_summary(
            "Needs setup · short blocker",
            ready=False,
            tooltip="short blocker",
        )
        controls.resize(360, controls.sizeHint().height())
        controls.show()
        qapp.processEvents()
        short_hint = controls.sizeHint().width()

        long_reason = "Needs setup · " + ("blocked configuration detail " * 20)
        controls.set_readiness_summary(
            long_reason,
            ready=False,
            tooltip=long_reason,
        )
        controls.resize(360, controls.sizeHint().height())
        qapp.processEvents()

        assert controls.sizeHint().width() == short_hint
        assert controls.readinessLabel.text() != long_reason
        assert controls.readinessLabel.text().endswith("…")
        assert controls.readinessLabel.toolTip() == long_reason
    finally:
        controls.close()
        controls.deleteLater()


def test_controls_panel_native_plan_preserves_monitor_parity():
    from xdart.modules.reduction import plan_from_live_scan

    args_1d = {
        "unit": "q_A^-1",
        "method": "csr",
        "numpoints": 250,
        "radial_range": (0.2, 4.4),
        "azimuth_range": (-30.0, 30.0),
        "monitor": "I0",
        "normalization_factor": 5.0,
        "error_model": "poisson",
        "polarization_factor": 0.95,
    }
    args_2d = {
        "unit": "q_A^-1",
        "method": "csr",
        "npt_rad": 80,
        "npt_azim": 90,
        "radial_range": (0.1, 5.0),
        "azimuth_range": (-90.0, 90.0),
        "chi_offset": 2.5,
        "monitor": "mon",
        "normalization_factor": 2.0,
        "error_model": "azimuthal",
        "polarization_factor": 0.9,
    }

    class FakeFrames:
        index = []

    class FakeScan:
        skip_2d = False
        gi = False
        global_mask = np.array([1, 4])
        detector_shape = (2, 3)
        frames = FakeFrames()
        bai_1d_args = dict(args_1d)
        bai_2d_args = dict(args_2d)

    legacy = plan_from_live_scan(FakeScan(), integrate_2d=True)
    native = build_native_int_reduction_plan_from_args(
        args_1d,
        args_2d,
        gi_enabled=False,
        integrate_1d=True,
        integrate_2d=True,
        detector_mask=FakeScan.global_mask,
        detector_shape=FakeScan.detector_shape,
    )

    assert _plan_snapshot(native) == _plan_snapshot(legacy)
    snapshot = _plan_snapshot(native)
    assert snapshot["integration_1d"]["monitor_key"] == "I0"
    assert snapshot["integration_2d"]["monitor_key"] == "mon"
    assert snapshot["mask"]["kind"] == "ndarray"
    assert snapshot["mask"]["shape"] == (2, 3)
    assert snapshot["mask"]["true_count"] == 2
    assert "normalization_factor" not in snapshot["integration_1d"]["extra"]
    assert "normalization_factor" not in snapshot["integration_2d"]["extra"]


def test_controls_panel_native_scan_builder_matches_legacy_plan():
    from xdart.modules.reduction import plan_from_live_scan

    args_1d = {
        "unit": "2th_deg",
        "method": "BBox",
        "numpoints": 321,
        "radial_range": (1.0, 4.0),
        "azimuth_range": (60.0, 120.0),
        "chi_offset": 90.0,
        "error_model": "poisson",
        "polarization_factor": 0.8,
        "correctSolidAngle": False,
        "dummy": -2.0,
        "delta_dummy": 0.1,
        "safe": False,
    }
    args_2d = {
        "unit": "q_A^-1",
        "method": "csr",
        "npt_rad": 77,
        "npt_azim": 88,
        "radial_range": (0.5, 5.0),
        "azimuth_range": (-45.0, 45.0),
        "chi_offset": 12.0,
        "error_model": "azimuthal",
        "polarization_factor": 0.9,
        "correctSolidAngle": True,
        "dummy": -3.0,
        "delta_dummy": 0.2,
        "safe": True,
    }

    class FakeFrames:
        index = []

    class FakeScan:
        skip_2d = False
        gi = False
        global_mask = np.array([1, 4])
        detector_shape = (2, 3)
        frames = FakeFrames()
        bai_1d_args = dict(args_1d)
        bai_2d_args = dict(args_2d)

    assert _plan_snapshot(build_native_int_reduction_plan_from_scan(
        FakeScan(), integrate_1d=True, integrate_2d=True
    )) == _plan_snapshot(plan_from_live_scan(
        FakeScan(), integrate_1d=True, integrate_2d=True
    ))


def test_controls_panel_native_gi_plan_defaults_orientation_to_4():
    args_plan = build_native_int_reduction_plan_from_args(
        {},
        {},
        gi_enabled=True,
        gi_incident_angle=0.1,
        integrate_2d=False,
    )
    assert args_plan.gi.sample_orientation == 4

    class FakeFrames:
        index = []

    class FakeScan:
        skip_2d = True
        gi = True
        _cached_fiber_integrator_angle = 0.1
        incidence_motor = None
        global_mask = None
        detector_shape = (2, 3)
        frames = FakeFrames()
        bai_1d_args = {}
        bai_2d_args = {}
        gi_config = {}

    scan_plan = build_native_int_reduction_plan_from_scan(
        FakeScan(), integrate_1d=True, integrate_2d=False
    )
    assert scan_plan.gi.sample_orientation == 4


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("Signal", "File"), "/data/raw/scan_0001.nxs"),
        (("Signal", "img_dir"), "/data/raw/images"),
        (("Signal", "poni_file"), "/data/calibration/cal.poni"),
        (("Signal", "mask_file"), "/data/calibration/mask.edf"),
        (("Project", "h5_dir"), "/data/processed/results"),
    ),
)
def test_compact_path_expands_for_direct_edit_then_collapses(
    qapp, path, value,
):
    row = FormRow(
        label="Path",
        path=path,
        value=value,
        browse=True,
    )
    row.show()
    qapp.processEvents()
    row.editor.clearFocus()
    qapp.processEvents()
    qapp.processEvents()
    edits = []
    drafts = []
    row.valueChanged.connect(lambda changed, text: edits.append((changed, text)))
    row.draftChanged.connect(lambda changed, text: drafts.append((changed, text)))
    try:
        assert row.editor.text() == Path(value).name
        assert row.current_value() == value

        row.editor.setFocus()
        qapp.processEvents()
        assert row.editor.text() == value
        assert row.current_value() == value
        assert edits == []
        assert drafts == []

        replacement = "/edited/directly/new-source.nxs"
        row.editor.selectAll()
        QtTest.QTest.keyClicks(row.editor, replacement)
        qapp.processEvents()
        assert row.current_value() == replacement
        assert drafts and drafts[-1] == (path, replacement)

        row.editor.clearFocus()
        qapp.processEvents()
        qapp.processEvents()
        assert edits[-1] == (path, replacement)
        assert row.current_value() == replacement
        assert row.editor.text() == Path(replacement).name
        assert row.editor.toolTip() == replacement
    finally:
        row.close()
        row.deleteLater()


def test_project_folder_remains_full_path_while_focused(qapp):
    value = "/data/very/long/project"
    row = FormRow(
        label="Folder",
        path=("Project", "project_folder"),
        value=value,
        browse=True,
    )
    row.show()
    qapp.processEvents()
    try:
        assert row.editor.text() == value
        row.editor.setFocus()
        qapp.processEvents()
        assert row.editor.text() == value
        row.editor.clearFocus()
        qapp.processEvents()
        assert row.editor.text() == value
    finally:
        row.close()
        row.deleteLater()


def test_rejected_compact_path_edit_restores_authoritative_path(qapp):
    path = ("Signal", "File")
    original = "/data/raw/scan_0001.nxs"
    row = FormRow(
        label="Image File",
        path=path,
        value=original,
        browse=True,
    )
    authoritative = ControlFormField(
        SectionId.SOURCE,
        "Image File",
        path,
        original,
        browse=True,
    )
    row.valueChanged.connect(
        lambda _path, _value: row.apply_field(authoritative)
    )
    row.show()
    qapp.processEvents()
    try:
        row.editor.setFocus()
        qapp.processEvents()
        row.editor.selectAll()
        QtTest.QTest.keyClicks(row.editor, "/rejected/missing.nxs")
        QtTest.QTest.keyClick(
            row.editor, QtCore.Qt.Key.Key_Return
        )
        qapp.processEvents()
        assert row.current_value() == original

        row.editor.clearFocus()
        qapp.processEvents()
        qapp.processEvents()
        assert row.current_value() == original
        assert row.editor.text() == Path(original).name
        assert row.editor.toolTip() == original
    finally:
        row.close()
        row.deleteLater()


def _admit_pending_run_configuration(widget):
    """Install the run admission a real ``imageWrangler.start()`` performs.

    X1 O-3 (§8.1): the run-state owner requires the handed-off frozen object
    and all four wrapper/worker carrier-and-ledger references to be ONE object
    before it will start a wrangler run.  These Controls cases drive
    ``start_wrangler()`` directly, so they have to stand in for the admission
    the production Start path would already have done — otherwise the double
    represents a run whose configuration was never admitted, which is exactly
    what the gate exists to refuse.
    """
    frozen = widget._require_controls_v2_run_handoff()
    if getattr(getattr(frozen, "source", None), "uri", None) is None:
        # X1 O-3 (§11.2.1): a wrangler run is refused unless its accepted
        # configuration names a SOURCE — the acquisition owner takes its source
        # identity from there and never from the mutable `scan.data_file`.
        # These cases drive `start_wrangler()` directly and so bypass the
        # readiness gate that makes a source mandatory before Start; the stand-
        # in supplies the one a real Start would already have carried.
        from dataclasses import replace as _replace

        from xrd_tools.session.run_configuration import FrozenSourceSpec

        frozen = _replace(frozen, source=FrozenSourceSpec(
            family="source", source_kind="image_file", uri="/raw/controls-test-source.h5"))
        # The handed-off object must be the SAME one: §8.1's five-reference
        # gate compares by identity, not by content.
        widget._pending_controls_v2_run_configuration = frozen
    for owner in (widget.wrangler, widget.wrangler.thread):
        for name in ("run_configuration", "_admitted_run_configuration"):
            setattr(owner, name, frozen)
    return frozen

@pytest.mark.parametrize("theme", ("dark", "light"))
def test_controls_panel_checked_disabled_matches_disabled_text_color(
    theme,
):
    from xdart.gui.themes import render_qss

    qss = render_qss(theme)

    def color_for(selector):
        start = qss.index(selector)
        body = qss[start:qss.index("}", start)]
        return next(
            line.strip() for line in body.splitlines()
            if line.strip().startswith("color:")
        )

    assert color_for(
        "QPushButton#controlsV2ToggleButton:checked:disabled"
    ) == color_for("QPushButton#controlsV2ToggleButton:disabled")



def test_apply_state_update_refuses_fast_path_when_fields_appear(qapp):
    """LV-UI-5: Standard→Grazing ADDS the θ-motor field; the in-place fast
    path must refuse (keys changed) so the full render mounts the new row —
    production falls back to ``set_state`` exactly as the shell does."""
    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.session.run_configuration import RunIntent
    from xdart.gui.tabs.scattering.controls_inventory import GI_MOTOR
    from xdart.gui.tabs.scattering.controls_projection import project_controls
    from xdart.gui.tabs.scattering.state_machine import RunPhase

    panel = ControlsPanel()
    try:
        intent = RunIntent()
        panel.set_state(project_controls(
            RunIntentStore(intent).snapshot(), None, RunPhase.IDLE))
        assert not [r for r in panel.findChildren(FormRow)
                    if tuple(r.path) == GI_MOTOR]

        intent.gi.enabled = True
        grazing = project_controls(
            RunIntentStore(intent).snapshot(), None, RunPhase.IDLE)
        assert panel.apply_state_update(grazing) is False
        panel.set_state(grazing)
        rows = [r for r in panel.findChildren(FormRow)
                if tuple(r.path) == GI_MOTOR]
        assert rows, "theta-motor row must mount on the Grazing switch"
    finally:
        panel.close()
        panel.deleteLater()


def test_apply_state_update_updates_action_buttons_without_spurious_repolish(
    qapp, monkeypatch,
):
    """Unchanged/action-only refreshes avoid polish; style changes do not."""
    from dataclasses import replace

    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.session.readiness import ControlAction
    from xrd_tools.session.run_configuration import RunIntent
    from xdart.gui.tabs.scattering.controls_projection import project_controls
    from xdart.gui.tabs.scattering.state_machine import RunPhase
    from xdart.gui.widgets.controls_panel import ActionButton

    store = RunIntentStore(RunIntent())
    disabled = project_controls(
        store.snapshot(), None, RunPhase.IDLE,
        reintegrate_available=False,
    )
    enabled = project_controls(
        store.snapshot(), None, RunPhase.IDLE,
        reintegrate_available=True,
    )
    panel = ControlsPanel()
    try:
        panel.set_state(disabled)
        before = next(
            button
            for button in panel.findChildren(ActionButton)
            if button.spec.action is ControlAction.REINTEGRATE_1D
        )
        row_before = panel.findChildren(FormRow)[0]
        editor_before = row_before.editor
        assert not before.isEnabled()
        assert "stable processed Browse artifact" in before.toolTip()
        calibrate_before = next(
            button
            for button in panel.findChildren(ActionButton)
            if button.spec.action is ControlAction.CALIBRATE
        )
        assert calibrate_before.text() == "⌖ Calibrate"

        assert panel.apply_state_update(enabled) is True
        qapp.processEvents()
        after = next(
            button
            for button in panel.findChildren(ActionButton)
            if button.spec.action is ControlAction.REINTEGRATE_1D
            and button.isEnabled()
        )
        assert after is before
        assert panel.findChildren(FormRow)[0] is row_before
        assert row_before.editor is editor_before
        assert "Replaces selected 1-D results" in after.toolTip()

        style = after.style()
        original_unpolish = style.unpolish
        original_polish = style.polish
        repolished: list[tuple[str, ActionButton]] = []

        def record_unpolish(widget):
            if isinstance(widget, ActionButton):
                repolished.append(("unpolish", widget))
            return original_unpolish(widget)

        def record_polish(widget):
            if isinstance(widget, ActionButton):
                repolished.append(("polish", widget))
            return original_polish(widget)

        monkeypatch.setattr(style, "unpolish", record_unpolish)
        monkeypatch.setattr(style, "polish", record_polish)

        # A controls-only refresh commonly projects an equal immutable state.
        # It must neither rebuild nor force every action through the style
        # engine: that global repolish is visible as a whole-panel flicker.
        assert panel.apply_state_update(enabled) is True
        assert repolished == []
        assert next(
            button
            for button in panel.findChildren(ActionButton)
            if button.spec.action is ControlAction.REINTEGRATE_1D
        ) is after

        active = project_controls(
            store.snapshot(), None, RunPhase.IDLE,
            operation_busy=True,
            calibration_active=True,
            reintegrate_available=True,
        )
        assert panel.apply_state_update(active) is True
        assert next(
            button
            for button in panel.findChildren(ActionButton)
            if button.spec.action is ControlAction.CALIBRATE
        ) is calibrate_before
        assert calibrate_before.text() == "Cancel Calibration"
        assert repolished == []

        # ``productionReady`` participates in the stylesheet selector.  A
        # genuine change to it must still update the dynamic property and
        # repolish exactly that one action button.
        actions = dict(active.profile.section_actions)
        actions[SectionId.EXPERIMENT] = tuple(
            replace(spec, production_ready=False)
            if spec.action is ControlAction.CALIBRATE
            else spec
            for spec in actions[SectionId.EXPERIMENT]
        )
        style_changed = replace(
            active,
            profile=replace(active.profile, section_actions=actions),
        )
        assert panel.apply_state_update(style_changed) is True
        assert calibrate_before.property("productionReady") is False
        assert repolished == [
            ("unpolish", calibrate_before),
            ("polish", calibrate_before),
        ]

        actions = dict(style_changed.profile.section_actions)
        processing = actions[SectionId.PROCESSING]
        actions[SectionId.PROCESSING] = tuple(reversed(processing))
        reordered = replace(
            style_changed,
            profile=replace(style_changed.profile, section_actions=actions),
        )
        assert panel.apply_state_update(reordered) is False
    finally:
        panel.close()
        panel.deleteLater()


def test_apply_state_update_locks_gi_more_without_rebuilding(qapp):
    """A controls-only operation refresh must not leave the GI popup live."""
    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.session.run_configuration import RunIntent
    from xdart.gui.tabs.scattering.controls_projection import project_controls
    from xdart.gui.tabs.scattering.state_machine import RunPhase

    intent = RunIntent()
    intent.gi.enabled = True
    store = RunIntentStore(intent)
    idle = project_controls(store.snapshot(), None, RunPhase.IDLE)
    busy = project_controls(
        store.snapshot(), None, RunPhase.IDLE,
        operation_busy=True,
        calibration_active=True,
    )
    panel = ControlsPanel()
    try:
        panel.set_state(idle)
        more = next(
            button
            for button in panel.experiment_card.body.findChildren(
                QtWidgets.QToolButton
            )
            if button.objectName() == "controlsV2MoreButton"
        )
        first_row = panel.findChildren(FormRow)[0]
        more.click()
        qapp.processEvents()
        assert panel._gi_options_popup is not None

        assert panel.apply_state_update(busy) is True
        qapp.processEvents()
        assert more is next(
            button
            for button in panel.experiment_card.body.findChildren(
                QtWidgets.QToolButton
            )
            if button.objectName() == "controlsV2MoreButton"
        )
        assert panel.findChildren(FormRow)[0] is first_row
        assert not more.isEnabled()
        assert panel._gi_options_popup is None
        assert {
            subsection.status.text()
            for subsection in panel.processing_card.body.findChildren(
                SubsectionCard
            )
        } == {"locked"}

        more.click()
        qapp.processEvents()
        assert panel._gi_options_popup is None
    finally:
        panel.close()
        panel.deleteLater()


def test_apply_state_update_refreshes_source_energy_popup_capture(qapp):
    """The Source More button must open the newly projected preference."""
    from dataclasses import replace

    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.session.run_configuration import RunIntent
    from xdart.gui.tabs.scattering.controls_projection import project_controls
    from xdart.gui.tabs.scattering.state_machine import RunPhase

    base = project_controls(
        RunIntentStore(RunIntent()).snapshot(), None, RunPhase.IDLE,
    )
    energy = ControlFormField(
        SectionId.SOURCE,
        "Energy Source",
        ("Source", "energy_preference"),
        "poni",
        kind=ControlFieldKind.COMBO,
        choices=("poni", "metadata"),
    )
    before = replace(
        base,
        bound_controls=replace(
            base.bound_controls,
            fields=base.bound_controls.fields + (energy,),
        ),
    )
    after = replace(
        before,
        bound_controls=replace(
            before.bound_controls,
            fields=tuple(
                replace(field, value="metadata")
                if field.path == energy.path
                else field
                for field in before.bound_controls.fields
            ),
        ),
    )
    panel = ControlsPanel()
    try:
        panel.set_state(before)
        button = next(
            candidate
            for candidate in panel.source_card.body.findChildren(
                QtWidgets.QToolButton
            )
            if candidate.property("role") == "sourceEnergy"
        )
        assert panel.apply_state_update(after) is True
        assert button is next(
            candidate
            for candidate in panel.source_card.body.findChildren(
                QtWidgets.QToolButton
            )
            if candidate.property("role") == "sourceEnergy"
        )

        button.click()
        qapp.processEvents()
        popup = panel._source_energy_popup
        assert popup is not None
        segmented = popup.findChild(SegmentedControl)
        assert segmented is not None
        assert segmented.current_value() == "metadata"
    finally:
        panel.close()
        panel.deleteLater()


def test_apply_state_update_refreshes_derived_subsection_statuses(qapp):
    from dataclasses import replace

    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.session.run_configuration import RunIntent
    from xdart.gui.tabs.scattering.controls_projection import project_controls
    from xdart.gui.tabs.scattering.state_machine import RunPhase

    before = project_controls(
        RunIntentStore(RunIntent()).snapshot(), None, RunPhase.IDLE,
    )
    after = replace(
        before,
        profile=replace(before.profile, detector_summary="Eiger4M · fitted"),
    )
    panel = ControlsPanel()
    try:
        panel.set_state(before)
        detector = next(
            subsection
            for subsection in panel.experiment_card.body.findChildren(
                SubsectionCard
            )
            if subsection.title.text() == "Detector"
        )
        assert detector.status.text() != "Eiger4M · fitted"
        assert panel.apply_state_update(after) is True
        assert detector.status.text() == "Eiger4M · fitted"
        assert detector.status.isVisibleTo(detector)
    finally:
        panel.close()
        panel.deleteLater()


def test_apply_state_update_refuses_render_schema_change_before_mutation(qapp):
    from dataclasses import replace

    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.session.run_configuration import RunIntent
    from xdart.gui.tabs.scattering.controls_projection import project_controls
    from xdart.gui.tabs.scattering.state_machine import RunPhase

    before = project_controls(
        RunIntentStore(RunIntent()).snapshot(), None, RunPhase.IDLE,
    )
    target = before.bound_controls.fields[0]
    after = replace(
        before,
        bound_controls=replace(
            before.bound_controls,
            fields=(replace(target, label=target.label + " changed"),)
            + before.bound_controls.fields[1:],
        ),
    )
    panel = ControlsPanel()
    try:
        panel.set_state(before)
        row = next(
            candidate
            for candidate in panel.findChildren(FormRow)
            if candidate.path == target.path
        )
        label = row.label.text()
        assert panel.apply_state_update(after) is False
        assert panel._bound_state is before.bound_controls
        assert row.label.text() == label
    finally:
        panel.close()
        panel.deleteLater()


def test_threshold_and_mask_saturated_are_independent_in_vnext(qapp):
    """Both switches are editable and each emits only its own exact path."""
    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.session.run_configuration import RunIntent
    from xdart.gui.tabs.scattering.controls_inventory import (
        MASK_SATURATION,
        THRESHOLD_ENABLED,
    )
    from xdart.gui.tabs.scattering.controls_projection import project_controls
    from xdart.gui.tabs.scattering.state_machine import RunPhase
    from xdart.gui.widgets.controls_panel import PillRow, RangeRow

    panel = ControlsPanel()
    try:
        panel.set_state(project_controls(
            RunIntentStore(RunIntent()).snapshot(), None, RunPhase.IDLE))
        row = next(
            r for r in panel.findChildren(RangeRow)
            if tuple(r._low_path) == ("Mask", "min")
        )
        tpath, btn = row._toggle
        assert tuple(tpath) == THRESHOLD_ENABLED
        assert not btn.isChecked()                 # manual band is off
        assert not row._low.isEnabled()
        assert not row._high.isEnabled()

        emitted = []
        panel.fieldValueChanged.connect(
            lambda p, v: emitted.append((tuple(p), v)))
        btn.setChecked(True)
        assert (THRESHOLD_ENABLED, True) in emitted

        pill_rows = [
            p for p in panel.findChildren(PillRow)
            if any(tuple(path) == MASK_SATURATION for path, _ in p._pills)
        ]
        assert pill_rows, "the Mask Saturated pill must stay visible"
        pill = next(
            b for path, b in pill_rows[0]._pills
            if tuple(path) == MASK_SATURATION
        )
        assert pill.isEnabled()
        assert pill.isChecked()
        assert pill.objectName() == "controlsV2PillButton"
        assert (MASK_SATURATION, True) in pill_rows[0].current_edits()
        pill.setChecked(False)
        assert (MASK_SATURATION, False) in emitted
        pill_top = pill_rows[0].layout().contentsMargins().top()
        conditioning = next(
            card
            for card in panel.processing_card.body.findChildren(SubsectionCard)
            if card.title.text() == "Conditioning"
        )
        assert pill_top == 3
        assert (
            conditioning.body_layout.spacing() + pill_top
            == conditioning.body_layout.contentsMargins().bottom()
        )

        # Scope guard (Codex P2): the seeded max is a detector-FAMILY display
        # default, not an acquisition-dtype fact — the RENDERED max-bound
        # widget must say so whether the band is disabled or enabled.
        assert "display default" in row._high.toolTip()

        manual = RunIntent()
        manual.threshold.mask_saturation = False
        manual.threshold.apply_threshold = True
        panel.set_state(project_controls(
            RunIntentStore(manual).snapshot(), None, RunPhase.IDLE))
        manual_row = next(
            r for r in panel.findChildren(RangeRow)
            if tuple(r._low_path) == ("Mask", "min")
        )
        assert manual_row._toggle[1].isChecked()
        assert manual_row._high.isEnabled()
        assert "display default" in manual_row._high.toolTip()
        manual_pill = next(
            b
            for pills in panel.findChildren(PillRow)
            for path, b in pills._pills
            if tuple(path) == MASK_SATURATION
        )
        assert manual_pill.isEnabled()
        assert not manual_pill.isChecked()
    finally:
        panel.close()
        panel.deleteLater()


def test_threshold_bounds_render_without_decimals_without_rounding_the_model(
    qapp,
):
    """Threshold formatting is presentation-only; edits retain float truth."""

    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.session.run_configuration import RunIntent
    from xdart.gui.tabs.scattering.controls_projection import project_controls
    from xdart.gui.tabs.scattering.state_machine import RunPhase

    def _threshold_row(host):
        return next(
            row for row in host.findChildren(RangeRow)
            if tuple(row._low_path) == ("Mask", "min")
        )

    intent = RunIntent()
    intent.threshold.apply_threshold = True
    intent.threshold.mask_saturation = False
    intent.threshold.threshold_min = 12.0
    intent.threshold.threshold_max = 4294967295.0
    panel = ControlsPanel()
    generic = RangeRow(
        label="Q",
        low={"path": ("Int1D", "radial_low"), "value": 0.25},
        high={"path": ("Int1D", "radial_high"), "value": 4.75},
    )
    try:
        panel.set_state(project_controls(
            RunIntentStore(intent).snapshot(), None, RunPhase.IDLE
        ))
        row = _threshold_row(panel)
        assert row._low.text() == "12"
        assert row._high.text() == "4294967295"
        assert (("Mask", "min"), "12.0") in row.current_edits()
        assert (
            (("Mask", "max"), "4294967295.0") in row.current_edits()
        )

        emitted = []
        row.valueChanged.connect(
            lambda path, value: emitted.append((tuple(path), value))
        )
        row._low.editingFinished.emit()
        assert emitted[-1] == (("Mask", "min"), "12.0")

        _user_types(qapp, panel, row._low, "12")
        focused = panel.focused_form_edit()
        assert focused is not None
        assert focused.path == ("Mask", "min")
        assert focused.value == "12"
        row._low.editingFinished.emit()
        assert emitted[-1] == (("Mask", "min"), "12")

        row._low.clearFocus()
        qapp.processEvents()
        # A later accepted exact value can share the same rounded display
        # bucket.  It must clear the old draft marker and remain authoritative
        # when the untouched editor is focused/harvested again.
        assert panel.apply_state_update(project_controls(
            RunIntentStore(intent).snapshot(), None, RunPhase.IDLE
        ))
        row = _threshold_row(panel)
        assert row._low.text() == "12"
        assert (("Mask", "min"), "12.0") in row.current_edits()
        row._low.setFocus()
        qapp.processEvents()
        focused = panel.focused_form_edit()
        assert focused is not None and focused.value == "12.0"
        row._low.clearFocus()
        qapp.processEvents()

        # A deliberately non-integral draft remains truthful after its
        # accepted commit and blur; zero-decimal formatting must never render
        # 1.5 as a different executed value such as 2.
        _user_types(qapp, panel, row._low, "1.5")
        row._low.editingFinished.emit()
        assert emitted[-1] == (("Mask", "min"), "1.5")
        row._low.clearFocus()
        qapp.processEvents()

        updated = RunIntent()
        updated.threshold.apply_threshold = True
        updated.threshold.mask_saturation = False
        updated.threshold.threshold_min = 1.5
        updated.threshold.threshold_max = 100.75
        assert panel.apply_state_update(project_controls(
            RunIntentStore(updated).snapshot(), None, RunPhase.IDLE
        ))
        row = _threshold_row(panel)
        assert row._low.text() == "1.5"
        assert row._high.text() == "100.75"
        assert (("Mask", "min"), "1.5") in row.current_edits()
        assert (("Mask", "max"), "100.75") in row.current_edits()

        # Other numeric ranges keep their existing precision.
        assert generic._low.text() == "0.25"
        assert generic._high.text() == "4.75"
    finally:
        generic.close()
        generic.deleteLater()
        panel.close()
        panel.deleteLater()


def test_max_bound_scope_caveat_survives_run_lock(qapp):
    """DESIGN_STOP secondary (2026-08-04): the lock reason outranks the
    tooltip table too, so the detector-scope caveat must ride the LOCKED
    max-bound reason — at construction under lock and across in-place
    lock/unlock state updates."""
    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.session.run_configuration import RunIntent
    from xdart.gui.tabs.scattering.controls_projection import project_controls
    from xdart.gui.tabs.scattering.state_machine import RunPhase
    from xdart.gui.widgets.controls_panel import RangeRow

    def _row(host):
        return next(
            r for r in host.findChildren(RangeRow)
            if tuple(r._low_path) == ("Mask", "min")
        )

    store = RunIntentStore(RunIntent())
    panel = ControlsPanel()
    try:
        # Construction while run-locked.
        panel.set_state(project_controls(
            store.snapshot(), None, RunPhase.RUNNING))
        locked = _row(panel)
        assert not locked._high.isEnabled()
        assert "locked" in locked._high.toolTip()
        assert "display default" in locked._high.toolTip()

        # In-place unlock, then re-lock, through the update path.
        if not panel.apply_state_update(project_controls(
                store.snapshot(), None, RunPhase.IDLE)):
            panel.set_state(project_controls(
                store.snapshot(), None, RunPhase.IDLE))
        assert "display default" in _row(panel)._high.toolTip()
        if not panel.apply_state_update(project_controls(
                store.snapshot(), None, RunPhase.RUNNING)):
            panel.set_state(project_controls(
                store.snapshot(), None, RunPhase.RUNNING))
        relocked = _row(panel)
        assert "locked" in relocked._high.toolTip()
        assert "display default" in relocked._high.toolTip()
    finally:
        panel.close()
        panel.deleteLater()


def test_range_toggle_presents_manual_on_while_preserving_auto_model(qapp):
    """A lit range toggle means its boxes apply; storage remains ``*_auto``."""
    from xdart.gui.widgets.controls_panel import RangeRow

    emitted = []
    row = RangeRow(
        label="Q",
        low={"path": ("Int1D", "radial_low"), "value": 0.0},
        high={"path": ("Int1D", "radial_high"), "value": 5.0},
        toggle={"path": ("Int1D", "radial_auto"), "value": True},
    )
    row.valueChanged.connect(lambda p, v: emitted.append((tuple(p), v)))
    try:
        btn = row._toggle[1]
        assert not btn.isChecked()                     # model is Auto
        assert (("Int1D", "radial_auto"), True) in row.current_edits()
        btn.setChecked(True)                           # entered bounds ON
        assert emitted == [(("Int1D", "radial_auto"), False)]
        assert (("Int1D", "radial_auto"), False) in row.current_edits()
    finally:
        row.close()
        row.deleteLater()


@pytest.mark.parametrize(
    "path",
    (
        ("Int1D", "radial_auto"),
        ("Int1D", "azim_auto"),
        ("Int2D", "radial_auto"),
        ("Int2D", "azim_auto"),
    ),
)
def test_all_integration_range_toggles_invert_only_the_ui_fact(qapp, path):
    """Every integration range uses lit=manual without changing storage."""

    stem = path[-1].removesuffix("_auto")
    low_path = (path[0], f"{stem}_low")
    high_path = (path[0], f"{stem}_high")
    row = RangeRow(
        label=stem,
        low={"path": low_path, "value": 0.0},
        high={"path": high_path, "value": 5.0},
        toggle={"path": path, "value": True},
    )
    try:
        button = row._toggle[1]
        assert not button.isChecked()
        assert "entered range" in button.toolTip()
        fields = {
            low_path: ControlFormField(
                SectionId.PROCESSING, "Low", low_path, 1.0,
            ),
            high_path: ControlFormField(
                SectionId.PROCESSING, "High", high_path, 4.0,
            ),
            path: ControlFormField(
                SectionId.PROCESSING, "Manual", path, False,
                kind=ControlFieldKind.BOOL,
            ),
        }
        assert row.apply_fields(fields)
        assert button.isChecked()
        assert "entered range" in button.toolTip()
        assert (path, False) in row.current_edits()

        fields[path] = ControlFormField(
            SectionId.PROCESSING, "Manual", path, True,
            kind=ControlFieldKind.BOOL,
        )
        assert row.apply_fields(fields)
        assert not button.isChecked()
        assert (path, True) in row.current_edits()
    finally:
        row.close()
        row.deleteLater()


def test_threshold_toggle_keeps_direct_apply_polarity(qapp):
    """Threshold's direct enable stays checked exactly when the band applies."""
    from xdart.gui.widgets.controls_panel import RangeRow

    emitted = []
    row = RangeRow(
        label="Threshold",
        low={"path": ("Mask", "min"), "value": None},
        high={"path": ("Mask", "max"), "value": None},
        toggle={"path": ("Mask", "Threshold"), "value": False},
    )
    row.valueChanged.connect(lambda p, v: emitted.append((tuple(p), v)))
    try:
        btn = row._toggle[1]
        assert not btn.isChecked()                    # off -> not applied
        assert (("Mask", "Threshold"), False) in row.current_edits()
        btn.setChecked(True)                          # user enables threshold
        assert emitted == [(("Mask", "Threshold"), True)]
        assert (("Mask", "Threshold"), True) in row.current_edits()
    finally:
        row.close()
        row.deleteLater()

# ── SW-7 (§10): wrangler swap resyncs the θ-motor choices to the owner ─────


# ── SW-8 (§10): Axis edits keep the gi_config mode copy in sync ─────────────


# ── POL-1: fresh-scan polarization default ON at 0.99 ──────────────────────


# ── Live chip + waiting status in the readiness bar (2026-07-13) ────────────


# ── DIR-1: the outgoing scan's frame list paints before the boundary clear ──


# ---------------------------------------------------------------------------
# T-2R (§14.11) — checked-commit atomicity: preflight, push-before-setter,
# rollback-all, contained exceptions, display-scan rollback, typed attempt
# result.  Each is RED at 1f8b4255 and GREEN at the T-2R correction.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# T-2.1 (§13.6/§13.7/§13.8/§13.11 + §14.11.B.4/D.2) — compound PONI carrier,
# source-token GI hydration, child-only-dir UNKNOWN, candidate SourceSpec.
# Each is RED at 1f8b4255 and GREEN at the T-2.1 correction.
# ---------------------------------------------------------------------------


def _write_poni(path, dist):
    path.write_text(
        f"Distance: {dist}\nPoni1: 0.01\nPoni2: 0.02\n"
        "Rot1: 0.0\nRot2: 0.0\nRot3: 0.0\nWavelength: 1.0e-10\n"
    )
    return str(path)


def test_section_number_presentation_option_preserves_canonical_default(qapp):
    compact = ControlsPanel(show_section_numbers=False)
    canonical = ControlsPanel()
    try:
        canonical_chips = canonical.findChildren(
            QtWidgets.QLabel, "controlsV2SectionChip")
        compact_chips = compact.findChildren(
            QtWidgets.QLabel, "controlsV2SectionChip")

        assert any(
            chip.text() == "1" and not chip.isHidden()
            for chip in canonical_chips)
        assert all(chip.isHidden() for chip in compact_chips)
    finally:
        canonical.close()
        canonical.deleteLater()
        compact.close()
        compact.deleteLater()


_COMBO_PATH = ("gi", "motor")


def _combo_row(*, value="Manual", choices=("Manual",)):
    return FormRow(
        label="Motor", path=_COMBO_PATH, value=value,
        kind="combo", choices=choices,
    )


def _combo_field(choices, value="Manual"):
    return ControlFormField(
        SectionId.EXPERIMENT, "Motor", _COMBO_PATH, value,
        ControlFieldKind.COMBO, tuple(choices),
    )


def _combo_items(row):
    return tuple(
        row.editor.itemText(index) for index in range(row.editor.count())
    )


def test_form_row_combo_reconciles_added_removed_and_reordered_choices(qapp):
    """Projected ``field.choices`` are the authoritative combo vocabulary."""
    row = _combo_row()
    emitted = []
    row.valueChanged.connect(lambda *values: emitted.append(values))
    try:
        assert row.apply_field(_combo_field(("Manual", "th", "eta")))
        assert _combo_items(row) == ("Manual", "th", "eta")

        assert row.apply_field(_combo_field(("Manual", "eta")))
        assert _combo_items(row) == ("Manual", "eta")

        assert row.apply_field(_combo_field(("eta", "Manual")))
        assert _combo_items(row) == ("eta", "Manual")

        assert row.editor.currentText() == "Manual"
        qapp.processEvents()
        assert emitted == []
    finally:
        row.close()
        row.deleteLater()


def test_form_row_combo_appends_absent_explicit_current_last(qapp):
    """An absent explicit current value is appended LAST, typed order kept."""
    row = _combo_row()
    try:
        assert row.apply_field(_combo_field(("Manual", "eta"), value="halpha"))
        assert _combo_items(row) == ("Manual", "eta", "halpha")
        assert row.editor.currentText() == "halpha"
    finally:
        row.close()
        row.deleteLater()


def test_form_row_combo_reconciliation_emits_no_field_value_changed(qapp):
    """A rebuild is silent on ``fieldValueChanged``; a real pick still emits."""
    panel = ControlsPanel()
    row = _combo_row(value="halpha", choices=("Manual", "halpha"))
    row.valueChanged.connect(panel.fieldValueChanged)
    emitted = []
    panel.fieldValueChanged.connect(lambda *values: emitted.append(values))
    try:
        assert row.apply_field(_combo_field(("Manual", "eta"), value="eta"))
        qapp.processEvents()
        assert emitted == []
        assert row.editor.currentText() == "eta"

        # E4-accepted seam: ANY programmatic index/text set stays silent (a
        # synchronous rebuild from currentTextChanged could destroy a native
        # popup mid-open); only a genuine user activation commits, deferred
        # one event-loop turn.
        row.editor.setCurrentIndex(0)
        qapp.processEvents()
        assert emitted == []
        row.editor.textActivated.emit("Manual")
        qapp.processEvents()
        assert emitted == [(_COMBO_PATH, "Manual")]
    finally:
        row.close()
        row.deleteLater()
        panel.close()
        panel.deleteLater()


def test_form_row_combo_restores_a_previously_blocked_signal_state(qapp):
    """``apply_field`` restores the PRIOR blocked state, never force-clears."""
    row = _combo_row()
    try:
        row.editor.blockSignals(True)
        assert row.apply_field(_combo_field(("Manual", "eta"), value="eta"))
        assert row.editor.signalsBlocked() is True
    finally:
        row.close()
        row.deleteLater()
