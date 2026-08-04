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
from pyqtgraph import QtWidgets
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
    GeomState,
    INTEGRATOR_BACKED_CONTROL_SPECS,
    INTEGRATION_CONTROL_PATHS,
    MeasMode,
    ProcessingPage,
    ResultCaps,
    RunTarget,
    SectionId,
    SourceCaps,
    StatusKind,
    Tool,
    build_control_panel_state,
    build_control_profile,
    build_native_int_reduction_plan_from_args,
    build_native_int_reduction_plan_from_scan,
)
from xdart.gui.tabs.static_scan.ui.controls_panel_v2 import (
    ControlsPanelV2,
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


def test_controls_panel_v2_int_inventory_includes_units_and_advanced_rows():
    specs = {spec.path: spec for spec in INTEGRATOR_BACKED_CONTROL_SPECS}
    required = {
        ("Int1D", "unit"),
        ("Int2D", "unit"),
        ("Int1D", "method"),
        ("Int2D", "method"),
        ("Int1D", "correctSolidAngle"),
        ("Int2D", "correctSolidAngle"),
        ("Int1D", "apply_polarization"),
        ("Int2D", "apply_polarization"),
        ("Int1D", "polarization_factor"),
        ("Int2D", "polarization_factor"),
        ("Int1D", "dummy"),
        ("Int2D", "dummy"),
        ("Int1D", "delta_dummy"),
        ("Int2D", "delta_dummy"),
        ("Int1D", "chi_offset"),
        ("Int2D", "chi_offset"),
        ("Int1D", "safe"),
        ("Int2D", "safe"),
    }

    assert required <= set(specs)
    assert required <= set(INTEGRATION_CONTROL_PATHS)
    assert specs[("Int1D", "unit")].widget_name == "unit_1D"
    assert specs[("Int2D", "unit")].widget_name == "unit_2D"
    assert specs[("Int1D", "method")].parameter_name == "method"
    assert specs[("Int2D", "method")].parameter_name == "method"
    for path in required:
        expected_group = "1d" if path[0] == "Int1D" else "2d"
        assert specs[path].parameter_group == expected_group


def test_controls_panel_v2_advanced_rows_not_rendered_inline(qapp, monkeypatch):
    """The Advanced (parameter_group-backed) params have ONE editing surface:
    the "Advanced" button on the Reintegrate row → the "Integration — Advanced
    Settings" dialog.  The old collapsed "Advanced" sub-panel duplicated that
    dialog field-for-field (maintainer decision 2026-07-12: removed) — no
    Advanced subsection renders, no advanced path leaks into the inline
    groups, and the button remains."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    def _subsections(widget):
        return {
            sub.title.text(): sub
            for sub in widget.controls_v2.processing_card.body.findChildren(
                SubsectionCard
            )
        }

    def _paths(sub):
        paths = {row.path for row in sub.body.findChildren(FormRow)}
        for row in sub.body.findChildren(RangeRow):
            paths.add(tuple(row._low_path))
            paths.add(tuple(row._high_path))
            toggle = getattr(row, "_toggle", None)
            if toggle is not None:
                paths.add(tuple(toggle[0]))
        for row in sub.body.findChildren(PillRow):
            paths.update(path for path, _ in row.current_edits())
        return paths

    widget = staticWidget()
    try:
        widget._refresh_controls_v2_profile_now()

        sections = _subsections(widget)
        assert {"1-D", "2-D"} <= set(sections)
        assert "Advanced" not in sections

        one_d_paths = _paths(sections["1-D"])
        two_d_paths = _paths(sections["2-D"])
        all_rendered = set().union(*(_paths(sub) for sub in sections.values()))

        assert {
            ("Int1D", "axis"),
            ("Int1D", "points"),
            ("Int1D", "radial_auto"),
            ("Int1D", "radial_low"),
            ("Int1D", "radial_high"),
            ("Int1D", "azim_auto"),
            ("Int1D", "azim_low"),
            ("Int1D", "azim_high"),
        } <= one_d_paths
        assert {
            ("Int2D", "axis"),
            ("Int2D", "radial_points"),
            ("Int2D", "azim_points"),
            ("Int2D", "radial_auto"),
            ("Int2D", "radial_low"),
            ("Int2D", "radial_high"),
            ("Int2D", "azim_auto"),
            ("Int2D", "azim_low"),
            ("Int2D", "azim_high"),
        } <= two_d_paths

        # Dialog-only params never render inline anywhere in the panel.
        advanced_only_paths = {
            ("Int1D", "unit"),
            ("Int2D", "unit"),
            ("Int1D", "method"),
            ("Int2D", "method"),
            ("Int1D", "correctSolidAngle"),
            ("Int2D", "correctSolidAngle"),
            ("Int1D", "apply_polarization"),
            ("Int2D", "apply_polarization"),
            ("Int1D", "polarization_factor"),
            ("Int2D", "polarization_factor"),
            ("Int1D", "dummy"),
            ("Int2D", "dummy"),
            ("Int1D", "delta_dummy"),
            ("Int2D", "delta_dummy"),
            ("Int1D", "chi_offset"),
            ("Int2D", "chi_offset"),
            ("Int1D", "safe"),
            ("Int2D", "safe"),
        }
        assert not (advanced_only_paths & all_rendered)

        # The dialog entry point survives: the Advanced action button is still
        # on the Reintegrate row.
        action_labels = {
            btn.text()
            for btn in widget.controls_v2.processing_card.body.findChildren(
                QtWidgets.QPushButton)
        }
        assert "Advanced" in action_labels
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_run_lock_disables_buttons_and_ignores_signals(
        qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    calls = []
    try:
        widget._on_controls_v2_field_changed(("GI", "Grazing"), True)
        widget._refresh_controls_v2_profile_now()
        more_button = _find_more_button(widget)
        source_energy_button = _find_source_energy_button(widget)
        assert more_button is not None
        assert source_energy_button is not None
        more_button.click()
        source_energy_button.click()
        qapp.processEvents()
        gi_popup = widget.controls_v2._gi_options_popup
        energy_popup = widget.controls_v2._source_energy_popup
        assert gi_popup is not None
        assert energy_popup is not None

        monkeypatch.setattr(
            widget,
            "_controls_v2_choose_source",
            lambda: calls.append("source"),
        )
        monkeypatch.setattr(
            widget.wrangler,
            "set_img_file",
            lambda: calls.append("browse"),
        )
        before = widget.integratorTree.get_gi_config()["sample_orientation"]

        widget._enter_run_state()

        locked = [
            button for button in widget.controls_v2.findChildren(QtWidgets.QAbstractButton)
            if button.objectName() in {
                "controlsV2ActionButton",
                "controlsV2MoreButton",
            }
        ]
        assert locked
        assert not any(button.isEnabled() for button in locked)
        assert not any(edit.isEnabled() for edit in gi_popup.findChildren(QtWidgets.QLineEdit))
        assert not any(
            button.isEnabled()
            for button in energy_popup.findChildren(QtWidgets.QAbstractButton)
        )

        widget._on_controls_v2_action(ControlAction.CHOOSE_SOURCE)
        widget._on_controls_v2_field_browse(("Signal", "File"))
        widget._on_controls_v2_field_changed(("GI", "sample_orientation"), "9")

        assert calls == []
        assert widget.integratorTree.get_gi_config()["sample_orientation"] == before
    finally:
        widget._exit_run_state(widget._new_projection_receipt())
        _reset_controls_v2_gi(widget)
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_combined_advanced_dialog_locked_during_run(
        qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget._show_integration_advanced()
        dlg = widget._integ_adv_combined_dlg
        assert dlg.isEnabled()

        widget._enter_run_state()
        assert not dlg.isEnabled()
        widget._show_integration_advanced()
        assert not dlg.isEnabled()

        widget._exit_run_state(widget._new_projection_receipt())
        assert dlg.isEnabled()
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_point_counts_refuse_below_minimum(qapp, monkeypatch):
    """§15.3 / §15.12-A.1 (updated from the retired clamp-to-one behavior): a
    below-minimum / negative point count is REFUSED at the checked idle commit —
    the permissive clamp is gone, so no carrier is mutated and the committed
    value stays at its valid default rather than silently becoming 1."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget._refresh_controls_v2_profile_now()
        before_1d = dict(widget.scan.bai_1d_args)
        before_2d = dict(widget.scan.bai_2d_args)

        widget._on_controls_v2_field_changed(("Int1D", "points"), "0")
        widget._on_controls_v2_field_changed(("Int1D", "points_oop"), "-4")
        widget._on_controls_v2_field_changed(("Int2D", "radial_points"), "0")
        widget._on_controls_v2_field_changed(("Int2D", "azim_points"), "-9")

        # Refused, not clamped: never 1, unchanged from the committed default.
        assert widget.scan.bai_1d_args["numpoints"] == before_1d["numpoints"] != 1
        # points_oop=-4 refused: npt_oop is neither clamped to 1 nor otherwise
        # mutated (it stays exactly as committed — absent, in the non-GI default).
        assert widget.scan.bai_1d_args.get("npt_oop") == before_1d.get("npt_oop")
        assert widget.scan.bai_1d_args.get("npt_oop") != 1
        assert widget.scan.bai_2d_args["npt_rad"] == before_2d["npt_rad"] != 1
        assert widget.scan.bai_2d_args["npt_azim"] == before_2d["npt_azim"] != 1
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_refresh_failure_warns_once(qapp, monkeypatch, caplog):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        monkeypatch.setattr(
            widget,
            "_controls_v2_state",
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        caplog.set_level("WARNING")

        widget._refresh_controls_v2_profile_now()
        widget._refresh_controls_v2_profile_now()

        warnings = [
            rec for rec in caplog.records
            if "Controls Panel V2 profile refresh failed" in rec.getMessage()
        ]
        assert len(warnings) == 1
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_native_int_advanced_rows_write_through_and_match_plan(
        qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        unit_1d = _combo_text(
            widget, "unit_1D", lambda text: text.startswith("2"))
        unit_2d = _combo_text(
            widget, "unit_2D", lambda text: text.startswith("2"))
        edits = (
            (("Int1D", "unit"), unit_1d),
            (("Int2D", "unit"), unit_2d),
            (("Int1D", "correctSolidAngle"), False),
            (("Int2D", "correctSolidAngle"), False),
            (("Int1D", "apply_polarization"), True),
            (("Int2D", "apply_polarization"), True),
            (("Int1D", "polarization_factor"), "0.73"),
            (("Int2D", "polarization_factor"), "0.81"),
            (("Int1D", "method"), "BBox"),
            (("Int2D", "method"), "BBox"),
            (("Int1D", "dummy"), "-2.0"),
            (("Int2D", "dummy"), "-3.0"),
            (("Int1D", "delta_dummy"), "0.25"),
            (("Int2D", "delta_dummy"), "0.5"),
            (("Int1D", "chi_offset"), "5.0"),
            (("Int2D", "chi_offset"), "12.0"),
            (("Int1D", "safe"), False),
            (("Int2D", "safe"), False),
        )
        _apply_v2_edits(widget, edits)

        a1 = widget.scan.bai_1d_args
        a2 = widget.scan.bai_2d_args
        assert a1["unit"] == "2th_deg"
        assert a2["unit"] == "2th_deg"
        assert a1["method"] == "BBox"
        assert a2["method"] == "BBox"
        assert a1["correctSolidAngle"] is False
        assert a2["correctSolidAngle"] is False
        assert a1["polarization_factor"] == pytest.approx(0.73)
        assert a2["polarization_factor"] == pytest.approx(0.81)
        assert a1["dummy"] == pytest.approx(-2.0)
        assert a2["dummy"] == pytest.approx(-3.0)
        assert a1["delta_dummy"] == pytest.approx(0.25)
        assert a2["delta_dummy"] == pytest.approx(0.5)
        assert a1["chi_offset"] == pytest.approx(5.0)
        assert a2["chi_offset"] == pytest.approx(12.0)
        assert a1["safe"] is False
        assert a2["safe"] is False
        assert _native_plan_snapshot(widget, commit_pending=False) == (
            _current_plan_snapshot(widget, commit_pending=False)
        )
    finally:
        widget.close()
        widget.deleteLater()


def test_advanced_polarization_none_survives_dialog_round_trip_s10(
        qapp, monkeypatch):
    """S10-1 GUARD (§10 fork class): ``polarization_factor=None`` (correction
    OFF — the args-side encoding V2 and ``_params_to_args`` both use) must
    survive the Advanced-dialog round trip.

    ``bai_*_args`` owns the fact; the 'Apply polarization factor' checkbox is a
    rendered VIEW of its None-encoding.  The regression: ``_args_to_params``
    rendered None as CHECKED, and because the V2 hydrate runs without the
    legacy disconnect bracket, the ``treeChangeBlocker`` exit fired
    ``process_change → sigUpdateArgs → get_args → _params_to_args`` — so
    merely OPENING the Advanced dialog (or editing any unrelated advanced
    field) silently rewrote ``polarization_factor`` from None to the tree's
    numeric value, enabling a correction the user never asked for and changing
    written data.
    """
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        integrator = widget.integratorTree
        # Pin the precondition through the PRODUCTION Controls seam (the panel's
        # fieldValueChanged handler), not by writing the retired display-scan
        # owner: R4-B made the Controls-owned intent the single writer of run
        # configuration, and the Advanced dialog hydrates from it.  Turning
        # polarization OFF here encodes ``polarization_factor=None`` on the
        # intent AND projects it to the scan.
        widget._on_controls_v2_field_changed(("Int1D", "apply_polarization"), False)
        widget._on_controls_v2_field_changed(("Int2D", "apply_polarization"), False)

        # The REAL dialog-open path: builds the dialog and hydrates the live
        # advanced trees from the (intent-owned) integration args.
        widget._show_integration_advanced()

        # The checkbox must render the None-encoding as unchecked...
        assert integrator.bai_1d_pars.child(
            "Apply polarization factor").value() is False
        assert integrator.bai_2d_pars.child(
            "Apply polarization factor").value() is False
        # ...and opening the dialog alone must not have rewritten the args.
        assert widget.scan.bai_1d_args["polarization_factor"] is None
        assert widget.scan.bai_2d_args["polarization_factor"] is None

        # Edit an UNRELATED advanced field through the real tree-change path
        # (sigTreeStateChanged → process_change → sigUpdateArgs → get_args).
        integrator.bai_1d_pars.child("dummy").setValue(-7.0)
        integrator.bai_2d_pars.child("dummy").setValue(-9.0)
        assert widget.scan.bai_1d_args["dummy"] == pytest.approx(-7.0)
        assert widget.scan.bai_2d_args["dummy"] == pytest.approx(-9.0)
        assert widget.scan.bai_1d_args["polarization_factor"] is None
        assert widget.scan.bai_2d_args["polarization_factor"] is None

        # A deliberate ON round-trips its numeric value unchanged.  Set it
        # through the Controls seam (single owner) so the dialog hydrate reflects
        # it, then re-hydrate the advanced trees.
        widget._on_controls_v2_field_changed(("Int1D", "polarization_factor"), 0.5)
        widget._controls_v2_hydrate_advanced_from_scan()
        assert integrator.bai_1d_pars.child(
            "Apply polarization factor").value() is True
        assert integrator.bai_1d_pars.child(
            "polarization_factor").value() == pytest.approx(0.5)
        integrator.bai_1d_pars.child("dummy").setValue(-8.0)
        assert widget.scan.bai_1d_args["polarization_factor"] == (
            pytest.approx(0.5))
        # The 2D dim was left OFF and must stay OFF through the same edit.
        assert widget.scan.bai_2d_args["polarization_factor"] is None
    finally:
        widget.close()
        widget.deleteLater()


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


def test_controls_panel_v2_backend_conflict_gates_run():
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


def test_controls_panel_v2_detector_status_uses_poni_summary(qapp):
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

    panel = ControlsPanelV2()
    panel.set_state(state)

    detector = next(
        card for card in panel.experiment_card.body.findChildren(SubsectionCard)
        if card.title.text() == "Detector"
    )
    assert detector.status.text() == "Eiger 1M · 200.4mm · fitted"


def test_controls_panel_v2_custom_title_keeps_specialized_gi_layout(qapp):
    state = build_control_panel_state(
        ControlState(),
        {("GI", "Grazing"): False},
    )
    panel = ControlsPanelV2(experiment_title="Configuration")
    try:
        panel.set_state(state)
        group = next(
            card for card in panel.experiment_card.findChildren(SubsectionCard)
            if card.title.text() == "Configuration"
        )
        assert [row.path for row in group.findChildren(SegmentedControl)] == [
            ("GI", "Grazing")
        ]
    finally:
        panel.close()
        panel.deleteLater()


def test_controls_panel_v2_section_ticks_and_source_synopsis(qapp):
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

    panel = ControlsPanelV2()
    panel.set_state(state)

    assert panel.source_card.status.text() == "Image Series"
    assert not panel.project_card.valid_marker.isHidden()
    assert not panel.source_card.valid_marker.isHidden()
    assert not panel.experiment_card.valid_marker.isHidden()


def test_controls_panel_v2_viewer_mode_shows_only_project(qapp):
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

    panel = ControlsPanelV2()
    panel.set_state(state)

    assert not panel.project_card.isHidden()
    assert panel.source_card.isHidden()
    assert panel.experiment_card.isHidden()
    assert panel.processing_card.isHidden()


def test_controls_panel_v2_requires_valid_project_before_setup_cards(qapp):
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

    panel = ControlsPanelV2()
    panel.set_state(ControlPanelRenderState(
        profile=profile,
        bound_controls=BoundControlState(fields=fields),
    ))

    assert not panel.project_card.isHidden()
    assert panel.source_card.isHidden()
    assert panel.experiment_card.isHidden()
    assert panel.processing_card.isHidden()


def test_make_mask_updates_mask_file_box(qapp, monkeypatch):
    """_on_mask_created writes the mask path to the wrangler param AND the V2
    Mask File box reflects it — the handler refreshes the panel, which re-reads
    the params and re-renders the row.  (Same refresh path the Calibrate→Poni
    autofill uses, so this covers both boxes.)"""
    monkeypatch.delenv("XDART_CONTROLS_PANEL_V2", raising=False)
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget._refresh_controls_v2_profile_now()
        focused = _find_form_row(widget, ("Project", "project_folder"))
        assert focused is not None
        focused.editor.setFocus()
        monkeypatch.setattr(widget.controls_v2, "focusWidget", lambda: focused.editor)

        mask_path = "/tmp/example-mask.edf"
        widget._on_mask_created(mask_path)
        # single source of truth: the wrangler param is updated
        assert widget.wrangler.parameters.child(
            "Signal", "mask_file").value() == mask_path
        # ...and the rendered Mask File box shows it
        row = _find_form_row(widget, ("Signal", "mask_file"))
        assert row is not None, "Mask File row not rendered"
        assert row.current_value() == mask_path
        assert widget._controls_v2_pending_editor is None
    finally:
        widget.close()
        widget.deleteLater()


def test_set_poni_field_updates_poni_box_with_focused_editor(
        qapp, monkeypatch, tmp_path):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    from xrd_tools.core.containers import PONI

    widget = staticWidget()
    try:
        poni_path = tmp_path / "calibrated.poni"
        PONI(
            dist=0.1794,
            poni1=0.0,
            poni2=0.0,
            detector="RayonixMx225",
            wavelength=0.7293e-10,
        ).to_poni_file(poni_path)

        widget._refresh_controls_v2_profile_now()
        focused = _find_form_row(widget, ("Project", "project_folder"))
        assert focused is not None
        focused.editor.setFocus()
        monkeypatch.setattr(widget.controls_v2, "focusWidget", lambda: focused.editor)

        widget._set_poni_field(str(poni_path))

        row = _find_form_row(widget, ("Signal", "poni_file"))
        assert row is not None, "Poni row not rendered"
        assert row.current_value() == str(poni_path)
        assert widget._controls_v2_pending_editor is None
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


def test_controls_panel_v2_source_energy_button_updates_native_preference(
        qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget._refresh_controls_v2_profile_now()

        btn = _find_source_energy_button(widget)
        assert btn is not None
        assert widget._controls_v2_energy_preference() == "poni"

        btn.click()
        popup = widget.controls_v2.findChild(QtWidgets.QWidget, "controlsV2EnergyPopup")
        assert popup is not None
        seg = popup.findChild(SegmentedControl)
        assert seg is not None
        assert seg.current_value() == "poni"
        metadata_button = next(
            button for button in seg.findChildren(QtWidgets.QPushButton)
            if button.text() == "Metadata"
        )

        metadata_button.click()

        assert widget._controls_v2_energy_preference() == "metadata"
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


def test_controls_panel_v2_integration_edits_update_native_state_immediately(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget._on_controls_v2_field_changed(("Int1D", "points"), "1234")
        assert widget.integratorTree.ui.npts_1D.text() != "1234"
        assert widget.scan.bai_1d_args["numpoints"] == 1234

        widget._on_controls_v2_field_changed(("Int2D", "radial_points"), "321")
        assert widget.integratorTree.ui.npts_radial_2D.text() != "321"
        assert widget.scan.bai_2d_args["npt_rad"] == 321

        widget._on_controls_v2_field_changed(("Int1D", "radial_auto"), False)
        assert widget.integratorTree.ui.radial_autoRange_1D.isChecked()
        assert widget.scan.bai_1d_args["radial_range"] is not None
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_gi_edits_update_native_state(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget._on_controls_v2_field_changed(("GI", "Grazing"), True)
        assert not widget.integratorTree.ui.gi_enable.isChecked()
        assert widget.scan.gi is True

        widget._on_controls_v2_field_changed(("GI", "sample_orientation"), "5")
        assert widget.integratorTree.ui.gi_sample_orientation.value() != 5
        assert widget._controls_v2_gi_config()["sample_orientation"] == 5
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

        # Clicking Grazing flips native scan state; the hidden legacy toggle is
        # no longer the V2 carrier.
        grazing_btn = next(btn for v, btn in seg._segments if v is True)
        grazing_btn.click()
        assert widget.scan.gi is True
        assert not widget.integratorTree.ui.gi_enable.isChecked()
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
        # disclosure, gated in controls_logic on the Grazing state).  With no
        # source loaded there is no real motor, so the θ-motor defaults to Manual
        # (not a phantom 'th'), which correctly reveals the manual θ-value input
        # inline alongside the motor selector.
        widget._on_controls_v2_field_changed(("GI", "Grazing"), True)
        widget._refresh_controls_v2_profile_now()
        assert _gi_detail_rows(widget) == {("GI", "th_motor"), ("GI", "th_val")}
        assert _find_more_button(widget) is not None
    finally:
        _reset_controls_v2_gi(widget)
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_gi_motor_never_shows_phantom_th(qapp, monkeypatch):
    """The Controls-V2 θ-motor row must never inject a phantom 'th': a fresh
    widget (no source) defaults to Manual, and a stale ``scan.incidence_motor ==
    'th'`` carried from the LiveScan default is dropped in favour of the shared
    default policy over the source's REAL motors.  'th' appears only when it is an
    actual motor of the loaded source.  (Regression: the V2 panel had its own
    'th'-first preference + a hardcoded ('Manual','th') fallback that bypassed the
    wrangler/integrator scoping.)"""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        # 1. Fresh widget, no motors -> Manual default, no phantom 'th' offered.
        choices = widget._controls_v2_native_int_choices().get(("GI", "th_motor"), ())
        assert "th" not in choices
        assert widget._controls_v2_default_gi_motor() == "Manual"

        # 2. A source offers real motors (none named 'th'); a stale incidence_motor
        #    of 'th' on the scan must NOT be shown — pick the preference motor.
        widget.integratorTree.set_gi_motor_options(["halpha", "detx", "dety"])
        widget.scan.incidence_motor = "th"       # legacy default carried on scan
        widget.scan.gi_config = {}
        cfg = widget._controls_v2_gi_config()
        assert cfg["incidence_motor"] == "halpha"   # named preference, not 'th'
        choices = widget._controls_v2_native_int_choices().get(("GI", "th_motor"), ())
        assert "th" not in choices
        assert set(choices) >= {"Manual", "halpha", "detx", "dety"}

        # 3. When 'th' IS a real motor of the source it is offered + preferred.
        widget.integratorTree.set_gi_motor_options(["th", "eta", "i0"])
        choices = widget._controls_v2_native_int_choices().get(("GI", "th_motor"), ())
        assert "th" in choices
    finally:
        _reset_controls_v2_gi(widget)
        widget.close()
        widget.deleteLater()


def test_dir_container_count_reports_direct_files_without_counting_frames(
    qapp, monkeypatch, tmp_path,
):
    """Directory Source status is name-only and direct-child-only."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    from tests.core.test_bluesky_nexus import _write_bluesky_nxwriter
    from xrd_tools.io import image as image_io

    d = tmp_path / "watch"
    d.mkdir()
    n_files = 12
    for i in range(n_files):
        _write_bluesky_nxwriter(d / f"scan_{i:05d}.nxs", n=2)

    monkeypatch.setattr(
        image_io, "count_frames",
        lambda path: pytest.fail(
            f"container directory listing opened a file: {path}"))

    widget = staticWidget()
    try:
        sig = widget.wrangler.parameters.child("Signal")
        sig.child("inp_type").setValue("Image Directory")
        sig.child("img_dir").setValue(str(d))
        sig.child("img_ext").setValue("nxs")

        assert _wait_until(
            qapp,
            lambda: widget._controls_v2_source_frame_count() == n_files,
        )
        assert widget._controls_v2_source_frame_count() == n_files
        assert widget._v2_source_count_is_files is True

        # A new direct child arrives: the displayed file count follows the
        # name/stat poll, still without counting container frames.
        _write_bluesky_nxwriter(d / f"scan_{n_files:05d}.nxs", n=5)
        widget._v2_frame_count_cache = None
        widget._controls_v2_source_widget.request_directory_poll()
        assert _wait_until(
            qapp,
            lambda: widget._controls_v2_source_frame_count() == n_files + 1,
        )
        assert widget._controls_v2_source_frame_count() == n_files + 1

        # The summary chip says files, never descriptor-frame totals.
        state = widget._controls_v2_state()
        assert state.frame_count_is_files is True
        from xdart.gui.tabs.static_scan.controls_logic import (
            build_control_profile,
        )
        profile = build_control_profile(state)
        text, _ready, _tooltip = staticWidget._controls_v2_run_summary(
            state, profile)
        if str(state.frame_count) in text:
            assert "file" in text
    finally:
        widget.close()
        widget.deleteLater()


def test_dir_container_count_small_dir_is_file_count(
    qapp, monkeypatch, tmp_path,
):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    from tests.core.test_bluesky_nexus import _write_bluesky_nxwriter

    d = tmp_path / "small"
    d.mkdir()
    for i in range(3):
        _write_bluesky_nxwriter(d / f"scan_{i:05d}.nxs", n=2)

    widget = staticWidget()
    try:
        sig = widget.wrangler.parameters.child("Signal")
        sig.child("inp_type").setValue("Image Directory")
        sig.child("img_dir").setValue(str(d))
        sig.child("img_ext").setValue("nxs")
        assert _wait_until(
            qapp,
            lambda: widget._controls_v2_source_frame_count() == 3,
        )
        assert widget._controls_v2_source_frame_count() == 3
        assert widget._v2_source_count_is_files is True
    finally:
        widget.close()
        widget.deleteLater()


def test_source_card_mounts_shared_widget_and_freezes_directory_intent(
    qapp, monkeypatch, tmp_path,
):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.scan_source_widget import ScanSourceWidget
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    from xrd_tools.sources.directory_index import DirectoryIndex
    from xrd_tools.sources.selection import DirectorySourceSpec

    (tmp_path / "scan_10.nxs").write_bytes(b"ten")
    (tmp_path / "scan_2.nxs").write_bytes(b"two")
    (tmp_path / "processed.nxs").write_bytes(b"processed")

    monkeypatch.setattr(
        DirectoryIndex,
        "probe_candidate",
        lambda *_args, **_kwargs: pytest.fail(
            "Source selection must not classify container contents"),
    )
    widget = staticWidget()
    try:
        source_widget = widget._controls_v2_source_widget
        assert isinstance(source_widget, ScanSourceWidget)
        assert source_widget._mode == "controls_source"
        embedded = widget.controls_v2.source_card.embedded_layout.itemAt(0)
        assert embedded.widget() is source_widget

        signal = widget.wrangler.parameters.child("Signal")
        signal.child("inp_type").setValue("Image Directory")
        signal.child("img_dir").setValue(str(tmp_path))
        signal.child("img_ext").setValue("nxs")
        assert _wait_until(
            qapp,
            lambda: widget._controls_v2_current_directory_observation()
            is not None,
        )
        observation = widget._controls_v2_current_directory_observation()
        assert len(observation.discovered_snapshot.candidates) == 3
        assert observation.ready_snapshot.candidates == ()
        assert observation.content_opens == 0
        assert widget._controls_v2_freeze_source_run_plan() is None
        spec = widget._controls_v2_freeze_source_spec()
        assert isinstance(spec, DirectorySourceSpec)
        assert spec.root == tmp_path
        assert spec.recursive is False
        assert spec.suffixes == (".nxs",)
        assert source_widget.directory_session is not None
    finally:
        widget.close()
        widget.deleteLater()


def test_h19_blank_container_directory_never_indexes_process_cwd(
    qapp, monkeypatch,
):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        signal = widget.wrangler.parameters.child("Signal")
        signal.child("inp_type").setValue("Image Directory")
        signal.child("img_ext").setValue("nxs")
        signal.child("img_dir").setValue("")

        assert widget._controls_v2_container_index_config() is None
        assert widget._controls_v2_source_widget.directory_session.configured is None
        assert widget._controls_v2_freeze_source_run_plan() is None
    finally:
        widget.close()
        widget.deleteLater()


def test_empty_container_directory_freezes_runnable_directory_intent(
    qapp, monkeypatch, tmp_path,
):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        signal = widget.wrangler.parameters.child("Signal")
        signal.child("inp_type").setValue("Image Directory")
        signal.child("img_dir").setValue(str(tmp_path))
        signal.child("img_ext").setValue("nxs")
        assert _wait_until(
            qapp,
            lambda: widget._controls_v2_current_directory_observation()
            is not None,
        )

        assert widget._controls_v2_freeze_source_run_plan() is None
        spec = widget._controls_v2_freeze_source_spec()
        assert spec.root == tmp_path
        assert spec.recursive is False
        assert spec.suffixes == (".nxs",)
    finally:
        widget.close()
        widget.deleteLater()


def test_run_boundary_propagates_directory_intent_without_catalog_owner(
    qapp, monkeypatch, tmp_path,
):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    started = []
    try:
        signal = widget.wrangler.parameters.child("Signal")
        signal.child("inp_type").setValue("Image Directory")
        signal.child("img_dir").setValue(str(tmp_path))
        signal.child("img_ext").setValue("nxs")
        assert _wait_until(
            qapp,
            lambda: widget._controls_v2_current_directory_observation()
            is not None,
        )
        widget.controls.liveButton.setChecked(True)
        monkeypatch.setattr(
            widget.wrangler.thread, "start", lambda: started.append(True))

        frozen = widget._prepare_controls_v2_run_configuration()
        assert frozen.source is not None
        # §8.1: stand in for the admission the production Start path performs;
        # the run-state owner refuses a wrangler run whose configuration is not
        # ONE admitted object across all five references.
        for _owner in (widget.wrangler, widget.wrangler.thread):
            for _name in ("run_configuration", "_admitted_run_configuration"):
                setattr(_owner, _name, frozen)
        widget.start_wrangler()

        assert started == [True]
        assert widget.wrangler.source_run_plan is None
        assert widget.wrangler.thread.source_run_plan is None
        assert widget.wrangler.source_index_session is None
        assert widget.wrangler.thread.source_index_session is None
        assert widget.wrangler.source_spec.root == tmp_path
        assert widget.wrangler.run_configuration is frozen
        assert widget.wrangler.thread.run_configuration is frozen
        assert "source_spec" not in vars(widget.wrangler.thread)
        assert widget.wrangler.source_frame_count_snapshot == {}
        assert widget.wrangler.thread.source_frame_count_snapshot == {}
        assert widget.wrangler.source_pending_count == 0
        assert widget.wrangler.thread.source_pending_count == 0
        assert widget.wrangler.img_file == ""
        assert widget.wrangler.thread.img_file == ""

        widget._clear_controls_v2_run_source_authority()
        assert widget.wrangler.source_run_plan is None
        assert widget.wrangler.source_spec is None
        assert widget.wrangler.source_index_session is None
        assert widget.wrangler.source_frame_count_snapshot == {}
        assert widget.wrangler.source_pending_count == 0
        assert widget.wrangler.thread.source_run_plan is None
        assert "source_spec" not in vars(widget.wrangler.thread)
        assert widget.wrangler.thread.source_index_session is None
        assert widget.wrangler.thread.source_frame_count_snapshot == {}
        assert widget.wrangler.thread.source_pending_count == 0
    finally:
        widget._exit_run_state(widget._new_projection_receipt())
        widget.close()
        widget.deleteLater()


def test_h19_frame_count_handoff_is_a_value_copy(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget._v2_container_final_count_memo = {
            "/raw/scan.nxs": ((123, 456), 6)}

        frozen = widget._controls_v2_freeze_container_frame_counts()
        widget._v2_container_final_count_memo[
            "/raw/scan.nxs"] = ((999, 999), 12)

        assert frozen == {"/raw/scan.nxs": ((123, 456), 6)}
    finally:
        widget.close()
        widget.deleteLater()


def test_h19_delayed_count_signal_never_pairs_old_count_with_new_stamp(
        qapp, monkeypatch, tmp_path):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    path = tmp_path / "scan.nxs"
    path.write_bytes(b"old")
    old_stamp = staticWidget._v2_file_stamp(path)
    # Model growth between worker emission and GUI-thread delivery.
    path.write_bytes(b"new container bytes")
    new_stamp = staticWidget._v2_file_stamp(path)
    assert new_stamp != old_stamp

    widget = staticWidget()
    try:
        widget._on_container_count_landed(
            str(path), 3, old_stamp, True)

        assert widget._v2_container_count_memo[str(path)] == (old_stamp, 3)
        assert widget._controls_v2_freeze_container_frame_counts() == {
            str(path): (old_stamp, 3)}
        assert widget._controls_v2_freeze_container_frame_counts()[
            str(path)][0] != new_stamp

        # The legacy click-to-count signal remains display-only.
        widget._v2_container_final_count_memo.clear()
        widget._on_container_count_landed(str(path), 9)
        assert widget._v2_container_count_memo[str(path)] == (new_stamp, 9)
        assert widget._controls_v2_freeze_container_frame_counts() == {}
    finally:
        widget.close()
        widget.deleteLater()


def test_h19_zero_ready_observation_never_falls_back_to_legacy_sidecar(
    qapp, monkeypatch, tmp_path,
):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    (tmp_path / "scan_data_000001.h5").write_bytes(b"sidecar")
    widget = staticWidget()
    try:
        signal = widget.wrangler.parameters.child("Signal")
        signal.child("inp_type").setValue("Image Directory")
        signal.child("img_dir").setValue(str(tmp_path))
        signal.child("img_ext").setValue("h5")
        assert _wait_until(
            qapp,
            lambda: widget._controls_v2_current_directory_observation()
            is not None,
        )
        observation = widget._controls_v2_current_directory_observation()
        assert observation.ready_snapshot.candidates == ()
        assert widget._controls_v2_first_metadata_file() == ""

        marker = object()
        widget._controls_v2_source_energy_cache = marker
        widget._controls_v2_metadata_probe_cache = marker
        widget._on_controls_v2_directory_observation(observation)
        assert widget._controls_v2_source_energy_cache is marker
        assert widget._controls_v2_metadata_probe_cache is marker
    finally:
        widget.close()
        widget.deleteLater()


def test_h19_close_stops_wrangler_before_shutting_directory_session(
    qapp, monkeypatch,
):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    events = []
    try:
        monkeypatch.setattr(
            widget, "_stop_wrangler_thread_on_close",
            lambda: events.append("wrangler"),
        )
        monkeypatch.setattr(
            widget._controls_v2_source_widget, "shutdown_probe_worker",
            lambda: events.append("session"),
        )
        widget.close()
        assert events[:2] == ["wrangler", "session"]
    finally:
        widget.deleteLater()


def test_h19_legacy_panel_never_requires_v2_directory_authority(
    qapp, monkeypatch, tmp_path,
):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "0")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        signal = widget.wrangler.parameters.child("Signal")
        signal.child("inp_type").setValue("Image Directory")
        signal.child("img_dir").setValue(str(tmp_path))
        signal.child("img_ext").setValue("nxs")
        assert widget._controls_v2_source_widget is None
        assert widget._controls_v2_container_index_config() is None
    finally:
        widget.close()
        widget.deleteLater()


def test_source_directory_config_roundtrip_rebinds_lazy_intent(
    qapp, monkeypatch, tmp_path,
):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    from xrd_tools.sources.directory_index import DirectoryIndex

    source_a = tmp_path / "source-a"
    source_b = tmp_path / "source-b"
    source_a.mkdir()
    source_b.mkdir()
    (source_a / "keep_scan_1.nxs").write_bytes(b"one")

    monkeypatch.setattr(
        DirectoryIndex,
        "probe_candidate",
        lambda *_args, **_kwargs: pytest.fail(
            "config restore must not inspect container contents"),
    )
    config_path = tmp_path / "source-session.json"
    widget = staticWidget()
    try:
        signal = widget.wrangler.parameters.child("Signal")
        signal.child("inp_type").setValue("Image Directory")
        signal.child("img_dir").setValue(str(source_a))
        signal.child("include_subdir").setValue(True)
        signal.child("img_ext").setValue("nxs")
        signal.child("Filter").setValue("keep -bad")
        signal.child("meta_ext").setValue("auto")
        assert _wait_until(
            qapp,
            lambda: widget._controls_v2_current_directory_observation()
            is not None,
        )
        widget.h5viewer.defaultWidget.save_defaults(fname=str(config_path))

        signal.child("inp_type").setValue("Image Series")
        signal.child("img_dir").setValue(str(source_b))
        signal.child("include_subdir").setValue(False)
        signal.child("img_ext").setValue("tif")
        signal.child("Filter").setValue("")
        signal.child("meta_ext").setValue("none")

        widget.h5viewer.defaultWidget.load_defaults(fname=str(config_path))
        assert _wait_until(
            qapp,
            lambda: widget._controls_v2_current_directory_observation()
            is not None,
        )

        restored = widget.wrangler.parameters.child("Signal")
        assert restored.child("inp_type").value() == "Image Directory"
        assert restored.child("img_dir").value() == str(source_a)
        assert restored.child("include_subdir").value() is True
        assert restored.child("img_ext").value() == "nxs"
        assert restored.child("Filter").value() == "keep -bad"
        assert restored.child("meta_ext").value() == "auto"

        source_widget = widget._controls_v2_source_widget
        session_config = source_widget.directory_session.configured
        assert session_config.root == source_a
        assert session_config.recursive is False
        assert session_config.name_filter == "keep -bad"
        assert session_config.suffixes == (".nxs",)
        assert source_widget.directory_subdirs_lazy is True
        assert widget._controls_v2_freeze_source_run_plan() is None
        spec = widget._controls_v2_freeze_source_spec()
        assert spec.root == source_a
        assert spec.recursive is True
        assert spec.name_filter == "keep -bad"
        assert spec.suffixes == (".nxs",)
    finally:
        widget.close()
        widget.deleteLater()


def test_gi_enable_repicks_default_over_leftover_manual(qapp, monkeypatch):
    """Live-found 2026-07-12 (LaB6, halpha listed but Manual selected): session
    restore writes the θ-motor VALUE programmatically without ever running
    set_gi_motor_options, so the default policy never fired — and enabling
    Grazing didn't re-evaluate it.  Enabling GI must apply the shared policy
    over the current choices when 'Manual' is a leftover; a DELIBERATE user
    Manual stays sticky."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        it = widget.integratorTree
        it.set_gi_motor_options(["halpha", "detx", "dety", "hx"])
        # _load_from_session's exact write: programmatic index set, no user pick.
        it.ui.gi_motor.setCurrentIndex(it.ui.gi_motor.findText("Manual"))
        widget.scan.incidence_motor = "0.1"   # stale numeric-theta carry
        widget.scan.gi_config = {}

        # The V2 Grazing pill: cfg, scan AND the integrator combo all re-pick.
        widget._on_controls_v2_field_changed(("GI", "Grazing"), True)
        cfg = widget._controls_v2_gi_config()
        assert cfg["incidence_motor"] == "halpha"
        assert widget.scan.gi_config["incidence_motor"] == "halpha"
        assert it.ui.gi_motor.currentText() == "halpha"  # surfaces stay equal

        # The integrator checkbox path in isolation re-picks too.
        it.ui.gi_enable.setChecked(False)
        it.ui.gi_motor.setCurrentIndex(it.ui.gi_motor.findText("Manual"))
        it.ui.gi_enable.setChecked(True)
        assert it.ui.gi_motor.currentText() == "halpha"

        # A DELIBERATE user Manual survives the toggle (F3 sticky rule).
        it.ui.gi_enable.setChecked(False)
        it.ui.gi_motor.setCurrentIndex(it.ui.gi_motor.findText("Manual"))
        it._on_gi_motor_user_pick()
        it.ui.gi_enable.setChecked(True)
        assert it.ui.gi_motor.currentText() == "Manual"
    finally:
        _reset_controls_v2_gi(widget)
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_gi_motor_is_a_pure_view_of_the_single_source(qapp, monkeypatch):
    """GUARD against the phantom-'th' bug CLASS (Controls-V2 field forking its
    source): the θ-motor row must be a pure VIEW of the single source of truth —
    the integrator ``gi_motor`` combo for the CHOICES and the shared
    ``pick_default_gi_motor`` for the DEFAULT — never a re-implementation.  For
    every motor list the V2 rendered choices equal the integrator combo's items
    and the V2 default equals the shared policy over those motors.  The phantom
    'th' was exactly a violation of this invariant (V2 had its own 'th'-first
    preference + hardcoded fallbacks); a future re-fork of any V2 native field's
    source of truth fails here.  (Design: design_controls_panel_v2_jun2026.md
    §10 — 'One source of truth per field — never a hand-set label / forked state'.)"""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    from xdart.gui.tabs.static_scan.gi_motor_defaults import pick_default_gi_motor

    widget = staticWidget()
    try:
        it = widget.integratorTree
        for motors in (["halpha", "detx", "dety"], ["th", "eta", "i0"],
                       ["samphi", "detx"], ["detx", "dety"], ["gonth", "hy"]):
            it.set_gi_motor_options(motors)
            # CHOICES: V2 renders EXACTLY the integrator combo's items (pure view).
            v2_choices = list(
                widget._controls_v2_native_int_choices().get(("GI", "th_motor"), ()))
            combo_items = [it.ui.gi_motor.itemText(i)
                           for i in range(it.ui.gi_motor.count())]
            assert v2_choices == combo_items, (motors, v2_choices, combo_items)
            # DEFAULT: V2 uses the shared policy over the real motors (no fork).
            assert widget._controls_v2_default_gi_motor() == pick_default_gi_motor(
                motors), motors
    finally:
        _reset_controls_v2_gi(widget)
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_every_combo_value_is_within_its_own_choices(qapp, monkeypatch):
    """GUARD generalising the phantom-'th' invariant to EVERY Controls-V2 combo
    field: a combo's rendered VALUE must always be a member of its rendered
    CHOICES.  Phantom-'th' was exactly a violation — the θ-motor field showed
    'th' while its choice list was {Manual, <real motors>} (value ∉ choices),
    because V2 forked its value/default derivation away from its choice
    derivation.  Value and choices flow through DIFFERENT code paths
    (_controls_v2_native_int_values vs _controls_v2_native_int_choices), so a
    future re-fork of ANY combo field — unit, axis, method, th_motor — where the
    two paths disagree fails here.  Held across GI on/off, several θ-motor lists,
    and both integration dims.  (Design: design_controls_panel_v2_jun2026.md §10
    — 'One source of truth per field — never a hand-set label / forked state'.)"""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        it = widget.integratorTree
        for gi in (False, True):
            widget._on_controls_v2_field_changed(("GI", "Grazing"), gi)
            for motors in (["halpha", "detx"], ["th", "eta", "i0"],
                           ["detx", "dety"]):
                it.set_gi_motor_options(motors)
                values = widget._controls_v2_native_int_values()
                choices = widget._controls_v2_native_int_choices()
                # Every rendered combo (a path that has a choice list) whose value
                # is currently shown must render a value drawn from that list.
                for path, opts in choices.items():
                    if path not in values:
                        continue  # field is mode-gated off right now (e.g. gi_mode)
                    assert values[path] in opts, (gi, motors, path, values[path], opts)
    finally:
        _reset_controls_v2_gi(widget)
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_unit_choices_mirror_the_legacy_unit_combos(qapp, monkeypatch):
    """GUARD: the V2 1D/2D UNIT fields are a pure VIEW of the legacy integrator
    unit combos (unit_1D / unit_2D) — same items, same order — never a forked
    hardcoded list.  Extends the phantom-'th' parity guard to the unit fields
    per the H21/Phase-8 scoping (the safe in-sequence increment toward retiring
    the duplicate state).

    AXIS is deliberately excluded: V2 renders the real GI/standard axis labels
    while the legacy axis combos are degenerate placeholders (axis1D shows only
    'Radial'), so V2 is authoritative there, not a view — axis is covered
    instead by the value-in-choices guard above."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        ui = widget.integratorTree.ui
        for gi in (False, True):
            widget._on_controls_v2_field_changed(("GI", "Grazing"), gi)
            choices = widget._controls_v2_native_int_choices()
            for path, name in ((("Int1D", "unit"), "unit_1D"),
                               (("Int2D", "unit"), "unit_2D")):
                combo = getattr(ui, name)
                legacy = tuple(combo.itemText(i) for i in range(combo.count()))
                assert tuple(choices[path]) == legacy, (gi, path, choices[path], legacy)
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

        # Forced rebuilds do not re-emit the retired legacy GI signal.
        for _ in range(5):
            widget._refresh_controls_v2_profile_now()
        assert emissions == []

        # A genuine V2 toggle updates native state without depending on the
        # legacy integrator signal.
        widget._on_controls_v2_field_changed(("GI", "Grazing"), False)
        assert emissions == []
        widget._on_controls_v2_field_changed(("GI", "Grazing"), False)
        assert emissions == []
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
        widget._controls_v2_last_schema_signature = None
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


def test_controls_panel_v2_active_run_refreshes_once_after_run(qapp, monkeypatch):
    """Active non-viewer runs should not rebuild V2 controls every progress tick.

    The profile is refreshed once the run exits, so the final state still
    appears without adding GUI churn during live/append reductions.
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
        widget.controls.batchButton.setChecked(False)
        widget._run_active = True

        widget._refresh_controls_v2_profile_now()

        assert calls == []
        assert widget._controls_v2_batch_refresh_deferred is True

        widget._exit_run_state(widget._new_projection_receipt())

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


def test_controls_panel_v2_metadata_energy_preference_becomes_authoritative(
        qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget.scan._persisted_wavelength_m = 1.0e-10
        widget.scan.source_energy_eV = 15_000.0

        widget._on_controls_v2_field_changed(
            ("Source", "energy_preference"), "metadata")
        state = widget._controls_v2_state()
        render_state = build_control_panel_state(
            state,
            widget._controls_v2_field_values(),
            widget._controls_v2_field_choices(),
        )
        energy = render_state.profile.fields[FieldId.BEAM_ENERGY]

        assert widget._controls_v2_energy_preference() == "metadata"
        assert state.geom.calibration_energy_eV == pytest.approx(15_000.0)
        assert state.geom.source_energy_eV is None
        assert energy.status is StatusKind.OK
        assert "disagree" not in energy.reason.lower()
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_state_summarizes_cached_poni_detector(
        qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    from xrd_tools.core.containers import PONI

    widget = staticWidget()
    try:
        widget.scan._cached_poni = PONI(
            dist=0.2004,
            poni1=0.0,
            poni2=0.0,
            detector="Eiger 1M",
        )

        state = widget._controls_v2_state()

        assert state.detector_summary == "Eiger 1M · 200.4mm · fitted"
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_not_ready_disables_run_row(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget.wrangler.project_folder = ""
        widget._refresh_controls_v2_profile(immediate=True)

        assert widget.controls.readinessDot.property("ready") is False
        assert "Choose a project folder" in widget.controls.readinessLabel.text()
        assert widget.controls.actionRow.isVisible()
        assert widget.controls.actionRow.isEnabled() is False
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_readiness_summary_shows_only_top_blocker():
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    state = ControlState(
        processing_mode="Int 2D",
        run_target=RunTarget.LOADED_SCAN,
    )
    profile = ControlProfile(
        processing_page=ProcessingPage.INT_2D,
        run_enabled=False,
        run_blockers=(
            "Choose a project folder.",
            "Run needs a frame source - use Reintegrate for the loaded scan.",
        ),
    )

    summary, ready, tooltip = staticWidget._controls_v2_run_summary(
        state, profile)

    assert ready is False
    assert summary == "Needs setup · Choose a project folder · Int 2D"
    assert "Run needs a frame source" not in summary
    assert "Run needs a frame source" in tooltip


def test_run_readiness_label_elides_without_widening_controls(qapp):
    from xdart.gui.tabs.static_scan.ui.static_controls import StaticControls

    controls = StaticControls()
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


def test_controls_panel_v2_cached_scan_poni_satisfies_calibration(
        qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    from xrd_tools.core.containers import PONI
    from xrd_tools.core.energy import wavelength_m_to_energy_eV

    widget = staticWidget()
    try:
        wavelength_m = 0.7293e-10
        widget.scan._cached_integrator = object()
        widget.scan._cached_poni = PONI(
            dist=0.1794,
            poni1=0.0,
            poni2=0.0,
            detector="RayonixMx225",
            wavelength=wavelength_m,
        )
        widget.wrangler.poni = None
        widget.wrangler.poni_file = ""

        state = widget._controls_v2_state()

        assert state.detector_summary == "RayonixMx225 · 179.4mm · fitted"
        assert state.geom.calibrated is True
        assert state.geom.calibration_energy_eV == pytest.approx(
            wavelength_m_to_energy_eV(wavelength_m),
            rel=1e-12,
        )
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_energy_values_use_poni_without_scan():
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    from xrd_tools.core.containers import PONI
    from xrd_tools.core.energy import wavelength_m_to_energy_eV

    wavelength_m = 1.0e-10
    host = SimpleNamespace(
        scan=None,
        _controls_v2_current_poni=lambda: PONI(
            dist=0.1,
            poni1=0.0,
            poni2=0.0,
            detector="RayonixMx225",
            wavelength=wavelength_m,
        ),
        _controls_v2_calibration_energy_eV=(
            staticWidget._controls_v2_calibration_energy_eV
        ),
        _controls_v2_source_energy_eV=staticWidget._controls_v2_source_energy_eV,
        _controls_v2_metadata_energy_eV=lambda scan: None,
        _controls_v2_energy_preference=lambda: "poni",
    )
    host._controls_v2_energy_values = MethodType(
        staticWidget._controls_v2_energy_values, host)

    calibration_energy_eV, source_energy_eV = host._controls_v2_energy_values()

    assert calibration_energy_eV == pytest.approx(
        wavelength_m_to_energy_eV(wavelength_m), rel=1e-12)
    assert source_energy_eV is None


def test_controls_panel_v2_calibration_energy_prefers_poni_over_scan_wavelength():
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    from xrd_tools.core.containers import PONI
    from xrd_tools.core.energy import wavelength_m_to_energy_eV

    poni_wavelength_m = 0.7293e-10
    scan = SimpleNamespace(_persisted_wavelength_m=1.0e-10)
    poni = PONI(
        dist=0.1794,
        poni1=0.0,
        poni2=0.0,
        detector="RayonixMx225",
        wavelength=poni_wavelength_m,
    )

    assert staticWidget._controls_v2_calibration_energy_eV(
        scan, poni=poni) == pytest.approx(
            wavelength_m_to_energy_eV(poni_wavelength_m),
            rel=1e-12,
        )


def test_controls_panel_v2_source_count_uses_container_frame_count(
        monkeypatch, tmp_path):
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    from xrd_tools.io import image as image_io

    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"")
    monkeypatch.setattr(image_io, "count_frames", lambda path: 50)

    count = staticWidget._controls_v2_count_source_frames(
        source_type="Image Series",
        img_file=str(master),
        img_dir="",
        img_ext="h5",
        include_subdir=False,
        file_filter="",
    )

    assert count == 50


def test_controls_panel_v2_source_count_reports_entire_image_series(
        monkeypatch, tmp_path):
    """The Source card describes the series, not the selected suffix onward."""
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    from xrd_tools.io import image as image_io

    paths = []
    for idx in range(1, 6):
        path = tmp_path / f"scan_{idx:04d}.tif"
        path.write_bytes(b"")
        paths.append(path)
    monkeypatch.setattr(
        image_io,
        "count_frames",
        lambda path: pytest.fail("single-image series count opened a file"),
    )

    count = staticWidget._controls_v2_count_source_frames(
        source_type="Image Series",
        img_file=str(paths[3]),
        img_dir="",
        img_ext="tif",
        include_subdir=False,
        file_filter="",
    )

    assert count == 5


def test_controls_panel_v2_source_count_directory_masters_is_file_count(
        monkeypatch, tmp_path):
    """DIR-2: a directory of masters reports the FILE count (2) — count_frames
    is never called for a container directory."""
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    from xrd_tools.io import image as image_io

    first = tmp_path / "a_master.h5"
    second = tmp_path / "b_master.h5"
    ignored = tmp_path / "not_a_container.h5"
    for path in (first, second, ignored):
        path.write_bytes(b"")

    monkeypatch.setattr(
        image_io, "count_frames",
        lambda path: pytest.fail("container directory count opened a file"))

    count = staticWidget._controls_v2_count_source_frames(
        source_type="Image Directory",
        img_file="",
        img_dir=str(tmp_path),
        img_ext="h5",
        include_subdir=False,
        file_filter="",
    )

    assert count == 2


def test_controls_panel_v2_source_count_counts_single_image_files(
        monkeypatch, tmp_path):
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    from xrd_tools.io import image as image_io

    for idx in range(3):
        (tmp_path / f"scan_{idx:04d}.tif").write_bytes(b"")
    monkeypatch.setattr(
        image_io,
        "count_frames",
        lambda path: pytest.fail("single-image directory count opened a file"),
    )

    count = staticWidget._controls_v2_count_source_frames(
        source_type="Image Directory",
        img_file="",
        img_dir=str(tmp_path),
        img_ext="tif",
        include_subdir=False,
        file_filter="",
    )

    assert count == 3


def test_controls_panel_v2_source_frame_count_is_unknown_in_live_mode(tmp_path):
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    raw_path = tmp_path / "scan_0001.tif"
    raw_path.write_bytes(b"")

    values = {
        ("Signal", "inp_type"): "Image Series",
        ("Signal", "File"): str(raw_path),
        ("Signal", "img_dir"): "",
        ("Signal", "img_ext"): "tif",
        ("Signal", "include_subdir"): False,
        ("Signal", "Filter"): "",
    }
    host = SimpleNamespace(
        _v2_frame_count_cache=None,
        _controls_v2_param_value=lambda path, default="": values.get(tuple(path), default),
        _controls_v2_live_source_active=lambda: True,
    )
    host._controls_v2_source_frame_count = MethodType(
        staticWidget._controls_v2_source_frame_count, host)

    assert host._controls_v2_source_frame_count() is None


def test_controls_panel_v2_frame_count_cache_tracks_directory_mtime(tmp_path):
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    values = {
        ("Signal", "inp_type"): "Image Directory",
        ("Signal", "File"): "",
        ("Signal", "img_dir"): str(tmp_path),
        ("NeXus File", "nexus_file"): "",
        ("Signal", "img_ext"): "tif",
        ("Signal", "include_subdir"): False,
        ("Signal", "Filter"): "",
    }
    host = SimpleNamespace(
        _v2_frame_count_cache=None,
        _controls_v2_param_value=lambda path, default="": values.get(tuple(path), default),
        _controls_v2_live_source_active=lambda: False,
        _controls_v2_source_cache_stamp=staticWidget._controls_v2_source_cache_stamp,
        _controls_v2_count_source_frames=staticWidget._controls_v2_count_source_frames,
    )
    host._controls_v2_source_frame_count = MethodType(
        staticWidget._controls_v2_source_frame_count, host)

    assert host._controls_v2_source_frame_count() == 0
    (tmp_path / "scan_0001.tif").write_bytes(b"")
    stamp = tmp_path.stat().st_mtime_ns + 1_000_000
    os.utime(tmp_path, ns=(stamp, stamp))

    assert host._controls_v2_source_frame_count() == 1


def test_controls_panel_v2_first_metadata_file_uses_resolved_preview_no_walk(tmp_path):
    """§15.12-D.2/D.3: the GUI profile takes the metadata authority ONLY from the
    wrangler's already-resolved direct-child preview — never an independent
    directory walk.  A matching file physically present in the directory does NOT
    surface until the wrangler has resolved it as a preview."""
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    values = {
        ("Signal", "inp_type"): "Image Directory",
        ("Signal", "img_dir"): str(tmp_path),
        ("Signal", "img_ext"): "tif",
        ("Signal", "Filter"): "",
        ("Signal", "include_subdir"): False,
    }
    wrangler = SimpleNamespace(img_file="", _directory_metadata_preview_path="")
    host = SimpleNamespace(
        wrangler=wrangler,
        _controls_v2_metadata_probe_cache=None,
        _controls_v2_param_value=lambda path, default="": values.get(tuple(path), default),
        _controls_v2_source_cache_stamp=staticWidget._controls_v2_source_cache_stamp,
    )
    host._controls_v2_first_metadata_file = MethodType(
        staticWidget._controls_v2_first_metadata_file, host)

    # No resolved preview yet -> empty, even though a matching file exists.
    raw = tmp_path / "scan_0001.tif"
    raw.write_bytes(b"")
    assert host._controls_v2_first_metadata_file() == ""

    # A resolved direct-child preview from the wrangler IS used (no walk).
    wrangler._directory_metadata_preview_path = str(raw)
    assert host._controls_v2_first_metadata_file() == str(raw)

    # A resolved single-image file takes precedence.
    wrangler.img_file = str(tmp_path / "chosen.tif")
    assert host._controls_v2_first_metadata_file() == str(tmp_path / "chosen.tif")


def test_controls_panel_v2_source_caps_delegate_to_headless_readiness(
        qapp, tmp_path):
    """H18: ``_controls_v2_source_caps`` no longer constructs SourceCaps inline
    (the ``has_frames = has_raw = raw_reachable = source_ready`` collapse is
    GONE) — it DELEGATES the tri-fields to
    ``xrd_tools.sources.readiness.describe_source_readiness`` over the real
    configured source path.  Merge policy pinned here:

    * non-live metadata family = panel truth OR source-served truth;
    * has_energy = panel-only (BEAM_ENERGY is a run-required field);
    * LIVE sources ride the core escape hatch for the tri-fields while their
      optimistic metadata/geometry claims stay ADVISORY (panel truth only).
    """
    import h5py
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    from xrd_tools.core.scan import SourceKind, SourceSpec
    from xrd_tools.sources.readiness import describe_source_readiness

    master = tmp_path / "scan_master.h5"
    raw = np.arange(2 * 8 * 8, dtype=np.uint32).reshape(2, 8, 8)
    with h5py.File(master, "w") as f:
        f.create_dataset("entry/data/data", data=raw)

    headless = describe_source_readiness(str(master))
    assert headless.raw_reachable is True     # real file, real frame-0 probe
    assert headless.has_metadata is False     # bare detector master

    widget = staticWidget()
    try:
        gui, ready = widget._controls_v2_source_caps(
            source_label=str(master),
            frame_count=2,
            live_unknown=False,
            has_metadata=False,
            has_motors=False,
            has_energy=False,
            has_geometry=False,
            has_psi_metadata=False,
        )
        assert ready is True
        assert gui == headless                # the delegation, field for field

        # Panel truth the source cannot serve (hydrated metadata, PONI
        # geometry, resolved energy) is OR-composed on top of the delegated
        # answer for non-live sources.
        gui_panel, _ready = widget._controls_v2_source_caps(
            source_label=str(master),
            frame_count=2,
            live_unknown=False,
            has_metadata=True,
            has_motors=True,
            has_energy=True,
            has_geometry=True,
            has_psi_metadata=True,
        )
        assert gui_panel.has_metadata is True and headless.has_metadata is False
        assert gui_panel.has_motors is True
        assert gui_panel.has_energy is True and headless.has_energy is False
        assert gui_panel.has_geometry is True
        assert (gui_panel.has_frames, gui_panel.has_raw,
                gui_panel.raw_reachable) == (True, True, True)

        # LIVE: the tri-fields come from the core true-live escape hatch, and
        # the LiveFrameSource's optimistic metadata/geometry claims are
        # advisory — the panel's actual state (here: nothing) wins, so a live
        # run without real metadata/calibration stays gated.
        live_headless = describe_source_readiness(
            SourceSpec(str(master), SourceKind.LIVE))
        assert live_headless.has_metadata is True     # the optimistic claim
        assert live_headless.has_geometry is True
        gui_live, ready_live = widget._controls_v2_source_caps(
            source_label=str(master),
            frame_count=0,
            live_unknown=True,
            has_metadata=False,
            has_motors=False,
            has_energy=False,
            has_geometry=False,
            has_psi_metadata=False,
        )
        assert ready_live is True
        assert (gui_live.has_frames, gui_live.has_raw,
                gui_live.raw_reachable) == (True, True, True)
        assert gui_live.has_metadata is False         # advisory claim overlaid
        assert gui_live.has_geometry is False
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_source_label_uses_configured_raw_source(
        qapp, monkeypatch, tmp_path):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    from xrd_tools.io import image as image_io

    widget = staticWidget()
    try:
        monkeypatch.setattr(image_io, "count_frames", lambda path: 1)
        raw_path = tmp_path / "raw" / "scan_0001.tif"
        raw_path.parent.mkdir()
        raw_path.write_bytes(b"")
        widget.fname = str(tmp_path / "processed" / "output.nxs")
        widget.wrangler.fname = widget.fname
        widget._controls_v2_param(("Signal", "File")).setValue(str(raw_path))

        state = widget._controls_v2_state()

        assert state.source_label == str(raw_path)
        assert state.source_caps.has_raw is True
        assert state.source_caps.has_frames is True
        assert state.frame_count == 1
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_frame_count_refresh_updates_without_rebuild(
        qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    count = {"frames": 1}
    try:
        monkeypatch.setattr(
            widget, "_controls_v2_source_label", lambda: "/tmp/raw_scan")
        monkeypatch.setattr(
            widget,
            "_controls_v2_source_frame_count",
            lambda: count["frames"],
        )
        widget._refresh_controls_v2_profile_now()
        assert "1 frames" in widget.controls_v2.source_card.status.text()

        rebuilds = []
        original_set_state = widget.controls_v2.set_state

        def _record_rebuild(state):
            rebuilds.append(state)
            original_set_state(state)

        monkeypatch.setattr(widget.controls_v2, "set_state", _record_rebuild)
        count["frames"] = 2

        widget._refresh_controls_v2_profile_now()

        assert rebuilds == []
        assert "2 frames" in widget.controls_v2.source_card.status.text()
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_loaded_scan_does_not_populate_source_or_run(
        qapp, monkeypatch, tmp_path):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    from xrd_tools.core.containers import PONI

    widget = staticWidget()
    try:
        project = tmp_path / "project"
        project.mkdir()
        processed = project / "processed.nxs"
        processed.write_bytes(b"")
        poni_path = project / "cal.poni"
        PONI(
            dist=0.1794,
            poni1=0.0,
            poni2=0.0,
            detector="RayonixMx225",
            wavelength=0.7293e-10,
        ).to_poni_file(poni_path)

        widget._controls_v2_param(("Project", "project_folder")).setValue(str(project))
        widget.wrangler.project_folder = str(project)
        widget._controls_v2_param(("Project", "h5_dir")).setValue(
            str(project / "xdart_processed_data")
        )
        widget.wrangler.h5_dir = str(project / "xdart_processed_data")
        widget._controls_v2_param(("Signal", "File")).setValue("")
        widget.wrangler.img_file = ""
        widget.scan.data_file = str(processed)
        widget._set_poni_field(str(poni_path))

        widget._refresh_controls_v2_profile_now()
        state = widget._controls_v2_state()
        summary, ready, _tooltip = widget._controls_v2_run_summary(
            state, widget.controls_v2.profile)

        assert state.loaded_scan_available is True
        assert state.source_label == ""
        assert state.frame_count == 0
        assert state.source_caps.has_frames is False
        assert widget.controls_v2.profile.can_run is False
        assert summary == "Needs setup · Run needs a frame source · Int 2D"
        assert "use Reintegrate" not in summary
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_raw_source_and_poni_enable_run_with_source_frames(
        qapp, monkeypatch, tmp_path):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    from xrd_tools.core.containers import PONI
    from xrd_tools.io import image as image_io

    widget = staticWidget()
    try:
        monkeypatch.setattr(image_io, "count_frames", lambda path: 1)
        project = tmp_path / "project"
        project.mkdir()
        raw_path = project / "raw" / "scan_0001.tif"
        raw_path.parent.mkdir()
        raw_path.write_bytes(b"")
        poni_path = project / "cal.poni"
        PONI(
            dist=0.1794,
            poni1=0.0,
            poni2=0.0,
            detector="RayonixMx225",
            wavelength=0.7293e-10,
        ).to_poni_file(poni_path)

        widget._controls_v2_param(("Project", "project_folder")).setValue(str(project))
        widget.wrangler.project_folder = str(project)
        widget._controls_v2_param(("Project", "h5_dir")).setValue(
            str(project / "xdart_processed_data")
        )
        widget.wrangler.h5_dir = str(project / "xdart_processed_data")
        widget._controls_v2_param(("Signal", "File")).setValue(str(raw_path))
        widget.wrangler.img_file = str(raw_path)
        widget._set_poni_field(str(poni_path))

        widget._refresh_controls_v2_profile_now()

        state = widget._controls_v2_state()
        assert state.frame_count == 1
        assert state.source_label == str(raw_path)
        assert state.geom.calibration_energy_eV is not None
        assert widget.controls_v2.profile.can_run is True
        assert widget.controls.actionRow.isEnabled() is True
        assert widget.controls.readinessDot.property("ready") is True
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_append_mismatch_same_target_stays_clickable(
        qapp, monkeypatch, tmp_path):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    from xrd_tools.core.containers import PONI
    from xrd_tools.io import image as image_io

    widget = staticWidget()
    try:
        monkeypatch.setattr(image_io, "count_frames", lambda path: 1)
        project = tmp_path / "project"
        project.mkdir()
        out_dir = project / "xdart_processed_data"
        out_dir.mkdir()
        raw_path = project / "raw" / "scan_0001.tif"
        raw_path.parent.mkdir()
        raw_path.write_bytes(b"")
        target = out_dir / "scan.nxs"
        target.write_bytes(b"")
        poni_path = project / "cal.poni"
        PONI(
            dist=0.1794,
            poni1=0.0,
            poni2=0.0,
            detector="RayonixMx225",
            wavelength=0.7293e-10,
        ).to_poni_file(poni_path)

        widget._controls_v2_param(("Project", "project_folder")).setValue(str(project))
        widget.wrangler.project_folder = str(project)
        widget._controls_v2_param(("Project", "h5_dir")).setValue(str(out_dir))
        widget.wrangler.h5_dir = str(out_dir)
        widget._controls_v2_param(("Signal", "File")).setValue(str(raw_path))
        widget.wrangler.img_file = str(raw_path)
        widget._set_poni_field(str(poni_path))
        widget.controls.set_write_mode("Append")
        widget.scan.data_file = str(target)
        widget.scan.reduction_config = {
            "gi": False,
            "bai_1d_args": {"unit": "q_A^-1"},
            "bai_2d_args": {"unit": "q_A^-1"},
        }
        widget.scan._display_reduction_config = dict(widget.scan.reduction_config)
        # Enable Grazing through the PRODUCTION Controls seam (single owner)
        # rather than writing the retired display-scan flag directly: this sets
        # both the Controls-owned intent (which the Run-click freeze reads) and
        # the projected scan (which _controls_v2_state's current_config reads),
        # so the processed(Standard)-vs-current(Grazing) mismatch is detected
        # both at state build and at Run click.
        widget._on_controls_v2_field_changed(("GI", "Grazing"), True)

        state = widget._controls_v2_state()
        profile = build_control_profile(state)

        assert state.processed_config is not None
        assert profile.can_run is True
        assert profile.run_blockers == ()
        assert "processed: Standard · current: Grazing" in (
            profile.append_confirm_reason
        )
        widget._refresh_controls_v2_profile_now()
        assert widget.controls.actionRow.isEnabled() is True
        assert widget.controls.startButton.isEnabled() is True
        assert "Confirm overwrite" in widget.controls.readinessLabel.text()
        assert "processed: Standard · current: Grazing" in (
            widget.controls.readinessLabel.toolTip()
        )
        assert "processed: Standard · current: Grazing" in (
            widget.wrangler._append_config_mismatch_message()
        )

        prompts = []

        def _cancel(check, processed, current):
            prompts.append((
                check.reason,
                processed.display_mode,
                current.display_mode,
            ))
            return False

        widget.wrangler._confirm_append_config_replace = _cancel
        widget.controls.startButton.click()
        qapp.processEvents()

        assert prompts == [(
            profile.append_confirm_reason,
            "Standard",
            "Grazing",
        )]
        assert getattr(widget.wrangler, "command", None) != "start"
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_append_mismatch_ignores_unrelated_displayed_scan(
        qapp, monkeypatch, tmp_path):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    from xrd_tools.core.containers import PONI
    from xrd_tools.io import image as image_io

    widget = staticWidget()
    try:
        monkeypatch.setattr(image_io, "count_frames", lambda path: 1)
        project = tmp_path / "project"
        project.mkdir()
        out_dir = project / "xdart_processed_data"
        out_dir.mkdir()
        raw_path = project / "raw" / "next_0001.tif"
        raw_path.parent.mkdir()
        raw_path.write_bytes(b"")
        displayed = out_dir / "displayed.nxs"
        displayed.write_bytes(b"")
        poni_path = project / "cal.poni"
        PONI(
            dist=0.1794,
            poni1=0.0,
            poni2=0.0,
            detector="RayonixMx225",
            wavelength=0.7293e-10,
        ).to_poni_file(poni_path)

        widget._controls_v2_param(("Project", "project_folder")).setValue(str(project))
        widget.wrangler.project_folder = str(project)
        widget._controls_v2_param(("Project", "h5_dir")).setValue(str(out_dir))
        widget.wrangler.h5_dir = str(out_dir)
        widget._controls_v2_param(("Signal", "File")).setValue(str(raw_path))
        widget.wrangler.img_file = str(raw_path)
        widget._set_poni_field(str(poni_path))
        widget.controls.set_write_mode("Append")
        widget.scan.data_file = str(displayed)
        widget.scan.reduction_config = {
            "gi": False,
            "bai_1d_args": {"unit": "q_A^-1"},
            "bai_2d_args": {"unit": "q_A^-1"},
        }
        widget.scan._display_reduction_config = dict(widget.scan.reduction_config)
        widget.scan.gi = True

        state = widget._controls_v2_state()
        profile = build_control_profile(state)

        assert state.processed_config is None
        assert profile.can_run is True
        assert widget.wrangler._append_config_mismatch_message() == ""
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_native_int_uses_binding_table(
        qapp, monkeypatch):
    """All V2 integration fields are harvested through the single binding table,
    so adding a future control is a one-row native-state change."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget._refresh_controls_v2_profile_now()
        values = widget._controls_v2_native_int_values()

        expected_paths = {
            spec.path
            for spec in INTEGRATOR_BACKED_CONTROL_SPECS
            if spec.path in values
        }
        assert expected_paths <= set(values)
        # (The retired per-field permissive setter `_set_controls_v2_native_int_field`
        # is deleted — writes now go ONLY through the validated stage/commit engine,
        # so there is no second setter authority left to round-trip here.)
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_native_gi_oop_points_feed_plan(
        qapp, monkeypatch):
    """Native GI fiber OOP points remain visible to V2 and reach the plan."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        _apply_v2_edits(widget, ((("GI", "Grazing"), True),))
        qip = _field_choice_text(
            widget,
            ("Int1D", "axis"),
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

        assert widget.ui.integratorFrame.isHidden()
        assert widget.integratorTree.ui.npts_oop_1D.isHidden()

        values = widget._controls_v2_native_int_values()
        assert values[("Int1D", "points_oop")] == "345"
        assert _visible_control_value(widget, ("Int1D", "points_oop")) == "345"

        _apply_prepared_run_state(widget)
        snapshot = _current_plan_snapshot(widget, commit_pending=False)
        assert snapshot["integration_1d"]["npt"] == 234
        assert snapshot["gi"]["mode_1d"] == "q_ip"
        assert snapshot["gi"]["npt_oop"] == 345
    finally:
        _reset_controls_v2_gi(widget)
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_standard_edits_feed_native_reduction_plan(
        qapp, monkeypatch):
    """V2 standard edits are authoritative for the scan-backed native plan."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        axis_1d = _field_choice_text(
            widget,
            ("Int1D", "axis"),
            lambda text: "2" in text and "θ" in text,
        )
        axis_2d = _field_choice_text(
            widget,
            ("Int2D", "axis"),
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

        _apply_v2_edits(widget, edits)

        assert _native_plan_snapshot(widget, commit_pending=False) == (
            _current_plan_snapshot(widget, commit_pending=False)
        )
        snapshot = _native_plan_snapshot(widget, commit_pending=False)
        assert snapshot["integration_1d"]["npt"] == 321
        assert snapshot["integration_1d"]["unit"] == "2th_deg"
        assert snapshot["integration_2d"]["npt_rad"] == 111
        assert snapshot["integration_2d"]["npt_azim"] == 77
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_gi_edits_feed_native_reduction_plan(
        qapp, monkeypatch):
    """GI edits feed the native plan without the hidden legacy parser."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        _apply_v2_edits(widget, ((("GI", "Grazing"), True),))
        q_total_or_current = _field_choice_text(
            widget,
            ("Int1D", "axis"),
            lambda text: text == "Q",
        )
        qip_qoop_or_current = _field_choice_text(
            widget,
            ("Int2D", "axis"),
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

        _apply_v2_edits(widget, edits)

        assert _native_plan_snapshot(widget, commit_pending=False) == (
            _current_plan_snapshot(widget, commit_pending=False)
        )
        snapshot = _native_plan_snapshot(widget, commit_pending=False)
        assert snapshot["gi"]["sample_orientation"] == 4
        assert snapshot["gi"]["mode_1d"] == "q_total"
        assert snapshot["gi"]["mode_2d"] == "qip_qoop"
        assert snapshot["gi"]["npt_oop"] == 33
    finally:
        _reset_controls_v2_gi(widget)
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_threshold_edits_feed_native_plan_overlay(
        qapp, monkeypatch):
    """Threshold + saturation are a native plan overlay."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        edits = (
            (("Mask", "Threshold"), True),
            (("Mask", "min"), "12.5"),
            (("Mask", "max"), "987.5"),
            (("MaskSat", "mask_sentinel"), True),
        )

        _apply_v2_edits(widget, edits)

        cfg = widget._controls_v2_threshold_config()
        assert cfg.apply_threshold is True
        assert cfg.threshold_min == pytest.approx(12.5)
        assert cfg.threshold_max == pytest.approx(987.5)
        assert cfg.mask_saturation is True
        assert _native_plan_snapshot(widget, commit_pending=False) == (
            _current_plan_snapshot(widget, commit_pending=False)
        )
    finally:
        widget.close()
        widget.deleteLater()


@pytest.mark.parametrize(
    ("integrate_1d", "integrate_2d"),
    (
        (True, False),
        (False, True),
        (True, True),
    ),
)
def test_controls_panel_v2_native_plan_matches_legacy_output_modes(
        qapp, monkeypatch, integrate_1d, integrate_2d):
    """The native plan seam must preserve one-output and both-output runs."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        edits = (
            (("Int1D", "points"), "345"),
            (("Int1D", "radial_auto"), False),
            (("Int1D", "radial_low"), "0.2"),
            (("Int1D", "radial_high"), "4.8"),
            (("Int2D", "radial_points"), "123"),
            (("Int2D", "azim_points"), "77"),
            (("Int2D", "radial_auto"), False),
            (("Int2D", "radial_low"), "0.3"),
            (("Int2D", "radial_high"), "4.7"),
        )
        _apply_v2_edits(widget, edits)

        assert _native_plan_snapshot(
            widget,
            integrate_1d=integrate_1d,
            integrate_2d=integrate_2d,
            commit_pending=False,
        ) == _current_plan_snapshot(
            widget,
            integrate_1d=integrate_1d,
            integrate_2d=integrate_2d,
            commit_pending=False,
        )
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_native_plan_preserves_monitor_parity():
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


def test_controls_panel_v2_native_scan_builder_matches_legacy_plan():
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


def test_controls_panel_v2_native_gi_plan_defaults_orientation_to_4():
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


def test_controls_panel_v2_native_run_plan_gate_configures_plan_caches(
        qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    monkeypatch.delenv("XDART_CONTROLS_V2_NATIVE_RUN_PLAN", raising=False)
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        cache = widget.wrangler.thread._plan_cache
        reint_cache = widget.integratorTree.integrator_thread._plan_cache
        scan = widget.scan
        scan.skip_2d = False
        scan.bai_1d_args.update({"numpoints": 123, "unit": "q_A^-1"})
        scan.bai_2d_args.update({"npt_rad": 45, "npt_azim": 67})
        widget._configure_controls_v2_native_run_plan()
        assert cache.plan_builder is not None
        assert reint_cache.plan_builder is not None
        assert _plan_snapshot(cache.get(scan)) == _plan_snapshot(
            build_native_int_reduction_plan_from_scan(scan)
        )

        monkeypatch.setenv("XDART_CONTROLS_V2_NATIVE_RUN_PLAN", "0")
        widget._configure_controls_v2_native_run_plan()
        assert cache.plan_builder is None
        assert reint_cache.plan_builder is None
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_native_plan_builder_is_snapshot_scoped(
        qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    monkeypatch.delenv("XDART_CONTROLS_V2_NATIVE_RUN_PLAN", raising=False)
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget.scan.bai_1d_args.update({"numpoints": 123, "unit": "q_A^-1"})
        builder = widget._controls_v2_native_run_plan_builder(
            widget._controls_v2_native_int_snapshot()
        )
        assert getattr(builder, "prepare_scan", None) is not None
        assert getattr(builder, "plan_cache_key", None) is not None
        closure_values = [
            cell.cell_contents for cell in (builder.__closure__ or ())
        ]
        prepare_values = [
            cell.cell_contents
            for cell in (builder.prepare_scan.__closure__ or ())
        ]
        assert widget not in closure_values
        assert widget not in prepare_values
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_native_reintegrate_plan_is_authoritative(
        qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    monkeypatch.delenv("XDART_CONTROLS_V2_NATIVE_RUN_PLAN", raising=False)
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        _apply_v2_edits(
            widget,
            (
                (("Int1D", "points"), "321"),
                (("Int1D", "radial_auto"), False),
                (("Int1D", "radial_low"), "0.1"),
                (("Int1D", "radial_high"), "4.2"),
                (("Int1D", "azim_auto"), False),
                (("Int1D", "azim_low"), "-50"),
                (("Int1D", "azim_high"), "60"),
                (("Int2D", "radial_points"), "77"),
                (("Int2D", "azim_points"), "88"),
                (("Int2D", "radial_auto"), False),
                (("Int2D", "radial_low"), "0.2"),
                (("Int2D", "radial_high"), "5.3"),
                (("Int2D", "azim_auto"), False),
                (("Int2D", "azim_low"), "-45"),
                (("Int2D", "azim_high"), "45"),
                (("Mask", "Threshold"), True),
                (("Mask", "min"), "3"),
                (("Mask", "max"), "999"),
                (("MaskSat", "mask_sentinel"), False),
            ),
        )
        thread = widget.integratorTree.integrator_thread
        thread.threshold_config = widget._controls_v2_threshold_config()
        cache = thread._plan_cache
        widget._configure_controls_v2_native_run_plan()
        assert cache.plan_builder is not None
        native_1d = _plan_snapshot(thread._plan_for_reintegration(integrate_2d=False))
        cache.invalidate()
        native_2d = _plan_snapshot(thread._plan_for_reintegration(integrate_2d=True))

        assert native_1d == _native_plan_snapshot(
            widget, integrate_1d=True, integrate_2d=False, commit_pending=False)
        assert native_2d == _native_plan_snapshot(
            widget, integrate_1d=False, integrate_2d=True, commit_pending=False)
    finally:
        widget.close()
        widget.deleteLater()


@pytest.mark.parametrize("gi_enabled", [False, True])
def test_controls_panel_v2_native_reintegrate_results_match_run_after_stale_legacy_click(
        qapp, monkeypatch, gi_enabled):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    monkeypatch.delenv("XDART_CONTROLS_V2_NATIVE_RUN_PLAN", raising=False)
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    from xdart.modules.live import LiveFrame
    import xdart.modules.reduction as reduction_adapters
    from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
    from xrd_tools.reduction import FrameReduction, ReductionResult

    def _range_pair(value):
        if value is None:
            return (0.0, 0.0)
        return float(value[0]), float(value[1])

    def _fake_run_reduction(plan_arg, scan_arg, **kwargs):
        frame_idx = int(scan_arg.frames[0].index)
        p1 = plan_arg.integration_1d
        p2 = plan_arg.integration_2d
        r1 = None
        if p1 is not None:
            rr = _range_pair(p1.radial_range)
            ar = _range_pair(p1.azimuth_range)
            r1 = IntegrationResult1D(
                radial=np.array([float(p1.npt), rr[0], rr[1]], dtype=float),
                intensity=np.array(
                    [float(p1.npt_rad), ar[0], ar[1]], dtype=float),
                sigma=None,
                unit=p1.unit,
            )
        r2 = None
        if p2 is not None:
            rr = _range_pair(p2.radial_range)
            ar = _range_pair(p2.azimuth_range)
            r2 = IntegrationResult2D(
                radial=np.array([float(p2.npt_rad), rr[0], rr[1]], dtype=float),
                azimuthal=np.array(
                    [float(p2.npt_azim), ar[0], ar[1]], dtype=float),
                intensity=np.add.outer(
                    np.array([float(p2.npt_rad), rr[0], rr[1]], dtype=float),
                    np.array([float(p2.npt_azim), ar[0], ar[1]], dtype=float),
                ),
                sigma=None,
                unit=p2.unit,
            )
        return ReductionResult(
            scan_name=scan_arg.name,
            frames={frame_idx: FrameReduction(
                frame_idx, result_1d=r1, result_2d=r2)},
            n_processed=1,
        )

    def _result_signature(plan):
        frame = LiveFrame(
            idx=7,
            map_raw=np.arange(16, dtype=float).reshape(4, 4),
            scan_info={"th": 0.24},
        )
        reduction_adapters.reduce_live_frame(frame, plan, scan_name="scan")
        sig = {}
        if frame.int_1d is not None:
            sig["1d"] = (
                frame.int_1d.unit,
                tuple(np.asarray(frame.int_1d.radial, dtype=float)),
                tuple(np.asarray(frame.int_1d.intensity, dtype=float)),
            )
        if frame.int_2d is not None:
            sig["2d"] = (
                frame.int_2d.unit,
                tuple(np.asarray(frame.int_2d.radial, dtype=float)),
                tuple(np.asarray(frame.int_2d.azimuthal, dtype=float)),
                tuple(np.asarray(frame.int_2d.intensity, dtype=float).ravel()),
            )
        return sig

    class ClobberingButton:
        clicked = False

        def __init__(self, widget):
            self.widget = widget

        def click(self):
            self.clicked = True
            scan = self.widget.scan
            scan.bai_1d_args = {
                "unit": "2th_deg",
                "numpoints": 3000,
                "radial_range": None,
                "azimuth_range": None,
            }
            scan.bai_2d_args = {
                "unit": "2th_deg",
                "npt_rad": 500,
                "npt_azim": 500,
                "radial_range": None,
                "azimuth_range": None,
            }
            scan.gi = False
            scan.gi_config = {}
            scan.incidence_motor = ""

    monkeypatch.setattr(reduction_adapters, "run_reduction", _fake_run_reduction)

    widget = staticWidget()
    try:
        edits = [
            (("Int1D", "points"), "37"),
            (("Int1D", "radial_auto"), False),
            (("Int1D", "radial_low"), "0.15"),
            (("Int1D", "radial_high"), "3.1"),
            (("Int1D", "azim_auto"), False),
            (("Int1D", "azim_low"), "-35"),
            (("Int1D", "azim_high"), "48"),
            (("Int2D", "radial_points"), "9"),
            (("Int2D", "azim_points"), "7"),
            (("Int2D", "radial_auto"), False),
            (("Int2D", "radial_low"), "0.2"),
            (("Int2D", "radial_high"), "3.4"),
            (("Int2D", "azim_auto"), False),
            (("Int2D", "azim_low"), "-42"),
            (("Int2D", "azim_high"), "51"),
            (("MaskSat", "mask_sentinel"), False),
        ]
        if gi_enabled:
            edits = [
                (("GI", "Grazing"), True),
                (("GI", "th_motor"), "Manual"),
                (("GI", "th_val"), "0.24"),
                (("GI", "sample_orientation"), "4"),
                (("GI", "tilt_angle"), "0.6"),
            ] + edits
        _apply_v2_edits(widget, edits)

        widget.scan.skip_2d = False
        _apply_prepared_run_state(widget)
        widget._configure_controls_v2_native_run_plan()
        run_cache = widget.wrangler.thread._plan_cache
        run_1d = run_cache.get(widget.scan, integrate_1d=True, integrate_2d=False)
        run_cache.invalidate()
        run_2d = run_cache.get(widget.scan, integrate_1d=False, integrate_2d=True)

        thread = widget.integratorTree.integrator_thread
        clobber_1d = ClobberingButton(widget)
        monkeypatch.setattr(widget.integratorTree.ui, "reintegrate1D", clobber_1d)
        widget._controls_v2_click_integrator_button("reintegrate1D")
        assert clobber_1d.clicked is True
        assert widget.scan.bai_1d_args["numpoints"] == 3000
        reintegrate_1d = thread._plan_for_reintegration(integrate_2d=False)

        clobber_2d = ClobberingButton(widget)
        monkeypatch.setattr(widget.integratorTree.ui, "reintegrate2D", clobber_2d)
        widget._controls_v2_click_integrator_button("reintegrate2D")
        assert clobber_2d.clicked is True
        assert widget.scan.bai_2d_args["npt_rad"] == 500
        reintegrate_2d = thread._plan_for_reintegration(integrate_2d=True)

        assert _plan_snapshot(reintegrate_1d) == _plan_snapshot(run_1d)
        assert _plan_snapshot(reintegrate_2d) == _plan_snapshot(run_2d)
        assert _result_signature(reintegrate_1d) == _result_signature(run_1d)
        assert _result_signature(reintegrate_2d) == _result_signature(run_2d)
    finally:
        if gi_enabled:
            _reset_controls_v2_gi(widget)
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_reintegrate_action_installs_native_builder(
        qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    monkeypatch.setenv("XDART_CONTROLS_V2_NATIVE_RUN_PLAN", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    class FakeButton:
        def __init__(self):
            self.clicked = False

        def click(self):
            self.clicked = True

    widget = staticWidget()
    fake = FakeButton()
    try:
        cache = widget.integratorTree.integrator_thread._plan_cache
        cache.plan_builder = None
        monkeypatch.setattr(widget.integratorTree.ui, "reintegrate1D", fake)

        widget._controls_v2_click_integrator_button("reintegrate1D")

        assert fake.clicked is True
        assert cache.plan_builder is not None
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_native_session_roundtrip_hydrates_visible_rows(
        qapp, monkeypatch, tmp_path):
    """V2-edited integration state survives close/open through native state."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    monkeypatch.setenv("XDART_SESSION_FILE", str(tmp_path / "session.json"))
    monkeypatch.delenv("XDART_SESSION_FRESH", raising=False)
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    edited = staticWidget()
    restored = None
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
        before = _native_plan_snapshot(edited)

        edited.close()
        edited.deleteLater()
        edited = None

        restored = staticWidget()
        restored._refresh_controls_v2_profile_now()

        assert _native_plan_snapshot(restored) == before
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
        widgets = [w for w in (edited, restored) if w is not None]
        _reset_controls_v2_gi(*widgets)
        for widget in widgets:
            widget.close()
            widget.deleteLater()


def test_config_save_uses_native_standard_state_not_stale_legacy_tree(
        qapp, monkeypatch, tmp_path):
    """Config Save serializes what Controls V2 shows, not a hidden carrier."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    from xrd_tools.core.containers import PONI

    poni_path = tmp_path / "rayonix.poni"
    PONI(
        dist=0.1794,
        poni1=0.01,
        poni2=0.02,
        detector="RayonixMx225",
    ).to_poni_file(poni_path)
    config_path = tmp_path / "image-directory.json"

    widget = staticWidget()
    try:
        widget._set_poni_field(str(poni_path))
        legacy_gi = widget.wrangler.parameters.child("GI").child("Grazing")
        legacy_gi.setValue(True)
        _apply_v2_edits(widget, (
            (("GI", "Grazing"), False),
            (("Int1D", "points"), "432"),
        ))

        assert legacy_gi.value() is True
        assert widget.scan.gi is False

        widget.h5viewer.defaultWidget.save_defaults(fname=str(config_path))
        saved = json.loads(config_path.read_text())
        active = saved["image_wrangler"]["image_wrangler"]
        canonical = saved[widget._CONFIG_STATE_KEY]

        assert active["GI"]["Grazing"] is False
        assert active["Signal"]["poni_file"] == str(poni_path)
        assert canonical["active_wrangler"] == "image_wrangler"
        assert canonical["poni_file"] == str(poni_path)
        assert canonical["controls_v2_int"]["gi"] is False
        assert canonical["controls_v2_int"]["bai_1d_args"]["numpoints"] == 432
    finally:
        widget.close()
        widget.deleteLater()


def test_config_roundtrip_restores_active_poni_mode_and_native_grid(
        qapp, monkeypatch, tmp_path):
    """Loading A after B leaves no B-owned PONI or GI/native state behind."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    from xrd_tools.core.containers import PONI

    rayonix_path = tmp_path / "rayonix.poni"
    eiger_path = tmp_path / "eiger.poni"
    rayonix = PONI(
        dist=0.1794,
        poni1=0.01,
        poni2=0.02,
        detector="RayonixMx225",
    )
    eiger = PONI(
        dist=0.1385,
        poni1=0.03,
        poni2=0.04,
        detector="Eiger4M",
    )
    rayonix.to_poni_file(rayonix_path)
    eiger.to_poni_file(eiger_path)
    config_a = tmp_path / "image-directory.json"
    config_b = tmp_path / "eiger.json"

    widget = staticWidget()
    try:
        widget._set_poni_field(str(rayonix_path))
        _apply_v2_edits(widget, (
            (("GI", "Grazing"), False),
            (("Int1D", "points"), "321"),
        ))
        widget.h5viewer.defaultWidget.save_defaults(fname=str(config_a))

        widget._set_poni_field(str(eiger_path))
        _apply_v2_edits(widget, (
            (("GI", "Grazing"), True),
            (("Int1D", "points"), "777"),
        ))
        widget.h5viewer.defaultWidget.save_defaults(fname=str(config_b))

        widget.h5viewer.defaultWidget.load_defaults(fname=str(config_b))
        widget.scan._cached_poni = eiger
        widget.h5viewer.defaultWidget.load_defaults(fname=str(config_a))
        qapp.processEvents()

        assert widget.scan.gi is False
        assert widget.wrangler.parameters.child("GI").child("Grazing").value() is False
        assert widget.scan.bai_1d_args["numpoints"] == 321
        assert widget._controls_v2_poni_path() == str(rayonix_path)
        assert widget.wrangler.poni.detector == "RayonixMx225"
        assert widget._controls_v2_current_poni().detector == "RayonixMx225"
        assert _visible_control_value(widget, ("GI", "Grazing")) is False
        assert _visible_control_value(widget, ("Int1D", "points")) == "321"
    finally:
        widget.close()
        widget.deleteLater()


def test_legacy_calibration_poni_restores_after_signal_schema_config(
        qapp, monkeypatch, tmp_path):
    """Old TIFF configs must replace a previously loaded Eiger calibration."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    from xrd_tools.core.containers import PONI

    tiff_poni_path = tmp_path / "tiff-rayonix.poni"
    eiger_poni_path = tmp_path / "eiger.poni"
    PONI(
        dist=0.1794,
        poni1=0.01,
        poni2=0.02,
        detector="RayonixMx225",
    ).to_poni_file(tiff_poni_path)
    PONI(
        dist=0.1385,
        poni1=0.03,
        poni2=0.04,
        detector="Eiger4M",
    ).to_poni_file(eiger_poni_path)
    tiff_config = tmp_path / "legacy-tiff.json"
    eiger_config = tmp_path / "eiger.json"

    widget = staticWidget()
    try:
        widget._set_poni_field(str(eiger_poni_path))
        widget.h5viewer.defaultWidget.save_defaults(fname=str(eiger_config))

        widget._set_poni_field(str(tiff_poni_path))
        widget.h5viewer.defaultWidget.save_defaults(fname=str(tiff_config))
        legacy = json.loads(tiff_config.read_text())
        legacy.pop(widget._CONFIG_STATE_KEY, None)
        image_tree = legacy["image_wrangler"]["image_wrangler"]
        saved_poni = image_tree["Signal"].pop("poni_file")
        image_tree.setdefault("Calibration", {})["poni_file"] = saved_poni
        tiff_config.write_text(json.dumps(legacy))

        widget.h5viewer.defaultWidget.load_defaults(fname=str(eiger_config))
        widget.h5viewer.defaultWidget.load_defaults(fname=str(tiff_config))
        qapp.processEvents()

        assert widget._controls_v2_poni_path() == str(tiff_poni_path)
        assert widget.wrangler.poni_file == str(tiff_poni_path)
        assert widget.wrangler.poni.detector == "RayonixMx225"
        assert widget._controls_v2_current_poni().detector == "RayonixMx225"
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_native_int_session_roundtrip_feeds_native_plan(
        qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    monkeypatch.setenv("XDART_CONTROLS_V2_NATIVE_RUN_PLAN", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    from xdart.utils.session import save_session

    edited = staticWidget()
    restored = None
    try:
        _apply_v2_edits(
            edited,
            (
                (("Int1D", "points"), "246"),
                (("Int1D", "radial_auto"), False),
                (("Int1D", "radial_low"), "0.4"),
                (("Int1D", "radial_high"), "3.9"),
                (("Int1D", "azim_auto"), False),
                (("Int1D", "azim_low"), "-80"),
                (("Int1D", "azim_high"), "70"),
                (("Int1D", "method"), "BBox"),
                (("Int1D", "apply_polarization"), True),
                (("Int1D", "polarization_factor"), "0.42"),
                (("Int2D", "radial_points"), "55"),
                (("Int2D", "azim_points"), "66"),
                (("Int2D", "radial_auto"), False),
                (("Int2D", "radial_low"), "0.5"),
                (("Int2D", "radial_high"), "4.4"),
                (("Int2D", "azim_auto"), False),
                (("Int2D", "azim_low"), "-45"),
                (("Int2D", "azim_high"), "45"),
                (("Int2D", "method"), "BBox"),
                (("Int2D", "apply_polarization"), True),
                (("Int2D", "polarization_factor"), "0.24"),
                (("Mask", "Threshold"), True),
                (("Mask", "min"), "11"),
                (("Mask", "max"), "900"),
                (("MaskSat", "mask_sentinel"), False),
            ),
        )
        before = _native_plan_snapshot(edited)

        edited.close()
        edited.deleteLater()
        edited = None

        # The native Controls V2 blob should win over a stale legacy bridge.
        save_session({
            "integrator": {
                "ui": {
                    "npts_1D": "999",
                    "npts_radial_2D": "998",
                    "threshold_min": "1",
                },
                "advanced": None,
            }
        })

        restored = staticWidget()
        after = _native_plan_snapshot(restored)

        assert after == before
        assert restored.scan.bai_1d_args["numpoints"] == 246
        assert restored.scan.bai_2d_args["npt_rad"] == 55
        assert _visible_control_value(restored, ("Int1D", "points")) == "246"
        assert _visible_control_value(restored, ("Int2D", "radial_points")) == "55"
        cfg = restored._controls_v2_threshold_config()
        assert cfg.apply_threshold is True
        assert cfg.threshold_min == pytest.approx(11)
        assert cfg.threshold_max == pytest.approx(900)
        assert cfg.mask_saturation is False
    finally:
        if edited is not None:
            edited.close()
            edited.deleteLater()
        if restored is not None:
            restored.close()
            restored.deleteLater()


def test_controls_panel_v2_threshold_edits_update_native_state(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget._on_controls_v2_field_changed(("Mask", "Threshold"), True)
        widget._on_controls_v2_field_changed(("Mask", "min"), "10")
        widget._on_controls_v2_field_changed(("Mask", "max"), "900")
        widget._on_controls_v2_field_changed(("MaskSat", "mask_sentinel"), False)

        cfg = widget._controls_v2_threshold_config()
        assert cfg.apply_threshold is True
        assert cfg.threshold_min == pytest.approx(10)
        assert cfg.threshold_max == pytest.approx(900)
        assert cfg.mask_saturation is False
        assert not widget.integratorTree.ui.threshold_enable.isChecked()
        assert widget.wrangler.parameters.child("Mask", "Threshold").value() is False
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


def test_controls_panel_v2_threshold_value_autoenables_native_state(
        qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget._on_controls_v2_field_changed(("Mask", "Threshold"), False)
        widget._on_controls_v2_field_changed(("Mask", "max"), "1000")

        cfg = widget._controls_v2_threshold_config()
        assert cfg.apply_threshold is True
        assert cfg.threshold_max == pytest.approx(1000.0)

        widget._on_controls_v2_field_changed(("Mask", "max"), "0")
        widget._on_controls_v2_field_changed(("Mask", "Threshold"), False)
        widget._on_controls_v2_field_changed(("Mask", "min"), "0")
        widget._on_controls_v2_field_changed(("Mask", "max"), "0")

        cfg = widget._controls_v2_threshold_config()
        assert cfg.apply_threshold is False
        assert cfg.threshold_min == 0.0
        assert cfg.threshold_max == 0.0
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_mask_saturated_survives_run_state(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget._on_controls_v2_field_changed(("MaskSat", "mask_sentinel"), True)
        assert widget._controls_v2_threshold_config().mask_saturation is True
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
        assert widget._controls_v2_threshold_config().mask_saturation is True
    finally:
        widget._exit_run_state(widget._new_projection_receipt())
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

        # Non-Int source conditioning still writes directly to the wrangler
        # parameter tree; the retired bridge only covered Int state.
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
        assert mtype.label.minimumWidth() == 92

        # Image Series: no Subdirs (directory-only), Meta Type still renders.
        widget._on_controls_v2_field_changed(("Signal", "inp_type"), "Image Series")
        widget._refresh_controls_v2_profile_now()
        assert _src_row(widget, ("Signal", "include_subdir")) is None
        mtype = _src_row(widget, ("Signal", "meta_ext"))
        assert mtype is not None
        assert mtype.label.minimumWidth() == 76
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_gi_options_popup_holds_orient_and_tilt(qapp, monkeypatch):
    """In Grazing mode: θ motor renders inline (compact label, descriptive
    tooltip); Orientation + Tilt Angle live behind a '…' button that opens a
    small popup, whose rows still write through."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    monkeypatch.setenv("XDART_SESSION_FRESH", "1")
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
        orientation_rows = [
            r for r in popup.findChildren(FormRow)
            if r.path == ("GI", "sample_orientation")
        ]
        assert orientation_rows[0].current_value() == "4"
        assert widget.integratorTree.get_gi_config()["sample_orientation"] == 4

        # A popup row updates the native GI provider used by integrator actions.
        orientation_rows[0].editor.setText("6")
        orientation_rows[0].editor.editingFinished.emit()
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
        # LV-UI-6 (accepted E4 behavior): the editor displays only the file
        # name; the FULL path lives in the tooltip — and the committed value
        # stays the full model path, never the displayed stem.
        assert editor.text() == "scan_0001.tif"
        assert editor.toolTip() == "/data/very/long/path/scan_0001.tif"
        assert rows[0].current_value() == "/data/very/long/path/scan_0001.tif"
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
        widget._controls_v2_last_signature = None
        widget._controls_v2_last_schema_signature = None
        widget._refresh_controls_v2_profile_now(preserve_focused_editor=False)

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
        widget._exit_run_state(widget._new_projection_receipt())
        widget.close()
        widget.deleteLater()



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

def test_controls_panel_v2_run_commits_focused_integration_edit(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    class FakeThread:
        batch_mode = False
        xye_only = False

        def start(self):
            pass

        # A real ``imageThread`` is a QThread, so it always answers the
        # run-owner admission probe.  T-3.1 refuses a Start when a PRESENT
        # owner cannot be observed, so this double must be faithful here or
        # it would stand in for a broken Qt owner rather than an idle one.
        def isRunning(self):
            return False

    widget = staticWidget()
    try:
        widget._refresh_controls_v2_profile_now()
        rows = [
            row
            for row in widget.controls_v2.processing_card.body.findChildren(FormRow)
            if row.path == ("Int1D", "points")
        ]
        assert rows
        _user_types(qapp, widget, rows[0].editor, "777")
        assert widget.integratorTree.ui.npts_1D.text() != "777"

        widget.wrangler.thread = FakeThread()
        monkeypatch.setattr(widget.wrangler, "setup", lambda: None)

        widget._prepare_controls_v2_run_configuration()
        _admit_pending_run_configuration(widget)
        widget.start_wrangler()

        assert widget.integratorTree.ui.npts_1D.text() != "777"
        assert widget.scan.bai_1d_args["numpoints"] == 777
        assert widget.wrangler.scan_args["bai_1d_args"]["numpoints"] == 777
    finally:
        widget._exit_run_state(widget._new_projection_receipt())
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
        _user_types(qapp, widget, rows[("Int1D", "points")].editor, "246")
        assert widget.integratorTree.ui.npts_1D.text() != "246"

        (args, _frozen) = _apply_prepared_run_state(widget)

        assert args is widget.wrangler.scan_args
        assert widget.integratorTree.ui.npts_1D.text() != "246"
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

        # A real ``imageThread`` is a QThread, so it always answers the
        # run-owner admission probe.  T-3.1 refuses a Start when a PRESENT
        # owner cannot be observed, so this double must be faithful here or
        # it would stand in for a broken Qt owner rather than an idle one.
        def isRunning(self):
            return False

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
        _user_types(qapp, widget, rows[("Int2D", "radial_points")].editor, "123")
        _user_types(qapp, widget, rows[("Int2D", "azim_points")].editor, "456")
        assert widget.integratorTree.ui.npts_radial_2D.text() != "123"

        widget.wrangler.thread = FakeThread()
        monkeypatch.setattr(widget.wrangler, "setup", lambda: None)

        widget._prepare_controls_v2_run_configuration()
        _admit_pending_run_configuration(widget)
        widget.start_wrangler()

        assert widget.integratorTree.ui.npts_radial_2D.text() != "123"
        assert widget.integratorTree.ui.npts_azim_2D.text() != "456"
        assert widget.wrangler.scan_args["bai_2d_args"]["npt_rad"] == 123
        assert widget.wrangler.scan_args["bai_2d_args"]["npt_azim"] == 456
    finally:
        widget._exit_run_state(widget._new_projection_receipt())
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
        _user_types(qapp, widget, rows[("Int1D", "points")].editor, "432")
        assert widget.integratorTree.ui.npts_1D.text() != "432"

        monkeypatch.setattr(widget.integratorTree.ui, "reintegrate1D", fake)
        widget._controls_v2_click_integrator_button("reintegrate1D")

        assert fake.clicked is True
        assert widget.integratorTree.ui.npts_1D.text() != "432"
        assert widget.scan.bai_1d_args["numpoints"] == 432
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_reintegrate_finish_unlocks_stale_running_phase(
        qapp, monkeypatch):
    """A stale wrangler run PHASE must not leave V2 grey after reintegrate.

    T-4.2 (§35.2/§35.7.A): this case previously used ``isRunning() == True`` with
    ``_run_phase == "idle"`` and called that "stale".  That shape is not stale — it
    is the production Stop/unwind window (``imageWrangler.stop()`` sets the phase
    idle before the worker finishes), and releasing the shared lifecycle through it
    reopens T-3's fast-Start window.  The TRUTHFUL stale-phase shape is an IDLE
    QThread carrying a leftover ``running`` phase, which is what this now drives;
    the active-thread case is owned by
    ``test_t42_closure_evidence::test_active_stopping_wrangler_without_session_holds_shared_locks``."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget._refresh_controls_v2_profile_now()
        rows = {
            row.path: row
            for row in widget.controls_v2.processing_card.body.findChildren(FormRow)
        }
        assert rows[("Int1D", "points")].editor.isEnabled()

        widget._enter_run_state()
        rows = {
            row.path: row
            for row in widget.controls_v2.processing_card.body.findChildren(FormRow)
        }
        assert not rows[("Int1D", "points")].editor.isEnabled()

        monkeypatch.setattr(widget.wrangler.thread, "isRunning", lambda: False)
        monkeypatch.setattr(
            widget.wrangler, "_run_phase", "running", raising=False)

        widget.integrator_thread_finished()

        assert widget._run_active is False
        rows = {
            row.path: row
            for row in widget.controls_v2.processing_card.body.findChildren(FormRow)
        }
        assert rows[("Int1D", "points")].editor.isEnabled()
    finally:
        widget._exit_run_state(widget._new_projection_receipt())
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_run_exit_unlocks_focused_editor(qapp, monkeypatch):
    """A focused V2 line edit must not block the run-exit rebuild/unlock."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget._refresh_controls_v2_profile_now()
        rows = {
            row.path: row
            for row in widget.controls_v2.processing_card.body.findChildren(FormRow)
        }
        editor = rows[("Int1D", "points")].editor
        editor.setFocus()
        assert widget.controls_v2.focusWidget() is editor
        assert editor.isEnabled()

        widget._enter_run_state()
        assert not editor.isEnabled()

        widget._exit_run_state(widget._new_projection_receipt())

        rows = {
            row.path: row
            for row in widget.controls_v2.processing_card.body.findChildren(FormRow)
        }
        assert rows[("Int1D", "points")].editor.isEnabled()
        assert widget._controls_v2_pending_editor is None
    finally:
        widget._exit_run_state(widget._new_projection_receipt())
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
        _user_types(qapp, widget, rows[("Int1D", "points")].editor, "543")
        assert widget.integratorTree.ui.npts_1D.text() != "543"

        def _fake_advanced():
            calls.append("advanced")
            assert widget.integratorTree.ui.npts_1D.text() != "543"
            assert widget.scan.bai_1d_args["numpoints"] == 543

        monkeypatch.setattr(widget, "_show_integration_advanced", _fake_advanced)
        widget._on_controls_v2_action(ControlAction.ADVANCED_PROCESSING)

        assert calls == ["advanced"]
    finally:
        widget.close()
        widget.deleteLater()


def test_controls_panel_v2_range_labels_follow_native_axis_state(
        qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
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
        assert "Q (Å⁻¹)" in labels
        assert "χ (°)" in labels
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
        edit_paths = [edit.path for edit in widget.controls_v2.current_form_edits()]
        assert ("Int1D", "radial_auto") in edit_paths
        assert ("Int1D", "radial_low") not in edit_paths
        assert ("Int1D", "radial_high") not in edit_paths

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


def test_controls_panel_v2_auto_stays_visibly_checked_while_run_locked(
        qapp, monkeypatch):
    """The range toggle remains visibly on while disabled during processing.

    LV-UI-1 semantics: checked now means EXPLICIT bounds (model
    ``radial_auto=False``).  The logical checked bit already survived the run
    lock, but the generic ``:disabled`` QSS rule painted the compact
    QToolButton like an unchecked control.  Require the more-specific
    checked+disabled rule in both themes.
    """
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    from xdart.gui.tabs.static_scan.ui.controls_panel_v2 import RangeRow
    from xdart.gui.themes import render_qss

    widget = staticWidget()
    try:
        widget._on_controls_v2_field_changed(("Int1D", "radial_auto"), False)
        widget._refresh_controls_v2_profile_now()
        widget._enter_run_state()

        ranges = {
            row._low_path: row
            for row in widget.controls_v2.processing_card.body.findChildren(RangeRow)
        }
        auto_button = ranges[("Int1D", "radial_low")]._toggle[1]
        assert auto_button.isChecked()
        assert not auto_button.isEnabled()
        selector = "QToolButton#controlsV2AutoButton:checked:disabled"
        assert selector in render_qss("dark")
        assert selector in render_qss("light")
    finally:
        widget._exit_run_state(widget._new_projection_receipt())
        widget.close()
        widget.deleteLater()



def test_apply_state_update_refuses_fast_path_when_fields_appear(qapp):
    """LV-UI-5: Standard→Grazing ADDS the θ-motor field; the in-place fast
    path must refuse (keys changed) so the full render mounts the new row —
    production falls back to ``set_state`` exactly as the shell does."""
    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.session.run_configuration import RunIntent
    from xdart.gui.tabs.scattering.controls_inventory import GI_MOTOR
    from xdart.gui.tabs.scattering.controls_projection import project_controls
    from xdart.gui.tabs.scattering.state_machine import RunPhase

    panel = ControlsPanelV2()
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


def test_range_toggle_follows_threshold_style_manual_semantics(qapp):
    """LV-UI-1: untoggled = auto range; toggled = the explicit input bounds.

    The MODEL field stays ``*_auto`` — the inversion lives only at the
    presentation seam, mirroring Threshold's enable toggle."""
    from xdart.gui.tabs.static_scan.ui.controls_panel_v2 import RangeRow

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
        assert not btn.isChecked()                     # auto -> untoggled
        assert (("Int1D", "radial_auto"), True) in row.current_edits()
        btn.setChecked(True)                           # explicit bounds ON
        assert emitted == [(("Int1D", "radial_auto"), False)]
        assert (("Int1D", "radial_auto"), False) in row.current_edits()
    finally:
        row.close()
        row.deleteLater()

def test_controls_panel_v2_pending_manual_range_survives_run_commit(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget._on_controls_v2_field_changed(("Int1D", "radial_auto"), False)
        widget._refresh_controls_v2_profile_now()
        ranges = {
            row._low_path: row
            for row in widget.controls_v2.processing_card.body.findChildren(RangeRow)
        }
        row = ranges[("Int1D", "radial_low")]
        _user_types(qapp, widget, row._low, "0.25")
        _user_types(qapp, widget, row._high, "4.25")

        widget._commit_controls_v2_pending_edits()

        assert widget.scan.bai_1d_args["radial_range"] == (0.25, 4.25)
        assert widget._controls_v2_native_reduction_plan(
            integrate_2d=False,
            commit_pending=False,
        ).integration_1d.radial_range == (0.25, 4.25)
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

        # A real ``imageThread`` is a QThread, so it always answers the
        # run-owner admission probe.  T-3.1 refuses a Start when a PRESENT
        # owner cannot be observed, so this double must be faithful here or
        # it would stand in for a broken Qt owner rather than an idle one.
        def isRunning(self):
            return False

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

        widget._prepare_controls_v2_run_configuration()
        _admit_pending_run_configuration(widget)
        widget.start_wrangler()

        assert seen_at_disable == [True]
        assert widget._controls_v2_threshold_config().mask_saturation is True
    finally:
        widget._exit_run_state(widget._new_projection_receipt())
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


# ── SW-7 (§10): wrangler swap resyncs the θ-motor choices to the owner ─────


def test_wrangler_swap_resyncs_gi_motor_choices_sw7(qapp, monkeypatch):
    """SW-7 GUARD (§10 fork class): the θ-motor CHOICES list is owned by the
    ACTIVE wrangler's motor discovery; the integrator combo (and the V2
    dropdown reading it) is a mirror.  The regression: after a wrangler swap
    the stale non-Manual combo list WON over the new wrangler's already-
    discovered motors (the `_gi_motor_choices` fallback only engaged on a
    Manual-only combo), so the V2 dropdown kept offering the previous source's
    motor names until the next discovery emit happened to fire."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        integrator = widget.integratorTree
        # The previous source's discovery left ITS motors in the combo (this is
        # the real handshake target the wrangler signal drives).
        integrator.set_gi_motor_options(["old_th", "detx"])
        stale = widget._controls_v2_native_int_choices()[("GI", "th_motor")]
        assert "old_th" in stale

        # The image wrangler has already discovered its own motors; swapping to
        # it must re-announce them over the stale list through the production
        # sigGIMotorOptions path.
        wr = widget.ui.wranglerStack.widget(0)
        wr.motors = ["halpha", "hy"]
        widget.set_wrangler(0)
        choices = widget._controls_v2_native_int_choices()[("GI", "th_motor")]
        assert "old_th" not in choices, (
            f"stale previous-source motors survived the wrangler swap: {choices}")
        assert "halpha" in choices and "Manual" in choices
        items = [integrator.ui.gi_motor.itemText(i)
                 for i in range(integrator.ui.gi_motor.count())]
        assert items == ["Manual", "halpha", "hy"]

        # An EMPTY discovery must not wipe: session-restored choices stay until
        # this wrangler's own discovery fires.
        wr.motors = []
        integrator.set_gi_motor_options(["restored_th"])
        widget.set_wrangler(0)
        keep = widget._controls_v2_native_int_choices()[("GI", "th_motor")]
        assert "restored_th" in keep
    finally:
        widget.close()
        widget.deleteLater()


# ── SW-8 (§10): Axis edits keep the gi_config mode copy in sync ─────────────


def test_axis_edit_keeps_gi_config_mode_in_sync_sw8(qapp, monkeypatch):
    """SW-8 GUARD (§10 fork class): gi_mode lives authoritatively in
    ``scan.bai_*_args``; ``scan.gi_config`` carries a persisted COPY that
    ``build_reduction_config`` feeds verbatim into written provenance.  The
    regression: an Axis edit updated the authority but never re-stamped the
    copy — the ONLY gi_mode edit path that didn't — so recorded provenance
    (and anything reading gi_config's mode) contradicted the args that
    actually integrate."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.integrator import GI_LABELS_1D, GI_LABELS_2D
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget._on_controls_v2_field_changed(("GI", "Grazing"), True)
        assert widget.scan.gi_config["gi_mode_1d"] == "q_total"

        widget._on_controls_v2_field_changed(
            ("Int1D", "axis"), GI_LABELS_1D[1])          # Qip -> q_ip
        widget._on_controls_v2_field_changed(
            ("Int2D", "axis"), GI_LABELS_2D[1])          # Q-χ -> q_chi
        assert widget.scan.bai_1d_args["gi_mode_1d"] == "q_ip"
        assert widget.scan.bai_2d_args["gi_mode_2d"] == "q_chi"
        # The persisted copy followed the authority...
        assert widget.scan.gi_config["gi_mode_1d"] == "q_ip"
        assert widget.scan.gi_config["gi_mode_2d"] == "q_chi"
        # ...and the recorded display provenance cannot contradict itself.
        cfg = staticWidget._scan_data_reduction_config_snapshot(widget.scan)
        assert (cfg["gi_config"]["gi_mode_1d"]
                == cfg["bai_1d_args"]["gi_mode_1d"] == "q_ip")
        assert (cfg["gi_config"]["gi_mode_2d"]
                == cfg["bai_2d_args"]["gi_mode_2d"] == "q_chi")
    finally:
        _reset_controls_v2_gi(widget)
        widget.close()
        widget.deleteLater()


# ── POL-1: fresh-scan polarization default ON at 0.99 ──────────────────────


def test_fresh_scan_polarization_defaults_on_at_099(qapp, monkeypatch):
    """POL-1 (maintainer decision, 2026-07-13): fresh scans apply the
    polarization correction by default at DEFAULT_POLARIZATION_FACTOR — SSRL
    beams are strongly horizontally polarized.  Deliberate OFF still encodes
    as ``polarization_factor=None`` (the S10-1 contract), and re-enabling
    lands the default factor, not 0.0."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.integrator import (
        DEFAULT_POLARIZATION_FACTOR,
    )
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        integrator = widget.integratorTree
        # The real dialog-open path seeds + hydrates the trees from scan args.
        widget._show_integration_advanced()
        a1 = widget.scan.bai_1d_args
        a2 = widget.scan.bai_2d_args
        assert a1["polarization_factor"] == pytest.approx(
            DEFAULT_POLARIZATION_FACTOR)
        assert a2["polarization_factor"] == pytest.approx(
            DEFAULT_POLARIZATION_FACTOR)
        assert integrator.bai_1d_pars.child(
            "Apply polarization factor").value() is True
        assert integrator.bai_1d_pars.child(
            "polarization_factor").value() == pytest.approx(
                DEFAULT_POLARIZATION_FACTOR)
        assert integrator.bai_2d_pars.child(
            "Apply polarization factor").value() is True

        # Deliberate OFF encodes as None (and survives — the S10-1 guard).
        widget._on_controls_v2_field_changed(
            ("Int1D", "apply_polarization"), False)
        assert a1["polarization_factor"] is None
        # Re-enabling restores the default factor, not 0.0.
        widget._on_controls_v2_field_changed(
            ("Int1D", "apply_polarization"), True)
        assert a1["polarization_factor"] == pytest.approx(
            DEFAULT_POLARIZATION_FACTOR)
    finally:
        widget.close()
        widget.deleteLater()


# ── Live chip + waiting status in the readiness bar (2026-07-13) ────────────


def test_readiness_bar_live_chip_and_waiting_status(qapp, monkeypatch):
    """Maintainer (2026-07-13): the readiness bar shows a bold colored 'Live'
    chip while the Live toggle is on, and reads "Waiting for new images…"
    while a live run idles between files (the thread's 'Watching for new
    files...' status), restoring the normal summary on any other status."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        controls = widget.controls
        assert controls.readinessLive.isHidden()

        # Toggling Live on re-renders the summary with the chip; off hides it.
        controls.liveButton.setChecked(True)
        assert not controls.readinessLive.isHidden()
        controls.liveButton.setChecked(False)
        assert controls.readinessLive.isHidden()

        # Live watch idling: the thread's watching status flips the bar to the
        # waiting text (chip on); any other status restores the summary.
        controls.liveButton.setChecked(True)
        monkeypatch.setattr(widget, "_controls_v2_run_active", lambda: True)
        widget._on_wrangler_status_text("Watching for new files...")
        assert "Waiting for new images" in controls.readinessLabel._full_text
        assert not controls.readinessLive.isHidden()

        widget._on_wrangler_status_text("frame_0001.tif")
        assert "Waiting for new images" not in controls.readinessLabel._full_text
    finally:
        widget.close()
        widget.deleteLater()


# ── DIR-1: the outgoing scan's frame list paints before the boundary clear ──


def test_frame_boundary_paints_outgoing_list_before_rescope_dir1(
        qapp, monkeypatch):
    """DIR-1 GUARD (bl17-2): per-frame updates only ARM the throttled list
    timer (no leading-edge fire), so with back-to-back short container scans
    the frame-driven boundary used to clear ``scan.frames.index`` before the
    outgoing scan's tail ever painted — the Frames panel showed 1-2 of 5
    frames and jumped to the next scan.  The boundary must run the LIGHT list
    flush for the outgoing scan (all 5 rows, one paint) before rescoping."""
    from types import MethodType, SimpleNamespace

    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    from xdart.modules.live import LiveFrame

    widget = staticWidget()
    try:
        # Scan A processed 5 frames; the throttled list flush never fired, so
        # the h5viewer list is stale (this is the burst-inside-one-window
        # state the boundary races).
        widget.scan.name = "scanA"
        with widget.scan.scan_lock:
            widget.scan.frames.index.extend([1, 2, 3, 4, 5])

        paints = []
        real_update = widget.h5viewer.update_data

        def recording_update(self_h5, *a, **k):
            paints.append((widget.scan.name,
                           len(widget.scan.frames.index)))
            return real_update(*a, **k)

        widget.h5viewer.update_data = MethodType(recording_update,
                                                 widget.h5viewer)

        # Scan B's first frame arrives (the frame-driven boundary).
        frame_b = LiveFrame(idx=1)
        frame_b.source_file = "/data/scanB_00001.nxs"
        frame_b.source_frame_idx = 0
        widget.wrangler.thread._published_frames = {1: frame_b}
        widget.wrangler.thread.batch_mode = False

        widget.update_data(1)

        # The OUTGOING scan's complete list painted BEFORE the rescope...
        assert ("scanA", 5) in paints, (
            f"outgoing scan's full list never painted; paints={paints}")
        # ...and the panel then rescoped to the new scan.
        assert widget.scan.name.startswith("scanB")
    finally:
        widget.close()
        widget.deleteLater()


def test_gi_mirror_no_reentry_no_intent_write(qapp, monkeypatch):
    """R4B-9 + P2 (O-1a-i.1 item 12): the run-boundary GI mirror is a pure
    hidden-output adapter.  It changes ONLY the intended hidden tree value and
    must not fire the ROOT tree signal (which drives ``imageWrangler.setup()``
    and the legacy session write), fire the child θ-motor handlers (which write
    ``wrangler.incidence_motor``), change the Controls-owned ``RunIntent``
    identity, or change the wrangler's engine carriers.  The root signal is
    observed DIRECTLY rather than by monkeypatching an already-connected
    ``setup`` slot (which would not replace the bound connection)."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        wrangler = widget.wrangler
        params = wrangler.parameters
        th_motor = params.child("GI", "th_motor")
        # Offer a real motor so the mirror actually CHANGES the value (a no-op
        # setValue would not emit even without the block).
        th_motor.setOpts(limits=["Manual", "halpha"], value="Manual")
        widget._on_controls_v2_field_changed(("GI", "Grazing"), True)
        widget._on_controls_v2_field_changed(("GI", "th_motor"), "halpha")

        # Observe the ROOT signal directly — it owns _setup_on_value_change
        # (-> setup()) and _save_to_session; silence here reliably proves
        # neither ran.  Plus the child θ-motor handlers (R4B-9).
        root_fires = []
        params.sigTreeStateChanged.connect(lambda *a: root_fires.append(1))
        child_fires = []
        th_motor.sigValueChanged.connect(lambda *a: child_fires.append(1))
        params.child("GI", "th_val").sigValueChanged.connect(
            lambda *a: child_fires.append(1))

        # fingerprint EXCLUDES generation, so it is stable across freeze() calls
        # unless the intent CONTENT changes.
        before_fingerprint = (
            widget._controls_v2_ensure_run_intent().freeze().fingerprint)
        before_motor = getattr(wrangler, "incidence_motor", None)
        before_gi = getattr(wrangler, "gi", None)

        frozen = widget._controls_v2_ensure_run_intent().freeze()
        widget._push_gi_to_wrangler(frozen)

        # Intended hidden output changed...
        assert th_motor.value() == "halpha"
        # ...and NOTHING else:
        assert root_fires == [], (
            "GI mirror fired the root tree signal (setup()/session write risk)")
        assert child_fires == [], (
            "GI mirror fired child θ-motor handlers (incidence_motor write-back)")
        after_fingerprint = (
            widget._controls_v2_ensure_run_intent().freeze().fingerprint)
        assert after_fingerprint == before_fingerprint, (
            "GI mirror changed the Controls-owned RunIntent content")
        assert getattr(wrangler, "incidence_motor", None) == before_motor
        assert getattr(wrangler, "gi", None) == before_gi
    finally:
        widget.close()
        widget.deleteLater()


def test_no_carrier_mutation_between_run_end_and_start(qapp, monkeypatch):
    """R4B-12 [b] (O-1a-i.3): with the timer/owner machinery deleted, NO carrier
    mutates between run-end and the next Start — deferred edits are a pure data
    delta until the fold inside preparation.  Red at d13dc545: the finish-handler
    singleShot owner applied edits at run-end (the property that kills the whole
    trial-write/timer class)."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        params = widget.wrangler.parameters
        mask_param = params.child("Signal", "mask_file")
        mask_before = mask_param.value()

        widget._enter_run_state()
        widget._on_controls_v2_field_changed(("GI", "Grazing"), True)
        widget._on_controls_v2_field_changed(("Signal", "mask_file"), "/tmp/x.edf")
        # End the run through the production finish slot, then pump the loop.
        widget.wrangler.finished.emit()
        qapp.processEvents()

        # NOTHING applied between run-end and Start:
        assert widget._controls_v2_ensure_run_intent().gi.enabled is False
        assert mask_param.value() == mask_before
        queued = [tuple(p) for p, _ in widget._controls_v2_deferred_field_edits]
        assert ("GI", "Grazing") in queued and ("Signal", "mask_file") in queued
    finally:
        widget.close()
        widget.deleteLater()


def test_deferred_edits_fold_into_intent_and_carriers_at_next_start(
        qapp, monkeypatch):
    """R4B-12 (O-1a-i.3): deferred edits (native GI + legacy mask) fold into the
    intent + hidden carriers at the next Start's preparation, and the frozen
    config reflects them.  'queued' notice, not 'saved'."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        notices = []
        widget.wrangler.showLabel.connect(lambda m: notices.append(str(m)))
        params = widget.wrangler.parameters
        mask_param = params.child("Signal", "mask_file")
        target = "/tmp/fold_mask.edf"

        widget._enter_run_state()
        widget._on_controls_v2_field_changed(("GI", "Grazing"), True)
        widget._on_controls_v2_field_changed(("Signal", "mask_file"), target)
        assert notices and "queued" in notices[-1].lower()
        assert "saved" not in notices[-1].lower()
        widget.wrangler.finished.emit()
        qapp.processEvents()
        assert widget._controls_v2_ensure_run_intent().gi.enabled is False
        assert mask_param.value() != target

        frozen = widget._prepare_controls_v2_run_configuration()
        assert frozen is not None
        assert frozen.gi.enabled is True
        assert mask_param.value() == target
        assert list(widget._controls_v2_deferred_field_edits) == []
    finally:
        widget.close()
        widget.deleteLater()


def test_fast_start_folds_deferred_values_before_publish(qapp, monkeypatch):
    """R4B-12 [a] (O-1a-i.3): defer edits during a run, Start immediately at run
    end (no event tick), assert the new frozen config already contains the folded
    values — the fold is synchronous inside preparation, so no window exists in
    which a frozen config publishes while deferred edits are unresolved."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget._enter_run_state()
        widget._on_controls_v2_field_changed(("GI", "Grazing"), True)
        # The run ends; a fast next Start receives no intervening event tick.
        widget._exit_run_state(widget._new_projection_receipt())

        intent = widget._controls_v2_ensure_run_intent()
        gen_before = intent.generation
        frozen = widget._prepare_controls_v2_run_configuration()
        assert frozen is not None
        assert frozen.gi.enabled is True
        assert list(widget._controls_v2_deferred_field_edits) == []
        # Preparation freezes a detached candidate without advancing the
        # canonical intent.  The post-admission owner commits that generation
        # exactly once.
        assert frozen.generation == gen_before + 1
        assert intent.generation == gen_before
        widget._apply_controls_v2_run_state(frozen)
        assert intent.generation == gen_before + 1
    finally:
        widget.close()
        widget.deleteLater()


def test_invalid_fold_aborts_prep_nothing_frozen(qapp, monkeypatch, caplog):
    """R4B-12 [c] (O-1a-i.3): a deferred legacy edit that cannot land aborts
    preparation at the fold — typed visible error, structured event, NO frozen
    config, full delta retained.  The i.3 fold has no separate validation/commit
    phase, so the i.2 silent-loss class cannot recur."""
    import logging

    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    monkeypatch.setenv("XDART_RUN_CONFIG_DEBUG", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        DeferredRunEditsPendingError,
        staticWidget,
    )

    widget = staticWidget()
    try:
        widget._pending_controls_v2_run_configuration = None
        params = widget.wrangler.parameters
        mask_param = params.child("Signal", "mask_file")

        widget._enter_run_state()
        widget._on_controls_v2_field_changed(("GI", "Grazing"), True)
        widget._on_controls_v2_field_changed(("Signal", "mask_file"), "/tmp/f.edf")
        monkeypatch.setattr(mask_param, "setValue", lambda *a, **k: None)
        widget.wrangler.finished.emit()

        with caplog.at_level(logging.INFO):
            with pytest.raises(DeferredRunEditsPendingError):
                widget._prepare_controls_v2_run_configuration()

        assert getattr(
            widget, "_pending_controls_v2_run_configuration", None) is None
        remaining = [
            tuple(p) for p, _ in widget._controls_v2_deferred_field_edits]
        assert ("Signal", "mask_file") in remaining, remaining
        assert "controls_deferred_fold_invalid" in caplog.text
    finally:
        widget.close()
        widget.deleteLater()


def test_deferred_source_edit_reconciles_source_index(qapp, monkeypatch):
    """R4B-12 [d] (O-1a-i.3): a deferred SOURCE edit reconciles the source index
    at the fold, exactly like a live idle source edit.  Red at d13dc545: the i.2
    owner never called _sync_controls_v2_source_index."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        calls = []
        orig = widget._sync_controls_v2_source_index
        monkeypatch.setattr(
            widget, "_sync_controls_v2_source_index",
            lambda: calls.append(1))

        widget._enter_run_state()
        widget._on_controls_v2_field_changed(("Signal", "include_subdir"), True)
        widget.wrangler.finished.emit()
        qapp.processEvents()
        calls.clear()  # only count reconciliation at the fold

        widget._controls_v2_fold_deferred_edits_into_intent()
        assert calls, (
            "deferred source edit did not reconcile the source index at fold")
    finally:
        widget.close()
        widget.deleteLater()


def test_deferred_multi_field_folds_together_once(qapp, monkeypatch):
    """R4B-12 (O-1a-i.3, pin): multiple deferred fields fold together atomically
    by construction (pure data), and a subsequent Run freezes once."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        params = widget.wrangler.parameters
        mask_param = params.child("Signal", "mask_file")
        target = "/tmp/multi_fold.edf"

        widget._enter_run_state()
        widget._on_controls_v2_field_changed(("GI", "Grazing"), True)
        widget._on_controls_v2_field_changed(("Signal", "mask_file"), target)
        widget.wrangler.finished.emit()
        qapp.processEvents()

        intent = widget._controls_v2_ensure_run_intent()
        gen_before = intent.generation
        frozen = widget._prepare_controls_v2_run_configuration()
        assert frozen.gi.enabled is True
        assert mask_param.value() == target
        assert list(widget._controls_v2_deferred_field_edits) == []
        assert frozen.generation == gen_before + 1
        assert intent.generation == gen_before
        widget._apply_controls_v2_run_state(frozen)
        assert intent.generation == gen_before + 1
    finally:
        widget.close()
        widget.deleteLater()


def test_deferred_edit_survives_finish_exception_and_folds_next_start(
        qapp, monkeypatch):
    """R4B-12 (O-1a-i.3): a deferred edit is never stranded by a finish-body
    exception — with no timer/owner it stays a pure data delta and folds at the
    next Start regardless of any exception on the run-exit path."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget._enter_run_state()
        widget._on_controls_v2_field_changed(("GI", "Grazing"), True)

        boom = [True]

        def _maybe_boom(*a, **k):
            if boom[0]:
                raise RuntimeError("injected finish-path failure")

        monkeypatch.setattr(widget.h5viewer, "set_run_writing", _maybe_boom)
        try:
            widget._exit_run_state(widget._new_projection_receipt())  # run-exit path raises
        except RuntimeError:
            pass
        # The edit survives as a queued delta despite the exit-path exception.
        assert list(widget._controls_v2_deferred_field_edits)
        assert widget._controls_v2_ensure_run_intent().gi.enabled is False

        # Clear the injected failure and the stranded run latch, then fold.
        boom[0] = False
        widget._run_active = False
        frozen = widget._prepare_controls_v2_run_configuration()
        assert frozen is not None
        assert frozen.gi.enabled is True
        assert list(widget._controls_v2_deferred_field_edits) == []
    finally:
        widget.close()
        widget.deleteLater()



def test_t1_stage_controls_transaction_is_pure_no_carrier_writes(qapp, monkeypatch):
    """T-1 (§9.10 step 3-stage): the pure stage writes NO production carrier.

    A native GI edit and a legacy mask edit are staged; the returned
    StagedControlsTransaction carries the candidate values, but the LIVE intent
    (object identity + generation), the hidden Qt params, and the display scan
    are all unchanged.  Staging leaves no trace of its redirect/side-effect
    suppression."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        StagedControlsTransaction,
        staticWidget,
    )

    widget = staticWidget()
    try:
        intent = widget._controls_v2_ensure_run_intent()
        gi_before = intent.gi.enabled
        gen_before = intent.generation
        mask_param = widget.wrangler.parameters.child("Signal", "mask_file")
        mask_before = mask_param.value()
        scan = getattr(widget, "scan", None)
        gi_scan_before = (
            bool(getattr(scan, "gi", False)) if scan is not None else None)

        staged = widget.stage_controls_transaction([
            (("GI", "Grazing"), True),
            (("Signal", "mask_file"), "/tmp/t1_pure_stage.edf"),
        ])

        assert isinstance(staged, StagedControlsTransaction)
        # The candidate carries the values...
        assert staged.staged_intent.gi.enabled is True
        assert staged.legacy_projection[("Signal", "mask_file")] == (
            "/tmp/t1_pure_stage.edf")
        # ...but NOTHING production changed.
        assert widget._controls_v2_ensure_run_intent() is intent  # identity kept
        assert intent.gi.enabled == gi_before
        assert intent.generation == gen_before
        assert mask_param.value() == mask_before
        if scan is not None:
            assert bool(getattr(scan, "gi", False)) == gi_scan_before
        assert getattr(widget, "_controls_v2_staging", False) is False
    finally:
        widget.close()
        widget.deleteLater()


def test_t1_stage_rejects_invalid_coercion_no_freeze(qapp, monkeypatch):
    """T-1 (§9.10 test 2): an invalid numeric on a legacy-backed field (BG.Scale)
    is a TYPED staging failure — not a silently-coerced 'landed' value.  A
    deferred invalid edit aborts preparation BEFORE any freeze, retains the
    journal, and surfaces DeferredRunEditsPendingError with a structured event.

    RED at 10e7b630: the i.3 fold classified an un-coercible legacy value as
    'landed' (coerce raised -> _controls_v2_deferred_field_landed returned True)
    and froze anyway, silently dropping the operator's edit (§9.3)."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    monkeypatch.setenv("XDART_RUN_CONFIG_DEBUG", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        DeferredRunEditsPendingError,
        staticWidget,
    )

    widget = staticWidget()
    try:
        widget._pending_controls_v2_run_configuration = None
        widget._enter_run_state()
        widget._on_controls_v2_field_changed(("BG", "Scale"), "not-a-number")
        widget.wrangler.finished.emit()
        qapp.processEvents()

        with pytest.raises(DeferredRunEditsPendingError):
            widget._prepare_controls_v2_run_configuration()

        assert getattr(
            widget, "_pending_controls_v2_run_configuration", None) is None
        remaining = [
            tuple(p) for p, _ in widget._controls_v2_deferred_field_edits]
        assert ("BG", "Scale") in remaining, remaining
    finally:
        widget.close()
        widget.deleteLater()


def test_t1_journal_lww_idle_edit_supersedes_older_deferred(qapp, monkeypatch):
    """T-1 (§9.10 test 5, clause 1): a newer IDLE edit supersedes an older
    DEFERRED value for the SAME path, by revision.  Run-active GI on, then idle
    GI off -> the next Start freezes GI OFF.

    RED at 10e7b630: the deferred queue and the idle edit were separate
    containers; the deferred value was folded AFTER the panel harvest and won
    regardless of the newer idle correction (§9.4)."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        widget._enter_run_state()
        widget._on_controls_v2_field_changed(("GI", "Grazing"), True)   # deferred A
        widget.wrangler.finished.emit()
        qapp.processEvents()
        # Newer idle correction B for the same path.
        widget._on_controls_v2_field_changed(("GI", "Grazing"), False)

        frozen = widget._prepare_controls_v2_run_configuration()
        assert frozen is not None
        assert frozen.gi.enabled is False   # newest edit (idle B) wins over A
    finally:
        widget.close()
        widget.deleteLater()


def test_t1_journal_lww_bad_deferred_recovered_by_idle_correction(
        qapp, monkeypatch):
    """T-1 (§9.10 test 5, clause 2): a bad deferred edit that fails Start is
    RECOVERABLE through the ordinary UI — an idle correction supersedes it (by
    revision) and the next Start succeeds with the corrected value.

    RED at 10e7b630: an un-coercible deferred value was silently 'landed', so the
    FIRST Start already froze (dropping the edit) and never refused — the
    recovery invariant could not even be exercised."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        DeferredRunEditsPendingError,
        staticWidget,
    )

    widget = staticWidget()
    try:
        widget._pending_controls_v2_run_configuration = None
        widget._enter_run_state()
        widget._on_controls_v2_field_changed(("BG", "Scale"), "not-a-number")
        widget.wrangler.finished.emit()
        qapp.processEvents()

        with pytest.raises(DeferredRunEditsPendingError):
            widget._prepare_controls_v2_run_configuration()   # refuses on bad A

        # Correct the field through the ordinary idle edit path (BG.Scale is an
        # integer-backed param, so use an integer correction value).
        widget._on_controls_v2_field_changed(("BG", "Scale"), 3)

        frozen = widget._prepare_controls_v2_run_configuration()
        assert frozen is not None
        param = widget.wrangler.parameters.child("BG", "Scale")
        assert int(param.value()) == 3
    finally:
        widget.close()
        widget.deleteLater()


def test_t1_invalid_second_field_leaves_earlier_native_field_unmutated(
        qapp, monkeypatch):
    """T-1 partial-mutation probe: a multi-field transaction whose SECOND field is
    invalid leaves the FIRST (native GI) field UNMUTATED — the pure stage
    validates the complete delta before ANY carrier/intent write, so no earlier
    field half-applies.

    RED at 10e7b630: the i.3 fold applied fields sequentially; GI landed on the
    live intent before the invalid BG.Scale was reached, leaving GI enabled while
    the fold reported success (§9.2)."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        DeferredRunEditsPendingError,
        staticWidget,
    )

    widget = staticWidget()
    try:
        assert widget._controls_v2_ensure_run_intent().gi.enabled is False
        widget._pending_controls_v2_run_configuration = None
        widget._enter_run_state()
        widget._on_controls_v2_field_changed(("GI", "Grazing"), True)
        widget._on_controls_v2_field_changed(("BG", "Scale"), "not-a-number")
        widget.wrangler.finished.emit()
        qapp.processEvents()

        with pytest.raises(DeferredRunEditsPendingError):
            widget._prepare_controls_v2_run_configuration()

        # The earlier native GI field was NOT applied to the live intent.
        assert widget._controls_v2_ensure_run_intent().gi.enabled is False
        assert getattr(
            widget, "_pending_controls_v2_run_configuration", None) is None
    finally:
        widget.close()
        widget.deleteLater()


def test_t1_gi_freeze_honors_explicit_motor_when_choices_unknown(
        qapp, monkeypatch):
    """T-1 item-5 ()-vs-None escape: when the GI θ-motor dropdown is NOT populated
    from a source, _prepare passes None (never ()) to freeze, so an explicit saved
    motor is HONORED — a motor GI run never silently degrades to a fixed-angle
    Manual run (display/frozen divergence, review F8)."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        # No source loaded: the θ-motor dropdown carries no real choices, so the
        # freeze helper reports 'unknown' (None), not empty-and-known (()).
        assert widget._controls_v2_gi_motor_choices_for_freeze() is None

        widget._on_controls_v2_field_changed(("GI", "Grazing"), True)
        widget._on_controls_v2_field_changed(("GI", "th_motor"), "halpha")

        frozen = widget._prepare_controls_v2_run_configuration()
        assert frozen is not None
        assert frozen.gi.enabled is True
        assert frozen.gi.incidence_motor == "halpha"     # raw retained
        assert frozen.gi.effective_motor == "halpha"     # honored, NOT Manual
    finally:
        widget.close()
        widget.deleteLater()


def _t1r_widget(monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    return staticWidget()


def test_t1r_source_edit_plus_invalid_leaves_all_state_unchanged(qapp, monkeypatch):
    """§12 test 1: staging a source edit followed by an invalid field mutates NO
    production state — source-energy preference, source-energy cache, source
    index/observation, and the display scan are byte/value-identical.

    RED at 4b59b7da: the redirect stage wrote _controls_v2_source_energy_preference
    (C1) and _controls_v2_axis_to_native wrote the display scan gi_config."""
    from xdart.gui.tabs.static_scan.static_scan_widget import ControlsTransactionError
    widget = _t1r_widget(monkeypatch)
    try:
        pref0 = getattr(widget, "_controls_v2_source_energy_preference", "poni")
        cache0 = getattr(widget, "_controls_v2_source_energy_cache", None)
        obs0 = getattr(widget, "_controls_v2_directory_observation", None)
        scan = getattr(widget, "scan", None)
        gi_cfg0 = copy.deepcopy(getattr(scan, "gi_config", None)) if scan else None
        gi0 = bool(getattr(scan, "gi", False)) if scan else None

        result = widget.stage_controls_transaction([
            (("Source", "energy_preference"), "metadata"),
            (("GI", "Grazing"), True),
            (("Int1D", "axis"), "Qip"),
            (("BG", "Scale"), "not-a-number"),
        ])
        assert isinstance(result, ControlsTransactionError)
        assert getattr(widget, "_controls_v2_source_energy_preference", "poni") == pref0
        assert getattr(widget, "_controls_v2_source_energy_cache", None) == cache0
        assert getattr(widget, "_controls_v2_directory_observation", None) is obs0
        if scan is not None:
            assert copy.deepcopy(getattr(scan, "gi_config", None)) == gi_cfg0
            assert bool(getattr(scan, "gi", False)) == gi0
    finally:
        widget.close()
        widget.deleteLater()


def test_t1r_unsupported_path_is_typed_refusal(qapp, monkeypatch):
    """§12 test 2: an unsupported control path is a typed ControlsTransactionError,
    not a silent skip, and the winning journal is retained.

    RED at 4b59b7da: the stage loop `continue`d past a param-less path."""
    from xdart.gui.tabs.static_scan.static_scan_widget import ControlsTransactionError
    widget = _t1r_widget(monkeypatch)
    try:
        result = widget.stage_controls_transaction([
            (("Nonexistent", "field"), "x"),
        ])
        assert isinstance(result, ControlsTransactionError)
        assert result.path == ("Nonexistent", "field")
    finally:
        widget.close()
        widget.deleteLater()


def test_t1r_invalid_native_numeric_is_typed_refusal(qapp, monkeypatch):
    """§12 test 3: an invalid native numeric (Int1D/points='not-a-number') is a
    typed refusal, not a permissive clamp to a default.

    RED at 4b59b7da: the native setter clamped invalid input to a default and
    returned a successful StagedControlsTransaction."""
    from xdart.gui.tabs.static_scan.static_scan_widget import ControlsTransactionError
    widget = _t1r_widget(monkeypatch)
    try:
        result = widget.stage_controls_transaction([
            (("Int1D", "points"), "not-a-number"),
        ])
        assert isinstance(result, ControlsTransactionError)
        assert result.path == ("Int1D", "points")
    finally:
        widget.close()
        widget.deleteLater()


def test_t1r_threshold_min_gt_max_is_typed_refusal(qapp, monkeypatch):
    """§12 test 4: a cross-field contradiction (threshold min>max) is a typed
    staging refusal that leaves the live intent unchanged and retains the journal
    — freeze is NOT the error boundary.

    RED at 4b59b7da: min>max staged and committed, installed an invalid live
    intent and cleared the journal; only RunIntent.freeze() raised later."""
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        ControlsTransactionError, DeferredRunEditsPendingError)
    widget = _t1r_widget(monkeypatch)
    try:
        # Pure stage: typed refusal, no install.
        result = widget.stage_controls_transaction([
            (("Mask", "min"), 100.0),
            (("Mask", "max"), 10.0),
        ])
        assert isinstance(result, ControlsTransactionError)

        # Through the production Start seam: refuse, retain journal, nothing frozen.
        widget._pending_controls_v2_run_configuration = None
        intent = widget._controls_v2_ensure_run_intent()
        min_before = intent.threshold.threshold_min
        widget._enter_run_state()
        widget._on_controls_v2_field_changed(("Mask", "min"), 100.0)
        widget._on_controls_v2_field_changed(("Mask", "max"), 10.0)
        widget.wrangler.finished.emit()
        qapp.processEvents()
        with pytest.raises(DeferredRunEditsPendingError):
            widget._prepare_controls_v2_run_configuration()
        assert widget._controls_v2_ensure_run_intent().threshold.threshold_min == min_before
        assert getattr(widget, "_pending_controls_v2_run_configuration", None) is None
        remaining = {tuple(p) for p, _ in widget._controls_v2_deferred_field_edits}
        assert ("Mask", "min") in remaining and ("Mask", "max") in remaining
    finally:
        widget.close()
        widget.deleteLater()


def test_t1r_gi_enable_then_axis_composes_against_staged_state(qapp, monkeypatch):
    """§12 test 5: a GI-enable followed by a GI-axis edit in ONE transaction
    composes against the STAGED GI state — the axis reducer sees gi_enabled and
    sets the GI mode, not a unit.

    RED at 4b59b7da: the axis reducer read the stale display scan (gi=False), so
    the second edit was evaluated as a Standard unit change (q_total retained)."""
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        GI_MODES_1D, StagedControlsTransaction)
    widget = _t1r_widget(monkeypatch)
    try:
        staged = widget.stage_controls_transaction([
            (("GI", "Grazing"), True),
            (("Int1D", "axis"), "Qip"),
        ])
        assert isinstance(staged, StagedControlsTransaction)
        assert staged.staged_intent.gi.enabled is True
        assert staged.staged_intent.bai_1d_args.get("gi_mode_1d") == "q_ip"
        assert "q_ip" in GI_MODES_1D
    finally:
        widget.close()
        widget.deleteLater()


def test_t1r_staging_gi_axis_does_not_mutate_display_scan(qapp, monkeypatch):
    """§12 test 6: staging a GI-axis change does NOT mutate the live display scan
    (the reducer touches only the candidate)."""
    from xdart.gui.tabs.static_scan.static_scan_widget import StagedControlsTransaction
    widget = _t1r_widget(monkeypatch)
    try:
        # Enable GI on the live intent first (idle apply) so the scan reflects GI.
        widget._on_controls_v2_field_changed(("GI", "Grazing"), True)
        scan = getattr(widget, "scan", None)
        gi_cfg0 = copy.deepcopy(getattr(scan, "gi_config", None)) if scan else None
        staged = widget.stage_controls_transaction([
            (("Int1D", "axis"), "Qoop"),
        ])
        assert isinstance(staged, StagedControlsTransaction)
        assert staged.staged_intent.bai_1d_args.get("gi_mode_1d") == "q_oop"
        if scan is not None:
            assert copy.deepcopy(getattr(scan, "gi_config", None)) == gi_cfg0
    finally:
        widget.close()
        widget.deleteLater()


def test_t1r_focused_draft_supersedes_older_deferred_by_revision(qapp, monkeypatch):
    """§12 test 7: an older deferred value followed by a NEWER still-focused real
    form draft selects the newer revision (no journal-beats-form rule).

    RED at 4b59b7da: the form harvest skipped a path already in the journal, so
    the older deferred value won over the newer focused draft."""
    widget = _t1r_widget(monkeypatch)
    try:
        widget._enter_run_state()
        widget._on_controls_v2_field_changed(("BG", "Scale"), 2)   # deferred A (older)
        widget.wrangler.finished.emit()
        qapp.processEvents()
        # Newer focused draft for the same path (idle), via the production slot.
        widget._on_controls_v2_field_draft_changed(("BG", "Scale"), 7)
        winners = dict(widget._controls_v2_collect_pending_edits())
        assert winners.get(("BG", "Scale")) == 7   # newer draft wins by revision
    finally:
        widget.close()
        widget.deleteLater()


def test_t1r_invalid_focused_draft_revisioned_and_survives_rebuild(qapp, monkeypatch):
    """§12 test 8: an invalid focused form draft is revisioned in the journal and
    SURVIVES a panel rebuild/refresh (it is not lost to a re-render).

    RED at 4b59b7da: focused drafts never entered the journal, so a rebuild lost
    the user's uncommitted text."""
    widget = _t1r_widget(monkeypatch)
    try:
        widget._on_controls_v2_field_draft_changed(("BG", "Scale"), "not-a-number")
        journal = widget._controls_v2_edit_journal_dict()
        assert ("BG", "Scale") in journal
        entry = journal[("BG", "Scale")]
        assert entry["value"] == "not-a-number" and entry["origin"] == "draft"
        rev_before = entry["revision"]
        # A panel rebuild/refresh must not drop the journaled draft.
        widget._refresh_controls_v2_profile(immediate=True)
        qapp.processEvents()
        journal2 = widget._controls_v2_edit_journal_dict()
        assert journal2.get(("BG", "Scale"), {}).get("value") == "not-a-number"
        assert journal2[("BG", "Scale")]["revision"] == rev_before
    finally:
        widget.close()
        widget.deleteLater()


def test_t1r_form_harvest_failure_refuses_preparation(qapp, monkeypatch):
    """§12 test 9: a form/draft collection failure REFUSES preparation (fail
    closed), it does not log-and-continue.

    RED at 4b59b7da: a form-harvest exception was caught and preparation
    continued with an empty harvest."""
    from xdart.gui.tabs.static_scan.static_scan_widget import DeferredRunEditsPendingError
    widget = _t1r_widget(monkeypatch)
    try:
        widget._pending_controls_v2_run_configuration = None
        widget._enter_run_state()
        widget._on_controls_v2_field_changed(("GI", "Grazing"), True)
        widget.wrangler.finished.emit()
        qapp.processEvents()

        def _boom():
            raise RuntimeError("injected form harvest failure")
        monkeypatch.setattr(widget.controls_v2, "focused_form_edit", _boom)

        with pytest.raises(DeferredRunEditsPendingError):
            widget._prepare_controls_v2_run_configuration()
        assert getattr(widget, "_pending_controls_v2_run_configuration", None) is None
    finally:
        widget.close()
        widget.deleteLater()


def test_t1r_commit_pending_cannot_bypass_journal(qapp, monkeypatch):
    """§12 test 10 / T-2 finding 1: a non-run form commit (reintegrate/advanced)
    routes through the SAME validated transaction owner, so a NEWER focused
    correction beats an OLDER deferred value by revision and is applied EXACTLY
    ONCE — the permissive live-setter path is retired.

    RED at 4b59b7da: _commit_controls_v2_pending_edits applied the form value
    directly via _apply_controls_v2_field_value without journaling it, so an
    older deferred journal entry won at the next Start."""
    widget = _t1r_widget(monkeypatch)
    try:
        # Older deferred value for BG.Scale (during a run).
        widget._enter_run_state()
        widget._on_controls_v2_field_changed(("BG", "Scale"), 2)   # deferred, older
        widget.wrangler.finished.emit()
        qapp.processEvents()
        # Idle: a newer focused form value published through the reintegrate/
        # advanced harvest seam (the form holds 9 while the carrier still holds the
        # old value); it must beat the older deferred by revision.
        class _Edit:
            def __init__(self, path, value):
                self.path = path
                self.value = value
        monkeypatch.setattr(
            widget.controls_v2, "focused_form_edit",
            lambda: _Edit(("BG", "Scale"), 9))

        bg = widget.wrangler.parameters.child("BG", "Scale")
        applied = []
        orig_set = bg.setValue
        monkeypatch.setattr(
            bg, "setValue",
            lambda v, *a, **k: (applied.append(int(v)), orig_set(v, *a, **k))[1])

        widget._commit_controls_v2_pending_edits()

        # The newer focused value (9) won over the older deferred (2) and was
        # applied through the validated transaction, exactly once — never 2.
        assert int(bg.value()) == 9
        assert applied.count(9) == 1
        assert 2 not in applied
        # The transaction consumed the journal (applied once, not left pending).
        assert widget._controls_v2_edit_journal_dict() == {}
    finally:
        widget.close()
        widget.deleteLater()


def test_t1r_gi_motor_observation_is_source_qualified(qapp, monkeypatch):
    """§12 test 11: motor choices are tied to the SOURCE they were observed from.
    Source A's halpha does not seed source B's freeze; UNKNOWN/KNOWN_EMPTY/
    KNOWN_NONEMPTY map correctly; a delayed stale A signal is ignored under B.

    RED at 4b59b7da: choices came from the persistent combo, so a source A→B edit
    froze B with A's motor, and known-empty was unreachable."""
    from xdart.gui.tabs.static_scan.static_scan_widget import GIMotorObservation
    widget = _t1r_widget(monkeypatch)
    try:
        token = {"v": "A"}
        monkeypatch.setattr(
            widget, "_controls_v2_source_token", lambda: token["v"])

        # Source A: metadata hydration reports halpha -> KNOWN_NONEMPTY for A.
        widget._controls_v2_record_gi_motor_observation(["halpha", "detx"])
        obs_a = widget._controls_v2_capture_gi_motor_observation()
        assert obs_a.state == GIMotorObservation.KNOWN_NONEMPTY
        # choices_for_freeze() is the source's real motor list.  Enabled run
        # admission validates the explicit identity against that list; it does
        # not apply the editable projection fallback.
        assert obs_a.choices_for_freeze() == ("halpha", "detx")

        # Switch to source B (no observation yet) -> UNKNOWN (A's is not reused).
        token["v"] = "B"
        obs_b = widget._controls_v2_capture_gi_motor_observation()
        assert obs_b.state == GIMotorObservation.UNKNOWN
        assert obs_b.choices_for_freeze() is None

        # A DELAYED stale A signal arriving under B is ignored.
        widget._controls_v2_record_gi_motor_observation(["halpha"], for_token="A")
        assert widget._controls_v2_capture_gi_motor_observation().state == (
            GIMotorObservation.UNKNOWN)

        # B hydrates to a genuinely empty motor list -> KNOWN_EMPTY -> ().
        widget._controls_v2_record_gi_motor_observation([])
        obs_b2 = widget._controls_v2_capture_gi_motor_observation()
        assert obs_b2.state == GIMotorObservation.KNOWN_EMPTY
        assert obs_b2.choices_for_freeze() == ()
    finally:
        widget.close()
        widget.deleteLater()


def test_t1r_unknown_preserves_motor_known_empty_refuses_enabled_run(
        qapp, monkeypatch):
    """§12 test 12: at freeze, UNKNOWN choices preserve the operator's explicit
    motor while KNOWN_EMPTY refuses that now-stale enabled-run identity.

    RED at 4b59b7da: the GUI mapped every no-real-choice case to None, so the
    core policy's known-empty distinction was unreachable.  The E6 owner split
    retains editable fallback but makes enabled run admission fail closed."""
    from xdart.gui.tabs.static_scan.static_scan_widget import GIMotorObservation
    widget = _t1r_widget(monkeypatch)
    try:
        token = {"v": "A"}
        monkeypatch.setattr(
            widget, "_controls_v2_source_token", lambda: token["v"])
        widget._on_controls_v2_field_changed(("GI", "Grazing"), True)
        widget._on_controls_v2_field_changed(("GI", "th_motor"), "halpha")

        # UNKNOWN: no observation for the source -> explicit motor honored.
        frozen = widget._prepare_controls_v2_run_configuration()
        assert frozen.gi.effective_motor == "halpha"

        # KNOWN_EMPTY: the source is probed and has no real motors, so an
        # enabled run cannot retain the explicit halpha identity.
        widget._controls_v2_record_gi_motor_observation([])
        with pytest.raises(ValueError, match="GI metadata motor 'halpha'"):
            widget._prepare_controls_v2_run_configuration()
    finally:
        widget.close()
        widget.deleteLater()


def test_t2_second_field_failure_rolls_back_earlier_field(qapp, monkeypatch):
    """§9.10 test 1: a two-field transaction whose SECOND legacy field fails
    readback restores the FIRST field to its exact pre-Start value (reverse-order
    verified rollback), retains the whole journal, and refuses the freeze — no
    generation bump.

    RED at 506dd978: T-1R detected the silent no-op and aborted but did NOT roll
    back the earlier field, leaving it mutated (reverse-rollback deferred to T-2)."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        DeferredRunEditsPendingError,
        staticWidget,
    )

    widget = staticWidget()
    try:
        widget._pending_controls_v2_run_configuration = None
        params = widget.wrangler.parameters
        mask_param = params.child("Signal", "mask_file")
        bg_param = params.child("BG", "File")
        mask_before = mask_param.value()
        bg_before = bg_param.value()
        gen_before = widget._controls_v2_ensure_run_intent().generation

        widget._enter_run_state()
        widget._on_controls_v2_field_changed(("Signal", "mask_file"), "/tmp/a.edf")
        widget._on_controls_v2_field_changed(("BG", "File"), "/tmp/b.edf")
        widget.wrangler.finished.emit()
        qapp.processEvents()
        # Field B's carrier silently no-ops.
        monkeypatch.setattr(bg_param, "setValue", lambda *a, **k: None)

        with pytest.raises(DeferredRunEditsPendingError):
            widget._prepare_controls_v2_run_configuration()

        # A rolled back; B unchanged; journal full; nothing frozen; no gen bump.
        assert mask_param.value() == mask_before
        assert bg_param.value() == bg_before
        assert getattr(
            widget, "_pending_controls_v2_run_configuration", None) is None
        assert widget._controls_v2_ensure_run_intent().generation == gen_before
        remaining = {tuple(p) for p, _ in widget._controls_v2_deferred_field_edits}
        assert ("Signal", "mask_file") in remaining
        assert ("BG", "File") in remaining
    finally:
        widget.close()
        widget.deleteLater()


def test_t2_setter_exception_is_commit_failure_with_verified_rollback(
        qapp, monkeypatch):
    """§9.10 test 3: a silent setter FAILURE (setValue raises, swallowed by the
    signal-blocked mirror) is caught by the per-write readback as a commit
    failure, and the earlier applied field is rolled back with verified readback.
    The readback — not the setter's return — is the authority.

    RED at 506dd978: T-1R aborted on the readback but left the earlier field
    mutated (no rollback)."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        DeferredRunEditsPendingError,
        staticWidget,
    )

    widget = staticWidget()
    try:
        widget._pending_controls_v2_run_configuration = None
        params = widget.wrangler.parameters
        mask_param = params.child("Signal", "mask_file")
        bg_param = params.child("BG", "File")
        mask_before = mask_param.value()

        widget._enter_run_state()
        widget._on_controls_v2_field_changed(("Signal", "mask_file"), "/tmp/a.edf")
        widget._on_controls_v2_field_changed(("BG", "File"), "/tmp/b.edf")
        widget.wrangler.finished.emit()
        qapp.processEvents()

        def _raise(*a, **k):
            raise RuntimeError("injected setter failure")

        monkeypatch.setattr(bg_param, "setValue", _raise)

        with pytest.raises(DeferredRunEditsPendingError):
            widget._prepare_controls_v2_run_configuration()

        assert mask_param.value() == mask_before  # verified rollback of A
        assert getattr(
            widget, "_pending_controls_v2_run_configuration", None) is None
    finally:
        widget.close()
        widget.deleteLater()


def test_t2_rollback_failure_surfaces_recovery_error_naming_carrier(
        qapp, monkeypatch):
    """§9.10 test 4: when the checked commit cannot RESTORE a carrier during
    rollback, Run stays refused with a DISTINCT recovery error that NAMES the
    un-restorable carrier, and the journal is retained.

    RED at 506dd978: T-1R had no rollback, hence no recovery-error concept."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        DeferredRunEditsPendingError,
        staticWidget,
    )

    widget = staticWidget()
    try:
        widget._pending_controls_v2_run_configuration = None
        params = widget.wrangler.parameters
        mask_param = params.child("Signal", "mask_file")
        bg_param = params.child("BG", "File")

        widget._enter_run_state()
        widget._on_controls_v2_field_changed(("Signal", "mask_file"), "/tmp/a.edf")
        widget._on_controls_v2_field_changed(("BG", "File"), "/tmp/b.edf")
        widget.wrangler.finished.emit()
        qapp.processEvents()

        # Field A applies forward (non-empty) but its RESTORE (to the empty prior)
        # silently no-ops -> A cannot be rolled back.  Field B no-ops forward ->
        # triggers the rollback of A.
        orig_mask_set = mask_param.setValue

        def _mask_set(value, *a, **k):
            if str(value):
                orig_mask_set(value, *a, **k)

        monkeypatch.setattr(mask_param, "setValue", _mask_set)
        monkeypatch.setattr(bg_param, "setValue", lambda *a, **k: None)

        with pytest.raises(DeferredRunEditsPendingError) as excinfo:
            widget._prepare_controls_v2_run_configuration()

        assert "mask_file" in str(excinfo.value)   # names the un-restorable carrier
        assert getattr(
            widget, "_pending_controls_v2_run_configuration", None) is None
        remaining = {tuple(p) for p, _ in widget._controls_v2_deferred_field_edits}
        assert ("Signal", "mask_file") in remaining
    finally:
        widget.close()
        widget.deleteLater()


def test_t2_source_transaction_reconciles_exactly_once(qapp, monkeypatch):
    """§9.10 test 6: a transaction with multiple source-selection fields reconciles
    the source index EXACTLY ONCE (the legacy projection is signal-blocked, so the
    source-tree handler cannot independently reconcile) and does not lose the
    edits."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        calls = []
        monkeypatch.setattr(
            widget, "_sync_controls_v2_source_index",
            lambda: calls.append(1))

        widget._enter_run_state()
        widget._on_controls_v2_field_changed(("Signal", "include_subdir"), True)
        widget._on_controls_v2_field_changed(("Signal", "img_ext"), "tif")
        widget.wrangler.finished.emit()
        qapp.processEvents()
        calls.clear()  # count only the Start-time reconciliation

        frozen = widget._prepare_controls_v2_run_configuration()
        assert frozen is not None                 # edits applied, not lost
        assert len(calls) == 1                     # exactly one reconciliation
        assert list(widget._controls_v2_deferred_field_edits) == []
    finally:
        widget.close()
        widget.deleteLater()


def test_t2_source_reconcile_failure_refuses_and_retains_journal(
        qapp, monkeypatch):
    """§9.10 test 7: a source reconciliation exception fails CLOSED — no freeze,
    the journal is retained, and Run is refused with a typed
    DeferredRunEditsPendingError; the transaction does not leave a partial
    source-owner swap or a false applied event.

    RED at 506dd978: T-1R's commit called _sync_controls_v2_source_index with no
    guard, so the raw exception propagated (not the typed refusal) and the intent
    was left installed (fail-open, §9.7)."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        DeferredRunEditsPendingError,
        staticWidget,
    )

    widget = staticWidget()
    try:
        widget._pending_controls_v2_run_configuration = None

        def _boom():
            raise RuntimeError("injected source reconcile failure")

        widget._enter_run_state()
        widget._on_controls_v2_field_changed(("Signal", "include_subdir"), True)
        widget.wrangler.finished.emit()
        qapp.processEvents()
        monkeypatch.setattr(
            widget, "_sync_controls_v2_source_index", _boom)

        with pytest.raises(DeferredRunEditsPendingError):
            widget._prepare_controls_v2_run_configuration()

        assert getattr(
            widget, "_pending_controls_v2_run_configuration", None) is None
        remaining = {tuple(p) for p, _ in widget._controls_v2_deferred_field_edits}
        assert ("Signal", "include_subdir") in remaining
    finally:
        widget.close()
        widget.deleteLater()


def test_t2_unrelated_edits_and_unchanged_start_reconcile_nothing(
        qapp, monkeypatch):
    """§12 test 13 / §12.7: mask/PONI/series-average edits and an unchanged Start
    cause ZERO source reconciliation and zero directory poll — only a source
    SELECTION change reconciles.

    RED at 506dd978: T-1R's commit reconciled on any Signal/Source
    (source_touched) edit, including mask/series-average."""
    widget = _t1r_widget(monkeypatch)
    try:
        calls = []
        monkeypatch.setattr(
            widget, "_sync_controls_v2_source_index", lambda: calls.append(1))
        widget._enter_run_state()
        widget._on_controls_v2_field_changed(("Signal", "mask_file"), "/tmp/m.edf")
        widget._on_controls_v2_field_changed(("Signal", "series_average"), "3")
        widget.wrangler.finished.emit()
        qapp.processEvents()
        calls.clear()

        frozen = widget._prepare_controls_v2_run_configuration()
        assert frozen is not None
        assert calls == []   # non-selection edits reconcile nothing (§12.7)

        calls.clear()
        widget._prepare_controls_v2_run_configuration()   # unchanged Start
        assert calls == []
    finally:
        widget.close()
        widget.deleteLater()


def test_t2_range_low_gt_high_is_typed_refusal(qapp, monkeypatch):
    """§12.9 P3b: a radial/azimuth range whose low exceeds its high is a typed
    staging refusal (not a freeze error or a silent swap).

    RED at 506dd978: the pure reducer did not validate range shape."""
    from xdart.gui.tabs.static_scan.static_scan_widget import ControlsTransactionError
    widget = _t1r_widget(monkeypatch)
    try:
        result = widget.stage_controls_transaction([
            (("Int1D", "radial_low"), 5.0),
            (("Int1D", "radial_high"), 1.0),
        ])
        assert isinstance(result, ControlsTransactionError)
    finally:
        widget.close()
        widget.deleteLater()


def test_apply_snapshot_to_scan_in_place_not_aliased(qapp, monkeypatch):
    """R4B-15 (O-1a-i item 9) + orchestrator amendment: projecting the Controls
    snapshot onto the display scan mutates the existing ``bai_*_args`` dicts IN
    PLACE (stable identity for held references — the S10 polarization round-trip)
    while introducing NO aliasing with the Controls-owned intent (the single
    writer of run configuration)."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        intent = widget._controls_v2_ensure_run_intent()
        widget._controls_v2_ensure_native_int_defaults()
        held_1d = widget.scan.bai_1d_args
        held_2d = widget.scan.bai_2d_args

        widget._controls_v2_apply_snapshot_to_scan(
            widget._controls_v2_native_int_snapshot())

        # Stable identity: the held references still ARE the scan's dicts.
        assert widget.scan.bai_1d_args is held_1d
        assert widget.scan.bai_2d_args is held_2d
        # No aliasing with the intent: distinct objects both ways.
        assert widget.scan.bai_1d_args is not intent.bai_1d_args
        assert widget.scan.bai_2d_args is not intent.bai_2d_args
        widget.scan.bai_1d_args["__probe_scan__"] = 123
        assert "__probe_scan__" not in intent.bai_1d_args
        intent.bai_1d_args["__probe_intent__"] = 456
        assert "__probe_intent__" not in widget.scan.bai_1d_args
    finally:
        widget.close()
        widget.deleteLater()


# ---------------------------------------------------------------------------
# T-2R (§14.11) — checked-commit atomicity: preflight, push-before-setter,
# rollback-all, contained exceptions, display-scan rollback, typed attempt
# result.  Each is RED at 1f8b4255 and GREEN at the T-2R correction.
# ---------------------------------------------------------------------------


def test_t2r_failed_forward_carrier_is_itself_restored(qapp, monkeypatch):
    """§14.11.A.3 / E1: a setter that writes the WRONG value then fails readback
    still gets restored — the failed-forward carrier is pushed onto the rollback
    stack BEFORE its setter and force-restored directly.

    RED at 1f8b4255: the carrier was appended to ``applied`` only AFTER a
    successful readback, so the one that failed forward was never rolled back."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        mask_path = ("Signal", "mask_file")
        bg_path = ("BG", "File")
        mask = widget._controls_v2_param(mask_path)
        bg = widget._controls_v2_param(bg_path)
        mask_before = mask.value()
        bg_before = bg.value()
        staged = widget.stage_controls_transaction(
            [(mask_path, "/tmp/t2r-mask.edf"), (bg_path, "/tmp/t2r-bg.edf")])

        original_bg_set = bg.setValue

        def set_wrong(_value, *args, **kwargs):
            return original_bg_set("/tmp/t2r-wrong.edf", *args, **kwargs)

        monkeypatch.setattr(bg, "setValue", set_wrong)
        result = widget.commit_controls_transaction(staged)

        assert not result.ok
        assert result.phase == "legacy_apply"
        assert mask.value() == mask_before
        assert bg.value() == bg_before
    finally:
        widget.close()
        widget.deleteLater()


def test_t2r_rollback_continues_past_first_restore_failure(qapp, monkeypatch):
    """§14.11.A.4 / E2: a restore failure on one carrier must NOT abort the
    restore of independently-restorable OLDER carriers; every un-restorable path
    is collected.

    RED at 1f8b4255: ``_controls_v2_rollback_legacy`` returned on the first
    restore failure, leaving earlier carriers mutated."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        first_path = ("Signal", "mask_file")
        second_path = ("BG", "File")
        trigger_path = ("Signal", "Filter")
        first = widget._controls_v2_param(first_path)
        second = widget._controls_v2_param(second_path)
        trigger = widget._controls_v2_param(trigger_path)
        first_before = first.value()
        second_before = second.value()
        staged = widget.stage_controls_transaction([
            (first_path, "/tmp/t2r-first.edf"),
            (second_path, "/tmp/t2r-second.edf"),
            (trigger_path, "t2r-trigger"),
        ])

        original_second_set = second.setValue

        def refuse_restore(value, *args, **kwargs):
            if value != second_before:
                return original_second_set(value, *args, **kwargs)
            return None

        monkeypatch.setattr(second, "setValue", refuse_restore)
        monkeypatch.setattr(trigger, "setValue", lambda *a, **k: None)
        result = widget.commit_controls_transaction(staged)

        assert not result.ok
        assert result.recovery_failed_path == second_path
        assert second_path in result.recovery_failed_paths
        assert first.value() == first_before   # older carrier still restored
    finally:
        widget.close()
        widget.deleteLater()


def test_t2r_getter_exception_stays_inside_boundary(qapp, monkeypatch):
    """§14.11.A.1/A.5 / E3: a getter that raises is contained — no raw exception
    escapes ``commit_controls_transaction`` and earlier carriers are untouched.

    RED at 1f8b4255: the prior value was read inside the forward loop AFTER an
    earlier write, so the raw ``RuntimeError`` escaped with A applied."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        first_path = ("Signal", "mask_file")
        second_path = ("BG", "File")
        first = widget._controls_v2_param(first_path)
        second = widget._controls_v2_param(second_path)
        first_before = first.value()
        staged = widget.stage_controls_transaction(
            [(first_path, "/tmp/t2r-first.edf"), (second_path, "/tmp/t2r-second.edf")])

        monkeypatch.setattr(
            second, "value",
            lambda: (_ for _ in ()).throw(RuntimeError("read failed")))
        result = widget.commit_controls_transaction(staged)   # must NOT raise

        assert not result.ok
        assert first.value() == first_before
    finally:
        widget.close()
        widget.deleteLater()


def test_t2r_missing_carrier_at_preflight_fails_closed(qapp, monkeypatch):
    """§14.11.A.1/A.6 / E4: a carrier missing at preflight is a typed refusal with
    ZERO writes — never the silent success of the old
    ``_controls_v2_apply_legacy_carrier`` (which returned True when the param was
    ``None``).

    RED at 1f8b4255: a missing carrier was treated as applied, so the commit
    succeeded and the earlier carrier was written."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        mask_path = ("Signal", "mask_file")
        bg_path = ("BG", "File")
        mask = widget.wrangler.parameters.child(*mask_path)
        mask_before = mask.value()
        staged = widget.stage_controls_transaction(
            [(mask_path, "/tmp/t2r-mask.edf"), (bg_path, "/tmp/t2r-bg.edf")])

        real_param = widget._controls_v2_param

        def fake_param(path):
            if tuple(path) == bg_path:
                return None    # carrier "missing" at preflight
            return real_param(path)

        monkeypatch.setattr(widget, "_controls_v2_param", fake_param)
        result = widget.commit_controls_transaction(staged)

        assert not result.ok
        assert result.phase == "preflight"
        assert result.failed_path == bg_path
        assert mask.value() == mask_before   # zero writes
    finally:
        widget.close()
        widget.deleteLater()


def test_t2r_partial_display_projection_is_rolled_back(qapp, monkeypatch):
    """§14.11.B.1 / E5: a mid-projection failure in
    ``_controls_v2_apply_snapshot_to_scan`` restores the display scan (in-place
    dict identity preserved) as well as the intent.

    RED at 1f8b4255: only the intent/self fields were snapshotted; the display
    scan stayed at its partially-projected value."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        intent = widget._controls_v2_ensure_run_intent()
        widget._controls_v2_ensure_native_int_defaults()
        held_1d = widget.scan.bai_1d_args
        intent_before = copy.deepcopy(intent.bai_1d_args)
        scan_before = copy.deepcopy(widget.scan.bai_1d_args)
        staged = widget.stage_controls_transaction([(("Int1D", "points"), "777")])

        def partial_then_raise(_snapshot, *a, **k):
            widget.scan.bai_1d_args["numpoints"] = 999_999
            raise RuntimeError("partial display projection")

        monkeypatch.setattr(
            widget, "_controls_v2_apply_snapshot_to_scan", partial_then_raise)
        result = widget.commit_controls_transaction(staged)

        assert not result.ok
        assert result.phase == "install"
        assert intent.bai_1d_args == intent_before
        assert widget.scan.bai_1d_args == scan_before
        assert widget.scan.bai_1d_args is held_1d   # in-place identity preserved
    finally:
        widget.close()
        widget.deleteLater()


def test_t2r_stale_recovery_does_not_mislabel_next_attempt(qapp, monkeypatch):
    """§14.11.C / E7: the fold's returned result is the sole diagnostic authority;
    a stale ambient recovery marker must not relabel a fresh staging failure.

    RED at 1f8b4255: ``_prepare`` read the ambient
    ``_controls_v2_last_fold_recovery`` (cleared only on success), so an
    unrelated staging failure surfaced the old carrier's recovery message."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        DeferredRunEditsPendingError,
        staticWidget,
    )

    widget = staticWidget()
    try:
        # A stale ambient marker from a prior attempt (now defunct).
        widget._controls_v2_last_fold_recovery = ("Signal", "mask_file")
        widget._controls_v2_record_edit(
            ("Int1D", "points"), "not-a-number", origin="draft")

        with pytest.raises(DeferredRunEditsPendingError) as excinfo:
            widget._prepare_controls_v2_run_configuration()

        message = str(excinfo.value)
        assert "Int1D/points" in message
        assert "could not be restored" not in message
    finally:
        widget.close()
        widget.deleteLater()


def test_t2r_refusal_message_is_phase_correct(qapp, monkeypatch):
    """§14.11.C / E10: an install failure surfaces its PHASE, not an empty path.

    RED at 1f8b4255: an install failure had ``failed_path=None``, so the message
    rendered ``deferred edit invalid ()`` with an empty descriptor."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        DeferredRunEditsPendingError,
        staticWidget,
    )

    widget = staticWidget()
    try:
        widget._pending_controls_v2_run_configuration = None
        widget._enter_run_state()
        widget._on_controls_v2_field_changed(("Int1D", "points"), 111)
        widget.wrangler.finished.emit()
        qapp.processEvents()

        def _raise(_snapshot, *a, **k):
            raise RuntimeError("injected install failure")

        monkeypatch.setattr(
            widget, "_controls_v2_apply_snapshot_to_scan", _raise)

        with pytest.raises(DeferredRunEditsPendingError) as excinfo:
            widget._prepare_controls_v2_run_configuration()

        message = str(excinfo.value)
        assert "install" in message         # phase surfaced
        assert "()" not in message          # not an empty descriptor
    finally:
        widget.close()
        widget.deleteLater()


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


def test_t2_1_candidate_source_is_b_with_unknown_motor(qapp, monkeypatch, tmp_path):
    """§13.11 test 5: a staged source A→B (then GI enable) derives candidate
    source B immediately and sets motor knowledge UNKNOWN — A's motor is never
    adopted into the candidate.

    RED at 1f8b4255: StagedControlsTransaction had no candidate_source_spec."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        GIMotorObservation,
        staticWidget,
    )

    widget = staticWidget()
    try:
        file_a = str(tmp_path / "scan_a.tif")
        file_b = str(tmp_path / "scan_b.tif")
        sig = widget.wrangler.parameters.child("Signal")
        sig.child("inp_type").setValue("Single Image")
        sig.child("File").setValue(file_a)
        # Record source A's motor observation (halpha) under A's token.
        widget._controls_v2_record_gi_motor_observation(["halpha"])
        assert widget._controls_v2_capture_gi_motor_observation().state == (
            GIMotorObservation.KNOWN_NONEMPTY)

        staged = widget.stage_controls_transaction([
            (("Signal", "File"), file_b),
            (("GI", "Grazing"), True),
        ])

        # Candidate source is B ...
        assert staged.candidate_source_spec is not None
        assert file_b in str(getattr(staged.candidate_source_spec, "uri", ""))
        assert file_b in tuple(str(x) for x in (staged.source_fingerprint or ()))
        # ... and its motor knowledge is UNKNOWN; A's halpha is never adopted.
        assert staged.gi_motor_observation.state == GIMotorObservation.UNKNOWN
        assert "halpha" not in staged.gi_motor_observation.motors
    finally:
        widget.close()
        widget.deleteLater()


def test_t2_1_delayed_and_older_hydration_are_ignored(qapp, monkeypatch):
    """§13.11 test 6: a production delayed-A result under source B, and an older
    same-root generation under a newer request, change NEITHER the stored motor
    knowledge NOR the visible theta dropdown.

    RED at 1f8b4255: sigGIMotorOptions carried a bare list with no source/epoch,
    so a delayed result was recorded as the current source."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        wr = widget.wrangler
        wr.inp_type = "Image Directory"
        integ_calls = []
        monkeypatch.setattr(
            widget.integratorTree, "set_gi_motor_options",
            lambda motors: integ_calls.append(tuple(motors)))
        emitted = []
        wr.sigGIMotorOptions.connect(lambda p: emitted.append(p))

        # Source A: request starts (gen bumped), motors halpha -> owner records A.
        wr.img_dir = "/tmp/source_A"
        wr._next_gi_hydration_generation()
        wr.motors = ["halpha"]
        wr._gi_motor_knowledge_proved = True
        wr.set_gi_motor_options()
        qapp.processEvents()
        payload_a = emitted[-1]

        # Switch to source B: newer request, motors gonth -> owner records B.
        wr.img_dir = "/tmp/source_B"
        wr._next_gi_hydration_generation()
        wr.motors = ["gonth"]
        wr.set_gi_motor_options()
        qapp.processEvents()
        stored_after_b = widget._controls_v2_gi_motor_observation
        integ_calls_after_b = len(integ_calls)

        # DELAYED A result arrives under B -> ignored (fingerprint mismatch).
        widget._on_gi_motor_options_changed(payload_a)
        assert widget._controls_v2_gi_motor_observation is stored_after_b
        assert len(integ_calls) == integ_calls_after_b   # dropdown untouched

        # An OLDER same-root generation under a newer request -> also ignored.
        wr.img_dir = "/tmp/source_R"
        wr._next_gi_hydration_generation()
        wr.motors = ["th"]
        wr.set_gi_motor_options()
        qapp.processEvents()
        payload_r_old = emitted[-1]
        wr._next_gi_hydration_generation()          # newer request, same root R
        wr.motors = ["eta"]
        wr.set_gi_motor_options()
        qapp.processEvents()
        stored_after_r = widget._controls_v2_gi_motor_observation
        integ_calls_after_r = len(integ_calls)

        widget._on_gi_motor_options_changed(payload_r_old)   # stale generation
        assert widget._controls_v2_gi_motor_observation is stored_after_r
        assert len(integ_calls) == integ_calls_after_r
    finally:
        widget.close()
        widget.deleteLater()


def test_t2_1_recursive_dir_preview_is_unknown_then_hydrates(qapp, monkeypatch, tmp_path):
    """§13.11 test 7: a recursive directory whose matching data lives ONLY in a
    subdirectory yields UNKNOWN (no eager recursive walk; explicit motor
    preserved); a later targeted hydration then populates the choices.

    RED at 1f8b4255: 'no direct-child preview' was classified KNOWN_EMPTY, so an
    explicit GI motor degraded to Manual."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        GIMotorObservation,
        staticWidget,
    )

    widget = staticWidget()
    try:
        root = tmp_path / "recursive_root"
        (root / "sub").mkdir(parents=True)
        (root / "sub" / "img_00001_master.h5").write_bytes(b"")
        wr = widget.wrangler
        wr.inp_type = "Image Directory"
        wr.img_dir = str(root)
        wr.include_subdir = True
        wr.img_ext = "h5"
        wr.file_filter = ""

        emitted = []
        wr.sigGIMotorOptions.connect(lambda p: emitted.append(p))

        # Direct-child-only preview finds nothing (no recursive walk into sub/).
        assert wr._directory_metadata_preview_file() == ""
        wr.motors = []
        wr._adopt_directory_metadata_preview("")
        qapp.processEvents()
        assert emitted[-1].state == GIMotorObservation.UNKNOWN
        obs = widget._controls_v2_capture_gi_motor_observation()
        assert obs.state == GIMotorObservation.UNKNOWN
        assert obs.choices_for_freeze() is None      # explicit motor preserved

        # Targeted hydration reports motors for THIS source.
        wr.motors = ["halpha"]
        wr._gi_motor_knowledge_proved = True
        wr.set_gi_motor_options()
        qapp.processEvents()
        assert emitted[-1].state == GIMotorObservation.KNOWN_NONEMPTY
        obs2 = widget._controls_v2_capture_gi_motor_observation()
        assert obs2.state == GIMotorObservation.KNOWN_NONEMPTY
        assert "halpha" in obs2.choices_for_freeze()
    finally:
        widget.close()
        widget.deleteLater()


def test_t2_1_targeted_empty_is_known_empty_and_manual(qapp, monkeypatch):
    """§13.11 test 8: a source PROVEN to have no motors is KNOWN_EMPTY, and the
    effective GI motor resolves to Manual.

    RED at 1f8b4255: sigGIMotorOptions carried a bare list with no knowledge
    state (payload has no ``.state``)."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        GIMotorObservation,
        staticWidget,
    )

    widget = staticWidget()
    try:
        wr = widget.wrangler
        emitted = []
        wr.sigGIMotorOptions.connect(lambda p: emitted.append(p))
        widget._on_controls_v2_field_changed(("GI", "Grazing"), True)
        widget._on_controls_v2_field_changed(("GI", "th_motor"), "halpha")

        # A targeted inspection proves NO motors -> KNOWN_EMPTY.
        wr.motors = []
        wr._gi_motor_knowledge_proved = True
        wr.set_gi_motor_options()
        qapp.processEvents()
        assert emitted[-1].state == GIMotorObservation.KNOWN_EMPTY
        assert widget._controls_v2_capture_gi_motor_observation().state == (
            GIMotorObservation.KNOWN_EMPTY)

        frozen = widget._prepare_controls_v2_run_configuration()
        assert frozen.gi.effective_motor == "Manual"
    finally:
        widget.close()
        widget.deleteLater()


def test_t2_1_compound_poni_deferred_parity(qapp, monkeypatch, tmp_path):
    """§13.11 test 9: a deferred PONI A→B installs path, values, object, wrangler
    carrier, thread carrier, and writer provenance as ONE compound value.

    RED at 1f8b4255: the signal-blocked projection updated only the path, so the
    thread kept PONI A's in-memory object while the frozen identity named B."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    from xrd_tools.core.containers import PONI

    widget = staticWidget()
    try:
        poni_a = _write_poni(tmp_path / "a.poni", 0.10)
        poni_b = _write_poni(tmp_path / "b.poni", 0.20)
        b_values = PONI.from_poni_file(poni_b).to_dict()

        # Establish live PONI A on the wrangler + thread.
        widget.wrangler.parameters.child("Signal", "poni_file").setValue(poni_a)
        assert widget.wrangler.poni is not None

        # Deferred edit A→B, then Start folds it through the checked commit.
        widget._enter_run_state()
        widget._on_controls_v2_field_changed(("Signal", "poni_file"), poni_b)
        widget.wrangler.finished.emit()
        qapp.processEvents()
        frozen = widget._prepare_controls_v2_run_configuration()

        assert widget.wrangler.poni_file == poni_b
        assert widget.wrangler.poni.to_dict() == b_values
        assert widget.wrangler.thread.poni.to_dict() == b_values
        assert widget._controls_v2_ensure_run_intent().poni_values == b_values
        assert frozen.poni_file == poni_b
        assert frozen.poni_values == b_values
    finally:
        widget.close()
        widget.deleteLater()


def test_t2_1_compound_poni_invalid_b_and_midcommit_restore_a(qapp, monkeypatch, tmp_path):
    """§13.11 test 9 (rollback): an invalid B refuses staging, and an injected
    mid-commit failure rolls back — both leave ALL of PONI A's carriers intact.

    RED at 1f8b4255: PONI was not part of the transaction, so an invalid B and a
    mid-commit failure could leave a torn (path B / object A) state."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        ControlsTransactionError,
        staticWidget,
    )

    widget = staticWidget()
    try:
        poni_a = _write_poni(tmp_path / "a.poni", 0.10)
        bad_b = tmp_path / "bad.poni"
        bad_b.write_text("- not\n- a\n- mapping\n")   # exists but unparseable
        widget.wrangler.parameters.child("Signal", "poni_file").setValue(poni_a)
        # get_poni_dict sets wrangler.poni on load; the thread carrier is synced
        # at setup — establish the full A baseline (as a prior run would leave it).
        widget.wrangler.thread.poni = widget.wrangler.poni
        a_values = widget.wrangler.poni.to_dict()
        a_object = widget.wrangler.poni

        # Invalid B -> staging refuses; A carriers untouched.
        staged = widget.stage_controls_transaction(
            [(("Signal", "poni_file"), str(bad_b))])
        assert isinstance(staged, ControlsTransactionError)
        assert widget.wrangler.poni is a_object
        assert widget.wrangler.thread.poni.to_dict() == a_values

        # Valid B but an injected mid-commit failure -> rollback restores A.
        good_b = _write_poni(tmp_path / "b.poni", 0.20)
        staged2 = widget.stage_controls_transaction(
            [(("Signal", "poni_file"), good_b)])
        assert staged2.poni_touched

        def _boom(_snapshot, *a, **k):
            raise RuntimeError("injected mid-commit failure")

        monkeypatch.setattr(
            widget, "_controls_v2_apply_snapshot_to_scan", _boom)
        result = widget.commit_controls_transaction(staged2)

        assert not result.ok
        assert widget.wrangler.poni is a_object
        assert widget.wrangler.poni_file == poni_a
        assert widget.wrangler.thread.poni.to_dict() == a_values
    finally:
        widget.close()
        widget.deleteLater()


def test_t2r_source_double_fault_names_source_carrier(qapp, monkeypatch):
    """§14.11.B.4 / E6: when source reconciliation fails AND the restore-sync also
    fails, the recovery failure explicitly NAMES the source carrier, and the
    caches/observation are restored where possible.

    RED at 1f8b4255: the restore-sync exception was logged and discarded, so
    recovery_failed_path never named the source owner."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        path = ("Signal", "include_subdir")
        current = bool(widget._controls_v2_param(path).value())
        staged = widget.stage_controls_transaction([(path, not current)])
        prior_observation = object()
        prior_energy_cache = ("old", 12.0)
        prior_probe_cache = ("old", "probe")
        widget._controls_v2_directory_observation = prior_observation
        widget._controls_v2_source_energy_cache = prior_energy_cache
        widget._controls_v2_metadata_probe_cache = prior_probe_cache
        calls = []

        def _sync_fails():
            calls.append(1)
            widget._controls_v2_directory_observation = ("partial", len(calls))
            raise RuntimeError("source reconciliation failed")

        monkeypatch.setattr(
            widget, "_sync_controls_v2_source_index", _sync_fails)
        result = widget.commit_controls_transaction(staged)

        assert not result.ok
        assert len(calls) == 2                       # forward + restore attempt
        assert result.recovery_failed_path == ("Source",)
        assert widget._controls_v2_directory_observation is prior_observation
        assert widget._controls_v2_source_energy_cache == prior_energy_cache
        assert widget._controls_v2_metadata_probe_cache == prior_probe_cache
    finally:
        widget.close()
        widget.deleteLater()


def test_t2r_same_value_source_edit_does_not_reconcile(qapp, monkeypatch):
    """§14.11.D.2: a same-value source edit whose effective selection is unchanged
    performs ZERO source reconciliation.

    RED at 1f8b4255: source_selection_touched was set from path membership, so a
    net-zero include_subdir edit still requested one reconciliation."""
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget = staticWidget()
    try:
        path = ("Signal", "include_subdir")
        current = widget._controls_v2_param(path).value()
        widget._controls_v2_record_edit(path, current, origin="deferred")
        calls = []
        monkeypatch.setattr(
            widget, "_sync_controls_v2_source_index",
            lambda: calls.append(1))

        assert widget._controls_v2_fold_deferred_edits_into_intent() is None
        assert calls == []
    finally:
        widget.close()
        widget.deleteLater()


def test_section_number_presentation_option_preserves_canonical_default(qapp):
    compact = ControlsPanelV2(show_section_numbers=False)
    canonical = ControlsPanelV2()
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
    panel = ControlsPanelV2()
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
