"""Stitch Advanced options: frame check, sample position and pyFAI corrections
reach the prepared plan through the page, and XU keeps only the frame check."""
from dataclasses import replace
import hashlib

import numpy as np
import pytest

from xdart.gui.pages.services import empty_host_services
from xdart.gui.pages.values import PageCleanup
from xdart.gui.tabs.stitch.page import StitchPage, frame_screen_note
from xdart.gui.tools.stitch_values import prepare_stitch_tool
from xrd_tools.analysis.stitch_operation import XuStitchOperationPlan
from tests.xdart.pages.test_stitch_page import _Status, _form, qapp  # noqa: F401
from tests.xdart.tools.test_stitch_native_values import _native_form


def _close(page, qapp):
    assert page.shutdown().status is PageCleanup.CLEAN
    page.deleteLater()
    qapp.processEvents()


def test_advanced_options_reach_the_prepared_plan(qapp, tmp_path):
    form = _native_form(tmp_path, "multigeometry", scans="1")
    dark = tmp_path / "dark.npy"
    np.save(dark, np.ones((195, 487)))
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.load_form(form)
        for key, value in (("frame_check", "Exclude empty and partial frames"),
                           ("sample_position", "On rotation axis"), ("solid_angle", "On"),
                           ("polarization", "0.95"), ("error_model", "Poisson"),
                           ("dark_path", str(dark))):
            page.set_field(key, value)
        built = page.build_form()
        assert (built.frame_screen, built.sample_position, built.polarization_factor,
                built.error_model, built.dark_path) == ("partial", "rotation_axis", .95, "poisson", str(dark))
        plan = prepare_stitch_tool(built).request.plan
        assert (plan.frame_screen.exclude, plan.sample_position) == ("partial", "rotation_axis")
        assert (plan.correct_solid_angle, plan.polarization_factor, plan.error_model) == (True, .95, "poisson")
        assert plan._dark_receipt.sha256 == hashlib.sha256(dark.read_bytes()).hexdigest()
        page.load_form(built)
        assert page.build_form().fingerprint == built.fingerprint
        assert page._rows["flat_cutoff"].editor.isEnabled() and built.flat_min_fraction == .1
        page.set_field("flat_cutoff", "50")
        assert page.build_form().flat_min_fraction == .5
        assert prepare_stitch_tool(page.build_form()).request.plan.flat_min_fraction == .5
        page.set_field("flat_cutoff", "many")
        with pytest.raises(ValueError, match="Flat cutoff"):
            page.build_form()
        page.set_field("flat_cutoff", "10")
        assert page.build_form().fingerprint == built.fingerprint  # the default changes nothing
        page.set_field("polarization", "")
        assert page.build_form().polarization_factor is None
        page.set_field("polarization", "strong")
        with pytest.raises(ValueError, match="Polarization"):
            page.build_form()
    finally:
        _close(page, qapp)


def test_xu_backend_takes_polarization_dark_and_flat_and_greys_the_rest(qapp, tmp_path):
    form = _native_form(tmp_path, "xu_hist", scans="1")
    dark = tmp_path / "dark.npy"
    np.save(dark, np.zeros((195, 1475)))
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.load_form(form)
        assert page.corrections_card.isEnabled()
        for key in ("sample_position", "solid_angle", "error_model"):
            assert not page._rows[key].editor.isEnabled(), key
        for key in ("polarization", "dark_path", "flat_path"):
            assert page._rows[key].editor.isEnabled(), key
        for key, value in (("frame_check", "Keep all frames"), ("polarization", "0.5"),
                           ("dark_path", str(dark)), ("error_model", "Poisson")):
            page.set_field(key, value)
        built = page.build_form()
        assert (built.frame_screen, built.polarization_factor, built.dark_path) == ("none", .5, str(dark))
        assert built.error_model is None  # greyed for XU: never forwarded
        plan = prepare_stitch_tool(built).request.plan
        assert type(plan) is XuStitchOperationPlan and plan.frame_screen.exclude == "none"
        assert plan.polarization_factor == .5 and plan._dark_receipt is not None
        page.set_field("backend", "pyFAI MultiGeometry")
        assert page._rows["sample_position"].editor.isEnabled()
        assert page._rows["solid_angle"].editor.isEnabled() and page._rows["error_model"].editor.isEnabled()
    finally:
        _close(page, qapp)


def test_form_refuses_pyfai_options_outside_native_pyfai(tmp_path):
    xu = _native_form(tmp_path, "xu_hist", scans="1")
    with pytest.raises(ValueError, match="no uncertainty"):
        replace(xu, error_model="poisson")
    with pytest.raises(ValueError, match="no uncertainty"):
        replace(xu, correct_solid_angle=False)
    assert replace(xu, polarization_factor=None).polarization_factor is None
    (tmp_path / "legacy").mkdir()
    legacy = _form(tmp_path / "legacy", "multigeometry")
    with pytest.raises(ValueError, match="native source selection"):
        replace(legacy, frame_screen="partial")


def test_frame_screen_note_names_excluded_and_kept_partial_frames():
    assert frame_screen_note(None) == ""
    assert frame_screen_note({"excluded": [], "partial": []}) == ""
    assert frame_screen_note({"excluded": [52, 164], "partial": [99, 106]}) == (
        "excluded dropped frames 52, 164 · partial frames kept 99, 106")
    assert frame_screen_note({"excluded": [52, 99], "partial": [99]}) == "excluded dropped frames 52, 99"


def test_the_page_takes_native_forms_only(qapp, tmp_path):
    (tmp_path / "legacy").mkdir()
    page = StitchPage(empty_host_services(_Status()))
    try:
        assert page.corrections_card.isEnabled()
        for key in ("frame_check", "sample_position", "solid_angle", "error_model"):
            assert page._rows[key].editor.isEnabled(), key
        with pytest.raises(ValueError, match="native source selection"):
            page.load_form(_form(tmp_path / "legacy", "multigeometry"))
        page.load_form(_native_form(tmp_path, "multigeometry", scans="1"))
        assert page.build_form().source_selection is not None
    finally:
        _close(page, qapp)


def test_preview_warns_when_the_flat_median_is_far_from_one(tmp_path):
    form = _native_form(tmp_path, "multigeometry", scans="1")
    rows, cols = form.detector_shape
    for name, value, warned in (("counts.npy", 1000., True), ("unit.npy", 1., False)):
        flat = tmp_path / name
        np.save(flat, np.full((rows, cols), value))
        warnings = prepare_stitch_tool(replace(form, flat_path=str(flat))).warnings
        assert any(item.startswith("Flat field median 1000") for item in warnings) is warned


def test_a_goniometer_json_sets_the_angle_mapping_from_its_positions(qapp, tmp_path):
    import json

    values = tmp_path / "MG_gonio_object.json"  # declares del_value/nu_value, not del_angle/nu_angle
    values.write_text(json.dumps({"trans_function": {"pos_names": ["del_value", "nu_value"]}}))
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.set_field("geometry_path", str(values))
        assert page._value("source_motors") == "del_value=del, nu_value=nu"
        page.set_field("source_motors", "del_value=delta, nu_value=nu")  # the user's motors stay
        page.set_field("geometry_path", str(values))
        assert page._value("source_motors") == "del_value=delta, nu_value=nu"
        page.set_field("geometry_path", str(tmp_path / "missing.json"))  # unreadable: untouched
        assert page._value("source_motors") == "del_value=delta, nu_value=nu"
    finally:
        _close(page, qapp)


def test_browse_opens_at_the_last_folder_with_int_file_types(qapp, tmp_path, monkeypatch):
    from pyqtgraph.Qt import QtWidgets
    from xdart.gui.widgets import file_dialogs
    from xdart.utils.browse import last_browse_dir, remember_browse_path

    monkeypatch.setenv("XDART_SESSION_FILE", str(tmp_path / "session.json"))
    monkeypatch.delenv("XDART_SESSION_FRESH", raising=False)
    last, calibration = tmp_path / "last", tmp_path / "calibration"
    last.mkdir(), calibration.mkdir()
    remember_browse_path(last)
    chosen = calibration / "MG_gonio_object.json"
    chosen.write_text("{}")
    seen = []

    def open_file(parent, title, start, file_filter):
        seen.append((title, start, file_filter))
        return str(chosen), file_filter

    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", open_file)
    page = StitchPage(empty_host_services(_Status()))
    try:
        page._browse("geometry_path")  # poni/json only, from the last folder picked anywhere
        assert seen == [("Choose calibration", str(last), file_dialogs.CALIBRATION_FILTER)]
        assert page._value("geometry_path") == str(chosen)
        assert last_browse_dir() == str(calibration)
        page._browse("mask_path")  # Int's mask types, from the folder just used
        assert seen[-1] == ("Choose detector mask", str(calibration), file_dialogs.MASK_FILTER)
    finally:
        _close(page, qapp)


def test_the_mode_box_lists_int_modes_and_leaves_for_them(qapp):
    from xdart.gui.pages.values import RSM_TOOL_KEY, SCATTERING_PAGE_KEY
    from xdart.gui.tabs.scattering.run_mode_projection import RUN_MODE_CHOICES

    page = StitchPage(empty_host_services(_Status()))
    requested = []
    page.toolRequested.connect(requested.append)
    try:
        combo = page.run_bar.modeCombo
        assert tuple(combo.itemText(row) for row in range(combo.count())) == RUN_MODE_CHOICES
        assert combo.currentText() == "Stitch 1D"
        assert not combo.model().item(combo.findText("Stitch 2D")).isEnabled()  # as on Int
        combo.setCurrentText("Int 2D")  # back to the integration workspace; the box stays
        assert requested == [str(SCATTERING_PAGE_KEY)] and combo.currentText() == "Stitch 1D"
        combo.setCurrentText("RSM")
        assert requested[-1] == str(RSM_TOOL_KEY) and combo.currentText() == "Stitch 1D"
    finally:
        _close(page, qapp)


def test_processing_rows_are_laid_out_as_int(qapp):
    from pyqtgraph.Qt import QtCore
    from xdart.gui.widgets.controls_panel import PROCESSING_ROW_LABEL_WIDTH, ControlsPanel

    page = StitchPage(empty_host_services(_Status()))
    try:
        rows = page._rows
        # the shared Controls names, so Int's styling (one box height) applies
        assert rows["unit"].editor.objectName() == "controlsComboBox"
        assert rows["npt_1d"].editor.objectName() == "controlsLineEdit"
        assert rows["npt_1d"].parent() is rows["unit"]  # Pts on the Axis row
        assert rows["unit"].label.minimumWidth() == PROCESSING_ROW_LABEL_WIDTH
        page.resize(1400, 1000)
        page.show()
        qapp.processEvents()
        panel = page.findChild(ControlsPanel)
        left = lambda editor: editor.mapTo(panel, QtCore.QPoint(0, 0)).x()
        edges = {left(rows["unit"].editor), left(page.radial_range._low),
                 left(page.threshold_range._low), left(rows["monitor"].editor)}
        assert len(edges) == 1, edges  # one left edge for Axis, ranges, Threshold, Monitor
    finally:
        _close(page, qapp)


def test_inputs_view_follows_int(qapp, tmp_path):
    spec = tmp_path / "data" / "LaB6_17keV"
    spec.parent.mkdir()
    spec.write_text("#F LaB6_17keV\n")
    calibration = tmp_path / "MG_gonio_object.json"
    calibration.write_text("{}")
    page = StitchPage(empty_host_services(_Status()))
    try:
        # Int's readiness line gates Run; Run then checks the inputs in full itself
        assert page.run_bar.readinessLabel.full_text() == "Choose an existing Project folder"
        assert not page.run_bar.startButton.isEnabled()
        low, high = page.threshold_range._low, page.threshold_range._high
        page._range_edited(("Stitch", "threshold_enabled"), False)  # Threshold off: limits greyed
        assert not low.isEnabled() and not high.isEnabled()
        page._range_edited(("Stitch", "threshold"), 700000.0)  # a manual band: editable, whole numbers
        page._range_edited(("Stitch", "threshold_enabled"), True)
        assert low.isEnabled() and high.isEnabled() and high.text() == "700000"

        page.set_field("project_root", str(tmp_path))
        page.set_field("geometry_path", str(calibration))
        page.set_sources([str(spec)])
        page.set_field("scan", "14.1")
        assert page._rows["image_dir"].editor.placeholderText() == f"auto · {spec.parent}"
        assert page._rows["image_stem"].editor.placeholderText() == "auto · LaB6_17keV_scan14_"
        names = page._rows["output_family"]  # an optional override under Advanced
        assert names.window() is page.advanced
        assert names.editor.placeholderText() == "auto · data-LaB6_17keV-14"
        page.set_field("scan", "[14,15]")  # a combined output is named automatically too
        assert names.editor.placeholderText() == "auto · data-LaB6_17keV-14-15"
        cards = page.controls
        for card in (cards.project_card, cards.experiment_card, cards.source_card, cards.processing_card):
            assert not card.valid_marker.isHidden(), card.title.text()
        assert cards.source_card.status.text() == "1 SPEC file · 1 output"
        assert page.run_bar.readinessLabel.full_text() == "Ready · Stitch 1D · 1 SPEC file · 1 output"
        assert page.run_bar.startButton.isEnabled()
    finally:
        _close(page, qapp)


def test_the_stitch_display_is_ints_scientific_view(qapp):
    from xdart.gui.tabs.scattering.scientific_view import ScientificView

    page = StitchPage(empty_host_services(_Status()))
    try:
        view = page.scientific
        assert type(view) is ScientificView and page.curve is view.curve
        # Int 1D's layout without the tools Stitch has no operation for, through every redraw
        for _ in range(2):
            assert view.norm.isHidden() and view.background.isHidden()
            assert view.raw_popup_button.isHidden() and view.image_splitter.isHidden()
            page._draw()
        choices = [view.plot_axis.itemData(row) for row in range(view.plot_axis.count())]
        assert choices == ["Q", "2theta"] and not view.plot_axis.isHidden()
        assert view.title.text() == "Current"
        with pytest.raises(ValueError, match="unknown scientific-view tools"):
            view.set_hidden_tools("slice")
    finally:
        _close(page, qapp)
