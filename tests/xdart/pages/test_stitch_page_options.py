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


def test_the_angle_mapping_follows_the_geometry_not_the_files_position_names(qapp, tmp_path):
    import json
    from xrd_tools.session.instrument import InstrumentGeometry

    values = tmp_path / "MG_gonio_object.json"  # declares del_value/nu_value, not del_angle/nu_angle
    values.write_text(json.dumps({"trans_function": {"pos_names": ["del_value", "nu_value"]}}))
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.set_field("geometry_path", str(values))
        assert page._value("source_motors") == "del_value=del, nu_value=nu"
        # the beamline calls del "delta": set once on the circle, every file maps to it
        page._set_instrument(InstrumentGeometry("psic", (("del", "delta"),)))
        assert page._value("source_motors") == "del_value=delta, nu_value=nu"
        other = tmp_path / "stacked.json"
        other.write_text(json.dumps({"trans_function": {"pos_names": ["del_angle", "nu_angle"]}}))
        page.set_field("geometry_path", str(other))
        assert page._value("source_motors") == "del_angle=delta, nu_angle=nu"
        page.set_field("geometry_path", str(tmp_path / "missing.json"))  # unreadable: untouched
        assert page._value("source_motors") == "del_angle=delta, nu_angle=nu"
        # the geometry travels with the settings, and older settings' del/nu names migrate
        restored = StitchPage(empty_host_services(_Status()))
        try:
            restored.apply_settings(page.settings())
            assert restored._instrument().motor("del") == "delta"
            legacy = {"fields": {"del_motor": "tth2", "nu_motor": "gam"}, "choices": {}}
            restored._set_calibration_settings(legacy)
            assert (restored._instrument().motor("del"), restored._instrument().motor("nu")) == ("tth2", "gam")
        finally:
            _close(restored, qapp)
    finally:
        _close(page, qapp)


def _edit_mapping(page, text):
    """As the user does: type in the Angle mapping and leave the box."""
    editor = page._rows["source_motors"].editor
    editor.setText(text)
    editor.editingFinished.emit()


def test_a_custom_position_keeps_its_explicit_motor(qapp, tmp_path):
    # Codex round 11 F2: a calibration's own position names, bound explicitly
    import json
    from pathlib import Path
    from xrd_tools.session.instrument import InstrumentGeometry

    form = _native_form(tmp_path, "multigeometry", scans="1")
    record = json.loads(Path(form.geometry_path).read_text())
    record["pos_names"] = record["trans_function"]["pos_names"] = ["dval", "nval"]
    record["trans_function"].update(rot1_expr="rot1_scale * nval + rot1_offset",
                                    rot2_expr="rot2_scale * dval + rot2_offset")
    custom = tmp_path / "custom.json"
    custom.write_text(json.dumps(record))
    form = replace(form, geometry_path=custom, source_motors=(("dval", "del"), ("nval", "nu")))
    assert prepare_stitch_tool(form).request is not None  # the headless owner admits it
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.load_form(form)
        assert page._value("source_motors") == "dval=del, nval=nu"
        assert page.build_form().source_motors == (("dval", "del"), ("nval", "nu"))
        assert prepare_stitch_tool(page.build_form()).request is not None
        page._set_instrument(InstrumentGeometry("psic", (("del", "delta"),)))  # names neither position
        assert page._value("source_motors") == "dval=del, nval=nu"
        restored = StitchPage(empty_host_services(_Status()))
        try:
            restored.apply_settings(page.settings())
            assert restored._value("source_motors") == "dval=del, nval=nu"
        finally:
            _close(restored, qapp)
        # editable: a custom position's motor is set here; a circle's renames it under Geometry
        assert not page._rows["source_motors"].editor.isReadOnly()
        _edit_mapping(page, "dval=del, nval=nu2")
        assert page._value("source_motors") == "dval=del, nval=nu2"
        page.set_field("geometry_path", str(tmp_path / "source" / "geometry.json"))  # del/nu positions
        assert page._value("source_motors") == "del_angle=delta, nu_angle=nu"
        _edit_mapping(page, "del_angle=tth9, nu_angle=nu")
        assert page._instrument().motor("del") == "tth9"
        assert page._value("source_motors") == "del_angle=tth9, nu_angle=nu"
    finally:
        _close(page, qapp)


def test_the_calibration_must_drive_the_geometrys_arm(qapp, tmp_path):
    # Codex round 11 F3: choosing fourc must not stitch with psic's del-on-nu calibration
    import json
    from xrd_tools.core.containers import PONI
    from xrd_tools.core.geometry import DetectorCalibration
    from xrd_tools.core.geometry.stacked_arm import StackedDetectorArm
    from xrd_tools.session.instrument import InstrumentGeometry

    form = _native_form(tmp_path, "multigeometry", scans="1")
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.load_form(form)
        assert page._readiness[1], page._readiness
        page._calibrate()  # through the window, as a user chooses the geometry
        editor = page._calibration_dialog.geometry_editor
        editor.geometry.setCurrentIndex(editor.geometry.findData("fourc"))
        editor.motor_edit("tth").setText("del")
        page._calibration_dialog.hide()
        assert not page._readiness[1] and "fourc has one detector circle" in page._readiness[0]
        assert not page.run_bar.startButton.isEnabled()
        with pytest.raises(ValueError, match="one detector circle"):
            page.build_form()
        page.begin_run()  # refused before anything is written
        assert not list((tmp_path / "output").glob("*.nexus"))
        # the same detector calibrated as fourc's 2θ arm: del is its motor, nu is never read
        calibration = DetectorCalibration(PONI(dist=.2, poni1=.01677, poni2=.04188, wavelength=1e-10,
                                               detector="Pilatus100k"), detector_config={"orientation": 3})
        single = tmp_path / "single.json"
        single.write_text(json.dumps(StackedDetectorArm(nu_motor=None, del_motor="tth")
                                     .to_pyfai_goniometer(calibration)))
        page.set_field("geometry_path", str(single))
        assert page._value("source_motors") == "del_angle=del"
        assert page._readiness[1], page._readiness
        request = prepare_stitch_tool(page.build_form()).request
        assert {name for item in request.manifest.contributions for name, _o, _v in item.values} == {"del"}
        page._set_instrument(InstrumentGeometry("psic"))  # psic's nu moves
        assert not page._readiness[1] and "nu held at zero" in page._readiness[0]
        # PONI + offsets: a fresh mapping takes the Geometry's arm, one reference per circle
        page.set_field("geometry_kind", "PONI + offsets")
        assert (page._value("source_motors"), page._value("poni_references")) == ("del=del, nu=nu", "del=0, nu=0")
        page.set_field("poni_references", "del=12.5, nu=0")
        # an explicit mapping is kept as it is (Codex round 12 F2): twoc refuses psic's del and nu
        page._set_instrument(InstrumentGeometry("twoc", (("tth", "del"),)))
        assert (page._value("source_motors"), page._value("poni_references")) == ("del=del, nu=nu",
                                                                                  "del=12.5, nu=0")
        assert not page._readiness[1] and "twoc has one detector circle" in page._readiness[0]
        _edit_mapping(page, "del=del")  # the user maps twoc's arm alone
        assert (page._value("source_motors"), page._value("poni_references")) == ("del=del", "del=12.5")
        assert page._readiness[1], page._readiness
        # xrayutilities fits psic's arm only (its single arm comes with the XU calibration work)
        page.set_field("geometry_kind", "Goniometer JSON")
        page.set_field("backend", "xrayutilities histogram")
        assert not page._readiness[1] and "xrayutilities backend" in page._readiness[0]
    finally:
        _close(page, qapp)


def _custom_positions_form(tmp_path):
    """A valid goniometer JSON whose positions are named dval and nval, bound to del and nu."""
    import json
    from pathlib import Path
    form = _native_form(tmp_path, "multigeometry", scans="1")
    record = json.loads(Path(form.geometry_path).read_text())
    record["pos_names"] = record["trans_function"]["pos_names"] = ["dval", "nval"]
    record["trans_function"].update(rot1_expr="rot1_scale * nval + rot1_offset",
                                    rot2_expr="rot2_scale * dval + rot2_offset")
    custom = tmp_path / "custom.json"
    custom.write_text(json.dumps(record))
    return replace(form, geometry_path=custom, source_motors=(("dval", "del"), ("nval", "nu")))


def _read_columns(form):
    request = prepare_stitch_tool(form).request
    return {name for item in request.manifest.contributions for name, _o, _v in item.values}


def test_a_custom_position_calibration_is_checked_by_the_circles_it_moves(qapp, tmp_path):
    # Codex round 12 F1: dval/nval bound to del and nu move both circles, whatever their names
    from xrd_tools.session.instrument import InstrumentGeometry
    form = _custom_positions_form(tmp_path)
    assert _read_columns(form) == {"del", "nu"}
    page = StitchPage(empty_host_services(_Status()))
    try:
        page._set_instrument(InstrumentGeometry("fourc", (("tth", "del"),)))  # chosen before the form
        page.load_form(form)
        assert page._value("source_motors") == "dval=del, nval=nu"  # kept (round 11 F2)
        assert not page._readiness[1] and "fourc has one detector circle" in page._readiness[0]
        assert not page.run_bar.startButton.isEnabled()
        with pytest.raises(ValueError, match="one detector circle"):
            page.build_form()
        page.begin_run()  # refused before anything is written
        assert not list((tmp_path / "output").glob("*.nexus"))
        page._set_instrument(InstrumentGeometry("psic"))
        assert page._readiness[1], page._readiness
        assert _read_columns(page.build_form()) == {"del", "nu"}
    finally:
        _close(page, qapp)


def test_a_poni_form_keeps_its_mapping_and_references_under_any_geometry(qapp, tmp_path):
    # Codex round 12 F2: loading never drops a PONI form's nu mapping or reference
    from xrd_tools.analysis.stitch_operation import StitchGeometryKind
    from xrd_tools.session.instrument import InstrumentGeometry
    form = _native_form(tmp_path, "multigeometry", scans="1")
    poni = tmp_path / "frame.poni"
    poni.write_text('poni_version: 2.1\nDetector: Pilatus100k\nDetector_config: {"orientation": 3}\n'
                    'Distance: 0.2\nPoni1: 0.01677\nPoni2: 0.04188\nRot1: 0\nRot2: 0\nRot3: 0\n'
                    'Wavelength: 1e-10\n')
    two = replace(form, geometry_path=poni, geometry_kind=StitchGeometryKind.PONI,
                  source_motors=(("del", "del"), ("nu", "nu")), poni_references=(("del", 0.), ("nu", 3.)))
    assert _read_columns(two) == {"del", "nu"}  # the headless owner reads both
    page = StitchPage(empty_host_services(_Status()))
    try:
        page._set_instrument(InstrumentGeometry("fourc", (("tth", "del"),)))  # the other geometry first
        page.load_form(two)
        assert (page._value("source_motors"), page._value("poni_references")) == ("del=del, nu=nu",
                                                                                  "del=0.0, nu=3.0")
        assert not page._readiness[1] and "fourc has one detector circle" in page._readiness[0]
        with pytest.raises(ValueError, match="one detector circle"):
            page.build_form()
        restored = StitchPage(empty_host_services(_Status()))  # the saved-settings entry point too
        try:
            restored.apply_settings(page.settings())
            assert restored._instrument().geometry == "fourc"
            assert (restored._value("source_motors"), restored._value("poni_references")) == (
                "del=del, nu=nu", "del=0.0, nu=3.0")
            assert not restored._readiness[1] and "fourc has one detector circle" in restored._readiness[0]
        finally:
            _close(restored, qapp)
        page._set_instrument(InstrumentGeometry("psic"))
        assert page._readiness[1], page._readiness
        built = page.build_form()
        assert (built.source_motors, built.poni_references) == ((("del", "del"), ("nu", "nu")),
                                                                (("del", 0.), ("nu", 3.)))
        assert _read_columns(built) == {"del", "nu"}
        # the reverse: a form mapping del alone, loaded under psic, stays del alone and is refused
        page.load_form(replace(two, source_motors=(("del", "del"),), poni_references=(("del", 0.),)))
        assert (page._value("source_motors"), page._value("poni_references")) == ("del=del", "del=0.0")
        assert not page._readiness[1] and "nu held at zero" in page._readiness[0]
        page._set_instrument(InstrumentGeometry("twoc", (("tth", "del"),)))
        assert page._readiness[1], page._readiness
        assert _read_columns(page.build_form()) == {"del"}
    finally:
        _close(page, qapp)


def test_a_horizontal_2theta_arm_maps_nu_and_refuses_a_vertical_calibration(qapp, tmp_path):
    import json
    from xrd_tools.core.containers import PONI
    from xrd_tools.core.geometry import DetectorCalibration
    from xrd_tools.core.geometry.stacked_arm import StackedDetectorArm
    from xrd_tools.session.instrument import InstrumentGeometry
    form = _native_form(tmp_path, "multigeometry", scans="1")
    calibration = DetectorCalibration(PONI(dist=.2, poni1=.01677, poni2=.04188, wavelength=1e-10,
                                           detector="Pilatus100k"), detector_config={"orientation": 3})
    files = {}
    for name, arm in (("horizontal", StackedDetectorArm(nu_motor="tth", del_motor=None)),
                      ("vertical", StackedDetectorArm(nu_motor=None, del_motor="tth"))):
        files[name] = tmp_path / f"{name}.json"
        files[name].write_text(json.dumps(arm.to_pyfai_goniometer(calibration)))
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.load_form(form)
        page._set_instrument(InstrumentGeometry("twoc", (("tth", "nu"),), "horizontal"))  # the scan's nu is 2θ
        page.set_field("geometry_path", str(files["horizontal"]))
        assert page._value("source_motors") == "nu_angle=nu"
        assert page._readiness[1], page._readiness
        assert _read_columns(page.build_form()) == {"nu"}
        page.set_field("geometry_path", str(files["vertical"]))
        assert not page._readiness[1] and "Choose the vertical scattering plane" in page._readiness[0]
        page.set_field("geometry_kind", "PONI + offsets")  # a fresh PONI mapping: the horizontal arm, nu
        assert (page._value("source_motors"), page._value("poni_references")) == ("nu=nu", "nu=0")
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
        # Stitch offers Single and Overlay only, and no Waterfall options; Int keeps them all
        modes = lambda combo: [combo.itemText(row) for row in range(combo.count())]  # noqa: E731
        assert modes(view.plot_mode) == ["Single", "Overlay"]
        assert view.plot_options_dialog._waterfall_group.isHidden()
        with pytest.raises(ValueError, match="unknown plot modes"):
            view.set_plot_modes("Stack")
        int_view = ScientificView()
        try:
            assert modes(int_view.plot_mode) == ["Single", "Overlay", "Average", "Sum", "Waterfall"]
            assert not int_view.plot_options_dialog._waterfall_group.isHidden()
        finally:
            int_view.deleteLater()
    finally:
        _close(page, qapp)
