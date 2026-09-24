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


def test_legacy_mode_greys_the_native_only_options(qapp, tmp_path):
    (tmp_path / "legacy").mkdir()
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.load_form(_form(tmp_path / "legacy", "multigeometry"))
        assert not page.corrections_card.isEnabled()
        for key in ("frame_check", "sample_position", "solid_angle", "error_model"):
            assert not page._rows[key].editor.isEnabled(), key
        page.load_form(_native_form(tmp_path, "multigeometry", scans="1"))
        assert page.corrections_card.isEnabled()
        for key in ("frame_check", "sample_position", "solid_angle", "error_model"):
            assert page._rows[key].editor.isEnabled(), key
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
