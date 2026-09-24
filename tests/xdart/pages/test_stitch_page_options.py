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
        page.set_field("polarization", "")
        assert page.build_form().polarization_factor is None
        page.set_field("polarization", "strong")
        with pytest.raises(ValueError, match="Polarization"):
            page.build_form()
    finally:
        _close(page, qapp)


def test_xu_backend_greys_pyfai_options_and_keeps_the_frame_check(qapp, tmp_path):
    form = _native_form(tmp_path, "xu_hist", scans="1")
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.load_form(form)
        assert not page.corrections_card.isEnabled()
        assert not page._rows["sample_position"].editor.isEnabled()
        for key, value in (("frame_check", "Keep all frames"), ("polarization", "0.5"),
                           ("dark_path", str(tmp_path / "absent.npy"))):
            page.set_field(key, value)
        built = page.build_form()
        assert (built.frame_screen, built.polarization_factor, built.dark_path) == ("none", .99, None)
        plan = prepare_stitch_tool(built).request.plan
        assert type(plan) is XuStitchOperationPlan and plan.frame_screen.exclude == "none"
        page.set_field("backend", "pyFAI MultiGeometry")
        assert page.corrections_card.isEnabled() and page._rows["sample_position"].editor.isEnabled()
    finally:
        _close(page, qapp)


def test_form_refuses_pyfai_options_outside_native_pyfai(tmp_path):
    xu = _native_form(tmp_path, "xu_hist", scans="1")
    with pytest.raises(ValueError, match="pinned solid angle"):
        replace(xu, error_model="poisson")
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
