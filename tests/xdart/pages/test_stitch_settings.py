"""Saved Stitch settings: File ▸ Save/Load Settings and the last session restore every
page input, the source files, the threshold policy and the calibration dialog, but
never its per-use confirmations."""
import json
from pathlib import Path

import numpy as np
import pytest

from xdart.gui.pages.catalog import STITCH_PAGE
from xdart.gui.pages.services import empty_host_services
from xdart.gui.pages.values import ActionCompleted, ActionRefused, PageCapability, PageCleanup
from xdart.gui.tabs.stitch.page import StitchPage, build_stitch_page
from xdart.gui.tabs.stitch.settings import SESSION_KEY, dump_settings, load_settings
from tests.xdart.pages.test_stitch_page import _Status, qapp  # noqa: F401
from tests.xdart.tools.test_stitch_native_values import _native_form


def _close(page, qapp):
    assert page.shutdown().status is PageCleanup.CLEAN
    page.deleteLater()
    qapp.processEvents()


def _edited_page(tmp_path, backend="multigeometry"):
    form = _native_form(tmp_path, backend, scans="1")
    dark = tmp_path / "dark.npy"
    np.save(dark, np.ones(form.detector_shape))
    page = StitchPage(empty_host_services(_Status()))
    page.load_form(form)
    for key, value in (("frame_check", "Exclude empty and partial frames"),
                       ("sample_position", "On rotation axis"), ("polarization", "0.95"),
                       ("error_model", "Poisson"), ("dark_path", str(dark)), ("flat_cutoff", "25"),
                       ("npt_1d", "80"), ("output_family", "renamed")):
        page.set_field(key, value)
    return page


@pytest.mark.parametrize(("backend", "manual"), (("multigeometry", False), ("multigeometry", True),
                                                 ("xu_hist", False)))
def test_settings_round_trip_every_input(qapp, tmp_path, backend, manual):
    page = _edited_page(tmp_path, backend)
    restored = StitchPage(empty_host_services(_Status()))
    try:
        if manual:  # a 2θ axis with a manual range and a manual intensity threshold
            for key, value in (("unit", "2θ (°)"), ("radial_auto", False), ("q_min", 10.0),
                               ("q_max", 60.0), ("threshold", 700000.0)):
                page.set_field(key, value)
            assert page._threshold_mode == "manual"
        built = page.build_form()
        if manual:
            assert built.unit == "2th_deg" and built.q_range == (10.0, 60.0)
            assert built.conditioning.mode == "manual" and built.conditioning.threshold_max == 700000.0
        document = load_settings(dump_settings(page.settings()))
        assert document["sources"]["Spec"] == [str(path) for path in built.source_selection.paths]
        restored.apply_settings(document)
        again = restored.build_form()
        assert again.fingerprint == built.fingerprint
        assert (again.flat_min_fraction, again.polarization_factor, again.npt_1d) == (.25, .95, 80)
        if backend == "xu_hist":  # the asset-owned rows are shown from the asset and locked
            assert restored._value("source_motors") == page._value("source_motors")
            assert not restored._rows["source_motors"].isEnabled()
    finally:
        _close(page, qapp)
        _close(restored, qapp)


def _three_nu_rows():
    return tuple(type("Frame", (), {"complete": True, "motors": {"del": 5., "nu": nu}})()
                 for nu in (-10., 3., 16.))


def test_calibration_dialog_settings_are_kept_but_not_its_confirmations(qapp, tmp_path):
    page = _edited_page(tmp_path)
    restored = StitchPage(empty_host_services(_Status()))
    try:
        page._calibrate()
        dialog = page._calibration_dialog
        assert dialog is not None
        # the extended arm needs completed frames on 3 nu positions before it can be chosen
        dialog.frames = _three_nu_rows()
        dialog._sync_arm_model()
        extended = dialog.arm_model.findData("stacked_extended")
        assert dialog.arm_model.isEnabled()
        dialog.arm_model.setCurrentIndex(extended)
        dialog.arm_model.activated.emit(extended)  # the user's choice
        dialog.refine_calibrant.setCurrentIndex(dialog.refine_calibrant.count() - 1)
        dialog._set_field_text("rms", ".02")
        dialog.refine_confirm.setChecked(True)
        dialog.compatible.setChecked(True)
        saved = page.settings()["calibration"]
        assert saved["choices"]["arm_model"]["data"] == "stacked_extended"
        assert saved["fields"]["rms"] == ".02"
        assert "refine_confirm" not in json.dumps(saved) and "compatible" not in json.dumps(saved)

        restored.apply_settings(page.settings())
        restored._calibrate()
        again = restored._calibration_dialog
        assert again.refine_calibrant.currentText() == dialog.refine_calibrant.currentText()
        custom = dict(page.settings()["calibration"])
        custom["choices"] = {**custom["choices"], "refine_calibrant": {"data": None, "text": "NIST-SRM-660c"}}
        again.apply_settings(custom)  # an editable choice keeps a typed-in entry
        assert again.refine_calibrant.currentText() == "NIST-SRM-660c"
        assert again._text("rms") == ".02"
        assert not again.refine_confirm.isChecked() and not again.compatible.isChecked()
        # a new session has no completed frames: five-parameter shown, the choice kept
        assert again.arm_model.currentData() == "stacked"
        assert again.settings()["choices"]["arm_model"]["data"] == "stacked_extended"
        again.frames = _three_nu_rows()  # once it can apply, the saved choice does
        again._sync_arm_model()
        assert again.arm_model.currentData() == "stacked_extended"
    finally:
        _close(page, qapp)
        _close(restored, qapp)


def test_file_save_and_load_through_the_settings_port(qapp, tmp_path):
    assert PageCapability.SETTINGS_PERSISTENCE in STITCH_PAGE.capabilities
    page = _edited_page(tmp_path)
    handle = build_stitch_page(empty_host_services(_Status()), None)
    other = handle.widget
    try:
        page._settings_path_chooser = lambda action, start: str(tmp_path / "stitch-settings")
        assert page.save() == ActionCompleted("stitch-settings-saved")
        saved = tmp_path / "stitch-settings.json"
        assert json.loads(saved.read_text())["kind"] == "xdart-stitch-settings"
        other._settings_path_chooser = lambda action, start: str(saved)
        assert handle.settings_io.load() == ActionCompleted("stitch-settings-loaded")
        assert other.build_form().fingerprint == page.build_form().fingerprint

        foreign = tmp_path / "profile.json"
        foreign.write_text(json.dumps({"kind": "xdart-run-intent-profile"}))
        before = other.settings()
        other._settings_path_chooser = lambda action, start: str(foreign)
        refused = handle.settings_io.load()
        assert type(refused) is ActionRefused and "not an xdart Stitch settings file" in refused.reason
        assert other.settings() == before
    finally:
        _close(page, qapp)
        _close(other, qapp)


def test_the_last_session_is_restored_and_a_fresh_start_ignores_it(qapp, tmp_path, monkeypatch):
    session = tmp_path / "session.json"
    monkeypatch.setenv("XDART_SESSION_FILE", str(session))
    monkeypatch.delenv("XDART_SESSION_FRESH", raising=False)
    first = build_stitch_page(empty_host_services(_Status()), None)
    first.widget.set_field("project_root", str(tmp_path))
    first.widget.set_field("npt_1d", "77")
    assert first.close().status is PageCleanup.CLEAN
    first.widget.deleteLater()
    assert json.loads(session.read_text())[SESSION_KEY]["fields"]["npt_1d"] in (77, "77")

    second = build_stitch_page(empty_host_services(_Status()), None)
    try:
        assert str(second.widget._value("npt_1d")) == "77"
        assert str(second.widget._value("project_root")) == str(tmp_path)
    finally:
        assert second.close().status is PageCleanup.CLEAN
        second.widget.deleteLater()

    before = session.read_bytes()
    monkeypatch.setenv("XDART_SESSION_FRESH", "1")
    fresh = build_stitch_page(empty_host_services(_Status()), None)
    try:
        assert str(fresh.widget._value("npt_1d")) != "77"
        fresh.widget.set_field("npt_1d", "55")
    finally:
        assert fresh.close().status is PageCleanup.CLEAN
        fresh.widget.deleteLater()
        qapp.processEvents()
    assert session.read_bytes() == before
