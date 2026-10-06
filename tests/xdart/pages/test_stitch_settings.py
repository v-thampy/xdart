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
        if manual:  # a 2θ axis with a manual range and χ window and a manual intensity threshold
            for key, value in (("unit", "2θ (°)"), ("radial_auto", False), ("q_min", 10.0),
                               ("q_max", 60.0), ("chi1d_auto", False), ("chi1d_min", -60.0),
                               ("chi1d_max", 45.0), ("threshold", 700000.0)):
                page.set_field(key, value)
            assert page._threshold_mode == "manual"
        built = page.build_form()
        if manual:
            assert built.unit == "2th_deg" and built.q_range == (10.0, 60.0)
            assert built.chi_range_1d == (-60.0, 45.0)
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


def test_a_blank_save_path_stays_blank_when_its_file_is_loaded_again(qapp, tmp_path):
    """G13 unit 3a (found in G12): the fields apply in the page's order, as RSM's do, not in the file's sorted
    order, in which the Save Path ("output_path") came before the Project ("project_root"), whose change fills a
    blank Save Path with <Project>/xdart_processed_data. The document is unchanged: it round-trips byte for byte."""
    page = StitchPage(empty_host_services(_Status()))
    restored = StitchPage(empty_host_services(_Status()))
    try:
        page.set_field("project_root", str(tmp_path))
        assert page._value("output_path") == str(tmp_path / "xdart_processed_data")  # a Project fills a blank one
        page.set_field("output_path", "")
        text = dump_settings(page.settings())
        assert '"output_path": ""' in text
        restored.apply_settings(load_settings(text))
        assert (restored._value("project_root"), restored._value("output_path")) == (str(tmp_path), "")
        assert dump_settings(restored.settings()) == text
        page.apply_settings(load_settings(text))  # the page's own file, loaded again
        assert page._value("output_path") == ""
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
        # the arm model is automatic (decision 40): nothing to save; the note follows the frames
        dialog.frames = _three_nu_rows()
        dialog._sync_arm_model()
        assert 'which the frames on 3 nu positions can determine' in dialog.arm_note.text()
        dialog.refine_calibrant.setCurrentIndex(dialog.refine_calibrant.count() - 1)
        dialog._set_field_text("rms", ".02")
        saved = page.settings()["calibration"]
        assert "arm_model" not in saved["choices"] and "backend" not in saved["choices"]
        assert saved["fields"]["rms"] == ".02"

        restored.apply_settings(page.settings())
        restored._calibrate()
        again = restored._calibration_dialog
        assert again.refine_calibrant.currentText() == dialog.refine_calibrant.currentText()
        custom = dict(page.settings()["calibration"])
        custom["choices"] = {**custom["choices"], "refine_calibrant": {"data": None, "text": "NIST-SRM-660c"}}
        again.apply_settings(custom)  # an editable choice keeps a typed-in entry
        assert again.refine_calibrant.currentText() == "NIST-SRM-660c"
        assert again._text("rms") == ".02"
        # an older settings file's model and backend choices are ignored, the rest applies
        older = {**custom, "choices": {**custom["choices"], "arm_model": {"data": "stacked_extended", "text": "x"},
                                        "backend": {"data": "xu_hist", "text": "y"}},
                 "fields": {**custom["fields"], "rms": ".025"}}
        again.apply_settings(older)
        assert again._text("rms") == ".025"
    finally:
        _close(page, qapp)
        _close(restored, qapp)


def test_omitted_calibration_keeps_restored_settings_before_the_window_is_opened(qapp):
    """A partial settings load preserves calibration settings restored at startup, even before Calibrate
    has created a window. An explicit null still clears them."""
    page = StitchPage(empty_host_services(_Status()))
    partial = {"kind": "xdart-stitch-settings", "version": 1, "fields": {}}
    try:
        page.apply_settings({**partial, "calibration": {"fields": {"rms": ".02"}, "choices": {}}})
        saved = page.settings()["calibration"]
        assert saved["fields"]["rms"] == ".02" and page._calibration_dialog is None
        page.apply_settings(partial)
        assert page.settings()["calibration"] == saved
        assert page._calibration_dialog is None
        page.apply_settings({**partial, "calibration": None})
        assert page.settings()["calibration"] is None
    finally:
        _close(page, qapp)


def test_a_load_waits_while_the_calibration_window_works(qapp, tmp_path):
    """G13 unit 3b (found in G12): a settings load waited for Stitch's own run only, so it could change the
    calibration window's inputs while a fit ran. It now waits for the window too, saying why, before any chooser;
    once the window is idle the load goes on."""
    page = _edited_page(tmp_path)
    asked = []
    try:
        page._calibrate()
        dialog = page._calibration_dialog
        assert dialog is not None and not page._owner.busy
        page._settings_path_chooser = lambda action, start: asked.append(action)  # then cancelled
        dialog.active = lambda: True  # a fit running
        assert page.load() == ActionRefused(
            "Stitch's calibration window is working; load settings when it is done")
        assert asked == []
        del dialog.active  # the fit done
        assert page.load() == ActionCompleted("stitch-settings-load-cancelled") and asked == ["load"]
    finally:
        _close(page, qapp)


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
        malformed = page.settings()
        malformed["calibration"] = {"fields": ["rms"], "choices": {}}
        foreign.write_text(json.dumps(malformed))
        refused = handle.settings_io.load()
        assert type(refused) is ActionRefused and "calibration is malformed" in refused.reason
        assert other.settings() == before
    finally:
        _close(page, qapp)
        _close(other, qapp)


def test_the_last_session_is_restored_and_a_fresh_start_ignores_it(qapp, tmp_path, monkeypatch):
    import time
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
        deadline = time.monotonic() + 5
        receipt = second.close()
        while receipt.status is PageCleanup.PENDING and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(.005)
            receipt = second.close()
        assert receipt.status is PageCleanup.CLEAN
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


def test_settings_that_cannot_apply_are_refused_and_leave_the_inputs(qapp, tmp_path, monkeypatch):
    page = _edited_page(tmp_path)
    document = page.settings()
    document["fields"]["source_kind"] = "TIFF stack"  # a source kind this version lacks
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(document))
    before = page.settings()
    try:
        page._settings_path_chooser = lambda action, start: str(broken)
        refused = page.load()
        assert type(refused) is ActionRefused and "cannot be applied" in refused.reason
        assert page.settings() == before
    finally:
        _close(page, qapp)

    session = tmp_path / "session.json"  # the same document as the last session's
    session.write_text(json.dumps({SESSION_KEY: document}))
    monkeypatch.setenv("XDART_SESSION_FILE", str(session))
    monkeypatch.delenv("XDART_SESSION_FRESH", raising=False)
    restored = build_stitch_page(empty_host_services(_Status()), None).widget
    default = StitchPage(empty_host_services(_Status()))
    try:
        assert restored.settings() == default.settings()
    finally:
        _close(restored, qapp)
        _close(default, qapp)


def _calibration_source(dialog):
    return dialog._text("source"), dialog._text("scan"), dialog.camera.value().to_xu()


def test_calibrate_keeps_a_calibration_source_other_than_the_processing_scan(qapp, tmp_path):
    page = _edited_page(tmp_path)
    restored, sourceless = (StitchPage(empty_host_services(_Status())) for _ in range(2))
    calibrant = (str(tmp_path / "powder-calibration.spec"), "2", ("x-", "z-"))  # 17-2's mirrored camera
    try:
        page._calibrate()  # a new dialog starts from the processing scan
        dialog = page._calibration_dialog
        assert _calibration_source(dialog)[1:] == ("1", ("x-", "z+"))  # Stitch's frame, the default
        dialog._set_field_text("source", calibrant[0])
        dialog._set_field_text("scan", calibrant[1])
        from xrd_tools.core.geometry.camera import DetectorCamera
        dialog.camera.set_value(DetectorCamera(*calibrant[2]))
        page._calibrate()  # opening it again keeps the calibrant scan
        assert _calibration_source(dialog) == calibrant

        restored.apply_settings(page.settings())
        restored._calibrate()  # so do the saved settings
        assert _calibration_source(restored._calibration_dialog) == calibrant
        unnamed = page.settings()
        unnamed["calibration"]["fields"]["source"] = ""  # saved without a calibration source
        sourceless.apply_settings(unnamed)
        sourceless._calibrate()
        assert _calibration_source(sourceless._calibration_dialog)[1:] == ("1", ("x-", "z+"))
    finally:
        _close(page, qapp)
        _close(restored, qapp)
        _close(sourceless, qapp)


def test_a_document_saved_without_a_calibration_window_closes_the_one_opened_since(qapp, tmp_path):
    """Codex's combined stage-2b review F3 (its same-page reproduction; pre-existing in Stitch): a document saved
    before Calibrate was opened holds ``calibration: null``. Loaded after the window was opened and edited (RMS
    .02) and hidden, it restores that: the next document has no calibration window, and Calibrate opens a new
    window on the defaults. The window is let go as a Project change lets it go: an idle one is shut down at once and leaves the
    page's windows, so repeated loads do not pile up hidden windows. A document without the section keeps the window and its edits, as before."""
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.set_field("project_root", str(tmp_path))
        original = page.settings()
        assert original["calibration"] is None
        page._calibrate()
        dialog = page._calibration_dialog
        default_rms = dialog._text("rms")
        dialog._set_field_text("rms", ".02")
        dialog.hide()
        assert default_rms != ".02" and page.settings()["calibration"]["fields"]["rms"] == ".02"
        page.apply_settings(original)
        assert page.settings()["calibration"] is None and page.settings() == original
        assert page._calibration_dialog is None and not dialog.isVisible()
        assert dialog not in page._authoring_dialogs  # idle: shut down and let go at once, its memory with it
        page._calibrate()
        again = page._calibration_dialog
        assert again is not dialog and again._text("rms") == default_rms
        again._set_field_text("rms", ".02")
        again.hide()
        partial = {key: value for key, value in original.items() if key != "calibration"}
        page.apply_settings(partial)  # no calibration section: the window stays, edits and all
        assert page._calibration_dialog is again and page.settings()["calibration"]["fields"]["rms"] == ".02"
    finally:
        _close(page, qapp)
