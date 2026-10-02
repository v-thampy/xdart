"""Saved RSM settings (AUD-9 stage 2b G12; ledger row 3, the maintainer, 2026-09-24: "File ▸ Save/Load Settings and
last-session restore covering every input, the Advanced options and any calibration panel, never per-use
confirmations, and a fresh start neither reading nor writing it; mirror the Stitch page"), through the shared
settings owner (``tools/page_settings``) as Stitch's. Stitch's own rows (test_stitch_settings.py) are unchanged; the
host's File menu reaching both pages is test_scattering_page.py's."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

pytest.importorskip("pyqtgraph")

from xdart.gui.pages.catalog import RSM_PAGE
from xdart.gui.pages.services import empty_host_services
from xdart.gui.pages.values import ActionCompleted, ActionRefused, PageCapability, PageCleanup
from xdart.gui.tabs.rsm.page import RSMPage, build_rsm_page
from xdart.gui.tabs.rsm.settings import RSM_SETTINGS, SESSION_KEY
from xdart.gui.tabs.scattering.controls_inventory import THRESHOLD_MAX, THRESHOLD_MIN
from xdart.gui.tools.page_settings import PageSettings, SettingsKind
from xrd_tools.io.analysis_artifact import AnalysisArtifactOverwrite
from xrd_tools.rsm.coordinate_frame import RSMCoordinateFrame
from tests.core.test_rsm_calibration_geometry import _canonical_partner
from tests.xdart.pages.test_rsm_page import (  # noqa: F401
    _Owner, _Status, _close, _page, _scan43_dialog_form, _sto_align, _synthetic_source, _with_source, qapp,
)

#: The scan-43 settings file (design §2 item 5: the "Scan 43" preset button went with the classic dialog)
_SCAN43 = Path(__file__).parent / "fixtures" / "rsm_scan43_settings.json"
#: Not saved field by field: the SPEC files row shows the selection, and Int's Threshold is one policy.
_NOT_FIELDS = {"spec_path", "threshold_enabled", "threshold_min", "threshold"}


@pytest.fixture
def session(tmp_path, monkeypatch):
    """The last session's file, here (the page builder reads it)."""
    path = tmp_path / "session.json"
    monkeypatch.setenv("XDART_SESSION_FILE", str(path))
    monkeypatch.delenv("XDART_SESSION_FRESH", raising=False)
    return path


def _every_input(root):
    """A value other than its default for every saved input: Project, Experiment (with the refinement toggle),
    Source (the Selection and the UB group), Processing (Coordinates, Grid, Offset, All-frame hot, Normalization)
    and every Advanced row (decoder, Angle mapping, frames, occurrences, Corrections, chunk and memory)."""
    return {
        "geometry_path": "calibration/rsm_canonical.json", "mask_path": "mask.npy", "refinement": False,
        "source_kind": "NeXus", "subdirs": True, "name_filter": "scan -bad", "image_dir": "source/images",
        "image_stem": "{spec}_scan{scan}_", "scan_mode": "All Scans", "scan": "[1-2]",
        "ub_from": "Typed", "ub_row1": "1 0 0", "ub_row2": "0 1 0", "ub_row3": "0 0 1",
        "ub_spec_path": "ub.spec", "ub_scan": "7",
        "project_root": str(root), "output_path": str(root / "named_rsm.nexus"),
        "coordinates": RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU.display_name, "grid": "Full 200×200×200",
        "offset": "0.5", "static_hot": "", "normalization": "Monitor", "monitor": "I0",
        "foil_column": "filters", "exposure_column": "sec", "foil_lengths": "1, 2, 3, 4",
        "adv_rows": "4", "adv_columns": "5", "adv_dtype": "uint16", "adv_header": "8",
        "adv_angle_mapping": "mu=mu, eta=eta, chi=chi, phi=phi, nu=Nu#1, del=del",
        "frame_first": "1", "frame_last": "9", "frame_step": "2",
        "adv_foil_occurrence": "1", "adv_exposure_occurrence": "1", "adv_monitor_occurrence": "2",
        "solid_angle": "Off", "polarization": "0.95", "dark_path": "dark.npy", "flat_path": "flat.npy",
        "flat_cutoff": "25", "adv_chunk": "4", "adv_max_frame_mib": "32", "adv_max_chunk_mib": "128",
    }


def test_settings_round_trip_every_rsm_input(qapp, tmp_path):
    """Every input, Int's threshold policy, each source kind's own files, the calibration window's settings and
    Append ⇄ Replace survive the document's text into another page. The table holds every saved field: an input
    added without a value here fails this row."""
    page, restored = _page(qapp, owner=_Owner()), _page(qapp, owner=_Owner())
    try:
        values = _every_input(tmp_path)
        assert set(values) == set(page.settings()["fields"]) == set(page._fields) - _NOT_FIELDS
        assert [key for key, value in values.items() if page._value(key) == value] == []  # each one a change
        for kind, paths in (("Spec", ["a.spec", "b.spec"]), ("Directory", ["folder"]), ("NeXus", ["x.nxs"])):
            page.set_field("source_kind", kind)
            page.set_sources(paths)
        for key, value in values.items():
            page.set_field(key, value)
        page._threshold_edited(THRESHOLD_MAX, 50000.0)  # Int's Manual: a High, then a Low
        page._threshold_edited(THRESHOLD_MIN, 2.0)
        calibration = {"fields": {"rms": ".02", "source": "calibrant.spec"}, "choices": {}}
        page._set_calibration_settings(calibration)
        page.run_bar.set_write_mode("Overwrite")
        page._write_mode_changed()
        assert {key: page._value(key) for key in values} == values

        document = RSM_SETTINGS.load(RSM_SETTINGS.dump(page.settings()))
        assert document["sources"] == {"Spec": ["a.spec", "b.spec"], "Directory": ["folder"], "NeXus": ["x.nxs"]}
        assert document["threshold"] == {"apply_threshold": True, "threshold_min": 2.0, "threshold_max": 50000.0,
                                         "mask_saturation": False}
        assert document["calibration"] == calibration and document["write_mode"] == "Overwrite"
        restored.apply_settings(document)
        assert {key: restored._value(key) for key in values} == values
        assert restored.settings() == page.settings()
        assert restored._threshold_state == page._threshold_state
        assert restored._source_paths == ("x.nxs",) and restored._value("spec_path") == "x.nxs"
        assert restored.run_bar.write_mode() == "Overwrite" and not restored._output_suggested
        assert restored._calibration_settings_now() == calibration
        restored.set_field("source_kind", "Spec")  # each kind keeps its own files
        assert restored._source_paths == ("a.spec", "b.spec")
    finally:
        _close(page, qapp)
        _close(restored, qapp)


def test_a_suggested_output_is_saved_as_following_the_source(qapp, tmp_path):
    """The Output follows the Source until a name is chosen (stage 2); a document keeps that, not the path it
    suggested then, so a restored page (or another Project) suggests its own. A chosen name is kept."""
    page, restored = _page(qapp, owner=_Owner()), _page(qapp, owner=_Owner())
    try:
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path)
        assert page._output_suggested and page._value("output_path").endswith("_rsm.nexus")
        assert page.settings()["fields"]["output_path"] == ""
        restored.apply_settings(page.settings())
        restored.resolve_source()
        assert restored._output_suggested and restored._value("output_path") == page._value("output_path")
        assert restored.build_form().fingerprint == page.build_form().fingerprint
        named = str(tmp_path / "mine_rsm.nexus")
        page.set_field("output_path", named)
        restored.apply_settings(page.settings())
        assert not restored._output_suggested and restored._value("output_path") == named
        assert restored.build_form().fingerprint == page.build_form().fingerprint
    finally:
        _close(page, qapp)
        _close(restored, qapp)


def test_an_older_or_partial_document_loads_what_it_holds(qapp, tmp_path):
    """Older or partial documents load what they can: a field or a section the document lacks (the sources, the
    threshold, the calibration window, Append ⇄ Replace) keeps the page's own, and one this version does not
    know is ignored."""
    page = _page(qapp, owner=_Owner())
    try:
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path)
        page._threshold_edited(THRESHOLD_MAX, 50000.0)
        page._set_calibration_settings({"fields": {"rms": ".02"}, "choices": {}})
        page.run_bar.set_write_mode("Overwrite")
        page._write_mode_changed()
        before = page.settings()
        page.apply_settings({"kind": "xdart-rsm-settings", "version": 1, "retired_section": {"x": 1},
                             "fields": {"grid": "Full 200×200×200", "retired_row": "x"}})
        assert page.settings() == {**before, "fields": {**before["fields"], "grid": "Full 200×200×200"}}
    finally:
        _close(page, qapp)


def test_file_save_and_load_through_the_settings_port_and_what_is_refused(qapp, tmp_path, session):
    """File ▸ Save/Load Settings (the host's, through the page's settings port; Stitch's chooser and messages):
    the saved document loads into another page, which then makes the same form. A foreign, newer or malformed
    document, a choice this version lacks, or a value its row cannot hold is refused with its reason, the inputs
    left as they were; a load while the page works is refused before any chooser. No confirmation is asked."""
    assert PageCapability.SETTINGS_PERSISTENCE in RSM_PAGE.capabilities
    assert RSM_SETTINGS == SettingsKind("xdart-rsm-settings", 1, "rsm_settings", "RSM")
    page = _page(qapp, owner=_Owner())
    handle = build_rsm_page(empty_host_services(_Status()), None)
    other = handle.widget
    try:
        assert type(other._saved_settings) is PageSettings and other._saved_settings.kind is RSM_SETTINGS
        assert handle.settings_io is other
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path)
        page.set_field("coordinates", RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU.display_name)
        page.set_field("grid", "Full 200×200×200")
        page._settings_path_chooser = lambda action, start: str(tmp_path / "rsm-settings")
        assert page.save() == ActionCompleted("rsm-settings-saved")
        saved = tmp_path / "rsm-settings.json"
        assert json.loads(saved.read_text())["kind"] == "xdart-rsm-settings"
        assert page.run_bar.readinessLabel.full_text() == "RSM settings saved: rsm-settings.json"
        asked = []
        other._settings_path_chooser = lambda action, start: asked.append(action) or str(saved)
        assert handle.settings_io.load() == ActionCompleted("rsm-settings-loaded")
        assert other.run_bar.readinessLabel.full_text() == "RSM settings loaded: rsm-settings.json"
        other.resolve_source()
        assert other.build_form().fingerprint == page.build_form().fingerprint

        good = json.loads(saved.read_text())
        foreign = tmp_path / "foreign.json"
        other._settings_path_chooser = lambda action, start: str(foreign)
        for document, reason in (
                ({"kind": "xdart-run-intent-profile"}, "not an xdart RSM settings file"),
                ({**good, "kind": "xdart-stitch-settings"}, "not an xdart RSM settings file"),
                ({**good, "version": 2}, "these RSM settings come from a newer xdart"),
                ({**good, "fields": ["grid"]}, "RSM settings fields are malformed"),
                ({**good, "calibration": {"fields": ["rms"], "choices": {}}}, "RSM settings calibration is malformed"),
                ({**good, "fields": {**good["fields"], "coordinates": "Q (lab)"}},
                 "Coordinates: 'Q (lab)' is not one of its choices"),
                ({**good, "fields": {**good["fields"], "grid": "Huge"}}, "Grid: 'Huge' is not one of its choices"),
                ({**good, "fields": {**good["fields"], "source_kind": "TIFF stack"}},
                 "Source: 'TIFF stack' is not one of its choices"),
                ({**good, "fields": {**good["fields"], "refinement": "yes"}},
                 "Use the calibration's refinement: 'yes' is not on or off"),
                ({**good, "fields": {**good["fields"], "scan": [43]}}, "Selection: [43] is not text or a number")):
            before = other.settings()
            foreign.write_text(json.dumps(document))
            assert handle.settings_io.load() == ActionRefused(reason), reason
            assert other.settings() == before, reason
            assert other.run_bar.readinessLabel.full_text() == f"Could not load the RSM settings: {reason}"
        foreign.write_text("{not json")
        assert handle.settings_io.load() == ActionRefused("not an xdart RSM settings file")

        other._settings_path_chooser = lambda action, start: asked.append(action) or str(saved)
        other._job_form = object()  # a queue between its RSMs: the page is working
        assert other.active() and handle.settings_io.load() == ActionRefused(
            "RSM is busy; load settings when it is idle")
        assert asked == ["load"]  # refused before any chooser
        other._job_form = None
        other._settings_path_chooser = lambda action, start: None
        assert handle.settings_io.load() == ActionCompleted("rsm-settings-load-cancelled")
        assert handle.settings_io.save() == ActionCompleted("rsm-settings-save-cancelled")
    finally:
        _close(page, qapp)
        _close(other, qapp)


def test_the_last_session_is_restored_and_a_fresh_start_neither_reads_nor_writes_it(qapp, tmp_path, session,
                                                                                      monkeypatch):
    """A clean close keeps the page's settings for the next session, which the page builder restores; a fresh
    start (``xdart -f``: XDART_SESSION_FRESH) restores nothing, and its close writes nothing."""
    first = build_rsm_page(empty_host_services(_Status()), None)
    first.widget.set_field("project_root", str(tmp_path))
    first.widget.set_field("grid", "Full 200×200×200")
    first.widget._threshold_edited(THRESHOLD_MAX, 50000.0)
    kept = first.widget._threshold_state
    assert first.close().status is PageCleanup.CLEAN
    first.widget.deleteLater()
    remembered = json.loads(session.read_text())[SESSION_KEY]
    assert remembered["fields"]["grid"] == "Full 200×200×200" and remembered["threshold"]["threshold_max"] == 50000.0

    second = build_rsm_page(empty_host_services(_Status()), None)
    try:
        assert second.widget._value("grid") == "Full 200×200×200"
        assert second.widget._value("project_root") == str(tmp_path) and second.widget._threshold_state == kept
    finally:
        assert second.close().status is PageCleanup.CLEAN
        second.widget.deleteLater()

    before = session.read_bytes()
    monkeypatch.setenv("XDART_SESSION_FRESH", "1")
    fresh = build_rsm_page(empty_host_services(_Status()), None)
    try:
        assert fresh.widget._value("grid") == "Quick 40×40×40" and fresh.widget._value("project_root") == ""
        fresh.widget.set_field("offset", "0.25")
    finally:
        assert fresh.close().status is PageCleanup.CLEAN
        fresh.widget.deleteLater()
        qapp.processEvents()
    assert session.read_bytes() == before


def test_settings_that_cannot_apply_roll_back_and_a_broken_last_session_opens_the_defaults(
        qapp, tmp_path, session, monkeypatch):
    """A document that fails partway is undone, the inputs left as they were, and refused with the reason; the last
    session's document refused at start-up leaves the page's defaults and says so."""
    page = _page(qapp, owner=_Owner())
    try:
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path)
        document = page.settings()
        document["fields"]["grid"] = "Full 200×200×200"
        document["calibration"] = {"fields": {"rms": ".02"}, "choices": {}}
        before = page.settings()
        calls = []
        set_calibration = page._set_calibration_settings

        def fails_once(values):  # the window gone when the calibration applies, after every field has
            calls.append(values)
            if len(calls) == 1:
                raise RuntimeError("the calibration window is gone")
            set_calibration(values)
        monkeypatch.setattr(page, "_set_calibration_settings", fails_once)
        broken = tmp_path / "broken.json"
        broken.write_text(json.dumps(document))
        page._settings_path_chooser = lambda action, start: str(broken)
        reason = "these RSM settings cannot be applied here (RuntimeError: the calibration window is gone)"
        assert page.load() == ActionRefused(reason)
        assert calls == [document["calibration"], before["calibration"]]  # applied, then rolled back
        assert page.settings() == before and page._value("grid") == "Quick 40×40×40"
        assert page.run_bar.readinessLabel.full_text() == f"Could not load the RSM settings: {reason}"
    finally:
        _close(page, qapp)

    document["fields"]["coordinates"] = "Q (lab)"
    session.write_text(json.dumps({SESSION_KEY: document}))
    restored = build_rsm_page(empty_host_services(_Status()), None).widget
    default = _page(qapp, owner=_Owner())
    try:
        assert restored.settings() == default.settings()
        assert restored.run_bar.readinessLabel.full_text() == "Could not restore the last RSM settings"
    finally:
        _close(restored, qapp)
        _close(default, qapp)


def test_calibrate_opens_on_the_saved_window_settings_and_keeps_its_own_calibration_source(qapp, tmp_path):
    """As Stitch's (test_stitch_settings): the calibration window's settings travel with the page's, a window
    opened on saved ones keeps their calibration source (the calibrant scan is usually another scan), one saved
    without a source takes the first selected scan, and a Project change keeps the window's inputs."""
    page, restored, sourceless = (_page(qapp, owner=_Owner()) for _ in range(3))
    try:
        _synthetic_source(tmp_path)
        for each in (page, restored, sourceless):
            _with_source(each, tmp_path)
        page._calibrate()  # a new window takes the first selected scan
        dialog = page._calibration_dialog
        processing = dialog._text("source")
        assert processing == str(tmp_path / "source" / "RSM_synth")
        calibrant = str(tmp_path / "powder.spec")
        dialog._set_field_text("source", calibrant)
        dialog._set_field_text("rms", ".02")
        saved = page.settings()["calibration"]
        assert saved == dialog.settings() and saved["fields"]["source"] == calibrant
        assert saved["fields"]["rms"] == ".02"
        dialog.hide()

        restored.apply_settings(page.settings())
        restored._calibrate()
        again = restored._calibration_dialog
        assert (again._text("source"), again._text("rms")) == (calibrant, ".02")
        again.hide()
        unnamed = page.settings()
        unnamed["calibration"]["fields"]["source"] = ""  # saved without a calibration source
        sourceless.apply_settings(unnamed)
        sourceless._calibrate()
        assert (sourceless._calibration_dialog._text("source"), sourceless._calibration_dialog._text("rms")) == (
            processing, ".02")
        sourceless._calibration_dialog.hide()

        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        page.set_field("project_root", str(elsewhere))
        page._calibrate()  # the window a Project change closes leaves its inputs to the next one
        assert page._calibration_dialog is not dialog
        assert (page._calibration_dialog._text("source"), page._calibration_dialog._text("rms")) == (calibrant, ".02")
        page._calibration_dialog.hide()
    finally:
        for each in (page, restored, sourceless):
            _close(each, qapp)


def test_a_document_saved_without_a_calibration_window_closes_the_one_opened_since(qapp, tmp_path):
    """Codex's combined stage-2b review F3 (its same-page reproduction): a document saved before Calibrate was
    opened holds ``calibration: null``. Loaded after the window was opened and edited (RMS .02) and hidden, it
    restores that: the next document has no calibration window, and Calibrate opens a new window on the
    defaults. The window is let go as a Project change lets it go (hidden; the page still shuts it down). A
    document without the section keeps the window and its edits (RSM's omission rule)."""
    page = _page(qapp, owner=_Owner())
    try:
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path)
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
        assert page._calibration_dialog is None and not dialog.isVisible() and dialog in page._authoring_dialogs
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


def test_the_scan43_settings_file_makes_the_retired_presets_form_on_any_project(qapp, tmp_path, session):
    """Design §2 item 5: the "Scan 43" preset button went with the classic dialog (G10), and its inputs ship as a
    settings file (``fixtures/rsm_scan43_settings.json``, made by the page). It names no Project, calibration
    window or Append/Replace, and its Output follows the Source: loaded onto a page whose Project holds the
    preset's scan and the canonical calibration, it makes the dialog's exact scan-43 form, as the page row does
    (test_rsm_page: Int's Auto threshold, Replace, the Calibration row's geometry, Max chunk MiB Auto)."""
    from xrd_tools.integrate.stitch_conditioning import StitchConditioning
    document = RSM_SETTINGS.load(_SCAN43.read_text(encoding="utf-8"))
    assert "project_root" not in document["fields"] and document["fields"]["output_path"] == ""
    assert not {"calibration", "write_mode"} & set(document)
    page = _page(qapp, owner=_Owner())
    try:
        _sto_align(tmp_path)  # the preset's scan on disk, and the canonical calibration
        _canonical_partner(tmp_path)
        page.set_field("project_root", str(tmp_path))
        page._settings_path_chooser = lambda action, start: str(_SCAN43)
        assert page.load() == ActionCompleted("rsm-settings-loaded")
        assert page._readiness[1] and page._output_suggested, page._readiness
        page.resolve_source()
        form = page.build_form()
        assert Path(form.output_path).name == "STO_align-43_rsm.nexus"  # the scan's own slot
        dialog = _scan43_dialog_form(tmp_path, Path(form.output_path))
        assert form.fingerprint == replace(
            dialog, conditioning=replace(dialog.conditioning, high_threshold=None),
            corrections=replace(dialog.corrections, threshold=StitchConditioning("auto")),
            overwrite=AnalysisArtifactOverwrite.REPLACE, geometry_asset=form.geometry_asset, max_chunk_bytes=None,
            output_family=form.output_family).fingerprint
    finally:
        _close(page, qapp)


def test_the_page_built_by_its_builder_is_unchanged_by_an_empty_session(qapp, session):
    """No last session: the builder's page is the plain page (restoring nothing changes nothing)."""
    built = build_rsm_page(empty_host_services(_Status()), None).widget
    plain = RSMPage(empty_host_services(_Status()))
    try:
        assert built.settings() == plain.settings() and not session.exists()
    finally:
        _close(built, qapp)
        _close(plain, qapp)
