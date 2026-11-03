"""NEXT-128-PEAK-FIT frozen GUI rows on Int's real page: T4 (displayed capture), T6 (the dialog), T8 (source changes
and lifecycle), T9 (settings) and T10 (one shared row and help) (Claude, 2026-10-31).

Frozen before the product change (DESIGN-PEAK-FIT-128.md, Codex REVIEW-1 R1-R6, and the PLAN's contract revisions
1-10). Real pages, the page's analysis slot and its worker, real fits, the shared rows and PageSettings. Only the
operating system's file choosers are answered (as W10 answers its chooser). T8's stale row and T9's busy row hold the
real template read on an Event, then release it (the browse-click rows' technique); nothing replaces the owner. New
APIs are reached inside test bodies, so a missing one fails its own row.
"""
from contextlib import contextmanager
from importlib import import_module
import json
from pathlib import Path
import threading

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets

from tests.xdart.scattering.test_fit_tools_preservation import _close_page, _page, _peaks, _settle, _show, _trace

PEAK_TOPICS = ("peak-fit.model", "peak-fit.background", "peak-fit.peaks", "peak-fit.range", "peak-fit.weighting")


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def isolated_library(tmp_path, monkeypatch):
    monkeypatch.setenv("XDART_PHASE_LIBRARY", str(tmp_path / "phase-library"))


def _owner():
    return import_module("xrd_tools.analysis.background_template")


@contextmanager
def _peak_page(qapp, monkeypatch, *, kind="peak"):
    page = _page()
    try:
        trace, anchor = _trace(*_peaks())
        yield page, _show(page, monkeypatch, kind, trace, anchor)
    finally:
        _close_page(page, qapp)


def _press(qapp, button):
    QtTest.QTest.mouseClick(button, QtCore.Qt.MouseButton.LeftButton)
    qapp.processEvents()


def _template_file(folder, name="iq_substrate.xye", scale=1.0):
    """The peaks' sloped background (``_peaks``: 50 + 10 x), stored in nm^-1."""
    x = np.linspace(1.0, 4.0, 601)
    path = Path(folder) / name
    np.savetxt(path, np.column_stack([x * 10.0, scale * (50.0 + 10.0 * x)]))
    return path


def _choose(qapp, row, path, unit="q_nm^-1"):
    """Choose a template file as the user does (NEXT-128-WINDOW-POLISH): a template background first, since the row
    waits for one (PF9), then Choose file…, the unit answered in the popup that asks for it (PF10)."""
    host = row.window()  # Phase Analysis's template background is its fitted one (its bkg_combo is the pre-fit one)
    combo = host.fit_background_row.editor if hasattr(host, "fit_background_row") else host.bkg_combo
    if not row.choose_btn.isEnabled():
        combo.setCurrentIndex(combo.findData("template"))
    row._file_chooser = lambda: str(path)
    row._ask_unit = lambda _path, _hint: unit
    _press(qapp, row.choose_btn)
    qapp.processEvents()


def _fit(qapp, page, dialog):
    dialog.fit_btn.click()
    update = _settle(page)
    assert page._consume_analysis_update(update)
    qapp.processEvents()
    return update


def _export_buttons(dialog):
    buttons = tuple(dialog.export_actions.buttons.values())
    assert buttons
    return buttons


def _hold_read(monkeypatch, path):
    """Hold the owner's real read of ``path`` until released (every other read passes)."""
    owner = _owner()
    entered, release = threading.Event(), threading.Event()
    real = owner.read_xye

    def held(target, *args, **kwargs):
        if Path(target) == Path(path) and not release.is_set():
            entered.set()
            assert release.wait(30)
        return real(target, *args, **kwargs)

    monkeypatch.setattr(owner, "read_xye", held)
    return entered, release


# ---- T10 one shared row, one settings owner, help on every option -----------------------------------------------

def test_t10_both_tools_mount_the_one_template_row_and_the_settings_owner(qapp, monkeypatch):
    from xdart.gui.analysis.background_template_row import BackgroundTemplateRow
    from xdart.gui.analysis.peak_fit_dialog import PEAK_FIT_SETTINGS
    from xdart.gui.tools.page_settings import PageSettings
    from xdart.gui.widgets.controls_panel import ActionButton
    page = _page()
    try:
        trace, anchor = _trace(*_peaks())
        peak = _show(page, monkeypatch, "peak", trace, anchor)
        phase = _show(page, monkeypatch, "phase", trace, anchor)
        assert type(peak.background_template_row) is BackgroundTemplateRow
        assert type(phase.background_template_row) is BackgroundTemplateRow
        assert isinstance(peak.settings_io, PageSettings) and isinstance(phase.settings_io, PageSettings)
        assert (PEAK_FIT_SETTINGS.kind, PEAK_FIT_SETTINGS.version) == ("xdart-peak-fit-settings", 1)
        assert "background_template" in PEAK_FIT_SETTINGS.sections
        assert "background_template" in phase.settings_io.kind.sections
        assert (phase.substrate_capture_btn, phase.substrate_clear_btn) == (
            phase.background_template_row.capture_btn, phase.background_template_row.clear_btn)
        for row in (peak.background_template_row, phase.background_template_row):
            for button in (row.capture_btn, row.choose_btn, row.clear_btn):
                assert isinstance(button, ActionButton) and button.toolTip(), button.text()
            assert row.status.toolTip()
            for control in (row.unit_row, row.frame_row, row.mode_row):
                assert control.toolTip() or control.editor.toolTip(), control.label.text()
        for button in (peak.save_settings_btn, peak.load_settings_btn):
            assert button.toolTip(), button.text()
        models = [peak.model_combo.itemText(index) for index in range(peak.model_combo.count())]
        backgrounds = [peak.bkg_combo.itemText(index) for index in range(peak.bkg_combo.count())]
        assert models[4:] == ["Pearson VII", "Split pseudo-Voigt"]
        assert backgrounds[4:] == ["Template", "Template + linear", "Template + Chebyshev (3)"]
    finally:
        _close_page(page, qapp)


def test_t10_every_peak_fitter_option_opens_its_mapped_help(qapp):
    from xdart.gui.analysis.peak_fit_dialog import PeakFitDialog
    from xdart.gui.widgets.context_help import owner_for
    from xdart.gui.widgets.controls_panel import ActionButton, FormRow, RangeRow
    from xrd_tools.help_topics import get_topic
    dialog = PeakFitDialog(vnext_submit=lambda *_args: None)
    try:
        dialog.show()
        qapp.processEvents()
        expected = {"model_row": "peak-fit.model", "bkg_row": "peak-fit.background",
                    "count_row": "peak-fit.peaks", "range_row": "peak-fit.range"}
        for name, topic in expected.items():
            row = getattr(dialog, name)
            assert row.property("helpTopic") == topic, (name, row.property("helpTopic"))
            assert isinstance(row.help_button, ActionButton) and row.help_button.toolTip(), name
        # NEXT-128-WINDOW-POLISH PF2: Auto sits in the Peaks row, whose "?" it shares (F1 on it opens the same topic)
        assert dialog.auto_row.property("helpTopic") == "peak-fit.peaks" and dialog.count_row.isAncestorOf(
            dialog.auto_row) and not hasattr(dialog.auto_row, "help_button")
        assert dialog.background_template_row.property("helpTopic") == "peak-fit.background"
        # PF6: the advanced values are the same rows, in the options column's card rather than the Advanced dialog
        advanced = [widget for widget in dialog.advanced_card.findChildren(QtWidgets.QWidget)
                    if isinstance(widget, (FormRow, RangeRow))]
        assert len(advanced) == 7
        topics = {row.label.text(): row.property("helpTopic") for row in advanced}
        assert topics == {"Peak centers": "peak-fit.peaks", "σ initial": "peak-fit.peaks",
                          "σ min / max": "peak-fit.peaks", "Center ± delta": "peak-fit.peaks",
                          "PV fraction": "peak-fit.model", "Max iterations": "peak-fit.peaks",
                          "Use uncertainties": "peak-fit.weighting"}
        assert all(get_topic(topic) is not None for topic in PEAK_TOPICS)
        _press(qapp, dialog.model_row.help_button)
        window = owner_for(dialog).window
        assert window is not None and window.topic_id == "peak-fit.model"
    finally:
        dialog.close()


# ---- T6 the dialog fits the added models and a file template ----------------------------------------------------

def test_t6_the_dialog_fits_pearson_vii_and_binds_a_file_template_for_export(qapp, monkeypatch, tmp_path):
    from xrd_tools.analysis.display_fit_operations import DisplayedPeakFitResult
    with _peak_page(qapp, monkeypatch) as (page, dialog):
        dialog.model_combo.setCurrentText("Pearson VII")
        _fit(qapp, page, dialog)
        result = page._peak_result
        assert type(result) is DisplayedPeakFitResult and result.code == "OK", page._notice_text
        assert dialog.vnext_fit_values()["model"] == "pearson7"
        row = dialog.background_template_row
        _choose(qapp, row, _template_file(tmp_path))
        dialog.bkg_combo.setCurrentText("Template + linear")
        qapp.processEvents()
        assert row.source is not None and row.identity is None and row.template is None
        assert "BACKGROUND_TEMPLATE_NOTEBOOK_UNAVAILABLE" in row.export_reason()
        assert not any(button.isEnabled() for button in _export_buttons(dialog))
        _fit(qapp, page, dialog)
        result = page._peak_result
        assert result.code == "OK" and len(result.background_template_identity) == 64, page._notice_text
        assert "bg_A" in result.parameter_names and "bgx_slope" in result.parameter_names
        assert row.identity == result.background_template_identity and row.export_reason() == ""
        assert all(button.isEnabled() for button in _export_buttons(dialog))
        values = dialog.vnext_fit_values()
        assert values["background"] == "template+linear"
        assert values["background_template_source"] == row.source.to_pairs()
        assert values["background_template_identity"] == row.identity


# ---- T8 source changes and lifecycle ----------------------------------------------------------------------------

def test_t8_a_stale_completion_never_binds_another_source(qapp, monkeypatch, tmp_path):
    first, second = _template_file(tmp_path, "iq_first.xye"), _template_file(tmp_path, "iq_second.xye", scale=2.0)
    with _peak_page(qapp, monkeypatch) as (page, dialog):
        row = dialog.background_template_row
        _choose(qapp, row, first)
        dialog.bkg_combo.setCurrentText("Template")
        entered, release = _hold_read(monkeypatch, first)
        try:
            dialog.fit_btn.click()
            assert entered.wait(30), "the fit reached the template read on the worker"
            _choose(qapp, row, second)
        finally:
            release.set()
        update = _settle(page)
        page._consume_analysis_update(update)
        qapp.processEvents()
        assert Path(row.source.path) == second and row.identity is None
        _fit(qapp, page, dialog)
        bound = page._peak_result
        assert bound.code == "OK" and row.identity == bound.background_template_identity
        assert row.identity == _owner().prepare_background_template(row.source).identity


def test_t8_closing_the_dialog_mid_fit_binds_nothing_and_a_reopened_one_starts_empty(qapp, monkeypatch, tmp_path):
    path = _template_file(tmp_path)
    with _peak_page(qapp, monkeypatch) as (page, dialog):
        _choose(qapp, dialog.background_template_row, path)
        dialog.bkg_combo.setCurrentText("Template")
        entered, release = _hold_read(monkeypatch, path)
        try:
            dialog.fit_btn.click()
            assert entered.wait(30)
            dialog.close()
            qapp.processEvents()
        finally:
            release.set()
        page._consume_analysis_update(_settle(page))
        qapp.processEvents()
        page._open_analysis_mount("peak")
        reopened = page._peak_dialog
        assert reopened.background_template_row.identity is None


def test_t8_a_changed_file_refuses_the_fit_until_it_is_chosen_again(qapp, monkeypatch, tmp_path):
    path = _template_file(tmp_path)
    with _peak_page(qapp, monkeypatch) as (page, dialog):
        row = dialog.background_template_row
        _choose(qapp, row, path)
        dialog.bkg_combo.setCurrentText("Template")
        _fit(qapp, page, dialog)
        first = row.identity
        assert first is not None and len(first) == 64
        before = page._current_analysis_request("peak", "peak")
        _template_file(tmp_path, scale=1.5)
        _fit(qapp, page, dialog)
        assert "BACKGROUND_TEMPLATE_CHANGED" in page._notice_text
        assert "BACKGROUND_TEMPLATE_CHANGED" in row.reason and row.export_reason()
        assert not any(button.isEnabled() for button in _export_buttons(dialog))
        _choose(qapp, row, path)
        assert row.identity is None and row.reason == ""
        assert page._current_analysis_request("peak", "peak") != before
        _fit(qapp, page, dialog)
        assert row.identity not in (None, first) and row.export_reason() == ""


# ---- T9 settings, through the shared PageSettings owner ---------------------------------------------------------

def test_t9_the_peak_fitter_saves_and_loads_its_source_and_fit_values(qapp, monkeypatch, tmp_path):
    path, document_path = _template_file(tmp_path), tmp_path / "peak-fit.json"
    with _peak_page(qapp, monkeypatch) as (page, dialog):
        dialog._settings_path_chooser = lambda action, start: str(document_path)
        row = dialog.background_template_row
        dialog.model_combo.setCurrentText("Split pseudo-Voigt")
        dialog.bkg_combo.setCurrentText("Template + Chebyshev (3)")
        dialog.auto_check.setChecked(False)
        dialog.npeaks_spin.setValue(3)
        dialog.adv_fraction.setText("0.3")
        dialog.adv_maxfev.setValue(400)
        _choose(qapp, row, path)
        _fit(qapp, page, dialog)
        saved, identity = dialog.vnext_fit_values(), row.identity
        assert identity is not None
        assert dialog.settings_io.save().__class__.__name__ == "ActionCompleted"
        document = json.loads(document_path.read_text(encoding="utf-8"))
        assert (document["kind"], document["version"]) == ("xdart-peak-fit-settings", 1)
        assert document["background_template"] == {"source": dict(row.source.to_pairs()), "identity": identity}
        assert all(not isinstance(value, str) or not value.startswith("(") for value in document["fields"].values())
        _press(qapp, row.clear_btn)  # PF9: Clear waits for a template background, so before Linear
        dialog.model_combo.setCurrentIndex(0)
        dialog.bkg_combo.setCurrentIndex(0)
        dialog.auto_check.setChecked(True)
        assert row.source is None
        assert dialog.settings_io.load().__class__.__name__ == "ActionCompleted"
        qapp.processEvents()
        assert dialog.vnext_fit_values() == saved
        assert row.source == _owner().BackgroundTemplateSource.from_pairs(document["background_template"]["source"])
        assert row.identity == identity and row.template is None
        before = dialog.vnext_fit_values()
        for text in ("{not json", json.dumps(dict(document, version=99)),
                     json.dumps(dict(document, fields=dict(document["fields"], model="moffat"))),
                     json.dumps(dict(document, background_template={"source": {"kind": "csv"}, "identity": None}))):
            document_path.write_text(text, encoding="utf-8")
            assert dialog.settings_io.load().__class__.__name__ == "ActionRefused", text[:30]
            assert dialog.vnext_fit_values() == before and row.identity == identity
        document_path.write_text(json.dumps(document), encoding="utf-8")
        entered, release = _hold_read(monkeypatch, path)
        try:
            _template_file(tmp_path, scale=1.0)  # the same content rewritten: new file facts, so the fit reads it
            dialog.fit_btn.click()
            assert entered.wait(30), "the fit reached the template read on the worker"
            assert dialog.settings_io.load().__class__.__name__ == "ActionRefused"  # busy: fitting
        finally:
            release.set()
        page._consume_analysis_update(_settle(page))
        path.unlink()
        assert dialog.settings_io.load().__class__.__name__ == "ActionCompleted"
        assert row.source is not None and "missing" in row.reason.casefold() and row.export_reason()
    with _peak_page(qapp, monkeypatch) as (_page2, fresh):
        assert fresh.background_template_row.source is None  # no automatic restore of the last document
        assert fresh.vnext_fit_values()["model"] == "pseudovoigt"


def test_t9_phase_analysis_saves_its_source_and_an_old_document_clears_it(qapp, monkeypatch, tmp_path):
    path, document_path = _template_file(tmp_path), tmp_path / "phase-analysis.json"
    with _peak_page(qapp, monkeypatch, kind="phase") as (_page_, dialog):
        dialog._settings_path_chooser = lambda action, start: str(document_path)
        row = dialog.background_template_row
        _choose(qapp, row, path)
        dialog.fit_background_row.editor.setCurrentText("Substrate")
        qapp.processEvents()
        assert dialog.settings_io.save().__class__.__name__ == "ActionCompleted"
        document = json.loads(document_path.read_text(encoding="utf-8"))
        assert (document["kind"], document["version"]) == ("xdart-phase-analysis-settings", 1)
        assert document["background_template"] == {"source": dict(row.source.to_pairs()), "identity": None}
        source = row.source
        _press(qapp, row.clear_btn)
        assert dialog.settings_io.load().__class__.__name__ == "ActionCompleted"
        assert row.source == source and row.identity is None
        old = {key: value for key, value in document.items() if key != "background_template"}
        document_path.write_text(json.dumps(old), encoding="utf-8")
        assert dialog.settings_io.load().__class__.__name__ == "ActionCompleted"
        assert row.source is None and row.template is None
        assert "substrate" in dialog.status.text().casefold()


# ---- T4 the displayed trace, named by its own file --------------------------------------------------------------

def test_t4_a_capture_its_file_cannot_reproduce_stays_usable_and_unnamed(qapp, monkeypatch, tmp_path):
    with _peak_page(qapp, monkeypatch) as (page, dialog):
        row = dialog.background_template_row
        dialog.bkg_combo.setCurrentText("Template")  # PF9: the row waits for a template background
        _press(qapp, row.capture_btn)  # the trace's receipt names /tmp/a.nxs, frame 3, which does not exist
        assert row.template is not None and row.source is not None and row.source.kind == "nexus"
        _fit(qapp, page, dialog)
        result = page._peak_result
        assert result.code == "OK" and result.background_template_identity == ""
        assert any("BACKGROUND_TEMPLATE_UNREPRODUCIBLE" in line for line in result.diagnostics)
        assert row.source is None and row.template is not None and "UNREPRODUCIBLE" in row.reason
        assert "does not hold this trace" in row.status.text()  # NEXT-128-WINDOW-POLISH PF10: the short status
        assert dialog.vnext_fit_values()["background_template"] is row.template
        assert "BACKGROUND_TEMPLATE_NOTEBOOK_UNAVAILABLE" in row.export_reason()
        assert not any(button.isEnabled() for button in _export_buttons(dialog))
        assert row.section() is None


def test_t4_phase_names_the_run_trace_by_its_file_after_the_first_fit(qapp, tmp_path):
    from xrd_tools.analysis.display_fit_operations import DisplayedPhaseFitPlan, run_displayed_phase_fit
    from xrd_tools.analysis.phase_references import PhaseLibrary
    from tests.xdart.scattering.test_phase_analysis_window import _cif, _open
    from tests.xdart.scattering.test_predicted_ring_overlays import run_page
    one = PhaseLibrary().add_cif(_cif(tmp_path, "one", 4.1))
    with run_page(qapp, tmp_path / "int", energies=(16, 16), mode="Int 1D") as (page, _view):
        dialog = _open(qapp, page)
        dialog.phase_source.add_library_entry(one.id)
        qapp.processEvents()
        row = dialog.background_template_row
        dialog.fit_background_row.editor.setCurrentText("Substrate")  # PF9: the row waits for a template background
        qapp.processEvents()
        _press(qapp, row.capture_btn)
        captured = row.template
        assert captured is not None and row.identity is None
        assert row.source == _owner().BackgroundTemplateSource(
            "nexus", str(captured.receipt.artifact), frame=int(captured.receipt.frame_label), mode="default")
        assert "BACKGROUND_TEMPLATE_NOTEBOOK_UNAVAILABLE" in row.export_reason()
        # declared repair (as W3/W4): the run's own trace has eight points, under the phase fit's ten-point floor
        trace, _anchor = _trace(*_peaks())
        dialog.set_vnext_trace(trace)
        values = dict(dialog.vnext_fit_values(), max_nfev=5)
        assert values["background_template"] is captured
        result = run_displayed_phase_fit(DisplayedPhaseFitPlan(trace, wavelength_angstrom=1.0, **values))
        assert result.code == "OK" and len(result.background_template_identity) == 64, result.diagnostics
        dialog.set_vnext_result(result)
        qapp.processEvents()
        assert row.template is None and row.identity == result.background_template_identity
        assert row.export_reason() == "" and all(button.isEnabled() for button in _export_buttons(dialog))
        assert row.section() == {"source": dict(row.source.to_pairs()), "identity": row.identity}


def test_t4_the_peak_fitter_names_the_run_trace_by_its_file_after_the_first_fit(qapp, tmp_path):
    from tests.xdart.scattering.test_predicted_ring_overlays import run_page
    with run_page(qapp, tmp_path / "int", energies=(16, 16), mode="Int 1D") as (page, _view):
        button = page._shell.tools.findChild(QtWidgets.QPushButton, "e3Tool_peak_fitting")
        assert button is not None and button.isEnabled()
        _press(qapp, button)
        dialog = page._peak_dialog
        row = dialog.background_template_row
        dialog.bkg_combo.setCurrentText("Template")  # PF9: the row waits for a template background
        _press(qapp, row.capture_btn)
        assert row.template is not None and row.source.kind == "nexus" and row.source.mode == "default"
        dialog.auto_check.setChecked(False)
        dialog.npeaks_spin.setValue(1)
        qapp.processEvents()
        _fit(qapp, page, dialog)
        result = page._peak_result
        assert result is not None and result.code == "OK", page._notice_text
        assert row.template is None and row.identity == result.background_template_identity
        assert row.export_reason() == ""
