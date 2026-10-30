"""PHASE-ANALYSIS-128 frozen GUI oracle W1-W10 on Int's real page (Claude, 2026-10-29).

One Phase Analysis window replaces Phase Fitting and Reference Phases: it opens through Int's real Tools command,
its list is the view's reference-phase plot options, and every notebook option has its row, tooltip and "?" help.
Real pages, Runs, workers, library and fits; only the operating system's file chooser is answered, as the existing
phase rows do. New APIs are reached inside test bodies, so a missing one fails its own row.
"""
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets

from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind

LAB6 = "calibrant:LaB6"
TOPICS = {"phases": "phase-analysis.phases", "shape": "phase-analysis.peak-shape",
          "background": "phase-analysis.background", "constraints": "phase-analysis.constraints",
          "batch": "phase-analysis.batch"}


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def isolated_library(tmp_path, monkeypatch):
    monkeypatch.setenv("XDART_PHASE_LIBRARY", str(tmp_path / "phase-library"))


@contextmanager
def empty_page(qapp):
    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.session.run_configuration import RunIntent
    from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
    from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
    from xdart.gui.tabs.scattering.page import ScatteringWorkspace
    page = ScatteringWorkspace(intents=RunIntentStore(RunIntent()), lifecycle=ScatteringCoordinator(),
                               sources=FilesystemSourceAdapter())
    page.show()
    qapp.processEvents()
    try:
        yield page
    finally:
        page.close_workspace()
        page.deleteLater()
        qapp.processEvents()


def _press(qapp, button):
    QtTest.QTest.mouseClick(button, QtCore.Qt.MouseButton.LeftButton)
    qapp.processEvents()


def _open(qapp, page):
    """Phase Analysis through Int's real Tools button."""
    button = page._shell.tools.findChild(QtWidgets.QPushButton, "e3Tool_phase_analysis")
    assert button is not None and button.isEnabled()
    _press(qapp, button)
    dialog = page._phase_dialog
    assert dialog is not None and dialog.isVisible()
    return dialog


def _options(page):
    return page._preferences.plot_options if hasattr(page, "_preferences") else page._view.plot_options


def _settle(qapp, view, predicate=lambda: True):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        qapp.processEvents()
        if not view.phase_worker.busy and predicate():
            qapp.processEvents()
            return
        time.sleep(.005)
    raise AssertionError("the shared phase worker did not settle")


def _added(qapp, source, predicate):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        qapp.processEvents()
        if not source.add_worker.busy and predicate():
            return
        time.sleep(.005)
    raise AssertionError("Add CIF did not finish")


def _marker(view, identifier):
    items = [item for item in view.curve.getPlotItem().items if item.data(0) == "phase:" + identifier]
    assert len(items) <= 1
    return items[0] if items else None


def _row(source, identifier):
    index = next(i for i, (entry, _shown) in enumerate(source.listed_entries()) if entry.id == identifier)
    return source.browser.scans.item(index)


def _check(qapp, source, identifier, checked):
    _row(source, identifier).setCheckState(QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked)
    qapp.processEvents()


def _ids(values):
    return tuple(entry.id for entry in values.get("phase_entries", ()))


def _cif(root, name, a):
    from tests.xdart.scattering.test_phase_reference_overlays import _cif as write
    return write(Path(root), name, a)


# ---- W1 one launcher, one window ---------------------------------------------------------------------------------

def test_w1_int_tools_show_one_phase_analysis_launcher_that_opens_the_bounded_window(qapp):
    from xdart.gui.tabs.scattering.analysis_mount import MOUNT_TARGETS
    from xdart.gui.tabs.scattering.tools_view import PHASE_ANALYSIS_TOOLTIP, ToolsView
    values = [entry[1] for entry in ToolsView._TOOLS]
    assert ("≈ Phase Analysis", "phase_analysis", PHASE_ANALYSIS_TOOLTIP) in ToolsView._TOOLS
    assert "phase_fitting" not in values and "reference_phases" not in values
    assert MOUNT_TARGETS[(ShellCommandKind.LAUNCH_TOOL, "phase_analysis")] == "phase"
    assert (ShellCommandKind.LAUNCH_TOOL, "phase_fitting") not in MOUNT_TARGETS
    assert MOUNT_TARGETS[(ShellCommandKind.ANALYSIS_ACTION, "phase_fit")] == "phase"
    assert "fit" in PHASE_ANALYSIS_TOOLTIP.casefold() and "reference" in PHASE_ANALYSIS_TOOLTIP.casefold()
    with empty_page(qapp) as page:
        tools = page._shell.tools
        assert tools.findChild(QtWidgets.QPushButton, "e3Tool_phase_fitting") is None
        assert tools.findChild(QtWidgets.QPushButton, "e3Tool_reference_phases") is None
        assert tools.findChild(QtWidgets.QPushButton, "e3Tool_phase_analysis").toolTip() == PHASE_ANALYSIS_TOOLTIP
        dialog = _open(qapp, page)
        assert dialog.windowTitle() == "Phase Analysis" and dialog.toolTip()
        available = dialog.screen().availableGeometry()
        # the maintainer after the mockup (2026-10-29): "the window can be wider. 1400x1200 is also fine"
        assert dialog.width() == min(1400, int(.95 * available.width()))
        assert dialog.height() == min(1200, int(.95 * available.height()))
        view = page._shell.scientific
        assert view.phase_list is dialog.phase_binding
        assert not hasattr(view, "reference_phases") and not hasattr(view, "show_reference_phases")
        assert dialog.rings_note is dialog.phase_source.note


# ---- W2 one list: the window edits the view's reference phases ---------------------------------------------------

def test_w2_every_add_reaches_the_plot_options_and_the_view_and_a_reopened_window_lists_it(qapp, tmp_path,
                                                                                          monkeypatch):
    import xdart.gui.analysis.phase_source as phase_source
    from xrd_tools.analysis.phase_references import PhaseLibrary
    from tests.xdart.scattering.test_predicted_ring_overlays import run_page
    first = _cif(tmp_path, "phaseA", 4.1)
    kept = PhaseLibrary().add_cif(_cif(tmp_path, "phaseB", 4.25))  # already in the library
    with run_page(qapp, tmp_path / "int", energies=(16, 16), mode="Int 1D") as (page, view):
        dialog = _open(qapp, page)
        source = dialog.phase_source
        monkeypatch.setattr(phase_source, "browse", lambda *args, **kwargs: [str(first)])
        _press(qapp, source.add_cif_btn)
        _added(qapp, source, lambda: len(_options(page).phase_references) == 1)
        cif = source.listed_entries()[0][0]
        # a user's choice is an activation (FormRow ignores programmatic index changes): declared oracle repair
        source.calibrant_row.editor.setCurrentText("LaB6")
        source.calibrant_row.editor.textActivated.emit("LaB6")
        qapp.processEvents()
        library_index = source.library_row.editor.findData(kept.id)
        source.library_row.editor.setCurrentIndex(library_index)
        source.library_row.editor.textActivated.emit(source.library_row.editor.currentText())
        qapp.processEvents()
        _press(qapp, source.add_lattice_btn)
        for key, text in (("name", "cell"), ("a", "4.3"), ("space_group", "221")):
            source.lattice_rows[key].editor.setText(text)
        _press(qapp, source.add_lattice_commit)
        options = _options(page)
        listed = [identifier for identifier, _shown in options.phase_references]
        assert listed[:3] == [cif.id, LAB6, kept.id] and len(listed) == 4
        _settle(qapp, view)
        for identifier in (cif.id, kept.id, listed[3]):
            assert _marker(view, identifier) is not None, identifier
        source.global_switch.editor.click()
        qapp.processEvents()
        source.cutoff_editor(cif.id).setValue(2.)
        qapp.processEvents()
        assert dict(_options(page).phase_floors)[cif.id] == 2.
        _check(qapp, source, kept.id, False)
        flags = dict(_options(page).phase_references)
        dialog.close()
        qapp.processEvents()
        assert page._phase_dialog is None and view.phase_list is None
        again = _open(qapp, page)
        assert [entry.id for entry, _shown in again.phase_source.listed_entries()] == listed
        assert dict((entry.id, shown) for entry, shown in again.phase_source.listed_entries()) == flags
        assert again.phase_source.cutoffs.per_phase and not again.phase_source.cutoffs.global_on


# ---- W3/W4 what is fitted: included CIF and lattice rows, in list order; calibrants are references only --------

def test_w3_w4_the_fit_takes_the_included_non_calibrant_rows_and_receipts_follow_their_cif_ids(qapp, tmp_path):
    from xrd_tools.analysis.phase_references import PhaseLibrary
    from xrd_tools.analysis.display_fit_operations import DisplayedPhaseFitPlan, run_displayed_phase_fit
    from tests.xdart.scattering.test_predicted_ring_overlays import run_page
    library = PhaseLibrary()
    one, two = library.add_cif(_cif(tmp_path, "one", 4.1)), library.add_cif(_cif(tmp_path, "two", 4.25))
    cell = library.add_lattice("cell", a=4.3, space_group="Pm-3m")
    with run_page(qapp, tmp_path / "int", energies=(16, 16), mode="Int 1D") as (page, view):
        selection = ((one.id, True), (LAB6, True), (cell.id, True), (two.id, False))
        page._handle_shell_command(ShellCommand(ShellCommandKind.SET_PLOT_OPTION, selection,
                                                ("other", "phase_references")))
        qapp.processEvents()
        dialog = _open(qapp, page)  # a list restored before the window exists
        source = dialog.phase_source
        assert [entry.id for entry, _shown in source.listed_entries()] == [one.id, LAB6, cell.id, two.id]
        values = dialog.vnext_fit_values()
        assert _ids(values) == (one.id, cell.id) and values["phase_names"] == ("one", "cell")
        assert LAB6 not in _ids(values)
        assert "reference" in _row(source, LAB6).toolTip().casefold()
        _settle(qapp, view)
        assert _marker(view, two.id) is None  # unchecked: neither drawn nor fitted
        source.global_switch.editor.click()
        qapp.processEvents()
        source.cutoff_editor(one.id).setValue(1.)
        qapp.processEvents()
        assert dialog.vnext_fit_values()["phase_min_intensities"] == (1., 5.)
        # declared oracle repair: the run's own trace has eight points (npt=8), under the fit's ten-point floor
        from tests.xdart.scattering.test_fit_tools_preservation import _peaks, _trace
        trace, _anchor = _trace(*_peaks())
        dialog.set_vnext_trace(trace)
        wavelength = 1.0
        result = run_displayed_phase_fit(DisplayedPhaseFitPlan(trace, wavelength_angstrom=wavelength,
                                                               **dict(dialog.vnext_fit_values(), max_nfev=5)))
        assert result.code == "OK", (result.code, result.diagnostics)
        dialog.set_vnext_result(result)
        qapp.processEvents()
        assert result.cif_receipts[0].sha256 in _row(source, one.id).toolTip()
        assert all(result.cif_receipts[0].sha256 not in _row(source, other).toolTip() for other in (LAB6, cell.id))
        _check(qapp, source, one.id, False)
        _check(qapp, source, cell.id, False)
        assert _ids(dialog.vnext_fit_values()) == ()
        before = page._analysis_identity
        _press(qapp, dialog.fit_btn)
        assert page._analysis_identity is before, "a calibrant-only list never starts a fit"
        assert "calibrant" in dialog.status.text().casefold()


# ---- W5 attach, replacement and close ---------------------------------------------------------------------------

def test_w5_the_binding_follows_its_view_across_replacement_close_and_an_explicit_refresh(qapp, tmp_path):
    from xrd_tools.analysis.phase_references import PhaseLibrary
    from tests.xdart.scattering.test_predicted_ring_overlays import run_page
    library = PhaseLibrary()
    one = library.add_cif(_cif(tmp_path, "one", 4.1))
    with run_page(qapp, tmp_path / "int", energies=(16, 16), mode="Int 1D") as (page, view):
        dialog = _open(qapp, page)
        binding = dialog.phase_binding
        assert view.phase_list is binding
        seen = []
        view.commandRequested.connect(seen.append)
        dialog.phase_source.add_library_entry(one.id)
        qapp.processEvents()
        assert [c.path for c in seen if c.kind is ShellCommandKind.SET_PLOT_OPTION].count(
            ("other", "phase_references")) == 1, "one edit, one command"
        view.attach_phase_list(binding)  # idempotent for the same binding
        view.detach_phase_list(object())  # another object: no effect
        assert view.phase_list is binding
        options = _options(page)
        fresh = page._shell.replace_scientific_with_blank()
        qapp.processEvents()
        assert fresh.phase_list is binding and view.phase_list is None
        assert fresh.phase_list.projection.phase_references == options.phase_references
        fresh_seen, old_seen = [], []
        fresh.commandRequested.connect(fresh_seen.append)
        view.commandRequested.connect(old_seen.append)
        dialog.phase_source.set_selection(())
        qapp.processEvents()
        assert fresh_seen and not old_seen, "edits reach only the current view"
        missing_path = _cif(tmp_path, "late", 4.4)
        late_id = __import__("hashlib").sha256(missing_path.read_bytes()).hexdigest()
        page._handle_shell_command(ShellCommand(ShellCommandKind.SET_PLOT_OPTION, ((late_id, True),),
                                                ("other", "phase_references")))
        _settle(qapp, fresh)
        assert _marker(fresh, late_id) is None  # not in the library yet
        library.add_cif(missing_path)
        dialog.phase_source.refresh_library()  # the explicit refresh (open, library, Add)
        qapp.processEvents()
        _settle(qapp, fresh)
        assert _marker(fresh, late_id) is not None, "an explicit refresh recovers a previously missing id"
        dialog.close()
        qapp.processEvents()
        assert fresh.phase_list is None


# ---- W6 availability --------------------------------------------------------------------------------------------

def test_w6_the_2d_viewer_greys_the_list_while_fit_keeps_its_own_reason(qapp, tmp_path):
    from tests.xdart.scattering.test_phase_reference_overlays import _navigation
    from tests.xdart.scattering.test_predicted_ring_overlays import run_page
    with run_page(qapp, tmp_path / "int", energies=(16, 16), mode="Int 1D") as (page, view):
        dialog = _open(qapp, page)
        fit_tip = dialog.fit_btn.toolTip()
        state = page._last_scientific_projection
        view.reconcile(replace(state, processing_mode="2D Viewer"), _navigation(page), completed=0, total=0,
                       detail="")
        qapp.processEvents()
        available, reason = view.phase_availability
        assert not available and "2D Viewer" in reason
        assert not dialog.phase_source.isEnabled() and "2D Viewer" in dialog.phase_source.toolTip()
        assert dialog.fit_btn.toolTip() == fit_tip


# ---- W7/W8 rows, tooltips, "?" help and Int's held Batch ---------------------------------------------------------

ROWS = {
    "shape": ("profile_row", "width_model_row", "texture_row"),
    "background": ("bkg_row", "fit_background_row", "fit_background_degree_row", "amorphous_row",
                   "amorphous_center_row", "amorphous_sigma_row"),
    "constraints": ("lattice_row", "q_shift_row", "cross_phase_row", "fit_range_row"),
    "batch": ("sequential_row",),
}
ADVANCED = {"constraints": ("_maxfev_row",), "background": ("snip_width_row",),
            "shape": ("free_height_row", "width_limits_row")}


def test_w7_every_option_has_its_row_tooltip_and_section_help(qapp):
    from xdart.gui.widgets.context_help import owner_for
    from xdart.gui.widgets.controls_panel import ActionButton, SubsectionCard
    with empty_page(qapp) as page:
        dialog = _open(qapp, page)
        for section, names in ROWS.items():
            for name in names:
                row = getattr(dialog, name)
                assert row.toolTip() or row.editor.toolTip(), name
                assert row.property("helpTopic") == TOPICS[section], (name, row.property("helpTopic"))
                assert isinstance(row.help_button, ActionButton) and row.help_button.toolTip(), name
        for section, names in ADVANCED.items():
            for name in names:
                assert getattr(dialog, name).property("helpTopic") == TOPICS[section], name
        for index, row in enumerate(dialog.march_rows):
            assert row.property("helpTopic") == TOPICS["shape"] and row.help_button, index
        for section, card in (("phases", dialog.phases_card), ("shape", dialog.shape_card),
                              ("background", dialog.background_card), ("constraints", dialog.constraints_card),
                              ("batch", dialog.batch_card)):
            assert card.property("helpTopic") == TOPICS[section] and card.help_button.toolTip(), section
        for row in (dialog.phase_source.calibrant_row, dialog.phase_source.library_row,
                    dialog.phase_source.global_switch, dialog.phase_source.global_row):
            assert row.property("helpTopic") == TOPICS["phases"] and row.help_button
        for button in (dialog.substrate_capture_btn, dialog.substrate_clear_btn):
            assert isinstance(button, ActionButton) and button.toolTip()
        for button in (dialog.load_settings_btn, dialog.save_settings_btn, dialog.advanced_btn, dialog.refresh_btn,
                       dialog.fit_btn, dialog.batch_btn):
            assert button.toolTip(), button.text()
        dialog.q_shift_row.editor.setFocus()
        dialog.q_shift_row.set_number(.02)
        _press(qapp, dialog.fit_background_row.help_button)
        window = owner_for(dialog).window
        assert window is not None and window.topic_id == TOPICS["background"]
        assert dialog.q_shift_row.current_value() == .02
        groups = {card.title.text() for card in dialog.advanced.findChildren(SubsectionCard)}
        assert groups == {"Solver", "Weighting", "Background", "Texture", "Peak widths"}
        splitter = dialog.findChild(QtWidgets.QSplitter)
        assert splitter is not None and splitter.orientation() == QtCore.Qt.Orientation.Horizontal
        assert dialog.fit_view.isVisible() and dialog.trend.isVisible() and dialog.table.isVisible()


def test_w8_int_lifts_the_texture_hold_and_keeps_batch_and_sequential_held(qapp):
    from xrd_tools.help_topics import diagnostic_text
    from xdart.gui.tabs.scattering.analysis_mount import HELD_REASONS
    assert "phase_texture" not in HELD_REASONS
    with empty_page(qapp) as page:
        dialog = _open(qapp, page)
        assert dialog.texture_row.editor.isEnabled()
        assert diagnostic_text("P3_7_PHASE_TEXTURE_UNAVAILABLE") not in dialog.texture_row.toolTip()
        batch_reason = diagnostic_text("P3_7_BATCH_DISPLAY_PROJECTION_UNAVAILABLE")
        assert not dialog.batch_btn.isEnabled() and not dialog.sequential_row.editor.isEnabled()
        assert batch_reason in dialog.sequential_row.toolTip() or batch_reason in dialog.sequential_row.editor.toolTip()
        assert dialog.sequential_batch() is False


# ---- W9 the substrate, captured through Int's own fit input -----------------------------------------------------

def test_w9_int_captures_the_displayed_trace_as_the_substrate_and_export_refuses_it(qapp, tmp_path):
    from xrd_tools.analysis.display_fit_operations import DisplayedPhaseFitPlan, DisplayedTraceInput
    from xdart.gui.tabs.scattering.analysis_mount import analysis_request_facts
    from tests.xdart.scattering.test_predicted_ring_overlays import run_page
    with run_page(qapp, tmp_path / "int", energies=(16, 16), mode="Int 1D") as (page, view):
        dialog = _open(qapp, page)
        assert dialog.substrate_template is None
        _press(qapp, dialog.substrate_capture_btn)
        template = dialog.substrate_template
        assert isinstance(template, DisplayedTraceInput) and template.axis_unit in {"q_A^-1", "1/angstrom", "A^-1"}
        assert template.label in dialog.background_card.status.text() and dialog.background_card.status.toolTip()
        dialog.fit_background_row.editor.setCurrentText("Substrate")
        qapp.processEvents()
        values = dialog.vnext_fit_values()
        assert values["background_template"] is template and values["fit_background"] == "template"
        buttons = tuple(dialog.export_actions.buttons.values())
        assert buttons and not any(button.isEnabled() for button in buttons)
        assert all("PHASE_TEMPLATE_NOTEBOOK_UNAVAILABLE" in button.toolTip() for button in buttons)
        trace, _anchor = page._capture_analysis_trace("phase")
        other = replace(template, label=template.label + " B")
        one = DisplayedPhaseFitPlan(trace, wavelength_angstrom=1., **values)
        two = DisplayedPhaseFitPlan(trace, wavelength_angstrom=1., **dict(values, background_template=other))
        assert analysis_request_facts(one) != analysis_request_facts(two)
        _press(qapp, dialog.substrate_clear_btn)
        assert dialog.substrate_template is None and all(button.isEnabled() for button in buttons)
        assert "substrate" in dialog.fit_background_row.toolTip().casefold()


# ---- W10 saved settings through the shared PageSettings owner ---------------------------------------------------

def test_w10_save_and_load_round_trip_every_option_and_the_list_and_refuse_bad_documents(qapp, tmp_path):
    from xrd_tools.analysis.phase_references import PhaseLibrary
    library = PhaseLibrary()
    one = library.add_cif(_cif(tmp_path, "one", 4.1))
    path = tmp_path / "phase-analysis.json"
    with empty_page(qapp) as page:
        dialog = _open(qapp, page)
        dialog._settings_path_chooser = lambda action, start: str(path)
        dialog.phase_source.add_library_entry(one.id)
        dialog.phase_source.add_calibrant("LaB6")
        qapp.processEvents()
        edits = {"profile_row": "Pearson VII", "width_model_row": "Scherrer size and strain",
                 "texture_row": "March-Dollase", "fit_background_row": "Chebyshev", "amorphous_row": "Gaussian"}
        for name, text in edits.items():
            getattr(dialog, name).editor.setCurrentText(text)
        for name, value in (("fit_background_degree_row", 5), ("q_shift_row", .02), ("amorphous_center_row", 2.),
                            ("amorphous_sigma_row", .4), ("snip_width_row", 9), ("lattice_row", 3.)):
            getattr(dialog, name).set_number(value)
        dialog.cross_phase_row.editor.setChecked(True)
        for row, value in zip(dialog.march_rows, (1, 1, 0)):
            row.set_number(value)
        qapp.processEvents()
        saved = dialog.vnext_fit_values()
        assert dialog.settings_io.save().__class__.__name__ == "ActionCompleted"
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["kind"] == "xdart-phase-analysis-settings" and document["version"] == 1
        assert all(not isinstance(value, str) or not value.startswith("(") for value in document["fields"].values())
        for name in edits:
            getattr(dialog, name).editor.setCurrentIndex(0)
        dialog.phase_source.set_selection(())
        qapp.processEvents()
        assert dialog.settings_io.load().__class__.__name__ == "ActionCompleted"
        qapp.processEvents()
        restored = dialog.vnext_fit_values()
        assert {key: restored[key] for key in saved if key != "phase_entries"} == \
            {key: saved[key] for key in saved if key != "phase_entries"}
        assert [identifier for identifier, _shown in _options(page).phase_references] == [one.id, LAB6]
        before = dialog.vnext_fit_values()
        for text in ("{not json", json.dumps(dict(document, version=99)),
                     json.dumps(dict(document, fields=dict(document["fields"], q_shift="far")))):
            path.write_text(text, encoding="utf-8")
            outcome = dialog.settings_io.load()
            assert outcome.__class__.__name__ == "ActionRefused", text[:20]
            assert dialog.vnext_fit_values() == before
        missing = dict(document, phases=dict(document["phases"], rows=[{"id": "0" * 64, "included": True}]))
        path.write_text(json.dumps(missing), encoding="utf-8")
        assert dialog.settings_io.load().__class__.__name__ == "ActionCompleted"
        assert "0" * 8 in dialog.status.text() or "missing" in dialog.status.text().casefold()


def test_w11_return_in_a_field_commits_it_and_presses_no_button(qapp):
    """PR #6 run 41005408144 (row added after the freeze for that defect): Return in an option box reached the
    window, which pressed its default button, the first section's "?", so Help opened on every committed value; off
    macOS a phase row's minimum box did the same (test_lt_phase_rows p3)."""
    from xdart.gui.widgets.context_help import HelpWindow

    def help_shown():
        return [window for window in QtWidgets.QApplication.topLevelWidgets()
                if isinstance(window, HelpWindow) and window.isVisible()]
    with empty_page(qapp) as page:
        dialog = _open(qapp, page)
        dialog.activateWindow()
        qapp.processEvents()
        assert not help_shown()
        line = dialog.lattice_pct.lineEdit()
        _press(qapp, line)
        line.selectAll()
        QtTest.QTest.keyClicks(line, "7")
        QtTest.QTest.keyClick(line, QtCore.Qt.Key.Key_Return)
        qapp.processEvents()
        assert dialog.vnext_fit_values()["lattice_pct"] == pytest.approx(.07, abs=0), "Return commits the value"
        assert not help_shown() and dialog.isVisible(), "and presses no button"
        # A focused button still answers its own Return.
        button = next(button for button in dialog.findChildren(QtWidgets.QPushButton) if button.text() == "?")
        button.setFocus()
        QtTest.QTest.keyClick(button, QtCore.Qt.Key.Key_Return)
        qapp.processEvents()
        assert help_shown(), "a focused ? opens Help on Return"
