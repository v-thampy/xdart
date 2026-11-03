"""CALIBRANT-GROUP-128: the built-in calibrants get their own list in Phase Analysis (the maintainer, 2026-11-03).

His words: "I'm trying to think of an elegant way to deal with the calibrant references. They are not used in the
fitting and I don't know if they belong in the phase analysis window". His clickable answer was "Separate group in
Phase Analysis, now": a "Reference lines (not fitted)" list below the phases, in the same window.

Rows C1-C6 run on the real ``PhaseFitDialog``, its shared ``PhaseSourceWidget``, Int's ``BrowserView`` lists and the
``ReferencePhaseBinding``. Every check, row selection and Remove comes from a real QTest mouse event, using the
LT-SMALL-FIXES helpers. New APIs are reached inside the test bodies, so a missing one fails its own row. These rows
were frozen before the product change: red on 20c1eac6.
"""
from contextlib import contextmanager

import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from tests.xdart.scattering.test_lt_phase_rows import _click, _indicator, _label_point
from tests.xdart.scattering.test_phase_reference_overlays import _cif

LAB6, CEO2 = "calibrant:LaB6", "calibrant:CeO2"
CAPTION = "Reference lines (not fitted)"
CHECKED, UNCHECKED = QtCore.Qt.CheckState.Checked, QtCore.Qt.CheckState.Unchecked


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def isolated_library(tmp_path, monkeypatch):
    monkeypatch.setenv("XDART_PHASE_LIBRARY", str(tmp_path / "phase-library"))


@contextmanager
def _window(qapp, submitted=None):
    """The Phase Analysis window as a host builds it, shown, on this test's own phase library."""
    from xdart.gui.analysis.phase_fit_dialog import PhaseFitDialog
    from xrd_tools.analysis.phase_references import PhaseLibrary
    calls = [] if submitted is None else submitted
    dialog = PhaseFitDialog(vnext_submit=lambda action, values: calls.append((action, values)))
    dialog.phase_source.library = PhaseLibrary()  # the isolated library (test_fit_dialogs_on_owners' pattern)
    try:
        dialog.show()
        dialog.activateWindow()
        qapp.processEvents()
        yield dialog
    finally:
        dialog.close()
        qapp.processEvents()


def _texts(scans):
    return [scans.item(row).text() for row in range(scans.count())]


def _selected(scans):
    return [item.text() for item in scans.selectedItems()]


# ---- C1 the phase list lists the phases; the reference list lists the calibrants ----------------------------------

def test_c1_the_phase_list_lists_the_phases_and_the_reference_list_the_calibrants(qapp, tmp_path):
    from xdart.gui.tabs.scattering.browser_view import BrowserView, keyed_list_profile
    with _window(qapp) as dialog:
        source = dialog.phase_source
        reference = source.reference_browser
        (cif,) = source.add_cif_paths((str(_cif(tmp_path, "phaseA", 4.1)),))
        source.add_calibrant("LaB6")
        cell = source.add_lattice("cell", a=4.3, space_group=221)
        source.add_calibrant("CeO2")
        qapp.processEvents()
        # one selection across both lists, in the order the entries joined it
        assert [identifier for identifier, _shown in source.selection] == [cif.id, LAB6, cell.id, CEO2]
        assert [entry.id for entry, _shown in source.listed_entries()] == [cif.id, LAB6, cell.id, CEO2]
        # each list keeps that order for its own entries
        assert source.browser.listed_results() == (cif.id, cell.id)
        assert _texts(source.browser.scans) == ["phaseA", "cell"]
        assert reference.listed_results() == (LAB6, CEO2)
        assert _texts(reference.scans) == ["LaB6", "CeO2"]
        assert source.browser.scans_label.text() == "Phases" and reference.scans_label.text() == CAPTION
        # the same shared owner and profile as the phase list: checks, and no minimum-intensity column
        assert isinstance(reference, BrowserView) and reference is not source.browser
        assert reference.profile == keyed_list_profile(CAPTION, checkable=True)
        assert reference.profile.result_values is None
        assert source.cutoff_editor(LAB6) is None and source.cutoff_editor(cif.id) is not None
        assert "never fitted" in reference.scans.item(0).toolTip()
        assert dialog.phase_source.isAncestorOf(reference) and dialog.phases_card.isAncestorOf(reference)
        # the fit takes the phases only
        assert [entry.id for entry in dialog.vnext_fit_values()["phase_entries"]] == [cif.id, cell.id]


# ---- C2 a check in either list edits the one selection, and so the view's plot options -----------------------------

def test_c2_a_reference_list_check_edits_the_one_selection_and_the_plot_options(qapp, tmp_path):
    with _window(qapp) as dialog:
        source = dialog.phase_source
        references = source.reference_browser.scans
        (cif,) = source.add_cif_paths((str(_cif(tmp_path, "phaseA", 4.1)),))
        source.add_calibrant("LaB6")
        qapp.processEvents()
        edits = []
        dialog.phase_binding.optionEdited.connect(lambda path, value: edits.append((path, value)))

        # a real click on the calibrant's check box hides it: one include edit; the phase keeps its own
        _click(qapp, references.viewport(), _indicator(references, references.item(0)).center())
        assert source.selection == ((cif.id, True), (LAB6, False))
        assert edits == [(("other", "phase_references"), ((cif.id, True), (LAB6, False)))]
        assert references.item(0).checkState() == UNCHECKED

        # a second click shows it again
        edits.clear()
        _click(qapp, references.viewport(), _indicator(references, references.item(0)).center())
        assert source.selection == ((cif.id, True), (LAB6, True))
        assert edits == [(("other", "phase_references"), ((cif.id, True), (LAB6, True)))]
        assert references.item(0).checkState() == CHECKED

        # a check in the phase list leaves the calibrant shown: both lists edit the same selection
        edits.clear()
        phases = source.browser.scans
        _click(qapp, phases.viewport(), _indicator(phases, phases.item(0)).center())
        assert source.selection == ((cif.id, False), (LAB6, True))
        assert edits == [(("other", "phase_references"), ((cif.id, False), (LAB6, True)))]
        assert references.item(0).checkState() == CHECKED and phases.item(0).checkState() == UNCHECKED


# ---- C3 one Remove for both lists: it takes the selected row, and only one list holds a selection -------------------

def test_c3_remove_takes_the_selected_row_of_either_list_and_one_list_is_selected_at_a_time(qapp, tmp_path):
    from xdart.gui.analysis.phase_source import _REMOVE_NONE_TIP
    with _window(qapp) as dialog:
        source = dialog.phase_source
        reference = source.reference_browser
        phases, references = source.browser.scans, reference.scans
        one, two = source.add_cif_paths((str(_cif(tmp_path, "phaseA", 4.1)), str(_cif(tmp_path, "phaseB", 4.25))))
        source.add_calibrant("LaB6")
        source.add_calibrant("CeO2")
        qapp.processEvents()

        # nothing selected: Remove is off and says why
        assert not source.remove_btn.isEnabled() and _REMOVE_NONE_TIP in source.remove_btn.toolTip()

        # a phase is selected; then a calibrant, which clears the phase's selection
        _click(qapp, phases.viewport(), _label_point(phases, phases.item(1)))
        assert _selected(phases) == ["phaseB"] and _selected(references) == [] and source.remove_btn.isEnabled()
        _click(qapp, references.viewport(), _label_point(references, references.item(0)))
        assert _selected(references) == ["LaB6"] and _selected(phases) == [] and source.remove_btn.isEnabled()

        # Remove takes only that calibrant
        _click(qapp, source.remove_btn)
        assert source.selection == ((one.id, True), (two.id, True), (CEO2, True))
        assert source.browser.listed_results() == (one.id, two.id) and reference.listed_results() == (CEO2,)
        assert _selected(phases) == [] and _selected(references) == []
        assert not source.remove_btn.isEnabled() and _REMOVE_NONE_TIP in source.remove_btn.toolTip()

        # a calibrant is selected; then a phase, which clears the calibrant's selection; Remove takes only the phase
        _click(qapp, references.viewport(), _label_point(references, references.item(0)))
        assert _selected(references) == ["CeO2"]
        _click(qapp, phases.viewport(), _label_point(phases, phases.item(0)))
        assert _selected(phases) == ["phaseA"] and _selected(references) == []
        _click(qapp, source.remove_btn)
        assert source.selection == ((two.id, True), (CEO2, True))
        assert source.browser.listed_results() == (two.id,) and reference.listed_results() == (CEO2,)
        assert not source.remove_btn.isEnabled()


# ---- C4 with only a calibrant checked, Fit refuses with the calibrant reason it gave before -------------------------

def test_c4_with_only_a_calibrant_checked_fit_refuses_with_the_calibrant_reason(qapp, tmp_path):
    from xdart.gui.analysis.phase_fit_dialog import _CALIBRANT_ONLY
    submitted = []
    with _window(qapp, submitted) as dialog:
        source = dialog.phase_source
        reference = source.reference_browser
        (cif,) = source.add_cif_paths((str(_cif(tmp_path, "phaseA", 4.1)),))
        source.add_calibrant("LaB6")
        qapp.processEvents()
        assert dialog.fit_refusal() == ""
        # the phase is unchecked with a real click, so only the calibrant in the reference list stays included
        phases = source.browser.scans
        _click(qapp, phases.viewport(), _indicator(phases, phases.item(0)).center())
        assert source.selection == ((cif.id, False), (LAB6, True))
        assert reference.checked_results() == (LAB6,) and source.browser.checked_results() == ()
        assert dialog.fit_refusal() == _CALIBRANT_ONLY
        _click(qapp, dialog.fit_btn)
        assert dialog.status.text() == _CALIBRANT_ONLY
        assert "fit" not in [action for action, _values in submitted]


# ---- C5 the reference list's tooltips, its help topic, and its availability with the rest of the list ---------------

def test_c5_the_reference_list_explains_itself_carries_the_phases_help_and_greys_with_the_list(qapp, tmp_path):
    with _window(qapp) as dialog:
        source = dialog.phase_source
        reference = source.reference_browser
        source.add_calibrant("LaB6")
        qapp.processEvents()
        purposes = {widget: widget.toolTip() for widget in (reference, reference.scans, reference.scans_label)}
        assert all(purposes.values()), purposes
        assert "fitted" in purposes[reference].casefold()
        for child in reference.findChildren(QtWidgets.QWidget):  # its hidden actions too, as the phase list's
            if isinstance(child, (QtWidgets.QPushButton, QtWidgets.QLineEdit, QtWidgets.QComboBox)):
                assert child.toolTip(), child.objectName()
        # the phase list no longer claims the calibrants: its own tooltip names the reference list
        assert CAPTION in source.browser.scans.toolTip()
        assert reference.property("helpTopic") == "phase-analysis.phases"
        # the view greys the whole list (the 2D Viewer); the reference list goes with it and says why
        reason = "Reference phases are unavailable in the 2D Viewer."
        dialog.phase_binding.set_available(False, reason)
        assert not reference.isEnabled() and not reference.scans.isEnabled()
        for widget, purpose in purposes.items():
            assert reason in widget.toolTip() and purpose in widget.toolTip(), widget
        dialog.phase_binding.set_available(True)
        assert reference.isEnabled()
        assert {widget: widget.toolTip() for widget in purposes} == purposes


# ---- C6 the layout: the phase controls, then the reference list with its Calibrant row, then the notices -------------

def test_c6_the_reference_list_sits_below_the_phase_controls_with_its_calibrant_row_and_a_compact_height(qapp,
                                                                                                         tmp_path):
    from xrd_tools.integrate.calibration import CALIBRANT_NAMES
    from xdart.gui.widgets.controls_panel import SectionCard
    with _window(qapp) as dialog:
        source = dialog.phase_source
        reference = source.reference_browser
        layout = source.layout()
        widgets = (source.browser, source.library_row, source.global_switch, source.global_row, source.lattice_form,
                   reference, source.calibrant_row, source.status, source.note)
        order = [layout.indexOf(widget) for widget in widgets]
        assert min(order) >= 0 and order == sorted(order), order
        assert layout.indexOf(source.calibrant_row) == layout.indexOf(reference) + 1, "the Calibrant row is right under"
        buttons = next(index for index in range(layout.count()) if layout.itemAt(index).layout() is not None
                       and layout.itemAt(index).layout().indexOf(source.remove_btn) >= 0)
        assert layout.indexOf(source.browser) < buttons < layout.indexOf(source.library_row)
        for name in CALIBRANT_NAMES:  # six reference lines: a longer list scrolls, as the phase list does
            source.add_calibrant(name)
        qapp.processEvents()
        assert reference.listed_results() == tuple("calibrant:" + name for name in CALIBRANT_NAMES)
        shown = (source.browser, source.add_cif_btn, source.library_row, source.global_switch, source.global_row,
                 reference, source.calibrant_row)
        tops = [widget.mapTo(source, QtCore.QPoint(0, 0)).y() for widget in shown]
        assert tops == sorted(tops), tops
        # PA3's phase list is kept; the reference list is no taller than it and never below its own minimum
        browser = source.browser
        assert browser.height() <= max(110, browser.minimumSizeHint().height()) < 220
        assert reference.minimumSizeHint().height() <= reference.height() <= browser.height()
        assert reference.scans.verticalScrollBar().maximum() > 0
        # the options column scrolls; no card is squeezed below its own minimum
        column = dialog.split.controls.widget()
        cards = [card for card in column.findChildren(SectionCard) if card.parentWidget() is column]
        assert cards and all(card.height() >= card.minimumSizeHint().height() for card in cards)
        assert dialog.split.controls.horizontalScrollBar().maximum() == 0
