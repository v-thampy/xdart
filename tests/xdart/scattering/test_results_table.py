"""The shared results table (``tabs/scattering/results_table.py``; FIT-TOOLS-2026-10, Codex's design review F7): the
one table a fit window mounts, and the rows of each fit result made once. Its consumers' rows pin what they show:
Int's Peak Fit and Phase Fit in ``test_fit_tools_preservation.py`` and ``test_p37_analysis_mount.py``."""
from __future__ import annotations

from types import SimpleNamespace

from pyqtgraph.Qt import QtCore, QtTest, QtWidgets

from xdart.gui.tabs.scattering.results_table import (
    PEAK_OUTCOME_HEADERS, PEAK_PARAMETER_HEADERS, PHASE_OUTCOME_HEADERS, PHASE_RESULT_HEADERS, ResultsTable,
    peak_outcome_rows, peak_parameter_rows, phase_outcome_rows, phase_result_rows,
)
from xrd_tools.analysis.display_fit_operations import DisplayedPeakFitResult, DisplayedPhaseFitResult
from xrd_tools.analysis.scan_operations import AnalysisDisposition


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _shown(table):
    headers = [table.horizontalHeaderItem(column).text() for column in range(table.columnCount())]
    return headers, [[table.item(row, column).text() for column in range(table.columnCount())]
                     for row in range(table.rowCount())]


def test_a_results_table_is_read_only_rows_of_text_under_its_headers() -> None:
    _app()
    table = ResultsTable(("A", "B"), name="someTable", max_height=130)
    try:
        assert table.objectName() == "someTable" and table.maximumHeight() == 130
        assert table.editTriggers() == QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        assert not table.verticalHeader().isVisible() and table.horizontalHeader().stretchLastSection()
        assert _shown(table) == (["A", "B"], [])
        table.show_rows(("Name", "Value", "Err"), (("x", 1.5, None), ("y", 2, 0.25)))
        assert _shown(table) == (["Name", "Value", "Err"], [["x", "1.5", "None"], ["y", "2", "0.25"]])
        table.clear_rows()
        assert _shown(table) == (["Name", "Value", "Err"], [])
    finally:
        table.deleteLater()


def test_results_table_owns_checkboxes_tips_and_calibration_row_selection() -> None:
    app = _app()
    table = ResultsTable(("Use", "Frame"), row_numbers=True, stretch_columns=True,
                         row_selection=True, minimum_rows=6, header_tooltips={0: "Prepare checked frames."})
    try:
        table.show_rows(("Use", "Frame"), (("", 10), ("", 20)),
                        check_states={(0, 0): True, (1, 0): False},
                        cell_tooltips={(0, 1): "Original detector frame 10."})
        table.resize(500, 300)
        table.show()
        app.processEvents()
        assert table.verticalHeader().isVisible()
        assert table.selectionBehavior() == QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        assert table.selectionMode() == QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        assert table.horizontalHeader().sectionResizeMode(0) == QtWidgets.QHeaderView.ResizeMode.Stretch
        assert table.minimumHeight() >= 6 * table.verticalHeader().defaultSectionSize()
        assert table.horizontalHeaderItem(0).toolTip() == "Prepare checked frames."
        assert table.item(0, 1).toolTip() == "Original detector frame 10."
        assert table.item(0, 0).checkState() == QtCore.Qt.CheckState.Checked
        checkbox = table.visualItemRect(table.item(1, 0)).topLeft() + QtCore.QPoint(10, 14)
        QtTest.QTest.mouseClick(table.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=checkbox)
        assert table.item(1, 0).checkState() == QtCore.Qt.CheckState.Checked
        table.selectRow(1)
        assert table.currentRow() == 1
        table.show_rows(("Use", "Frame"), (("", 30),), check_states={(0, 0): False})
        assert table.item(0, 0).checkState() == QtCore.Qt.CheckState.Unchecked
        assert not table.item(0, 1).toolTip()
        assert table.horizontalHeaderItem(0).toolTip() == "Prepare checked frames."
    finally:
        table.close()
        table.deleteLater()


def test_a_displayed_peak_fits_rows_are_its_parameters_values_and_errors() -> None:
    result = DisplayedPeakFitResult(AnalysisDisposition.COMPLETED, "OK", parameter_names=("p0_center", "p0_sigma"),
                                    parameter_values=(2.1, 0.2), parameter_stderr=(0.01, None))
    assert peak_parameter_rows(result) == (PEAK_PARAMETER_HEADERS, (("p0_center", 2.1, 0.01),
                                                                     ("p0_sigma", 0.2, None)))
    assert PEAK_PARAMETER_HEADERS == ("Parameter", "Value", "StdErr")


def test_a_displayed_phase_fits_rows_are_its_phases_then_its_parameters() -> None:
    result = DisplayedPhaseFitResult(
        AnalysisDisposition.COMPLETED, "OK", parameter_names=("scale",), parameter_values=(3.0,),
        parameter_stderr=(0.2,), phase_fractions=(("alpha", 0.6), ("beta", 0.4)),
        lattice_parameters=(("alpha", (("a", 4.1), ("c", 5.0))),))
    assert phase_result_rows(result) == (PHASE_RESULT_HEADERS, (
        ("alpha", 0.6, "a=4.1, c=5.0", "phase", ""), ("beta", 0.4, "", "phase", ""),
        ("scale", 3.0, "", "parameter", 0.2)))
    assert PHASE_RESULT_HEADERS == ("Name", "Value", "Lattice", "Kind", "StdErr")


def test_an_analyzer_peak_fits_rows_are_its_peaks_with_the_dialogs_formats() -> None:
    fwhm = SimpleNamespace(value=0.0589)
    payload = SimpleNamespace(peak_centers=[1.60012, 2.4], peak_centers_err=[0.000123, None],
                              peak_sigmas=[0.025, 0.03], peak_amplitudes=[900.4, 600.0],
                              params={"p0_fwhm": fwhm})
    assert peak_outcome_rows(payload) == (PEAK_OUTCOME_HEADERS, (
        ("1", "1.6001", "0.00012", "0.0589", "900.4"),
        ("2", "2.4", "nan", f"{2.3548 * 0.03:.4g}", "600")))
    assert PEAK_OUTCOME_HEADERS == ("#", "Center", "Center ±", "FWHM", "Amplitude")


def test_an_analyzer_phase_fits_rows_are_its_phases_fractions_and_cell() -> None:
    entry = {"phase_fractions": {"TiN": 0.75, "HfO2": 0.25}, "lattice_params": {"TiN": {"a": 4.24, "b": 4.24}}}
    assert phase_outcome_rows(entry, ()) == (PHASE_OUTCOME_HEADERS, (
        ("TiN", "0.750", "4.2400", "4.2400", "—"), ("HfO2", "0.250", "—", "—", "—")))
    # LT-SMALL-FIXES-20261014 V2: an absent fraction shows "—", as the owner presents it, and its reason as a row
    assert phase_outcome_rows(None, ("TiN",)) == (PHASE_OUTCOME_HEADERS, (("TiN", "—", "—", "—", "—"),))
    withheld = {"phase_fractions_reason": "fractions withheld: the fit did not converge"}
    assert phase_outcome_rows(withheld, ("TiN",)) == (PHASE_OUTCOME_HEADERS, (
        ("TiN", "—", "—", "—", "—"), ("fractions withheld: the fit did not converge", "—", "—", "—", "—")))


def test_a_lattice_fits_withheld_fraction_reason_reaches_both_phase_tables(tmp_path) -> None:
    # Split from tests/core/test_lattice_phase_fit_contract.py (its store half stays there): the core runner has no GUI
    # libraries, so the shared tables' rows of a real lattice fit are pinned here.
    from tests.core.test_lattice_phase_fit_contract import _plan, _stored_lattice_fit
    from xrd_tools.analysis.display_fit_operations import run_displayed_phase_fit
    from xrd_tools.analysis.phase_references import PhaseLibrary
    result, entry = _stored_lattice_fit()
    _, rows = phase_outcome_rows(entry, ("positions",))
    assert rows[0][0] == "positions" and rows[0][1] == "—"
    assert rows[0][2:] == ("4.1000", "4.1000", "4.1000")
    assert any(result.phase_fractions_reason in str(cell) for row in rows for cell in row)
    library = PhaseLibrary(tmp_path / "library")
    lattice = library.add_lattice("positions", a=4.1, space_group="Pm-3m")
    displayed = run_displayed_phase_fit(_plan(tmp_path, (lattice,)))
    _, rows = phase_result_rows(displayed)
    phase_rows = [row for row in rows if row[3] == "phase"]
    assert len(phase_rows) == 1 and phase_rows[0][0:2] == ("positions", "—")
    assert "a=" in phase_rows[0][2]
    assert any(displayed.phase_fractions_reason in str(cell) for row in rows for cell in row)
