"""The shared results table, beside ``FitView``: a fit's (or any analysis's) result rows, read-only, under headers.

FIT-TOOLS-2026-10 (Codex's design review F7: "Extract the existing fit table rendering/formatting into the shared
table owner; reuse that owner across both fits"). Int's Peak Fit and Phase Fit built this table twice and filled it
three ways; it is now this one owner, and the rows of each fit result are made here, once, so Stitch's and RSM's
fit windows show the same rows. A window mounts ``ResultsTable`` and calls ``show_rows`` with one of the row
projections below (or its own rows); it builds no table of its own (``test_gui_reuse_fence``: ``QTableWidget``,
``QTableView``).
"""
from __future__ import annotations

from collections.abc import Sequence

from pyqtgraph.Qt import QtWidgets

#: A displayed peak fit's parameters, as Int shows them (``peak_parameter_rows``).
PEAK_PARAMETER_HEADERS = ("Parameter", "Value", "StdErr")
#: A displayed phase fit's phases and parameters (``phase_result_rows``).
PHASE_RESULT_HEADERS = ("Name", "Value", "Lattice", "Kind", "StdErr")
#: The peaks of an analyzer's peak fit (the dialog's provider path: ``peak_outcome_rows``).
PEAK_OUTCOME_HEADERS = ("#", "Center", "Center ±", "FWHM", "Amplitude")
#: The phases of an analyzer's phase fit (``phase_outcome_rows``).
PHASE_OUTCOME_HEADERS = ("Phase", "Fraction", "a", "b", "c")


class ResultsTable(QtWidgets.QTableWidget):
    """Rows of values under headers, read-only: no row numbers, the last column stretched, each value shown as its
    text. ``show_rows`` replaces what is shown; ``clear_rows`` empties it, keeping the headers."""

    def __init__(self, headers: Sequence[str] = (), parent=None, *, name: str = "resultsTable",
                 max_height: int | None = None) -> None:
        super().__init__(0, len(headers), parent)
        self.setObjectName(name)
        self.setHorizontalHeaderLabels(list(headers))
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        if max_height is not None:
            self.setMaximumHeight(max_height)
        self.horizontalHeader().setStretchLastSection(True)

    def show_rows(self, headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
        """Show ``rows`` under ``headers``: each value as ``str(value)`` (an unknown error reads "None")."""
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(list(headers))
        self.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                self.setItem(row, column, QtWidgets.QTableWidgetItem(str(value)))

    def clear_rows(self) -> None:
        """No rows; the headers stay."""
        self.setRowCount(0)


def peak_parameter_rows(result) -> tuple[tuple[str, ...], tuple[tuple[object, ...], ...]]:
    """A displayed peak fit's rows (``DisplayedPeakFitResult``): each fitted parameter, its value and its error."""
    return PEAK_PARAMETER_HEADERS, tuple(zip(result.parameter_names, result.parameter_values,
                                             result.parameter_stderr, strict=True))


def phase_result_rows(result) -> tuple[tuple[str, ...], tuple[tuple[object, ...], ...]]:
    """A displayed phase fit's rows (``DisplayedPhaseFitResult``): each phase's fraction and lattice, then each
    fitted parameter with its error."""
    lattice = dict(result.lattice_parameters)
    rows = [(name, value, ", ".join(f"{key}={number}" for key, number in lattice.get(name, ())), "phase", "")
            for name, value in result.phase_fractions]
    rows += [(name, value, "", "parameter", error) for name, value, error
             in zip(result.parameter_names, result.parameter_values, result.parameter_stderr, strict=True)]
    return PHASE_RESULT_HEADERS, tuple(rows)


def peak_outcome_rows(payload) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """An analyzer peak fit's peaks (a ``PeakFitResult1D``): each centre and its error, FWHM (the fit's own, else
    the Gaussian-equivalent of its sigma) and amplitude."""
    centers = list(payload.peak_centers or [])
    errors = list(getattr(payload, "peak_centers_err", []) or [])
    sigmas = list(payload.peak_sigmas or [])
    amplitudes = list(payload.peak_amplitudes or [])
    params = getattr(payload, "params", None)
    rows = []
    for index, centre in enumerate(centers):
        fwhm = None
        if params is not None:
            parameter = params.get(f"p{index}_fwhm")
            if parameter is not None:
                fwhm = parameter.value
        if fwhm is None and index < len(sigmas):
            fwhm = 2.3548 * sigmas[index]
        error = errors[index] if index < len(errors) and errors[index] is not None else float("nan")
        amplitude = amplitudes[index] if index < len(amplitudes) else float("nan")
        rows.append((str(index + 1), f"{centre:.5g}", f"{error:.2g}",
                     f"{fwhm:.4g}" if fwhm is not None else "—", f"{amplitude:.4g}"))
    return PEAK_OUTCOME_HEADERS, tuple(rows)


def phase_outcome_rows(entry, names: Sequence[str]) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """An analyzer phase fit's phases (a ``FitResultStore`` entry, or None): each phase's fraction and a, b, c;
    ``names`` are listed when the entry has no fractions."""
    fractions = (entry or {}).get("phase_fractions", {}) or {}
    lattice = (entry or {}).get("lattice_params", {}) or {}
    rows = []
    for name in list(fractions) or list(names):
        cell = lattice.get(name, {})
        rows.append((name, f"{fractions.get(name, float('nan')):.3f}",
                     *(f"{cell[key]:.4f}" if cell.get(key) is not None else "—" for key in ("a", "b", "c"))))
    return PHASE_OUTCOME_HEADERS, tuple(rows)



__all__ = [
    "PEAK_OUTCOME_HEADERS", "PEAK_PARAMETER_HEADERS", "PHASE_OUTCOME_HEADERS", "PHASE_RESULT_HEADERS",
    "ResultsTable", "peak_outcome_rows", "peak_parameter_rows", "phase_outcome_rows", "phase_result_rows",
]
