# -*- coding: utf-8 -*-
"""Batch peak-fit results: parameters vs frame number (analyzer framework Step 4).

A read-only popup that plots the per-frame parameter series a batch run produced
— one curve per peak, a family selector to switch between centers / FWHM /
amplitude, and CSV export.  It is fed the headless ``(labels, columns)`` table
that :func:`xrd_tools.analysis.runner.batch_params_table` builds, so the plot is
pure presentation: no fitting or analysis logic lives here."""
import csv
import logging

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

logger = logging.getLogger(__name__)

#: param-family key -> friendly label.  Families a peak analyzer emits are
#: ``center`` / ``center_err`` / ``fwhm`` / ``amplitude`` (see _peak_params).
_FAMILY_LABELS = {
    "center": "Peak center",
    "center_err": "Center uncertainty",
    "fwhm": "FWHM",
    "amplitude": "Amplitude",
}
#: families whose VALUES are in the pattern's x-unit (vs. intensity).
_X_UNIT_FAMILIES = {"center", "center_err", "fwhm"}

_CURVE_PENS = [(189, 147, 249), (80, 250, 123), (255, 184, 108),
               (139, 233, 253), (255, 121, 198), (241, 250, 140)]


def split_family(key):
    """``'center_0' -> ('center', 0)``; ``'center_err_2' -> ('center_err', 2)``;
    a key with no trailing ``_<int>`` -> ``(key, 0)``."""
    head, _, tail = key.rpartition("_")
    if head and tail.isdigit():
        return head, int(tail)
    return key, 0


def group_families(columns):
    """``{param_key: series}`` -> ordered ``{family: [(key, peak_index), ...]}``
    (each family's peaks sorted by index) so a family's curves plot together."""
    fam: "dict[str, list]" = {}
    for key in columns:
        f, i = split_family(key)
        fam.setdefault(f, []).append((key, i))
    for f in fam:
        fam[f].sort(key=lambda ki: ki[1])
    return fam


class BatchFitResultsDialog(QtWidgets.QDialog):
    """Non-modal popup: parameter-vs-frame plot for a completed batch fit."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("batchFitResultsDialog")
        self.setWindowTitle("Batch fit — parameters vs frame")
        self.resize(640, 480)
        self._labels = []
        self._columns = {}
        self._families = {}
        self._x_unit = ""
        self._build_ui()

    def _build_ui(self):
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(11, 11, 11, 11)
        lay.setSpacing(8)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(7)
        row.addWidget(QtWidgets.QLabel("Parameter"))
        self.family_combo = QtWidgets.QComboBox()
        self.family_combo.currentIndexChanged.connect(self._redraw)
        row.addWidget(self.family_combo)
        row.addStretch(1)
        self.save_btn = QtWidgets.QPushButton("Save CSV…")
        self.save_btn.clicked.connect(self._save_csv)
        row.addWidget(self.save_btn)
        lay.addLayout(row)

        self.plot = pg.PlotWidget()
        self.plot.addLegend(offset=(-10, 10))
        self.plot.setLabel("bottom", "Frame")
        lay.addWidget(self.plot)

        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("batchFitStatus")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

    def set_results(self, labels, columns, x_unit=""):
        """Load a completed batch table and draw the default (first) family."""
        self._labels = list(labels or [])
        self._columns = dict(columns or {})
        self._x_unit = x_unit or ""
        self._families = group_families(self._columns)
        self.family_combo.blockSignals(True)
        self.family_combo.clear()
        for fam in self._families:
            self.family_combo.addItem(_FAMILY_LABELS.get(fam, fam), fam)
        self.family_combo.blockSignals(False)
        if not self._columns:
            self.status.setText(
                f"{len(self._labels)} frames processed — no peaks fit "
                "(check the model / range / Auto settings, then re-run Batch).")
        else:
            self.status.setText(f"{len(self._labels)} frames fit.")
        self._redraw()

    def _frame_x(self):
        """Numeric x from the frame labels (falls back to sequence position)."""
        xs = []
        for label in self._labels:
            try:
                xs.append(float(label))
            except (TypeError, ValueError):
                xs.append(float(len(xs)))
        return np.asarray(xs, dtype=float)

    def _redraw(self):
        self.plot.clear()
        if not self._columns:
            return
        fam = self.family_combo.currentData()
        if fam is None:
            return
        x = self._frame_x()
        for n, (key, _idx) in enumerate(self._families.get(fam, [])):
            y = np.asarray(self._columns[key], dtype=float)
            color = _CURVE_PENS[n % len(_CURVE_PENS)]
            self.plot.plot(x, y, pen=pg.mkPen(color, width=2), name=key,
                           symbol="o", symbolSize=5, symbolBrush=color)
        unit = self._x_unit if fam in _X_UNIT_FAMILIES else "Intensity"
        label = _FAMILY_LABELS.get(fam, fam)
        self.plot.setLabel("left", f"{label} ({unit})" if unit else label)

    def _save_csv(self):
        if not self._columns:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save batch fit parameters", "batch_fit_params.csv",
            "CSV files (*.csv)")
        if not path:
            return
        keys = list(self._columns)
        try:
            with open(path, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["frame"] + keys)
                for i, label in enumerate(self._labels):
                    writer.writerow([label] + [self._columns[k][i] for k in keys])
            self.status.setText(f"Saved {path}")
        except Exception:
            logger.exception("batch CSV save failed")
            self.status.setText("Could not save CSV (see log).")
