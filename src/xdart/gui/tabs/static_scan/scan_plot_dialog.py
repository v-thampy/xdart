# -*- coding: utf-8 -*-
"""Scan Plot popup (Direction-A Tools) — step 1: metadata plotting.

Plot per-frame scan metadata: any column vs any column, several overlaid to
compare, with an optional normalization column (y / norm).  NOT tied to the
loaded ``.nxs`` — a source picker opens any "scan" the headless source layer
classifies (processed NeXus / Eiger / TIFF-or-RAW sequence / SPEC) via
``xrd_tools.sources``; the per-frame columns come from
``xrd_tools.io.read_scan_data`` (processed NeXus) or the source's own metadata.

Step 1 = metadata only.  The "Plot ROI" path (ROI stats as computed columns) is
a later increment — see docs/design/design_scan_plotter_metadata_roi_jun2026.md.
"""

import logging
import os

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from .peak_fit_util import CURVE_PENS

logger = logging.getLogger(__name__)


def load_scan_table(uri):
    """Return ``(label, {column: ndarray})`` of per-frame metadata for a scan
    URI, classifying it via the headless source layer.

    Processed NeXus → the full ``scan_data`` table (every motor + counter).
    Other kinds (Eiger / TIFF-or-RAW series / NeXus stack) → best-effort:
    ``frame_index`` plus any per-frame ``motors`` the source exposes.  Thin GUI
    orchestration over headless functions (no analysis here)."""
    from xrd_tools.io import read_scan_data
    from xrd_tools.sources import SourceKind, guess_source_kind, open_source

    label = os.path.basename(str(uri))
    kind = guess_source_kind(uri)
    if kind is SourceKind.PROCESSED_NEXUS:
        table = read_scan_data(uri)
        if table:
            return label, table
        # fall through to the generic path if there was no scan_data group
    table = {}
    try:
        src = open_source(uri)
    except Exception:
        logger.exception("scan-plot: could not open source %s", uri)
        return label, table
    try:
        idx = np.asarray(list(src.frame_indices), dtype=float)
        table["frame_index"] = idx
        motors = getattr(src, "motors", None) or {}
        for name, values in motors.items():
            arr = np.asarray(values, dtype=float)
            if arr.ndim == 1 and arr.shape[0] == idx.shape[0]:
                table[str(name)] = arr
    except Exception:
        logger.exception("scan-plot: could not read metadata from %s", uri)
    return label, table


def _numeric_columns(table):
    """Plottable columns: 1-D, numeric, finite-bearing."""
    out = []
    for name, arr in table.items():
        a = np.asarray(arr)
        if a.ndim == 1 and a.dtype.kind in "fiu" and np.any(np.isfinite(a)):
            out.append(name)
    return out


class ScanPlotDialog(QtWidgets.QDialog):
    """Plot scan metadata columns vs each other, overlaid, with normalization."""

    def __init__(self, default_uri=None, parent=None):
        super().__init__(parent)
        self._table = {}
        self._columns = []
        self.setObjectName("scanPlotDialog")
        self.setWindowTitle("Scan Plot")
        self.resize(640, 620)
        self._build_ui()
        if default_uri:
            self.load_uri(default_uri)

    def _build_ui(self):
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(11, 11, 11, 11)
        lay.setSpacing(8)

        # Source row.
        src_row = QtWidgets.QHBoxLayout()
        src_row.setSpacing(7)
        src_row.addWidget(QtWidgets.QLabel("Source"))
        self.source_label = QtWidgets.QLineEdit()
        self.source_label.setReadOnly(True)
        self.source_label.setPlaceholderText("(no scan chosen)")
        src_row.addWidget(self.source_label, 1)
        self.choose_btn = QtWidgets.QPushButton("Choose…")
        self.choose_btn.setToolTip(
            "Open a scan: processed NeXus, Eiger, the first image of a "
            "TIFF/RAW sequence, or a SPEC file")
        src_row.addWidget(self.choose_btn)
        lay.addLayout(src_row)

        # Axes row: X / Normalize combos (Y is the multi-select below).
        axes_row = QtWidgets.QHBoxLayout()
        axes_row.setSpacing(7)
        axes_row.addWidget(QtWidgets.QLabel("X"))
        self.x_combo = QtWidgets.QComboBox()
        self.x_combo.setMinimumWidth(120)
        axes_row.addWidget(self.x_combo)
        axes_row.addWidget(QtWidgets.QLabel("Normalize"))
        self.norm_combo = QtWidgets.QComboBox()
        self.norm_combo.setMinimumWidth(110)
        self.norm_combo.setToolTip("Divide Y by this column (None = off)")
        axes_row.addWidget(self.norm_combo)
        axes_row.addStretch(1)
        self.save_btn = QtWidgets.QPushButton("Save CSV…")
        self.save_btn.setEnabled(False)
        axes_row.addWidget(self.save_btn)
        lay.addLayout(axes_row)

        # Y multi-select (check several to overlay) + the plot, side by side.
        body = QtWidgets.QHBoxLayout()
        body.setSpacing(7)
        y_col = QtWidgets.QVBoxLayout()
        y_col.setSpacing(2)
        y_col.addWidget(QtWidgets.QLabel("Y (overlay)"))
        self.y_list = QtWidgets.QListWidget()
        self.y_list.setObjectName("scanPlotYList")
        self.y_list.setMaximumWidth(160)
        self.y_list.setToolTip("Check one or more columns to plot vs X")
        y_col.addWidget(self.y_list, 1)
        body.addLayout(y_col)
        self.plot = pg.PlotWidget()
        self.plot.addLegend(offset=(-10, 10))
        body.addWidget(self.plot, 1)
        lay.addLayout(body, 1)

        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("peakFitStatus")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

        self.choose_btn.clicked.connect(self._choose_source)
        self.x_combo.currentIndexChanged.connect(self._redraw)
        self.norm_combo.currentIndexChanged.connect(self._redraw)
        self.y_list.itemChanged.connect(self._redraw)
        self.save_btn.clicked.connect(self._save_csv)

    # ---- source loading -------------------------------------------------
    def _choose_source(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose a scan", "",
            "Scans (*.nxs *.h5 *.hdf5 *.cxi *.tif *.tiff *.spec);;All files (*)")
        if path:
            self.load_uri(path)

    def load_uri(self, uri):
        label, table = load_scan_table(uri)
        self.set_table(label, table)

    def set_table(self, label, table):
        """Load a per-frame column table + populate the X/Y/Normalize selectors."""
        self._table = dict(table or {})
        self._columns = _numeric_columns(self._table)
        self.source_label.setText(label or "")
        cols = self._columns

        self.x_combo.blockSignals(True)
        self.norm_combo.blockSignals(True)
        self.y_list.blockSignals(True)
        self.x_combo.clear()
        self.norm_combo.clear()
        self.y_list.clear()
        self.x_combo.addItems(cols)
        self.norm_combo.addItem("None", None)
        for c in cols:
            self.norm_combo.addItem(c, c)
        for c in cols:
            item = QtWidgets.QListWidgetItem(c)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.y_list.addItem(item)
        # Defaults: X = frame_index if present else first column; Y = first
        # column that isn't X.
        if cols:
            x_default = "frame_index" if "frame_index" in cols else cols[0]
            self.x_combo.setCurrentText(x_default)
            y_default = next((c for c in cols if c != x_default), None)
            if y_default is not None:
                items = self.y_list.findItems(y_default, QtCore.Qt.MatchFlag.MatchExactly)
                if items:
                    items[0].setCheckState(QtCore.Qt.CheckState.Checked)
        self.x_combo.blockSignals(False)
        self.norm_combo.blockSignals(False)
        self.y_list.blockSignals(False)

        self.save_btn.setEnabled(bool(cols))
        if not self._table:
            self.status.setText("No metadata found for this source.")
        elif not cols:
            self.status.setText(
                f"{len(self._table)} column(s), none numerically plottable.")
        else:
            self.status.setText(
                f"{len(cols)} plottable column(s). Pick X and check Y column(s).")
        self._redraw()

    def _checked_y(self):
        out = []
        for i in range(self.y_list.count()):
            item = self.y_list.item(i)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                out.append(item.text())
        return out

    def _redraw(self):
        self.plot.clear()
        if not self._columns:
            return
        xcol = self.x_combo.currentText()
        if xcol not in self._table:
            return
        x = np.asarray(self._table[xcol], dtype=float)
        norm_col = self.norm_combo.currentData()
        norm = None
        if norm_col and norm_col in self._table:
            norm = np.asarray(self._table[norm_col], dtype=float)
            norm = np.where(norm == 0, np.nan, norm)   # avoid /0
        normalized = norm is not None
        for n, ycol in enumerate(self._checked_y()):
            y = np.asarray(self._table[ycol], dtype=float)
            if normalized:
                y = y / norm
            color = CURVE_PENS[n % len(CURVE_PENS)]
            self.plot.plot(x, y, pen=pg.mkPen(color, width=2), name=ycol,
                           symbol="o", symbolSize=4, symbolBrush=color)
        self.plot.setLabel("bottom", xcol)
        self.plot.setLabel("left", "value / " + norm_col if normalized else "value")

    def _save_csv(self):
        if not self._columns:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save scan table", "scan_table.csv", "CSV files (*.csv)")
        if not path:
            return
        import csv
        cols = self._columns
        n = max((len(self._table[c]) for c in cols), default=0)
        try:
            with open(path, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(cols)
                for r in range(n):
                    writer.writerow([
                        self._table[c][r] if r < len(self._table[c]) else ""
                        for c in cols])
            self.status.setText(f"Saved {path}")
        except Exception:
            logger.exception("scan-table CSV save failed")
            self.status.setText("Could not save CSV (see log).")
