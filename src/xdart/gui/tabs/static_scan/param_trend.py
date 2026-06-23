# -*- coding: utf-8 -*-
"""Shared parameter-vs-frame trend (row 3) for the fitting tools.

A mixin both PeakFitDialog and PhaseFitDialog use: it owns the accumulator +
redraw + CSV logic, operating on widgets the host dialog builds
(``param_plot`` / ``param_family_combo`` / ``param_overlay_check`` /
``param_save_btn``) and state it initialises (``_param_accumulator`` /
``_param_family_keys`` / ``_x_label`` / ``status``).  The pure helpers live in
the Qt-free :mod:`peak_fit_util`; this is the thin Qt layer over them.
"""

import csv
import logging

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from .peak_fit_util import (CURVE_PENS, FAMILY_LABELS, X_UNIT_FAMILIES,
                            accumulator_to_table, group_families)

logger = logging.getLogger(__name__)


class ParamTrendMixin:
    """Accumulate ``{frame_index: params}`` and draw one curve per peak vs frame.

    Keyed by frame index, so a live drop / out-of-order re-fit updates rather
    than duplicates, and batch fills it densely.  Reset only at the start of a
    fresh run (Reload / live-enable / Batch) — never per frame."""

    def reset_param_trend(self):
        """Drop the accumulated vs-frame series + clear row 3."""
        self._param_accumulator = {}
        self._param_family_keys = ()
        self.param_family_combo.blockSignals(True)
        self.param_family_combo.clear()
        self.param_family_combo.blockSignals(False)
        self.param_family_combo2.blockSignals(True)
        self.param_family_combo2.clear()
        self.param_family_combo2.addItem("None", None)
        self.param_family_combo2.blockSignals(False)
        self.param_family_combo.setEnabled(False)
        self.param_family_combo2.setEnabled(False)
        self.param_overlay_check.setEnabled(False)
        self.param_save_btn.setEnabled(False)
        self.param_plot.clear()
        self.param_right_vb.clear()
        self.param_right_axis.setVisible(False)

    def _accumulate_frame_params(self, frame_idx, params):
        """Store one frame's params (update-not-duplicate) and refresh row 3."""
        if not params:
            return
        self._param_accumulator[int(frame_idx)] = dict(params)
        self._sync_param_families()
        self._redraw_param_plot()
        has_data = bool(self._param_accumulator)
        self.param_family_combo.setEnabled(has_data)
        self.param_family_combo2.setEnabled(has_data)
        self.param_overlay_check.setEnabled(has_data)
        self.param_save_btn.setEnabled(has_data)

    @staticmethod
    def _fill_family_combo(combo, families, *, include_none):
        """Repopulate a family combo, preserving the current selection by data.
        ``include_none`` prepends a ``None`` sentinel (the right-axis 'off')."""
        current = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        if include_none:
            combo.addItem("None", None)
        for fam in families:
            combo.addItem(FAMILY_LABELS.get(fam, fam), fam)
        if current is not None:
            i = combo.findData(current)
            if i >= 0:
                combo.setCurrentIndex(i)
        combo.blockSignals(False)

    def _sync_param_families(self):
        """Keep both family combos in sync with the families present so far,
        preserving each current selection."""
        keys = []
        for params in self._param_accumulator.values():
            for k in params:
                if k not in keys:
                    keys.append(k)
        families = tuple(group_families(keys))
        # Compare as a set: only rebuild when the SET of families changes, not
        # when their discovery order happens to differ between frames.
        if frozenset(families) == frozenset(self._param_family_keys):
            return
        self._fill_family_combo(self.param_family_combo, families,
                                include_none=False)
        self._fill_family_combo(self.param_family_combo2, families,
                                include_none=True)
        self._param_family_keys = families

    def _redraw_param_plot(self):
        self.param_plot.clear()
        self.param_right_vb.clear()
        if self.param_legend is not None:
            self.param_legend.clear()
        if not self._param_accumulator:
            self.param_right_axis.setVisible(False)
            return
        labels, columns = accumulator_to_table(self._param_accumulator)
        x = np.asarray([float(label) for label in labels], dtype=float)
        grouped = group_families(columns)
        overlay = self.param_overlay_check.isChecked()
        color_idx = 0

        # Left axis: the primary family.
        fam = self.param_family_combo.currentData()
        if fam is not None:
            members = grouped.get(fam, [])
            if not overlay:
                members = members[:1]       # just the first peak
            for key, _idx in members:
                y = np.asarray(columns[key], dtype=float)
                color = CURVE_PENS[color_idx % len(CURVE_PENS)]
                color_idx += 1
                self.param_plot.plot(x, y, pen=pg.mkPen(color, width=2),
                                     name=key, symbol="o", symbolSize=5,
                                     symbolBrush=color)
            self.param_plot.setLabel("left", self._family_axis_label(fam))

        # Right axis: an optional SECOND family (cross-family overlay) so e.g.
        # peak center (~2) and amplitude (~1e5) read cleanly together.
        fam2 = self.param_family_combo2.currentData()
        any_right = False
        if fam2 is not None and fam2 != fam:
            members2 = grouped.get(fam2, [])
            if not overlay:
                members2 = members2[:1]
            for key, _idx in members2:
                y = np.asarray(columns[key], dtype=float)
                color = CURVE_PENS[color_idx % len(CURVE_PENS)]
                color_idx += 1
                item = pg.PlotDataItem(
                    x, y, pen=pg.mkPen(color, width=2,
                                       style=QtCore.Qt.PenStyle.DashLine),
                    name=key, symbol="t", symbolSize=5, symbolBrush=color)
                self.param_right_vb.addItem(item)
                if self.param_legend is not None:
                    self.param_legend.addItem(item, f"{key} (R)")
                any_right = True
            self.param_right_axis.setLabel(self._family_axis_label(fam2))
        self.param_right_axis.setVisible(any_right)
        self.param_right_vb.setVisible(any_right)

    def _family_axis_label(self, fam):
        unit = self._x_label if fam in X_UNIT_FAMILIES else "Intensity"
        label = FAMILY_LABELS.get(fam, fam)
        return f"{label} ({unit})" if unit else label

    def _save_param_csv(self):
        if not self._param_accumulator:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save parameters vs frame", "fit_params_vs_frame.csv",
            "CSV files (*.csv)")
        if not path:
            return
        labels, columns = accumulator_to_table(self._param_accumulator)
        keys = list(columns)
        try:
            with open(path, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["frame"] + keys)
                for i, label in enumerate(labels):
                    writer.writerow([label] + [columns[k][i] for k in keys])
            self.status.setText(f"Saved {path}")
        except Exception:
            logger.exception("param-vs-frame CSV save failed")
            self.status.setText("Could not save CSV (see log).")

    def _build_param_trend_row(self, lay):
        """Build the row-3 widgets (track combo + overlay + save + plot) into the
        host dialog's layout.  The host connects the signals + inits the state."""
        track_row = QtWidgets.QHBoxLayout()
        track_row.setSpacing(7)
        track_row.addWidget(QtWidgets.QLabel("Track"))
        self.param_family_combo = QtWidgets.QComboBox()
        self.param_family_combo.setToolTip(
            "Which fitted parameter to plot vs frame number")
        self.param_family_combo.setEnabled(False)
        self.param_overlay_check = QtWidgets.QCheckBox("Overlay")
        self.param_overlay_check.setToolTip(
            "Plot every series in the selected parameter on one axis, not just "
            "the first")
        # A second family on a right-hand axis (cross-family overlay).
        self.param_family_combo2 = QtWidgets.QComboBox()
        self.param_family_combo2.setToolTip(
            "Overlay a SECOND parameter family against a right-hand axis (None = "
            "off) — for families of very different magnitude")
        self.param_family_combo2.addItem("None", None)
        self.param_family_combo2.setEnabled(False)
        track_row.addWidget(self.param_family_combo)
        track_row.addWidget(self.param_overlay_check)
        track_row.addWidget(QtWidgets.QLabel("Right"))
        track_row.addWidget(self.param_family_combo2)
        track_row.addStretch(1)
        self.param_save_btn = QtWidgets.QPushButton("Save CSV…")
        self.param_save_btn.setToolTip("Export the parameters-vs-frame table")
        self.param_save_btn.setEnabled(False)
        track_row.addWidget(self.param_save_btn)
        lay.addLayout(track_row)

        self.param_plot = pg.PlotWidget()
        self.param_plot.setMinimumHeight(150)
        self.param_legend = self.param_plot.addLegend(offset=(-10, 10))
        self.param_plot.setLabel("bottom", "Frame")
        from .plot_axes import attach_right_axis
        self.param_right_vb, self.param_right_axis = attach_right_axis(
            self.param_plot)
        lay.addWidget(self.param_plot, 2)

    def _connect_param_trend(self):
        """Wire the row-3 signals (call after _build_param_trend_row)."""
        self.param_family_combo.currentIndexChanged.connect(self._redraw_param_plot)
        self.param_family_combo2.currentIndexChanged.connect(self._redraw_param_plot)
        self.param_overlay_check.toggled.connect(self._redraw_param_plot)
        self.param_save_btn.clicked.connect(self._save_param_csv)
