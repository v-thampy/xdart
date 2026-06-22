# -*- coding: utf-8 -*-
"""Self-contained Peak Fitting popup (Direction-A Tools).

Opens from the bottom-left Tools card.  Grabs the currently displayed 1-D
integrated pattern, fits it with the headless ``xrd_tools.analysis.fitting``
API over a user-chosen range (auto-detecting peaks by default), and shows
data + fit + residual in its own pyqtgraph plots plus a results table — all
isolated from the main display (no change to the main plot render path).
Non-modal so the user can change the selected frame and re-fit.

The fitting backend is headless/Qt-free; this module is the thin Qt front-end.
``lmfit`` (the ``xrd-tools[fitting]`` extra) is imported lazily so xdart still
launches without it — the dialog shows a friendly install hint instead.
"""

import logging

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

logger = logging.getLogger(__name__)

_MAX_PEAKS = 12

# (display label, fit_peaks model string)
_MODELS = [
    ("Pseudo-Voigt", "pseudovoigt"),
    ("Gaussian", "gaussian"),
    ("Lorentzian", "lorentzian"),
    ("Voigt", "voigt"),
]
# (display label, fit_peaks background string)
_BACKGROUNDS = [
    ("Linear", "linear"),
    ("Constant", "constant"),
    ("None", "none"),
    ("Chebyshev (3)", "chebyshev3"),
]


class PeakFitDialog(QtWidgets.QDialog):
    """Peak-fit the active 1-D pattern.

    Parameters
    ----------
    pattern_provider : callable
        Zero-arg callable returning ``(x, y, x_label)`` for the currently
        selected frame, or ``None`` when nothing is selectable.  Called on
        :meth:`refresh_pattern` (dialog open / Reload) so the dialog always
        fits what the user is looking at.
    """

    def __init__(self, pattern_provider, parent=None):
        super().__init__(parent)
        self._provider = pattern_provider
        self._x = None
        self._y = None
        self._x_label = "q"
        # Persisted fit range (in x-units); None => whole pattern.  Survives
        # Reload so stepping through frames keeps the chosen window.
        self._fit_lo = None
        self._fit_hi = None
        self._sync_guard = False
        self.setObjectName("peakFitDialog")
        self.setWindowTitle("Peak Fitting")
        self.resize(560, 660)
        self._build_ui()

    # ---- UI construction ------------------------------------------------
    def _build_ui(self):
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(11, 11, 11, 11)
        lay.setSpacing(8)

        # Row 1: model / background / peak count
        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(7)
        self.model_combo = QtWidgets.QComboBox()
        for label, _ in _MODELS:
            self.model_combo.addItem(label)
        self.bkg_combo = QtWidgets.QComboBox()
        for label, _ in _BACKGROUNDS:
            self.bkg_combo.addItem(label)
        self.auto_check = QtWidgets.QCheckBox("Auto")
        self.auto_check.setToolTip(
            "Detect peaks automatically (scipy find_peaks) over the fit range")
        self.auto_check.setChecked(True)
        self.npeaks_spin = QtWidgets.QSpinBox()
        self.npeaks_spin.setRange(1, _MAX_PEAKS)
        self.npeaks_spin.setValue(1)
        self.npeaks_spin.setEnabled(False)        # auto on by default
        self.refresh_btn = QtWidgets.QPushButton("Reload")
        self.refresh_btn.setToolTip("Re-grab the currently selected frame's pattern")
        self.fit_btn = QtWidgets.QPushButton("Fit")
        self.fit_btn.setObjectName("peakFitGo")
        controls.addWidget(QtWidgets.QLabel("Model"))
        controls.addWidget(self.model_combo)
        controls.addWidget(QtWidgets.QLabel("Background"))
        controls.addWidget(self.bkg_combo)
        controls.addWidget(self.auto_check)
        controls.addWidget(QtWidgets.QLabel("Peaks"))
        controls.addWidget(self.npeaks_spin)
        controls.addStretch(1)
        controls.addWidget(self.refresh_btn)
        controls.addWidget(self.fit_btn)
        lay.addLayout(controls)

        # Row 2: fit range (current x-unit) — synced with a draggable region
        range_row = QtWidgets.QHBoxLayout()
        range_row.setSpacing(7)
        self.range_lo = QtWidgets.QLineEdit()
        self.range_hi = QtWidgets.QLineEdit()
        for e in (self.range_lo, self.range_hi):
            e.setValidator(QtGui.QDoubleValidator(self))
            e.setMaximumWidth(90)
            e.setPlaceholderText("—")
        self.full_btn = QtWidgets.QPushButton("Full")
        self.full_btn.setToolTip("Reset the fit range to the whole pattern")
        range_row.addWidget(QtWidgets.QLabel("Range"))
        range_row.addWidget(self.range_lo)
        range_row.addWidget(QtWidgets.QLabel("to"))
        range_row.addWidget(self.range_hi)
        range_row.addWidget(self.full_btn)
        range_row.addStretch(1)
        lay.addLayout(range_row)

        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("peakFitStatus")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

        # Plots: data+fit on top (with a draggable fit-range region), residual below
        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(240)
        self.plot.addLegend(offset=(-10, 10))
        self.region = pg.LinearRegionItem(brush=(189, 147, 249, 30))
        self.region.setZValue(-10)
        self.plot.addItem(self.region)
        self.region.hide()
        self.resid_plot = pg.PlotWidget()
        self.resid_plot.setMaximumHeight(120)
        self.resid_plot.setXLink(self.plot)
        self.resid_plot.addLine(y=0, pen=pg.mkPen((130, 130, 140), width=1))
        lay.addWidget(self.plot, 3)
        lay.addWidget(self.resid_plot, 1)

        # Results table
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["#", "Center", "Center ±", "FWHM", "Amplitude"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setMaximumHeight(160)
        self.table.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(self.table)

        self.fit_btn.clicked.connect(self._do_fit)
        self.refresh_btn.clicked.connect(self.refresh_pattern)
        self.full_btn.clicked.connect(self._on_full)
        self.auto_check.toggled.connect(
            lambda on: self.npeaks_spin.setEnabled(not on))
        self.region.sigRegionChanged.connect(self._region_to_fields)
        self.range_lo.editingFinished.connect(self._fields_to_region)
        self.range_hi.editingFinished.connect(self._fields_to_region)

    # ---- range sync -----------------------------------------------------
    def _data_extent(self):
        if self._x is None or self._x.size == 0:
            return (0.0, 1.0)
        return (float(np.nanmin(self._x)), float(np.nanmax(self._x)))

    def _region_to_fields(self):
        if self._sync_guard:
            return
        lo, hi = sorted(self.region.getRegion())
        self._fit_lo, self._fit_hi = lo, hi
        self._sync_guard = True
        try:
            self.range_lo.setText(f"{lo:.4g}")
            self.range_hi.setText(f"{hi:.4g}")
        finally:
            self._sync_guard = False

    def _fields_to_region(self):
        if self._sync_guard:
            return
        ext = self._data_extent()
        try:
            lo = float(self.range_lo.text()) if self.range_lo.text() else ext[0]
            hi = float(self.range_hi.text()) if self.range_hi.text() else ext[1]
        except ValueError:
            return
        lo, hi = sorted((lo, hi))
        self._fit_lo, self._fit_hi = lo, hi
        self._sync_guard = True
        try:
            self.region.setRegion((lo, hi))
        finally:
            self._sync_guard = False

    def _on_full(self):
        self._fit_lo = self._fit_hi = None
        lo, hi = self._data_extent()
        self._sync_guard = True
        try:
            self.region.setRegion((lo, hi))
            self.range_lo.setText("")
            self.range_hi.setText("")
        finally:
            self._sync_guard = False

    def _fit_range(self):
        """The (lo, hi) fit window, clamped to the current data extent."""
        lo_ext, hi_ext = self._data_extent()
        lo = lo_ext if self._fit_lo is None else max(lo_ext, self._fit_lo)
        hi = hi_ext if self._fit_hi is None else min(hi_ext, self._fit_hi)
        if hi <= lo:
            lo, hi = lo_ext, hi_ext
        return lo, hi

    # ---- data + fit -----------------------------------------------------
    def refresh_pattern(self):
        """Re-grab the active frame's pattern, draw the raw data, reset the fit."""
        self._clear_fit()
        data = None
        try:
            data = self._provider()
        except Exception:
            logger.exception("peak-fit: failed to read the current pattern")
        if not data:
            self._x = self._y = None
            self.status.setText("No frame selected — pick a frame, then Reload.")
            self.plot.clear()
            self.plot.addItem(self.region)
            self.region.hide()
            return
        x, y, x_label = data
        self._x = np.asarray(x, dtype=float)
        self._y = np.asarray(y, dtype=float)
        self._x_label = x_label or "q"
        self.plot.clear()
        self.plot.addItem(self.region)
        self.plot.setLabel("bottom", self._x_label)
        self.plot.setLabel("left", "Intensity")
        self.plot.plot(self._x, self._y, pen=pg.mkPen((210, 210, 220), width=1),
                       name="data")
        # Region bounds = data extent; keep the chosen window across reloads.
        lo_ext, hi_ext = self._data_extent()
        self.region.setBounds([lo_ext, hi_ext])
        self._sync_guard = True
        try:
            self.region.setRegion(self._fit_range())
        finally:
            self._sync_guard = False
        self.region.show()
        self._region_to_fields()
        n = int(np.sum(np.isfinite(self._y)))
        self.status.setText(f"Loaded {n} points. Set the model and click Fit.")

    def _clear_fit(self):
        self.resid_plot.clear()
        self.resid_plot.addLine(y=0, pen=pg.mkPen((130, 130, 140), width=1))
        self.table.setRowCount(0)

    def _detect_peaks(self, x, y):
        """Auto-detect peak centers in (x, y) via scipy.signal.find_peaks.

        Returns up to ``_MAX_PEAKS`` most-prominent centers (sorted by x)."""
        from scipy.signal import find_peaks
        rng = float(np.nanmax(y) - np.nanmin(y))
        if not np.isfinite(rng) or rng <= 0:
            return []
        idx, props = find_peaks(
            y, prominence=0.04 * rng, distance=max(1, y.size // 80))
        if idx.size == 0:
            return []
        proms = props.get("prominences", np.ones(idx.size))
        if idx.size > _MAX_PEAKS:                 # keep the strongest
            keep = np.argsort(proms)[-_MAX_PEAKS:]
            idx = idx[keep]
        return sorted(float(x[i]) for i in idx)

    def _do_fit(self):
        if self._x is None or self._y is None:
            self.refresh_pattern()
            if self._x is None:
                return
        try:
            from xrd_tools.analysis.fitting import fit_peaks
        except Exception:
            self.status.setText(
                "Peak fitting needs lmfit — install it with "
                "`pip install \"xrd-tools[fitting]\"`, then reopen.")
            return

        model = _MODELS[self.model_combo.currentIndex()][1]
        background = _BACKGROUNDS[self.bkg_combo.currentIndex()][1]
        # Restrict to the fit range AND the finite samples (NaN-masked detector
        # gaps break lmfit).
        lo, hi = self._fit_range()
        finite = (np.isfinite(self._x) & np.isfinite(self._y)
                  & (self._x >= lo) & (self._x <= hi))
        x = self._x[finite]
        y = self._y[finite]

        positions = None
        if self.auto_check.isChecked():
            positions = self._detect_peaks(x, y)
            if not positions:
                self.status.setText(
                    "No peaks auto-detected in this range — narrow the range or "
                    "uncheck Auto and set a count.")
                return
            n_peaks = len(positions)
            self.npeaks_spin.blockSignals(True)
            self.npeaks_spin.setValue(min(n_peaks, _MAX_PEAKS))
            self.npeaks_spin.blockSignals(False)
        else:
            n_peaks = self.npeaks_spin.value()

        if x.size < max(5, 3 * n_peaks):
            self.status.setText("Not enough finite points in the range to fit.")
            return
        try:
            result = fit_peaks(x, y, positions=positions, model=model,
                               n_peaks=n_peaks, background=background)
        except Exception as exc:
            logger.exception("peak-fit: fit_peaks failed")
            self.status.setText(f"Fit failed: {exc}")
            return

        self._draw_result(x, y, result, auto=self.auto_check.isChecked())

    def _draw_result(self, x, y, result, auto=False):
        # redraw data over the FULL pattern (context), then fit over the range
        self.plot.clear()
        self.plot.addItem(self.region)
        self.plot.setLabel("bottom", self._x_label)
        self.plot.setLabel("left", "Intensity")
        self.plot.plot(self._x, self._y, pen=pg.mkPen((210, 210, 220), width=1),
                       name="data")
        best = np.asarray(result.best_fit, dtype=float)
        self.plot.plot(x, best, pen=pg.mkPen((189, 147, 249), width=2), name="fit")
        try:
            comps = result.fit_result.eval_components(x=x)
            bg = sum(v for k, v in comps.items() if str(k).startswith("bg"))
            if np.ndim(bg) == 1:
                self.plot.plot(x, np.asarray(bg, dtype=float),
                               pen=pg.mkPen((130, 200, 160), width=1,
                                            style=QtCore.Qt.PenStyle.DashLine),
                               name="background")
        except Exception:
            logger.debug("peak-fit: no background component to draw", exc_info=True)

        self._clear_fit()
        try:
            self.resid_plot.plot(x, y - best, pen=pg.mkPen((230, 133, 151), width=1))
            self.resid_plot.setLabel("left", "resid")
        except Exception:
            logger.debug("peak-fit: residual draw failed", exc_info=True)

        centers = list(result.peak_centers or [])
        cerrs = list(getattr(result, "peak_centers_err", []) or [])
        sigmas = list(result.peak_sigmas or [])
        amps = list(result.peak_amplitudes or [])
        params = getattr(result, "params", None)
        self.table.setRowCount(len(centers))
        for i, c in enumerate(centers):
            fwhm = None
            if params is not None:
                p = params.get(f"p{i}_fwhm")
                if p is not None:
                    fwhm = p.value
            if fwhm is None and i < len(sigmas):
                fwhm = 2.3548 * sigmas[i]
            cerr = cerrs[i] if i < len(cerrs) and cerrs[i] is not None else float("nan")
            amp = amps[i] if i < len(amps) else float("nan")
            vals = [str(i + 1), f"{c:.5g}", f"{cerr:.2g}",
                    f"{fwhm:.4g}" if fwhm is not None else "—", f"{amp:.4g}"]
            for col, v in enumerate(vals):
                self.table.setItem(i, col, QtWidgets.QTableWidgetItem(v))

        ok = bool(getattr(result, "success", True))
        how = "auto-detected" if auto else "fixed-count"
        self.status.setText(
            ("Fit converged." if ok else "Fit did NOT converge (best effort).")
            + f"  {len(centers)} {how} peak(s), "
            + f"{result.model_name} + {result.background_name}.")
