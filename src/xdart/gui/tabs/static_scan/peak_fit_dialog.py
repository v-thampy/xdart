# -*- coding: utf-8 -*-
"""Self-contained Peak Fitting popup (Direction-A Tools, first increment).

Opens from the bottom-left Tools card.  Grabs the currently displayed 1-D
integrated pattern, fits it with the headless ``xrd_tools.analysis.fitting``
API, and shows data + fit + residual in its own pyqtgraph plots plus a results
table — all isolated from the main display (no change to the main plot render
path).  Non-modal so the user can change the selected frame and re-fit.

The fitting backend is headless/Qt-free; this module is the thin Qt front-end.
``lmfit`` (the ``xrd-tools[fitting]`` extra) is imported lazily so xdart still
launches without it — the dialog shows a friendly install hint instead.
"""

import logging

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

logger = logging.getLogger(__name__)

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
        :meth:`refresh_pattern` (dialog open) and at each Fit so the dialog
        always fits what the user is looking at.
    """

    def __init__(self, pattern_provider, parent=None):
        super().__init__(parent)
        self._provider = pattern_provider
        self._x = None
        self._y = None
        self._x_label = "q"
        self.setObjectName("peakFitDialog")
        self.setWindowTitle("Peak Fitting")
        self.resize(560, 620)
        self._build_ui()

    # ---- UI construction ------------------------------------------------
    def _build_ui(self):
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(11, 11, 11, 11)
        lay.setSpacing(8)

        # Controls row
        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(7)
        self.model_combo = QtWidgets.QComboBox()
        for label, _ in _MODELS:
            self.model_combo.addItem(label)
        self.bkg_combo = QtWidgets.QComboBox()
        for label, _ in _BACKGROUNDS:
            self.bkg_combo.addItem(label)
        self.npeaks_spin = QtWidgets.QSpinBox()
        self.npeaks_spin.setRange(1, 12)
        self.npeaks_spin.setValue(1)
        self.fit_btn = QtWidgets.QPushButton("Fit")
        self.fit_btn.setObjectName("peakFitGo")
        self.refresh_btn = QtWidgets.QPushButton("Reload")
        self.refresh_btn.setToolTip("Re-grab the currently selected frame's pattern")
        controls.addWidget(QtWidgets.QLabel("Model"))
        controls.addWidget(self.model_combo)
        controls.addWidget(QtWidgets.QLabel("Background"))
        controls.addWidget(self.bkg_combo)
        controls.addWidget(QtWidgets.QLabel("Peaks"))
        controls.addWidget(self.npeaks_spin)
        controls.addStretch(1)
        controls.addWidget(self.refresh_btn)
        controls.addWidget(self.fit_btn)
        lay.addLayout(controls)

        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("peakFitStatus")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

        # Plots: data+fit on top, residual below (shared x).
        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(240)
        self.plot.addLegend(offset=(-10, 10))
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
        hh = self.table.horizontalHeader()
        hh.setStretchLastSection(True)
        lay.addWidget(self.table)

        self.fit_btn.clicked.connect(self._do_fit)
        self.refresh_btn.clicked.connect(self.refresh_pattern)

    # ---- data + fit -----------------------------------------------------
    def refresh_pattern(self):
        """Re-grab the active frame's pattern, draw the raw data, clear any fit."""
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
            return
        x, y, x_label = data
        self._x = np.asarray(x, dtype=float)
        self._y = np.asarray(y, dtype=float)
        self._x_label = x_label or "q"
        self.plot.clear()
        self.plot.setLabel("bottom", self._x_label)
        self.plot.setLabel("left", "Intensity")
        self.plot.plot(self._x, self._y, pen=pg.mkPen((210, 210, 220), width=1),
                       name="data")
        n = int(np.sum(np.isfinite(self._y)))
        self.status.setText(f"Loaded {n} points. Set the model and click Fit.")

    def _clear_fit(self):
        self.resid_plot.clear()
        self.resid_plot.addLine(y=0, pen=pg.mkPen((130, 130, 140), width=1))
        self.table.setRowCount(0)

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
        n_peaks = self.npeaks_spin.value()
        # Fit only the finite samples (NaN-masked detector gaps break lmfit).
        finite = np.isfinite(self._x) & np.isfinite(self._y)
        x = self._x[finite]
        y = self._y[finite]
        if x.size < max(5, 3 * n_peaks):
            self.status.setText("Not enough finite points to fit.")
            return
        try:
            result = fit_peaks(x, y, model=model, n_peaks=n_peaks,
                               background=background)
        except Exception as exc:
            logger.exception("peak-fit: fit_peaks failed")
            self.status.setText(f"Fit failed: {exc}")
            return

        self._draw_result(x, y, result)

    def _draw_result(self, x, y, result):
        # redraw data (so legend + fit layer cleanly), then fit + background
        self.plot.clear()
        self.plot.setLabel("bottom", self._x_label)
        self.plot.setLabel("left", "Intensity")
        self.plot.plot(x, y, pen=pg.mkPen((210, 210, 220), width=1), name="data")
        best = np.asarray(result.best_fit, dtype=float)
        self.plot.plot(x, best, pen=pg.mkPen((189, 147, 249), width=2),
                       name="fit")
        # Background component, if the model exposes one.
        try:
            comps = result.fit_result.eval_components(x=x)
            bg = sum(v for k, v in comps.items() if str(k).startswith("bg"))
            if np.ndim(bg) == 1:
                self.plot.plot(x, np.asarray(bg, dtype=float),
                               pen=pg.mkPen((130, 200, 160), width=1,
                                            style=QtCore.Qt.PenStyle.DashLine),
                               name="background")
        except Exception:
            logger.debug("peak-fit: no background component to draw",
                         exc_info=True)

        # Residual
        self._clear_fit()
        try:
            resid = y - best
            self.resid_plot.plot(x, resid, pen=pg.mkPen((230, 133, 151), width=1))
            self.resid_plot.setLabel("left", "resid")
        except Exception:
            logger.debug("peak-fit: residual draw failed", exc_info=True)

        # Results table
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
                fwhm = 2.3548 * sigmas[i]   # Gaussian approx fallback
            cerr = cerrs[i] if i < len(cerrs) and cerrs[i] is not None else float("nan")
            amp = amps[i] if i < len(amps) else float("nan")
            vals = [str(i + 1), f"{c:.5g}", f"{cerr:.2g}",
                    f"{fwhm:.4g}" if fwhm is not None else "—",
                    f"{amp:.4g}"]
            for col, v in enumerate(vals):
                self.table.setItem(i, col, QtWidgets.QTableWidgetItem(v))

        ok = bool(getattr(result, "success", True))
        self.status.setText(
            ("Fit converged." if ok else "Fit did NOT converge (showing best effort).")
            + f"  {len(centers)} peak(s), {result.model_name} + {result.background_name}.")
