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
        # Parameter-vs-frame trend (analyzer framework Step: unified 3-row plot).
        # frame_index -> flat params dict; written by BOTH live (per fit) and
        # batch (per frame).  Keyed by index so live drops / out-of-order re-fits
        # update-not-duplicate.  Reset only on a fresh run, never per frame.
        self._param_accumulator = {}
        self._param_family_keys = ()      # cached families currently in the combo
        self.setObjectName("peakFitDialog")
        self.setWindowTitle("Peak Fitting")
        self.resize(560, 820)
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
        self.live_check = QtWidgets.QCheckBox("Live")
        self.live_check.setToolTip(
            "Re-fit automatically as new frames arrive during a scan "
            "(latest-wins — the fit tracks the newest frame).")
        self.fit_btn = QtWidgets.QPushButton("Fit")
        self.fit_btn.setObjectName("peakFitGo")
        self.batch_btn = QtWidgets.QPushButton("Batch")
        self.batch_btn.setToolTip(
            "Fit every frame in the scan with these settings, then plot the "
            "parameters vs frame number.")
        controls.addWidget(QtWidgets.QLabel("Model"))
        controls.addWidget(self.model_combo)
        controls.addWidget(QtWidgets.QLabel("Background"))
        controls.addWidget(self.bkg_combo)
        controls.addWidget(self.auto_check)
        controls.addWidget(QtWidgets.QLabel("Peaks"))
        controls.addWidget(self.npeaks_spin)
        controls.addStretch(1)
        controls.addWidget(self.refresh_btn)
        controls.addWidget(self.live_check)
        controls.addWidget(self.fit_btn)
        controls.addWidget(self.batch_btn)
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

        # Advanced options (collapsible) — all backed by fit_peaks / PeakFitPlan
        # today, so they apply identically to manual Fit, Live, and Batch.
        self.advanced_btn = QtWidgets.QPushButton("Advanced ▾")
        self.advanced_btn.setCheckable(True)
        self.advanced_btn.setObjectName("peakFitAdvancedToggle")
        adv_row = QtWidgets.QHBoxLayout()
        adv_row.addWidget(self.advanced_btn)
        adv_row.addStretch(1)
        lay.addLayout(adv_row)

        self.advanced_box = QtWidgets.QWidget()
        self.advanced_box.setVisible(False)
        adv = QtWidgets.QGridLayout(self.advanced_box)
        adv.setContentsMargins(2, 2, 2, 2)
        adv.setHorizontalSpacing(8)
        adv.setVerticalSpacing(5)

        def _num():
            e = QtWidgets.QLineEdit()
            e.setValidator(QtGui.QDoubleValidator(self))
            e.setMaximumWidth(90)
            e.setPlaceholderText("auto")
            return e

        self.adv_centers = QtWidgets.QLineEdit()
        self.adv_centers.setPlaceholderText("auto-detect — e.g. 2.1, 3.5, 4.8")
        self.adv_centers.setToolTip(
            "Manual peak centers (current x-unit), comma-separated.  When set, "
            "these seed the fit and override Auto-detect — for shoulders / weak "
            "peaks find_peaks misses.")
        self.adv_sigma_init = _num()
        self.adv_sigma_init.setToolTip("Initial peak width σ (blank = auto)")
        self.adv_sigma_min = _num()
        self.adv_sigma_max = _num()
        self.adv_center_delta = _num()
        self.adv_center_delta.setToolTip(
            "Constrain each center to ± this much around its start (blank = free)")
        self.adv_fraction = _num()
        self.adv_fraction.setPlaceholderText("0.5")
        self.adv_fraction.setToolTip(
            "Pseudo-Voigt Gaussian/Lorentzian mix, 0–1 (Pseudo-Voigt only)")
        self.adv_maxfev = QtWidgets.QSpinBox()
        self.adv_maxfev.setRange(0, 1000000)
        self.adv_maxfev.setSingleStep(500)
        self.adv_maxfev.setSpecialValueText("auto")
        self.adv_maxfev.setMaximumWidth(110)
        self.adv_maxfev.setToolTip("Max solver iterations (max_nfev); 0 = lmfit default")

        adv.addWidget(QtWidgets.QLabel("Peak centers"), 0, 0)
        adv.addWidget(self.adv_centers, 0, 1, 1, 3)
        adv.addWidget(QtWidgets.QLabel("σ initial"), 1, 0)
        adv.addWidget(self.adv_sigma_init, 1, 1)
        adv.addWidget(QtWidgets.QLabel("σ min / max"), 1, 2)
        sig_row = QtWidgets.QHBoxLayout()
        sig_row.setSpacing(4)
        sig_row.addWidget(self.adv_sigma_min)
        sig_row.addWidget(self.adv_sigma_max)
        adv.addLayout(sig_row, 1, 3)
        adv.addWidget(QtWidgets.QLabel("Center ± delta"), 2, 0)
        adv.addWidget(self.adv_center_delta, 2, 1)
        adv.addWidget(QtWidgets.QLabel("PV fraction"), 2, 2)
        adv.addWidget(self.adv_fraction, 2, 3)
        adv.addWidget(QtWidgets.QLabel("Max iterations"), 3, 0)
        adv.addWidget(self.adv_maxfev, 3, 1)
        adv.setColumnStretch(3, 1)
        lay.addWidget(self.advanced_box)

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

        # Row 3: parameter(s) vs frame number — fills as frames are fit (live or
        # batch).  A family combo picks which parameter; Overlay shows every peak
        # in that family.  X-axis is FRAME INDEX (not q), so it is NOT x-linked.
        track_row = QtWidgets.QHBoxLayout()
        track_row.setSpacing(7)
        track_row.addWidget(QtWidgets.QLabel("Track"))
        self.param_family_combo = QtWidgets.QComboBox()
        self.param_family_combo.setToolTip(
            "Which fitted parameter to plot vs frame number")
        self.param_family_combo.setEnabled(False)
        self.param_overlay_check = QtWidgets.QCheckBox("Overlay")
        self.param_overlay_check.setToolTip(
            "Plot every peak in the selected parameter (e.g. all centers) "
            "on one axis, not just the first")
        track_row.addWidget(self.param_family_combo)
        track_row.addWidget(self.param_overlay_check)
        track_row.addStretch(1)
        self.param_save_btn = QtWidgets.QPushButton("Save CSV…")
        self.param_save_btn.setToolTip("Export the parameters-vs-frame table")
        self.param_save_btn.setEnabled(False)
        track_row.addWidget(self.param_save_btn)
        lay.addLayout(track_row)

        self.param_plot = pg.PlotWidget()
        self.param_plot.setMinimumHeight(150)
        self.param_plot.addLegend(offset=(-10, 10))
        self.param_plot.setLabel("bottom", "Frame")
        lay.addWidget(self.param_plot, 2)

        # Results table
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["#", "Center", "Center ±", "FWHM", "Amplitude"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setMaximumHeight(130)
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
        self.param_family_combo.currentIndexChanged.connect(self._redraw_param_plot)
        self.param_overlay_check.toggled.connect(self._redraw_param_plot)
        self.param_save_btn.clicked.connect(self._save_param_csv)
        self.advanced_btn.toggled.connect(self._on_advanced_toggled)

    def _on_advanced_toggled(self, on):
        self.advanced_box.setVisible(on)
        self.advanced_btn.setText("Advanced ▴" if on else "Advanced ▾")

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
        self.reset_param_trend()      # Reload starts a fresh vs-frame trend
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
        self._show_pattern(x, y, x_label)
        n = int(np.sum(np.isfinite(self._y)))
        self.status.setText(f"Loaded {n} points. Set the model and click Fit.")

    def set_live_pattern(self, x, y, x_label):
        """Show a pattern pushed by the live runner — the data appears at once;
        the fit overlay arrives later via :meth:`_draw_outcome` on the worker
        result.  Called by staticWidget's live controller per frame."""
        self._clear_fit()
        self._show_pattern(x, y, x_label)

    def set_batch_running(self, running):
        """Reflect a batch run in flight: the Batch button becomes Cancel and
        the single-frame controls are disabled (staticWidget owns the run)."""
        self.batch_btn.setText("Cancel" if running else "Batch")
        self.fit_btn.setEnabled(not running)
        self.live_check.setEnabled(not running)

    def set_batch_progress(self, done, total):
        self.status.setText(f"Batch fitting… {done}/{total}")

    def _show_pattern(self, x, y, x_label):
        """Draw the raw pattern + the fit-range region and set self._x / self._y."""
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

    def build_fit_request(self):
        """Build ``(AnalysisInput, PeakFitAnalyzer)`` from the current pattern +
        controls, or ``None`` (with a status message) if it can't be fit.

        Shared by the manual Fit button (synchronous) AND the live worker
        (background), so live + manual fit identically — the dialog is just one
        consumer of the agnostic Analyzer contract."""
        if self._x is None or self._y is None:
            return None
        try:
            from xrd_tools.analysis.fitting import fit_peaks  # noqa: F401  lmfit probe
        except Exception:
            self.status.setText(
                "Peak fitting needs lmfit — install it with "
                "`pip install \"xrd-tools[fitting]\"`, then reopen.")
            return None
        from xrd_tools.analysis.plans import PeakFitPlan
        from xrd_tools.analysis.runner import AnalysisInput, PeakFitAnalyzer

        model = _MODELS[self.model_combo.currentIndex()][1]
        background = _BACKGROUNDS[self.bkg_combo.currentIndex()][1]
        # Restrict to the fit range AND the finite samples (NaN-masked detector
        # gaps break lmfit).
        lo, hi = self._fit_range()
        finite = (np.isfinite(self._x) & np.isfinite(self._y)
                  & (self._x >= lo) & (self._x <= hi))
        x = self._x[finite]
        y = self._y[finite]

        # Manual centers (Advanced) take precedence over Auto-detect.
        positions = self._manual_centers(lo, hi)
        if positions:
            n_peaks = len(positions)
            self.npeaks_spin.blockSignals(True)
            self.npeaks_spin.setValue(min(n_peaks, _MAX_PEAKS))
            self.npeaks_spin.blockSignals(False)
        elif self.auto_check.isChecked():
            positions = self._detect_peaks(x, y)
            if not positions:
                self.status.setText(
                    "No peaks auto-detected in this range — narrow the range, "
                    "uncheck Auto and set a count, or enter centers in Advanced.")
                return None
            n_peaks = len(positions)
            self.npeaks_spin.blockSignals(True)
            self.npeaks_spin.setValue(min(n_peaks, _MAX_PEAKS))
            self.npeaks_spin.blockSignals(False)
        else:
            n_peaks = self.npeaks_spin.value()

        if x.size < max(5, 3 * n_peaks):
            self.status.setText("Not enough finite points in the range to fit.")
            return None
        plan = PeakFitPlan(
            positions=tuple(positions) if positions else None,
            model=model, n_peaks=n_peaks, background=background,
            **self._advanced_kwargs())
        inp = AnalysisInput(label="current", x=x, y=y, x_unit=self._x_label)
        return inp, PeakFitAnalyzer(plan)

    # ---- advanced options -----------------------------------------------
    @staticmethod
    def _line_float(line_edit):
        t = line_edit.text().strip()
        if not t:
            return None
        try:
            return float(t)
        except ValueError:
            return None

    def _manual_centers(self, lo, hi):
        """Parsed in-range manual peak centers from Advanced, or None."""
        text = self.adv_centers.text().strip()
        if not text:
            return None
        vals = []
        for part in text.replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                vals.append(float(part))
            except ValueError:
                continue
        vals = [v for v in vals if lo <= v <= hi]
        return sorted(vals) or None

    def _advanced_kwargs(self):
        """The optional PeakFitPlan kwargs set in the Advanced box (only the
        ones the user filled in)."""
        out = {}
        si = self._line_float(self.adv_sigma_init)
        if si is not None:
            out["sigma_init"] = si
        smin = self._line_float(self.adv_sigma_min)
        smax = self._line_float(self.adv_sigma_max)
        if smin is not None and smax is not None and smax > smin:
            out["sigma_bounds"] = (smin, smax)
        delta = self._line_float(self.adv_center_delta)
        if delta is not None and delta > 0:
            out["center_bounds_delta"] = delta
        frac = self._line_float(self.adv_fraction)
        if frac is not None:
            out["fraction_init"] = min(max(frac, 0.0), 1.0)
        nfev = self.adv_maxfev.value()
        if nfev > 0:
            out["fit_kwargs"] = {"max_nfev": int(nfev)}
        return out

    def _do_fit(self):
        if self._x is None or self._y is None:
            self.refresh_pattern()
            if self._x is None:
                return
        req = self.build_fit_request()
        if req is None:
            return
        inp, analyzer = req
        outcome = analyzer.analyze(inp)
        if not outcome or not outcome.ok:
            self.status.setText(
                f"Fit failed: {outcome.message if outcome else 'no result'}")
            return
        self._draw_outcome(outcome, auto=self.auto_check.isChecked())

    def _draw_outcome(self, outcome, auto=False):
        # data over the FULL pattern (context); the analyzer's Overlay traces
        # (fit / background / residual) over the fitted range.
        overlay = outcome.overlay
        payload = outcome.result.payload
        self.plot.clear()
        self.plot.addItem(self.region)
        self.plot.setLabel("bottom", self._x_label)
        self.plot.setLabel("left", "Intensity")
        self.plot.plot(self._x, self._y, pen=pg.mkPen((210, 210, 220), width=1),
                       name="data")
        ox = overlay.x
        if "fit" in overlay.traces:
            self.plot.plot(ox, overlay.traces["fit"],
                           pen=pg.mkPen((189, 147, 249), width=2), name="fit")
        if "background" in overlay.traces:
            self.plot.plot(ox, overlay.traces["background"],
                           pen=pg.mkPen((130, 200, 160), width=1,
                                        style=QtCore.Qt.PenStyle.DashLine),
                           name="background")
        self._clear_fit()
        if "residual" in overlay.traces:
            self.resid_plot.plot(ox, overlay.traces["residual"],
                                 pen=pg.mkPen((230, 133, 151), width=1))
            self.resid_plot.setLabel("left", "resid")

        centers = list(payload.peak_centers or [])
        cerrs = list(getattr(payload, "peak_centers_err", []) or [])
        sigmas = list(payload.peak_sigmas or [])
        amps = list(payload.peak_amplitudes or [])
        params = getattr(payload, "params", None)
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

        ok = bool(getattr(payload, "success", True))
        how = "auto-detected" if auto else "fixed-count"
        self.status.setText(
            ("Fit converged." if ok else "Fit did NOT converge (best effort).")
            + f"  {len(centers)} {how} peak(s), "
            + f"{payload.model_name} + {payload.background_name}.")

        # Feed the vs-frame trend when this outcome is for a real frame (live /
        # batch label = frame index); manual Fit uses label="current" -> skipped.
        try:
            frame_idx = int(outcome.label)
        except (TypeError, ValueError):
            frame_idx = None
        if frame_idx is not None:
            self._accumulate_frame_params(frame_idx, outcome.params)

    # ---- parameter-vs-frame trend (row 3) -------------------------------
    def reset_param_trend(self):
        """Drop the accumulated vs-frame series + clear row 3.  Called at the
        start of a fresh run (Reload, live-enable, Batch) — never per frame."""
        self._param_accumulator = {}
        self._param_family_keys = ()
        self.param_family_combo.blockSignals(True)
        self.param_family_combo.clear()
        self.param_family_combo.blockSignals(False)
        self.param_family_combo.setEnabled(False)
        self.param_overlay_check.setEnabled(False)
        self.param_save_btn.setEnabled(False)
        self.param_plot.clear()

    def _accumulate_frame_params(self, frame_idx, params):
        """Store one frame's params (update-not-duplicate) and refresh row 3."""
        if not params:
            return
        self._param_accumulator[int(frame_idx)] = dict(params)
        self._sync_param_families()
        self._redraw_param_plot()
        has_data = bool(self._param_accumulator)
        self.param_family_combo.setEnabled(has_data)
        self.param_overlay_check.setEnabled(has_data)
        self.param_save_btn.setEnabled(has_data)

    def _sync_param_families(self):
        """Keep the family combo in sync with the families present so far,
        preserving the current selection."""
        from .peak_fit_util import FAMILY_LABELS, group_families
        keys = []
        for params in self._param_accumulator.values():
            for k in params:
                if k not in keys:
                    keys.append(k)
        families = tuple(group_families(keys))
        if families == self._param_family_keys:
            return
        current = self.param_family_combo.currentData()
        self.param_family_combo.blockSignals(True)
        self.param_family_combo.clear()
        for fam in families:
            self.param_family_combo.addItem(FAMILY_LABELS.get(fam, fam), fam)
        if current is not None:
            i = self.param_family_combo.findData(current)
            if i >= 0:
                self.param_family_combo.setCurrentIndex(i)
        self.param_family_combo.blockSignals(False)
        self._param_family_keys = families

    def _redraw_param_plot(self):
        import pyqtgraph as _pg
        from .peak_fit_util import (CURVE_PENS, FAMILY_LABELS, X_UNIT_FAMILIES,
                                    accumulator_to_table, group_families)
        self.param_plot.clear()
        if not self._param_accumulator:
            return
        labels, columns = accumulator_to_table(self._param_accumulator)
        x = np.asarray([float(l) for l in labels], dtype=float)
        fam = self.param_family_combo.currentData()
        if fam is None:
            return
        members = group_families(columns).get(fam, [])
        if not self.param_overlay_check.isChecked():
            members = members[:1]           # just the first peak
        for n, (key, _idx) in enumerate(members):
            y = np.asarray(columns[key], dtype=float)
            color = CURVE_PENS[n % len(CURVE_PENS)]
            self.param_plot.plot(x, y, pen=_pg.mkPen(color, width=2), name=key,
                                 symbol="o", symbolSize=5, symbolBrush=color)
        unit = self._x_label if fam in X_UNIT_FAMILIES else "Intensity"
        label = FAMILY_LABELS.get(fam, fam)
        self.param_plot.setLabel("left", f"{label} ({unit})" if unit else label)

    def _save_param_csv(self):
        from .peak_fit_util import accumulator_to_table
        if not self._param_accumulator:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save parameters vs frame", "fit_params_vs_frame.csv",
            "CSV files (*.csv)")
        if not path:
            return
        import csv
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
