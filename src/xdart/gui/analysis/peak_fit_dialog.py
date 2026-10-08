# -*- coding: utf-8 -*-
"""Self-contained Peak Fitting popup (Direction-A Tools).

Opens from the bottom-left Tools card. Fits the currently displayed 1-D integrated pattern over a user-chosen range
(auto-detecting peaks by default) and shows data, fit and residual, a parameter trend and a results table, isolated
from the main display. Non-modal so the user can change the selected frame and re-fit.

It builds no display or editor widget of its own (hard rule 14; FIT-TOOLS-2026-10 phase 0): its rows are the shared
Controls rows (``FormRow``, ``RangeRow``) in a ``SectionCard``, its rarely changed values the shared
``AdvancedDialog``, its plots ``FitView`` (whose fit range is the dragged range), its trend ``TrendView`` and its table
``ResultsTable``. Int's page drives it through ``vnext_submit`` (its analysis slot runs the headless displayed-fit
operation); without it, the dialog's own path fits through the headless analyzer. ``lmfit`` (the ``xrd-tools[fitting]``
extra) is imported lazily so xdart still launches without it.
"""

import logging

from xrd_tools.help_topics import diagnostic_text

import numpy as np
from dataclasses import replace
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from xdart.gui.tabs.scattering.advanced_editor import AdvancedDialog
from xdart.gui.tabs.scattering.fit_view import FitPanel, FitProjection, FitSeries, FitView, fit_result_series
from xdart.gui.tabs.scattering.results_table import (
    PEAK_OUTCOME_HEADERS, ResultsTable, peak_outcome_rows, peak_parameter_rows,
)
from xdart.gui.tabs.scattering.trend_view import TrendView
from xdart.gui.widgets.controls_panel import FormRow, RangeRow, SectionCard

logger = logging.getLogger(__name__)

_MAX_PEAKS = 12

# Auto-detect prominence floor: a candidate must rise at least this many
# robust noise-sigmas above the local floor to count as a peak.  Keeps
# peakless/diffuse scatter (e.g. a grazing-incidence cut with no rings in
# range) from latching onto noise spikes and grinding a doomed 12-component
# fit — instead it auto-detects nothing and says so immediately.
_NOISE_SIGMA = 5.0

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


def _path(name):
    return ("peak_fit", name)


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

    def __init__(self, pattern_provider=None, parent=None, *, analysis_context=None,
                 vnext_submit=None, allow_batch=False):
        super().__init__(parent)
        self._vnext = callable(vnext_submit)
        self._vnext_submit = vnext_submit
        self._allow_batch = bool(allow_batch)
        self._export_trace = None
        self._export_snapshot_provider = None
        self._provider = (
            analysis_context.current_pattern_tuple
            if analysis_context is not None else pattern_provider)
        if self._provider is None:
            self._provider = lambda: None
        self._x = None
        self._y = None
        self._x_label = "q"
        # Persisted fit range (in x-units); None => whole pattern.  Survives
        # Reload so stepping through frames keeps the chosen window.
        self._fit_lo = None
        self._fit_hi = None
        self.setObjectName("peakFitDialog")
        self.setWindowTitle("Peak Fitting")
        self.resize(560, 860)
        self._build_ui()
        if self._vnext:
            self.live_check.setChecked(False); self.live_check.setEnabled(False)
            self.live_check.setToolTip(diagnostic_text("P3_7_LIVE_FITTING_UNAVAILABLE"))
            self.batch_btn.setEnabled(bool(allow_batch))
            if not allow_batch:
                self.batch_btn.setToolTip(diagnostic_text("P3_7_BATCH_DISPLAY_PROJECTION_UNAVAILABLE"))
        from .fit_export import FitExportActions
        self.export_actions = FitExportActions(self, tool='peak', snapshot=self.export_snapshot,
                                               settings=self.vnext_fit_values)

    # ---- UI construction ------------------------------------------------
    def _build_ui(self):
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(11, 11, 11, 11)
        lay.setSpacing(8)

        # The fit's rows: model, background, peaks (Auto or a count) and the fit range, on the shared rows.
        card = SectionCard("Peak fit", collapsible=False)
        models, backgrounds = [label for label, _ in _MODELS], [label for label, _ in _BACKGROUNDS]
        self.model_row = FormRow(label="Model", path=_path("model"), value=models[0], kind="combo", choices=models)
        self.bkg_row = FormRow(label="Background", path=_path("background"), value=backgrounds[0], kind="combo",
                               choices=backgrounds)
        self.auto_row = FormRow(label="Auto", path=_path("auto"), value=True, kind="bool")
        self.count_row = FormRow(label="Peaks", path=_path("n_peaks"), value="1", kind="int", minimum=1,
                                 maximum=_MAX_PEAKS)
        self.range_row = RangeRow(label="Range", low={"path": _path("range_low"), "value": "", "placeholder": "—"},
                                  high={"path": _path("range_high"), "value": "", "placeholder": "—"})
        self.model_combo, self.bkg_combo = self.model_row.editor, self.bkg_row.editor
        self.auto_check, self.npeaks_spin = self.auto_row.editor, self.count_row.editor
        self.range_lo, self.range_hi = self.range_row.editors
        self.auto_check.setToolTip("Detect peaks automatically (scipy find_peaks) over the fit range")
        self.npeaks_spin.setEnabled(False)        # auto on by default
        for edit in (self.range_lo, self.range_hi):
            edit.setValidator(QtGui.QDoubleValidator(self))
        self.full_btn = QtWidgets.QPushButton("Full")
        self.full_btn.setToolTip("Reset the fit range to the whole pattern")
        self.range_row.add_trailing_widget(self.full_btn)
        for row in (self.model_row, self.bkg_row, self.auto_row, self.count_row, self.range_row):
            card.add_row(row)
        lay.addWidget(card)

        # Actions: Advanced, Reload, Live, Fit and Batch.
        actions = QtWidgets.QHBoxLayout()
        actions.setSpacing(7)
        self.advanced_btn = QtWidgets.QPushButton("Advanced…")
        self.advanced_btn.setObjectName("peakFitAdvancedToggle")
        self.refresh_btn = QtWidgets.QPushButton("Reload")
        self.refresh_btn.setToolTip("Re-grab the currently selected frame's pattern")
        self.live_row = FormRow(label="Live", path=_path("live"), value=False, kind="bool")
        self.live_check = self.live_row.editor
        self.live_check.setToolTip(
            "Re-fit automatically as new frames arrive during a scan "
            "(latest-wins — the fit tracks the newest frame).")
        self.fit_btn = QtWidgets.QPushButton("Fit")
        self.fit_btn.setObjectName("peakFitGo")
        self.batch_btn = QtWidgets.QPushButton("Batch")
        self.batch_btn.setToolTip(
            "Fit every frame in the scan with these settings, then plot the "
            "parameters vs frame number.")
        actions.addWidget(self.advanced_btn)
        actions.addStretch(1)
        actions.addWidget(self.refresh_btn)
        actions.addWidget(self.live_row)
        actions.addWidget(self.fit_btn)
        actions.addWidget(self.batch_btn)
        lay.addLayout(actions)

        self._build_advanced()

        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("peakFitStatus")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

        # Data and fit above (the fit range dragged on it), the residual below: the shared fit display.
        self.fit_view = FitView(self, name="peakFitView", summary=False, stretch=(3, 1), upper_min_height=240,
                                lower_max_height=120, link_x=True)
        self.plot, self.resid_plot = self.fit_view.plot, self.fit_view.residuals
        self.fit_view.reconcile(FitProjection(lower=FitPanel(zero_line=True)))
        lay.addWidget(self.fit_view, 4)

        # Parameters vs frame (live or batch), on the shared trend; its x is the frame, not q.
        self.trend = TrendView(self, name="peakFitTrend")
        self.trend.messageChanged.connect(self.status.setText)
        lay.addWidget(self.trend, 2)

        self.table = ResultsTable(PEAK_OUTCOME_HEADERS, self, name="peakFitTable", max_height=130)
        lay.addWidget(self.table)

        self.fit_btn.clicked.connect(self._do_fit)
        self.refresh_btn.clicked.connect(self.refresh_pattern)
        self.full_btn.clicked.connect(self._on_full)
        self.advanced_btn.clicked.connect(self._show_advanced)
        self.auto_check.toggled.connect(lambda on: self.npeaks_spin.setEnabled(not on))
        self.fit_view.rangeChanged.connect(self._region_to_fields)
        self.range_row.valueChanged.connect(lambda _path, _value: self._fields_to_region())

    def _build_advanced(self):
        """The rarely changed fit values, on the shared Advanced dialog's rows: they apply identically to Fit, Live
        and Batch (all backed by fit_peaks / PeakFitPlan)."""
        self.advanced = AdvancedDialog(self, title="Advanced — Peak Fitting", object_name="peakFitAdvancedDialog")
        rows = {
            "centers": FormRow(label="Peak centers", path=_path("centers"), value=""),
            "sigma_init": FormRow(label="σ initial", path=_path("sigma_init"), value=""),
            "sigma_bounds": RangeRow(label="σ min / max",
                                     low={"path": _path("sigma_min"), "value": "", "placeholder": "auto"},
                                     high={"path": _path("sigma_max"), "value": "", "placeholder": "auto"}),
            "center_delta": FormRow(label="Center ± delta", path=_path("center_delta"), value=""),
            "fraction": FormRow(label="PV fraction", path=_path("fraction"), value=""),
            "max_nfev": FormRow(label="Max iterations", path=_path("max_nfev"), value="", kind="int", minimum=0,
                                maximum=1_000_000, allow_blank=True),
        }
        self.advanced.add_group("Peaks", tuple(rows.values()))
        uncertainty = FormRow(label="Use uncertainties", path=_path("use_uncertainties"), value=False, kind="bool")
        self.advanced.add_group("Weighting", (uncertainty,))
        self.use_uncertainties = uncertainty.editor
        self.use_uncertainties.setEnabled(False)
        self.use_uncertainties.setToolTip("Use admitted measured sigma; no uncertainties are inferred")
        self._maxfev_row = rows["max_nfev"]
        self.adv_centers = rows["centers"].editor
        self.adv_sigma_init = rows["sigma_init"].editor
        self.adv_sigma_min, self.adv_sigma_max = rows["sigma_bounds"].editors
        self.adv_center_delta = rows["center_delta"].editor
        self.adv_fraction = rows["fraction"].editor
        self.adv_maxfev = rows["max_nfev"].editor
        self.adv_centers.setPlaceholderText("auto-detect — e.g. 2.1, 3.5, 4.8")
        self.adv_centers.setToolTip(
            "Manual peak centers (current x-unit), comma-separated.  When set, "
            "these seed the fit and override Auto-detect — for shoulders / weak "
            "peaks find_peaks misses.")
        for edit in (self.adv_sigma_init, self.adv_sigma_min, self.adv_sigma_max, self.adv_center_delta,
                     self.adv_fraction):
            edit.setValidator(QtGui.QDoubleValidator(self))
        for edit in (self.adv_sigma_init, self.adv_center_delta):
            edit.setPlaceholderText("auto")
        self.adv_sigma_init.setToolTip("Initial peak width σ (blank = auto)")
        self.adv_center_delta.setToolTip("Constrain each center to ± this much around its start (blank = free)")
        self.adv_fraction.setPlaceholderText("0.5")
        self.adv_fraction.setToolTip("Pseudo-Voigt Gaussian/Lorentzian mix, 0–1 (Pseudo-Voigt only)")
        self.adv_maxfev.setPlaceholderText("auto")
        self.adv_maxfev.setToolTip("Max solver iterations (max_nfev); blank or 0 = lmfit default")

    def _show_advanced(self):
        self.advanced.show(); self.advanced.raise_(); self.advanced.activateWindow()

    def _refusal(self):
        """The first whole-number row whose text is refused, named as the rows name it ("" when none)."""
        for row in (self.count_row, self._maxfev_row):
            if row.refusal():
                return f"{row.label.text()}: {row.refusal()}"
        return ""

    # ---- range sync -----------------------------------------------------
    def _data_extent(self):
        if self._x is None or self._x.size == 0:
            return (0.0, 1.0)
        return (float(np.nanmin(self._x)), float(np.nanmax(self._x)))

    def _region_to_fields(self, lo, hi):
        """The fit range dragged on the plot: kept, and written into the Range row."""
        self._fit_lo, self._fit_hi = lo, hi
        self.range_lo.setText(f"{lo:.4g}")
        self.range_hi.setText(f"{hi:.4g}")

    def _fields_to_region(self):
        """The Range row's values: kept, and shown as the plot's fit range."""
        ext = self._data_extent()
        try:
            lo = float(self.range_lo.text()) if self.range_lo.text() else ext[0]
            hi = float(self.range_hi.text()) if self.range_hi.text() else ext[1]
        except ValueError:
            return
        lo, hi = sorted((lo, hi))
        self._fit_lo, self._fit_hi = lo, hi
        if self._x is not None:
            self.fit_view.set_fit_range((lo, hi), limits=ext)

    def _on_full(self):
        self._fit_lo = self._fit_hi = None
        if self._x is not None:
            self.fit_view.set_fit_range(self._data_extent(), limits=self._data_extent())
        self.range_lo.setText("")
        self.range_hi.setText("")

    def batch_x_range(self):
        """The (lo, hi) x-window to clip each frame to in a batch run — for the
        Peak Fitter this is the chosen fit range (the shared batch contract; the
        Phase Fitter returns the full extent)."""
        return self._fit_range()

    def _fit_range(self):
        """The (lo, hi) fit window, clamped to the current data extent."""
        lo_ext, hi_ext = self._data_extent()
        lo = lo_ext if self._fit_lo is None else max(lo_ext, self._fit_lo)
        hi = hi_ext if self._fit_hi is None else min(hi_ext, self._fit_hi)
        if hi <= lo:
            lo, hi = lo_ext, hi_ext
        return lo, hi

    # ---- data + fit -----------------------------------------------------
    def keyPressEvent(self, event):
        # Esc must not discard the fit setup (model/peaks/range/results); the
        # default QDialog Esc -> reject closes the popup.  Match the metadata
        # plotter.  Other keys pass through.
        if event.key() == QtCore.Qt.Key.Key_Escape:
            event.accept()
            return
        super().keyPressEvent(event)

    def refresh_pattern(self):
        """Re-grab the active frame's pattern, draw the raw data, reset the fit."""
        if self._vnext:
            self._vnext_submit("reload", None)
            return
        self._clear_fit()
        # Reload starts a fresh vs-frame trend — but NOT while a live run is
        # accumulating (don't discard the user's live trend on a manual Reload).
        if not self.live_check.isChecked():
            self.reset_param_trend()
        data = None
        try:
            data = self._provider()
        except Exception:
            logger.exception("peak-fit: failed to read the current pattern")
        if not data:
            self._x = self._y = None
            self.status.setText("No frame selected — pick a frame, then Reload.")
            self.fit_view.reconcile(replace(self.fit_view.projection, upper=FitPanel()))
            self.fit_view.set_fit_range(None)
            return
        x, y, x_label = data
        self._show_pattern(x, y, x_label)
        n = int(np.sum(np.isfinite(self._y)))
        self.status.setText(f"Loaded {n} points. Set the model and click Fit.")

    def set_live_pattern(self, x, y, x_label):
        """Show a pattern pushed by the live runner — the data appears at once;
        the fit overlay arrives later via :meth:`_draw_outcome` on the worker
        result. Called by the workspace live controller per frame."""
        self._show_pattern(x, y, x_label, clear_fit=True)

    def set_batch_running(self, running):
        """Reflect a batch run in flight: the Batch button becomes Cancel and
        the single-frame controls are disabled (the workspace owns the run)."""
        self.batch_btn.setText("Cancel" if running else "Batch")
        self.fit_btn.setEnabled(not running)
        self.live_check.setEnabled(not running and not self._vnext)
        self.batch_btn.setEnabled(not self._vnext or self._allow_batch)

    def set_batch_progress(self, done, total):
        self.status.setText(f"Batch fitting… {done}/{total}")

    def _pattern_panel(self, *series, markers=()):
        """The upper panel: the pattern's points ("data", the 1-D plot's markers), then ``series`` (fit,
        background) and the peak ``markers``."""
        return FitPanel((FitSeries(self._x, self._y, "data", "data"), *series), self._x_label, "Intensity",
                        markers=markers)

    def _show_pattern(self, x, y, x_label, *, clear_fit=False):
        """Draw the raw pattern and the fit range (kept across reloads, clamped to the data); set _x / _y.
        ``clear_fit``: the last fit goes too (its residual and rows), in the same one redraw."""
        self._x = np.asarray(x, dtype=float)
        self._y = np.asarray(y, dtype=float)
        self._x_label = x_label or "q"
        self.trend.set_value_unit(self._x_label)
        lower = FitPanel(zero_line=True) if clear_fit else self.fit_view.projection.lower
        self.fit_view.reconcile(FitProjection(self._pattern_panel(), lower))
        if clear_fit:
            self.table.clear_rows()
        self.fit_view.set_fit_range(self._fit_range(), limits=self._data_extent())
        self._region_to_fields(*self.fit_view.fit_range)

    def _clear_fit(self):
        self.fit_view.reconcile(replace(self.fit_view.projection, lower=FitPanel(zero_line=True)))
        self.table.clear_rows()

    def _detect_peaks(self, x, y):
        """Auto-detect peak centers in (x, y) via scipy.signal.find_peaks.

        Returns up to ``_MAX_PEAKS`` most-prominent centers (sorted by x).

        The prominence floor is the LARGER of a fraction of the data range
        and a robust multiple of the noise level, so peakless/diffuse scatter
        auto-detects NOTHING (caller then shows the clean "no peaks" hint)
        rather than latching onto noise spikes and grinding a doomed fit.
        ``y`` is already range- and NaN-masked by the caller, so its indices
        align with ``x``."""
        from scipy.signal import find_peaks
        y = np.asarray(y, dtype=float)
        rng = float(np.nanmax(y) - np.nanmin(y))
        if not np.isfinite(rng) or rng <= 0:
            return []
        # Robust noise estimate: MAD of successive differences.  Real peaks
        # are sparse so they barely perturb it; diff doubles i.i.d. variance
        # (=> /sqrt(2)) and 1.4826 rescales MAD to a Gaussian sigma.
        d = np.diff(y)
        mad = float(np.median(np.abs(d - np.median(d)))) if d.size else 0.0
        noise = 1.4826 * mad / np.sqrt(2.0)
        prominence = max(0.04 * rng, _NOISE_SIGMA * noise)
        # Light boxcar smoothing collapses 1-2 sample noise spikes -- their
        # prominence drops below the noise floor -- while barely touching a
        # real, multi-sample peak.  Detect on the smoothed trace so peakless
        # scatter finds NOTHING (clean "no peaks" hint) instead of fitting a
        # spike as garbage or grinding a doomed 12-component fit.
        w = max(3, (y.size // 150) | 1)          # odd window, >= 3 samples
        ys = np.convolve(y, np.ones(w) / w, mode="same")
        # width >= 2 samples is a second guard against single-sample spikes:
        # a real diffraction peak is broader than one point, a noise outlier
        # is not.
        idx, props = find_peaks(
            ys, prominence=prominence, distance=max(1, y.size // 80), width=2)
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
        refusal = self._refusal()
        if refusal:
            self.status.setText(refusal)
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
        if nfev:
            out["fit_kwargs"] = {"max_nfev": int(nfev)}
        return out

    def vnext_fit_values(self):
        bounds = self._fit_range() if self._x is not None else None
        centers = tuple(self._manual_centers(*bounds) or ()) if bounds else ()
        smin, smax = self._line_float(self.adv_sigma_min), self._line_float(self.adv_sigma_max)
        fraction = self._line_float(self.adv_fraction)
        nfev = self.adv_maxfev.value()
        values = {
            "selection_mode": "auto" if self.auto_check.isChecked() else "count",
            "n_peaks": len(centers) or int(self.npeaks_spin.value()),
            "model": _MODELS[self.model_combo.currentIndex()][1],
            "background": _BACKGROUNDS[self.bkg_combo.currentIndex()][1],
            "fit_bounds": bounds, "manual_centers": centers,
            "sigma_init": self._line_float(self.adv_sigma_init),
            "sigma_bounds": ((smin, smax) if smin is not None and smax is not None else None),
            "center_bounds_delta": self._line_float(self.adv_center_delta),
            "fraction_init": 0.5 if fraction is None else fraction,
            "max_nfev": int(nfev) if nfev else None,
        }
        if self.use_uncertainties.isChecked():
            values['use_uncertainties'] = True
        return values

    def _do_fit(self):
        refusal = self._refusal()
        if refusal:
            self.status.setText(refusal)
            return
        if self._vnext:
            if self._x is None:  # nothing shown yet: the displayed trace first, as Reload, so its fit can be drawn
                self._vnext_submit("reload", None)
                if self._x is None:
                    return
            self._vnext_submit("fit", self.vnext_fit_values())
            return
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

    def set_vnext_trace(self, trace):
        self._export_trace = trace
        self.use_uncertainties.setEnabled(trace.sigma is not None)
        if trace.sigma is None:
            self.use_uncertainties.setChecked(False)
        self._show_pattern(trace.axis, trace.intensity, trace.axis_unit or trace.label, clear_fit=True)
        self.status.setText("Displayed trace captured. Click Fit.")

    def clear_input(self):
        """Release a detached displayed input when its host changes scope or closes."""
        self._x = self._y = self._export_trace = None
        self.fit_view.reconcile(FitProjection(FitPanel(), FitPanel(zero_line=True)))
        self.table.clear_rows()
        self.fit_btn.setEnabled(False)
        self.use_uncertainties.setChecked(False)
        self.use_uncertainties.setEnabled(False)

    def export_snapshot(self):
        if self._export_snapshot_provider is not None:
            return self._export_snapshot_provider()
        if self._export_trace is None:
            return None
        from .fit_export import snapshot_from_trace, plain_settings
        return snapshot_from_trace(self._export_trace, settings=plain_settings(self.vnext_fit_values()))

    def set_vnext_result(self, result):
        lo, hi = self._fit_range()
        mask = np.isfinite(self._x) & np.isfinite(self._y) & (self._x >= lo) & (self._x <= hi)
        upper, markers, lower = fit_result_series(self._x[mask], result)
        self.fit_view.reconcile(FitProjection(self._pattern_panel(*upper, markers=markers),
                                              FitPanel(lower, zero_line=True)))
        self.table.show_rows(*peak_parameter_rows(result))
        detail = "; ".join(result.diagnostics)
        self.status.setText(result.code if result.fit_success is None else
                            (result.message or result.code) + (f"; {detail}" if detail else ""))
        self.status.setToolTip("\n".join((result.plan_fingerprint,
            result.policy_fingerprint, result.result_fingerprint)))

    def closeEvent(self, event):
        if self._vnext:
            self._vnext_submit("close", None)
        super().closeEvent(event)

    def _draw_outcome(self, outcome, auto=False):
        # data over the FULL pattern (context); the analyzer's Overlay traces
        # (fit / background / residual) over the fitted range: the total fit solid, the background dashed
        overlay = outcome.overlay
        payload = outcome.result.payload
        ox = overlay.x
        upper = [FitSeries(ox, overlay.traces[name], name, name) for name in ("fit", "background")
                 if name in overlay.traces]
        residual = (FitSeries(ox, overlay.traces["residual"], "residual"),) if "residual" in overlay.traces else ()
        self.fit_view.reconcile(FitProjection(self._pattern_panel(*upper), FitPanel(
            residual, y_label="resid" if residual else "", zero_line=True)))
        headers, rows = peak_outcome_rows(payload)
        self.table.show_rows(headers, rows)

        ok = bool(getattr(payload, "success", True))
        how = "auto-detected" if auto else "fixed-count"
        self.status.setText(
            ("Fit converged." if ok else "Fit did NOT converge (best effort).")
            + f"  {len(rows)} {how} peak(s), "
            + f"{payload.model_name} + {payload.background_name}.")

        # Feed the vs-frame trend when this outcome is for a real frame (live /
        # batch label = frame index); manual Fit uses label="current" -> skipped.
        try:
            frame_idx = int(outcome.label)
        except (TypeError, ValueError):
            frame_idx = None
        if frame_idx is not None:
            self._accumulate_frame_params(frame_idx, outcome.params)

    # ---- the parameter trend ---------------------------------------------
    def reset_param_trend(self):
        """A fresh vs-frame trend (Reload, a new live or batch run)."""
        self.trend.clear()

    def _accumulate_frame_params(self, frame_idx, params):
        """One frame's parameters on the trend (a re-fit of the frame replaces its row)."""
        self.trend.set_row(str(int(frame_idx)), int(frame_idx), params)
