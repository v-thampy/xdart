# -*- coding: utf-8 -*-
"""Self-contained Phase Fitting popup (Direction-A Tools).

Structure-informed multi-phase fitting of the active 1-D pattern: load one or more CIFs, set the fit options (profile /
background / lattice tolerance / texture — the notebook 03/04 knobs), and fit. Without Int's page it drives the
headless ``xrd_tools.analysis.runner.PhaseFitAnalyzer`` through the same contract the Peak Fitter uses, so Batch
reuses the shared worker and the parameter trend (phase fractions / lattice vs frame).

It builds no display or editor widget of its own (hard rule 14; FIT-TOOLS-2026-10 phase 0): its phases are the shared
Data Browser's keyed list (``keyed_list_profile``), reconciled after each Add or Remove; its options the shared
Controls rows in a ``SectionCard``; its rarely changed values the shared ``AdvancedDialog``; its plots ``FitView``,
its trend ``TrendView``, its table ``ResultsTable``; Browse the shared ``file_dialogs``.

Needs the ``xrd-tools[fitting]`` extra: ``lmfit`` (the fit) AND ``pymatgen`` (CIF parsing). Both are imported lazily
so xdart still launches without them — the dialog shows a friendly install hint instead.
"""

import logging
import os

from dataclasses import replace

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from xdart.gui.tabs.scattering.advanced_editor import AdvancedDialog
from xdart.gui.tabs.scattering.browser_view import BrowserResult, BrowserView, keyed_list_profile
from xdart.gui.tabs.scattering.fit_view import FitPanel, FitProjection, FitSeries, FitView, fit_result_series
from xdart.gui.tabs.scattering.results_table import (
    PHASE_OUTCOME_HEADERS, ResultsTable, phase_outcome_rows, phase_result_rows,
)
from xdart.gui.tabs.scattering.trend_view import TrendView
from xdart.gui.widgets.controls_panel import FormRow, SectionCard
from xdart.gui.widgets.file_dialogs import browse

logger = logging.getLogger(__name__)

#: A phase's crystal structure.
CIF_FILTER = "CIF files (*.cif);;All files (*)"

# (display label, PhaseFitter phase_profile)
_PROFILES = [
    ("Pseudo-Voigt", "pseudovoigt"),
    ("Gaussian", "gaussian"),
    ("Lorentzian", "lorentzian"),
    ("Voigt", "voigt"),
]
# (display label, prefit_background)
_BACKGROUNDS = [
    ("None", "none"),
    ("SNIP", "snip"),
    ("Chebyshev", "chebyshev"),
]
# (display label, texture model)
_TEXTURES = [
    ("None", "none"),
    ("March-Dollase", "march_dollase"),
    ("Free (per-peak)", "free"),
]


def _path(name):
    return ("phase_fit", name)


def _combo(label, name, choices):
    labels = [text for text, _ in choices]
    return FormRow(label=label, path=_path(name), value=labels[0], kind="combo", choices=labels)


class PhaseFitDialog(QtWidgets.QDialog):
    """Phase-fit the active 1-D pattern against CIF-defined phases.

    Parameters
    ----------
    pattern_provider : callable
        Zero-arg callable returning ``(x, y, x_label)`` for the currently
        selected frame, or ``None``.  Called on :meth:`refresh_pattern`.
    """

    def __init__(self, pattern_provider=None, parent=None, *, analysis_context=None,
                 vnext_submit=None, allow_batch=False):
        super().__init__(parent)
        self._vnext = callable(vnext_submit)
        self._vnext_submit = vnext_submit
        self._allow_batch = bool(allow_batch)
        self._export_trace = None
        self._export_wavelength = None
        self._export_snapshot_provider = None
        self._analysis_context = analysis_context
        self._provider = (
            analysis_context.current_pattern_tuple
            if analysis_context is not None else pattern_provider)
        if self._provider is None:
            self._provider = lambda: None
        self._x = None
        self._y = None
        self._x_label = "q"
        #: the phases added from CIFs: (path, PhaseModel), or (path, name) when Int's page fits them
        self._phases = []
        #: what the phases' list shows for each (its row's text and tooltip: a fit's CIF receipt once fitted)
        self._phase_entries = []
        self.setObjectName("phaseFitDialog")
        self.setWindowTitle("Phase Fitting")
        self.resize(600, 900)
        self._build_ui()
        if self._vnext:
            self.texture_combo.setCurrentIndex(0); self.texture_combo.setEnabled(False)
            self.texture_combo.setToolTip("P3_7_PHASE_TEXTURE_UNAVAILABLE")
            self.batch_btn.setEnabled(bool(allow_batch))
            if not allow_batch:
                self.batch_btn.setToolTip("P3_7_BATCH_DISPLAY_PROJECTION_UNAVAILABLE")
        from .fit_export import FitExportActions
        self.export_actions = FitExportActions(self, tool='phase', snapshot=self.export_snapshot,
                                               settings=self.vnext_fit_values)

    # ---- UI construction ------------------------------------------------
    def _build_ui(self):
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(11, 11, 11, 11)
        lay.setSpacing(8)

        # Phases: the CIFs, on the shared browser's keyed list, with Add and Remove.
        phases = SectionCard("Phases", collapsible=False)
        self.cif_browser = BrowserView(self, profile=keyed_list_profile("CIF files"))
        self.cif_browser.setObjectName("phaseCifBrowser")
        self.cif_browser.setMinimumWidth(0)
        self.cif_browser.setMaximumHeight(150)
        self.cif_list = self.cif_browser.scans
        self.cif_list.setToolTip("CIF files defining the phases to fit")
        buttons = QtWidgets.QHBoxLayout()
        self.add_cif_btn = QtWidgets.QPushButton("Add CIF…")
        self.add_cif_btn.setToolTip("Browse for a CIF and add it as a phase (+)")
        self.remove_cif_btn = QtWidgets.QPushButton("Remove")
        self.remove_cif_btn.setToolTip("Remove the selected phase")
        buttons.addWidget(self.add_cif_btn)
        buttons.addWidget(self.remove_cif_btn)
        buttons.addStretch(1)
        phases.add_row(self.cif_browser)
        row = QtWidgets.QWidget()
        row.setLayout(buttons)
        phases.add_row(row)
        lay.addWidget(phases)

        # Options (the notebook knobs), on the shared rows.
        options = SectionCard("Phase fit", collapsible=False)
        self.profile_row = _combo("Profile", "profile", _PROFILES)
        self.bkg_row = _combo("Background", "background", _BACKGROUNDS)
        self.texture_row = _combo("Texture", "texture", _TEXTURES)
        self.lattice_row = FormRow(label="Lattice ±%", path=_path("lattice_pct"), value="5")
        self.profile_combo, self.bkg_combo = self.profile_row.editor, self.bkg_row.editor
        self.texture_combo, self.lattice_pct = self.texture_row.editor, self.lattice_row.editor
        self.lattice_pct.setValidator(QtGui.QDoubleValidator(self))
        self.lattice_pct.setToolTip("Lattice tolerance ±% (a/b/c float within this band)")
        for item in (self.profile_row, self.bkg_row, self.texture_row, self.lattice_row):
            options.add_row(item)
        lay.addWidget(options)

        # Run controls.
        run_row = QtWidgets.QHBoxLayout()
        self.advanced_btn = QtWidgets.QPushButton("Advanced…")
        self.refresh_btn = QtWidgets.QPushButton("Reload")
        self.refresh_btn.setToolTip("Re-grab the currently selected frame's pattern")
        self.fit_btn = QtWidgets.QPushButton("Fit")
        self.fit_btn.setObjectName("peakFitGo")
        self.batch_btn = QtWidgets.QPushButton("Batch")
        self.batch_btn.setToolTip(
            "Phase-fit every frame, then plot fractions / lattice vs frame.")
        run_row.addWidget(self.advanced_btn)
        run_row.addStretch(1)
        run_row.addWidget(self.refresh_btn)
        run_row.addWidget(self.fit_btn)
        run_row.addWidget(self.batch_btn)
        lay.addLayout(run_row)

        self._build_advanced()

        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("peakFitStatus")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

        # Data + fit (top) and residual (below), on the shared fit display (Codex's Stitch/Int reuse
        # audit R2); plot and resid_plot are the display's own widgets, which only its reconcile draws on.
        self.fit_view = FitView(self, name="phaseFitView", summary=False, stretch=(3, 1),
                                upper_min_height=220, lower_max_height=110, link_x=True)
        self.plot, self.resid_plot = self.fit_view.plot, self.fit_view.residuals
        self.fit_view.reconcile(FitProjection(lower=FitPanel(zero_line=True)))
        lay.addWidget(self.fit_view, 4)

        # Parameters (fractions / lattice) vs frame, on the shared trend.
        self.trend = TrendView(self, name="phaseFitTrend")
        self.trend.messageChanged.connect(self.status.setText)
        lay.addWidget(self.trend, 2)

        # Results table: phase / fraction / a / b / c, on the shared table.
        self.table = ResultsTable(PHASE_OUTCOME_HEADERS, self, name="phaseFitTable", max_height=120)
        lay.addWidget(self.table)

        self.add_cif_btn.clicked.connect(self._add_cif)
        self.remove_cif_btn.clicked.connect(self._remove_cif)
        self.refresh_btn.clicked.connect(self.refresh_pattern)
        self.fit_btn.clicked.connect(self._do_fit)
        self.advanced_btn.clicked.connect(self._show_advanced)

    def _build_advanced(self):
        """Min intensity and solver iterations, on the shared Advanced dialog's rows."""
        self.advanced = AdvancedDialog(self, title="Advanced — Phase Fitting", object_name="phaseFitAdvancedDialog")
        minimum = FormRow(label="Min peak intensity %", path=_path("min_intensity"), value="5")
        self._maxfev_row = FormRow(label="Max iterations", path=_path("max_nfev"), value="", kind="int", minimum=0,
                                   maximum=1_000_000, allow_blank=True)
        self.advanced.add_group("Phases", (minimum, self._maxfev_row))
        uncertainty = FormRow(label="Use uncertainties", path=_path("use_uncertainties"), value=False, kind="bool")
        self.advanced.add_group("Weighting", (uncertainty,))
        self.use_uncertainties = uncertainty.editor
        self.use_uncertainties.setEnabled(False)
        self.use_uncertainties.setToolTip("Use admitted measured sigma; no uncertainties are inferred")
        self.min_intensity, self.max_nfev = minimum.editor, self._maxfev_row.editor
        self.min_intensity.setValidator(QtGui.QDoubleValidator(self))
        self.min_intensity.setToolTip("Drop template peaks below this % of a phase's max intensity")
        self.max_nfev.setPlaceholderText("auto")
        self.max_nfev.setToolTip("Max solver iterations (max_nfev); blank or 0 = default")

    def _show_advanced(self):
        self.advanced.show(); self.advanced.raise_(); self.advanced.activateWindow()

    def _refusal(self):
        """Max iterations' text refused, named as the row names it ("" when none)."""
        refusal = self._maxfev_row.refusal()
        return f"{self._maxfev_row.label.text()}: {refusal}" if refusal else ""

    # ---- phases (CIF) ---------------------------------------------------
    def _sync_phase_list(self):
        """The phases' list shows the phases, in order; the one chosen stays chosen."""
        self.cif_browser.reconcile_results(tuple(self._phase_entries), self.cif_browser.current_result())

    def _add_cif(self):
        if self._vnext:
            if len(self._phases) >= 8:
                self.status.setText("PHASE_COUNT_LIMIT_EXCEEDED"); return
            picks = browse(self, "Choose CIF", file_filter=CIF_FILTER)
            if not picks:
                return
            path = picks[0]
            if any(path == existing for existing, _name in self._phases):
                self.status.setText("Duplicate CIF path ignored."); return
            base = os.path.splitext(os.path.basename(path))[0] or "phase"
            if len(base.encode("utf-8")) > 256:
                self.status.setText("PHASE_NAME_LIMIT_EXCEEDED"); return
            used = {name for _path, name in self._phases}; name = base; suffix = 2
            while name in used:
                name = f"{base}_{suffix}"; suffix += 1
            if len(name.encode("utf-8")) > 256:
                self.status.setText("PHASE_NAME_LIMIT_EXCEEDED"); return
            self._phases.append((path, name))
            self._phase_entries.append(BrowserResult(str(len(self._phase_entries)), name))
            self._renumber_entries()
            return
        try:
            from xrd_tools.analysis.phase import PhaseModel  # noqa: F401
            import pymatgen  # noqa: F401  CIF parsing backend
        except Exception:
            self.status.setText(
                "Phase fitting needs pymatgen — install it with "
                "`pip install \"xrd-tools[fitting]\"`, then reopen.")
            return
        from xrd_tools.analysis.phase import PhaseModel
        paths = browse(self, "Add CIF phase(s)", many=True, file_filter=CIF_FILTER)
        for path in paths:
            try:
                phase = PhaseModel.from_cif(path)
            except Exception as exc:
                logger.exception("CIF load failed: %s", path)
                self.status.setText(f"Could not load {os.path.basename(path)}: {exc}")
                continue
            self._phases.append((path, phase))
            self._phase_entries.append(BrowserResult("", f"{phase.name}  ({os.path.basename(path)})", path))
        self._renumber_entries()
        if self._phases:
            self.status.setText(
                f"{len(self._phases)} phase(s) loaded. Set options and click Fit.")

    def _renumber_entries(self):
        """Each row keyed by its place (two phases may come from one file), then the list reconciled."""
        self._phase_entries = [replace(entry, identifier=str(index))
                               for index, entry in enumerate(self._phase_entries)]
        self._sync_phase_list()

    def _remove_cif(self):
        row = self.cif_browser.current_result_row()
        if not 0 <= row < len(self._phases):
            return
        self._phases.pop(row)
        self._phase_entries.pop(row)
        self._renumber_entries()

    # ---- data + fit -----------------------------------------------------
    def keyPressEvent(self, event):
        # Esc must not discard the fit setup (phases/range/results); the default
        # QDialog Esc -> reject closes the popup.  Match the metadata plotter.
        # Other keys pass through.
        if event.key() == QtCore.Qt.Key.Key_Escape:
            event.accept()
            return
        super().keyPressEvent(event)

    def refresh_pattern(self):
        """Re-grab the active frame's pattern, draw it, reset the fit/trend."""
        if self._vnext:
            self._vnext_submit("reload", None)
            return
        self._clear_fit()
        self.reset_param_trend()
        data = None
        try:
            data = self._provider()
        except Exception:
            logger.exception("phase-fit: failed to read the current pattern")
        if not data:
            self._x = self._y = None
            self.status.setText("No frame selected — pick a frame, then Reload.")
            self.fit_view.reconcile(replace(self.fit_view.projection, upper=FitPanel()))
            return
        x, y, x_label = data
        self._show_pattern(x, y, x_label)
        n_phases = len(self._phases)
        self.status.setText(
            f"Loaded {int(np.sum(np.isfinite(self._y)))} points; "
            f"{n_phases} phase(s). " + ("Click Fit." if n_phases else "Add a CIF."))

    def set_live_pattern(self, x, y, x_label):
        self._show_pattern(x, y, x_label, clear_fit=True)

    def _show_pattern(self, x, y, x_label, *, clear_fit=False):
        """Draw the pattern; ``clear_fit``: the last fit goes too (its residual and rows), in the same redraw."""
        self._x = np.asarray(x, dtype=float)
        self._y = np.asarray(y, dtype=float)
        self._x_label = x_label or "q"
        self.trend.set_value_unit(self._x_label)
        lower = FitPanel(zero_line=True) if clear_fit else self.fit_view.projection.lower
        self.fit_view.reconcile(FitProjection(self._pattern_panel(), lower))
        if clear_fit:
            self.table.clear_rows()

    def _pattern_panel(self, *series, markers=()):
        """The upper panel: the pattern as a line ("data"), then ``series`` (fit, background,
        components) and the peak ``markers``."""
        return FitPanel((FitSeries(self._x, self._y, "data_line", "data"), *series),
                        self._x_label, "Intensity", markers=markers)

    def _clear_fit(self):
        self.fit_view.reconcile(replace(self.fit_view.projection, lower=FitPanel(zero_line=True)))
        self.table.clear_rows()

    def build_fit_request(self):
        """Build ``(AnalysisInput, PhaseFitAnalyzer)`` from the pattern + CIFs +
        options, or ``None`` (with a status message).  Shared by manual Fit and
        the batch worker, so they fit identically."""
        if self._x is None or self._y is None:
            return None
        if not self._phases:
            self.status.setText("Add at least one CIF phase to fit.")
            return None
        refusal = self._refusal()
        if refusal:
            self.status.setText(refusal)
            return None
        try:
            from xrd_tools.analysis.fitting import fit_peaks  # noqa: F401  lmfit probe
        except Exception:
            self.status.setText(
                "Phase fitting needs lmfit — install `xrd-tools[fitting]`, reopen.")
            return None
        from xrd_tools.analysis.fitting import FitConfig
        from xrd_tools.analysis.plans import PhaseFitPlan
        from xrd_tools.analysis.runner import AnalysisInput, PhaseFitAnalyzer

        finite = np.isfinite(self._x) & np.isfinite(self._y)
        x = self._x[finite]
        y = self._y[finite]
        if x.size < 10:
            self.status.setText("Not enough finite points to fit.")
            return None

        profile = _PROFILES[self.profile_combo.currentIndex()][1]
        background = _BACKGROUNDS[self.bkg_combo.currentIndex()][1]
        texture = _TEXTURES[self.texture_combo.currentIndex()][1]
        try:
            lattice_pct = max(0.0, float(self.lattice_pct.text())) / 100.0
        except ValueError:
            lattice_pct = 0.05
        try:
            min_int = float(self.min_intensity.text())
        except ValueError:
            min_int = 5.0

        init_kw = {}
        if background != "none":
            init_kw["prefit_background"] = background
        fit_kw = {"phase_profile": profile, "lattice_pct": lattice_pct,
                  "texture": texture}
        if self.max_nfev.value():
            fit_kw["max_nfev"] = int(self.max_nfev.value())

        phases = [pm for _, pm in self._phases]
        config = FitConfig(init_kw=init_kw, fit_kw=fit_kw,
                           phase_names=[p.name for p in phases],
                           min_intensity=min_int)
        plan = PhaseFitPlan(config=config)
        inp = AnalysisInput(label="current", x=x, y=y, x_unit=self._x_label)
        return inp, PhaseFitAnalyzer(plan, phases=phases)

    def vnext_fit_values(self):
        try: lattice = float(self.lattice_pct.text()) / 100.0
        except ValueError: lattice = 0.05
        try: minimum = float(self.min_intensity.text())
        except ValueError: minimum = 5.0
        nfev = self.max_nfev.value()
        values = {
            "cif_paths": tuple(path for path, _name in self._phases),
            "phase_names": tuple(name for _path, name in self._phases),
            "expected_hashes": (None,) * len(self._phases),
            "prefit_background": _BACKGROUNDS[self.bkg_combo.currentIndex()][1],
            "phase_profile": _PROFILES[self.profile_combo.currentIndex()][1],
            "lattice_pct": lattice, "min_intensity": minimum,
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
        self.status.setText("Fitting phases… (this can take a few seconds)")
        QtWidgets.QApplication.processEvents()
        outcome = analyzer.analyze(inp)
        if not outcome or not outcome.ok:
            self.status.setText(
                f"Phase fit failed: {outcome.message if outcome else 'no result'}")
            return
        self._draw_outcome(outcome)

    def set_vnext_trace(self, trace):
        self._export_trace = trace
        self.use_uncertainties.setEnabled(trace.sigma is not None)
        if trace.sigma is None:
            self.use_uncertainties.setChecked(False)
        self._show_pattern(trace.axis, trace.intensity, trace.axis_unit or trace.label, clear_fit=True)
        self.status.setText("Displayed trace captured. Add CIFs and click Fit.")

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
        return snapshot_from_trace(self._export_trace, wavelength=self._export_wavelength,
                                   settings=plain_settings(self.vnext_fit_values()))

    def set_vnext_result(self, result):
        fit_x = self._x[np.isfinite(self._x) & np.isfinite(self._y)]
        upper, markers, lower = fit_result_series(fit_x, result,
            tuple((receipt.phase_name, values) for receipt, values in zip(
                result.cif_receipts, result.phase_components, strict=True)))
        self.fit_view.reconcile(FitProjection(self._pattern_panel(*upper, markers=markers),
                                              FitPanel(lower, zero_line=True)))
        for index, receipt in enumerate(result.cif_receipts):
            if index < len(self._phase_entries):  # the row shows the CIF the fit read: its receipt
                self._phase_entries[index] = replace(
                    self._phase_entries[index], label=receipt.phase_name, detail="\n".join(map(str, (
                        receipt.lexical_path, receipt.resolved_path, receipt.byte_count, receipt.sha256,
                        receipt.expected_sha256, receipt.pre_state, receipt.post_state,
                        receipt.receipt_fingerprint))))
        self._sync_phase_list()
        self.table.show_rows(*phase_result_rows(result))
        detail = "; ".join(result.diagnostics)
        self.status.setText((result.message or result.code) +
            (f"; wavelength={result.wavelength_angstrom}" if result.wavelength_angstrom else "") +
            (f"; {detail}" if detail else ""))
        self.status.setToolTip("\n".join((result.plan_fingerprint,
            result.policy_fingerprint, result.result_fingerprint)))

    def closeEvent(self, event):
        if self._vnext:
            self._vnext_submit("close", None)
        super().closeEvent(event)

    def _draw_outcome(self, outcome, auto=False):
        overlay = outcome.overlay
        traces = {} if overlay is None else overlay.traces
        fit = (FitSeries(overlay.x, traces["fit"], "fit", "fit"),) if "fit" in traces else ()
        residual = ((FitSeries(overlay.x, traces["residual"], "residual"),)
                    if "residual" in traces else ())
        self.fit_view.reconcile(FitProjection(
            self._pattern_panel(*fit),
            FitPanel(residual, y_label="resid" if residual else "", zero_line=True)))
        self._fill_phase_table(outcome)

        # Feed the vs-frame trend for live / batch (label = frame index).
        try:
            frame_idx = int(outcome.label)
        except (TypeError, ValueError):
            frame_idx = None
        if frame_idx is not None:
            self._accumulate_frame_params(frame_idx, outcome.params)

    def _fill_phase_table(self, outcome):
        entry = None
        try:
            store = outcome.result.payload
            if len(store):
                entry = store[0]
        except Exception:
            entry = None
        self.table.show_rows(*phase_outcome_rows(entry, [getattr(phase, "name", phase) for _, phase in self._phases]))
        ok = bool(entry.get("success", False)) if entry else False
        rchi = entry.get("redchi") if entry else None
        msg = "Phase fit converged." if ok else "Phase fit did NOT converge (best effort)."
        if rchi is not None and np.isfinite(rchi):
            msg += f"  reduced χ² = {rchi:.4g}."
        self.status.setText(msg)

    # ---- batch hooks (driven by the workspace) --------------------------
    def batch_x_range(self):
        """Phase fitting has no fit-range UI — fit the whole finite pattern, so
        the batch input-builder applies no x-window (the shared batch contract)."""
        return (-np.inf, np.inf)

    def set_batch_running(self, running):
        self.batch_btn.setText("Cancel" if running else "Batch")
        self.fit_btn.setEnabled(not running)
        self.batch_btn.setEnabled(not self._vnext or self._allow_batch)

    def set_batch_progress(self, done, total):
        self.status.setText(f"Phase-fitting… {done}/{total}")

    # ---- the parameter trend ---------------------------------------------
    def reset_param_trend(self):
        """A fresh vs-frame trend (Reload, a new batch run)."""
        self.trend.clear()

    def _accumulate_frame_params(self, frame_idx, params):
        """One frame's parameters on the trend (a re-fit of the frame replaces its row)."""
        self.trend.set_row(str(int(frame_idx)), int(frame_idx), params)
