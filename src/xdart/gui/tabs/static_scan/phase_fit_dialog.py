# -*- coding: utf-8 -*-
"""Self-contained Phase Fitting popup (Direction-A Tools).

Structure-informed multi-phase fitting of the active 1-D pattern: load one or
more CIFs, set the fit options (profile / background / lattice tolerance /
texture — the notebook 03/04 knobs), and fit.  Drives the headless
``xrd_tools.analysis.runner.PhaseFitAnalyzer`` through the SAME contract the
Peak Fitter uses, so Batch reuses the shared worker + the vs-frame trend (phase
fractions / lattice vs frame).

Needs the ``xrd-tools[fitting]`` extra: ``lmfit`` (the fit) AND ``pymatgen``
(CIF parsing).  Both are imported lazily so xdart still launches without them —
the dialog shows a friendly install hint instead.
"""

import logging
import os

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from .param_trend import ParamTrendMixin

logger = logging.getLogger(__name__)

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


class PhaseFitDialog(ParamTrendMixin, QtWidgets.QDialog):
    """Phase-fit the active 1-D pattern against CIF-defined phases.

    Parameters
    ----------
    pattern_provider : callable
        Zero-arg callable returning ``(x, y, x_label)`` for the currently
        selected frame, or ``None``.  Called on :meth:`refresh_pattern`.
    """

    def __init__(self, pattern_provider, parent=None):
        super().__init__(parent)
        self._provider = pattern_provider
        self._x = None
        self._y = None
        self._x_label = "q"
        #: list of (path, PhaseModel) added from CIFs.
        self._phases = []
        self._param_accumulator = {}
        self._param_family_keys = ()
        self.setObjectName("phaseFitDialog")
        self.setWindowTitle("Phase Fitting")
        self.resize(600, 860)
        self._build_ui()

    # ---- UI construction ------------------------------------------------
    def _build_ui(self):
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(11, 11, 11, 11)
        lay.setSpacing(8)

        # Phases: a list of CIFs + Add / Remove.
        phases_row = QtWidgets.QHBoxLayout()
        phases_row.setSpacing(7)
        phases_row.addWidget(QtWidgets.QLabel("Phases"))
        self.cif_list = QtWidgets.QListWidget()
        self.cif_list.setObjectName("phaseCifList")
        self.cif_list.setMaximumHeight(86)
        self.cif_list.setToolTip("CIF files defining the phases to fit")
        phases_row.addWidget(self.cif_list, 1)
        btn_col = QtWidgets.QVBoxLayout()
        btn_col.setSpacing(4)
        self.add_cif_btn = QtWidgets.QPushButton("Add CIF…")
        self.add_cif_btn.setToolTip("Browse for a CIF and add it as a phase (+)")
        self.remove_cif_btn = QtWidgets.QPushButton("Remove")
        self.remove_cif_btn.setToolTip("Remove the selected phase")
        btn_col.addWidget(self.add_cif_btn)
        btn_col.addWidget(self.remove_cif_btn)
        btn_col.addStretch(1)
        phases_row.addLayout(btn_col)
        lay.addLayout(phases_row)

        # Options (the notebook knobs).
        opts = QtWidgets.QHBoxLayout()
        opts.setSpacing(7)
        self.profile_combo = QtWidgets.QComboBox()
        for label, _ in _PROFILES:
            self.profile_combo.addItem(label)
        self.bkg_combo = QtWidgets.QComboBox()
        for label, _ in _BACKGROUNDS:
            self.bkg_combo.addItem(label)
        self.texture_combo = QtWidgets.QComboBox()
        for label, _ in _TEXTURES:
            self.texture_combo.addItem(label)
        self.lattice_pct = QtWidgets.QLineEdit("5")
        self.lattice_pct.setValidator(QtGui.QDoubleValidator(self))
        self.lattice_pct.setMaximumWidth(54)
        self.lattice_pct.setToolTip("Lattice tolerance ±% (a/b/c float within this band)")
        opts.addWidget(QtWidgets.QLabel("Profile"))
        opts.addWidget(self.profile_combo)
        opts.addWidget(QtWidgets.QLabel("Background"))
        opts.addWidget(self.bkg_combo)
        opts.addWidget(QtWidgets.QLabel("Texture"))
        opts.addWidget(self.texture_combo)
        opts.addWidget(QtWidgets.QLabel("Lattice ±%"))
        opts.addWidget(self.lattice_pct)
        opts.addStretch(1)
        lay.addLayout(opts)

        # Advanced (collapsible): min intensity + solver iterations.
        self.advanced_btn = QtWidgets.QPushButton("Advanced ▾")
        self.advanced_btn.setCheckable(True)
        adv_row = QtWidgets.QHBoxLayout()
        adv_row.addWidget(self.advanced_btn)
        adv_row.addStretch(1)
        lay.addLayout(adv_row)
        self.advanced_box = QtWidgets.QWidget()
        self.advanced_box.setVisible(False)
        adv = QtWidgets.QHBoxLayout(self.advanced_box)
        adv.setContentsMargins(2, 2, 2, 2)
        self.min_intensity = QtWidgets.QLineEdit("5")
        self.min_intensity.setValidator(QtGui.QDoubleValidator(self))
        self.min_intensity.setMaximumWidth(60)
        self.min_intensity.setToolTip(
            "Drop template peaks below this % of a phase's max intensity")
        self.max_nfev = QtWidgets.QSpinBox()
        self.max_nfev.setRange(0, 1000000)
        self.max_nfev.setSingleStep(500)
        self.max_nfev.setSpecialValueText("auto")
        self.max_nfev.setMaximumWidth(110)
        self.max_nfev.setToolTip("Max solver iterations (max_nfev); 0 = default")
        adv.addWidget(QtWidgets.QLabel("Min peak intensity %"))
        adv.addWidget(self.min_intensity)
        adv.addWidget(QtWidgets.QLabel("Max iterations"))
        adv.addWidget(self.max_nfev)
        adv.addStretch(1)
        lay.addWidget(self.advanced_box)

        # Run controls.
        run_row = QtWidgets.QHBoxLayout()
        run_row.addStretch(1)
        self.refresh_btn = QtWidgets.QPushButton("Reload")
        self.refresh_btn.setToolTip("Re-grab the currently selected frame's pattern")
        self.fit_btn = QtWidgets.QPushButton("Fit")
        self.fit_btn.setObjectName("peakFitGo")
        self.batch_btn = QtWidgets.QPushButton("Batch")
        self.batch_btn.setToolTip(
            "Phase-fit every frame, then plot fractions / lattice vs frame.")
        run_row.addWidget(self.refresh_btn)
        run_row.addWidget(self.fit_btn)
        run_row.addWidget(self.batch_btn)
        lay.addLayout(run_row)

        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("peakFitStatus")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

        # Data + fit (top) and residual (below).
        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(220)
        self.plot.addLegend(offset=(-10, 10))
        self.resid_plot = pg.PlotWidget()
        self.resid_plot.setMaximumHeight(110)
        self.resid_plot.setXLink(self.plot)
        self.resid_plot.addLine(y=0, pen=pg.mkPen((130, 130, 140), width=1))
        lay.addWidget(self.plot, 3)
        lay.addWidget(self.resid_plot, 1)

        # Row 3: parameters (fractions / lattice) vs frame (shared trend).
        self._build_param_trend_row(lay)

        # Results table: phase / fraction / a / b / c.
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Phase", "Fraction", "a", "b", "c"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setMaximumHeight(120)
        self.table.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(self.table)

        self.add_cif_btn.clicked.connect(self._add_cif)
        self.remove_cif_btn.clicked.connect(self._remove_cif)
        self.refresh_btn.clicked.connect(self.refresh_pattern)
        self.fit_btn.clicked.connect(self._do_fit)
        self.advanced_btn.toggled.connect(self._on_advanced_toggled)
        self._connect_param_trend()

    def _on_advanced_toggled(self, on):
        self.advanced_box.setVisible(on)
        self.advanced_btn.setText("Advanced ▴" if on else "Advanced ▾")

    # ---- phases (CIF) ---------------------------------------------------
    def _add_cif(self):
        try:
            from xrd_tools.analysis.phase import PhaseModel  # noqa: F401
            import pymatgen  # noqa: F401  CIF parsing backend
        except Exception:
            self.status.setText(
                "Phase fitting needs pymatgen — install it with "
                "`pip install \"xrd-tools[fitting]\"`, then reopen.")
            return
        from xrd_tools.analysis.phase import PhaseModel
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add CIF phase(s)", "", "CIF files (*.cif);;All files (*)")
        for path in paths:
            try:
                phase = PhaseModel.from_cif(path)
            except Exception as exc:
                logger.exception("CIF load failed: %s", path)
                self.status.setText(f"Could not load {os.path.basename(path)}: {exc}")
                continue
            self._phases.append((path, phase))
            item = QtWidgets.QListWidgetItem(f"{phase.name}  ({os.path.basename(path)})")
            item.setToolTip(path)
            self.cif_list.addItem(item)
        if self._phases:
            self.status.setText(
                f"{len(self._phases)} phase(s) loaded. Set options and click Fit.")

    def _remove_cif(self):
        row = self.cif_list.currentRow()
        if row < 0:
            return
        self.cif_list.takeItem(row)
        if 0 <= row < len(self._phases):
            self._phases.pop(row)

    # ---- data + fit -----------------------------------------------------
    def refresh_pattern(self):
        """Re-grab the active frame's pattern, draw it, reset the fit/trend."""
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
            self.plot.clear()
            return
        x, y, x_label = data
        self._show_pattern(x, y, x_label)
        n_phases = len(self._phases)
        self.status.setText(
            f"Loaded {int(np.sum(np.isfinite(self._y)))} points; "
            f"{n_phases} phase(s). " + ("Click Fit." if n_phases else "Add a CIF."))

    def set_live_pattern(self, x, y, x_label):
        self._clear_fit()
        self._show_pattern(x, y, x_label)

    def _show_pattern(self, x, y, x_label):
        self._x = np.asarray(x, dtype=float)
        self._y = np.asarray(y, dtype=float)
        self._x_label = x_label or "q"
        self.plot.clear()
        self.plot.setLabel("bottom", self._x_label)
        self.plot.setLabel("left", "Intensity")
        self.plot.plot(self._x, self._y, pen=pg.mkPen((210, 210, 220), width=1),
                       name="data")

    def _clear_fit(self):
        self.resid_plot.clear()
        self.resid_plot.addLine(y=0, pen=pg.mkPen((130, 130, 140), width=1))
        self.table.setRowCount(0)

    def build_fit_request(self):
        """Build ``(AnalysisInput, PhaseFitAnalyzer)`` from the pattern + CIFs +
        options, or ``None`` (with a status message).  Shared by manual Fit and
        the batch worker, so they fit identically."""
        if self._x is None or self._y is None:
            return None
        if not self._phases:
            self.status.setText("Add at least one CIF phase to fit.")
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
        if self.max_nfev.value() > 0:
            fit_kw["max_nfev"] = int(self.max_nfev.value())

        phases = [pm for _, pm in self._phases]
        config = FitConfig(init_kw=init_kw, fit_kw=fit_kw,
                           phase_names=[p.name for p in phases],
                           min_intensity=min_int)
        plan = PhaseFitPlan(config=config)
        inp = AnalysisInput(label="current", x=x, y=y, x_unit=self._x_label)
        return inp, PhaseFitAnalyzer(plan, phases=phases)

    def _do_fit(self):
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

    def _draw_outcome(self, outcome, auto=False):
        overlay = outcome.overlay
        self.plot.clear()
        self.plot.setLabel("bottom", self._x_label)
        self.plot.setLabel("left", "Intensity")
        self.plot.plot(self._x, self._y, pen=pg.mkPen((210, 210, 220), width=1),
                       name="data")
        if overlay is not None and "fit" in overlay.traces:
            self.plot.plot(overlay.x, overlay.traces["fit"],
                           pen=pg.mkPen((189, 147, 249), width=2), name="fit")
        self._clear_fit()
        if overlay is not None and "residual" in overlay.traces:
            self.resid_plot.plot(overlay.x, overlay.traces["residual"],
                                 pen=pg.mkPen((230, 133, 151), width=1))
            self.resid_plot.setLabel("left", "resid")

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
        fracs = (entry or {}).get("phase_fractions", {}) or {}
        lattice = (entry or {}).get("lattice_params", {}) or {}
        names = list(fracs) or [name for name, _ in
                                ((p.name, p) for _, p in self._phases)]
        self.table.setRowCount(len(names))
        for r, name in enumerate(names):
            lat = lattice.get(name, {})
            vals = [name, f"{fracs.get(name, float('nan')):.3f}"]
            for key in ("a", "b", "c"):
                v = lat.get(key)
                vals.append(f"{v:.4f}" if v is not None else "—")
            for col, v in enumerate(vals):
                self.table.setItem(r, col, QtWidgets.QTableWidgetItem(v))
        ok = bool(entry.get("success", False)) if entry else False
        rchi = entry.get("redchi") if entry else None
        msg = "Phase fit converged." if ok else "Phase fit did NOT converge (best effort)."
        if rchi is not None and np.isfinite(rchi):
            msg += f"  reduced χ² = {rchi:.4g}."
        self.status.setText(msg)

    # ---- batch hooks (driven by staticWidget) ---------------------------
    def set_batch_running(self, running):
        self.batch_btn.setText("Cancel" if running else "Batch")
        self.fit_btn.setEnabled(not running)

    def set_batch_progress(self, done, total):
        self.status.setText(f"Phase-fitting… {done}/{total}")
