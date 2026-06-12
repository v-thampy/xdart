"""Interactive phase-fitting viewer.

Combines ipywidgets controls with a live plotly
:class:`~plotly.graph_objects.FigureWidget` showing the data, current
fit, per-phase / amorphous / background components, and residuals.
Pressing **Fit** runs :meth:`PhaseFitter.fit` with the current control
values and updates the figure in place.

Controls exposed
----------------
* **Phases** — multi-select over registered phases.
* **Caglioti widths** — toggle Caglioti (U, V, W) vs fixed σ per phase.
* **Prefit background** — none / snip / chebyshev (pre-subtracted).
* **Fit background** — none / polynomial{N} / chebyshev{N} (refined).
* **Amorphous peak** — none / gaussian / pseudovoigt / lorentzian /
  voigt / lorentzian_squared.
* **Amorphous init** — center, sigma text fields.
* **min_intensity** — pymatgen template cutoff (0–100 scale).
* **lattice_pct** — fractional tolerance on lattice refinement.
* **max_nfev** — iteration cap.

Example
-------
>>> from ssrl_xrd_tools.analysis.fitting.phase_fitting import PhaseFitter
>>> from ssrl_xrd_tools.gui.widgets import PhaseFitViewer
>>>
>>> viewer = PhaseFitViewer(pattern=result_1d, phases=[ortho, mono, tetra, tin])
>>> viewer.widget    # display in the notebook
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

__all__ = ["PhaseFitViewer"]


# Shared with viz.plotly
_PHASE_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
]


def _lazy_imports():
    """Import ipywidgets / plotly only when the viewer is actually used."""
    try:
        import ipywidgets as widgets
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PhaseFitViewer needs ipywidgets and plotly. "
            "Install with `conda install -c conda-forge ipywidgets plotly`."
        ) from exc
    return widgets, go


class PhaseFitViewer:
    """Interactive multi-phase fit explorer.

    Parameters
    ----------
    pattern : IntegrationResult1D or (q, I[, sigma]) tuple
        The 1D pattern to fit.
    phases : iterable of PhaseModel
        Candidate phases.  The user selects which ones to include in
        each fit via the phase multi-select control.
    height : int
        Total widget height (main panel + residual).
    amorphous_defaults : dict, optional
        Starting values for the amorphous peak init fields
        (``{'center': ..., 'sigma': ...}``).  Default ``center=1.5``,
        ``sigma=0.3``.

    Attributes
    ----------
    fitter : PhaseFitter or None
        The most recently built fitter (``None`` before the first fit).
    result : MultiPhaseResult or None
        The most recent fit result.
    figure_widget : plotly.graph_objects.FigureWidget
        The live plotly figure.  Persistent across fits.
    widget : ipywidgets.VBox
        The assembled controls + figure widget — display this in the
        notebook.
    """

    def __init__(
        self,
        pattern: Any,
        phases: Iterable[Any],
        *,
        height: int = 650,
        figsize: tuple[int, int] | None = None,
        amorphous_defaults: dict | None = None,
        fit_background_template: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        widgets, go = _lazy_imports()

        self._widgets_mod = widgets
        self._go = go

        # ---- Normalise pattern into x/y/sigma ----
        self._pattern_obj = pattern
        x, y, sigma = self._unpack_pattern(pattern)
        self._x = np.asarray(x, dtype=float)
        self._y = np.asarray(y, dtype=float)
        self._sigma = None if sigma is None else np.asarray(sigma, dtype=float)

        self._phases = list(phases)
        if not self._phases:
            raise ValueError("PhaseFitViewer requires at least one phase.")

        # ---- Optional fit-background template ----
        # Stored as (x_ref, y_ref); forwarded to PhaseFitter when the
        # user picks "template" in the fit-background dropdown.
        if fit_background_template is not None:
            x_ref, y_ref = fit_background_template
            self._fit_background_template: tuple[np.ndarray, np.ndarray] | None = (
                np.asarray(x_ref, dtype=float),
                np.asarray(y_ref, dtype=float),
            )
        else:
            self._fit_background_template = None

        # Fit state — populated after the first Fit button click.
        self.fitter: Any | None = None
        self.result: Any | None = None
        self._last_elapsed: float | None = None
        self._prefit_cache: np.ndarray | None = None

        # Build the three UI blocks.
        amorphous_defaults = amorphous_defaults or {}
        controls = self._build_controls(amorphous_defaults)
        self.figure_widget = self._build_figure(figsize, height)
        self._status = widgets.HTML(value="<i>Not fit yet.</i>")

        self.widget = widgets.VBox([controls, self._status, self.figure_widget])

    # ------------------------------------------------------------------
    # UI construction helpers (called once from __init__)
    # ------------------------------------------------------------------

    def _build_controls(self, amorphous_defaults: dict) -> "widgets.VBox":
        """Create all ipywidgets controls grouped by workflow stage.

        Layout
        ------
        Column 1 – **Data / Phases**   : phase select, q-range, display toggles
        Column 2 – **Background**      : prefit bg, fit bg, amorphous
        Column 3 – **Fit constraints**  : bounds, profile, Fit button
        """
        widgets = self._widgets_mod

        # -- Shared styles / layouts (tighter than before) --
        _S = {"description_width": "initial"}
        _W = "240px"                                        # standard width
        _T = widgets.Layout(width=_W)                       # text / dropdown
        _SL = widgets.Layout(width=_W)                      # sliders
        _CB = widgets.Layout(width="200px")                 # checkboxes
        _H = widgets.Layout(width="117px")                  # half-width
        _col = widgets.Layout(width="280px", padding="0 8px 0 0")  # column box

        am_center0 = float(amorphous_defaults.get("center", 1.5))
        am_sigma0 = float(amorphous_defaults.get("sigma", 0.3))
        phase_names = [
            getattr(p, "name", f"phase{i}")
            for i, p in enumerate(self._phases)
        ]

        # ================================================================
        #  Column 1 — Data / Phases
        # ================================================================
        self._phase_select = widgets.SelectMultiple(
            options=phase_names, value=tuple(phase_names),
            rows=min(max(len(phase_names), 3), 6),
            description="Phases", style=_S, layout=_T,
        )
        self._phase_select.observe(self._on_phase_select_changed, names="value")
        self._min_intensity = widgets.FloatText(
            value=5.0, step=0.5, description="min I", style=_S, layout=_T,
        )

        qmin_all = float(np.nanmin(self._x))
        qmax_all = float(np.nanmax(self._x))
        q_step = max((qmax_all - qmin_all) / 500.0, 1e-4)
        self._q_range = widgets.FloatRangeSlider(
            value=(qmin_all, qmax_all), min=qmin_all, max=qmax_all,
            step=q_step, description="q-range", readout_format=".3f",
            continuous_update=True, style=_S, layout=_SL,
        )
        self._q_range.observe(self._on_q_range_changed, names="value")

        self._log_y = widgets.Checkbox(
            value=False, description="log y",
            style=_S, layout=widgets.Layout(width="90px"),
        )
        self._log_y.observe(self._on_log_toggled, names="value")
        self._show_peak_markers = widgets.Checkbox(
            value=False, description="peaks",
            tooltip="Show calculated peak positions from CIF / structural data",
            style=_S, layout=widgets.Layout(width="80px"),
        )
        self._show_peak_markers.observe(self._on_peak_markers_toggled, names="value")
        self._show_hkl = widgets.Checkbox(
            value=False, description="hkl",
            tooltip="Show Miller indices (hkl) at peak positions",
            style=_S, layout=widgets.Layout(width="70px"),
        )
        self._show_hkl.observe(self._on_peak_markers_toggled, names="value")

        display_row = widgets.HBox(
            [self._log_y, self._show_peak_markers, self._show_hkl],
            layout=widgets.Layout(width=_W),
        )

        col_data = widgets.VBox([
            self._phase_select,
            self._min_intensity,
            self._q_range,
            display_row,
        ], layout=_col)

        # ================================================================
        #  Column 2 — Background
        # ================================================================
        self._prefit_bg = widgets.Dropdown(
            options=[("none", "none"), ("snip", "snip"), ("chebyshev", "chebyshev")],
            value="none", description="prefit bg", style=_S, layout=_T,
        )
        self._snip_width = widgets.IntText(
            value=80, description="snip width", style=_S, layout=_T,
        )
        # "template" is always listed so a config that referenced it
        # round-trips cleanly, but it only actually works if a
        # substrate template was passed in at construction time.  The
        # fit-click handler raises a clear error otherwise.
        _has_tmpl = self._fit_background_template is not None
        _suffix = "" if _has_tmpl else " (no template)"
        self._fit_bg = widgets.Dropdown(
            options=[("none", "none"), ("linear", "linear"),
                     ("poly", "poly"), ("cheby", "cheby"),
                     ("spline", "spline"),
                     (f"template{_suffix}", "template"),
                     (f"template+linear{_suffix}", "template+linear"),
                     (f"template+poly{_suffix}", "template+poly"),
                     (f"template+cheb{_suffix}", "template+cheb")],
            value="none", description="fit bg", style=_S, layout=_H,
        )
        self._fit_bg_degree = widgets.BoundedIntText(
            value=3, min=2, max=15, step=1, description="deg", style=_S, layout=_H,
        )
        self._fit_bg_row = widgets.HBox(
            [self._fit_bg, self._fit_bg_degree],
            layout=widgets.Layout(width=_W),
        )
        self._fit_bg.observe(self._on_fit_bg_type_changed, names="value")

        self._amorphous = widgets.Dropdown(
            options=[("none", "none"), ("gaussian", "gaussian"),
                     ("pseudovoigt", "pseudovoigt"), ("lorentzian", "lorentzian"),
                     ("voigt", "voigt"), ("lorentzian²", "lorentzian_squared")],
            value="none", description="amorphous", style=_S, layout=_T,
        )
        self._am_center = widgets.FloatText(
            value=am_center0, step=0.05, description="am center", style=_S, layout=_T,
        )
        self._am_sigma = widgets.FloatText(
            value=am_sigma0, step=0.05, description="am σ", style=_S, layout=_T,
        )
        self._lock_cross_phase = widgets.Checkbox(
            value=False, description="lock a,b,c order",
            style=_S, layout=widgets.Layout(width=_W),
        )
        self._bg_subtract = widgets.ToggleButton(
            value=False, description="subtract bg",
            tooltip="Toggle between raw data and background-subtracted data",
            layout=widgets.Layout(width="130px", height="32px"),
        )
        self._bg_subtract.observe(self._on_bg_subtract_toggled, names="value")

        col_bg = widgets.VBox([
            self._prefit_bg, self._snip_width,
            self._fit_bg_row,
            self._amorphous, self._am_center, self._am_sigma,
            self._lock_cross_phase,
            self._bg_subtract,
        ], layout=_col)

        # ================================================================
        #  Column 3 — Fit constraints & action
        # ================================================================
        self._phase_profile = widgets.Dropdown(
            options=[
                ("pseudo-Voigt", "pseudovoigt"), ("Gaussian", "gaussian"),
                ("Lorentzian", "lorentzian"),    ("Voigt", "voigt"),
                ("Lorentzian²", "lorentzian_squared"),
                ("Pearson VII", "pearson7"),     ("Pearson IV", "pearson4"),
                ("split Lorentz.", "splitlorentzian"),
                ("Moffat", "moffat"),            ("Student's t", "studentst"),
                ("skew Gauss.", "skewedgaussian"),
            ],
            value="pseudovoigt", description="profile", style=_S, layout=_T,
        )
        self._caglioti = widgets.Checkbox(
            value=True, description="Caglioti U,V,W", style=_S, layout=_CB,
        )
        self._scherrer = widgets.Checkbox(
            value=False, description="Scherrer D,ε", style=_S, layout=_CB,
        )
        # Mutex: checking one clears the other. If both are off, a fixed
        # scalar sigma is used.
        self._width_model_row = widgets.HBox(
            [self._caglioti, self._scherrer],
            layout=widgets.Layout(width=_W),
        )
        self._caglioti.observe(self._on_caglioti_toggled, names="value")
        self._scherrer.observe(self._on_scherrer_toggled, names="value")
        self._texture = widgets.Dropdown(
            options=[
                ("none", "none"),
                ("March-Dollase", "march_dollase"),
                ("free (per-peak)", "free"),
            ],
            value="none", description="texture", style=_S, layout=_T,
        )
        small_int = widgets.Layout(width="52px")
        self._march_h = widgets.IntText(value=0, description="h₀", style=_S, layout=small_int)
        self._march_k = widgets.IntText(value=0, description="k₀", style=_S, layout=small_int)
        self._march_l = widgets.IntText(value=1, description="l₀", style=_S, layout=small_int)
        self._march_axis_row = widgets.HBox(
            [widgets.Label("MD axis:", layout=widgets.Layout(width="55px")),
             self._march_h, self._march_k, self._march_l],
            layout=widgets.Layout(width=_W),
        )
        self._texture.observe(self._on_texture_changed, names="value")
        self._march_axis_row.layout.display = "none"

        self._pk_scale_range = widgets.FloatRangeSlider(
            value=(0.5, 1.5), min=0.0, max=10.0, step=0.1,
            description="pk scale", readout_format=".1f",
            style=_S, layout=widgets.Layout(width=_W, display="none"),
        )

        self._lattice_pct = widgets.FloatSlider(
            value=0.05, min=0.0, max=0.2, step=0.005,
            description="lat ±", readout_format=".3f", style=_S, layout=_SL,
        )
        self._q_shift_bound = widgets.FloatSlider(
            value=0.05, min=0.0, max=0.2, step=0.005,
            description="q-shift ±", readout_format=".3f", style=_S, layout=_SL,
        )
        self._width_max = widgets.FloatText(
            value=0.0, step=0.01, description="σ max", style=_S, layout=_H,
        )
        self._width_min = widgets.FloatText(
            value=0.0, step=0.001, description="σ min", style=_S, layout=_H,
        )
        self._width_row = widgets.HBox(
            [self._width_min, self._width_max],
            layout=widgets.Layout(width=_W),
        )
        self._max_nfev = widgets.IntText(
            value=2000, description="max nfev", style=_S, layout=_T,
        )

        self._fit_button = widgets.Button(
            description="Fit", button_style="primary", icon="check",
            layout=widgets.Layout(width="240px", height="36px"),
        )
        self._fit_button.on_click(self._on_fit_clicked)

        # Placeholder row for batch controls (Sequential, Fit All)
        # that BatchPhaseFitViewer can inject via `extra_right_controls`.
        self._extra_right_row = widgets.HBox(
            [], layout=widgets.Layout(width="auto"),
        )

        col_fit = widgets.VBox([
            self._phase_profile, self._width_model_row,
            self._texture, self._march_axis_row, self._pk_scale_range,
            self._lattice_pct, self._q_shift_bound,
            self._width_row, self._max_nfev,
            self._fit_button,
            self._extra_right_row,
        ], layout=_col)

        # ================================================================
        #  Section labels (thin HTML headers above each column)
        # ================================================================
        _hdr = "font-weight:600; font-size:12px; color:#555; margin:0 0 4px 0"
        hdr_data = widgets.HTML(f"<p style='{_hdr}'>Data / Phases</p>")
        hdr_bg   = widgets.HTML(f"<p style='{_hdr}'>Background</p>")
        hdr_fit  = widgets.HTML(f"<p style='{_hdr}'>Fit Constraints</p>")

        data_section = widgets.VBox([hdr_data, col_data])
        bg_section   = widgets.VBox([hdr_bg,   col_bg])
        fit_section  = widgets.VBox([hdr_fit,  col_fit])

        return widgets.HBox([data_section, bg_section, fit_section])

    def _build_figure(
        self, figsize: tuple[int, int] | None, height: int,
    ) -> "go.FigureWidget":
        """Create the plotly FigureWidget with all trace placeholders."""
        go = self._go
        from plotly.subplots import make_subplots

        phase_names = [
            getattr(p, "name", f"phase{i}")
            for i, p in enumerate(self._phases)
        ]
        zeros = np.zeros_like(self._x)

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.75, 0.25], vertical_spacing=0.04,
        )

        def _add(name, row=1, **kw):
            fig.add_trace(
                go.Scatter(x=self._x, y=zeros if "y" not in kw else kw.pop("y"),
                           mode="lines", name=name, **kw),
                row=row, col=1,
            )
            return len(fig.data) - 1

        self._data_idx = _add(
            "data", y=self._y, line=dict(color="black", width=1),
        )
        self._total_idx = _add(
            "total fit", line=dict(color="#e41a1c", width=1.5), visible=False,
        )
        self._bg_idx = _add(
            "background", line=dict(color="#888", width=1, dash="dash"), visible=False,
        )
        self._prefit_idx = _add(
            "prefit bg", line=dict(color="#666", width=1, dash="dashdot"), visible=False,
        )
        self._data_minus_prefit_idx = _add(
            "data − prefit", line=dict(color="#1f77b4", width=1), visible=False,
        )
        self._am_idx = _add(
            "amorphous", line=dict(color="#4daf4a", width=1, dash="dot"), visible=False,
        )
        self._phase_trace_start = len(fig.data)
        for i, name in enumerate(phase_names):
            color = _PHASE_COLORS[i % len(_PHASE_COLORS)]
            _add(name, line=dict(color=color, width=1), visible=False)

        self._residual_idx = _add(
            "residual", row=2,
            line=dict(color="#555", width=1), showlegend=False, visible=False,
        )

        fig.add_hline(y=0, line_color="#aaa", line_width=0.5, row=2, col=1)
        fig.update_yaxes(title_text="Intensity", row=1, col=1)
        fig.update_yaxes(title_text="residual", row=2, col=1)
        fig.update_xaxes(title_text="q (Å⁻¹)", row=2, col=1)

        if figsize is not None:
            fig_w, fig_h = int(figsize[0]), int(figsize[1])
        else:
            fig_w, fig_h = 1100, int(height)
        fig.update_layout(
            height=fig_h, width=fig_w,
            hovermode="x unified",
            margin=dict(l=60, r=20, t=20, b=50),
            legend=dict(font=dict(size=10)),
        )
        return go.FigureWidget(fig)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_pattern(pattern: Any):
        """Accept IntegrationResult1D or (q, I[, sigma]) tuples."""
        from ssrl_xrd_tools.core.containers import IntegrationResult1D
        if isinstance(pattern, IntegrationResult1D):
            return pattern.radial, pattern.intensity, pattern.sigma
        if isinstance(pattern, tuple):
            if len(pattern) == 2:
                return pattern[0], pattern[1], None
            if len(pattern) == 3:
                return pattern
        raise TypeError(
            "pattern must be an IntegrationResult1D or a (q, I[, sigma]) tuple."
        )

    def _build_fitter_kwargs(self) -> tuple[dict, dict]:
        """Split control values into PhaseFitter init vs fit kwargs."""
        init_kw: dict = {}

        prefit = self._prefit_bg.value
        if prefit != "none":
            init_kw["prefit_background"] = prefit
            if prefit == "snip":
                init_kw["prefit_background_kwargs"] = {
                    "snip_width": int(self._snip_width.value),
                }

        fit_bg_spec = self._current_fit_bg_spec()
        if fit_bg_spec is not None:
            init_kw["fit_background"] = fit_bg_spec
            if fit_bg_spec == "template" or fit_bg_spec.startswith("template+"):
                # PhaseFitter expects the reference spectrum passed
                # separately (it doesn't JSON-serialise cleanly through
                # the fit-config, so we keep it on the widget).
                if self._fit_background_template is None:
                    raise RuntimeError(
                        f"Fit background is {fit_bg_spec!r} but no "
                        "substrate template was provided to PhaseFitViewer. "
                        "Pass fit_background_template=(q_ref, I_ref) at "
                        "construction time."
                    )
                init_kw["fit_background_template"] = self._fit_background_template

        if self._amorphous.value != "none":
            init_kw["amorphous_peak"] = self._amorphous.value
            init_kw["amorphous_init"] = {
                "center": float(self._am_center.value),
                "sigma": float(self._am_sigma.value),
            }

        texture = str(self._texture.value)
        march_axis = (
            int(self._march_h.value),
            int(self._march_k.value),
            int(self._march_l.value),
        )
        # Width model: mutex between Caglioti and Scherrer; if neither,
        # a fixed scalar sigma is used.
        if bool(self._scherrer.value):
            width_model = "scherrer"
        elif bool(self._caglioti.value):
            width_model = "caglioti"
        else:
            width_model = "fixed"
        fit_kw: dict = {
            "width_model": width_model,
            "phase_profile": str(self._phase_profile.value),
            "lattice_pct": float(self._lattice_pct.value),
            "q_shift_bound": float(self._q_shift_bound.value),
            "max_nfev": int(self._max_nfev.value),
            "lock_cross_phase": bool(self._lock_cross_phase.value),
            "texture": texture,
            "march_axis": march_axis,
            "pk_scale_range": tuple(float(v) for v in self._pk_scale_range.value),
        }

        # Optional σ cap / floor — 0 / negative ⇒ auto.
        width_max = float(self._width_max.value)
        if width_max > 0.0:
            fit_kw["width_max"] = width_max
        width_min = float(self._width_min.value)
        if width_min > 0.0:
            fit_kw["width_min"] = width_min

        # q-range — only pass if the slider differs from the full span.
        qlo, qhi = (float(v) for v in self._q_range.value)
        qmin_all = float(np.nanmin(self._x))
        qmax_all = float(np.nanmax(self._x))
        if not (np.isclose(qlo, qmin_all) and np.isclose(qhi, qmax_all)):
            fit_kw["q_range"] = (qlo, qhi)

        return init_kw, fit_kw

    def _selected_phase_names(self) -> list[str]:
        # Delegate to the public property; kept for internal call sites.
        return self.selected_phase_names

    def _current_fit_bg_spec(self) -> str | None:
        """Compose the fit-background spec string from the type + degree pair."""
        kind = self._fit_bg.value
        if kind == "none":
            return None
        if kind == "linear":
            return "polynomial1"
        if kind == "template":
            return "template"
        if kind == "template+linear":
            return "template+polynomial1"
        deg = int(self._fit_bg_degree.value)
        if kind == "poly":
            return f"polynomial{deg}"
        if kind == "cheby":
            return f"chebyshev{deg}"
        if kind == "spline":
            return f"spline{max(deg, 4)}"
        if kind == "template+poly":
            return f"template+polynomial{deg}"
        if kind == "template+cheb":
            return f"template+chebyshev{deg}"
        return None

    def _on_fit_bg_type_changed(self, change):
        """Disable the degree field for fit-bg kinds that don't use it."""
        kind = change["new"]
        self._fit_bg_degree.disabled = kind in (
            "none", "linear", "template", "template+linear",
        )

    def _on_caglioti_toggled(self, change):
        """Mutex: turning on Caglioti turns off Scherrer."""
        if bool(change["new"]) and self._scherrer.value:
            self._scherrer.unobserve(self._on_scherrer_toggled, names="value")
            try:
                self._scherrer.value = False
            finally:
                self._scherrer.observe(self._on_scherrer_toggled, names="value")

    def _on_scherrer_toggled(self, change):
        """Mutex: turning on Scherrer turns off Caglioti."""
        if bool(change["new"]) and self._caglioti.value:
            self._caglioti.unobserve(self._on_caglioti_toggled, names="value")
            try:
                self._caglioti.value = False
            finally:
                self._caglioti.observe(self._on_caglioti_toggled, names="value")

    def _on_texture_changed(self, change):
        """Show/hide the March-Dollase axis row or free peak-scale slider."""
        val = change["new"]
        self._march_axis_row.layout.display = (
            "flex" if val == "march_dollase" else "none"
        )
        self._pk_scale_range.layout.display = (
            "flex" if val == "free" else "none"
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_log_toggled(self, change):
        use_log = bool(change["new"])
        self.figure_widget.update_yaxes(
            type="log" if use_log else "linear", row=1, col=1,
        )
        # Re-render so y-values get clamped to a positive floor (log) or
        # restored to their true values (linear).
        if self.fitter is not None and self.result is not None:
            self._update_figure(self._selected_phase_names())
        elif use_log:
            # No fit yet — just clamp the data trace.
            fw = self.figure_widget
            y_raw = np.asarray(self._y, dtype=float)
            floor = max(y_raw[y_raw > 0].min() * 0.1, 1e-1) if np.any(y_raw > 0) else 1e-1
            with fw.batch_update():
                fw.data[self._data_idx].y = np.where(
                    y_raw > floor, y_raw, floor,
                ).tolist()
        else:
            # Restore un-clamped data.
            fw = self.figure_widget
            with fw.batch_update():
                fw.data[self._data_idx].y = np.asarray(
                    self._y, dtype=float,
                ).tolist()
            self._apply_q_range_clip()

    def _on_q_range_changed(self, change):
        """Slider callback — re-clip the viewport."""
        self._apply_q_range_clip(new_range=change["new"])

    def _apply_q_range_clip(self, new_range=None):
        """Clip the displayed x-axis range to match the q-range selector.

        Masks the data / residual / component curves to NaN outside the
        range so plotly's auto y-range doesn't still try to fit the
        hidden points. Called both from the slider observer and at the
        end of :meth:`_update_figure`.
        """
        if new_range is None:
            new_range = self._q_range.value
        qlo, qhi = float(new_range[0]), float(new_range[1])
        if qhi <= qlo:
            return
        fw = self.figure_widget
        mask = (self._x >= qlo) & (self._x <= qhi)

        def _clip(y):
            arr = np.asarray(y, dtype=float).copy()
            arr[~mask] = np.nan
            return arr.tolist()

        # Recompute the prefit baseline when the q-range changes, because
        # SNIP / Chebyshev baselines depend on the visible data window.
        subtract_active = (
            getattr(self, "_bg_subtract", None) is not None
            and self._bg_subtract.value
            and self._prefit_bg.value != "none"
        )
        if subtract_active:
            self._get_prefit_baseline(force=True)

        with fw.batch_update():
            base = np.asarray(self._y, dtype=float)
            if subtract_active and self._prefit_cache is not None:
                if self._prefit_cache.shape == base.shape:
                    base = base - self._prefit_cache
            fw.data[self._data_idx].y = _clip(base)
            for idx in (self._total_idx, self._bg_idx, self._prefit_idx,
                        self._data_minus_prefit_idx, self._am_idx,
                        self._residual_idx):
                trace = fw.data[idx]
                if trace.visible and trace.y is not None:
                    trace.y = _clip(trace.y)
            for slot in range(len(self._phases)):
                trace = fw.data[self._phase_trace_start + slot]
                if trace.visible and trace.y is not None:
                    trace.y = _clip(trace.y)

            fw.layout.xaxis.range = [qlo, qhi]
            fw.layout.xaxis2.range = [qlo, qhi]
            fw.layout.yaxis.autorange = True
            fw.layout.yaxis2.autorange = True

        # Refresh peak markers if they're visible (their q-range filter changed)
        if self._show_peak_markers.value:
            self._update_peak_markers()

    def _get_prefit_baseline(self, *, force: bool = False) -> np.ndarray | None:
        """Return the prefit baseline array, computing only when needed.

        Parameters
        ----------
        force : bool
            If *True* the cache is discarded and the baseline is recomputed
            from scratch.  Use this when the q-range changes, because
            SNIP / Chebyshev baselines depend on the data window.  The
            bg-subtract toggle and post-fit display path pass *False* so
            they reuse whatever was last computed.

        Returns ``None`` when prefit_bg == 'none'.
        """
        if self._prefit_bg.value == "none":
            self._prefit_cache = None
            return None

        if not force and self._prefit_cache is not None:
            return self._prefit_cache

        from ssrl_xrd_tools.analysis.fitting.phase_fitting import PhaseFitter

        init_kw, _ = self._build_fitter_kwargs()
        init_kw = {k: v for k, v in init_kw.items()
                   if k.startswith("prefit_")}
        try:
            if self._sigma is not None:
                preview = PhaseFitter(
                    self._x, self._y, sigma=self._sigma, **init_kw,
                )
            else:
                preview = PhaseFitter(self._x, self._y, **init_kw)
        except Exception as e:  # pragma: no cover — runtime feedback
            self._status.value = (
                f"<span style='color:#b00'>Preview error: {e}</span>"
            )
            return None
        arr = np.asarray(preview.background, dtype=float)
        self._prefit_cache = arr
        return arr

    def _on_bg_subtract_toggled(self, _change):
        """Toggle raw ↔ bg-subtracted display.

        * If a fit has been run, rebuild the full figure — the flag is
          read inside ``_update_figure``.
        * Otherwise, show just the raw (or baseline-subtracted) data.
        """
        if self.fitter is not None and self.result is not None:
            self._update_figure(self._selected_phase_names())
            return

        fw = self.figure_widget
        subtract = bool(self._bg_subtract.value)
        prefit = self._get_prefit_baseline() if subtract else None

        with fw.batch_update():
            # Hide any stale fit traces.
            for idx in (
                self._total_idx, self._bg_idx, self._am_idx,
                self._residual_idx,
            ):
                fw.data[idx].visible = False
            for slot in range(len(self._phases)):
                fw.data[self._phase_trace_start + slot].visible = False

            if subtract and prefit is not None:
                fw.data[self._data_idx].y = (self._y - prefit).tolist()
                fw.data[self._prefit_idx].visible = False
                fw.data[self._data_minus_prefit_idx].visible = False
                self._status.value = (
                    f"<i>Showing data − {self._prefit_bg.value} baseline "
                    f"(no fit yet).</i>"
                )
            else:
                fw.data[self._data_idx].y = np.asarray(
                    self._y, dtype=float
                ).tolist()
                if subtract:
                    self._status.value = (
                        "<i>Prefit bg = none — nothing to subtract. "
                        "Pick 'snip' or 'chebyshev' first.</i>"
                    )
                fw.data[self._prefit_idx].visible = False
                fw.data[self._data_minus_prefit_idx].visible = False

        self._apply_q_range_clip()

    def _on_fit_clicked(self, _button):
        import time

        # Reset button to "in progress" state
        self._fit_button.button_style = "primary"
        self._fit_button.description = "Fitting…"
        self._fit_button.icon = ""

        # Imported lazily so the widget module doesn't pay the cost at import.
        from ssrl_xrd_tools.analysis.fitting.phase_fitting import PhaseFitter

        selected = self._selected_phase_names()
        try:
            init_kw, fit_kw = self._build_fitter_kwargs()
        except Exception as e:
            self._status.value = (
                f"<span style='color:#b00'>Config error: {e}</span>"
            )
            self._fit_button.button_style = "danger"
            self._fit_button.description = "Fit ✗"
            return

        # Zero phases is allowed as long as there is an amorphous or in-fit
        # background component — useful for baseline-only fits.
        has_bg_model = bool(
            init_kw.get("fit_background") or init_kw.get("amorphous_profile")
        )
        if not selected and not has_bg_model:
            self._status.value = (
                "<span style='color:#b00'>Select at least one phase, or "
                "enable an amorphous/in-fit background component.</span>"
            )
            self._fit_button.description = "Fit"
            self._fit_button.icon = "check"
            return

        # Build a fresh fitter every click so init kwargs (prefit, etc) apply.
        try:
            if self._sigma is not None:
                fitter = PhaseFitter(self._x, self._y, sigma=self._sigma, **init_kw)
            else:
                fitter = PhaseFitter(self._x, self._y, **init_kw)
            for name in selected:
                phase = next(p for p in self._phases if getattr(p, "name", None) == name)
                fitter.add_phase(phase, min_intensity=float(self._min_intensity.value))
        except Exception as e:  # pragma: no cover — runtime feedback
            self._status.value = f"<span style='color:#b00'>Build error: {e}</span>"
            self._fit_button.button_style = "danger"
            self._fit_button.description = "Fit ✗"
            return

        self._status.value = "<i>Fitting…</i>"
        try:
            # build_model must be called with matching width_model/texture
            # so phase-eval signatures line up with build_parameters() in
            # fit().
            fitter.build_model(
                width_model=str(fit_kw.get("width_model", "caglioti")),
                texture=str(fit_kw.get("texture", "none")),
                march_axis=tuple(fit_kw.get("march_axis", (0, 0, 1))),
            )
            t0 = time.perf_counter()
            result = fitter.fit(**fit_kw)
            elapsed = time.perf_counter() - t0
        except Exception as e:  # pragma: no cover
            self._status.value = f"<span style='color:#b00'>Fit error: {e}</span>"
            self._fit_button.button_style = "danger"
            self._fit_button.description = "Fit ✗"
            return

        self.fitter = fitter
        self.result = result
        self._last_elapsed = elapsed
        self._update_figure(selected)
        self._update_status(elapsed)

    # ------------------------------------------------------------------
    # Figure / status updates
    # ------------------------------------------------------------------

    def _update_figure(self, selected_phase_names: list[str]):
        fw = self.figure_widget
        assert self.fitter is not None and self.result is not None

        params = self.result.params
        fitter = self.fitter
        prefit = np.asarray(fitter.background, dtype=float)
        y_model = np.asarray(fitter.eval_model(params), dtype=float)

        # PhaseFitter.x may be a subset of self._x (q-range slicing) — align
        # model-domain arrays to the full plot x.
        fit_x = np.asarray(fitter.x, dtype=float)
        full_x = self._x
        use_subset = fit_x.shape != full_x.shape or not np.allclose(fit_x, full_x)

        def _to_full(arr, *, pad=np.nan):
            """Embed a fit-domain array into a full-x-length array."""
            if not use_subset:
                return np.asarray(arr, dtype=float)
            out = np.full_like(full_x, pad, dtype=float)
            mask = (full_x >= fit_x.min()) & (full_x <= fit_x.max())
            # Interpolate so the line doesn't have gaps inside the fit range.
            out[mask] = np.interp(full_x[mask], fit_x, np.asarray(arr, dtype=float))
            return out

        y_model_full = _to_full(y_model)
        prefit_full = _to_full(prefit)
        # Store the fit's prefit baseline into the shared cache so that
        # _get_prefit_baseline() can return it without recomputation when
        # the bg-subtract toggle or _update_figure is called later.
        self._prefit_cache = (
            np.asarray(prefit_full, dtype=float)
            if self._prefit_bg.value != "none" else None
        )

        # Known-array coercion helper — plotly's FigureWidget has a long-
        # standing bug where ndarray round-trips through `_remove_overlapping_props`
        # trigger `ValueError: truth value of array is ambiguous`. Converting
        # to plain lists before assignment avoids it entirely.
        use_log = bool(self._log_y.value)
        if use_log:
            # Floor = 10% of the smallest positive data value, or 0.1
            # if everything is non-positive.  Values at or below the
            # floor are replaced with NaN so they simply disappear
            # instead of dragging the y-axis to -infinity.
            y_pos = self._y[self._y > 0]
            _log_floor = float(y_pos.min() * 0.1) if len(y_pos) > 0 else 0.1
        else:
            _log_floor = 0.0  # unused, but avoids NameError

        def _ylist(a):
            arr = np.asarray(a, dtype=float)
            # Replace non-finite with None so plotly draws gaps cleanly.
            arr = np.where(np.isfinite(arr), arr, np.nan)
            if use_log:
                # Clamp to a positive floor so log axis doesn't explode
                # on zero / negative values (background subtraction,
                # component tails, residuals).
                arr = np.where(arr > _log_floor, arr, np.nan)
            return arr.tolist()

        subtract = bool(self._bg_subtract.value)
        has_prefit = self._prefit_bg.value != "none"
        # Offset subtracted from data/total when the toggle is on. Every
        # individual component (amorphous, phases, fit background) is
        # already plotted WITHOUT the prefit baseline, so the subtract
        # toggle only changes what we do to the raw data / total curve.
        offset_full = prefit_full if (subtract and has_prefit) else 0.0

        with fw.batch_update():
            # Raw data — optionally baseline-subtracted.
            data_display = np.asarray(self._y, dtype=float) - (
                prefit_full if (subtract and has_prefit) else 0.0
            )
            fw.data[self._data_idx].y = _ylist(data_display)

            # Total fit.  y_model_full already includes the prefit
            # baseline (eval_model adds self.background), so subtracting
            # it matches the displayed data.
            fw.data[self._total_idx].y = _ylist(
                y_model_full - offset_full
                if (subtract and has_prefit) else y_model_full
            )
            fw.data[self._total_idx].visible = True

            # Prefit baseline — shown as a separate trace when we have
            # one and the user hasn't subtracted it away.
            if has_prefit and not subtract:
                fw.data[self._prefit_idx].y = _ylist(prefit_full)
                fw.data[self._prefit_idx].visible = True
            else:
                fw.data[self._prefit_idx].visible = False

            # The dedicated "data − prefit" preview trace is unused
            # during a fit — the main data trace now plays that role.
            fw.data[self._data_minus_prefit_idx].visible = False

            # In-fit (refined) background — plotted on its own, never
            # summed with the prefit.  Shown in the same coordinate
            # system as the data: raw when subtract=False, raw when
            # subtract=True (the fit bg is independent of the prefit).
            if getattr(fitter, "_bg_model", None) is not None:
                bg_curve = fitter.eval_fit_background(params)
                if bg_curve is not None:
                    fw.data[self._bg_idx].y = _ylist(
                        _to_full(np.asarray(bg_curve))
                    )
                    fw.data[self._bg_idx].visible = True
                else:
                    fw.data[self._bg_idx].visible = False
            else:
                fw.data[self._bg_idx].visible = False

            # Amorphous — plotted as its own gaussian/pV/etc. curve, no
            # prefit or fit bg added.
            if getattr(fitter, "_amorphous_model", None) is not None:
                am_curve = fitter.eval_amorphous(params)
                if am_curve is not None:
                    fw.data[self._am_idx].y = _ylist(
                        _to_full(np.asarray(am_curve))
                    )
                    fw.data[self._am_idx].visible = True
                else:
                    fw.data[self._am_idx].visible = False
            else:
                fw.data[self._am_idx].visible = False

            # Per-phase components — each plotted on its own.
            original_names = [getattr(p, "name", f"phase{i}")
                              for i, p in enumerate(self._phases)]
            fitted_names = [p.name for p in fitter.phases]
            for slot, name in enumerate(original_names):
                trace = fw.data[self._phase_trace_start + slot]
                if name in fitted_names:
                    i = fitted_names.index(name)
                    y_phase = fitter.eval_phase(i, params)
                    trace.y = _ylist(_to_full(np.asarray(y_phase)))
                    trace.visible = True
                else:
                    trace.visible = False

            # Residual (data − total fit) on the full plot x, with nan
            # outside the fit range so the line drops out there. The
            # prefit subtraction cancels out in the residual so this is
            # toggle-independent.
            residual_full = np.asarray(self._y, dtype=float) - y_model_full
            fw.data[self._residual_idx].y = _ylist(residual_full)
            fw.data[self._residual_idx].visible = True

        # Reapply any active q-range clipping so the freshly-updated
        # fit/data traces respect the viewport.
        self._apply_q_range_clip()

    def _update_status(self, elapsed: float):
        r = self.result
        if r is None:
            return
        fracs = r.phase_fractions()
        frac_html = "  ".join(
            f"<b>{k}</b>: {v:.3f}" for k, v in fracs.items()
        )
        color = "#0a0" if r.success else "#b60"
        self._status.value = (
            f"<span style='color:{color}'>"
            f"fit {'OK' if r.success else 'stopped'}</span> &nbsp; "
            f"time {elapsed:.2f} s &nbsp; "
            f"redχ² {r.redchi:.3g} &nbsp;&nbsp; {frac_html}"
        )
        # Color-coded Fit button feedback
        self._fit_button.button_style = "success" if r.success else "danger"
        self._fit_button.description = "Fit ✓" if r.success else "Fit ✗"

    # ------------------------------------------------------------------
    # Public API — used by BatchPhaseFitViewer and external callers
    # ------------------------------------------------------------------

    def fit(self):
        """Programmatic trigger — same as clicking the Fit button."""
        self._on_fit_clicked(self._fit_button)
        return self.result

    @staticmethod
    def unpack_pattern(pattern: Any):
        """Accept IntegrationResult1D or (q, I[, sigma]) tuples.

        Returns (q, I, sigma) where sigma may be None.
        """
        return PhaseFitViewer._unpack_pattern(pattern)

    def swap_pattern(self, pattern: Any) -> None:
        """Replace the displayed pattern data and reset fit state.

        Parameters
        ----------
        pattern : IntegrationResult1D or (q, I[, sigma])
            The new pattern to display.
        """
        x, y, sigma = self._unpack_pattern(pattern)
        self._x = np.asarray(x, dtype=float)
        self._y = np.asarray(y, dtype=float)
        self._sigma = None if sigma is None else np.asarray(sigma, dtype=float)
        self._pattern_obj = pattern

        # Reset fit state
        self.fitter = None
        self.result = None
        self._prefit_cache = None

        # Update q-range slider bounds
        qmin, qmax = float(np.nanmin(self._x)), float(np.nanmax(self._x))
        self._q_range.min = qmin
        self._q_range.max = qmax
        self._q_range.value = (qmin, qmax)

        # Redraw raw data, hide fit traces
        fw = self.figure_widget
        with fw.batch_update():
            fw.data[self._data_idx].y = self._y.tolist()
            for tidx in (
                self._total_idx, self._bg_idx, self._prefit_idx,
                self._data_minus_prefit_idx, self._am_idx,
                self._residual_idx,
            ):
                fw.data[tidx].visible = False
            for slot in range(len(self._phases)):
                fw.data[self._phase_trace_start + slot].visible = False
        self._status.value = "<i>Not fit yet.</i>"
        self._fit_button.button_style = "primary"
        self._fit_button.description = "Fit"
        self._fit_button.icon = "check"
        self._apply_q_range_clip()

    def restore_result(self, result, fitter, elapsed: float = 0.0) -> None:
        """Restore a previously computed fit result and update the figure.

        Parameters
        ----------
        result : MultiPhaseResult
            A fit result (e.g. from a batch run).
        fitter : PhaseFitter
            The fitter that produced the result.
        elapsed : float
            Wall-clock time for display in the status bar.
        """
        self.fitter = fitter
        self.result = result
        self._last_elapsed = elapsed
        self._prefit_cache = None  # recompute on demand
        self._update_figure(self.selected_phase_names)
        self._update_status(elapsed)

    @property
    def selected_phase_names(self) -> list[str]:
        """Names of the currently selected phases in the multi-select."""
        return list(self._phase_select.value)

    @selected_phase_names.setter
    def selected_phase_names(self, names: list[str]) -> None:
        self._phase_select.value = tuple(names)

    def get_fitter_kwargs(self) -> tuple[dict, dict]:
        """Snapshot the current widget state as ``(init_kw, fit_kw)`` dicts.

        ``init_kw`` is for ``PhaseFitter(...)`` constructor kwargs.
        ``fit_kw`` is for ``PhaseFitter.fit(...)`` kwargs.
        """
        return self._build_fitter_kwargs()

    @property
    def min_intensity(self) -> float:
        """Current min-intensity threshold for phase peak filtering."""
        return float(self._min_intensity.value)

    @property
    def extra_right_controls(self) -> "widgets.HBox":
        """HBox at the bottom of the right column for injecting extra widgets.

        Usage (from BatchPhaseFitViewer)::

            self._viewer.extra_right_controls.children = [
                self._sequential, self._fit_all_btn,
            ]
        """
        return self._extra_right_row

    def invalidate_prefit(self) -> None:
        """Clear the cached prefit baseline so it's recomputed on next use."""
        self._prefit_cache = None

    def set_control_state(
        self,
        init_kw: dict,
        fit_kw: dict,
        phase_names: list[str] | None = None,
        min_intensity: float | None = None,
    ) -> None:
        """Apply a configuration dict to the viewer's control widgets.

        This is the public counterpart of reading widget values via
        :meth:`get_fitter_kwargs` — it writes values back.  Unknown keys
        are silently skipped so that older configs remain compatible with
        newer widget code.

        Parameters
        ----------
        init_kw : dict
            PhaseFitter constructor kwargs (prefit_background, amorphous, etc.).
        fit_kw : dict
            PhaseFitter.fit() kwargs (caglioti, phase_profile, etc.).
        phase_names : list of str, optional
            Which phases to select in the multi-select widget.
        min_intensity : float, optional
            Minimum peak intensity threshold.
        """
        # --- Prefit background ---
        prefit = init_kw.get("prefit_background", "none")
        self._prefit_bg.value = prefit
        if prefit == "snip":
            pkw = init_kw.get("prefit_background_kwargs", {})
            if "snip_width" in pkw:
                self._snip_width.value = pkw["snip_width"]

        # --- Fit background ---
        fit_bg = init_kw.get("fit_background")
        if fit_bg is None:
            self._fit_bg.value = "none"
        elif fit_bg == "template":
            # Silently drop back to "none" if no template is available in
            # this viewer (e.g. loading a template-config into a viewer
            # that was constructed without a substrate spectrum).
            if self._fit_background_template is not None:
                self._fit_bg.value = "template"
            else:
                self._fit_bg.value = "none"
        elif fit_bg.startswith("template+"):
            extra = fit_bg.split("+", 1)[1]
            if self._fit_background_template is None:
                self._fit_bg.value = "none"
            elif extra == "polynomial1":
                self._fit_bg.value = "template+linear"
            elif extra.startswith("polynomial"):
                self._fit_bg.value = "template+poly"
                self._fit_bg_degree.value = int(extra.replace("polynomial", ""))
            elif extra.startswith("chebyshev"):
                self._fit_bg.value = "template+cheb"
                self._fit_bg_degree.value = int(extra.replace("chebyshev", ""))
            else:
                self._fit_bg.value = "template"
        elif fit_bg.startswith("polynomial"):
            deg = int(fit_bg.replace("polynomial", ""))
            if deg == 1:
                self._fit_bg.value = "linear"
            else:
                self._fit_bg.value = "poly"
                self._fit_bg_degree.value = deg
        elif fit_bg.startswith("chebyshev"):
            self._fit_bg.value = "cheby"
            self._fit_bg_degree.value = int(fit_bg.replace("chebyshev", ""))
        elif fit_bg.startswith("spline"):
            self._fit_bg.value = "spline"
            self._fit_bg_degree.value = int(fit_bg.replace("spline", ""))

        # --- Amorphous ---
        am = init_kw.get("amorphous_peak", "none")
        self._amorphous.value = am
        am_init = init_kw.get("amorphous_init", {})
        if "center" in am_init:
            self._am_center.value = am_init["center"]
        if "sigma" in am_init:
            self._am_sigma.value = am_init["sigma"]

        # --- Fit kwargs ---
        # Width model: prefer explicit "width_model", fall back to legacy
        # boolean "caglioti".
        if "width_model" in fit_kw:
            wm = str(fit_kw["width_model"]).lower().strip()
            # Silence observers while we set both at once to avoid mutex
            # bouncing on load.
            self._caglioti.unobserve(self._on_caglioti_toggled, names="value")
            self._scherrer.unobserve(self._on_scherrer_toggled, names="value")
            try:
                self._caglioti.value = (wm == "caglioti")
                self._scherrer.value = (wm == "scherrer")
            finally:
                self._caglioti.observe(self._on_caglioti_toggled, names="value")
                self._scherrer.observe(self._on_scherrer_toggled, names="value")
        elif "caglioti" in fit_kw:
            self._caglioti.unobserve(self._on_caglioti_toggled, names="value")
            self._scherrer.unobserve(self._on_scherrer_toggled, names="value")
            try:
                self._caglioti.value = bool(fit_kw["caglioti"])
                if self._caglioti.value:
                    self._scherrer.value = False
            finally:
                self._caglioti.observe(self._on_caglioti_toggled, names="value")
                self._scherrer.observe(self._on_scherrer_toggled, names="value")
        if "phase_profile" in fit_kw:
            self._phase_profile.value = fit_kw["phase_profile"]
        if "lattice_pct" in fit_kw:
            self._lattice_pct.value = fit_kw["lattice_pct"]
        if "q_shift_bound" in fit_kw:
            self._q_shift_bound.value = fit_kw["q_shift_bound"]
        if "max_nfev" in fit_kw:
            self._max_nfev.value = fit_kw["max_nfev"]
        if "lock_cross_phase" in fit_kw:
            self._lock_cross_phase.value = fit_kw["lock_cross_phase"]
        if "texture" in fit_kw:
            self._texture.value = fit_kw["texture"]
        if "march_axis" in fit_kw:
            h, k, l = fit_kw["march_axis"]
            self._march_h.value = h
            self._march_k.value = k
            self._march_l.value = l
        if "width_max" in fit_kw:
            self._width_max.value = fit_kw["width_max"]
        if "width_min" in fit_kw:
            self._width_min.value = fit_kw["width_min"]
        if "pk_scale_range" in fit_kw:
            self._pk_scale_range.value = tuple(fit_kw["pk_scale_range"])

        # --- Phase selection ---
        if phase_names:
            # Filter to only phases that exist in current options to avoid
            # TraitError when the config references phases not currently loaded.
            available = set(self._phase_select.options)
            filtered = tuple(n for n in phase_names if n in available)
            if filtered:
                self._phase_select.value = filtered
        if min_intensity is not None:
            self._min_intensity.value = min_intensity

    # ------------------------------------------------------------------
    # Peak position markers
    # ------------------------------------------------------------------

    def _on_phase_select_changed(self, _change):
        """Refresh peak markers when the phase selection changes."""
        if self._show_peak_markers.value:
            self._update_peak_markers()

    def _on_peak_markers_toggled(self, _change):
        """Show / hide vertical markers at calculated peak positions."""
        self._update_peak_markers()

    def _update_peak_markers(self):
        """Draw or clear vertical line shapes and optional HKL annotations."""
        fw = self.figure_widget
        show = bool(self._show_peak_markers.value)
        show_hkl = bool(self._show_hkl.value)

        _TAG = "pk_marker"
        _ATAG = "pk_hkl"

        # Strip old markers and annotations
        existing_shapes = [
            s for s in (fw.layout.shapes or ())
            if getattr(s, "name", None) != _TAG
        ]
        existing_annots = [
            a for a in (fw.layout.annotations or ())
            if getattr(a, "name", None) != _ATAG
        ]

        if not show:
            fw.layout.shapes = existing_shapes
            fw.layout.annotations = existing_annots
            return

        qlo, qhi = float(self._q_range.value[0]), float(self._q_range.value[1])
        min_I = self.min_intensity
        selected = set(self.selected_phase_names)
        shapes = list(existing_shapes)
        annotations = list(existing_annots)

        for i, phase in enumerate(self._phases):
            if getattr(phase, "name", None) not in selected:
                continue
            color = _PHASE_COLORS[i % len(_PHASE_COLORS)]
            for pk in phase.peaks:
                if pk.intensity < min_I:
                    continue
                if not (qlo <= pk.q <= qhi):
                    continue
                shapes.append(dict(
                    type="line",
                    x0=pk.q, x1=pk.q, y0=0, y1=1,
                    xref="x", yref="y domain",
                    line=dict(color=color, width=1.2, dash="dot"),
                    opacity=0.5,
                    layer="below",
                    name=_TAG,
                ))
                if show_hkl and hasattr(pk, "hkl") and pk.hkl is not None:
                    _hkl = pk.hkl
                    if len(_hkl) == 4:
                        _hkl = (_hkl[0], _hkl[1], _hkl[3])
                    h, k, l = (int(v) for v in _hkl)
                    annotations.append(dict(
                        x=pk.q, y=1.0,
                        xref="x", yref="y domain",
                        text=f"{h}{k}{l}",
                        showarrow=False,
                        font=dict(size=11, color=color),
                        textangle=-90,
                        yshift=4,
                        name=_ATAG,
                    ))

        fw.layout.shapes = shapes
        fw.layout.annotations = annotations

    def _ipython_display_(self):
        from IPython.display import display
        display(self.widget)
