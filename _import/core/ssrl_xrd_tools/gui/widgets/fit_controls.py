"""Compact ipywidgets-based control panels for XRD phase / peak fitting.

Provides:
- ``PhaseFitControls`` — controls for structure-dependent ``PhaseFitter``
- ``PeakFitControls`` — controls for structure-agnostic ``fit_peaks``

Both classes expose:
- ``.widget``: the assembled ipywidgets layout (Accordion + buttons)
- ``.get_params()``: returns the current settings as a dict
- ``on_fit`` callback: called with the params dict when the Fit button is clicked

Layouts use ``Accordion`` to keep the on-screen footprint small — only one
section is expanded at a time. Sub-sections use ``HBox``/``VBox`` with explicit
widget widths for a clean grid look.
"""
from __future__ import annotations

from typing import Any, Callable

import ipywidgets as widgets
from IPython.display import display

__all__ = ["PhaseFitControls", "PeakFitControls"]


def _label(text: str, width: str = "auto") -> widgets.HTML:
    return widgets.HTML(f"<b>{text}</b>", layout=widgets.Layout(width=width))


def _slider_layout(width: str = "260px") -> widgets.Layout:
    return widgets.Layout(width=width)


class PhaseFitControls:
    """Control panel for structure-dependent multi-phase fitting (PhaseFitter).

    Parameters
    ----------
    phase_names : list of str
        Names of loaded phases (each becomes a toggle button).
    on_fit : callable, optional
        Callback ``f(params_dict) -> None`` invoked when ▶ Fit is clicked.
    """

    def __init__(
        self,
        phase_names: list[str] | None = None,
        on_fit: Callable[[dict], None] | None = None,
    ):
        self._phase_names = phase_names or []
        self._on_fit = on_fit

        # --- Phase toggles ---
        self._phase_toggles: dict[str, widgets.ToggleButton] = {
            name: widgets.ToggleButton(
                value=True,
                description=name,
                button_style="success",
                tooltip=f"Include {name} phase in fit",
                layout=widgets.Layout(width="auto", min_width="110px"),
            )
            for name in self._phase_names
        }

        # --- Background controls ---
        self.use_snip = widgets.Checkbox(
            value=True, description="SNIP pre-subtract",
            indent=False, layout=widgets.Layout(width="180px"),
        )
        self.snip_width = widgets.IntSlider(
            value=30, min=5, max=200, step=5,
            description="SNIP w:", continuous_update=False,
            layout=_slider_layout(),
        )
        self.bg_model = widgets.Dropdown(
            options=["none", "chebyshev2", "chebyshev3", "chebyshev4", "linear"],
            value="none", description="BG fit:",
            layout=widgets.Layout(width="220px"),
        )

        # --- Peak model ---
        self.use_caglioti = widgets.Checkbox(
            value=True, description="Caglioti U/V/W",
            indent=False, layout=widgets.Layout(width="180px"),
        )
        self.fit_method = widgets.Dropdown(
            options=["leastsq", "least_squares", "nelder"],
            value="leastsq", description="Method:",
            layout=widgets.Layout(width="220px"),
        )

        # --- Constraints ---
        self.q_shift_bound = widgets.FloatSlider(
            value=0.05, min=0.001, max=0.5, step=0.005,
            description="|Δq| max:", continuous_update=False,
            readout_format=".3f",
            layout=_slider_layout(),
        )
        self.lattice_pct = widgets.FloatSlider(
            value=5.0, min=0.1, max=20.0, step=0.5,
            description="Lat tol %:", continuous_update=False,
            readout_format=".1f",
            layout=_slider_layout(),
        )
        self.q_min = widgets.BoundedFloatText(
            value=1.0, min=0.0, max=20.0, step=0.1,
            description="q min:",
            layout=widgets.Layout(width="160px"),
        )
        self.q_max = widgets.BoundedFloatText(
            value=5.0, min=0.0, max=20.0, step=0.1,
            description="q max:",
            layout=widgets.Layout(width="160px"),
        )

        # --- MCMC ---
        self.run_mcmc = widgets.Checkbox(
            value=False, description="Run MCMC",
            indent=False, layout=widgets.Layout(width="120px"),
        )
        self.mcmc_steps = widgets.IntSlider(
            value=500, min=100, max=5000, step=100,
            description="Steps:", continuous_update=False,
            layout=_slider_layout(),
        )

        # --- Fit button + status ---
        self._fit_button = widgets.Button(
            description="▶ Fit", button_style="primary",
            tooltip="Run fit with current settings",
            layout=widgets.Layout(width="120px"),
        )
        self._fit_button.on_click(self._do_fit)
        self._status = widgets.HTML(
            "<i>Ready</i>", layout=widgets.Layout(margin="0 0 0 12px"),
        )

        # --- Section layouts ---
        phase_section = widgets.HBox(
            list(self._phase_toggles.values()),
            layout=widgets.Layout(flex_flow="row wrap"),
        )
        bg_section = widgets.VBox([
            widgets.HBox([self.use_snip, self.snip_width]),
            self.bg_model,
        ])
        peak_section = widgets.HBox([self.use_caglioti, self.fit_method])
        constraint_section = widgets.VBox([
            widgets.HBox([self.q_shift_bound, self.lattice_pct],
                         layout=widgets.Layout(flex_flow="row wrap")),
            widgets.HBox([self.q_min, self.q_max]),
        ])
        mcmc_section = widgets.HBox([self.run_mcmc, self.mcmc_steps])

        # Accordion for compactness — only one section shown at a time
        self._accordion = widgets.Accordion(
            children=[phase_section, bg_section, peak_section,
                      constraint_section, mcmc_section],
            layout=widgets.Layout(width="700px"),
        )
        self._accordion.set_title(0, "Phases")
        self._accordion.set_title(1, "Background")
        self._accordion.set_title(2, "Peak model")
        self._accordion.set_title(3, "Constraints")
        self._accordion.set_title(4, "MCMC")
        self._accordion.selected_index = 0

        self.widget = widgets.VBox(
            [self._accordion, widgets.HBox([self._fit_button, self._status])],
            layout=widgets.Layout(width="720px"),
        )

    # ---- public API ----
    @property
    def enabled_phases(self) -> list[str]:
        return [n for n, t in self._phase_toggles.items() if t.value]

    def get_params(self) -> dict[str, Any]:
        return dict(
            use_snip=self.use_snip.value,
            snip_width=self.snip_width.value,
            bg_model=self.bg_model.value,
            use_caglioti=self.use_caglioti.value,
            q_shift_bound=self.q_shift_bound.value,
            lattice_float_pct=self.lattice_pct.value / 100.0,
            q_range=(self.q_min.value, self.q_max.value),
            fit_method=self.fit_method.value,
            run_mcmc=self.run_mcmc.value,
            mcmc_steps=self.mcmc_steps.value,
            enabled_phases=self.enabled_phases,
        )

    def set_status(self, html: str, color: str = "black") -> None:
        self._status.value = f"<span style='color:{color}'>{html}</span>"

    # ---- internals ----
    def _do_fit(self, button: Any = None) -> None:
        self.set_status("Fitting…", color="#b86b00")
        if self._on_fit is None:
            self.set_status("No on_fit callback set", color="red")
            return
        try:
            self._on_fit(self.get_params())
            self.set_status("Fit complete ✓", color="green")
        except Exception as exc:
            self.set_status(f"Fit failed: {exc}", color="red")

    def _ipython_display_(self) -> None:
        display(self.widget)


class PeakFitControls:
    """Control panel for structure-agnostic individual peak fitting (fit_peaks).

    Parameters
    ----------
    on_fit : callable, optional
        Callback ``f(params_dict) -> None`` invoked when ▶ Fit is clicked.
    """

    def __init__(self, on_fit: Callable[[dict], None] | None = None):
        self._on_fit = on_fit

        # --- Peak settings ---
        self.n_peaks = widgets.BoundedIntText(
            value=3, min=1, max=20,
            description="# peaks:",
            layout=widgets.Layout(width="160px"),
        )
        self.peak_model = widgets.Dropdown(
            options=["pseudovoigt", "gaussian", "lorentzian", "voigt", "lorentzian_squared"],
            value="pseudovoigt", description="Model:",
            layout=widgets.Layout(width="220px"),
        )
        self.peak_positions = widgets.Text(
            value="",
            placeholder="e.g. 2.0, 2.14, 2.5, 2.84",
            description="Positions:",
            layout=widgets.Layout(width="450px"),
        )
        self.sigma_init = widgets.FloatSlider(
            value=0.03, min=0.001, max=1.0, step=0.005,
            description="σ init:", continuous_update=False,
            readout_format=".3f",
            layout=_slider_layout(),
        )
        self.sigma_min = widgets.FloatSlider(
            value=0.005, min=0.001, max=0.5, step=0.001,
            description="σ min:", continuous_update=False,
            readout_format=".3f",
            layout=_slider_layout(),
        )
        self.sigma_max = widgets.FloatSlider(
            value=0.2, min=0.01, max=2.0, step=0.01,
            description="σ max:", continuous_update=False,
            readout_format=".3f",
            layout=_slider_layout(),
        )
        self.center_bounds_delta = widgets.FloatSlider(
            value=0.1, min=0.01, max=1.0, step=0.01,
            description="centre ± Δ:", continuous_update=False,
            readout_format=".2f",
            layout=_slider_layout(),
        )

        # --- Background ---
        self.use_snip = widgets.Checkbox(
            value=False, description="SNIP pre-subtract",
            indent=False, layout=widgets.Layout(width="180px"),
        )
        self.snip_width = widgets.IntSlider(
            value=30, min=5, max=200, step=5,
            description="SNIP w:", continuous_update=False,
            layout=_slider_layout(),
        )
        self.bg_model = widgets.Dropdown(
            options=["none", "constant", "linear", "chebyshev2", "chebyshev3", "chebyshev4"],
            value="chebyshev3", description="BG fit:",
            layout=widgets.Layout(width="220px"),
        )

        # --- Amorphous peak ---
        self.fit_amorphous = widgets.Checkbox(
            value=True, description="Fit amorphous",
            indent=False, layout=widgets.Layout(width="160px"),
        )
        self.amorphous_center = widgets.FloatSlider(
            value=1.5, min=0.5, max=3.0, step=0.05,
            description="centre:", continuous_update=False,
            readout_format=".2f",
            layout=_slider_layout(),
        )
        self.amorphous_sigma = widgets.FloatSlider(
            value=0.3, min=0.05, max=2.0, step=0.05,
            description="σ:", continuous_update=False,
            readout_format=".2f",
            layout=_slider_layout(),
        )

        # --- Q range ---
        self.q_min = widgets.BoundedFloatText(
            value=1.0, min=0.0, max=20.0, step=0.1,
            description="q min:",
            layout=widgets.Layout(width="160px"),
        )
        self.q_max = widgets.BoundedFloatText(
            value=5.0, min=0.0, max=20.0, step=0.1,
            description="q max:",
            layout=widgets.Layout(width="160px"),
        )

        # --- Fit button + status ---
        self._fit_button = widgets.Button(
            description="▶ Fit", button_style="primary",
            tooltip="Run fit with current settings",
            layout=widgets.Layout(width="120px"),
        )
        self._fit_button.on_click(self._do_fit)
        self._status = widgets.HTML(
            "<i>Ready</i>", layout=widgets.Layout(margin="0 0 0 12px"),
        )

        # --- Section layouts ---
        peak_section = widgets.VBox([
            widgets.HBox([self.n_peaks, self.peak_model]),
            self.peak_positions,
            widgets.HBox([self.sigma_init, self.sigma_min, self.sigma_max],
                         layout=widgets.Layout(flex_flow="row wrap")),
            self.center_bounds_delta,
        ])
        bg_section = widgets.VBox([
            widgets.HBox([self.use_snip, self.snip_width]),
            self.bg_model,
        ])
        amorphous_section = widgets.VBox([
            self.fit_amorphous,
            widgets.HBox([self.amorphous_center, self.amorphous_sigma],
                         layout=widgets.Layout(flex_flow="row wrap")),
        ])
        qrange_section = widgets.HBox([self.q_min, self.q_max])

        self._accordion = widgets.Accordion(
            children=[peak_section, bg_section, amorphous_section, qrange_section],
            layout=widgets.Layout(width="700px"),
        )
        self._accordion.set_title(0, "Peaks")
        self._accordion.set_title(1, "Background")
        self._accordion.set_title(2, "Amorphous")
        self._accordion.set_title(3, "q range")
        self._accordion.selected_index = 0

        self.widget = widgets.VBox(
            [self._accordion, widgets.HBox([self._fit_button, self._status])],
            layout=widgets.Layout(width="720px"),
        )

    # ---- public API ----
    @property
    def positions_list(self) -> list[float] | None:
        txt = self.peak_positions.value.strip()
        if not txt:
            return None
        try:
            return [float(x.strip()) for x in txt.split(",") if x.strip()]
        except ValueError:
            return None

    def get_params(self) -> dict[str, Any]:
        return dict(
            n_peaks=self.n_peaks.value,
            positions=self.positions_list,
            model=self.peak_model.value,
            sigma_init=self.sigma_init.value,
            sigma_bounds=(self.sigma_min.value, self.sigma_max.value),
            center_bounds_delta=self.center_bounds_delta.value,
            use_snip=self.use_snip.value,
            snip_width=self.snip_width.value,
            background=self.bg_model.value,
            fit_amorphous=self.fit_amorphous.value,
            amorphous_center=self.amorphous_center.value,
            amorphous_sigma=self.amorphous_sigma.value,
            q_range=(self.q_min.value, self.q_max.value),
        )

    def set_status(self, html: str, color: str = "black") -> None:
        self._status.value = f"<span style='color:{color}'>{html}</span>"

    # ---- internals ----
    def _do_fit(self, button: Any = None) -> None:
        self.set_status("Fitting…", color="#b86b00")
        if self._on_fit is None:
            self.set_status("No on_fit callback set", color="red")
            return
        try:
            self._on_fit(self.get_params())
            self.set_status("Fit complete ✓", color="green")
        except Exception as exc:
            self.set_status(f"Fit failed: {exc}", color="red")

    def _ipython_display_(self) -> None:
        display(self.widget)
