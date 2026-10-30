"""Reusable Jupyter fit host: shared pattern/control owners, injected public scientific callbacks.

It also hosts displayed phase requests; no PhaseFitter construction or copied fitting policy lives here.
"""
from __future__ import annotations

import html
from dataclasses import replace

import numpy as np


class PatternSelector:
    """Cheap descriptor selection, optionally a bounded set of overlay indices."""
    def __init__(self, labels, *, on_select=None):
        import ipywidgets as widgets
        self.selection = widgets.Dropdown(options=[(label, i) for i, label in enumerate(labels)],
                                          value=0, description="Input")
        self.overlays = widgets.SelectMultiple(options=[(label, i) for i, label in enumerate(labels)],
                                               value=(), description="Overlays", rows=min(4, len(labels)))
        self._callback = on_select
        self.selection.observe(self._changed, names="value")
        self.widget = widgets.HBox([self.selection, self.overlays])

    def _changed(self, change):
        if self._callback is not None:
            self._callback(change["new"])

    def dispose(self):
        from .fit_controls import _close_widgets
        self.selection.unobserve(self._changed, names="value")
        self._callback = None
        _close_widgets(self.widget)


class FitRangeControls:
    """Scientific fit bounds separate from typed presentation limits and Reset."""
    def __init__(self, extent, *, fit_bounds=None, label="native axis", on_display=None, fit_enabled=True):
        import ipywidgets as widgets
        lo, hi = map(float, extent)
        self.fit_bounds = widgets.FloatRangeSlider(value=fit_bounds or (lo, hi), min=lo, max=hi,
                                                   step=(hi-lo)/500, continuous_update=False,
                                                   description=f"Fit {label}") if fit_enabled else None
        self.display_min = widgets.FloatText(value=lo, description="Display min")
        self.display_max = widgets.FloatText(value=hi, description="Display max")
        self.reset = widgets.Button(description="Reset view")
        self._extent, self._callback = (lo, hi), on_display
        self.display_min.observe(self._changed, names="value")
        self.display_max.observe(self._changed, names="value")
        self.reset.on_click(self._reset)
        display = widgets.HBox([self.display_min, self.display_max, self.reset])
        self.widget = widgets.VBox(([self.fit_bounds] if fit_enabled else []) + [display])

    def _changed(self, change):
        if self._callback is not None and self.display_min.value < self.display_max.value:
            self._callback()

    def _reset(self, button):
        self.display_min.value, self.display_max.value = self._extent
        self._changed(None)

    def dispose(self):
        from .fit_controls import _close_widgets
        self.display_min.unobserve(self._changed, names="value")
        self.display_max.unobserve(self._changed, names="value")
        self.reset.on_click(self._reset, remove=True)
        self._callback = None
        _close_widgets(self.widget)


class PeakFitViewer:
    """One admitted selected input, one heavy fit, compact rows, and injected load/fit/batch callbacks.

    ``load_input(index)`` returns a FitInputSnapshot, ``fit_input(snapshot, settings)`` an AnalysisOutcome.
    It never fits merely because presentation or input selection changes. ``from_recipe`` supplies the existing
    public recipe calls; old PatternViewer, PeakFitControls and PhaseFitControls callers keep their defaults.
    """
    def __init__(self, labels, load_input, fit_input, *, settings=None, tool="peak", outcome=None, batch_fit=None, contexts=None):
        import ipywidgets as widgets
        import plotly.graph_objects as go
        from .pattern_viewer import PatternViewer
        from .fit_controls import PeakFitControls, PhaseFitControls
        self._load, self._fit, self._batch = load_input, fit_input, batch_fit
        self.tool, self._disposed = tool, False
        self.snapshot, self.outcome = load_input(0), outcome
        self._settings = dict(settings or {})
        self._committed_model = self._settings.get("model", "pseudovoigt")
        self._context_refs = dict(contexts or {})
        self.selector = PatternSelector(labels, on_select=self.select)
        self.pattern_viewer = PatternViewer()  # the shared owner, extended with an analyzer outcome model
        self.figure_widget = self.pattern_viewer.figure_widget
        self._log = widgets.Checkbox(value=False, description="Log intensity")
        from xrd_tools.analysis.axis_units import canonical_q_unit
        unit = self.snapshot.axes[0].unit
        unit_options = [("Native", unit)]
        if len(self.snapshot.axes) == 1 and self.snapshot.wavelength_angstrom is not None and (
                canonical_q_unit(unit) == "q_A^-1" or unit in {"2th_deg", "2th_rad"}):
            unit_options += [("q (A^-1)", "q_A^-1"), ("2theta (deg)", "2th_deg")]
        self._axis_unit = widgets.Dropdown(options=unit_options, value=unit, description="Axis")
        self._axis_unit.observe(self._presentation_changed, names="value")
        self._overlays = widgets.SelectMultiple(options=["fit", "background", "residual", "components"],
                                                 value=("fit", "background", "residual"), description="Fit traces")
        self._log.observe(self._presentation_changed, names="value")
        self._overlays.observe(self._presentation_changed, names="value")
        self.selector.overlays.observe(self._presentation_changed, names="value")
        axis = self.snapshot.axes[0]
        finite = axis.values[np.isfinite(axis.values)]
        from xrd_tools.notebooks.recipe import native_fit_range
        self.ranges = FitRangeControls((finite.min(), finite.max()), fit_bounds=native_fit_range(self._settings, self.snapshot),
                                       label=f"{axis.name} ({axis.unit})", on_display=self.render)
        self.region_ranges = None
        if tool.startswith("region"):
            from .region_controls import RegionRangeControls
            self.region_ranges = RegionRangeControls(self.snapshot, bounds=self._settings.get("region_bounds"))
            self.region_ranges.set_on_display(self.render)
        cls = PhaseFitControls if tool == "phase" else PeakFitControls
        self.controls = (cls(list(self._settings.get("phase_names", ())), on_fit=lambda _settings: self.fit())
                         if tool == "phase" else cls(on_fit=lambda _settings: self.fit()))
        self.controls.configure_displayed(self._settings)
        self._table = widgets.HTML()
        self._csv = widgets.Textarea(value="", description="CSV", layout=widgets.Layout(width="700px", height="100px"))
        self._fit_all = widgets.Button(description="Fit batch", disabled=batch_fit is None)
        self._fit_all.on_click(self._run_batch)
        self.widget = widgets.VBox([self.selector.widget, widgets.HBox([self._log, self._axis_unit, self._overlays]),
                                    self.region_ranges.widget if self.region_ranges is not None else self.ranges.widget,
                                    self.controls.widget, self._fit_all,
                                    self.figure_widget, self._table, self._csv])
        self.render()
        self.set_table([outcome] if outcome is not None else [])

    @classmethod
    def from_recipe(cls, recipe, *, bundle_root, data_root=None, outcome=None):
        from xrd_tools.notebooks import load_snapshot, run_recipe_batch
        from xrd_tools.notebooks.recipe import analyze_recipe_snapshot
        references = (recipe.input, *recipe.batch)
        def load(index):
            return load_snapshot(references[index], bundle_root=bundle_root, data_root=data_root)
        def fit(snapshot, settings):
            current = replace(recipe, settings=settings)
            return analyze_recipe_snapshot(current, snapshot, data_root=bundle_root if data_root is None else data_root)
        def batch(settings):
            return run_recipe_batch(replace(recipe, settings=settings), bundle_root=bundle_root, data_root=data_root)
        return cls([ref.label for ref in references], load, fit, settings=dict(recipe.settings), tool=recipe.tool,
                   outcome=outcome, batch_fit=batch if recipe.batch else None, contexts={ref.label: ref for ref in references})

    def select(self, index):
        self.snapshot = self._load(index)
        self.outcome = None
        # A new input's native extent replaces the old fit-range widget; no previous fit is adopted for it.
        axis = self.snapshot.axes[0]
        old = self.ranges
        self.ranges = FitRangeControls((float(np.nanmin(axis.values)), float(np.nanmax(axis.values))),
                                      label=f"{axis.name} ({axis.unit})", on_display=self.render)
        if self.region_ranges is not None:
            from .region_controls import RegionRangeControls
            self.region_ranges.dispose()
            self.region_ranges = RegionRangeControls(self.snapshot)
            self.region_ranges.set_on_display(self.render)
        children = list(self.widget.children)
        children[2] = self.region_ranges.widget if self.region_ranges is not None else self.ranges.widget
        self.widget.children = tuple(children)
        old.dispose()
        self.set_table([])
        self.render()

    def current_settings(self):
        settings = self.controls.get_params()
        if self.tool in {"peak", "phase"}:
            from xrd_tools.notebooks.recipe import apply_native_fit_range
            settings = apply_native_fit_range(settings, self.ranges.fit_bounds.value, self.snapshot)
        elif self.region_ranges is not None:
            settings["region_bounds"] = self.region_ranges.bounds
        return settings

    def fit(self):
        if self._disposed:
            return None
        settings = self.current_settings()
        self.outcome = self._fit(self.snapshot, settings)
        self._committed_model = settings.get("model", "pseudovoigt")
        self.render()
        self.set_table([self.outcome])
        return self.outcome

    def _run_batch(self, button):
        if self._batch is not None and not self._disposed:
            outcomes = self._batch(self.current_settings())
            self.set_table(outcomes)

    def _presentation_changed(self, change):
        self.render()

    def render(self):
        if self._disposed:
            return
        from xrd_tools.viz.plotly import plot_analysis_outcome, convert_analysis_axis
        import plotly.graph_objects as go
        fig = plot_analysis_outcome(self.snapshot, self.outcome, overlays=self._overlays.value,
                                    log_y=self._log.value,
                                    x_range=(self.ranges.display_min.value, self.ranges.display_max.value),
                                    axis_unit=self._axis_unit.value,
                                    peak_model=self._committed_model,
                                    plane_axis=self.region_ranges.plane[0] if self.region_ranges is not None else None,
                                    plane_coordinate=self.region_ranges.slice_coordinate.value if self.region_ranges is not None else None)
        # Bound comparison overlays to four; they are loaded only for this render and released immediately.
        if len(self.snapshot.axes) == 1:
            for index in self.selector.overlays.value[:4]:
                other = self._load(index)
                fig.add_trace(go.Scattergl(x=convert_analysis_axis(other, other.axes[0].values, self._axis_unit.value),
                                           y=other.fitted_values(), mode="lines",
                                           name=other.key.result_identity), row=1, col=1)
        with self.figure_widget.batch_update():
            self.figure_widget.data = ()
            for trace in fig.data:
                self.figure_widget.add_trace(trace)
            self.figure_widget.layout = fig.layout

    def set_table(self, outcomes):
        from xrd_tools.analysis.fit_table import outcomes_table, table_csv
        contexts = {**self._context_refs, self.snapshot.key.result_identity: self.snapshot}
        settings = self.current_settings()
        model = settings.get("model", settings.get("phase_profile", settings.get("profile", "pseudovoigt" if self.tool in {"peak", "phase"} else "gaussian")))
        table = outcomes_table(outcomes, axis_unit=self.snapshot.axes[0].unit, contexts=contexts,
                               model=model, settings=settings)
        headings = ("Input", "Component", "Parameter", "Value", "Std err", "Unit", "Status", "Message")
        body = "".join("<tr>" + "".join(f"<td>{html.escape('' if v is None else str(v))}</td>" for v in
                                      (r.input, r.component, r.parameter, r.value, r.stderr, r.unit, r.status, r.message))
                       + "</tr>" for r in table.rows)
        self._table.value = "<table><tr>" + "".join(f"<th>{h}</th>" for h in headings) + "</tr>" + body + "</table>"
        self._csv.value = table_csv(table)

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        self._log.unobserve(self._presentation_changed, names="value")
        self._axis_unit.unobserve(self._presentation_changed, names="value")
        self._overlays.unobserve(self._presentation_changed, names="value")
        self.selector.overlays.unobserve(self._presentation_changed, names="value")
        self._fit_all.on_click(self._run_batch, remove=True)
        self.selector.dispose()
        self.ranges.dispose()
        if self.region_ranges is not None:
            self.region_ranges.dispose()
        self.controls.dispose()
        self.pattern_viewer.dispose()
        from .fit_controls import _close_widgets
        _close_widgets(self.widget)
        self._load = self._fit = self._batch = None
