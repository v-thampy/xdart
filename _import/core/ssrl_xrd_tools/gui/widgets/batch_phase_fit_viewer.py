"""Batch wrapper around :class:`PhaseFitViewer` for multi-pattern fitting.

Adds a pattern selector, **Fit All** / **Sequential** controls, config
save/load, and a phase-fraction-vs-index plot.  The underlying
:class:`PhaseFitViewer` is reused: switching patterns swaps the data
and re-renders the last fit for that pattern (if available).

Example
-------
>>> from ssrl_xrd_tools.gui.widgets import BatchPhaseFitViewer
>>> viewer = BatchPhaseFitViewer(patterns, phases, labels=labels)
>>> viewer.widget
"""
from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = ["BatchPhaseFitViewer"]


def _lazy_imports():
    try:
        import ipywidgets as widgets
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError(
            "BatchPhaseFitViewer needs ipywidgets and plotly. "
            "Install with `conda install -c conda-forge ipywidgets plotly`."
        ) from exc
    return widgets, go, make_subplots


class BatchPhaseFitViewer:
    """Multi-pattern wrapper around :class:`PhaseFitViewer`.

    Parameters
    ----------
    patterns : list
        Each element is an ``IntegrationResult1D`` or ``(q, I[, sigma])``
        tuple — anything :class:`PhaseFitViewer` accepts.
    phases : list of PhaseModel
        Candidate crystal phases.
    labels : list of str, optional
        Per-pattern labels shown in the dropdown.  Defaults to
        ``"#0", "#1", …``.
    height, figsize, amorphous_defaults
        Forwarded to the inner :class:`PhaseFitViewer`.
    """

    def __init__(
        self,
        patterns: Sequence[Any],
        phases: Iterable[Any],
        *,
        labels: Sequence[str] | None = None,
        height: int = 650,
        figsize: tuple[int, int] | None = None,
        amorphous_defaults: dict | None = None,
        fit_background_template: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        widgets, go, make_subplots = _lazy_imports()
        self._widgets = widgets
        self._go = go
        self._make_subplots = make_subplots

        self._patterns = list(patterns)
        self._phases = list(phases)
        self._labels = (
            list(labels) if labels is not None
            else [f"#{i}" for i in range(len(self._patterns))]
        )
        if len(self._labels) != len(self._patterns):
            raise ValueError(
                f"len(labels)={len(self._labels)} != "
                f"len(patterns)={len(self._patterns)}"
            )

        self._height = height
        self._figsize = figsize
        self._amorphous_defaults = amorphous_defaults or {}
        # Keep a reference so Fit-All rebuilds of the inner viewer (if
        # any) can re-forward the template; also used by the batch-mode
        # fit_sequence call below.
        self._fit_background_template = fit_background_template

        # ------ Import the inner viewer class lazily ------
        from ssrl_xrd_tools.gui.widgets.phase_fit_viewer import PhaseFitViewer
        self._PhaseFitViewer = PhaseFitViewer

        # ------ Per-pattern state ------
        # results[i] = (MultiPhaseResult, params_snapshot, elapsed) or None
        self._results: list[tuple | None] = [None] * len(self._patterns)

        # ------ Build inner viewer for the first pattern ------
        self._viewer = PhaseFitViewer(
            self._patterns[0],
            self._phases,
            height=height,
            figsize=figsize,
            amorphous_defaults=self._amorphous_defaults,
            fit_background_template=self._fit_background_template,
        )

        # ------ Batch controls ------
        _lbl = widgets.Layout(width="auto")
        _btn = widgets.Layout(width="130px", height="36px")

        self._pattern_dropdown = widgets.Dropdown(
            options={label: idx for idx, label in enumerate(self._labels)},
            value=0,
            description="Pattern",
            style={"description_width": "auto"},
            layout=widgets.Layout(width="300px"),
        )
        self._pattern_dropdown.observe(self._on_pattern_changed, names="value")

        self._sequential = widgets.Checkbox(
            value=False, description="Sequential",
            tooltip="Use result from pattern N as starting guess for N+1",
            layout=_lbl,
        )

        self._fit_all_btn = widgets.Button(
            description="Fit All",
            button_style="warning",
            layout=_btn,
        )
        self._fit_all_btn.on_click(self._on_fit_all)

        self._save_config_btn = widgets.Button(
            description="Save Config",
            button_style="info",
            layout=_btn,
        )
        self._save_config_btn.on_click(self._on_save_config)

        self._load_config_btn = widgets.Button(
            description="Load Config",
            button_style="info",
            layout=_btn,
        )
        self._load_config_btn.on_click(self._on_load_config)

        self._config_name = widgets.Text(
            value="fit_config.json",
            description="",
            layout=widgets.Layout(width="200px"),
        )

        # Optional file chooser (ipyfilechooser) for browse dialogs
        self._file_chooser = None
        try:
            from ipyfilechooser import FileChooser
            self._file_chooser = FileChooser(
                path=".",
                filter_pattern="*.json",
                title="",
                show_hidden=False,
                select_default=False,
                layout=widgets.Layout(width="auto"),
            )
            self._file_chooser.register_callback(self._on_file_chosen)
        except ImportError:
            pass

        self._batch_status = widgets.HTML(value="")
        self._progress = widgets.IntProgress(
            value=0, min=0, max=len(self._patterns),
            description="", layout=widgets.Layout(width="300px"),
        )
        self._progress.layout.visibility = "hidden"

        # Phase fraction plot
        self._frac_fig = self._build_frac_figure()

        # ------ Assemble layout ------
        top_row = widgets.HBox([
            self._pattern_dropdown,
            self._sequential,
            self._fit_all_btn,
        ])

        # Config in a collapsible accordion so it doesn't clutter the main view
        config_contents = [
            self._save_config_btn,
            self._load_config_btn,
            self._config_name,
        ]
        if self._file_chooser is not None:
            config_row = widgets.VBox([
                widgets.HBox(config_contents),
                self._file_chooser,
            ])
        else:
            config_row = widgets.HBox(config_contents)

        self._config_accordion = widgets.Accordion(
            children=[config_row],
            layout=widgets.Layout(width="auto"),
        )
        self._config_accordion.set_title(0, "Config Save / Load")
        self._config_accordion.selected_index = None  # collapsed by default

        progress_row = widgets.HBox([self._progress, self._batch_status])

        self.widget = widgets.VBox([
            top_row,
            self._config_accordion,
            progress_row,
            self._viewer.widget,
            self._frac_fig,
        ])

    # ------------------------------------------------------------------
    # FitConfig bridge
    # ------------------------------------------------------------------

    def get_config(self) -> "FitConfig":
        """Snapshot the current viewer widget state as a :class:`FitConfig`."""
        from ssrl_xrd_tools.analysis.fitting.batch import FitConfig
        init_kw, fit_kw = self._viewer.get_fitter_kwargs()
        return FitConfig(
            init_kw=init_kw,
            fit_kw=fit_kw,
            phase_names=self._viewer.selected_phase_names,
            min_intensity=self._viewer.min_intensity,
        )

    def set_config(self, config: "FitConfig") -> None:
        """Apply a :class:`FitConfig` to the viewer widgets.

        Delegates to :meth:`PhaseFitViewer.set_control_state` which
        silently skips unknown keys so old configs work with newer code.
        """
        self._viewer.set_control_state(
            init_kw=config.init_kw,
            fit_kw=config.fit_kw,
            phase_names=config.phase_names,
            min_intensity=config.min_intensity,
        )

    # ------------------------------------------------------------------
    # Pattern switching
    # ------------------------------------------------------------------

    def _on_pattern_changed(self, change):
        idx = change["new"]
        self._switch_pattern(idx)

    def _switch_pattern(self, idx: int):
        """Swap the viewer's data to pattern *idx* and restore any saved result."""
        pat = self._patterns[idx]

        # swap_pattern resets fit state, redraws raw data, updates q-range
        self._viewer.swap_pattern(pat)

        # Restore saved result if we have one
        saved = self._results[idx]
        if saved is not None:
            result, fitter, elapsed = saved
            self._viewer.restore_result(result, fitter, elapsed)

    # ------------------------------------------------------------------
    # Fit All
    # ------------------------------------------------------------------

    def _on_fit_all(self, _btn):
        """Fit every pattern using the current widget config."""
        n = len(self._patterns)
        self._progress.max = n
        self._progress.value = 0
        self._progress.layout.visibility = "visible"
        self._fit_all_btn.disabled = True

        config = self.get_config()
        sequential = bool(self._sequential.value)
        prev_params = None

        from ssrl_xrd_tools.analysis.fitting.phase_fitting import PhaseFitter

        selected_phases = [
            p for p in self._phases
            if getattr(p, "name", None) in config.phase_names
        ]
        if not selected_phases:
            self._batch_status.value = (
                "<span style='color:#b00'>No phases selected.</span>"
            )
            self._fit_all_btn.disabled = False
            self._progress.layout.visibility = "hidden"
            return

        # FitConfig deliberately doesn't carry the (ndarray) template —
        # it doesn't JSON-round-trip.  Re-attach it here per-iteration.
        _fit_bg_cfg = config.init_kw.get("fit_background") or ""
        needs_template = (
            _fit_bg_cfg == "template" or _fit_bg_cfg.startswith("template+")
        )
        if needs_template and self._fit_background_template is None:
            self._batch_status.value = (
                f"<span style='color:#b00'>fit_background={_fit_bg_cfg!r} "
                "but no substrate template was provided to "
                "BatchPhaseFitViewer. Pass "
                "fit_background_template=(q_ref, I_ref) at construction."
                "</span>"
            )
            self._fit_all_btn.disabled = False
            self._progress.layout.visibility = "hidden"
            return

        for i, pat in enumerate(self._patterns):
            x, y, sigma = self._viewer.unpack_pattern(pat)
            init_kw = dict(config.init_kw)
            if needs_template:
                init_kw["fit_background_template"] = self._fit_background_template

            try:
                if sigma is not None:
                    fitter = PhaseFitter(
                        np.asarray(x), np.asarray(y),
                        sigma=np.asarray(sigma), **init_kw,
                    )
                else:
                    fitter = PhaseFitter(
                        np.asarray(x), np.asarray(y), **init_kw,
                    )
                for ph in selected_phases:
                    fitter.add_phase(
                        ph, min_intensity=config.min_intensity,
                    )
            except Exception as exc:
                self._batch_status.value = (
                    f"<span style='color:#b00'>"
                    f"Pattern {self._labels[i]} build error: {exc}</span>"
                )
                self._results[i] = None
                self._progress.value = i + 1
                continue

            fit_kw = dict(config.fit_kw)
            if sequential and prev_params is not None:
                fit_kw["params"] = prev_params

            t0 = time.perf_counter()
            try:
                result = fitter.fit(**fit_kw)
                elapsed = time.perf_counter() - t0
            except Exception as exc:
                self._batch_status.value = (
                    f"<span style='color:#b00'>"
                    f"Pattern {self._labels[i]} fit error: {exc}</span>"
                )
                self._results[i] = None
                self._progress.value = i + 1
                continue

            self._results[i] = (result, fitter, elapsed)

            if sequential:
                prev_params = deepcopy(result.params)

            self._progress.value = i + 1
            ok = result.success
            fracs = result.phase_fractions()
            frac_str = "  ".join(f"{k}={v:.3f}" for k, v in fracs.items())
            tag = self._labels[i]
            self._batch_status.value = (
                f"<span style='color:{'#0a0' if ok else '#b60'}'>"
                f"[{i+1}/{n}] {tag}: "
                f"{'OK' if ok else 'STOP'} "
                f"redχ²={result.redchi:.3g} {frac_str}</span>"
            )

            # Live-update: show this pattern's fit in the viewer and
            # refresh the phase-fraction plot after each fit.
            self._pattern_dropdown.unobserve(
                self._on_pattern_changed, names="value",
            )
            self._pattern_dropdown.value = i
            self._pattern_dropdown.observe(
                self._on_pattern_changed, names="value",
            )
            self._viewer.swap_pattern(pat)
            self._viewer.restore_result(result, fitter, elapsed)
            self._update_frac_plot()

        self._fit_all_btn.disabled = False
        n_ok = sum(1 for r in self._results if r is not None and r[0].success)
        self._batch_status.value = (
            f"<b>Batch done:</b> {n_ok}/{n} converged. "
            f"Use the pattern dropdown to inspect individual fits."
        )

    # ------------------------------------------------------------------
    # Phase fraction plot
    # ------------------------------------------------------------------

    def _build_frac_figure(self):
        go = self._go
        fig = go.FigureWidget()
        fig.update_layout(
            height=300,
            width=1100,
            margin=dict(l=60, r=20, t=30, b=50),
            xaxis_title="Pattern",
            yaxis_title="Phase fraction",
            yaxis_range=[0, 1],
            legend=dict(orientation="h", y=1.12),
            template="plotly_white",
        )
        return fig

    def _update_frac_plot(self):
        """Rebuild the phase fraction vs. index plot from stored results."""
        go = self._go
        fig = self._frac_fig

        # Collect data
        indices = []
        labels = []
        frac_data: dict[str, list] = {}  # phase_name -> list of fracs
        for i, entry in enumerate(self._results):
            if entry is None:
                continue
            result, _, _ = entry
            indices.append(i)
            labels.append(self._labels[i])
            for name, frac in result.phase_fractions().items():
                frac_data.setdefault(name, []).append(frac)

        if not indices:
            return

        from ssrl_xrd_tools.gui.widgets.phase_fit_viewer import _PHASE_COLORS
        fig.data = []  # clear existing traces
        for j, (name, fracs) in enumerate(frac_data.items()):
            color = _PHASE_COLORS[j % len(_PHASE_COLORS)]
            fig.add_trace(go.Scatter(
                x=labels,
                y=fracs,
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=2),
                marker=dict(color=color, size=6),
            ))

    # ------------------------------------------------------------------
    # Config save / load
    # ------------------------------------------------------------------

    def _on_save_config(self, _btn):
        path = self._config_name.value.strip()
        if not path:
            self._batch_status.value = (
                "<span style='color:#b00'>Enter a filename.</span>"
            )
            return
        config = self.get_config()
        config.save(path)
        self._batch_status.value = f"Config saved to <code>{path}</code>"

    def _on_load_config(self, _btn):
        """Load config from the path in the text field."""
        path = self._config_name.value.strip()
        if not path:
            self._batch_status.value = (
                "<span style='color:#b00'>Enter a filename to load.</span>"
            )
            return
        p = Path(path)
        if not p.exists():
            self._batch_status.value = (
                f"<span style='color:#b00'>File not found: "
                f"<code>{path}</code></span>"
            )
            return
        self.load_config(p)

    def _on_file_chosen(self, chooser):
        """Callback from ipyfilechooser — populate text field with selection."""
        selected = chooser.selected
        if selected:
            self._config_name.value = str(selected)

    def load_config(self, path: str | Path) -> None:
        """Load a :class:`FitConfig` from JSON and apply to the viewer."""
        from ssrl_xrd_tools.analysis.fitting.batch import FitConfig
        config = FitConfig.load(path)
        self.set_config(config)
        self._batch_status.value = f"Config loaded from <code>{path}</code>"

    # ------------------------------------------------------------------
    # Result store export
    # ------------------------------------------------------------------

    def to_result_store(self) -> "FitResultStore":
        """Export all fitted results as a :class:`FitResultStore`."""
        from ssrl_xrd_tools.analysis.fitting.batch import FitResultStore
        store = FitResultStore()
        for i, entry in enumerate(self._results):
            if entry is None:
                continue
            result, _, elapsed = entry
            store.append(
                result,
                index=i,
                label=self._labels[i],
                elapsed=elapsed,
            )
        return store

    def to_dataframe(self):
        """Shortcut: export results to a pandas DataFrame."""
        return self.to_result_store().to_dataframe()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def viewer(self) -> Any:
        """The underlying :class:`PhaseFitViewer`."""
        return self._viewer

    def _ipython_display_(self):
        from IPython.display import display
        display(self.widget)
