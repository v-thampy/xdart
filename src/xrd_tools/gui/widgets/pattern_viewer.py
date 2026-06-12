"""Interactive 1D powder pattern viewer using ipywidgets + plotly.

Uses ``plotly.graph_objects.FigureWidget`` for fast in-place updates and
``ipywidgets`` for native VS Code-friendly controls.

Features
--------
- Overlay multiple patterns with labels and legend
- Vertical offset stacking (waterfall view)
- Phase marker lines from PhaseModel objects
- Linear / log intensity toggle
- ``.figure_widget`` exposes the underlying plotly FigureWidget
- ``.widget`` is the assembled controls + figure VBox
"""
from __future__ import annotations

from typing import Any

import numpy as np
import ipywidgets as widgets
import plotly.graph_objects as go
from IPython.display import display

__all__ = ["PatternViewer"]

# Default colour cycle for phase markers
_PHASE_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
]


class PatternViewer:
    """Interactive 1D diffraction pattern viewer (ipywidgets + plotly).

    Parameters
    ----------
    patterns : list, optional
        List of either ``IntegrationResult1D`` or ``(q, I, label)`` tuples.
    phase_models : list, optional
        Crystal phases whose peak positions are overlaid as vertical lines.
    height : int
        Figure height in pixels.

    Attributes
    ----------
    figure_widget : plotly.graph_objects.FigureWidget
        The underlying plotly figure.
    widget : ipywidgets.VBox
        Assembled controls + figure widget.
    """

    def __init__(
        self,
        patterns: list[Any] | None = None,
        phase_models: list[Any] | None = None,
        height: int = 450,
    ):
        self._patterns: list[tuple[np.ndarray, np.ndarray, str]] = []
        self._phase_models = phase_models or []

        self.figure_widget = go.FigureWidget(
            layout=go.Layout(
                width=820,
                height=height,
                xaxis=dict(title="q (Å⁻¹)"),
                yaxis=dict(title="Intensity"),
                margin=dict(l=60, r=20, t=30, b=50),
                legend=dict(font=dict(size=9), x=1.0, xanchor="right", y=1.0),
                hovermode="x unified",
                showlegend=True,
            ),
        )

        # --- Controls ---
        self._log_toggle = widgets.ToggleButton(
            value=False, description="Log I",
            tooltip="Logarithmic intensity axis",
            layout=widgets.Layout(width="80px"),
        )
        self._offset = widgets.FloatSlider(
            value=0.0, min=0.0, max=10.0, step=0.1,
            description="Offset:",
            continuous_update=False,
            readout_format=".1f",
            layout=widgets.Layout(width="280px"),
        )
        self._show_phases = widgets.ToggleButton(
            value=True, description="Phases",
            tooltip="Show phase marker lines",
            layout=widgets.Layout(width="90px"),
        )

        for w in (self._log_toggle, self._offset, self._show_phases):
            w.observe(self._on_change, names="value")

        controls = widgets.HBox(
            [self._log_toggle, self._offset, self._show_phases],
            layout=widgets.Layout(flex_flow="row wrap", align_items="center"),
        )

        self.widget = widgets.VBox([controls, self.figure_widget])

        if patterns:
            self.set_patterns(patterns)

    # ---- public API ----
    def set_patterns(self, patterns: list[Any]) -> None:
        """Set patterns from a list of (q, I, label) tuples or IntegrationResult1D."""
        from xrd_tools.core.containers import IntegrationResult1D

        self._patterns = []
        for i, p in enumerate(patterns):
            if isinstance(p, IntegrationResult1D):
                self._patterns.append((p.radial, p.intensity, f"Pattern {i}"))
            elif isinstance(p, (tuple, list)) and len(p) >= 2:
                label = p[2] if len(p) > 2 else f"Pattern {i}"
                self._patterns.append((np.asarray(p[0]), np.asarray(p[1]), str(label)))
        self._render()

    def set_phase_models(self, phases: list[Any]) -> None:
        self._phase_models = phases
        self._render()

    # ---- internals ----
    def _on_change(self, change: Any = None) -> None:
        self._render()

    def _render(self) -> None:
        n_patterns = len(self._patterns)
        n_existing = len(self.figure_widget.data)

        # Phase marker shapes
        shapes: list[dict] = []
        if self._show_phases.value and self._phase_models and n_patterns > 0:
            q_lo = float(min(q.min() for q, _, _ in self._patterns))
            q_hi = float(max(q.max() for q, _, _ in self._patterns))
            for pidx, phase in enumerate(self._phase_models):
                color = _PHASE_COLORS[pidx % len(_PHASE_COLORS)]
                for pk in phase.peaks:
                    if pk.intensity > 2.0 and q_lo <= pk.q <= q_hi:
                        shapes.append(dict(
                            type="line",
                            x0=pk.q, x1=pk.q,
                            y0=0, y1=1, yref="paper",
                            line=dict(color=color, width=0.7, dash="dash"),
                        ))

        with self.figure_widget.batch_update():
            # Remove excess traces (subset assignment is allowed)
            if n_existing > n_patterns:
                self.figure_widget.data = self.figure_widget.data[:n_patterns]

            # Update existing traces in place + add any new ones
            for idx, (q, intensity, label) in enumerate(self._patterns):
                y = intensity.copy().astype(float)
                if self._log_toggle.value:
                    y = np.where(y > 0, np.log10(y), 0.0)
                y = y + idx * self._offset.value

                if idx < len(self.figure_widget.data):
                    tr = self.figure_widget.data[idx]
                    tr.x = q
                    tr.y = y
                    tr.name = label
                else:
                    self.figure_widget.add_trace(
                        go.Scattergl(
                            x=q, y=y, mode="lines", name=label,
                            line=dict(width=1),
                        )
                    )

            self.figure_widget.layout.shapes = shapes
            self.figure_widget.layout.yaxis.title.text = (
                "log\u2081\u2080(I)" if self._log_toggle.value else "Intensity"
            )

    def _ipython_display_(self) -> None:
        display(self.widget)
