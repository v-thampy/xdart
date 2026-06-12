"""Interactive 2D detector image viewer using ipywidgets + plotly.

Uses ``plotly.graph_objects.FigureWidget`` for fast in-place updates and
``ipywidgets`` for compact, native VS Code-friendly controls.

Features
--------
- Logarithmic / linear colormap scaling
- Adjustable percentile clipping (vmin / vmax)
- Colormap selector
- Automatic downsampling for fast rendering
- ``.figure_widget`` exposes the underlying plotly FigureWidget for embedding
- ``.widget`` is the assembled controls + figure VBox
"""
from __future__ import annotations

from typing import Any

import numpy as np
import ipywidgets as widgets
import plotly.graph_objects as go
from IPython.display import display

__all__ = ["ImageViewer"]

# Maximum displayed image dimension (downsampled before sending to plotly).
# Plotly's Heatmap is fast at ~600x600; larger sizes start to drag.
_MAX_DISPLAY_PIXELS = 600


def _downsample(img: np.ndarray, max_dim: int = _MAX_DISPLAY_PIXELS) -> np.ndarray:
    """Strided downsample so the largest dimension <= ``max_dim``."""
    if img is None:
        return img
    h, w = img.shape[:2]
    factor = max(1, max(h, w) // max_dim)
    return img[::factor, ::factor] if factor > 1 else img


class ImageViewer:
    """Interactive 2D image viewer (ipywidgets + plotly).

    Parameters
    ----------
    image : ndarray, optional
        2D detector image.
    title : str
        Plot title.
    log_scale : bool
        Use logarithmic colour scale.
    colormap : str
        Plotly colour scale name (e.g. ``"Viridis"``, ``"Plasma"``).
    height : int
        Figure height in pixels.

    Attributes
    ----------
    figure_widget : plotly.graph_objects.FigureWidget
        The underlying plotly figure — mutate it directly for advanced use.
    widget : ipywidgets.VBox
        The assembled controls + figure widget. Display this (or just the
        instance — ``ImageViewer`` implements ``_ipython_display_``).
    """

    def __init__(
        self,
        image: np.ndarray | None = None,
        title: str = "Detector image",
        log_scale: bool = True,
        colormap: str = "Viridis",
        height: int = 480,
    ):
        self._image = image
        self._title = title

        # Plotly FigureWidget — initialised with a placeholder heatmap so the
        # data trace exists and can be updated in place.
        self.figure_widget = go.FigureWidget(
            data=[
                go.Heatmap(
                    z=np.zeros((2, 2)),
                    colorscale=colormap,
                    colorbar=dict(title="I", thickness=12),
                    hovertemplate="x=%{x}<br>y=%{y}<br>I=%{z:.3g}<extra></extra>",
                )
            ],
            layout=go.Layout(
                title=dict(text=title, font=dict(size=13)),
                width=620,
                height=height,
                xaxis=dict(title="x (pixels)"),
                yaxis=dict(title="y (pixels)", scaleanchor="x"),
                margin=dict(l=55, r=20, t=40, b=50),
            ),
        )

        # --- Controls ---
        self._log_toggle = widgets.ToggleButton(
            value=log_scale,
            description="Log",
            tooltip="Logarithmic colour scale",
            layout=widgets.Layout(width="70px"),
        )
        self._cmap = widgets.Dropdown(
            options=["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Greys", "Hot"],
            value=colormap,
            description="Cmap:",
            layout=widgets.Layout(width="180px"),
        )
        self._vmin_pct = widgets.FloatSlider(
            value=1.0, min=0.0, max=50.0, step=0.5,
            description="Min %:",
            continuous_update=False,
            readout_format=".1f",
            layout=widgets.Layout(width="240px"),
        )
        self._vmax_pct = widgets.FloatSlider(
            value=99.0, min=50.0, max=100.0, step=0.5,
            description="Max %:",
            continuous_update=False,
            readout_format=".1f",
            layout=widgets.Layout(width="240px"),
        )
        self._info = widgets.HTML("<b>Shape:</b> —")

        # Wire up observers
        for w in (self._log_toggle, self._cmap, self._vmin_pct, self._vmax_pct):
            w.observe(self._on_change, names="value")

        # Compact control row
        controls = widgets.HBox(
            [self._log_toggle, self._cmap, self._vmin_pct, self._vmax_pct],
            layout=widgets.Layout(flex_flow="row wrap", align_items="center"),
        )

        self.widget = widgets.VBox([self._info, controls, self.figure_widget])

        if image is not None:
            self._render()

    # ---- public API ----
    @property
    def image(self) -> np.ndarray | None:
        return self._image

    @image.setter
    def image(self, img: np.ndarray | None) -> None:
        self._image = img
        self._render()

    @property
    def title(self) -> str:
        return self._title

    @title.setter
    def title(self, t: str) -> None:
        self._title = t
        self.figure_widget.layout.title.text = t

    # ---- internals ----
    def _on_change(self, change: Any = None) -> None:
        self._render()

    def _render(self) -> None:
        if self._image is None:
            self._info.value = "<b>Shape:</b> —"
            return

        data = _downsample(self._image).astype(float)
        self._info.value = (
            f"<b>Shape:</b> {self._image.shape}  "
            f"<span style='color:#888'>(displayed: {data.shape})</span>"
        )

        positive = data[data > 0]
        if positive.size == 0:
            positive = np.array([1.0])

        if self._log_toggle.value:
            vmin = float(np.log10(max(np.nanpercentile(positive, self._vmin_pct.value), 1e-1)))
            vmax = float(np.log10(max(np.nanpercentile(data, self._vmax_pct.value), 1.0)))
            z = np.where(data > 0, np.log10(data), 0.0)
            ctitle = "log\u2081\u2080(I)"
        else:
            vmin = float(np.nanpercentile(data, self._vmin_pct.value))
            vmax = float(np.nanpercentile(data, self._vmax_pct.value))
            z = data
            ctitle = "I"

        with self.figure_widget.batch_update():
            tr = self.figure_widget.data[0]
            tr.z = z
            tr.zmin = vmin
            tr.zmax = vmax
            tr.colorscale = self._cmap.value
            tr.colorbar.title.text = ctitle
            self.figure_widget.layout.title.text = self._title

    def _ipython_display_(self) -> None:
        display(self.widget)
