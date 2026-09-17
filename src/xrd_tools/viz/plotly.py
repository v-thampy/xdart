"""Plotly plotting helpers for 1D XRD data and fit results.

These are thin, notebook-friendly helpers built on ``plotly.graph_objects``.
They return plain :class:`plotly.graph_objects.Figure` objects so the
caller decides how to render (``fig.show()``, embed in a FigureWidget,
etc.).  The helpers never mutate their inputs.

Helpers
-------
plot_pattern_fit
    Interactive view of a multi-phase :class:`~xrd_tools.analysis.fitting.phase_fitting.PhaseFitter`
    fit — data, total model, per-phase contributions, amorphous peak,
    background, and residuals.
plot_phase_fractions
    Bar / grouped-bar view of phase fractions across a set of fits
    (e.g. a composition series).
plot_peak_fit
    Individual-peak fit overlay from :class:`~xrd_tools.analysis.fitting.fit.PeakFitResult1D`
    with per-peak components broken out.

The helpers import ``plotly`` lazily so simply importing the ``viz``
package does not require plotly to be installed.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "plot_pattern_fit",
    "plot_phase_fractions",
    "plot_peak_fit",
    "plot_waterfall",
    "plot_peak_fit_frame",
    "plot_thermal_history",
]


# Default colour cycle — matches ``gui.widgets.pattern_viewer``.
_PHASE_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
]


def _import_go():
    """Lazy import of ``plotly.graph_objects`` with a friendly error."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "plotly is required for xrd_tools.viz.plotly. "
            "Install with `conda install -c conda-forge plotly`."
        ) from exc
    return go


def plot_waterfall(
    dataset: Any,
    *,
    intensity_var: str | None = None,
    x_coord: str = "q",
    y_coord: str = "frame",
    log_intensity: bool = False,
    color_percentiles: tuple[float, float] = (1.0, 99.5),
    colorscale: str = "Viridis",
    title: str | None = None,
    height: int = 620,
):
    """Plot intensity against any two one-dimensional Dataset coordinates.

    For example, use ``y_coord="time"`` or ``y_coord="temperature"`` for
    a series indexed by elapsed time or temperature. Coordinate ``long_name``
    and ``units`` attributes supply axis and hover labels, falling back to
    the coordinate name. Intensity dimensions are ordered as (y, x) for
    display; coordinate order and the input Dataset are preserved.

    ``log_intensity`` and ``color_percentiles`` affect only the display.
    No row normalization is applied.
    """
    go = _import_go()
    if intensity_var is None:
        intensity_var = (
            "intensity_normalized"
            if "intensity_normalized" in dataset else "intensity")
    x = dataset.coords[x_coord]
    y = dataset.coords[y_coord]
    intensity = dataset[intensity_var]
    if x.ndim != 1 or y.ndim != 1 or x.dims == y.dims:
        raise ValueError("waterfall coordinates must span two distinct dimensions")
    values = np.asarray(intensity.transpose(y.dims[0], x.dims[0]).values, dtype=float)
    shown = values
    color_title = intensity_var
    if log_intensity:
        positive = values[values > 0]
        floor = float(np.nanpercentile(positive, 0.1)) if positive.size else 1.0
        shown = np.log10(np.clip(values, floor, None))
        color_title = f"log10({intensity_var})"
    finite = shown[np.isfinite(shown)]
    if finite.size:
        zmin, zmax = np.nanpercentile(finite, color_percentiles)
    else:
        zmin = zmax = None
    def axis_label(coordinate, name):
        label = str(coordinate.attrs.get("long_name", name))
        unit = str(coordinate.attrs.get("units", ""))
        return f"{label} ({unit})" if unit else label

    x_label, y_label = axis_label(x, x_coord), axis_label(y, y_coord)
    fig = go.Figure(go.Heatmap(
        x=np.asarray(x.values),
        y=np.asarray(y.values),
        z=shown,
        zmin=zmin,
        zmax=zmax,
        colorscale=colorscale,
        colorbar=dict(title=color_title),
        hovertemplate=(
            f"{x_label}=%{{x:.5g}}<br>{y_label}=%{{y:.6g}}<br>"
            f"{color_title}=%{{z:.5g}}<extra></extra>"),
    ))
    fig.update_layout(
        height=height,
        title=title or f"Waterfall: {intensity_var}",
        xaxis_title=x_label,
        yaxis_title=y_label,
        margin=dict(l=70, r=30, t=55, b=60),
    )
    return fig


def plot_peak_fit_frame(
    fit_dataset: Any,
    fit_index: int,
    *,
    title: str | None = None,
    height: int = 600,
):
    """Interactive data/fit/residual review for one fitted pattern."""
    go = _import_go()
    from plotly.subplots import make_subplots

    pos = int(fit_index)
    q = np.asarray(fit_dataset.coords["q_fit"].values, dtype=float)
    fit = np.asarray(fit_dataset["fit"].isel(fit_pattern=pos).values, dtype=float)
    residual = np.asarray(
        fit_dataset["residual"].isel(fit_pattern=pos).values, dtype=float)
    data = fit + residual
    background = (
        np.asarray(fit_dataset["background"].isel(fit_pattern=pos).values, dtype=float)
        if "background" in fit_dataset else None)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.76, 0.24],
        vertical_spacing=0.04,
    )
    fig.add_trace(go.Scattergl(
        x=q, y=data, mode="lines", name="data", line=dict(color="#222", width=1)),
        row=1, col=1)
    fig.add_trace(go.Scattergl(
        x=q, y=fit, mode="lines", name="fit", line=dict(color="#d62728", width=2)),
        row=1, col=1)
    if background is not None and np.any(np.isfinite(background)):
        fig.add_trace(go.Scattergl(
            x=q, y=background, mode="lines", name="background",
            line=dict(color="#777", width=1, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scattergl(
        x=q, y=residual, mode="lines", name="residual",
        line=dict(color="#1f77b4", width=1), showlegend=False), row=2, col=1)
    for name in sorted(n for n in fit_dataset.data_vars if n.startswith("center_")):
        if "err" in name:
            continue
        center = float(fit_dataset[name].isel(fit_pattern=pos).values)
        if np.isfinite(center):
            fig.add_vline(x=center, line_width=1, line_dash="dot",
                          line_color="#2ca02c", row=1, col=1)
    source_pattern = (
        int(fit_dataset.coords["pattern"].values[pos])
        if "pattern" in fit_dataset.coords else pos)
    ok = bool(fit_dataset["fit_success"].values[pos])
    fig.update_layout(
        height=height,
        title=title or f"Pattern {source_pattern} fit ({'OK' if ok else 'check'})",
        hovermode="x unified",
        margin=dict(l=65, r=25, t=55, b=55),
    )
    fig.update_yaxes(title_text="Intensity", row=1, col=1)
    fig.update_yaxes(title_text="Residual", row=2, col=1)
    fig.update_xaxes(title_text="q", row=2, col=1)
    return fig


def plot_thermal_history(
    dataset: Any,
    *,
    time_coord: str = "time",
    title: str | None = None,
    height: int = 760,
):
    """Plot lattice, inferred temperature, and heating/cooling rate."""
    go = _import_go()
    from plotly.subplots import make_subplots

    time = np.asarray(dataset.coords[time_coord].values, dtype=float)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    if "lattice_A" in dataset:
        for reflection in dataset.coords["reflection"].values:
            y = dataset["lattice_A"].sel(reflection=reflection).values
            fig.add_trace(go.Scattergl(
                x=time, y=y, mode="lines", name=f"a {reflection}"), row=1, col=1)
    if "lattice_mean_A" in dataset:
        fig.add_trace(go.Scattergl(
            x=time, y=dataset["lattice_mean_A"].values, mode="lines",
            name="a combined", line=dict(color="#111", width=2)), row=1, col=1)
    temperature_name = (
        "temperature_smoothed_K"
        if "temperature_smoothed_K" in dataset else "temperature_K")
    fig.add_trace(go.Scattergl(
        x=time, y=dataset[temperature_name].values, mode="lines",
        name="temperature", line=dict(color="#d62728", width=2)), row=2, col=1)
    fig.add_trace(go.Scattergl(
        x=time, y=dataset["temperature_rate_K_per_s"].values, mode="lines",
        name="dT/dt", line=dict(color="#1f77b4", width=1.5)), row=3, col=1)
    time_unit = str(dataset.coords[time_coord].attrs.get("units", ""))
    fig.update_yaxes(title_text="Lattice a (A)", row=1, col=1)
    fig.update_yaxes(title_text="Temperature (K)", row=2, col=1)
    fig.update_yaxes(title_text="Rate (K/s)", row=3, col=1)
    fig.update_xaxes(
        title_text=f"time ({time_unit})" if time_unit else time_coord, row=3, col=1)
    fig.update_layout(
        height=height,
        title=title or "Lattice and thermal history",
        hovermode="x unified",
        margin=dict(l=75, r=25, t=55, b=60),
    )
    return fig


# ---------------------------------------------------------------------------
# Pattern + fit overlay
# ---------------------------------------------------------------------------

def plot_pattern_fit(
    fitter: Any,
    result: Any | None = None,
    *,
    show_phases: bool = True,
    show_amorphous: bool = True,
    show_background: bool = True,
    show_residual: bool = True,
    height: int = 600,
    title: str | None = None,
    log_y: bool = False,
):
    """Overlay a :class:`PhaseFitter` result on its data (interactive plotly).

    Parameters
    ----------
    fitter : PhaseFitter
        The fitter used to produce ``result``.  Its ``x`` / ``y`` /
        ``y_fit`` / ``background`` attributes are used for the data
        trace and the prefit baseline.
    result : MultiPhaseResult or None
        Result from ``fitter.fit()``.  If ``None``, the data-only figure
        is returned.
    show_phases : bool
        Add one line per phase (each shifted onto the same baseline as
        the model so components line up visually).
    show_amorphous : bool
        Overlay the amorphous peak contribution if present.
    show_background : bool
        Overlay the in-fit background if present.
    show_residual : bool
        Add a second panel with ``data − model`` residuals.
    height : int
        Figure height in pixels (includes the residual sub-panel if
        enabled).
    title : str, optional
        Figure title.
    log_y : bool
        Use a log intensity axis on the main panel.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go = _import_go()
    from plotly.subplots import make_subplots

    x = np.asarray(fitter.x, dtype=float)
    y = np.asarray(fitter.y, dtype=float)

    if show_residual and result is not None:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.75, 0.25],
            vertical_spacing=0.04,
        )
        main_row, res_row = 1, 2
    else:
        fig = go.Figure()
        main_row = res_row = None

    def _add(trace, **kw):
        if main_row is not None:
            fig.add_trace(trace, row=main_row, col=1, **kw)
        else:
            fig.add_trace(trace, **kw)

    # --- Data ---
    _add(go.Scatter(
        x=x, y=y, mode="lines", name="data",
        line=dict(color="black", width=1),
    ))

    if result is None:
        fig.update_layout(
            height=height,
            xaxis_title="q (Å⁻¹)",
            yaxis_title="Intensity",
            hovermode="x unified",
            title=title,
        )
        if log_y:
            if main_row is not None:
                fig.update_yaxes(type="log", row=main_row, col=1)
            else:
                fig.update_yaxes(type="log")
        return fig

    params = result.params
    # Prefit baseline: everything the composite sees must be added back
    # to display on the same axis as ``y``.
    prefit = np.asarray(fitter.background, dtype=float)

    # Total model (composite eval already handles prefit via eval_model)
    y_model = fitter.eval_model(params)
    _add(go.Scatter(
        x=x, y=y_model, mode="lines", name="total fit",
        line=dict(color="#e41a1c", width=1.5),
    ))

    # In-fit background
    if show_background and getattr(fitter, "_bg_model", None) is not None:
        bg = fitter.eval_fit_background(params)
        if bg is not None:
            _add(go.Scatter(
                x=x, y=bg + prefit, mode="lines", name="background",
                line=dict(color="#888", width=1, dash="dash"),
            ))

    # Amorphous peak
    if show_amorphous and getattr(fitter, "_amorphous_model", None) is not None:
        am = fitter.eval_amorphous(params)
        if am is not None:
            _add(go.Scatter(
                x=x, y=am + prefit, mode="lines", name="amorphous",
                line=dict(color="#4daf4a", width=1, dash="dot"),
            ))

    # Per-phase contributions
    if show_phases:
        for i, phase in enumerate(fitter.phases):
            y_phase = fitter.eval_phase(i, params)
            color = _PHASE_COLORS[i % len(_PHASE_COLORS)]
            _add(go.Scatter(
                x=x, y=y_phase + prefit, mode="lines",
                name=f"{phase.name}",
                line=dict(color=color, width=1),
            ))

    if log_y:
        fig.update_yaxes(type="log", row=main_row, col=1)

    # --- Residual ---
    if show_residual:
        residual = y - y_model
        fig.add_trace(
            go.Scatter(
                x=x, y=residual, mode="lines", name="residual",
                line=dict(color="#555", width=1),
                showlegend=False,
            ),
            row=res_row, col=1,
        )
        fig.add_hline(y=0, line_color="#aaa", line_width=0.5, row=res_row, col=1)
        fig.update_yaxes(title_text="residual", row=res_row, col=1)
        fig.update_xaxes(title_text="q (Å⁻¹)", row=res_row, col=1)
        fig.update_yaxes(title_text="Intensity", row=main_row, col=1)
    else:
        fig.update_layout(
            xaxis_title="q (Å⁻¹)",
            yaxis_title="Intensity",
        )

    fig.update_layout(
        height=height,
        hovermode="x unified",
        title=title,
        legend=dict(font=dict(size=10)),
        margin=dict(l=60, r=20, t=50 if title else 30, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Phase fractions across a sequence of fits
# ---------------------------------------------------------------------------

def plot_phase_fractions(
    results: Sequence[Any],
    labels: Sequence[str] | None = None,
    *,
    title: str | None = None,
    height: int = 400,
    stacked: bool = True,
):
    """Bar chart of phase fractions across several fits.

    Parameters
    ----------
    results : sequence of MultiPhaseResult
        Fits whose ``phase_fractions()`` are plotted.
    labels : sequence of str, optional
        X-axis labels for each fit.  Default: ``"#0"``, ``"#1"``, ...
    title : str, optional
        Figure title.
    height : int
        Figure height in pixels.
    stacked : bool
        If True (default), draw a stacked bar per fit summing to 1.
        If False, draw grouped bars (one bar per phase per fit).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go = _import_go()
    if not results:
        raise ValueError("plot_phase_fractions: results is empty.")

    if labels is None:
        labels = [f"#{i}" for i in range(len(results))]
    if len(labels) != len(results):
        raise ValueError(
            f"len(labels)={len(labels)} does not match len(results)={len(results)}"
        )

    # Union of all phase names (preserve order of first occurrence)
    phase_names: list[str] = []
    rows: list[dict[str, float]] = []
    for r in results:
        frac = r.phase_fractions()
        rows.append(frac)
        for name in frac:
            if name not in phase_names:
                phase_names.append(name)

    fig = go.Figure()
    for p_idx, name in enumerate(phase_names):
        color = _PHASE_COLORS[p_idx % len(_PHASE_COLORS)]
        ys = [float(row.get(name, 0.0)) for row in rows]
        fig.add_trace(go.Bar(
            x=list(labels), y=ys, name=name,
            marker_color=color,
        ))

    fig.update_layout(
        barmode="stack" if stacked else "group",
        height=height,
        yaxis_title="phase fraction",
        xaxis_title="",
        title=title,
        legend=dict(font=dict(size=10)),
        margin=dict(l=60, r=20, t=50 if title else 30, b=50),
    )
    if stacked:
        fig.update_yaxes(range=[0.0, 1.0])
    return fig


# ---------------------------------------------------------------------------
# Independent peak fit overlay
# ---------------------------------------------------------------------------

def plot_peak_fit(
    x: np.ndarray,
    y: np.ndarray,
    peak_result: Any,
    *,
    show_components: bool = True,
    title: str | None = None,
    height: int = 450,
):
    """Overlay an independent multi-peak fit (from :func:`fit_peaks`).

    Parameters
    ----------
    x, y : ndarray
        The fitted data (used for the points trace).
    peak_result : PeakFitResult1D
        The result returned by :func:`xrd_tools.analysis.fitting.fit.fit_peaks`.
    show_components : bool
        If True (default), draw each peak component individually in
        addition to the total fit.
    title : str, optional
    height : int

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go = _import_go()
    from plotly.subplots import make_subplots

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Raw lmfit result
    fit_result = peak_result.fit_result
    best = np.asarray(fit_result.best_fit, dtype=float)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.04,
    )

    fig.add_trace(
        go.Scatter(x=x, y=y, mode="lines", name="data",
                   line=dict(color="black", width=1)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=best, mode="lines", name="total fit",
                   line=dict(color="#e41a1c", width=1.5)),
        row=1, col=1,
    )

    if show_components:
        try:
            components = fit_result.eval_components(x=x)
        except Exception:  # pragma: no cover — lmfit edge case
            components = {}
        for j, (cname, cvals) in enumerate(components.items()):
            color = _PHASE_COLORS[j % len(_PHASE_COLORS)]
            fig.add_trace(
                go.Scatter(
                    x=x, y=np.asarray(cvals), mode="lines",
                    name=cname.rstrip("_"),
                    line=dict(color=color, width=1, dash="dot"),
                ),
                row=1, col=1,
            )

    residual = y - best
    fig.add_trace(
        go.Scatter(x=x, y=residual, mode="lines", name="residual",
                   line=dict(color="#555", width=1), showlegend=False),
        row=2, col=1,
    )
    fig.add_hline(y=0, line_color="#aaa", line_width=0.5, row=2, col=1)

    fig.update_yaxes(title_text="Intensity", row=1, col=1)
    fig.update_yaxes(title_text="residual", row=2, col=1)
    fig.update_xaxes(title_text="q (Å⁻¹)", row=2, col=1)
    fig.update_layout(
        height=height,
        hovermode="x unified",
        title=title,
        legend=dict(font=dict(size=10)),
        margin=dict(l=60, r=20, t=50 if title else 30, b=50),
    )
    return fig
