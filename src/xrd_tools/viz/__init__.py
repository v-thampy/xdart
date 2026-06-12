"""Plotting helpers for xrd_tools.

This package collects lightweight, backend-specific plotting helpers
that are independent of any GUI. The goal is to cut the amount of
``ax.errorbar / ax.set / ax.legend`` boilerplate needed in notebooks
and in the analysis modules themselves.

Submodules
----------
mpl
    Matplotlib helpers for static, publication-style figures.
    See :func:`xrd_tools.viz.mpl.plot_1d` and
    :func:`xrd_tools.viz.mpl.plot_image`.

plotly
    Interactive helpers for notebook exploration built on
    ``plotly.graph_objects``.  See
    :func:`xrd_tools.viz.plotly.plot_pattern_fit`,
    :func:`plot_phase_fractions`, and :func:`plot_peak_fit`.
"""
from __future__ import annotations

from xrd_tools.viz.mpl import plot_1d, plot_image
from xrd_tools.viz.plotly import (
    plot_pattern_fit,
    plot_phase_fractions,
    plot_peak_fit,
)

__all__ = [
    "plot_1d",
    "plot_image",
    "plot_pattern_fit",
    "plot_phase_fractions",
    "plot_peak_fit",
]
