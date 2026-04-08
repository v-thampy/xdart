"""Plotting helpers for ssrl_xrd_tools.

This package collects lightweight, backend-specific plotting helpers
that are independent of any GUI. The goal is to cut the amount of
``ax.errorbar / ax.set / ax.legend`` boilerplate needed in notebooks
and in the analysis modules themselves.

Submodules
----------
mpl
    Matplotlib helpers for static, publication-style figures.
    See :func:`ssrl_xrd_tools.viz.mpl.plot_1d` and
    :func:`ssrl_xrd_tools.viz.mpl.plot_image`.

Planned
-------
plotly
    Interactive helpers built on plotly for notebook exploration
    (hover tooltips, zoom/pan, linked axes). Not yet implemented —
    will be added alongside the interactive XRD viewer work.
"""
from __future__ import annotations

from ssrl_xrd_tools.viz.mpl import plot_1d, plot_image

__all__ = ["plot_1d", "plot_image"]
