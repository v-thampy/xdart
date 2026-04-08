"""Matplotlib plotting helpers for 1D and 2D XRD data.

These are thin convenience wrappers around ``Axes.errorbar`` and
``Axes.imshow`` that handle the boilerplate most notebooks repeat
(labels, log scales, legends, colorbars) while staying fully
transparent: everything except the bookkeeping is passed through to
matplotlib via ``**kwargs``, and the returned handles (line, image,
colorbar) can be further tweaked by the caller.

These helpers never call ``plt.show()`` or ``plt.figure()`` on their
own — the caller is expected to own the ``Figure``/``Axes`` layout —
so they compose cleanly inside multi-panel figures and inside GUI
embeddings.

Examples
--------
Simple 1D overlay in a notebook::

    import matplotlib.pyplot as plt
    from ssrl_xrd_tools.viz.mpl import plot_1d

    fig, ax = plt.subplots()
    plot_1d(ax, q, I, yerr=sigma, label='raw',
            attrs=dict(xlabel=r'Q (Å$^{-1}$)', ylabel='Intensity',
                       yscale='log'))
    plot_1d(ax, q, I_fit, label='fit', fmt='-', clear=False)

2D detector image with an attached colorbar::

    from ssrl_xrd_tools.viz.mpl import plot_image

    fig, ax = plt.subplots()
    im, cb = plot_image(ax, img, cb_label='Counts',
                        attrs=dict(xlabel='px', ylabel='px',
                                   title='Pilatus frame 42'),
                        cmap='viridis', vmin=0, vmax=500)
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.container import ErrorbarContainer
from matplotlib.image import AxesImage
from mpl_toolkits.axes_grid1 import make_axes_locatable


__all__ = ["plot_1d", "plot_image"]


# ---------------------------------------------------------------------------
# 1D line / scatter
# ---------------------------------------------------------------------------

def plot_1d(
    ax: Axes,
    x,
    y,
    yerr=None,
    *,
    label: str | None = None,
    fmt: str = ".",
    attrs: dict[str, Any] | None = None,
    clear: bool = False,
    legend_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ErrorbarContainer:
    """Plot ``y`` vs ``x`` on ``ax`` with optional error bars and axis setup.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes. Not created by this function.
    x, y : array-like
        The 1D data to plot.
    yerr : array-like, optional
        Uncertainties on ``y``, passed to :meth:`Axes.errorbar` as-is.
    label : str, optional
        Series label. When non-None, a legend is drawn.
    fmt : str
        Matplotlib format string passed through to ``ax.errorbar``.
        Use ``'.'`` for scatter, ``'-'`` for a line, ``'o-'`` for both.
    attrs : dict, optional
        Keyword arguments forwarded to :meth:`Axes.set` in one shot.
        Typical keys: ``xlabel``, ``ylabel``, ``title``, ``xscale``,
        ``yscale``, ``xlim``, ``ylim``.
    clear : bool, default False
        If True, call ``ax.cla()`` before plotting. Default is False so
        multiple calls compose naturally; notebooks that want
        overwrite-in-place behaviour can pass ``clear=True``.
    legend_kwargs : dict, optional
        Keyword arguments forwarded to :meth:`Axes.legend` when a legend
        is drawn. Defaults to ``dict(frameon=True, markerscale=1)``.
    **kwargs
        Additional keyword arguments forwarded to :meth:`Axes.errorbar`
        (e.g. ``color``, ``alpha``, ``capsize``, ``elinewidth``).

    Returns
    -------
    matplotlib.container.ErrorbarContainer
        The container returned by ``ax.errorbar``. Callers that want to
        tweak the line or caps afterwards can use its ``lines`` attribute.
    """
    if attrs is None:
        attrs = {}

    if clear:
        ax.cla()

    container = ax.errorbar(x, y, yerr=yerr, fmt=fmt, label=label, **kwargs)
    if attrs:
        ax.set(**attrs)
    if label is not None:
        lk = {"frameon": True, "markerscale": 1}
        if legend_kwargs:
            lk.update(legend_kwargs)
        ax.legend(**lk)
    return container


# ---------------------------------------------------------------------------
# 2D image / detector frame
# ---------------------------------------------------------------------------

def plot_image(
    ax: Axes,
    data,
    *,
    attrs: dict[str, Any] | None = None,
    cax: Axes | None = None,
    cb_label: str | None = None,
    cb_fontsize: int = 14,
    clear: bool = False,
    origin: str = "lower",
    aspect: str | float = "auto",
    **kwargs: Any,
) -> tuple[AxesImage, Colorbar | None]:
    """Display a 2D array on ``ax`` with optional colorbar and axis setup.

    Defaults are chosen for detector / reciprocal-space maps:

    * ``origin='lower'`` so that row-0 is plotted at the bottom, which
      is what almost every XRD/RSM convention expects.
    * ``aspect='auto'`` so rectangular Q–χ maps are not squished into a
      square by matplotlib's default ``'equal'`` aspect ratio. Pass
      ``aspect='equal'`` explicitly for raw square-pixel detector
      images where pixel aspect matters.
    * ``ax.grid(False)`` after plotting — matplotlib's default grid
      draws on top of the image and is almost never what you want.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    data : 2D array-like
        Image data passed through to :meth:`Axes.imshow`.
    attrs : dict, optional
        Forwarded to :meth:`Axes.set` (xlabel, ylabel, title, xlim, ylim…).
    cax : matplotlib.axes.Axes or None, optional
        Axes into which the colorbar should be drawn. If ``None`` and a
        colorbar is desired (either ``cb_label`` was given, or you
        simply want the default colorbar), one is auto-created to the
        right of ``ax`` via ``mpl_toolkits.axes_grid1.make_axes_locatable``.
        Pass ``cax=False`` to suppress the colorbar entirely.
    cb_label : str, optional
        Label for the colorbar. When provided, a colorbar is always drawn.
    cb_fontsize : int, default 14
        Font size for the colorbar label.
    clear : bool, default False
        If True, call ``ax.clear()`` before plotting.
    origin : {'lower', 'upper'}, default 'lower'
        Forwarded to ``ax.imshow``. Default is ``'lower'`` (see above).
    aspect : {'auto', 'equal'} or float, default 'auto'
        Forwarded to ``ax.imshow``. Default ``'auto'`` is suitable for
        Q–χ maps; use ``'equal'`` for raw detector images.
    **kwargs
        Additional keyword arguments forwarded to :meth:`Axes.imshow`
        (e.g. ``cmap``, ``vmin``, ``vmax``, ``norm``, ``extent``,
        ``interpolation``).

    Returns
    -------
    (image, colorbar) : tuple
        The :class:`~matplotlib.image.AxesImage` handle and the
        :class:`~matplotlib.colorbar.Colorbar` instance (or ``None`` if
        no colorbar was drawn). Callers can tweak the colour limits
        with ``image.set_clim(lo, hi)`` later.
    """
    if attrs is None:
        attrs = {}

    if clear:
        ax.clear()

    im = ax.imshow(data, origin=origin, aspect=aspect, **kwargs)
    if attrs:
        ax.set(**attrs)
    ax.grid(False)

    # Colorbar handling
    cb: Colorbar | None = None
    want_cb = cax is not False and (cax is not None or cb_label is not None)
    if want_cb:
        if cax is None or cax is True:
            # Auto-create a right-side colorbar axis that shares the
            # parent's height and doesn't steal space from ``ax``.
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.08)
        cb = plt.colorbar(im, cax=cax, use_gridspec=True)
        if cb_label is not None:
            cb.set_label(cb_label, fontsize=cb_fontsize)

    return im, cb
