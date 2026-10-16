"""The shared fit display, beside ``ScientificView``: a fit's data and model above, its residuals below.

Codex's Stitch/Int reuse audit (2026-10-01, R2): "Extend the scientific display owner for fit/data/residual
panels, so both Stitch diagnostics and Int's analysis fit windows can use the same display capability".
A passive view, as ``ScientificView`` is: a ``FitProjection`` in (``reconcile``), drawn on the 1-D plot's
own conventions (``RectViewBox`` box zoom, middle-click to show all and pyqtgraph's menu with Export on the
right button; the seaborn style; the cursor readout on the upper panel). Each series names its role, and
the role decides the look, here and nowhere else: data points, a data line, the fit, the background, a
component, groups of points coloured as the 1-D plot colours its traces (``trace_colors``), rejected
points, residuals. Zoom, export and the readout are pyqtgraph's.

A fit window's fit range is the view's too (FIT-TOOLS-2026-10, Codex's design review F7: "a fit-range region is a
``FitView`` extension, not a page's plot"): ``set_fit_range`` shows a range over the upper panel, kept over every
redraw, and ``rangeChanged`` says where the user dragged it. ``fit_result_series`` is a displayed fit result
(``DisplayedPeakFitResult``, ``DisplayedPhaseFitResult``) as the view's series, for every host's fit window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.gui_utils import GridAxisItem, RectViewBox
from xdart.gui.themes import apply_seaborn_plot_style
from xdart.gui.widgets.scientific_plot import PlotCursorReadout, trace_colors

_DATA = (210, 210, 220)
_FIT = (189, 147, 249)
_BACKGROUND = (130, 200, 160)
_RESIDUAL = (230, 133, 151)
_REJECTED = (230, 110, 110)
_ZERO = (130, 130, 140)
_MARKER = (240, 200, 90)
_RANGE = (189, 147, 249, 30)

#: Each role's look: points ("o" filled, "x" crosses) or a line, its colour (None: the theme's
#: foreground; "group": the 1-D plot's trace colour for the series' group) and size or width.
_ROLES = {
    "data": ("o", _DATA, 4),
    "data_line": ("line", _DATA, 1),
    "fit": ("line", _FIT, 2),
    "background": ("dash", _BACKGROUND, 1.3),
    "component": ("dash", None, 1),
    "group": ("o", "group", 5),
    "rejected": ("x", _REJECTED, 7),
    "residual": ("line", _RESIDUAL, 1),
    "residual_points": ("o", _RESIDUAL, 5),
}


@dataclass(frozen=True, slots=True)
class FitSeries:
    """One drawn series: its x and y, its role (the look), its legend name (None: not in the legend) and,
    for the ``group`` role, its group's index in the 1-D plot's trace colours."""

    x: np.ndarray
    y: np.ndarray
    role: str = "data"
    name: str | None = None
    group: int = 0

    def __post_init__(self) -> None:
        if self.role not in _ROLES:
            raise ValueError(f"unknown fit series role {self.role!r}")


@dataclass(frozen=True, slots=True)
class FitPanel:
    """One panel: its series in drawing order, its axis labels and title, a zero line, and vertical markers
    (peak positions)."""

    series: tuple[FitSeries, ...] = ()
    x_label: str = ""
    y_label: str = ""
    title: str | None = None
    zero_line: bool = False
    markers: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class FitProjection:
    """What the view shows: the upper panel (data and model), the lower one (residuals), and the summary."""

    upper: FitPanel = FitPanel()
    lower: FitPanel = FitPanel()
    summary: str = ""


def draw_series(plot, series: FitSeries, *, group_count: int = 1):
    """Draw one series on ``plot`` (a ``PlotWidget`` or ``PlotItem``) in its role's look; return the item.
    The one place a fit series' look is decided, for the view and for any plot drawn through it."""
    kind, colour, size = _ROLES[series.role]
    if colour == "group":
        colour = trace_colors("Default", max(group_count, series.group + 1))[series.group]
    x, y = np.asarray(series.x, dtype=float), np.asarray(series.y, dtype=float)
    if kind == "o":
        return plot.plot(x, y, pen=None, symbol="o", symbolSize=size, symbolBrush=colour,
                         symbolPen=None, name=series.name)
    if kind == "x":
        return plot.plot(x, y, pen=None, symbol="x", symbolSize=size, symbolPen=pg.mkPen(colour),
                         name=series.name)
    style = QtCore.Qt.PenStyle.DashLine if kind == "dash" else QtCore.Qt.PenStyle.SolidLine
    pen = pg.mkPen(width=size, style=style) if colour is None else pg.mkPen(colour, width=size, style=style)
    return plot.plot(x, y, pen=pen, name=series.name)


def draw_marker(plot, position: float):
    """A vertical marker line (a peak position) on ``plot``."""
    from .scientific_curves import draw_marker as shared_marker
    return shared_marker(plot, position, color=_MARKER)


def draw_zero_line(plot):
    """The residual zero line on ``plot``."""
    return plot.addLine(y=0, pen=pg.mkPen(_ZERO, width=1))


def fit_result_series(x, result, components=()):
    """A displayed fit result as the view's series: its fit, background and named ``components`` (name, values)
    above, the peak markers, and its residual below; every array on the fitted ``x``."""
    upper = []
    if result.fit is not None:
        upper.append(FitSeries(x, result.fit, "fit", "fit"))
    if result.background is not None:
        upper.append(FitSeries(x, result.background, "background", "background"))
    upper += [FitSeries(x, values, "component", name) for name, values in components]
    markers = (() if result.marker_positions is None
               else tuple(float(position) for position in result.marker_positions))
    lower = () if result.residual is None else (FitSeries(x, result.residual, "residual"),)
    return tuple(upper), markers, lower


def _draw_panel(plot: pg.PlotWidget, panel: FitPanel) -> None:
    plot.clear()
    if panel.zero_line:
        draw_zero_line(plot)
    for side, label in (("bottom", panel.x_label), ("left", panel.y_label)):
        axis = plot.getPlotItem().getAxis(side)
        axis.setLabel(label)
        axis.showLabel(bool(label))  # an unlabelled axis keeps no empty label row
    plot.setTitle(panel.title)
    groups = 1 + max((series.group for series in panel.series if series.role == "group"), default=0)
    for series in panel.series:
        draw_series(plot, series, group_count=groups)
    for position in panel.markers:
        draw_marker(plot, position)


class FitView(QtWidgets.QWidget):
    """A fit's summary, an upper panel (data and model) and a lower one (residuals), from a
    ``FitProjection``. ``plot`` and ``residuals`` are the panels' plot widgets, read by their consumers'
    tests; only ``reconcile`` draws on them, and the fit range (``set_fit_range``) stays over its redraws."""

    #: The user moved the fit range on the upper panel: its (low, high), sorted. ``set_fit_range`` does not emit it.
    rangeChanged = QtCore.Signal(float, float)

    def __init__(self, parent=None, *, name: str = "fitView", summary: bool = True,
                 stretch: tuple[int, int] = (1, 1), upper_min_height: int | None = None,
                 lower_max_height: int | None = None, link_x: bool = False) -> None:
        super().__init__(parent)
        self.setObjectName(name)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.summary = None
        if summary:
            self.summary = QtWidgets.QLabel()
            self.summary.setObjectName(f"{name}Summary")
            self.summary.setWordWrap(True)
            self.summary.setTextInteractionFlags(self.summary.textInteractionFlags()
                                                 | QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            layout.addWidget(self.summary)
        self.plot = pg.PlotWidget(viewBox=RectViewBox(), axisItems={"left": GridAxisItem("left")})
        apply_seaborn_plot_style(self.plot.getPlotItem(), grid=True)
        self.plot.addLegend(offset=(-10, 10))
        self._readout = PlotCursorReadout(self.plot)
        self.residuals = pg.PlotWidget(viewBox=RectViewBox(), axisItems={"left": GridAxisItem("left")})
        apply_seaborn_plot_style(self.residuals.getPlotItem(), grid=True)
        if upper_min_height is not None:
            self.plot.setMinimumHeight(upper_min_height)
        if lower_max_height is not None:
            self.residuals.setMaximumHeight(lower_max_height)
        if link_x:
            self.residuals.setXLink(self.plot)
        layout.addWidget(self.plot, stretch[0])
        layout.addWidget(self.residuals, stretch[1])
        self._projection = FitProjection()
        # the fit range: one region, behind the series, re-added after each redraw clears the upper panel
        self._range: tuple[float, float] | None = None
        self._setting_range = False
        self._region = pg.LinearRegionItem(brush=_RANGE)
        self._region.setZValue(-10)
        self._region.hide()
        self._region.sigRegionChanged.connect(self._region_moved)
        self.plot.addItem(self._region)

    @property
    def projection(self) -> FitProjection:
        """The projection shown."""
        return self._projection

    def reconcile(self, projection: FitProjection) -> None:
        """Show ``projection``: both panels redrawn from it, and the summary."""
        if type(projection) is not FitProjection:
            raise TypeError("FitView shows a FitProjection")
        if self.summary is not None:
            self.summary.setText(projection.summary)
        _draw_panel(self.plot, projection.upper)
        self.plot.addItem(self._region)  # the fit range, shown or not, outlives the redraw
        _draw_panel(self.residuals, projection.lower)
        self._projection = projection

    @property
    def fit_range(self) -> tuple[float, float] | None:
        """The fit range shown (low, high), or None."""
        return self._range

    def set_fit_range(self, bounds: tuple[float, float] | None, *,
                      limits: tuple[float, float] | None = None) -> None:
        """Show the fit range ``bounds`` (low, high) over the upper panel, draggable within ``limits`` (the data's
        extent; None: anywhere), the range clamped to them; None hides it. It stays over every redraw until it is
        set again, and setting it does not emit ``rangeChanged``."""
        self._setting_range = True
        try:
            if bounds is None:
                self._range = None
                self._region.hide()
                return
            self._region.setBounds([None, None] if limits is None else sorted(float(value) for value in limits))
            self._region.setRegion(tuple(sorted(float(value) for value in bounds)))
            self._range = tuple(sorted(float(value) for value in self._region.getRegion()))
            self._region.show()
        finally:
            self._setting_range = False

    def _region_moved(self) -> None:
        if self._setting_range or self._range is None:
            return
        self._range = tuple(sorted(float(value) for value in self._region.getRegion()))
        self.rangeChanged.emit(*self._range)
