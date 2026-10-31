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

NEXT-128-PEAK-FIT (the maintainer: the tool windows follow the main window): an axis given its unit is labelled as
Int labels it (``_axis_presentation``: q (Å⁻¹)); ``top_bar`` puts Int's display top bar above the upper panel (a title,
Int's colour maps and Log: ``color_map_combo``/``log_toggle``), where Log draws the upper panel with Int's own
transform applied once to all its series, so they stay aligned, and the colour map colours the fit's components; and
``components`` makes a toggle (``components_toggle``) that shows or hides them, which the window mounts where it
wants it (the trend's row). Fit components are solid lines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.gui_utils import GridAxisItem, RectViewBox
from xdart.gui.themes import apply_seaborn_plot_style
from xdart.gui.widgets.scientific_plot import PlotCursorReadout, scaled_intensity, scaled_intensity_label, trace_colors

_DATA = (210, 210, 220)
_FIT = (189, 147, 249)
_BACKGROUND = (130, 200, 160)
#: PHASE-ANALYSIS-128: a phase fit's own components, a background refined with the phases and an amorphous peak
_FIT_BACKGROUND = (110, 170, 230)
_AMORPHOUS = (235, 150, 95)
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
    "fit_background": ("dash", _FIT_BACKGROUND, 1.3),
    "amorphous": ("dash", _AMORPHOUS, 1.3),
    "component": ("line", None, 1.2),
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
    #: the x axis's unit: when given, the axis is labelled as Int labels it (``_axis_presentation``)
    x_unit: str = ""


@dataclass(frozen=True, slots=True)
class FitProjection:
    """What the view shows: the upper panel (data and model), the lower one (residuals), and the summary."""

    upper: FitPanel = FitPanel()
    lower: FitPanel = FitPanel()
    summary: str = ""


def draw_series(plot, series: FitSeries, *, group_count: int = 1, color=None):
    """Draw one series on ``plot`` (a ``PlotWidget`` or ``PlotItem``) in its role's look (``color``: a colour map's
    colour in place of the role's); return the item. The one place a fit series' look is decided, for the view and
    for any plot drawn through it."""
    kind, colour, size = _ROLES[series.role]
    if color is not None:
        colour = color
    elif colour == "group":
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
    """A displayed fit result as the view's series: its fit, background, a phase fit's fitted background and
    amorphous peak when it has them, and named ``components`` (name, values) above (a peak fit's own peaks when none
    are given: ``peak_component_curves``), the peak markers, and its residual below; every array on the fitted
    ``x``."""
    if not components and getattr(result, "model", "") and result.parameter_names:
        from xrd_tools.analysis.fitting.fit import peak_component_curves
        components = peak_component_curves(result.model, result.parameter_names, result.parameter_values, x)
    upper = []
    if result.fit is not None:
        upper.append(FitSeries(x, result.fit, "fit", "fit"))
    if result.background is not None:
        upper.append(FitSeries(x, result.background, "background", "background"))
    for role, name in (("fit_background", "fitted background"), ("amorphous", "amorphous")):
        values = getattr(result, role, None)
        if values is not None:
            upper.append(FitSeries(x, values, role, name))
    upper += [FitSeries(x, values, "component", name) for name, values in components]
    markers = (() if result.marker_positions is None
               else tuple(float(position) for position in result.marker_positions))
    lower = () if result.residual is None else (FitSeries(x, result.residual, "residual"),)
    return tuple(upper), markers, lower


def _draw_panel(plot: pg.PlotWidget, panel: FitPanel, *, colors=None) -> None:
    plot.clear()
    if panel.zero_line:
        draw_zero_line(plot)
    for side, label in (("bottom", panel.x_label), ("left", panel.y_label)):
        axis = plot.getPlotItem().getAxis(side)
        if side == "bottom" and panel.x_unit:
            from .shell_widgets import _axis_presentation
            label, unit = _axis_presentation(panel.x_label, panel.x_unit)
            axis.enableAutoSIPrefix(False)  # as Int's axes: a unit is never rescaled
            axis.setLabel(label, units=unit)
        else:
            axis.setLabel(label, units="")
        axis.showLabel(bool(label))  # an unlabelled axis keeps no empty label row
    plot.setTitle(panel.title)
    groups = 1 + max((series.group for series in panel.series if series.role == "group"), default=0)
    for index, series in enumerate(panel.series):
        draw_series(plot, series, group_count=groups, color=None if colors is None else colors.get(index))
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
                 lower_max_height: int | None = None, link_x: bool = False, top_bar: bool = False,
                 components: bool | None = None) -> None:
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
        self.title = self.color_map = self.log_scale = self.components_toggle = None
        if top_bar:  # Int's display top bar (ScientificView's): a title, its colour maps and Log
            from .shell_widgets import color_map_combo, log_toggle
            bar = QtWidgets.QHBoxLayout()
            self.title = QtWidgets.QLabel("")
            self.title.setObjectName(f"{name}Title")
            self.title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.title.setToolTip("The trace shown and fitted.")
            self.color_map = color_map_combo()
            self.color_map.setToolTip("The colour map the fit's components are drawn in, as Int colours its traces.")
            self.log_scale = log_toggle()
            self.log_scale.setToolTip("Draw the data and the fit on a log intensity scale (Int's Log); the "
                                      "residuals stay linear.")
            bar.addWidget(self.title, 1)
            bar.addWidget(self.color_map)
            bar.addWidget(self.log_scale)
            layout.addLayout(bar)
            self.color_map.currentTextChanged.connect(lambda _name: self._redraw_upper())
            self.log_scale.toggled.connect(lambda _on: self._redraw_upper())
        if components is not None:  # unparented until its window mounts it (the trend's row)
            self.components_toggle = QtWidgets.QPushButton("Components")
            self.components_toggle.setObjectName(f"{name}Components")
            self.components_toggle.setCheckable(True)
            self.components_toggle.setChecked(bool(components))
            self.components_toggle.setToolTip("Show each fitted component (every peak, or every phase) as its own "
                                              "line over the data.")
            self.components_toggle.toggled.connect(lambda _on: self._redraw_upper())
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
        self._projection = projection
        self._redraw_upper()
        _draw_panel(self.residuals, projection.lower)

    def set_title(self, text: str) -> None:
        """The top bar's title: the trace shown."""
        if self.title is not None:
            self.title.setText(text)

    def _redraw_upper(self) -> None:
        """The upper panel as the top bar and the components toggle present it: components shown or not, coloured
        from the colour map, and Int's Log applied once to every series (one shift: they stay aligned)."""
        panel = self._projection.upper
        shown = (self.components_toggle is None or self.components_toggle.isChecked())
        series = tuple(item for item in panel.series if shown or item.role != "component")
        colors = None
        if self.color_map is not None:
            parts = [index for index, item in enumerate(series) if item.role == "component"]
            palette = trace_colors(self.color_map.currentText(), max(len(parts), 1))
            colors = {index: palette[position] for position, index in enumerate(parts)}
        y_label = panel.y_label
        if self.log_scale is not None and self.log_scale.isChecked() and series:
            joined = scaled_intensity(np.concatenate([np.asarray(item.y, dtype=float) for item in series]), "Log")
            bounds = np.cumsum([np.asarray(item.y).size for item in series])[:-1]
            series = tuple(FitSeries(item.x, values, item.role, item.name, item.group)
                           for item, values in zip(series, np.split(joined, bounds)))
            y_label = scaled_intensity_label(panel.y_label, "Log")
        from dataclasses import replace
        _draw_panel(self.plot, replace(panel, series=series, y_label=y_label), colors=colors)
        self.plot.addItem(self._region)  # the fit range, shown or not, outlives the redraw

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
