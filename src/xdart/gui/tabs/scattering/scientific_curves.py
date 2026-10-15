"""The shared Figure Curve item owner for keyed and normal scientific panes."""
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

from xdart.gui.widgets.scientific_plot import HOVER_PX


def hkl_label(hkl):
    """``(110)`` for single-digit non-negative indices, else ``(1 -1 10)``."""
    hkl = tuple(int(index) for index in hkl)
    if all(0 <= index <= 9 for index in hkl):
        return "(" + "".join(map(str, hkl)) + ")"
    return "(" + " ".join(map(str, hkl)) + ")"


def reference_entry(value, color):
    """One hovered reference line's tooltip entry (C7): a swatch in its phase's legend colour, then its hkl families,
    its d-spacing and, for a CIF phase, its relative intensity; a calibrant line, which has no hkl, shows its
    d-spacing alone."""
    swatch = '<span style="color:#%02x%02x%02x">■</span>' % tuple(int(channel) for channel in color[:3])
    if not value.hkls:
        return f"{swatch} d = {value.d_spacing:.3f} Å"
    text = f"{swatch} {' '.join(hkl_label(hkl) for hkl in value.hkls)} · d {value.d_spacing:.3f} Å"
    return text if value.intensity is None else f"{text} · {value.intensity:.0f} %"


def add_curve_item(plot, curve, color, *, xy=None):
    points = curve.xy if xy is None else xy
    # an instrument schematic's detector-arm circles dash as a prediction does, its label leaders dot (LT-007)
    style = (QtCore.Qt.PenStyle.DashLine if curve.role in ('predicted', 'detector')
             else QtCore.Qt.PenStyle.DotLine if curve.role in ('tie', 'leader')
             else QtCore.Qt.PenStyle.SolidLine)
    item = plot.plot(points[:, 0], points[:, 1],
                     pen=pg.mkPen(color, width=2, style=style), name=curve.name)
    item.setToolTip(f'{curve.role}: {curve.tooltip}')
    item.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
    item.setAcceptHoverEvents(False)
    return item


def add_text_item(plot, text, color, *, xy=None):
    """Shared passive Figure label, in the caller's displayed coordinates, placed by its anchor."""
    item = pg.TextItem(text.text, color=color, anchor=getattr(text, 'anchor', (0, 0)))
    item.setPos(*(text.xy if xy is None else xy))
    item.setToolTip(text.tooltip)
    item.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
    item.setAcceptHoverEvents(False)
    plot.addItem(item)
    return item


def draw_marker(plot, position, *, color=(240, 200, 90), family=None, tooltip=''):
    """One passive vertical reference marker, shared by fits and overlays."""
    line = pg.InfiniteLine(float(position), angle=90, pen=pg.mkPen(color, width=1))
    line.setToolTip(tooltip)
    line.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
    line.setAcceptHoverEvents(False)
    if family is not None:
        line.setData(0, family)
    plot.addItem(line, ignoreBounds=True)
    return line


class ReferenceLines(pg.ItemGroup):
    """One passive item per reference family: vertical lines at view x positions, rising from the view's bottom.

    A plain group, not a plot data item, so trace bookkeeping (``listDataItems``) never sees it; its single
    curve child draws every line and is re-spanned whenever the view range changes. A line whose value carries a
    relative intensity (a CIF phase's) stops at that fraction of the displayed height, in displayed coordinates and
    so identically in linear and log views (C8); any other spans the view. ``hover_entries`` names the lines under
    the cursor (C7), for the plot's ``PlotCursorReadout``.
    """

    def __init__(self, positions, *, color, family=None, tooltip='', values=()):
        super().__init__()
        self.reference_positions = tuple(float(position) for position in positions)
        #: each line's detached value (C7), aligned with ``reference_positions``; empty: no hover entries
        self.line_values = tuple(values) if len(values) == len(self.reference_positions) else ()
        self._fractions = np.array([1. if value.intensity is None else max(0., min(1., value.intensity / 100.))
                                    for value in self.line_values] or [1.] * len(self.reference_positions))
        self.color = color
        self.lines = pg.PlotCurveItem(pen=pg.mkPen(color, width=1.4, style=QtCore.Qt.PenStyle.DashLine),
                                      connect='finite')  # dashed, apart from measured traces (D4)
        self._x = np.asarray(self.reference_positions, dtype=float)
        self.addItem(self.lines)
        for item in (self, self.lines):
            item.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
            item.setAcceptHoverEvents(False)
        if family is not None:
            self.setData(0, family)
        self.set_tooltip(tooltip)

    def set_tooltip(self, tooltip):
        self.setToolTip(tooltip)
        self.lines.setToolTip(tooltip)

    def viewRangeChanged(self, *_args):
        super().viewRangeChanged()
        self.span()

    def span(self):
        box = self.getViewBox()
        if box is None:
            return
        low, high = box.viewRange()[1]
        x = np.repeat(self._x, 3)
        x[2::3] = np.nan
        y = np.empty_like(x)
        y[0::3], y[1::3], y[2::3] = low, low + self._fractions * (high - low), np.nan
        self.lines.setData(x, y, connect='finite')

    def hover_entries(self, point, pixel):
        """Tooltip entries for the drawn lines within ``HOVER_PX`` screen pixels of view ``point`` (``pixel``: the
        view's size of one screen pixel), in line order: the distance to each drawn stick as a segment, the image
        index's convention (HELP-HOVER-ENDPOINT-REACH), so a corner beyond a stick's end is not within reach."""
        box = self.getViewBox()
        if not self.line_values or box is None or not self.isVisible():
            return ()
        low, high = box.viewRange()[1]
        tops = low + self._fractions * (high - low)
        across = (self._x - point.x()) / abs(pixel[0])
        along = np.maximum(0., np.maximum(point.y() - tops, low - point.y())) / abs(pixel[1])
        under = np.hypot(across, along) <= HOVER_PX
        return tuple(reference_entry(self.line_values[index], self.color) for index in np.flatnonzero(under))


class AngularSupportCurve(pg.PlotDataItem):
    """One shown trace's angular coverage on the 1-D strip (COVERAGE-SUPPORT-20261013): the share of its χ window
    that unmasked pixels reach in each radial bin, in percent over the trace's own displayed axis and colour.
    ``angular_support`` is the profile drawn, never copied or resampled; hover names the bin's degrees."""

    def __init__(self, x, profile, *, name, color):
        self.angular_support = profile
        self._x = np.array(x, dtype=float)
        fill = pg.mkColor(color)
        fill.setAlpha(40)
        super().__init__(self._x, 100.0 * profile.fraction, name=name, pen=pg.mkPen(color=color, width=1.2),
                         fillLevel=0.0, brush=fill, connect='finite')
        self.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)

    def hover_entries(self, point, _context=None):
        """The entry for the bin under view ``point``'s x (D1: the bin mean; D2: degrees of the window), none
        beyond the drawn bins."""
        finite = np.isfinite(self._x)
        if not self.isVisible() or not finite.any():
            return ()
        distance = np.where(finite, np.abs(self._x - point.x()), np.inf)
        index = int(np.argmin(distance))
        neighbours = self._x[max(index - 1, 0):index + 2]
        half_bin = 0.5 * float(np.nanmax(np.abs(np.diff(neighbours)))) if neighbours.size > 1 else 0.0
        degrees = float(self.angular_support.supported_deg[index])
        if distance[index] > half_bin or not np.isfinite(degrees):
            return ()
        window = self.angular_support.window_deg
        return (f"{self.name()}: {100.0 * degrees / window:.1f} % of {window:g}° ({degrees:.2f}° over this bin)",)


def draw_reference_lines(plot, positions, *, color=(240, 200, 90), family=None, tooltip='', values=()):
    """All of one family's reference positions as one passive item (one object per phase); ``values`` are the
    lines' detached reflections (C7, C8)."""
    item = ReferenceLines(positions, color=color, family=family, tooltip=tooltip, values=values)
    plot.addItem(item, ignoreBounds=True)
    item.span()
    return item
