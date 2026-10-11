"""The shared Figure Curve item owner for keyed and normal scientific panes."""
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore


def add_curve_item(plot, curve, color, *, xy=None):
    points = curve.xy if xy is None else xy
    style = (QtCore.Qt.PenStyle.DashLine if curve.role == 'predicted'
             else QtCore.Qt.PenStyle.DotLine if curve.role == 'tie'
             else QtCore.Qt.PenStyle.SolidLine)
    item = plot.plot(points[:, 0], points[:, 1],
                     pen=pg.mkPen(color, width=2, style=style), name=curve.name)
    item.setToolTip(f'{curve.role}: {curve.tooltip}')
    item.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
    item.setAcceptHoverEvents(False)
    return item


def add_text_item(plot, text, color, *, xy=None):
    """Shared passive Figure label, in the caller's displayed coordinates."""
    item = pg.TextItem(text.text, color=color)
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
    """One passive item per reference family: vertical lines at view x positions, each spanning the visible y range.

    A plain group, not a plot data item, so trace bookkeeping (``listDataItems``) never sees it; its single
    curve child draws every line and is re-spanned whenever the view range changes.
    """

    def __init__(self, positions, *, color, family=None, tooltip=''):
        super().__init__()
        self.reference_positions = tuple(float(position) for position in positions)
        self.lines = pg.PlotCurveItem(pen=pg.mkPen(color, width=1), connect='finite')
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
        x = np.repeat(np.asarray(self.reference_positions, dtype=float), 3)
        x[2::3] = np.nan
        self.lines.setData(x, np.tile((low, high, np.nan), len(self.reference_positions)), connect='finite')


def draw_reference_lines(plot, positions, *, color=(240, 200, 90), family=None, tooltip=''):
    """All of one family's reference positions as one passive item (one object per phase)."""
    item = ReferenceLines(positions, color=color, family=family, tooltip=tooltip)
    plot.addItem(item, ignoreBounds=True)
    item.span()
    return item
