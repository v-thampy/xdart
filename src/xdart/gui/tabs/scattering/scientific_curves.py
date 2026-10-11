"""The shared Figure Curve item owner for keyed and normal scientific panes."""
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
