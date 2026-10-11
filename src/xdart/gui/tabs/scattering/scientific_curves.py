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
