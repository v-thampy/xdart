"""Small shared curve-presentation helpers; no result or operation ownership."""

import numpy as np
from matplotlib import colormaps
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore


class PlotCursorReadout:
    """Keep the Int cursor label and scene-to-view behavior on one plot."""

    def __init__(self, curve):
        self.curve = curve
        self.label = pg.LabelItem(justify="right")
        self.label.setParentItem(curve.getPlotItem())
        self.label.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(-14, 8))
        self.proxy = pg.SignalProxy(
            signal=curve.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_moved,
        )

    def mouse_moved(self, positions):
        position = positions
        if type(positions) is tuple:
            if not positions:
                return
            position = positions[0]
        if not isinstance(position, QtCore.QPointF):
            return
        plot = self.curve.getPlotItem()
        if not plot.sceneBoundingRect().contains(position):
            self.label.setText("")
            self.curve.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            return
        point = plot.vb.mapSceneToView(position)
        self.label.setText(f"x={point.x():.2f}, y={point.y():.2e}")
        self.curve.setCursor(QtCore.Qt.CursorShape.CrossCursor)


def scaled_intensity(values: np.ndarray, scale: str) -> np.ndarray:
    if scale == "Linear":
        return values
    result = np.asarray(values, dtype=float).copy()
    finite = result[np.isfinite(result)]
    if not finite.size:
        return result
    if scale == "Log":
        minimum = float(np.min(finite))
        if minimum < 1.0:
            result -= minimum - 1.0
        return np.log10(result)
    if scale == "Sqrt":
        return np.sign(result) * np.sqrt(np.abs(result))
    raise ValueError(f"unsupported intensity scale: {scale}")


def scaled_intensity_label(label: str, scale: str) -> str:
    if scale == "Log":
        return f"Log {label}"
    if scale == "Sqrt":
        return f"√{label}"
    return label


def overlay_step(arrays, offset_percent: float) -> float:
    minimum = np.inf
    maximum = -np.inf
    for values in arrays:
        finite = values[np.isfinite(values)]
        if finite.size:
            minimum = min(minimum, float(np.min(finite)))
            maximum = max(maximum, float(np.max(finite)))
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        return 0.0
    return float(offset_percent) * (maximum - minimum) / 100.0


_TRACE_COLORS = (
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    (188, 189, 34), (23, 190, 207),
)


def trace_colors(name: str, count: int) -> tuple:
    if name != "Default" and name in colormaps:
        positions = [.65] if count == 1 else np.linspace(.15, .85, count)
        return tuple(tuple(int(channel) for channel in color)
                     for color in colormaps[name](positions, bytes=True))
    return tuple(_TRACE_COLORS[index % len(_TRACE_COLORS)] for index in range(count))
