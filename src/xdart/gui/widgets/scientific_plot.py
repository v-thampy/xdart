"""Small shared curve-presentation helpers; no result or operation ownership."""

import numpy as np
from matplotlib import colormaps
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

#: how near (screen pixels) the cursor must come to a hoverable reference line or ring (C7)
HOVER_PX = 4.


class HoverTooltip(QtCore.QObject):
    """One display owner's hover tooltip over a graphics view (C7): shown at a scene position, hidden when it no
    longer applies (only while it is the tooltip shown), and kept over Qt's own item tooltip while shown."""

    def __init__(self, view):
        super().__init__(view)
        self._view, self._text = view, ""
        viewport = getattr(view, "viewport", None)
        self._viewport = viewport() if callable(viewport) else view
        self._viewport.installEventFilter(self)

    @property
    def text(self) -> str:
        return self._text

    def show(self, scene_position, text: str) -> None:
        local = self._view.mapFromScene(scene_position)
        self._text = text
        QtWidgets.QToolTip.showText(self._viewport.mapToGlobal(local), text, self._viewport)

    def hide(self) -> None:
        # Ours whether or not Qt reports it visible yet: a show still settling is hidden too.
        shown = QtWidgets.QToolTip.text() == self._text
        if self._text and shown:
            QtWidgets.QToolTip.hideText()
        self._text = ""

    def eventFilter(self, watched, event):  # noqa: N802 - Qt override
        return bool(self._text) and event.type() == QtCore.QEvent.Type.ToolTip


def data_curve_style(color):
    """Universal Int style for measured 1-D data, including keyed projections."""
    return dict(pen=pg.mkPen(color=color, width=1.4, style=QtCore.Qt.PenStyle.SolidLine),
                symbol="o", symbolSize=4, symbolBrush=color, symbolPen=color, connect="finite")


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
        #: the plot's hover tooltip: entries from items that offer them (ReferenceLines' reflections, C7)
        self.hover = HoverTooltip(curve)

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
            self.hover.hide()
            return
        point = plot.vb.mapSceneToView(position)
        self.label.setText(f"x={point.x():.2f}, y={point.y():.2e}")
        self.curve.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        pixel = plot.vb.viewPixelSize()
        entries = [entry for item in plot.items if hasattr(item, "hover_entries")
                   for entry in item.hover_entries(point, pixel)]
        if entries:
            self.hover.show(position, "<br>".join(entries))
        else:
            self.hover.hide()

    def clear_hover(self):
        """The hovered drawing changed or left: its tooltip goes."""
        self.hover.hide()


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


#: Reference-phase colours (D4): Okabe–Ito's seven and a grey that reads on both themes. None is a Default trace
#: colour (``_TRACE_COLORS``); a user-chosen trace colormap carries no such promise.
REFERENCE_COLORS = (
    (230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66), (0, 114, 178),
    (213, 94, 0), (204, 121, 167), (96, 96, 96),
)


def reference_colors(count: int) -> tuple:
    """The first ``count`` reference-phase colours, repeating past eight."""
    return tuple(REFERENCE_COLORS[index % len(REFERENCE_COLORS)] for index in range(count))


def trace_colors(name: str, count: int) -> tuple:
    if name != "Default" and name in colormaps:
        positions = [.65] if count == 1 else np.linspace(.15, .85, count)
        return tuple(tuple(int(channel) for channel in color)
                     for color in colormaps[name](positions, bytes=True))
    return tuple(_TRACE_COLORS[index % len(_TRACE_COLORS)] for index in range(count))
