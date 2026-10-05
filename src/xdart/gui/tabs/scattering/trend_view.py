"""The shared trend, beside ``FitView``: fitted parameters across results, a second family on a right-hand axis.

FIT-TOOLS-2026-10 (Codex's design review F4, F7): Int's Peak Fit and Phase Fit built this row themselves
(``ParamTrendMixin``: their own plot, combos, check box and CSV dialog); it is now this one owner, which every fit
window mounts. The Scan Plot mounts its passive generic-series profile (``SeriesProjection``); its right-hand
axis uses the same ``attach_right_axis`` and ``add_right_series`` owners.

A row is one result's flat parameters (``center_0``, ``fwhm_1``, ``frac_TiN`` …), keyed by its identity, so a
re-fit updates its row and two results at the same x stay two rows. Its x is the host's: a frame index (Int), a
scan column's value, or None when that result has none (it is listed in the CSV, counted in the note, not drawn);
or, with ``order``, its place among the results. Parameters group into families (one curve per peak: the
``peak_fit_util`` helpers); the Track family is drawn on the left axis, a Right family on its own right-hand axis
(dashed, "(R)" in the legend), one series each unless Overlay draws them all. Families in the fitted pattern's
coordinate are labelled with its unit (``set_value_unit``), the others "Intensity". Save CSV… writes the rows
through the shared Browse rule (``file_dialogs``).
"""
from __future__ import annotations

import csv
import logging
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.analysis.peak_fit_util import CURVE_PENS, FAMILY_LABELS, X_UNIT_FAMILIES, group_families
from xdart.gui.gui_utils import RectViewBox
from xdart.gui.themes import apply_seaborn_plot_style, plot_font
from xdart.gui.widgets.controls_panel import FormRow

logger = logging.getLogger(__name__)

#: Save CSV's file types.
CSV_FILTER = "CSV files (*.csv)"


def attach_right_axis(plot_widget):
    """Attach a right-hand Y axis backed by a linked ViewBox sharing the x range. Returns ``(view_box, axis)``; the
    axis starts hidden (``set_right_axis_visible``). Add right-axis curves with ``add_right_series`` and clear them
    with ``view_box.clear()``."""
    plot_item = plot_widget.getPlotItem()
    vb = pg.ViewBox()
    plot_item.showAxis("right")
    plot_item.scene().addItem(vb)
    right_axis = plot_item.getAxis("right")
    right_axis.linkToView(vb)
    vb.setXLink(plot_item)
    right_axis.setVisible(False)
    # The right ViewBox shares the x range and AUTO-RANGES its y to fit the right-axis series on every redraw.
    # Leaving it independently mouse/menu interactive is the classic linked-ViewBox footgun: a stray drag/wheel would
    # pan-or-zoom (and latch off autorange on) just this axis, so the right series silently drifts off-scale. Its own
    # mouse handling is off: the user interacts with the primary (left) view; x stays linked, y keeps tracking the data.
    vb.setMouseEnabled(x=False, y=False)
    vb.setMenuEnabled(False)

    def _sync_geometry():
        vb.setGeometry(plot_item.vb.sceneBoundingRect())
        vb.linkedViewChanged(plot_item.vb, vb.XAxis)

    plot_item.vb.sigResized.connect(_sync_geometry)
    _sync_geometry()
    return vb, right_axis


def set_right_axis_visible(view_box, axis, visible, label=None):
    """Show/hide the right axis and its ViewBox; set its label when shown."""
    axis.setVisible(bool(visible))
    view_box.setVisible(bool(visible))
    if visible and label is not None:
        axis.setLabel(label)


def add_right_series(view_box, legend, x, y, *, pen, name, symbol="o", symbol_size=4, symbol_brush=None):
    """Add one curve to the right-axis ``view_box`` and a ``"… (R)"`` entry to the main ``legend`` (which cannot
    track items outside its PlotItem); return the ``PlotDataItem``."""
    item = pg.PlotDataItem(x, y, pen=pen, name=name, symbol=symbol, symbolSize=symbol_size,
                           symbolBrush=symbol_brush if symbol_brush is not None else pen.color())
    view_box.addItem(item)
    if legend is not None:
        legend.addItem(item, f"{name} (R)")
    return item


@dataclass(frozen=True, slots=True)
class TrendRow:
    """One result on the trend: its identity (the row's key), its x (None: unavailable) and its flat parameters."""

    identity: str
    x: float | None
    params: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class PlotSeries:
    """An already prepared 1-D series, independently assigned to the left or right axis."""

    name: str
    x: object
    y: object
    right: bool = False


@dataclass(frozen=True, slots=True)
class SeriesProjection:
    """A passive generic-series view: science and column choices belong to its host."""

    series: tuple[PlotSeries, ...] = ()
    x_label: str = ""
    left_label: str = "value"
    right_label: str = "value (right)"
    log_left: bool = False
    current_index: int | None = None


class TrendView(QtWidgets.QWidget):
    """Fitted parameters across results (the module docstring): the Track/Overlay/Right rows and Save CSV… over a
    plot styled as Int's 1-D plot, with a right-hand axis."""

    #: A line for the host's status: where the CSV was saved, or that it could not be.
    messageChanged = QtCore.Signal(str)
    #: Original source rows hit by the gesture; multiple rows are ambiguous.
    pointRowsClicked = QtCore.Signal(object)

    def __init__(self, parent=None, *, name: str = "trendView", x_label: str = "Frame", x_name: str = "frame",
                 order: bool = False, save_title: str = "Save parameters vs frame",
                 save_name: str = "fit_params_vs_frame.csv", min_height: int = 150,
                 series_only: bool = False, identity_column: str | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(name)
        self._x_name, self._order = x_name, bool(order)
        self._save_title, self._save_name = save_title, save_name
        self._series_only = bool(series_only)
        self._identity_column = identity_column
        self._rows: dict[str, TrendRow] = {}
        self._families: tuple[str, ...] = ()
        self._value_unit = ""
        self._point_series = []
        self._frame_markers = []
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        bar = QtWidgets.QHBoxLayout()
        self.family_row = FormRow(label="Track", path=("trend", "family"), value="", kind="combo")
        self.overlay_row = FormRow(label="Overlay", path=("trend", "overlay"), value=False, kind="bool")
        self.right_row = FormRow(label="Right", path=("trend", "right_family"), value="", kind="combo")
        #: the family on the left axis, the one on the right axis (None: off) and the Overlay toggle
        self.family, self.right_family = self.family_row.editor, self.right_row.editor
        self.overlay = self.overlay_row.editor
        self.family.clear()
        self.right_family.clear()
        self.right_family.addItem("None", None)
        self.family.setToolTip("Which fitted parameter to plot across the results")
        self.overlay.setToolTip("Plot every series in the selected parameter on one axis, not just the first")
        self.right_family.setToolTip("Overlay a second parameter family against a right-hand axis (None = off) — "
                                     "for families of very different magnitude")
        self.save = QtWidgets.QPushButton("Save CSV…")
        self.save.setToolTip("Export the parameters table")
        for row in (self.family_row, self.overlay_row, self.right_row):
            bar.addWidget(row)
        bar.addStretch(1)
        bar.addWidget(self.save)
        layout.addLayout(bar)
        self.note = QtWidgets.QLabel("")
        self.note.setObjectName(f"{name}Note")
        self.note.hide()
        layout.addWidget(self.note)
        self.plot = pg.PlotWidget(viewBox=RectViewBox())
        apply_seaborn_plot_style(self.plot.getPlotItem(), grid=True)
        self.plot.setMinimumHeight(min_height)
        self.legend = self.plot.addLegend(offset=(-10, 10))
        self.plot.setLabel("bottom", x_label)
        self.right_view_box, self.right_axis = attach_right_axis(self.plot)
        layout.addWidget(self.plot, 1)
        self.family.currentIndexChanged.connect(self.redraw)
        self.right_family.currentIndexChanged.connect(self.redraw)
        self.overlay.toggled.connect(self.redraw)
        self.save.clicked.connect(self._save_csv)
        self._set_enabled(False)
        if self._series_only:
            for widget in (self.family_row, self.overlay_row, self.right_row, self.save, self.note):
                widget.hide()
            self.legend.setLabelTextSize(f"{plot_font().pointSize() + 2}pt")

    def reconcile_series(self, projection: SeriesProjection) -> None:
        """Draw prepared arbitrary series through the same plot/right-axis owner as fitted trends.
        The right axis stays linear when the left is logarithmic. Empty and all-nonfinite series are omitted."""
        if not self._series_only:
            raise RuntimeError("reconcile_series needs the series_only profile")
        if type(projection) is not SeriesProjection:
            raise TypeError("series must be a SeriesProjection")
        prepared = []
        for series in projection.series:
            if type(series) is not PlotSeries:
                raise TypeError("series must contain PlotSeries values")
            x, y = np.asarray(series.x, dtype=float), np.asarray(series.y, dtype=float)
            if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
                raise ValueError("series axes must be equal-length 1-D arrays")
            if projection.current_index is not None and (
                    type(projection.current_index) is not int
                    or not 0 <= projection.current_index < len(x)):
                raise ValueError("current index must be an original row within every series")
            prepared.append((series, x, y))
        for box, marker in self._frame_markers:
            box.removeItem(marker)
        self._frame_markers = []
        self.plot.clear()
        self.right_view_box.clear()
        self.legend.clear()
        self._point_series = []
        any_left = any_right = False
        for index, (series, x, y) in enumerate(prepared):
            if not (np.isfinite(x).any() and np.isfinite(y).any()):
                continue
            color = CURVE_PENS[index % len(CURVE_PENS)]
            pen = pg.mkPen(color, width=3)
            if series.right:
                item = add_right_series(self.right_view_box, self.legend, x, y, pen=pen, name=series.name,
                                 symbol="o", symbol_size=7, symbol_brush=color)
                any_right = True
            else:
                item = self.plot.plot(x, y, pen=pen, name=series.name, symbol="o", symbolSize=7, symbolBrush=color)
                any_left = True
            item.setData(x, y, data=list(range(len(x))))
            self._point_series.append(item)
            item.sigPointsClicked.connect(self._point_click)
            row = projection.current_index
            if (row is not None and np.isfinite(x[row]) and np.isfinite(y[row])
                    and (series.right or not projection.log_left or y[row] > 0)):
                marker = pg.PlotDataItem([x[row]], [y[row]], pen=None, symbol="o", symbolSize=14,
                    symbolPen=pg.mkPen("#ffdf00", width=3), symbolBrush=pg.mkBrush(0, 0, 0, 0), data=[row])
                marker.setObjectName("currentFrameMarker")
                marker.setToolTip("Selected image's exact source row; click a statistic point to request its frame.")
                marker.setZValue(20)
                box = self.right_view_box if series.right else self.plot.getPlotItem()
                box.addItem(marker, ignoreBounds=True)
                self._frame_markers.append((box, marker))
                marker.sigPointsClicked.connect(self._point_click)
        self.plot.setLabel("bottom", projection.x_label)
        self.plot.setLabel("left", projection.left_label if any_left else "")
        set_right_axis_visible(self.right_view_box, self.right_axis, any_right, projection.right_label)
        self.plot.setLogMode(x=False, y=projection.log_left)
        self.right_axis.setLogMode(False)

    def _point_click(self, _item, points, event):
        # Scatter accepts the first curve's event. Check all visible symbol footprints
        # to detect cross-curve ambiguity, in each axis's actual scene coordinates.
        rows = {point.data() for point in points}
        for curve in self._point_series:
            scatter = curve.scatter
            if scatter.isVisible() and scatter.isEnabled():
                rows.update(point.data() for point in scatter.pointsAt(scatter.mapFromScene(event.scenePos())))
        self.pointRowsClicked.emit(tuple(sorted(row for row in rows if type(row) is int)))

    @property
    def rows(self) -> tuple[TrendRow, ...]:
        """The rows, in the order their results were first set."""
        return tuple(self._rows.values())

    def set_value_unit(self, unit: str) -> None:
        """The fitted pattern's coordinate (its unit or label), which the centre and width families are in."""
        if unit != self._value_unit:
            self._value_unit = unit
            self.redraw()

    def set_row(self, identity: str, x: float | None, params: Mapping[str, float]) -> None:
        """One result's parameters: its row is replaced (a re-fit), or added; a result with no parameters adds
        nothing."""
        if not params:
            return
        self._rows[str(identity)] = TrendRow(str(identity), None if x is None else float(x), dict(params))
        self._sync_families()
        self.redraw()
        self._set_enabled(True)

    def clear(self) -> None:
        """No rows: the families and the plot go, Right back to None."""
        self._rows = {}
        self._families = ()
        for combo, none in ((self.family, False), (self.right_family, True)):
            blocker = QtCore.QSignalBlocker(combo)
            combo.clear()
            if none:
                combo.addItem("None", None)
            del blocker
        self._set_enabled(False)
        self.redraw()

    def table(self) -> tuple[tuple[str, ...], tuple[float | None, ...], dict[str, list[float]]]:
        """The rows as a table: their identities and x in the trend's order (by x, those without one last; or the
        results' order), and each parameter's column (NaN where a result has none), parameters in the order met."""
        ordered = self._ordered()
        keys: list[str] = []
        for _x, row in ordered:
            keys += [key for key in row.params if key not in keys]
        columns = {key: [float(row.params.get(key, np.nan)) for _x, row in ordered] for key in keys}
        return tuple(row.identity for _x, row in ordered), tuple(x for x, _row in ordered), columns

    def redraw(self, *_args) -> None:
        """The Track family on the left axis and the Right family on the right one, from the rows with an x."""
        self.plot.clear()
        self.right_view_box.clear()
        if self.legend is not None:
            self.legend.clear()  # the right-axis entries were added by hand: PlotItem.clear cannot reach them
        identities, xs, columns = self.table()
        drawn = [index for index, x in enumerate(xs) if x is not None]
        missing = len(xs) - len(drawn)
        self.note.setText(f"{missing} result(s) without {self._x_name} are listed, not drawn" if missing else "")
        self.note.setVisible(bool(missing))
        if not drawn:
            self.right_axis.setVisible(False)
            self.right_view_box.setVisible(False)
            return
        x = np.asarray([xs[index] for index in drawn], dtype=float)
        grouped = group_families(columns)
        overlay = self.overlay.isChecked()
        colour = 0
        family = self.family.currentData()
        if family is not None:
            for key, _peak in grouped.get(family, [])[:None if overlay else 1]:
                pen = CURVE_PENS[colour % len(CURVE_PENS)]
                colour += 1
                y = np.asarray([columns[key][index] for index in drawn], dtype=float)
                self.plot.plot(x, y, pen=pg.mkPen(pen, width=2), name=key, symbol="o", symbolSize=5, symbolBrush=pen)
            self.plot.setLabel("left", self._axis_label(family))
        right = self.right_family.currentData()
        any_right = False
        if right is not None and right != family:
            for key, _peak in grouped.get(right, [])[:None if overlay else 1]:
                pen = CURVE_PENS[colour % len(CURVE_PENS)]
                colour += 1
                y = np.asarray([columns[key][index] for index in drawn], dtype=float)
                add_right_series(self.right_view_box, self.legend, x, y,
                                 pen=pg.mkPen(pen, width=2, style=QtCore.Qt.PenStyle.DashLine), name=key,
                                 symbol="t", symbol_size=5, symbol_brush=pen)
                any_right = True
            self.right_axis.setLabel(self._axis_label(right))
        self.right_axis.setVisible(any_right)
        self.right_view_box.setVisible(any_right)

    def _ordered(self) -> list[tuple[float | None, TrendRow]]:
        rows = list(self._rows.values())
        if self._order:
            return [(float(position), row) for position, row in enumerate(rows)]
        # rows with an x, by x (ties in the results' order); a row without one is never compared, it goes last
        known = sorted(((x, position) for position, row in enumerate(rows) if (x := row.x) is not None))
        return [(x, rows[position]) for x, position in known] + [(None, row) for row in rows if row.x is None]

    def _sync_families(self) -> None:
        """Both family lists follow the families the rows have, each keeping its choice; rebuilt only when the set
        of families changes."""
        keys: list[str] = []
        for row in self._rows.values():
            keys += [key for key in row.params if key not in keys]
        families = tuple(group_families(keys))
        if frozenset(families) == frozenset(self._families):
            return
        for combo, none in ((self.family, False), (self.right_family, True)):
            current = combo.currentData()
            blocker = QtCore.QSignalBlocker(combo)
            combo.clear()
            if none:
                combo.addItem("None", None)
            for name in families:
                combo.addItem(FAMILY_LABELS.get(name, name), name)
            if current is not None and combo.findData(current) >= 0:
                combo.setCurrentIndex(combo.findData(current))
            del blocker
        self._families = families

    def _axis_label(self, family: str) -> str:
        unit = self._value_unit if family in X_UNIT_FAMILIES else "Intensity"
        label = FAMILY_LABELS.get(family, family)
        return f"{label} ({unit})" if unit else label

    def _set_enabled(self, enabled: bool) -> None:
        for widget in (self.family, self.right_family, self.overlay, self.save):
            widget.setEnabled(enabled)

    def _save_csv(self) -> None:
        if not self._rows:
            return
        from xdart.gui.widgets.file_dialogs import browse
        picks = browse(self, self._save_title, save=True, file_filter=CSV_FILTER, default_name=self._save_name)
        if not picks:
            return
        path = picks[0]
        identities, xs, columns = self.table()
        keys = list(columns)
        try:
            with open(path, "w", newline="") as handle:
                writer = csv.writer(handle)
                prefix = [self._identity_column] if self._identity_column is not None else []
                writer.writerow(prefix + [self._x_name] + keys)
                for index, identity in enumerate(identities):
                    x = xs[index]
                    # Keep Int's integral frame CSV spelling while exporting the actual numeric x.
                    value = "" if x is None else int(x) if float(x).is_integer() else x
                    prefix = [identity] if self._identity_column is not None else []
                    writer.writerow(prefix + [value] + [columns[key][index] for key in keys])
        except OSError:
            logger.exception("trend CSV save failed")
            self.messageChanged.emit("Could not save CSV (see log).")
            return
        self.messageChanged.emit(f"Saved {path}")


__all__ = ["CSV_FILTER", "PlotSeries", "SeriesProjection", "TrendRow", "TrendView", "add_right_series",
           "attach_right_axis", "set_right_axis_visible"]
