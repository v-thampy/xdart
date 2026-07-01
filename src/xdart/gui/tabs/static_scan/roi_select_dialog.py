# -*- coding: utf-8 -*-
"""ROI selection popup for the Scan Plot tool.

Pops the scan's first raw frame with one or more draggable ``pg.RectROI``
rectangles, each two-way synced to numeric ``center_row / center_col /
width_row / width_col`` fields — the 2-D generalization of the Peak Fitter's
fit-range ``LinearRegionItem`` ↔ fields sync.  Each signal ROI carries its own
reducer (sum/mean/max/min/std) and an optional paired *background* ROI combined
by subtract or divide.  ``Compute`` emits the assembled
:class:`xrd_tools.analysis.plans.RoiSignal` list; the ROI math itself is
headless (``run_roi_signals``) — this dialog is the thin picker.

The image is drawn row-major (array axis 0 = rows = the y axis, axis 1 = cols =
the x axis), so a ``RectROI`` at item ``(x, y)`` maps directly to a
``RoiSpec(center_x=col, center_y=row, …)``.
"""

import logging

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from .peak_fit_util import CURVE_PENS
from xdart.gui.widgets.image_widget import _ceiling_safe_levels

logger = logging.getLogger(__name__)

#: reducer options (label == the headless reducer string).
_REDUCERS = ("mean", "sum", "max", "min", "std")
#: background operations (label, RoiSignal.background_op).
_BG_OPS = [("Subtract", "subtract"), ("Divide", "divide")]


def _rect_center_size(rect):
    """``pg.RectROI`` geometry → ``(center_row, center_col, width_row,
    width_col)`` in image-pixel coords (x = col, y = row)."""
    pos = rect.pos()
    size = rect.size()
    cx = float(pos.x()) + float(size.x()) / 2.0
    cy = float(pos.y()) + float(size.y()) / 2.0
    return cy, cx, float(size.y()), float(size.x())


class _RoiEntry:
    """One signal ROI: its rectangle + reducer + optional background."""

    __slots__ = ("name", "rect", "reducer", "bg_enabled", "bg_op", "bg_rect")

    def __init__(self, name, rect):
        self.name = name
        self.rect = rect
        self.reducer = "mean"
        self.bg_enabled = False
        self.bg_op = "subtract"
        self.bg_rect = None


class RoiSelectDialog(QtWidgets.QDialog):
    """Pick rectangular ROIs on the first frame; emit them as RoiSignals."""

    #: emitted on Compute with a list of
    #: :class:`xrd_tools.analysis.plans.RoiSignal`.
    sigCompute = QtCore.Signal(object)

    def __init__(self, image, parent=None):
        super().__init__(parent)
        self._image = np.asarray(image, dtype=float)
        self._image_shape = self._image.shape       # (rows, cols)
        self._rois = []
        self._guard = False                         # region<->fields recursion guard
        self.setObjectName("roiSelectDialog")
        self.setWindowTitle("Select ROIs")
        self.resize(560, 680)
        self._build_ui()
        self._add_roi()                             # start with one ROI

    # ---- UI -------------------------------------------------------------
    def _build_ui(self):
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(11, 11, 11, 11)
        lay.setSpacing(8)

        # Image with a fixed aspect ratio, row-major so (x, y) == (col, row).
        self.image_plot = pg.PlotWidget()
        self.image_plot.setMinimumHeight(300)
        self.image_plot.setAspectLocked(True)
        self.image_plot.invertY(True)               # row 0 at the top
        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.image_plot.addItem(self.image_item)
        # Intensity bar + colormap matching the main Image Viewer mode: a
        # ColorBarItem linked to the frame (the same widget pgImageWidget uses).
        self._cmap = pg.colormap.getFromMatplotlib("viridis")
        self.colorbar = pg.ColorBarItem(width=15)
        self.colorbar.setImageItem(self.image_item,
                                   insert_in=self.image_plot.getPlotItem())
        self.colorbar.setColorMap(self._cmap)
        lay.addWidget(self.image_plot, 3)

        # Default / Log intensity scale, mirroring the Image Viewer's buttons.
        scale_row = QtWidgets.QHBoxLayout()
        scale_row.setSpacing(6)
        scale_row.addWidget(QtWidgets.QLabel("Intensity"))
        self.scale_default_btn = QtWidgets.QPushButton("Default")
        self.scale_log_btn = QtWidgets.QPushButton("Log")
        self._scale_group = QtWidgets.QButtonGroup(self)
        self._scale_group.setExclusive(True)
        for b in (self.scale_default_btn, self.scale_log_btn):
            b.setCheckable(True)
            self._scale_group.addButton(b)
        self.scale_default_btn.setChecked(True)
        scale_row.addWidget(self.scale_default_btn)
        scale_row.addWidget(self.scale_log_btn)
        scale_row.addStretch(1)
        lay.addLayout(scale_row)
        self.scale_log_btn.toggled.connect(self._apply_scale)
        self._apply_scale()                         # initial linear render

        # ROI list + Add / Remove.
        roi_row = QtWidgets.QHBoxLayout()
        roi_row.setSpacing(7)
        self.roi_list = QtWidgets.QListWidget()
        self.roi_list.setObjectName("roiList")
        self.roi_list.setMaximumHeight(90)
        self.roi_list.setToolTip("Signal ROIs — each becomes a plotted column")
        roi_row.addWidget(self.roi_list, 1)
        btn_col = QtWidgets.QVBoxLayout()
        btn_col.setSpacing(4)
        self.add_btn = QtWidgets.QPushButton("Add ROI")
        self.remove_btn = QtWidgets.QPushButton("Remove")
        btn_col.addWidget(self.add_btn)
        btn_col.addWidget(self.remove_btn)
        btn_col.addStretch(1)
        roi_row.addLayout(btn_col)
        lay.addLayout(roi_row)

        # Editor for the selected ROI: center/size fields + reducer + background.
        self.editor = QtWidgets.QWidget()
        ed = QtWidgets.QGridLayout(self.editor)
        ed.setContentsMargins(2, 2, 2, 2)
        ed.setHorizontalSpacing(7)
        ed.setVerticalSpacing(5)

        def _num():
            e = QtWidgets.QLineEdit()
            e.setValidator(QtGui.QDoubleValidator(self))
            e.setMaximumWidth(80)
            return e

        self.f_crow, self.f_ccol = _num(), _num()
        self.f_wrow, self.f_wcol = _num(), _num()
        ed.addWidget(QtWidgets.QLabel("Center (row, col)"), 0, 0)
        ed.addWidget(self.f_crow, 0, 1)
        ed.addWidget(self.f_ccol, 0, 2)
        ed.addWidget(QtWidgets.QLabel("Size (row, col)"), 1, 0)
        ed.addWidget(self.f_wrow, 1, 1)
        ed.addWidget(self.f_wcol, 1, 2)

        ed.addWidget(QtWidgets.QLabel("Reducer"), 2, 0)
        self.reducer_combo = QtWidgets.QComboBox()
        for r in _REDUCERS:
            self.reducer_combo.addItem(r)
        ed.addWidget(self.reducer_combo, 2, 1, 1, 2)

        self.bg_check = QtWidgets.QCheckBox("Background ROI")
        self.bg_check.setToolTip(
            "Pair this signal ROI with a background ROI, combined per the "
            "operation below")
        ed.addWidget(self.bg_check, 3, 0)
        self.bg_op_combo = QtWidgets.QComboBox()
        for label, _op in _BG_OPS:
            self.bg_op_combo.addItem(label)
        self.bg_op_combo.setEnabled(False)
        ed.addWidget(self.bg_op_combo, 3, 1, 1, 2)

        self.bg_label = QtWidgets.QLabel("Bkg center / size")
        self.f_bg_crow, self.f_bg_ccol = _num(), _num()
        self.f_bg_wrow, self.f_bg_wcol = _num(), _num()
        ed.addWidget(self.bg_label, 4, 0)
        bg_box = QtWidgets.QHBoxLayout()
        bg_box.setSpacing(4)
        for w in (self.f_bg_crow, self.f_bg_ccol, self.f_bg_wrow, self.f_bg_wcol):
            bg_box.addWidget(w)
        ed.addLayout(bg_box, 4, 1, 1, 2)
        ed.setColumnStretch(2, 1)
        lay.addWidget(self.editor)

        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("peakFitStatus")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

        # Compute / Close.
        action_row = QtWidgets.QHBoxLayout()
        self.mask_sat_check = QtWidgets.QCheckBox("Mask saturated")
        self.mask_sat_check.setToolTip(
            "Exclude dead/saturated pixels (the dtype-ceiling module mask) from "
            "the ROI stats — matches the integrator's saturated-pixel masking. "
            "The uint32 dummy + non-finite pixels are always excluded.")
        action_row.addWidget(self.mask_sat_check)
        action_row.addStretch(1)
        self.compute_btn = QtWidgets.QPushButton("Compute")
        self.compute_btn.setObjectName("roiCompute")
        self.compute_btn.setToolTip(
            "Reduce these ROIs over every raw frame and add each as a plotted "
            "column")
        self.close_btn = QtWidgets.QPushButton("Close")
        action_row.addWidget(self.compute_btn)
        action_row.addWidget(self.close_btn)
        lay.addLayout(action_row)

        self.add_btn.clicked.connect(self._add_roi)
        self.remove_btn.clicked.connect(self._remove_roi)
        self.roi_list.currentRowChanged.connect(self._on_select)
        for f in (self.f_crow, self.f_ccol, self.f_wrow, self.f_wcol):
            f.editingFinished.connect(self._fields_to_rect)
        for f in (self.f_bg_crow, self.f_bg_ccol, self.f_bg_wrow, self.f_bg_wcol):
            f.editingFinished.connect(self._bg_fields_to_rect)
        self.reducer_combo.currentIndexChanged.connect(self._on_reducer_changed)
        self.bg_check.toggled.connect(self._on_bg_toggled)
        self.bg_op_combo.currentIndexChanged.connect(self._on_bg_op_changed)
        self.compute_btn.clicked.connect(self._on_compute)
        self.close_btn.clicked.connect(self.close)
        self._set_bg_fields_visible(False)

    def keyPressEvent(self, event):
        # Don't let a stray Esc (e.g. after editing a numeric field) dismiss the
        # ROI picker mid-selection — Close is the explicit exit.
        if event.key() == QtCore.Qt.Key.Key_Escape:
            event.accept()
            return
        super().keyPressEvent(event)

    def _apply_scale(self, *_args):
        """Render the frame at the chosen intensity scale (Default = linear,
        Log), reusing the main Image Viewer's ceiling-safe autoscale + colorbar
        so ROI picking sees the same levels the viewer does.  The RectROI
        overlay lives on the plot (not the image item), so re-rendering here
        never disturbs the picked boxes."""
        disp = np.asarray(np.copy(self._image), dtype=float)
        if disp.size == 0 or not np.isfinite(disp).any():
            self.image_item.setImage(disp)
            return
        if self.scale_log_btn.isChecked():
            min_val = float(np.nanmin(disp))
            if min_val < 1:
                disp -= (min_val - 1)
            disp = np.log10(disp)
            levels = _ceiling_safe_levels(disp, self._image, (0.1, 99.9))
            self.image_item.setImage(disp, levels=levels)
            self.colorbar.axis.setLogMode(True)
        else:
            levels = _ceiling_safe_levels(disp, self._image, (2, 98))
            self.image_item.setImage(disp, levels=levels)
            self.colorbar.axis.setLogMode(False)
        if np.isfinite(disp).any():
            low, high = float(np.nanmin(disp)), float(np.nanmax(disp))
        else:
            low, high = 0.0, 1.0
        self.colorbar.lo_lim, self.colorbar.hi_lim = low, high
        self.colorbar.setLevels(values=levels)

    # ---- ROI lifecycle --------------------------------------------------
    def _default_box(self):
        """A centered starter rectangle ``([x0, y0], [w, h])`` (¼ of the frame)."""
        rows, cols = self._image_shape
        w = max(4.0, cols / 4.0)
        h = max(4.0, rows / 4.0)
        return [cols / 2.0 - w / 2.0, rows / 2.0 - h / 2.0], [w, h]

    def _full_frame_box(self):
        """The whole-detector rectangle ``([0, 0], [cols, rows])`` — the default
        first ROI (ROI doc §1: one ROI defaults to the entire frame)."""
        rows, cols = self._image_shape
        return [0.0, 0.0], [float(cols), float(rows)]

    def _new_name(self):
        existing = {e.name for e in self._rois}
        i = 1
        while f"roi{i}" in existing:
            i += 1
        return f"roi{i}"

    def _make_rect(self, pos, size, color, dashed=False):
        style = QtCore.Qt.PenStyle.DashLine if dashed else QtCore.Qt.PenStyle.SolidLine
        pen = pg.mkPen(color, width=2, style=style)
        rect = pg.RectROI(pos, size, pen=pen, movable=True, resizable=True)
        rect.addScaleHandle([0, 0], [1, 1])         # a second corner handle
        rect.setZValue(10)
        self.image_plot.addItem(rect)
        return rect

    def _add_roi(self):
        # The first ROI defaults to the whole frame; later ones to a ¼ box.
        pos, size = self._full_frame_box() if not self._rois else self._default_box()
        color = CURVE_PENS[len(self._rois) % len(CURVE_PENS)]
        rect = self._make_rect(pos, size, color)
        entry = _RoiEntry(self._new_name(), rect)
        rect.sigRegionChanged.connect(
            lambda *_a, e=entry: self._on_rect_changed(e, bg=False))
        self._rois.append(entry)
        self.roi_list.addItem(entry.name)
        self.roi_list.setCurrentRow(len(self._rois) - 1)
        self._update_status()

    def _remove_roi(self):
        row = self.roi_list.currentRow()
        if not (0 <= row < len(self._rois)):
            return
        entry = self._rois.pop(row)
        self.image_plot.removeItem(entry.rect)
        if entry.bg_rect is not None:
            self.image_plot.removeItem(entry.bg_rect)
        self.roi_list.takeItem(row)
        self._update_status()

    def _current_entry(self):
        row = self.roi_list.currentRow()
        if 0 <= row < len(self._rois):
            return self._rois[row]
        return None

    # ---- selection / field <-> rect sync --------------------------------
    def _on_select(self, _row):
        entry = self._current_entry()
        self.editor.setEnabled(entry is not None)
        if entry is None:
            return
        self._guard = True
        try:
            self.reducer_combo.setCurrentText(entry.reducer)
            self.bg_check.setChecked(entry.bg_enabled)
            self.bg_op_combo.setCurrentIndex(
                0 if entry.bg_op == "subtract" else 1)
            self._rect_to_fields(entry.rect,
                                 self.f_crow, self.f_ccol, self.f_wrow, self.f_wcol)
            if entry.bg_rect is not None:
                self._rect_to_fields(entry.bg_rect, self.f_bg_crow, self.f_bg_ccol,
                                     self.f_bg_wrow, self.f_bg_wcol)
        finally:
            self._guard = False
        self._set_bg_fields_visible(entry.bg_enabled)
        self._apply_bg_gate()

    @staticmethod
    def _rect_to_fields(rect, f_crow, f_ccol, f_wrow, f_wcol):
        crow, ccol, wrow, wcol = _rect_center_size(rect)
        f_crow.setText(f"{crow:.1f}")
        f_ccol.setText(f"{ccol:.1f}")
        f_wrow.setText(f"{wrow:.1f}")
        f_wcol.setText(f"{wcol:.1f}")

    def _on_rect_changed(self, entry, bg):
        """A rectangle was dragged/resized → mirror it into the fields (only when
        that entry is selected)."""
        if self._guard or entry is not self._current_entry():
            return
        self._guard = True
        try:
            if bg:
                self._rect_to_fields(entry.bg_rect, self.f_bg_crow, self.f_bg_ccol,
                                     self.f_bg_wrow, self.f_bg_wcol)
            else:
                self._rect_to_fields(entry.rect, self.f_crow, self.f_ccol,
                                     self.f_wrow, self.f_wcol)
        finally:
            self._guard = False

    @staticmethod
    def _read_box(f_crow, f_ccol, f_wrow, f_wcol):
        """Read 4 fields → ``([x0, y0], [w, h])`` or ``None`` if incomplete."""
        try:
            crow = float(f_crow.text())
            ccol = float(f_ccol.text())
            wrow = float(f_wrow.text())
            wcol = float(f_wcol.text())
        except ValueError:
            return None
        wrow = max(1.0, wrow)
        wcol = max(1.0, wcol)
        return [ccol - wcol / 2.0, crow - wrow / 2.0], [wcol, wrow]

    def _fields_to_rect(self):
        if self._guard:
            return
        entry = self._current_entry()
        if entry is None:
            return
        box = self._read_box(self.f_crow, self.f_ccol, self.f_wrow, self.f_wcol)
        if box is None:
            return
        self._apply_box(entry.rect, box)

    def _bg_fields_to_rect(self):
        if self._guard:
            return
        entry = self._current_entry()
        if entry is None or entry.bg_rect is None:
            return
        box = self._read_box(self.f_bg_crow, self.f_bg_ccol,
                             self.f_bg_wrow, self.f_bg_wcol)
        if box is None:
            return
        self._apply_box(entry.bg_rect, box)

    def _apply_box(self, rect, box):
        pos, size = box
        self._guard = True
        try:
            rect.setSize(size, update=False)
            rect.setPos(pos)
        finally:
            self._guard = False

    # ---- reducer / background -------------------------------------------
    def _on_reducer_changed(self, _idx):
        entry = self._current_entry()
        if entry is not None and not self._guard:
            entry.reducer = self.reducer_combo.currentText()
        self._apply_bg_gate()

    def _apply_bg_gate(self):
        """A paired background is only defined for the mean/sum reducers (the
        density rule, ROI doc §6.2); disable the background controls — and turn an
        active background off — for max/min/std."""
        entry = self._current_entry()
        allow = entry is not None and entry.reducer in ("mean", "sum")
        self.bg_check.setEnabled(allow)
        if not allow and self.bg_check.isChecked():
            self.bg_check.setChecked(False)         # -> _on_bg_toggled hides it

    def _on_bg_op_changed(self, _idx):
        entry = self._current_entry()
        if entry is not None and not self._guard:
            entry.bg_op = _BG_OPS[self.bg_op_combo.currentIndex()][1]

    def _on_bg_toggled(self, on):
        self.bg_op_combo.setEnabled(on)
        self._set_bg_fields_visible(on)
        entry = self._current_entry()
        if entry is None or self._guard:
            return
        entry.bg_enabled = bool(on)
        if on and entry.bg_rect is None:
            # A background box offset from the signal box.
            pos, size = self._default_box()
            pos = [pos[0] + size[0] * 0.6, pos[1] + size[1] * 0.6]
            color = CURVE_PENS[(self._rois.index(entry) + 3) % len(CURVE_PENS)]
            entry.bg_rect = self._make_rect(pos, size, color, dashed=True)
            entry.bg_rect.sigRegionChanged.connect(
                lambda *_a, e=entry: self._on_rect_changed(e, bg=True))
            self._guard = True
            try:
                self._rect_to_fields(entry.bg_rect, self.f_bg_crow, self.f_bg_ccol,
                                     self.f_bg_wrow, self.f_bg_wcol)
            finally:
                self._guard = False
        if entry.bg_rect is not None:
            entry.bg_rect.setVisible(bool(on))

    def _set_bg_fields_visible(self, on):
        for w in (self.bg_label, self.f_bg_crow, self.f_bg_ccol,
                  self.f_bg_wrow, self.f_bg_wcol):
            w.setVisible(bool(on))

    def _update_status(self):
        self.status.setText(
            f"{len(self._rois)} ROI(s). Drag a box or edit its fields, then "
            "Compute.")

    # ---- output ---------------------------------------------------------
    def roi_signals(self):
        """The current ROIs as :class:`RoiSignal`s (geometry read live from the
        rectangles)."""
        from xrd_tools.analysis.plans import RoiSignal
        from xrd_tools.core.roi import RoiSpec

        out = []
        for entry in self._rois:
            crow, ccol, wrow, wcol = _rect_center_size(entry.rect)
            roi = RoiSpec(center_x=ccol, center_y=crow, width_x=wcol,
                          width_y=wrow, name=entry.name)
            bg = None
            if entry.bg_enabled and entry.bg_rect is not None:
                bcrow, bccol, bwrow, bwcol = _rect_center_size(entry.bg_rect)
                bg = RoiSpec(center_x=bccol, center_y=bcrow, width_x=bwcol,
                             width_y=bwrow, name=f"{entry.name}_bg")
            out.append(RoiSignal(roi=roi, reducer=entry.reducer, background=bg,
                                 background_op=entry.bg_op, name=entry.name))
        return out

    def mask_saturated(self):
        """Whether dead/saturated pixels should be excluded from the ROI stats."""
        return self.mask_sat_check.isChecked()

    def _on_compute(self):
        if not self._rois:
            self.status.setText("Add at least one ROI first.")
            return
        self.sigCompute.emit(self.roi_signals())
