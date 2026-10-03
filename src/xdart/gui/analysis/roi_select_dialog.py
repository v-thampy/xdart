# -*- coding: utf-8 -*-
"""ROI selection popup for the Scan Plot tool.

Mounts ScientificView's detector picker with independently keyed signal and
background rectangles, each two-way synced to numeric ``center_row / center_col /
width_row / width_col`` fields — the 2-D generalization of the Peak Fitter's
fit-range ``LinearRegionItem`` ↔ fields sync.  Each signal ROI carries its own
reducer (sum/mean/max/min/std) and an optional paired *background* ROI combined
by subtract or divide.  ``Compute`` emits the assembled
:class:`xrd_tools.analysis.plans.RoiSignal` list; the ROI math itself is
headless (``run_roi_signals``) — this dialog is the thin picker.

The image is drawn row-major (array axis 0 = rows = the y axis, axis 1 = cols =
the x axis), so a region at ``(x, y)`` maps directly to a
``RoiSpec(center_x=col, center_y=row, …)``.
"""

import logging

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from .peak_fit_util import CURVE_PENS
from xdart.gui.widgets.controls_panel import FormRow
from xdart.gui.tabs.scattering.browser_view import BrowserResult, BrowserView, keyed_list_profile
from xdart.gui.tabs.scattering.scientific_view import ScientificView
from xdart.gui.tabs.scattering.shell_values import RectRegion, ShellCommandKind

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
    """One signal ROI's choices and borrowed interaction handles; ScientificView owns the marks."""

    __slots__ = ("name", "rect", "reducer", "bg_enabled", "bg_op", "bg_rect")

    def __init__(self, name, rect):
        self.name = name
        self.rect = rect
        self.reducer = "mean"
        self.bg_enabled = False
        self.bg_op = "subtract"
        self.bg_rect = None


class RoiSelectDialog(QtWidgets.QDialog):
    """Pick rectangular ROIs in detector pixels; emit them as RoiSignals."""

    #: emitted on Compute with a list of
    #: :class:`xrd_tools.analysis.plans.RoiSignal`.
    sigCompute = QtCore.Signal(object)
    sigPreviewFrame = QtCore.Signal(int)

    def __init__(self, image, parent=None, *, frame_labels=(), frame_label=None):
        super().__init__(parent)
        self._image = np.asarray(image, dtype=float)
        self._image_shape = self._image.shape       # (rows, cols)
        self._frame_labels = tuple(frame_labels)
        self._frame_label = frame_label
        self._pending_label = None
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

        self.scientific = ScientificView(self, detector_picker=True)
        self.scientific.setMinimumHeight(300)
        self._image_key = self._apply_scale()
        self.scientific.commandRequested.connect(self._preview_command)
        self.scientific.keyedNamedRegionChanged.connect(self._on_mark_changed)
        pane = self.scientific.keyed_panel(self._image_key).pane
        # Borrowed compatibility handles; the shared owner constructs and retains every display widget.
        self.image_plot, self.image_item, self.colorbar = pane.plot, pane.image, pane.color_scale
        self.scale_default_btn, self.scale_log_btn = self.scientific.default_scale, self.scientific.detector_log
        lay.addWidget(self.scientific, 3)

        # ROI list + Add / Remove.
        roi_row = QtWidgets.QHBoxLayout()
        roi_row.setSpacing(7)
        self.roi_browser = BrowserView(self, profile=keyed_list_profile("Signal ROIs"))
        self.roi_list = self.roi_browser.scans
        self.roi_list.setObjectName("roiList")
        self.roi_list.setMaximumHeight(90)
        self.roi_list.setToolTip("Signal ROIs — each becomes a plotted column")
        roi_row.addWidget(self.roi_browser, 1)
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

        self._numeric_rows = {}

        def _num(name):
            row = FormRow(label="", path=("roi", name), value="")
            row.label.hide()
            row.editor.setValidator(QtGui.QDoubleValidator(self))
            row.editor.setMaximumWidth(80)
            self._numeric_rows[row.editor] = row
            return row.editor

        self.f_crow, self.f_ccol = _num("center_row"), _num("center_column")
        self.f_wrow, self.f_wcol = _num("width_row"), _num("width_column")
        ed.addWidget(QtWidgets.QLabel("Center (row, col)"), 0, 0)
        ed.addWidget(self._numeric_rows[self.f_crow], 0, 1)
        ed.addWidget(self._numeric_rows[self.f_ccol], 0, 2)
        ed.addWidget(QtWidgets.QLabel("Size (row, col)"), 1, 0)
        ed.addWidget(self._numeric_rows[self.f_wrow], 1, 1)
        ed.addWidget(self._numeric_rows[self.f_wcol], 1, 2)

        ed.addWidget(QtWidgets.QLabel("Reducer"), 2, 0)
        self.reducer_row = FormRow(label="", path=("roi", "reducer"), value="mean", kind="combo",
                                   choices=_REDUCERS)
        self.reducer_row.label.hide()
        self.reducer_combo = self.reducer_row.editor
        ed.addWidget(self.reducer_row, 2, 1, 1, 2)

        self.bg_row = FormRow(label="Background ROI", path=("roi", "background"), value=False, kind="bool")
        self.bg_check = self.bg_row.editor
        self.bg_check.setToolTip(
            "Pair this signal ROI with a background ROI, combined per the "
            "operation below")
        ed.addWidget(self.bg_row, 3, 0)
        self.bg_op_row = FormRow(label="", path=("roi", "background_op"), value="Subtract", kind="combo",
                                 choices=[label for label, _op in _BG_OPS])
        self.bg_op_row.label.hide()
        self.bg_op_combo = self.bg_op_row.editor
        self.bg_op_combo.setEnabled(False)
        ed.addWidget(self.bg_op_row, 3, 1, 1, 2)

        self.bg_label = QtWidgets.QLabel("Bkg center / size")
        self.f_bg_crow, self.f_bg_ccol = _num("background_center_row"), _num("background_center_column")
        self.f_bg_wrow, self.f_bg_wcol = _num("background_width_row"), _num("background_width_column")
        ed.addWidget(self.bg_label, 4, 0)
        bg_box = QtWidgets.QHBoxLayout()
        bg_box.setSpacing(4)
        for w in (self.f_bg_crow, self.f_bg_ccol, self.f_bg_wrow, self.f_bg_wcol):
            bg_box.addWidget(self._numeric_rows[w])
        ed.addLayout(bg_box, 4, 1, 1, 2)
        ed.setColumnStretch(2, 1)
        lay.addWidget(self.editor)

        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("peakFitStatus")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

        # Compute / Close.
        action_row = QtWidgets.QHBoxLayout()
        self.mask_sat_row = FormRow(label="Mask saturated", path=("roi", "mask_saturated"), value=False, kind="bool")
        self.mask_sat_check = self.mask_sat_row.editor
        self.mask_sat_check.setToolTip(
            "Exclude dead/saturated pixels (the dtype-ceiling module mask) from "
            "the ROI stats — matches the integrator's saturated-pixel masking. "
            "The uint32 dummy + non-finite pixels are always excluded.")
        action_row.addWidget(self.mask_sat_row)
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
        label = self._frame_label if self._pending_label is None else self._pending_label
        index = self._frame_labels.index(label) if self._frame_labels else 0
        title = f"Frame {self._frame_label}" if self._frame_labels else ""
        if self._pending_label is not None:
            title += f" · loading frame {self._pending_label}…"
        return self.scientific.reconcile_detector_image(
            self._image, tuple(str(value) for value in self._frame_labels), index, title)

    def _preview_command(self, command):
        if (command.kind is ShellCommandKind.SET_PANEL_INDEX
                and 0 <= command.value < len(self._frame_labels)):
            self.sigPreviewFrame.emit(self._frame_labels[command.value])

    def set_preview_pending(self, label=None, *, message=None):
        """Keep accepted pixels and marks visible while another preview settles."""
        self._pending_label = label
        self.compute_btn.setEnabled(label is None)
        self._apply_scale()
        if message is not None:
            self.status.setText(message)

    def adopt_preview(self, image, label):
        """Update the shared image in place; changed shapes cannot reinterpret marks."""
        if np.asarray(image).shape != self._image_shape:
            self.set_preview_pending(message="Frame shape differs; keeping the previous image and ROIs.")
            return False
        self._image = np.asarray(image, dtype=float)
        self._frame_label = label
        self.set_preview_pending()
        self._update_status()
        return True

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

    def _make_mark(self, name, pos, size, color, dashed=False):
        region = RectRegion((float(pos[0]), float(pos[0] + size[0])),
                            (float(pos[1]), float(pos[1] + size[1])), ("column", "px"), ("row", "px"))
        self.scientific.set_keyed_named_region(self._image_key, name, region, color=color, dashed=dashed)
        return self.scientific.keyed_region_item(self._image_key, name)

    def _sync_roi_list(self, row):
        current = self._rois[row].name if 0 <= row < len(self._rois) else None
        self.roi_browser.reconcile_results(tuple(BrowserResult(entry.name, entry.name) for entry in self._rois), current)
        self._on_select(row)

    def _on_mark_changed(self, _key, name, _region):
        for entry in self._rois:
            if name in (entry.name, f"{entry.name}_bg"):
                self._on_rect_changed(entry, bg=name.endswith("_bg"))
                break

    def _add_roi(self):
        # The first ROI defaults to the whole frame; later ones to a ¼ box.
        pos, size = self._full_frame_box() if not self._rois else self._default_box()
        color = CURVE_PENS[len(self._rois) % len(CURVE_PENS)]
        name = self._new_name()
        rect = self._make_mark(name, pos, size, color)
        entry = _RoiEntry(name, rect)
        self._rois.append(entry)
        self._sync_roi_list(len(self._rois) - 1)
        self._update_status()

    def _remove_roi(self):
        row = self.roi_list.currentRow()
        if not (0 <= row < len(self._rois)):
            return
        entry = self._rois.pop(row)
        self.scientific.set_keyed_named_region(self._image_key, entry.name, None)
        self.scientific.set_keyed_named_region(self._image_key, f"{entry.name}_bg", None)
        self._sync_roi_list(min(row, len(self._rois) - 1))
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
            entry = next(e for e in self._rois if rect is e.rect or rect is e.bg_rect)
            bg = rect is entry.bg_rect
            self._make_mark(f"{entry.name}_bg" if bg else entry.name, pos, size, rect.pen.color(), dashed=bg)
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
            entry.bg_rect = self._make_mark(f"{entry.name}_bg", pos, size, color, dashed=True)
            self._guard = True
            try:
                self._rect_to_fields(entry.bg_rect, self.f_bg_crow, self.f_bg_ccol,
                                     self.f_bg_wrow, self.f_bg_wcol)
            finally:
                self._guard = False
        if entry.bg_rect is not None:
            region = self.scientific.keyed_named_region(self._image_key, f"{entry.name}_bg")
            self.scientific.set_keyed_named_region(self._image_key, f"{entry.name}_bg", region,
                color=entry.bg_rect.pen.color(), dashed=True, visible=bool(on))

    def _set_bg_fields_visible(self, on):
        for w in (self.bg_label, self.f_bg_crow, self.f_bg_ccol,
                  self.f_bg_wrow, self.f_bg_wcol):
            self._numeric_rows.get(w, w).setVisible(bool(on))

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
        if self._pending_label is not None:
            return
        if not self._rois:
            self.status.setText("Add at least one ROI first.")
            return
        self.sigCompute.emit(self.roi_signals())
