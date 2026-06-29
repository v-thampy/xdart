# -*- coding: utf-8 -*-
"""Display frame widget — main controller for 2D image and 1D plot display.

@author: thampy

This module was refactored in Phase 3 of the GUI architecture cleanup.
Data-fetching/processing methods live in ``display_data.py`` (DisplayDataMixin)
and plot-rendering/waterfall/slice methods live in ``display_plot.py``
(DisplayPlotMixin).  This file retains the widget shell, initialization,
update orchestration, viewer modes, and 2D image rendering.
"""

# Standard library imports
import logging
import os
import re
import threading

logger = logging.getLogger(__name__)

# Other imports
import matplotlib.pyplot as plt
import numpy as np

# Qt imports
from PySide6.QtCore import Qt as pyQt
import pyqtgraph as pg
from pyqtgraph import Qt
from pyqtgraph.Qt import QtWidgets

# This module imports
from .ui.displayFrameUI import Ui_Form
from ...gui_utils import RectViewBox, get_rect
from ...widgets import pgImageWidget, pmeshImageWidget
from .integrator import GI_LABELS_1D, GI_LABELS_2D
from .display_constants import (
    AA_inv, Th, Chi, Deg, Qip_s, Qoop_s, Qtot_s,
    plotUnits, imageUnits,
    x_labels_1D, x_units_1D, x_labels_2D, x_units_2D,
    y_labels_2D, y_units_2D,
    gi_plotUnits, gi_imageUnits,
    gi_x_labels_1D, gi_x_units_1D,
    gi_x_labels_2D, gi_x_units_2D,
    gi_y_labels_2D, gi_y_units_2D,
    GI_MODES_1D, GI_MODES_2D,
    GI_2D_AXES, STD_2D_AXES,
    _downsample_for_display,
)
from .display_data import DisplayDataMixin
from .display_plot import DisplayPlotMixin
from .display_logic import (
    Mode, LoadStatus, PanelRole, compute_display_state,
    build_payload, render_plan, controller_for, ImagePayload,
    empty_display_state, PANEL_LAYOUT,
    resolve_selection, resolve_render_ids,
    default_plot_unit, pretty_unit, sentinel_mask, integer_saturation_ceiling,
    combine_flat_masks, nan_gaps_in_thumbnail,
    stitch_plot_payload, stitch_image_payload,
)
from .display_controllers import register_default_controllers

QFileDialog = QtWidgets.QFileDialog
QInputDialog = QtWidgets.QInputDialog
QCombo = QtWidgets.QComboBox
QDialog = QtWidgets.QDialog
_translate = Qt.QtCore.QCoreApplication.translate

formats = [
    str(f.data(), encoding='utf-8').lower() for f in
    Qt.QtGui.QImageReader.supportedImageFormats()
]


def _axis_key_from_label(label):
    """Canonical key for matching plot/cake axes without relying on row order."""
    text = str(label or '')
    lower = text.lower()
    if Qoop_s in text or 'qoop' in lower or 'q_oop' in lower:
        return 'qoop_A^-1'
    if Qip_s in text or 'qip' in lower or 'q_ip' in lower:
        return 'qip_A^-1'
    if 'exit' in lower:
        return 'exit_angle_deg'
    if '2th' in lower or f'2{Th}' in text:
        return '2th_deg'
    if (Chi in text or 'chi' in lower) and 'gi' in lower:
        return 'chigi_deg'
    if Chi in text or 'chi' in lower:
        return 'chi_deg'
    if 'q' in lower or AA_inv in text:
        return 'q_A^-1'
    return lower.strip()


def _combo_text(combo, index):
    """Return item text from a real QComboBox or the lightweight test fakes."""
    try:
        return combo.itemText(index)
    except AttributeError:
        items = getattr(combo, '_items', None)
        if items is not None and 0 <= index < len(items):
            return items[index]
    try:
        current = combo.currentIndex()
        if int(current) == int(index):
            return combo.currentText()
    except Exception:
        pass
    return ''


class _IntensityRangeSlider(QtWidgets.QWidget):
    """Small two-handle horizontal range slider for display intensity."""

    sigRangeChanged = Qt.QtCore.Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._vmin = 0.0
        self._vmax = 1.0
        self._lo = 0.0
        self._hi = 1.0
        self._drag = None
        self.setMinimumSize(Qt.QtCore.QSize(150, 24))
        self.setMaximumHeight(28)
        self.setFocusPolicy(pyQt.StrongFocus)
        self.setToolTip("Manual intensity range.")

    def values(self):
        return self._lo, self._hi

    def domain(self):
        return self._vmin, self._vmax

    def has_valid_domain(self):
        return np.isfinite(self._vmin) and np.isfinite(self._vmax) and self._vmax > self._vmin

    def _fractions(self):
        if not self.has_valid_domain():
            return 0.0, 1.0
        span = self._vmax - self._vmin
        return ((self._lo - self._vmin) / span,
                (self._hi - self._vmin) / span)

    def setDomain(self, vmin, vmax, *, lower=None, upper=None,
                  preserve_fraction=True, emit=False):
        try:
            vmin, vmax = float(vmin), float(vmax)
        except (TypeError, ValueError):
            self.setEnabled(False)
            return
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            self.setEnabled(False)
            return

        old_frac = self._fractions()
        self._vmin, self._vmax = vmin, vmax
        self.setEnabled(True)
        if lower is None or upper is None:
            if preserve_fraction:
                lower = vmin + old_frac[0] * (vmax - vmin)
                upper = vmin + old_frac[1] * (vmax - vmin)
            else:
                lower, upper = vmin, vmax
        self.setValues(lower, upper, emit=emit)

    def setValues(self, lower, upper, *, emit=True):
        if not self.has_valid_domain():
            return
        lo = float(np.clip(lower, self._vmin, self._vmax))
        hi = float(np.clip(upper, self._vmin, self._vmax))
        if hi < lo:
            lo, hi = hi, lo
        changed = (abs(lo - self._lo) > 1e-12 or abs(hi - self._hi) > 1e-12)
        self._lo, self._hi = lo, hi
        self.update()
        if emit and changed:
            self.sigRangeChanged.emit(self._lo, self._hi)

    def _track_rect(self):
        margin = 9
        h = self.height()
        return Qt.QtCore.QRectF(margin, h / 2 - 3, max(1, self.width() - 2 * margin), 6)

    def _x_for_value(self, value):
        rect = self._track_rect()
        if not self.has_valid_domain():
            return rect.left()
        frac = (value - self._vmin) / (self._vmax - self._vmin)
        return rect.left() + float(np.clip(frac, 0.0, 1.0)) * rect.width()

    def _value_for_x(self, x):
        rect = self._track_rect()
        frac = (float(x) - rect.left()) / rect.width()
        frac = float(np.clip(frac, 0.0, 1.0))
        return self._vmin + frac * (self._vmax - self._vmin)

    def paintEvent(self, event):
        painter = Qt.QtGui.QPainter(self)
        painter.setRenderHint(Qt.QtGui.QPainter.RenderHint.Antialiasing)
        rect = self._track_rect()
        disabled = not self.isEnabled()
        track = Qt.QtGui.QColor("#3a3d4d" if not disabled else "#282a36")
        fill = Qt.QtGui.QColor("#bd93f9" if not disabled else "#6272a4")
        handle = Qt.QtGui.QColor("#f8f8f2" if not disabled else "#6272a4")
        painter.setPen(pyQt.NoPen)
        painter.setBrush(track)
        painter.drawRoundedRect(rect, 3, 3)
        lo_x = self._x_for_value(self._lo)
        hi_x = self._x_for_value(self._hi)
        sel = Qt.QtCore.QRectF(lo_x, rect.top(), max(1.0, hi_x - lo_x), rect.height())
        painter.setBrush(fill)
        painter.drawRoundedRect(sel, 3, 3)
        painter.setBrush(handle)
        for x in (lo_x, hi_x):
            painter.drawEllipse(Qt.QtCore.QPointF(x, rect.center().y()), 5.5, 5.5)

    def mousePressEvent(self, event):
        if not self.isEnabled():
            return
        x = event.position().x()
        lo_x = self._x_for_value(self._lo)
        hi_x = self._x_for_value(self._hi)
        self._drag = "lo" if abs(x - lo_x) <= abs(x - hi_x) else "hi"
        self._move_handle(x)

    def mouseMoveEvent(self, event):
        if self._drag is not None:
            self._move_handle(event.position().x())

    def mouseReleaseEvent(self, event):
        self._drag = None

    def _move_handle(self, x):
        value = self._value_for_x(x)
        if self._drag == "lo":
            self.setValues(min(value, self._hi), self._hi)
        elif self._drag == "hi":
            self.setValues(self._lo, max(value, self._lo))

# Switch to using white background and black foreground
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


class displayFrameWidget(DisplayDataMixin, DisplayPlotMixin, Qt.QtWidgets.QWidget):
    # Emitted whenever the user changes the plot method combo
    # (Single / Overlay / Waterfall / Sum / Average). Listeners (e.g. the
    # H5Viewer) use this to switch listData selection mode so accumulating
    # plot methods don't require shift/ctrl multi-select.
    sigPlotMethodChanged = Qt.QtCore.Signal(str)

    # Emitted when the user clicks Clear while in a viewer mode.  The host
    # (staticWidget) clears the H5Viewer file-list selection so the cleared plot,
    # the selection and the title all agree -- the displayframe owns the title but
    # not the list selection.
    sigCleared = Qt.QtCore.Signal()

    # Feature flag: during a processing run, keep the last-rendered 2D panels
    # (raw image + cake) on screen instead of blanking them when the in-flight
    # frame's 2D data isn't available yet — so the 2D panels persist exactly
    # like the 1D plot does (which keeps its curve).  Set False to restore the
    # previous behavior (2D panels blank during the run).  This is the single
    # revert switch for the panel-consistency feature.
    PERSIST_2D_DURING_PROCESSING = True

    """Widget for displaying 2D image data and 1D plots from LiveScan
    objects.

    Inherits data-access helpers from ``DisplayDataMixin`` and
    plot-rendering helpers from ``DisplayPlotMixin``.

    attributes:
        curve1: pyqtgraph pen, overall data line
        curve2: pyqtgraph pen, individual frame data line
        histogram: pyqtgraph HistogramLUTWidget, used for adjusting min
            and max level for image
        image: pyqtgraph ImageItem, displays the 2D data
        image_plot: pyqtgraph plot, for 2D data
        image_win: pyqtgraph GraphicsLayoutWidget, layout for the 2D
            data
        imageViewBox: RectViewBox, used to set behavior of the image
            plot
        plot: pyqtgraph plot, for 1D data
        plot_layout: QVBoxLayout, for holding the 1D plotting widgets
        plot_win: pyqtgraph GraphicsLayoutWidget, layout for the 1D
            data
        scan: LiveScan, unused.
        frame: LiveFrame, currently loaded frame object
        frame_ids: List of LiveFrame indices currently loaded
        frames: Dictionary of currently loaded LiveFrames
        data_1d: Dictionary object holding all 1D data in memory
        data_2d: Dictionary object holding all 2D data in memory
        ui: Ui_Form from qtdesigner

    methods:
        get_frames_map_raw: Gets averaged 2D raw data from frames
        get_scan_map_raw: Gets averaged (and normalized) 2D raw data for all images
        get_frames_int_2d: Gets averaged 2D rebinned data from frames
        get_scan_int_2d: Gets overall 2D data for the scan
        update: Updates the displayed image and plot
        update_image: Updates image data based on selections
        update_plot: Updates plot data based on selections
    """

    def __init__(self, scan, frame, frame_ids, frames, data_1d, data_2d,
                 parent=None, data_lock=None, publication_store=None):
        super().__init__(parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        # Top-bar polish (Vivek): width = text + margin, computed from font
        # metrics (no hand-tuned pixels, nothing elided), ONE height for the
        # whole row (combos and buttons have different native heights on
        # macOS).  NOTE: AdjustToContents + editable-centered combos were
        # tried and reverted -- they broke the popups on macOS.
        _ROW_H = 28
        # Match the bottom (plot-controls) toolbar height: frame_top was
        # capped at 35 while imageToolbar renders at 40, making the top row
        # visibly shorter.
        self.ui.frame_top.setMinimumSize(Qt.QtCore.QSize(0, 40))
        self.ui.frame_top.setMaximumSize(Qt.QtCore.QSize(16777215, 40))
        # Pin BOTH cluster containers to one height (34 inside the 40 row)
        # so their painted boxes match -- left shrink-wrapped a few px
        # shorter than right under the mac style.  Children stay 28 and
        # center inside.
        for _f in (self.ui.frame_4, self.ui.frame_6):
            _f.setMinimumSize(Qt.QtCore.QSize(0, 34))
            _f.setMaximumSize(Qt.QtCore.QSize(16777215, 34))
            # Horizontal policy Maximum = shrink-wrap: each cluster hugs its
            # content (instead of splitting the spare width 50/50 with the
            # other cluster) and can be squeezed below it; the spare space
            # all goes to the title in the middle.
            _sp = _f.sizePolicy()
            _sp.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.Maximum)
            _f.setSizePolicy(_sp)
        # Borderless boxes (Vivek): no frame lines on the cluster/title
        # containers or the title label.
        for _f in (self.ui.frame_top, self.ui.frame_4, self.ui.frame_5,
                   self.ui.frame_6, self.ui.labelCurrent):
            try:
                _f.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
                _f.setLineWidth(0)
            except Exception:
                logger.debug("frame border clear failed", exc_info=True)
        # The title takes the stretch: lift labelCurrent's generated 600px
        # maximum (capped the center on wide windows) and let it shrink
        # small; frame_5 (its container) unconstrained likewise.
        self.ui.labelCurrent.setMinimumSize(Qt.QtCore.QSize(0, 0))
        self.ui.labelCurrent.setMaximumSize(
            Qt.QtCore.QSize(16777215, 16777215))
        if hasattr(self.ui, 'frame_5'):
            self.ui.frame_5.setMinimumSize(Qt.QtCore.QSize(0, 0))
            self.ui.frame_5.setMaximumSize(
                Qt.QtCore.QSize(16777215, 34))
            _lay = self.ui.frame_5.layout()
            if _lay is not None:
                for _i in range(_lay.count()):
                    _cw = _lay.itemAt(_i).widget()
                    if _cw is not None:
                        _lay.setAlignment(_cw, pyQt.AlignVCenter)
        # Center EVERY top-row cell vertically: frame_4 (Norm Channel/Set
        # Bkg) anchored top while frame_6 (Log/Raw Image) was centered, so
        # the two clusters sat ~3px apart -- the 'Log looks low' symptom.
        _top_lay = self.ui.frame_top.layout()
        if _top_lay is not None:
            for _i in range(_top_lay.count()):
                _cw = _top_lay.itemAt(_i).widget()
                if _cw is not None:
                    _top_lay.setAlignment(_cw, pyQt.AlignVCenter)
        for _c in (self.ui.normChannel, self.ui.scale, self.ui.cmap):
            displayFrameWidget._fit_combo_width(_c, max_w=130)
            _c.setFixedHeight(_ROW_H)
        # The colormap selector shows a short value ("Default") and lives in the
        # compact scale pill — trim it ~15% so the pill is tighter (Vivek).
        _cmw = self.ui.cmap.maximumWidth()
        self.ui.cmap.setFixedWidth(int(_cmw * 0.85) if 0 < _cmw < 16777215 else 110)
        # Widen the BG button (+7%, then another +10% = x1.177) so the 'Clear BG'
        # label has comfortable room (Vivek).
        displayFrameWidget._fit_button_width(self.ui.setBkg, scale=1.177)
        self.ui.setBkg.setFixedHeight(_ROW_H)
        # The SCALE combo moves into the Options popup ('Other' row); the
        # colormap stays in the bar.  One checkable 'Log' toggle = Linear <->
        # Log via the scale combo, so the existing Log path (incl. its
        # negative/small-value shift in pgImageWidget.update_image) is reused
        # verbatim.  The right cluster is rebuilt in final order (Raw |
        # colormap | Log, Log at the corner) once the Raw button exists.
        self.ui.horizontalLayout_9.removeWidget(self.ui.scale)
        self.ui.scale.setParent(None)
        self._logBtn = QtWidgets.QPushButton('Log')
        self._logBtn.setCheckable(True)
        self._logBtn.setFixedHeight(_ROW_H)
        displayFrameWidget._fit_button_width(self._logBtn)
        self._logBtn.setFocusPolicy(pyQt.StrongFocus)
        _parent_layout = self.ui.frame_6.parentWidget().layout()
        if _parent_layout is not None:
            _parent_layout.setAlignment(self.ui.frame_6, pyQt.AlignVCenter)
        # Options now hosts the scale/cmap combos -- it must be reachable
        # from launch (the generated UI starts it disabled until the first
        # 1D layout setup enables it).
        self.ui.wf_options.setEnabled(True)
        self._logBtn.toggled.connect(
            lambda on: self.ui.scale.setCurrentText('Log' if on else 'Linear'))
        # Keep the toggle honest when the combo changes (e.g. Sqrt in the
        # Options popup, or a restored session).
        def _sync_log_btn(*_a):
            self._logBtn.blockSignals(True)
            self._logBtn.setChecked(self.ui.scale.currentText() == 'Log')
            self._logBtn.blockSignals(False)
        self.ui.scale.currentIndexChanged.connect(_sync_log_btn)
        # Shared reentrant lock guarding data_1d / data_2d.  When created
        # standalone (tests, viewer mode) fall back to a private lock.
        self.data_lock = data_lock if data_lock is not None else threading.RLock()
        self.publication_store = publication_store
        self._init_data_objects(scan, frame, frame_ids, frames, data_1d, data_2d)
        self._init_display_panes()
        self._init_plot_panes()
        self._connect_signals()
        self._init_controls()
        self._reflow_controls()
        self._init_intensity_controls()
        self._set_tooltips()
        self._set_equal_primary_panel_heights()
        self._install_share_geometry_hooks()

    # ── Initialization helpers ─────────────────────────────────────

    def _set_tooltips(self):
        """Hover tooltips for the display-frame controls (PySide6 setToolTip)."""
        tips = {
            'normChannel': 'Normalize intensity by this monitor/counter channel.',
            'setBkg': 'Use the current frame(s) as a background to subtract.',
            'scale': 'Intensity scale: Linear / Log / Sqrt.',
            'cmap': 'Colormap for the 2D images.',
            'imageUnit': '2D cake radial axis: Q-χ or 2θ-χ.',
            'shareAxis': "Lock the 1D plot x-axis to the 2D cake's x-axis.",
            'plotUnit': '1D plot x-axis (Q, 2θ, or χ from the 2D cake).',
            'plotMethod': 'Combine frames: Single / Overlay / Waterfall / '
                          'Sum / Average.',
            'slice': 'Restrict the 1D pattern to a χ range (needs 2D data).',
            'slice_center': 'Center of the χ slice (degrees).',
            'slice_width': 'Width of the χ slice (degrees).',
            'wf_options': 'Waterfall / Overlay / Legend options.',
            'clear_1D': 'Clear accumulated overlay/waterfall curves.',
        }
        for name, tip in tips.items():
            w = getattr(self.ui, name, None)
            if w is not None:
                w.setToolTip(tip)
        if getattr(self, '_showImageBtn', None) is not None:
            self._showImageBtn.setToolTip(
                'Show the raw detector image for the selected frame.')

    def _reflow_controls(self):
        """Consolidate the 1D plot controls into the middle bar
        (``imageToolbar``, which sits between the 2D cake and the 1D plot)
        and collapse the now-empty bottom bar so the 1D plot gets that
        height back.

        Left→right the middle bar becomes: the 1D controls grouped by
        function (unit + X-Range, then Single/Overlay + Options, then
        Legend + Clear), a stretch, then the 2D-only controls (Share Axis,
        2D unit) at the far right under the cake.  The Offset control folds
        into the Options popup, so it leaves the bar entirely.

        Per-mode show/hide of the 2D-only controls + slice is handled by
        :meth:`_set_2d_controls_visible`.
        """
        mid = self.ui.horizontalLayout_2     # imageToolbar (middle bar)
        bot = self.ui.horizontalLayout       # plotToolBar (bottom, emptied)

        # Detach whatever is currently in the middle bar (imageUnit,
        # shareAxis, spacers) so we can re-add in the new order.
        while mid.count():
            mid.takeAt(0)

        # Offset + Legend fold into the Options popup — pull them out of
        # the toolbar (they get re-parented into the dialog when it's built).
        for w in (self.ui.yOffsetLabel, self.ui.yOffset, self.ui.showLegend):
            bot.removeWidget(w)
            w.setParent(None)

        # Move the remaining 1D controls out of the bottom bar.
        ones = (self.ui.plotUnit, self.ui.slice, self.ui.slice_center,
                self.ui.slice_width, self.ui.plotMethod, self.ui.wf_options,
                self.ui.clear_1D)
        for w in ones:
            bot.removeWidget(w)
        while bot.count():
            bot.takeAt(0)

        # Rebuild the middle bar: 1D controls, stretch, then 2D controls.
        for w in ones:
            mid.addWidget(w)
        mid.addStretch(1)
        mid.addWidget(self.ui.shareAxis)
        mid.addWidget(self.ui.imageUnit)

        # The bottom bar is empty now.  It is collapsed in normal modes and
        # reused as the Image Viewer top intensity row.
        bot.setContentsMargins(0, 0, 8, 0)
        bot.setSpacing(8)
        bot.addStretch(1)
        self._prepare_viewer_intensity_toolbar()

    def _prepare_viewer_intensity_toolbar(self):
        """Reuse the emptied legacy plot toolbar as Image Viewer's top row."""
        toolbar = self.ui.plotToolBar
        toolbar.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        toolbar.setLineWidth(0)
        toolbar.setMidLineWidth(0)
        try:
            layout = self.ui.verticalLayout_3
            two_d = self.ui.twoDWindow
            target = layout.indexOf(two_d)
            if target >= 0 and layout.indexOf(toolbar) != target:
                layout.insertWidget(target, toolbar)
        except Exception:
            logger.debug("viewer intensity toolbar placement failed", exc_info=True)

        # Start hidden; PANEL_LAYOUT owns visibility/height from here on.
        self.ui.plotToolBar.setMaximumHeight(0)
        self.ui.plotToolBar.setMinimumHeight(0)
        self.ui.plotToolBar.setVisible(False)

    def _set_2d_controls_visible(self, visible: bool):
        """Show/hide the controls that only make sense with 2D data:
        the Share Axis + 2D-unit buttons and the X-Range slice trio
        (the slice is computed from the 2D cake).  The plain 1D controls
        (unit, Single/Overlay, Options, Legend, Clear) stay visible."""
        for w in (self.ui.shareAxis, self.ui.imageUnit, self.ui.slice,
                  self.ui.slice_center, self.ui.slice_width):
            w.setVisible(visible)

    def _set_middle_1d_controls_visible(self, visible: bool):
        """Show/hide toolbar controls that are analysis controls, not viewer
        intensity controls."""
        for w in (self.ui.plotUnit, self.ui.plotMethod,
                  self.ui.wf_options, self.ui.clear_1D):
            w.setVisible(visible)

    def _init_intensity_controls(self):
        """Viewer / 1D-only display-scale controls hosted in the middle row."""
        self._intensityWidget = QtWidgets.QFrame(self.ui.imageToolbar)
        self._intensityWidget.setObjectName("viewerIntensityControls")
        self._intensityWidget.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._intensityWidget.setStyleSheet("""
        QFrame#viewerIntensityControls {
            border: 0px;
            background: transparent;
        }
        QFrame#viewerIntensityControls QLabel {
            border: 0px;
            background: transparent;
        }
        """)
        lay = QtWidgets.QHBoxLayout(self._intensityWidget)
        lay.setContentsMargins(0, 0, 8, 0)
        lay.setSpacing(8)
        self._intensityLabel = QtWidgets.QLabel("Intensity", self._intensityWidget)
        self._intensitySlider = _IntensityRangeSlider(self._intensityWidget)
        self._intensityAuto = QtWidgets.QPushButton("Autoscale", self._intensityWidget)
        self._intensityAuto.setCheckable(True)
        self._intensityAuto.setChecked(True)
        self._intensityAuto.setFixedHeight(28)
        displayFrameWidget._fit_button_width(self._intensityAuto, pad=34)
        self._intensityAuto.setMinimumWidth(112)
        lay.addWidget(self._intensityLabel)
        lay.addWidget(self._intensitySlider)
        lay.addWidget(self._intensityAuto)
        self._move_intensity_controls_for_mode(Mode.INT_1D)
        self._intensityWidget.setVisible(False)
        self._intensitySlider.sigRangeChanged.connect(self._on_intensity_range_changed)
        self._intensityAuto.toggled.connect(self._on_intensity_autoscale_toggled)

    def _move_intensity_controls_for_mode(self, mode):
        """Host the intensity controls on the row that is visible for mode."""
        if not hasattr(self, "_intensityWidget"):
            return
        try:
            if mode is Mode.IMAGE_VIEWER:
                self.ui.horizontalLayout.addWidget(self._intensityWidget)
            else:
                idx = self.ui.horizontalLayout_2.indexOf(self.ui.shareAxis)
                if idx < 0:
                    idx = self.ui.horizontalLayout_2.count()
                self.ui.horizontalLayout_2.insertWidget(idx, self._intensityWidget)
        except Exception:
            logger.debug("intensity control placement failed", exc_info=True)

    def _set_equal_primary_panel_heights(self):
        """Give the 2D and 1D primary panels equal splitter space."""
        splitter = getattr(self.ui, "splitter", None)
        if splitter is None:
            return

        def _apply():
            try:
                if (
                    self.ui.imageWindow.maximumHeight() != 0
                    and self.ui.plotWindow.maximumHeight() != 0
                ):
                    splitter.setSizes([1, 1])
            except RuntimeError:
                pass

        _apply()
        try:
            Qt.QtCore.QTimer.singleShot(0, _apply)
        except RuntimeError:
            pass

    def _init_data_objects(self, scan, frame, frame_ids, frames, data_1d, data_2d):
        """Initialize data references, plotting state, and index tracking."""
        self.ui.slice.setText(Chi)

        # Plotting parameters
        self.ui.cmap.clear()
        self.ui.cmap.addItems(['Default'] + plt.colormaps())
        self.ui.cmap.setCurrentIndex(0)
        self.cmap = self.ui.cmap.currentText()
        self.plotMethod = self.ui.plotMethod.currentText()
        self.scale = self.ui.scale.currentText()
        self.wf_yaxis = 'Frame #'
        self.wf_start = 0
        self.wf_stop = None  # None → slice through the last frame
        self.wf_step = 1

        # Data object references
        self.scan = scan
        self.frame = frame
        self.frame_ids = frame_ids
        self.frames = frames
        self.frame_names = []
        self.overlaid_idxs = []
        # Flip stage 2: the payload-carried Overlay/Waterfall accumulator
        # (generation-keyed WaterfallHistory). Lives alongside the legacy triple
        # until the render path delegates to it (stage 3/4); reset at the same
        # accumulator-lifecycle sites (new_scan, clear_overlay).
        self._waterfall_history = None
        self.data_1d = data_1d
        self.data_2d = data_2d
        self.bkg_1d = 0.
        self.bkg_2d = 0.
        self.bkg_map_raw = 0.
        # XYE-viewer background: (x, y) of the averaged background pattern, or None.
        # The viewer files have per-file grids, so unlike the scan-grid bkg_1d the
        # XYE bkg carries its own x and is interpolated onto each trace at render.
        self._bkg_xye = None
        self._norm_channel_map = {}

        # Viewer mode: None (normal), 'image', or 'xye'
        self.viewer_mode = None
        self._wrangler = None

        # Frame index tracking
        self.idxs = []
        self.idxs_1d = []
        self.idxs_2d = []
        self.overall = False

        # Stage 2: monotonic display generation.  Bumped on the events that
        # must invalidate a stale render — mode switch, new selection, new
        # scan/file load — so a worker result computed against an old
        # generation can be dropped (full enforcement lands in Stage 5).
        self.display_generation = 0
        # Persistent stitch display: '1d'/'2d' while a Stitch mode is selected in
        # the wrangler dropdown AND a result exists; None ⇒ the per-frame view.
        # Set by the host via sigStitchModeChanged + stitch_thread_finished;
        # gated by result-existence in _active_stitch_mode.
        self.stitch_display_mode = None
        # D2 (greenfield Phase 3): off-GUI-thread rehydration of evicted frames.
        # The store reads an evicted frame's heavy payload through THIS widget's
        # disk reader; the background worker calls it off the GUI thread.  Async
        # is OFF by default (synchronous blocking reads — preserves the headless
        # test behaviour); the live app turns it on via enable_async_hydration().
        self._hydration_worker = None
        self._async_hydration_enabled = False
        # Step 7b: off-GUI-thread whole-scan aggregation (Sum/Average over a scan
        # longer than the bounded store).  The worker computes from the on-disk
        # stack ⊕ in-memory tail; results are cached per (dim, method, channel)
        # with the generation they were computed under (stale ones are dropped).
        # Shares the async on/off flag with hydration (enable_async_hydration);
        # OFF => computed synchronously inline so headless renders see it at once.
        self._aggregation_worker = None
        self._agg_cache: dict = {}
        self._agg_pending: set = set()
        if self.publication_store is not None:
            try:
                self.publication_store.set_hydrator(self._rehydrate_publication)
            except Exception:
                logger.debug("set_hydrator failed", exc_info=True)
        # True once an empty/no-data update has blanked all panels; reset
        # when a data render draws.  Lets update() no-op on repeated empty
        # updates instead of re-clearing every time.
        self._display_blanked = False
        self._last_selection_sig = None

        # True while a wrangler/integrator run is in progress.  Set by
        # staticWidget at run start, cleared at run end (incl. Stop).  Drives
        # the PERSIST_2D_DURING_PROCESSING feature: while a run is active, the
        # 2D panels keep their last-rendered content instead of blanking when
        # the in-flight frame's 2D data isn't available yet.
        self._processing_active = False

        # Stage 5: register the mode controllers (Scan/ImageViewer/XYEViewer)
        # into the open registry; _live_display_state dispatches through them.
        register_default_controllers()

        self.get_idxs()

        # Plotting variables
        self.normChannel = None
        self.overlay = None
        self._last_plot_unit = -1
        self._plot_axis_info = []  # populated by set_axes()
        self._was_skip_2d = False  # track 1D-only state for transitions
        self._payload_x_axis_label = None
        self._payload_y_axis_label = None
        self._using_publication_plot_payload = False
        self._plot_autorange_requested = False

        # Cached display data
        self.image_data = (None, None)
        self.binned_data = (None, None)
        self.plot_data = [np.zeros(0), np.zeros(0)]
        self.plot_data_range = [[0, 0], [0, 0]]

    def _init_display_panes(self):
        """Set up the raw image and binned 2D image display panes."""
        # Raw image pane
        self.image_layout = Qt.QtWidgets.QHBoxLayout(self.ui.imageFrame)
        self.image_layout.setContentsMargins(0, 0, 0, 0)
        self.image_layout.setSpacing(0)
        self.image_widget = pgImageWidget(lockAspect=True, raw=True)
        self.image_layout.addWidget(self.image_widget)

        # Binned (regrouped) image pane
        self.binned_layout = Qt.QtWidgets.QHBoxLayout(self.ui.binnedFrame)
        self.binned_layout.setContentsMargins(0, 0, 0, 0)
        self.binned_layout.setSpacing(0)
        self.binned_widget = pgImageWidget()
        self.binned_layout.addWidget(self.binned_widget)

    def _init_plot_panes(self):
        """Set up 1D plot, waterfall plot, and mouse tracking."""
        self.plot_layout = Qt.QtWidgets.QHBoxLayout(self.ui.plotFrame)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_layout.setSpacing(0)

        # 1D plot
        self.plot_win = pg.GraphicsLayoutWidget()
        self.plot_layout.addWidget(self.plot_win)
        self.plot_viewBox = RectViewBox()
        self.plot = self.plot_win.addPlot(viewBox=self.plot_viewBox)
        # Seaborn-darkgrid + talk-context styling: gridlines on,
        # tick/label fonts ~11pt.  Background colour comes from
        # ``apply_dark_theme``'s pg.setConfigOption.
        from xdart.gui.themes import apply_seaborn_plot_style
        apply_seaborn_plot_style(self.plot)
        self.curves = []
        self.legend = self.plot.addLegend()
        from PySide6.QtWidgets import QGraphicsItem
        self.legend.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, False)

        self.pos_label = pg.LabelItem(justify='right')
        self.plot_win.addItem(self.pos_label)
        self.pos_label.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(-20, 10))
        self.pos_label.setFixedWidth(1)
        self.trackMouse()

        # Waterfall plot
        self.wf_widget = pgImageWidget()
        self.setup_wf_widget()
        self.plot_layout.addWidget(self.wf_widget)

        if self.plotMethod == 'Waterfall':
            self.plot_win.setParent(None)
            self.plot_layout.addWidget(self.wf_widget)
        else:
            self.wf_widget.setParent(None)
            self.plot_layout.addWidget(self.plot_win)

    def _connect_signals(self):
        """Wire all signal/slot connections for display controls."""
        # Global controls
        self.ui.normChannel.activated.connect(self.normUpdate)
        self.ui.setBkg.clicked.connect(self.setBkg)
        self.ui.scale.currentIndexChanged.connect(self.update_views)
        self.ui.cmap.currentIndexChanged.connect(self.update_views)
        # shareAxis / showLegend / slice are checkable QPushButtons now —
        # use ``toggled`` (bool) rather than the QCheckBox-only stateChanged.
        # Share Axis runs a synchronous, geometry-heavy render + relink; during a
        # live scan that momentarily freezes the GUI while the writer keeps
        # emitting per-frame display events (the named stall).  _on_share_axis_changed
        # DEFERS that work to the next event-loop pass ONLY while processing, so the
        # click returns immediately; idle (incl. headless tests) stays synchronous.
        self.ui.shareAxis.toggled.connect(self._on_share_axis_changed)

        # 2D image controls — the Q-χ / 2θ-χ toggle re-renders through the
        # payload path (cake_image owns the on-the-fly Q↔2θ conversion now), so
        # the cake unit is consistent on every render, not just this redraw.
        self.ui.imageUnit.activated.connect(self.update)
        self.ui.imageUnit.activated.connect(self._update_slice_range)

        # 1D plot controls
        self.ui.plotMethod.currentIndexChanged.connect(self._on_plotMethod_changed)
        self.ui.yOffset.valueChanged.connect(self.update_plot_view)
        self.ui.plotUnit.activated.connect(self._on_plotUnit_changed)
        self.ui.plotUnit.activated.connect(self.request_plot_autorange)
        # Stage 4: drive the redraw through the payload pipeline (update) rather
        # than the legacy update_plot; _on_plotUnit_changed / request_plot_
        # autorange still set the label + autorange request consumed by update.
        self.ui.plotUnit.activated.connect(self.update)
        self.ui.showLegend.toggled.connect(self.update_legend)
        self.ui.slice.toggled.connect(self._sync_slice_controls)
        self.ui.slice.toggled.connect(self.update)
        self.ui.slice.toggled.connect(self._update_slice_range)
        self.ui.slice_center.valueChanged.connect(self.update_plot_range)
        self.ui.slice_width.valueChanged.connect(self.update_plot_range)
        self.ui.wf_options.clicked.connect(self.popup_wf_options)

        # Action buttons.  (The in-panel Save buttons were removed — use
        # pyqtgraph's right-click Export, or File ▸ Export.  The
        # save_image / save_1D methods are still wired to those menu
        # actions in static_scan_widget.)
        self.ui.clear_1D.clicked.connect(self.clear_1D)

    def _init_controls(self):
        """Initialize image units, waterfall options, and preview button."""
        self.set_axes()
        if hasattr(self, 'refresh_norm_channels'):
            self.refresh_norm_channels()
        self._set_slice_range(initialize=True)

        # Waterfall options popup widgets
        self.wf_dialog = QDialog()
        self.wf_yaxis_widget = QCombo()
        self.wf_start_widget = QtWidgets.QDoubleSpinBox()
        self.wf_stop_widget = QtWidgets.QDoubleSpinBox()
        self.wf_step_widget = QtWidgets.QDoubleSpinBox()
        self.wf_accept_button = QtWidgets.QPushButton('Okay')
        self.wf_cancel_button = QtWidgets.QPushButton('Cancel')

        # Raw image preview button
        self._showImageBtn = QtWidgets.QPushButton('Raw')
        self._showImageBtn.setFixedHeight(28)   # match the top-row controls
        displayFrameWidget._fit_button_width(self._showImageBtn)
        self._showImageBtn.setToolTip('Show raw image preview for selected frame')
        self._showImageBtn.setFocusPolicy(pyQt.StrongFocus)
        # The colormap selector ("Default") and the Log scale toggle are always
        # shown together, so group them into one segmented pill (shared border,
        # a hairline divider between, flush edges).  Raw stays a separate button
        # to the left of the pill.
        self._scaleGroup = QtWidgets.QFrame()
        self._scaleGroup.setObjectName('displayScaleGroup')
        _sg = QtWidgets.QHBoxLayout(self._scaleGroup)
        _sg.setContentsMargins(0, 0, 0, 0)
        _sg.setSpacing(0)
        _scale_div = QtWidgets.QFrame()
        _scale_div.setObjectName('displayScaleDivider')
        _scale_div.setFrameShape(QtWidgets.QFrame.VLine)
        _sg.addWidget(self.ui.cmap)
        _sg.addWidget(_scale_div)
        _sg.addWidget(self._logBtn)
        # Rebuild the right cluster in its final order: Raw | [colormap ┊ Log].
        _l9 = self.ui.horizontalLayout_9
        while _l9.count():
            _l9.takeAt(0)
        for _w in (self._showImageBtn, self._scaleGroup):
            _l9.addWidget(_w)
            _l9.setAlignment(_w, pyQt.AlignVCenter)
        _l9.setSpacing(8)
        self._showImageBtn.clicked.connect(self._show_image_preview)
        self._showImageBtn.setVisible(False)
        self._image_preview_dialog = None
        self._image_preview_widget = None

    # ── Index management ──────────────────────────────────────────

    def get_idxs(self):
        """ Return selected frame indices.

        Thread-safety: snapshots of data_1d / data_2d keys are taken under
        ``data_lock`` to avoid racing with integrator / file-handler worker
        threads that mutate these dicts concurrently.
        """
        self.idxs, self.idxs_1d, self.idxs_2d = [], [], []
        if len(self.frame_ids) == 0 or self.frame_ids[0] == 'No data':
            return

        if self.viewer_mode is not None:
            # Viewer modes (image/xye/nexus) are file browsers: the loaded
            # ``frame_ids`` ARE the selection and must never be resolved
            # against the integration scan's frame index (§8 invariant —
            # viewer controllers never consult ``scan.frames``).  That index
            # may be stale-populated from a prior run; in particular, opening
            # the SAME file in Image Viewer that was just integrated via an
            # Int 1D (XYE) batch makes ``len(frame_ids) == len(scan.frames)``,
            # which flips ``resolve_selection``'s ``overall`` heuristic True
            # and rebases ``ids`` onto the stale scan labels.  Those don't
            # intersect the viewer's loaded data keys, so ``idxs`` come back
            # empty and the panel renders blank.
            try:
                ids = tuple(sorted(int(i) for i in self.frame_ids))
            except (TypeError, ValueError):
                return
            self.overall = False
            with self.data_lock:
                data_1d_keys = set(self.data_1d.keys())
                data_2d_keys = set(self.data_2d.keys())
        else:
            with self.data_lock:
                with self.scan.scan_lock:
                    # Stage 1: selection logic is the pure ``resolve_selection``
                    # / ``resolve_render_ids`` (unit-tested headlessly).
                    try:
                        ids, self.overall = resolve_selection(
                            self.frame_ids, self.scan.frames.index)
                    except ValueError:
                        return

            store = getattr(self, 'publication_store', None)
            if store is not None:
                data_1d_keys = set()
                data_2d_keys = set()
                try:
                    from .display_publication import publication_availability
                    pub_1d, pub_2d, _raw = publication_availability(
                        store, labels=ids)
                    data_1d_keys |= set(pub_1d)
                    data_2d_keys |= set(pub_2d)
                except Exception:
                    logger.debug("publication availability lookup failed",
                                 exc_info=True)
            else:
                with self.data_lock:
                    # Legacy/test-host path only: normal scan display no longer
                    # treats these mirrors as loaded when a publication store is
                    # present.
                    data_1d_keys = set(self.data_1d.keys())
                    data_2d_keys = set(self.data_2d.keys())

        self.idxs = list(ids)
        # ``ids`` is already the effective set (all-or-selected), so intersect
        # it directly with the loaded keys for each panel.
        self.idxs_1d = list(resolve_render_ids(ids, False, (), data_1d_keys))
        self.idxs_2d = list(resolve_render_ids(ids, False, (), data_2d_keys))

    # ── Display generation (Stage 2) ──────────────────────────────────

    def _bump_display_generation(self):
        """Advance the monotonic display generation.  A render/worker result
        stamped with an older generation is stale and may be dropped (full
        enforcement: Stage 5)."""
        self.display_generation += 1
        pending = getattr(self, "_agg_pending", None)
        if pending is not None:
            pending.clear()
        return self.display_generation

    # ── D2: off-GUI-thread frame hydration (greenfield Phase 3) ──────────────
    def enable_async_hydration(self) -> None:
        """Turn on background rehydration of evicted frames (live app only).

        Headless tests leave this OFF so the render path keeps its synchronous
        blocking reads (their assertions expect data on the first render); the
        live app calls this once so scroll-back / Set-Bkg no longer freeze the
        GUI thread on a ~5 s ``.nxs`` open."""
        self._async_hydration_enabled = True
        self._ensure_hydration_worker()

    def _ensure_hydration_worker(self):
        if self._hydration_worker is None and self.publication_store is not None:
            from .frame_hydration_worker import FrameHydrationWorker
            # parent=None (NOT self): a QThread parented to the widget is
            # C++-deleted when the widget is destroyed even if its thread is
            # still running ('QThread: Destroyed while running').  With no Qt
            # parent the Python handle (kept by stop_hydration_worker on a slow
            # read) is the sole owner, so the thread is never force-deleted mid
            # read (codex P1).
            worker = FrameHydrationWorker(self.publication_store, parent=None)
            worker.sigHydrated.connect(self._on_frame_hydrated)
            worker.start()
            self._hydration_worker = worker
        return self._hydration_worker

    def _request_frame_hydration(self, label) -> None:
        """Queue a background rehydration for an evicted frame (no-op unless the
        live app enabled async hydration)."""
        if not self._async_hydration_enabled:
            return
        worker = self._ensure_hydration_worker()
        if worker is not None:
            worker.request(label, self.display_generation)

    def _on_frame_hydrated(self, label, generation) -> None:
        """A background hydration finished: the heavy payload is now resident in
        the store.  Drop a stale result (the selection/mode moved on), else
        re-render so the panel upgrades from its thumbnail to the full frame.

        THROTTLED: a unit-switch REBUILD backfills hundreds of evicted frames,
        firing this once per frame.  Render immediately when we haven't rendered
        recently (sparse scroll-back hydration stays instant), but coalesce a
        burst into one render per ~120 ms so the progressive fill stays smooth
        instead of O(N) renders (the data is already resident, so the short
        trailing delay is invisible)."""
        if int(generation) != self.display_generation:
            return
        import time
        now = time.monotonic()
        last = getattr(self, "_last_hydration_render", 0.0)
        self._pending_hydration_render = True
        if now - last >= 0.12:
            self._last_hydration_render = now
            self._pending_hydration_render = False
            try:
                self.update()
            except Exception:
                logger.debug("re-render after hydration failed for %s", label,
                             exc_info=True)
            return
        # Inside the throttle window: coalesce into one trailing render.
        if getattr(self, "_hydration_render_scheduled", False):
            return
        self._hydration_render_scheduled = True

        def _go():
            self._hydration_render_scheduled = False
            if not getattr(self, "_pending_hydration_render", False):
                return
            self._pending_hydration_render = False
            self._last_hydration_render = time.monotonic()
            try:
                self.update()
            except Exception:
                logger.debug("re-render after hydration failed", exc_info=True)
        Qt.QtCore.QTimer.singleShot(120, _go)

    def stop_hydration_worker(self) -> None:
        """Stop + join the background worker (idempotent; call at teardown).

        P1: disconnect the signal first so a late cross-thread emit can't
        re-enter a half-torn-down widget; and only release the handle once the
        thread has actually stopped — if a slow ``.nxs`` read is still in flight
        we KEEP the handle (and log) rather than let the QThread object be
        destroyed while its thread runs ('QThread: Destroyed while running')."""
        worker = self._hydration_worker
        if worker is None:
            return
        try:
            worker.sigHydrated.disconnect(self._on_frame_hydrated)
        except Exception:
            pass
        stopped = True
        try:
            stopped = worker.stop()
        except Exception:
            logger.debug("hydration worker stop failed", exc_info=True)
            stopped = False
        if stopped:
            self._hydration_worker = None
        else:
            logger.warning("frame-hydration worker did not stop within timeout; "
                           "keeping the handle so the QThread isn't destroyed "
                           "while its read is still in flight")

    # ── Step 7b: off-GUI-thread whole-scan aggregation ───────────────────────
    def _ensure_aggregation_worker(self):
        if self._aggregation_worker is None:
            from .aggregation_worker import AggregationWorker
            # parent=None for the same reason as the hydration worker: keep the
            # Python handle the sole owner so the QThread is never C++-deleted
            # while a chunked read is still in flight.
            worker = AggregationWorker(parent=None)
            worker.sigAggregated.connect(self._on_aggregated)
            worker.start()
            self._aggregation_worker = worker
        return self._aggregation_worker

    def _aggregate_display_is_primary(self, scan, dim: str) -> bool:
        """True when the displayed mode is the top-level persisted stack.

        Whole-scan aggregates read the primary on-disk stack.  For GI scans that
        is safe only when the current display mode matches the persisted
        ``gi_config`` mode; otherwise a non-primary mode may be partial/lazy and
        must not fall back to a bounded resident subset.
        """
        from xdart.modules.scan_aggregate import mode_aggregation_allowed

        if not getattr(scan, "gi", False):
            return mode_aggregation_allowed(None, None)

        gi_config = getattr(scan, "gi_config", None) or {}
        key = "gi_mode_1d" if dim == "1d" else "gi_mode_2d"
        # FAIL CLOSED: a GI scan whose gi_config does not record the primary mode
        # (e.g. a .nxs written before gi_config existed, reloaded) must NOT serve a
        # whole-scan aggregate.  Defaulting primary=displayed would make the gate
        # ALWAYS pass, defeating the anti-truncation protection the moment the user
        # switches to a non-primary GI mode — so defer (blank) instead.
        if key not in gi_config:
            return False
        if dim == "1d":
            displayed = getattr(scan, "bai_1d_args", {}).get(
                "gi_mode_1d", "q_total")
        else:
            displayed = getattr(scan, "bai_2d_args", {}).get(
                "gi_mode_2d", "qip_qoop")
        return mode_aggregation_allowed(displayed, gi_config[key])

    def _whole_scan_aggregate(self, *, dim, method):
        """Return the whole-scan Sum/Average for the current Overall selection as
        an ``Aggregated1D``/``Aggregated2D``, or ``None`` when it isn't available
        this render (defer to the resident-store / legacy path).

        Primary-mode-scoped (ADR-0003): only served when the displayed mode IS
        the primary on-disk stack.  For GI, the persisted ``scan.gi_config``
        records that primary mode; non-primary modes defer rather than silently
        aggregating a partial stack or a bounded in-memory subset.
        Async (live): dispatch to the worker and return None now (re-render on
        completion).  Sync (headless): compute inline."""
        scan = getattr(self, "scan", None)
        if scan is None:
            return None
        gate = getattr(self, "_aggregate_display_is_primary", None)
        if gate is None:  # unbound duck-widget tests call this method directly
            gate = displayFrameWidget._aggregate_display_is_primary.__get__(
                self, type(self))
        if not gate(scan, str(dim)):
            return None
        norm_channel = None
        try:
            norm_channel = self.get_normChannel() or None
        except Exception:
            norm_channel = None
        generation = self.display_generation
        key = (dim, method, norm_channel)
        cached = self._agg_cache.get(key)
        if cached is not None and cached[0] == generation:
            return cached[1]
        if getattr(self, "_async_hydration_enabled", False):
            worker = self._ensure_aggregation_worker()
            pending_key = (key, generation)
            if worker is not None and pending_key not in self._agg_pending:
                self._agg_pending.add(pending_key)
                worker.request(key, generation, scan, dim, method, norm_channel)
            return None                  # not ready this render
        # Headless / synchronous: compute inline so the first render has it.
        from xdart.modules.scan_aggregate import (
            whole_scan_aggregate_1d, whole_scan_aggregate_2d)
        fn = whole_scan_aggregate_2d if dim == "2d" else whole_scan_aggregate_1d
        try:
            result = fn(scan, method=method, norm_channel=norm_channel)
        except Exception:
            logger.debug("inline aggregation failed for %s", key, exc_info=True)
            result = None
        if result is not None:
            self._agg_cache[key] = (generation, result)
        return result

    def _on_aggregated(self, key, generation, result) -> None:
        """A background aggregate finished: cache it (on the GUI thread) and
        re-render, unless a newer generation has superseded it."""
        self._agg_pending.discard((key, int(generation)))
        if int(generation) != self.display_generation:
            return
        if result is None:
            # None means "not ready / unavailable for this attempt", not a
            # terminal empty aggregate.  Do not cache it or self-trigger a
            # repaint loop; the next scan/display update will retry.
            return
        self._agg_cache[key] = (generation, result)
        try:
            self.update()
        except Exception:
            logger.debug("re-render after aggregation failed for %s", key,
                         exc_info=True)

    def stop_aggregation_worker(self) -> None:
        """Stop + join the aggregation worker (idempotent; teardown)."""
        worker = self._aggregation_worker
        if worker is None:
            return
        try:
            worker.sigAggregated.disconnect(self._on_aggregated)
        except Exception:
            pass
        stopped = True
        try:
            stopped = worker.stop()
        except Exception:
            logger.debug("aggregation worker stop failed", exc_info=True)
            stopped = False
        if stopped:
            self._aggregation_worker = None
        else:
            logger.warning("aggregation worker did not stop within timeout; "
                           "keeping the handle so the QThread isn't destroyed "
                           "while its read is still in flight")

    def _note_selection_generation(self):
        """Bump the generation when the *effective* selection changes.

        Keyed on the resolved ``idxs`` (+ overall), so it is robust to how
        ``frame_ids`` was mutated (assignment, ``.clear()``, ``.append()``).
        A new scan/file load resets the selection, so this also covers most
        load events; explicit load-lifecycle bumps land with the
        controllers in Stage 5.  The first call only records the baseline."""
        sig = (tuple(self.idxs), bool(self.overall))
        if sig != self._last_selection_sig:
            if self._last_selection_sig is not None:
                self._bump_display_generation()
            self._last_selection_sig = sig

    def _active_stitch_mode(self):
        """``'1d'``/``'2d'`` when the persistent stitch display should show, else
        None.  ``stitch_display_mode`` follows the wrangler Mode dropdown, but the
        stitch only renders once the matching result actually exists on the scan —
        so selecting a Stitch mode before a run (or after a new scan cleared the
        result) keeps the per-frame view rather than flashing an empty stitch."""
        m = getattr(self, 'stitch_display_mode', None)
        scan = getattr(self, 'scan', None)
        if m == '1d' and getattr(scan, 'stitched_1d', None) is not None:
            return '1d'
        if m == '2d' and getattr(scan, 'stitched_2d', None) is not None:
            return '2d'
        return None

    def _live_mode(self):
        """Map the widget's viewer state to a :class:`Mode`.  Normal mode is
        INT_1D when the scan is 1D-only (skip_2d) — plot-only, matching
        _apply_1d_only_visibility — else INT_2D (raw|cake / plot).  A live stitch
        result (STITCH_1D/2D) takes precedence over the per-frame integration
        view but not over a file viewer."""
        if self.viewer_mode == 'image':
            return Mode.IMAGE_VIEWER
        if self.viewer_mode == 'xye':
            return Mode.XYE_VIEWER
        if self.viewer_mode == 'nexus':
            return Mode.NEXUS_VIEWER
        stitch = self._active_stitch_mode()
        if stitch == '1d':
            return Mode.STITCH_1D
        if stitch == '2d':
            return Mode.STITCH_2D
        if getattr(self.scan, 'skip_2d', False):
            return Mode.INT_1D
        return Mode.INT_2D

    def _live_display_state(self):
        """Build the :class:`DisplayState` for the current widget inputs by
        dispatching to the mode controller (Stage 5).

        The single place the GUI snapshots its state for the display layer.
        Each controller owns its mode's selection rules — viewer controllers
        never consult scan.frames or the integration unit (§8); the scan
        controller reads the scan frame index for Overall aggregation."""
        mode = self._live_mode()
        return controller_for(mode).compute_state(self, mode)

    def update_plot_range(self):
        if self.ui.slice.isChecked():
            self.update()

    # ── Update orchestration ──────────────────────────────────────

    def _updated(self):
        """Check if there is data to update
        """
        # A live stitch result is a whole-scan synthetic independent of per-frame
        # cache readiness — render it whenever it exists (so it survives eviction
        # of the per-frame data and re-renders on every tick).
        if self._active_stitch_mode() is not None:
            return True

        # In viewer mode, bypass the scan.name check — no HDF5 scan is loaded
        if self.viewer_mode is not None:
            if len(self.frame_ids) == 0:
                return False
            if self.viewer_mode == 'image' and len(self.data_2d) == 0:
                return False
            if self.viewer_mode == 'xye' and len(self.data_1d) == 0:
                return False
            if self.viewer_mode == 'nexus' and len(self.data_1d) == 0:
                return False
            return True

        if (len(self.frame_ids) == 0) or (self.scan.name == 'null_main'):
            return False

        if len(self.idxs_1d) == 0:
            return False

        store = getattr(self, 'publication_store', None)
        if store is not None:
            try:
                if any(store.get(int(idx)) is not None for idx in self.idxs_1d):
                    return True
            except Exception:
                logger.debug("publication-store readiness check failed",
                             exc_info=True)

        if len(self.data_1d) == 0:
            return False

        return True

    def update(self):
        """Re-entrancy-guarded panel update.

        Drops a RE-ENTRANT call (a signal — aggregate-worker completion,
        autorange — firing while a render is already on the stack): the in-flight
        render already reflects the current state, and genuinely-new state lands
        on the next event-loop-driven update.  Defense-in-depth so a pathological
        render chain can never starve the event loop into a hard freeze.
        """
        if getattr(self, "_in_update", False):
            return True
        self._in_update = True
        try:
            return self._update_impl()
        finally:
            self._in_update = False

    def _update_impl(self):
        """Update the image and plot panels for the current selection.

        Mode-agnostic: snapshot one :class:`DisplayState` (via the mode
        controller), build its payload, and hand both to
        :meth:`render_display`, which lays panels out by the state's
        ``layout`` and draws-or-clears each — no ``if viewer_mode == ...``
        dispatch here.
        """
        if hasattr(self, 'refresh_norm_channels'):
            self.refresh_norm_channels()
        self.get_idxs()
        self._note_selection_generation()   # bump generation on selection change

        if not self._updated():
            # Nothing to draw yet for the current selection.  Only render the
            # EXPLICIT blank when there is genuinely nothing cached — a fresh
            # file, a cleared scan, or a failed load with no fallback.  When
            # prior-scan / other-frame data is still cached (a new-scan gap, or
            # a not-yet-loaded selection whose load is in flight), keep the
            # current display instead of flashing blank; the imminent real
            # render replaces it.  Kills the blank flicker at scan start and on
            # frame selection without leaving stale content when there is
            # truly nothing to show.
            if getattr(self, "_display_blanked", False):
                return True
            scan_store_active = (
                getattr(self, "viewer_mode", None) is None
                and getattr(self, "publication_store", None) is not None
            )
            if scan_store_active:
                # Normal scan mode now treats PublicationStore as the
                # authoritative display source.  Old data_1d/data_2d mirrors
                # may still exist as transitional/viewer storage, but they must
                # not keep stale plots/cakes alive when the store says the
                # selected scan rows are missing.
                has_cached = False
            else:
                with self.data_lock:
                    has_cached = bool(self.data_1d) or bool(self.data_2d)
            # Panel-consistency: while a run is active, keep the current display
            # instead of blanking when there's nothing new to draw.  A silent
            # batch run populates the GUI caches only at the end, so without this
            # the empty-render path blanks all panels mid-run (2D goes blank, and
            # clear_plot_view drops the 1D legend) — the inconsistency Vivek saw.
            # Keeping the display freezes ALL panels (1D + 2D) until the run's
            # data lands, so they persist together.
            if has_cached or (getattr(self, 'PERSIST_2D_DURING_PROCESSING', True)
                              and getattr(self, '_processing_active', False)):
                return True
            empty = empty_display_state(self._live_mode(), self.display_generation)
            result = self.render_display(empty, None)
            self._display_blanked = True
            return result

        state = self._live_display_state()
        ctrl = controller_for(state.mode)
        # Share Axis silently re-points the plotUnit combo at the cake's x-axis.
        # Post Step-5 flip the integration 1D draw reads that combo when it builds
        # the payload, so sync it BEFORE build_payload — otherwise a single Share
        # Axis toggle would draw the 1D in the stale unit (render_display also
        # applies it, idempotent, for direct callers).  getattr: bare holders.
        if state.mode in (Mode.INT_1D, Mode.INT_2D):
            _ass = getattr(self, '_apply_share_axis_state', None)
            if _ass is not None:
                _ass()
        payload = ctrl.build_payload(self, state)  # store=None ⇒ delegate draws
        result = self.render_display(state, payload)
        self._display_blanked = False
        return result

    # ── Render (Stage 3) ──────────────────────────────────────────────

    # Per-role draw delegates: render owns the *decision* (what to draw vs
    # clear, gen-drop, blanking); the legacy methods own the pixel push.
    # RAW_2D / PLOT_1D differ by mode (viewer vs normal); CAKE_2D is normal
    # only.  These collapse into mode controllers in Stage 5.
    def _draw_delegate(self, role, mode):
        # Payload-only viewer modes (Image / XYE / NeXus): a ``None`` payload
        # means blank that panel — there is no legacy draw fallback.  Normal
        # integration plots intentionally delegate to update_plot; raw images
        # still keep update_image as their fallback.
        if role is PanelRole.RAW_2D:
            if mode in (Mode.IMAGE_VIEWER, Mode.NEXUS_VIEWER):
                return self.clear_image_view
            # Item-2 flip: the Int raw panel now renders SOLELY from the
            # raw_image payload (gap-masking + detector_shape live there),
            # mirroring CAKE_2D -- a None raw payload normally blanks the panel
            # (no legacy update_image fallback).  Panel-consistency: while a run
            # is active, return None so render skips this panel (keep last) like
            # the 1D plot + cake.  update_image is retained as dead-but-rollback-
            # able code (its gap-mask logic is duplicated in the raw_image builder).
            if (getattr(self, 'PERSIST_2D_DURING_PROCESSING', True)
                    and getattr(self, '_processing_active', False)):
                return None
            return self.clear_image_view
        if role is PanelRole.PLOT_1D:
            if mode in (Mode.XYE_VIEWER, Mode.NEXUS_VIEWER):
                return self.clear_plot_view
            # Stage 4: the production CONTROLS (plotUnit/slice/slice-range,
            # plotMethod, slice converter) now drive the redraw through update()
            # -> the payload pipeline.  The PLOT_1D None-payload fallback is KEPT
            # as update_plot (a safety net + rollback path) pending the live
            # checkpoint that confirms integration_plot_payload covers every case;
            # removing it is the live-gated final step (see update_plot's note).
            return self.update_plot
        if role is PanelRole.CAKE_2D:
            # CAKE_2D renders solely from the payload (cake_image); a None cake
            # payload normally blanks the panel (no legacy update_binned
            # fallback).  Panel-consistency: while a run is active, keep the last
            # cake on screen instead of blanking (None delegate -> render skips
            # this panel) so it persists like the 1D plot.
            if (getattr(self, 'PERSIST_2D_DURING_PROCESSING', True)
                    and getattr(self, '_processing_active', False)):
                return None
            return self.clear_binned_view
        return None

    def _clear_delegate(self, role):
        return {
            PanelRole.RAW_2D: self.clear_image_view,
            PanelRole.CAKE_2D: self.clear_binned_view,
            PanelRole.PLOT_1D: self.clear_plot_view,
        }.get(role)

    def _payload_for_role(self, role, payload):
        if payload is None:
            return None
        if role is PanelRole.PLOT_1D:
            return payload.plot
        if role is PanelRole.RAW_2D:
            return payload.raw_image
        if role is PanelRole.CAKE_2D:
            return payload.cake_image
        return None

    def _draw_payload(self, role, payload_value, state):
        if payload_value is None:
            return False

        if role in (PanelRole.RAW_2D, PanelRole.CAKE_2D):
            if not isinstance(payload_value, ImagePayload):
                return False
            return self._draw_image_payload(role, payload_value, state)

        if role is not PanelRole.PLOT_1D:
            return False

        # A stitch is a single synthetic whole-scan curve with no per-frame
        # selection state — draw it directly rather than through update_plot_view
        # (which keys legends/waterfall offsets off frame ids the stitch lacks).
        if state is not None and state.mode in (Mode.STITCH_1D, Mode.STITCH_2D):
            return self._draw_stitch_plot(payload_value)

        traces = tuple(getattr(payload_value, "traces", ()) or ())
        if not traces:
            self.clear_plot_view()
            return True

        ref_x = np.asarray(traces[0].x, dtype=float)
        rows = []
        names = []
        for trace in traces:
            x = np.asarray(trace.x, dtype=float)
            y = np.asarray(trace.y, dtype=float)
            if x.shape != ref_x.shape or not np.allclose(x, ref_x, equal_nan=True):
                y = np.interp(ref_x, x, y)
            rows.append(y)
            names.append(str(trace.label))

        ydata = np.vstack(rows)
        if self.bkg_1d is not None:
            try:
                ydata = ydata - self.bkg_1d
            except ValueError:
                logger.debug(
                    "Skipping publication plot background with shape %s for %s",
                    np.shape(self.bkg_1d), ydata.shape,
                )

        self.plot_data = [ref_x, ydata]
        self.frame_names = names
        # Flip stage 3: an Overlay/Waterfall payload carries the FULL accumulator
        # (overlaid_ids = every captured frame, which may exceed this render's
        # render_ids after eviction), and the waterfall y-axis (_wf_y_axis) keys off
        # self.overlaid_idxs -- so prefer it.  Single/Sum/Average payloads leave
        # overlaid_ids empty, so those keep the per-render selection.
        overlaid = getattr(payload_value, "overlaid_ids", None)
        self.overlaid_idxs = list(overlaid) if overlaid else list(state.render_ids)
        # Carry the immutable accumulator back onto the widget so the NEXT render
        # appends onto it (the payload-owned successor to the legacy mutable triple).
        # Only Overlay/Waterfall payloads supply a history; others leave it alone.
        history = getattr(payload_value, "plot_history", None)
        if history is not None:
            self._waterfall_history = history
        axis = payload_value.axis_x
        self._payload_x_axis_label = (axis.label, axis.unit)
        # XYE labels its bottom axis from the file prefix; _current_plot_axis_label
        # reads _viewer_x_axis_label first in xye mode, so keep it in sync.
        if state.mode is Mode.XYE_VIEWER:
            self._viewer_x_axis_label = (axis.label, axis.unit)
        axis_y = getattr(payload_value, "axis_y", None)
        self._payload_y_axis_label = (
            (axis_y.label, axis_y.unit) if axis_y is not None else None
        )

        if ref_x.size == 0 or ydata.size == 0 or not np.isfinite(ydata).any():
            self.clear_plot_view()
            return True

        self.plot_data_range = [
            [np.nanmin(ref_x), np.nanmax(ref_x)],
            [np.nanmin(ydata), np.nanmax(ydata)],
        ]
        self._using_publication_plot_payload = True
        try:
            self.update_plot_view()
        finally:
            self._using_publication_plot_payload = False
        # Honor a pending autorange (unit change, mode switch -> data_changed):
        # the legacy 1D draw consumed _plot_autorange_requested in
        # draw_plot_state; the payload path bypasses it, so consume it here.
        if getattr(self, '_plot_autorange_requested', False):
            self._plot_autorange_requested = False
            _ar = getattr(self, '_autorange_plot_view', None)
            if _ar is not None:
                _ar()
        return True

    def _draw_image_payload(self, role, payload, state=None):
        data = np.asarray(payload.image, dtype=float)
        if data.ndim != 2 or data.size == 0 or not np.isfinite(data).any():
            if role is PanelRole.RAW_2D:
                self.clear_image_view()
            else:
                self.clear_binned_view()
            return True

        def _axis_values(axis, size):
            values = getattr(axis, "values", None)
            if values is None:
                return np.arange(size)
            values = np.asarray(values, dtype=float)
            if values.shape != (size,):
                return np.arange(size)
            return values

        # pyqtgraph images expect the first array axis to map to x and the
        # second to y.  HDF5 image-like datasets are conventionally
        # (rows=y, columns=x), so transpose for display.
        image = data.T
        x = _axis_values(payload.axis_x, image.shape[0])
        y = _axis_values(payload.axis_y, image.shape[1])
        # get_rect is affine.  Any nonlinear unit toggle (Q <-> 2θ) must have
        # already resampled the image onto a uniform displayed grid before this
        # point; otherwise interior peaks draw at the wrong coordinate.
        rect = get_rect(x, y)
        widget = self.image_widget if role is PanelRole.RAW_2D else self.binned_widget
        display_data = _downsample_for_display(image, widget)
        widget.setImage(display_data, scale=self.scale, cmap=self.cmap)
        widget.setRect(rect)
        if role is not PanelRole.RAW_2D:
            # Display-only trim: cut the cake's visible axes to the data's
            # bounding box.  The STORED grid keeps the full default range
            # (e.g. GI q-chi integrates chi -180..180 while the physical
            # wedge is ~+/-90) -- without the trim half the view is empty.
            displayFrameWidget._trim_view_to_data(widget, image, x, y)
        displayFrameWidget._set_image_widget_colorbar_visible(widget, True)
        # Levels come from pgImageWidget.update_image's nanpercentile autoscale
        # (the (1,99) Linear default), for the Image Viewer exactly as for the
        # Int 2D raw/cake panels.  The wrangler Intensity Threshold is an
        # integration mask parameter, NOT a colour scale, so it must not set
        # display levels here (coupling it to vmin/vmax washed the image out).
        if role is not PanelRole.RAW_2D:
            widget.image_plot.setLabel(
                "bottom", payload.axis_x.label, units=pretty_unit(payload.axis_x.unit),
            )
            widget.image_plot.setLabel(
                "left", payload.axis_y.label, units=pretty_unit(payload.axis_y.unit),
            )
        if role is PanelRole.RAW_2D:
            # No setLabel-with-units for the raw panel: passing units engages
            # pyqtgraph's auto-SI-prefix machinery, whose stale
            # autoSIPrefixScale multiplied the TICK LABELS (x1000 -> the
            # 0..2.5e6 'Pixels' axes / label flicker on reloaded files)
            # while the actual view range and hover stayed pixel-correct.
            displayFrameWidget._set_raw_pixel_axes(widget)
            self.image_data = (image, rect)
        else:
            self.binned_data = (image, rect)
            # Re-attach the slice-band ROI on the cake.  Legacy update_binned_view
            # did this at its tail (line ~1799); the Step-3 cake-payload flip
            # dropped it and it survived only as a side effect of the legacy
            # update_plot->get_int_1d 1D draw.  Step 5 routes the 1D draw through
            # the pure payload (no show_slice_overlay), so the cake renderer must
            # own the re-attach.  show_slice_overlay self-guards (clears + returns
            # unless slicing a 2D-derived axis).  getattr: render_display unit
            # tests bind _draw_image_payload onto a holder without it.
            _sso = getattr(self, 'show_slice_overlay', None)
            if _sso is not None:
                _sso()
        return True

    def _draw_stitch_plot(self, plot_payload):
        """Draw a stitch's single merged 1-D curve directly onto ``self.plot``.

        A stitch is one synthetic whole-scan trace with no per-frame selection,
        so it bypasses ``update_plot_view`` (which keys legends + waterfall
        offsets off per-frame ids).  Shared by the persistent payload path
        (:meth:`_draw_payload`) and the legacy :meth:`render_stitch_result`.
        Returns True if it drew, False otherwise."""
        traces = tuple(getattr(plot_payload, "traces", ()) or ())
        if not traces:
            self.clear_plot_view()
            return True
        try:
            x = np.asarray(traces[0].x, dtype=float)
            y = np.asarray(traces[0].y, dtype=float)
            self.plot.clear()
            self.plot.plot(x, y, pen=pg.mkPen('#2b6cb0', width=1.5), name='Stitch')
            self.plot.setLabel('bottom', plot_payload.axis_x.label,
                               units=pretty_unit(plot_payload.axis_x.unit))
            self.plot.setLabel('left', 'Intensity')
            self.plot.autoRange()
            return True
        except Exception:
            logger.error("stitch 1D draw failed", exc_info=True)
            return False

    def render_stitch_result(self, result_1d=None, result_2d=None):
        """Legacy one-shot stitch draw (``scan.stitched_1d`` / ``stitched_2d``).

        Superseded by the persistent :class:`StitchDisplayController` (the stitch
        is now a first-class display source routed through ``update()``); kept as a
        direct-draw helper / rollback path.  Returns True if anything drew."""
        drew = False
        pp = stitch_plot_payload(result_1d)
        if pp is not None and self._draw_stitch_plot(pp):
            drew = True
        ip = stitch_image_payload(result_2d)
        if ip is not None:
            try:
                self._draw_image_payload(PanelRole.CAKE_2D, ip)
                drew = True
            except Exception:
                logger.error("stitch 2D draw failed", exc_info=True)
        return drew

    @staticmethod
    def _trim_view_to_data(widget, image, x, y):
        """Set the view range to the non-dummy data extent (display only).

        Skipped when the user has zoomed (auto-range off) so it never fights
        manual navigation; with auto-range on it replaces the full-rect
        autoscale that left e.g. a GI q-chi cake half empty."""
        try:
            vb = widget.image_plot.getViewBox()
            auto = vb.autoRangeEnabled()
            if not (auto[0] or auto[1]):
                # Auto-range is off: either the USER zoomed (respect it) or it
                # is just OUR previous trim -- setRange() itself disables
                # auto-range, so without this check the first trim froze the
                # view forever and an axis-KIND change (e.g. transmission q-chi
                # -> GI qip-qoop) kept the stale window instead of rescaling.
                last = getattr(widget, '_cake_trim_view', None)
                cur = vb.viewRange()

                def _same(a, b):
                    return all(
                        abs(p - q) <= 1e-9 + 1e-6 * max(abs(p), abs(q))
                        for pa, pb in zip(a, b) for p, q in zip(pa, pb)
                    )

                if last is None or not _same(cur, last):
                    return                  # genuine user navigation
            has = np.isfinite(image) & (image > 0)
            if not has.any():
                return
            x_idx = np.where(has.any(axis=1))[0]   # image is (x, y)
            y_idx = np.where(has.any(axis=0))[0]
            pad = 2                                # bins of margin
            x_lo = float(x[max(0, x_idx[0] - pad)])
            x_hi = float(x[min(len(x) - 1, x_idx[-1] + pad)])
            y_lo = float(y[max(0, y_idx[0] - pad)])
            y_hi = float(y[min(len(y) - 1, y_idx[-1] + pad)])
            if x_hi > x_lo and y_hi > y_lo:
                vb.setRange(xRange=(x_lo, x_hi), yRange=(y_lo, y_hi),
                            padding=0.02)
                # Remember what WE set so the next render can tell our trim
                # apart from a user zoom.
                widget._cake_trim_view = [list(r) for r in vb.viewRange()]
        except Exception:
            logger.debug("cake view trim skipped", exc_info=True)

    def _current_image_axis_key(self):
        """Canonical key for the current 2D cake x-axis."""
        scan = getattr(self, 'scan', None)
        if scan is None:
            return None
        if getattr(scan, 'gi', False):
            gi_args = getattr(scan, 'bai_2d_args', {}) or {}
            gi_mode_2d = gi_args.get('gi_mode_2d', 'qip_qoop')
            gi_idx = GI_MODES_2D.index(gi_mode_2d) if gi_mode_2d in GI_MODES_2D else 0
            return _axis_key_from_label(
                f"{gi_x_labels_2D[gi_idx]} ({gi_x_units_2D[gi_idx]})"
            )
        return '2th_deg' if self.ui.imageUnit.currentIndex() == 1 else 'q_A^-1'

    def _plot_axis_key(self, index):
        info = (
            self._plot_axis_info[index]
            if hasattr(self, '_plot_axis_info') and 0 <= index < len(self._plot_axis_info)
            else {}
        )
        key = info.get('unit_key')
        if key:
            return key
        return _axis_key_from_label(_combo_text(self.ui.plotUnit, index))

    def _share_axis_plot_index(self):
        """Return plotUnit index matching the cake x-axis, or None."""
        scan = getattr(self, 'scan', None)
        if scan is None or getattr(scan, 'skip_2d', False):
            return None
        target_key = self._current_image_axis_key()
        if target_key is None:
            return None
        try:
            count = int(self.ui.plotUnit.count())
        except Exception:
            return None
        for idx in range(count):
            info = (
                self._plot_axis_info[idx]
                if hasattr(self, '_plot_axis_info') and idx < len(self._plot_axis_info)
                else {}
            )
            if info and not (
                info.get('axis') == 'radial' or info.get('source') == '1d_2d'
            ):
                continue
            if self._plot_axis_key(idx) == target_key:
                return idx
        return None

    def _set_plot_unit_index_silently(self, index):
        blocker = None
        try:
            blocker = self.ui.plotUnit.blockSignals(True)
        except Exception:
            blocker = None
        try:
            if self.ui.plotUnit.currentIndex() != index:
                self.ui.plotUnit.setCurrentIndex(index)
        finally:
            try:
                self.ui.plotUnit.blockSignals(False if blocker is None else blocker)
            except Exception:
                pass

    def _apply_share_axis_state(self):
        """Synchronize Share Axis by axis identity, never combo row number."""
        target_idx = self._share_axis_plot_index()
        can_share = target_idx is not None
        was_checked = self.ui.shareAxis.isChecked()
        self.ui.shareAxis.setEnabled(can_share)
        if not can_share:
            if was_checked:
                self.ui.shareAxis.setChecked(False)
            displayFrameWidget._set_share_link(self, False)
            self.ui.plotUnit.setEnabled(True)
            return False
        if was_checked:
            self._set_plot_unit_index_silently(target_idx)
            self.ui.plotUnit.setEnabled(False)
            displayFrameWidget._set_share_link(self, True)
            return True
        displayFrameWidget._set_share_link(self, False)
        self.ui.plotUnit.setEnabled(True)
        return False

    def _active_bottom_plot(self):
        """Return the visible bottom plot: waterfall image plot or line plot."""
        try:
            if self._waterfall_active():
                wf = getattr(self, 'wf_widget', None)
                plot = getattr(wf, 'image_plot', None)
                if plot is not None:
                    return plot
        except Exception:
            logger.debug("active bottom plot lookup failed", exc_info=True)
        return getattr(self, 'plot', None)

    def _active_bottom_window(self):
        """Return the GraphicsLayoutWidget holding the visible bottom plot."""
        try:
            if self._waterfall_active():
                wf = getattr(self, 'wf_widget', None)
                win = getattr(wf, 'image_win', None)
                if win is not None:
                    return win
        except Exception:
            logger.debug("active bottom window lookup failed", exc_info=True)
        return getattr(self, 'plot_win', None)

    def _set_share_link(self, on: bool) -> None:
        """Share Axis: FREEZE the 1D y-axis and scale the 1D x-EXTENT so the shared
        Q columns line up under the cake (the cake is never touched).

        Per Vivek's spec the 1D's y-axis must NOT move.  Earlier fixes
        (e29c070/f93cb78) padded the 1D's layout margins to match screen spans,
        which shifted the y-axis to the middle of the pane (the regression Vivek
        flagged); 4055654's bare XLink links the DATA range but not the screen
        geometry cross-widget, so the panes never lined up.  Instead we set the 1D's
        x-RANGE geometrically: the 1D pane is wider and its y-axis sits further left
        than the cake's plot area (which is pushed right by the cake's y-axis +
        colorbar), so the 1D simply shows a WIDER range -- extending to negative Q on
        the left -- and a Q on the 1D sits directly below the same Q on the cake with
        the y-axis frozen in place.  The align runs deferred (the layout must settle
        first), re-runs on resize, and re-runs whenever the cake's x-range changes
        (zoom/pan) so the columns stay aligned -- the cake drives the 1D."""
        ip = getattr(getattr(self, 'binned_widget', None),
                     'image_plot', None)
        get_vb = getattr(ip, 'getViewBox', None)
        vb = get_vb() if callable(get_vb) else None
        if vb is None:
            return          # duck holders in tests / widget not fully built
        already = getattr(self, '_share_link_on', False)
        if on and not already:
            self._share_link_on = True
            # Two-way wiring:
            #   cake x-range change  -> re-align the 1D under the cake (forward),
            #   1D/waterfall x-range -> drive the cake to match (inverse, the
            #     "2D follows 1D" zoom Vivek asked for).
            # The ``_share_axis_syncing`` guard (set around each programmatic
            # setXRange) keeps one direction's range-set from re-triggering the
            # other.  getattr so duck holders (no bound slots) just skip.
            handler = getattr(self, '_on_cake_xrange_changed', None)
            if handler is not None:
                self._share_cake_handler = handler   # stable ref for disconnect
                try:
                    vb.sigXRangeChanged.connect(handler)
                except Exception:
                    logger.debug("share-axis connect failed", exc_info=True)
            inv = getattr(self, '_on_plot_xrange_changed', None)
            if inv is not None:
                self._share_plot_handler = inv
                self._share_plot_vbs = []
                for plot in (
                    getattr(self, 'plot', None),
                    getattr(getattr(self, 'wf_widget', None),
                            'image_plot', None),
                ):
                    pvb = (plot.getViewBox()
                           if hasattr(plot, 'getViewBox') else None)
                    if pvb is not None:
                        try:
                            pvb.sigXRangeChanged.connect(inv)
                            self._share_plot_vbs.append(pvb)
                        except Exception:
                            logger.debug("share-axis inverse connect failed",
                                         exc_info=True)
            displayFrameWidget._schedule_align(self)
        elif not on and already:
            self._share_link_on = False
            handler = getattr(self, '_share_cake_handler', None)
            if handler is not None:
                try:
                    vb.sigXRangeChanged.disconnect(handler)
                except (TypeError, RuntimeError):
                    pass
                self._share_cake_handler = None
            inv = getattr(self, '_share_plot_handler', None)
            if inv is not None:
                for pvb in getattr(self, '_share_plot_vbs', []) or []:
                    try:
                        pvb.sigXRangeChanged.disconnect(inv)
                    except (TypeError, RuntimeError):
                        pass
                self._share_plot_handler = None
                self._share_plot_vbs = []
            # Detach; _on_share_axis_toggled re-arms the 1D's own autorange.
            for plot in (
                getattr(self, 'plot', None),
                getattr(getattr(self, 'wf_widget', None), 'image_plot', None),
            ):
                try:
                    if plot is not None:
                        plot.setXLink(None)
                except Exception:
                    logger.debug("share-axis unlink failed", exc_info=True)

    def _on_cake_xrange_changed(self, *args) -> None:
        """Re-align the 1D under the cake whenever the cake's x-range changes.

        Skipped while ``_share_axis_syncing`` -- that means the inverse
        (2D-follows-1D) align is the one driving the cake, and forwarding it
        back to the 1D would fight the user's 1D zoom."""
        if (getattr(self, '_share_link_on', False)
                and not getattr(self, '_share_axis_syncing', False)):
            displayFrameWidget._schedule_align(self)

    def _on_plot_xrange_changed(self, *args) -> None:
        """User zoomed/panned the 1D (or waterfall) -> drive the cake to match
        (the "2D follows 1D" direction).

        Skipped while ``_share_axis_syncing`` (the forward align is repositioning
        the 1D) so the two directions never chase each other."""
        if (getattr(self, '_share_link_on', False)
                and not getattr(self, '_share_axis_syncing', False)):
            displayFrameWidget._schedule_align_cake(self)

    def _on_share_geometry_changed(self, *args) -> None:
        """Re-align after a layout geometry change (internal splitter drag or a
        cake/bottom-plot resize/show) -- no-op unless Share Axis is on."""
        if getattr(self, '_share_link_on', False):
            displayFrameWidget._schedule_align(self)

    def _schedule_align(self) -> None:
        # DEBOUNCED (freeze fix): each call supersedes the previous pending align
        # and defers ~50 ms.  A burst of schedules (the setup_*_layout hooks +
        # resizeEvent that an align's own setXRange can re-trigger) collapses to a
        # SINGLE align after the burst settles, instead of running one align per
        # event-loop tick — which, with a hundreds-of-frame waterfall repainting
        # each time, froze the GUI.  The convergence guard in _align_plot_under_cake
        # then makes that one align a no-op once the panes are aligned, so it
        # terminates; the 50 ms gap keeps the event loop responsive meanwhile.
        seq = getattr(self, '_align_seq', 0) + 1
        self._align_seq = seq

        def _go():
            if seq != getattr(self, '_align_seq', 0):
                return                       # superseded by a later schedule
            displayFrameWidget._align_plot_under_cake(self)
        Qt.QtCore.QTimer.singleShot(50, _go)

    @staticmethod
    def _global_xspan(win, vb):
        """Global-screen x-extent [left, right] of a viewbox's scene rect.

        The geometric align lines panes up by SCREEN columns (not equal data
        ranges), so the spans must be in a common coordinate system -- global
        screen pixels -- even though each pane lives in its own window.

        ASSUMES an axis-aligned scene transform (translation + uniform scale):
        only the rect's two opposite corners' x() are mapped, so a rotated /
        sheared viewbox would misalign.  Safe today -- no path rotates these
        pyqtgraph viewboxes -- but revisit this mapping if one ever does."""
        r = vb.sceneBoundingRect()
        left = win.mapToGlobal(win.mapFromScene(r.topLeft())).x()
        right = win.mapToGlobal(win.mapFromScene(r.bottomRight())).x()
        return float(left), float(right)

    def _align_plot_under_cake(self) -> None:
        """Set the 1D plot's x-RANGE so the shared Q values land at the same screen
        columns as the cake, WITHOUT moving the 1D's y-axis (no margin changes).

        The cake maps data->screen over its viewbox span [cx0, cx1] showing range
        [cq0, cq1]; we set the 1D's range so its data->screen map coincides over the
        1D's own (wider, further-left) span [px0, px1].  The 1D y-axis is frozen
        (x-auto off, y-auto on); the cake is never touched.  Idempotent + convergent,
        so re-running on every cake range change / resize is safe."""
        try:
            if not getattr(self, '_share_link_on', False):
                return
            bw = getattr(self, 'binned_widget', None)
            cake_win = getattr(bw, 'image_win', None)
            plot_win = self._active_bottom_window()
            bottom_plot = self._active_bottom_plot()
            if (cake_win is None or plot_win is None
                    or bottom_plot is None
                    or not cake_win.isVisible() or not plot_win.isVisible()):
                return
            cvb = bw.image_plot.getViewBox()
            pvb = bottom_plot.getViewBox()

            cx0, cx1 = displayFrameWidget._global_xspan(cake_win, cvb)
            px0, px1 = displayFrameWidget._global_xspan(plot_win, pvb)
            if (cx1 - cx0) <= 1.0 or (px1 - px0) <= 1.0:
                return
            xr = cvb.viewRange()[0]
            cq0, cq1 = float(xr[0]), float(xr[1])
            scale = (cq1 - cq0) / (cx1 - cx0)        # data units per screen pixel
            pq0 = cq0 - (cx0 - px0) * scale          # 1D extends left (often negative Q)
            pq1 = cq0 + (px1 - cx0) * scale          # ...and a touch past the cake on the right
            # CONVERGENCE GUARD (freeze fix): only setXRange when the bottom plot
            # is not already aligned (within ~0.5% of the span).  setXRange
            # triggers a relayout/repaint that re-schedules _align (via the
            # setup_*_layout hooks + resizeEvent); without this guard that is an
            # UNBOUNDED align cascade, and with a hundreds-of-frame waterfall
            # ImageItem repainting each iteration it FREEZES the GUI.  Once the
            # range matches the computed target the next _align is a no-op, so the
            # cascade terminates in 1-2 ticks.
            span = pq1 - pq0
            if span > 0:
                cur0, cur1 = (float(v) for v in pvb.viewRange()[0])
                tol = 0.005 * span
                if abs(cur0 - pq0) <= tol and abs(cur1 - pq1) <= tol:
                    return
            # Mark this range-set as ours so the bottom plot's resulting
            # sigXRangeChanged doesn't bounce back as a "2D follows 1D" inverse.
            self._share_axis_syncing = True
            try:
                bottom_plot.enableAutoRange(x=False, y=True)  # freeze x; keep y auto
                bottom_plot.setXRange(pq0, pq1, padding=0)    # extends left if needed
            finally:
                self._share_axis_syncing = False
        except Exception:
            logger.debug("share-axis geometric align failed", exc_info=True)

    def _schedule_align_cake(self) -> None:
        """Debounced scheduler for the inverse (2D-follows-1D) align -- mirrors
        ``_schedule_align`` with its own sequence counter so the two directions'
        pending timers don't cancel each other."""
        seq = getattr(self, '_align_cake_seq', 0) + 1
        self._align_cake_seq = seq

        def _go():
            if seq != getattr(self, '_align_cake_seq', 0):
                return
            displayFrameWidget._align_cake_under_plot(self)
        Qt.QtCore.QTimer.singleShot(50, _go)

    def _align_cake_under_plot(self) -> None:
        """Inverse of :meth:`_align_plot_under_cake`: set the CAKE's x-range so its
        screen columns line up with the (user-zoomed) 1D -- the "2D follows 1D"
        direction.

        The 1D maps its range [pq0, pq1] over screen span [px0, px1]; we set the
        cake's range over its own span [cx0, cx1] to the same data->screen map.
        Convergence-guarded + ``_share_axis_syncing`` wrapped, exactly like the
        forward align, so it terminates and never fights the forward direction."""
        try:
            if not getattr(self, '_share_link_on', False):
                return
            bw = getattr(self, 'binned_widget', None)
            cake_win = getattr(bw, 'image_win', None)
            plot_win = self._active_bottom_window()
            bottom_plot = self._active_bottom_plot()
            if (cake_win is None or plot_win is None
                    or bottom_plot is None
                    or not cake_win.isVisible() or not plot_win.isVisible()):
                return
            cvb = bw.image_plot.getViewBox()
            pvb = bottom_plot.getViewBox()

            cx0, cx1 = displayFrameWidget._global_xspan(cake_win, cvb)
            px0, px1 = displayFrameWidget._global_xspan(plot_win, pvb)
            if (cx1 - cx0) <= 1.0 or (px1 - px0) <= 1.0:
                return
            xr = pvb.viewRange()[0]
            pq0, pq1 = float(xr[0]), float(xr[1])
            scale = (pq1 - pq0) / (px1 - px0)        # data units per screen pixel
            cq0 = pq0 + (cx0 - px0) * scale
            cq1 = pq0 + (cx1 - px0) * scale
            span = cq1 - cq0
            if span <= 0:
                return
            cur0, cur1 = (float(v) for v in cvb.viewRange()[0])
            tol = 0.005 * span
            if abs(cur0 - cq0) <= tol and abs(cur1 - cq1) <= tol:
                return
            self._share_axis_syncing = True
            try:
                cvb.enableAutoRange(x=False)
                cvb.setXRange(cq0, cq1, padding=0)
            finally:
                self._share_axis_syncing = False
            # Remember the cake was pinned by the inverse sync so unchecking
            # Share Axis can re-arm its autorange (forward-only zooms leave it
            # untouched, preserving the prior "keep cake zoom on uncheck").
            self._cake_x_pinned_by_share = True
        except Exception:
            logger.debug("share-axis inverse align failed", exc_info=True)

    def _install_share_geometry_hooks(self) -> None:
        """Re-align the shared axes on any layout geometry change, not just this
        widget's own resize.

        Dragging an internal splitter resizes the cake/plot panes WITHOUT
        resizing this widget, so ``resizeEvent`` never fires (the reported
        "expand the plot window and the 1D doesn't follow" regression).  Hook the
        splitters' ``splitterMoved`` directly, and install an event filter on the
        cake / bottom-plot windows for Resize/Show -- those fire after the layout
        reflow settles, giving the align correct timing.  All hooks no-op unless
        Share Axis is on."""
        for name in ('splitter', 'splitter_2', 'splitter_3'):
            sp = getattr(self.ui, name, None)
            sig = getattr(sp, 'splitterMoved', None)
            if sig is not None:
                try:
                    sig.connect(self._on_share_geometry_changed)
                except Exception:
                    logger.debug("share-axis splitter hook failed",
                                 exc_info=True)
        for w in (getattr(getattr(self, 'binned_widget', None),
                          'image_win', None),
                  getattr(self, 'plot_win', None),
                  getattr(getattr(self, 'wf_widget', None),
                          'image_win', None)):
            inst = getattr(w, 'installEventFilter', None)
            if callable(inst):
                try:
                    inst(self)
                except Exception:
                    logger.debug("share-axis event-filter install failed",
                                 exc_info=True)

    def eventFilter(self, obj, event):
        """Schedule a share-axis realign when a watched cake/plot window resizes
        or is shown (installed by ``_install_share_geometry_hooks``).  Never
        consumes the event."""
        try:
            et = event.type()
            if et in (Qt.QtCore.QEvent.Type.Resize,
                      Qt.QtCore.QEvent.Type.Show) \
                    and getattr(self, '_share_link_on', False):
                displayFrameWidget._schedule_align(self)
        except Exception:
            logger.debug("share-axis eventFilter failed", exc_info=True)
        return super().eventFilter(obj, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Re-line the shared 1D under the cake after geometry changes.
        if getattr(self, '_share_link_on', False):
            displayFrameWidget._schedule_align(self)

    def render_display(self, state, payload):
        """Draw the display from ``state`` + ``payload``.  (Named
        ``render_display`` to avoid shadowing ``QWidget.render``.)

        Thin: it executes the pure :func:`render_plan` decision — drop a
        stale-generation payload, then draw the panels the state wants and
        clear the rest (so a panel left from a previous mode/selection is
        always blanked, §8).  The pixel push is delegated to the legacy
        draw/clear methods; the *decision* lives in render_plan.
        """
        plan = render_plan(state, payload)
        if plan.drop:
            # Payload computed against a superseded generation — never render
            # it over the current state (§8 generation invariant).
            logger.debug("render: dropping stale payload gen=%s vs state gen=%s",
                         getattr(payload, 'generation', None), state.generation)
            return True

        mode = state.mode

        # Normal-mode input prep: Share-Axis link + 1D-only panel visibility.
        if mode in (Mode.INT_1D, Mode.INT_2D):
            self._apply_share_axis_state()
            self._apply_1d_only_visibility()
        elif mode in (Mode.STITCH_1D, Mode.STITCH_2D):
            # Stitch modes are not in the _apply_1d_only_visibility path; assert
            # their geometry here (idempotent) so the cake/plot pane is visible
            # and so leaving stitch restores the INT geometry on the next render.
            self._apply_layout(mode)

        # Clear the panels this state does not want (kills stale content).
        for role in plan.clear:
            clear = self._clear_delegate(role)
            if clear is not None:
                clear()

        # Draw the panels it does want.  Exception handling matches the
        # legacy update(): normal-mode draws caught only TypeError (a
        # missing-data frame) and let anything else propagate; the viewer
        # draws were wrapped in a broad debug-logged guard.
        is_viewer = mode in (Mode.IMAGE_VIEWER, Mode.XYE_VIEWER, Mode.NEXUS_VIEWER)
        for role in plan.draw:
            payload_value = self._payload_for_role(role, payload)
            if payload_value is not None and self._draw_payload(role, payload_value, state):
                continue
            if role is PanelRole.PLOT_1D:
                self._payload_x_axis_label = None
                self._payload_y_axis_label = None
            draw = self._draw_delegate(role, mode)
            if draw is None:
                continue
            try:
                draw()
            except TypeError:
                return False
            except Exception:
                if not is_viewer:
                    raise
                logger.debug("render: viewer draw of %s failed", role, exc_info=True)

        # (Removed _floor_plot_xaxis: it floored the 1D x-axis at 0 to hide the
        # spurious negative-Q flat line from ZERO-filled GI-pad bins.  Those empty
        # bins are now NaN (gid._nan_empty_1d, 165fb7b) so they aren't plotted and
        # autoRange ignores them; the floor is obsolete AND it clamped the Share
        # Axis geometry alignment, which legitimately extends into negative Q.)

        # 2D title + image-preview popup (normal mode; viewer draw methods
        # set their own title).  Skip on a non-READY (EMPTY/ERROR) state —
        # there is no current frame to label or preview, and ``update_2d_label``
        # would index an empty ``frame_ids`` (IndexError on the explicit-blank
        # render at scan start).
        if (mode in (Mode.INT_1D, Mode.INT_2D)
                and state.load_status is LoadStatus.READY):
            self.update_2d_label()
            self._update_image_preview()
        # The Image / XYE viewer title (filename(s) / ``Filename #N``) used to be
        # a side effect of the legacy ``_update_image_viewer`` / ``_update_xye_viewer``
        # draws; now that those modes render from their payloads, set it here for
        # the selected frame(s).
        elif (mode in (Mode.IMAGE_VIEWER, Mode.XYE_VIEWER)
                and state.load_status is LoadStatus.READY):
            self._set_viewer_title(list(state.render_ids))
        refresh_intensity = getattr(
            self, "_refresh_intensity_controls_after_render", None)
        if refresh_intensity is not None:
            refresh_intensity(mode)
        return True

    def update_views(self):
        """Updates 2D (if flag is selected) and 1D views
        """
        # Refresh the render-style caches BEFORE the no-new-data gate: a Log
        # toggle / colormap change with nothing loaded yet must still apply
        # to the FIRST real render (the gate otherwise left self.scale
        # stale at 'Linear' while the button showed checked).
        self.cmap = self.ui.cmap.currentText()
        self.plotMethod = self.ui.plotMethod.currentText()
        self.scale = self.ui.scale.currentText()

        if not self._updated():
            return True

        if self.viewer_mode is not None:
            # Viewer modes render through the payload path (_draw_image_payload /
            # update_plot_view), which read self.scale / self.cmap directly.  Go
            # through update() so a Linear/Log (or colormap) change redraws the
            # Image/XYE viewer correctly instead of using the Int-mode draws.
            self.update()
            return

        self.update_image_view()
        self.update_binned_view()
        self.update_2d_label()
        self.update_plot_view()

    # ── Viewer / 1D intensity range controls ─────────────────────

    def _intensity_controls_visible_for_mode(self, mode):
        if mode in (Mode.IMAGE_VIEWER, Mode.XYE_VIEWER, Mode.INT_1D):
            return True
        return False

    def _set_intensity_controls_visible(self, visible):
        try:
            self._intensityWidget.setVisible(bool(visible))
        except Exception:
            logger.debug("intensity control visibility update failed", exc_info=True)

    @staticmethod
    def _finite_minmax(data):
        arr = np.asarray(data, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return None
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return None
        return lo, hi

    def _image_display_minmax(self, widget):
        data = getattr(widget, "displayed_image", None)
        return displayFrameWidget._finite_minmax(data)

    def _image_current_levels(self, widget):
        levels = getattr(getattr(widget, "imageItem", None), "levels", None)
        if levels is None:
            return None
        try:
            lo, hi = float(levels[0]), float(levels[1])
        except (TypeError, ValueError, IndexError):
            return None
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            return lo, hi
        return None

    def _plot_display_minmax(self):
        try:
            plot = self._active_bottom_plot()
            if plot is not None:
                y_range = plot.getViewBox().viewRange()[1]
            else:
                y_range = None
        except Exception:
            y_range = None
        data_range = displayFrameWidget._finite_minmax(
            self.plot_data[1] if len(self.plot_data) > 1 else ()
        )
        if data_range is not None:
            return data_range
        if y_range is not None:
            return float(y_range[0]), float(y_range[1])
        return None

    def _plot_current_levels(self):
        try:
            plot = self._active_bottom_plot()
            if plot is None:
                return None
            lo, hi = plot.getViewBox().viewRange()[1]
            lo, hi = float(lo), float(hi)
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                return lo, hi
        except Exception:
            logger.debug("plot y-range lookup failed", exc_info=True)
        return None

    def _intensity_target_for_mode(self, mode):
        if mode is Mode.IMAGE_VIEWER:
            return ("image", self.image_widget)
        if mode in (Mode.XYE_VIEWER, Mode.INT_1D):
            if getattr(self, "_waterfall_active", lambda: False)():
                return ("waterfall", self.wf_widget)
            return ("plot", self._active_bottom_plot())
        return (None, None)

    def _refresh_intensity_controls_after_render(self, mode):
        if not hasattr(self, "_intensityWidget"):
            return
        visible = self._intensity_controls_visible_for_mode(mode)
        self._set_intensity_controls_visible(visible)
        if not visible:
            return

        target, obj = self._intensity_target_for_mode(mode)
        if target in ("image", "waterfall"):
            domain = self._image_display_minmax(obj)
            current = self._image_current_levels(obj)
        elif target == "plot":
            domain = self._plot_display_minmax()
            current = self._plot_current_levels()
        else:
            domain = current = None

        if domain is None:
            self._intensitySlider.setEnabled(False)
            return

        if self._intensityAuto.isChecked():
            values = current if current is not None else domain
            self._intensitySlider.setDomain(
                domain[0], domain[1], lower=values[0], upper=values[1],
                preserve_fraction=False, emit=False,
            )
            return

        self._intensitySlider.setDomain(
            domain[0], domain[1], preserve_fraction=True, emit=False)
        lo, hi = self._intensitySlider.values()
        self._apply_intensity_range(mode, lo, hi)

    def _on_intensity_autoscale_toggled(self, checked):
        mode = self._live_mode()
        if checked:
            try:
                target, obj = self._intensity_target_for_mode(mode)
                if target == "plot":
                    plot = self._active_bottom_plot()
                    if plot is not None:
                        plot.enableAutoRange(axis='y')
                else:
                    self.update()
            except Exception:
                logger.debug("intensity autoscale restore failed", exc_info=True)
            return

        # Seed manual mode from what is currently on screen, then apply it.
        target, obj = self._intensity_target_for_mode(mode)
        if target in ("image", "waterfall"):
            domain = self._image_display_minmax(obj)
            current = self._image_current_levels(obj)
        elif target == "plot":
            domain = self._plot_display_minmax()
            current = self._plot_current_levels()
        else:
            domain = current = None
        if domain is None:
            self._intensitySlider.setEnabled(False)
            return
        values = current if current is not None else domain
        self._intensitySlider.setDomain(
            domain[0], domain[1], lower=values[0], upper=values[1],
            preserve_fraction=False, emit=False,
        )
        self._apply_intensity_range(mode, *self._intensitySlider.values())

    def _on_intensity_range_changed(self, lo, hi):
        if self._intensityAuto.isChecked():
            return
        self._apply_intensity_range(self._live_mode(), lo, hi)

    def _apply_intensity_range(self, mode, lo, hi):
        target, obj = self._intensity_target_for_mode(mode)
        if target in ("image", "waterfall") and obj is not None:
            displayFrameWidget._apply_image_levels(obj, (lo, hi))
        elif target == "plot":
            plot = self._active_bottom_plot()
            if plot is not None:
                try:
                    plot.setYRange(float(lo), float(hi), padding=0)
                except Exception:
                    logger.debug("manual plot y-range failed", exc_info=True)

    # ── 1D-only visibility ────────────────────────────────────────

    def _apply_layout(self, mode):
        """Apply the *full* panel geometry for ``mode`` from ``PANEL_LAYOUT``.

        Idempotent and self-contained: every managed widget's visibility +
        (min,max) height/width is set unconditionally, with no reliance on the
        prior mode's state.  This is the single source of panel geometry —
        ``set_viewer_display_mode`` and ``_apply_1d_only_visibility`` route
        through here instead of poking heights/visibility themselves, which
        kills the "mode A collapsed X and mode B forgot to restore it" leak
        class (the twoDWindow zero-height blank, the stuck 1D-only residue).

        Geometry only: control enable/disable, the plotUnit 2D-entry rebuild,
        the slice uncheck and the splitter equalization stay with their callers
        for now (a later sub-step can fold the controls in too).
        """
        spec = PANEL_LAYOUT[mode]
        ui = self.ui
        move_intensity = getattr(self, "_move_intensity_controls_for_mode", None)
        if move_intensity is not None:
            move_intensity(mode)
        # Visibility.
        ui.frame_top.setVisible(spec.frame_top_vis)
        ui.twoDWindow.setVisible(spec.twoDWindow_vis)
        ui.imageToolbar.setVisible(spec.imageToolbar_vis)
        ui.frame_4.setVisible(spec.frame_4_vis)
        ui.frame_6.setVisible(spec.frame_6_vis)
        ui.plotToolBar.setVisible(spec.plotToolBar_vis)
        self._showImageBtn.setVisible(spec.show_image_btn_vis)
        # Heights (min, max).
        ui.twoDWindow.setMinimumHeight(spec.twoDWindow_h[0])
        ui.twoDWindow.setMaximumHeight(spec.twoDWindow_h[1])
        ui.imageWindow.setMinimumHeight(spec.imageWindow_h[0])
        ui.imageWindow.setMaximumHeight(spec.imageWindow_h[1])
        ui.plotWindow.setMinimumHeight(spec.plotWindow_h[0])
        ui.plotWindow.setMaximumHeight(spec.plotWindow_h[1])
        ui.imageToolbar.setMinimumHeight(spec.imageToolbar_h[0])
        ui.imageToolbar.setMaximumHeight(spec.imageToolbar_h[1])
        ui.plotToolBar.setMinimumHeight(spec.plotToolBar_h[0])
        ui.plotToolBar.setMaximumHeight(spec.plotToolBar_h[1])
        # Cake-panel width (collapsed to show raw only).
        ui.binnedFrame.setMinimumWidth(spec.binnedFrame_w[0])
        ui.binnedFrame.setMaximumWidth(spec.binnedFrame_w[1])

    def _apply_1d_only_visibility(self):
        """Apply the 1D-only vs full-2D *control state* for the current
        processing mode (``scan.skip_2d``), and the matching panel geometry.

        Geometry (panel heights/widths/visibility + the raw-preview button) is
        owned by :meth:`_apply_layout`; this method keeps only the control-state
        bits the geometry table deliberately does not own: the 2D-only controls
        (Share Axis, 2D unit, X-Range slice), the slice uncheck, and rebuilding
        the plotUnit combo for the mode (``set_axes`` is skip-aware, so Int 1D
        omits the 2D-derived axes outright — no post-hoc removal).
        """
        # In viewer mode, set_viewer_display_mode() controls panels
        if self.viewer_mode is not None:
            return
        skip = getattr(self.scan, 'skip_2d', False)
        # Full panel geometry for this processing mode (idempotent).
        self._apply_layout(Mode.INT_1D if skip else Mode.INT_2D)
        if skip:
            # 2D-only controls (Share Axis, 2D unit, X-Range slice) off.
            if self.ui.slice.isChecked():
                self.ui.slice.setChecked(False)
            self._set_2d_controls_visible(False)
        else:
            self._set_2d_controls_visible(True)
        set_intensity_visible = getattr(
            self, "_set_intensity_controls_visible", None)
        if set_intensity_visible is not None:
            set_intensity_visible(skip)
        # Rebuild the plotUnit combo only on a 1D-only<->2D *transition* (so the
        # user's current selection is preserved on every other render).
        # ``set_axes`` is skip-aware: Int 1D drops the 2D-derived axes, Int 2D
        # restores them.
        if skip != self._was_skip_2d:
            self.set_axes()
            # On a 1D-only -> 2D transition the 2D panel was collapsed; the
            # viewer-mode path re-equalizes the primary panels but the
            # update_views path did not, leaving the 2D panel contracted
            # (UI-3).  Re-split the primary panels 50/50, on the transition
            # only (not every render).
            if not skip:
                self._set_equal_primary_panel_heights()
            self._was_skip_2d = skip

    # ── Axis configuration ────────────────────────────────────────

    def set_axes(self):
        """Populate plotUnit / imageUnit combos for standard or GI mode.

        Each plotUnit entry is tracked in ``self._plot_axis_info``, a list
        of dicts with keys:

        - ``'source'``: ``'1d'`` or ``'2d'``
        - ``'slice_axis'``: label of the other 2D axis to slice along
          (only meaningful when source == '2d')
        - ``'axis'``: ``'radial'`` or ``'azimuthal'`` position in the 2D
          result (only for source == '2d')

        In GI mode the plotUnit combo shows the 1D integration axis plus
        both axes from the 2D integration (with slicing enabled only for
        the 2D-derived axes).  In standard mode, the existing behaviour
        is preserved (Q, 2θ, χ) but now annotated with source metadata.
        """
        # Block signals while rebuilding to avoid spurious callbacks
        self.ui.plotUnit.blockSignals(True)
        self.ui.imageUnit.blockSignals(True)

        self.ui.plotUnit.clear()
        self.ui.imageUnit.clear()
        self._plot_axis_info = []
        target_plot_idx = 0

        # Int 1D (skip_2d) has no cake, so a 2D-derived axis (χ, or the GI
        # Q_ip/Q_oop reciprocal axes) can't be sliced from anything — offering it
        # would plot nothing.  Build the combo skip-aware: in Int 1D include only
        # axes computable from the 1D result ('1d' / '1d_2d'), excluding every
        # 'source'=='2d' entry.  Int 2D keeps them all.  This replaces the old
        # add-then-remove dance in _apply_1d_only_visibility (one source of truth).
        skip = getattr(self.scan, 'skip_2d', False)

        if self.scan.gi:
            gi_mode_1d = self.scan.bai_1d_args.get('gi_mode_1d', 'q_total')
            gi_mode_2d = self.scan.bai_2d_args.get('gi_mode_2d', 'qip_qoop')
            idx_1d = GI_MODES_1D.index(gi_mode_1d) if gi_mode_1d in GI_MODES_1D else 0
            idx_2d = GI_MODES_2D.index(gi_mode_2d) if gi_mode_2d in GI_MODES_2D else 0

            label_1d = gi_plotUnits[idx_1d]
            radial_label, azimuthal_label = GI_2D_AXES[gi_mode_2d]

            # --- Q axis (1D integration result) ---
            # If 2D radial matches the 1D label, merge as '1d_2d'
            if radial_label == label_1d:
                self.ui.plotUnit.addItem(_translate("Form", label_1d))
                self._plot_axis_info.append({
                    'source': '1d_2d', 'slice_axis': azimuthal_label,
                    'axis': 'radial',
                })
            else:
                self.ui.plotUnit.addItem(_translate("Form", label_1d))
                self._plot_axis_info.append({
                    'source': '1d', 'slice_axis': None, 'axis': None,
                })

            # --- 2θ conversion option (only when 1D is Q polar/total) ---
            if gi_mode_1d == 'q_total':
                tth_label = f"2{Th} ({Deg})"
                if radial_label == label_1d:
                    self.ui.plotUnit.addItem(_translate("Form", tth_label))
                    self._plot_axis_info.append({
                        'source': '1d_2d', 'slice_axis': azimuthal_label,
                        'axis': 'radial',
                    })
                else:
                    self.ui.plotUnit.addItem(_translate("Form", tth_label))
                    self._plot_axis_info.append({
                        'source': '1d', 'slice_axis': None, 'axis': None,
                    })

            # --- 2D-derived axes (Int 2D only; sliced from the cake) ---
            if not skip and radial_label != label_1d:
                self.ui.plotUnit.addItem(_translate("Form", radial_label))
                self._plot_axis_info.append({
                    'source': '2d', 'slice_axis': azimuthal_label,
                    'axis': 'radial',
                })

            if (not skip and azimuthal_label != label_1d
                    and azimuthal_label != radial_label):
                self.ui.plotUnit.addItem(_translate("Form", azimuthal_label))
                self._plot_axis_info.append({
                    'source': '2d', 'slice_axis': radial_label,
                    'axis': 'azimuthal',
                })

            # imageUnit: single label for the 2D mode
            self.ui.imageUnit.addItem(_translate("Form", gi_imageUnits[idx_2d]))
            self.ui.plotUnit.setEnabled(True)
            self.ui.imageUnit.setEnabled(False)
            unit_1d = str(self.scan.bai_1d_args.get('unit', '')).lower()
            if gi_mode_1d == 'q_total' and '2th' in unit_1d:
                target_plot_idx = 1
        else:
            unit_1d = str(self.scan.bai_1d_args.get('unit', '')).lower()
            if 'chi' in unit_1d:
                # Azimuthal-integration mode (I vs χ via integrate_radial).  The
                # native 1D result IS I(χ), so χ is the default plot axis.  It is
                # source='1d_2d': the bare readout is the pooled I(χ), but the
                # user can also restrict it to a Q band by ticking X Range
                # (slicing the cake along its RADIAL axis).  Q and 2θ are derived
                # from the 2D cake (source='2d', Int 2D only) by slicing along χ.
                # This makes χ-mode symmetric with Q-mode -- Q/2θ projections, the
                # slice range, and Share Axis all behave the same way (the only
                # difference is which axis is the native 1D result).
                self.ui.plotUnit.addItem(_translate("Form", f"{Chi} ({Deg})"))
                self._plot_axis_info.append({
                    # slice_axis=None: resolved dynamically in _set_slice_range to
                    # the cake's radial axis (Q or 2θ per imageUnit toggle).
                    'source': '1d_2d', 'slice_axis': None, 'axis': 'azimuthal',
                })
                # Q / 2θ radial profiles from the cake (Int 2D only; slice along χ)
                if not skip:
                    for label in plotUnits[:2]:
                        self.ui.plotUnit.addItem(_translate("Form", label))
                        self._plot_axis_info.append({
                            'source': '2d',
                            'slice_axis': f'{Chi} ({Deg})',
                            'axis': 'radial',
                        })
                target_plot_idx = 0
            else:
                # Standard mode: Q, 2θ from 1D but can also slice via 2D chi;
                # χ purely from 2D
                for label in plotUnits[:2]:
                    self.ui.plotUnit.addItem(_translate("Form", label))
                    self._plot_axis_info.append({
                        'source': '1d_2d',
                        'slice_axis': f'{Chi} ({Deg})',
                        'axis': 'radial',
                    })
                # χ is derived from 2D (Int 2D only — no cake to slice in Int 1D)
                if not skip:
                    self.ui.plotUnit.addItem(_translate("Form", plotUnits[2]))
                    self._plot_axis_info.append({
                        'source': '2d',
                        'slice_axis': None,  # determined dynamically by imageUnit
                        'axis': 'azimuthal',
                    })
                # Default the plot unit to the entry matching the 1D integration
                # unit (so a 2θ integration opens on a 2θ axis).
                canon_1d = '2th_deg' if '2th' in unit_1d else 'q_A^-1'
                target_plot_idx = default_plot_unit(
                    canon_1d, ('q_A^-1', '2th_deg'))

            for label in imageUnits:
                self.ui.imageUnit.addItem(_translate("Form", label))
            self.ui.plotUnit.setEnabled(True)
            self.ui.imageUnit.setEnabled(True)

        if self.ui.plotUnit.count() > 0:
            self.ui.plotUnit.setCurrentIndex(
                min(target_plot_idx, self.ui.plotUnit.count() - 1)
            )
        self.ui.plotUnit.blockSignals(False)
        self.ui.imageUnit.blockSignals(False)
        # Re-baseline the Overlay/Waterfall unit-change tracker to the freshly
        # rebuilt combo so a PROGRAMMATIC combo rebuild never registers as a
        # spurious user unit change on the next overlay render.  That stale
        # _last_plot_unit was the trigger that sent a live overlay down the
        # REBUILD partial-read path and collapsed/re-stacked the waterfall.
        self._last_plot_unit = self.ui.plotUnit.currentIndex()

        # Update slice enable/disable for current selection
        self._on_plotUnit_changed()

    def _on_plotUnit_changed(self, _=None):
        """Enable/disable slice controls based on whether the selected
        plotUnit axis is derived from 2D integration (slice-able) or 1D
        (not slice-able).  Also updates the slice label to reflect the
        axis being sliced along.
        """
        idx = self.ui.plotUnit.currentIndex()
        if not hasattr(self, '_plot_axis_info') or idx < 0:
            return
        if idx >= len(self._plot_axis_info):
            return

        info = self._plot_axis_info[idx]
        skip_2d = getattr(self.scan, 'skip_2d', False)
        # Slicing requires 2D data and the axis must come from 2D
        can_slice = (not skip_2d) and info['source'] in ('2d', '1d_2d')

        # The X Range button is available when slicing is possible; the
        # center/width spinboxes are only live once X Range is *checked*.
        self.ui.slice.setEnabled(can_slice)
        if not can_slice:
            self.ui.slice.setChecked(False)
            self.clear_slice_overlay()
        self._sync_slice_controls()

        # R2-1: refresh the X-Range label + bounds to the *complementary* 2D
        # axis for this plotUnit (read from _plot_axis_info[idx].slice_axis —
        # Q_ip→Q_oop, Q→χ), driven from here so it tracks plotUnit AND mode/GI
        # changes (set_axes ends by calling this) — not lazily on first click.
        self._set_slice_range()

        # Share Axis is keyed to the cake x-axis unit, not the currently
        # selected plotUnit row.  That lets it switch a χ plot back to Q/2θ/qip
        # when possible, and disables only when no matching 1D axis exists.
        self._apply_share_axis_state()

    def _sync_slice_controls(self, _=None):
        """Enable the slice center/width spinboxes only while the X Range
        button is both available and checked."""
        active = self.ui.slice.isEnabled() and self.ui.slice.isChecked()
        self.ui.slice_center.setEnabled(active)
        self.ui.slice_width.setEnabled(active)

    def _on_share_axis_changed(self, checked):
        """Apply a Share Axis toggle: the full render + relink + the bottom-panel
        rescale, in that order.

        While a scan is PROCESSING the writer thread emits a per-frame display
        signal, so running the synchronous render inline froze the GUI and let the
        events queue into a burst (the 'Share Axis is slow + frames race ahead'
        report).  Defer the work to the next event-loop pass during processing so
        the click returns immediately; when idle (including headless tests, which
        have no running event loop) run it synchronously so behavior is unchanged.
        The render is idempotent, so deferring only changes WHEN it runs."""
        if getattr(self, "_processing_active", False):
            Qt.QtCore.QTimer.singleShot(
                0, lambda c=bool(checked): self._apply_share_axis_change(c))
        else:
            self._apply_share_axis_change(checked)

    def _apply_share_axis_change(self, checked):
        self.update()
        self._on_share_axis_toggled(checked)

    def _on_share_axis_toggled(self, checked):
        """Rescale the active bottom panel to its own data when Share Axis is off.

        While shared, ``_align_plot_under_cake`` pins the bottom plot's x-range
        (x-auto off) geometrically so its Q columns line up under the cake; on
        uncheck ``_set_share_link`` detaches but leaves the view frozen at that
        range, so re-arm ``autoRange`` on the active bottom plot (waterfall or 1D
        line) to refit its own data.  Processing-mode only — Share Axis is hidden
        in viewer modes, and the only programmatic uncheck is INT-gated."""
        if not checked and getattr(self, "viewer_mode", None) is None:
            # Rescale the ACTIVE bottom panel back to its own data.  Earlier this
            # only refit self.plot (the 1D line); when the bottom panel is the
            # waterfall image, _align had frozen its x at the shared cake range
            # (x-auto off), so un-sharing left it stuck (Vivek-reported).  Refit
            # the 1D line AND the waterfall (whichever is the visible bottom plot)
            # so both come back to their own extent.  Order matters: autoRange()
            # internally disables auto, so re-arm with enableAutoRange() after.
            plots = [self.plot]
            ab = self._active_bottom_plot()
            if ab is not None and ab is not self.plot:
                plots.append(ab)
            for _p in plots:
                try:
                    _p.autoRange()          # immediate fit (disables auto)
                    _p.enableAutoRange()    # re-arm continuous tracking
                except Exception:
                    logger.debug("autoscale on Share Axis off failed",
                                 exc_info=True)
            # If the inverse ("2D follows 1D") sync had pinned the cake's
            # x-range, re-arm its autorange too so it refits its own data on
            # uncheck.  Forward-only zooms never pin it, so this preserves the
            # prior behavior of leaving a cake-zoom intact after unsharing.
            if getattr(self, '_cake_x_pinned_by_share', False):
                try:
                    cvb = self.binned_widget.image_plot.getViewBox()
                    cvb.autoRange()
                    cvb.enableAutoRange()
                except Exception:
                    logger.debug("cake autoscale on Share Axis off failed",
                                 exc_info=True)
                self._cake_x_pinned_by_share = False

        # Update slice range label
        self._set_slice_range()

    def _autorange_plot_view(self, *args):
        """Refit the 1D plot view to the current data and KEEP auto-ranging on.

        A 1D-unit change re-expresses the x-values (Q<->2θ span very different
        ranges), so the view must auto-range to the new data instead of staying
        frozen at the previous unit's range.

        Do a one-shot ``autoRange()`` for an IMMEDIATE refit (synchronous — the
        headless tests and the user both need the view to reflect the new data
        right away), then re-arm continuous auto with ``enableAutoRange()``.
        Order matters: ``autoRange()`` internally calls
        ``setRange(disableAutoRange=True)`` which turns continuous tracking OFF,
        so the trailing ``enableAutoRange()`` is what keeps the y-axis following
        new live traces (instead of freezing and clipping a taller peak) until
        the user manually zooms."""
        try:
            if self.viewer_mode is None and self.ui.shareAxis.isChecked():
                # Shared (processing modes only): the cake owns x.  Only
                # refit y, never grab x back (the bidirectional link would
                # drag the cake to the 1D's wider data range).  Viewer modes
                # must keep full autorange -- a checked-but-hidden Share
                # Axis froze the XYE viewer at the old cake's x-range.
                self.plot.enableAutoRange(x=False, y=True)
                return
            self.plot.autoRange()        # immediate fit (disables auto)
            self.plot.enableAutoRange()  # re-arm continuous tracking
        except Exception:
            logger.debug("1D autoscale on unit change failed", exc_info=True)

    # ── 2D image rendering ────────────────────────────────────────

    def _nan_thumbnail_gaps(self, data, frame_mask=None):
        """NaN the detector gap pixels in a downsampled thumbnail in place.

        The detector mask (``scan.global_mask`` + any per-frame ``frame_mask``)
        is stored as flat indices into the *full-resolution* detector shape.
        The full-res path applies it directly, but the thumbnail path normally
        relies on the mask being baked into the preview at creation.  A frame
        whose thumbnail was generated without the bake — notably the last frame
        persisted at end-of-scan — then shows the 0-valued module gaps as dark
        instead of NaN.  Map the flat indices into the thumbnail's smaller shape
        via the cached full-res shape (``_raw_full_shape``, set by
        ``get_frames_map_raw`` whenever a resident raw is seen) and set those
        pixels to NaN, so the raw panel masks gaps consistently with the
        full-res path.  No-op when the shape isn't known or there is no gap mask.

        Tactical bridge: the publication/payload unification folds this
        thumbnail-vs-full-res masking divergence into one display contract.
        The flat-index union + downsample-coordinate mapping live in the
        Qt-free ``display_logic`` (``combine_flat_masks`` / ``nan_gaps_in_thumbnail``)
        so the legacy path here and the publication ``raw_image`` builder mask
        gaps identically.
        """
        _scan = getattr(self, 'scan', None)
        # Authoritative full-res shape from the scan (persisted in the .nxs);
        # falls back to the live widget cache, then None (no-op).  Explicit
        # is-None checks (not truthiness) so a stray ndarray can't raise.
        full_shape = getattr(_scan, 'detector_shape', None)
        if full_shape is None:
            full_shape = getattr(self, '_raw_full_shape', None)
        gap = combine_flat_masks(getattr(_scan, 'global_mask', None), frame_mask)
        nan_gaps_in_thumbnail(data, gap, full_shape)

    def update_image(self):
        """Updates image plotted in image frame.

        Applies the detector-level mask and global mask to the raw image.
        If the data is a downsampled thumbnail (mask already baked in as
        NaN), the mask application is skipped because the flat indices
        would not match the thumbnail's smaller shape.
        """
        self._image_levels_override = None
        # Panel-consistency: while a run is active, keep the last raw image on
        # screen rather than blanking when the in-flight frame's data isn't
        # available yet — so the raw panel persists like the 1D plot.
        keep_last = (getattr(self, 'PERSIST_2D_DURING_PROCESSING', True)
                     and getattr(self, '_processing_active', False))
        mask = None
        if self.overall and len(self.frame_ids) > 1:
            # G2: aggregate via per-frame dict instead of the deleted
            # scan.overall_raw accumulator.  Stays correct after v2
            # reload (the accumulator didn't), and after replace-frames
            # reintegration (the accumulator drifted).
            data, raw_source = self.get_frames_map_raw(
                list(self.scan.frames.index),
                prefer_thumbnail=True,
                return_source=True,
                require_all=True,
            )
            if data is None:
                if not keep_last:
                    self.clear_image_view()
                return
        else:
            data, raw_source = self.get_frames_map_raw(return_source=True)
            if data is None:
                if not keep_last:
                    self.clear_image_view()
                return

            # Apply Mask — store first (A3/A4).  The bounded mirror remains a
            # transition fallback, but normal scan display should not need it to
            # know the current frame's detector mask.
            frame_2d = None
            try:
                pub = self._publication_from_store_for_display(
                    self.idxs_2d[0], allow_blocking_read=False)
                if pub is not None:
                    _frame_1d, frame_2d = self._publication_legacy_parts(pub)
            except Exception:
                logger.debug("store mask lookup failed", exc_info=True)
            if not frame_2d:
                with self.data_lock:
                    frame_2d = self.data_2d.get(self.idxs_2d[0])
            mask = frame_2d.get('mask') if frame_2d is not None else None
        # Capture the saturation ceiling from the RAW integer dtype (iinfo.max)
        # BEFORE the float conversion loses it — the display then learns the
        # ceiling from the detector bit depth rather than assuming 16-bit.
        _sat_ceiling = integer_saturation_ceiling(data)
        data = np.asarray(data, dtype=float)

        # Mask detector sentinels (saturation ceiling / uint32 ceiling) to NaN,
        # for PARITY with the payload path (display_publication.raw_image, which
        # already calls sentinel_mask).  The legacy update_image path — which the
        # live Int-raw panel uses (the cake reads the store; the raw panel does
        # not) — previously skipped this, so a TIFF whose invalid pixels sit at
        # the ceiling rendered them as bright bands/speckle instead of masked.
        # Gated by the "Mask Saturated" toggle; no-op when no sentinels are
        # present (the <1e-4 fraction guard).
        data = sentinel_mask(
            data,
            mask_saturation=bool(getattr(self.scan, 'mask_sentinel', True)),
            ceiling=_sat_ceiling,
        )

        # Apply the detector + global mask.  Full-resolution raw uses the flat
        # detector indices directly.  Thumbnails normally bake the mask in at
        # creation, but a frame whose thumbnail was generated without it (e.g.
        # the last frame persisted at end-of-scan) shows the 0-valued module
        # gaps as dark; re-apply the gap mask in thumbnail coordinates so both
        # paths mask gaps identically.
        if raw_source == 'raw':
            # Bound each flat index to data.size (combine_flat_masks) rather than
            # the old all-or-nothing max()<size guard, so an out-of-range index
            # can't suppress masking the in-range gaps -- identical to the
            # payload path's masking.
            _gap = combine_flat_masks(
                mask, getattr(getattr(self, 'scan', None), 'global_mask', None),
                size=data.size)
            if _gap is not None:
                data[np.unravel_index(_gap, data.shape)] = np.nan
        elif raw_source is not None:
            self._nan_thumbnail_gaps(data, mask)

        # Subtract background
        bkg = np.asarray(self.bkg_map_raw)
        if bkg.shape == () or bkg.shape == data.shape:
            data -= self.bkg_map_raw
        else:
            logger.debug(
                "Skipping raw-image background with shape %s for display shape %s",
                bkg.shape, data.shape,
            )

        if data.size == 0 or not np.isfinite(data).any():
            if not keep_last:
                self.clear_image_view()
            return

        data = data.T[:, ::-1]

        # Get Bounding Rectangle
        rect = get_rect(np.arange(data.shape[0]), np.arange(data.shape[1]))

        self.image_data = (data, rect)
        self.update_image_view()

    def update_image_view(self):
        if self.image_data is None:
            # Int 1D mode (or nothing drawn yet): a scale/cmap switch
            # re-renders all views, but there is no raw panel content.
            return
        data, rect = self.image_data

        display_data = _downsample_for_display(data, self.image_widget)
        self.image_widget.setImage(display_data, scale=self.scale, cmap=self.cmap)
        self.image_widget.setRect(rect)
        displayFrameWidget._set_image_widget_colorbar_visible(
            self.image_widget, True)
        displayFrameWidget._apply_image_levels(
            self.image_widget,
            getattr(self, "_image_levels_override", None),
        )
        self._image_levels_override = None

        displayFrameWidget._set_raw_pixel_axes(self.image_widget)

    def update_binned_view(self):
        if self.binned_data is None:
            return                      # no cake drawn yet (e.g. Int 1D mode)
        data, rect = self.binned_data

        display_data = _downsample_for_display(data, self.binned_widget)
        self.binned_widget.setImage(display_data, scale=self.scale, cmap=self.cmap)
        self.binned_widget.setRect(rect)
        displayFrameWidget._set_image_widget_colorbar_visible(
            self.binned_widget, True)

        imageUnit = self.ui.imageUnit.currentIndex()
        if self.scan.gi:
            gi_mode_2d = self.scan.bai_2d_args.get('gi_mode_2d', 'qip_qoop')
            gi_idx = GI_MODES_2D.index(gi_mode_2d) if gi_mode_2d in GI_MODES_2D else 0
            _xl2 = gi_x_labels_2D[gi_idx]
            _xu2 = gi_x_units_2D[gi_idx]
            _yl2 = gi_y_labels_2D[gi_idx]
            _yu2 = gi_y_units_2D[gi_idx]
        else:
            _xl2 = x_labels_2D[imageUnit] if imageUnit < len(x_labels_2D) else x_labels_2D[0]
            _xu2 = x_units_2D[imageUnit] if imageUnit < len(x_units_2D) else x_units_2D[0]
            _yl2 = y_labels_2D[imageUnit] if imageUnit < len(y_labels_2D) else y_labels_2D[0]
            _yu2 = y_units_2D[imageUnit] if imageUnit < len(y_units_2D) else y_units_2D[0]
        self.binned_widget.image_plot.setLabel("bottom", _xl2, units=pretty_unit(_xu2))
        self.binned_widget.image_plot.setLabel("left", _yl2, units=pretty_unit(_yu2))

        self.show_slice_overlay()
        return data

    @staticmethod
    def _set_raw_pixel_axes(widget):
        """Use literal detector pixels; do not let pyqtgraph scale to kPixels."""
        plot = getattr(widget, 'image_plot', None)
        if plot is None:
            return
        try:
            # Avoid ``units='Pixels'`` here: AxisItem treats units as SI-scaleable
            # and can relabel large detector axes as kPixels. The Image Viewer
            # should show detector pixels literally.
            plot.setLabel("bottom", 'x (Pixels)')
            plot.setLabel("left", 'y (Pixels)')
        except Exception:
            logger.debug("raw pixel axis label update failed", exc_info=True)
            return
        for axis_name in ('bottom', 'left'):
            try:
                axis = plot.getAxis(axis_name)
                if hasattr(axis, 'enableAutoSIPrefix'):
                    axis.enableAutoSIPrefix(False)
                if hasattr(axis, 'setScale'):
                    axis.setScale(1.0)
                # pg 0.14: tick strings multiply by autoSIPrefixScale, and
                # updateAutoSIPrefix recomputes it from whatever the CURRENT
                # range is.  Verified live (Jun 2026): at label time the
                # axis still held the transient pre-autorange range
                # (+/-0.615), siScale(0.615) -> milli prefix -> pixel ticks
                # rendered x1000 (the 0..2.5e6 'Pixels' axes).  Clear the
                # stale scale explicitly and force an axis repaint.
                if hasattr(axis, 'autoSIPrefixScale'):
                    axis.autoSIPrefixScale = 1.0
                if hasattr(axis, 'labelUnitPrefix'):
                    axis.labelUnitPrefix = ''
                axis.picture = None
                axis.update()
            except Exception:
                logger.debug("raw pixel axis scale update failed", exc_info=True)

    def update_2d_label(self):
        """Updates 2D Label
        """
        # Sets title text
        label = self.scan.name
        if len(label) > 40:
            label = f'{label[:18]}...{label[-18:]}'

        # Single/Overlay/Waterfall: the cake + raw show the CURRENT (latest-
        # selected) frame (see _display_ids_for_2d), so the title shows that
        # frame's number -- not the bare name / "[Average]".  Only Sum/Average
        # show the aggregate (handled by the chain below).  single_img scans have
        # no per-frame index, so they keep the bare-name title.
        method = getattr(self, 'plotMethod', '') or ''
        if method not in ('Sum', 'Average') and not self.scan.single_img:
            idxs = getattr(self, 'idxs_2d', None) or getattr(self, 'idxs_1d', None) or []
            if idxs:
                self.ui.labelCurrent.setText(f'{label}_{idxs[-1]}')
                return

        if (self.overall or self.scan.single_img) and (len(self.frame_ids) > 1):
            self.ui.labelCurrent.setText(label)
        elif self.scan.series_average:
            self.ui.labelCurrent.setText(label)
        elif len(self.frame_ids) > 1:
            self.ui.labelCurrent.setText(f'{label} [Average]')
        elif self.frame_ids:
            self.ui.labelCurrent.setText(f'{label}_{self.frame_ids[0]}')
        else:
            # No selection yet (e.g. a new scan before its first frame) — show
            # the scan name alone rather than indexing an empty frame_ids list.
            self.ui.labelCurrent.setText(label)

    # ── Normalization / background handlers ───────────────────────

    def normUpdate(self):
        """Update plots if norm channel exists"""
        self.normChannel = self.get_normChannel()
        if self.normChannel:
            # scan_data may now carry non-numeric columns (N2): treat a
            # non-numeric / zero norm channel as "no normalization".
            try:
                norm_sum = float(self.scan.scan_data[self.normChannel].sum())
            except (TypeError, ValueError):
                norm_sum = 0.0
            if norm_sum == 0.:
                self.normChannel = None
        # Clear stale plot_data so update_plot() rebuilds all overlay curves
        self.plot_data = [np.zeros(0), np.zeros(0)]
        self.frame_names = []
        self.overlaid_idxs = []
        self._waterfall_history = None
        self.update()

    def _clear_bkg(self):
        """Drop every background (all modes) and reset the button."""
        self.bkg_1d = 0.
        self.bkg_2d = 0.
        self.bkg_map_raw = 0.
        self._bkg_xye = None
        self.ui.setBkg.setText('Set BG')

    def _viewer_selection(self):
        """Selected viewer frame labels (ints).  Mirrors get_idxs' viewer rule:
        the loaded frame_ids ARE the selection."""
        ids = self.idxs or self.frame_ids
        out = []
        for i in ids:
            try:
                out.append(int(i))
            except (TypeError, ValueError):
                continue
        return out

    def _viewer_raw_for(self, label):
        """The selected viewer frame's raw array (full or thumbnail), sourced like
        _image_viewer_raw_payload: publication store first, then the data_2d
        mirror."""
        store = getattr(self, 'publication_store', None)
        if store is not None:
            pub = store.get(label)
            if pub is not None:
                raw = pub.view.raw
                if raw is None:
                    raw = (getattr(pub.raw_ref, 'thumbnail', None)
                           if pub.raw_ref is not None else None)
                if raw is None:
                    raw = pub.view.thumbnail
                if raw is not None:
                    return raw
        with self.data_lock:
            frame_2d = self.data_2d.get(label)
        if frame_2d is not None:
            raw = frame_2d.get('map_raw')
            if raw is None:
                raw = frame_2d.get('thumbnail')
            return raw
        return None

    def _set_bkg_image_viewer(self):
        """Image Viewer Set BG: average the selected raw frame(s) into
        ``bkg_map_raw`` (resized to the displayed thumbnail at render)."""
        accum = None
        count = 0
        for label in self._viewer_selection():
            raw = self._viewer_raw_for(label)
            if raw is None:
                continue
            raw = np.asarray(raw, dtype=float)
            if raw.ndim != 2:
                continue
            if accum is None:
                accum = raw
            elif accum.shape == raw.shape:
                accum = accum + raw
            else:
                continue
            count += 1
        if accum is None or count == 0:
            return False
        self.bkg_map_raw = accum / count
        return True

    def _set_bkg_xye_viewer(self):
        """XYE Viewer Set BG: average the selected XYE 1D(s) onto the first
        selection's grid and store ``(x, y)``; subtracted (interpolated) per trace
        in _xye_plot_payload."""
        ref_x = None
        accum = None
        count = 0
        for label in self._viewer_selection():
            xy = self._viewer_xye_for(label)
            if xy is None:
                continue
            x, y = xy
            if x.shape != y.shape or x.size == 0:
                continue
            if ref_x is None:
                ref_x = x
            elif x.shape != ref_x.shape or not np.allclose(x, ref_x, equal_nan=True):
                y = np.interp(ref_x, x, y)
            accum = y if accum is None else accum + y
            count += 1
        if accum is None or count == 0:
            return False
        self._bkg_xye = (ref_x, accum / count)
        return True

    def _viewer_xye_for(self, label):
        """The selected XYE frame's (x, y), sourced like _xye_plot_payload:
        publication store first, then the data_1d mirror."""
        store = getattr(self, 'publication_store', None)
        if store is not None:
            pub = store.get(label)
            if (pub is not None and pub.view.intensity_1d is not None
                    and pub.view.axis_1d is not None
                    and pub.view.axis_1d.values is not None):
                return (np.asarray(pub.view.axis_1d.values, dtype=float),
                        np.asarray(pub.view.intensity_1d, dtype=float))
        with self.data_lock:
            fr = self.data_1d.get(label)
        int_1d = getattr(fr, 'int_1d', None) if fr is not None else None
        if int_1d is None:
            return None
        return (np.asarray(int_1d.radial, dtype=float),
                np.asarray(int_1d.intensity, dtype=float))

    def setBkg(self):
        """Sets selected points as background.
        If background is already selected, it unsets it.

        Mode-aware: Int 1D/2D averages the selected scan frames' 1D/2D/raw; the
        Image Viewer averages the selected raw frame(s) into bkg_map_raw; the XYE
        Viewer averages the selected XYE 1D(s).  The background is cleared on a
        mode change (set_viewer_display_mode), so each mode is independent."""
        if (len(self.frame_ids) == 0) or (len(self.idxs) == 0):
            return

        if self.ui.setBkg.text() != 'Set BG':
            self._clear_bkg()
            self.update()
            return

        # Viewer modes own their own background sourcing (no scan-frame integration).
        if self.viewer_mode == 'image':
            if self._set_bkg_image_viewer():
                self.ui.setBkg.setText('Clear BG')
            self.update()
            return
        if self.viewer_mode == 'xye':
            if self._set_bkg_xye_viewer():
                self.ui.setBkg.setText('Clear BG')
            self.update()
            return
        if self.viewer_mode == 'nexus':
            return  # no display to subtract a background from

        # Int 1D/2D (viewer_mode is None here -- viewers early-returned above).
        idxs = self.frame_ids
        if self.overall:
            idxs = sorted(list(self.scan.frames.index))

        # #6: refuse a PARTIAL 2D background rather than silently averaging
        # only the frames whose int_2d happens to be available — a partial
        # average is a wrong background, not a smaller one.  require_all=True
        # returns None when not every selected frame contributes; a None
        # here is only partial coverage (not a 1D-only scan) if the
        # subset-average path WOULD have returned something.
        bkg_2d, _, _ = self.get_frames_int_2d(
            idxs, require_all=True, allow_blocking_read=True)
        if (
            bkg_2d is None
            and self.get_frames_int_2d(
                idxs, allow_blocking_read=True)[0] is not None
        ):
            logger.error(
                "Set BG refused: the 2D background covers only part of the "
                "selection (some frames' 2D data is unavailable) — a partial "
                "average would be a wrong background.")
            try:
                QtWidgets.QMessageBox.warning(
                    self, "Background not set",
                    "Some selected frames have no 2D data, so the background "
                    "would cover only part of the selection.  Background not "
                    "set — pick a fully-covered selection.")
            except Exception:
                pass
            return  # button stays 'Set BG'; no wrong background applied

        self.bkg_1d, _ = self.get_frames_int_1d(
            idxs, rv='average', allow_blocking_read=True)
        self.bkg_2d = bkg_2d
        # Set-Bkg is a one-shot user action on an idle scan: block-and-read
        # an evicted frame's raw from disk rather than defer to the async
        # worker (which would leave it out -> require_all None -> a silent
        # bkg_map_raw = 0 over the whole 2D map).
        self.bkg_map_raw = self.get_frames_map_raw(
            idxs, require_all=True, allow_blocking_read=True)
        if self.bkg_map_raw is None:
            # F5: be honest about a no-op 2D background.  Pre-F5
            # this silently set bkg=0.: 1D/2D bkg subtraction
            # would still apply but the user saw "Clear BG" on
            # the button as if 2D was wired up too.  Without
            # raw frames (e.g. reloaded v2 file without
            # resolvable source files), there's nothing to
            # subtract in the 2D map view; log it.
            logger.warning(
                "setBkg: no raw image data available for selected "
                "frames; 2D background subtraction inactive "
                "(1D / int_2d background still applied).  This "
                "usually means the .nxs was reloaded without "
                "access to the original source files."
            )
            self.bkg_map_raw = 0.
        self.ui.setBkg.setText('Clear BG')
        self.update()
        return

    # ── Viewer modes ──────────────────────────────────────────────

    def clear_overlay(self):
        """Drop accumulated overlay curves + names."""
        self.plot_data = [np.zeros(0), np.zeros(0)]
        self.plot_data_range = [[0, 0], [0, 0]]
        self.frame_names = []
        self.overlaid_idxs = []
        self._waterfall_history = None
        # Forget which scan the (now empty) accumulator belonged to, so the next
        # render re-stamps it for the current scan.
        self._overlay_scan_key = None

    # ── Panel clears (safety net for empty selections) ────────────
    # When a render path has no data for the current selection it must
    # blank its panel instead of returning early and leaving the last
    # frame on screen — otherwise a mode switch or an unhydrated frame
    # shows a stale image/cake/curve that looks like real data.

    @staticmethod
    def _clear_image_widget(widget):
        """Clear a pyqtgraph image widget without drawing fake zero data."""
        try:
            widget.raw_image = np.zeros(0)
            widget.displayed_image = np.zeros(0)
        except Exception:
            pass
        item = getattr(widget, "imageItem", None)
        try:
            if item is not None and hasattr(item, "clear"):
                item.clear()
            elif hasattr(widget, "clear"):
                widget.clear()
        except Exception:
            logger.debug("image widget clear failed", exc_info=True)
        displayFrameWidget._set_image_widget_colorbar_visible(widget, False)

    @staticmethod
    def _set_image_widget_colorbar_visible(widget, visible):
        """Hide colorbars when an image panel is intentionally blank."""
        hist = getattr(widget, "histogram", None)
        if hist is None:
            return
        try:
            hist.setVisible(bool(visible))
        except Exception:
            logger.debug("image colorbar visibility update failed", exc_info=True)

    @staticmethod
    def _apply_image_levels(widget, levels):
        """Apply display-only color levels without changing image pixels."""
        if levels is None:
            return
        try:
            lo, hi = float(levels[0]), float(levels[1])
        except (TypeError, ValueError, IndexError):
            return
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return
        try:
            widget.imageItem.setLevels((lo, hi))
        except Exception:
            logger.debug("image item level update failed", exc_info=True)
        hist = getattr(widget, "histogram", None)
        if hist is not None and hasattr(hist, "setLevels"):
            try:
                hist.setLevels(values=(lo, hi))
            except Exception:
                logger.debug("image colorbar level update failed", exc_info=True)

    def clear_image_view(self):
        """Blank the raw 2D image panel."""
        try:
            self.image_data = None
            self._clear_image_widget(self.image_widget)
        except Exception:
            logger.debug("clear_image_view failed", exc_info=True)

    def clear_binned_view(self):
        """Blank the 2D cake panel."""
        try:
            self.binned_data = None
            self._clear_image_widget(self.binned_widget)
            # A blanked cake must not keep a stale slice-band ROI floating over
            # the empty panel.  Pre Step-5 the legacy get_int_1d re-attached the
            # band every sliced 1D draw; now the cake renderer owns it
            # (_draw_image_payload), so a cleared cake must drop it.
            # clear_slice_overlay self-guards on overlay=None.
            _cso = getattr(self, 'clear_slice_overlay', None)
            if _cso is not None:
                _cso()
        except Exception:
            logger.debug("clear_binned_view failed", exc_info=True)

    def clear_plot_view(self):
        """Remove all 1D curves and reset cached plot state."""
        try:
            self.clear_overlay()
            # removeItem, not curve.clear() -- see update_plot_view: clear()
            # leaves the items registered on the plot and they accumulate.
            for curve in self.curves:
                self.plot.removeItem(curve)
            self.curves.clear()
            if getattr(self, 'legend', None) is not None:
                self.legend.clear()
            if getattr(self, 'wf_widget', None) is not None:
                self._clear_image_widget(self.wf_widget)
            self._payload_x_axis_label = None
            self._payload_y_axis_label = None
        except Exception:
            logger.debug("clear_plot_view failed", exc_info=True)

    def clear_display_state(self, title=None):
        """Blank all rendered panels and cached display data."""
        self._raw_resolve_failed = set()   # re-arm the raw hydrate retries
        self._raw_full_shape = None        # detector shape is per-scan (sizes vary)
        self.clear_image_view()
        self.clear_binned_view()
        self.clear_plot_view()
        if title is not None:
            self.ui.labelCurrent.setText(title)

    def set_processing_active(self, active):
        """Mark a wrangler/integrator run as in progress (or finished).

        Called by ``staticWidget`` at run start/end (incl. Stop).  While active,
        the PERSIST_2D_DURING_PROCESSING feature keeps the last-rendered 2D
        panels on screen instead of blanking them when the in-flight frame's 2D
        data isn't available yet — so the 2D panels persist like the 1D plot.
        """
        self._processing_active = bool(active)
        # Reset the waterfall-repaint throttle at every run boundary so the next
        # update_wf always repaints in full -- in particular the end-of-scan flush
        # (run just ended -> active False) must show the COMPLETE stack even if the
        # last in-scan repaint was throttled.
        self._wf_last_draw_t = 0.0

    def _show_viewer_set_bkg(self, show):
        """Surface ONLY the Set BG button in a viewer mode.

        The Set BG button lives in ``frame_4`` next to Norm Channel, which the
        layout table hides in viewer modes.  Show ``frame_4`` but keep Norm Channel
        hidden, so Set BG sits left-justified where Norm Channel is in the Int
        modes.  ``show=False`` hides it (NeXus has no display to subtract from)."""
        try:
            self.ui.normChannel.setVisible(False)
            self.ui.frame_4.setVisible(bool(show))
            self.ui.setBkg.setVisible(bool(show))
            self.ui.setBkg.setEnabled(bool(show))
            if show:
                for _lay in (self.ui.frame_4.layout(),
                             self.ui.setBkg.parentWidget().layout()):
                    if _lay is not None:
                        try:
                            _lay.setContentsMargins(0, 0, 0, 0)
                            _lay.setAlignment(
                                self.ui.setBkg, pyQt.AlignLeft | pyQt.AlignVCenter)
                        except Exception:
                            pass
        except Exception:
            pass

    def set_viewer_display_mode(self, mode):
        """Configure display panels for viewer modes.

        Args:
            mode: 'image' — show only the raw 2D image panel,
                  'xye'   — show only the 1D plot panel,
                  'nexus' — show title-only center; details are in metadata,
                  None    — restore normal layout.
        """
        # Stage 2: a mode switch must invalidate any stale render computed
        # for the previous mode (the exact failure this guards against).
        if mode != self.viewer_mode:
            self._bump_display_generation()
            # Background is scoped to the mode it was set in: drop it on a real
            # mode change so a viewer's background never bleeds into Int (or the
            # other viewer).  Each mode starts with no background.
            self._clear_bkg()
            try:
                self._intensityAuto.blockSignals(True)
                self._intensityAuto.setChecked(True)
                self._intensityAuto.blockSignals(False)
            except Exception:
                pass
        self.viewer_mode = mode
        self._viewer_is_xdart = False
        self._viewer_x_axis_label = None
        self._payload_x_axis_label = None
        self._payload_y_axis_label = None
        if mode == 'image':
            title = 'Image Viewer'
        elif mode == 'xye':
            title = 'XYE Viewer'
        elif mode == 'nexus':
            title = 'NeXus Viewer'
        else:
            title = ''
        # A mode transition must not carry the previous mode's visible
        # image/curve or cached overlay data into the new one.
        self.clear_display_state(title)

        # Full panel geometry for this mode (idempotent, table-driven).  This
        # owns *all* of the height/width/visibility that used to be poked per
        # branch below — including the twoDWindow zero-height restore that a
        # prior 1D-only mode required (the Int 1D (XYE) -> Image Viewer blank).
        # A viewer mode maps to its own Mode; None/'' falls back to the current
        # processing mode's geometry.
        layout_mode = {
            'image': Mode.IMAGE_VIEWER,
            'xye': Mode.XYE_VIEWER,
            'nexus': Mode.NEXUS_VIEWER,
        }.get(mode)
        if layout_mode is None:
            layout_mode = (Mode.INT_1D if getattr(self.scan, 'skip_2d', False)
                           else Mode.INT_2D)
        self._apply_layout(layout_mode)
        set_intensity_visible = getattr(
            self, "_set_intensity_controls_visible", None)
        if set_intensity_visible is not None:
            set_intensity_visible(
                layout_mode in (Mode.IMAGE_VIEWER, Mode.XYE_VIEWER, Mode.INT_1D)
            )

        # Control-state the geometry table deliberately does not own.
        if mode == 'xye':
            # The XYE file owns its x-axis, so hide the transform combo; the
            # 2D-only controls (Share Axis, 2D unit, slice) are meaningless.
            # Unlink the (possibly checked) Share Axis WITHOUT unchecking --
            # the stale XLink froze the viewer 1D at the old cake's range;
            # returning to an INT mode re-links via _apply_share_axis_state.
            displayFrameWidget._set_share_link(self, False)
            _plot = getattr(self, 'plot', None)   # duck holders in tests
            if _plot is not None:
                _plot.enableAutoRange()
            set_middle_visible = getattr(
                self, "_set_middle_1d_controls_visible", None)
            if set_middle_visible is not None:
                set_middle_visible(False)
            self._set_2d_controls_visible(False)
            # frame_6 is shown so the Linear/Log scale applies to the 1D
            # plot; the colormap stays too (Vivek) — the XYE waterfall image
            # uses it, and Int 1D (XYE) processing mode shows it as well.
            if self.ui.cmap.parent() is not None:
                self.ui.cmap.setVisible(True)
            self.ui.cmap.setEnabled(True)
            self.ui.scale.setEnabled(True)
            self._show_viewer_set_bkg(True)   # Set BG subtracts a background XYE
        elif mode in ('image', 'nexus'):
            # Raw image / schema preview need no extra control state beyond the
            # geometry table.  The Linear/Log scale + colormap apply to the raw
            # image, so make sure both are shown/enabled (cmap may have been
            # hidden by a prior XYE-viewer visit).
            # cmap is back in the top bar (always parented), but keep the
            # guard: setVisible(True) on a PARENTLESS widget floats it as a
            # top-level window (the stray 'Default' popup bug).
            if self.ui.cmap.parent() is not None:
                self.ui.cmap.setVisible(True)
            self.ui.cmap.setEnabled(True)
            self.ui.scale.setEnabled(True)
            # Same stale-share unlink as the xye branch: the NeXus viewer
            # renders 1-D dataset previews into self.plot, which a checked
            # Share Axis left x-linked to the hidden cake with x-auto off --
            # previews froze at the old cake's range (and panning them
            # dragged the hidden cake, breaking the trim re-arm check).
            displayFrameWidget._set_share_link(self, False)
            _plot = getattr(self, 'plot', None)
            if _plot is not None:
                _plot.enableAutoRange()
            if mode == 'image':
                set_middle_visible = getattr(
                    self, "_set_middle_1d_controls_visible", None)
                if set_middle_visible is not None:
                    set_middle_visible(False)
                self._set_2d_controls_visible(False)
            # Image Viewer: Set BG subtracts a background raw frame; NeXus has no
            # subtractable display, so its button stays hidden.
            self._show_viewer_set_bkg(mode == 'image')
            if mode == 'nexus':
                self._set_equal_primary_panel_heights()
        else:
            # Normal mode — re-enable all process-mode controls.
            self.ui.normChannel.setVisible(True)   # viewer modes hid it
            self.ui.normChannel.setEnabled(True)
            self.ui.setBkg.setEnabled(True)
            self.ui.shareAxis.setEnabled(True)
            self.ui.imageUnit.setEnabled(True)
            self.ui.scale.setEnabled(True)
            self.ui.cmap.setEnabled(True)
            if self.ui.cmap.parent() is not None:
                self.ui.cmap.setVisible(True)   # restore if hidden by XYE viewer
            set_middle_visible = getattr(
                self, "_set_middle_1d_controls_visible", None)
            if set_middle_visible is not None:
                set_middle_visible(True)
            self.ui.plotUnit.setVisible(True)
            self.ui.plotUnit.setEnabled(True)
            # Re-baseline the Overlay/Waterfall unit tracker on the return to a
            # processing mode: a viewer round-trip leaves plotUnit hidden/rebuilt
            # and never updated _last_plot_unit, which then read as a spurious unit
            # change and sent the next live overlay down the REBUILD partial-read
            # collapse.  Sync it to the now-restored combo so only a genuine user
            # unit toggle registers.
            self._last_plot_unit = self.ui.plotUnit.currentIndex()
            self.ui.plotMethod.setEnabled(True)
            # Slice enable/disable depends on which axis is selected
            self._on_plotUnit_changed()
            self.ui.showLegend.setEnabled(True)
            self.ui.clear_1D.setEnabled(True)
            # Re-apply 2D-control visibility for the current processing mode
            # (Int 1D hides the 2D controls + slice; Int 2D shows all).  This
            # re-asserts the geometry via _apply_layout (idempotent).
            self._apply_1d_only_visibility()
            self._set_equal_primary_panel_heights()

    @staticmethod
    def _truncate_name(name, limit=100, head=48, tail=48):
        """Middle-ellipsis only very long filenames.  With the viewer top
        bar trimmed to just the label there's room for the full name, so the
        limit is generous (100 chars); longer names show the first ``head``
        and last ``tail`` characters with ``...`` between."""
        if name and len(name) > limit:
            return f'{name[:head]}...{name[-tail:]}'
        return name

    def _viewer_default_title(self):
        """The pristine title for the current viewer mode (no selection)."""
        return {
            'image': 'Image Viewer',
            'xye': 'XYE Viewer',
            'nexus': 'NeXus Viewer',
        }.get(getattr(self, 'viewer_mode', None), 'Viewer')

    def _set_viewer_title(self, idxs):
        """Set the title label in viewer modes from the selected frame's
        source file: plain filename for single-image formats (tiff/raw/xye),
        ``Filename #N`` for multi-image HDF5/NeXus files.  ``idxs`` is the
        list of selected frame keys; extra selections (XYE overlay) add a
        ``(+k more)`` suffix."""
        idxs = list(idxs) if idxs else []
        if not idxs:
            self.ui.labelCurrent.setText(self._viewer_default_title())
            return
        idx0 = idxs[0]
        src = ''
        with self.data_lock:
            d1 = self.data_1d.get(idx0)
        if d1 is not None:
            info = getattr(d1, 'scan_info', None) or {}
            src = info.get('source_file', '') or ''
        name = os.path.basename(src) if src else ''
        ext = os.path.splitext(name)[1].lower()
        if not name:
            title = 'Viewer'
        elif ext in ('.h5', '.hdf5', '.nxs'):
            title = f'{self._truncate_name(name)} #{idx0}'
        else:
            title = self._truncate_name(name)
        if len(idxs) > 1:
            title += f'  (+{len(idxs) - 1} more)'
        self.ui.labelCurrent.setText(title)

    # ── Image preview dialog ──────────────────────────────────────

    @staticmethod
    def _fit_combo_width(combo, *, max_w=200, arrow=34):
        """Fixed width = widest item text + dropdown-arrow allowance."""
        try:
            fm = combo.fontMetrics()
            texts = [combo.itemText(i) for i in range(combo.count())] or ['']
            w = max(fm.horizontalAdvance(t) for t in texts)
            combo.setFixedWidth(min(w + arrow, max_w))
        except Exception:
            logger.debug("combo width fit failed", exc_info=True)

    @staticmethod
    def _fit_button_width(btn, *, pad=26, scale=1.0):
        """Fixed width = (label text + padding) * scale."""
        try:
            w = btn.fontMetrics().horizontalAdvance(btn.text()) + pad
            btn.setFixedWidth(int(round(w * scale)))
        except Exception:
            logger.debug("button width fit failed", exc_info=True)

    @staticmethod
    def integration_view_image(thumb, scan):
        """Return the raw image AS THE INTEGRATION SAW IT.

        Applies the detector/global mask and the run's intensity threshold
        as NaN (pgImageWidget renders NaN transparent), mirroring
        ``_resolve_frame_mask`` + ``_apply_threshold_inline`` on the worker.
        The Image Viewer mode deliberately does NOT use this -- it shows the
        untouched raw image."""
        img = np.asarray(thumb, dtype=np.float32).copy()
        gm = getattr(scan, 'global_mask', None)
        if gm is not None:
            try:
                flat = np.asarray(gm).ravel().astype(np.int64)
                ok = (flat >= 0) & (flat < img.size)
                img.ravel()[flat[ok]] = np.nan
            except Exception:
                logger.debug("preview mask apply failed", exc_info=True)
        if bool(getattr(scan, 'apply_threshold', False)):
            try:
                tmin = float(getattr(scan, 'threshold_min', 0) or 0)
                tmax = float(getattr(scan, 'threshold_max', 0) or 0)
                img[(img < tmin) | (img > tmax)] = np.nan
            except Exception:
                logger.debug("preview threshold apply failed", exc_info=True)
        return img

    def _show_image_preview(self):
        """Open a popup dialog showing the raw image thumbnail for the
        currently selected frame."""
        # Determine the selected frame index
        idx = None
        if self.idxs_1d:
            idx = self.idxs_1d[-1]  # last selected frame

        if idx is None:
            QtWidgets.QMessageBox.information(
                self, 'No Data', 'No frame is currently selected.')
            return

        # Create the preview dialog if it doesn't exist yet
        if self._image_preview_dialog is None:
            dlg = QDialog(self)
            dlg.setWindowTitle('Raw Image Preview')
            dlg.resize(600, 600)
            layout = QtWidgets.QVBoxLayout(dlg)
            layout.setContentsMargins(2, 2, 2, 2)
            pw = pgImageWidget(lockAspect=True, raw=True)
            layout.addWidget(pw)
            self._image_preview_dialog = dlg
            self._image_preview_widget = pw

        # Show the dialog first so _update_image_preview's visibility
        # check passes, then update the image content.
        self._image_preview_dialog.show()
        self._image_preview_dialog.raise_()
        self._update_image_preview(idx, show_message=True)

    def _update_image_preview(self, idx=None, show_message=False):
        """Update the image preview dialog with the thumbnail for *idx*.

        If *idx* is None the last selected 1D frame is used.  When
        *show_message* is True an info dialog is shown if there is no
        thumbnail available; otherwise the call silently returns.
        """
        if self._image_preview_dialog is None:
            return
        if not self._image_preview_dialog.isVisible():
            return

        if idx is None:
            idx = self.idxs_1d[-1] if self.idxs_1d else None
        if idx is None:
            return

        # Try the publication store first.  The old data_1d/data_2d mirrors are
        # still used by viewer modes, but integration previews should survive
        # once Role-A mirror writes are gone.
        thumb = None
        full_res = False
        try:
            pub = self._publication_from_store_for_display(
                int(idx), allow_blocking_read=False)
            if pub is not None:
                view = getattr(pub, 'view', None)
                thumb = getattr(view, 'thumbnail', None)
                if thumb is None:
                    thumb = getattr(view, 'raw', None)
                    full_res = thumb is not None
                if thumb is None:
                    raw_ref = getattr(pub, 'raw_ref', None)
                    thumb = getattr(raw_ref, 'thumbnail', None)
                if thumb is None:
                    raw_ref = getattr(pub, 'raw_ref', None)
                    thumb = getattr(raw_ref, 'map_raw', None)
                    full_res = thumb is not None
        except Exception:
            logger.debug("store image-preview lookup failed", exc_info=True)

        if thumb is None:
            with self.data_lock:
                frame = self.data_1d.get(int(idx))
            if frame is not None:
                thumb = getattr(frame, 'thumbnail', None)

        # Fall back to 2D data dict.  Snapshot under data_lock: a concurrent
        # eviction between an `in` check and the read raised KeyError on the
        # GUI thread.
        if thumb is None:
            with self.data_lock:
                d2 = self.data_2d.get(int(idx))
            if d2 is not None:
                thumb = d2.get('map_raw')
                full_res = thumb is not None

        if thumb is None or (hasattr(thumb, 'size') and thumb.size == 0):
            if show_message:
                QtWidgets.QMessageBox.information(
                    self, 'No Image',
                    f'No image data available for frame {idx}.')
            return

        # Show the image AS INTEGRATED (mask + threshold as NaN), through the
        # standard pgImageWidget path so the (2, 98) nanpercentile levels
        # apply.  Mask/threshold only on the FULL-RES map_raw path:
        # thumbnails are downsampled (<=256px) with the mask already baked
        # in as NaN, so the full-res flat mask indices would land on
        # unrelated pixels (speckles), and their values are bg-subtracted /
        # interpolated -- not the raw counts the worker thresholds.
        if full_res:
            img = displayFrameWidget.integration_view_image(
                thumb, getattr(self, 'scan', None))
        else:
            img = np.asarray(thumb, dtype=np.float32)
        # Correct orientation: transpose and flip vertically.
        self._image_preview_widget.setImage(img.T[:, ::-1])
        self._image_preview_dialog.setWindowTitle(
            f'Raw Image Preview \u2014 Frame {idx}')
