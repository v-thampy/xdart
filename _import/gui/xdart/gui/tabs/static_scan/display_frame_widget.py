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
    default_plot_unit, pretty_unit,
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

# Switch to using white background and black foreground
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


class displayFrameWidget(DisplayDataMixin, DisplayPlotMixin, Qt.QtWidgets.QWidget):
    # Emitted whenever the user changes the plot method combo
    # (Single / Overlay / Waterfall / Sum / Average). Listeners (e.g. the
    # H5Viewer) use this to switch listData selection mode so accumulating
    # plot methods don't require shift/ctrl multi-select.
    sigPlotMethodChanged = Qt.QtCore.Signal(str)

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
        displayFrameWidget._fit_button_width(self.ui.setBkg)
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
        self._set_tooltips()
        self._set_equal_primary_panel_heights()

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

        # Rebuild the middle bar: 1D controls, stretch, then 2D controls.
        for w in ones:
            mid.addWidget(w)
        mid.addStretch(1)
        mid.addWidget(self.ui.shareAxis)
        mid.addWidget(self.ui.imageUnit)

        # The bottom bar is empty now — collapse it so the 1D plot grows.
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
        self.data_1d = data_1d
        self.data_2d = data_2d
        self.bkg_1d = 0.
        self.bkg_2d = 0.
        self.bkg_map_raw = 0.
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
        self.ui.shareAxis.toggled.connect(self.update)
        # On *unchecking* Share Axis, release the x-link and rescale the 1D
        # plot to its own data (update() relinks/unlinks but leaves the
        # range frozen at the cake's).  Connected after update so the unlink
        # has already happened.
        self.ui.shareAxis.toggled.connect(self._on_share_axis_toggled)

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
        self.ui.plotUnit.activated.connect(self.update_plot)
        self.ui.showLegend.toggled.connect(self.update_legend)
        self.ui.slice.toggled.connect(self._sync_slice_controls)
        self.ui.slice.toggled.connect(self.update_plot)
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
        # Rebuild the right cluster in its final order: Raw | colormap | Log
        # (Log at the right corner in every mode).
        _l9 = self.ui.horizontalLayout_9
        while _l9.count():
            _l9.takeAt(0)
        for _w in (self._showImageBtn, self.ui.cmap, self._logBtn):
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

                # Snapshot current dict keys while the lock is held, then
                # release before doing list-comprehension work.
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
        return self.display_generation

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

    def _live_mode(self):
        """Map the widget's viewer state to a :class:`Mode`.  Normal mode is
        INT_1D when the scan is 1D-only (skip_2d) — plot-only, matching
        _apply_1d_only_visibility — else INT_2D (raw|cake / plot)."""
        if self.viewer_mode == 'image':
            return Mode.IMAGE_VIEWER
        if self.viewer_mode == 'xye':
            return Mode.XYE_VIEWER
        if self.viewer_mode == 'nexus':
            return Mode.NEXUS_VIEWER
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
            self.update_plot()

    # ── Update orchestration ──────────────────────────────────────

    def _updated(self):
        """Check if there is data to update
        """
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
        if (len(self.data_1d) == 0) or (len(self.idxs_1d) == 0):
            return False

        return True

    def update(self):
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
            return self.update_image
        if role is PanelRole.PLOT_1D:
            if mode in (Mode.XYE_VIEWER, Mode.NEXUS_VIEWER):
                return self.clear_plot_view
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
        self.overlaid_idxs = list(state.render_ids)
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
        widget.image_plot.setLabel(
            "bottom", payload.axis_x.label, units=pretty_unit(payload.axis_x.unit),
        )
        widget.image_plot.setLabel(
            "left", payload.axis_y.label, units=pretty_unit(payload.axis_y.unit),
        )
        if role is PanelRole.RAW_2D:
            displayFrameWidget._set_raw_pixel_axes(widget)
            self.image_data = (image, rect)
        else:
            self.binned_data = (image, rect)
        return True

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

    def _set_share_link(self, on: bool) -> None:
        """Share Axis = dev semantics: the native pyqtgraph XLink.

        The 1D plot's frame is untouched (its y-axis stays at the left end
        of its pane); the x-range is GEOMETRY-mapped so the screen columns
        the two panes share line up vertically, and the link is
        bidirectional -- zooming either plot moves both.  The only addition
        over dev: the 1D's continuous x-auto-range is disabled while shared,
        so render refits never drag the cake (that interaction, exposed by
        the cake's trim-to-data, was the 'inverted share' regression)."""
        ip = getattr(getattr(self, 'binned_widget', None),
                     'image_plot', None)
        if ip is None or not hasattr(ip, 'getViewBox'):
            return          # duck holders in tests / widget not fully built
        already = getattr(self, '_share_link_on', False)
        if on and not already:
            self._share_link_on = True
            self.plot.setXLink(ip)
            try:
                self.plot.enableAutoRange(x=False, y=True)
                # Snap NOW: pyqtgraph only syncs a fresh link on the next
                # master change, so force one initial linkedViewChanged.
                vbp = self.plot.getViewBox()
                vbp.linkedViewChanged(ip.getViewBox(), vbp.XAxis)
            except Exception:
                logger.debug("share x-auto disable failed", exc_info=True)
        elif not on and already:
            self._share_link_on = False
            self.plot.setXLink(None)

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

            for label in imageUnits:
                self.ui.imageUnit.addItem(_translate("Form", label))
            self.ui.plotUnit.setEnabled(True)
            self.ui.imageUnit.setEnabled(True)
            # Default the plot unit to the entry matching the 1D integration
            # unit (so a 2θ integration opens on a 2θ axis).  Standard combo
            # order is (Q, 2θ, χ); normalise the pyFAI unit to canonical then
            # route the index choice through the pure default_plot_unit.
            unit_1d = str(self.scan.bai_1d_args.get('unit', '')).lower()
            canon_1d = ('2th_deg' if '2th' in unit_1d
                        else 'chi_deg' if 'chi' in unit_1d else 'q_A^-1')
            target_plot_idx = default_plot_unit(
                canon_1d, ('q_A^-1', '2th_deg', 'chi_deg'))

        if self.ui.plotUnit.count() > 0:
            self.ui.plotUnit.setCurrentIndex(
                min(target_plot_idx, self.ui.plotUnit.count() - 1)
            )
        self.ui.plotUnit.blockSignals(False)
        self.ui.imageUnit.blockSignals(False)

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

    def _on_share_axis_toggled(self, checked):
        """Rescale the 1D plot to its own data when Share Axis is turned off.

        While shared, the 1D plot's x-axis is XLinked to the 2D cake; the
        ``update`` slot calls ``setXLink(None)`` on uncheck but pyqtgraph
        leaves the view frozen at the cake's range, so the user sees a
        stuck axis.  Re-enable autoRange so it fits the 1D curve."""
        if not checked:
            try:
                # Immediate fit, then re-arm continuous tracking (see
                # _autorange_plot_view for why the order matters).
                self.plot.autoRange()
                self.plot.enableAutoRange()
            except Exception:
                logger.debug("1D autoscale on Share Axis off failed",
                             exc_info=True)

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

            # Apply Mask — O8: snapshot under data_lock so a
            # concurrent writer (integrator publish, fileHandlerThread
            # load) can't evict ``self.idxs_2d[0]`` between the
            # ``in`` check and the value read.  ``.get(...)`` returns
            # None for an evicted key; falling back to None mask is
            # the same as having no mask, so render continues.
            with self.data_lock:
                frame_2d = self.data_2d.get(self.idxs_2d[0])
            mask = frame_2d['mask'] if frame_2d is not None else None
        data = np.asarray(data, dtype=float)

        # Apply detector + global mask only to full-resolution raw data.
        # Thumbnails already bake the mask into the preview before
        # downsampling; flat detector indices point at unrelated pixels there.
        if raw_source == 'raw':
            global_mask = (
                self.scan.global_mask if self.scan.global_mask is not None else []
            )
            mask = mask if mask is not None else []
            mask = np.asarray(np.unique(np.append(mask, global_mask)), dtype=int)
            if len(mask) > 0 and mask.max() < data.size:
                mask = np.unravel_index(mask, data.shape)
                data[mask] = np.nan

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
            except Exception:
                logger.debug("raw pixel axis scale update failed", exc_info=True)

    def update_2d_label(self):
        """Updates 2D Label
        """
        # Sets title text
        label = self.scan.name
        if len(label) > 40:
            label = f'{label[:18]}...{label[-18:]}'

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
        self.update()

    def setBkg(self):
        """Sets selected points as background.
        If background is already selected, it unsets it"""
        if (len(self.frame_ids) == 0) or (len(self.idxs) == 0):
            return

        if self.ui.setBkg.text() == 'Set Bkg':
            idxs = self.frame_ids
            if self.overall:
                idxs = sorted(list(self.scan.frames.index))

            # #6: refuse a PARTIAL 2D background rather than silently averaging
            # only the frames whose int_2d happens to be available — a partial
            # average is a wrong background, not a smaller one.  require_all=True
            # returns None when not every selected frame contributes; a None
            # here is only partial coverage (not a 1D-only scan) if the
            # subset-average path WOULD have returned something.
            bkg_2d, _, _ = self.get_frames_int_2d(idxs, require_all=True)
            if bkg_2d is None and self.get_frames_int_2d(idxs)[0] is not None:
                logger.error(
                    "Set Bkg refused: the 2D background covers only part of the "
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
                return  # button stays 'Set Bkg'; no wrong background applied

            self.bkg_1d, _ = self.get_frames_int_1d(idxs, rv='average')
            self.bkg_2d = bkg_2d
            self.bkg_map_raw = self.get_frames_map_raw(idxs, require_all=True)
            if self.bkg_map_raw is None:
                # F5: be honest about a no-op 2D background.  Pre-F5
                # this silently set bkg=0.: 1D/2D bkg subtraction
                # would still apply but the user saw "Clear Bkg" on
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
            self.ui.setBkg.setText('Clear Bkg')
        else:
            self.bkg_1d = 0.
            self.bkg_2d = 0.
            self.bkg_map_raw = 0.
            self.ui.setBkg.setText('Set Bkg')

        self.update()
        return

    # ── Viewer modes ──────────────────────────────────────────────

    def clear_overlay(self):
        """Drop accumulated overlay curves + names."""
        self.plot_data = [np.zeros(0), np.zeros(0)]
        self.plot_data_range = [[0, 0], [0, 0]]
        self.frame_names = []
        self.overlaid_idxs = []

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
            self.ui.plotUnit.setVisible(False)
            self._set_2d_controls_visible(False)
            # frame_6 is shown so the Linear/Log scale applies to the 1D
            # plot; the colormap stays too (Vivek) — the XYE waterfall image
            # uses it, and Int 1D (XYE) processing mode shows it as well.
            if self.ui.cmap.parent() is not None:
                self.ui.cmap.setVisible(True)
            self.ui.cmap.setEnabled(True)
            self.ui.scale.setEnabled(True)
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
            if mode == 'nexus':
                self._set_equal_primary_panel_heights()
        else:
            # Normal mode — re-enable all process-mode controls.
            self.ui.normChannel.setEnabled(True)
            self.ui.setBkg.setEnabled(True)
            self.ui.shareAxis.setEnabled(True)
            self.ui.imageUnit.setEnabled(True)
            self.ui.scale.setEnabled(True)
            self.ui.cmap.setEnabled(True)
            if self.ui.cmap.parent() is not None:
                self.ui.cmap.setVisible(True)   # restore if hidden by XYE viewer
            self.ui.plotUnit.setVisible(True)
            self.ui.plotUnit.setEnabled(True)
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

    def _set_viewer_title(self, idxs):
        """Set the title label in viewer modes from the selected frame's
        source file: plain filename for single-image formats (tiff/raw/xye),
        ``Filename #N`` for multi-image HDF5/NeXus files.  ``idxs`` is the
        list of selected frame keys; extra selections (XYE overlay) add a
        ``(+k more)`` suffix."""
        idxs = list(idxs) if idxs else []
        if not idxs:
            self.ui.labelCurrent.setText('Image Viewer')
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
    def _fit_button_width(btn, *, pad=26):
        """Fixed width = label text + padding."""
        try:
            btn.setFixedWidth(btn.fontMetrics().horizontalAdvance(btn.text()) + pad)
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

        # Try to get thumbnail from loaded 1D data
        thumb = None
        full_res = False
        frame = self.data_1d.get(int(idx))
        if frame is not None:
            thumb = getattr(frame, 'thumbnail', None)

        # Fall back to 2D data dict
        if thumb is None and int(idx) in self.data_2d:
            d2 = self.data_2d[int(idx)]
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
