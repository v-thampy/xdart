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
    empty_display_state,
    resolve_selection, resolve_render_ids,
    xye_unit_from_filename, x_axis_for_unit, default_plot_unit,
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

# Switch to using white background and black foreground
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


class displayFrameWidget(DisplayDataMixin, DisplayPlotMixin, Qt.QtWidgets.QWidget):
    # Emitted whenever the user changes the plot method combo
    # (Single / Overlay / Waterfall / Sum / Average). Listeners (e.g. the
    # H5Viewer) use this to switch listData selection mode so accumulating
    # plot methods don't require shift/ctrl multi-select.
    sigPlotMethodChanged = Qt.QtCore.Signal(str)

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

        # 2D image controls
        self.ui.imageUnit.activated.connect(self.update_binned)
        self.ui.imageUnit.activated.connect(self._update_slice_range)

        # 1D plot controls
        self.ui.plotMethod.currentIndexChanged.connect(self._on_plotMethod_changed)
        self.ui.yOffset.valueChanged.connect(self.update_plot_view)
        self.ui.plotUnit.activated.connect(self._on_plotUnit_changed)
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
        self._showImageBtn = QtWidgets.QPushButton('Show Image')
        self._showImageBtn.setMinimumSize(QtWidgets.QWidget().minimumSize())
        self._showImageBtn.setMaximumSize(Qt.QtCore.QSize(90, 16777215))
        self._showImageBtn.setToolTip('Show raw image preview for selected frame')
        self._showImageBtn.setFocusPolicy(pyQt.StrongFocus)
        self.ui.horizontalLayout_9.addSpacerItem(
            QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Policy.Fixed,
                                  QtWidgets.QSizePolicy.Policy.Minimum))
        self.ui.horizontalLayout_9.addWidget(self._showImageBtn)
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

        with self.data_lock:
            with self.scan.scan_lock:
                # Stage 1: selection logic is the pure ``resolve_selection``
                # / ``resolve_render_ids`` (unit-tested headlessly).
                try:
                    ids, self.overall = resolve_selection(
                        self.frame_ids, self.scan.frames.index)
                except ValueError:
                    return

            self.idxs = list(ids)
            # Snapshot current dict keys while the lock is held, then release
            # before doing list-comprehension work.
            data_1d_keys = set(self.data_1d.keys())
            data_2d_keys = set(self.data_2d.keys())

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
            if has_cached:
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
        if role is PanelRole.RAW_2D:
            return (self._update_image_viewer if mode is Mode.IMAGE_VIEWER
                    else self.update_image)
        if role is PanelRole.PLOT_1D:
            return (self._update_xye_viewer if mode is Mode.XYE_VIEWER
                    else self.update_plot)
        if role is PanelRole.CAKE_2D:
            return self.update_binned
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
            return self._draw_image_payload(role, payload_value)

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

    def _draw_image_payload(self, role, payload):
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
        widget.image_plot.setLabel(
            "bottom", payload.axis_x.label, units=payload.axis_x.unit,
        )
        widget.image_plot.setLabel(
            "left", payload.axis_y.label, units=payload.axis_y.unit,
        )
        if role is PanelRole.RAW_2D:
            self.image_data = (image, rect)
        else:
            self.binned_data = (image, rect)
        return True

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
            if self.ui.shareAxis.isChecked() and (self.ui.imageUnit.currentIndex() < 2):
                self.ui.plotUnit.setCurrentIndex(self.ui.imageUnit.currentIndex())
                self.ui.plotUnit.setEnabled(False)
                self.plot.setXLink(self.binned_widget.image_plot)
            else:
                self.plot.setXLink(None)
                self.ui.plotUnit.setEnabled(True)
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
        return True

    def update_views(self):
        """Updates 2D (if flag is selected) and 1D views
        """
        if not self._updated():
            return True

        self.cmap = self.ui.cmap.currentText()
        self.plotMethod = self.ui.plotMethod.currentText()
        self.scale = self.ui.scale.currentText()

        if self.viewer_mode == 'image':
            # Image viewer: only update the raw image panel
            self.update_image_view()
            return

        self.update_image_view()
        self.update_binned_view()
        self.update_2d_label()
        self.update_plot_view()

    # ── 1D-only visibility ────────────────────────────────────────

    def _apply_1d_only_visibility(self):
        """Show or hide 2D panes based on scan.skip_2d.

        In 1D-only mode (skip_2d), collapse the 2D image panels and
        image toolbar while keeping the top toolbar (Norm Channel, Scale,
        Set Bkg, etc.) visible.  Also removes pure-2D entries (like χ)
        from the plotUnit combo so the user cannot select them.
        """
        # In viewer mode, set_viewer_display_mode() controls panels
        if self.viewer_mode is not None:
            return
        skip = getattr(self.scan, 'skip_2d', False)
        if skip:
            # Hide the 2D image but KEEP the middle control bar
            # (imageToolbar) — it now holds the 1D plot controls.  Shrink
            # imageWindow to the title + control bar (frame_top 35 +
            # imageToolbar 40).
            self.ui.twoDWindow.setMaximumHeight(0)
            self.ui.twoDWindow.setMinimumHeight(0)
            self.ui.imageToolbar.setMinimumHeight(40)
            self.ui.imageToolbar.setMaximumHeight(40)
            self.ui.imageWindow.setMinimumHeight(80)
            self.ui.imageWindow.setMaximumHeight(85)
            # 2D-only controls (Share Axis, 2D unit, X-Range slice) off.
            if self.ui.slice.isChecked():
                self.ui.slice.setChecked(False)
            self._set_2d_controls_visible(False)

            # Show the raw image preview button in 1D-only mode
            self._showImageBtn.setVisible(True)

            # Remove pure-2D entries from plotUnit (e.g. χ)
            self.ui.plotUnit.blockSignals(True)
            i = 0
            while i < self.ui.plotUnit.count():
                if i < len(self._plot_axis_info):
                    info = self._plot_axis_info[i]
                    if info['source'] == '2d':
                        self.ui.plotUnit.removeItem(i)
                        self._plot_axis_info.pop(i)
                        continue
                i += 1
            self.ui.plotUnit.blockSignals(False)
            self._was_skip_2d = True
        else:
            # Restore the 2D image + all controls.
            self.ui.twoDWindow.setMinimumHeight(0)
            self.ui.twoDWindow.setMaximumHeight(16777215)
            self.ui.imageToolbar.setMinimumHeight(40)
            self.ui.imageToolbar.setMaximumHeight(40)
            self.ui.imageWindow.setMinimumHeight(200)
            self.ui.imageWindow.setMaximumHeight(16777215)
            self._set_2d_controls_visible(True)
            # Hide the raw image preview button in 2D modes
            self._showImageBtn.setVisible(False)
            # Only rebuild plotUnit when transitioning from 1D-only mode,
            # otherwise preserve the user's current plotUnit selection.
            if self._was_skip_2d:
                self.set_axes()
                self._was_skip_2d = False

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

            # --- 2D-derived axes ---
            if radial_label != label_1d:
                self.ui.plotUnit.addItem(_translate("Form", radial_label))
                self._plot_axis_info.append({
                    'source': '2d', 'slice_axis': azimuthal_label,
                    'axis': 'radial',
                })

            if azimuthal_label != label_1d and azimuthal_label != radial_label:
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
            # χ is derived from 2D
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

        # Share Axis only makes sense when the 1D plot and the 2D cake can
        # be on the SAME x-axis — i.e. the 1D plotUnit is the radial axis
        # (Q or 2θ), which the cake also uses for its x.  When the 1D plot
        # is on χ (the cake's azimuthal/y axis) there's nothing to share,
        # so disable it.  Also disabled in 1D-only mode (no 2D cake).
        can_share = (not skip_2d) and (
            info.get('axis') == 'radial' or info['source'] == '1d_2d'
        )
        was_checked = self.ui.shareAxis.isChecked()
        self.ui.shareAxis.setEnabled(can_share)
        if not can_share and was_checked:
            # Auto-disabling Share Axis (e.g. switched 1D to χ): unchecking
            # emits ``toggled`` → update() (unlinks the axes) +
            # _on_share_axis_toggled() (rescales the 1D plot to its data).
            self.ui.shareAxis.setChecked(False)

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
                self.plot.enableAutoRange()
                self.plot.autoRange()
            except Exception:
                logger.debug("1D autoscale on Share Axis off failed",
                             exc_info=True)

        # Update slice range label
        self._set_slice_range()

    # ── 2D image rendering ────────────────────────────────────────

    def update_image(self):
        """Updates image plotted in image frame.

        Applies the detector-level mask and global mask to the raw image.
        If the data is a downsampled thumbnail (mask already baked in as
        NaN), the mask application is skipped because the flat indices
        would not match the thumbnail's smaller shape.
        """
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
                self.clear_image_view()
                return
        else:
            data, raw_source = self.get_frames_map_raw(return_source=True)
            if data is None:
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
            self.clear_image_view()
            return

        data = data.T[:, ::-1]

        # Get Bounding Rectangle
        rect = get_rect(np.arange(data.shape[0]), np.arange(data.shape[1]))

        self.image_data = (data, rect)
        self.update_image_view()

    def update_image_view(self):
        data, rect = self.image_data

        display_data = _downsample_for_display(data, self.image_widget)
        self.image_widget.setImage(display_data, scale=self.scale, cmap=self.cmap)
        self.image_widget.setRect(rect)

        self.image_widget.image_plot.setLabel("bottom", 'x (Pixels)')
        self.image_widget.image_plot.setLabel("left", 'y (Pixels)')

    def update_binned(self):
        """Updates image plotted in image frame.

        Note: when shareAxis is checked, the plotUnit sync and
        update_plot() are already handled by the main display_update
        flow before update_binned is called.  We only need to refresh
        the view here (no data re-accumulation).
        """
        if self.ui.shareAxis.isChecked() and (self.ui.imageUnit.currentIndex() < 2):
            self.ui.plotUnit.setCurrentIndex(self.ui.imageUnit.currentIndex())
            self.update_plot_view()

        # Always aggregate from per-frame data_2d.  In Overall mode require
        # every frame's 2D row to be present; otherwise a bounded cache would
        # quietly average only the rows that happen to be loaded.
        if self.overall and len(self.frame_ids) > 1:
            intensity, xdata, ydata = self.get_frames_int_2d(
                list(self.scan.frames.index), require_all=True,
            )
        else:
            intensity, xdata, ydata = self.get_frames_int_2d()

        if intensity is None:
            self.clear_binned_view()
            return

        # Subtract background
        if self.bkg_2d is not None:
            intensity -= self.bkg_2d

        rect = get_rect(xdata, ydata)
        self.binned_data = (intensity, rect)
        self.update_binned_view()

        return

    def update_binned_view(self):
        data, rect = self.binned_data

        display_data = _downsample_for_display(data, self.binned_widget)
        self.binned_widget.setImage(display_data, scale=self.scale, cmap=self.cmap)
        self.binned_widget.setRect(rect)

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
        self.binned_widget.image_plot.setLabel("bottom", _xl2, units=_xu2)
        self.binned_widget.image_plot.setLabel("left", _yl2, units=_yu2)

        self.show_slice_overlay()
        return data

    def update_2d_label(self):
        """Updates 2D Label
        """
        # Sets title text
        label = self.scan.name
        if len(label) > 40:
            label = f'{label[:10]}.....{label[-30:]}'

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
        if self.normChannel and (self.scan.scan_data[self.normChannel].sum() == 0.):
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

            self.bkg_1d, _ = self.get_frames_int_1d(idxs, rv='average')
            self.bkg_2d, _, _ = self.get_frames_int_2d(idxs)
            self.bkg_map_raw = self.get_frames_map_raw(idxs)
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
            for curve in self.curves:
                curve.clear()
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
        self._viewer_x_axis_label = None
        self._payload_x_axis_label = None
        self._payload_y_axis_label = None
        is_viewer = mode in ('image', 'xye', 'nexus')
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
        # In a viewer the top bar carries only the filename label (frame_5):
        # the norm/bkg controls (frame_4) and scale/cmap (frame_6) are
        # process-mode concerns.  The bottom toolbar is permanently
        # collapsed (its controls moved to the middle bar in _reflow).
        self.ui.frame_4.setVisible(not is_viewer)
        self.ui.frame_6.setVisible(not is_viewer)
        self.ui.plotToolBar.setVisible(False)

        if mode == 'image':
            # Top bar (filename only) + raw image; no controls, collapse 1D.
            self.ui.frame_top.setVisible(True)
            self.ui.twoDWindow.setVisible(True)
            self.ui.imageToolbar.setVisible(False)
            self.ui.imageWindow.setMinimumHeight(200)
            self.ui.imageWindow.setMaximumHeight(16777215)
            self.ui.plotWindow.setMinimumHeight(0)
            self.ui.plotWindow.setMaximumHeight(0)
            # Hide binned frame — only raw image relevant
            self.ui.binnedFrame.setMaximumWidth(0)
            self.ui.binnedFrame.setMinimumWidth(0)
            self._showImageBtn.setVisible(False)
        elif mode == 'xye':
            # 1D overlay view: hide the 2D image but KEEP the middle control
            # bar (Single/Options/Legend/Clear are useful for overlays).
            # The XYE file owns its x-axis, so hide the transform combo.
            self.ui.frame_top.setVisible(True)
            self.ui.twoDWindow.setVisible(False)
            self.ui.imageToolbar.setVisible(True)
            self.ui.plotUnit.setVisible(False)
            self._set_2d_controls_visible(False)
            self.ui.imageWindow.setMinimumHeight(80)
            self.ui.imageWindow.setMaximumHeight(85)
            self.ui.plotWindow.setMinimumHeight(200)
            self.ui.plotWindow.setMaximumHeight(16777215)
        elif mode == 'nexus':
            # Schema/detail view with bounded dataset previews: selected
            # 1D datasets draw in the plot panel, selected 2D datasets draw
            # in the image panel, and metadata-only rows blank both.
            self.ui.frame_top.setVisible(True)
            self.ui.twoDWindow.setVisible(True)
            self.ui.imageToolbar.setVisible(False)
            self.ui.imageWindow.setMinimumHeight(200)
            self.ui.imageWindow.setMaximumHeight(16777215)
            self.ui.plotWindow.setMinimumHeight(200)
            self.ui.plotWindow.setMaximumHeight(16777215)
            self.ui.binnedFrame.setMaximumWidth(0)
            self.ui.binnedFrame.setMinimumWidth(0)
            self._showImageBtn.setVisible(False)
        else:
            # Normal mode — restore panels + the middle control bar.
            self.ui.frame_top.setVisible(True)
            self.ui.twoDWindow.setVisible(True)
            self.ui.imageToolbar.setVisible(True)
            self.ui.imageWindow.setMinimumHeight(200)
            self.ui.imageWindow.setMaximumHeight(16777215)
            self.ui.plotWindow.setMinimumHeight(200)
            self.ui.plotWindow.setMaximumHeight(16777215)
            # Restore binned frame
            self.ui.binnedFrame.setMinimumWidth(0)
            self.ui.binnedFrame.setMaximumWidth(16777215)
            # Re-enable all controls
            self.ui.normChannel.setEnabled(True)
            self.ui.setBkg.setEnabled(True)
            self.ui.shareAxis.setEnabled(True)
            self.ui.imageUnit.setEnabled(True)
            self.ui.scale.setEnabled(True)
            self.ui.cmap.setEnabled(True)
            self.ui.plotUnit.setVisible(True)
            self.ui.plotUnit.setEnabled(True)
            self.ui.plotMethod.setEnabled(True)
            # Slice enable/disable depends on which axis is selected
            self._on_plotUnit_changed()
            self.ui.showLegend.setEnabled(True)
            self.ui.clear_1D.setEnabled(True)
            # Re-apply 2D-control visibility for the current processing
            # mode (Int 1D hides the 2D controls + slice; Int 2D shows all).
            self._apply_1d_only_visibility()

    def _update_xye_viewer(self):
        """Render 1D line data in XYE viewer mode (no scan/integration).

        Builds plot_data and frame_names from the loaded XYE data,
        then delegates to the standard update_plot_view() so that
        Single/Overlay/Waterfall/Sum/Average all work.
        """
        if len(self.idxs_1d) == 0:
            self.clear_plot_view()
            return

        # Title bar: the xye filename(s) — same convention as the image viewer.
        self._set_viewer_title(self.idxs_1d)

        # Axis label comes from the file's prefix (iq \u2192 Q, itth \u2192 2\u03b8), not a
        # transform combo \u2014 an XYE file can't be re-transformed.  Stage 4:
        # route through the pure xye_unit_from_filename + x_axis_for_unit so
        # an UNKNOWN prefix labels the axis plain 'x' (never an assumed 2\u03b8).
        first_frame = self.data_1d[self.idxs_1d[0]]
        source_name = ''
        if getattr(first_frame, 'scan_info', None):
            source_name = first_frame.scan_info.get('source_file', '')
        first_unit = xye_unit_from_filename(source_name)
        xlabel, xunits = x_axis_for_unit(first_unit)
        self._viewer_x_axis_label = (xlabel, xunits)

        # Overlaying xye files of different units mixes incompatible x-axes —
        # we label from the first file (above); warn if the selection mixes
        # known prefixes (iq with itth) so the user knows the axis is the
        # first file's, not a shared one.
        if len(self.idxs_1d) > 1:
            units = set()
            for idx in self.idxs_1d:
                fr = self.data_1d.get(idx)
                sinfo = getattr(fr, 'scan_info', None) if fr is not None else None
                src = sinfo.get('source_file', '') if sinfo else ''
                u = xye_unit_from_filename(src)
                if u != 'unknown':
                    units.add(u)
            if len(units) > 1:
                logger.warning(
                    'XYE overlay mixes different x-axis units %s; labelling '
                    'the axis from the first file (%s). Overlaid curves are '
                    'on incompatible axes.', sorted(units), first_unit,
                )

        # Build xdata and ydata arrays from all selected indices.
        ref_x = np.asarray(first_frame.int_1d.radial, dtype=float)
        frame_names = []
        rows = []
        for idx in self.idxs_1d:
            frame = self.data_1d[idx]
            int_1d = frame.int_1d
            if int_1d is None:
                continue
            fname = frame.scan_info.get('source_file', f'xye_{idx}')
            frame_names.append(os.path.basename(fname))
            xdata = np.asarray(int_1d.radial, dtype=float)
            ydata = np.asarray(int_1d.intensity, dtype=float)
            if xdata.shape != ref_x.shape or not np.allclose(xdata, ref_x):
                ydata = np.interp(ref_x, xdata, ydata)
            rows.append(ydata)

        if not rows:
            self.clear_plot_view()
            return

        xdata = ref_x
        ydata = np.vstack(rows) if len(rows) > 1 else rows[0][np.newaxis, :]

        # In Overlay/Waterfall: accumulate, skip duplicates.
        current_method = self.ui.plotMethod.currentText()
        if current_method in ('Overlay', 'Waterfall') and \
                len(self.plot_data[0]) > 0 and \
                self.plot_data[0].shape == xdata.shape:
            for name, row in zip(frame_names, ydata):
                if name not in self.frame_names:
                    self.plot_data[1] = np.vstack(
                        (self.plot_data[1], row[np.newaxis, :]))
                    self.frame_names.append(name)
        else:
            self.plot_data = [xdata, ydata]
            self.frame_names = list(frame_names)

        xdata, ydata = self.plot_data
        self.plot_data_range = [
            [xdata.min(), xdata.max()],
            [ydata.min(), ydata.max()],
        ]

        self.plot.setLabel('bottom', xlabel, units=xunits)
        self.plot.setLabel('left', 'Intensity')

        self.update_plot_view()

    def _update_image_viewer(self):
        """Render raw image data in viewer mode with optional mask/threshold."""
        if len(self.idxs_2d) == 0:
            self.clear_image_view()
            return
        # O8: snapshot under data_lock so a concurrent writer can't
        # evict the key between the lookup and the read.
        with self.data_lock:
            frame_2d = self.data_2d.get(self.idxs_2d[0])
        if frame_2d is None:
            self.clear_image_view()
            return
        # The detector array may not be hydrated yet (the loader publishes a
        # thumbnail-backed preview first, then the full frame), or it may have
        # been evicted from the bounded raw cache.  Fall back to the stored
        # thumbnail rather than crashing on ``np.asarray(None)`` (blank panel).
        raw = frame_2d.get('map_raw')
        if raw is None:
            raw = frame_2d.get('thumbnail')
        if raw is None:
            self.clear_image_view()
            return
        data = self._sanitize_display_image(raw)

        # Dead/hot-pixel sentinels (e.g. uint32 max 4294967295 from Eiger
        # masters) blow out the autoscale and render the frame blank/dark.
        # Mask them to NaN so the real dynamic range drives the colormap.
        if data.size == 0 or not np.isfinite(data).any():
            self._set_viewer_title(self.idxs_2d)
            self.clear_image_view()
            return

        # Apply threshold from wrangler if enabled
        w = self._wrangler
        if w is not None:
            if getattr(w, 'apply_threshold', False):
                lo = getattr(w, 'threshold_min', 0)
                hi = getattr(w, 'threshold_max', 0)
                if hi > lo:
                    data[data < lo] = np.nan
                    data[data > hi] = np.nan

            # Apply mask file if shape matches
            mask_file = getattr(w, 'mask_file', '')
            if mask_file and os.path.exists(mask_file):
                try:
                    from ssrl_xrd_tools.io.image import read_image as _read_img
                    mask = np.asarray(_read_img(mask_file), dtype=bool)
                    if mask.shape == data.shape:
                        data[mask] = np.nan
                except Exception:
                    logger.debug("Failed to load or apply mask file %s", mask_file, exc_info=True)

        if data.size == 0 or not np.isfinite(data).any():
            self._set_viewer_title(self.idxs_2d)
            self.clear_image_view()
            return

        data = data.T[:, ::-1]
        rect = get_rect(np.arange(data.shape[0]), np.arange(data.shape[1]))
        self.image_data = (data, rect)
        self._set_viewer_title(self.idxs_2d)
        self.update_image_view()

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
        frame = self.data_1d.get(int(idx))
        if frame is not None:
            thumb = getattr(frame, 'thumbnail', None)

        # Fall back to 2D data dict
        if thumb is None and int(idx) in self.data_2d:
            d2 = self.data_2d[int(idx)]
            thumb = d2.get('map_raw')

        if thumb is None or (hasattr(thumb, 'size') and thumb.size == 0):
            if show_message:
                QtWidgets.QMessageBox.information(
                    self, 'No Image',
                    f'No image data available for frame {idx}.')
            return

        # Display with correct orientation: transpose and flip vertically
        self._image_preview_widget.setImage(
            thumb.T[:, ::-1],
            autoRange=True,
            autoLevels=True,
        )
        self._image_preview_dialog.setWindowTitle(
            f'Raw Image Preview \u2014 Frame {idx}')
