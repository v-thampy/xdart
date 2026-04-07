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

    """Widget for displaying 2D image data and 1D plots from EwaldSphere
    objects.

    Inherits data-access helpers from ``DisplayDataMixin`` and
    plot-rendering helpers from ``DisplayPlotMixin``.

    attributes:
        curve1: pyqtgraph pen, overall data line
        curve2: pyqtgraph pen, individual arch data line
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
        sphere: EwaldSphere, unused.
        arch: EwaldArch, currently loaded arch object
        arch_ids: List of EwaldArch indices currently loaded
        arches: Dictionary of currently loaded EwaldArches
        data_1d: Dictionary object holding all 1D data in memory
        data_2d: Dictionary object holding all 2D data in memory
        ui: Ui_Form from qtdesigner

    methods:
        get_arches_map_raw: Gets averaged 2D raw data from arches
        get_sphere_map_raw: Gets averaged (and normalized) 2D raw data for all images
        get_arches_int_2d: Gets averaged 2D rebinned data from arches
        get_sphere_int_2d: Gets overall 2D data for the sphere
        update: Updates the displayed image and plot
        update_image: Updates image data based on selections
        update_plot: Updates plot data based on selections
    """

    def __init__(self, sphere, arch, arch_ids, arches, data_1d, data_2d,
                 parent=None, data_lock=None):
        super().__init__(parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        # Shared reentrant lock guarding data_1d / data_2d.  When created
        # standalone (tests, viewer mode) fall back to a private lock.
        self.data_lock = data_lock if data_lock is not None else threading.RLock()
        self._init_data_objects(sphere, arch, arch_ids, arches, data_1d, data_2d)
        self._init_display_panes()
        self._init_plot_panes()
        self._connect_signals()
        self._init_controls()

    # ── Initialization helpers ─────────────────────────────────────

    def _init_data_objects(self, sphere, arch, arch_ids, arches, data_1d, data_2d):
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
        self.wf_step = 1

        # Data object references
        self.sphere = sphere
        self.arch = arch
        self.arch_ids = arch_ids
        self.arches = arches
        self.arch_names = []
        self.data_1d = data_1d
        self.data_2d = data_2d
        self.bkg_1d = 0.
        self.bkg_2d = 0.
        self.bkg_map_raw = 0.

        # Viewer mode: None (normal), 'image', or 'xye'
        self.viewer_mode = None
        self._wrangler = None

        # Arch index tracking
        self.idxs = []
        self.idxs_1d = []
        self.idxs_2d = []
        self.overall = False
        self.get_idxs()

        # Plotting variables
        self.normChannel = None
        self.overlay = None
        self._last_plot_unit = -1
        self._plot_axis_info = []  # populated by set_axes()
        self._was_skip_2d = False  # track 1D-only state for transitions

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
        self.ui.shareAxis.stateChanged.connect(self.update)
        self.ui.update2D.stateChanged.connect(self.enable_2D_buttons)

        # 2D image controls
        self.ui.imageUnit.activated.connect(self.update_binned)
        self.ui.imageUnit.activated.connect(self._update_slice_range)

        # 1D plot controls
        self.ui.plotMethod.currentIndexChanged.connect(self._on_plotMethod_changed)
        self.ui.yOffset.valueChanged.connect(self.update_plot_view)
        self.ui.plotUnit.activated.connect(self._on_plotUnit_changed)
        self.ui.plotUnit.activated.connect(self.update_plot)
        self.ui.showLegend.stateChanged.connect(self.update_legend)
        self.ui.slice.stateChanged.connect(self.update_plot)
        self.ui.slice.stateChanged.connect(self._update_slice_range)
        self.ui.slice_center.valueChanged.connect(self.update_plot_range)
        self.ui.slice_width.valueChanged.connect(self.update_plot_range)
        self.ui.wf_options.clicked.connect(self.popup_wf_options)

        # Action buttons
        self.ui.clear_1D.clicked.connect(self.clear_1D)
        self.ui.save_2D.clicked.connect(self.save_image)
        self.ui.save_1D.clicked.connect(self.save_1D)

    def _init_controls(self):
        """Initialize image units, waterfall options, and preview button."""
        self.set_axes()
        self._set_slice_range(initialize=True)

        # Waterfall options popup widgets
        self.wf_dialog = QDialog()
        self.wf_yaxis_widget = QCombo()
        self.wf_start_widget = QtWidgets.QDoubleSpinBox()
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
        """ Return selected arch indices.

        Thread-safety: snapshots of data_1d / data_2d keys are taken under
        ``data_lock`` to avoid racing with integrator / file-handler worker
        threads that mutate these dicts concurrently.
        """
        self.idxs, self.idxs_1d, self.idxs_2d = [], [], []
        if len(self.arch_ids) == 0 or self.arch_ids[0] == 'No data':
            return

        with self.data_lock:
            with self.sphere.sphere_lock:
                if len(self.arch_ids) == len(self.sphere.arches.index) > 1:
                    self.overall = True
                    self.idxs = sorted(np.asarray(self.sphere.arches.index, dtype=int))
                else:
                    self.overall = False
                    self.idxs = sorted(self.arch_ids)

            try:
                self.idxs = list(np.asarray(self.idxs, dtype=int))
            except ValueError:
                return
            # Snapshot current dict keys while the lock is held, then release
            # before doing list-comprehension work.
            data_1d_keys = set(self.data_1d.keys())
            data_2d_keys = set(self.data_2d.keys())

        self.idxs_1d = [int(idx) for idx in self.idxs if idx in data_1d_keys]
        self.idxs_2d = [int(idx) for idx in self.idxs if idx in data_2d_keys]

    def update_plot_range(self):
        if self.ui.slice.isChecked():
            self.update_plot()

    # ── Update orchestration ──────────────────────────────────────

    def _updated(self):
        """Check if there is data to update
        """
        # In viewer mode, bypass the sphere.name check — no HDF5 scan is loaded
        if self.viewer_mode is not None:
            if len(self.arch_ids) == 0:
                return False
            if self.viewer_mode == 'image' and len(self.data_2d) == 0:
                return False
            if self.viewer_mode == 'xye' and len(self.data_1d) == 0:
                return False
            return True

        if (len(self.arch_ids) == 0) or (self.sphere.name == 'null_main'):
            return False
        if (len(self.data_1d) == 0) or (len(self.idxs_1d) == 0):
            return False

        return True

    def update(self):
        """Updates image and plot frames based on toolbar options
        """
        self.get_idxs()

        if not self._updated():
            return True

        # ── Viewer mode: simplified rendering ────────────────────────
        if self.viewer_mode == 'image':
            try:
                self._update_image_viewer()
            except Exception:
                logger.debug('Image viewer update failed', exc_info=True)
            return True
        if self.viewer_mode == 'xye':
            try:
                self._update_xye_viewer()
            except Exception:
                logger.debug('XYE viewer update failed', exc_info=True)
            return True

        # ── Normal mode ──────────────────────────────────────────────
        if self.ui.shareAxis.isChecked() and (self.ui.imageUnit.currentIndex() < 2):
            self.ui.plotUnit.setCurrentIndex(self.ui.imageUnit.currentIndex())
            self.ui.plotUnit.setEnabled(False)
            self.plot.setXLink(self.binned_widget.image_plot)
        else:
            self.plot.setXLink(None)
            self.ui.plotUnit.setEnabled(True)

        self._apply_1d_only_visibility()

        try:
            self.update_plot()
        except TypeError:
            return False

        if self.ui.update2D.isChecked():
            try:
                self.update_image()
            except TypeError:
                return False
            try:
                self.update_binned()
            except TypeError:
                return False

        # Apply label to 2D view
        if self.ui.update2D.isChecked():
            self.update_2d_label()

        # Update the image preview popup if it is open
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

        if self.ui.update2D.isChecked():
            self.update_image_view()
            self.update_binned_view()
            self.update_2d_label()
        self.update_plot_view()

    # ── 1D-only visibility ────────────────────────────────────────

    def _apply_1d_only_visibility(self):
        """Show or hide 2D panes based on sphere.skip_2d.

        In 1D-only mode (skip_2d), collapse the 2D image panels and
        image toolbar while keeping the top toolbar (Norm Channel, Scale,
        Set Bkg, etc.) visible.  Also removes pure-2D entries (like χ)
        from the plotUnit combo so the user cannot select them.
        """
        # In viewer mode, set_viewer_display_mode() controls panels
        if self.viewer_mode is not None:
            return
        skip = getattr(self.sphere, 'skip_2d', False)
        if skip:
            # Hide 2D panels and image toolbar, keep frame_top visible
            self.ui.twoDWindow.setMaximumHeight(0)
            self.ui.twoDWindow.setMinimumHeight(0)
            self.ui.imageToolbar.setMaximumHeight(0)
            self.ui.imageToolbar.setMinimumHeight(0)
            # Shrink imageWindow to just the top toolbar height
            self.ui.imageWindow.setMinimumHeight(35)
            self.ui.imageWindow.setMaximumHeight(35)
            self.ui.update2D.setChecked(False)
            self.ui.update2D.setEnabled(False)
            if self.ui.slice.isChecked():
                self.ui.slice.setChecked(False)
            self.ui.slice.setEnabled(False)
            self.ui.slice_center.setEnabled(False)
            self.ui.slice_width.setEnabled(False)
            self.ui.imageUnit.setEnabled(False)

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
            # Restore 2D panels and image toolbar
            self.ui.twoDWindow.setMinimumHeight(0)
            self.ui.twoDWindow.setMaximumHeight(16777215)
            self.ui.imageToolbar.setMinimumHeight(40)
            self.ui.imageToolbar.setMaximumHeight(40)
            self.ui.imageWindow.setMinimumHeight(200)
            self.ui.imageWindow.setMaximumHeight(16777215)
            self.ui.update2D.setEnabled(True)
            self.ui.update2D.setChecked(True)
            self.ui.imageUnit.setEnabled(True)
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

        if self.sphere.gi:
            gi_mode_1d = self.sphere.bai_1d_args.get('gi_mode_1d', 'q_total')
            gi_mode_2d = self.sphere.bai_2d_args.get('gi_mode_2d', 'qip_qoop')
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
        skip_2d = getattr(self.sphere, 'skip_2d', False)
        # Slicing requires 2D data and the axis must come from 2D
        can_slice = (not skip_2d) and info['source'] in ('2d', '1d_2d')

        # Enable/disable slice UI
        self.ui.slice.setEnabled(can_slice)
        self.ui.slice_center.setEnabled(can_slice)
        self.ui.slice_width.setEnabled(can_slice)
        if not can_slice:
            self.ui.slice.setChecked(False)
            self.clear_slice_overlay()

        # Update slice range label
        self._set_slice_range()

    def enable_2D_buttons(self):
        """Placeholder: disable 2D-related buttons when update2D is unchecked.

        Currently a no-op; 2D button visibility is managed elsewhere.
        """

    # ── 2D image rendering ────────────────────────────────────────

    def update_image(self):
        """Updates image plotted in image frame.

        Applies the detector-level mask and global mask to the raw image.
        If the data is a downsampled thumbnail (mask already baked in as
        NaN), the mask application is skipped because the flat indices
        would not match the thumbnail's smaller shape.
        """
        mask = None
        if self.overall and len(self.arch_ids) > 1:
            data = self.get_sphere_map_raw()
        else:
            data = self.get_arches_map_raw()
            if data is None:
                return

            # Apply Mask
            arch_2d = self.data_2d[self.idxs_2d[0]]
            mask = arch_2d['mask']
        data = np.asarray(data, dtype=float)

        # Apply detector + global mask.
        global_mask = self.sphere.global_mask if self.sphere.global_mask is not None else []
        mask = mask if mask is not None else []
        mask = np.asarray(np.unique(np.append(mask, global_mask)), dtype=int)
        if len(mask) > 0 and mask.max() < data.size:
            mask = np.unravel_index(mask, data.shape)
            data[mask] = np.nan

        # Subtract background
        data -= self.bkg_map_raw

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

        # Always aggregate from per-arch data_2d. The legacy
        # `self.overall → get_sphere_int_2d()` shortcut returned a 1×1
        # zero array when `sphere.bai_2d` wasn't populated (common with
        # NeXus files that store per-arch results only), which made the
        # 2D pane go blank as soon as all arches were selected.
        intensity, xdata, ydata = self.get_arches_int_2d()
        if intensity is None and self.overall and len(self.arch_ids) > 1:
            # Fall back to the precomputed sphere total if available.
            intensity, xdata, ydata = self.get_sphere_int_2d()

        if intensity is None:
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
        if self.sphere.gi:
            gi_mode_2d = self.sphere.bai_2d_args.get('gi_mode_2d', 'qip_qoop')
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
        label = self.sphere.name
        if len(label) > 40:
            label = f'{label[:10]}.....{label[-30:]}'

        if (self.overall or self.sphere.single_img) and (len(self.arch_ids) > 1):
            self.ui.labelCurrent.setText(label)
        elif self.sphere.series_average:
            self.ui.labelCurrent.setText(label)
        elif len(self.arch_ids) > 1:
            self.ui.labelCurrent.setText(f'{label} [Average]')
        else:
            self.ui.labelCurrent.setText(f'{label}_{self.arch_ids[0]}')

    # ── Normalization / background handlers ───────────────────────

    def normUpdate(self):
        """Update plots if norm channel exists"""
        self.normChannel = self.get_normChannel()
        if self.normChannel and (self.sphere.scan_data[self.normChannel].sum() == 0.):
            self.normChannel = None
        # Clear stale plot_data so update_plot() rebuilds all overlay curves
        self.plot_data = [np.zeros(0), np.zeros(0)]
        self.arch_names = []
        self.update()

    def setBkg(self):
        """Sets selected points as background.
        If background is already selected, it unsets it"""
        if (len(self.arch_ids) == 0) or (len(self.idxs) == 0):
            return

        if self.ui.setBkg.text() == 'Set Bkg':
            idxs = self.arch_ids
            if self.overall:
                idxs = sorted(list(self.sphere.arches.index))

            self.bkg_1d, _ = self.get_arches_int_1d(idxs, rv='average')
            self.bkg_2d, _, _ = self.get_arches_int_2d(idxs)
            self.bkg_map_raw = self.get_arches_map_raw(idxs)
            if self.bkg_map_raw is None:
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

    def set_viewer_display_mode(self, mode):
        """Configure display panels for viewer modes.

        Args:
            mode: 'image' — show only the raw 2D image panel,
                  'xye'   — show only the 1D plot panel,
                  None    — restore normal layout.
        """
        self.viewer_mode = mode
        if mode == 'image':
            # Show 2D image panel, collapse 1D plot panel
            self.ui.imageWindow.setMinimumHeight(200)
            self.ui.imageWindow.setMaximumHeight(16777215)
            self.ui.plotWindow.setMinimumHeight(0)
            self.ui.plotWindow.setMaximumHeight(0)
            # Hide binned frame — only raw image relevant
            self.ui.binnedFrame.setMaximumWidth(0)
            self.ui.binnedFrame.setMinimumWidth(0)

            # Disable all controls not relevant in image viewer mode
            self.ui.normChannel.setEnabled(False)
            self.ui.setBkg.setEnabled(False)
            self.ui.update2D.setChecked(False)
            self.ui.update2D.setEnabled(False)
            self.ui.shareAxis.setEnabled(False)
            self.ui.imageUnit.setEnabled(False)
            self.ui.slice.setEnabled(False)
            self._showImageBtn.setVisible(False)

            # Keep these active: scale (Linear/Log), cmap, save_2D
            self.ui.scale.setEnabled(True)
            self.ui.cmap.setEnabled(True)
            self.ui.save_2D.setEnabled(True)
        elif mode == 'xye':
            # Collapse 2D image panel, show 1D plot panel
            self.ui.imageWindow.setMinimumHeight(0)
            self.ui.imageWindow.setMaximumHeight(0)
            self.ui.plotWindow.setMinimumHeight(200)
            self.ui.plotWindow.setMaximumHeight(16777215)
            self.ui.update2D.setChecked(False)
            self.ui.update2D.setEnabled(False)
            # Disable controls not applicable to raw XYE data
            self.ui.slice.setEnabled(False)
            self.ui.slice_center.setEnabled(False)
            self.ui.slice_width.setEnabled(False)
            self.ui.plotUnit.setEnabled(False)
            self.ui.shareAxis.setEnabled(False)
            # Keep plot method, legend, clear, save functional
            self.ui.plotMethod.setEnabled(True)
            self.ui.yOffset.setEnabled(True)
            self.ui.showLegend.setEnabled(True)
            self.ui.clear_1D.setEnabled(True)
            self.ui.save_1D.setEnabled(True)
            self.ui.wf_options.setEnabled(True)
        else:
            # Normal mode — restore both panels
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
            self.ui.update2D.setEnabled(True)
            self.ui.update2D.setChecked(True)
            self.ui.shareAxis.setEnabled(True)
            self.ui.imageUnit.setEnabled(True)
            self.ui.scale.setEnabled(True)
            self.ui.cmap.setEnabled(True)
            self.ui.save_2D.setEnabled(True)
            self.ui.plotUnit.setEnabled(True)
            self.ui.plotMethod.setEnabled(True)
            # Slice enable/disable depends on which axis is selected
            self._on_plotUnit_changed()
            self.ui.showLegend.setEnabled(True)
            self.ui.clear_1D.setEnabled(True)
            self.ui.save_1D.setEnabled(True)

    def _update_xye_viewer(self):
        """Render 1D line data in XYE viewer mode (no sphere/integration).

        Builds plot_data and arch_names from the loaded XYE data,
        then delegates to the standard update_plot_view() so that
        Single/Overlay/Waterfall/Sum/Average all work.
        """
        if len(self.idxs_1d) == 0:
            return

        # Determine axis label from first arch
        first_arch = self.data_1d[self.idxs_1d[0]]
        unit = getattr(first_arch.int_1d, 'unit', '2th_deg')
        if 'q' in unit.lower():
            xlabel = u'Q (\u212b\u207b\u00b9)'
            xunits = u'\u212b\u207b\u00b9'
        else:
            xlabel = u'2\u03b8'
            xunits = u'\u00b0'

        # Build xdata and ydata arrays from all selected indices.
        ref_x = np.asarray(first_arch.int_1d.radial, dtype=float)
        arch_names = []
        rows = []
        for idx in self.idxs_1d:
            arch = self.data_1d[idx]
            int_1d = arch.int_1d
            if int_1d is None:
                continue
            fname = arch.scan_info.get('source_file', f'xye_{idx}')
            arch_names.append(os.path.basename(fname))
            xdata = np.asarray(int_1d.radial, dtype=float)
            ydata = np.asarray(int_1d.intensity, dtype=float)
            if xdata.shape != ref_x.shape or not np.allclose(xdata, ref_x):
                ydata = np.interp(ref_x, xdata, ydata)
            rows.append(ydata)

        if not rows:
            return

        xdata = ref_x
        ydata = np.vstack(rows) if len(rows) > 1 else rows[0][np.newaxis, :]

        # In Overlay/Waterfall: accumulate, skip duplicates.
        current_method = self.ui.plotMethod.currentText()
        if current_method in ('Overlay', 'Waterfall') and \
                len(self.plot_data[0]) > 0 and \
                self.plot_data[0].shape == xdata.shape:
            for name, row in zip(arch_names, ydata):
                if name not in self.arch_names:
                    self.plot_data[1] = np.vstack(
                        (self.plot_data[1], row[np.newaxis, :]))
                    self.arch_names.append(name)
        else:
            self.plot_data = [xdata, ydata]
            self.arch_names = list(arch_names)

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
            return
        arch_2d = self.data_2d[self.idxs_2d[0]]
        data = np.asarray(arch_2d['map_raw'], dtype=float)

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

        data = data.T[:, ::-1]
        rect = get_rect(np.arange(data.shape[0]), np.arange(data.shape[1]))
        self.image_data = (data, rect)
        self.update_image_view()

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
        arch = self.data_1d.get(int(idx))
        if arch is not None:
            thumb = getattr(arch, 'thumbnail', None)

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
