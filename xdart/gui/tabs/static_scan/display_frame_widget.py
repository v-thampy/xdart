# -*- coding: utf-8 -*-
"""
@author: thampy
"""

# Standard library imports
import logging
import os
import re
import time
import copy

logger = logging.getLogger(__name__)

# Other imports
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from pathlib import Path

# Qt imports
from PySide6.QtCore import Qt as pyQt
import pyqtgraph as pg
from pyqtgraph import Qt
from pyqtgraph.Qt import QtWidgets
import pyqtgraph.exporters
from pyqtgraph import ROI

# This module imports
from .ui.displayFrameUI import Ui_Form
from ...gui_utils import RectViewBox, get_rect
import xdart.utils as ut
from ...widgets import pgImageWidget, pmeshImageWidget
from xdart.utils import split_file_name
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


class displayFrameWidget(Qt.QtWidgets.QWidget):
    """Widget for displaying 2D image data and 1D plots from EwaldSphere
    objects.

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

    def __init__(self, sphere, arch, arch_ids, arches, data_1d, data_2d, parent=None):
        super().__init__(parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
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

    def get_idxs(self):
        """ Return selected arch indices
        """
        self.idxs, self.idxs_1d, self.idxs_2d = [], [], []
        if len(self.arch_ids) == 0 or self.arch_ids[0] == 'No data':
            return

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
        self.idxs_1d = [int(idx) for idx in self.idxs if idx in self.data_1d.keys()]
        self.idxs_2d = [int(idx) for idx in self.idxs if idx in self.data_2d.keys()]

    def update_plot_range(self):
        if self.ui.slice.isChecked():
            self.update_plot()

    def setup_1d_layout(self):
        """Setup the layout for 1D plot
        """
        self.wf_widget.setParent(None)
        self.plot_layout.addWidget(self.plot_win)

        self.ui.wf_options.setEnabled(False)
        if len(self.plot_data[1]) > 1:
            self.ui.wf_options.setEnabled(True)
            self.wf_yaxis_widget.setEnabled(False)

    def setup_wf_widget(self):
        self.plot_layout.addWidget(self.wf_widget)

        # Waterfall Plot setup
        if self.plotMethod == 'Waterfall':
            self.plot_win.setParent(None)
            self.plot_layout.addWidget(self.wf_widget)
        else:
            self.wf_widget.setParent(None)
            self.plot_layout.addWidget(self.plot_win)

    def setup_wf_layout(self):
        """Setup the layout for WF plot
        """
        self.plot_win.setParent(None)
        self.plot_layout.addWidget(self.wf_widget)

        self.ui.wf_options.setEnabled(True)
        self.wf_yaxis_widget.setEnabled(True)

    def popup_wf_options(self):
        """
        Popup Qt Window to select options for Waterfall Plot
        Options include Y-axis unit and number of points to skip
        """
        if self.wf_dialog.layout() is None:
            self.setup_wf_options_widget()

        self.wf_dialog.show()

    def setup_wf_options_widget(self):
        """
        Setup y-axis option for Waterfall plot
        Setup first image and step size for wf and overlay plots
        """
        layout = QtWidgets.QGridLayout()
        self.wf_dialog.setLayout(layout)

        layout.addWidget(QtWidgets.QLabel('Y-Axis'), 0, 0)
        layout.addWidget(QtWidgets.QLabel('Start'), 0, 1)
        layout.addWidget(QtWidgets.QLabel('Step'), 0, 2)

        layout.addWidget(self.wf_yaxis_widget, 1, 0)
        layout.addWidget(self.wf_start_widget, 1, 1)
        layout.addWidget(self.wf_step_widget, 1, 2)

        layout.addWidget(self.wf_accept_button, 2, 1)
        layout.addWidget(self.wf_cancel_button, 2, 2)

        arch = self.data_1d[self.idxs_1d[0]]
        counters = list(arch.scan_info.keys())
        counters = ['Frame #', 'Time (s)', 'Time (minutes)'] + counters
        self.wf_yaxis_widget.addItems(counters)

        self.wf_start_widget.setDecimals(0)
        self.wf_start_widget.setRange(1, 1000)

        self.wf_step_widget.setDecimals(0)
        self.wf_step_widget.setRange(1, 100)

        self.wf_accept_button.clicked.connect(self.get_wf_option)
        self.wf_cancel_button.clicked.connect(self.close_wf_popup)

    def get_wf_option(self):
        self.wf_yaxis = self.wf_yaxis_widget.currentText()

        self.wf_start = int(self.wf_start_widget.value()) - 1
        self.wf_step = int(self.wf_step_widget.value())

        self.close_wf_popup()
        self.update_plot_view()

    def close_wf_popup(self):
        self.wf_dialog.close()

    def enable_2D_buttons(self):
        """Disable buttons if update 2D is unchecked"""
        pass

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
            # 2θ is a unit conversion of Q, mirrors Q's source and slicing
            if gi_mode_1d == 'q_total':
                tth_label = f"2{Th} ({Deg})"
                if radial_label == label_1d:
                    # Q matched 2D radial → 2θ also gets '1d_2d'
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
            # Add radial axis of 2D if not already covered by the 1D entry
            if radial_label != label_1d:
                self.ui.plotUnit.addItem(_translate("Form", radial_label))
                self._plot_axis_info.append({
                    'source': '2d', 'slice_axis': azimuthal_label,
                    'axis': 'radial',
                })

            # Add azimuthal axis of 2D
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
        # For files with different x-grids, interpolate onto the first
        # file's grid so overlay/sum/average work correctly.
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
        # In Single/Sum/Average: always replace with current selection.
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
        # Mask indices are flat-pixel offsets into the full detector image.
        # If 'data' is a downsampled thumbnail (mask baked in as NaN) or
        # a 2D integration result, the indices won't match — skip masking.
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
        # self.image_widget.setImage(data.T[:, ::-1], scale=self.scale, cmap=self.cmap)
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

        # if 'Overall' in self.arch_ids:
        if self.overall and len(self.arch_ids) > 1:
            intensity, xdata, ydata = self.get_sphere_int_2d()
        else:
            intensity, xdata, ydata = self.get_arches_int_2d()

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

        # intensity is (npt_rad, npt_azim) from ssrl_xrd_tools.
        # pyqtgraph setImage expects (x, y) = (radial, azimuthal).
        # The rect (from get_rect) maps pixel indices to the correct
        # axis values, so no manual flip is needed.
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

    def update_plot(self):
        """Updates data in plot frame
        """
        if (self.sphere.name == 'null_main') or (len(self.arch_ids) == 0):
            data = (np.arange(100), np.arange(100))
            return data

        # Get 1D data for all arches
        ydata, xdata = self.get_arches_int_1d()

        # 2D-derived axes require 2D data; fall back gracefully if unavailable
        if xdata is None or ydata is None:
            _idx = self.ui.plotUnit.currentIndex()
            _info = (self._plot_axis_info[_idx]
                     if hasattr(self, '_plot_axis_info')
                        and 0 <= _idx < len(self._plot_axis_info)
                     else None)
            needs_2d = ((_info and _info['source'] in ('2d', '1d_2d'))
                        or self.ui.slice.isChecked())
            if needs_2d and getattr(self.sphere, 'skip_2d', False):
                try:
                    self.window().statusBar().showMessage(
                        "Chi slicing requires 2D integration (1D Only is enabled).", 4000)
                except Exception:
                    logger.debug("Failed to show status bar message about chi slicing", exc_info=True)
            # Fall back: disable slice, retry with plain 1D
            self.ui.slice.setChecked(False)
            ydata, xdata = self.get_arches_int_1d()
            if xdata is None or ydata is None:
                return

        if self.sphere.series_average:
            arch_names = [self.sphere.name]
        else:
            arch_names = [f'{self.sphere.name}_{i}' for i in self.idxs]

        # When slicing is active, include slice parameters in arch names
        # so the same image with different slice ranges can be overlaid.
        if self.ui.slice.isEnabled() and self.ui.slice.isChecked():
            center = self.ui.slice_center.value()
            width = self.ui.slice_width.value()
            suffix = f' [{center:.1f}\u00b1{width:.1f}]'
            arch_names = [n + suffix for n in arch_names]

        # Subtract background
        if self.bkg_1d is not None:
            ydata -= self.bkg_1d
        if ydata.ndim == 1:
            ydata = ydata[np.newaxis, :]

        current_plot_unit = self.ui.plotUnit.currentIndex()
        unit_changed = current_plot_unit != self._last_plot_unit
        self._last_plot_unit = current_plot_unit

        # In Overlay/Waterfall: accumulate new arches, skip duplicates.
        # In Single/Sum/Average: always replace with current selection.
        current_method = self.ui.plotMethod.currentText()
        accumulate = (current_method in ('Overlay', 'Waterfall')
                      and (not unit_changed)
                      and len(self.plot_data[0]) > 0)

        if accumulate:
            for arch_name, row in zip(arch_names, ydata):
                if arch_name not in self.arch_names:
                    old_x = self.plot_data[0]
                    if old_x.shape == xdata.shape and np.allclose(old_x, xdata):
                        # Same grid — just append
                        self.plot_data[1] = np.vstack((self.plot_data[1], row))
                    else:
                        # Different grid — merge and interpolate.
                        # np.interp within each curve's original range,
                        # NaN outside (no extrapolation).
                        merged_x = np.union1d(old_x, xdata)
                        merged_x.sort()

                        def _reinterp(src_x, src_y, dst_x):
                            """Interpolate src onto dst, NaN outside range."""
                            out = np.interp(dst_x, src_x, src_y)
                            out[dst_x < src_x[0]] = np.nan
                            out[dst_x > src_x[-1]] = np.nan
                            return out

                        old_y = self.plot_data[1]
                        if old_y.ndim == 1:
                            old_y = old_y[np.newaxis, :]
                        new_old = np.array([_reinterp(old_x, r, merged_x)
                                            for r in old_y])
                        new_row = _reinterp(xdata, row, merged_x)
                        self.plot_data = [merged_x,
                                          np.vstack((new_old, new_row))]
                    self.arch_names.append(arch_name)
        else:
            # Fresh start: Single/Sum/Average, unit changed, or no existing data
            self.plot_data = [xdata, ydata]
            self.arch_names = list(arch_names)

        xdata, ydata = self.plot_data
        if xdata.size == 0 or ydata.size == 0:
            return
        self.plot_data_range = [
            [np.nanmin(xdata), np.nanmax(xdata)],
            [np.nanmin(ydata), np.nanmax(ydata)],
        ]

        self.update_plot_view()

        # self.save_1D(auto=True)

    def _on_plotMethod_changed(self):
        """Handle plotMethod combo box changes.

        Switching away from Overlay/Waterfall requires a full
        update_plot() so that plot_data is rebuilt from the current
        selection instead of carrying forward accumulated curves.
        """
        new_method = self.ui.plotMethod.currentText()
        if new_method not in ('Overlay', 'Waterfall'):
            # Reset accumulated data — rebuild from current selection
            self.plot_data = [np.array([]), np.array([])]
            self.arch_names = []
            self.update_plot()
        else:
            self.update_plot_view()

    def update_plot_view(self):
        """Updates 1D view of data in plot frame
        """
        if (len(self.arch_ids) == 0) or len(self.data_1d) == 0:
            return

        # Clear curves
        [curve.clear() for curve in self.curves]
        self.curves.clear()

        self.plotMethod = self.ui.plotMethod.currentText()

        self.ui.yOffset.setEnabled(False)
        if (self.plotMethod in ['Overlay', 'Single']) and (len(self.arch_names) > 1):
            self.ui.yOffset.setEnabled(True)

        n_curves = len(self.plot_data[1])
        # Only switch to WF plot if more than three curves. Definitely switch if more than 15!
        if (self.plotMethod == 'Waterfall' and n_curves > 3) or n_curves > 15:
            self.update_wf()
        else:
            self.update_1d_view()

    def update_1d_view(self):
        """Updates data in 1D plot Frame
        """
        self.setup_1d_layout()

        xdata_, ydata_ = self.plot_data
        s_xdata, ydata = xdata_.copy(), ydata_.copy()

        int_label = 'I'
        if self.normChannel:
            int_label = f'I / {self.normChannel}'

        self.plot.getAxis("left").setLogMode(False)
        self.plot.getAxis("bottom").setLogMode(False)
        ylabel = f'{int_label} (a.u.)'
        if self.scale == 'Log':
            if ydata.size == 0:
                return
            if ydata.min() < 1:
                ydata -= (ydata.min() - 1.)
            ydata = np.log10(ydata)
            self.plot.getAxis("left").setLogMode(True)
            ylabel = f'Log {int_label}(a.u.)'
        elif self.scale == 'Log-Log':
            if ydata.min() < 1:
                ydata -= (ydata.min() - 1.)
            ydata = np.log10(ydata)
            self.plot.getAxis("left").setLogMode(True)
            ylabel = f'Log {int_label}(a.u.)'

            s_xdata = np.log10(s_xdata)
            self.plot.getAxis("bottom").setLogMode(True)
        elif self.scale == 'Sqrt':
            if ydata.min() < 0.:
                ydata_ = np.sqrt(np.abs(ydata))
                ydata_[ydata < 0] *= -1
                ydata = ydata_
            else:
                ydata = np.sqrt(ydata)
            ylabel = f'<math>&radic;</math>{int_label} (a.u.)'

        if ((self.plotMethod in ['Overlay', 'Waterfall']) or
                ((self.plotMethod == 'Single') and (ydata.shape[0] > 1))):
            ydata = ydata[self.wf_start::self.wf_step]
            self.setup_curves()

            offset = self.ui.yOffset.value()
            y_offset = offset / 100 * (self.plot_data_range[1][1] - self.plot_data_range[1][0])
            for nn, (curve, s_ydata) in enumerate(zip(self.curves, ydata)):
                curve.setData(s_xdata, s_ydata + y_offset*nn)

        else:
            self.setup_curves()
            s_ydata = ydata
            if self.plotMethod == 'Average':
                s_ydata = np.nanmean(s_ydata, 0)
            elif self.plotMethod == 'Sum':
                s_ydata = np.nansum(s_ydata, 0)

            self.curves[0].setData(s_xdata, s_ydata.squeeze())

        # Apply labels to plot — parse from plotUnit combo text
        plot_text = self.ui.plotUnit.currentText()
        # Extract label and units from "Label (Units)" format
        m = re.match(r'^(.+?)\s*\((.+)\)$', plot_text)
        if m:
            _xl, _xu = m.group(1).strip(), m.group(2).strip()
        else:
            _xl, _xu = plot_text, ''
        self.plot.setLabel("bottom", _xl, units=_xu)
        self.plot.setLabel("left", ylabel)

        return s_xdata, s_ydata

    def update_wf(self):
        """Updates data in 1D plot Frame
        """
        self.setup_wf_layout()

        xdata_, data_ = self.plot_data
        s_xdata, data = xdata_.copy(), data_.copy()
        data = data[self.wf_start::self.wf_step, :]

        # Set YAxis Unit
        if self.wf_yaxis == 'Frame #':
            s_ydata = np.asarray(np.arange(data.shape[0]) + self.wf_start + 1, dtype=float)
        else:
            # TODO: Fix below for more WF options
            try:
                if self.wf_yaxis == 'Time (s)':
                    s_ydata = np.asarray([self.data_1d[idx].scan_info['epoch'] for idx in self.idxs])
                    s_ydata -= s_ydata.min()
                elif self.wf_yaxis == 'Time (minutes)':
                    s_ydata = np.asarray([self.data_1d[idx].scan_info['epoch'] for idx in self.idxs])/60.
                    s_ydata -= s_ydata.min()
                else:
                    s_ydata = np.asarray([self.data_1d[idx].scan_info[self.wf_yaxis] for idx in self.idxs])

                s_ydata = s_ydata[self.wf_start::self.wf_step]
            except KeyError:
                logger.debug('Counter not present in metadata')

        # rect = get_rect(s_xdata, np.arange(data.shape[0]))
        rect = get_rect(s_xdata, s_ydata)

        self.wf_widget.setImage(data.T, scale=self.scale, cmap=self.cmap)
        self.wf_widget.setRect(rect)

        # Parse label from plotUnit combo text
        plot_text = self.ui.plotUnit.currentText()
        m = re.match(r'^(.+?)\s*\((.+)\)$', plot_text)
        if m:
            _xl, _xu = m.group(1).strip(), m.group(2).strip()
        else:
            _xl, _xu = plot_text, ''
        self.wf_widget.image_plot.setLabel("bottom", _xl, units=_xu)
        self.wf_widget.image_plot.setLabel("left", self.wf_yaxis)

    def get_arches_map_raw(self, idxs=None):
        """Return 2D arch data for multiple arches (averaged).

        Falls back to the stored thumbnail when full-resolution raw data
        is not available (e.g. when loading from NeXus files that only
        store integration results + thumbnails).
        """
        if idxs is None:
            idxs = self.idxs_2d

        intensity, ctr = 0., 0
        for nn, idx in enumerate(idxs):
            arch_1d = self.data_1d.get(int(idx))
            arch_2d = self.data_2d.get(int(idx), {})
            raw = arch_2d.get('map_raw')
            bg = arch_2d.get('bg_raw', 0)
            # Try thumbnail from data_2d, then fall back to data_1d
            thumb = arch_2d.get('thumbnail')
            if thumb is None and arch_1d is not None:
                thumb = getattr(arch_1d, 'thumbnail', None)
            for kk in range(3):
                try:
                    scan_info = arch_1d.scan_info if arch_1d is not None else {}
                    if raw is not None:
                        intensity += self.normalize(raw - bg, scan_info)
                    elif thumb is not None:
                        # Use thumbnail as fallback when raw isn't stored
                        intensity += self.normalize(
                            np.asarray(thumb, dtype=float), scan_info)
                    else:
                        break
                    ctr += 1
                    break
                except ValueError:
                    time.sleep(0.5)

        if ctr > 0:
            intensity /= ctr
        else:
            return None

        return np.asarray(intensity, dtype=float)

    def get_sphere_map_raw(self):
        """Returns data and QRect for data in sphere
        """
        with self.sphere.sphere_lock:
            map_raw = np.asarray(self.sphere.overall_raw, dtype=float)
            if map_raw.ndim < 2:
                self.sphere.load_from_h5(data_only=True)
                map_raw = np.asarray(self.sphere.overall_raw, dtype=float)

            norm_fac = len(self.sphere.arches.index)
            if self.normChannel:
                norm = self.sphere.scan_data[self.normChannel].sum()
                if norm > 0:
                    norm_fac = norm

            return map_raw/norm_fac

    def get_arches_int_2d(self, idxs=None):
        """Return 2D arch data for multiple arches (averaged)"""
        if idxs is None:
            idxs = self.idxs_2d

        if len(idxs) > 1:
            intensity = self.get_int_2d(self.arches['sum_int_2d'])
            xdata, ydata = self.get_xydata(self.arches['sum_int_2d'])
            return intensity, xdata, ydata


        # intensity, n_arches = 0., 0
        # for nn, idx in enumerate(idxs):
        try:
            idx = idxs[0]
        except IndexError:
            return None, None, None

        arch_1d = self.data_1d[int(idx)]
        arch_2d = self.data_2d[int(idx)]
        _gi2d = arch_2d.get('gi_2d', {})
        intensity = self.get_int_2d(arch_2d['int_2d'], arch_1d, gi_2d=_gi2d)

        if intensity.ndim != 2:
            return None, None, None

        xdata, ydata = self.get_xydata(arch_2d['int_2d'], gi_2d=_gi2d)
        return np.asarray(intensity, dtype=float), xdata, ydata

    def get_sphere_int_2d(self):
        """Returns data and QRect for data in sphere
        """
        with self.sphere.sphere_lock:
            int_2d = self.sphere.bai_2d

        if int_2d is None:
            return np.zeros((1, 1)), np.array([]), np.array([])

        intensity = self.get_int_2d(int_2d, normalize=True)

        xdata, ydata = self.get_xydata(int_2d)
        return intensity, xdata, ydata

    def get_arches_int_1d(self, idxs=None, rv='all'):
        """Return 1D data for multiple arches"""
        if idxs is None:
            idxs = self.idxs_1d

        ydata = None
        xdata = None
        for idx in idxs:
            arch_1d = self.data_1d.get(int(idx), None)
            if arch_1d is None:
                continue
            arch_2d = self.data_2d.get(int(idx), None)
            x, y = self.get_int_1d(arch_1d, arch_2d, idx)
            if x is None or y is None:
                continue
            if ydata is None:
                xdata = x
                ydata = y
            else:
                ydata = np.vstack((ydata, y))

        if ydata is None:
            return None, None

        if ydata.ndim == 2:
            if rv == 'average':
                ydata = np.nanmean(ydata, 0)
            elif rv == 'sum':
                ydata = np.nansum(ydata, 0)

        return ydata, xdata

    def get_int_2d(self, int_2d, arch_1d=None, normalize=True, gi_2d=None):
        """Returns the appropriate 2D data depending on the chosen axes.
        In GI mode, int_2d already holds the selected mode's data.
        """
        if int_2d is None:
            return np.zeros((1, 1))
        # int_2d is always the correct result (GI or standard)
        intensity_2d = int_2d.intensity
        intensity = np.asarray(intensity_2d.copy(), dtype=float)

        if normalize:
            if arch_1d is not None:
                intensity = self.normalize(intensity, arch_1d.scan_info)
            else:
                norm_fac = len(self.sphere.arches.index)
                if self.normChannel:
                    norm = self.sphere.scan_data[self.normChannel].sum()
                    if norm > 0:
                        norm_fac = norm
                intensity /= norm_fac

        return intensity

    def get_int_1d(self, arch, arch_2d, idx):
        """Returns 1D integrated data for arch.

        Uses ``self._plot_axis_info`` to determine whether the selected
        plotUnit axis comes from the 1D integration (direct readout) or
        the 2D integration (requires slicing/projection from the 2D map).
        When the axis is 2D-derived *and* slicing is enabled, only the
        selected range of the orthogonal axis is averaged.
        """
        _plot_idx = self.ui.plotUnit.currentIndex()
        info = (self._plot_axis_info[_plot_idx]
                if hasattr(self, '_plot_axis_info')
                   and 0 <= _plot_idx < len(self._plot_axis_info)
                else {'source': '1d', 'slice_axis': None, 'axis': None})

        # Pure 2D axes always need 2D data; hybrid (1d_2d) only when slicing
        _needs_2d = (info['source'] == '2d') or \
                    (info['source'] == '1d_2d' and self.ui.slice.isChecked())

        # --- Fast path: pure 1D readout (no 2D data needed) ---
        if not _needs_2d:
            int_1d = arch.int_1d
            if int_1d is None:
                return None, None
            intensity = int_1d.intensity
            ydata = self.normalize(intensity, arch.scan_info)
            xdata = self.get_xdata(arch)
            return xdata, ydata

        # --- 2D path: project from 2D map ---
        if arch_2d is None:
            return None, None

        intensity = self.get_int_2d(arch_2d['int_2d'], arch, normalize=False,
                                    gi_2d=arch_2d.get('gi_2d', {}))
        if intensity.ndim < 2:
            return None, None

        _i2d = arch_2d['int_2d']
        radial = _i2d.radial if _i2d is not None else np.array([])
        azimuthal = _i2d.azimuthal if _i2d is not None else np.array([])

        # Determine which 2D axis is the "display" axis and which is
        # the "slice" axis.
        # IntegrationResult2D.intensity shape is [radial, azimuthal].
        axis_type = info.get('axis', 'radial')

        if axis_type == 'radial':
            # Display along radial, slice along azimuthal
            xdata = radial
            slice_data = azimuthal
            # mean over azimuthal (axis 1) → 1D along radial
            reduce_axis = 1
        elif axis_type == 'azimuthal':
            # Display along azimuthal, slice along radial
            xdata = azimuthal
            slice_data = radial
            # mean over radial (axis 0) → 1D along azimuthal
            reduce_axis = 0
        else:
            # Fallback for legacy standard-mode paths
            xdata = radial
            slice_data = azimuthal
            reduce_axis = 1

        # Apply slice range if enabled
        _inds = np.s_[:]
        if self.ui.slice.isChecked():
            center = self.ui.slice_center.value()
            width = self.ui.slice_width.value()
            _range = [center - width, center + width]
            _inds = (_range[0] <= slice_data) & (slice_data <= _range[1])

        if reduce_axis == 0:
            # Reducing over radial (axis 0): _inds filters radial rows
            ydata = np.nanmean(intensity[_inds, :], axis=0)
        else:
            # Reducing over azimuthal (axis 1): _inds filters azimuthal cols
            ydata = np.nanmean(intensity[:, _inds], axis=1)

        self.show_slice_overlay()

        ydata = self.normalize(ydata, arch.scan_info)
        return xdata, ydata

    def get_xydata(self, int_2d, gi_2d=None):
        """Reads the unit box and returns appropriate xdata.

        In GI mode, int_2d already holds the selected mode's result,
        so radial/azimuthal axes are always correct.

        args:
            int_2d: IntegrationResult2D, primary integration result
            gi_2d: dict of IntegrationResult2D for GI modes (unused, kept
                   for API compatibility)

        returns:
            xdata, ydata: numpy arrays for radial and azimuthal axes.
        """
        if int_2d is None:
            return np.array([]), np.array([])
        return int_2d.radial, int_2d.azimuthal

    def _get_wavelength(self, arch=None):
        """Return the X-ray wavelength in metres.

        Tries several sources in order:
        1. ``arch.integrator.wavelength`` (available during live processing)
        2. ``self.sphere.mg_args['wavelength']`` (persisted in NXS)
        3. The calibration group in the HDF5 file

        Returns None if the wavelength cannot be determined.
        """
        # 1. From the arch's integrator (fastest, works during live runs)
        if arch is not None:
            ai = getattr(arch, 'integrator', None)
            wl = getattr(ai, 'wavelength', None) if ai else None
            if wl and wl > 0:
                return wl

        # 2. From sphere.mg_args (loaded when NXS is opened)
        wl = self.sphere.mg_args.get('wavelength', None) if hasattr(self.sphere, 'mg_args') else None
        if wl and wl > 0:
            return wl

        # 3. Read from the HDF5 calibration group
        try:
            import h5py
            with h5py.File(self.sphere.data_file, 'r') as f:
                wl = float(f['entry/calibration/wavelength'][()]) # type: ignore
                if wl > 0:
                    return wl
        except Exception:
            logger.debug("Failed to read wavelength from HDF5 calibration group in %s", self.sphere.data_file, exc_info=True)

        return None

    def get_xdata(self, arch):
        """Reads the unit box and returns appropriate xdata for 1D plot.

        Handles on-the-fly Q ↔ 2θ conversion when the plotUnit selection
        differs from the integration unit stored in int_1d.

        args:
            arch: EwaldArch copy (data_1d entry) holding int_1d and gi_1d

        returns:
            xdata: numpy array, x axis data for plot.
        """
        int_1d = getattr(arch, 'int_1d', None)
        if int_1d is None:
            return np.array([])

        radial = int_1d.radial
        plot_label = self.ui.plotUnit.currentText()

        # Determine if conversion is needed by comparing plotUnit label
        # to the stored integration unit
        data_unit = getattr(int_1d, 'unit', 'q_A^-1')
        want_tth = (Th in plot_label)  # plotUnit label contains θ
        have_tth = ('2th' in data_unit)

        if want_tth and not have_tth:
            # Data is in Q, display wants 2θ: convert Q → 2θ
            wl = self._get_wavelength(arch)
            if wl and wl > 0:
                lam_A = wl * 1e10
                arg = np.clip(radial * lam_A / (4 * np.pi), -1, 1)
                return 2 * np.degrees(np.arcsin(arg))
        elif not want_tth and have_tth and (AA_inv in plot_label):
            # Data is in 2θ, display wants Q: convert 2θ → Q
            wl = self._get_wavelength(arch)
            if wl and wl > 0:
                lam_A = wl * 1e10
                return (4 * np.pi / lam_A) * np.sin(np.radians(radial / 2))

        return radial

    def normalize(self, int_data, scan_info):
        """Reads the norm, raw, pcount option box and returns
        appropriate ydata.

        args:
            box: QComboBox, list of choices for picking data to return
            int_data: int_nd_data object, data to parse

        returns:
            data: numpy array, region selected based on
                choices in box
            corners: tuple, the bounds of the non-zero region of the
                dataset
        """
        try:
            intensity = np.asarray(int_data.copy(), dtype=float)
        except AttributeError:
            return np.zeros((10, 10))

        # normChannel = self.ui.normChannel.currentText()
        # if normChannel in scan_info.keys():
        #     self.normChannel = normChannel
        # elif normChannel.upper() in scan_info.keys():
        #     self.normChannel = normChannel.upper()
        # elif normChannel.lower() in scan_info.keys():
        #     self.normChannel = normChannel.lower()
        # else:
        #     self.normChannel = None

        normChannel = self.get_normChannel(scan_data_keys=scan_info.keys())
        if normChannel and (scan_info[normChannel] > 0):
            intensity /= scan_info[normChannel]

        return intensity

    def get_normChannel(self, scan_data_keys=None):
        """Check to see if normalization channel exists in metadata and return name"""
        normChannel = self.ui.normChannel.currentText()
        if normChannel == 'sec':
            normChannel = {'sec', 'seconds', 'Seconds', 'Sec', 'SECONDS', 'SEC'}
        elif normChannel == 'Monitor':
            normChannel = {'Monitor', 'monitor', 'mon', 'Mon', 'MON', 'MONITOR'}
        else:
            normChannel = {normChannel, normChannel.lower(), normChannel.upper()}
        if scan_data_keys is None:
            scan_data_keys = self.sphere.scan_data.columns
        normChannel = normChannel.intersection(scan_data_keys)
        return normChannel.pop() if len(normChannel) > 0 else None

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
            # if 'Overall' in self.arch_ids:
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

    def setup_curves(self):
        """Initialize curves for line plots
        """
        self.curves.clear()
        self.legend.clear()

        arch_ids = self.arch_names[self.wf_start::self.wf_step]
        if (self.plotMethod in ['Sum', 'Average'] and
                len(self.arch_names) > 1):
            arch_ids = f'{self.plotMethod} [{self.arch_names[0]}'
            for arch_name in self.arch_names[1:]:
                arch_ids += f', {arch_name}'
            arch_ids = [arch_ids + ']']

        colors = self.get_colors()
        self.curves = [self.plot.plot(
            pen=color,
            symbolBrush=color,
            symbolPen=color,
            symbolSize=4,
            name=arch_id,
        ) for (color, arch_id) in zip(colors, arch_ids)]

        if not self.ui.showLegend.isChecked():
            self.legend.clear()

    def clear_1D(self):
        """Initialize curves for line plots
        """
        self.arch_names.clear()
        self.arch_ids.clear()
        self.plot_data = [np.zeros(0), np.zeros(0)]
        self.setup_1d_layout()
        self.plot.clear()
        # Re-add legend (plot.clear() removes it)
        self.legend = self.plot.addLegend()

    def update_legend(self):
        if not self.ui.showLegend.isChecked():
            self.legend.hide()
        else:
            self.legend.show()

    def _set_slice_range(self, _=None, initialize=False):
        """Configure the slice label, range, and step size based on
        which axis is being sliced along.

        Uses ``self._plot_axis_info`` when available to determine the
        slice axis from the 2D integration metadata.  Falls back to the
        legacy index-based logic for standard mode.
        """
        idx = self.ui.plotUnit.currentIndex()
        info = (self._plot_axis_info[idx]
                if hasattr(self, '_plot_axis_info')
                   and 0 <= idx < len(self._plot_axis_info)
                else None)

        # Determine the slice axis label
        if info and info['source'] in ('2d', '1d_2d') and info.get('slice_axis'):
            slice_label = info['slice_axis']
        elif info and info['source'] == '1d':
            # 1D axis selected — default slice along Chi for standard mode
            slice_label = f'{Chi} ({Deg})'
        else:
            # Legacy fallback for standard mode
            if idx == 2:
                # Chi selected — slice along Q or 2Th
                if self.ui.imageUnit.currentIndex() != 1:
                    slice_label = f'Q ({AA_inv})'
                else:
                    slice_label = f'2{Th} ({Deg})'
            else:
                slice_label = f'{Chi} ({Deg})'

        # Extract the short label for display (strip units in parens)
        short_label = re.sub(r'\s*\(.*\)', '', slice_label).strip()

        # Configure ranges based on what we're slicing
        is_q_like = any(s in slice_label for s in (AA_inv, 'Q'))
        is_angle = any(s in slice_label for s in (Deg, Chi, Th, 'angle', 'Exit'))

        if is_q_like and not is_angle:
            # Q-type axis (Å⁻¹)
            self.ui.slice.setText(f'{short_label} Range')
            self.ui.slice_center.setRange(0, 25)
            self.ui.slice_width.setRange(0, 30)
            self.ui.slice_center.setSingleStep(0.1)
            self.ui.slice_width.setSingleStep(0.1)
            if initialize:
                self.ui.slice_center.setValue(2)
                self.ui.slice_width.setValue(0.5)
        else:
            # Angle-type axis (degrees)
            self.ui.slice.setText(f'{short_label} Range')
            self.ui.slice_center.setRange(-180, 180)
            self.ui.slice_width.setRange(0, 270)
            self.ui.slice_center.setSingleStep(1)
            self.ui.slice_width.setSingleStep(1)
            if initialize:
                self.ui.slice_center.setValue(0)
                self.ui.slice_width.setValue(10)

    def _update_slice_range(self):
        """Handle imageUnit changes that affect slice range units.

        In standard mode with χ selected as plotUnit, switching between
        Q-χ and 2θ-χ imageUnit requires converting the slice range
        between Q and 2θ.  In GI mode or when the axis metadata
        directly specifies the slice axis, just refresh the range.
        """
        if not self.ui.slice.isChecked():
            self.clear_slice_overlay()
            return

        plotUnit = self.ui.plotUnit.currentIndex()
        info = (self._plot_axis_info[plotUnit]
                if hasattr(self, '_plot_axis_info')
                   and 0 <= plotUnit < len(self._plot_axis_info)
                else None)

        # In GI mode or when metadata explicitly defines the slice axis,
        # no unit conversion is needed — just refresh
        if self.sphere.gi or (info and info['source'] not in ('2d', '1d_2d')):
            self.update_plot()
            return

        # Standard mode, chi axis: handle Q ↔ 2θ conversion
        if not self.sphere.gi and info and info.get('axis') == 'azimuthal':
            imageUnit = self.ui.imageUnit.currentIndex()
            cen = self.ui.slice_center.value()
            wid = self.ui.slice_width.value()
            _range = np.array([cen - wid, cen + wid])

            try:
                arch_for_wl = self.data_1d[self.idxs_1d[0]]
            except (IndexError, KeyError):
                self.update_plot()
                return
            wavelength = self._get_wavelength(arch_for_wl)
            if wavelength is None or wavelength <= 0:
                self.update_plot()
                return

            if imageUnit == 0:
                if self.ui.slice.text() == f'2{Th} Range':
                    _range = ((4 * np.pi / (wavelength * 1e10))
                              * np.sin(np.radians(_range / 2)))
            else:
                if self.ui.slice.text() == 'Q Range':
                    _range = (2 * np.degrees(
                        np.arcsin(_range * (wavelength * 1e10) / (4 * np.pi))))

            cen = (_range[-1] + _range[0]) / 2.
            wid = (_range[-1] - _range[0]) / 2.
            self.ui.slice_center.setValue(cen)
            self.ui.slice_width.setValue(wid)

        self._set_slice_range()
        self.show_slice_overlay()

    def show_slice_overlay(self):
        """
        Shows the slice integration region on 2D Binned plot.

        The overlay orientation depends on which axis the user is
        displaying (radial vs azimuthal):
        - Displaying radial → slicing along azimuthal → horizontal band
        - Displaying azimuthal → slicing along radial → vertical band
        """
        self.clear_slice_overlay()

        if not self.ui.slice.isChecked():
            return

        idx = self.ui.plotUnit.currentIndex()
        info = (self._plot_axis_info[idx]
                if hasattr(self, '_plot_axis_info')
                   and 0 <= idx < len(self._plot_axis_info)
                else None)

        # Only show overlay for 2D-derived axes
        if info and info['source'] not in ('2d', '1d_2d'):
            return

        center = self.ui.slice_center.value()
        width = self.ui.slice_width.value()
        _range = [center - width, center + width]

        binned_data, rect = self.binned_data

        if rect is None:
            return

        # Determine orientation from axis metadata
        axis_type = info.get('axis', 'radial') if info else 'radial'

        if axis_type == 'radial':
            # Displaying radial axis → slice band is along azimuthal (y-axis)
            _range = [max(rect.top(), _range[0]),
                      min(rect.top() + rect.height(), _range[1])]
            width = (_range[1] - _range[0]) / 2.
            self.overlay = ROI(
                [rect.left(), _range[0]], [rect.width(), 2 * width],
                pen=(255, 255, 255),
                maxBounds=rect
            )
        else:
            # Displaying azimuthal axis → slice band is along radial (x-axis)
            _range = [max(rect.left(), _range[0]),
                      min(rect.left() + rect.width(), _range[1])]
            width = (_range[1] - _range[0]) / 2.
            self.overlay = ROI(
                [_range[0], rect.top()], [2 * width, rect.height()],
                pen=(255, 255, 255),
                maxBounds=rect
            )

        self.binned_widget.imageViewBox.addItem(self.overlay, ignoreBounds=True)

    def clear_slice_overlay(self):
        """Clear the overlay that shows integration slice"""
        if self.overlay is not None:
            self.binned_widget.imageViewBox.removeItem(self.overlay)
            self.overlay = None

    def save_image(self):
        """Saves currently displayed image. Formats are automatically
        grabbed from Qt. Also implements tiff saving.
        """
        ext_filter = "Images ("
        for f in formats:
            ext_filter += "*." + f + " "

        dialog = QFileDialog()
        fname, _ = dialog.getSaveFileName(
            dialog,
            filter=ext_filter,
            caption='Save as...',
            options=QFileDialog.DontUseNativeDialog
        )
        if fname == '':
            return

        # Choose the right widget depending on viewer mode
        if self.viewer_mode == 'image':
            data, rect = self.image_data
            scene = self.image_widget.imageViewBox.scene()
        else:
            data, rect = self.binned_data
            scene = self.binned_widget.imageViewBox.scene()

        exporter = pyqtgraph.exporters.ImageExporter(scene)
        h = exporter.params.param('height').value()
        w = exporter.params.param('width').value()
        if h == 0 or w == 0:
            logger.warning("Cannot export image with zero dimensions (%dx%d)", w, h)
            return
        h_new = 2000
        w_new = int(np.round(w/h * h_new, 0))
        exporter.params.param('height').setValue(h_new)
        exporter.params.param('width').setValue(w_new)
        exporter.export(fname)

        directory, base_name, ext = split_file_name(fname)
        save_fname = os.path.join(directory, base_name)

        # Save as Numpy array
        np.save(f'{save_fname}.npy', data)

        # In image viewer mode, also save a pyFAI-compatible TIFF
        # from the raw detector-frame data (not the transposed display).
        if self.viewer_mode == 'image' and len(self.idxs_2d) > 0:
            try:
                import fabio
                raw = np.asarray(
                    self.data_2d[self.idxs_2d[0]]['map_raw'], dtype=np.float32)
                tif_path = os.path.join(directory, f'{base_name}_npy.tif')
                fabio.tifimage.TifImage(data=raw).write(tif_path)
                logger.info("Saved pyFAI-compatible TIFF: %s", tif_path)
            except Exception:
                logger.exception("Failed to save TIFF for pyFAI")

    def save_1D(self, auto=False):
        """Saves currently displayed data. Currently supports .xye
        and .csv.
        """
        fname = f'{self.sphere.name}'
        if not auto:
            path = QFileDialog.getExistingDirectory(
                self,
                caption="Select Directory to Save Images",
                dir="",
                options=(QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog)
            )

            inp_dialog = QtWidgets.QInputDialog()
            suffix, ok = inp_dialog.getText(inp_dialog, 'Enter Suffix to be added to File Name', 'Suffix', text='')
            if not ok:
                return
            if suffix != '':
                fname += f'_{suffix}'
        else:
            path = os.path.dirname(self.sphere.data_file)
            path = os.path.join(path, self.sphere.name)
            Path(path).mkdir(parents=True, exist_ok=True)

        fname = os.path.join(path, fname)

        xdata, ydata = self.plot_data
        if self.plotMethod in ['Average', 'Sum']:
            if self.plotMethod == 'Average':
                s_ydata = np.nanmean(ydata, 0)
            else:
                s_ydata = np.nansum(ydata, 0)

            # Write to xye
            xye_fname = f'{fname}.xye'
            ut.write_xye(xye_fname, xdata, s_ydata)

        idxs = [arch.replace(f'{self.sphere.name}_', '') for arch in self.arch_names]
        for nn, (s_ydata, idx) in enumerate(zip(ydata, idxs)):
            # Write to xye
            xye_fname = f'{fname}_{str(idx).zfill(4)}.xye'
            ut.write_xye(xye_fname, xdata, s_ydata)

        if not auto:
            scene = self.plot_viewBox.scene()
            exporter = pyqtgraph.exporters.ImageExporter(scene)
            h = exporter.params.param('height').value()
            w = exporter.params.param('width').value()
            h_new = 600
            w_new = int(np.round(w/h * h_new, 0))
            exporter.params.param('height').setValue(h_new)
            exporter.params.param('width').setValue(w_new)
            exporter.export(fname + '.png')

    def get_colors(self):
        # Define color tuples

        colors = (1, 1, 1)
        if self.cmap == 'Default':
            colors_tuples = [plt.get_cmap('tab10'), plt.get_cmap('Set3'), plt.get_cmap('tab20b', 5)]
            for nn, color_tuples in enumerate(colors_tuples):
                if nn == 0:
                    colors = np.asarray(color_tuples.colors)
                else:
                    colors = np.vstack((colors, np.asarray(color_tuples.colors)[:, 0:3]))

            colors_tuples = plt.get_cmap('jet')
            more_colors = colors_tuples(np.linspace(0, 1, len(self.arch_names)))
            colors = np.vstack((colors, more_colors[:, 0:3]))

        else:
            try:
                colors_tuples = plt.get_cmap(self.cmap)
            except ValueError:
                colors_tuples = plt.get_cmap('jet', 256)
            colors = colors_tuples(np.linspace(0, 1, len(self.arch_names)))[:, 0:3]

        colors = np.round(colors * [255, 255, 255]).astype(int)
        colors = [tuple(color[:3]) for color in colors]

        return colors

    def get_profile_chi(self, arch):
        """
        Args:
            arch (EwaldArch Object):

        Returns:
            intensity (ndarray): Intensity integrated along Chi
                                 over a range of Q specified by UI
        """
        pass

    def get_chi_1d(self, arch):
        """
        Args:
            arch (EwaldArch Object):

        Returns:
            intensity (ndarray): Intensity integrated along Chi
                                 over a range of Q specified by UI
        """
        pass

    def trackMouse(self):
        """
        Sets up mouse trancking
        Returns: x, y (z) coordinates on screen

        """
        proxy = pg.SignalProxy(signal=self.plot.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        proxy.signal.connect(self.mouseMoved)

    def mouseMoved(self, pos):
        if len(self.curves) == 0:
            return

        if self.plot.sceneBoundingRect().contains(pos):
            vb = self.plot.vb
            self.plot_win.setCursor(pyQt.CrossCursor)
            mousePoint = vb.mapSceneToView(pos)
            self.pos_label.setText(f'x={mousePoint.x():.2f}, y={mousePoint.y():.2e}')
        else:
            self.pos_label.setText('')
            self.plot_win.setCursor(pyQt.ArrowCursor)

    def update_wf_pmesh(self):
        """Updates data in 1D plot Frame
        """
        self.setup_wf_layout()

        xdata_, data_ = self.plot_data
        s_xdata, data = xdata_.copy(), data_.copy()
        data = data[self.wf_start::self.wf_step, :]

        x_max, x_min = np.max(s_xdata), np.min(s_xdata)
        x_step = (x_max - x_min)/len(s_xdata)
        s_xdata = np.append(s_xdata, [x_max + x_step])
        s_xdata -= x_step/2
        s_xdata = np.tile(s_xdata, (data.shape[0]+1, 1)).T

        # Set YAxis Unit
        if self.wf_yaxis == 'Frame #':
            s_ydata = np.asarray(np.arange(data.shape[0]) + self.wf_start + 1, dtype=float)
        else:
            # TODO: Fix below for more WF options
            if self.wf_yaxis == 'Time (s)':
                s_ydata = np.asarray([self.data_1d[idx].scan_info['epoch'] for idx in self.idxs])
                s_ydata -= s_ydata.min()
            elif self.wf_yaxis == 'Time (minutes)':
                s_ydata = np.asarray([self.data_1d[idx].scan_info['epoch'] for idx in self.idxs])/60.
                s_ydata -= s_ydata.min()
            else:
                s_ydata = np.asarray([self.data_1d[idx].scan_info[self.wf_yaxis] for idx in self.idxs])

            s_ydata = s_ydata[self.wf_start::self.wf_step]

        y_max, y_min = np.max(s_ydata), np.min(s_ydata)
        y_step = (y_max - y_min)/len(s_ydata)
        s_ydata = np.append(s_ydata, [y_max + y_step])
        s_ydata -= y_step/2.
        s_ydata = np.tile(s_ydata, (data.shape[1]+1, 1))

        # self.wf_widget.setImage(data.T, scale=self.scale, cmap=self.cmap)
        levels = np.nanpercentile(data, (1, 98))
        self.wf_widget.imageItem.setLevels(levels)
        self.wf_widget.imageItem.setData(s_xdata, s_ydata, data.T)
        self.wf_widget.imageItem.informViewBoundsChanged()

        # rect = get_rect(s_xdata, np.arange(data.shape[0]))
        rect = get_rect(s_xdata[:, 0], s_ydata[0])
        self.wf_widget.setRect(rect)

        plotUnit = self.ui.plotUnit.currentIndex()
        self.wf_widget.image_plot.setLabel("bottom", x_labels_1D[plotUnit],
                                           units=x_units_1D[plotUnit])
        self.wf_widget.image_plot.setLabel("left", self.wf_yaxis)

        return data
